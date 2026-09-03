"""Run the EVA02-CLIP-B/16 detector with LoRA adapters attached.

Three modes:

    --mode all        one adapter, by default the one trained on every distortion
    --mode ensemble   several adapters in turn, then their mean
    --mode paired     take only the undistorted images of a split, score them
                      clean, then score the very same images again with one
                      randomly drawn distortion applied at a random severity,
                      exactly as during training. Because both passes share one
                      image set and one fixed distortion draw, the difference
                      between them isolates what the distortion costs and how
                      much of that the adapters recover.

Which adapters run is free to choose with --adapters:

    --adapters jpeg,motion_blur     only these two
    --adapters all,jpeg             mix the all-distortions adapter with one more
    --list-adapters                 show what is trained and exit

Two inputs:

    --image path/to/one.jpg     a single image
    --split test                the whole test split, with ROC-AUC and accuracy
                                reported over all / clean / distorted images

Examples:

    python3 lora/lora_infer.py --mode ensemble --image sample.jpg \
        --head /data2/aidetection/runs/evaclipb_gap_distorted_only/best.pt

    python3 lora/lora_infer.py --mode all --split test \
        --head /data2/aidetection/runs/evaclipb_gap_distorted_only/best.pt

All adapters share one trunk in memory: the model is loaded once and the active
adapter is switched between passes, so an ensemble run over a folder costs N
forward passes over the data rather than N model loads per image.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from augmentations import BUILTIN_DISTORTION_NAMES  # noqa: E402
from aigc_detector.augmentation_pipeline import DistortionPipeline, DistortionPolicy  # noqa: E402
from aigc_detector.data import AIGCImageDataset, ImageSample, load_samples, seed_worker  # noqa: E402
from aigc_detector.distributed import initialize_distributed, set_global_seed  # noqa: E402
from aigc_detector.lora_model import (  # noqa: E402
    build_lora_detector,
    load_detector_checkpoint,
    load_lora_adapters,
)
from aigc_detector.metrics import sigmoid_numpy  # noqa: E402
from aigc_detector.training import _configure_strict_fp32  # noqa: E402


DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
DEFAULT_ADAPTERS_ROOT = PROJECT_ROOT / "runs_lora"
ALL_ADAPTER_NAME = "all"
BASELINE_NAME = "__baseline__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("all", "ensemble", "paired"), default=None)
    parser.add_argument("--head", type=Path, default=None, help="Detector checkpoint (best.pt) providing the head.")
    parser.add_argument("--adapters-root", type=Path, default=DEFAULT_ADAPTERS_ROOT)
    parser.add_argument("--image", type=Path, default=None, help="Score a single image instead of a split.")
    parser.add_argument("--split", default=None, help="Dataset split to score, for example 'test'.")
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument(
        "--adapters",
        default=None,
        help=(
            "Comma-separated adapter names to run, in the order given, for example "
            "'jpeg,motion_blur' or 'all,jpeg'. Any directory under --adapters-root is "
            "accepted, so adapters for future distortions need no code change. "
            "Defaults: --mode all runs 'all'; --mode ensemble runs every working distortion."
        ),
    )
    parser.add_argument(
        "--distortions",
        default=None,
        help="Deprecated alias for --adapters, kept so existing commands keep working.",
    )
    parser.add_argument(
        "--list-adapters",
        action="store_true",
        help="Print the trained adapters under --adapters-root and exit.",
    )
    parser.add_argument(
        "--backbone-local-dir",
        default=str(PROJECT_ROOT / "pretrained" / "eva02_clip_b16"),
        help="Offline EVA-CLIP directory. Pass 'none' to download from the Hugging Face hub instead.",
    )
    parser.add_argument("--backbone-cache-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability at or above which an image is fake.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--precision", default="fp32", choices=("fp32", "bf16"))
    parser.add_argument("--output", type=Path, default=None, help="Write per-image predictions to this CSV.")
    parser.add_argument(
        "--distortion-ops",
        default=None,
        help=(
            "--mode paired only: comma-separated distortions to draw from. "
            "Defaults to every working distortion."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help=(
            "--mode paired only: how many distinct distortions to apply in sequence to each "
            "image. 1 keeps one distortion per image; 2 and 3 draw that many different "
            "distortions and apply them back to back. Every applied distortion always gets "
            "its own random severity."
        ),
    )
    parser.add_argument("--severity-min", type=int, default=1, help="--mode paired only.")
    parser.add_argument("--severity-max", type=int, default=5, help="--mode paired only.")
    parser.add_argument(
        "--distortion-seed",
        type=int,
        default=None,
        help=(
            "--mode paired only: seed for the distortion draw. One draw is shared by every "
            "adapter, so the comparison stays paired. Defaults to --seed."
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the adapter-disabled reference pass over a split.",
    )
    arguments = parser.parse_args()
    if arguments.list_adapters:
        return arguments
    if arguments.mode is None:
        parser.error("--mode is required.")
    if arguments.head is None:
        parser.error("--head is required.")
    if arguments.adapters and arguments.distortions:
        parser.error("Pass either --adapters or its deprecated alias --distortions, not both.")
    if arguments.mode == "paired":
        if arguments.image is not None:
            parser.error("--mode paired scores a whole split, not a single image.")
        if arguments.split is None:
            arguments.split = "test"
        if not 1 <= arguments.severity_min <= arguments.severity_max <= 5:
            parser.error("Severity must satisfy 1 <= --severity-min <= --severity-max <= 5.")
        if arguments.rounds < 1:
            parser.error("--rounds must be at least 1.")
    elif (arguments.image is None) == (arguments.split is None):
        parser.error("Pass exactly one of --image or --split.")
    if not 0.0 < arguments.threshold < 1.0:
        parser.error("--threshold must lie strictly between 0 and 1.")
    return arguments


def resolve_backbone_local_dir(value: str | None) -> str | None:
    """Allow --backbone-local-dir none to fall back to the Hugging Face hub."""
    if value is None or str(value).strip().casefold() in {"", "none", "null", "hub"}:
        return None
    return str(Path(value).expanduser())


def discover_adapters(adapters_root: Path) -> dict[str, Path]:
    """Every trained adapter directly under ``adapters_root``, in sorted order."""
    if not adapters_root.is_dir():
        return {}
    return {
        directory.name: directory.resolve()
        for directory in sorted(adapters_root.iterdir())
        if directory.is_dir() and (directory / "adapter_model.safetensors").is_file()
    }


def print_adapter_listing(adapters_root: Path) -> None:
    available = discover_adapters(adapters_root)
    print(f"adapters root: {adapters_root.resolve()}")
    if not available:
        print("  (none trained yet)")
        return
    width = max(len(name) for name in available)
    for name, directory in available.items():
        metadata_path = directory / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            operations = ",".join(metadata.get("operations", [])) or "?"
            detail = (
                f"epochs={metadata.get('training', {}).get('epochs_run', '?')} "
                f"best_epoch={metadata.get('best_epoch', '?')} "
                f"operations={operations}"
            )
        else:
            detail = "INCOMPLETE (no metadata.json; the training run was interrupted)"
        print(f"  {name:<{width}}  {detail}")


def resolve_adapters(arguments: argparse.Namespace) -> dict[str, Path]:
    available = discover_adapters(arguments.adapters_root)
    selection = arguments.adapters or arguments.distortions

    if selection:
        names = [name.strip() for name in selection.split(",") if name.strip()]
        if not names:
            raise ValueError("--adapters was given but lists no adapter names.")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"--adapters lists the same adapter more than once: {duplicates}.")
        if arguments.mode == "all" and len(names) != 1:
            raise ValueError(
                f"--mode all scores exactly one adapter, but {len(names)} were requested: {names}. "
                "Use --mode ensemble to score several and average them."
            )
    elif arguments.mode == "all":
        names = [ALL_ADAPTER_NAME]
    else:
        names = list(BUILTIN_DISTORTION_NAMES)


    missing = [name for name in names if name not in available]
    if missing:
        raise FileNotFoundError(
            f"These adapters are not trained under {arguments.adapters_root.resolve()}: {missing}.\n"
            f"Available: {sorted(available) or 'none'}.\n"
            "Train them with lora/lora_train.py, or list what exists with --list-adapters."
        )
    return {name: available[name] for name in names}


def check_adapter_compatibility(adapters: dict[str, Path], checkpoint: dict[str, Any], head_path: Path) -> None:
    expected_size = int(checkpoint["preprocessing"]["image_size"])
    resolved_head = str(head_path.resolve())
    for name, directory in adapters.items():
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            print(f"[warn] adapter {name!r} has no metadata.json; compatibility cannot be checked.")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        adapter_size = int(metadata.get("preprocessing", {}).get("image_size", expected_size))
        if adapter_size != expected_size:
            raise ValueError(
                f"Adapter {name!r} was trained at image_size {adapter_size}, but the head expects "
                f"{expected_size}. They are not interchangeable."
            )
        trained_head = metadata.get("head_checkpoint")
        if trained_head and trained_head != resolved_head:
            # Not an error: the trunk is identical across detector variants, so
            # cross-attachment is legitimate. It just is not the trained pairing.
            print(
                f"[warn] adapter {name!r} was trained against {trained_head}, "
                f"now attached to {resolved_head}."
            )


def autocast_context(precision: str, device: torch.device):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


@torch.no_grad()
def score_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    """Return per-sample probabilities ordered by dataset index."""
    indices: list[int] = []
    logits: list[float] = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with autocast_context(precision, device):
            batch_logits = model(pixel_values)
        indices.extend(int(value) for value in batch["index"].tolist())
        logits.extend(float(value) for value in batch_logits.float().cpu().tolist())
    order = np.argsort(np.asarray(indices, dtype=np.int64), kind="stable")
    if len(set(indices)) != len(indices):
        raise RuntimeError("Scoring produced duplicate dataset indices.")
    return sigmoid_numpy(np.asarray(logits, dtype=np.float64)[order])


def score_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    """ROC-AUC and accuracy for probability scores (not logits)."""
    if labels.size == 0:
        return {"samples": 0}
    predictions = (scores >= threshold).astype(np.int64)
    result: dict[str, Any] = {
        "samples": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "mean_probability": float(scores.mean()),
    }
    if np.unique(labels).size == 2:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))
    else:
        result["roc_auc"] = None
        result["note"] = "ROC-AUC needs both classes in the evaluated subset."
    return result


def metrics_by_group(
    labels: np.ndarray,
    scores: np.ndarray,
    is_distorted: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    clean_mask = is_distorted == 0
    distorted_mask = is_distorted == 1
    return {
        "all": score_metrics(labels, scores, threshold=threshold),
        "clean": score_metrics(labels[clean_mask], scores[clean_mask], threshold=threshold),
        "distorted": score_metrics(labels[distorted_mask], scores[distorted_mask], threshold=threshold),
    }


def build_loader(
    samples: list[ImageSample],
    *,
    preprocessing: dict[str, Any],
    image_mean: list[float],
    image_std: list[float],
    arguments: argparse.Namespace,
    device: torch.device,
    distortion_pipeline: DistortionPipeline | None = None,
    base_seed: int | None = None,
) -> DataLoader:
    dataset = AIGCImageDataset(
        samples,
        image_size=int(preprocessing["image_size"]),
        image_mean=image_mean,
        image_std=image_std,
        base_seed=arguments.seed if base_seed is None else base_seed,
        distortion_pipeline=distortion_pipeline,
    )
    generator = torch.Generator()
    generator.manual_seed(arguments.seed)
    return DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def collect_scores(
    model: torch.nn.Module,
    peft_model: Any,
    adapters: Sequence[str],
    loader: DataLoader,
    device: torch.device,
    precision: str,
    *,
    with_baseline: bool,
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for name in adapters:
        peft_model.set_adapter(name)
        scores[name] = score_dataset(model, loader, device, precision)

    # The baseline pass runs last on purpose. It is the only pass that has to
    # tear the adapters down, so nothing that follows depends on that state
    # being restored correctly.
    if with_baseline:
        with peft_model.disable_adapter():
            scores[BASELINE_NAME] = score_dataset(model, loader, device, precision)
        inert = [name for name in adapters if np.array_equal(scores[name], scores[BASELINE_NAME])]
        if inert:
            raise RuntimeError(
                f"These adapters scored identically to the detector with no adapter attached: {inert}. "
                "They are attached but not routed through the forward pass, so their numbers are "
                "meaningless. Check the installed peft version against requirements.txt."
            )
    return scores


def write_predictions(
    path: Path,
    samples: list[ImageSample],
    scores: dict[str, np.ndarray],
    adapters: Sequence[str],
    mean_scores: np.ndarray,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_name", "label", "is_distorted"]
    if BASELINE_NAME in scores:
        fieldnames.append("prob_baseline")
    fieldnames += [f"prob_{name}" for name in adapters] + ["prob_mean", "verdict"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, sample in enumerate(samples):
            row = {
                "image_name": sample.path.name,
                "label": "" if sample.label is None else int(sample.label),
                "is_distorted": "" if sample.is_distorted is None else int(sample.is_distorted),
                "prob_mean": f"{mean_scores[position]:.6f}",
                "verdict": "fake" if mean_scores[position] >= threshold else "real",
            }
            if BASELINE_NAME in scores:
                row["prob_baseline"] = f"{scores[BASELINE_NAME][position]:.6f}"
            for name in adapters:
                row[f"prob_{name}"] = f"{scores[name][position]:.6f}"
            writer.writerow(row)


def run_single_image(
    arguments: argparse.Namespace,
    model: torch.nn.Module,
    peft_model: Any,
    adapters: dict[str, Path],
    preprocessing: dict[str, Any],
    device: torch.device,
) -> None:
    image_path = arguments.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    samples = [ImageSample(path=image_path, label=None, sample_id=image_path.stem)]
    loader = build_loader(
        samples,
        preprocessing=preprocessing,
        image_mean=model.image_mean,
        image_std=model.image_std,
        arguments=arguments,
        device=device,
    )
    names = list(adapters)
    scores = collect_scores(
        model, peft_model, names, loader, device, arguments.precision, with_baseline=False
    )
    probabilities = [float(scores[name][0]) for name in names]
    mean_probability = float(np.mean(probabilities))
    verdict = "fake" if mean_probability >= arguments.threshold else "real"

    print(f"image: {image_path}")
    width = max(len(name) for name in names)
    for name, probability in zip(names, probabilities):
        print(f"  {name:<{width}}  {probability:.6f}")
    print(f"  {'mean':<{width}}  {mean_probability:.6f}")
    print(f"verdict: {verdict}  (threshold {arguments.threshold})")
    print(
        json.dumps(
            {
                "event": "single_image",
                "mode": arguments.mode,
                "image": str(image_path),
                "probabilities": dict(zip(names, probabilities)),
                "mean_probability": mean_probability,
                "threshold": arguments.threshold,
                "verdict": verdict,
            },
            ensure_ascii=False,
        )
    )


def run_split(
    arguments: argparse.Namespace,
    model: torch.nn.Module,
    peft_model: Any,
    adapters: dict[str, Path],
    preprocessing: dict[str, Any],
    device: torch.device,
) -> None:
    samples, _ = load_samples(arguments.dataset_config, arguments.split, require_labels=False)
    loader = build_loader(
        samples,
        preprocessing=preprocessing,
        image_mean=model.image_mean,
        image_std=model.image_std,
        arguments=arguments,
        device=device,
    )
    names = list(adapters)
    scores = collect_scores(
        model,
        peft_model,
        names,
        loader,
        device,
        arguments.precision,
        with_baseline=not arguments.no_baseline,
    )

    stacked = np.asarray([scores[name] for name in names])
    mean_scores = stacked.mean(axis=0)
    max_scores = stacked.max(axis=0)
    labels = np.asarray([-1 if sample.label is None else int(sample.label) for sample in samples], dtype=np.int64)
    is_distorted = np.asarray(
        [-1 if sample.is_distorted is None else int(sample.is_distorted) for sample in samples],
        dtype=np.int64,
    )

    summary: dict[str, Any] = {
        "event": "split_complete",
        "mode": arguments.mode,
        "split": arguments.split,
        "samples": len(samples),
        "adapters": names,
        "threshold": arguments.threshold,
        "head": str(arguments.head.resolve()),
    }
    if (labels < 0).any():
        summary["metrics"] = None
        summary["note"] = "The split has no labels; only per-image predictions were produced."
    else:
        summary["ensemble_metrics"] = metrics_by_group(
            labels, mean_scores, is_distorted, threshold=arguments.threshold
        )
        summary["ensemble_max_metrics"] = metrics_by_group(
            labels, max_scores, is_distorted, threshold=arguments.threshold
        )
        summary["per_adapter_metrics"] = {
            name: metrics_by_group(labels, scores[name], is_distorted, threshold=arguments.threshold)
            for name in names
        }
        if BASELINE_NAME in scores:
            summary["baseline_metrics"] = metrics_by_group(
                labels, scores[BASELINE_NAME], is_distorted, threshold=arguments.threshold
            )

    if arguments.output is not None:
        write_predictions(arguments.output, samples, scores, names, mean_scores, arguments.threshold)
        summary["output"] = str(arguments.output.resolve())

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if "ensemble_metrics" in summary:
        print_metrics_table(summary)


def print_metrics_table(summary: dict[str, Any]) -> None:
    rows: list[tuple[str, dict[str, Any]]] = []
    if "baseline_metrics" in summary:
        rows.append(("baseline (no adapter)", summary["baseline_metrics"]))
    rows += list(summary["per_adapter_metrics"].items())
    rows.append(("MEAN of adapters", summary["ensemble_metrics"]))
    rows.append(("MAX of adapters", summary["ensemble_max_metrics"]))

    width = max(len(name) for name, _ in rows)
    header = f"{'adapter':<{width}}  {'all auc':>8} {'all acc':>8}  {'clean auc':>10} {'clean acc':>10}  {'dist auc':>9} {'dist acc':>9}"
    print()
    print(header)
    print("-" * len(header))
    for name, groups in rows:
        cells = []
        for group in ("all", "clean", "distorted"):
            metrics = groups[group]
            auc = metrics.get("roc_auc")
            accuracy = metrics.get("accuracy")
            cells.append("n/a" if auc is None else f"{auc:.4f}")
            cells.append("n/a" if accuracy is None else f"{accuracy:.4f}")
        print(
            f"{name:<{width}}  {cells[0]:>8} {cells[1]:>8}  {cells[2]:>10} {cells[3]:>10}  {cells[4]:>9} {cells[5]:>9}"
        )


def build_evaluation_pipeline(arguments: argparse.Namespace) -> DistortionPipeline:
    """Draw ``--rounds`` distinct distortions per image, each at its own random severity.

    With the default of one round this mirrors how a per-distortion adapter was
    trained. Higher round counts stack different distortions back to back, which
    is what the challenge test set does to its own images.
    """
    if arguments.distortion_ops:
        operations = [name.strip() for name in arguments.distortion_ops.split(",") if name.strip()]
        unknown = sorted(set(operations) - set(BUILTIN_DISTORTION_NAMES))
        if unknown:
            raise ValueError(
                f"Unknown or non-working distortions: {unknown}. "
                f"Available: {list(BUILTIN_DISTORTION_NAMES)}."
            )
    else:
        operations = list(BUILTIN_DISTORTION_NAMES)
    if arguments.rounds > len(operations):
        raise ValueError(
            f"--rounds {arguments.rounds} needs at least that many distortions to draw from, "
            f"but only {len(operations)} are enabled: {operations}. "
            "Lower --rounds or widen --distortion-ops."
        )
    policy = DistortionPolicy(
        min_operations=arguments.rounds,
        max_operations=arguments.rounds,
        severity_min=arguments.severity_min,
        severity_max=arguments.severity_max,
        sample_without_replacement=True,
    )
    return DistortionPipeline(policy=policy, enabled_operations=operations)


def applied_distortion_plan(
    dataset: AIGCImageDataset,
    pipeline: DistortionPipeline,
    count: int,
) -> tuple[list[list[str]], list[list[int]]]:
    """Recover which distortions each image received, without decoding any image.

    ``AIGCImageDataset`` derives a per-sample RNG from (base_seed, epoch, index)
    and hands it straight to the pipeline, so replaying those two deterministic
    steps reproduces the exact plan the loader used, in order.
    """
    names: list[list[str]] = []
    severities: list[list[int]] = []
    for index in range(count):
        plan = pipeline.sample_plan(dataset._rng_for_index(index))
        names.append([step[0] for step in plan])
        severities.append([int(step[1]) for step in plan])
    return names, severities


def write_paired_predictions(
    path: Path,
    samples: list[ImageSample],
    clean_scores: dict[str, np.ndarray],
    distorted_scores: dict[str, np.ndarray],
    adapters: Sequence[str],
    applied_names: list[list[str]],
    applied_severities: list[list[int]],
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_name", "label", "applied_distortions", "applied_severities"]
    if BASELINE_NAME in clean_scores:
        fieldnames += ["prob_baseline_clean", "prob_baseline_distorted"]
    for name in adapters:
        fieldnames += [f"prob_{name}_clean", f"prob_{name}_distorted"]
    fieldnames += [
        "prob_mean_clean",
        "prob_mean_distorted",
        "prob_max_clean",
        "prob_max_distorted",
        "verdict_mean_clean",
        "verdict_mean_distorted",
        "verdict_max_clean",
        "verdict_max_distorted",
    ]

    clean_stack = np.asarray([clean_scores[name] for name in adapters])
    distorted_stack = np.asarray([distorted_scores[name] for name in adapters])
    clean_mean, distorted_mean = clean_stack.mean(axis=0), distorted_stack.mean(axis=0)
    clean_max, distorted_max = clean_stack.max(axis=0), distorted_stack.max(axis=0)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, sample in enumerate(samples):
            row = {
                "image_name": sample.path.name,
                "label": "" if sample.label is None else int(sample.label),
                "applied_distortions": "|".join(applied_names[position]),
                "applied_severities": "|".join(str(value) for value in applied_severities[position]),
                "prob_mean_clean": f"{clean_mean[position]:.6f}",
                "prob_mean_distorted": f"{distorted_mean[position]:.6f}",
                "prob_max_clean": f"{clean_max[position]:.6f}",
                "prob_max_distorted": f"{distorted_max[position]:.6f}",
                "verdict_mean_clean": "fake" if clean_mean[position] >= threshold else "real",
                "verdict_mean_distorted": "fake" if distorted_mean[position] >= threshold else "real",
                "verdict_max_clean": "fake" if clean_max[position] >= threshold else "real",
                "verdict_max_distorted": "fake" if distorted_max[position] >= threshold else "real",
            }
            if BASELINE_NAME in clean_scores:
                row["prob_baseline_clean"] = f"{clean_scores[BASELINE_NAME][position]:.6f}"
                row["prob_baseline_distorted"] = f"{distorted_scores[BASELINE_NAME][position]:.6f}"
            for name in adapters:
                row[f"prob_{name}_clean"] = f"{clean_scores[name][position]:.6f}"
                row[f"prob_{name}_distorted"] = f"{distorted_scores[name][position]:.6f}"
            writer.writerow(row)


def print_paired_table(summary: dict[str, Any]) -> None:
    rows: list[tuple[str, dict[str, Any]]] = []
    if "baseline" in summary["scores"]:
        rows.append(("baseline (no adapter)", summary["scores"]["baseline"]))
    rows += [(name, summary["scores"][name]) for name in summary["adapters"]]
    rows.append(("MEAN of adapters", summary["scores"]["mean"]))
    rows.append(("MAX of adapters", summary["scores"]["max"]))

    width = max(len(name) for name, _ in rows)
    header = (
        f"{'adapter':<{width}}  {'clean auc':>9} {'clean acc':>9}   "
        f"{'dist auc':>8} {'dist acc':>8}   {'d auc':>7} {'d acc':>7}"
    )
    policy = summary["distortion_policy"]
    print()
    print(
        f"paired evaluation on {summary['samples']} undistorted test images, "
        f"{policy['operations_per_image']} distortion(s) per image, "
        f"severity {policy['severity_min']}-{policy['severity_max']}"
    )
    print(header)
    print("-" * len(header))
    for name, entry in rows:
        clean, distorted = entry["clean"], entry["distorted"]
        print(
            f"{name:<{width}}  {clean['roc_auc']:>9.4f} {clean['accuracy']:>9.4f}   "
            f"{distorted['roc_auc']:>8.4f} {distorted['accuracy']:>8.4f}   "
            f"{distorted['roc_auc'] - clean['roc_auc']:>+7.4f} "
            f"{distorted['accuracy'] - clean['accuracy']:>+7.4f}"
        )


def print_aggregation_table(summary: dict[str, Any]) -> None:
    """Second view: how the per-adapter probabilities are combined into one score."""
    rows: list[tuple[str, dict[str, Any]]] = []
    if "baseline" in summary["scores"]:
        rows.append(("baseline (no adapter)", summary["scores"]["baseline"]))
    rows.append(("aggregation: mean", summary["scores"]["mean"]))
    rows.append(("aggregation: max", summary["scores"]["max"]))

    width = max(len(name) for name, _ in rows)
    header = (
        f"{'combination':<{width}}  {'clean auc':>9} {'clean acc':>9}   "
        f"{'dist auc':>8} {'dist acc':>8}   {'d auc':>7} {'d acc':>7}"
    )
    print()
    print(f"aggregation of {len(summary['adapters'])} adapters into one score")
    print(header)
    print("-" * len(header))
    for name, entry in rows:
        clean, distorted = entry["clean"], entry["distorted"]
        print(
            f"{name:<{width}}  {clean['roc_auc']:>9.4f} {clean['accuracy']:>9.4f}   "
            f"{distorted['roc_auc']:>8.4f} {distorted['accuracy']:>8.4f}   "
            f"{distorted['roc_auc'] - clean['roc_auc']:>+7.4f} "
            f"{distorted['accuracy'] - clean['accuracy']:>+7.4f}"
        )


def print_applied_breakdown(summary: dict[str, Any]) -> None:
    breakdown = summary.get("by_applied_distortion")
    if not breakdown:
        return
    width = max(len("applied distortion"), max(len(name) for name in breakdown))
    has_baseline = "baseline" in summary["scores"]
    header = f"{'applied distortion':<{width}}  {'n':>6}  {'mean auc':>9}  {'max auc':>9}"
    if has_baseline:
        header += f"  {'baseline auc':>13}  {'mean-base':>9}"
    rounds = summary["distortion_policy"]["operations_per_image"]
    print()
    if rounds == 1:
        print("distorted pass, split by the distortion each image received")
    else:
        print(
            f"distorted pass, split by distortion; each image received {rounds} of them "
            "and is counted under every one"
        )
    print(header)
    print("-" * len(header))
    for name, entry in breakdown.items():
        mean_auc = entry["mean"].get("roc_auc")
        max_auc = entry["max"].get("roc_auc")
        line = f"{name:<{width}}  {entry['samples']:>6}  "
        line += "      n/a" if mean_auc is None else f"{mean_auc:>9.4f}"
        line += "        n/a" if max_auc is None else f"  {max_auc:>9.4f}"
        if has_baseline:
            base_auc = entry["baseline"].get("roc_auc")
            line += "            n/a" if base_auc is None else f"  {base_auc:>13.4f}"
            if mean_auc is not None and base_auc is not None:
                line += f"  {mean_auc - base_auc:>+9.4f}"
        print(line)


def run_paired(
    arguments: argparse.Namespace,
    model: torch.nn.Module,
    peft_model: Any,
    adapters: dict[str, Path],
    preprocessing: dict[str, Any],
    device: torch.device,
) -> None:
    """Score the undistorted test images, then the very same images distorted."""
    all_samples, _ = load_samples(arguments.dataset_config, arguments.split, require_labels=True)
    samples = [sample for sample in all_samples if sample.is_distorted == 0]
    if not samples:
        raise ValueError(
            f"Split {arguments.split!r} contains no images marked as undistorted. "
            "The manifest needs an 'is_distorted' column."
        )
    labels = np.asarray([int(sample.label) for sample in samples], dtype=np.int64)
    if np.unique(labels).size != 2:
        raise ValueError("The undistorted subset must contain both classes to compute ROC-AUC.")

    pipeline = build_evaluation_pipeline(arguments)
    distortion_seed = arguments.seed if arguments.distortion_seed is None else arguments.distortion_seed
    loader_kwargs = {
        "preprocessing": preprocessing,
        "image_mean": model.image_mean,
        "image_std": model.image_std,
        "arguments": arguments,
        "device": device,
    }
    clean_loader = build_loader(samples, **loader_kwargs)
    # One distorted dataset shared by every pass: the draw is fixed by
    # (distortion_seed, index), so all adapters and the baseline see byte-identical
    # inputs and the clean/distorted comparison is genuinely paired.
    distorted_loader = build_loader(
        samples, **loader_kwargs, distortion_pipeline=pipeline, base_seed=distortion_seed
    )
    applied_names, applied_severities = applied_distortion_plan(
        distorted_loader.dataset, pipeline, len(samples)
    )

    names = list(adapters)
    with_baseline = not arguments.no_baseline
    print(json.dumps({"event": "paired_pass", "stage": "clean", "samples": len(samples)}))
    clean_scores = collect_scores(
        model, peft_model, names, clean_loader, device, arguments.precision, with_baseline=with_baseline
    )
    print(json.dumps({"event": "paired_pass", "stage": "distorted", "samples": len(samples)}))
    distorted_scores = collect_scores(
        model, peft_model, names, distorted_loader, device, arguments.precision, with_baseline=with_baseline
    )

    clean_stack = np.asarray([clean_scores[name] for name in names])
    distorted_stack = np.asarray([distorted_scores[name] for name in names])
    clean_mean, distorted_mean = clean_stack.mean(axis=0), distorted_stack.mean(axis=0)
    # Max is the other natural way to combine specialists: one adapter shouting
    # "fake" is enough, where the mean would dilute it across the eight.
    clean_max, distorted_max = clean_stack.max(axis=0), distorted_stack.max(axis=0)

    def pair(clean: np.ndarray, distorted: np.ndarray) -> dict[str, Any]:
        return {
            "clean": score_metrics(labels, clean, threshold=arguments.threshold),
            "distorted": score_metrics(labels, distorted, threshold=arguments.threshold),
        }

    scores: dict[str, Any] = {name: pair(clean_scores[name], distorted_scores[name]) for name in names}
    scores["mean"] = pair(clean_mean, distorted_mean)
    scores["max"] = pair(clean_max, distorted_max)
    if with_baseline:
        scores["baseline"] = pair(clean_scores[BASELINE_NAME], distorted_scores[BASELINE_NAME])

    breakdown: dict[str, Any] = {}
    # With more than one round an image carries several distortions, so it is
    # counted under each of them rather than under one combined label.
    for operation in sorted({name for plan in applied_names for name in plan}):
        mask = np.asarray([operation in plan for plan in applied_names])
        entry: dict[str, Any] = {
            "samples": int(mask.sum()),
            "mean": score_metrics(labels[mask], distorted_mean[mask], threshold=arguments.threshold),
            "max": score_metrics(labels[mask], distorted_max[mask], threshold=arguments.threshold),
        }
        if with_baseline:
            entry["baseline"] = score_metrics(
                labels[mask], distorted_scores[BASELINE_NAME][mask], threshold=arguments.threshold
            )
        breakdown[operation] = entry

    summary = {
        "event": "paired_complete",
        "mode": "paired",
        "split": arguments.split,
        "samples": len(samples),
        "samples_in_split": len(all_samples),
        "adapters": names,
        "threshold": arguments.threshold,
        "head": str(arguments.head.resolve()),
        "distortion_policy": {
            "operations": list(pipeline.operations),
            "operations_per_image": arguments.rounds,
            "severity_min": arguments.severity_min,
            "severity_max": arguments.severity_max,
            "seed": distortion_seed,
        },
        "scores": scores,
        "by_applied_distortion": breakdown,
    }
    if arguments.output is not None:
        write_paired_predictions(
            arguments.output,
            samples,
            clean_scores,
            distorted_scores,
            names,
            applied_names,
            applied_severities,
            arguments.threshold,
        )
        summary["output"] = str(arguments.output.resolve())

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print_paired_table(summary)
    print_aggregation_table(summary)
    print_applied_breakdown(summary)


def main() -> None:
    arguments = parse_args()
    if arguments.list_adapters:
        print_adapter_listing(arguments.adapters_root)
        return
    environment = initialize_distributed()
    _configure_strict_fp32()
    set_global_seed(arguments.seed, deterministic=False)

    checkpoint = load_detector_checkpoint(arguments.head)
    adapters = resolve_adapters(arguments)
    check_adapter_compatibility(adapters, checkpoint, arguments.head)

    model, preprocessing = build_lora_detector(
        checkpoint,
        local_dir=resolve_backbone_local_dir(arguments.backbone_local_dir),
        cache_dir=arguments.backbone_cache_dir,
        freeze_head=True,
    )
    peft_model = load_lora_adapters(model, adapters)
    model.to(environment.device)
    model.eval()

    print(
        json.dumps(
            {
                "event": "setup",
                "mode": arguments.mode,
                "adapters": list(adapters),
                "head_experiment": checkpoint.get("experiment"),
                "image_size": int(preprocessing["image_size"]),
                "device": str(environment.device),
                "precision": arguments.precision,
            },
            ensure_ascii=False,
        )
    )

    if arguments.mode == "paired":
        run_paired(arguments, model, peft_model, adapters, preprocessing, environment.device)
    elif arguments.image is not None:
        run_single_image(arguments, model, peft_model, adapters, preprocessing, environment.device)
    else:
        run_split(arguments, model, peft_model, adapters, preprocessing, environment.device)


if __name__ == "__main__":
    main()
