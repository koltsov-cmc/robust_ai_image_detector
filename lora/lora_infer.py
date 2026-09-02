"""Run the EVA02-CLIP-B/16 detector with LoRA adapters attached.

Two modes:

    --mode all        one adapter, the one trained on every distortion
    --mode ensemble   every per-distortion adapter in turn, then their mean

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
    parser.add_argument("--mode", required=True, choices=("all", "ensemble"))
    parser.add_argument("--head", required=True, type=Path, help="Detector checkpoint (best.pt) providing the head.")
    parser.add_argument("--adapters-root", type=Path, default=DEFAULT_ADAPTERS_ROOT)
    parser.add_argument("--image", type=Path, default=None, help="Score a single image instead of a split.")
    parser.add_argument("--split", default=None, help="Dataset split to score, for example 'test'.")
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument(
        "--distortions",
        default=None,
        help="Comma-separated adapter subset for --mode ensemble. Defaults to all working distortions.",
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
        "--no-baseline",
        action="store_true",
        help="Skip the adapter-disabled reference pass over a split.",
    )
    arguments = parser.parse_args()
    if (arguments.image is None) == (arguments.split is None):
        parser.error("Pass exactly one of --image or --split.")
    if not 0.0 < arguments.threshold < 1.0:
        parser.error("--threshold must lie strictly between 0 and 1.")
    return arguments


def resolve_backbone_local_dir(value: str | None) -> str | None:
    """Allow --backbone-local-dir none to fall back to the Hugging Face hub."""
    if value is None or str(value).strip().casefold() in {"", "none", "null", "hub"}:
        return None
    return str(Path(value).expanduser())


def resolve_adapters(arguments: argparse.Namespace) -> dict[str, Path]:
    if arguments.mode == "all":
        names = [ALL_ADAPTER_NAME]
    elif arguments.distortions:
        names = [name.strip() for name in arguments.distortions.split(",") if name.strip()]
        unknown = sorted(set(names) - set(BUILTIN_DISTORTION_NAMES))
        if unknown:
            raise ValueError(f"Unknown or non-working distortions: {unknown}.")
    else:
        names = list(BUILTIN_DISTORTION_NAMES)

    adapters: dict[str, Path] = {}
    missing: list[str] = []
    for name in names:
        directory = (arguments.adapters_root / name).resolve()
        if (directory / "adapter_model.safetensors").is_file():
            adapters[name] = directory
        else:
            missing.append(str(directory))
    if missing:
        raise FileNotFoundError(
            "These adapters have not been trained yet:\n  " + "\n  ".join(missing)
            + "\nTrain them with lora/lora_train.py."
        )
    return adapters


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
) -> DataLoader:
    dataset = AIGCImageDataset(
        samples,
        image_size=int(preprocessing["image_size"]),
        image_mean=image_mean,
        image_std=image_std,
        base_seed=arguments.seed,
        distortion_pipeline=None,
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

    mean_scores = np.mean([scores[name] for name in names], axis=0)
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


def main() -> None:
    arguments = parse_args()
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

    if arguments.image is not None:
        run_single_image(arguments, model, peft_model, adapters, preprocessing, environment.device)
    else:
        run_split(arguments, model, peft_model, adapters, preprocessing, environment.device)


if __name__ == "__main__":
    main()
