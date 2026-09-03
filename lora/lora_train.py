"""Train LoRA adapters for the EVA02-CLIP-B/16 AIGC detector.

Two modes:

    # one adapter that sees every distortion
    python3 lora/lora_train.py --mode all \
        --head /data2/aidetection/runs/evaclipb_gap_distorted_only/best.pt

    # one adapter per distortion, trained back to back
    python3 lora/lora_train.py --mode per-distortion \
        --head /data2/aidetection/runs/evaclipb_gap_distorted_only/best.pt

The distortion list comes from ``augmentations.BUILTIN_DISTORTION_NAMES``, so a
newly added working distortion automatically becomes another adapter with no
change to this file.

The detector head is loaded from ``--head`` and frozen. LoRA is initialised to
zero, so at step 0 the adapted model is bit-for-bit the base detector and the
frozen head is exactly matched to it; training then only moves the trunk. That
is what keeps each adapter a pure, detachable trunk modification that can be
attached to either released detector variant.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from augmentations import BUILTIN_DISTORTION_NAMES  # noqa: E402
from aigc_detector.augmentation_pipeline import DistortionPipeline, DistortionPolicy  # noqa: E402
from aigc_detector.config import load_yaml  # noqa: E402
from aigc_detector.data import (  # noqa: E402
    AIGCImageDataset,
    ImageSample,
    _load_ntire_shards,
    _resolve_relative,
    load_samples,
    seed_worker,
)
from aigc_detector.distributed import initialize_distributed, set_global_seed  # noqa: E402
from aigc_detector.lora_model import (  # noqa: E402
    LORA_TARGET_REGEX,
    attach_lora,
    build_lora_detector,
    load_detector_checkpoint,
    print_lora_report,
    save_lora_adapter,
)
from aigc_detector.metrics import binary_metrics  # noqa: E402
from aigc_detector.training import EarlyStopping, _build_scheduler, _configure_strict_fp32  # noqa: E402


DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
DEFAULT_SUBSET = PROJECT_ROOT / "lora" / "lora_train_subset.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs_lora"
ALL_ADAPTER_NAME = "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("all", "per-distortion"),
        help="'all' trains one adapter over every distortion; 'per-distortion' trains one adapter each.",
    )
    parser.add_argument("--head", required=True, type=Path, help="Detector checkpoint (best.pt) providing the frozen head.")
    parser.add_argument(
        "--subset",
        type=Path,
        default=None,
        help="Training manifest CSV. Defaults to lora/lora_train_subset.csv when --shards is not given.",
    )
    parser.add_argument(
        "--shards",
        default=None,
        help=(
            "Train on every image of these train shards instead of a subset CSV, for "
            "example --shards 0,1,2,3,4,5 or --shards shard_0,shard_1. Images belonging "
            "to the validation split are excluded automatically."
        ),
    )
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--distortions",
        default=None,
        help="Comma-separated subset of distortions for --mode per-distortion. Defaults to all working ones.",
    )
    parser.add_argument(
        "--backbone-local-dir",
        default=str(PROJECT_ROOT / "pretrained" / "eva02_clip_b16"),
        help="Offline EVA-CLIP directory. Pass 'none' to download from the Hugging Face hub instead.",
    )
    parser.add_argument("--backbone-cache-dir", type=Path, default=None)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-regex", default=LORA_TARGET_REGEX)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--log-interval-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=("fp32", "bf16"),
        help="fp32 matches the native detector exactly; bf16 autocast is roughly 2-3x faster.",
    )
    parser.add_argument(
        "--validation-max-samples",
        type=int,
        default=None,
        help="Cap the validation split. Useful to shorten the nine-adapter run.",
    )

    parser.add_argument("--severity-min", type=int, default=1)
    parser.add_argument("--severity-max", type=int, default=5)
    parser.add_argument("--all-min-operations", type=int, default=1)
    parser.add_argument("--all-max-operations", type=int, default=3)

    parser.add_argument("--resume", action="store_true", help="Skip adapters that already have saved weights.")
    parser.add_argument("--overwrite", action="store_true", help="Retrain adapters that already have saved weights.")
    return parser.parse_args()


def resolve_backbone_local_dir(value: str | None) -> str | None:
    """Allow --backbone-local-dir none to fall back to the Hugging Face hub."""
    if value is None or str(value).strip().casefold() in {"", "none", "null", "hub"}:
        return None
    return str(Path(value).expanduser())


def _peft_version() -> str:
    import peft

    return str(peft.__version__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    seconds = max(0, round(float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_subset_samples(path: Path) -> list[ImageSample]:
    if not path.is_file():
        raise FileNotFoundError(
            f"LoRA training subset does not exist: {path}. Build it with `python3 lora/make_subset.py`."
        )
    samples: list[ImageSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            image_path = str(row.get("path", "")).strip()
            if not image_path:
                raise ValueError(f"Empty 'path' column in {path}:{row_number}.")
            label = int(str(row["label"]).strip())
            if label not in {0, 1}:
                raise ValueError(f"Binary label must be 0 or 1, got {label} in {path}:{row_number}.")
            samples.append(
                ImageSample(path=Path(image_path), label=label, sample_id=Path(image_path).stem)
            )
    if not samples:
        raise ValueError(f"{path} contains zero rows.")
    labels = [int(sample.label) for sample in samples]
    if 0 not in labels or 1 not in labels:
        raise ValueError(f"{path} must contain both binary classes.")
    return samples


def load_shard_samples(dataset_config_path: Path, shards: str) -> tuple[list[ImageSample], list[str]]:
    """Load every labelled image of the requested train shards."""
    config = load_yaml(dataset_config_path)
    if config.get("format") != "ntire_2026":
        raise ValueError(
            f"{dataset_config_path} is not in the 'ntire_2026' format, so it has no shard directories."
        )
    root = _resolve_relative(Path(config["_config_dir"]), config.get("root", "../ntire"))
    train_directory = _resolve_relative(root, config.get("train_directory", "train"))
    if not train_directory.is_dir():
        raise FileNotFoundError(f"Train directory does not exist: {train_directory}")

    names: list[str] = []
    for raw in shards.split(","):
        raw = raw.strip()
        if not raw:
            continue
        name = raw if raw.startswith("shard_") else f"shard_{raw}"
        if name in names:
            raise ValueError(f"--shards lists {name!r} more than once.")
        names.append(name)
    if not names:
        raise ValueError("--shards was given but lists no shards.")

    available = sorted(path.name for path in train_directory.glob("shard_*") if path.is_dir())
    missing = [name for name in names if name not in available]
    if missing:
        raise FileNotFoundError(
            f"These shards are not extracted under {train_directory}: {missing}. Available: {available}."
        )

    samples = _load_ntire_shards(
        train_directory,
        [train_directory / name for name in names],
        image_column=str(config.get("train_image_column", "image_name")),
        label_column=str(config.get("train_label_column", "label")),
        class_mapping={str(key): int(value) for key, value in config.get("classes", {}).items()},
    )
    return samples, names


def build_pipeline(operations: Sequence[str], arguments: argparse.Namespace) -> DistortionPipeline:
    """One fixed operation for a per-distortion adapter, 1-3 sampled ones for 'all'."""
    if len(operations) == 1:
        minimum = maximum = 1
    else:
        # Sampling is without replacement, so the cap can never exceed the pool.
        maximum = min(arguments.all_max_operations, len(operations))
        minimum = min(arguments.all_min_operations, maximum)
    policy = DistortionPolicy(
        min_operations=minimum,
        max_operations=maximum,
        severity_min=arguments.severity_min,
        severity_max=arguments.severity_max,
        sample_without_replacement=True,
    )
    return DistortionPipeline(policy=policy, enabled_operations=list(operations))


def make_loader(
    dataset: AIGCImageDataset,
    *,
    arguments: argparse.Namespace,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(arguments.seed)
    return DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def autocast_context(precision: str, device: torch.device):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def assert_gradient_flows(
    model: nn.Module,
    image_size: int,
    device: torch.device,
    precision: str,
) -> None:
    """Fail immediately if the forward pass builds no graph.

    Without this, a trunk whose adapters are attached but not routed produces a
    loss with no grad_fn, and the only symptom is an opaque autograd error on
    the first backward call.
    """
    was_training = model.training
    model.train()
    probe = torch.zeros(1, 3, image_size, image_size, device=device)
    with autocast_context(precision, device):
        logits = model(probe)
    model.train(was_training)
    if not logits.requires_grad or logits.grad_fn is None:
        raise RuntimeError(
            "The adapted model produced a logit with no gradient graph, so nothing can train. "
            "Either the LoRA layers are attached but not routed through the forward pass, or the "
            "trunk is still being run under torch.no_grad(). Check that aigc_detector/ and lora/ "
            "come from the same revision, delete stale __pycache__ directories, and verify the "
            "installed peft version against requirements.txt."
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    labels: list[int] = []
    logits: list[float] = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with autocast_context(precision, device):
            batch_logits = model(pixel_values)
        labels.extend(int(value) for value in batch["label"].tolist())
        logits.extend(float(value) for value in batch_logits.float().cpu().tolist())
    model.train(was_training)
    return binary_metrics(labels=labels, logits=logits)


def train_one_adapter(
    *,
    adapter_name: str,
    operations: Sequence[str],
    arguments: argparse.Namespace,
    checkpoint: dict[str, Any],
    train_samples: list[ImageSample],
    validation_samples: list[ImageSample],
    train_source: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    output_dir = (arguments.output_root / adapter_name).resolve()
    started_at = time.perf_counter()
    print(json.dumps({"event": "adapter_start", "adapter": adapter_name, "operations": list(operations)}))

    # A fresh trunk per adapter: peft rewrites trunk modules in place, and
    # rebuilding is far cheaper than reasoning about leftover adapter state.
    model, preprocessing = build_lora_detector(
        checkpoint,
        local_dir=resolve_backbone_local_dir(arguments.backbone_local_dir),
        cache_dir=arguments.backbone_cache_dir,
        freeze_head=True,
    )
    model.to(device)

    pipeline = build_pipeline(operations, arguments)
    dataset_kwargs = {
        "image_size": int(preprocessing["image_size"]),
        "image_mean": model.image_mean,
        "image_std": model.image_std,
    }
    train_dataset = AIGCImageDataset(
        train_samples, base_seed=arguments.seed, distortion_pipeline=pipeline, **dataset_kwargs
    )
    # The validation datasets never see set_epoch, so their distortion draw is
    # fixed across epochs and the early-stopping metric is not resampled noise.
    distorted_validation = AIGCImageDataset(
        validation_samples, base_seed=arguments.seed + 1, distortion_pipeline=pipeline, **dataset_kwargs
    )
    clean_validation = AIGCImageDataset(
        validation_samples, base_seed=arguments.seed + 1, distortion_pipeline=None, **dataset_kwargs
    )

    train_loader = make_loader(train_dataset, arguments=arguments, device=device, shuffle=True)
    distorted_loader = make_loader(distorted_validation, arguments=arguments, device=device, shuffle=False)
    clean_loader = make_loader(clean_validation, arguments=arguments, device=device, shuffle=False)

    # Baseline = this same head and trunk before any adapter exists, which is
    # exactly the released detector. Measuring it here rather than through
    # peft's disable_adapter() context means no adapter state has to be torn
    # down and restored around it.
    baseline = {
        "distorted": evaluate(model, distorted_loader, device, arguments.precision),
        "clean": evaluate(model, clean_loader, device, arguments.precision),
    }
    print(json.dumps({"event": "baseline", "adapter": adapter_name, **baseline}, ensure_ascii=False))

    peft_model, report = attach_lora(
        model,
        rank=arguments.lora_rank,
        alpha=arguments.lora_alpha,
        dropout=arguments.lora_dropout,
        target_regex=arguments.target_regex,
    )
    print_lora_report(report)
    if report["head_trainable"]:
        raise RuntimeError("The detector head must stay frozen while a LoRA adapter trains.")
    model.to(device)
    assert_gradient_flows(model, int(preprocessing["image_size"]), device, arguments.precision)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=arguments.learning_rate, weight_decay=arguments.weight_decay)
    total_steps = arguments.epochs * len(train_loader)
    scheduler = _build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=min(round(total_steps * arguments.warmup_ratio), max(0, total_steps - 1)),
        min_lr_ratio=arguments.min_lr_ratio,
    )
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=arguments.patience)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    best_record: dict[str, Any] | None = None
    global_step = 0
    epochs_run = 0

    for epoch in range(arguments.epochs):
        epoch_started_at = time.perf_counter()
        epochs_run = epoch + 1
        train_dataset.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        sample_count = 0

        for step_in_epoch, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(arguments.precision, device):
                logits = model(pixel_values)
                loss = criterion(logits.float(), labels)
            loss.backward()
            if arguments.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(trainable, arguments.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            batch_size = labels.numel()
            loss_sum += float(loss.detach().item()) * batch_size
            sample_count += batch_size

            if arguments.log_interval_steps > 0 and global_step % arguments.log_interval_steps == 0:
                elapsed = time.perf_counter() - epoch_started_at
                estimated_epoch = elapsed * len(train_loader) / step_in_epoch
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "adapter": adapter_name,
                            "epoch": epoch + 1,
                            "step_in_epoch": step_in_epoch,
                            "steps_per_epoch": len(train_loader),
                            "loss": float(loss.detach().item()),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "train_samples_per_second": sample_count / max(elapsed, 1.0e-9),
                            "epoch_eta": format_duration(estimated_epoch - elapsed),
                        },
                        ensure_ascii=False,
                    )
                )

        distorted_metrics = evaluate(model, distorted_loader, device, arguments.precision)
        clean_metrics = evaluate(model, clean_loader, device, arguments.precision)
        improved, should_stop = early_stopping.update(distorted_metrics["roc_auc"])
        record = {
            "adapter": adapter_name,
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": loss_sum / max(1, sample_count),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started_at,
            # Early stopping follows the distorted split, because that is what the
            # adapter is for. The clean split is a drift diagnostic: if it falls
            # below the baseline, the adapter is buying robustness with accuracy.
            "validation_distorted": distorted_metrics,
            "validation_clean": clean_metrics,
            "clean_delta_vs_baseline": clean_metrics["roc_auc"] - baseline["clean"]["roc_auc"],
            "distorted_delta_vs_baseline": distorted_metrics["roc_auc"] - baseline["distorted"]["roc_auc"],
            "best_roc_auc": early_stopping.best,
            "improved": improved,
            "early_stop": should_stop,
        }
        print(json.dumps({"event": "epoch_end", **record}, ensure_ascii=False))
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if improved:
            save_lora_adapter(peft_model, output_dir)
            best_record = record
        if should_stop:
            print(json.dumps({"event": "early_stopping", "adapter": adapter_name, "epoch": epoch + 1}))
            break

    if best_record is None:
        raise RuntimeError(f"Adapter {adapter_name!r} never improved on its initial score; nothing was saved.")

    metadata = {
        "adapter": adapter_name,
        "created_at_utc": utc_now(),
        "operations": list(operations),
        # Read back from the pipeline, not from the CLI arguments: the operation
        # count is clamped to the size of the enabled set.
        "distortion_policy": {
            "min_operations": pipeline.policy.min_operations,
            "max_operations": pipeline.policy.max_operations,
            "severity_min": pipeline.policy.severity_min,
            "severity_max": pipeline.policy.severity_max,
            "applied_always": True,
        },
        "head_checkpoint": str(Path(arguments.head).resolve()),
        "head_experiment": checkpoint.get("experiment"),
        "head_frozen": True,
        "backbone_id": checkpoint["model"]["id"],
        "preprocessing": {
            "image_size": int(preprocessing["image_size"]),
            "image_mean": model.image_mean,
            "image_std": model.image_std,
        },
        "lora": {
            "rank": arguments.lora_rank,
            "alpha": arguments.lora_alpha,
            "dropout": arguments.lora_dropout,
            "target_regex": arguments.target_regex,
            "wrapped_modules": report["wrapped_modules"],
            "wrapped_by_suffix": report["wrapped_by_suffix"],
            "trainable_parameters": report["trainable_parameters"],
        },
        "training": {
            "source": train_source,
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "epochs_configured": arguments.epochs,
            "epochs_run": epochs_run,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "precision": arguments.precision,
            "seed": arguments.seed,
        },
        "baseline_metrics": baseline,
        "best_epoch": best_record["epoch"],
        "best_metrics": {
            "validation_distorted": best_record["validation_distorted"],
            "validation_clean": best_record["validation_clean"],
        },
        "wall_seconds": time.perf_counter() - started_at,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "adapter_complete",
                "adapter": adapter_name,
                "output": str(output_dir),
                "best_epoch": best_record["epoch"],
                "best_distorted_roc_auc": best_record["validation_distorted"]["roc_auc"],
                "baseline_distorted_roc_auc": baseline["distorted"]["roc_auc"],
                "duration": format_duration(metadata["wall_seconds"]),
            },
            ensure_ascii=False,
        )
    )
    return metadata


def cap_validation(samples: list[ImageSample], maximum: int) -> list[ImageSample]:
    """Shrink the validation split while keeping both classes represented."""
    if maximum >= len(samples):
        return samples
    if maximum < 2:
        raise ValueError("--validation-max-samples must be at least 2.")
    by_label: dict[int, list[ImageSample]] = {0: [], 1: []}
    for sample in samples:
        by_label[int(sample.label)].append(sample)
    real_quota = min(len(by_label[0]), maximum // 2)
    generated_quota = min(len(by_label[1]), maximum - real_quota)
    if real_quota == 0 or generated_quota == 0:
        raise ValueError("The validation split must keep both classes after capping.")
    kept = by_label[0][:real_quota] + by_label[1][:generated_quota]
    return sorted(kept, key=lambda sample: sample.sample_id)


def resolve_adapter_plan(arguments: argparse.Namespace) -> list[tuple[str, tuple[str, ...]]]:
    available = tuple(BUILTIN_DISTORTION_NAMES)
    if arguments.distortions:
        requested = tuple(name.strip() for name in arguments.distortions.split(",") if name.strip())
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"Unknown or non-working distortions: {unknown}. Available: {list(available)}.")
    else:
        requested = available

    if arguments.mode == "all":
        return [(ALL_ADAPTER_NAME, requested)]
    return [(name, (name,)) for name in requested]


def main() -> None:
    arguments = parse_args()
    if arguments.resume and arguments.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")

    environment = initialize_distributed()
    _configure_strict_fp32()
    set_global_seed(arguments.seed, deterministic=False)

    checkpoint = load_detector_checkpoint(arguments.head)
    validation_samples, _ = load_samples(arguments.dataset_config, "validation", require_labels=True)
    if arguments.validation_max_samples is not None:
        validation_samples = cap_validation(validation_samples, arguments.validation_max_samples)
    validation_ids = {sample.sample_id for sample in validation_samples}

    if arguments.shards:
        if arguments.subset is not None:
            raise ValueError("Pass either --shards or --subset, not both.")
        shard_samples, shard_names = load_shard_samples(arguments.dataset_config, arguments.shards)
        # The validation split is carved out of the train shards, so training on
        # whole shards has to drop those images or early stopping is meaningless.
        train_samples = [
            sample for sample in shard_samples if sample.sample_id not in validation_ids
        ]
        train_source = {
            "kind": "shards",
            "shards": shard_names,
            "shard_total": len(shard_samples),
            "validation_excluded": len(shard_samples) - len(train_samples),
        }
        labels = [int(sample.label) for sample in train_samples]
        if 0 not in labels or 1 not in labels:
            raise ValueError(f"Shards {shard_names} do not contain both binary classes.")
    else:
        subset_path = arguments.subset if arguments.subset is not None else DEFAULT_SUBSET
        train_samples = load_subset_samples(subset_path)
        train_source = {"kind": "subset", "path": str(Path(subset_path).resolve())}

    overlap = {sample.sample_id for sample in train_samples} & validation_ids
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} training images also appear in the validation split. "
            "Rebuild the subset with `python3 lora/make_subset.py --overwrite`."
        )

    plan = resolve_adapter_plan(arguments)
    print(
        json.dumps(
            {
                "event": "setup",
                "mode": arguments.mode,
                "adapters": [name for name, _ in plan],
                "head": str(Path(arguments.head).resolve()),
                "head_experiment": checkpoint.get("experiment"),
                "image_size": int(checkpoint["preprocessing"]["image_size"]),
                "train_source": train_source,
                "train_samples": len(train_samples),
                "train_real": sum(1 for sample in train_samples if sample.label == 0),
                "train_ai_generated": sum(1 for sample in train_samples if sample.label == 1),
                "validation_samples": len(validation_samples),
                "device": str(environment.device),
                "device_name": torch.cuda.get_device_name(environment.device),
                "precision": arguments.precision,
                "torch": torch.__version__,
                "peft": _peft_version(),
            },
            ensure_ascii=False,
        )
    )

    summaries: list[dict[str, Any]] = []
    for adapter_name, operations in plan:
        adapter_dir = arguments.output_root / adapter_name
        has_weights = (adapter_dir / "adapter_model.safetensors").is_file()
        # metadata.json is written only after the epoch loop finishes, so it is
        # the completion marker. Weights alone mean an interrupted run: they were
        # saved on an improving epoch that was never followed by a clean exit.
        is_complete = has_weights and (adapter_dir / "metadata.json").is_file()
        if is_complete:
            if arguments.resume:
                print(json.dumps({"event": "adapter_skipped", "adapter": adapter_name, "reason": "already trained"}))
                continue
            if not arguments.overwrite:
                raise FileExistsError(
                    f"{adapter_dir} already holds a completed adapter. "
                    "Pass --resume to skip it or --overwrite to retrain it."
                )
        elif has_weights:
            print(
                json.dumps(
                    {
                        "event": "adapter_incomplete",
                        "adapter": adapter_name,
                        "reason": "weights exist but metadata.json does not; the previous run was interrupted",
                        "action": "retraining from scratch",
                    }
                )
            )
        summaries.append(
            train_one_adapter(
                adapter_name=adapter_name,
                operations=operations,
                arguments=arguments,
                checkpoint=checkpoint,
                train_samples=train_samples,
                validation_samples=validation_samples,
                train_source=train_source,
                device=environment.device,
            )
        )

    print(
        json.dumps(
            {
                "event": "training_complete",
                "mode": arguments.mode,
                "trained": [
                    {
                        "adapter": summary["adapter"],
                        "best_distorted_roc_auc": summary["best_metrics"]["validation_distorted"]["roc_auc"],
                        "baseline_distorted_roc_auc": summary["baseline_metrics"]["distorted"]["roc_auc"],
                        "best_clean_roc_auc": summary["best_metrics"]["validation_clean"]["roc_auc"],
                        "baseline_clean_roc_auc": summary["baseline_metrics"]["clean"]["roc_auc"],
                    }
                    for summary in summaries
                ],
                "output_root": str(arguments.output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
