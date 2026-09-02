from __future__ import annotations

import json
import math
import os
import shutil
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .data import AIGCImageDataset, ImageSample, load_samples, seed_worker
from .distributed import cleanup_distributed, initialize_distributed, set_global_seed
from .augmentation_pipeline import DistortionPipeline
from .experiments import PROJECT_ROOT, Experiment, get_experiment
from .metrics import binary_metrics, flatten_gathered
from .model import EvaClipGAPClassifier


@dataclass
class EarlyStopping:
    patience: int
    best: float = -math.inf
    epochs_without_improvement: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        improved = value > self.best
        if improved:
            self.best = value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        should_stop = self.epochs_without_improvement >= self.patience
        return improved, should_stop


def _configure_strict_fp32() -> None:
    """Use FP32 matmuls rather than BF16/FP16 or NVIDIA TF32 shortcuts."""
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def _reset_experiment_output_dir(output_dir: Path, runs_root: Path) -> None:
    """Replace exactly one direct child of runs/ with a new empty directory."""
    runs_root = runs_root.resolve()
    if output_dir.is_symlink():
        raise RuntimeError(
            f"Refusing to clear a symbolic-link experiment directory: {output_dir}"
        )
    resolved_output = output_dir.resolve()
    if resolved_output.parent != runs_root:
        raise RuntimeError(
            f"Refusing to clear {resolved_output}: it is not a direct child of {runs_root}."
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError(
                f"Refusing to replace non-directory experiment output: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def _load_train_and_validation_samples(experiment: Experiment) -> tuple[list[ImageSample], list[ImageSample]]:
    train_samples, _ = load_samples(
        experiment.dataset_config_path,
        "train",
        require_labels=True,
    )
    validation_samples, _ = load_samples(
        experiment.dataset_config_path,
        "validation",
        require_labels=True,
    )
    return train_samples, validation_samples


def _make_datasets_and_loaders(
    experiment: Experiment,
    image_mean: list[float],
    image_std: list[float],
    train_samples: list[ImageSample],
    validation_samples: list[ImageSample],
    device: torch.device,
    distortion_pipeline: DistortionPipeline | None,
) -> tuple[AIGCImageDataset, DataLoader, DataLoader, dict[str, int]]:
    train_dataset = AIGCImageDataset(
        train_samples,
        image_size=experiment.image_size,
        image_mean=image_mean,
        image_std=image_std,
        base_seed=experiment.seed,
        distortion_pipeline=distortion_pipeline,
    )
    validation_dataset = AIGCImageDataset(
        validation_samples,
        image_size=experiment.image_size,
        image_mean=image_mean,
        image_std=image_std,
        base_seed=experiment.seed,
        distortion_pipeline=None,
    )

    generator = torch.Generator()
    generator.manual_seed(experiment.seed)
    loader_kwargs = {
        "batch_size": experiment.batch_size,
        "num_workers": experiment.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    labels = [int(sample.label) for sample in train_samples if sample.label is not None]
    counts = {
        "train_total": len(train_samples),
        "train_real": labels.count(0),
        "train_ai_generated": labels.count(1),
        "validation_total": len(validation_samples),
    }
    if counts["train_real"] == 0 or counts["train_ai_generated"] == 0:
        raise ValueError("The training split must contain both binary classes.")
    return train_dataset, train_loader, validation_loader, counts


@torch.no_grad()
def _validate(
    model: EvaClipGAPClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        logits = model(pixel_values)
        for index, label, logit in zip(
            batch["index"].tolist(),
            batch["label"].tolist(),
            logits.cpu().tolist(),
        ):
            rows.append({"index": int(index), "label": int(label), "logit": float(logit)})

    rows = flatten_gathered([rows])
    return binary_metrics(
        labels=[int(row["label"]) for row in rows],
        logits=[float(row["logit"]) for row in rows],
    )


def _checkpoint_payload(
    model: EvaClipGAPClassifier,
    experiment: Experiment,
    *,
    epoch: int,
    global_step: int,
    best_roc_auc: float,
    epochs_without_improvement: int,
    validation_metrics: dict[str, float],
    image_mean: list[float],
    image_std: list[float],
) -> dict[str, Any]:
    return {
        "format_version": 3,
        "experiment": experiment.name,
        "epoch": epoch,
        "global_step": global_step,
        "best_roc_auc": best_roc_auc,
        "epochs_without_improvement": epochs_without_improvement,
        "validation_metrics": validation_metrics,
        "model": {
            "id": experiment.model_id,
            "architecture": "EVA02-CLIP-B/16",
            "hidden_size": model.hidden_size,
            "patch_size": list(model.patch_size),
            "num_prefix_tokens": model.num_prefix_tokens,
            "head": "gap_patch_tokens_plus_linear_768_to_1",
        },
        "preprocessing": {
            "image_size": experiment.image_size,
            "resize": "bicubic_squish",
            "image_mean": image_mean,
            "image_std": image_std,
            "precision": "fp32",
        },
        "head_state_dict": {
            key: value.detach().cpu() for key, value in model.head.state_dict().items()
        },
    }


def run_training(experiment_name: str) -> None:
    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if requested_world_size != 1:
        raise RuntimeError(
            "Training is restricted to one process on one GPU. "
            "Run `python train.py --experiment NAME`, not torchrun."
        )

    experiment = get_experiment(experiment_name)
    environment = initialize_distributed()
    status_path: Path | None = None
    started_at = _utc_now()
    try:
        _configure_strict_fp32()
        set_global_seed(experiment.seed, deterministic=False)

        output_dir = experiment.output_dir
        _reset_experiment_output_dir(output_dir, PROJECT_ROOT / "runs")
        status_path = output_dir / "status.json"
        _write_status(
            status_path,
            {
                "status": "running",
                "experiment": experiment.name,
                "started_at_utc": started_at,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "device": str(environment.device),
                "device_name": torch.cuda.get_device_name(environment.device),
            },
        )

        # Manifests are checked before any model download.
        train_samples, validation_samples = _load_train_and_validation_samples(experiment)
        distortion_config = experiment.distortion_config
        distortion_pipeline = (
            DistortionPipeline.from_config(distortion_config)
            if distortion_config is not None
            else None
        )
        model = EvaClipGAPClassifier.from_pretrained(
            experiment.model_config,
            image_size=experiment.image_size,
        ).float()
        image_mean = list(model.image_mean)
        image_std = list(model.image_std)
        model.to(environment.device)

        train_dataset, train_loader, validation_loader, counts = _make_datasets_and_loaders(
            experiment,
            image_mean,
            image_std,
            train_samples,
            validation_samples,
            environment.device,
            distortion_pipeline,
        )

        head_parameters = list(model.head.parameters())
        optimizer = AdamW(
            head_parameters,
            lr=experiment.learning_rate,
            weight_decay=experiment.weight_decay,
        )
        total_steps = experiment.max_epochs * len(train_loader)
        warmup_steps = min(
            round(total_steps * experiment.warmup_ratio),
            max(0, total_steps - 1),
        )
        scheduler = _build_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=experiment.min_lr_ratio,
        )
        criterion = nn.BCEWithLogitsLoss()

        print(
            json.dumps(
                {
                    "event": "setup",
                    "experiment": experiment.name,
                    "device": str(environment.device),
                    "precision": "fp32_strict_no_tf32",
                    "augmentation_mode": experiment.augmentation_mode,
                    "max_epochs": experiment.max_epochs,
                    "early_stopping_patience": experiment.early_stopping_patience,
                    "steps_per_epoch": len(train_loader),
                    "trainable_parameters": sum(parameter.numel() for parameter in head_parameters),
                    "trainable_parameter_names": model.trainable_parameter_names(),
                    "distortion_operations": (
                        list(distortion_pipeline.operations)
                        if distortion_pipeline is not None
                        else []
                    ),
                    "distortion_stage": (
                        "after_bicubic_squish_to_model_input"
                        if distortion_pipeline is not None
                        else None
                    ),
                    "jpeg_ai_backend": (
                        None
                        if distortion_pipeline is None
                        or distortion_pipeline.jpeg_ai_backend is None
                        else {
                            "name": distortion_pipeline.jpeg_ai_backend.name,
                            "version": distortion_pipeline.jpeg_ai_backend.version,
                            "profile": distortion_pipeline.jpeg_ai_backend.profile,
                        }
                    ),
                    **counts,
                },
                ensure_ascii=False,
            )
        )

        history_path = output_dir / "history.jsonl"
        global_step = 0
        early_stopping = EarlyStopping(patience=experiment.early_stopping_patience)
        completed_epochs = 0

        for epoch in range(experiment.max_epochs):
            epoch_started_at = time.perf_counter()
            train_dataset.set_epoch(epoch)
            model.train()
            local_loss_sum = 0.0
            local_sample_count = 0

            for step_in_epoch, batch in enumerate(train_loader, start=1):
                pixel_values = batch["pixel_values"].to(environment.device, non_blocking=True)
                labels = batch["label"].to(environment.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(pixel_values)
                loss = criterion(logits, labels)
                loss.backward()
                if experiment.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(head_parameters, experiment.max_grad_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1

                batch_size = labels.numel()
                local_loss_sum += float(loss.detach().item()) * batch_size
                local_sample_count += batch_size

                if (
                    experiment.log_interval_steps > 0
                    and global_step % experiment.log_interval_steps == 0
                ):
                    progress_measured_at = time.perf_counter()
                    epoch_elapsed_seconds = progress_measured_at - epoch_started_at
                    current_samples_per_second = (
                        local_sample_count / max(epoch_elapsed_seconds, 1.0e-9)
                    )
                    estimated_epoch_seconds = (
                        epoch_elapsed_seconds * len(train_loader) / step_in_epoch
                    )
                    epoch_eta_seconds = max(
                        0.0, estimated_epoch_seconds - epoch_elapsed_seconds
                    )
                    remaining_full_epochs = experiment.max_epochs - (epoch + 1)
                    training_eta_seconds = (
                        epoch_eta_seconds
                        + remaining_full_epochs * estimated_epoch_seconds
                    )
                    estimated_full_training_seconds = (
                        estimated_epoch_seconds * experiment.max_epochs
                    )
                    print(
                        json.dumps(
                            {
                                "event": "train_step",
                                "epoch": epoch + 1,
                                "step_in_epoch": step_in_epoch,
                                "steps_per_epoch": len(train_loader),
                                "loss": float(loss.detach().item()),
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "train_samples_per_second": current_samples_per_second,
                                "estimated_epoch_duration": _format_duration(
                                    estimated_epoch_seconds
                                ),
                                "epoch_eta": _format_duration(epoch_eta_seconds),
                                "estimated_full_training_duration": _format_duration(
                                    estimated_full_training_seconds
                                ),
                                "training_eta": _format_duration(training_eta_seconds),
                            },
                            ensure_ascii=False,
                        )
                    )

            train_loss = local_loss_sum / max(1, local_sample_count)
            validation_metrics = _validate(model, validation_loader, environment.device)
            current_roc_auc = validation_metrics["roc_auc"]
            improved, should_stop = early_stopping.update(current_roc_auc)

            epoch_seconds = time.perf_counter() - epoch_started_at
            samples_per_second = local_sample_count / max(epoch_seconds, 1.0e-9)
            completed_epochs = epoch + 1
            epoch_record = {
                "epoch": completed_epochs,
                "global_step": global_step,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": epoch_seconds,
                "train_samples_per_second": samples_per_second,
                "validation": validation_metrics,
                "best_roc_auc": early_stopping.best,
                "improved": improved,
                "epochs_without_improvement": early_stopping.epochs_without_improvement,
                "early_stop": should_stop,
            }
            print(json.dumps({"event": "epoch_end", **epoch_record}, ensure_ascii=False))
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

            payload = _checkpoint_payload(
                model,
                experiment,
                epoch=epoch,
                global_step=global_step,
                best_roc_auc=early_stopping.best,
                epochs_without_improvement=early_stopping.epochs_without_improvement,
                validation_metrics=validation_metrics,
                image_mean=image_mean,
                image_std=image_std,
            )
            torch.save(payload, output_dir / f"checkpoint_epoch_{completed_epochs:03d}.pt")
            if improved:
                torch.save(payload, experiment.best_checkpoint_path)

            if should_stop:
                print(
                    json.dumps(
                        {
                            "event": "early_stopping",
                            "epoch": completed_epochs,
                            "reason": (
                                f"No clean validation ROC-AUC improvement for "
                                f"{experiment.early_stopping_patience} consecutive epochs."
                            ),
                            "best_roc_auc": early_stopping.best,
                        },
                        ensure_ascii=False,
                    )
                )
                break

        print(
            json.dumps(
                {
                    "event": "training_complete",
                    "experiment": experiment.name,
                    "completed_epochs": completed_epochs,
                    "best_roc_auc": early_stopping.best,
                    "best_checkpoint": str(experiment.best_checkpoint_path),
                },
                ensure_ascii=False,
            )
        )
        _write_status(
            status_path,
            {
                "status": "completed",
                "experiment": experiment.name,
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "device": str(environment.device),
                "device_name": torch.cuda.get_device_name(environment.device),
                "completed_epochs": completed_epochs,
                "best_roc_auc": early_stopping.best,
                "best_checkpoint": str(experiment.best_checkpoint_path),
            },
        )
    except BaseException as error:
        if status_path is not None:
            _write_status(
                status_path,
                {
                    "status": "failed",
                    "experiment": experiment.name,
                    "started_at_utc": started_at,
                    "finished_at_utc": _utc_now(),
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise
    finally:
        cleanup_distributed(environment)
