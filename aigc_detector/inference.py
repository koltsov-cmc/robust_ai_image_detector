from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import AIGCImageDataset, load_samples, seed_worker
from .distributed import cleanup_distributed, initialize_distributed, set_global_seed
from .experiments import get_experiment
from .metrics import binary_metrics, flatten_gathered, sigmoid_numpy
from .model import EvaClipGAPClassifier


def _configure_strict_fp32() -> None:
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _load_checkpoint(path: Path, expected_experiment: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Best checkpoint does not exist: {path}. Train experiment '{expected_experiment}' first."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 3:
        raise ValueError(f"Unsupported checkpoint format in {path}.")
    if checkpoint.get("experiment") != expected_experiment:
        raise ValueError(
            f"Checkpoint belongs to experiment {checkpoint.get('experiment')!r}, "
            f"not {expected_experiment!r}."
        )
    return checkpoint


def _rounded_roc_auc(subset: list[dict[str, Any]]) -> float | None:
    if {int(row["label"]) for row in subset} != {0, 1}:
        return None
    return round(
        binary_metrics(
            [int(row["label"]) for row in subset],
            [float(row["_logit"]) for row in subset],
        )["roc_auc"],
        3,
    )


@torch.no_grad()
def run_inference(experiment_name: str) -> Path:
    requested_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if requested_world_size != 1:
        raise RuntimeError(
            "Inference is restricted to one process on one GPU. "
            "Run `python inference.py --experiment NAME`, not torchrun."
        )

    experiment = get_experiment(experiment_name)
    environment = initialize_distributed()
    try:
        _configure_strict_fp32()
        set_global_seed(experiment.seed, deterministic=False)
        checkpoint = _load_checkpoint(experiment.best_checkpoint_path, experiment.name)

        samples, _ = load_samples(
            experiment.dataset_config_path,
            "test",
            require_labels=False,
        )

        preprocessing = checkpoint["preprocessing"]
        model_config = experiment.model_config
        model_config["id"] = checkpoint["model"]["id"]
        model = EvaClipGAPClassifier.from_pretrained(
            model_config,
            image_size=int(preprocessing["image_size"]),
        ).float()
        model.head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        model.to(environment.device)
        model.eval()

        dataset = AIGCImageDataset(
            samples,
            image_size=preprocessing["image_size"],
            image_mean=preprocessing["image_mean"],
            image_std=preprocessing["image_std"],
            base_seed=experiment.seed,
            distortion_pipeline=None,
        )
        generator = torch.Generator()
        generator.manual_seed(experiment.seed)
        loader = DataLoader(
            dataset,
            batch_size=experiment.inference_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=experiment.inference_num_workers,
            pin_memory=environment.device.type == "cuda",
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        rows: list[dict[str, Any]] = []
        for batch in loader:
            pixel_values = batch["pixel_values"].to(environment.device, non_blocking=True)
            logits = model(pixel_values)
            for index, sample_id, path, label, is_distorted, logit in zip(
                batch["index"].tolist(),
                batch["sample_id"],
                batch["path"],
                batch["label"].tolist(),
                batch["is_distorted"].tolist(),
                logits.cpu().tolist(),
            ):
                rows.append(
                    {
                        "index": int(index),
                        "image_name": sample_id,
                        "_path": path,
                        "label": int(label),
                        "is_distorted": int(is_distorted),
                        "_logit": float(logit),
                    }
                )

        rows = flatten_gathered([rows])
        probabilities = sigmoid_numpy([float(row["_logit"]) for row in rows])
        for row, probability in zip(rows, probabilities.tolist()):
            row["pred"] = float(probability)

        output_path = experiment.prediction_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Exact challenge submission columns. Labels and paths stay internal so
        # a labeled local test set can be scored without polluting submission.
        fieldnames = ["image_name", "pred"]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        clean_rows = [row for row in rows if int(row["is_distorted"]) == 0]
        distorted_rows = [row for row in rows if int(row["is_distorted"]) == 1]
        summary = {
            "roc_auc_all": _rounded_roc_auc(rows),
            "roc_auc_clean": _rounded_roc_auc(clean_rows),
            "roc_auc_distorted": _rounded_roc_auc(distorted_rows),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return output_path
    finally:
        cleanup_distributed(environment)
