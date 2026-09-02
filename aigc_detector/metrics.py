from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    positive = logits >= 0
    result = np.empty_like(logits)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)
    return result


def binary_metrics(labels: list[int] | np.ndarray, logits: list[float] | np.ndarray) -> dict[str, float]:
    label_array = np.asarray(labels, dtype=np.int64)
    logit_array = np.asarray(logits, dtype=np.float64)
    if label_array.shape != logit_array.shape:
        raise ValueError(f"labels and logits must have the same shape, got {label_array.shape} and {logit_array.shape}.")
    if np.unique(label_array).size != 2:
        raise ValueError("ROC-AUC requires both binary classes in the evaluated split.")

    probabilities = sigmoid_numpy(logit_array)
    predictions = (logit_array >= 0.0).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(label_array, logit_array)),
        "average_precision": float(average_precision_score(label_array, logit_array)),
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(label_array, predictions)),
        "mean_probability": float(probabilities.mean()),
    }


def flatten_gathered(parts: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = [row for part in parts for row in part]
    rows.sort(key=lambda row: int(row["index"]))
    indices = [int(row["index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise RuntimeError("Evaluation/inference produced duplicate dataset indices.")
    return rows
