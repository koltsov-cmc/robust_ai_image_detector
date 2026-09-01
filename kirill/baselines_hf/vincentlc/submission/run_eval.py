# run_eval.py
# Run the detector over a labelled image folder and report detection metrics.
#
# Usage:
#   python run_eval.py \
#       --images_dir /data2/aidetection/ntire/test/test_images \
#       --labels /data2/aidetection/ntire/test/test_labels.csv \
#       --out_dir ./results \
#       --device cuda --batch_size 32 --num_workers 8
#
# Writes <out_dir>/predictions.csv (image_name,score,label,is_distorted,distortions)
# and <out_dir>/metrics.json (overall + per-subset metrics, timing, peak memory).

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.model import Model


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class LabelledImageDataset(torch.utils.data.Dataset):
    """Yields images already resized to the model resolution so they can be batched."""

    def __init__(self, image_dir, df, resize):
        self.image_dir = Path(image_dir)
        self.df = df.reset_index(drop=True)
        self.resize = resize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = load_image(self.image_dir / row["image_name"])
        if self.resize is not None:
            img = self.resize(img)
        return {"name": row["image_name"], "image": img, "label": int(row["label"])}


def roc_auc(labels, scores):
    """Rank based ROC AUC with tie correction. Avoids a scikit-learn dependency."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if (labels == 1).sum() == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / np.arange(1, len(sorted_labels) + 1)
    return float((precision * sorted_labels).sum() / (labels == 1).sum())


def compute_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    preds = (scores >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    n = len(labels)
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    return {
        "n": n,
        "n_real": int((labels == 0).sum()),
        "n_generated": int((labels == 1).sum()),
        "accuracy": (tp + tn) / n if n else float("nan"),
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "tpr_recall_generated": tpr,
        "tnr_specificity_real": tnr,
        "precision": precision,
        "f1": 2 * precision * tpr / (precision + tpr) if (precision + tpr) else float("nan"),
        "auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def parse_distortions(value):
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--weights_dir", default="weights")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--no_preresize",
        action="store_true",
        help="Resize inside the model instead of the dataloader; forces batch_size=1.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = Model(device=args.device, model_data_dir=args.weights_dir)

    df = pd.read_csv(args.labels)
    df = df[["image_name", "label", "distortions", "distortion_scales", "is_distorted"]].reset_index(drop=True)
    print(f"Images in labels file: {len(df)}")

    resize = None if args.no_preresize else model.resize
    batch_size = 1 if args.no_preresize else args.batch_size

    dataset = LabelledImageDataset(args.images_dir, df, resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    rows = []
    start = time.perf_counter()
    for batch in tqdm(loader, desc="Inference", dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            scores = model.predict(images).detach().cpu()
        for name, score, label in zip(batch["name"], scores, batch["label"]):
            rows.append({"image_name": name, "score": float(score), "label": int(label)})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time = time.perf_counter() - start

    preds = pd.DataFrame(rows).merge(
        df[["image_name", "distortions", "distortion_scales", "is_distorted"]],
        on="image_name",
        how="left",
    )
    preds.to_csv(out_dir / "predictions.csv", index=False)

    metrics = {
        "overall": compute_metrics(preds["label"].values, preds["score"].values, args.threshold),
        "threshold": args.threshold,
        "total_time_sec": total_time,
        "images_per_sec": len(preds) / total_time if total_time else float("nan"),
        "batch_size": batch_size,
        "preresize_in_dataloader": not args.no_preresize,
    }

    if device.type == "cuda":
        metrics["peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        metrics["peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    for flag, key in [(0, "clean"), (1, "distorted")]:
        subset = preds[preds["is_distorted"] == flag]
        if len(subset):
            metrics[key] = compute_metrics(subset["label"].values, subset["score"].values, args.threshold)

    per_distortion = {}
    preds["_distortion_list"] = preds["distortions"].apply(parse_distortions)
    for name in sorted({d for lst in preds["_distortion_list"] for d in lst}):
        subset = preds[preds["_distortion_list"].apply(lambda lst: name in lst)]
        per_distortion[name] = compute_metrics(subset["label"].values, subset["score"].values, args.threshold)
    metrics["per_distortion"] = per_distortion

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    o = metrics["overall"]
    print("\n=== Overall ===")
    print(f"images={o['n']} real={o['n_real']} generated={o['n_generated']}")
    print(f"AUC={o['auc']:.4f}  AP={o['average_precision']:.4f}")
    print(f"acc@{args.threshold}={o['accuracy']:.4f}  balanced_acc={o['balanced_accuracy']:.4f}")
    print(f"TPR(generated)={o['tpr_recall_generated']:.4f}  TNR(real)={o['tnr_specificity_real']:.4f}")
    for key in ("clean", "distorted"):
        if key in metrics:
            m = metrics[key]
            print(f"{key:>9}: n={m['n']} AUC={m['auc']:.4f} acc={m['accuracy']:.4f}")
    if per_distortion:
        print("\n=== Per distortion ===")
        for name, m in per_distortion.items():
            print(f"{name:>28}: n={m['n']:5d} AUC={m['auc']:.4f} acc={m['accuracy']:.4f}")
    print(f"\ntime={total_time:.1f}s  ({metrics['images_per_sec']:.1f} img/s)")
    print(f"Wrote {out_dir / 'predictions.csv'} and {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
