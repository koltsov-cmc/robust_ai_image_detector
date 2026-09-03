# run_submission.py
# Usage:
#   python run_submission.py --images_dir /path/to/images --out_dir /path/to/out --save_every 200 --device cuda --batch_size 32
#
# Creates/updates: <out_dir>/predictions.csv with columns: image_name,score
# Appends results every N images and supports resume (skips already processed image_name).

import argparse
import csv
from pathlib import Path

from PIL import Image
import numpy as np

from src.model import Model
import sys
from tqdm.auto import tqdm

import pandas as pd

import torch
import torch.nn.functional as F
import json
import time


def save_df_attrs_to_json(df: pd.DataFrame, results_path: str) -> str:
    csv_path = Path(results_path)
    json_path = csv_path.with_suffix(".json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dict(df.attrs), f, ensure_ascii=False, indent=2, default=str)

    return str(json_path)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def downsample(image: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Уменьшает изображение [C, H, W] в scale раз по H и W.
    scale: float в (0, 1)
    """
    if not (isinstance(image, torch.Tensor) and image.ndim == 3):
        raise ValueError("image должен быть тензором формы [C, H, W]")
    if not (0.0 < scale < 1.0):
        raise ValueError("scale должен быть в диапазоне (0, 1)")

    c, h, w = image.shape
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # F.interpolate ожидает [N, C, H, W]
    x = image.unsqueeze(0)
    y = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return y.squeeze(0)


def crop(image: torch.Tensor, scale: float) -> torch.Tensor:
    c, h, w = image.shape
    crop_h = max(1, int(round(h * scale)))
    crop_w = max(1, int(round(w * scale)))

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    return image[:, top:top + crop_h, left:left + crop_w]


def square_crop(image: torch.Tensor, size: int) -> torch.Tensor:
    c, h, w = image.shape

    pad_h = max(0, size - h)
    pad_w = max(0, size - w)

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = image.unsqueeze(0)
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        image = x.squeeze(0)
        _, h, w = image.shape

    top = (h - size) // 2
    left = (w - size) // 2
    return image[:, top:top + size, left:left + size]


def square_crop_resize_long_side(image: torch.Tensor, size: int) -> torch.Tensor:
    c, h, w = image.shape
    long_side = max(h, w)

    if long_side != size:
        scale = size / long_side
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        x = image.unsqueeze(0)  # [1, C, H, W]
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        image = x.squeeze(0)
        _, h, w = image.shape

    pad_h = max(0, size - h)
    pad_w = max(0, size - w)

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = image.unsqueeze(0)
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        image = x.squeeze(0)
        _, h, w = image.shape

    top = (h - size) // 2
    left = (w - size) // 2
    return image[:, top:top + size, left:left + size]


def square_crop_resize_short_side(image: torch.Tensor, size: int) -> torch.Tensor:
    c, h, w = image.shape
    short_side = min(h, w)

    if short_side != size:
        scale = size / short_side
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        x = image.unsqueeze(0)  # [1, C, H, W]
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        image = x.squeeze(0)
        _, h, w = image.shape

    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = image.unsqueeze(0)
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        image = x.squeeze(0)
        _, h, w = image.shape

    top = (h - size) // 2
    left = (w - size) // 2
    return image[:, top:top + size, left:left + size]

def identiti(image, fictive):
    return image

class DfDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels, load_img_func, distort, distort_param):
        self.image_dir = Path(image_dir)
        self.load_img_func = load_img_func
        self.distort = distort
        self.distort_param = distort_param
        df = pd.read_csv(labels)
        self.df = df.loc[df["is_distorted"] == 0]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        name = self.df.loc[idx, "image_name"]
        label = self.df.loc[idx, "label"]
        path = self.image_dir / name
        img = self.load_img_func(path)
        img = self.distort(img, self.distort_param)
        return {"name": name, "image": img, "label": label}


def evaluate(args, model: Model, labels_csv, results_csv, distort, distort_param):
    device = torch.device(args.device)
    image_dir = Path(args.images_dir)
    batch_size = int(args.batch_size)

    dataset = DfDataset(image_dir, labels_csv, load_image, distort, distort_param)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(args.num_workers), drop_last=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    results = []

    time_start = time.perf_counter()
    for data in tqdm(data_loader):
        names = data["name"]
        labels = data["label"]
        img = data["image"].to(device)
        with torch.no_grad():
            scores = model.predict(img).detach().cpu()
        for score, name, label in zip(scores, names, labels):
            results.append({"image_name": name, "score": score.item(), "label": int(label)})
    torch.cuda.synchronize(device)
        
    total_time = time.perf_counter() - time_start

    peak_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # MB
    peak_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    df = pd.DataFrame(results)
    df.attrs["total_time_sec"] = total_time
    df.attrs["peak_memory_allocated_mb"] = peak_memory_allocated
    df.attrs["peak_memory_reserved_mb"] = peak_memory_reserved
    df.to_csv(results_csv, index=False)
    save_df_attrs_to_json(df, results_csv)

    return df, total_time, peak_memory_allocated, peak_memory_reserved

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--labels", default="")
    parser.add_argument("--num_workers", default=12)

    args = parser.parse_args()

    model = Model(device=args.device, model_data_dir="weights")

    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    distortions = {
        "crop": square_crop,
        "long_crop": square_crop_resize_long_side,
        "short_crop": square_crop_resize_short_side
    }

    # for distort_name, func in distortions.items():
    #     for size in [512, 256]: 
    #         results_path = out_dir / f"results_{distort_name}_{size}.csv"
    #         evaluate(args, model, labels_path, results_path, func, size)

    args.batch_size = 1
    args.num_workers = 1
    results_path = out_dir / f"results_original.csv"
    evaluate(args, model, labels_path, results_path, identiti, True)

if __name__ == "__main__":
    main()
