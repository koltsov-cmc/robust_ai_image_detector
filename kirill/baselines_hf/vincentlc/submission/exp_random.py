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


def resized_crop(image: torch.Tensor, top_H: int, top_W: int, crop_size: int, target_size: int):
    croped = image[:, top_H:top_H + crop_size, top_W:top_W + crop_size]
    if crop_size != target_size:
        x = croped.unsqueeze(0)  # [1, C, H, W]
        x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
        croped = x.squeeze(0)
    return croped


class DfDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels, load_img_func):
        self.image_dir = Path(image_dir)
        self.load_img_func = load_img_func
        df = pd.read_csv(labels)
        self.df = df.loc[df["is_distorted"] == 0]
        self.num_crops = 20

    def __len__(self):
        return len(self.df) * self.num_crops
    
    def __getitem__(self, idx):
        idx, sample = idx // self.num_crops, idx % self.num_crops
        if sample < (self.num_crops // 2):
            resized = ""
        else:
            resized = "resized_"
            sample -= (self.num_crops // 2)
        crop_name = f"{resized}{sample}"
        name = self.df.loc[idx, "image_name"]
        label = self.df.loc[idx, "label"]
        top_H = self.df.loc[idx, f"top_H_{crop_name}"]
        top_W = self.df.loc[idx, f"top_W_{crop_name}"]
        size = self.df.loc[idx, f"size_{crop_name}"]
        path = self.image_dir / name
        img = self.load_img_func(path)
        img = resized_crop(img, top_H, top_W, size, 256)
        return {"name": name, "crop_name": crop_name, "image": img, "label": label}


def evaluate(args, model: Model, labels_csv, results_csv):
    device = torch.device(args.device)
    image_dir = Path(args.images_dir)
    batch_size = int(args.batch_size)

    dataset = DfDataset(image_dir, labels_csv, load_image)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(args.num_workers), drop_last=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    results = []

    time_start = time.perf_counter()
    for data in tqdm(data_loader):
        names = data["name"]
        crop_names = data["crop_name"]
        labels = data["label"]
        img = data["image"].to(device)
        with torch.no_grad():
            scores = model.predict(img).detach().cpu()
        for score, name, crop_name, label in zip(scores, names, crop_names, labels):
            results.append({"image_name": name, "crop_name": crop_name, "score": score.item(), "label": int(label)})
    torch.cuda.synchronize(device)
        
    total_time = time.perf_counter() - time_start

    peak_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # MB
    peak_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    df = pd.DataFrame(results)
    df.attrs["total_time_sec"] = total_time
    df.attrs["frames"] = len(dataset)
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

    results_path = out_dir / "results_random_crop.csv"
    evaluate(args, model, labels_path, results_path)

if __name__ == "__main__":
    main()
