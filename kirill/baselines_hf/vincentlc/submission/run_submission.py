# run_submission.py
# Usage:
#   python run_submission.py --images_dir /path/to/images --out_dir /path/to/out --save_every 200 --device cuda --batch_size 32
#
# Creates/updates: <out_dir>/predictions.csv with columns: image_name,score
# Appends results every N images and supports resume (skips already processed image_name).

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
import numpy as np

from src.model import Model
import sys
from tqdm.auto import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(images_dir):
    images = []
    for p in Path(images_dir).rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            images.append(p)
    images.sort()
    return images


def load_processed(csv_path):
    processed = set()
    if not csv_path.exists():
        return processed

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["image_name"])

    return processed


def save_rows(csv_path, rows):
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["image_name", "score"])

        writer.writerows(rows)


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "predictions.csv"

    image_paths = list_images(images_dir)
    processed = load_processed(csv_path)

    print(f"Total images: {len(image_paths)}")
    print(f"Already processed: {len(processed)}")

    model = Model(device=args.device, model_data_dir="weights")

    buffer = []
    batch_images = []
    batch_names = []

    processed_counter = 0

    remaining = [p for p in image_paths if p.name not in processed]

    pbar = tqdm(
        total=len(remaining),
        desc="Inference",
        unit="img",
        dynamic_ncols=True,
        file=sys.stdout,   
        disable=False,     
        mininterval=0.5,
    )

    for path in remaining:
        name = path.name

        img = load_image(path)
        batch_images.append(img)
        batch_names.append(name)

        pbar.update(1)

        if len(batch_images) == args.batch_size:
            images = torch.stack(batch_images).to(args.device)

            with torch.no_grad():
                scores = model.predict(images)

            scores = scores.detach().cpu()

            for n, s in zip(batch_names, scores):
                buffer.append((n, float(s)))

            batch_images.clear()
            batch_names.clear()

            processed_counter += len(scores)
            pbar.set_postfix_str(f"saved={processed_counter}", refresh=False)

            if len(buffer) >= args.save_every:
                save_rows(csv_path, buffer)
                buffer.clear()
                #print(f"[checkpoint] wrote {args.save_every} rows -> {csv_path}", flush=True)

    if batch_images:

        images = torch.stack(batch_images).to(args.device)

        with torch.no_grad():
            scores = model.predict(images)

        scores = scores.detach().cpu()

        for n, s in zip(batch_names, scores):
            buffer.append((n, float(s)))

    if buffer:
        save_rows(csv_path, buffer)

    pbar.close()

    print("Finished inference")


if __name__ == "__main__":
    main()
