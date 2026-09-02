"""Build the fixed LoRA training subset and write it to lora_train_subset.csv.

This is a one-off preparation step. Every adapter is then trained on exactly the
rows this script selected, so the nine adapters differ only in the distortion
they see, never in the images.

The subset is drawn from shard_5, which the native detector never trained on
(its train split is ``all_except_validation``, i.e. shard_0 and shard_1). shard_5
is however also the native validation split, so the images that split already
uses are excluded here by asking the shared loader for them rather than by
re-deriving the sampling: same function, same seed, same result.

    python3 lora/make_subset.py
    python3 lora/make_subset.py --shard shard_5 --size 15000 --seed 3407
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aigc_detector.config import load_yaml  # noqa: E402
from aigc_detector.data import (  # noqa: E402
    ImageSample,
    _load_ntire_shards,
    _resolve_relative,
    load_samples,
)


DEFAULT_DATASET_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "lora" / "lora_train_subset.csv"
CSV_FIELDNAMES = ("image_name", "label", "shard", "path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--shard", default="shard_5", help="Shard directory to sample from.")
    parser.add_argument("--size", type=int, default=15000, help="Total number of images to select.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--keep-validation-overlap",
        action="store_true",
        help=(
            "Do not exclude the native validation split. Only use this when the "
            "adapters are validated on something else; otherwise it leaks."
        ),
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="Check that every selected image exists on disk (slower on a network filesystem).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV.")
    return parser.parse_args()


def load_shard_samples(dataset_config_path: Path, shard_name: str) -> list[ImageSample]:
    config = load_yaml(dataset_config_path)
    if config.get("format") != "ntire_2026":
        raise ValueError(
            f"{dataset_config_path} is not in the 'ntire_2026' format; this script reads shard directories."
        )
    root = _resolve_relative(Path(config["_config_dir"]), config.get("root", "../ntire"))
    train_directory = _resolve_relative(root, config.get("train_directory", "train"))
    shard_directory = train_directory / shard_name
    if not shard_directory.is_dir():
        available = sorted(path.name for path in train_directory.glob("shard_*") if path.is_dir())
        raise FileNotFoundError(
            f"Shard directory does not exist: {shard_directory}. Available shards: {available}."
        )
    return _load_ntire_shards(
        train_directory,
        [shard_directory],
        image_column=str(config.get("train_image_column", "image_name")),
        label_column=str(config.get("train_label_column", "label")),
        class_mapping={str(key): int(value) for key, value in config.get("classes", {}).items()},
    )


def stratified_sample(
    samples: list[ImageSample],
    *,
    size: int,
    seed: int,
) -> list[ImageSample]:
    by_label: dict[int, list[ImageSample]] = {0: [], 1: []}
    for sample in samples:
        if sample.label not in by_label:
            raise ValueError(f"Unexpected label {sample.label!r} for {sample.path}.")
        by_label[int(sample.label)].append(sample)

    real_quota = size // 2
    generated_quota = size - real_quota
    shortages = []
    if len(by_label[0]) < real_quota:
        shortages.append(f"real: need {real_quota}, have {len(by_label[0])}")
    if len(by_label[1]) < generated_quota:
        shortages.append(f"ai_generated: need {generated_quota}, have {len(by_label[1])}")
    if shortages:
        raise ValueError(
            "The eligible pool is too small for a balanced subset of "
            f"{size} images ({'; '.join(shortages)}). Lower --size or pick another shard."
        )

    rng = random.Random(seed)
    # Sort first so the pool order does not depend on directory iteration order.
    selected = rng.sample(sorted(by_label[0], key=lambda item: item.sample_id), real_quota)
    selected += rng.sample(sorted(by_label[1], key=lambda item: item.sample_id), generated_quota)
    return sorted(selected, key=lambda item: item.sample_id)


def write_subset(path: Path, shard_name: str, samples: list[ImageSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "image_name": sample.path.name,
                    "label": int(sample.label),
                    "shard": shard_name,
                    "path": str(sample.path),
                }
            )


def main() -> None:
    arguments = parse_args()
    if arguments.size <= 1:
        raise ValueError("--size must be at least 2.")
    if arguments.output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"{arguments.output} already exists. Pass --overwrite to rebuild the subset. "
            "Rebuilding changes which images every adapter trains on."
        )

    shard_samples = load_shard_samples(arguments.dataset_config, arguments.shard)
    shard_counts = Counter(int(sample.label) for sample in shard_samples)

    excluded_ids: set[str] = set()
    if not arguments.keep_validation_overlap:
        validation_samples, _ = load_samples(
            arguments.dataset_config, "validation", require_labels=True
        )
        excluded_ids = {sample.sample_id for sample in validation_samples}

    eligible = [sample for sample in shard_samples if sample.sample_id not in excluded_ids]
    overlap = len(shard_samples) - len(eligible)
    selected = stratified_sample(eligible, size=arguments.size, seed=arguments.seed)

    leaked = {sample.sample_id for sample in selected} & excluded_ids
    if leaked:
        raise RuntimeError(f"{len(leaked)} selected images are also in the validation split.")

    if arguments.verify_files:
        missing = [str(sample.path) for sample in selected if not sample.path.is_file()]
        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(f"{len(missing)} selected images are missing:\n{preview}")

    write_subset(arguments.output, arguments.shard, selected)
    selected_counts = Counter(int(sample.label) for sample in selected)
    print(
        json.dumps(
            {
                "event": "subset_written",
                "output": str(arguments.output),
                "shard": arguments.shard,
                "shard_total": len(shard_samples),
                "shard_real": shard_counts[0],
                "shard_ai_generated": shard_counts[1],
                "validation_excluded": overlap,
                "eligible_pool": len(eligible),
                "selected_total": len(selected),
                "selected_real": selected_counts[0],
                "selected_ai_generated": selected_counts[1],
                "seed": arguments.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
