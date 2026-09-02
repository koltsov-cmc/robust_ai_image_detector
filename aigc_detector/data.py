from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .config import load_yaml
from .augmentation_pipeline import DistortionPipeline


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int | None
    sample_id: str
    is_distorted: bool | None = None


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]

    if suffix in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}.")
                rows.append(row)
        return rows

    raise ValueError(
        f"Unsupported manifest format '{suffix}' for {path}. "
        "Use CSV, TSV, JSONL, or NDJSON."
    )


def _resolve_relative(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _parse_label(value: Any, class_mapping: dict[str, int]) -> int:
    if isinstance(value, bool):
        label = int(value)
    elif isinstance(value, (int, np.integer)):
        label = int(value)
    elif isinstance(value, float) and value.is_integer():
        label = int(value)
    else:
        text = str(value).strip()
        normalized = text.casefold()
        normalized_mapping = {str(key).casefold(): int(mapped) for key, mapped in class_mapping.items()}
        if normalized in normalized_mapping:
            label = normalized_mapping[normalized]
        else:
            try:
                label = int(text)
            except ValueError as error:
                raise ValueError(
                    f"Unknown label {value!r}. Define it under 'classes' in configs/dataset.yaml."
                ) from error

    if label not in {0, 1}:
        raise ValueError(f"Binary label must be 0 or 1, got {label!r}.")
    return label


def _require_mapping(config: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' in {source} must be a YAML mapping.")
    return value


def _shard_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"shard_(\d+)", path.name)
    return (int(match.group(1)), path.name) if match else (2**31 - 1, path.name)


def _configured_shard_names(value: Any, available: list[Path], *, field: str) -> list[str]:
    available_names = [path.name for path in available]
    if value == "all":
        return available_names
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{field}' must be 'all' or a non-empty YAML list of shard numbers/names.")
    result = [str(item) if str(item).startswith("shard_") else f"shard_{item}" for item in value]
    missing = sorted(set(result) - set(available_names))
    if missing:
        raise FileNotFoundError(
            f"Configured NTIRE shards are not extracted under the train directory: {missing}. "
            f"Available extracted shards: {available_names}. ZIP files are not read directly."
        )
    return result


def _detect_column(
    rows: list[dict[str, Any]],
    configured: str | None,
    candidates: tuple[str, ...],
    *,
    manifest: Path,
) -> str:
    if not rows:
        raise ValueError(f"Manifest contains zero rows: {manifest}")
    keys = tuple(rows[0])
    if configured:
        if configured not in keys:
            raise ValueError(f"Column {configured!r} is absent from {manifest}; found {list(keys)}.")
        return configured
    for candidate in candidates:
        if candidate in keys:
            return candidate
    meaningful = [key for key in keys if key and not key.casefold().startswith("unnamed:")]
    if len(meaningful) == 1:
        return meaningful[0]
    raise ValueError(
        f"Could not identify a column in {manifest}. Expected one of {list(candidates)}; "
        f"found {list(keys)}. Set the column explicitly in configs/dataset.yaml."
    )


def _load_ntire_shards(
    train_directory: Path,
    shard_directories: list[Path],
    *,
    image_column: str,
    label_column: str,
    class_mapping: dict[str, int],
) -> list[ImageSample]:
    del train_directory
    samples: list[ImageSample] = []
    seen_ids: set[str] = set()
    for shard in shard_directories:
        images_directory = shard / "images"
        labels_path = shard / "labels.csv"
        if not images_directory.is_dir():
            raise FileNotFoundError(f"NTIRE shard has no images directory: {images_directory}")
        rows = _read_manifest(labels_path)
        detected_image_column = _detect_column(
            rows,
            image_column,
            ("image_name", "filename", "file_name", "path"),
            manifest=labels_path,
        )
        detected_label_column = _detect_column(
            rows,
            label_column,
            ("label", "label_num", "target"),
            manifest=labels_path,
        )
        for row_number, row in enumerate(rows, start=2):
            image_name = str(row.get(detected_image_column, "")).strip()
            if not image_name:
                raise ValueError(
                    f"Empty image name in {labels_path}:{row_number}, column {detected_image_column!r}."
                )
            sample_id = Path(image_name).stem
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate NTIRE image id {sample_id!r} found in {labels_path}.")
            seen_ids.add(sample_id)
            samples.append(
                ImageSample(
                    path=(images_directory / image_name).resolve(),
                    label=_parse_label(row.get(detected_label_column), class_mapping),
                    sample_id=sample_id,
                )
            )
    return samples


def _sample_validation(
    samples: list[ImageSample],
    *,
    maximum: int | None,
    seed: int,
    sampling: str,
) -> list[ImageSample]:
    if maximum is None or maximum >= len(samples):
        return samples
    if maximum <= 1:
        raise ValueError("NTIRE validation max_samples must be at least 2.")
    rng = random.Random(seed)
    if sampling == "random":
        selected = rng.sample(samples, maximum)
    elif sampling == "balanced":
        by_label = {
            label: [sample for sample in samples if sample.label == label] for label in (0, 1)
        }
        real_count = maximum // 2
        generated_count = maximum - real_count
        if len(by_label[0]) < real_count or len(by_label[1]) < generated_count:
            raise ValueError(
                "Not enough examples of both classes for the requested balanced validation subset."
            )
        selected = rng.sample(by_label[0], real_count) + rng.sample(
            by_label[1], generated_count
        )
    else:
        raise ValueError("NTIRE validation sampling must be 'balanced' or 'random'.")
    return sorted(selected, key=lambda sample: sample.sample_id)


def _load_optional_test_labels(
    manifest: Path,
    *,
    image_column: str | None,
    label_column: str | None,
    class_mapping: dict[str, int],
) -> dict[str, tuple[int, bool | None]]:
    if not manifest.is_file():
        return {}
    rows = _read_manifest(manifest)
    detected_image_column = _detect_column(
        rows,
        image_column,
        ("image_name", "filename", "file_name", "path"),
        manifest=manifest,
    )
    detected_label_column = _detect_column(
        rows,
        label_column,
        ("label", "label_num", "target"),
        manifest=manifest,
    )
    keys = tuple(rows[0])
    distortion_column = next(
        (candidate for candidate in ("is_distorted", "distorted") if candidate in keys),
        None,
    )
    labels: dict[str, tuple[int, bool | None]] = {}
    for row in rows:
        raw_name = str(row.get(detected_image_column, "")).strip()
        if not raw_name:
            continue
        label = _parse_label(row.get(detected_label_column), class_mapping)
        if distortion_column is None or row.get(distortion_column) in {None, ""}:
            is_distorted = None
        else:
            raw_distorted = str(row[distortion_column]).strip().casefold()
            if raw_distorted in {"0", "false", "no"}:
                is_distorted = False
            elif raw_distorted in {"1", "true", "yes"}:
                is_distorted = True
            else:
                raise ValueError(
                    f"Expected binary {distortion_column!r} in {manifest}, got "
                    f"{row[distortion_column]!r}."
                )
        for key in (raw_name, Path(raw_name).name, Path(raw_name).stem):
            labels[key] = (label, is_distorted)
    return labels


def _load_ntire_test(
    root: Path,
    split_spec: dict[str, Any],
    *,
    class_mapping: dict[str, int],
    require_labels: bool,
) -> list[ImageSample]:
    image_directory = _resolve_relative(
        root, split_spec.get("directory", "test/test_images")
    )
    if not image_directory.is_dir():
        raise FileNotFoundError(f"NTIRE test image directory does not exist: {image_directory}")

    raw_manifest = split_spec.get("images_manifest")
    if raw_manifest:
        manifest = _resolve_relative(root, raw_manifest)
        rows = _read_manifest(manifest)
        image_column = _detect_column(
            rows,
            split_spec.get("image_column"),
            ("image_name", "filename", "file_name", "path"),
            manifest=manifest,
        )
        image_names = [str(row.get(image_column, "")).strip() for row in rows]
    else:
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        image_names = [
            path.name
            for path in sorted(image_directory.iterdir())
            if path.is_file() and path.suffix.casefold() in extensions
        ]

    labels_manifest_value = split_spec.get("labels_manifest")
    labels = (
        _load_optional_test_labels(
            _resolve_relative(root, labels_manifest_value),
            image_column=split_spec.get("labels_image_column"),
            label_column=split_spec.get("label_column"),
            class_mapping=class_mapping,
        )
        if labels_manifest_value
        else {}
    )

    default_extension = str(split_spec.get("default_extension", ".jpg"))
    samples: list[ImageSample] = []
    seen_ids: set[str] = set()
    for row_number, raw_name in enumerate(image_names, start=2):
        if not raw_name:
            raise ValueError(f"Empty image name in NTIRE test manifest row {row_number}.")
        name_path = Path(raw_name)
        file_name = name_path.name if name_path.suffix else name_path.name + default_extension
        sample_id = Path(file_name).stem
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate NTIRE test image id: {sample_id!r}.")
        seen_ids.add(sample_id)
        label_record = labels.get(raw_name, labels.get(name_path.name, labels.get(sample_id)))
        label = None if label_record is None else label_record[0]
        is_distorted = None if label_record is None else label_record[1]
        if require_labels and label is None:
            raise ValueError(f"No label found for NTIRE test image {raw_name!r}.")
        samples.append(
            ImageSample(
                path=(image_directory / file_name).resolve(),
                label=label,
                sample_id=sample_id,
                is_distorted=is_distorted,
            )
        )
    return samples


def _load_ntire_samples(
    config: dict[str, Any],
    split_name: str,
    *,
    require_labels: bool,
) -> list[ImageSample]:
    config_path = Path(config["_config_path"])
    config_directory = Path(config["_config_dir"])
    root = _resolve_relative(config_directory, config.get("root", "../ntire"))
    splits = _require_mapping(config, "splits", config_path)
    if split_name not in splits or not isinstance(splits[split_name], dict):
        raise KeyError(f"NTIRE split {split_name!r} is not configured in {config_path}.")
    split_spec = splits[split_name]
    class_mapping = {str(key): int(value) for key, value in config.get("classes", {}).items()}

    if split_name == "test":
        samples = _load_ntire_test(
            root,
            split_spec,
            class_mapping=class_mapping,
            require_labels=require_labels,
        )
    elif split_name in {"train", "validation"}:
        train_directory = _resolve_relative(root, config.get("train_directory", "train"))
        if not train_directory.is_dir():
            raise FileNotFoundError(
                f"NTIRE train directory does not exist: {train_directory}. "
                "Expected ntire/train beside this repository."
            )
        shard_glob = str(config.get("shard_glob", "shard_*"))
        available = sorted(
            (path for path in train_directory.glob(shard_glob) if path.is_dir()),
            key=_shard_sort_key,
        )
        if not available:
            raise FileNotFoundError(
                f"No extracted NTIRE shard directories matching {shard_glob!r} in {train_directory}."
            )

        shard_value = split_spec.get("shards", "all")
        if shard_value == "all_except_validation":
            validation_spec = splits.get("validation")
            if not isinstance(validation_spec, dict):
                raise ValueError("A validation mapping is required for all_except_validation.")
            excluded = set(
                _configured_shard_names(
                    validation_spec.get("shards"), available, field="splits.validation.shards"
                )
            )
            shard_names = [path.name for path in available if path.name not in excluded]
            if not shard_names:
                raise ValueError("No training shards remain after excluding validation shards.")
        else:
            shard_names = _configured_shard_names(
                shard_value, available, field=f"splits.{split_name}.shards"
            )
        chosen = [path for path in available if path.name in set(shard_names)]
        samples = _load_ntire_shards(
            train_directory,
            chosen,
            image_column=str(config.get("train_image_column", "image_name")),
            label_column=str(config.get("train_label_column", "label")),
            class_mapping=class_mapping,
        )
        if split_name == "validation":
            raw_maximum = split_spec.get("max_samples")
            samples = _sample_validation(
                samples,
                maximum=None if raw_maximum is None else int(raw_maximum),
                seed=int(split_spec.get("seed", 3407)),
                sampling=str(split_spec.get("sampling", "balanced")),
            )
    else:
        raise KeyError(f"Unsupported NTIRE split: {split_name!r}.")

    if not samples:
        raise ValueError(f"NTIRE split {split_name!r} contains zero samples.")
    if bool(config.get("verify_files", False)):
        missing = [str(sample.path) for sample in samples if not sample.path.is_file()]
        if missing:
            preview = "\n".join(missing[:10])
            suffix = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise FileNotFoundError(f"Missing {len(missing)} image files:\n{preview}{suffix}")
    return samples


def _rows_for_split(config: dict[str, Any], split_name: str) -> list[dict[str, Any]]:
    config_dir = Path(config["_config_dir"])
    splits = config.get("splits")
    if not isinstance(splits, dict) or split_name not in splits:
        available = sorted(splits) if isinstance(splits, dict) else []
        raise KeyError(f"Split '{split_name}' is absent. Available splits: {available}")

    split_spec = splits[split_name]
    shared_manifest = config.get("manifest")

    if shared_manifest is not None:
        rows = _read_manifest(_resolve_relative(config_dir, shared_manifest))
        columns = config.get("columns", {})
        split_column = columns.get("split", "split")
        split_value = split_spec.get("value", split_name) if isinstance(split_spec, dict) else split_spec
        return [row for row in rows if str(row.get(split_column, "")) == str(split_value)]

    if isinstance(split_spec, list):
        if not all(isinstance(row, dict) for row in split_spec):
            raise ValueError(f"All inline records in split '{split_name}' must be mappings.")
        return [dict(row) for row in split_spec]

    if isinstance(split_spec, str):
        return _read_manifest(_resolve_relative(config_dir, split_spec))

    if isinstance(split_spec, dict):
        if "records" in split_spec:
            records = split_spec["records"]
            if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
                raise ValueError(f"'{split_name}.records' must be a list of mappings.")
            return [dict(row) for row in records]
        if "manifest" in split_spec:
            rows = _read_manifest(_resolve_relative(config_dir, split_spec["manifest"]))
            if "value" in split_spec:
                columns = config.get("columns", {})
                split_column = split_spec.get("split_column", columns.get("split", "split"))
                rows = [row for row in rows if str(row.get(split_column, "")) == str(split_spec["value"])]
            return rows

    raise ValueError(
        f"Unsupported specification for split '{split_name}'. "
        "Use a manifest path, a mapping with 'manifest', or inline 'records'."
    )


def load_samples(
    dataset_config_path: str | Path,
    split_name: str,
    *,
    require_labels: bool,
) -> tuple[list[ImageSample], dict[str, Any]]:
    config = load_yaml(dataset_config_path)
    if config.get("format") == "ntire_2026":
        return (
            _load_ntire_samples(config, split_name, require_labels=require_labels),
            config,
        )

    config_dir = Path(config["_config_dir"])
    image_root = _resolve_relative(config_dir, config.get("root", "."))
    columns = config.get("columns", {})
    path_column = str(columns.get("path", "path"))
    label_column = str(columns.get("label", "label"))
    id_column = columns.get("id")
    class_mapping = {str(key): int(value) for key, value in config.get("classes", {}).items()}

    rows = _rows_for_split(config, split_name)
    samples: list[ImageSample] = []
    for row_number, row in enumerate(rows, start=2):
        raw_path = row.get(path_column)
        if raw_path is None or not str(raw_path).strip():
            raise ValueError(
                f"Missing path column '{path_column}' in split '{split_name}', row {row_number}."
            )
        path = _resolve_relative(image_root, str(raw_path).strip())

        raw_label = row.get(label_column)
        if raw_label is None or str(raw_label).strip() == "":
            if require_labels:
                raise ValueError(
                    f"Missing label column '{label_column}' in split '{split_name}', row {row_number}."
                )
            label = None
        else:
            label = _parse_label(raw_label, class_mapping)

        if id_column is not None and row.get(id_column) not in {None, ""}:
            sample_id = str(row[id_column])
        else:
            sample_id = str(raw_path).replace("\\", "/")
        samples.append(ImageSample(path=path, label=label, sample_id=sample_id))

    if not samples:
        raise ValueError(
            f"Split '{split_name}' contains zero samples. The checked-in manifests are placeholders; "
            "fill configs/dataset.yaml with real paths before training/inference."
        )

    if bool(config.get("verify_files", False)):
        missing = [str(sample.path) for sample in samples if not sample.path.is_file()]
        if missing:
            preview = "\n".join(missing[:10])
            suffix = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
            raise FileNotFoundError(f"Missing {len(missing)} image files:\n{preview}{suffix}")

    return samples, config


def _splitmix64(value: int) -> int:
    """Stable 64-bit mixer used to derive a per-(epoch, sample) RNG seed."""
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


class AIGCImageDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: list[ImageSample],
        *,
        image_size: int | tuple[int, int],
        image_mean: Iterable[float],
        image_std: Iterable[float],
        base_seed: int,
        distortion_pipeline: DistortionPipeline | None = None,
    ) -> None:
        if isinstance(image_size, int):
            self.image_height = image_size
            self.image_width = image_size
        else:
            self.image_height, self.image_width = map(int, image_size)
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_size must contain positive dimensions.")

        self.samples = samples
        self.mean = torch.tensor(list(image_mean), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(list(image_std), dtype=torch.float32).view(3, 1, 1)
        if self.mean.numel() != 3 or self.std.numel() != 3:
            raise ValueError("EVA-CLIP preprocessing must provide three mean and std values.")
        if torch.any(self.std <= 0):
            raise ValueError("All normalization standard deviations must be positive.")
        self.base_seed = int(base_seed)
        self.distortion_pipeline = distortion_pipeline
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _rng_for_index(self, index: int) -> random.Random:
        combined = (
            (self.base_seed & 0xFFFFFFFFFFFFFFFF)
            ^ ((self.epoch + 1) * 0xD1B54A32D192ED03)
            ^ ((index + 1) * 0x94D049BB133111EB)
        ) & 0xFFFFFFFFFFFFFFFF
        sample_seed = _splitmix64(combined)
        return random.Random(sample_seed)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Failed to read image at dataset index {index}: {sample.path}") from error

        # All experimental distortions operate in model-input space.  Resizing
        # first bounds their CPU and memory cost independently of the source
        # image resolution and gives pixel-sized parameters (for example a
        # 15-pixel motion-blur kernel) an unambiguous scale.
        image = image.resize(
            (self.image_width, self.image_height),
            resample=Image.Resampling.BICUBIC,
        )
        rng = self._rng_for_index(index)
        if self.distortion_pipeline is not None:
            image = self.distortion_pipeline(image, rng)

        array = np.asarray(image, dtype=np.float32).copy()
        pixel_values = torch.from_numpy(array).permute(2, 0, 1).div_(255.0)
        pixel_values = (pixel_values - self.mean) / self.std

        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(-1.0 if sample.label is None else float(sample.label), dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.int64),
            "path": str(sample.path),
            "sample_id": sample.sample_id,
            "is_distorted": torch.tensor(
                -1 if sample.is_distorted is None else int(sample.is_distorted),
                dtype=torch.int64,
            ),
        }


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)
