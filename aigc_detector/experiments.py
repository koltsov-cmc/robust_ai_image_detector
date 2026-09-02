from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# These are the only supported experiment YAML files. The CLI deliberately does
# not accept an arbitrary --config path: an experiment name always resolves to
# exactly one repository-relative training config.
TRAINING_CONFIG_PATHS: dict[str, Path] = {
    "evaclipb_gap_clean": Path("configs/train_gap_clean.yaml"),
    "evaclipb_gap_distorted_only": Path("configs/train_gap_distorted.yaml"),
}
INFERENCE_CONFIG_PATH = Path("configs/inference_gap.yaml")
DATASET_CONFIG_PATH = Path("configs/dataset.yaml")


def _mapping(config: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' in {source} must be a YAML mapping.")
    return value


def _positive_int(value: Any, key: str, source: Path, *, allow_zero: bool = False) -> int:
    parsed = int(value)
    lower_bound = 0 if allow_zero else 1
    if parsed < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"'{key}' in {source} must be a {qualifier} integer.")
    return parsed


@dataclass(frozen=True)
class Experiment:
    name: str
    augmentation_mode: str
    training_config_path: Path
    inference_config_path: Path
    dataset_config_path: Path

    seed: int
    model_id: str
    image_size: int
    batch_size: int
    num_workers: int
    inference_batch_size: int
    inference_num_workers: int

    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    min_lr_ratio: float
    max_grad_norm: float
    log_interval_steps: int

    _model_config: dict[str, Any]
    _distortion_config: dict[str, Any] | None

    @property
    def output_dir(self) -> Path:
        return PROJECT_ROOT / "runs" / self.name

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_dir / "best.pt"

    @property
    def prediction_path(self) -> Path:
        return PROJECT_ROOT / "predictions" / f"{self.name}.csv"

    @property
    def model_config(self) -> dict[str, Any]:
        return deepcopy(self._model_config)

    @property
    def distortion_config(self) -> dict[str, Any] | None:
        return deepcopy(self._distortion_config)


def _load_experiment(name: str, relative_training_path: Path) -> Experiment:
    training_path = (PROJECT_ROOT / relative_training_path).resolve()
    inference_path = (PROJECT_ROOT / INFERENCE_CONFIG_PATH).resolve()
    dataset_path = (PROJECT_ROOT / DATASET_CONFIG_PATH).resolve()

    training_config = load_yaml(training_path)
    inference_config = load_yaml(inference_path)

    configured_name = str(training_config.get("experiment", ""))
    if configured_name != name:
        raise ValueError(
            f"Experiment name in {training_path} is {configured_name!r}; expected {name!r}."
        )

    augmentation_mode = str(training_config.get("augmentation_mode", ""))
    if augmentation_mode not in {"clean", "distorted_only"}:
        raise ValueError(
            f"'augmentation_mode' in {training_path} must be 'clean' or 'distorted_only'."
        )

    model = _mapping(training_config, "model", training_path)
    preprocessing = _mapping(training_config, "preprocessing", training_path)
    train_loader = _mapping(training_config, "dataloader", training_path)
    training = _mapping(training_config, "training", training_path)
    optimizer = _mapping(training, "optimizer", training_path)
    scheduler = _mapping(training, "scheduler", training_path)
    early_stopping = _mapping(training, "early_stopping", training_path)
    inference_loader = _mapping(inference_config, "dataloader", inference_path)

    distortion_config = training_config.get("distortions")
    if augmentation_mode == "clean":
        if distortion_config is not None:
            raise ValueError(
                f"'distortions' in clean experiment {training_path} must be null."
            )
    elif not isinstance(distortion_config, dict):
        raise ValueError(
            f"'distortions' in distorted-only experiment {training_path} must be a mapping."
        )
    if isinstance(distortion_config, dict):
        distortion_config = deepcopy(distortion_config)
        distortion_config["_config_dir"] = str(training_path.parent)

    model_id = str(model.get("id", "")).strip()
    if not model_id:
        raise ValueError(f"'model.id' in {training_path} must not be empty.")

    model_config = deepcopy(model)
    model_config["id"] = model_id

    return Experiment(
        name=name,
        augmentation_mode=augmentation_mode,
        training_config_path=training_path,
        inference_config_path=inference_path,
        dataset_config_path=dataset_path,
        seed=int(training_config.get("seed", 3407)),
        model_id=model_id,
        image_size=_positive_int(
            preprocessing.get("image_size"), "preprocessing.image_size", training_path
        ),
        batch_size=_positive_int(
            train_loader.get("batch_size"), "dataloader.batch_size", training_path
        ),
        num_workers=_positive_int(
            train_loader.get("num_workers"),
            "dataloader.num_workers",
            training_path,
            allow_zero=True,
        ),
        inference_batch_size=_positive_int(
            inference_loader.get("batch_size"),
            "dataloader.batch_size",
            inference_path,
        ),
        inference_num_workers=_positive_int(
            inference_loader.get("num_workers"),
            "dataloader.num_workers",
            inference_path,
            allow_zero=True,
        ),
        max_epochs=_positive_int(
            training.get("max_epochs"), "training.max_epochs", training_path
        ),
        early_stopping_patience=_positive_int(
            early_stopping.get("patience"),
            "training.early_stopping.patience",
            training_path,
        ),
        learning_rate=float(optimizer.get("learning_rate")),
        weight_decay=float(optimizer.get("weight_decay")),
        warmup_ratio=float(scheduler.get("warmup_ratio")),
        min_lr_ratio=float(scheduler.get("min_lr_ratio")),
        max_grad_norm=float(training.get("max_grad_norm")),
        log_interval_steps=_positive_int(
            training.get("log_interval_steps"),
            "training.log_interval_steps",
            training_path,
            allow_zero=True,
        ),
        _model_config=model_config,
        _distortion_config=deepcopy(distortion_config),
    )


def experiment_names() -> tuple[str, ...]:
    return tuple(TRAINING_CONFIG_PATHS)


def get_experiment(name: str) -> Experiment:
    try:
        relative_path = TRAINING_CONFIG_PATHS[name]
    except KeyError as error:
        available = ", ".join(experiment_names())
        raise ValueError(f"Unknown experiment '{name}'. Available experiments: {available}") from error
    return _load_experiment(name, relative_path)
