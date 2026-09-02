from __future__ import annotations

import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from filelock import FileLock

from augmentations import (
    BUILTIN_DISTORTION_NAMES,
    DISTORTION_NAMES,
    JPEGAIBackend,
    ReferenceSoftwareJPEGAIBackend,
    apply_distortion,
)


class SerializedJPEGAIBackend:
    """Serialize real JPEG AI subprocesses across DataLoader workers."""

    def __init__(
        self,
        backend: ReferenceSoftwareJPEGAIBackend,
        *,
        lock_path: str | Path,
        lock_timeout_seconds: float,
    ) -> None:
        self.backend = backend
        self.lock_path = str(Path(lock_path).expanduser().resolve())
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.name = backend.name
        self.version = backend.version
        self.profile = backend.profile

    def round_trip(
        self, image: np.ndarray, target_bpp: float
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        with FileLock(self.lock_path, timeout=self.lock_timeout_seconds):
            decoded, facts = self.backend.round_trip(image, target_bpp)
        return decoded, dict(facts)


@dataclass(frozen=True)
class DistortionPolicy:
    """Sampling policy around the downloaded root-level augmentations package."""

    min_operations: int = 1
    max_operations: int = 3
    severity_min: int = 1
    severity_max: int = 5
    sample_without_replacement: bool = True

    def validate(self, operation_count: int) -> None:
        if not 1 <= self.min_operations <= self.max_operations:
            raise ValueError("Expected 1 <= min_operations <= max_operations.")
        if not 1 <= self.severity_min <= self.severity_max <= 5:
            raise ValueError("Severity must be in the inclusive range [1, 5].")
        if self.sample_without_replacement and self.max_operations > operation_count:
            raise ValueError("Cannot sample more distinct operations than are enabled.")


class DistortionPipeline:
    """Apply 1-N downloaded transforms sequentially without persisting metadata."""

    def __init__(
        self,
        policy: DistortionPolicy,
        enabled_operations: list[str] | tuple[str, ...] | None = None,
        jpeg_ai_backend: JPEGAIBackend | None = None,
    ) -> None:
        operations = tuple(enabled_operations or BUILTIN_DISTORTION_NAMES)
        if not operations:
            raise ValueError("At least one distortion operation must be enabled.")
        if len(operations) != len(set(operations)):
            raise ValueError("The distortion operation list must not contain duplicates.")

        unknown = sorted(set(operations) - set(DISTORTION_NAMES))
        if unknown:
            raise ValueError(f"Unknown downloaded distortion operations: {unknown}")
        if "jpeg_ai" in operations and jpeg_ai_backend is None:
            raise ValueError(
                "jpeg_ai requires a separately configured JPEG AI reference-software backend. "
                "Set JPEG_AI_REPOSITORY and JPEG_AI_PYTHON before launch."
            )

        policy.validate(len(operations))
        self.policy = policy
        self.operations = operations
        self.jpeg_ai_backend = jpeg_ai_backend

    @staticmethod
    def _build_jpeg_ai_backend(config: dict[str, Any]) -> JPEGAIBackend:
        backend_config = config.get("jpeg_ai_backend")
        if not isinstance(backend_config, dict):
            raise ValueError(
                "A 'jpeg_ai_backend' mapping is required when jpeg_ai is enabled."
            )

        repository_env = str(backend_config.get("repository_env", "JPEG_AI_REPOSITORY"))
        python_env = str(backend_config.get("python_executable_env", "JPEG_AI_PYTHON"))
        repository_value = os.environ.get(repository_env) or backend_config.get("repository")
        python_value = os.environ.get(python_env) or backend_config.get("python_executable")
        if not repository_value:
            raise ValueError(
                f"jpeg_ai is enabled, but {repository_env} is unset and no repository path "
                "is configured in YAML."
            )
        if not python_value:
            raise ValueError(
                f"jpeg_ai is enabled, but {python_env} is unset and no codec Python path "
                "is configured in YAML."
            )

        config_directory = Path(str(config.get("_config_dir", ".")))
        repository = Path(str(repository_value)).expanduser()
        if not repository.is_absolute():
            repository = config_directory / repository
        python_executable = Path(str(python_value)).expanduser()
        if not python_executable.is_absolute():
            python_executable = config_directory / python_executable
        python_executable = python_executable.resolve()
        if not python_executable.is_file():
            raise FileNotFoundError(
                f"JPEG AI Python executable does not exist: {python_executable}"
            )

        backend = ReferenceSoftwareJPEGAIBackend(
            repository=repository,
            python_executable=python_executable,
            profile=str(backend_config.get("profile", "base")),
            tools_on=bool(backend_config.get("tools_on", False)),
            version=backend_config.get("version"),
            timeout_seconds=float(backend_config.get("timeout_seconds", 600.0)),
        )
        lock_env = str(backend_config.get("lock_path_env", "JPEG_AI_LOCK_PATH"))
        lock_path = os.environ.get(lock_env) or backend_config.get(
            "lock_path", "/tmp/ntire_jpeg_ai_gpu5.lock"
        )
        return SerializedJPEGAIBackend(
            backend,
            lock_path=lock_path,
            lock_timeout_seconds=float(
                backend_config.get("lock_timeout_seconds", 7200.0)
            ),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DistortionPipeline":
        policy = DistortionPolicy(
            min_operations=int(config.get("min_operations", 1)),
            max_operations=int(config.get("max_operations", 3)),
            severity_min=int(config.get("severity_min", 1)),
            severity_max=int(config.get("severity_max", 5)),
            sample_without_replacement=bool(config.get("sample_without_replacement", True)),
        )
        operations = config.get("operations")
        effective_operations = tuple(operations or BUILTIN_DISTORTION_NAMES)
        jpeg_ai_backend = (
            cls._build_jpeg_ai_backend(config) if "jpeg_ai" in effective_operations else None
        )
        return cls(
            policy=policy,
            enabled_operations=effective_operations,
            jpeg_ai_backend=jpeg_ai_backend,
        )

    def sample_plan(self, rng: random.Random) -> list[tuple[str, int, int]]:
        operation_count = rng.randint(self.policy.min_operations, self.policy.max_operations)
        if self.policy.sample_without_replacement:
            names = rng.sample(self.operations, operation_count)
        else:
            names = [rng.choice(self.operations) for _ in range(operation_count)]
        return [
            (
                name,
                rng.randint(self.policy.severity_min, self.policy.severity_max),
                rng.getrandbits(63),
            )
            for name in names
        ]

    def __call__(self, image: Image.Image, rng: random.Random) -> Image.Image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        for name, severity, seed in self.sample_plan(rng):
            array, _metadata = apply_distortion(
                array,
                distortion_type=name,
                severity=severity,
                seed=seed,
                jpeg_ai_backend=self.jpeg_ai_backend,
            )
        return Image.fromarray(array, mode="RGB")
