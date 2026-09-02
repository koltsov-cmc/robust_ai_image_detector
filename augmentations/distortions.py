"""Reproducible image distortions for detector-robustness pipelines.

The module provides nine distortion types, each with five severity levels.
All built-in transforms accept an RGB ``uint8`` NumPy image and preserve its
shape and dtype. Random choices are derived only from the supplied seed and
are returned in the metadata. JPEG AI is deliberately delegated to an
explicit codec backend so it can never be silently approximated by JPEG 1.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


SEVERITY_LEVELS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class DistortionSpec:
    family: str
    levels: Mapping[int, Mapping[str, float | int]]


def _levels(parameter: str, values: list[float | int]) -> dict[int, dict[str, float | int]]:
    return {severity: {parameter: value} for severity, value in zip(SEVERITY_LEVELS, values, strict=True)}


DISTORTION_SPECS: dict[str, DistortionSpec] = {
    "jpeg": DistortionSpec("compression", _levels("quality", [95, 80, 60, 40, 20])),
    "gaussian_blur": DistortionSpec("blur", _levels("sigma", [0.5, 1.0, 1.5, 2.0, 3.0])),
    "motion_blur": DistortionSpec("blur", _levels("kernel_size", [3, 5, 7, 11, 15])),
    "gaussian_noise": DistortionSpec("noise", _levels("sigma", [2, 5, 10, 15, 25])),
    "brightness_shift": DistortionSpec("brightness", _levels("magnitude", [0.05, 0.10, 0.20, 0.30, 0.40])),
    "saturation_shift": DistortionSpec("color", _levels("magnitude", [0.10, 0.20, 0.35, 0.50, 0.70])),
    "downsample_upscale": DistortionSpec("spatial", _levels("scale", [0.90, 0.75, 0.60, 0.40, 0.25])),
    "random_crop_resize": DistortionSpec("spatial", _levels("area_ratio", [0.95, 0.90, 0.80, 0.70, 0.60])),
    # The five target rates in the official reference software's
    # cfg/pipeline.json, ordered here from mild to severe distortion.
    "jpeg_ai": DistortionSpec("compression", _levels("target_bpp", [1.00, 0.75, 0.50, 0.25, 0.12])),
}

DISTORTION_NAMES = tuple(DISTORTION_SPECS)
BUILTIN_DISTORTION_NAMES = (
    "jpeg",
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "brightness_shift",
    "saturation_shift",
    "downsample_upscale",
    "random_crop_resize",
)


class JPEGAIUnavailableError(RuntimeError):
    """Raised when JPEG AI is requested without a real codec backend."""


class JPEGAICodecError(RuntimeError):
    """Raised when the configured JPEG AI reference codec cannot run."""


class JPEGAIBackend(Protocol):
    """Boundary implemented by a real JPEG AI encode/decode integration."""

    name: str
    version: str
    profile: str

    def round_trip(
        self, image: np.ndarray, target_bpp: float
    ) -> tuple[np.ndarray, Mapping[str, float | int | str]]:
        """Encode/decode an image and report positive ``actual_bpp`` and ``bitstream_bytes`` values."""


class ReferenceSoftwareJPEGAIBackend:
    """Adapter for the official JPEG AI reference-software encoder and decoder.

    The upstream CLI expects ``--set_target_bpp`` in units of bpp multiplied
    by 100.  It also expects PNG input and emits a decoded PNG image.
    """

    name = "jpeg-ai-reference-software"

    def __init__(
        self,
        repository: str | Path,
        *,
        python_executable: str | Path | None = None,
        profile: str = "base",
        tools_on: bool = False,
        version: str | None = None,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.repository = Path(repository).expanduser().resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.profile = profile
        self.tools_on = tools_on
        self.timeout_seconds = float(timeout_seconds)

        if profile not in {"simple", "base", "high"}:
            raise ValueError("JPEG AI profile must be one of: simple, base, high")
        if not self.repository.is_dir():
            raise JPEGAICodecError(f"JPEG AI reference-software repository does not exist: {self.repository}")
        tools_cfg = "tools_on.json" if self.tools_on else "tools_off.json"
        required = (
            self.repository / "src" / "reco" / "coders" / "encoder.py",
            self.repository / "src" / "reco" / "coders" / "decoder.py",
            self.repository / "cfg" / "pipeline.json",
            self.repository / "cfg" / tools_cfg,
            self.repository / "cfg" / "profiles" / f"{self.profile}.json",
            self.repository / "cfg" / "BRM" / "regen_list.json",
        )
        missing = [str(path.relative_to(self.repository)) for path in required if not path.is_file()]
        if missing:
            raise JPEGAICodecError(f"JPEG AI reference-software checkout is missing required files: {', '.join(missing)}")
        self.version = version or self._detect_version()

    def _detect_version(self) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        return completed.stdout.strip() or "unknown"

    def _run(self, command: list[str], stage: str) -> float:
        started = time.perf_counter()
        try:
            subprocess.run(
                command,
                cwd=self.repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise JPEGAICodecError(f"JPEG AI {stage} timed out after {self.timeout_seconds:g} seconds") from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "no codec output").strip()[-2000:]
            raise JPEGAICodecError(f"JPEG AI {stage} failed: {detail}") from error
        except OSError as error:
            raise JPEGAICodecError(f"JPEG AI {stage} could not start: {error}") from error
        return time.perf_counter() - started

    def round_trip(
        self, image: np.ndarray, target_bpp: float
    ) -> tuple[np.ndarray, Mapping[str, float | int | str]]:
        _validate_image(image)
        if not math.isfinite(target_bpp) or target_bpp <= 0:
            raise ValueError("target_bpp must be a positive finite number")
        target_bppm100 = int(round(target_bpp * 100.0))
        if target_bppm100 <= 0:
            raise ValueError("target_bpp is below the reference CLI precision of 0.01 bpp")

        tools_cfg = "cfg/tools_on.json" if self.tools_on else "cfg/tools_off.json"
        profile_cfg = f"cfg/profiles/{self.profile}.json"
        with tempfile.TemporaryDirectory(prefix="jpeg-ai-roundtrip-") as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "input.png"
            bitstream_path = temporary_path / "image.bits"
            output_path = temporary_path / "decoded.png"
            _pil(image).save(input_path, format="PNG")

            encoder_seconds = self._run(
                [
                    self.python_executable,
                    "-m",
                    "src.reco.coders.encoder",
                    str(input_path),
                    str(bitstream_path),
                    "--set_target_bpp",
                    str(target_bppm100),
                    "--cfg",
                    tools_cfg,
                    profile_cfg,
                ],
                "encoder",
            )
            if not bitstream_path.is_file():
                raise JPEGAICodecError("JPEG AI encoder completed without creating a bitstream")

            decoder_seconds = self._run(
                [
                    self.python_executable,
                    "-m",
                    "src.reco.coders.decoder",
                    str(bitstream_path),
                    str(output_path),
                ],
                "decoder",
            )
            if not output_path.is_file():
                raise JPEGAICodecError("JPEG AI decoder completed without creating a decoded PNG")

            bitstream_bytes = bitstream_path.stat().st_size
            with Image.open(output_path) as decoded_image:
                decoded = np.asarray(decoded_image.convert("RGB"), dtype=np.uint8).copy()

        height, width = image.shape[:2]
        return decoded, {
            "target_bppm100": target_bppm100,
            "actual_bpp": bitstream_bytes * 8.0 / (height * width),
            "bitstream_bytes": bitstream_bytes,
            "bitstream_extension": ".bits",
            "source_width": width,
            "source_height": height,
            "rate_selection": "set_target_bpp",
            "repository_commit": self.version,
            "tools_config": "on" if self.tools_on else "off",
            "encoder_seconds": encoder_seconds,
            "decoder_seconds": decoder_seconds,
        }


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an HWC RGB uint8 NumPy array")


def _pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image, mode="RGB")


def _motion_blur(image: np.ndarray, kernel_size: int, direction: str) -> np.ndarray:
    vectors = {
        "horizontal": (0, 1),
        "vertical": (1, 0),
        "diagonal_down": (1, 1),
        "diagonal_up": (1, -1),
    }
    dy, dx = vectors[direction]
    half = kernel_size // 2
    height, width = image.shape[:2]
    # Float32 is sufficient for the largest possible sum (15 * 255) and halves
    # both memory traffic and temporary-buffer size compared with float64.
    source = image.astype(np.float32)
    padded = np.pad(source, ((half, half), (half, half), (0, 0)), mode="edge")
    accumulated = np.zeros_like(source)
    for offset in range(-half, half + 1):
        top = half + dy * offset
        left = half + dx * offset
        accumulated += padded[top : top + height, left : left + width]
    accumulated *= np.float32(1.0 / kernel_size)
    return np.uint8(np.clip(np.rint(accumulated), 0, 255))


def _base_metadata(name: str, severity: int, seed: int, parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "transform_type": name,
        "family": DISTORTION_SPECS[name].family,
        "severity": severity,
        "seed": seed,
        "actual_parameters": dict(parameters),
    }


def apply_distortion(
    image: np.ndarray,
    distortion_type: str,
    severity: int,
    seed: int,
    *,
    jpeg_ai_backend: JPEGAIBackend | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one declared distortion and return the image plus full metadata."""

    _validate_image(image)
    if distortion_type not in DISTORTION_SPECS:
        raise ValueError(f"unknown distortion: {distortion_type}")
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"severity must be one of {SEVERITY_LEVELS}, got {severity}")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    declared = dict(DISTORTION_SPECS[distortion_type].levels[severity])
    rng = np.random.default_rng(seed)
    pil = _pil(image)

    if distortion_type == "jpeg":
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=int(declared["quality"]), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            output = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
        parameters = declared

    elif distortion_type == "gaussian_blur":
        output = np.asarray(pil.filter(ImageFilter.GaussianBlur(float(declared["sigma"]))), dtype=np.uint8).copy()
        parameters = declared

    elif distortion_type == "motion_blur":
        directions = ("horizontal", "vertical", "diagonal_down", "diagonal_up")
        direction = directions[seed % len(directions)]
        output = _motion_blur(image, int(declared["kernel_size"]), direction)
        parameters = {**declared, "direction": direction}

    elif distortion_type == "gaussian_noise":
        sigma = np.float32(declared["sigma"])
        # Generator.standard_normal supports native float32 output, avoiding
        # two full-resolution float64 arrays for every noisy image.
        noise = rng.standard_normal(size=image.shape, dtype=np.float32)
        noise *= sigma
        noisy = image.astype(np.float32)
        noisy += noise
        output = np.uint8(np.clip(np.rint(noisy), 0, 255))
        parameters = {**declared, "noise_seed": seed}

    elif distortion_type in {"brightness_shift", "saturation_shift"}:
        magnitude = float(declared["magnitude"])
        direction = "increase" if seed % 2 == 0 else "decrease"
        factor = 1.0 + magnitude if direction == "increase" else 1.0 - magnitude
        enhancer = ImageEnhance.Brightness(pil) if distortion_type == "brightness_shift" else ImageEnhance.Color(pil)
        output = np.asarray(enhancer.enhance(factor), dtype=np.uint8).copy()
        parameters = {**declared, "direction": direction, "factor": factor}

    elif distortion_type == "downsample_upscale":
        scale = float(declared["scale"])
        width, height = pil.size
        down_width = max(1, int(round(width * scale)))
        down_height = max(1, int(round(height * scale)))
        reduced = pil.resize((down_width, down_height), Image.Resampling.BILINEAR)
        output = np.asarray(reduced.resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8).copy()
        parameters = {
            **declared,
            "downsample_width": down_width,
            "downsample_height": down_height,
            "resampling": "bilinear",
        }

    elif distortion_type == "random_crop_resize":
        area_ratio = float(declared["area_ratio"])
        width, height = pil.size
        side_ratio = math.sqrt(area_ratio)
        crop_width = max(1, min(width, int(round(width * side_ratio))))
        crop_height = max(1, min(height, int(round(height * side_ratio))))
        left = int(rng.integers(0, width - crop_width + 1))
        top = int(rng.integers(0, height - crop_height + 1))
        box = (left, top, left + crop_width, top + crop_height)
        cropped = pil.crop(box)
        output = np.asarray(cropped.resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8).copy()
        parameters = {
            **declared,
            "crop_box": list(box),
            "crop_width": crop_width,
            "crop_height": crop_height,
            "crop_seed": seed,
            "resampling": "bilinear",
        }

    elif distortion_type == "jpeg_ai":
        if jpeg_ai_backend is None:
            raise JPEGAIUnavailableError(
                "JPEG AI backend is required; install/configure a real JPEG AI codec and pass jpeg_ai_backend"
            )
        target_bpp = float(declared["target_bpp"])
        decoded, codec_facts = jpeg_ai_backend.round_trip(image.copy(), target_bpp)
        _validate_image(decoded)
        if decoded.shape != image.shape:
            raise ValueError(f"JPEG AI backend changed image shape from {image.shape} to {decoded.shape}")
        for required_fact in ("actual_bpp", "bitstream_bytes"):
            if required_fact not in codec_facts:
                raise JPEGAICodecError(f"JPEG AI backend did not report required fact: {required_fact}")
        raw_actual_bpp = codec_facts["actual_bpp"]
        raw_bitstream_bytes = codec_facts["bitstream_bytes"]
        try:
            actual_bpp = float(raw_actual_bpp)
        except (TypeError, ValueError, OverflowError) as error:
            raise JPEGAICodecError("JPEG AI backend reported non-numeric codec facts") from error
        if isinstance(raw_actual_bpp, (bool, np.bool_)) or not math.isfinite(actual_bpp) or actual_bpp <= 0:
            raise JPEGAICodecError("JPEG AI backend reported an invalid actual_bpp")
        if isinstance(raw_bitstream_bytes, (bool, np.bool_)) or not isinstance(
            raw_bitstream_bytes, (int, np.integer)
        ):
            raise JPEGAICodecError("JPEG AI backend reported a non-integer bitstream_bytes value")
        bitstream_bytes = int(raw_bitstream_bytes)
        if bitstream_bytes <= 0:
            raise JPEGAICodecError("JPEG AI backend reported an invalid bitstream_bytes value")
        output = decoded.copy()
        parameters = {
            **dict(codec_facts),
            "actual_bpp": actual_bpp,
            "bitstream_bytes": bitstream_bytes,
            **declared,
            "codec_name": jpeg_ai_backend.name,
            "codec_version": jpeg_ai_backend.version,
            "codec_profile": jpeg_ai_backend.profile,
        }

    else:  # pragma: no cover - registry and dispatch must evolve together.
        raise RuntimeError(f"distortion is registered but not implemented: {distortion_type}")

    return output, _base_metadata(distortion_type, severity, seed, parameters)
