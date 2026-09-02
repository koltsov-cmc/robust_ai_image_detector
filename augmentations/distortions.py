"""Reproducible image distortions for detector-robustness pipelines.

The module provides 29 distortion types, each with five severity levels. All
built-in transforms accept an RGB ``uint8`` NumPy image and preserve its shape
and dtype. Random choices are derived only from the supplied seed and are
returned in the metadata. Twenty lightweight transforms reuse the parameter
tables from ``aug_utils_val_private`` through dependency-free approximations;
metadata explicitly marks this as parameter-table compatibility, not bytewise
algorithm compatibility. The module does not import the ZIP's PyTorch, Kornia,
Albumentations, CUDA, watermark, or neural-codec dependencies. JPEG AI remains
delegated to an explicit real-codec backend and is never approximated by JPEG.
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


def _level_rows(rows: list[Mapping[str, float | int]]) -> dict[int, dict[str, float | int]]:
    return {severity: dict(row) for severity, row in zip(SEVERITY_LEVELS, rows, strict=True)}


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
    # Lightweight ports from aug_utils_val_private. Names are normalized to
    # snake_case, while the five-level parameter tables remain source-compatible.
    "lens_blur": DistortionSpec("blur", _levels("radius", [1, 2, 4, 6, 8])),
    "color_shift": DistortionSpec("color", _levels("amount", [1, 3, 6, 8, 12])),
    "impulse_noise": DistortionSpec("noise", _levels("density", [0.001, 0.005, 0.01, 0.015, 0.02])),
    "jitter": DistortionSpec("spatial", _levels("amount", [0.05, 0.10, 0.20, 0.50, 1.00])),
    "quantization": DistortionSpec("compression", _levels("levels", [20, 16, 13, 10, 7])),
    "linear_contrast_change": DistortionSpec("brightness", _levels("amount", [0.0, 0.15, -0.4, 0.3, -0.6])),
    "multiplicative_noise": DistortionSpec("noise", _levels("variance", [0.001, 0.005, 0.01, 0.015, 0.035])),
    "pixelate": DistortionSpec("spatial", _levels("strength", [0.01, 0.05, 0.10, 0.20, 0.50])),
    "rgb_shift": DistortionSpec("color", _levels("radius", [10, 20, 30, 40, 50])),
    "random_aspect_crop_resize": DistortionSpec("spatial", _levels("fraction", [0.8, 0.7, 0.6, 0.5, 0.4])),
    "jpeg_recompression_1": DistortionSpec("compression", _levels("n_compressions", [2, 3, 3, 4, 5])),
    "jpeg_recompression_2": DistortionSpec("compression", _level_rows([
        {"source_level": 0, "quality_min": 15, "quality_max": 45},
        {"source_level": 1, "quality_min": 10, "quality_max": 45},
        {"source_level": 2, "quality_min": 10, "quality_max": 40},
        {"source_level": 3, "quality_min": 10, "quality_max": 30},
        {"source_level": 4, "quality_min": 7, "quality_max": 25},
    ])),
    "jpeg_recompression_comb": DistortionSpec("compression", _levels("n_compressions", [2, 3, 3, 4, 5])),
    "jpeg2000": DistortionSpec("compression", _levels("ratio", [16, 32, 45, 120, 170])),
    "glass_blur": DistortionSpec("blur", _levels("max_delta", [1, 2, 3, 4, 6])),
    "random_tone_curve": DistortionSpec("color", _levels("scale", [0.05, 0.15, 0.20, 0.30, 0.40])),
    "clahe": DistortionSpec("color", _level_rows([
        {"source_level": 0, "clip_min": 1.0, "clip_max": 2.5},
        {"source_level": 1, "clip_min": 1.5, "clip_max": 3.5},
        {"source_level": 2, "clip_min": 3.5, "clip_max": 4.5},
        {"source_level": 3, "clip_min": 4.5, "clip_max": 5.5},
        {"source_level": 4, "clip_min": 5.5, "clip_max": 6.5},
    ])),
    # The ZIP defines ISO and shot noise twice; Python uses the later tables.
    "iso_noise": DistortionSpec("noise", _level_rows([
        {"source_level": 0, "intensity_min": 0.20, "intensity_max": 0.30},
        {"source_level": 1, "intensity_min": 0.25, "intensity_max": 0.40},
        {"source_level": 2, "intensity_min": 0.35, "intensity_max": 0.55},
        {"source_level": 3, "intensity_min": 0.40, "intensity_max": 0.65},
        {"source_level": 4, "intensity_min": 0.50, "intensity_max": 0.75},
    ])),
    "shot_noise": DistortionSpec("noise", _level_rows([
        {"source_level": 0, "scale_min": 0.025, "scale_max": 0.075},
        {"source_level": 1, "scale_min": 0.050, "scale_max": 0.150},
        {"source_level": 2, "scale_min": 0.075, "scale_max": 0.200},
        {"source_level": 3, "scale_min": 0.125, "scale_max": 0.250},
        {"source_level": 4, "scale_min": 0.200, "scale_max": 0.300},
    ])),
    "perspective": DistortionSpec("spatial", _level_rows([
        {"source_level": 0, "scale_min": 0.025, "scale_max": 0.075},
        {"source_level": 1, "scale_min": 0.075, "scale_max": 0.120},
        {"source_level": 2, "scale_min": 0.100, "scale_max": 0.250},
        {"source_level": 3, "scale_min": 0.250, "scale_max": 0.300},
        {"source_level": 4, "scale_min": 0.300, "scale_max": 0.400},
    ])),
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
    "lens_blur",
    "color_shift",
    "impulse_noise",
    "jitter",
    "quantization",
    "linear_contrast_change",
    "multiplicative_noise",
    "pixelate",
    "rgb_shift",
    "random_aspect_crop_resize",
    "jpeg_recompression_1",
    "jpeg_recompression_2",
    "jpeg_recompression_comb",
    "jpeg2000",
    "glass_blur",
    "random_tone_curve",
    "clahe",
    "iso_noise",
    "shot_noise",
    "perspective",
)

LIGHTWEIGHT_PORT_NAMES = frozenset(DISTORTION_NAMES[9:])

SOURCE_ARCHIVE_SHA256 = "bc5940e392a6bd840afd081d51b98da698011c93bcf3e0179f286edbccc68040"
SOURCE_TRANSFORM_NAMES = {
    "lens_blur": "lensblur",
    "color_shift": "colorshift",
    "impulse_noise": "impulsenoise",
    "jitter": "jitter",
    "quantization": "quantization",
    "linear_contrast_change": "lincontrchange",
    "multiplicative_noise": "multnoise",
    "pixelate": "pixelate",
    "rgb_shift": "rgbshift",
    "random_aspect_crop_resize": "randomaspectcrop",
    "jpeg_recompression_1": "jpeg_recompression_1",
    "jpeg_recompression_2": "jpeg_recompression_2",
    "jpeg_recompression_comb": "jpeg_recompression_comb",
    "jpeg2000": "jpeg2000",
    "glass_blur": "glassblur",
    "random_tone_curve": "randomtonecurve",
    "clahe": "clahe",
    "iso_noise": "isonoise",
    "shot_noise": "shotnoise",
    "perspective": "perspective",
}


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


def _jpeg_round_trip(image: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    buffer = io.BytesIO()
    _pil(image).save(buffer, format="JPEG", quality=int(quality), optimize=False)
    encoded_size = buffer.tell()
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        output = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
    return output, encoded_size


def _jpeg2000_round_trip(image: np.ndarray, ratio: int) -> tuple[np.ndarray, int]:
    buffer = io.BytesIO()
    try:
        _pil(image).save(
            buffer,
            format="JPEG2000",
            quality_mode="rates",
            quality_layers=[int(ratio)],
        )
    except (KeyError, OSError, ValueError) as error:
        raise RuntimeError(
            "JPEG2000 requires a Pillow build with OpenJPEG support"
        ) from error
    encoded_size = buffer.tell()
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        output = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
    return output, encoded_size


def _disk_blur(image: np.ndarray, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = (xx * xx + yy * yy <= (radius + 0.5) ** 2).astype(np.float64)
    kernel /= kernel.sum()
    padded = np.pad(image.astype(np.float64), ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    height, width = image.shape[:2]
    accumulated = np.zeros_like(image, dtype=np.float64)
    for kernel_y, kernel_x in np.argwhere(kernel > 0):
        accumulated += kernel[kernel_y, kernel_x] * padded[
            kernel_y : kernel_y + height,
            kernel_x : kernel_x + width,
        ]
    return np.uint8(np.clip(np.rint(accumulated), 0, 255))


def _bilinear_resample(image: np.ndarray, source_x: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    source_x = np.clip(source_x, 0.0, width - 1.0)
    source_y = np.clip(source_y, 0.0, height - 1.0)
    x0 = np.floor(source_x).astype(np.int64)
    y0 = np.floor(source_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (source_x - x0)[..., None]
    wy = (source_y - y0)[..., None]
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _piecewise_curve(image: np.ndarray, input_points: list[float], output_points: list[float]) -> np.ndarray:
    normalized = image.astype(np.float64) / 255.0
    transformed = np.empty_like(normalized)
    for channel in range(3):
        transformed[..., channel] = np.interp(normalized[..., channel], input_points, output_points)
    return np.uint8(np.clip(np.rint(transformed * 255.0), 0, 255))


def _clahe_channel(channel: np.ndarray, clip_limit: float, grid_size: tuple[int, int]) -> np.ndarray:
    """Apply dependency-free CLAHE with bilinear interpolation between tile LUTs."""

    height, width = channel.shape
    grid_y = min(grid_size[0], height)
    grid_x = min(grid_size[1], width)
    tile_height = math.ceil(height / grid_y)
    tile_width = math.ceil(width / grid_x)
    luts = np.empty((grid_y, grid_x, 256), dtype=np.float64)

    for tile_y in range(grid_y):
        top = tile_y * tile_height
        bottom = min(height, top + tile_height)
        for tile_x in range(grid_x):
            left = tile_x * tile_width
            right = min(width, left + tile_width)
            tile = channel[top:bottom, left:right]
            histogram = np.bincount(tile.ravel(), minlength=256).astype(np.int64)
            threshold = max(1, int(round(float(clip_limit) * tile.size / 256.0)))
            excess = int(np.maximum(histogram - threshold, 0).sum())
            histogram = np.minimum(histogram, threshold)
            histogram += excess // 256
            histogram[: excess % 256] += 1
            cumulative = histogram.cumsum()
            nonzero = np.flatnonzero(cumulative)
            if not len(nonzero) or cumulative[-1] == cumulative[nonzero[0]]:
                luts[tile_y, tile_x] = np.arange(256)
            else:
                base = cumulative[nonzero[0]]
                luts[tile_y, tile_x] = np.clip(
                    (cumulative - base) * 255.0 / (cumulative[-1] - base),
                    0,
                    255,
                )

    tile_positions_y = (np.arange(height) + 0.5) / tile_height - 0.5
    tile_positions_x = (np.arange(width) + 0.5) / tile_width - 0.5
    low_y_raw = np.floor(tile_positions_y).astype(np.int64)
    low_x_raw = np.floor(tile_positions_x).astype(np.int64)
    weight_y = (tile_positions_y - low_y_raw)[:, None]
    weight_x = (tile_positions_x - low_x_raw)[None, :]
    low_y = np.clip(low_y_raw, 0, grid_y - 1)
    high_y = np.clip(low_y_raw + 1, 0, grid_y - 1)
    low_x = np.clip(low_x_raw, 0, grid_x - 1)
    high_x = np.clip(low_x_raw + 1, 0, grid_x - 1)

    top_left = luts[low_y[:, None], low_x[None, :], channel]
    top_right = luts[low_y[:, None], high_x[None, :], channel]
    bottom_left = luts[high_y[:, None], low_x[None, :], channel]
    bottom_right = luts[high_y[:, None], high_x[None, :], channel]
    top = top_left * (1.0 - weight_x) + top_right * weight_x
    bottom = bottom_left * (1.0 - weight_x) + bottom_right * weight_x
    return np.uint8(np.clip(np.rint(top * (1.0 - weight_y) + bottom * weight_y), 0, 255))


def _perspective_coefficients(
    destination: list[tuple[float, float]],
    source: list[tuple[float, float]],
) -> tuple[float, ...]:
    matrix: list[list[float]] = []
    targets: list[float] = []
    for (x, y), (u, v) in zip(destination, source, strict=True):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        targets.append(u)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        targets.append(v)
    return tuple(float(value) for value in np.linalg.solve(np.asarray(matrix), np.asarray(targets)))


def _base_metadata(name: str, severity: int, seed: int, parameters: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        "transform_type": name,
        "family": DISTORTION_SPECS[name].family,
        "severity": severity,
        "seed": seed,
        "actual_parameters": dict(parameters),
    }
    if name in LIGHTWEIGHT_PORT_NAMES:
        metadata.update({
            "implementation": "numpy-pillow-lightweight-port",
            "source_archive": "aug_utils_val_private",
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_registry_member": "aug_utils_val_private/utils_data.py",
            "source_implementation_member": "aug_utils_val_private/distortions.py",
            "source_transform": SOURCE_TRANSFORM_NAMES[name],
            "compatibility": "parameter-table-only",
        })
    return metadata


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

    elif distortion_type == "lens_blur":
        radius = int(declared["radius"])
        output = _disk_blur(image, radius)
        parameters = {**declared, "kernel_size": 2 * radius + 1, "kernel_shape": "disk"}

    elif distortion_type == "color_shift":
        amount = int(declared["amount"])
        height, width = image.shape[:2]
        gray = np.dot(image.astype(np.float64), np.array([0.299, 0.587, 0.114]))
        gradient_y = np.gradient(gray, axis=0) if height > 1 else np.zeros_like(gray)
        gradient_x = np.gradient(gray, axis=1) if width > 1 else np.zeros_like(gray)
        edge = np.hypot(gradient_x, gradient_y)
        if edge.max() > edge.min():
            edge = (edge - edge.min()) / (edge.max() - edge.min())
        edge_image = Image.fromarray(np.uint8(np.clip(np.rint(edge * 255.0), 0, 255)), mode="L")
        edge = np.asarray(edge_image.filter(ImageFilter.GaussianBlur(4.0)), dtype=np.float64) / 255.0
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        shift_x = int(round(math.cos(angle) * amount))
        shift_y = int(round(math.sin(angle) * amount))
        if shift_x == 0 and shift_y == 0:
            shift_x = amount
        source_x = np.clip(np.arange(width) - shift_x, 0, width - 1)
        source_y = np.clip(np.arange(height) - shift_y, 0, height - 1)
        shifted_green = image[..., 1][np.ix_(source_y, source_x)]
        result = image.astype(np.float64)
        result[..., 1] = shifted_green * edge + result[..., 1] * (1.0 - edge)
        output = np.uint8(np.clip(np.rint(result), 0, 255))
        parameters = {
            **declared,
            "channel": "green",
            "shift_x": shift_x,
            "shift_y": shift_y,
            "edge_blur_sigma": 4.0,
            "shift_seed": seed,
        }

    elif distortion_type == "impulse_noise":
        density = float(declared["density"])
        salt_fraction = 0.5
        number = max(1, int(density * image.size))
        coordinates = rng.integers(0, image.size, size=number)
        number_salt = int(salt_fraction * number)
        flat = image.copy().reshape(-1)
        flat[coordinates[:number_salt]] = 255
        flat[coordinates[number_salt:]] = 0
        output = flat.reshape(image.shape)
        parameters = {
            **declared,
            "salt_fraction": salt_fraction,
            "requested_elements": number,
            "noise_seed": seed,
        }

    elif distortion_type == "jitter":
        amount = float(declared["amount"])
        iterations = 5
        height, width = image.shape[:2]
        grid_y, grid_x = np.mgrid[:height, :width]
        jittered = image.astype(np.float64)
        for _ in range(iterations):
            shift_y = rng.normal(0.0, amount, size=(height, width))
            shift_x = rng.normal(0.0, amount, size=(height, width))
            jittered = _bilinear_resample(jittered, grid_x - shift_x, grid_y - shift_y)
        output = np.uint8(np.clip(np.rint(jittered), 0, 255))
        parameters = {**declared, "iterations": iterations, "noise_seed": seed, "interpolation": "bilinear"}

    elif distortion_type == "quantization":
        levels = int(declared["levels"])
        step = 255.0 / (levels - 1)
        output = np.uint8(np.clip(np.rint(np.rint(image.astype(np.float64) / step) * step), 0, 255))
        parameters = {**declared, "step": step, "method": "uniform-per-channel"}

    elif distortion_type == "linear_contrast_change":
        amount = float(declared["amount"])
        input_points = [0.0, 0.3, 0.5, 0.7, 1.0]
        output_points = [0.0, 0.25 - amount / 4.0, 0.5, 0.75 + amount / 4.0, 1.0]
        output = _piecewise_curve(image, input_points, output_points)
        parameters = {**declared, "input_points": input_points, "output_points": output_points}

    elif distortion_type == "multiplicative_noise":
        variance = float(declared["variance"])
        normalized = image.astype(np.float64) / 255.0
        noise = rng.normal(0.0, math.sqrt(variance), size=image.shape)
        output = np.uint8(np.clip(np.rint((normalized + normalized * noise) * 255.0), 0, 255))
        parameters = {**declared, "standard_deviation": math.sqrt(variance), "noise_seed": seed}

    elif distortion_type == "pixelate":
        strength = float(declared["strength"])
        scale = max(0.01, 0.95 - strength ** 0.6)
        width, height = pil.size
        down_width = max(1, int(round(width * scale)))
        down_height = max(1, int(round(height * scale)))
        reduced = pil.resize((down_width, down_height), Image.Resampling.NEAREST)
        output = np.asarray(reduced.resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8).copy()
        parameters = {
            **declared,
            "scale": scale,
            "downsample_width": down_width,
            "downsample_height": down_height,
            "resampling": "nearest",
        }

    elif distortion_type == "rgb_shift":
        radius = int(declared["radius"])
        shifts = rng.integers(-radius, radius + 1, size=3)
        output = np.uint8(np.clip(image.astype(np.int16) + shifts.reshape(1, 1, 3), 0, 255))
        parameters = {**declared, "channel_shifts": [int(value) for value in shifts], "shift_seed": seed}

    elif distortion_type == "random_aspect_crop_resize":
        fraction = float(declared["fraction"])
        width, height = pil.size
        aspect_ratio = float(rng.uniform(0.5, 2.0))
        crop_height = max(1, min(height, int(math.ceil((fraction * height) / 16.0) * 16)))
        crop_width = max(1, min(width, int(math.ceil((fraction * height * aspect_ratio) / 16.0) * 16)))
        left = int(rng.integers(0, width - crop_width + 1))
        top = int(rng.integers(0, height - crop_height + 1))
        box = (left, top, left + crop_width, top + crop_height)
        cropped = pil.crop(box)
        output = np.asarray(cropped.resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8).copy()
        parameters = {
            **declared,
            "aspect_ratio": aspect_ratio,
            "crop_box": list(box),
            "crop_width": crop_width,
            "crop_height": crop_height,
            "crop_seed": seed,
            "resized_to_original_shape": True,
            "resampling": "bilinear",
        }

    elif distortion_type == "jpeg_recompression_1":
        n_compressions = int(declared["n_compressions"])
        qualities: list[int] = []
        encoded_sizes: list[int] = []
        output = image.copy()
        for _ in range(n_compressions):
            quality = int(rng.integers(7, 44))
            output, encoded_size = _jpeg_round_trip(output, quality)
            qualities.append(quality)
            encoded_sizes.append(encoded_size)
        parameters = {
            **declared,
            "quality_min": 7,
            "quality_max": 43,
            "qualities": qualities,
            "encoded_bytes": encoded_sizes,
            "compression_seed": seed,
        }

    elif distortion_type == "jpeg_recompression_2":
        quality_min = int(declared["quality_min"])
        quality_max = int(declared["quality_max"])
        n_compressions = 3
        qualities = []
        encoded_sizes = []
        output = image.copy()
        for _ in range(n_compressions):
            quality = int(rng.integers(quality_min, quality_max + 1))
            output, encoded_size = _jpeg_round_trip(output, quality)
            qualities.append(quality)
            encoded_sizes.append(encoded_size)
        parameters = {
            **declared,
            "n_compressions": n_compressions,
            "qualities": qualities,
            "encoded_bytes": encoded_sizes,
            "compression_seed": seed,
        }

    elif distortion_type == "jpeg_recompression_comb":
        n_compressions = int(declared["n_compressions"])
        sequence: list[dict[str, Any]] = []
        output = image.copy()
        for _ in range(n_compressions):
            if float(rng.random()) < 0.5:
                quality = int(rng.integers(7, 44))
                output, encoded_size = _jpeg_round_trip(output, quality)
                sequence.append({"codec": "jpeg", "quality": quality, "encoded_bytes": encoded_size})
            else:
                ratio = int(rng.integers(16, 171))
                output, encoded_size = _jpeg2000_round_trip(output, ratio)
                sequence.append({"codec": "jpeg2000", "ratio": ratio, "encoded_bytes": encoded_size})
        parameters = {
            **declared,
            "jpeg_quality_range": [7, 43],
            "jpeg2000_ratio_range": [16, 170],
            "sequence": sequence,
            "compression_seed": seed,
        }

    elif distortion_type == "jpeg2000":
        ratio = int(declared["ratio"])
        output, encoded_size = _jpeg2000_round_trip(image, ratio)
        parameters = {**declared, "encoded_bytes": encoded_size, "quality_mode": "rates"}

    elif distortion_type == "glass_blur":
        max_delta = int(declared["max_delta"])
        sigma = 0.5
        iterations = 4
        glass = np.asarray(pil.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float64)
        height, width = image.shape[:2]
        grid_y, grid_x = np.mgrid[:height, :width]
        for _ in range(iterations):
            displacement_y = rng.integers(-max_delta, max_delta + 1, size=(height, width))
            displacement_x = rng.integers(-max_delta, max_delta + 1, size=(height, width))
            source_y = np.clip(grid_y + displacement_y, 0, height - 1)
            source_x = np.clip(grid_x + displacement_x, 0, width - 1)
            glass = glass[source_y, source_x]
        blurred = Image.fromarray(np.uint8(np.clip(np.rint(glass), 0, 255)), mode="RGB").filter(
            ImageFilter.GaussianBlur(sigma)
        )
        output = np.asarray(blurred, dtype=np.uint8).copy()
        parameters = {
            **declared,
            "sigma": sigma,
            "iterations": iterations,
            "source_mode": "fast",
            "implementation_mode": "dense-random-remap",
            "noise_seed": seed,
        }

    elif distortion_type == "random_tone_curve":
        scale = float(declared["scale"])
        input_points = [0.0, 0.25, 0.75, 1.0]
        channel_points: list[list[float]] = []
        normalized = image.astype(np.float64) / 255.0
        transformed = np.empty_like(normalized)
        for channel in range(3):
            low = float(np.clip(rng.normal(0.25, scale), 0.0, 0.5))
            high = float(np.clip(rng.normal(0.75, scale), 0.5, 1.0))
            output_points = [0.0, low, high, 1.0]
            transformed[..., channel] = np.interp(normalized[..., channel], input_points, output_points)
            channel_points.append(output_points)
        output = np.uint8(np.clip(np.rint(transformed * 255.0), 0, 255))
        parameters = {
            **declared,
            "input_points": input_points,
            "channel_output_points": channel_points,
            "curve_seed": seed,
            "interpolation": "piecewise-linear",
        }

    elif distortion_type == "clahe":
        clip_limit = float(rng.uniform(float(declared["clip_min"]), float(declared["clip_max"])))
        ycbcr = np.asarray(pil.convert("YCbCr"), dtype=np.uint8).copy()
        ycbcr[..., 0] = _clahe_channel(ycbcr[..., 0], clip_limit, (12, 12))
        output = np.asarray(Image.fromarray(ycbcr, mode="YCbCr").convert("RGB"), dtype=np.uint8).copy()
        parameters = {
            **declared,
            "clip_limit": clip_limit,
            "tile_grid_size": [12, 12],
            "channel": "YCbCr-Y",
            "parameter_seed": seed,
        }

    elif distortion_type == "iso_noise":
        intensity = float(rng.uniform(float(declared["intensity_min"]), float(declared["intensity_max"])))
        color_shift = float(rng.uniform(0.01, 0.20))
        normalized = image.astype(np.float64) / 255.0
        luminance = np.dot(normalized, np.array([0.299, 0.587, 0.114]))[..., None]
        luminance_noise = rng.normal(0.0, 1.0, size=(image.shape[0], image.shape[1], 1))
        luminance_noise *= intensity * 0.12 * np.sqrt(np.maximum(luminance, 1.0 / 255.0))
        chroma_noise = rng.normal(0.0, intensity * color_shift * 0.08, size=image.shape)
        noisy = normalized + luminance_noise + chroma_noise
        output = np.uint8(np.clip(np.rint(noisy * 255.0), 0, 255))
        parameters = {
            **declared,
            "intensity": intensity,
            "color_shift": color_shift,
            "noise_seed": seed,
            "model": "signal-dependent-gaussian",
        }

    elif distortion_type == "shot_noise":
        scale = float(rng.uniform(float(declared["scale_min"]), float(declared["scale_max"])))
        peak = max(1.0, 1.0 / (scale * scale))
        normalized = image.astype(np.float64) / 255.0
        output = np.uint8(np.clip(np.rint(rng.poisson(normalized * peak) / peak * 255.0), 0, 255))
        parameters = {**declared, "scale": scale, "poisson_peak": peak, "noise_seed": seed}

    elif distortion_type == "perspective":
        scale = float(rng.uniform(float(declared["scale_min"]), float(declared["scale_max"])))
        width, height = pil.size
        if width < 2 or height < 2:
            source_quad = [(0.0, 0.0)] * 4
            output = image.copy()
        else:
            max_offset = scale * min(width - 1, height - 1)
            source_quad = [
                (float(rng.uniform(0.0, max_offset)), float(rng.uniform(0.0, max_offset))),
                (float(width - 1 - rng.uniform(0.0, max_offset)), float(rng.uniform(0.0, max_offset))),
                (float(width - 1 - rng.uniform(0.0, max_offset)), float(height - 1 - rng.uniform(0.0, max_offset))),
                (float(rng.uniform(0.0, max_offset)), float(height - 1 - rng.uniform(0.0, max_offset))),
            ]
            destination = [
                (0.0, 0.0),
                (float(width - 1), 0.0),
                (float(width - 1), float(height - 1)),
                (0.0, float(height - 1)),
            ]
            coefficients = _perspective_coefficients(destination, source_quad)
            transformed = pil.transform(
                (width, height),
                Image.Transform.PERSPECTIVE,
                coefficients,
                resample=Image.Resampling.BICUBIC,
            )
            output = np.asarray(transformed, dtype=np.uint8).copy()
        parameters = {
            **declared,
            "scale": scale,
            "source_quad": [[x, y] for x, y in source_quad],
            "warp_seed": seed,
            "resampling": "bicubic",
            "kept_original_shape": True,
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

    _validate_image(output)
    if output.shape != image.shape:
        raise RuntimeError(f"{distortion_type} changed image shape from {image.shape} to {output.shape}")
    return output, _base_metadata(distortion_type, severity, seed, parameters)
