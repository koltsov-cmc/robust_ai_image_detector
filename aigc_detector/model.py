from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


EVA_CLIP_HUB_ID = "timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k"
EVA_CLIP_HIDDEN_SIZE = 768
EVA_CLIP_PATCH_SIZE = 16
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _require_open_clip():
    try:
        import open_clip
    except ImportError as error:
        raise ImportError(
            "open_clip_torch and timm are required for EVA02-CLIP-B/16. "
            "Install requirements.txt."
        ) from error
    return open_clip


def _as_pair(value: Any, *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        pair = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"Expected {name} to be an integer or a pair, got {value!r}.")
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"Expected positive {name}, got {pair!r}.")
    return pair


def _rgb_stats(preprocess_config: dict[str, Any]) -> tuple[list[float], list[float]]:
    mean = [float(value) for value in preprocess_config.get("mean", [])]
    std = [float(value) for value in preprocess_config.get("std", [])]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(
            "The EVA-CLIP checkpoint must provide three-channel RGB normalization "
            f"statistics, got mean={mean}, std={std}."
        )
    if any(value <= 0 for value in std):
        raise ValueError(f"EVA-CLIP normalization std must be positive, got {std}.")
    return mean, std


def load_evaclip_backbone(
    model_config: dict[str, Any],
    *,
    image_size: int,
) -> tuple[nn.Module, list[float], list[float]]:
    """Load the public EVA02-CLIP-B/16 checkpoint and retain its visual trunk only."""
    open_clip = _require_open_clip()

    model_id = str(model_config["id"]).strip()
    if model_id != EVA_CLIP_HUB_ID:
        raise ValueError(
            "This experiment is fixed to the public EVA02-CLIP-B/16 checkpoint "
            f"{EVA_CLIP_HUB_ID!r}, got {model_id!r}."
        )

    local_dir_value = model_config.get("local_dir")
    if local_dir_value not in {None, ""}:
        local_dir = Path(str(local_dir_value)).expanduser()
        if not local_dir.is_absolute():
            local_dir = PROJECT_ROOT / local_dir
        local_dir = local_dir.resolve()
        required_files = (
            local_dir / "open_clip_config.json",
            local_dir / "open_clip_model.safetensors",
        )
        missing_files = [str(path) for path in required_files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                "The configured offline EVA-CLIP directory is incomplete. Missing: "
                + ", ".join(missing_files)
            )
        model_source = f"local-dir:{local_dir}"
    else:
        model_source = f"hf-hub:{model_id}"

    cache_dir = model_config.get("cache_dir")
    create_kwargs: dict[str, Any] = {
        "device": "cpu",
        "precision": "fp32",
        "force_image_size": int(image_size),
        "require_pretrained": True,
        "weights_only": True,
    }
    if cache_dir not in {None, ""}:
        create_kwargs["cache_dir"] = str(cache_dir)

    # OpenCLIP maps the complete CLIP checkpoint into a timm-backed EVA tower.
    # The text tower and CLIP projection are then discarded: this detector
    # consumes raw contextualized patch tokens from the visual trunk.
    clip_model = open_clip.create_model(model_source, **create_kwargs)
    preprocess_config = open_clip.get_model_preprocess_cfg(clip_model)
    image_mean, image_std = _rgb_stats(preprocess_config)

    visual = getattr(clip_model, "visual", None)
    backbone = getattr(visual, "trunk", None)
    if backbone is None:
        raise RuntimeError(
            "The selected EVA-CLIP checkpoint did not expose the expected timm visual trunk. "
            "Check the installed open_clip_torch/timm versions from requirements.txt."
        )
    return backbone, image_mean, image_std


class EvaClipGAPClassifier(nn.Module):
    """Frozen EVA02-CLIP-B/16 patch encoder followed by GAP and one logit."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(getattr(backbone, "num_features", 0))
        patch_embed = getattr(backbone, "patch_embed", None)
        self.patch_size = _as_pair(
            getattr(patch_embed, "patch_size", None),
            name="EVA-CLIP patch size",
        )
        self.num_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 0))
        self.image_mean = list(image_mean or [0.48145466, 0.4578275, 0.40821073])
        self.image_std = list(image_std or [0.26862954, 0.26130258, 0.27577711])

        if self.hidden_size != EVA_CLIP_HIDDEN_SIZE:
            raise ValueError(
                "This experiment is fixed to EVA02-CLIP-B/16 "
                f"(hidden_size={EVA_CLIP_HIDDEN_SIZE}), got {self.hidden_size}."
            )
        if self.patch_size != (EVA_CLIP_PATCH_SIZE, EVA_CLIP_PATCH_SIZE):
            raise ValueError(
                "This experiment is fixed to EVA02-CLIP-B/16 "
                f"(patch_size={EVA_CLIP_PATCH_SIZE}), got {self.patch_size}."
            )
        if self.num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must not be negative.")

        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.head = nn.Linear(self.hidden_size, 1, bias=True)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    @classmethod
    def from_pretrained(
        cls,
        model_config: dict[str, Any],
        *,
        image_size: int,
    ) -> "EvaClipGAPClassifier":
        backbone, image_mean, image_std = load_evaclip_backbone(
            model_config,
            image_size=image_size,
        )
        return cls(backbone, image_mean=image_mean, image_std=image_std)

    def train(self, mode: bool = True) -> "EvaClipGAPClassifier":
        super().train(mode)
        # A linear probe keeps the feature extractor frozen and in eval mode.
        self.backbone.eval()
        return self

    def encode_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the visual trunk. A linear probe never needs trunk gradients."""
        with torch.no_grad():
            return self.backbone.forward_features(pixel_values)

    def extract_gap_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        patch_height, patch_width = self.patch_size
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError(
                f"Input spatial size {(height, width)} must be divisible by "
                f"patch size {self.patch_size}."
            )

        tokens = self.encode_tokens(pixel_values)

        expected_patch_count = (height // patch_height) * (width // patch_width)
        expected_sequence_length = self.num_prefix_tokens + expected_patch_count
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            shape = tuple(tokens.shape) if isinstance(tokens, torch.Tensor) else type(tokens).__name__
            raise RuntimeError(
                "Unexpected EVA-CLIP visual output: expected a [batch, tokens, channels] "
                f"tensor, got {shape}."
            )
        if tokens.shape[1] != expected_sequence_length or tokens.shape[2] != self.hidden_size:
            raise RuntimeError(
                "Unexpected EVA-CLIP token layout: "
                f"got {tuple(tokens.shape)}, expected sequence length {expected_sequence_length} "
                f"({self.num_prefix_tokens} prefix + {expected_patch_count} patches) "
                f"and hidden size {self.hidden_size}."
            )

        patch_tokens = tokens[:, self.num_prefix_tokens :, :]
        return patch_tokens.mean(dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        gap_features = self.extract_gap_features(pixel_values)
        return self.head(gap_features).squeeze(-1)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]
