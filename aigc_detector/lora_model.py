"""LoRA-adapted variant of the EVA02-CLIP-B/16 GAP detector.

The linear-probe classifier in :mod:`aigc_detector.model` runs its trunk under
``torch.no_grad()``. That is correct for a probe but blocks the gradient a LoRA
adapter needs, so this module provides a subclass that keeps the gradient path
open. The base trunk weights stay frozen either way: only the injected LoRA
matrices ever require a gradient.

An adapter produced here lives entirely inside the visual trunk, which both
released detector variants share bit for bit. Only the 769-parameter head
differs between them, so the same adapter directory can be attached to either.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .model import EvaClipGAPClassifier


# Matched with ``re.fullmatch`` against trunk-relative module names. Anchoring on
# ``blocks.<n>.attn.`` matters: a bare "proj" suffix would also capture the
# ``patch_embed.proj`` convolution, which must stay untouched.
LORA_TARGET_REGEX = r"blocks\.\d+\.attn\.(q_proj|k_proj|v_proj|proj)"

# Module names that must never be wrapped, whatever regex the caller passes.
FORBIDDEN_TARGET_SUBSTRINGS = ("patch_embed", "head")


def _require_peft():
    try:
        import peft
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "peft is required to train or attach LoRA adapters. "
            "Install it with `python3 -m pip install -r requirements.txt`."
        ) from error
    return peft


class EvaClipLoRAClassifier(EvaClipGAPClassifier):
    """EVA02-CLIP-B/16 GAP detector whose trunk can carry trainable LoRA adapters."""

    def train(self, mode: bool = True) -> "EvaClipLoRAClassifier":
        # Deliberately bypass the parent override that pins the trunk to eval():
        # LoRA dropout has to be active while the adapter trains.
        nn.Module.train(self, mode)
        return self

    def encode_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # No torch.no_grad() here: the gradient has to reach the LoRA matrices.
        return self.backbone.forward_features(pixel_values)


def load_detector_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a ``runs/<experiment>/best.pt`` checkpoint written by the native pipeline."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 3:
        raise ValueError(
            f"Unsupported checkpoint format in {checkpoint_path}: "
            f"expected format_version 3, got {checkpoint.get('format_version')!r}."
        )
    for key in ("head_state_dict", "preprocessing", "model"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_path} has no {key!r} entry.")
    return checkpoint


def build_lora_detector(
    checkpoint: dict[str, Any],
    *,
    local_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    freeze_head: bool = True,
) -> tuple[EvaClipLoRAClassifier, dict[str, Any]]:
    """Rebuild the detector described by ``checkpoint`` with a gradient-capable trunk.

    Preprocessing (input size and RGB statistics) and the backbone id come from
    the checkpoint itself rather than from a separate YAML file, so an adapter
    can never be trained against a different input contract than the head it is
    paired with.
    """
    preprocessing = checkpoint["preprocessing"]
    image_size = int(preprocessing["image_size"])

    model_config: dict[str, Any] = {"id": str(checkpoint["model"]["id"])}
    if local_dir not in {None, ""}:
        model_config["local_dir"] = str(local_dir)
    if cache_dir not in {None, ""}:
        model_config["cache_dir"] = str(cache_dir)

    model = EvaClipLoRAClassifier.from_pretrained(model_config, image_size=image_size).float()
    model.head.load_state_dict(checkpoint["head_state_dict"], strict=True)

    # The head defines the fixed direction in GAP feature space that the adapter
    # learns to aim distorted features at. Freezing it is what keeps the adapter
    # a pure, detachable trunk modification.
    if freeze_head:
        model.head.requires_grad_(False)

    # Keep the checkpoint's statistics rather than the ones open_clip reports, so
    # the dataset normalises exactly as the head expects.
    model.image_mean = [float(value) for value in preprocessing["image_mean"]]
    model.image_std = [float(value) for value in preprocessing["image_std"]]
    return model, preprocessing


def attach_lora(
    model: EvaClipLoRAClassifier,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_regex: str = LORA_TARGET_REGEX,
) -> tuple[Any, dict[str, Any]]:
    """Inject LoRA into the trunk in place and return the PeftModel plus a report.

    ``peft`` rewrites the matched children of the trunk in place, so after this
    call ``model.backbone.forward_features`` already routes through the adapter.
    The returned PeftModel handle is only needed for saving and adapter switching.
    """
    peft = _require_peft()

    pattern = re.compile(target_regex)
    planned = sorted(
        name
        for name, module in model.backbone.named_modules()
        if isinstance(module, nn.Linear) and pattern.fullmatch(name)
    )
    if not planned:
        available = sorted(
            {
                name.rsplit(".", 1)[-1]
                for name, module in model.backbone.named_modules()
                if isinstance(module, nn.Linear)
            }
        )
        raise RuntimeError(
            f"The LoRA target regex {target_regex!r} matched no nn.Linear module in the trunk. "
            f"Available linear-layer suffixes: {available}."
        )
    forbidden = [
        name
        for name in planned
        if any(marker in name for marker in FORBIDDEN_TARGET_SUBSTRINGS)
    ]
    if forbidden:
        raise RuntimeError(
            f"The LoRA target regex {target_regex!r} matched modules that must stay frozen: "
            f"{forbidden}."
        )

    peft_model = peft.get_peft_model(
        model.backbone,
        peft.LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            bias="none",
            target_modules=target_regex,
            init_lora_weights=True,
        ),
    )
    return peft_model, lora_target_report(model, planned=planned, target_regex=target_regex)


def lora_target_report(
    model: EvaClipLoRAClassifier,
    *,
    planned: list[str],
    target_regex: str,
) -> dict[str, Any]:
    """Verify that every planned module really got wrapped, and summarise the result."""
    from peft.tuners.lora import LoraLayer

    wrapped = sorted(
        name for name, module in model.backbone.named_modules() if isinstance(module, LoraLayer)
    )
    if not wrapped:
        raise RuntimeError(
            "LoRA injection wrapped zero modules. Refusing to train an adapter that "
            "cannot change the trunk."
        )
    missing = sorted(set(planned) - set(wrapped))
    unexpected = sorted(set(wrapped) - set(planned))
    if missing or unexpected:
        raise RuntimeError(
            "LoRA injection did not match the planned module set. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    blocks = sorted({int(match.group(1)) for name in wrapped if (match := re.match(r"blocks\.(\d+)\.", name))})
    return {
        "target_regex": target_regex,
        "wrapped_modules": len(wrapped),
        "wrapped_by_suffix": dict(sorted(Counter(name.rsplit(".", 1)[-1] for name in wrapped).items())),
        "wrapped_blocks": blocks,
        "wrapped_module_names": wrapped,
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "trainable_tensors": len(trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "head_trainable": any(name.startswith("head.") for name, _ in trainable),
        "non_lora_trainable_tensors": sorted(
            name for name, _ in trainable if ".lora_" not in name
        ),
    }


def print_lora_report(report: dict[str, Any], *, max_listed: int = 8) -> None:
    """Print where the adapter actually landed, so a silent no-op is impossible to miss."""
    total = report["total_parameters"]
    trainable = report["trainable_parameters"]
    percentage = 100.0 * trainable / total if total else 0.0
    names = report["wrapped_module_names"]
    print("[lora] target regex:", report["target_regex"])
    print(
        f"[lora] wrapped {report['wrapped_modules']} modules "
        f"across blocks {report['wrapped_blocks'][0]}-{report['wrapped_blocks'][-1]} "
        f"({len(report['wrapped_blocks'])} blocks)"
    )
    for suffix, count in report["wrapped_by_suffix"].items():
        print(f"[lora]   {suffix}: {count}")
    preview = names[:max_listed]
    print("[lora] modules:", ", ".join(preview) + (f", ... (+{len(names) - len(preview)})" if len(names) > len(preview) else ""))
    print(
        f"[lora] trainable {trainable:,} / {total:,} parameters ({percentage:.4f}%) "
        f"in {report['trainable_tensors']} tensors"
    )
    print(f"[lora] head trainable: {report['head_trainable']}")
    if report["non_lora_trainable_tensors"]:
        print("[lora] non-LoRA trainable tensors:", report["non_lora_trainable_tensors"])


def save_lora_adapter(peft_model: Any, directory: str | Path) -> Path:
    """Write ``adapter_config.json`` and ``adapter_model.safetensors`` into ``directory``."""
    output_dir = Path(directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(output_dir))
    weights = output_dir / "adapter_model.safetensors"
    if not weights.is_file():
        raise RuntimeError(f"peft did not write {weights}.")
    return output_dir


def load_lora_adapters(
    model: EvaClipLoRAClassifier,
    adapters: dict[str, Path],
) -> Any:
    """Attach one or more saved adapters to a trunk and return the PeftModel handle.

    All adapters share a single trunk in memory, so ``peft_model.set_adapter(name)``
    switches between them without reloading any weights.
    """
    peft = _require_peft()
    if not adapters:
        raise ValueError("At least one adapter directory is required.")

    names = list(adapters)
    for name, directory in adapters.items():
        config_path = Path(directory) / "adapter_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Adapter {name!r} is missing {config_path}. Train it with lora_train.py first."
            )

    peft_model = peft.PeftModel.from_pretrained(
        model.backbone,
        str(adapters[names[0]]),
        adapter_name=names[0],
        is_trainable=False,
    )
    for name in names[1:]:
        peft_model.load_adapter(str(adapters[name]), adapter_name=name, is_trainable=False)
    peft_model.set_adapter(names[0])
    return peft_model
