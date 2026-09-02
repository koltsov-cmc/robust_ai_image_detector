from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeEnvironment:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed() -> RuntimeEnvironment:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError(
            "This project intentionally permits one process on one GPU only; WORLD_SIZE must equal 1."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Training/inference is GPU-only and will not silently fall back "
            "to CPU. Check the CUDA-enabled PyTorch installation and NVIDIA driver."
        )
    visible_device_count = torch.cuda.device_count()
    if visible_device_count != 1:
        raise RuntimeError(
            f"Exactly one CUDA device must be visible, but PyTorch sees {visible_device_count}. "
            "Select one physical GPU before launch, for example: CUDA_VISIBLE_DEVICES=3 python ..."
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    return RuntimeEnvironment(
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
        distributed=False,
    )


def set_global_seed(seed: int, rank: int = 0, deterministic: bool = False) -> None:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed % (2**32))
    # torch.manual_seed() delegates to cuda.manual_seed_all(); seed the CPU
    # generator explicitly so no non-selected GPU is touched.
    torch.random.default_generator.manual_seed(effective_seed)
    if torch.cuda.is_available():
        # Seed only the current logical cuda:0. Do not initialize/touch other
        # visible L40 devices in this deliberately single-GPU experiment.
        torch.cuda.manual_seed(effective_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False


def barrier(environment: RuntimeEnvironment) -> None:
    del environment


def cleanup_distributed(environment: RuntimeEnvironment) -> None:
    del environment


def reduce_sum(value: float | int, environment: RuntimeEnvironment) -> float:
    del environment
    return float(value)


def gather_object(value: Any, environment: RuntimeEnvironment) -> list[Any]:
    del environment
    return [value]
