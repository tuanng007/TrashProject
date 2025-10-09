"""Optimizer and scheduler builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch.optim import AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR, _LRScheduler


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    name: Optional[str] = "onecycle"
    max_lr: Optional[float] = None
    steps_per_epoch: Optional[int] = None
    epochs: Optional[int] = None
    gamma: float = 0.1
    step_size: int = 10


def build_optimizer(
    model: torch.nn.Module,
    config: OptimConfig,
    backbone_params: Optional[Iterable] = None,
    backbone_lr: Optional[float] = None,
) -> Optimizer:
    """Create optimizer with optional differential learning rates."""
    if backbone_params is not None and backbone_lr is not None:
        backbone_list = list(backbone_params)
        backbone_ids = {id(p) for p in backbone_list}
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        params = [
            {"params": backbone_list, "lr": backbone_lr},
            {"params": head_params, "lr": config.lr},
        ]
    else:
        params = model.parameters()

    name = config.name.lower()
    if name == "adamw":
        return AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if name == "sgd":
        return SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer: {config.name}")


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
) -> Optional[_LRScheduler]:
    """Create learning rate scheduler from configuration."""
    if not config or config.name is None:
        return None

    name = config.name.lower()
    if name == "onecycle":
        if config.max_lr is None or config.steps_per_epoch is None or config.epochs is None:
            raise ValueError("OneCycle requires max_lr, steps_per_epoch, and epochs.")
        return OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            steps_per_epoch=config.steps_per_epoch,
            epochs=config.epochs,
        )
    if name == "cosine":
        if config.epochs is None:
            raise ValueError("Cosine scheduler requires epochs (T_max).")
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    if name == "step":
        return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    raise ValueError(f"Unsupported scheduler: {config.name}")
