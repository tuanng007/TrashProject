"""Loss utilities for waste classification training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    name: str = "cross_entropy"
    gamma: float = 2.0  # Focal loss focusing parameter
    alpha: Optional[Iterable[float]] = None  # Class weights for focal loss


def compute_class_weights(class_counts: Iterable[int]) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    counts = torch.tensor(list(class_counts), dtype=torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("All class counts must be positive.")
    inv_freq = counts.sum() / (len(counts) * counts)
    return inv_freq / inv_freq.sum() * len(counts)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[Iterable[float]] = None,
) -> torch.Tensor:
    """Focal loss for multi-class classification."""
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    targets = targets.long()
    one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
    pt = (probs * one_hot).sum(dim=1)
    weights = (1 - pt) ** gamma
    if alpha is not None:
        alpha_tensor = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
        weights = weights * alpha_tensor[targets]
    loss = -(weights * log_probs.gather(1, targets.view(-1, 1)).squeeze(1))
    return loss.mean()


def build_loss(
    loss_config: LossConfig,
    class_counts: Optional[Iterable[int]] = None,
) -> callable:
    """Return a loss function based on configuration."""
    if loss_config.name.lower() == "cross_entropy":
        weight = None
        if class_counts is not None:
            weight = compute_class_weights(class_counts)
        return lambda logits, targets: F.cross_entropy(
            logits, targets, weight=weight.to(logits.device) if weight is not None else None
        )
    if loss_config.name.lower() == "focal":
        alpha = None
        if loss_config.alpha is not None:
            alpha = loss_config.alpha
        elif class_counts is not None:
            alpha = compute_class_weights(class_counts).tolist()
        return lambda logits, targets: focal_loss(
            logits,
            targets,
            gamma=loss_config.gamma,
            alpha=alpha,
        )
    raise ValueError(f"Unsupported loss: {loss_config.name}")

