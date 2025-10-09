"""Evaluation metrics for waste classification."""

from __future__ import annotations

from typing import Dict

import torch
from sklearn.metrics import classification_report, confusion_matrix


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def compute_classification_report(
    y_true, y_pred, target_names=None
) -> Dict[str, Dict[str, float]]:
    """Return macro precision/recall/F1 and per-class metrics."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return report


def compute_confusion_matrix(y_true, y_pred) -> torch.Tensor:
    return torch.tensor(confusion_matrix(y_true, y_pred))

