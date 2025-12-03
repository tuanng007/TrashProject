"""Utilities for seeding all relevant RNGs to make runs reproducible."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU/CUDA) and enforce deterministic ops."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Might raise for unsupported ops; keeps results repeatable when possible.
    torch.use_deterministic_algorithms(True)

