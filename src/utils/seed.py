"""Utilities for seeding all relevant RNGs to make runs reproducible."""

import os
import random
import sys

import numpy as np
import torch

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU/CUDA).

    deterministic=True enforces deterministic algorithms; set False to avoid cuBLAS constraints.
    """
    # Needed for deterministic cuBLAS kernels on CUDA >= 10.2
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:  # pragma: no cover - defensive path
            print(
                f"[seed_everything] Deterministic algorithms not fully supported, falling back: {exc}",
                file=sys.stderr,
            )
            torch.use_deterministic_algorithms(False)
