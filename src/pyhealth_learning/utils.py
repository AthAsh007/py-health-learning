"""Small utilities shared across examples."""

from __future__ import annotations

import random

import numpy as np

__all__ = ["set_seed", "section"]


def set_seed(seed: int = 42) -> None:
    """Seed Python and NumPy RNGs for reproducible example runs."""
    random.seed(seed)
    np.random.seed(seed)


def section(title: str) -> None:
    """Print a tidy section header (used by the example scripts)."""
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")
