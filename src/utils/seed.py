"""Reproducibility helpers.

`set_seed` should be called once, as early as possible in a run, before any
library that draws random numbers is used. imbalanced-learn and scikit-learn
estimators are also given an explicit `random_state` from the config, which is
what actually pins their behaviour -- seeding the global RNGs here covers the
incidental randomness (train/test splits, shuffles, sampling in notebooks).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> int:
    """Seed Python's `random`, numpy's legacy global RNG, and `PYTHONHASHSEED`.

    Returns the seed so callers can log it.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def new_rng(seed: int) -> np.random.Generator:
    """A fresh, independent numpy Generator -- preferred over the global RNG
    for anything new. Kept here so every module draws from the same convention.
    """
    return np.random.default_rng(seed)
