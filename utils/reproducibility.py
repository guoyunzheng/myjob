"""Opt-in process seeding; strict resume subsequently restores the saved RNG."""
import random
import numpy as np
import torch


def seed_training(seed, rank=0):
    if seed is None:
        return  # Preserve legacy launches that did not seed model/noise RNG.
    effective_seed = seed + rank
    if not 0 <= effective_seed < 2**32:
        raise ValueError("seed + rank must be in [0, 2**32).")
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)


def data_seed(seed):
    return 0 if seed is None else seed
