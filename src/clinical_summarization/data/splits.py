"""Deterministic, patient-level train/val/test splits.

A split is derived from a hash of (seed, subject_id), so every note of the same patient always lands in the
same split. No randomness at runtime, nothing to store, and identical across machines.
"""
from __future__ import annotations

import hashlib

import pandas as pd

SPLITS = ("train", "val", "test")


def split_of(subject_id: object, seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> str:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    if isinstance(subject_id, float) and subject_id.is_integer():
        # pandas reads an integer column with a blank cell as float64: 10001.0 must stay patient 10001.
        subject_id = int(subject_id)
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64          # uniform in [0, 1)
    if u < ratios[0]:
        return "train"
    if u < ratios[0] + ratios[1]:
        return "val"
    return "test"


def assign_splits(df: pd.DataFrame, id_col: str = "subject_id", seed: int = 42,
                  ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> pd.Series:
    """Return a Series of split names aligned with ``df``."""
    if id_col not in df.columns:
        raise KeyError(f"column {id_col!r} is required for patient-level splits")
    return df[id_col].map(lambda s: split_of(s, seed, ratios))
