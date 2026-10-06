"""Load discharge notes and BHC targets and build note_id-aligned (input, target) pairs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..text import expand_abbreviations, mask_deidentified
from .sections import remove_bhc
from .splits import assign_splits


def _read(path: str | Path, usecols: list[str]) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, usecols=lambda c: c in usecols)
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")
    return df[usecols]


def load_notes(path: str | Path, text_col: str = "text") -> pd.DataFrame:
    df = _read(path, ["note_id", "subject_id", text_col]).rename(columns={text_col: "text"})
    if df["note_id"].duplicated().any():
        raise ValueError("note_id must be unique in the notes file")
    return df


def load_targets(path: str | Path, target_col: str = "target") -> pd.DataFrame:
    df = _read(path, ["note_id", target_col]).rename(columns={target_col: "target"})
    if df["note_id"].duplicated().any():
        raise ValueError("note_id must be unique in the targets file")
    return df


def build_pairs(notes: pd.DataFrame, targets: pd.DataFrame, *, remove_reference: bool = True,
                expand_abbrev: bool = False, seed: int = 42) -> pd.DataFrame:
    """Join notes and targets on ``note_id`` (never by position) and prepare model inputs.

    Columns: note_id, subject_id, split, input_text, target, bhc_removed.
    """
    df = notes.merge(targets, on="note_id", how="inner", validate="one_to_one")
    inputs, removed = [], []
    for text in df["text"].fillna(""):
        if remove_reference:
            text, did = remove_bhc(text)
        else:
            did = False
        text = mask_deidentified(text)
        if expand_abbrev:
            text = expand_abbreviations(text)
        inputs.append(text.strip())
        removed.append(did)
    df = df.assign(input_text=inputs, bhc_removed=removed).drop(columns=["text"])
    df["split"] = assign_splits(df, "subject_id", seed=seed)
    df = df[df["input_text"].str.len() > 0]
    return df[["note_id", "subject_id", "split", "input_text", "target", "bhc_removed"]].reset_index(drop=True)
