"""Text utilities: reasoning-tag stripping, whitespace normalisation and word-boundary-safe abbreviation expansion."""
from __future__ import annotations

import re
from collections.abc import Mapping

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_DEID_BLANK = re.compile(r"_{3,}")

# Deliberately small and unambiguous. Ambiguous short forms (e.g. "SZ", "BPD", "d/c") are left alone.
DEFAULT_ABBREVIATIONS: dict[str, str] = {
    "HTN": "hypertension",
    "HLD": "hyperlipidemia",
    "DM": "diabetes mellitus",
    "CHF": "congestive heart failure",
    "COPD": "chronic obstructive pulmonary disease",
    "CAD": "coronary artery disease",
    "CKD": "chronic kidney disease",
    "SOB": "shortness of breath",
    "NKDA": "no known drug allergies",
    "s/p": "status post",
    "h/o": "history of",
    "r/o": "rule out",
    "c/w": "consistent with",
}


def normalize_whitespace(text: str | None) -> str:
    """Collapse all runs of whitespace into single spaces."""
    return _WHITESPACE.sub(" ", text or "").strip()


def strip_reasoning(text: str | None) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models (also an unterminated trailing block)."""
    if not text:
        return ""
    out = _THINK_BLOCK.sub(" ", text)
    out = _THINK_UNCLOSED.sub(" ", out)
    return normalize_whitespace(out)


def mask_deidentified(text: str, token: str = "[REDACTED]") -> str:
    """Replace MIMIC de-identification blanks (``___``) with an explicit token."""
    return _DEID_BLANK.sub(token, text)


def expand_abbreviations(text: str, mapping: Mapping[str, str] | None = None) -> str:
    """Expand abbreviations only when they stand alone as tokens.

    Unlike plain ``str.replace``, "ART" is never replaced inside "ARTERIAL" and "DM" never inside "ADMIT".
    Matching is case-sensitive, longest key first.
    """
    mapping = dict(mapping or DEFAULT_ABBREVIATIONS)
    if not text or not mapping:
        return text
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"(?<![A-Za-z0-9/])(" + "|".join(re.escape(k) for k in keys) + r")(?![A-Za-z0-9/])")
    return pattern.sub(lambda m: mapping[m.group(1)], text)
