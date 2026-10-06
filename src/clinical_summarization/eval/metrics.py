"""Metric computation over note_id-aligned predictions and references."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..text import strip_reasoning


def align(predictions: Mapping[str, str], references: Mapping[str, str], *, allow_subset: bool = False
          ) -> tuple[list[str], list[str], list[str]]:
    """Return (ids, preds, refs) in the same order. Refuses to compare by position.

    By default the two mappings must contain exactly the same note_ids; with ``allow_subset`` only the
    predicted ids are used, but every one of them must have a reference.
    """
    pred_ids, ref_ids = set(predictions), set(references)
    missing = pred_ids - ref_ids
    if missing:
        raise ValueError(f"{len(missing)} predictions have no reference (e.g. {sorted(missing)[:3]})")
    if not allow_subset and pred_ids != ref_ids:
        raise ValueError(f"{len(ref_ids - pred_ids)} references have no prediction; pass allow_subset=True "
                         "only if the prediction set is the intended evaluation sample")
    ids = sorted(pred_ids)
    return ids, [strip_reasoning(predictions[i]) for i in ids], [references[i] for i in ids]


def compute(predictions: Mapping[str, str], references: Mapping[str, str],
            metrics: Sequence[str] = ("rouge",), allow_subset: bool = False) -> dict:
    """Compute metrics (requires the ``eval`` extra). Returns a dict of metric name -> scores."""
    import evaluate

    ids, preds, refs = align(predictions, references, allow_subset=allow_subset)
    results: dict = {"n": len(ids)}
    for m in metrics:
        if m == "rouge":
            results["rouge"] = evaluate.load("rouge").compute(predictions=preds, references=refs, use_stemmer=True)
        elif m == "bleu":
            results["bleu"] = evaluate.load("bleu").compute(predictions=preds, references=[[r] for r in refs])
        elif m == "bertscore":
            s = evaluate.load("bertscore").compute(predictions=preds, references=refs, lang="en")
            results["bertscore"] = {k: sum(s[k]) / len(s[k]) for k in ("precision", "recall", "f1")}
        else:
            raise ValueError(f"unknown metric {m!r}")
    return results
