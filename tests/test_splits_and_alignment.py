import pandas as pd
import pytest

from clinical_summarization.data.loaders import build_pairs
from clinical_summarization.data.splits import split_of
from clinical_summarization.eval.metrics import align


def test_split_is_deterministic_and_patient_level():
    assert split_of(10001) == split_of(10001)
    counts = pd.Series([split_of(i) for i in range(20000)]).value_counts(normalize=True)
    assert abs(counts["train"] - 0.8) < 0.02 and abs(counts["test"] - 0.1) < 0.02


def test_build_pairs_joins_by_note_id_not_position():
    notes = pd.DataFrame({"note_id": ["a", "b"], "subject_id": [1, 1],
                          "text": ["Chief Complaint:\nA\nBrief Hospital Course:\nref A\nDischarge Disposition:\nHome",
                                   "Chief Complaint:\nB"]})
    targets = pd.DataFrame({"note_id": ["b", "a"], "target": ["target B", "target A"]})   # reversed order
    pairs = build_pairs(notes, targets).set_index("note_id")
    assert pairs.loc["a", "target"] == "target A" and pairs.loc["b", "target"] == "target B"
    assert "ref A" not in pairs.loc["a", "input_text"] and bool(pairs.loc["a", "bhc_removed"])
    assert pairs.loc["a", "split"] == pairs.loc["b", "split"]        # same patient -> same split


def test_align_refuses_mismatched_ids():
    with pytest.raises(ValueError):
        align({"a": "x", "c": "y"}, {"a": "r1", "b": "r2"})
    ids, preds, refs = align({"b": "pb", "a": "<think>t</think>pa"}, {"a": "ra", "b": "rb"})
    assert ids == ["a", "b"] and preds == ["pa", "pb"] and refs == ["ra", "rb"]
