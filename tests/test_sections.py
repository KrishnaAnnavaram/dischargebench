from clinical_summarization.data.sections import extract_bhc, find_sections, remove_bhc

NOTE = """Chief Complaint:
Shortness of breath

History of Present Illness:
72 year old with CHF presenting with dyspnea.

Brief Hospital Course:
Patient was diuresed with IV furosemide.
Atrial fibrillation: rate controlled with metoprolol.
# Hypokalemia: repleted.

Discharge Medications:
1. Furosemide 40 mg PO daily

Discharge Disposition:
Home
"""


def test_finds_known_sections_in_order():
    names = [s.name for s in find_sections(NOTE)]
    assert names == ["Chief Complaint", "History Of Present Illness", "Brief Hospital Course",
                     "Discharge Medications", "Discharge Disposition"]


def test_remove_bhc_removes_whole_section_including_problem_lines():
    out, removed = remove_bhc(NOTE)
    assert removed
    assert "diuresed" not in out
    assert "rate controlled" not in out          # a "Word:" line inside the BHC must not end the section early
    assert "Hypokalemia" not in out
    assert "Discharge Medications:" in out and "History of Present Illness:" in out


def test_extract_bhc_body():
    body = extract_bhc(NOTE)
    assert body.startswith("Patient was diuresed") and "repleted." in body


def test_note_without_bhc_is_unchanged():
    text = "Chief Complaint:\nChest pain\n"
    assert remove_bhc(text) == (text, False)
    assert extract_bhc(text) is None
