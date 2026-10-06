from clinical_summarization.text import expand_abbreviations, mask_deidentified, strip_reasoning


def test_strip_reasoning_removes_think_blocks():
    out = strip_reasoning("<think>let me reason\nstep by step</think>\nPatient admitted with CHF exacerbation.")
    assert out == "Patient admitted with CHF exacerbation."


def test_strip_reasoning_handles_unclosed_block():
    assert strip_reasoning("Summary text. <think>unfinished reasoning") == "Summary text."


def test_expansion_respects_word_boundaries():
    text = "ARTERIAL line placed. PERRLA. ADMIT note. Hx of HTN and DM, s/p CABG."
    out = expand_abbreviations(text)
    assert "ARTERIAL" in out and "PERRLA" in out and "ADMIT" in out        # never touched inside words
    assert "hypertension" in out and "diabetes mellitus" in out and "status post" in out


def test_mask_deidentified():
    assert mask_deidentified("Name: ___ seen by Dr. ___") == "Name: [REDACTED] seen by Dr. [REDACTED]"
