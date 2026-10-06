"""Section parsing for discharge notes, and removal of the Brief Hospital Course (the reference summary).

The BHC section ends at the next *known* top-level header. A generic "Word:" pattern is not used for this,
because problem-based lines inside the BHC (e.g. "Atrial fibrillation: rate controlled") would end the
section early and leak the rest of the reference into the model input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

BHC_HEADERS = ("Brief Hospital Course", "Hospital Course")

KNOWN_HEADERS = (
    "Chief Complaint",
    "Major Surgical or Invasive Procedure",
    "History of Present Illness",
    "Past Medical History",
    "Social History",
    "Family History",
    "Physical Exam",
    "Pertinent Results",
    "Medications on Admission",
    "Discharge Medications",
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Followup Instructions",
    "Follow-up Instructions",
    "Allergies",
    "Facility",
) + BHC_HEADERS


def _header_regex(names: tuple[str, ...]) -> re.Pattern[str]:
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.compile(rf"^[ \t]*({alt})[ \t]*:", re.IGNORECASE | re.MULTILINE)


_ANY_HEADER = _header_regex(KNOWN_HEADERS)


@dataclass(frozen=True)
class Section:
    name: str
    start: int  # offset of the header
    end: int    # offset where the next known header starts (or end of text)


def find_sections(text: str) -> list[Section]:
    """Return known top-level sections in order of appearance."""
    matches = list(_ANY_HEADER.finditer(text or ""))
    sections = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(Section(m.group(1).title(), m.start(), end))
    return sections


def _is_bhc(section: Section) -> bool:
    return section.name.lower() in {h.lower() for h in BHC_HEADERS}


def extract_bhc(text: str) -> str | None:
    """Return the BHC section body (without its header), or None if the note has none."""
    for s in find_sections(text):
        if _is_bhc(s):
            body = text[s.start:s.end]
            return body.split(":", 1)[1].strip() if ":" in body else body.strip()
    return None


def remove_bhc(text: str) -> tuple[str, bool]:
    """Remove every BHC section from a note. Returns (text_without_bhc, removed_anything)."""
    if not text:
        return text, False
    spans = [(s.start, s.end) for s in find_sections(text) if _is_bhc(s)]
    if not spans:
        return text, False
    out, last = [], 0
    for start, end in spans:
        out.append(text[last:start])
        last = end
    out.append(text[last:])
    return "".join(out), True
