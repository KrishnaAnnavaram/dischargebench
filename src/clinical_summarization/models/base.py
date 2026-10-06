"""The interface every model adapter implements."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Summarizer(Protocol):
    name: str

    def summarize(self, texts: list[str]) -> list[str]:
        """Return one summary per input, in the same order."""
        ...
