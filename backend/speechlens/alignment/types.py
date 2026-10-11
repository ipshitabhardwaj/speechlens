from __future__ import annotations

from dataclasses import dataclass

from speechlens.features.extract import PauseInterval


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float
    score: float


@dataclass(frozen=True)
class AlignmentResult:
    method: str
    words: list[WordTiming] | None
    pauses: list[PauseInterval]
    unavailable: list[str]
