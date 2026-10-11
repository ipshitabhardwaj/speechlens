from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Flaw:
    rule_id: str
    category: str
    severity: str
    start_seconds: float
    end_seconds: float
    participant_measurement: dict
    reference_measurement: dict
    deviation: float
    evidence: str
    interpretation: str
    recommendation: str
    scope: str = "local"
