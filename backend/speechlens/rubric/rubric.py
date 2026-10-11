from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from speechlens.config import load_config
from speechlens.features.extract import FeatureBundle
from speechlens.flaws.types import Flaw


@dataclass(frozen=True)
class DimensionScore:
    name: str
    score: float
    weight: float
    available: bool
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class RubricResult:
    dimensions: list[DimensionScore]
    overall_score: float
    version: str


def _f0_iqr_semitones(f0_array, voiced_flag) -> float | None:
    voiced_f0 = f0_array[voiced_flag]
    if len(voiced_f0) == 0:
        return None
    median_f0 = float(np.median(voiced_f0))
    if median_f0 <= 0:
        return None
    semitones = 12 * np.log2(voiced_f0 / median_f0)
    return float(np.percentile(semitones, 75) - np.percentile(semitones, 25))


def _score_pacing(
    participant: FeatureBundle,
    reference: FeatureBundle,
    cfg: dict,
) -> DimensionScore:
    if (participant.rate is None or reference.rate is None
            or reference.rate.articulation_rate_wpm <= 0):
        return DimensionScore(
            name="pacing", score=0.0, weight=cfg["weight"],
            available=False, unavailable_reason="missing rate data",
        )

    ratio = participant.rate.articulation_rate_wpm / reference.rate.articulation_rate_wpm
    deviation = abs(ratio - 1.0)
    score = max(0.0, cfg["base_score"] - cfg["penalty_per_unit"] * deviation)

    return DimensionScore(name="pacing", score=score, weight=cfg["weight"], available=True)


def _score_pause_control(flaws: list[Flaw], cfg: dict) -> DimensionScore:
    penalty = 0.0
    for flaw in flaws:
        if flaw.rule_id != "long_pause":
            continue
        penalty += {
            "mild": cfg["penalty_mild"],
            "moderate": cfg["penalty_moderate"],
            "severe": cfg["penalty_severe"],
        }.get(flaw.severity, 0)

    score = max(0.0, cfg["base_score"] - penalty)
    return DimensionScore(name="pause_control", score=score, weight=cfg["weight"], available=True)


def _score_vocal_variation(
    participant: FeatureBundle,
    reference: FeatureBundle,
    unavailable_reasons: list[str],
    cfg: dict,
) -> DimensionScore:
    pitch_unavailable = [r for r in unavailable_reasons if r.startswith("flat_pitch:")]
    if pitch_unavailable:
        return DimensionScore(
            name="vocal_variation", score=0.0, weight=cfg["weight"],
            available=False, unavailable_reason=pitch_unavailable[0],
        )

    part_iqr = _f0_iqr_semitones(participant.f0.f0, participant.f0.voiced_flag)
    ref_iqr = _f0_iqr_semitones(reference.f0.f0, reference.f0.voiced_flag)

    if part_iqr is None or ref_iqr is None or ref_iqr <= 0:
        return DimensionScore(
            name="vocal_variation", score=0.0, weight=cfg["weight"],
            available=False, unavailable_reason="flat_pitch: could not compute F0 IQR",
        )

    iqr_ratio = min(part_iqr / ref_iqr, 1.0)
    score = cfg["base_score"] * max(0.0, iqr_ratio)

    return DimensionScore(name="vocal_variation", score=score, weight=cfg["weight"], available=True)


def score_rubric(
    participant: FeatureBundle,
    reference: FeatureBundle,
    flaws: list[Flaw],
    unavailable: list[str],
) -> RubricResult:
    cfg = load_config("rubric_v1")
    dims_cfg = cfg["dimensions"]

    dimensions = [
        _score_pacing(participant, reference, dims_cfg["pacing"]),
        _score_pause_control(flaws, dims_cfg["pause_control"]),
        _score_vocal_variation(participant, reference, unavailable, dims_cfg["vocal_variation"]),
    ]

    available = [d for d in dimensions if d.available]
    if available:
        total_weight = sum(d.weight for d in available)
        overall = sum(d.score * d.weight for d in available) / total_weight
    else:
        overall = 0.0

    return RubricResult(
        dimensions=dimensions,
        overall_score=round(overall, 2),
        version=cfg["version"],
    )
