from __future__ import annotations

import numpy as np

from speechlens.config import load_config
from speechlens.features.extract import FeatureBundle, FrameTimes, VADResult
from speechlens.flaws.types import Flaw


def _speech_span(vad: VADResult, frame_times: FrameTimes) -> tuple[float, float]:
    speech_idx = np.where(vad.is_speech)[0]
    if len(speech_idx) == 0:
        return (0.0, 0.0)
    hop_sec = frame_times.hop_length / frame_times.sample_rate
    return (float(frame_times.times[speech_idx[0]]),
            float(frame_times.times[speech_idx[-1]] + hop_sec))


def _f0_iqr_semitones(f0_array, voiced_flag) -> float | None:
    voiced_f0 = f0_array[voiced_flag]
    if len(voiced_f0) == 0:
        return None
    median_f0 = float(np.median(voiced_f0))
    if median_f0 <= 0:
        return None
    semitones = 12 * np.log2(voiced_f0 / median_f0)
    return float(np.percentile(semitones, 75) - np.percentile(semitones, 25))


def _long_pause_severity(excess_ratio: float, bands: dict) -> str:
    if excess_ratio >= bands["severe"]["min_excess_ratio"]:
        return "severe"
    if excess_ratio >= bands["moderate"]["min_excess_ratio"]:
        return "moderate"
    return "mild"


def _rate_severity(excess: float, bands: dict) -> str:
    if excess >= bands["severe"]["min_excess"]:
        return "severe"
    if excess >= bands["moderate"]["min_excess"]:
        return "moderate"
    return "mild"


def _pitch_severity(iqr_ratio: float, bands: dict) -> str:
    if iqr_ratio < bands["severe"]["max_ratio"]:
        return "severe"
    if iqr_ratio < bands["moderate"]["max_ratio"]:
        return "moderate"
    return "mild"


def detect_long_pauses(
    participant: FeatureBundle,
    reference: FeatureBundle,
    alignment_method: str,
) -> list[Flaw]:
    cfg = load_config("rules_v1")["long_pause"]
    floor = cfg["absolute_floor_seconds"]
    factor = cfg["reference_factor"]
    bands = cfg["severity_bands"]
    recommendation = cfg["recommendation"]

    ref_pause_durs = [p.end - p.start for p in reference.vad.pauses]
    if ref_pause_durs:
        ref_median = float(np.median(ref_pause_durs))
        ref_p90 = float(np.percentile(ref_pause_durs, 90))
    else:
        ref_median = 0.0
        ref_p90 = 0.0

    threshold = max(floor, factor * ref_median)

    ref_measurement = {
        "median_pause_seconds": round(ref_median, 4),
        "p90_pause_seconds": round(ref_p90, 4),
        "pause_count": len(ref_pause_durs),
    }

    mode_note = (
        f"Pauses are not matched one-to-one in {alignment_method} mode; "
        "each participant pause is compared against the reference distribution."
    )

    flaws: list[Flaw] = []
    for pause in participant.vad.pauses:
        dur = pause.end - pause.start
        if dur <= threshold:
            continue

        excess_ratio = dur / threshold
        severity = _long_pause_severity(excess_ratio, bands)

        evidence = (
            f"Participant pause of {dur:.2f} s at {pause.start:.2f}–{pause.end:.2f} s. "
            f"Reference median pause {ref_median:.2f} s ({len(ref_pause_durs)} pauses). "
            f"Threshold {threshold:.2f} s (max of {floor} s floor, {factor}× reference median). "
            f"{mode_note}"
        )
        interpretation = (
            f"A {dur:.1f} s pause ({excess_ratio:.1f}× the threshold) "
            f"may disrupt the listener’s engagement at this point in the delivery."
        )

        flaws.append(Flaw(
            rule_id="long_pause",
            category=cfg["category"],
            severity=severity,
            start_seconds=pause.start,
            end_seconds=pause.end,
            participant_measurement={"pause_duration_seconds": round(dur, 4)},
            reference_measurement=ref_measurement,
            deviation=round(excess_ratio, 4),
            evidence=evidence,
            interpretation=interpretation,
            recommendation=recommendation,
        ))

    return flaws


def detect_rate_deviation(
    participant: FeatureBundle,
    reference: FeatureBundle,
) -> list[Flaw]:
    cfg = load_config("rules_v1")["rate_deviation"]
    upper = cfg["upper_bound"]
    lower = cfg["lower_bound"]
    bands = cfg["severity_bands"]

    if participant.rate is None or reference.rate is None:
        return []
    if reference.rate.articulation_rate_wpm <= 0:
        return []

    ratio = participant.rate.articulation_rate_wpm / reference.rate.articulation_rate_wpm

    if lower <= ratio <= upper:
        return []

    if ratio > upper:
        excess = ratio - upper
        direction = "faster"
        recommendation = cfg["recommendation_fast"]
    else:
        excess = lower - ratio
        direction = "slower"
        recommendation = cfg["recommendation_slow"]

    severity = _rate_severity(excess, bands)
    pct = abs(ratio - 1.0) * 100

    start, end = _speech_span(participant.vad, participant.frame_times)

    evidence = (
        f"Participant articulation rate {participant.rate.articulation_rate_wpm:.1f} WPM "
        f"vs reference {reference.rate.articulation_rate_wpm:.1f} WPM "
        f"(ratio {ratio:.2f})."
    )
    interpretation = (
        f"Speaking {pct:.0f}% {direction} than the reference "
        f"may affect clarity and listener comprehension."
    )

    return [Flaw(
        rule_id="rate_deviation",
        category=cfg["category"],
        severity=severity,
        start_seconds=start,
        end_seconds=end,
        participant_measurement={
            "articulation_rate_wpm": round(participant.rate.articulation_rate_wpm, 2),
        },
        reference_measurement={
            "articulation_rate_wpm": round(reference.rate.articulation_rate_wpm, 2),
        },
        deviation=round(ratio, 4),
        evidence=evidence,
        interpretation=interpretation,
        recommendation=recommendation,
        scope="recording",
    )]


def detect_flat_pitch(
    participant: FeatureBundle,
    reference: FeatureBundle,
) -> tuple[list[Flaw], list[str]]:
    cfg = load_config("rules_v1")["flat_pitch"]
    min_fraction = cfg["min_iqr_fraction"]
    min_voiced = cfg["min_voiced_frames"]
    bands = cfg["severity_bands"]
    recommendation = cfg["recommendation"]

    part_voiced_count = int(participant.f0.voiced_flag.sum())
    ref_voiced_count = int(reference.f0.voiced_flag.sum())

    if part_voiced_count < min_voiced or ref_voiced_count < min_voiced:
        reason = (
            f"flat_pitch: too few voiced frames "
            f"(participant {part_voiced_count}, reference {ref_voiced_count}, "
            f"minimum {min_voiced})"
        )
        return [], [reason]

    part_iqr = _f0_iqr_semitones(participant.f0.f0, participant.f0.voiced_flag)
    ref_iqr = _f0_iqr_semitones(reference.f0.f0, reference.f0.voiced_flag)

    if part_iqr is None or ref_iqr is None or ref_iqr <= 0:
        return [], ["flat_pitch: could not compute F0 IQR"]

    iqr_ratio = part_iqr / ref_iqr

    if iqr_ratio >= min_fraction:
        return [], []

    severity = _pitch_severity(iqr_ratio, bands)
    start, end = _speech_span(participant.vad, participant.frame_times)

    evidence = (
        f"Participant F0 IQR {part_iqr:.2f} st vs reference {ref_iqr:.2f} st "
        f"(ratio {iqr_ratio:.2f}). "
        f"Participant median F0 {participant.f0.median_f0_hz:.1f} Hz, "
        f"reference median F0 {reference.f0.median_f0_hz:.1f} Hz."
    )
    interpretation = (
        f"Pitch variation at {iqr_ratio:.0%} of the reference level "
        f"may make the delivery sound monotone to the listener."
    )

    return [Flaw(
        rule_id="flat_pitch",
        category=cfg["category"],
        severity=severity,
        start_seconds=start,
        end_seconds=end,
        participant_measurement={
            "f0_iqr_semitones": round(part_iqr, 4),
            "median_f0_hz": round(participant.f0.median_f0_hz, 2),
            "voiced_fraction": round(participant.f0.voiced_fraction, 4),
        },
        reference_measurement={
            "f0_iqr_semitones": round(ref_iqr, 4),
            "median_f0_hz": round(reference.f0.median_f0_hz, 2),
            "voiced_fraction": round(reference.f0.voiced_fraction, 4),
        },
        deviation=round(iqr_ratio, 4),
        evidence=evidence,
        interpretation=interpretation,
        recommendation=recommendation,
        scope="recording",
    )], []


def detect_all_flaws(
    participant: FeatureBundle,
    reference: FeatureBundle,
    alignment_method: str = "energy_fallback",
) -> tuple[list[Flaw], list[str]]:
    flaws: list[Flaw] = []
    unavailable: list[str] = []

    flaws.extend(detect_long_pauses(participant, reference, alignment_method))
    flaws.extend(detect_rate_deviation(participant, reference))

    pitch_flaws, pitch_unavailable = detect_flat_pitch(participant, reference)
    flaws.extend(pitch_flaws)
    unavailable.extend(pitch_unavailable)

    return flaws, unavailable
