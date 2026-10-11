from __future__ import annotations

import numpy as np

from speechlens.alignment.types import WordTiming
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


def _local_rate_severity(excess: float, bands: dict) -> str:
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


# ── pause-to-gap assignment ──────────────────────────────────────────────

def _assign_pauses_to_gaps(
    pauses: list,
    word_timings: list[WordTiming],
) -> dict[int, float]:
    """Map each word-gap index to the total pause duration in that gap.

    Gap i = between word i and word i+1. Uses word onsets as boundaries
    (more reliable than word ends per the spike finding).
    """
    if len(word_timings) < 2:
        return {}

    onsets = [w.start for w in word_timings]
    gap_durations: dict[int, float] = {}

    for pause in pauses:
        pause_mid = (pause.start + pause.end) / 2
        for i in range(len(onsets) - 1):
            if onsets[i] <= pause_mid < onsets[i + 1]:
                dur = pause.end - pause.start
                gap_durations[i] = gap_durations.get(i, 0.0) + dur
                break

    return gap_durations


# ── long pause detection ─────────────────────────────────────────────────

def detect_long_pauses(
    participant: FeatureBundle,
    reference: FeatureBundle,
    alignment_method: str,
    participant_words: list[WordTiming] | None = None,
    reference_words: list[WordTiming] | None = None,
    excluded_indices: set[int] | None = None,
) -> list[Flaw]:
    cfg = load_config("rules_v1")["long_pause"]
    floor = cfg["absolute_floor_seconds"]
    factor = cfg["reference_factor"]
    bands = cfg["severity_bands"]
    recommendation = cfg["recommendation"]

    if (
        participant_words is not None
        and reference_words is not None
        and len(participant_words) >= 2
        and len(reference_words) >= 2
    ):
        return _detect_long_pauses_matched(
            participant, reference, participant_words, reference_words,
            excluded_indices or set(),
            floor, factor, bands, recommendation,
        )

    return _detect_long_pauses_unmatched(
        participant, reference, alignment_method,
        floor, factor, bands, recommendation,
    )


def _detect_long_pauses_unmatched(
    participant: FeatureBundle,
    reference: FeatureBundle,
    alignment_method: str,
    floor: float,
    factor: float,
    bands: dict,
    recommendation: str,
) -> list[Flaw]:
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
            f"may disrupt the listener's engagement at this point in the delivery."
        )

        flaws.append(Flaw(
            rule_id="long_pause",
            category="pause_control",
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


def _detect_long_pauses_matched(
    participant: FeatureBundle,
    reference: FeatureBundle,
    participant_words: list[WordTiming],
    reference_words: list[WordTiming],
    excluded_indices: set[int],
    floor: float,
    factor: float,
    bands: dict,
    recommendation: str,
) -> list[Flaw]:
    cfg = load_config("rules_v1")["long_pause"]
    n_words = min(len(participant_words), len(reference_words))

    part_gaps = _assign_pauses_to_gaps(participant.vad.pauses, participant_words)
    ref_gaps = _assign_pauses_to_gaps(reference.vad.pauses, reference_words)

    flaws: list[Flaw] = []
    for i in range(n_words - 1):
        if i in excluded_indices or (i + 1) in excluded_indices:
            continue

        part_pause = part_gaps.get(i, 0.0)
        ref_pause = ref_gaps.get(i, 0.0)

        threshold = max(floor, factor * ref_pause) if ref_pause > 0 else floor

        if part_pause <= threshold:
            continue

        excess_ratio = part_pause / threshold
        severity = _long_pause_severity(excess_ratio, bands)
        preceding_word = participant_words[i].word

        evidence = (
            f"Participant pause of {part_pause:.2f} s after word '{preceding_word}' "
            f"(gap {i}). "
            f"Reference pause at same position: {ref_pause:.2f} s. "
            f"Threshold {threshold:.2f} s (max of {floor} s floor, "
            f"{factor}× matched reference pause)."
        )
        interpretation = (
            f"A {part_pause:.1f} s pause ({excess_ratio:.1f}× the threshold) "
            f"after '{preceding_word}' may disrupt the listener's engagement."
        )

        # Use VAD pause boundaries for the flaw time span
        pause_start = participant_words[i].end
        pause_end = participant_words[i + 1].start
        # Find the actual VAD pause that falls in this gap
        for p in participant.vad.pauses:
            p_mid = (p.start + p.end) / 2
            if participant_words[i].start <= p_mid < participant_words[i + 1].start:
                pause_start = p.start
                pause_end = p.end
                break

        flaws.append(Flaw(
            rule_id="long_pause",
            category=cfg["category"],
            severity=severity,
            start_seconds=pause_start,
            end_seconds=pause_end,
            participant_measurement={
                "pause_duration_seconds": round(part_pause, 4),
                "preceding_word": preceding_word,
                "gap_index": i,
            },
            reference_measurement={
                "matched_pause_seconds": round(ref_pause, 4),
            },
            deviation=round(excess_ratio, 4),
            evidence=evidence,
            interpretation=interpretation,
            recommendation=recommendation,
        ))

    return flaws


# ── rate deviation (recording-level) ─────────────────────────────────────

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


# ── local rate (word-indexed, sliding window) ────────────────────────────

def detect_local_rate(
    participant_words: list[WordTiming],
    reference_words: list[WordTiming],
    excluded_indices: set[int],
) -> list[Flaw]:
    cfg = load_config("rules_v1")["local_rate"]
    N = cfg["window_words"]
    upper = cfg["upper_bound"]
    lower = cfg["lower_bound"]
    bands = cfg["severity_bands"]

    n_words = min(len(participant_words), len(reference_words))
    if n_words < N:
        return []

    flagged_ranges: list[tuple[int, int, float]] = []

    for i in range(n_words - N + 1):
        word_indices = range(i, i + N)
        if any(j in excluded_indices for j in word_indices):
            continue

        p_span = participant_words[i + N - 1].start - participant_words[i].start
        r_span = reference_words[i + N - 1].start - reference_words[i].start

        if r_span <= 0:
            continue

        ratio = p_span / r_span

        if ratio > upper or ratio < lower:
            flagged_ranges.append((i, i + N - 1, ratio))

    if not flagged_ranges:
        return []

    # Merge overlapping ranges
    merged: list[tuple[int, int, list[float]]] = []
    cs, ce, cr = flagged_ranges[0][0], flagged_ranges[0][1], [flagged_ranges[0][2]]

    for start, end, ratio in flagged_ranges[1:]:
        if start <= ce:
            ce = max(ce, end)
            cr.append(ratio)
        else:
            merged.append((cs, ce, cr))
            cs, ce, cr = start, end, [ratio]
    merged.append((cs, ce, cr))

    flaws: list[Flaw] = []
    for start_idx, end_idx, ratios in merged:
        flaw_start = participant_words[start_idx].start
        if end_idx + 1 < len(participant_words):
            flaw_end = participant_words[end_idx + 1].start
        else:
            flaw_end = participant_words[end_idx].end

        mean_ratio = sum(ratios) / len(ratios)
        p_span = participant_words[end_idx].start - participant_words[start_idx].start
        r_span = reference_words[end_idx].start - reference_words[start_idx].start

        if mean_ratio > upper:
            excess = mean_ratio - upper
            direction = "slower"
            recommendation = cfg["recommendation_slow"]
        else:
            excess = lower - mean_ratio
            direction = "faster"
            recommendation = cfg["recommendation_fast"]

        severity = _local_rate_severity(excess, bands)

        word_names = [participant_words[j].word for j in range(start_idx, end_idx + 1)]
        evidence = (
            f"Local rate ratio {mean_ratio:.2f} over words {start_idx}–{end_idx} "
            f"({' '.join(word_names)}). "
            f"Participant onset span {p_span:.2f} s vs reference {r_span:.2f} s."
        )
        interpretation = (
            f"Speaking {direction} than the reference in this section "
            f"may affect pacing and listener comprehension."
        )

        flaws.append(Flaw(
            rule_id="local_rate",
            category=cfg["category"],
            severity=severity,
            start_seconds=flaw_start,
            end_seconds=flaw_end,
            participant_measurement={
                "onset_span_seconds": round(p_span, 4),
                "word_range": f"{start_idx}-{end_idx}",
            },
            reference_measurement={
                "onset_span_seconds": round(r_span, 4),
            },
            deviation=round(mean_ratio, 4),
            evidence=evidence,
            interpretation=interpretation,
            recommendation=recommendation,
        ))

    return flaws


# ── flat pitch ───────────────────────────────────────────────────────────

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


# ── score exclusion ──────────────────────────────────────────────────────

def _compute_excluded_indices(
    participant_words: list[WordTiming] | None,
    reference_words: list[WordTiming] | None,
) -> tuple[set[int], list[str]]:
    """Return indices of low-score words to exclude, and any warnings."""
    if participant_words is None or reference_words is None:
        return set(), []

    cfg = load_config("rules_v1")["word_comparison"]
    threshold = cfg["score_threshold"]
    max_fraction = cfg["max_excluded_fraction"]

    n_words = min(len(participant_words), len(reference_words))
    if n_words == 0:
        return set(), []

    excluded: set[int] = set()
    for i in range(n_words):
        p_score = participant_words[i].score
        r_score = reference_words[i].score
        if p_score < threshold or r_score < threshold:
            excluded.add(i)

    warnings: list[str] = []
    if n_words > 0 and len(excluded) / n_words > max_fraction:
        warnings.append(
            f"word_comparison: {len(excluded)}/{n_words} words "
            f"({len(excluded)/n_words:.0%}) have low alignment scores "
            f"(below {threshold}). The transcript may not match the audio. "
            f"Word-level rules are unavailable."
        )

    return excluded, warnings


# ── orchestrator ─────────────────────────────────────────────────────────

def detect_all_flaws(
    participant: FeatureBundle,
    reference: FeatureBundle,
    alignment_method: str = "energy_fallback",
    participant_words: list[WordTiming] | None = None,
    reference_words: list[WordTiming] | None = None,
) -> tuple[list[Flaw], list[str]]:
    flaws: list[Flaw] = []
    unavailable: list[str] = []

    excluded, score_warnings = _compute_excluded_indices(
        participant_words, reference_words,
    )
    too_many_excluded = len(score_warnings) > 0

    # Long pauses: matched when word timings available, unmatched otherwise
    if too_many_excluded:
        flaws.extend(detect_long_pauses(participant, reference, alignment_method))
    else:
        flaws.extend(detect_long_pauses(
            participant, reference, alignment_method,
            participant_words, reference_words, excluded,
        ))

    # Overall rate deviation (always recording-level)
    flaws.extend(detect_rate_deviation(participant, reference))

    # Local rate (only when word timings available and not too many excluded)
    if participant_words and reference_words and not too_many_excluded:
        flaws.extend(detect_local_rate(
            participant_words, reference_words, excluded,
        ))
    elif alignment_method == "energy_fallback":
        unavailable.append(
            "local_rate: no word-level timestamps (energy_fallback mode)"
        )

    if too_many_excluded:
        unavailable.extend(score_warnings)

    # Flat pitch (unchanged)
    pitch_flaws, pitch_unavailable = detect_flat_pitch(participant, reference)
    flaws.extend(pitch_flaws)
    unavailable.extend(pitch_unavailable)

    return flaws, unavailable
