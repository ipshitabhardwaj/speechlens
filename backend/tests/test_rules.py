"""Tests for speechlens.flaws.rules and speechlens.rubric — flaw detection and scoring."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from speechlens.config import load_config, clear_cache
from speechlens.features.extract import FeatureBundle, extract_features
from speechlens.flaws.rules import (
    detect_all_flaws,
    detect_flat_pitch,
    detect_long_pauses,
    detect_rate_deviation,
)
from speechlens.rubric.rubric import score_rubric

SR = load_config("features")["audio"]["analysis_sample_rate"]


def _sine(freq: float = 220.0, duration: float = 2.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration: float = 1.0) -> np.ndarray:
    return np.zeros(int(SR * duration), dtype=np.float32)


def _features(signal: np.ndarray, transcript: str = "one two three four five") -> FeatureBundle:
    duration = len(signal) / SR
    return extract_features(signal, SR, transcript, duration)


# ── Identical recordings ──────────────────────────────────────────────────

class TestIdenticalRecordings:
    def test_zero_flaws(self) -> None:
        sig = _sine(220.0, 3.0, 0.3)
        feat = _features(sig)
        flaws, unavailable = detect_all_flaws(feat, feat)
        assert len(flaws) == 0, f"Expected 0 flaws, got {len(flaws)}: {flaws}"

    def test_maximum_rubric_scores(self) -> None:
        sig = _sine(220.0, 3.0, 0.3)
        feat = _features(sig)
        flaws, unavailable = detect_all_flaws(feat, feat)
        rubric = score_rubric(feat, feat, flaws, unavailable)
        for dim in rubric.dimensions:
            if dim.available:
                assert dim.score == 100.0, (
                    f"Dimension {dim.name} score {dim.score}, expected 100"
                )
        assert rubric.overall_score == 100.0


# ── Gain invariance ───────────────────────────────────────────────────────

class TestGainInvariance:
    def test_half_gain_zero_flaws(self) -> None:
        sig = _sine(220.0, 3.0, 0.3)
        sig_half = sig * 0.5
        transcript = "one two three four five"
        ref_feat = _features(sig, transcript)
        part_feat = _features(sig_half, transcript)
        flaws, _ = detect_all_flaws(part_feat, ref_feat)
        assert len(flaws) == 0, f"Expected 0 flaws at 0.5x gain, got {len(flaws)}: {flaws}"


# ── Long pause detection ─────────────────────────────────────────────────

class TestLongPause:
    def test_inserted_silence_detected(self) -> None:
        """Reference = continuous speech. Participant = same speech with a 2 s
        silence inserted at a known time. Exactly one long_pause with boundaries
        within 25 ms of the insertion."""
        insert_time = 1.5
        silence_dur = 2.0
        transcript = "one two three four five"

        ref_sig = _sine(220.0, 3.0, 0.3)
        ref_feat = _features(ref_sig, transcript)

        speech1 = _sine(220.0, insert_time, 0.3)
        gap = _silence(silence_dur)
        speech2 = _sine(220.0, 3.0 - insert_time, 0.3)
        part_sig = np.concatenate([speech1, gap, speech2])
        part_feat = _features(part_sig, transcript)

        flaws, _ = detect_all_flaws(part_feat, ref_feat)
        pause_flaws = [f for f in flaws if f.rule_id == "long_pause"]

        assert len(pause_flaws) == 1, (
            f"Expected exactly 1 long_pause, got {len(pause_flaws)}"
        )

        flaw = pause_flaws[0]
        tolerance = 0.025  # 25 ms
        assert abs(flaw.start_seconds - insert_time) < tolerance, (
            f"Pause start {flaw.start_seconds:.4f}s, expected {insert_time:.4f}s ± {tolerance}s"
        )
        expected_end = insert_time + silence_dur
        assert abs(flaw.end_seconds - expected_end) < tolerance, (
            f"Pause end {flaw.end_seconds:.4f}s, expected {expected_end:.4f}s ± {tolerance}s"
        )

    def test_flaw_has_full_schema(self) -> None:
        ref_sig = _sine(220.0, 3.0, 0.3)
        ref_feat = _features(ref_sig)

        speech = _sine(220.0, 1.5, 0.3)
        gap = _silence(2.0)
        part_sig = np.concatenate([speech, gap, speech])
        part_feat = _features(part_sig)

        flaws, _ = detect_all_flaws(part_feat, ref_feat)
        pause_flaws = [f for f in flaws if f.rule_id == "long_pause"]
        assert len(pause_flaws) == 1
        flaw = pause_flaws[0]

        assert flaw.rule_id == "long_pause"
        assert flaw.category == "pause_control"
        assert flaw.severity in ("mild", "moderate", "severe")
        assert flaw.start_seconds > 0
        assert flaw.end_seconds > flaw.start_seconds
        assert "pause_duration_seconds" in flaw.participant_measurement
        assert "median_pause_seconds" in flaw.reference_measurement
        assert "p90_pause_seconds" in flaw.reference_measurement
        assert "pause_count" in flaw.reference_measurement
        assert flaw.deviation > 1.0
        assert "not matched one-to-one" in flaw.evidence
        assert len(flaw.interpretation) > 0
        assert len(flaw.recommendation) > 0

    def test_reference_pause_distribution_reported(self) -> None:
        ref_speech = _sine(220.0, 1.0, 0.3)
        ref_gap = _silence(0.4)
        ref_sig = np.concatenate([ref_speech, ref_gap, ref_speech])
        ref_feat = _features(ref_sig)

        part_speech = _sine(220.0, 1.0, 0.3)
        part_gap = _silence(2.0)
        part_sig = np.concatenate([part_speech, part_gap, part_speech])
        part_feat = _features(part_sig)

        flaws, _ = detect_all_flaws(part_feat, ref_feat)
        pause_flaws = [f for f in flaws if f.rule_id == "long_pause"]
        assert len(pause_flaws) == 1
        ref_meas = pause_flaws[0].reference_measurement
        assert ref_meas["pause_count"] >= 1


# ── Rate deviation ────────────────────────────────────────────────────────

class TestRateDeviation:
    def test_fast_rate_detected(self) -> None:
        """Participant signal is half the duration of reference with the same
        transcript, giving 2x articulation rate — well beyond the 1.3 upper bound."""
        transcript = "one two three four five"
        ref_sig = _sine(220.0, 4.0, 0.3)
        part_sig = _sine(220.0, 2.0, 0.3)

        ref_feat = _features(ref_sig, transcript)
        part_feat = _features(part_sig, transcript)

        flaws, _ = detect_all_flaws(part_feat, ref_feat)
        rate_flaws = [f for f in flaws if f.rule_id == "rate_deviation"]

        assert len(rate_flaws) == 1
        flaw = rate_flaws[0]
        assert flaw.scope == "recording"

        expected_ratio = (
            part_feat.rate.articulation_rate_wpm / ref_feat.rate.articulation_rate_wpm
        )
        assert abs(flaw.deviation - expected_ratio) < 0.01, (
            f"Deviation {flaw.deviation}, expected ratio {expected_ratio:.2f}"
        )

    def test_rate_flaw_has_correct_ratio(self) -> None:
        transcript = "one two three four five"
        ref_sig = _sine(220.0, 5.0, 0.3)
        part_sig = _sine(220.0, 2.0, 0.3)

        ref_feat = _features(ref_sig, transcript)
        part_feat = _features(part_sig, transcript)

        ratio = part_feat.rate.articulation_rate_wpm / ref_feat.rate.articulation_rate_wpm
        assert ratio > 1.3

        flaws, _ = detect_all_flaws(part_feat, ref_feat)
        rate_flaws = [f for f in flaws if f.rule_id == "rate_deviation"]
        assert len(rate_flaws) == 1
        assert abs(rate_flaws[0].deviation - ratio) < 0.01


# ── Flat pitch ────────────────────────────────────────────────────────────

class TestFlatPitch:
    def test_too_few_voiced_frames_unavailable(self) -> None:
        """When either recording has too few voiced frames, flat_pitch is
        unavailable rather than flagged."""
        rng = np.random.default_rng(99)
        noise = (rng.standard_normal(SR * 2) * 0.05).astype(np.float32)
        tone = _sine(220.0, 3.0, 0.3)

        noise_feat = _features(noise)
        tone_feat = _features(tone)

        flaws, unavailable = detect_all_flaws(noise_feat, tone_feat)
        pitch_flaws = [f for f in flaws if f.rule_id == "flat_pitch"]
        assert len(pitch_flaws) == 0

        pitch_unavailable = [u for u in unavailable if u.startswith("flat_pitch:")]
        assert len(pitch_unavailable) == 1
        assert "too few voiced frames" in pitch_unavailable[0]

    def test_unavailable_not_flagged(self) -> None:
        """Unavailable means no flaw, not a flaw with unavailable severity."""
        silence = _silence(2.0)
        tone = _sine(220.0, 3.0, 0.3)

        silence_feat = _features(silence)
        tone_feat = _features(tone)

        flaws, unavailable = detect_flat_pitch(silence_feat, tone_feat)
        assert len(flaws) == 0
        assert any("flat_pitch" in u for u in unavailable)


# ── Rubric ────────────────────────────────────────────────────────────────

class TestRubric:
    def test_determinism(self) -> None:
        sig = _sine(220.0, 3.0, 0.3)
        feat = _features(sig)
        flaws, unavailable = detect_all_flaws(feat, feat)

        r1 = score_rubric(feat, feat, flaws, unavailable)
        r2 = score_rubric(feat, feat, flaws, unavailable)

        assert r1.overall_score == r2.overall_score
        for d1, d2 in zip(r1.dimensions, r2.dimensions):
            assert d1.score == d2.score
            assert d1.available == d2.available

    def test_yaml_threshold_changes_outcome(self, tmp_path, monkeypatch) -> None:
        """Modifying a YAML threshold changes the flaw/rubric outcome without
        any code change."""
        import speechlens.config as config_mod

        config_dir = Path(config_mod._CONFIG_DIR)
        for f in config_dir.iterdir():
            if f.is_file():
                shutil.copy(f, tmp_path / f.name)

        ref_sig = _sine(220.0, 3.0, 0.3)
        speech = _sine(220.0, 1.0, 0.3)
        gap = _silence(2.0)
        part_sig = np.concatenate([speech, gap, speech])
        transcript = "one two three four five"

        ref_feat = _features(ref_sig, transcript)
        part_feat = _features(part_sig, transcript)

        # Default config: 2 s pause > 1.5 s floor → flaw
        flaws_default, _ = detect_all_flaws(part_feat, ref_feat)
        pause_default = [f for f in flaws_default if f.rule_id == "long_pause"]
        assert len(pause_default) == 1

        # Raise the floor to 999 s → same pause is no longer a flaw
        rules_path = tmp_path / "rules_v1.yaml"
        with open(rules_path) as f:
            rules = yaml.safe_load(f)
        rules["long_pause"]["absolute_floor_seconds"] = 999.0
        with open(rules_path, "w") as f:
            yaml.dump(rules, f)

        monkeypatch.setattr(config_mod, "_CONFIG_DIR", tmp_path)
        clear_cache()

        flaws_modified, _ = detect_all_flaws(part_feat, ref_feat)
        pause_modified = [f for f in flaws_modified if f.rule_id == "long_pause"]
        assert len(pause_modified) == 0

        # Rubric scores differ
        rubric_default = score_rubric(part_feat, ref_feat, flaws_default, [])
        rubric_modified = score_rubric(part_feat, ref_feat, flaws_modified, [])

        pc_default = next(d for d in rubric_default.dimensions if d.name == "pause_control")
        pc_modified = next(d for d in rubric_modified.dimensions if d.name == "pause_control")
        assert pc_modified.score > pc_default.score

    def test_energy_fallback_local_rate_unavailable(self) -> None:
        """When using energy_fallback, local_rate is listed as unavailable."""
        sig = _sine(220.0, 3.0, 0.3)
        feat = _features(sig)
        flaws, unavailable = detect_all_flaws(feat, feat, "energy_fallback")
        lr_unavailable = [u for u in unavailable if u.startswith("local_rate:")]
        assert len(lr_unavailable) == 1
        assert "energy_fallback" in lr_unavailable[0]

    def test_unavailable_dimension_excluded(self) -> None:
        """When vocal_variation is unavailable, it is excluded from the overall
        score and weights are renormalised."""
        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(SR * 2) * 0.05).astype(np.float32)
        tone = _sine(220.0, 3.0, 0.3)

        noise_feat = _features(noise)
        tone_feat = _features(tone)

        flaws, unavailable = detect_all_flaws(noise_feat, tone_feat)
        rubric = score_rubric(noise_feat, tone_feat, flaws, unavailable)

        vv = next(d for d in rubric.dimensions if d.name == "vocal_variation")
        assert not vv.available
        assert vv.unavailable_reason is not None

        available_dims = [d for d in rubric.dimensions if d.available]
        assert len(available_dims) < len(rubric.dimensions)

        if available_dims:
            total_weight = sum(d.weight for d in available_dims)
            expected_overall = sum(d.score * d.weight for d in available_dims) / total_weight
            assert abs(rubric.overall_score - expected_overall) < 0.01
