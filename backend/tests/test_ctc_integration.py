"""CTC integration tests — skipped when torch is not installed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from speechlens.alignment.ctc import CTCAligner
from speechlens.audio.decode import decode_audio
from speechlens.features.extract import extract_features
from speechlens.flaws.rules import detect_all_flaws

DEV_FIXTURES = Path(__file__).resolve().parent.parent / "dev_fixtures"

pytestmark = pytest.mark.skipif(
    not (DEV_FIXTURES / "tts_ideal.wav").exists(),
    reason="dev fixtures not generated",
)


@pytest.fixture(scope="module")
def aligner():
    return CTCAligner()


@pytest.fixture(scope="module")
def transcript():
    return (DEV_FIXTURES / "passage.txt").read_text(encoding="utf-8").strip()


def _load_and_align(aligner, filename, transcript):
    decoded = decode_audio(DEV_FIXTURES / filename)
    duration = len(decoded.samples) / decoded.sample_rate
    feat = extract_features(decoded.samples, decoded.sample_rate, transcript, duration)
    alignment = aligner.align(decoded.samples, decoded.sample_rate, transcript, feat.vad)
    return feat, alignment


@pytest.fixture(scope="module")
def ideal(aligner, transcript):
    return _load_and_align(aligner, "tts_ideal.wav", transcript)


@pytest.fixture(scope="module")
def pauses(aligner, transcript):
    return _load_and_align(aligner, "tts_pauses.wav", transcript)


@pytest.fixture(scope="module")
def fast(aligner, transcript):
    return _load_and_align(aligner, "tts_fast.wav", transcript)


class TestIdealVsPauses:
    def test_three_long_pause_flaws(self, ideal, pauses):
        ref_feat, ref_align = ideal
        part_feat, part_align = pauses

        flaws, unavailable = detect_all_flaws(
            part_feat, ref_feat, part_align.method,
            part_align.words, ref_align.words,
        )
        pause_flaws = [f for f in flaws if f.rule_id == "long_pause"]
        assert len(pause_flaws) == 3, (
            f"Expected 3 long_pause flaws, got {len(pause_flaws)}"
        )

    def test_preceding_words(self, ideal, pauses):
        ref_feat, ref_align = ideal
        part_feat, part_align = pauses

        flaws, _ = detect_all_flaws(
            part_feat, ref_feat, part_align.method,
            part_align.words, ref_align.words,
        )
        pause_flaws = [f for f in flaws if f.rule_id == "long_pause"]
        preceding = [f.participant_measurement["preceding_word"] for f in pause_flaws]
        assert preceding == ["DEVELOP", "IMPRESSION", "POINTS"], (
            f"Expected DEVELOP, IMPRESSION, POINTS but got {preceding}"
        )

    def test_matched_mode_evidence(self, ideal, pauses):
        ref_feat, ref_align = ideal
        part_feat, part_align = pauses

        flaws, _ = detect_all_flaws(
            part_feat, ref_feat, part_align.method,
            part_align.words, ref_align.words,
        )
        pause_flaws = [f for f in flaws if f.rule_id == "long_pause"]
        for flaw in pause_flaws:
            assert "matched reference pause" in flaw.evidence
            assert "matched_pause_seconds" in flaw.reference_measurement


class TestIdealVsFast:
    def test_rate_deviation_fires(self, ideal, fast):
        ref_feat, ref_align = ideal
        part_feat, part_align = fast

        flaws, _ = detect_all_flaws(
            part_feat, ref_feat, part_align.method,
            part_align.words, ref_align.words,
        )
        rate_flaws = [f for f in flaws if f.rule_id == "rate_deviation"]
        assert len(rate_flaws) == 1
        assert rate_flaws[0].deviation > 1.3

    def test_print_all_flaws(self, ideal, fast, capsys):
        ref_feat, ref_align = ideal
        part_feat, part_align = fast

        flaws, unavailable = detect_all_flaws(
            part_feat, ref_feat, part_align.method,
            part_align.words, ref_align.words,
        )
        print(f"\n--- ideal vs fast: {len(flaws)} flaws ---")
        for f in flaws:
            print(f"  {f.rule_id} [{f.severity}] {f.start_seconds:.2f}-{f.end_seconds:.2f}s "
                  f"deviation={f.deviation:.2f}")
        if unavailable:
            print(f"  unavailable: {unavailable}")


class TestIdenticalRecordings:
    def test_no_flaws(self, ideal):
        feat, alignment = ideal

        flaws, unavailable = detect_all_flaws(
            feat, feat, alignment.method,
            alignment.words, alignment.words,
        )
        assert len(flaws) == 0, (
            f"Expected 0 flaws for identical recordings, got {len(flaws)}: "
            + ", ".join(f.rule_id for f in flaws)
        )


class TestWrongTranscript:
    def test_low_score_warning(self, aligner):
        wrong_transcript = (
            "The sun of dark communication is one of the most terrible "
            "things a banana can develop. Whether we are giving a purple "
            "elephant or a spaceship pizza, the ability to zebra our "
            "ideas clearly and penguin can make a lasting umbrella."
        )
        decoded = decode_audio(DEV_FIXTURES / "tts_ideal.wav")
        duration = len(decoded.samples) / decoded.sample_rate
        feat = extract_features(decoded.samples, decoded.sample_rate, wrong_transcript, duration)
        alignment = aligner.align(decoded.samples, decoded.sample_rate, wrong_transcript, feat.vad)

        assert alignment.words is not None
        scores = [w.score for w in alignment.words]
        low_count = sum(1 for s in scores if s < -1.0)
        assert low_count > 5, (
            f"Expected many low-score words with wrong transcript, got {low_count} below -1.0"
        )

        flaws, unavailable = detect_all_flaws(
            feat, feat, alignment.method,
            alignment.words, alignment.words,
        )
        score_warnings = [u for u in unavailable if "word_comparison" in u]
        assert len(score_warnings) == 1, (
            f"Expected a word_comparison warning, got: {unavailable}"
        )
        assert "transcript may not match" in score_warnings[0]
