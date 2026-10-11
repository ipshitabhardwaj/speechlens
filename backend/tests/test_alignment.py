"""Tests for speechlens.alignment — adapter protocol and energy_fallback."""
from __future__ import annotations

import numpy as np

from speechlens.alignment import Aligner, AlignmentResult, EnergyFallbackAligner
from speechlens.config import load_config
from speechlens.features.extract import (
    PauseInterval,
    VADResult,
    compute_frame_times,
    compute_rms,
    compute_vad,
)

SR = load_config("features")["audio"]["analysis_sample_rate"]


def _sine(freq: float = 220.0, duration: float = 2.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration, int(SR * duration), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_vad(samples: np.ndarray) -> VADResult:
    ft = compute_frame_times(len(samples), SR)
    rms = compute_rms(samples, ft).rms
    return compute_vad(rms, ft)


class TestEnergyFallbackAligner:
    def test_conforms_to_protocol(self) -> None:
        aligner = EnergyFallbackAligner()
        assert isinstance(aligner, Aligner)

    def test_method_name(self) -> None:
        samples = _sine(duration=2.0)
        vad = _make_vad(samples)
        result = EnergyFallbackAligner().align(samples, SR, "hello world", vad)
        assert result.method == "energy_fallback"

    def test_words_always_none(self) -> None:
        samples = _sine(duration=2.0)
        vad = _make_vad(samples)
        result = EnergyFallbackAligner().align(samples, SR, "hello world", vad)
        assert result.words is None

    def test_no_code_path_returns_word_timings(self) -> None:
        """Exhaustive check: no input combination produces word timings."""
        aligner = EnergyFallbackAligner()

        for samples in [_sine(duration=2.0), np.zeros(SR * 2, dtype=np.float32)]:
            vad = _make_vad(samples)
            for transcript in ["", "hello", "one two three four five six seven"]:
                result = aligner.align(samples, SR, transcript, vad)
                assert result.words is None, (
                    f"energy_fallback returned word timings for transcript={transcript!r}"
                )

    def test_unavailable_reports_word_timestamps(self) -> None:
        samples = _sine(duration=2.0)
        vad = _make_vad(samples)
        result = EnergyFallbackAligner().align(samples, SR, "test", vad)
        assert len(result.unavailable) == 1
        assert "word_timestamps" in result.unavailable[0]

    def test_pauses_forwarded_from_vad(self) -> None:
        speech = _sine(220.0, 1.0, 0.3)
        gap = np.zeros(int(SR * 0.6), dtype=np.float32)
        sig = np.concatenate([speech, gap, speech])
        vad = _make_vad(sig)
        assert len(vad.pauses) >= 1

        result = EnergyFallbackAligner().align(sig, SR, "hello world", vad)
        assert len(result.pauses) == len(vad.pauses)
        for rp, vp in zip(result.pauses, vad.pauses):
            assert rp.start == vp.start
            assert rp.end == vp.end

    def test_returns_alignment_result(self) -> None:
        samples = _sine(duration=2.0)
        vad = _make_vad(samples)
        result = EnergyFallbackAligner().align(samples, SR, "test", vad)
        assert isinstance(result, AlignmentResult)
