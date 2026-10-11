"""Tests for speechlens.features.extract — RMS, VAD, F0, rate, timing, and determinism."""
from __future__ import annotations

import time

import numpy as np
import pytest

from speechlens.config import load_config
from speechlens.features.extract import (
    FeatureBundle,
    compute_f0,
    compute_frame_times,
    compute_rate,
    compute_rms,
    compute_vad,
    extract_features,
)

SR = load_config("features")["audio"]["analysis_sample_rate"]  # 16000
HOP = load_config("features")["framing"]["hop_length"]         # 160 (10 ms)
RMS_FRAME = load_config("features")["rms"]["frame_length"]     # 400 (25 ms)
F0_FRAME = load_config("features")["f0"]["frame_length"]       # 1024 (64 ms)


def _sine(freq: float = 440.0, duration: float = 2.0, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration: float = 1.0, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration), dtype=np.float32)


def _speech_silence_speech(
    speech_dur: float = 1.0,
    silence_dur: float = 0.5,
    freq: float = 220.0,
    amplitude: float = 0.3,
    noise_amplitude: float = 0.0,
    sr: int = SR,
) -> np.ndarray:
    speech1 = _sine(freq, speech_dur, sr, amplitude)
    if noise_amplitude > 0:
        gap = (np.random.default_rng(42).standard_normal(int(sr * silence_dur)) * noise_amplitude).astype(np.float32)
    else:
        gap = _silence(silence_dur, sr)
    speech2 = _sine(freq, speech_dur, sr, amplitude)
    return np.concatenate([speech1, gap, speech2])


# ── Time axis ──────────────────────────────────────────────────────────────

class TestFrameTimes:
    def test_frame_times_seconds(self) -> None:
        n = SR * 2
        ft = compute_frame_times(n, SR)
        assert ft.times[0] == 0.0
        expected_step = HOP / SR
        actual_step = float(ft.times[1] - ft.times[0])
        assert abs(actual_step - expected_step) < 1e-9

    def test_rms_and_pyin_frame_counts_match(self) -> None:
        """RMS and pYIN both produce the same number of frames as compute_frame_times.

        With librosa centre=True (the default), frame count = 1 + n_samples // hop,
        independent of frame_length. This test verifies that RMS (frame_length=400)
        and pYIN (frame_length=1024) agree.
        """
        import librosa
        for duration in [1.0, 2.0, 3.7]:
            n_samples = int(SR * duration)
            samples = _sine(freq=220.0, duration=duration, amplitude=0.3)
            ft = compute_frame_times(n_samples, SR)
            expected_n = len(ft.times)

            rms = librosa.feature.rms(
                y=samples, frame_length=RMS_FRAME, hop_length=HOP,
            )[0]
            f0, _, _ = librosa.pyin(
                y=samples.astype(np.float64), fmin=65.0, fmax=400.0,
                sr=SR, frame_length=F0_FRAME, hop_length=HOP,
            )
            assert len(rms) == expected_n, (
                f"dur={duration}: RMS frames {len(rms)} != expected {expected_n}"
            )
            assert len(f0) == expected_n, (
                f"dur={duration}: pYIN frames {len(f0)} != expected {expected_n}"
            )

    def test_step_onset_within_one_hop(self) -> None:
        """A tone starting at a known time: the frame where RMS first reaches
        half its steady-state power is within one hop (10 ms) of the true onset.

        The old impulse test failed because a single-sample impulse spreads its
        energy across every frame whose centre-padded window covers it — with a
        128 ms window that smeared the peak by up to 64 ms. A step onset is the
        correct test: the 25 ms RMS window transitions cleanly at the boundary.
        """
        onset_time = 0.5
        duration = 2.0
        samples = np.zeros(int(SR * duration), dtype=np.float32)
        onset_sample = int(onset_time * SR)
        tone = _sine(freq=440.0, duration=duration - onset_time, amplitude=0.3)
        samples[onset_sample:onset_sample + len(tone)] = tone

        ft = compute_frame_times(len(samples), SR)
        rms = compute_rms(samples, ft)

        steady_rms = float(np.median(rms.rms[-100:]))
        half_power = steady_rms * 0.5

        onset_frame = int(np.argmax(rms.rms >= half_power))
        detected_onset = float(ft.times[onset_frame])
        tolerance = HOP / SR  # 10 ms
        assert abs(detected_onset - onset_time) <= tolerance, (
            f"Onset at {detected_onset:.4f}s, expected {onset_time:.4f}s ± {tolerance:.4f}s"
        )


# ── RMS ────────────────────────────────────────────────────────────────────

class TestRMS:
    def test_sine_rms_equals_a_over_sqrt2(self) -> None:
        A = 0.4
        samples = _sine(freq=440.0, duration=2.0, amplitude=A)
        ft = compute_frame_times(len(samples), SR)
        result = compute_rms(samples, ft)

        expected_rms = A / np.sqrt(2)
        mid = len(result.rms) // 4
        end = 3 * len(result.rms) // 4
        mean_rms = float(np.mean(result.rms[mid:end]))
        assert abs(mean_rms - expected_rms) / expected_rms < 0.01, (
            f"Mean RMS {mean_rms:.6f}, expected {expected_rms:.6f}"
        )

    def test_rms_db_relative_to_median_speech(self) -> None:
        samples = _sine(duration=2.0, amplitude=0.3)
        ft = compute_frame_times(len(samples), SR)
        vad = compute_vad(compute_rms(samples, ft).rms, ft)
        result = compute_rms(samples, ft, speech_mask=vad.is_speech)

        assert result.median_speech_rms_db == 0.0
        mid = len(result.rms_db) // 4
        end = 3 * len(result.rms_db) // 4
        assert all(abs(v) < 1.0 for v in result.rms_db[mid:end])


# ── VAD and pauses ─────────────────────────────────────────────────────────

class TestVAD:
    def test_gain_invariant_pause_detection(self) -> None:
        sig_full = _speech_silence_speech(speech_dur=1.0, silence_dur=0.5, amplitude=0.3)
        sig_half = sig_full * 0.5

        ft_full = compute_frame_times(len(sig_full), SR)
        ft_half = compute_frame_times(len(sig_half), SR)
        rms_full = compute_rms(sig_full, ft_full).rms
        rms_half = compute_rms(sig_half, ft_half).rms
        vad_full = compute_vad(rms_full, ft_full)
        vad_half = compute_vad(rms_half, ft_half)

        assert len(vad_full.pauses) == len(vad_half.pauses)
        for p_full, p_half in zip(vad_full.pauses, vad_half.pauses):
            assert abs(p_full.start - p_half.start) < 0.02
            assert abs(p_full.end - p_half.end) < 0.02

    def test_known_silence_boundaries(self) -> None:
        """Detected boundaries within 25 ms of ground truth.

        Tolerance is 25 ms: half the 25 ms RMS window (12.5 ms edge smear)
        plus one 10 ms hop for frame quantisation = 22.5 ms, rounded to 25 ms.
        """
        speech_dur = 1.0
        silence_dur = 0.6
        sig = _speech_silence_speech(speech_dur=speech_dur, silence_dur=silence_dur)

        ft = compute_frame_times(len(sig), SR)
        rms = compute_rms(sig, ft).rms
        vad = compute_vad(rms, ft)

        assert len(vad.pauses) == 1, f"Expected 1 pause, got {len(vad.pauses)}"
        pause = vad.pauses[0]
        expected_start = speech_dur
        expected_end = speech_dur + silence_dur
        tolerance_ms = 25
        tolerance_s = tolerance_ms / 1000
        assert abs(pause.start - expected_start) < tolerance_s, (
            f"Pause start {pause.start:.4f}s, expected {expected_start:.4f}s ± {tolerance_ms}ms"
        )
        assert abs(pause.end - expected_end) < tolerance_s, (
            f"Pause end {pause.end:.4f}s, expected {expected_end:.4f}s ± {tolerance_ms}ms"
        )

    def test_works_with_noise_in_gaps(self) -> None:
        sig = _speech_silence_speech(
            speech_dur=1.0,
            silence_dur=0.6,
            amplitude=0.3,
            noise_amplitude=0.005,
        )
        ft = compute_frame_times(len(sig), SR)
        rms = compute_rms(sig, ft).rms
        vad = compute_vad(rms, ft)
        assert len(vad.pauses) == 1

    def test_min_pause_duration_configurable(self) -> None:
        min_pause = load_config("features")["vad"]["min_pause_duration_seconds"]
        sig = _speech_silence_speech(speech_dur=1.0, silence_dur=min_pause * 0.5)
        ft = compute_frame_times(len(sig), SR)
        rms = compute_rms(sig, ft).rms
        vad = compute_vad(rms, ft)
        assert len(vad.pauses) == 0

    def test_leading_trailing_silence_not_pauses(self) -> None:
        leading = _silence(0.5)
        speech = _sine(220.0, 1.5, amplitude=0.3)
        trailing = _silence(0.3)
        sig = np.concatenate([leading, speech, trailing])

        ft = compute_frame_times(len(sig), SR)
        rms = compute_rms(sig, ft).rms
        vad = compute_vad(rms, ft)

        assert vad.leading_silence is not None
        assert abs(vad.leading_silence.start) < 0.01
        assert abs(vad.leading_silence.end - 0.5) < 0.03

        assert vad.trailing_silence is not None
        assert len(vad.pauses) == 0


# ── F0 (pYIN) ─────────────────────────────────────────────────────────────

class TestF0:
    def test_220hz_tone(self) -> None:
        samples = _sine(freq=220.0, duration=2.0, amplitude=0.3)
        ft = compute_frame_times(len(samples), SR)
        vad = compute_vad(compute_rms(samples, ft).rms, ft)
        result = compute_f0(samples, ft, vad_speech=vad.is_speech)
        assert abs(result.median_f0_hz - 220.0) / 220.0 < 0.01, (
            f"Median F0 {result.median_f0_hz:.2f} Hz, expected 220 Hz ± 1%"
        )

    def test_110hz_tone(self) -> None:
        samples = _sine(freq=110.0, duration=2.0, amplitude=0.3)
        ft = compute_frame_times(len(samples), SR)
        vad = compute_vad(compute_rms(samples, ft).rms, ft)
        result = compute_f0(samples, ft, vad_speech=vad.is_speech)
        assert abs(result.median_f0_hz - 110.0) / 110.0 < 0.01, (
            f"Median F0 {result.median_f0_hz:.2f} Hz, expected 110 Hz ± 1%"
        )

    def test_330hz_tone(self) -> None:
        samples = _sine(freq=330.0, duration=2.0, amplitude=0.3)
        ft = compute_frame_times(len(samples), SR)
        vad = compute_vad(compute_rms(samples, ft).rms, ft)
        result = compute_f0(samples, ft, vad_speech=vad.is_speech)
        assert abs(result.median_f0_hz - 330.0) / 330.0 < 0.01, (
            f"Median F0 {result.median_f0_hz:.2f} Hz, expected 330 Hz ± 1%"
        )

    def test_unvoiced_frames_are_nan(self) -> None:
        samples = _sine(freq=220.0, duration=2.0, amplitude=0.3)
        ft = compute_frame_times(len(samples), SR)
        vad = compute_vad(compute_rms(samples, ft).rms, ft)
        result = compute_f0(samples, ft, vad_speech=vad.is_speech)

        unvoiced = ~result.voiced_flag
        if unvoiced.any():
            assert np.all(np.isnan(result.f0[unvoiced]))
            assert not np.any(result.f0[unvoiced] == 0.0)

    def test_silence_gives_exactly_zero_voiced_frames(self) -> None:
        """Pure silence: VAD marks all frames as non-speech, so voiced_fraction == 0."""
        samples = _silence(2.0)
        ft = compute_frame_times(len(samples), SR)
        vad = compute_vad(compute_rms(samples, ft).rms, ft)
        result = compute_f0(samples, ft, vad_speech=vad.is_speech)

        assert result.voiced_fraction == 0.0
        assert int(result.voiced_flag.sum()) == 0

    def test_white_noise_voiced_fraction(self) -> None:
        """White noise: report and check voiced fraction with VAD gating."""
        rng = np.random.default_rng(42)
        samples = (rng.standard_normal(SR * 2) * 0.1).astype(np.float32)
        ft = compute_frame_times(len(samples), SR)

        import librosa
        cfg = load_config("features")["f0"]
        _, raw_voiced, raw_prob = librosa.pyin(
            y=samples.astype(np.float64),
            fmin=cfg["fmin"], fmax=cfg["fmax"],
            sr=SR, frame_length=cfg["frame_length"], hop_length=HOP,
        )
        print(f"\nWhite noise raw pYIN: {raw_voiced.sum()}/{len(raw_voiced)} voiced frames")
        print(f"  prob distribution: min={raw_prob.min():.3f} p25={np.percentile(raw_prob, 25):.3f} "
              f"median={np.median(raw_prob):.3f} p75={np.percentile(raw_prob, 75):.3f} max={raw_prob.max():.3f}")

        vad = compute_vad(compute_rms(samples, ft).rms, ft)
        result = compute_f0(samples, ft, vad_speech=vad.is_speech)

        print(f"  After VAD gating: {result.voiced_flag.sum()}/{len(result.voiced_flag)} voiced, "
              f"fraction={result.voiced_fraction:.4f}")

        assert result.voiced_fraction < 0.05, (
            f"White noise voiced fraction {result.voiced_fraction:.4f} exceeds 5%"
        )

    def test_voiced_fraction_reported(self) -> None:
        samples = _sine(freq=440.0, duration=2.0, amplitude=0.3)
        ft = compute_frame_times(len(samples), SR)
        vad = compute_vad(compute_rms(samples, ft).rms, ft)
        result = compute_f0(samples, ft, vad_speech=vad.is_speech)

        assert 0.0 <= result.voiced_fraction <= 1.0
        assert result.voiced_fraction > 0.5

    def test_pyin_wall_clock_60s(self) -> None:
        samples = _sine(freq=220.0, duration=60.0, amplitude=0.3)
        ft = compute_frame_times(len(samples), SR)
        vad = compute_vad(compute_rms(samples, ft).rms, ft)

        start = time.perf_counter()
        compute_f0(samples, ft, vad_speech=vad.is_speech)
        elapsed = time.perf_counter() - start

        print(f"\npYIN wall-clock time for 60s audio (hop={HOP}, frame={F0_FRAME}): {elapsed:.2f}s")
        assert elapsed < 120, f"pYIN took {elapsed:.1f}s for 60s audio"


# ── Rate ───────────────────────────────────────────────────────────────────

class TestRate:
    def test_speaking_and_articulation_rate(self) -> None:
        leading = _silence(0.3)
        speech1 = _sine(220.0, 1.0, amplitude=0.3)
        gap = _silence(0.5)
        speech2 = _sine(220.0, 1.0, amplitude=0.3)
        trailing = _silence(0.2)
        sig = np.concatenate([leading, speech1, gap, speech2, trailing])
        duration = len(sig) / SR
        transcript = "the quick brown fox jumps over the lazy dog"

        ft = compute_frame_times(len(sig), SR)
        rms = compute_rms(sig, ft).rms
        vad = compute_vad(rms, ft)
        rate = compute_rate(transcript, duration, vad)

        assert rate.word_count == 9
        assert rate.speaking_rate_wpm > 0
        assert rate.articulation_rate_wpm > 0
        assert rate.articulation_rate_wpm > rate.speaking_rate_wpm

    def test_rate_definition(self) -> None:
        transcript = "one two three four five"
        duration = 10.0
        vad = compute_vad(
            np.ones(100, dtype=np.float32),
            compute_frame_times(SR * 10, SR),
        )
        rate = compute_rate(transcript, duration, vad)
        assert rate.word_count == 5


# ── Determinism ────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_identical_output_on_repeat(self) -> None:
        samples = _speech_silence_speech(speech_dur=1.0, silence_dur=0.4, amplitude=0.3)
        transcript = "hello world test"
        duration = len(samples) / SR

        r1 = extract_features(samples, SR, transcript, duration)
        r2 = extract_features(samples, SR, transcript, duration)

        np.testing.assert_array_equal(r1.rms.rms, r2.rms.rms)
        np.testing.assert_array_equal(r1.rms.rms_db, r2.rms.rms_db)
        np.testing.assert_array_equal(r1.vad.is_speech, r2.vad.is_speech)
        assert len(r1.vad.pauses) == len(r2.vad.pauses)
        for p1, p2 in zip(r1.vad.pauses, r2.vad.pauses):
            assert p1.start == p2.start
            assert p1.end == p2.end
        np.testing.assert_array_equal(r1.f0.f0, r2.f0.f0)
        np.testing.assert_array_equal(r1.f0.voiced_flag, r2.f0.voiced_flag)
        assert r1.rate.speaking_rate_wpm == r2.rate.speaking_rate_wpm
        assert r1.rate.articulation_rate_wpm == r2.rate.articulation_rate_wpm
