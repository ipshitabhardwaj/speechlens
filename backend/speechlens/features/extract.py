"""Frame-level feature extraction: RMS, VAD, F0, and recording-level rates.

Frame centring convention
-------------------------
Every frame-level array is indexed by frame number. The time of frame i is:
    t[i] = i * hop_length / sample_rate
This matches librosa's default centre=True padding, where the analysis window
is centred at that time. All features share the same hop_length (10 ms) so
their time axes are aligned. RMS/VAD use a 25 ms window; pYIN uses a 64 ms
window for better low-frequency resolution.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray

from speechlens.config import load_config


@dataclass(frozen=True)
class FrameTimes:
    """Time axis shared by all frame-level features."""
    times: NDArray[np.float64]
    hop_length: int
    sample_rate: int


@dataclass(frozen=True)
class RMSResult:
    rms: NDArray[np.float32]
    rms_db: NDArray[np.float32]
    median_speech_rms: float
    median_speech_rms_db: float


@dataclass(frozen=True)
class PauseInterval:
    start: float
    end: float


@dataclass(frozen=True)
class VADResult:
    is_speech: NDArray[np.bool_]
    pauses: list[PauseInterval]
    leading_silence: PauseInterval | None
    trailing_silence: PauseInterval | None
    speech_time_seconds: float
    total_pause_time_seconds: float


@dataclass(frozen=True)
class F0Result:
    f0: NDArray[np.float64]
    voiced_flag: NDArray[np.bool_]
    voiced_probability: NDArray[np.float64]
    voiced_fraction: float
    median_f0_hz: float


@dataclass(frozen=True)
class RateResult:
    """Recording-level speaking rates.

    speaking_rate: transcript words / (duration - leading silence - trailing silence),
        in words per minute. Measures overall pace including pauses.
    articulation_rate: transcript words / speech time excluding all pauses,
        in words per minute. Measures pace of actual speech production.
    Both are recording-level only. Local (per-word) rate requires word
    timestamps from the CTC aligner and comes in Phase 2.
    """
    speaking_rate_wpm: float
    articulation_rate_wpm: float
    word_count: int
    total_duration_seconds: float
    speech_time_seconds: float


@dataclass
class FeatureBundle:
    frame_times: FrameTimes
    rms: RMSResult
    vad: VADResult
    f0: F0Result
    rate: RateResult | None = None


def compute_frame_times(n_samples: int, sample_rate: int) -> FrameTimes:
    cfg = load_config("features")
    hop = cfg["framing"]["hop_length"]
    # With librosa's default centre=True, frame count is 1 + n_samples // hop
    # regardless of frame_length. This matches both RMS and pYIN.
    n_frames = 1 + n_samples // hop
    times = np.arange(n_frames) * hop / sample_rate
    return FrameTimes(times=times, hop_length=hop, sample_rate=sample_rate)


def compute_rms(
    samples: NDArray[np.float32],
    frame_times: FrameTimes,
    speech_mask: NDArray[np.bool_] | None = None,
) -> RMSResult:
    cfg = load_config("features")["rms"]
    frame_length = cfg["frame_length"]

    rms = librosa.feature.rms(
        y=samples,
        frame_length=frame_length,
        hop_length=frame_times.hop_length,
    )[0].astype(np.float32)

    n = len(frame_times.times)
    if len(rms) > n:
        rms = rms[:n]

    eps = 1e-10
    if speech_mask is not None and speech_mask.any():
        mask = speech_mask[:len(rms)]
        median_speech_rms = float(np.median(rms[mask]))
    else:
        median_speech_rms = float(np.median(rms))

    ref = max(median_speech_rms, eps)
    rms_db = (20 * np.log10(np.maximum(rms, eps) / ref)).astype(np.float32)

    return RMSResult(
        rms=rms,
        rms_db=rms_db,
        median_speech_rms=median_speech_rms,
        median_speech_rms_db=0.0,
    )


def compute_vad(rms: NDArray[np.float32], frame_times: FrameTimes) -> VADResult:
    cfg = load_config("features")["vad"]
    threshold_db = cfg["threshold_db"]
    min_pause_dur = cfg["min_pause_duration_seconds"]

    eps = 1e-10
    median_rms = float(np.median(rms))
    ref = max(median_rms, eps)
    rms_db_vs_median = 20 * np.log10(np.maximum(rms, eps) / ref)

    is_speech = rms_db_vs_median > threshold_db

    times = frame_times.times
    hop_sec = frame_times.hop_length / frame_times.sample_rate
    duration = float(times[-1] + hop_sec) if len(times) > 0 else 0.0

    speech_indices = np.where(is_speech)[0]
    if len(speech_indices) == 0:
        return VADResult(
            is_speech=is_speech,
            pauses=[],
            leading_silence=PauseInterval(0.0, duration) if duration > 0 else None,
            trailing_silence=None,
            speech_time_seconds=0.0,
            total_pause_time_seconds=0.0,
        )

    first_speech = int(speech_indices[0])
    last_speech = int(speech_indices[-1])

    leading = None
    if first_speech > 0:
        leading = PauseInterval(0.0, float(times[first_speech]))

    trailing = None
    trailing_start = float(times[last_speech] + hop_sec)
    if trailing_start < duration:
        trailing = PauseInterval(trailing_start, duration)

    pauses: list[PauseInterval] = []
    silence_regions = np.where(~is_speech[first_speech:last_speech + 1])[0] + first_speech
    if len(silence_regions) > 0:
        breaks = np.where(np.diff(silence_regions) > 1)[0]
        segments = np.split(silence_regions, breaks + 1)
        for seg in segments:
            start_t = float(times[seg[0]])
            end_t = float(times[seg[-1]] + hop_sec)
            if (end_t - start_t) >= min_pause_dur:
                pauses.append(PauseInterval(start_t, end_t))

    speech_frame_count = int(is_speech[first_speech:last_speech + 1].sum())
    speech_time = speech_frame_count * hop_sec
    total_pause_time = sum(p.end - p.start for p in pauses)

    return VADResult(
        is_speech=is_speech,
        pauses=pauses,
        leading_silence=leading,
        trailing_silence=trailing,
        speech_time_seconds=speech_time,
        total_pause_time_seconds=total_pause_time,
    )


def compute_f0(
    samples: NDArray[np.float32],
    frame_times: FrameTimes,
    vad_speech: NDArray[np.bool_] | None = None,
) -> F0Result:
    cfg = load_config("features")["f0"]
    fmin = cfg["fmin"]
    fmax = cfg["fmax"]
    frame_length = cfg["frame_length"]

    # voiced_probability is a raw pYIN diagnostic, not a calibrated confidence.
    # It is kept in the output but not used for filtering.
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=samples.astype(np.float64),
        fmin=fmin,
        fmax=fmax,
        sr=frame_times.sample_rate,
        frame_length=frame_length,
        hop_length=frame_times.hop_length,
    )

    n = len(frame_times.times)
    if len(f0) > n:
        f0 = f0[:n]
        voiced_flag = voiced_flag[:n]
        voiced_prob = voiced_prob[:n]
    elif len(f0) < n:
        pad = n - len(f0)
        f0 = np.concatenate([f0, np.full(pad, np.nan)])
        voiced_flag = np.concatenate([voiced_flag, np.zeros(pad, dtype=bool)])
        voiced_prob = np.concatenate([voiced_prob, np.zeros(pad)])

    if vad_speech is not None:
        vad_mask = vad_speech[:len(voiced_flag)]
        if len(vad_mask) < len(voiced_flag):
            vad_mask = np.concatenate([vad_mask, np.zeros(len(voiced_flag) - len(vad_mask), dtype=bool)])
        not_speech = ~vad_mask
        voiced_flag = voiced_flag & vad_mask
        f0[not_speech] = np.nan

    voiced_fraction = float(voiced_flag.sum() / len(voiced_flag)) if len(voiced_flag) > 0 else 0.0
    voiced_f0 = f0[voiced_flag]
    median_f0 = float(np.median(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    return F0Result(
        f0=f0,
        voiced_flag=voiced_flag,
        voiced_probability=voiced_prob,
        voiced_fraction=voiced_fraction,
        median_f0_hz=median_f0,
    )


def compute_rate(transcript: str, duration_seconds: float, vad: VADResult) -> RateResult:
    """Compute recording-level speaking and articulation rates.

    speaking_rate = word_count / effective_duration * 60
        where effective_duration = total_duration - leading_silence - trailing_silence
    articulation_rate = word_count / speech_time * 60
        where speech_time excludes all detected pauses

    Both in words per minute. Recording-level only; local rate needs word
    timestamps from the CTC aligner (Phase 2).
    """
    words = transcript.strip().split()
    word_count = len(words)

    leading_dur = (vad.leading_silence.end - vad.leading_silence.start) if vad.leading_silence else 0.0
    trailing_dur = (vad.trailing_silence.end - vad.trailing_silence.start) if vad.trailing_silence else 0.0
    effective_duration = duration_seconds - leading_dur - trailing_dur

    speaking_rate = (word_count / effective_duration * 60) if effective_duration > 0 else 0.0
    articulation_rate = (word_count / vad.speech_time_seconds * 60) if vad.speech_time_seconds > 0 else 0.0

    return RateResult(
        speaking_rate_wpm=speaking_rate,
        articulation_rate_wpm=articulation_rate,
        word_count=word_count,
        total_duration_seconds=duration_seconds,
        speech_time_seconds=vad.speech_time_seconds,
    )


def extract_features(
    samples: NDArray[np.float32],
    sample_rate: int,
    transcript: str | None = None,
    duration_seconds: float | None = None,
) -> FeatureBundle:
    frame_times = compute_frame_times(len(samples), sample_rate)
    rms_initial = compute_rms(samples, frame_times)
    vad = compute_vad(rms_initial.rms, frame_times)
    rms = compute_rms(samples, frame_times, speech_mask=vad.is_speech)
    f0 = compute_f0(samples, frame_times, vad_speech=vad.is_speech)

    rate = None
    if transcript is not None and duration_seconds is not None:
        rate = compute_rate(transcript, duration_seconds, vad)

    return FeatureBundle(
        frame_times=frame_times,
        rms=rms,
        vad=vad,
        f0=f0,
        rate=rate,
    )
