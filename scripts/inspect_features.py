"""Inspect feature extraction on a real recording.

Usage:
    uv run python scripts/inspect_features.py <audio_path> <transcript_path>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from speechlens.audio.decode import decode_audio
from speechlens.features.extract import extract_features


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <audio_path> <transcript_path>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    transcript_path = Path(sys.argv[2])

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)
    if not transcript_path.exists():
        print(f"Transcript file not found: {transcript_path}")
        sys.exit(1)

    transcript = transcript_path.read_text(encoding="utf-8").strip()

    print(f"Audio file: {audio_path}")
    print(f"Transcript: {transcript[:80]}{'...' if len(transcript) > 80 else ''}")
    print(f"Word count: {len(transcript.split())}")
    print()

    decoded = decode_audio(audio_path)
    duration = len(decoded.samples) / decoded.sample_rate
    print(f"Duration: {duration:.3f}s")
    print(f"Sample rate: {decoded.sample_rate} Hz")
    print(f"Samples: {len(decoded.samples)}")
    print(f"Original SR: {decoded.metadata.original_sample_rate} Hz")
    print(f"Channels: {decoded.metadata.channels}")
    print(f"Format: {decoded.metadata.format}")
    print(f"Resampler: {decoded.metadata.resampler}")
    print()

    t0 = time.perf_counter()
    bundle = extract_features(decoded.samples, decoded.sample_rate, transcript, duration)
    elapsed = time.perf_counter() - t0

    ft = bundle.frame_times
    print(f"Frames: {len(ft.times)}")
    print(f"Hop: {ft.hop_length} samples ({ft.hop_length / ft.sample_rate * 1000:.1f} ms)")
    print()

    # Leading / trailing silence
    if bundle.vad.leading_silence:
        ls = bundle.vad.leading_silence
        print(f"Leading silence: {ls.start:.3f}s – {ls.end:.3f}s ({ls.end - ls.start:.3f}s)")
    else:
        print("Leading silence: none")
    if bundle.vad.trailing_silence:
        ts = bundle.vad.trailing_silence
        print(f"Trailing silence: {ts.start:.3f}s – {ts.end:.3f}s ({ts.end - ts.start:.3f}s)")
    else:
        print("Trailing silence: none")
    print()

    # Pauses
    print(f"Pauses: {len(bundle.vad.pauses)}")
    for i, p in enumerate(bundle.vad.pauses):
        print(f"  [{i}] {p.start:.3f}s – {p.end:.3f}s  ({p.end - p.start:.3f}s)")
    print()

    print(f"Speech time: {bundle.vad.speech_time_seconds:.3f}s")
    print(f"Total pause time: {bundle.vad.total_pause_time_seconds:.3f}s")
    print()

    # Rate
    if bundle.rate:
        print(f"Speaking rate: {bundle.rate.speaking_rate_wpm:.1f} WPM")
        print(f"Articulation rate: {bundle.rate.articulation_rate_wpm:.1f} WPM")
    else:
        print("Rate: not computed (no transcript)")
    print()

    # F0
    print(f"Median F0: {bundle.f0.median_f0_hz:.1f} Hz")
    voiced_f0 = bundle.f0.f0[bundle.f0.voiced_flag]
    if len(voiced_f0) > 0:
        median_f0 = np.median(voiced_f0)
        semitones = 12 * np.log2(voiced_f0 / median_f0)
        iqr = float(np.percentile(semitones, 75) - np.percentile(semitones, 25))
        print(f"F0 IQR: {iqr:.2f} semitones (relative to median)")
        print(f"F0 range: {voiced_f0.min():.1f} – {voiced_f0.max():.1f} Hz")
    else:
        print("F0 IQR: no voiced frames")

    speech_frames = int(bundle.vad.is_speech.sum())
    voiced_speech = int((bundle.f0.voiced_flag & bundle.vad.is_speech[:len(bundle.f0.voiced_flag)]).sum())
    if speech_frames > 0:
        frac = voiced_speech / speech_frames
        print(f"Voiced fraction of speech frames: {voiced_speech}/{speech_frames} = {frac:.4f}")
    else:
        print("Voiced fraction of speech frames: no speech frames")
    print()

    print(f"Extraction wall-clock: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
