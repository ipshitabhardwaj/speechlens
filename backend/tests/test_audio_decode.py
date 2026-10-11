"""Tests for speechlens.audio.decode — audio decoding, validation, and 16 kHz conversion."""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import av
import numpy as np
import pytest
import soundfile as sf

from speechlens.audio.decode import (
    AudioMetadata,
    CorruptedFileError,
    DecodedAudio,
    DurationExceededError,
    EmptyFileError,
    FileTooLargeError,
    TooShortError,
    decode_audio,
    save_analysis_copy,
    store_original,
)
from speechlens.config import load_config

ANALYSIS_SAMPLE_RATE = load_config("features")["audio"]["analysis_sample_rate"]


def _make_sine(freq: float = 440.0, duration: float = 1.0, sr: int = 44100, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray, sr: int = 44100, channels: int = 1) -> None:
    if channels == 2 and samples.ndim == 1:
        samples = np.column_stack([samples, samples])
    sf.write(str(path), samples, sr, subtype="PCM_16")


def _write_mp3(path: Path, samples: np.ndarray, sr: int = 44100) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream("mp3", rate=sr, layout="mono")
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
    frame.rate = sr
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_m4a(path: Path, samples: np.ndarray, sr: int = 44100) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream("aac", rate=sr, layout="mono")
    chunk_size = 1024
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i : i + chunk_size]
        frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="flt", layout="mono")
        frame.rate = sr
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    return tmp_path


class TestDecodeFormats:
    def test_decode_wav(self, tmp: Path) -> None:
        wav_path = tmp / "test.wav"
        samples = _make_sine(duration=1.0)
        _write_wav(wav_path, samples)

        result = decode_audio(wav_path)

        assert isinstance(result, DecodedAudio)
        assert result.sample_rate == ANALYSIS_SAMPLE_RATE
        assert result.samples.dtype == np.float32
        assert result.samples.ndim == 1
        assert result.metadata.format == "wav"
        assert result.metadata.original_sample_rate == 44100
        assert result.metadata.channels == 1
        assert abs(result.metadata.duration_seconds - 1.0) < 0.05

    def test_decode_mp3(self, tmp: Path) -> None:
        mp3_path = tmp / "test.mp3"
        samples = _make_sine(duration=1.0)
        _write_mp3(mp3_path, samples)

        result = decode_audio(mp3_path)

        assert result.sample_rate == ANALYSIS_SAMPLE_RATE
        assert result.metadata.format == "mp3"
        assert result.metadata.channels == 1
        assert result.metadata.duration_seconds > 0.8

    def test_decode_m4a(self, tmp: Path) -> None:
        m4a_path = tmp / "test.m4a"
        samples = _make_sine(duration=1.0)
        _write_m4a(m4a_path, samples)

        result = decode_audio(m4a_path)

        assert result.sample_rate == ANALYSIS_SAMPLE_RATE
        assert result.metadata.format == "m4a"
        assert result.metadata.channels == 1
        assert result.metadata.duration_seconds > 0.8


class TestStereoDownmix:
    def test_stereo_downmixed_to_mono(self, tmp: Path) -> None:
        wav_path = tmp / "stereo.wav"
        samples = _make_sine(duration=1.5)
        _write_wav(wav_path, samples, channels=2)

        result = decode_audio(wav_path)

        assert result.samples.ndim == 1
        assert result.metadata.channels == 2


class TestOriginalStorage:
    def test_store_original_byte_identical(self, tmp: Path) -> None:
        src = tmp / "original.wav"
        samples = _make_sine(duration=1.5)
        _write_wav(src, samples)
        original_bytes = src.read_bytes()

        dest = tmp / "stored" / "copy.wav"
        store_original(src, dest)

        assert dest.read_bytes() == original_bytes


class TestAnalysisCopy:
    def test_analysis_copy_16khz_mono_float(self, tmp: Path) -> None:
        wav_path = tmp / "test.wav"
        samples = _make_sine(duration=1.0, sr=48000)
        _write_wav(wav_path, samples, sr=48000)

        result = decode_audio(wav_path)

        assert result.sample_rate == ANALYSIS_SAMPLE_RATE
        assert result.samples.dtype == np.float32
        assert result.samples.ndim == 1

        analysis_path = tmp / "analysis.wav"
        save_analysis_copy(result, analysis_path)
        data, sr = sf.read(str(analysis_path))
        assert sr == ANALYSIS_SAMPLE_RATE
        assert data.ndim == 1

    def test_resampler_named_in_metadata(self, tmp: Path) -> None:
        wav_path = tmp / "test.wav"
        _write_wav(wav_path, _make_sine(duration=1.5))

        result = decode_audio(wav_path)

        assert result.metadata.resampler == "soxr"


class TestNoPeakNormalisation:
    def test_half_amplitude_measures_about_6db_lower(self, tmp: Path) -> None:
        sr = 44100
        full = _make_sine(duration=1.0, sr=sr, amplitude=0.5)
        half = _make_sine(duration=1.0, sr=sr, amplitude=0.25)

        full_path = tmp / "full.wav"
        half_path = tmp / "half.wav"
        _write_wav(full_path, full, sr=sr)
        _write_wav(half_path, half, sr=sr)

        full_result = decode_audio(full_path)
        half_result = decode_audio(half_path)

        full_rms = np.sqrt(np.mean(full_result.samples ** 2))
        half_rms = np.sqrt(np.mean(half_result.samples ** 2))

        db_diff = 20 * np.log10(full_rms / half_rms)
        assert 5.0 < db_diff < 7.0, f"Expected ~6 dB difference, got {db_diff:.1f} dB"


class TestValidation:
    def test_reject_over_3_minutes(self, tmp: Path) -> None:
        wav_path = tmp / "long.wav"
        sr = 8000
        samples = _make_sine(duration=181.0, sr=sr)
        _write_wav(wav_path, samples, sr=sr)

        with pytest.raises(DurationExceededError, match="exceeds maximum"):
            decode_audio(wav_path)

    def test_reject_empty_file(self, tmp: Path) -> None:
        empty_path = tmp / "empty.wav"
        empty_path.write_bytes(b"")

        with pytest.raises(EmptyFileError, match="empty"):
            decode_audio(empty_path)

    def test_reject_corrupted_file(self, tmp: Path) -> None:
        corrupt_path = tmp / "corrupt.wav"
        corrupt_path.write_bytes(b"this is not audio data at all" * 10)

        with pytest.raises(CorruptedFileError, match="corrupted|Cannot decode"):
            decode_audio(corrupt_path)

    def test_reject_oversized_file(self, tmp: Path) -> None:
        big_path = tmp / "big.wav"
        sr = 8000
        max_mb = load_config("features")["audio"]["max_file_size_mb"]
        # Write enough samples to exceed the file size limit
        n_samples = int(max_mb * 1_000_000 / 2) + sr  # PCM_16 = 2 bytes/sample + WAV header
        samples = _make_sine(duration=n_samples / sr, sr=sr)
        _write_wav(big_path, samples, sr=sr)
        assert big_path.stat().st_size > max_mb * 1_000_000

        with pytest.raises(FileTooLargeError, match="exceeds maximum"):
            decode_audio(big_path)

    def test_reject_too_short(self, tmp: Path) -> None:
        short_path = tmp / "short.wav"
        sr = 44100
        min_dur = load_config("features")["audio"]["min_duration_seconds"]
        samples = _make_sine(duration=min_dur * 0.5, sr=sr)
        _write_wav(short_path, samples, sr=sr)

        with pytest.raises(TooShortError, match="shorter than minimum"):
            decode_audio(short_path)

    def test_file_size_checked_before_decode(self, tmp: Path) -> None:
        """File size is rejected before attempting to decode."""
        big_path = tmp / "big.bin"
        max_bytes = int(load_config("features")["audio"]["max_file_size_mb"] * 1_000_000)
        big_path.write_bytes(b"\x00" * (max_bytes + 1))

        with pytest.raises(FileTooLargeError):
            decode_audio(big_path)


class TestMetadata:
    def test_metadata_fields(self, tmp: Path) -> None:
        wav_path = tmp / "test.wav"
        sr = 22050
        samples = _make_sine(duration=2.0, sr=sr)
        _write_wav(wav_path, samples, sr=sr)

        result = decode_audio(wav_path)
        meta = result.metadata

        assert meta.original_sample_rate == sr
        assert meta.channels == 1
        assert abs(meta.duration_seconds - 2.0) < 0.05
        assert meta.format == "wav"
        assert isinstance(meta.resampler, str) and len(meta.resampler) > 0
