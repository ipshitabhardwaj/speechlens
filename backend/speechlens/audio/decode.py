from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import soundfile as sf
import soxr
from numpy.typing import NDArray

from speechlens.config import load_config


class AudioError(Exception):
    pass


class EmptyFileError(AudioError):
    pass


class CorruptedFileError(AudioError):
    pass


class DurationExceededError(AudioError):
    pass


class FileTooLargeError(AudioError):
    pass


class TooShortError(AudioError):
    pass


@dataclass(frozen=True)
class AudioMetadata:
    original_sample_rate: int
    channels: int
    duration_seconds: float
    format: str
    resampler: str


@dataclass(frozen=True)
class DecodedAudio:
    samples: NDArray[np.float32]
    sample_rate: int
    metadata: AudioMetadata


def _load_audio_config() -> dict:
    return load_config("features")["audio"]


def _detect_format(header: bytes) -> str | None:
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if header[:3] == b"ID3" or header[0:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if header[4:8] == b"ftyp":
        return "m4a"
    if header[:4] == b"fLaC":
        return "flac"
    if header[:4] == b"OggS":
        return "ogg"
    return None


def decode_audio(file_path: Path) -> DecodedAudio:
    """Decode an audio file to 16 kHz mono float32 for analysis.

    Checks file size before reading, container duration before decoding all
    frames, and actual sample count after decoding. No peak normalisation.
    """
    cfg = _load_audio_config()
    analysis_sr = cfg["analysis_sample_rate"]
    max_duration = cfg["max_duration_seconds"]
    min_duration = cfg["min_duration_seconds"]
    max_file_size_bytes = int(cfg["max_file_size_mb"] * 1_000_000)

    file_path = Path(file_path)

    if not file_path.exists():
        raise AudioError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise EmptyFileError("Audio file is empty (0 bytes)")

    if file_size > max_file_size_bytes:
        size_mb = file_size / 1_000_000
        raise FileTooLargeError(
            f"File size {size_mb:.1f} MB exceeds maximum of {cfg['max_file_size_mb']:.0f} MB"
        )

    header = file_path.read_bytes()[:12]
    detected_format = _detect_format(header) or file_path.suffix.lstrip(".")

    try:
        container = av.open(str(file_path))
    except av.error.InvalidDataError:
        raise CorruptedFileError("Cannot decode audio: file is corrupted or not a supported format")
    except av.error.FileNotFoundError:
        raise AudioError(f"File not found: {file_path}")
    except Exception as e:
        raise CorruptedFileError(f"Cannot decode audio: {e}")

    audio_streams = container.streams.audio
    if not audio_streams:
        container.close()
        raise CorruptedFileError("No audio stream found in file")

    stream = audio_streams[0]
    original_sample_rate = stream.rate or stream.codec_context.sample_rate
    channels = stream.channels or stream.codec_context.channels

    container_duration = float(container.duration / av.time_base) if container.duration else None
    if container_duration is not None and container_duration > max_duration:
        container.close()
        raise DurationExceededError(
            f"Audio duration {container_duration:.1f}s exceeds maximum of {max_duration:.0f}s"
        )

    frames: list[NDArray] = []
    decoded_samples = 0
    max_samples = int(max_duration * original_sample_rate) + original_sample_rate
    try:
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            frames.append(arr.astype(np.float32))
            decoded_samples += arr.shape[-1]
            if decoded_samples > max_samples:
                container.close()
                raise DurationExceededError(
                    f"Audio duration exceeds maximum of {max_duration:.0f}s (aborted during decode)"
                )
    except DurationExceededError:
        raise
    except Exception as e:
        container.close()
        raise CorruptedFileError(f"Error decoding audio frames: {e}")

    container.close()

    if not frames:
        raise CorruptedFileError("No audio frames decoded from file")

    audio = np.concatenate(frames)
    actual_duration = len(audio) / original_sample_rate

    if actual_duration > max_duration:
        raise DurationExceededError(
            f"Audio duration {actual_duration:.1f}s exceeds maximum of {max_duration:.0f}s"
        )

    if actual_duration < min_duration:
        raise TooShortError(
            f"Audio duration {actual_duration:.2f}s is shorter than minimum of {min_duration:.1f}s"
        )

    resampled = soxr.resample(audio, original_sample_rate, analysis_sr).astype(np.float32)

    metadata = AudioMetadata(
        original_sample_rate=original_sample_rate,
        channels=channels,
        duration_seconds=actual_duration,
        format=detected_format,
        resampler="soxr",
    )

    return DecodedAudio(
        samples=resampled,
        sample_rate=analysis_sr,
        metadata=metadata,
    )


def store_original(source_path: Path, dest_path: Path) -> None:
    """Copy the original upload byte-identical for playback."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_path), str(dest_path))


def save_analysis_copy(decoded: DecodedAudio, dest_path: Path) -> None:
    """Save the 16 kHz mono analysis copy as WAV."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest_path), decoded.samples, decoded.sample_rate, subtype="FLOAT")
