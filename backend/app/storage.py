from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "analysis_outputs"

_recordings: dict[str, dict] = {}
_analyses: dict[str, dict] = {}


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_recording(file_bytes: bytes, filename: str) -> dict:
    from speechlens.audio.decode import decode_audio

    _ensure_dirs()
    rec_id = uuid.uuid4().hex[:12]
    rec_dir = UPLOADS_DIR / rec_id
    rec_dir.mkdir()

    suffix = Path(filename).suffix.lower() or ".wav"
    audio_path = rec_dir / f"original{suffix}"
    audio_path.write_bytes(file_bytes)

    try:
        decoded = decode_audio(audio_path)
    except Exception:
        import shutil
        shutil.rmtree(rec_dir, ignore_errors=True)
        raise

    duration = len(decoded.samples) / decoded.sample_rate
    warnings: list[str] = []
    if decoded.metadata.channels > 1:
        warnings.append(f"Downmixed from {decoded.metadata.channels} channels to mono")

    meta = {
        "id": rec_id,
        "filename": filename,
        "audio_file": audio_path.name,
        "duration": round(duration, 3),
        "sample_rate": decoded.sample_rate,
        "original_sample_rate": decoded.metadata.original_sample_rate,
        "format": decoded.metadata.format,
        "channels": decoded.metadata.channels,
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (rec_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _recordings[rec_id] = meta
    return meta


def get_recording(rec_id: str) -> dict | None:
    if rec_id in _recordings:
        return _recordings[rec_id]
    meta_path = UPLOADS_DIR / rec_id / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _recordings[rec_id] = meta
        return meta
    return None


def get_recording_audio_path(rec_id: str) -> Path | None:
    meta = get_recording(rec_id)
    if meta is None:
        return None
    return UPLOADS_DIR / rec_id / meta["audio_file"]


def create_analysis(
    participant_id: str,
    reference_id: str,
    transcript: str,
    rubric_version: str,
) -> dict:
    _ensure_dirs()
    analysis_id = uuid.uuid4().hex[:12]
    (OUTPUTS_DIR / analysis_id).mkdir()

    analysis = {
        "id": analysis_id,
        "participant_id": participant_id,
        "reference_id": reference_id,
        "transcript": transcript,
        "rubric_version": rubric_version,
        "status": "queued",
        "stage": None,
        "error": None,
        "result": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _analyses[analysis_id] = analysis
    return analysis


def get_analysis(analysis_id: str) -> dict | None:
    if analysis_id in _analyses:
        return _analyses[analysis_id]
    result_path = OUTPUTS_DIR / analysis_id / "result.json"
    if result_path.exists():
        analysis = json.loads(result_path.read_text(encoding="utf-8"))
        _analyses[analysis_id] = analysis
        return analysis
    return None


def _run_pipeline(analysis_id: str) -> None:
    from speechlens.audio.decode import decode_audio
    from speechlens.features.extract import extract_features
    from speechlens.alignment.energy_fallback import EnergyFallbackAligner
    from speechlens.flaws.rules import detect_all_flaws
    from speechlens.rubric.rubric import score_rubric

    analysis = _analyses[analysis_id]

    analysis["status"] = "running"
    analysis["stage"] = "decoding"

    part_path = get_recording_audio_path(analysis["participant_id"])
    ref_path = get_recording_audio_path(analysis["reference_id"])
    part_decoded = decode_audio(part_path)
    ref_decoded = decode_audio(ref_path)
    part_dur = len(part_decoded.samples) / part_decoded.sample_rate
    ref_dur = len(ref_decoded.samples) / ref_decoded.sample_rate

    analysis["stage"] = "extracting_features"
    transcript = analysis["transcript"]
    part_feat = extract_features(
        part_decoded.samples, part_decoded.sample_rate, transcript, part_dur,
    )
    ref_feat = extract_features(
        ref_decoded.samples, ref_decoded.sample_rate, transcript, ref_dur,
    )

    analysis["stage"] = "aligning"
    aligner = EnergyFallbackAligner()
    alignment = aligner.align(
        part_decoded.samples, part_decoded.sample_rate, transcript, part_feat.vad,
    )

    analysis["stage"] = "comparing"
    flaws, unavailable = detect_all_flaws(part_feat, ref_feat, alignment.method)
    unavailable.extend(alignment.unavailable)

    analysis["stage"] = "scoring"
    rubric = score_rubric(part_feat, ref_feat, flaws, unavailable)

    result = {
        "flaws": [dataclasses.asdict(f) for f in flaws],
        "unavailable": unavailable,
        "rubric": {
            "overall_score": rubric.overall_score,
            "version": rubric.version,
            "dimensions": [dataclasses.asdict(d) for d in rubric.dimensions],
        },
        "alignment_method": alignment.method,
        "participant_duration": round(part_dur, 3),
        "reference_duration": round(ref_dur, 3),
        "participant_rate_wpm": (
            round(part_feat.rate.articulation_rate_wpm, 1)
            if part_feat.rate else None
        ),
        "reference_rate_wpm": (
            round(ref_feat.rate.articulation_rate_wpm, 1)
            if ref_feat.rate else None
        ),
    }

    analysis["status"] = "complete"
    analysis["stage"] = None
    analysis["result"] = result

    out_path = OUTPUTS_DIR / analysis_id / "result.json"
    out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")


async def run_analysis(analysis_id: str) -> None:
    analysis = _analyses[analysis_id]
    try:
        await asyncio.to_thread(_run_pipeline, analysis_id)
    except Exception as e:
        analysis["status"] = "error"
        analysis["stage"] = None
        analysis["error"] = str(e)
