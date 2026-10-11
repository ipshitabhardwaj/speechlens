from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    RecordingResponse,
    RubricInfoResponse,
    RubricRuleResponse,
)
from app.storage import (
    create_analysis,
    get_analysis,
    get_recording,
    get_recording_audio_path,
    run_analysis,
    save_recording,
)
from speechlens.config import load_config

router = APIRouter(prefix="/api")


@router.post("/recordings", response_model=RecordingResponse)
async def upload_recording(file: UploadFile) -> RecordingResponse:
    cfg = load_config("features")["audio"]
    max_bytes = int(cfg["max_file_size_mb"] * 1024 * 1024)

    contents = await file.read()
    if len(contents) > max_bytes:
        size_mb = len(contents) / (1024 * 1024)
        raise HTTPException(400, f"File too large: {size_mb:.1f} MB (max {cfg['max_file_size_mb']} MB)")
    if len(contents) == 0:
        raise HTTPException(400, "Empty file")

    try:
        meta = await asyncio.to_thread(save_recording, contents, file.filename or "upload.wav")
    except Exception as exc:
        raise HTTPException(422, str(exc))

    return RecordingResponse(
        id=meta["id"],
        duration=meta["duration"],
        sample_rate=meta["sample_rate"],
        warnings=meta["warnings"],
    )


@router.post("/analyses", response_model=AnalysisResponse)
async def start_analysis(req: AnalysisRequest) -> AnalysisResponse:
    if get_recording(req.participant_id) is None:
        raise HTTPException(404, f"Participant recording {req.participant_id} not found")
    if get_recording(req.reference_id) is None:
        raise HTTPException(404, f"Reference recording {req.reference_id} not found")
    if not req.transcript.strip():
        raise HTTPException(400, "Transcript must not be empty")

    analysis = create_analysis(
        req.participant_id, req.reference_id, req.transcript, req.rubric_version,
    )
    asyncio.create_task(run_analysis(analysis["id"]))

    return AnalysisResponse(
        id=analysis["id"],
        participant_id=analysis["participant_id"],
        reference_id=analysis["reference_id"],
        status=analysis["status"],
        created_at=analysis["created_at"],
    )


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_status(analysis_id: str) -> AnalysisResponse:
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(404, f"Analysis {analysis_id} not found")

    return AnalysisResponse(
        id=analysis["id"],
        participant_id=analysis["participant_id"],
        reference_id=analysis["reference_id"],
        status=analysis["status"],
        stage=analysis.get("stage"),
        error=analysis.get("error"),
        result=analysis.get("result"),
        created_at=analysis["created_at"],
    )


@router.get("/recordings/{recording_id}/audio")
async def get_audio(recording_id: str) -> FileResponse:
    audio_path = get_recording_audio_path(recording_id)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(404, f"Recording {recording_id} not found")

    suffix = audio_path.suffix.lower()
    media_types = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(audio_path, media_type=media_type)


@router.get("/rubric", response_model=RubricInfoResponse)
async def get_rubric() -> RubricInfoResponse:
    rubric_cfg = load_config("rubric_v1")
    rules_cfg = load_config("rules_v1")

    dimensions = []
    for name, dim in rubric_cfg["dimensions"].items():
        dimensions.append({"name": name, **dim})

    rules = []
    skip_keys = {"category", "severity_bands", "recommendation", "recommendation_fast", "recommendation_slow"}
    for rule_id in ["long_pause", "rate_deviation", "flat_pitch", "local_rate"]:
        rule = rules_cfg[rule_id]
        thresholds = {k: v for k, v in rule.items() if k not in skip_keys}
        rules.append(RubricRuleResponse(
            rule_id=rule_id,
            category=rule["category"],
            thresholds=thresholds,
        ))

    return RubricInfoResponse(
        version=rubric_cfg["version"],
        dimensions=dimensions,
        rules=rules,
    )
