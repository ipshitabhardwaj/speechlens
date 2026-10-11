from __future__ import annotations

from pydantic import BaseModel


class RecordingResponse(BaseModel):
    id: str
    duration: float
    sample_rate: int
    warnings: list[str] = []


class AnalysisRequest(BaseModel):
    participant_id: str
    reference_id: str
    transcript: str
    rubric_version: str = "1"


class FlawResponse(BaseModel):
    rule_id: str
    category: str
    severity: str
    start_seconds: float
    end_seconds: float
    participant_measurement: dict
    reference_measurement: dict
    deviation: float
    evidence: str
    interpretation: str
    recommendation: str
    scope: str


class WordTimingResponse(BaseModel):
    word: str
    start: float
    end: float
    score: float


class DimensionResponse(BaseModel):
    name: str
    score: float
    weight: float
    available: bool
    unavailable_reason: str | None = None


class RubricScoreResponse(BaseModel):
    overall_score: float
    version: str
    dimensions: list[DimensionResponse]


class AnalysisResultResponse(BaseModel):
    flaws: list[FlawResponse]
    unavailable: list[str]
    rubric: RubricScoreResponse
    alignment_method: str
    participant_duration: float
    reference_duration: float
    participant_rate_wpm: float | None = None
    reference_rate_wpm: float | None = None
    participant_words: list[WordTimingResponse] | None = None
    reference_words: list[WordTimingResponse] | None = None


class AnalysisResponse(BaseModel):
    id: str
    participant_id: str
    reference_id: str
    status: str
    stage: str | None = None
    error: str | None = None
    result: AnalysisResultResponse | None = None
    created_at: str


class RubricRuleResponse(BaseModel):
    rule_id: str
    category: str
    thresholds: dict


class RubricInfoResponse(BaseModel):
    version: str
    dimensions: list[dict]
    rules: list[RubricRuleResponse]
