"""End-to-end API test using dev fixtures."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

DEV_FIXTURES = Path(__file__).resolve().parent.parent / "dev_fixtures"

pytestmark = pytest.mark.skipif(
    not (DEV_FIXTURES / "tts_ideal.wav").exists(),
    reason="dev fixtures not generated",
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


class TestRecordingUpload:
    def test_upload_wav(self, client: TestClient) -> None:
        wav = (DEV_FIXTURES / "tts_ideal.wav").read_bytes()
        resp = client.post(
            "/api/recordings",
            files={"file": ("tts_ideal.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["duration"] > 30
        assert data["sample_rate"] == 16000

    def test_upload_empty_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/api/recordings",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        assert resp.status_code == 400

    def test_audio_retrieval(self, client: TestClient) -> None:
        wav = (DEV_FIXTURES / "tts_ideal.wav").read_bytes()
        upload = client.post(
            "/api/recordings",
            files={"file": ("tts_ideal.wav", wav, "audio/wav")},
        ).json()
        resp = client.get(f"/api/recordings/{upload['id']}/audio")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert len(resp.content) > 0


class TestAnalysis:
    def _upload(self, client: TestClient, filename: str) -> str:
        wav = (DEV_FIXTURES / filename).read_bytes()
        return client.post(
            "/api/recordings",
            files={"file": (filename, wav, "audio/wav")},
        ).json()["id"]

    def test_full_analysis_pipeline(self, client: TestClient) -> None:
        ref_id = self._upload(client, "tts_ideal.wav")
        part_id = self._upload(client, "tts_pauses.wav")
        transcript = (DEV_FIXTURES / "passage.txt").read_text(encoding="utf-8").strip()

        resp = client.post("/api/analyses", json={
            "participant_id": part_id,
            "reference_id": ref_id,
            "transcript": transcript,
        })
        assert resp.status_code == 200
        analysis_id = resp.json()["id"]
        assert resp.json()["status"] == "queued"
        assert resp.json()["participant_id"] == part_id
        assert resp.json()["reference_id"] == ref_id

        deadline = time.time() + 120
        while time.time() < deadline:
            status = client.get(f"/api/analyses/{analysis_id}").json()
            if status["status"] in ("complete", "error"):
                break
            time.sleep(0.5)

        assert status["status"] == "complete", f"Analysis failed: {status.get('error')}"

        result = status["result"]
        assert result["alignment_method"] in ("ctc", "energy_fallback")

        pause_flaws = [f for f in result["flaws"] if f["rule_id"] == "long_pause"]
        assert len(pause_flaws) == 3
        for flaw in pause_flaws:
            assert flaw["start_seconds"] > 0
            assert flaw["end_seconds"] > flaw["start_seconds"]
            assert "evidence" in flaw
            assert "interpretation" in flaw

        if result["alignment_method"] == "ctc":
            assert result["participant_words"] is not None
            assert result["reference_words"] is not None
            assert len(result["participant_words"]) > 0
            for pw in pause_flaws:
                assert "preceding_word" in pw["participant_measurement"]

        rubric = result["rubric"]
        assert rubric["version"] == "1"
        assert len(rubric["dimensions"]) == 3
        assert 0 <= rubric["overall_score"] <= 100

        assert result["participant_duration"] > 0
        assert result["reference_duration"] > 0

    def test_missing_recording_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/analyses", json={
            "participant_id": "nonexistent",
            "reference_id": "nonexistent",
            "transcript": "hello",
        })
        assert resp.status_code == 404

    def test_empty_transcript_rejected(self, client: TestClient) -> None:
        ref_id = self._upload(client, "tts_ideal.wav")
        resp = client.post("/api/analyses", json={
            "participant_id": ref_id,
            "reference_id": ref_id,
            "transcript": "   ",
        })
        assert resp.status_code == 400


class TestRubric:
    def test_get_rubric(self, client: TestClient) -> None:
        resp = client.get("/api/rubric")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "1"
        assert len(data["dimensions"]) == 3
        assert len(data["rules"]) == 4
        rule_ids = {r["rule_id"] for r in data["rules"]}
        assert rule_ids == {"long_pause", "rate_deviation", "flat_pitch", "local_rate"}
