from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_speechlens_import():
    import speechlens
    import speechlens.audio
    import speechlens.features
    import speechlens.alignment
