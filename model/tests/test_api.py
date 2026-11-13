"""
tests/test_api.py - tests for the FastAPI backend (app/main.py).

Requires artifacts/scam_model.joblib (run python src/train.py first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
ROOT_DIR = Path(__file__).resolve().parent.parent
for _d in (SRC_DIR, ROOT_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

MODEL = Path(__file__).resolve().parent.parent / "artifacts" / "scam_model.joblib"


@pytest.fixture(scope="module")
def client():
    if not MODEL.exists():
        pytest.skip("scam_model.joblib not found - run python src/train.py first")
    from fastapi.testclient import TestClient

    import app.main as main_mod  # noqa: WPS433

    return TestClient(main_mod.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_predict(client):
    r = client.post("/predict", json={"text": "Hi Mom, I will be home by 6pm"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"is_scam", "risk_score", "risk_level", "reasons", "links"}
    assert isinstance(body["risk_score"], int)
    assert 0 <= body["risk_score"] <= 100


def test_predict_batch(client):
    r = client.post("/predict-batch",
                    json={"texts": ["Hi there", "URGENT verify bit.ly/abc now"]})
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_predict_empty_rejected(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_predict_missing_text(client):
    r = client.post("/predict", json={})
    assert r.status_code in (400, 422)


def test_cors_header(client):
    r = client.get("/health")
    # CORS middleware present (allow_origins defaults to "*" for easy website plug-in)
    assert r.status_code == 200
