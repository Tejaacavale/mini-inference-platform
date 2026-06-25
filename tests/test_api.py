"""API tests.

The model is monkeypatched with a deterministic stub so tests run fast and
offline (no model download, no torch inference). The batcher, endpoints, and
schema wiring are still exercised end to end.
"""
import pytest
from fastapi.testclient import TestClient

from app import model


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    """Replace the real model with a deterministic keyword-based stub."""
    def fake_predict(texts):
        out = []
        for t in texts:
            if "bad" in t.lower() or "terrible" in t.lower():
                out.append({"label": "NEGATIVE", "score": 0.99})
            else:
                out.append({"label": "POSITIVE", "score": 0.98})
        return out

    monkeypatch.setattr(model, "_pipeline", object())  # mark as "loaded"
    monkeypatch.setattr(model, "load_model", lambda: model._pipeline)
    monkeypatch.setattr(model, "predict", fake_predict)
    monkeypatch.setattr(model, "is_loaded", lambda: True)


@pytest.fixture
def client():
    # Import inside the fixture so the monkeypatched model is in place before
    # the app's lifespan runs.
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200


def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code == 200


def test_predict_single(client):
    r = client.post("/v1/predict", json={"text": "I love this, it is great"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"POSITIVE", "NEGATIVE"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["label"] == "POSITIVE"


def test_predict_single_negative(client):
    r = client.post("/v1/predict", json={"text": "this is terrible and bad"})
    assert r.status_code == 200
    assert r.json()["label"] == "NEGATIVE"


def test_predict_batch(client):
    payload = {"texts": ["wonderful day", "terrible service"]}
    r = client.post("/v1/predict/batch", json=payload)
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
    assert results[0]["label"] == "POSITIVE"
    assert results[1]["label"] == "NEGATIVE"


def test_metrics_endpoint(client):
    client.post("/v1/predict", json={"text": "great"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "requests_total" in r.text
