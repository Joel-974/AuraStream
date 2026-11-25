"""
Integration tests for the Ingestion Adapter FastAPI endpoints.
"""

from fastapi.testclient import TestClient
from fastapi import FastAPI
from classification.ingestion_adapter import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_classify_single_report_endpoint():
    payload = {
        "report_id": "test-rep-001",
        "text": "Flood waters rising rapidly, several families trapped on rooftops needing rescue boats",
        "modalities_available": ["text"],
    }
    response = client.post("/api/v1/classify-report", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["report_id"] == "test-rep-001"
    assert "classification" in res
    assert "decision" in res
    assert res["classification"]["status"] in ("predicted", "low_confidence")
    assert res["decision"]["suggested_priority"] in (
        "immediate",
        "high",
        "medium",
        "low",
        "requires_review",
    )


def test_classify_single_report_empty_text():
    payload = {
        "report_id": "test-rep-empty",
        "text": "",
    }
    response = client.post("/api/v1/classify-report", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["classification"]["status"] == "no_text"
    assert res["decision"]["suggested_priority"] == "invalid"
    assert res["decision"]["human_review_required"] is True


def test_batch_classify_reports_endpoint():
    payload = [
        {
            "report_id": "batch-1",
            "text": "Emergency shelter opened at community hall with food donations",
        },
        {
            "report_id": "batch-2",
            "text": "Power lines down and roads blocked by fallen trees",
        },
    ]
    response = client.post("/api/v1/batch-classify", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert len(res) == 2
    assert res[0]["report_id"] == "batch-1"
    assert res[1]["report_id"] == "batch-2"
