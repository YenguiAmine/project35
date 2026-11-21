from fastapi.testclient import TestClient

from project35.api.main import app

client = TestClient(app)


def test_swagger_openapi_is_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/v1/data/validate/" in response.json()["paths"]
    assert client.get("/docs").status_code == 200


def test_health_reports_phase_one():
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "phase": "1-data-foundation"}


def test_contract_endpoint_marks_unknown_project_values():
    response = client.get("/api/v1/data/contract/")
    assert response.status_code == 200
    assert response.json()["target_column"] == "to_be_confirmed_from_official_topic"


def test_validate_records_returns_quality_report(sample_records):
    response = client.post("/api/v1/data/validate/", json={"target_column": "target", "required_columns": ["age", "segment"], "records": sample_records})
    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == 4
    assert body["target_distribution"] == {"0": 2, "1": 2}
    assert body["missing_by_column"]["age"] == 0


def test_validate_records_rejects_missing_required_column(sample_records):
    response = client.post("/api/v1/data/validate/", json={"target_column": "target", "required_columns": ["missing_feature"], "records": sample_records})
    assert response.status_code == 422
    assert "Colonnes obligatoires absentes" in response.json()["detail"]


def test_validate_records_rejects_missing_target(sample_records):
    response = client.post("/api/v1/data/validate/", json={"target_column": "not_a_target", "records": sample_records})
    assert response.status_code == 422
    assert "Cible absente" in response.json()["detail"]


def test_validate_records_rejects_empty_payload():
    response = client.post("/api/v1/data/validate/", json={"target_column": "target", "records": []})
    assert response.status_code == 422
