from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from rest_framework.test import APIClient

from predictions.services import ModelUnavailable
from predictions.monitoring import PSIReferenceUnavailable
from project35.monitoring.psi import PSIError
from project35.data.phase1 import build_baseline_pipeline
from project35.explainability.shap_service import ShapExplanationService


class FakeClassifier:
    feature_names_in_ = np.array(["balance", "segment"])

    def predict(self, frame):
        return np.array([1] * len(frame))

    def predict_proba(self, frame):
        return np.tile(np.array([[0.2, 0.8]]), (len(frame), 1))


@pytest.fixture
def client():
    return APIClient()


def active_metadata():
    return {
        "name": "project35-classifier",
        "alias": "champion",
        "version": "3",
        "run_id": "run-123",
        "source": "runs:/run-123/baseline_pipeline",
        "metrics": {"f1": 0.82, "roc_auc": 0.88},
        "tags": {"validation_status": "approved"},
    }


def test_health_is_available_without_registry_connection(client):
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.data["status"] == "degraded"
    assert response.data["model_loaded"] is False


def test_swagger_and_openapi_expose_scoring_routes(client):
    schema = client.get("/api/schema/")
    assert schema.status_code == 200
    assert "/api/predict/" in schema.data["paths"]
    assert "/api/model-info/" in schema.data["paths"]
    assert "/api/explain/" in schema.data["paths"]
    assert "/api/explain/global/" in schema.data["paths"]
    assert "/api/monitoring/psi/" in schema.data["paths"]
    assert client.get("/api/docs/").status_code == 200


@patch("predictions.views.model_provider.get_active", return_value=(FakeClassifier(), active_metadata()))
def test_predict_returns_class_probability_request_id_and_version(_model, client):
    response = client.post(
        "/api/predict/",
        {"instances": [{"balance": 1200.5, "segment": "A"}]},
        format="json",
    )
    assert response.status_code == 200
    assert response.data["prediction"] == [1]
    assert response.data["probabilities"] == [[0.2, 0.8]]
    assert response.data["model_version"] == "3"
    assert response.data["request_id"]


@patch("predictions.views.model_provider.get_active", return_value=(FakeClassifier(), active_metadata()))
def test_predict_rejects_missing_or_unknown_features(_model, client):
    response = client.post("/api/predict/", {"instances": [{"balance": 10}]}, format="json")
    assert response.status_code == 400
    assert response.data["instances"]["missing_features"] == ["segment"]

    response = client.post(
        "/api/predict/",
        {"instances": [{"balance": 10, "segment": "A", "secret_extra": "x"}]},
        format="json",
    )
    assert response.status_code == 400
    assert response.data["instances"]["unknown_features"] == ["secret_extra"]


def test_predict_rejects_empty_and_oversized_batches(client):
    assert client.post("/api/predict/", {"instances": []}, format="json").status_code == 400
    records = [{"balance": 1, "segment": "A"}] * 1001
    assert client.post("/api/predict/", {"instances": records}, format="json").status_code == 400


@patch("predictions.views.model_provider.get_active", return_value=(FakeClassifier(), active_metadata()))
def test_model_info_returns_registry_metadata(_model, client):
    response = client.get("/api/model-info/")
    assert response.status_code == 200
    assert response.data["version"] == "3"
    assert response.data["metrics"]["roc_auc"] == 0.88


@patch("predictions.views.model_provider.get_active", side_effect=ModelUnavailable("registry down"))
def test_model_unavailable_returns_503_without_internal_details(_model, client):
    response = client.post(
        "/api/predict/", {"instances": [{"balance": 10, "segment": "A"}]}, format="json"
    )
    assert response.status_code == 503
    assert "registry down" not in str(response.data)


@patch("predictions.views.model_provider.get_active", return_value=(FakeClassifier(), active_metadata()))
@patch("predictions.views.get_explanation_service")
def test_local_explanation_api_returns_shap_values_and_png(_service, _model, client):
    _service.return_value.explain_one.return_value = {
        "class_index": 1,
        "class_value": 1,
        "output_scale": "probability",
        "explainer_method": "PermutationExplainer",
        "reference_sha256": "a" * 64,
        "base_value": 0.5,
        "prediction_probability": 0.8,
        "additivity_residual": 0.0,
        "shap_values": {"balance": 0.2, "segment": 0.1},
        "waterfall_png_base64": "iVBORw0KGgo=",
        "interpretation": "Contribution SHAP, sans causalité.",
    }
    response = client.post(
        "/api/explain/",
        {"instance": {"balance": 1200, "segment": "A"}, "class_index": 1},
        format="json",
    )
    assert response.status_code == 200
    assert response.data["shap_values"]["balance"] == 0.2
    assert response.data["explainer_method"] == "PermutationExplainer"
    assert response.data["waterfall_png_base64"].startswith("iVBOR")
    assert response.data["model_version"] == "3"
    assert response.data["reference_sha256"] == "a" * 64


@patch("predictions.views.model_provider.get_active", return_value=(FakeClassifier(), active_metadata()))
@patch("predictions.views.get_explanation_service")
def test_global_explanation_api_returns_mean_absolute_shap(_service, _model, client):
    _service.return_value.explain_global.return_value = {
        "class_index": 1,
        "class_value": 1,
        "output_scale": "probability",
        "explainer_method": "PermutationExplainer",
        "reference_sha256": "b" * 64,
        "rows_explained": 12,
        "mean_absolute_shap": {"balance": 0.3, "segment": 0.1},
        "summary_png_base64": "iVBORw0KGgo=",
        "beeswarm_png_base64": "iVBORw0KGgo=",
        "interpretation": "Importance SHAP, sans causalité.",
    }
    response = client.get("/api/explain/global/?class_index=1")
    assert response.status_code == 200
    assert response.data["mean_absolute_shap"]["balance"] == 0.3
    assert response.data["explainer_method"] == "PermutationExplainer"
    assert response.data["model_name"] == "project35-classifier"


def test_tree_explainer_is_exposed_through_local_and_global_api(client):
    frame = pd.DataFrame({
        "usage": [0.05, 0.12, 0.22, 0.31, 0.48, 0.57, 0.74, 0.88, 0.93, 0.99],
        "region": ["north", "south", "north", "west", "south", "east", "west", "east", "north", "south"],
    })
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    model = build_baseline_pipeline(frame, classifier_name="decision_tree", classifier_params={"max_depth": 3})
    model.fit(frame, labels)
    service = ShapExplanationService(frame, max_background_rows=8, max_global_rows=6, reference_sha256="c" * 64)

    with patch("predictions.views.model_provider.get_active", return_value=(model, active_metadata())), patch(
        "predictions.views.get_explanation_service", return_value=service
    ):
        local = client.post(
            "/api/explain/",
            {"instance": {"usage": 0.2, "region": "north"}, "class_index": 1},
            format="json",
        )
        global_result = client.get("/api/explain/global/?class_index=1")

    assert local.status_code == 200
    assert local.data["explainer_method"] == "TreeExplainer"
    assert set(local.data["shap_values"]) == set(frame.columns)
    assert abs(local.data["additivity_residual"]) < 1e-5
    assert global_result.status_code == 200
    assert global_result.data["explainer_method"] == "TreeExplainer"
    assert set(global_result.data["mean_absolute_shap"]) == set(frame.columns)


@patch("predictions.views.get_psi_profile")
def test_psi_api_returns_report_and_alert(_profile, client):
    _profile.return_value.calculate.return_value = {
        "report_id": "psi-report-1",
        "reference_sha256": "a" * 64,
        "current_sha256": "b" * 64,
        "reference_rows": 100,
        "current_rows": 2,
        "reference_window": "configured_reference",
        "current_window": "2026-11-20T00:00:00Z",
        "observed_at": "2026-11-20T00:00:00+00:00",
        "bins": 10,
        "epsilon": 1e-6,
        "thresholds": {"watch": 0.1, "drift": 0.25},
        "status": "drift",
        "alert": True,
        "features": {
            "balance": {
                "psi": 0.31,
                "status": "drift",
                "reference_proportions": {"bin:0": 1.0},
                "current_proportions": {"bin:0": 0.5, "overflow": 0.5},
                "contributions": {"bin:0": 0.1},
                "bin_definition": {"kind": "numeric_quantile_bins", "internal_edges": []},
            }
        },
    }
    response = client.post(
        "/api/monitoring/psi/",
        {
            "records": [{"balance": 150.0, "segment": "A"}, {"balance": 200.0, "segment": "B"}],
            "current_window": "2026-11-20T00:00:00Z",
        },
        format="json",
    )
    assert response.status_code == 200
    assert response.data["alert"] is True
    assert response.data["features"]["balance"]["psi"] == 0.31


def test_psi_api_loads_reference_and_calculates_real_profile(tmp_path, monkeypatch, client):
    reference_path = tmp_path / "approved_reference.csv"
    reference = pd.DataFrame({
        "balance": [10.0, 20.0, 30.0, 40.0],
        "segment": ["A", "B", "A", "B"],
    })
    reference.to_csv(reference_path, index=False)
    monkeypatch.setenv("PROJECT35_PSI_REFERENCE", str(reference_path))

    response = client.post(
        "/api/monitoring/psi/",
        {
            "current_window": "same-as-reference",
            "records": reference.to_dict(orient="records"),
        },
        format="json",
    )

    assert response.status_code == 200
    assert response.data["status"] == "stable"
    assert response.data["features"]["balance"]["psi"] == pytest.approx(0.0)
    assert len(response.data["reference_sha256"]) == 64
    assert len(response.data["current_sha256"]) == 64


@patch("predictions.views.get_psi_profile", side_effect=PSIReferenceUnavailable("reference missing"))
def test_psi_api_reports_missing_reference_as_service_unavailable(_profile, client):
    response = client.post(
        "/api/monitoring/psi/",
        {"records": [{"balance": 1.0, "segment": "A"}]},
        format="json",
    )
    assert response.status_code == 503
