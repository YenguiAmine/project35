import base64

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from project35.data.phase1 import build_baseline_pipeline

from project35.explainability.shap_service import ExplanationError, ShapExplanationService


@pytest.fixture
def fitted_model_and_background():
    frame = pd.DataFrame({
        "balance": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        "age": [20, 25, 30, 45, 50, 55],
    })
    labels = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression().fit(frame, labels)
    return model, frame


def test_local_shap_is_additive_and_has_original_feature_names(fitted_model_and_background):
    model, background = fitted_model_and_background
    service = ShapExplanationService(background, max_evals=5)

    result = service.explain_one(model, {"age": 52, "balance": 2.5})

    assert set(result["shap_values"]) == set(background.columns)
    assert result["output_scale"] == "probability"
    assert abs(result["additivity_residual"]) < 1e-6
    png = base64.b64decode(result["waterfall_png_base64"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_global_shap_returns_sorted_importance_and_plot(fitted_model_and_background):
    model, background = fitted_model_and_background
    result = ShapExplanationService(background, max_evals=5).explain_global(model)

    importances = list(result["mean_absolute_shap"].values())
    assert set(result["mean_absolute_shap"]) == set(background.columns)
    assert importances == sorted(importances, reverse=True)
    assert base64.b64decode(result["summary_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert base64.b64decode(result["beeswarm_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")


def test_shap_rejects_unknown_input_features(fitted_model_and_background):
    model, background = fitted_model_and_background
    with pytest.raises(ExplanationError, match="Schéma incompatible"):
        ShapExplanationService(background).explain_one(model, {"balance": 1, "unknown": 2})


class MixedTypeClassifier:
    feature_names_in_ = np.array(["balance", "segment"])
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        score = 1 / (1 + np.exp(-(frame["balance"].astype(float) / 5 + (frame["segment"] == "B") * 0.5)))
        return np.column_stack([1 - score, score])


def test_shap_supports_mixed_numeric_and_categorical_features():
    background = pd.DataFrame({"balance": [-3.0, -1.0, 1.0, 3.0], "segment": ["A", "B", "A", "B"]})
    service = ShapExplanationService(background, max_global_rows=3, max_evals=5)
    model = MixedTypeClassifier()

    local = service.explain_one(model, {"balance": 2.0, "segment": "B"})
    global_result = service.explain_global(model)

    assert set(local["shap_values"]) == {"balance", "segment"}
    assert abs(local["additivity_residual"]) < 1e-6
    assert set(global_result["mean_absolute_shap"]) == {"balance", "segment"}


def test_tree_pipeline_uses_tree_explainer_and_groups_one_hot_features():
    frame = pd.DataFrame({
        "usage": [0.05, 0.12, 0.22, 0.31, 0.48, 0.57, 0.74, 0.88, 0.93, 0.99],
        "region": ["north", "south", "north", "west", "south", "east", "west", "east", "north", "south"],
    })
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    model = build_baseline_pipeline(frame, classifier_name="decision_tree", classifier_params={"max_depth": 3})
    model.fit(frame, labels)
    service = ShapExplanationService(frame, max_background_rows=8, max_global_rows=6)

    local = service.explain_one(model, {"usage": 0.2, "region": "north"}, class_index=1)
    global_result = service.explain_global(model, class_index=1)

    assert local["explainer_method"] == "TreeExplainer"
    assert set(local["shap_values"]) == set(frame.columns)
    assert abs(local["additivity_residual"]) < 1e-5
    assert base64.b64decode(local["waterfall_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert global_result["explainer_method"] == "TreeExplainer"
    assert set(global_result["mean_absolute_shap"]) == set(frame.columns)
    assert base64.b64decode(global_result["summary_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert base64.b64decode(global_result["beeswarm_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
