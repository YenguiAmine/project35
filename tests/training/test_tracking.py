import numpy as np
import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from rest_framework.test import APIClient
from unittest.mock import patch
from sklearn.dummy import DummyClassifier

from project35.data.contract import DataContract
from project35.gating.service import evaluate_registered_aliases, log_gate_report
from project35.registry.service import (
    RegistryGovernanceError,
    assign_alias,
    load_alias_model,
    register_run_model,
    resolve_alias,
)
from project35.training.trainer import train_and_log
from predictions.services import ChampionModelProvider


def test_training_logs_metrics_signature_and_artifacts(tmp_path):
    rng = np.random.default_rng(35)
    rows = 100
    spend = rng.normal(500, 100, rows)
    segment = rng.choice(["basic", "plus", "premium"], rows)
    target = (spend > 500).astype(int)
    csv_path = tmp_path / "synthetic_test_only.csv"
    pd.DataFrame({"spend": spend, "segment": segment, "target": target}).to_csv(csv_path, index=False)

    contract = DataContract(
        dataset_path=csv_path,
        target_column="target",
        positive_class=1,
        f1_average="binary",
        random_state=35,
    )
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("phase2-test", artifact_location=(tmp_path / "artifacts").as_uri())
    train_features = pd.DataFrame({"spend": spend, "segment": segment})
    dummy = DummyClassifier(strategy="most_frequent").fit(train_features, target)
    input_example = train_features.head(2)
    with mlflow.start_run(experiment_id=experiment_id, run_name="weak-champion") as dummy_run:
        mlflow.sklearn.log_model(
            sk_model=dummy,
            name="baseline_pipeline",
            signature=infer_signature(input_example, dummy.predict(input_example)),
            input_example=input_example,
        )
        dummy_run_id = dummy_run.info.run_id

    evaluation_rng = np.random.default_rng(99)
    eval_spend = evaluation_rng.normal(500, 100, 120)
    eval_segment = evaluation_rng.choice(["basic", "plus", "premium"], 120)
    eval_target = (eval_spend > 500).astype(int)
    evaluation_path = tmp_path / "independent_evaluation.csv"
    evaluation = pd.DataFrame({"spend": eval_spend, "segment": eval_segment, "target": eval_target})
    evaluation.to_csv(evaluation_path, index=False)
    result = train_and_log(
        contract,
        tracking_uri=tracking_uri,
        experiment_name="phase2-test",
        run_name="pytest-baseline",
    )

    assert result["run_id"]
    assert result["metrics"]["f1"] >= 0
    assert "roc_auc" in result["metrics"]
    run = client.get_run(result["run_id"])
    assert run.data.params["f1_average"] == "binary"
    assert run.data.tags["dataset.sha256"] == result["dataset_sha256"]
    paths = {item.path for item in client.list_artifacts(result["run_id"], "evaluation")}
    assert "evaluation/evaluation_report.json" in paths
    assert "evaluation/confusion_matrix.png" in paths
    model_files = {item.path for item in client.list_artifacts(result["run_id"], "baseline_pipeline")}
    assert "baseline_pipeline/MLmodel" in model_files

    champion = register_run_model(dummy_run_id, "project35-test", alias="champion", tracking_uri=tracking_uri)
    challenger = register_run_model(result["run_id"], "project35-test", alias="challenger", tracking_uri=tracking_uri)
    assert champion["version"] == "1"
    assert challenger["version"] == "2"
    assert resolve_alias("project35-test", "champion", tracking_uri=tracking_uri)["version"] == "1"
    with pytest.raises(RegistryGovernanceError, match="rapport de gate PASS"):
        assign_alias("project35-test", challenger["version"], "champion", tracking_uri=tracking_uri)
    loaded = load_alias_model("project35-test", "champion", tracking_uri=tracking_uri)
    assert len(loaded.predict(pd.DataFrame([{"spend": 550.0, "segment": "premium"}]))) == 1
    evaluation_sha256 = __import__("hashlib").sha256(evaluation_path.read_bytes()).hexdigest()
    passing_gate = evaluate_registered_aliases(
        "project35-test",
        evaluation,
        target_column="target",
        f1_average="binary",
        positive_class=1,
        tracking_uri=tracking_uri,
        evaluation_sha256=evaluation_sha256,
    )
    assert passing_gate["decision"] == "PASS"
    log_gate_report(passing_gate, tracking_uri=tracking_uri)
    promoted = assign_alias(
        "project35-test",
        challenger["version"],
        "champion",
        tracking_uri=tracking_uri,
        gate_report_id=passing_gate["gate_report_id"],
    )
    assert promoted["version"] == "2"

    failed_candidate = register_run_model(
        dummy_run_id, "project35-test", alias="challenger", tracking_uri=tracking_uri
    )
    failed_gate = evaluate_registered_aliases(
        "project35-test",
        evaluation,
        target_column="target",
        f1_average="binary",
        positive_class=1,
        tracking_uri=tracking_uri,
        evaluation_sha256=evaluation_sha256,
    )
    assert failed_candidate["version"] == "3"
    assert failed_gate["decision"] == "FAIL"
    log_gate_report(failed_gate, tracking_uri=tracking_uri)
    with pytest.raises(RegistryGovernanceError, match="ne prouve pas une promotion valide"):
        assign_alias(
            "project35-test",
            failed_candidate["version"],
            "champion",
            tracking_uri=tracking_uri,
            gate_report_id=failed_gate["gate_report_id"],
        )
    assert resolve_alias("project35-test", "champion", tracking_uri=tracking_uri)["version"] == "2"

    # Exercise the actual registry -> provider -> Django REST prediction path.
    provider = ChampionModelProvider(
        model_name="project35-test", alias="champion", tracking_uri=tracking_uri
    )
    with patch("predictions.views.model_provider", provider):
        response = APIClient().post(
            "/api/predict/",
            {"instances": [{"spend": 550.0, "segment": "premium"}]},
            format="json",
        )
    assert response.status_code == 200
    assert response.data["model_version"] == "2"
    assert len(response.data["prediction"]) == 1
    assert len(response.data["probabilities"][0]) == 2
