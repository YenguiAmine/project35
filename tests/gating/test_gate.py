import numpy as np
import pandas as pd
import pytest

from project35.gating.service import GateError, compare_models


class BinaryClassifier:
    classes_ = np.array([0, 1])
    feature_names_in_ = np.array(["score"])

    def __init__(self, good=False):
        self.good = good

    def predict_proba(self, frame):
        if self.good:
            positive = np.where(frame["score"].to_numpy() >= 2, 0.95, 0.05)
        else:
            positive = np.full(len(frame), 0.5)
        return np.column_stack([1 - positive, positive])

    def predict(self, frame):
        return (self.predict_proba(frame)[:, 1] > 0.5).astype(int)


@pytest.fixture
def evaluation():
    return pd.DataFrame({"score": [0, 1, 2, 3], "target": [0, 0, 1, 1]})


def aliases():
    return (
        {"version": "1", "run_id": "champion-run"},
        {"version": "2", "run_id": "challenger-run"},
    )


def test_gate_passes_only_when_f1_and_roc_auc_both_improve(evaluation):
    champion, challenger = aliases()
    report = compare_models(
        BinaryClassifier(),
        BinaryClassifier(good=True),
        evaluation,
        model_name="test-model",
        champion=champion,
        challenger=challenger,
        target_column="target",
        f1_average="binary",
        positive_class=1,
        evaluation_sha256="a" * 64,
    )

    assert report["decision"] == "PASS"
    assert report["criteria"]["f1_strict_improvement"]["passed"] is True
    assert report["criteria"]["roc_auc_strict_improvement"]["passed"] is True
    assert report["metrics"]["challenger"]["f1"] == 1.0


def test_equal_or_worse_challenger_fails_and_preserves_both_deltas(evaluation):
    champion, challenger = aliases()
    report = compare_models(
        BinaryClassifier(),
        BinaryClassifier(),
        evaluation,
        model_name="test-model",
        champion=champion,
        challenger=challenger,
        target_column="target",
        f1_average="binary",
        positive_class=1,
        evaluation_sha256="b" * 64,
    )

    assert report["decision"] == "FAIL"
    assert report["metrics"]["delta"] == {"f1": 0.0, "roc_auc": 0.0}


def test_configured_minimum_improvements_are_enforced(evaluation):
    champion, challenger = aliases()
    report = compare_models(
        BinaryClassifier(),
        BinaryClassifier(good=True),
        evaluation,
        model_name="test-model",
        champion=champion,
        challenger=challenger,
        target_column="target",
        f1_average="binary",
        positive_class=1,
        min_f1_improvement=1.0,
        evaluation_sha256="c" * 64,
    )
    assert report["decision"] == "FAIL"
    assert report["criteria"]["f1_strict_improvement"]["passed"] is False


def test_gate_rejects_non_matching_schema_and_same_version(evaluation):
    champion, challenger = aliases()
    with pytest.raises(GateError, match="Schéma champion incompatible"):
        compare_models(
            BinaryClassifier(),
            BinaryClassifier(good=True),
            evaluation.rename(columns={"score": "other"}),
            model_name="test-model",
            champion=champion,
            challenger=challenger,
            target_column="target",
            f1_average="binary",
            positive_class=1,
            evaluation_sha256="e" * 64,
        )


def test_gate_requires_evaluation_hash_and_identified_psi_trigger(evaluation):
    champion, challenger = aliases()
    shared = dict(
        model_name="test-model",
        champion=champion,
        challenger=challenger,
        target_column="target",
        f1_average="binary",
        positive_class=1,
    )
    with pytest.raises(GateError, match="SHA-256"):
        compare_models(BinaryClassifier(), BinaryClassifier(good=True), evaluation, **shared)
    with pytest.raises(GateError, match="psi_report_id"):
        compare_models(
            BinaryClassifier(), BinaryClassifier(good=True), evaluation,
            evaluation_sha256="d" * 64,
            trigger="psi_drift",
            **shared,
        )

    with pytest.raises(GateError, match="versions distinctes"):
        compare_models(
            BinaryClassifier(),
            BinaryClassifier(good=True),
            evaluation,
            model_name="test-model",
            champion=champion,
            challenger={**challenger, "version": champion["version"]},
            target_column="target",
            f1_average="binary",
            positive_class=1,
            evaluation_sha256="f" * 64,
        )
