from pathlib import Path

import pandas as pd
import pytest

from project35.data.contract import DataContract, DataContractError, validate_frame
from project35.data.phase1 import build_baseline_pipeline, build_splits


def test_quality_report_captures_missing_values_and_duplicates(sample_frame):
    frame = pd.concat([sample_frame, sample_frame.iloc[[0]]], ignore_index=True)
    frame.loc[1, "age"] = None
    report = validate_frame(frame, target_column="target", required_columns=("age", "segment"))
    assert report["rows"] == 5
    assert report["duplicate_rows"] == 1
    assert report["missing_by_column"]["age"] == 1
    assert report["target_distribution"] == {"0": 3, "1": 2}
    assert report["target_balance"]["minority_share"] == 0.4
    assert report["feature_profile"]["age"]["numeric_summary"]["50%"] == 33
    assert report["feature_profile"]["segment"]["unique_count"] == 3


def test_quality_report_accepts_boolean_features(sample_frame):
    sample_frame["auto_renew"] = [True, False, True, False]
    report = validate_frame(sample_frame, target_column="target")
    assert report["feature_profile"]["auto_renew"]["dtype"] == "bool"
    assert "numeric_summary" not in report["feature_profile"]["auto_renew"]


def test_quality_report_rejects_target_with_nulls(sample_frame):
    sample_frame.loc[0, "target"] = None
    with pytest.raises(DataContractError, match="cible contient des valeurs manquantes"):
        validate_frame(sample_frame, target_column="target")


def test_split_keeps_target_out_of_features(sample_frame):
    frame = pd.concat([sample_frame] * 12, ignore_index=True)
    contract = DataContract(dataset_path=Path("unused"), target_column="target", random_state=35)
    splits = build_splits(frame, contract)
    assert "target" not in splits.X_train.columns
    assert len(splits.X_train) + len(splits.X_validation) + len(splits.X_test) == len(frame)


def test_baseline_pipeline_fits_numeric_and_categorical_features(sample_frame):
    frame = pd.concat([sample_frame] * 8, ignore_index=True)
    X, y = frame.drop(columns="target"), frame["target"]
    model = build_baseline_pipeline(X)
    model.fit(X, y)
    predictions = model.predict(pd.DataFrame([{"age": 36, "segment": "NEW_CATEGORY"}]))
    assert len(predictions) == 1


def test_tree_candidate_uses_same_preprocessing_pipeline(sample_frame):
    frame = pd.concat([sample_frame] * 8, ignore_index=True)
    X, y = frame.drop(columns="target"), frame["target"]
    model = build_baseline_pipeline(
        X,
        classifier_name="random_forest",
        classifier_params={"n_estimators": 10, "max_depth": 3},
        random_state=35,
    )
    model.fit(X, y)
    assert len(model.predict(pd.DataFrame([{"age": 36, "segment": "NEW_CATEGORY"}]))) == 1


def test_pipeline_rejects_unapproved_estimator(sample_frame):
    with pytest.raises(DataContractError, match="classifier_name"):
        build_baseline_pipeline(sample_frame.drop(columns="target"), classifier_name="unapproved")
