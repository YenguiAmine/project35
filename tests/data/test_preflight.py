import hashlib

import pandas as pd
import pytest
import yaml

from project35.data.contract import DataContractError
from scripts.preflight_data import preflight


def _write_inputs(tmp_path, *, approval_status="approved"):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    (root / "data").mkdir()
    frame = pd.DataFrame(
        {
            "account_id": range(80),
            "amount": [float(i * 10) for i in range(80)],
            "segment": ["business" if i % 2 else "standard" for i in range(80)],
            "target": [i % 2 for i in range(80)],
        }
    )
    frame.to_csv(root / "data" / "customers.csv", index=False)
    dataset_hash = hashlib.sha256((root / "data" / "customers.csv").read_bytes()).hexdigest()
    contract = {
        "dataset_path": "data/customers.csv",
        "target_column": "target",
        "positive_class": 1,
        "f1_average": "binary",
        "id_columns": ["account_id"],
        "required_columns": ["amount", "segment", "target"],
        "split_strategy": "stratified",
        "random_state": 35,
    }
    (root / "configs" / "data_contract.yaml").write_text(yaml.safe_dump(contract), encoding="utf-8")
    provenance = {
        "dataset_name": "approved-test-fixture",
        "source_url": "https://example.test/data",
        "source_version": "fixture-v1",
        "retrieved_on": "2026-11-20",
        "dataset_sha256": dataset_hash,
        "license_name": "test-only",
        "license_url": "https://example.test/license",
        "approval_status": approval_status,
        "approved_by": "test owner",
        "target_column": "target",
        "target_definition": "Synthetic target for tests only",
        "positive_class": 1,
        "positive_class_definition": "1 means positive in the fixture",
        "split_policy": "stratified, rows are independent fixture observations",
        "sensitive_fields_reviewed": True,
        "redistribution_allowed": False,
    }
    provenance_path = root / "configs" / "data_provenance.yaml"
    provenance_path.write_text(yaml.safe_dump(provenance), encoding="utf-8")
    return root / "configs" / "data_contract.yaml", provenance_path


def test_preflight_reports_hash_quality_classes_and_splits(tmp_path):
    contract_path, provenance_path = _write_inputs(tmp_path)
    report = preflight(contract_path, provenance_path, require_approved=True)

    assert report["status"] == "ready_for_review"
    assert report["training_approved"] is True
    assert len(report["dataset_sha256"]) == 64
    assert report["provenance"]["dataset_sha256"] == report["dataset_sha256"]
    assert report["quality"]["rows"] == 80
    assert report["target_classes"] == ["0", "1"]
    assert report["splits"]["train"]["rows"] + report["splits"]["validation"]["rows"] + report["splits"]["test"]["rows"] == 80
    assert "account_id" not in report["feature_columns_after_id_date_handling"]


def test_preflight_rejects_unapproved_provenance_for_training(tmp_path):
    contract_path, provenance_path = _write_inputs(tmp_path, approval_status="pending")
    with pytest.raises(DataContractError, match="approval_status=approved"):
        preflight(contract_path, provenance_path, require_approved=True)


def test_preflight_rejects_mismatched_target_in_provenance(tmp_path):
    contract_path, provenance_path = _write_inputs(tmp_path)
    provenance = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
    provenance["target_column"] = "other_target"
    provenance_path.write_text(yaml.safe_dump(provenance), encoding="utf-8")

    with pytest.raises(DataContractError, match="ne correspondent pas"):
        preflight(contract_path, provenance_path)


def test_preflight_rejects_dataset_changed_after_provenance_hash(tmp_path):
    contract_path, provenance_path = _write_inputs(tmp_path)
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    csv_path = contract_path.parents[1] / contract["dataset_path"]
    with csv_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(DataContractError, match="dataset_sha256"):
        preflight(contract_path, provenance_path, require_approved=True)
