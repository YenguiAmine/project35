from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.preflight_data import preflight
from project35.data.contract import DataContract, DataContractError
from project35.gating.service import (
    GateError,
    evaluate_registered_aliases,
    log_gate_report,
)
from project35.registry.service import RegistryGovernanceError, assign_alias, register_run_model
from project35.training.trainer import train_and_log


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Entraîne un challenger, le compare au champion et ne promeut qu'avec un gate PASS vérifiable."
    )
    parser.add_argument("--data-config", default="configs/data_contract.yaml")
    parser.add_argument("--provenance", default="configs/data_provenance.yaml")
    parser.add_argument("--gate-config", default="configs/gate.yaml")
    parser.add_argument("--mlflow-config", default="configs/mlflow.yaml")
    parser.add_argument("--evaluation-data", default=None, help="Holdout indépendant labellisé, jamais utilisé au fit.")
    parser.add_argument("--psi-report", default=None, help="Si fourni, le rapport doit avoir alert=true et status=drift.")
    parser.add_argument("--run-name", default="challenger-candidate")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    try:
        contract = DataContract.from_yaml(args.data_config)
        readiness = preflight(args.data_config, args.provenance, require_approved=True)
        gate_path = Path(args.gate_config)
        if not gate_path.is_file():
            parser.error(f"Configuration gate absente: {gate_path}; copiez configs/gate.example.yaml vers configs/gate.yaml.")
        gate = yaml.safe_load(gate_path.read_text(encoding="utf-8")) or {}
        mlflow_path = Path(args.mlflow_config)
        mlflow_config = yaml.safe_load(mlflow_path.read_text(encoding="utf-8")) if mlflow_path.is_file() else {}
        tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or (mlflow_config or {}).get("tracking_uri")
        model_name = gate.get("model_name") or (mlflow_config or {}).get("registered_model_name", "project35-classifier")
        evaluation_path = args.evaluation_data or gate.get("evaluation_dataset_path")
        if not evaluation_path or "REPLACE_" in str(evaluation_path):
            raise GateError("Configurer evaluation_dataset_path dans la configuration gate.")

        trigger = "manual"
        psi_report_id = None
        if args.psi_report:
            psi_path = Path(args.psi_report)
            psi_report = json.loads(psi_path.read_text(encoding="utf-8"))
            if psi_report.get("status") != "drift" or psi_report.get("alert") is not True:
                raise GateError("Le workflow PSI ne démarre que sur un rapport status=drift avec alert=true.")
            trigger = "psi_drift"
            psi_report_id = psi_report.get("report_id")
            if not psi_report_id:
                raise GateError("Le rapport PSI déclencheur doit inclure report_id.")

        training = train_and_log(
            contract,
            tracking_uri=tracking_uri,
            experiment_name=(mlflow_config or {}).get("experiment_name", "project35-classification"),
            run_name=args.run_name,
            classifier_name=gate.get("challenger_classifier", "logistic_regression"),
            classifier_params=gate.get("challenger_params") or {},
            data_provenance=readiness["provenance"],
        )
        candidate = register_run_model(
            training["run_id"],
            model_name,
            alias="challenger",
            tracking_uri=tracking_uri,
            description=f"Challenger déclenché par {trigger}; run MLflow {training['run_id']}.",
        )
        report = evaluate_registered_aliases(
            model_name,
            pd.read_csv(evaluation_path),
            target_column=contract.target_column,
            f1_average=contract.f1_average,
            positive_class=contract.positive_class,
            min_f1_improvement=float(gate.get("min_f1_improvement", 0.0)),
            min_roc_auc_improvement=float(gate.get("min_roc_auc_improvement", 0.0)),
            tracking_uri=tracking_uri,
            evaluation_sha256=hashlib.sha256(Path(evaluation_path).read_bytes()).hexdigest(),
            trigger=trigger,
            psi_report_id=psi_report_id,
        )
        log_gate_report(report, tracking_uri=tracking_uri)
        if report["decision"] == "PASS":
            promotion = assign_alias(
                model_name,
                candidate["version"],
                "champion",
                tracking_uri=tracking_uri,
                gate_report_id=report["gate_report_id"],
            )
            report["promotion"] = {"status": "PROMOTED", **promotion}
        else:
            report["promotion"] = {"status": "NOT_PROMOTED", "reason": "One or more configured criteria failed."}

        report_dir = Path(gate.get("report_dir", "reports/gates"))
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{report['gate_report_id']}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except (DataContractError, GateError, RegistryGovernanceError, OSError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Rapport gate: {report_path}")
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
