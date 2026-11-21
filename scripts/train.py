from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.preflight_data import preflight
from project35.data.contract import DataContract, DataContractError
from project35.training.trainer import train_and_log


def main() -> int:
    parser = argparse.ArgumentParser(description="Entraîne la baseline et trace le run dans MLflow.")
    parser.add_argument("--data-config", default="configs/data_contract.yaml")
    parser.add_argument("--provenance", default="configs/data_provenance.yaml")
    parser.add_argument("--mlflow-config", default="configs/mlflow.yaml")
    parser.add_argument("--run-name", default="baseline-logistic-regression")
    parser.add_argument(
        "--classifier",
        choices=("logistic_regression", "random_forest", "extra_trees", "decision_tree"),
        default="logistic_regression",
    )
    parser.add_argument("--classifier-params", default="{}", help="Paramètres estimator en objet JSON.")
    args = parser.parse_args()
    try:
        contract = DataContract.from_yaml(args.data_config)
        readiness = preflight(args.data_config, args.provenance, require_approved=True)
        mlflow_path = Path(args.mlflow_config)
        mlflow_config = yaml.safe_load(mlflow_path.read_text(encoding="utf-8")) if mlflow_path.is_file() else {}
        result = train_and_log(
            contract,
            tracking_uri=(mlflow_config or {}).get("tracking_uri"),
            experiment_name=(mlflow_config or {}).get("experiment_name", "project35-classification"),
            run_name=args.run_name,
            classifier_name=args.classifier,
            classifier_params=json.loads(args.classifier_params),
            data_provenance=readiness["provenance"],
        )
    except (DataContractError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
