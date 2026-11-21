from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project35.registry.service import register_run_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Enregistre un run terminé dans le Model Registry MLflow.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--alias", choices=("champion", "challenger"))
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--mlflow-config", default="configs/mlflow.yaml")
    parser.add_argument("--description", default=None)
    parser.add_argument("--gate-report-id", default=None, help="Rapport MLflow PASS requis pour déplacer un champion existant.")
    args = parser.parse_args()
    config_path = Path(args.mlflow_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    tracking_uri = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or (config or {}).get("tracking_uri")
    model_name = args.model_name or (config or {}).get("registered_model_name", "project35-classifier")
    result = register_run_model(
        args.run_id,
        model_name,
        alias=args.alias,
        tracking_uri=tracking_uri,
        description=args.description,
        gate_report_id=args.gate_report_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
