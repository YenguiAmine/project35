from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.preflight_data import preflight, sha256_file
from project35.data.contract import DataContract, DataContractError
from project35.data.phase1 import run_phase1


def main() -> int:
    parser = argparse.ArgumentParser(description="Valide les données et entraîne la baseline de phase 1.")
    parser.add_argument("--config", default="configs/data_contract.yaml")
    parser.add_argument("--provenance", default="configs/data_provenance.yaml")
    parser.add_argument("--report", default="reports/phase1/data_quality.json")
    args = parser.parse_args()
    try:
        readiness = preflight(args.config, args.provenance, require_approved=True)
        contract = DataContract.from_yaml(args.config)
        result = run_phase1(contract)
        if sha256_file(contract.dataset_path) != readiness["dataset_sha256"]:
            raise DataContractError("Le dataset a changé depuis le préflight approuvé; relancer la validation.")
    except (DataContractError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "data_readiness": readiness,
        "quality": result["quality"],
        "split_rows": result["split_rows"],
        "feature_columns": result["feature_columns"],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
