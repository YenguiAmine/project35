from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from project35.data.contract import DataContract, DataContractError, load_csv, validate_frame
from project35.data.phase1 import build_splits

PROVENANCE_FIELDS = (
    "dataset_name",
    "source_url",
    "source_version",
    "retrieved_on",
    "dataset_sha256",
    "license_name",
    "license_url",
    "approval_status",
    "approved_by",
    "target_column",
    "target_definition",
    "positive_class",
    "positive_class_definition",
    "split_policy",
    "sensitive_fields_reviewed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_provenance(path: str | Path, *, require_approved: bool) -> dict[str, Any]:
    provenance_path = Path(path)
    if not provenance_path.is_file():
        raise DataContractError(f"Fiche provenance introuvable: {provenance_path}")
    try:
        provenance = yaml.safe_load(provenance_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise DataContractError(f"Fiche provenance YAML invalide: {exc}") from exc
    if not isinstance(provenance, dict):
        raise DataContractError("La fiche provenance doit contenir un objet YAML.")
    missing = [field for field in PROVENANCE_FIELDS if field not in provenance]
    placeholder = [
        field for field in PROVENANCE_FIELDS
        if isinstance(provenance.get(field), str)
        and (not provenance[field].strip() or "REPLACE_" in provenance[field])
    ]
    if missing or placeholder:
        raise DataContractError(f"Fiche provenance incomplète (missing={missing}, placeholders={placeholder}).")
    try:
        date.fromisoformat(str(provenance["retrieved_on"]))
    except ValueError as exc:
        raise DataContractError("retrieved_on doit être une date ISO YYYY-MM-DD.") from exc
    if provenance["approval_status"] not in {"pending", "approved"}:
        raise DataContractError("approval_status doit être pending ou approved.")
    if require_approved and provenance["approval_status"] != "approved":
        raise DataContractError("L'entraînement métier exige une provenance avec approval_status=approved.")
    if provenance["sensitive_fields_reviewed"] is not True:
        raise DataContractError("Faire documenter la revue des champs sensibles avant de continuer.")
    if not isinstance(provenance.get("redistribution_allowed"), bool):
        raise DataContractError("redistribution_allowed doit être explicitement true ou false.")
    return provenance


def preflight(
    config_path: str | Path,
    provenance_path: str | Path,
    *,
    require_approved: bool = False,
) -> dict[str, Any]:
    contract = DataContract.from_yaml(config_path)
    provenance = load_provenance(provenance_path, require_approved=require_approved)
    if provenance["target_column"] != contract.target_column:
        raise DataContractError("target_column du contrat et de la provenance ne correspondent pas.")
    frame = load_csv(contract.dataset_path)
    dataset_hash = sha256_file(contract.dataset_path)
    expected_hash = provenance.get("dataset_sha256")
    if expected_hash is not None and (
        not isinstance(expected_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        or expected_hash != dataset_hash
    ):
        raise DataContractError("dataset_sha256 de la provenance est invalide ou ne correspond pas au CSV.")
    if require_approved and expected_hash is None:
        raise DataContractError("La provenance approuvée doit contenir le dataset_sha256 validé.")
    quality = validate_frame(
        frame,
        target_column=contract.target_column,
        required_columns=contract.required_columns,
    )
    class_labels = list(frame[contract.target_column].dropna().unique())
    if contract.f1_average == "binary":
        if len(class_labels) != 2:
            raise DataContractError("f1_average=binary exige exactement deux classes dans le dataset.")
        if contract.positive_class is None:
            raise DataContractError("Renseigner positive_class dans le contrat binaire.")
        if contract.positive_class not in class_labels:
            raise DataContractError("positive_class du contrat est absent de la cible observée.")
    elif len(class_labels) < 2:
        raise DataContractError("La cible doit avoir au moins deux classes.")
    elif len(class_labels) > 2 and contract.positive_class is not None:
        raise DataContractError("positive_class doit rester null pour une cible multiclasse.")
    if str(provenance["positive_class"]) != str(contract.positive_class):
        raise DataContractError("positive_class du contrat et de la provenance ne correspondent pas.")

    splits = build_splits(frame, contract)
    split_summary = {}
    for name in ("train", "validation", "test"):
        labels = getattr(splits, f"y_{name}")
        split_summary[name] = {
            "rows": int(len(labels)),
            "class_counts": {str(key): int(value) for key, value in labels.value_counts(dropna=False).items()},
        }
        if set(labels.unique()) != set(class_labels):
            raise DataContractError(f"Le split {name} ne contient pas toutes les classes de la cible.")

    return {
        "status": "ready_for_review",
        "dataset_path": str(contract.dataset_path.resolve()),
        "dataset_sha256": dataset_hash,
        "provenance_file": str(Path(provenance_path).resolve()),
        "approval_status": provenance["approval_status"],
        "provenance": {
            "dataset_name": provenance["dataset_name"],
            "source_url": provenance["source_url"],
            "source_version": provenance["source_version"],
            "dataset_sha256": dataset_hash,
            "license_name": provenance["license_name"],
            "license_url": provenance["license_url"],
            "redistribution_allowed": provenance["redistribution_allowed"],
        },
        "target_column": contract.target_column,
        "target_classes": [str(label) for label in class_labels],
        "split_strategy": contract.split_strategy,
        "random_state": contract.random_state,
        "quality": quality,
        "splits": split_summary,
        "feature_columns_after_id_date_handling": splits.X_train.columns.tolist(),
        "training_approved": bool(require_approved and provenance["approval_status"] == "approved"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Contrôle données, provenance, cible, hash et faisabilité des splits sans entraîner de modèle."
    )
    parser.add_argument("--config", default="configs/data_contract.yaml")
    parser.add_argument("--provenance", default="configs/data_provenance.yaml")
    parser.add_argument("--require-approved", action="store_true")
    parser.add_argument("--output", default="reports/preflight/data_readiness.json")
    args = parser.parse_args()
    try:
        report = preflight(args.config, args.provenance, require_approved=args.require_approved)
    except (DataContractError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Rapport de pré-validation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
