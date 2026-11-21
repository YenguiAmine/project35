from __future__ import annotations

import json
import math
import re
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException

from project35.data.contract import DataContractError

ALLOWED_ALIASES = {"champion", "challenger"}


class RegistryGovernanceError(RuntimeError):
    """Raised when a registry operation could bypass the alias governance rules."""


def _client(tracking_uri: str | None = None) -> MlflowClient:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri or mlflow.get_tracking_uri())


def register_run_model(
    run_id: str,
    model_name: str,
    *,
    alias: str | None = None,
    tracking_uri: str | None = None,
    description: str | None = None,
    gate_report_id: str | None = None,
) -> dict[str, Any]:
    """Register the `baseline_pipeline` artifact from a completed MLflow run."""
    if alias is not None and alias not in ALLOWED_ALIASES:
        raise RegistryGovernanceError(f"Alias invalide: {alias}. Valeurs permises: {sorted(ALLOWED_ALIASES)}")
    client = _client(tracking_uri)
    try:
        run = client.get_run(run_id)
    except MlflowException as exc:
        raise DataContractError(f"Run MLflow introuvable: {run_id}") from exc
    status = RunStatus.from_string(run.info.status) if isinstance(run.info.status, str) else run.info.status
    if status != RunStatus.FINISHED:
        raise RegistryGovernanceError(f"Le run {run_id} doit être FINISHED avant l'enregistrement.")

    model_uri = f"runs:/{run_id}/baseline_pipeline"
    try:
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    except MlflowException as exc:
        raise RegistryGovernanceError(f"Enregistrement impossible pour {model_uri}: {exc}") from exc

    client.set_model_version_tag(model_name, model_version.version, "source_run_id", run_id)
    client.set_model_version_tag(model_name, model_version.version, "validation_status", "pending")
    if description:
        client.update_model_version(model_name, model_version.version, description=description)
    if alias:
        assign_alias(
            model_name,
            model_version.version,
            alias,
            tracking_uri=tracking_uri,
            gate_report_id=gate_report_id,
        )
    return {
        "name": model_name,
        "version": str(model_version.version),
        "run_id": run_id,
        "source": model_uri,
        "alias": alias,
    }


def assign_alias(
    model_name: str,
    version: str | int,
    alias: str,
    *,
    tracking_uri: str | None = None,
    gate_report_id: str | None = None,
) -> dict[str, str]:
    """Assign an alias; moving an existing champion requires a persisted passing gate report."""
    if alias not in ALLOWED_ALIASES:
        raise RegistryGovernanceError(f"Alias invalide: {alias}. Valeurs permises: {sorted(ALLOWED_ALIASES)}")
    client = _client(tracking_uri)
    try:
        model = client.get_registered_model(model_name)
    except MlflowException as exc:
        raise DataContractError(f"Modèle enregistré introuvable: {model_name}") from exc
    current_version = (model.aliases or {}).get(alias)
    requested_version = str(version)
    try:
        candidate = client.get_model_version(model_name, requested_version)
    except MlflowException as exc:
        raise DataContractError(f"Version {requested_version} absente pour {model_name}") from exc

    verified_report = None
    if alias == "champion" and current_version and current_version != requested_version:
        verified_report = _verify_promotion_gate(
            client,
            model_name=model_name,
            challenger_version=requested_version,
            champion_version=str(current_version),
            gate_report_id=gate_report_id,
        )

    client.set_registered_model_alias(model_name, alias, requested_version)
    if verified_report is not None:
        client.set_model_version_tag(model_name, requested_version, "promotion_gate_report_id", gate_report_id)
        client.set_model_version_tag(model_name, requested_version, "validation_status", "approved")
        promotion_event = {
            "gate_report_id": gate_report_id,
            "decision": "PROMOTED",
            "model_name": model_name,
            "previous_champion_version": str(current_version),
            "new_champion_version": requested_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with tempfile.TemporaryDirectory(prefix="project35-promotion-") as temp_dir:
            local = f"{temp_dir}/promotion.json"
            with open(local, "w", encoding="utf-8") as stream:
                json.dump(promotion_event, stream, indent=2, ensure_ascii=False)
            client.log_artifact(candidate.run_id, local, artifact_path=f"promotion_gates/{gate_report_id}")
    return {"name": model_name, "alias": alias, "version": requested_version}


def _verify_promotion_gate(
    client: MlflowClient,
    *,
    model_name: str,
    challenger_version: str,
    champion_version: str,
    gate_report_id: str | None,
) -> dict[str, Any]:
    if not gate_report_id:
        raise RegistryGovernanceError(
            f"L'alias @champion pointe vers {champion_version}; un rapport de gate PASS vérifiable est requis."
        )
    try:
        parsed_id = uuid.UUID(gate_report_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise RegistryGovernanceError("gate_report_id doit être l'UUID d'un rapport MLflow persisté.") from exc

    try:
        candidate = client.get_model_version(model_name, challenger_version)
        artifact_path = client.download_artifacts(
            candidate.run_id,
            f"promotion_gates/{parsed_id}.json",
        )
        with open(artifact_path, encoding="utf-8") as stream:
            report = json.load(stream)
    except Exception as exc:
        raise RegistryGovernanceError("Rapport de gate introuvable ou illisible dans le run challenger.") from exc

    expected = {
        "gate_report_id": str(parsed_id),
        "model_name": model_name,
        "champion_version": champion_version,
        "challenger_version": challenger_version,
        "candidate_run_id": candidate.run_id,
        "decision": "PASS",
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    criteria = report.get("criteria") or {}
    if not isinstance(criteria, dict):
        criteria = {}
    metrics = report.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    f1_criteria = criteria.get("f1_strict_improvement")
    auc_criteria = criteria.get("roc_auc_strict_improvement")
    if not isinstance(f1_criteria, dict):
        f1_criteria = {}
    if not isinstance(auc_criteria, dict):
        auc_criteria = {}
    dataset_hash = report.get("evaluation_dataset_sha256")
    try:
        numbers_valid = all(
            math.isfinite(float(metrics[alias][metric]))
            for alias in ("champion", "challenger")
            for metric in ("f1", "roc_auc")
        )
        evaluation_rows = int(report.get("evaluation_rows") or 0)
        f1_delta = float(metrics["challenger"]["f1"]) - float(metrics["champion"]["f1"])
        auc_delta = float(metrics["challenger"]["roc_auc"]) - float(metrics["champion"]["roc_auc"])
        f1_minimum = float(f1_criteria["minimum_delta"])
        auc_minimum = float(auc_criteria["minimum_delta"])
    except (KeyError, TypeError, ValueError):
        numbers_valid = False
        evaluation_rows = 0
        f1_delta = auc_delta = f1_minimum = auc_minimum = float("nan")
    if (
        mismatches
        or not f1_criteria.get("passed")
        or not auc_criteria.get("passed")
        or not criteria.get("same_evaluation_population")
        or not criteria.get("same_feature_order")
        or not criteria.get("same_class_order")
        or not isinstance(dataset_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", dataset_hash)
        or evaluation_rows < 1
        or not numbers_valid
        or not math.isfinite(f1_delta)
        or not math.isfinite(auc_delta)
        or not math.isfinite(f1_minimum)
        or not math.isfinite(auc_minimum)
        or f1_delta <= f1_minimum
        or auc_delta <= auc_minimum
    ):
        raise RegistryGovernanceError("Le rapport de gate ne prouve pas une promotion valide pour ces versions.")
    return report


def resolve_alias(model_name: str, alias: str, *, tracking_uri: str | None = None) -> dict[str, Any]:
    if alias not in ALLOWED_ALIASES:
        raise RegistryGovernanceError(f"Alias invalide: {alias}. Valeurs permises: {sorted(ALLOWED_ALIASES)}")
    client = _client(tracking_uri)
    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as exc:
        raise DataContractError(f"Alias @{alias} non défini pour {model_name}") from exc
    run = client.get_run(model_version.run_id)
    return {
        "name": model_name,
        "alias": alias,
        "version": str(model_version.version),
        "run_id": model_version.run_id,
        "source": model_version.source,
        "metrics": dict(run.data.metrics),
        "tags": dict(model_version.tags),
    }


def load_alias_model(model_name: str, alias: str = "champion", *, tracking_uri: str | None = None):
    """Load a registry model through the mutable `models:/name@alias` URI."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    resolve_alias(model_name, alias, tracking_uri=tracking_uri)
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
