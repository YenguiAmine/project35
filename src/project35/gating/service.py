from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import f1_score, roc_auc_score

from project35.registry.service import resolve_alias


class GateError(ValueError):
    """The comparison cannot safely be performed with the supplied inputs."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(model: Any, frame: pd.DataFrame, y_true: pd.Series, *, f1_average: str, positive_class: Any) -> dict[str, float]:
    classes = np.asarray(getattr(model, "classes_", []))
    if len(classes) < 2 or not callable(getattr(model, "predict_proba", None)):
        raise GateError("Chaque modèle doit avoir au moins deux classes et predict_proba.")
    predictions = np.asarray(model.predict(frame))
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    if predictions.shape != (len(frame),) or probabilities.shape != (len(frame), len(classes)):
        raise GateError("Le modèle a retourné des dimensions de prédiction invalides.")
    if not np.isfinite(probabilities).all() or ((probabilities < -1e-8) | (probabilities > 1 + 1e-8)).any():
        raise GateError("Les probabilités contiennent des valeurs invalides.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise GateError("Les probabilités de classe ne somment pas à 1.")

    if f1_average == "binary":
        if len(classes) != 2:
            raise GateError("f1_average=binary exige un problème à deux classes.")
        if positive_class is None or positive_class not in classes:
            raise GateError("positive_class doit être défini et présent dans les classes des modèles.")
        positive_index = int(np.flatnonzero(classes == positive_class)[0])
        y_binary = (y_true.to_numpy() == positive_class).astype(int)
        if np.unique(y_binary).size != 2:
            raise GateError("La fenêtre d'évaluation doit contenir les deux classes.")
        auc = roc_auc_score(y_binary, probabilities[:, positive_index])
        f1 = f1_score(y_true, predictions, average="binary", pos_label=positive_class, zero_division=0)
    else:
        if f1_average not in {"macro", "weighted", "micro"}:
            raise GateError("f1_average doit être binary, macro, weighted ou micro.")
        if len(classes) == 2:
            pos = positive_class if positive_class is not None else classes[1]
            if pos not in classes:
                raise GateError("La classe positive configurée est absente des modèles.")
            index = int(np.flatnonzero(classes == pos)[0])
            y_binary = (y_true.to_numpy() == pos).astype(int)
            auc = roc_auc_score(y_binary, probabilities[:, index])
        else:
            if positive_class is not None:
                raise GateError("positive_class n'est pas accepté pour ROC-AUC multiclasse.")
            auc = roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted", labels=classes)
        f1 = f1_score(y_true, predictions, average=f1_average, zero_division=0)
    return {"f1": float(f1), "roc_auc": float(auc)}


def _load_registry_model(model_name: str, version: str, tracking_uri: str | None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{model_name}/{version}")


def compare_models(
    champion_model: Any,
    challenger_model: Any,
    evaluation: pd.DataFrame,
    *,
    model_name: str,
    champion: dict[str, Any],
    challenger: dict[str, Any],
    target_column: str,
    f1_average: str,
    positive_class: Any = None,
    min_f1_improvement: float = 0.0,
    min_roc_auc_improvement: float = 0.0,
    evaluation_sha256: str | None = None,
    trigger: str = "manual",
    psi_report_id: str | None = None,
) -> dict[str, Any]:
    if evaluation.empty or evaluation.columns.duplicated().any():
        raise GateError("L'échantillon d'évaluation doit être non vide avec des colonnes uniques.")
    if target_column not in evaluation:
        raise GateError(f"Cible d'évaluation absente: {target_column}.")
    if evaluation[target_column].isna().any():
        raise GateError("La cible d'évaluation contient des valeurs manquantes.")
    if not math.isfinite(min_f1_improvement) or not math.isfinite(min_roc_auc_improvement):
        raise GateError("Les marges d'amélioration doivent être finies.")
    if min_f1_improvement < 0 or min_roc_auc_improvement < 0:
        raise GateError("Les marges d'amélioration ne peuvent pas être négatives.")
    if not isinstance(evaluation_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", evaluation_sha256):
        raise GateError("Le SHA-256 du jeu d'évaluation est requis pour un rapport de gate auditable.")
    if trigger not in {"manual", "psi_drift"}:
        raise GateError("trigger doit être manual ou psi_drift.")
    if trigger == "psi_drift" and not psi_report_id:
        raise GateError("Un déclenchement PSI exige psi_report_id.")

    X = evaluation.drop(columns=[target_column])
    y = evaluation[target_column]
    versions = (str(champion["version"]), str(challenger["version"]))
    if versions[0] == versions[1]:
        raise GateError("Champion et challenger doivent être deux versions distinctes.")
    model_features = []
    for alias, model in (("champion", champion_model), ("challenger", challenger_model)):
        names = getattr(model, "feature_names_in_", None)
        if names is None:
            raise GateError(f"Le modèle {alias} ne publie pas feature_names_in_.")
        expected = [str(name) for name in names]
        if expected != list(X.columns):
            missing = sorted(set(expected) - set(X.columns))
            extra = sorted(set(X.columns) - set(expected))
            raise GateError(f"Schéma {alias} incompatible (missing={missing}, unknown={extra}).")
        model_features.append(expected)
        classes = np.asarray(getattr(model, "classes_", []))
        if len(classes) < 2:
            raise GateError(f"Le modèle {alias} ne contient pas plusieurs classes.")
    if model_features[0] != model_features[1]:
        raise GateError("Champion et challenger n'ont pas le même ordre de features.")
    if not np.array_equal(np.asarray(champion_model.classes_), np.asarray(challenger_model.classes_)):
        raise GateError("Champion et challenger n'ont pas le même ordre de classes.")

    champion_metrics = _metrics(champion_model, X, y, f1_average=f1_average, positive_class=positive_class)
    challenger_metrics = _metrics(challenger_model, X, y, f1_average=f1_average, positive_class=positive_class)
    f1_delta = challenger_metrics["f1"] - champion_metrics["f1"]
    auc_delta = challenger_metrics["roc_auc"] - champion_metrics["roc_auc"]
    f1_pass = f1_delta > min_f1_improvement
    auc_pass = auc_delta > min_roc_auc_improvement
    report_id = str(uuid.uuid4())

    return {
        "gate_report_id": report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if f1_pass and auc_pass else "FAIL",
        "model_name": model_name,
        "champion_version": versions[0],
        "champion_run_id": champion["run_id"],
        "challenger_version": versions[1],
        "candidate_run_id": challenger["run_id"],
        "evaluation_dataset_sha256": evaluation_sha256,
        "trigger": trigger,
        "psi_report_id": psi_report_id,
        "evaluation_rows": int(len(evaluation)),
        "target_column": target_column,
        "f1_average": f1_average,
        "positive_class": positive_class,
        "features": list(X.columns),
        "metrics": {
            "champion": champion_metrics,
            "challenger": challenger_metrics,
            "delta": {"f1": f1_delta, "roc_auc": auc_delta},
        },
        "criteria": {
            "f1_strict_improvement": {
                "minimum_delta": min_f1_improvement,
                "passed": bool(f1_pass),
            },
            "roc_auc_strict_improvement": {
                "minimum_delta": min_roc_auc_improvement,
                "passed": bool(auc_pass),
            },
            "same_evaluation_population": True,
            "same_feature_order": True,
            "same_class_order": True,
        },
    }


def evaluate_registered_aliases(
    model_name: str,
    evaluation: pd.DataFrame,
    *,
    target_column: str,
    f1_average: str,
    positive_class: Any = None,
    min_f1_improvement: float = 0.0,
    min_roc_auc_improvement: float = 0.0,
    tracking_uri: str | None = None,
    evaluation_sha256: str | None = None,
    trigger: str = "manual",
    psi_report_id: str | None = None,
) -> dict[str, Any]:
    champion = resolve_alias(model_name, "champion", tracking_uri=tracking_uri)
    challenger = resolve_alias(model_name, "challenger", tracking_uri=tracking_uri)
    champion_model = _load_registry_model(model_name, champion["version"], tracking_uri)
    challenger_model = _load_registry_model(model_name, challenger["version"], tracking_uri)
    return compare_models(
        champion_model,
        challenger_model,
        evaluation,
        model_name=model_name,
        champion=champion,
        challenger=challenger,
        target_column=target_column,
        f1_average=f1_average,
        positive_class=positive_class,
        min_f1_improvement=min_f1_improvement,
        min_roc_auc_improvement=min_roc_auc_improvement,
        evaluation_sha256=evaluation_sha256,
        trigger=trigger,
        psi_report_id=psi_report_id,
    )


def log_gate_report(report: dict[str, Any], *, tracking_uri: str | None = None) -> str:
    """Persist the decision next to the challenger source run in MLflow artifacts."""
    client = MlflowClient(tracking_uri=tracking_uri) if tracking_uri else MlflowClient()
    candidate_run_id = report.get("candidate_run_id")
    gate_report_id = report.get("gate_report_id")
    if not candidate_run_id or not gate_report_id:
        raise GateError("Le rapport doit identifier le run candidat et le gate_report_id.")
    relative_path = f"promotion_gates/{gate_report_id}.json"
    with tempfile.TemporaryDirectory(prefix="project35-gate-") as temp_dir:
        local_path = Path(temp_dir) / f"{gate_report_id}.json"
        local_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        client.log_artifact(candidate_run_id, str(local_path), artifact_path="promotion_gates")
    client.set_tag(candidate_run_id, f"promotion_gate.{gate_report_id}.decision", report["decision"])
    return relative_path


def run_gate_from_csv(
    model_name: str,
    dataset_path: str | Path,
    *,
    target_column: str,
    f1_average: str,
    positive_class: Any = None,
    min_f1_improvement: float = 0.0,
    min_roc_auc_improvement: float = 0.0,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    path = Path(dataset_path)
    if not path.is_file():
        raise GateError(f"Fichier d'évaluation introuvable: {path}.")
    try:
        evaluation = pd.read_csv(path)
    except Exception as exc:
        raise GateError("Lecture du fichier d'évaluation impossible.") from exc
    if evaluation.empty:
        raise GateError("Le fichier d'évaluation ne contient aucune ligne.")
    return evaluate_registered_aliases(
        model_name,
        evaluation,
        target_column=target_column,
        f1_average=f1_average,
        positive_class=positive_class,
        min_f1_improvement=min_f1_improvement,
        min_roc_auc_improvement=min_roc_auc_improvement,
        tracking_uri=tracking_uri,
        evaluation_sha256=_hash_file(path),
    )
