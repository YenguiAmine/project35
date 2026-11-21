from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier

from project35.data.contract import DataContract, DataContractError, load_csv
from project35.data.phase1 import build_baseline_pipeline, build_splits


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _dataset_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_metrics(model, X: pd.DataFrame, y: pd.Series, contract: DataContract) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    predictions = model.predict(X)
    classes = np.asarray(model.named_steps["classifier"].classes_)
    probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    average = contract.f1_average
    if average == "binary":
        if len(classes) != 2:
            raise DataContractError("f1_average=binary exige exactement deux classes.")
        positive_class = contract.positive_class
        if positive_class is None:
            raise DataContractError("Renseigner positive_class pour calculer les métriques binaires.")
        if positive_class not in classes:
            raise DataContractError(f"positive_class={positive_class!r} absent des classes de la cible.")
        pos_index = int(np.flatnonzero(classes == positive_class)[0])
        positive_scores = probabilities[:, pos_index] if probabilities is not None else None
        metrics = {
            "f1": float(f1_score(y, predictions, average="binary", pos_label=positive_class, zero_division=0)),
            "precision": float(precision_score(y, predictions, average="binary", pos_label=positive_class, zero_division=0)),
            "recall": float(recall_score(y, predictions, average="binary", pos_label=positive_class, zero_division=0)),
            "accuracy": float(accuracy_score(y, predictions)),
        }
        if positive_scores is not None:
            y_binary = (y.to_numpy() == positive_class).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(y_binary, positive_scores))
            metrics["average_precision"] = float(average_precision_score(y_binary, positive_scores))
        return metrics, predictions, positive_scores if positive_scores is not None else np.array([])

    metrics = {
        "f1": float(f1_score(y, predictions, average=average, zero_division=0)),
        "precision": float(precision_score(y, predictions, average=average, zero_division=0)),
        "recall": float(recall_score(y, predictions, average=average, zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
    }
    if probabilities is not None:
        if len(classes) == 2:
            y_binary = (y.to_numpy() == classes[1]).astype(int)
            positive_scores = probabilities[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_binary, positive_scores))
            metrics["average_precision"] = float(average_precision_score(y_binary, positive_scores))
            return metrics, predictions, positive_scores
        metrics["roc_auc"] = float(roc_auc_score(y, probabilities, multi_class="ovr", average="weighted", labels=classes))
    return metrics, predictions, probabilities if probabilities is not None else np.array([])


def _log_evaluation_plots(y_true: pd.Series, predictions: np.ndarray, scores: np.ndarray, output_dir: Path, positive_class: Any = None) -> None:
    labels = np.unique(y_true)
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(display, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Prédit", ylabel="Réel", title="Matrice de confusion — validation")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    if len(labels) == 2 and scores.size:
        positive = positive_class if positive_class in labels else labels[1]
        y_binary = (y_true.to_numpy() == positive).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, scores)
        precision, recall, _ = precision_recall_curve(y_binary, scores)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label="ROC")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Référence aléatoire")
        ax.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="Courbe ROC — validation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curve.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, label="Precision-Recall")
        ax.set(xlabel="Recall", ylabel="Precision", title="Courbe Precision-Recall — validation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "precision_recall_curve.png", dpi=140)
        plt.close(fig)


def train_and_log(
    contract: DataContract,
    *,
    tracking_uri: str | None = None,
    experiment_name: str = "project35-classification",
    run_name: str = "baseline-logistic-regression",
    registered_model_name: str | None = None,
    classifier_name: str = "logistic_regression",
    classifier_params: dict[str, Any] | None = None,
    data_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train on train split, score on validation, and log a fully reproducible MLflow run."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    frame = load_csv(contract.dataset_path)
    if data_provenance is not None:
        expected_dataset_hash = data_provenance.get("dataset_sha256")
        if expected_dataset_hash != _dataset_digest(contract.dataset_path):
            raise DataContractError("Le dataset a changé depuis le préflight approuvé; relancer la validation.")
    splits = build_splits(frame, contract)
    model = build_baseline_pipeline(
        splits.X_train,
        classifier_name=classifier_name,
        classifier_params=classifier_params,
        random_state=contract.random_state,
    )
    model.fit(splits.X_train, splits.y_train)
    metrics, predictions, scores = _validation_metrics(model, splits.X_validation, splits.y_validation, contract)

    with tempfile.TemporaryDirectory(prefix="project35-mlflow-") as tmp:
        artifact_dir = Path(tmp)
        _log_evaluation_plots(splits.y_validation, predictions, scores, artifact_dir, contract.positive_class)
        report = {
            "dataset_sha256": _dataset_digest(contract.dataset_path),
            "estimator": classifier_name,
            "estimator_parameters": (model.named_steps["classifier"].get_params(deep=False)),
            "rows": len(frame),
            "features": splits.X_train.columns.tolist(),
            "split_rows": {"train": len(splits.X_train), "validation": len(splits.X_validation), "test": len(splits.X_test)},
            "validation_metrics": metrics,
            "f1_average": contract.f1_average,
            "positive_class": contract.positive_class,
            "data_provenance": data_provenance,
        }
        report_path = artifact_dir / "evaluation_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        input_example = splits.X_validation.head(min(5, len(splits.X_validation)))
        signature = infer_signature(input_example, model.predict(input_example))
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({
                "estimator": classifier_name,
                "random_state": contract.random_state,
                "split_strategy": contract.split_strategy,
                "test_size": contract.test_size,
                "validation_size": contract.validation_size,
                "f1_average": contract.f1_average,
                "positive_class": str(contract.positive_class),
                "preprocessing": "ColumnTransformer(num: median+StandardScaler; cat: most_frequent+OneHotEncoder(ignore))",
            })
            for parameter_name, parameter_value in model.named_steps["classifier"].get_params(deep=False).items():
                mlflow.log_param(
                    f"classifier.{parameter_name}",
                    json.dumps(parameter_value, sort_keys=True, ensure_ascii=False, default=str),
                )
            mlflow.log_metrics(metrics)
            mlflow.set_tags({
                "dataset.sha256": report["dataset_sha256"],
                "dataset.rows": str(len(frame)),
                "dataset.target": contract.target_column,
                "python.version": platform.python_version(),
                "pandas.version": _package_version("pandas"),
                "sklearn.version": _package_version("scikit-learn"),
                "mlflow.version": _package_version("mlflow"),
                "evaluation.split": "validation",
            })
            mlflow.log_dict(report, "reports/evaluation_report.json")
            mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")
            trusted_model_types = ["numpy.dtype"]
            # The estimator is trained in this workflow, so trust only sklearn's
            # tree storage type when one of our supported tree ensembles is used.
            if isinstance(
                model.named_steps["classifier"],
                (DecisionTreeClassifier, ExtraTreesClassifier, RandomForestClassifier),
            ):
                trusted_model_types.append("sklearn.tree._tree.Tree")
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="baseline_pipeline",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                skops_trusted_types=trusted_model_types,
            )
            return {"run_id": run.info.run_id, "experiment_id": run.info.experiment_id, "metrics": metrics, "model_uri": model_info.model_uri, "dataset_sha256": report["dataset_sha256"]}
