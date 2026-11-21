from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project35.data.contract import DataContract, DataContractError, load_csv, validate_frame


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def build_splits(frame: pd.DataFrame, contract: DataContract) -> DatasetSplits:
    validate_frame(frame, target_column=contract.target_column, required_columns=contract.required_columns)
    excluded_columns = {contract.target_column, *contract.id_columns}
    source_feature_columns = [c for c in frame.columns if c not in excluded_columns]
    if not source_feature_columns:
        raise DataContractError("Aucune feature disponible après exclusion de la cible et des identifiants.")
    X, y = frame.loc[:, source_feature_columns].copy(), frame[contract.target_column].copy()
    parsed_dates = {}
    for date_column in contract.date_columns:
        if date_column not in X.columns:
            raise DataContractError(f"Colonne de date absente: {date_column}")
        parsed = pd.to_datetime(X[date_column], errors="coerce")
        if contract.split_strategy == "chronological" and parsed.isna().any():
            raise DataContractError(f"Dates manquantes ou invalides dans {date_column}.")
        parsed_dates[date_column] = parsed
        # Raw timestamps should not become one-hot categories; derive stable calendar features.
        X[f"{date_column}__year"] = parsed.dt.year
        X[f"{date_column}__month"] = parsed.dt.month
        X[f"{date_column}__day"] = parsed.dt.day
        X[f"{date_column}__weekday"] = parsed.dt.dayofweek
        X = X.drop(columns=[date_column])
    if contract.split_strategy == "chronological":
        if not contract.date_columns:
            raise DataContractError("Le split chronologique exige au moins une date dans date_columns.")
        date_col = contract.date_columns[0]
        order = parsed_dates[date_col]
        indices = order.sort_values(kind="stable").index
        test_n, val_n = max(1, round(len(indices) * contract.test_size)), max(1, round(len(indices) * contract.validation_size))
        train_idx = indices[: len(indices) - test_n - val_n]
        val_idx = indices[len(indices) - test_n - val_n : len(indices) - test_n]
        test_idx = indices[len(indices) - test_n :]
        if len(train_idx) < 2 or len(val_idx) == 0:
            raise DataContractError("Trop peu de lignes pour les splits configurés.")
        return DatasetSplits(X.loc[train_idx], X.loc[val_idx], X.loc[test_idx], y.loc[train_idx], y.loc[val_idx], y.loc[test_idx])
    stratify = y if y.value_counts().min() >= 3 else None
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=contract.test_size, random_state=contract.random_state, stratify=stratify)
        relative_validation_size = contract.validation_size / (1 - contract.test_size)
        second_stratify = y_train_val if y_train_val.value_counts().min() >= 2 else None
        X_train, X_validation, y_train, y_validation = train_test_split(X_train_val, y_train_val, test_size=relative_validation_size, random_state=contract.random_state, stratify=second_stratify)
    except ValueError as exc:
        raise DataContractError(f"Split impossible avec la taille/distribution des classes: {exc}") from exc
    return DatasetSplits(X_train, X_validation, X_test, y_train, y_validation, y_test)


def build_baseline_pipeline(
    X_train: pd.DataFrame,
    *,
    classifier_name: str = "logistic_regression",
    classifier_params: dict | None = None,
    random_state: int = 35,
) -> Pipeline:
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns))
    if categorical_columns:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns))
    if not transformers:
        raise DataContractError("Aucune feature numérique ou catégorielle exploitable.")
    parameters = dict(classifier_params or {})
    if classifier_name == "logistic_regression":
        defaults = {"max_iter": 1000, "random_state": random_state}
        defaults.update(parameters)
        classifier = LogisticRegression(**defaults)
    elif classifier_name == "random_forest":
        defaults = {"n_estimators": 200, "random_state": random_state, "n_jobs": -1}
        defaults.update(parameters)
        classifier = RandomForestClassifier(**defaults)
    elif classifier_name == "extra_trees":
        defaults = {"n_estimators": 200, "random_state": random_state, "n_jobs": -1}
        defaults.update(parameters)
        classifier = ExtraTreesClassifier(**defaults)
    elif classifier_name == "decision_tree":
        defaults = {"random_state": random_state}
        defaults.update(parameters)
        classifier = DecisionTreeClassifier(**defaults)
    else:
        raise DataContractError(
            "classifier_name doit être logistic_regression, random_forest, extra_trees ou decision_tree."
        )
    # Keep preprocessing inside the fitted estimator to reuse it identically at serving time.
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("classifier", classifier),
    ])


def run_phase1(contract: DataContract) -> dict:
    frame = load_csv(contract.dataset_path)
    quality = validate_frame(frame, target_column=contract.target_column, required_columns=contract.required_columns)
    splits = build_splits(frame, contract)
    model = build_baseline_pipeline(splits.X_train, random_state=contract.random_state)
    model.fit(splits.X_train, splits.y_train)
    return {"quality": quality, "split_rows": {"train": len(splits.X_train), "validation": len(splits.X_validation), "test": len(splits.X_test)}, "feature_columns": splits.X_train.columns.tolist(), "baseline_pipeline": model, "splits": splits}
