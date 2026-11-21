from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


class DataContractError(ValueError):
    """Raised when data or its configuration violates the declared contract."""


@dataclass(frozen=True)
class DataContract:
    dataset_path: Path
    target_column: str
    positive_class: Any = None
    f1_average: str = "macro"
    id_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 35
    split_strategy: str = "stratified"

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "DataContract":
        path = Path(config_path)
        if not path.is_file():
            raise DataContractError(f"Configuration introuvable: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        target, dataset = raw.get("target_column"), raw.get("dataset_path")
        if not target or str(target).startswith("REPLACE_"):
            raise DataContractError("Renseigner target_column depuis l'énoncé officiel.")
        if not dataset or "REPLACE_" in str(dataset):
            raise DataContractError("Renseigner dataset_path avec le dataset officiel.")
        test_size, validation_size = float(raw.get("test_size", 0.2)), float(raw.get("validation_size", 0.2))
        if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
            raise DataContractError("test_size et validation_size doivent être positifs et leur somme < 1.")
        strategy = str(raw.get("split_strategy", "stratified"))
        if strategy not in {"stratified", "chronological"}:
            raise DataContractError("split_strategy accepte stratified ou chronological.")
        f1_average = str(raw.get("f1_average", "macro"))
        if f1_average not in {"binary", "macro", "weighted", "micro"}:
            raise DataContractError("f1_average accepte binary, macro, weighted ou micro.")
        root = path.resolve().parent.parent
        dataset_path = Path(dataset)
        if not dataset_path.is_absolute():
            dataset_path = root / dataset_path
        return cls(
            dataset_path=dataset_path,
            target_column=str(target),
            positive_class=raw.get("positive_class"),
            f1_average=f1_average,
            id_columns=tuple(raw.get("id_columns") or ()),
            date_columns=tuple(raw.get("date_columns") or ()),
            required_columns=tuple(raw.get("required_columns") or ()),
            test_size=test_size,
            validation_size=validation_size,
            random_state=int(raw.get("random_state", 35)),
            split_strategy=strategy,
        )


def load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise DataContractError(f"Dataset introuvable: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise DataContractError(f"Lecture CSV impossible: {exc}") from exc
    validate_frame(frame)
    return frame


def validate_frame(frame: pd.DataFrame, *, target_column: str | None = None, required_columns: tuple[str, ...] = ()) -> dict[str, Any]:
    if frame.empty:
        raise DataContractError("Le dataset ne contient aucune ligne.")
    if len(frame.columns) == 0:
        raise DataContractError("Le dataset ne contient aucune colonne.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise DataContractError(f"Noms de colonnes dupliqués: {duplicates}")
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise DataContractError(f"Colonnes obligatoires absentes: {missing}")
    if target_column:
        if target_column not in frame.columns:
            raise DataContractError(f"Cible absente: {target_column}")
        if frame[target_column].isna().any():
            raise DataContractError("La cible contient des valeurs manquantes.")
        if frame[target_column].nunique(dropna=True) < 2:
            raise DataContractError("La cible doit contenir au moins deux classes.")
    feature_columns = [column for column in frame.columns if column != target_column]
    feature_profile: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        values = frame[column]
        profile: dict[str, Any] = {
            "dtype": str(values.dtype),
            "unique_count": int(values.nunique(dropna=True)),
            "missing_count": int(values.isna().sum()),
            "missing_rate": float(values.isna().mean()),
        }
        profile["constant_or_all_missing"] = profile["unique_count"] <= 1
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            numeric = values.describe(percentiles=[0.25, 0.5, 0.75])
            profile["numeric_summary"] = {
                name: (None if pd.isna(numeric[name]) else float(numeric[name]))
                for name in ("min", "25%", "50%", "mean", "75%", "max", "std")
            }
        feature_profile[str(column)] = profile
    target_distribution = (
        {str(k): int(v) for k, v in frame[target_column].value_counts(dropna=False).items()}
        if target_column
        else None
    )
    target_balance = None
    if target_distribution:
        counts = list(target_distribution.values())
        total = sum(counts)
        target_balance = {
            "class_proportions": {label: count / total for label, count in target_distribution.items()},
            "minority_share": min(counts) / total,
            "majority_to_minority_ratio": max(counts) / min(counts),
        }
    return {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_by_column": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "dtypes": {str(k): str(v) for k, v in frame.dtypes.items()},
        "target_distribution": target_distribution,
        "target_balance": target_balance,
        "feature_profile": feature_profile,
    }
