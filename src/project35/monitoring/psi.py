from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class PSIError(ValueError):
    """Raised when reference/current monitoring data violate the contract."""


_MISSING = "special:missing"
_OTHER = "special:other"


@dataclass
class FeatureReference:
    kind: str
    edges: list[float] | None = None
    minimum: float | None = None
    maximum: float | None = None
    categories: list[str] | None = None
    reference_counts: dict[str, int] | None = None


@dataclass
class PSIProfile:
    """Immutable-in-use reference bins and proportions learned from a baseline window."""

    features: dict[str, FeatureReference]
    reference_rows: int
    reference_sha256: str
    bins: int = 10
    epsilon: float = 1e-6
    watch_threshold: float = 0.10
    drift_threshold: float = 0.25

    @classmethod
    def fit(
        cls,
        reference: pd.DataFrame,
        *,
        bins: int = 10,
        epsilon: float = 1e-6,
        watch_threshold: float = 0.10,
        drift_threshold: float = 0.25,
    ) -> "PSIProfile":
        if reference.empty or len(reference.columns) == 0:
            raise PSIError("La fenêtre de référence doit contenir des lignes et des features.")
        if reference.columns.duplicated().any():
            raise PSIError("La référence contient des noms de colonnes dupliqués.")
        if bins < 2:
            raise PSIError("bins doit être supérieur ou égal à 2.")
        if epsilon <= 0 or not np.isfinite(epsilon):
            raise PSIError("epsilon doit être strictement positif et fini.")
        if (
            not np.isfinite(watch_threshold)
            or not np.isfinite(drift_threshold)
            or watch_threshold < 0
            or drift_threshold <= watch_threshold
        ):
            raise PSIError("Les seuils doivent vérifier 0 <= watch_threshold < drift_threshold.")

        profiles: dict[str, FeatureReference] = {}
        for column in reference.columns:
            values = reference[column]
            is_numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
            if is_numeric:
                numeric = pd.to_numeric(values, errors="coerce")
                valid = numeric.dropna().to_numpy(dtype=float)
                if not np.isfinite(valid).all():
                    raise PSIError(f"La feature numérique {column} contient une valeur infinie.")
                if len(valid):
                    quantiles = np.quantile(valid, np.linspace(0.0, 1.0, bins + 1)[1:-1])
                    edges = np.unique(quantiles).astype(float).tolist()
                else:
                    edges = []
                profile = FeatureReference(
                    kind="numeric",
                    edges=edges,
                    minimum=float(valid.min()) if len(valid) else None,
                    maximum=float(valid.max()) if len(valid) else None,
                )
            else:
                categories = sorted({f"value:{value}" for value in values.dropna().map(str)})
                profile = FeatureReference(kind="categorical", categories=categories)
            counts = cls._counts(values, profile)
            profile.reference_counts = counts
            profiles[str(column)] = profile

        digest = hashlib.sha256(reference.to_csv(index=False).encode("utf-8")).hexdigest()
        return cls(
            features=profiles,
            reference_rows=len(reference),
            reference_sha256=digest,
            bins=bins,
            epsilon=float(epsilon),
            watch_threshold=float(watch_threshold),
            drift_threshold=float(drift_threshold),
        )

    @staticmethod
    def _labels(profile: FeatureReference) -> list[str]:
        if profile.kind == "numeric":
            return ["underflow", *[f"bin:{i}" for i in range(len(profile.edges or []) + 1)], "overflow", _MISSING]
        return [*(profile.categories or []), _OTHER, _MISSING]

    @classmethod
    def _counts(cls, values: pd.Series, profile: FeatureReference) -> dict[str, int]:
        labels = cls._labels(profile)
        counts = dict.fromkeys(labels, 0)
        missing = values.isna()
        if profile.kind == "numeric":
            numeric = pd.to_numeric(values, errors="coerce")
            invalid = (~missing) & numeric.isna()
            if invalid.any():
                raise PSIError("Une valeur non numérique apparaît dans une feature numérique.")
            finite_values = numeric.loc[~missing].to_numpy(dtype=float)
            if not np.isfinite(finite_values).all():
                raise PSIError("Les features numériques ne peuvent contenir l'infini.")
            if len(finite_values):
                edges = np.asarray(profile.edges or [], dtype=float)
                indices = np.searchsorted(edges, finite_values, side="right")
                for value, index in zip(finite_values, indices):
                    if profile.minimum is not None and value < profile.minimum:
                        bucket = "underflow"
                    elif profile.maximum is not None and value > profile.maximum:
                        bucket = "overflow"
                    else:
                        bucket = f"bin:{int(index)}"
                    counts[bucket] += 1
        else:
            known = set(profile.categories or [])
            for value in values.loc[~missing].map(str):
                key = f"value:{value}"
                counts[key if key in known else _OTHER] += 1
        counts[_MISSING] += int(missing.sum())
        return counts

    @staticmethod
    def _proportions(counts: dict[str, int], epsilon: float) -> dict[str, float]:
        denominator = sum(counts.values()) + epsilon * len(counts)
        return {label: (count + epsilon) / denominator for label, count in counts.items()}

    def calculate(
        self,
        current: pd.DataFrame,
        *,
        current_window: str | None = None,
        observed_at: str | None = None,
    ) -> dict[str, Any]:
        if current.empty:
            raise PSIError("La fenêtre courante doit contenir au moins une ligne.")
        expected, actual = set(self.features), set(str(c) for c in current.columns)
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        if missing or extra:
            raise PSIError(f"Schéma courant incompatible (missing={missing}, unknown={extra}).")

        feature_results: dict[str, dict[str, Any]] = {}
        for name, profile in self.features.items():
            counts = self._counts(current[name], profile)
            reference = self._proportions(profile.reference_counts or {}, self.epsilon)
            population = self._proportions(counts, self.epsilon)
            contribution = {
                label: (population[label] - reference[label]) * np.log(population[label] / reference[label])
                for label in reference
            }
            psi = float(sum(contribution.values()))
            if psi > self.drift_threshold:
                status = "drift"
            elif psi >= self.watch_threshold:
                status = "watch"
            else:
                status = "stable"
            feature_results[name] = {
                "psi": psi,
                "status": status,
                "reference_proportions": reference,
                "current_proportions": population,
                "contributions": contribution,
                "bin_definition": (
                    {
                        "kind": "numeric_quantile_bins",
                        "internal_edges": profile.edges or [],
                        "reference_min": profile.minimum,
                        "reference_max": profile.maximum,
                        "special_bins": ["underflow", "overflow", _MISSING],
                    }
                    if profile.kind == "numeric"
                    else {
                        "kind": "categorical_reference_values",
                        "reference_categories": profile.categories or [],
                        "special_bins": [_OTHER, _MISSING],
                    }
                ),
            }

        severity = {"stable": 0, "watch": 1, "drift": 2}
        overall_status = max((row["status"] for row in feature_results.values()), key=severity.get)
        current_digest = hashlib.sha256(current.to_csv(index=False).encode("utf-8")).hexdigest()
        return {
            "report_id": str(uuid.uuid4()),
            "reference_sha256": self.reference_sha256,
            "current_sha256": current_digest,
            "reference_rows": self.reference_rows,
            "current_rows": int(len(current)),
            "reference_window": "configured_reference",
            "current_window": current_window,
            "observed_at": observed_at,
            "bins": self.bins,
            "epsilon": self.epsilon,
            "thresholds": {"watch": self.watch_threshold, "drift": self.drift_threshold},
            "status": overall_status,
            "alert": overall_status == "drift",
            "features": feature_results,
        }
