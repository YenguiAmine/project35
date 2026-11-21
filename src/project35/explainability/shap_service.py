from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


class ExplanationError(ValueError):
    """The model or reference data cannot produce a valid explanation."""


@dataclass
class ShapExplanationService:
    """Bounded SHAP over original inputs, using TreeExplainer for sklearn trees.

    Tree pipelines are explained after their fitted preprocessing and encoded
    contributions are grouped back to the source feature. Other estimators use
    model-agnostic Permutation SHAP. Values explain class probability, not causality.
    """

    background: pd.DataFrame
    max_background_rows: int = 40
    max_global_rows: int = 50
    max_evals: int | None = None
    reference_sha256: str | None = None

    def __post_init__(self):
        if self.background.empty:
            raise ExplanationError("Le jeu de référence SHAP est vide.")
        if self.max_background_rows < 1 or self.max_global_rows < 1:
            raise ExplanationError("Les limites de calcul SHAP doivent être positives.")
        self.background = self.background.sample(
            n=min(len(self.background), self.max_background_rows), random_state=35
        ).reset_index(drop=True)

    @staticmethod
    def _columns(model: Any, frame: pd.DataFrame) -> list[str]:
        names = getattr(model, "feature_names_in_", None)
        columns = [str(name) for name in names] if names is not None else list(frame.columns)
        missing = sorted(set(columns) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(columns))
        if missing or extra:
            raise ExplanationError(f"Schéma incompatible (missing={missing}, unknown={extra}).")
        return columns

    @staticmethod
    def _probability_function(model: Any, columns: list[str], class_index: int, categories: dict[str, list[Any]]):
        classes = list(getattr(model, "classes_", []))
        if not callable(getattr(model, "predict_proba", None)):
            raise ExplanationError("Le modèle doit exposer predict_proba pour l'explication.")
        if class_index < 0 or class_index >= len(classes):
            raise ExplanationError("class_index ne correspond à aucune classe du modèle.")

        def predict_probability(values):
            frame = pd.DataFrame(values, columns=columns)
            # SHAP's tabular Independent masker uses numpy.isclose and numeric
            # masking. Encode categoricals for SHAP, then decode them before
            # calling the original sklearn pipeline.
            for name, values_for_name in categories.items():
                reverse = {float(index): value for index, value in enumerate(values_for_name)}
                indices = pd.to_numeric(frame[name], errors="raise").astype(float)
                if not indices.isin(reverse).all():
                    raise ExplanationError(f"Valeur catégorielle encodée inconnue pour {name}.")
                frame[name] = indices.map(reverse)
            result = np.asarray(model.predict_proba(frame), dtype=float)
            return result[:, class_index]

        return classes[class_index], predict_probability

    def _explainer(self, model: Any, columns: list[str], class_index: int, extra_frame: pd.DataFrame | None = None):
        background = self.background.loc[:, columns].copy()
        combined = pd.concat([background, extra_frame.loc[:, columns]], ignore_index=True) if extra_frame is not None else background
        categorical = [name for name in columns if not pd.api.types.is_numeric_dtype(combined[name].dtype)]
        categories = {name: list(pd.unique(combined[name])) for name in categorical}
        encoded_background = background.copy()
        for name, values in categories.items():
            mapping = {value: float(index) for index, value in enumerate(values)}
            encoded_background[name] = encoded_background[name].map(mapping).astype(float)
        encoded_background = encoded_background.astype(float)
        class_value, predict_probability = self._probability_function(model, columns, class_index, categories)
        explainer = shap.Explainer(predict_probability, shap.maskers.Independent(encoded_background))
        evals = self.max_evals or 2 * len(columns) + 1
        return class_value, predict_probability, explainer, evals, categories

    @staticmethod
    def _tree_context(model: Any, columns: list[str]):
        """Return a probability TreeExplainer and mapping for a fitted tree pipeline."""
        if not isinstance(model, Pipeline) or "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
            return None
        classifier = model.named_steps["classifier"]
        if not isinstance(classifier, (DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier)):
            return None
        preprocessor = model.named_steps["preprocessor"]
        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
        source_groups: dict[str, list[int]] = {name: [] for name in columns}
        output_indices = getattr(preprocessor, "output_indices_", {})
        for transformer_name, _, raw_columns in preprocessor.transformers_:
            if transformer_name == "remainder" or transformer_name not in output_indices:
                continue
            span = output_indices[transformer_name]
            start, stop = span.start or 0, span.stop or 0
            source_columns = [raw_columns] if isinstance(raw_columns, str) else list(raw_columns)
            for output_index in range(start, stop):
                encoded_name = feature_names[output_index].split("__", 1)[-1]
                matches = [name for name in source_columns if encoded_name == name or encoded_name.startswith(f"{name}_")]
                if not matches:
                    raise ExplanationError(f"Impossible de rattacher la feature transformée {encoded_name} à son origine.")
                source_groups[max(matches, key=len)].append(output_index)
        if any(not source_groups[name] for name in columns):
            raise ExplanationError("Le prétraitement arbre ne conserve pas toutes les features d'origine.")
        return classifier, preprocessor, feature_names, source_groups

    @staticmethod
    def _dense(values):
        return values.toarray() if hasattr(values, "toarray") else np.asarray(values)

    @staticmethod
    def _select_tree_class_values(values: Any, class_index: int) -> np.ndarray:
        if isinstance(values, list):
            if class_index >= len(values):
                raise ExplanationError("TreeExplainer n'a pas retourné la classe demandée.")
            return np.asarray(values[class_index], dtype=float)
        array = np.asarray(values, dtype=float)
        if array.ndim == 3:
            if class_index >= array.shape[-1]:
                raise ExplanationError("TreeExplainer n'a pas retourné la classe demandée.")
            return array[..., class_index]
        if array.ndim == 1:
            return array.reshape(1, -1)
        return array

    @staticmethod
    def _group_tree_values(values: np.ndarray, columns: list[str], groups: dict[str, list[int]]) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return np.column_stack([matrix[:, groups[name]].sum(axis=1) for name in columns])

    def _tree_local(self, model: Any, row: pd.DataFrame, columns: list[str], class_index: int) -> dict[str, Any] | None:
        context = self._tree_context(model, columns)
        if context is None:
            return None
        classifier, preprocessor, feature_names, groups = context
        transformed_background = self._dense(preprocessor.transform(self.background.loc[:, columns]))
        transformed_row = self._dense(preprocessor.transform(row.loc[:, columns]))
        explainer = shap.TreeExplainer(
            classifier,
            data=transformed_background,
            feature_perturbation="interventional",
            model_output="probability",
            feature_names=feature_names,
        )
        values = self._select_tree_class_values(explainer.shap_values(transformed_row, check_additivity=False), class_index)
        grouped_values = self._group_tree_values(values, columns, groups)[0]
        expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
        if class_index >= len(expected):
            raise ExplanationError("TreeExplainer n'a pas retourné la valeur de base de la classe demandée.")
        base_value = float(expected[class_index])
        prediction = float(model.predict_proba(row.loc[:, columns])[:, class_index][0])
        residual = prediction - (base_value + float(grouped_values.sum()))
        if not np.isfinite(grouped_values).all() or not np.isfinite(base_value):
            raise ExplanationError("TreeExplainer a produit des valeurs non finies.")
        exp = shap.Explanation(
            values=grouped_values,
            base_values=base_value,
            data=row.iloc[0].to_numpy(),
            feature_names=columns,
        )
        shap.plots.waterfall(exp, max_display=min(15, len(columns)), show=False)
        return {
            "class_index": class_index,
            "class_value": model.classes_[class_index].item() if isinstance(model.classes_[class_index], np.generic) else model.classes_[class_index],
            "output_scale": "probability",
            "explainer_method": "TreeExplainer",
            "reference_sha256": self.reference_sha256,
            "base_value": base_value,
            "prediction_probability": prediction,
            "additivity_residual": residual,
            "shap_values": {name: float(value) for name, value in zip(columns, grouped_values)},
            "waterfall_png_base64": self._png_base64(plt.gcf()),
            "interpretation": "Contribution Tree SHAP à la probabilité prédite, agrégée par feature source; non causale.",
        }

    def _tree_global(self, model: Any, class_index: int) -> dict[str, Any] | None:
        columns = self._columns(model, self.background)
        context = self._tree_context(model, columns)
        if context is None:
            return None
        classifier, preprocessor, feature_names, groups = context
        frame = self.background.loc[:, columns].head(self.max_global_rows).reset_index(drop=True)
        transformed_background = self._dense(preprocessor.transform(self.background.loc[:, columns]))
        transformed_frame = self._dense(preprocessor.transform(frame))
        explainer = shap.TreeExplainer(
            classifier,
            data=transformed_background,
            feature_perturbation="interventional",
            model_output="probability",
            feature_names=feature_names,
        )
        values = self._select_tree_class_values(explainer.shap_values(transformed_frame, check_additivity=False), class_index)
        grouped_values = self._group_tree_values(values, columns, groups)
        importance = np.mean(np.abs(grouped_values), axis=0)
        ordering = np.argsort(importance)[::-1]
        summary = {columns[i]: float(importance[i]) for i in ordering}

        fig, ax = plt.subplots(figsize=(9, max(3, min(12, 0.32 * len(columns) + 1))))
        positions = np.arange(len(ordering))
        ax.barh(positions, importance[ordering], color="#2463a6")
        ax.set_yticks(positions, [columns[i] for i in ordering])
        ax.invert_yaxis()
        ax.set_xlabel("mean(|SHAP value|) — classe demandée, probabilité")
        ax.set_title("Importance globale Tree SHAP par feature source")
        fig.tight_layout()
        summary_png = self._png_base64(fig)

        fig, ax = plt.subplots(figsize=(10, max(4, min(14, 0.4 * len(columns) + 1))))
        rng = np.random.default_rng(35)
        for position, feature_index in enumerate(ordering):
            jitter = rng.uniform(-0.22, 0.22, size=len(frame))
            ax.scatter(grouped_values[:, feature_index], position + jitter, s=13, alpha=0.65, color="#2463a6")
        ax.axvline(0, color="#555555", linewidth=0.8)
        ax.set_yticks(positions, [columns[i] for i in ordering])
        ax.invert_yaxis()
        ax.set_xlabel("Contribution SHAP à la probabilité de classe")
        ax.set_title("Distribution globale Tree SHAP (points = observations)")
        fig.tight_layout()
        beeswarm_png = self._png_base64(fig)
        classes = list(classifier.classes_)
        return {
            "class_index": class_index,
            "class_value": classes[class_index].item() if isinstance(classes[class_index], np.generic) else classes[class_index],
            "output_scale": "probability",
            "explainer_method": "TreeExplainer",
            "reference_sha256": self.reference_sha256,
            "rows_explained": len(frame),
            "mean_absolute_shap": summary,
            "summary_png_base64": summary_png,
            "beeswarm_png_base64": beeswarm_png,
            "interpretation": "Importance Tree SHAP regroupée par feature source; points individuels non causals.",
        }

    @staticmethod
    def _png_base64(fig) -> str:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def explain_one(self, model: Any, instance: dict[str, Any], class_index: int = 1) -> dict[str, Any]:
        row = pd.DataFrame([instance])
        columns = self._columns(model, row)
        row = row.loc[:, columns]
        tree_result = self._tree_local(model, row, columns, class_index)
        if tree_result is not None:
            return tree_result
        class_value, predict_probability, explainer, evals, categories = self._explainer(model, columns, class_index, row)
        encoded_row = row.copy()
        for name, values in categories.items():
            mapping = {value: float(index) for index, value in enumerate(values)}
            encoded_row[name] = encoded_row[name].map(mapping).astype(float)
        encoded_row = encoded_row.astype(float)
        result = explainer(encoded_row, max_evals=evals)
        values = np.asarray(result.values, dtype=float).reshape(-1)
        base_value = float(np.asarray(result.base_values).reshape(-1)[0])
        prediction = float(predict_probability(encoded_row.to_numpy())[0])
        residual = prediction - (base_value + float(values.sum()))
        if not np.isfinite(values).all() or not np.isfinite(base_value):
            raise ExplanationError("SHAP a produit des valeurs non finies.")

        exp = shap.Explanation(
            values=values,
            base_values=base_value,
            data=row.iloc[0].to_numpy(),
            feature_names=columns,
        )
        shap.plots.waterfall(exp, max_display=min(15, len(columns)), show=False)
        waterfall = self._png_base64(plt.gcf())
        return {
            "class_index": class_index,
            "class_value": class_value.item() if isinstance(class_value, np.generic) else class_value,
            "output_scale": "probability",
            "explainer_method": "PermutationExplainer",
            "reference_sha256": self.reference_sha256,
            "base_value": base_value,
            "prediction_probability": prediction,
            "additivity_residual": residual,
            "shap_values": {name: float(value) for name, value in zip(columns, values)},
            "waterfall_png_base64": waterfall,
            "interpretation": "Contribution à la probabilité prédite; une explication SHAP n'établit pas de causalité.",
        }

    def explain_global(self, model: Any, class_index: int = 1) -> dict[str, Any]:
        tree_result = self._tree_global(model, class_index)
        if tree_result is not None:
            return tree_result
        columns = self._columns(model, self.background)
        frame = self.background.loc[:, columns].head(self.max_global_rows).reset_index(drop=True)
        class_value, _, explainer, evals, categories = self._explainer(model, columns, class_index)
        encoded_frame = frame.copy()
        for name, values_for_name in categories.items():
            mapping = {value: float(index) for index, value in enumerate(values_for_name)}
            encoded_frame[name] = encoded_frame[name].map(mapping).astype(float)
        encoded_frame = encoded_frame.astype(float)
        result = explainer(encoded_frame, max_evals=evals)
        values = np.asarray(result.values, dtype=float)
        if values.ndim == 3:
            values = values[:, :, class_index]
        importance = np.mean(np.abs(values), axis=0)
        ordering = np.argsort(importance)[::-1]
        summary = {columns[i]: float(importance[i]) for i in ordering}

        plt.figure(figsize=(9, max(3, min(12, 0.32 * len(columns) + 1))))
        shap.summary_plot(values, encoded_frame, feature_names=columns, show=False, plot_type="bar", max_display=20)
        summary_png = self._png_base64(plt.gcf())
        plt.figure(figsize=(10, max(4, min(14, 0.4 * len(columns) + 1))))
        shap.summary_plot(values, encoded_frame, feature_names=columns, show=False, plot_type="dot", max_display=20)
        beeswarm_png = self._png_base64(plt.gcf())
        return {
            "class_index": class_index,
            "class_value": class_value.item() if isinstance(class_value, np.generic) else class_value,
            "output_scale": "probability",
            "explainer_method": "PermutationExplainer",
            "reference_sha256": self.reference_sha256,
            "rows_explained": len(frame),
            "mean_absolute_shap": summary,
            "summary_png_base64": summary_png,
            "beeswarm_png_base64": beeswarm_png,
            "interpretation": "Importance moyenne des contributions SHAP absolues; elle ne mesure pas un effet causal.",
        }
