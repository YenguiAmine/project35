from __future__ import annotations

import os
from threading import RLock
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

from project35.data.contract import DataContractError
from project35.registry.service import resolve_alias


class ModelUnavailable(RuntimeError):
    """The configured registry model or alias cannot currently be loaded."""


class ChampionModelProvider:
    """Resolve the registry alias on requests and reload only when its version changes."""

    def __init__(self, model_name: str | None = None, alias: str | None = None, tracking_uri: str | None = None):
        self.model_name = model_name or os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "project35-classifier")
        self.alias = alias or os.getenv("MLFLOW_MODEL_ALIAS", "champion")
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        self._model = None
        self._metadata: dict[str, Any] | None = None
        self._lock = RLock()

    def get_active(self):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            metadata = resolve_alias(self.model_name, self.alias, tracking_uri=self.tracking_uri)
        except (DataContractError, MlflowException, OSError) as exc:
            raise ModelUnavailable(f"Modèle @{self.alias} indisponible dans le registre.") from exc

        with self._lock:
            if self._model is None or self._metadata["version"] != metadata["version"]:
                model_uri = f"models:/{self.model_name}@{self.alias}"
                try:
                    candidate = mlflow.sklearn.load_model(model_uri)
                except Exception as exc:
                    # Keep the last good in-memory version, but fail this request instead of hiding a bad promotion.
                    raise ModelUnavailable(f"Impossible de charger {model_uri}.") from exc
                self._model = candidate
                self._metadata = metadata
            else:
                self._metadata = metadata
            return self._model, dict(self._metadata)

    def cached_metadata(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._metadata) if self._metadata else None


model_provider = ChampionModelProvider()
