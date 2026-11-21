from __future__ import annotations

import os
import hashlib
from functools import lru_cache
from pathlib import Path

import pandas as pd

from project35.explainability.shap_service import ExplanationError, ShapExplanationService


@lru_cache(maxsize=4)
def _service_for(path_text: str, mtime_ns: int) -> ShapExplanationService:
    del mtime_ns  # included in the cache key so edits reload the reference data
    path = Path(path_text)
    if not path.is_file():
        raise ExplanationError("Le fichier de référence SHAP configuré est introuvable.")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ExplanationError("Impossible de lire le fichier de référence SHAP.") from exc
    service = ShapExplanationService(
        frame,
        max_background_rows=int(os.getenv("SHAP_MAX_BACKGROUND_ROWS", "40")),
        max_global_rows=int(os.getenv("SHAP_MAX_GLOBAL_ROWS", "50")),
        max_evals=int(os.getenv("SHAP_MAX_EVALS", "0")) or None,
    )
    service.reference_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return service


def get_explanation_service() -> ShapExplanationService:
    configured = os.getenv("PROJECT35_SHAP_BACKGROUND")
    if not configured:
        raise ExplanationError("PROJECT35_SHAP_BACKGROUND doit désigner un CSV de référence.")
    path = Path(configured).expanduser().resolve()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return _service_for(str(path), mtime_ns)
