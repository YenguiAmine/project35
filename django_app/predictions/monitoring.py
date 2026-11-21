from __future__ import annotations

import os
import hashlib
from functools import lru_cache
from pathlib import Path

import pandas as pd

from project35.monitoring.psi import PSIError, PSIProfile


class PSIReferenceUnavailable(RuntimeError):
    """The reference file or its monitoring configuration cannot be loaded."""


@lru_cache(maxsize=4)
def _profile_for(path_text: str, mtime_ns: int, size: int) -> PSIProfile:
    del mtime_ns, size  # cache-key values make changes to the reference invalidate the cache
    path = Path(path_text)
    if not path.is_file():
        raise PSIReferenceUnavailable("Le fichier de référence PSI configuré est introuvable.")
    try:
        frame = pd.read_csv(path)
        profile = PSIProfile.fit(
            frame,
            bins=int(os.getenv("PSI_BINS", "10")),
            epsilon=float(os.getenv("PSI_EPSILON", "0.000001")),
            watch_threshold=float(os.getenv("PSI_WATCH_THRESHOLD", "0.10")),
            drift_threshold=float(os.getenv("PSI_DRIFT_THRESHOLD", "0.25")),
        )
        profile.reference_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return profile
    except PSIError as exc:
        raise PSIReferenceUnavailable("La référence PSI ou ses paramètres sont invalides.") from exc
    except Exception as exc:
        raise PSIReferenceUnavailable("Impossible de charger la référence PSI ou sa configuration.") from exc


def get_psi_profile() -> PSIProfile:
    configured = os.getenv("PROJECT35_PSI_REFERENCE")
    if not configured:
        raise PSIReferenceUnavailable("PROJECT35_PSI_REFERENCE doit désigner le CSV de référence.")
    path = Path(configured).expanduser().resolve()
    try:
        stat = path.stat()
        return _profile_for(str(path), stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        raise PSIReferenceUnavailable("Le fichier de référence PSI configuré est introuvable.") from exc
