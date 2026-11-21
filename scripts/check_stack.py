from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import yaml


def _get_json(url: str) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"GET {url} returned a non-object JSON response")
    return payload


def _get_text(url: str) -> str:
    try:
        with urlopen(url, timeout=5) as response:
            return response.read().decode("utf-8")
    except (OSError, URLError) as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc


def _get_spec(url: str) -> dict[str, Any]:
    text = _get_text(url)
    try:
        spec = json.loads(text)
    except json.JSONDecodeError:
        spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise RuntimeError(f"GET {url} returned an invalid OpenAPI document")
    return spec


def check_stack(fastapi_url: str, django_url: str, mlflow_url: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    fast_health = _get_json(f"{fastapi_url.rstrip('/')}/health/")
    if fast_health.get("status") != "ok":
        raise RuntimeError("FastAPI health response is not ok")
    checks["fastapi"] = {"status": "ok", "health": fast_health}

    fast_openapi = _get_json(f"{fastapi_url.rstrip('/')}/openapi.json")
    if "/api/v1/data/validate/" not in fast_openapi.get("paths", {}):
        raise RuntimeError("FastAPI OpenAPI does not expose data validation")
    checks["fastapi_openapi"] = {"status": "ok", "path_count": len(fast_openapi["paths"])}

    django_health = _get_json(f"{django_url.rstrip('/')}/health/")
    checks["django"] = {
        "status": django_health.get("status"),
        "model_loaded": django_health.get("model_loaded"),
        "model_version": django_health.get("model_version"),
    }

    django_openapi = _get_spec(f"{django_url.rstrip('/')}/api/schema/")
    paths = django_openapi.get("paths", {})
    required_paths = {"/api/predict/", "/api/explain/", "/api/monitoring/psi/"}
    missing = sorted(required_paths - set(paths))
    if missing:
        raise RuntimeError(f"Django OpenAPI is missing expected paths: {missing}")
    checks["django_openapi"] = {"status": "ok", "path_count": len(paths)}

    mlflow_health = _get_text(f"{mlflow_url.rstrip('/')}/health")
    checks["mlflow"] = {"status": "ok", "health": mlflow_health}
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check the Project 35 local Compose services.")
    parser.add_argument("--fastapi", default="http://127.0.0.1:8000")
    parser.add_argument("--django", default="http://127.0.0.1:8001")
    parser.add_argument("--mlflow", default="http://127.0.0.1:5000")
    args = parser.parse_args()
    try:
        result = check_stack(args.fastapi, args.django, args.mlflow)
    except RuntimeError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", "checks": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
