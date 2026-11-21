from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from project35.data.contract import DataContractError, validate_frame

app = FastAPI(
    title="Project 35 — Data & API Contract",
    description=(
        "API de développement pour vérifier le contrat de données dès la phase 1. "
        "La documentation interactive est disponible sur /docs. L'API de scoring finale "
        "reste prévue dans Django REST conformément au sujet Project 35."
    ),
    version="0.1.0",
)


class DataValidationRequest(BaseModel):
    target_column: str = Field(min_length=1)
    required_columns: list[str] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


@app.get("/health/", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "1-data-foundation"}


@app.get("/api/v1/data/contract/", tags=["data"])
def contract_info() -> dict[str, Any]:
    return {
        "target_column": "to_be_confirmed_from_official_topic",
        "dataset": "to_be_confirmed_from_official_topic",
        "validation": ["non_empty", "unique_column_names", "required_columns", "target_non_null", "target_has_multiple_classes"],
    }


@app.post("/api/v1/data/validate/", tags=["data"])
def validate_records(request: DataValidationRequest) -> dict[str, Any]:
    try:
        frame = pd.DataFrame.from_records(request.records)
        return validate_frame(frame, target_column=request.target_column, required_columns=tuple(request.required_columns))
    except DataContractError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
