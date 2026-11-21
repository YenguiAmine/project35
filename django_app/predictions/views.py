from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.exceptions import APIException, ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

from predictions.serializers import (
    HealthResponseSerializer,
    ModelInfoResponseSerializer,
    PredictionRequestSerializer,
    PredictionResponseSerializer,
)
from predictions.explain_serializers import (
    ExplainRequestSerializer,
    GlobalExplanationResponseSerializer,
    LocalExplanationResponseSerializer,
)
from predictions.explainability import ExplanationError, get_explanation_service
from predictions.monitoring import PSIReferenceUnavailable, get_psi_profile
from predictions.monitoring_serializers import PSIRequestSerializer, PSIResponseSerializer
from project35.monitoring.psi import PSIError
from predictions.services import ModelUnavailable, model_provider

logger = logging.getLogger(__name__)


class RegistryUnavailable(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = "Le modèle champion est temporairement indisponible."
    default_code = "model_unavailable"


class ExplanationUnavailable(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = "Le service d'explication est indisponible; vérifiez le modèle et la référence SHAP."
    default_code = "explanation_unavailable"


class MonitoringUnavailable(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = "La surveillance PSI est temporairement indisponible."
    default_code = "monitoring_unavailable"


def _plain_list(values):
    return np.asarray(values).tolist()


class HealthView(APIView):
    authentication_classes = []
    permission_classes = []

    @extend_schema(responses=HealthResponseSerializer)
    def get(self, request):
        metadata = model_provider.cached_metadata()
        return Response({
            "status": "ok" if metadata else "degraded",
            "service": "project35-django-rest",
            "model_loaded": metadata is not None,
            "model_name": model_provider.model_name,
            "alias": model_provider.alias,
            "model_version": metadata["version"] if metadata else None,
        })


class PredictView(APIView):
    authentication_classes = []
    permission_classes = []

    @extend_schema(request=PredictionRequestSerializer, responses={200: PredictionResponseSerializer})
    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        instances = serializer.validated_data["instances"]
        keys = set(instances[0])
        if not keys:
            raise ValidationError({"instances": "Chaque observation doit contenir au moins une feature."})
        if any(set(item) != keys for item in instances):
            raise ValidationError({"instances": "Toutes les observations d'un lot doivent avoir le même schéma."})

        request_id = uuid.uuid4()
        started = time.perf_counter()
        try:
            model, metadata = model_provider.get_active()
        except ModelUnavailable as exc:
            raise RegistryUnavailable() from exc

        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            expected = set(str(name) for name in feature_names)
            missing, extra = sorted(expected - keys), sorted(keys - expected)
            if missing or extra:
                raise ValidationError({"instances": {"missing_features": missing, "unknown_features": extra}})
            frame = pd.DataFrame(instances, columns=list(feature_names))
        else:
            frame = pd.DataFrame(instances)

        try:
            predictions = _plain_list(model.predict(frame))
            proba_method = getattr(model, "predict_proba", None)
            probabilities = _plain_list(proba_method(frame)) if callable(proba_method) else None
        except (ValueError, TypeError) as exc:
            raise ValidationError({"instances": "Les valeurs ne respectent pas le schéma attendu par le modèle."}) from exc

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "prediction request_id=%s model=%s version=%s rows=%s latency_ms=%s status=ok",
            request_id,
            metadata["name"],
            metadata["version"],
            len(instances),
            latency_ms,
        )
        return Response({
            "request_id": request_id,
            "model_name": metadata["name"],
            "model_version": metadata["version"],
            "prediction": predictions,
            "probabilities": probabilities,
        })


class ModelInfoView(APIView):
    authentication_classes = []
    permission_classes = []

    @extend_schema(responses=ModelInfoResponseSerializer)
    def get(self, request):
        try:
            _, metadata = model_provider.get_active()
        except ModelUnavailable as exc:
            raise RegistryUnavailable() from exc
        return Response(metadata)


class ExplainView(APIView):
    authentication_classes = []
    permission_classes = []

    @extend_schema(request=ExplainRequestSerializer, responses={200: LocalExplanationResponseSerializer})
    def post(self, request):
        serializer = ExplainRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            model, metadata = model_provider.get_active()
            result = get_explanation_service().explain_one(
                model,
                serializer.validated_data["instance"],
                class_index=serializer.validated_data["class_index"],
            )
            result.update(model_name=metadata["name"], model_version=metadata["version"])
        except ExplanationError as exc:
            raise ValidationError({"explanation": str(exc)}) from exc
        except ModelUnavailable as exc:
            raise RegistryUnavailable() from exc
        except Exception as exc:
            logger.exception("SHAP local explanation failed")
            raise ExplanationUnavailable() from exc
        return Response(result)


class GlobalExplainView(APIView):
    authentication_classes = []
    permission_classes = []

    @extend_schema(responses={200: GlobalExplanationResponseSerializer})
    def get(self, request):
        try:
            class_index = int(request.query_params.get("class_index", "1"))
            if class_index < 0:
                raise ValueError
            model, metadata = model_provider.get_active()
            result = get_explanation_service().explain_global(model, class_index=class_index)
            result.update(model_name=metadata["name"], model_version=metadata["version"])
        except ValueError as exc:
            raise ValidationError({"class_index": "Doit être un entier positif ou nul."}) from exc
        except ExplanationError as exc:
            raise ValidationError({"explanation": str(exc)}) from exc
        except ModelUnavailable as exc:
            raise RegistryUnavailable() from exc
        except Exception as exc:
            logger.exception("SHAP global explanation failed")
            raise ExplanationUnavailable() from exc
        return Response(result)


class PSIMonitoringView(APIView):
    authentication_classes = []
    permission_classes = []

    @extend_schema(request=PSIRequestSerializer, responses={200: PSIResponseSerializer})
    def post(self, request):
        serializer = PSIRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        records = serializer.validated_data["records"]
        columns = set(records[0])
        if not columns or any(set(record) != columns for record in records):
            raise ValidationError({"records": "Chaque observation doit avoir le même schéma non vide."})
        try:
            current = pd.DataFrame.from_records(records)
            observed_at = serializer.validated_data.get("observed_at") or datetime.now(timezone.utc)
            report = get_psi_profile().calculate(
                current,
                current_window=serializer.validated_data.get("current_window"),
                observed_at=observed_at.isoformat(),
            )
        except PSIReferenceUnavailable as exc:
            raise MonitoringUnavailable() from exc
        except PSIError as exc:
            raise ValidationError({"monitoring": str(exc)}) from exc
        except Exception as exc:
            logger.exception("PSI monitoring calculation failed")
            raise MonitoringUnavailable() from exc
        logger.info(
            "psi report_id=%s rows=%s status=%s features=%s reference_sha256=%s current_sha256=%s",
            report["report_id"],
            report["current_rows"],
            report["status"],
            len(report["features"]),
            report["reference_sha256"],
            report["current_sha256"],
        )
        return Response(report)
