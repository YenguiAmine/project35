from django.urls import path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from predictions.views import ExplainView, GlobalExplainView, HealthView, ModelInfoView, PredictView, PSIMonitoringView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("api/predict/", PredictView.as_view(), name="predict"),
    path("api/model-info/", ModelInfoView.as_view(), name="model-info"),
    path("api/explain/", ExplainView.as_view(), name="explain-local"),
    path("api/explain/global/", GlobalExplainView.as_view(), name="explain-global"),
    path("api/monitoring/psi/", PSIMonitoringView.as_view(), name="monitoring-psi"),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
]
