from rest_framework import serializers


class PredictionRequestSerializer(serializers.Serializer):
    instances = serializers.ListField(
        child=serializers.DictField(child=serializers.JSONField()),
        allow_empty=False,
        max_length=1000,
    )


class PredictionResponseSerializer(serializers.Serializer):
    request_id = serializers.UUIDField()
    model_name = serializers.CharField()
    model_version = serializers.CharField()
    prediction = serializers.ListField(child=serializers.JSONField())
    probabilities = serializers.ListField(
        child=serializers.ListField(child=serializers.FloatField()), allow_null=True
    )


class HealthResponseSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=("ok", "degraded"))
    service = serializers.CharField()
    model_loaded = serializers.BooleanField()
    model_name = serializers.CharField()
    alias = serializers.CharField()
    model_version = serializers.CharField(allow_null=True)


class ModelInfoResponseSerializer(serializers.Serializer):
    name = serializers.CharField()
    alias = serializers.CharField()
    version = serializers.CharField()
    run_id = serializers.CharField()
    metrics = serializers.DictField(child=serializers.FloatField())
    tags = serializers.DictField(child=serializers.CharField())
