from rest_framework import serializers


class ExplainRequestSerializer(serializers.Serializer):
    instance = serializers.DictField(child=serializers.JSONField())
    class_index = serializers.IntegerField(min_value=0, default=1)


class LocalExplanationResponseSerializer(serializers.Serializer):
    model_name = serializers.CharField()
    model_version = serializers.CharField()
    reference_sha256 = serializers.CharField()
    class_index = serializers.IntegerField()
    class_value = serializers.JSONField()
    output_scale = serializers.CharField()
    explainer_method = serializers.CharField()
    base_value = serializers.FloatField()
    prediction_probability = serializers.FloatField()
    additivity_residual = serializers.FloatField()
    shap_values = serializers.DictField(child=serializers.FloatField())
    waterfall_png_base64 = serializers.CharField()
    interpretation = serializers.CharField()


class GlobalExplanationResponseSerializer(serializers.Serializer):
    model_name = serializers.CharField()
    model_version = serializers.CharField()
    reference_sha256 = serializers.CharField()
    class_index = serializers.IntegerField()
    class_value = serializers.JSONField()
    output_scale = serializers.CharField()
    explainer_method = serializers.CharField()
    rows_explained = serializers.IntegerField()
    mean_absolute_shap = serializers.DictField(child=serializers.FloatField())
    summary_png_base64 = serializers.CharField()
    beeswarm_png_base64 = serializers.CharField()
    interpretation = serializers.CharField()
