from rest_framework import serializers


class PSIRequestSerializer(serializers.Serializer):
    records = serializers.ListField(
        child=serializers.DictField(child=serializers.JSONField()),
        allow_empty=False,
        max_length=10000,
    )
    current_window = serializers.CharField(required=False, allow_blank=False, max_length=160)
    observed_at = serializers.DateTimeField(required=False)


class PSIFeatureResultSerializer(serializers.Serializer):
    psi = serializers.FloatField()
    status = serializers.ChoiceField(choices=("stable", "watch", "drift"))
    reference_proportions = serializers.DictField(child=serializers.FloatField())
    current_proportions = serializers.DictField(child=serializers.FloatField())
    contributions = serializers.DictField(child=serializers.FloatField())
    bin_definition = serializers.JSONField()


class PSIResponseSerializer(serializers.Serializer):
    report_id = serializers.CharField()
    reference_sha256 = serializers.CharField()
    current_sha256 = serializers.CharField()
    reference_rows = serializers.IntegerField()
    current_rows = serializers.IntegerField()
    reference_window = serializers.CharField()
    current_window = serializers.CharField(allow_null=True)
    observed_at = serializers.DateTimeField()
    bins = serializers.IntegerField()
    epsilon = serializers.FloatField()
    thresholds = serializers.DictField(child=serializers.FloatField())
    status = serializers.ChoiceField(choices=("stable", "watch", "drift"))
    alert = serializers.BooleanField()
    features = serializers.DictField(child=PSIFeatureResultSerializer())
