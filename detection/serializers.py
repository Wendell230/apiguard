"""Serializers para predição, logs de tráfego e modelos ML."""
from rest_framework import serializers

from .models import MLModel, TrafficLog


class PredictRequestSerializer(serializers.Serializer):
    """
    Validação de entrada do endpoint /api/predict/.
    Espera o subconjunto de features do CIC-DDoS2019 mais relevantes.
    """
    source_ip = serializers.IPAddressField(required=False, allow_blank=True, default='')
    protocol = serializers.CharField(max_length=20, required=False, default='')
    packet_count = serializers.IntegerField(min_value=0, required=False, default=0)
    duration = serializers.FloatField(min_value=0, required=False, default=0.0)

    # Features numéricas de tráfego (subconjunto CIC-DDoS2019)
    flow_duration = serializers.FloatField(default=0.0)
    total_fwd_packets = serializers.FloatField(default=0.0)
    total_bwd_packets = serializers.FloatField(default=0.0)
    total_length_fwd_packets = serializers.FloatField(default=0.0)
    total_length_bwd_packets = serializers.FloatField(default=0.0)
    fwd_packet_length_max = serializers.FloatField(default=0.0)
    fwd_packet_length_min = serializers.FloatField(default=0.0)
    fwd_packet_length_mean = serializers.FloatField(default=0.0)
    fwd_packet_length_std = serializers.FloatField(default=0.0)
    bwd_packet_length_max = serializers.FloatField(default=0.0)
    bwd_packet_length_min = serializers.FloatField(default=0.0)
    bwd_packet_length_mean = serializers.FloatField(default=0.0)
    bwd_packet_length_std = serializers.FloatField(default=0.0)
    flow_bytes_s = serializers.FloatField(default=0.0)
    flow_packets_s = serializers.FloatField(default=0.0)
    flow_iat_mean = serializers.FloatField(default=0.0)
    flow_iat_std = serializers.FloatField(default=0.0)
    flow_iat_max = serializers.FloatField(default=0.0)
    flow_iat_min = serializers.FloatField(default=0.0)
    syn_flag_count = serializers.FloatField(default=0.0)
    rst_flag_count = serializers.FloatField(default=0.0)
    psh_flag_count = serializers.FloatField(default=0.0)
    ack_flag_count = serializers.FloatField(default=0.0)
    avg_packet_size = serializers.FloatField(default=0.0)
    avg_fwd_segment_size = serializers.FloatField(default=0.0)
    avg_bwd_segment_size = serializers.FloatField(default=0.0)

    def get_feature_vector(self) -> dict:
        """Retorna apenas as features numéricas para o modelo."""
        NUMERIC_FIELDS = [
            'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
            'total_length_fwd_packets', 'total_length_bwd_packets',
            'fwd_packet_length_max', 'fwd_packet_length_min',
            'fwd_packet_length_mean', 'fwd_packet_length_std',
            'bwd_packet_length_max', 'bwd_packet_length_min',
            'bwd_packet_length_mean', 'bwd_packet_length_std',
            'flow_bytes_s', 'flow_packets_s',
            'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
            'syn_flag_count', 'rst_flag_count', 'psh_flag_count', 'ack_flag_count',
            'avg_packet_size', 'avg_fwd_segment_size', 'avg_bwd_segment_size',
        ]
        return {k: self.validated_data[k] for k in NUMERIC_FIELDS}


class TrafficLogSerializer(serializers.ModelSerializer):
    created_by_name = serializers.CharField(source='created_by.name', read_only=True)

    class Meta:
        model = TrafficLog
        fields = (
            'id', 'timestamp', 'source_ip', 'protocol', 'packet_count',
            'duration', 'features_json', 'prediction', 'attack_type',
            'probability', 'response_time_ms', 'created_by', 'created_by_name',
        )
        read_only_fields = fields


class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = ('id', 'name', 'version', 'file_path', 'accuracy', 'is_active', 'description', 'created_at')
        read_only_fields = ('id', 'created_at', 'file_path')


class MLModelUploadSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    version = serializers.CharField(max_length=20)
    description = serializers.CharField(required=False, default='')
    accuracy = serializers.FloatField(required=False, min_value=0, max_value=1)
    model_file = serializers.FileField()

    def validate_model_file(self, value):
        if not value.name.endswith('.pkl'):
            raise serializers.ValidationError('O arquivo deve ser um .pkl.')
        # Limite de 200 MB
        if value.size > 200 * 1024 * 1024:
            raise serializers.ValidationError('Arquivo muito grande (máx. 200 MB).')
        return value
