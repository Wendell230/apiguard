"""Serializers para predição, logs de tráfego e modelos ML."""
from collections import OrderedDict

from rest_framework import serializers

from .models import MLModel, TrafficLog

# Mapeamento: campo snake_case (JSON) → nome original no CSV do dataset
COLUMN_MAP = OrderedDict([
    ('flow_duration',       'Flow Duration'),
    ('tot_fwd_pkts',        'Tot Fwd Pkts'),
    ('tot_bwd_pkts',        'Tot Bwd Pkts'),
    ('totlen_fwd_pkts',     'TotLen Fwd Pkts'),
    ('totlen_bwd_pkts',     'TotLen Bwd Pkts'),
    ('fwd_pkt_len_max',     'Fwd Pkt Len Max'),
    ('fwd_pkt_len_min',     'Fwd Pkt Len Min'),
    ('fwd_pkt_len_mean',    'Fwd Pkt Len Mean'),
    ('fwd_pkt_len_std',     'Fwd Pkt Len Std'),
    ('bwd_pkt_len_max',     'Bwd Pkt Len Max'),
    ('bwd_pkt_len_min',     'Bwd Pkt Len Min'),
    ('bwd_pkt_len_mean',    'Bwd Pkt Len Mean'),
    ('bwd_pkt_len_std',     'Bwd Pkt Len Std'),
    ('flow_byts_s',         'Flow Byts/s'),
    ('flow_pkts_s',         'Flow Pkts/s'),
    ('flow_iat_mean',       'Flow IAT Mean'),
    ('flow_iat_std',        'Flow IAT Std'),
    ('flow_iat_max',        'Flow IAT Max'),
    ('flow_iat_min',        'Flow IAT Min'),
    ('fwd_iat_tot',         'Fwd IAT Tot'),
    ('fwd_iat_mean',        'Fwd IAT Mean'),
    ('fwd_iat_std',         'Fwd IAT Std'),
    ('fwd_iat_max',         'Fwd IAT Max'),
    ('fwd_iat_min',         'Fwd IAT Min'),
    ('bwd_iat_tot',         'Bwd IAT Tot'),
    ('bwd_iat_mean',        'Bwd IAT Mean'),
    ('bwd_iat_std',         'Bwd IAT Std'),
    ('bwd_iat_max',         'Bwd IAT Max'),
    ('bwd_iat_min',         'Bwd IAT Min'),
    ('fwd_psh_flags',       'Fwd PSH Flags'),
    ('bwd_psh_flags',       'Bwd PSH Flags'),
    ('fwd_urg_flags',       'Fwd URG Flags'),
    ('bwd_urg_flags',       'Bwd URG Flags'),
    ('fwd_header_len',      'Fwd Header Len'),
    ('bwd_header_len',      'Bwd Header Len'),
    ('fwd_pkts_s',          'Fwd Pkts/s'),
    ('bwd_pkts_s',          'Bwd Pkts/s'),
    ('pkt_len_min',         'Pkt Len Min'),
    ('pkt_len_max',         'Pkt Len Max'),
    ('pkt_len_mean',        'Pkt Len Mean'),
    ('pkt_len_std',         'Pkt Len Std'),
    ('pkt_len_var',         'Pkt Len Var'),
    ('fin_flag_cnt',        'FIN Flag Cnt'),
    ('syn_flag_cnt',        'SYN Flag Cnt'),
    ('rst_flag_cnt',        'RST Flag Cnt'),
    ('psh_flag_cnt',        'PSH Flag Cnt'),
    ('ack_flag_cnt',        'ACK Flag Cnt'),
    ('urg_flag_cnt',        'URG Flag Cnt'),
    ('cwe_flag_count',      'CWE Flag Count'),
    ('ece_flag_cnt',        'ECE Flag Cnt'),
    ('down_up_ratio',       'Down/Up Ratio'),
    ('pkt_size_avg',        'Pkt Size Avg'),
    ('fwd_seg_size_avg',    'Fwd Seg Size Avg'),
    ('bwd_seg_size_avg',    'Bwd Seg Size Avg'),
    ('fwd_byts_b_avg',      'Fwd Byts/b Avg'),
    ('fwd_pkts_b_avg',      'Fwd Pkts/b Avg'),
    ('fwd_blk_rate_avg',    'Fwd Blk Rate Avg'),
    ('bwd_byts_b_avg',      'Bwd Byts/b Avg'),
    ('bwd_pkts_b_avg',      'Bwd Pkts/b Avg'),
    ('bwd_blk_rate_avg',    'Bwd Blk Rate Avg'),
    ('subflow_fwd_pkts',    'Subflow Fwd Pkts'),
    ('subflow_fwd_byts',    'Subflow Fwd Byts'),
    ('subflow_bwd_pkts',    'Subflow Bwd Pkts'),
    ('subflow_bwd_byts',    'Subflow Bwd Byts'),
    ('init_fwd_win_byts',   'Init Fwd Win Byts'),
    ('init_bwd_win_byts',   'Init Bwd Win Byts'),
    ('fwd_act_data_pkts',   'Fwd Act Data Pkts'),
    ('fwd_seg_size_min',    'Fwd Seg Size Min'),
    ('active_mean',         'Active Mean'),
    ('active_std',          'Active Std'),
    ('active_max',          'Active Max'),
    ('active_min',          'Active Min'),
    ('idle_mean',           'Idle Mean'),
    ('idle_std',            'Idle Std'),
    ('idle_max',            'Idle Max'),
    ('idle_min',            'Idle Min'),
])


class PredictRequestSerializer(serializers.Serializer):
    """
    Validação de entrada do endpoint /api/predict/.
    Campos de metadados + 76 features do dataset 5G DoS/DDoS.
    Todos os campos de feature têm default=0.0 para facilitar uso.
    """
    # Metadados opcionais (não vão para o modelo)
    source_ip  = serializers.IPAddressField(required=False, allow_blank=True, default='')
    protocol   = serializers.CharField(max_length=20, required=False, default='')
    packet_count = serializers.IntegerField(min_value=0, required=False, default=0)
    duration   = serializers.FloatField(min_value=0, required=False, default=0.0)

    # 76 features — nomes em snake_case, mapeados para o CSV em COLUMN_MAP
    flow_duration    = serializers.FloatField(default=0.0)
    tot_fwd_pkts     = serializers.FloatField(default=0.0)
    tot_bwd_pkts     = serializers.FloatField(default=0.0)
    totlen_fwd_pkts  = serializers.FloatField(default=0.0)
    totlen_bwd_pkts  = serializers.FloatField(default=0.0)
    fwd_pkt_len_max  = serializers.FloatField(default=0.0)
    fwd_pkt_len_min  = serializers.FloatField(default=0.0)
    fwd_pkt_len_mean = serializers.FloatField(default=0.0)
    fwd_pkt_len_std  = serializers.FloatField(default=0.0)
    bwd_pkt_len_max  = serializers.FloatField(default=0.0)
    bwd_pkt_len_min  = serializers.FloatField(default=0.0)
    bwd_pkt_len_mean = serializers.FloatField(default=0.0)
    bwd_pkt_len_std  = serializers.FloatField(default=0.0)
    flow_byts_s      = serializers.FloatField(default=0.0)
    flow_pkts_s      = serializers.FloatField(default=0.0)
    flow_iat_mean    = serializers.FloatField(default=0.0)
    flow_iat_std     = serializers.FloatField(default=0.0)
    flow_iat_max     = serializers.FloatField(default=0.0)
    flow_iat_min     = serializers.FloatField(default=0.0)
    fwd_iat_tot      = serializers.FloatField(default=0.0)
    fwd_iat_mean     = serializers.FloatField(default=0.0)
    fwd_iat_std      = serializers.FloatField(default=0.0)
    fwd_iat_max      = serializers.FloatField(default=0.0)
    fwd_iat_min      = serializers.FloatField(default=0.0)
    bwd_iat_tot      = serializers.FloatField(default=0.0)
    bwd_iat_mean     = serializers.FloatField(default=0.0)
    bwd_iat_std      = serializers.FloatField(default=0.0)
    bwd_iat_max      = serializers.FloatField(default=0.0)
    bwd_iat_min      = serializers.FloatField(default=0.0)
    fwd_psh_flags    = serializers.FloatField(default=0.0)
    bwd_psh_flags    = serializers.FloatField(default=0.0)
    fwd_urg_flags    = serializers.FloatField(default=0.0)
    bwd_urg_flags    = serializers.FloatField(default=0.0)
    fwd_header_len   = serializers.FloatField(default=0.0)
    bwd_header_len   = serializers.FloatField(default=0.0)
    fwd_pkts_s       = serializers.FloatField(default=0.0)
    bwd_pkts_s       = serializers.FloatField(default=0.0)
    pkt_len_min      = serializers.FloatField(default=0.0)
    pkt_len_max      = serializers.FloatField(default=0.0)
    pkt_len_mean     = serializers.FloatField(default=0.0)
    pkt_len_std      = serializers.FloatField(default=0.0)
    pkt_len_var      = serializers.FloatField(default=0.0)
    fin_flag_cnt     = serializers.FloatField(default=0.0)
    syn_flag_cnt     = serializers.FloatField(default=0.0)
    rst_flag_cnt     = serializers.FloatField(default=0.0)
    psh_flag_cnt     = serializers.FloatField(default=0.0)
    ack_flag_cnt     = serializers.FloatField(default=0.0)
    urg_flag_cnt     = serializers.FloatField(default=0.0)
    cwe_flag_count   = serializers.FloatField(default=0.0)
    ece_flag_cnt     = serializers.FloatField(default=0.0)
    down_up_ratio    = serializers.FloatField(default=0.0)
    pkt_size_avg     = serializers.FloatField(default=0.0)
    fwd_seg_size_avg = serializers.FloatField(default=0.0)
    bwd_seg_size_avg = serializers.FloatField(default=0.0)
    fwd_byts_b_avg   = serializers.FloatField(default=0.0)
    fwd_pkts_b_avg   = serializers.FloatField(default=0.0)
    fwd_blk_rate_avg = serializers.FloatField(default=0.0)
    bwd_byts_b_avg   = serializers.FloatField(default=0.0)
    bwd_pkts_b_avg   = serializers.FloatField(default=0.0)
    bwd_blk_rate_avg = serializers.FloatField(default=0.0)
    subflow_fwd_pkts = serializers.FloatField(default=0.0)
    subflow_fwd_byts = serializers.FloatField(default=0.0)
    subflow_bwd_pkts = serializers.FloatField(default=0.0)
    subflow_bwd_byts = serializers.FloatField(default=0.0)
    init_fwd_win_byts = serializers.FloatField(default=0.0)
    init_bwd_win_byts = serializers.FloatField(default=0.0)
    fwd_act_data_pkts = serializers.FloatField(default=0.0)
    fwd_seg_size_min  = serializers.FloatField(default=0.0)
    active_mean      = serializers.FloatField(default=0.0)
    active_std       = serializers.FloatField(default=0.0)
    active_max       = serializers.FloatField(default=0.0)
    active_min       = serializers.FloatField(default=0.0)
    idle_mean        = serializers.FloatField(default=0.0)
    idle_std         = serializers.FloatField(default=0.0)
    idle_max         = serializers.FloatField(default=0.0)
    idle_min         = serializers.FloatField(default=0.0)

    def get_feature_vector(self) -> dict:
        """Retorna dict com nomes originais do CSV → valores, na ordem do COLUMN_MAP."""
        return {
            original_name: float(self.validated_data.get(field_name, 0.0))
            for field_name, original_name in COLUMN_MAP.items()
        }


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
    name        = serializers.CharField(max_length=100)
    version     = serializers.CharField(max_length=20)
    description = serializers.CharField(required=False, default='')
    accuracy    = serializers.FloatField(required=False, min_value=0, max_value=1)
    model_file  = serializers.FileField()

    def validate_model_file(self, value):
        if not value.name.endswith('.pkl'):
            raise serializers.ValidationError('O arquivo deve ser um .pkl.')
        if value.size > 500 * 1024 * 1024:
            raise serializers.ValidationError('Arquivo muito grande (máx. 500 MB).')
        return value
