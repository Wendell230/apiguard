"""Views da app detection: predição, logs, dashboard e gestão de modelos."""
import os
from datetime import date, timedelta
from pathlib import Path

from django.conf import settings
from django.db.models import Avg, Count, Q
from django.utils import timezone
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework.viewsets import ReadOnlyModelViewSet

from .ml_service import ml_service
from .models import MLModel, TrafficLog
from .permissions import IsAdmin
from .serializers import (
    MLModelSerializer,
    MLModelUploadSerializer,
    PredictRequestSerializer,
    TrafficLogSerializer,
)


class PredictView(APIView):
    """
    POST /api/predict/
    Recebe features de tráfego, classifica com Random Forest e persiste o log.
    """
    permission_classes = (IsAuthenticated,)
    throttle_classes   = (ScopedRateThrottle,)
    throttle_scope     = 'predict'

    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        # Inferência
        result = ml_service.predict(serializer.get_feature_vector())

        # Persiste log
        log = TrafficLog.objects.create(
            source_ip=data.get('source_ip') or None,
            protocol=data.get('protocol', ''),
            packet_count=data.get('packet_count'),
            duration=data.get('duration'),
            features_json=serializer.get_feature_vector(),
            prediction=result['prediction'],
            attack_type=result['attack_type'],
            probability=result['probability'],
            response_time_ms=result['response_time_ms'],
            created_by=request.user,
        )

        return Response({
            'log_id': log.id,
            'prediction': result['prediction'],
            'attack_type': result['attack_type'],
            'probability': result['probability'],
            'response_time_ms': result['response_time_ms'],
            'is_mock': result['is_mock'],
            'timestamp': log.timestamp,
        }, status=status.HTTP_200_OK)


class TrafficLogViewSet(ReadOnlyModelViewSet):
    """
    GET /api/logs/       → lista paginada com filtros
    GET /api/logs/<id>/  → detalhe
    """
    permission_classes = (IsAuthenticated,)
    serializer_class = TrafficLogSerializer
    filterset_fields = ['prediction', 'attack_type', 'protocol']
    search_fields = ['source_ip', 'attack_type']
    ordering_fields = ['timestamp', 'probability', 'response_time_ms']
    ordering = ['-timestamp']

    def get_queryset(self):
        qs = TrafficLog.objects.select_related('created_by')

        # Filtro por data
        date_from = self.request.query_params.get('date_from')
        date_to = self.request.query_params.get('date_to')
        if date_from:
            qs = qs.filter(timestamp__date__gte=date_from)
        if date_to:
            qs = qs.filter(timestamp__date__lte=date_to)

        # Filtro por IP
        ip = self.request.query_params.get('source_ip')
        if ip:
            qs = qs.filter(source_ip=ip)

        return qs


class DashboardView(APIView):
    """GET /api/dashboard/ — métricas resumidas para o painel de controle."""
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        now = timezone.now()
        last_24h = now - timedelta(hours=24)
        last_7d  = now - timedelta(days=7)

        total_today   = TrafficLog.objects.filter(timestamp__gte=last_24h).count()
        attacks_today = TrafficLog.objects.filter(timestamp__gte=last_24h, prediction='ATTACK').count()
        benign_today  = total_today - attacks_today

        pct_benign = round((benign_today / total_today * 100), 1) if total_today else 0

        avg_latency = TrafficLog.objects.filter(
            timestamp__gte=last_24h
        ).aggregate(avg=Avg('response_time_ms'))['avg']

        # Top tipos de ataque (últimos 7 dias)
        top_attacks = (
            TrafficLog.objects.filter(timestamp__gte=last_7d, prediction='ATTACK')
            .values('attack_type')
            .annotate(count=Count('id'))
            .order_by('-count')[:5]
        )

        # Status do modelo ativo
        active_model = MLModel.objects.filter(is_active=True).first()
        model_status = {
            'name': active_model.name if active_model else 'Mock (sem modelo treinado)',
            'version': active_model.version if active_model else '0.0',
            'accuracy': active_model.accuracy if active_model else None,
            'is_mock': active_model is None,
        }

        return Response({
            'today': {
                'total': total_today,
                'attacks': attacks_today,
                'benign': benign_today,
                'pct_benign': pct_benign,
            },
            'avg_latency_ms': round(avg_latency, 3) if avg_latency else 0,
            'top_attacks_7d': list(top_attacks),
            'active_model': model_status,
        })


class ModelUpdateView(APIView):
    """POST /api/model/update/ — upload de novo modelo .pkl (somente admin)."""
    permission_classes = (IsAdmin,)
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        serializer = MLModelUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        # Salva arquivo na pasta ml/models/
        models_dir = Path(settings.ML_MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        file_obj = data['model_file']
        filename = f"model_v{data['version']}.pkl"
        file_path = models_dir / filename

        with open(file_path, 'wb') as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        # Cria registro e ativa o novo modelo
        ml_model = MLModel.objects.create(
            name=data['name'],
            version=data['version'],
            file_path=str(file_path),
            accuracy=data.get('accuracy'),
            description=data.get('description', ''),
            is_active=True,
        )

        # Recarrega o serviço singleton
        ml_service.reload()

        return Response(MLModelSerializer(ml_model).data, status=status.HTTP_201_CREATED)


class ModelListView(APIView):
    """GET /api/model/list/ — lista todas as versões de modelos."""
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        models = MLModel.objects.all()
        return Response(MLModelSerializer(models, many=True).data)
