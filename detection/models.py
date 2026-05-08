"""Modelos de dados para logs de tráfego e versões de modelos ML."""
from django.conf import settings
from django.db import models


class MLModel(models.Model):
    """Registro de versões do modelo Random Forest treinado."""
    name = models.CharField(max_length=100, verbose_name='Nome')
    version = models.CharField(max_length=20, verbose_name='Versão')
    file_path = models.CharField(max_length=500, verbose_name='Caminho do arquivo .pkl')
    accuracy = models.FloatField(null=True, blank=True, verbose_name='Acurácia')
    is_active = models.BooleanField(default=False, verbose_name='Ativo')
    description = models.TextField(blank=True, verbose_name='Descrição')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Modelo ML'
        verbose_name_plural = 'Modelos ML'
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.name} v{self.version} ({"ativo" if self.is_active else "inativo"})'

    def save(self, *args, **kwargs):
        # Garante que somente um modelo esteja ativo por vez
        if self.is_active:
            MLModel.objects.exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)


class TrafficLog(models.Model):
    """Log de análise de tráfego de rede classificado pelo modelo."""

    class Prediction(models.TextChoices):
        BENIGN = 'BENIGN', 'Benigno'
        ATTACK = 'ATTACK', 'Ataque'

    class AttackType(models.TextChoices):
        BENIGN = 'BENIGN', 'Benigno'
        SYN_FLOOD = 'SYN Flood', 'SYN Flood'
        UDP_FLOOD = 'UDP Flood', 'UDP Flood'
        HTTP_FLOOD = 'HTTP Flood', 'HTTP Flood'
        DNS_AMPLIFICATION = 'DNS Amplification', 'DNS Amplification'
        UNKNOWN = 'Unknown', 'Desconhecido'

    timestamp = models.DateTimeField(auto_now_add=True, verbose_name='Timestamp')
    source_ip = models.GenericIPAddressField(null=True, blank=True, verbose_name='IP de origem')
    protocol = models.CharField(max_length=20, blank=True, verbose_name='Protocolo')
    packet_count = models.PositiveIntegerField(null=True, blank=True, verbose_name='Qtd. pacotes')
    duration = models.FloatField(null=True, blank=True, verbose_name='Duração (s)')
    features_json = models.JSONField(verbose_name='Features de entrada')
    prediction = models.CharField(
        max_length=10, choices=Prediction.choices, verbose_name='Predição'
    )
    attack_type = models.CharField(
        max_length=50, choices=AttackType.choices, verbose_name='Tipo de ataque'
    )
    probability = models.FloatField(verbose_name='Probabilidade')
    response_time_ms = models.FloatField(verbose_name='Tempo de resposta (ms)')
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='traffic_logs',
        verbose_name='Criado por',
    )

    class Meta:
        verbose_name = 'Log de Tráfego'
        verbose_name_plural = 'Logs de Tráfego'
        ordering = ['-timestamp']

    def __str__(self):
        return f'[{self.timestamp:%Y-%m-%d %H:%M}] {self.source_ip} → {self.attack_type}'
