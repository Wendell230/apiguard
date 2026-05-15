"""Testes dos endpoints de detecção: predict, logs e dashboard."""
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from authentication.models import User
from detection.models import TrafficLog

# Payload mínimo — todos os campos têm default=0.0 no serializer
VALID_PREDICT_PAYLOAD = {
    'source_ip': '192.168.1.100',
    'protocol': 'TCP',
    'packet_count': 100,
    'duration': 1.0,
    # Algumas features com valores reais
    'flow_duration': 500000.0,
    'tot_fwd_pkts': 50.0,
    'tot_bwd_pkts': 30.0,
    'syn_flag_cnt': 1.0,
    'ack_flag_cnt': 78.0,
    'flow_byts_s': 8000.0,
    'flow_pkts_s': 160.0,
    # Demais fields assumem default 0.0
}


class DetectionTests(APITestCase):
    def setUp(self):
        self.analyst = User.objects.create_user(
            email='analista@apiguard.com',
            password='senha123',
            name='Analista',
            role=User.Role.ANALYST,
        )
        self.admin = User.objects.create_user(
            email='admin@apiguard.com',
            password='adminsenha123',
            name='Admin',
            role=User.Role.ADMIN,
        )
        self.predict_url     = reverse('predict')
        self.dashboard_url   = reverse('dashboard')
        self.model_list_url  = reverse('model-list')
        self.model_update_url = reverse('model-update')

    def _auth(self, user):
        pwd = 'adminsenha123' if user.role == User.Role.ADMIN else 'senha123'
        resp = self.client.post(reverse('auth-login'), {
            'email': user.email, 'password': pwd,
        }, format='json')
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {resp.data["access"]}')

    def test_predict_sem_autenticacao(self):
        resp = self.client.post(self.predict_url, VALID_PREDICT_PAYLOAD, format='json')
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_predict_com_payload_valido(self):
        self._auth(self.analyst)
        resp = self.client.post(self.predict_url, VALID_PREDICT_PAYLOAD, format='json')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn('prediction', resp.data)
        self.assertIn('attack_type', resp.data)
        self.assertIn('probability', resp.data)
        self.assertIn('response_time_ms', resp.data)
        self.assertIn('log_id', resp.data)
        self.assertIn(resp.data['prediction'], ('BENIGN', 'ATTACK'))

    def test_predict_persiste_log(self):
        self._auth(self.analyst)
        count_before = TrafficLog.objects.count()
        self.client.post(self.predict_url, VALID_PREDICT_PAYLOAD, format='json')
        self.assertEqual(TrafficLog.objects.count(), count_before + 1)

    def test_predict_campo_invalido(self):
        self._auth(self.analyst)
        payload = {**VALID_PREDICT_PAYLOAD, 'packet_count': -1}
        resp = self.client.post(self.predict_url, payload, format='json')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_logs_lista_paginada(self):
        self._auth(self.analyst)
        for _ in range(3):
            self.client.post(self.predict_url, VALID_PREDICT_PAYLOAD, format='json')
        resp = self.client.get(reverse('traffic-logs-list'))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn('results', resp.data)
        self.assertGreaterEqual(resp.data['count'], 3)

    def test_log_detalhe(self):
        self._auth(self.analyst)
        pred = self.client.post(self.predict_url, VALID_PREDICT_PAYLOAD, format='json')
        log_id = pred.data['log_id']
        resp = self.client.get(reverse('traffic-logs-detail', args=[log_id]))
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['id'], log_id)

    def test_dashboard_retorna_metricas(self):
        self._auth(self.analyst)
        resp = self.client.get(self.dashboard_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn('today', resp.data)
        self.assertIn('avg_latency_ms', resp.data)
        self.assertIn('active_model', resp.data)

    def test_model_list_requer_autenticacao(self):
        resp = self.client.get(self.model_list_url)
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_model_list_autenticado(self):
        self._auth(self.analyst)
        resp = self.client.get(self.model_list_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_model_update_somente_admin(self):
        """Analista não pode fazer upload de modelo."""
        self._auth(self.analyst)
        resp = self.client.post(self.model_update_url, {}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_403_FORBIDDEN)
