"""Testes de autenticação JWT."""
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from authentication.models import User


class AuthTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email='analista@apiguard.com',
            password='senha_segura_123',
            name='Analista Teste',
            role=User.Role.ANALYST,
        )
        self.login_url = reverse('auth-login')
        self.refresh_url = reverse('auth-refresh')
        self.logout_url = reverse('auth-logout')
        self.me_url = reverse('auth-me')

    def _login(self):
        resp = self.client.post(self.login_url, {
            'email': 'analista@apiguard.com',
            'password': 'senha_segura_123',
        }, format='json')
        return resp.data

    def test_login_com_credenciais_validas(self):
        resp = self.client.post(self.login_url, {
            'email': 'analista@apiguard.com',
            'password': 'senha_segura_123',
        }, format='json')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn('access', resp.data)
        self.assertIn('refresh', resp.data)
        self.assertEqual(resp.data['user']['email'], 'analista@apiguard.com')

    def test_login_com_credenciais_invalidas(self):
        resp = self.client.post(self.login_url, {
            'email': 'analista@apiguard.com',
            'password': 'senha_errada',
        }, format='json')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)

    def test_refresh_token(self):
        tokens = self._login()
        resp = self.client.post(self.refresh_url, {'refresh': tokens['refresh']}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertIn('access', resp.data)

    def test_me_sem_autenticacao(self):
        resp = self.client.get(self.me_url)
        self.assertEqual(resp.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_me_com_autenticacao(self):
        tokens = self._login()
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {tokens["access"]}')
        resp = self.client.get(self.me_url)
        self.assertEqual(resp.status_code, status.HTTP_200_OK)
        self.assertEqual(resp.data['email'], 'analista@apiguard.com')
        self.assertEqual(resp.data['role'], User.Role.ANALYST)

    def test_logout(self):
        tokens = self._login()
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {tokens["access"]}')
        resp = self.client.post(self.logout_url, {'refresh': tokens['refresh']}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_200_OK)

    def test_logout_sem_token_refresh(self):
        tokens = self._login()
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {tokens["access"]}')
        resp = self.client.post(self.logout_url, {}, format='json')
        self.assertEqual(resp.status_code, status.HTTP_400_BAD_REQUEST)
