"""Ponto de entrada ASGI para o projeto APIGuard."""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'apiguard.settings')

application = get_asgi_application()
