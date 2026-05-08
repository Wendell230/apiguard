"""Ponto de entrada WSGI para o projeto APIGuard."""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'apiguard.settings')

application = get_wsgi_application()
