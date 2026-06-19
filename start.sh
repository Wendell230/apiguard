#!/bin/bash
set -e

python manage.py migrate --noinput
python manage.py collectstatic --noinput --clear

python manage.py shell -c "
import os
from authentication.models import User
email = os.environ.get('DJANGO_SUPERUSER_EMAIL', 'admin@apiguard.com')
password = os.environ.get('DJANGO_SUPERUSER_PASSWORD', 'Admin@2026')
name = os.environ.get('DJANGO_SUPERUSER_NAME', 'Admin')
if not User.objects.filter(is_superuser=True).exists():
    User.objects.create_superuser(email=email, password=password, name=name)
    print('Admin criado:', email)
else:
    print('Admin ja existe.')
"

python manage.py shell -c "
from pathlib import Path
from detection.models import MLModel
from django.conf import settings

if not MLModel.objects.filter(is_active=True).exists():
    model_path = Path(settings.ML_MODELS_DIR) / 'modelo_rf_otimizado.pkl'
    if model_path.exists():
        MLModel.objects.create(
            name='Random Forest CIC-DDoS2019',
            version='1.0',
            file_path=str(model_path),
            accuracy=0.8505,
            is_active=True,
            description='Treinado com 26 features do CIC-DDoS2019. Acuracia 0.8505 | F1: 0.8561',
        )
        print('Modelo RF registrado no banco.')
    else:
        print('AVISO: modelo_rf_otimizado.pkl nao encontrado.')
else:
    print('Modelo ativo ja registrado.')
"

exec gunicorn apiguard.wsgi --bind 0.0.0.0:$PORT --workers 2 --timeout 120
