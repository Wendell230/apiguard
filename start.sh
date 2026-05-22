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

exec gunicorn apiguard.wsgi --bind 0.0.0.0:$PORT --workers 2 --timeout 120
