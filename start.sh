#!/bin/bash
set -e

python manage.py migrate --noinput
python manage.py collectstatic --noinput --clear

# Cria superusuário automaticamente se as variáveis estiverem definidas
if [ -n "$DJANGO_SUPERUSER_EMAIL" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ]; then
    python manage.py shell -c "
from authentication.models import User
email = '$DJANGO_SUPERUSER_EMAIL'
password = '$DJANGO_SUPERUSER_PASSWORD'
name = '${DJANGO_SUPERUSER_NAME:-Admin}'
if not User.objects.filter(email=email).exists():
    User.objects.create_superuser(email=email, password=password, name=name)
    print('Superusuario criado:', email)
else:
    print('Superusuario ja existe:', email)
"
fi

exec gunicorn apiguard.wsgi --bind 0.0.0.0:$PORT --workers 2 --timeout 120
