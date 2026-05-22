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
if not User.objects.filter(email=email).exists():
    User.objects.create_superuser(email=email, password=password)
    print(f'Superusuário {email} criado com sucesso.')
else:
    print(f'Superusuário {email} já existe.')
"
fi

exec gunicorn apiguard.wsgi --bind 0.0.0.0:$PORT --workers 2 --timeout 120
