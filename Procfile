web: python manage.py migrate --noinput && python manage.py collectstatic --noinput --clear && gunicorn apiguard.wsgi --bind 0.0.0.0:$PORT --workers 2 --timeout 120
