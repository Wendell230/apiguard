"""URLs raiz do projeto APIGuard."""
from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/', include('authentication.urls')),
    path('api/', include('detection.urls')),
    # Interface web — serve o SPA
    path('', TemplateView.as_view(template_name='index.html'), name='frontend'),
]
