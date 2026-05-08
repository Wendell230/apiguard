"""Rotas da app detection."""
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    DashboardView,
    ModelListView,
    ModelUpdateView,
    PredictView,
    TrafficLogViewSet,
)

router = DefaultRouter()
router.register(r'logs', TrafficLogViewSet, basename='traffic-logs')

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
    path('model/update/', ModelUpdateView.as_view(), name='model-update'),
    path('model/list/', ModelListView.as_view(), name='model-list'),
    path('', include(router.urls)),
]
