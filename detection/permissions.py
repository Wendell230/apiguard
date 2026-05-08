"""Permissões customizadas baseadas em roles."""
from rest_framework.permissions import BasePermission


class IsAdmin(BasePermission):
    """Permite acesso apenas a usuários com role=admin."""
    message = 'Apenas administradores podem realizar esta ação.'

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and request.user.is_admin)


class IsAdminOrAnalyst(BasePermission):
    """Permite acesso a admins e analistas."""
    message = 'Acesso restrito a administradores e analistas.'

    def has_permission(self, request, view):
        if not (request.user and request.user.is_authenticated):
            return False
        return request.user.role in ('admin', 'analyst')
