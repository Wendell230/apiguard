"""Views de autenticação: login, refresh, logout e perfil."""
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.exceptions import TokenError
from rest_framework_simplejwt.tokens import RefreshToken

from .serializers import LoginSerializer, UserSerializer


class LoginView(APIView):
    permission_classes = (AllowAny,)

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']

        refresh = RefreshToken.for_user(user)
        return Response({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'user': UserSerializer(user).data,
        }, status=status.HTTP_200_OK)


class RefreshView(APIView):
    permission_classes = (AllowAny,)

    def post(self, request):
        token = request.data.get('refresh')
        if not token:
            return Response(
                {'detail': 'O campo refresh é obrigatório.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            refresh = RefreshToken(token)
            return Response({'access': str(refresh.access_token)})
        except TokenError as exc:
            return Response({'detail': str(exc)}, status=status.HTTP_401_UNAUTHORIZED)


class LogoutView(APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request):
        token = request.data.get('refresh')
        if not token:
            return Response(
                {'detail': 'O campo refresh é obrigatório.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            RefreshToken(token).blacklist()
        except TokenError:
            pass
        return Response({'detail': 'Logout realizado com sucesso.'}, status=status.HTTP_200_OK)


class MeView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        return Response(UserSerializer(request.user).data)
