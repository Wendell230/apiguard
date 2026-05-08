"""Serializers para autenticação e dados de usuário."""
from django.contrib.auth import authenticate
from rest_framework import serializers

from .models import User


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        user = authenticate(username=attrs['email'], password=attrs['password'])
        if not user:
            raise serializers.ValidationError('Credenciais inválidas.')
        if not user.is_active:
            raise serializers.ValidationError('Conta desativada.')
        attrs['user'] = user
        return attrs


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'email', 'name', 'role', 'date_joined')
        read_only_fields = fields


class UserCreateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ('email', 'name', 'role', 'password')

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)
