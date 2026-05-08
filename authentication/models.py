"""Modelo de usuário customizado com suporte a roles."""
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('O e-mail é obrigatório.')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('role', User.Role.ADMIN)
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    class Role(models.TextChoices):
        ADMIN = 'admin', 'Administrador'
        ANALYST = 'analyst', 'Analista'
        INTEGRATOR = 'integrator', 'Integrador'

    email = models.EmailField(unique=True, verbose_name='E-mail')
    name = models.CharField(max_length=150, verbose_name='Nome completo')
    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.ANALYST,
        verbose_name='Perfil',
    )
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    date_joined = models.DateTimeField(auto_now_add=True)

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name']

    class Meta:
        verbose_name = 'Usuário'
        verbose_name_plural = 'Usuários'

    def __str__(self):
        return f'{self.name} <{self.email}>'

    @property
    def is_admin(self):
        return self.role == self.Role.ADMIN
