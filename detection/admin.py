from django.contrib import admin

from .models import MLModel, TrafficLog


@admin.register(TrafficLog)
class TrafficLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'source_ip', 'protocol', 'prediction', 'attack_type', 'probability', 'response_time_ms')
    list_filter = ('prediction', 'attack_type', 'protocol')
    search_fields = ('source_ip',)
    readonly_fields = ('timestamp', 'features_json', 'response_time_ms')
    ordering = ('-timestamp',)


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'version', 'accuracy', 'is_active', 'created_at')
    list_filter = ('is_active',)
    readonly_fields = ('created_at',)
