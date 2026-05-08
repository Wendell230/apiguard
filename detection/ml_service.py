"""
Serviço singleton de predição ML.
Carrega o modelo .pkl ativo na inicialização e expõe o método predict().
Se nenhum modelo treinado existir, usa um simulador para desenvolvimento.
"""
import random
import time
from pathlib import Path
from typing import Any

import joblib

from django.conf import settings


class _MockModel:
    """Modelo simulado para desenvolvimento sem .pkl treinado."""

    CLASSES = ['BENIGN', 'SYN Flood', 'UDP Flood', 'HTTP Flood', 'DNS Amplification']
    # Distribuição realista: maioria benigna
    WEIGHTS = [0.70, 0.10, 0.08, 0.07, 0.05]

    def predict_proba(self, X):
        probs = []
        for _ in X:
            # Distribui probabilidade de forma aleatória mas ponderada
            chosen = random.choices(self.CLASSES, weights=self.WEIGHTS, k=1)[0]
            idx = self.CLASSES.index(chosen)
            row = [0.01] * len(self.CLASSES)
            row[idx] = 0.75 + random.random() * 0.20
            total = sum(row)
            probs.append([p / total for p in row])
        return probs

    def predict(self, X):
        probas = self.predict_proba(X)
        return [self.CLASSES[p.index(max(p))] for p in probas]

    @property
    def classes_(self):
        return self.CLASSES


class MLService:
    """Singleton que gerencia o ciclo de vida do modelo Random Forest."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model = None
        self._scaler = None
        self._is_mock = False
        self._load_active_model()
        self._initialized = True

    def _load_active_model(self):
        """Tenta carregar o modelo ativo do banco; cai no mock se não encontrar."""
        try:
            # Importação local para evitar importação circular no boot do Django
            from detection.models import MLModel
            active = MLModel.objects.filter(is_active=True).first()
            if active:
                model_path = Path(active.file_path)
                if model_path.exists():
                    self._model = joblib.load(model_path)
                    # Tenta carregar scaler na mesma pasta
                    scaler_path = model_path.parent / 'scaler.pkl'
                    if scaler_path.exists():
                        self._scaler = joblib.load(scaler_path)
                    self._is_mock = False
                    return
        except Exception:
            pass

        # Fallback para modelo simulado
        self._model = _MockModel()
        self._is_mock = True

    def reload(self):
        """Força o recarregamento do modelo (chamado após upload de novo .pkl)."""
        self._initialized = False
        self._load_active_model()
        self._initialized = True

    def predict(self, features_dict: dict[str, Any]) -> dict:
        """
        Classifica um vetor de features de tráfego de rede.

        Retorna:
            prediction: 'BENIGN' ou 'ATTACK'
            attack_type: classe específica do ataque
            probability: confiança da predição (0–1)
            response_time_ms: latência da inferência
            is_mock: se a predição veio do simulador
        """
        start = time.perf_counter()

        # Converte dict para lista ordenada de valores numéricos
        feature_vector = [float(v) for v in features_dict.values()]

        # Aplica scaler se disponível
        X = [feature_vector]
        if self._scaler is not None:
            X = self._scaler.transform(X)

        probas = self._model.predict_proba(X)[0]
        classes = self._model.classes_

        top_idx = probas.index(max(probas)) if isinstance(probas, list) else probas.argmax()
        attack_type = classes[top_idx]
        probability = float(max(probas) if isinstance(probas, list) else probas[top_idx])

        prediction = 'BENIGN' if attack_type == 'BENIGN' else 'ATTACK'

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            'prediction': prediction,
            'attack_type': attack_type,
            'probability': round(probability, 4),
            'response_time_ms': round(elapsed_ms, 3),
            'is_mock': self._is_mock,
        }


# Instância global — carregada uma única vez
ml_service = MLService()
