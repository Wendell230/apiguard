"""
Serviço singleton de predição ML.
Carrega o modelo .pkl ativo e colunas_modelo.pkl na inicialização.
Fallback: modelo simulado se nenhum .pkl treinado existir.
"""
import random
import time
from pathlib import Path
from typing import Any

import joblib

from django.conf import settings


class _MockModel:
    """Modelo simulado para desenvolvimento sem .pkl treinado."""

    CLASSES = ['BENIGN', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris',
               'DDoS', 'PortScan', 'FTP-Patator', 'SSH-Patator', 'Bot', 'Infiltration',
               'Web Attack Brute Force', 'Web Attack XSS', 'Web Attack Sql Injection',
               'Heartbleed']
    WEIGHTS = [0.60, 0.06, 0.06, 0.05, 0.04, 0.05, 0.04, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.005, 0.005]

    def predict_proba(self, X):
        result = []
        for _ in X:
            chosen = random.choices(self.CLASSES, weights=self.WEIGHTS, k=1)[0]
            idx = self.CLASSES.index(chosen)
            row = [0.005] * len(self.CLASSES)
            row[idx] = 0.75 + random.random() * 0.20
            total = sum(row)
            result.append([p / total for p in row])
        return result

    def predict(self, X):
        return [self.CLASSES[p.index(max(p))] for p in self.predict_proba(X)]

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
        self._feature_columns = None
        self._is_mock = False
        self._load_active_model()
        self._initialized = True

    def _load_active_model(self):
        """Tenta carregar o modelo ativo do banco; fallback para mock."""
        try:
            from detection.models import MLModel
            active = MLModel.objects.filter(is_active=True).first()
            if active:
                model_path = Path(active.file_path)
                if model_path.exists():
                    self._model = joblib.load(model_path)
                    # Carrega lista de colunas salva pelo train.py
                    cols_path = model_path.parent / 'colunas_modelo.pkl'
                    if cols_path.exists():
                        self._feature_columns = joblib.load(cols_path)
                    self._is_mock = False
                    return
        except Exception:
            pass

        # Tenta carregar colunas_modelo.pkl global mesmo sem modelo ativo
        try:
            cols_path = Path(settings.ML_MODELS_DIR) / 'colunas_modelo.pkl'
            if cols_path.exists():
                self._feature_columns = joblib.load(cols_path)
        except Exception:
            pass

        self._model = _MockModel()
        self._is_mock = True

    def reload(self):
        """Recarrega o modelo após upload de novo .pkl."""
        self._initialized = False
        self._load_active_model()
        self._initialized = True

    def predict(self, features_dict: dict[str, Any]) -> dict:
        """
        Classifica um vetor de features de tráfego.

        features_dict: chaves são nomes originais do CSV (ex: 'Flow Duration').
        Retorna prediction, attack_type, probability, response_time_ms, is_mock.
        """
        start = time.perf_counter()

        # Ordena features conforme a lista salva no treino; 0.0 para ausentes
        if self._feature_columns:
            feature_vector = [float(features_dict.get(col, 0.0)) for col in self._feature_columns]
        else:
            feature_vector = [float(v) for v in features_dict.values()]

        X = [feature_vector]
        probas = self._model.predict_proba(X)[0]
        classes = self._model.classes_

        if isinstance(probas, list):
            top_idx = probas.index(max(probas))
            probability = float(probas[top_idx])
        else:
            top_idx = int(probas.argmax())
            probability = float(probas[top_idx])

        attack_type = classes[top_idx]
        prediction = 'BENIGN' if str(attack_type).upper() == 'BENIGN' else 'ATTACK'

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            'prediction': prediction,
            'attack_type': str(attack_type),
            'probability': round(probability, 4),
            'response_time_ms': round(elapsed_ms, 3),
            'is_mock': self._is_mock,
        }


ml_service = MLService()
