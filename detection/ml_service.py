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
        """
        Ordem de carregamento:
        1. Modelo marcado como ativo no banco de dados
        2. ml/models/modelo_rf_otimizado.pkl (arquivo do TCC, commitado no repo)
        3. Fallback: modelo simulado (_MockModel)
        """
        # 1. Modelo ativo no banco
        try:
            from detection.models import MLModel
            active = MLModel.objects.filter(is_active=True).first()
            if active:
                model_path = Path(active.file_path)
                if model_path.exists():
                    model = joblib.load(model_path)
                    cols_path = model_path.parent / 'colunas_modelo.pkl'
                    cols = joblib.load(cols_path) if cols_path.exists() else None
                    # Valida que o modelo e as colunas são compatíveis
                    if cols and hasattr(model, 'n_features_in_') and model.n_features_in_ != len(cols):
                        raise ValueError(
                            f'Incompatibilidade: modelo espera {model.n_features_in_} features, '
                            f'colunas_modelo.pkl tem {len(cols)}.'
                        )
                    self._model = model
                    self._feature_columns = cols
                    self._is_mock = False
                    return
        except Exception as e:
            print(f'[ml_service] AVISO ao carregar modelo do banco: {e}')

        # 2. Arquivo padrão do TCC commitado no repositório
        try:
            default_model = Path(settings.ML_MODELS_DIR) / 'modelo_rf_otimizado.pkl'
            default_cols  = Path(settings.ML_MODELS_DIR) / 'colunas_modelo.pkl'
            if default_model.exists():
                model = joblib.load(default_model)
                cols = joblib.load(default_cols) if default_cols.exists() else None
                if cols and hasattr(model, 'n_features_in_') and model.n_features_in_ != len(cols):
                    raise ValueError(
                        f'modelo_rf_otimizado.pkl espera {model.n_features_in_} features, '
                        f'colunas_modelo.pkl tem {len(cols)} — usando mock.'
                    )
                self._model = model
                self._feature_columns = cols
                self._is_mock = False
                return
        except Exception as e:
            print(f'[ml_service] AVISO ao carregar modelo padrão: {e}')

        # 3. Mock para desenvolvimento sem modelo
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
