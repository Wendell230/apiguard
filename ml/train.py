"""
Treinamento do modelo Random Forest para detecção de DoS/DDoS.

Uso:
    python ml/train.py --models-dir ml/models/ --version 1.0

O script:
1. Carrega X_preprocessed.npy e y.npy gerados pelo preprocess.py
2. Treina RandomForestClassifier com GridSearchCV
3. Avalia com cross-validation 5-fold
4. Salva model_v{versao}.pkl + metadados JSON
5. Imprime relatório completo
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split


def load_data(models_dir: Path):
    X_path = models_dir / 'X_preprocessed.npy'
    y_path = models_dir / 'y.npy'

    if not X_path.exists() or not y_path.exists():
        print('[train] ERRO: execute ml/preprocess.py antes de treinar.')
        sys.exit(1)

    X = np.load(X_path)
    y = np.load(y_path)
    print(f'[train] Dados carregados: {X.shape[0]:,} amostras, {X.shape[1]} features.')
    return X, y


def grid_search(X_train, y_train) -> RandomForestClassifier:
    """Otimização de hiperparâmetros com GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20, 40],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None],
    }
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print('[train] Iniciando GridSearchCV (pode demorar alguns minutos)...')
    gs = GridSearchCV(
        base, param_grid, cv=cv, scoring='f1_weighted',
        n_jobs=-1, verbose=1, refit=True,
    )
    gs.fit(X_train, y_train)
    print(f'[train] Melhores parâmetros: {gs.best_params_}')
    print(f'[train] Melhor F1 (CV): {gs.best_score_:.4f}')
    return gs.best_estimator_


def cross_validate_model(model, X, y) -> dict:
    """Validação cruzada estratificada 5-fold."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(
        model, X, y, cv=cv,
        scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
        return_train_score=False,
    )
    return {
        'accuracy': float(results['test_accuracy'].mean()),
        'f1': float(results['test_f1_weighted'].mean()),
        'precision': float(results['test_precision_weighted'].mean()),
        'recall': float(results['test_recall_weighted'].mean()),
        'accuracy_std': float(results['test_accuracy'].std()),
    }


def train(models_dir: str, version: str, test_size: float = 0.2):
    models_dir = Path(models_dir)
    X, y = load_data(models_dir)

    # Carrega metadados do pré-processamento
    meta_path = models_dir / 'preprocessing_metadata.json'
    classes = None
    if meta_path.exists():
        with open(meta_path) as f:
            pre_meta = json.load(f)
        classes = pre_meta.get('classes', [])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print(f'[train] Treino: {len(X_train):,} | Teste: {len(X_test):,}')

    model = grid_search(X_train, y_train)

    # Avaliação no conjunto de teste
    y_pred = model.predict(X_test)
    print('\n[train] === Relatório de Classificação (conjunto de teste) ===')
    print(classification_report(y_test, y_pred, target_names=classes))

    # Validação cruzada
    print('[train] Executando cross-validation 5-fold...')
    cv_results = cross_validate_model(model, X, y)
    print(f'[train] CV Acurácia : {cv_results["accuracy"]:.4f} ± {cv_results["accuracy_std"]:.4f}')
    print(f'[train] CV F1        : {cv_results["f1"]:.4f}')
    print(f'[train] CV Precisão  : {cv_results["precision"]:.4f}')
    print(f'[train] CV Recall    : {cv_results["recall"]:.4f}')

    # Importância das features
    if classes:
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:10]
        print('\n[train] Top 10 features mais importantes:')
        feature_names = pre_meta.get('features', [f'f{i}' for i in range(X.shape[1])])
        for rank, idx in enumerate(sorted_idx, 1):
            name = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
            print(f'  {rank:2d}. {name}: {importances[idx]:.4f}')

    # Salva modelo
    model_filename = f'model_v{version}.pkl'
    model_path = models_dir / model_filename
    joblib.dump(model, model_path)
    print(f'\n[train] Modelo salvo: {model_path}')

    # Salva metadados
    metadata = {
        'version': version,
        'trained_at': datetime.now().isoformat(),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'test_accuracy': float((y_pred == y_test).mean()),
        'test_f1': float(f1_score(y_test, y_pred, average='weighted')),
        'test_precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'test_recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'cv': cv_results,
        'best_params': model.get_params(),
        'classes': classes or [],
        'model_file': str(model_path),
    }
    meta_out = models_dir / f'model_v{version}_metadata.json'
    with open(meta_out, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    print(f'[train] Metadados salvos: {meta_out}')

    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina o modelo Random Forest.')
    parser.add_argument('--models-dir', default='ml/models/', help='Diretório dos artefatos')
    parser.add_argument('--version', default='1.0', help='Versão do modelo (ex: 1.0)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporção do conjunto de teste')
    args = parser.parse_args()
    train(args.models_dir, args.version, args.test_size)
