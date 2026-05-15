"""
Treinamento do modelo Random Forest — pipeline do TCC 5G DoS/DDoS.

Uso:
    python ml/train.py --models-dir ml/models/ --version 1.0 [--output-dir saida_DT]

Fluxo:
1. Carrega X_preprocessed.npy e y.npy (gerados pelo preprocess.py)
2. Treina RandomForest inicial (n_estimators=100, max_depth=5)
3. Ajusta hiperparâmetros com GridSearchCV (cv=5)
4. Avalia com validação cruzada 5-fold
5. Gera gráficos: matriz de confusão, árvore, importância das variáveis
6. Salva modelo como model_v{versao}.pkl + colunas_modelo.pkl + metadados JSON
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')  # backend sem display — funciona em servidor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import export_text, plot_tree


def load_data(models_dir: Path):
    X_path = models_dir / 'X_preprocessed.npy'
    y_path = models_dir / 'y.npy'
    le_path = models_dir / 'label_encoder.pkl'
    cols_path = models_dir / 'colunas_modelo.pkl'

    if not X_path.exists():
        raise FileNotFoundError('Execute ml/preprocess.py primeiro.')

    X = np.load(X_path)
    y = np.load(y_path)
    le = joblib.load(le_path) if le_path.exists() else None
    colunas = joblib.load(cols_path) if cols_path.exists() else [f'f{i}' for i in range(X.shape[1])]

    print(f'[train] {X.shape[0]:,} amostras | {X.shape[1]} features')
    return X, y, le, colunas


def train(models_dir: str, version: str, output_dir: str, test_size: float = 0.3):
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, le, colunas = load_data(models_dir)
    classes = le.classes_.tolist() if le else list(map(str, np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f'[train] Treino: {len(X_train):,} | Teste: {len(X_test):,}')

    # --- Modelo inicial ---
    print('\n[train] Treinando modelo inicial (n_estimators=100, max_depth=5)...')
    modelo_inicial = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo_inicial.fit(X_train, y_train)
    y_pred_inicial = modelo_inicial.predict(X_test)

    print('\n=== Avaliação Inicial ===')
    print('Acurácia:', accuracy_score(y_test, y_pred_inicial))
    print(classification_report(y_test, y_pred_inicial, target_names=classes))

    # Validação cruzada do modelo inicial
    scores_cv = cross_val_score(modelo_inicial, X, y, cv=5, scoring='accuracy')
    print('=== Validação Cruzada (modelo inicial) ===')
    print('Acurácias por fold:', scores_cv)
    print('Acurácia média:', np.mean(scores_cv))

    # Matriz de confusão — modelo inicial
    cm = confusion_matrix(y_test, y_pred_inicial)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.title('Matriz de Confusão — Modelo Inicial')
    plt.savefig(output_dir / 'matriz_confusao_inicial.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[train] Gráfico salvo: {output_dir}/matriz_confusao_inicial.png')

    # --- GridSearchCV ---
    print('\n=== Ajuste de Hiperparâmetros (GridSearchCV) ===')
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 8, 12],
        'min_samples_split': [2, 4, 8],
        'min_samples_leaf': [1, 2, 4],
    }
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1,
    )
    gs.fit(X, y)
    print('Melhores parâmetros:', gs.best_params_)
    print('Melhor acurácia (CV):', gs.best_score_)

    modelo_otimizado = gs.best_estimator_

    # --- Avaliação do modelo otimizado no teste ---
    y_pred = modelo_otimizado.predict(X_test)
    print('\n=== Avaliação Modelo Otimizado ===')
    print('Acurácia treino:', accuracy_score(y_train, modelo_otimizado.predict(X_train)))
    print('Acurácia teste:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=classes))

    # --- Visualização da primeira árvore ---
    estimator = modelo_otimizado.estimators_[0]
    plt.figure(figsize=(20, 10))
    plot_tree(estimator, feature_names=colunas, class_names=classes, filled=True, max_depth=3)
    plt.title('Primeira Árvore do Random Forest Otimizado')
    plt.savefig(output_dir / 'primeira_arvore_rf_otimizada.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Regras da primeira árvore
    regras = export_text(estimator, feature_names=colunas)
    with open(output_dir / 'regras_primeira_arvore_rf.txt', 'w') as f:
        f.write(regras)
    print(f'[train] Regras salvas: {output_dir}/regras_primeira_arvore_rf.txt')

    # --- Importância das variáveis ---
    importancias = modelo_otimizado.feature_importances_
    importancia_df = pd.DataFrame({'Variável': colunas, 'Importância': importancias})
    importancia_df = importancia_df[importancia_df['Importância'] > 0].sort_values('Importância', ascending=False)

    print('\n=== Importância das Variáveis ===')
    print(importancia_df.to_string(index=False))

    plt.figure(figsize=(10, max(6, 0.3 * len(importancia_df))))
    sns.barplot(data=importancia_df, x='Importância', y='Variável', palette='viridis')
    plt.title('Importância das Variáveis no Random Forest')
    plt.xlabel('Importância')
    plt.ylabel('Variável')
    plt.tight_layout()
    plt.savefig(output_dir / 'importancia_variaveis_rf.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Salva modelo e artefatos ---
    model_filename = f'model_v{version}.pkl'
    model_path = models_dir / model_filename
    joblib.dump(modelo_otimizado, model_path)

    # Garante que colunas_modelo.pkl está na pasta de modelos
    joblib.dump(colunas, models_dir / 'colunas_modelo.pkl')

    metadata = {
        'version': version,
        'trained_at': datetime.now().isoformat(),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'test_f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'test_precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'test_recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'cv_accuracy_mean': float(np.mean(scores_cv)),
        'best_params': gs.best_params_,
        'classes': classes,
        'model_file': str(model_path),
        'output_dir': str(output_dir),
    }
    meta_path = models_dir / f'model_v{version}_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    print(f'\n[train] Modelo salvo: {model_path}')
    print(f'[train] Metadados: {meta_path}')
    print(f'[train] Gráficos em: {output_dir}/')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina o modelo Random Forest.')
    parser.add_argument('--models-dir', default='ml/models/', help='Diretório dos artefatos .npy')
    parser.add_argument('--version', default='1.0', help='Versão do modelo')
    parser.add_argument('--output-dir', default='saida_DT', help='Diretório para gráficos')
    parser.add_argument('--test-size', type=float, default=0.3, help='Proporção do conjunto de teste')
    args = parser.parse_args()
    train(args.models_dir, args.version, args.output_dir, args.test_size)
