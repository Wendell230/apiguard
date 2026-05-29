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
import time
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
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.tree import export_text, plot_tree


def load_data(models_dir: Path):
    X_path = models_dir / 'X_preprocessed.npy'
    y_path = models_dir / 'y.npy'
    le_path = models_dir / 'label_encoder.pkl'
    cols_path = models_dir / 'colunas_modelo.pkl'

    if not X_path.exists():
        raise FileNotFoundError('Execute ml/preprocess.py primeiro.')

    X = np.load(X_path)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_encoded = np.load(y_path)
    le = joblib.load(le_path) if le_path.exists() else None
    colunas = joblib.load(cols_path) if cols_path.exists() else [f'f{i}' for i in range(X.shape[1])]

    # Decodifica inteiros → strings para que model.classes_ tenha os nomes reais
    y = le.inverse_transform(y_encoded) if le is not None else y_encoded.astype(str)

    print(f'[train] {X.shape[0]:,} amostras | {X.shape[1]} features')
    return X, y, le, colunas


def train(models_dir: str, version: str, output_dir: str,
          max_train: int = 300_000, max_test: int = 200_000):
    t_total_start = time.perf_counter()

    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, le, colunas = load_data(models_dir)
    classes = le.classes_.tolist() if le else list(map(str, np.unique(y)))

    # ── Split 70/30 estratificado ────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    del X, y  # libera o array completo da memória
    import gc; gc.collect()

    # ── Subamostragem estratificada do treino (max_train amostras) ─
    if len(X_train) > max_train:
        X_train, y_train = resample(
            X_train, y_train,
            n_samples=max_train,
            stratify=y_train,
            random_state=42,
        )
        print(f'[train] Subamostragem treino: {max_train:,} amostras estratificadas')

    # ── Subamostragem estratificada do teste (max_test amostras) ──
    if len(X_test) > max_test:
        X_test, y_test = resample(
            X_test, y_test,
            n_samples=max_test,
            stratify=y_test,
            random_state=42,
        )
        print(f'[train] Subamostragem teste:  {max_test:,} amostras estratificadas')

    print(f'[train] Treino: {len(X_train):,} | Teste: {len(X_test):,}')
    print(f'[train] Classes: {classes}')
    for cls in classes:
        n = (y_train == cls).sum()
        print(f'         {cls:<25} {n:>8,} ({n/len(y_train)*100:.1f}%)')

    # ── Parâmetros fixos do Random Forest ───────────────────────
    RF_PARAMS = dict(
        n_estimators=100,
        max_depth=20,
        max_samples=0.8,
        n_jobs=-1,
        random_state=42,
    )
    print(f'\n[train] Parâmetros: {RF_PARAMS}')

    # ── Treinamento ──────────────────────────────────────────────
    print(f'[train] Treinando Random Forest...')
    t0 = time.perf_counter()
    modelo = RandomForestClassifier(**RF_PARAMS)
    modelo.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    print(f'[train] Treino concluído em {t_fit/60:.1f} min')

    # ── Validação cruzada 5-fold no conjunto de treino ───────────
    print(f'[train] Validação cruzada (5-fold) nos {len(X_train):,} de treino...')
    t0 = time.perf_counter()
    scores_cv = cross_val_score(modelo, X_train, y_train,
                                cv=5, scoring='accuracy', n_jobs=-1)
    print(f'[train] CV concluído em {(time.perf_counter()-t0)/60:.1f} min')
    print(f'  Acurácias por fold: {np.round(scores_cv, 4)}')
    print(f'  Média: {np.mean(scores_cv):.4f} ± {np.std(scores_cv):.4f}')

    # ── Avaliação no teste ───────────────────────────────────────
    y_pred = modelo.predict(X_test)
    acc   = accuracy_score(y_test, y_pred)
    f1_w  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_w  = recall_score(y_test, y_pred, average='weighted', zero_division=0)

    report = classification_report(y_test, y_pred, target_names=classes,
                                   digits=4, zero_division=0)
    print('\n=== Avaliação no Conjunto de Teste ===')
    print(report)

    # ── Matriz de confusão ───────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot(
        ax=ax, colorbar=False, xticks_rotation=15
    )
    ax.set_title('Matriz de Confusão — Random Forest')
    plt.tight_layout()
    plt.savefig(output_dir / 'matriz_confusao.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Primeira árvore (visualização) ───────────────────────────
    estimator = modelo.estimators_[0]
    plt.figure(figsize=(20, 10))
    plot_tree(estimator, feature_names=colunas, class_names=classes,
              filled=True, max_depth=3)
    plt.title('Primeira Árvore do Random Forest (profundidade 3)')
    plt.savefig(output_dir / 'primeira_arvore_rf.png', dpi=150, bbox_inches='tight')
    plt.close()

    regras = export_text(estimator, feature_names=colunas, max_depth=5)
    with open(output_dir / 'regras_primeira_arvore_rf.txt', 'w') as f:
        f.write(regras)

    # ── Importância das variáveis ────────────────────────────────
    imp_df = pd.DataFrame({'Variável': colunas,
                           'Importância': modelo.feature_importances_})
    imp_df = imp_df.sort_values('Importância', ascending=False)

    print('=== Importância das Variáveis ===')
    print(imp_df.to_string(index=False))

    plt.figure(figsize=(10, max(6, 0.35 * len(imp_df))))
    sns.barplot(data=imp_df, x='Importância', y='Variável', palette='viridis')
    plt.title('Importância das Variáveis — Random Forest')
    plt.xlabel('Importância (Gini)')
    plt.tight_layout()
    plt.savefig(output_dir / 'importancia_variaveis_rf.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Salva modelo e artefatos ─────────────────────────────────
    model_path = models_dir / f'model_v{version}.pkl'
    joblib.dump(modelo, model_path)
    joblib.dump(colunas, models_dir / 'colunas_modelo.pkl')

    t_total = time.perf_counter() - t_total_start

    metadata = {
        'version': version,
        'trained_at': datetime.now().isoformat(),
        'training_time_min': round(t_total / 60, 2),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'rf_params': RF_PARAMS,
        'test_accuracy': float(acc),
        'test_f1_weighted': float(f1_w),
        'test_precision_weighted': float(prec_w),
        'test_recall_weighted': float(rec_w),
        'cv_accuracy_mean': float(np.mean(scores_cv)),
        'cv_accuracy_std': float(np.std(scores_cv)),
        'classes': classes,
        'model_file': str(model_path),
        'output_dir': str(output_dir),
    }
    with open(models_dir / f'model_v{version}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    # ── Relatório final ──────────────────────────────────────────
    print('\n' + '═' * 62)
    print(f'  RELATÓRIO FINAL — Random Forest v{version}')
    print('═' * 62)
    print(f'  Amostras treino    : {len(X_train):>10,}')
    print(f'  Amostras teste     : {len(X_test):>10,}')
    print(f'  Features           : {len(colunas):>10}')
    print(f'  n_estimators       : {RF_PARAMS["n_estimators"]:>10}')
    print(f'  max_depth          : {RF_PARAMS["max_depth"]:>10}')
    print(f'  max_samples        : {RF_PARAMS["max_samples"]:>10}')
    print('─' * 62)
    print(f'  Acurácia (teste)   : {acc:>10.4f}')
    print(f'  F1  weighted       : {f1_w:>10.4f}')
    print(f'  Precision weighted : {prec_w:>10.4f}')
    print(f'  Recall weighted    : {rec_w:>10.4f}')
    print(f'  CV Acurácia (5-fold): {np.mean(scores_cv):.4f} ± {np.std(scores_cv):.4f}')
    print('─' * 62)
    print(report)
    print(f'  Tempo total        : {t_total/60:.1f} min')
    print(f'  Modelo salvo em    : {model_path}')
    print('═' * 62)

    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina o modelo Random Forest.')
    parser.add_argument('--models-dir', default='ml/models/', help='Diretório dos artefatos .npy')
    parser.add_argument('--version', default='1.0', help='Versão do modelo')
    parser.add_argument('--output-dir', default='saida_DT', help='Diretório para gráficos')
    parser.add_argument('--max-train', type=int, default=300_000,
                        help='Máx. amostras estratificadas de treino (padrão: 300 000)')
    parser.add_argument('--max-test', type=int, default=200_000,
                        help='Máx. amostras estratificadas de teste  (padrão: 200 000)')
    args = parser.parse_args()
    train(args.models_dir, args.version, args.output_dir, args.max_train, args.max_test)
