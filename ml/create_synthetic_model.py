"""
Gera um modelo Random Forest treinado em dados sintéticos.
Usado para desenvolvimento e deploy quando o dataset real não está disponível.
Salva modelo_rf_otimizado.pkl e colunas_modelo.pkl em ml/models/.
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

COLUNAS_USADAS = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
    'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
    'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
    'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]

# Índices das features mais discriminativas para cada classe
IDX = {col: i for i, col in enumerate(COLUNAS_USADAS)}

CLASSES = ['BENIGN', 'SYN Flood', 'UDP Flood', 'HTTP Flood', 'DNS Amplification']
N = 800  # amostras por classe


def gerar_classe(classe: str, n: int) -> np.ndarray:
    """Gera n amostras sintéticas com padrões realistas para cada classe."""
    rng = np.random.default_rng(seed=CLASSES.index(classe))
    X = rng.uniform(0, 100, size=(n, len(COLUNAS_USADAS)))

    if classe == 'BENIGN':
        # Tráfego balanceado, pacotes médios, flags normais
        X[:, IDX['Tot Fwd Pkts']]   = rng.uniform(5, 50, n)
        X[:, IDX['Tot Bwd Pkts']]   = rng.uniform(4, 40, n)
        X[:, IDX['SYN Flag Cnt']]   = rng.uniform(0, 2, n)
        X[:, IDX['ACK Flag Cnt']]   = rng.uniform(5, 30, n)
        X[:, IDX['Flow Pkts/s']]    = rng.uniform(5, 50, n)
        X[:, IDX['Flow Byts/s']]    = rng.uniform(1000, 50000, n)
        X[:, IDX['Pkt Len Mean']]   = rng.uniform(200, 800, n)
        X[:, IDX['Flow Duration']]  = rng.uniform(500000, 5000000, n)

    elif classe == 'SYN Flood':
        # Muitos SYN, zero ACK, pacotes pequenos, unidirecional
        X[:, IDX['Tot Fwd Pkts']]   = rng.uniform(10000, 100000, n)
        X[:, IDX['Tot Bwd Pkts']]   = rng.uniform(0, 5, n)
        X[:, IDX['SYN Flag Cnt']]   = rng.uniform(10000, 100000, n)
        X[:, IDX['ACK Flag Cnt']]   = rng.uniform(0, 2, n)
        X[:, IDX['Flow Pkts/s']]    = rng.uniform(50000, 500000, n)
        X[:, IDX['Flow Byts/s']]    = rng.uniform(1000000, 10000000, n)
        X[:, IDX['Pkt Len Mean']]   = rng.uniform(40, 60, n)
        X[:, IDX['Flow Duration']]  = rng.uniform(100000, 1000000, n)

    elif classe == 'UDP Flood':
        # Pacotes grandes, alta taxa, sem flags TCP
        X[:, IDX['Tot Fwd Pkts']]   = rng.uniform(50000, 500000, n)
        X[:, IDX['Tot Bwd Pkts']]   = rng.uniform(0, 5, n)
        X[:, IDX['SYN Flag Cnt']]   = rng.uniform(0, 1, n)
        X[:, IDX['ACK Flag Cnt']]   = rng.uniform(0, 1, n)
        X[:, IDX['Flow Pkts/s']]    = rng.uniform(100000, 1000000, n)
        X[:, IDX['Flow Byts/s']]    = rng.uniform(100000000, 500000000, n)
        X[:, IDX['Pkt Len Mean']]   = rng.uniform(1000, 1472, n)
        X[:, IDX['Flow Duration']]  = rng.uniform(100000, 500000, n)

    elif classe == 'HTTP Flood':
        # PSH+ACK alto, bidirecional, pacotes médios-grandes
        X[:, IDX['Tot Fwd Pkts']]   = rng.uniform(5000, 50000, n)
        X[:, IDX['Tot Bwd Pkts']]   = rng.uniform(4000, 45000, n)
        X[:, IDX['PSH Flag Cnt']]   = rng.uniform(8000, 80000, n)
        X[:, IDX['ACK Flag Cnt']]   = rng.uniform(9000, 90000, n)
        X[:, IDX['SYN Flag Cnt']]   = rng.uniform(50, 200, n)
        X[:, IDX['Flow Pkts/s']]    = rng.uniform(5000, 50000, n)
        X[:, IDX['Flow Byts/s']]    = rng.uniform(5000000, 50000000, n)
        X[:, IDX['Pkt Len Mean']]   = rng.uniform(400, 1460, n)
        X[:, IDX['Flow Duration']]  = rng.uniform(1000000, 5000000, n)

    elif classe == 'DNS Amplification':
        # Resposta muito maior que requisição, UDP, alta assimetria
        X[:, IDX['Tot Fwd Pkts']]         = rng.uniform(1, 10, n)
        X[:, IDX['Tot Bwd Pkts']]         = rng.uniform(5000, 50000, n)
        X[:, IDX['TotLen Fwd Pkts']]      = rng.uniform(100, 1000, n)
        X[:, IDX['TotLen Bwd Pkts']]      = rng.uniform(5000000, 50000000, n)
        X[:, IDX['Bwd Pkt Len Mean']]     = rng.uniform(1000, 4000, n)
        X[:, IDX['Down/Up Ratio']]        = rng.uniform(50, 100, n)
        X[:, IDX['Flow Byts/s']]          = rng.uniform(10000000, 100000000, n)
        X[:, IDX['Flow Duration']]        = rng.uniform(100000, 1000000, n)
        X[:, IDX['SYN Flag Cnt']]         = rng.uniform(0, 1, n)

    return X


def main():
    output_dir = Path(__file__).parent / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)

    print('[synthetic] Gerando dados sintéticos...')
    X_list, y_list = [], []
    for cls in CLASSES:
        X_cls = gerar_classe(cls, N)
        X_list.append(X_cls)
        y_list.extend([cls] * N)

    X = np.vstack(X_list)
    y = np.array(y_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f'[synthetic] Treinando Random Forest ({len(X_train)} amostras, {X.shape[1]} features)...')
    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'[synthetic] Acurácia no teste (dados sintéticos): {acc:.4f}')
    print(classification_report(y_test, y_pred))

    # Salva no mesmo formato do script do TCC
    modelo_path = output_dir / 'modelo_rf_otimizado.pkl'
    cols_path   = output_dir / 'colunas_modelo.pkl'
    joblib.dump(modelo, modelo_path)
    joblib.dump(COLUNAS_USADAS, cols_path)

    print(f'[synthetic] modelo_rf_otimizado.pkl → {modelo_path}')
    print(f'[synthetic] colunas_modelo.pkl       → {cols_path}')
    print('[synthetic] Pronto! Agora rode: python manage.py runserver')


if __name__ == '__main__':
    main()
