"""
Pré-processamento do dataset 5G DoS/DDoS.

Uso:
    python ml/preprocess.py --input entrada/5G_DoS_DDoS_dataset.csv --output ml/models/

O script:
1. Carrega CSV do dataset 5G DoS/DDoS
2. Trata valores nulos e infinitos (substitui por 0)
3. Salva colunas_modelo.pkl para uso posterior na inferência
4. Salva X_preprocessed.npy e y.npy para o treino
Obs: não usa StandardScaler — conforme pipeline do TCC.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Colunas exatas do dataset (mesma ordem do TCC)
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

LABEL_COLUMN = 'Label'


def preprocess(input_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'[preprocess] Carregando: {input_path}')
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    # Converte features para numérico; erros viram NaN
    for col in COLUNAS_USADAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Preenche NaN e inf com 0 (mesma lógica do TCC)
    df[COLUNAS_USADAS] = df[COLUNAS_USADAS].fillna(0)
    df[COLUNAS_USADAS] = df[COLUNAS_USADAS].replace([np.inf, -np.inf], 0)

    X = df[COLUNAS_USADAS].values
    y = df[LABEL_COLUMN].values

    print(f'[preprocess] {len(X):,} amostras | {X.shape[1]} features')
    print(f'[preprocess] Distribuição:\n{pd.Series(y).value_counts().to_string()}')

    # Codifica labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Salva artefatos
    np.save(output_dir / 'X_preprocessed.npy', X)
    np.save(output_dir / 'y.npy', y_encoded)
    joblib.dump(COLUNAS_USADAS, output_dir / 'colunas_modelo.pkl')
    joblib.dump(le, output_dir / 'label_encoder.pkl')

    metadata = {
        'features': COLUNAS_USADAS,
        'classes': le.classes_.tolist(),
        'n_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'class_distribution': {cls: int((y == cls).sum()) for cls in le.classes_},
    }
    with open(output_dir / 'preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f'[preprocess] Artefatos salvos em: {output_dir}')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pré-processa o dataset 5G DoS/DDoS.')
    parser.add_argument('--input', required=True, help='Caminho do CSV')
    parser.add_argument('--output', default='ml/models/', help='Diretório de saída')
    args = parser.parse_args()
    preprocess(args.input, args.output)
