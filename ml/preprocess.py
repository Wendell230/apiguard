"""
Pré-processamento do dataset CIC-DDoS2019.

Uso:
    python ml/preprocess.py --input /caminho/para/dataset.csv --output ml/models/

O script:
1. Carrega CSV(s) do CIC-DDoS2019
2. Trata valores nulos e infinitos
3. Remove features com variância zero
4. Normaliza com StandardScaler
5. Salva scaler.pkl e X_preprocessed.npy / y.npy para o treino
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold


# Features numéricas a usar (subconjunto relevante CIC-DDoS2019)
FEATURE_COLUMNS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
]

LABEL_COLUMN = 'Label'


def load_dataset(input_path: Path) -> pd.DataFrame:
    """Carrega um CSV ou todos os CSVs de um diretório."""
    if input_path.is_dir():
        frames = [pd.read_csv(p) for p in input_path.glob('*.csv')]
        if not frames:
            raise FileNotFoundError(f'Nenhum CSV encontrado em {input_path}')
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(input_path)

    # Normaliza nomes de colunas (remove espaços extras)
    df.columns = df.columns.str.strip()
    print(f'[preprocess] Dataset carregado: {len(df):,} amostras, {df.shape[1]} colunas.')
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extrai features e label; remove NaN e inf."""
    # Seleciona apenas colunas existentes
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available)
    if missing:
        print(f'[preprocess] AVISO: colunas ausentes no dataset → {missing}')

    X = df[available].copy()
    y = df[LABEL_COLUMN].copy()

    # Remove linhas sem label
    mask = y.notna()
    X, y = X[mask], y[mask]

    # Substitui inf por NaN e preenche com mediana da coluna
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(numeric_only=True), inplace=True)

    # Garante tipos numéricos
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    print(f'[preprocess] Após limpeza: {len(X):,} amostras, {X.shape[1]} features.')
    return X, y


def remove_zero_variance(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove features com variância zero (constantes)."""
    selector = VarianceThreshold(threshold=0)
    selector.fit(X)
    kept = X.columns[selector.get_support()].tolist()
    removed = list(set(X.columns) - set(kept))
    if removed:
        print(f'[preprocess] Features removidas (variância zero): {removed}')
    X = X[kept]
    return X, kept


def preprocess(input_path: str, output_dir: str):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    X, y = clean(df)
    X, kept_features = remove_zero_variance(X)

    # Codifica labels para inteiros
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Salva artefatos
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    joblib.dump(le, output_dir / 'label_encoder.pkl')
    np.save(output_dir / 'X_preprocessed.npy', X_scaled)
    np.save(output_dir / 'y.npy', y_encoded)

    metadata = {
        'features': kept_features,
        'classes': le.classes_.tolist(),
        'n_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'class_distribution': {
            cls: int((y == cls).sum()) for cls in le.classes_
        },
    }
    with open(output_dir / 'preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f'[preprocess] Artefatos salvos em: {output_dir}')
    print(f'[preprocess] Classes: {le.classes_.tolist()}')
    print(f'[preprocess] Distribuição:\n{pd.Series(y).value_counts().to_string()}')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pré-processa o dataset CIC-DDoS2019.')
    parser.add_argument('--input', required=True, help='CSV ou diretório com CSVs')
    parser.add_argument('--output', default='ml/models/', help='Diretório de saída')
    args = parser.parse_args()
    preprocess(args.input, args.output)
