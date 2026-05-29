"""
Pré-processamento do dataset CIC-DDoS2019 (11 arquivos CSV).

Uso — processar todos os CSVs e salvar artefatos para treino:
    python ml/preprocess.py --csv-dir /caminho/csvs --output ml/models/

Uso — modo teste (primeiras 1000 linhas de cada arquivo):
    python ml/preprocess.py --csv-dir /caminho/csvs --test-mode

Uso legado — arquivo único (compatibilidade retroativa):
    python ml/preprocess.py --input arquivo.csv --output ml/models/

O script:
1. Lê os 11 CSVs do CIC-DDoS2019 em chunks (encoding cp1252, sep auto)
2. Strip em nomes de colunas e valores Infinity/NaN
3. Mapeia labels para as classes da API
4. Renomeia as 26 features para os nomes abreviados compatíveis com a API
5. Salva X_preprocessed.npy, y.npy, colunas_modelo.pkl, label_encoder.pkl
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────
#  Arquivos do dataset CIC-DDoS2019
# ─────────────────────────────────────────────────────────────
CSV_FILES = [
    'DrDoS_NTP.csv',
    'DrDoS_DNS.csv',
    'TFTP.csv',
    'Syn.csv',
    'UDPLag.csv',
    'DrDoS_UDP.csv',
    'DrDoS_SSDP.csv',
    'DrDoS_SNMP.csv',
    'DrDoS_NetBIOS.csv',
    'DrDoS_MSSQL.csv',
    'DrDoS_LDAP.csv',
]

# ─────────────────────────────────────────────────────────────
#  26 features do CIC-DDoS2019 (nomes longos, após strip)
#  → renomeadas para os nomes abreviados usados pela API (COLUMN_MAP)
# ─────────────────────────────────────────────────────────────
CIC_TO_ABBREV = {
    'Flow Duration':               'Flow Duration',
    'Total Fwd Packets':           'Tot Fwd Pkts',
    'Total Backward Packets':      'Tot Bwd Pkts',
    'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
    'Total Length of Bwd Packets': 'TotLen Bwd Pkts',
    'Fwd Packet Length Max':       'Fwd Pkt Len Max',
    'Fwd Packet Length Min':       'Fwd Pkt Len Min',
    'Fwd Packet Length Mean':      'Fwd Pkt Len Mean',
    'Fwd Packet Length Std':       'Fwd Pkt Len Std',
    'Bwd Packet Length Max':       'Bwd Pkt Len Max',
    'Bwd Packet Length Min':       'Bwd Pkt Len Min',
    'Bwd Packet Length Mean':      'Bwd Pkt Len Mean',
    'Bwd Packet Length Std':       'Bwd Pkt Len Std',
    'Flow Bytes/s':                'Flow Byts/s',
    'Flow Packets/s':              'Flow Pkts/s',
    'Flow IAT Mean':               'Flow IAT Mean',
    'Flow IAT Std':                'Flow IAT Std',
    'Flow IAT Max':                'Flow IAT Max',
    'Flow IAT Min':                'Flow IAT Min',
    'SYN Flag Count':              'SYN Flag Cnt',
    'RST Flag Count':              'RST Flag Cnt',
    'PSH Flag Count':              'PSH Flag Cnt',
    'ACK Flag Count':              'ACK Flag Cnt',
    'Average Packet Size':         'Pkt Size Avg',
    'Avg Fwd Segment Size':        'Fwd Seg Size Avg',
    'Avg Bwd Segment Size':        'Bwd Seg Size Avg',
}

# Nomes finais que serão salvos em colunas_modelo.pkl (compatíveis com a API)
COLUNAS_MODELO = list(CIC_TO_ABBREV.values())

# ─────────────────────────────────────────────────────────────
#  Mapeamento de labels do dataset → classes da API
# ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    'BENIGN':       'BENIGN',
    # Amplificação / reflexão
    'DrDoS_NTP':    'DNS Amplification',
    'DrDoS_DNS':    'DNS Amplification',
    'DrDoS_SSDP':   'DNS Amplification',
    'DrDoS_SNMP':   'DNS Amplification',
    'DrDoS_NetBIOS':'DNS Amplification',
    'DrDoS_MSSQL':  'DNS Amplification',
    'DrDoS_LDAP':   'DNS Amplification',
    # SYN Flood
    'Syn':          'SYN Flood',
    'SYN':          'SYN Flood',
    # UDP Flood / TFTP
    'UDPLag':       'UDP Flood',
    'UDP-lag':      'UDP Flood',
    'DrDoS_UDP':    'UDP Flood',
    'UDP':          'UDP Flood',
    'TFTP':         'UDP Flood',
}

# Valores que representam infinito no CSV (como strings)
_INF_VALUES = ['Infinity', '-Infinity', 'infinity', '-infinity',
               'inf', '-inf', 'Inf', '-Inf', 'INF', '-INF']

# ─────────────────────────────────────────────────────────────
#  Colunas do dataset legado (76 features, nomes abreviados)
#  Mantido para compatibilidade com --input modo antigo
# ─────────────────────────────────────────────────────────────
COLUNAS_USADAS_LEGADO = [
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


# ─────────────────────────────────────────────────────────────
#  Funções auxiliares
# ─────────────────────────────────────────────────────────────

def _find_label_col(columns):
    """Encontra a coluna de label (pode vir com espaços ou variações de case)."""
    for col in columns:
        if col.strip().lower() == 'label':
            return col
    return None


def _clean_chunk(chunk, cic_cols, label_col):
    """
    Limpa um chunk:
    - substitui valores Infinity por NaN
    - converte features para float
    - remove linhas com NaN
    - mapeia labels
    Retorna DataFrame filtrado ou None se vazio.
    """
    # Substitui strings de infinito por NaN
    chunk.replace(_INF_VALUES, np.nan, inplace=True)

    # Converte features para numérico
    for col in cic_cols:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

    # Remove float inf gerados por pd.to_numeric('Infinity') → float('inf')
    chunk[cic_cols] = chunk[cic_cols].replace([np.inf, -np.inf], np.nan)

    # Remove linhas com NaN em qualquer feature
    chunk.dropna(subset=cic_cols, inplace=True)

    if chunk.empty:
        return None

    # Normaliza e mapeia labels
    chunk[label_col] = chunk[label_col].astype(str).str.strip()
    chunk['_label_mapped'] = chunk[label_col].map(LABEL_MAP)
    chunk = chunk[chunk['_label_mapped'].notna()].copy()

    if chunk.empty:
        return None

    return chunk


def _load_single_csv(path, chunk_size=50_000, max_rows=None):
    """
    Lê um CSV do CIC-DDoS2019 em chunks e converte direto para numpy.

    Retorna (X: float32 array, y: str array) ou (None, None) em caso de erro.
    Converter para numpy por chunk evita acumular DataFrames na memória.
    max_rows: para de ler após este número de linhas válidas (None = sem limite).
    """
    path = Path(path)
    cic_cols = list(CIC_TO_ABBREV.keys())
    abbrev_cols = list(CIC_TO_ABBREV.values())
    X_parts = []
    y_parts = []
    rows_collected = 0

    try:
        reader = pd.read_csv(
            path,
            sep=None,
            engine='python',
            encoding='cp1252',
            on_bad_lines='skip',
            chunksize=chunk_size,
        )
        for chunk in reader:
            if max_rows is not None and rows_collected >= max_rows:
                break

            chunk.columns = chunk.columns.str.strip()

            label_col = _find_label_col(chunk.columns)
            if label_col is None:
                continue

            for col in cic_cols:
                if col not in chunk.columns:
                    chunk[col] = 0.0

            chunk = chunk[cic_cols + [label_col]].copy()
            cleaned = _clean_chunk(chunk, cic_cols, label_col)
            del chunk
            if cleaned is None:
                continue

            # Trunca se atingiu o limite
            if max_rows is not None:
                remaining = max_rows - rows_collected
                if len(cleaned) > remaining:
                    cleaned = cleaned.iloc[:remaining]

            # Renomeia e converte para numpy imediatamente — libera o DataFrame
            cleaned.rename(columns=CIC_TO_ABBREV, inplace=True)
            X_parts.append(cleaned[abbrev_cols].values.astype(np.float32))
            y_parts.append(cleaned['_label_mapped'].values)
            rows_collected += len(cleaned)
            del cleaned

    except Exception as e:
        print(f'  [ERRO ao ler {path.name}] {e}')
        return None, None

    if not X_parts:
        return None, None

    return np.concatenate(X_parts), np.concatenate(y_parts)


# ─────────────────────────────────────────────────────────────
#  Função principal: carrega todos os CSVs
# ─────────────────────────────────────────────────────────────

def load_all_csvs(csv_dir='.', test_mode=False, chunk_size=50_000,
                  max_rows_per_file=500_000, output_dir=None):
    """
    Itera sobre os 11 arquivos do CIC-DDoS2019, lê em chunks e mantém
    apenas arrays numpy em memória (sem acumular DataFrames).

    Parâmetros
    ----------
    csv_dir : str | Path
    test_mode : bool        — se True, limita a 100 k linhas por arquivo
    chunk_size : int        — linhas por chunk de leitura (padrão: 50 000)
    max_rows_per_file : int — amostras máximas por arquivo (padrão: 500 000)
    output_dir : str | Path | None — se informado, salva artefatos para train.py

    Retorna
    -------
    X : np.ndarray  shape (n, 26)  float64
    y : np.ndarray  shape (n,)     str
    """
    import gc

    csv_dir = Path(csv_dir)
    limit = 100_000 if test_mode else max_rows_per_file
    X_parts = []
    y_parts = []
    total_rows = 0

    for filename in CSV_FILES:
        fpath = csv_dir / filename
        if not fpath.exists():
            print(f'[preprocess] AVISO: {filename} não encontrado em {csv_dir}')
            continue

        print(f'[preprocess] ▶ {filename} ...', end=' ', flush=True)
        X_file, y_file = _load_single_csv(fpath, chunk_size=chunk_size, max_rows=limit)

        if X_file is None or len(X_file) == 0:
            print('vazio ou sem dados válidos.')
            continue

        n = len(X_file)
        total_rows += n
        classes, counts = np.unique(y_file, return_counts=True)
        dist_str = '  |  '.join(f'{c}: {v:,}' for c, v in zip(classes, counts))
        print(f'{n:,} linhas  [{dist_str}]')

        X_parts.append(X_file)
        y_parts.append(y_file)
        # Libera referências imediatamente para o GC recuperar memória
        del X_file, y_file
        gc.collect()

    if not X_parts:
        print('\n[preprocess] ERRO: nenhum arquivo carregado.')
        return np.array([]), np.array([])

    print(f'\n[preprocess] Juntando {len(X_parts)} arquivo(s)...')
    X = np.concatenate(X_parts).astype(np.float64)
    y = np.concatenate(y_parts)
    del X_parts, y_parts
    gc.collect()

    print(f'[preprocess] Total: {total_rows:,} amostras | {len(COLUNAS_MODELO)} features')
    classes, counts = np.unique(y, return_counts=True)
    print('[preprocess] Distribuição final:')
    for cls, cnt in zip(classes, counts):
        print(f'  {cls:<25} {cnt:>10,}  ({cnt/total_rows*100:.1f}%)')

    if output_dir is not None:
        _save_artifacts(X, y, output_dir)

    return X, y


def _save_artifacts(X, y, output_dir):
    """Salva artefatos de pré-processamento para uso em train.py."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    np.save(output_dir / 'X_preprocessed.npy', X)
    np.save(output_dir / 'y.npy', y_encoded)
    joblib.dump(COLUNAS_MODELO, output_dir / 'colunas_modelo.pkl')
    joblib.dump(le, output_dir / 'label_encoder.pkl')

    metadata = {
        'features': COLUNAS_MODELO,
        'n_features': len(COLUNAS_MODELO),
        'classes': le.classes_.tolist(),
        'n_samples': int(len(X)),
        'class_distribution': {cls: int((y == cls).sum()) for cls in le.classes_},
        'label_map': LABEL_MAP,
        'source_files': CSV_FILES,
    }
    with open(output_dir / 'preprocessing_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f'\n[preprocess] Artefatos salvos em: {output_dir}')
    print(f'  colunas_modelo.pkl  → {len(COLUNAS_MODELO)} features')
    print(f'  X_preprocessed.npy → {X.shape}')
    print(f'  y.npy              → {y_encoded.shape}')
    print(f'  Classes: {le.classes_.tolist()}')


# ─────────────────────────────────────────────────────────────
#  Função legada: arquivo CSV único
# ─────────────────────────────────────────────────────────────

def preprocess(input_path, output_dir):
    """Modo legado: processa um único CSV com as 76 features abreviadas."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'[preprocess] Carregando (modo legado): {input_path}')
    df = pd.read_csv(input_path, encoding='cp1252', on_bad_lines='skip', low_memory=False)
    df.columns = df.columns.str.strip()

    colunas = COLUNAS_USADAS_LEGADO
    for col in colunas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[colunas] = df[colunas].replace([np.inf, -np.inf], np.nan).fillna(0)

    label_col = _find_label_col(df.columns) or 'Label'
    X = df[colunas].values
    y = df[label_col].values

    print(f'[preprocess] {len(X):,} amostras | {X.shape[1]} features')
    print(f'[preprocess] Distribuição:\n{pd.Series(y).value_counts().to_string()}')

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    np.save(output_dir / 'X_preprocessed.npy', X)
    np.save(output_dir / 'y.npy', y_encoded)
    joblib.dump(colunas, output_dir / 'colunas_modelo.pkl')
    joblib.dump(le, output_dir / 'label_encoder.pkl')

    metadata = {
        'features': colunas,
        'classes': le.classes_.tolist(),
        'n_samples': int(len(X)),
        'n_features': int(X.shape[1]),
        'class_distribution': {cls: int((y == cls).sum()) for cls in le.classes_},
    }
    with open(output_dir / 'preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f'[preprocess] Artefatos salvos em: {output_dir}')
    return metadata


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pré-processa o dataset CIC-DDoS2019 (11 arquivos CSV).'
    )
    # Modo novo: múltiplos CSVs
    parser.add_argument('--csv-dir', default=None,
                        help='Diretório com os 11 CSVs do CIC-DDoS2019')
    parser.add_argument('--test-mode', action='store_true',
                        help='Lê apenas as primeiras 100 k linhas de cada arquivo')
    parser.add_argument('--max-rows', type=int, default=500_000,
                        help='Máx. de linhas por arquivo no modo completo (padrão: 500 000)')
    parser.add_argument('--chunk-size', type=int, default=50_000,
                        help='Linhas por chunk ao ler (padrão: 50 000)')

    # Modo legado: CSV único
    parser.add_argument('--input', default=None,
                        help='(Legado) Caminho de um único CSV')

    parser.add_argument('--output', default='ml/models/',
                        help='Diretório de saída dos artefatos (padrão: ml/models/)')

    args = parser.parse_args()

    if args.input:
        # Modo legado
        preprocess(args.input, args.output)
    elif args.csv_dir:
        # Modo CIC-DDoS2019
        load_all_csvs(
            csv_dir=args.csv_dir,
            test_mode=args.test_mode,
            chunk_size=args.chunk_size,
            max_rows_per_file=args.max_rows,
            output_dir=None if args.test_mode else args.output,
        )
    else:
        parser.error('Informe --csv-dir ou --input.')
