# APIGuard — Detecção de DoS/DDoS em Redes 5G e IoT

API RESTful para classificação em tempo real de tráfego de rede usando Random Forest treinado no dataset **CIC-DDoS2019**.

## Tecnologias

- Python 3.11+ / Django 4.2 / Django REST Framework
- SimpleJWT (autenticação)
- scikit-learn (modelo Random Forest)
- SQLite (desenvolvimento)

---

## Instalação e configuração

```bash
# 1. Clone o repositório e acesse o diretório
git clone <url> apiguard && cd apiguard

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure o ambiente
cp .env.example .env
# Edite o .env e defina SECRET_KEY, DEBUG, etc.

# 5. Aplique as migrações
python manage.py migrate

# 6. Crie um superusuário (admin)
python manage.py createsuperuser --email admin@apiguard.com

# 7. Inicie o servidor
python manage.py runserver
```

A API estará disponível em `http://localhost:8000/api/`.

---

## Treinamento do modelo ML

> Necessita do dataset CIC-DDoS2019 (CSV).  
> Download: https://www.unb.ca/cic/datasets/ddos-2019.html

```bash
# Passo 1: pré-processamento
python ml/preprocess.py --input /caminho/para/dataset.csv --output ml/models/

# Passo 2: treinamento
python ml/train.py --models-dir ml/models/ --version 1.0

# Passo 3: registre o modelo no banco (via admin ou API)
# POST /api/model/update/ com o arquivo model_v1.0.pkl
```

---

## Executar os testes

```bash
python manage.py test tests
```

---

## Endpoints

### Autenticação

#### POST /api/auth/login/
```bash
curl -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@apiguard.com", "password": "suasenha"}'
```
Resposta:
```json
{
  "access": "<jwt_access_token>",
  "refresh": "<jwt_refresh_token>",
  "user": {"id": 1, "email": "admin@apiguard.com", "name": "Admin", "role": "admin"}
}
```

#### POST /api/auth/refresh/
```bash
curl -X POST http://localhost:8000/api/auth/refresh/ \
  -H "Content-Type: application/json" \
  -d '{"refresh": "<jwt_refresh_token>"}'
```

#### POST /api/auth/logout/
```bash
curl -X POST http://localhost:8000/api/auth/logout/ \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"refresh": "<jwt_refresh_token>"}'
```

#### GET /api/auth/me/
```bash
curl http://localhost:8000/api/auth/me/ \
  -H "Authorization: Bearer <access_token>"
```

---

### Predição de tráfego

#### POST /api/predict/

Estrutura do JSON de entrada:
```json
{
  "source_ip": "192.168.1.100",
  "protocol": "TCP",
  "packet_count": 1000,
  "duration": 5.0,
  "flow_duration": 5000000.0,
  "total_fwd_packets": 500.0,
  "total_bwd_packets": 500.0,
  "total_length_fwd_packets": 50000.0,
  "total_length_bwd_packets": 50000.0,
  "fwd_packet_length_max": 1514.0,
  "fwd_packet_length_min": 64.0,
  "fwd_packet_length_mean": 100.0,
  "fwd_packet_length_std": 20.0,
  "bwd_packet_length_max": 1514.0,
  "bwd_packet_length_min": 64.0,
  "bwd_packet_length_mean": 100.0,
  "bwd_packet_length_std": 20.0,
  "flow_bytes_s": 10000.0,
  "flow_packets_s": 200.0,
  "flow_iat_mean": 5000.0,
  "flow_iat_std": 1000.0,
  "flow_iat_max": 10000.0,
  "flow_iat_min": 100.0,
  "syn_flag_count": 1.0,
  "rst_flag_count": 0.0,
  "psh_flag_count": 100.0,
  "ack_flag_count": 499.0,
  "avg_packet_size": 100.0,
  "avg_fwd_segment_size": 100.0,
  "avg_bwd_segment_size": 100.0
}
```

```bash
curl -X POST http://localhost:8000/api/predict/ \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

Resposta:
```json
{
  "log_id": 42,
  "prediction": "ATTACK",
  "attack_type": "SYN Flood",
  "probability": 0.9231,
  "response_time_ms": 1.452,
  "is_mock": false,
  "timestamp": "2026-05-08T10:00:00Z"
}
```

---

### Logs de tráfego

#### GET /api/logs/
```bash
# Lista paginada
curl "http://localhost:8000/api/logs/" \
  -H "Authorization: Bearer <access_token>"

# Com filtros
curl "http://localhost:8000/api/logs/?prediction=ATTACK&date_from=2026-05-01&source_ip=10.0.0.1" \
  -H "Authorization: Bearer <access_token>"
```

#### GET /api/logs/{id}/
```bash
curl http://localhost:8000/api/logs/42/ \
  -H "Authorization: Bearer <access_token>"
```

---

### Dashboard

#### GET /api/dashboard/
```bash
curl http://localhost:8000/api/dashboard/ \
  -H "Authorization: Bearer <access_token>"
```

Resposta:
```json
{
  "today": {
    "total": 1500,
    "attacks": 45,
    "benign": 1455,
    "pct_benign": 97.0
  },
  "avg_latency_ms": 1.8,
  "top_attacks_7d": [
    {"attack_type": "SYN Flood", "count": 120},
    {"attack_type": "UDP Flood", "count": 60}
  ],
  "active_model": {
    "name": "RandomForest CIC-DDoS2019",
    "version": "1.0",
    "accuracy": 0.9923,
    "is_mock": false
  }
}
```

---

### Modelos ML (somente admin)

#### POST /api/model/update/
```bash
curl -X POST http://localhost:8000/api/model/update/ \
  -H "Authorization: Bearer <admin_access_token>" \
  -F "name=RandomForest CIC-DDoS2019" \
  -F "version=1.1" \
  -F "accuracy=0.9930" \
  -F "description=Treinado com dados de abril/2026" \
  -F "model_file=@ml/models/model_v1.1.pkl"
```

#### GET /api/model/list/
```bash
curl http://localhost:8000/api/model/list/ \
  -H "Authorization: Bearer <access_token>"
```

---

## Classes de ataque detectadas

| Classe              | Descrição                         |
|---------------------|-----------------------------------|
| `BENIGN`            | Tráfego legítimo                  |
| `SYN Flood`         | Ataque de inundação TCP SYN       |
| `UDP Flood`         | Ataque de inundação UDP           |
| `HTTP Flood`        | Ataque de inundação HTTP          |
| `DNS Amplification` | Amplificação via DNS              |

---

## Estrutura do projeto

```
apiguard/
├── manage.py
├── requirements.txt
├── .env.example
├── apiguard/           # settings, urls, wsgi, asgi
├── authentication/     # modelo User, JWT, roles
├── detection/          # predict, logs, dashboard, ml_service
├── ml/
│   ├── preprocess.py   # pré-processamento CIC-DDoS2019
│   ├── train.py        # treino RandomForest + GridSearchCV
│   └── models/         # .pkl e metadados salvos aqui
└── tests/              # test_auth.py, test_detection.py
```

## Roles de usuário

| Role         | Predict | Logs | Dashboard | Upload modelo |
|--------------|---------|------|-----------|---------------|
| `admin`      | ✅      | ✅   | ✅        | ✅            |
| `analyst`    | ✅      | ✅   | ✅        | ❌            |
| `integrator` | ✅      | ✅   | ✅        | ❌            |
