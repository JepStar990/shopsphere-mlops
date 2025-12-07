***

# README.md

# ShopSphere MLOps

End‑to‑end MLOps stack for retail analytics.  
It includes **data ingestion → feature building → model training & registry → API serving → monitoring & dashboards**, and stores artifacts on **MinIO (S3)**.

**Core use cases**

*   **CLV** (regression; pyfunc-ready): features & training, MLflow registry, FastAPI serving, drift monitoring with Evidently.
*   **Campaign Propensity** (classification): features & training, MLflow registry, FastAPI scoring, AUC drift monitoring.
*   **Recommender** (Implicit **ALS**): interactions → model training, MLflow **pyfunc**, FastAPI recommendations, coverage/novelty monitoring.
*   **Customer Segmentation** (KMeans): features → **pyfunc** pipeline, MLflow registry, FastAPI assignment.
*   **Dynamic Pricing** (elasticity regression baseline): features → training, registry, FastAPI suggestion + guardrails monitoring.

**Stack**

*   **Docker Compose** services:  
    `MinIO` (S3), `Postgres`, `MLflow` (Tracking+Registry), `Prefect Server`, `Prefect Worker`, `FastAPI`, `Prometheus`, `Grafana`, `Monitor Exporter`.
*   **Python**: Prefect 2.x, MLflow 2.16.x, pandas, scikit‑learn, **implicit**, LightFM (optional), Evidently, boto3, pyarrow.
*   **Artifacts**: MLflow → S3 (MinIO bucket `mlflow`).
*   **Dashboards**: Grafana “ShopSphere - MLOps Overview”.

***

## 1) Repository Layout

    shopsphere-mlops/
    ├─ infra/
    │  ├─ docker-compose.yaml
    │  ├─ env.example        # copy to .env before composing
    │  ├─ prometheus/prometheus.yml
    │  └─ grafana/provisioning/
    │     ├─ datasources/datasource.yml
    │     └─ dashboards/
    │        ├─ dashboards.yml
    │        └─ shopsphere.json
    ├─ docker/
    │  ├─ api/Dockerfile
    │  ├─ mlflow/Dockerfile
    │  ├─ prefect/Dockerfile
    │  └─ monitor/Dockerfile
    ├─ src/
    │  ├─ api/
    │  │  ├─ main.py
    │  │  └─ routers/
    │  │     ├─ clv.py
    │  │     ├─ propensity.py
    │  │     ├─ recommend.py
    │  │     ├─ pricing.py
    │  │     └─ segmentation.py
    │  ├─ common/
    │  │  ├─ io.py
    │  │  ├─ features.py
    │  │  └─ promotion.py
    │  ├─ monitoring/
    │  │  ├─ drift_report.py
    │  │  └─ metrics_exporter.py
    │  ├─ segmentation/train_kmeans_pyfunc.py
    │  ├─ recommender/train_als.py
    │  ├─ campaign_response/train.py
    │  └─ pricing/train.py
    ├─ prefect_flows/
    │  ├─ ingest_flow.py
    │  ├─ features_flow.py
    │  ├─ train_flow.py                        # CLV
    │  ├─ segmentation_train_flow.py
    │  ├─ recommender_train_flow.py
    │  ├─ campaign_train_flow.py
    │  ├─ pricing_train_flow.py
    │  ├─ monitor_flow.py                      # CLV drift
    │  ├─ monitor_propensity_auc.py
    │  ├─ monitor_recommender_metrics.py
    │  └─ monitor_pricing_alerts.py
    ├─ data/
    │  ├─ raw/      # your CSVs
    │  ├─ bronze/   # parquet created by flows
    │  ├─ gold/     # final features & UI
    │  ├─ reports/  # Evidently HTML
    │  └─ monitoring/  # *.txt metrics scraped by exporter
    ├─ requirements.txt
    └─ README.md

***

## 2) Quickstart (Linux VM)

> **Prereqs**
>
> *   Docker & Compose plugin installed.
> *   Your raw dataset CSVs placed in `data/raw/` (already done).

```bash
# 0) Prepare .env
cd ~/shopsphere-mlops/infra
cp env.example .env
# (Optional) edit secrets inside .env

# 1) Bring up the stack (first run could take several minutes)
docker compose up -d --build

# 2) Check services
docker compose ps

# 3) If the Prefect worker ever races the server, use ephemeral runs; see §5.
```

**UIs**

*   **MLflow**:         `http://<VM-IP>:5000`
*   **Prefect UI**:     `http://<VM-IP>:4200`
*   **MinIO Console**:  `http://<VM-IP>:9001` (login: MINIO creds from `.env`)
*   **Prometheus**:     `http://<VM-IP>:9090`
*   **Grafana**:        `http://<VM-IP>:3000` (admin / GF\_SECURITY\_ADMIN\_PASSWORD)

***

## 3) Build → Train → Promote (per use case)

> You can run flows **in the long‑lived worker** or **ephemerally**. Ephemeral runs are robust when a worker is not required.

### 3.1 CLV

```bash
# Ingest → Features → Train (register & promote)
docker compose exec prefect_worker python /app/prefect_flows/ingest_flow.py
docker compose exec prefect_worker python /app/prefect_flows/features_flow.py
docker compose exec prefect_worker python /app/prefect_flows/train_flow.py

# Restart API to load Production model
docker compose restart api
```

**Test**

```bash
curl -s -X POST http://localhost:8080/score/clv \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"59540","features":{"recency_days":12,"tx_count":5,"monetary":350,
       "avg_discount":0.05,"avg_quantity":2.0,"premium_tx_share":0.3,
       "events_view_count":10,"events_add_to_cart_count":3,"events_purchase_count":2,
       "avg_session_duration_sec":120,"age":48,"is_male":1,"loyalty_level":1}}'
```

### 3.2 Segmentation (KMeans pyfunc)

```bash
# Features + training + promote
docker compose exec prefect_worker python /app/prefect_flows/segmentation_train_flow.py
docker compose exec -T prefect_worker python - <<'PY'
from src.common.promotion import promote_latest_model
print("Promoted seg:", promote_latest_model("segmentation_model","Production"))
PY
docker compose restart api
```

**Test**

```bash
curl -s -X POST http://localhost:8080/segment \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"59540","features":{"recency_days":12,"tx_count":5,"monetary":350,"avg_discount":0.05,"avg_qty":2,"age":48,"loyalty_level":1}}'
```

### 3.3 Recommender (Implicit ALS pyfunc)

```bash
docker compose exec prefect_worker python /app/prefect_flows/recommender_train_flow.py
docker compose exec -T prefect_worker python - <<'PY'
from src.common.promotion import promote_latest_model
print("Promoted rec:", promote_latest_model("recommender_als_model","Production"))
PY
docker compose restart api
```

**Test**

```bash
curl -s -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"59540","k":5}'
```

### 3.4 Campaign Propensity

```bash
docker compose exec prefect_worker python /app/prefect_flows/campaign_train_flow.py
docker compose exec -T prefect_worker python - <<'PY'
from src.common.promotion import promote_latest_model
print("Promoted camp:", promote_latest_model("campaign_model","Production"))
PY
docker compose restart api
```

**Test**

```bash
curl -s -X POST http://localhost:8080/score/propensity \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"59540","features":{"age":48,"is_male":1,"loyalty_level":1,
       "uplift_mean":0.1,"events_view_count":10,"events_add_to_cart_count":3,"events_purchase_count":2}}'
```

### 3.5 Pricing

```bash
docker compose exec prefect_worker python /app/prefect_flows/pricing_train_flow.py
docker compose exec -T prefect_worker python - <<'PY'
from src.common.promotion import promote_latest_model
print("Promoted price:", promote_latest_model("pricing_model","Production"))
PY
docker compose restart api
```

**Test**

```bash
curl -s -X POST http://localhost:8080/price \
  -H "Content-Type: application/json" \
  -d '{"product_id":"408","features":{"avg_price":72.5,"units":100,"revenue":5000,
       "avg_discount":0.1,"premium_share":1.0},"min_price":50,"max_price":150}'
```

***

## 4) Monitoring & Dashboards

### 4.1 Metrics exporter

A lightweight container exposes gauges at `:9100/metrics`:

*   `clv_drift_score`
*   `propensity_auc`, `propensity_auc_delta`
*   `recommender_coverage`, `recommender_novelty`
*   `pricing_guardrail_violations`

These are read from `data/monitoring/*.txt`, which are **written by monitoring flows**.

### 4.2 Monitoring flows

```bash
# CLV drift (Evidently HTML to data/reports/clv_drift.html, plus drift score txt)
docker compose exec prefect_worker python /app/prefect_flows/monitor_flow.py

# Propensity AUC drift
docker compose exec prefect_worker python /app/prefect_flows/monitor_propensity_auc.py

# Recommender coverage & novelty
docker compose exec prefect_worker python /app/prefect_flows/monitor_recommender_metrics.py

# Pricing guardrails
docker compose exec prefect_worker python /app/prefect_flows/monitor_pricing_alerts.py
```

**Validate exporter**

```bash
curl -s http://localhost:9100/metrics | egrep "clv_drift_score|propensity_auc|recommender_coverage|recommender_novelty|pricing_guardrail_violations"
```

### 4.3 Grafana dashboard

Provisioned at: `infra/grafana/provisioning/dashboards/shopsphere.json`  
Title: **ShopSphere - MLOps Overview**

**Panels**

*   **API**: requests/sec by endpoint, latency (ms) by endpoint.
*   **CLV**: drift score (stat & series).
*   **Propensity**: AUC and AUC Δ vs reference.
*   **Recommender**: coverage (stat), novelty (log scale series).
*   **Pricing**: guardrail violations (stat).
*   **Health**: Prometheus job `up` summary.
*   **Links**: MLflow / Prefect / MinIO / Prometheus / Grafana.

Open Grafana: `http://<VM-IP>:3000`, search **ShopSphere - MLOps Overview**.

***

## 5) Running flows ephemerally (no worker needed)

If the worker isn’t ready, run any flow in a one‑off container:

```bash
docker compose run --rm prefect_worker python /app/prefect_flows/ingest_flow.py
docker compose run --rm prefect_worker python /app/prefect_flows/features_flow.py
docker compose run --rm prefect_worker python /app/prefect_flows/train_flow.py
```

Same applies for **monitoring** and other training flows.

***

## 6) Model Registry, Promotion & Artifacts

### 6.1 Promotion utility

We ship `src/common/promotion.py`:

```python
from mlflow.tracking import MlflowClient

def promote_latest_model(model_name: str, stage: str = "Production", archive_existing: bool = True) -> str | None:
    ...
```

Use it inside flows (or manually) to move the **latest** version to **Production**.

### 6.2 Artifacts

*   **Preferred**: MLflow `download_artifacts` resolves Registry/Run URIs

```python
import mlflow
from mlflow.artifacts import download_artifacts
mlflow.set_tracking_uri("http://mlflow:5000")
local = download_artifacts("models:/recommender_als_model/Production")
```

*   **Direct S3** via MinIO (for side files)

```python
import os, boto3
s3 = boto3.client("s3",
  endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
  aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
  aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
s3.download_file(Bucket="mlflow", Key="0/<run_id>/artifacts/path", Filename="/tmp/file")
```

***

## 7) API Reference

**Base**: `http://<VM-IP>:8080/`

*   `GET /health` → `{ "status": "ok" }`
*   `GET /metrics` → Prometheus exposition
*   `POST /score/clv`  
    Request: `{ customer_id, features{...} }`  
    Response: `{ ok, error, customer_id, clv_180d, latency_ms }`
*   `POST /score/propensity`  
    Request: `{ customer_id, features{...} }`  
    Response: `{ customer_id, prob_response, ok, error }`
*   `POST /segment`  
    Request: `{ customer_id, features{...} }`  
    Response: `{ customer_id, cluster_id, ok, error }`
*   `POST /recommend`  
    Request: `{ customer_id, k }`  
    Response: `{ customer_id, rec_list[], ok, error }`
*   `POST /price`  
    Request: `{ product_id, features{...}, min_price?, max_price? }`  
    Response: `{ product_id, price_suggested, ok, error }`

***

## 8) Configuration

### 8.1 `infra/.env`

```env
# MinIO
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Postgres
PGUSER=mlops
PGPASSWORD=mlops

# Grafana
GF_SECURITY_ADMIN_PASSWORD=admin
```

### 8.2 Environment (containers)

*   **MLflow/Workers** (already set in compose):  
    `MLFLOW_TRACKING_URI` = `http://mlflow:5000`  
    `MLFLOW_S3_ENDPOINT_URL` = `http://minio:9000`  
    `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ADDRESSING_STYLE=path`

***

## 9) Troubleshooting

| Symptom                                    | Cause                                        | Fix                                                                                                                  |
| ------------------------------------------ | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `service "prefect_worker" is not running`  | Worker exited or not started                 | `docker compose up -d --build prefect_worker` or use **ephemeral runs** `docker compose run --rm prefect_worker ...` |
| Prefect worker shows `ConnectError` to API | Worker starts before Prefect server is ready | Use compose **healthcheck** & `depends_on` (already provided)                                                        |
| MLflow artifact upload `NoSuchBucket`      | `mlflow` bucket not created                  | `mc mb -p local/mlflow` or use MinIO Console                                                                         |
| MLflow `NoCredentialsError`                | Missing S3 env in worker                     | Ensure env vars under `prefect_worker.environment`                                                                   |
| API returns fallback / can’t load model    | No Production model yet                      | Train, **promote**, then `docker compose restart api`                                                                |
| Grafana shows no metrics                   | Prometheus target misnamed                   | Ensure `prometheus.yml` contains jobs `api` and `monitor-exporter`                                                   |

***

## 10) Roadmap & Extensions

*   Schedule flows with Prefect deployments/automations.
*   CI/CD (GitHub Actions): linting, tests, image build, flow checks.
*   Canary model rollout via MLflow stages + feature flags.
*   Infra metrics (Node Exporter + cAdvisor) and add host/container dashboards.
*   Proper schema registry & contracts for features (e.g., Pydantic / Great Expectations).
*   Model explainability snapshots, feature attribution drift.

***

## 11) License / Credits

*   Open‑source components as per their licenses (MLflow, Prefect, Prometheus, Grafana, Evidently, Implicit, LightFM).
*   Sample code © ShopSphere project team.

***

