FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Pin Prefect to 2.x (to match server) and MLflow to match server (2.16)
RUN pip install --no-cache-dir \
    "prefect==2.20.18" \
    "mlflow==2.16.0" \
    dvc[s3] \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    boto3 \
    minio \
    pyarrow \
    evidently \
    prometheus-client \
    python-dotenv

WORKDIR /app

