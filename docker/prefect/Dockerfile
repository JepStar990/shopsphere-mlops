FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Align Prefect/MLflow to your server versions and add recommender libs
RUN pip install --no-cache-dir \
    "prefect==2.20.18" \
    "mlflow==2.16.0" \
    pandas scikit-learn xgboost lightgbm \
    boto3 minio pyarrow scipy numpy \
    evidently dvc[s3] prometheus-client python-dotenv \
    implicit \
    lightfm

WORKDIR /app
