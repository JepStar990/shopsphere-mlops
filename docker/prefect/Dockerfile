# docker/prefect/Dockerfile
FROM python:3.11-slim

# System deps (optional, useful for building some wheels)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Python deps (for flows)
RUN pip install --no-cache-dir \
    prefect \
    dvc[s3] \
    pandas \
    scikit-learn \
    xgboost \
    lightgbm \
    mlflow \
    boto3 \
    minio \
    pyarrow \
    evidently \
    prometheus-client \
    python-dotenv

WORKDIR /app

# We mount src/ and prefect_flows/ at runtime via docker-compose, so no COPY is needed here
# The agent command is provided in docker-compose.yml

