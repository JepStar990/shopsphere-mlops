FROM python:3.11-slim

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
    pyarrow

WORKDIR /app

# Copy flows and src for future runs/exec
COPY prefect_flows /app/prefect_flows
COPY src /app/src

EXPOSE 4200
