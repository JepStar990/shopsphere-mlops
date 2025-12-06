FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow[extras]==2.16 \
    boto3 \
    psycopg2-binary

EXPOSE 5000

# Env vars supplied by docker-compose
CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000
