FROM python:3.11-slim

RUN pip install --no-cache-dir evidently pandas pyarrow

WORKDIR /app
COPY src/monitoring /app/monitoring

CMD ["python", "-c", "print('Evidently container ready')"]
