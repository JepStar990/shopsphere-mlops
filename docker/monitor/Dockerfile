FROM python:3.11-slim
RUN pip install --no-cache-dir prometheus-client
WORKDIR /app
COPY src/monitoring/metrics_exporter.py /app/metrics_exporter.py
CMD ["python", "/app/metrics_exporter.py"]
