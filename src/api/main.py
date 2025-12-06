from fastapi import FastAPI, Response
from api.routers import clv, propensity, recommend, pricing

# Prometheus metrics
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="ShopSphere ML APIs")

# Simple metrics
requests_total = Counter("api_requests_total", "Total API requests", ["endpoint"])
score_latency_g = Gauge("api_score_latency_ms", "Score latency in milliseconds", ["endpoint"])

# Routers
app.include_router(clv.router, prefix="/score/clv")
app.include_router(propensity.router, prefix="/score/propensity")
app.include_router(recommend.router, prefix="/recommend")
app.include_router(pricing.router, prefix="/price")


@app.get("/health")
def health():
    requests_total.labels(endpoint="/health").inc()
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
