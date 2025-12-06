from fastapi import APIRouter
from pydantic import BaseModel
import time

router = APIRouter()

class CLVRequest(BaseModel):
    customer_id: str
    features: dict

@router.post("/")
def score(payload: CLVRequest):
    # Placeholder scoring; integrate MLflow model later
    start = time.time()
    clv = float(payload.features.get("monetary", 0.0))
    latency_ms = (time.time() - start) * 1000
    try:
        from api.main import requests_total, score_latency_g
        requests_total.labels(endpoint="/score/clv").inc()
        score_latency_g.labels(endpoint="/score/clv").set(latency_ms)
    except Exception:
        pass
    return {"customer_id": payload.customer_id, "clv_180d": clv}
