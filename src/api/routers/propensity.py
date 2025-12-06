from fastapi import APIRouter
from pydantic import BaseModel
import time

router = APIRouter()

class PropensityRequest(BaseModel):
    customer_id: str
    features: dict

@router.post("/")
def score(payload: PropensityRequest):
    start = time.time()
    prob = 0.5  # Placeholder
    latency_ms = (time.time() - start) * 1000
    try:
        from api.main import requests_total, score_latency_g
        requests_total.labels(endpoint="/score/propensity").inc()
        score_latency_g.labels(endpoint="/score/propensity").set(latency_ms)
    except Exception:
        pass
    return {"customer_id": payload.customer_id, "prob_response": prob}
