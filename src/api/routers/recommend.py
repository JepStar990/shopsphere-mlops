from fastapi import APIRouter
from pydantic import BaseModel
import time

router = APIRouter()

class RecommendRequest(BaseModel):
    customer_id: str
    k: int = 5

@router.post("/")
def recommend(payload: RecommendRequest):
    start = time.time()
    recs = [{"product_id": f"P{i}", "score": 1.0 - i * 0.1} for i in range(payload.k)]
    latency_ms = (time.time() - start) * 1000
    try:
        from api.main import requests_total, score_latency_g
        requests_total.labels(endpoint="/recommend").inc()
        score_latency_g.labels(endpoint="/recommend").set(latency_ms)
    except Exception:
        pass
    return {"customer_id": payload.customer_id, "rec_list": recs}
