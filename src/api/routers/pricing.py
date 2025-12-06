from fastapi import APIRouter
from pydantic import BaseModel
import time

router = APIRouter()

class PricingRequest(BaseModel):
    product_id: str
    features: dict

@router.post("/")
def price(payload: PricingRequest):
    start = time.time()
    price_suggested = float(payload.features.get("base_price", 100.0))
    latency_ms = (time.time() - start) * 1000
    try:
        from api.main import requests_total, score_latency_g
        requests_total.labels(endpoint="/price").inc()
        score_latency_g.labels(endpoint="/price").set(latency_ms)
    except Exception:
        pass
    return {"product_id": payload.product_id, "price_suggested": price_suggested}
