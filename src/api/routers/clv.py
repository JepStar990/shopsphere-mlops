
from fastapi import APIRouter
from pydantic import BaseModel
import time
import os

router = APIRouter()
_model = None

class CLVRequest(BaseModel):
    customer_id: str
    features: dict

@router.on_event("startup")
def load_model():
    global _model
    try:
        import mlflow.pyfunc as pyfunc
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        _model = pyfunc.load_model("models:/clv_model/Production")
        print("Loaded MLflow model: models:/clv_model/Production")
    except Exception as e:
        print(f"Could not load MLflow model: {e}")
        _model = None

@router.post("/")
def score(payload: CLVRequest):
    start = time.time()
    if _model is not None:
        try:
            pred = _model.predict([payload.features])[0]
            clv = float(pred)
        except Exception:
            clv = float(payload.features.get("monetary", 0.0))
    else:
        # Fallback: use monetary as proxy
        clv = float(payload.features.get("monetary", 0.0))

    latency_ms = (time.time() - start) * 1000
    try:
        from api.main import requests_total, score_latency_g
        requests_total.labels(endpoint="/score/clv").inc()
        score_latency_g.labels(endpoint="/score/clv").set(latency_ms)
    except Exception:
        pass
    return {"customer_id": payload.customer_id, "clv_180d": clv}

