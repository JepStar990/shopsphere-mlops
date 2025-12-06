from fastapi import APIRouter
from pydantic import BaseModel
import time, os
import pandas as pd

router = APIRouter()
_model = None

class CLVRequest(BaseModel):
    customer_id: str
    features: dict

@router.on_event("startup")
def load_model():
    global _model
    try:
        import mlflow
        import mlflow.pyfunc as pyfunc
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        _model = pyfunc.load_model("models:/clv_model/Production")
        print("Loaded MLflow model: models:/clv_model/Production")
    except Exception as e:
        print(f"Could not load MLflow model: {e}")
        _model = None

@router.post("/")
def score(payload: CLVRequest):
    start = time.time()
    try:
        if _model is None:
            raise RuntimeError("Model not loaded. Ensure clv_model is in Production and restart API.")

        # Coerce to DataFrame for pyfunc
        X = pd.DataFrame([payload.features])
        pred = _model.predict(X)[0]
        clv = float(pred)
        ok = True
        err = None
    except Exception as ex:
        ok = False
        err = str(ex)
        # Fallback: return monetary if provided, else 0
        clv = float(payload.features.get("monetary", 0.0))

    latency_ms = (time.time() - start) * 1000.0
    try:
        from api.main import requests_total, score_latency_g
        requests_total.labels(endpoint="/score/clv").inc()
        score_latency_g.labels(endpoint="/score/clv").set(latency_ms)
    except Exception:
        pass

    return {"ok": ok, "error": err, "customer_id": payload.customer_id, "clv_180d": clv, "latency_ms": latency_ms}
