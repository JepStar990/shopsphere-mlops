from fastapi import APIRouter
from pydantic import BaseModel
import os, pandas as pd
import mlflow, mlflow.pyfunc as pyfunc

router = APIRouter()
_model = None

@router.on_event("startup")
def load_model():
    global _model
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://mlflow:5000"))
        _model = pyfunc.load_model("models:/pricing_model/Production")
        print("Loaded pricing_model")
    except Exception as e:
        print(f"Could not load pricing_model: {e}")
        _model = None

class PricingRequest(BaseModel):
    product_id: str
    features: dict  # {"avg_price":..., "units":..., "revenue":..., "avg_discount":..., "premium_share":...}
    min_price: float | None = None
    max_price: float | None = None

@router.post("/")
def price(payload: PricingRequest):
    X = pd.DataFrame([payload.features])
    try:
        if _model is not None:
            sensitivity = float(_model.predict(X)[0])
            # simple rule-of-thumb price suggestion: move opposite sensitivity
            base = float(X.get("avg_price", pd.Series([payload.features.get("avg_price", 100.0)])).iloc[0])
            suggested = max(0.0, base * (1.0 - 0.2 * sensitivity))
            # apply guardrails
            if payload.min_price is not None:
                suggested = max(suggested, payload.min_price)
            if payload.max_price is not None:
                suggested = min(suggested, payload.max_price)
            return {"product_id": payload.product_id, "price_suggested": suggested, "ok": True}
        else:
            return {"product_id": payload.product_id, "price_suggested": payload.features.get("avg_price", 100.0), "ok": False, "error": "model_not_loaded"}
    except Exception as ex:
        return {"product_id": payload.product_id, "price_suggested": payload.features.get("avg_price", 100.0), "ok": False, "error": str(ex)}
