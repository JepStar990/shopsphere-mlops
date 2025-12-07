from fastapi import APIRouter
from pydantic import BaseModel
import os, pandas as pd
import mlflow.pyfunc as pyfunc
import mlflow

router = APIRouter()
_model = None

@router.on_event("startup")
def load_model():
    global _model
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://mlflow:5000"))
        _model = pyfunc.load_model("models:/campaign_model/Production")
        print("Loaded campaign_model")
    except Exception as e:
        print(f"Could not load campaign_model: {e}")
        _model = None

class PropensityRequest(BaseModel):
    customer_id: str
    features: dict

@router.post("/")
def score(payload: PropensityRequest):
    X = pd.DataFrame([payload.features])
    try:
        if _model is not None:
            prob = float(_model.predict_proba(X)[0,1]) if hasattr(_model, "predict_proba") else float(_model.predict(X)[0])
            ok, err = True, None
        else:
            prob, ok, err = 0.5, False, "model_not_loaded"
    except Exception as ex:
        prob, ok, err = 0.5, False, str(ex)
    return {"customer_id": payload.customer_id, "prob_response": prob, "ok": ok, "error": err}

