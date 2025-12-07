# src/api/routers/recommend.py
from fastapi import APIRouter
from pydantic import BaseModel
import os, pandas as pd
import mlflow, mlflow.pyfunc as pyfunc

router = APIRouter()
_model = None

class RecommendRequest(BaseModel):
    customer_id: str
    k: int = 5

@router.on_event("startup")
def load_model():
    global _model
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://mlflow:5000"))
        _model = pyfunc.load_model("models:/recommender_als_model/Production")
        print("Loaded recommender_als_model")
    except Exception as e:
        print(f"Could not load recommender model: {e}")
        _model = None

@router.post("/")
def recommend(payload: RecommendRequest):
    if _model is None:
        return {"customer_id": payload.customer_id, "rec_list": [], "ok": False, "error": "model_not_loaded"}
    X = pd.DataFrame([{"customer_id": payload.customer_id, "k": payload.k}])
    try:
        out = _model.predict(X)[0]["rec_list"]
        return {"customer_id": payload.customer_id, "rec_list": out, "ok": True}
    except Exception as ex:
        return {"customer_id": payload.customer_id, "rec_list": [], "ok": False, "error": str(ex)}
