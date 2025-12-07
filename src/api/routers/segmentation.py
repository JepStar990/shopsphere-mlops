from fastapi import APIRouter
from pydantic import BaseModel
import os, pandas as pd
import mlflow, mlflow.pyfunc as pyfunc

router = APIRouter()
_model = None

class SegmentationRequest(BaseModel):
    customer_id: str
    features: dict

@router.on_event("startup")
def load_model():
    global _model
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://mlflow:5000"))
        _model = pyfunc.load_model("models:/segmentation_model/Production")
        print("Loaded segmentation_model")
    except Exception as e:
        print(f"Could not load segmentation_model: {e}")
        _model = None

@router.post("/")
def segment(payload: SegmentationRequest):
    if _model is None:
        return {"customer_id": payload.customer_id, "cluster_id": None, "ok": False, "error": "model_not_loaded"}
    X = pd.DataFrame([payload.features])
    try:
        cluster_id = int(_model.predict(X)[0])
        return {"customer_id": payload.customer_id, "cluster_id": cluster_id, "ok": True}
    except Exception as ex:
        return {"customer_id": payload.customer_id, "cluster_id": None, "ok": False, "error": str(ex)}
