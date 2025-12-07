from fastapi import APIRouter
from pydantic import BaseModel
import os, pickle
import pandas as pd

router = APIRouter()
_model = None

@router.on_event("startup")
def load_model():
    # Load latest artifact for simplicity (from mlruns on disk or MinIO)
    # Here we expect a local artifact; for production use MLflow pyfunc or artifacts via S3.
    # We’ll scan a conventional path if you choose to mount mlruns.
    pass

class SegmentationRequest(BaseModel):
    customer_id: str
    features: dict  # same columns used in training

@router.post("/")
def segment(payload: SegmentationRequest):
    # minimal inline inference: require same feature columns
    X = pd.DataFrame([payload.features])
    # if you saved the scaler+kmeans pipeline to a known location, load it and predict
    # For now, return a placeholder or compute simple heuristic segment:
    # You can wire real loading by reading from MinIO via boto3.
    score = X.get("monetary", pd.Series([0.0])).iloc[0]
    cluster_id = 0 if score < 100 else 1
    return {"customer_id": payload.customer_id, "cluster_id": int(cluster_id)}
