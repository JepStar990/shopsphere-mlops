from fastapi import APIRouter
from pydantic import BaseModel
import os, pandas as pd

router = APIRouter()
_co = None
_user_items = None

class RecommendRequest(BaseModel):
    customer_id: str
    k: int = 5

@router.on_event("startup")
def load_artifacts():
    global _co, _user_items
    # Load user-item from gold; and cooccurrence artifact from last run if available
    try:
        ui_path = "/app/data/gold/user_item.parquet"
        _user_items = pd.read_parquet(ui_path)
    except Exception as e:
        print(f"Could not load {ui_path}: {e}")
    # Minimal: attempt to read cooccurrence from a fixed path (mounted artifact if you prefer).
    try:
        # If you logged to MLflow, you could download artifacts via MLflow API or S3.
        # Here we assume a local copy exists for simplicity.
        _co = pd.read_parquet("/app/data/gold/cooccurrence.parquet") if os.path.exists("/app/data/gold/cooccurrence.parquet") else None
    except Exception as e:
        print(f"Could not load cooccurrence: {e}")

@router.post("/")
def recommend(payload: RecommendRequest):
    k = payload.k
    # get items bought by the user
    if _user_items is None:
        return {"customer_id": payload.customer_id, "rec_list": [], "ok": False, "error": "no_user_items"}
    user_items = set(_user_items.loc[_user_items["customer_id"] == payload.customer_id, "product_id"])
    if not user_items:
        return {"customer_id": payload.customer_id, "rec_list": [], "ok": True}

    # If we have co-occurrence scores, rank candidates
    if _co is not None:
        candidates = _co[_co["item"].isin(user_items)].groupby("item_rec")["score"].sum().sort_values(ascending=False)
        recs = [{"product_id": str(pid), "score": float(score)} for pid, score in candidates.head(k).items()]
        return {"customer_id": payload.customer_id, "rec_list": recs, "ok": True}
    else:
        # fallback: recommend the most popular items not owned
        popular = (_user_items.groupby("product_id")["strength"].sum().sort_values(ascending=False)).index.tolist()
        popular = [p for p in popular if p not in user_items][:k]
        recs = [{"product_id": str(pid), "score": 1.0} for pid in popular]
        return {"customer_id": payload.customer_id, "rec_list": recs, "ok": True}
