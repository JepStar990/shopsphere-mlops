from prefect import flow, task
from pathlib import Path
import os, pandas as pd
from sklearn.metrics import roc_auc_score
import mlflow, mlflow.pyfunc

BRONZE = Path("/app/data/bronze")
GOLD = Path("/app/data/gold")
MON = Path("/app/data/monitoring")
AUC_FILE = MON / "campaign_auc.txt"
AUC_DELTA_FILE = MON / "campaign_auc_delta.txt"

@task
def ensure_dirs():
    MON.mkdir(parents=True, exist_ok=True)

@task
def load_data():
    # Features built earlier
    feats = pd.read_parquet(GOLD / "campaign_features.parquet")
    # Labels proxy: purchase_event > 0
    y = (feats.get("events_purchase_count", 0) > 0).astype(int).values
    return feats, y

@task
def load_model():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://mlflow:5000"))
    try:
        model = mlflow.pyfunc.load_model("models:/campaign_model/Production")
        return model
    except Exception:
        return None

@task
def compute_auc(model, feats: pd.DataFrame, y):
    if model is None:
        return 0.0, 0.0, 0.0
    X = feats[["age","is_male","loyalty_level","uplift_mean",
               "events_view_count","events_add_to_cart_count","events_purchase_count"]].fillna(0.0)
    # split reference/current windows by date if available; otherwise first 60% vs last 40%
    n = len(X)
    split = int(n*0.6)
    X_ref, y_ref = X.iloc[:split], y[:split]
    X_cur, y_cur = X.iloc[split:], y[split:]
    # probability
    if hasattr(model, "predict_proba"):
        p_ref = model.predict_proba(X_ref)[:,1]
        p_cur = model.predict_proba(X_cur)[:,1]
    else:
        p_ref = model.predict(X_ref)
        p_cur = model.predict(X_cur)
    auc_ref = roc_auc_score(y_ref, p_ref) if len(set(y_ref)) > 1 else 0.0
    auc_cur = roc_auc_score(y_cur, p_cur) if len(set(y_cur)) > 1 else 0.0
    return auc_ref, auc_cur, auc_cur - auc_ref

@task
def write_metrics(auc_ref, auc_cur, auc_delta):
    AUC_FILE.write_text(f"{auc_cur:.6f}")
    AUC_DELTA_FILE.write_text(f"{auc_delta:.6f}")

@flow(name="monitor_propensity_auc")
def run():
    ensure_dirs()
    feats, y = load_data()
    model = load_model()
    auc_ref, auc_cur, auc_delta = compute_auc(model, feats, y)
    write_metrics(auc_ref, auc_cur, auc_delta)
    print(f"[monitor] propensity_auc={auc_cur:.4f} delta={auc_delta:.4f}")

if __name__ == "__main__":
    run()
