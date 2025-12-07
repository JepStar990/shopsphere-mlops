from prefect import flow, task
from pathlib import Path
import pandas as pd
import os, mlflow, mlflow.pyfunc
import numpy as np

BRONZE = Path("/app/data/bronze")
MON = Path("/app/data/monitoring")
COVERAGE_FILE = MON / "recommender_coverage.txt"
NOVELTY_FILE = MON / "recommender_novelty.txt"

@task
def ensure_dirs():
    MON.mkdir(parents=True, exist_ok=True)

@task
def load_tx():
    tx = pd.read_parquet(BRONZE / "transactions.parquet")
    return tx

@task
def load_model():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://mlflow:5000"))
    try:
        return mlflow.pyfunc.load_model("models:/recommender_als_model/Production")
    except Exception:
        return None

@task
def compute_metrics(model, tx: pd.DataFrame, k: int = 5):
    if model is None or tx.empty:
        return 0.0, 0.0
    users = tx["customer_id"].astype(str).unique().tolist()
    items_pop = tx.groupby("product_id")["quantity"].sum()
    popularity = items_pop / items_pop.sum()  # normalized popularity
    inv_pop = lambda pid: 1.0 / (popularity.get(pid, 1e-9))

    # generate top-k recs per user (batch predict)
    X = pd.DataFrame([{"customer_id": u, "k": k} for u in users])
    out = model.predict(X)
    # coverage: distinct recommended items / total catalog
    rec_items = set()
    novelty_scores = []
    for r in out:
        rec_list = r.get("rec_list", [])
        for e in rec_list:
            pid = str(e["product_id"])
            rec_items.add(pid)
            novelty_scores.append(inv_pop(pid))
    coverage = len(rec_items) / tx["product_id"].nunique()
    novelty = float(np.mean(novelty_scores)) if novelty_scores else 0.0
    return coverage, novelty

@task
def write_metrics(coverage, novelty):
    COVERAGE_FILE.write_text(f"{coverage:.6f}")
    NOVELTY_FILE.write_text(f"{novelty:.6f}")

@flow(name="monitor_recommender")
def run():
    ensure_dirs()
    tx = load_tx()
    model = load_model()
    coverage, novelty = compute_metrics(model, tx, k=5)
    write_metrics(coverage, novelty)
    print(f"[monitor] recommender coverage={coverage:.4f} novelty={novelty:.4f}")

if __name__ == "__main__":
    run()
