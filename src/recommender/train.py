import os
import mlflow
import pandas as pd

def train_cooccurrence(ui_path: str):
    """
    Build simple co-occurrence scores: for each item, items co-purchased with it.
    Store a mapping as an artifact (CSV).
    """
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("recommender_experiment")

    ui = pd.read_parquet(ui_path)
    # build item -> set of users
    item_users = ui.groupby("product_id")["customer_id"].apply(set).to_dict()
    # co-occurrence: score(i,j) = |users(i) ∩ users(j)|
    items = list(item_users.keys())
    rows = []
    for i in items:
        ui_i = item_users[i]
        for j in items:
            if j == i:
                continue
            score = len(ui_i.intersection(item_users[j]))
            if score > 0:
                rows.append((i, j, score))
    co = pd.DataFrame(rows, columns=["item","item_rec","score"])

    with mlflow.start_run(run_name="cooccurrence"):
        out_path = "cooccurrence.parquet"
        co.to_parquet(out_path, index=False)
        mlflow.log_artifact(out_path)
        # You can register a pyfunc later; for now we’ll load this artifact in the API.
