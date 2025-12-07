from prefect import flow, task
from pathlib import Path
import os, pandas as pd
import mlflow, mlflow.pyfunc

BRONZE = Path("/app/data/bronze")
MON = Path("/app/data/monitoring")
VIOL_FILE = MON / "pricing_guardrail_violations.txt"

@task
def ensure_dirs():
    MON.mkdir(parents=True, exist_ok=True)

@task
def load_data():
    products = pd.read_parquet(BRONZE / "products.parquet")
    # Build a minimal feature frame for pricing inference
    tx = pd.read_parquet(BRONZE / "transactions.parquet")
    agg = tx.groupby("product_id").agg(
        units=("quantity","sum"),
        revenue=("gross_revenue","sum"),
        avg_discount=("discount_applied","mean")
    ).reset_index()
    df = agg.merge(products[["product_id","category","base_price","is_premium"]], on="product_id", how="left")
    df["premium_share"] = df["is_premium"].fillna(0.0)
    df["avg_price"] = df["base_price"].fillna(0.0)
    return df

@task
def load_model():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://mlflow:5000"))
    try:
        return mlflow.pyfunc.load_model("models:/pricing_model/Production")
    except Exception:
        return None

@task
def check_guardrails(model, df: pd.DataFrame):
    if model is None or df.empty:
        return 0
    # company policy guardrails (example)
    min_price = 0.5
    max_price = 2_000.0
    count = 0
    X = df[["avg_price","units","revenue","avg_discount","premium_share"]].fillna(0.0)
    preds = model.predict(X)
    for p in preds:
        if p < min_price or p > max_price:
            count += 1
    return count

@task
def write(count: int):
    VIOL_FILE.write_text(str(count))

@flow(name="monitor_pricing_guardrails")
def run():
    ensure_dirs()
    df = load_data()
    model = load_model()
    count = check_guardrails(model, df)
    write(count)
    print(f"[monitor] pricing_guardrail_violations={count}")

if __name__ == "__main__":
    run()
