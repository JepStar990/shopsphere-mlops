from prefect import flow, task
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from pathlib import Path
import os

FEATURES_PATH = Path("/app/data/gold/clv_features.parquet")
REGISTERED_MODEL_NAME = "clv_model"

@task
def set_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("clv_experiment")

@task
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)

@task
def train_and_register(df: pd.DataFrame) -> float:
    df = df.copy()
    # Target engineered in features: clv_180d
    y = df["clv_180d"].astype(float)
    features = ["recency_days","tx_count","monetary","avg_discount","avg_quantity",
                "premium_tx_share","events_view_count","events_add_to_cart_count",
                "events_purchase_count","avg_session_duration_sec","age","is_male","loyalty_level"]
    X = df[features].fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="clv_rf_baseline"):
        model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, preds)
        mlflow.log_metric("mape", float(mape))
        mlflow.log_params({"n_estimators": 200, "model": "RandomForestRegressor"})

        # Log model and register
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )
        return mape

@task
def promote_to_production():
    from mlflow.client import MlflowClient
    client = MlflowClient()
    # Get latest version in 'None' stage and move to 'Production'
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        return
    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True
    )

@flow(name="train_clv")
def train_clv_flow():
    set_mlflow()
    df = load_features()
    mape = train_and_register(df)
    print(f"CLV MAPE: {mape:.4f}")
    promote_to_production()

if __name__ == "__main__":
    train_clv_flow()
