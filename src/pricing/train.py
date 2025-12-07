import os
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

def train_pricing(feats_path: str):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("pricing_experiment")

    df = pd.read_parquet(feats_path).copy()
    # Target proxy: price_sensitivity (from features builder)
    y = df["price_sensitivity"].fillna(0.0)
    X = df[["avg_price","units","revenue","avg_discount","premium_share"]].fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with mlflow.start_run(run_name="pricing_gbr"):
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        mlflow.log_metric("r2", float(r2))
        mlflow.log_params({"model": "GBR"})
        mlflow.sklearn.log_model(model, "model", registered_model_name="pricing_model")
