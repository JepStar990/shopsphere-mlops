import os
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def train_campaign_classifier(feats_path: str, label_path: str = None):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("campaign_response_experiment")

    df = pd.read_parquet(feats_path).copy()
    # Create synthetic label if none: conversion proxy from engagement
    if label_path and os.path.exists(label_path):
        y = pd.read_parquet(label_path)["converted"].astype(int)
    else:
        y = ((df.get("events_purchase_count", 0) > 0).astype(int))

    X = df[["age","is_male","loyalty_level","uplift_mean",
            "events_view_count","events_add_to_cart_count","events_purchase_count"]].fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    with mlflow.start_run(run_name="rf_campaign"):
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        mlflow.log_metric("auc", float(auc))
        mlflow.log_params({"model": "RandomForestClassifier", "n_estimators": 300})
        mlflow.sklearn.log_model(model, "model", registered_model_name="campaign_model")
