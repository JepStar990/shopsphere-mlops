import os
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def train_kmeans(feats_path: str, k: int = 6):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("segmentation_experiment")

    df = pd.read_parquet(feats_path)
    X = df[["recency_days","tx_count","monetary","avg_discount","avg_qty","age","loyalty_level"]].fillna(0.0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    with mlflow.start_run(run_name=f"kmeans_k={k}"):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(Xs)

        mlflow.log_params({"k": k})
        # save cluster labels alongside customer_id as an artifact
        out = pd.DataFrame({"customer_id": df["customer_id"], "cluster_id": labels})
        out_path = "segmentation_labels.parquet"
        out.to_parquet(out_path, index=False)
        mlflow.log_artifact(out_path)

        # log model (scaler + kmeans) as a pipeline
        import pickle, tempfile
        pipe = {"scaler": scaler, "kmeans": km, "feature_cols": list(X.columns)}
        with tempfile.NamedTemporaryFile("wb", suffix=".pkl", delete=False) as f:
            pickle.dump(pipe, f)
            model_file = f.name
        mlflow.log_artifact(model_file, artifact_path="model")
        # You can also register a pyfunc wrapper later if desired
