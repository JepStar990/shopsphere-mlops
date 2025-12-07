import os
import mlflow, mlflow.pyfunc
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class SegmentationModel(mlflow.pyfunc.PythonModel):
    def __init__(self, scaler, kmeans, feature_cols):
        self.scaler = scaler
        self.kmeans = kmeans
        self.feature_cols = feature_cols

    def predict(self, context, model_input):
        """
        model_input: DataFrame with features matching self.feature_cols
        Returns: list of cluster_id integers
        """
        X = model_input[self.feature_cols].fillna(0.0)
        Xs = self.scaler.transform(X)
        labels = self.kmeans.predict(Xs)
        return labels.tolist()

def train_segmentation_pyfunc(feats_path: str, k: int = 6):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("segmentation_experiment")

    df = pd.read_parquet(feats_path)
    feature_cols = ["recency_days","tx_count","monetary","avg_discount","avg_qty","age","loyalty_level"]
    X = df[feature_cols].fillna(0.0)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(Xs)

    with mlflow.start_run(run_name=f"kmeans_pyfunc_k={k}"):
        mlflow.log_params({"k": k})
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SegmentationModel(scaler, km, feature_cols),
            registered_model_name="segmentation_model"
        )
