from prefect import flow, task
import pandas as pd
from pathlib import Path
from src.common.features import segmentation_features
from src.segmentation.train import train_kmeans

BRONZE = Path("/app/data/bronze")
GOLD = Path("/app/data/gold")
FEATS = GOLD / "segmentation_features.parquet"

@task
def build_features():
    customers = pd.read_parquet(BRONZE / "customers.parquet")
    transactions = pd.read_parquet(BRONZE / "transactions.parquet")
    feats = segmentation_features(customers, transactions)
    FEATS.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATS, index=False)

@task
def train():
    train_kmeans(str(FEATS), k=6)

@flow(name="segmentation_train")
def run():
    build_features()
    train()

if __name__ == "__main__":
    run()
