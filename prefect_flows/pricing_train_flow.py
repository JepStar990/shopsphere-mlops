from prefect import flow, task
import pandas as pd
from pathlib import Path
from src.common.features import pricing_features
from src.pricing.train import train_pricing

BRONZE = Path("/app/data/bronze")
GOLD = Path("/app/data/gold")
FEATS = GOLD / "pricing_features.parquet"

@task
def build_features():
    tx = pd.read_parquet(BRONZE / "transactions.parquet")
    products = pd.read_parquet(BRONZE / "products.parquet")
    feats = pricing_features(tx, products)
    FEATS.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATS, index=False)

@task
def train():
    train_pricing(str(FEATS))

@flow(name="pricing_train")
def run():
    build_features()
    train()

if __name__ == "__main__":
    run()
