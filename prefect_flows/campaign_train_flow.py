from prefect import flow, task
import pandas as pd
from pathlib import Path
from src.common.features import campaign_features
from src.campaign_response.train import train_campaign_classifier

BRONZE = Path("/app/data/bronze")
GOLD = Path("/app/data/gold")
FEATS = GOLD / "campaign_features.parquet"

@task
def build_features():
    customers = pd.read_parquet(BRONZE / "customers.parquet")
    campaigns = pd.read_parquet(BRONZE / "campaigns.parquet")
    events = pd.read_parquet(BRONZE / "events.parquet")
    feats = campaign_features(customers, campaigns, events)
    FEATS.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATS, index=False)

@task
def train():
    train_campaign_classifier(str(FEATS))

@flow(name="campaign_response_train")
def run():
    build_features()
    train()

if __name__ == "__main__":
    run()
