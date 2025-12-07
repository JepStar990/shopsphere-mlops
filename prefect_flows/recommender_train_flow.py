from prefect import flow, task
import pandas as pd
from pathlib import Path
from src.common.features import build_user_item_matrix
from src.recommender.train import train_cooccurrence

BRONZE = Path("/app/data/bronze")
GOLD = Path("/app/data/gold")
UI = GOLD / "user_item.parquet"

@task
def build_ui():
    tx = pd.read_parquet(BRONZE / "transactions.parquet")
    ui = build_user_item_matrix(tx)
    UI.parent.mkdir(parents=True, exist_ok=True)
    ui.to_parquet(UI, index=False)

@task
def train():
    train_cooccurrence(str(UI))

@flow(name="recommender_train")
def run():
    build_ui()
    train()

if __name__ == "__main__":
    run()
