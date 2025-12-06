from prefect import flow, task
import pandas as pd
from pathlib import Path
from src.common.features import build_clv_feature_table

BRONZE = Path("/app/data/bronze")
GOLD = Path("/app/data/gold")
GOLD_FEATURES = GOLD / "clv_features.parquet"


@task
def load_bronze():
    customers = pd.read_parquet(BRONZE / "customers.parquet")
    products = pd.read_parquet(BRONZE / "products.parquet")
    campaigns = pd.read_parquet(BRONZE / "campaigns.parquet")   # reserved for future
    transactions = pd.read_parquet(BRONZE / "transactions.parquet")
    events = pd.read_parquet(BRONZE / "events.parquet")
    return customers, products, campaigns, transactions, events


@task
def build_features(customers, products, campaigns, transactions, events) -> pd.DataFrame:
    # Build CLV feature table per customer
    return build_clv_feature_table(
        customers=customers,
        transactions=transactions,
        products=products,
        events=events,
    )


@task
def write_features(df: pd.DataFrame):
    GOLD.mkdir(parents=True, exist_ok=True)
    df.to_parquet(GOLD_FEATURES, index=False)


@flow(name="features_build")
def features_build_flow():
    customers, products, campaigns, transactions, events = load_bronze()
    feats = build_features(customers, products, campaigns, transactions, events)
    write_features(feats)


if __name__ == "__main__":
    features_build_flow()
