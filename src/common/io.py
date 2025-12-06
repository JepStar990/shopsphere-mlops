from pathlib import Path
import pandas as pd

RAW_DIR = Path("/app/data/raw")
BRONZE_DIR = Path("/app/data/bronze")

def read_raw_customers() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "customers.csv", parse_dates=["signup_date"])
    return df

def read_raw_products() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "products.csv", parse_dates=["launch_date"])
    return df

def read_raw_campaigns() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "campaigns.csv", parse_dates=["start_date", "end_date"])
    return df

def read_raw_transactions() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "transactions.csv", parse_dates=["timestamp"])
    # sanitize/rename if needed
    return df

def read_raw_events() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "events.csv", parse_dates=["timestamp"])
    return df

def write_bronze(df: pd.DataFrame, name: str):
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BRONZE_DIR / f"{name}.parquet", index=False)
