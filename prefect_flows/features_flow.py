from prefect import flow, task
from src.common.io import (
    read_raw_customers, read_raw_products, read_raw_campaigns, read_raw_transactions, read_raw_events,
    write_bronze
)

@task
def ingest_customers():
    df = read_raw_customers()
    write_bronze(df, "customers")

@task
def ingest_products():
    df = read_raw_products()
    write_bronze(df, "products")

@task
def ingest_campaigns():
    df = read_raw_campaigns()
    write_bronze(df, "campaigns")

@task
def ingest_transactions():
    df = read_raw_transactions()
    write_bronze(df, "transactions")

@task
def ingest_events():
    df = read_raw_events()
    write_bronze(df, "events")

@flow(name="ingest_all")
def ingest_all_flow():
    ingest_customers()
    ingest_products()
    ingest_campaigns()
    ingest_transactions()
    ingest_events()

if __name__ == "__main__":
    ingest_all_flow()
