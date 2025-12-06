import pandas as pd

LOYALTY_MAP = {"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4}

def compute_rfm_from_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency (days since last purchase), Frequency (# transactions), Monetary (sum of gross_revenue),
    and simple transaction stats per customer from transactions.
    """
    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    # Exclude refunds
    if "refund_flag" in tx.columns:
        tx = tx[tx["refund_flag"] == 0]

    now_ts = tx["timestamp"].max()

    agg = tx.groupby("customer_id").agg(
        recency_days=("timestamp", lambda x: (now_ts - x.max()).days),
        tx_count=("transaction_id", "count"),
        total_revenue=("gross_revenue", "sum"),
        avg_discount=("discount_applied", "mean"),
        avg_quantity=("quantity", "mean"),
    ).reset_index()

    return agg

def enrich_with_products(transactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate share of transactions with premium products per customer.
    """
    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    # Exclude refunds
    if "refund_flag" in tx.columns:
        tx = tx[tx["refund_flag"] == 0]

    prod = products.copy()
    merged = tx.merge(prod[["product_id", "is_premium"]], on="product_id", how="left")
    premium_stats = merged.groupby("customer_id").agg(
        premium_tx_share=("is_premium", lambda x: float(pd.Series(x).fillna(0).mean()))
    ).reset_index()
    return premium_stats

def engagement_features_from_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    Simple engagement features from events: counts per event_type and avg session_duration.
    """
    ev = events.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])

    counts = ev.pivot_table(index="customer_id", columns="event_type", values="event_id", aggfunc="count", fill_value=0)
    counts.columns = [f"events_{c}_count" for c in counts.columns]
    counts = counts.reset_index()

    # Session duration
    if "session_duration_sec" in ev.columns:
        dur = ev.groupby("customer_id")["session_duration_sec"].mean().rename("avg_session_duration_sec").reset_index()
        out = counts.merge(dur, on="customer_id", how="left")
    else:
        out = counts

    return out

def join_customer_demographics(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Map demographics & loyalty to numeric features.
    """
    df = customers.copy()
    df["loyalty_level"] = df["loyalty_tier"].map(LOYALTY_MAP).fillna(0).astype(int)
    df["is_male"] = (df["gender"].str.lower() == "male").astype(int)
    demo = df[["customer_id", "age", "is_male", "loyalty_level"]].copy()
    return demo

def build_clv_feature_table(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final feature table per customer for CLV training.
    """
    rfm = compute_rfm_from_transactions(transactions)
    premium = enrich_with_products(transactions, products)
    engagement = engagement_features_from_events(events)
    demo = join_customer_demographics(customers)

    feats = rfm.merge(premium, on="customer_id", how="left") \
               .merge(engagement, on="customer_id", how="left") \
               .merge(demo, on="customer_id", how="left")

    # Fill missing with 0 for counts and reasonable defaults
    for col in feats.columns:
        if col.startswith("events_") or col in ["avg_session_duration_sec", "premium_tx_share"]:
            feats[col] = feats[col].fillna(0)

    # Rename monetary and set a demo target (we'll use total_revenue as proxy for clv_180d)
    feats = feats.rename(columns={"total_revenue": "monetary"})
    feats["clv_180d"] = feats["monetary"].fillna(0.0)

    return feats
