import numpy as np
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

# --- Segmentation features from transactions + customers ---
def segmentation_features(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    if "refund_flag" in tx.columns:
        tx = tx[tx["refund_flag"] == 0]

    now_ts = tx["timestamp"].max()
    rfm = tx.groupby("customer_id").agg(
        recency_days=("timestamp", lambda x: (now_ts - x.max()).days),
        tx_count=("transaction_id", "count"),
        monetary=("gross_revenue", "sum"),
        avg_discount=("discount_applied", "mean"),
        avg_qty=("quantity", "mean")
    ).reset_index()

    demo = customers.copy()
    demo["loyalty_level"] = demo["loyalty_tier"].map({"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4}).fillna(0).astype(int)
    demo["is_male"] = (demo["gender"].str.lower() == "male").astype(int)
    demo = demo[["customer_id", "age", "is_male", "loyalty_level"]]

    feats = rfm.merge(demo, on="customer_id", how="left")
    # Normalize selected numeric features for KMeans
    for col in ["recency_days", "tx_count", "monetary", "avg_discount", "avg_qty", "age", "loyalty_level"]:
        feats[col] = feats[col].fillna(0.0)
    return feats


# --- Campaign response features (join customers + campaigns + events history) ---
def campaign_features(customers: pd.DataFrame, campaigns: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    # basic engagement counts per customer
    counts = ev.pivot_table(index="customer_id", columns="event_type",
                            values="event_id", aggfunc="count", fill_value=0)
    counts.columns = [f"events_{c}_count" for c in counts.columns]
    counts = counts.reset_index()

    demo = customers.copy()
    demo["loyalty_level"] = demo["loyalty_tier"].map({"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4}).fillna(0).astype(int)
    demo["is_male"] = (demo["gender"].str.lower() == "male").astype(int)
    demo = demo[["customer_id", "age", "is_male", "loyalty_level", "acquisition_channel", "country"]]

    # For a simple baseline, assign each customer to the latest running campaign channel (proxy)
    last_campaign = campaigns.copy()
    last_campaign["start_date"] = pd.to_datetime(last_campaign["start_date"])
    last_campaign["end_date"] = pd.to_datetime(last_campaign["end_date"])
    # Use expected_uplift as a proxy feature; in reality we’d join response labels from a table
    camp_feat = last_campaign.groupby("channel").agg(
        uplift_mean=("expected_uplift", "mean")
    ).reset_index()

    # Merge everything into customer space
    feats = demo.merge(counts, on="customer_id", how="left")
    feats["channel_pref"] = feats["acquisition_channel"].fillna("Unknown")
    feats = feats.merge(camp_feat, left_on="channel_pref", right_on="channel", how="left").drop(columns=["channel"])
    for c in feats.columns:
        if c.startswith("events_"):
            feats[c] = feats[c].fillna(0)
    feats["uplift_mean"] = feats["uplift_mean"].fillna(0.0)
    return feats


# --- Recommender features: user-item interaction matrix (co-occurrence baseline) ---
def build_user_item_matrix(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    if "refund_flag" in tx.columns:
        tx = tx[tx["refund_flag"] == 0]
    # implicit feedback: quantity as strength
    ui = tx.groupby(["customer_id", "product_id"]).agg(
        strength=("quantity", "sum")
    ).reset_index()
    return ui


# --- Pricing features: compute elasticity proxies at product level ---
def pricing_features(transactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    if "refund_flag" in tx.columns:
        tx = tx[tx["refund_flag"] == 0]

    prod = products.copy()
    df = tx.merge(prod[["product_id", "category", "base_price", "is_premium"]], on="product_id", how="left")
    # simple product aggregates
    agg = df.groupby("product_id").agg(
        avg_price=("base_price", "mean"),
        units=("quantity", "sum"),
        revenue=("gross_revenue", "sum"),
        avg_discount=("discount_applied", "mean"),
        premium_share=("is_premium", "mean")
    ).reset_index()
    # target proxy: revenue sensitivity to discount (very rough)
    agg["price_sensitivity"] = agg["avg_discount"].fillna(0.0) * agg["units"].fillna(0.0)
    return agg
