# src/monitoring/drift_report.py
import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def build_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, out_html: str) -> float:
    """
    Run Evidently's DataDriftPreset on reference/current,
    save interactive HTML to out_html, and return drift_score (share of drifted columns).
    Compatible with modern Evidently 'Snapshot' result object from report.run(...).
    """
    # Ensure both datasets have the same schema
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    ref = reference_df[common_cols].copy()
    cur = current_df[common_cols].copy()

    # 1) Define a report; 2) run to get the Snapshot; 3) save HTML from the Snapshot
    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=cur, reference_data=ref)  # Snapshot object
    snapshot.save_html(out_html)

    # Extract dataset-level drift share from the Snapshot JSON (string) -> dict
    r = json.loads(snapshot.json())  # <-- parse JSON string to dict

    drift_share = 0.0
    for m in r.get("metrics", []):
        res = m.get("result", {})
        ds = res.get("dataset_drift")
        if isinstance(ds, dict) and "share_of_drifted_columns" in ds:
            drift_share = float(ds["share_of_drifted_columns"])
            break

    return drift_share
