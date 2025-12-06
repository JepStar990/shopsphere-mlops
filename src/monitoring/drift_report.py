from evidently.report import Report
from evidently.metrics import DataDriftPreset
import pandas as pd

def build_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, out_html: str) -> float:
    """
    Builds an Evidently DataDrift report, saves HTML, and returns a drift score in [0,1],
    defined as share of drifted columns according to Evidently.
    """
    # Ensure same columns intersection to avoid schema issues
    common_cols = [c for c in reference_df.columns if c in current_df.columns]
    ref = reference_df[common_cols].copy()
    cur = current_df[common_cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(out_html)

    # Extract share of drifted columns from Evidently result dict
    r = report.as_dict()
    drift_share = 0.0
    # Traverse to find dataset-level drift info (structure may vary with Evidently version)
    for m in r.get("metrics", []):
        res = m.get("result", {}) if isinstance(m, dict) else {}
        ds = res.get("dataset_drift")
        if isinstance(ds, dict) and "share_of_drifted_columns" in ds:
            drift_share = float(ds["share_of_drifted_columns"])
            break

    return drift_share
