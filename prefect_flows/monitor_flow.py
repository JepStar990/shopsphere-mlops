from prefect import flow, task
from pathlib import Path
import pandas as pd
from src.monitoring.drift_report import build_drift_report

GOLD = Path("/app/data/gold")
REF_DIR = Path("/app/data/reference")
REPORT_DIR = Path("/app/data/reports")
MONITOR_DIR = Path("/app/data/monitoring")

CUR_FEATURES = GOLD / "clv_features.parquet"
REF_FEATURES = REF_DIR / "clv_features_ref.parquet"
REPORT_HTML = REPORT_DIR / "clv_drift.html"
DRIFT_SCORE_FILE = MONITOR_DIR / "clv_drift_score.txt"

@task
def ensure_dirs():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)

@task
def load_current() -> pd.DataFrame:
    return pd.read_parquet(CUR_FEATURES)

@task
def load_or_init_reference(cur: pd.DataFrame) -> pd.DataFrame:
    if REF_FEATURES.exists():
        return pd.read_parquet(REF_FEATURES)
    # First run: take a stable reference snapshot (e.g., first 60% rows)
    ref = cur.sample(frac=0.6, random_state=42) if len(cur) > 10 else cur.copy()
    ref.to_parquet(REF_FEATURES, index=False)
    return ref

@task
def run_evidently(ref: pd.DataFrame, cur: pd.DataFrame) -> float:
    drift_score = build_drift_report(
        reference_df=ref,
        current_df=cur,
        out_html=str(REPORT_HTML)
    )
    return float(drift_score)

@task
def write_score(score: float):
    DRIFT_SCORE_FILE.write_text(f"{score:.6f}")

@flow(name="clv_monitor_drift")
def monitor_flow():
    ensure_dirs()
    cur = load_current()
    ref = load_or_init_reference(cur)
    score = run_evidently(ref, cur)
    write_score(score)
    print(f"[monitor] clv_drift_score={score:.4f} | report={REPORT_HTML}")

if __name__ == "__main__":
    monitor_flow()
