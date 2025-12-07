from prometheus_client import start_http_server, Gauge
from pathlib import Path
import time

SCORE_FILE = Path("/app/data/monitoring/clv_drift_score.txt")
AUC_FILE = Path("/app/data/monitoring/campaign_auc.txt")
AUC_DELTA_FILE = Path("/app/data/monitoring/campaign_auc_delta.txt")
PRICING_VIOLATIONS_FILE = Path("/app/data/monitoring/pricing_guardrail_violations.txt")
REC_COVERAGE_FILE = Path("/app/data/monitoring/recommender_coverage.txt")
REC_NOVELTY_FILE = Path("/app/data/monitoring/recommender_novelty.txt")

def read_float(p: Path) -> float:
    try: return float(p.read_text().strip())
    except: return 0.0

if __name__ == "__main__":
    clv_g = Gauge("clv_drift_score", "Share of drifted columns from Evidently (0..1)")
    auc_g = Gauge("propensity_auc", "Current AUC for campaign propensity model")
    auc_delta_g = Gauge("propensity_auc_delta", "Current AUC minus reference AUC")
    pricing_viol_g = Gauge("pricing_guardrail_violations", "Pricing guardrail violations in last run")
    rec_cov_g = Gauge("recommender_coverage", "Recommender coverage across catalog (0..1)")
    rec_nov_g = Gauge("recommender_novelty", "Recommender novelty (avg inverse popularity)")

    start_http_server(9100)
    while True:
        clv_g.set(read_float(SCORE_FILE))
        auc_g.set(read_float(AUC_FILE))
        auc_delta_g.set(read_float(AUC_DELTA_FILE))
        pricing_viol_g.set(read_float(PRICING_VIOLATIONS_FILE))
        rec_cov_g.set(read_float(REC_COVERAGE_FILE))
        rec_nov_g.set(read_float(REC_NOVELTY_FILE))
        time.sleep(15)
