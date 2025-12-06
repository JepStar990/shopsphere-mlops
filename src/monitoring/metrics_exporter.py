from prometheus_client import start_http_server, Gauge
import time
from pathlib import Path

SCORE_FILE = Path("/app/data/monitoring/clv_drift_score.txt")

def read_score() -> float:
    try:
        return float(SCORE_FILE.read_text().strip())
    except Exception:
        return 0.0

if __name__ == "__main__":
    gauge = Gauge("clv_drift_score", "Share of drifted columns from Evidently (0..1)")
    start_http_server(9100)
    while True:
        gauge.set(read_score())
        time.sleep(15)
