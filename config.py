"""
config.py
All constants for the insider detection pipeline.
Change WALLET here; everything else propagates automatically.
"""

from pathlib import Path

# ── Wallet under investigation ────────────────────────────────────────────────
# Confirm full address at: https://polymarket.com/@0xafee
# The DOJ complaint also lists it; press sources show the prefix "0xafEe..."
WALLET = "0xee50a31c3f5a7c77824b12a941a54388a2827ed6"  # <-- update if needed

# ── Trading window (AlphaRaccoon) ─────────────────────────────────────────────
# Oct 1 - Dec 10 2025, widened slightly around the known Oct 15-Dec 4 trading
# period. Previous values here were off by a full year (2024 not 2025) which
# caused 00_fetch.py to pull an empty price_histories.json — verify any future
# change with: python -c "import datetime; print(datetime.datetime.fromtimestamp(TS, tz=datetime.timezone.utc))"
WINDOW_START_TS = 1759276800  # 2025-10-01  (Unix UTC)
WINDOW_END_TS   = 1765324800  # 2025-12-10

# ── Polymarket API roots ───────────────────────────────────────────────────────
DATA_API  = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Stage 1 thresholds ────────────────────────────────────────────────────────
S1_WEIGHTS = {"adjusted_roi": 0.40, "wilson_win_rate": 0.35, "age_flag": 0.25}
S1_FLAG_PERCENTILE = 0.95          # flag top 5% by S1 score

WILSON_Z            = 1.96         # 95% confidence
WILSON_LB_THRESHOLD = 0.75         # flag if Wilson lower bound > 75%

AGE_FLAG_MAX_DAYS   = 14           # account younger than this...
AGE_FLAG_MIN_PROFIT = 1_000        # ...AND profit above this...
AGE_FLAG_MAX_MARKETS = 3           # ...AND traded ≤ this many markets

# ── Stage 2 signal thresholds ─────────────────────────────────────────────────
# Signal 1 — Timing
TIMING_PRICE_JUMP_THRESHOLD = 10   # points (0–100 scale)
TIMING_JUMP_STRONG          = 15   # points — used for the hard threshold
TIMING_MAX_LEAD_SECONDS     = 3600 # flag if trade is >1hr before jump

# Signal 2 — Concentration
HHI_CATEGORY_THRESHOLD      = 0.90 # 90% volume in one category → concentrated
SINGLE_EVENT_DOMINANCE      = 0.90 # 90% of total volume in one event

# Signal 3 — Market Impact
IMPACT_WINDOW_SECONDS       = 300  # 5-minute window post-trade
IMPACT_VOLUME_LOOKBACK      = 86400  # 24hr rolling average for comparison

# Signal 4 — Wallet Network
NETWORK_TIME_WINDOW_SECONDS = 120  # 2-minute co-trade window
NETWORK_SIZE_SIMILARITY     = 0.20 # within 20% trade size
NETWORK_JACCARD_THRESHOLD   = 0.60 # market overlap threshold
NETWORK_MIN_CLUSTER_SIZE    = 3    # flag clusters of 3+

# ── Stage 3 model ─────────────────────────────────────────────────────────────
# Bayesian skill threshold: if estimated p(skill) > this, classify as skilled
SKILL_THRESHOLD = 0.80

# ── Stage 4 Claude classifier ─────────────────────────────────────────────────
CLASSIFIER_MODEL      = "claude-haiku-4-5-20251001"
CLASSIFIER_BATCH_SIZE = 25