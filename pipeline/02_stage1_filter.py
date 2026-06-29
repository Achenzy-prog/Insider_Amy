"""
pipeline/02_stage1_filter.py
============================
Stage 1 Anomaly Filter — reduces wallet universe to suspicious subset.

Reads:
  data/processed/trades_clean.parquet
  data/processed/markets_clean.parquet
  data/processed/peer_stats.parquet
  data/processed/wins.parquet        (optional)
  data/raw/profile.json

Writes:
  data/processed/s1_scores.parquet   — per-market z-scores + final S1 for target wallet

Prints a human-readable summary of all three metrics and the composite S1 score.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import WALLET, RAW_DIR, PROC_DIR, S1_WEIGHTS, WILSON_Z
from algo.metrics import (
    compute_wallet_roi,
    adjusted_roi,
    adjusted_roi_percentile,
    compute_win_rate_stats,
    account_age_flag,
    s1_composite,
)


# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading preprocessed data...")
trades   = pd.read_parquet(PROC_DIR / "trades_clean.parquet")
markets  = pd.read_parquet(PROC_DIR / "markets_clean.parquet")
peers    = pd.read_parquet(PROC_DIR / "peer_stats.parquet")

wins_path = PROC_DIR / "wins.parquet"
wins = pd.read_parquet(wins_path) if wins_path.exists() else None

with open(RAW_DIR / "profile.json") as f:
    profile = json.load(f)


# ── Metric 1: Market-adjusted ROI ─────────────────────────────────────────────

print("\n── Metric 1: Market-adjusted ROI ──")
wallet_rois = compute_wallet_roi(trades, WALLET)

rows = []
for cid, roi in wallet_rois.items():
    peer_row = peers[peers["condition_id"] == cid]
    if peer_row.empty:
        continue
    mean = peer_row["market_mean_roi"].iloc[0]
    std  = peer_row["market_std_roi"].iloc[0]
    n    = peer_row["n_traders"].iloc[0]

    z    = adjusted_roi(roi, mean, std)
    pct  = adjusted_roi_percentile(z)

    # Pull market title for readability
    mkt_title = "Unknown"
    mrow = markets[markets["condition_id"] == cid]
    if not mrow.empty:
        mkt_title = mrow.iloc[0].get("title", "Unknown")

    rows.append({
        "condition_id":     cid,
        "title":            mkt_title,
        "wallet_roi":       roi,
        "market_mean_roi":  mean,
        "market_std_roi":   std,
        "n_peer_traders":   n,
        "adjusted_roi_z":   z,
        "adjusted_roi_pct": pct,
    })

m1_df = pd.DataFrame(rows).sort_values("adjusted_roi_pct", ascending=False)
print(m1_df[["title", "wallet_roi", "adjusted_roi_z", "adjusted_roi_pct", "n_peer_traders"]].to_string(index=False))


# ── Metric 2: Wilson score win rate ───────────────────────────────────────────

print("\n── Metric 2: Wilson Score Win Rate ──")
win_stats = compute_win_rate_stats(wins, trades, WALLET, z=WILSON_Z)
print(f"  Wins:              {win_stats['wins']}")
print(f"  Total positions:   {win_stats['n']}")
print(f"  Raw win rate:      {win_stats['raw_win_rate']:.1%}")
print(f"  Wilson lower bound:{win_stats['wilson_lower_bound']:.3f}")
print(f"  Flagged (> 0.75):  {win_stats['flagged']}")


# ── Metric 3: Account age flag ────────────────────────────────────────────────

print("\n── Metric 3: Account Age Flag ──")
age = account_age_flag(profile, wallet_rois, trades, WALLET)
print(f"  Account age at first trade: {age['days_old']} days")
print(f"  Total estimated profit:     ${age['total_profit']:,.0f}")
print(f"  Markets traded:             {age['n_markets']}")
print(f"  Flagged (burner pattern):   {age['flagged']}")


# ── S1 Composite ──────────────────────────────────────────────────────────────

print("\n── S1 Composite Score ──")

# Use mean adjusted_roi_pct across markets as the wallet-level metric
mean_roi_pct = m1_df["adjusted_roi_pct"].mean() if len(m1_df) > 0 else 0.5
wb           = win_stats["wilson_lower_bound"]
af           = age["flag_value"]

s1 = s1_composite(mean_roi_pct, wb, af, weights=S1_WEIGHTS)

print(f"  adjusted_ROI percentile (mean): {mean_roi_pct:.3f}  × {S1_WEIGHTS['adjusted_roi']}")
print(f"  Wilson lower bound:             {wb:.3f}  × {S1_WEIGHTS['wilson_win_rate']}")
print(f"  Age flag:                       {af:.0f}      × {S1_WEIGHTS['age_flag']}")
print(f"  ─────────────────────────────────────")
print(f"  S1 composite score:             {s1:.4f}")
print(f"  Flag threshold (top 25%):        0.75")
print(f"  → {'⚠ FLAGGED' if s1 >=0.75 else 'not flagged at this threshold'}")


# ── Save ──────────────────────────────────────────────────────────────────────

# Attach wallet-level summary to the per-market table
m1_df["wallet"] = WALLET
m1_df["wilson_lower_bound"] = wb
m1_df["age_flag"]           = af
m1_df["s1_composite"]       = s1

m1_df.to_parquet(PROC_DIR / "s1_scores.parquet", index=False)
print(f"\n✓ Saved s1_scores.parquet ({len(m1_df)} rows)")
