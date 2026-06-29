"""
pipeline/03_stage2_signals.py
==============================
Stage 2 Signal Modules — runs all four signals on the target wallet.

Reads:
  data/processed/trades_clean.parquet
  data/processed/markets_clean.parquet
  data/raw/price_histories.json

Writes:
  data/processed/signal1_timing.parquet
  data/processed/signal2_concentration.json
  data/processed/signal3_impact.parquet
  data/processed/signal4_network.json
  data/processed/signals_summary.json
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import (
    WALLET, RAW_DIR, PROC_DIR,
    TIMING_PRICE_JUMP_THRESHOLD, TIMING_JUMP_STRONG, TIMING_MAX_LEAD_SECONDS,
    HHI_CATEGORY_THRESHOLD, SINGLE_EVENT_DOMINANCE,
    IMPACT_WINDOW_SECONDS,
    NETWORK_TIME_WINDOW_SECONDS, NETWORK_SIZE_SIMILARITY, NETWORK_JACCARD_THRESHOLD,
    NETWORK_MIN_CLUSTER_SIZE,
)
from algo.signals import (
    compute_timing_signal,
    compute_concentration_signal,
    compute_market_impact_signal,
    build_wallet_network,
    find_coordinated_clusters,
)


# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading data...")
trades  = pd.read_parquet(PROC_DIR / "trades_clean.parquet")
markets = pd.read_parquet(PROC_DIR / "markets_clean.parquet")

with open(RAW_DIR / "price_histories.json") as f:
    price_histories = json.load(f)

print(f"  {len(trades)} trades, {len(markets)} markets, {len(price_histories)} price histories")


# ── Signal 1: Timing ──────────────────────────────────────────────────────────

print("\n── Signal 1: Timing ──")
timing_df = compute_timing_signal(
    trades_df        = trades,
    price_histories  = price_histories,
    wallet           = WALLET,
    min_jump_pts     = TIMING_PRICE_JUMP_THRESHOLD,
    strong_jump_pts  = TIMING_JUMP_STRONG,
    max_lead_sec     = TIMING_MAX_LEAD_SECONDS,
)

if len(timing_df) > 0:
    n_flagged = timing_df["timing_flagged"].sum()
    print(f"  Trades analysed:    {len(timing_df)}")
    print(f"  Flagged (>1hr lead, >15pt jump): {n_flagged}")
    print(f"  Mean Δt (seconds):  {timing_df['delta_t_seconds'].mean():.0f}")
    print(f"  Mean timing z-score:{timing_df['timing_z'].mean():.2f}")
    print("\n  Per-trade breakdown:")
    print(timing_df[["trade_ts", "delta_t_seconds", "jump_size_pts", "timing_z", "timing_flagged"]]
          .to_string(index=False))
    timing_df.to_parquet(PROC_DIR / "signal1_timing.parquet", index=False)
else:
    print("  No timing results — price history may be sparse or trades outside window")


# ── Signal 2: Concentration ───────────────────────────────────────────────────

print("\n── Signal 2: Concentration ──")
conc = compute_concentration_signal(
    trades_df        = trades,
    markets_df       = markets,
    wallet           = WALLET,
    hhi_threshold    = HHI_CATEGORY_THRESHOLD,
    single_dominance = SINGLE_EVENT_DOMINANCE,
)

print(f"  Total volume:        ${conc['total_volume_usdc']:,.0f}")
print(f"  Category HHI:        {conc['category_hhi']:.4f}  (flag if ≥ {HHI_CATEGORY_THRESHOLD})")
print(f"  Category breakdown:  {conc['category_breakdown']}")
print(f"  HHI flagged:         {conc['hhi_flagged']}")
print(f"  Single-event share:  {conc['single_event_share']:.1%}  (flag if ≥ {SINGLE_EVENT_DOMINANCE:.0%})")
print(f"  Dominant market:     {conc['dominant_condition_id']}")
print(f"  Dominance flagged:   {conc['dominance_flagged']}")

with open(PROC_DIR / "signal2_concentration.json", "w") as f:
    json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in conc.items()}, f, indent=2, default=str)


# ── Signal 3: Market Impact ───────────────────────────────────────────────────

print("\n── Signal 3: Market Impact ──")
impact_df = compute_market_impact_signal(
    trades_df        = trades,
    price_histories  = price_histories,
    wallet           = WALLET,
    impact_window_sec= IMPACT_WINDOW_SECONDS,
)

if len(impact_df) > 0:
    n_flagged = impact_df["impact_flagged"].sum()
    print(f"  Trades analysed:      {len(impact_df)}")
    print(f"  Flagged (large trade + big move): {n_flagged}")
    print(f"  Mean 5-min price move:{impact_df['price_move_5min_pts'].mean():.1f} pts")
    print(f"  Mean size/daily ratio:{impact_df['size_vs_daily_avg'].mean():.2f}x")
    impact_df.to_parquet(PROC_DIR / "signal3_impact.parquet", index=False)
else:
    print("  No impact results — price history may be sparse")


# ── Signal 4: Wallet Network ──────────────────────────────────────────────────

print("\n── Signal 4: Wallet Network ──")
print("  Building co-trading graph... (may take a minute on large datasets)")

G = build_wallet_network(
    trades_df         = trades,
    time_window_sec   = NETWORK_TIME_WINDOW_SECONDS,
    size_similarity   = NETWORK_SIZE_SIMILARITY,
    jaccard_threshold = NETWORK_JACCARD_THRESHOLD,
)

clusters = find_coordinated_clusters(G, min_cluster_size=NETWORK_MIN_CLUSTER_SIZE)
target_in_cluster = [c for c in clusters if WALLET.lower() in {w.lower() for w in c}]

print(f"  Wallets in graph:       {G.number_of_nodes()}")
print(f"  Edges (co-trade links): {G.number_of_edges()}")
print(f"  Clusters (≥3 wallets):  {len(clusters)}")
print(f"  Target wallet in cluster: {bool(target_in_cluster)}")

if target_in_cluster:
    cluster = target_in_cluster[0]
    print(f"  Cluster size: {len(cluster)} wallets")
    print(f"  Members: {list(cluster)[:10]}{'...' if len(cluster) > 10 else ''}")

network_summary = {
    "n_nodes":              G.number_of_nodes(),
    "n_edges":              G.number_of_edges(),
    "n_clusters":           len(clusters),
    "target_in_cluster":    bool(target_in_cluster),
    "target_cluster_size":  len(target_in_cluster[0]) if target_in_cluster else 0,
    "target_cluster_wallets": list(target_in_cluster[0]) if target_in_cluster else [],
}
with open(PROC_DIR / "signal4_network.json", "w") as f:
    json.dump(network_summary, f, indent=2)


# ── Summary ───────────────────────────────────────────────────────────────────

summary = {
    "wallet": WALLET,
    "signal1_timing": {
        "n_trades_with_timing": len(timing_df) if len(timing_df) > 0 else 0,
        "n_flagged": int(timing_df["timing_flagged"].sum()) if len(timing_df) > 0 else 0,
        "mean_timing_z": float(timing_df["timing_z"].mean()) if len(timing_df) > 0 else None,
    },
    "signal2_concentration": {
        "category_hhi":      conc["category_hhi"],
        "hhi_flagged":       conc["hhi_flagged"],
        "single_event_share": conc["single_event_share"],
        "dominance_flagged": conc["dominance_flagged"],
    },
    "signal3_impact": {
        "n_trades": len(impact_df) if len(impact_df) > 0 else 0,
        "n_flagged": int(impact_df["impact_flagged"].sum()) if len(impact_df) > 0 else 0,
    },
    "signal4_network": network_summary,
}

with open(PROC_DIR / "signals_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n✓ Saved all signal outputs to {PROC_DIR}")
