"""
diagnose_wins.py
Traces exactly why wins.parquet still comes out as 0/1 (or similar) even
after switching to outcomePrices-based resolution labelling.

Checks each step of the new logic in isolation:
  1. Does target_trades (wallet's own trades) have the expected ~22-23 rows?
  2. Does resolution_map have entries for those condition_ids?
  3. For each trade, what does settled_price actually come out to?

Run from project root: python diagnose_wins.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import RAW_DIR, PROC_DIR, WALLET


def load(name):
    with open(RAW_DIR / f"{name}.json") as f:
        return json.load(f)


print("=== Step 1: target_trades ===")
trades = pd.read_parquet(PROC_DIR / "trades_clean.parquet")
target_trades = trades[trades["wallet"].str.lower() == WALLET.lower()].copy()
print(f"Trades for target wallet: {len(target_trades)}")
print(f"Columns available: {list(target_trades.columns)}")

if len(target_trades) == 0:
    print("⚠ STOP — no trades match this wallet in trades_clean.parquet.")
    print("  Did you re-run 00_fetch.py / 01_preprocess.py with the CURRENT")
    print("  wallet address after changing config.py?")
    sys.exit(0)

print(f"\nSample outcome_index values: {target_trades['outcome_index'].head(10).tolist()}")
print(f"outcome_index dtype: {target_trades['outcome_index'].dtype}")
print(f"Any nulls in outcome_index? {target_trades['outcome_index'].isna().sum()}")

print("\n=== Step 2: resolution_map from markets.json ===")
raw_markets = load("markets")
print(f"Total market records: {len(raw_markets)}")

resolution_map = {}
parse_failures = 0
for m in raw_markets:
    cid = m.get("conditionId")
    raw_prices = m.get("outcomePrices")
    if not cid or not raw_prices:
        continue
    try:
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        prices = [float(p) for p in prices]
    except (json.JSONDecodeError, TypeError, ValueError):
        parse_failures += 1
        continue
    resolution_map[cid] = prices

print(f"Markets with parseable outcomePrices: {len(resolution_map)}")
print(f"Parse failures: {parse_failures}")
print(f"Sample resolution_map entries: {dict(list(resolution_map.items())[:3])}")

print("\n=== Step 3: per-trade resolution check ===")
target_cids = set(target_trades["condition_id"])
print(f"Unique condition_ids in target's trades: {len(target_cids)}")
overlap = target_cids & set(resolution_map.keys())
print(f"Overlap with resolution_map: {len(overlap)}")

if len(overlap) == 0:
    print("⚠ ZERO OVERLAP — condition_ids don't match between trades and markets.json")
    print(f"  Sample trade cid: {list(target_cids)[0] if target_cids else 'N/A'}")
    print(f"  Sample markets.json cid: {list(resolution_map.keys())[0] if resolution_map else 'N/A'}")

print("\nDetailed per-trade breakdown (first 10):")
n_resolved = 0
n_unresolved = 0
n_win = 0
n_loss = 0
for _, t in target_trades.head(30).iterrows():
    cid = t["condition_id"]
    idx = t.get("outcome_index")
    prices = resolution_map.get(cid)

    status = "NO_MARKET_DATA"
    settled = None
    if prices is not None and idx is not None and not pd.isna(idx) and int(idx) < len(prices):
        settled = prices[int(idx)]
        if settled >= 0.95:
            status = "WIN"
            n_resolved += 1
            n_win += 1
        elif settled <= 0.05:
            status = "LOSS"
            n_resolved += 1
            n_loss += 1
        else:
            status = "STILL_OPEN"
            n_unresolved += 1

    print(f"  cid={cid[:20]}...  idx={idx}  settled_price={settled}  → {status}")

print(f"\nSummary across all {len(target_trades)} trades (not just first 10 shown):")
all_resolved, all_win = 0, 0
for _, t in target_trades.iterrows():
    cid = t["condition_id"]
    idx = t.get("outcome_index")
    prices = resolution_map.get(cid)
    if prices is not None and idx is not None and not pd.isna(idx) and int(idx) < len(prices):
        settled = prices[int(idx)]
        if settled >= 0.95:
            all_resolved += 1
            all_win += 1
        elif settled <= 0.05:
            all_resolved += 1

print(f"  Resolved markets found: {all_resolved}")
print(f"  Wins among those:       {all_win}")