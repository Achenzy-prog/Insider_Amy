"""
pipeline/01_preprocess.py
=========================
Cleans and normalises raw JSON from 00_fetch.py into analysis-ready DataFrames.

Outputs (Parquet, written to data/processed/):
  trades_clean.parquet    — one row per trade, typed and enriched
  markets_clean.parquet   — one row per market, with category and liquidity info
  peer_stats.parquet      — per-market: mean_roi, std_roi, n_traders
  wins.parquet            — win/loss per market, derived from market_resolution.json
                            (CLOB token.winner flags — see 00_fetch.py Step 3)
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from config import RAW_DIR, PROC_DIR, WALLET


def load(name):
    path = RAW_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


# ── 1. Trades ─────────────────────────────────────────────────────────────────

print("[1/4] Cleaning trades...")
raw_trades = load("trades")

trades = pd.DataFrame(raw_trades)

# Normalise column names (API may return camelCase or snake_case)
trades.columns = [c.lower() for c in trades.columns]
rename = {
    "proxywallet":   "wallet",
    "conditionid":   "condition_id",
    "transactionhash": "tx_hash",
    "outcomeindex":  "outcome_index",
    "eventslug":     "event_slug",
}
trades.rename(columns={k: v for k, v in rename.items() if k in trades.columns}, inplace=True)

# Types
trades["timestamp"] = pd.to_datetime(trades["timestamp"], unit="s", utc=True)
trades["price"]     = trades["price"].astype(float)
trades["size"]      = trades["size"].astype(float)

# Dollar value of each trade (size × price, since size is in shares and price is in USDC per share)
trades["usdc_value"] = trades["size"] * trades["price"]

# Direction: BUY YES or SELL YES are the meaningful insider bets
trades["is_target_wallet"] = trades["wallet"].str.lower() == WALLET.lower()

# Sort chronologically
trades.sort_values("timestamp", inplace=True)
trades.reset_index(drop=True, inplace=True)

trades.to_parquet(PROC_DIR / "trades_clean.parquet", index=False)
print(f"  ✓ trades_clean.parquet  ({len(trades)} rows)")


# ── 2. Markets ────────────────────────────────────────────────────────────────

print("[2/4] Cleaning markets...")
raw_markets = load("markets")

markets = pd.DataFrame(raw_markets)
markets.columns = [c.lower() for c in markets.columns]
rename = {
    "conditionid":  "condition_id",
    "question":     "title",
    "volume":       "volume_usdc",
    "liquidity":    "liquidity_usdc",
    "marketslug":   "slug",
    "enddate":      "end_date",
    "startdate":    "start_date",
}
markets.rename(columns={k: v for k, v in rename.items() if k in markets.columns}, inplace=True)

# Category and event_slug both live INSIDE the nested 'events' list, not at
# the top level. Gamma's /markets response shape:
#   events: [{ "id": ..., "slug": "...", "ticker": "...", "tags": [...] }]
def extract_category(row):
    events = row.get("events") or []
    if isinstance(events, list) and events:
        ev = events[0]
        tags = ev.get("tags") or []
        if isinstance(tags, list) and tags:
            first = tags[0]
            if isinstance(first, dict):
                return first.get("label") or first.get("slug") or "Unknown"
            return str(first)
    return "Unknown"

def extract_event_slug(row):
    events = row.get("events") or []
    if isinstance(events, list) and events:
        return events[0].get("slug", "Unknown")
    return "Unknown"

markets["category"]   = markets.apply(extract_category, axis=1)
markets["event_slug"] = markets.apply(extract_event_slug, axis=1)

# Number of unique traders per market isn't in the Gamma response directly;
# we'll compute it downstream from market_all_trades.json instead (peer_stats.parquet).
markets["num_traders"] = 0  # placeholder, overwritten by peer_stats join if needed

# Numeric coercions
for col in ["volume_usdc", "liquidity_usdc"]:
    if col in markets.columns:
        markets[col] = pd.to_numeric(markets[col], errors="coerce").fillna(0)

markets.to_parquet(PROC_DIR / "markets_clean.parquet", index=False)
print(f"  ✓ markets_clean.parquet  ({len(markets)} rows)")


# ── 3. Peer statistics per market ─────────────────────────────────────────────
# For each market: compute the distribution of ROI across all traders.
# This is what Stage 1 needs for market-adjusted ROI.

print("[3/4] Computing peer ROI distributions...")
raw_all = load("market_all_trades")

peer_rows = []
for cid, mkt_trades in raw_all.items():
    if not mkt_trades:
        continue
    df = pd.DataFrame(mkt_trades)
    df.columns = [c.lower() for c in df.columns]

    # Normalise wallet column
    wallet_col = next((c for c in ["proxywallet", "wallet", "maker_address"] if c in df.columns), None)
    if wallet_col is None:
        continue
    df.rename(columns={wallet_col: "wallet"}, inplace=True)

    for col in ["price", "size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["usdc_value"] = df.get("size", 0) * df.get("price", 0)
    df["side"]       = df.get("side", "BUY").str.upper()

    # Per-wallet: rough ROI proxy = (sell_value - buy_value) / buy_value
    # We can only do this from fills; it's an approximation.
    def wallet_roi(wdf):
        buys  = wdf[wdf["side"] == "BUY"]["usdc_value"].sum()
        sells = wdf[wdf["side"] == "SELL"]["usdc_value"].sum()
        if buys == 0:
            return np.nan
        return (sells - buys) / buys

    roi_series = df.groupby("wallet").apply(wallet_roi).dropna()

    peer_rows.append({
        "condition_id":    cid,
        "market_mean_roi": roi_series.mean(),
        "market_std_roi":  roi_series.std() if len(roi_series) > 1 else 1.0,
        "n_traders":       roi_series.count(),
    })

peer_stats = pd.DataFrame(peer_rows)
peer_stats.to_parquet(PROC_DIR / "peer_stats.parquet", index=False)
print(f"  ✓ peer_stats.parquet  ({len(peer_stats)} markets)")


# ── 4. Win/loss labels derived from market resolution ─────────────────────────
# Two earlier approaches were tried and both failed:
#   1. /positions endpoint (closed_positions.json) — returned only 1 record
#      for AlphaRaccoon despite 22+ known trades. Unreliable/incomplete.
#   2. outcomePrices from Gamma's /markets?condition_id=X — this query
#      parameter turned out to be silently ignored by the API, so every
#      market record returned was a duplicate of the SAME market, and that
#      market's outcomePrices was just a live/unresolved snapshot anyway.
#
# Current approach: 00_fetch.py now queries the CLOB API's /markets/{cid}
# path endpoint (which reliably filters by condition_id) and saves each
# token's `winner: true/false` flag to market_resolution.json. This is the
# actual on-chain settlement flag, not an inferred price threshold — the
# correct ground truth.

print("[4/4] Labelling wins and losses from market resolution...")

target_trades = trades[trades["wallet"].str.lower() == WALLET.lower()].copy()

if len(target_trades) == 0:
    print("  ⚠ No trades found for target wallet — check WALLET in config.py")
    wins = pd.DataFrame()
else:
    resolution_path = RAW_DIR / "market_resolution.json"
    if not resolution_path.exists():
        print("  ⚠ market_resolution.json not found — re-run 00_fetch.py with the")
        print("    updated CLOB-based market fetch to generate it.")
        wins = pd.DataFrame()
    else:
        with open(resolution_path) as f:
            market_resolution = json.load(f)  # {cid: {token_id: winner_bool}}

        # Each trade has an 'asset' field — this IS the token_id the wallet
        # traded (confirmed from trades.json: 'asset': '42970893...'). Match
        # that token_id against market_resolution[cid] to get the winner flag.
        rows = []
        for _, t in target_trades.iterrows():
            cid = t["condition_id"]
            token_id = str(t.get("asset", ""))
            token_winners = market_resolution.get(cid)
            if token_winners is None or token_id not in token_winners:
                continue  # market not yet resolved, or token_id format mismatch

            is_win = bool(token_winners[token_id])
            rows.append({
                "condition_id": cid,
                "token_id":     token_id,
                "is_win":       is_win,
            })

        wins = pd.DataFrame(rows)

        # One row per market (a wallet may have several trades in the same
        # market — collapse to avoid overweighting markets with many fills).
        if len(wins) > 0:
            wins = wins.drop_duplicates(subset="condition_id")

        if len(wins) > 0:
            wins.to_parquet(PROC_DIR / "wins.parquet", index=False)
            n_w = int(wins["is_win"].sum())
            print(f"  ✓ wins.parquet  ({len(wins)} resolved markets, {n_w} wins)")
        else:
            print("  ⚠ No resolved markets found — markets may still be open,")
            print("    or token_id format in trades doesn't match market_resolution keys.")
            print("    Run diagnose_wins.py for a detailed breakdown.")

print("\n=== Preprocess complete ===")