"""
experiment1.py
==============
Experiment 1: Known Insider Cases — Ground Truth Validation

For each wallet in experiment_wallets.json, fetches trade data and computes:
  - Market-adjusted ROI
  - Corrected win rate (Wilson score)
  - Timing score
  - Concentration score (HHI)
  - Market impact score
  - S1 composite score

Then prints a comparison table: Insiders vs Skilled vs Average traders.

Run from project root: python experiment1.py
"""

import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from config import DATA_API, GAMMA_API, CLOB_API, WINDOW_START_TS, WINDOW_END_TS
from algo.metrics import (
    compute_wallet_roi,
    adjusted_roi,
    adjusted_roi_percentile,
    compute_win_rate_stats,
    s1_composite,
)
from algo.signals import (
    compute_concentration_signal,
    compute_timing_signal,
    compute_market_impact_signal,
)

MAX_WORKERS = 8
OUT_DIR = Path("data/experiment1")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get(url, params=None, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                time.sleep(2 ** (attempt + 2))
            else:
                break
        except Exception:
            time.sleep(2 ** attempt)
    return None


def paginate(url, params, page_size=500):
    results, offset = [], 0
    while True:
        batch = get(url, {**params, "limit": page_size, "offset": offset})
        if not batch:
            break
        results.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return results


# ── Load wallets ──────────────────────────────────────────────────────────────

with open("experiment_wallets.json") as f:
    wallets = json.load(f)

print(f"Loaded {len(wallets)} wallets: {list(wallets.keys())}\n")


# ── Fetch data for each wallet ────────────────────────────────────────────────

all_rows = []

for name, info in wallets.items():
    wallet_addr = info["id"].replace(" ", "")  # strip any accidental spaces
    category    = info["category"]
    event       = info["event"]

    print(f"[{category}] {name} ({wallet_addr[:10]}...)")

    # 1. Fetch all trades for this wallet
    trades_raw = paginate(f"{DATA_API}/trades", {
        "user": wallet_addr,
        "takerOnly": "false",
    })
    if not trades_raw:
        print(f"  ⚠ No trades found — skipping")
        continue

    trades = pd.DataFrame(trades_raw)
    trades.columns = [c.lower() for c in trades.columns]
    rename = {
        "proxywallet": "wallet",
        "conditionid": "condition_id",
        "eventslug":   "event_slug",
        "outcomeindex": "outcome_index",
        "transactionhash": "tx_hash",
    }
    trades.rename(columns={k: v for k, v in rename.items() if k in trades.columns}, inplace=True)
    trades["timestamp"]  = pd.to_datetime(trades["timestamp"], unit="s", utc=True)
    trades["price"]      = pd.to_numeric(trades["price"], errors="coerce")
    trades["size"]       = pd.to_numeric(trades["size"], errors="coerce")
    trades["usdc_value"] = trades["size"] * trades["price"]
    trades["wallet"]     = wallet_addr

    market_ids = list({t for t in trades["condition_id"] if t})
    print(f"  {len(trades_raw)} trades across {len(market_ids)} markets")

    # 2. Fetch market metadata (CLOB path-based — correct endpoint)
    def fetch_market(cid):
        data = get(f"{CLOB_API}/markets/{cid}")
        if data and (data.get("condition_id") == cid or data.get("conditionId") == cid):
            return (cid, data)
        return None

    market_map = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_market, cid): cid for cid in market_ids}
        for future in as_completed(futures):
            result = future.result()
            if result:
                market_map[result[0]] = result[1]

    markets_list = list(market_map.values())

    # Build minimal markets_df for concentration signal
    markets_df = pd.DataFrame([{
        "condition_id": m.get("condition_id") or m.get("conditionId"),
        "category":     event,  # use the event column from experiment_wallets.json directly
                                # since the CLOB API's tags field is unreliable
        "event_slug":   m.get("market_slug") or m.get("slug") or "",
    } for m in markets_list])

    # 3. Fetch peer trades (for market-adjusted ROI denominator)
    def fetch_peer_trades(cid):
        data = paginate(f"{DATA_API}/trades", {"market": cid, "takerOnly": "false"})
        return (cid, data) if data else None

    peer_map = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_peer_trades, cid): cid for cid in market_ids}
        for future in as_completed(futures):
            result = future.result()
            if result:
                peer_map[result[0]] = result[1]

    # Build peer stats
    peer_rows = []
    for cid, mkt_trades in peer_map.items():
        df_p = pd.DataFrame(mkt_trades)
        df_p.columns = [c.lower() for c in df_p.columns]
        wc = next((c for c in ["proxywallet", "wallet"] if c in df_p.columns), None)
        if not wc:
            continue
        df_p.rename(columns={wc: "wallet"}, inplace=True)
        for col in ["price", "size"]:
            if col in df_p.columns:
                df_p[col] = pd.to_numeric(df_p[col], errors="coerce")
        df_p["usdc_value"] = df_p.get("size", 0) * df_p.get("price", 0)
        df_p["side"] = df_p.get("side", "BUY").str.upper()

        def wroi(wdf):
            buys  = wdf[wdf["side"] == "BUY"]["usdc_value"].sum()
            sells = wdf[wdf["side"] == "SELL"]["usdc_value"].sum()
            return (sells - buys) / buys if buys > 0 else np.nan

        roi_s = df_p.groupby("wallet").apply(wroi).dropna()
        peer_rows.append({
            "condition_id":    cid,
            "market_mean_roi": roi_s.mean(),
            "market_std_roi":  roi_s.std() if len(roi_s) > 1 else 1.0,
            "n_traders":       roi_s.count(),
        })

    peers = pd.DataFrame(peer_rows)

    # 4. Fetch price histories for timing signal
    market_tokens = {}
    for m in markets_list:
        cid = m.get("condition_id") or m.get("conditionId")
        tokens = m.get("tokens")
        if tokens and cid:
            yes_token = next((t for t in tokens if t.get("outcome","").lower()=="yes"), tokens[0])
            market_tokens[cid] = yes_token.get("token_id")

    def fetch_ph(item):
        cid, token_id = item
        data = get(f"{CLOB_API}/prices-history", params={
            "market": token_id,
            "fidelity": 1,
            "startTs": WINDOW_START_TS,
            "endTs":   WINDOW_END_TS,
        })
        if data and data.get("history"):
            return (cid, data["history"])
        return None

    price_histories = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_ph, item): item for item in market_tokens.items()}
        for future in as_completed(futures):
            result = future.result()
            if result:
                price_histories[result[0]] = result[1]

    # 5. Win/loss from token.winner
    market_resolution = {}
    for m in markets_list:
        cid = m.get("condition_id") or m.get("conditionId")
        tokens = m.get("tokens")
        if tokens and cid:
            market_resolution[cid] = {t.get("token_id"): t.get("winner", False) for t in tokens}

    # Compute wins
    wins_rows = []
    for _, t in trades.iterrows():
        cid      = t["condition_id"]
        token_id = str(t.get("asset", ""))
        token_winners = market_resolution.get(cid)
        if token_winners and token_id in token_winners:
            wins_rows.append({"condition_id": cid, "is_win": bool(token_winners[token_id])})
    wins_df = pd.DataFrame(wins_rows).drop_duplicates(subset="condition_id") if wins_rows else None

    # ── Compute all metrics ───────────────────────────────────────────────────

    # Market-adjusted ROI
    wallet_rois = compute_wallet_roi(trades, wallet_addr)
    adj_roi_scores = []
    for cid, roi in wallet_rois.items():
        peer_row = peers[peers["condition_id"] == cid] if len(peers) > 0 else pd.DataFrame()
        if peer_row.empty:
            continue
        z   = adjusted_roi(roi, peer_row["market_mean_roi"].iloc[0], peer_row["market_std_roi"].iloc[0])
        pct = adjusted_roi_percentile(z)
        adj_roi_scores.append(pct)
    mean_adj_roi_pct = np.mean(adj_roi_scores) if adj_roi_scores else 0.5

    # Wilson win rate
    win_stats = compute_win_rate_stats(wins_df, trades, wallet_addr)
    wilson_lb = win_stats["wilson_lower_bound"]
    win_rate  = win_stats["raw_win_rate"]
    n_wins    = win_stats["wins"]
    n_total   = win_stats["n"]

    # Concentration (HHI)
    if len(markets_df) > 0:
        conc = compute_concentration_signal(trades, markets_df, wallet_addr)
        hhi               = conc["category_hhi"]
        single_event_share = conc["single_event_share"]
    else:
        hhi, single_event_share = 0.0, 0.0

    # Timing signal
    if price_histories:
        timing_df = compute_timing_signal(trades, price_histories, wallet_addr)
        n_timing_flagged = int(timing_df["timing_flagged"].sum()) if len(timing_df) > 0 else 0
        mean_timing_z    = float(timing_df["timing_z"].mean()) if len(timing_df) > 0 else 0.0
    else:
        n_timing_flagged, mean_timing_z = 0, 0.0

    # Market impact
    if price_histories:
        impact_df = compute_market_impact_signal(trades, price_histories, wallet_addr)
        n_impact_flagged = int(impact_df["impact_flagged"].sum()) if len(impact_df) > 0 else 0
    else:
        n_impact_flagged = 0

    # S1 composite
    s1 = s1_composite(mean_adj_roi_pct, wilson_lb, 0.0)

    row = {
        "name":               name,
        "category":           category,
        "event":              event,
        "n_trades":           len(trades),
        "n_markets":          len(market_ids),
        "win_rate":           f"{n_wins}/{n_total} ({win_rate:.1%})",
        "wilson_lb":          round(wilson_lb, 4),
        "adj_roi_pct":        round(mean_adj_roi_pct, 4),
        "hhi":                round(hhi, 4),
        "single_event_share": f"{single_event_share:.1%}",
        "timing_flagged":     n_timing_flagged,
        "mean_timing_z":      round(mean_timing_z, 3),
        "impact_flagged":     n_impact_flagged,
        "s1_composite":       round(s1, 4),
    }
    all_rows.append(row)
    print(f"  S1={s1:.4f}  HHI={hhi:.4f}  Wilson={wilson_lb:.4f}  wins={n_wins}/{n_total}")
    print()


# ── Results table ─────────────────────────────────────────────────────────────

results = pd.DataFrame(all_rows)
results = results.sort_values(["category", "s1_composite"], ascending=[True, False])

print("\n" + "=" * 80)
print("EXPERIMENT 1 RESULTS — Insider vs Skilled vs Average")
print("=" * 80)

for cat in ["Insider", "Skilled", "Average"]:
    subset = results[results["category"] == cat]
    if len(subset) == 0:
        continue
    print(f"\n── {cat} traders ──")
    print(subset[[
        "name", "event", "win_rate", "wilson_lb",
        "hhi", "timing_flagged", "s1_composite"
    ]].to_string(index=False))

print("\n── Group averages ──")
numeric_cols = ["wilson_lb", "adj_roi_pct", "hhi", "timing_flagged", "impact_flagged", "s1_composite"]
print(results.groupby("category")[numeric_cols].mean().round(4).to_string())

# Save
results.to_csv(OUT_DIR / "results.csv", index=False)
print(f"\n✓ Full results saved to {OUT_DIR}/results.csv")