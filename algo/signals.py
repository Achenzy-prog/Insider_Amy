"""
algo/signals.py
===============
Stage 2 signal modules: timing, concentration, market impact, wallet network.
All functions are pure and independently testable.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional


# ── Signal 1: Timing ──────────────────────────────────────────────────────────

def find_next_price_jump(
    price_history: list[dict],
    trade_ts: pd.Timestamp,
    min_jump_points: float = 10.0,
) -> Optional[dict]:
    """
    Given a market's price history (list of {t, p} dicts) and a trade timestamp,
    find the next price jump ≥ min_jump_points that occurs *after* trade_ts.

    Returns: {jump_ts, jump_size, delta_t_seconds} or None if no jump found.

    delta_t_seconds < 0 → trade was before the jump (suspicious)
    delta_t_seconds > 0 → trade was after the jump (unremarkable)
    """
    if not price_history:
        return None

    # Build a sorted series of (timestamp, price) tuples
    candles = sorted(price_history, key=lambda x: x.get("t", 0))
    prices  = [(pd.Timestamp(c["t"], unit="s", tz="UTC"), float(c["p"])) for c in candles]

    # Find jumps: consecutive candles where |Δprice| ≥ threshold
    for i in range(1, len(prices)):
        ts_prev, p_prev = prices[i - 1]
        ts_curr, p_curr = prices[i]
        jump = abs(p_curr - p_prev) * 100  # convert 0–1 to 0–100 scale

        if jump >= min_jump_points and ts_curr > trade_ts:
            delta_t = (trade_ts - ts_curr).total_seconds()  # negative = trade before jump
            return {
                "jump_ts":          ts_curr,
                "jump_size_pts":    jump,
                "delta_t_seconds":  delta_t,
            }
    return None


def timing_z_score(
    delta_t: float,
    all_delta_ts: list[float],
) -> float:
    """
    Normalise delta_t relative to all other traders on the same market.
    More negative z-score = traded earlier relative to peers = more suspicious.
    """
    arr = np.array(all_delta_ts)
    mu  = arr.mean()
    std = arr.std()
    if std < 1e-9:
        return 0.0
    return float((delta_t - mu) / std)


def compute_timing_signal(
    trades_df: pd.DataFrame,
    price_histories: dict,
    wallet: str,
    min_jump_pts:   float = 10.0,
    strong_jump_pts: float = 15.0,
    max_lead_sec:   float = 3600.0,
) -> pd.DataFrame:
    """
    For each (wallet, market) trade, compute:
      - delta_t to next price jump
      - timing z-score relative to all traders on that market
      - boolean flag: traded > 1hr before a >15pt jump

    Returns a DataFrame with one row per target-wallet trade.
    """
    target_trades = trades_df[trades_df["wallet"].str.lower() == wallet.lower()].copy()
    rows = []

    for cid, group in target_trades.groupby("condition_id"):
        ph = price_histories.get(cid, [])
        if not ph:
            continue

        # All traders' timing on this market (for z-score denominator)
        all_on_market = trades_df[trades_df["condition_id"] == cid]
        all_deltas = []
        for _, row in all_on_market.iterrows():
            result = find_next_price_jump(ph, row["timestamp"], min_jump_pts)
            if result:
                all_deltas.append(result["delta_t_seconds"])

        # Target wallet's trades on this market
        for _, row in group.iterrows():
            result = find_next_price_jump(ph, row["timestamp"], min_jump_pts)
            if result is None:
                continue
            dt = result["delta_t_seconds"]
            z  = timing_z_score(dt, all_deltas) if all_deltas else 0.0
            rows.append({
                "condition_id":        cid,
                "trade_ts":            row["timestamp"],
                "delta_t_seconds":     dt,
                "jump_size_pts":       result["jump_size_pts"],
                "timing_z":            z,
                "timing_flagged":      dt < -max_lead_sec and result["jump_size_pts"] >= strong_jump_pts,
            })

    return pd.DataFrame(rows)


# ── Signal 2: Concentration Score ─────────────────────────────────────────────

def herfindahl_index(values: dict) -> float:
    """
    Herfindahl–Hirschman Index (HHI) over a distribution of values.
    HHI = sum of (share_i)^2 where share_i = value_i / total.
    Range: [1/n, 1.0]. Value of 1.0 = total concentration in one category.
    """
    total = sum(values.values())
    if total == 0:
        return 0.0
    shares = [v / total for v in values.values()]
    return float(sum(s ** 2 for s in shares))


def compute_concentration_signal(
    trades_df: pd.DataFrame,
    markets_df: pd.DataFrame,
    wallet: str,
    hhi_threshold:   float = 0.90,
    single_dominance: float = 0.90,
) -> dict:
    """
    Signal 2: Two sub-metrics.
      1. Category HHI: trades concentrated in one market category?
      2. Single-event dominance: one event's volume > X% of total?

    Returns a dict with both sub-metrics and flags.
    """
    df = trades_df[trades_df["wallet"].str.lower() == wallet.lower()].copy()
    # trades_df already carries its own 'event_slug' (from the Data API trades
    # endpoint, which returns it directly per-trade). Only merge 'category'
    # from markets_clean — merging 'event_slug' too would create a column
    # collision and the markets-level version is less reliable (derived from
    # the first nested event only).
    df = df.merge(markets_df[["condition_id", "category"]], on="condition_id", how="left")

    total_vol = df["usdc_value"].sum()

    # Sub-metric 1: Category HHI
    cat_vol = df.groupby("category")["usdc_value"].sum().to_dict()
    cat_hhi = herfindahl_index(cat_vol)

    # Sub-metric 2: Single-event dominance
    event_vol = df.groupby("condition_id")["usdc_value"].sum()
    if len(event_vol) > 0 and total_vol > 0:
        max_event_share = float(event_vol.max() / total_vol)
        dominant_event  = event_vol.idxmax()
    else:
        max_event_share = 0.0
        dominant_event  = None

    return {
        "total_volume_usdc":   total_vol,
        "category_hhi":        cat_hhi,
        "category_breakdown":  cat_vol,
        "single_event_share":  max_event_share,
        "dominant_condition_id": dominant_event,
        "hhi_flagged":         cat_hhi >= hhi_threshold,
        "dominance_flagged":   max_event_share >= single_dominance,
    }


# ── Signal 3: Market Impact ────────────────────────────────────────────────────

def compute_market_impact_signal(
    trades_df: pd.DataFrame,
    price_histories: dict,
    wallet: str,
    impact_window_sec: int = 300,
) -> pd.DataFrame:
    """
    For each large trade by the target wallet:
      - Measure the price move in the next 5 minutes
      - Compare trade size to rolling 24hr average volume on that market

    A large trade preceding a big price move in an illiquid market is a strong signal.
    """
    target = trades_df[trades_df["wallet"].str.lower() == wallet.lower()].copy()
    all_vols = trades_df.groupby("condition_id")["usdc_value"].sum()

    rows = []
    for _, trade in target.iterrows():
        cid = trade["condition_id"]
        ph  = price_histories.get(cid, [])
        if not ph:
            continue

        # Sort price candles
        candles = sorted(ph, key=lambda x: x.get("t", 0))
        prices  = [(pd.Timestamp(c["t"], unit="s", tz="UTC"), float(c["p"])) for c in candles]

        # Find price just before the trade
        trade_ts = trade["timestamp"]
        pre_prices  = [(ts, p) for ts, p in prices if ts <= trade_ts]
        post_prices = [(ts, p) for ts, p in prices if ts > trade_ts and
                       (ts - trade_ts).total_seconds() <= impact_window_sec]

        if not pre_prices or not post_prices:
            continue

        price_before = pre_prices[-1][1]
        price_after  = post_prices[-1][1]
        price_move   = abs(price_after - price_before) * 100  # in points

        # 24hr average volume on this market (rough proxy: total / trading days)
        mkt_total_vol = all_vols.get(cid, 0)
        avg_daily_vol = mkt_total_vol / 60  # rough normaliser

        trade_size   = trade["usdc_value"]
        size_ratio   = trade_size / avg_daily_vol if avg_daily_vol > 0 else 0.0

        rows.append({
            "condition_id":        cid,
            "trade_ts":            trade_ts,
            "trade_usdc":          trade_size,
            "price_move_5min_pts": price_move,
            "size_vs_daily_avg":   size_ratio,
            "impact_flagged":      price_move >= 5.0 and size_ratio >= 0.5,
        })

    return pd.DataFrame(rows)


# ── Signal 4: Wallet Network Graph ────────────────────────────────────────────

def build_wallet_network(
    trades_df: pd.DataFrame,
    time_window_sec:  int   = 120,
    size_similarity:  float = 0.20,
    jaccard_threshold: float = 0.60,
) -> nx.Graph:
    """
    Build a wallet co-trading graph.

    Edges are added between two wallets when ANY of:
      1. They trade the same market within `time_window_sec` with similar size
      2. Their traded-market Jaccard similarity > `jaccard_threshold`

    (On-chain funding source links require Polygon RPC data — see README.)

    Returns a networkx Graph; nodes are wallet addresses.
    """
    G = nx.Graph()

    # Add all wallets as nodes
    wallets = trades_df["wallet"].unique()
    for w in wallets:
        G.add_node(w)

    # Pre-compute each wallet's set of markets
    wallet_markets = (
        trades_df.groupby("wallet")["condition_id"]
        .apply(set)
        .to_dict()
    )

    # Criterion 1: same market, same time window, similar size
    for cid, group in trades_df.groupby("condition_id"):
        group = group.sort_values("timestamp")
        records = group[["wallet", "timestamp", "usdc_value"]].to_dict("records")

        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                r1, r2 = records[i], records[j]
                dt = abs((r2["timestamp"] - r1["timestamp"]).total_seconds())
                if dt > time_window_sec:
                    break  # sorted, so all later pairs will be further apart

                if r1["wallet"] == r2["wallet"]:
                    continue

                # Size similarity: within 20%
                s1, s2 = r1["usdc_value"], r2["usdc_value"]
                if s1 > 0 and s2 > 0:
                    ratio = min(s1, s2) / max(s1, s2)
                    if ratio >= (1 - size_similarity):
                        w1, w2 = r1["wallet"], r2["wallet"]
                        if G.has_edge(w1, w2):
                            G[w1][w2]["co_trade_count"] = G[w1][w2].get("co_trade_count", 0) + 1
                        else:
                            G.add_edge(w1, w2, co_trade_count=1, jaccard=0.0)

    # Criterion 2: Jaccard similarity on traded markets
    wallet_list = list(wallet_markets.keys())
    for i in range(len(wallet_list)):
        for j in range(i + 1, len(wallet_list)):
            w1, w2 = wallet_list[i], wallet_list[j]
            s1, s2 = wallet_markets[w1], wallet_markets[w2]
            if not s1 or not s2:
                continue
            jaccard = len(s1 & s2) / len(s1 | s2)
            if jaccard >= jaccard_threshold:
                if G.has_edge(w1, w2):
                    G[w1][w2]["jaccard"] = jaccard
                else:
                    G.add_edge(w1, w2, co_trade_count=0, jaccard=jaccard)

    return G


def find_coordinated_clusters(
    G: nx.Graph,
    min_cluster_size: int = 3,
) -> list[set]:
    """
    Return connected components of size ≥ min_cluster_size in the co-trading graph.
    These are candidate coordinated rings.
    """
    return [
        c for c in nx.connected_components(G)
        if len(c) >= min_cluster_size
    ]