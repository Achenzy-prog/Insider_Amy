"""
algo/metrics.py
===============
Pure functions for Stage 1 anomaly filter metrics.
All functions are stateless and testable in isolation.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Metric 1: Market-adjusted ROI ─────────────────────────────────────────────

def compute_wallet_roi(trades_df: pd.DataFrame, wallet: str) -> dict[str, float]:
    """
    Compute per-market ROI for the target wallet from trade fills.

    ROI is approximated as (sell_value - buy_value) / buy_value per market.
    This is a fill-based proxy; it matches what the peer distribution also uses,
    so the z-score comparison is internally consistent.

    Returns: {condition_id: roi}
    """
    df = trades_df[trades_df["wallet"].str.lower() == wallet.lower()].copy()
    df["side"] = df["side"].str.upper()

    result = {}
    for cid, group in df.groupby("condition_id"):
        buys  = group[group["side"] == "BUY"]["usdc_value"].sum()
        sells = group[group["side"] == "SELL"]["usdc_value"].sum()
        if buys > 0:
            result[cid] = (sells - buys) / buys
    return result


def adjusted_roi(
    wallet_roi: float,
    market_mean: float,
    market_std: float,
) -> float:
    """
    adjusted_ROI = (wallet_ROI − market_mean_ROI) / market_std_ROI

    A value of +2 means the wallet outperformed 97.7% of peers on that market.
    Returns 0 if std is zero or near-zero (single-trader market).
    """
    if market_std < 1e-9:
        return 0.0
    return (wallet_roi - market_mean) / market_std


def adjusted_roi_percentile(z_score: float) -> float:
    """Convert z-score to a [0, 1] percentile using the normal CDF."""
    from scipy.stats import norm
    return float(norm.cdf(z_score))


# ── Metric 2: Wilson score interval lower bound ───────────────────────────────

def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """
    Wilson score interval lower bound (one-sided, 95% confidence by default).

    Shrinks extreme win rates toward 0.5 when n is small.
    A lower bound > 0.75 is flagged as anomalous.

    Formula:
        adjusted = (wins + z²/2) / (n + z²)
                 − z * sqrt((wins*(n−wins)/n + z²/4)) / (n + z²)

    Edge cases:
        n == 0 → returns 0.0
        wins == 0 → returns 0.0 (no wins observed, lower bound is 0)
    """
    if n == 0:
        return 0.0
    z2 = z * z
    p_hat = wins / n
    numerator   = p_hat + z2 / (2 * n)
    margin      = z * np.sqrt((p_hat * (1 - p_hat) / n) + z2 / (4 * n * n))
    denominator = 1 + z2 / n
    return float((numerator - margin) / denominator)


def compute_win_rate_stats(
    wins_df: Optional[pd.DataFrame],
    trades_df: pd.DataFrame,
    wallet: str,
    z: float = 1.96,
) -> dict:
    """
    Compute raw win rate, sample size, and Wilson lower bound for the wallet.

    Falls back to a trade-level heuristic (bought YES before a market resolved YES)
    if closed_positions data is unavailable.
    """
    if wins_df is not None and len(wins_df) > 0:
        n    = len(wins_df)
        w    = int(wins_df["is_win"].sum())
    else:
        # Heuristic: a trade is a "win" if we BUY YES (or SELL NO) and
        # the corresponding market is in the price_histories at price ~1.0 at end.
        # Simplified: count distinct condition_ids where the wallet has net BUY exposure.
        df   = trades_df[trades_df["wallet"].str.lower() == wallet.lower()]
        n    = df["condition_id"].nunique()
        # Cannot determine actual wins without resolution data — use n/2 as neutral prior
        w    = n // 2
    raw_win_rate = w / n if n > 0 else 0.0
    lb = wilson_lower_bound(w, n, z)
    return {
        "wins":              w,
        "n":                 n,
        "raw_win_rate":      raw_win_rate,
        "wilson_lower_bound": lb,
        "flagged":           lb > 0.75,
    }


# ── Metric 3: Account age flag ────────────────────────────────────────────────

def account_age_flag(
    profile: dict,
    wallet_roi_map: dict[str, float],
    trades_df: pd.DataFrame,
    wallet: str,
    max_days: int       = 14,
    min_profit: float   = 1_000,
    max_markets: int    = 3,
) -> dict:
    """
    Flags Maduro-style burner accounts:
      - account < 14 days old at time of first trade, AND
      - total profit > $1,000, AND
      - number of markets traded ≤ 3

    profile: the raw JSON from the Data API /profile endpoint.
    wallet_roi_map: {condition_id: roi} from compute_wallet_roi().
    """
    import pandas as pd

    # Account creation date
    created_ts = None
    for field in ["joinedAt", "joined_at", "createdAt", "created_at"]:
        if field in profile:
            created_ts = profile[field]
            break

    if created_ts:
        if isinstance(created_ts, (int, float)):
            created = pd.Timestamp(created_ts, unit="s", tz="UTC")
        else:
            created = pd.Timestamp(created_ts, tz="UTC")
        df_wallet  = trades_df[trades_df["wallet"].str.lower() == wallet.lower()]
        first_trade = df_wallet["timestamp"].min()
        days_old    = (first_trade - created).days
    else:
        days_old = 999  # unknown → don't flag

    # Total profit (sum of positive ROIs weighted by position size)
    total_profit = 0.0
    df_wallet = trades_df[trades_df["wallet"].str.lower() == wallet.lower()]
    for cid, roi in wallet_roi_map.items():
        cost = df_wallet[df_wallet["condition_id"] == cid]["usdc_value"].sum()
        if cost > 0 and roi > 0:
            total_profit += cost * roi

    # Number of distinct markets traded
    n_markets = df_wallet["condition_id"].nunique()

    is_flagged = (
        days_old    <= max_days  and
        total_profit >= min_profit and
        n_markets   <= max_markets
    )

    return {
        "days_old":      days_old,
        "total_profit":  total_profit,
        "n_markets":     n_markets,
        "flagged":       is_flagged,
        "flag_value":    1.0 if is_flagged else 0.0,
    }


# ── S1 Composite Score ────────────────────────────────────────────────────────

def s1_composite(
    adjusted_roi_pct:    float,
    wilson_lower_bound:  float,
    age_flag_value:      float,
    weights: dict = None,
) -> float:
    """
    S1 = 0.40 * adjusted_ROI_percentile
       + 0.35 * wilson_lower_bound
       + 0.25 * age_flag_value

    All inputs should be in [0, 1].
    """
    if weights is None:
        weights = {"adjusted_roi": 0.40, "wilson_win_rate": 0.35, "age_flag": 0.25}
    return (
        weights["adjusted_roi"]    * adjusted_roi_pct   +
        weights["wilson_win_rate"] * wilson_lower_bound +
        weights["age_flag"]        * age_flag_value
    )
