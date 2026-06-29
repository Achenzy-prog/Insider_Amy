"""
tests/test_signals.py
Unit tests for Stage 2 signal functions.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import timezone
from algo.signals import (
    find_next_price_jump,
    timing_z_score,
    herfindahl_index,
    build_wallet_network,
    find_coordinated_clusters,
)


def ts(unix_sec: int) -> pd.Timestamp:
    return pd.Timestamp(unix_sec, unit="s", tz="UTC")


# ── find_next_price_jump ──────────────────────────────────────────────────────

class TestFindNextPriceJump:
    def _history(self, prices: list[tuple[int, float]]) -> list[dict]:
        return [{"t": t, "p": p} for t, p in prices]

    def test_finds_jump_after_trade(self):
        """A >10pt jump occurring after the trade should be found."""
        history = self._history([
            (1000, 0.30),
            (1060, 0.31),
            (1120, 0.75),  # big jump
            (1180, 0.76),
        ])
        result = find_next_price_jump(history, ts(1050), min_jump_points=10.0)
        assert result is not None
        assert result["jump_size_pts"] > 40
        assert result["delta_t_seconds"] < 0  # trade was before the jump

    def test_ignores_jump_before_trade(self):
        """A jump that happened before the trade should not be returned."""
        history = self._history([
            (1000, 0.30),
            (1060, 0.80),  # jump at t=1060
            (1120, 0.81),
        ])
        result = find_next_price_jump(history, ts(1100), min_jump_points=10.0)
        # The jump at t=1060 is before trade at t=1100 → should not be returned
        assert result is None

    def test_small_jump_not_flagged(self):
        """A 5pt movement should not be found when threshold is 10."""
        history = self._history([
            (1000, 0.50),
            (1060, 0.55),  # only 5pt move
        ])
        result = find_next_price_jump(history, ts(1000), min_jump_points=10.0)
        assert result is None

    def test_empty_history(self):
        assert find_next_price_jump([], ts(1000)) is None

    def test_delta_t_sign(self):
        """delta_t should be negative when trade is before the jump."""
        history = self._history([(1000, 0.3), (2000, 0.8)])
        result  = find_next_price_jump(history, ts(1500), min_jump_points=10.0)
        assert result is not None
        assert result["delta_t_seconds"] < 0  # trade at 1500, jump at 2000


# ── timing_z_score ────────────────────────────────────────────────────────────

class TestTimingZScore:
    def test_at_mean(self):
        deltas = [-3600, -1800, 0, 1800, 3600]
        z = timing_z_score(np.mean(deltas), deltas)
        assert z == pytest.approx(0.0, abs=0.01)

    def test_extreme_early_trade(self):
        """A trade 2hrs before peers should have a strongly negative z-score."""
        deltas = [0, 300, 600, 900, 1200]
        z = timing_z_score(-7200, deltas)
        assert z < -3  # very suspicious

    def test_zero_std(self):
        """All traders at the same time → std=0, return 0."""
        z = timing_z_score(100.0, [100.0, 100.0, 100.0])
        assert z == 0.0


# ── herfindahl_index ──────────────────────────────────────────────────────────

class TestHHI:
    def test_monopoly(self):
        assert herfindahl_index({"a": 1}) == pytest.approx(1.0)

    def test_duopoly_equal(self):
        assert herfindahl_index({"a": 50, "b": 50}) == pytest.approx(0.5)

    def test_zero_values_ignored(self):
        hhi = herfindahl_index({"a": 100, "b": 0, "c": 0})
        assert hhi == pytest.approx(1.0)


# ── build_wallet_network ──────────────────────────────────────────────────────

def make_trades(**kwargs) -> pd.DataFrame:
    """Helper: create a minimal trades DataFrame."""
    return pd.DataFrame(kwargs)


class TestWalletNetwork:
    def _base_trades(self):
        return pd.DataFrame({
            "wallet":       ["0xAAA", "0xBBB", "0xCCC", "0xDDD"],
            "condition_id": ["mkt1",  "mkt1",  "mkt2",  "mkt3"],
            "timestamp":    [ts(1000), ts(1060), ts(2000), ts(3000)],
            "usdc_value":   [1000.0,  1050.0,  500.0,   100.0],
            "side":         ["BUY",   "BUY",   "BUY",   "BUY"],
        })

    def test_co_trade_edge_created(self):
        """Two wallets trading same market within 2 minutes with similar size → edge."""
        trades = self._base_trades()
        G = build_wallet_network(trades, time_window_sec=120, size_similarity=0.20)
        assert G.has_edge("0xAAA", "0xBBB")

    def test_no_edge_when_far_apart(self):
        """Trades more than 2 minutes apart and low Jaccard should not create an edge.
        Two wallets on only one shared market but 5 min apart: Jaccard=1.0 still fires.
        This test disables Jaccard (threshold=1.1) to isolate the time-window criterion."""
        trades = pd.DataFrame({
            "wallet":       ["0xAAA", "0xBBB"],
            "condition_id": ["mkt1",  "mkt1"],
            "timestamp":    [ts(1000), ts(1000 + 300)],  # 5 min apart
            "usdc_value":   [1000.0, 1000.0],
            "side":         ["BUY", "BUY"],
        })
        # Set jaccard_threshold > 1.0 to disable that criterion; only time window applies
        G = build_wallet_network(trades, time_window_sec=120, jaccard_threshold=1.1)
        assert not G.has_edge("0xAAA", "0xBBB")

    def test_jaccard_edge(self):
        """Two wallets sharing >60% of markets should get a Jaccard edge."""
        trades = pd.DataFrame({
            "wallet":       ["0xAAA"] * 5 + ["0xBBB"] * 5,
            "condition_id": ["m1","m2","m3","m4","m5"] * 2,
            "timestamp":    [ts(i * 100000) for i in range(10)],  # far apart in time
            "usdc_value":   [100.0] * 10,
            "side":         ["BUY"] * 10,
        })
        G = build_wallet_network(trades, jaccard_threshold=0.60)
        assert G.has_edge("0xAAA", "0xBBB")

    def test_cluster_detection(self):
        """A 3-wallet cluster should be returned by find_coordinated_clusters."""
        trades = pd.DataFrame({
            "wallet":       ["0xAAA", "0xBBB", "0xCCC"],
            "condition_id": ["mkt1",  "mkt1",  "mkt1"],
            "timestamp":    [ts(1000), ts(1010), ts(1020)],
            "usdc_value":   [1000.0, 950.0, 980.0],
            "side":         ["BUY", "BUY", "BUY"],
        })
        G        = build_wallet_network(trades, time_window_sec=120)
        clusters = find_coordinated_clusters(G, min_cluster_size=3)
        assert len(clusters) >= 1
        members = clusters[0]
        assert "0xAAA" in members
        assert "0xBBB" in members
        assert "0xCCC" in members
