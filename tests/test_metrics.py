"""
tests/test_metrics.py
Unit tests for all Stage 1 metric functions.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from algo.metrics import (
    adjusted_roi,
    adjusted_roi_percentile,
    wilson_lower_bound,
    s1_composite,
)
from algo.signals import herfindahl_index


# ── adjusted_roi ──────────────────────────────────────────────────────────────

class TestAdjustedROI:
    def test_average_performer(self):
        """A wallet performing at the market mean should have z=0."""
        z = adjusted_roi(wallet_roi=0.5, market_mean=0.5, market_std=0.1)
        assert z == pytest.approx(0.0)

    def test_outperformer(self):
        """2 std above mean → z=2."""
        z = adjusted_roi(wallet_roi=0.7, market_mean=0.5, market_std=0.1)
        assert z == pytest.approx(2.0)

    def test_underperformer(self):
        z = adjusted_roi(wallet_roi=0.3, market_mean=0.5, market_std=0.1)
        assert z == pytest.approx(-2.0)

    def test_zero_std(self):
        """Single-trader market: std=0 should return 0 without dividing by zero."""
        z = adjusted_roi(wallet_roi=1.0, market_mean=0.5, market_std=0.0)
        assert z == 0.0

    def test_extreme_outlier(self):
        """AlphaRaccoon-like: massive outperformance on a small market."""
        z = adjusted_roi(wallet_roi=5.0, market_mean=0.1, market_std=0.2)
        assert z > 10  # extremely flagged


class TestAdjustedROIPercentile:
    def test_median(self):
        pct = adjusted_roi_percentile(0.0)
        assert pct == pytest.approx(0.5, abs=0.01)

    def test_99th_percentile(self):
        pct = adjusted_roi_percentile(2.33)
        assert pct > 0.98

    def test_bounds(self):
        assert 0 <= adjusted_roi_percentile(-5.0) <= 1
        assert 0 <= adjusted_roi_percentile(5.0) <= 1


# ── wilson_lower_bound ────────────────────────────────────────────────────────

class TestWilsonLowerBound:
    def test_zero_trades(self):
        """n=0 should not crash."""
        lb = wilson_lower_bound(wins=0, n=0)
        assert lb == 0.0

    def test_perfect_record_small_n(self):
        """3/3 wins should be shrunk significantly from 1.0."""
        lb = wilson_lower_bound(wins=3, n=3)
        assert lb < 0.75  # shrunk by small sample

    def test_perfect_record_large_n(self):
        """22/23 wins (AlphaRaccoon) should have lb well above 0.75."""
        lb = wilson_lower_bound(wins=22, n=23)
        assert lb > 0.75

    def test_known_value(self):
        """Reference calculation: 50/100 wins, z=1.96."""
        lb = wilson_lower_bound(wins=50, n=100, z=1.96)
        # Expected: ~0.404 (standard result)
        assert lb == pytest.approx(0.404, abs=0.005)

    def test_symmetry_floor(self):
        """0 wins should give lower bound of 0."""
        lb = wilson_lower_bound(wins=0, n=10)
        assert lb == pytest.approx(0.0, abs=0.01)

    def test_flag_threshold_alpharaccoon(self):
        """AlphaRaccoon's 22/23 should be flagged (lb > 0.75)."""
        lb = wilson_lower_bound(wins=22, n=23)
        assert lb > 0.75


# ── herfindahl_index ──────────────────────────────────────────────────────────

class TestHerfindahlIndex:
    def test_perfect_concentration(self):
        """All volume in one category → HHI = 1.0."""
        hhi = herfindahl_index({"politics": 1000, "sports": 0, "crypto": 0})
        assert hhi == pytest.approx(1.0)

    def test_equal_distribution(self):
        """Equal split across N → HHI = 1/N."""
        hhi = herfindahl_index({"a": 100, "b": 100, "c": 100, "d": 100})
        assert hhi == pytest.approx(0.25, abs=0.01)

    def test_empty(self):
        hhi = herfindahl_index({})
        assert hhi == 0.0

    def test_single_category(self):
        hhi = herfindahl_index({"politics": 500})
        assert hhi == pytest.approx(1.0)

    def test_alpharaccoon_like(self):
        """AlphaRaccoon traded almost exclusively in Google/tech markets."""
        hhi = herfindahl_index({"tech": 2_700_000, "sports": 10_000, "crypto": 5_000})
        assert hhi > 0.95  # extremely concentrated


# ── s1_composite ──────────────────────────────────────────────────────────────

class TestS1Composite:
    def test_all_max(self):
        """All inputs at 1.0 → score = 1.0."""
        s = s1_composite(1.0, 1.0, 1.0)
        assert s == pytest.approx(1.0)

    def test_all_zero(self):
        s = s1_composite(0.0, 0.0, 0.0)
        assert s == pytest.approx(0.0)

    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        from config import S1_WEIGHTS
        assert sum(S1_WEIGHTS.values()) == pytest.approx(1.0, abs=0.001)

    def test_alpharaccoon_scenario(self):
        """
        AlphaRaccoon: 99th percentile ROI, Wilson lb ~0.83 (22/23), no age flag.
        Expect S1 well above 0.95 threshold.
        """
        lb = wilson_lower_bound(22, 23)  # ~0.83
        s = s1_composite(
            adjusted_roi_pct   = 0.99,
            wilson_lower_bound = lb,
            age_flag_value     = 0.0,    # account not new
        )
        assert s > 0.60  # 0.40*0.99 + 0.35*0.83 + 0.25*0.0 ≈ 0.69

    def test_custom_weights(self):
        weights = {"adjusted_roi": 0.5, "wilson_win_rate": 0.5, "age_flag": 0.0}
        s = s1_composite(0.8, 0.6, 1.0, weights=weights)
        assert s == pytest.approx(0.5 * 0.8 + 0.5 * 0.6)
