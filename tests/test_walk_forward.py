"""Tests that the walk-forward training window has no look-ahead (Point 4)."""
from __future__ import annotations

import pandas as pd

from quant.signals import _train_window


def _make_df(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    return pd.DataFrame({"ds": dates, "y": range(len(dates))})


def test_train_window_uses_only_past_data():
    df = _make_df("2020-01-01", "2026-05-22")
    as_of = pd.Timestamp("2024-06-15")
    win = _train_window(df, as_of, years=3)
    # No date in the window can be > as_of.
    assert win["ds"].max() <= as_of
    # Window must start within 3 years before as_of (allow 1-day rounding).
    assert win["ds"].min() >= as_of - pd.DateOffset(years=3) - pd.Timedelta(days=1)


def test_train_window_is_three_years_wide_on_long_history():
    df = _make_df("2010-01-01", "2026-01-01")
    as_of = pd.Timestamp("2025-01-01")
    win = _train_window(df, as_of, years=3)
    span_days = (win["ds"].max() - win["ds"].min()).days
    # 3 calendar years, allow trading-day shrinkage.
    assert 365 * 2.8 <= span_days <= 365 * 3 + 5


def test_train_window_truncates_short_history():
    # Only 1 year of data available; window cannot fabricate extra past.
    df = _make_df("2024-06-01", "2025-06-01")
    as_of = pd.Timestamp("2025-06-01")
    win = _train_window(df, as_of, years=3)
    assert win["ds"].min() == df["ds"].min()
    assert win["ds"].max() == df["ds"].max()
