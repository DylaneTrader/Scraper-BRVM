"""Tests for the liquidity filter (Point 1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant.data_loader import (
    DEFAULT_FLAT_STREAK,
    MAX_ILLIQUID_RATIO,
    _flat_streak_mask,
    load_ticker,
)


def _write_csv(tmp_path: Path, ticker: str, rows: list[tuple]) -> Path:
    path = tmp_path / f"{ticker}_historique.csv"
    header = "Date;Ouverture;Plus_Haut;Plus_Bas;Cloture;Volume_Titres;Volume_FCFA;Variation_Pct"
    lines = [header]
    for date_str, close, volume in rows:
        close_fr = str(close).replace(".", ",")
        lines.append(f"{date_str};{close_fr};{close_fr};{close_fr};{close_fr};{volume};0;")
    path.write_text("﻿" + "\n".join(lines), encoding="utf-8")
    return path


def test_flat_streak_mask_marks_runs_longer_than_threshold():
    closes = pd.Series([100, 100, 100, 100, 100, 105, 105, 110])  # 5-in-a-row of 100
    mask = _flat_streak_mask(closes, threshold=DEFAULT_FLAT_STREAK)
    # First 5 are all part of a flat run of length 5 (> 3) -> all True.
    assert mask.iloc[:5].all()
    # The pair 105,105 has length 2, not > 3 -> False.
    assert not mask.iloc[5:7].any()
    assert not mask.iloc[7]


def test_flat_streak_mask_does_not_flag_short_runs():
    closes = pd.Series([100, 100, 100, 100, 102, 103])  # exactly 4 same closes
    mask = _flat_streak_mask(closes, threshold=DEFAULT_FLAT_STREAK)
    # Run of length 4 > 3 -> flagged
    assert mask.iloc[:4].all()
    # Threshold=4 -> run of 4 is NOT > 4
    mask2 = _flat_streak_mask(closes, threshold=4)
    assert not mask2.iloc[:4].any()


def test_load_ticker_filters_zero_volume(tmp_path):
    rows = [
        ("2024-01-02", 1000, 100),
        ("2024-01-03", 1010, 0),    # zero volume -> filtered
        ("2024-01-04", 1020, 50),
        ("2024-01-05", 1030, 0),    # zero volume -> filtered
        ("2024-01-08", 1040, 75),
    ]
    path = _write_csv(tmp_path, "TEST_xx", rows)
    res = load_ticker(path)
    assert res.ticker == "TEST_xx"
    assert res.rows_raw == 5
    assert res.rows_kept == 3
    assert res.illiquid_ratio == pytest.approx(2 / 5)


def test_load_ticker_excludes_when_ratio_above_threshold(tmp_path):
    # 6/10 illiquid -> ratio 0.6 > 0.40 -> excluded
    rows = []
    for i in range(10):
        vol = 0 if i < 6 else 100
        rows.append((f"2024-01-{i+2:02d}", 1000 + i, vol))
    path = _write_csv(tmp_path, "ILQ_xx", rows)
    res = load_ticker(path)
    assert res.illiquid_ratio > MAX_ILLIQUID_RATIO
    assert res.excluded is True


def test_load_ticker_keeps_liquid_ticker(tmp_path):
    rows = [(f"2024-01-{i+2:02d}", 1000 + i * 3, 100 + i) for i in range(10)]
    path = _write_csv(tmp_path, "LIQ_xx", rows)
    res = load_ticker(path)
    assert res.excluded is False
    assert res.illiquid_ratio == 0.0
    assert res.rows_kept == 10


def test_load_ticker_filters_flat_streaks(tmp_path):
    # 5 identical closes (flat streak length 5 > 3) followed by 3 movers.
    rows = [(f"2024-01-{i+2:02d}", 1000, 100) for i in range(5)]  # flat
    rows += [(f"2024-01-{i+7:02d}", 1000 + i, 100) for i in range(1, 4)]
    path = _write_csv(tmp_path, "FLT_xx", rows)
    res = load_ticker(path)
    # 5 flat rows out of 8 -> ratio 0.625 -> excluded
    assert res.illiquid_ratio == pytest.approx(5 / 8)
    assert res.excluded is True
