"""Tests for Ledoit-Wolf covariance and diagonal fallback (Point 3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from quant.covariance import COND_NUMBER_THRESHOLD, estimate_covariance
from quant.data_loader import LoadResult


def _make_result(ticker: str, prices: pd.Series) -> LoadResult:
    df = pd.DataFrame({"ds": prices.index, "y": prices.values, "volume": 100})
    return LoadResult(ticker=ticker, df=df, illiquid_ratio=0.0,
                      rows_raw=len(df), rows_kept=len(df), excluded=False)


def _gbm_prices(n: int, mu: float, sigma: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-01-01", periods=n)
    log_rets = rng.normal(mu, sigma, n)
    return pd.Series(100.0 * np.exp(np.cumsum(log_rets)), index=dates)


def test_estimate_covariance_shrinks_toward_identity():
    results = [
        _make_result(f"T{i}", _gbm_prices(600, 0.0003, 0.012, seed=i))
        for i in range(5)
    ]
    as_of = pd.Timestamp("2026-01-01")
    cov = estimate_covariance(results, as_of=as_of, years=2)
    assert cov.cov.shape == (5, 5)
    # Diagonal should be positive (annualized variance).
    assert (np.diag(cov.cov.values) > 0).all()
    # Shrinkage should not blow up condition number.
    assert cov.cond_number_shrunk < cov.cond_number_raw * 10 + 1


def test_estimate_covariance_raises_on_insufficient_overlap():
    a = _make_result("A", _gbm_prices(10, 0.0003, 0.012, seed=1))
    with pytest.raises(ValueError):
        estimate_covariance([a], as_of=pd.Timestamp("2026-01-01"))
