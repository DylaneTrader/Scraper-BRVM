"""Robust covariance estimation for Black-Litterman (Point 3).

Uses Ledoit-Wolf shrinkage on the last 2 years of daily returns. Falls back to
a diagonal (variance-only) matrix when the condition number stays > 1000 after
shrinkage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .data_loader import LoadResult

log = logging.getLogger(__name__)

COV_LOOKBACK_YEARS = 2
COND_NUMBER_THRESHOLD = 1000.0


@dataclass
class CovarianceResult:
    cov: pd.DataFrame  # annualized covariance
    tickers: list[str]
    cond_number_raw: float
    cond_number_shrunk: float
    used_fallback: bool  # True if diagonal fallback was triggered


def _build_returns_panel(
    results: Iterable[LoadResult],
    as_of: pd.Timestamp,
    years: int = COV_LOOKBACK_YEARS,
) -> pd.DataFrame:
    cutoff_low = as_of - pd.DateOffset(years=years)
    series = {}
    for r in results:
        if r.excluded or r.df.empty:
            continue
        df = r.df[(r.df["ds"] >= cutoff_low) & (r.df["ds"] <= as_of)]
        if len(df) < 30:
            continue
        s = df.set_index("ds")["y"].astype(float)
        series[r.ticker] = s
    if not series:
        return pd.DataFrame()
    prices = pd.DataFrame(series).sort_index()
    # Inner-join on common dates to keep returns aligned; gaps imply illiquidity.
    prices = prices.dropna(how="any")
    returns = prices.pct_change().dropna(how="any")
    return returns


def _cond_number(matrix: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return float("inf")


def estimate_covariance(
    results: Iterable[LoadResult],
    as_of: pd.Timestamp,
    years: int = COV_LOOKBACK_YEARS,
) -> CovarianceResult:
    returns = _build_returns_panel(results, as_of, years)
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Insufficient overlapping data to estimate covariance.")

    tickers = list(returns.columns)
    raw_cov = returns.cov().values
    cond_raw = _cond_number(raw_cov)

    lw = LedoitWolf().fit(returns.values)
    shrunk = lw.covariance_
    cond_shrunk = _cond_number(shrunk)

    used_fallback = False
    if cond_shrunk > COND_NUMBER_THRESHOLD:
        log.warning(
            "Cov condition number %.1f > %.0f after shrinkage; "
            "falling back to diagonal variance matrix.",
            cond_shrunk, COND_NUMBER_THRESHOLD,
        )
        shrunk = np.diag(np.diag(shrunk))
        used_fallback = True
        cond_shrunk = _cond_number(shrunk)

    # Scale daily covariance to the signal horizon (30 trading days, matching
    # Prophet's forecast window). Keeps BL views and covariance on the same
    # time scale -> avoids fake Sharpe ratios from annualizing a 30d view.
    HORIZON_TRADING_DAYS = 30
    horizon_cov = shrunk * HORIZON_TRADING_DAYS
    cov_df = pd.DataFrame(horizon_cov, index=tickers, columns=tickers)

    log.info(
        "Cov estimated: n=%d tickers, cond raw=%.1f -> shrunk=%.1f fallback=%s",
        len(tickers), cond_raw, cond_shrunk, used_fallback,
    )
    return CovarianceResult(
        cov=cov_df,
        tickers=tickers,
        cond_number_raw=cond_raw,
        cond_number_shrunk=cond_shrunk,
        used_fallback=used_fallback,
    )
