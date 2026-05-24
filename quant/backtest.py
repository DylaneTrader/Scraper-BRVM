"""Walk-forward backtest (Point 4).

At each monthly rebalancing date `t`:
  1. Train Prophet on data strictly < `t`.
  2. Build BL views from the forecasts.
  3. Re-estimate covariance on returns < `t`.
  4. Rebalance to the resulting BL weights, held for one month.

Also runs a "naive" backtest that uses the full-sample model at every date
(look-ahead biased) so the two can be compared head-to-head.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .covariance import estimate_covariance
from .data_loader import LoadResult
from .portfolio import optimize_portfolio
from .signals import TickerSignal, forecast_ticker

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    backtest_type: str
    dates: list[str]
    algo: list[float]
    bh: list[float]
    bench: list[float]
    metrics: dict


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="MS"))


def _price_panel(results: Iterable[LoadResult]) -> pd.DataFrame:
    series = {}
    for r in results:
        if r.excluded or r.df.empty:
            continue
        series[r.ticker] = r.df.set_index("ds")["y"].astype(float)
    return pd.DataFrame(series).sort_index().ffill()


def _metrics(curve: pd.Series) -> dict:
    if len(curve) < 2:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "cagr": 0.0}
    rets = curve.pct_change().dropna()
    days = (curve.index[-1] - curve.index[0]).days
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (365.0 / max(days, 1)) - 1.0
    vol = rets.std() * np.sqrt(12)  # monthly rebalanced curve
    sharpe = (rets.mean() * 12) / vol if vol > 0 else 0.0
    running_max = curve.cummax()
    drawdown = (curve / running_max - 1.0).min()
    return {
        "sharpe": round(float(sharpe), 3),
        "max_drawdown": round(float(drawdown), 3),
        "cagr": round(float(cagr), 3),
    }


def run_walk_forward(
    results: list[LoadResult],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> BacktestResult:
    rebal_dates = _month_starts(start, end)
    prices = _price_panel(results)

    algo_curve = [1.0]
    bh_curve = [1.0]
    bench_curve = [1.0]
    curve_dates = [rebal_dates[0]]

    bh_weights: pd.Series | None = None
    bench_weights: pd.Series | None = None

    for i in range(len(rebal_dates) - 1):
        t = rebal_dates[i]
        t_next = rebal_dates[i + 1]

        # Build signals using ONLY data strictly before t.
        signals: list[TickerSignal] = []
        for r in results:
            sig = forecast_ticker(r, as_of=t - pd.Timedelta(days=1))
            if sig is not None:
                signals.append(sig)
        if len(signals) < 2:
            curve_dates.append(t_next)
            algo_curve.append(algo_curve[-1])
            bh_curve.append(bh_curve[-1])
            bench_curve.append(bench_curve[-1])
            continue

        cov = estimate_covariance(results, as_of=t - pd.Timedelta(days=1))
        try:
            opt = optimize_portfolio(cov, signals)
        except Exception as exc:  # numerical edge cases at early dates
            log.warning("optimization failed at %s: %s", t.date(), exc)
            curve_dates.append(t_next)
            algo_curve.append(algo_curve[-1])
            bh_curve.append(bh_curve[-1])
            bench_curve.append(bench_curve[-1])
            continue

        weights = pd.Series(opt["allocation"]).reindex(prices.columns).fillna(0.0)

        # Period return: price change over [t, t_next] for tradable tickers.
        window = prices.loc[t:t_next].ffill()
        if len(window) < 2:
            curve_dates.append(t_next)
            algo_curve.append(algo_curve[-1])
            bh_curve.append(bh_curve[-1])
            bench_curve.append(bench_curve[-1])
            continue
        ret_vec = (window.iloc[-1] / window.iloc[0]) - 1.0
        algo_ret = float((weights * ret_vec.fillna(0)).sum())

        # Buy & hold = first valid allocation, held forever.
        if bh_weights is None:
            bh_weights = weights.copy()
        bh_ret = float((bh_weights * ret_vec.fillna(0)).sum())

        # Benchmark = equal-weight across tradable tickers, set once.
        if bench_weights is None:
            tradable = ret_vec.dropna().index
            bench_weights = pd.Series(1.0 / len(tradable), index=tradable).reindex(
                prices.columns).fillna(0.0)
        bench_ret = float((bench_weights * ret_vec.fillna(0)).sum())

        algo_curve.append(algo_curve[-1] * (1 + algo_ret))
        bh_curve.append(bh_curve[-1] * (1 + bh_ret))
        bench_curve.append(bench_curve[-1] * (1 + bench_ret))
        curve_dates.append(t_next)

    algo_series = pd.Series(algo_curve, index=pd.to_datetime(curve_dates))
    metrics = {
        "algo": _metrics(algo_series),
        "bh": _metrics(pd.Series(bh_curve, index=algo_series.index)),
        "bench": _metrics(pd.Series(bench_curve, index=algo_series.index)),
    }

    # Scale to base 100 for dashboard parity.
    base = 100.0
    return BacktestResult(
        backtest_type="walk_forward",
        dates=[d.date().isoformat() for d in algo_series.index],
        algo=[round(v * base, 4) for v in algo_curve],
        bh=[round(v * base, 4) for v in bh_curve],
        bench=[round(v * base, 4) for v in bench_curve],
        metrics=metrics,
    )
