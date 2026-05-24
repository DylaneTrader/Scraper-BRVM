"""Black-Litterman allocation using Prophet signals as views.

Uses PyPortfolioOpt's BlackLittermanModel with the Ledoit-Wolf covariance from
`covariance.py`. Prophet's 30-day return target becomes the absolute view for
each ticker; view uncertainty derives from Prophet's interval width.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from .covariance import CovarianceResult
from .signals import TickerSignal

log = logging.getLogger(__name__)

MAX_WEIGHT = 0.25  # cap matches the empirical 0.25 ceiling in the existing HTML
MIN_WEIGHT = 0.0


def build_views(
    signals: Iterable[TickerSignal],
    tickers: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Map signals onto BL views: {ticker -> expected 30-day return},
    and per-view confidences (omega diagonals) derived from interval width.

    Vues et covariance partagent le même horizon (30j) pour éviter les Sharpe
    irréalistes qui surviennent quand on annualise une prédiction court terme.
    """
    by_ticker = {s.ticker: s for s in signals}
    views: dict[str, float] = {}
    confidences: dict[str, float] = {}
    for t in tickers:
        s = by_ticker.get(t)
        if s is None or s.last_price <= 0:
            continue
        ret_30d = s.target_30d / s.last_price - 1.0
        views[t] = ret_30d  # already on the 30-day horizon
        rel_width = (s.yhat_upper - s.yhat_lower) / max(s.yhat, 1e-9)
        # Wider interval -> larger view variance -> BL relies more on prior.
        confidences[t] = max(rel_width ** 2, 1e-4)
    return views, confidences


def optimize_portfolio(
    cov_result: CovarianceResult,
    signals: Iterable[TickerSignal],
    max_weight: float = MAX_WEIGHT,
) -> dict:
    from pypfopt import BlackLittermanModel, EfficientFrontier

    tickers = cov_result.tickers
    views, view_vars = build_views(signals, tickers)
    if not views:
        raise ValueError("No usable views to feed Black-Litterman.")

    # Equal-weight prior pi: neutral when no market-cap data is available for BRVM.
    pi = pd.Series(0.0, index=tickers)

    omega = np.diag([view_vars.get(t, 1.0) for t in views.keys()])
    bl = BlackLittermanModel(
        cov_matrix=cov_result.cov,
        pi=pi,
        absolute_views=views,
        omega=omega,
    )
    bl_returns = bl.bl_returns()
    bl_cov = bl.bl_cov()

    # Risk-free rate scaled to the 30-day horizon (3% annual ~= 0.25% monthly).
    RF_30D = (1.03) ** (30 / 365) - 1.0

    # The 25% per-ticker cap (weight_bounds) already provides regularization;
    # no L2 penalty needed (and it conflicts with max_sharpe's reformulation).
    ef = EfficientFrontier(bl_returns, bl_cov, weight_bounds=(MIN_WEIGHT, max_weight))
    ef.max_sharpe(risk_free_rate=RF_30D)
    weights = ef.clean_weights(cutoff=1e-4)

    perf = ef.portfolio_performance(risk_free_rate=RF_30D, verbose=False)
    return {
        "allocation": {t: float(w) for t, w in weights.items() if w > 0},
        "expected_return_30d": float(perf[0]),
        "expected_volatility_30d": float(perf[1]),
        "sharpe_ratio_30d": float(perf[2]),
        "horizon_days": 30,
        "n_views": len(views),
        "covariance": {
            "cond_number_raw": cov_result.cond_number_raw,
            "cond_number_shrunk": cov_result.cond_number_shrunk,
            "used_diagonal_fallback": cov_result.used_fallback,
        },
    }
