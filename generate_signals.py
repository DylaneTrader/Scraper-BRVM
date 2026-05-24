"""Orchestrator: produce data/signals/signals_latest.json and data/algo/portfolio_latest.json.

Run locally or from GitHub Actions. Designed to fail loudly so CI can gate the
git push on a successful run.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant.backtest import run_walk_forward
from quant.covariance import estimate_covariance
from quant.data_loader import load_universe
from quant.portfolio import optimize_portfolio
from quant.signals import (
    detect_concentration,
    forecast_ticker,
    signals_to_dashboard_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("generate_signals")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--skip-backtest", action="store_true",
        help="Skip walk-forward backtest (long-running; useful for quick runs).",
    )
    parser.add_argument(
        "--backtest-start", type=str, default="2023-01-01",
    )
    parser.add_argument(
        "--backtest-end", type=str, default=None,
        help="Default: today.",
    )
    args = parser.parse_args()

    actions_dir = args.root / "data" / "actions"
    out_signals = args.root / "data" / "signals" / "signals_latest.json"
    out_algo = args.root / "data" / "algo" / "portfolio_latest.json"
    out_signals.parent.mkdir(parents=True, exist_ok=True)
    out_algo.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading universe from %s", actions_dir)
    results = load_universe(actions_dir)
    log.info("Loaded %d tickers (%d excluded by liquidity filter)",
             len(results), sum(1 for r in results if r.excluded))

    as_of = pd.Timestamp(datetime.now(tz=timezone.utc).date())
    signals = []
    diagnostics = []
    for r in results:
        diagnostics.append({
            "ticker": r.ticker,
            "illiquid_ratio": round(r.illiquid_ratio, 4),
            "rows_raw": r.rows_raw,
            "rows_kept": r.rows_kept,
            "excluded": r.excluded,
        })
        sig = forecast_ticker(r, as_of=as_of)
        if sig is not None:
            signals.append(sig)
    log.info("Generated %d signals (of %d tickers)", len(signals), len(results))

    concentration = detect_concentration(signals)

    signals_payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "as_of": as_of.date().isoformat(),
        "training_window_years": 3,
        "horizon_days": 30,
        "concentration_diagnostic": concentration,
        "liquidity_diagnostics": diagnostics,
        "signals": signals_to_dashboard_dict(signals),
    }
    out_signals.write_text(json.dumps(signals_payload, indent=2, ensure_ascii=False))
    log.info("Wrote %s", out_signals)

    # Portfolio.
    cov = estimate_covariance(results, as_of=as_of)
    opt = optimize_portfolio(cov, signals)

    # Backtest.
    backtest_payload: dict | None = None
    if not args.skip_backtest:
        bt_end = pd.Timestamp(args.backtest_end) if args.backtest_end else as_of
        log.info("Running walk-forward backtest %s -> %s", args.backtest_start, bt_end.date())
        bt = run_walk_forward(
            results,
            start=pd.Timestamp(args.backtest_start),
            end=bt_end,
        )
        backtest_payload = {
            "backtest_type": bt.backtest_type,
            "dates": bt.dates,
            "algo": bt.algo,
            "bh": bt.bh,
            "bench": bt.bench,
            "metrics": bt.metrics,
        }

    algo_payload = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "allocation": opt["allocation"],
        "horizon_days": opt["horizon_days"],
        "expected_return_30d": opt["expected_return_30d"],
        "expected_volatility_30d": opt["expected_volatility_30d"],
        "sharpe_ratio_30d": opt["sharpe_ratio_30d"],
        "covariance_diagnostic": opt["covariance"],
        "backtest": backtest_payload,
    }
    out_algo.write_text(json.dumps(algo_payload, indent=2, ensure_ascii=False))
    log.info("Wrote %s", out_algo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
