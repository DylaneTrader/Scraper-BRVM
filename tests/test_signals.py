"""Tests for concentration detection and signal classification (Point 2)."""
from __future__ import annotations

from quant.signals import (
    CONCENTRATION_THRESHOLD,
    TickerSignal,
    _classify,
    detect_concentration,
)


def _mk(ticker: str, signal: str) -> TickerSignal:
    return TickerSignal(
        ticker=ticker, last_price=100.0, last_date="2026-01-01",
        signal=signal, confidence="Haute", target_30d=100.0, trend="UP",
        yhat=100.0, yhat_lower=95.0, yhat_upper=105.0,
        train_start="2023-01-01", train_end="2026-01-01",
        train_rows=750, train_annualized_return=0.0,
    )


def test_classify_buckets():
    assert _classify(120, 100) == "ACHAT FORT"  # +20%
    assert _classify(105, 100) == "ACHAT"       # +5%
    assert _classify(100, 100) == "NEUTRE"      # 0%
    assert _classify(95, 100) == "VENTE"        # -5%
    assert _classify(80, 100) == "VENTE FORTE"  # -20%


def test_classify_handles_zero_last_price():
    assert _classify(100, 0) == "NEUTRE"


def test_detect_concentration_flags_when_above_threshold():
    # 8/10 same signal -> ratio 0.8 > 0.7 -> concentrated
    signals = [_mk(f"T{i}", "VENTE FORTE") for i in range(8)]
    signals += [_mk("T8", "NEUTRE"), _mk("T9", "ACHAT")]
    diag = detect_concentration(signals)
    assert diag["concentrated"] is True
    assert diag["dominant_signal"] == "VENTE FORTE"
    assert diag["ratio"] > CONCENTRATION_THRESHOLD
    assert diag["n"] == 10


def test_detect_concentration_passes_when_diverse():
    signals = [
        _mk("T1", "ACHAT FORT"), _mk("T2", "ACHAT"), _mk("T3", "NEUTRE"),
        _mk("T4", "VENTE"), _mk("T5", "VENTE FORTE"),
    ]
    diag = detect_concentration(signals)
    assert diag["concentrated"] is False
    assert diag["ratio"] <= CONCENTRATION_THRESHOLD


def test_detect_concentration_empty_input():
    diag = detect_concentration([])
    assert diag["concentrated"] is False
    assert diag["n"] == 0
