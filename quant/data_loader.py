"""BRVM historical CSV loader with illiquidity filtering (Point 1).

Filters trading days where Volume_Titres == 0 or where the close price is
unchanged across more than `flat_streak_threshold` consecutive sessions
(proxy for no real transaction).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


HISTORIQUE_SUFFIX = "_historique.csv"
MAX_ILLIQUID_RATIO = 0.40
DEFAULT_FLAT_STREAK = 3


@dataclass
class LoadResult:
    ticker: str
    df: pd.DataFrame  # filtered, columns: ds, y, volume
    illiquid_ratio: float
    rows_raw: int
    rows_kept: int
    excluded: bool  # True if illiquid_ratio > MAX_ILLIQUID_RATIO


def _read_brvm_csv(path: Path) -> pd.DataFrame:
    # Files use ';' separator, ',' decimal, UTF-8 BOM.
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Cloture"]).sort_values("Date").reset_index(drop=True)
    df["Volume_Titres"] = pd.to_numeric(df["Volume_Titres"], errors="coerce").fillna(0)
    return df


def _flat_streak_mask(close: pd.Series, threshold: int) -> pd.Series:
    """True for rows that sit inside a flat streak strictly longer than `threshold`.

    A streak of identical closes of length L marks all L rows as flat when L > threshold.
    Using `> threshold` (not >=) matches the spec: "more than 3 sessions".
    """
    same_as_prev = close.eq(close.shift())
    # Group consecutive same-as-previous runs.
    group_id = (~same_as_prev).cumsum()
    run_len = same_as_prev.groupby(group_id).transform("sum") + 1  # +1 for the anchor row
    # Anchor row itself is part of the run; mark it too when the run is long.
    return run_len > threshold


def load_ticker(
    csv_path: Path,
    flat_streak_threshold: int = DEFAULT_FLAT_STREAK,
) -> LoadResult:
    ticker = csv_path.name.replace(HISTORIQUE_SUFFIX, "")
    raw = _read_brvm_csv(csv_path)
    rows_raw = len(raw)
    if rows_raw == 0:
        return LoadResult(ticker, raw.iloc[:0], 1.0, 0, 0, True)

    zero_volume = raw["Volume_Titres"] <= 0
    flat = _flat_streak_mask(raw["Cloture"], flat_streak_threshold)
    illiquid = zero_volume | flat

    illiquid_ratio = float(illiquid.mean())
    kept = raw.loc[~illiquid, ["Date", "Cloture", "Volume_Titres"]].copy()
    kept = kept.rename(columns={"Date": "ds", "Cloture": "y", "Volume_Titres": "volume"})

    excluded = illiquid_ratio > MAX_ILLIQUID_RATIO
    return LoadResult(
        ticker=ticker,
        df=kept.reset_index(drop=True),
        illiquid_ratio=illiquid_ratio,
        rows_raw=rows_raw,
        rows_kept=int(len(kept)),
        excluded=excluded,
    )


def discover_tickers(actions_dir: Path) -> list[Path]:
    return sorted(p for p in actions_dir.glob(f"*{HISTORIQUE_SUFFIX}") if p.is_file())


def load_universe(
    actions_dir: Path,
    flat_streak_threshold: int = DEFAULT_FLAT_STREAK,
) -> list[LoadResult]:
    return [load_ticker(p, flat_streak_threshold) for p in discover_tickers(actions_dir)]
