"""
ETF Builder BRVM - Streamlit
Construction d'un ETF répliquant un indice BRVM (BRVM30 par défaut)
à partir des données scrappées localement.

Lancement :
    streamlit run etf_builder.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

from brvm_tickers import ACTIONS, INDICES, safe_filename


DATA_DIR = Path(__file__).parent / "data"
ACTIONS_DIR = DATA_DIR / "actions"
INDICES_DIR = DATA_DIR / "indices"
SOCIETES_DIR = DATA_DIR / "societes"
DIVIDENDES_FILE = DATA_DIR / "dividendes" / "dividendes_historique.csv"

NUM_COLS = ["Ouverture", "Plus_Haut", "Plus_Bas", "Cloture", "Variation_Pct"]


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
def _read_brvm_csv(path: Path) -> pd.DataFrame:
    """Lit un CSV BRVM (séparateur ';' et virgule décimale)."""
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume_Titres" in df.columns:
        df["Volume_Titres"] = pd.to_numeric(df["Volume_Titres"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_action(ticker: str) -> pd.DataFrame | None:
    fname = safe_filename(ticker) + "_historique.csv"
    path = ACTIONS_DIR / fname
    if not path.exists():
        return None
    df = _read_brvm_csv(path)
    return df[["Date", "Cloture", "Volume_Titres"]].rename(
        columns={"Cloture": ticker, "Volume_Titres": f"{ticker}_vol"}
    )


@st.cache_data(show_spinner=False)
def load_index(symbol: str) -> pd.DataFrame | None:
    fname = safe_filename(symbol) + "_historique.csv"
    path = INDICES_DIR / fname
    if not path.exists():
        return None
    df = _read_brvm_csv(path)
    return df[["Date", "Cloture"]].rename(columns={"Cloture": symbol})


def _to_float(s: str | float | int | None) -> float | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace("\xa0", "").replace(" ", "").replace("%", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def load_societes() -> pd.DataFrame:
    rows = []
    for path in SOCIETES_DIR.glob("*_societe.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(obj)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalisation des champs numériques utiles
    df["Nb_Titres"] = df.get("Nombre_Actions", df.get("Nombre de titres")).apply(_to_float)
    df["Flottant_Pct_num"] = df.get("Flottant_Pct").apply(_to_float)
    df["BNPA_num"] = df.get("BNPA").apply(_to_float)
    df["PER_num"] = df.get("PER").apply(_to_float)
    df["Dividende_num"] = df.get("Dividende").apply(_to_float)
    df["Resultat_Net_num"] = df.get("Résultat net").apply(_to_float)
    df["CA_num"] = df.get("Chiffre d'affaires").apply(_to_float)
    df = df.rename(columns={"ticker": "Ticker"})
    return df


@st.cache_data(show_spinner=False)
def load_dividendes() -> pd.DataFrame:
    if not DIVIDENDES_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(DIVIDENDES_FILE, sep=";", decimal=",", encoding="utf-8-sig")
    return df


@st.cache_data(show_spinner=False)
def build_prices_panel(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Construit le panel de cours de clôture pour les tickers donnés."""
    frames = []
    for t in tickers:
        df = load_action(t)
        if df is None:
            continue
        frames.append(df[["Date", t]].set_index("Date"))
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1).sort_index()
    return panel


@st.cache_data(show_spinner=False)
def build_volumes_panel(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Panel des volumes en titres (Volume_Titres) pour la liquidité."""
    frames = []
    for t in tickers:
        df = load_action(t)
        if df is None:
            continue
        col = f"{t}_vol"
        if col not in df.columns:
            continue
        frames.append(df[["Date", col]].set_index("Date").rename(columns={col: t}))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


# ---------------------------------------------------------------------------
# Filtres de liquidité
# ---------------------------------------------------------------------------
def compute_liquidity_metrics(
    tickers: list[str],
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    societes: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule un tableau de métriques de liquidité par ticker sur la période fournie."""
    rows = []
    soc_idx = societes.set_index("Ticker") if not societes.empty else pd.DataFrame()

    n_total = len(prices.index) if not prices.empty else 0
    for t in tickers:
        px = prices[t] if t in prices.columns else pd.Series(dtype=float)
        vol = volumes[t] if (volumes is not None and t in volumes.columns) else pd.Series(dtype=float)

        last_px = float(px.dropna().iloc[-1]) if px.notna().any() else np.nan
        days_traded = int((vol.fillna(0) > 0).sum()) if not vol.empty else 0
        share_traded = days_traded / n_total if n_total else 0.0
        avg_vol = float(vol.mean()) if not vol.empty else 0.0
        med_vol = float(vol.median()) if not vol.empty else 0.0
        # Valeur transigée moyenne (FCFA) ≈ vol * cours du jour
        traded_value = (vol * px).dropna() if not vol.empty else pd.Series(dtype=float)
        adv_fcfa = float(traded_value.mean()) if not traded_value.empty else 0.0

        # Capi & flottant via fiche société
        nb_titres = float(soc_idx.loc[t, "Nb_Titres"]) if t in soc_idx.index and pd.notna(soc_idx.loc[t, "Nb_Titres"]) else np.nan
        flottant_pct = float(soc_idx.loc[t, "Flottant_Pct_num"]) if t in soc_idx.index and pd.notna(soc_idx.loc[t, "Flottant_Pct_num"]) else np.nan
        market_cap = (last_px * nb_titres) if (not np.isnan(last_px) and not np.isnan(nb_titres)) else np.nan
        free_float_cap = (market_cap * flottant_pct / 100.0) if (not np.isnan(market_cap) and not np.isnan(flottant_pct)) else np.nan

        rows.append({
            "Ticker": t,
            "Dernier cours": last_px,
            "Jours cotés": days_traded,
            "% jours cotés": share_traded * 100,
            "Volume moyen (titres)": avg_vol,
            "Volume médian (titres)": med_vol,
            "ADV (FCFA)": adv_fcfa,
            "Nb titres": nb_titres,
            "Capi (FCFA)": market_cap,
            "Flottant %": flottant_pct,
            "Capi flottante (FCFA)": free_float_cap,
        })
    return pd.DataFrame(rows)


def apply_liquidity_filters(
    metrics: pd.DataFrame,
    *,
    min_share_traded_pct: float | None = None,
    min_avg_vol: float | None = None,
    min_adv_fcfa: float | None = None,
    min_market_cap: float | None = None,
    min_free_float_cap: float | None = None,
    min_price: float | None = None,
    min_flottant_pct: float | None = None,
    drop_na: bool = False,
) -> pd.DataFrame:
    """Applique les filtres de liquidité paramétrés. Renvoie le sous-DataFrame retenu."""
    df = metrics.copy()
    masks = []

    def add(col, mn):
        if mn is None:
            return
        if drop_na:
            masks.append(df[col].fillna(-np.inf) >= mn)
        else:
            masks.append(df[col].isna() | (df[col] >= mn))

    add("% jours cotés", min_share_traded_pct)
    add("Volume moyen (titres)", min_avg_vol)
    add("ADV (FCFA)", min_adv_fcfa)
    add("Capi (FCFA)", min_market_cap)
    add("Capi flottante (FCFA)", min_free_float_cap)
    add("Dernier cours", min_price)
    add("Flottant %", min_flottant_pct)

    if masks:
        keep = masks[0]
        for m in masks[1:]:
            keep = keep & m
        df = df[keep].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Pondérations
# ---------------------------------------------------------------------------
def equal_weights(tickers: list[str]) -> pd.Series:
    n = len(tickers)
    return pd.Series(1.0 / n, index=tickers)


def market_cap_weights(tickers: list[str], societes: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Capi-pondéré : nb titres * dernier cours."""
    last_px = prices[tickers].ffill().iloc[-1]
    nb = societes.set_index("Ticker").reindex(tickers)["Nb_Titres"]
    cap = (last_px * nb).fillna(0.0)
    if cap.sum() <= 0:
        return equal_weights(tickers)
    return cap / cap.sum()


def free_float_weights(tickers: list[str], societes: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    last_px = prices[tickers].ffill().iloc[-1]
    soc = societes.set_index("Ticker").reindex(tickers)
    nb = soc["Nb_Titres"].fillna(0.0)
    flottant = (soc["Flottant_Pct_num"].fillna(100.0) / 100.0).clip(0, 1)
    cap_ff = last_px * nb * flottant
    if cap_ff.sum() <= 0:
        return equal_weights(tickers)
    return cap_ff / cap_ff.sum()


def inverse_vol_weights(tickers: list[str], returns: pd.DataFrame) -> pd.Series:
    vol = returns[tickers].std()
    inv = 1.0 / vol.replace(0, np.nan)
    inv = inv.fillna(0.0)
    if inv.sum() <= 0:
        return equal_weights(tickers)
    return inv / inv.sum()


def min_tracking_error_weights(
    tickers: list[str],
    asset_returns: pd.DataFrame,
    bench_returns: pd.Series,
    max_weight: float = 0.30,
) -> pd.Series:
    """Minimisation de la tracking error vs benchmark, contraintes long-only + cap."""
    R = asset_returns[tickers].dropna(how="all").fillna(0.0)
    b = bench_returns.reindex(R.index).fillna(0.0).values
    A = R.values
    n = len(tickers)

    def te(w: np.ndarray) -> float:
        diff = A @ w - b
        return float(np.std(diff))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = np.full(n, 1.0 / n)
    res = minimize(te, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-9})
    w = res.x if res.success else x0
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        w = np.full(n, 1.0 / n)
    return pd.Series(w / w.sum(), index=tickers)


def max_sharpe_weights(
    tickers: list[str],
    returns: pd.DataFrame,
    rf_annual: float = 0.03,
    max_weight: float = 0.30,
) -> pd.Series:
    R = returns[tickers].dropna(how="all").fillna(0.0)
    mu = R.mean().values * 252
    cov = R.cov().values * 252
    rf = rf_annual
    n = len(tickers)

    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        if vol <= 0:
            return 1e6
        return -(ret - rf) / vol

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = np.full(n, 1.0 / n)
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-9})
    w = res.x if res.success else x0
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        w = np.full(n, 1.0 / n)
    return pd.Series(w / w.sum(), index=tickers)


# ---------------------------------------------------------------------------
# Backtest et métriques
# ---------------------------------------------------------------------------
def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    cols = [c for c in weights.index if c in asset_returns.columns]
    w = weights.reindex(cols).fillna(0.0).values
    R = asset_returns[cols].fillna(0.0).values
    return pd.Series(R @ w, index=asset_returns.index, name="ETF")


def cumulative_curve(rets: pd.Series, base: float = 100.0) -> pd.Series:
    return base * (1.0 + rets.fillna(0.0)).cumprod()


def annualized_return(rets: pd.Series, periods: int = 252) -> float:
    if len(rets) == 0:
        return 0.0
    total = (1.0 + rets.fillna(0.0)).prod()
    years = len(rets) / periods
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def annualized_vol(rets: pd.Series, periods: int = 252) -> float:
    return float(rets.std() * np.sqrt(periods))


def max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def tracking_error(etf_rets: pd.Series, bench_rets: pd.Series, periods: int = 252) -> float:
    diff = (etf_rets - bench_rets).dropna()
    if len(diff) == 0:
        return 0.0
    return float(diff.std() * np.sqrt(periods))


def beta_alpha(etf_rets: pd.Series, bench_rets: pd.Series, periods: int = 252) -> tuple[float, float]:
    df = pd.concat([etf_rets, bench_rets], axis=1).dropna()
    if len(df) < 2:
        return 0.0, 0.0
    cov = df.cov().iloc[0, 1]
    var = df.iloc[:, 1].var()
    beta = float(cov / var) if var > 0 else 0.0
    alpha = float(df.iloc[:, 0].mean() - beta * df.iloc[:, 1].mean()) * periods
    return beta, alpha


# ---------------------------------------------------------------------------
# Helpers réutilisables (UI + grid search)
# ---------------------------------------------------------------------------
WEIGHTING_METHODS = [
    "Équipondéré",
    "Capi-pondéré",
    "Free-float pondéré",
    "Inverse de la volatilité",
    "Tracking error min (vs indice)",
    "Sharpe maximal",
]
REBAL_LABELS = ["Aucun (buy & hold)", "Mensuel", "Trimestriel", "Semestriel", "Annuel"]


def rebalance_freq_map(label: str) -> str | None:
    return {
        "Aucun (buy & hold)": None,
        "Mensuel": "ME",
        "Trimestriel": "QE",
        "Semestriel": "2QE",
        "Annuel": "YE",
    }[label]


def backtest_with_rebalance(
    asset_returns: pd.DataFrame,
    target_weights: pd.Series,
    freq: str | None,
) -> pd.Series:
    if freq is None:
        return portfolio_returns(asset_returns, target_weights)
    rebal_dates = pd.date_range(asset_returns.index.min(), asset_returns.index.max(), freq=freq)
    rebal_dates = asset_returns.index[asset_returns.index.searchsorted(rebal_dates).clip(0, len(asset_returns) - 1)]
    rebal_set = set(pd.DatetimeIndex(sorted(set(rebal_dates))))

    cols = list(target_weights.index)
    R = asset_returns[cols].fillna(0.0).values
    w = target_weights.values.copy()
    pf_ret = np.zeros(R.shape[0])

    for i, dt in enumerate(asset_returns.index):
        r = R[i]
        pf_ret[i] = float(w @ r)
        new_val = w * (1.0 + r)
        s = new_val.sum()
        w = new_val / s if s > 0 else target_weights.values.copy()
        if dt in rebal_set:
            w = target_weights.values.copy()

    return pd.Series(pf_ret, index=asset_returns.index, name="ETF")


def compute_weights(
    method: str,
    tickers: list[str],
    asset_returns: pd.DataFrame,
    bench_rets: pd.Series,
    prices: pd.DataFrame,
    societes: pd.DataFrame,
    max_weight: float,
    rf: float,
) -> pd.Series:
    if method == "Équipondéré":
        w = equal_weights(tickers)
    elif method == "Capi-pondéré":
        w = market_cap_weights(tickers, societes, prices)
    elif method == "Free-float pondéré":
        w = free_float_weights(tickers, societes, prices)
    elif method == "Inverse de la volatilité":
        w = inverse_vol_weights(tickers, asset_returns)
    elif method == "Tracking error min (vs indice)":
        w = min_tracking_error_weights(tickers, asset_returns, bench_rets, max_weight=max_weight)
    elif method == "Sharpe maximal":
        w = max_sharpe_weights(tickers, asset_returns, rf_annual=rf, max_weight=max_weight)
    else:
        raise ValueError(method)
    if method in {"Équipondéré", "Capi-pondéré", "Free-float pondéré", "Inverse de la volatilité"}:
        w = w.clip(upper=max_weight)
        s = w.sum()
        if s > 0:
            w = w / s
    return w


def run_backtest(
    method: str,
    tickers: list[str],
    asset_returns: pd.DataFrame,
    bench_rets: pd.Series,
    prices: pd.DataFrame,
    societes: pd.DataFrame,
    max_weight: float,
    rf: float,
    rebalance_label: str,
) -> dict:
    """Calcule poids + rendements ETF + métriques sur la sélection donnée."""
    if not tickers:
        return {}
    asset_rets_sel = asset_returns[tickers].dropna(how="all")
    weights = compute_weights(method, tickers, asset_rets_sel, bench_rets, prices, societes, max_weight, rf)
    etf_rets = backtest_with_rebalance(asset_rets_sel, weights, rebalance_freq_map(rebalance_label))
    idx = etf_rets.index.intersection(bench_rets.index)
    etf_rets = etf_rets.loc[idx]
    b = bench_rets.loc[idx]
    curve = cumulative_curve(etf_rets)
    ann_etf = annualized_return(etf_rets)
    vol_etf = annualized_vol(etf_rets)
    beta, alpha = beta_alpha(etf_rets, b)
    return {
        "weights": weights,
        "etf_rets": etf_rets,
        "bench_rets": b,
        "curve": curve,
        "ann_return": ann_etf,
        "ann_vol": vol_etf,
        "tracking_error": tracking_error(etf_rets, b),
        "beta": beta,
        "alpha": alpha,
        "sharpe": (ann_etf - rf) / vol_etf if vol_etf > 0 else 0.0,
        "max_drawdown": max_drawdown(curve),
        "n_assets": int((weights > 1e-4).sum()),
    }


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ETF Builder BRVM", page_icon=":bar_chart:", layout="wide")

st.title("ETF Builder BRVM")
st.caption("Construis un ETF répliquant un indice BRVM à partir des données locales.")

societes = load_societes()

# Sidebar : paramètres ----------------------------------------------------
with st.sidebar:
    st.header("Paramètres")

    bench_symbol = st.selectbox(
        "Indice de référence",
        options=INDICES,
        index=INDICES.index("BRVM30"),
        help="Indice servant de benchmark à l'ETF.",
    )

    universe_choice = st.radio(
        "Univers d'actions candidates",
        ["Toutes les actions BRVM", "Filtrer par secteur"],
        index=0,
    )

    if universe_choice == "Filtrer par secteur" and not societes.empty:
        sectors = sorted(societes["Secteur"].dropna().unique().tolist())
        sel_sectors = st.multiselect("Secteurs", sectors, default=sectors)
        universe = societes[societes["Secteur"].isin(sel_sectors)]["Ticker"].dropna().tolist()
    else:
        universe = ACTIONS

    method = st.selectbox(
        "Méthode de pondération",
        WEIGHTING_METHODS,
        index=4,
    )

    n_max = st.slider("Nombre max d'actifs dans l'ETF", 5, min(40, len(universe)), 15)
    cap_w = st.slider("Plafond de poids par action (%)", 5, 100, 25) / 100.0

    today = pd.Timestamp.today().normalize()
    default_start = today - pd.DateOffset(years=2)
    start_date = st.date_input("Date de début", value=default_start.date())
    end_date = st.date_input("Date de fin", value=today.date())

    rebalance = st.selectbox(
        "Rebalancement",
        REBAL_LABELS,
        index=2,
    )
    rf = st.number_input("Taux sans risque annuel (%)", value=3.0, step=0.5) / 100.0

    st.markdown("---")
    with st.expander("💧 Contraintes de liquidité", expanded=False):
        st.caption("Filtre l'univers candidat avant la sélection top-N. Mets 0 pour désactiver.")
        liq_min_share_traded = st.slider(
            "Min % de jours cotés sur la période",
            0, 100, 30,
            help="Pourcentage de jours avec volume > 0 sur la fenêtre."
        )
        liq_min_avg_vol = st.number_input(
            "Min volume moyen quotidien (titres)",
            min_value=0, value=0, step=100,
            help="Moyenne du nb de titres échangés par jour."
        )
        liq_min_adv_fcfa = st.number_input(
            "Min ADV en FCFA (volume × cours moyen)",
            min_value=0, value=0, step=1_000_000,
            help="Average Daily Value : valeur moyenne échangée par séance."
        )
        liq_min_market_cap = st.number_input(
            "Min capitalisation totale (FCFA)",
            min_value=0, value=0, step=1_000_000_000,
            help="Cours × nb total de titres."
        )
        liq_min_free_float_cap = st.number_input(
            "Min capitalisation flottante (FCFA)",
            min_value=0, value=0, step=1_000_000_000,
            help="Capi totale × % flottant."
        )
        liq_min_price = st.number_input(
            "Min cours en FCFA",
            min_value=0.0, value=0.0, step=100.0,
            help="Évite les actions à très petit nominal."
        )
        liq_min_flottant_pct = st.slider(
            "Min flottant (%)",
            0, 100, 0,
            help="Part flottante minimale exigée."
        )
        liq_drop_na = st.checkbox(
            "Exclure les titres sans donnée pour un critère",
            value=False,
            help="Si décoché : un titre dont la métrique est inconnue passe le filtre.",
        )

# Chargement benchmark + univers -----------------------------------------
bench_df = load_index(bench_symbol)
if bench_df is None or bench_df.empty:
    st.error(f"Indice {bench_symbol} introuvable dans data/indices/.")
    st.stop()

bench_df = bench_df.set_index("Date")
bench_df = bench_df.loc[str(start_date):str(end_date)]
bench_rets = bench_df[bench_symbol].pct_change().dropna()

# Panel des prix + volumes sur l'univers
prices = build_prices_panel(tuple(universe))
volumes = build_volumes_panel(tuple(universe))
if prices.empty:
    st.error("Aucun cours disponible pour l'univers sélectionné.")
    st.stop()
prices = prices.loc[str(start_date):str(end_date)]
if not volumes.empty:
    volumes = volumes.loc[str(start_date):str(end_date)]

# Aligner sur les dates du benchmark
common_idx = bench_df.index.intersection(prices.index)
prices = prices.reindex(common_idx).ffill()
volumes = volumes.reindex(common_idx) if not volumes.empty else volumes
bench_aligned = bench_df.reindex(common_idx).ffill()
bench_rets = bench_aligned[bench_symbol].pct_change().dropna()

# Calcul des rendements quotidiens
asset_returns = prices.pct_change().dropna(how="all")

# Métriques de liquidité sur l'univers + filtrage
liq_metrics = compute_liquidity_metrics(universe, prices, volumes, societes)
liq_filtered = apply_liquidity_filters(
    liq_metrics,
    min_share_traded_pct=liq_min_share_traded if liq_min_share_traded > 0 else None,
    min_avg_vol=liq_min_avg_vol if liq_min_avg_vol > 0 else None,
    min_adv_fcfa=liq_min_adv_fcfa if liq_min_adv_fcfa > 0 else None,
    min_market_cap=liq_min_market_cap if liq_min_market_cap > 0 else None,
    min_free_float_cap=liq_min_free_float_cap if liq_min_free_float_cap > 0 else None,
    min_price=liq_min_price if liq_min_price > 0 else None,
    min_flottant_pct=liq_min_flottant_pct if liq_min_flottant_pct > 0 else None,
    drop_na=liq_drop_na,
)
liquid_universe = liq_filtered["Ticker"].tolist()

if not liquid_universe:
    st.error("Aucun titre ne passe les filtres de liquidité. Détends les contraintes.")
    st.stop()

# Sélection des n_max actions les plus liquides
ranking = asset_returns.apply(lambda s: s.notna().sum(), axis=0).sort_values(ascending=False)
candidates = [t for t in ranking.index if t in liquid_universe]
selected = candidates[:n_max]

st.sidebar.markdown("---")
st.sidebar.write(
    f"**Univers** : {len(universe)} → **{len(liquid_universe)}** après liquidité → "
    f"**{len(selected)}** retenus (top-{n_max})"
)
selected = st.sidebar.multiselect(
    "Ajuster la sélection",
    options=candidates,
    default=selected,
)

if not selected:
    st.warning("Sélectionne au moins une action.")
    st.stop()

# ------------------------------------------------------------------------
# Calcul des poids + backtest pour la méthode courante
# ------------------------------------------------------------------------
asset_returns_sel = asset_returns[selected].dropna(how="all")

result = run_backtest(
    method, selected, asset_returns, bench_rets, prices, societes,
    cap_w, rf, rebalance,
)
weights = result["weights"]
etf_rets = result["etf_rets"]
bench_rets_c = result["bench_rets"]
etf_curve = result["curve"]
bench_curve = cumulative_curve(bench_rets_c)
common = etf_rets.index

# ------------------------------------------------------------------------
# Affichage
# ------------------------------------------------------------------------
tab_portf, tab_perf, tab_compo, tab_liq, tab_grid, tab_data, tab_doc = st.tabs(
    [
        ":briefcase: Portefeuille",
        ":chart_with_upwards_trend: Performance",
        ":pie_chart: Composition",
        ":droplet: Liquidité",
        ":mag: Grid Search",
        ":open_file_folder: Données",
        ":books: Documentation",
    ]
)

with tab_portf:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indice", bench_symbol)
    c2.metric("Méthode", method)
    c3.metric("Actions", len(weights))
    c4.metric("Période", f"{common.min().date()} → {common.max().date()}" if len(common) else "-")

    poids_df = (
        pd.DataFrame({"Ticker": weights.index, "Poids": weights.values})
        .sort_values("Poids", ascending=False)
        .reset_index(drop=True)
    )
    poids_df["Poids %"] = (poids_df["Poids"] * 100).round(2)

    # Enrichir avec nom + secteur si dispo
    if not societes.empty:
        soc_lite = societes[["Ticker", "Nom", "Secteur"]].drop_duplicates("Ticker")
        poids_df = poids_df.merge(soc_lite, on="Ticker", how="left")

    fig_w = px.bar(
        poids_df, x="Ticker", y="Poids %",
        hover_data=["Nom", "Secteur"] if "Nom" in poids_df.columns else None,
        title="Pondérations de l'ETF",
    )
    st.plotly_chart(fig_w, use_container_width=True)
    st.dataframe(poids_df, use_container_width=True, hide_index=True)

with tab_perf:
    ann_etf = annualized_return(etf_rets)
    ann_bench = annualized_return(bench_rets_c)
    vol_etf = annualized_vol(etf_rets)
    vol_bench = annualized_vol(bench_rets_c)
    te = tracking_error(etf_rets, bench_rets_c)
    beta, alpha = beta_alpha(etf_rets, bench_rets_c)
    sharpe_etf = (ann_etf - rf) / vol_etf if vol_etf > 0 else 0.0
    mdd = max_drawdown(etf_curve)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Perf annualisée ETF", f"{ann_etf*100:.2f}%", f"vs {ann_bench*100:.2f}% bench")
    m2.metric("Volatilité ann. ETF", f"{vol_etf*100:.2f}%", f"vs {vol_bench*100:.2f}% bench")
    m3.metric("Tracking error", f"{te*100:.2f}%")
    m4.metric("Sharpe ETF", f"{sharpe_etf:.2f}")

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Beta", f"{beta:.2f}")
    n2.metric("Alpha annualisé", f"{alpha*100:.2f}%")
    n3.metric("Max drawdown", f"{mdd*100:.2f}%")
    n4.metric("Jours de données", f"{len(etf_rets)}")

    curve_df = pd.DataFrame({"ETF": etf_curve, bench_symbol: bench_curve}).dropna()
    fig_perf = px.line(curve_df, title=f"Performance cumulée (base 100) — ETF vs {bench_symbol}")
    fig_perf.update_layout(yaxis_title="Base 100", xaxis_title=None, legend_title=None)
    st.plotly_chart(fig_perf, use_container_width=True)

    excess = (etf_rets - bench_rets_c).dropna()
    excess_curve = (1 + excess).cumprod() - 1
    fig_excess = px.area(excess_curve * 100, title="Excès de performance cumulé vs benchmark (%)")
    fig_excess.update_layout(yaxis_title="%", xaxis_title=None, showlegend=False)
    st.plotly_chart(fig_excess, use_container_width=True)

    # Drawdown
    dd = etf_curve / etf_curve.cummax() - 1.0
    fig_dd = px.area(dd * 100, title="Drawdown ETF (%)")
    fig_dd.update_layout(yaxis_title="%", xaxis_title=None, showlegend=False)
    st.plotly_chart(fig_dd, use_container_width=True)

with tab_compo:
    if not societes.empty:
        sec_df = poids_df.groupby("Secteur", dropna=False)["Poids"].sum().reset_index()
        sec_df["Secteur"] = sec_df["Secteur"].fillna("Non classé")
        fig_sec = px.pie(sec_df, names="Secteur", values="Poids", title="Allocation sectorielle de l'ETF")
        st.plotly_chart(fig_sec, use_container_width=True)

    fig_pie = px.pie(poids_df, names="Ticker", values="Poids", title="Répartition par titre")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Matrice de corrélation des actions sélectionnées
    corr = asset_returns_sel.corr()
    fig_corr = px.imshow(
        corr, x=corr.columns, y=corr.columns, color_continuous_scale="RdBu", zmin=-1, zmax=1,
        title="Corrélation des rendements quotidiens"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab_liq:
    st.subheader("Métriques de liquidité de l'univers")
    st.caption(
        "Calculées sur la période sélectionnée. Le filtrage en sidebar exclut les titres "
        "qui ne respectent pas tes seuils avant la sélection top-N."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Univers initial", len(universe))
    c2.metric("Après filtres liquidité", len(liquid_universe))
    c3.metric("Top-N final", len(selected))

    liq_show = liq_metrics.copy()
    liq_show["Retenu"] = liq_show["Ticker"].isin(liquid_universe)
    liq_show["Sélectionné"] = liq_show["Ticker"].isin(selected)
    cols_fmt = {
        "Dernier cours": "{:,.0f}",
        "% jours cotés": "{:.1f}",
        "Volume moyen (titres)": "{:,.0f}",
        "Volume médian (titres)": "{:,.0f}",
        "ADV (FCFA)": "{:,.0f}",
        "Nb titres": "{:,.0f}",
        "Capi (FCFA)": "{:,.0f}",
        "Flottant %": "{:.2f}",
        "Capi flottante (FCFA)": "{:,.0f}",
    }
    st.dataframe(
        liq_show.style.format(cols_fmt, na_rep="-").apply(
            lambda r: ["background-color: #d4edda" if r["Sélectionné"]
                       else ("background-color: #fff3cd" if r["Retenu"] else "background-color: #f8d7da")] * len(r),
            axis=1,
        ),
        use_container_width=True, hide_index=True,
    )
    st.caption("🟢 Sélectionné dans l'ETF · 🟡 Passe les filtres mais hors top-N · 🔴 Exclu par les filtres")

    if not liq_metrics.empty:
        fig_adv = px.bar(
            liq_metrics.sort_values("ADV (FCFA)", ascending=False),
            x="Ticker", y="ADV (FCFA)",
            title="ADV (Average Daily Value, FCFA)",
        )
        st.plotly_chart(fig_adv, use_container_width=True)

        fig_cap = px.scatter(
            liq_metrics.dropna(subset=["Capi (FCFA)", "ADV (FCFA)"]),
            x="Capi (FCFA)", y="ADV (FCFA)", text="Ticker",
            log_x=True, log_y=True,
            title="ADV vs Capitalisation (échelle log)",
        )
        fig_cap.update_traces(textposition="top center")
        st.plotly_chart(fig_cap, use_container_width=True)


with tab_grid:
    st.subheader("Grid Search — exploration de la grille de paramètres")
    st.caption(
        "Combine plusieurs valeurs pour chaque dimension. L'app teste tous les couples "
        "(méthode × n_max × cap × rebal × seuils de liquidité) puis classe les ETF selon l'objectif retenu."
    )

    st.markdown("**Allocation**")
    col1, col2 = st.columns(2)
    with col1:
        gs_methods = st.multiselect(
            "Méthodes de pondération",
            WEIGHTING_METHODS,
            default=["Tracking error min (vs indice)", "Capi-pondéré", "Équipondéré"],
        )
        gs_n_values = st.multiselect(
            "Nombre max d'actifs (n_max)",
            options=[5, 8, 10, 12, 15, 20, 25, 30],
            default=[10, 15, 20],
        )
    with col2:
        gs_caps = st.multiselect(
            "Plafonds par titre (%)",
            options=[5, 10, 15, 20, 25, 30, 40, 50, 100],
            default=[15, 25, 40],
        )
        gs_rebals = st.multiselect(
            "Fréquences de rebalancement",
            REBAL_LABELS,
            default=["Trimestriel"],
        )

    st.markdown("**Liquidité** — coche les seuils que tu veux balayer (les autres restent fixés à la sidebar)")
    use_share = st.checkbox("Balayer % jours cotés", value=False, key="gs_use_share")
    if use_share:
        gs_share_traded = st.multiselect(
            "Min % jours cotés",
            options=[0, 25, 50, 70, 80, 90, 95, 99],
            default=[0, 50, 90],
        )
    else:
        gs_share_traded = [liq_min_share_traded]

    use_adv = st.checkbox("Balayer ADV minimum (FCFA)", value=False, key="gs_use_adv")
    if use_adv:
        gs_adv = st.multiselect(
            "Min ADV (FCFA)",
            options=[0, 1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000, 250_000_000],
            default=[0, 10_000_000, 50_000_000],
            format_func=lambda x: f"{x:,}".replace(",", " ") if x else "0",
        )
    else:
        gs_adv = [liq_min_adv_fcfa]

    use_ffcap = st.checkbox("Balayer capitalisation flottante minimum (FCFA)", value=False, key="gs_use_ffcap")
    if use_ffcap:
        gs_ffcap = st.multiselect(
            "Min capi flottante (FCFA)",
            options=[0, 10_000_000_000, 50_000_000_000, 100_000_000_000, 250_000_000_000, 500_000_000_000],
            default=[0, 50_000_000_000, 250_000_000_000],
            format_func=lambda x: f"{x:,}".replace(",", " ") if x else "0",
        )
    else:
        gs_ffcap = [liq_min_free_float_cap]

    use_avgvol = st.checkbox("Balayer volume moyen minimum (titres)", value=False, key="gs_use_avgvol")
    if use_avgvol:
        gs_avgvol = st.multiselect(
            "Min volume moyen (titres)",
            options=[0, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000],
            default=[0, 1_000, 10_000],
        )
    else:
        gs_avgvol = [liq_min_avg_vol]

    objective = st.selectbox(
        "Objectif de classement",
        [
            "Tracking error (min)",
            "Sharpe (max)",
            "Perf annualisée (max)",
            "Alpha annualisé (max)",
            "Max drawdown (min en valeur absolue)",
            "Score composite (Sharpe − TE)",
        ],
        index=0,
    )

    n_liq_combos = len(gs_share_traded) * len(gs_adv) * len(gs_ffcap) * len(gs_avgvol)
    n_alloc_combos = len(gs_methods) * len(gs_n_values) * len(gs_caps) * len(gs_rebals)
    n_combos = n_liq_combos * n_alloc_combos
    st.caption(
        f"**{n_combos}** combinaisons = {n_alloc_combos} allocations × {n_liq_combos} configs liquidité. "
        f"Univers initial : **{len(universe)}** titres."
    )
    if n_combos > 1500:
        st.warning(f"⚠️ {n_combos} combos — l'exécution peut prendre plusieurs minutes. Réduis la grille si besoin.")

    run_grid = st.button(":rocket: Lancer le Grid Search", type="primary", disabled=(n_combos == 0))

    if run_grid:
        results_rows = []
        progress = st.progress(0.0)
        status = st.empty()
        i = 0

        # Pré-calcul des univers liquides pour chaque combo de seuils (cache local)
        liq_universe_cache: dict[tuple, list[str]] = {}

        def _liquid_universe_for(share, adv, ffcap, avgvol):
            key = (share, adv, ffcap, avgvol)
            if key in liq_universe_cache:
                return liq_universe_cache[key]
            filt = apply_liquidity_filters(
                liq_metrics,
                min_share_traded_pct=share if share > 0 else None,
                min_avg_vol=avgvol if avgvol > 0 else None,
                min_adv_fcfa=adv if adv > 0 else None,
                min_market_cap=liq_min_market_cap if liq_min_market_cap > 0 else None,
                min_free_float_cap=ffcap if ffcap > 0 else None,
                min_price=liq_min_price if liq_min_price > 0 else None,
                min_flottant_pct=liq_min_flottant_pct if liq_min_flottant_pct > 0 else None,
                drop_na=liq_drop_na,
            )["Ticker"].tolist()
            # Ordre : top liquidité (jours cotés) intersecté avec l'ordre du ranking
            ordered = [t for t in ranking.index if t in filt]
            liq_universe_cache[key] = ordered
            return ordered

        for share_v in gs_share_traded:
            for adv_v in gs_adv:
                for ffcap_v in gs_ffcap:
                    for avgvol_v in gs_avgvol:
                        liq_uni = _liquid_universe_for(share_v, adv_v, ffcap_v, avgvol_v)
                        n_liq = len(liq_uni)

                        for m in gs_methods:
                            for n_v in gs_n_values:
                                sel_g = liq_uni[:n_v]
                                n_eff = len(sel_g)
                                for cap_v in gs_caps:
                                    for reb_v in gs_rebals:
                                        i += 1
                                        progress.progress(i / max(n_combos, 1))
                                        if i % 25 == 0 or i == n_combos:
                                            status.caption(f"{i}/{n_combos} — {m} · n={n_v} · cap={cap_v}% · liq={n_liq} titres")

                                        base_row = {
                                            "Méthode": m,
                                            "n_max": n_v,
                                            "Cap %": cap_v,
                                            "Rebal": reb_v,
                                            "% jours cotés min": share_v,
                                            "ADV min (FCFA)": adv_v,
                                            "Capi flot. min (FCFA)": ffcap_v,
                                            "Vol moyen min": avgvol_v,
                                            "Univers liquide": n_liq,
                                            "n titres ETF": n_eff,
                                        }

                                        if n_eff < 2:
                                            results_rows.append({
                                                **base_row,
                                                "Perf ann.": np.nan, "Vol ann.": np.nan,
                                                "Tracking error": np.nan, "Sharpe": np.nan,
                                                "Beta": np.nan, "Alpha ann.": np.nan,
                                                "Max DD": np.nan,
                                                "Erreur": "Univers trop petit",
                                            })
                                            continue
                                        try:
                                            r = run_backtest(
                                                m, sel_g, asset_returns, bench_rets, prices, societes,
                                                cap_v / 100.0, rf, reb_v,
                                            )
                                            if not r:
                                                continue
                                            results_rows.append({
                                                **base_row,
                                                "Perf ann.": r["ann_return"],
                                                "Vol ann.": r["ann_vol"],
                                                "Tracking error": r["tracking_error"],
                                                "Sharpe": r["sharpe"],
                                                "Beta": r["beta"],
                                                "Alpha ann.": r["alpha"],
                                                "Max DD": r["max_drawdown"],
                                            })
                                        except Exception as exc:
                                            results_rows.append({
                                                **base_row,
                                                "Perf ann.": np.nan, "Vol ann.": np.nan,
                                                "Tracking error": np.nan, "Sharpe": np.nan,
                                                "Beta": np.nan, "Alpha ann.": np.nan,
                                                "Max DD": np.nan,
                                                "Erreur": str(exc),
                                            })
        progress.empty()
        status.empty()

        if not results_rows:
            st.warning("Aucun résultat — vérifie ta grille.")
        else:
            grid_df = pd.DataFrame(results_rows)
            st.session_state["grid_results"] = grid_df

    grid_df = st.session_state.get("grid_results")
    if grid_df is not None and not grid_df.empty:
        sort_map = {
            "Tracking error (min)": ("Tracking error", True),
            "Sharpe (max)": ("Sharpe", False),
            "Perf annualisée (max)": ("Perf ann.", False),
            "Alpha annualisé (max)": ("Alpha ann.", False),
            "Max drawdown (min en valeur absolue)": ("Max DD", False),
        }
        if objective == "Score composite (Sharpe − TE)":
            grid_df = grid_df.assign(**{"Score": grid_df["Sharpe"] - grid_df["Tracking error"]})
            grid_df = grid_df.sort_values("Score", ascending=False)
        else:
            col_sort, asc = sort_map[objective]
            grid_df = grid_df.sort_values(col_sort, ascending=asc, na_position="last")

        fmt_grid = grid_df.copy()
        for c in ["Perf ann.", "Vol ann.", "Tracking error", "Alpha ann.", "Max DD"]:
            if c in fmt_grid.columns:
                fmt_grid[c] = fmt_grid[c].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
        for c in ["Sharpe", "Beta", "Score"]:
            if c in fmt_grid.columns:
                fmt_grid[c] = fmt_grid[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
        for c in ["ADV min (FCFA)", "Capi flot. min (FCFA)", "Vol moyen min"]:
            if c in fmt_grid.columns:
                fmt_grid[c] = fmt_grid[c].map(lambda x: f"{int(x):,}".replace(",", " ") if pd.notna(x) else "-")

        st.markdown(f"**{len(grid_df)} configurations** — top 5 :")
        st.dataframe(fmt_grid.head(5), use_container_width=True, hide_index=True)

        st.markdown("**Résultats complets :**")
        st.dataframe(fmt_grid, use_container_width=True, hide_index=True)

        # Heatmap configurable
        st.markdown("**Heatmap** — choisir 2 dimensions à pivoter (méthode/rebal/seuils filtrés)")
        possible_dims = ["n_max", "Cap %", "% jours cotés min", "ADV min (FCFA)",
                         "Capi flot. min (FCFA)", "Vol moyen min"]
        # Garder uniquement les dimensions qui ont >1 valeur unique
        varying = [d for d in possible_dims if grid_df[d].nunique(dropna=True) > 1]
        if len(varying) >= 2:
            c1, c2, c3, c4 = st.columns(4)
            dim_x = c1.selectbox("Axe X", varying, index=min(1, len(varying) - 1), key="hm_x")
            dim_y = c2.selectbox("Axe Y", [d for d in varying if d != dim_x],
                                 index=0, key="hm_y")
            sel_method = c3.selectbox("Méthode (filtre)", grid_df["Méthode"].unique().tolist(), key="hm_m")
            sel_metric = c4.selectbox(
                "Métrique",
                ["Tracking error", "Sharpe", "Perf ann.", "Alpha ann.", "Max DD", "Vol ann.", "Univers liquide"],
                key="hm_metric",
            )
            sub = grid_df[grid_df["Méthode"] == sel_method].copy()
            if len(sub) >= 2 and dim_x in sub.columns and dim_y in sub.columns:
                pivot = sub.pivot_table(index=dim_y, columns=dim_x, values=sel_metric, aggfunc="mean")
                scale = "Blues" if sel_metric in {"Sharpe", "Perf ann.", "Alpha ann.", "Univers liquide"} else "Reds"
                fig_hm = px.imshow(
                    pivot, text_auto=".2f", aspect="auto", color_continuous_scale=scale,
                    title=f"{sel_metric} — {sel_method} ({dim_y} × {dim_x})",
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            else:
                st.info("Pas assez de points pour cette combinaison.")
        else:
            st.info("Heatmap indisponible : il faut au moins 2 dimensions ayant >1 valeur dans la grille.")

        # Diagnostic du filtre liquidité
        if grid_df["% jours cotés min"].nunique() > 1 or grid_df["ADV min (FCFA)"].nunique() > 1 \
                or grid_df["Capi flot. min (FCFA)"].nunique() > 1 or grid_df["Vol moyen min"].nunique() > 1:
            st.markdown("**Impact des filtres de liquidité sur la taille de l'univers**")
            liq_dims = [d for d in ["% jours cotés min", "ADV min (FCFA)", "Capi flot. min (FCFA)", "Vol moyen min"]
                        if grid_df[d].nunique() > 1]
            uni_view = grid_df.drop_duplicates(subset=liq_dims)[liq_dims + ["Univers liquide"]]
            uni_view = uni_view.sort_values("Univers liquide", ascending=False)
            st.dataframe(uni_view, use_container_width=True, hide_index=True)

        csv_grid = grid_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger les résultats (CSV)", csv_grid, "grid_search.csv", "text/csv")


with tab_data:
    st.subheader("Cours alignés")
    st.dataframe(prices[selected].tail(200), use_container_width=True)
    st.subheader("Pondérations")
    csv = poids_df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger les pondérations (CSV)", csv, "etf_weights.csv", "text/csv")
    st.subheader("Performance ETF (rendements quotidiens)")
    st.dataframe(
        pd.DataFrame({"ETF": etf_rets, bench_symbol: bench_rets_c}).tail(200),
        use_container_width=True,
    )


with tab_doc:
    st.markdown(
        r"""
# 📚 Documentation — ETF Builder BRVM

Cette app construit un **ETF synthétique** répliquant un indice BRVM à partir
des cours historiques scrappés. Elle te laisse explorer plusieurs méthodes de
pondération sous contraintes (liquidité, plafond par titre, rebalancement) et
les comparer sur métriques de risque/performance.

---

## 1. Univers et liquidité

### Univers candidat
- 47 actions cotées BRVM (cf. `brvm_tickers.py`).
- Filtrage optionnel par **secteur** (BRVM-CB, BRVM-SF, BRVM-EN, etc.).

### Filtres de liquidité
Seuils paramétrables, appliqués **avant** la sélection top-N :

| Filtre | Définition | Cas d'usage |
|---|---|---|
| **% jours cotés** | jours avec volume > 0 / total jours | exclut les titres dormants |
| **Volume moyen (titres)** | moyenne du nombre de titres échangés/jour | seuil d'activité brut |
| **ADV** (Average Daily Value, FCFA) | $\overline{V \times P}$ | impact de marché — référence pour pentes/clients institutionnels |
| **Capi totale** (FCFA) | $P_{last} \times N_{titres}$ | filtre les small caps |
| **Capi flottante** | $\text{Capi} \times \text{flottant\%}$ | proxy de la liquidité réelle pour un ETF |
| **Cours min (FCFA)** | dernier cours | évite les titres à très petit nominal |
| **Flottant min (%)** | part flottante | exigence ISC pour l'éligibilité indice |

L'option *"Exclure les titres sans donnée"* impose la disponibilité de la métrique :
décochée, un titre dont la fiche société n'expose pas le flottant passe le filtre.

### Sélection top-N
Parmi les titres qui passent les filtres, on retient les **N plus liquides** au sens
*nb de jours cotés* (proxy de couverture). Tu peux ensuite ajuster la sélection
manuellement (multiselect en sidebar).

---

## 2. Méthodes de pondération

Soit $N$ le nombre de titres retenus, $w_i$ le poids du titre $i$, $r_i$ son rendement
quotidien, $\sigma_i$ sa volatilité, $r_b$ le rendement du benchmark.

### 2.1 Équipondéré
$$w_i = \frac{1}{N}$$
Robuste, sans biais d'estimation ; performe bien quand les rendements sont peu
différenciés ou en absence de signal fiable.

### 2.2 Capi-pondéré (*market cap weighted*)
$$w_i = \frac{P_i \cdot N_i^{titres}}{\sum_j P_j \cdot N_j^{titres}}$$
Standard des indices boursiers (S&P 500, BRVM Composite). Concentre sur les
mégacaps.

### 2.3 Free-float pondéré
$$w_i = \frac{P_i \cdot N_i^{titres} \cdot f_i}{\sum_j P_j \cdot N_j^{titres} \cdot f_j}$$
où $f_i$ est le pourcentage flottant. Méthode des indices investissables modernes
(MSCI, BRVM 30) : on pondère sur ce qui est réellement échangeable.

### 2.4 Inverse de la volatilité (*risk parity light*)
$$w_i = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}$$
Sur-pondère les titres calmes, sous-pondère les volatils. Meilleure diversification
du risque mais ignore les corrélations.

### 2.5 Tracking Error minimum (vs indice)
Cherche le portefeuille long-only qui réplique le mieux l'indice :
$$\min_w \; \mathrm{TE}(w) = \mathrm{Std}\big(R w - r_b\big) \quad \text{s.c.} \quad \sum_i w_i = 1, \; 0 \le w_i \le c$$
où $c$ est le plafond par titre. Résolu par **SLSQP** (scipy). Utile pour
construire un ETF de réplication partielle (sampling) sur un univers liquide.

### 2.6 Sharpe maximal (*tangent portfolio*)
$$\max_w \; \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}} \quad \text{s.c.} \quad \sum_i w_i = 1, \; 0 \le w_i \le c$$
avec $\mu$ = moyenne annualisée des rendements, $\Sigma$ = matrice de covariance
annualisée. Sensible à l'estimation de $\mu$ ; à utiliser avec précaution
(préférer un horizon long).

### Plafond par titre $c$
- Pour les méthodes 2.1–2.4 : on **clipe** les poids à $c$ puis on renormalise.
- Pour 2.5–2.6 : la contrainte est intégrée à l'optimiseur (bornes SLSQP).

---

## 3. Backtest

### Drift entre rebalancements
Entre deux dates de rebalancement, les poids dérivent avec les rendements :
$$w_{i,t+1}^{drift} = \frac{w_{i,t}(1 + r_{i,t})}{\sum_j w_{j,t}(1 + r_{j,t})}$$
Aux dates pivot (mensuel/trimestriel/…), on **réinitialise** vers les poids cibles.
Mode "Buy & Hold" : aucun rebalancement, drift libre.

### Fréquences disponibles
Aucun · Mensuel (`ME`) · Trimestriel (`QE`) · Semestriel (`2QE`) · Annuel (`YE`).
Codes pandas modernes (post-2.2) — fin de période.

---

## 4. Métriques

| Métrique | Formule | Interprétation |
|---|---|---|
| **Perf annualisée** | $\left(\prod (1+r_t)\right)^{252/n} - 1$ | rendement moyen géométrique annualisé |
| **Volatilité annualisée** | $\sigma_r \cdot \sqrt{252}$ | dispersion annualisée |
| **Tracking Error** | $\mathrm{Std}(r_{ETF} - r_b) \cdot \sqrt{252}$ | écart de réplication ; un *bon ETF* < 1 % |
| **Beta** | $\mathrm{Cov}(r_{ETF}, r_b) / \mathrm{Var}(r_b)$ | sensibilité linéaire au benchmark |
| **Alpha annualisé** | $(\overline{r_{ETF}} - \beta \cdot \overline{r_b}) \cdot 252$ | sur-performance résiduelle (Jensen) |
| **Sharpe** | $(\mu_{ETF} - r_f) / \sigma_{ETF}$ | rémunération par unité de risque total |
| **Max Drawdown** | $\min_t \big( C_t / \max_{s \le t} C_s - 1 \big)$ | pire perte cumulée depuis un sommet |

$252$ = nombre de jours de trading par an (convention).

---

## 5. Grid Search

L'onglet *Grid Search* balaie le produit cartésien d'**allocations** × **filtres de liquidité** :

$$\text{Combos} = \underbrace{M \times N_{max} \times C \times R}_{\text{allocation}} \times \underbrace{S \times A \times F \times V}_{\text{liquidité}}$$

avec :
- **Allocation** : $M$ = méthodes, $N_{max}$ = tailles d'ETF, $C$ = plafonds par titre, $R$ = fréquences de rebalancement.
- **Liquidité** (optionnel — coches dans l'UI) : $S$ = % jours cotés min, $A$ = ADV min, $F$ = capi flottante min, $V$ = volume moyen min.
  Les seuils non balayés restent figés à leur valeur de la sidebar.

Pour chaque tuple de seuils de liquidité, l'app reconstruit l'univers admissible,
trie par jours cotés et applique le top-$N_{max}$. Le pré-calcul est cacheé en
mémoire pour ne pas recomputer le même filtre. La colonne *Univers liquide*
expose la taille de l'univers admissible pour chaque combo, et la colonne
*n titres ETF* le nombre de titres effectivement pris.

### Objectifs de classement
- **Tracking error (min)** — adapté aux ETF de réplication.
- **Sharpe (max)** — adapté aux ETF actifs / smart-beta.
- **Perf annualisée (max)** — peut sur-fitter, à interpréter avec prudence.
- **Alpha annualisé (max)** — recherche de génération d'alpha pur.
- **Max drawdown** (min en valeur absolue) — robustesse aux baisses.
- **Score composite Sharpe − TE** — compromis rendement/risque/réplication.

### Heatmap
Pivot des résultats sur (`n_max` × `Cap %`) pour une méthode + rebalancement
fixés. Permet de repérer visuellement les zones plates/pic.

---

## 6. Limites & avertissements

- **Pas de coûts de transaction** ni d'impact de marché (à ajouter pour un
  vrai PnL : ~0.7 % de frais BRVM aller-retour, et plus pour les small caps).
- **Pas de dividendes réinvestis dans la perf** (on travaille sur les cours nuls
  de coupons). Voir `data/dividendes/` pour enrichir.
- **Sharpe-max** est très sensible à $\mu$ ; sur un échantillon court il peut
  produire des allocations extrêmes même avec cap.
- **Tracking error in-sample** : l'optimiseur voit les rendements du benchmark
  utilisés pour le scoring → biais d'optimisme. Pour un usage réel, calibrer
  sur une fenêtre passée et tester sur une fenêtre out-of-sample (split temporel).
- Données BRVM : peu d'observations sur certains titres (liquidité faible),
  d'où l'importance des **filtres de liquidité**.
"""
    )
