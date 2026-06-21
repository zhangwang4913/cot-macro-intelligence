"""
Analytics engine — COT positioning metrics.

Signals computed per market:
  - Percentile ranks (1Y/3Y/5Y) and Winsorized z-scores  [primary display]
  - COT Index (0-100 range-normalized, per industry standard)
  - Positioning state (NEUTRAL / MODERATE LONG|SHORT / EXTREME LONG|SHORT)
  - Commercial divergence (informed money vs speculators on opposite sides)
  - Position signal (UNWIND | SQUEEZE when extreme + weekly reversal)
  - TFF alignment score (lev_money + asset_mgr consensus)
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import mstats

import database
from data_pipeline import TRACKED_MARKETS

log = logging.getLogger("cot.analytics")

PRIMARY_CATEGORY = {
    "disagg": "managed_money",
    "tff":    "lev_money",
}

# Per professional framework: for disagg markets, producers have fundamental edge.
# For TFF, dealers vs lev money (less reliable for equities, valid for FX/rates).
COMMERCIAL_CATEGORY = {
    "disagg": "prod_merc",
    "tff":    "dealer",
}

SPECULATOR_CATEGORY = {
    "disagg": "managed_money",
    "tff":    "lev_money",
}

ASSET_CLASS_ORDER = ["commodity", "fx", "equity", "rates", "crypto"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CategoryMetrics:
    category: str
    long_contracts: int
    short_contracts: int
    net_contracts: int
    open_interest: int
    net_pct_oi: float
    wow_change: int
    pct_1y: float
    pct_3y: float
    pct_5y: float
    z_1y: float
    z_3y: float
    z_5y: float
    sparkline: str
    cot_index_1y: float = 50.0   # 0-100 range-normalized (industry standard)
    cot_index_3y: float = 50.0


@dataclass
class MarketMetrics:
    market_code: str
    display_name: str
    asset_class: str
    report_type: str
    latest_date: Optional[date]
    categories: dict[str, CategoryMetrics] = field(default_factory=dict)
    alignment_score: Optional[float] = None
    alignment_label: Optional[str]  = None
    positioning_state: str = "NEUTRAL"       # NEUTRAL / MODERATE LONG|SHORT / EXTREME LONG|SHORT
    divergence_signal: Optional[str] = None  # e.g. "PROD SHORT / MM LONG"
    position_signal: Optional[str] = None    # "UNWIND" | "SQUEEZE" | None


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------

def _winsorized_zscore(series: pd.Series, current: float) -> float:
    if len(series) < 4:
        return float("nan")
    w = mstats.winsorize(series.values.astype(float), limits=(0.01, 0.01))
    mu  = float(np.nanmean(w))
    std = float(np.nanstd(w, ddof=1))
    if std == 0 or np.isnan(std):
        return 0.0
    return round((current - mu) / std, 3)


def _percentile_rank(series: pd.Series, current: float) -> float:
    if series.empty:
        return float("nan")
    return round(float((series <= current).mean() * 100), 1)


def _cot_index(series: pd.Series, current: float, lookback: int) -> float:
    """
    Industry-standard COT Index: 0 = period low, 100 = period high.
    Formula: (current - period_min) / (period_max - period_min) * 100
    """
    seg = series.iloc[:lookback].dropna() if len(series) >= lookback else series.dropna()
    if len(seg) < 2:
        return 50.0
    mn, mx = float(seg.min()), float(seg.max())
    if mx == mn:
        return 50.0
    return round(max(0.0, min(100.0, (current - mn) / (mx - mn) * 100)), 1)


def _compute_stats(
    history: pd.DataFrame,
    current_net_pct: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """
    Returns (pct_1y, pct_3y, pct_5y, z_1y, z_3y, z_5y, cot_idx_1y, cot_idx_3y).
    history is sorted DESC (newest first), column net_pct_oi.
    """
    s = history["net_pct_oi"].dropna().reset_index(drop=True)

    def _window(w: int):
        seg = s.iloc[:w]
        if len(seg) < 4:
            return float("nan"), float("nan")
        pct = _percentile_rank(seg, current_net_pct)
        z   = _winsorized_zscore(seg, current_net_pct)
        return pct, z

    p1, z1 = _window(52)
    p3, z3 = _window(156)
    p5, z5 = _window(260)
    ci1 = _cot_index(s, current_net_pct, 52)
    ci3 = _cot_index(s, current_net_pct, 156)
    return p1, p3, p5, z1, z3, z5, ci1, ci3


def _make_sparkline(series: pd.Series, width: int = 20) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    vals = series.dropna().iloc[::-1].values[-width:]
    if len(vals) == 0:
        return "─" * width
    mn, mx = vals.min(), vals.max()
    if mx - mn == 0:
        return "▄" * len(vals)
    normalised = ((vals - mn) / (mx - mn) * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in normalised)


def _alignment_score(lev_net: int, am_net: int, oi: int) -> tuple[float, str]:
    if oi == 0:
        return 0.0, "N/A"
    score = round((lev_net + am_net) / (2 * oi), 4)
    if score > 0.05:
        label = "ALIGNED LONG"
    elif score < -0.05:
        label = "ALIGNED SHORT"
    else:
        label = "DIVERGING"
    return score, label


# ---------------------------------------------------------------------------
# New signal helpers
# ---------------------------------------------------------------------------

def _positioning_state(pct_1y: float) -> str:
    """
    Maps 1Y percentile rank to the professional positioning state framework.
    Drives position sizing guidance in AI insights.
    """
    if pct_1y != pct_1y:   # NaN
        return "NEUTRAL"
    if pct_1y >= 90: return "EXTREME LONG"
    if pct_1y >= 80: return "MODERATE LONG"
    if pct_1y <= 10: return "EXTREME SHORT"
    if pct_1y <= 20: return "MODERATE SHORT"
    return "NEUTRAL"


def _commercial_divergence(cat_metrics: dict, report_type: str) -> Optional[str]:
    """
    Detects when informed money (commercials/dealers) and speculators
    are positioned in opposite directions — a classic COT divergence setup.

    Disagg: producers (fundamental knowledge) vs managed money.
    TFF: dealers vs leveraged funds.
    Note: Per professional framework, this signal is less reliable for equity
    index futures where dealers are market-makers, not fundamentalists.
    """
    comm_key = COMMERCIAL_CATEGORY.get(report_type)
    spec_key = SPECULATOR_CATEGORY.get(report_type)
    if not comm_key or not spec_key:
        return None
    comm = cat_metrics.get(comm_key)
    spec = cat_metrics.get(spec_key)
    if not comm or not spec:
        return None
    comm_long = comm.net_contracts > 0
    spec_long = spec.net_contracts > 0
    if comm_long == spec_long:
        return None   # Same direction — no divergence
    cs = "LONG" if comm_long else "SHORT"
    ss = "LONG" if spec_long else "SHORT"
    label = "PROD" if report_type == "disagg" else "DEAL"
    slabel = "MM" if report_type == "disagg" else "LEV"
    return f"{label} {cs} / {slabel} {ss}"


def _position_signal(cat: CategoryMetrics) -> Optional[str]:
    """
    Detects early unwind or short-squeeze:
    UNWIND  — position at extreme long (85th+) but specs reduced ≥0.8% OI this week
    SQUEEZE — position at extreme short (15th-) but specs added ≥0.8% OI this week
    """
    if cat.open_interest == 0:
        return None
    wow_pct = abs(cat.wow_change) / cat.open_interest * 100
    if cat.pct_1y >= 85 and cat.wow_change < 0 and wow_pct >= 0.8:
        return "UNWIND"
    if cat.pct_1y <= 15 and cat.wow_change > 0 and wow_pct >= 0.8:
        return "SQUEEZE"
    return None


# ---------------------------------------------------------------------------
# Per-market computation
# ---------------------------------------------------------------------------

def compute_market_metrics(
    db_path,
    market_code: str,
    meta: dict,
) -> Optional[MarketMetrics]:
    report_type = meta["report_type"]
    if report_type == "disagg":
        categories_needed = ["prod_merc", "managed_money", "swap", "other_rept", "nonrept"]
    else:
        categories_needed = ["dealer", "lev_money", "asset_mgr", "other_rept"]

    cat_metrics: dict[str, CategoryMetrics] = {}
    latest_date = None

    for cat in categories_needed:
        hist = database.get_history(db_path, market_code, cat, weeks=260)
        if hist.empty:
            continue

        hist = hist.sort_values("report_date", ascending=False).reset_index(drop=True)
        row0 = hist.iloc[0]

        if latest_date is None:
            try:
                latest_date = row0["report_date"].date() if hasattr(row0["report_date"], "date") else row0["report_date"]
            except Exception:
                latest_date = None

        lng = int(row0["long_contracts"])
        sht = int(row0["short_contracts"])
        net = int(row0["net_contracts"])
        oi  = int(row0["open_interest"])
        pct = float(row0["net_pct_oi"]) if row0["net_pct_oi"] is not None else (net / oi * 100 if oi else 0.0)

        wow = 0
        if len(hist) >= 2:
            wow = net - int(hist.iloc[1]["net_contracts"])

        p1, p3, p5, z1, z3, z5, ci1, ci3 = _compute_stats(hist, pct)
        spark = _make_sparkline(hist["net_contracts"])

        cm = CategoryMetrics(
            category=cat,
            long_contracts=lng,
            short_contracts=sht,
            net_contracts=net,
            open_interest=oi,
            net_pct_oi=round(pct, 3),
            wow_change=wow,
            pct_1y=p1, pct_3y=p3, pct_5y=p5,
            z_1y=z1, z_3y=z3, z_5y=z5,
            sparkline=spark,
            cot_index_1y=ci1,
            cot_index_3y=ci3,
        )
        cat_metrics[cat] = cm

    if not cat_metrics:
        return None

    # Determine primary category signals
    primary_key = PRIMARY_CATEGORY.get(report_type, "managed_money")
    primary     = cat_metrics.get(primary_key)

    pos_state  = _positioning_state(primary.pct_1y if primary else float("nan"))
    divergence = _commercial_divergence(cat_metrics, report_type)
    pos_signal = _position_signal(primary) if primary else None

    m = MarketMetrics(
        market_code=market_code,
        display_name=meta["display"],
        asset_class=meta["asset_class"],
        report_type=report_type,
        latest_date=latest_date,
        categories=cat_metrics,
        positioning_state=pos_state,
        divergence_signal=divergence,
        position_signal=pos_signal,
    )

    # TFF alignment score
    if report_type == "tff":
        lev = cat_metrics.get("lev_money")
        am  = cat_metrics.get("asset_mgr")
        if lev and am:
            oi_ref = lev.open_interest or 1
            m.alignment_score, m.alignment_label = _alignment_score(
                lev.net_contracts, am.net_contracts, oi_ref
            )

    return m


def compute_all_metrics(db_path) -> dict[str, MarketMetrics]:
    result = {}
    for code, meta in TRACKED_MARKETS.items():
        try:
            m = compute_market_metrics(db_path, code, meta)
            if m is not None:
                result[code] = m
        except Exception as exc:
            log.warning("Metrics failed for %s: %s", code, exc)
    return result


# ---------------------------------------------------------------------------
# LLM context filter
# ---------------------------------------------------------------------------

def filter_extremes(
    all_metrics: dict[str, MarketMetrics],
    threshold: float = 10.0,
) -> dict[str, MarketMetrics]:
    hi = 100.0 - threshold
    out = {}
    for code, m in all_metrics.items():
        primary_cat = PRIMARY_CATEGORY.get(m.report_type)
        cat = m.categories.get(primary_cat or "")
        if cat is None:
            continue
        pcts = [p for p in (cat.pct_1y, cat.pct_3y, cat.pct_5y) if p == p]
        if any(p <= threshold or p >= hi for p in pcts):
            out[code] = m
    return out


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def zscore_to_style(z: float) -> str:
    if z != z: return "dim"
    if z > 2.0:  return "bright_red"
    if z > 1.0:  return "red"
    if z > 0.5:  return "yellow"
    if z > -0.5: return "white"
    if z > -1.0: return "cyan"
    if z > -2.0: return "green"
    return "bright_green"


def pct_bar(pct: float, width: int = 10) -> str:
    if pct != pct:
        return "─" * width
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def format_net(n: int) -> str:
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:+.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:+.1f}K"
    return f"{n:+d}"
