"""
Analytics engine.

Mandate resolutions:
  #4  Rollover Noise   – net_pct_oi is Winsorized at the 1st/99th percentile
                          before z-score computation. Percentile ranks are the
                          PRIMARY display signal and are inherently rollover-safe
                          because extreme OI drops shift ALL values in that window
                          proportionally, preserving relative rank.
  #5  LLM Optimization – filter_extremes() returns only markets where ANY
                          lookback window is at or beyond the 10th/90th pct
                          threshold. Only those go to the Insight Engine.
  #6  Specialist Board  – PRIMARY_CATEGORY and SPECIALIST_MAP encode which
                          metric maps to which committee role.
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

# Which category is the main "speculator" view per report type
PRIMARY_CATEGORY = {
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
    wow_change: int       # week-over-week Δnet
    pct_1y: float         # 0-100 percentile rank vs 52 weeks
    pct_3y: float
    pct_5y: float
    z_1y: float           # Winsorized z-score vs 52 weeks
    z_3y: float
    z_5y: float
    sparkline: str        # 20-char Unicode block art


@dataclass
class MarketMetrics:
    market_code: str
    display_name: str
    asset_class: str
    report_type: str
    latest_date: Optional[date]
    categories: dict[str, CategoryMetrics] = field(default_factory=dict)
    alignment_score: Optional[float] = None   # TFF only: -1.0 to +1.0
    alignment_label: Optional[str]  = None    # "ALIGNED LONG" | "ALIGNED SHORT" | "DIVERGING"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _winsorized_zscore(series: pd.Series, current: float) -> float:
    """
    Mandate #4: Winsorize the lookback window at the 1st/99th percentile
    before computing mean/std. This prevents single rollover week outliers
    from inflating the denominator and generating fake extreme z-scores.
    """
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


def _compute_stats(
    history: pd.DataFrame,
    current_net_pct: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Return (pct_1y, pct_3y, pct_5y, z_1y, z_3y, z_5y).
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
    return p1, p3, p5, z1, z3, z5


def _make_sparkline(series: pd.Series, width: int = 20) -> str:
    """
    ASCII sparkline using Unicode block elements ▁–█ (U+2581–U+2588).
    series: net_contracts, newest-first. We reverse to chronological order.
    """
    blocks = " ▁▂▃▄▅▆▇█"
    vals = series.dropna().iloc[::-1].values[-width:]
    if len(vals) == 0:
        return "─" * width
    mn, mx = vals.min(), vals.max()
    rng = mx - mn
    if rng == 0:
        return "▄" * len(vals)
    normalised = ((vals - mn) / rng * (len(blocks) - 1)).round().astype(int)
    return "".join(blocks[i] for i in normalised)


def _alignment_score(
    lev_net: int,
    am_net: int,
    oi: int,
) -> tuple[float, str]:
    if oi == 0:
        return 0.0, "N/A"
    score = round((lev_net + am_net) / (2 * oi), 4)
    if score > 0.05:
        label = "ALIGNED LONG ↑"
    elif score < -0.05:
        label = "ALIGNED SHORT ↓"
    else:
        label = "DIVERGING  ↔"
    return score, label


# ---------------------------------------------------------------------------
# Per-market computation
# ---------------------------------------------------------------------------

def compute_market_metrics(
    db_path,
    market_code: str,
    meta: dict,
) -> Optional[MarketMetrics]:
    report_type = meta["report_type"]
    categories_needed: list[str]
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

        lng  = int(row0["long_contracts"])
        sht  = int(row0["short_contracts"])
        net  = int(row0["net_contracts"])
        oi   = int(row0["open_interest"])
        pct  = float(row0["net_pct_oi"]) if row0["net_pct_oi"] is not None else (net / oi * 100 if oi else 0.0)

        wow = 0
        if len(hist) >= 2:
            wow = net - int(hist.iloc[1]["net_contracts"])

        p1, p3, p5, z1, z3, z5 = _compute_stats(hist, pct)
        spark = _make_sparkline(hist["net_contracts"])

        cat_metrics[cat] = CategoryMetrics(
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
        )

    if not cat_metrics:
        return None

    m = MarketMetrics(
        market_code=market_code,
        display_name=meta["display"],
        asset_class=meta["asset_class"],
        report_type=report_type,
        latest_date=latest_date,
        categories=cat_metrics,
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
    """Compute MarketMetrics for every tracked market. Run in thread worker."""
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
# LLM context filter  (Mandate #5)
# ---------------------------------------------------------------------------

def filter_extremes(
    all_metrics: dict[str, MarketMetrics],
    threshold: float = 10.0,
) -> dict[str, MarketMetrics]:
    """
    Return only markets where the primary speculator category has ANY lookback
    at or beyond the threshold percentile (default 10th / 90th).
    Prevents narrative dilution by feeding Claude only the actionable signals.
    """
    hi = 100.0 - threshold
    out = {}
    for code, m in all_metrics.items():
        primary_cat = PRIMARY_CATEGORY.get(m.report_type)
        cat = m.categories.get(primary_cat or "")
        if cat is None:
            continue
        pcts = [p for p in (cat.pct_1y, cat.pct_3y, cat.pct_5y) if not (p != p)]  # NaN-safe
        if any(p <= threshold or p >= hi for p in pcts):
            out[code] = m
    return out


# ---------------------------------------------------------------------------
# Formatting helpers for UI
# ---------------------------------------------------------------------------

def zscore_to_style(z: float) -> str:
    if z != z:  # NaN
        return "dim"
    if z > 2.0:   return "bright_red"
    if z > 1.0:   return "red"
    if z > 0.5:   return "yellow"
    if z > -0.5:  return "white"
    if z > -1.0:  return "cyan"
    if z > -2.0:  return "green"
    return "bright_green"


def pct_bar(pct: float, width: int = 10) -> str:
    """Render a text bar: [████░░░░░░] 82%"""
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
