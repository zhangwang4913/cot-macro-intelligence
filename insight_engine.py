"""
LLM-powered insight engine.
Supports Claude (ANTHROPIC_API_KEY) and Gemini (GEMINI_API_KEY).
GEMINI_API_KEY takes priority; ANTHROPIC_API_KEY is fallback.

Prompts are structured around the professional COT decision framework:
  - COT as a positioning FILTER not a signal
  - 3-layer funnel: regime → positioning extreme → sizing/risk
  - Actionable trade thesis with explicit sizing guidance
"""

import hashlib
import logging
import os
from pathlib import Path

import database
from analytics import (
    MarketMetrics,
    PRIMARY_CATEGORY,
    COMMERCIAL_CATEGORY,
    filter_extremes,
    format_net,
    _positioning_state,
)

log = logging.getLogger("cot.insight")

CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# System prompts — professional COT framework
# ---------------------------------------------------------------------------

MARKET_SYSTEM_PROMPT = """You are a macro hedge fund analyst. Your job is to read CFTC COT positioning data and deliver a professional trade assessment — the same way an institutional PM would brief a morning meeting.

COT is a POSITIONING FILTER, not a buy/sell signal. It tells you how crowded a trade is and what size is appropriate given where everyone else is sitting.

Output format (use these exact headers):

**POSITIONING STATE**
State: [EXTREME LONG / MODERATE LONG / NEUTRAL / MODERATE SHORT / EXTREME SHORT]
Cite the 1Y and 3Y percentile ranks and z-scores. Note the COT Index reading.

**TRADE THESIS**
One clear sentence: "The thesis is [long/short] [asset] because [reason]."
If there is no viable thesis at current positioning, state why.

**SIZING GUIDANCE**
Based on positioning state, give explicit guidance:
- NEUTRAL (20–80th %ile): Full size. Normal ATR-based stops and targets.
- MODERATE (80–90th or 10–20th %ile): 50–75% size. Tighter stops. Reduced profit target.
- EXTREME (90th+ or sub-10th %ile): 0–25% size OR no new entry. De-risk existing positions. Note any options strategy.
- CONTRARIAN EXTREME (fading the crowd): 25–50% size. Price action confirmation required before entry.

**KEY RISKS**
What would invalidate the thesis? What catalyst could trigger rapid position unwind? Mention divergence signals if present.

Be direct. Cite specific percentile ranks, z-scores, and COT Index readings. No hedging language. Under 320 words."""


MACRO_SYSTEM_PROMPT = """You are a macro hedge fund PM running the 3-step professional COT decision process. Only markets at positioning extremes (beyond the 10th or 90th percentile) are provided.

**STEP 1: REGIME DETERMINATION**
Score at least 4 of these 5 asset class groups:
- Equities (S&P/Nasdaq/Russell): leveraged fund direction
- Bonds (2Y/5Y/10Y/30Y): leveraged fund direction (risk-off = net long)
- Gold: managed money direction (safe-haven bid = net long extreme)
- Commodities (oil/copper): managed money direction
- USD/FX: DXY specs + high-beta FX (AUD, EUR) direction

Determine regime: RISK-ON / RISK-OFF / MIXED / FRAGILE
State conviction level (low/medium/high) and the one factor that tips the balance.

**STEP 2: CROWDED TRADE RISKS**
Identify the 3 most crowded positions by absolute z-score. For each:
- Unwind scenario: what triggers it?
- Cascade risk: could it drag related assets?
- Any commercial divergence (smart money on the other side)?

**STEP 3: ACTIONABLE SETUPS**
For each high-conviction setup, provide exactly:
- Asset and direction (long/short)
- Positioning state and recommended size % (Neutral=100%, Mod=50-75%, Extreme=0-25%, Contrarian=25-50%)
- Entry trigger: what price action or fundamental event confirms the setup
- Invalidation: one specific condition that kills the trade

Professional rules embedded:
- Never initiate full size into an 85th+ percentile crowd
- Contrarian setups require price confirmation — patience is the edge
- Commercial divergence (informed money vs speculators) is a high-conviction input
- For equities, focus on leveraged funds + asset managers; dealer commercial signal is less reliable

Be direct. Cite specific percentile ranks, z-scores, and COT Index readings. Under 500 words."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_market_prompt(m: MarketMetrics) -> str:
    primary_cat = PRIMARY_CATEGORY.get(m.report_type, "managed_money")
    comm_cat    = COMMERCIAL_CATEGORY.get(m.report_type)
    cat  = m.categories.get(primary_cat)
    comm = m.categories.get(comm_cat) if comm_cat else None

    lines = [
        f"Market: {m.display_name} ({m.asset_class.upper()})",
        f"Report type: {m.report_type.upper()}  |  Latest as of: {m.latest_date} (Tuesday close, 3-day release lag)",
        f"Positioning State: {m.positioning_state}",
    ]

    if m.divergence_signal:
        lines.append(f"*** DIVERGENCE SIGNAL: {m.divergence_signal} — informed money vs speculators on opposite sides ***")
    if m.position_signal:
        lines.append(f"*** POSITION SIGNAL: {m.position_signal} — extreme position with weekly reversal ***")

    lines += ["", "=== PRIMARY SPECULATOR POSITIONING ==="]
    if cat:
        lines += [
            f"Category      : {primary_cat}",
            f"Net contracts : {format_net(cat.net_contracts)}",
            f"Net % of OI   : {cat.net_pct_oi:+.2f}%",
            f"WoW change    : {format_net(cat.wow_change)}",
            f"Open Interest : {cat.open_interest:,}",
            "",
            "Percentile Ranks (rolling lookbacks):",
            f"  1Y (52w)  : {cat.pct_1y:.1f}th pct  [z={cat.z_1y:+.2f}]  [COT Index: {cat.cot_index_1y:.0f}/100]",
            f"  3Y (156w) : {cat.pct_3y:.1f}th pct  [z={cat.z_3y:+.2f}]  [COT Index: {cat.cot_index_3y:.0f}/100]",
            f"  5Y (260w) : {cat.pct_5y:.1f}th pct  [z={cat.z_5y:+.2f}]",
            "",
            "Sizing framework: Neutral(20-80%)=full | Moderate(80-90%)=50-75% | Extreme(90%+)=0-25%",
        ]

    if comm:
        lines += [
            "",
            f"=== INFORMED MONEY ({comm_cat.upper()}) ===",
            f"  Net: {format_net(comm.net_contracts)}  ({'+' if comm.net_contracts >= 0 else ''}{comm.net_pct_oi:.2f}% of OI)",
            f"  WoW: {format_net(comm.wow_change)}",
        ]

    lines += ["", "=== ALL CATEGORIES ==="]
    for cat_name, c in m.categories.items():
        lines.append(
            f"  {cat_name:<14}: net={format_net(c.net_contracts)}  "
            f"OI%={c.net_pct_oi:+.2f}%  WoW={format_net(c.wow_change)}"
        )

    if m.alignment_score is not None:
        lines += [
            "",
            "=== TFF CONSENSUS (Lev Funds + Asset Managers) ===",
            f"  Alignment: {m.alignment_label}  (score: {m.alignment_score:+.4f})",
        ]

    return "\n".join(lines)


def _build_macro_prompt(extremes: dict[str, MarketMetrics]) -> str:
    lines = [
        f"Cross-asset EXTREME positioning snapshot — {len(extremes)} markets flagged.",
        "As-of: Tuesday close (CFTC 3-day release lag applies).",
        "",
        f"{'Market':<16} {'Class':<10} {'State':<16} {'1Y%':>5} {'3Y%':>5} {'1Yz':>6} {'CI1Y':>5} {'Net%OI':>7} {'WoW':>8} {'Signal':<12} {'Align':<14}",
        "-" * 110,
    ]
    for m in extremes.values():
        primary_cat = PRIMARY_CATEGORY.get(m.report_type, "managed_money")
        cat = m.categories.get(primary_cat)
        if not cat:
            continue
        sig   = m.position_signal or m.divergence_signal or "-"
        align = m.alignment_label or "-"
        lines.append(
            f"{m.display_name:<16} {m.asset_class:<10} {m.positioning_state:<16} "
            f"{cat.pct_1y:>5.1f} {cat.pct_3y:>5.1f} {cat.z_1y:>+6.2f} "
            f"{cat.cot_index_1y:>5.0f} {cat.net_pct_oi:>+7.2f}% "
            f"{format_net(cat.wow_change):>8} {sig:<12} {align:<14}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class InsightEngine:
    def __init__(self, db_path: Path):
        self.db_path = db_path

        gemini_key    = os.environ.get("GEMINI_API_KEY", "").strip()
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

        if gemini_key:
            from google import genai
            self._gemini   = genai.Client(api_key=gemini_key)
            self._provider = "gemini"
            log.info("Insight engine: Gemini (%s)", GEMINI_MODEL)
        elif anthropic_key:
            from anthropic import AsyncAnthropic
            self._client   = AsyncAnthropic(api_key=anthropic_key)
            self._provider = "claude"
            log.info("Insight engine: Claude (%s)", CLAUDE_MODEL)
        else:
            raise RuntimeError(
                "No LLM API key found. Set GEMINI_API_KEY or ANTHROPIC_API_KEY."
            )

    def _fingerprint(self, m: MarketMetrics) -> str:
        primary_cat = PRIMARY_CATEGORY.get(m.report_type, "managed_money")
        cat = m.categories.get(primary_cat)
        raw = f"{m.latest_date}|{cat.net_pct_oi:.3f}|{cat.pct_3y:.1f}|{m.positioning_state}" if cat else str(m.latest_date)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _macro_fingerprint(self, extremes: dict[str, MarketMetrics]) -> str:
        keys = "|".join(f"{c}:{m.latest_date}:{m.positioning_state}" for c, m in sorted(extremes.items()))
        return hashlib.md5(keys.encode()).hexdigest()[:12]

    async def _call_llm(self, system: str, user_msg: str, max_tokens: int) -> str:
        if self._provider == "claude":
            return await self._call_claude(system, user_msg, max_tokens)
        return await self._call_gemini(system, user_msg, max_tokens)

    async def _call_claude(self, system: str, user_msg: str, max_tokens: int) -> str:
        full = []
        async with self._client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            async for chunk in stream.text_stream:
                full.append(chunk)
        return "".join(full)

    async def _call_gemini(self, system: str, user_msg: str, max_tokens: int) -> str:
        from google.genai import types
        combined = f"{system}\n\n{user_msg}"
        response = self._gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=combined,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text or ""

    async def get_market_insight(
        self,
        market_code: str,
        metrics: MarketMetrics,
        force_refresh: bool = False,
    ) -> str:
        fingerprint = self._fingerprint(metrics)
        if not force_refresh:
            cached = database.get_cached_insight(self.db_path, market_code, fingerprint)
            if cached:
                return cached

        text = await self._call_llm(
            MARKET_SYSTEM_PROMPT, _build_market_prompt(metrics), max_tokens=1800
        )
        database.cache_insight(self.db_path, market_code, text, fingerprint)
        return text

    async def get_macro_insight(
        self,
        all_metrics: dict[str, MarketMetrics],
        force_refresh: bool = False,
    ) -> str:
        extremes = filter_extremes(all_metrics, threshold=10.0)
        if not extremes:
            return "No markets are currently at positioning extremes (beyond 10th or 90th percentile)."

        fingerprint = self._macro_fingerprint(extremes)
        if not force_refresh:
            cached = database.get_cached_insight(self.db_path, "__macro__", fingerprint)
            if cached:
                return cached

        text = await self._call_llm(
            MACRO_SYSTEM_PROMPT, _build_macro_prompt(extremes), max_tokens=2800
        )
        database.cache_insight(self.db_path, "__macro__", text, fingerprint)
        return text
