"""
LLM-powered insight engine.
Supports Claude (ANTHROPIC_API_KEY) and Gemini (GEMINI_API_KEY).
Provider is auto-detected from whichever env var is present.
"""

import hashlib
import logging
import os
from pathlib import Path

import database
from analytics import (
    MarketMetrics,
    PRIMARY_CATEGORY,
    filter_extremes,
    format_net,
)

log = logging.getLogger("cot.insight")

CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

MARKET_SYSTEM_PROMPT = """You are an expert macro analyst specializing in CFTC Commitment of Traders data.
Analyze the positioning data and provide professional insights. Focus on:
1. Current positioning extreme vs history
2. Momentum vs mean-reversion setup
3. Alignment between commercial hedgers and speculators
4. Risk/reward implications for the trade
Be concise, professional, and actionable. No more than 300 words."""

MACRO_SYSTEM_PROMPT = """You are a macro strategist analyzing cross-asset positioning from CFTC COT data.
Only markets at positioning extremes (beyond the 10th or 90th percentile on any lookback) are provided.

Provide a concise cross-asset macro analysis covering:
1. Current macro regime (Risk-On / Risk-Off / Neutral / Fragile) and conviction level
2. The 3 most crowded trades by z-score — what is the unwind risk for each?
3. The 2-3 best risk/reward setups where positioning supports a high-conviction trade
4. Any notable divergences between 1Y and 5Y percentile ranks (short-term vs secular dislocation)

Be direct, cite specific percentile ranks and z-scores. No hedging language. Under 400 words."""

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_market_prompt(m: MarketMetrics) -> str:
    primary_cat = PRIMARY_CATEGORY.get(m.report_type, "managed_money")
    cat = m.categories.get(primary_cat)
    lines = [
        f"Market: {m.display_name} ({m.asset_class.upper()})",
        f"Report type: {m.report_type.upper()}",
        f"Latest data as of: {m.latest_date} [Tuesday close - report released following Friday]",
        "",
        "=== PRIMARY SPECULATOR POSITIONING ===",
    ]
    if cat:
        lines += [
            f"Category: {primary_cat}",
            f"Net contracts : {format_net(cat.net_contracts)}",
            f"Net % of OI   : {cat.net_pct_oi:+.2f}%",
            f"WoW change    : {format_net(cat.wow_change)}",
            f"Open Interest : {cat.open_interest:,}",
            "",
            "Percentile Ranks (rolling, robust to rollover noise):",
            f"  1Y (52-week) : {cat.pct_1y:.1f}th percentile  [z={cat.z_1y:+.2f}]",
            f"  3Y (156-week): {cat.pct_3y:.1f}th percentile  [z={cat.z_3y:+.2f}]",
            f"  5Y (260-week): {cat.pct_5y:.1f}th percentile  [z={cat.z_5y:+.2f}]",
        ]

    lines += ["", "=== ALL CATEGORIES ==="]
    for cat_name, c in m.categories.items():
        lines.append(
            f"  {cat_name:<14}: net={format_net(c.net_contracts)}  "
            f"OI%={c.net_pct_oi:+.2f}%  "
            f"WoW={format_net(c.wow_change)}"
        )

    if m.alignment_score is not None:
        lines += [
            "",
            "=== TFF ALIGNMENT ===",
            f"Lev Funds + Asset Mgrs alignment: {m.alignment_label}",
            f"Alignment score: {m.alignment_score:+.4f}",
        ]

    return "\n".join(lines)


def _build_macro_prompt(extremes: dict[str, MarketMetrics]) -> str:
    lines = [
        f"Cross-asset EXTREME positioning summary ({len(extremes)} markets flagged).",
        "All values are as-of Tuesday close (CFTC release lag = 3 days).",
        "",
        f"{'Market':<16} {'Class':<10} {'1Y%':>5} {'3Y%':>5} {'5Y%':>5} "
        f"{'1Yz':>6} {'Net%OI':>7} {'WoW':>8} {'Align':<16}",
        "-" * 84,
    ]
    for m in extremes.values():
        primary_cat = PRIMARY_CATEGORY.get(m.report_type, "managed_money")
        cat = m.categories.get(primary_cat)
        if not cat:
            continue
        align = m.alignment_label or "-"
        lines.append(
            f"{m.display_name:<16} {m.asset_class:<10} "
            f"{cat.pct_1y:>5.1f} {cat.pct_3y:>5.1f} {cat.pct_5y:>5.1f} "
            f"{cat.z_1y:>+6.2f} {cat.net_pct_oi:>+7.2f}% "
            f"{format_net(cat.wow_change):>8} {align:<16}"
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
                "No LLM API key found. Set ANTHROPIC_API_KEY or GEMINI_API_KEY."
            )

    # ── fingerprinting ────────────────────────────────────────────────────

    def _fingerprint(self, m: MarketMetrics) -> str:
        primary_cat = PRIMARY_CATEGORY.get(m.report_type, "managed_money")
        cat = m.categories.get(primary_cat)
        raw = f"{m.latest_date}|{cat.net_pct_oi:.3f}|{cat.pct_3y:.1f}" if cat else str(m.latest_date)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _macro_fingerprint(self, extremes: dict[str, MarketMetrics]) -> str:
        keys = "|".join(f"{c}:{m.latest_date}" for c, m in sorted(extremes.items()))
        return hashlib.md5(keys.encode()).hexdigest()[:12]

    # ── LLM calls ─────────────────────────────────────────────────────────

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

    # ── Public API ────────────────────────────────────────────────────────

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
            MARKET_SYSTEM_PROMPT, _build_market_prompt(metrics), max_tokens=1500
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
            return "No markets are currently at positioning extremes (>90th or <10th percentile)."

        fingerprint = self._macro_fingerprint(extremes)
        if not force_refresh:
            cached = database.get_cached_insight(self.db_path, "__macro__", fingerprint)
            if cached:
                return cached

        text = await self._call_llm(
            MACRO_SYSTEM_PROMPT, _build_macro_prompt(extremes), max_tokens=2500
        )
        database.cache_insight(self.db_path, "__macro__", text, fingerprint)
        return text
