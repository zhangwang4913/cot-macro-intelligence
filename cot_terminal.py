"""
COT Macro Intelligence Terminal
Entry point and full Textual TUI.

Usage:
    py cot_terminal.py             # Launch TUI (loads from existing DB)
    py cot_terminal.py --backfill  # Download 2010-present, then launch TUI
    py cot_terminal.py --refresh   # Refresh current year data only, then launch

Mandate resolutions:
  #2  DuckDB concurrency  – All DB calls run inside @work(thread=True) workers.
                            The main asyncio thread never touches duckdb directly.
                            database.py enforces a threading.Lock on every open().
  #6  Specialist Board    – [M] macro insight produces the 6-section Committee
                            output; market detail shows alignment score for the
                            Bottleneck Architect and Contrarian roles explicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    DataTable,
    Footer,
    Label,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual import work

import analytics
import database
from analytics import (
    MarketMetrics,
    PRIMARY_CATEGORY,
    ASSET_CLASS_ORDER,
    format_net,
    pct_bar,
    zscore_to_style,
)
from data_pipeline import COTPipeline, TRACKED_MARKETS
from insight_engine import InsightEngine

load_dotenv()
log = logging.getLogger("cot.app")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

_BASE = Path(__file__).parent
DB_PATH   = Path(os.environ.get("COT_DB_PATH",   str(_BASE / "cot_data.db")))
DATA_DIR  = Path(os.environ.get("COT_DATA_DIR",  str(_BASE / "data")))
BACKFILL_FROM = int(os.environ.get("COT_BACKFILL_FROM", "2010"))


# ---------------------------------------------------------------------------
# Custom messages
# ---------------------------------------------------------------------------

class MarketSelected(Message):
    def __init__(self, market_code: str) -> None:
        super().__init__()
        self.market_code = market_code


# ---------------------------------------------------------------------------
# Sidebar widgets
# ---------------------------------------------------------------------------

class MarketItem(Widget):
    """Single clickable market row in sidebar."""

    DEFAULT_CSS = ""

    def __init__(self, market_code: str, display_name: str, **kwargs):
        super().__init__(**kwargs)
        self.market_code  = market_code
        self.display_name = display_name

    def render(self) -> str:
        return self.display_name

    def on_click(self) -> None:
        self.post_message(MarketSelected(self.market_code))


class MarketSidebar(ScrollableContainer):
    """Left sidebar: markets grouped by asset class."""

    def compose(self) -> ComposeResult:
        grouped: dict[str, list[tuple[str, dict]]] = {ac: [] for ac in ASSET_CLASS_ORDER}
        for code, meta in TRACKED_MARKETS.items():
            ac = meta.get("asset_class", "other")
            if ac in grouped:
                grouped[ac].append((code, meta))

        for asset_class in ASSET_CLASS_ORDER:
            markets = grouped.get(asset_class, [])
            if not markets:
                continue
            yield Label(asset_class.upper(), classes="asset-class-header")
            for code, meta in markets:
                safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", code)
                yield MarketItem(code, meta["display"], id=f"mi_{safe_id}")


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

class HeatmapTable(DataTable):
    """Color-coded z-score heatmap for all tracked markets."""

    COLUMNS = [
        ("Market",    16),
        ("Class",     10),
        ("1Y %ile",    7),
        ("3Y %ile",    7),
        ("5Y %ile",    7),
        ("1Y Z",       6),
        ("WoW Δ",      9),
        ("OI %",       8),
    ]

    def on_mount(self) -> None:
        self.cursor_type = "row"
        for label, width in self.COLUMNS:
            self.add_column(label, width=width)

    def refresh_data(self, all_metrics: dict[str, MarketMetrics]) -> None:
        self.clear()
        order = sorted(
            all_metrics.values(),
            key=lambda m: (ASSET_CLASS_ORDER.index(m.asset_class)
                           if m.asset_class in ASSET_CLASS_ORDER else 99,
                           m.display_name),
        )
        for m in order:
            pcat = PRIMARY_CATEGORY.get(m.report_type, "managed_money")
            cat  = m.categories.get(pcat)
            if cat is None:
                continue

            z = cat.z_3y
            style = zscore_to_style(z)

            def _t(val: str) -> Text:
                return Text(val, style=style)

            p1  = f"{cat.pct_1y:.0f}" if cat.pct_1y == cat.pct_1y else "—"
            p3  = f"{cat.pct_3y:.0f}" if cat.pct_3y == cat.pct_3y else "—"
            p5  = f"{cat.pct_5y:.0f}" if cat.pct_5y == cat.pct_5y else "—"
            z3  = f"{z:+.2f}"         if z == z           else "—"
            wow = format_net(cat.wow_change)
            oi  = f"{cat.net_pct_oi:+.1f}%"

            self.add_row(
                _t(m.display_name),
                Text(m.asset_class[:8], style="dim"),
                _t(p1), _t(p3), _t(p5), _t(z3),
                Text(wow, style="cyan" if cat.wow_change >= 0 else "magenta"),
                Text(oi,  style=style),
                key=m.market_code,
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            self.post_message(MarketSelected(str(event.row_key.value)))


# ---------------------------------------------------------------------------
# Detail panel
# ---------------------------------------------------------------------------

class DetailPanel(Static):
    """Full market breakdown shown in the Detail tab."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metrics: MarketMetrics | None = None

    def update_market(self, m: MarketMetrics) -> None:
        self._metrics = m
        self.refresh(layout=True)

    def render(self) -> str:
        m = self._metrics
        if m is None:
            return "[dim]Select a market from the sidebar.[/dim]"

        pcat = PRIMARY_CATEGORY.get(m.report_type, "managed_money")
        cat  = m.categories.get(pcat)
        lines: list[str] = []

        # Header
        date_str = str(m.latest_date) if m.latest_date else "—"
        lines += [
            f"[bold accent]{m.display_name}[/]  [{m.asset_class}]  "
            f"[dim]As-of: {date_str} (Tuesday close)[/]",
            "",
        ]

        if cat:
            # Key metrics row
            lines += [
                f"[bold]Net Position:[/] {format_net(cat.net_contracts)}   "
                f"[bold]OI %:[/] {cat.net_pct_oi:+.2f}%   "
                f"[bold]WoW Δ:[/] {format_net(cat.wow_change)}   "
                f"[bold]OI:[/] {cat.open_interest:,}",
                "",
                "[bold]Percentile Ranks  (Winsorized z-scores in brackets)[/]",
                f"  1Y  {pct_bar(cat.pct_1y)} {cat.pct_1y:5.1f}th   [z={cat.z_1y:+.2f}]",
                f"  3Y  {pct_bar(cat.pct_3y)} {cat.pct_3y:5.1f}th   [z={cat.z_3y:+.2f}]",
                f"  5Y  {pct_bar(cat.pct_5y)} {cat.pct_5y:5.1f}th   [z={cat.z_5y:+.2f}]",
                "",
            ]

        # Category breakdown table
        lines.append("[bold]Category Breakdown[/]")
        lines.append(f"  {'Category':<15} {'Long':>10} {'Short':>10} {'Net':>10} {'OI%':>8}")
        lines.append("  " + "─" * 55)
        for cat_name, c in m.categories.items():
            marker = "◀" if cat_name == pcat else " "
            lines.append(
                f"  {marker}{cat_name:<14} {c.long_contracts:>10,} {c.short_contracts:>10,} "
                f"{format_net(c.net_contracts):>10} {c.net_pct_oi:>+7.2f}%"
            )

        # Sparkline
        if cat:
            lines += [
                "",
                "[bold]Net Position History — 20W sparkline[/]",
                f"  {cat.sparkline}",
            ]

        # TFF alignment
        if m.alignment_score is not None:
            align_style = (
                "bright_green" if "LONG" in (m.alignment_label or "")
                else "bright_red" if "SHORT" in (m.alignment_label or "")
                else "yellow"
            )
            lines += [
                "",
                f"[bold]TFF Alignment (Lev Funds + Asset Mgrs):[/] "
                f"[{align_style}]{m.alignment_label}[/]  "
                f"score={m.alignment_score:+.4f}",
            ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Insight panel
# ---------------------------------------------------------------------------

class InsightPanel(Vertical):
    """Bottom panel showing Claude-generated analysis."""

    def compose(self) -> ComposeResult:
        yield Label("● INSIGHT   [I]=Market  [M]=Macro  [Shift+I]=Force refresh", id="insight-label")
        yield RichLog(id="insight-log", markup=True, highlight=True, wrap=True)

    def show_loading(self, msg: str = "Generating insight…") -> None:
        log_widget = self.query_one("#insight-log", RichLog)
        log_widget.clear()
        log_widget.write(f"[dim]{msg}[/dim]")

    def show_text(self, text: str) -> None:
        log_widget = self.query_one("#insight-log", RichLog)
        log_widget.clear()
        log_widget.write(text)

    def show_error(self, msg: str) -> None:
        log_widget = self.query_one("#insight-log", RichLog)
        log_widget.clear()
        log_widget.write(f"[bold red]ERROR:[/bold red] {msg}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class COTApp(App):
    CSS_PATH = Path(__file__).parent / "cot_terminal.tcss"

    BINDINGS = [
        Binding("i",       "market_insight",    "Market Insight"),
        Binding("I",       "market_insight_force", "Force Refresh Insight", show=False),
        Binding("m",       "macro_insight",     "Macro [Committee]"),
        Binding("r",       "refresh_data",      "Refresh"),
        Binding("b",       "backfill_data",     "Backfill"),
        Binding("q",       "quit",              "Quit"),
        Binding("ctrl+c",  "quit",              "Quit", show=False),
    ]

    # Reactive state
    selected_code: reactive[str | None]         = reactive(None)
    all_metrics:   reactive[dict]               = reactive({})
    last_update:   reactive[str]                = reactive("No data loaded")
    is_busy:       reactive[bool]               = reactive(False)

    def __init__(self, db_path: Path, data_dir: Path):
        super().__init__()
        self._db_path   = db_path
        self._data_dir  = data_dir
        self._engine: InsightEngine | None = None
        self._refresh_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        # Header bar
        with Horizontal(id="app-header"):
            yield Label("COT MACRO INTELLIGENCE TERMINAL", id="header-title")
            yield Label(self.last_update, id="header-status")

        # Main split
        with Horizontal(id="main-area"):
            with Container(id="sidebar"):
                yield MarketSidebar()

            with Vertical(id="right-panel"):
                with TabbedContent():
                    with TabPane("Heatmap", id="tab-heatmap"):
                        yield HeatmapTable(id="heatmap-table")
                    with TabPane("Detail", id="tab-detail"):
                        with ScrollableContainer(id="detail-container"):
                            yield DetailPanel(id="detail-panel")

                yield InsightPanel(id="insight-panel")

        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        database.init_db(self._db_path)
        self._init_engine()
        self._load_metrics()

    def _init_engine(self) -> None:
        try:
            self._engine = InsightEngine(self._db_path)
        except RuntimeError as e:
            log.warning("Insight engine unavailable: %s", e)
            self._engine = None

    # ------------------------------------------------------------------
    # Reactive watchers
    # ------------------------------------------------------------------

    def watch_last_update(self, value: str) -> None:
        try:
            self.query_one("#header-status", Label).update(value)
        except Exception:
            pass

    def watch_all_metrics(self, metrics: dict) -> None:
        try:
            self.query_one(HeatmapTable).refresh_data(metrics)
        except Exception:
            pass
        # Refresh detail panel if a market is selected
        if self.selected_code and self.selected_code in metrics:
            self._refresh_detail(self.selected_code)

    def watch_selected_code(self, code: str | None) -> None:
        # Highlight sidebar item
        for item in self.query(MarketItem):
            item.remove_class("selected")
        if code:
            try:
                safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", code)
                self.query_one(f"#mi_{safe_id}", MarketItem).add_class("selected")
            except Exception:
                pass
            self._refresh_detail(code)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @on(MarketSelected)
    def on_market_selected(self, event: MarketSelected) -> None:
        self.selected_code = event.market_code

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_market_insight(self) -> None:
        self._trigger_market_insight(force=False)

    def action_market_insight_force(self) -> None:
        self._trigger_market_insight(force=True)

    def action_macro_insight(self) -> None:
        if not self._engine:
            self.query_one(InsightPanel).show_error("ANTHROPIC_API_KEY not configured.")
            return
        self._fetch_macro_insight()

    def action_refresh_data(self) -> None:
        self._run_refresh()

    def action_backfill_data(self) -> None:
        self._run_backfill()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_detail(self, code: str) -> None:
        m = self.all_metrics.get(code)
        if m:
            try:
                self.query_one("#detail-panel", DetailPanel).update_market(m)
            except Exception:
                pass

    def _trigger_market_insight(self, force: bool) -> None:
        if not self._engine:
            self.query_one(InsightPanel).show_error("ANTHROPIC_API_KEY not configured.")
            return
        if not self.selected_code:
            self.query_one(InsightPanel).show_error("Select a market first (sidebar or heatmap).")
            return
        m = self.all_metrics.get(self.selected_code)
        if m is None:
            self.query_one(InsightPanel).show_error("No data loaded for this market.")
            return
        self._fetch_market_insight(self.selected_code, m, force)

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    @work(thread=True, exclusive=False)
    def _load_metrics(self) -> None:
        """Load all analytics from DB in a thread worker."""
        try:
            metrics = analytics.compute_all_metrics(self._db_path)
            max_date = database.get_max_date(self._db_path)
            status = f"Last data: {max_date}" if max_date else "DB empty — press B to backfill"
            self.call_from_thread(setattr, self, "all_metrics",   metrics)
            self.call_from_thread(setattr, self, "last_update",   status)
        except Exception as exc:
            self.call_from_thread(
                setattr, self, "last_update", f"Load error: {exc}"
            )

    @work(thread=True, exclusive=True)
    def _run_refresh(self) -> None:
        """Refresh current-year CFTC data."""
        self.call_from_thread(setattr, self, "last_update", "Refreshing…")
        try:
            pipe = COTPipeline(self._data_dir, self._db_path)
            rows, latest = pipe.refresh(
                progress_cb=lambda msg: self.call_from_thread(
                    setattr, self, "last_update", msg
                )
            )
            self.call_from_thread(
                setattr, self, "last_update", f"Refreshed: +{rows} rows | latest {latest}"
            )
            self._load_metrics()
        except Exception as exc:
            self.call_from_thread(setattr, self, "last_update", f"Refresh error: {exc}")

    @work(thread=True, exclusive=True)
    def _run_backfill(self) -> None:
        """Download full history (2010–present)."""
        self.call_from_thread(setattr, self, "last_update", "Backfilling 2010–present…")
        try:
            pipe  = COTPipeline(self._data_dir, self._db_path)
            total = pipe.backfill(
                from_year=BACKFILL_FROM,
                progress_cb=lambda msg: self.call_from_thread(
                    setattr, self, "last_update", msg
                ),
            )
            self.call_from_thread(
                setattr, self, "last_update", f"Backfill complete: {total:,} rows"
            )
            self._load_metrics()
        except Exception as exc:
            self.call_from_thread(setattr, self, "last_update", f"Backfill error: {exc}")

    @work(exclusive=True)
    async def _fetch_market_insight(
        self, code: str, m: MarketMetrics, force: bool
    ) -> None:
        panel = self.query_one(InsightPanel)
        panel.show_loading(f"Generating insight for {m.display_name}…")
        try:
            text = await self._engine.get_market_insight(code, m, force_refresh=force)
            panel.show_text(text)
        except Exception as exc:
            panel.show_error(str(exc))

    @work(exclusive=True)
    async def _fetch_macro_insight(self) -> None:
        panel = self.query_one(InsightPanel)
        panel.show_loading("Running cross-asset macro analysis for Investment Committee…")
        try:
            text = await self._engine.get_macro_insight(self.all_metrics)
            panel.show_text(text)
        except Exception as exc:
            panel.show_error(str(exc))


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _run_backfill_headless(db_path: Path, data_dir: Path) -> None:
    """Backfill without launching TUI. Called with --backfill flag."""
    database.init_db(db_path)
    pipe = COTPipeline(data_dir, db_path)
    print(f"Starting backfill from {BACKFILL_FROM}…")
    total = pipe.backfill(
        from_year=BACKFILL_FROM,
        progress_cb=lambda msg: print(f"  {msg}"),
    )
    print(f"Done: {total:,} rows inserted.")


def _run_refresh_headless(db_path: Path, data_dir: Path) -> None:
    database.init_db(db_path)
    pipe = COTPipeline(data_dir, db_path)
    print("Refreshing current year…")
    rows, latest = pipe.refresh(progress_cb=lambda msg: print(f"  {msg}"))
    print(f"Done: +{rows} rows | latest date: {latest}")


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="COT Macro Intelligence Terminal")
    parser.add_argument("--backfill", action="store_true", help="Download history then quit")
    parser.add_argument("--refresh",  action="store_true", help="Refresh current year then quit")
    parser.add_argument("--db",   default=str(DB_PATH),   help="DuckDB path")
    parser.add_argument("--data", default=str(DATA_DIR),  help="Data cache directory")
    args = parser.parse_args()

    db_path   = Path(args.db)
    data_dir  = Path(args.data)

    if args.backfill:
        _run_backfill_headless(db_path, data_dir)
        return
    if args.refresh:
        _run_refresh_headless(db_path, data_dir)
        return

    app = COTApp(db_path=db_path, data_dir=data_dir)
    app.run()


if __name__ == "__main__":
    main()
