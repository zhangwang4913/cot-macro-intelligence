"""
DuckDB persistence layer.

Connection strategy (Mandate #2):
- All connections are short-lived: open → operate → close.
- A module-level threading.Lock gates every open() call so no two
  connections ever coexist, preventing DuckDB's hard write-lock error.
- TUI read paths pass read_only=True as an explicit safety flag even
  though the lock already guarantees exclusivity.
- All callers must run inside @work(thread=True) workers — never on
  Textual's main asyncio thread — so the blocking Lock.acquire() cannot
  stall the event loop.
"""

import threading
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

log = logging.getLogger("cot.db")

_DB_LOCK = threading.Lock()

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS cot_positions (
        report_date     DATE    NOT NULL,
        release_date    DATE,               -- audit only; do not use for signal timing
        market_code     VARCHAR NOT NULL,
        market_name     VARCHAR NOT NULL,
        report_type     VARCHAR NOT NULL,   -- 'disagg' | 'tff'
        category        VARCHAR NOT NULL,
        long_contracts  BIGINT  NOT NULL DEFAULT 0,
        short_contracts BIGINT  NOT NULL DEFAULT 0,
        net_contracts   BIGINT  NOT NULL DEFAULT 0,
        open_interest   BIGINT  NOT NULL DEFAULT 0,
        net_pct_oi      DOUBLE,
        PRIMARY KEY (report_date, market_code, report_type, category)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pos_lookup
        ON cot_positions (market_code, category, report_date DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS markets_meta (
        market_code  VARCHAR PRIMARY KEY,
        market_name  VARCHAR,
        display_name VARCHAR NOT NULL,
        asset_class  VARCHAR NOT NULL,
        report_type  VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS insight_cache (
        cache_key        VARCHAR   PRIMARY KEY,
        generated_at     TIMESTAMP NOT NULL,
        insight_text     VARCHAR   NOT NULL,
        data_fingerprint VARCHAR   NOT NULL
    )
    """,
]


@contextmanager
def _conn(db_path: Path, read_only: bool = False):
    with _DB_LOCK:
        c = duckdb.connect(str(db_path), read_only=read_only)
        try:
            yield c
        finally:
            c.close()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _conn(db_path, read_only=False) as c:
        for stmt in _SCHEMA:
            c.execute(stmt.strip())
    log.info("DB ready: %s", db_path)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def upsert_positions(db_path: Path, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with _conn(db_path, read_only=False) as c:
        c.register("_stg", df)
        c.execute("""
            INSERT INTO cot_positions
                SELECT * FROM _stg
            ON CONFLICT (report_date, market_code, report_type, category)
            DO UPDATE SET
                release_date    = EXCLUDED.release_date,
                market_name     = EXCLUDED.market_name,
                long_contracts  = EXCLUDED.long_contracts,
                short_contracts = EXCLUDED.short_contracts,
                net_contracts   = EXCLUDED.net_contracts,
                open_interest   = EXCLUDED.open_interest,
                net_pct_oi      = EXCLUDED.net_pct_oi
        """)
        c.unregister("_stg")
    return len(df)


def upsert_market_meta(db_path: Path, meta: dict) -> None:
    rows = [
        (code, m.get("name", ""), m["display"], m["asset_class"], m["report_type"])
        for code, m in meta.items()
    ]
    with _conn(db_path, read_only=False) as c:
        for row in rows:
            c.execute("""
                INSERT INTO markets_meta VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (market_code) DO UPDATE SET
                    display_name = EXCLUDED.display_name,
                    asset_class  = EXCLUDED.asset_class,
                    report_type  = EXCLUDED.report_type
            """, list(row))


def cache_insight(db_path: Path, key: str, text: str, fingerprint: str) -> None:
    with _conn(db_path, read_only=False) as c:
        c.execute("""
            INSERT INTO insight_cache VALUES (?, NOW(), ?, ?)
            ON CONFLICT (cache_key) DO UPDATE SET
                generated_at     = NOW(),
                insight_text     = EXCLUDED.insight_text,
                data_fingerprint = EXCLUDED.data_fingerprint
        """, [key, text, fingerprint])


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_history(
    db_path: Path,
    market_code: str,
    category: str,
    weeks: int = 260,
) -> pd.DataFrame:
    with _conn(db_path, read_only=True) as c:
        return c.execute("""
            SELECT report_date, net_contracts, open_interest, net_pct_oi,
                   long_contracts, short_contracts
            FROM cot_positions
            WHERE market_code = ? AND category = ?
            ORDER BY report_date DESC
            LIMIT ?
        """, [market_code, category, weeks]).df()


def get_all_latest(db_path: Path) -> pd.DataFrame:
    with _conn(db_path, read_only=True) as c:
        return c.execute("""
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY market_code, category
                           ORDER BY report_date DESC
                       ) AS rn
                FROM cot_positions
            )
            SELECT * EXCLUDE (rn) FROM ranked WHERE rn = 1
        """).df()


def get_max_date(db_path: Path) -> Optional[str]:
    with _conn(db_path, read_only=True) as c:
        row = c.execute("SELECT MAX(report_date) FROM cot_positions").fetchone()
    return str(row[0]) if row and row[0] else None


def get_db_stats(db_path: Path) -> dict:
    with _conn(db_path, read_only=True) as c:
        rows = c.execute("SELECT COUNT(*) FROM cot_positions").fetchone()[0]
        markets = c.execute(
            "SELECT COUNT(DISTINCT market_code) FROM cot_positions"
        ).fetchone()[0]
        max_d = c.execute("SELECT MAX(report_date) FROM cot_positions").fetchone()[0]
        min_d = c.execute("SELECT MIN(report_date) FROM cot_positions").fetchone()[0]
    return {"rows": rows, "markets": markets, "max_date": max_d, "min_date": min_d}


def get_cached_insight(db_path: Path, key: str, fingerprint: str) -> Optional[str]:
    with _conn(db_path, read_only=True) as c:
        row = c.execute("""
            SELECT insight_text FROM insight_cache
            WHERE cache_key = ? AND data_fingerprint = ?
        """, [key, fingerprint]).fetchone()
    return row[0] if row else None
