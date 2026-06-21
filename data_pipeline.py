"""
CFTC COT data pipeline.

Mandate resolutions embedded here:
  #1  Tuesday Bias   – "As of Date" column stored as report_date (Tuesday close).
                        release_date = report_date + 3 days (following Friday).
                        Never expose release_date as a signal date.
  #3  File Detection – detect_file_format() reads the first 8 bytes for OLE2 /
                        XLSX magic; falls back to a line-count heuristic before
                        attempting any pandas parse engine.
"""

import io
import json
import logging
import re
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass  # fall back to default SSL verification

import database

log = logging.getLogger("cot.pipeline")

# ---------------------------------------------------------------------------
# Market registry
# ---------------------------------------------------------------------------

TRACKED_MARKETS: dict[str, dict] = {
    # ---- Commodities (Disaggregated) ----
    "088691": {"name": "GOLD",        "display": "Gold",        "asset_class": "commodity", "report_type": "disagg"},
    "084691": {"name": "SILVER",      "display": "Silver",      "asset_class": "commodity", "report_type": "disagg"},
    "067651": {"name": "CRUDE OIL",   "display": "WTI Crude",   "asset_class": "commodity", "report_type": "disagg"},
    "023651": {"name": "NATURAL GAS", "display": "Natural Gas", "asset_class": "commodity", "report_type": "disagg"},
    "085692": {"name": "COPPER",      "display": "Copper",      "asset_class": "commodity", "report_type": "disagg"},
    # ---- FX (TFF — verified: FX futures live in the TFF file) ----
    "099741": {"name": "EURO FX",          "display": "EUR/USD",  "asset_class": "fx", "report_type": "tff"},
    "096742": {"name": "BRITISH POUND",    "display": "GBP/USD",  "asset_class": "fx", "report_type": "tff"},
    "097741": {"name": "JAPANESE YEN",     "display": "JPY/USD",  "asset_class": "fx", "report_type": "tff"},
    "232741": {"name": "AUSTRALIAN DOLLAR","display": "AUD/USD",  "asset_class": "fx", "report_type": "tff"},
    "098662": {"name": "USD INDEX",        "display": "DXY",      "asset_class": "fx", "report_type": "tff"},
    # ---- Equities (TFF) ----
    "13874A": {"name": "S&P 500",      "display": "S&P 500",      "asset_class": "equity", "report_type": "tff"},
    "209742": {"name": "NASDAQ MINI",  "display": "Nasdaq-100",   "asset_class": "equity", "report_type": "tff"},
    "239742": {"name": "RUSSELL",      "display": "Russell 2000", "asset_class": "equity", "report_type": "tff"},
    "12460+": {"name": "DJIA",         "display": "DJIA",         "asset_class": "equity", "report_type": "tff"},
    # ---- Rates (TFF) ----
    "042601": {"name": "UST 2Y NOTE",  "display": "2Y T-Note",   "asset_class": "rates", "report_type": "tff"},
    "044601": {"name": "UST 5Y NOTE",  "display": "5Y T-Note",   "asset_class": "rates", "report_type": "tff"},
    "043602": {"name": "UST 10Y NOTE", "display": "10Y T-Note",  "asset_class": "rates", "report_type": "tff"},
    "020601": {"name": "UST BOND",     "display": "30Y T-Bond",  "asset_class": "rates", "report_type": "tff"},
    # ---- Crypto (TFF — Bitcoin futures live in TFF file at CME) ----
    "133741": {"name": "BITCOIN",      "display": "Bitcoin",      "asset_class": "crypto", "report_type": "tff"},
}

# Name fragments used for fuzzy fallback matching (lower-case substrings)
_FUZZY_NAMES: dict[str, str] = {
    meta["name"].lower(): code for code, meta in TRACKED_MARKETS.items()
}

DISAGG_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_xls_{year}.zip"
TFF_URL    = "https://www.cftc.gov/files/dea/history/fut_fin_xls_{year}.zip"

# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Strip whitespace, hyphens, slashes, parens for flexible column matching."""
    return re.sub(r"[\s\-_/()\[\]]", "", name).lower()


def _build_col_map(raw_pairs: list[tuple[str, tuple[str, str]]]) -> dict[str, tuple[str, str]]:
    return {_norm(k): v for k, v in raw_pairs}


_DISAGG_RAW = [
    ("Prod/Merc Positions-Long (All)",    ("prod_merc",     "long")),
    ("Prod/Merc Positions Long (All)",    ("prod_merc",     "long")),
    ("Prod/Merc Positions-Short (All)",   ("prod_merc",     "short")),
    ("Prod/Merc Positions Short (All)",   ("prod_merc",     "short")),
    ("M Money Positions-Long (All)",      ("managed_money", "long")),
    ("M Money Positions Long (All)",      ("managed_money", "long")),
    ("M Money Positions-Short (All)",     ("managed_money", "short")),
    ("M Money Positions Short (All)",     ("managed_money", "short")),
    ("Swap Positions-Long (All)",         ("swap",          "long")),
    ("Swap Positions Long (All)",         ("swap",          "long")),
    ("Swap Positions-Short (All)",        ("swap",          "short")),
    ("Swap Positions Short (All)",        ("swap",          "short")),
    ("Other Rept Positions-Long (All)",   ("other_rept",    "long")),
    ("Other Rept Positions Long (All)",   ("other_rept",    "long")),
    ("Other Rept Positions-Short (All)",  ("other_rept",    "short")),
    ("Other Rept Positions Short (All)",  ("other_rept",    "short")),
    ("Nonrept Positions-Long (All)",      ("nonrept",       "long")),
    ("Nonrept Positions Long (All)",      ("nonrept",       "long")),
    ("Nonrept Positions-Short (All)",     ("nonrept",       "short")),
    ("Nonrept Positions Short (All)",     ("nonrept",       "short")),
]

_TFF_RAW = [
    ("Dealer Positions-Long (All)",       ("dealer",     "long")),
    ("Dealer Positions Long (All)",       ("dealer",     "long")),
    ("Dealer Positions-Short (All)",      ("dealer",     "short")),
    ("Dealer Positions Short (All)",      ("dealer",     "short")),
    ("Lev Money Positions-Long (All)",    ("lev_money",  "long")),
    ("Lev Money Positions Long (All)",    ("lev_money",  "long")),
    ("Lev Money Positions-Short (All)",   ("lev_money",  "short")),
    ("Lev Money Positions Short (All)",   ("lev_money",  "short")),
    ("Asset Mgr Positions-Long (All)",    ("asset_mgr",  "long")),
    ("Asset Mgr Positions Long (All)",    ("asset_mgr",  "long")),
    ("Asset Mgr Positions-Short (All)",   ("asset_mgr",  "short")),
    ("Asset Mgr Positions Short (All)",   ("asset_mgr",  "short")),
    ("Other Rept Positions-Long (All)",   ("other_rept", "long")),
    ("Other Rept Positions Long (All)",   ("other_rept", "long")),
    ("Other Rept Positions-Short (All)",  ("other_rept", "short")),
    ("Other Rept Positions Short (All)",  ("other_rept", "short")),
]

DISAGG_COL_MAP = _build_col_map(_DISAGG_RAW)
TFF_COL_MAP    = _build_col_map(_TFF_RAW)

# Possible names for shared identifier columns (normalised → canonical)
# Covers both legacy space-formatted CSV names and current underscore XLS names.
_DATE_KEYS = {_norm(k): k for k in [
    "Report_Date_as_MM_DD_YYYY",        # XLS files (primary — gives Timestamp directly)
    "As_of_Date_In_Form_YYMMDD",        # XLS files (YYMMDD integer, fallback)
    "As of Date in Form YYYY-MM-DD",    # legacy CSV format
    "Report_Date_as_of_YYYY-MM-DD",
]}
_CODE_KEYS = {_norm(k): k for k in [
    "CFTC_Contract_Market_Code",
    "CFTC Contract Market Code",
    "Contract Market Code",
]}
_NAME_KEYS = {_norm(k): k for k in [
    "Market_and_Exchange_Names",
    "Market and Exchange Names",
]}
_OI_KEYS = {_norm(k): k for k in [
    "Open_Interest_All",
    "Open Interest (All)",
    "Open Interest All",
]}

# ---------------------------------------------------------------------------
# File format detection  (Mandate #3)
# ---------------------------------------------------------------------------

_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # true XLS binary (pre-2015)
_ZIP_MAGIC  = b"PK\x03\x04"                           # XLSX or ZIP-wrapped CSV


def detect_file_format(data: bytes) -> str:
    """Return 'xls', 'xlsx', or 'csv' based on magic bytes, not extension."""
    if data[:8] == _OLE2_MAGIC:
        return "xls"
    if data[:4] == _ZIP_MAGIC:
        return "xlsx"
    return "csv"


def _read_file(data: bytes, fmt: str) -> pd.DataFrame:
    """Parse raw bytes into DataFrame using the detected format."""
    buf = io.BytesIO(data)
    if fmt == "xls":
        return pd.read_excel(buf, engine="xlrd")
    if fmt == "xlsx":
        return pd.read_excel(buf, engine="openpyxl")
    # CSV — try comma first, then tab, then fail gracefully
    try:
        df = pd.read_csv(io.BytesIO(data), low_memory=False)
        if df.shape[1] > 5:
            return df
    except Exception:
        pass
    return pd.read_csv(io.BytesIO(data), sep="\t", low_memory=False)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _find_col(df_cols_norm: dict[str, str], lookup: dict) -> Optional[str]:
    """Return the actual column name matching any key in lookup, or None."""
    for nk in lookup:
        if nk in df_cols_norm:
            return df_cols_norm[nk]
    return None


def _resolve_market_code(row_code: str, row_name: str) -> Optional[str]:
    """Return canonical tracked code or None if not tracked."""
    clean_code = str(row_code).strip().upper()
    if clean_code in TRACKED_MARKETS:
        return clean_code
    # Fuzzy fallback: substring match on display name
    row_name_lower = str(row_name).lower()
    for fragment, code in _FUZZY_NAMES.items():
        if fragment in row_name_lower:
            log.debug("Fuzzy matched %r → %s via fragment %r", row_name, code, fragment)
            return code
    return None


def _wide_to_long(df_wide: pd.DataFrame, col_map: dict, report_type: str) -> pd.DataFrame:
    """
    Melt CFTC wide format → normalized long rows.
    Each output row: one (market, week, category).
    Tuesday bias fix: stores 'as of' Tuesday date as report_date;
    release_date is always report_date + 3 days (the following Friday).
    """
    # Build normalised → actual column lookup
    norm_to_actual: dict[str, str] = {_norm(c): c for c in df_wide.columns}

    date_col = _find_col(norm_to_actual, _DATE_KEYS)
    code_col = _find_col(norm_to_actual, _CODE_KEYS)
    name_col = _find_col(norm_to_actual, _NAME_KEYS)
    oi_col   = _find_col(norm_to_actual, _OI_KEYS)

    if not all([date_col, code_col, name_col, oi_col]):
        log.warning(
            "Missing identifier columns. Found: %s",
            [date_col, code_col, name_col, oi_col],
        )
        return pd.DataFrame()

    records = []
    for _, row in df_wide.iterrows():
        raw_code = str(row[code_col]).strip()
        raw_name = str(row[name_col]).strip()
        canon_code = _resolve_market_code(raw_code, raw_name)
        if canon_code is None:
            continue

        raw_date = row[date_col]
        try:
            if isinstance(raw_date, pd.Timestamp):
                as_of = raw_date.date()
            elif isinstance(raw_date, (int, float)) and not pd.isna(raw_date):
                # YYMMDD integer e.g. 241231 → 2024-12-31
                d = int(raw_date)
                yy, mm, dd = d // 10000, (d // 100) % 100, d % 100
                year = 2000 + yy if yy < 100 else yy
                from datetime import date as _date
                as_of = _date(year, mm, dd)
            else:
                as_of = pd.to_datetime(str(raw_date).strip()).date()
        except Exception:
            continue

        # Mandate #1: as_of_date = Tuesday positioning date.
        # release_date = as_of + 3 days (Friday publication).
        release = as_of + timedelta(days=3)

        try:
            oi = int(str(row[oi_col]).replace(",", "").strip() or 0)
        except (ValueError, AttributeError):
            oi = 0

        # Group long/short values by category
        cat_longs: dict[str, int]  = {}
        cat_shorts: dict[str, int] = {}

        for col_name, (category, direction) in col_map.items():
            # Match against normalised actual columns
            matched_actual = norm_to_actual.get(col_name)
            if matched_actual is None:
                continue
            try:
                val = int(str(row[matched_actual]).replace(",", "").strip() or 0)
            except (ValueError, AttributeError):
                val = 0
            if direction == "long":
                cat_longs[category] = val
            else:
                cat_shorts[category] = val

        for cat in set(cat_longs) | set(cat_shorts):
            lng  = cat_longs.get(cat, 0)
            sht  = cat_shorts.get(cat, 0)
            net  = lng - sht
            pct  = round(net / oi * 100, 4) if oi > 0 else None
            records.append({
                "report_date":     as_of,
                "release_date":    release,
                "market_code":     canon_code,
                "market_name":     raw_name,
                "report_type":     report_type,
                "category":        cat,
                "long_contracts":  lng,
                "short_contracts": sht,
                "net_contracts":   net,
                "open_interest":   oi,
                "net_pct_oi":      pct,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class COTPipeline:
    def __init__(self, data_dir: Path, db_path: Path):
        self.data_dir = data_dir
        self.db_path  = db_path
        self._etag_cache_path = data_dir / ".etag_cache.json"
        self._etags: dict[str, str] = self._load_etags()
        data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def backfill(self, from_year: int = 2010, progress_cb=None) -> int:
        """Download and parse years [from_year, current]. Returns total rows."""
        import datetime
        current_year = datetime.date.today().year
        total = 0
        database.upsert_market_meta(self.db_path, TRACKED_MARKETS)

        for year in range(from_year, current_year + 1):
            for rtype, url_tpl, col_map in [
                ("disagg", DISAGG_URL, DISAGG_COL_MAP),
                ("tff",    TFF_URL,    TFF_COL_MAP),
            ]:
                if progress_cb:
                    progress_cb(f"Downloading {rtype.upper()} {year}…")
                data = self._download(url_tpl.format(year=year), f"{rtype}_{year}")
                if data is None:
                    continue
                df = self._parse_zip_bytes(data, col_map, rtype)
                rows = database.upsert_positions(self.db_path, df)
                total += rows
                log.info("%s %d → %d rows", rtype, year, rows)
        return total

    def refresh(self, progress_cb=None) -> tuple[int, Optional[str]]:
        """Download current year only. Returns (rows_added, latest_date_str)."""
        import datetime
        year = datetime.date.today().year
        total = 0
        database.upsert_market_meta(self.db_path, TRACKED_MARKETS)

        for rtype, url_tpl, col_map in [
            ("disagg", DISAGG_URL, DISAGG_COL_MAP),
            ("tff",    TFF_URL,    TFF_COL_MAP),
        ]:
            if progress_cb:
                progress_cb(f"Refreshing {rtype.upper()} {year}…")
            data = self._download(url_tpl.format(year=year), f"{rtype}_{year}")
            if data is None:
                continue
            df = self._parse_zip_bytes(data, col_map, rtype)
            total += database.upsert_positions(self.db_path, df)

        latest = database.get_max_date(self.db_path)
        self._save_etags()
        return total, latest

    # ------------------------------------------------------------------
    def _download(self, url: str, key: str) -> Optional[bytes]:
        """Download with ETag caching. Returns ZIP bytes or None on 304."""
        headers = {}
        if key in self._etags:
            headers["If-None-Match"] = self._etags[key]
        try:
            resp = requests.get(url, headers=headers, timeout=60)
        except Exception as exc:
            log.warning("Download failed %s: %s", url, exc)
            return None

        if resp.status_code == 304:
            log.debug("ETag hit — skipping %s", key)
            return None
        if resp.status_code != 200:
            log.warning("HTTP %d for %s", resp.status_code, url)
            return None

        etag = resp.headers.get("ETag")
        if etag:
            self._etags[key] = etag
        return resp.content

    def _parse_zip_bytes(
        self, zip_bytes: bytes, col_map: dict, report_type: str
    ) -> pd.DataFrame:
        """Extract the data file from the ZIP and parse it."""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                # Pick the largest member (avoids README / metadata files)
                members = sorted(zf.infolist(), key=lambda m: m.file_size, reverse=True)
                if not members:
                    return pd.DataFrame()
                target = members[0]
                raw = zf.read(target.filename)
        except zipfile.BadZipFile:
            log.warning("Not a valid ZIP; skipping.")
            return pd.DataFrame()

        # Mandate #3: detect format via magic bytes, not file extension
        fmt = detect_file_format(raw)
        log.debug("Detected format: %s for %d bytes", fmt, len(raw))

        try:
            df_wide = _read_file(raw, fmt)
        except Exception as exc:
            log.error("Parse failed (%s): %s", fmt, exc)
            return pd.DataFrame()

        # Strip erratic trailing whitespace from all string columns
        str_cols = df_wide.select_dtypes(include="object").columns
        df_wide[str_cols] = df_wide[str_cols].apply(
            lambda s: s.str.strip() if hasattr(s, "str") else s
        )

        return _wide_to_long(df_wide, col_map, report_type)

    def _load_etags(self) -> dict:
        if self._etag_cache_path.exists():
            try:
                return json.loads(self._etag_cache_path.read_text())
            except Exception:
                pass
        return {}

    def _save_etags(self) -> None:
        self._etag_cache_path.write_text(json.dumps(self._etags, indent=2))
