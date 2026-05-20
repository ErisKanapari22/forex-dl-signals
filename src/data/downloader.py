# src/data/downloader.py
"""
Pull historical OHLCV bars from MetaTrader 5 and persist them to data/raw/.

MT5 connection lifecycle
------------------------
The ``MT5Connector`` context manager calls ``mt5.initialize()`` on entry and
``mt5.shutdown()`` on exit — even if an exception is raised inside the block.
Never call ``initialize`` / ``shutdown`` yourself outside this context manager;
dangling connections cause silent failures on subsequent calls.

History-depth limitation  ← READ THIS BEFORE DEBUGGING MISSING BARS
----------------------------------------------------------------------
MT5 only keeps bars in RAM that the terminal has already streamed from the
broker's server.  ``copy_rates_range()`` will silently return a *shorter*
result — no error, no exception — when history hasn't been loaded yet.

How to fix it:
  1. Open the MetaTrader 5 desktop terminal.
  2. Open a chart for the symbol/timeframe you need.
  3. Press Ctrl+← (scroll left) and hold until bars stop appearing.
     Alternatively: Tools → Options → Charts → "Max bars in history" = unlimited.
  4. Wait ~30 seconds for the terminal to finish fetching from the server.
  5. Re-run this script.

This project's ``_check_bar_count()`` function warns you automatically when
the returned count is suspiciously low.

Timestamp convention
---------------------
``copy_rates_range()`` returns a structured numpy array whose ``time`` field
is a Unix epoch (seconds since 1970-01-01 00:00 UTC).  We convert it to a
``pd.DatetimeIndex`` with ``utc=True``.  No broker-timezone arithmetic is
needed — the epoch is always UTC by definition.

Output layout
-------------
  data/raw/{SYMBOL}_{TIMEFRAME}.csv          — OHLCV bars
  data/raw/{SYMBOL}_{TIMEFRAME}_meta.json    — provenance sidecar
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import pandas as pd

from src.utils.io import DotDict, load_config

log = logging.getLogger(__name__)

# ── MT5 import guard ──────────────────────────────────────────────────────────
# MetaTrader5 is a Windows-only compiled extension.  Wrapping the import lets
# the module be imported on Linux CI runners (e.g., for unit tests that mock
# the download) without an ImportError blowing up the whole test suite.
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    mt5 = None          # type: ignore[assignment]
    _MT5_AVAILABLE = False


# ── Timeframe string → MT5 constant ──────────────────────────────────────────
def _build_tf_map() -> dict[str, int]:
    """Build mapping lazily so we don't crash at import time on Linux."""
    if not _MT5_AVAILABLE:
        return {}
    return {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }


# ── Context manager ───────────────────────────────────────────────────────────
@contextmanager
def MT5Connector(cfg: DotDict) -> Generator[None, None, None]:
    """
    Context manager for the MT5 connection lifecycle.

    Credentials (account / server / password) come from cfg.trading.mt5,
    which is populated from .env via load_config().  For read-only data pulls
    on a terminal that's already logged in, credentials are optional —
    ``mt5.initialize()`` with no arguments attaches to the running terminal.

    Usage::

        with MT5Connector(cfg):
            rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
    """
    if not _MT5_AVAILABLE:
        raise RuntimeError(
            "MetaTrader5 package not importable.  This package is Windows-only "
            "and requires the MT5 desktop terminal to be installed and running."
        )

    mt5_cfg  = cfg.get("trading", {}).get("mt5", {})
    account  = mt5_cfg.get("account")
    server   = mt5_cfg.get("server")
    password = mt5_cfg.get("password")

    # If credentials are present in config/env, pass them explicitly.
    # Otherwise, attach to the already-running, already-logged-in terminal.
    if account and server and password:
        ok = mt5.initialize(
            login=int(account),
            server=str(server),
            password=str(password),
        )
    else:
        ok = mt5.initialize()

    if not ok:
        err = mt5.last_error()
        raise ConnectionError(
            f"mt5.initialize() failed — error {err}.  "
            "Ensure MetaTrader 5 is running and logged into your demo account."
        )

    info = mt5.terminal_info()
    log.info(
        "MT5 connected | build=%s | connected=%s | community_account=%s",
        getattr(info, "build", "?"),
        getattr(info, "connected", "?"),
        getattr(info, "community_account", "?"),
    )

    try:
        yield
    finally:
        mt5.shutdown()
        log.info("MT5 shutdown complete.")


# ── Core download ─────────────────────────────────────────────────────────────
def download_rates(
    symbol: str,
    timeframe_str: str,
    utc_from: datetime,
    utc_to: datetime,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from MT5 for *symbol* in [*utc_from*, *utc_to*].

    Parameters
    ----------
    symbol        : MT5 symbol name, e.g. ``"EURUSD"``.
    timeframe_str : Config string, e.g. ``"H1"``.  Must be in _build_tf_map().
    utc_from      : Start of requested range (UTC).  May be tz-aware or naive.
    utc_to        : End   of requested range (UTC).  May be tz-aware or naive.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume, spread.
        Index: UTC-aware DatetimeIndex named ``"time"``.
        Sorted ascending.  No de-duplication (cleaner does that).

    Raises
    ------
    ValueError   : Unknown timeframe string.
    RuntimeError : MT5 returned zero bars (history not loaded or bad symbol).
    """
    tf_map = _build_tf_map()
    if timeframe_str not in tf_map:
        raise ValueError(
            f"Unknown timeframe '{timeframe_str}'.  "
            f"Supported values: {sorted(tf_map)}"
        )
    mt5_tf = tf_map[timeframe_str]

    # MT5 treats naive datetimes as UTC.  Strip tzinfo if present to keep the
    # API happy — passing tz-aware datetimes causes a TypeError in some builds.
    from_naive = utc_from.replace(tzinfo=None)
    to_naive   = utc_to.replace(tzinfo=None)

    log.info(
        "  Requesting %s %s | %s → %s",
        symbol, timeframe_str, from_naive.date(), to_naive.date(),
    )

    rates = mt5.copy_rates_range(symbol, mt5_tf, from_naive, to_naive)

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(
            f"MT5 returned no data for {symbol} {timeframe_str}.  "
            f"MT5 last error: {err}.  "
            "Common causes: symbol not in Market Watch, terminal not connected, "
            "or history not yet streamed (open chart and scroll left first)."
        )

    log.info("  Received %d raw bars.", len(rates))

    df = pd.DataFrame(rates)

    # Convert Unix epoch → UTC-aware DatetimeIndex
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").sort_index()

    # Rename tick_volume → volume; keep only OHLCV + spread
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    keep_cols = [c for c in ("open", "high", "low", "close", "volume", "spread")
                 if c in df.columns]
    return df[keep_cols].copy()


# ── Persistence helpers ───────────────────────────────────────────────────────
def raw_csv_path(raw_dir: Path, symbol: str, timeframe: str) -> Path:
    return raw_dir / f"{symbol}_{timeframe}.csv"


def raw_meta_path(raw_dir: Path, symbol: str, timeframe: str) -> Path:
    return raw_dir / f"{symbol}_{timeframe}_meta.json"


def save_raw(
    df: pd.DataFrame,
    raw_dir: Path,
    symbol: str,
    timeframe: str,
    utc_from: datetime,
    utc_to: datetime,
) -> Path:
    """Write df to CSV + provenance sidecar JSON.  Creates raw_dir if needed."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_csv_path(raw_dir, symbol, timeframe)
    df.to_csv(csv_path)
    log.info("  Saved raw CSV   → %s  (%d bars)", csv_path.name, len(df))

    meta = {
        "symbol":         symbol,
        "timeframe":      timeframe,
        "requested_from": utc_from.isoformat(),
        "requested_to":   utc_to.isoformat(),
        "actual_start":   df.index[0].isoformat(),
        "actual_end":     df.index[-1].isoformat(),
        "n_bars":         len(df),
        "downloaded_at":  datetime.now(timezone.utc).isoformat(),
    }
    meta_path = raw_meta_path(raw_dir, symbol, timeframe)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("  Saved metadata  → %s", meta_path.name)

    return csv_path


def load_raw(raw_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load a previously saved raw CSV.  Restores UTC-aware DatetimeIndex."""
    path = raw_csv_path(raw_dir, symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            "Run the downloader first:  python scripts/download_data.py"
        )
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    return df


# ── Bar-count sanity check ────────────────────────────────────────────────────
# Expected bars per trading week (Mon open → Fri close, no weekends).
# Used only for the warning heuristic — not a hard validation.
_EXPECTED_BARS_PER_WEEK: dict[str, float] = {
    "M1":  5 * 24 * 60,
    "M5":  5 * 24 * 12,
    "M15": 5 * 24 * 4,
    "M30": 5 * 24 * 2,
    "H1":  5 * 24,       # 120
    "H4":  5 * 6,        # 30
    "D1":  5,
    "W1":  1,
}

def _check_bar_count(
    df: pd.DataFrame,
    timeframe: str,
    utc_from: datetime,
    utc_to: datetime,
) -> None:
    """Warn when actual bar count is < 90% of expected.  Indicates history gap."""
    if timeframe not in _EXPECTED_BARS_PER_WEEK:
        return
    total_days = (utc_to - utc_from).days
    weeks      = total_days / 7.0
    # 90% threshold: accounts for holidays and session edges
    expected   = int(weeks * _EXPECTED_BARS_PER_WEEK[timeframe] * 0.90)
    actual     = len(df)

    if actual < expected:
        log.warning(
            "  ⚠  Bar count LOW for %s: got %d, expected ≥ %d. "
            "History may not be fully loaded in MT5 terminal. "
            "Open the %s chart, scroll left to %s, let bars populate, then re-run.",
            timeframe, actual, expected, timeframe, utc_from.date(),
        )
    else:
        log.info("  Bar count OK: %d (expected ≥ %d).", actual, expected)


# ── Public entry point ────────────────────────────────────────────────────────
def run_download(cfg: DotDict | None = None) -> dict[tuple[str, str], Path]:
    """
    Download all active instruments × all configured timeframes.

    Reads:  cfg.instruments.active, cfg.timeframes.primary/context,
            cfg.data.history_years, cfg.paths.data.raw

    Returns a dict mapping (symbol, timeframe) → saved CSV Path.
    Failed pairs are logged and skipped (partial results returned).
    """
    if cfg is None:
        cfg = load_config()

    history_years = cfg.data.history_years
    utc_to        = datetime.now(timezone.utc)
    utc_from      = utc_to - timedelta(days=365 * history_years)

    symbols    = list(cfg.instruments.active)
    timeframes = [cfg.timeframes.primary] + list(cfg.timeframes.context)
    raw_dir    = cfg.paths.data.raw          # already an absolute Path from io.py

    log.info(
        "Download plan: %d symbol(s) × %d timeframe(s) | "
        "%d years back from %s",
        len(symbols), len(timeframes), history_years, utc_to.date(),
    )

    results: dict[tuple[str, str], Path] = {}

    with MT5Connector(cfg):
        for symbol in symbols:
            for tf in timeframes:
                log.info("=== %s / %s ===", symbol, tf)
                try:
                    df   = download_rates(symbol, tf, utc_from, utc_to)
                    path = save_raw(df, raw_dir, symbol, tf, utc_from, utc_to)
                    _check_bar_count(df, tf, utc_from, utc_to)
                    results[(symbol, tf)] = path
                except Exception as exc:
                    log.error(
                        "Download FAILED for %s %s: %s", symbol, tf, exc,
                        exc_info=True,
                    )

    log.info("Download complete: %d / %d pairs saved.", len(results), len(symbols) * len(timeframes))
    return results