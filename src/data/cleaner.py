# src/data/cleaner.py
"""
Load raw OHLCV bars and produce a clean, aligned DataFrame.

Canonical bar-timestamp convention
------------------------------------
Every bar in this project is labelled by its **open time**, expressed in UTC.
The implicit close time is ``open_time + timeframe_duration``.

For a broker running on UTC+2 (e.g., most ECN demo servers):
  - A D1 bar whose server clock reads midnight 2024-01-08 00:00 (UTC+2)
    has a UTC open timestamp of 2024-01-07 22:00 UTC (Sunday evening).
    This bar represents the **Monday** trading session.
  - We store and process this as-is.  All downstream code must be aware that
    a D1 timestamp at 22:00 UTC is the open of the *next calendar day's* session.
  - MT5 returns these as Unix epochs so no tz conversion is needed — we simply
    decode the epoch to a UTC-aware datetime.

No look-ahead guarantee
------------------------
Steps are ordered so that only **past** information is ever used to fill gaps:

  1. Dedup (keep last)         ← no future data, just taking the latest quote
  2. Drop weekends             ← structural remove, time-direction neutral
  3. Sort ascending            ← guard against unsorted input
  4. Reindex to complete grid  ← exposes gaps as NaN (no data assigned yet)
  5. ffill(limit=N)            ← propagates the most recent *past* bar forward
  6. dropna                    ← removes large gaps; never fills backward
  7. OHLC validation           ← read-only check

``bfill`` is NEVER called.  Resampling is NOT performed (no aggregation that
could aggregate future bars into past rows).

D1 gap handling
---------------
Mapping D1 to a "business day" frequency is tricky because:
  - Forex trades Sunday night, so the week-start bar has a Sunday timestamp.
  - Public holidays vary by country and broker.
We skip the reindex step for D1 entirely and rely on dedup + weekend-drop +
dropna.  D1 data from MT5 is clean enough that this is sufficient.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import DotDict, load_config
from src.data.downloader import load_raw, raw_csv_path

log = logging.getLogger(__name__)

# ── Forex session constants ───────────────────────────────────────────────────
# Forex opens Sunday ~21:00 UTC (some brokers: 22:00) and closes Friday ~21:00.
# Adjust FOREX_SESSION_OPEN if your broker starts the weekly session later.
FOREX_SESSION_OPEN_HOUR_UTC: int = 21   # Sunday hour at which market opens


def _is_off_market(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Return a boolean array: True for timestamps outside forex trading hours.

    Rules (conservative, broker-agnostic):
    - Saturday (dayofweek == 5): always off-market.
    - Sunday before FOREX_SESSION_OPEN_HOUR_UTC: pre-session, off-market.

    Friday bars are intentionally NOT masked — close time is broker-dependent
    and the terminal simply won't produce bars after the session ends.
    """
    dow  = np.array(index.dayofweek)   # Monday=0 … Sunday=6
    hour = np.array(index.hour)

    saturday     = (dow == 5)
    early_sunday = (dow == 6) & (hour < FOREX_SESSION_OPEN_HOUR_UTC)

    return saturday | early_sunday


# ── Timeframe → pandas frequency alias ───────────────────────────────────────
# pandas 1.5.3 uses uppercase aliases: "T" (minute), "H" (hour), "B" (bday).
# None means: skip the reindex/gap-fill step for this timeframe.
_TF_TO_FREQ: dict[str, str | None] = {
    "M1":  "T",
    "M5":  "5T",
    "M15": "15T",
    "M30": "30T",
    "H1":  "H",
    "H4":  "4H",
    "D1":  None,   # See module docstring: D1 gap detection is skipped
    "W1":  None,
}


# ── Main cleaner ──────────────────────────────────────────────────────────────
def clean(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fill_limit: int = 3,
) -> pd.DataFrame:
    """
    Produce a clean OHLCV DataFrame from raw MT5 data.

    Parameters
    ----------
    df         : Raw DataFrame, UTC-aware DatetimeIndex, from ``load_raw()``.
    symbol     : Used for log messages only.
    timeframe  : Config timeframe string, e.g. ``"H1"``.
    fill_limit : Max consecutive NaN bars to forward-fill.
                 Gaps larger than this are treated as genuine session breaks
                 (holidays / broker outages) and dropped.  At H1, fill_limit=3
                 covers clock-change gaps (23:00→01:00 DST jumps).

    Returns
    -------
    pd.DataFrame  — same columns as input, UTC-aware DatetimeIndex, no NaNs.
    """
    n_raw = len(df)
    log.info("[%s %s] Starting clean — %d raw bars.", symbol, timeframe, n_raw)

    # ── Step 1: Deduplicate ───────────────────────────────────────────────────
    # keep="last" because the terminal may have updated a bar's data on re-query.
    n_before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    n_dupes = n_before - len(df)
    if n_dupes:
        log.warning("  Dropped %d duplicate timestamp(s).", n_dupes)

    # ── Step 2: Drop off-market bars ──────────────────────────────────────────
    off_mask = _is_off_market(df.index)
    n_off = int(off_mask.sum())
    if n_off:
        log.info(
            "  Dropping %d off-market bar(s) (Saturday / early-Sunday). "
            "Normal if broker includes partial weekend rows.",
            n_off,
        )
        df = df[~off_mask]

    # ── Step 3: Sort ascending ────────────────────────────────────────────────
    df = df.sort_index()

    # ── Step 4: Reindex to complete regular grid → expose gaps as NaN rows ───
    freq = _TF_TO_FREQ.get(timeframe)
    n_missing = 0

    if freq is not None and len(df) > 1:
        full_index = pd.date_range(
            start=df.index[0],
            end=df.index[-1],
            freq=freq,
            tz="UTC",
        )
        # Remove weekend/off-market slots from the expected grid too
        full_index = full_index[~_is_off_market(full_index)]

        n_expected = len(full_index)
        n_actual   = len(df)
        n_missing  = n_expected - n_actual

        if n_missing > 0:
            log.info(
                "  Gap analysis: %d missing bar(s) out of %d expected (%.2f%%). "
                "Will ffill gaps ≤ %d bar(s); larger gaps → dropped.",
                n_missing, n_expected,
                100.0 * n_missing / n_expected,
                fill_limit,
            )
        else:
            log.info("  No missing bars found in the regular grid.")

        # reindex inserts NaN rows for missing timestamps — no data is
        # fabricated here, we're just making the gap explicit.
        df = df.reindex(full_index)

    elif freq is None:
        log.info(
            "  Reindex step skipped for %s (see D1 note in module docstring).",
            timeframe,
        )

    # ── Step 5: Forward-fill small gaps ───────────────────────────────────────
    # ffill propagates the last known bar forward — strictly no future leakage.
    # limit= caps the fill so large outage gaps aren't silently bridged.
    if n_missing > 0:
        df = df.ffill(limit=fill_limit)

    # ── Step 6: Drop rows still NaN after fill ─────────────────────────────────
    n_before_drop = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    n_dropped_na = n_before_drop - len(df)
    if n_dropped_na:
        log.info(
            "  Dropped %d row(s) still NaN after fill_limit=%d "
            "(likely holidays or outages > %d bars).",
            n_dropped_na, fill_limit, fill_limit,
        )

    # ── Step 7: OHLC sanity validation ────────────────────────────────────────
    _validate_ohlc(df, symbol, timeframe)

    # ── Summary ───────────────────────────────────────────────────────────────
    pct_kept = 100.0 * len(df) / n_raw if n_raw else 0.0
    log.info(
        "[%s %s] Clean complete: %d raw → %d clean bars (%.1f%% retained) | "
        "%s → %s",
        symbol, timeframe, n_raw, len(df), pct_kept,
        df.index[0].date(), df.index[-1].date(),
    )
    return df


def _validate_ohlc(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    """
    Soft-check OHLC relationships: H ≥ max(O,C) and L ≤ min(O,C) and H ≥ L.

    Logs warnings rather than raising — bad ticks occasionally appear in real
    broker feeds and we don't want one bad row to abort the whole pipeline.
    Investigate any warnings; they are rare on major pairs.
    """
    if df.empty:
        return

    oc_max = df[["open", "close"]].max(axis=1)
    oc_min = df[["open", "close"]].min(axis=1)

    bad_high = df["high"] < oc_max
    bad_low  = df["low"]  > oc_min
    bad_hl   = df["high"] < df["low"]

    n_bad = int((bad_high | bad_low | bad_hl).sum())
    if n_bad:
        log.warning(
            "[%s %s] OHLC sanity FAILED on %d bar(s): "
            "high<max(O,C)=%d | low>min(O,C)=%d | high<low=%d. "
            "Investigate these timestamps.",
            symbol, timeframe, n_bad,
            int(bad_high.sum()), int(bad_low.sum()), int(bad_hl.sum()),
        )
    else:
        log.info("[%s %s] OHLC sanity check passed.", symbol, timeframe)


# ── Persistence helpers ───────────────────────────────────────────────────────
def processed_csv_path(processed_dir: Path, symbol: str, timeframe: str) -> Path:
    return processed_dir / f"{symbol}_{timeframe}_clean.csv"


def save_processed(
    df: pd.DataFrame,
    processed_dir: Path,
    symbol: str,
    timeframe: str,
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_csv_path(processed_dir, symbol, timeframe)
    df.to_csv(path)
    log.info("Saved processed → %s  (%d bars)", path.name, len(df))
    return path


def load_processed(processed_dir: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    path = processed_csv_path(processed_dir, symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}\n"
            "Run the cleaner first:  python scripts/download_data.py --clean-only"
        )
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    return df


# ── Public entry point ────────────────────────────────────────────────────────
def run_clean(cfg: DotDict | None = None) -> dict[tuple[str, str], Path]:
    """
    Clean all active (symbol, timeframe) pairs.

    Reads from cfg.paths.data.raw, writes to cfg.paths.data.processed.
    Returns a dict mapping (symbol, timeframe) → Path of saved clean CSV.
    """
    if cfg is None:
        cfg = load_config()

    symbols    = list(cfg.instruments.active)
    timeframes = [cfg.timeframes.primary] + list(cfg.timeframes.context)
    raw_dir    = cfg.paths.data.raw
    proc_dir   = cfg.paths.data.processed

    results: dict[tuple[str, str], Path] = {}

    for symbol in symbols:
        for tf in timeframes:
            log.info("=== Cleaning %s / %s ===", symbol, tf)
            try:
                df_raw   = load_raw(raw_dir, symbol, tf)
                df_clean = clean(df_raw, symbol, tf)
                path     = save_processed(df_clean, proc_dir, symbol, tf)
                results[(symbol, tf)] = path
            except FileNotFoundError as exc:
                log.error("Cannot clean %s %s — file missing: %s", symbol, tf, exc)
            except Exception as exc:
                log.error(
                    "Cleaning FAILED for %s %s: %s", symbol, tf, exc,
                    exc_info=True,
                )

    log.info("Cleaning complete: %d / %d pairs.", len(results), len(symbols) * len(timeframes))
    return results