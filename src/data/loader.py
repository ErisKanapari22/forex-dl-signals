# src/data/loader.py
"""
Chronological train / val / test splitting and walk-forward CV.

Why not random k-fold?
-----------------------
Financial time-series are auto-correlated.  Shuffling bars and putting some
in train and the temporally-adjacent ones in val is data leakage — the model
sees the statistical neighbourhood of every validation bar during training.
The result is validation metrics that look great but collapse on live data.

We enforce three rules:
  1. All splits are contiguous time slices (no shuffling, ever).
  2. The test set is fixed at the *end* of the series and is never touched
     during any training or validation loop.  It is a true holdout.
  3. Walk-forward CV folds are generated only from train + val data.
     The test set is not passed to ``make_walk_forward()``.

Split by position, not by date
--------------------------------
Splitting by position (integer index into the cleaned DataFrame) is more
robust than splitting by date.  Reasons:
  - Holiday gaps create uneven calendar density; position gives exact ratios.
  - You can't accidentally pick a date that falls inside a weekend gap.
  - Reproducible: the same cleaned CSV always produces the same split.

Walk-forward modes
-------------------
  Expanding (default):
      Train window grows with each fold.  Start from the beginning each time.
      Good when data volume is limited and you don't want to discard early bars.
      The risk: the model may over-weight ancient price regimes.

  Sliding (fixed window):
      Train window of fixed size slides forward, dropping early bars.
      Good when market structure has changed and old data is misleading.
      Requires ``initial_train_bars`` to be large enough to learn from.

This module returns DataFrames only.
PyTorch tensors and sliding-window sequences are built in the dataset layer
(Chat 04/05).  Keeping them separate makes this module testable without a GPU.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from src.utils.io import DotDict, load_config
from src.data.cleaner import load_processed

log = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class DataSplit:
    """
    Holds train / val / test DataFrames for one (symbol, timeframe) pair.

    Invariants checked at construction:
      - train ends strictly before val starts
      - val   ends strictly before test starts
    """
    symbol:    str
    timeframe: str
    train:     pd.DataFrame
    val:       pd.DataFrame
    test:      pd.DataFrame

    def __post_init__(self) -> None:
        assert len(self.train) > 0, "Train split is empty."
        assert len(self.val)   > 0, "Val split is empty."
        assert len(self.test)  > 0, "Test split is empty."
        assert self.train.index[-1] < self.val.index[0], (
            f"Leakage: train ends at {self.train.index[-1]} but "
            f"val starts at {self.val.index[0]}."
        )
        assert self.val.index[-1] < self.test.index[0], (
            f"Leakage: val ends at {self.val.index[-1]} but "
            f"test starts at {self.test.index[0]}."
        )

    def summary(self) -> str:
        def _fmt(df: pd.DataFrame, label: str) -> str:
            return (
                f"{label}={len(df):,} bars "
                f"[{df.index[0].date()} → {df.index[-1].date()}]"
            )
        return (
            f"{self.symbol} {self.timeframe} | "
            + _fmt(self.train, "train") + " | "
            + _fmt(self.val,   "val")   + " | "
            + _fmt(self.test,  "test")
        )


@dataclass
class WalkForwardFold:
    """One (train, val) pair from walk-forward cross-validation."""
    fold_idx: int
    train:    pd.DataFrame
    val:      pd.DataFrame

    def summary(self) -> str:
        return (
            f"Fold {self.fold_idx:02d} | "
            f"train={len(self.train):,} "
            f"[{self.train.index[0].date()} → {self.train.index[-1].date()}] | "
            f"val={len(self.val):,} "
            f"[{self.val.index[0].date()} → {self.val.index[-1].date()}]"
        )


# ── Splitter ──────────────────────────────────────────────────────────────────
def split_chronological(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    train_ratio: float,
    val_ratio: float,
) -> DataSplit:
    """
    Slice *df* into train / val / test using integer-position boundaries.

    The test ratio is implied: 1 - train_ratio - val_ratio.

    Parameters
    ----------
    df           : Clean, ascending, UTC-indexed DataFrame from load_processed().
    symbol       : Stored in DataSplit for provenance.
    timeframe    : Stored in DataSplit for provenance.
    train_ratio  : Fraction allocated to training   (e.g. 0.70).
    val_ratio    : Fraction allocated to validation (e.g. 0.15).

    Returns
    -------
    DataSplit with train / val / test as non-overlapping DataFrames.

    Raises
    ------
    ValueError : Fewer than 100 bars (too few to split meaningfully).
    """
    n = len(df)
    if n < 100:
        raise ValueError(
            f"Only {n} bars in {symbol} {timeframe} — too few to split. "
            "Check download and clean steps."
        )

    i_val  = int(n * train_ratio)
    i_test = int(n * (train_ratio + val_ratio))

    # Ensure each split has at least 1 bar (can't happen at sane ratios, but guard)
    if i_val == 0 or i_test <= i_val or i_test >= n:
        raise ValueError(
            f"Split boundaries degenerate for n={n}, "
            f"train_ratio={train_ratio}, val_ratio={val_ratio}."
        )

    split = DataSplit(
        symbol=symbol,
        timeframe=timeframe,
        train=df.iloc[:i_val].copy(),
        val=df.iloc[i_val:i_test].copy(),
        test=df.iloc[i_test:].copy(),
    )
    log.info("Split | %s", split.summary())
    return split


# ── Walk-forward generator ────────────────────────────────────────────────────
def walk_forward_splits(
    df: pd.DataFrame,
    initial_train_bars: int,
    val_bars: int,
    step_bars: int,
    expanding: bool = True,
) -> Iterator[WalkForwardFold]:
    """
    Yield (train, val) folds by sliding a window forward through *df*.

    ⚠  Pass ONLY train + val data here — never the test set.

    Parameters
    ----------
    df                 : DataFrame covering train + val period only.
    initial_train_bars : Minimum bars in the first training fold.
                         Must be > model sequence_length so the model can
                         actually form at least one training sample.
    val_bars           : Bars in each validation window (fixed across folds).
    step_bars          : Bars to advance per fold.
                         step_bars = val_bars    → non-overlapping val windows.
                         step_bars < val_bars    → overlapping val windows
                                                   (more folds, not independent).
    expanding          : True  → growing train window (recommended default).
                         False → fixed-size sliding train window.

    Yields
    ------
    WalkForwardFold for each valid fold.

    Example (expanding, initial=1000, val=240, step=120)
    -------
        Fold 00: train[  0:1000] val[1000:1240]
        Fold 01: train[  0:1120] val[1120:1360]
        Fold 02: train[  0:1240] val[1240:1480]
        ...
    """
    n         = len(df)
    fold_idx  = 0
    train_end = initial_train_bars

    while train_end + val_bars <= n:
        train_start = 0 if expanding else (train_end - initial_train_bars)
        fold = WalkForwardFold(
            fold_idx=fold_idx,
            train=df.iloc[train_start:train_end].copy(),
            val=df.iloc[train_end:train_end + val_bars].copy(),
        )
        log.debug("Walk-forward: %s", fold.summary())
        yield fold

        train_end += step_bars
        fold_idx  += 1

    if fold_idx == 0:
        log.warning(
            "Walk-forward generated 0 folds.  "
            "initial_train_bars=%d + val_bars=%d > n=%d. "
            "Reduce initial_train_bars or val_bars.",
            initial_train_bars, val_bars, n,
        )
    else:
        log.info("Walk-forward CV: %d fold(s) generated.", fold_idx)


# ── Public entry points ───────────────────────────────────────────────────────
def load_splits(
    cfg: DotDict | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
) -> DataSplit:
    """
    Load a processed CSV and return a DataSplit.

    Defaults to the first active instrument and the primary timeframe.
    Override with explicit symbol / timeframe for context timeframes.
    """
    if cfg is None:
        cfg = load_config()

    symbol    = symbol    or cfg.instruments.active[0]
    timeframe = timeframe or cfg.timeframes.primary

    df = load_processed(cfg.paths.data.processed, symbol, timeframe)

    return split_chronological(
        df,
        symbol=symbol,
        timeframe=timeframe,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
    )


def make_walk_forward(
    split: DataSplit,
    initial_train_bars: int | None = None,
    val_bars: int | None = None,
    step_bars: int | None = None,
    expanding: bool = True,
) -> Iterator[WalkForwardFold]:
    """
    Build walk-forward folds from a DataSplit's train + val period.

    The test set in *split* is NEVER touched — it remains a sealed holdout.

    Defaults (when parameters are None):
      initial_train_bars = 60% of len(train + val)
      val_bars           = 10% of len(train + val)
      step_bars          = val_bars // 2  (50% overlap between val windows)

    These defaults give a reasonable number of folds for a 5-year history at H1.
    Tune them explicitly for production training runs.
    """
    trainval = pd.concat([split.train, split.val]).sort_index()
    n = len(trainval)

    _initial = initial_train_bars or int(0.60 * n)
    _val     = val_bars           or int(0.10 * n)
    _step    = step_bars          or max(1, _val // 2)

    log.info(
        "Walk-forward config | initial_train=%d, val_window=%d, step=%d, "
        "expanding=%s | source: %s %s train+val (%d bars)",
        _initial, _val, _step, expanding,
        split.symbol, split.timeframe, n,
    )

    return walk_forward_splits(trainval, _initial, _val, _step, expanding)