# src/utils/logger.py
"""
Logging setup from config.

Call ``setup_logging(cfg)`` once at the top of every entry-point script,
immediately after ``load_config()``.  Module-level loggers created before
this call will retroactively pick up the new handlers because we configure
the *root* logger.

Idempotent: a second call with the same or different cfg is a no-op — safe
if multiple modules call it defensively.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from src.utils.io import DotDict


def setup_logging(cfg: DotDict) -> None:
    """
    Configure the root logger from cfg.logging.

    Adds two handlers:
      StreamHandler   — always active, writes to stderr.
      RotatingFileHandler — active when cfg.logging.file_enabled is True.

    The log directory is created if it does not exist.
    """
    root = logging.getLogger()

    # Idempotency guard: if handlers already attached, do nothing.
    if root.handlers:
        return

    log_cfg = cfg.logging
    level   = getattr(logging, log_cfg.level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt=log_cfg.format,
        datefmt=log_cfg.datefmt,
    )

    root.setLevel(level)

    # ── Console ───────────────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # ── Rotating file ─────────────────────────────────────────────────────────
    if log_cfg.file_enabled:
        log_path = Path(log_cfg.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=int(log_cfg.rotate_mb * 1024 * 1024),
            backupCount=log_cfg.backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)