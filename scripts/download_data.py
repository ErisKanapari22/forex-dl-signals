# scripts/download_data.py
"""
CLI entry point: download raw OHLCV from MT5 and/or clean it.

Usage
-----
  # Full pipeline (default)
  python scripts/download_data.py

  # Download only — skip cleaning
  python scripts/download_data.py --download-only

  # Re-clean existing raw files — no MT5 connection needed
  python scripts/download_data.py --clean-only

  # Override active instruments for this run
  python scripts/download_data.py --instruments EURUSD GBPUSD

  # Override timeframes for this run
  python scripts/download_data.py --timeframes H1 H4

  # Force config reload (ignore singleton cache)
  python scripts/download_data.py --reload-config
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Ensure project root is on sys.path when run as a script ──────────────────
# This is needed if the project is NOT installed with `pip install -e .`.
# If it IS installed (recommended), this is a harmless no-op.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.io import load_config
from src.utils.logger import setup_logging
from src.data.downloader import run_download
from src.data.cleaner import run_clean


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download MT5 OHLCV data and/or clean it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--download-only",
        action="store_true",
        help="Only download raw data from MT5; skip cleaning.",
    )
    mode.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean existing raw files; skip MT5 connection.",
    )
    p.add_argument(
        "--instruments",
        nargs="+",
        metavar="SYMBOL",
        help="Override cfg.instruments.active for this run.",
    )
    p.add_argument(
        "--timeframes",
        nargs="+",
        metavar="TF",
        help="Override cfg.timeframes for this run (primary + context combined).",
    )
    p.add_argument(
        "--reload-config",
        action="store_true",
        help="Force re-read of YAML files (bypasses singleton cache).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config(reload=args.reload_config)
    setup_logging(cfg)

    import logging
    log = logging.getLogger(__name__)
    log.info("=== forex-dl-signals | download_data ===")
    log.info("Environment : %s", cfg.project.environment)
    log.info("Project root: %s", cfg._meta.project_root)

    # ── CLI overrides (mutate cfg in-place for this run only) ─────────────────
    # DotDict is a dict subclass, so we can update it directly.
    # We don't write back to YAML — this is a run-time override only.
    if args.instruments:
        log.info("Overriding instruments: %s", args.instruments)
        cfg["instruments"]["active"] = args.instruments

    if args.timeframes:
        log.info("Overriding timeframes: %s", args.timeframes)
        # Flatten into primary + context structure (first becomes primary)
        cfg["timeframes"]["primary"] = args.timeframes[0]
        cfg["timeframes"]["context"] = args.timeframes[1:]

    # ── Run pipeline ──────────────────────────────────────────────────────────
    exit_code = 0

    if not args.clean_only:
        log.info("--- Phase 1: Download ---")
        try:
            results = run_download(cfg)
            if not results:
                log.error("No data downloaded — check MT5 connection and logs above.")
                exit_code = 1
            else:
                log.info(
                    "Downloaded %d pair(s): %s",
                    len(results),
                    [f"{s}/{tf}" for (s, tf) in results],
                )
        except Exception as exc:
            log.exception("Download phase failed: %s", exc)
            return 1

    if not args.download_only:
        log.info("--- Phase 2: Clean ---")
        try:
            results = run_clean(cfg)
            if not results:
                log.error("No data cleaned — check raw files exist and logs above.")
                exit_code = 1
            else:
                log.info(
                    "Cleaned %d pair(s): %s",
                    len(results),
                    [f"{s}/{tf}" for (s, tf) in results],
                )
        except Exception as exc:
            log.exception("Clean phase failed: %s", exc)
            return 1

    log.info("=== Done. Exit code: %d ===", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())