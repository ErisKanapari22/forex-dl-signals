# src/utils/io.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)


# ── Project root discovery ────────────────────────────────────────────────────
def _find_project_root() -> Path:
    candidate = Path(__file__).resolve()
    for parent in [candidate, *candidate.parents]:
        if (parent / "setup.py").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT: Path = _find_project_root()
CONFIG_DIR: Path = Path(os.getenv("CONFIG_DIR", str(PROJECT_ROOT / "config")))


# ── DotDict ───────────────────────────────────────────────────────────────────
class DotDict(dict):
    """Dict with attribute-style access. Nested dicts auto-converted."""

    def __init__(self, data: dict | None = None):
        super().__init__(data or {})
        for key, value in self.items():
            if isinstance(value, dict):
                super().__setitem__(key, DotDict(value))

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"Config has no key '{key}'. "
                f"Available keys: {list(self.keys())}"
            ) from None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        log.warning("Config file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _resolve_paths(cfg: DotDict, root: Path) -> DotDict:
    if "paths" not in cfg:
        return cfg

    def _resolve_node(node: Any) -> Any:
        if isinstance(node, dict):
            return DotDict({k: _resolve_node(v) for k, v in node.items()})
        if isinstance(node, str):
            p = Path(node)
            return root / p if not p.is_absolute() else p
        return node

    cfg["paths"] = _resolve_node(cfg["paths"])

    if "logging" in cfg and "file_path" in cfg.logging:
        lp = Path(cfg.logging.file_path)
        cfg["logging"]["file_path"] = root / lp if not lp.is_absolute() else lp

    return cfg


def _validate(cfg: DotDict) -> None:
    if "data" in cfg:
        d = cfg.data
        if all(k in d for k in ("train_ratio", "val_ratio", "test_ratio")):
            total = d.train_ratio + d.val_ratio + d.test_ratio
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Train/val/test ratios must sum to 1.0, got {total:.6f}"
                )

    if "instruments" in cfg:
        if not cfg.instruments.get("active", []):
            raise ValueError("instruments.active is empty in config.yaml")

    if "signal" in cfg.get("trading", {}):
        ct = cfg.trading.signal.get("confidence_threshold", 0.5)
        if not (0.0 < ct < 1.0):
            raise ValueError(f"confidence_threshold must be in (0, 1), got {ct}")


# ── Singleton ─────────────────────────────────────────────────────────────────
_CONFIG_CACHE: DotDict | None = None


def load_config(
    config_dir: Path | str | None = None,
    reload: bool = False,
) -> DotDict:
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and not reload:
        return _CONFIG_CACHE

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    cfg_dir = Path(config_dir) if config_dir else CONFIG_DIR

    base      = _load_yaml(cfg_dir / "config.yaml")
    model_cfg = _load_yaml(cfg_dir / "model_config.yaml")
    trade_cfg = _load_yaml(cfg_dir / "trading_config.yaml")

    merged = _deep_merge(base, {"model": model_cfg, "trading": trade_cfg})
    cfg = DotDict(merged)
    cfg = _resolve_paths(cfg, PROJECT_ROOT)
    _validate(cfg)

    cfg["_meta"] = DotDict({
        "project_root": PROJECT_ROOT,
        "config_dir": cfg_dir,
    })

    _CONFIG_CACHE = cfg
    log.info("Config loaded: %s | env=%s", cfg.project.name, cfg.project.environment)
    return cfg