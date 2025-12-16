# solana_trading_bot_bundle/utils/__init__.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time as _t
from pathlib import Path

# -------------------------------
# AppData helpers (robust)
# -------------------------------
APP_NAME = "SOLOTradingBot"

def _appdata_root() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if os.name == "posix" and hasattr(os, "uname") and os.uname().sysname.lower() == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    return Path(xdg) / APP_NAME if xdg else Path.home() / ".local" / "share" / APP_NAME

def appdata_dir() -> Path:
    return _appdata_root()

def logs_dir() -> Path:
    return appdata_dir() / "logs"

def data_dir() -> Path:
    return appdata_dir() / "data"

def ensure_app_dirs() -> None:
    for p in (appdata_dir(), logs_dir(), data_dir()):
        p.mkdir(parents=True, exist_ok=True)

def prefer_appdata_path(relative_name: str) -> Path:
    """Return a Path inside AppData for the given filename, ensuring parent exists."""
    p = appdata_dir() / relative_name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def prefer_appdata_file(relative_name: str) -> str:
    """Back-compat: same as prefer_appdata_path but returns str."""
    return str(prefer_appdata_path(relative_name))

# Canonical important paths
def config_path() -> Path: return prefer_appdata_path("config.yaml")
def env_path()    -> Path: return prefer_appdata_path(".env")
def db_path()     -> Path: return prefer_appdata_path("tokens.sqlite3")
def token_cache_path() -> Path: return prefer_appdata_path("tokens.sqlite3")

# Optionally import env bootstrap helpers if they exist (no hard dependency)
try:
    from .env_loader import (  # type: ignore
        load_env_first_found,
        ensure_appdata_env_bootstrap,
        get_active_private_key,
        ensure_one_time_credentials,
    )
except Exception:
    def load_env_first_found(override: bool = False):
        return None
    def ensure_appdata_env_bootstrap(*a, **k):
        return None
    def get_active_private_key():
        return None
    def ensure_one_time_credentials(*a, **k):
        return None

# -------------------------------
# Tiny TTL cache for market_data
# -------------------------------
class _TTLCache:
    def __init__(self):
        # key -> (stored_at_seconds, value, ttl_seconds)
        self._d: dict[str, tuple[float, object, float]] = {}

    def get(self, key: str):
        item = self._d.get(key)
        if not item:
            return None
        ts, val, ttl = item
        if _t.time() - ts > ttl:
            try:
                del self._d[key]
            except Exception:
                pass
            return None
        return val

    def set(self, key: str, val, ttl: float = 15.0):
        self._d[key] = (_t.time(), val, ttl)

# What trading_bot/market_data.py imports:
price_cache = _TTLCache()

__all__ = [
    # appdata helpers
    "APP_NAME", "appdata_dir", "logs_dir", "data_dir", "ensure_app_dirs",
    "prefer_appdata_path", "prefer_appdata_file",
    "config_path", "env_path", "db_path", "token_cache_path",
    # env helpers (best-effort)
    "load_env_first_found", "ensure_appdata_env_bootstrap",
    "get_active_private_key", "ensure_one_time_credentials",
    # cache
    "price_cache",
]
