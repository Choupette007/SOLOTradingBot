# solana_trading_bot_bundle/trading_bot/utils/__init__.py
from __future__ import annotations

import os, json, logging, datetime as _dt
from pathlib import Path

# YAML is used by load_config
try:
    import yaml  # pyyaml
except Exception as _e:  # still keep module importable even if pyyaml missing
    yaml = None
    logging.getLogger(__name__).warning("PyYAML not available; load_config() will return {}")

# -----------------
# Public API surface
# -----------------
__all__ = [
    "CONFIG_PATH",
    "load_config",
    "custom_json_encoder",
    "WHITELISTED_TOKENS",
    # convenient re-exports from env_loader (many callers import them from utils)
    "ensure_appdata_env_bootstrap",
    "load_env_first_found",
    "prefer_appdata_file",
    "token_cache_path",
    "log_file_path",
    "ensure_one_time_credentials",
    "get_active_private_key",
]

# Default config path (override with SOLO_CONFIG_PATH if you like)
CONFIG_PATH: Path = Path(os.getenv("SOLO_CONFIG_PATH", Path.cwd() / "config.yaml"))

def load_config(path: Path = CONFIG_PATH) -> dict:
    """
    Safe YAML loader. Returns {} on any error so downstream code keeps running.
    database.py imports this symbol from utils. (Keep the name exactly.)
    """
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logging.getLogger(__name__).debug("load_config failed for %s: %s", path, e, exc_info=True)
        return {}

def custom_json_encoder(obj):
    """Used when we json.dumps token rows in the DB code."""
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    try:
        return float(obj)
    except Exception:
        return str(obj)

# Optional allow-list used by blacklist hygiene (DB module imports it).
WHITELISTED_TOKENS: dict[str, str] = {}

# Re-export common env helpers so callers can continue importing from utils.*
try:
    from .env_loader import (
        ensure_appdata_env_bootstrap,
        load_env_first_found,
        prefer_appdata_file,
        token_cache_path,
        log_file_path,
        ensure_one_time_credentials,
        get_active_private_key,
    )
except Exception:
    # Keep module importable even if env_loader isn’t present yet
    def ensure_appdata_env_bootstrap(): ...
    def load_env_first_found(*, override=False): return None
    def prefer_appdata_file(p): return str(p)
    token_cache_path = lambda: str(Path.home() / "Library/Application Support/SOLOTradingBot/tokens.sqlite3")  # type: ignore
    log_file_path    = lambda: str(Path.home() / "Library/Application Support/SOLOTradingBot/logs/bot.log")   # type: ignore
    def ensure_one_time_credentials(): ...
    def get_active_private_key(): return None
