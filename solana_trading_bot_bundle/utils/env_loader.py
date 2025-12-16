# solana_trading_bot_bundle/utils/env_loader.py
from __future__ import annotations

import logging
import os
import sys
import shutil
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv, find_dotenv
import dotenv  # for monkey-patch

logger = logging.getLogger(__name__)

APP_DIR_NAME = "SOLOTradingBot"

# Relative resource names used by other modules
token_cache_path: str = "tokens.sqlite3"
log_file_path: str = "logs/bot.log"


def _exe_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parents[2]


def _appdata_base_dir() -> Path:
    if os.name == "nt":
        base = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or str(Path.home() / "AppData" / "Local")
        )
        return Path(base) / APP_DIR_NAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME

    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / APP_DIR_NAME
    return Path.home() / ".local" / "share" / APP_DIR_NAME


def _appdata_env_path() -> Path:
    return _appdata_base_dir() / ".env"


def _candidate_env_paths() -> list[Path]:
    exe_dir = _exe_dir()
    return [
        Path.cwd() / ".env",  # project CWD (dev)
        exe_dir / ".env",     # alongside exe/binary
        _appdata_env_path(),  # user appdata
    ]


def load_env_first_found(override: bool = False) -> Optional[Path]:
    """
    Priority:
      1) DOTENV_PATH env var (if set and exists)
      2) Per-user LocalAppData path: %LOCALAPPDATA%/SOLOTradingBot/.env
      3) Fallback to candidate list (CWD, exe_dir, appdata)
    Returns the Path loaded or None.
    """
    # 1) explicit override via env var
    dotenv_override = os.environ.get("DOTENV_PATH")
    if dotenv_override:
        try:
            p = Path(dotenv_override)
            if p.exists():
                load_dotenv(dotenv_path=str(p), override=override)
                logger.info("Loaded .env from DOTENV_PATH: %s", str(p))
                return p
            else:
                logger.warning("DOTENV_PATH set but file not found: %s", str(p))
        except Exception:
            logger.exception("Error loading .env from DOTENV_PATH: %s", str(dotenv_override))

    # 2) authoritative LocalAppData path (per-user)
    try:
        preferred = _appdata_env_path()
        if preferred.exists():
            load_dotenv(dotenv_path=str(preferred), override=override)
            logger.info("Loaded preferred .env from LocalAppData: %s", str(preferred))
            return preferred
    except Exception:
        logger.exception("Error loading preferred LocalAppData .env")

    # 3) fallback: previous candidate behavior
    for p in _candidate_env_paths():
        try:
            if p.exists():
                load_dotenv(dotenv_path=str(p), override=override)
                logger.info("Loaded .env from candidate path: %s", str(p))
                return p
        except Exception:
            logger.exception("Error loading .env from candidate path: %s", str(p))

    logger.warning("No .env file found by loader.")
    return None


def ensure_appdata_env_bootstrap(template_names: tuple[str, ...] = (".env", "default.env")) -> Path:
    dst = _appdata_env_path()
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return dst

    exe_dir = _exe_dir()
    sources = [
        Path.cwd() / template_names[0],
        exe_dir / template_names[0],
        exe_dir / template_names[-1],
    ]
    for src in sources:
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Bootstrapped appdata .env from %s to %s", str(src), str(dst))
            return dst

    dst.write_text(
        "SOLANA_PRIVATE_KEY=\n"
        "BIRDEYE_API_KEY=\n"
        "RUGCHECK_JWT_TOKEN=\n"
        "RUGCHECK_API_KEY=\n"
        "RUGCHECK_ENABLE=true\n"
        "RUGCHECK_DISCOVERY_CHECK=true\n"
        "RUGCHECK_DISCOVERY_FILTER=false\n",
        encoding="utf-8",
    )
    logger.info("Created skeleton appdata .env at %s", str(dst))
    return dst


def prefer_appdata_path(relative_name: str, *, ensure_parent: bool = True, default_contents: Optional[str] = None) -> Path:
    rel = str(relative_name).lstrip("/\\")
    p = _appdata_base_dir() / rel
    if ensure_parent:
        p.parent.mkdir(parents=True, exist_ok=True)
    if default_contents is not None and not p.exists():
        p.write_text(default_contents, encoding="utf-8")
    return p


def prefer_appdata_file(relative_name: str, *, ensure_parent: bool = True, default_contents: Optional[str] = None) -> str:
    return str(prefer_appdata_path(relative_name, ensure_parent=ensure_parent, default_contents=default_contents))


def get_active_private_key() -> Optional[str]:
    v = os.environ.get("SOLANA_PRIVATE_KEY")
    if not v:
        return None
    v = v.strip().strip('"').strip("'")
    return v or None


def ensure_one_time_credentials() -> None:
    return None


__all__ = [
    "APP_DIR_NAME",
    "token_cache_path",
    "log_file_path",
    "load_env_first_found",
    "ensure_appdata_env_bootstrap",
    "prefer_appdata_path",
    "prefer_appdata_file",
    "get_active_private_key",
    "ensure_one_time_credentials",
]

# ---------------------------------------------------------------------------
# Load environment on import (performed once here; override=True so local copies
# do not silently clobber the chosen authoritative values).
# ---------------------------------------------------------------------------
ENV_PATH_LOADED = load_env_first_found(override=True)
logger.info("ENV_PATH_LOADED=%s", str(ENV_PATH_LOADED))

# --- Prevent accidental subsequent loads from other modules by monkey-patching
try:
    def _load_dotenv_noop(*args, **kwargs):
        logger.debug("Skipped redundant load_dotenv call (authoritative loader already ran); args=%s kwargs=%s", args, kwargs)
        return False

    dotenv.load_dotenv = _load_dotenv_noop
    os.environ.setdefault("SOLO_ENV_LOADED", str(ENV_PATH_LOADED) if ENV_PATH_LOADED else "1")
    logger.debug("Patched dotenv.load_dotenv to no-op and set SOLO_ENV_LOADED=%s", os.environ.get("SOLO_ENV_LOADED"))
except Exception:
    logger.exception("Failed to patch dotenv.load_dotenv; redundant loads may still occur.")