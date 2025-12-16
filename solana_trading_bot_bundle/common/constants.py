# solana_trading_bot_bundle/common/constants.py
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional, Mapping, Final

# Public API re-exported by common/__init__.py
__all__ = [
    "APP_NAME",
    "local_appdata_dir",
    "appdata_dir",
    "logs_dir",
    "data_dir",
    "config_path",
    "env_path",
    "db_path",
    "token_cache_path",
    "ensure_app_dirs",
    "prefer_appdata_file",
    "CAP_DISPLAY_ALIAS",
    "display_cap",
]

# -----------------------------------------------------------------------------
# App naming
# -----------------------------------------------------------------------------
APP_NAME: Final[str] = "SOLOTradingBot"  # used as the directory name across platforms

# -----------------------------------------------------------------------------
# Platform-aware base dirs
# -----------------------------------------------------------------------------
def _windows_local_appdata() -> Optional[Path]:
    """Return Windows LocalAppData (LOCALAPPDATA), or None."""
    val = os.getenv("LOCALAPPDATA") or os.getenv("LOCAL_APPDATA")
    if not val:
        return None
    try:
        p = Path(val).expanduser()
        if p.exists() or p.parent.exists():
            return p
    except Exception:
        pass
    return None


def _windows_roaming_appdata() -> Optional[Path]:
    """Return Windows Roaming (APPDATA), or None."""
    val = os.getenv("APPDATA")
    if not val:
        return None
    try:
        p = Path(val).expanduser()
        if p.exists() or p.parent.exists():
            return p
    except Exception:
        pass
    return None


def _darwin_app_support() -> Path:
    """macOS application support root."""
    return Path.home() / "Library" / "Application Support"


def _xdg_data_home() -> Path:
    """Linux/Unix XDG data home root."""
    val = os.getenv("XDG_DATA_HOME")
    return Path(val).expanduser() if val else (Path.home() / ".local" / "share")


def local_appdata_dir() -> Path:
    r"""
    Cross-platform "local app data" root for this user.

    - Windows:  %LOCALAPPDATA%
    - macOS:    ~/Library/Application Support
    - Linux:    ~/.local/share
    """
    system = platform.system().lower()
    if system.startswith("win"):
        # Prefer LOCALAPPDATA, fall back to Roaming APPDATA, then ~
        return _windows_local_appdata() or _windows_roaming_appdata() or Path.home()
    if system == "darwin":
        return _darwin_app_support()
    # Linux / everything else
    return _xdg_data_home()


def appdata_dir() -> Path:
    r"""
    Full application data directory:

    - Windows:  %LOCALAPPDATA%\SOLOTradingBot
    - macOS:    ~/Library/Application Support/SOLOTradingBot
    - Linux:    ~/.local/share/SOLOTradingBot
    """
    return local_appdata_dir() / APP_NAME


def logs_dir() -> Path:
    """Directory where rotating logs are stored."""
    return appdata_dir() / "logs"


def data_dir() -> Path:
    """Directory for general runtime data (DB, caches)."""
    return appdata_dir()


def config_path() -> Path:
    """Default location for YAML config."""
    return appdata_dir() / "config.yaml"


def env_path() -> Path:
    """Default location for a .env file (optional)."""
    return appdata_dir() / ".env"


def db_path() -> Path:
    """Default SQLite DB location (tokens/trades/etc.)."""
    # Matches your logs that reference tokens.sqlite3
    return appdata_dir() / "tokens.sqlite3"


def token_cache_path() -> Path:
    """Optional token cache JSON path."""
    return appdata_dir() / "token_cache.json"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_app_dirs() -> None:
    """
    Create the app data hierarchy if missing. Safe to call multiple times.
    Never raises on filesystem errors.
    """
    try:
        appdata_dir().mkdir(parents=True, exist_ok=True)
        logs_dir().mkdir(parents=True, exist_ok=True)
        # data_dir() is same as appdata_dir(); created above.
    except Exception:
        # never crash import for FS reasons
        pass


def prefer_appdata_file(
    relative_name: str,
    *,
    ensure_parent: bool = True,
    default_contents: Optional[str] = None,
) -> Path:
    r"""
    Resolve a file path *inside* the appdata directory.

    Example:
        prefer_appdata_file("config.yaml")

    If `default_contents` is provided and the file does not exist,
    it will be created with those contents.
    """
    p = appdata_dir() / relative_name
    try:
        if ensure_parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        if default_contents is not None and not p.exists():
            p.write_text(default_contents, encoding="utf-8")
    except Exception:
        # ignore filesystem errors here; caller can still handle
        pass
    return p


# Create directories immediately so first-run testers don't have to
ensure_app_dirs()

# -----------------------------------------------------------------------------
# UI display aliases (cosmetic only)
# -----------------------------------------------------------------------------
# Keep internal behavior/filters using original bucket keys (e.g., "High Cap").
# Only the *displayed* text is aliased here; zero logic/DB impact.
CAP_DISPLAY_ALIAS: Mapping[str, str] = {"High Cap": "Large Cap"}


def display_cap(name: str) -> str:
    """Return the display label for a given cap-bucket name."""
    return CAP_DISPLAY_ALIAS.get(name, name)

# -----------------------------------------------------------------------------
# Notes (raw string to avoid backslash warnings in docstrings)
# -----------------------------------------------------------------------------
__doc__ = r"""
This module centralizes all cross-platform paths for the SOLOTradingBot app.

It intentionally:
  - Uses one app directory (APP_NAME) across platforms.
  - Creates directories on import so non-technical testers have a working
    folder structure “out of the box”.
  - Keeps helpers small and dependency-free.

Paths:
  Windows:
    - App data root:   %LOCALAPPDATA%  (fallback: %APPDATA%)
    - App folder:      %LOCALAPPDATA%\SOLOTradingBot
    - Logs:            %LOCALAPPDATA%\SOLOTradingBot\logs
    - Config:          %LOCALAPPDATA%\SOLOTradingBot\config.yaml
    - DB:              %LOCALAPPDATA%\SOLOTradingBot\tokens.sqlite3

  macOS:
    - App data root:   ~/Library/Application Support
    - App folder:      ~/Library/Application Support/SOLOTradingBot
    - Logs:            ~/Library/Application Support/SOLOTradingBot/logs
    - Config:          ~/Library/Application Support/SOLOTradingBot/config.yaml
    - DB:              ~/Library/Application Support/SOLOTradingBot/tokens.sqlite3

  Linux/other (XDG):
    - App data root:   ~/.local/share
    - App folder:      ~/.local/share/SOLOTradingBot
    - Logs:            ~/.local/share/SOLOTradingBot/logs
    - Config:          ~/.local/share/SOLOTradingBot/config.yaml
    - DB:              ~/.local/share/SOLOTradingBot/tokens.sqlite3
"""
