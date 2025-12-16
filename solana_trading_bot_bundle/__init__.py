# solana_trading_bot_bundle/__init__.py
from __future__ import annotations

# Package version (falls back to 0.0.0 when not installed)
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("solana-trading-bot-bundle")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except Exception:
    __version__ = "0.0.0"

# Soft import of key constants; provide safe fallbacks if submodule isn't present yet.
try:
    from .common.constants import (
        APP_NAME,
        local_appdata_dir,
        roaming_appdata_dir,
        appdata_join,
    )
except Exception:  # pragma: no cover
    APP_NAME = "SOLOTradingBot"
    local_appdata_dir = None
    roaming_appdata_dir = None

    def appdata_join(*parts: str):
        raise RuntimeError("constants module missing; cannot resolve appdata paths")

# Public API surface (include lazy names so `import *` and IDEs see them)
__all__ = [
    "__version__",
    "APP_NAME",
    "local_appdata_dir",
    "roaming_appdata_dir",
    "appdata_join",
    # Lazy names:
    "load_config",
    "setup_logging",
]

def __getattr__(name: str):
    """Lazy access to selected helpers to avoid import cycles at startup."""
    if name == "load_config":
        # NOTE: point to the actual home of utils_exec
        from .trading_bot.utils_exec import load_config
        return load_config
    if name == "setup_logging":
        from .trading_bot.utils_exec import setup_logging
        return setup_logging
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list[str]:
    # Include lazy attributes so IDE autocompletion and `dir()` see them.
    return sorted(set(globals().keys()) | {"load_config", "setup_logging"})


