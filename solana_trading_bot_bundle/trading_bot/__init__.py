# solana_trading_bot_bundle/trading_bot/__init__.py
from __future__ import annotations

# Make this package importable as both:
#   - solana_trading_bot_bundle.trading_bot
#   - trading_bot   (many legacy imports use this)
import sys as _sys
import importlib as _importlib

# Register a top-level alias so "import trading_bot" works anywhere.
_sys.modules.setdefault("trading_bot", _sys.modules.get(__name__))

__all__ = ["database", "eligibility", "fetching", "market_data", "trading", "utils_exec"]

def __getattr__(name: str):
    if name in __all__:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + __all__)

