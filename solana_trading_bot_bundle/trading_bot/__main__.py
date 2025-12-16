#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys

def _resolve_funcs():
    """
    Return (main_loop, bootstrap_fn|None) from the bundled module,
    regardless of whether we're imported relatively or absolutely.
    """
    # Try relative first (when run as -m solana_trading_bot_bundle.trading_bot)
    try:
        from ..solana_trading_bot import main_loop as _main_loop  # type: ignore
        try:
            from ..solana_trading_bot import _bootstrap_db_once as _bootstrap  # type: ignore
        except Exception:
            _bootstrap = None
        return _main_loop, _bootstrap
    except Exception:
        # Fall back to absolute import
        from solana_trading_bot_bundle.solana_trading_bot import main_loop as _main_loop  # type: ignore
        try:
            from solana_trading_bot_bundle.solana_trading_bot import _bootstrap_db_once as _bootstrap  # type: ignore
        except Exception:
            _bootstrap = None
        return _main_loop, _bootstrap

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Solana trading bot runner")
    p.add_argument("-c", "--config", dest="config", default=None,
                   help="Path to config.yaml (optional; will use AppData default if omitted)")
    return p.parse_args(argv)

def main(argv=None):
    args = _parse_args(argv)
    main_loop, bootstrap = _resolve_funcs()

    async def _runner():
        # Run DB bootstrap once (if exposed) before the main loop
        if bootstrap:
            try:
                await bootstrap()  # async bootstrap
            except TypeError:
                # (In case someone made it sync later)
                bootstrap()
        await main_loop(args.config)

    asyncio.run(_runner())

if __name__ == "__main__":
    main()
