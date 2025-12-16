from __future__ import annotations

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

import aiohttp

# Auto-load your .env just like the helper does
def _appdata_base() -> Path:
    try:
        from solana_trading_bot_bundle.common.constants import appdata_dir as _appdata_dir  # type: ignore
        base = _appdata_dir() if callable(_appdata_dir) else Path(_appdata_dir)
    except Exception:
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / ".local" / "share")) / "SOLOTradingBot"
    base.mkdir(parents=True, exist_ok=True)
    return base

def _env_file_path() -> Path:
    try:
        from solana_trading_bot_bundle.common.constants import env_path as _env_path  # type: ignore
        envp = _env_path() if callable(_env_path) else Path(_env_path)
    except Exception:
        envp = _appdata_base() / ".env"
    return Path(envp)

def _load_dotenv_into_env() -> None:
    envp = _env_file_path()
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=envp)
        return
    except Exception:
        pass
    # very small fallback parser
    if envp.exists():
        for raw in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if v.startswith('"') and v.endswith('"') and len(v) >= 2:
                v = v[1:-1]
            os.environ.setdefault(k.strip(), v)

# ✅ Corrected import path to where your helper actually lives
from solana_trading_bot_bundle.trading_bot.rugcheck_auth import ensure_valid_rugcheck_headers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TradingBot")

async def main(argv: list[str]) -> int:
    _load_dotenv_into_env()

    # Optional flags:
    force_refresh = ("--refresh" in argv) or (os.getenv("RUGCHECK_LOGIN_FORCE") == "1")
    silent = (os.getenv("RUGCHECK_LOGIN_SILENT") == "1")

    # Must have a private key available if a refresh is needed
    if force_refresh and not (os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("WALLET_PRIVATE_KEY")):
        logger.error("SOLANA_PRIVATE_KEY (or WALLET_PRIVATE_KEY) is required for --refresh.")
        return 2

    async with aiohttp.ClientSession() as session:
        hdrs = await ensure_valid_rugcheck_headers(session=session, force_refresh=force_refresh)
        token = hdrs.get("Authorization", "").removeprefix("Bearer ").strip()

    if not token:
        logger.error("Failed to obtain RugCheck token.")
        return 2

    # The helper already persisted to .env; we just print for convenience.
    if not silent:
        print(json.dumps({"token": token}), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
