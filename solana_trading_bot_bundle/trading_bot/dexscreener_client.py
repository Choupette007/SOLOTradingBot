# dexscreener_client.py
"""
Safe DexScreener client.

Features
--------
* Global rate-limit (approximately 1 req / 1.2 s)
* TTLCache (5 min) – avoids duplicate calls for the same mint
* CircuitBreaker429 integration – stops hammering the API after 5 consecutive 429s
* Exponential back-off on 429
* Full logging
* Thread-safe (single-worker executor)

Public API
----------
fetch_pairs(mint: str) -> Optional[dict]
best_pair_summary(mint: str) -> dict
"""

# dexscreener_client.py — FIXED FOR NOVEMBER 2025
"""
Safe DexScreener client — WORKING VERSION
"""

from __future__ import annotations

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict

import requests
from cachetools import TTLCache

# ----------------------------------------------------------------------
# Bot utilities
# ----------------------------------------------------------------------
from solana_trading_bot_bundle.trading_bot.utils import CircuitBreaker429

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logger = logging.getLogger("TradingBot")

# ----------------------------------------------------------------------
# Global state
# ----------------------------------------------------------------------
_last_call: float = 0.0
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)

# 5-minute cache (mint -> full JSON response)
_cache: TTLCache = TTLCache(maxsize=1_000, ttl=300)

# Circuit breaker – 5 consecutive 429s → 2-minute cooldown
_circuit_breaker = CircuitBreaker429(threshold=5, cooldown_seconds=120)

# ----------------------------------------------------------------------
# Constants — FIXED ENDPOINT
# ----------------------------------------------------------------------
# OLD (BROKEN): https://api.dexscreener.com/latest/dex/tokens/{}
# NEW (WORKING): https://api.dexscreener.com/token/v1/{mint}/solana
DEX_URL = "https://api.dexscreener.com/token/v1/{}/solana"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Origin": "https://dexscreener.com",
    "Referer": "https://dexscreener.com/",
    "Sec-CH-UA": '"Chromium";v="131", "Google Chrome";v="131", "Not=A?Brand";v="99"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
}

# ----------------------------------------------------------------------
# Internal low-level request
# ----------------------------------------------------------------------
def _rate_limited_get(url: str, timeout: int = 10) -> Optional[Dict]:
    global _last_call

    # Rate limit: ~1 req per 1.2 seconds
    with _lock:
        elapsed = time.time() - _last_call
        if elapsed < 1.2:
            time.sleep(1.2 - elapsed)
        _last_call = time.time()

    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            
            if resp.status_code == 200:
                _circuit_breaker.record(is_429=False)
                return resp.json()

            if resp.status_code == 429:
                _circuit_breaker.record(is_429=True)
                wait = (2 ** attempt) + 2
                logger.warning("DexScreener 429 — backing off %s seconds (attempt %s)", wait, attempt + 1)
                time.sleep(wait)
                continue

            if resp.status_code == 404:
                logger.debug("DexScreener 404: no Solana pairs for token")
                _circuit_breaker.record(is_429=False)
                return None

            logger.debug("DexScreener HTTP %s: %s", resp.status_code, resp.text[:200])
            return None

        except requests.exceptions.RequestException as e:
            if attempt == 2:
                logger.debug("DexScreener request failed after retries: %s", e)
            else:
                time.sleep(2)
    return None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def fetch_pairs(mint: str) -> Optional[Dict]:
    mint = mint.strip()
    if not mint:
        return None

    if _circuit_breaker.is_open():
        logger.info("DexScreener circuit breaker OPEN — skipping %s", mint)
        return None

    if mint in _cache:
        return _cache[mint]

    future = _executor.submit(_rate_limited_get, DEX_URL.format(mint))
    result = future.result()

    if result and isinstance(result, dict) and "pairs" in result:
        _cache[mint] = result
        logger.debug("Cached DexScreener data for %s", mint[:8])
    elif result is None:
        logger.debug("No DexScreener pairs found for %s", mint[:8])

    return result


def best_pair_summary(mint: str) -> dict:
    js = fetch_pairs(mint)
    if not js or "pairs" not in js or not js["pairs"]:
        return {
            "pairs": [],
            "best_volume_h24": 0.0,
            "best_liquidity_usd": 0.0,
        }

    best_vol = best_liq = 0.0
    summaries = []

    for p in js["pairs"]:
        # Filter only Solana pairs
        if p.get("chainId") != "solana":
            continue

        vol = float(p.get("volume", {}).get("h24", 0) or 0)
        liq = float(p.get("liquidity", {}).get("usd", 0) or 0)
        price = p.get("priceUsd")

        summaries.append({
            "dex": p.get("dexId"),
            "pairAddress": p.get("pairAddress"),
            "quote": p.get("quoteToken", {}).get("symbol", "UNKNOWN"),
            "priceUsd": price,
            "volume_h24": vol,
            "liquidity_usd": liq,
            "url": p.get("url") or f"https://dexscreener.com/solana/{p.get('pairAddress')}",
            "fdv": p.get("fdv"),
            "mc": p.get("marketCap"),
        })
        best_vol = max(best_vol, vol)
        best_liq = max(best_liq, liq)

    return {
        "pairs": summaries,
        "best_volume_h24": best_vol,
        "best_liquidity_usd": best_liq,
    }