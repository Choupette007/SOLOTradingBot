#!/usr/bin/env python3
"""
Reference fix for async price enrichment using aiohttp.

Symptoms in your logs:
  "Price enrichment skipped (error): 'ClientSession' object is not iterable"

Likely cause:
  The code mistakenly tries to iterate over an aiohttp.ClientSession instance
  instead of using it as an async context manager and awaiting requests.

This file provides:
- A small, copy-pasteable function `enrich_prices_async(session, tokens)` that
  demonstrates proper aiohttp usage, timeouts, and per-request error handling.
- A sync wrapper `enrich_prices(tokens)` you can call from existing synchronous code
  using asyncio.run().

Drop this into your codebase (e.g. replace the incorrect enrichment function)
or use as a reference to patch the buggy file.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Iterable, List, Dict, Any, Optional

try:
    import aiohttp
except Exception:
    raise RuntimeError("Please install aiohttp: pip install aiohttp")

logger = logging.getLogger("price-enrich-fix")
_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=10)


async def _fetch_price(session: aiohttp.ClientSession, token_addr: str, endpoint: str) -> Optional[Dict[str, Any]]:
    """
    Fetch price/enrichment for a single token address. Returns parsed JSON or None.
    Replace `endpoint` with the actual Birdeye/Dexscreener endpoint your project uses.
    """
    url = f"{endpoint.rstrip('/')}/{token_addr}"
    try:
        async with session.get(url, timeout=_DEFAULT_TIMEOUT) as resp:
            text = await resp.text()
            if resp.status != 200:
                logger.debug("Enrichment request for %s returned HTTP %s: %s", token_addr, resp.status, text[:200])
                return None
            try:
                return json.loads(text)
            except Exception:
                logger.debug("Failed to parse JSON for %s: %s", token_addr, text[:200])
                return None
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("Enrichment request error for %s: %s", token_addr, e, exc_info=True)
        return None


async def enrich_prices_async(tokens: Iterable[str], endpoint: str, concurrency: int = 10) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Enrich prices for tokens concurrently. Returns a map token -> enrichment (or None).
    - tokens: iterable of token addresses (strings)
    - endpoint: base endpoint for enrichment (e.g., "https://api.birdeye.example/v1/token")
    - concurrency: number of concurrent requests
    """
    tokens_list = list(dict.fromkeys(tokens))  # preserve order, dedupe
    results: Dict[str, Optional[Dict[str, Any]]] = {}

    connector = aiohttp.TCPConnector(limit=max(10, concurrency), force_close=False)
    headers = {
        "User-Agent": "price-enrich-fix/1.0",
        # Add API key header if needed:
        # "Authorization": f"Bearer {os.environ.get('BIRDEYE_API_KEY')}"
    }

    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(connector=connector, headers=headers, timeout=_DEFAULT_TIMEOUT) as session:
        async def _worker(addr: str):
            async with sem:
                val = await _fetch_price(session, addr, endpoint)
                results[addr] = val

        tasks = [asyncio.create_task(_worker(addr)) for addr in tokens_list]
        # Wait and shield from exceptions (we log, but don't let one failure cancel all)
        for t in asyncio.as_completed(tasks):
            try:
                await t
            except Exception as e:
                logger.debug("Price enrichment task failed: %s", e, exc_info=True)

    return results


def enrich_prices(tokens: Iterable[str], endpoint: str, concurrency: int = 10) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Synchronous wrapper: run the async enrichment and return results.
    Use this when the calling code is synchronous.
    """
    return asyncio.run(enrich_prices_async(tokens, endpoint, concurrency=concurrency))


# Quick self-test (run `python tools/fix_price_enrichment_async.py --test` to verify basic loop)
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--endpoint", default="https://api.birdeye.example/v1/token")
    args = p.parse_args()
    if args.test:
        # Dummy tokens to verify concurrency/flow; the endpoint is fake so results will be None
        toklist = ["TokenAddr1", "TokenAddr2", "TokenAddr3"]
        print("Running test enrichment (endpoint=%s) ..." % args.endpoint)
        out = enrich_prices(toklist, args.endpoint, concurrency=3)
        print("Results keys:", list(out.keys()))