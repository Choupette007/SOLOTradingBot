
# optional_checks.py
# Drop-in helper to make Rugcheck optional and add a Birdeye holder-count filter.
# Usage:
#   1) File must be on PYTHONPATH.
#   2) In eligibility code (where we already filter tokens), import:
#        from optional_checks import maybe_enforce_optional_checks, load_api_settings
#      Call inside the token predicate after the basic thresholds pass:
#        api = load_api_settings(cfg, logger)
#        if not await maybe_enforce_optional_checks(token, api, session, logger):
#            return False
#   3) Config keys used (add if missing):
#        discovery:
#          min_holder_count: 0        # 0 disables the holder-count check
#        features:
#          rugcheck_optional: true     # skip Rugcheck if API down / token missing
#        apis:
#          birdeye_api_key_env: BIRDEYE_API_KEY
#          rugcheck_jwt_env: RUGCHECK_JWT
#
# Notes:
#  - If Birdeye API key is missing or a 401 occurs, the holder-count check is skipped (passes).
#  - If Rugcheck is enabled but the call errors or no JWT is present AND features.rugcheck_optional is true,
#    the Rugcheck check is skipped (passes).

from __future__ import annotations

import os
import json
import asyncio
from typing import Any, Dict, Optional

import aiohttp

BIRDEYE_HOLDER_ENDPOINT = "https://public-api.birdeye.so/defi/token/holder_count"

def _cfg_get(d: Dict[str, Any], path: str, default: Any=None) -> Any:
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def load_api_settings(cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    """Resolve API settings & feature flags from config + environment."""
    return {
        "min_holder_count": int(_cfg_get(cfg, "discovery.min_holder_count", 0) or 0),
        "rugcheck_optional": bool(_cfg_get(cfg, "features.rugcheck_optional", True)),
        "birdeye_key": os.getenv(_cfg_get(cfg, "apis.birdeye_api_key_env", "BIRDEYE_API_KEY") or "BIRDEYE_API_KEY"),
        "rugcheck_jwt": os.getenv(_cfg_get(cfg, "apis.rugcheck_jwt_env", "RUGCHECK_JWT") or "RUGCHECK_JWT"),
    }

async def _fetch_holder_count(session: aiohttp.ClientSession, mint: str, api: Dict[str, Any], logger) -> Optional[int]:
    """Return holder count or None if API unavailable/misconfigured."""
    min_req = api.get("min_holder_count", 0)
    if not min_req or min_req <= 0:
        return None  # disabled
    key = api.get("birdeye_key")
    if not key:
        logger.debug("[holders] No Birdeye API key set; skipping holder-count check.")
        return None
    headers = {"X-API-KEY": key, "accept": "application/json"}
    params = {"address": mint, "chain": "solana"}
    try:
        async with session.get(BIRDEYE_HOLDER_ENDPOINT, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 401:
                logger.warning("[holders] Birdeye 401 for %s; skipping holder-count check.", mint)
                return None
            if r.status != 200:
                logger.warning("[holders] Birdeye %s for %s; skipping.", r.status, mint)
                return None
            data = await r.json()
            # Birdeye typically returns {"success":true,"data":{"holder": 12345}} or similar
            # accept a few shapes defensively:
            holder = None
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], dict):
                    holder = data["data"].get("holder") or data["data"].get("holderCount") or data["data"].get("count")
                if holder is None:
                    holder = data.get("holder") or data.get("holderCount") or data.get("count")
            if holder is None:
                logger.debug("[holders] Unexpected Birdeye payload for %s: %s", mint, data)
                return None
            return int(holder)
    except asyncio.TimeoutError:
        logger.warning("[holders] Birdeye timeout for %s; skipping.", mint)
        return None
    except Exception as e:
        logger.warning("[holders] Error for %s: %s; skipping.", mint, e)
        return None

async def _check_rugcheck(session: aiohttp.ClientSession, mint: str, api: Dict[str, Any], logger) -> Optional[bool]:
    """Return True/False if Rugcheck clearly passes/fails, or None to skip."""
    jwt = api.get("rugcheck_jwt")
    if not jwt:
        logger.debug("[rugcheck] No JWT found; %s", "skipping (optional)" if api.get("rugcheck_optional", True) else "enforcing fail")
        return None if api.get("rugcheck_optional", True) else True  # True here means 'do not block'
    url = f"https://api.rugcheck.xyz/v1/tokens/{mint}"
    headers = {"Authorization": f"Bearer {jwt}", "accept": "application/json"}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 401:
                logger.warning("[rugcheck] 401 for %s; treating as skip.", mint)
                return None
            if r.status != 200:
                logger.warning("[rugcheck] %s for %s; treating as skip.", r.status, mint)
                return None
            data = await r.json()
            # Very simple rule: if explicit "is_scam" or risk level critical -> fail; else pass
            is_scam = False
            risk = None
            if isinstance(data, dict):
                is_scam = bool(data.get("is_scam") or data.get("scam") or False)
                risk = (data.get("risk") or data.get("risk_level") or "").lower()
            if is_scam or risk in {"critical", "high"}:
                return False
            return True
    except asyncio.TimeoutError:
        logger.warning("[rugcheck] timeout for %s; treating as skip.", mint)
        return None
    except Exception as e:
        logger.warning("[rugcheck] error for %s: %s; treating as skip.", mint, e)
        return None

async def maybe_enforce_optional_checks(token: Dict[str, Any], api: Dict[str, Any],
                                        session: aiohttp.ClientSession, logger) -> bool:
    """
    Apply optional holder-count and Rugcheck gating.
    Returns True if the token should remain eligible, False if it should be filtered out.
    """
    mint = token.get("address") or token.get("mint") or token.get("token_address")
    if not mint:
        return False  # no address to verify

    # Holder count
    holders = await _fetch_holder_count(session, mint, api, logger)
    min_req = api.get("min_holder_count", 0) or 0
    if holders is not None and min_req > 0 and holders < min_req:
        logger.debug("[holders] %s has %s holders (< %s) → filtered", mint, holders, min_req)
        return False

    # Rugcheck
    rc = await _check_rugcheck(session, mint, api, logger)
    if rc is False:
        logger.debug("[rugcheck] %s flagged by Rugcheck → filtered", mint)
        return False
    # rc is True (explicit pass) or None (skip): keep
    return True
