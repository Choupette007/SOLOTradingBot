# price_enrichment_helper_file.py
# Modified to delegate Birdeye/price enrichment to the package fetching implementation when available,
# falling back to the local defensive implementation otherwise.
#
# Backwards-compatible wrapper added for fetch_birdeye_tokens so callers may use either:
#   - fallback style: fetch_birdeye_tokens(session, max_tokens=...)
#   - delegated style: fetch_birdeye_tokens(session, solana_client, max_tokens=...)
# The wrapper inspects the delegated/fallback callable and dispatches appropriately.

from __future__ import annotations

import os
import re
import time
import asyncio
import aiohttp
import logging
import json
import math
import inspect
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime, timezone

# Prefer solders.Pubkey; fallback to solana.publickey.PublicKey or a regex-only check
try:
    from solders.pubkey import Pubkey  # preferred
except Exception:
    try:
        from solana.publickey import PublicKey as _PublicKey  # type: ignore
        class Pubkey:
            @staticmethod
            def from_string(s: str):
                return _PublicKey(s)
    except Exception:
        _PUBKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
        class Pubkey:
            @staticmethod
            def from_string(s: str):
                if not isinstance(s, str) or not _PUBKEY_RE.match(s):
                    raise ValueError("invalid pubkey")
                return s

from solana_trading_bot_bundle.trading_bot.utils_exec import (
    load_config,
    format_market_cap,
    add_to_blacklist,
    price_cache,
    WHITELISTED_TOKENS,
)

# Optional creation-time cache helpers
try:
    from solana_trading_bot_bundle.trading_bot.utils_exec import get_cached_creation_time, cache_creation_time  # type: ignore
except Exception:
    get_cached_creation_time = None
    cache_creation_time = None

logger = logging.getLogger("PriceEnrichment")
# Do NOT call basicConfig in a library module — leave logging configuration to the app.

# -----------------------------
# Import shared Birdeye cooperative gate (preferred) -- single clean import attempt
# -----------------------------
# Try to reuse a shared birdeye implementation from the packaged fetching module.
# This reduces duplication across modules and ensures consistent behavior.
_fetching_module = None
try:
    from solana_trading_bot_bundle.trading_bot import fetching as _fetching  # type: ignore
    _fetching_module = _fetching
    logger.debug("Delegating price enrichment to packaged fetching module.")
except Exception:
    _fetching_module = None
    # not fatal: we'll fall back to the local implementations below

# Also try to pull a shared birdeye_gate from package birdeye_client if present.
try:
    # Prefer package-qualified import
    from solana_trading_bot_bundle.trading_bot.birdeye_client import birdeye_gate  # type: ignore
    _HAS_SHARED_GATE = True
except Exception:
    try:
        # Fallback if module is on PYTHONPATH at top level
        from birdeye_client import birdeye_gate  # type: ignore
        _HAS_SHARED_GATE = True
    except Exception:
        birdeye_gate = None
        _HAS_SHARED_GATE = False

from contextlib import asynccontextmanager

# -------------------------
# Small helpers
# -------------------------
_BASE58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

def _is_valid_base58_addr(s: Optional[str]) -> bool:
    try:
        if not s or not isinstance(s, str):
            return False
        s = s.strip()
        if not (32 <= len(s) <= 44):
            return False
        return bool(_BASE58_RE.match(s))
    except Exception:
        return False

def _parse_float_like(v: Optional[Any]) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        s = str(v).strip()
        if s == "":
            return None
        s = s.replace(",", "").replace("$", "")
        if s.lower() in ("nan", "null", "none", "inf", "infinity"):
            return None
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None

def _safe_int(v: Any) -> Optional[int]:
    """
    Safely convert untrusted/unknown data to int. Returns None if conversion fails.
    """
    try:
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return int(v)
        s = str(v).strip()
        if s == "":
            return None
        if '.' in s:
            f = float(s)
            if math.isnan(f) or math.isinf(f):
                return None
            return int(f)
        return int(s)
    except Exception:
        return None

def _now_monotonic() -> float:
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        return time.monotonic()

def _redact_key(k: Optional[str]) -> str:
    if not k:
        return "<missing>"
    s = str(k)
    if len(s) <= 10:
        return s[:2] + "..." + s[-2:]
    return s[:4] + "..." + s[-4:]

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

# Robust env-number helpers to tolerate quoted values, inline comments, and stray text.
def _env_float(name: str, default: Any) -> float:
    """
    Robustly parse a float from environment variable `name`.
    Accepts quoted values and strips inline comments after '#' and non-numeric suffixes.
    Falls back to `default` when parsing fails.
    """
    raw = os.getenv(name, None)
    if raw is None:
        try:
            return float(default)
        except Exception:
            return 0.0
    s = str(raw).strip()
    # remove inline comment
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    # strip surrounding quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    # find first numeric token
    m = re.search(r'[-+]?\d+(?:\.\d+)?', s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            pass
    # last attempt: direct cast
    try:
        return float(s)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0

def _env_int(name: str, default: Any) -> int:
    """
    Parse integer from env var, using _env_float as a robust foundation.
    """
    try:
        f = _env_float(name, default)
        return int(f)
    except Exception:
        try:
            return int(default)
        except Exception:
            return 0

# -------------------------
# Tunables & constants
# -------------------------
cfg = load_config() or {}
disc_cfg = cfg.get("discovery", {}) or {}
pe_cfg = cfg.get("price_enrich", {}) or {}

# Use robust env parsing to avoid ValueError on quoted/commented env values
_ENRICH_TOKEN_TIMEOUT_S = _env_float("PRICE_ENRICH_TOKEN_TIMEOUT_S", pe_cfg.get("token_timeout", 6))
_SMALL_BACKOFF_S = _env_float("PRICE_ENRICH_SMALL_BACKOFF_S", pe_cfg.get("small_backoff", 0.2))
_HTTP_TOTAL_TIMEOUT_S = _env_float("PRICE_ENRICH_HTTP_TIMEOUT_S", pe_cfg.get("http_timeout", 8))

_PRICE_CACHE_TTL_POS_S = _env_float("PRICE_CACHE_TTL_S", pe_cfg.get("price_cache_ttl", 60))
_PRICE_CACHE_TTL_NEG_S = _env_float("PRICE_CACHE_NEG_TTL_S", pe_cfg.get("price_cache_neg_ttl", 300))
_PRICE_CACHE_MAX_SIZE  = _env_int("PRICE_CACHE_MAX_SIZE", pe_cfg.get("price_cache_max_size", 5000))

# Birdeye RPS defaults (cap to Starter plan 15 RPS)
_BIRDEYE_RPS_CAP = 15
_env_rps = os.getenv("BIRDEYE_RPS")
_default_rps = int(pe_cfg.get("birdeye_rps", 5))
_BIRDEYE_RPS = max(1, int(_env_rps) if _env_rps is not None and str(_env_rps).isdigit() else _default_rps)
_BIRDEYE_RPS = min(_BIRDEYE_RPS, _BIRDEYE_RPS_CAP)

_BIRDEYE_PAGE_LIMIT = int(os.getenv("BIRDEYE_PAGE_LIMIT", str(pe_cfg.get("page_limit", 100))))
_BIRDEYE_TINY_PROBE_LIMIT = int(os.getenv("BIRDEYE_TINY_PROBE_LIMIT", str(pe_cfg.get("tiny_probe_limit", 5))))
_BIRDEYE_MAX_PAGES_PRIMARY = int(os.getenv("BIRDEYE_MAX_PAGES_PRIMARY", str(pe_cfg.get("max_pages_primary", 6))))
_BIRDEYE_MAX_PAGES_REFILL  = int(os.getenv("BIRDEYE_MAX_PAGES_REFILL", str(pe_cfg.get("max_pages_refill", 4))))

_BIRDEYE_SHORT_CACHE_S = float(os.getenv("BIRDEYE_SHORT_CACHE_S", "30.0"))
_BIRDEYE_DOWN_BACKOFF_DEFAULT = float(os.getenv("BIRDEYE_DOWN_BACKOFF_DEFAULT", "30.0"))
_BIRDEYE_TOKENLIST_PAGE_LIMIT_CAP = 500  # guard upper bound

_DEFAULT_UA = {"Accept": "application/json", "User-Agent": os.getenv("PRICE_ENRICH_UA", "SOLOTradingBot/1.0")}

# ------------------------
# Birdeye: dual support (auto-detect + explicit override)
# ------------------------
BIRDEYE_API_KEY = (os.getenv("BIRDEYE_API_KEY") or os.getenv("BIRDEYE_KEY") or "").strip()
BIRDEYE_MODE = os.getenv("BIRDEYE_MODE", "public").strip().lower()  # "auto"|"pro"|"public"|"off"

def _guess_pro_by_key(key: str) -> bool:
    return bool(key and len(key) >= 30)

if BIRDEYE_MODE == "pro":
    BIRDEYE_PRO = True
elif BIRDEYE_MODE == "off":
    BIRDEYE_PRO = False
elif BIRDEYE_MODE in ("public", "free"):
    BIRDEYE_PRO = False
else:
    BIRDEYE_PRO = _guess_pro_by_key(BIRDEYE_API_KEY)

_user_base = (os.getenv("BIRDEYE_BASE_URL") or "").strip()
if _user_base:
    BIRDEYE_BASE_URL = _user_base.rstrip("/")
else:
    BIRDEYE_BASE_URL = ("https://api.birdeye.so" if BIRDEYE_PRO else "https://public-api.birdeye.so")

# If public and no explicit env override, reduce RPS to be more conservative
if not BIRDEYE_PRO and _env_rps is None:
    _BIRDEYE_RPS = min(_BIRDEYE_RPS, 2)

def _birdeye_headers(api_key: str, pro: bool) -> Dict[str, str]:
    """
    Return canonical headers expected by Birdeye v3 Starter & Pro.
    Keep headers minimal to avoid intermediaries misinterpreting them.
    """
    base = {
        "User-Agent": _DEFAULT_UA["User-Agent"],
        "Accept": "application/json",
        "x-chain": "solana",
    }
    if api_key:
        base["X-API-KEY"] = api_key
    return base

# Normalize whitelist to a set for O(1) checks
try:
    if isinstance(WHITELISTED_TOKENS, (list, tuple, set)):
        WHITELISTED_SET: Set[str] = set(str(x).strip() for x in WHITELISTED_TOKENS if x)
    elif isinstance(WHITELISTED_TOKENS, dict):
        WHITELISTED_SET = set(str(k).strip() for k in WHITELISTED_TOKENS.keys())
    else:
        WHITELISTED_SET = set()
except Exception:
    WHITELISTED_SET = set()

# ------------------------
# Module caches
# ------------------------
_BIRDEYE_CACHE: Dict[str, Any] = {"until": 0.0, "items": []}
_BIRDEYE_CACHE_LOCK = asyncio.Lock()

# Local price cache wrapper (fast in-memory layer) using monotonic timestamps
_local_price_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}

def _price_cache_get(addr: str) -> Optional[Dict[str, Any]]:
    try:
        entry = _local_price_cache.get(addr)
        if entry:
            ts, data = entry
            if (_now_monotonic() - ts) <= _PRICE_CACHE_TTL_POS_S:
                return data
            else:
                _local_price_cache.pop(addr, None)
        # mirror to shared price_cache if it's dict-like
        try:
            if isinstance(price_cache, dict):
                p = price_cache.get(addr)
                if isinstance(p, dict) and "data" in p:
                    return p.get("data")
                return p
        except Exception:
            pass
    except Exception:
        pass
    return None

def _price_cache_put(addr: str, data: Optional[Dict[str, Any]]) -> None:
    try:
        _local_price_cache[addr] = (_now_monotonic(), data)
        try:
            if isinstance(price_cache, dict):
                price_cache[addr] = {"ts": time.time(), "data": data}
        except Exception:
            pass
        if len(_local_price_cache) > _PRICE_CACHE_MAX_SIZE:
            oldest = min(_local_price_cache.items(), key=lambda kv: kv[1][0])[0]
            _local_price_cache.pop(oldest, None)
    except Exception:
        pass

# ------------------------
# Cooperative shutdown
# ------------------------
from threading import Event as _ThreadEvent
_SHUTDOWN = _ThreadEvent()

def signal_shutdown() -> None:
    """
    Signal cooperative shutdown for long-running tasks. Safe to call from sync contexts.
    """
    _SHUTDOWN.set()

def clear_shutdown_signal() -> None:
    """
    Clear previous shutdown signal (useful in tests / interactive sessions).
    """
    _SHUTDOWN.clear()

def _should_stop() -> bool:
    """
    Returns True if a cooperative shutdown has been signalled.
    """
    return _SHUTDOWN.is_set()

# ------------------------
# Utilities
# ------------------------
def _normalize_creation_timestamp(val: Any) -> Optional[int]:
    """
    Normalize various timestamp formats/units to an integer POSIX seconds timestamp.
    Heuristics:
      - If value looks like milliseconds (>= 1e11), divide by 1000.
      - If value looks like microseconds (>= 1e14), divide by 1e6.
      - Otherwise assume seconds.
    Validate against Solana launch and now.
    """
    try:
        v = float(val)
    except Exception:
        return None
    # heuristics
    if v <= 0:
        return None
    # if value looks like microseconds
    if v >= 1e14:
        s = v / 1_000_000.0
    # if value looks like milliseconds
    elif v >= 1e11:
        s = v / 1000.0
    else:
        s = v
    try:
        ts = int(s)
    except Exception:
        return None
    # validate plausible range: after Solana launch, before now + small slack
    solana_launch_s = 1581465600  # 2020-02-12 00:00:00 UTC in seconds
    now_s = int(time.time()) + 60
    if not (solana_launch_s <= ts <= now_s):
        return None
    return ts

from contextlib import asynccontextmanager

@asynccontextmanager
async def _birdeye_gate_request(pacing_no_gate: float = 0.02):
    """
    Defensive wrapper to enter birdeye_gate.request() as an async context manager.

    - Handles birdeye_gate being None.
    - If birdeye_gate.request() returns a coroutine, await it to obtain the context manager.
    - If request() already returns an async context manager, use it directly.
    - Falls back to a no-op context (with optional tiny sleep) on unexpected errors.
    """
    # No shared gate: small conservative sleep to avoid bursts (configurable)
    if birdeye_gate is None:
        try:
            if pacing_no_gate:
                await asyncio.sleep(pacing_no_gate)
        except Exception:
            pass

        # yield control to caller; nothing to clean up on exit
        try:
            yield
        finally:
            # no explicit return from finally (avoid the Pylance diagnostic)
            # noop cleanup
            pass
        # function exits here

    # If we do have a shared gate, try to use it and fall back to a paced no-op.
    try:
        req = birdeye_gate.request()
        # If request() returned a coroutine, await it to get the actual async context manager
        if asyncio.iscoroutine(req):
            cm = await req
        else:
            cm = req
        # Enter the context manager produced by the gate
        async with cm:
            yield
    except Exception:
        # Defensive fallback: allow the caller to proceed without the gate, with tiny pacing
        try:
            if pacing_no_gate:
                await asyncio.sleep(pacing_no_gate)
        except Exception:
            pass

        try:
            yield
        finally:
            # noop cleanup; do not use `return` inside finally
            pass

# -----------------------------------------------------------------------------
# Delegation layer: if the packaged fetching module provides these helpers, use them.
# This prevents duplicate implementations and keeps behavior consistent across modules.
# -----------------------------------------------------------------------------
_fetch_delegate_ok = False
if _fetching_module is not None:
    try:
        _dfetch_fetch_birdeye_tokens = getattr(_fetching_module, "fetch_birdeye_tokens", None)
        _dfetch_fetch_birdeye_creation_time = getattr(_fetching_module, "fetch_birdeye_creation_time", None)
        _dfetch_fetch_birdeye_price = getattr(_fetching_module, "fetch_birdeye_price", None)
        _dfetch_fetch_birdeye_multi_price = getattr(_fetching_module, "fetch_birdeye_multi_price", None)
        _dfetch_enrich_tokens_with_price_change = getattr(_fetching_module, "enrich_tokens_with_price_change", None)

        # We consider delegation available if at least enrich_tokens_with_price_change exists,
        # or the multi_price + single price functions are present.
        if callable(_dfetch_enrich_tokens_with_price_change) or callable(_dfetch_fetch_birdeye_multi_price) or callable(_dfetch_fetch_birdeye_price):
            _fetch_delegate_ok = True
            # Export delegated references (these names are used by other modules)
            fetch_birdeye_tokens = _dfetch_fetch_birdeye_tokens
            fetch_birdeye_creation_time = _dfetch_fetch_birdeye_creation_time
            fetch_birdeye_price = _dfetch_fetch_birdeye_price
            fetch_birdeye_multi_price = _dfetch_fetch_birdeye_multi_price
            enrich_tokens_with_price_change = _dfetch_enrich_tokens_with_price_change
            logger.debug("Price enrichment functions delegated to packaged fetching module.")
    except Exception:
        _fetch_delegate_ok = False

# -----------------------------------------------------------------------------
# If delegation was not possible, define the local defensive implementations.
# (These implementations are the same robust code the module had previously.)
# -----------------------------------------------------------------------------
if not _fetch_delegate_ok:

    # ------------------------
    # fetch_birdeye_tokens (defensive, fallback-capable)
    # ------------------------
    async def fetch_birdeye_tokens(
        session: aiohttp.ClientSession,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch tokens from Birdeye with robust behavior:
          - Try pro path (/defi/v3/token/list) first, then public alternatives.
          - Use minimal params for public endpoints; retry with simplified params on 400.
          - Respect Retry-After on 429, mark Birdeye down and back off through shared gate.
          - Short-term cache to avoid bursts.
        """
        if _should_stop():
            return []

        if BIRDEYE_MODE == "off":
            return []

        api_key = BIRDEYE_API_KEY
        target = max(1, int(max_tokens or int(pe_cfg.get("birdeye_max_tokens", 250))))
        page_limit_cfg = int(pe_cfg.get("birdeye_page_limit", _BIRDEYE_PAGE_LIMIT))
        # smaller pages for public endpoints to avoid rejections
        if not BIRDEYE_PRO:
            page_limit = max(10, min(50, page_limit_cfg))
        else:
            page_limit = max(1, min(page_limit_cfg, _BIRDEYE_TOKENLIST_PAGE_LIMIT_CAP, target))

        max_pages_cfg = int(pe_cfg.get("birdeye_max_pages", _BIRDEYE_MAX_PAGES_PRIMARY))
        needed_pages = max(1, math.ceil(target / page_limit))
        max_pages = min(max_pages_cfg, needed_pages)
        success_floor = min(max(12, page_limit), target)

        base_host = BIRDEYE_BASE_URL.rstrip('/')
        endpoint_candidates = [
            f"{base_host}/defi/v3/token/list",   # pro / v3 path
            f"{base_host}/defi/tokenlist",       # public older path
            f"{base_host}/defi/token/list",      # alternative
            f"{base_host}/defi/v2/token/list",
        ]

        headers = _birdeye_headers(api_key, BIRDEYE_PRO)
        timeout = aiohttp.ClientTimeout(total=20, sock_connect=6, sock_read=10)

        async with _BIRDEYE_CACHE_LOCK:
            now_ts = time.time()
            if now_ts < float(_BIRDEYE_CACHE.get("until", 0.0)) and _BIRDEYE_CACHE.get("items"):
                cached = _BIRDEYE_CACHE["items"]
                logger.debug("Birdeye: served %d items from short cache", len(cached))
                return cached[:target]

        def make_params(limit: int, offset: int, minimal: bool = True):
            params = {"chain": "solana", "limit": limit, "offset": offset}
            if not minimal:
                # only add discovery filters if configured and present
                try:
                    min_liq = int(disc_cfg.get("min_liquidity", 0) or 0)
                    if min_liq > 0:
                        params["min_liquidity"] = min_liq
                except Exception:
                    pass
                try:
                    min_vol = int(disc_cfg.get("min_volume_24h_usd", 0) or 0)
                    if min_vol > 0:
                        params["min_volume_24h_usd"] = min_vol
                except Exception:
                    pass
            return params

        items: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        minimal = True
        stop_all = False  # when True, stop trying more endpoints (e.g., after 429/backoff)

        # page through candidates until we have enough
        for endpoint in endpoint_candidates:
            if _should_stop() or stop_all:
                break
            # modest pause to avoid immediate bursting between endpoints
            await asyncio.sleep(0.02)

            offset = 0
            page = 0
            error_count = 0
            while page < max_pages and len(items) < target and not stop_all and not _should_stop():
                params = make_params(page_limit, offset, minimal=minimal)
                try:
                    # cooperative acquire per-request (preferred)
                    async with _birdeye_gate_request():
                        resp = await session.get(endpoint, headers=headers, params=params, timeout=timeout)

                    async with resp as r:
                        if r.status == 429:
                            retry_after = r.headers.get("Retry-After") or r.headers.get("retry-after")
                            try:
                                wait = float(retry_after) if retry_after is not None else _BIRDEYE_DOWN_BACKOFF_DEFAULT
                            except Exception:
                                wait = _BIRDEYE_DOWN_BACKOFF_DEFAULT
                            wait = max(1.0, min(300.0, wait))
                            # Cooperatively mark the shared gate down (preferred)
                            if birdeye_gate is not None:
                                try:
                                    await birdeye_gate.note_429()
                                    await birdeye_gate.mark_down(wait)
                                except Exception:
                                    logger.debug("birdeye_gate.mark_down/note_429 failed", exc_info=True)
                            else:
                                # fallback: sleep locally to avoid hammering
                                logger.debug("No shared gate; local sleep for %.1fs", wait)
                            logger.warning("Birdeye 429 on %s; backing off %.1fs", endpoint, wait)
                            # stop trying endpoints for now
                            stop_all = True
                            break

                        if r.status == 400:
                            # Public endpoints may reject optional params; retry with minimal params once
                            if not minimal:
                                logger.warning("Birdeye 400 for %s params=%s; retrying with minimal params", endpoint, params)
                                minimal = True
                                continue
                            else:
                                logger.debug("Birdeye 400 for %s params=%s", endpoint, params)
                                error_count += 1
                                break
                        if r.status != 200:
                            body = (await r.text())[:500]
                            logger.debug("Birdeye page HTTP %s on %s: %s", r.status, endpoint, body)
                            error_count += 1
                            break

                        data = await r.json(content_type=None)
                        payload = data.get("data") or data
                        added_page: List[Dict[str, Any]] = []
                        if isinstance(payload, dict):
                            raw = payload.get("items") or payload.get("tokens") or payload.get("results") or []
                            if isinstance(raw, list):
                                for it in raw:
                                    if isinstance(it, dict):
                                        addr = (it.get("address") or it.get("tokenAddress") or it.get("token_address") or "").strip()
                                        if not addr:
                                            continue
                                        if addr in seen:
                                            continue
                                        seen.add(addr)
                                        added_page.append(it)
                        elif isinstance(payload, list):
                            for it in payload:
                                if isinstance(it, dict):
                                    addr = (it.get("address") or it.get("tokenAddress") or it.get("token_address") or "").strip()
                                    if not addr:
                                        continue
                                    if addr in seen:
                                        continue
                                    seen.add(addr)
                                    added_page.append(it)
                        if added_page:
                            items.extend(added_page)
                        else:
                            # empty page -> stop paging this endpoint
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.debug("Birdeye request failed for %s: %s", endpoint, e, exc_info=True)
                    error_count += 1
                    if error_count > 2:
                        break
                except Exception as ex:
                    logger.exception("Unexpected error while fetching Birdeye pages from %s: %s", endpoint, ex)
                    break
                finally:
                    offset += page_limit
                    page += 1
                    # modest pacing between pages
                    await asyncio.sleep(0.18 if not BIRDEYE_PRO else 0.10)

            if len(items) >= success_floor:
                logger.info("Birdeye: endpoint %s produced %d items (floor=%d)", endpoint, len(items), success_floor)
                break
            logger.debug("Birdeye: endpoint %s produced %d items; trying next host", endpoint, len(items))

        # final attempt: try the primary endpoint with minimal params one more time
        if not stop_all and len(items) < success_floor and endpoint_candidates:
            endpoint = endpoint_candidates[0]
            try:
                params = make_params(page_limit, 0, minimal=True)
                async with _birdeye_gate_request():
                    resp = await session.get(endpoint, headers=headers, params=params, timeout=timeout)

                async with resp as r:
                    if r.status == 200:
                        data = await r.json(content_type=None)
                        payload = data.get("data") or data
                        li: List[Dict[str, Any]] = []
                        if isinstance(payload, dict):
                            raw = payload.get("items") or payload.get("tokens") or []
                            if isinstance(raw, list):
                                for it in raw:
                                    if isinstance(it, dict):
                                        addr = (it.get("address") or it.get("tokenAddress") or it.get("token_address") or "").strip()
                                        if not addr:
                                            continue
                                        if addr in seen:
                                            continue
                                        seen.add(addr)
                                        li.append(it)
                        elif isinstance(payload, list):
                            for it in payload:
                                if isinstance(it, dict):
                                    addr = (it.get("address") or it.get("tokenAddress") or it.get("token_address") or "").strip()
                                    if not addr:
                                        continue
                                    if addr in seen:
                                        continue
                                    seen.add(addr)
                                    li.append(it)
                        if li:
                            items.extend(li)
                            logger.info("Birdeye auto-fill added %d item(s); raw total=%d (floor=%d).", len(li), len(items), success_floor)
            except Exception:
                logger.debug("Final auto-fill attempt failed", exc_info=True)

        if not items:
            logger.warning("Birdeye returned no items this cycle")
            return []

        # build token dicts
        tokens: List[Dict[str, Any]] = []
        now_ts_int = int(time.time())
        for it in items[:target]:
            try:
                token_address = (it.get("address") or it.get("tokenAddress") or it.get("token_address") or "").strip()
                if not token_address:
                    continue
                symbol = it.get("symbol") or it.get("tokenSymbol") or "UNKNOWN"
                name = it.get("name") or symbol
                vol24 = it.get("volume_24h_usd") or it.get("v24hUSD") or it.get("volume24hUSD") or 0
                liq = it.get("liquidity") or it.get("liquidity_usd") or 0
                mc = it.get("market_cap") or it.get("mc") or 0
                raw_price = it.get("price") if it.get("price") is not None else it.get("priceUsd")
                p_val = _parse_float_like(raw_price)
                final_price = float(p_val) if (p_val is not None and p_val > 0) else None

                created_ts = 0
                recent_listing = it.get("recent_listing_time") or it.get("recentListingTime") or None
                if recent_listing:
                    try:
                        norm = _normalize_creation_timestamp(recent_listing)
                        created_ts = norm or 0
                    except Exception:
                        created_ts = 0

                tokens.append({
                    "address": token_address,
                    "token_address": token_address,
                    "symbol": symbol,
                    "name": name,
                    "volume_24h": float(_parse_float_like(vol24) or 0.0),
                    "liquidity": float(_parse_float_like(liq) or 0.0),
                    "market_cap": float(_parse_float_like(mc) or 0.0),
                    "creation_timestamp": created_ts or 0,
                    "timestamp": now_ts_int,
                    "categories": ["no_creation_time"] if (created_ts is None or created_ts == 0) else [],
                    "price": final_price,
                    "price_change_1h": _parse_float_like(it.get("price_change_1h") or it.get("v1h") or None),
                    "price_change_6h": _parse_float_like(it.get("price_change_6h") or it.get("v6h") or None),
                    "price_change_24h": _parse_float_like(it.get("price_change_24h") or it.get("v24h") or None),
                    "source": "birdeye",
                })
            except Exception:
                logger.debug("Skipping malformed item in Birdeye results", exc_info=True)
                continue

        async with _BIRDEYE_CACHE_LOCK:
            _BIRDEYE_CACHE["until"] = time.time() + max(_BIRDEYE_SHORT_CACHE_S, 30.0)
            _BIRDEYE_CACHE["items"] = list(tokens)

        if len(tokens) >= success_floor:
            logger.info("Birdeye PASS: fetched %d ≥ %d tokens.", len(tokens), success_floor)
        else:
            logger.warning("Birdeye SOFT-OK: fetched %d < %d tokens (rate/host limits likely).", len(tokens), success_floor)

        return tokens[:target]

    # ------------------------
    # fetch_birdeye_creation_time
    # ------------------------
    async def fetch_birdeye_creation_time(token_address: str, session: aiohttp.ClientSession) -> Optional[datetime]:
        """
        Query Birdeye for creation/launch time (best-effort).
        """
        if _should_stop():
            return None
        if token_address in WHITELISTED_SET:
            return None
        if not _is_valid_base58_addr(token_address):
            return None

        if callable(get_cached_creation_time):
            try:
                cached = await get_cached_creation_time(token_address)
                if cached:
                    return cached
            except Exception:
                logger.debug("get_cached_creation_time failed", exc_info=True)

        api_key = BIRDEYE_API_KEY
        if BIRDEYE_PRO and not api_key:
            return None

        headers = _birdeye_headers(api_key, BIRDEYE_PRO)
        timeout = aiohttp.ClientTimeout(total=12, sock_connect=6, sock_read=8)
        creation_url = f"{BIRDEYE_BASE_URL.rstrip('/')}/defi/token_creation_info"

        attempts = 0
        backoff = 0.8
        while attempts < 3 and not _should_stop():
            attempts += 1
            try:
                async with _birdeye_gate_request():
                    resp = await session.get(creation_url, headers=headers, params={"address": token_address}, timeout=timeout)

                async with resp as r:
                    if r.status == 429:
                        # cooperative down with shared gate
                        retry_after = r.headers.get("Retry-After") or r.headers.get("retry-after")
                        try:
                            wait = float(retry_after) if retry_after is not None else 15.0
                        except Exception:
                            wait = 15.0
                        wait = max(1.0, min(300.0, wait))
                        if birdeye_gate is not None:
                            try:
                                await birdeye_gate.note_429()
                                await birdeye_gate.mark_down(wait)
                            except Exception:
                                logger.debug("birdeye_gate.mark_down/note_429 failed", exc_info=True)
                        else:
                            logger.debug("No shared gate; sleeping %.1fs", wait)
                        logger.warning("Birdeye token_creation_info 429 for %s; backing off %.1fs", token_address, wait)
                        return None
                    if r.status == 404:
                        if callable(cache_creation_time):
                            try:
                                await cache_creation_time(token_address, None)
                            except Exception:
                                logger.debug("Negative cache failed", exc_info=True)
                        return None
                    if r.status != 200:
                        logger.debug("Birdeye creation info HTTP %s for %s", r.status, token_address)
                        if r.status in (401, 403):
                            return None
                        await asyncio.sleep(backoff)
                        backoff = min(10.0, backoff * 2.0)
                        continue
                    data_obj = await r.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug("Birdeye creation_time transient error for %s: %s", token_address, e)
                await asyncio.sleep(backoff)
                backoff = min(10.0, backoff * 2.0)
                continue
            except Exception as e:
                logger.exception("Birdeye creation_time unexpected error for %s: %s", token_address, e)
                return None

            if not isinstance(data_obj, dict):
                return None

            d = data_obj.get("data") or data_obj
            created_val: Optional[float] = None

            for key in ("creation_time", "created_timestamp", "createdTimestamp", "createdAt", "mintTime", "created_time", "created_at"):
                try:
                    val = d.get(key) if isinstance(d, dict) else None
                    if val is not None:
                        created_val = float(val)
                        break
                except Exception:
                    continue

            if created_val is None:
                return None

            norm_ts = _normalize_creation_timestamp(created_val)
            if norm_ts is None:
                return None

            dt = datetime.fromtimestamp(norm_ts, tz=timezone.utc)
            if callable(cache_creation_time):
                try:
                    await cache_creation_time(token_address, dt)
                except Exception:
                    logger.debug("cache_creation_time failed", exc_info=True)
            return dt

        return None

    # ------------------------
    # fetch_birdeye_price
    # ------------------------
    async def fetch_birdeye_price(session: aiohttp.ClientSession, addr: str) -> Optional[Dict[str, Any]]:
        """
        Query Birdeye price endpoint for a single token.
        """
        if BIRDEYE_MODE == "off":
            return None
        if BIRDEYE_PRO and not BIRDEYE_API_KEY:
            return None
        if not _is_valid_base58_addr(addr):
            return None

        cached = _price_cache_get(addr)
        if cached is not None:
            return cached

        base = BIRDEYE_BASE_URL.rstrip('/')
        url = f"{base}/defi/price"
        headers = _birdeye_headers(BIRDEYE_API_KEY, BIRDEYE_PRO)
        timeout = aiohttp.ClientTimeout(total=_HTTP_TOTAL_TIMEOUT_S)
        backoff = 0.8
        attempts = 0
        while attempts < 3 and not _should_stop():
            attempts += 1
            try:
                async with _birdeye_gate_request():
                    resp = await session.get(url, headers=headers, params={"address": addr}, timeout=timeout)

                async with resp as r:
                    text = await r.text()
                    if r.status in (401, 403):
                        logger.warning("Birdeye auth error (%s) — key redacted=%s", r.status, _redact_key(BIRDEYE_API_KEY))
                        return None
                    if r.status == 429:
                        ra = r.headers.get("Retry-After") or r.headers.get("retry-after")
                        try:
                            ra_val = float(ra) if ra else None
                        except Exception:
                            ra_val = None
                        sleep_s = min(10.0, ra_val or backoff)
                        sleep_s = max(1.0, sleep_s)
                        # cooperative backoff
                        if birdeye_gate is not None:
                            try:
                                await birdeye_gate.note_429()
                                await birdeye_gate.mark_down(sleep_s)
                            except Exception:
                                logger.debug("birdeye_gate.mark_down/note_429 failed", exc_info=True)
                        else:
                            logger.debug("No shared gate; local sleep for %.1fs", sleep_s)
                        logger.warning("Birdeye 429 for %s; backing off %.1fs", addr, sleep_s)
                        await asyncio.sleep(sleep_s)
                        backoff = min(10.0, backoff * 2.0)
                        continue
                    if r.status == 404:
                        _price_cache_put(addr, None)
                        return None
                    if r.status != 200:
                        logger.debug("Birdeye HTTP %s for %s: %s", r.status, addr, (text or "")[:300])
                        await asyncio.sleep(backoff)
                        backoff = min(10.0, backoff * 2.0)
                        continue

                    try:
                        data = json.loads(text)
                    except Exception:
                        data = await r.json(content_type=None)
                    pd = data.get("data") or data
                    price = _parse_float_like(pd.get("value") or pd.get("price") or pd.get("priceUsd") or pd.get("price_usd"))
                    if price is None:
                        _price_cache_put(addr, None)
                        return None
                    pc1 = _parse_float_like(pd.get("priceChange1h") or pd.get("price_change_1h") or pd.get("v1h")) or 0.0
                    pc6 = _parse_float_like(pd.get("priceChange6h") or pd.get("price_change_6h") or pd.get("v6h")) or 0.0
                    pc24 = _parse_float_like(pd.get("priceChange24h") or pd.get("price_change_24h") or pd.get("v24h")) or 0.0
                    result = {
                        "price": float(price),
                        "price_change_1h": float(pc1),
                        "price_change_6h": float(pc6),
                        "price_change_24h": float(pc24),
                    }
                    _price_cache_put(addr, result)
                    return result
            except (asyncio.TimeoutError, aiohttp.ClientConnectionError) as e:
                logger.debug("Birdeye transient error for %s: %s", addr, e)
                await asyncio.sleep(backoff)
                backoff = min(10.0, backoff * 2.0)
                continue
            except Exception as e:
                logger.exception("Birdeye unexpected error for %s: %s", addr, e)
                return None
        return None

    # ------------------------
    # fetch_birdeye_multi_price (new)
    # ------------------------
            
    async def fetch_birdeye_multi_price(session: aiohttp.ClientSession, addrs: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        out: Dict[str, Optional[Dict[str, Any]]] = {}
        if not addrs:
            return out
        if BIRDEYE_MODE == "off":
            for a in addrs:
                out[a] = None
            return out
        addresses = [a for a in dict.fromkeys(addrs) if _is_valid_base58_addr(a)]
        if not addresses:
            for a in addrs:
                out[a] = None
            return out

        base = BIRDEYE_BASE_URL.rstrip("/")
        url = f"{base}/defi/multi_price"
        headers = _birdeye_headers(BIRDEYE_API_KEY, BIRDEYE_PRO)
        timeout = aiohttp.ClientTimeout(total=_HTTP_TOTAL_TIMEOUT_S)
        max_per_call = 100  # per Starter plan
        chunk_sleep = max(0.0, (1.0 / float(max(1, _BIRDEYE_RPS))) - 0.01) if _BIRDEYE_RPS > 0 else 0.05

        async def _do_request(params: Dict[str, Any]) -> Tuple[int, Any, Optional[str], Dict[str, str]]:
            try:
                async with _birdeye_gate_request():
                    resp = await session.get(url, headers=headers, params=params, timeout=timeout)
                async with resp as r:
                    text = await r.text()
                    status = r.status
                    headers_out = dict(r.headers)
                    try:
                        data = await r.json(content_type=None)
                    except Exception:
                        data = None
                    return status, (data if data is not None else text), (text[:400]), headers_out
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug("Birdeye multi_price transient error for params=%s: %s", params, e, exc_info=True)
                raise

        for i in range(0, len(addresses), max_per_call):
            chunk = addresses[i:i + max_per_call]
            params_preferred = {"ui_amount_mode": "raw", "list_address": ",".join(chunk)}

            try:
                status, payload, text_head, resp_headers = await _do_request(params_preferred)
            except Exception:
                for a in chunk:
                    out[a] = None
                await asyncio.sleep(chunk_sleep)
                continue

            # If primary response is a 429, handle backoff and stop
            if status == 429:
                retry_after = None
                try:
                    retry_after = float(resp_headers.get("Retry-After") or resp_headers.get("retry-after"))
                except Exception:
                    pass
                wait = max(1.0, min(300.0, retry_after or _BIRDEYE_DOWN_BACKOFF_DEFAULT))
                if birdeye_gate is not None:
                    try:
                        await birdeye_gate.note_429()
                        await birdeye_gate.mark_down(wait)
                    except Exception:
                        logger.debug("birdeye_gate.mark_down/note_429 failed", exc_info=True)
                logger.warning("Birdeye multi_price 429; backing off %.1fs", wait)
                for a in chunk:
                    out[a] = None
                break

            # If non-200, detect param-format message and try legacy param name as fallback
            if status != 200:
                resp_text = text_head or ""
                try:
                    msg = None
                    if isinstance(payload, dict):
                        msg = payload.get("message") or payload.get("error") or None
                    if msg and "list_address" in str(msg).lower():
                        params_alt = {"ui_amount_mode": "raw", "addresses": ",".join(chunk)}
                        try:
                            status2, payload2, text_head2, resp_headers2 = await _do_request(params_alt)
                            status, payload, text_head, resp_headers = status2, payload2, text_head2, resp_headers2
                        except Exception:
                            for a in chunk:
                                out[a] = None
                            await asyncio.sleep(chunk_sleep)
                            continue
                except Exception:
                    pass

            # If 429 after fallback
            if status == 429:
                retry_after = None
                try:
                    retry_after = float(resp_headers.get("Retry-After") or resp_headers.get("retry-after"))
                except Exception:
                    pass
                wait = max(1.0, min(300.0, retry_after or _BIRDEYE_DOWN_BACKOFF_DEFAULT))
                if birdeye_gate is not None:
                    try:
                        await birdeye_gate.note_429()
                        await birdeye_gate.mark_down(wait)
                    except Exception:
                        logger.debug("birdeye_gate.mark_down/note_429 failed", exc_info=True)
                logger.warning("Birdeye multi_price 429 on fallback; backing off %.1fs", wait)
                for a in chunk:
                    out[a] = None
                break

            if status != 200:
                logger.debug("Birdeye multi_price HTTP %s for chunk start %d: %s", status, i, (text_head or "")[:300])
                for a in chunk:
                    out[a] = None
                await asyncio.sleep(chunk_sleep)
                continue

            # At this point we have a 200 and payload should be dict mapping addresses -> info
            payload_obj = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
            if not isinstance(payload_obj, dict):
                for a in chunk:
                    out[a] = None
                await asyncio.sleep(chunk_sleep)
                continue

            for a in chunk:
                info = payload_obj.get(a) or payload_obj.get(a.strip()) or None
                if not info:
                    out[a] = None
                    continue
                try:
                    value = _parse_float_like(info.get("value") or info.get("price") or info.get("priceUsd") or info.get("price_usd"))
                    pc24 = _parse_float_like(info.get("priceChange24h") or info.get("price_change_24h") or info.get("priceChange24h") or info.get("priceChange24H")) or 0.0
                    liq = _parse_float_like(info.get("liquidity") or info.get("liquidity_usd") or info.get("liquidityUsd")) or 0.0
                    update_unix = _safe_int(info.get("updateUnixTime") or info.get("update_unix") or info.get("updateUnix")) or None
                    if value is None:
                        out[a] = None
                    else:
                        out[a] = {
                            "price": float(value),
                            "price_change_24h": float(pc24),
                            "liquidity": float(liq),
                            "update_unix": int(update_unix) if update_unix is not None else None,
                        }
                except Exception:
                    out[a] = None

            await asyncio.sleep(chunk_sleep)

        # ensure all original addrs are present in out
        for a in addrs:
            if a not in out:
                out[a] = None
        return out

    # ------------------------
    # enrich_tokens_with_price_change (defensive, single implementation)
    # ------------------------
    async def enrich_tokens_with_price_change(
        tokens,
        session: Optional[aiohttp.ClientSession] = None,
        concurrency: int = 25,
        per_token_timeout: float = 7.0,
        *args,
        **kwargs,
    ):
        """
        Enrich a list of token dicts with Birdeye price data.

        Defensive behavior:
        - Recover from positional-arg swaps (session/tokens).
        - Normalize single dict -> list.
        - Clamp concurrency relative to Birdeye RPS to reduce 429s.
        - Best-effort enrich creation_timestamp.
        """
        # Detect and recover if caller accidentally passed session as first positional arg
        try:
            is_client_session_first = isinstance(tokens, aiohttp.ClientSession)
        except Exception:
            is_client_session_first = False

        if is_client_session_first:
            # If second param looks like tokens list, swap them
            if isinstance(session, (list, tuple)):
                logger.warning(
                    "enrich_tokens_with_price_change: detected positional-args swap (ClientSession passed as first arg). Swapping."
                )
                tokens, session = session, tokens
            else:
                logger.warning(
                    "enrich_tokens_with_price_change: called with ClientSession as first argument and no tokens provided. Skipping."
                )
                return tokens

        # Accept single dict -> list
        if isinstance(tokens, dict):
            tokens = [tokens]

        if not tokens:
            return tokens

        # Build address -> token mapping
        addr_to_tokens: Dict[str, List[Dict[str, Any]]] = {}
        for t in tokens:
            if not isinstance(t, dict):
                continue
            addr = (t.get("address") or t.get("token_address") or "").strip()
            if not addr:
                continue
            addr_to_tokens.setdefault(addr, []).append(t)
        addrs = list(addr_to_tokens.keys())
        if not addrs:
            return tokens

        # Simplified defensive concurrency normalization
        try:
            c_val = _parse_float_like(concurrency)
            if c_val is None:
                concurrency = int(pe_cfg.get("default_concurrency", 25))
            else:
                concurrency = max(1, int(c_val))
        except Exception:
            concurrency = int(pe_cfg.get("default_concurrency", 25))

        # Normalize per_token_timeout
        try:
            pt_val = _parse_float_like(per_token_timeout)
            if pt_val is None:
                per_token_timeout = float(pe_cfg.get("per_token_timeout", 7.0))
            else:
                per_token_timeout = float(pt_val)
        except Exception:
            per_token_timeout = float(pe_cfg.get("per_token_timeout", 7.0))

        # clamp concurrency relative to RPS to avoid many 429s
        concurrency = max(1, int(concurrency))
        concurrency = min(concurrency, max(1, _BIRDEYE_RPS * 3))

        sem = asyncio.Semaphore(concurrency)
        results: Dict[str, Optional[Dict[str, Any]]] = {}

        # use caller session if provided, otherwise create and close our own
        own_session = False
        if session is None:
            own_session = True
            session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=6, sock_read=6))

        def should_stop() -> bool:
            return callable(globals().get("_should_stop")) and _should_stop()

        # FIRST: attempt bulk multi_price call to cover many addresses in fewer requests
        try:
            try:
                bulk_map = await fetch_birdeye_multi_price(session, addrs)
            except Exception:
                bulk_map = {}
            # apply bulk_map to results
            for addr in addrs:
                if addr in bulk_map and bulk_map[addr] is not None:
                    bm = bulk_map[addr]
                    # map bulk fields into the same shape used by per-token fetch
                    results[addr] = {
                        "price": float(bm.get("price")) if bm.get("price") is not None else None,
                        "price_change_1h": 0.0,
                        "price_change_6h": 0.0,
                        "price_change_24h": float(bm.get("price_change_24h") or 0.0),
                    }
                elif addr in bulk_map and bulk_map[addr] is None:
                    results[addr] = None
            # remove addresses already resolved by bulk from the remaining list
            remaining_addrs = [a for a in addrs if a not in results or results[a] is None]
        except Exception:
            remaining_addrs = addrs[:]

        # If there are remaining addresses that bulk did not resolve, fetch them individually (constrained by semaphore)
        async def _fetch_price_for(addr: str) -> Tuple[str, Optional[Dict[str, Any]]]:
            if should_stop():
                return (addr, None)
            try:
                async with sem:
                    # if a bulk result exists and is None, still allow per-token fetch to try (maybe bulk missed it)
                    coro = fetch_birdeye_price(session, addr)
                    if per_token_timeout and per_token_timeout > 0:
                        try:
                            res = await asyncio.wait_for(coro, timeout=per_token_timeout)
                        except asyncio.TimeoutError:
                            logger.debug("Price fetch timeout for %s", addr)
                            return (addr, None)
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logger.debug("Price fetch error for %s: %s", addr, e, exc_info=True)
                            return (addr, None)
                    else:
                        try:
                            res = await coro
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logger.debug("Price fetch error for %s: %s", addr, e, exc_info=True)
                            return (addr, None)
                    return (addr, res)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("Price fetch unexpected error for %s: %s", addr, e, exc_info=True)
                return (addr, None)

        # schedule tasks only for remaining_addrs that were not satisfied by bulk (or where bulk returned None)
        to_query = [a for a in addrs if a not in results or results[a] is None]
        tasks = [asyncio.create_task(_fetch_price_for(a)) for a in to_query]
        try:
            for fut in asyncio.as_completed(tasks):
                if should_stop():
                    break
                try:
                    addr, pdata = await fut
                    # only overwrite if we don't already have a positive bulk result
                    results[addr] = pdata
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("Error awaiting a price task", exc_info=True)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass

        # apply price results
        for addr, toks in addr_to_tokens.items():
            pdata = results.get(addr)
            if pdata and isinstance(pdata, dict):
                p = pdata.get("price")
                pc1 = pdata.get("price_change_1h")
                pc6 = pdata.get("price_change_6h")
                pc24 = pdata.get("price_change_24h")
                for t in toks:
                    try:
                        if p is not None:
                            t["price"] = float(p)
                        if pc1 is not None:
                            t["price_change_1h"] = float(pc1)
                        if pc6 is not None:
                            t["price_change_6h"] = float(pc6)
                        if pc24 is not None:
                            t["price_change_24h"] = float(pc24)
                    except Exception:
                        logger.debug("Failed to apply price to token %s", t.get("address"), exc_info=True)
            else:
                for t in toks:
                    try:
                        t.setdefault("price", None)
                        t.setdefault("price_change_1h", 0.0)
                        t.setdefault("price_change_6h", 0.0)
                        t.setdefault("price_change_24h", 0.0)
                    except Exception:
                        pass

        # Optionally enrich creation_time for tokens that lack it (best-effort)
        enrich_creation = bool(pe_cfg.get("enrich_creation_time", True))
        if enrich_creation:
            missing_ct: List[str] = []
            for tok in tokens:
                if not isinstance(tok, dict):
                    continue
                addr = (tok.get("address") or tok.get("token_address") or "").strip()
                if not addr:
                    continue
                if not tok.get("creation_timestamp"):
                    missing_ct.append(addr)
            missing_ct = list(dict.fromkeys(missing_ct))
            if missing_ct:
                ct_sem = asyncio.Semaphore(max(1, min(4, max(1, concurrency // 2))))
                async def _fetch_ct(addr: str) -> Tuple[str, Optional[datetime]]:
                    if should_stop():
                        return (addr, None)
                    try:
                        async with ct_sem:
                            coro = fetch_birdeye_creation_time(addr, session)
                            try:
                                dt = await asyncio.wait_for(coro, timeout=per_token_timeout)
                            except asyncio.TimeoutError:
                                dt = None
                            return (addr, dt)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        return (addr, None)

                ct_tasks = [asyncio.create_task(_fetch_ct(a)) for a in missing_ct]
                try:
                    for fut in asyncio.as_completed(ct_tasks):
                        if should_stop():
                            break
                        try:
                            addr, dt = await fut
                            if dt:
                                for t in addr_to_tokens.get(addr, []):
                                    try:
                                        ts_val = _safe_int(dt.timestamp())
                                        if ts_val is not None:
                                            t["creation_timestamp"] = ts_val
                                        cats = t.get("categories") or []
                                        if "no_creation_time" in cats:
                                            try:
                                                cats = [c for c in cats if c != "no_creation_time"]
                                                t["categories"] = cats
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.debug("Error awaiting creation_time task", exc_info=True)
                finally:
                    for tt in ct_tasks:
                        if not tt.done():
                            tt.cancel()
                    try:
                        await asyncio.gather(*ct_tasks, return_exceptions=True)
                    except Exception:
                        pass

        # close session we created
        if own_session and session is not None:
            try:
                await session.close()
            except Exception:
                pass

        return tokens

# ------------------------
# Technical indicators (unchanged)
# ------------------------
def compute_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if not prices or len(prices) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = prices[i] - prices[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses += -delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0 and avg_gain == 0:
        return 50.0
    for i in range(period + 1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)

def compute_simple_moving_average(prices: List[float], period: int) -> Optional[float]:
    if not prices or len(prices) < period:
        return None
    return float(sum(prices[-period:]) / period)

def compute_bollinger(prices: List[float], period: int = 20, mult: float = 2.0) -> Optional[Tuple[float, float, float]]:
    if not prices or len(prices) < period:
        return None
    slice_prices = prices[-period:]
    mean = sum(slice_prices) / period
    variance = sum((p - mean) ** 2 for p in slice_prices) / (period - 1) if period > 1 else 0.0
    stddev = math.sqrt(variance)
    return (mean - mult * stddev, mean, mean + mult * stddev)

def compute_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict[str, float]]:
    if not prices or len(prices) < slow + 1:
        return None
    def ema_series(nums: List[float], period: int) -> List[Optional[float]]:
        n = len(nums)
        out: List[Optional[float]] = [None] * n
        if n < period:
            return out
        sma = sum(nums[0:period]) / float(period)
        out[period - 1] = sma
        prev = sma
        k = 2.0 / (period + 1)
        for i in range(period, n):
            prev = nums[i] * k + prev * (1 - k)
            out[i] = prev
        return out
    fast_ema = ema_series(prices, fast)
    slow_ema = ema_series(prices, slow)
    macd_series: List[Optional[float]] = [None] * len(prices)
    for i in range(len(prices)):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_series[i] = fast_ema[i] - slow_ema[i]
    macd_values = [v for v in macd_series if v is not None]
    if not macd_values or len(macd_values) < signal:
        return None
    def ema_on_list(nums: List[float], period: int) -> List[float]:
        out = []
        sma = sum(nums[0:period]) / float(period)
        prev = sma
        out.extend([float("nan")] * (period - 1))
        out.append(sma)
        k = 2.0 / (period + 1)
        for i in range(period, len(nums)):
            prev = nums[i] * k + prev * (1 - k)
            out.append(prev)
        return out
    signal_series = ema_on_list(macd_values, signal)
    macd_positions = [i for i, v in enumerate(macd_series) if v is not None]
    pos_index = len(macd_positions) - 1
    macd_val = macd_values[pos_index]
    signal_val = signal_series[pos_index] if pos_index < len(signal_series) else signal_series[-1]
    if math.isnan(signal_val):
        return None
    hist = macd_val - signal_val
    return {"macd": float(macd_val), "signal": float(signal_val), "hist": float(hist)}

# -----------------------------------------------------------------------------
# Backwards-compatible wrapper for fetch_birdeye_tokens
# -----------------------------------------------------------------------------
# At this point either:
#  - fetch_birdeye_tokens refers to the delegated function (which may require solana_client arg), or
#  - the module defined a local fallback (which uses the fallback signature).
# We wrap the currently-exported callable to accept both calling styles.

_orig_fetch = globals().get("fetch_birdeye_tokens", None)

def _is_sol_client_like(obj) -> bool:
    """Heuristic to detect an AsyncClient-like solana client."""
    if obj is None:
        return False
    # AsyncClient has methods like get_balance/get_account_info, but be permissive.
    return any(hasattr(obj, name) for name in ("get_account_info", "get_balance", "get_recent_blockhash"))

def _make_fetch_wrapper(orig_callable):
    if orig_callable is None:
        async def _noimpl(session, *args, **kwargs):
            logger.debug("fetch_birdeye_tokens wrapper: no underlying implementation available")
            return []
        return _noimpl

    # Inspect signature if possible
    try:
        sig = inspect.signature(orig_callable)
        params = list(sig.parameters.keys())
    except Exception:
        params = []

    # Determine whether the underlying callable expects a solana_client param (commonly second positional)
    expects_solana_param = any(p in ("solana_client", "client", "rpc_client", "sol") for p in params[1:3])

    async def _wrapper(session: aiohttp.ClientSession, *args, **kwargs):
        """
        Compatibility wrapper that accepts both:
         - (session, max_tokens=...)
         - (session, solana_client, max_tokens=...)
        and forwards to the underlying implementation in a best-effort manner.
        """
        # prefer explicit kwarg if provided
        if expects_solana_param:
            # try to extract solana_client from kwargs or positional args
            sol = None
            max_tokens = kwargs.pop("max_tokens", None)

            # check kwargs for common names
            for key in ("solana_client", "client", "rpc_client", "sol"):
                if key in kwargs:
                    sol = kwargs.pop(key)
                    break

            if sol is None and len(args) >= 1:
                # assume first positional arg is solana_client
                sol = args[0]
                # if second positional arg exists and appears to be numeric, treat as max_tokens
                if len(args) >= 2 and isinstance(args[1], (int, float, str)) and str(args[1]).isdigit():
                    max_tokens = int(args[1])
            # If caller supplied a sol-like object as max_tokens accidentally, accept but prefer sol clients
            if sol is None and isinstance(max_tokens, (object,)) and _is_sol_client_like(max_tokens):
                sol = max_tokens
                max_tokens = None

            try:
                # Call underlying with (session, solana_client, max_tokens=...)
                return await orig_callable(session, sol, max_tokens=max_tokens)
            except TypeError:
                # fallback: try positional calling
                try:
                    if sol is None:
                        return await orig_callable(session, *args, max_tokens=max_tokens, **kwargs)
                    return await orig_callable(session, sol, *([max_tokens] if max_tokens is not None else []), **kwargs)
                except Exception:
                    # last-resort: attempt to call with only session and any provided numeric max_tokens
                    try:
                        return await orig_callable(session, max_tokens=max_tokens, **kwargs)
                    except Exception:
                        logger.exception("fetch_birdeye_tokens wrapper: failed to call underlying impl")
                        return []
            except Exception:
                logger.exception("fetch_birdeye_tokens wrapper: underlying impl raised")
                return []
        else:
            # Underlying does not expect solana_client; call in fallback style.
            # If caller passed a sol client as first positional arg, treat it as mistake and skip it.
            max_tokens = kwargs.pop("max_tokens", None)
            args_list = list(args)
            # If args_list[0] looks like a sol client, drop it
            if args_list and _is_sol_client_like(args_list[0]):
                args_list = args_list[1:]
            # If remaining first positional arg looks numeric, treat as max_tokens
            if args_list:
                first = args_list[0]
                if isinstance(first, (int, float)) or (isinstance(first, str) and str(first).isdigit()):
                    max_tokens = int(first)
            try:
                return await orig_callable(session, max_tokens=max_tokens, **kwargs)
            except TypeError:
                # try positional fallback
                try:
                    if max_tokens is not None:
                        return await orig_callable(session, max_tokens)
                    return await orig_callable(session)
                except Exception:
                    logger.exception("fetch_birdeye_tokens wrapper: failed to call underlying impl (fallback path)")
                    return []
            except Exception:
                logger.exception("fetch_birdeye_tokens wrapper: underlying impl raised")
                return []

    return _wrapper

# Replace exported name with wrapper (preserve original under internal name)
if _orig_fetch is not None and callable(_orig_fetch):
    globals()["_fetch_birdeye_tokens_impl"] = _orig_fetch
globals()["fetch_birdeye_tokens"] = _make_fetch_wrapper(_orig_fetch)

# -----------------------------------------------------------------------------
# Exported symbols
# -----------------------------------------------------------------------------
__all__ = [
    "fetch_birdeye_tokens",
    "fetch_birdeye_creation_time",
    "fetch_birdeye_price",
    "fetch_birdeye_multi_price",
    "enrich_tokens_with_price_change",
    "compute_rsi",
    "compute_macd",
    "compute_bollinger",
    "compute_simple_moving_average",
    "signal_shutdown",
    "clear_shutdown_signal",
]