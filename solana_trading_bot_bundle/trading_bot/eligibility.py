# solana_trading_bot_bundle/trading_bot/eligibility.py
from __future__ import annotations

import os
import logging
import asyncio
import time
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import aiohttp
import json
import sqlite3

# Configure via environment:
# - DUMP_TOKEN_ADDRS: comma-separated addresses to dump (prefix matching allowed)
# - DUMP_TOKEN_ON_REJECT: "1" to dump ANY rejected token object (be careful — noisy)
DUMP_TOKEN_ADDRS = {a.strip() for a in os.getenv("DUMP_TOKEN_ADDRS", "").split(",") if a.strip()}
DUMP_TOKEN_ON_REJECT = os.getenv("DUMP_TOKEN_ON_REJECT", "0") == "1"

# ---- Noise filter: stables & WSOL shouldn't be shortlisted in categories ----
KNOWN_STABLES = {
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    "27G8MtK7VtTcCHkpASjSDdkWWYfoqT6ggEuKidVJidD4",  # JLP
    "BNso1VUJnh4zcfpZa6986Ea66P6TCp59hvtNJ8b1X85",   # BNSOL
}
ALWAYS_HIDE_IN_CATEGORIES = KNOWN_STABLES | {
    "So11111111111111111111111111111111111111112",  # WSOL
}

from .utils_exec import (
    add_to_blacklist,
    load_config,
    get_rugcheck_token,  # fallback if rugcheck_auth headers aren’t available
    _best_first,
    WHITELISTED_TOKENS,
)

logger = logging.getLogger("TradingBot")

# Standardize eligibility rejections at INFO so they're easy to grep in logs
REJECT_TAG = "ELIG-REJECT"


def _log_reject(addr: str, reason: str) -> None:
    logger.info("%s %s: %s", REJECT_TAG, addr, reason)

from .eligibility_override import should_override_extreme_price   # type: ignore

# --- Rugcheck headers import (supports both of your helper locations) ---
try:
    # utils variant
    from utils.rugcheck_auth import ensure_valid_rugcheck_headers as _ensure_headers  # type: ignore
    from utils.rugcheck_auth import get_rugcheck_headers as _get_headers  # type: ignore
    logger.debug("Imported rugcheck_auth from utils.rugcheck_auth")
except Exception as e:
    logger.debug("utils.rugcheck_auth not available: %s", e, exc_info=True)
    try:
        # bundle variant
        from solana_trading_bot_bundle.trading_bot.rugcheck_auth import (  # type: ignore
            ensure_valid_rugcheck_headers as _ensure_headers,
            get_rugcheck_headers as _get_headers,
        )
        logger.debug("Imported rugcheck_auth from bundle rugcheck_auth")
    except Exception as e2:
        logger.debug("bundle rugcheck_auth not available: %s", e2, exc_info=True)
        _ensure_headers = None  # type: ignore
        _get_headers = None     # type: ignore

# --- RugcheckClient integration (optional) ---
try:
    from .rugcheck_client import make_rugcheck_client_from_env  # type: ignore
    logger.debug("Imported make_rugcheck_client_from_env")
except Exception:
    try:
        from solana_trading_bot_bundle.trading_bot.rugcheck_client import make_rugcheck_client_from_env  # type: ignore
        logger.debug("Imported make_rugcheck_client_from_env from bundle")
    except Exception as e:
        make_rugcheck_client_from_env = None  # type: ignore
        logger.debug("No rugcheck_client factory available: %s", e, exc_info=True)

# Concurrency guard for lazy client creation + shutdown helper
_rug_client = None  # type: ignore
_rug_client_lock: Optional[asyncio.Lock] = None

def _ensure_rug_client_lock() -> Optional[asyncio.Lock]:
    """
    Lazily create an asyncio.Lock for serializing _rug_client creation.
    Returns None if lock construction fails (extremely unlikely).
    """
    global _rug_client_lock
    if _rug_client_lock is None:
        try:
            _rug_client_lock = asyncio.Lock()
        except Exception:
            _rug_client_lock = None
    return _rug_client_lock

async def _get_rug_client():
    """
    Lazy-create and start a shared RugcheckClient instance.
    Serialized with an asyncio.Lock to avoid races when multiple coroutines
    request the client at the same time.
    Returns None if no factory available or start() fails.
    """
    global _rug_client
    # Fast-path: already initialized
    if _rug_client is not None:
        return _rug_client

    if make_rugcheck_client_from_env is None:
        return None

    lock = _ensure_rug_client_lock()
    if lock is not None:
        async with lock:
            # Double-check inside lock
            if _rug_client is not None:
                return _rug_client
            try:
                _rug_client = make_rugcheck_client_from_env()
            except Exception as e:
                logger.debug("make_rugcheck_client_from_env() failed: %s", e, exc_info=True)
                _rug_client = None
                return None

            # Try to start; if start fails, attempt to close and return None
            try:
                await _rug_client.start()
            except Exception as e:
                logger.exception("Rugcheck client.start() failed — falling back to legacy: %s", e)
                # Tear down partially-initialized client
                try:
                    close_fn = getattr(_rug_client, "close", None)
                    if callable(close_fn):
                        maybe = close_fn()
                        if asyncio.iscoroutine(maybe):
                            await maybe
                except Exception:
                    logger.debug("Failed to close partially-started rugcheck client", exc_info=True)
                _rug_client = None
                return None
            return _rug_client
    else:
        # Fallback without lock (shouldn't generally happen)
        try:
            _rug_client = make_rugcheck_client_from_env()
        except Exception as e:
            logger.debug("make_rugcheck_client_from_env() failed (no-lock path): %s", e, exc_info=True)
            _rug_client = None
            return None
        try:
            await _rug_client.start()
        except Exception as e:
            logger.exception("Rugcheck client.start() failed (no-lock path): %s", e)
            try:
                close_fn = getattr(_rug_client, "close", None)
                if callable(close_fn):
                    maybe = close_fn()
                    if asyncio.iscoroutine(maybe):
                        await maybe
            except Exception:
                logger.debug("Failed to close partially-started rugcheck client (no-lock path)", exc_info=True)
            _rug_client = None
            return None
        return _rug_client


async def shutdown_rugcheck_client(timeout: float = 5.0) -> None:
    """
    Close the shared RugcheckClient instance if present.
    Call during test teardown or short-lived scripts to avoid leaked resources.
    """
    global _rug_client
    rc = globals().get("_rug_client") or None
    if not rc:
        return
    try:
        close_fn = getattr(rc, "close", None)
        if callable(close_fn):
            res = close_fn()
            if asyncio.iscoroutine(res):
                try:
                    await asyncio.wait_for(res, timeout=timeout)
                except Exception:
                    logger.debug("shutdown_rugcheck_client: close timed out or errored", exc_info=True)
        # also try 'stop' if client exposes it
        stop_fn = getattr(rc, "stop", None)
        if callable(stop_fn):
            res2 = stop_fn()
            if asyncio.iscoroutine(res2):
                try:
                    await asyncio.wait_for(res2, timeout=timeout)
                except Exception:
                    logger.debug("shutdown_rugcheck_client: stop timed out or errored", exc_info=True)
    except Exception:
        logger.debug("shutdown_rugcheck_client: encountered error", exc_info=True)
    finally:
        try:
            _rug_client = None
        except Exception:
            pass


def shutdown_rugcheck_client_sync(timeout: float = 5.0) -> None:
    """
    Synchronous wrapper for shutdown_rugcheck_client for callers that are not async.
    """
    try:
        asyncio.run(shutdown_rugcheck_client(timeout=timeout))
    except RuntimeError:
        # If called inside an already-running loop (tests), just schedule it
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(shutdown_rugcheck_client(timeout=timeout))
        except Exception:
            logger.debug("shutdown_rugcheck_client_sync: failed to schedule async shutdown", exc_info=True)

# -----------------------------------------------------------------------------#
# Optional TTL caches (avoid refetching within a short window)
# -----------------------------------------------------------------------------#
try:
    from cachetools import TTLCache
except Exception as e:  # very small fallback if cachetools isn't installed
    logger.warning("cachetools not installed — using simple TTLCache fallback. Install cachetools for production: %s", e)
    class TTLCache(dict):  # type: ignore
        def __init__(self, maxsize: int, ttl: int) -> None:
            super().__init__()
            self._ttl = ttl
        def __setitem__(self, key, value) -> None:  # store (value, expires_at)
            super().__setitem__(key, (value, time.time() + self._ttl))
        def __getitem__(self, key):
            v, exp = super().__getitem__(key)
            if time.time() > exp:
                super().pop(key, None)
                raise KeyError(key)
            return v
        def get(self, key, default=None):
            try:
                return self.__getitem__(key)
            except KeyError:
                return default

# Cache horizons (seconds)
_RUGCHECK_TTL = 30 * 60
_HOLDERS_TTL = 10 * 60
_CONC_TTL    = 10 * 60

_rugcheck_cache: TTLCache = TTLCache(maxsize=5000, ttl=_RUGCHECK_TTL)
_holders_cache: TTLCache  = TTLCache(maxsize=10000, ttl=_HOLDERS_TTL)  # key: token -> int
_conc_cache: TTLCache     = TTLCache(maxsize=5000, ttl=_CONC_TTL)      # key: (token, top_n) -> float

# -----------------------------------------------------------------------------#
# Env fallback thresholds (used only if config values are missing)
# -----------------------------------------------------------------------------#
MIN_LIQ_USD_ENV   = float(os.getenv("MIN_LIQ_USD", "0"))     # e.g., 1000
MIN_VOL24_USD_ENV = float(os.getenv("MIN_VOL24_USD", "0"))   # e.g., 2500

# -----------------------------------------------------------------------------#
# Small helpers
# -----------------------------------------------------------------------------#
def get_env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _lower(s: Any) -> str:
    try:
        return str(s or "").strip().lower()
    except Exception:
        return ""


def _label_names_from_any(labels_any: Any) -> List[str]:
    """
    Normalize labels coming from RugCheck (array of dicts/strings/mixed) into a list of lowercase names.
    Accepts: [{"label":"scam"}, "dangerous", {"name": "Honeypot"}]
    -> ["scam","dangerous","honeypot"]
    """
    out: List[str] = []
    if labels_any is None:
        return out
    if isinstance(labels_any, dict):
        for k, v in labels_any.items():
            if v:
                out.append(_lower(k))
        return out
    if isinstance(labels_any, list):
        for item in labels_any:
            if isinstance(item, dict):
                name = item.get("label") or item.get("name") or item.get("value") or item.get("type")
                out.append(_lower(name))
            else:
                out.append(_lower(item))
        return out
    out.append(_lower(labels_any))
    return out


def _disc_get(cfg: Dict[str, Any], cat: str, key: str, default: Any = None) -> Any:
    """
    Safe accessor for discovery subkeys. Returns default if discovery/cat/key is missing.
    """
    try:
        return ((cfg or {}).get("discovery") or {}).get(cat, {}).get(key, default)
    except Exception:
        return default

# -----------------------------------------------------------------------------#
# Helper: robust numeric coercion + persisted-shortlist validator
# -----------------------------------------------------------------------------#

def _safe_float(x: Any, default: float = 0.0) -> Optional[float]:
    """
    Best-effort conversion to float for values coming from DB/API payloads.
    Handles None, numeric types, strings with commas, leading/trailing $/percent,
    parentheses negatives and scientific notation. Returns None if parsing fails.
    """
    if x is None:
        return None
    try:
        # numeric types (excluding bool)
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            v = float(x)
            if math.isfinite(v):
                return v
            return None
        s = str(x).strip()
        if s == "":
            return None
        # strip currency/thousand separators
        s = s.replace(",", "").replace("$", "")
        # percent -> numeric (e.g. "5.2%" -> 5.2)
        if s.endswith("%"):
            s = s[:-1]
        # parentheses negative: "(123)" -> "-123"
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        v = float(s)
        if math.isfinite(v):
            return v
    except Exception:
        try:
            # fallback: Decimal parse then to float
            from decimal import Decimal
            v = Decimal(str(x))
            fv = float(v)
            if math.isfinite(fv):
                return fv
        except Exception:
            return None
    return None


def _coerce_token_numerics(token: Dict[str, Any]) -> Dict[str, Any]:
    """
    In-place normalize common numeric fields on token dict:
      - liquidity, volume_24h, market_cap, price, price_change_1h/6h/24h
    Tries several common field names and stores canonical numeric fields back on the token.
    Returns the token for convenience.
    """
    def pick_float(*cands):
        for c in cands:
            try:
                v = token.get(c)
                if v is None:
                    continue
                fv = _safe_float(v)
                if fv is not None:
                    return fv
            except Exception:
                continue
        return None

    liq = pick_float("liquidity", "liquidity_usd", "dex_liquidity", "ray_liquidity_usd", "liquidityUsd")
    token["liquidity"] = float(liq or 0.0)

    vol = pick_float("volume_24h", "v24hUSD", "volume", "volume24h", "v24hUsd")
    token["volume_24h"] = float(vol or 0.0)

    mc = pick_float("market_cap", "mc", "fdv", "fdvUsd", "mcap")
    token["market_cap"] = float(mc or 0.0)
    token["mc"] = token.get("market_cap", token.get("mc", 0.0))

    price = pick_float("price", "token_price", "last_price")
    token["price"] = float(price or 0.0)

    token["price_change_1h"] = float(pick_float("price_change_1h", "priceChange1h") or 0.0)
    token["price_change_6h"] = float(pick_float("price_change_6h", "priceChange6h") or 0.0)
    token["price_change_24h"] = float(pick_float("price_change_24h", "priceChange24h", "priceChange") or 0.0)

    return token


async def validate_persisted_shortlist(
    tokens: Iterable[Dict[str, Any]],
    cfg: Dict[str, Any],
    session: aiohttp.ClientSession | None = None,
) -> List[Dict[str, Any]]:
    """
    Validate/normalize tokens loaded from the DB fallback.

    Steps:
      - If token contains a raw JSON `data` blob, attempt to extract missing fields from it.
      - Coerce/normalize numeric fields (liquidity, volume_24h, market_cap, price, price-change).
      - Re-run enforce_scoring_hard_floor() with the provided config.
      - Re-run recent momentum guard (_recent_momentum_allowed()).
      - Discard tokens with zero liquidity AND zero volume.
      - Return the sanitized list (does not mutate original input objects).
    """
    out: List[Dict[str, Any]] = []
    cfg_local = cfg or {}
    for raw in (tokens or []):
        try:
            token = dict(raw) if isinstance(raw, dict) else {"address": str(raw)}
            # If stored JSON blob is present, try to fill missing fields
            if not token.get("symbol") and isinstance(token.get("data"), str):
                try:
                    j = json.loads(token.get("data") or "{}")
                    if isinstance(j, dict):
                        for k in ("symbol", "name", "price", "liquidity", "volume_24h", "market_cap"):
                            if not token.get(k) and (k in j):
                                token[k] = j[k]
                except Exception:
                    # ignore malformed embedded JSON
                    pass

            # Normalise numeric fields
            _coerce_token_numerics(token)

            # Re-run hard-floor gating
            try:
                hf_ok, hf_reason = enforce_scoring_hard_floor(token, cfg_local)
            except Exception:
                hf_ok, hf_reason = True, "hard_floor helper error"
            if not hf_ok:
                logger.info("Persisted shortlist token rejected by hard_floor: %s (%s)", token.get("address"), hf_reason)
                continue

            # Re-run recent momentum guard
            try:
                recent_ok, recent_reason = _recent_momentum_allowed(token, cfg=cfg_local)
            except Exception:
                # if the helper fails, allow the token through (conservative)
                recent_ok, recent_reason = True, "momentum helper error"
            if not recent_ok:
                logger.info("Persisted shortlist token rejected by recent_momentum: %s (%s)", token.get("address"), recent_reason)
                continue

            # Require at least some depth
            liq = float(token.get("liquidity") or 0.0)
            vol = float(token.get("volume_24h") or 0.0)
            if liq <= 0.0 and vol <= 0.0:
                logger.info("Persisted shortlist token lacks liquidity/volume, skipping: %s (liq=%s vol=%s)", token.get("address"), liq, vol)
                continue

            # Accept the normalized token
            out.append(token)
        except Exception as e:
            logger.debug("validate_persisted_shortlist: failed for row %r: %s", raw, e, exc_info=True)
            continue

    return out    

def enforce_scoring_hard_floor(token: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Enforce scoring.hard_floor rules if enabled in config.
    Returns (allowed: bool, reason: str).
    This mirrors the discovery prefilter but is safe to call from other code paths.
    """
    try:
        scoring_hf = (cfg.get("scoring") or {}).get("hard_floor") or {}
        if not scoring_hf.get("enable", False):
            return True, "hard_floor disabled"

        addr = token.get("address") or ""
        allow_whitelist = bool(scoring_hf.get("allow_whitelist", True))
        if allow_whitelist and addr and addr in (WHITELISTED_TOKENS or {}):
            return True, "whitelisted bypass"

        def _num(x, d=0.0):
            try:
                return float(x or 0.0)
            except Exception:
                return float(d)

        min_liq = _num(scoring_hf.get("min_liquidity_usd", 0.0))
        min_vol = _num(scoring_hf.get("min_volume_24h_usd", 0.0))
        min_mc = _num(scoring_hf.get("min_market_cap_usd", 0.0))
        max_mc = _num(scoring_hf.get("max_market_cap_usd", 0.0))

        liq = _num(token.get("liquidity") or token.get("dex_liquidity") or token.get("ray_liquidity_usd") or token.get("liquidity_usd"))
        vol = _num(token.get("v24hUSD") or token.get("volume_24h") or token.get("volume24h") or token.get("dex_volume_24h"))
        mc = _num(token.get("market_cap") or token.get("mc") or 0.0)

        if min_mc and mc < min_mc:
            return False, f"market_cap {mc:.0f} < min_market_cap_usd {min_mc:.0f}"
        if max_mc and mc > max_mc:
            return False, f"market_cap {mc:.0f} > max_market_cap_usd {max_mc:.0f}"
        if min_liq and liq < min_liq:
            return False, f"liquidity {liq:.0f} < min_liquidity_usd {min_liq:.0f}"
        if min_vol and vol < min_vol:
            return False, f"volume24h {vol:.0f} < min_volume_24h_usd {min_vol:.0f}"

        # optional checks that may rely on external lookups — best-effort placeholders:
        min_holders = int(scoring_hf.get("min_holders", 0) or 0)
        if min_holders:
            holders = token.get("holderCount") or token.get("holders") or 0
            try:
                if int(holders or 0) < min_holders:
                    return False, f"holders {holders} < min_holders {min_holders}"
            except Exception:
                pass

        return True, "passes hard_floor"
    except Exception as e:
        logger.debug("enforce_scoring_hard_floor error: %s", e, exc_info=True)
        # Being conservative: if helper fails, allow token so other checks can decide.
        return True, "hard_floor helper error (allowed)"
    
    
# ---------- RugCheck status composer (for clear logs/UI) ----------
def _compose_rugcheck_status(config: Dict[str, Any]) -> Tuple[str, bool, bool, bool]:
    """
    Returns (status_str, core_enabled, discovery_enabled, filter_enabled)
    """
    try:
        core_enabled = bool(((config.get("rugcheck") or {}).get("enabled", False)))
        disc_cfg = (config.get("discovery") or {})
        env_disc = get_env_bool("RUGCHECK_DISCOVERY_CHECK", False)
        env_filter = get_env_bool("RUGCHECK_DISCOVERY_FILTER", False)
        cfg_in_disc = bool(disc_cfg.get("rugcheck_in_discovery", False))
        cfg_require_pass = disc_cfg.get("require_rugcheck_pass", None)
        filter_enabled = bool(env_filter or (cfg_require_pass is True))
        discovery_enabled = bool(env_disc or cfg_in_disc or (cfg_require_pass is True))
        status_str = (
            f"core={'enabled' if core_enabled else 'disabled'}, "
            f"discovery={'enabled' if discovery_enabled else 'disabled'}, "
            f"filter={'enabled' if filter_enabled else 'disabled'}"
        )
        return status_str, core_enabled, discovery_enabled, filter_enabled
    except Exception:
        return "core=? , discovery=? , filter=?", False, False, False


# ---------------- New helper: require recent positive momentum ----------------
def _recent_momentum_allowed(
    token: Dict[str, Any],
    *,
    cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    try:
        disc = cfg.get("discovery") or {}
        min_1h = _num(disc.get("min_price_change_1h", 0.0), 0.0)
        min_6h = _num(disc.get("min_price_change_6h", 0.0), 0.0)
        # New flag: allow_missing_price_change -> when True, missing price-change fields
        # will be treated as 0.0 instead of causing an immediate reject.
        allow_missing = bool(disc.get("allow_missing_price_change", True))
    except Exception:
        min_1h, min_6h, allow_missing = 0.0, 0.0, True

    p1_src = _best_first(
        token.get("price_change_1h"),
        token.get("priceChange1h"),
        (token.get("dexscreener") or {}).get("priceChange1h"),
        (token.get("birdeye") or {}).get("price_change_1h"),
    )
    p6_src = _best_first(
        token.get("price_change_6h"),
        token.get("priceChange6h"),
        (token.get("dexscreener") or {}).get("priceChange6h"),
        (token.get("birdeye") or {}).get("price_change_6h"),
    )

    # If operator asked for strict enforcement, require presence; otherwise coerce missing to 0.0
    if ("min_price_change_1h" in (disc or {})) and p1_src is None:
        if not allow_missing:
            return False, "missing price_change_1h while min_price_change_1h configured"
        p1 = 0.0
    else:
        p1 = _num(p1_src, 0.0)

    if ("min_price_change_6h" in (disc or {})) and p6_src is None:
        if not allow_missing:
            return False, "missing price_change_6h while min_price_change_6h configured"
        p6 = 0.0
    else:
        p6 = _num(p6_src, 0.0)

    if p1 < float(min_1h):
        return False, f"price_change_1h={p1}% < min_price_change_1h={min_1h}%"
    if p6 < float(min_6h):
        return False, f"price_change_6h={p6}% < min_price_change_6h={min_6h}%"

    return True, f"price_change_1h={p1}%, price_change_6h={p6}% meet minima"


# ---------------- New: Price-change guard helper (signed logic) -----------------
def _price_change_allowed(
    token: Dict[str, Any],
    price_change_pct: float,
    *,
    cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    try:
        disc = cfg.get("discovery") or {}
        max_pct = _num(disc.get("max_price_change", 350), 350.0)
        max_pct_hard = _num(disc.get("max_price_change_hard", max(max_pct * 4, 2000)), max(max_pct * 4, 2000.0))
        min_price_for_pct = _num(disc.get("min_price_for_pct", 0.0001), 0.0001)
        min_liq_for_big_pct = _num(disc.get("min_liq_for_big_pct", 5000.0), 5000.0)
        min_vol_for_big_pct = _num(disc.get("min_vol_for_big_pct", 10000.0), 10000.0)
        max_drop_pct = disc.get("max_price_drop")
        max_drop_pct = _num(max_drop_pct, max_pct) if max_drop_pct is not None else max_pct
    except Exception:
        max_pct, max_pct_hard, min_price_for_pct, min_liq_for_big_pct, min_vol_for_big_pct, max_drop_pct = (
            350.0, 2000.0, 0.0001, 5000.0, 10000.0, 350.0
        )

    pct = float(price_change_pct or 0.0)

    if pct < 0.0:
        apct = abs(pct)
        if apct > float(max_drop_pct):
            return False, f"price_change_24h={pct}% negative drop too large (>{max_drop_pct}%)"
        return True, f"price_change_24h={pct}% negative within allowed drop ({max_drop_pct}%)"

    if pct > float(max_pct_hard):
        return False, f"price_change_24h={pct}% too extreme (>{max_pct_hard}%)"

    if pct <= float(max_pct):
        return True, f"price_change_24h={pct}% <= max={max_pct}%"

    liq = _num(_best_first(token.get("liquidity"), token.get("liquidity_usd"), 0.0), 0.0)
    vol24 = _num(_best_first(token.get("volume_24h"), token.get("v24hUSD"), token.get("v24"), token.get("volume"), 0.0), 0.0)
    price_val = None
    try:
        price_val = None if token.get("price") is None else float(token.get("price"))
    except Exception:
        price_val = None

    if price_val is not None and price_val > 0 and price_val < float(min_price_for_pct):
        if liq >= float(min_liq_for_big_pct) or vol24 >= float(min_vol_for_big_pct):
            return True, f"tiny_price={price_val} but market depth present (liq={liq},vol24={vol24})"
        return False, f"tiny_price={price_val} -> pct unreliable (pct={pct}%)"

    if liq >= float(min_liq_for_big_pct) or vol24 >= float(min_vol_for_big_pct):
        return True, f"pct={pct}% > {max_pct}% but market depth ok (liq={liq},vol24={vol24})"

    return False, f"price_change_24h={pct}% > {max_pct}% and market depth too low (liq={liq},vol24={vol24})"


# -----------------------------------------------------------------------------#
# Centralized HTTP helper (retries for 429/5xx, consistent headers)
# -----------------------------------------------------------------------------#
async def _json_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
    max_retries: int = 2,
    backoff_initial: float = 1.5,
) -> Optional[Any]:
    hdrs = {"accept": "application/json", "User-Agent": "SOLOTradingBot/1.0"}
    if headers:
        hdrs.update(headers)

    attempt = 0
    backoff = backoff_initial
    while True:
        try:
            async with session.request(
                method.upper(),
                url,
                headers=hdrs,
                params=params,
                json=json_body,
                timeout=timeout,
            ) as resp:
                if resp.status == 200:
                    try:
                        return await resp.json(content_type=None)
                    except Exception as e:
                        logger.warning("Failed to decode JSON from %s: %s", url, e, exc_info=True)
                        return None
                if resp.status in (429, 500, 502, 503, 504) and attempt < max_retries:
                    attempt += 1
                    body = await resp.text()
                    logger.warning("HTTP %s from %s, retrying in %.1fs (attempt %d/%d). Body: %s",
                                   resp.status, url, backoff, attempt, max_retries, (body or "")[:200])
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                body = await resp.text()
                logger.debug("Non-200 from %s: %s %s", url, resp.status, (body or "")[:300])
                return None
        except Exception as e:
            if attempt < max_retries:
                attempt += 1
                logger.warning("Request error to %s: %s; retrying in %.1fs (%d/%d)", url, e, backoff, attempt, max_retries)
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            logger.exception("Request error to %s (final): %s", url, e)
            return None


# -----------------------------------------------------------------------------#
# RugCheck verification
# -----------------------------------------------------------------------------#
async def verify_token_with_rugcheck(token_address, token, session, config):
    """
    Returns (risk_score: float, labels: list[str], message: str)
    """
    rcfg = (config.get("rugcheck") or {}) if isinstance(config, dict) else {}
    if not bool(rcfg.get("enabled", False)):
        return 0.0, [], "RugCheck disabled"

    cache = _rugcheck_cache  # type: ignore
    if token_address in cache:
        try:
            return cache[token_address]
        except Exception:
            logger.debug("RugCheck cache corrupted for %s — ignoring cache entry", token_address, exc_info=True)
            try:
                del cache[token_address]
            except Exception:
                pass

    client = await _get_rug_client() if make_rugcheck_client_from_env is not None else None

    data = None
    last_err = None

    api_base = str(rcfg.get("api_base", "https://api.rugcheck.xyz/v1")).rstrip("/")

    if client:
        try:
            # Prefer the unified helper that prefers the rich /report endpoint then falls back to summary.
            data = await client.get_token_report(token_address)
        except Exception as e:
            # Preserve diagnostic info and attempt a forced header refresh on auth failures.
            last_err = f"{type(e).__name__}: {e}"
            logger.debug("Rugcheck client.get_token_report error for %s: %s", token_address, last_err, exc_info=True)

            status = getattr(e, "status", None) or None
            # If unauthorized, try a forced refresh of local JWT helper then retry once.
            if status in (401, 403) and _ensure_headers:
                try:
                    maybe = _ensure_headers(session, force_refresh=True)  # type: ignore
                    if asyncio.iscoroutine(maybe):
                        await maybe
                except Exception:
                    logger.debug("ensure_valid_rugcheck_headers refresh failed (client path)", exc_info=True)
                try:
                    data = await client.get_token_report(token_address)
                except Exception as e2:
                    last_err = f"{type(e2).__name__}: {e2}"
                    logger.debug("Rugcheck client.get_token_report retry failed for %s: %s", token_address, e2, exc_info=True)

    if data is None:
        endpoints = [
            f"{api_base}/tokens/scan/solana/{token_address}",
            f"{api_base}/tokens/{token_address}",
        ]

        async def _fetch_legacy(path: str):
            nonlocal data, last_err
            headers: Dict[str, str] = {}
            try:
                maybe = _ensure_headers(session, force_refresh=False) if _ensure_headers else None
                if asyncio.iscoroutine(maybe):
                    maybe = await maybe
                if isinstance(maybe, dict):
                    headers.update(maybe)
            except Exception as e:
                logger.debug("ensure_valid_rugcheck_headers failed: %s", e, exc_info=True)
            try:
                maybe = _get_headers() if _get_headers else None
                if isinstance(maybe, dict):
                    headers.update(maybe)
            except Exception as e:
                logger.debug("get_rugcheck_headers failed: %s", e, exc_info=True)
            if "authorization" not in {k.lower() for k in headers.keys()}:
                try:
                    tok = get_rugcheck_token()
                    if tok:
                        headers["Authorization"] = f"Bearer {tok}"
                except Exception as e:
                    logger.debug("get_rugcheck_token failed: %s", e, exc_info=True)
            api_key = rcfg.get("api_key") or os.getenv("RUGCHECK_API_KEY") or ""
            if api_key and "x-api-key" not in {k.lower() for k in headers.keys()}:
                headers["X-API-KEY"] = str(api_key)
            timeout_total = float(rcfg.get("timeout_sec", 15.0))
            timeout = aiohttp.ClientTimeout(total=timeout_total, sock_connect=min(7, timeout_total), sock_read=min(10, timeout_total))
            try:
                async with session.get(path, headers=headers, timeout=timeout) as resp:
                    if resp.status == 200:
                        try:
                            data_local = await resp.json(content_type=None)
                        except Exception:
                            txt = await resp.text()
                            data_local = {"raw": txt}
                        data = data_local
                        return resp.status
                    last_err = f"HTTP {resp.status} at {path}"
                    logger.debug("Rugcheck legacy HTTP %s for %s: %s", resp.status, token_address, path)
                    return resp.status
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                logger.debug("Rugcheck legacy request to %s failed: %s", path, e, exc_info=True)
                return None

        st = await _fetch_legacy(endpoints[0])
        if st in (401, 403):
            st = await _fetch_legacy(endpoints[0])
        if data is None:
            st = await _fetch_legacy(endpoints[1])
            if st in (401, 403):
                await _fetch_legacy(endpoints[1])

    if data is None:
        msg = f"RugCheck API error for {token_address}: {last_err or 'unknown'}"
        logger.warning(msg)
        if bool(rcfg.get("allow_if_api_down", True)):
            logger.warning("RugCheck allow_if_api_down=True -> permitting token %s while API is down (last_err=%s)", token_address, last_err)
            tup = (0.0, [], "API down - allowed")
            try:
                cache[token_address] = tup  # type: ignore
            except Exception:
                logger.debug("Failed to cache rugcheck allow_if_api_down result for %s", token_address, exc_info=True)
            return tup
        tup = (9999.0, [], msg)
        try:
            cache[token_address] = tup  # type: ignore
        except Exception:
            logger.debug("Failed to cache rugcheck failure for %s", token_address, exc_info=True)
        return tup

    payload = data.get("data", data) if isinstance(data, dict) else {}
    labels_raw = payload.get("labels") or payload.get("riskLabels") or payload.get("flags") or []
    labels_list = _label_names_from_any(labels_raw)

    cand_scores = [
        payload.get("risk_score"),
        payload.get("riskScore"),
        payload.get("score"),
        (payload.get("risk") or {}).get("score") if isinstance(payload, dict) else None,
        (payload.get("result") or {}).get("risk_score") if isinstance(payload, dict) else None,
        (payload.get("analysis") or {}).get("risk_score") if isinstance(payload, dict) else None,
    ]
    risk_score = 0.0
    for v in cand_scores:
        try:
            if v is not None:
                risk_score = float(v)
                break
        except Exception:
            logger.debug("Failed to coerce rugcheck score candidate %r for %s", v, token_address, exc_info=True)
            continue

    danger = {str(x).lower() for x in (rcfg.get("danger_labels_blocklist") or ["dangerous", "scam", "honeypot"])}
    present = set(labels_list)
    if present & danger:
        tup = (9999.0, labels_list, f"Blocked by labels: {sorted(present & danger)}")
        try:
            cache[token_address] = tup  # type: ignore
        except Exception:
            logger.debug("Failed to cache rugcheck blocked result for %s", token_address, exc_info=True)
        return tup

    tup = (risk_score, labels_list, "OK")
    try:
        cache[token_address] = tup  # type: ignore
    except Exception:
        logger.debug("Failed to cache rugcheck OK result for %s", token_address, exc_info=True)
    return tup


# -----------------------------------------------------------------------------#
# Holders helpers (Birdeye/Solscan/Helius/public)
# -----------------------------------------------------------------------------#
async def _fetch_holders_birdeye(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> Optional[int]:
    p = (cfg.get("providers", {}) or {}).get("birdeye", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        logger.debug("Birdeye provider not configured for holders (no api_key)")
        return None
    url = p.get("holders_url") or "https://api.birdeye.so/defi/token/holders"
    headers = {"x-api-key": api_key}
    payload = {"address": token_address, "page": 1, "page_size": 1}

    data = await _json_request(
        session, "POST", url, headers=headers, json_body=payload, timeout=timeout
    )
    if not data:
        logger.debug("Birdeye holders returned no data for %s", token_address)
        return None
    d = data.get("data", data)
    total = d.get("total")
    return int(total) if total is not None else None

async def _fetch_holders_solscan_pro(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> Optional[int]:
    p = (cfg.get("providers", {}) or {}).get("solscan_pro", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        logger.debug("Solscan Pro provider not configured for holders (no api_key)")
        return None

    url = "https://pro-api.solscan.io/v2.0/token/holders"
    headers = {"token": api_key}
    params = {"address": token_address, "page": 1, "page_size": 10}

    data = await _json_request(
        session, "GET", url, headers=headers, params=params, timeout=timeout
    )
    if not data:
        logger.debug("Solscan Pro holders returned no data for %s", token_address)
        return None
    d = data.get("data") or {}
    total = d.get("total")
    return int(total) if total is not None else None


async def _fetch_holders_helius(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    timeout: aiohttp.ClientTimeout,
) -> Optional[int]:
    p = (cfg.get("providers", {}) or {}).get("helius", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        logger.debug("Helius provider not configured for holders (no api_key)")
        return None
    url = p.get("url") or f"https://mainnet.helius-rpc.com/?api-key={api_key}"
    method = p.get("method_summary") or "getTokenHoldersSummary"

    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": {"mint": token_address}}
    data = await _json_request(session, "POST", url, json_body=payload, timeout=timeout)
    if not data:
        logger.debug("Helius holders returned no data for %s", token_address)
        return None
    result = (data or {}).get("result") | {} if isinstance((data or {}).get("result"), dict) else (data or {}).get("result", {})
    total = result.get("holderCount") or result.get("total")
    return int(total) if total is not None else None


async def _fetch_holders_public_solscan(
    token_address: str,
    session: aiohttp.ClientSession,
    timeout: aiohttp.ClientTimeout,
) -> Optional[int]:
    base = "https://public-api.solscan.io/token/holders"
    params = {"tokenAddress": token_address, "limit": 1, "offset": 0}
    data = await _json_request(session, "GET", base, params=params, timeout=timeout)
    if not data:
        logger.debug("Public Solscan holders returned no data for %s", token_address)
        return None
    if isinstance(data, dict):
        if "total" in data:
            return int(data["total"])
        d = data.get("data")
        if isinstance(d, dict) and "total" in d:
            return int(d["total"])
    return None


async def fetch_holder_count(
    token_address: str,
    session: aiohttp.ClientSession,
    config: Dict[str, Any],
) -> Tuple[Optional[int], str]:
    cached = _holders_cache.get(token_address)
    if isinstance(cached, int):
        return cached, "cache"

    hcfg = config.get("holders", {}) or {}
    timeout_sec = int(hcfg.get("timeout_seconds", 8))
    timeout = aiohttp.ClientTimeout(total=timeout_sec, sock_connect=5, sock_read=timeout_sec)

    count = await _fetch_holders_birdeye(token_address, session, hcfg, timeout)
    if isinstance(count, int) and count >= 0:
        _holders_cache[token_address] = count
        return count, "birdeye"

    count = await _fetch_holders_solscan_pro(token_address, session, hcfg, timeout)
    if isinstance(count, int) and count >= 0:
        _holders_cache[token_address] = count
        return count, "solscan_pro"

    count = await _fetch_holders_helius(token_address, session, hcfg, timeout)
    if isinstance(count, int) and count >= 0:
        _holders_cache[token_address] = count
        return count, "helius"

    if (hcfg.get("providers", {}).get("public_solscan", {}) or {}).get("enabled", True):
        count = await _fetch_holders_public_solscan(token_address, session, timeout)
        if isinstance(count, int) and count >= 0:
            _holders_cache[token_address] = count
            return count, "public_solscan"

    logger.debug("All holder providers returned no data for %s", token_address)
    return None, "unavailable"


def _min_holders_for_categories(categories: List[str], config: Dict[str, Any]) -> Optional[int]:
    hcfg = config.get("holders", {}) or {}
    mins = hcfg.get("min_holders", {}) or {}
    vals: List[int] = []
    for cat in categories:
        v = mins.get(cat)
        if isinstance(v, (int, float)):
            vals.append(int(v))
    return max(vals) if vals else None

# -----------------------------------------------------------------------------#
# Concentration — top N share
# -----------------------------------------------------------------------------#
async def _birdeye_supply_and_top(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    top_n: int,
    timeout: aiohttp.ClientTimeout,
) -> Optional[Tuple[float, float]]:
    p = (cfg.get("providers", {}) or {}).get("birdeye", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        logger.debug("Birdeye provider not configured for concentration (no api_key)")
        return None
    headers = {"x-api-key": api_key}
    holders_url  = p.get("top_holders_url") or p.get("holders_url") or "https://api.birdeye.so/defi/token/holders"
    overview_url = p.get("overview_url") or "https://api.birdeye.so/defi/token/overview"

    payload = {"address": token_address, "page": 1, "page_size": int(max(1, top_n))}
    hdata = await _json_request(session, "POST", holders_url, headers=headers, json_body=payload, timeout=timeout)
    if not hdata:
        logger.debug("Birdeye top holders returned no data for %s", token_address)
        return None
    d = hdata.get("data", hdata)
    items = d.get("items") or d.get("holders") or []
    top_sum = 0.0
    for it in items:
        qty = _best_first(it.get("balance"), it.get("amount"), it.get("uiAmount"))
        if qty is None:
            continue
        top_sum += _num(qty, 0.0)

    payload_overview = {"address": token_address}
    odata = await _json_request(session, "POST", overview_url, headers=headers, json_body=payload_overview, timeout=timeout)
    if not odata:
        logger.debug("Birdeye overview returned no data for %s", token_address)
        return None
    od = odata.get("data", odata)
    supply = _best_first(od.get("supply"), od.get("circulating_supply"), od.get("total_supply"))
    if supply is None:
        logger.debug("Birdeye overview missing supply for %s", token_address)
        return None
    total_supply = _num(supply, 0.0)
    if total_supply <= 0:
        logger.debug("Birdeye supply non-positive for %s: %s", token_address, total_supply)
        return None
    return top_sum, total_supply


async def _solscan_supply_and_top(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    top_n: int,
    timeout: aiohttp.ClientTimeout,
) -> Optional[Tuple[float, float]]:
    p = (cfg.get("providers", {}) or {}).get("solscan_pro", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        logger.debug("Solscan Pro provider not configured for concentration (no api_key)")
        return None
    headers = {"token": api_key}

    holders_url = "https://pro-api.solscan.io/v2.0/token/holders"
    params = {"address": token_address, "page": 1, "page_size": int(max(1, top_n))}
    hdata = await _json_request(session, "GET", holders_url, headers=headers, params=params, timeout=timeout)
    if not hdata:
        logger.debug("Solscan Pro holders returned no data for %s", token_address)
        return None
    hd = hdata.get("data") or {}
    items = hd.get("items") or []
    decimals = hd.get("decimals")
    top_sum_raw = 0.0
    for it in items:
        amt = it.get("amount")
        if amt is None:
            continue
        top_sum_raw += _num(amt, 0.0)

    info_url = "https://pro-api.solscan.io/v2.0/token/info"
    params_info = {"address": token_address}
    idata = await _json_request(session, "GET", info_url, headers=headers, params=params_info, timeout=timeout)
    if not idata:
        logger.debug("Solscan Pro token info returned no data for %s", token_address)
        return None
    idd = idata.get("data") or {}
    supply_raw = idd.get("supply")
    dec = idd.get("decimals") if idd.get("decimals") is not None else decimals
    if supply_raw is None or dec is None:
        logger.debug("Solscan Pro missing supply/decimals for %s", token_address)
        return None
    scale = 10 ** int(dec)
    total_supply_units = _num(supply_raw, 0.0) / scale
    top_sum_units = top_sum_raw / scale
    if total_supply_units <= 0:
        logger.debug("Solscan Pro total supply <= 0 for %s", token_address)
        return None
    return top_sum_units, total_supply_units


async def _helius_supply_and_top(
    token_address: str,
    session: aiohttp.ClientSession,
    cfg: Dict[str, Any],
    top_n: int,
    timeout: aiohttp.ClientTimeout,
) -> Optional[Tuple[float, float]]:
    p = (cfg.get("providers", {}) or {}).get("helius", {}) or {}
    api_key = p.get("api_key") or ""
    if not api_key:
        logger.debug("Helius provider not configured for concentration (no api_key)")
        return None
    url = p.get("url") or f"https://mainnet.helius-rpc.com/?api-key={api_key}"
    method_top = p.get("method_top") or "getTokenTopHolders"
    method_summary = p.get("method_summary") or "getTokenHoldersSummary"

    payload_top = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method_top,
        "params": {"mint": token_address, "limit": int(max(1, top_n))},
    }
    tdata = await _json_request(session, "POST", url, json_body=payload_top, timeout=timeout)
    if not tdata:
        logger.debug("Helius top holders returned no data for %s", token_address)
        return None
    titems = (tdata or {}).get("result") or []
    top_sum_units = 0.0
    for it in titems:
        qty = _best_first(it.get("uiAmount"), it.get("amountUi"))
        if qty is None:
            continue
        top_sum_units += _num(qty, 0.0)

    payload_sum = {"jsonrpc": "2.0", "id": 2, "method": method_summary, "params": {"mint": token_address}}
    sdata = await _json_request(session, "POST", url, json_body=payload_sum, timeout=timeout)
    if not sdata:
        logger.debug("Helius summary returned no data for %s", token_address)
        return None
    result = (sdata or {}).get("result") or {}
    supply_units = _best_first(result.get("supplyUi"), result.get("supply"), result.get("circulatingUi"))
    if supply_units is None:
        logger.debug("Helius supply missing for %s", token_address)
        return None
    total_supply_units = _num(supply_units, 0.0)
    if total_supply_units <= 0:
        logger.debug("Helius total supply <= 0 for %s", token_address)
        return None
    return top_sum_units, total_supply_units


async def _public_solscan_supply_and_top(
    token_address: str,
    session: aiohttp.ClientSession,
    top_n: int,
    timeout: aiohttp.ClientTimeout,
) -> Optional[Tuple[float, float]]:
    holders_url = "https://public-api.solscan.io/token/holders"
    params = {"tokenAddress": token_address, "limit": int(max(1, top_n)), "offset": 0}
    hdata = await _json_request(session, "GET", holders_url, params=params, timeout=timeout)
    if not hdata:
        logger.debug("Public Solscan top holders returned no data for %s", token_address)
        return None
    data = hdata
    if isinstance(hdata, dict) and isinstance(hdata.get("data"), dict):
        items = hdata["data"].get("items")
    else:
        items = (data.get("holders") or data.get("items") or []) if isinstance(data, dict) else []
    top_sum_units = 0.0
    for it in items or []:
        qty = _best_first(it.get("uiAmount"), it.get("amountUi"), it.get("amount"))
        if qty is None:
            continue
        top_sum_units += _num(qty, 0.0)

    info_url = "https://public-api.solscan.io/token/meta"
    params2 = {"tokenAddress": token_address}
    idata = await _json_request(session, "GET", info_url, params=params2, timeout=timeout)
    if not idata:
        logger.debug("Public Solscan token meta returned no data for %s", token_address)
        return None
    supply_units = _best_first(
        (idata.get("data") or {}).get("supplyUi") if isinstance(idata.get("data"), dict) else None,
        idata.get("supplyUi"),
        idata.get("supply"),
    )
    if supply_units is None:
        logger.debug("Public Solscan supply missing for %s", token_address)
        return None
    total_supply_units = _num(supply_units, 0.0)
    if total_supply_units <= 0:
        logger.debug("Public Solscan supply <= 0 for %s", token_address)
        return None
    return top_sum_units, total_supply_units


async def fetch_top_holder_concentration(

    token_address: str,
    session: aiohttp.ClientSession,
    config: Dict[str, Any],
    top_n: int,
) -> Tuple[Optional[float], str]:
    cache_key = (token_address, int(max(1, top_n)))
    cached = _conc_cache.get(cache_key)
    if isinstance(cached, (int, float)):
        return float(cached), "cache"

    hcfg = config.get("holders", {}) or {}
    timeout_sec = int(hcfg.get("timeout_seconds", 8))
    timeout = aiohttp.ClientTimeout(total=timeout_sec, sock_connect=5, sock_read=timeout_sec)

    pair = await _birdeye_supply_and_top(token_address, session, hcfg, top_n, timeout)
    if pair:
        top_sum, supply = pair
        pct = (top_sum / supply) * 100.0
        _conc_cache[cache_key] = pct
        return pct, "birdeye"

    pair = await _solscan_supply_and_top(token_address, session, hcfg, top_n, timeout)
    if pair:
        top_sum, supply = pair
        pct = (top_sum / supply) * 100.0
        _conc_cache[cache_key] = pct
        return pct, "solscan_pro"

    pair = await _helius_supply_and_top(token_address, session, hcfg, top_n, timeout)
    if pair:
        top_sum, supply = pair
        pct = (top_sum / supply) * 100.0
        _conc_cache[cache_key] = pct
        return pct, "helius"

    if (hcfg.get("providers", {}).get("public_solscan", {}) or {}).get("enabled", True):
        pair = await _public_solscan_supply_and_top(token_address, session, top_n, timeout)
        if pair:
            top_sum, supply = pair
            pct = (top_sum / supply) * 100.0
            _conc_cache[cache_key] = pct
            return pct, "public_solscan"

    logger.debug("All concentration providers returned no data for %s", token_address)
    return None, "unavailable"


def _max_conc_for_categories(categories: List[str], config: Dict[str, Any]) -> Optional[float]:
    ccfg = config.get("concentration", {}) or {}
    per = ccfg.get("max_percent_per_category", {}) or {}
    caps: List[float] = []
    for cat in categories:
        v = per.get(cat)
        if isinstance(v, (int, float)):
            caps.append(float(v))
    if caps:
        return float(min(caps))
    global_cap = ccfg.get("max_percent")
    if isinstance(global_cap, (int, float)):
        return float(global_cap)
    return None


# -----------------------------------------------------------------------------#
# Eligibility (with holders + concentration)  — plus single-primary category
# -----------------------------------------------------------------------------#

def _primary_category_list(categories: List[str]) -> List[str]:
    """
    Reduce any list of category tags to a single primary tag.
    Precedence: newly_launched > large_cap > mid_cap > low_cap > unknown_cap
    """
    s = {c.strip().lower() for c in categories or []}
    if "newly_launched" in s:
        return ["newly_launched"]
    if "large_cap" in s:
        return ["large_cap"]
    if "mid_cap" in s:
        return ["mid_cap"]
    if "low_cap" in s:
        return ["low_cap"]
    return ["unknown_cap"] if s else ["unknown_cap"]


async def is_token_eligible(
    token: Dict[str, Any],
    session: aiohttp.ClientSession,
    config: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Check if a token is eligible for trading based on discovery criteria, RugCheck (optional at discovery),
    and optional holders/concentration policies.
    Returns: (eligible: bool, [single_primary_category])
    """
    token_address = token.get("address")
    if not token_address:
        logger.warning("Token missing address: %s", token)
        return False, []

    # Determine if we should dump token details for debugging
    should_dump_addr = False
    for a in DUMP_TOKEN_ADDRS:
        if a and str(token_address).startswith(a):
            should_dump_addr = True
            break

    def _maybe_dump(when: str = "pre-eligibility") -> None:
        # when: "pre-eligibility" or "reject"
        try:
            if should_dump_addr or DUMP_TOKEN_ON_REJECT:
                logger.info("DEBUG TOKEN DUMP (%s) %s: %s", when, token_address, json.dumps(token, indent=2, default=str, ensure_ascii=False))
        except Exception:
            logger.exception("Failed to dump token for debug")
       
    # Immediately ignore WSOL/stables in category selection flows
    if token_address in ALWAYS_HIDE_IN_CATEGORIES:
        logger.debug("Ignoring stable/WSOL token in eligibility flow: %s", token_address)
        return False, []
    
    # Enforce configured scoring.hard_floor (sync helper)
    try:
        ok, hf_reason = enforce_scoring_hard_floor(token, config)
        if not ok:
            # annotate + reject consistently with other rejects
            reasons.append(hf_reason)
            return _reject(hf_reason)
    except Exception:
        # If enforcement fails unexpectedly, be conservative: allow token so other checks can decide
        logger.debug("hard_floor enforcement failed (continuing): %s", traceback.format_exc()) 
        
    # Helper to centralize reject logging + optional dump
    def _reject(reason: str, annotate: bool = True, dump_when: str = "reject") -> Tuple[bool, List[str]]:
        try:
            if reason:
                _log_reject(token_address, reason)
            if annotate:
                token_local = dict(token)
                token_local["eligibility_reasons"] = token_local.get("eligibility_reasons", []) + [reason] if reason else token_local.get("eligibility_reasons", [])
                # attempt to attach to original token object as well for callers
                try:
                    token.update(token_local)
                except Exception:
                    logger.debug("Failed to attach eligibility_reasons to token for %s", token_address, exc_info=True)
            # dump if requested either specifically for address prefix, or global reject flag
            if should_dump_addr or DUMP_TOKEN_ON_REJECT:
                try:
                    logger.info("DEBUG TOKEN DUMP (%s) %s: %s", dump_when, token_address, json.dumps(token, indent=2, default=str, ensure_ascii=False))
                except Exception:
                    logger.exception("Failed to dump token for debug")
        except Exception:
            logger.exception("Error while rejecting token")
        # Return a safe empty categories list here to avoid referencing a variable that may not yet be defined.
        return False, []

    # --- NEW: Block symbol-squat SOL-like tokens early (mint-based whitelist required) ---
    try:
        sym = (token.get("symbol") or "").strip().lower()
        addr = (token_address or "").strip()

        # suspicious symbols (exact-match; extend if you want substring rules)
        _sol_like_syms = {
            "sol", "solana", "wsol", "wrapped sol", "wrapped solana", "wrapped-sol", "w sol"
        }

        # If the symbol looks like SOL but mint is not trusted, reject via central helper
        if sym in _sol_like_syms and addr not in TRUSTED_SOL_MINTS:
            reason = "symbol looks like SOL but mint not whitelisted"
            return _reject(reason)
    except Exception:
        # Fail-safe: log and continue (do not crash eligibility)
        logger.debug("SOL-symbol guard check failed for %s", token_address, exc_info=True)    

    # Optionally dump pre-eligibility matched tokens (rare; helpful)
    if should_dump_addr:
        _maybe_dump("pre-eligibility")

    categories: List[str] = []
    reasons: List[str] = []

    # --- Dexscreener / Birdeye fallbacks ---
    ds = token.get("dexscreener") or {}
    be = token.get("birdeye") or {}

    # Normalize core fields (prefer normalized, then Birdeye, then Dexscreener)
    market_cap = _num(_best_first(
        token.get("mc"),
        token.get("fdv"),
        token.get("market_cap"),
        be.get("market_cap") if isinstance(be, dict) else None,
        ds.get("fdv"),
        ds.get("mcap"),
        ds.get("marketCap"),
    ), 0)

    liquidity = _num(_best_first(
        token.get("liquidity"),
        token.get("liquidity_usd"),
        ds.get("liquidityUsd"),
        (ds.get("liquidity") or {}).get("usd") if isinstance(ds.get("liquidity"), dict) else None,
    ), 0)

    volume_24h = _num(_best_first(
        token.get("volume_24h"),
        token.get("v24hUSD"),
        token.get("volume"),
        ds.get("v24hUsd") or ds.get("v24hUSD"),
        ds.get("volume24h") or ds.get("h24"),
        ds.get("volume"),
    ), 0)

    # creation: accept seconds or ms; also accept Dexscreener pairCreatedAt
    creation_ts_sec = _best_first(
        token.get("creation_timestamp"),
        (token.get("pairCreatedAt") or 0) / 1000.0 if token.get("pairCreatedAt") else None,
        (ds.get("pairCreatedAt") or 0) / 1000.0 if ds.get("pairCreatedAt") else None,
    )
    creation_ts_sec = _num(creation_ts_sec or 0)
    pair_created_at_ms = int(creation_ts_sec * 1000) if creation_ts_sec else 0

    # Assign categories by market cap (robust to missing config keys)
    try:
        if market_cap > 0:
            # safe lookups so missing discovery/mid_cap/low_cap don't raise KeyError
            disc = (config or {}).get("discovery") or {}
            low_cfg = (disc.get("low_cap") or {}) 
            mid_cfg = (disc.get("mid_cap") or {})
            low_max = _num(low_cfg.get("max_market_cap"), 1e6)
            mid_max = _num(mid_cfg.get("max_market_cap"), 1e8)

            if market_cap < low_max:
                categories.append("low_cap")
            elif market_cap < mid_max:
                categories.append("mid_cap")
            else:
                categories.append("large_cap")
    except Exception as e:
        # include the actual exception object in the formatted message and pass exc_info keyword
        logger.debug("Failed to determine market cap category for %s: %s", token_address, e, exc_info=True)

    # "newly_launched" by age
    if pair_created_at_ms:
        age_minutes = (time.time() - pair_created_at_ms / 1000.0) / 60.0
        try:
            max_new_age = _num(_disc_get(config, "newly_launched", "max_token_age_minutes"), 180.0)
        except Exception:
            max_new_age = 180.0
        if age_minutes <= max_new_age:
            categories.append("newly_launched")

    # If nothing yet, keep token visible downstream
    if not categories:
        categories = ["unknown_cap"]

    # Reduce to a single primary category *before* thresholds/logics below
    categories = _primary_category_list(categories)

    # ---------------------- Guardrails: market cap ----------------------
    _min_mc = 1000  # default; can be overridden via config["discovery"]["min_market_cap"]
    try:
        _min_mc = _num(config.get("discovery", {}).get("min_market_cap", 1000), 1000)
    except Exception:
        pass

    if market_cap > 0 and market_cap < _min_mc:
        reason = f"market_cap={market_cap} too low (<{_min_mc})"
        reasons.append(reason)
        return _reject(reason)

    # Optional upper bound sanity
    try:
        mid_cfg = (config or {}).get("discovery", {}).get("mid_cap", {}) or {}
        mid_cap_max = _num(mid_cfg.get("max_market_cap"), 1e8)
        if market_cap > 0 and market_cap > mid_cap_max and categories == ["mid_cap"]:
            reason = f"market_cap={market_cap} too high (>mid_cap_max={mid_cap_max})"
            reasons.append(reason)
            return _reject(reason)
    except Exception:
        pass

    # ---------------------- Guardrails: liquidity ----------------------
    def _liquidity_threshold(cat: str) -> float:
        try:
            v = _disc_get(config, cat, "liquidity_threshold", None)
            return _num(v, MIN_LIQ_USD_ENV)
        except Exception:
            return MIN_LIQ_USD_ENV

    cat = categories[0]
    if cat == "low_cap" and liquidity < _liquidity_threshold("low_cap"):
        reason = f"liquidity={liquidity} too low for low_cap"
        reasons.append(reason)
        return _reject(reason)
    if cat == "mid_cap" and liquidity < _liquidity_threshold("mid_cap"):
        reason = f"liquidity={liquidity} too low for mid_cap"
        reasons.append(reason)
        return _reject(reason)
    if cat == "large_cap" and liquidity < _liquidity_threshold("large_cap"):
        reason = f"liquidity={liquidity} too low for large_cap"
        reasons.append(reason)
        return _reject(reason)
    if cat == "newly_launched":
        try:
            nl_th = _num(_disc_get(config, "newly_launched", "liquidity_threshold"), MIN_LIQ_USD_ENV)
        except Exception:
            nl_th = MIN_LIQ_USD_ENV
        if liquidity < nl_th:
            reason = f"liquidity={liquidity} too low for newly_launched"
            reasons.append(reason)
            return _reject(reason)

    # ---------------------- Guardrails: volume ----------------------
    def _volume_threshold(cat: str) -> float:
        try:
            v = _disc_get(config, cat, "volume_threshold", None)
            return _num(v, MIN_VOL24_USD_ENV)
        except Exception:
            return MIN_VOL24_USD_ENV

    if cat == "low_cap" and volume_24h < _volume_threshold("low_cap"):
        reason = f"volume_24h={volume_24h} too low for low_cap"
        reasons.append(reason)
        return _reject(reason)
    if cat == "mid_cap" and volume_24h < _volume_threshold("mid_cap"):
        reason = f"volume_24h={volume_24h} too low for mid_cap"
        reasons.append(reason)
        return _reject(reason)
    if cat == "large_cap" and volume_24h < _volume_threshold("large_cap"):
        reason = f"volume_24h={volume_24h} too low for large_cap"
        reasons.append(reason)
        return _reject(reason)
    if cat == "newly_launched" and volume_24h < _volume_threshold("newly_launched"):
        reason = f"volume_24h={volume_24h} too low for newly_launched"
        reasons.append(reason)
        return _reject(reason)

    # ---------------------- Guardrail: recent momentum (1h/6h) ----------------------
    recent_ok, recent_reason = _recent_momentum_allowed(token, cfg=config)
    if not recent_ok:
        reason = recent_reason
        reasons.append(reason)
        return _reject(reason)
    else:
        token.setdefault("eligibility_reasons", [])
        token["eligibility_reasons"].append(f"recent_momentum_ok: {recent_reason}")

    # ---------------------- Guardrail: price change ----------------------
    price_change_24h = _num(_best_first(
        token.get("price_change_24h"),
        token.get("priceChange24h"),
        ds.get("priceChange24h"),
        ds.get("priceChange"),
    ), 0)

    allowed, pc_reason = _price_change_allowed(token, price_change_24h, cfg=config)

    if not allowed:
        # Attempt an audited, conservative override instead of immediate reject.
        # For override we require RugCheck to have been run (and pass). Run a lightweight RugCheck now
        # (verify_token_with_rugcheck caches results so this is cheap if run later as well).
        try:
            rugcheck_ok = False
            rcfg = (config.get("rugcheck") or {}) if isinstance(config, dict) else {}
            core_rc_enabled = bool(rcfg.get("enabled", False))

            if core_rc_enabled:
                try:
                    risk_score, risk_labels_raw, rc_msg = await verify_token_with_rugcheck(
                        token_address, token, session, config
                    )
                except Exception:
                    # If RugCheck call fails, treat rugcheck as failing for override purposes.
                    logger.debug("verify_token_with_rugcheck failed during override check for %s", token_address, exc_info=True)
                    risk_score = 9999.0
                    risk_labels_raw = []
                    rc_msg = "rugcheck_call_error"

                token["rugcheck_score"] = float(risk_score)
                token["rugcheck_labels"] = _label_names_from_any(risk_labels_raw)
                token["_rugcheck_msg"] = str(rc_msg or "")

                # If the RugCheck helper explicitly marked the API as down (or returned an "allowed" fallback),
                # treat that as NOT acceptable for an automatic override (defensive).
                rc_api_down = False
                try:
                    rc_msg_lc = "" if rc_msg is None else str(rc_msg).lower()
                    # detect phrases like "api down" or the fallback "api down - allowed"
                    if "api down" in rc_msg_lc or "allowed" in rc_msg_lc and "api" in rc_msg_lc:
                        rc_api_down = True
                except Exception:
                    rc_api_down = False

                # Compute max_rug_score according to category (reuse same defaults as later in file)
                def _max_rug_local(cat: str, default_val: float = 80.0) -> float:
                    try:
                        return _num(_disc_get(config, cat, "max_rugcheck_score", default_val), default_val)
                    except Exception:
                        return float(default_val)

                if cat == "newly_launched":
                    max_rug_score = _max_rug_local("newly_launched")
                elif cat == "mid_cap":
                    max_rug_score = _max_rug_local("mid_cap")
                elif cat == "large_cap":
                    max_rug_score = _max_rug_local("large_cap")
                else:
                    max_rug_score = _max_rug_local("low_cap")

                # Only mark rugcheck_ok True if risk_score is within the allowed threshold AND API was not reported down.
                rugcheck_ok = (float(risk_score) <= float(max_rug_score)) and (not rc_api_down)
            else:
                # RugCheck not enabled; do NOT treat this as a pass for override purposes.
                rugcheck_ok = False
        except Exception:
            # Any unexpected error -> do not allow override
            logger.exception("Unexpected error while computing rugcheck_ok for override for %s", token_address)
            rugcheck_ok = False

        # Conservative depth thresholds required for any override (category-specific if configured)
        try:
            min_liq_for_override = _num(_disc_get(config, cat, "liquidity_threshold", 5000.0), 5000.0)
            min_vol_for_override = _num(_disc_get(config, cat, "volume_threshold", 10000.0), 10000.0)
        except Exception:
            min_liq_for_override, min_vol_for_override = 5000.0, 10000.0

        # If rugcheck did not explicitly pass, block override now
        if not rugcheck_ok:
            reason = f"override blocked: rugcheck not enabled/passing (score={token.get('rugcheck_score', 'N/A')})"
            reasons.append(reason)
            return _reject(reason)

        # Require either liquidity OR 24h volume minimum to consider override
        if float(liquidity or 0.0) < float(min_liq_for_override) and float(volume_24h or 0.0) < float(min_vol_for_override):
            reason = (
                f"override blocked: insufficient market depth for safe override "
                f"(liq={liquidity}<{min_liq_for_override}, vol24h={volume_24h}<{min_vol_for_override})"
            )
            reasons.append(reason)
            return _reject(reason)

        # At this point rugcheck_ok == True AND depth thresholds satisfied -> proceed to DB-assisted override helper.
        # (existing threaded call to should_override_extreme_price remains unchanged below)

        # Now attempt override (this will consult cached metrics or Dexscreener)
        # Determine DB path to use for journaling (config first, env fallback second)
        DB_PATH_DEFAULT = os.path.expanduser(r"C:\Users\Admin\AppData\Local\SOLOTradingBot\tokens.sqlite3")
        db_path = None
        try:
            db_path = config.get("db_path") or os.getenv("SOLO_TOKENS_DB") or DB_PATH_DEFAULT
        except Exception:
            db_path = DB_PATH_DEFAULT

        # run the blocking override helper in a thread so we don't block the event loop
        allow_override = False
        override_detail: Dict[str, Any] = {}

        def _run_override_with_own_conn() -> tuple[bool, Dict[str, Any]]:
            """
            Open sqlite3 connection in this worker thread, call helper, close conn.
            Returns (allowed: bool, detail: dict).
            Accepts helper forms that return either bool or (bool, detail).
            """
            try:
                conn_local = sqlite3.connect(db_path, timeout=5)
            except Exception as e:
                logger.warning("Override DB open failed for %s: %s", token_address, e)
                return False, {}

            try:
                try:
                    res = should_override_extreme_price(
                        conn_local,
                        token_address,
                        float(price_change_24h),
                        float(volume_24h if volume_24h is not None else 0.0),
                        float(liquidity if liquidity is not None else 0.0),
                        bool(rugcheck_ok),
                    )
                    # Normalize helper return: support bool or (bool, detail)
                    if isinstance(res, (tuple, list)) and len(res) >= 1:
                        allowed = bool(res[0])
                        detail = res[1] if len(res) > 1 and isinstance(res[1], dict) else {}
                    else:
                        allowed = bool(res)
                        detail = {}
                    return allowed, detail
                except Exception as e:
                    # Promote to warning so override failures are visible (previously debug)
                    logger.warning("should_override_extreme_price raised for %s: %s", token_address, e, exc_info=True)
                    return False, {}
            finally:
                try:
                    conn_local.close()
                except Exception:
                    pass

        # Execute the blocking work on a thread so the asyncio event loop is not blocked.
        try:
            if hasattr(asyncio, "to_thread"):
                allow_override, override_detail = await asyncio.to_thread(_run_override_with_own_conn)
            else:
                loop = asyncio.get_event_loop()
                allow_override, override_detail = await loop.run_in_executor(None, _run_override_with_own_conn)
        except Exception as e:
            logger.warning("Override thread execution failure for %s: %s", token_address, e, exc_info=True)
            allow_override, override_detail = False, {}

        # Continue logic based on allow_override...
        if allow_override:
            token.setdefault("eligibility_reasons", [])
            token["eligibility_reasons"].append(f"ELIG-OVERRIDE by dexscreener/cached_metrics (price_change={price_change_24h})")
            # attach structured override detail (small) for observability/UI
            try:
                token["eligibility_override"] = override_detail
            except Exception:
                logger.debug("Failed to attach override_detail for %s", token_address, exc_info=True)
            logger.info("ELIG-OVERRIDE %s price_change=%.2f detected_by=override reason=%s evidence=%s",
                        token_address, float(price_change_24h),
                        override_detail.get("reason") if isinstance(override_detail, dict) else None,
                        json.dumps(override_detail.get("evidence", {})) if isinstance(override_detail, dict) else "")
        else:
            reason = pc_reason
            reasons.append(reason)
            return _reject(reason)
    else:
        token.setdefault("eligibility_reasons", [])
        token["eligibility_reasons"].append(f"price_change_ok: {pc_reason}")

    # ---------------------- Rugcheck at discovery (optional) ----------------------
    disc_cfg = (config.get("discovery") or {})
    use_rc = get_env_bool("RUGCHECK_DISCOVERY_CHECK", False) or bool(disc_cfg.get("rugcheck_in_discovery", False))
    filter_rc = get_env_bool("RUGCHECK_DISCOVERY_FILTER", False)
    if disc_cfg.get("require_rugcheck_pass", None) is not None:
        filter_rc = bool(disc_cfg.get("require_rugcheck_pass"))
        if filter_rc:
            use_rc = True

    rc_status, core_on, disc_on, filt_on = _compose_rugcheck_status(config)
    token.setdefault("_rc_status", rc_status)

    # If we are skipping RugCheck at discovery, just annotate and continue.
    if not use_rc:
        token.setdefault("safety", token.get("safety", "unknown"))
        token.setdefault("dangerous", bool(token.get("dangerous", False)))
        token.setdefault("eligibility_reasons", [])
        token["eligibility_reasons"].append(f"rugcheck: {rc_status} (skipped at discovery)")
        logger.debug(
            "RugCheck at discovery — %s — skipping for %s (safety=%s, dangerous=%s)",
            rc_status, token_address, token.get("safety"), token.get("dangerous")
        )
        risk_score = 0.0
        risk_label_names: List[str] = []
    else:
        # Run verification
        try:
            risk_score, risk_labels_raw, _ = await verify_token_with_rugcheck(
                token_address, token, session, config
            )
        except Exception as e:
            logger.warning("RugCheck call failed for %s: %s", token_address, e, exc_info=True)
            # If RugCheck fails here, honor allow_if_api_down if configured
            rcfg = (config.get("rugcheck") or {})
            if bool(rcfg.get("allow_if_api_down", True)):
                logger.warning("RugCheck allow_if_api_down=True -> permitting %s while API failed: %s", token_address, e)
                risk_score, risk_labels_raw = 0.0, []
            else:
                # treat as hard reject
                reason = f"RugCheck API failed: {e}"
                return _reject(reason)
        risk_label_names = _label_names_from_any(risk_labels_raw)

    token["rugcheck_labels"] = risk_label_names
    token["rugcheck_score"] = float(risk_score)
    token["safety"] = "ok"
    token["dangerous"] = False

    if use_rc and filter_rc:
        block_dangerous_env = get_env_bool("BLOCK_RUGCHECK_DANGEROUS", True)
        block_dangerous_cfg = disc_cfg.get("require_rugcheck_pass", None)
        block_dangerous = block_dangerous_cfg if block_dangerous_cfg is not None else block_dangerous_env

        danger_blocklist: Set[str] = {"dangerous", "scam", "honeypot"}
        try:
            cfg_labels = (config.get("rugcheck", {}) or {}).get("danger_labels_blocklist")
            if isinstance(cfg_labels, list) and cfg_labels:
                danger_blocklist = {_lower(x) for x in (set(cfg_labels) | danger_blocklist)}
        except Exception as e:
            logger.warning("Failed to load rugcheck.danger_labels_blocklist: %s", e)

        if block_dangerous:
            hit = set(risk_label_names) & danger_blocklist
            if hit:
                msg = f"Dangerous label(s): {sorted(hit)}"
                _log_reject(token_address, msg)
                try:
                    await add_to_blacklist(token_address, msg)
                except Exception:
                    logger.exception("Failed to add to blacklist for %s", token_address)
                token["safety"] = "dangerous"
                token["dangerous"] = True
                token = dict(token)
                token["eligibility_reasons"] = [msg]
                return _reject(msg)

        def _max_rug(cat: str, default_val: float = 80.0) -> float:
            try:
                return _num(_disc_get(config, cat, "max_rugcheck_score", default_val), default_val)
            except Exception:
                return float(default_val)

        if cat == "newly_launched":
            max_rug_score = _max_rug("newly_launched")
        elif cat == "mid_cap":
            max_rug_score = _max_rug("mid_cap")
        elif cat == "large_cap":
            max_rug_score = _max_rug("large_cap")
        else:
            max_rug_score = _max_rug("low_cap")

        if risk_score > max_rug_score:
            msg = f"RugCheck high risk (score={risk_score} >= limit={max_rug_score})"
            _log_reject(token_address, msg)
            try:
                await add_to_blacklist(token_address, msg)
            except Exception:
                logger.exception("Failed to add to blacklist for %s", token_address)
            token["safety"] = "dangerous"
            token["dangerous"] = True
            token = dict(token)
            token["eligibility_reasons"] = [msg]
            return _reject(msg)

    if ("dangerous" in risk_label_names) or ("scam" in risk_label_names) or ("honeypot" in risk_label_names):
        token["safety"] = "dangerous"
        token["dangerous"] = True
    elif risk_score >= 1e6:
        token["safety"] = "unknown"
        token["dangerous"] = False
    else:
        token["safety"] = "ok"
        token["dangerous"] = False

    # -------------------- Holders / Concentration (unchanged) --------------------
    hcfg = config.get("holders", {}) or {}
    if hcfg.get("enabled", True):
        min_required = _min_holders_for_categories(categories, config)
        if isinstance(min_required, int):
            allow_if_down = bool(hcfg.get("allow_if_api_down", True))
            holders_count, source_h = await fetch_holder_count(token_address, session, config)
            if holders_count is None:
                note = f"holders unavailable (src={source_h})"
                if allow_if_down:
                    logger.warning(
                        "Holder count unavailable for %s; proceeding due to allow_if_api_down (src=%s)",
                        token_address, source_h,
                    )
                    categories.append("pending_holder_count")
                else:
                    _log_reject(token_address, note)
                    try:
                        await add_to_blacklist(token_address, note)
                    except Exception:
                        logger.exception("Failed to add to blacklist for %s", token_address)
                    token = dict(token)
                    token["eligibility_reasons"] = [note]
                    return _reject(note)
            else:
                token["holders"] = holders_count
                token["holders_source"] = source_h
                if holders_count < min_required:
                    msg = f"holders={holders_count} < min_required={min_required} (src={source_h})"
                    _log_reject(token_address, msg)
                    try:
                        await add_to_blacklist(token_address, msg)
                    except Exception:
                        logger.exception("Failed to add to blacklist for %s", token_address)
                    token = dict(token)
                    token["eligibility_reasons"] = [msg]
                    return _reject(msg)

    ccfg = config.get("concentration", {}) or {}
    if ccfg.get("enabled", False):
        top_n = int(max(1, ccfg.get("top_n", 10)))
        allow_if_down_c = bool(ccfg.get("allow_if_api_down", True))
        max_pct_allowed = _max_conc_for_categories(categories, config)

        pct, source_c = await fetch_top_holder_concentration(token_address, session, config, top_n)
        if pct is None:
            note = f"concentration unavailable (src={source_c})"
            if not allow_if_down_c:
                _log_reject(token_address, note)
                try:
                    await add_to_blacklist(token_address, note)
                except Exception:
                    logger.exception("Failed to add to blacklist for %s", token_address)
                token = dict(token)
                token["eligibility_reasons"] = [note]
                return _reject(note)
            else:
                logger.warning(
                    "Concentration unavailable for %s; proceeding due to allow_if_api_down (src=%s)",
                    token_address, source_c,
                )
        else:
            token["topN_concentration_pct"] = pct
            token["topN_concentration_source"] = source_c
            if isinstance(max_pct_allowed, (int, float)) and pct > float(max_pct_allowed):
                msg = f"top{top_n}_concentration={pct:.2f}% > limit={float(max_pct_allowed):.2f}% (src={source_c})"
                _log_reject(token_address, msg)
                try:
                    await add_to_blacklist(token_address, msg)
                except Exception:
                    logger.exception("Failed to add to blacklist for %s", token_address)
                token = dict(token)
                token["eligibility_reasons"] = [msg]
                return _reject(msg)

    logger.debug("ELIGIBLE %s | category=%s", token_address, categories)
    return True, categories


# -----------------------------------------------------------------------------#
# Scoring + Selection
# -----------------------------------------------------------------------------#
def score_token(token: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Score a token based on various metrics (field names normalized).
    Returns [0,1].

    Note: price-change contribution is amplified when market depth indicates a real runner.
    """
    ds = token.get("dexscreener") or {}
    be = token.get("birdeye") or {}

    market_cap = _num(_best_first(
        token.get("mc"),
        token.get("fdv"),
        token.get("market_cap"),
        be.get("market_cap") if isinstance(be, dict) else None,
        ds.get("fdv"),
        ds.get("mcap"),
        ds.get("marketCap"),
    ), 0)

    liquidity = _num(_best_first(
        token.get("liquidity"),
        token.get("liquidity_usd"),
        ds.get("liquidityUsd"),
        (ds.get("liquidity") or {}).get("usd") if isinstance(ds.get("liquidity"), dict) else None,
    ), 0)

    volume_24h = _num(_best_first(
        token.get("volume_24h"),
        token.get("v24hUSD"),
        token.get("volume"),
        ds.get("v24hUsd") or ds.get("v24hUSD"),
        ds.get("volume24h") or ds.get("h24"),
        ds.get("volume"),
    ), 0)

    # Treat negative 24h % as zero for scoring (we don't want negative momentum to be rewarded).
    price_change_24h = _num(_best_first(
        token.get("price_change_24h"),
        token.get("priceChange24h"),
        ds.get("priceChange24h"),
        ds.get("priceChange"),
    ), 0.0)
    price_change_24h = max(0.0, price_change_24h)

    weights = config.get(
        "weights",
        {
            "market_cap": 0.3,
            "liquidity": 0.2,
            "volume": 0.3,
            "price_change": 0.2,
            "new_token_bonus": 0.3,
        },
    )

    score = 0.0

    max_market_cap = _num(
        config.get("discovery", {}).get("mid_cap", {}).get("max_market_cap", 1e9),
        1e9,
    )
    if market_cap > 0 and max_market_cap > 0:
        score += (max(0.0, max_market_cap - market_cap) / max_market_cap) * weights.get("market_cap", 0.3)

    max_liquidity = max(
        _num(config.get("discovery", {}).get("low_cap", {}).get("liquidity_threshold", 1.0), 1.0),
        _num(config.get("discovery", {}).get("mid_cap", {}).get("liquidity_threshold", 1.0), 1.0),
        _num(config.get("discovery", {}).get("large_cap", {}).get("liquidity_threshold", 1.0), 1.0),
    )
    if liquidity > 0 and max_liquidity > 0:
        score += min(liquidity / max_liquidity, 1.0) * weights.get("liquidity", 0.2)

    max_volume = max(
        _num(config.get("discovery", {}).get("low_cap", {}).get("volume_threshold", 1.0), 1.0),
        _num(config.get("discovery", {}).get("mid_cap", {}).get("volume_threshold", 1.0), 1.0),
        _num(config.get("discovery", {}).get("large_cap", {}).get("volume_threshold", 1.0), 1.0),
    )
    if volume_24h > 0 and max_volume > 0:
        score += min(volume_24h / max_volume, 1.0) * weights.get("volume", 0.3)

    # Price change scoring: encourage real runners.
    max_price_change_allowed = _num(config.get("discovery", {}).get("max_price_change", 350), 350)
    # Depth amplification: if token has decent depth amplify price-change contribution
    min_liq_for_big_pct = _num(config.get("discovery", {}).get("min_liq_for_big_pct", 5000.0), 5000.0)
    min_vol_for_big_pct = _num(config.get("discovery", {}).get("min_vol_for_big_pct", 10000.0), 10000.0)
    depth_multiplier = 1.0
    if liquidity >= min_liq_for_big_pct or volume_24h >= min_vol_for_big_pct:
        depth_multiplier = 1.5  # boost price-change influence for tokens with real market depth

    if price_change_24h > 0 and max_price_change_allowed > 0:
        pct_score = min(price_change_24h / max_price_change_allowed, 1.0) * weights.get("price_change", 0.2)
        score += pct_score * depth_multiplier

    if "newly_launched" in token.get("categories", []):
        score += weights.get("new_token_bonus", 0.0)

    return max(0.0, min(score, 1.0))

def _apply_tech_tiebreaker(
    eligible_tokens: List[Dict[str, Any]],
    config: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """
    In-place (and return) deterministic tech tie-breaker for shortlists.

    Behavior:
    - If signals.ranking_tilt is disabled in config, no-op and return tokens.
    - Prefer to use dispatch scoring if available (score_token_dispatch), otherwise
      prefer existing 'score' field, otherwise fall back to a lightweight technical
      score (_tech_score from tradingOld).
    - Sorts tokens in-place (descending) by composite key:
        1) score or dispatch result
        2) liquidity (desc)
        3) volume_24h (desc)
        4) market cap (desc)
      This stabilizes ordering for identical scores.
    - Marks each token with "_tiebreak_applied": True (non-destructive).
    - Returns the sorted list (same object for convenience).
    """
    try:
        log = logger or logging.getLogger("TradingBot")
        if not (eligible_tokens and bool((config or {}).get("signals", {}).get("ranking_tilt", True))):
            return eligible_tokens

        # helper to get numeric safely
        def _num_for(t: dict, key: str) -> float:
            try:
                v = t.get(key)
                return float(v) if v is not None else 0.0
            except Exception:
                return 0.0

        # Compose primary key function: prefer score_token_dispatch if present
        dispatch = globals().get("score_token_dispatch")  # may be imported elsewhere
        if callable(dispatch):
            def _primary(t: dict) -> float:
                try:
                    s = dispatch(t, config)  # may return None
                except Exception:
                    s = None
                if s is None:
                    s = t.get("score", None)
                if s is None:
                    # fallback to local technical proxy if available in globals
                    _tech = globals().get("_tech_score")
                    try:
                        s = _tech(t) if callable(_tech) else 0.0
                    except Exception:
                        s = 0.0
                try:
                    return float(s)
                except Exception:
                    return 0.0
        else:
            def _primary(t: dict) -> float:
                s = t.get("score", None)
                if s is None:
                    _tech = globals().get("_tech_score")
                    try:
                        s = _tech(t) if callable(_tech) else 0.0
                    except Exception:
                        s = 0.0
                try:
                    return float(s)
                except Exception:
                    return 0.0

        # Sort using composite key (descending). Stable sort of Python ensures deterministic order for ties.
        eligible_tokens.sort(
            key=lambda x: (
                -_primary(x),
                -_num_for(x, "liquidity"),
                -_num_for(x, "volume_24h"),
                -_num_for(x, "mc"),
            )
        )

        # Mark tokens so downstream observability can tell the tie-breaker ran
        try:
            for t in eligible_tokens:
                if isinstance(t, dict):
                    t.setdefault("_tiebreak_applied", True)
        except Exception:
            logger.debug("Failed to mark tiebreak_applied", exc_info=True)

        log.debug("Applied tech tiebreaker to shortlist (count=%d).", len(eligible_tokens))
        return eligible_tokens

    except Exception as e:
        try:
            logging.getLogger("TradingBot").debug("Tech tiebreaker failed (falling back): %s", e, exc_info=True)
        except Exception:
            pass
        return eligible_tokens


# -----------------------------------------------------------------------------#
# Shortlisting (canonical buckets; no alias lists that cause UI duplication)
# -----------------------------------------------------------------------------#

# (1) Canonical bucket synonyms
_BUCKET_SYNONYMS: Dict[str, Set[str]] = {
    "high": {"high", "high_cap", "large", "large_cap"},
    "mid": {"mid", "mid_cap", "medium", "medium_cap"},
    "low": {"low", "low_cap", "small", "small_cap"},
    "new": {"new", "newly_launched", "newly_listed", "new_tokens"},
}


def _canon_bucket_for(categories: List[str]) -> Optional[str]:
    if not categories:
        return None
    cats = {str(c).lower().strip() for c in categories}
    for canon, syns in _BUCKET_SYNONYMS.items():
        if cats & syns:
            return canon
    return None


# (2) Defaults that keep obvious stables/wrappers out of discovery
_DEFAULT_EXCLUDE_SYMBOLS: Set[str] = {
    "SOL", "WSOL", "W SOL", "wSOL",
    "USDC", "USDT",
    "Wrapped SOL", "Wrapped Solana",
}
_DEFAULT_EXCLUDE_NAMES: Set[str] = {"wrapped solana", "wrapped sol"}  # substr match (lowercase)
_DEFAULT_EXCLUDE_ADDR: Set[str] = {
    # canonical Solana wrappers / stables
    "So11111111111111111111111111111111111111112",  # wSOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}


def _mc(tok: Dict[str, Any]) -> float:
    return _num(
        tok.get("mc")
        or tok.get("market_cap")
        or tok.get("fdv")
        or tok.get("fdvUsd")
        or tok.get("marketCap")
        or 0.0
    )


def _liq(tok: Dict[str, Any]) -> float:
    v = tok.get("liquidity")
    if isinstance(v, dict):
        return _num(v.get("usd"), 0.0)
    return _num(v, 0.0)


def _vol24(tok: Dict[str, Any]) -> float:
    v = tok.get("volume_24h")
    if v is not None:
        return _num(v, 0.0)
    v = tok.get("volume")
    if isinstance(v, dict):
        return _num(v.get("h24"), 0.0)
    return _num(v, 0.0)


def _created_s(tok: Dict[str, Any]) -> int:
    raw = tok.get("creation_timestamp") or tok.get("pairCreatedAt") or tok.get("createdAt") or 0
    ts = _num(raw, 0.0)
    if ts > 10**12:  # ms -> s
        ts = ts / 1000.0
    return int(ts or 0)


# (4) Config readers
def _cfg() -> Dict[str, Any]:
    try:
        return load_config() or {}
    except Exception:
        logger.debug("load_config() failed in _cfg()", exc_info=True)
        return {}


def _disc(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return (cfg.get("discovery") or {})


def _per_bucket_limit(cfg: Dict[str, Any], default: int = 5) -> int:
    d = _disc(cfg)
    v = d.get("shortlist_per_bucket", default)
    try:
        return max(1, int(v))
    except Exception:
        return default


def _rules(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = _disc(cfg)
    return {
        "new": {
            "max_age_min": _num((d.get("newly_launched") or {}).get("max_token_age_minutes"), 180),
            "liq_min":      _num((d.get("newly_launched") or {}).get("liquidity_threshold"), 100),
            "vol_min":      _num((d.get("newly_launched") or {}).get("volume_threshold"), 50),
        },
        "low": {
            "mc_max": _num((d.get("low_cap") or {}).get("max_market_cap"), 100_000),
            "liq_min": _num((d.get("low_cap") or {}).get("liquidity_threshold"), 50),
            "vol_min": _num((d.get("low_cap") or {}).get("volume_threshold"), 50),
        },
        "mid": {
            "mc_max": _num((d.get("mid_cap") or {}).get("max_market_cap"), 500_000),
            "liq_min": _num((d.get("mid_cap") or {}).get("liquidity_threshold"), 300),
            "vol_min": _num((d.get("mid_cap") or {}).get("volume_threshold"), 100),
        },
        "high": {
            "liq_min": _num((d.get("large_cap") or {}).get("liquidity_threshold"), 1000),
            "vol_min": _num((d.get("large_cap") or {}).get("volume_threshold"), 500),
        },
    }


# (5) Global de-dup
def _better_row(new_t: Dict[str, Any], old_t: Dict[str, Any]) -> bool:
    a = (_liq(new_t), _vol24(new_t), _mc(new_t))
    b = (_liq(old_t), _vol24(old_t), _mc(old_t))
    return a > b


def _dedupe_best_by_address(tokens: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    merges = 0
    for t in tokens:
        addr = (t.get("address") or t.get("token_address") or "").strip()
        if not addr:
            continue
        prev = best.get(addr)
        if prev is None:
            best[addr] = t
        else:
            merges += 1
            best[addr] = t if _better_row(t, prev) else prev
    if merges:
        logger.info("Eligibility: merged %d duplicate token rows by address.", merges)
    return best


# (6) Build exclude lists (config + defaults + constants if available)
def _exclude_filters(cfg: Dict[str, Any]) -> Dict[str, Set[str]]:
    d = _disc(cfg)
    sym_cfg = {s.strip() for s in (d.get("exclude_symbols") or []) if s and isinstance(s, str)}
    addr_cfg = {s.strip() for s in (d.get("exclude_addresses") or []) if s and isinstance(s, str)}
    # merge with defaults
    sym = set(_DEFAULT_EXCLUDE_SYMBOLS) | sym_cfg
    addr = set(_DEFAULT_EXCLUDE_ADDR) | addr_cfg

    # IMPORTANT: do NOT union a “whitelist” here; we only want to hide those from High-Cap Top-5,
    # which is filtered later in solana_trading_bot.run_discovery_cycle(...)
    # (i.e., whitelist ≠ exclude)

    try:
        # Keep canonical stables/wrappers out of discovery buckets
        addr |= set(ALWAYS_HIDE_IN_CATEGORIES)  # type: ignore
    except Exception:
        logger.debug("Failed to union ALWAYS_HIDE_IN_CATEGORIES", exc_info=True)

    return {"symbols": sym, "addresses": addr}


# Trusted SOL-like mints (canonical WSOL + any other explicitly trusted SOL-like mints)
TRUSTED_SOL_MINTS: Set[str] = {
    "So11111111111111111111111111111111111111112",  # canonical WSOL
    # Add any other explicit trusted SOL-like mint addresses here if you truly trust them
}


# (7) One canonical category per token
def _primary_category(tok: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[str]:
    cat = _canon_bucket_for(tok.get("categories") or [])
    if cat in ("new", "low", "mid", "high"):
        return cat

    rules = _rules(cfg)
    now_s = int(time.time())
    created = _created_s(tok)
    age_min = None if not created else max(0.0, (now_s - created) / 60.0)

    liq = _liq(tok)
    vol = _vol24(tok)
    mc  = _mc(tok)

    r_new = rules["new"]
    if age_min is not None and age_min <= r_new["max_age_min"]:
        if liq >= r_new["liq_min"] and vol >= r_new["vol_min"]:
            return "new"

    r_low, r_mid, r_high = rules["low"], rules["mid"], rules["high"]
    if mc <= r_low["mc_max"]:
        if liq >= r_low["liq_min"] and vol >= r_low["vol_min"]:
            return "low"
        return None
    if mc <= r_mid["mc_max"]:
        if liq >= r_mid["liq_min"] and vol >= r_mid["vol_min"]:
            return "mid"
        return None

    if liq >= r_high["liq_min"] and vol >= r_high["vol_min"]:
        return "high"

    return None


def select_top_five_per_category(
    tokens: List[Dict[str, Any]],
    per_bucket: int = 5,
    blacklist: Optional[Set[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    cfg = _cfg()
    limit = max(1, int(per_bucket or _per_bucket_limit(cfg, default=5)))
    excludes = _exclude_filters(cfg)
    exclude_syms = {s.lower() for s in excludes["symbols"]}
    exclude_names_lc = _DEFAULT_EXCLUDE_NAMES  # lower-case substrings
    exclude_addrs = excludes["addresses"]
    bl = set(blacklist or set())

    # honor env/consts if present
    try:
        bypass_stables = get_env_bool("BYPASS_STABLE_FILTER", False)  # type: ignore
    except Exception:
        bypass_stables = False

    prepped: List[Dict[str, Any]] = []
    dropped_by_exclude = 0

    for t in tokens or []:
        addr = (t.get("address") or t.get("token_address") or "").strip()
        if not addr or addr in bl:
            continue

        # Symbol/name based excludes
        sym = (t.get("symbol") or t.get("baseSymbol") or "").strip()
        name = (t.get("name") or t.get("baseName") or "").strip()
        sym_lc = sym.lower()
        name_lc = name.lower()

        excluded = False
        if not bypass_stables:
            if addr in exclude_addrs:
                excluded = True
            elif sym and sym_lc in exclude_syms:
                excluded = True
            elif any(substr in name_lc for substr in exclude_names_lc):
                excluded = True

        if excluded:
            dropped_by_exclude += 1
            continue

        # normalize numerics
        t = dict(t)
        t.setdefault("liquidity", _liq(t))
        t.setdefault("volume_24h", _vol24(t))
        if "mc" not in t:
            t["mc"] = _mc(t)

        # score (use your util if available)
        if "score" not in t or t["score"] is None:
            try:
                t["score"] = score_token(t, cfg)  # type: ignore
            except Exception:
                logger.debug("score_token failed for %s — applying fallback linear score", addr, exc_info=True)
                w = (cfg.get("weights") or {})
                t["score"] = (
                    _mc(t) * _num(w.get("market_cap"), 0.3)
                    + _liq(t) * _num(w.get("liquidity"), 0.2)
                    + _vol24(t) * _num(w.get("volume"), 0.3)
                    + _num(t.get("price_change_24h") or 0.0) * _num(w.get("price_change"), 0.2)
                )

        if not t.get("categories"):
            t["categories"] = ["unknown_cap"]

        prepped.append(t)

    if dropped_by_exclude:
        logger.info("Eligibility: excluded %d rows by symbol/name/address filters.", dropped_by_exclude)

    # de-dup globally
    best_by_addr = _dedupe_best_by_address(prepped)

    # bucket strictly by primary category (no cross-bucket padding)
    buckets: Dict[str, List[Dict[str, Any]]] = {"high": [], "mid": [], "low": [], "new": []}

    for tok in best_by_addr.values():
        canon = _primary_category(tok, cfg)
        if not canon:
            continue
        if len(buckets[canon]) < limit:
            buckets[canon].append(tok)

    logger.info(
        "Shortlist per-bucket (canonical): high=%d mid=%d low=%d new=%d",
        len(buckets["high"]), len(buckets["mid"]), len(buckets["low"]), len(buckets["new"]),
    )
    return buckets


# Back-compat async wrapper for older imports/call sites
async def shortlist_by_category(

    tokens: List[Dict[str, Any]],
    blacklist: Optional[Set[str]] = None,
    top_n: int = 5,
    new_minutes: int = 180,  # retained for signature compatibility; not used here
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Legacy async wrapper. Accepts either `top_n` or `per_bucket` (kwargs),
    and an optional `blacklist`. Returns the same as `select_top_five_per_category`.
    """
    per_bucket = int(kwargs.get("per_bucket", top_n))
    return select_top_five_per_category(tokens, per_bucket=per_bucket, blacklist=blacklist)


# ==== BEGIN FILTER-REASON SUMMARY PATCH ======================================
# This summarizes *why* candidates are rejected, so discovery "0 coins" is debuggable.
# Call log_filter_summary(...) from your orchestrator right after you build `eligible`.

@dataclass
class FilterStats:
    total_in: int = 0
    kept: int = 0
    reasons: Counter = field(default_factory=Counter)

    @property
    def rejected(self) -> int:
        return max(self.total_in - self.kept, 0)

    def add(self, reason: str) -> None:
        if reason:
            self.reasons[reason] += 1


def _infer_primary_bucket(tok: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """
    Use this module's canonical category logic to pick one of: new|low|mid|high.
    Falls back to thresholds if the token didn't carry categories.
    """
    # Reuse canonical logic defined above in this file
    canon = _primary_category(tok, cfg)  # returns "new"/"low"/"mid"/"high" or None
    return canon or "low"  # harmless fallback


def summarize_filter_outcomes(

    all_candidates: List[Dict[str, Any]],
    kept_candidates: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> FilterStats:
    stats = FilterStats()
    stats.total_in = len(all_candidates)
    kept_set = { (t.get("address") or t.get("token_address") or "").strip() for t in (kept_candidates or []) }
    stats.kept = len(kept_set)

    rules = _rules(cfg)  # uses your discovery thresholds per bucket

    for t in all_candidates or []:
        addr = (t.get("address") or t.get("token_address") or "").strip()
        if not addr:
            stats.add("missing_address")
            continue
        if addr in kept_set:
            continue  # not rejected

        # Determine which bucket’s thresholds to compare against
        bucket = _infer_primary_bucket(t, cfg)
        r = rules.get(bucket, {})
        liq_min = float(r.get("liq_min", 0))
        vol_min = float(r.get("vol_min", 0))

        # Numerics (reuse file helpers)
        liq = _liq(t)
        vol = _vol24(t)
        mc  = _mc(t)
        created_s = _created_s(t)  # 0 if unknown

        # --- market cap sanity (your discovery.min_market_cap, if set) ---
        min_mc = float((cfg.get("discovery") or {}).get("min_market_cap", 0) or 0)
        if mc > 0 and min_mc and mc < min_mc:
            stats.add("marketcap<min")

        # --- liquidity & volume thresholds per bucket ---
        if liq_min and liq < liq_min:
            stats.add(f"liquidity<{bucket}_min")
        if vol_min and vol < vol_min:
            stats.add(f"volume24h<{bucket}_min")

        # --- "new" age window (only if token lands in "new") ---
        if bucket == "new":
            max_age_min = float(((_disc(cfg).get("newly_launched") or {}).get("max_token_age_minutes") or 0) or 0)
            if created_s:
                age_min = max(0.0, (time.time() - created_s) / 60.0)
                if max_age_min and age_min > max_age_min:
                    stats.add("too_old_for_new")

        # --- price change guardrail (matches your discovery.max_price_change) ---
        pc = t.get("price_change_24h") or (t.get("dexscreener") or {}).get("priceChange24h") or 0
        try:
            pc = float(pc)
        except Exception:
            pc = 0.0
        max_pc = float((cfg.get("discovery") or {}).get("max_price_change", 350) or 350)
        if abs(pc) > max_pc:
            # We still record these; with the new logic some may be allowed if depth present.
            stats.add("price_change_24h>max")

        # --- Rugcheck filtering, if enabled at discovery ---
        disc = (cfg.get("discovery") or {})
        require_rc_pass = bool(disc.get("require_rugcheck_pass", False))
        if require_rc_pass:
            labels = {str(x).lower() for x in (t.get("rugcheck_labels") or [])}
            dangerous = bool(t.get("dangerous"))
            if dangerous or labels.intersection({"dangerous", "scam", "honeypot"}):
                stats.add("rugcheck_danger")
            # If RC not present at all
            if not labels and not t.get("rugcheck_score"):
                stats.add("rugcheck_missing")

        # --- cosmetic gaps that often cause later drops ---
        sym = (t.get("symbol") or t.get("baseSymbol") or "").strip()
        if not sym:
            stats.add("missing_symbol")
        src = t.get("source")
        if not src:
            stats.add("missing_source")

    return stats


def log_filter_summary(
    all_candidates: List[Dict[str, Any]],
    kept_candidates: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    logger: logging.Logger,
    *,
    label: str = "pre-shortlist",
) -> None:
    stats = summarize_filter_outcomes(all_candidates, kept_candidates, cfg)
    top_reasons = ", ".join(f"{k}:{v}" for k, v in stats.reasons.most_common(12))
    logger.info(
        "FILTER-SUMMARY [%s] in=%d kept=%d rejected=%d | %s",
        label, stats.total_in, stats.kept, stats.rejected, top_reasons or "no-reasons"
    )