# SOLOTradingBot\solana_trading_bot_bundle\trading_bot\fetching.py

from __future__ import annotations

import os
import json
import time
import asyncio
import logging
import traceback
# keep inspect aliasing usage below for introspection; avoid duplicate bare import
import math
import re
from typing import Optional, Any, Callable, Dict, List, Tuple, Iterable
from pathlib import Path
from datetime import datetime, timezone
import socket  # IPv4/IPv6 connector control

import aiohttp
# aiosqlite intentionally retained if DB cache helpers are used elsewhere
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Solana / solders
from solana.rpc.async_api import AsyncClient

# Prefer solders.Pubkey; fall back to solana.PublicKey or regex-based validator for dev environments.
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

# ---------------------------------------------------------------------
# Ensure BirdeyeRateLimiter & parse helper are available (best-effort imports + minimal fallback)
# ---------------------------------------------------------------------
try:
    # preferred packaged path (when installed as package)
    from solana_trading_bot_bundle.trading_bot.birdeye_client import birdeye_gate, parse_retry_after_seconds  # type: ignore
except Exception:
    try:
        # fallback to local module path when running from repo root
        from birdeye_client import birdeye_gate, parse_retry_after_seconds  # type: ignore
    except Exception:
        # Provide conservative fallbacks so higher-level code can still run in degraded mode.
        birdeye_gate = None  # type: ignore
        def parse_retry_after_seconds(headers):
            """
            Minimal Retry-After parser fallback.

            - Accepts either a numeric value (seconds) or a HTTP-date string.
            - Returns float seconds to wait (>=0.0), or 0.0 if not parseable.
            """
            if not headers:
                return 0.0
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if not ra:
                return 0.0
            try:
                return float(ra)
            except Exception:
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(str(ra))
                    if dt:
                        remaining = dt.timestamp() - time.time()
                        return max(0.0, remaining)
                except Exception:
                    pass
            return 0.0

# ---------------------------------------------------------------------
# Ensure BirdeyeClient class is importable (best-effort imports with None fallback)
# ---------------------------------------------------------------------
try:
    # Preferred packaged path (when installed as a package)
    from solana_trading_bot_bundle.trading_bot.birdeye_client import BirdeyeClient
except Exception:
    try:
        # Fallback for repo layout where trading_bot is a top-level package/module
        from trading_bot.birdeye_client import BirdeyeClient  # type: ignore
    except Exception:
        try:
            # Fallback for running from repo root where module may be importable directly
            from birdeye_client import BirdeyeClient  # type: ignore
        except Exception:
            # If none of the imports succeed, disable client usage safely
            BirdeyeClient = None  # type: ignore

# ---------------------------------------------------------------------
# Defensive numeric helpers
# ---------------------------------------------------------------------
def _fnum(o: dict, key: str, default: float = 0.0) -> float:
    """Defensive numeric extractor (module scope)."""
    try:
        v = (o or {}).get(key, default)
        if v is None:
            return float(default)
        if isinstance(v, str):
            s = v.strip().replace(",", "").replace("$", "")
            if s == "":
                return float(default)
            v = s
        f = float(v)
        if f != f:
            return float(default)
        return f
    except Exception:
        return float(default)


def _parse_float_like(v: Any) -> Optional[float]:
    """Return float or None for many common strings/values."""
    try:
        if v is None or v == "":
            return None
        if isinstance(v, (int, float)):
            f = float(v)
            if f != f:
                return None
            return f
        s = str(v).strip().replace(",", "").replace("$", "")
        if s == "":
            return None
        f = float(s)
        if f != f:
            return None
        return f
    except Exception:
        return None


def _safe_price_from_token_row(row: Dict[str, Any]) -> Optional[float]:
    """Try multiple fields and fallbacks, return None if no usable price."""
    for k in ("price", "priceUsd", "price_usd", "lastPrice"):
        v = _parse_float_like(row.get(k))
        if v is not None and v > 0:
            return v
    for alt in ("pair", "baseToken", "token", "data"):
        sub = row.get(alt)
        if isinstance(sub, dict):
            for k in ("priceUsd", "price"):
                v = _parse_float_like(sub.get(k))
                if v is not None and v > 0:
                    return v
    try:
        addr = (row.get("address") or row.get("token_address") or "").strip()
        fb = FALLBACK_PRICES.get(addr, {}) if isinstance(FALLBACK_PRICES, dict) else {}
        v = _parse_float_like(fb.get("price"))
        if v is not None and v > 0:
            return v
    except Exception:
        pass
    try:
        mc = _parse_float_like(row.get("market_cap") or row.get("mc") or row.get("fdv") or row.get("marketCap"))
        circ = _parse_float_like(row.get("circulating_supply") or row.get("circulatingSupply") or row.get("supply"))
        if mc is not None and circ is not None and circ > 0:
            derived = mc / circ
            if derived > 0:
                return derived
    except Exception:
        pass
    return None

# Added: small defensive float helper used by Birdeye token normalization
def _bd_num(x, default=0.0) -> float:
    try:
        return float(x if x not in (None, "") else default)
    except Exception:
        return float(default)

# Forward declaration so top-level wrappers can refer to refresh_token_cache
# Real implementation appears later in this module and will override this stub.
async def refresh_token_cache(max_items: Optional[int] = None) -> int:
    """
    Refresh the persisted shortlist cache. Compatibility behavior:
      1) If a legacy token_cache.json exists and is parseable, use it.
      2) Otherwise read from the canonical DB (eligible_tokens) via database.list_eligible_tokens.
      3) If DB provided tokens and JSON path is writable, write token_cache.json for legacy consumers.
    Returns number of tokens loaded (0 on failure/none).
    """
    try:
        from solana_trading_bot_bundle.trading_bot.fetching_shortlist_compat import load_shortlist_compat
    except Exception:
        # If helper unavailable, try direct DB call
        try:
            from solana_trading_bot_bundle.trading_bot import database
            tokens = await database.list_eligible_tokens(limit=int(max_items or 500))
            return len(tokens or [])
        except Exception:
            return 0

    try:
        tokens = await load_shortlist_compat(max_rows=int(max_items or 500))
        if not tokens:
            return 0
        return len(tokens)
    except Exception:
        return 0

# --- Ensure indicators / patterns local aliases are wired (defensive) -------
try:
    import numpy as _np  # may already be imported above; safe to reimport
except Exception:
    _np = None

# tiny array-like stubs used as safe fallbacks
class _EmptyArrayStub:
    size = 0
    def __len__(self): return 0
    def __getitem__(self, idx): raise IndexError("empty")

def _empty_float_array():
    return _np.empty(0, dtype=float) if _np is not None else _EmptyArrayStub()

# Return Python lists (not stub objects) for boolean pattern stubs to prevent IndexError
def _empty_bool_array():
    return _np.empty(0, dtype=bool).tolist() if _np is not None else []

# Wire ind_rsi
if not ("ind_rsi" in globals() and callable(globals().get("ind_rsi"))):
    try:
        # prefer local package relative import
        from . import indicators as _ind_mod  # type: ignore
        ind_rsi = getattr(_ind_mod, "rsi", None)
    except Exception:
        ind_rsi = None
    if not callable(ind_rsi):
        try:
            # absolute-package fallback
            import importlib
            _ind_mod = importlib.import_module("solana_trading_bot_bundle.trading_bot.indicators")
            ind_rsi = getattr(_ind_mod, "rsi", None)
        except Exception:
            ind_rsi = None
    if not callable(ind_rsi):
        # final safe stub: returns empty ndarray-like object
        def ind_rsi(*args, **kwargs):
            return _empty_float_array()

# Safely schedule the refresh job from sync contexts (GUI callbacks).
from .utils_exec import safe_create_task  # robust scheduler that handles sync contexts

async def _refresh_token_cache_safely(max_items: Optional[int] = None) -> int:
    """
    Awaitable wrapper around refresh_token_cache that logs any unexpected
    exceptions and returns 0 on failure. Use this when scheduling a
    background job so failures don't go silent.
    """
    try:
        return await refresh_token_cache(max_items=max_items)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("refresh_token_cache crashed unexpectedly")
        return 0

def schedule_refresh_token_cache(max_items: Optional[int] = None, *, name: Optional[str] = None):
    """
    Schedule the refresh_token_cache job from sync or async contexts.
    Returns whatever safe_create_task returns.
    """
    return safe_create_task(_refresh_token_cache_safely(max_items=max_items), name=name or "refresh_token_cache")

# Ensure ind_rsi final bind (safe attempt)
import importlib
_logger = logging.getLogger(__name__)

try:
    _ind_mod = importlib.import_module("solana_trading_bot_bundle.trading_bot.indicators")
    _rsi = getattr(_ind_mod, "rsi", None)
    if callable(_rsi):
        ind_rsi = _rsi
    else:
        _logger.debug("indicators.rsi not callable; keeping existing ind_rsi (or fallback).")
except Exception:
    _logger.debug("Failed to import indicators module; using ind_rsi fallback.", exc_info=True)
    def ind_rsi(*args, **kwargs):
        return _empty_float_array()

# Wire classify_patterns_arrays (keep fallback)
if not ("classify_patterns_arrays" in globals() and callable(globals().get("classify_patterns_arrays"))):
    try:
        from . import patterns as _p_mod  # type: ignore
        classify_patterns_arrays = getattr(_p_mod, "classify_patterns", getattr(_p_mod, "classify_patterns_arrays", None))
    except Exception:
        classify_patterns_arrays = None
    if not callable(classify_patterns_arrays):
        try:
            import importlib
            _p_mod = importlib.import_module("solana_trading_bot_bundle.trading_bot.patterns")
            classify_patterns_arrays = getattr(_p_mod, "classify_patterns", getattr(_p_mod, "classify_patterns_arrays", None))
        except Exception:
            classify_patterns_arrays = None
    if not callable(classify_patterns_arrays):
        # final safe stub: return mapping name -> empty bool-lists (prevent IndexError on [-1])
        def classify_patterns_arrays(ohlcv, names):
            return {n: [] for n in names}

# Project constants / paths
from solana_trading_bot_bundle.common.constants import appdata_dir

# Optional feature flags (env + config aware hard switches)
try:
    from solana_trading_bot_bundle.common.feature_flags import (
        is_enabled_raydium,
        is_enabled_birdeye,
        FORCE_DISABLE_RAYDIUM,
        FORCE_DISABLE_BIRDEYE,
    )
except Exception:
    def _env_on(name: str, default: bool) -> bool:
        v = os.getenv(name, "1" if default else "0").strip().lower()
        return v in ("1", "true", "yes", "on", "y")
    def is_enabled_raydium(*_args, **_kwargs) -> bool:
        return _env_on("RAYDIUM_ENABLE", False)
    def is_enabled_birdeye(*_args, **_kwargs) -> bool:
        return _env_on("BIRDEYE_ENABLE", False)
    FORCE_DISABLE_RAYDIUM = _env_on("FORCE_DISABLE_RAYDIUM", False)
    FORCE_DISABLE_BIRDEYE = _env_on("FORCE_DISABLE_BIRDEYE", False)

# Local utilities / config loader
try:
    from .utils_exec import WHITELISTED_TOKENS, load_config
except ImportError:
    WHITELISTED_TOKENS: Set[str] = set()

    def load_config() -> Dict[str, Any]:
        return {}

# DB cache helpers
from .database import (
    get_cached_token_data,
    cache_token_data,
    get_cached_creation_time,
    cache_creation_time,
)

# ---------------------------------------------------------------------
# DB writer shim: tolerate both cache_token_data(token) and (addr, token)
# ---------------------------------------------------------------------
import inspect as _ins

async def _maybe_await(x):
    import inspect as _inspect
    return (await x) if _inspect.isawaitable(x) else x

try:
    _sig = _ins.signature(cache_token_data)
    _CACHE_TOKEN_DATA_PARAM_COUNT = sum(
        1 for p in _sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    )
except Exception:
    _CACHE_TOKEN_DATA_PARAM_COUNT = 1

async def _write_token_row(addr: str, token: dict) -> None:
    token.setdefault("address", addr)
    if _CACHE_TOKEN_DATA_PARAM_COUNT <= 1:
        await _maybe_await(cache_token_data(token))
    else:
        await _maybe_await(cache_token_data(addr, token))

# ---------------------------------------------------------------------
# Logging (single definition)
# ---------------------------------------------------------------------
logger = logging.getLogger("TradingBot")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------
# Load .env (AppData -> package root -> cwd) and debug redacted API key
# ---------------------------------------------------------------------
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # safe no-op when dotenv isn't installed

def _redact_key(k: Optional[str]) -> str:
    if not k:
        return "<missing>"
    s = str(k)
    if len(s) <= 10:
        return s[:2] + "..." + s[-2:]
    return s[:4] + "..." + s[-4:]

def _load_dotenv_from_candidates() -> Optional[Path]:
    candidates: List[Path] = []
    appdata = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
    if appdata:
        candidates.append(Path(appdata) / "SOLOTradingBot" / ".env")
    try:
        pkg_env = Path(__file__).resolve().parent.parent / ".env"
        candidates.append(pkg_env)
    except Exception:
        pass
    candidates.append(Path.cwd() / ".env")
    for p in candidates:
        try:
            if p.exists():
                if load_dotenv:
                    load_dotenv(dotenv_path=str(p))
                    try:
                        logger.info("Loaded .env from %s", p)
                    except Exception:
                        print(f"Loaded .env from {p}")
                else:
                    try:
                        logger.info("python-dotenv not installed; skipping .env load at %s", p)
                    except Exception:
                        print(f"python-dotenv not installed; skipping .env load at {p}")
                return p
        except Exception:
            continue
    try:
        logger.debug("No .env found in candidates: %s", candidates)
    except Exception:
        pass
    return None

try:
    _loaded = _load_dotenv_from_candidates()
    try:
        _present = bool(os.getenv("BIRDEYE_API_KEY"))
        logger.debug("BIRDEYE_API_KEY present=%s redacted=%s", _present, _redact_key(os.getenv("BIRDEYE_API_KEY")))
    except Exception:
        pass
except Exception:
    try:
        logger.debug("Failed to run .env loader", exc_info=True)
    except Exception:
        pass

# ---------------------------------------------------------------------
# Appdata paths
# ---------------------------------------------------------------------
def _appdir_path(filename: str) -> Path:
    try:
        base = appdata_dir() if callable(appdata_dir) else Path(appdata_dir)
    except Exception:
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / ".local" / "share")) / "SOLOTradingBot"
    base.mkdir(parents=True, exist_ok=True)
    return base / filename

STATUS_FILE = _appdir_path("rugcheck_status.json")
FAILURES_FILE = _appdir_path("rugcheck_failures.json")

# ---------------------------------------------------------------------
# Env defaults (config can override)
# ---------------------------------------------------------------------
def _env_bool(var: str, default: bool) -> bool:
    return os.getenv(var, str(default)).strip().lower() in ("1", "true", "yes", "on", "y")

RUGCHECK_ENABLE = _env_bool("RUGCHECK_ENABLE", True)
RUGCHECK_HARD_FAIL = _env_bool("RUGCHECK_HARD_FAIL", False)
BLOCK_RUGCHECK_DANGEROUS = _env_bool("BLOCK_RUGCHECK_DANGEROUS", True)

RUGCHECK_DISCOVERY_CHECK  = _env_bool("RUGCHECK_DISCOVERY_CHECK", False)
RUGCHECK_DISCOVERY_FILTER = _env_bool("RUGCHECK_DISCOVERY_FILTER", False)

RAYDIUM_ENABLE = _env_bool("RAYDIUM_ENABLE", False)

_MIN_MB_FLOOR = 8
try:
    _ray_bytes_env = os.getenv("RAYDIUM_MAX_BYTES")
    if _ray_bytes_env is not None and _ray_bytes_env.strip():
        RAYDIUM_MAX_BYTES = max(1, int(_ray_bytes_env))
    else:
        _mb_env = os.getenv("RAYDIUM_MAX_DOWNLOAD_MB")
        _mb = int(_mb_env) if (_mb_env is not None and _mb_env.strip()) else 40
        _mb = max(_MIN_MB_FLOOR, _mb)
        RAYDIUM_MAX_BYTES = _mb * 1024 * 1024
except Exception:
    RAYDIUM_MAX_BYTES = 40 * 1024 * 1024

RAYDIUM_MAX_PAIRS = int(os.getenv("RAYDIUM_MAX_PAIRS", "100"))
RAYDIUM_PAGE_SIZE_ENV = int(os.getenv("RAYDIUM_PAGE_SIZE", "50"))
RAYDIUM_MAX_POOLS_ENV = int(os.getenv("RAYDIUM_MAX_POOLS", "1500"))

BIRDEYE_ENABLE_ENV = _env_bool("BIRDEYE_ENABLE", False)
BIRDEYE_MAX_TOKENS_ENV = int(os.getenv("BIRDEYE_MAX_TOKENS", "500"))

DEFAULT_SOLANA_TIMEOUT = int(os.getenv("SOLANA_RPC_TIMEOUT", "15"))
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")

DEXSCREENER_PAGES_ENV = int(os.getenv("DEX_PAGES") or os.getenv("DEXSCREENER_PAGES", "10"))
DEXSCREENER_PER_PAGE_ENV = int(os.getenv("DEX_PER_PAGE", "100"))
DEXSCREENER_MAX_ENV = int(os.getenv("DEX_MAX", os.getenv("DEXSCREENER_MAX", "0")))
DEXSCREENER_QUERY_ENV = os.getenv("DEXSCREENER_QUERY", "solana")
DEXSCREENER_QUERIES_ENV = [q.strip() for q in os.getenv("DEXSCREENER_QUERIES", "").split(",") if q.strip()]

DEX_USER_AGENT_ENV = os.getenv("DEX_USER_AGENT", "").strip()
DEFAULT_BROWSER_UA = (
    DEX_USER_AGENT_ENV
    or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
DEX_FORCE_IPV4_ENV = os.getenv("DEX_FORCE_IPV4", "auto").strip().lower()

def _dex_force_ipv4() -> bool:
    if DEX_FORCE_IPV4_ENV in ("1", "true", "yes", "on", "force"):
        return True
    if DEX_FORCE_IPV4_ENV in ("0", "false", "no", "off"):
        return False
    return os.name == "nt"

try:
    _FORCE_IPV4 = _dex_force_ipv4()
except Exception:
    _FORCE_IPV4 = False

logger.debug("DEX/BIRDEYE IPv4 preference: _FORCE_IPV4=%s (DEX_FORCE_IPV4_ENV=%r)", _FORCE_IPV4, DEX_FORCE_IPV4_ENV)

async def _make_dex_session() -> Tuple[aiohttp.ClientSession, Dict[str, str]]:
    family = socket.AF_INET if _dex_force_ipv4() else 0
    connector = aiohttp.TCPConnector(
        family=family, limit=30, ttl_dns_cache=300, enable_cleanup_closed=True
    )
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)
    sess = aiohttp.ClientSession(connector=connector, timeout=timeout)
    headers = {"Accept": "application/json", "User-Agent": DEFAULT_BROWSER_UA}
    return sess, headers

# --- Helper: extract first base58-like substring (Solana mint) ----------
def _extract_base58_address(s: Any) -> Optional[str]:
    """
    Return first base58-like substring that looks like a Solana Pubkey
    (32..44 chars, base58 alphabet). Returns None if none found.
    """
    try:
        if not s:
            return None
        ss = str(s).strip()
        # match Base58 subset used by Solana (no 0,O,I,l)
        m = re.search(r"([1-9A-HJ-NP-Za-km-z]{32,44})", ss)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------
# Birdeye client integration helpers (lazy sync client called via threads)
# ---------------------------------------------------------------------
_birdeye_client = None
def _get_birdeye_client():
    """
    Lazily create and return a BirdeyeClient (synchronous/requests-based).
    We will call its methods inside asyncio.to_thread(...) to avoid blocking the event loop.
    """
    global _birdeye_client
    if _birdeye_client is not None:
        return _birdeye_client
    if BirdeyeClient is None:
        logger.debug("BirdeyeClient class not importable; client-based probe disabled.")
        return None
    api_key = os.getenv("BIRDEYE_API_KEY", "") or ""
    if not api_key:
        logger.debug("BIRDEYE_API_KEY not set; client-based probe disabled.")
        return None
    base = globals().get("BIRDEYE_BASE_URL") or "https://public-api.birdeye.so"
    # conservative default RPS (override with BIRDEYE_RPS env during testing)
    try:
        rps = max(1, int(os.getenv("BIRDEYE_RPS", "3")))
    except Exception:
        rps = 3
    try:
        _birdeye_client = BirdeyeClient(api_key=api_key, base_url=base, rps_limit=rps)
        logger.info("Instantiated BirdeyeClient (rps=%s base=%s)", rps, base)
        return _birdeye_client
    except Exception as e:
        logger.warning("Failed to instantiate BirdeyeClient: %s", e)
        return None

# ---------------------------------------------------------------------
# Birdeye cooldown / host-block state (centralized) with async lock ----------
from typing import cast

_BIRDEYE_HOST_BLOCK_UNTIL: Dict[str, float] = {}
_BIRDEYE_COOLDOWN_UNTIL: float = 0.0
_BIRDEYE_LAST_BACKOFF: float = 0.0
_BIRDEYE_DOWN_BACKOFF_DEFAULT: float = 30.0

# module-level cache for Birdeye token list
BIRDEYE_CACHE: Dict[str, Any] = {"until": 0.0, "items": []}

# lazy asyncio.Lock for state updates (create on first use inside an event loop)
_BIRDEYE_STATE_LOCK: Optional[asyncio.Lock] = None
def _ensure_birdeye_lock():
    global _BIRDEYE_STATE_LOCK
    if _BIRDEYE_STATE_LOCK is None:
        try:
            _BIRDEYE_STATE_LOCK = asyncio.Lock()
        except Exception:
            # fallback dummy async context manager (no-op)
            class _DummyAsyncLock:
                async def __aenter__(self): return None
                async def __aexit__(self, exc_type, exc, tb): return False
            _BIRDEYE_STATE_LOCK = _DummyAsyncLock()
    return _BIRDEYE_STATE_LOCK

def _birdeye_is_blocked(host: str) -> bool:
    return time.time() < _BIRDEYE_HOST_BLOCK_UNTIL.get(host, 0.0)

async def _birdeye_block_host(host: str, seconds: float) -> None:
    lock = _ensure_birdeye_lock()
    async with lock:
        _BIRDEYE_HOST_BLOCK_UNTIL[host] = max(_BIRDEYE_HOST_BLOCK_UNTIL.get(host, 0.0), time.time() + seconds)

def _birdeye_is_cooled_down() -> bool:
    return time.time() >= _BIRDEYE_COOLDOWN_UNTIL

async def _birdeye_cooldown(seconds: float) -> None:
    global _BIRDEYE_COOLDOWN_UNTIL
    lock = _ensure_birdeye_lock()
    async with lock:
        _BIRDEYE_COOLDOWN_UNTIL = max(_BIRDEYE_COOLDOWN_UNTIL, time.time() + seconds)

# ---- Quality thresholds (optional; leave 0 to disable) -----------------------
MIN_LIQUIDITY_USD   = float(os.getenv("MIN_LIQUIDITY_USD",   "0"))
MIN_VOLUME24_USD    = float(os.getenv("MIN_VOLUME24_USD",    "0"))
MIN_MARKETCAP_USD   = float(os.getenv("MIN_MARKETCAP_USD",   "0"))

def _num(v, d=0.0) -> float:
    try:
        return float(v if v not in (None, "") else d)
    except Exception:
        return float(d)

def _mc(obj: Dict[str, Any]) -> float:
    return _num(
        obj.get("market_cap")
        or obj.get("mc")
        or obj.get("fdv")
        or obj.get("fdvUsd")
        or obj.get("marketCap")
        or 0.0,
        0.0,
    )

def _better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    a_key = (_num(a.get("liquidity")), _num(a.get("volume_24h")), _mc(a))
    b_key = (_num(b.get("liquidity")), _num(b.get("volume_24h")), _mc(b))
    return a_key > b_key

def _passes_minimums(tok: Dict[str, Any]) -> bool:
    if MIN_LIQUIDITY_USD   and _num(tok.get("liquidity"))  < MIN_LIQUIDITY_USD:   return False
    if MIN_VOLUME24_USD    and _num(tok.get("volume_24h")) < MIN_VOLUME24_USD:    return False
    if MIN_MARKETCAP_USD   and _mc(tok)                    < MIN_MARKETCAP_USD:   return False
    return True

from .utils_exec import _best_first

# ---------------------------------------------------------------------
# Cooperative shutdown
# ---------------------------------------------------------------------
from threading import Event as _ThreadEvent
_SHUTDOWN = _ThreadEvent()

def signal_shutdown() -> None:
    _SHUTDOWN.set()

def clear_shutdown_signal() -> None:
    _SHUTDOWN.clear()

def _should_stop() -> bool:
    if _SHUTDOWN.is_set():
        return True
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    return loop.is_closed()

# ---------------------------------------------------------------------
# Helpers to read config with sane fallbacks
# ---------------------------------------------------------------------
def _cfg() -> Dict[str, Any]:
    try:
        return load_config() or {}
    except Exception:
        return {}

def _disc(key: str, default: Any) -> Any:
    c = _cfg()
    return (c.get("discovery") or {}).get(key, default)

def _dex_queries() -> List[str]:
    q = _disc("dexscreener_queries", None)
    if isinstance(q, list) and q:
        return [str(x).strip() for x in q if str(x).strip()]
    if DEXSCREENER_QUERIES_ENV:
        return DEXSCREENER_QUERIES_ENV
    return [DEXSCREENER_QUERY_ENV]

def _dex_pages_perpage() -> Tuple[int, int]:
    pages = int(_disc("dexscreener_pages", DEXSCREENER_PAGES_ENV))
    per_page = int(_disc("dexscreener_per_page", DEXSCREENER_PER_PAGE_ENV))
    return max(1, pages), max(1, per_page)

def _dex_post_cap() -> int:
    cap = 0
    try:
        cap = int(_disc("dexscreener_post_cap", 0))
    except Exception:
        cap = 0
    if cap <= 0:
        try:
            cap = int(DEXSCREENER_MAX_ENV or 0)
        except Exception:
            cap = 0
    return max(0, cap)

def _ray_limits(max_pairs_hint: Optional[int] = None) -> Tuple[int, int]:
    cfg_max_pairs = int(_disc("raydium_max_pairs", RAYDIUM_MAX_PAIRS))
    parse_cap = int(max_pairs_hint or cfg_max_pairs)
    parse_cap = min(parse_cap, int(_disc("raydium_max_pools", RAYDIUM_MAX_POOLS_ENV)))
    page_size = int(_disc("raydium_page_size", RAYDIUM_PAGE_SIZE_ENV))
    page_size = max(50, min(page_size, 250))
    return max(1, parse_cap), page_size

def _bird_max() -> int:
    try:
        v = int(_disc("birdeye_max_tokens", BIRDEYE_MAX_TOKENS_ENV))
        return max(0, v)
    except Exception:
        return max(0, BIRDEYE_MAX_TOKENS_ENV)

def _rc_discovery_switches() -> Tuple[bool, bool]:
    check = bool(_disc("rugcheck_in_discovery", RUGCHECK_DISCOVERY_CHECK))
    filt = bool(_disc("require_rugcheck_pass", RUGCHECK_DISCOVERY_FILTER))
    return check, filt

# ---------------------------------------------------------------------
# Fallback prices (DISABLED by default)
# ---------------------------------------------------------------------
FALLBACK_PRICES: Dict[str, Dict[str, float]] = {}
if os.getenv("FALLBACK_ENABLE", "0").lower() not in ("1", "true", "yes"):
    FALLBACK_PRICES.clear()

logger.warning(f"USING FETCHING FROM: {__file__}")
logger.warning("FALLBACK enabled? %s (count=%d)", bool(FALLBACK_PRICES), len(FALLBACK_PRICES))

# ---------------------------------------------------------------------
# In-memory failures & status
# ---------------------------------------------------------------------
RUGCHECK_FAILURES: List[Dict[str, str]] = []

def _write_status(enabled: bool, available: bool, message: str) -> None:
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with STATUS_FILE.open("w", encoding="utf-8") as f:
            json.dump(
                {"enabled": enabled, "available": available, "message": message, "timestamp": int(time.time())},
                f, indent=2,
            )
    except Exception as e:
        logger.warning("Failed to write Rugcheck status: %s", e)

def _save_failures() -> None:
    try:
        FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with FAILURES_FILE.open("w", encoding="utf-8") as f:
            json.dump(RUGCHECK_FAILURES, f, indent=2)
    except Exception as e:
        logger.error("Failed to save Rugcheck failures: %s", e)

# NEW: seed the GUI banner immediately on startup (no network calls)
def ensure_rugcheck_status_file() -> None:
    try:
        if "RUGCHECK_ENABLE" in globals():
            enabled = bool(globals()["RUGCHECK_ENABLE"])
        else:
            enabled = str(os.getenv("RUGCHECK_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on")
        try:
            hdrs = get_rugcheck_headers() if "get_rugcheck_headers" in globals() else {}
        except Exception:
            hdrs = {}
        available = bool(hdrs.get("Authorization"))
        # reflect hard-fail policy in banner
        policy = "hard_fail" if RUGCHECK_HARD_FAIL else "allow_if_api_down"
        msg = f"JWT configured (policy={policy})" if available else f"JWT missing (policy={policy})"
        _write_status(enabled, available, msg)
    except Exception as e:
        _write_status(False, False, f"status unavailable: {e}")

_MAX_FAILURES = 500

def _append_failure(addr: str, reason: str) -> None:
    RUGCHECK_FAILURES.append({"address": addr, "reason": reason})
    if len(RUGCHECK_FAILURES) > _MAX_FAILURES:
        del RUGCHECK_FAILURES[: len(RUGCHECK_FAILURES) - _MAX_FAILURES]
    _save_failures()

def log_error_with_stacktrace(message: str, error: Exception) -> None:
    logger.error("%s: %s\n%s", message, traceback.format_exc())

# ---------------------------------------------------------------------
# Sanity tweak #1: normalize config with safe defaults (discovery/bot)
# ---------------------------------------------------------------------
def _normalized_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = raw or {}
    disc = dict(raw.get("discovery") or {})
    # default buckets
    def _b(defaults):
        b = dict(defaults)
        b.update(dict(disc.get(defaults["name"], {})))
        b.pop("name", None)
        return b

    # sensible defaults that won't stall discovery
    low_cap_defaults = {
        "name": "low_cap",
        "max_market_cap": 750_000,
        "liquidity_threshold": 10_000,
        "volume_threshold": 10_000,
        "max_rugcheck_score": 5000,
    }
    mid_cap_defaults = {
        "name": "mid_cap",
        "max_market_cap": 5_000_000,
        "liquidity_threshold": 50_000,
        "volume_threshold": 40_000,
        "max_rugcheck_score": 7000,
    }
    large_cap_defaults = {
        "name": "large_cap",
        "liquidity_threshold": 100_000,
        "volume_threshold": 100_000,
        "max_rugcheck_score": 9000,
    }
    new_defaults = {
        "name": "newly_launched",
        "max_token_age_minutes": 180,
        "liquidity_threshold": 5_000,
        "volume_threshold": 5_000,
        "max_rugcheck_score": 2000,
    }

    disc_out = {
        "low_cap": _b(low_cap_defaults),
        "mid_cap": _b(mid_cap_defaults),
        "large_cap": _b(large_cap_defaults),
        "newly_launched": _b(new_defaults),
        "max_price_change": float(disc.get("max_price_change", 85)),  # absolute %
    }

    bot = dict(raw.get("bot") or {})
    bot.setdefault("cycle_interval", 30)
    bot.setdefault("cooldown_seconds", 3)
    bot.setdefault("dry_run", False)

    out = dict(raw)
    out["discovery"] = disc_out
    out["bot"] = bot
    return out

# -----------------------------------------------------------------------------
# Sanity tweak #2: clamp discovery/fetch limits to safe ranges
# -----------------------------------------------------------------------------
def _clamp_discovery_limits(cfg: Dict[str, Any]) -> None:
    # Dexscreener
    disc = cfg.setdefault("discovery", {})
    dex = disc.setdefault("dexscreener", {})
    dex.setdefault("pages", 2)
    dex.setdefault("per_page", 100)
    dex.setdefault("max", 0)
    dex["pages"] = max(1, min(int(dex.get("pages", 2)), 3))
    dex["per_page"] = max(50, min(int(dex.get("per_page", 100)), 150))
    dex["max"] = max(0, min(int(dex.get("max", 0)), 1000))

    # Raydium
    ray = disc.setdefault("raydium", {})
    ray.setdefault("max_pairs", 100)
    ray.setdefault("max_download_mb", 40)  # small to avoid stalls on Windows
    ray["max_pairs"] = max(25, min(int(ray.get("max_pairs", 100)), 250))
    ray["max_download_mb"] = max(16, min(int(ray.get("max_download_mb", 40)), 85))

    # Birdeye
    be = disc.setdefault("birdeye", {})
    be.setdefault("max_tokens", 250)
    be["max_tokens"] = max(50, min(int(be.get("max_tokens", 250)), 750))
    

# ---------------------------------------------------------------------
# Rugcheck header helpers + shared RugcheckClient integration
# ---------------------------------------------------------------------
try:
    try:
        from .rugcheck_auth import ensure_valid_rugcheck_headers as _rc_hdrs
    except Exception:
        _rc_hdrs = None
    from .rugcheck_auth import get_rugcheck_headers
except Exception:
    try:
        try:
            from solana_trading_bot_bundle.trading_bot.rugcheck_auth import ensure_valid_rugcheck_headers as _rc_hdrs
        except Exception:
            _rc_hdrs = None
        from solana_trading_bot_bundle.trading_bot.rugcheck_auth import get_rugcheck_headers
    except Exception:
        _rc_hdrs = None
        def get_rugcheck_headers() -> Dict[str, str]:
            tok = (os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT") or os.getenv("RUGCHECK_API_TOKEN") or "")
            return {"Authorization": f"Bearer {tok}"} if tok else {}

# Prefer a centralized RugcheckClient if available (handles rate/backoff/JWT)
try:
    from .rugcheck_client import make_rugcheck_client_from_env  # type: ignore
except Exception:
    try:
        from solana_trading_bot_bundle.trading_bot.rugcheck_client import make_rugcheck_client_from_env  # type: ignore
    except Exception:
        make_rugcheck_client_from_env = None  # type: ignore

_rug_client = None  # module-level singleton

async def _get_rug_client():
    global _rug_client
    if _rug_client is not None:
        return _rug_client
    if make_rugcheck_client_from_env is None:
        return None
    _rug_client = make_rugcheck_client_from_env()
    try:
        await _rug_client.start()
    except Exception:
        return _rug_client
    return _rug_client

# ---------------------------------------------------------------------
# Streaming JSON helper (defensive, size-capped)
# ---------------------------------------------------------------------
async def _safe_json_stream(
    resp: aiohttp.ClientResponse,
    max_bytes: Optional[int] = None,
    chunk_size: int = 262_144,  # 256 KiB chunks keep memory flatter
) -> Any:
    """
    Stream-response -> JSON with a hard byte cap.

    - Aborts early if Content-Length exceeds cap.
    - Streams in chunks; raises MemoryError if cap is exceeded mid-stream.
    - Returns {} on empty/204 bodies.
    - Tries a second decode pass with explicit UTF-8 if the first fails.
    """
    # --- resolve byte cap safely (no NameError if constant is missing)
    try:
        default_cap = int(RAYDIUM_MAX_BYTES)  # type: ignore[name-defined]
    except Exception:
        default_cap = 40 * 1024 * 1024  # 40 MiB final fallback

    limit = int(max_bytes) if max_bytes is not None else default_cap
    if limit < 1:
        limit = default_cap

    # --- hard fail on clearly-oversize content-length
    cl = resp.content_length
    if cl is not None and cl > limit:
        # Make sure to release the connection pool slot ASAP
        try:
            await resp.release()
        finally:
            pass
        raise MemoryError(f"Response too large: {cl} bytes > {limit}")

    # --- stream the body under the cap
    buf = bytearray()
    try:
        async for chunk in resp.content.iter_chunked(chunk_size):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > limit:
                # Stop reading more data; free socket promptly
                try:
                    await resp.release()
                finally:
                    pass
                raise MemoryError(f"Response exceeded {limit} bytes")
    except asyncio.CancelledError:
        # Propagate cancellations cleanly (task shutdown, etc.)
        raise
    except Exception:
        # Ensure the connection is not left hanging on unexpected errors
        try:
            await resp.release()
        finally:
            pass
        raise

    # --- empty body handling
    if not buf or resp.status == 204:
        return {}

    # --- decode JSON (two attempts)
    try:
        return json.loads(buf)
    except json.JSONDecodeError:
        # Retry with explicit UTF-8 string decode (tolerant replacement)
        try:
            text = buf.decode("utf-8", errors="replace")
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON ({e}) from {len(buf)} bytes") from e

# ---------------------------------------------------------------------
# Rugcheck validation (boolean for trading-time checks)
# ---------------------------------------------------------------------
# Patch: replace the existing validate_rugcheck implementation with a client-first, 404-caching-safe version.

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=30))
async def validate_rugcheck(token_address: str, session: aiohttp.ClientSession) -> bool:
    """
    Prefer shared RugcheckClient for requests (handles JWT/API-key/backoff/404-caching).
    Fall back to legacy per-session header fetching if client is unavailable.

    Differences vs previous implementation:
      - Uses rugcheck_client.get_report() which centralizes 404 handling and JWT refresh.
      - Treats missing data (None) as 'API down' behavior controlled by RUGCHECK_HARD_FAIL / allow_if_api_down.
      - Avoids spamming ERROR logs on /tokens/scan 404s (client caches them).
    """
    if not RUGCHECK_ENABLE:
        _write_status(False, False, "Rugcheck disabled by config; tokens allowed")
        return True
    if token_address in WHITELISTED_TOKENS:
        _write_status(True, True, "Rugcheck active (whitelist bypass)")
        return True

    # Try to get a shared RugcheckClient (may be None)
    try:
        client = await _get_rug_client()
    except Exception:
        client = None

    # Legacy header acquisition (only used for the fallback path)
    headers: Dict[str, str] = {}
    if client is None:
        try:
            headers = await _rc_hdrs(session, force_refresh=False) if _rc_hdrs else get_rugcheck_headers()
        except Exception:
            headers = {}

        if not headers.get("Authorization"):
            msg = "Rugcheck unavailable: missing/invalid JWT"
            logger.warning("%s for %s", msg, token_address)
            _append_failure(token_address, msg)
            _write_status(True, False, msg)
            return not RUGCHECK_HARD_FAIL

    try:
        data = None
        status = None

        # --- Preferred path: RugcheckClient helpers (get_report) ---
        if client is not None:
            try:
                # client.get_report returns parsed JSON dict (or None)
                data = await client.get_report(token_address)
                status = 200 if data is not None else None
            except Exception as e:
                # client raises on persistent non-200 (ClientResponseError) or rethrows network errors.
                status = getattr(e, "status", None) or None
                last_err = f"{type(e).__name__}: {e}"
                logger.debug("Rugcheck client.get_report error for %s: %s", token_address, last_err, exc_info=True)

                # If unauthorized, attempt a forced refresh via module helper then retry once.
                if status in (401, 403) and _rc_hdrs:
                    try:
                        maybe = _rc_hdrs(session, force_refresh=True)  # type: ignore
                        if asyncio.iscoroutine(maybe):
                            await maybe
                    except Exception:
                        logger.debug("ensure_valid_rugcheck_headers refresh failed (client path)", exc_info=True)
                    try:
                        data = await client.get_report(token_address)
                        status = 200 if data is not None else status
                    except Exception as e2:
                        status = getattr(e2, "status", None) or status
                        logger.debug("Rugcheck client.get_report retry failed for %s: %s", token_address, e2, exc_info=True)

        # --- Fallback path: legacy session-based GET to /report (preserve previous behavior) ---
        if data is None:
            # Compose endpoints and headers for legacy path
            api_base = str((load_config() or {}).get("rugcheck", {}).get("api_base", "https://api.rugcheck.xyz/v1")).rstrip("/")
            report_url = f"{api_base}/tokens/{token_address}/report"

            # try to ensure headers include Authorization / X-API-KEY
            try:
                # use module-level _rc_hdrs (ensure_valid_rugcheck_headers) if available
                maybe = _rc_hdrs(session, force_refresh=False) if _rc_hdrs else None
                if asyncio.iscoroutine(maybe):
                    maybe = await maybe
                if isinstance(maybe, dict):
                    headers.update(maybe)
            except Exception:
                pass
            try:
                # use module-level get_rugcheck_headers (fallback) if available
                maybe2 = get_rugcheck_headers() if "get_rugcheck_headers" in globals() else None
                if isinstance(maybe2, dict):
                    headers.update(maybe2)
            except Exception:
                pass

            # If still no Authorization header, try to obtain it via get_rugcheck_headers()
            try:
                if not any(k.lower() == "authorization" for k in headers.keys()):
                    maybe_hdrs = get_rugcheck_headers() if "get_rugcheck_headers" in globals() else {}
                    if isinstance(maybe_hdrs, dict):
                        auth = {k: v for k, v in maybe_hdrs.items() if k.lower() == "authorization"}
                        if auth:
                            headers.update(auth)
            except Exception:
                pass

            # Ensure X-API-KEY present if configured and not already set
            try:
                cfg = load_config() or {}
                api_key = (cfg.get("rugcheck", {}) or {}).get("api_key") or os.getenv("RUGCHECK_API_KEY") or ""
                if api_key and not any(k.lower() == "x-api-key" for k in headers.keys()):
                    headers["X-API-KEY"] = str(api_key)
            except Exception:
                pass

            timeout = aiohttp.ClientTimeout(total=12, sock_connect=min(7, 12), sock_read=min(10, 12))
            try:
                async with session.get(report_url, headers=headers, timeout=timeout) as r:
                    status = r.status
                    if r.status == 200:
                        try:
                            data = await r.json(content_type=None)
                        except Exception:
                            txt = await r.text()
                            logger.debug("Rugcheck legacy report: non-JSON body for %s: %s", token_address, (txt or "")[:200])
                            data = {"raw": txt}
                    else:
                        text = await r.text()
                        logger.debug("Rugcheck legacy report HTTP %s for %s: %s", r.status, token_address, (text or "")[:300])
            except Exception as e:
                logger.debug("Rugcheck legacy request failed for %s: %s", token_address, e, exc_info=True)
                data = None
                status = None

        # --- Interpret result ---
        if data is None:
            msg = f"Rugcheck HTTP {status or 'no-response'}"
            logger.warning("%s for %s", msg, token_address)
            _append_failure(token_address, msg)
            _write_status(True, False, msg)
            return not RUGCHECK_HARD_FAIL

        # Extract a risk-level/label in a robust way (support multiple shapes)
        risk_level = None
        try:
            if isinstance(data, dict):
                # prefer explicit top-level fields
                risk_level = _best_first(
                    data.get("risk_level"),
                    data.get("label"),
                    (data.get("risk") or {}).get("label"),
                    (data.get("result") or {}).get("risk_level"),
                    (data.get("analysis") or {}).get("risk_level"),
                )
            else:
                risk_level = None
        except Exception:
            risk_level = None

        rl_norm = (str(risk_level or "unknown")).lower()

        if BLOCK_RUGCHECK_DANGEROUS and rl_norm in ("high", "medium"):
            msg = f"High/Medium risk ({risk_level})"
            logger.warning("Rugcheck blocked %s: %s", token_address, msg)
            _append_failure(token_address, msg)
            _write_status(True, True, "Rugcheck active")
            return False

        _write_status(True, True, "Rugcheck active")
        return True

    except (aiohttp.ClientError, ValueError) as e:
        msg = f"Rugcheck error: {e}"
        logger.warning("%s for %s", msg, token_address)
        _append_failure(token_address, str(e))
        _write_status(True, False, msg)
        return not RUGCHECK_HARD_FAIL

# Lightweight annotator used during discovery (config-aware)
async def _annotate_rugcheck_fields(token_address: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
    safety = "unknown"
    dangerous = False
    check, filt = _rc_discovery_switches()
    if RUGCHECK_ENABLE and check:
        try:
            ok = await validate_rugcheck(token_address, session)
            safety = "ok" if ok else "dangerous"
            dangerous = not ok
            if (not ok) and filt and token_address not in WHITELISTED_TOKENS:
                return {"drop": True}
        except Exception:
            safety = "unknown"
            dangerous = False
    return {"safety": safety, "dangerous": dangerous}

# ---------------------------------------------------------------------
# Dexscreener (search)
#  - Classic paged version (stable, predictable, fewer duplicates)
#  - Auto-fallback to sharded search with alt-queries if classic is sparse
#  Both paths DEDUPE BY BASE TOKEN ADDRESS before returning.
#  IMPORTANT: Solana mints are case-sensitive — we DO NOT lowercase them.
# ---------------------------------------------------------------------
from string import ascii_lowercase, digits

#   DEX_USE_SHARDS=false  -> we still try classic first, and only shard if classic is sparse
#   DEX_USE_SHARDS=true   -> (kept for back-compat; wrapper no longer depends on it strictly)
DEX_USE_SHARDS_ENV = os.getenv("DEX_USE_SHARDS", "false").lower() in ("1", "true", "yes", "on")


def _prefer(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return the better row between a and b using (liquidity, vol24h, fdv/mcap).
    Tie-breaker: prefer the row that has a non-null, positive price when liquidity
    and volume are equal (reduces chance of overwriting a priced row with a
    zero/unknown-priced row).
    """
    def f(x, k, d=0.0):
        try:
            return float((x or {}).get(k, d) or d)
        except Exception:
            return float(d)

    liq_a, liq_b = f(a, "liquidity"), f(b, "liquidity")
    if liq_a != liq_b:
        return a if liq_a > liq_b else b

    vol_a, vol_b = f(a, "volume_24h"), f(b, "volume_24h")
    if vol_a != vol_b:
        return a if vol_a > vol_b else b

    # Prefer rows that have a real price (non-null, >0) when liquidity & volume tie.
    def _has_price(x: Dict[str, Any]) -> bool:
        try:
            # check common price keys; treat only positive parsed floats as a "has price"
            for key in ("price", "priceUsd", "price_usd", "lastPrice"):
                v = (x or {}).get(key)
                if v is not None:
                    pv = _parse_float_like(v)
                    if pv is not None and pv > 0:
                        return True
        except Exception:
            pass
        return False

    hp_a, hp_b = _has_price(a), _has_price(b)
    if hp_a != hp_b:
        return a if hp_a else b

    # Final fallback: compare market-cap / fdv (existing behaviour)
    def mc_val(x: Dict[str, Any]) -> float:
        try:
            return float(
                (x.get("market_cap")
                 or x.get("mc")
                 or x.get("fdv")
                 or x.get("fdvUsd")
                 or x.get("marketCap")
                 or 0.0)
            )
        except Exception:
            return 0.0

    return a if mc_val(a) >= mc_val(b) else b


def _dedupe_by_base_address(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse duplicates by base token address, keeping best row.
    Preserve original Solana address casing (case-sensitive base58).
    Also keep both 'address' and 'token_address' populated for downstream code.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        addr = (r.get("address") or r.get("token_address") or "").strip()  # NO .lower()
        if not addr:
            continue
        prev = best.get(addr)
        best[addr] = r if prev is None else _prefer(prev, r)

    out: List[Dict[str, Any]] = []
    for r in best.values():
        a = (r.get("address") or r.get("token_address") or "").strip()
        r["address"] = a
        r["token_address"] = a
        out.append(r)
    return out


def _dex_query_shards(pages_to_fetch: int) -> List[str]:
    """
    Generate shard suffixes (" a", " b", ...) to broaden Dexscreener search.
    Front-load vowels + 't' then the rest + digits to bias early, common results.
    """
    base = list("aeioubt") + [c for c in ascii_lowercase if c not in "aeioubt"] + list(digits)
    n = max(1, int(pages_to_fetch))
    return [f" {s}" for s in base[:n]]


# ---------- Windows-friendly browser-like headers ----------
def _default_dex_headers(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    ua = os.getenv("DEX_USER_AGENT", "").strip() or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    )
    hdrs = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://dexscreener.com",
        "Referer": "https://dexscreener.com/",
        "Connection": "keep-alive",
        "User-Agent": ua,
    }
    if overrides:
        hdrs.update(overrides)
    return hdrs


# [snip rest of file unchanged for brevity in this view]
# (The rest of the file is unchanged except for the two added `global` declarations
#  explained above. The full body continues as originally provided by the user.)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=6, max=120),
    retry=retry_if_exception_type((aiohttp.ClientError, ValueError, asyncio.TimeoutError)),
)
async def fetch_dexscreener_search_classic(
    session: aiohttp.ClientSession,
    query: str = DEXSCREENER_QUERY_ENV,
    pages: Optional[int] = None,
    per_page: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Classic Dexscreener search using sequential ?page= pagination.
    Dedupes by base address. Uses browser-like headers and tries both URL variants.
    """
    cfg_pages, cfg_per_page = _dex_pages_perpage()
    total_pages = max(1, int(pages) if pages is not None else cfg_pages)
    per = max(1, min(int(per_page) if per_page is not None else cfg_per_page, 200))

    # try without and with trailing slash — observed to behave differently behind some CDNs
    bases = [
        "https://api.dexscreener.com/latest/dex/search",
        "https://api.dexscreener.com/latest/dex/search/",
    ]
    hdrs = _default_dex_headers(headers)
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)

    rows: List[Dict[str, Any]] = []
    now_ms = int(time.time() * 1000)
    sol_launch_ms = 1581465600000  # 2020-02-12T00:00:00Z

    for page in range(1, total_pages + 1):
        if _should_stop():
            break
        got_page = False

        for base in bases:
            url = f"{base}?q={query}&page={page}"
            try:
                async with session.get(url, headers=hdrs, timeout=timeout) as resp:
                    if resp.status == 429:
                        ra = resp.headers.get("Retry-After") or resp.headers.get("X-RateLimit-Reset")
                        sleep_s = float(ra) - time.time() if ra and str(ra).isdigit() else 15.0
                        await asyncio.sleep(max(5.0, sleep_s))
                        continue
                    if resp.status != 200:
                        body = (await resp.text())[:300]
                        logger.warning("Dexscreener page=%d HTTP %s: %s", page, resp.status, body)
                        continue

                    data = await resp.json(content_type=None)
                    pairs = (data.get("pairs") or [])[:per]
                    for pair in pairs:
                        if pair.get("chainId") != "solana":
                            continue
                        token_address = (pair.get("baseToken") or {}).get("address")
                        if not token_address:
                            continue
                        try:
                            Pubkey.from_string(token_address)
                        except Exception:
                            continue
                        token_address = token_address.strip()

                        created = pair.get("pairCreatedAt") or 0
                        if not (sol_launch_ms <= (created or 0) <= now_ms):
                            created = 0

                        # --- normalize priceChange and market cap for classic path ---
                        _pc = pair.get("priceChange") or {}
                        _pc1h = float((_pc.get("h1") or _pc.get("1h") or 0) or 0)
                        _pc6h = float((_pc.get("h6") or _pc.get("6h") or 0) or 0)
                        _pc24h = float((_pc.get("h24") or _pc.get("24h") or 0) or 0)

                        # created (pairCreatedAt might be ms or seconds; we keep earlier logic of zeroing out invalid ranges)
                        created = pair.get("pairCreatedAt") or pair.get("createdAt") or 0
                        try:
                            created_val = float(created or 0)
                        except Exception:
                            created_val = 0
                        # (classic path used created==0 as "unknown")
                        if not (1581465600000 <= (created_val if created_val > 10_000_000_000 else created_val * 1000) <= int(time.time() * 1000)):
                            created = 0

                        # defensive market cap / volume / liquidity parsing
                        _mcap = _parse_float_like(
                            pair.get("fdv") or pair.get("fdvUsd") or pair.get("marketCap") or pair.get("marketCapUsd") or 0
                        ) or 0.0

                        # volume and liquidity may be nested objects; parse defensively
                        volume_24h = _parse_float_like(
                            (pair.get("volume") or {}).get("h24") or (pair.get("volume") or {}).get("24h") or 0
                        ) or 0.0
                        liquidity = _parse_float_like(
                            (pair.get("liquidity") or {}).get("usd") or pair.get("liquidity") or 0
                        ) or 0.0

                        # Price: do not coerce to 0; prefer None if missing.
                        price_val = _parse_float_like(pair.get("priceUsd") or pair.get("price") or None)

                        rows.append({
                            "address": token_address,
                            "token_address": token_address,
                            "symbol": (pair.get("baseToken") or {}).get("symbol", "UNKNOWN"),
                            "name": (pair.get("baseToken") or {}).get("name", "UNKNOWN"),
                            "volume_24h": float(volume_24h),
                            "liquidity": float(liquidity),
                            "market_cap": float(_mcap),
                            "creation_timestamp": (int(created) // 1000) if created else 0,
                            "timestamp": int(time.time()),
                            "categories": ["no_creation_time"] if not created else [],
                            # Price: keep None when unknown (do NOT coerce to 0)
                            "price": float(price_val) if (price_val is not None and price_val > 0) else None,
                            # flattened changes for downstream DB/UI
                            "price_change_1h": _pc1h,
                            "price_change_6h": _pc6h,
                            "price_change_24h": _pc24h,
                            "pair_address": pair.get("pairAddress"),
                            "source": "dexscreener",
                        })

                    got_page = True
                    break  # success on one base variant
            except (aiohttp.ClientError, ValueError, asyncio.TimeoutError) as e:
                logger.warning("Dexscreener page=%d fetch failed (%s): %s", page, base, e)
                continue

        # small politeness delay (helps with bot heuristics on Windows)
        await asyncio.sleep(0.25 if got_page else 0.6)

    deduped = _dedupe_by_base_address(rows)
    cap = _dex_post_cap()
    if cap and len(deduped) > cap:
        deduped = deduped[:cap]
    logger.info("Dexscreener classic: %d unique", len(deduped))
    return deduped

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=180),
    retry=retry_if_exception_type((aiohttp.ClientError, ValueError)),
)
async def fetch_dexscreener_search(
    session: aiohttp.ClientSession,
    query: str = DEXSCREENER_QUERY_ENV,
    pages: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Wrapper: try classic first; if too few uniques, fallback to sharded search
    over multiple alt queries and merge (dedup by base address).
    """
    hdrs = _default_dex_headers(headers)
    cfg_pages, cfg_per_page = _dex_pages_perpage()
    pages_to_fetch = int(pages) if pages is not None else cfg_pages
    per_page_cap = cfg_per_page

    # --- 1) Classic pass ---
    deduped = await fetch_dexscreener_search_classic(
        session, query=query, pages=pages_to_fetch, per_page=per_page_cap, headers=hdrs
    )

    # If classic yielded enough, stop here
    fallback_min = int(os.getenv("DEX_FALLBACK_MIN_UNIQUES", "10"))
    if len(deduped) >= fallback_min:
        return deduped

    # --- 2) Fallback: sharded search over alt queries ---
    # Keep this gentle to avoid rate limits, but broad enough to escape filters.
    alt_env = [q.strip() for q in os.getenv("DEXSCREENER_ALT_QUERIES", "").split(",") if q.strip()]
    alt_defaults = [query, f"{query} ", "sol", "sola", "a", "e", "i", "o", "u", "t"]
    alt = (alt_env or alt_defaults)[:8]  # cap attempts

    logger.info("Dexscreener fallback: trying shards/alt queries: %r", alt)

    shards = _dex_query_shards(max(1, pages_to_fetch))
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)

    async def fetch_shard(q: str, sh: str) -> List[Dict[str, Any]]:
        url = f"https://api.dexscreener.com/latest/dex/search?q={q}{sh}"
        try:
            async with session.get(url, headers=hdrs, timeout=timeout) as resp:
                if resp.status == 429:
                    # gentle throttling; skip this shard
                    await asyncio.sleep(1.2)
                    return []
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
                pairs = (data.get("pairs") or [])[:per_page_cap]
                now_ms = int(time.time() * 1000)
                sol_launch_ms = 1581465600000
                out: List[Dict[str, Any]] = []
                for pair in pairs:
                    # tolerant chain check
                    chain_val = (pair.get("chainId") or pair.get("chain") or "")
                    try:
                        chain_check = str(chain_val).strip().lower()
                    except Exception:
                        chain_check = ""
                    if "sol" not in chain_check and "solana" not in chain_check:
                        continue

                    # robust base token address extraction
                    token_address = (pair.get("baseToken") or {}).get("address") or (pair.get("base") or {}).get("mint") or pair.get("baseMint") or ""
                    if not token_address:
                        continue
                    token_address = str(token_address).strip()
                    try:
                        Pubkey.from_string(token_address)
                    except Exception:
                        continue

                    # created may be seconds or ms; normalize to ms and validate
                    created_raw = pair.get("pairCreatedAt") or pair.get("createdAt") or pair.get("createdTime") or 0
                    created_ms = None
                    try:
                        if created_raw:
                            created_val = float(created_raw)
                            if created_val < 10_000_000_000:
                                created_ms = int(created_val * 1000)
                            else:
                                created_ms = int(created_val)
                    except Exception:
                        created_ms = None
                    if not (created_ms and sol_launch_ms <= created_ms <= now_ms):
                        created_ms = 0

                    # --- normalize priceChange and market cap for shard path ---
                    _pc = pair.get("priceChange") or {}
                    _pc1h = float((_pc.get("h1") or _pc.get("1h") or 0) or 0)
                    _pc6h = float((_pc.get("h6") or _pc.get("6h") or 0) or 0)
                    _pc24h = float((_pc.get("h24") or _pc.get("24h") or 0) or 0)

                    # defensive market cap / volume / liquidity parsing
                    _mcap = _parse_float_like(
                        pair.get("fdv") or pair.get("fdvUsd") or pair.get("marketCap") or pair.get("marketCapUsd") or 0
                    ) or 0.0

                    volume_24h = _parse_float_like(
                        (pair.get("volume") or {}).get("h24") or (pair.get("volume") or {}).get("24h") or 0
                    ) or 0.0
                    liquidity = _parse_float_like(
                        (pair.get("liquidity") or {}).get("usd") or pair.get("liquidity") or 0
                    ) or 0.0

                    # Price: do not coerce to 0; prefer None if missing.
                    price_val = _parse_float_like(pair.get("priceUsd") or pair.get("price") or None)

                    out.append({
                        "address": token_address,
                        "token_address": token_address,
                        "symbol": (pair.get("baseToken") or {}).get("symbol", "UNKNOWN"),
                        "name": (pair.get("baseToken") or {}).get("name", "UNKNOWN"),
                        "volume_24h": float(volume_24h),
                        "liquidity": float(liquidity),
                        "market_cap": float(_mcap),
                        "creation_timestamp": (created_ms // 1000) if created_ms else 0,
                        "timestamp": int(time.time()),
                        "categories": ["no_creation_time"] if not created_ms else [],
                        "price": float(price_val) if (price_val is not None and price_val > 0) else None,
                        "price_change_1h": _pc1h,
                        "price_change_6h": _pc6h,
                        "price_change_24h": _pc24h,
                        "pair_address": pair.get("pairAddress"),
                        "source": "dexscreener",
                    })
                return out
        except (aiohttp.ClientError, ValueError, asyncio.TimeoutError):
            return []

    # Moderate parallelism across alt queries × shards
    tasks = [fetch_shard(q, sh) for q in alt for sh in shards]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    extra: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, list):
            extra.extend(r)

    merged = _dedupe_by_base_address(deduped + extra)
    cap = _dex_post_cap()
    if cap and len(merged) > cap:
        merged = merged[:cap]

    return merged

# ---------------------------------------------------------------------
# Raydium (pairs) — streaming early-stop
# ---------------------------------------------------------------------
async def fetch_raydium_tokens(
    session: aiohttp.ClientSession,
    solana_client: AsyncClient,  # kept for signature parity
    max_pairs: Optional[int] = None,
    *,
    max_download_bytes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    # --- Feature flags: bail out early if disabled ---
    try:
        if 'FORCE_DISABLE_RAYDIUM' in globals() and FORCE_DISABLE_RAYDIUM:
            logger.info("Raydium disabled by FORCE_DISABLE_RAYDIUM — skipping fetch.")
            return []
        if 'is_enabled_raydium' in globals():
            try:
                enabled = is_enabled_raydium()  # some builds take 0 args
            except TypeError:
                # fallback: some builds take (config); else read env
                enabled = os.getenv("RAYDIUM_ENABLE", "false").strip().lower() not in ("0", "false", "no", "off")
            if not enabled:
                logger.info("Raydium disabled by config/env — skipping fetch.")
                return []
        else:
            if os.getenv("RAYDIUM_ENABLE", "false").strip().lower() in ("0", "false", "no", "off"):
                logger.info("Raydium disabled via env/config — skipping fetch.")
                return []
    except Exception:
        logger.info("Raydium disabled (flag error fallback) — skipping fetch.")
        return []

    # ---- helpers ----
    def _num(v, d: float = 0.0) -> float:
        try:
            return float(v if v not in (None, "") else d)
        except Exception:
            return float(d)

    def _better_row(new_t: Dict[str, Any], old_t: Dict[str, Any]) -> bool:
        """Prefer higher (liquidity, volume_24h, market_cap)."""
        a = (_num(new_t.get("liquidity")), _num(new_t.get("volume_24h")), _num(new_t.get("market_cap")))
        b = (_num(old_t.get("liquidity")), _num(old_t.get("volume_24h")), _num(old_t.get("market_cap")))
        return a > b

    # Light quality gates (envs; default to 0)
    min_liq = float(os.getenv("MIN_LIQUIDITY_USD", "0"))
    min_vol = float(os.getenv("MIN_VOLUME24_USD", "0"))
    min_mc  = float(os.getenv("MIN_MARKETCAP_USD", "0"))

    def _passes_minimums(liq: float, vol: float, mc: float) -> bool:
        return (
            (not min_liq or liq >= min_liq) and
            (not min_vol or vol >= min_vol) and
            (not min_mc  or mc  >= min_mc)
        )

    # Unique-token cap
    try:
        parse_cap, _ = _ray_limits(max_pairs_hint=max_pairs)  # if present in your codebase
        parse_cap = int(parse_cap or 0)
    except Exception:
        parse_cap = int(max_pairs or 0)
    if parse_cap <= 0:
        parse_cap = int(os.getenv("RAYDIUM_PARSE_CAP", "300"))

    # Byte budget (streaming); max_download_bytes takes precedence
    if max_download_bytes is not None:
        byte_budget = int(max_download_bytes)
    else:
        # prefer env; fallback 16 MiB
        byte_budget = int(os.getenv("RAYDIUM_READ_BYTE_BUDGET", str(16 * 1024 * 1024)))

    url = "https://api.raydium.io/v2/main/pairs"
    headers = {"Accept": "application/json", "User-Agent": "SOLOTradingBot/1.0"}
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)

    # ---- streaming array-of-objects parser (no extra deps) ----
    async def _iter_top_level_objects_from_stream(resp: aiohttp.ClientResponse, *, budget: int):
        """
        Yields dict objects from a JSON like: [ {...}, {...}, ... ]
        Stops if 'budget' bytes consumed.
        """
        buf = bytearray()
        consumed = 0
        depth = 0
        in_string = False
        escape = False
        saw_array_start = False
        obj_start = None  # index where current object starts in buf

        async for chunk in resp.content.iter_chunked(65536):
            buf.extend(chunk)
            consumed += len(chunk)
            if consumed > budget:
                break

            i = 0
            while i < len(buf):
                c = chr(buf[i])

                if not saw_array_start:
                    if c == '[':
                        saw_array_start = True
                    i += 1
                    continue

                if obj_start is None:
                    if c == '{':
                        obj_start = i
                        depth = 1
                        in_string = False
                        escape = False
                    i += 1
                    continue

                if in_string:
                    if escape:
                        escape = False
                    elif c == '\\':
                        escape = True
                    elif c == '"':
                        in_string = False
                else:
                    if c == '"':
                        in_string = True
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            obj_bytes = bytes(buf[obj_start:i+1])
                            try:
                                yield json.loads(obj_bytes)
                            except Exception:
                                pass
                            # drop processed bytes
                            del buf[:i+1]
                            i = -1
                            obj_start = None
                i += 1
        # ignore trailing partials

    # ---- request + stream ----
    best_by_addr: Dict[str, Dict[str, Any]] = {}
    now_ts = int(time.time())

    try:
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status == 429:
                reset = resp.headers.get("X-RateLimit-Reset")
                try:
                    to_sleep = max(5.0, float(reset) - time.time()) if reset else 15.0
                except Exception:
                    to_sleep = 15.0
                logger.warning("Raydium 429; sleeping %.1fs", to_sleep)
                await asyncio.sleep(to_sleep)
                return []

            if resp.status != 200:
                body = (await resp.text())[:500]
                logger.error("Raydium HTTP %s: %s", resp.status, body)
                return []

            # Stream objects and stop early when we have enough uniques
            async for p in _iter_top_level_objects_from_stream(resp, budget=byte_budget):
                # Extract base token address robustly
                addr = (
                    ((p.get("base") or {}).get("address"))
                    or p.get("baseMint")
                    or p.get("mint")
                    or ""
                )
                addr = (addr or "").strip()
                if not addr:
                    continue
                try:
                    Pubkey.from_string(addr)
                except Exception:
                    continue

                # Metrics
                def _vol24(obj: Dict[str, Any]) -> float:
                    return _num(obj.get("volume24h") or (obj.get("volume") or {}).get("h24") or 0)

                liq = _num(p.get("liquidity") or (p.get("liquidity") or {}).get("usd") or 0)
                vol = _vol24(p)
                mc  = _num(p.get("fdv") or p.get("marketCap") or 0)

                if not _passes_minimums(liq, vol, mc):
                    continue

                # created time (ms) -> seconds
                created_ms = None
                for k in ("createdAt", "createdTimestamp", "createdTime", "timestamp", "created_at"):
                    v = p.get(k)
                    if v is not None:
                        try:
                            v = float(v)
                            if v and v < 10_000_000_000:
                                v *= 1000.0
                            created_ms = v
                            break
                        except Exception:
                            pass
                creation_timestamp = int(created_ms / 1000) if created_ms else 0

                symbol = (
                    (p.get("base") or {}).get("symbol")
                    or (p.get("name") or "").split("/")[0]
                    or p.get("symbol")
                    or "UNKNOWN"
                )
                name = (
                    (p.get("base") or {}).get("name")
                    or p.get("name")
                    or symbol
                )

                tok = {
                    "address": addr,
                    "symbol": symbol,
                    "name": name,
                    "volume_24h": vol,
                    "liquidity": liq,
                    "market_cap": mc,
                    "creation_timestamp": creation_timestamp,
                    "timestamp": now_ts,
                    "price": _num(p.get("price") or 0),
                    "source": "raydium",
                }

                prev = best_by_addr.get(addr)
                if prev is None or _better_row(tok, prev):
                    best_by_addr[addr] = tok

                if len(best_by_addr) >= parse_cap:
                    break  # EARLY STOP

    except asyncio.TimeoutError:
        logger.warning("Raydium request timed out.")
        return []
    except MemoryError as e:
        logger.warning("Raydium response exceeded safe size: %s", e)
        return []
    except (aiohttp.ClientError, ValueError) as e:
        logger.error("Raydium error: %s", e, exc_info=True)
        return []

    tokens = list(best_by_addr.values())
    logger.info(
        "Raydium streamed: %d unique tokens (cap=%d, budget=%d bytes)",
        len(tokens), parse_cap, byte_budget
    )
    return tokens

# ---------------------------------------------------------------------
# Birdeye (tokens) — canonical public-first implementation
# ---------------------------------------------------------------------
# Host selection: explicit base URL > BIRDEYE_MODE > public starter gateway
BIRDEYE_MODE = os.getenv("BIRDEYE_MODE", "public").strip().lower()
_user_base = (os.getenv("BIRDEYE_BASE_URL") or "").strip()
if _user_base:
    BIRDEYE_BASE_URL = _user_base.rstrip("/")
else:
    if BIRDEYE_MODE in ("pro", "v2"):
        BIRDEYE_BASE_URL = "https://api.birdeye.so"  # pro base
    else:
        BIRDEYE_BASE_URL = "https://public-api.birdeye.so"  # public starter gateway

# Derived flag for pro
BIRDEYE_PRO = BIRDEYE_MODE in ("pro", "v2")

try:
    logger.info("Birdeye mode=%s base_url=%s BIRDEYE_API_KEY_present=%s",
                BIRDEYE_MODE, BIRDEYE_BASE_URL, bool(os.getenv("BIRDEYE_API_KEY")))
except Exception:
    pass

def _birdeye_headers_public(api_key: Optional[str]) -> Dict[str, str]:
    """Headers for the public gateway / Starter plan.

    Minimal headers per documentation: X-API-KEY and x-chain (plus common Accept/User-Agent).
    """
    k = (api_key or "").strip()
    headers = {
        "Accept": "application/json",
        "User-Agent": "SOLOTradingBot/1.0",
        "x-chain": "solana",
    }
    if k:
        headers["X-API-KEY"] = k
    return headers

def _rl_from_headers_safe(h: Dict[str, str]) -> float:
    """Safe read of rate-limit headers => seconds to wait (small cushion)."""
    try:
        reset = float(h.get("x-ratelimit-reset", "0"))
        now = time.time()
        if reset > now:
            return max(0.8, reset - now)
    except Exception:
        pass
    try:
        ra = float(h.get("retry-after", "0"))
        return max(0.8, ra)
    except Exception:
        return 0.0

def _chunks(seq: Iterable, n: int):
    it = list(seq)
    for i in range(0, len(it), n):
        yield it[i:i+n]


async def _birdeye_multi_price_fallback_per_address(
    session: aiohttp.ClientSession,
    addresses: List[str],
    api_key: str,
    birdeye_gate: Optional[Any],
    base_url: str,
    timeout: aiohttp.ClientTimeout,
) -> Dict[str, Any]:
    """
    Fallback: per-address GET to /defi/price with strict, conservative throttling.
    Returns mapping address -> payload (or None if absent).
    """
    # ensure module-level backoff names are declared global before any use in this function
    global _BIRDEYE_LAST_BACKOFF, _BIRDEYE_COOLDOWN_UNTIL

    out: Dict[str, Any] = {}
    if not addresses:
        return out

    per_url = f"{base_url.rstrip('/')}/defi/price"
    # build headers via existing helper if available; otherwise plain Bearer
    try:
        headers = _birdeye_headers_public(api_key)
    except Exception:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # conservative semaphore derived from RPS env (1.5x rps)
    try:
        rps = max(1, int(os.getenv("BIRDEYE_RPS", "2")))
    except Exception:
        rps = 2
    sem = asyncio.Semaphore(max(1, int(rps * 1.5)))

    for addr in addresses:
        async with sem:
            try:
                if birdeye_gate is not None:
                    gate_ctx = await birdeye_gate.request()
                    async with gate_ctx:
                        async with session.get(per_url, headers=headers, params={"address": addr}, timeout=timeout) as r:
                            if r.status == 200:
                                j = await r.json(content_type=None)
                                out[addr] = j.get("data") or j
                            elif r.status == 429:
                                wait = parse_retry_after_seconds(r.headers) or _BIRDEYE_DOWN_BACKOFF_DEFAULT
                                wait = max(1.0, min(300.0, float(wait)))
                                try:
                                    await birdeye_gate.mark_down(wait)
                                except Exception:
                                    logger.debug("birdeye_gate.mark_down failed", exc_info=True)
                                logger.warning("Birdeye per-address fallback got 429; sleeping %.1fs", wait)
                                await asyncio.sleep(wait)
                                out[addr] = None
                            else:
                                out[addr] = None
                else:
                    async with session.get(per_url, headers=headers, params={"address": addr}, timeout=timeout) as r:
                        if r.status == 200:
                            j = await r.json(content_type=None)
                            out[addr] = j.get("data") or j
                        elif r.status == 429:
                            wait = parse_retry_after_seconds(r.headers) or _BIRDEYE_DOWN_BACKOFF_DEFAULT
                            wait = max(1.0, min(300.0, float(wait)))
                            # set module-local fallback as defensive measure
                            try:
                                _BIRDEYE_LAST_BACKOFF = min(25, max(8, (_BIRDEYE_LAST_BACKOFF * 2) if _BIRDEYE_LAST_BACKOFF else 8))
                                _BIRDEYE_COOLDOWN_UNTIL = time.time() + max(_BIRDEYE_LAST_BACKOFF, wait)
                                logger.warning("Birdeye module-local backoff set for %.1fs", _BIRDEYE_COOLDOWN_UNTIL - time.time())
                            except Exception:
                                pass
                            await asyncio.sleep(wait)
                            out[addr] = None
                        else:
                            out[addr] = None
            except Exception:
                logger.debug("per-address birdeye fallback error for %s", addr, exc_info=True)
                out[addr] = None
    return out

async def birdeye_multi_price(
    session: aiohttp.ClientSession,
    addresses: List[str],
    api_key: str,
    birdeye_gate: Optional[Any],
    base_url: Optional[str] = None,
    chunk_size: int = 50,
    timeout: Optional[aiohttp.ClientTimeout] = None,
    max_retries: int = 2,
    retry_backoff_base: float = 0.8,
) -> Dict[str, Any]:
    """
    Batch price lookups using Birdeye /defi/multi_price.

    Preferred request order (more compatible with public gateway / observed behavior):
      1) GET ?list_address=addr1,addr2,...  (works in your curl example)
      2) POST json={"list_address": "addr1,addr2,..."} (some gateways accept string payload)
      3) POST json={"addresses": [...]}
      4) Fallback: per-address GET to /defi/price

    The function is defensive about rate-limit (429) and will use `birdeye_gate`
    if provided to cooperatively mark the service down.
    """
    # module-level backoff names
    global _BIRDEYE_LAST_BACKOFF, _BIRDEYE_COOLDOWN_UNTIL

    out: Dict[str, Any] = {}
    if not addresses:
        return out

    base = (base_url or globals().get("BIRDEYE_BASE_URL", "")).rstrip("/")
    if not base:
        raise RuntimeError("birdeye_multi_price requires base_url or BIRDEYE_BASE_URL global")
    multi_url = f"{base}/defi/multi_price"

    # headers builder
    try:
        headers = _birdeye_headers_public(api_key)
    except Exception:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    if timeout is None:
        timeout = aiohttp.ClientTimeout(total=12, sock_connect=6, sock_read=8)

    def _is_list_address_error(text: str) -> bool:
        try:
            if not text:
                return False
            return "list_address" in text and ("required" in text or "is required" in text)
        except Exception:
            return False

    # Helper: chunk addresses into batches
    def _chunks(seq: Iterable[str], n: int):
        it = list(seq)
        for i in range(0, len(it), n):
            yield it[i:i+n]

    # Normalize & dedupe input addresses
    addrs = [str(a).strip() for a in dict.fromkeys(list(addresses)) if str(a).strip()]
    if not addrs:
        return {}

    # Emit a single (redacted) info log so operators can see which base/key we're using.
    try:
        red = (api_key[:4] + "..." + api_key[-4:]) if api_key else "<none>"
        logger.info("Birdeye multi_price using base=%s key=%s", base, red)
    except Exception:
        # never fail the function because logging failed
        pass

    # Try chunked calls
    for chunk in _chunks(addrs, int(chunk_size)):
        attempt = 0
        comma_list = ",".join(chunk)
        success_for_chunk = False

        while attempt <= max_retries and not success_for_chunk:
            attempt += 1
            # 1) Try GET with list_address=<comma-separated> first (matches working curl)
            try:
                if birdeye_gate is not None:
                    gate_ctx = await birdeye_gate.request()
                    async with gate_ctx:
                        async with session.get(multi_url, headers=headers, params={"list_address": comma_list}, timeout=timeout) as r:
                            text = await r.text()
                            if r.status == 200:
                                try:
                                    j = json.loads(text)
                                except Exception:
                                    j = await r.json(content_type=None)
                                data = j.get("data") or j
                                if isinstance(data, dict):
                                    out.update({k: v for k, v in data.items() if isinstance(k, str)})
                                elif isinstance(data, list):
                                    for it in data:
                                        if isinstance(it, dict):
                                            addr = (it.get("address") or it.get("tokenAddress") or "").strip()
                                            if addr:
                                                out[addr] = it
                                success_for_chunk = True
                                break
                            elif r.status == 429:
                                wait = parse_retry_after_seconds(r.headers) or _BIRDEYE_DOWN_BACKOFF_DEFAULT
                                wait = max(1.0, min(300.0, float(wait)))
                                try:
                                    await birdeye_gate.mark_down(wait)
                                except Exception:
                                    pass
                                logger.warning("Birdeye multi_price GET chunk got 429; backing off %.1fs", wait)
                                await asyncio.sleep(wait)
                                # try again depending on attempt loop
                                continue
                            elif r.status in (400,):
                                # try alternative shapes below; keep the response body for diagnostics
                                logger.info("Birdeye multi_price GET returned 400; trying POST fallbacks. body[:512]=%s", (text or "")[:512])
                            elif r.status in (401, 403):
                                logger.error("Birdeye multi_price GET auth error (%s)", r.status)
                                success_for_chunk = True  # stop attempts for this chunk (auth won't be fixed by retries)
                                break
                            else:
                                # unexpected non-200; log and proceed to fallback shapes
                                logger.debug("Birdeye multi_price GET unexpected status=%s body=%s", r.status, (text or "")[:512])
                else:
                    async with session.get(multi_url, headers=headers, params={"list_address": comma_list}, timeout=timeout) as r:
                        text = await r.text()
                        if r.status == 200:
                            try:
                                j = json.loads(text)
                            except Exception:
                                j = await r.json(content_type=None)
                            data = j.get("data") or j
                            if isinstance(data, dict):
                                out.update({k: v for k, v in data.items() if isinstance(k, str)})
                            elif isinstance(data, list):
                                for it in data:
                                    if isinstance(it, dict):
                                        addr = (it.get("address") or it.get("tokenAddress") or "").strip()
                                        if addr:
                                            out[addr] = it
                            success_for_chunk = True
                            break
                        elif r.status == 429:
                            wait = parse_retry_after_seconds(r.headers) or _BIRDEYE_DOWN_BACKOFF_DEFAULT
                            wait = max(1.0, min(300.0, float(wait)))
                            # module-local fallback backoff
                            try:
                                _BIRDEYE_LAST_BACKOFF = min(25, max(8, (_BIRDEYE_LAST_BACKOFF * 2) if _BIRDEYE_LAST_BACKOFF else 8))
                                _BIRDEYE_COOLDOWN_UNTIL = time.time() + max(_BIRDEYE_LAST_BACKOFF, wait)
                            except Exception:
                                pass
                            logger.warning("Birdeye multi_price GET chunk got 429; backing off %.1fs", wait)
                            await asyncio.sleep(wait)
                            continue
                        elif r.status in (400,):
                            logger.info("Birdeye multi_price GET returned 400; will try POST fallbacks. body[:512]=%s", (await r.text() or "")[:512])
                        elif r.status in (401, 403):
                            logger.error("Birdeye multi_price GET auth error (%s)", r.status)
                            success_for_chunk = True
                            break
                        else:
                            logger.debug("Birdeye multi_price GET unexpected status=%s", r.status)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug("Birdeye multi_price GET transient error (try=%d/%d): %s", attempt, max_retries, e)
                await asyncio.sleep(retry_backoff_base * (2 ** (attempt - 1)))
                continue
            except Exception:
                logger.exception("Birdeye multi_price GET unexpected failure", exc_info=True)
                await asyncio.sleep(retry_backoff_base * (2 ** (attempt - 1)))
                continue

            # If GET either returned 400 or didn't succeed, try POST fallbacks:
            # 2) POST json={"list_address": "addr1,addr2,..."} (string) — many public gateways accept this
            try:
                payload = {"list_address": comma_list}
                if birdeye_gate is not None:
                    gate_ctx = await birdeye_gate.request()
                    async with gate_ctx:
                        async with session.post(multi_url, headers=headers, json=payload, timeout=timeout) as r:
                            text = await r.text()
                            if r.status == 200:
                                try:
                                    j = json.loads(text)
                                except Exception:
                                    j = await r.json(content_type=None)
                                data = j.get("data") or j
                                if isinstance(data, dict):
                                    out.update({k: v for k, v in data.items() if isinstance(k, str)})
                                elif isinstance(data, list):
                                    for it in data:
                                        if isinstance(it, dict):
                                            addr = (it.get("address") or it.get("tokenAddress") or "").strip()
                                            if addr:
                                                out[addr] = it
                                success_for_chunk = True
                                break
                            elif r.status == 429:
                                wait = parse_retry_after_seconds(r.headers) or _BIRDEYE_DOWN_BACKOFF_DEFAULT
                                wait = max(1.0, min(300.0, float(wait)))
                                try:
                                    await birdeye_gate.mark_down(wait)
                                except Exception:
                                    logger.debug("birdeye_gate.mark_down failed", exc_info=True)
                                logger.warning("Birdeye multi_price POST(list_address string) got 429; backing off %.1fs", wait)
                                await asyncio.sleep(wait)
                                continue
                            else:
                                logger.info("Birdeye multi_price POST(list_address string) HTTP %s body[:512]=%s", r.status, (text or "")[:512])
                else:
                    async with session.post(multi_url, headers=headers, json={"list_address": comma_list}, timeout=timeout) as r:
                        text = await r.text()
                        if r.status == 200:
                            try:
                                j = json.loads(text)
                            except Exception:
                                j = await r.json(content_type=None)
                            data = j.get("data") or j
                            if isinstance(data, dict):
                                out.update({k: v for k, v in data.items() if isinstance(k, str)})
                            elif isinstance(data, list):
                                for it in data:
                                    if isinstance(it, dict):
                                        addr = (it.get("address") or it.get("tokenAddress") or "").strip()
                                        if addr:
                                            out[addr] = it
                            success_for_chunk = True
                            break
                        elif r.status == 429:
                            wait = parse_retry_after_seconds(r.headers) or _BIRDEYE_DOWN_BACKOFF_DEFAULT
                            wait = max(1.0, min(300.0, float(wait)))
                            try:
                                _BIRDEYE_LAST_BACKOFF = min(25, max(8, (_BIRDEYE_LAST_BACKOFF * 2) if _BIRDEYE_LAST_BACKOFF else 8))
                                _BIRDEYE_COOLDOWN_UNTIL = time.time() + max(_BIRDEYE_LAST_BACKOFF, wait)
                            except Exception:
                                pass
                            logger.warning("Birdeye multi_price POST(list_address string) got 429; backing off %.1fs", wait)
                            await asyncio.sleep(wait)
                            continue
                        else:
                            logger.info("Birdeye multi_price POST(list_address string) HTTP %s body[:512]=%s", r.status, (text or "")[:512])
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug("Birdeye multi_price POST(list_address string) transient error (try=%d/%d): %s", attempt, max_retries, e)
                await asyncio.sleep(retry_backoff_base * (2 ** (attempt - 1)))
                continue
            except Exception:
                logger.exception("Birdeye multi_price POST(list_address string) unexpected failure", exc_info=True)
                await asyncio.sleep(retry_backoff_base * (2 ** (attempt - 1)))
                continue

            # 3) Legacy POST json={"addresses": [...]} (older/other gateways)
            try:
                if birdeye_gate is not None:
                    gate_ctx = await birdeye_gate.request()
                    async with gate_ctx:
                        async with session.post(multi_url, headers=headers, json={"addresses": chunk}, timeout=timeout) as r:
                            text = await r.text()
                            if r.status == 200:
                                try:
                                    j = json.loads(text)
                                except Exception:
                                    j = await r.json(content_type=None)
                                data = j.get("data") or j
                                if isinstance(data, dict):
                                    out.update({k: v for k, v in data.items() if isinstance(k, str)})
                                elif isinstance(data, list):
                                    for it in data:
                                        if isinstance(it, dict):
                                            addr = (it.get("address") or it.get("tokenAddress") or "").strip()
                                            if addr:
                                                out[addr] = it
                                success_for_chunk = True
                                break
                            elif r.status == 429:
                                wait = parse_retry_after_seconds(r.headers) or _BIRDEYE_DOWN_BACKOFF_DEFAULT
                                wait = max(1.0, min(300.0, float(wait)))
                                try:
                                    await birdeye_gate.mark_down(wait)
                                except Exception:
                                    logger.debug("birdeye_gate.mark_down failed", exc_info=True)
                                logger.warning("Birdeye multi_price fallback chunk got 429; backing off %.1fs", wait)
                                await asyncio.sleep(wait)
                                continue
                            elif r.status in (400, 404):
                                logger.info("Birdeye multi_price fallback returned %s; will fall back to per-address for %d addresses", r.status, len(chunk))
                                # fall through to per-address fallback
                                break
                            else:
                                logger.debug("Birdeye multi_price fallback HTTP %s body[:512]=%s", r.status, (text or "")[:512])
                                await asyncio.sleep(retry_backoff_base * (2 ** (attempt - 1)))
                                continue
                else:
                    async with session.post(multi_url, headers=headers, json={"addresses": chunk}, timeout=timeout) as r:
                        text = await r.text()
                        if r.status == 200:
                            try:
                                j = json.loads(text)
                            except Exception:
                                j = await r.json(content_type=None)
                            data = j.get("data") or j
                            if isinstance(data, dict):
                                out.update({k: v for k, v in data.items() if isinstance(k, str)})
                            elif isinstance(data, list):
                                for it in data:
                                    if isinstance(it, dict):
                                        addr = (it.get("address") or it.get("tokenAddress") or "").strip()
                                        if addr:
                                            out[addr] = it
                            success_for_chunk = True
                            break
                        elif r.status == 429:
                            wait = parse_retry_after_seconds(r.headers) or _BIRDEYE_DOWN_BACKOFF_DEFAULT
                            wait = max(1.0, min(300.0, float(wait)))
                            try:
                                _BIRDEYE_LAST_BACKOFF = min(25, max(8, (_BIRDEYE_LAST_BACKOFF * 2) if _BIRDEYE_LAST_BACKOFF else 8))
                                _BIRDEYE_COOLDOWN_UNTIL = time.time() + max(_BIRDEYE_LAST_BACKOFF, wait)
                                logger.warning("Birdeye module-local backoff set for %.1fs", _BIRDEYE_COOLDOWN_UNTIL - time.time())
                            except Exception:
                                pass
                            await asyncio.sleep(wait)
                            continue
                        elif r.status in (400, 404):
                            logger.info("Birdeye multi_price fallback returned %s; will fall back to per-address for %d addresses", r.status, len(chunk))
                            break
                        else:
                            await asyncio.sleep(retry_backoff_base * (2 ** (attempt - 1)))
                            continue
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug("Birdeye multi_price fallback transient error (try=%d/%d): %s", attempt, max_retries, e)
                await asyncio.sleep(retry_backoff_base * (2 ** (attempt - 1)))
                continue
            except Exception:
                logger.exception("Birdeye multi_price fallback unexpected failure", exc_info=True)
                await asyncio.sleep(retry_backoff_base * (2 ** (attempt - 1)))
                continue

            # If we reach here without success, break the retry loop; next iteration attempt>max_retries will exit
            break

        # If we never got success_for_chunk, fallback to per-address retrieval for that chunk
        if not success_for_chunk:
            try:
                fb = await _birdeye_multi_price_fallback_per_address(session, chunk, api_key, birdeye_gate, base, timeout)
                out.update(fb)
            except Exception:
                logger.exception("birdeye_multi_price per-address fallback failed for chunk")
                for a in chunk:
                    out.setdefault(a, None)

    return out

# --- replace apply_batched_birdeye_prices with normalized-address mapping ----
async def apply_batched_birdeye_prices(
    session: aiohttp.ClientSession,
    tokens: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    birdeye_gate_obj: Optional[Any] = None,
    base_url: Optional[str] = None,
    chunk_size: int = 50,
):
    """
    Fetch batched price payloads for tokens and apply canonical fields onto token dicts in place.

    - Normalizes addresses (extract base58 mint) before calling Birdeye.
    - Maintains an addr_map mapping normalized_addr -> token so results map correctly.
    - Preserves original raw address in token['_raw_address'] for debugging/UI.
    """
    if not tokens:
        return

    api_key = api_key if api_key is not None else os.getenv("BIRDEYE_API_KEY", "")
    if not api_key:
        logger.debug("apply_batched_birdeye_prices: no BIRDEYE_API_KEY available; skipping")
        return

    # Build normalized mapping
    addr_map: Dict[str, Dict[str, Any]] = {}
    addresses: List[str] = []
    for t in tokens:
        raw = (t.get("address") or t.get("token_address") or "") or ""
        raw = str(raw).strip()
        norm = _extract_base58_address(raw) or raw
        # store original raw for traceability
        t.setdefault("_raw_address", raw)
        # prefer validated Pubkey if possible (skip obviously invalid)
        try:
            Pubkey.from_string(norm)
        except Exception:
            # keep raw as last resort and continue; do not send empty addresses
            logger.debug("apply_batched_birdeye_prices: skipping invalid address %r", raw)
            continue
        # if normalization changed the value, log once (debug)
        if norm != raw:
            logger.debug("Normalized address: %r -> %r", raw, norm)
            # optionally update token address to canonical form so downstream logic uses it
            t["address"] = norm
            t["token_address"] = norm
        addr_map[norm] = t
        addresses.append(norm)

    if not addresses:
        return

    try:
        results = await birdeye_multi_price(
            session=session,
            addresses=addresses,
            api_key=api_key,
            birdeye_gate=birdeye_gate_obj if birdeye_gate_obj is not None else globals().get("birdeye_gate"),
            base_url=base_url or globals().get("BIRDEYE_BASE_URL"),
            chunk_size=chunk_size,
        )
    except Exception:
        logger.exception("apply_batched_birdeye_prices: multi_price failed")
        results = {}

    # apply results: be resilient to keys returned in either normalized or raw form
    for norm_addr, tok in addr_map.items():
        # try normalized key first, then raw
        payload = results.get(norm_addr)
        if payload is None:
            raw_key = tok.get("_raw_address")
            if raw_key:
                payload = results.get(raw_key)
        if not payload:
            continue
        if isinstance(payload, dict):
            # common keys: price, priceUsd, price_change_1h, v1h, v6h, v24h
            p = payload.get("price") if payload.get("price") is not None else payload.get("priceUsd")
            tok["price"] = float(_parse_float_like(p)) if _parse_float_like(p) is not None else tok.get("price")
            tok["price_change_1h"] = _parse_float_like(payload.get("price_change_1h") or payload.get("v1h") or payload.get("h1") or None) or tok.get("price_change_1h")
            tok["price_change_6h"] = _parse_float_like(payload.get("price_change_6h") or payload.get("v6h") or payload.get("h6") or None) or tok.get("price_change_6h")
            tok["price_change_24h"] = _parse_float_like(payload.get("price_change_24h") or payload.get("v24h") or payload.get("h24") or None) or tok.get("price_change_24h")
        else:
            continue

# --- replace/enhance enrich_tokens_with_price_change to use normalized addresses --
async def enrich_tokens_with_price_change(
    session: aiohttp.ClientSession | None = None,
    tokens: Optional[List[Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
    blacklist: Optional[set] = None,
    failure_count: Optional[Dict[str, int]] = None,
    concurrency: int = 25,
    per_token_timeout: float = 7.0,
) -> List[Dict[str, Any]]:
    """
    Enrich tokens with Birdeye price/change fields using normalized addresses.

    - Normalizes addresses before enrichment.
    - Uses apply_batched_birdeye_prices which now accepts normalized mapping.
    """
    if not tokens:
        return tokens or []

    own_session = False
    if session is None:
        session = aiohttp.ClientSession()
        own_session = True

    try:
        api_key = os.getenv("BIRDEYE_API_KEY", "") or ""
        if not api_key:
            if logger:
                try:
                    logger.debug("enrich_tokens_with_price_change: no BIRDEYE_API_KEY; skipping enrichment")
                except Exception:
                    pass
            return tokens

        # shallow copy
        tokens = list(tokens or [])

        # First pass: normalized batched enrichment
        try:
            await apply_batched_birdeye_prices(
                session=session,
                tokens=tokens,
                api_key=api_key,
                birdeye_gate_obj=globals().get("birdeye_gate"),
                base_url=globals().get("BIRDEYE_BASE_URL"),
                chunk_size=int(os.getenv("BIRDEYE_CHUNK_SIZE", "50")),
            )
        except Exception:
            if logger:
                try:
                    logger.debug("apply_batched_birdeye_prices failed in enrich_tokens_with_price_change", exc_info=True)
                except Exception:
                    pass

        # Collect addresses still missing a usable positive price (use normalized addresses if present)
        remaining_addrs: List[str] = []
        addr_map: Dict[str, Dict[str, Any]] = {}
        for t in tokens:
            try:
                raw = (t.get("address") or t.get("token_address") or "") or ""
                raw = str(raw).strip()
                norm = _extract_base58_address(raw) or raw
                # ensure canonical assignment so mapping keys match those we used above
                t["address"] = norm
                t["token_address"] = norm
                t.setdefault("_raw_address", raw)
                try:
                    Pubkey.from_string(norm)
                except Exception:
                    continue
                addr_map[norm] = t
                p = t.get("price")
                if p is None or (isinstance(p, (int, float)) and float(p) <= 0):
                    remaining_addrs.append(norm)
            except Exception:
                continue

        # Fallback multi call on remaining_addrs (normalized)
        if remaining_addrs:
            try:
                chunk_size = int(os.getenv("BIRDEYE_CHUNK_SIZE", "50"))
                multi = await birdeye_multi_price(
                    session=session,
                    addresses=remaining_addrs,
                    api_key=api_key,
                    birdeye_gate=globals().get("birdeye_gate"),
                    base_url=globals().get("BIRDEYE_BASE_URL"),
                    chunk_size=chunk_size,
                )
            except Exception:
                multi = {}
                if logger:
                    try:
                        logger.debug("birdeye_multi_price failed in enrich_tokens_with_price_change", exc_info=True)
                    except Exception:
                        pass

            if isinstance(multi, dict):
                for addr_key, payload in list(multi.items()):
                    try:
                        # addr_key might be normalized or the raw original; map both
                        tok = addr_map.get(addr_key)
                        if not tok:
                            # try mapping by raw address value
                            for k, v in addr_map.items():
                                if v.get("_raw_address") == addr_key:
                                    tok = v
                                    break
                        if not tok or not payload:
                            continue
                        if isinstance(payload, dict):
                            p = payload.get("price") or payload.get("value") or payload.get("priceUsd") or payload.get("price_usd")
                            pf = _parse_float_like(p)
                            if pf is not None and pf > 0:
                                tok["price"] = float(pf)
                            # price changes
                            try:
                                v1 = _parse_float_like(payload.get("priceChange1h") or payload.get("price_change_1h") or payload.get("v1h") or payload.get("h1"))
                                if v1 is not None: tok["price_change_1h"] = float(v1)
                            except Exception:
                                pass
                            try:
                                v6 = _parse_float_like(payload.get("priceChange6h") or payload.get("price_change_6h") or payload.get("v6h") or payload.get("h6"))
                                if v6 is not None: tok["price_change_6h"] = float(v6)
                            except Exception:
                                pass
                            try:
                                v24 = _parse_float_like(payload.get("priceChange24h") or payload.get("price_change_24h") or payload.get("v24h") or payload.get("h24"))
                                if v24 is not None: tok["price_change_24h"] = float(v24)
                            except Exception:
                                pass
                    except Exception:
                        continue

        return tokens

    except Exception:
        if logger:
            try:
                logger.debug("enrich_tokens_with_price_change top-level failure", exc_info=True)
            except Exception:
                pass
        return tokens or []

    finally:
        if own_session and session is not None:
            try:
                await session.close()
            except Exception:
                pass
            
# Tunables
BIRDEYE_TINY_PROBE_LIMIT = int(os.getenv("BIRDEYE_TINY_PROBE_LIMIT", "20"))
BIRDEYE_PAGE_LIMIT = int(os.getenv("BIRDEYE_PAGE_LIMIT", "100"))  # API supports up to 100
BIRDEYE_MAX_PAGES_PRIMARY = int(os.getenv("BIRDEYE_MAX_PAGES_PRIMARY", "6"))
BIRDEYE_MAX_PAGES_REFILL = int(os.getenv("BIRDEYE_MAX_PAGES_REFILL", "4"))

# Preferred sort keys for v3 token list
_BIRDEYE_SORT_KEYS = ("liquidity", "volume_24h_usd", None)

async def _ipv_session(force_ipv4: bool = False, force_ipv6: bool = False) -> aiohttp.ClientSession:
    """Short-lived session using IPv4/IPv6-only connector when required."""
    fam = 0
    if force_ipv4:
        fam = socket.AF_INET
    elif force_ipv6:
        fam = socket.AF_INET6

    connector = aiohttp.TCPConnector(family=fam or 0, limit=30, ttl_dns_cache=300, enable_cleanup_closed=True)
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)
    return aiohttp.ClientSession(connector=connector, timeout=timeout)

async def fetch_birdeye_tokens(
    session: aiohttp.ClientSession,
    solana_client: AsyncClient,
    max_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Bulk discovery using GET {BIRDEYE_BASE_URL}/defi/v3/token/list
    Supports offset/limit paging and query params: sort_by, sort_type, min_liquidity,
    min_volume_24h_usd, min_recent_listing_time, max_recent_listing_time.
    """
    if _should_stop():
        return []

    # feature gating (quick exits)
    try:
        if 'FORCE_DISABLE_BIRDEYE' in globals() and FORCE_DISABLE_BIRDEYE:
            logger.info("Birdeye disabled via FORCE_DISABLE_BIRDEYE")
            return []
        if 'is_enabled_birdeye' in globals():
            try:
                enabled = is_enabled_birdeye()
            except TypeError:
                enabled = BIRDEYE_ENABLE_ENV
            if not enabled:
                logger.info("Birdeye disabled by config/env; skipping.")
                return []
    except Exception:
        pass

    global _BIRDEYE_COOLDOWN_UNTIL, _BIRDEYE_LAST_BACKOFF

    # Prefer the shared gate's monotonic down-until if available
    if 'birdeye_gate' in globals() and birdeye_gate is not None:
        try:
            down_mon = float(birdeye_gate.stats().get("down_until_monotonic", 0.0))
            if time.monotonic() < down_mon:
                logger.warning("Birdeye gate indicates down until %.3f (monotonic); skipping fetch.", down_mon)
                return []
        except Exception:
            # If gate.stats failed, fall back to module-local wall-clock cooldown check
            now = time.time()
            if now < _BIRDEYE_COOLDOWN_UNTIL:
                logger.warning("Birdeye on global cooldown (module-local fallback); skipping fetch.")
                return []
    else:
        # legacy behavior (module-local wall-clock cooldown)
        now = time.time()
        if now < _BIRDEYE_COOLDOWN_UNTIL:
            logger.warning("Birdeye on global cooldown; skipping fetch.")
            return []

    api_key = os.getenv("BIRDEYE_API_KEY", "")
    if not api_key:
        logger.warning("BIRDEYE_API_KEY not set; skipping Birdeye.")
        return []

    # Defensive top-level wrapper so unexpected errors don't return None to callers
    try:
        cfg_max = _bird_max()
        target = max(1, int(max_tokens if max_tokens is not None else cfg_max))

        page_limit_cfg = int(_disc("birdeye_page_limit", BIRDEYE_PAGE_LIMIT))
        page_limit = min(max(1, page_limit_cfg), 100, target)

        max_pages_cfg = int(_disc("birdeye_max_pages", BIRDEYE_MAX_PAGES_PRIMARY))
        needed_pages = max(1, math.ceil(target / page_limit))
        max_pages = min(max_pages_cfg, needed_pages)

        success_floor = min(max(12, page_limit), target)

        base_list_url = f"{BIRDEYE_BASE_URL.rstrip('/')}/defi/v3/token/list"
        logger.debug("Birdeye: base_list_url=%s (using v3 token list) BIRDEYE_MODE=%s", base_list_url, BIRDEYE_MODE)
        headers = _birdeye_headers_public(api_key)
        timeout = aiohttp.ClientTimeout(total=20, sock_connect=6, sock_read=10)

        # ensure now is defined before using cache (fix UnboundLocalError if present)
        now = time.time()

        # 30s in-memory cache
        _cache = getattr(fetch_birdeye_tokens, "_CACHE", {"until": 0.0, "items": []})
        if now < float(_cache.get("until", 0)) and _cache.get("items"):
            cached = _cache["items"]
            logger.debug("Birdeye: served %d items from cache", len(cached))
            return cached[:target]

        # small jitter
        await asyncio.sleep(0.08 + (time.time() % 0.12))

        async def _do_get(url: str, params: dict, family: str = "normal"):
            if family == "normal":
                return await session.get(url, headers=headers, params=params, timeout=timeout)
            tmp = await _ipv_session(force_ipv6=(family == "ipv6"), force_ipv4=(family == "ipv4"))
            try:
                resp = await tmp.get(url, headers=headers, params=params, timeout=timeout)
                async def _closer():
                    await tmp.close()
                setattr(resp, "close_parent", _closer)
                return resp
            except Exception:
                await tmp.close()
                raise

        # tiny probe uses the closure _do_get and base_list_url directly
        async def _tiny_probe(
            limit: Optional[int] = None,
            api_key_local: Optional[str] = None,
            do_get_local: Optional[Callable[..., Any]] = None,
            base_list_url_local: Optional[str] = None,
            birdeye_gate_local: Optional[Any] = None,
        ) -> List[Dict[str, Any]]:
            """
            Tiny probe to seed token discovery. Accepts dependency-injection parameters
            so it can be called safely from tests or the surrounding scope.
            """
            try:
                api_key_val = api_key_local if api_key_local is not None else api_key
                _do_get_fn = do_get_local if do_get_local is not None else _do_get
                base_list_url_val = base_list_url_local if base_list_url_local is not None else base_list_url
                gate = birdeye_gate_local if birdeye_gate_local is not None else globals().get("birdeye_gate", None)

                # Determine configured tiny-probe limit (priority order):
                if limit is None:
                    try:
                        cfg = globals().get("_BIRDEYE_TINY_PROBE_LIMIT", None)
                        if cfg is None:
                            cfg = int(globals().get("BIRDEYE_TINY_PROBE_LIMIT", 0)) or None
                    except Exception:
                        cfg = None
                    if cfg is None:
                        cfg = 20
                    try:
                        probe_limit = int(cfg)
                    except Exception:
                        probe_limit = 5
                else:
                    probe_limit = int(limit)
                probe_limit = max(1, int(probe_limit))
                logger.debug("Birdeye tiny_probe using limit=%s", probe_limit)

                # defensive fallbacks if closures/globals weren't present
                if _do_get_fn is None:
                    raise RuntimeError("_do_get helper not available for tiny_probe; pass it as do_get_local")

                if not base_list_url_val:
                    raise RuntimeError("base_list_url not available for tiny_probe; pass it as base_list_url_local")

                params = {"chain": "solana", "limit": int(probe_limit), "offset": 0}
                family_order = ("ipv4", "ipv6") if bool(globals().get("_FORCE_IPV4", False)) else ("ipv6", "ipv4")
                redact = globals().get("_redact_key", lambda k: "<redacted>")

                for fam in family_order:
                    resp_ctx = None
                    try:
                        if gate is not None:
                            gate_ctx = await gate.request()
                            async with gate_ctx:
                                try:
                                    resp_ctx = await _do_get_fn(base_list_url_val, params, family=fam)
                                except TypeError:
                                    resp_ctx = await _do_get_fn(base_list_url_val, params)
                        else:
                            resp_ctx = await _do_get_fn(base_list_url_val, params, family=fam)

                        async with resp_ctx as resp:
                            text = await resp.text()
                            logger.debug(
                                "Birdeye tiny_probe HTTP %s params=%s family=%s body=%s",
                                resp.status, params, fam, (text or "")[:2000]
                            )

                            if resp.status == 429:
                                wait = None
                                try:
                                    wait = parse_retry_after_seconds(resp.headers)
                                except Exception:
                                    wait = None
                                wait = wait or float(globals().get("_BIRDEYE_DOWN_BACKOFF_DEFAULT", _BIRDEYE_DOWN_BACKOFF_DEFAULT))
                                wait = max(1.0, min(300.0, float(wait)))
                                if gate is not None:
                                    try:
                                        await gate.mark_down(wait)
                                    except Exception:
                                        logger.debug("birdeye_gate.mark_down failed", exc_info=True)
                                logger.warning("Birdeye tiny_probe 429; backing off %.1fs", wait)
                                return []
                            if resp.status in (401, 403):
                                logger.warning(
                                    "Birdeye tiny_probe auth error (%s) — key redacted=%s",
                                    resp.status, redact(api_key_val)
                                )
                                return []
                            if resp.status != 200:
                                logger.debug("Birdeye tiny_probe non-200 status %s: %s", resp.status, (text or "")[:300])
                                return []

                            # parse JSON robustly
                            try:
                                data = json.loads(text)
                            except Exception:
                                try:
                                    data = await resp.json(content_type=None)
                                except Exception:
                                    logger.debug("Birdeye tiny_probe JSON parse failed", exc_info=True)
                                    return []

                            payload = (data.get("data") if isinstance(data, dict) else data) or data
                            if isinstance(payload, dict):
                                raw = payload.get("items") or payload.get("tokens") or []
                                if isinstance(raw, list):
                                    return [it for it in raw if isinstance(it, dict)]
                                return []
                            elif isinstance(payload, list):
                                return [it for it in payload if isinstance(it, dict)]
                            else:
                                return []
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.debug("Birdeye tiny_probe transient error (family=%s): %s", fam, e, exc_info=True)
                        continue
                    except Exception:
                        logger.exception("Birdeye tiny_probe unexpected error (family=%s)", fam)
                        continue
                    finally:
                        closer = getattr(resp_ctx, "close_parent", None)
                        if callable(closer):
                            try:
                                await closer()
                            except Exception:
                                pass
                return []
            except Exception:
                logger.exception("Birdeye tiny_probe top-level unexpected error")
                return []

        async def _force_fill_public(start_offset: int, need: int) -> int:
            added = 0
            off = start_offset
            for _ in range(BIRDEYE_MAX_PAGES_REFILL):
                if _should_stop():
                    break
                try:
                    if birdeye_gate is not None:
                        gate_ctx = await birdeye_gate.request()
                        async with gate_ctx:
                            async with session.get(base_list_url, headers=headers, params={"chain": "solana", "limit": page_limit, "offset": off}, timeout=timeout) as r:
                                if r.status != 200:
                                    break
                                data = await r.json(content_type=None)
                    else:
                        async with session.get(base_list_url, headers=headers, params={"chain": "solana", "limit": page_limit, "offset": off}, timeout=timeout) as r:
                            if r.status != 200:
                                break
                            data = await r.json(content_type=None)

                    payload = data.get("data") or data
                    li = payload.get("items") if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
                    if li:
                        added += len(li)
                        off += page_limit
                        # extend local items via closure
                        items.extend(li)
                    else:
                        break
                except Exception:
                    break
                await asyncio.sleep(1.0)
            return added

        items: List[Dict[str, Any]] = []

        # seed with tiny probe (inject the closure helpers so tiny_probe uses the same connectors)
        try:
            seed = await _tiny_probe(
                limit=None,
                api_key_local=api_key,
                do_get_local=_do_get,
                base_list_url_local=base_list_url,
                birdeye_gate_local=globals().get("birdeye_gate", None),
            )
            if seed:
                items.extend(seed)
                logger.info("Birdeye tiny-probe seeded %d items", len(seed))
                await asyncio.sleep(0.6)
        except Exception:
            logger.debug("Birdeye tiny-probe failed", exc_info=True)

        offset = 0
        for _page in range(max_pages):
            if _should_stop() or len(items) >= target:
                break

            got_page = False
            for sort_by in _BIRDEYE_SORT_KEYS:
                params = {"chain": "solana", "limit": page_limit, "offset": offset}
                if sort_by:
                    params["sort_by"] = sort_by
                    params["sort_type"] = "desc"

                # apply configured minima
                try:
                    min_liq = float(_disc("min_liquidity", MIN_LIQUIDITY_USD) or MIN_LIQUIDITY_USD)
                    if min_liq > 0:
                        params["min_liquidity"] = int(min_liq)
                except Exception:
                    pass
                try:
                    min_vol = float(_disc("min_volume_24h_usd", MIN_VOLUME24_USD) or MIN_VOLUME24_USD)
                    if min_vol > 0:
                        params["min_volume_24h_usd"] = int(min_vol)
                except Exception:
                    pass

                page_captured = False
                family_order = ("normal", "ipv4", "ipv6") if _FORCE_IPV4 else ("normal", "ipv6", "ipv4")
                for fam in family_order:
                    resp_ctx = None
                    try:
                        # Acquire shared gate if available before making the HTTP request
                        if birdeye_gate is not None:
                            gate_ctx = await birdeye_gate.request()
                            async with gate_ctx:
                                try:
                                    resp_ctx = await _do_get(base_list_url, params, family=fam)
                                except TypeError:
                                    resp_ctx = await _do_get(base_list_url, params)
                        else:
                            resp_ctx = await _do_get(base_list_url, params, family=fam)

                        async with resp_ctx as resp:
                            if resp.status == 429:
                                wait = _rl_from_headers_safe(resp.headers)
                                # prefer to mark shared gate down if present
                                if birdeye_gate is not None:
                                    try:
                                        await birdeye_gate.mark_down(max(1.0, wait))
                                    except Exception:
                                        logger.debug("birdeye_gate.mark_down failed", exc_info=True)
                                else:
                                    _BIRDEYE_LAST_BACKOFF = min(25, max(8, (_BIRDEYE_LAST_BACKOFF * 2) if _BIRDEYE_LAST_BACKOFF else 8))
                                    _BIRDEYE_COOLDOWN_UNTIL = time.time() + max(_BIRDEYE_LAST_BACKOFF, wait)
                                logger.warning("Birdeye 429; backing off %.1fs", max(1.0, wait))
                                await asyncio.sleep(min(1.5, wait))
                                continue
                            if resp.status != 200:
                                continue
                            data = await resp.json(content_type=None)
                            payload = data.get("data") or data
                            page_items = []
                            if isinstance(payload, dict):
                                page_items = payload.get("items") or payload.get("tokens") or []
                            elif isinstance(payload, list):
                                page_items = payload
                            if isinstance(page_items, list) and page_items:
                                items.extend(page_items)
                                page_captured = True
                                got_page = True
                            break
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.warning("Birdeye request failed via %s: %s", fam, e)
                        continue
                    finally:
                        closer = getattr(resp_ctx, "close_parent", None)
                        if callable(closer):
                            try:
                                await closer()
                            except Exception:
                                pass

                if page_captured:
                    break

            if not got_page:
                break

            offset += page_limit
            await asyncio.sleep(0.9)

        # refill if below floor
        if len(items) < success_floor:
            added = await _force_fill_public(offset, success_floor)
            logger.info("Birdeye refill added %d items; total=%d (floor=%d)", added, len(items), success_floor)

        if not items:
            logger.warning("Birdeye returned no items this cycle")
            return []

        # --- Apply batched price enrichment to raw items (reduces per-token GET fan-out) ---
        try:
            # items are raw token dicts from Birdeye; many contain "address"
            await apply_batched_birdeye_prices(
                session=session,
                tokens=items,
                api_key=api_key,
                birdeye_gate_obj=globals().get("birdeye_gate", None),
                base_url=BIRDEYE_BASE_URL,
                chunk_size=int(_disc("birdeye_chunk_size", 50))
            )
        except Exception:
            logger.debug("apply_batched_birdeye_prices failed; continuing without batched prices", exc_info=True)

        # Normalize / dedupe and optionally request precise creation_time when needed
        seen: set[str] = set()
        tokens: List[Dict[str, Any]] = []
        for it in items:
            if _should_stop():
                break
            try:
                token_address = (it.get("address") or it.get("tokenAddress") or it.get("token_address") or "").strip()
                if not token_address:
                    continue
                try:
                    Pubkey.from_string(token_address)
                except Exception:
                    continue

                rc = await _annotate_rugcheck_fields(token_address, session)
                if rc.get("drop"):
                    continue
                if token_address in seen:
                    continue
                seen.add(token_address)

                symbol = it.get("symbol") or it.get("tokenSymbol") or "UNKNOWN"
                name = it.get("name") or symbol
                vol24 = it.get("volume_24h_usd") or it.get("v24hUSD") or it.get("volume24hUSD") or 0
                liq = it.get("liquidity") or it.get("liquidity_usd") or 0
                mc = it.get("market_cap") or it.get("mc") or 0

                # Prefer any price already set by batched enrichment, else bulk field
                raw_price = it.get("price") if it.get("price") is not None else (it.get("priceUsd") if it.get("priceUsd") is not None else None)
                p_val = _parse_float_like(raw_price)
                fb_dict = FALLBACK_PRICES.get(token_address, {}) if isinstance(FALLBACK_PRICES, dict) else {}
                fb_price = _parse_float_like(fb_dict.get("price")) if fb_dict else None

                if p_val is not None and p_val > 0:
                    final_price = float(p_val)
                elif fb_price is not None and fb_price > 0:
                    final_price = float(fb_price)
                else:
                    final_price = None

                # Prefer recent_listing_time from v3 bulk (cheap)
                created_ts = 0
                recent_listing = it.get("recent_listing_time") or it.get("recentListingTime") or None
                if recent_listing:
                    try:
                        cr = float(recent_listing)
                        if cr > 10**12:
                            cr = cr // 1000
                        created_ts = int(cr)
                    except Exception:
                        created_ts = 0

                # Only call token_creation_info when we really need precise timestamp and we're pro
                if created_ts == 0:
                    try:
                        ct_dt = None
                        # fetch_birdeye_creation_time already gates/backs off internally
                        ct_dt = await fetch_birdeye_creation_time(token_address, session)
                    except Exception:
                        ct_dt = None
                    if ct_dt:
                        created_ts = int(ct_dt.timestamp())

                tokens.append({
                    "address": token_address,
                    "symbol": symbol,
                    "name": name,
                    "volume_24h": _bd_num(vol24),
                    "liquidity": _bd_num(liq),
                    "market_cap": _bd_num(mc),
                    "creation_timestamp": created_ts,
                    "timestamp": int(time.time()),
                    "categories": ["no_creation_time"] if created_ts == 0 else [],
                    "links": (it.get("extensions") or {}).get("socials", []),
                    "score": None,
                    "price": final_price,
                    "price_change_1h": _parse_float_like(it.get("price_change_1h") or it.get("v1h") or None),
                    "price_change_6h": _parse_float_like(it.get("price_change_6h") or it.get("v6h") or None),
                    "price_change_24h": _parse_float_like(it.get("price_change_24h") or it.get("v24h") or None),
                    "source": "birdeye",
                    "safety": rc.get("safety", "unknown"),
                    "dangerous": bool(rc.get("dangerous", False)),
                })
            except Exception:
                continue

        if tokens:
            _BIRDEYE_LAST_BACKOFF = 0
            _BIRDEYE_COOLDOWN_UNTIL = 0
            fetch_birdeye_tokens._CACHE = {"until": time.time() + 30.0, "items": list(tokens)}

        if len(tokens) >= success_floor:
            logger.info("Birdeye PASS: fetched %d ≥ %d tokens.", len(tokens), success_floor)
        else:
            logger.warning("Birdeye SOFT-OK: fetched %d < %d tokens (rate/limits likely).", len(tokens), success_floor)

        return tokens[:target]

    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("fetch_birdeye_tokens top-level failure")
        try:
            fetch_birdeye_tokens._CACHE = {"until": time.time() + 5.0, "items": []}
        except Exception:
            pass
        return []

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def fetch_birdeye_creation_time(
    token_address: str,
    session: aiohttp.ClientSession
) -> Optional[datetime]:
    """
    Creation time lookup.

    Behavior:
      - If running in pro mode (BIRDEYE_PRO True), attempt the precise:
          GET {BIRDEYE_BASE_URL}/defi/token_creation_info?address=<token>
        (may be private/pro-only).
      - Otherwise (public/starter), use the price endpoint fallback:
          GET {BIRDEYE_BASE_URL}/defi/price?address=<token>
        Inspect the returned payload for fields such as updateUnixTime/update_time
        or creation_time-like keys and only accept values that pass sanity checks.

    This avoids calling a non-existent endpoint (404) on Starter/public plan.
    """
    global _BIRDEYE_COOLDOWN_UNTIL, _BIRDEYE_LAST_BACKOFF

    # Prefer the shared gate's monotonic down-until if available
    if "birdeye_gate" in globals() and birdeye_gate is not None:
        try:
            down_mon = float(birdeye_gate.stats().get("down_until_monotonic", 0.0))
            if time.monotonic() < down_mon:
                # Early-exit quietly when gate says down
                return None
        except Exception:
            # fallback to module-local wall-clock check
            now = time.time()
            if now < _BIRDEYE_COOLDOWN_UNTIL:
                return None
    else:
        # legacy behavior (module-local wall-clock cooldown)
        now = time.time()
        if now < _BIRDEYE_COOLDOWN_UNTIL:
            return None

    api_key = os.getenv("BIRDEYE_API_KEY", "")
    if not api_key or token_address in WHITELISTED_TOKENS:
        return None

    try:
        Pubkey.from_string(token_address)
    except Exception:
        return None

    # cached check
    cached = None
    try:
        cached = await get_cached_creation_time(token_address)
    except Exception:
        cached = None
    if cached:
        return cached

    headers = _birdeye_headers_public(api_key)
    timeout = aiohttp.ClientTimeout(total=12, sock_connect=6, sock_read=8)

    # endpoints
    creation_url = f"{BIRDEYE_BASE_URL.rstrip('/')}/defi/token_creation_info"
    price_url = f"{BIRDEYE_BASE_URL.rstrip('/')}/defi/price"
    # multi_price endpoint available if you want to batch price queries
    multi_price_url = f"{BIRDEYE_BASE_URL.rstrip('/')}/defi/multi_price"

    data_obj: Optional[Dict[str, Any]] = None

    # Helper to handle 429 cooperative backoff (shared gate preferred)
    def _handle_429_and_backoff(resp_headers: Dict[str, Any]) -> None:
        # modify module-level fallback variables when shared gate is not available
        global _BIRDEYE_LAST_BACKOFF, _BIRDEYE_COOLDOWN_UNTIL
        # Prefer the shared parser if present, else fallback
        try:
            if "parse_retry_after_seconds" in globals() and callable(globals().get("parse_retry_after_seconds")):
                wait = parse_retry_after_seconds(resp_headers) or 0.0
            else:
                wait = _rl_from_headers_safe(resp_headers)
        except Exception:
            wait = _rl_from_headers_safe(resp_headers)
        try:
            wait = float(wait) if (wait is not None) else 0.0
        except Exception:
            wait = 0.0
        wait = max(1.0, min(300.0, wait))

        # If shared gate is not present, update module-local fallback values
        if not ("birdeye_gate" in globals() and birdeye_gate is not None):
            try:
                _BIRDEYE_LAST_BACKOFF = min(25, max(8, (_BIRDEYE_LAST_BACKOFF * 2) if _BIRDEYE_LAST_BACKOFF else 8))
                _BIRDEYE_COOLDOWN_UNTIL = time.time() + max(_BIRDEYE_LAST_BACKOFF, wait)
                logger.warning("Birdeye module-local backoff set for %.1fs", _BIRDEYE_COOLDOWN_UNTIL - time.time())
            except Exception:
                logger.debug("Failed to set module-local backoff", exc_info=True)

    # 1) Primary precise endpoint: only attempt if pro mode (avoid 404 on Starter)
    if BIRDEYE_PRO:
        try:
            # Acquire shared gate if available
            if "birdeye_gate" in globals() and birdeye_gate is not None:
                gate_ctx = await birdeye_gate.request()
                async with gate_ctx:
                    async with session.get(creation_url, headers=headers, params={"address": token_address}, timeout=timeout) as r:
                        if r.status == 429:
                            # parse & set cooperative backoff
                            wait = (parse_retry_after_seconds(r.headers)
                                    if "parse_retry_after_seconds" in globals() and callable(globals().get("parse_retry_after_seconds"))
                                    else _rl_from_headers_safe(r.headers))
                            wait = max(1.0, min(300.0, float(wait or 1.0)))
                            try:
                                await birdeye_gate.mark_down(wait)
                            except Exception:
                                logger.debug("birdeye_gate.mark_down failed", exc_info=True)
                            # also set module-local fallback
                            _handle_429_and_backoff(r.headers)
                            # behave like before: raise to be caught and let fallback run
                            raise aiohttp.ClientError("birdeye-429")
                        if r.status == 200:
                            data_obj = await r.json(content_type=None)
                        elif r.status == 404:
                            # treat as "no data for this token"
                            if callable(cache_creation_time):
                                try:
                                    await cache_creation_time(token_address, None)
                                except Exception:
                                    pass
                            return None
            else:
                # No shared gate: direct request with legacy handling
                async with session.get(creation_url, headers=headers, params={"address": token_address}, timeout=timeout) as r:
                    if r.status == 429:
                        _handle_429_and_backoff(r.headers)
                        raise aiohttp.ClientError("birdeye-429")
                    if r.status == 200:
                        data_obj = await r.json(content_type=None)
                    elif r.status == 404:
                        if callable(cache_creation_time):
                            try:
                                await cache_creation_time(token_address, None)
                            except Exception:
                                pass
                        return None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            # treat as no data and allow fallback to price endpoint
            data_obj = None

    # 2) Fallback: price endpoint (Starter/public supported)
    #    Inspect returned payload for potential timestamp-like fields that we can
    #    use as a best-effort creation / update time, with sanity checks.
    if not isinstance(data_obj, dict):
        try:
            if "birdeye_gate" in globals() and birdeye_gate is not None:
                gate_ctx = await birdeye_gate.request()
                async with gate_ctx:
                    async with session.get(price_url, headers=headers, params={"address": token_address}, timeout=timeout) as r2:
                        if r2.status == 429:
                            wait = (parse_retry_after_seconds(r2.headers)
                                    if "parse_retry_after_seconds" in globals() and callable(globals().get("parse_retry_after_seconds"))
                                    else _rl_from_headers_safe(r2.headers))
                            wait = max(1.0, min(300.0, float(wait or 1.0)))
                            try:
                                await birdeye_gate.mark_down(wait)
                            except Exception:
                                logger.debug("birdeye_gate.mark_down failed", exc_info=True)
                            _handle_429_and_backoff(r2.headers)
                            raise aiohttp.ClientError("birdeye-429")
                        if r2.status == 200:
                            data_obj = await r2.json(content_type=None)
                        elif r2.status == 404:
                            return None
            else:
                async with session.get(price_url, headers=headers, params={"address": token_address}, timeout=timeout) as r2:
                    if r2.status == 429:
                        _handle_429_and_backoff(r2.headers)
                        raise aiohttp.ClientError("birdeye-429")
                    if r2.status == 200:
                        data_obj = await r2.json(content_type=None)
                    elif r2.status == 404:
                        return None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            data_obj = None

    if not isinstance(data_obj, dict):
        return None

    d = data_obj.get("data") or data_obj

    created_ms: Optional[float] = None

    # try canonical creation fields first (if present)
    for key in ("creation_time", "created_timestamp", "createdTimestamp", "createdAt", "mintTime", "created_time"):
        try:
            if isinstance(d, dict) and d.get(key) is not None:
                created_ms = float(d.get(key))
                break
        except Exception:
            continue

    # fallback: inspect possible "update" or unix time fields in price payload
    if created_ms is None:
        for key in ("updateUnixTime", "update_unix_time", "update_time", "updateTimestamp", "updateUnix"):
            try:
                v = None
                if isinstance(d, dict):
                    v = d.get(key)
                if v is None and isinstance(d.get("price"), dict):
                    v = d["price"].get(key)
                if v is not None:
                    created_ms = float(v)
                    break
            except Exception:
                continue

    if created_ms is None:
        return None

    try:
        if created_ms < 10_000_000_000:
            created_ms = created_ms * 1000.0
    except Exception:
        return None

    now_ms = time.time() * 1000.0
    solana_launch_ms = 1581465600000.0
    if not (solana_launch_ms <= created_ms <= now_ms):
        return None

    dt = datetime.fromtimestamp(created_ms / 1000.0, tz=timezone.utc)
    try:
        await cache_creation_time(token_address, dt)
    except Exception:
        pass

    # reset legacy module-level backoff state for compatibility when a successful lookup occurs
    try:
        _BIRDEYE_LAST_BACKOFF = 0
        _BIRDEYE_COOLDOWN_UNTIL = 0
    except Exception:
        pass

    return dt

# ---------------------------------------------------------------------
# Dexscreener creation time
# ---------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def fetch_dexscreener_creation_time(
    token_address: str,
    session: aiohttp.ClientSession,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[datetime]:
    """
    Robust Dexscreener creation-time fetch using the new /token/v1/{token}/solana endpoint,
    with an IPv4-only fallback on connector/DNS failures. Returns a UTC datetime or None.
    """
    if _should_stop():
        return None
    if token_address in WHITELISTED_TOKENS:
        return None
    try:
        Pubkey.from_string(token_address)
    except Exception:
        return None

    cached = await get_cached_creation_time(token_address)
    if cached:
        return cached

    # Browser-like headers (safe defaults)
    try:
        DEFAULT_BROWSER_UA  # type: ignore[name-defined]
    except NameError:
        DEFAULT_BROWSER_UA = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    try:
        BROWSERISH_DEX_HEADERS  # type: ignore[name-defined]
    except NameError:
        BROWSERISH_DEX_HEADERS = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Origin": "https://dexscreener.com",
            "Referer": "https://dexscreener.com/",
            "Sec-CH-UA": '"Chromium";v="131", "Google Chrome";v="131", "Not=A?Brand";v="99"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": DEFAULT_BROWSER_UA,
        }

    hdrs = dict(BROWSERISH_DEX_HEADERS)
    if headers:
        hdrs.update(headers)

    timeout = aiohttp.ClientTimeout(total=12, sock_connect=6, sock_read=8)

    # New 2025 endpoint
    url = f"https://api.dexscreener.com/token/v1/{token_address}/solana"

    async def _do_request(sess: aiohttp.ClientSession) -> Optional[dict]:
        try:
            async with sess.get(url, headers=hdrs, timeout=timeout) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = 10.0
                    if retry_after:
                        try:
                            sleep_s = max(5.0, float(retry_after))
                        except Exception:
                            pass
                    logger.warning("DexScreener 429 — sleeping %.1fs", sleep_s)
                    await asyncio.sleep(sleep_s)
                    raise aiohttp.ClientError("rate-limited")

                if resp.status == 404:
                    logger.debug("DexScreener 404: no pairs for %s", token_address)
                    return None

                if resp.status != 200:
                    body = (await resp.text())[:500]
                    logger.warning("DexScreener HTTP %s for %s: %s", resp.status, url, body)
                    return None

                try:
                    return await resp.json(content_type=None)
                except Exception as je:
                    logger.debug("DexScreener JSON decode failed for %s: %s", token_address, je)
                    return None

        except asyncio.TimeoutError:
            logger.warning("DexScreener timeout for %s", token_address)
            raise  # let tenacity handle retry
        except aiohttp.ClientError:
            # Propagate the client error to the caller so caller can inspect it for DNS/connect issues
            raise

    data: Optional[dict] = None
    last_exc: Optional[BaseException] = None

    # Primary attempt (capture exceptions so we can decide on IPv4 fallback)
    try:
        data = await _do_request(session)
    except Exception as e:
        last_exc = e
        logger.debug("DexScreener primary request error for %s: %s", token_address, e)

    # If we failed and it looks like a DNS/connector/getaddrinfo issue, try IPv4-only session
    if data is None and last_exc is not None:
        emsg = str(last_exc).lower()
        is_dns_err = False
        # Common indicators: getaddrinfo failed, Temporary failure in name resolution, underlying OSError
        if "getaddrinfo" in emsg or "temporary failure in name resolution" in emsg or isinstance(getattr(last_exc, "__cause__", None), OSError):
            is_dns_err = True

        if is_dns_err:
            dex_sess = None
            try:
                try:
                    dex_sess, _ = await _make_dex_session()  # type: ignore[name-defined]
                except Exception as e:
                    logger.debug("Failed to create IPv4 dex session: %s", e)
                    dex_sess = None
                if dex_sess is not None:
                    try:
                        data = await _do_request(dex_sess)
                    except Exception as e2:
                        logger.debug("DexScreener IPv4 fallback failed for %s: %s", token_address, e2)
                    finally:
                        try:
                            await dex_sess.close()
                        except Exception:
                            pass
            except Exception as e:
                logger.debug("Unexpected error during IPv4 fallback for %s: %s", token_address, e)

    # If still no usable data, give up
    if not data or not isinstance(data, dict):
        return None

    # New response format usually: {"pairs": [...]}
    pairs = data.get("pairs") or data.get("data") or []
    if not pairs or not isinstance(pairs, list):
        return None

    earliest_ms: Optional[float] = None
    now_ms = time.time() * 1000.0
    solana_launch_ms = 1581465600000.0  # Feb 12 2020

    for p in pairs:
        if _should_stop():
            return None

        # accept chainId or chain (and be tolerant of casing)
        chain_val = p.get("chainId") or p.get("chain") or ""
        try:
            if isinstance(chain_val, str):
                chain_check = chain_val.strip().lower()
            else:
                chain_check = str(chain_val).strip().lower()
        except Exception:
            chain_check = ""

        if "sol" not in chain_check and "solana" not in chain_check:
            continue

        # pick created fields (pairCreatedAt, createdAt, createdTime, created_timestamp, created_at, mintTime)
        created = None
        for key in ("pairCreatedAt", "createdAt", "createdTime", "created_timestamp", "created_at", "mintTime"):
            if key in p and p.get(key) is not None:
                created = p.get(key)
                break
            # sometimes nested in sub-objects
            sub = p.get("data") or p.get("score") or {}
            if isinstance(sub, dict) and key in sub and sub.get(key) is not None:
                created = sub.get(key)
                break

        if not created:
            continue

        try:
            created_ms = float(created)
            # If it's a small integer (seconds) convert to ms
            if created_ms < 10_000_000_000:
                created_ms *= 1000.0
        except Exception:
            # created not parseable
            continue

        if not (solana_launch_ms <= created_ms <= now_ms):
            continue

        if earliest_ms is None or created_ms < earliest_ms:
            earliest_ms = created_ms

    if earliest_ms is None:
        return None

    dt = datetime.fromtimestamp(earliest_ms / 1000.0, tz=timezone.utc)
    try:
        await cache_creation_time(token_address, dt)
    except Exception as e:
        logger.debug("Failed to cache creation time for %s: %s", token_address, e)

    logger.debug("DexScreener creation time for %s → %s", token_address, dt.isoformat())
    return dt


# ---------------------------------------------------------------------
# Unified creation time helper (used by refresh hook)
# ---------------------------------------------------------------------
async def get_latest_tokens(max_items: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch latest tokens from Dexscreener, Birdeye, and Raydium, then combine,
    dedupe by exact on-chain address, and return a normalized list of tokens.
    """
    if _should_stop():
        logger.info("Shutdown signaled; skipping latest token fetch.")
        return []

    connector = aiohttp.TCPConnector(limit=50, enable_cleanup_closed=True)
    session_timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)

    # Try to obtain a dedicated Dex session (may prefer IPv4 + browser UA)
    try:
        dex_session, dex_headers = await _make_dex_session()  # type: ignore[name-defined]
    except Exception:
        dex_session = None
        dex_headers = {
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
            ),
            "Origin": "https://dexscreener.com",
            "Referer": "https://dexscreener.com/",
        }

    # Primary short-lived session used for Birdeye and Raydium
    async with aiohttp.ClientSession(connector=connector, timeout=session_timeout) as session:
        solana_client = AsyncClient(SOLANA_RPC_URL, timeout=DEFAULT_SOLANA_TIMEOUT)

        # Start Birdeye in background so results are available concurrently with Dexscreener
        bird_task: Optional[asyncio.Task] = None
        try:
            bird_cap = _bird_max()
            if bird_cap > 0:
                target = max(1, min(bird_cap, int(max_items or bird_cap)))
                try:
                    bird_task = asyncio.create_task(fetch_birdeye_tokens(session, solana_client, max_tokens=target))
                    _export_birdeye_bg_task(bird_task)                       # <<-- INSERTED HERE
                    logger.debug("Started background Birdeye discovery task (target=%d)", target)
                except Exception:
                    logger.debug("Failed to start bird_task; will fetch inline later", exc_info=True)
                    bird_task = None
                    _export_birdeye_bg_task(None)                            # <<-- INSERTED HERE (failure path)
            else:
                bird_task = None
        except Exception:
            logger.debug("Failed to bootstrap birdeye background task", exc_info=True)
            bird_task = None
            _export_birdeye_bg_task(None)                                    

        try:
            # ---------- Dexscreener FIRST (fast path) ----------
            dex: List[Dict[str, Any]] = []
            if not _should_stop():
                try:
                    queries = _dex_queries()
                    pages, _ = _dex_pages_perpage()
                    # Use dedicated dex_session when available; otherwise reuse 'session'
                    _ds = dex_session if dex_session is not None else session
                    rs = await asyncio.gather(
                        *[fetch_dexscreener_search(_ds, query=q, pages=pages, headers=dex_headers) for q in queries],
                        return_exceptions=True
                    )
                    for r in rs:
                        if isinstance(r, list):
                            dex.extend(r)
                except Exception as e:
                    logger.error("Dexscreener list error: %s", e, exc_info=True)
                    dex = []

            # ---------- Birdeye (if enabled) ----------
            bird: List[Dict[str, Any]] = []
            if not _should_stop():
                try:
                    if bird_task is not None:
                        try:
                            bird = await bird_task
                            logger.debug("Birdeye background task completed with %d items", len(bird or []))
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            # Background task failed — fall back to inline fetch
                            logger.warning("Birdeye background task failed: %s — falling back to inline fetch", e)
                            bird = []
                            try:
                                bird_cap = _bird_max()
                                if bird_cap > 0:
                                    target = max(1, min(bird_cap, int(max_items or bird_cap)))
                                    bird = await fetch_birdeye_tokens(session, solana_client, max_tokens=target)
                            except Exception as e2:
                                logger.error("Birdeye fallback inline list error: %s", e2, exc_info=True)
                                bird = []
                    else:
                        # No background task (e.g. bird_cap==0 or task creation failed): do the legacy inline fetch
                        bird_cap = _bird_max()
                        if bird_cap > 0:
                            target = max(1, min(bird_cap, int(max_items or bird_cap)))
                            bird = await fetch_birdeye_tokens(session, solana_client, max_tokens=target)
                        else:
                            logger.info("Birdeye disabled by config; skipping.")
                            bird = []
                except Exception as e:
                    logger.error("Birdeye list error: %s", e, exc_info=True)
                    bird = []

            # ---------- Raydium LAST + strictly bounded ----------
            ray: List[Dict[str, Any]] = []
            if not _should_stop() and is_enabled_raydium():
                try:
                    parse_cap, _ = _ray_limits(max_items)
                    # hard cap: if Raydium slow, give up quickly to keep UI responsive
                    ray = await asyncio.wait_for(
                        fetch_raydium_tokens(session, solana_client, max_pairs=parse_cap),
                        timeout=6.0,
                    )
                except (asyncio.TimeoutError, MemoryError):
                    logger.warning("Raydium skipped (timeout/size).")
                    ray = []
                except Exception as e:
                    logger.error("Raydium list error: %s", e, exc_info=True)
                    ray = []

            # Close dedicated dex_session if we created one via _make_dex_session
            if dex_session is not None:
                try:
                    await dex_session.close()
                except Exception:
                    pass

            # ---------- Combine & Dedupe (exact on-chain address) ----------
            combined = (ray or []) + (dex or []) + (bird or [])
            if _should_stop():
                return []

            best_by_addr: Dict[str, Dict[str, Any]] = {}
            for t in combined:
                addr = (t.get("address") or t.get("token_address") or "").strip()
                if not addr:
                    continue
                try:
                    # Validate address; skip obviously invalid entries
                    Pubkey.from_string(addr)
                except Exception:
                    continue

                prev = best_by_addr.get(addr)
                if prev is None:
                    t["address"] = addr
                    t["token_address"] = addr
                    best_by_addr[addr] = t
                    continue

                # Prefer the better row (use module _better if present)
                try:
                    if "_better" in globals() and callable(globals().get("_better")):
                        replace = _better(t, prev)
                    else:
                        replace = (_fnum(t, "liquidity"), _fnum(t, "volume_24h"), _mc(t)) > (_fnum(prev, "liquidity"), _fnum(prev, "volume_24h"), _mc(prev))
                    if replace:
                        t["address"] = addr
                        t["token_address"] = addr
                        best_by_addr[addr] = t
                except Exception:
                    # keep existing prev if comparison fails
                    pass

            rows = list(best_by_addr.values())

            # ---------- Normalize & Robust price handling (do NOT coerce to 0) ----------
            out: List[Dict[str, Any]] = []
            for tok in rows:
                addr = tok.get("address")
                tok["liquidity"] = _fnum(tok, "liquidity", 0.0)
                tok["volume_24h"] = _fnum(tok, "volume_24h", 0.0)
                mc_val = _mc(tok)
                tok["market_cap"] = mc_val
                tok["mc"] = mc_val

                # Price resolution: try multiple sources, return None if unknown
                price = _safe_price_from_token_row(tok)
                tok["price"] = float(price) if (price is not None and price > 0) else None
                tok["priceUsd"] = tok.get("price")

                # Price changes
                pc = tok.get("priceChange") or {}
                tok["price_change_1h"] = _fnum(pc.get("h1") or pc.get("1h"), 0.0)
                tok["price_change_6h"] = _fnum(pc.get("h6") or pc.get("6h"), 0.0)
                tok["price_change_24h"] = _fnum(pc.get("h24") or pc.get("24h"), 0.0)

                # Creation timestamp normalization
                created_raw = tok.get("creation_timestamp") or tok.get("createdAt") or tok.get("pairCreatedAt") or 0
                try:
                    created_val = int(created_raw)
                    if created_val > 10**12:
                        created_val //= 1000
                    tok["creation_timestamp"] = int(created_val)
                except Exception:
                    tok["creation_timestamp"] = tok.get("creation_timestamp", 0)

                tok.setdefault("safety", "unknown")
                tok.setdefault("dangerous", False)

                out.append(tok)

            out.sort(
                key=lambda x: (x.get("liquidity", 0.0) * 2.0) + (x.get("volume_24h", 0.0) * 1.5) + (x.get("market_cap", 0.0)),
                reverse=True,
            )

            if max_items:
                out = out[: int(max_items)]

            logger.info("Combined pool: ray=%d, dex=%d, bird=%d → unique_by_addr=%d (returned=%d)",
                        len(ray or []), len(dex or []), len(bird or []), len(best_by_addr), len(out))
            return out

        except Exception as exc:
            logger.exception("get_latest_tokens encountered an unexpected error: %s", exc)
            raise
        finally:
            try:
                await solana_client.close()
            except Exception:
                pass

# --- Back-compat shim -------------------------------------------------
async def collect_candidates(limit: int = 500):
    """
    Backwards-compatible alias for older code paths that still import
    `collect_candidates`. Calls the new aggregator and normalizes keys.
    """
    rows = await get_latest_tokens(max_items=limit)
    for r in rows:
        addr = r.get("address") or r.get("token_address")
        if addr:
            r["address"] = addr
            r.setdefault("token_address", addr)
    return rows

# If module defines __all__, expose the alias too:
# --- exporting symbols safely ---
if "__all__" not in globals():
    __all__: List[str] = []

if "collect_candidates" not in __all__:
    __all__.append("collect_candidates")
# expose the new enrichment helper for external imports
if "enrich_tokens_with_price_change" not in __all__:
    __all__.append("enrich_tokens_with_price_change")

# Expose the latest background birdeye task (if any) and provide a helper that
# returns a small snapshot (try waiting briefly for the background task,
# else fall back to in-memory cache). Paste this near the other helpers in fetching.py.

# Module-global slot other modules can use to pick up the background task.
_birdeye_bg_task: Optional[asyncio.Task] = None

def _export_birdeye_bg_task(task: Optional[asyncio.Task]) -> None:
    """Internal helper to store a pointer to the background birdeye task."""
    global _birdeye_bg_task
    try:
        _birdeye_bg_task = task
    except Exception:
        _birdeye_bg_task = None

async def get_birdeye_snapshot_count(timeout: float = 1.2) -> int:
    """
    Try to return a current count of Birdeye-discovered tokens:
     - If a background task is present, wait a short `timeout` for it and return its length.
     - Otherwise, fall back to fetch_birdeye_tokens._CACHE (30s cache) if available.
     - Does not start new long-running network jobs; conservative and safe for logging.
    """
    # 1) If background task exists, await it briefly
    try:
        task = globals().get("_birdeye_bg_task") or None
        if task is not None and not task.done():
            try:
                res = await asyncio.wait_for(task, timeout=timeout)
                return len(res or [])
            except asyncio.TimeoutError:
                # background task not ready; fall through to cache fallback
                pass
            except Exception:
                # background task failed — fall through to cache fallback
                pass
        elif task is not None and task.done():
            try:
                res = task.result()
                return len(res or [])
            except Exception:
                # task finished but errored — fall through
                pass
    except Exception:
        pass

    # 2) Fallback: return cached items if fetch_birdeye_tokens populated it
    try:
        cached = getattr(fetch_birdeye_tokens, "_CACHE", None)
        if isinstance(cached, dict):
            items = cached.get("items") or []
            return len(items)
    except Exception:
        pass

    # 3) Last-resort: no data available quickly
    return 0

# ---- Shared clients shutdown helper ------------------------------------------------
async def shutdown_shared_clients(timeout: float = 5.0) -> None:
    """
    Close any long-lived/shared clients created by this module (birdeye client, rugcheck client).
    Call from test teardown or short-lived scripts to avoid unclosed-ClientSession warnings.
    """
    global _birdeye_client, _rug_client

    to_await = []
    closed_labels = []

    # Close birdeye client if present and has a close() method
    bc = globals().get("_birdeye_client")
    if bc:
        try:
            close_fn = getattr(bc, "close", None) or (getattr(bc, "session", None) and getattr(bc.session, "close", None))
            if callable(close_fn):
                res = close_fn()
                if asyncio.iscoroutine(res):
                    to_await.append(res)
                    closed_labels.append("_birdeye_client")
                else:
                    # sync close invoked immediately
                    closed_labels.append("_birdeye_client")
            else:
                logger.debug("No close() found on birdeye client object")
        except Exception:
            logger.debug("Failed to schedule birdeye client close", exc_info=True)

    # Close shared RugcheckClient if present and has an async close()
    rc = globals().get("_rug_client")
    if rc:
        try:
            close_fn = getattr(rc, "close", None)
            if callable(close_fn):
                res = close_fn()
                if asyncio.iscoroutine(res):
                    to_await.append(res)
                    closed_labels.append("_rug_client")
                else:
                    closed_labels.append("_rug_client")
            else:
                logger.debug("No close() found on rugcheck client object")
        except Exception:
            logger.debug("Failed to schedule rugcheck client close", exc_info=True)

    if to_await:
        try:
            await asyncio.wait_for(asyncio.gather(*to_await, return_exceptions=True), timeout=timeout)
            logger.info("shutdown_shared_clients: closed %d client(s): %s", len(to_await), ", ".join(closed_labels))
        except Exception:
            logger.debug("shutdown_shared_clients: timed out or encountered error", exc_info=True)

    # Only clear module-level singletons after close attempts completed
    try:
        _birdeye_client = None
    except Exception:
        pass
    try:
        _rug_client = None
    except Exception:
        pass


def shutdown_shared_clients_sync(timeout: float = 5.0) -> None:
    """
    Synchronous wrapper for shutdown_shared_clients for callers that are not async.
    """
    try:
        asyncio.run(shutdown_shared_clients(timeout=timeout))
    except RuntimeError:
        # If called from within a running loop (e.g. certain test runners), schedule via create_task
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(shutdown_shared_clients(timeout=timeout))
        except Exception:
            logger.debug("shutdown_shared_clients_sync: failed to schedule async shutdown", exc_info=True)
                        
# ---------------------------------------------------------------------
# Optional: quick sync wrapper (handy for CLI/local testing)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio as _a
    async def _main():
        try:
            toks = await get_latest_tokens(max_items=int(os.getenv("TEST_MAX_ITEMS", "500")))
            print(f"Got {len(toks)} tokens (sample 3):")
            for t in toks[:3]:
                print(json.dumps({k: t.get(k) for k in ("address","symbol","liquidity","volume_24h","mc")}, indent=2))
        except Exception as e:
            print("Error:", e)
    _a.run(_main())

# =====================================================================
# === Signals & Patterns (RSI, MACD, Bollinger, TD9, Candlestick)   ===
# === Lightweight, NumPy-only, post-shortlist enrichment utilities  ===
# =====================================================================

# ---- local config shim for signals (do NOT override module-wide _cfg) ----
def _cfg_signals() -> Dict[str, Any]:
    """
    Signals-specific config accessor. Prefer a module-level `load_config`
    if present; otherwise fall back to a global CONFIG dict. This keeps the
    signals helper local and avoids clobbering the module-wide _cfg().
    """
    try:
        lc = globals().get("load_config")
        if callable(lc):
            return lc()
    except Exception:
        pass
    try:
        cfg = globals().get("CONFIG")
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass
    return {}

def _np_arr(a):
    if _np is None:
        raise ImportError("NumPy is required for indicators; please add numpy to requirements.")
    return _np.asarray(a, dtype=float)

def _signals_cfg():
    c = _cfg_signals() if callable(_cfg_signals) else {}
    sig = (c.get("signals") if isinstance(c, dict) else None) or {}
    pats = (c.get("patterns") if isinstance(c, dict) else None) or {}
    return {
        "enable": bool(sig.get("enable", True)),
        "rsi": {"enable": bool(sig.get("rsi", {}).get("enable", True)), "period": int(sig.get("rsi", {}).get("period", 14))},
        "macd": {
            "enable": bool(sig.get("macd", {}).get("enable", True)),
            "fast": int(sig.get("macd", {}).get("fast", 12)),
            "slow": int(sig.get("macd", {}).get("slow", 26)),
            "signal": int(sig.get("macd", {}).get("signal", 9)),
        },
        "bollinger": {
            "enable": bool(sig.get("bollinger", {}).get("enable", True)),
            "period": int(sig.get("bollinger", {}).get("period", 20)),
            "stddev": float(sig.get("bollinger", {}).get("stddev", 2.0)),
        },
        "td9": {"enable": bool(sig.get("td9", {}).get("enable", True)), "lookback": int(sig.get("td9", {}).get("lookback", 30))},
        "patterns": {
            "enable": bool(pats.get("enable", True)),
            "list": list(pats.get("list", ["bullish_engulfing","bearish_engulfing","hammer","shooting_star","doji"])),
        },
        "ohlcv": {
            "interval": (sig.get("ohlcv", {}) or {}).get("interval", "1m"),
            "limit": int((sig.get("ohlcv", {}) or {}).get("limit", 200)),
        }
    }

def _ema(arr, period: int):
    x = _np_arr(arr)
    if x.size < period:
        out = _np.empty_like(x); out[:] = _np.nan
        return out
    k = 2.0 / (period + 1)
    out = _np.empty_like(x); out[:] = _np.nan
    sma = _np.nanmean(x[:period])
    out[period-1] = sma
    for i in range(period, x.size):
        out[i] = x[i] * k + out[i-1] * (1 - k)
    return out

def ind_rsi(close, period: int = 14):
    c = _np_arr(close)
    if c.size < period + 1:
        rsis = _np.empty_like(c); rsis[:] = _np.nan
        return rsis
    diff = _np.diff(c, prepend=c[0])
    gains = _np.clip(diff, 0, None)
    losses = -_np.clip(diff, None, 0)
    rsis = _np.empty_like(c); rsis[:] = _np.nan
    avg_gain = _np.convolve(gains, _np.ones(period), 'valid')[:1].mean()
    avg_loss = _np.convolve(losses, _np.ones(period), 'valid')[:1].mean()
    if avg_loss == 0:
        rsis[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsis[period] = 100.0 - (100.0 / (1.0 + rs))
    ag, al = avg_gain, avg_loss
    for i in range(period + 1, c.size):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            rsis[i] = 100.0
        else:
            rs = ag / al
            rsis[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsis

def ind_macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    c = _np_arr(close)
    ema_fast = _ema(c, fast)
    ema_slow = _ema(c, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}

def ind_bollinger(close, period: int = 20, stddev: float = 2.0):
    c = _np_arr(close)
    n = c.size
    mid = _np.full(n, _np.nan)
    upper = _np.full(n, _np.nan)
    lower = _np.full(n, _np.nan)
    width = _np.full(n, _np.nan)
    percent_b = _np.full(n, _np.nan)
    if n < period:
        return {"mid": mid, "upper": upper, "lower": lower, "width": width, "percent_b": percent_b}
    # O(n*period) rolling — fine for 100–500 bars
    for i in range(period-1, n):
        w = c[i-period+1:i+1]
        m = _np.mean(w); s = _np.std(w, ddof=0)
        mid[i] = m
        upper[i] = m + stddev * s
        lower[i] = m - stddev * s
        width[i] = (upper[i] - lower[i]) / m if m != 0 and not _np.isnan(m) else _np.nan
        if not _np.isnan(upper[i]) and not _np.isnan(lower[i]) and upper[i] != lower[i]:
            percent_b[i] = (c[i] - lower[i]) / (upper[i] - lower[i])
    return {"mid": mid, "upper": upper, "lower": lower, "width": width, "percent_b": percent_b}

def ind_td9(close, lookback: int = 30):
    c = _np_arr(close); n = c.size
    up = _np.zeros(n, dtype=int)     # bearish setup count
    down = _np.zeros(n, dtype=int)   # bullish setup count
    setup = _np.full(n, "", dtype=object)
    td9_up = _np.zeros(n, dtype=bool)
    td9_down = _np.zeros(n, dtype=bool)
    for i in range(n):
        if i < 4:
            continue
        if c[i] > c[i-4]:
            up[i] = (up[i-1] + 1) if up[i-1] > 0 else 1
            down[i] = 0
        elif c[i] < c[i-4]:
            down[i] = (down[i-1] + 1) if down[i-1] > 0 else 1
            up[i] = 0
        else:
            up[i] = 0; down[i] = 0
        if up[i] == 9:
            setup[i] = "td9_up"; td9_up[i] = True
        elif down[i] == 9:
            setup[i] = "td9_down"; td9_down[i] = True
    if lookback and n > lookback:
        start = n - lookback
        td9_up[:start] = False
        td9_down[:start] = False
    return {"up": up, "down": down, "setup": setup, "td9_up": td9_up, "td9_down": td9_down}

# -------------------------- Candle Patterns --------------------------
def pat_bullish_engulfing(open_, close):
    o, c = _np_arr(open_), _np_arr(close)
    prev_bear = c[:-1] < o[:-1]
    curr_bull = c[1:]  > o[1:]
    engulf    = (c[1:] >= o[:-1]) & (o[1:] <= c[:-1])
    out = _np.zeros_like(c, dtype=bool); out[1:] = prev_bear & curr_bull & engulf
    return out

def pat_bearish_engulfing(open_, close):
    o, c = _np_arr(open_), _np_arr(close)
    prev_bull = c[:-1] > o[:-1]
    curr_bear = c[1:]  < o[1:]
    engulf    = (o[1:] >= c[:-1]) & (c[1:] <= o[:-1])
    out = _np.zeros_like(c, dtype=bool); out[1:] = prev_bull & curr_bear & engulf
    return out

def pat_hammer(open_, high, low, close, tol: float = 0.3):
    o, h, l, c = map(_np_arr, (open_, high, low, close))
    body = _np.abs(c - o)
    upper = h - _np.maximum(c, o)
    lower = _np.minimum(c, o) - l
    return (lower >= 2*body) & (upper <= body) & ((h - c) <= tol * (h - l))

def pat_shooting_star(open_, high, low, close, tol: float = 0.3):
    o, h, l, c = map(_np_arr, (open_, high, low, close))
    body = _np.abs(c - o)
    upper = h - _np.maximum(c, o)
    lower = _np.minimum(c, o) - l
    return (upper >= 2*body) & (lower <= body) & ((c - l) <= tol * (h - l))

def pat_doji(open_, close, eps: float = 1e-6, rel: float = 0.1):
    o, c = _np_arr(open_), _np_arr(close)
    rng = _np.maximum(_np.abs(c - o), eps)
    return (_np.abs(c - o) / rng) < rel

# --- classify_patterns_arrays: use concrete (static) return typing --------
# Return lists of booleans (safe for runtime and static analysis).
def classify_patterns_arrays(ohlcv: Dict[str, Iterable[float]], names: Iterable[str]) -> Dict[str, List[bool]]:
    """
    Classify the given OHLCV arrays for the requested pattern names.

    Returns a mapping: pattern_name -> list[bool] (last element corresponds to last bar).
    We convert numpy arrays to Python lists before returning so the type is stable
    whether or not numpy is available.
    """
    # _np_arr should already be defined elsewhere to produce either numpy arrays or lists.
    o, h, l, c = map(_np_arr, (ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]))
    out: Dict[str, List[bool]] = {}

    for name in names:
        try:
            if name == "bullish_engulfing":
                res = pat_bullish_engulfing(o, c)
            elif name == "bearish_engulfing":
                res = pat_bearish_engulfing(o, c)
            elif name == "hammer":
                res = pat_hammer(o, h, l, c)
            elif name == "shooting_star":
                res = pat_shooting_star(o, h, l, c)
            elif name == "doji":
                res = pat_doji(o, c)
            else:
                # Unknown pattern: empty list
                out[name] = []
                continue

            # Ensure the returned value is a plain Python list[bool] for static typing stability.
            if hasattr(res, "tolist"):
                out[name] = list(res.tolist())
            else:
                # coerce iterable -> list
                out[name] = list(res)
        except Exception:
            # Defensive: if pattern computation fails, return an empty list for that pattern.
            out[name] = []

    return out

# ------------------------ Enrichment Orchestrator ---------------------

# (replace the existing enrich_token_with_signals implementation with the following)

async def enrich_token_with_signals(token: Dict[str, Any], ohlcv: Dict[str, Iterable[float]], cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Mutates and returns `token` with last-bar signals.
    Expected OHLCV: dict with keys: 'open','high','low','close','volume' arrays.
    """
    cfg = cfg or _signals_cfg()
    if not cfg.get("enable", True):
        return token
    close = _np_arr(ohlcv["close"])
    if close.size < 30:  # minimal bars
        return token

    # RSI
    if cfg["rsi"]["enable"]:
        _rsi = ind_rsi(close, cfg["rsi"]["period"])
        token["rsi"] = float(_rsi[-1]) if _rsi.size else None

    # MACD
    if cfg["macd"]["enable"]:
        m = ind_macd(close, cfg["macd"]["fast"], cfg["macd"]["slow"], cfg["macd"]["signal"])
        token["macd"] = float(m["macd"][-1]) if m["macd"].size else None
        token["macd_signal"] = float(m["signal"][-1]) if m["signal"].size else None
        token["macd_hist"] = float(m["hist"][-1]) if m["hist"].size else None

    # Bollinger
    if cfg["bollinger"]["enable"]:
        bb = ind_bollinger(close, cfg["bollinger"]["period"], cfg["bollinger"]["stddev"])
        token["bb_percent_b"] = float(bb["percent_b"][-1]) if bb["percent_b"].size else None
        token["bb_width"]     = float(bb["width"][-1]) if bb["width"].size else None

    # TD9
    if cfg["td9"]["enable"]:
        td = ind_td9(close, cfg["td9"]["lookback"])
        token["td9_up"]   = bool(td["td9_up"][-1]) if td["td9_up"].size else False
        token["td9_down"] = bool(td["td9_down"][-1]) if td["td9_down"].size else False

    # Patterns (defensive: support numpy arrays and plain Python lists)
    if cfg["patterns"]["enable"]:
        pats = classify_patterns_arrays(ohlcv, cfg["patterns"]["list"])
        out_patterns: List[str] = []
        for name, v in (pats.items() if isinstance(pats, dict) else []):
            try:
                # compute a size compatible with numpy arrays (.size) and Python lists (len)
                size = getattr(v, "size", None)
                if size is None:
                    try:
                        size = len(v)
                    except Exception:
                        size = 0
                if size and bool(v[-1]):
                    out_patterns.append(name)
            except Exception:
                # be conservative: skip any pattern that errors
                continue
        token["patterns"] = out_patterns

    return token

async def batch_enrich_tokens_with_signals(
    tokens: List[Dict[str, Any]],
    fetch_ohlcv_func,  # async callable: (address, interval, limit) -> Dict[str, List[float]]
    interval: Optional[str] = None,
    limit: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
    concurrency: int = 8,
) -> List[Dict[str, Any]]:
    """Generic post-shortlist enricher. You provide how to fetch OHLCV for a token.
    We call it for top-N tokens and attach signals/patterns to each.
    """
    cfg_base = _signals_cfg()
    interval = interval or cfg_base["ohlcv"]["interval"]
    limit = int(limit or cfg_base["ohlcv"]["limit"])
    cfg = cfg or cfg_base

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _enrich_one(tok: Dict[str, Any]):
        addr = tok.get("address") or tok.get("token_address")
        if not addr:
            return tok
        async with sem:
            try:
                ohlcv = await fetch_ohlcv_func(addr, interval=interval, limit=limit)
                if isinstance(ohlcv, dict) and all(k in ohlcv for k in ("open","high","low","close","volume")):
                    await enrich_token_with_signals(tok, ohlcv, cfg=cfg)
            except Exception as e:
                try:
                    logger.debug("batch_enrich: OHLCV fetch/enrich failed for %s: %s", addr, e)
                except Exception:
                    pass
        return tok

    return await asyncio.gather(*[_enrich_one(t) for t in tokens])