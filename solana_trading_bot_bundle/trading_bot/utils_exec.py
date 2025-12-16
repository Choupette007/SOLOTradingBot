# solana_trading_bot_bundle/trading_bot/utils.py
# Updated utils.py — re-exports pattern helpers and provides safe stubs when the modules are not present.
# This preserves backward compatibility with callers that import pattern detectors from utils.
from __future__ import annotations

import base64
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union, Sequence
import asyncio
import threading
from concurrent.futures import Future as ThreadFuture
import random
import statistics
import binascii  
import traceback

import yaml
from cachetools import TTLCache
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.signature import Signature

# Try to re-export candlestick/pattern helpers so they are available via utils.
# If the canonical modules are absent, provide conservative stubs to keep imports working.
try:
    # Prefer the local "patterns.py" aggregator if present (returns simple lists / booleans)
    from .patterns import (  # type: ignore
        is_doji,
        is_hammer,
        is_inverted_hammer,
        is_bullish_engulfing,
        is_bearish_engulfing,
        is_three_white_soldiers,
        is_three_black_crows,
        is_morning_star,
        is_evening_star,
        classify_patterns as classify_patterns_arrays,
    )
except Exception:
    try:
        # Fall back to candlestick_patterns (alternate implementation)
        from .candlestick_patterns import (  # type: ignore
            is_doji,
            is_hammer,
            is_inverted_hammer,
            is_bullish_engulfing,
            is_bearish_engulfing,
            is_three_white_soldiers,
            is_three_black_crows,
            is_morning_star,
            is_evening_star,
            classify_patterns as _classify_patterns_dict,
        )

        # Provide a simple, robust adapter that normalizes output to a list of active pattern names
        def classify_patterns_arrays(df) -> list:
            """
            Adapter around alternate classify_patterns that may return either:
              - dict[name] -> per-bar boolean-list/array, or
              - list of pattern-names (legacy)
            We return a list of active pattern names for the latest bar, or [] if unknown.
            """
            try:
                d = _classify_patterns_dict(df)
                # If already a list of names, return it directly
                if isinstance(d, list):
                    return d
                if isinstance(d, dict):
                    # Attempt to detect latest index for per-bar sequences
                    latest_index = None
                    sample = next(iter(d.values()), None)
                    try:
                        if sample is not None and hasattr(sample, "__len__"):
                            latest_index = max(0, len(sample) - 1)
                    except Exception:
                        latest_index = None
                    out = []
                    for k, v in d.items():
                        try:
                            if latest_index is not None:
                                # v may be numpy array or list-like
                                val = v[latest_index]
                                if bool(val):
                                    out.append(k)
                        except Exception:
                            # best-effort: if v is truthy scalar, include
                            try:
                                if bool(v):
                                    out.append(k)
                            except Exception:
                                pass
                    return out
            except Exception:
                pass
            return []
    except Exception:
        # Provide safe, conservative stubs so imports from utils do not fail.
        def is_doji(*_a, **_k) -> bool:
            return False

        def is_hammer(*_a, **_k) -> bool:
            return False

        def is_inverted_hammer(*_a, **_k) -> bool:
            return False

        def is_bullish_engulfing(*_a, **_k) -> bool:
            return False

        def is_bearish_engulfing(*_a, **_k) -> bool:
            return False

        def is_three_white_soldiers(*_a, **_k) -> bool:
            return False

        def is_three_black_crows(*_a, **_k) -> bool:
            return False

        def is_morning_star(*_a, **_k) -> bool:
            return False

        def is_evening_star(*_a, **_k) -> bool:
            return False

        def classify_patterns_arrays(*_a, **_k) -> list:
            return []

# rolling_median / rolling_mad: attempt to import from an indicators module if present,
# otherwise provide small stubs so utils.__all__ remains accurate.
try:
    from .indicators import rolling_median, rolling_mad  # type: ignore
except Exception:
    def rolling_median(seq, window=5):
        # conservative stub: return simple rolling median fallback if seq is list-like
        try:
            if not seq:
                return []
            seq_list = list(seq)
            n = len(seq_list)
            out = []
            for i in range(n):
                start = max(0, i - window + 1)
                w = seq_list[start : i + 1]
                out.append(float(statistics.median(w)))
            return out
        except Exception:
            return []

    def rolling_mad(seq, window=5):
        # conservative stub: compute median absolute deviation over rolling window if possible,
        # otherwise return zeros.
        try:
            seq_list = list(seq or [])
            n = len(seq_list)
            out = []
            for i in range(n):
                start = max(0, i - window + 1)
                w = seq_list[start : i + 1]
                med = statistics.median(w) if w else 0.0
                mad = statistics.median([abs(x - med) for x in w]) if w else 0.0
                out.append(float(mad))
            return out
        except Exception:
            try:
                return [0.0] * len(seq or [])
            except Exception:
                return []

# Add small volume spike helper (was referenced but missing)
def is_volume_spike(vol_seq, window: int = 20, factor: float = 3.0) -> bool:
    """
    Return True if the last volume in vol_seq is >= factor * median(previous_window).
    Robust to short sequences; returns False on errors.
    """
    try:
        seq = list(vol_seq or [])
        n = len(seq)
        if n == 0:
            return False
        last = float(seq[-1])
        prev = seq[max(0, n - window - 1) : n - 1]  # exclude last
        if not prev:
            # fall back to simple heuristic: last >= factor
            return last >= float(factor)
        med = float(statistics.median(prev))
        if med <= 0:
            return last >= float(factor)
        return last >= med * float(factor)
    except Exception:
        return False

# Prefer modern solders types (solana-py >= 0.36)
try:
    from solders.transaction import Transaction, VersionedTransaction  # type: ignore
except Exception:
    try:
        from solana.transaction import Transaction  # type: ignore
    except Exception as e:
        logging.getLogger("TradingBot").error(
            "Failed to import Transaction type from 'solders' or 'solana'. Ensure compatible versions are installed. "
            "Recommended: solana>=0.36,<0.37 and solders>=0.26,<0.27. Underlying error: %s",
            e,
        )
        raise
    VersionedTransaction = None  # type: ignore

# Optional signing support with fallback
try:
    from solders.keypair import Keypair  # type: ignore
except Exception:
    try:
        from solana.keypair import Keypair  # type: ignore
    except Exception:
        Keypair = None  # type: ignore

from solana_trading_bot_bundle.common.constants import (
    APP_NAME,
    local_appdata_dir,
    appdata_dir,
    logs_dir,
    data_dir,
    config_path,
    env_path,
    db_path,
    token_cache_path,
    ensure_app_dirs,
    prefer_appdata_file,
)

# Single canonical logger used across modules
logger = logging.getLogger("TradingBot")
logging.getLogger("SoloTradingBot").propagate = True

# -----------------------------------------------------------------------------
# Whitelist & caches
# -----------------------------------------------------------------------------
WHITELISTED_TOKENS: Dict[str, str] = {
    "So11111111111111111111111111111111111111112": "SOL",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
}

price_cache = TTLCache(maxsize=1000, ttl=300)
token_balance_cache = TTLCache(maxsize=1000, ttl=60)
token_account_existence_cache = TTLCache(maxsize=1000, ttl=300)


def compute_buy_size_usd(target_usd: float, sol_price: float) -> float:
    try:
        return target_usd / float(sol_price or 1.0)
    except Exception:
        return 0.1

# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def _to_path(p: Any) -> Path:
    try:
        v = p() if callable(p) else p
    except Exception:
        v = p
    try:
        v2 = prefer_appdata_file(v)
    except Exception:
        v2 = v
    return Path(v2)

def _appdata() -> Path:
    try:
        return _to_path(appdata_dir)
    except Exception:
        return Path(os.getenv("LOCALAPPDATA") or (Path.home() / ".local" / "share")) / APP_NAME

def _logs_dir() -> Path:
    p = _to_path(logs_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _config_path() -> Path:
    return _to_path(config_path)

def _env_file_path() -> Path:
    """
    Deterministic .env file path. Prefer LOCALAPPDATA if available (Windows),
    otherwise fall back to APPDATA / user home appdata folder.
    """
    try:
        # Prefer local appdata so we have a single authoritative .env (avoids mixed Roaming vs Local).
        local = os.getenv("LOCALAPPDATA") or os.getenv("XDG_CONFIG_HOME")
        if local:
            p = Path(local) / APP_NAME / ".env"
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:
        pass
    try:
        # Fallback to configured env_path constant
        return _to_path(env_path)
    except Exception:
        # Last resort: home dir
        return Path.home() / ".config" / APP_NAME / ".env"

# global ENV_PATH used by dotenv helpers
ENV_PATH = _env_file_path()

# Ensure base folders exist early
try:
    ensure_app_dirs()
except Exception:
    pass
try:
    _logs_dir()
except Exception:
    pass

# -----------------------------------------------------------------------------
# Config & logging helpers
# -----------------------------------------------------------------------------
_missing_cfg_last_log_ts: float = 0.0

def _resolve_config_path(path: Optional[str] = None) -> Path:
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.append(_config_path())
    candidates.append(_appdata() / "config.yaml")
    candidates.append(Path.cwd() / "config.yaml")
    for c in candidates:
        try:
            c.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            if c.exists() or os.access(str(c.parent), os.W_OK):
                return c
        except Exception:
            continue
    p = _appdata() / "config.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _create_default_config(cfg_path: Path) -> None:
    try:
        if not cfg_path.exists() or (cfg_path.stat().st_size == 0):
            default_cfg = {
                "logging": {
                    "file": str(_logs_dir() / "bot.log"),
                    "log_level": "INFO",
                    "log_rotation_size_mb": 10,
                    "log_max_files": 5,
                },
                "discovery": {
                    "dexscreener_pages": 5,
                    "dexscreener_per_page": 75,
                    "dexscreener_post_cap": 0,
                    "raydium_max_pairs": 400,
                    "raydium_page_size": 100,
                    "raydium_max_pools": 800,
                    "birdeye_max_tokens": 200,
                    "rugcheck_in_discovery": False,
                    "require_rugcheck_pass": False,
                },
                "database": {
                    "token_cache_path": str(_to_path(token_cache_path)),
                },
            }
            cfg_path.write_text("# Auto-generated default config\n" + yaml.safe_dump(default_cfg), encoding="utf-8")
    except Exception as e:
        logging.getLogger("TradingBot").debug("Could not create default config at %s: %s", cfg_path, e)

def load_config(path: str | None = None) -> Dict:
    global _missing_cfg_last_log_ts
    cfg_path = _resolve_config_path(path)
    _create_default_config(cfg_path)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.debug("Loaded configuration from %s", cfg_path)
        return config
    except Exception as e:
        now = time.time()
        if now - _missing_cfg_last_log_ts > 30:
            logger.error("Failed to load config from %s: %s", cfg_path, e)
            _missing_cfg_last_log_ts = now
        return {}

_LOG_SENTINEL_ATTR = "_solo_logging_file"

def setup_logging(config: dict[str, Any] | None) -> logging.Logger:
    log_cfg = (config or {}).get("logging", {}) if isinstance(config, dict) else {}
    from solana_trading_bot_bundle.common.constants import logs_dir as _logs_dir_fn

    raw_file = log_cfg.get("file") or (_logs_dir_fn() / "bot.log")
    log_file = Path(raw_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level_name = str(log_cfg.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    max_size_mb = int(log_cfg.get("log_rotation_size_mb", 10))
    max_files = int(log_cfg.get("log_max_files", 5))

    root = logging.getLogger()
    root.setLevel(level)

    if getattr(root, _LOG_SENTINEL_ATTR, None) == str(log_file):
        for h in root.handlers:
            h.setLevel(level)
        return logging.getLogger("TradingBot")

    file_handler: Optional[RotatingFileHandler] = None
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == str(log_file):
            file_handler = h
            break
    if file_handler is None:
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=max_files,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        file_handler.setLevel(level)
        root.addHandler(file_handler)
    else:
        file_handler.setLevel(level)

    has_console = any(isinstance(h, logging.StreamHandler) and not hasattr(h, "baseFilename") for h in root.handlers)
    if not has_console:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        sh.setLevel(level)
        root.addHandler(sh)

    for name in ("TradingBot", "TradingBotGUI", "SoloTradingBot"):
        lx = logging.getLogger(name)
        lx.handlers.clear()
        lx.propagate = True
        lx.setLevel(level)

    setattr(root, _LOG_SENTINEL_ATTR, str(log_file))
    logging.getLogger("TradingBot").info("Logging configured: level=%s, file=%s", level_name, str(log_file))
    return logging.getLogger("TradingBot")

# -----------------------------------------------------------------------------
# .env helpers
# -----------------------------------------------------------------------------
ENV_PATH = _env_file_path()

def _parse_dotenv_file(path: Path) -> dict:
    data: Dict[str, str] = {}
    try:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip().strip('"')
    except Exception as e:
        logger.debug("Failed to parse .env: %s", e)
    return data

def _write_env_key(key: str, value: str) -> None:
    try:
        ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        current = _parse_dotenv_file(ENV_PATH)
        current[key] = value
        lines = ["# SOLOTradingBot .env"]
        for k, v in current.items():
            v_clean = str(v).replace("\n", "").replace("\r", "").replace('"', '\\"')
            lines.append(f'{k}="{v_clean}"')
        ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if os.name != "nt":
            try:
                os.chmod(ENV_PATH, 0o600)
            except Exception:
                pass
    except Exception as e:
        logger.warning("Failed to write %s to .env: %s", key, e)

# -----------------------------------------------------------------------------
# Rugcheck helpers (token login + headers)
# -----------------------------------------------------------------------------
def _base64url_decode_noverify(seg: str) -> bytes:
    seg_b = seg.encode() if isinstance(seg, str) else seg
    pad = b"=" * ((4 - (len(seg_b) % 4)) % 4)
    import base64 as _b64
    return _b64.urlsafe_b64decode(seg_b + pad)

def _jwt_exp_unix(jwt_token: str) -> Optional[int]:
    try:
        parts = jwt_token.split(".")
        if len(parts) != 3:
            return None
        payload = json.loads(_base64url_decode_noverify(parts[1]).decode("utf-8"))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None

def _is_token_still_valid(jwt_token: str, leeway_seconds: int = 300) -> bool:
    exp = _jwt_exp_unix(jwt_token)
    if not exp:
        return False
    now = int(time.time())
    return (exp - now) > leeway_seconds

def _load_wallet_keypair_from_env() -> Any:
    if Keypair is None:
        raise RuntimeError("Keypair type not available; cannot sign Rugcheck login.")
    pk = (os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("WALLET_PRIVATE_KEY") or "").strip()
    if not pk:
        raise ValueError("Set SOLANA_PRIVATE_KEY or WALLET_PRIVATE_KEY in your .env to auto-login Rugcheck.")
    try:
        return Keypair.from_base58_string(pk)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        arr = json.loads(pk)
        if isinstance(arr, list) and len(arr) == 64 and all(isinstance(x, int) for x in arr):
            return Keypair.from_bytes(bytes(arr))
    except Exception:
        pass
    raise ValueError("Private key format not recognized (expect base58 64-byte secret or JSON array of 64 ints).")

def get_rugcheck_token(refresh: bool = False) -> Optional[str]:
    tok = (os.getenv("RUGCHECK_JWT_TOKEN") or "").strip()
    if tok and not refresh and _is_token_still_valid(tok):
        return tok

    try:
        import base58  # type: ignore
        import requests  # type: ignore

        wallet = _load_wallet_keypair_from_env()
        message_data = {
            "message": "Sign-in to Rugcheck.xyz",
            "timestamp": int(time.time() * 1000),
            "publicKey": str(wallet.pubkey()),
        }
        message_json = json.dumps(message_data, separators=(",", ":")).encode("utf-8")
        signature = wallet.sign_message(message_json)
        sig_b58 = str(signature)
        signature_data = list(base58.b58decode(sig_b58))
        payload = {
            "signature": {"data": signature_data, "type": "ed25519"},
            "wallet": str(wallet.pubkey()),
            "message": message_data,
        }
        resp = requests.post(
            "https://api.rugcheck.xyz/auth/login/solana",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            new_tok = data.get("token") or data.get("jwt") or data.get("access_token")
            if new_tok:
                os.environ["RUGCHECK_JWT_TOKEN"] = new_tok
                _write_env_key("RUGCHECK_JWT_TOKEN", new_tok)
                return new_tok
            logger.warning("Rugcheck login succeeded but token field not found in response.")
        else:
            logger.warning("Rugcheck login failed: %s %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Rugcheck auto-login error: %s", e)

    return tok or None

async def get_rugcheck_token_async(refresh: bool = False) -> Optional[str]:
    return await asyncio.to_thread(get_rugcheck_token, refresh)

def get_rugcheck_headers(force_refresh: bool = False) -> Dict[str, str]:
    tok = get_rugcheck_token(refresh=force_refresh)
    return {"Authorization": f"Bearer {tok}"} if tok else {}

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
async def add_to_blacklist(token_address: str, reason: str) -> None:
    cfg = load_config()
    db_file = _to_path((cfg.get("database", {}) or {}).get("token_cache_path", token_cache_path))
    try:
        import aiosqlite
        async with aiosqlite.connect(db_file) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS blacklist (
                    address   TEXT PRIMARY KEY,
                    reason    TEXT,
                    timestamp INTEGER
                )
                """
            )
            await db.execute(
                "INSERT OR REPLACE INTO blacklist (address, reason, timestamp) VALUES (?, ?, ?)",
                (token_address, reason, int(datetime.now().timestamp())),
            )
            await db.commit()
        logger.info("Added %s to blacklist: %s", token_address, reason)
    except Exception as e:
        logger.error("Failed to add %s to blacklist: %s", token_address, e)

def deduplicate_tokens(tokens: List[Dict]) -> List[Dict]:
    seen: set[str] = set()
    out: List[Dict] = []
    for t in tokens or []:
        addr = t.get("address")
        if addr and addr not in seen:
            seen.add(addr)
            out.append(t)
    return out

def remove_none_keys(d: Dict) -> Dict:
    return {k: v for k, v in (d or {}).items() if v is not None and v != ""}

def _best_first(*vals):
    """
    Return the first non-empty/non-None value among arguments (0 is valid).
    Consider empty strings, None, empty lists and empty dicts as "missing".
    """
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return None

def custom_json_encoder(obj: object) -> str:
    if isinstance(obj, (Pubkey, Signature)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# -----------------------------------------------------------------------------
# Robust safe_create_task
# -----------------------------------------------------------------------------
def safe_create_task(coro, *, name: Optional[str] = None) -> Union[asyncio.Task, ThreadFuture, threading.Thread]:
    """
    Schedule a coroutine safely from both async and sync contexts.

    Returns:
      - asyncio.Task when scheduled in the current running loop
      - concurrent.futures.Future when scheduled via run_coroutine_threadsafe on another thread's loop
      - threading.Thread when a daemon thread was used to run asyncio.run(coro)
    """
    label = name or "<task>"

    # Attach logging callback for asyncio.Task-like objects
    def _attach_task_logging(task_like, label_inner: str) -> None:
        try:
            if hasattr(task_like, "add_done_callback"):
                def _done_cb(t):
                    try:
                        if isinstance(t, asyncio.Task):
                            t.result()
                        else:
                            # concurrent.futures.Future
                            t.result()
                    except asyncio.CancelledError:
                        logger.info("Background task %s cancelled", label_inner)
                    except Exception as exc:
                        logger.exception("Background task %s raised exception: %s", label_inner, exc)
                task_like.add_done_callback(_done_cb)
        except Exception:
            logger.exception("Failed to attach done-callback to background task %s", label_inner)

    # 1) If there's a running loop in this thread, use create_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        try:
            task = asyncio.create_task(coro, name=label) if hasattr(asyncio, "create_task") else asyncio.ensure_future(coro)
            _attach_task_logging(task, label)
            return task
        except Exception:
            logger.debug("asyncio.create_task failed; will try cross-thread scheduling", exc_info=True)

    # 2) Try to schedule on the default event loop (maybe running in another thread)
    try:
        loop_obj = asyncio.get_event_loop()
    except Exception:
        loop_obj = None

    if loop_obj is not None and getattr(loop_obj, "is_running", lambda: False)():
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop_obj)
            _attach_task_logging(fut, label)
            return fut
        except Exception:
            logger.debug("run_coroutine_threadsafe failed; falling back to thread runner", exc_info=True)

    # 3) Last-resort: run in a daemon thread using asyncio.run
    def _thread_runner():
        try:
            asyncio.run(coro)
        except Exception as e:
            logger.exception("Background thread coroutine %s crashed: %s", label, e)

    thr = threading.Thread(target=_thread_runner, name=f"bgtask-{label}", daemon=True)
    thr.start()
    logger.info("Started background coroutine %s in daemon thread %s", label, thr.name)
    return thr

# -----------------------------------------------------------------------------
# Backoff / Circuit-breaker helpers
# -----------------------------------------------------------------------------
async def backoff_sleep(attempt: int, base: float = 1.5, max_sleep: float = 120.0, jitter: bool = True) -> float:
    sleep = min(max_sleep, base * (2 ** max(0, attempt - 1)))
    if jitter:
        factor = random.uniform(0.8, 1.25)
        sleep = max(0.0, sleep * factor)
    sleep = max(0.1, float(sleep))
    await asyncio.sleep(sleep)
    return sleep

class CircuitBreaker429:
    def __init__(self, threshold: int = 5, cooldown_seconds: int = 60):
        self.threshold = int(threshold)
        self.cooldown_seconds = int(cooldown_seconds)
        self._count = 0
        self._last_trip_time: Optional[float] = None

    def record(self, is_429: bool) -> None:
        if is_429:
            self._count += 1
            if self._count >= self.threshold:
                self._last_trip_time = time.time()
        else:
            self._count = 0
            self._last_trip_time = None

    def is_open(self) -> bool:
        if self._last_trip_time is None:
            return False
        if (time.time() - self._last_trip_time) > self.cooldown_seconds:
            self._count = 0
            self._last_trip_time = None
            return False
        return True

    def remaining_cooldown(self) -> float:
        if self._last_trip_time is None:
            return 0.0
        rc = max(0.0, self.cooldown_seconds - (time.time() - self._last_trip_time))
        return rc
    
# -----------------------------------------------------------------------------
# Buy-sizing / risk-aware sizing
# -----------------------------------------------------------------------------
def compute_buy_size_usd(wallet_sol_balance: float, sol_usd_price: float, config: Dict[str, Any]) -> float:
    """
    Compute USD amount for a buy following risk-based sizing with safe fallbacks.

    Rules:
      - Prefer bot.max_risk_pct_of_portfolio if prefer_risk_sizing is True and portfolio value available.
      - Fall back to legacy buy_percentage if necessary.
      - Enforce min_usd_per_buy and max_order_usd clamps.
      - Respect global trading.min_order_usd when present.
    """
    botcfg = (config or {}).get("bot", {}) or {}
    tradingcfg = (config or {}).get("trading", {}) or {}

    prefer_risk = bool(botcfg.get("prefer_risk_sizing", True))
    max_risk_pct = float(botcfg.get("max_risk_pct_of_portfolio", 0.05))
    buy_percentage = float(botcfg.get("buy_percentage", 0.10))
    min_usd = float(botcfg.get("min_usd_per_buy", tradingcfg.get("min_order_usd", 50.0)))
    max_usd = float(botcfg.get("max_order_usd", tradingcfg.get("max_order_usd", 500.0)))

    # portfolio USD estimate
    portfolio_usd = 0.0
    try:
        if wallet_sol_balance is not None and sol_usd_price and sol_usd_price > 0:
            portfolio_usd = float(wallet_sol_balance) * float(sol_usd_price)
    except Exception:
        portfolio_usd = 0.0

    # compute candidate sizes
    risk_buy = portfolio_usd * float(max_risk_pct) if portfolio_usd > 0 else 0.0
    legacy_buy = portfolio_usd * float(buy_percentage) if portfolio_usd > 0 else 0.0

    if prefer_risk and risk_buy > 0:
        chosen = max(min_usd, min(risk_buy, max_usd))
        reason = "risk_pct"
    elif legacy_buy > 0:
        chosen = max(min_usd, min(legacy_buy, max_usd))
        reason = "legacy_pct"
    else:
        chosen = min_usd
        reason = "min_floor"

    # respect global trading minimum if configured
    global_min = float(tradingcfg.get("min_order_usd", min_usd))
    if chosen < global_min:
        chosen = global_min

    logger.debug(
        "compute_buy_size_usd: wallet_sol=%.6f sol_price=%.4f portfolio_usd=%.2f prefer_risk=%s max_risk_pct=%.4f "
        "risk_buy=%.2f legacy_buy=%.2f chosen_usd=%.2f reason=%s",
        wallet_sol_balance or 0.0, sol_usd_price or 0.0, portfolio_usd, prefer_risk, max_risk_pct,
        risk_buy, legacy_buy, chosen, reason,
    )

    return float(chosen)

# --- Jupiter / RPC send helpers: improved logging + backoff + multi-RPC fallback --- #

async def _extract_blockhash_from_client(solana_client) -> str | None:
    """Return a recent blockhash string from the AsyncClient (best-effort)."""
    try:
        res = await solana_client.get_latest_blockhash()
        # Attempt several defensive access patterns
        if hasattr(res, "value") and res.value is not None:
            val = getattr(res, "value", None)
            bh = getattr(val, "blockhash", None) or (val.get("blockhash") if isinstance(val, dict) else None)
            if bh:
                return str(bh)
        if isinstance(res, dict):
            v = res.get("value") or {}
            bh = v.get("blockhash") or v.get("blockHash")
            if bh:
                return str(bh)
        # Last-resort attribute access
        try:
            return str(res.value.blockhash)
        except Exception:
            pass
    except Exception:
        logger.debug("_extract_blockhash_from_client failed", exc_info=True)
    return None


async def send_signed_tx_with_retries(
    signed_tx_bytes: bytes,
    solana_client,
    *,
    session=None,
    rpc_endpoints: Iterable[str] | None = None,
    max_attempts: int = 4,
    skip_preflight: bool = False,
    preflight_commitment: str = "confirmed",
    backoff_base: float = 1.5,
) -> str | None:
    """
    Send a fully-signed transaction (raw bytes) with retries, exponential backoff and optional RPC rotation.

    Returns txid string on success, otherwise None.
    """
    endpoints = list(rpc_endpoints or [])
    clients: list[tuple[str | None, Any]] = [(None, solana_client)]
    created_clients: list[Any] = []

    # lazy-create AsyncClient instances for endpoints
    async def _make_client(url: str):
        try:
            c = AsyncClient(url)
            created_clients.append(c)
            return c
        except Exception:
            logger.debug("Failed to create AsyncClient for %s", url, exc_info=True)
            return None

    # build clients list but lazily create
    for u in endpoints:
        clients.append((u, None))  # placeholder, will be created lazily

    last_error = None

    try:
        for attempt in range(1, max_attempts + 1):
            # iterate through clients (primary first, then endpoints)
            for idx, (rpc_id, client) in enumerate(clients):
                # lazily create client if placeholder
                if client is None and rpc_id is not None:
                    client = await _make_client(rpc_id)
                    clients[idx] = (rpc_id, client)

                if client is None:
                    logger.debug("send_signed_tx_with_retries: no client for rpc_id=%s (skipping)", rpc_id)
                    continue

                # short hex preview
                try:
                    short_hex = binascii.hexlify(signed_tx_bytes[:32]).decode("ascii")
                except Exception:
                    short_hex = "<hex-failed>"

                logger.info(
                    "send_signed_tx_with_retries: attempt=%d/%d client=%s size=%d skip_preflight=%s",
                    attempt, max_attempts, rpc_id or getattr(client, "_provider", getattr(client, "endpoint_uri", "<primary>")), len(signed_tx_bytes), skip_preflight,
                )
                logger.debug("send_signed_tx_with_retries: payload_preview_first32=%s", short_hex)

                try:
                    # Try the preferred send_raw_transaction signature (many wrappers accept opts as TxOpts or dict)
                    try:
                        # Try passing opts as a simple dict first (widest compatibility)
                        resp = await client.send_raw_transaction(
                            signed_tx_bytes,
                            opts={"skip_preflight": skip_preflight, "preflight_commitment": preflight_commitment},
                        )
                    except Exception:
                        # fallback: try without opts (older wrappers) or with named TxOpts if available
                        try:
                            resp = await client.send_raw_transaction(signed_tx_bytes)
                        except Exception as inner_e:
                            raise inner_e

                    logger.debug("send_signed_tx_with_retries: raw send response=%r", resp)

                    # Normalize some common response shapes to extract signature
                    sig = None
                    try:
                        if isinstance(resp, dict):
                            # solana-py may return {'result': 'signature'} or {'result': {'value':'sig'}} or {'value': 'sig'}
                            if "result" in resp:
                                r = resp["result"]
                                if isinstance(r, dict):
                                    # nested value form
                                    sig = r.get("value") or r.get("signature") or r.get("txhash")
                                    if isinstance(sig, dict):
                                        sig = sig.get("signature") or sig.get("txhash") or None
                                elif isinstance(r, str):
                                    sig = r
                            elif "value" in resp and isinstance(resp["value"], str):
                                sig = resp["value"]
                            elif resp.get("signature"):
                                sig = resp.get("signature")
                        else:
                            # object-like return: attempt attributes
                            sig = getattr(resp, "value", None) or getattr(resp, "signature", None) or getattr(resp, "result", None)
                            if isinstance(sig, dict):
                                sig = sig.get("value") or sig.get("signature") or None
                    except Exception:
                        logger.debug("send_signed_tx_with_retries: response normalization failed", exc_info=True)

                    if sig:
                        logger.info("send_signed_tx_with_retries: success attempt=%d client=%s sig=%s", attempt, rpc_id or "<primary>", str(sig))
                        return str(sig)

                    # if resp is a string, sometimes it's the signature
                    if isinstance(resp, str) and len(resp) > 8:
                        logger.info("send_signed_tx_with_retries: success (string resp) attempt=%d client=%s", attempt, rpc_id or "<primary>")
                        return resp

                    # No signature -> log and treat as failure for retry
                    last_error = RuntimeError(f"No signature in response: {repr(resp)[:400]}")
                    logger.warning("send_signed_tx_with_retries: attempt=%d client=%s returned no signature; resp=%r", attempt, rpc_id or "<primary>", resp)

                except Exception as e:
                    last_error = e
                    err_text = str(e).lower()
                    is_blockhash = any(k in err_text for k in ("blockhash", "expired", "expired blockhash", "blockhash not found"))
                    is_rate = any(k in err_text for k in ("429", "rate limit", "too many requests", "503", "502"))
                    logger.warning("send_signed_tx_with_retries: attempt=%d client=%s error=%s", attempt, rpc_id or "<primary>", str(e)[:200])
                    if is_blockhash:
                        # Blockhash errors are not recoverable by re-sending the same signed tx.
                        logger.warning("send_signed_tx_with_retries: blockhash/expiry error detected; aborting re-submit.")
                        return None
                    # otherwise we'll retry (and possibly rotate RPC endpoint)

                # small backoff between trying different RPCs within same attempt
                try:
                    await backoff_sleep(1, base=0.5, max_sleep=2.0, jitter=True)
                except Exception:
                    await asyncio.sleep(0.3)

            # End client loop for this attempt — perform exponential backoff before next attempt if not last
            if attempt < max_attempts:
                try:
                    slept = await backoff_sleep(attempt, base=backoff_base, jitter=True)
                    logger.debug("send_signed_tx_with_retries: backing off %.2fs before next attempt", slept)
                except Exception:
                    await asyncio.sleep(min(2.0, backoff_base * attempt))

        # exhausted attempts
        logger.error("send_signed_tx_with_retries: exhausted %d attempts; last_error=%r", max_attempts, last_error)
        if last_error:
            logger.debug("send_signed_tx_with_retries last exception traceback:\n%s", "".join(traceback.format_exception(type(last_error), last_error, last_error.__traceback__)))
        return None

    finally:
        # close clients we created (do not close provided solana_client)
        for c in created_clients:
            try:
                await c.close()
            except Exception:
                pass


async def get_buy_amount(
    token: Optional[Dict[str, Any]],
    wallet_balance: float,
    sol_price: float,
    config: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Determine buy amount (SOL) and USD amount for a given token.

    Returns:
      (amount_sol, usd_amount)
    """
    token = token or {}

    botcfg = (config or {}).get("bot", {}) or {}
    tradingcfg = (config or {}).get("trading", {}) or {}

    dry_run = bool(botcfg.get("dry_run", tradingcfg.get("dry_run", True)))
    per_trade_sol = float(botcfg.get("per_trade_sol", 0.0))  # if >0, cap in SOL

    usd_target = compute_buy_size_usd(wallet_balance, sol_price, config)
    usd_min_config = float(botcfg.get("min_usd_per_buy", tradingcfg.get("min_order_usd", 0.0)))
    usd_target = max(usd_target, usd_min_config)

    amount_sol = 0.0
    try:
        if sol_price and sol_price > 0:
            amount_sol = float(usd_target) / float(sol_price)
    except Exception:
        amount_sol = 0.0

    if per_trade_sol and per_trade_sol > 0:
        amount_sol = min(amount_sol, float(per_trade_sol))

    if not dry_run and wallet_balance is not None:
        amount_sol = min(amount_sol, float(wallet_balance))

    if dry_run and amount_sol <= 0 and usd_target > 0 and sol_price and sol_price > 0:
        amount_sol = float(usd_target) / float(sol_price)

    usd_amount = amount_sol * sol_price if sol_price and sol_price > 0 else 0.0

    try:
        tok_name = token.get("symbol") or token.get("address") or "unknown"
        logger.info(
            "Buy sizing: token=%s wallet_sol=%.6f sol_price=%.4f portfolio_usd=%.2f "
            "usd_target=%.2f amount_sol=%.6f usd_amount=%.2f dry_run=%s",
            tok_name,
            wallet_balance or 0.0,
            sol_price or 0.0,
            (wallet_balance or 0.0) * (sol_price or 0.0),
            usd_target,
            amount_sol,
            usd_amount,
            dry_run,
        )
    except Exception:
        pass

    return float(amount_sol), float(usd_amount)

# -----------------------------------------------------------------------------
# Labels + Scoring
# -----------------------------------------------------------------------------
def normalize_labels(labels_any: Any) -> List[str]:
    out: list[str] = []
    if isinstance(labels_any, dict):
        labels_any = [labels_any]
    if not isinstance(labels_any, (list, tuple)):
        return out
    for item in labels_any:
        if isinstance(item, str):
            out.append((item or "").strip().lower())
        elif isinstance(item, dict):
            name = item.get("label") or item.get("name") or item.get("id")
            if name:
                out.append(str(name).strip().lower())
    return list(dict.fromkeys([x for x in out if x]))

def is_flagged(labels_any: Any, bad_keys: Optional[set[str]] = None) -> bool:
    bad = bad_keys or {"honeypot", "scam", "blocked", "malicious", "dangerous"}
    labs = set(normalize_labels(labels_any))
    return len(bad.intersection(labs)) > 0

def score_token(token: Dict[str, Any]) -> float:
    def _num(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    if is_flagged(token.get("rugcheck_labels")):
        return 0.0
    status = str(token.get("rugcheck_status") or "").strip().lower()
    if status in {"fail", "blocked"}:
        return 0.0

    mc  = _num(token.get("mc") or token.get("market_cap") or token.get("fdv"), 0.0)
    liq = _num(token.get("liquidity"), 0.0)
    vol = _num(token.get("v24hUSD") or token.get("volume_24h"), 0.0)

    pc = token.get("priceChange") or {}
    pc1h  = _num(token.get("price_change_1h",  pc.get("h1",  0.0)), 0.0)
    pc6h  = _num(token.get("price_change_6h",  pc.get("h6",  0.0)), 0.0)
    pc24h = _num(token.get("price_change_24h", pc.get("h24", 0.0)), 0.0)

    created_raw = token.get("pairCreatedAt") or token.get("creation_timestamp") or 0
    try:
        created_ms = int(created_raw)
    except Exception:
        created_ms = 0
    if 0 < created_ms < 10_000_000_000:
        created_ms *= 1000
    age_h = (time.time() - (created_ms / 1000.0)) / 3600.0 if created_ms else None

    MID_MIN, MID_MAX = 100_000.0, 500_000.0
    if mc <= 0:
        mc_s = 0.0
    elif mc < MID_MIN:
        mc_s = _clip(mc / MID_MIN) * 0.8
    elif mc <= MID_MAX:
        mc_s = 1.0
    else:
        mc_s = _clip((MID_MAX / max(mc, 1.0)) ** 0.4)

    LIQ_SOFT = 25_000.0
    VOL_SOFT = 150_000.0
    liq_s = _clip(math.tanh(liq / max(LIQ_SOFT, 1.0)))
    vol_s = _clip(math.tanh(vol / max(VOL_SOFT, 1.0)))

    def _mom(x: float, cap: float) -> float:
        if x <= 0:
            return 0.0
        x = min(x, cap)
        return _clip(math.tanh(x / cap))
    m1  = _mom(pc1h, 40.0)
    m6  = _mom(pc6h, 120.0)
    m24 = _mom(pc24h, 250.0)
    momentum = (0.50 * m1) + (0.30 * m6) + (0.20 * m24)

    if age_h is None:
        age_s = 0.7
    else:
        if age_h < (15.0 / 60.0):
            age_s = 0.2
        elif age_h < 2.0:
            age_s = 0.7
        elif age_h < 72.0:
            age_s = 1.0
        elif age_h < 720.0:
            age_s = 0.8
        else:
            age_s = 0.6

    birdeye_present = bool(token.get("created_at")) or (str(token.get("source") or "").lower() == "birdeye")
    presence_mult = 1.10 if birdeye_present else 0.85

    cats = (token.get("categories") or []) or []
    cat_mult = 1.0
    if "mid_cap" in cats:
        cat_mult *= 1.10
    if "large_cap" in cats:
        cat_mult *= 0.92
    if "low_cap" in cats:
        cat_mult *= 0.97

    base = (0.35 * mc_s) + (0.25 * liq_s) + (0.20 * vol_s) + (0.20 * momentum)
    final = base * age_s * presence_mult * cat_mult
    score = max(0.0, min(100.0, final * 100.0))
    return float(score)

def _birdeye_present(token: Dict[str, Any]) -> bool:
    return bool(token.get("created_at")) or (str(token.get("source") or "").lower() == "birdeye")

def score_token_conservative(token: Dict[str, Any]) -> float:
    if is_flagged(token.get("rugcheck_labels")):
        return 0.0
    status = str(token.get("rugcheck_status") or "").strip().lower()
    if status in {"fail", "blocked"}:
        return 0.0

    def _num(x: Any, d: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return d

    def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    mc  = _num(token.get("mc") or token.get("market_cap") or token.get("fdv"))
    liq = _num(token.get("liquidity"))
    vol = _num(token.get("v24hUSD") or token.get("volume_24h"))

    pc = token.get("priceChange") or {}
    pc1h  = _num(token.get("price_change_1h",  pc.get("h1",  0.0)))
    pc6h  = _num(token.get("price_change_6h",  pc.get("h6",  0.0)))
    pc24h = _num(token.get("price_change_24h", pc.get("h24", 0.0)))

    LO, HI = 50_000.0, 2_500_000.0
    if mc <= 0:
        mc_s = 0.0
    elif mc < LO:
        mc_s = _clip(mc / LO) * 0.7
    elif mc <= HI:
        mc_s = 1.0
    else:
        mc_s = _clip((HI / max(mc, 1.0)) ** 0.25)

    liq_s = _clip(math.tanh(liq / 20_000.0))
    vol_s = _clip(math.tanh(vol / 100_000.0))

    def _mom(x: float, cap: float) -> float:
        if x <= 0:
            return 0.0
        x = min(x, cap)
        return _clip(math.tanh(x / cap))
    m1  = _mom(pc1h, 25.0)
    m6  = _mom(pc6h, 80.0)
    m24 = _mom(pc24h, 200.0)
    momentum = (0.35 * m1) + (0.35 * m6) + (0.30 * m24)

    created_raw = token.get("pairCreatedAt") or token.get("creation_timestamp") or 0
    try:
        created_ms = int(created_raw)
    except Exception:
        created_ms = 0
    if 0 < created_ms < 10_000_000_000:
        created_ms *= 1000
    age_h = (time.time() - (created_ms / 1000.0)) / 3600.0 if created_ms else None
    if age_h is None:
        age_s = 0.7
    elif age_h < 2.0:
        age_s = 0.6
    elif age_h < 168.0:
        age_s = 1.0
    elif age_h < 720.0:
        age_s = 0.8
    else:
        age_s = 0.6

    presence_mult = 1.05 if _birdeye_present(token) else 0.9

    base = (0.30 * mc_s) + (0.35 * liq_s) + (0.25 * vol_s) + (0.10 * momentum)
    final = base * age_s * presence_mult
    return float(max(0.0, min(100.0, final * 100.0)))

# Replace existing score_token_dispatch with the following in utils_exec.py

async def score_token_dispatch(token: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """
    Dispatch scoring mode, plus optional BBands-flip gating/bonus and hard-floor gating.

    New configurable hard-floor gating (in config under `scoring.hard_floor`) supports:
      scoring:
        hard_floor:
          enable: true
          min_liquidity_usd: 5000.0
          min_volume_24h_usd: 100.0
          allow_whitelist: true

    Behavior:
      - If hard_floor.enable is true and either liquidity < min_liquidity_usd OR
        24h volume < min_volume_24h_usd, return a hard-reject score (-1e9)
        and set token["_reject_reason"] for diagnostics.
      - Whitelisted tokens can bypass the floor if allow_whitelist is true.
      - Existing bbands_flip logic unchanged (still can hard-reject bb_short flip).
    """
    def _num(x: Any, d: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(d)

    # Read scoring mode
    mode = str(((cfg or {}).get("scoring", {}) or {}).get("mode") or "midcap").lower()
    if mode == "conservative":
        score = score_token_conservative(token)
    else:
        score = score_token(token)

    # Hard-floor gating config (prefer explicit config, otherwise defaults)
    try:
        scoring_cfg = (cfg or {}).get("scoring", {}) or {}
        hard_cfg = scoring_cfg.get("hard_floor", {}) if isinstance(scoring_cfg, dict) else {}
    except Exception:
        hard_cfg = {}

    hf_enable = bool(hard_cfg.get("enable", True))
    try:
        min_liq = float(hard_cfg.get("min_liquidity_usd", hard_cfg.get("min_liquidity", 5000.0)))
    except Exception:
        min_liq = 5000.0
    try:
        min_vol = float(hard_cfg.get("min_volume_24h_usd", hard_cfg.get("min_volume", 50.0)))
    except Exception:
        min_vol = 50.0
    allow_whitelist = bool(hard_cfg.get("allow_whitelist", True))

    # Determine effective numeric liquidity/volume from many common fields
    liq_candidates = (
        token.get("liquidity"),
        token.get("dex_liquidity"),
        token.get("ray_liquidity_usd"),
        token.get("v24hUSD"),
        token.get("reserve_usd"),
        token.get("pool_liquidity_usd"),
    )
    vol_candidates = (
        token.get("v24hUSD"),
        token.get("volume_24h"),
        token.get("volume24h"),
        token.get("dex_volume_24h"),
    )

    # Best-effort numeric extraction (pick the max sensible value)
    liq_val = 0.0
    for c in liq_candidates:
        try:
            if c is None:
                continue
            v = float(c)
            if v and v > liq_val:
                liq_val = v
        except Exception:
            continue

    vol_val = 0.0
    for c in vol_candidates:
        try:
            if c is None:
                continue
            v = float(c)
            if v and v > vol_val:
                vol_val = v
        except Exception:
            continue

    # Whitelist check (categories or explicit mapping)
    cats = (token.get("categories") or []) or []
    is_whitelisted = ("whitelisted" in cats) or (str(token.get("address") or "") in WHITELISTED_TOKENS)

    # Apply hard-floor gating
    if hf_enable:
        if not (allow_whitelist and is_whitelisted):
            # If liquidity OR volume is below configured minimum, hard-reject
            try:
                liq_bad = (min_liq and float(liq_val) < float(min_liq))
            except Exception:
                liq_bad = False
            try:
                vol_bad = (min_vol and float(vol_val) < float(min_vol))
            except Exception:
                vol_bad = False

            if liq_bad or vol_bad:
                token["_reject_reason"] = f"hard_floor_liq<{min_liq}_vol<{min_vol}"
                try:
                    logger.info(
                        "Rejected by hard-floor gating: addr=%s liq=%.2f vol=%.2f min_liq=%.2f min_vol=%.2f",
                        token.get("address") or token.get("token_address") or token.get("symbol") or "<unknown>",
                        float(liq_val), float(vol_val), float(min_liq), float(min_vol),
                    )
                except Exception:
                    pass
                return -1e9  # hard-reject sentinel

    # Existing bbands_flip logic preserved
    bb_cfg: Dict[str, Any] = (((cfg or {}).get("signals") or {}).get("bbands_flip") or {})
    if bool(bb_cfg.get("enable", True)):
        bb_long = bool(token.get("bb_long"))
        bb_short = bool(token.get("bb_short"))
        if bb_short and bool(bb_cfg.get("reject_shorts", True)):
            token["_reject_reason"] = "bb_short_flip"
            return -1e9  # hard-reject
        if bb_long:
            # Safely coerce numeric inputs to floats/ints with fallbacks
            basis = token.get("bb_basis")
            upper = token.get("bb_upper")
            lower = token.get("bb_lower")

            try:
                width_min = float(bb_cfg.get("min_width", 0.0))
            except Exception:
                width_min = 0.0

            width_pct = 0.0
            try:
                if basis is not None:
                    b = float(basis)
                    if b != 0 and upper is not None and lower is not None:
                        width_pct = (float(upper) - float(lower)) / b
            except Exception:
                width_pct = 0.0

            # Explicitly extract bonus configs and cast safely (avoid `or` inside int())
            long_bonus_val = bb_cfg.get("long_bonus_points", 15)
            try:
                long_bonus = int(long_bonus_val)
            except Exception:
                long_bonus = 15

            narrow_bonus_val = bb_cfg.get("long_bonus_points_narrow", 5)
            try:
                narrow_bonus = int(narrow_bonus_val)
            except Exception:
                narrow_bonus = 5

            bump = long_bonus if width_pct >= width_min else narrow_bonus
            score += bump

    # Keep a small floor for whitelisted items
    minimum_floor = 5.0
    if ("whitelisted" in cats) and score < minimum_floor:
        score = minimum_floor

    return float(score)

def log_scoring_telemetry(tokens: List[Dict[str, Any]], where: str = "eligible") -> None:
    if not tokens:
        logger.info(f"[SCORE] {where}: empty set")
        return
    total = 0.0
    have_be = 0
    n = 0
    for t in tokens:
        try:
            total += float(t.get("score", 0.0))
            have_be += 1 if _birdeye_present(t) else 0
            n += 1
        except Exception:
            continue
    avg = total / max(1, n)
    pct = (have_be / max(1, n)) * 100.0
    logger.info(f"[SCORE] {where}: avg={avg:.1f} birdeye_coverage={have_be}/{n} ({pct:.1f}%)")

# ---------- formatting helpers ----------------------------------------------------------
def format_market_cap(value: float) -> str:
    try:
        num = float(value)
    except Exception:
        return "N/A"

    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif abs_num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif abs_num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"
    
def format_usd(value: float | int | None, *, compact: bool = False, decimals: int = 2) -> str:
    try:
        num = float(value)
    except Exception:
        return "$0.00"

    if not math.isfinite(num):
        return "$0.00"

    sign = "-" if num < 0 else ""
    n = abs(num)

    if compact:
        if n >= 1_000_000_000:
            return f"{sign}${n/1_000_000_000:.{decimals}f}B"
        if n >= 1_000_000:
            return f"{sign}${n/1_000_000:.{decimals}f}M"
        if n >= 1_000:
            return f"{sign}${n/1_000:.{decimals}f}K"
        return f"{sign}${n:.{decimals}f}"

    return f"{sign}${n:,.{decimals}f}"

def format_ts_human(ts: int | float | str | None, *, with_time: bool = True) -> str:
    if ts is None:
        return "-"
    try:
        t = float(ts)
    except Exception:
        return "-"
    if 0 < t < 10_000_000_000:
        dt = datetime.fromtimestamp(t)
    else:
        dt = datetime.fromtimestamp(t / 1000.0)
    return dt.strftime("%Y-%m-%d %H:%M:%S") if with_time else dt.strftime("%Y-%m-%d")
    
def fmt_usd(value):
    return format_usd(value, compact=True, decimals=2)

def fmt_ts(ts):
    return format_ts_human(ts, with_time=True)

# ----------------------------------------------------------------------------- 
# Buy amount sizing (per-cycle and per-asset)
# -----------------------------------------------------------------------------

async def get_buy_amount(
    token: Optional[Dict[str, Any]] = None,
    wallet_balance: float | None = None,
    sol_price: float | None = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """
    Determine how much SOL to spend on a single buy.

    Supports both:
      - Fixed cycle amount (default behavior)
      - Per-asset dynamic sizing (if enabled and token data provided)

    Returns: (amount_sol: float, usd_equivalent: float)
    """
    try:
        bot_cfg = (config or {}).get("bot", {}) or {}
        sol_price = float(sol_price or 1.0)
        wallet_balance = float(wallet_balance or 0.0)

        # Default fixed buy size (in SOL)
        default_buy_sol = float(bot_cfg.get("buy_amount_sol", 0.1))

        # If per-asset sizing is disabled or no token data, use fixed amount
        if not bot_cfg.get("per_asset_sizing", False) or not token:
            amount_sol = default_buy_sol
        else:
            # Example dynamic logic — adjust to your strategy
            # Here: scale based on market cap tier (smaller cap = smaller position)
            try:
                mcap = float(token.get("market_cap") or token.get("fdv") or 0)
                if mcap <= 50_000:
                    amount_sol = default_buy_sol * 0.5
                elif mcap <= 200_000:
                    amount_sol = default_buy_sol * 0.8
                elif mcap <= 1_000_000:
                    amount_sol = default_buy_sol
                else:
                    amount_sol = default_buy_sol * 1.2
            except Exception:
                amount_sol = default_buy_sol

        # Safety caps
        max_buy_sol = float(bot_cfg.get("max_buy_amount_sol", 1.0))
        min_buy_sol = float(bot_cfg.get("min_buy_amount_sol", 0.01))

        amount_sol = max(min_buy_sol, min(amount_sol, max_buy_sol))
        amount_sol = min(amount_sol, wallet_balance * 0.9)  # never use >90% of balance

        usd_amount = amount_sol * sol_price

        return float(amount_sol), float(usd_amount)

    except Exception as e:
        logger.warning("get_buy_amount failed, using fallback 0.05 SOL: %s", e)
        fallback_sol = 0.05
        return fallback_sol, fallback_sol * float(sol_price or 170.0)
    
# -----------------------------------------------------------------------------
# Jupiter swap execution (robust)
# -----------------------------------------------------------------------------
async def execute_jupiter_swap_with_logging(
    quote: dict,
    user_pubkey: str,
    wallet,
    solana_client,
    *,
    session=None,
    compute_unit_price_micro_lamports: int | None = None,
    wrap_and_unwrap_sol: bool = True,
    use_shared_accounts: bool = True,
    rpc_endpoints: Iterable[str] | None = None,
    max_send_attempts: int = 4,
) -> str | None:
    """
    Robust wrapper to execute a Jupiter swap-like quote.

    Behavior:
      - Inspect quote for obvious errors and route/impact heuristics and abort early if unsafe.
      - If signed tx bytes present, send via send_signed_tx_with_retries.
      - If unsigned transaction bytes present and wallet supports sign, attempt best-effort re-sign.
      - Returns txid string on success or None on failure.
    """
    last_exc = None

    def _env_float(name: str, default: float) -> float:
        try:
            v = os.getenv(name)
            if v is None:
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    try:
        if not isinstance(quote, dict):
            logger.error("execute_jupiter_swap_with_logging: quote is not dict (type=%s)", type(quote))
            return None

        # ---------- Pre-buy probe (quote-based heuristics) ----------
        try:
            # 1) explicit error fields -> abort
            if quote.get("error") or quote.get("errorCode"):
                logger.error(
                    "Preflight Jupiter probe indicates not tradable: %s",
                    remove_none_keys({"error": quote.get("error"), "errorCode": quote.get("errorCode")}),
                )
                return None

            # 2) missing/empty route plan -> abort
            route_plan = quote.get("routePlan") or quote.get("route_plan") or quote.get("route") or None
            if route_plan in (None, [], ""):
                logger.error("Preflight Jupiter probe: no route plan present in quote; aborting. Quote keys: %s", list(quote.keys()))
                return None

            # 3) price impact guard (env JUP_MAX_PRICE_IMPACT_PCT; default 5%)
            price_impact_pct = None
            try:
                for f in ("priceImpactPct", "price_impact_pct", "priceImpact"):
                    if f in quote and quote.get(f) is not None:
                        price_impact_pct = float(quote.get(f) or 0.0)
                        break
            except Exception:
                price_impact_pct = None
            max_allowed_pct = _env_float("JUP_MAX_PRICE_IMPACT_PCT", 5.0)
            if price_impact_pct is not None and price_impact_pct > float(max_allowed_pct):
                logger.warning("Preflight Jupiter probe: price impact %.3f%% > allowed %.3f%%; aborting", price_impact_pct, float(max_allowed_pct))
                return None

            # 4) minimal USD swap value guard (env JUP_MIN_SWAP_USD; default 1.0)
            swap_usd_val = 0.0
            try:
                swap_usd_val = float(quote.get("swapUsdValue") or quote.get("swap_usd_value") or quote.get("swapUsd") or 0.0)
            except Exception:
                swap_usd_val = 0.0
            min_swap_usd = _env_float("JUP_MIN_SWAP_USD", 1.0)
            if swap_usd_val is not None and swap_usd_val < float(min_swap_usd):
                logger.warning("Preflight Jupiter probe: swapUsdValue %.2f < min_allowed %.2f; aborting", float(swap_usd_val or 0.0), float(min_swap_usd))
                return None

            # 5) quick AMM-level heuristics if present
            try:
                amms = quote.get("mostReliableAmmsQuoteReport") or quote.get("amms") or None
                if isinstance(amms, dict):
                    if (not amms.get("legs") and not amms.get("route")) or amms.get("outAmount") in (0, "0", None):
                        logger.debug("Preflight Jupiter probe: amms report empty/insufficient liquidity: %s", amms)
                        return None
            except Exception:
                pass

        except Exception as e_probe:
            # Defensive: if probe logic errors, do not block execution; record and continue
            last_exc = e_probe
            logger.debug("Preflight probe inspection failed; continuing with send attempts. probe_error=%s", e_probe, exc_info=True)

        # ---------- Locate signed tx bytes (common Jupiter keys) ----------
        signed_bytes = None
        key_used = None
        for k in ("swapTransaction", "swap_tx", "signed_tx", "raw_transaction", "signedTransaction"):
            v = quote.get(k)
            if not v:
                continue
            key_used = k
            if isinstance(v, (bytes, bytearray)):
                signed_bytes = bytes(v)
            else:
                try:
                    s = str(v)
                    if s.startswith("base64,"):
                        s = s.split("base64,", 1)[1]
                    signed_bytes = base64.b64decode(s)
                except Exception as e:
                    last_exc = e
                    logger.debug("execute_jupiter_swap_with_logging: failed to decode %s: %s", k, e, exc_info=True)
                    signed_bytes = None
            break

        # -------------- If we have signed bytes -> robust send with retries --------------
        if signed_bytes:
            try:
                try:
                    first32 = binascii.hexlify(signed_bytes[:32]).decode()
                except Exception:
                    first32 = "<hex-failed>"
                logger.debug(
                    "execute_jupiter_swap_with_logging: found signed payload key=%s first32=%s size=%d",
                    key_used, first32, len(signed_bytes),
                )

                txid = await send_signed_tx_with_retries(
                    signed_bytes,
                    solana_client,
                    session=session,
                    rpc_endpoints=rpc_endpoints,
                    max_attempts=max_send_attempts,
                    skip_preflight=False,
                    preflight_commitment="confirmed",
                )
                if txid:
                    return txid
                last_exc = Exception("send_signed_tx_with_retries returned None")
            except Exception as e_send:
                last_exc = e_send
                logger.debug("execute_jupiter_swap_with_logging: send_signed_tx_with_retries failed", exc_info=True)

        # -------------- If no signed bytes, attempt best-effort re-sign (legacy Transaction) --------------
        unsigned_b64 = None
        ub_key = None
        for k in ("unsignedTransaction", "transaction", "tx", "swapTransactionUnsigned"):
            if quote.get(k):
                unsigned_b64 = quote.get(k)
                ub_key = k
                break

        if unsigned_b64 and wallet is not None:
            try:
                raw_unsigned = bytes(unsigned_b64) if isinstance(unsigned_b64, (bytes, bytearray)) else base64.b64decode(str(unsigned_b64))
                txobj = None
                try:
                    from solana.transaction import Transaction as LegacyTransaction  # type: ignore
                    try:
                        txobj = LegacyTransaction.deserialize(raw_unsigned)  # type: ignore[attr-defined]
                    except Exception:
                        txobj = None
                except Exception:
                    txobj = None

                if txobj and hasattr(wallet, "sign_transaction"):
                    # refresh blockhash if possible
                    bh = await _extract_blockhash_from_client(solana_client)
                    try:
                        if bh:
                            try:
                                txobj.recent_blockhash = bh  # type: ignore[attr-defined]
                            except Exception:
                                pass
                        # wallet.sign_transaction may be sync or async
                        signed = await wallet.sign_transaction(txobj) if asyncio.iscoroutinefunction(wallet.sign_transaction) else wallet.sign_transaction(txobj)
                        if hasattr(signed, "serialize"):
                            signed_bytes = signed.serialize()
                        else:
                            try:
                                signed_bytes = bytes(signed)
                            except Exception:
                                signed_bytes = None

                        if signed_bytes:
                            txid = await send_signed_tx_with_retries(
                                signed_bytes,
                                solana_client,
                                session=session,
                                rpc_endpoints=rpc_endpoints,
                                max_attempts=max_send_attempts,
                            )
                            if txid:
                                return txid
                    except Exception as e_sign:
                        last_exc = e_sign
                        logger.debug("execute_jupiter_swap_with_logging: signing/resend path failed", exc_info=True)
            except Exception as e_dec:
                last_exc = e_dec
                logger.debug("execute_jupiter_swap_with_logging: unsigned tx path failed to decode/sign", exc_info=True)

        # If we reach here, nothing succeeded. Final diagnostics & return.
        logger.error("JUPITER-SEND: all attempts exhausted; last error: %s", repr(last_exc) if last_exc else "none")
        if last_exc:
            try:
                logger.debug("JUPITER-SEND last exception traceback:\n%s", "".join(traceback.format_exception(type(last_exc), last_exc, last_exc.__traceback__)))
            except Exception:
                pass

        logger.error("execute_jupiter_swap_with_logging: no signed payload found or all send attempts failed; quote_keys=%s", list(quote.keys()))
        return None

    except Exception as e:
        last_exc = e
        logger.exception("execute_jupiter_swap_with_logging: unexpected error")
        logger.error("JUPITER-SEND: all attempts exhausted; last error: %s", repr(last_exc) if last_exc else "none")
        try:
            logger.debug("JUPITER-SEND last exception traceback:\n%s", "".join(traceback.format_exception(type(last_exc), last_exc, last_exc.__traceback__)))
        except Exception:
            pass
        return None

# Compatibility alias — keep old import paths working
# Backwards-compatible alias (preserve previous import name)
async def execute_jupiter_swap(*a, **kw):
    """Compatibility wrapper for older callers expecting execute_jupiter_swap."""
    return await execute_jupiter_swap_with_logging(*a, **kw)

# -----------------------------------------------------------------------------
# Secret management (first-run prompt + persistence)
# -----------------------------------------------------------------------------
def _is_json_64int_secret(s: str) -> bool:
    try:
        arr = json.loads(s)
        return isinstance(arr, list) and len(arr) == 64 and all(isinstance(x, int) and 0 <= x <= 255 for x in arr)
    except Exception:
        return False

def _looks_base58_secret(s: str) -> bool:
    return isinstance(s, str) and len(s.strip()) >= 64 and s.strip().isalnum()

def validate_solana_secret(s: str) -> bool:
    s = (s or "").strip()
    return _is_json_64int_secret(s) or _looks_base58_secret(s)

def validate_birdeye_key(s: str) -> bool:
    return isinstance(s, str) and len(s.strip()) >= 12

def get_secret_from_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v and isinstance(v, str) and v.strip():
        return v.strip()
    data = _parse_dotenv_file(ENV_PATH)
    v2 = data.get(name)
    return v2.strip() if isinstance(v2, str) and v2.strip() else None

def set_secret(name: str, value: str) -> None:
    os.environ[name] = value
    _write_env_key(name, value)

def ensure_required_secrets(interactive: bool = True) -> Dict[str, bool]:
    import sys

    sol = get_secret_from_env("SOLANA_PRIVATE_KEY")
    bir = get_secret_from_env("BIRDEYE_API_KEY")

    have_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()

    flags = {"SOLANA_PRIVATE_KEY": False, "BIRDEYE_API_KEY": False}
    if sol and validate_solana_secret(sol):
        flags["SOLANA_PRIVATE_KEY"] = True
    if bir and validate_birdeye_key(bir):
        flags["BIRDEYE_API_KEY"] = True

    if interactive and have_tty:
        if not flags["SOLANA_PRIVATE_KEY"]:
            print("\n=== First-run setup: SOLANA_PRIVATE_KEY ===")
            print("Paste your Solana secret EITHER as JSON array of 64 ints OR as a base58 64-byte secret.")
            sol_in = input("SOLANA_PRIVATE_KEY: ").strip()
            if validate_solana_secret(sol_in):
                set_secret("SOLANA_PRIVATE_KEY", sol_in)
                flags["SOLANA_PRIVATE_KEY"] = True
            else:
                print("(!) That doesn't look valid. You can also put it in your .env later.")

        if not flags["BIRDEYE_API_KEY"]:
            print("\n=== First-run setup: BIRDEYE_API_KEY ===")
            bir_in = input("BIRDEYE_API_KEY: ").strip()
            if validate_birdeye_key(bir_in):
                set_secret("BIRDEYE_API_KEY", bir_in)
                flags["BIRDEYE_API_KEY"] = True
            else:
                print("(!) That doesn't look valid. You can also put it in your .env later.")

    return flags

# -----------------------------------------------------------------------------
# Config helpers (Dexscreener post-cap)
# -----------------------------------------------------------------------------
def get_dex_post_cap(config: Dict) -> int:
    try:
        disc = dict((config or {}).get("discovery") or {})
    except Exception:
        disc = {}
    try:
        cap = int(disc.get("dexscreener_post_cap", 0))
    except Exception:
        cap = 0
    if cap <= 0:
        try:
            cap = int(os.getenv("DEX_MAX") or os.getenv("DEXSCREENER_MAX", "0"))
        except Exception:
            cap = 0
    return max(0, cap)

# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------
__all__ = [
    "WHITELISTED_TOKENS",
    "price_cache",
    "token_balance_cache",
    "token_account_existence_cache",
    "load_config",
    "setup_logging",
    "ENV_PATH",
    "_parse_dotenv_file",
    "_write_env_key",
    "get_rugcheck_token",
    "get_rugcheck_headers",
    "get_rugcheck_token_async",
    "add_to_blacklist",
    "deduplicate_tokens",
    "remove_none_keys",
    "_best_first",
    "custom_json_encoder",
    "safe_create_task",
    # <<< OPTIMAL PLACEMENT FOR NEW SIZING HELPERS >>>
    "get_buy_amount",          # ← Insert here
    "compute_buy_size_usd",    # ← Insert here
    # <<<
    "backoff_sleep",
    "CircuitBreaker429",
    "normalize_labels",
    "is_flagged",
    "score_token",
    "format_market_cap",
    "format_usd",
    "format_ts_human",
    "execute_jupiter_swap",
    # secret management
    "get_secret_from_env",
    "set_secret",
    "validate_solana_secret",
    "validate_birdeye_key",
    "ensure_required_secrets",
    # scoring selector + telemetry
    "score_token_dispatch",
    "score_token_conservative",
    "log_scoring_telemetry",
    # config helpers
    "get_dex_post_cap",
    # convenience shorthands
    "fmt_usd",
    "fmt_ts",
    # volume & pattern helpers (re-exports)
    "rolling_median",
    "rolling_mad",
    "is_volume_spike",
    "is_evening_star",
    "is_morning_star",
    "is_three_black_crows",
    "is_three_white_soldiers",
    "is_hammer",
    "is_inverted_hammer",
    "is_bullish_engulfing",
    "is_bearish_engulfing",
    "is_doji",
]

# -----------------------------------------------------------------------------
# Simple unit tests (run when executed directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import unittest

    class UtilsTests(unittest.TestCase):
        def test_rolling_median_basic(self):
            seq = [1, 3, 2, 5]
            expected = [1.0, 2.0, 2.5, 3.5]
            res = rolling_median(seq, window=2)
            self.assertEqual(len(res), len(expected))
            for a, b in zip(res, expected):
                self.assertAlmostEqual(float(a), float(b), places=6)

        def test_rolling_mad_basic(self):
            seq = [1, 2, 3]
            res = rolling_mad(seq, window=2)
            # Accept either computed mad or zeros; ensure length matches and numeric
            self.assertEqual(len(res), len(seq))
            for v in res:
                self.assertIsInstance(v, float)

        def test_is_volume_spike_true(self):
            seq = [1, 1, 1, 10]
            self.assertTrue(is_volume_spike(seq, window=3, factor=3.0))

        def test_is_volume_spike_short_sequence(self):
            seq = [5]
            # fallback heuristic: last >= factor
            self.assertTrue(is_volume_spike(seq, window=3, factor=3.0))

        def test_classify_patterns_arrays_returns_list(self):
            res = classify_patterns_arrays(None)
            self.assertIsInstance(res, list)

    unittest.main()