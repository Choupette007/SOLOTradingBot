from __future__ import annotations

import os
import time
import asyncio
import logging
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone
import re
import signal
import random
from typing import TYPE_CHECKING, Any, Dict, List, Set, Optional, Callable, Iterable
import concurrent.futures
import sqlite3
from pathlib import Path


# Ensure module-level logger exists before any fallback logging
logger: logging.Logger = logging.getLogger("TradingBot")

import atexit
try:
    # Defensive import: prefer the concrete DB module so we can close the shared connection
    # at process exit from synchronous contexts (signal handlers, tests, etc.).
    import solana_trading_bot_bundle.trading_bot.database as db
except Exception:
    db = None


def _close_shared_db_at_exit() -> None:
    """
    Best-effort synchronous cleanup at interpreter exit to ensure the module-shared
    aiosqlite connection is closed and its worker thread can terminate.
    """
    if db is None:
        return
    try:
        # Use the synchronous wrapper so this is safe from non-async exit handlers.
        close_sync = getattr(db, "close_shared_db_sync", None)
        if callable(close_sync):
            close_sync()
        else:
            # Fallback: try to run the async close if no sync wrapper is available.
            try:
                asyncio.run(getattr(db, "close_shared_db")())
            except Exception:
                # best-effort; ignore
                pass
    except Exception:
        try:
            logger.exception("Failed to close shared DB at process exit")
        except Exception:
            pass


atexit.register(_close_shared_db_at_exit)

from .canonical import extract_canonical_mint

# Safe optional import of persisted-shortlist validator (normalises + re-applies hard-floor)
try:
    from .eligibility import validate_persisted_shortlist  # type: ignore
except Exception:
    validate_persisted_shortlist = None  # type: ignore

def setup_logging(config: Optional[Dict] = None) -> None:
    """
    Best-effort logging setup.
    - If a project-specific setup_logging exists (packaged or local), call it.
    - Otherwise perform a minimal logging.basicConfig so the module logs sensibly.
    This is defensive and never raises.
    """
    try:
        # Try packaged helper first
        from solana_trading_bot_bundle.trading_bot.logging_setup import setup_logging as _setup  # type: ignore
        try:
            _setup(config)
            return
        except Exception:
            # If packaged helper raises, continue to local attempt/fallback
            pass
    except Exception:
        pass

    try:
        # Try local helper
        from .logging_setup import setup_logging as _setup_local  # type: ignore
        try:
            _setup_local(config)
            return
        except Exception:
            pass
    except Exception:
        pass

    # Final safe fallback: configure a simple root handler if none exist.
    try:
        root = logging.getLogger()
        if not root.handlers:
            level = logging.INFO
            if isinstance(config, dict):
                try:
                    lvl = (config.get("logging") or {}).get("level")
                    if isinstance(lvl, str) and hasattr(logging, lvl.upper()):
                        level = getattr(logging, lvl.upper())
                except Exception:
                    pass
            logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        # Ensure module logger inherits or uses a reasonable level
        try:
            logger.setLevel(logger.level or logging.INFO)
        except Exception:
            pass
    except Exception:
        # swallow any logging setup errors - the rest of the module should still run
        try:
            logger.debug("Fallback setup_logging failed", exc_info=True)
        except Exception:
            pass


def log_scoring_telemetry(tokens: Optional[List[Dict]] = None, *, where: str = "") -> None:
    """
    Lightweight telemetry/log helper used to emit summary info about candidate scoring.
    Non-fatal, best-effort: prefer to do a compact INFO-level summary and a DEBUG-level
    sample listing. If a richer telemetry helper exists in the package, callers will
    typically have wired that; this fallback ensures name is defined and useful.
    """
    try:
        toks = list(tokens or [])
        n = len(toks)
        try:
            logger.info("Scoring telemetry (%s): %d token(s) evaluated", where or "unknown", n)
        except Exception:
            pass

        if n == 0:
            return

        # Count categories if present
        try:
            from collections import Counter
            cats = []
            for t in toks:
                try:
                    cs = t.get("categories") or []
                    if isinstance(cs, (list, tuple)):
                        cats.extend([str(c) for c in cs])
                except Exception:
                    continue
            if cats:
                cnt = Counter(cats)
                # show top 5 categories
                most = ", ".join(f"{k}:{v}" for k, v in cnt.most_common(5))
                logger.info("Scoring categories (%s): %s", where or "unknown", most)
        except Exception:
            pass

        # Show top 5 by score if available (defensive)
        try:
            scored = [t for t in toks if isinstance(t, dict) and t.get("score") is not None]
            if scored:
                scored.sort(key=lambda x: float(x.get("score", 0) or 0.0), reverse=True)
                top = scored[:5]
                s_preview = ", ".join(
                    f"{(t.get('symbol') or t.get('address') or 'UNK')}({float(t.get('score') or 0):.2f})"
                    for t in top
                )
                logger.info("Top scored (%s): %s", where or "unknown", s_preview)
            else:
                # fallback: show symbol/address of first up to 5 tokens
                preview = ", ".join(
                    f"{(t.get('symbol') or t.get('address') or 'UNK')}" for t in toks[:5]
                )
                logger.debug("Sample tokens (%s): %s", where or "unknown", preview)
        except Exception:
            pass

    except Exception:
        try:
            logger.debug("log_scoring_telemetry failed", exc_info=True)
        except Exception:
            pass

# Best-effort binding for shared helpers (load_config, caches, sizing, jupiter execution, dedupe)
# Place this near the top of trading.py after the other imports so callers like load_config() exist.
try:
    # Prefer relative (local) utils
    from .utils_exec import (
        load_config,
        price_cache,
        WHITELISTED_TOKENS,
        token_balance_cache,
        get_buy_amount,
        execute_jupiter_swap,
        deduplicate_tokens,
    )
except Exception:
    try:
        # Fallback to packaged path if running from the bundle layout
        from solana_trading_bot_bundle.trading_bot.utils_exec import (
            load_config,
            price_cache,
            WHITELISTED_TOKENS,
            token_balance_cache,
            get_buy_amount,
            execute_jupiter_swap,
            deduplicate_tokens,
        )
    except Exception:
        # Conservative fallbacks so trading.py runs in dry-run/test/watch-only environments.
        import asyncio
        from typing import Any, Dict, List, Optional

        def load_config(path: Optional[str] = None) -> Dict:
            # Minimal empty config to avoid KeyError; operators should replace with real loader.
            return {}

        # lightweight in-memory caches (dict-like)
        price_cache = {}
        token_balance_cache = {}
        WHITELISTED_TOKENS = set()

        async def get_buy_amount(
            token: Optional[Dict[str, Any]] = None,
            wallet_balance: float | None = None,
            sol_price: float | None = None,
            config: Optional[Dict[str, Any]] = None,
        ):
            """
            Conservative async fallback: return tiny SOL amount and corresponding USD amount.
            Signature mirrors utils.get_buy_amount which returns (amount_sol, usd_amount).
            """
            try:
                sol = float(sol_price or 1.0)
            except Exception:
                sol = 1.0
            amount_sol = 0.01
            usd_amount = amount_sol * sol
            return float(amount_sol), float(usd_amount)

        
        async def execute_jupiter_swap(
            quote: Dict,
            user_pubkey: str,
            wallet: Any,
            solana_client: Any,
            *args,
            **kwargs,
        ) -> Optional[str]:
            """
            Dry-run / fallback stub for execute_jupiter_swap.

            Accepts the real function signature plus optional keyword args (session, compute_unit_price_micro_lamports,
            wrap_and_unwrap_sol, use_shared_accounts) so callers can pass them without causing TypeError when the
            real utils implementation isn't available (e.g., in test/dry-run environments).
            """
            return "SIMULATED-JUPITER-TX"

        def deduplicate_tokens(tokens: List[Dict]) -> List[Dict]:
            out: List[Dict] = []
            seen = set()
            for t in (tokens or []):
                try:
                    addr = t.get("address") or t.get("token_address") or t.get("mint") or ""
                    if not addr:
                        # fallback: use repr to dedupe if no address
                        key = repr(t)
                    else:
                        key = str(addr)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(t)
                except Exception:
                    try:
                        out.append(t)
                    except Exception:
                        pass
            return out


# --- Solana AsyncClient (runtime-safe, late import) --------------------------
if TYPE_CHECKING:
    from solana.rpc.async_api import AsyncClient as AsyncClientType
else:
    AsyncClientType = Any  # runtime stub; actual type resolved at import time


def _new_async_client(rpc_url: str) -> AsyncClientType:
    try:
        from solana.rpc.async_api import AsyncClient as _AC  # type: ignore
        return _AC(rpc_url)
    except ImportError:
        # If operator explicitly allowed watch-only/dry-run, return a minimal stub
        if str(os.getenv("FORCE_WATCH_ONLY", "")).strip().lower() in ("1", "true") or str(
            os.getenv("DRY_RUN", "0")
        ).lower() in ("1", "true", "yes"):

            class _StubValue:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            class _StubAsyncClient:
                async def get_balance(self, *a, **k):
                    return _StubValue(value=0)

                async def get_latest_blockhash(self, *a, **k):
                    return _StubValue(context=_StubValue(slot=0), value=_StubValue(blockhash="0" * 44))

                async def get_version(self, *a, **k):
                    return {"solana-core": "stub"}

                async def get_token_account_balance(self, *a, **k):
                    return _StubValue(value=_StubValue(ui_amount=0.0))

                async def get_token_supply(self, *a, **k):
                    return _StubValue(value=_StubValue(decimals=9))

                async def get_token_largest_accounts(self, *a, **k):
                    return _StubValue(value=[])

                async def close(self):
                    return None

                def __getattr__(self, name):
                    # fail fast & clear if later callers use unsupported RPC surface
                    raise AttributeError(
                        f"_StubAsyncClient missing RPC method '{name}' — add it to stub if required"
                    )

            return _StubAsyncClient()
        # otherwise re-raise the ImportError
        raise


# -------------------------- DB helpers (robust import with exception logging) -------------------------
_persist_discovered_tokens: Optional[Callable[..., Any]] = None

# Attempt to acquire a module-like `_db` object (prefer relative import)
_db: Optional[Any] = None
try:
    from . import database as _db  # type: ignore
except Exception:
    try:
        from solana_trading_bot_bundle.trading_bot import database as _db  # type: ignore
    except Exception:
        _db = None

# Robust DB helper binder to avoid import-time circular/ordering failures.
# Strategy:
#  - Install conservative no-op stubs for the names trading.py expects.
#  - Attempt to bind real implementations from the module object `_db` if present.
#  - If no module is available, construct a minimal module-like fallback pointing at
#    the no-op stubs and log a single warning.
async def _noop_async(*a, **k):
    return None


def _noop_sync(*a, **k):
    return None

# sync helper that copies canonicalized rows from eligible_tokens into discovered_tokens whenever eligible_tokens is updated
async def _sync_eligible_to_discovered_once(db_path: str | Path, *, logger: logging.Logger | None = None) -> int:
    """Copy missing rows from eligible_tokens -> discovered_tokens using canonical_address where available.

    Returns the number of inserted rows.
    This is intentionally conservative: it only INSERTs rows that do not already exist in discovered_tokens.
    """
    if logger is None:
        _logger = logging.getLogger("TradingBot")
    else:
        _logger = logger

    dbp = str(db_path)
    inserted = 0
    try:
        conn = sqlite3.connect(dbp)
        cur = conn.cursor()

        # Ensure discovered_tokens table exists
        try:
            cur.execute("SELECT 1 FROM discovered_tokens LIMIT 1;")
        except sqlite3.OperationalError:
            _logger.debug("discovered_tokens table missing; skipping eligible->discovered sync")
            conn.close()
            return 0

        # Determine discovered_tokens columns available
        discovered_cols = [r[1] for r in cur.execute("PRAGMA table_info(discovered_tokens);").fetchall()]

        # We'll populate a minimal safe set of columns if present
        candidate_cols = []
        for c in ("address", "canonical_address", "symbol", "source", "created_at"):
            if c in discovered_cols:
                candidate_cols.append(c)
        if not candidate_cols:
            _logger.debug("discovered_tokens has no target columns; skipping sync")
            conn.close()
            return 0

        col_list = ",".join(candidate_cols)
        placeholders = ",".join("?" for _ in candidate_cols)

        # Select eligible tokens (address, canonical_address, symbol, source, created_at if present)
        eligible_cols = [r[1] for r in cur.execute("PRAGMA table_info(eligible_tokens);").fetchall()]
        select_cols = []
        for c in ("address", "canonical_address", "symbol", "source", "created_at"):
            if c in eligible_cols:
                select_cols.append(c)
        if not select_cols:
            _logger.debug("eligible_tokens table has no selectable columns; skipping sync")
            conn.close()
            return 0

        sel_sql = "SELECT " + ", ".join(select_cols) + " FROM eligible_tokens"
        rows = list(cur.execute(sel_sql).fetchall())

        # For each eligible row, determine a canonical key to check existence and insert if missing
        for row in rows:
            # Map selected columns to their names
            rec = dict(zip(select_cols, row))
            raw_addr = rec.get("address")
            canon = rec.get("canonical_address") or extract_canonical_mint(raw_addr)

            # Skip if neither address nor canonical exists
            lookup_cond = None
            lookup_val = None
            if canon:
                lookup_cond = "canonical_address = ?"
                lookup_val = canon
            elif raw_addr:
                lookup_cond = "address = ?"
                lookup_val = raw_addr
            else:
                # nothing to insert
                continue

            exists = cur.execute(f"SELECT 1 FROM discovered_tokens WHERE {lookup_cond} LIMIT 1", (lookup_val,)).fetchone()
            if exists:
                continue

            # Build insert values in same column order
            values = []
            for c in candidate_cols:
                if c == "address":
                    values.append(raw_addr)
                elif c == "canonical_address":
                    values.append(canon)
                else:
                    values.append(rec.get(c))
            try:
                cur.execute(f"INSERT INTO discovered_tokens ({col_list}) VALUES ({placeholders})", values)
                inserted += 1
            except Exception as e:
                _logger.debug("failed to insert discovered_tokens row for %s: %s", lookup_val, e)

        conn.commit()
        conn.close()
    except Exception as e:
        logging.getLogger("TradingBot").exception("eligible->discovered sync failed: %s", e)
        return 0

    if inserted:
        logging.getLogger("TradingBot").info("eligible->discovered sync: inserted %d rows into discovered_tokens", inserted)
    return inserted

# Static analyzer stubs for runtime-bound helpers (place after _noop_async/_noop_sync)
from typing import Any, Callable, Awaitable, Optional, Dict, Tuple

# The real implementations are bound at runtime from the database/module.
# These stubs simply help static analyzers (Pylance/pyright) and do not change runtime behavior.
persist_eligible_shortlist: Callable[..., Awaitable[int]] = _noop_async
persist_discovered_tokens: Callable[..., Awaitable[int]] = _noop_async

get_token_trade_status: Callable[..., Awaitable[Optional[Dict[str, Any]]]] = _noop_async
get_cached_token_data: Callable[..., Awaitable[Optional[Dict[str, Any]]]] = _noop_async
get_cached_creation_time: Callable[..., Awaitable[Optional[Any]]] = _noop_async

cache_token_data: Callable[..., Awaitable[None]] = _noop_async
cache_creation_time: Callable[..., Awaitable[None]] = _noop_async

db_add_to_blacklist: Callable[..., Awaitable[None]] = _noop_async
mark_token_sold: Callable[..., Awaitable[None]] = _noop_async
update_token_record: Callable[..., Awaitable[None]] = _noop_async

# Bulk/other DB helpers (if referenced elsewhere)
bulk_upsert_tokens: Callable[..., Awaitable[None]] = _noop_async
persist_eligible_shortlist: Callable[..., Awaitable[None]] = _noop_async
get_open_positions: Callable[..., Awaitable[list]] = _noop_async

# Typing alias import fix (if code uses `Tuple` in annotations)
# from typing import Tuple  # already imported above

# Map of expected DB symbol names -> global name used in this module.
_DB_ALIAS: Dict[str, str] = {
    "persist_eligible_shortlist": "persist_eligible_shortlist",
    "persist_discovered_tokens": "persist_discovered_tokens",
    "get_token_trade_status": "get_token_trade_status",
    "get_cached_token_data": "get_cached_token_data",
    "get_cached_creation_time": "get_cached_creation_time",
    "cache_token_data": "cache_token_data",
    "cache_creation_time": "cache_creation_time",
    "update_token_record": "update_token_record",
    # aliasing DB names to the trading module globals expected
    "add_to_blacklist": "db_add_to_blacklist",
    "load_blacklist": "db_load_blacklist",
    "mark_token_sold": "mark_token_sold",
    "init_db": "init_db",
    "clear_expired_blacklist": "clear_expired_blacklist",
    "review_blacklist": "review_blacklist",
    # bulk helpers that trading calls indirectly
    "bulk_upsert_tokens": "bulk_upsert_tokens",
    "bulk_upsert_eligible_tokens": "bulk_upsert_eligible_tokens",
    "ensure_discovered_tokens_schema": "ensure_discovered_tokens_schema",
}

# If you have synchronous helpers that callers will call without awaiting, add them here.
_sync_helpers: Set[str] = set()  # e.g., {"some_sync_helper_name"}

# Install safe defaults (async no-ops for awaited helpers, sync no-ops for sync helpers)
for src_name, global_name in _DB_ALIAS.items():
    globals()[global_name] = _noop_sync if src_name in _sync_helpers else _noop_async

# Compatibility alias used elsewhere in trading.py
_persist_discovered_tokens = globals().get("persist_discovered_tokens", _noop_async)


def _bind_db_helpers_from_module(db_mod: Optional[Any]) -> None:
    """
    Attach known helper functions from a database module object to this module's
    globals. Forgiving: does not raise if names are missing or binding fails.
    """
    global _persist_discovered_tokens
    if not db_mod:
        return

    for src_name, global_name in _DB_ALIAS.items():
        try:
            if hasattr(db_mod, src_name):
                val = getattr(db_mod, src_name)
                globals()[global_name] = val
                try:
                    logger.debug("Bound DB helper %s -> %s", src_name, global_name)
                except Exception:
                    pass
        except Exception:
            try:
                logger.debug("Binding DB helper %s from module failed", src_name, exc_info=True)
            except Exception:
                pass

    # Keep pointer for packaged helper if present
    try:
        if hasattr(db_mod, "persist_discovered_tokens"):
            _persist_discovered_tokens = getattr(db_mod, "persist_discovered_tokens")
    except Exception:
        pass


# ----- Replaced section: DB helper binding + fallback + dry-run persistence helpers -----
# Attempt to bind real helpers from the module object if present.
try:
    _bind_db_helpers_from_module(_db)
except Exception:
    try:
        logger.debug("Initial DB helper bind attempt failed (continuing with stubs).", exc_info=True)
    except Exception:
        pass

# If no module object is available, construct a minimal _db fallback that uses the no-op stubs.
if _db is None:
    # Minimal async DB object used by callers when they "async with _db.connect_db() as db:"
    class _DummyDB:
        async def execute(self, *a, **k):
            return None

        async def commit(self, *a, **k):
            return None

        async def close(self, *a, **k):
            return None



    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _dummy_connect_db_contextmanager(*a, **k):
        db = _DummyDB()
        try:
            yield db
        finally:
            try:
                await db.close()
            except Exception:
                pass

    # Provide module-like _db fallback with async callables and CM factory
    import types

    _db = types.SimpleNamespace(
        init_db=globals().get("init_db", _noop_async),
        connect_db=_dummy_connect_db_contextmanager,
        load_blacklist=globals().get("db_load_blacklist", _noop_async),
        add_to_blacklist=globals().get("db_add_to_blacklist", _noop_async),
        clear_expired_blacklist=globals().get("clear_expired_blacklist", _noop_async),
        review_blacklist=globals().get("review_blacklist", _noop_async),
        get_open_positions=globals().get("get_open_positions", _noop_async),
        update_token_record=globals().get("update_token_record", _noop_async),
        persist_discovered_tokens=globals().get("persist_discovered_tokens", _noop_async),
        persist_eligible_shortlist=globals().get("persist_eligible_shortlist", _noop_async),
        bulk_upsert_tokens=globals().get("bulk_upsert_tokens", _noop_async),
        ensure_discovered_tokens_schema=globals().get("ensure_discovered_tokens_schema", _noop_async),
        mark_token_sold=globals().get("mark_token_sold", _noop_async),
    )

    logger.warning("Using fallback async-aware DB stubs; database-backed features disabled.")
    
# -------------------------------------------------------------------
# Ensure-token helper + DRY-RUN persistence helpers 
# -------------------------------------------------------------------

async def _ensure_token_row_exists(token_address: str, symbol: Optional[str] = None, name: Optional[str] = None) -> bool:
    """
    Best-effort: ensure a minimal `tokens` row exists for token_address.
    Returns True if the row exists or was created, False on error.
    Non-blocking and tolerant: logs debug on failure and returns False.
    """
    if not token_address:
        return False
    try:
        if not hasattr(_db, "connect_db") or _db.connect_db is None:
            try:
                logger.debug("_ensure_token_row_exists: no _db.connect_db available")
            except Exception:
                pass
            return False

        async with _db.connect_db() as db:
            try:
                now = int(time.time())
                sql = "INSERT OR IGNORE INTO tokens(address, name, symbol, timestamp, is_trading) VALUES (?, ?, ?, ?, ?);"
                params = (token_address, (name or "UNKNOWN"), (symbol or "UNKNOWN"), now, 1)
                await db.execute(sql, params)
                try:
                    await db.commit()
                except Exception:
                    # commit may be optional depending on connection semantics; ignore commit errors
                    pass
                return True
            except sqlite3.IntegrityError as ie:
                try:
                    logger.debug("Token upsert integrity error for %s: %s", token_address, ie)
                except Exception:
                    pass
                return False
            except Exception:
                try:
                    logger.debug("Token upsert failed for %s", token_address, exc_info=True)
                except Exception:
                    pass
                return False
    except Exception:
        try:
            logger.debug("_ensure_token_row_exists top-level failure", exc_info=True)
        except Exception:
            pass
        return False


async def _persist_dryrun_buy_into_trade_history(
    token_address: str,
    symbol: str,
    buy_price: float,
    buy_amount_sol: float,
    buy_txid: str,
    buy_time: int,
    config: Optional[dict] = None,
) -> None:
    """
    Persist a DRY_RUN buy into trade_history (best-effort).
    This function first tries a high-level record_trade, falling back
    to a raw INSERT that tolerates schema variants. It ensures a minimal
    tokens row exists first to avoid FK errors.
    """
    try:
        # opt-in guard: config flag or env var
        cfg_flag = False
        try:
            if isinstance(config, dict):
                cfg_flag = bool((config.get("trading") or {}).get("persist_dry_run_trades", False))
        except Exception:
            cfg_flag = False
        env_flag = str(os.getenv("PERSIST_DRYRUN_TRADES", "0")).strip().lower() in ("1", "true", "yes", "on", "y")
        if not (cfg_flag or env_flag):
            return

        # Compute token quantity (token units) from SOL amount when price available
        token_qty = None
        try:
            bp = float(buy_price or 0.0)
            amt_sol = float(buy_amount_sol or 0.0)
            if bp > 0 and amt_sol > 0:
                token_qty = float(amt_sol) / float(bp)
        except Exception:
            token_qty = None

        # Ensure tokens row exists so record_trade / raw insert won't fail FK constraints
        try:
            await _ensure_token_row_exists(token_address, symbol=symbol)
        except Exception:
            try:
                logger.debug("ensure token row existence helper failed (ignored)", exc_info=True)
            except Exception:
                pass

        # 1) Try high-level record_trade if available (async expected)
        rec = globals().get("record_trade") or getattr(_db, "record_trade", None)
        if callable(rec):
            try:
                # attempt keyword-style call first
                await rec(
                    token={"address": token_address, "symbol": symbol},
                    buy_price=(float(buy_price) if buy_price is not None else None),
                    buy_amount=(float(token_qty) if token_qty is not None else None),
                    buy_txid=str(buy_txid or ""),
                    buy_time=int(buy_time or int(time.time())),
                    simulated=True,
                )
                try:
                    logger.debug("Persisted DRYRUN buy via record_trade for %s", token_address)
                except Exception:
                    pass
                return
            except TypeError:
                # positional fallback (historic signature)
                try:
                    await rec(token_address, symbol, float(buy_price or 0.0), float(token_qty or 0.0), str(buy_txid or ""), int(buy_time or int(time.time())))
                    try:
                        logger.debug("Persisted DRYRUN buy via alternate record_trade signature for %s", token_address)
                    except Exception:
                        pass
                    return
                except Exception:
                    # fall through to raw INSERT
                    pass
            except Exception:
                # fall through to raw INSERT
                pass

        # 2) Fallback: raw INSERT via _db.connect_db() (best-effort; tolerant to schema variants)
        try:
            if not hasattr(_db, "connect_db") or _db.connect_db is None:
                try:
                    logger.debug("No _db.connect_db available for raw insert into trade_history (skip)")
                except Exception:
                    pass
                return

            async with _db.connect_db() as db:
                # Discover columns (best-effort)
                cols: list[str] = []
                try:
                    res = await db.execute("PRAGMA table_info(trade_history);")
                    try:
                        rows = await res.fetchall()
                    except Exception:
                        rows = res or []
                    for r in rows:
                        try:
                            if isinstance(r, dict) or hasattr(r, "keys"):
                                nm = r.get("name") or r.get(1) or None
                            else:
                                nm = r[1] if len(r) > 1 else None
                            if nm:
                                cols.append(str(nm))
                        except Exception:
                            continue
                except Exception:
                    cols = []

                cand_map = {
                    "token_address": token_address,
                    "symbol": symbol,
                    "buy_price": float(buy_price or 0.0),
                    "buy_amount": (float(token_qty) if token_qty is not None else float(buy_amount_sol or 0.0)),
                    "buy_txid": str(buy_txid or ""),
                    "buy_time": int(buy_time or int(time.time())),
                    "simulated": 1,
                }

                insert_cols: list[str] = []
                params: list = []
                if cols:
                    for k in ("token_address", "symbol", "buy_price", "buy_amount", "buy_txid", "buy_time", "simulated"):
                        if k in cols:
                            insert_cols.append(k)
                            params.append(cand_map[k])
                else:
                    insert_cols = ["token_address", "symbol", "buy_price", "buy_amount", "buy_txid", "buy_time"]
                    params = [cand_map[k] for k in insert_cols]

                if not insert_cols:
                    try:
                        logger.debug("No suitable trade_history columns discovered; skipping raw insert.")
                    except Exception:
                        pass
                    return

                placeholders = ", ".join(["?"] * len(insert_cols))
                sql = f"INSERT INTO trade_history ({', '.join(insert_cols)}) VALUES ({placeholders});"
                try:
                    await db.execute(sql, tuple(params))
                    try:
                        await db.commit()
                    except Exception:
                        pass
                    try:
                        logger.debug("Persisted DRYRUN buy via raw INSERT into trade_history for %s", token_address)
                    except Exception:
                        pass
                except Exception:
                    try:
                        logger.debug("Raw INSERT into trade_history failed for %s", token_address, exc_info=True)
                    except Exception:
                        pass
        except Exception:
            try:
                logger.debug("Fallback DB insert failed while recording DRYRUN buy", exc_info=True)
            except Exception:
                pass

    except Exception:
        try:
            logger.debug("Unexpected error in _persist_dryrun_buy_into_trade_history", exc_info=True)
        except Exception:
            pass


async def _persist_dryrun_sell_into_trade_history(
    token_address: str,
    symbol: str,
    sell_price: float,
    sell_amount_tokens: Optional[float],
    sell_txid: str,
    sell_time: int,
    config: Optional[dict] = None,
) -> None:
    """
    Best-effort persist a DRY_RUN sell into trade_history.
    Tries record_trade first, falls back to tolerant raw INSERT.
    Ensures tokens row exists first to avoid FK errors.
    """
    try:
        cfg_flag = False
        try:
            if isinstance(config, dict):
                cfg_flag = bool((config.get("trading") or {}).get("persist_dry_run_trades", False))
        except Exception:
            cfg_flag = False
        env_flag = str(os.getenv("PERSIST_DRYRUN_TRADES", "0")).strip().lower() in ("1", "true", "yes", "on", "y")
        if not (cfg_flag or env_flag):
            return

        # Ensure tokens row exists so record_trade / raw insert won't fail FK constraints
        try:
            await _ensure_token_row_exists(token_address, symbol=symbol)
        except Exception:
            try:
                logger.debug("ensure token row existence helper failed (ignored)", exc_info=True)
            except Exception:
                pass

        # 1) Try high-level record_trade if available
        rec = globals().get("record_trade") or getattr(_db, "record_trade", None)
        if callable(rec):
            try:
                await rec(
                    token={"address": token_address, "symbol": symbol},
                    sell_price=(float(sell_price) if sell_price is not None else None),
                    sell_amount=(float(sell_amount_tokens) if sell_amount_tokens is not None else None),
                    sell_txid=str(sell_txid or ""),
                    sell_time=int(sell_time or int(time.time())),
                    simulated=True,
                )
                try:
                    logger.debug("Persisted DRYRUN sell via record_trade for %s", token_address)
                except Exception:
                    pass
                return
            except Exception:
                # ignore and fall through to raw insert
                pass

        # 2) Fallback: raw INSERT via _db.connect_db()
        try:
            if not hasattr(_db, "connect_db") or _db.connect_db is None:
                try:
                    logger.debug("No _db.connect_db available for raw insert into trade_history (skip sell)")
                except Exception:
                    pass
                return

            async with _db.connect_db() as db:
                # Discover columns
                cols: list[str] = []
                try:
                    res = await db.execute("PRAGMA table_info(trade_history);")
                    try:
                        rows = await res.fetchall()
                    except Exception:
                        rows = res or []
                    for r in rows:
                        try:
                            if isinstance(r, dict) or hasattr(r, "keys"):
                                nm = r.get("name") or r.get(1) or None
                            else:
                                nm = r[1] if len(r) > 1 else None
                            if nm:
                                cols.append(str(nm))
                        except Exception:
                            continue
                except Exception:
                    cols = []

                cand_map = {
                    "token_address": token_address,
                    "symbol": symbol,
                    "sell_price": float(sell_price or 0.0),
                    "sell_amount": (float(sell_amount_tokens) if sell_amount_tokens is not None else None),
                    "sell_txid": str(sell_txid or ""),
                    "sell_time": int(sell_time or int(time.time())),
                    "simulated": 1,
                }

                insert_cols: list[str] = []
                params: list = []
                if cols:
                    for k in ("token_address", "symbol", "sell_price", "sell_amount", "sell_txid", "sell_time", "simulated"):
                        if k in cols:
                            insert_cols.append(k)
                            params.append(cand_map[k])
                else:
                    insert_cols = ["token_address", "symbol", "sell_price", "sell_amount", "sell_txid", "sell_time"]
                    params = [cand_map[k] for k in insert_cols]

                if not insert_cols:
                    try:
                        logger.debug("No suitable trade_history columns discovered for sell; skipping raw insert.")
                    except Exception:
                        pass
                    return

                placeholders = ", ".join(["?"] * len(insert_cols))
                sql = f"INSERT INTO trade_history ({', '.join(insert_cols)}) VALUES ({placeholders});"
                try:
                    await db.execute(sql, tuple(params))
                    try:
                        await db.commit()
                    except Exception:
                        pass
                    try:
                        logger.debug("Persisted DRYRUN sell via raw INSERT into trade_history for %s", token_address)
                    except Exception:
                        pass
                except Exception:
                    try:
                        logger.debug("Raw INSERT into trade_history failed for sell %s", token_address, exc_info=True)
                    except Exception:
                        pass
        except Exception:
            try:
                logger.debug("Fallback DB insert failed while recording DRYRUN sell", exc_info=True)
            except Exception:
                pass

    except Exception:
        try:
            logger.debug("Unexpected error in _persist_dryrun_sell_into_trade_history", exc_info=True)
        except Exception:
            pass

# Ensure a safe default exists so callers cannot hit an UnboundLocalError even if imports fail.
def _apply_tech_tiebreaker(tokens: list, config: dict, *, logger: Optional[logging.Logger] = None) -> list:
    """Default no-op tie-breaker — overridden by import if available."""
    return tokens

# Try to acquire the tech tie-breaker from eligibility and override the default if found.
# This uses a simple import loop and will leave the no-op if nothing can be imported.
for _mod_path, _attr in (
    (".eligibility", "_apply_tech_tiebreaker"),
    ("solana_trading_bot_bundle.trading_bot.eligibility", "_apply_tech_tiebreaker"),
    (".eligibility", "apply_tech_tiebreaker"),
    ("solana_trading_bot_bundle.trading_bot.eligibility", "apply_tech_tiebreaker"),
):
    try:
        _pkg = __import__(_mod_path, fromlist=[_attr.split(".")[0]])
        _candidate = getattr(_pkg, _attr, None)
        if callable(_candidate):
            # Assign into module globals so the name is not treated as a local
            # variable inside the function scope (which produces UnboundLocalError
            # when the name is referenced earlier). Using globals() avoids that.
            globals()["_apply_tech_tiebreaker"] = _candidate  # type: ignore[assignment]
            logger.debug("Bound _apply_tech_tiebreaker from %s.%s (via globals())", _mod_path, _attr)
            break
    except Exception:
        # continue to next candidate silently
        continue
else:
    logger.debug("_apply_tech_tiebreaker not found in eligibility; using no-op fallback.")
                
# Prefer eligibility.hard_floor implementation from module; safe fallback if unavailable
try:
    # prefer relative local
    from .eligibility import enforce_scoring_hard_floor  # type: ignore
except Exception:
    try:
        # fallback to bundled package layout
        from solana_trading_bot_bundle.trading_bot.eligibility import enforce_scoring_hard_floor  # type: ignore
    except Exception:
        # safe no-op fallback so trading still runs if eligibility missing
        def enforce_scoring_hard_floor(token: dict, cfg: dict) -> tuple[bool, str]:
            # Allow by default (do not hard-block) if the real function isn't available.
            # This avoids accidental crashes in environments where eligibility cannot be imported.
            return True, "enforce_scoring_hard_floor unavailable (allowing by default)"                 
       
# Best-effort candlestick patterns classifier import (package -> relative -> disabled)
def _resolve_classify_patterns() -> Optional[Callable]:
    """Late-resolve the candlestick patterns classifier if available."""
    try:
        from solana_trading_bot_bundle.trading_bot.candlestick_patterns import classify_patterns
        logger.debug("Candlestick patterns classifier loaded (packaged path).")
        return classify_patterns  # type: ignore[return-value]
    except ImportError:
        try:
            from .candlestick_patterns import classify_patterns  # type: ignore
            logger.debug("Candlestick patterns classifier loaded (local path).")
            return classify_patterns  # type: ignore[return-value]
        except ImportError:
            logger.debug("Candlestick patterns classifier not available; pattern signals disabled.", exc_info=False)
            return None

# resolved on-demand by callers
_classify_patterns: Optional[Callable] = None

# --- Fetching (prefer packaged bundle; fallback to local; final shims) ------
def _resolve_fetching_impl():
    """
    Return a small namespace (dict-like) of fetching functions and a source tag.
    Tries (in order): packaged bundle -> relative local -> final shims.
    """
    out = {"fetch_dexscreener_search": None, "fetch_raydium_tokens": None,
           "fetch_birdeye_tokens": None, "ensure_rugcheck_status_file": None,
           "fetch_raydium_pool_for_mint": None, "batch_enrich_tokens_with_signals": None,
           "source": "missing"}

    try:
        from solana_trading_bot_bundle.trading_bot import fetching as _f
        out["fetch_dexscreener_search"] = getattr(_f, "fetch_dexscreener_search", None)
        out["fetch_raydium_tokens"] = getattr(_f, "fetch_raydium_tokens", None)
        out["fetch_birdeye_tokens"] = getattr(_f, "fetch_birdeye_tokens", None)
        out["ensure_rugcheck_status_file"] = getattr(_f, "ensure_rugcheck_status_file", None)
        out["fetch_raydium_pool_for_mint"] = getattr(_f, "fetch_raydium_pool_for_mint", None)
        out["batch_enrich_tokens_with_signals"] = getattr(_f, "batch_enrich_tokens_with_signals", None)
        out["source"] = "bundle"
        return out
    except ImportError:
        pass

    try:
        from . import fetching as _f
        out["fetch_dexscreener_search"] = getattr(_f, "fetch_dexscreener_search", None)
        out["fetch_raydium_tokens"] = getattr(_f, "fetch_raydium_tokens", None)
        out["fetch_birdeye_tokens"] = getattr(_f, "fetch_birdeye_tokens", None)
        out["ensure_rugcheck_status_file"] = getattr(_f, "ensure_rugcheck_status_file", None)
        out["fetch_raydium_pool_for_mint"] = getattr(_f, "fetch_raydium_pool_for_mint", None)
        out["batch_enrich_tokens_with_signals"] = getattr(_f, "batch_enrich_tokens_with_signals", None)
        out["source"] = "local"
        return out
    except ImportError:
        # final shims
        async def _noop_async(*a, **k):
            return []
        def _noop_sync(*a, **k):
            return None
        out["fetch_dexscreener_search"] = _noop_async
        out["fetch_raydium_tokens"] = _noop_async
        out["fetch_birdeye_tokens"] = _noop_async
        out["ensure_rugcheck_status_file"] = _noop_sync
        out["fetch_raydium_pool_for_mint"] = _noop_async
        out["batch_enrich_tokens_with_signals"] = None
        out["source"] = "missing"
        return out

# Resolve once at import time (cheap)
_FETCH_IMPL = _resolve_fetching_impl()
fetch_dexscreener_search = _FETCH_IMPL["fetch_dexscreener_search"]
fetch_raydium_tokens = _FETCH_IMPL["fetch_raydium_tokens"]
fetch_birdeye_tokens = _FETCH_IMPL["fetch_birdeye_tokens"]
ensure_rugcheck_status_file = _FETCH_IMPL["ensure_rugcheck_status_file"]
fetch_raydium_pool_for_mint = _FETCH_IMPL["fetch_raydium_pool_for_mint"]
batch_enrich_tokens_with_signals = _FETCH_IMPL["batch_enrich_tokens_with_signals"]
USING_FETCHING = _FETCH_IMPL["source"]

logger.debug("USING FETCHING FROM: %s", USING_FETCHING)

# Try to import fetching module helpers that trading.py now references.
try:
    from .fetching import _export_birdeye_bg_task, get_birdeye_snapshot_count, shutdown_shared_clients
except Exception:
    try:
        from solana_trading_bot_bundle.trading_bot.fetching import _export_birdeye_bg_task, get_birdeye_snapshot_count, shutdown_shared_clients
    except Exception:
        # Safe no-op fallbacks so trading still runs when fetching helpers aren't available.
        def _export_birdeye_bg_task(task: Optional[asyncio.Task]) -> None:
            return None

        async def get_birdeye_snapshot_count(timeout: float = 1.2) -> int:
            return 0

        async def shutdown_shared_clients(timeout: float = 5.0) -> None:
            return None

# --- Metrics Engine (best-effort import; no-op if unavailable) --------------
import asyncio, time, logging  # make sure these are available above

# Lazy resolver + safe emit wrapper for metrics to avoid blocking / startup errors.
METRICS = None
_make_store = None
logger = logger  # assumes 'logger' already defined in this module

def _resolve_metrics_store_once():
    """
    Try to import and instantiate the metrics store once.
    Returns the store instance or None. Uses ImportError for missing modules
    and logs unexpected failures at debug level.
    """
    global METRICS, _make_store
    if METRICS is not None or _make_store is not None:
        return METRICS or _make_store
    try:
        from solana_trading_bot_bundle.trading_bot.metrics_engine import make_store as _ms  # type: ignore
        _make_store = _ms
        try:
            METRICS = _make_store(logger=logger, replay_on_init=True)
            logger.debug("Metrics engine loaded (packaged path).")
        except Exception:
            logger.debug("Metrics make_store available but instantiation failed.", exc_info=True)
        return METRICS or _make_store
    except ImportError:
        try:
            from .metrics_engine import make_store as _ms  # type: ignore
            _make_store = _ms
            try:
                METRICS = _make_store(logger=logger, replay_on_init=True)
                logger.debug("Metrics engine loaded (local path).")
            except Exception:
                logger.debug("Local metrics make_store present but instantiation failed.", exc_info=True)
            return METRICS or _make_store
        except ImportError:
            logger.debug("Metrics engine not available; metrics recording disabled.", exc_info=False)
            METRICS = None
            _make_store = None
            return None
        except Exception:
            logger.debug("Unexpected error importing local metrics engine.", exc_info=True)
            METRICS = None
            _make_store = None
            return None
    except Exception:
        logger.debug("Unexpected error importing packaged metrics engine.", exc_info=True)
        METRICS = None
        _make_store = None
        return None

# Public alias for tests/external wiring: returns the factory if available (callable) or None.
def make_metrics_store():
    _resolve_metrics_store_once()
    return _make_store

# Internal cached wrapper for the resolved metrics store
_metrics_store = None

def _safe_emit(fn, payload):
    """
    Call metrics emission in a safe manner:
      - If fn is coroutine function or returns a coroutine, schedule it with create_task.
      - Catch and log all exceptions at debug level.
    This guarantees metrics never raise into trading logic.
    """
    try:
        res = fn(payload) if payload is not None else fn()
        if asyncio.iscoroutine(res):
            try:
                asyncio.create_task(res)
            except RuntimeError:
                # No running loop; try a sync fallback carefully
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(res)
                    else:
                        loop.run_until_complete(res)
                except Exception:
                    logger.debug("Metrics emit failed in sync fallback", exc_info=True)
    except Exception:
        logger.debug("Metrics emission failed", exc_info=True)

def _get_metrics_store():
    """
    Return the active metrics store instance (prefer _metrics_store, then METRICS).
    Ensure resolver is attempted at first call.
    """
    global _metrics_store
    if _metrics_store is not None:
        return _metrics_store
    _resolve_metrics_store_once()
    _metrics_store = METRICS or None
    return _metrics_store

def _metrics_on_fill(
    token_addr: str,
    symbol: str,
    side: str,
    qty: float,
    price_usd: float,
    fee_usd: float,
    txid: str | None,
    simulated: bool,
    **meta
) -> None:
    """
    Best-effort call into a MetricsStore-like object.
    Prefer structured record_fill if available, fall back to generic record.
    Emissions are protected and non-blocking where possible.
    """
    try:
        ms = _get_metrics_store()
        if not ms:
            return
        if hasattr(ms, "record_fill"):
            try:
                payload = {
                    "token_addr": token_addr,
                    "side": side.upper(),
                    "qty": qty,
                    "price_usd": price_usd,
                    "fee_usd": fee_usd,
                    "dry_run": simulated,
                    "txid": txid,
                    "symbol": symbol,
                    **meta,
                }
                _safe_emit(ms.record_fill, payload)
                return
            except Exception:
                logger.debug("Metrics.record_fill path failed; falling back.", exc_info=True)
        if hasattr(ms, "record"):
            try:
                payload = {
                    "type": "fill",
                    "token": token_addr,
                    "symbol": symbol,
                    "side": side.upper(),
                    "qty": qty,
                    "price_usd": price_usd,
                    "fee_usd": fee_usd,
                    "txid": txid,
                    "simulated": simulated,
                    **meta,
                }
                _safe_emit(ms.record, payload)
            except Exception:
                logger.debug("Metrics.record fallback failed", exc_info=True)
    except Exception:
        try:
            logger.debug("Metrics on_fill failed", exc_info=True)
        except Exception:
            pass

def _metrics_snapshot_equity_point(logger: logging.Logger | None = None) -> None:
    """
    Best-effort call to snapshot equity / point-in-time metrics.
    No-op if metrics store lacks such API. Use safe emit wrapper.
    """
    try:
        ms = _get_metrics_store()
        if not ms:
            return

        # Prefer a dedicated method if present and call via _safe_emit without spurious None arg
        if hasattr(ms, "snapshot_equity_point"):
            try:
                _safe_emit(ms.snapshot_equity_point)
                return
            except Exception:
                # use provided logger if available, otherwise module logger
                (logger or logging.getLogger("TradingBot")).debug("Metrics snapshot_equity_point failed", exc_info=True)

        # Otherwise try a generic record() API with a small payload
        if hasattr(ms, "record"):
            try:
                _safe_emit(ms.record, {"type": "equity_snapshot", "ts": int(time.time())})
                return
            except Exception:
                (logger or logging.getLogger("TradingBot")).debug("Metrics snapshot.record failed", exc_info=True)
    except Exception:
        (logger or logging.getLogger("TradingBot")).debug("Metrics snapshot failed", exc_info=True)
        
def _db_path_from_config(config: Optional[dict]) -> Path:
    """
    Resolve a DB path from the config dictionary.

    Precedence:
      1. config['database']['token_cache_path']
      2. config['paths']['token_cache_path']
      3. config['token_cache_path']
      4. token_cache_path() helper (callable or value)
      5. _default_db_path() fallback
    """
    try:
        db_cfg = (config or {}).get("database") or {}
        paths_cfg = (config or {}).get("paths") or {}
        raw = (
            db_cfg.get("token_cache_path")
            or paths_cfg.get("token_cache_path")
            or (config or {}).get("token_cache_path")
        )
        if raw:
            return Path(raw)
    except Exception:
        # swallow and fall through to helper-based defaults
        pass

    # token_cache_path may be a callable or a value imported from constants
    try:
        val = token_cache_path() if callable(token_cache_path) else token_cache_path
        if val:
            return Path(val)
    except Exception:
        # final defensive fallback
        try:
            return _default_db_path()  # if you have such helper in scope
        except Exception:
            # Last resort: current working directory 'tokens.sqlite3'
            return Path.cwd() / "tokens.sqlite3"        
            
# -------------------------------------------------------------------------
# Candlestick / BB helpers (best-effort; inert if numpy/pandas absent)
# -------------------------------------------------------------------------
_patterns_module = None
_pd = None
_np = None
try:
    import numpy as _np  # type: ignore
except ImportError:
    _np = None
try:
    import pandas as _pd  # type: ignore
except ImportError:
    _pd = None
try:
    # Support a numpy-based patterns module or a local candlestick_patterns module
    import candlestick_patterns as _patterns_module  # type: ignore
except ImportError:
    try:
        # package-style import
        from solana_trading_bot_bundle.trading_bot import candlestick_patterns as _patterns_module  # type: ignore
    except ImportError:
        _patterns_module = None

# Indicators optional import (vectorized helpers like bollinger)
try:
    from . import indicators as _ind  # local package path (preferred)
except Exception:
    try:
        from solana_trading_bot_bundle.trading_bot import indicators as _ind  # packaged path
    except Exception:
        _ind = None
def attach_patterns_if_available(token: dict) -> None:
    """
    If token has OHLC arrays or a prebuilt pandas DataFrame, attempt to attach
    pattern booleans (pat_<name>_last) in-place. Defensive: returns silently if
    required libs or data are missing.

    This version memoizes a timestamp tok["_ohlc_df_ts"] when it builds a DataFrame
    so other helpers can detect and invalidate stale cached frames.
    """
    try:
        if not _patterns_module:
            return
        df = None
        if _pd and isinstance(token.get("_ohlc_df"), _pd.DataFrame):
            df = token.get("_ohlc_df")
        else:
            if _pd and isinstance(token.get("_ohlc_close"), (list, tuple)):
                try:
                    close = list(token.get("_ohlc_close") or [])
                    open_ = list(token.get("_ohlc_open") or [])
                    high = list(token.get("_ohlc_high") or [])
                    low = list(token.get("_ohlc_low") or [])
                    vol = list(token.get("_ohlc_volume") or token.get("_ohlc_vol") or [])
                    if len(close) and len(close) == len(open_) == len(high) == len(low):
                        df = _pd.DataFrame({
                            "open": open_,
                            "high": high,
                            "low": low,
                            "close": close,
                            "volume": vol if vol else [0] * len(close)
                        })
                        try:
                            import time as _time
                            token["_ohlc_df_ts"] = int(_time.time())
                        except Exception:
                            token["_ohlc_df_ts"] = None
                except Exception:
                    df = None
        if df is None:
            return
        try:
            pat_hits = _patterns_module.classify_patterns(df)  # defensive: may raise
        except Exception:
            pat_hits = {}
        for name, arr in (pat_hits or {}).items():
            try:
                if isinstance(arr, (list, tuple)):
                    token[f"pat_{name}_last"] = bool(arr[-1])
                else:
                    try:
                        if hasattr(arr, "iloc"):
                            token[f"pat_{name}_last"] = bool(arr.iloc[-1])
                        else:
                            token[f"pat_{name}_last"] = bool(arr[-1])
                    except Exception:
                        try:
                            token[f"pat_{name}_last"] = bool(list(arr)[-1])
                        except Exception:
                            token[f"pat_{name}_last"] = False
            except Exception:
                token[f"pat_{name}_last"] = False
    except Exception:
        try:
            logger.debug("attach_patterns_if_available failed for %s", token.get("symbol", token.get("address")), exc_info=True)
        except Exception:
            pass

def _attach_bbands_if_available(token: dict, window: int = 20, stdev: float = 2.0) -> None:
    """
    If token contains _ohlc_close (list) or _ohlc_df (pandas), compute Bollinger Bands
    (basis, upper, lower) and attach boolean signals bb_long, bb_short for last row.
    Defensive: no-op if pandas/numpy not available.
    """
    try:
        if not _pd or not _np:
            return
        df = None
        if isinstance(token.get("_ohlc_df"), _pd.DataFrame):
            df = token.get("_ohlc_df")
        elif isinstance(token.get("_ohlc_close"), (list, tuple)):
            close = list(token.get("_ohlc_close") or [])
            if len(close) < window:
                return
            df = _pd.DataFrame({"close": close})
        else:
            return

        if "close" not in df.columns:
            return
        # compute rolling mean and std
        try:
            basis = df["close"].rolling(window=window, min_periods=window).mean()
            dev = df["close"].rolling(window=window, min_periods=window).std()
            upper = basis + dev * stdev
            lower = basis - dev * stdev
            # attach last values if present
            token["bb_basis"] = float(basis.iloc[-1]) if not _np.isnan(basis.iloc[-1]) else None
            token["bb_upper"] = float(upper.iloc[-1]) if not _np.isnan(upper.iloc[-1]) else None
            token["bb_lower"] = float(lower.iloc[-1]) if not _np.isnan(lower.iloc[-1]) else None
            # simple signals: price cross over / under
            close_last = float(df["close"].iloc[-1])
            token["bb_long"] = (token.get("bb_lower") is not None and close_last <= token["bb_lower"])
            token["bb_short"] = (token.get("bb_upper") is not None and close_last >= token["bb_upper"])
        except Exception:
            # safe to ignore any numeric oddities
            return
    except Exception:
        try:
            logger.debug("attach_bbands_if_available failed for %s", token.get("symbol", token.get("address")), exc_info=True)
        except Exception:
            pass

# ---- Additions: batch attach helper + DataFrame memoization + indicators use ----
# These helpers are safe/fallback-first; they will no-op if optional deps are missing.

def _make_df_from_ohlcv_dict(ohlcv: dict):
    """
    Build and return a pandas DataFrame from an ohlcv dict (open/high/low/close, volume).
    Returns None on any error or if pandas not available.
    Also does not set token fields; caller should set _ohlc_df and _ohlc_df_ts.
    """
    try:
        if ohlcv is None:
            return None
        import pandas as _pd  # lazy import
        open_ = list(ohlcv.get("open") or [])
        high = list(ohlcv.get("high") or [])
        low = list(ohlcv.get("low") or [])
        close = list(ohlcv.get("close") or [])
        vol = list(ohlcv.get("volume") or ohlcv.get("vol") or [])
        if not (len(close) and len(close) == len(open_) == len(high) == len(low)):
            return None
        df = _pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol if vol else [0] * len(close)
        })
        return df
    except Exception:
        return None


def _batch_attach_bbands_and_patterns(tokens: list[dict], *, window: int = 20, stdev: float = 2.0, pat_list: list | None = None):
    """
    Batch process a list of tokens that have `_ohlc_open/_ohlc_close/...` (or `_ohlc_df`) populated.
    Returns a summary dict with counts for observability. Defensive: never raises; best-effort only.
    """
    summary = {"n_attempted": 0, "n_bbands": 0, "n_patterns": 0, "errors": 0}
    if not tokens:
        return summary
    try:
        import time as _time
        # TTL for cached DataFrame (seconds)
        _OHLCCACHE_TTL_S = 300  # 5 minutes

        # lazy imports
        try:
            import numpy as _np
        except Exception:
            _np = None
        try:
            import pandas as _pd
        except Exception:
            _pd = None
        try:
            from . import indicators as _ind  # pure-Python & vectorized helpers (preferred)
        except Exception:
            _ind = None

        # If pattern classifier missing and indicators missing, nothing to batch
        if _classify_patterns is None and _ind is None:
            return summary

        # collect tokens to process (with enough bars), invalidating stale cached DataFrames
        to_process = []
        minlen = int(window or 20)
        for tok in tokens:
            try:
                # prefer an existing df
                df = tok.get("_ohlc_df")
                df_ts = tok.get("_ohlc_df_ts")
                # If df exists, check TTL and invalidate if stale
                if df is not None and isinstance(df_ts, (int, float)):
                    try:
                        if int(_time.time()) - int(df_ts) > _OHLCCACHE_TTL_S:
                            # stale - drop cached DF so we'll rebuild below
                            tok.pop("_ohlc_df", None)
                            tok.pop("_ohlc_df_ts", None)
                            df = None
                    except Exception:
                        tok.pop("_ohlc_df", None)
                        tok.pop("_ohlc_df_ts", None)
                        df = None

                if df is None:
                    # build ohlcv dict from exposed lists if available
                    close = tok.get("_ohlc_close") or []
                    if isinstance(close, (list, tuple)) and len(close) >= minlen:
                        to_process.append(tok)
                    else:
                        continue
                else:
                    try:
                        if len(df) >= minlen:
                            to_process.append(tok)
                    except Exception:
                        continue
            except Exception:
                continue

        if not to_process:
            return summary

        # For each token, ensure _ohlc_df memoized and compute BBs via indicators if possible,
        # then call classifier.classify_patterns once (preferring DataFrame interface).
        for tok in to_process:
            summary["n_attempted"] += 1
            try:
                # memoize DataFrame if pandas available and not present (or was invalidated above)
                df = tok.get("_ohlc_df")
                if df is None and _pd is not None:
                    try:
                        o = {
                            "open": tok.get("_ohlc_open") or [],
                            "high": tok.get("_ohlc_high") or [],
                            "low": tok.get("_ohlc_low") or [],
                            "close": tok.get("_ohlc_close") or [],
                            "volume": tok.get("_ohlc_volume") or []
                        }
                        df_candidate = _make_df_from_ohlcv_dict(o)
                        if df_candidate is not None:
                            tok["_ohlc_df"] = df_candidate
                            try:
                                tok["_ohlc_df_ts"] = int(_time.time())
                            except Exception:
                                tok["_ohlc_df_ts"] = None
                            df = df_candidate
                    except Exception:
                        df = None

                attached_bb = False
                attached_patterns = False

                # compute BBs using vectorized path if indicators present
                try:
                    if df is not None and _ind is not None:
                        try:
                            res = None
                            try:
                                res = _ind.bollinger(list(df["close"].to_numpy()), period=window, stddev=stdev)
                            except Exception:
                                try:
                                    res = _ind.bollinger(list(df["close"].to_list()), period=window, stddev=stdev)
                                except Exception:
                                    res = None
                            if res:
                                mid = res.get("mid")
                                upper = res.get("upper")
                                lower = res.get("lower")
                                if mid is not None and len(mid) >= window:
                                    last_idx = len(mid) - 1
                                    try:
                                        token_mid = float(mid[last_idx])
                                    except Exception:
                                        token_mid = None
                                    try:
                                        token_upper = float(upper[last_idx]) if upper is not None else None
                                    except Exception:
                                        token_upper = None
                                    try:
                                        token_lower = float(lower[last_idx]) if lower is not None else None
                                    except Exception:
                                        token_lower = None
                                    tok["bb_basis"] = token_mid
                                    tok["bb_upper"] = token_upper
                                    tok["bb_lower"] = token_lower
                                    try:
                                        close_last = float(df["close"].iloc[-1])
                                    except Exception:
                                        close_last = None
                                    tok["bb_long"] = bool(close_last is not None and token_lower is not None and close_last <= token_lower)
                                    tok["bb_short"] = bool(close_last is not None and token_upper is not None and close_last >= token_upper)
                                    attached_bb = True
                        except Exception:
                            pass

                    # If indicators path didn't attach bb, fall back to existing function
                    if not attached_bb and tok.get("bb_basis") is None:
                        try:
                            _attach_bbands_if_available(tok, window=window, stdev=stdev)
                            # consider attached if fields now present
                            if tok.get("bb_basis") is not None:
                                attached_bb = True
                        except Exception:
                            pass
                except Exception:
                    pass

                if attached_bb:
                    summary["n_bbands"] += 1

                # Patterns: prefer classifier with DataFrame if available
                if _classify_patterns is not None:
                    try:
                        try:
                            cfg_min_bars = int((load_config() or {}).get("signals", {}).get("patterns", {}).get("min_bars", minlen))
                        except Exception:
                            cfg_min_bars = minlen
                        try:
                            close_len = len(df) if df is not None else len(tok.get("_ohlc_close") or [])
                        except Exception:
                            close_len = 0
                        if close_len >= cfg_min_bars:
                            if df is not None:
                                try:
                                    pat_hits = _classify_patterns(df)
                                except Exception:
                                    try:
                                        od = {
                                            "open": list(df["open"].to_list()),
                                            "high": list(df["high"].to_list()),
                                            "low": list(df["low"].to_list()),
                                            "close": list(df["close"].to_list()),
                                            "volume": list(df["volume"].to_list()) if "volume" in df.columns else []
                                        }
                                        pat_hits = _classify_patterns(od)
                                    except Exception:
                                        pat_hits = {}
                            else:
                                od = {
                                    "open": list(tok.get("_ohlc_open") or []),
                                    "high": list(tok.get("_ohlc_high") or []),
                                    "low": list(tok.get("_ohlc_low") or []),
                                    "close": list(tok.get("_ohlc_close") or []),
                                    "volume": list(tok.get("_ohlc_volume") or [])
                                }
                                try:
                                    pat_hits = _classify_patterns(od)
                                except Exception:
                                    pat_hits = {}
                            # attach last-value booleans using robust extraction
                            any_pat = False
                            for name, arr in (pat_hits or {}).items():
                                try:
                                    val = None
                                    if isinstance(arr, (list, tuple)):
                                        val = bool(arr[-1]) if arr else False
                                    else:
                                        try:
                                            if hasattr(arr, "iloc"):
                                                val = bool(arr.iloc[-1])
                                            else:
                                                val = bool(arr[-1])
                                        except Exception:
                                            try:
                                                val = bool(list(arr)[-1])
                                            except Exception:
                                                val = False
                                    tok[f"pat_{name}_last"] = bool(val)
                                    tok[f"pat_{name}_5m"] = bool(val)
                                    if val:
                                        any_pat = True
                                except Exception:
                                    tok[f"pat_{name}_last"] = False
                                    tok[f"pat_{name}_5m"] = False
                            if any_pat:
                                attached_patterns = True
                    except Exception:
                        pass

                if attached_patterns:
                    summary["n_patterns"] += 1

            except Exception:
                summary["errors"] += 1
                continue

        # Final logging & optional metrics emit
        try:
            logger.info("Batch signals: processed=%d bb_attached=%d patterns_attached=%d errors=%d (window=%d, stdev=%s)",
                        summary["n_attempted"], summary["n_bbands"], summary["n_patterns"], summary["errors"], int(window), str(stdev))
        except Exception:
            pass

        # If METRICS supports a simple counter or snapshot, emit a small record
        try:
            if METRICS is not None and hasattr(METRICS, "record"):
                METRICS.record({
                    "type": "signals_batch_summary",
                    "n_processed": summary["n_attempted"],
                    "n_bbands": summary["n_bbands"],
                    "n_patterns": summary["n_patterns"],
                    "n_errors": summary["errors"],
                    "window": int(window),
                    "stdev": float(stdev),
                })
        except Exception:
            pass

        return summary

    except Exception:
        # ensure we always return a dict summary even on top-level failure
        try:
            logger.debug("batch attach failed unexpectedly", exc_info=True)
        except Exception:
            pass
        return summary

async def _enrich_shortlist_with_signals(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich a shortlist of tokens with Bollinger Bands and pattern signals.

    Defensive: optional deps (batch_enrich_tokens_with_signals, _classify_patterns,
    pandas/numpy, indicators) may be missing. Always returns the input list on any
    failure. Performs:
      - calls batch_enrich_tokens_with_signals(...) if available to populate ohlcv fields
      - for each token, ensures lightweight _ohlc_* lists exist (from enricher or fetcher)
      - groups tokens with enough bars and calls _batch_attach_bbands_and_patterns(...)
      - falls back to attach_patterns_if_available(...) per-token

    Notes:
      - Honors both signals.enabled and signals.patterns.enable config flags.
      - Uses a bounded per-token fetch timeout from the signals config.
      - Never raises; on unexpected errors returns the original tokens list.
    """
    # Early guard: batch enricher must exist
    if batch_enrich_tokens_with_signals is None:
        try:
            logger.debug("Signals enrichment unavailable: batch enricher not present; skipping.")
        except Exception:
            pass
        return tokens

    # Resolve OHLCV fetcher (may be None)
    try:
        fetcher = _resolve_ohlcv_fetcher()
    except Exception:
        fetcher = None

    if not fetcher:
        try:
            logger.debug("Signals enrichment skipped: no OHLCV fetcher found/resolved.")
        except Exception:
            pass
        return tokens

    # Run the batch enrichment implementation (may be package or local)
    try:
        enriched = await batch_enrich_tokens_with_signals(tokens, fetch_ohlcv_func=fetcher, concurrency=8)
    except Exception as e:
        try:
            logger.warning("Batch enrichment raised; skipping signals enrichment: %s", e)
        except Exception:
            pass
        return tokens

    # Safety: if enrichment didn't return a list, fall back to original tokens
    if not isinstance(enriched, list):
        return tokens

    # Resolve signals configuration and pattern list (defensive)
    try:
        cfg = load_config() if callable(load_config) else {}
    except Exception:
        cfg = {}
    _signals_cfg = _signals_cfg_from_config(cfg)
    signals_enabled = bool((_signals_cfg.get("enabled", True)))
    patterns_enabled = bool((_signals_cfg.get("patterns") or {}).get("enable", True))
    patterns_cfg = (cfg.get("patterns") or {}) if isinstance(cfg, dict) else {}
    pat_list = patterns_cfg.get("list") or [
        "bullish_engulfing", "bearish_engulfing", "hammer", "shooting_star",
        "morning_star", "evening_star", "doji", "hanging_man"
    ]

    # Honor both global signals flag and patterns-specific enable flag
    if not signals_enabled or not patterns_enabled:
        try:
            logger.debug("Signals or pattern detection disabled by config; returning enriched without attaching patterns.")
        except Exception:
            pass
        return enriched

    # Helper to obtain 5m OHLCV; prefer batch enricher fields, then fetcher(addr, "5m")
    async def _ensure_ohlcv5(tok: dict):
        # Prefer fields commonly set by batch enricher
        try:
            cands = []
            try:
                cands.append(tok.get("ohlcv_5m"))
            except Exception:
                pass
            try:
                cands.append(tok.get("_ohlcv_5m"))
            except Exception:
                pass
            try:
                ohlcv_container = tok.get("ohlcv")
                if isinstance(ohlcv_container, dict):
                    cands.append(ohlcv_container.get("5m"))
            except Exception:
                pass

            for c in (cands or []):
                if isinstance(c, dict) and all(k in c for k in ("open", "high", "low", "close")):
                    return {
                        "open": list(c.get("open") or []),
                        "high": list(c.get("high") or []),
                        "low": list(c.get("low") or []),
                        "close": list(c.get("close") or []),
                        "volume": list(c.get("volume") or []),
                    }
        except Exception:
            pass

        # Fallback to fetcher with a bounded timeout from config
        try:
            addr = tok.get("address") or tok.get("token_address") or tok.get("mint") or (tok.get("baseToken") or {}).get("address")
            if not addr or not callable(fetcher):
                return None
            per_timeout = int(_signals_cfg.get("per_ohlcv_timeout_s", 6))
            import asyncio as _asyncio
            try:
                o = await _asyncio.wait_for(fetcher(addr, interval="5m", limit=200), timeout=per_timeout)
            except _asyncio.TimeoutError:
                return None
            except Exception:
                return None
            if isinstance(o, dict) and all(k in o for k in ("open", "high", "low", "close")):
                return {
                    "open": list(o.get("open") or []),
                    "high": list(o.get("high") or []),
                    "low": list(o.get("low") or []),
                    "close": list(o.get("close") or []),
                    "volume": list(o.get("volume") or []),
                }
        except Exception:
            pass

        return None

    # Prepare candidates: build lightweight _ohlc_* lists and pick those with enough bars
    try:
        candidate_for_batch: list[dict] = []
        _min_bars = int((_signals_cfg.get("patterns") or {}).get("min_bars", 20))
        _bb_window = int((_signals_cfg.get("bbands") or {}).get("window", 20))
        _bb_stdev = float((_signals_cfg.get("bbands") or {}).get("stdev", 2.0))

        for tok in (enriched or []):
            try:
                # If batch enricher already provided lists under various keys, prefer them (do not clobber)
                if "_ohlc_close" not in tok or not isinstance(tok.get("_ohlc_close"), (list, tuple)):
                    o5 = tok.get("ohlcv_5m") or tok.get("_ohlcv_5m") or ((tok.get("ohlcv") or {}).get("5m") if isinstance(tok.get("ohlcv"), dict) else None)
                    if isinstance(o5, dict) and all(k in o5 for k in ("open", "close", "high", "low")):
                        try:
                            tok.setdefault("_ohlc_open", list(o5.get("open") or []))
                            tok.setdefault("_ohlc_high", list(o5.get("high") or []))
                            tok.setdefault("_ohlc_low", list(o5.get("low") or []))
                            tok.setdefault("_ohlc_close", list(o5.get("close") or []))
                            tok.setdefault("_ohlc_volume", list(o5.get("volume") or o5.get("vol") or []))
                        except Exception:
                            pass

                # If still missing, try to fetch 5m OHLCV
                if not isinstance(tok.get("_ohlc_close"), (list, tuple)) or not tok.get("_ohlc_close"):
                    try:
                        ohlcv5 = await _ensure_ohlcv5(tok)
                        if ohlcv5:
                            tok.setdefault("_ohlc_open", list(ohlcv5.get("open") or []))
                            tok.setdefault("_ohlc_high", list(ohlcv5.get("high") or []))
                            tok.setdefault("_ohlc_low", list(ohlcv5.get("low") or []))
                            tok.setdefault("_ohlc_close", list(ohlcv5.get("close") or []))
                            tok.setdefault("_ohlc_volume", list(ohlcv5.get("volume") or []))
                    except Exception:
                        pass

                cl = tok.get("_ohlc_close") or []
                if not isinstance(cl, (list, tuple)):
                    # nothing usable for this token; still attempt lightweight attach
                    try:
                        attach_patterns_if_available(tok)
                    except Exception:
                        pass
                    continue

                if len(cl) >= _min_bars:
                    # optional: memoize DataFrame if pandas available and helper exists
                    if "_ohlc_df" not in tok:
                        try:
                            df = _make_df_from_ohlcv_dict({
                                "open": tok.get("_ohlc_open") or [],
                                "high": tok.get("_ohlc_high") or [],
                                "low": tok.get("_ohlc_low") or [],
                                "close": tok.get("_ohlc_close") or [],
                                "volume": tok.get("_ohlc_volume") or []
                            })
                            if df is not None:
                                tok["_ohlc_df"] = df
                                try:
                                    import time as _time
                                    tok["_ohlc_df_ts"] = int(_time.time())
                                except Exception:
                                    tok["_ohlc_df_ts"] = None
                        except Exception:
                            pass
                    candidate_for_batch.append(tok)
                else:
                    # short series: still expose lists but don't add to batch candidates
                    try:
                        attach_patterns_if_available(tok)
                    except Exception:
                        pass
            except Exception:
                # skip token on any normalization error
                continue

        # Run batch helper (best-effort). The helper is allowed to be missing (noop).
        try:
            if candidate_for_batch:
                try:
                    logger.debug("Running batch_attach on %d tokens (bb_window=%d)", len(candidate_for_batch), _bb_window)
                except Exception:
                    pass
                try:
                    _batch_attach_bbands_and_patterns(candidate_for_batch, window=_bb_window, stdev=_bb_stdev, pat_list=pat_list)
                except Exception:
                    # ensure any failure in batch helper doesn't bubble up
                    logger.debug("batch_attach_bbands_and_patterns failed", exc_info=True)
        except Exception:
            # degrade to per-token attach for tokens not handled
            pass

        # For tokens not processed (short arrays or earlier fail), fall back to per-token attach
        for tok in (enriched or []):
            try:
                # if patterns already attached, skip
                if any(k.startswith("pat_") for k in tok.keys()):
                    continue
                # otherwise try the robust per-token helper (this will attempt BBs too)
                try:
                    attach_patterns_if_available(tok)
                except Exception:
                    logger.debug("attach_patterns_if_available failed (fallback) for %s", tok.get("symbol", tok.get("address")), exc_info=True)
            except Exception:
                continue

        return enriched
    except Exception as e:
        try:
            logger.warning("Signals enrichment failed (continuing without): %s", e)
        except Exception:
            pass
        return tokens    
   
def _record_metric_fill(
    token_addr: str,
    symbol: str,
    side: str,
    quote: dict | None,
    buy_amount_sol: float | None = None,
    token_price_sol: float | None = None,
    txid: str | None = None,
    simulated: bool = False,
    source: str = "jupiter",
    **extra_metadata
) -> None:
    """
    Safe wrapper to record a standardized fill event to metrics store.

    Best-effort extraction of:
      - qty (token units)
      - price_usd (per-token)
      - fee_usd

    Uses, in order of preference:
      - explicit values from the Jupiter `quote` dict
      - buy_amount_sol and token_price_sol provided by caller
      - cached SOL->USD price passed in extra_metadata['sol_price_usd'] or price_cache

    All work is defensive: missing libs/fields won't raise.
    """
    # Fast path: nothing to do if metrics subsystem unavailable
    if METRICS is None:
        try:
            logger.debug("Metrics store not available; skipping fill recording.")
        except Exception:
            pass
        return

    try:
        # defaults
        qty = 0.0
        price_usd = 0.0
        fee_usd = 0.0
        amount_sol = float(buy_amount_sol or 0.0)

        # Prefer a supplied SOL→USD price (caller should pass sol_price_usd). Fall back to price_cache.
        try:
            sol_price_usd = float(extra_metadata.get("sol_price_usd") or price_cache.get("SOLUSD") or 0.0)
        except Exception:
            sol_price_usd = float(price_cache.get("SOLUSD") or 0.0)

        LAMPORTS_PER_SOL_LOCAL = 1_000_000_000

        # Helper: best-effort decimals resolver (non-blocking, uses price_cache if available)
        def _resolve_decimals(mint_addr: str | None, fallback: int = 6) -> int:
            try:
                if not mint_addr:
                    return fallback
                v = price_cache.get(f"decimals:{mint_addr}")
                if v is not None:
                    return int(v)
            except Exception:
                pass
            return fallback

        # Parse quote when available (Jupiter-like)
        if quote and isinstance(quote, dict):
            try:
                in_amount_raw = quote.get("inAmount", 0)
                out_amount_raw = quote.get("outAmount", 0)
                # Some quotes use numeric strings; coerce safely
                try:
                    in_amount_val = float(in_amount_raw)
                except Exception:
                    try:
                        in_amount_val = float(str(in_amount_raw))
                    except Exception:
                        in_amount_val = 0.0
                try:
                    out_amount_val = float(out_amount_raw)
                except Exception:
                    try:
                        out_amount_val = float(str(out_amount_raw))
                    except Exception:
                        out_amount_val = 0.0

                # Determine token mint from quote metadata if provided
                token_mint = None
                try:
                    token_mint = quote.get("outputMint") or quote.get("tokenMint") or token_addr
                except Exception:
                    token_mint = token_addr

                # BUY: inAmount is SOL lamports, outAmount is token base units
                if side.upper() == "BUY":
                    if in_amount_val:
                        amount_sol = float(in_amount_val) / LAMPORTS_PER_SOL_LOCAL
                    # qty from outAmount using token decimals
                    if out_amount_val:
                        token_decimals = _resolve_decimals(token_mint, fallback=6)
                        try:
                            qty = float(out_amount_val) / (10 ** int(token_decimals))
                        except Exception:
                            try:
                                qty = float(out_amount_val) / (10 ** 6)
                            except Exception:
                                qty = 0.0
                else:
                    # SELL: inAmount is token base units, outAmount is SOL lamports
                    if in_amount_val:
                        token_mint = quote.get("inputMint") or token_mint or token_addr
                        token_decimals = _resolve_decimals(token_mint, fallback=6)
                        try:
                            qty = float(in_amount_val) / (10 ** int(token_decimals))
                        except Exception:
                            try:
                                qty = float(in_amount_val) / (10 ** 6)
                            except Exception:
                                qty = 0.0
                    if out_amount_val:
                        amount_sol = float(out_amount_val) / LAMPORTS_PER_SOL_LOCAL

                # Fee extraction (platformFee may be in lamports or token base units)
                fee_usd = 0.0
                try:
                    platform_fee = quote.get("platformFee")
                    if isinstance(platform_fee, dict):
                        fee_amt = platform_fee.get("amount") or platform_fee.get("lamports") or 0
                        try:
                            fee_amt_val = float(fee_amt)
                        except Exception:
                            fee_amt_val = 0.0
                        # Best-effort: treat as lamports -> USD
                        fee_usd = (fee_amt_val / LAMPORTS_PER_SOL_LOCAL) * (sol_price_usd or 0.0)
                    else:
                        # approximate from priceImpactPct if present
                        if "priceImpactPct" in quote:
                            try:
                                impact_pct = float(quote.get("priceImpactPct") or 0.0)
                                fee_usd = (amount_sol or 0.0) * (sol_price_usd or 0.0) * (abs(impact_pct) / 100.0)
                            except Exception:
                                fee_usd = 0.0
                except Exception:
                    fee_usd = 0.0

            except Exception:
                # defensive: ignore quote parsing errors
                pass

        # Fallback qty inference if still unknown
        try:
            if (not qty or qty <= 0.0) and amount_sol and token_price_sol and token_price_sol > 0:
                qty = float(amount_sol) / float(token_price_sol)
        except Exception:
            pass

        # Final price_usd computation (prefer derived from amount_sol & qty using sol_price_usd)
        try:
            if qty and qty > 0 and amount_sol and (sol_price_usd and sol_price_usd > 0):
                price_usd = (float(amount_sol or 0.0) * float(sol_price_usd or 0.0)) / float(qty)
            elif token_price_sol and (sol_price_usd and sol_price_usd > 0):
                price_usd = float(token_price_sol) * float(sol_price_usd or 0.0)
            else:
                if qty and qty > 0 and amount_sol:
                    price_sol = float(amount_sol) / float(qty)
                    price_usd = price_sol * float(sol_price_usd or 1.0)
                else:
                    price_usd = 0.0
        except Exception:
            price_usd = 0.0

        # Record to METRICS (support both record_fill or generic record)
        try:
            if hasattr(METRICS, "record_fill"):
                METRICS.record_fill(
                    token_addr=token_addr,
                    side=side.upper(),
                    qty=qty,
                    price_usd=price_usd,
                    fee_usd=fee_usd,
                    dry_run=bool(simulated),
                    symbol=symbol,
                    txid=txid,
                    source=source,
                    amount_sol=amount_sol,
                    token_price_sol=token_price_sol,
                    **(extra_metadata or {}),
                )
            elif hasattr(METRICS, "record"):
                METRICS.record(
                    {
                        "type": "fill",
                        "token": token_addr,
                        "symbol": symbol,
                        "side": side.upper(),
                        "qty": qty,
                        "price_usd": price_usd,
                        "amount_sol": amount_sol,
                        "fee_usd": fee_usd,
                        "txid": txid,
                        "simulated": simulated,
                        "source": source,
                        "quote": quote,
                        "token_price_sol": token_price_sol,
                        "sol_price_usd": sol_price_usd,
                        **(extra_metadata or {}),
                    }
                )
            try:
                logger.debug(
                    "Recorded metric fill: %s %s qty=%.6f price_usd=%.6f simulated=%s",
                    side,
                    symbol,
                    qty,
                    price_usd,
                    simulated,
                )
            except Exception:
                pass

            # Also call the generic _metrics_on_fill wrapper as a best-effort side-channel.
            # Keep this nested try/except so metrics problems cannot bubble into trading logic.
            try:
                _metrics_on_fill(
                    token_addr=token_addr,
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    price_usd=price_usd,
                    fee_usd=fee_usd,
                    txid=txid,
                    simulated=bool(simulated),
                    source=source,
                    **(extra_metadata or {}),
                )
            except Exception:
                try:
                    logger.debug("Aux metrics_on_fill invocation failed", exc_info=True)
                except Exception:
                    pass

        except Exception:
            # swallow metrics errors
            try:
                logger.debug("Failed to emit metric record for %s", symbol, exc_info=True)
            except Exception:
                pass

    except Exception as e:
        try:
            logger.debug("Failed to record metric fill for %s: %s", symbol, e, exc_info=False)
        except Exception:
            pass

# --- Shared helper: derive a unix 'creation' timestamp in *seconds* ----------
def _derive_creation_ts_s(d: Dict[str, Any]) -> int:
    """
    Robustly derive a unix 'creation' timestamp in *seconds* from common fields.
    Handles ms -> s, ISO8601 strings, ints/floats, and simple nested dicts.
    Priority order favors already-canonical fields.
    """
    from datetime import datetime, timezone

    def _to_intish(v) -> int:
        if v is None:
            return 0
        if isinstance(v, (int, float)):
            try:
                return int(v)
            except Exception:
                return 0
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return 0
            if s.isdigit():
                try:
                    return int(s)
                except Exception:
                    return 0
            # ISO8601 string?
            try:
                s2 = s.replace("Z", "+00:00")
                dt = datetime.fromisoformat(s2)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp())
            except Exception:
                return 0
        if isinstance(v, dict):
            # common nested carriers like {"ms":...}, {"ts":...}
            for k in ("ms", "sec", "seconds", "ts", "timestamp", "createdAt", "listedAt"):
                if k in v:
                    return _to_intish(v[k])
            return 0
        return 0

    # Try in this order; some sources use milliseconds
    for key in (
        "creation_timestamp",  
        "created_at",          
        "pairCreatedAt",       
        "listedAt",            
        "createdAt",           
        "timestamp",           
    ):
        val = d.get(key)
        ts = _to_intish(val)
        if ts > 2_000_000_000:  # clearly milliseconds -> seconds
            ts //= 1000
        if ts > 0:
            return ts
    return 0

# --- Categories backfill (restore/compute) -------------------------------------
def _restore_or_compute_categories(tokens: list[dict]) -> list[dict]:
    """
    Ensure each token has a 'categories' list before selector runs.
    If missing, compute a coarse cap-bucket and attach:
      - 'low_cap'    : mc < 100_000
      - 'mid_cap'    : 100_000 <= mc < 500_000
      - 'large_cap'  : mc >= 500_000

    If an explicit 'bucket' exists ('low_cap'/'mid_cap'/'large_cap'/'newly_launched'),
    prefer that. This function is tolerant to numeric strings with commas/currency
    symbols and missing fields.
    """
    out: list[dict] = []

    def _to_num_loose(x: Any) -> float:
        """Loosely parse ints/floats or numeric-looking strings (commas, $) to float."""
        try:
            if x is None:
                return 0.0
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip()
            if not s:
                return 0.0
            # remove common formatting
            s = s.replace(",", "").replace("$", "")
            return float(s)
        except Exception:
            return 0.0

    for t in tokens or []:
        tt = dict(t or {})
        cats = tt.get("categories")
        # If categories already present and non-empty, keep them
        if isinstance(cats, list) and cats:
            out.append(tt)
            continue

        # Prefer an existing 'bucket' if present (tolerant to non-string values)
        try:
            bucket_raw = tt.get("bucket", "") or ""
            bucket = str(bucket_raw).strip().lower()
        except Exception:
            bucket = ""

        if bucket not in ("low_cap", "mid_cap", "large_cap", "newly_launched"):
            # derive from market cap if needed; accept multiple aliases
            mc = _to_num_loose(tt.get("market_cap", tt.get("mc", tt.get("fdv", 0.0))))
            if mc < 100_000:
                bucket = "low_cap"
            elif mc < 500_000:
                bucket = "mid_cap"
            else:
                bucket = "large_cap"

        tt["categories"] = [bucket]
        out.append(tt)

    return out


def select_top_five_per_category(signals):
    """
    Group signals by their 'category' key and return a dict mapping category ->
    list of top 5 signals ordered by descending 'score' (falls back to 0).

    signals: iterable of dict-like objects expected to contain at least:
      - "category" (optional): grouping key (uses "uncategorized" if missing or falsey)
      - "score" (optional): numeric or string-convertible-to-float score; non-numeric falls back to 0

    Returns:
      dict: { category_name: [top_5_signal_dicts_in_descending_score] }
    """
    grouped = {}
    for s in signals or []:
        try:
            cat = s.get("category") or "uncategorized"
        except Exception:
            cat = "uncategorized"
        grouped.setdefault(cat, []).append(s)

    top_per_cat = {}
    for cat, items in grouped.items():
        # sort by 'score' (descending), treating missing/non-numeric as 0
        def _score(x):
            try:
                return float(x.get("score", 0))
            except Exception:
                return 0.0

        try:
            sorted_items = sorted(items, key=_score, reverse=True)
        except Exception:
            # If sorting fails for any reason, keep original order and slice
            sorted_items = list(items)[:]
        top_per_cat[cat] = sorted_items[:5]

    return top_per_cat

# --- AsyncClient / Commitment (Pylance-safe) --------------------------------
try:
    # Preferred: solana-py Commitment object (what AsyncClient expects)
    from solana.rpc.commitment import Commitment as _RpcCommitment
    COMMIT_CONFIRMED = _RpcCommitment("confirmed")
    COMMIT_PROCESSED = _RpcCommitment("processed")
except Exception:
    COMMIT_CONFIRMED = None
    COMMIT_PROCESSED = None

# ---- Core constants / paths
from solana_trading_bot_bundle.common.constants import (
    APP_NAME, local_appdata_dir, appdata_dir, logs_dir, data_dir,
    config_path, env_path, db_path, token_cache_path, ensure_app_dirs, prefer_appdata_file
)

# ---- Third-party
import aiohttp
from dotenv import load_dotenv
from solders.commitment_config import CommitmentLevel
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from spl.token.instructions import get_associated_token_address

# ---- Feature gates (env + config aware)
from solana_trading_bot_bundle.common.feature_flags import (
    is_enabled_raydium,
    is_enabled_birdeye,
    resolved_run_flags,  # optional: for truthful log line
)

# ---- Signals config helpers (defaults + small accessor) ----
_DEFAULT_SIGNALS_CFG = {
    "enabled": True,
    "batch_mode": True,
    "bbands": {"window": 20, "stdev": 2.0},
    "patterns": {"min_bars": 20},
    "per_ohlcv_timeout_s": 6,  # per-token OHLC fetch timeout when using fetcher
}

def _signals_cfg_from_config(cfg: dict | None) -> dict:
    """Read signals-related settings from main config with safe fallbacks."""
    try:
        scfg = (cfg or {}).get("signals") or {}
        out = dict(_DEFAULT_SIGNALS_CFG)
        # shallow merge bbands and patterns subkeys
        bb = out["bbands"].copy()
        bb.update(scfg.get("bbands") or {})
        out["bbands"] = bb
        pat = out["patterns"].copy()
        pat.update(scfg.get("patterns") or {})
        out["patterns"] = pat
        out["batch_mode"] = bool(scfg.get("batch_mode", out["batch_mode"]))
        out["enabled"] = bool(scfg.get("enabled", out["enabled"]))
        out["per_ohlcv_timeout_s"] = int(scfg.get("per_ohlcv_timeout_s", out["per_ohlcv_timeout_s"]))
        return out
    except Exception:
        return dict(_DEFAULT_SIGNALS_CFG)

# Package utilities

# Use the canonical RugCheck verifier implemented in eligibility.py
from .eligibility import verify_token_with_rugcheck
from .trading_helpers import _to_float, _to_int, truthy_env

from typing import Callable, Awaitable, Union, Any
from inspect import iscoroutinefunction

async def _resolve_buy_amount_sol(
    *,
    get_buy_amount_fn,
    config: dict,
    sol_price: float,
    wallet_balance: float,
    token: dict | None,
) -> float:
    def _clamp(v: Any) -> float:
        try:
            v = float(v)
            # guard NaN/inf/zero/negatives
            if v != v or v in (float("inf"), float("-inf")) or v <= 0:
                return 0.01
            return v
        except Exception:
            return 0.01

    # Call the provided sizing function (async or sync)
    if iscoroutinefunction(get_buy_amount_fn):
        try:
            v = await get_buy_amount_fn(
                token_data=token,
                wallet_balance=wallet_balance,
                sol_price=sol_price,
                config=config,
            )
        except TypeError:
            # legacy signature support
            v = await get_buy_amount_fn(config, sol_price)
    else:
        try:
            v = get_buy_amount_fn(
                token_data=token,
                wallet_balance=wallet_balance,
                sol_price=sol_price,
                config=config,
            )
        except TypeError:
            # legacy signature support
            v = get_buy_amount_fn(config, sol_price)

    # --- Normalize return to a scalar SOL amount ---
    if isinstance(v, (tuple, list)) and v:
        # (amount_sol, usd_amount) -> amount_sol
        v = v[0]
    elif isinstance(v, dict):
        # tolerate dict-style returns
        v = v.get("amount_sol") or v.get("sol") or next(
            (x for x in v.values() if isinstance(x, (int, float))), 0.0
        )
    elif v is None:
        v = 0.0

    return _clamp(v)

# Market/chain data
from .market_data import (
    validate_token_mint,
    get_token_price_in_sol,
    get_sol_balance,        # comes from market_data
    get_sol_price,
    get_jupiter_quote,
    check_token_account,
    create_token_account,
)
# --- DB schema helper (tolerant import) --------------------------------------
# Try the schema function; if it's not present, fall back to init_db; else no-op.
_ensure_schema_impl: Callable[[], Awaitable[None]] | Callable[[], None] | None
try:
    # Preferred name if present in this codebase
    from .database import ensure_eligible_tokens_schema as _ensure_schema_impl  # type: ignore[attr-defined]
except Exception:
    try:
        # Fall back to init_db (which ensures/patches schema idempotently)
        from .database import init_db as _ensure_schema_impl  # type: ignore[attr-defined]
    except Exception:
        _ensure_schema_impl = None  # last resort: no-op marker
        
# --- Discovered-tokens persistence (tolerant import) -------------------------
try:
    from .database import persist_discovered_tokens  # real implementation
except Exception:
    # No-op fallback so the call site won't crash if the symbol is missing
    async def persist_discovered_tokens(tokens, prune_hours: int = 24):
        return 0      

async def _ensure_schema_once(logger: logging.Logger) -> None:
    """Run whichever schema helper we found, if any, without blocking the event loop."""
    fn = _ensure_schema_impl
    if fn is None:
        logger.debug("DB schema helper not available; continuing without schema bootstrap.")
        return
    try:
        if asyncio.iscoroutinefunction(fn):
            await fn()                     # async impl
        else:
            await asyncio.to_thread(fn)    # sync impl without blocking loop
    except Exception:
        logger.debug("Schema bootstrap failed (continuing):", exc_info=True)

def _default_db_path() -> Path:
    # Prefer explicit env var, otherwise app local dir on Windows, or ~/.local/share on others
    env = os.getenv("SOLO_BOT_DB")
    if env:
        return Path(env)
    win_base = os.getenv("LOCALAPPDATA")
    if win_base:
        return Path(win_base) / "SOLOTradingBot" / "tokens.sqlite3"
    # Fallback for *nix
    return Path(os.path.expanduser("~")) / ".local" / "share" / "SOLOTradingBot" / "tokens.sqlite3"

def _db_path_from_config(config: dict | None) -> Path:
    try:
        db_cfg = (config or {}).get("database") or {}
        paths_cfg = (config or {}).get("paths") or {}
        raw = (
            db_cfg.get("token_cache_path")
            or paths_cfg.get("token_cache_path")
            or (config or {}).get("token_cache_path")
        )
        if raw:
            return Path(raw)
    except Exception:
        pass
    try:
        return Path(token_cache_path())  # normal path
    except Exception:
        return _default_db_path()        # defensive fallback

    # Fallback to bundled helper (imported above from constants)
    return Path(token_cache_path())  # type: ignore[name-defined]

async def _load_persisted_shortlist_from_db(
    config: Optional[Dict[str, Any]] = None,
    max_age_s: int = 300,   # 5m TTL so we don't reuse stale shortlists
    min_count: int = 10     # require at least N rows or force live discovery
) -> List[Dict[str, Any]]:
    """
    Async wrapper to read the most recent shortlist from the local tokens DB.

    - Runs the blocking sqlite3 work in a thread so it does not block the event loop.
    - Returns [] if the DB is missing, empty, too old, or too small.
    - If available, passes loaded rows through validate_persisted_shortlist() to
      normalize numeric fields and re-apply hard-floor / momentum guards so stale
      or low-liquidity rows are not re-introduced.
    """
    cfg = config or {}

    def _blocking_load(cfg_inner, max_age_inner, min_count_inner):
        conn = None
        try:
            # Resolve DB path (use helper if available)
            try:
                db_path_obj = _db_path_from_config(cfg_inner) if callable(globals().get("_db_path_from_config")) else _default_db_path()
                db_path = str(db_path_obj)
            except Exception:
                logger.debug("_db_path_from_config failed; skipping DB shortlist.", exc_info=True)
                return []

            # Quick existence check
            try:
                if not os.path.exists(db_path):
                    logger.info("Shortlist DB not found at %s; skipping DB-based shortlist.", db_path)
                    return []
            except Exception:
                logger.debug("Error checking DB path existence (%s); skipping DB shortlist.", db_path, exc_info=True)
                return []

            import sqlite3
            try:
                conn = sqlite3.connect(db_path, timeout=5)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
            except Exception as e:
                logger.debug("Failed to open shortlist DB %s: %s", db_path, e, exc_info=True)
                return []

            # Ensure table exists and inspect columns
            try:
                cur.execute("PRAGMA table_info(eligible_tokens);")
                cols = [row["name"] for row in cur.fetchall()]
            except Exception:
                logger.info("Shortlist table 'eligible_tokens' not present or PRAGMA failed; skipping DB-based shortlist.")
                return []

            if not cols:
                logger.info("Shortlist table 'eligible_tokens' seems empty; skipping DB-based shortlist.")
                return []

            # Determine ordering/timestamp fields robustly
            order_col = "created_at" if "created_at" in cols else ("updated_ts" if "updated_ts" in cols else None)
            shortlist_flag_col = "is_shortlist" if "is_shortlist" in cols else None

            # Build tolerant SQL
            sql = "SELECT * FROM eligible_tokens"
            where_clauses = []
            if shortlist_flag_col:
                where_clauses.append(f"{shortlist_flag_col}=1")
            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)
            if order_col:
                sql += f" ORDER BY {order_col} DESC"
            sql += " LIMIT 200"

            try:
                cur.execute(sql)
                rows = cur.fetchall()
            except Exception:
                logger.debug("Shortlist DB query failed (%s) using SQL: %s", db_path, sql, exc_info=True)
                return []

            if not rows:
                logger.info("Shortlist DB query returned 0 rows; skipping DB-based shortlist.")
                return []

            now = time.time()
            out: List[Dict[str, Any]] = []
            seen: Set[str] = set()
            youngest_ts: Optional[float] = None

            def _get_ts(row: sqlite3.Row) -> Optional[float]:
                try:
                    if "created_at" in row.keys() and row["created_at"] is not None:
                        return float(row["created_at"])
                    if "updated_ts" in row.keys() and row["updated_ts"] is not None:
                        return float(row["updated_ts"])
                    if "updated_at" in row.keys() and row["updated_at"] is not None:
                        v = row["updated_at"]
                        try:
                            return float(v)
                        except Exception:
                            return None
                except Exception:
                    return None
                return None

            for r in rows:
                try:
                    addr = None
                    if "address" in r.keys() and r["address"]:
                        addr = r["address"]
                    elif "token_address" in r.keys() and r["token_address"]:
                        addr = r["token_address"]
                    if not addr:
                        continue
                    if addr in seen:
                        continue
                    seen.add(addr)

                    ts = _get_ts(r)
                    if ts is not None:
                        try:
                            age = now - float(ts)
                            if max_age_s and age > max_age_s:
                                # rows are newest-first; once stale, stop scanning
                                break
                            youngest_ts = float(ts) if youngest_ts is None else max(youngest_ts, float(ts))
                        except Exception:
                            pass

                    tok: Dict[str, Any] = {
                        "address": addr,
                        "symbol": (r["symbol"] if "symbol" in r.keys() and r["symbol"] else (r.get("baseSymbol") or "UNKNOWN")),
                        "name": (r["name"] if "name" in r.keys() and r["name"] else (r.get("baseName") or "UNKNOWN")),
                        "categories": ["shortlist"],
                    }

                    # Copy tolerant numeric/signal fields
                    for k in (
                        "price", "price_change_1h", "price_change_6h", "price_change_24h",
                        "volume_24h", "v24hUSD", "volume24h", "liquidity", "market_cap", "score", "bucket"
                    ):
                        if k in r.keys() and r[k] is not None:
                            tok[k] = r[k]

                    for k in ("rsi_5m", "rsi_15m", "rsi_1h", "sma_5m_5", "sma_5m_10", "sma_5m_20", "atr_5m", "atr_15m"):
                        if k in r.keys() and r[k] is not None:
                            tok[k] = r[k]

                    out.append(tok)
                except Exception:
                    logger.debug("Malformed row in eligible_tokens skipped", exc_info=True)
                    continue

            # shortlist-level guards
            if not out:
                logger.info("Persisted shortlist empty after TTL filtering; forcing live discovery.")
                return []

            if len(out) < min_count_inner:
                logger.info("Persisted shortlist size %d < min_count %d; forcing live discovery.", len(out), min_count_inner)
                return []

            if youngest_ts is not None:
                age_s = int(now - youngest_ts)
                logger.info("Loaded %d token(s) from persisted shortlist DB %s (youngest age=%ds)", len(out), db_path, age_s)
            else:
                logger.info("Loaded %d token(s) from persisted shortlist DB %s", len(out), db_path)

            return out

        except Exception as e:
            logger.warning("Failed loading persisted shortlist from DB: %s", e)
            return []

        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass

    # Execute blocking loader in a threadpool
    try:
        loop = asyncio.get_running_loop()
        rows: List[Dict[str, Any]] = await loop.run_in_executor(None, _blocking_load, cfg, max_age_s, min_count)
    except RuntimeError:
        # no running loop -> fallback
        rows = await asyncio.to_thread(_blocking_load, cfg, max_age_s, min_count)
    except Exception as e:
        logger.exception("Failed to load persisted shortlist from DB: %s", e)
        rows = []

    if not rows:
        return []

    # validate/normalize persisted rows if helper available
    if validate_persisted_shortlist is not None:
        try:
            validated = await validate_persisted_shortlist(rows, cfg, session=None)
            orig = len(rows)
            kept = len(validated or [])
            dropped = orig - kept
            logger.info("Persisted shortlist validation: kept=%d dropped=%d (orig=%d)", kept, dropped, orig)
            rows = validated or []
        except Exception as e:
            logger.exception("validate_persisted_shortlist() failed — returning raw DB rows: %s", e)
            # fall back to raw rows
    else:
        logger.debug("validate_persisted_shortlist not available; returning raw DB rows (consider enabling validation).")

    # If after validation the list is below min_count, treat as empty to avoid fallback
    if min_count and len(rows) < int(min_count):
        logger.info("After validation persisted shortlist too small (have=%d < min_count=%d); skipping DB fallback.", len(rows), min_count)
        return []

    return list(rows or [])

    # Run the blocking loader in a thread to avoid blocking the event loop.
    try:
        return await asyncio.to_thread(_blocking_load, config, max_age_s, min_count)
    except Exception as e:
        logger.warning("Async wrapper for persisted shortlist failed: %s", e, exc_info=True)
        return []

# ---- Raydium shortlist enricher (safe, budgeted per-mint lookups) ----

async def _raydium_enrich_shortlist(session: aiohttp.ClientSession, shortlist: list[dict]) -> list[dict]:
    if os.getenv("ENABLE_RAYDIUM", "0") != "1":
        logger.info("Raydium disabled by ENABLE_RAYDIUM=0 - skipping enrichment.")
        return shortlist

    max_mints      = int(os.getenv("RAY_MINTS_PER_CYCLE", "40"))
    concurrency    = int(os.getenv("RAY_CONCURRENCY", "3"))
    per_timeout    = float(os.getenv("RAY_REQ_TIMEOUT_S", "6"))
    global_budget  = float(os.getenv("RAY_GLOBAL_BUDGET_S", "10"))
    min_liq_usd    = float(os.getenv("RAY_MIN_LIQ_USD", "50000"))
    break_on_empty = os.getenv("RAY_BREAK_ON_EMPTY", "1") == "1"
    empty_abort    = int(os.getenv("RAY_FIRST_EMPTY_ABORT", "2"))

    # first N unique mints
    mints, seen = [], set()
    for t in shortlist:
        a = t.get("address")
        if a and a not in seen:
            seen.add(a)
            mints.append(a)
        if len(mints) >= max_mints:
            break
    if not mints:
        return shortlist

    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    empty_batches = 0

    async def _fetch_one(mint: str) -> Dict:
        try:
            async with sem:
                info = await fetch_raydium_pool_for_mint(session, mint, timeout_s=per_timeout)
                if not info:
                    return {}
                if float(info.get("ray_liquidity_usd", 0) or 0) < min_liq_usd:
                    return {}
                return info
        except Exception:
            return {}

    out_map: Dict[str, Dict] = {}
    chunk = max(1, concurrency * 3)
    for i in range(0, len(mints), chunk):
        if time.time() - t0 > global_budget:
            logger.info("Raydium budget exhausted (%.1fs), stopping.", global_budget)
            break
        batch = mints[i:i+chunk]
        results = await asyncio.gather(*(_fetch_one(m) for m in batch), return_exceptions=True)
        added = 0
        for r in results:
            if isinstance(r, dict) and r.get("address"):
                out_map[r["address"]] = r
                added += 1
        if added == 0:
            empty_batches += 1
            if break_on_empty and empty_abort and empty_batches >= empty_abort:
                logger.info("Raydium enrichment aborting early due to consecutive empties.")
                break

    # merge fields back
    for t in shortlist:
        addr = t.get("address")
        info = out_map.get(addr)
        if info:
            t.setdefault("sources", []).append("raydium")
            t["liquidity"]   = max(float(t.get("liquidity", 0) or 0), float(info.get("ray_liquidity_usd", 0) or 0))
            t["ray_pool_id"] = info.get("ray_pool_id", "")
            t["ray_amm"]     = info.get("ray_amm", "")
            t["ray_version"] = info.get("ray_version", "")
    return shortlist

# --- LAZY import to break circular dependency (works for both package & script) ---
async def _enrich_tokens_with_price_change_lazy(
    session,
    tokens,
    logger,
    blacklist,
    failure_count,
):
    """
    Lazy wrapper that locates an available `enrich_tokens_with_price_change` implementation
    (packaged -> local -> injected global), optionally gates Birdeye usage for the duration
    of the call, and returns a merged/normalized list of enrichment rows.

    Key properties:
      - If no enrichment implementation is available, returns a neutral fallback list.
      - Supports async and sync enrichers (sync impls run in asyncio.to_thread).
      - Temporarily sets BIRDEYE_ENABLE="0" for the call when gating requires it and
        restores the prior env value afterwards.
      - Normalizes provider field aliases and merges enrichment rows back onto original
        tokens by address where possible.
      - Defensive: never raises; always returns a list.
    """
    import asyncio
    import os
    import time

    # Normalize input
    tokens = list(tokens or [])

    # small numeric helpers
    def _f(x, d=0.0):
        try:
            return float(x)
        except Exception:
            try:
                return float(d)
            except Exception:
                return 0.0

    def _i(x, d=0):
        try:
            return int(x)
        except Exception:
            try:
                return int(d)
            except Exception:
                return 0

    def _neutral_fallback(src):
        """Produce a neutral-enriched list preserving identity fields and canonical keys."""
        out = []
        for t in (src or []):
            try:
                t2 = dict(t or {})
            except Exception:
                t2 = {}
            def _fnum(x, dflt=0.0):
                try:
                    return float(x)
                except Exception:
                    try:
                        return float(dflt)
                    except Exception:
                        return 0.0
            def _inum(x, dflt=0):
                try:
                    return int(x)
                except Exception:
                    try:
                        return int(dflt)
                    except Exception:
                        return 0
            t2.setdefault("price", _fnum(t2.get("price", 0.0)))
            mc_val = _fnum(t2.get("mc", t2.get("market_cap", 0.0)))
            t2.setdefault("mc", mc_val)
            t2.setdefault("market_cap", mc_val)
            t2.setdefault("liquidity", _fnum(t2.get("liquidity", 0.0)))
            vol = (
                t2.get("volume_24h")
                or t2.get("dex_volume_24h")
                or t2.get("v24hUSD")
                or t2.get("volume24h")
                or 0.0
            )
            t2["volume_24h"] = _fnum(vol, 0.0)
            t2.setdefault("holderCount", _inum(t2.get("holderCount", 0)))
            t2.setdefault("pct_change_5m", 0.0)
            t2.setdefault("pct_change_1h", 0.0)
            t2.setdefault("pct_change_24h", 0.0)
            t2.setdefault("dexscreenerUrl", t2.get("dexscreenerUrl", "") or "")
            t2.setdefault("dsPairAddress", t2.get("dsPairAddress", "") or "")
            try:
                from .utils_exec import format_market_cap  # type: ignore
            except Exception:
                try:
                    from solana_trading_bot_bundle.trading_bot.utils import format_market_cap  # type: ignore
                except Exception:
                    format_market_cap = None
            try:
                if format_market_cap:
                    t2["mcFormatted"] = format_market_cap(_fnum(t2.get("market_cap", t2.get("mc", 0.0))))
                else:
                    t2["mcFormatted"] = str(_fnum(t2.get("market_cap", t2.get("mc", 0.0))))
            except Exception:
                t2["mcFormatted"] = str(_fnum(t2.get("market_cap", t2.get("mc", 0.0))))
            out.append(t2)
        return out

    # -------------------------
    # Resolve enrichment implementation (packaged -> local -> injected global)
    # -------------------------
    enrich_fn = None
    try:
        import solana_trading_bot_bundle.trading_bot.fetching as _fetch_mod  # type: ignore
        enrich_fn = getattr(_fetch_mod, "enrich_tokens_with_price_change", None)
    except Exception:
        enrich_fn = None

    if not callable(enrich_fn):
        try:
            from .fetching import enrich_tokens_with_price_change as _enrich_impl  # type: ignore
            enrich_fn = _enrich_impl
        except Exception:
            enrich_fn = enrich_fn

    if not callable(enrich_fn):
        candidate = globals().get("enrich_tokens_with_price_change")
        if callable(candidate):
            enrich_fn = candidate

    if not callable(enrich_fn):
        try:
            logger.debug("Price-change enrichment unavailable (no impl); using neutral fallback.")
        except Exception:
            pass
        return _neutral_fallback(tokens)

    # -------------------------
    # Decide Birdeye gating for this call
    # -------------------------
    disable_birdeye = False
    _prev_birdeye = None
    try:
        _birdeye_allowed_fn = globals().get("_birdeye_allowed")
        _global_cfg = globals().get("GLOBAL_CONFIG")
        if callable(_birdeye_allowed_fn) and not _birdeye_allowed_fn(_global_cfg):
            disable_birdeye = True
    except Exception:
        pass

    # env overrides and per-run breaker
    try:
        if os.getenv("BIRDEYE_ENABLE", "1") == "0":
            disable_birdeye = True
        if os.getenv("FORCE_DISABLE_BIRDEYE", "0") == "1":
            disable_birdeye = True
        if "_BIRDEYE_401_SEEN" in globals() and globals().get("_BIRDEYE_401_SEEN"):
            disable_birdeye = True
        if not os.getenv("BIRDEYE_API_KEY"):
            # if there is no key, treat Birdeye as effectively disabled for safety
            disable_birdeye = True
    except Exception:
        pass

    # Apply temporary env change if needed and remember previous state
    try:
        if disable_birdeye:
            _prev_birdeye = os.environ.get("BIRDEYE_ENABLE")
            os.environ["BIRDEYE_ENABLE"] = "0"
    except Exception:
        # If we can't mutate env, continue without gating (best-effort)
        _prev_birdeye = None

    # -------------------------
    # Call enrichment implementation safely (ensure restore in finally below)
    # -------------------------
    result = None
    try:
        if asyncio.iscoroutinefunction(enrich_fn):
            # try the most common rich signature first; gracefully degrade on TypeError
            try:
                result = await enrich_fn(session=session, tokens=tokens, logger=logger, blacklist=blacklist, failure_count=failure_count)
            except TypeError:
                try:
                    result = await enrich_fn(session=session, tokens=tokens, blacklist=blacklist)
                except TypeError:
                    try:
                        result = await enrich_fn(tokens, session)
                    except Exception as e:
                        raise
        else:
            # sync impl -> run in thread to avoid blocking event loop
            try:
                result = await asyncio.to_thread(enrich_fn, session, tokens, logger, blacklist, failure_count)
            except TypeError:
                try:
                    result = await asyncio.to_thread(enrich_fn, session, tokens)
                except Exception as e:
                    raise
    except Exception as e:
        try:
            logger.warning("Price-change enrichment raised %s; using neutral fallback.", e)
        except Exception:
            pass
        return _neutral_fallback(tokens)
    finally:
        # restore BIRDEYE_ENABLE exactly as it was before the call
        try:
            if disable_birdeye:
                if _prev_birdeye is None:
                    os.environ.pop("BIRDEYE_ENABLE", None)
                else:
                    os.environ["BIRDEYE_ENABLE"] = _prev_birdeye
        except Exception:
            # swallow restore errors; do not let env restore failures break enrichment
            pass

    # -------------------------
    # Sanity checks and normalization
    # -------------------------
    if not isinstance(result, list) or not result:
        try:
            logger.warning("Price-change enrichment returned empty; using neutral fallback.")
        except Exception:
            pass
        return _neutral_fallback(tokens)

    def _has_any_signal(row: dict) -> bool:
        try:
            return (
                _f(row.get("price"), 0.0) > 0.0
                or _f(row.get("priceChange5m", row.get("price_change_5m", 0.0))) != 0.0
                or _f(row.get("priceChange1h", row.get("price_change_1h", 0.0))) != 0.0
                or _f(row.get("priceChange24h", row.get("price_change_24h", 0.0))) != 0.0
            )
        except Exception:
            return False

    try:
        if sum(1 for r in result if isinstance(r, dict) and _has_any_signal(r)) == 0:
            try:
                logger.warning("Price-change enrichment looks all-zero; using neutral fallback.")
            except Exception:
                pass
            return _neutral_fallback(tokens)
    except Exception:
        try:
            logger.debug("Price-change enrichment shape unexpected; using neutral fallback.")
        except Exception:
            pass
        return _neutral_fallback(tokens)

    # Merge enriched rows back onto original tokens by address and normalize
    try:
        src_by_addr = {
            t.get("address"): t for t in tokens if isinstance(t, dict) and isinstance(t.get("address"), str) and t.get("address")
        }

        sanitized = []
        for item in result:
            try:
                item = dict(item) if isinstance(item, dict) else {}
                addr = item.get("address")
                base = src_by_addr.get(addr, {}) if addr else {}

                t = dict(base)  # start with original identity fields

                # pull fields from enrichment row (blended providers)
                price = item.get("price", item.get("lastPrice"))
                if price is None:
                    price = item.get("value") or item.get("last_price")

                mc = item.get("mc", item.get("market_cap"))
                if mc is None:
                    mc = item.get("fdv")

                liq = item.get("liquidity")
                holder_cnt = item.get("holderCount")

                pc1h = item.get("priceChange1h", item.get("price_change_1h"))
                pc6h = item.get("priceChange6h", item.get("price_change_6h"))
                pc24h = item.get("priceChange24h", item.get("price_change_24h"))
                pc5m = item.get("priceChange5m", item.get("price_change_5m"))

                vol_candidates = [
                    item.get("volume_24h"),
                    item.get("dex_volume_24h"),
                    item.get("v24hUSD"),
                    item.get("volume24h"),
                    base.get("volume_24h") if isinstance(base, dict) else None,
                    base.get("v24hUSD") if isinstance(base, dict) else None,
                    base.get("volume24h") if isinstance(base, dict) else None,
                ]
                best_vol = next((v for v in vol_candidates if isinstance(v, (int, float)) and float(v) > 0.0), 0.0)
                t["volume_24h"] = _f(best_vol, _f(base.get("volume_24h", 0.0) if isinstance(base, dict) else 0.0))

                # augment: prefer positive enrichment values, don't overwrite with zeros/None
                existing_price = _f(t.get("price", 0.0))
                if isinstance(price, (int, float)) and float(price) > 0:
                    t["price"] = float(price)
                else:
                    t["price"] = existing_price

                existing_mc = _f(t.get("mc", t.get("market_cap", 0.0)))
                if isinstance(mc, (int, float)) and float(mc) > 0:
                    t["mc"] = float(mc)
                    t["market_cap"] = float(mc)
                else:
                    t["mc"] = existing_mc
                    if _f(t.get("market_cap", 0.0)) <= 0:
                        t["market_cap"] = existing_mc

                t["liquidity"] = _f(liq, t.get("liquidity", 0.0))
                t["holderCount"] = _i(holder_cnt, t.get("holderCount", 0))

                # Percent changes (0 can be legit) — normalize to GUI keys
                t["pct_change_1h"] = _f(pc1h, 0.0)
                t["pct_change_6h"] = _f(pc6h, 0.0)
                t["pct_change_24h"] = _f(pc24h, 0.0)
                t["pct_change_5m"] = _f(pc5m, 0.0)

                # timestamp mapping; normalize ms->s and clamp future
                ts_candidates = [
                    item.get("updated_at"),
                    item.get("last_trade_unix"),
                    item.get("created_at"),
                    item.get("pairCreatedAt"),
                    item.get("timestamp"),
                ]
                ts_new = next((x for x in ts_candidates if isinstance(x, (int, float))), None)
                if ts_new is not None:
                    try:
                        ts_new_i = int(ts_new)
                        if ts_new_i > 10_000_000_000:
                            ts_new_i //= 1000
                        now = int(time.time())
                        if ts_new_i > now + 600:
                            ts_new_i = now
                        t["timestamp"] = max(int(t.get("timestamp") or 0), ts_new_i)
                    except Exception:
                        pass

                ds_url = item.get("dexscreenerUrl") or t.get("dexscreenerUrl") or ""
                ds_pair = item.get("dsPairAddress") or t.get("dsPairAddress") or ""
                t["dexscreenerUrl"] = ds_url if isinstance(ds_url, str) else ""
                t["dsPairAddress"] = ds_pair if isinstance(ds_pair, str) else ""

                try:
                    from .utils_exec import format_market_cap  # type: ignore
                except Exception:
                    try:
                        from solana_trading_bot_bundle.trading_bot.utils import format_market_cap  # type: ignore
                    except Exception:
                        format_market_cap = None
                try:
                    if format_market_cap:
                        t["mcFormatted"] = format_market_cap(_f(t.get("market_cap", t.get("mc", 0.0)), 0.0))
                    else:
                        t["mcFormatted"] = str(_f(t.get("market_cap", t.get("mc", 0.0)), 0.0))
                except Exception:
                    t["mcFormatted"] = str(_f(t.get("market_cap", t.get("mc", 0.0)), 0.0))

                if "address" not in t and isinstance(base, dict) and "address" in base:
                    t["address"] = base["address"]

                sanitized.append(t)
            except Exception:
                # skip malformed enrichment row but keep going
                continue

        if not sanitized:
            return _neutral_fallback(tokens)

        return sanitized

    except Exception as e:
        try:
            logger.debug("Merging enrichment results failed: %s", e, exc_info=True)
        except Exception:
            pass
        return _neutral_fallback(tokens)   

def _merge_by_address(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge heterogeneous token rows by address, keeping the 'best' row and
    backfilling missing fields from others. 'Best' is chosen by:
      1) higher liquidity, then
      2) higher 24h volume (v24hUSD), then
      3) newer created/listed timestamp.

    This implementation also ensures canonical fields expected by the GUI and
    downstream logic are present:
      - 'v24hUSD'  (canonical per-cycle 24h volume from providers)
      - 'volume_24h' (GUI canonical key)
      - 'pct_change_5m', 'pct_change_1h', 'pct_change_24h' (period deltas)
    """
    def _f(x: Any, d: float = 0.0) -> float:
        try:
            if isinstance(x, dict):  # liquidity may arrive as {"usd": ...}
                return float(x.get("usd", d))
            return float(x)
        except Exception:
            return float(d)

    def _ts(x: Any) -> int:
        # accept seconds or ms
        try:
            t = int(x)
            return t // 1000 if t > 2_000_000_000 else t
        except Exception:
            return 0

    def _creation_ts(d: dict[str, Any]) -> int:
        for key in (
            "creation_timestamp",
            "pairCreatedAt",
            "created_at",
            "listedAt",
            "createdAt",
            "timestamp",
        ):
            v = d.get(key)
            ts = _ts(v)
            if ts > 0:
                return ts
        return 0

    def _v24h(d: dict[str, Any]) -> float:
        # normalize many provider aliases into a canonical numeric v24hUSD
        candidates = (
            d.get("v24hUSD"),
            d.get("volume_24h"),
            d.get("dex_volume_24h"),
            d.get("volume24h"),
            d.get("volume"),
            (d.get("volume", {}) or {}).get("h24") if isinstance(d.get("volume"), dict) else None,
        )
        for c in candidates:
            try:
                if c is None:
                    continue
                val = float(c)
                if val and val > 0:
                    return val
            except Exception:
                try:
                    # Some providers use strings with commas
                    s = str(c).replace(",", "")
                    val = float(s)
                    if val and val > 0:
                        return val
                except Exception:
                    continue
        return 0.0

    def _score(d: dict[str, Any]) -> tuple[float, float, int]:
        liq = _f(d.get("liquidity"), 0.0)
        vol = _v24h(d)
        ts = _ts(d.get("pairCreatedAt") or d.get("listedAt") or d.get("createdAt"))
        return (liq, vol, ts)

    merged: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        addr = row.get("address") or (row.get("baseToken") or {}).get("address")
        if not addr:
            continue
        cand: dict[str, Any] = dict(row)

        # Normalize volume/liquidity aliases so downstream coalesce can pick them up
        # set canonical numeric v24hUSD
        v = _v24h(cand)
        cand["v24hUSD"] = v if v > 0 else cand.get("v24hUSD", 0.0)
        # set canonical volume_24h for GUI
        cand["volume_24h"] = cand.get("volume_24h") or cand.get("v24hUSD") or cand.get("volume24h") or 0.0

        cand["liquidity"] = _f(cand.get("liquidity"), 0.0)

        # Provide dex_ aliases used by coalesce step
        cand.setdefault("dex_volume_24h", cand.get("v24hUSD"))
        if cand.get("dex_liquidity") is None:
            cand["dex_liquidity"] = cand.get("liquidity")
        if cand.get("dex_market_cap") is None and cand.get("fdv") is not None:
            cand["dex_market_cap"] = cand.get("fdv")

        # ensure every candidate carries a canonical creation_timestamp (seconds)
        cand["creation_timestamp"] = _creation_ts(cand)

        best = merged.get(addr)
        if best is None:
            merged[addr] = cand
            continue

        # choose the better of (best, cand) and also backfill missing values
        if _score(cand) > _score(best):
            for k, v in best.items():
                if k not in cand or cand.get(k) in (None, "", 0, 0.0):
                    cand[k] = v
            merged[addr] = cand
        else:
            for k, v in cand.items():
                if k not in best or best.get(k) in (None, "", 0, 0.0):
                    best[k] = v
            merged[addr] = best

    out = list(merged.values())
    # Final normalize: ensure canonical keys exist on outputs
    for d in out:
        d["creation_timestamp"] = _creation_ts(d)
        # ensure volume fields
        try:
            if not d.get("v24hUSD"):
                d["v24hUSD"] = _v24h(d)
        except Exception:
            d["v24hUSD"] = d.get("v24hUSD") or 0.0
        d["volume_24h"] = d.get("volume_24h") or d.get("v24hUSD") or 0.0
        # Ensure percent-change aliases exist (fill from common variants if present)
        d.setdefault("priceChange5m", d.get("price_change_5m") or d.get("pct_change_5m") or 0.0)
        d.setdefault("priceChange1h", d.get("price_change_1h") or d.get("pct_change_1h") or 0.0)
        d.setdefault("priceChange24h", d.get("price_change_24h") or d.get("pct_change_24h") or 0.0)
    return out

# --- Revised _build_live_shortlist (fix persistence NameError + respect dexscreener_post_cap=0) ---
async def _build_live_shortlist(
    *,
    session,
    solana_client,  # kept for signature parity; not used here
    config: dict[str, Any],
    queries: List[str],
    blacklist: Set[str],            # kept for parity; not used by this function
    failure_count: dict[str, int],  # kept for parity; not used by this function
) -> List[dict[str, Any]]:
    """
    Live discovery → merge → dedupe → pre-cut → enrich → diagnostics → coalesce → shortlist → persist.
    Also persists the raw discovered set for the GUI "Discovered Tokens" tab.
    """
    logger.info("LIVE-SHORTLIST(build=v3) starting with queries=%s", queries)

    disc = (config.get("discovery") or {})

    # Respect an explicit 0 in config (meaning "no cap"). Parse defensively.
    raw_post_cap = disc.get("dexscreener_post_cap", 300)
    try:
        post_cap = int(raw_post_cap)
    except Exception:
        post_cap = 300

    # pages similarly: allow explicit 0? (but default to 1 minimum)
    raw_pages = disc.get("dexscreener_pages", 1)
    try:
        pages = int(raw_pages)
        if pages < 1:
            pages = 1
    except Exception:
        pages = 1

    # 1) FETCH (Dexscreener paged search; keep your per-query loop)
    rows: List[dict[str, Any]] = []
    seen: Set[str] = set()
    for q in queries:
        try:
            chunk = await fetch_dexscreener_search(session, query=q, pages=pages)
        except Exception as e:
            logger.warning("Dexscreener fetch failed for query=%s: %s", q, e)
            chunk = []
        for pair in (chunk or []):
            addr = pair.get("address")
            if not addr or addr in seen:
                continue
            seen.add(addr)
            rows.append(pair)
        # Soft cap early if we're already above the intended working set
        # If post_cap is 0, treat as "no early soft cap" (i.e., don't break here).
        try:
            if post_cap and len(rows) >= (post_cap * 2):
                break
        except Exception:
            # if post_cap malformed, ignore and continue fetching until done
            pass

    logger.info("Pre-dedupe counts: total=%d", len(rows))

        # 2) MERGE/DEDUPE to address-canonical list
    merged: List[dict[str, Any]] = _merge_by_address(rows)
    logger.info("After merge_by_address: %d", len(merged))
    
    # ---------- NEW: discovery-level scoring.hard_floor pre-filter  ----------
    # Try to resolve the eligibility helper (sync) and apply it in-place to `merged`.
    try:
        try:
            # prefer local relative import
            from .eligibility import enforce_scoring_hard_floor  # type: ignore
        except Exception:
            from solana_trading_bot_bundle.trading_bot.eligibility import enforce_scoring_hard_floor  # type: ignore
    except Exception:
        enforce_scoring_hard_floor = None  # type: ignore

    try:
        if enforce_scoring_hard_floor:
            before_n = len(merged)
            kept: List[dict[str, Any]] = []
            rejected: List[dict[str, Any]] = []
            for t in merged:
                try:
                    allow, reason = enforce_scoring_hard_floor(t, config)
                    if not allow:
                        # annotate for diagnostics/UI
                        t["_reject_reason"] = reason or "hard_floor_reject"
                        # optional extra fields for debugging
                        try:
                            t["_liquidity_usd"] = float(t.get("liquidity") or t.get("dex_liquidity") or t.get("ray_liquidity_usd") or 0.0)
                            t["_v24h_usd"] = float(t.get("v24hUSD") or t.get("volume_24h") or t.get("volume24h") or 0.0)
                        except Exception:
                            pass
                        rejected.append(t)
                    else:
                        kept.append(t)
                except Exception:
                    # on any extraction error, be conservative and keep token
                    kept.append(t)
            merged = kept
            after_n = len(merged)
            try:
                logger.info(
                    "Discovery hard_floor prefilter: kept=%d rejected=%d (min_liq=%s min_vol=%s)",
                    after_n, (before_n - after_n),
                    str(((config.get("scoring") or {}).get("hard_floor") or {}).get("min_liquidity_usd")),
                    str(((config.get("scoring") or {}).get("hard_floor") or {}).get("min_volume_24h_usd")),
                )
                if rejected:
                    sample = rejected[:5]
                    samp_str = ", ".join(
                        f"{(s.get('symbol') or s.get('address') or 'UNK')}({_to_int(round(s.get('_liquidity_usd') or 0))}/${_to_int(round(s.get('_v24h_usd') or 0))})"  # type: ignore
                        for s in sample
                    )
                    logger.debug("Discovery hard_floor sample rejected: %s", samp_str)
            except Exception:
                logger.debug("Discovery hard_floor logging failed", exc_info=True)
    except Exception:
        logger.debug("Discovery hard_floor application failed (continuing)", exc_info=True)

    # Compatibility shim: ensure _db has an ensure_discovered_tokens_schema helper
    # Some builds expose a helper with a different name. Provide a safe alias or a
    # minimal implementation so persistence does not fail with AttributeError.
    try:
        if not hasattr(_db, "ensure_discovered_tokens_schema"):
            # Prefer an existing explicit helper if present
            for cand in (
                "ensure_discovered_tokens_schema",
                "ensure_tokens_schema",
                "ensure_shortlist_tokens_schema",
                "ensure_eligible_tokens_schema",
                "init_db",
                "init_schema",
            ):
                if hasattr(_db, cand) and callable(getattr(_db, cand)):
                    _db.ensure_discovered_tokens_schema = getattr(_db, cand)  # type: ignore[attr-defined]
                    logger.info("DB shim: mapped ensure_discovered_tokens_schema -> %s", cand)
                    break
            else:
                # Provide a minimal async implementation that creates the discovered_tokens table
                async def _ensure_discovered_tokens_schema_minimal(db_conn=None):
                    # If called without a connection, open one via the public connect_db helper
                    if db_conn is None:
                        async with _db.connect_db() as _db_conn:
                            await _ensure_discovered_tokens_schema_minimal(_db_conn)
                        return
                    try:
                        await db_conn.execute("""
                            CREATE TABLE IF NOT EXISTS discovered_tokens (
                                address TEXT PRIMARY KEY,
                                data TEXT,
                                name TEXT,
                                symbol TEXT,
                                price REAL,
                                liquidity REAL,
                                market_cap REAL,
                                v24hUSD REAL,
                                volume_24h REAL,
                                dexscreenerUrl TEXT,
                                dsPairAddress TEXT,
                                links TEXT,
                                created_at INTEGER,
                                creation_timestamp INTEGER
                            );
                        """)
                        # ensure the JSON upsert helper (bulk_upsert_tokens) can set `data`
                        await db_conn.commit()
                        logger.info("DB shim: created minimal discovered_tokens table")
                    except Exception:
                        logger.debug("DB shim: failed to create minimal discovered_tokens table", exc_info=True)

                _db.ensure_discovered_tokens_schema = _ensure_discovered_tokens_schema_minimal  # type: ignore[attr-defined]
    except Exception:
        # Defensive: don't let the shim crash imports
        logger.debug("Failed to apply DB shim for discovered_tokens schema", exc_info=True)

    # 3) PERSIST raw discovery for GUI "Discovered Tokens"
    try:
        # Restore expected config/local variables used by persistence logic
        prune_hours = int((config.get("discovery") or {}).get("discovered_prune_hours", 24))
        now_s = int(time.time())

        # ---- Normalize creation/created_at on the discovered set BEFORE persisting ----
        for _t in merged:
            ts = _derive_creation_ts_s(_t)  # seconds, robust helper
            _t["creation_timestamp"] = ts if ts > 0 else 0

            # created_at is what the DB prune uses; make sure it's present & sane
            try:
                created_at = int(_t.get("created_at") or 0)
            except Exception:
                created_at = 0
            if created_at <= 0:
                # prefer derived creation ts; otherwise fall back to "now"
                created_at = ts or now_s
            _t["created_at"] = created_at

            # keep a generic timestamp aligned as a fallback for any downstream readers
            try:
                stamp = int(_t.get("timestamp") or 0)
            except Exception:
                stamp = 0
            if stamp <= 0:
                _t["timestamp"] = created_at
        # ------------------------------------------------------------------------------

        # Robust existence check: call package helper if available; otherwise inline upsert
        persist_rows = []
        for r in merged:
            t = dict(r)
            # Ensure numeric canonical volume is present for GUI (prefer v24hUSD -> dex_volume_24h -> volume24h)
            try:
                v = (
                    float(t.get("v24hUSD") or 0.0)
                    or float(t.get("dex_volume_24h") or 0.0)
                    or float(t.get("volume24h") or 0.0)
                )
            except Exception:
                try:
                    v = float(str(t.get("v24hUSD") or t.get("dex_volume_24h") or t.get("volume24h") or 0).replace(",", ""))
                except Exception:
                    v = 0.0
            t["v24hUSD"] = float(t.get("v24hUSD") or v or 0.0)
            t["volume_24h"] = float(t.get("volume_24h") or t.get("v24hUSD") or t.get("dex_volume_24h") or 0.0)

            # Ensure liquidity numeric
            try:
                t["liquidity"] = float(t.get("liquidity") or t.get("dex_liquidity") or 0.0)
            except Exception:
                t["liquidity"] = 0.0

            # Ensure price if present
            try:
                if not t.get("price"):
                    p = t.get("price") or t.get("lastPrice") or t.get("value") or t.get("last_price")
                    t["price"] = float(p) if p is not None else 0.0
            except Exception:
                t["price"] = float(t.get("price") or 0.0)

            # Normalize pct_change aliases (best-effort)
            try:
                t["pct_change_5m"] = float(t.get("pct_change_5m") or t.get("priceChange5m") or t.get("price_change_5m") or 0.0)
            except Exception:
                t["pct_change_5m"] = 0.0
            try:
                t["pct_change_1h"] = float(t.get("pct_change_1h") or t.get("priceChange1h") or t.get("price_change_1h") or 0.0)
            except Exception:
                t["pct_change_1h"] = 0.0
            try:
                t["pct_change_24h"] = float(t.get("pct_change_24h") or t.get("priceChange24h") or t.get("price_change_24h") or 0.0)
            except Exception:
                t["pct_change_24h"] = 0.0

            persist_rows.append(t)

        # Prefer packaged helper if present
        packaged = globals().get("_persist_discovered_tokens")
        if packaged and callable(packaged):
            try:
                await packaged(persist_rows, prune_hours=prune_hours)  # type: ignore[misc]
            except Exception:
                # If packaged helper failed, fall through to inline fallback so we still persist
                logger.debug("Packaged _persist_discovered_tokens failed; falling back to inline.", exc_info=True)
                packaged = None

        if not packaged:
            # Inline fallback using _db helpers — robust to variations in helper names.
            async with _db.connect_db() as db:
                # Try to use an ensure_schema helper if present; tolerate different names:
                ensure_fn = getattr(_db, "ensure_discovered_tokens_schema", None) or getattr(_db, "ensure_tokens_schema", None) or getattr(_db, "ensure_schema", None)
                if callable(ensure_fn):
                    try:
                        await ensure_fn(db)
                    except Exception:
                        # If ensure fails, log and continue; next step will attempt to upsert/create table
                        logger.debug("ensure_discovered_tokens_schema/ensure_* failed", exc_info=True)
                else:
                    # As a last resort, run a minimal CREATE TABLE that matches what bulk_upsert_tokens expects.
                    try:
                        await db.execute(
                            """
                            CREATE TABLE IF NOT EXISTS discovered_tokens (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                address TEXT UNIQUE,
                                symbol TEXT,
                                name TEXT,
                                created_at INTEGER,
                                creation_timestamp INTEGER,
                                price REAL,
                                liquidity REAL,
                                market_cap REAL,
                                v24hUSD REAL,
                                volume_24h REAL,
                                dexscreenerUrl TEXT,
                                dsPairAddress TEXT,
                                links TEXT,
                                raw_json TEXT
                            );
                            """
                        )
                        await db.commit()
                        logger.debug("Created minimal discovered_tokens table via SQL fallback.")
                    except Exception:
                        logger.debug("Failed to create minimal discovered_tokens table via SQL fallback", exc_info=True)

                # Now attempt bulk upsert (existing helper)
                try:
                    await _db.bulk_upsert_tokens(db, "discovered_tokens", persist_rows)
                    if prune_hours and prune_hours > 0:
                        cutoff = now_s - int(prune_hours * 3600)
                        await db.execute("DELETE FROM discovered_tokens WHERE created_at < ?;", (cutoff,))
                        await db.commit()
                except Exception:
                    # If bulk_upsert_tokens is not present or fails, try a naive per-row INSERT/REPLACE
                    try:
                        for tok in persist_rows:
                            try:
                                await db.execute(
                                    "INSERT OR REPLACE INTO discovered_tokens (address, symbol, name, created_at, creation_timestamp, price, liquidity, market_cap, v24hUSD, volume_24h, dexscreenerUrl, dsPairAddress, links, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                                    (
                                        tok.get("address"),
                                        tok.get("symbol"),
                                        tok.get("name"),
                                        int(tok.get("created_at") or 0),
                                        int(tok.get("creation_timestamp") or 0),
                                        float(tok.get("price") or 0.0),
                                        float(tok.get("liquidity") or 0.0),
                                        float(tok.get("market_cap") or tok.get("mc") or 0.0),
                                        float(tok.get("v24hUSD") or 0.0),
                                        float(tok.get("volume_24h") or 0.0),
                                        tok.get("dexscreenerUrl") or "",
                                        tok.get("dsPairAddress") or "",
                                        json.dumps(tok.get("links") or []),
                                        json.dumps(tok),
                                    ),
                                )
                            except Exception:
                                # best-effort per-row upsert; ignore failing rows
                                logger.debug("Per-row upsert failed for %s", tok.get("address"), exc_info=True)
                        await db.commit()
                    except Exception:
                        logger.debug("Fallback per-row persistence failed", exc_info=True)

        logger.info(
            "Persisted %d discovered tokens into discovered_tokens (prune=%dh)",
            len(persist_rows), prune_hours
        )
    except Exception:
        # Visible warning for operators, plus debug-level stack for diagnosis
        logger.warning("Persisting discovered tokens failed (see debug for details)")
        logger.debug("Persist discovered_tokens exception details:", exc_info=True)

    # 4) OPTIONAL pre-cut (keep top-N by liquidity & v24hUSD before heavier steps)
    # If post_cap == 0, interpret as "no pre_cut cap" => keep full merged list.
    try:
        if post_cap == 0:
            pre_cut = sorted(
                merged,
                key=lambda d: (
                    float(d.get("liquidity", 0.0)),
                    float(d.get("v24hUSD", d.get("volume24h", 0.0))),
                ),
                reverse=True,
            )
        else:
            pre_cut = sorted(
                merged,
                key=lambda d: (
                    float(d.get("liquidity", 0.0)),
                    float(d.get("v24hUSD", d.get("volume24h", 0.0))),
                ),
                reverse=True,
            )[:max(0, post_cap)]
    except Exception:
        # Fail-safe: if sorting fails, keep merged as pre_cut
        pre_cut = merged[:]

    # 5) CAP workset so enrichment can cover 100% this cycle (align with Birdeye per-cycle cap)
    ENRICH_MAX = int(os.getenv("ENRICH_MAX", "60"))
    workset = pre_cut if len(pre_cut) <= ENRICH_MAX else pre_cut[:ENRICH_MAX]
    if len(pre_cut) > ENRICH_MAX:
        logger.info("Workset capped for enrichment: %d (of %d pre_cut)", ENRICH_MAX, len(pre_cut))

    # 6) ENRICH using your existing lazy enrichment helper
    try:
        if not workset:
            logger.info("Workset is empty after pre-cut; skipping enrichment.")
            enriched: List[dict[str, Any]] = []
        else:
            logger.info("Enriching %d tokens with price change data", len(workset))
            enriched = await _enrich_tokens_with_price_change_lazy(
                session=session,
                tokens=workset,
                logger=logger,
                blacklist=blacklist,
                failure_count=failure_count,
            )
        if not enriched:
            logger.warning("Enrichment returned no data; falling back to pre_cut (%d)", len(pre_cut))
            enriched = pre_cut
            for t in enriched:
                t.setdefault("_enriched", False)
    except Exception:
        logger.warning("Signal enrichment failed; using pre_cut without enrichment", exc_info=True)
        enriched = pre_cut
        for t in enriched:
            t.setdefault("_enriched", False)

    # 7) DIAGNOSTICS: enrichment coverage
    def _count_truthy(xs, key):
        c = 0
        for x in xs:
            v = x.get(key)
            if v is not None and (not isinstance(v, float) or v == v):  # not NaN
                c += 1
        return c

    n = len(enriched)
    have_mc    = _count_truthy(enriched, "market_cap")
    have_liq   = _count_truthy(enriched, "liquidity")
    have_vol24 = _count_truthy(enriched, "volume_24h")
    have_age   = _count_truthy(enriched, "token_age_minutes")
    logger.info("Enrichment coverage: n=%d mc=%d liq=%d vol24=%d age=%d", n, have_mc, have_liq, have_vol24, have_age)

    # 8) COALESCE metrics (fallback to Dexscreener fields when Birdeye missing)
    def _coalesce_metrics(t: dict[str, Any]) -> dict[str, Any]:
        if t.get("market_cap") is None:
            t["market_cap"] = t.get("fdv") or t.get("dex_market_cap")
        if t.get("liquidity") is None:
            t["liquidity"] = t.get("dex_liquidity") or t.get("reserve_usd") or t.get("liquidity")
        if t.get("volume_24h") is None:
            t["volume_24h"] = t.get("dex_volume_24h") or t.get("v24hUSD") or t.get("volume24h")
        return t

    enriched = [_coalesce_metrics(t) for t in enriched]

    # 9) Attempt to (re)bind tiebreaker & hard_floor into module globals (no local bindings)
    # This must be module-level (not inside _build_live_shortlist) so callers in functions
    # can use the global name without creating an accidental local binding.
    try:
        # Prefer the local private names if available
        from .eligibility import _apply_tech_tiebreaker as _tb, enforce_scoring_hard_floor as _hf  # type: ignore
        globals()["_apply_tech_tiebreaker"] = _tb
        globals()["enforce_scoring_hard_floor"] = _hf
    except Exception:
        try:
            # Packaged fallback path
            from solana_trading_bot_bundle.trading_bot.eligibility import _apply_tech_tiebreaker as _tb, enforce_scoring_hard_floor as _hf  # type: ignore
            globals()["_apply_tech_tiebreaker"] = _tb
            globals()["enforce_scoring_hard_floor"] = _hf
        except Exception:
            try:
                # Public API variant name
                from .eligibility import apply_tech_tiebreaker as _tb, enforce_scoring_hard_floor as _hf  # type: ignore
                globals()["_apply_tech_tiebreaker"] = _tb
                globals()["enforce_scoring_hard_floor"] = _hf
            except Exception:
                try:
                    from solana_trading_bot_bundle.trading_bot.eligibility import apply_tech_tiebreaker as _tb, enforce_scoring_hard_floor as _hf  # type: ignore
                    globals()["_apply_tech_tiebreaker"] = _tb
                    globals()["enforce_scoring_hard_floor"] = _hf
                except Exception:
                    # Last-resort: assign safe no-op fallbacks into globals
                    def _tb_fallback(tokens: list, config: dict, *, logger: Optional[logging.Logger] = None) -> list:
                        return list(tokens or [])
                    def _hf_fallback(token: dict, cfg: dict) -> tuple[bool, str]:
                        return True, "hard_floor missing (fallback allow)"
                    globals()["_apply_tech_tiebreaker"] = _tb_fallback
                    globals()["enforce_scoring_hard_floor"] = _hf_fallback

    # Now callers can safely call globals()["_apply_tech_tiebreaker"] by name `_apply_tech_tiebreaker`
    # without creating local binding problems.          
                      
    # 10) Fallback shortlist if coverage is weak
    eligible_tokens_preselect: List[dict[str, Any]] | None = None
    coverage_ratio = (have_mc / max(1, n)) if n else 0.0
    if coverage_ratio < 0.2 and n > 0:
        logger.warning("Weak enrichment (mc coverage=%.0f%%). Using fallback shortlist by 24h volume.", coverage_ratio * 100.0)
        MIN_FALLBACK = int(os.getenv("FALLBACK_SHORTLIST_MIN", "5"))
        fallback = sorted(enriched, key=lambda t: float(t.get("volume_24h") or 0.0), reverse=True)[:MIN_FALLBACK]
        for t in fallback:
            t.setdefault("_fallback_eligible", True)
        eligible_tokens_preselect = fallback

    tokens_for_selection = eligible_tokens_preselect if eligible_tokens_preselect is not None else enriched

    # 11) Bucket/shortlist with existing helper
    logger.info("Selecting mid-cap-centric shortlist from %d eligible tokens", len(tokens_for_selection))
    shortlist = await select_top_five_per_category(tokens_for_selection)

    # 11.5) FINAL CANONICALIZE for GUI/DB before persistence & return
    def _canon(t: dict[str, Any]) -> dict[str, Any]:
        """
        Final per-token canonicalization used before persisting or returning
        shortlist rows for the GUI. Ensures the GUI and downstream logic see
        stable/canonical keys (market_cap/mc, volume_24h / v24hUSD, pct_change_*,
        created_at/creation_timestamp, etc).
        """
        def _f(x, d=0.0):
            try:
                return float(x)
            except Exception:
                return float(d)

        def _i(x, d=0):
            try:
                return int(x)
            except Exception:
                return int(d)

        t = dict(t)  # copy so we don't mutate caller's object

        # price / mc / liq mirroring
        if "mc" in t and ("market_cap" not in t or _f(t.get("market_cap"), 0.0) <= 0):
            t["market_cap"] = _f(t["mc"], 0.0)
        if "market_cap" in t and ("mc" not in t or _f(t.get("mc"), 0.0) <= 0):
            t["mc"] = _f(t["market_cap"], 0.0)

        # Ensure canonical 24h volume is populated from any common alias
        vol_val = (
            t.get("volume_24h")
            or t.get("v24hUSD")
            or t.get("dex_volume_24h")
            or t.get("volume24h")
            or (t.get("volume") or {}).get("h24") if isinstance(t.get("volume"), dict) else t.get("volume")
            or 0.0
        )
        t["volume_24h"] = _f(vol_val, 0.0)
        # Keep v24hUSD for compatibility
        t["v24hUSD"] = _f(t.get("v24hUSD") or t["volume_24h"], t["volume_24h"])

        # Normalize percent-change aliases into stable pct_change_* keys used by GUI
        # Accept provider variants like priceChange1h, price_change_1h, pct_change_1h
        try:
            t["pct_change_5m"] = _f(
                t.get("pct_change_5m")
                or t.get("priceChange5m")
                or t.get("price_change_5m")
                or t.get("price_change_5m_pct")
                or t.get("priceChange5mPct")
                or 0.0
            )
            t["pct_change_1h"] = _f(
                t.get("pct_change_1h")
                or t.get("priceChange1h")
                or t.get("price_change_1h")
                or t.get("priceChange1hPct")
                or 0.0
            )
            t["pct_change_24h"] = _f(
                t.get("pct_change_24h")
                or t.get("priceChange24h")
                or t.get("price_change_24h")
                or t.get("priceChange24hPct")
                or 0.0
            )
        except Exception:
            # fallback defaults
            t.setdefault("pct_change_5m", 0.0)
            t.setdefault("pct_change_1h", 0.0)
            t.setdefault("pct_change_24h", 0.0)

        # holders/liquidity present and numeric
        t["holderCount"] = _i(t.get("holderCount"), 0)
        t["liquidity"] = _f(t.get("liquidity"), 0.0)

        # Dexscreener fields as strings
        t["dexscreenerUrl"] = t.get("dexscreenerUrl") or ""
        t["dsPairAddress"] = t.get("dsPairAddress") or ""

        # Pretty market cap for UI
        try:
            from .utils_exec import format_market_cap
        except ImportError:
            from solana_trading_bot_bundle.trading_bot.utils_exec import format_market_cap

        t["mcFormatted"] = format_market_cap(_f(t.get("market_cap", t.get("mc", 0.0)), 0.0))

        # ---- canonical creation timestamp (seconds) for bucket logic ----
        ts = _derive_creation_ts_s(t)   # uses the shared helper
        t["creation_timestamp"] = ts if ts > 0 else 0

        # mirror into created_at if DB/GUI consumers expect it
        if _i(t.get("created_at"), 0) <= 0:
            t["created_at"] = t["creation_timestamp"]

        # mark enrichment default if missing
        t.setdefault("_enriched", True)

        return t

    shortlist = [_canon(t) for t in (shortlist or [])]

    # 12) Persist shortlist for GUI "Shortlist/Eligible" view
    try:
        await persist_eligible_shortlist(shortlist, prune_hours=168)
        logger.info("Persisted %d tokens into eligible_tokens (shortlist view)", len(shortlist))

        # Keep discovered_tokens in sync so the GUI (which reads discovered_tokens)
        # immediately sees the same canonical mints the buy pipeline uses.
        try:
            # Resolve DB path using the helper already in this module.
            db_path = _db_path_from_config(config) if callable(globals().get("_db_path_from_config")) else _default_db_path()
            try:
                inserted = await _sync_eligible_to_discovered_once(db_path, logger=logger)
                if inserted:
                    logger.info("Synced %d eligible tokens into discovered_tokens for GUI visibility", inserted)
            except Exception:
                logger.exception("eligible->discovered post-persist sync failed")
        except Exception:
            # Non-fatal: if resolving DB path or scheduling the sync fails, continue.
            logger.debug("Failed to run eligible->discovered sync (non-fatal)", exc_info=True)

    except Exception:
        logger.debug("Persisting shortlist failed", exc_info=True)

    return shortlist

# --- Build+trim+enrich live shortlist once -----------------------------------
async def _fetch_live_shortlist_once(
    *,
    session: aiohttp.ClientSession,
    solana_client,
    config: dict,
    queries: list[str],
    blacklist: set[str] | list[str],
    failure_count: dict,
    logger: logging.Logger
) -> list[dict]:
    """
    Build a live shortlist once, then apply:
      1) select_top_five_per_category
      2) _enrich_shortlist_with_signals (mark _enriched=False if enrichment empty)

    Returns [] if nothing viable could be produced.
    """
    try:
        tokens = await _build_live_shortlist(
            session=session,
            solana_client=solana_client,
            config=config,
            queries=queries,
            blacklist=blacklist,
            failure_count=failure_count,
        )
    except Exception:
        logger.debug("Live shortlist build failed", exc_info=True)
        return []

    if not tokens:
        return []

    # Keep shortlist small for trading path
    try:
        tokens = await select_top_five_per_category(tokens)
    except Exception:
        logger.debug("select_top_five_per_category failed", exc_info=True)

    # Enrich, but keep neutral path if it returns empty
    try:
        enriched = await _enrich_shortlist_with_signals(tokens)
        if enriched:
            tokens = enriched
        else:
            logger.warning(
                "Live enrichment returned empty; keeping un-enriched shortlist."
            )
            for t in tokens:
                t.setdefault("_enriched", False)
    except Exception:
        logger.debug("Signal enrichment on live shortlist failed", exc_info=True)

    # --- FINAL CANONICALIZE (recommended) ---
    try:
        def _f(x, d=0.0):
            try:
                return float(x)
            except Exception:
                return float(d)

        def _i(x, d=0):
            try:
                return int(x)
            except Exception:
                return int(d)

        def _canon_token(t: dict) -> dict:
            t = dict(t or {})

            # Ensure canonical market cap mirrors
            mc = _f(t.get("market_cap", t.get("mc", 0.0)))
            t["mc"] = mc
            t["market_cap"] = mc

            # Canonical 24h volume for GUI (coalesce common provider keys)
            if t.get("volume_24h") is None:
                t["volume_24h"] = _f(
                    t.get("dex_volume_24h") or t.get("v24hUSD") or t.get("volume24h"),
                    0.0
                )

            # Safe defaults the GUI expects
            t.setdefault("price", _f(t.get("price"), 0.0))
            t.setdefault("liquidity", _f(t.get("liquidity"), 0.0))
            t.setdefault("holderCount", _i(t.get("holderCount"), 0))
            t.setdefault("pct_change_5m", _f(t.get("pct_change_5m"), 0.0))
            t.setdefault("pct_change_1h", _f(t.get("pct_change_1h"), 0.0))
            t.setdefault("pct_change_24h", _f(t.get("pct_change_24h"), 0.0))

            # Pretty market cap
            try:
                from .utils_exec import format_market_cap
            except Exception:
                from solana_trading_bot_bundle.trading_bot.utils_exec import format_market_cap
            t["mcFormatted"] = format_market_cap(mc)

            return t

        tokens = [_canon_token(t) for t in (tokens or [])]
    except Exception:
        logger.debug("final canonicalize failed; continuing.", exc_info=True)

    return tokens or []

# --- Dexscreener per-token cache / rate limit (for creation time & any per-token fallbacks) ---

_DS_TOKEN_CACHE: dict[str, tuple[float, dict]] = {}  # addr -> (ts, data)
_DS_CACHE_TTL = int(os.getenv("DS_TOKEN_CACHE_TTL", "600"))  # seconds
_DS_SEM = asyncio.Semaphore(int(os.getenv("DS_TOKEN_CONCURRENCY", "6")))

# Birdeye per-cycle disable flag when a 401 is encountered
_BIRDEYE_401_SEEN = False

# --- PID / Heartbeat files (module-scope constants) ---
# prefer_appdata_file must already be imported from your constants module
PID_FILE        = prefer_appdata_file("bot.pid")
HEARTBEAT_FILE  = prefer_appdata_file("heartbeat")
STARTED_AT_FILE = prefer_appdata_file("started_at")  # single source of truth

_last_hb = 0  # for throttled heartbeat


def _write_pid_file() -> None:
    """
    Best-effort atomic pid write. If PID file already exists, overwrite only if owner matches
    or if we can take over later through acquire_single_instance_or_explain.
    """
    try:
        pid_path = str(Path(PID_FILE))
        created = _atomic_create_pid_file(pid_path, os.getpid())
        if not created:
            # Best-effort overwrite (non-atomic) when another method already created the file.
            try:
                Path(pid_path).write_text(str(os.getpid()), encoding="utf-8")
            except Exception:
                logger.debug("Fallback PID write failed", exc_info=True)
    except Exception:
        logger.debug("PID write failed", exc_info=True)

    # Write started_at once (durable)
    try:
        p = Path(STARTED_AT_FILE)
        if not p.exists():
            # atomic create semantics using os.open with O_CREAT|O_EXCL
            try:
                _atomic = False
                try:
                    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(str(int(_time.time())))
                    _atomic = True
                except FileExistsError:
                    _atomic = False
                except Exception:
                    _atomic = False
                if not _atomic:
                    p.write_text(str(int(_time.time())), encoding="utf-8")
            except Exception:
                logger.debug("started_at write fallback failed", exc_info=True)
    except Exception:
        logger.debug("started_at write failed", exc_info=True)


def _heartbeat(throttle_s: int = 5) -> None:
    """
    Write STARTED_AT_FILE (once) and update HEARTBEAT_FILE periodically.
    Uses atomic write where practical.
    """
    global _last_hb
    now = int(time.time())

    # Ensure started_at exists (write-once)
    try:
        started_at_path = Path(STARTED_AT_FILE)
        if not started_at_path.exists():
            try:
                fd = os.open(str(started_at_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(str(now))
            except FileExistsError:
                pass
            except Exception:
                try:
                    started_at_path.write_text(str(now), encoding="utf-8")
                except Exception:
                    logger.debug("Started-at write failed", exc_info=True)
    except Exception:
        logger.debug("Started-at write failed", exc_info=True)

    # Throttled heartbeat
    if now - _last_hb < throttle_s:
        return
    _last_hb = now
    try:
        # atomic replace via temporary file then os.replace minimizes race
        hb_path = Path(HEARTBEAT_FILE)
        tmp = hb_path.with_suffix(".tmp")
        try:
            tmp.write_text(str(now), encoding="utf-8")
            os.replace(str(tmp), str(hb_path))
        except Exception:
            # fallback direct write
            hb_path.write_text(str(now), encoding="utf-8")
    except Exception:
        logger.debug("Heartbeat write failed", exc_info=True)

# --- Single-instance guard & heartbeat loop ------------------------------------
import errno
import time as _time
from pathlib import Path

def _read_int_file(p: str) -> int | None:
    """
    Read an integer from file p. Return None on missing file, empty content, parse
    error, or other read/parsing problems. Avoid noisy tracebacks for the common
    'file missing' case (heartbeat/pid absent on first run).
    """
    try:
        path = Path(p)
        if not path.exists():
            # common and expected (first run or cleaned up) — return None quietly
            return None
        txt = path.read_text(encoding="utf-8").strip()
        if not txt:
            return None
        try:
            return int(txt)
        except ValueError:
            # malformed content — debug enough to help but don't log full traceback
            logger.debug("Failed to parse integer from %s: %r", p, txt)
            return None
    except FileNotFoundError:
        # race: file removed between exists() and read_text(); treat as missing
        return None
    except PermissionError:
        logger.debug("Permission denied reading int file %s", p, exc_info=False)
        return None
    except Exception:
        # Unexpected: log full traceback for diagnosis
        try:
            logger.debug("Failed to read int from %s", p, exc_info=True)
        except Exception:
            pass
        return None

def _process_alive(pid: int) -> bool:
    """
    Return True if PID appears to be a running process.
    - Prefer psutil when available.
    - Fallbacks:
        - On POSIX: use os.kill(pid, 0) (works for existence check).
        - On other platforms without psutil: conservatively return True.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    if not isinstance(pid, int) or pid <= 0:
        return False

    if psutil:
        try:
            p = psutil.Process(pid)
            return p.is_running() and (p.status() != psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            return False
        except Exception:
            try:
                logger.debug("psutil.Process check failed for pid=%s", pid, exc_info=True)
            except Exception:
                pass
            # conservative fallback to True to avoid takeover when unsure
            return True

    # No psutil: POSIX check using kill(0)
    try:
        if os.name == "posix":
            try:
                os.kill(pid, 0)
            except OSError as e:
                if e.errno == errno.ESRCH:
                    return False
                # EPERM means exists but no permission to signal -> treat as alive
                if e.errno == errno.EPERM:
                    return True
                return True
            else:
                return True
        # On Windows without psutil we can't reliably check; be conservative
        return True
    except Exception:
        try:
            logger.debug("Fallback process alive check failed for pid=%s", pid, exc_info=True)
        except Exception:
            pass
        return True
    
def _atomic_create_pid_file(pid_path: str, pid: int) -> bool:
    """
    Try to atomically create PID file with exclusive semantics.
    Returns True if creation succeeded (we created it), False if file already exists or on failure.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = 0o644
    try:
        fd = os.open(pid_path, flags, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(pid))
            f.flush()
            os.fsync(f.fileno())
        return True
    except FileExistsError:
        return False
    except Exception:
        try:
            logger.debug("Atomic create pid file failed for %s", pid_path, exc_info=True)
        except Exception:
            pass
        return False    

def acquire_single_instance_or_explain(stale_after_s: int = 180) -> bool:
    """
    Try to acquire single-instance lock.
    Uses atomic create first to avoid the check-then-write race. If PID exists, check freshness
    and process aliveness. If stale, attempt takeover via removal + atomic create (with retries).
    """
    pid_path = Path(PID_FILE)
    hb_path = Path(HEARTBEAT_FILE)

    # Try atomic create first — fast path when no existing instance
    try:
        got = _atomic_create_pid_file(str(pid_path), os.getpid())
        if got:
            _heartbeat(throttle_s=0)
            logger.info("PID guard: acquired lock (atomic create).")
            return True
    except Exception:
        # continue to legacy logic on any failure
        pass

    # If file exists, read previous PID & heartbeat time
    prev_pid = _read_int_file(str(pid_path))
    hb_ts = _read_int_file(str(hb_path)) or 0
    now = int(_time.time())
    fresh = (now - hb_ts) <= max(30, stale_after_s)

    # If previous PID looks alive and heartbeat is fresh, refuse to start.
    try:
        if prev_pid and _process_alive(prev_pid) and fresh:
            logger.warning(
                "Another bot instance appears to be running (pid=%s, heartbeat %ds ago). Refusing to start.",
                prev_pid, max(0, now - hb_ts)
            )
            return False
    except Exception:
        # on error be conservative
        logger.warning("Single-instance check encountered an error; refusing to start.")
        return False

    # Stale or dead -> try to take over: remove stale PID file and try atomic create again
    logger.info(
        "PID guard: previous instance looks stale or dead (pid=%s, last heartbeat=%s). Attempting takeover.",
        prev_pid, f"{int(now - hb_ts)}s ago" if hb_ts else "unknown"
    )

    # Try to remove stale pid file and create our pid file atomically (retry loop to avoid race)
    for attempt in range(3):
        try:
            # remove stale file
            try:
                pid_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                logger.debug("Failed to unlink stale pid file (attempt %d)", attempt, exc_info=True)

            # attempt atomic create again
            try:
                got = _atomic_create_pid_file(str(pid_path), os.getpid())
                if got:
                    _heartbeat(throttle_s=0)
                    logger.info("PID guard: acquired lock after takeover.")
                    return True
            except Exception:
                pass

            # small delay between retries
            time.sleep(0.1 + attempt * 0.05)
        except Exception:
            break

    logger.warning("PID guard: failed to acquire lock after takeover attempts; refusing to start.")
    return False


async def heartbeat_task(stop_event: asyncio.Event, interval: int = 5) -> None:
    """
    Background task that updates heartbeat regularly.
    """
    try:
        while not stop_event.is_set():
            _heartbeat(throttle_s=interval)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                # timeout just means "emit another heartbeat"
                continue
    except asyncio.CancelledError:
        # Optional last stamp
        _heartbeat(throttle_s=0)
        raise
    except Exception:
        logger.debug("heartbeat_task error", exc_info=True)

def release_single_instance() -> None:
    """
    Remove PID/heartbeat files on clean exit, but only if we appear to own them.
    Safer than unconditional removal.
    """
    try:
        existing_pid = _read_int_file(PID_FILE)
        if existing_pid is None or existing_pid == os.getpid():
            for p in (PID_FILE, HEARTBEAT_FILE, STARTED_AT_FILE):
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    logger.debug("Cleanup failed for %s", p, exc_info=True)
        else:
            logger.info("Not removing PID file; owned by PID %s.", existing_pid)
    except Exception:
        logger.debug("PID cleanup failed", exc_info=True)


# --- Uniform trade-event logger (single definition) ---
def _log_trade_event(event: str, **kw):
    try:
        msg = " | ".join([event] + [f"{k}={kw[k]}" for k in sorted(kw.keys())])
        logger.info(msg)
    except Exception:
        logger.info("%s %s", event, kw)


def _peek_tokens(tokens: list[dict], n: int = 5) -> str:
    out = []
    for t in tokens[:n]:
        sym = t.get("symbol", "?")
        addr = t.get("address", "?")
        out.append(f"{sym}({addr[:6]}…)")
    return ", ".join(out)

# =========================
# DRY RUN / TEST MODE HELPERS (master switch)
# =========================
def _dry_run_on(cfg: Dict[str, Any] | None = None) -> bool:
    """
    Returns True if DRY RUN mode is active via either the environment or config.

    Sources:
      - ENV: DRY_RUN in {"1","true","yes","on","y"} (case-insensitive)
      - CONFIG: config["trading"]["dry_run"] == True (or config["bot"]["dry_run"] for back-compat)

    Use this at execution choke points (buy/sell executors, ATA creation, send layer).
    """

    env_flag = str(os.getenv("DRY_RUN", "0")).lower() in ("1", "true", "yes", "on", "y")
    if cfg is None:
        return env_flag
    # Prefer 'trading' section; fall back to 'bot' for older configs.
    t = (cfg.get("trading") or {})
    b = (cfg.get("bot") or {})
    return env_flag or bool(t.get("dry_run", b.get("dry_run", False)))

def _dry_run_txid(kind: str = "OP") -> str:
    """Generate a synthetic txid for paper trades (e.g., DRYRUN-BUY / DRYRUN-SELL)."""
    return f"DRYRUN-{str(kind or 'OP').upper()}"

def _skip_onchain_send(cfg: Dict[str, Any] | None = None) -> bool:
    """
    True if we should NOT broadcast:
      - DRY RUN is on
      - or ENV DISABLE_SEND_TX in {"1","true","yes","on","y"}
      - or config.trading.send_transactions is False
    """

    if _dry_run_on(cfg):
        return True
    if str(os.getenv("DISABLE_SEND_TX", "0")).lower() in ("1", "true", "yes", "on", "y"):
        return True
    if cfg:
        t = (cfg.get("trading") or {})
        if t.get("send_transactions") is False:
            return True
    return False

LAMPORTS_PER_SOL = 1_000_000_000  # keep your existing constant if already defined

async def has_min_balance(
    wallet,
    config: dict,
    session: aiohttp.ClientSession | None = None,
    min_sol: float | None = None,
    solana_client: Any = None,
) -> bool:
    """
    Check whether wallet has at least min_sol (float). Uses provided solana_client if present;
    otherwise creates/owns a client and closes it before returning.
    Returns False on any error.

    This implementation is defensive about:
      - wallet being a Keypair-like with .pubkey(), or already a Pubkey/string
      - many RPC response shapes (object with .value, nested .value, dict with 'result'/'value', stub objects)
      - ensuring any client created here is closed
    """
    try:
        threshold = float(min_sol) if (min_sol is not None) else float((config.get("bot") or {}).get("required_sol", 0.0))
    except Exception:
        threshold = float(min_sol or 0.0)

    if threshold <= 0:
        return True

    client_owned = False
    client = solana_client
    rpc_url = (config.get("solana") or {}).get("rpc_endpoint") or "https://api.mainnet-beta.solana.com"

    # Helper to extract a lamports int from many possible RPC response shapes
    def _extract_lamports(resp: Any) -> Optional[int]:
        try:
            if resp is None:
                return None

            # solana-py typed response: has .value which may be int or object
            if hasattr(resp, "value"):
                v = getattr(resp, "value", None)
                if isinstance(v, int):
                    return v
                # nested object with .lamports or .value
                if hasattr(v, "lamports"):
                    maybe = getattr(v, "lamports", None)
                    if isinstance(maybe, int):
                        return maybe
                if hasattr(v, "value"):
                    maybe = getattr(v, "value", None)
                    if isinstance(maybe, int):
                        return maybe

            # RPC raw-like: .result -> dict with "value"
            if hasattr(resp, "result"):
                res = getattr(resp, "result", None)
                if isinstance(res, dict):
                    val = res.get("value")
                    if isinstance(val, int):
                        return val
                    if isinstance(val, dict):
                        if isinstance(val.get("lamports"), int):
                            return val.get("lamports")
                        if isinstance(val.get("value"), int):
                            return val.get("value")

            # dict-shaped responses
            if isinstance(resp, dict):
                if "result" in resp and isinstance(resp["result"], dict):
                    v = resp["result"].get("value")
                    if isinstance(v, int):
                        return v
                    if isinstance(v, dict):
                        if isinstance(v.get("lamports"), int):
                            return v.get("lamports")
                        if isinstance(v.get("value"), int):
                            return v.get("value")
                if "value" in resp and isinstance(resp["value"], int):
                    return resp["value"]

            # last-ditch: small wrapper object with .lamports
            if hasattr(resp, "lamports") and isinstance(getattr(resp, "lamports"), int):
                return getattr(resp, "lamports")
        except Exception:
            return None
        return None

    # Resolve pubkey argument in a robust way
    try:
        if hasattr(wallet, "pubkey") and callable(getattr(wallet, "pubkey")):
            pubkey_arg = wallet.pubkey()
        else:
            pubkey_arg = wallet
    except Exception:
        pubkey_arg = wallet

    try:
        if client is None:
            client = _new_async_client(rpc_url)
            client_owned = True

        # prefer to pass the commitment object if available
        if COMMIT_PROCESSED is not None:
            resp = await client.get_balance(pubkey_arg, commitment=COMMIT_PROCESSED)
        else:
            resp = await client.get_balance(pubkey_arg)

        lamports = _extract_lamports(resp)
        if isinstance(lamports, int):
            sol = lamports / LAMPORTS_PER_SOL
            logger.debug("Wallet balance: %.6f SOL (need >= %.6f)", sol, threshold)
            return sol >= threshold

        # If we didn't find lamports but response includes a ui_amount field path (some variants),
        # try to extract ui_amount and treat it as SOL (not lamports)
        try:
            ui_amount = None
            if hasattr(resp, "value"):
                val = getattr(resp, "value", None)
                if hasattr(val, "ui_amount"):
                    ui_amount = getattr(val, "ui_amount", None)
                elif isinstance(val, dict) and "uiAmount" in val:
                    ui_amount = val.get("uiAmount")
                elif isinstance(val, dict) and "ui_amount" in val:
                    ui_amount = val.get("ui_amount")

            if isinstance(resp, dict):
                if "result" in resp and isinstance(resp["result"], dict):
                    rval = resp["result"].get("value", {})
                    if isinstance(rval, dict):
                        ui_amount = ui_amount or rval.get("uiAmount") or rval.get("ui_amount")
                # top-level value.uiAmount
                if isinstance(resp.get("value"), dict):
                    ui_amount = ui_amount or resp.get("value", {}).get("uiAmount") or resp.get("value", {}).get("ui_amount")

            if ui_amount is not None:
                try:
                    sol = float(ui_amount)
                    logger.debug("Wallet balance (via ui_amount): %.6f SOL (need >= %.6f)", sol, threshold)
                    return sol >= threshold
                except Exception:
                    pass
        except Exception:
            pass

        logger.warning("has_min_balance: unexpected balance response shape: %r", resp)
        return False
    except Exception as e:
        logger.warning("has_min_balance: balance check failed: %s", e, exc_info=True)
        return False
    finally:
        if client_owned and client is not None:
            try:
                await client.close()
            except Exception:
                pass

    # unreachable: preserved explicit return as defensive fallback
    return False

# =========================
# Birdeye master gate (env + optional config)
# =========================
def _birdeye_allowed(config: Dict[str, Any] | None = None) -> bool:
    """
    Central switch for Birdeye usage.
    Priority: ENV overrides > config['sources']['birdeye_enabled'] (defaults True).
    - FORCE_DISABLE_BIRDEYE=1 -> OFF
    - BIRDEYE_ENABLE=0        -> OFF
    - else -> ON unless config disables
    """

    if str(os.getenv("FORCE_DISABLE_BIRDEYE", "0")).strip().lower() in ("1", "true", "yes", "on", "y"):
        return False
    env_enable = str(os.getenv("BIRDEYE_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on", "y")
    if not env_enable:
        return False
    try:
        if config and isinstance(config, dict):
            return bool((config.get("sources") or {}).get("birdeye_enabled", True))
    except Exception:
        pass
    return True

# =========================
# PATCH: Aggressive exits (partial TPs + trailing)
# =========================

def _load_exit_cfg(config: Dict[str, Any], token: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Build the exit config used by the trading engine.
    If token is provided and the token matches the configured "qualified_runner_filters",
    apply overrides from qualified_runner_exits.
    """
    # 1) base bot-level defaults (backwards-compatible names)
    b = config.get("bot", {}) if isinstance(config, dict) else {}
    exit_cfg: Dict[str, Any] = {
        "stop_loss_x": float(b.get("stop_loss", 0.85)),
        "tp1_x": float(b.get("tp1_x", 1.30)),
        "tp2_x": float(b.get("tp2_x", 2.00)),
        "tp1_pct_to_sell": float(b.get("tp1_pct_to_sell", 0.50)),
        "tp2_pct_of_remaining": float(b.get("tp2_pct_of_remaining", 0.60)),
        "breakeven_plus": float(b.get("breakeven_plus", 0.02)),
        "trail_pct_after_tp1": float(b.get("trail_pct_after_tp1", 0.18)),
        "trail_pct_moonbag": float(b.get("trail_pct_moonbag", 0.30)),
        "no_tp_cutoff_hours": float(b.get("no_tp_cutoff_hours", 6.0)),
        # optional extras (kept for backward compatibility)
        "no_tp_min_profit_x": float(b.get("no_tp_min_profit_x", 1.10)),
        "max_hold_hours": float(b.get("max_hold_hours", 48.0)),
        "sell_slippage_bps": int(b.get("sell_slippage_bps", 150)),
    }

    # 2) qualified-runner conditional overrides (if configured)
    try:
        q_filters = config.get("qualified_runner_filters", {}) or {}
        q_exits = config.get("qualified_runner_exits", {}) or {}
    except Exception:
        q_filters, q_exits = {}, {}

    def _to_num_loose(x, default: float = 0.0) -> float:
        """Convert many common representations to float: dicts with 'usd', strings with commas, ints, floats."""
        try:
            if x is None:
                return float(default)
            # dict like {"usd": 12345}
            if isinstance(x, dict):
                # prefer usd key if present
                for k in ("usd", "USD", "value"):
                    if k in x:
                        return float(x.get(k) or default)
                # otherwise try the first numeric-looking value
                vals = [v for v in x.values() if isinstance(v, (int, float, str))]
                if vals:
                    return float(vals[0])
                return float(default)
            # strings with commas or currency symbols
            if isinstance(x, str):
                s = x.strip().replace(",", "").replace("$", "")
                if s == "":
                    return float(default)
                return float(s)
            return float(x)
        except Exception:
            return float(default)

    def _token_matches_qualified(tok: Dict[str, Any]) -> bool:
        # If no token or no filters configured, do not match
        if not tok or not q_filters:
            return False

        reasons: list[str] = []

        try:
            liq = _to_num_loose(tok.get("liquidity") or tok.get("liquidity_usd") or tok.get("liq") or tok.get("liquidity_usd_usd") or 0.0)
            vol = _to_num_loose(tok.get("volume_24h") or tok.get("volume") or tok.get("vol24") or tok.get("v24hUSD") or 0.0)
            mc  = _to_num_loose(tok.get("market_cap") or tok.get("mc") or tok.get("fdv") or 0.0)
        except Exception:
            # conservative: if parsing fails, do not match
            return False

        # Threshold checks (only enforced when provided > 0)
        min_liq = _to_num_loose(q_filters.get("min_liquidity_usd", 0))
        if min_liq > 0 and liq < min_liq:
            reasons.append(f"liq {liq:.0f} < {min_liq:.0f}")
        min_vol = _to_num_loose(q_filters.get("min_volume_24h_usd", 0))
        if min_vol > 0 and vol < min_vol:
            reasons.append(f"vol {vol:.0f} < {min_vol:.0f}")
        min_mc = _to_num_loose(q_filters.get("min_market_cap_usd", 0))
        if min_mc > 0 and mc < min_mc:
            reasons.append(f"mc {mc:.0f} < {min_mc:.0f}")

        # RugCheck safety (optional). Heuristic:
        # - If token has explicit 'safe' or 'labels_ok', treat as safe.
        # - Otherwise, if any danger label present, reject.
        # - If no rugcheck info at all, treat as not qualified (conservative).
        if q_filters.get("require_rugcheck_safe"):
            labels = set()
            try:
                raw = tok.get("labels") or tok.get("rugcheck_labels") or tok.get("rugcheck_label") or []
                if isinstance(raw, str):
                    labels.add(raw.strip().lower())
                else:
                    for r in (raw or []):
                        try:
                            labels.add(str(r).strip().lower())
                        except Exception:
                            continue
                meta = tok.get("rugcheck_meta") or {}
                for r in (meta.get("labels_sample") or []):
                    labels.add(str(r).strip().lower())
            except Exception:
                labels = set()

            # If we found explicit indicators of safety, accept them
            if labels:
                if ("safe" in labels) or ("labels_ok" in labels) or ("no_labels" in labels):
                    pass  # treat as ok
                else:
                    # check for a danger label set if present in module scope
                    try:
                        danger = set(globals().get("DANGER_LABELS", set()))
                        if not danger:
                            # fallback conservative list
                            danger = {"rugpull", "scam", "honeypot", "malicious", "blocked", "high_risk", "no_liquidity"}
                        if labels & danger:
                            reasons.append(f"rugcheck danger: {', '.join(sorted(labels & danger))}")
                        else:
                            # labels present but no danger labels -> accept
                            pass
                    except Exception:
                        reasons.append("rugcheck unknown (error)")
            else:
                reasons.append("no rugcheck labels")

        if reasons:
            # optional debug log if logger exists
            try:
                logger.debug("Qualified-runner check failed for %s: %s", tok.get("symbol") or tok.get("address"), "; ".join(reasons))
            except Exception:
                pass
            return False

        return True

    # 3) apply overrides if token qualifies
    try:
        if token and q_exits and _token_matches_qualified(token):
            try:
                logger.info(
                    "Qualified runner promotion: applying qualified_runner_exits for token %s (%s)",
                    token.get("symbol") or token.get("name") or token.get("address"),
                    token.get("address") or ""
                )
            except Exception:
                pass

            for k, v in q_exits.items():
                if k in exit_cfg:
                    try:
                        exit_cfg[k] = float(v)
                    except Exception:
                        exit_cfg[k] = v
                else:
                    try:
                        exit_cfg[k] = float(v)
                    except Exception:
                        exit_cfg[k] = v
    except Exception:
        pass

    return exit_cfg

async def _get_trade_state(token_address: str) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "tp1_done": False,
        "tp2_done": False,
        "highest_price": None,
        "breakeven_floor": None,
        "no_tp_deadline": None,
        "hard_exit_deadline": None,
        "trail_pct": None,
    }
    try:
        status = await get_token_trade_status(token_address)
        if status and isinstance(status, dict):
            ts = status.get("trade_state") or {}
            if not ts and "highest_price" in status:
                ts = {k: status.get(k) for k in state.keys() if k in status}
            if isinstance(ts, dict):
                state.update({k: ts.get(k, state[k]) for k in state})
                return state
    except Exception:
        pass

    try:
        cached = await get_cached_token_data(token_address)
        if cached and isinstance(cached, dict):
            ts = cached.get("trade_state") or {}
            if isinstance(ts, dict):
                state.update({k: ts.get(k, state[k]) for k in state})
    except Exception:
        pass
    return state

async def _save_trade_state(token_address: str, trade_state: Dict[str, Any]) -> None:
    try:
        await update_token_record(
            token={"address": token_address, "trade_state": dict(trade_state)},
            buy_price=None, buy_txid=None, buy_time=None, is_trading=True
        )
    except Exception:
        logger.debug("Failed to persist trade_state for %s", token_address, exc_info=True)

def _update_highest(state: Dict[str, Any], current_price: float) -> None:
    hp = state.get("highest_price")
    if hp is None or current_price > float(hp or 0):
        state["highest_price"] = float(current_price)

def _sell_amount_lamports(token_balance_tokens: float, pct: float, default_decimals: int = 9) -> int:
    """
    Compute base-unit amount to sell from a whole-token balance.
    token_balance_tokens: balance in WHOLE tokens (not lamports/base units)
    pct: 0..1 fraction of the balance to sell
    default_decimals: SPL token decimals (9 for SOL-wrapped style tokens unless overridden)
    """
    pct = max(0.0, min(1.0, float(pct)))
    return max(1, int(token_balance_tokens * pct * (10 ** default_decimals)))

# ---- Jupiter helpers -------------------------------------------------
import asyncio
from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError

# Trip this to True for the remainder of the *current cycle* when Jupiter DNS/connector fails.
# Remember to reset `_JUP_OFFLINE = False` at the start of each trading cycle.
_JUP_OFFLINE = False

async def _safe_get_jupiter_quote(
    *,
    input_mint: str,
    output_mint: str,
    amount: int,
    user_pubkey: str,
    session: "aiohttp.ClientSession",
    slippage_bps: int,
) -> tuple[dict | None, str | None]:
    """
    Wrap market_data.get_jupiter_quote. If DNS/network fails once in a cycle,
    mark _JUP_OFFLINE and stop calling Jupiter again until next cycle.
    Returns (quote_dict_or_None, error_or_None).
    """
    global _JUP_OFFLINE
    if _JUP_OFFLINE:
        return None, "JUPITER_OFFLINE"

    try:
        # CALL THE REAL FUNCTION (do not recurse)
        quote, err = await get_jupiter_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=amount,
            user_pubkey=user_pubkey,
            session=session,
            slippage_bps=slippage_bps,
        )
        return quote, err
    except (ClientConnectorError, ClientOSError, ServerDisconnectedError, asyncio.TimeoutError) as e:
        _JUP_OFFLINE = True
        logger.warning("Jupiter looks offline this cycle (%s); skipping further probes.", e)
        return None, "JUPITER_OFFLINE"
    except Exception as e:
        # Other errors shouldn't trip the offline breaker
        return None, str(e)


async def _is_jupiter_tradable(
    input_mint: str,
    output_mint: str,
    lamports: int,
    user_pubkey: str,
    session: "aiohttp.ClientSession",
    slippage_bps: int,
) -> tuple[bool, str | None]:
    """
    Quick probe: ask Jupiter for a tiny quote via the safe wrapper.
    Returns (is_tradable, error_text_if_any).
    """
    try:
        quote, error = await _safe_get_jupiter_quote(
            input_mint=input_mint,
            output_mint=output_mint,
            amount=lamports,
            user_pubkey=user_pubkey,
            session=session,
            slippage_bps=slippage_bps,
        )
        if not quote or error:
            return False, (error or "No quote")
        return True, None
    except Exception as e:
        return False, str(e)

# ---- Correct program IDs (previous short caused ValueError: wrong size) ----
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
# --------------------------------------------------------------------------------------

def log_error_with_stacktrace(message: str, error: Exception) -> None:
    logger.error(f"{message}: {str(error)}\n{traceback.format_exc()}")

def _clamp_bps(bps: int, lo: int = 1, hi: int = 5000) -> int:
    """Clamp slippage basis points to a sane range to avoid accidental extremes."""
    try:
        return max(lo, min(hi, int(bps)))
    except Exception:
        return 50

# =============================================================================
# Mid-cap focus & stable-like skip (NEW)
# =============================================================================

# Symbols we never want to trade (stablecoins, wrappers, staked-SOL wrappers, etc.)
CORE_STABLE_SYMS = {
    "usdc", "usdt", "wusdc", "wusdt", "usdc.e", "usdt.e", "usdcet", "usdtet",
    "wsol", "jupsol"
}
# Canonical mints to skip outright
CORE_STABLE_ADDRS = {
    # WSOL
    "So11111111111111111111111111111111111111112",
    # USDC
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    # USDT
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    # Jupiter Staked SOL (as reported)
    "jupSoLaHXQiZZTSfEWMTRRgpnyFm8f6sZdosWBjx93v",
}

def _is_core_stable_like(token: Dict[str, Any]) -> bool:
    sym = (token.get("symbol") or token.get("baseToken", {}).get("symbol") or "").strip().lower()
    addr = (token.get("address") or token.get("token_address") or "").strip()
    name = (token.get("name") or "").strip().lower()
    # Treat anything that looks like *staked sol* as a wrapper we do not trade
    if "staked sol" in name or "stsol" in sym:
        return True
    return (sym in CORE_STABLE_SYMS) or (addr in CORE_STABLE_ADDRS)


# Trusted SOL-like mints (canonical WSOL + any other explicitly trusted SOL-like mints)
TRUSTED_SOL_MINTS: Set[str] = {
    "So11111111111111111111111111111111111111112",  # canonical WSOL
    # add other explicitly trusted SOL-like mint addresses here if you truly trust them
}

def _is_sol_like_symbol(token: Dict[str, Any]) -> bool:
    """
    True if the token's symbol/name looks like SOL/WSOL/Solana (untrusted).
    Use mint whitelist (TRUSTED_SOL_MINTS) to determine if it's allowed.
    """
    try:
        sym = (token.get("symbol") or token.get("baseSymbol") or "").strip().lower()
        name = (token.get("name") or token.get("baseName") or "").strip().lower()
        sol_like = {"sol", "solana", "wsol", "wrapped sol", "wrapped solana", "wrapped-sol", "w sol"}
        # check exact symbol matches or name contains suspicious substring
        if sym in sol_like:
            return True
        for s in sol_like:
            if s in name:
                return True
    except Exception:
        pass
    return False

# Tunable targets (env) for mid-cap overweight:
MID_CAP_TARGET  = int(os.getenv("MID_CAP_TARGET", "12"))  # how many mids to aim for
LOW_CAP_SPILLOVER  = int(os.getenv("LOW_CAP_SPILLOVER", "3"))
HIGH_CAP_SPILLOVER = int(os.getenv("HIGH_CAP_SPILLOVER", "2"))

# -------------------------- External data fetchers --------------------------

async def fetch_dexscreener_search(
    session: aiohttp.ClientSession,
    query: str = "solana",
    pages: int = 10,
) -> List[Dict]:
    logger.info(f"Fetching Solana pairs from Dexscreener with query: {query}")
    import os, random

    # --- Dexscreener rate-limit controls ---
    _DS_MAX_CONC = int(os.getenv("DEXSCREENER_MAX_CONC", "2"))  # tiny; DS is strict
    _DS_TIMEOUT  = aiohttp.ClientTimeout(total=12)
    _DS_HEADERS  = {"accept": "application/json", "User-Agent": "SOLOTradingBot/1.0"}
    _DS_SEM      = asyncio.Semaphore(_DS_MAX_CONC)

    class _DexRateLimit(Exception):
        def __init__(self, retry_after: float | None = None):
            super().__init__("Dexscreener rate limit")
            self.retry_after = retry_after

    async def fetch_page_once(page: int) -> List[Dict]:
        # NOTE: uses outer-scope `session` and `query`
        url = f"https://api.dexscreener.com/latest/dex/search/?q={query}&page={page}"
        try:
            async with _DS_SEM:
                async with session.get(url, headers=_DS_HEADERS, timeout=_DS_TIMEOUT) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        pairs = data.get("pairs", []) or []
                        return [{
                            "address": pair.get("baseToken", {}).get("address"),
                            "symbol":  pair.get("baseToken", {}).get("symbol"),
                            "name":    pair.get("baseToken", {}).get("name"),
                            "v24hUSD": float((pair.get("volume", {}) or {}).get("h24", 0) or 0),
                            "liquidity": float((pair.get("liquidity", {}) or {}).get("usd", 0) or 0),
                            "mc": float(pair.get("fdv", 0) or 0),
                            "pairCreatedAt": pair.get("pairCreatedAt", 0),
                            "links": (pair.get("info", {}) or {}).get("socials", []),
                        } for pair in pairs
                            if pair.get("chainId") == "solana"
                            and (pair.get("baseToken", {}) or {}).get("address")]
                    if resp.status == 429:
                        ra = resp.headers.get("Retry-After")
                        try:
                            retry_after = float(ra) if ra is not None else None
                        except Exception:
                            retry_after = None
                        raise _DexRateLimit(retry_after)
                    text = (await resp.text())[:200]
                    logger.warning(f"Dexscreener non-200 {resp.status} on page {page}: {text}")
                    return []
        except _DexRateLimit:
            raise
        except Exception as e:
            log_error_with_stacktrace(f"Dexscreener fetch failed on page {page}", e)
            return []

    async def fetch_page(page: int) -> List[Dict]:
        backoff, max_backoff = 1.0, 30.0
        attempts = 0
        while attempts < 5:
            attempts += 1
            try:
                return await fetch_page_once(page)
            except _DexRateLimit as rl:
                sleep_s = rl.retry_after if rl.retry_after is not None else backoff
                logger.warning(f"Dexscreener rate limit hit on page {page}, sleeping {sleep_s:.1f}s")
                await asyncio.sleep(sleep_s + random.uniform(0, 0.5))
                backoff = min(backoff * 2.0, max_backoff)
                continue
            except Exception as e:
                logger.warning(f"Dexscreener fetch exception on page {page} (attempt {attempts}): {e}")
                await asyncio.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 2.0, max_backoff)
        logger.error(f"Dexscreener fetch failed on page {page} after {attempts} attempts")
        return []

    # Limit total pages to avoid bursts
    page_limit = int(os.getenv("DEX_PAGE_LIMIT", "6"))
    pages = max(1, min(pages, page_limit))
    page_indices = list(range(1, pages + 1))

    # Fetch with per-host concurrency controlled by _DS_SEM
    results = await asyncio.gather(*(fetch_page(p) for p in page_indices), return_exceptions=True)

    # Flatten + dedupe by token address
    out: List[Dict] = []
    seen: set[str] = set()
    for r in results:
        if isinstance(r, list):
            for t in r:
                addr = t.get("address")
                if addr and addr not in seen:
                    seen.add(addr)
                    out.append(t)
        else:
            logger.warning(f"Dexscreener page task failed: {r}")

    return out    

async def fetch_raydium_tokens(
    session: aiohttp.ClientSession,
    solana_client: Any,
    max_pairs: int = 100
) -> List[Dict]:
    def _truthy_env(name: str, default: str = "0") -> bool:
        v = os.getenv(name, default).strip().lower()
        return v in ("1", "true", "yes", "on", "y")

    # master gates
    if _truthy_env("FORCE_DISABLE_RAYDIUM", "0"):
        logger.info("Raydium disabled by FORCE_DISABLE_RAYDIUM - skipping fetch.")
        return []
    if not _truthy_env("RAYDIUM_ENABLE", "0"):
        logger.info("Raydium disabled by RAYDIUM_ENABLE=0 - skipping fetch.")
        return []
    if max_pairs <= 0:
        logger.info("Raydium disabled by config (max_pairs=0) - skipping fetch.")
        return []

    tokens: List[Dict] = []
    url = "https://api.raydium.io/v2/main/pairs"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "TradingBot/1.0"
    }

    try:
        max_mb = float(os.getenv("RAYDIUM_MAX_DOWNLOAD_MB", "40"))
    except Exception:
        max_mb = 40.0
    max_bytes = int(max_mb * 1024 * 1024)

    try:
        logger.info("Fetching tokens from Raydium")
        timeout = aiohttp.ClientTimeout(total=20)
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status == 429:
                logger.warning("Raydium rate limit (429); skipping this cycle.")
                return []
            if resp.status != 200:
                body = (await resp.text())[:512]
                logger.error(f"Raydium API failed: HTTP {resp.status}, Response: {body}")
                return []

            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit() and int(cl) > max_bytes:
                logger.warning("Raydium response exceeded safe size (Content-Length=%s > %s bytes). Skipping.", cl, max_bytes)
                return []

            buf = bytearray()
            async for chunk in resp.content.iter_chunked(64 * 1024):
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    logger.warning("Raydium streamed body exceeded safe size (%s > %s bytes). Skipping.", len(buf), max_bytes)
                    return []

            try:
                data = json.loads(bytes(buf))
            except MemoryError:
                logger.error("Raydium parse MemoryError on %s bytes.", len(buf))
                return []
            except Exception as e:
                logger.error("Raydium JSON parse error: %s", e)
                return []

            pairs = data if isinstance(data, list) else data.get("data", [])
            if not isinstance(pairs, list):
                logger.error(f"Raydium API returned unexpected structure: {type(data)}")
                return []

            logger.info("Fetched %d pairs from Raydium", len(pairs))
            for pair in pairs[:max_pairs]:
                if not isinstance(pair, dict):
                    continue

                token_address = pair.get("baseMint")
                name_val = pair.get("name", "") or "UNKNOWN"
                symbol = name_val.split("/")[0] if "/" in name_val else name_val
                if not token_address or not symbol:
                    continue

                try:
                    Pubkey.from_string(token_address)
                except Exception:
                    logger.warning("Invalid Raydium token address: %s", token_address)
                    continue

                # tolerant volume/liquidity
                v24 = pair.get("volume24h")
                if isinstance(v24, dict):
                    v24 = v24.get("usd", 0)
                v24 = float(v24 or 0)

                liq = pair.get("liquidity")
                if isinstance(liq, dict):
                    liq = liq.get("usd", 0)
                liq = float(liq or 0)

                market_cap = float(pair.get("fdv", 0) or pair.get("marketCap", 0) or 0)

                try:
                    cached_token = await get_cached_token_data(token_address)
                    if cached_token and cached_token.get("v24hUSD", 0) >= v24:
                        continue
                except Exception:
                    pass

                try:
                    pair_created_at = await get_token_creation_time(token_address, solana_client, config=None, session=session)
                except Exception:
                    pair_created_at = None

                token = {
                    "address": token_address,
                    "symbol": symbol,
                    "name": name_val,
                    "v24hUSD": v24,
                    "liquidity": liq,
                    "mc": market_cap,
                    "pairCreatedAt": int(pair_created_at.timestamp() * 1000) if pair_created_at else 0,
                    "links": (pair.get("extensions") or {}).get("socials", []),
                    # NEW
                    "creation_timestamp": _derive_creation_ts_s({
                        "pairCreatedAt": int(pair_created_at.timestamp() * 1000) if pair_created_at else 0
                    }),
                }
                try:
                    await cache_token_data(token)
                except Exception:
                    pass
                tokens.append(token)

            logger.info("Processed %d valid Raydium tokens", len(tokens))
            return tokens

    except Exception as e:
        log_error_with_stacktrace("Raydium API fetch failed", e)
        return []
    
# ---------------------------------------------------------------------
# NEW: Birdeye base URL selection (respect env override + pro/public modes)
# ---------------------------------------------------------------------
BIRDEYE_MODE = os.getenv("BIRDEYE_MODE", "public").strip().lower()
_user_birdeye_base = (os.getenv("BIRDEYE_BASE_URL") or "").strip()
if _user_birdeye_base:
    BIRDEYE_BASE_URL = _user_birdeye_base.rstrip("/")
else:
    if BIRDEYE_MODE in ("pro", "v2"):
        BIRDEYE_BASE_URL = "https://api.birdeye.so"
    else:
        BIRDEYE_BASE_URL = "https://public-api.birdeye.so"

# Birdeye 401-cycle breaker (already present in file) 

async def fetch_birdeye_tokens(
    session: aiohttp.ClientSession,
    solana_client: Any,
    max_tokens: int = 50,
    config: Dict[str, Any] | None = None,
) -> List[Dict]:
    """
    Birdeye fetch with hard OFF support and clean 401 handling.

    Updated endpoints:
      - Prefer the public/pro v3 token list endpoint:
          {BIRDEYE_BASE_URL}/defi/v3/token/list?chain=solana&limit=...&offset=...
      - Use X-API-KEY header (when present).
      - Compatible with both public and pro hostnames; tolerant to payload shapes:
        supports top-level list, {data: {items: [...]}} and {data: [...] } variants.
    NO-OP when:
      - _birdeye_allowed(config) is False, or
      - missing BIRDEYE_API_KEY, or
      - max_tokens <= 0
    """
    global _BIRDEYE_401_SEEN  # <-- keep the per-cycle breaker on 401

    if not _birdeye_allowed(config):
        logger.info("Birdeye disabled by config/env - skipping tokenlist fetch.")
        return []

    api_key = (os.getenv("BIRDEYE_API_KEY") or "").strip()
    if not api_key:
        logger.info("Birdeye disabled - missing BIRDEYE_API_KEY.")
        return []
    if max_tokens <= 0:
        logger.info("Birdeye disabled by config (max_tokens=0) - skipping fetch.")
        return []

    logger.info("Fetching tokens from Birdeye (base=%s)", BIRDEYE_BASE_URL)
    try:
        base_list_url = f"{BIRDEYE_BASE_URL.rstrip('/')}/defi/v3/token/list"
        headers = {
            "X-API-KEY": api_key,
            "x-chain": "solana",
            "Accept": "application/json",
            "User-Agent": "TradingBot/1.0",
        }
        timeout = aiohttp.ClientTimeout(total=15)

        params = {"chain": "solana", "limit": int(max_tokens), "offset": 0}

        async with session.get(base_list_url, headers=headers, params=params, timeout=timeout) as resp:
            if resp.status == 429:
                logger.warning("Birdeye tokenlist rate limit (429); skipping this cycle.")
                return []
            if resp.status == 401:
                _BIRDEYE_401_SEEN = True  # flip breaker so the rest of the cycle avoids Birdeye
                body = await resp.text()
                logger.error("Birdeye unauthorized (401). Check key/plan. Resp[:256]=%s", body[:256])
                return []
            if resp.status != 200:
                body = await resp.text()
                logger.error("Birdeye API error: HTTP %s, Response: %s", resp.status, (body or "")[:512])
                return []

            try:
                data = await resp.json(content_type=None)
            except Exception as e:
                try:
                    txt = await resp.text()
                    data = json.loads(txt)
                except Exception:
                    logger.error("Birdeye JSON parse error: %s", e)
                    return []

            # Accept multiple shapes: {data: {items: [...]}} or {data: [...] } or top-level list
            payload = None
            if isinstance(data, dict):
                payload = data.get("data") or data.get("items") or data
            else:
                payload = data

            # Normalize list of token dicts
            token_list: List[Dict] = []
            if isinstance(payload, dict):
                # try common keys 'items' or 'tokens'
                if isinstance(payload.get("items"), list):
                    token_list = payload.get("items", [])
                elif isinstance(payload.get("tokens"), list):
                    token_list = payload.get("tokens", [])
                else:
                    # try to find a list-valued key
                    for v in payload.values():
                        if isinstance(v, list):
                            token_list = v
                            break
            elif isinstance(payload, list):
                token_list = payload
            else:
                logger.error("Birdeye tokenlist payload unexpected: %s", type(payload))
                return []

            valid_tokens: List[Dict] = []
            for token in token_list:
                if not isinstance(token, dict):
                    continue
                token_address = token.get("address") or token.get("tokenAddress") or token.get("token_address")
                if not token_address:
                    continue
                if token_address in WHITELISTED_TOKENS:
                    continue
                try:
                    Pubkey.from_string(token_address)
                except Exception:
                    logger.warning("Invalid Birdeye token address: %s", token_address)
                    continue

                cached_token = await get_cached_token_data(token_address)
                try:
                    v24h = float(token.get("v24hUSD") or token.get("volume_24h_usd") or token.get("v24h") or token.get("v24hUSD", 0) or 0)
                except Exception:
                    v24h = 0.0
                if cached_token and cached_token.get("v24hUSD", 0) >= v24h:
                    continue

                # IMPORTANT: pass config through so the same gate applies to creation time
                pair_created_at = await get_token_creation_time(token_address, solana_client, config=config, session=session)

                valid_token = {
                    "address": token_address,
                    "symbol": token.get("symbol") or token.get("tokenSymbol") or "UNKNOWN",
                    "name": token.get("name") or token.get("tokenName") or "UNKNOWN",
                    "v24hUSD": float(v24h or 0.0),
                    "liquidity": float(token.get("liquidity") or token.get("liquidity_usd") or 0.0),
                    "mc": float(token.get("mc") or token.get("market_cap") or token.get("fdv") or 0.0),
                    "pairCreatedAt": int(pair_created_at.timestamp() * 1000) if pair_created_at else 0,
                    "links": (token.get("extensions") or {}).get("socials", []),
                    "created_at": token.get("created_at"),
                    # canonical seconds field
                    "creation_timestamp": _derive_creation_ts_s({
                        "pairCreatedAt": int(pair_created_at.timestamp() * 1000) if pair_created_at else 0,
                        "created_at": token.get("created_at"),
                    }),
                }
                try:
                    await cache_token_data(valid_token)
                except Exception:
                    pass
                valid_tokens.append(valid_token)

            logger.info("Processed %d valid Birdeye tokens", len(valid_tokens))
            return valid_tokens

    except Exception as e:
        log_error_with_stacktrace("Birdeye token fetch failed", e)
        return []
# -------------------------- Creation time helpers --------------------------

def _truthy_env(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y")

async def fetch_dexscreener_creation_time(
    token_address: str,
    session: aiohttp.ClientSession,
) -> datetime | None:
    """
    Fetch creation time from Dexscreener token endpoint (/token/v1/{token}/solana).
    Handles numeric timestamps (seconds or ms) and ISO8601 strings (e.g. "2025-11-12T09:41:23.000Z").
    Uses an in-process cache and a concurrency semaphore to avoid hammering Dexscreener.
    Returns a timezone-aware UTC datetime or None.
    """
    try:
        import time, random
        from datetime import datetime, timezone
    except Exception:
        pass

    if not token_address:
        return None

    # Don't query for whitelisted tokens
    if token_address in WHITELISTED_TOKENS:
        return None

    # 1) DB cache first
    try:
        cached_time = await get_cached_creation_time(token_address)
        if cached_time:
            return cached_time
    except Exception:
        logger.debug("get_cached_creation_time failed for %s", token_address, exc_info=True)

    # Ensure module-level cache/semaphore exist with safe defaults
    global _DS_TOKEN_CACHE, _DS_CACHE_TTL, _DS_SEM
    if "_DS_TOKEN_CACHE" not in globals() or not isinstance(globals().get("_DS_TOKEN_CACHE"), dict):
        _DS_TOKEN_CACHE = {}
    if "_DS_CACHE_TTL" not in globals():
        _DS_CACHE_TTL = int(os.getenv("DS_TOKEN_CACHE_TTL", "600"))
    if "_DS_SEM" not in globals():
        _DS_SEM = asyncio.Semaphore(int(os.getenv("DS_TOKEN_CONCURRENCY", "6")))

    # 2) in-process cache
    now = time.time()
    cached = _DS_TOKEN_CACHE.get(token_address)
    if cached and (now - cached[0]) < int(_DS_CACHE_TTL):
        data = cached[1]
    else:
        url = f"https://api.dexscreener.com/token/v1/{token_address}/solana"
        headers = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": os.getenv("DEX_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"),
            "Origin": "https://dexscreener.com",
            "Referer": "https://dexscreener.com/",
        }
        timeout = aiohttp.ClientTimeout(total=12, sock_connect=6, sock_read=8)

        backoff = 1.5
        max_backoff = 30.0
        attempts = 0
        data = None

        while attempts < 6:
            attempts += 1
            try:
                async with _DS_SEM:
                    async with session.get(url, headers=headers, timeout=timeout) as resp:
                        st = resp.status
                        if st == 429:
                            ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                            try:
                                ra_val = float(ra) if ra is not None else None
                            except Exception:
                                ra_val = None
                            sleep_s = (ra_val if ra_val and ra_val > 0 else backoff) + random.uniform(0.0, 0.8)
                            logger.warning("Dexscreener 429 for %s; sleeping %.1fs (attempt %d)", token_address, sleep_s, attempts)
                            await asyncio.sleep(min(sleep_s, max_backoff))
                            backoff = min(backoff * 1.8, max_backoff)
                            continue
                        if st == 404:
                            logger.debug("Dexscreener returned 404 for token %s", token_address)
                            return None
                        if st != 200:
                            txt = (await resp.text())[:500]
                            logger.warning("Dexscreener token HTTP %s for %s: %s", st, token_address, txt)
                            return None

                        try:
                            data = await resp.json(content_type=None)
                        except Exception:
                            try:
                                txt = await resp.text()
                                data = json.loads(txt)
                            except Exception as je:
                                logger.debug("Dexscreener JSON decode failed for %s: %s", token_address, je, exc_info=True)
                                return None

                        try:
                            _DS_TOKEN_CACHE[token_address] = (time.time(), data)
                        except Exception:
                            pass

                        break
            except asyncio.TimeoutError:
                logger.warning("Dexscreener token request timed out for %s (attempt %d)", token_address, attempts)
                backoff = min(backoff * 1.8, max_backoff)
                await asyncio.sleep(backoff + random.uniform(0, 0.5))
                continue
            except aiohttp.ClientError as e:
                logger.debug("Dexscreener client error for %s: %s (attempt %d)", token_address, e, attempts, exc_info=True)
                backoff = min(backoff * 1.8, max_backoff)
                await asyncio.sleep(backoff + random.uniform(0, 0.5))
                continue
            except Exception as e:
                logger.debug("Unexpected error fetching Dexscreener token %s: %s", token_address, e, exc_info=True)
                return None

        if not data:
            return None

    # Parse pairs and accept numeric and ISO timestamps
    try:
        pairs = []
        if isinstance(data, dict):
            if isinstance(data.get("pairs"), list):
                pairs = data.get("pairs")
            elif isinstance(data.get("data"), dict) and isinstance(data["data"].get("pairs"), list):
                pairs = data["data"]["pairs"]
            elif isinstance(data.get("data"), list):
                pairs = data["data"]
            else:
                for k in ("pairs", "data", "results"):
                    if isinstance(data.get(k), list):
                        pairs = data.get(k)
                        break
        elif isinstance(data, list):
            pairs = data

        if not pairs:
            return None

        earliest_ms = None
        now_ms = time.time() * 1000.0
        sol_launch_ms = 1581465600000.0  # Feb 12 2020

        for p in pairs:
            if not isinstance(p, dict):
                continue

            # chain tolerance
            chain_val = p.get("chainId") or p.get("chain") or ""
            try:
                chain_check = str(chain_val).strip().lower()
            except Exception:
                chain_check = ""
            if "sol" not in chain_check and "solana" not in chain_check:
                continue

            # look for several created keys and parse numeric or ISO strings
            created_candidate = None
            for key in ("pairCreatedAt", "createdAt", "createdTime", "created_timestamp", "created_at", "mintTime"):
                if key in p and p.get(key) is not None:
                    created_candidate = p.get(key)
                    break
                sub = p.get("data") or p.get("score") or {}
                if isinstance(sub, dict) and key in sub and sub.get(key) is not None:
                    created_candidate = sub.get(key)
                    break

            if created_candidate is None:
                continue

            # Try numeric first (seconds or ms)
            created_ms = None
            try:
                # numeric-like (int/float) or numeric string
                if isinstance(created_candidate, (int, float)):
                    created_ms = float(created_candidate)
                else:
                    s = str(created_candidate).strip()
                    # If it's a pure-digit string, parse as number
                    if re.fullmatch(r"\d+", s):
                        created_ms = float(s)
                    else:
                        # Try to parse as float (covers "1234567890.0")
                        try:
                            created_ms = float(s)
                        except Exception:
                            created_ms = None
            except Exception:
                created_ms = None

            # Try numeric first (seconds or ms) and accept ISO8601 strings (e.g. "2025-11-12T09:41:23.000Z").
            # Requires: `import re` and `from datetime import datetime, timezone` at top of file.
            created_ms = None
            try:
                if created_candidate is None:
                    created_ms = None
                elif isinstance(created_candidate, (int, float)):
                    # numeric value (could be seconds or ms)
                    created_ms = float(created_candidate)
                else:
                    s = str(created_candidate).strip()
                    if not s:
                        created_ms = None
                    else:
                        # Pure numeric string: integer or float (e.g. "1630000000" or "1630000000.0")
                        if re.fullmatch(r"\d+(\.\d+)?", s):
                            created_ms = float(s)
                        else:
                            # Try ISO8601 parse (handles trailing "Z" by converting to +00:00)
                            try:
                                iso = s
                                if iso.endswith("Z"):
                                    iso = iso.replace("Z", "+00:00")
                                dt = datetime.fromisoformat(iso)
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                created_ms = dt.timestamp() * 1000.0
                            except Exception:
                                # Final fallback: try float conversion for odd numeric-like strings
                                try:
                                    created_ms = float(s)
                                except Exception:
                                    created_ms = None
            except Exception:
                created_ms = None

            # Normalize seconds→ms if it looks like seconds
            try:
                if created_ms is not None and created_ms < 10_000_000_000:
                    created_ms = created_ms * 1000.0
            except Exception:
                created_ms = None

            # If still None, skip this pair
            if created_ms is None:
                continue

            # sanity check and accept (sol_launch_ms and now_ms defined earlier)
            try:
                if created_ms < sol_launch_ms or created_ms > now_ms:
                    continue
            except Exception:
                continue

            if earliest_ms is None or created_ms < earliest_ms:
                earliest_ms = created_ms

            
        dt = datetime.fromtimestamp(earliest_ms / 1000.0, tz=timezone.utc)
        try:
            await cache_creation_time(token_address, dt)
        except Exception:
            logger.debug("Failed to cache Dexscreener creation time for %s", token_address, exc_info=True)

        logger.debug("Dexscreener creation time for %s -> %s", token_address, dt.isoformat())
        return dt

    except Exception as e:
        logger.debug("Error parsing Dexscreener response for %s: %s", token_address, e, exc_info=True)
        return None


async def fetch_birdeye_creation_time(
    token_address: str,
    session: aiohttp.ClientSession,
) -> datetime | None:
    """
    Fetch creation time from Birdeye using the public-api.birdeye.so endpoints.

    Behavior / contract (per Birdeye guidance):
      - Base URL: https://public-api.birdeye.so  (use BIRDEYE_BASE_URL if present)
      - Token list endpoint:  /defi/v3/token/list
      - Required headers: X-API-KEY, x-chain (other headers are optional; we send Accept for JSON)
      - Rate limits (Starter): 15 RPS, 3M CUs/month — callers should respect these limits externally.
      - Returns a timezone-aware UTC datetime on success, or None if not available.

    Notes:
      - Respects feature gates and the module-level _BIRDEYE_401_SEEN breaker (sets it on HTTP 401).
      - Uses get_cached_creation_time / cache_creation_time when available to avoid unnecessary calls.
      - Accepts and normalizes a variety of timestamp shapes (seconds, ms, ISO8601).
    """
    global _BIRDEYE_401_SEEN

    if not token_address:
        return None

    try:
        if token_address in WHITELISTED_TOKENS:
            logger.debug("Birdeye creation-time: token is whitelisted, skipping: %s", token_address)
            return None
    except Exception:
        pass

    # Feature gating & key presence
    try:
        if _truthy_env("FORCE_DISABLE_BIRDEYE", "0"):
            logger.debug("Birdeye creation-time disabled by FORCE_DISABLE_BIRDEYE")
            return None
        if not _truthy_env("BIRDEYE_ENABLE", "0"):
            logger.debug("Birdeye creation-time disabled by BIRDEYE_ENABLE=0")
            return None
        api_key = (os.getenv("BIRDEYE_API_KEY") or "").strip()
        if not api_key:
            logger.debug("Birdeye creation-time disabled: missing BIRDEYE_API_KEY")
            return None
        if "_BIRDEYE_401_SEEN" in globals() and _BIRDEYE_401_SEEN:
            logger.debug("Birdeye creation-time skipped: 401 seen earlier this cycle")
            return None
    except Exception:
        return None

    # DB/in-process cache first (tolerant)
    try:
        cached = await get_cached_creation_time(token_address)
        if cached:
            try:
                if isinstance(cached, datetime):
                    return cached
                if isinstance(cached, (int, float)):
                    v = float(cached)
                    # ms vs s
                    if v > 10_000_000_000:
                        return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
                    return datetime.fromtimestamp(v, tz=timezone.utc)
                if isinstance(cached, str):
                    s = cached.strip()
                    if re.fullmatch(r"\d+(\.\d+)?", s):
                        v = float(s)
                        if v > 10_000_000_000:
                            return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
                        return datetime.fromtimestamp(v, tz=timezone.utc)
                    try:
                        iso = s
                        if iso.endswith("Z"):
                            iso = iso.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(iso)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt
                    except Exception:
                        pass
            except Exception:
                logger.debug("Cached creation-time in unknown format for %s: %s", token_address, repr(cached))
    except Exception:
        logger.debug("get_cached_creation_time failed for %s", token_address, exc_info=True)

    # Resolve base URL (prefer configured global BIRDEYE_BASE_URL if present)
    base = globals().get("BIRDEYE_BASE_URL") or "https://public-api.birdeye.so"
    endpoint = "/defi/v3/token/list"
    url = f"{base.rstrip('/')}{endpoint}"

    # Chain header (default "solana")
    chain = (os.getenv("BIRDEYE_CHAIN") or "solana").strip()

    headers = {
        "X-API-KEY": api_key,
        "x-chain": chain,
        "Accept": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=10)

    params = {"address": token_address}
    payload = {"address": token_address}

    try:
        # Try GET first
        async with session.get(url, headers=headers, params=params, timeout=timeout) as resp:
            st = resp.status
            if st == 429:
                logger.warning("Birdeye creation time rate limit (429) for %s - skipping this cycle.", token_address)
                return None
            if st == 401:
                _BIRDEYE_401_SEEN = True
                body = await resp.text()
                logger.error("Birdeye creation time unauthorized (401). Check key/plan. Resp[:256]=%s", body[:256])
                return None
            if st in (400, 404):
                logger.debug("Birdeye creation time returned %s for %s", st, token_address)
                return None
            if st == 405:
                get_failed = True
            elif st != 200:
                body = await resp.text()
                logger.warning("Birdeye creation time failed for %s: HTTP %s, Resp[:256]=%s", token_address, st, (body or "")[:256])
                return None
            else:
                get_failed = False
                try:
                    data = await resp.json(content_type=None)
                except Exception:
                    try:
                        txt = await resp.text()
                        data = json.loads(txt)
                    except Exception as e:
                        logger.debug("Birdeye creation time JSON parse failed for %s: %s", token_address, e, exc_info=True)
                        return None

        # If GET method not allowed, try POST
        if 'get_failed' in locals() and get_failed:
            try:
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                    st = resp.status
                    if st == 429:
                        logger.warning("Birdeye creation time rate limit (429) for %s - skipping this cycle.", token_address)
                        return None
                    if st == 401:
                        _BIRDEYE_401_SEEN = True
                        body = await resp.text()
                        logger.error("Birdeye creation time unauthorized (401). Check key/plan. Resp[:256]=%s", body[:256])
                        return None
                    if st in (400, 404):
                        logger.debug("Birdeye creation time returned %s for %s", st, token_address)
                        return None
                    if st != 200:
                        body = await resp.text()
                        logger.warning("Birdeye creation time failed (POST) for %s: HTTP %s, Resp[:256]=%s", token_address, st, (body or "")[:256])
                        return None
                    try:
                        data = await resp.json(content_type=None)
                    except Exception:
                        try:
                            txt = await resp.text()
                            data = json.loads(txt)
                        except Exception as e:
                            logger.debug("Birdeye creation time JSON parse failed (POST) for %s: %s", token_address, e, exc_info=True)
                            return None
            except aiohttp.ClientError as e:
                logger.debug("Birdeye POST client error for %s: %s", token_address, e, exc_info=True)
                return None
            except asyncio.TimeoutError:
                logger.debug("Birdeye creation time POST request timed out for %s", token_address)
                return None

        # Normalize response payload into a list of candidate token objects
        candidates: list = []
        try:
            if isinstance(data, dict):
                # Common shapes: {data: [ ... ]} or {data: { items: [...] }} or top-level single token
                if isinstance(data.get("data"), list):
                    candidates = data.get("data", [])
                elif isinstance(data.get("data"), dict):
                    for k in ("items", "tokens", "list"):
                        if isinstance(data["data"].get(k), list):
                            candidates = data["data"].get(k)
                            break
                    else:
                        # treat data as a single token object
                        candidates = [data["data"]]
                else:
                    # try other list-valued keys
                    for k in ("result", "tokens", "items", "list"):
                        if isinstance(data.get(k), list):
                            candidates = data.get(k)
                            break
                    else:
                        # fall back to treating top-level dict as single candidate
                        candidates = [data]
            elif isinstance(data, list):
                candidates = data
            else:
                candidates = []
        except Exception:
            candidates = []

        if not candidates:
            return None

        # Search candidates for a creation timestamp
        created_ts = None
        for item in candidates:
            if not isinstance(item, dict):
                continue
            # direct keys
            for k in ("created_timestamp", "createdAt", "created_at", "mintTime", "created"):
                if k in item and item.get(k) not in (None, ""):
                    created_ts = item.get(k)
                    break
            if created_ts is not None:
                break
            # nested structures
            for nested in ("meta", "attributes", "data"):
                sub = item.get(nested)
                if isinstance(sub, dict):
                    for k in ("created_timestamp", "createdAt", "created_at", "mintTime", "created"):
                        if k in sub and sub.get(k) not in (None, ""):
                            created_ts = sub.get(k)
                            break
                    if created_ts is not None:
                        break
            if created_ts is not None:
                break

        if created_ts is None:
            return None

        # Normalize to milliseconds (float)
        ms_val = None
        try:
            if isinstance(created_ts, (int, float)):
                ms_val = float(created_ts)
            else:
                s = str(created_ts).strip()
                if re.fullmatch(r"\d+(\.\d+)?", s):
                    ms_val = float(s)
                else:
                    iso = s
                    if iso.endswith("Z"):
                        iso = iso.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(iso)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ms_val = dt.timestamp() * 1000.0
        except Exception:
            ms_val = None

        if ms_val is None:
            return None

        # convert seconds->ms if needed
        if ms_val < 10_000_000_000:
            ms_val *= 1000.0

        now_ms = time.time() * 1000.0
        sol_launch_ms = 1581465600000.0  # Feb 12 2020 sanity lower bound
        if not (sol_launch_ms <= ms_val <= now_ms):
            logger.debug("Birdeye creation time out of range for %s: %s", token_address, ms_val)
            return None

        dt = datetime.fromtimestamp(ms_val / 1000.0, tz=timezone.utc)

        # Cache the parsed datetime if helper available
        try:
            await cache_creation_time(token_address, dt)
        except Exception:
            try:
                await cache_creation_time(token_address, int(dt.timestamp()))
            except Exception:
                logger.debug("Failed to cache Birdeye creation time for %s", token_address, exc_info=True)

        logger.debug("Birdeye creation time for %s -> %s", token_address, dt.isoformat())
        return dt

    except Exception as e:
        logger.debug("Birdeye creation time request/parsing failed for %s: %s", token_address, e, exc_info=True)
        return None

async def get_token_creation_time(
    token_address: str,
    solana_client: Any,
    config: dict[str, Any] | None = None,
    session: aiohttp.ClientSession | None = None,
) -> datetime | None:
    """
    Chooses provider order based on Birdeye availability:
      - If _birdeye_allowed(config) and API key present -> try Birdeye first then Dexscreener.
      - Otherwise -> Dexscreener only.

    This function is unchanged in behavior except that it remains compatible with the
    updated fetch_birdeye_creation_time which now uses the new Birdeye endpoints and headers.
    """
    logger.info("Fetching token creation time for %s", token_address)

    # Decide whether Birdeye should be tried (unified gate + has key)
    birdeye_on = _birdeye_allowed(config) and bool((os.getenv("BIRDEYE_API_KEY") or "").strip())

    _owns = False
    if session is None:
        session = aiohttp.ClientSession()
        _owns = True
    try:
        # Keep same ordering logic: try Birdeye first if enabled
        fetchers = [fetch_dexscreener_creation_time]  # fallbacks must be defined in module where this is used
        if birdeye_on:
            fetchers = [fetch_birdeye_creation_time, fetch_dexscreener_creation_time]

        for fetch_func in fetchers:
            try:
                creation_time = await fetch_func(token_address, session)
                if creation_time:
                    return creation_time
            except Exception as e:
                logger.warning(
                    "Failed to fetch creation time for %s using %s: %s",
                    token_address,
                    getattr(fetch_func, "__name__", "wrapped_fetcher"),
                    str(e),
                )
        return None
    finally:
        if _owns:
            await session.close()

# -------------------------- Eligibility / scoring --------------------------

# try to keep a reference to the real implementation if available
_try_verify = None
try:
    # prefer the eligibility module's implementation if import succeeded earlier
    from .eligibility import verify_token_with_rugcheck as _try_verify  # type: ignore
except Exception:
    _try_verify = None

async def verify_token_with_rugcheck(token_address: str, token: Dict, session: aiohttp.ClientSession, config: Dict) -> Tuple[float, List[Dict], str]:
    """
    Delegating wrapper:
      - If the real eligibility.verify_token_with_rugcheck is available, call it.
      - Otherwise fall back to a safe placeholder.
    """
    if callable(_try_verify):
        try:
            # Support both async and sync implementations
            from inspect import iscoroutinefunction
            import asyncio
            if iscoroutinefunction(_try_verify):
                return await _try_verify(token_address, token, session, config)
            else:
                # run sync implementation in a worker thread to avoid blocking
                return await asyncio.to_thread(_try_verify, token_address, token, session, config)
        except Exception:
            logger.exception("eligibility.verify_token_with_rugcheck raised; falling back to placeholder.")

    # Placeholder fallback (existing behaviour)
    logger.warning(
        "RugCheck API not configured; falling back to placeholder result for %s (%s)",
        token.get("symbol", "UNKNOWN"),
        token_address,
    )
    max_rugcheck_score = (config or {}).get("discovery", {}).get("low_cap", {}).get("max_rugcheck_score", 5000)
    if "newly_launched" in (token or {}).get("categories", []):
        max_rugcheck_score = (config or {}).get("discovery", {}).get("newly_launched", {}).get("max_rugcheck_score", 2000)
    return 0.0, [], "Placeholder: RugCheck not available"

def _log_reject(token_address: str, reasons: str) -> None:
    """
    Centralized logging for rejected tokens used by is_token_eligible.

    Keep this small and defensive so logging a rejection never causes the
    eligibility check to raise. 
    """
    try:
        # Prefer module-level logger if available
        if 'logger' in globals() and logger is not None:
            # Use INFO so rejections are visible at normal verbosity; change to debug if noisy.
            logger.info("Token %s rejected: %s", token_address, reasons)
        else:
            # Fallback for environments where logger isn't initialized yet
            print(f"Token {token_address} rejected: {reasons}")
    except Exception:
        # Last-resort swallow so logging never breaks the eligibility check
        try:
            print(f"Token {token_address} rejected: {reasons}")
        except Exception:
            pass

async def is_token_eligible(token: Dict, session: aiohttp.ClientSession, config: Dict) -> Tuple[bool, List[str]]:
    logger.info(f"Checking eligibility for {token.get('symbol', 'UNKNOWN')} ({token.get('address', 'UNKNOWN')})")
    token_address = token.get('address')
    if not token_address:
        logger.warning(f"Token missing address: {token}")
        return False, []

    # >>> Hard skip for stables / wrappers (jupSOL, USDC, USDT, WSOL, etc.)
    if _is_core_stable_like(token):
        return False, []

    categories: List[str] = []
    reasons: List[str] = []

    # Loose numeric parsing helpers
    def _to_float_loose(v, default=0.0):
        try:
            if v is None or v == "":
                return float(default)
            return float(v)
        except Exception:
            try:
                s = str(v).strip().replace(",", "").replace("$", "")
                return float(s) if s else float(default)
            except Exception:
                return float(default)

    market_cap = _to_float_loose(token.get('mc') or token.get('fdv') or token.get('market_cap') or 0.0)
    liquidity = _to_float_loose(token.get('liquidity') or token.get('dex_liquidity') or token.get('liquidity_usd') or 0.0)
    volume_24h = _to_float_loose(token.get('v24hUSD') or token.get('volume_24h') or token.get('volume24h') or 0.0)
    pair_created_at = token.get('pairCreatedAt', 0)

    # Defensive config extraction with safe defaults
    discovery_cfg = (config.get("discovery") or {})
    low_cfg = (discovery_cfg.get("low_cap") or {})
    mid_cfg = (discovery_cfg.get("mid_cap") or {})
    large_cfg = (discovery_cfg.get("large_cap") or {})
    new_cfg = (discovery_cfg.get("newly_launched") or {})

    # Assign category by market cap when present. If market_cap is missing/zero,
    # prefer newly_launched check (age) then fall back to low_cap (permissive).
    try:
        if market_cap and market_cap > 0:
            if market_cap < float(low_cfg.get("max_market_cap", 100000)):
                categories.append('low_cap')
            elif market_cap < float(mid_cfg.get("max_market_cap", 500000)):
                categories.append('mid_cap')
            else:
                categories.append('large_cap')
        else:
            # No market cap -> use creation time to decide if newly launched, else low_cap
            try:
                now_s = time.time()
                if pair_created_at:
                    # pair_created_at may be ms or seconds
                    pc = int(pair_created_at)
                    if pc > 10**12:
                        pc = pc // 1000
                    age_minutes = (now_s - float(pc)) / 60.0 if pc > 0 else None
                    max_age = float(new_cfg.get("max_token_age_minutes", 180))
                    if age_minutes is not None and age_minutes <= max_age:
                        categories.append("newly_launched")
                    else:
                        categories.append("low_cap")
                else:
                    categories.append("low_cap")
            except Exception:
                categories.append("low_cap")
    except Exception:
        categories.append("low_cap")

    # Helper to only enforce positive thresholds
    def _enforce_threshold(metric_value: float, threshold_raw: Any, label: str) -> bool:
        try:
            thr = float(threshold_raw or 0.0)
        except Exception:
            thr = 0.0
        if thr <= 0:
            return True  # no threshold configured -> pass
        if metric_value >= thr:
            return True
        # fails threshold
        reasons.append(f"{label}<{int(thr)}")
        return False

    # Category-specific checks: only enforce the thresholds for the categories the token is in.
    # For tokens in multiple categories (rare), require all applicable checks to pass.
    ok = True
    for cat in categories:
        try:
            if cat == "low_cap":
                # low-cap thresholds (apply only if non-zero in config)
                liq_thr = low_cfg.get("liquidity_threshold", 0)
                vol_thr = low_cfg.get("volume_threshold", 0)
                if not _enforce_threshold(liquidity, liq_thr, "liquidity<low_min"):
                    ok = False
                if not _enforce_threshold(volume_24h, vol_thr, "volume24h<low_min"):
                    ok = False
            elif cat == "mid_cap":
                liq_thr = mid_cfg.get("liquidity_threshold", 0)
                vol_thr = mid_cfg.get("volume_threshold", 0)
                if not _enforce_threshold(liquidity, liq_thr, "liquidity<mid_min"):
                    ok = False
                if not _enforce_threshold(volume_24h, vol_thr, "volume24h<mid_min"):
                    ok = False
            elif cat == "large_cap":
                liq_thr = large_cfg.get("liquidity_threshold", 0)
                vol_thr = large_cfg.get("volume_threshold", 0)
                if not _enforce_threshold(liquidity, liq_thr, "liquidity<large_min"):
                    ok = False
                if not _enforce_threshold(volume_24h, vol_thr, "volume24h<large_min"):
                    ok = False
            elif cat == "newly_launched":
                liq_thr = new_cfg.get("liquidity_threshold", 0)
                vol_thr = new_cfg.get("volume_threshold", 0)
                if not _enforce_threshold(liquidity, liq_thr, "liquidity<new_min"):
                    ok = False
                if not _enforce_threshold(volume_24h, vol_thr, "volume24h<new_min"):
                    ok = False
        except Exception:
            # Be permissive on unexpected errors in checks
            logger.debug("Error applying thresholds for category %s on %s", cat, token_address, exc_info=True)
            continue

    # Extra sanity gates
    # reject extremely tiny market_cap unconditionally (defensive)
    try:
        if market_cap and market_cap < 1000:
            reasons.append("market_cap<1000")
            ok = False
    except Exception:
        pass

    # price change sanity: if absolute 24h change too extreme relative to config, reject
    try:
        price_change_24h = _to_float_loose(token.get('priceChange24h') or token.get('price_change_24h') or 0)
        max_price_change = float((discovery_cfg.get("max_price_change", 85)))
        if max_price_change and abs(price_change_24h) > abs(max_price_change):
            reasons.append(f"price_change24h>{max_price_change}%")
            ok = False
    except Exception:
        pass

    # If final verdict is reject, log reasons for diagnostics
    if not ok:
        try:
            # keep reasons unique and readable
            unique_reasons = ", ".join(sorted(set(reasons)))
            _log_reject(token_address, unique_reasons)
        except Exception:
            logger.info("%s rejected: %s", token_address, reasons)
        return False, categories

    # else accept
    return True, categories

async def select_top_five_per_category(eligible_tokens: List[Dict]) -> List[Dict]:
    """
    Return a shortlist biased toward mid-cap tokens.

    Behavior:
      - If no tokens, return [].
      - If no tokens have categories, fall back to top-N by volume/liquidity.
      - Otherwise, prefer mid caps up to MID_CAP_TARGET, allowing limited spillover from low/large caps.

    Defensive: tolerates missing or malformed fields and keeps existing semantics.
    """
    # Nicety: log clearly when nothing to buy this cycle
    if not eligible_tokens:
        logger.info("No buy candidates this cycle (eligible_tokens=0)")
        return []

    # Mid-cap-centric shortlist (overweight mid, limited spillover from low/high)
    try:
        logger.info("Selecting mid-cap-centric shortlist from %d eligible tokens", len(eligible_tokens))
    except Exception:
        pass

    # Split into groups
    groups = {"low_cap": [], "mid_cap": [], "large_cap": [], "newly_launched": []}
    for t in eligible_tokens:
        try:
            cats = (t.get("categories") or [])
            if "mid_cap" in cats:
                groups["mid_cap"].append(t)
            elif "low_cap" in cats:
                groups["low_cap"].append(t)
            elif "large_cap" in cats:
                groups["large_cap"].append(t)
            elif "newly_launched" in cats:
                groups["newly_launched"].append(t)
        except Exception:
            # skip malformed entries
            continue

    # ---- Fallback if no categories made it through ----
    total_cats = sum(len(v) for v in groups.values())
    if total_cats == 0:
        # Prefer tokens explicitly marked by upstream fallback
        marked = [t for t in eligible_tokens if t.get("_fallback_eligible")]
        pool = marked if marked else eligible_tokens

        def _f(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        pool_sorted = sorted(
            pool,
            key=lambda d: (
                _f(d.get("volume_24h") or d.get("v24hUSD") or d.get("volume24h")),
                _f(d.get("liquidity") or d.get("dex_liquidity")),
            ),
            reverse=True,
        )

        # Size via env (default 5) so you don’t need to change the signature
        try:
            take = int(os.getenv("FALLBACK_SHORTLIST_MIN", "5"))
        except Exception:
            take = 5

        shortlist = pool_sorted[:max(1, take)]
        # Tag for clarity (optional)
        for t in shortlist:
            try:
                t.setdefault("categories", [])
                if "fallback" not in t["categories"]:
                    t["categories"].append("fallback")
            except Exception:
                continue

        try:
            logger.warning(
                "Selector fallback engaged: no categories present; returning top %d by vol/liquidity (pool=%d).",
                len(shortlist), len(pool_sorted)
            )
        except Exception:
            pass

        try:
            return deduplicate_tokens(shortlist)
        except Exception:
            # If dedupe helper fails, return the raw shortlist defensively
            return shortlist

    # ---- Normal path: categories present ----
    # Sort each by your existing score (already set with utils.score_token)
    for k in groups:
        try:
            groups[k].sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        except Exception:
            # fallback: leave unsorted if sort fails
            pass

    # Take mid caps first
    picks: List[Dict[str, Any]] = []
    try:
        target = int(MID_CAP_TARGET or 0)
    except Exception:
        target = 0
    if target > 0:
        picks.extend(groups["mid_cap"][:target])
    else:
        picks.extend(groups["mid_cap"])

    # If we did not hit the mid target, fill with low caps first, then large caps
    need_more = max(0, (target - len(picks))) if target > 0 else 0
    if need_more > 0:
        try:
            picks.extend(groups["low_cap"][:max(LOW_CAP_SPILLOVER or 0, need_more)])
        except Exception:
            pass
        try:
            need_more = max(0, target + (LOW_CAP_SPILLOVER or 0) - len(picks))
        except Exception:
            need_more = max(0, target - len(picks))
        if need_more > 0:
            try:
                picks.extend(groups["large_cap"][:max(HIGH_CAP_SPILLOVER or 0, need_more)])
            except Exception:
                pass

    # Optionally: we keep newly_launched out of shortlist for now.
    # dedupe just in case
    try:
        picks = deduplicate_tokens(picks)
    except Exception:
        # If dedupe fails, keep picks as-is
        pass

    try:
        logger.info(
            "Shortlist (mid-first): mid=%d low=%d high=%d new=%d total=%d",
            len(groups["mid_cap"]), len(groups["low_cap"]), len(groups["large_cap"]),
            len(groups["newly_launched"]), len(picks)
        )
    except Exception:
        pass

    return picks


# ---- Signals enrichment wiring (RSI/MACD/Bollinger/TD9 + patterns) ----

def _resolve_ohlcv_fetcher():
    """
    Try to discover an async OHLCV fetcher in the current runtime.
    Must return: async fn(token_address, *, interval: str, limit: int) -> dict(open, high, low, close, volume)
    """
    try:
        import solana_trading_bot_bundle.trading_bot.market_data as market_data  # type: ignore
        if hasattr(market_data, "fetch_ohlcv"):
            async def _f(addr: str, *, interval: str = "1m", limit: int = 200):
                o = await market_data.fetch_ohlcv(token_address=addr, timeframe=interval, limit=limit)
                return {"open": o["open"], "high": o["high"], "low": o["low"], "close": o["close"], "volume": o["volume"]}
            return _f
        if hasattr(market_data, "get_ohlcv"):
            async def _g(addr: str, *, interval: str = "1m", limit: int = 200):
                o = await market_data.get_ohlcv(token_address=addr, timeframe=interval, limit=limit)
                return {"open": o["open"], "high": o["high"], "low": o["low"], "close": o["close"], "volume": o["volume"]}
            return _g
    except Exception:
        pass
    cand = globals().get("fetch_ohlcv_for_signals") or globals().get("get_ohlcv") or globals().get("fetch_ohlcv")
    if callable(cand):
        return cand
    return None

# -------------------------- Wallet / trading actions --------------------------

async def get_token_balance(
    wallet,
    token_mint: str,
    solana_client,
    token_data: dict | None = None,
) -> float:
    """
    Local helper aligned with market_data.get_token_balance signature:
    (wallet, token_mint, solana_client, token_data=None)

    Robust behavior:
      - Uses get_token_accounts_by_owner(mint=...) to find any token accounts owned by wallet.
      - If none exist, returns 0.0 (quietly, DEBUG level).
      - If one or more exist, attempts to extract ui_amount from jsonParsed response;
        falls back to get_token_account_balance for that account if needed.
      - Caches result in token_balance_cache (existing cache).
      - Avoids noisy stack traces for expected RPC "could not find account"/"Invalid param".
    """
    try:
        # Resolve owner pubkey (accepts wallet Keypair-like or a pubkey string)
        try:
            owner_pub = wallet.pubkey() if hasattr(wallet, "pubkey") and callable(getattr(wallet, "pubkey")) else wallet
        except Exception:
            owner_pub = wallet

        owner_str = str(owner_pub)

        cache_key = f"balance:{owner_str}:{token_mint}"
        cached = token_balance_cache.get(cache_key)
        if cached is not None:
            return float(cached)

        # Resolve mint Pubkey where appropriate (solders.Pubkey used elsewhere in this file)
        try:
            mint_obj = Pubkey.from_string(token_mint)
        except Exception:
            # Fall back to raw string if conversion fails (solana client may accept either)
            mint_obj = token_mint

        # Use get_token_accounts_by_owner with mint filter to find any ATAs for this owner+mint
        try:
            if COMMIT_CONFIRMED is not None:
                resp = await solana_client.get_token_accounts_by_owner(owner_str, mint=mint_obj, encoding="jsonParsed", commitment=COMMIT_CONFIRMED)
            else:
                resp = await solana_client.get_token_accounts_by_owner(owner_str, mint=mint_obj, encoding="jsonParsed")
        except Exception as e:
            # If the RPC itself fails in an unexpected way, log debug and treat as zero (don't spam ERROR)
            logger.debug("get_token_accounts_by_owner failed for owner=%s mint=%s: %s", owner_str, token_mint, str(e))
            token_balance_cache[cache_key] = 0.0
            return 0.0

        # Normalize possible response shapes (dict or object)
        accounts_list = []
        try:
            if isinstance(resp, dict):
                accounts_list = resp.get("result", {}).get("value", []) or []
            elif getattr(resp, "result", None) is not None:
                # object-like wrapper with .result.value
                rv = getattr(resp, "result", None)
                if isinstance(rv, dict):
                    accounts_list = rv.get("value", []) or []
                else:
                    # some typed responses expose .value directly
                    accounts_list = getattr(resp, "value", []) or []
            else:
                accounts_list = getattr(resp, "value", []) or []
        except Exception:
            accounts_list = []

        if not accounts_list:
            # No token account exists for this mint -> balance zero (quiet)
            logger.debug("No token account found for owner=%s mint=%s (treated as zero)", owner_str, token_mint)
            token_balance_cache[cache_key] = 0.0
            return 0.0

        # Sum ui_amount across all token accounts for this mint (rare to have multiple, but safe)
        total_ui = 0.0
        for acc in accounts_list:
            try:
                pubkey = None
                ui_amount = None

                # Account may be dict-shaped (jsonParsed) with parsed tokenAmount.uiAmount
                if isinstance(acc, dict):
                    pubkey = acc.get("pubkey") or acc.get("pubkey", None)
                    parsed = acc.get("account", {}).get("data", {}).get("parsed", {})
                    if parsed:
                        ui_amount = parsed.get("info", {}).get("tokenAmount", {}).get("uiAmount")
                else:
                    # object-like acc (typed response)
                    pubkey = getattr(acc, "pubkey", None) or getattr(acc, "address", None)
                    # try to probe parsed data
                    try:
                        parsed = getattr(acc, "account", None)
                        # if parsed is a dict-like structure, attempt extraction
                        if isinstance(parsed, dict):
                            ui_amount = parsed.get("data", {}).get("parsed", {}).get("info", {}).get("tokenAmount", {}).get("uiAmount")
                    except Exception:
                        ui_amount = None

                # If ui_amount not found in parsed payload, fall back to get_token_account_balance on the token account pubkey
                if ui_amount is None and pubkey:
                    try:
                        if COMMIT_CONFIRMED is not None:
                            bal_resp = await solana_client.get_token_account_balance(pubkey, commitment=COMMIT_CONFIRMED)
                        else:
                            bal_resp = await solana_client.get_token_account_balance(pubkey)
                        # robustly parse possible shapes
                        if isinstance(bal_resp, dict):
                            ui_amount = float(bal_resp.get("result", {}).get("value", {}).get("uiAmount") or 0.0)
                        elif getattr(bal_resp, "value", None) is not None:
                            v = getattr(bal_resp, "value")
                            ui_amount = float(getattr(v, "ui_amount", getattr(v, "uiAmount", 0.0)) or 0.0)
                        else:
                            # fallback: try attributes
                            ui_amount = float(getattr(bal_resp, "ui_amount", getattr(bal_resp, "uiAmount", 0.0)) or 0.0)
                    except Exception as e:
                        # Treat "could not find account" / "Invalid param" as zero quietly.
                        msg = str(e)
                        if "could not find account" in msg or "Invalid param" in msg:
                            logger.debug("Token account missing when fetching balance (owner=%s mint=%s pubkey=%s): %s", owner_str, token_mint, pubkey, msg)
                            ui_amount = 0.0
                        else:
                            logger.debug("get_token_account_balance failed for pubkey=%s owner=%s mint=%s: %s", pubkey, owner_str, token_mint, msg)
                            ui_amount = 0.0

                total_ui += float(ui_amount or 0.0)
            except Exception as e:
                # Be defensive: skip malformed account entries but keep going
                logger.debug("Error reading token account entry for owner=%s mint=%s: %s", owner_str, token_mint, str(e))
                continue

        # Cache and return float ui_amount (same semantic as original)
        token_balance_cache[cache_key] = float(total_ui)
        return float(total_ui)

    except Exception as e:
        # Unexpected top-level errors: log at debug (avoid noisy error spam for expected missing-account cases)
        sym = (token_data or {}).get("symbol", "UNKNOWN")
        logger.debug("Unexpected error fetching token balance for %s (%s): %s", sym, token_mint, str(e), exc_info=True)
        try:
            token_balance_cache[cache_key] = 0.0
        except Exception:
            pass
        return 0.0

# Cache for token mint decimals
token_decimals_cache: Dict[str, int] = {}

async def get_token_decimals(
    token_mint: str,
    solana_client: Any,
    logger: logging.Logger,
) -> int:
    """
    Fetch SPL token mint decimals (cached).
    Returns 9 if unknown or on RPC limitations.
    """
    if token_mint in token_decimals_cache:
        return token_decimals_cache[token_mint]

    # Fast path: get_token_supply exposes decimals
    try:
        supply = await solana_client.get_token_supply(Pubkey.from_string(token_mint))
        dec = getattr(getattr(supply, "value", None), "decimals", None)
        if isinstance(dec, int) and 0 <= dec <= 18:
            token_decimals_cache[token_mint] = dec
            return dec
    except Exception:
        pass

    # Fallback: try largest accounts -> read one account's balance (includes decimals)
    try:
        la = await solana_client.get_token_largest_accounts(Pubkey.from_string(token_mint))
        value = getattr(la, "value", None)
        if isinstance(value, list) and value:
            v0 = value[0]
            first_address = v0.get("address") if isinstance(v0, dict) else getattr(v0, "address", None)
            if first_address:
                bal = await solana_client.get_token_account_balance(Pubkey.from_string(first_address))
                dec = getattr(getattr(bal, "value", None), "decimals", None)
                if isinstance(dec, int) and 0 <= dec <= 18:
                    token_decimals_cache[token_mint] = dec
                    return dec
    except Exception:
        pass

    logger.debug(f"Using default decimals=9 for {token_mint}")
    token_decimals_cache[token_mint] = 9
    return 9

async def real_on_chain_buy(
    token: dict[str, Any],
    buy_amount: float,
    wallet: Keypair,
    solana_client: Any,
    session: aiohttp.ClientSession,
    blacklist: set[str],
    failure_count: dict[str, int],
    config: dict[str, Any],
) -> str | None:
    """
    Robust buy executor that supports DRY-RUN and WATCH-ONLY properly.
    - On dry-run (skip onchain send) it synthesizes a txid, emits metrics,
      and persists a simulated buy via update_token_record (preferred) or
      _persist_dryrun_buy_into_trade_history (fallback).
    - Avoids early-return when ATA missing in dry-run: simulates buy anyway.
    """
    token_address = token.get("address")
    symbol = token.get("symbol", "UNKNOWN")
    if not token_address:
        logger.warning(f"Token missing address: {token}")
        return None

    try:
        # Safety: never buy stables/stake-wrappers if they slipped through
        if _is_core_stable_like(token):
            logger.info(f"Skipping stable/wrapper {symbol} ({token_address})")
            return None

        # Validate mint; if invalid and not whitelisted, blacklist
        if not await validate_token_mint(token_address, solana_client):
            logger.warning(f"Invalid token mint {token_address} for {symbol}")
            if token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, "Invalid token mint")
                blacklist.add(token_address)
            return None

        # Determine token price in SOL
        token_price = await get_token_price_in_sol(token_address, session, price_cache, token)
        if token_price <= 0:
            logger.warning(f"Invalid token price for {symbol} ({token_address}): {token_price}")
            failure_count[token_address] = failure_count.get(token_address, 0) + 1
            if failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, "Persistent invalid token price")
                blacklist.add(token_address)
            return None

        lamports = int(buy_amount * 1_000_000_000)
        expected_tokens_est = (buy_amount / token_price) if token_price > 0 else 0.0
        logger.debug(f"Buying ~{expected_tokens_est:.6f} units of {symbol} ({token_address}) for {buy_amount:.6f} SOL")

        # Preflight Jupiter tradability probe (keeps behavior in original file)
        tradable, t_error = await _is_jupiter_tradable(
            input_mint="So11111111111111111111111111111111111111112",
            output_mint=token_address,
            lamports=max(1000000, int(lamports * 0.01)),
            user_pubkey=str(wallet.pubkey()),
            session=session,
            slippage_bps=_clamp_bps(int(config['bot'].get('buy_slippage_bps', 50)))
        )
        if not tradable:
            logger.error(f"Preflight Jupiter probe indicates {symbol} not tradable: {t_error}")
            failure_count[token_address] = failure_count.get(token_address, 0) + 1
            if failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, f"Not tradable: {t_error}")
                blacklist.add(token_address)
            return None

        # Check / ensure ATA: in dry-run we do not abort if ATA missing;
        # only attempt ATA creation in live mode.
        assoc_addr_exists = await check_token_account(
            str(wallet.pubkey()), token_address, solana_client, blacklist, failure_count, token
        )

        if not assoc_addr_exists:
            if _skip_onchain_send(config):
                # Dry-run: log, but continue to simulate buy
                logger.info("Watch-only: no ATA yet for %s (%s) — simulating buy in dry-run.", symbol, token_address)
                # proceed without creating ATA
            else:
                # Live path: attempt to create ATA
                try:
                    assoc_addr = await create_token_account(
                        wallet=wallet,
                        mint_address=token_address,
                        solana_client=solana_client,
                        token_data=token,
                    )
                except TypeError as e:
                    logger.error("Failed to create token account for %s (%s): %s", symbol, token_address, e)
                    assoc_addr = None
                except Exception as e:
                    logger.exception("Failed to create token account for %s (%s): %s", symbol, token_address, e)
                    assoc_addr = None

                if not assoc_addr:
                    logger.error("ATA creation unavailable for %s (%s); blacklisting.", symbol, token_address)
                    if token_address not in WHITELISTED_TOKENS:
                        await db_add_to_blacklist(token_address, "Failed to create token account")
                        blacklist.add(token_address)
                    return None

        # Get Jupiter quote (safe wrapper)
        quote, error = await _safe_get_jupiter_quote(
            input_mint="So11111111111111111111111111111111111111112",
            output_mint=token_address,
            amount=lamports,
            user_pubkey=str(wallet.pubkey()),
            session=session,
            slippage_bps=_clamp_bps(int(config['bot'].get('buy_slippage_bps', 50)))
        )
        if not quote or error:
            logger.error(f"Failed to get Jupiter quote for {symbol} ({token_address}): {error}")
            failure_count[token_address] = failure_count.get(token_address, 0) + 1
            if failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, f"Jupiter quote error: {error}")
                blacklist.add(token_address)
            return None

        # Decide dry-run / send behavior
        if _skip_onchain_send(config):
            txid = _dry_run_txid("buy")
            _log_trade_event(
                "DRYRUN_BUY",
                token=symbol,
                address=token_address,
                lamports=lamports,
                slippage_bps=_clamp_bps(int(config['bot'].get('buy_slippage_bps', 50))),
                txid=txid
            )

            # Record metrics for dry-run buy
            _record_metric_fill(
                token_addr=token_address,
                symbol=symbol,
                side="BUY",
                quote=quote,
                buy_amount_sol=buy_amount,
                token_price_sol=token_price,
                txid=txid,
                simulated=True,
                source="jupiter",
            )

            # Persist a simulated buy state so sells/watch logic can operate on it.
            now_ts = int(time.time())
            exit_cfg = _load_exit_cfg(config)
            trade_state = {
                "tp1_done": False,
                "tp2_done": False,
                "highest_price": float(token_price),
                "breakeven_floor": float(token_price) * (1.0 + exit_cfg["breakeven_plus"]),
                "no_tp_deadline": now_ts + int(exit_cfg["no_tp_cutoff_hours"] * 3600),
                "hard_exit_deadline": now_ts + int(exit_cfg["max_hold_hours"] * 3600),
                "trail_pct": None,
            }

            # Preferred persistence: update_token_record if available (keeps schema consistent)
            try:
                if callable(globals().get("update_token_record")):
                    await update_token_record(
                        token={
                            "address": token_address,
                            "name": token.get("name", "UNKNOWN"),
                            "symbol": symbol,
                            "volume_24h": float(token.get('v24hUSD', 0)),
                            "liquidity": float(token.get('liquidity', 0)),
                            "market_cap": float(token.get('mc', 0) or token.get('fdv', 0)),
                            "price": float(token.get('price', 0) or 0),
                            "price_change_1h": float(token.get('price_change_1h', 0) or token.get('priceChange1h', 0) or 0),
                            "price_change_6h": float(token.get('price_change_6h', 0) or 0),
                            "price_change_24h": float(token.get('price_change_24h', 0) or token.get('priceChange24h', 0) or 0),
                            "categories": token.get('categories', []),
                            "timestamp": now_ts,
                            "trade_state": trade_state,
                        },
                        buy_price=float(token_price),
                        buy_txid=str(txid),
                        buy_time=now_ts,
                        is_trading=True,
                    )
                else:
                    # Fallback: persist via the specialized dryrun helper when update_token_record isn't present
                    await _persist_dryrun_buy_into_trade_history(
                        token_address=token_address,
                        symbol=symbol,
                        buy_price=float(token_price),
                        buy_amount_sol=float(buy_amount),
                        buy_txid=txid,
                        buy_time=now_ts,
                        config=config,
                    )
            except Exception:
                logger.debug("Persisting dry-run buy failed; continuing (non-fatal).", exc_info=True)

        else:
            # Live execution path: execute Jupiter swap
            txid = await execute_jupiter_swap(
                        quote,
                        str(wallet.pubkey()),
                        wallet,
                        solana_client,
                        session=session,
                    )
            if not txid:
                logger.error(f"Failed to execute swap for {symbol} ({token_address})")
                failure_count[token_address] = failure_count.get(token_address, 0) + 1
                if failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
                    await db_add_to_blacklist(token_address, "Swap execution failed")
                    blacklist.add(token_address)
                return None

            # Record metrics for live buy
            _record_metric_fill(
                token_addr=token_address,
                symbol=symbol,
                side="BUY",
                quote=quote,
                buy_amount_sol=buy_amount,
                token_price_sol=token_price,
                txid=txid,
                simulated=False,
                source="jupiter",
            )

            # Persist live buy
            now_ts = int(time.time())
            exit_cfg = _load_exit_cfg(config)
            trade_state = {
                "tp1_done": False,
                "tp2_done": False,
                "highest_price": float(token_price),
                "breakeven_floor": float(token_price) * (1.0 + exit_cfg["breakeven_plus"]),
                "no_tp_deadline": now_ts + int(exit_cfg["no_tp_cutoff_hours"] * 3600),
                "hard_exit_deadline": now_ts + int(exit_cfg["max_hold_hours"] * 3600),
                "trail_pct": None,
            }

            try:
                await update_token_record(
                    token={
                        "address": token_address,
                        "name": token.get("name", "UNKNOWN"),
                        "symbol": symbol,
                        "volume_24h": float(token.get('v24hUSD', 0)),
                        "liquidity": float(token.get('liquidity', 0)),
                        "market_cap": float(token.get('mc', 0) or token.get('fdv', 0)),
                        "price": float(token.get('price', 0) or 0),
                        "price_change_1h": float(token.get('price_change_1h', 0) or token.get('priceChange1h', 0) or 0),
                        "price_change_6h": float(token.get('price_change_6h', 0) or 0),
                        "price_change_24h": float(token.get('price_change_24h', 0) or token.get('priceChange24h', 0) or 0),
                        "categories": token.get('categories', []),
                        "timestamp": now_ts,
                        "trade_state": trade_state,
                    },
                    buy_price=float(token_price),
                    buy_txid=str(txid),
                    buy_time=now_ts,
                    is_trading=True,
                )
            except Exception:
                logger.debug("Persisting live buy failed (non-fatal)", exc_info=True)

        logger.info(f"Successfully bought {symbol} ({token_address}) for {buy_amount:.6f} SOL, txid: {txid}")
        return txid

    except Exception as e:
        log_error_with_stacktrace(f"Error executing buy for {symbol} ({token_address})", e)
        failure_count[token_address] = failure_count.get(token_address, 0) + 1
        if failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
            await db_add_to_blacklist(token_address, "Repeated buy failures")
            blacklist.add(token_address)
        return None

async def real_on_chain_sell(
    token: dict[str, Any],
    wallet: Keypair,
    solana_client: Any,
    session: aiohttp.ClientSession,
    blacklist: set[str],
    failure_count: dict[str, int],
    config: dict[str, Any],
) -> str | None:
    token_address = token.get("address")
    symbol = token.get("symbol", "UNKNOWN")
    if not token_address:
        logger.warning(f"Token missing address: {token}")
        return None
    try:
        # Basic safety checks
        if not await validate_token_mint(token_address, solana_client):
            logger.warning(f"Invalid token mint {token_address} for {symbol}")
            if token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, "Invalid token mint")
                blacklist.add(token_address)
            return None
        
        assoc_addr_exists = await check_token_account(
            str(wallet.pubkey()), token_address, solana_client, blacklist, failure_count, token
        )

        # Detect dry-run / simulate / send-tx disabled
        dry_run = _skip_onchain_send(config)

        if not assoc_addr_exists:
            if dry_run:
                # In dry-run we expect no ATA until a real buy happens; don't blacklist
                logger.info("Watch-only: no ATA yet for %s (%s) - expected in dry-run.", symbol, token_address)
                return None
            else:
                logger.error("No token account exists for %s (%s)", symbol, token_address)
                if token_address not in WHITELISTED_TOKENS:
                    await db_add_to_blacklist(token_address, "No token account")
                    blacklist.add(token_address)
                return None

        token_balance = await get_token_balance(wallet, token_address, solana_client, token)
        if token_balance <= 0:
            logger.debug(f"No tokens available to sell for {symbol} ({token_address})")
            return None

        status = await get_token_trade_status(token_address)
        if not status or status.get("buy_price") is None or status.get("buy_time") is None:
            if dry_run:
                # In dry-run there may never be a recorded buy; don't blacklist
                logger.info("No valid buy record for %s (%s) - expected in dry-run.", symbol, token_address)
                return None
            else:
                logger.warning(f"No valid buy record for {symbol} ({token_address})")
                if token_address not in WHITELISTED_TOKENS:
                    await db_add_to_blacklist(token_address, "No buy record")
                    blacklist.add(token_address)
                return None

        exit_cfg = _load_exit_cfg(config)

        buy_price = float(status.get("buy_price") or 0)
        buy_time = int(status.get("buy_time") or 0)
        now_ts = int(time.time())

        # FIX: signature (no logger)
        current_price = await get_token_price_in_sol(token_address, session, price_cache, token)
        if current_price <= 0 or buy_price <= 0:
            logger.warning(
                f"Invalid price(s) for {symbol} ({token_address}): current={current_price}, buy={buy_price}"
            )
            failure_count[token_address] = failure_count.get(token_address, 0) + 1
            if not dry_run and failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
                await db_add_to_blacklist(token_address, "Persistent invalid current/buy price")
                blacklist.add(token_address)
            return None

        # Mint decimals (use everywhere we compute amounts)
        decimals = await get_token_decimals(token_address, solana_client, logger)

        # Per-trade state
        ts = await _get_trade_state(token_address)
        _update_highest(ts, current_price)

        profit_ratio = current_price / buy_price
        hours_held = (now_ts - buy_time) / 3600.0

        # ----- TIME-BASED EXITS -----

        # Stagnant exit (full)
        if (not ts.get("tp1_done", False)
            and hours_held >= exit_cfg["no_tp_cutoff_hours"]
            and profit_ratio < exit_cfg["no_tp_min_profit_x"]):
            logger.info(
                f"[{symbol}] No TP1 within {exit_cfg['no_tp_cutoff_hours']}h and "
                f"profit {profit_ratio:.2f}x < {exit_cfg['no_tp_min_profit_x']}x → exit full."
            )
            amount = _sell_amount_lamports(token_balance, 1.0, default_decimals=decimals)
            # FIX: signature (no logger kwarg)
            quote, error = await _safe_get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount=amount,
                user_pubkey=str(wallet.pubkey()),
                session=session,
                slippage_bps=exit_cfg["sell_slippage_bps"]
            )
            if not quote or error:
                logger.error(f"Failed stagnant-exit quote for {symbol} ({token_address}): {error}")
                return None

            if _skip_onchain_send(config):
                txid = _dry_run_txid("sell")
                _log_trade_event(
                    "DRYRUN_SELL",
                    token=symbol,
                    address=token_address,
                    amount_lamports=amount,
                    reason="STALE",
                    slippage_bps=exit_cfg["sell_slippage_bps"],
                    txid=txid
                )
                # Record metrics for dry-run sell (stagnant exit)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=True,
                    source="jupiter",
                    reason="STALE",
                )

                # Persist DRYRUN sell into trade_history (best-effort) BEFORE marking sold
                try:
                    token_qty = float(token_balance) if token_balance is not None else (float(amount) / (10 ** int(decimals)))
                except Exception:
                    try:
                        token_qty = float(amount) / (10 ** int(decimals))
                    except Exception:
                        token_qty = None
                try:
                    await _persist_dryrun_sell_into_trade_history(
                        token_address=token_address,
                        symbol=symbol,
                        sell_price=float(current_price),
                        sell_amount_tokens=(float(token_qty) if token_qty is not None else None),
                        sell_txid=txid,
                        sell_time=now_ts,
                        config=config,
                    )
                except Exception:
                    logger.debug("Persist DRYRUN sell helper failed (ignored)", exc_info=True)

            else:
                txid = await execute_jupiter_swap(
                        quote,
                        str(wallet.pubkey()),
                        wallet,
                        solana_client,
                        session=session,
                    )
                if not txid:
                    logger.error(f"Failed stagnant-exit sell for {symbol} ({token_address})")
                    return None
                # Record metrics for live sell (stagnant exit)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=False,
                    source="jupiter",
                    reason="STALE",
                )

            await mark_token_sold(
                token_address=token_address,
                sell_price=float(current_price),
                sell_txid=str(txid),
                sell_time=now_ts,
            )
            logger.info(f"[{symbol}] Stagnant exit complete. txid={txid}")
            return txid

        # Max-hold hard exit (full)
        if hours_held >= exit_cfg["max_hold_hours"]:
            logger.info(f"[{symbol}] Max-hold {exit_cfg['max_hold_hours']}h reached → exit full.")
            amount = _sell_amount_lamports(token_balance, 1.0, default_decimals=decimals)
            quote, error = await _safe_get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount=amount,
                user_pubkey=str(wallet.pubkey()),
                session=session,
                slippage_bps=exit_cfg["sell_slippage_bps"]
            )
            if not quote or error:
                logger.error(f"Failed max-hold quote for {symbol} ({token_address}): {error}")
                return None

            if _skip_onchain_send(config):
                txid = _dry_run_txid("sell")
                _log_trade_event(
                    "DRYRUN_SELL",
                    token=symbol,
                    address=token_address,
                    amount_lamports=amount,
                    reason="MAX_HOLD",
                    slippage_bps=exit_cfg["sell_slippage_bps"],
                    txid=txid
                )
                # Record metrics for dry-run sell (max-hold exit)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=True,
                    source="jupiter",
                    reason="MAX_HOLD",
                )

                # Persist DRYRUN sell into trade_history (best-effort) BEFORE marking sold
                try:
                    token_qty = float(token_balance) if token_balance is not None else (float(amount) / (10 ** int(decimals)))
                except Exception:
                    try:
                        token_qty = float(amount) / (10 ** int(decimals))
                    except Exception:
                        token_qty = None
                try:
                    await _persist_dryrun_sell_into_trade_history(
                        token_address=token_address,
                        symbol=symbol,
                        sell_price=float(current_price),
                        sell_amount_tokens=(float(token_qty) if token_qty is not None else None),
                        sell_txid=txid,
                        sell_time=now_ts,
                        config=config,
                    )
                except Exception:
                    logger.debug("Persist DRYRUN sell helper failed (ignored)", exc_info=True)

            else:
                txid = await execute_jupiter_swap(
                        quote,
                        str(wallet.pubkey()),
                        wallet,
                        solana_client,
                        session=session,
                    )
                if not txid:
                    logger.error(f"Failed max-hold sell for {symbol} ({token_address})")
                    return None
                # Record metrics for live sell (max-hold exit)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=False,
                    source="jupiter",
                    reason="MAX_HOLD",
                )

            await mark_token_sold(
                token_address=token_address,
                sell_price=float(current_price),
                sell_txid=str(txid),
                sell_time=now_ts
            )
            logger.info(f"[{symbol}] Max-hold exit complete. txid={txid}")
            return txid

        # ----- PRICE-BASED EXITS -----

        # Pre-TP1 stop-loss (full)
        if not ts.get("tp1_done", False) and profit_ratio <= exit_cfg["stop_loss_x"]:
            logger.info(f"[{symbol}] Stop-loss hit at {profit_ratio:.2f}x (<= {exit_cfg['stop_loss_x']}x) -> sell all.")
            amount = _sell_amount_lamports(token_balance, 1.0, default_decimals=decimals)
            quote, error = await _safe_get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount=amount,
                user_pubkey=str(wallet.pubkey()),
                session=session,
                slippage_bps=exit_cfg["sell_slippage_bps"]
            )
            if not quote or error:
                logger.error(f"Failed stop-loss quote for {symbol} ({token_address}): {error}")
                return None

            if _skip_onchain_send(config):
                txid = _dry_run_txid("sell")
                _log_trade_event(
                    "DRYRUN_SELL",
                    token=symbol,
                    address=token_address,
                    amount_lamports=amount,
                    reason="STOP_LOSS",
                    slippage_bps=exit_cfg["sell_slippage_bps"],
                    txid=txid
                )
                # Record metrics for dry-run sell (stop-loss)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=True,
                    source="jupiter",
                    reason="STOP_LOSS",
                )

                # Persist DRYRUN sell into trade_history (best-effort) BEFORE marking sold
                try:
                    token_qty = float(token_balance) if token_balance is not None else (float(amount) / (10 ** int(decimals)))
                except Exception:
                    try:
                        token_qty = float(amount) / (10 ** int(decimals))
                    except Exception:
                        token_qty = None
                try:
                    await _persist_dryrun_sell_into_trade_history(
                        token_address=token_address,
                        symbol=symbol,
                        sell_price=float(current_price),
                        sell_amount_tokens=(float(token_qty) if token_qty is not None else None),
                        sell_txid=txid,
                        sell_time=now_ts,
                        config=config,
                    )
                except Exception:
                    logger.debug("Persist DRYRUN sell helper failed (ignored)", exc_info=True)

            else:
                txid = await execute_jupiter_swap(
                        quote,
                        str(wallet.pubkey()),
                        wallet,
                        solana_client,
                        session=session,
                    )
                if not txid:
                    logger.error(f"Failed stop-loss sell for {symbol} ({token_address})")
                    return None
                # Record metrics for live sell (stop-loss)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=False,
                    source="jupiter",
                    reason="STOP_LOSS",
                )

            await mark_token_sold(token_address=token_address, sell_price=float(current_price), sell_txid=str(txid), sell_time=now_ts)
            logger.info(f"[{symbol}] Stop-loss executed. txid={txid}")
            return txid

        # TP1: partial take profit + start trailing (partial)
        if not ts.get("tp1_done", False) and profit_ratio >= exit_cfg["tp1_x"]:
            pct = exit_cfg["tp1_pct_to_sell"]
            amount = _sell_amount_lamports(token_balance, pct, default_decimals=decimals)
            logger.info(f"[{symbol}] TP1 {profit_ratio:.2f}x → sell {pct*100:.1f}% and start {exit_cfg['trail_pct_after_tp1']*100:.0f}% trailing.")
            quote, error = await _safe_get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount=amount,
                user_pubkey=str(wallet.pubkey()),
                session=session,
                slippage_bps=exit_cfg["sell_slippage_bps"]
            )
            if not quote or error:
                logger.error(f"Failed TP1 quote for {symbol} ({token_address}): {error}")
                return None

            if _skip_onchain_send(config):
                txid = _dry_run_txid("sell")
                _log_trade_event(
                    "DRYRUN_SELL",
                    token=symbol,
                    address=token_address,
                    amount_lamports=amount,
                    reason="TP1",
                    slippage_bps=exit_cfg["sell_slippage_bps"],
                    txid=txid
                )
                # Record metrics for dry-run sell (TP1)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=True,
                    source="jupiter",
                    reason="TP1",
                )

                # Persist DRYRUN partial sell into trade_history (best-effort)
                try:
                    token_qty = None
                    try:
                        token_qty = float(amount) / (10 ** int(decimals))
                    except Exception:
                        token_qty = None
                    await _persist_dryrun_sell_into_trade_history(
                        token_address=token_address,
                        symbol=symbol,
                        sell_price=float(current_price),
                        sell_amount_tokens=(float(token_qty) if token_qty is not None else None),
                        sell_txid=txid,
                        sell_time=now_ts,
                        config=config,
                    )
                except Exception:
                    logger.debug("Persist DRYRUN sell helper failed (ignored)", exc_info=True)

            else:
                txid = await execute_jupiter_swap(
                        quote,
                        str(wallet.pubkey()),
                        wallet,
                        solana_client,
                        session=session,
                    )
                if not txid:
                    logger.error(f"Failed TP1 sell for {symbol} ({token_address})")
                    return None
                # Record metrics for live sell (TP1)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=False,
                    source="jupiter",
                    reason="TP1",
                )

            # Activate trailing & breakeven protection (partial sell does NOT mark sold)
            ts["tp1_done"] = True
            ts["trail_pct"] = exit_cfg["trail_pct_after_tp1"]
            ts["breakeven_floor"] = max(float(ts.get("breakeven_floor") or 0), buy_price * (1.0 + exit_cfg["breakeven_plus"]))
            _update_highest(ts, current_price)
            await _save_trade_state(token_address, ts)
            logger.info(f"[{symbol}] TP1 partial sell done. txid={txid}")
            return txid  # one action per cycle

        # TP2: second partial + widen trailing (partial)
        if ts.get("tp1_done", False) and (not ts.get("tp2_done", False)) and profit_ratio >= exit_cfg["tp2_x"]:
            pct = exit_cfg["tp2_pct_of_remaining"]
            amount = _sell_amount_lamports(token_balance, pct, default_decimals=decimals)
            logger.info(f"[{symbol}] TP2 {profit_ratio:.2f}x -> sell {pct*100:.1f}% of remaining; widen trail to {exit_cfg['trail_pct_moonbag']*100:.0f}%.")
            quote, error = await _safe_get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",
                amount=amount,
                user_pubkey=str(wallet.pubkey()),
                session=session,
                slippage_bps=exit_cfg["sell_slippage_bps"]
            )
            if not quote or error:
                logger.error(f"Failed TP2 quote for {symbol} ({token_address}): {error}")
                return None

            if _skip_onchain_send(config):
                txid = _dry_run_txid("sell")
                _log_trade_event(
                    "DRYRUN_SELL",
                    token=symbol,
                    address=token_address,
                    amount_lamports=amount,
                    reason="TP2",
                    slippage_bps=exit_cfg["sell_slippage_bps"],
                    txid=txid
                )
                # Record metrics for dry-run sell (TP2)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=True,
                    source="jupiter",
                    reason="TP2",
                )

                # Persist DRYRUN partial sell into trade_history (best-effort)
                try:
                    token_qty = None
                    try:
                        token_qty = float(amount) / (10 ** int(decimals))
                    except Exception:
                        token_qty = None
                    await _persist_dryrun_sell_into_trade_history(
                        token_address=token_address,
                        symbol=symbol,
                        sell_price=float(current_price),
                        sell_amount_tokens=(float(token_qty) if token_qty is not None else None),
                        sell_txid=txid,
                        sell_time=now_ts,
                        config=config,
                    )
                except Exception:
                    logger.debug("Persist DRYRUN sell helper failed (ignored)", exc_info=True)

            else:
                txid = await execute_jupiter_swap(
                        quote,
                        str(wallet.pubkey()),
                        wallet,
                        solana_client,
                        session=session,
                    )
                if not txid:
                    logger.error(f"Failed TP2 sell for {symbol} ({token_address})")
                    return None
                # Record metrics for live sell (TP2)
                _record_metric_fill(
                    token_addr=token_address,
                    symbol=symbol,
                    side="SELL",
                    quote=quote,
                    buy_amount_sol=None,
                    token_price_sol=current_price,
                    txid=txid,
                    simulated=False,
                    source="jupiter",
                    reason="TP2",
                )

            ts["tp2_done"] = True
            ts["trail_pct"] = exit_cfg["trail_pct_moonbag"]
            _update_highest(ts, current_price)
            await _save_trade_state(token_address, ts)
            logger.info(f"[{symbol}] TP2 partial sell done. txid={txid}")
            return txid  # one action per cycle

        # Trailing stops (post-TP1 or post-TP2) (full)
        if ts.get("tp1_done", False):
            hp = float(ts.get("highest_price") or current_price)
            trail_pct = float(ts.get("trail_pct") or exit_cfg["trail_pct_after_tp1"])
            trail_floor = hp * (1.0 - trail_pct)
            # Ensure floor respects breakeven after TP1
            breakeven_floor = float(ts.get("breakeven_floor") or (buy_price * (1.0 + exit_cfg["breakeven_plus"])))
            active_floor = max(trail_floor, breakeven_floor)

            if current_price <= active_floor:
                logger.info(f"[{symbol}] Trailing stop @ {current_price:.8f} SOL (floor {active_floor:.8f}) -> exit full.")
                amount = _sell_amount_lamports(token_balance, 1.0, default_decimals=decimals)
                quote, error = await _safe_get_jupiter_quote(
                    input_mint=token_address,
                    output_mint="So11111111111111111111111111111111111111112",
                    amount=amount,
                    user_pubkey=str(wallet.pubkey()),
                    session=session,
                    slippage_bps=exit_cfg["sell_slippage_bps"]
                )
                if not quote or error:
                    logger.error(f"Failed trailing-stop quote for {symbol} ({token_address}): {error}")
                    return None

                if _skip_onchain_send(config):
                    txid = _dry_run_txid("sell")
                    _log_trade_event(
                        "DRYRUN_SELL",
                        token=symbol,
                        address=token_address,
                        amount_lamports=amount,
                        reason="TRAIL",
                        slippage_bps=exit_cfg["sell_slippage_bps"],
                        txid=txid
                    )
                    # Record metrics for dry-run sell (trailing stop)
                    _record_metric_fill(
                        token_addr=token_address,
                        symbol=symbol,
                        side="SELL",
                        quote=quote,
                        buy_amount_sol=None,
                        token_price_sol=current_price,
                        txid=txid,
                        simulated=True,
                        source="jupiter",
                        reason="TRAIL",
                    )

                    # Persist DRYRUN sell into trade_history (best-effort) BEFORE marking sold
                    try:
                        token_qty = float(token_balance) if token_balance is not None else (float(amount) / (10 ** int(decimals)))
                    except Exception:
                        try:
                            token_qty = float(amount) / (10 ** int(decimals))
                        except Exception:
                            token_qty = None
                    try:
                        await _persist_dryrun_sell_into_trade_history(
                            token_address=token_address,
                            symbol=symbol,
                            sell_price=float(current_price),
                            sell_amount_tokens=(float(token_qty) if token_qty is not None else None),
                            sell_txid=txid,
                            sell_time=now_ts,
                            config=config,
                        )
                    except Exception:
                        logger.debug("Persist DRYRUN sell helper failed (ignored)", exc_info=True)

                else:
                    txid = await execute_jupiter_swap(
                        quote,
                        str(wallet.pubkey()),
                        wallet,
                        solana_client,
                        session=session,
                    )
                    if not txid:
                        logger.error(f"Failed trailing-stop sell for {symbol} ({token_address})")
                        return None
                    # Record metrics for live sell (trailing stop)
                    _record_metric_fill(
                        token_addr=token_address,
                        symbol=symbol,
                        side="SELL",
                        quote=quote,
                        buy_amount_sol=None,
                        token_price_sol=current_price,
                        txid=txid,
                        simulated=False,
                        source="jupiter",
                        reason="TRAIL",
                    )

                await mark_token_sold(token_address=token_address, sell_price=float(current_price), sell_txid=str(txid), sell_time=now_ts)
                logger.info(f"[{symbol}] Trailing stop exit complete. txid={txid}")
                return txid

            # Keep tracking the high and persist occasionally
            if current_price > hp * 1.001:  # tiny hysteresis to reduce DB churn
                ts["highest_price"] = float(current_price)
                await _save_trade_state(token_address, ts)

        # Nothing to do this cycle
        return None

    except Exception as e:
        log_error_with_stacktrace(f"Error executing sell for {symbol} ({token_address})", e)
        failure_count[token_address] = failure_count.get(token_address, 0) + 1

        # Only blacklist repeated failures when actually sending transactions
        if (not _skip_onchain_send(config)) and failure_count[token_address] >= 5 and token_address not in WHITELISTED_TOKENS:
            await db_add_to_blacklist(token_address, "Repeated sell failures")
            blacklist.add(token_address)

        return None

# -------------------------- Diagnostics: open-position snapshot --------------------------

async def _log_positions_snapshot(
    open_positions: List[Dict[str, Any]],
    session: aiohttp.ClientSession,
    solana_client: Any,            # <- use Any so Pylance doesn't require a real type
    config: Dict[str, Any],
) -> None:
    if not open_positions:
        logger.info("[WATCH] No open positions.")
        return

    exit_cfg = _load_exit_cfg(config)

    # Ensure we have a price cache object. Prefer an existing module-level cache if present,
    # otherwise use a small local dict to avoid NameError.
    price_cache = globals().get("price_cache", {})

    for pos in open_positions:
        addr = pos.get("address")
        sym = pos.get("symbol", "UNKNOWN")
        if not addr:
            continue

        try:
            # Attempt to load persisted trade status; tolerate missing import or DB issues.
            try:
                status = await get_token_trade_status(addr)
            except NameError:
                # get_token_trade_status not defined (missing import/removed). Continue without status.
                logger.warning("[WATCH] get_token_trade_status is not defined; skipping trade status for %s", addr)
                status = None
            except Exception as _e:
                # Log other unexpected errors but continue with snapshot
                logger.warning("[WATCH] Failed to obtain token trade status for %s: %s", addr, _e, exc_info=True)
                status = None

            # Always derive buy_price / buy_time from the (possibly None) status so they exist below.
            buy_price = float((status or {}).get("buy_price") or 0.0)
            buy_time = int((status or {}).get("buy_time") or 0)

            # Current price (uses price_cache that is guaranteed to exist)
            try:
                current_price = await get_token_price_in_sol(
                    addr, session, price_cache, {"address": addr, "symbol": sym}
                )
            except Exception as _e:
                logger.warning("[WATCH] Failed to get current price for %s: %s", addr, _e, exc_info=True)
                current_price = 0.0

            profit_x = (current_price / buy_price) if (current_price > 0 and buy_price > 0) else 0.0
            hours_held = (time.time() - buy_time) / 3600.0 if buy_time else 0.0

            # Trade-state fields (if any)
            ts = ((status or {}).get("trade_state") or {})
            tp1_done = bool(ts.get("tp1_done", False))
            tp2_done = bool(ts.get("tp2_done", False))
            highest_price = float(ts.get("highest_price") or 0.0)
            trail_pct = ts.get("trail_pct")  # None, 0.18, or 0.30
            breakeven_floor = ts.get("breakeven_floor")

            trail_floor = None
            if highest_price > 0 and trail_pct not in (None, 0, 0.0):
                trail_floor = highest_price * (1.0 - float(trail_pct))

            # Active floor after TP1 = max(trailing floor, breakeven+)
            active_floor = None
            if tp1_done:
                be = float(breakeven_floor or (buy_price * (1.0 + exit_cfg["breakeven_plus"])))
                tr = float(trail_floor or 0.0)
                active_floor = max(be, tr)

            logger.info(
                "[WATCH] %s %s | price=%.8f SOL | buy=%.8f | pnl=%.2fx | held=%.2fh | "
                "TP1=%s TP2=%s | high=%s | trail=%s | floor=%s",
                sym,
                f"{addr[:4]}…{addr[-4:]}",
                current_price,
                buy_price,
                profit_x,
                hours_held,
                tp1_done,
                tp2_done,
                f"{highest_price:.8f}" if highest_price else "-",
                f"{float(trail_pct)*100:.1f}%" if trail_pct not in (None, 0, 0.0) else "-",
                f"{active_floor:.8f}" if active_floor else "-",
            )

        except Exception as e:
            logger.warning(f"[WATCH] Snapshot failed for {sym} ({addr}): {e}", exc_info=True)


# --- Best-effort Rugcheck client shutdown helper (safe to call from main cleanup) ---
async def _shutdown_rugcheck_client(timeout: float = 5.0) -> None:
    """
    Try to call the shared shutdown helper for the Rugcheck client if it exists.
    Defensive: tolerates missing modules/atoms and will not raise.
    Supports async or sync implementations (calls sync version in a thread).
    """
    try:
        # Prefer package-style import first, then relative import
        shutdown_fn = None
        try:
            from solana_trading_bot_bundle.trading_bot.eligibility import shutdown_rugcheck_client  # type: ignore
            shutdown_fn = shutdown_rugcheck_client
        except Exception:
            try:
                from .eligibility import shutdown_rugcheck_client  # type: ignore
                shutdown_fn = shutdown_rugcheck_client
            except Exception:
                shutdown_fn = None

        if not callable(shutdown_fn):
            # nothing to do
            return

        # If it's an async function, await it; if sync, run it in a thread.
        try:
            if asyncio.iscoroutinefunction(shutdown_fn):
                await asyncio.wait_for(shutdown_fn(timeout=timeout), timeout=timeout + 0.5)
            else:
                # call sync shutdown in a thread (some impls may be sync wrappers)
                await asyncio.to_thread(shutdown_fn, timeout)
        except asyncio.TimeoutError:
            logger.debug("shutdown_rugcheck_client: timed out after %ss", timeout)
        except Exception:
            logger.debug("shutdown_rugcheck_client() raised", exc_info=True)
    except Exception:
        # ultimate fail-safe: do not let shutdown errors propagate
        try:
            logger.debug("Unexpected error while attempting rugcheck shutdown", exc_info=True)
        except Exception:
            pass
# -------------------------- Main loop --------------------------
async def main() -> None:
    logger.info("Starting trading bot")
    solana_client: Optional[AsyncClientType] = None
    try:  # ==== OUTER TRY ====
        # --- .env loading (do this first) ---
        from dotenv import load_dotenv, find_dotenv
        from pathlib import Path

        candidate_envs = [
            # preferred app-data location used by the GUI
            Path.home() / "AppData" / "Local" / "SOLOTradingBot" / ".env",
            # repo / working dir fallback
            Path.cwd() / ".env",
        ]
        loaded_from = None
        for p in candidate_envs:
            try:
                if p.exists():
                    load_dotenv(dotenv_path=str(p), override=True)
                    loaded_from = str(p)
                    break
            except Exception:
                pass
        if not loaded_from:
            discovered = find_dotenv(usecwd=True)
            if discovered:
                load_dotenv(discovered, override=True)
                loaded_from = discovered

        # ---- Load config as early as possible and validate it BEFORE further startup logging ----
        try:
            config = load_config()
        except Exception as e:
            logger.error("Failed to load configuration via load_config(): %s", e, exc_info=True)
            return

        # Defensive validation: ensure we have a dict and minimal required keys to proceed.
        if not isinstance(config, dict):
            logger.error("Configuration error: load_config() did not return a dict. Aborting startup.")
            return

        # Minimal required configuration keys (safe-fail early with explanatory messages).
        missing_reqs = []
        try:
            if not isinstance(config.get("solana"), dict) or not config["solana"].get("rpc_endpoint"):
                missing_reqs.append("solana.rpc_endpoint")
        except Exception:
            missing_reqs.append("solana.rpc_endpoint")
        try:
            if not isinstance(config.get("wallet"), dict) or not config["wallet"].get("private_key_env"):
                missing_reqs.append("wallet.private_key_env")
        except Exception:
            missing_reqs.append("wallet.private_key_env")

        if missing_reqs:
            logger.error("Configuration missing required keys: %s. Aborting startup.", ", ".join(missing_reqs))
            return

        # Initialize logging now that we have config (so subsequent logs use configured handlers/levels).
        try:
            setup_logging(config)
        except Exception as e:
            # If logging setup fails, still proceed but warn that logging might be degraded.
            logger.exception("setup_logging(config) failed; continuing with default logging. Error: %s", e)

        # quick env echo so we know what actually applied (logged after setup_logging)
        logger.info("ENV: loaded from %s", loaded_from or "<none>")
        logger.info(
            "ENV FLAGS: DRY_RUN=%s DISABLE_SEND_TX=%s JUPITER_QUOTE_ONLY=%s JUPITER_EXECUTE=%s",
            os.getenv("DRY_RUN", "0"), os.getenv("DISABLE_SEND_TX", "0"),
            os.getenv("JUPITER_QUOTE_ONLY", "0"), os.getenv("JUPITER_EXECUTE", "0"),
        )
        logger.info(
            "Birdeye: ENABLE=%s FORCE_DISABLE=%s KEY_PRESENT=%s RPS=%s cycle_cap=%s run_cap=%s",
            os.getenv("BIRDEYE_ENABLE", "0") in ("1", "true", "True"),
            os.getenv("FORCE_DISABLE_BIRDEYE", "0"),
            "yes" if os.getenv("BIRDEYE_API_KEY") else "no",
            os.getenv("BIRDEYE_RPS", "?"),
            os.getenv("BIRDEYE_MAX_CALLS_PER_CYCLE", "?"),
            os.getenv("BIRDEYE_MAX_CALLS_PER_RUN", "?"),
        )
        logger.info(
            "Dexscreener: DEX_PAGES=%s DEX_PER_PAGE=%s DEX_MAX=%s DEX_FORCE_IPV4=%s",
            os.getenv("DEX_PAGES", "?"), os.getenv("DEX_PER_PAGE", "?"),
            os.getenv("DEX_MAX", "?"), os.getenv("DEX_FORCE_IPV4", "auto"),
        )

        # --- Testing override: loosen selection thresholds ---
        TEST_LOOSEN = bool(int(os.getenv("TEST_LOOSEN_SELECTION", "0")))
        if TEST_LOOSEN:
            sel = (config.get("selection") or {})
            sel.setdefault("min_liquidity_usd", 0)
            sel.setdefault("min_volume_24h_usd", 0)
            sel.setdefault("min_age_minutes", 0)
            sel.setdefault("allow_missing_mc", True)
            sel.setdefault("allow_missing_liq", True)
            config["selection"] = sel
            logger.info("Selection thresholds loosened for testing (TEST_LOOSEN_SELECTION=1).")

        # --- SINGLE-INSTANCE GUARD (claim before any side effects) ---
        if not acquire_single_instance_or_explain(int(os.getenv("PID_STALE_AFTER_S", "180"))):
            logger.info("trading.main() exiting because acquire_single_instance_or_explain() refused to start (another instance active).")
            return

        # Start a background heartbeat that stays fresh even during long cycles
        stop_hb = asyncio.Event()
        hb_task = asyncio.create_task(heartbeat_task(stop_hb, int(os.getenv("HEARTBEAT_INTERVAL_S", "5"))))

        # Graceful shutdown integration: create an asyncio Event that signal handlers will set.
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _signal_handler(signame):
            try:
                logger.info("Received signal %s: scheduling shutdown", signame)
            except Exception:
                pass
            # set the stop event in the running loop
            try:
                loop.call_soon_threadsafe(stop_event.set)
            except Exception:
                # best-effort fallback
                try:
                    stop_event.set()
                except Exception:
                    pass

        # Register handlers for SIGINT and SIGTERM where available
        for sig in ("SIGINT", "SIGTERM"):
            try:
                signum = getattr(signal, sig)
                try:
                    loop.add_signal_handler(signum, lambda s=sig: _signal_handler(s))
                except NotImplementedError:
                    # Not supported on some platforms (Windows / uvloop); fall back to signal.signal
                    signal.signal(signum, lambda *_args, s=sig: _signal_handler(s))
            except Exception:
                # ignore inability to register signals
                pass
            
        # Persist our PID & emit an immediate heartbeat
        _write_pid_file()
        _heartbeat(throttle_s=0)
        # --- END GUARD ---

        # Emit a clear log that main started (helps debug early-return)
        logger.info("trading.main() started; PID=%s", os.getpid())

        # ---------------------------------------------------------------------
        # RPC client (create ONCE) - guarded to avoid ImportError when solana-py missing
        # and to provide a more complete stub surface for watch-only/testing.
        # ---------------------------------------------------------------------
        try:
            solana_client = _new_async_client(config["solana"]["rpc_endpoint"])
        except Exception as e:
            # If solana-py isn't installed or client construction failed, allow watch-only/dry-run.
            # Operators can force watch-only with FORCE_WATCH_ONLY=1 or by setting DRY_RUN in env/config.
            watch_only = _dry_run_on(config) or os.getenv("FORCE_WATCH_ONLY", "0").strip() == "1"
            logger.warning(
                "Failed to create real Solana AsyncClient: %s. watch_only=%s", getattr(e, "args", e), watch_only
            )
            if watch_only:
                logger.warning("Running in watch-only stub mode (no real RPC). On-chain ops disabled.")
                # Minimal, but more complete AsyncClient-like stub: implement the shapes used
                # in this module so callers don't hit AttributeError/TypeError unexpectedly.
                class _StubValue:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                    def __repr__(self):
                        return f"_StubValue({self.__dict__})"

                class _StubAsyncClient:
                    async def get_balance(self, *args, **kwargs):
                        # callers expect an object with 'value' as int or dict-like
                        return _StubValue(value=0)

                    async def get_latest_blockhash(self, *args, **kwargs):
                        # callers expect .context.slot and .value.blockhash
                        ctx = _StubValue(slot=0)
                        val = _StubValue(blockhash="0" * 44)
                        return _StubValue(context=ctx, value=val)

                    async def get_version(self, *args, **kwargs):
                        # callers accept dict or object
                        return {"solana-core": "stub"}

                    async def get_token_account_balance(self, *args, **kwargs):
                        # return object with .value.ui_amount
                        return _StubValue(value=_StubValue(ui_amount=0.0))

                    async def get_token_supply(self, *args, **kwargs):
                        # return object with .value.decimals
                        return _StubValue(value=_StubValue(decimals=9))

                    async def get_token_largest_accounts(self, *args, **kwargs):
                        # return object with .value list
                        return _StubValue(value=[])

                    # Some other helper call sites may use `close`/`is_connected`/`get_block_time`.
                    async def close(self):
                        return None

                    # defensive: attribute fallback so unknown calls raise AttributeError early with helpful message
                    def __getattr__(self, name):
                        raise AttributeError(f"_StubAsyncClient missing RPC method '{name}'; add it to the stub if required for watch-only mode")

                solana_client = _StubAsyncClient()
            else:
                logger.error(
                    "Solana RPC client required for live trading but unavailable. Install solana-py or set DRY_RUN=1 / FORCE_WATCH_ONLY=1 to run in watch-only mode."
                )
                return

        # Wallet (user requested to keep behavior - require wallet presence)
        private_key = os.getenv(config["wallet"]["private_key_env"])
        if not private_key:
            logger.error("%s not set", config["wallet"]["private_key_env"])
            return
        wallet = Keypair.from_base58_string(private_key)
        logger.info("Wallet loaded: %s", wallet.pubkey())

        # RPC quick check (resilient to response shape changes)
        try:
            bh = await solana_client.get_latest_blockhash()
            slot = getattr(getattr(bh, "context", None), "slot", None)
            blockhash = getattr(getattr(bh, "value", None), "blockhash", None)
            short_bh = (blockhash[:12] + "…") if isinstance(blockhash, str) else str(blockhash)

            ver = await solana_client.get_version()

            # Normalize version payload to something we can query safely
            v = getattr(ver, "value", ver)
            core = None
            if isinstance(v, dict):
                core = (
                    v.get("solana-core")
                    or v.get("solana_core")
                    or v.get("solanaCore")
                    or v.get("solana")
                    or v.get("version")
                )
            else:
                # Sometimes it's a typed object; probe common attribute names
                core = (
                    getattr(v, "solana_core", None)
                    or getattr(v, "solanaCore", None)
                    or getattr(v, "solana", None)
                    or getattr(v, "version", None)
                )

            logger.info(
                "RPC OK via %s - slot=%s, blockhash=%s, core=%s",
                config["solana"]["rpc_endpoint"], slot, short_bh, core or "unknown"
            )
        except Exception as e:
            logger.warning("RPC health check failed: %s", e)

        # -----------------------
        # DB init (defensive, do not raise NameError during runtime)
        # -----------------------
        try:
            if _db is None:
                logger.warning("Database module not available; proceeding without DB init.")
            else:
                # Prefer canonical names, but tolerate alternates
                init_candidates = [getattr(_db, "init_db", None),
                                   getattr(_db, "initialize_db", None),
                                   getattr(_db, "db_init", None)]
                init_fn = next((fn for fn in init_candidates if callable(fn)), None)
                if init_fn:
                    try:
                        await init_fn()
                    except Exception:
                        logger.debug("Database init function raised; continuing without DB init.", exc_info=True)
                else:
                    logger.warning("Database module does not expose an init function; continuing without DB init.")
        except Exception:
            logger.debug("Unexpected error during DB init step; continuing.", exc_info=True)

        await _ensure_schema_once(logger)

        # -----------------------
        # Load blacklist (defensive: fallback to empty set)
        # -----------------------
        try:
            if _db is None:
                logger.warning("Database module not available; using empty blacklist.")
                blacklist = set()
            else:
                load_candidates = [getattr(_db, "load_blacklist", None),
                                   getattr(_db, "db_load_blacklist", None)]
                load_fn = next((fn for fn in load_candidates if callable(fn)), None)
                if load_fn:
                    try:
                        blacklist = await load_fn()
                        # ensure a set-like result
                        if not isinstance(blacklist, (set, list, tuple)):
                            # try to coerce
                            try:
                                blacklist = set(blacklist or [])
                            except Exception:
                                blacklist = set()
                        else:
                            blacklist = set(blacklist)
                    except Exception:
                        logger.warning("Loading blacklist failed; proceeding with empty blacklist.", exc_info=True)
                        blacklist = set()
                else:
                    logger.warning("Database module missing load_blacklist; proceeding with empty blacklist.")
                    blacklist = set()
        except Exception:
            logger.warning("Unexpected error loading blacklist; proceeding with empty blacklist.", exc_info=True)
            blacklist = set()

        failure_count: Dict[str, int] = {}
        cycle_index = 0
        cycle_index = 0
        # consecutive failure protection to avoid infinite hammering on fatal errors
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "10"))

        async with aiohttp.ClientSession() as session:
            while not stop_event.is_set():
                # Stop flag check (operator-controlled file)
                try:
                    with open("bot_stop_flag.txt", "r") as f:
                        if f.read().strip() == "1":
                            logger.info("Stop flag detected, exiting trading bot")
                            break
                except FileNotFoundError:
                    pass

                # ===== PER-CYCLE START =====
                try:
                    # inside the per-cycle start (where _JUP_OFFLINE is reset)
                    # Reset Jupiter offline breaker for this cycle
                    global _JUP_OFFLINE
                    _JUP_OFFLINE = False

                    # Reset Birdeye 401 breaker for this cycle (so a transient 401 doesn't disable Birdeye for entire run)
                    global _BIRDEYE_401_SEEN
                    _BIRDEYE_401_SEEN = False

                    # -----------------------
                    # Hygiene: clear expired blacklist & optional review_blacklist
                    # This block is defensive: it will not raise NameError if a helper is missing.
                    # -----------------------
                    try:
                        clear_fn = None
                        if _db is not None:
                            clear_fn = getattr(_db, "clear_expired_blacklist", None)
                            if not callable(clear_fn):
                                # try some historical/alternate names
                                for alt in ("db_clear_expired_blacklist", "clear_blacklist_expired", "purge_expired_blacklist"):
                                    if hasattr(_db, alt) and callable(getattr(_db, alt)):
                                        clear_fn = getattr(_db, alt)
                                        break
                        if callable(clear_fn):
                            try:
                                await clear_fn(max_age_hours=24)
                            except Exception:
                                logger.debug("clear_expired_blacklist helper raised; continuing.", exc_info=True)
                        else:
                            logger.debug("DB does not provide clear_expired_blacklist; skipping blacklist hygiene this cycle.")
                    except Exception:
                        logger.debug("Unexpected error during blacklist hygiene; continuing.", exc_info=True)

                    try:
                        review_fn = getattr(_db, "review_blacklist", None) if _db is not None else None
                        if callable(review_fn):
                            try:
                                await review_fn()
                            except Exception:
                                logger.debug("review_blacklist raised; continuing.", exc_info=True)
                    except Exception:
                        logger.debug("Unexpected error invoking review_blacklist; continuing.", exc_info=True)

                    # Balance (telemetry)
                    startup_balance = await get_sol_balance(wallet, solana_client)
                    required_min = float((config.get("bot") or {}).get("required_sol", 0.0))
                    logger.info("Startup SOL balance: %.6f (min required=%.6f)", startup_balance, required_min)

                    # Source queries for trading pass
                    _discovery_cfg = (config.get("discovery") or {})
                    queries = _discovery_cfg.get("dexscreener_queries") or ["solana"]
                    queries = list(dict.fromkeys(queries))
                    logger.info("Trading: using dexscreener queries %s", queries)

                    # Build shortlist (DB first, then live)
                    eligible_tokens: List[Dict[str, Any]] = []
                    using_db_shortlist = False

                    prefer_db_shortlist = bool((config.get("trading") or {}).get("prefer_persisted_shortlist", True))
                    db_max_age_s = int((config.get("trading") or {}).get("db_shortlist_max_age_s", 300))
                    db_min_count = int((config.get("trading") or {}).get("db_shortlist_min_count", 10))

                    try:
                        eligible_tokens_from_db = await _load_persisted_shortlist_from_db( config=config, max_age_s=db_max_age_s, min_count=db_min_count )
                    except Exception as _e:
                        logger.warning("DB shortlist load failed: %s; falling back to live.", _e)
                        eligible_tokens_from_db = []

                    if prefer_db_shortlist and eligible_tokens_from_db:
                        eligible_tokens = eligible_tokens_from_db
                        using_db_shortlist = True
                        logger.info("Using %d token(s) from persisted shortlist.", len(eligible_tokens))
                    else:
                        logger.info("Persisted shortlist stale/small/empty; running live discovery.")
                        eligible_tokens = await _build_live_shortlist(
                            session=session,
                            solana_client=solana_client,
                            config=config,
                            queries=queries,
                            blacklist=blacklist,
                            failure_count=failure_count,
                        )

                    # Common shortlist processing
                    fallback_hours = int((config.get("trading") or {}).get("db_fallback_hours", 1))
                    _fallback_age_s = max(0, fallback_hours) * 3600

                    if using_db_shortlist:
                        try:
                            enriched = await _enrich_shortlist_with_signals(eligible_tokens)
                            eligible_tokens = enriched or eligible_tokens
                            if not enriched:
                                for t in eligible_tokens:
                                    t.setdefault("_enriched", False)
                        except Exception:
                            logger.warning("Signal enrichment failed on DB shortlist; attempting DB fallback.", exc_info=True)
                            db_fallback = await _load_persisted_shortlist_from_db(
                                config=config, max_age_s=_fallback_age_s, min_count=5
                            )
                            if db_fallback:
                                # Backfill categories on DB fallback too
                                eligible_tokens = _restore_or_compute_categories(db_fallback)
                                using_db_shortlist = True
                                logger.info("Pulled %d tokens from DB fallback.", len(eligible_tokens))
                            else:
                                for t in eligible_tokens:
                                    t.setdefault("_enriched", False)

                        try:
                            await persist_eligible_shortlist(eligible_tokens, prune_hours=168)
                        except Exception:
                            logger.debug("Persisting shortlist (DB path refresh) failed", exc_info=True)

                    else:
                        # Ensure categories exist before per-bucket selection (live path)
                        eligible_tokens = _restore_or_compute_categories(eligible_tokens)
                        eligible_tokens = await select_top_five_per_category(eligible_tokens)
                        try:
                            enriched = await _enrich_shortlist_with_signals(eligible_tokens)
                            if enriched:
                                eligible_tokens = enriched
                            else:
                                logger.warning("Live shortlist collapsed; attempting DB fallback (<=%ds).", _fallback_age_s)
                                db_fallback = await _load_persisted_shortlist_from_db(
                                    config=config, max_age_s=_fallback_age_s, min_count=5
                                )
                                if db_fallback:
                                    # Backfill categories on DB fallback so downstream logic/UI has buckets
                                    eligible_tokens = _restore_or_compute_categories(db_fallback)
                                    using_db_shortlist = True
                                    logger.info("Pulled %d tokens from DB fallback.", len(eligible_tokens))
                                else:
                                    # Backfill again before the fallback selector
                                    eligible_tokens = _restore_or_compute_categories(eligible_tokens)
                                    eligible_tokens = await select_top_five_per_category(eligible_tokens)
                                    for t in eligible_tokens:
                                        t.setdefault("_enriched", False)
                        except Exception:
                            logger.warning("Signal enrichment failed on live shortlist; attempting DB fallback.", exc_info=True)
                            db_fallback = await _load_persisted_shortlist_from_db(
                                config=config, max_age_s=_fallback_age_s, min_count=5
                            )
                            if db_fallback:
                                # Backfill categories on DB fallback too
                                eligible_tokens = _restore_or_compute_categories(db_fallback)
                                using_db_shortlist = True
                                logger.info("Pulled %d tokens from DB fallback.", len(eligible_tokens))
                            else:
                                for t in eligible_tokens:
                                    t.setdefault("_enriched", False)
                                    
                    _apply_tech_tiebreaker(eligible_tokens, config, logger=logger)
                    logger.info("Found %d eligible tokens", len(eligible_tokens))
                    log_scoring_telemetry(eligible_tokens, where="shortlist")

                    if using_db_shortlist and not eligible_tokens:
                        logger.info("DB shortlist yielded zero; forcing live discovery once.")
                        eligible_tokens = await _fetch_live_shortlist_once(
                            session=session, solana_client=solana_client, config=config,
                            queries=queries, blacklist=blacklist, failure_count=failure_count, logger=logger
                        )

                    # Mode flags
                    tsec = (config.get("trading") or {})
                    bsec = (config.get("bot") or {})

                    dry = _dry_run_on(config)
                    send_disabled = _skip_onchain_send(config)
                    simulate = bool(tsec.get("simulate", bsec.get("simulate", False)))

                    env_dry = os.getenv("DRY_RUN", "0") == "1"
                    env_disable_send = os.getenv("DISABLE_SEND_TX", "0") == "1"
                    env_jup_quote_only = os.getenv("JUPITER_QUOTE_ONLY", "0") == "1"
                    env_jup_execute = os.getenv("JUPITER_EXECUTE", "0") == "1"

                    dry = dry or env_dry
                    simulate = simulate or env_dry or env_jup_quote_only
                    send_disabled = send_disabled or env_disable_send or env_jup_quote_only
                    if env_jup_execute and not env_jup_quote_only:
                        simulate = False

                    _dry_run = bool(locals().get("dry_run", locals().get("dry", False)))
                    _send_disabled = bool(locals().get("send_disabled", False))
                    _simulate = bool(locals().get("simulate", False))

                    try:
                        _has_balance = await has_min_balance(wallet, config, session=session)
                    except Exception as _e:
                        logger.warning("Balance check failed; treating as no-balance. (%s)", _e)
                        _has_balance = False

                    _enabled = bool((config.get("trading") or {}).get("enabled", True))
                    _send_tx = not _send_disabled
                    can_trade = True if (_dry_run or _send_disabled or _simulate) else bool(_has_balance)

                    logger.info(
                        "TRADING-CYCLE: shortlist=%d enabled=%s dry_run=%s simulate=%s send_tx=%s can_trade=%s",
                        len(eligible_tokens), _enabled, _dry_run, _simulate, _send_tx, can_trade
                    )

                    # ---------------- Buys ----------------
                    candidates = eligible_tokens[:10]
                    logger.info("TRADING-CYCLE: entering buy loop with %d candidate(s)", len(candidates))
                    try:
                        if not candidates:
                            logger.info("No buy candidates this cycle")
                        else:
                            WSOL = "So11111111111111111111111111111111111111112"
                            USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
                            user_pubkey = str(wallet.pubkey())
                            # Use module-level constant if present, otherwise local
                            LAMPORTS_PER_SOL_LOCAL = globals().get("LAMPORTS_PER_SOL", 1_000_000_000)

                            # Local helpers to read config/env with fallbacks
                            def _cfg_int(cfg: dict, key: str, env: str, default: int) -> int:
                                try:
                                    v = cfg.get(key)
                                    if v is None:
                                        v = int(os.getenv(env, str(default)))
                                    return int(v)
                                except Exception:
                                    try:
                                        return int(os.getenv(env, str(default)))
                                    except Exception:
                                        return int(default)

                            def _cfg_float(cfg: dict, key: str, env: str, default: float) -> float:
                                try:
                                    v = cfg.get(key)
                                    if v is None:
                                        v = os.getenv(env, None)
                                        if v is None:
                                            return float(default)
                                    return float(v)
                                except Exception:
                                    try:
                                        return float(os.getenv(env, str(default)))
                                    except Exception:
                                        return float(default)

                            def _cfg_bool(cfg: dict, key: str, env: str, default: bool = False) -> bool:
                                try:
                                    if key in cfg and cfg.get(key) is not None:
                                        v = cfg.get(key)
                                        # accept booleans or truthy strings/numbers
                                        if isinstance(v, bool):
                                            return v
                                        try:
                                            return str(v).strip().lower() in ("1", "true", "yes", "on", "y")
                                        except Exception:
                                            return bool(v)
                                    ev = os.getenv(env, None)
                                    if ev is not None:
                                        return str(ev).strip().lower() in ("1", "true", "yes", "on", "y")
                                    return bool(default)
                                except Exception:
                                    ev = os.getenv(env, "0")
                                    return str(ev).strip().lower() in ("1", "true", "yes", "on", "y")

                            # Resolve trading / bot sections once
                            tsec = (config.get("trading") or {})
                            bsec = (config.get("bot") or {})

                            # Candidate limit (config.yaml key trading.candidate_limit or env CANDIDATE_LIMIT)
                            candidate_limit = _cfg_int(tsec, "candidate_limit", "CANDIDATE_LIMIT", 10)
                            candidates = candidates[:max(1, candidate_limit)]

                            # Optional per-candidate reason logging at INFO level when enabled
                            log_candidate_reasons_info = _cfg_bool(tsec, "log_candidate_reasons_info", "LOG_CANDIDATE_REASONS_INFO", False)

                            def _log_skip(symbol: str, token_addr: str, reason: str, log_at_info_if_enabled: bool = True):
                                """
                                Log the reason a candidate was skipped. If log_candidate_reasons_info is True,
                                emit at INFO level, otherwise DEBUG.
                                """
                                msg = f"Skipping {symbol} ({token_addr}): {reason}"
                                try:
                                    if log_candidate_reasons_info and log_at_info_if_enabled:
                                        logger.info(msg)
                                    else:
                                        logger.debug(msg)
                                except Exception:
                                    # Fallback safe logging
                                    try:
                                        logger.info(msg)
                                    except Exception:
                                        pass

                            # Resolve SOL price (try primary provider, then cache, then Jupiter, then env fallback)
                            sol_price: Optional[float] = None
                            try:
                                sol_price = await get_sol_price(session)
                                if sol_price and sol_price > 0:
                                    try:
                                        price_cache["SOLUSD"] = float(sol_price)
                                    except Exception:
                                        pass
                            except Exception:
                                sol_price = None

                            if not sol_price:
                                try:
                                    cached = float(price_cache.get("SOLUSD") or 0.0)
                                    sol_price = cached if cached > 0 else None
                                except Exception:
                                    sol_price = None

                            if not sol_price:
                                try:
                                    quote, qerr = await _safe_get_jupiter_quote(
                                        input_mint=WSOL,
                                        output_mint=USDC,
                                        amount=LAMPORTS_PER_SOL_LOCAL,  # 1 SOL
                                        user_pubkey=user_pubkey,
                                        session=session,
                                        slippage_bps=50,
                                    )
                                    if quote and not qerr:
                                        usdc_out_raw = float(quote.get("outAmount") or 0.0)  # USDC 6dp
                                        derived = usdc_out_raw / 1_000_000 if usdc_out_raw > 0 else 0.0
                                        if derived > 0:
                                            sol_price = derived
                                            try:
                                                price_cache["SOLUSD"] = float(sol_price)
                                            except Exception:
                                                pass
                                except Exception:
                                    pass

                            if not sol_price:
                                try:
                                    env_fallback = os.getenv("SOL_FALLBACK_PRICE")
                                    sol_price = float(env_fallback) if env_fallback else None
                                except Exception:
                                    sol_price = None

                            if not sol_price:
                                logger.warning("Unable to determine SOL price; skipping buys this cycle.")
                            else:
                                # Normalize SOL→USD price once for the cycle and store in cache for metric helpers
                                try:
                                    sol_price_usd = float(sol_price)
                                    if sol_price_usd <= 0:
                                        sol_price_usd = float(price_cache.get("SOLUSD") or 0.0)
                                except Exception:
                                    sol_price_usd = float(price_cache.get("SOLUSD") or 0.0)
                                try:
                                    price_cache["SOLUSD"] = float(sol_price_usd or 0.0)
                                except Exception:
                                    pass

                                # --- Sizing ---
                                wallet_balance = await get_sol_balance(wallet, solana_client)

                                buy_amount_sol = await _resolve_buy_amount_sol(
                                    get_buy_amount_fn=get_buy_amount,
                                    config=config,
                                    sol_price=sol_price,
                                    wallet_balance=wallet_balance,
                                    token=None,  # pass a token dict if you tilt sizing per-asset
                                )

                                # Optional: do not size above available balance (minus a dust buffer)
                                try:
                                    max_afford = max(0.0, float(wallet_balance) - float((tsec.get("wallet_reserve_sol") or bsec.get("wallet_reserve_sol") or 0.0003)))
                                    if buy_amount_sol > max_afford:
                                        logger.info(
                                            "Clamping buy amount from %.6f SOL to wallet max %.6f SOL",
                                            buy_amount_sol,
                                            max_afford,
                                        )
                                        buy_amount_sol = max_afford
                                except Exception:
                                    pass

                                # Bail early if buy amount non-positive or wallet practically empty
                                if not buy_amount_sol or float(wallet_balance or 0.0) <= 0.0:
                                    logger.info("No available balance or zero buy_amount_sol (%.6f); skipping buys this cycle.", float(buy_amount_sol or 0.0))
                                else:
                                    # --- Compute slippage once for the cycle ---
                                    slippage_bps = _clamp_bps(int(tsec.get("slippage_bps", 150)))

                                    # rank/tilt candidates with signals (best-effort)
                                    try:
                                        enriched = await _enrich_shortlist_with_signals(candidates)
                                        if enriched:
                                            candidates = enriched
                                        else:
                                            for t in candidates:
                                                t.setdefault("_enriched", False)
                                            logger.warning("Candidate enrichment returned empty; keeping un-enriched candidates.")
                                    except Exception:
                                        logger.debug("Candidate enrichment failed; continuing with raw candidates.", exc_info=True)

                                    _apply_tech_tiebreaker(candidates, config, logger=logger)

                                    # Per-cycle limits and tuning from config/env
                                    max_buys = _cfg_int(tsec, "max_buys_per_cycle", "MAX_BUYS_PER_CYCLE", 5)
                                    min_buy_sol = _cfg_float(tsec, "min_buy_sol", "MIN_BUY_SOL", 0.01)
                                    buy_cooldown = _cfg_float(tsec, "buy_cooldown_s", "BUY_COOLDOWN_S", 2.0)

                                    # Use a local wallet_balance snapshot so we don't overcommit within this cycle
                                    local_balance = float(wallet_balance or 0.0)
                                    buys_done = 0  # ensure local counter exists

                                    for tok in candidates:
                                        if buys_done >= max_buys:
                                            break

                                        try:
                                            token_addr = tok.get("address")
                                            symbol = tok.get("symbol") or "UNKNOWN"
                                            if not token_addr:
                                                _log_skip(symbol, str(tok), "no address present")
                                                continue

                                            # Skip core stable-like tokens (safety)
                                            if _is_core_stable_like(tok):
                                                _log_skip(symbol, token_addr, "core stable-like / wrapper")
                                                continue

                                            # Skip if blacklisted
                                            try:
                                                if token_addr in blacklist:
                                                    _log_skip(symbol, token_addr, "token is blacklisted")
                                                    continue
                                            except Exception:
                                                pass

                                            # Per-token sizing override
                                            per_token_buy_amount_sol = buy_amount_sol
                                            if (config.get("bot") or {}).get("per_asset_sizing", True):
                                                try:
                                                    per_token_buy_amount_sol = await _resolve_buy_amount_sol(
                                                        get_buy_amount_fn=get_buy_amount,
                                                        config=config,
                                                        sol_price=sol_price,
                                                        wallet_balance=local_balance,
                                                        token=tok,
                                                    )
                                                except Exception as e:
                                                    logger.warning(
                                                        "Per-asset sizing failed for %s: %s. Falling back to cycle size.",
                                                        tok.get("symbol") or tok.get("address") or "unknown",
                                                        e,
                                                    )
                                                    per_token_buy_amount_sol = buy_amount_sol

                                            # Sanity checks: positive and >= min
                                            try:
                                                per_token_buy_amount_sol = float(per_token_buy_amount_sol or 0.0)
                                            except Exception:
                                                per_token_buy_amount_sol = 0.0

                                            if per_token_buy_amount_sol < min_buy_sol:
                                                _log_skip(symbol, token_addr, f"per-token buy amount {per_token_buy_amount_sol:.6f} < min_buy_sol {min_buy_sol:.6f}")
                                                continue

                                            # Affordability check using local snapshot
                                            reserve_buffer = float(tsec.get("wallet_reserve_sol", bsec.get("wallet_reserve_sol", 0.0003)))
                                            max_affordable = max(0.0, local_balance - reserve_buffer)
                                            if per_token_buy_amount_sol > max_affordable:
                                                _log_skip(symbol, token_addr, f"insufficient funds: need {per_token_buy_amount_sol:.6f}, available {max_affordable:.6f}")
                                                continue

                                            logger.info("Attempting BUY %s (%s) amount=%.6f SOL (available=%.6f)", symbol, token_addr, per_token_buy_amount_sol, local_balance)

                                            # Execute buy - real_on_chain_buy handles dry-run/send-disabled internally
                                            try:
                                                txid = await real_on_chain_buy(
                                                    token=tok,
                                                    buy_amount=per_token_buy_amount_sol,
                                                    wallet=wallet,
                                                    solana_client=solana_client,
                                                    session=session,
                                                    blacklist=blacklist,
                                                    failure_count=failure_count,
                                                    config=config,
                                                )
                                            except Exception as e:
                                                logger.warning("real_on_chain_buy raised for %s (%s): %s", symbol, token_addr, e, exc_info=True)
                                                txid = None

                                            if txid:
                                                buys_done += 1
                                                # conservative local balance update
                                                try:
                                                    local_balance = max(0.0, local_balance - float(per_token_buy_amount_sol))
                                                except Exception:
                                                    pass
                                                logger.info("BUY SUCCESS %s (%s) txid=%s buys_done=%d/%d", symbol, token_addr, txid, buys_done, max_buys)
                                            else:
                                                logger.info("BUY attempt returned no txid for %s (%s)", symbol, token_addr)

                                            # brief cooldown between buys (reduce rate-limit pressure)
                                            try:
                                                await asyncio.sleep(float(buy_cooldown))
                                            except Exception:
                                                pass

                                        except Exception as e:
                                            logger.warning("Error handling buy candidate %s: %s", tok.get("address"), e, exc_info=True)
                                            continue

                                    logger.info("Buy loop completed: buys_done=%d (max=%d)", buys_done, max_buys)

                    except Exception as e:
                        logger.error("BUY-LOOP error: %s", e, exc_info=True)

                    # ---------------- Sells ----------------
                    try:
                        # Resolve get_open_positions safely:
                        # Prefer the DB module implementation when available so the no-op
                        # global stub (installed earlier as a safe default) does not
                        # accidentally shadow a real implementation.
                        _get_open_positions_fn = getattr(_db, "get_open_positions", None) if _db is not None else None
                        # If DB module doesn't provide it, fall back to a global binding (legacy/injected)
                        if not callable(_get_open_positions_fn):
                            _get_open_positions_fn = globals().get("get_open_positions") if "get_open_positions" in globals() else None

                        # If the selected callable is the noop stub, emit a warning so operators can see the mismatch.
                        try:
                            if callable(_get_open_positions_fn) and getattr(_get_open_positions_fn, "__name__", "") == "_noop_async":
                                logger.warning("Using noop get_open_positions() stub; sells will see no open positions. _db.get_open_positions exists=%s", hasattr(_db, "get_open_positions") if _db is not None else False)
                        except Exception:
                            pass

                        # Fetch open positions in a protected manner
                        try:
                            open_positions = await _get_open_positions_fn() if callable(_get_open_positions_fn) else []
                        except Exception as e:
                            logger.warning("Failed to fetch open positions: %s", e)
                            open_positions = []

                        # Observability: log how many positions we got and which callable was used
                        try:
                            cnt = len(open_positions) if isinstance(open_positions, (list, tuple)) else (0 if open_positions is None else "non-list")
                            logger.info("Sell loop: fetched open_positions via %s -> count=%s", getattr(_get_open_positions_fn, "__name__", repr(_get_open_positions_fn)), cnt)
                        except Exception:
                            pass

                        if not open_positions:
                            logger.debug("No open positions to process for sells this cycle.")
                        else:
                            # Resolve cooldown (defensive parsing)
                            try:
                                cooldown_sec = int(float((config.get("bot") or {}).get("cooldown_seconds", 3)))
                            except Exception:
                                cooldown_sec = 3

                            # Periodic snapshot (protected)
                            try:
                                if cycle_index % 5 == 0:
                                    await _log_positions_snapshot(open_positions, session, solana_client, config)
                            except Exception:
                                logger.debug("positions snapshot failed", exc_info=True)

                            # Helper to persist a paper-sell (defensive, uses whichever persistence is available)                       
                            async def _safe_update_token_record_for_sell(token: dict, sell_txid: str, sell_time: int, is_trading: bool = False) -> bool:
                                """
                                Persist a sell record tolerant to variations in update_token_record signature.
                                Tries keyword call first, then tries a few fallback signatures.
                                Returns True if a persistence call was made successfully, False otherwise.
                                """
                                # Prefer top-level update_token_record then _db.update_token_record
                                fn = globals().get("update_token_record")
                                if not callable(fn):
                                    fn = getattr(_db, "update_token_record", None)
                                if not callable(fn):
                                    return False

                                # Common kwargs used elsewhere in codebase
                                kwargs = {
                                    "token": token,
                                    "sell_txid": sell_txid,
                                    "sell_time": sell_time,
                                    "is_trading": is_trading,
                                }

                                try:
                                    # Try the keyword form first
                                    await fn(**kwargs)
                                    return True
                                except TypeError:
                                    # Try a few fallback permutations / legacy names
                                    try:
                                        await fn(token, sell_txid, sell_time, is_trading)
                                        return True
                                    except Exception:
                                        try:
                                            # some impls expect 'sell_time' as 'sell_ts' or 'sell_timestamp'
                                            alt_kwargs = dict(kwargs)
                                            alt_kwargs.pop("is_trading", None)
                                            for alt_time_key in ("sell_ts", "sell_timestamp"):
                                                try:
                                                    await fn(token=token, sell_txid=sell_txid, **{alt_time_key: sell_time})
                                                    return True
                                                except Exception:
                                                    continue
                                        except Exception:
                                            pass
                                except Exception:
                                    logger.debug("Persisting sell record failed for %s", token.get("address"), exc_info=True)

                                return False

                            # Iterate open positions sequentially (safe, clear behavior)
                            for pos in open_positions:
                                try:
                                    addr = (pos.get("address") if isinstance(pos, dict) else None) or ""
                                    if not addr:
                                        logger.debug("Skipping position with no address: %s", pos)
                                        await asyncio.sleep(0.25)
                                        continue

                                    sym = (pos.get("symbol") if isinstance(pos, dict) else None) or "UNKNOWN"
                                    token_data = {"address": addr, "symbol": sym}

                                    # Dry-run / simulate / send-disabled path: don't call real sell; persist a paper-sell                                    
                                    
                                    try:
                                        now_ts = int(time.time())

                                        # fetch current price and token qty (best-effort, tolerant)
                                        try:
                                            current_price = await get_token_price_in_sol(addr, session, price_cache, {"address": addr, "symbol": sym})
                                        except Exception:
                                            current_price = None

                                        try:
                                            token_qty = await get_token_balance(wallet, addr, solana_client, {"address": addr, "symbol": sym})
                                            # get_token_balance returns float token units; if zero or error, set None
                                            if token_qty is None or token_qty <= 0:
                                                token_qty = None
                                        except Exception:
                                            token_qty = None

                                        # If we are in dry-run / disabled-send, record a paper-sell and skip the live sell call
                                        if _skip_onchain_send(config):
                                            txid = _dry_run_txid("sell")

                                            _log_trade_event(
                                                "DRYRUN_SELL",
                                                token=sym,
                                                address=addr,
                                                amount_lamports=None,
                                                reason="MANUAL_SHORTCIRCUIT",
                                                txid=txid
                                            )

                                            # Record metrics for dry-run sell (best-effort)
                                            try:
                                                _record_metric_fill(
                                                    token_addr=addr,
                                                    symbol=sym,
                                                    side="SELL",
                                                    quote=None,
                                                    buy_amount_sol=None,
                                                    token_price_sol=(float(current_price) if current_price is not None else None),
                                                    txid=txid,
                                                    simulated=True,
                                                    source="paper",
                                                    reason="SHORTCIRCUIT",
                                                )
                                            except Exception:
                                                logger.debug("Dry-run metric recording failed (ignored)", exc_info=True)

                                            # Persist DRYRUN sell into trade_history (best-effort) BEFORE marking sold
                                            try:
                                                await _persist_dryrun_sell_into_trade_history(
                                                    token_address=addr,
                                                    symbol=sym,
                                                    sell_price=(float(current_price) if current_price is not None else None),
                                                    sell_amount_tokens=(float(token_qty) if token_qty is not None else None),
                                                    sell_txid=txid,
                                                    sell_time=now_ts,
                                                    config=config,
                                                )
                                            except Exception:
                                                logger.debug("Persist DRYRUN sell helper raised (ignored)", exc_info=True)

                                            # Optionally persist a token-level sell record if update helper available
                                            try:
                                                await _safe_update_token_record_for_sell({"address": addr, "symbol": sym}, str(txid), now_ts, is_trading=False)
                                            except Exception:
                                                pass

                                            # cooldown / continue to next position
                                            try:
                                                await asyncio.sleep(cooldown_sec)
                                            except Exception:
                                                pass
                                            continue

                                    except Exception:
                                        logger.debug("Persist DRYRUN sell helper raised (ignored)", exc_info=True)

                                    # Live sell path
                                    try:
                                        txid = await real_on_chain_sell(
                                            token_data, wallet, solana_client, session, blacklist, failure_count, config
                                        )
                                    except Exception as e:
                                        logger.warning("real_on_chain_sell raised for %s (%s): %s", sym, addr, e, exc_info=True)
                                        txid = None

                                    if txid:
                                        logger.info("LIVE-SELL %s (%s) txid=%s", sym, addr, txid)
                                    else:
                                        logger.warning("Sell returned no txid for %s (%s)", sym, addr)

                                    try:
                                        await asyncio.sleep(cooldown_sec)
                                    except Exception:
                                        pass

                                except Exception as e:
                                    log_error_with_stacktrace(
                                        f"Error processing sell for {pos.get('symbol','UNKNOWN')} ({pos.get('address')})", e
                                    )
                                    try:
                                        await asyncio.sleep(0.5)
                                    except Exception:
                                        pass

                    except Exception as e:
                        logger.error("SELL-LOOP error: %s", e, exc_info=True)
                        
                    # ===== PER-CYCLE END =====

                    # ---- end-of-cycle sleep & bookkeeping ----
                    interval: int = int(float((config.get("bot") or {}).get("cycle_interval", 30)))
                    jitter_s: float = float(os.getenv("CYCLE_SLEEP_JITTER_S", "5"))
                    jitter: float = float(random.uniform(0.0, max(0.0, jitter_s)))
                    sleep_for: float = float(max(0.0, float(interval))) + jitter

                    logger.info(
                        "Completed trading cycle, waiting %.1f seconds (interval=%d jitter=%.2f)",
                        sleep_for,
                        interval,
                        jitter,
                    )
                    _heartbeat()
                    await asyncio.sleep(float(sleep_for))

                    # Successful end of cycle -> reset consecutive failures and increment cycle index
                    try:
                        consecutive_failures = 0
                    except Exception:
                        consecutive_failures = 0
                    # Increment cycle index only after a successful cycle completion
                    cycle_index += 1

                except Exception as e:
                    # Keep the loop alive on per-cycle failures, but respect external stop requests.
                    # Also track consecutive failures to avoid endless hammering of external services
                    try:
                        logger.error("Error in trading cycle: %s", e, exc_info=True)
                    except Exception:
                        pass

                    # increment consecutive failure counter and log progress
                    try:
                        consecutive_failures = (int(consecutive_failures) + 1) if isinstance(consecutive_failures, int) else 1
                        logger.warning("Consecutive cycle failures: %d/%d", consecutive_failures, MAX_CONSECUTIVE_FAILURES)
                    except Exception:
                        try:
                            consecutive_failures = 1
                        except Exception:
                            consecutive_failures = 1

                    # If stop was requested externally, break immediately
                    try:
                        if stop_event.is_set():
                            break
                    except Exception:
                        pass

                    # If too many consecutive failures, escalate and exit cleanly to avoid hammering
                    try:
                        if MAX_CONSECUTIVE_FAILURES and consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            logger.error(
                                "Exceeded max consecutive failures (%d). Aborting to avoid hammering external services.",
                                MAX_CONSECUTIVE_FAILURES,
                            )
                            try:
                                stop_event.set()
                            except Exception:
                                pass
                            break
                    except Exception:
                        pass

                    # Otherwise wait a bit (small randomized sleep) before retrying
                    try:
                        await asyncio.sleep(10 + float(random.uniform(0.0, 2.0)))
                    except Exception:
                        pass
                    continue
                # ===== PER-CYCLE END =====
                    
    except Exception as e:  # <-- outer except: fatal error in main loop
        logger.error("Fatal error in trading.main(): %s", e, exc_info=True)

    finally:
        # Stop heartbeat task and cleanup PID/heartbeat files
        try:
            if 'stop_hb' in locals():
                stop_hb.set()
            if 'hb_task' in locals():
                hb_task.cancel()
                try:
                    await asyncio.gather(hb_task, return_exceptions=True)
                except Exception:
                    logger.debug("Heartbeat gather cancel raised", exc_info=True)
        except Exception:
            logger.debug("Heartbeat task shutdown failed", exc_info=True)

        # final heartbeat is nice-to-have
        try:
            _heartbeat(throttle_s=0)
        except Exception:
            logger.debug("Final heartbeat failed", exc_info=True)

        # Best-effort: shutdown any shared clients created by fetching.py (Birdeye / Rugcheck clients)
        try:
            fn = globals().get("shutdown_shared_clients")
            if callable(fn):
                # If it's an async function, await it; if sync, run in a thread.
                try:
                    if asyncio.iscoroutinefunction(fn):
                        await fn(timeout=5.0)
                    else:
                        # run sync shutdown in a worker thread so we don't block the event loop
                        await asyncio.to_thread(fn, timeout=5.0)
                except TypeError:
                    # Some callables may not accept the timeout kwarg; try calling without it.
                    try:
                        if asyncio.iscoroutinefunction(fn):
                            await fn()
                        else:
                            await asyncio.to_thread(fn)
                    except Exception:
                        logger.debug("shutdown_shared_clients call raised", exc_info=True)
            else:
                logger.debug("shutdown_shared_clients not available; skipping fetching client shutdown.")
        except Exception:
            logger.debug("shutdown_shared_clients failed or not available", exc_info=True)

        # Close RPC client if opened
        try:
            if solana_client is not None:
                await solana_client.close()
        except Exception:
            logger.debug("solana_client.close() failed", exc_info=True)

        # Best-effort: shutdown shared Rugcheck client to avoid leaked resources (aiohttp/tasks).
        try:
            await _shutdown_rugcheck_client(timeout=5.0)
        except Exception:
            logger.debug("Rugcheck client shutdown attempt failed", exc_info=True)
            
        # Close module-shared DB connection (async) so aiosqlite worker threads can exit cleanly.
        try:
            if db is not None:
                try:
                    await db.close_shared_db()
                except Exception:
                    logger.debug("Failed to close shared DB connection", exc_info=True)
        except Exception:
            logger.debug("Error while closing shared DB connection", exc_info=True)

        # Release PID / single-instance guard (best-effort)
        try:
            try:
                release_single_instance()
            except Exception:
                logger.debug("release_single_instance failed", exc_info=True)
        except Exception:
            logger.debug("Error during release_single_instance", exc_info=True)    

if __name__ == "__main__":
    asyncio.run(main())
