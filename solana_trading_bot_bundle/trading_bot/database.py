# solana_trading_bot_bundle/trading_bot/database.py
from __future__ import annotations

import aiosqlite
import logging
import json
import time
import re
import os
import traceback
import inspect
import asyncio
import sys
from datetime import datetime
from typing import Dict, Set, Optional, List, Tuple, Any, Sequence
from functools import lru_cache
from contextlib import asynccontextmanager

from solana_trading_bot_bundle.common.constants import (
    token_cache_path,  # fallback if not set in config.yaml
)

from .canonical import extract_canonical_mint

# Defensive import of small helpers from .utils_exec to avoid circular import errors
# at module import time when trading.py imports database early.
try:
    from .utils_exec import (
        load_config,
        custom_json_encoder,
        WHITELISTED_TOKENS,
    )
except Exception:
    # Provide safe, conservative fallbacks so database.py can be imported
    # even if utils cannot be resolved right now (circular import / startup order).
    # These fallbacks mimic the minimal contract used by database.py:
    # - load_config() -> returns empty dict
    # - custom_json_encoder(o) -> best-effort JSON encoder
    # - WHITELISTED_TOKENS -> empty mapping/set
    def load_config(*a, **k):
        # Real load_config will be available later when utils imports finish.
        return {}

    def custom_json_encoder(o):
        # Best-effort serialiser for unusual objects used only for logging/payloads.
        # Try common strategies then fall back to str().
        try:
            # dataclasses / objects with __dict__
            if hasattr(o, "__dict__"):
                return o.__dict__
            # objects implementing isoformat (datetimes)
            if hasattr(o, "isoformat"):
                return o.isoformat()
            # sequences/dicts handled by json by default; let json try them
            return str(o)
        except Exception:
            return str(o)

    # Set WHITELISTED_TOKENS to empty mapping to avoid KeyError usage patterns.
    WHITELISTED_TOKENS = {}

logger = logging.getLogger("TradingBot")


# =========================
# Connection / Path helpers
# =================

class _ConnCtx:
    def __init__(self, path: str):
        self._path = path
        self._db: Optional[aiosqlite.Connection] = None

    async def __aenter__(self) -> aiosqlite.Connection:
        self._db = await aiosqlite.connect(self._path)
        try:
            await self._db.execute("PRAGMA journal_mode=WAL;")
            await self._db.execute("PRAGMA busy_timeout=30000;")
            await self._db.execute("PRAGMA synchronous=NORMAL;")
            await self._db.execute("PRAGMA foreign_keys=ON;")
            self._db.row_factory = aiosqlite.Row
            # Ensure schema exists on every connection (idempotent)
            await _ensure_core_schema(self._db)
        except Exception:
            # Non-fatal: keep connection usable
            logger.debug(
                "Schema bootstrap on enter encountered a non-fatal issue:\n%s",
                traceback.format_exc()
            )
        return self._db

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._db is not None:
                await self._db.close()
        finally:
            self._db = None


def _default_token_cache_path_value() -> str:
    """
    Safely obtain the default token cache path.
    `token_cache_path` may be either a string/Path or a function returning one.
    Also normalize JSON defaults to SQLite.
    """
    try:
        val = token_cache_path() if callable(token_cache_path) else token_cache_path
    except TypeError:
        val = token_cache_path

    p = os.path.abspath(os.path.expanduser(os.path.expandvars(str(val))))
    # Sanitize cross-user Windows profile paths (if any)
    p = _sanitize_win_user_profile_path(p)
    p = os.path.normpath(p)
    if p.lower().endswith(".json"):
        p = os.path.join(os.path.dirname(p), "tokens.sqlite3")
    return p


_WARNED_WINPATH = False
def _warn_once_winpath(msg: str) -> None:
    global _WARNED_WINPATH
    if not _WARNED_WINPATH:
        logger.warning(msg)
        _WARNED_WINPATH = True


def _looks_windows_path(s: str) -> bool:
    return ("\\" in s) and (":" in s)


def _sanitize_win_user_profile_path(raw_path: str) -> str:
    r"""
    If raw_path looks like a Windows per-user AppData path for a DIFFERENT user,
    rewrite it to use the current process' LOCALAPPDATA base. Otherwise return raw_path unchanged.

    Examples rewritten:
      C:\Users\johnk\AppData\Local\SOLOTradingBot -> <LOCALAPPDATA>\SOLOTradingBot

    This helps when a config was authored on another machine/account and points at
    an absolute profile path that the current user cannot access.
    """
    try:
        if os.name != "nt" or not raw_path:
            return raw_path
        rp = str(raw_path)
        low = rp.lower()
        # look for the standard user profile AppData\Local anchor
        marker = "\\appdata\\local"
        idx = low.find(marker)
        if idx == -1:
            return raw_path
        # If the prefix before \AppData\Local is not our current LOCALAPPDATA, rewrite
        cur_local = (os.getenv("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")).rstrip("\\/")
        if low.startswith(cur_local.lower()):
            return raw_path  # already pointing at this user's LocalAppData
        # Compose new path by taking the suffix after \AppData\Local and joining to current LOCALAPPDATA
        suffix = rp[idx + len(marker):].lstrip("\\/")
        new_path = os.path.join(cur_local, suffix) if suffix else cur_local
        return new_path
    except Exception:
        return raw_path


def _resolve_db_path() -> str:
    default_dbp = _default_token_cache_path_value()
    # Ensure default also sanitized (defensive)
    default_dbp = _sanitize_win_user_profile_path(default_dbp)
    default_dbp = os.path.normpath(default_dbp)

    try:
        import yaml  # optional
        cfg_path = os.path.join(os.getcwd(), "config.yaml")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    db_section = (cfg.get("database") or {}) if isinstance(cfg, dict) else {}
    cfg_dbp = db_section.get("token_cache_path") or db_section.get("path")

    if cfg_dbp:
        candidate = os.path.expanduser(os.path.expandvars(str(cfg_dbp)))
        # sanitize Windows-profile-style paths to the current LOCALAPPDATA if needed
        candidate = _sanitize_win_user_profile_path(candidate)
        candidate = os.path.normpath(candidate)
        if os.name != "nt" and _looks_windows_path(candidate):
            _warn_once_winpath(
                f"Configured token_cache_path looks Windows-style on this OS ({candidate}). "
                "Ignoring and using default AppData path instead."
            )
            dbp = default_dbp
        else:
            dbp = os.path.abspath(candidate)
    else:
        dbp = default_dbp

    base = os.path.basename(dbp)
    if (not os.path.splitext(base)[1]) or dbp.endswith(("/", "\\")):
        dbp = os.path.join(dbp, "tokens.sqlite3") if os.path.isdir(dbp) else os.path.join(os.path.dirname(dbp), "tokens.sqlite3")

    if dbp.lower().endswith(".json"):
        logger.info("Upgrading legacy JSON DB path to SQLite: %s -> tokens.sqlite3", dbp)
        dbp = os.path.join(os.path.dirname(dbp), "tokens.sqlite3")

    # -------------------------
    # Diagnostic & defensive fallback
    # Purpose: log useful diagnostics and detect/write-test the parent dir.
    # If the write test fails, fall back to a safe per-user default DB path.
    # -------------------------
    try:
        logger.info("Resolved DB path (pre-create): %s", dbp)
        # Attempt to report process/user info (best-effort)
        try:
            proc_user = os.getlogin()
        except Exception:
            proc_user = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
        logger.info("Process user: %s", proc_user)
        logger.info("LOCALAPPDATA: %s", os.environ.get("LOCALAPPDATA"))
        logger.info("APPDATA: %s", os.environ.get("APPDATA"))

        parent = os.path.dirname(dbp) or os.getcwd()
        logger.info("DB parent directory: %s (exists=%s, isdir=%s)", parent, os.path.exists(parent), os.path.isdir(parent))

        # Try to create parent dir idempotently
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create DB parent dir %s: %s", parent, e)

        # Quick write test to parent directory
        try:
            testfn = f".db_write_test_{os.getpid()}_{int(time.time())}"
            testpath = os.path.join(parent, testfn)
            with open(testpath, "w", encoding="utf-8") as tf:
                tf.write("ok")
            try:
                os.remove(testpath)
            except Exception:
                pass
            logger.info("Write test to DB parent succeeded: %s", parent)
        except Exception as write_exc:
            logger.error("Write test to DB parent FAILED for %s: %s", parent, write_exc, exc_info=True)
            # Fallback: try per-user default app dir
            try:
                fallback = _default_token_cache_path_value()
                fallback = _sanitize_win_user_profile_path(fallback)
                fallback = os.path.normpath(fallback)
                if fallback != dbp:
                    logger.warning("Falling back from configured DB path to default per-user DB path: %s -> %s", dbp, fallback)
                    dbp = fallback
                    parent2 = os.path.dirname(dbp) or os.getcwd()
                    try:
                        os.makedirs(parent2, exist_ok=True)
                    except Exception as e2:
                        logger.error("Could not create fallback DB parent dir %s: %s", parent2, e2, exc_info=True)
                else:
                    logger.error("Fallback path equals resolved path; cannot recover from write failure: %s", fallback)
            except Exception as e3:
                logger.error("Failed to compute/apply fallback DB path: %s", e3, exc_info=True)
    except Exception:
        logger.exception("Unexpected error while performing DB path diagnostics/fallback")

    # Final ensure parent exists (best-effort)
    try:
        os.makedirs(os.path.dirname(dbp), exist_ok=True)
    except Exception:
        logger.debug("Final os.makedirs failed for %s (ignored)", os.path.dirname(dbp), exc_info=True)

    return dbp


@lru_cache(maxsize=1)
def _cached_resolve_db_path() -> str:
    """
    Cache the resolved DB path for the lifetime of the process to avoid
    repeating expensive path diagnostics and write-tests on every DB call.
    If you need to force a re-evaluation (e.g. config changed at runtime),
    call _cached_resolve_db_path.cache_clear() before calling _connect().
    """
    return _resolve_db_path()


# ------------------------------------------------------------------
# Shared aiosqlite connection: lazy singleton + async context manager
# ------------------------------------------------------------------
# Module-level shared connection and lock
_aiosqlite_conn: Optional[aiosqlite.Connection] = None
_aiosqlite_lock = asyncio.Lock()
_db_path_for_init: Optional[str] = None

async def _ensure_shared_conn(dbp: Optional[str] = None) -> aiosqlite.Connection:
    """
    Ensure a single shared aiosqlite.Connection exists for the process.
    Lazily creates the connection (and bootstraps schema) on first use.
    """
    global _aiosqlite_conn, _db_path_for_init
    async with _aiosqlite_lock:
        if _aiosqlite_conn is None:
            path = dbp or _cached_resolve_db_path()
            _db_path_for_init = path
            conn = await aiosqlite.connect(path)
            try:
                # Pragmatic pragmas for better concurrency
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA busy_timeout=30000;")
                await conn.execute("PRAGMA synchronous=NORMAL;")
                await conn.execute("PRAGMA foreign_keys=ON;")
                conn.row_factory = aiosqlite.Row
                # Ensure schema is present (idempotent)
                await _ensure_core_schema(conn)
                await conn.commit()
            except Exception:
                # Non-fatal: keep connection available even if some setup statements fail
                logger.debug("Shared DB connection bootstrap encountered non-fatal error", exc_info=True)
                try:
                    await conn.commit()
                except Exception:
                    pass
            _aiosqlite_conn = conn
        return _aiosqlite_conn


from contextlib import asynccontextmanager

@asynccontextmanager
async def _shared_connect_ctx(dbp: Optional[str] = None):
    """
    Async context manager yielding the shared connection. Does NOT close the
    connection on exit — callers must call close_shared_db() at shutdown if
    they want the connection closed.
    """
    conn = await _ensure_shared_conn(dbp)
    try:
        yield conn
    finally:
        # Intentionally do not close the shared connection here.
        # This keeps the number of underlying threads used by aiosqlite small.
        pass


# Backwards-compatible public alias expected by earlier imports.
# Previously connect_db returned an object that could be used with `async with`.
# We preserve that contract by returning the new asynccontextmanager.
def _connect(dbp: Optional[str] = None):
    return _shared_connect_ctx(dbp)


def connect_db(dbp: Optional[str] = None):
    """
    Back-compat wrapper so modules can import `connect_db` from this module.
    Returns an async context manager that yields the shared connection.
    """
    return _connect(dbp)


async def close_shared_db() -> None:
    """
    Close the shared aiosqlite connection (call during process shutdown).
    Safe to call multiple times.
    """
    global _aiosqlite_conn
    async with _aiosqlite_lock:
        if _aiosqlite_conn is not None:
            try:
                await _aiosqlite_conn.close()
            except Exception:
                logger.debug("close_shared_db: error closing connection", exc_info=True)
            _aiosqlite_conn = None

async def init_shared_db(dbp: Optional[str] = None) -> None:
    """
    Public wrapper to eagerly initialize the module-shared aiosqlite connection.

    Usage:
      await init_shared_db()            # uses resolved DB path
      await init_shared_db("/path/to/tokens.sqlite3")
    This is idempotent and safe to call multiple times.
    """
    try:
        # ensure connection is created (and schema bootstrapped)
        await _ensure_shared_conn(dbp)
        logger.debug("init_shared_db: shared connection ensured (dbp=%s)", dbp or "<resolved>")
    except Exception:
        logger.exception("init_shared_db: failed to initialize shared DB connection")
        raise

# small helper wrapper for executing SQL via an existing connection
async def _exec(db: aiosqlite.Connection, sql: str, params: Tuple = ()) -> None:
    await db.execute(sql, params)


# ============================
# Compat view helper (minimal)
# ============================
async def _create_compat_views(db: aiosqlite.Connection) -> None:
    await db.executescript("""
        DROP VIEW IF EXISTS eligible_tokens_view;
        CREATE VIEW eligible_tokens_view AS
        SELECT
            address        AS token_address,
            name,
            symbol,
            volume_24h,
            liquidity,
            market_cap,
            price,
            price_change_1h,
            price_change_6h,
            price_change_24h,
            score,
            categories,
            timestamp,
            data,
            created_at
        FROM eligible_tokens;
    """)
    await db.commit()


# =====================
# Core schema bootstrap
# =====================

async def _ensure_core_schema(db: aiosqlite.Connection) -> None:
    await _exec(db, """
        CREATE TABLE IF NOT EXISTS tokens (
            address TEXT PRIMARY KEY,
            name TEXT,
            symbol TEXT,
            volume_24h REAL,
            liquidity REAL,
            market_cap REAL,
            price REAL,
            price_change_1h REAL,
            price_change_6h REAL,
            price_change_24h REAL,
            categories TEXT,
            timestamp INTEGER,
            buy_price REAL,
            buy_txid TEXT,
            buy_time INTEGER,
            sell_price REAL,
            sell_txid TEXT,
            sell_time INTEGER,
            is_trading BOOLEAN
        );
    """)

    await _exec(db, """
        CREATE TABLE IF NOT EXISTS blacklist (
            address TEXT PRIMARY KEY,
            reason TEXT,
            timestamp INTEGER
        );
    """)

    await _exec(db, """
        CREATE TABLE IF NOT EXISTS eligible_tokens (
            address TEXT PRIMARY KEY,
            name TEXT,
            symbol TEXT,
            volume_24h REAL,
            liquidity REAL,
            market_cap REAL,
            price REAL,
            price_change_1h REAL,
            price_change_6h REAL,
            price_change_24h REAL,
            score REAL,
            categories TEXT,
            timestamp INTEGER,
            data TEXT,
            created_at INTEGER
        );
    """)

    # Backfill columns if table pre-existed
    try:
        cols = []
        async with db.execute("PRAGMA table_info(eligible_tokens)") as cur:
            async for row in cur:
                cols.append(row[1])
        if "data" not in cols:
            await _exec(db, "ALTER TABLE eligible_tokens ADD COLUMN data TEXT;")
        if "created_at" not in cols:
            await _exec(db, "ALTER TABLE eligible_tokens ADD COLUMN created_at INTEGER;")
        # Ensure canonical_address exists (safe/idempotent)
        if "canonical_address" not in cols:
            try:
                await _exec(db, "ALTER TABLE eligible_tokens ADD COLUMN canonical_address TEXT;")
            except Exception:
                logger.debug("Could not add 'canonical_address' to eligible_tokens (maybe already exists)", exc_info=True)
    except Exception:
        # Non-fatal: continue even if PRAGMA or ALTER fails
        logger.debug("eligible_tokens schema backfill check failed (continuing)", exc_info=True)

    await _exec(db, """
        CREATE TABLE IF NOT EXISTS cached_token_data (
            address TEXT PRIMARY KEY,
            symbol TEXT,
            market_cap REAL,
            data TEXT
        );
    """)
    await _exec(db, """
        CREATE TABLE IF NOT EXISTS cached_creation_time (
            address TEXT PRIMARY KEY,
            creation_time TEXT
        );
    """)
    
    await _exec(db, """
        CREATE TABLE IF NOT EXISTS trade_history (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT,
            symbol TEXT,
            buy_price REAL,
            sell_price REAL,
            buy_amount REAL,
            sell_amount REAL,
            buy_txid TEXT,
            sell_txid TEXT,
            buy_time INTEGER,
            sell_time INTEGER,
            profit REAL,
            simulated INTEGER DEFAULT 0,
            FOREIGN KEY (token_address) REFERENCES tokens(address)
        );
    """)

    # Ensure 'simulated' column exists on older DBs (idempotent migration)
    try:
        th_cols = []
        async with db.execute("PRAGMA table_info(trade_history)") as cur:
            async for row in cur:
                th_cols.append(row[1])
        if "simulated" not in th_cols:
            try:
                await _exec(db, "ALTER TABLE trade_history ADD COLUMN simulated INTEGER DEFAULT 0;")
                logger.info("Added 'simulated' column to trade_history (migration applied)")
            except Exception:
                # If the ALTER fails for concurrency reasons or other, log debug and continue
                logger.debug("Could not add 'simulated' column to trade_history (maybe already exists)", exc_info=True)
    except Exception:
        logger.debug("trade_history schema inspection failed (continuing)", exc_info=True)
        
    await _exec(db, """
        CREATE TABLE IF NOT EXISTS shortlist_tokens (
            address TEXT PRIMARY KEY,
            data    TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        );
    """)
    try:
        sl_cols = []
        async with db.execute("PRAGMA table_info(shortlist_tokens)") as cur:
            async for row in cur:
                sl_cols.append(row[1])
        if "data" not in sl_cols:
            await _exec(db, "ALTER TABLE shortlist_tokens ADD COLUMN data TEXT;")
        if "created_at" not in sl_cols:
            await _exec(db, "ALTER TABLE shortlist_tokens ADD COLUMN created_at INTEGER;")
    except Exception:
        logger.debug("shortlist_tokens schema backfill check failed (continuing)", exc_info=True)

    await _exec(db, """
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
    # Ensure discovered_tokens has canonical_address column
    try:
        dcols = []
        async with db.execute("PRAGMA table_info(discovered_tokens)") as cur:
            async for row in cur:
                dcols.append(row[1])
        if "canonical_address" not in dcols:
            try:
                await _exec(db, "ALTER TABLE discovered_tokens ADD COLUMN canonical_address TEXT;")
            except Exception:
                logger.debug("Could not add 'canonical_address' to discovered_tokens (maybe already exists)", exc_info=True)
    except Exception:
        logger.debug("discovered_tokens schema backfill check failed (continuing)", exc_info=True)

    # Indexes (idempotent)
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_eligible_tokens_timestamp ON eligible_tokens(timestamp);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_eligible_tokens_score ON eligible_tokens(score);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_tokens_is_trading ON tokens(is_trading);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_trade_history_token ON trade_history(token_address);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_trade_history_buy_time ON trade_history(buy_time);")
    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_trade_history_sell_time ON trade_history(sell_time);")

    # optional index to speed queries by simulated flag (best-effort)
    try:
        await _exec(db, "CREATE INDEX IF NOT EXISTS idx_trade_history_simulated ON trade_history(simulated);")
    except Exception:
        logger.debug("Could not create idx_trade_history_simulated (ignored)", exc_info=True)

    await _exec(db, "CREATE INDEX IF NOT EXISTS idx_shortlist_created_at ON shortlist_tokens(created_at);")

    # Clean up any rows with null addresses (defensive)
    await _exec(db, "DELETE FROM tokens WHERE address IS NULL;")
    await _exec(db, "DELETE FROM eligible_tokens WHERE address IS NULL;")

    # Create compatibility views and commit
    await _create_compat_views(db)
    await db.commit()


# =====================
# Schema init/migration
# =====================
async def init_db(conn: Optional[aiosqlite.Connection] = None) -> None:
    if conn is not None:
        await _ensure_core_schema(conn)
        await _create_compat_views(conn)
        try:
            async with conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                tables = {row[0] async for row in cur}
            expected = {
                'tokens','blacklist','eligible_tokens','cached_token_data',
                'cached_creation_time','trade_history','shortlist_tokens','discovered_tokens'
            }
            missing = expected - tables
            if missing:
                raise aiosqlite.OperationalError(f"Missing tables after init: {missing}")
            logger.info("Database initialized successfully (existing connection)")
        except Exception as e:
            logger.error("Verification on provided connection failed: %s\n%s", e, traceback.format_exc())
            raise
        return

    dbp = _resolve_db_path()
    logger.info(f"Initializing database at {dbp}")
    try:
        async with _connect(dbp) as db:
            # Ensure core schema (this will apply migrations such as adding simulated column)
            await _ensure_core_schema(db)
            await _create_compat_views(db)
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                tables = {row[0] async for row in cur}
            expected = {
                'tokens','blacklist','eligible_tokens','cached_token_data',
                'cached_creation_time','trade_history','shortlist_tokens','discovered_tokens'
            }
            missing = expected - tables
            if missing:
                raise aiosqlite.OperationalError(f"Missing tables after init: {missing}")
            logger.info("Database initialized successfully")
    except aiosqlite.OperationalError as e:
        logger.error(f"Failed to initialize database at {dbp}: {e}\n{traceback.format_exc()}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error initializing database at {dbp}: {e}\n{traceback.format_exc()}")
        raise

# ==================
# Blacklist helpers
# ==================
async def load_blacklist() -> Set[str]:
    try:
        async with _connect() as db:
            bl: Set[str] = set()
            async with db.execute("SELECT address, reason, timestamp FROM blacklist") as cur:
                rows = await cur.fetchall()
                for address, reason, ts in rows:
                    bl.add(address)
                    logger.debug(
                        "Blacklist entry: %s, Reason: %s, Timestamp: %s",
                        address, reason, datetime.fromtimestamp(ts)
                    )
            logger.info("Loaded %d blacklisted tokens", len(bl))
            return bl
    except Exception as e:
        logger.error(f"Failed to load blacklist: {e}\n{traceback.format_exc()}")
        return set()

async def add_to_blacklist(token_address: str, reason: str) -> None:
    try:
        async with _connect() as db:
            await db.execute(
                "INSERT INTO blacklist(address, reason, timestamp) VALUES (?, ?, ?) "
                "ON CONFLICT(address) DO UPDATE SET reason=excluded.reason, timestamp=excluded.timestamp;",
                (token_address, reason, int(time.time())),
            )
            await db.commit()
        logger.info("Added %s to blacklist: %s", token_address, reason)
    except Exception as e:
        logger.error(f"Failed to add {token_address} to blacklist: {e}\n{traceback.format_exc()}")


async def clear_whitelisted_from_blacklist() -> None:
    try:
        if not WHITELISTED_TOKENS:
            return
        async with _connect() as db:
            qmarks = ",".join(["?"] * len(WHITELISTED_TOKENS))
            await db.execute(
                f"DELETE FROM blacklist WHERE address IN ({qmarks});",
                tuple(WHITELISTED_TOKENS.keys())
            )
            await db.commit()
        logger.info("Removed whitelisted tokens from blacklist: %s", list(WHITELISTED_TOKENS.keys()))
    except Exception as e:
        logger.error(f"Failed to clear whitelisted tokens from blacklist: {e}\n{traceback.format_exc()}")


async def clear_expired_blacklist(max_age_hours: float = 24, *, hours: Optional[float] = None) -> None:
    effective_hours = float(hours) if hours is not None else float(max_age_hours)
    cutoff = int(time.time()) - int(effective_hours * 3600)
    try:
        async with _connect() as db:
            before = db.total_changes
            await db.execute("DELETE FROM blacklist WHERE timestamp < ?;", (cutoff,))
            await db.commit()
            changed = db.total_changes - before
        logger.info(
            "Cleared %d expired blacklist entries older than %.1f hours",
            changed, effective_hours
        )
    except Exception as e:
        logger.error(f"Failed to clear expired blacklist: {e}\n{traceback.format_exc()}")


async def review_blacklist() -> None:
    try:
        async with _connect() as db:
            async with db.execute("SELECT address, reason FROM blacklist") as cur:
                rows = await cur.fetchall()
            transient = ("rate limit", "http 429", "temporary")
            removed = 0
            for address, reason in rows:
                r = (reason or "").lower()
                if any(t in r for t in transient):
                    await db.execute("DELETE FROM blacklist WHERE address = ?;", (address,))
                    logger.info("Removed transient blacklist entry: %s (%s)", address, reason)
                    removed += 1
            if removed:
                await db.commit()
        logger.info("Reviewed blacklist, removed %d transient entries", removed)
    except Exception as e:
        logger.error(f"Failed to review blacklist: {e}\n{traceback.format_exc()}")


# =========================
# Creation-time cache helpers
# =========================
async def cache_creation_time(token_address: str, creation_time: Optional[datetime]) -> None:
    try:
        async with _connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cached_creation_time (
                    address TEXT PRIMARY KEY,
                    creation_time TEXT
                );
            """)
            creation_time_str = creation_time.isoformat() if creation_time else None
            await db.execute(
                "INSERT INTO cached_creation_time(address, creation_time) VALUES(?, ?) "
                "ON CONFLICT(address) DO UPDATE SET creation_time=excluded.creation_time;",
                (token_address, creation_time_str),
            )
            await db.commit()
            logger.debug("Cached creation time for %s: %s", token_address, creation_time_str)
    except Exception as e:
        logger.error(f"Failed to cache creation time for {token_address}: {e}\n{traceback.format_exc()}")


async def get_cached_creation_time(token_address: str) -> Optional[datetime]:
    try:
        async with _connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cached_creation_time (
                    address TEXT PRIMARY KEY,
                    creation_time TEXT
                );
            """)
            async with db.execute("SELECT creation_time FROM cached_creation_time WHERE address = ?;", (token_address,)) as cur:
                row = await cur.fetchone()
                if row and row[0]:
                    logger.debug("Retrieved cached creation time for %s: %s", token_address, row[0])
                    return datetime.fromisoformat(row[0])
                return None
    except Exception as e:
        logger.error(f"Failed to retrieve cached creation time for {token_address}: {e}\n{traceback.format_exc()}")
        return None


# ======================
# Token data cache table
# ======================
def _to_optional_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            f = float(v)
            if f != f:
                return None
            return f
        except Exception:
            return None
    try:
        s = str(v).strip()
        if s == "" or s.lower() in ("none", "null", "-", "—"):
            return None
        s = s.replace(",", "").replace("$", "")
        f = float(s)
        if f != f:
            return None
        return f
    except Exception:
        return None


async def cache_token_data(token: Dict) -> None:
    token_address = token.get("address")
    if not token_address:
        logger.warning("Cannot cache token data: missing address for %s", token.get("symbol", "UNKNOWN"))
        return

    market_cap_val: Optional[float] = None
    try:
        for k in ("market_cap", "mc", "fdv"):
            v = token.get(k)
            if v is None:
                continue
            try:
                market_cap_val = float(v)
                break
            except Exception:
                try:
                    s = str(v).replace(",", "").strip()
                    market_cap_val = float(s) if s != "" else None
                    if market_cap_val is not None:
                        break
                except Exception:
                    market_cap_val = None
    except Exception:
        market_cap_val = None

    try:
        async with _connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cached_token_data (
                    address TEXT PRIMARY KEY,
                    symbol TEXT,
                    market_cap REAL,
                    data TEXT
                );
            """)
            token_data_json = json.dumps(token, default=custom_json_encoder)

            await db.execute(
                "INSERT INTO cached_token_data(address, symbol, market_cap, data) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(address) DO UPDATE SET "
                "symbol=excluded.symbol, market_cap=excluded.market_cap, data=excluded.data;",
                (
                    token_address,
                    token.get("symbol", "UNKNOWN"),
                    (market_cap_val if market_cap_val is not None else 0),
                    token_data_json,
                ),
            )
            await db.commit()
            logger.debug("Cached token data for %s (%s)", token.get("symbol", "UNKNOWN"), token_address)
    except Exception as e:
        logger.error(
            "Failed to cache token data for %s (%s): %s\n%s",
            token.get("symbol", "UNKNOWN"),
            token_address,
            e,
            traceback.format_exc(),
        )


async def get_cached_token_data(token_address: str) -> Optional[Dict]:
    try:
        async with _connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cached_token_data (
                    address TEXT PRIMARY KEY,
                    symbol TEXT,
                    market_cap REAL,
                    data TEXT
                );
            """)
            async with db.execute("SELECT data FROM cached_token_data WHERE address = ?;", (token_address,)) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                try:
                    data = json.loads(row[0])
                    if not isinstance(data, dict):
                        logger.error("Invalid cached data format for %s: %s", token_address, type(data))
                        return None
                    logger.debug("Retrieved cached token data for %s: %s...", token_address, json.dumps(data)[:100])
                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse cached data for {token_address}: {e}\n{traceback.format_exc()}")
                    return None
    except Exception as e:
        logger.error(f"Failed to retrieve cached token data for {token_address}: {e}\n{traceback.format_exc()}")
        return None


async def clear_birdeye_cache() -> None:
    try:
        async with _connect() as db:
            await db.execute("DELETE FROM cached_token_data WHERE address = ?;", ("birdeye_tokenlist",))
            await db.commit()
            logger.info("Cleared birdeye_tokenlist cache")
    except Exception as e:
        logger.error(f"Failed to clear birdeye_tokenlist cache: {e}\n{traceback.format_exc()}")


# =========================
# Tokens / Trades tables
# =========================
async def update_token_record(
    token: Dict,
    buy_price: float,
    buy_txid: str,
    buy_time: int,
    is_trading: bool = True,
) -> None:
    try:
        categories_json = json.dumps(token.get("categories", []), default=custom_json_encoder)
        async with _connect() as db:
            await db.execute("""
                INSERT INTO tokens (
                    address, name, symbol, volume_24h, liquidity, market_cap, price,
                    price_change_1h, price_change_6h, price_change_24h, categories,
                    timestamp, buy_price, buy_txid, buy_time, is_trading
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    name=excluded.name,
                    symbol=excluded.symbol,
                    volume_24h=excluded.volume_24h,
                    liquidity=excluded.liquidity,
                    market_cap=excluded.market_cap,
                    price=excluded.price,
                    price_change_1h=excluded.price_change_1h,
                    price_change_6h=excluded.price_change_6h,
                    price_change_24h=excluded.price_change_24h,
                    categories=excluded.categories,
                    timestamp=excluded.timestamp,
                    buy_price=excluded.buy_price,
                    buy_txid=excluded.buy_txid,
                    buy_time=excluded.buy_time,
                    is_trading=excluded.is_trading;
            """, (
                token.get("address"),
                token.get("name", "UNKNOWN"),
                token.get("symbol", "UNKNOWN"),
                float(token.get("volume_24h", 0)),
                float(token.get("liquidity", 0)),
                float(_to_optional_float(token.get("market_cap")) or 0.0),
                _to_optional_float(token.get("price")),
                _to_optional_float(token.get("price_change_1h")),
                _to_optional_float(token.get("price_change_6h")),
                _to_optional_float(token.get("price_change_24h")),
                categories_json,
                int(token.get("timestamp", time.time())),
                float(buy_price),
                buy_txid,
                int(buy_time),
                bool(is_trading),
            ))
            await db.commit()
        logger.debug("Updated token record for %s (%s)", token.get("symbol", "UNKNOWN"), token.get("address"))
    except Exception as e:
        logger.error(f"Failed to update token record for {token.get('address','UNKNOWN')}: {e}\n{traceback.format_exc()}")


# ---------- replace existing record_trade with this updated version ----------
async def record_trade(
    token: Dict,
    buy_price: Optional[float] = None,
    buy_amount: Optional[float] = None,
    buy_txid: Optional[str] = None,
    buy_time: Optional[int] = None,
    sell_price: Optional[float] = None,
    sell_amount: Optional[float] = None,
    sell_txid: Optional[str] = None,
    sell_time: Optional[int] = None,
    simulated: bool = False,
) -> None:
    """
    Persist a trade row into trade_history.

    - token: token dict (must contain 'address' and optionally 'symbol')
    - buy_price: per-token price (SOL)
    - buy_amount: token quantity (recommended). If you only have SOL amount, convert before calling:
        token_qty = sol_spent / buy_price
    - simulated: True for DRY_RUN / paper trades
    """
    # Compute profit if we have both price and amount for a full round-trip
    profit = None
    try:
        if sell_price is not None and sell_amount is not None and buy_price is not None and buy_amount is not None:
            qty = min(float(buy_amount), float(sell_amount))
            profit = (float(sell_price) - float(buy_price)) * qty
    except Exception:
        profit = None

    try:
        async with _connect() as db:
            await db.execute("""
                INSERT INTO trade_history (
                    token_address, symbol, buy_price, sell_price, buy_amount, sell_amount,
                    buy_txid, sell_txid, buy_time, sell_time, profit, simulated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                token.get("address"),
                token.get("symbol", "UNKNOWN"),
                (float(buy_price) if buy_price is not None else None),
                (float(sell_price) if sell_price is not None else None),
                (float(buy_amount) if buy_amount is not None else None),
                (float(sell_amount) if sell_amount is not None else None),
                buy_txid,
                sell_txid,
                (int(buy_time) if buy_time is not None else None),
                (int(sell_time) if sell_time is not None else None),
                (float(profit) if profit is not None else None),
                (1 if simulated else 0),
            ))
            await db.commit()
        if simulated:
            logger.info("Persisted simulated trade for %s tx=%s/%s", token.get("address"), buy_txid, sell_txid)
        else:
            logger.debug("Recorded trade for %s (%s)", token.get("symbol", "UNKNOWN"), token.get("address"))
    except Exception as e:
        logger.error(f"Failed to record trade for {token.get('address','UNKNOWN')}: {e}\n{traceback.format_exc()}")


async def get_open_positions() -> List[Dict]:
    try:
        async with _connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM tokens WHERE is_trading = 1;") as cur:
                rows = await cur.fetchall()
                return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_open_positions failed: {e}\n{traceback.format_exc()}")
        return []


async def mark_token_sold(
    token_address: str,
    sell_price: float,
    sell_txid: str,
    sell_time: int,
) -> None:
    try:
        async with _connect() as db:
            await db.execute(
                """
                UPDATE tokens
                   SET sell_price = ?,
                       sell_txid  = ?,
                       sell_time  = ?,
                       is_trading = 0
                 WHERE address = ?;
                """,
                (float(sell_price), str(sell_txid), int(sell_time), token_address),
            )
            await db.commit()
        logger.debug("Marked %s sold @ %.8f (tx %s)", token_address, sell_price, sell_txid)
    except Exception as e:
        logger.error(f"mark_token_sold failed for {token_address}: {e}\n{traceback.format_exc()}")


# ============================
# eligible_tokens (shortlist)
# ============================

async def prune_old_eligible_tokens(max_age_hours: float = 168) -> None:
    cutoff = int(time.time()) - int(max_age_hours * 3600)
    try:
        async with _connect() as db:
            before = db.total_changes
            await db.execute("DELETE FROM eligible_tokens WHERE timestamp < ?;", (cutoff,))
            await db.commit()
            deleted = db.total_changes - before
        logger.info("Pruned %d old eligible_tokens entries older than %.1f hours", deleted, max_age_hours)
    except Exception as e:
        logger.error(f"Failed to prune old eligible tokens: {e}\n{traceback.format_exc()}")


async def get_token_trade_status(token_address: str) -> Optional[Dict]:
    try:
        async with _connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM tokens WHERE address = ?;", (token_address,)) as cur:
                row = await cur.fetchone()
                if row:
                    logger.debug("Retrieved trade status for %s", token_address)
                    return dict(row)
                return None
    except Exception as e:
        logger.error(f"Failed to retrieve trade status for {token_address}: {e}\n{traceback.format_exc()}")
        return None


async def upsert_eligible_token(token: Dict) -> None:
    try:
        categories_json = json.dumps(token.get("categories", []), default=custom_json_encoder)
        async with _connect() as db:
            await db.execute("""
                INSERT INTO eligible_tokens (
                    address, name, symbol, volume_24h, liquidity, market_cap, price,
                    price_change_1h, price_change_6h, price_change_24h, score, categories, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    name=excluded.name,
                    symbol=excluded.symbol,
                    volume_24h=excluded.volume_24h,
                    liquidity=excluded.liquidity,
                    market_cap=excluded.market_cap,
                    price=excluded.price,
                    price_change_1h=excluded.price_change_1h,
                    price_change_6h=excluded.price_change_6h,
                    price_change_24h=excluded.price_change_24h,
                    score=excluded.score,
                    categories=excluded.categories,
                    timestamp=excluded.timestamp;
            """, (
                token.get("address"),
                token.get("name", "UNKNOWN"),
                token.get("symbol", "UNKNOWN"),
                float(token.get("volume_24h", 0)),
                float(token.get("liquidity", 0)),
                float(_to_optional_float(token.get("market_cap")) or 0.0),
                _to_optional_float(token.get("price")),
                _to_optional_float(token.get("price_change_1h")),
                _to_optional_float(token.get("price_change_6h")),
                _to_optional_float(token.get("price_change_24h")),
                float(token.get("score", 0)),
                categories_json,
                int(token.get("timestamp", time.time())),
            ))

            json_blob = json.dumps(token, default=custom_json_encoder)
            await db.execute(
                "UPDATE eligible_tokens "
                "SET data = ?, created_at = COALESCE(created_at, strftime('%s','now')) "
                "WHERE address = ?;",
                (json_blob, token.get("address")),
            )

            await db.commit()
        logger.debug("Upserted eligible_token %s (%s)", token.get("symbol", "UNKNOWN"), token.get("address"))
    except Exception as e:
        logger.error("Failed to upsert eligible_token %s: %s\n%s",
                     token.get("address", "UNKNOWN"), e, traceback.format_exc())


async def bulk_upsert_eligible_tokens(tokens: List[Dict]) -> int:
    if not tokens:
        return 0
    try:
        async with _connect() as db:
            rows = []
            now = int(time.time())
            for t in tokens:
                rows.append((
                    t.get("address"),
                    t.get("name", "UNKNOWN"),
                    t.get("symbol", "UNKNOWN"),
                    float(t.get("volume_24h", 0)),
                    float(t.get("liquidity", 0)),
                    float(_to_optional_float(t.get("market_cap")) or 0.0),
                    _to_optional_float(t.get("price")),
                    _to_optional_float(t.get("price_change_1h")),
                    _to_optional_float(t.get("price_change_6h")),
                    _to_optional_float(t.get("price_change_24h")),
                    float(t.get("score", 0)),
                    json.dumps(t.get("categories", []), default=custom_json_encoder),
                    int(t.get("timestamp", now)),
                ))
            sql = """
                INSERT INTO eligible_tokens (
                    address, name, symbol, volume_24h, liquidity, market_cap, price,
                    price_change_1h, price_change_6h, price_change_24h, score, categories, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(address) DO UPDATE SET
                    name=excluded.name,
                    symbol=excluded.symbol,
                    volume_24h=excluded.volume_24h,
                    liquidity=excluded.liquidity,
                    market_cap=excluded.market_cap,
                    price=excluded.price,
                    price_change_1h=excluded.price_change_1h,
                    price_change_6h=excluded.price_change_6h,
                    price_change_24h=excluded.price_change_24h,
                    score=excluded.score,
                    categories=excluded.categories,
                    timestamp=excluded.timestamp;
            """
            before = db.total_changes
            await db.executemany(sql, rows)

            json_rows = []
            for t in tokens:
                addr = t.get("address")
                if addr:
                    json_rows.append((json.dumps(t, default=custom_json_encoder), addr))
            if json_rows:
                await db.executemany(
                    "UPDATE eligible_tokens "
                    "SET data = ?, created_at = COALESCE(created_at, strftime('%s','now')) "
                    "WHERE address = ?;",
                    json_rows,
                )

            await db.commit()
            written = db.total_changes - before
        logger.debug("Bulk upserted %d eligible_tokens", written)
        return written
    except Exception as e:
        logger.error("Bulk upsert eligible_tokens failed: %s\n%s", e, traceback.format_exc())
        return 0


# Robust invoker to support multiple historical bulk_fn signatures.
async def _maybe_await_result(maybe_awaitable):
    if asyncio.iscoroutine(maybe_awaitable) or inspect.isawaitable(maybe_awaitable):
        return await maybe_awaitable
    return maybe_awaitable


async def _invoke_bulk_upsert_fn(bulk_fn, db_conn: Optional[aiosqlite.Connection], table_name: str, rows):
    """
    Call bulk_fn with the best-matching argument list depending on its signature.
    Supports:
      - bulk_fn(db, table, rows)
      - bulk_fn(table, rows)
      - bulk_fn(db, rows)
      - bulk_fn(rows)
    Returns whatever bulk_fn returns (commonly an int), or None on failure.
    """
    try:
        sig = inspect.signature(bulk_fn)
        pos_params = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        param_count = len(pos_params)
    except Exception:
        param_count = None

    # try sensible ordering
    attempts = []
    if param_count == 3:
        attempts.append(("db_table_rows", (db_conn, table_name, rows)))
    if param_count == 2:
        attempts.extend([
            ("db_rows", (db_conn, rows)),
            ("table_rows", (table_name, rows)),
        ])
    if param_count == 1:
        attempts.append(("rows", (rows,)))

    # Fallback attempts if signature introspection inconclusive
    attempts.extend([
        ("db_table_rows", (db_conn, table_name, rows)),
        ("table_rows", (table_name, rows)),
        ("db_rows", (db_conn, rows)),
        ("rows", (rows,)),
    ])

    last_exc = None
    for name, args in attempts:
        try:
            result = bulk_fn(*args)
            res = await _maybe_await_result(result)
            return res
        except TypeError as te:
            # wrong signature, try next
            last_exc = te
            continue
        except Exception as e:
            # If bulk function attempted to run and raised some other error, log and propagate None
            logger.debug("Bulk helper '%s' raised error when invoked with %s: %s", getattr(bulk_fn, "__name__", repr(bulk_fn)), name, traceback.format_exc())
            return None

    # All attempts failed due to signature mismatch
    logger.debug("bulk upsert helper signature mismatch; last exception: %s", last_exc)
    return None


async def list_eligible_tokens(
    limit: int = 100,
    min_score: Optional[float] = None,
    newer_than: Optional[int] = None,
    order_by_score_desc: bool = True,
) -> List[Dict]:
    try:
        async with _connect() as db:
            db.row_factory = aiosqlite.Row

            where: List[str] = []
            params: List[Any] = []

            if isinstance(min_score, (int, float)):
                where.append("score >= ?")
                params.append(float(min_score))
            if isinstance(newer_than, int):
                where.append("timestamp >= ?")
                params.append(int(newer_than))

            STABLES = (
                "usdc", "usdt", "usdce", "pyusd", "dai", "tusd",
                "usde", "susd", "lusd", "eurc"
            )
            WRAPPERS_LPS = (
                "jitosol", "bsol", "msol", "lsol", "stsol", "jlp",
                "lp"
            )

            placeholders = ",".join(["?"] * (len(STABLES) + len(WRAPPERS_LPS)))
            where.append(f"COALESCE(LOWER(symbol), '') NOT IN ({placeholders})")
            params.extend([*STABLES, *WRAPPERS_LPS])

            where.append(
                "LOWER(name) NOT LIKE ? AND LOWER(name) NOT LIKE ?"
            )
            params.extend(["%staked%", "%staking%"])

            where.append(
                "(categories IS NULL OR ("
                "LOWER(categories) NOT LIKE ? AND "
                "LOWER(categories) NOT LIKE ? AND "
                "LOWER(categories) NOT LIKE ?))"
            )
            params.extend(['%"stable"%', '%"lp"%', '%"staking"%'])

            where_sql = f"WHERE {' AND '.join(where)}" if where else ""
            order_sql = (
                "ORDER BY score DESC, timestamp DESC"
                if order_by_score_desc else
                "ORDER BY timestamp DESC"
            )

            sql = f"""
                SELECT address, name, symbol, volume_24h, liquidity, market_cap, price,
                       price_change_1h, price_change_6h, price_change_24h, score, categories, timestamp
                FROM eligible_tokens
                {where_sql}
                {order_sql}
                LIMIT ?;
            """
            params.append(int(max(1, limit)))

            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
                res = []
                for r in rows:
                    d = dict(r)
                    try:
                        d["categories"] = json.loads(d.get("categories") or "[]")
                    except Exception:
                        pass
                    res.append(d)
                return res
    except Exception as e:
        logger.error("list_eligible_tokens failed: %s\n%s", e, traceback.format_exc())
        return []


# ================
# Shortlist bridge
# ================
def _to_eligible_row(t: Dict) -> Optional[Dict]:
    now = int(time.time())

    vol = t.get("volume_24h")
    if vol is None:
        vol = t.get("v24hUSD") or t.get("v24h")
        if vol is None:
            vol_field = t.get("volume")
            if isinstance(vol_field, dict):
                vol = vol_field.get("h24")
    if vol is None:
        vol = 0

    mc = t.get("market_cap")
    if mc is None:
        mc = t.get("mc", t.get("fdv", 0))

    price = t.get("price", None)

    pc1h = t.get("price_change_1h", t.get("priceChange1h", None))
    pc6h = t.get("price_change_6h", t.get("priceChange6h", None))
    pc24h = t.get("price_change_24h", t.get("priceChange24h", None))

    ts = t.get("timestamp")
    if ts is None:
        ts = t.get("pairCreatedAt", t.get("createdAt", now))

    try:
        ts = int(ts)
        if ts > 10_000_000_000:
            ts //= 1000
    except Exception:
        ts = now

    cats = t.get("categories", [])
    if isinstance(cats, str):
        cats = [cats]

    sym = str(t.get("symbol") or "").lower()
    name = str(t.get("name") or "").lower()

    has_lp_word = bool(re.search(r"\blp\b", name))
    has_staking = ("staked" in name) or ("staking" in name)

    if has_lp_word or has_staking or sym in ("jitosol", "bsol", "jlp", "msol", "lsol", "psol", "stsol"):
        return None

    return {
        "address": t.get("address"),
        "name": t.get("name", "UNKNOWN"),
        "symbol": t.get("symbol", "UNKNOWN"),
        "volume_24h": float(vol or 0),
        "liquidity": float(t.get("liquidity", 0)),
        "market_cap": float(mc or 0),
        "price": float(price or 0),
        "price_change_1h": float(pc1h or 0),
        "price_change_6h": float(pc6h or 0),
        "price_change_24h": float(pc24h or 0),
        "score": float(t.get("score", 0) or 0),
        "categories": cats,
        "timestamp": int(ts or now),
    }


# Robust persist_eligible_shortlist using flexible bulk invocation
async def persist_eligible_shortlist(
    top_tokens: Any,
    prune_hours: int = 168,
) -> int:
    """
    Normalize incoming shortlist results and persist into eligible_tokens.
    Returns number of rows written (best-effort) or 0 on failure/no-rows.
    """
    try:
        async with _connect() as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS eligible_tokens (
                    address TEXT PRIMARY KEY,
                    name TEXT,
                    symbol TEXT,
                    volume_24h REAL,
                    liquidity REAL,
                    market_cap REAL,
                    price REAL,
                    price_change_1h REAL,
                    price_change_6h REAL,
                    price_change_24h REAL,
                    score REAL,
                    categories TEXT,
                    timestamp INTEGER,
                    data TEXT,
                    created_at INTEGER
                );
            """)
            cur = await db.execute("PRAGMA table_info(eligible_tokens);")
            cols_rows = await cur.fetchall()
            cols = [r[1] for r in cols_rows]
            if "data" not in cols:
                try:
                    await db.execute("ALTER TABLE eligible_tokens ADD COLUMN data TEXT;")
                except Exception:
                    logger.debug("Could not add 'data' column to eligible_tokens (maybe already exists)", exc_info=True)
            if "created_at" not in cols:
                try:
                    await db.execute("ALTER TABLE eligible_tokens ADD COLUMN created_at INTEGER;")
                except Exception:
                    logger.debug("Could not add 'created_at' column to eligible_tokens (maybe already exists)", exc_info=True)
            await db.commit()
    except Exception:
        logger.debug("persist_eligible_shortlist: schema ensure failed (continuing).", exc_info=True)

    rows_to_upsert: List[Dict[str, Any]] = []
    try:
        if isinstance(top_tokens, dict):
            for cat, items in top_tokens.items():
                for t in (items or []):
                    r = _to_eligible_row(t) if callable(globals().get("_to_eligible_row")) else dict(t or {})
                    if not r:
                        continue
                    cats = set(r.get("categories") or [])
                    cats.add("shortlist")
                    cats.add(str(cat))
                    r["categories"] = list(cats)
                    rows_to_upsert.append(r)
        elif isinstance(top_tokens, (list, tuple)):
            for t in top_tokens:
                r = _to_eligible_row(t) if callable(globals().get("_to_eligible_row")) else dict(t or {})
                if not r:
                    continue
                cats = set(r.get("categories") or [])
                cats.add("shortlist")
                cats.add("unknown")
                r["categories"] = list(cats)
                rows_to_upsert.append(r)
        else:
            try:
                for cat, t in top_tokens:
                    r = _to_eligible_row(t) if callable(globals().get("_to_eligible_row")) else dict(t or {})
                    if not r:
                        continue
                    cats = set(r.get("categories") or [])
                    cats.add("shortlist")
                    cats.add(str(cat))
                    r["categories"] = list(cats)
                    rows_to_upsert.append(r)
            except Exception:
                rows_to_upsert = list(top_tokens or [])
    except Exception:
        logger.exception("persist_eligible_shortlist: normalization failed")
        return 0

    if not rows_to_upsert:
        return 0

    written = 0
    mod = sys.modules.get(__name__) or __import__(__name__)

    bulk_fn = getattr(mod, "bulk_upsert_eligible_tokens", None) or globals().get("bulk_upsert_eligible_tokens")
    if callable(bulk_fn):
        try:
            async with _connect() as db:
                res = await _invoke_bulk_upsert_fn(bulk_fn, db, "eligible_tokens", rows_to_upsert)
                if isinstance(res, int) and res > 0:
                    written = int(res)
                elif res is None:
                    logger.warning("bulk upsert helper returned None or failed; falling back to per-row upsert")
                else:
                    # Some bulk helpers may have no return but succeed; assume success when no exception
                    written = len(rows_to_upsert)
        except Exception:
            logger.exception("bulk_upsert_eligible_tokens failed; falling back to per-row upsert")

    if written == 0:
        try:
            async with _connect() as db:
                for r in rows_to_upsert:
                    try:
                        await db.execute(
                            """
                            INSERT OR REPLACE INTO eligible_tokens
                            (address, name, symbol, volume_24h, liquidity, market_cap, price, price_change_1h, price_change_6h, price_change_24h, score, categories, timestamp, data, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                            """,
                            (
                                r.get("address"),
                                r.get("name"),
                                r.get("symbol"),
                                float(r.get("volume_24h") or 0.0),
                                float(r.get("liquidity") or 0.0),
                                float(r.get("market_cap") or r.get("mc") or 0.0),
                                float(r.get("price") or 0.0),
                                float(r.get("price_change_1h") or r.get("priceChange1h") or 0.0),
                                float(r.get("price_change_6h") or 0.0),
                                float(r.get("price_change_24h") or 0.0),
                                float(r.get("score") or 0.0),
                                json.dumps(r.get("categories") or []),
                                int(r.get("timestamp") or 0),
                                json.dumps(r.get("data") or r),
                                int(r.get("created_at") or r.get("creation_timestamp") or 0),
                            ),
                        )
                    except Exception:
                        logger.debug("eligible_tokens per-row upsert failed for %s", r.get("address"), exc_info=True)
                try:
                    await db.commit()
                except Exception:
                    logger.debug("eligible_tokens commit failed", exc_info=True)
            written = len(rows_to_upsert)
        except Exception:
            logger.exception("persist_eligible_shortlist per-row upsert failed")
            return 0

    if prune_hours and prune_hours > 0:
        prune_fn = getattr(mod, "prune_old_eligible_tokens", None) or globals().get("prune_old_eligible_tokens")
        if callable(prune_fn):
            try:
                if asyncio.iscoroutinefunction(prune_fn):
                    await prune_fn(prune_hours)
                else:
                    await asyncio.to_thread(prune_fn, prune_hours)
            except Exception:
                logger.debug("prune_old_eligible_tokens failed", exc_info=True)
        else:
            try:
                cutoff = int(time.time()) - int(prune_hours * 3600)
                async with _connect() as db:
                    await db.execute("DELETE FROM eligible_tokens WHERE created_at < ?;", (cutoff,))
                    await db.commit()
            except Exception:
                logger.debug("Inline prune of eligible_tokens failed", exc_info=True)

    return int(written)


async def persist_discovered_tokens(tokens: Any, prune_hours: int = 24) -> Optional[int]:
    if not tokens:
        return 0
    rows: List[Dict[str, Any]] = list(tokens)
    mod = sys.modules.get(__name__) or __import__(__name__)

    ensure_fn = getattr(mod, "ensure_discovered_tokens_schema", None) or getattr(mod, "ensure_tokens_schema", None) or getattr(mod, "ensure_schema", None)
    if callable(ensure_fn):
        try:
            if asyncio.iscoroutinefunction(ensure_fn):
                await ensure_fn()
            else:
                await asyncio.to_thread(ensure_fn)
        except Exception:
            logger.debug("ensure_discovered_tokens_schema helper failed (continuing)", exc_info=True)

    bulk_fn = getattr(mod, "bulk_upsert_tokens", None) or globals().get("bulk_upsert_tokens")
    if callable(bulk_fn):
        try:
            async with _connect() as db:
                res = await _invoke_bulk_upsert_fn(bulk_fn, db, "discovered_tokens", rows)
                if isinstance(res, int):
                    if prune_hours and prune_hours > 0:
                        try:
                            cutoff = int(time.time()) - int(prune_hours * 3600)
                            async with _connect() as db2:
                                await db2.execute("DELETE FROM discovered_tokens WHERE created_at < ?;", (cutoff,))
                                await db2.commit()
                        except Exception:
                            logger.debug("persist_discovered_tokens: prune after bulk failed", exc_info=True)
                    return int(res)
                # if bulk helper returned None or non-int, fall back to per-row
        except Exception:
            logger.exception("bulk_upsert_tokens failed for discovered_tokens; falling back to per-row upsert")

    try:
        async with _connect() as db:
            for tok in rows:
                try:
                    created_at_val = tok.get("created_at") or tok.get("creation_timestamp") or 0
                    try:
                        if hasattr(created_at_val, "timestamp"):
                            created_at_val = int(created_at_val.timestamp())
                        else:
                            created_at_val = int(created_at_val or 0)
                    except Exception:
                        created_at_val = int(created_at_val or 0)

                    await db.execute(
                        """
                        INSERT OR REPLACE INTO discovered_tokens
                        (address, data, name, symbol, created_at, creation_timestamp, price, liquidity, market_cap, v24hUSD, volume_24h, dexscreenerUrl, dsPairAddress, links)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            tok.get("address"),
                            json.dumps(tok),                       # write JSON into `data` column (schema uses `data`)
                            tok.get("name"),
                            tok.get("symbol"),
                            int(created_at_val or 0),
                            int(tok.get("creation_timestamp") or 0),
                            float(tok.get("price") or 0.0),
                            float(tok.get("liquidity") or 0.0),
                            float(tok.get("market_cap") or tok.get("mc") or 0.0),
                            float(tok.get("v24hUSD") or 0.0),
                            float(tok.get("volume_24h") or tok.get("v24hUSD") or 0.0),
                            tok.get("dexscreenerUrl") or "",
                            tok.get("dsPairAddress") or "",
                            json.dumps(tok.get("links") or []),
                        ),
                    )
                except Exception:
                    logger.debug("persist_discovered_tokens: per-row upsert failed for %s", tok.get("address"), exc_info=True)
            try:
                await db.commit()
            except Exception:
                logger.debug("commit failed after per-row upsert (discovered_tokens)", exc_info=True)

            if prune_hours and prune_hours > 0:
                try:
                    cutoff = int(time.time()) - int(prune_hours * 3600)
                    await db.execute("DELETE FROM discovered_tokens WHERE created_at < ?;", (cutoff,))
                    await db.commit()
                except Exception:
                    logger.debug("persist_discovered_tokens: prune failed", exc_info=True)

        return len(rows)
    except Exception:
        logger.exception("Failed to persist discovered_tokens")
        return None


# ==============================
# GUI/Runner alignment helpers
# ==============================
async def ensure_eligible_tokens_schema(db: Optional[aiosqlite.Connection] = None) -> None:
    if db is None:
        async with _connect() as _db:
            await ensure_eligible_tokens_schema(_db)
        return

    await db.execute("""
        CREATE TABLE IF NOT EXISTS eligible_tokens (
            address TEXT PRIMARY KEY,
            name TEXT,
            symbol TEXT,
            volume_24h REAL,
            liquidity REAL,
            market_cap REAL,
            price REAL,
            price_change_1h REAL,
            price_change_6h REAL,
            price_change_24h REAL,
            score REAL,
            categories TEXT,
            timestamp INTEGER,
            data TEXT,
            created_at INTEGER
        );
    """)
    cols = []
    async with db.execute("PRAGMA table_info(eligible_tokens)") as cur:
        async for row in cur:
            cols.append(row[1])
    if "data" not in cols:
        await db.execute("ALTER TABLE eligible_tokens ADD COLUMN data TEXT;")
    if "created_at" not in cols:
        await db.execute("ALTER TABLE eligible_tokens ADD COLUMN created_at INTEGER DEFAULT (strftime('%s','now'));")
    await db.commit()


async def ensure_shortlist_tokens_schema(db: Optional[aiosqlite.Connection] = None) -> None:
    if db is None:
        async with _connect() as _db:
            await ensure_shortlist_tokens_schema(_db)
        return

    await db.execute("""
        CREATE TABLE IF NOT EXISTS shortlist_tokens (
            address TEXT PRIMARY KEY,
            data    TEXT,
            created_at INTEGER DEFAULT (strftime('%s','now'))
        );
    """)
    cols = []
    async with db.execute("PRAGMA table_info(shortlist_tokens)") as cur:
        async for row in cur:
            cols.append(row[1])
    if "data" not in cols:
        await db.execute("ALTER TABLE shortlist_tokens ADD COLUMN data TEXT;")
    if "created_at" not in cols:
        await db.execute("ALTER TABLE shortlist_tokens ADD COLUMN created_at INTEGER;")
    await db.commit()


async def ensure_discovered_tokens_schema(db: Optional[aiosqlite.Connection] = None) -> None:
    if db is None:
        async with _connect() as _db:
            await ensure_discovered_tokens_schema(_db)
        return

    await db.execute("""
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
    await db.commit()

def _ensure_canonical_in_row(row: Dict[str, Any], addr_field: str = "address") -> Dict[str, Any]:
    """Return a copy of row with canonical_address set (if addr present). Non-destructive."""
    r = dict(row or {})
    raw = r.get(addr_field)
    if raw and not r.get("canonical_address"):
        try:
            r["canonical_address"] = extract_canonical_mint(raw)
        except Exception:
            r["canonical_address"] = raw
    return r

async def upsert_token_row(db: aiosqlite.Connection, table: str, token: Dict) -> None:
    if table not in {"shortlist_tokens", "eligible_tokens", "discovered_tokens"}:
        raise ValueError(f"Invalid table name: {table}")

    addr = token.get("address")
    if not addr:
        return

    payload = json.dumps(
        token,
        default=custom_json_encoder,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    sql = (
        f"INSERT INTO {table} (address, data, created_at) "
        "VALUES (?, ?, strftime('%s','now')) "
        "ON CONFLICT(address) DO UPDATE SET data=excluded.data, created_at=excluded.created_at;"
    )
    await db.execute(sql, (addr, payload))


async def bulk_upsert_tokens(db: aiosqlite.Connection, table: str, tokens: List[Dict]) -> None:
    for t in tokens or []:
        await upsert_token_row(db, table, t)
    await db.commit()


# ============================
# Atomic shortlist replacement
# ============================
async def clear_eligible_tokens() -> None:
    try:
        async with _connect() as db:
            await ensure_eligible_tokens_schema(db)
            await db.execute("DELETE FROM eligible_tokens;")
            await db.commit()
        logger.info("Cleared eligible_tokens")
    except Exception as e:
        logger.error(f"Failed to clear eligible_tokens: {e}\n{traceback.format_exc()}")


async def save_eligible_tokens(tokens: Sequence[Dict[str, Any]]) -> int:
    try:
        async with _connect() as db:
            await ensure_eligible_tokens_schema(db)
            await db.execute("DELETE FROM eligible_tokens;")

            now = int(time.time())
            core_rows = []
            json_rows = []
            for t in tokens or []:
                addr = t.get("address")
                if not addr:
                    continue
                core_rows.append((
                    addr,
                    t.get("name", "UNKNOWN"),
                    t.get("symbol", "UNKNOWN"),
                    float(t.get("volume_24h", 0)),
                    float(t.get("liquidity", 0)),
                    float(_to_optional_float(t.get("market_cap")) or 0.0),
                    _to_optional_float(t.get("price")),
                    _to_optional_float(t.get("price_change_1h")),
                    _to_optional_float(t.get("price_change_6h")),
                    _to_optional_float(t.get("price_change_24h")),
                    float(t.get("score", 0)),
                    json.dumps(t.get("categories", []), default=custom_json_encoder),
                    int(t.get("timestamp", now)),
                ))
                json_rows.append((json.dumps(t, default=custom_json_encoder), addr))

            if core_rows:
                await db.executemany("""
                    INSERT INTO eligible_tokens (
                        address, name, symbol, volume_24h, liquidity, market_cap, price,
                        price_change_1h, price_change_6h, price_change_24h, score, categories, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, core_rows)

            if json_rows:
                await db.executemany(
                    "UPDATE eligible_tokens "
                    "SET data = ?, created_at = COALESCE(created_at, strftime('%s','now')) "
                    "WHERE address = ?;",
                    json_rows,
                )

            await db.commit()
            written = len(core_rows)
        logger.info("Persisted %d shortlisted tokens into eligible_tokens", written)
        return written
    except Exception as e:
        logger.error(f"Failed to save eligible_tokens: {e}\n{traceback.format_exc()}")
        return 0
