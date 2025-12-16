from __future__ import annotations

# --- robust import bootstrap (must be FIRST) -----------------------
import sys, os
from pathlib import Path

_PKG_ROOT     = Path(__file__).resolve().parent        # .../solana_trading_bot_bundle
_PROJECT_ROOT = _PKG_ROOT.parent                       # repo root (e.g. .../NewFolder)

# Try a normal import; if that fails, prepend the project root.
try:
    import solana_trading_bot_bundle  # noqa: F401
except ModuleNotFoundError:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure the package (and key subpackages) have __init__.py
# Safe no-op if they already exist; keeps imports consistent whether run as a module or script.
for sub in ("", "utils", "trading_bot"):
    init_file = _PKG_ROOT / sub / "__init__.py"
    if not init_file.exists():
        try:
            init_file.write_text("# package marker\n", encoding="utf-8")
        except Exception:
            pass
# --- end bootstrap --------------------------------------------------

import json
import time
import asyncio
import signal
import logging
import traceback
import random
import contextlib
from typing import Any, Dict, List, Optional, Tuple


# utils/env loader: support both layouts (subpackage or flat module)
# Prioritise the top-level utils package where your env_loader actually resides:
#   solana_trading_bot_bundle/utils/env_loader.py
# Fallback to flatter layouts where trading_bot/utils/env_loader.py may exist.
try:
    from solana_trading_bot_bundle.utils.env_loader import (
        ensure_appdata_env_bootstrap,
        load_env_first_found,
        prefer_appdata_file,
        token_cache_path,
        log_file_path,
        ensure_one_time_credentials,
        get_active_private_key,
    )
except ModuleNotFoundError:
    try:
        # If running from a flat repo layout (not installed package)
        from trading_bot.utils.env_loader import (
            ensure_appdata_env_bootstrap,
            load_env_first_found,
            prefer_appdata_file,
            token_cache_path,
            log_file_path,
            ensure_one_time_credentials,
            get_active_private_key,
        )
    except Exception:
        # Last-ditch attempt: older layout where env_loader might be under trading_bot package inside bundle
        try:
            from solana_trading_bot_bundle.trading_bot.utils.env_loader import (
                ensure_appdata_env_bootstrap,
                load_env_first_found,
                prefer_appdata_file,
                token_cache_path,
                log_file_path,
                ensure_one_time_credentials,
                get_active_private_key,
            )
        except Exception:
            # Provide tiny no-op fallbacks so launcher won't crash immediately.
            def ensure_appdata_env_bootstrap(*a, **k): return None
            def load_env_first_found(*a, **k): return None
            def prefer_appdata_file(*a, **k): return str(Path.cwd())
            token_cache_path = "tokens.sqlite3"
            log_file_path = "logs/bot.log"
            def ensure_one_time_credentials(*a, **k): return None
            def get_active_private_key(): return None

# 1) Bootstrap .env into appdata if missing
ensure_appdata_env_bootstrap()
# 2) Load .env (cwd → exe dir → appdata). Do this exactly once.
_used_env = load_env_first_found(override=False)
print(f"[ENV] Loaded .env from: {_used_env}" if _used_env else "[ENV] No .env found; using process env only")

# Load core helpers (load_config, setup_logging) — prefer the renamed utils_exec module,
# then fall back to the legacy trading_bot.utils, then to a flat layout.
try:
    from solana_trading_bot_bundle.trading_bot.utils_exec import (
        load_config,
        setup_logging,
    )
except Exception:
    try:
        from solana_trading_bot_bundle.trading_bot.utils import (
            load_config,
            setup_logging,
        )
    except Exception:
        try:
            # flat repo layout fallback
            from trading_bot.utils import load_config, setup_logging
        except Exception:
            # Conservative stubs to avoid immediate crash. These are minimal and should be
            # replaced by the real implementations in normal runs.
            def load_config(path: str | None = None) -> Dict:
                return {}
            def setup_logging(config: dict | None = None) -> logging.Logger:
                logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
                return logging.getLogger("TradingBot")

# --- Resolve config path robustly: prefer YAML in working dir or appdata ---
def _resolve_config_path() -> str:
    import os
    from pathlib import Path
    # 1) Honor explicit env if set
    p = os.environ.get("SOLO_BOT_CONFIG")
    if p and Path(p).exists():
        return p
    # 2) Prefer ./config.yaml (zero-setup for testers)
    here_yaml = Path(os.getcwd()) / "config.yaml"
    if here_yaml.exists():
        return str(here_yaml)
    # 3) AppData / platform defaults
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "SOLOTradingBot"
    yaml_path = base / "config.yaml"
    json_path = base / "config.json"
    if yaml_path.exists():
        return str(yaml_path)
    if json_path.exists():
        return str(json_path)
    # 4) Legacy fallback
    here_json = Path(os.getcwd()) / "config.json"
    return str(here_json)


from solana_trading_bot_bundle.trading_bot.market_data import get_sol_price

from solana_trading_bot_bundle.trading_bot.database import (
    # Ensure schema up-front and use connect_db() for normalized paths
    connect_db,            # async context manager
    init_db,               # can be called without a connection
    ensure_shortlist_tokens_schema,
    ensure_eligible_tokens_schema,
    bulk_upsert_tokens,
    save_eligible_tokens,
    clear_expired_blacklist,
    review_blacklist,
)

from solana_trading_bot_bundle.trading_bot.eligibility import (
    is_token_eligible,
    select_top_five_per_category,
    log_filter_summary,
    ALWAYS_HIDE_IN_CATEGORIES,
   
)

from solana_trading_bot_bundle.trading_bot.fetching import (
    fetch_dexscreener_search,
    fetch_raydium_tokens,
    fetch_birdeye_tokens,
    ensure_rugcheck_status_file,
)

from solana_trading_bot_bundle.price_enrichment_helper_file import (
    enrich_tokens_with_price_change,
)

# Trading cycle (run after discovery when enabled)
# NOTE: We import the trading entrypoint lazily inside _async_main_loop()
# to avoid ModuleNotFoundError when running from a bundled layout.

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logger = logging.getLogger("TradingBot")
if not logger.handlers:
    lvl = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# -- Rugcheck banner/status touch (no other behavior changed) ----
try:
    ensure_rugcheck_status_file()
except Exception as _e:
    logger.debug("ensure_rugcheck_status_file error: %s", _e, exc_info=True)

# ---------------------------------------------------------------
# Safe env helpers (tolerate quotes/comments/underscores)
# ---------------------------------------------------------------
def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().strip('"').strip("'")
    return v if v != "" else default

def _env_int(name: str, default: int) -> int:
    v = _env_str(name)
    if not v:
        return default
    try:
        return int(v.replace("_", " ").split()[0])
    except Exception:
        # last resort: extract first integer sequence
        import re
        m = re.search(r"-?\d+", v.replace("_", ""))
        return int(m.group()) if m else default

def _env_float(name: str, default: float) -> float:
    v = _env_str(name)
    if not v:
        return default
    try:
        return float(v.replace("_", " ").split()[0])
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = _env_str(name)
    return default if v is None else v.lower() in ("1", "true", "yes", "on", "y", "t")

def _env_list(name: str, fallback_names: Optional[List[str]] = None, default: Optional[List[str]] = None) -> List[str]:
    """
    Parse a CSV-ish env var into a de-duplicated string list, preserving order.
    Optionally check additional names (in order) until one is set.
    """
    candidates = [name] + (fallback_names or [])
    raw = None
    for n in candidates:
        val = os.getenv(n)
        if val is not None and str(val).strip():
            raw = str(val)
            break
    if raw is None:
        return list(default or [])
    parts = [p.strip() for p in raw.replace("\\n", ",").replace(";", ",").split(",")]
    seen = set()
    out: List[str] = []
    for p in parts:
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out

# ---------------------------------------------------------------
# Env knobs (safe)
# ---------------------------------------------------------------
DISCOVERY_CYCLE_SECONDS = _env_float("DISCOVERY_CYCLE_SECONDS", 60.0)
AIOHTTP_CONN_LIMIT       = _env_int("AIOHTTP_CONN_LIMIT", 50)
ELIGIBILITY_CONCURRENCY  = _env_int("ELIGIBILITY_CONCURRENCY", 50)
SHORTLIST_PER_CATEGORY   = _env_int("SHORTLIST_PER_CATEGORY", 5)

# Dexscreener knobs:
# - legacy single query fallback (DEXSCREENER_QUERY/DEX_QUERY)
# - multi-query fanout preferred (DEXSCREENER_QUERIES / DEX_QUERIES as CSV)
DEX_QUERY    = _env_str("DEXSCREENER_QUERY", _env_str("DEX_QUERY", "solana")) or "solana"
DEX_QUERIES  = _env_list("DEXSCREENER_QUERIES", fallback_names=["DEX_QUERIES"], default=[DEX_QUERY]) or [DEX_QUERY]

# stop flag file
STOP_FLAG_PATH = Path(prefer_appdata_file("bot_stop_flag.txt"))

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def log_error_with_stacktrace(message: str, error: Exception) -> None:
    logger.error("%s: %s\n%s", message, error, traceback.format_exc())

def _num(x, d=0.0) -> float:
    try:
        return float(x if x is not None else d)
    except Exception:
        return float(d)

def _should_stop() -> bool:
    try:
        return STOP_FLAG_PATH.exists()
    except Exception:
        return False

def _request_stop() -> None:
    try:
        STOP_FLAG_PATH.write_text("stop", encoding="utf-8")
    except Exception:
        pass

def _get_rpc_url(config: Dict[str, Any]) -> str:
    """Gracefully obtain an RPC URL; never KeyError."""
    rpc = (
        (config.get("solana") or {}).get("rpc_url")
        or _env_str("SOLANA_RPC_URL")
        or "https://api.mainnet-beta.solana.com"
    )
    if (config.get("solana") or {}).get("rpc_url") is None:
        logger.warning("Config missing 'solana.rpc_url'; using fallback: %s", rpc)
    return rpc

# ----------------------------------------------------------------
# Merge — normalize address + safer tie-breaks + carry creation time
# ----------------------------------------------------------------
def _merge_best_by_address(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge duplicates by *base token* address.
    Prefer higher 24h volume; tie-break on liquidity.
    Preserve earliest creation_timestamp if previous had it.
    """
    def _n(x, d=0.0):
        try:
            return float(x if x is not None else d)
        except Exception:
            return float(d)

    best: Dict[str, Dict[str, Any]] = {}
    for t in tokens or []:
        addr_raw = t.get("address") or ""
        addr = addr_raw.strip().lower()
        if not addr:
            continue

        prev = best.get(addr)
        if prev is None:
            best[addr] = t
            continue

        vol_t = _n(t.get("volume_24h", t.get("v24hUSD", 0)))
        vol_p = _n(prev.get("volume_24h", prev.get("v24hUSD", 0)))
        liq_t = _n(t.get("liquidity", 0))
        liq_p = _n(prev.get("liquidity", 0))

        replace = (vol_t > vol_p) or (vol_t == vol_p and liq_t > liq_p)
        if replace:
            if not t.get("creation_timestamp") and prev.get("creation_timestamp"):
                t["creation_timestamp"] = prev["creation_timestamp"]
            best[addr] = t

    return list(best.values())



# ----------------------------------------------------------------
# Eligibility concurrency
# ----------------------------------------------------------------
async def _concurrent_eligibility(
    tokens: List[Dict[str, Any]],
    session,
    config: Dict[str, Any],
    concurrency: int = 50,
) -> List[Dict[str, Any]]:
    if not tokens:
        return []

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _one(tok: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with sem:
            try:
                ok, cats = await is_token_eligible(tok, session, config)
                if ok:
                    tok["categories"] = cats
                    return tok
            except Exception as e:
                logger.debug("Eligibility error for %s: %s", tok.get("address"), e, exc_info=True)
            return None

    tasks = [asyncio.create_task(_one(t)) for t in tokens]
    out: List[Dict[str, Any]] = []
    for coro in asyncio.as_completed(tasks):
        res = await coro
        if res:
            out.append(res)
    return out

# ----------------------------------------------------------------
# Discovery cycle (fetch → merge → enrich → eligibility → shortlist → persist)
# ----------------------------------------------------------------
async def run_discovery_cycle(
    session,          # aiohttp.ClientSession
    solana_client,    # AsyncClient
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    sources_cfg = config.get("sources", {}) or {}
    raydium_enabled  = bool(sources_cfg.get("raydium_enabled", True))
    birdeye_enabled  = bool(sources_cfg.get("birdeye_enabled", True))

    disc_cfg = config.get("discovery", {}) or {}
    dexscreener_pages = int(disc_cfg.get("dexscreener_pages", 8))
    raydium_max_pairs = int(disc_cfg.get("raydium_max_pairs", 100))
    birdeye_max_tokens = int(disc_cfg.get("birdeye_max_tokens", 50))

    # --- Multi-query fanout control --------------------------------
    def _coerce_queries(v) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            return [s for s in [x.strip() for x in v.replace("\\n", ",").replace(";", ",").split(",")] if s]
        if isinstance(v, (list, tuple, set)):
            out: List[str] = []
            seen = set()
            for x in v:
                s = str(x).strip()
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    out.append(s)
            return out
        return []

    cfg_queries = _coerce_queries(disc_cfg.get("dexscreener_queries"))
    dex_queries: List[str] = cfg_queries or DEX_QUERIES or [DEX_QUERY]
    # small safety: cap to a reasonable count if someone passes 200+ tokens
    MAX_FANOUT = max(1, _env_int("DEX_MAX_QUERIES", 24))
    if len(dex_queries) > MAX_FANOUT:
        logger.warning("dexscreener_queries capped from %d to %d by DEX_MAX_QUERIES", len(dex_queries), MAX_FANOUT)
        dex_queries = dex_queries[:MAX_FANOUT]

    # 1) Fetch (label tasks so we can log per-source failures) -------
    tasks = []
    labels: List[str] = []

    # SOL price
    tasks.append(asyncio.create_task(get_sol_price(session))); labels.append("sol")

    # Dexscreener multi-query fanout (each query gets its own task)
    for q in dex_queries:
        tasks.append(asyncio.create_task(fetch_dexscreener_search(session, query=q, pages=dexscreener_pages)))
        labels.append(f"dex:{q}")

    # Raydium/Birdeye
    if raydium_enabled:
        tasks.append(asyncio.create_task(fetch_raydium_tokens(session, solana_client, max_pairs=raydium_max_pairs))); labels.append("ray")
    if birdeye_enabled:
        tasks.append(asyncio.create_task(fetch_birdeye_tokens(session, solana_client, max_tokens=birdeye_max_tokens))); labels.append("bird")

    results = await asyncio.gather(*tasks, return_exceptions=True)
    resmap = {labels[i]: results[i] for i in range(len(labels))}

    def _as_list(name: str) -> List[Dict[str, Any]]:
        r = resmap.get(name)
        if isinstance(r, Exception):
            logger.warning("%s fetch failed: %s", name.upper(), r)
            return []
        if r is None:
            return []
        if isinstance(r, list):
            return r
        try:
            return list(r)
        except Exception:
            return []

    # collect/flatten dex results across all queries
    dex: List[Dict[str, Any]] = []
    for lbl in labels:
        if lbl.startswith("dex:"):
            dex.extend(_as_list(lbl))

    ray  = _as_list("ray") if raydium_enabled else []
    bird = _as_list("bird") if birdeye_enabled else []

    logger.info(
        "Fetched: Dex=%d (queries=%d), Raydium=%d, Birdeye=%d",
        len(dex), len(dex_queries), len(ray), len(bird)
    )

    # 2) Normalize + merge -----------------------------------------
    def _extract_base_address(p: Dict[str, Any]) -> Optional[str]:
        try:
            candidates = [
                p.get("address"),
                p.get("token_address"),
                p.get("baseMint"),
                (p.get("baseToken") or {}).get("address") if isinstance(p.get("baseToken"), dict) else None,
            ]
            for c in candidates:
                if isinstance(c, str) and c.strip():
                    return c.strip()
        except Exception:
            pass
        return None

    combined: List[Dict[str, Any]] = []
    missing_addr = 0
    for arr in (dex, ray, bird):
        for p in arr or []:
            addr = _extract_base_address(p)
            if not addr:
                missing_addr += 1
                continue
            combined.append({
                "address": addr,
                "symbol": p.get("symbol", "UNKNOWN"),
                "name": p.get("name", "UNKNOWN"),
                "volume_24h": float(p.get("volume_24h", p.get("v24hUSD", 0) or 0)),
                "liquidity": float(p.get("liquidity", 0) or 0),
                "market_cap": float(p.get("market_cap", p.get("mc", 0) or 0)),
                "creation_timestamp": int(p.get("creation_timestamp", 0) or 0),
                "price": float(p.get("price", 0) or 0),
                "price_change_1h": float(p.get("price_change_1h", 0) or 0),
                "price_change_6h": float(p.get("price_change_6h", 0) or 0),
                "price_change_24h": float(p.get("price_change_24h", 0) or 0),
                "categories": p.get("categories", []),
                "links": p.get("links", []),
                "source": p.get("source", "unknown"),
            })

    pre_total = len(dex) + len(ray) + len(bird)
    logger.info(
        "Pre-dedupe counts: dex=%d ray=%d bird=%d total=%d, usable_with_address=%d, missing_address=%d",
        len(dex), len(ray), len(bird), pre_total, len(combined), missing_addr
    )

    merged = _merge_best_by_address(combined)
    logger.info("After merge_by_address: %d", len(merged))
    # ---- Do-not-trade filter (stables/WSOL/etc.) BEFORE eligibility/RugCheck ----
    # We treat "whitelist" here as "do-not-trade": exclude from trading & skip RugCheck.
    def _addr_lower(t: Dict[str, Any]) -> str:
        return (t.get("address") or t.get("token_address") or t.get("mint") or "").strip().lower()

    # Read addresses from config (preferred) and union with canonical stables/WSOL
    wl_cfg = set(
        str(a).strip().lower()
        for a in (
            disc_cfg.get("whitelist", {}).get("addresses", [])
            or config.get("whitelist", {}).get("addresses", [])
            or disc_cfg.get("do_not_trade", {}).get("addresses", [])
            or config.get("do_not_trade", {}).get("addresses", [])
            or []
        )
        if a
    )
    # Also include the built-in hide set (USDC/USDT/WSOL, etc.)
    builtin_hide = {str(a).strip().lower() for a in (ALWAYS_HIDE_IN_CATEGORIES or [])}

    DO_NOT_TRADE = wl_cfg | builtin_hide

    if DO_NOT_TRADE:
        pre_len = len(merged)
        merged = [t for t in merged if _addr_lower(t) not in DO_NOT_TRADE]
        removed = pre_len - len(merged)
        if removed:
            logger.info(
                "Do-not-trade filter removed %d token(s) prior to eligibility (RugCheck skipped for these).",
                removed,
            )

    # Mid-cap probe (optional)
    def _nprobe(x, d=0.0):
        try:
            return float(x if x is not None else d)
        except Exception:
            return float(d)

    mid_caps = [t for t in merged if 100_000 <= _nprobe(t.get("market_cap") or t.get("mc"), 0.0) < 500_000]
    logger.info("MID-CAP candidates this cycle: %d", len(mid_caps))
    if mid_caps:
        mid_caps.sort(key=lambda x: (_nprobe(x.get("liquidity"), 0.0), _nprobe(x.get("volume_24h"), 0.0)), reverse=True)
        m = mid_caps[0]
        logger.info("Example MID-CAP: %s (%s) mc=%s liq=%s vol24h=%s src=%s",
                    m.get("symbol"), m.get("address"),
                    (m.get("market_cap") or m.get("mc")),
                    m.get("liquidity"), m.get("volume_24h"),
                    m.get("source"))

    # 3) Price enrichment (best-effort)
    if merged:
        try:
            # Use keyword args to avoid accidental positional-arg misbinding
            await enrich_tokens_with_price_change(tokens=merged, session=session)
        except Exception as e:
            logger.exception("Price enrichment skipped (error): %s", e)

    # 4) Eligibility (concurrent)
    eligible = await _concurrent_eligibility(
        merged, session=session, config=config, concurrency=ELIGIBILITY_CONCURRENCY
    )
    logger.info("Eligible (pre-shortlist): %d", len(eligible))

    # ---- MANDATORY: reason summary (merged -> eligible) ----
    # Shows which rules are deleting candidates when Eligible==0.
    try:
        log_filter_summary(merged, eligible, config, logger, label="pre-shortlist")
    except Exception as e:
        logger.debug("log_filter_summary(pre-shortlist) failed silently: %s", e)

    # Helper to label buckets when the selector returns a flat list
    def _classify_bucket(t: Dict[str, Any]) -> str:
        cats = t.get("categories") or []
        if not isinstance(cats, (list, set, tuple)):
            cats = [cats]
        cats_l = {str(c).lower() for c in cats}
        # explicit category beats MC heuristic
        if "new" in cats_l:
            return "new"
        if "large" in cats_l or "high" in cats_l or "big" in cats_l:
            return "high"
        if "mid" in cats_l or "medium" in cats_l:
            return "mid"
        if "low" in cats_l or "small" in cats_l:
            return "low"
        # heuristic by MC
        mc = _nprobe(t.get("market_cap") or t.get("mc"), 0.0)
        if mc >= 500_000:
            return "high"
        if mc >= 100_000:
            return "mid"
        return "low"

    # 5) Shortlist (top N per category)
    disc_cfg = config.get("discovery", {}) or {}
    per_bucket = int(disc_cfg.get("shortlist_per_bucket", SHORTLIST_PER_CATEGORY))
    raw_shortlist = select_top_five_per_category(eligible, per_bucket=per_bucket, blacklist=set())

    # Flatten + count per bucket robustly
    bucket_counts = {"high": 0, "mid": 0, "low": 0, "new": 0}
    shortlist: List[Dict[str, Any]] = []

    if isinstance(raw_shortlist, dict):
        # Accept both naming styles: high/large, low/small
        def _extend(bucket_key: str, alt_key: str | None = None):
            src = (raw_shortlist.get(bucket_key) or []) + (raw_shortlist.get(alt_key, []) if alt_key else [])
            for t in src or []:
                t["_bucket"] = "high" if bucket_key in ("high", "large") else \
                               "low"  if bucket_key in ("low", "small") else bucket_key
                shortlist.append(t)
            bkey = "high" if bucket_key in ("high", "large") else \
                   "low"  if bucket_key in ("low", "small") else bucket_key
            bucket_counts[bkey] += len(src or [])
        _extend("high", "large")
        _extend("mid")
        _extend("low", "small")
        _extend("new")
    elif isinstance(raw_shortlist, list):
        for t in raw_shortlist:
            b = _classify_bucket(t)
            t["_bucket"] = b
            bucket_counts[b] = bucket_counts.get(b, 0) + 1
            shortlist.append(t)
    else:
        shortlist = []

    # ---- OPTIONAL: reason summary (eligible -> shortlist) ----
    # Confirms if bucket limits/top-N dropped otherwise-eligible tokens.
    try:
        log_filter_summary(eligible, shortlist, config, logger, label="post-bucket-shortlist")
    except Exception as e:
        logger.debug("log_filter_summary(post-bucket-shortlist) failed silently: %s", e)

    logger.info(
        "Shortlist (final): total=%d (per_bucket=%d) | buckets -> high=%d mid=%d low=%d new=%d",
        len(shortlist), per_bucket,
        bucket_counts.get("high", 0),
        bucket_counts.get("mid", 0),
        bucket_counts.get("low", 0),
        bucket_counts.get("new", 0),
    )

    # 6) Persist (shortlist view + history view)
    try:
        written = await save_eligible_tokens(shortlist)
        logger.info("Persisted %d tokens into eligible_tokens (shortlist view)", int(written))
    except Exception as e:
        logger.error("Failed to persist shortlist into eligible_tokens: %s", e, exc_info=True)

    # Use DB layer to resolve the correct path per-OS
    try:
        async with connect_db() as db:  # no explicit path => normalized inside DB module
            await ensure_shortlist_tokens_schema(db)
            await bulk_upsert_tokens(db, "shortlist_tokens", shortlist)
    except Exception as e:
        logger.warning("Failed to upsert shortlist_tokens (history view): %s", e)

    return merged, shortlist


# ----------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Local effective flags helper (config + env) to avoid NameError
# ----------------------------------------------------------------
# Stickiness so flags never get *lowered* during runtime (your requirement)
_STICKY_FLAGS = {"dry_run": False, "simulate": False, "send_tx_forced_off": False}

def _local_effective_flags(config):
    """Return (enabled, dry_run, simulate, send_tx) from config.yaml + environment.
    Guaranteed monotonic: once dry_run/simulate are True, they stay True for the process.
    If send_tx is ever False (by config or env), it stays False.
    """
    tcfg = (config or {}).get("trading", {}) or {}
    enabled = bool(tcfg.get("enabled") or tcfg.get("enable"))
    # Base from config
    dry_run = bool(tcfg.get("dry_run"))
    simulate = bool(tcfg.get("simulate"))
    send_tx = bool(tcfg.get("send_transactions"))
    # Env overlays
    import os as _os
    env_dry = _os.getenv("DRY_RUN", "0") == "1"
    env_quote = _os.getenv("JUPITER_QUOTE_ONLY", "0") == "1"
    env_disable_send = _os.getenv("DISABLE_SEND_TX", "0") == "1"
    env_execute = _os.getenv("JUPITER_EXECUTE", "0") == "1"

    dry_run = dry_run or env_dry or env_quote
    simulate = simulate or env_dry or env_quote
    # send_tx is false if disabled OR quote-only
    if env_disable_send or env_quote:
        send_tx = False
    # If explicitly executing and not quote-only, allow simulate False (live) if send_tx True
    if env_execute and not env_quote and send_tx:
        simulate = False

    # --- stickiness (never lower) ---
    dry_run = dry_run or _STICKY_FLAGS["dry_run"]
    simulate = simulate or _STICKY_FLAGS["simulate"]
    if not send_tx:
        _STICKY_FLAGS["send_tx_forced_off"] = True
    if _STICKY_FLAGS["send_tx_forced_off"]:
        send_tx = False

    _STICKY_FLAGS["dry_run"] = dry_run
    _STICKY_FLAGS["simulate"] = simulate

    return enabled, dry_run, simulate, send_tx


async def _async_main_loop(config_path: Optional[str] = None) -> None:
    # Resolve config path (prefer ./config.yaml, then appdata yaml/json, then ./config.json)
    if config_path is None:
        config_path = _resolve_config_path()
    logger.debug(f'Using config path: {config_path}')
    import aiohttp
    from solana.rpc.async_api import AsyncClient

    # initial config + logging
    config = load_config(config_path)
    setup_logging(config)
    logger.info("Starting discovery loop")
    enabled, dry_run, simulate, send_tx = _local_effective_flags(config)
    logger.info(
        "TRADING FLAGS: enabled=%s dry_run=%s simulate=%s send_tx=%s | env DRY_RUN=%s DISABLE_SEND_TX=%s JUPITER_EXECUTE=%s JUPITER_QUOTE_ONLY=%s",
        enabled, dry_run, simulate, send_tx,
        os.getenv("DRY_RUN"), os.getenv("DISABLE_SEND_TX"),
        os.getenv("JUPITER_EXECUTE"), os.getenv("JUPITER_QUOTE_ONLY"),
    )

    # Ensure DB schema exists before any reads/writes (creates blacklist, eligible_tokens, etc.)
    await init_db()
    await ensure_eligible_tokens_schema()

    # aiohttp session/client
    timeout = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)
    connector = aiohttp.TCPConnector(limit=AIOHTTP_CONN_LIMIT, enable_cleanup_closed=True)

    sol_client: Optional[AsyncClient] = None
    trading_task: Optional[asyncio.Task] = None
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            rpc_url = _get_rpc_url(config)
            sol_client = AsyncClient(rpc_url)

            # Blacklist maintenance
            await clear_expired_blacklist(24.0)
            await review_blacklist()

            # Signals
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, _request_stop)
                except NotImplementedError:
                    pass  # Windows

            # Clear stop flag if present
            try:
                STOP_FLAG_PATH.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

            # ---- Start trading loop once, if enabled ----
            if enabled and trading_task is None:
                try:
                    # IMPORTANT: this imports your bundle trading entrypoint
                    from solana_trading_bot_bundle.trading_bot.trading import main as trading_main
                except Exception as e:
                    logger.error("Unable to import trading loop (trading.main): %s", e, exc_info=True)
                else:
                    trading_task = asyncio.create_task(trading_main())
                    _, d1, s1, st1 = _local_effective_flags(config)
                    logger.info("Trading loop started (background task). dry_run=%s simulate=%s send_tx=%s", d1, s1, st1)

            # ---- main discovery loop (no per-cycle trading call here) ----
            while not _should_stop():
                try:
                    # Re-read config each iteration so toggling flags in config.yaml takes effect
                    config = load_config(config_path)

                    merged, shortlist = await run_discovery_cycle(session, sol_client, config)

                    # Visibility into each discovery pass (and current trading toggles)
                    tcfg = (config or {}).get("trading", {}) or {}
                    enb, dryy, simm, sendx = _local_effective_flags(config)
                    logger.info("TRADING-CYCLE: shortlist=%d enabled=%s dry_run=%s simulate=%s send=%s",
                        len(shortlist), enb, dryy, simm, sendx)

                except Exception as e:
                    logger.error("Discovery cycle error: %s", e, exc_info=True)

                # sleep with jitter
                jitter = random.uniform(0.0, min(5.0, DISCOVERY_CYCLE_SECONDS * 0.25))
                await asyncio.sleep(DISCOVERY_CYCLE_SECONDS + jitter)

            logger.info("Stop requested; exiting main loop.")

    finally:
        # graceful shutdown of trading background task
        if trading_task:
            try:
                trading_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await trading_task
            except Exception:
                logger.exception("Error while stopping trading task")

        try:
            if sol_client is not None:
                await sol_client.close()
        except Exception:
            pass

def bootstrap() -> None:
    """Optional bootstrap hook expected by the launcher."""
    logger.debug("bootstrap(): no-op")

def main_loop(config_path: Optional[str] = None) -> Optional[asyncio.Task]:
    """Synchronous entry point expected by the launcher.
    If an event loop is running (e.g. from the GUI), schedule the async loop as a Task and return it.
    Otherwise, run it to completion and return None.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        logger.info("Detected running asyncio loop — scheduling main loop as a task.")
        return loop.create_task(_async_main_loop(config_path))
    else:
        logger.info("No running asyncio loop — executing main loop with asyncio.run().")
        asyncio.run(_async_main_loop(config_path))
        return None