import os
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

def _json_candidate_from_dbpath(sqlite_path: str) -> str:
    return str(Path(sqlite_path).with_name("token_cache.json"))

def dump_token_cache_json(tokens: List[Dict[str, Any]], out_path: str) -> None:
    """
    Synchronous helper that writes token_cache.json atomically.
    Use asyncio.to_thread(...) around this when calling from async code.
    """
    payload = {"tokens": tokens, "updated_at": int(time.time())}
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # atomic replace
    os.replace(tmp, out_path)

async def load_shortlist_from_json(path: str) -> Optional[List[Dict[str, Any]]]:
    try:
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        # run sync I/O in thread to avoid blocking loop
        def _read():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        data = await asyncio.to_thread(_read)
        if isinstance(data, dict):
            tokens = data.get("tokens") or data.get("items") or []
            if isinstance(tokens, list):
                return tokens
            return None
        if isinstance(data, list):
            return data
        return None
    except Exception:
        return None

async def load_shortlist_from_sqlite(max_rows: int = 500) -> Optional[List[Dict[str, Any]]]:
    """
    Query the canonical DB via the database module. Returns list or None on error.
    """
    try:
        # Import here to avoid circular import at module load time
        from solana_trading_bot_bundle.trading_bot import database
        # database.list_eligible_tokens is async
        tokens = await database.list_eligible_tokens(limit=max_rows)
        return tokens or []
    except Exception:
        return None

async def load_shortlist_compat(max_rows: int = 500, prefer_json: bool = True) -> List[Dict[str, Any]]:
    """
    Prefers JSON token_cache.json if present and parseable, otherwise falls back to SQLite.
    Returns a list (possibly empty).
    """
    # try to resolve sqlite path via database helper if available
    sqlite_path = None
    try:
        from solana_trading_bot_bundle.trading_bot import database
        sqlite_path = database._cached_resolve_db_path()
    except Exception:
        sqlite_path = None

    json_candidate = None
    if sqlite_path:
        json_candidate = _json_candidate_from_dbpath(sqlite_path)
    else:
        # fallback conventional location (per-user AppData)
        try:
            base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
            base = os.path.join(base, "SOLOTradingBot")
            json_candidate = os.path.join(base, "token_cache.json")
        except Exception:
            json_candidate = None

    # 1) optional JSON path first
    if prefer_json and json_candidate:
        tokens = await load_shortlist_from_json(json_candidate)
        if tokens:
            return tokens

    # 2) sqlite fallback
    tokens = await load_shortlist_from_sqlite(max_rows=max_rows)
    if tokens:
        # attempt to write a compatible token_cache.json so legacy readers find it
        if json_candidate:
            try:
                # write in thread
                await asyncio.to_thread(dump_token_cache_json, tokens, json_candidate)
            except Exception:
                # don't fail on write error; just return tokens list
                pass
        return tokens

    # 3) none found
    return []