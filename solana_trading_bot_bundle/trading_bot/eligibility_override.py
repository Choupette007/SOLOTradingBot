# New/updated helper: should_override_extreme_price
# This helper is synchronous (blocking): it performs any necessary DB writes and uses requests
# (or other blocking I/O). It returns a tuple: (allowed: bool, detail: dict).

import sqlite3
import time
import json
from typing import Dict, Any, Tuple, Optional

# Prefer the small local dexscreener client if present; fallback to requests if not.
try:
    # local relative import (when running as package module)
    from .dexscreener_client import best_pair_summary  # type: ignore
except Exception:
    try:
        # packaged bundle import path
        from solana_trading_bot_bundle.trading_bot.dexscreener_client import best_pair_summary  # type: ignore
    except Exception:
        best_pair_summary = None  # type: ignore

# Keep a light requests fallback if the dexscreener client isn't available.
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

# Keep the audit table creation idempotent so helper can create it if missing
def _ensure_audit_table(conn: sqlite3.Connection) -> None:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS override_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                address TEXT NOT NULL,
                allowed INTEGER NOT NULL,
                reason TEXT,
                explanation TEXT,
                source TEXT,
                evidence_json TEXT,
                price_change_24h REAL,
                volume_24h_usd REAL,
                liquidity_usd REAL,
                rugcheck_score REAL
            );
            """
        )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def _insert_audit_row(
    conn: sqlite3.Connection,
    address: str,
    allowed: bool,
    reason: Optional[str],
    explanation: Optional[str],
    source: Optional[str],
    evidence: Optional[dict],
    price_change_24h: float,
    volume_24h_usd: float,
    liquidity_usd: float,
    rugcheck_score: float,
) -> None:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO override_audit
            (ts,address,allowed,reason,explanation,source,evidence_json,price_change_24h,volume_24h_usd,liquidity_usd,rugcheck_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(time.time()),
                address,
                1 if allowed else 0,
                reason or None,
                explanation or None,
                source or None,
                json.dumps(evidence or {}) if evidence is not None else None,
                float(price_change_24h or 0.0),
                float(volume_24h_usd or 0.0),
                float(liquidity_usd or 0.0),
                float(rugcheck_score or 0.0),
            ),
        )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def _query_dexscreener_for_pair(address: str) -> Optional[Dict[str, Any]]:
    """
    Lightweight blocking probe for Dexscreener pair evidence.

    Implementation notes:
    - Prefer the bundled/local dexscreener_client.best_pair_summary (fast, small JSON).
    - If that isn't available, fall back to a minimal requests GET to the same endpoint.
    - Return a compact evidence dict (keys: source, pair, pair_url, liquidity, volume_24h)
      or None if nothing useful found.
    """
    # Preferred fast path: use best_pair_summary helper if present
    try:
        if callable(best_pair_summary):
            try:
                summary = best_pair_summary(address)  # returns {"pairs": [...], "best_volume_h24": ..., "best_liquidity_usd": ...}
                if not summary:
                    return None
                best_liq = float(summary.get("best_liquidity_usd") or 0.0)
                best_vol = float(summary.get("best_volume_h24") or 0.0)
                # pick first pair entry for URL/pairAddress if available (summaries list is small)
                pairs = summary.get("pairs") or []
                first = pairs[0] if isinstance(pairs, (list, tuple)) and pairs else {}
                evidence = {
                    "source": "dexscreener_client",
                    "pair": first.get("pairAddress") or "",
                    "pair_url": first.get("url") or first.get("pair_url") or "",
                    "liquidity": float(first.get("liquidity_usd") or best_liq or 0.0),
                    "volume_24h": float(first.get("volume_h24") or best_vol or 0.0),
                }
                # If both values zero, treat as no evidence
                if (evidence["liquidity"] or 0.0) <= 0.0 and (evidence["volume_24h"] or 0.0) <= 0.0:
                    return None
                return evidence
            except Exception:
                # fall through to requests fallback if available
                pass
        # Fallback: direct requests call (if requests available)
        if requests is None:
            return None
        DEX_URL = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
        resp = requests.get(DEX_URL, timeout=6, headers={"User-Agent": "SOLOTradingBot/1.0", "Accept": "application/json"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        pairs = data.get("pairs") or []
        best = None
        best_score = 0.0
        for p in pairs:
            try:
                vol = float((p.get("volume") or {}).get("h24", 0) or 0)
                liq = float(((p.get("liquidity") or {}).get("usd") or 0) or 0)
                score = vol + liq
                if score > best_score:
                    best_score = score
                    best = p
            except Exception:
                continue
        if not best:
            return None
        evidence = {
            "source": "dexscreener",
            "pair": best.get("pairAddress") or "",
            "pair_url": best.get("url") or best.get("dexscreenerUrl") or "",
            "liquidity": float(((best.get("liquidity") or {}).get("usd") or 0) or 0),
            "volume_24h": float(((best.get("volume") or {}).get("h24") or 0) or 0),
        }
        if (evidence["liquidity"] or 0.0) <= 0.0 and (evidence["volume_24h"] or 0.0) <= 0.0:
            return None
        return evidence
    except Exception:
        return None


def should_override_extreme_price(
    conn: sqlite3.Connection,
    address: str,
    price_change_24h: float,
    volume_24h: float,
    liquidity: float,
    rugcheck_ok: bool,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether to override an extreme price-change rejection for `address`.
    This is a synchronous helper intended to be run on a background thread (not the asyncio loop).
    It MUST be passed a sqlite3.Connection that is usable in the current thread (i.e., created in that thread).

    Returns:
      (allowed: bool, detail: dict)

    detail keys (suggested):
      - reason: short machine-friendly token e.g. "dexscreener_depth_ok"
      - explanation: human text
      - evidence: dict with numeric fields & small urls
      - source: "dexscreener" | "cached_metrics" | "none"
      - rugcheck_score: float (if available)
    """
    detail: Dict[str, Any] = {"reason": "unknown", "explanation": "", "evidence": {}, "source": None, "rugcheck_score": None}
    allowed = False

    try:
        # Ensure audit table exists (idempotent)
        _ensure_audit_table(conn)
    except Exception:
        pass

    try:
        # Quick heuristic gates: require some depth or volume and rugcheck_ok
        try:
            vol = float(volume_24h or 0.0)
        except Exception:
            vol = 0.0
        try:
            liq = float(liquidity or 0.0)
        except Exception:
            liq = 0.0

        # If rugcheck explicitly failed, do not override
        if not bool(rugcheck_ok):
            detail.update({
                "reason": "rugcheck_failed",
                "explanation": "RugCheck indicates high risk or API inaccessible",
                "source": "rugcheck",
                "evidence": {"rugcheck_ok": rugcheck_ok},
            })
            allowed = False
            _insert_audit_row(conn, address, allowed, detail.get("reason"), detail.get("explanation"), detail.get("source"),
                              detail.get("evidence"), float(price_change_24h or 0.0), vol, liq, float(detail.get("rugcheck_score") or 0.0))
            return allowed, detail

        # 1) Prefer cached internal metrics if present
        try:
            cur = conn.cursor()
            cur.execute("SELECT liquidity, volume_24h, note FROM override_cached_metrics WHERE address = ? LIMIT 1;", (address,))
            row = cur.fetchone()
            if row:
                try:
                    c_liq = float(row[0] or 0.0)
                    c_vol = float(row[1] or 0.0)
                except Exception:
                    c_liq, c_vol = 0.0, 0.0
                # If cached metrics show good depth, accept
                if c_liq >= 5000.0 or c_vol >= 10000.0:
                    detail.update({
                        "reason": "cached_metrics_depth_ok",
                        "explanation": "Cached metrics show sufficient liquidity/volume",
                        "source": "cached_metrics",
                        "evidence": {"cached_liquidity": c_liq, "cached_volume_24h": c_vol, "note": row[2]},
                    })
                    allowed = True
                    _insert_audit_row(conn, address, allowed, detail.get("reason"), detail.get("explanation"), detail.get("source"),
                                      detail.get("evidence"), float(price_change_24h or 0.0), c_vol, c_liq, float(detail.get("rugcheck_score") or 0.0))
                    return allowed, detail
        except Exception:
            # ignore DB errors and continue to Dexscreener probe
            pass

        # 2) Probe Dexscreener for real-time pair depth evidence (fast, blocking)
        evidence = _query_dexscreener_for_pair(address)
        if evidence:
            ev_liq = float(evidence.get("liquidity") or 0.0)
            ev_vol = float(evidence.get("volume_24h") or 0.0)

            # thresholds — align to config in caller if desired; use conservative defaults here
            if (ev_liq >= 5000.0 or ev_vol >= 10000.0):
                detail.update({
                    "reason": "dexscreener_pair_depth_ok",
                    "explanation": f"Dexscreener shows pair with liquidity={ev_liq} and vol24={ev_vol}",
                    "source": evidence.get("source") or "dexscreener",
                    "evidence": evidence,
                })
                allowed = True
            else:
                detail.update({
                    "reason": "dexscreener_pair_depth_insufficient",
                    "explanation": "Dexscreener pair found but liquidity/volume below thresholds",
                    "source": evidence.get("source") or "dexscreener",
                    "evidence": evidence,
                })
                allowed = False
            _insert_audit_row(conn, address, allowed, detail.get("reason"), detail.get("explanation"), detail.get("source"),
                              detail.get("evidence"), float(price_change_24h or 0.0), ev_vol, ev_liq, float(detail.get("rugcheck_score") or 0.0))
            return allowed, detail

        # 3) Last-resort: allow override only if on-chain provided volume/liquidity (passed args)
        if liq >= 5000.0 or vol >= 10000.0:
            detail.update({
                "reason": "passed_metrics_ok",
                "explanation": "Supplied liquidity/volume args exceed thresholds",
                "source": "supplied_metrics",
                "evidence": {"liquidity": liq, "volume_24h": vol},
            })
            allowed = True
        else:
            detail.update({
                "reason": "no_evidence_depth",
                "explanation": "No dexscreener pair and supplied metrics below thresholds",
                "source": "none",
                "evidence": {"liquidity": liq, "volume_24h": vol},
            })
            allowed = False

        _insert_audit_row(conn, address, allowed, detail.get("reason"), detail.get("explanation"), detail.get("source"),
                          detail.get("evidence"), float(price_change_24h or 0.0), vol, liq, float(detail.get("rugcheck_score") or 0.0))
        return allowed, detail
    except Exception as e:
        try:
            detail.update({"reason": "internal_error", "explanation": str(e)})
            try:
                _insert_audit_row(conn, address, False, detail.get("reason"), detail.get("explanation"), "internal", detail.get("evidence"),
                                  float(price_change_24h or 0.0), float(volume_24h or 0.0), float(liquidity or 0.0), float(detail.get("rugcheck_score") or 0.0))
            except Exception:
                pass
        except Exception:
            pass
        return False, detail