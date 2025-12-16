
# scoring_engine.py
"""
SOLO Meme Coin Bot — Scoring Engine
-----------------------------------
Config-driven pre-filter and scoring pipeline to rank tokens BEFORE
applying your entry strategy (e.g., BBands Flip).

- Pure Python (no 3rd-party deps)
- Compatible with your existing token dicts (Dexscreener/Birdeye fields)
- Returns rich breakdowns for GUI
- Honors RugCheck flags (if present) and your momentum guard limits

Usage (quick):
    from scoring_engine import score_tokens, score_one, select_top_n_per_bucket

    scored = score_tokens(tokens, cfg, logger)
    top = select_top_n_per_bucket(scored, n=5)

Expected token fields (best-effort fallbacks applied):
    - address, symbol, name
    - created_at (unix ts) or age_minutes
    - liquidity (USD), mc/market_cap (USD)
    - volume_10m, volume_30m, volume_1h, volume_24h (USD)   # any subset ok
    - price_change_10m, price_change_1h, price_change_6h, price_change_24h (%)
    - social: {"x_mentions_30m": int, "tg_members": int, ...}  # optional
    - rugcheck: {"status": "pass"/"fail"/"warn", "labels": [...], "locked": bool, ...}  # optional
    - categories: ["New", "Low", "Mid", "High"] or synonyms (we normalize)

Config (examples):
    scoring:
      weights:
        age: 0.15
        liquidity: 0.20
        mcap: 0.10
        volume: 0.25
        momentum: 0.15
        social: 0.10
        security: 0.05
      thresholds:
        min_total: 55                 # overall min to pass
        min_liquidity_usd: 50000
        max_mcap_usd: 500000
        min_volume_30m_usd: 10000
      bands:                          # per-cap bucket overrides (optional)
        New:
          min_total: 60
          max_age_minutes: 180
        Low:
          min_total: 58
        Mid:
          min_total: 56
        High:
          min_total: 54
      momentum_guard:                 # mirrors your trading.momentum_guard
        min_price_change_1h: 0.0
        min_price_change_6h: 0.0

Notes:
- All scores are normalized to 0..100 and combined by weights.
- Missing fields degrade gracefully.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------- Utilities ----------

def _now_ts() -> int:
    try:
        return int(time.time())
    except Exception:
        return 0

def _num(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _pct(x, default=0.0) -> float:
    # helpful if a % shows up as "12.3" or "0.123" — assume it's already % units
    return _num(x, default)

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


# ---------- Normalization helpers (0..100) ----------

def _score_age_minutes(age_min: float, sweet_spot_max: float = 180.0, max_ok: float = 1440.0) -> float:
    """
    Newer tends to be better for discovery (but too old still ok).
    0..100 where <= sweet_spot_max gets 100, then linearly decays to 0 by max_ok.
    """
    if age_min <= 0:
        return 100.0
    if age_min <= sweet_spot_max:
        return 100.0
    if age_min >= max_ok:
        return 0.0
    frac = 1.0 - ((age_min - sweet_spot_max) / max(1.0, (max_ok - sweet_spot_max)))
    return 100.0 * _clamp(frac, 0.0, 1.0)

def _score_liquidity(usd: float, min_ok: float = 5_000.0, target: float = 50_000.0, cap: float = 200_000.0) -> float:
    """
    0..100: below min_ok -> 0, at target -> 80, at cap+ -> 100.
    """
    if usd <= 0:
        return 0.0
    if usd < min_ok:
        return 20.0 * (usd / max(1.0, min_ok))  # small uplift
    if usd >= cap:
        return 100.0
    if usd >= target:
        # map [target..cap] -> [80..100]
        frac = (usd - target) / max(1.0, cap - target)
        return 80.0 + 20.0 * _clamp(frac, 0.0, 1.0)
    # map [min_ok..target] -> [20..80]
    frac = (usd - min_ok) / max(1.0, target - min_ok)
    return 20.0 + 60.0 * _clamp(frac, 0.0, 1.0)

def _score_mcap(usd: float, sweet_min: float = 50_000.0, sweet_max: float = 500_000.0) -> float:
    """
    Reward mid microcaps; penalize too tiny and too large.
    0..100: peak around the sweet band; linear falloff outside.
    """
    if usd <= 0:
        return 0.0
    if usd < sweet_min:
        # map [0..sweet_min] -> [0..70]
        return 70.0 * (usd / max(1.0, sweet_min))
    if usd <= sweet_max:
        # map [sweet_min..sweet_max] -> [70..100]
        frac = (usd - sweet_min) / max(1.0, (sweet_max - sweet_min))
        return 70.0 + 30.0 * _clamp(frac, 0.0, 1.0)
    # above sweet band map to [70..20] and tail to 0
    # assume 10x past sweet_max is near 0
    tail_max = sweet_max * 10.0
    if usd >= tail_max:
        return 0.0
    frac = (usd - sweet_max) / max(1.0, (tail_max - sweet_max))
    return 70.0 - 50.0 * _clamp(frac, 0.0, 1.0)

def _score_volume_surge(v10m: float, v30m: float, v1h: float, base_24h: float) -> float:
    """
    Combines short-term surges with baseline liquidity. 0..100.
    - Strong if 10m>5k, 30m>20k, 1h>50k and 24h>100k (example scales).
    """
    # Normalize components
    s10 = _clamp(v10m / 5_000.0, 0.0, 2.0)   # up to 2x
    s30 = _clamp(v30m / 20_000.0, 0.0, 2.0)
    s1h = _clamp(v1h  / 50_000.0, 0.0, 2.0)
    b24 = _clamp(base_24h / 100_000.0, 0.0, 2.0)
    # Weighted blend
    raw = (0.35*s10 + 0.35*s30 + 0.20*s1h + 0.10*b24) / 1.0
    return 100.0 * _clamp(raw / 1.5, 0.0, 1.0)  # tame to 0..100

def _score_momentum(p10m: float, p1h: float, p6h: float, p24h: float) -> float:
    """
    Emphasize near-term > mid-term > daily. Negative hurts sharply.
    """
    comps = []
    for w, p in ((0.45, p10m), (0.30, p1h), (0.15, p6h), (0.10, p24h)):
        if p is None:
            continue
        pv = float(p)
        if pv < 0:
            comps.append(w * (50.0 * (1.0 + pv/100.0)))  # negative compress
        else:
            comps.append(w * min(100.0, pv))             # cap at 100% for scoring
    score = sum(comps)
    # Normalize to 0..100
    return _clamp(score, 0.0, 100.0)

def _score_social(x_mentions_30m: float, tg_members: float, x_growth_rate: float=0.0) -> float:
    """
    Simple social pulse. Optional if your enrichment provides it.
    """
    xm = _clamp(x_mentions_30m / 50.0, 0.0, 2.0)     # 50 mentions -> strong
    tg = _clamp(tg_members / 2000.0, 0.0, 2.0)       # 2k members -> strong
    gr = _clamp(x_growth_rate, 0.0, 2.0)
    raw = 0.6*xm + 0.3*tg + 0.1*gr
    return 100.0 * _clamp(raw / 1.5, 0.0, 1.0)

def _score_security(status: str, labels: List[str], locked: Optional[bool]) -> float:
    """
    RugCheck-style signal. Pass=90, Warn=60, Fail=0. Labels may reduce.
    """
    s = (status or "").lower()
    bad = {"dangerous","scam","honeypot","malicious","blocked"}
    penalty = 0.0
    for lbl in labels or []:
        if str(lbl).lower() in bad:
            penalty += 30.0
    if s == "pass":
        base = 90.0
    elif s == "warn":
        base = 60.0
    elif s == "fail":
        base = 0.0
    else:
        base = 50.0  # unknown
    if locked is False:
        penalty += 10.0
    return _clamp(base - penalty, 0.0, 100.0)


# ---------- Dataclasses ----------

@dataclass
class ScoreBreakdown:
    age: float = 0.0
    liquidity: float = 0.0
    mcap: float = 0.0
    volume: float = 0.0
    momentum: float = 0.0
    social: float = 0.0
    security: float = 0.0
    total: float = 0.0
    passed: bool = False
    reasons: List[str] = None  # why failed or notes

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reasons"] = self.reasons or []
        return d


# ---------- Core Scoring ----------

DEFAULT_WEIGHTS = {
    "age": 0.15,
    "liquidity": 0.20,
    "mcap": 0.10,
    "volume": 0.25,
    "momentum": 0.15,
    "social": 0.10,
    "security": 0.05,
}

def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    w = dict(DEFAULT_WEIGHTS)
    if isinstance(weights, dict):
        w.update({k: float(v) for k, v in weights.items() if k in DEFAULT_WEIGHTS})
    s = sum(w.values()) or 1.0
    return {k: v/s for k, v in w.items()}

def _category_alias(cat: str) -> str:
    if not cat:
        return "Unknown"
    c = str(cat).lower()
    if c in ("high","large","big","hc","lc","large cap","high cap"):
        return "High"
    if c in ("mid","medium","mc","mid cap","medium cap"):
        return "Mid"
    if c in ("low","small","sc","low cap","small cap"):
        return "Low"
    if c in ("new","fresh","recent"):
        return "New"
    return cat

def _compute_age_minutes(token: Dict[str, Any]) -> float:
    age = _num(token.get("age_minutes"))
    if age > 0:
        return age
    created = _num(token.get("created_at"), 0.0)
    if created > 0:
        return max(0.0, (_now_ts() - created) / 60.0)
    return 0.0

def _momentum_guard_ok(cfg: Dict[str, Any], token: Dict[str, Any]) -> Tuple[bool, str]:
    mg = ((cfg.get("scoring") or {}).get("momentum_guard")) or ((cfg.get("trading") or {}).get("momentum_guard")) or {}
    min_1h = _num(mg.get("min_price_change_1h", 0.0))
    min_6h = _num(mg.get("min_price_change_6h", 0.0))
    pc1h = _pct(token.get("price_change_1h", token.get("priceChange1h")))
    pc6h = _pct(token.get("price_change_6h", token.get("priceChange6h")))
    if pc1h < min_1h:
        return False, f"pc1h<{min_1h}"
    if pc6h < min_6h:
        return False, f"pc6h<{min_6h}"
    return True, "ok"

def score_one(token: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    scfg = (cfg.get("scoring") or {})
    weights = _normalize_weights(scfg.get("weights") or {})
    thr = scfg.get("thresholds") or {}
    bands = scfg.get("bands") or {}

    reasons: List[str] = []

    # Pre-cut checks
    liq = _num(token.get("liquidity", token.get("liq")))
    mc  = _num(token.get("mc", token.get("market_cap")))
    v30 = _num(token.get("volume_30m", _safe_get(token, "volume", "30m", default=0.0)))
    pc10m = _pct(token.get("price_change_10m", token.get("priceChange10m")))
    pc1h  = _pct(token.get("price_change_1h",  token.get("priceChange1h")))
    pc6h  = _pct(token.get("price_change_6h",  token.get("priceChange6h")))
    pc24h = _pct(token.get("price_change_24h", token.get("priceChange24h")))

    min_liq = _num(thr.get("min_liquidity_usd", 0.0))
    max_mc  = _num(thr.get("max_mcap_usd", 0.0))
    min_v30 = _num(thr.get("min_volume_30m_usd", 0.0))

    if min_liq and liq < min_liq:
        reasons.append(f"liquidity<{min_liq}")
    if max_mc and mc > max_mc > 0:
        reasons.append(f"mcap>{max_mc}")
    if min_v30 and v30 < min_v30:
        reasons.append(f"vol30m<{min_v30}")

    # Momentum guard (soft pre-check; we don't hard-fail here—weighting will penalize)
    ok_mg, mg_reason = _momentum_guard_ok(cfg, token)
    if not ok_mg:
        reasons.append(f"momentum_guard:{mg_reason}")

    # Scores
    age_min = _compute_age_minutes(token)
    age_s = _score_age_minutes(age_min,
                               sweet_spot_max=_num(_safe_get(bands, "New", "max_age_minutes"), 180.0),
                               max_ok=1440.0)

    liq_s = _score_liquidity(liq,
                             min_ok=_num(thr.get("min_liquidity_usd", 5_000.0)),
                             target=max(_num(thr.get("min_liquidity_usd", 5_000.0)) * 10.0, 50_000.0),
                             cap=200_000.0)

    mcap_s = _score_mcap(mc,
                         sweet_min=max(10_000.0, _num(thr.get("min_liquidity_usd", 5_000.0))),  # roughly relate to liq
                         sweet_max=_num(thr.get("max_mcap_usd", 500_000.0)) or 500_000.0)

    v10 = _num(token.get("volume_10m", 0.0))
    v1h = _num(token.get("volume_1h", token.get("volume1h", 0.0)))
    v24 = _num(token.get("volume_24h", token.get("volume24h", 0.0)))
    vol_s = _score_volume_surge(v10, v30, v1h, v24)

    mom_s = _score_momentum(pc10m, pc1h, pc6h, pc24h)

    # Social & Security
    x_m = _num(_safe_get(token, "social", "x_mentions_30m"), 0.0)
    tg  = _num(_safe_get(token, "social", "tg_members"), 0.0)
    soc_s = _score_social(x_m, tg)

    rstatus = _safe_get(token, "rugcheck", "status") or ""
    rlabels = _safe_get(token, "rugcheck", "labels") or []
    rlocked = _safe_get(token, "rugcheck", "locked")
    sec_s = _score_security(str(rstatus), list(rlabels) if isinstance(rlabels, list) else [], rlocked if isinstance(rlocked, bool) else None)

    # Weighted sum
    total = (
        weights["age"]       * age_s +
        weights["liquidity"] * liq_s +
        weights["mcap"]      * mcap_s +
        weights["volume"]    * vol_s +
        weights["momentum"]  * mom_s +
        weights["social"]    * soc_s +
        weights["security"]  * sec_s
    ) * 100.0  # optional amplifier to expand range (comment out if too large)

    # bring back to 0..100
    total = _clamp(total / 100.0, 0.0, 100.0)

    # Pass/fail thresholds
    cat = _category_alias((token.get("categories") or token.get("category") or "Unknown"))
    if isinstance(cat, list):
        cat = cat[0] if cat else "Unknown"
    min_total = _num(thr.get("min_total", 55.0))
    min_total = _num(_safe_get(bands, cat, "min_total"), min_total)

    passed = total >= min_total and (len([r for r in reasons if r.startswith("liquidity<")]) == 0)

    breakdown = ScoreBreakdown(
        age=age_s, liquidity=liq_s, mcap=mcap_s, volume=vol_s,
        momentum=mom_s, social=soc_s, security=sec_s,
        total=total, passed=passed, reasons=reasons
    )
    out = {
        **token,
        "_score": breakdown.as_dict(),
        "_score_category": cat,
        "_score_min_total": min_total,
    }
    return out

def score_tokens(tokens: List[Dict[str, Any]], cfg: Dict[str, Any], logger=None) -> List[Dict[str, Any]]:
    out = []
    for t in tokens or []:
        try:
            out.append(score_one(t, cfg))
        except Exception as e:
            if logger:
                logger.debug("Scoring failed for %s: %s", t.get("symbol") or t.get("address"), e, exc_info=True)
    # Sort: passed first by total desc, then failed by total desc
    def _key(x):
        s = _safe_get(x, "_score", "total", default=0.0)
        p = bool(_safe_get(x, "_score", "passed", default=False))
        return (1 if p else 0, s)
    out.sort(key=_key, reverse=True)
    return out

def select_top_n_per_bucket(scored: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    buckets = {"New": [], "Low": [], "Mid": [], "High": [], "Unknown": []}
    for t in scored:
        cat = _safe_get(t, "_score_category", default="Unknown") or "Unknown"
        if cat not in buckets:
            buckets[cat] = []
        buckets[cat].append(t)
    final: List[Dict[str, Any]] = []
    for cat, items in buckets.items():
        # keep passed first
        passed = [x for x in items if _safe_get(x, "_score", "passed", default=False)]
        failed = [x for x in items if not _safe_get(x, "_score", "passed", default=False)]
        final.extend(passed[:n])
        if len(passed) < n and failed:
            # optionally top-up with highest failed to maintain N rows in GUI
            final.extend(failed[: (n - len(passed)) ])
    return final
