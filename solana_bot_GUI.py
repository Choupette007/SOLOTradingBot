# -------------------- top-of-file preamble --------------------
from __future__ import annotations

import os
import time
import re
import sys
import json
import math
from pathlib import Path
from typing import Iterable
from typing import Any, Dict, Optional, Iterable

import streamlit as st
import pandas as pd

# ✅ MUST be the first Streamlit call (before any st.markdown / st.write)
st.set_page_config(
    page_title="SOLO Meme Coin Trading Bot",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Package bootstrap so 'solana_trading_bot_bundle' is importable everywhere ---
# Place this BEFORE any 'from solana_trading_bot_bundle ...' imports.
try:
    _GUI_ROOT = Path(__file__).resolve().parent
    _PKG_ROOT = _GUI_ROOT  # repo root (folder that contains solana_trading_bot_bundle)
    _PKG_DIR = _PKG_ROOT / "solana_trading_bot_bundle"

    # Make repo root importable in THIS process…
    if str(_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PKG_ROOT))

    # …and in ALL child processes (bot subprocess, etc.)
    _existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(_PKG_ROOT)] + ([p for p in _existing.split(os.pathsep) if p] if _existing else [])
    )

    # Ensure subpackages are real packages for the bot’s -m import
    for sub in ("", "utils", "trading_bot", "common"):
        ip = _PKG_DIR / sub / "__init__.py"
        ip.parent.mkdir(parents=True, exist_ok=True)
        if not ip.exists():
            ip.write_text("# package marker\n", encoding="utf-8")
except Exception as _e:
    # logger isn't initialized yet; use print to avoid NameError
    print(f"[package bootstrap] skipped: {_e}", flush=True)

# Safety defaults for app paths and flag files (prevents NameError on import and linter warnings)
import sys as _sys
from pathlib import Path

# Ensure APP_DIR exists early (only if not already provided by the script)
if "APP_DIR" not in globals() or not globals().get("APP_DIR"):
    if os.name == "nt":
        _base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home() / "AppData" / "Local")
        APP_DIR: Path = Path(_base) / "SOLOTradingBot"
    elif _sys.platform == "darwin":
        APP_DIR: Path = Path.home() / "Library" / "Application Support" / "SOLOTradingBot"
    else:
        APP_DIR: Path = Path.home() / ".local" / "share" / "SOLOTradingBot"
else:
    # Coerce existing value to Path
    try:
        APP_DIR = Path(globals().get("APP_DIR"))
    except Exception:
        APP_DIR = Path.home() / ".local" / "share" / "SOLOTradingBot"

# ensure folder exists (non-fatal)
try:
    APP_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# File defaults (coerce existing values to Path, or set sensible defaults)
if "STATUS_FILE" not in globals() or not globals().get("STATUS_FILE"):
    STATUS_FILE: Path = APP_DIR / "rugcheck_status.json"
else:
    STATUS_FILE = Path(globals().get("STATUS_FILE"))

if "FAILURES_FILE" not in globals() or not globals().get("FAILURES_FILE"):
    FAILURES_FILE: Path = APP_DIR / "rugcheck_failures.json"
else:
    FAILURES_FILE = Path(globals().get("FAILURES_FILE"))

# Provide STOP_FLAG_PATH and PID_FILE early so other early code can use them safely
if "STOP_FLAG_PATH" not in globals() or not globals().get("STOP_FLAG_PATH"):
    STOP_FLAG_PATH: Path = APP_DIR / "bot_stop_flag.txt"
else:
    STOP_FLAG_PATH = Path(globals().get("STOP_FLAG_PATH"))

if "PID_FILE" not in globals() or not globals().get("PID_FILE"):
    PID_FILE: Path = APP_DIR / "bot.pid"
else:
    PID_FILE = Path(globals().get("PID_FILE"))

# Insert (or replace) these helper functions near the top of the file,
# after the imports and before any DataFrame / styler usage.

def _fmt_money_for_styler(x) -> str:
    """
    Formatter used with pandas Styler.format for money columns.
    Returns '-' when value is None/NaN, shows more decimals for small prices.
    """
    try:
        if x is None:
            return "-"
        # treat NaN as missing
        if isinstance(x, float) and math.isnan(x):
            return "-"
        if isinstance(x, (int, float)) and abs(float(x)) < 1.0:
            return f"${float(x):,.6f}"
        return f"${float(x):,.0f}"
    except Exception:
        return "-"

def _fmt_pct_for_styler(x) -> str:
    """
    Formatter used with pandas Styler.format for percent columns.
    Returns 'N/A' when value is None/NaN, otherwise formatted +/-.2f%.
    """
    try:
        if x is None:
            return "N/A"
        if isinstance(x, float) and math.isnan(x):
            return "N/A"
        return f"{float(x):+.2f}%"
    except Exception:
        return "N/A"

# --- Import the cosmetic alias helper (safe fallback if constants wasn't updated yet) ---
try:
    from solana_trading_bot_bundle.common import constants as _const
    display_cap = getattr(_const, "display_cap", lambda s: {"High Cap": "Large Cap"}.get(s, s))
except Exception:
    # fallback if package not available at import-time
    display_cap = lambda s: {"High Cap": "Large Cap"}.get(s, s)

# ---------- Sticky Tab (remember + restore across reruns) ----------
DEFAULT_TAB = "⚙️ Bot Control"

if "active_tab" not in st.session_state:
    st.session_state.active_tab = st.query_params.get("tab") or DEFAULT_TAB

def goto_tab(tab_label: str) -> None:
    st.session_state.active_tab = tab_label
    st.query_params["tab"] = tab_label
    wanted_js = json.dumps(tab_label)
    st.markdown(
        f"""
        <script>
          (function(){{
            const wanted = {wanted_js};
            function tryClick(){{
              const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
              for (const t of tabs) {{ if (t.innerText.trim() === wanted) {{ t.click(); return; }} }}
              setTimeout(tryClick, 80);
            }}
            setTimeout(tryClick, 0);
          }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

_wanted = st.session_state.get("active_tab", DEFAULT_TAB)
st.markdown(
    f"""
    <script>
      (function(){{
        const wanted = {json.dumps(_wanted)};
        function tryClick(){{
          const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
          for (const t of tabs) {{ if (t.innerText.trim() === wanted) {{ t.click(); return; }} }}
          setTimeout(tryClick, 80);
        }}
        setTimeout(tryClick, 0);
      }})();
    </script>
    """,
    unsafe_allow_html=True,
)

# ---- Market-cap buckets & helpers (GUI-side) -------------------------------
# >>> Stable/pegged/wrapped/LP detection (GUI-side)
MC_THRESHOLDS = {
    "high_min": 50_000_000,   # >= 50M → High (tune in the GUI)
    "mid_min":    150_000,    # 150K .. < 50M → Mid
}

GUI_HIDE_STABLES = False

# Symbols we always treat as "stable-like" or pegged/wrapped majors.
# Use lowercase; we normalize the incoming symbol to lowercase.
_STABLE_LIKE_SYMBOLS = {
    # fiat-stables
    "usdc", "usdt", "usdt.e", "usdc.e", "pai", "usdd", "dai",
    # solana staking wrappers / common pegs
    "jitosol", "msol", "stsol", "bsol", "lsol",
    # majors (wrapped/pegged) commonly cluttering "High"
    "sol", "wsol", "weth", "wbtc", "eth", "btc",
    # common generic LP symbol
    "lp",
}

# Name substrings that strongly indicate stables, wrapped majors, or LPs.
# All checks are done in lowercase.
_STABLE_NAME_HINTS = (
    # fiat-stable hints
    "usd", "usdc", "usdt", "usdd", "dai", "pai", "tether", "circle",
    # wrapped majors / bridges
    "wrapped btc", "wrapped bitcoin",
    "wrapped eth", "wrapped ether",
    "(wormhole)", "wormhole",
    # staking wrappers / perps / LPs
    "staked sol", "binance staked sol", "jito staked sol",
    "perps", "perps lp", "liquidity provider", "lp token",
)

def _is_stable_like_row(row: dict) -> bool:
    sym = str((row.get("symbol") or "")).strip().lower()
    name = str((row.get("name") or "")).strip().lower()

    # direct symbol blocks
    if sym in _STABLE_LIKE_SYMBOLS:
        return True

    # catch obvious LP names without false positives like "help"/"alpha"
    if " lp" in name or name.endswith(" lp"):
        return True

    # generic name hints
    return any(h in name for h in _STABLE_NAME_HINTS)

def _coerce_float(x, default=0.0) -> float:
    try:
        v = float(x)
        # handle NaN: v != v is True for NaN
        return v if v == v else float(default)
    except Exception:
        return float(default)

def market_cap_bucket(
    mc_like,
    *,
    mid_floor: float | None = None,
    high_floor: float | None = None,
) -> str:
    """
    Returns 'High' | 'Mid' | 'Low' based on thresholds.
    Uses MC_THRESHOLDS by default, but callers can override.
    """
    if high_floor is None:
        high_floor = float(MC_THRESHOLDS["high_min"])
    if mid_floor is None:
        mid_floor = float(MC_THRESHOLDS["mid_min"])

    mc = _coerce_float(mc_like, 0.0)
    if mc <= 0:
        return "Low"  # treat unknown/zero as Low for grouping

    if mc >= high_floor:
        return "High"
    if mc >= mid_floor:
        return "Mid"
    return "Low"

def market_cap_badge(
    mc_like,
    *,
    mid_floor: float | None = None,
    high_floor: float | None = None,
) -> str:
    """
    Badge used in tables. Accepts mc/fdv/strings; graceful on None/NaN.
    Default thresholds come from MC_THRESHOLDS, but can be overridden.
    """
    mc = _coerce_float(mc_like, 0.0)
    if mc <= 0:
        return "⚪"  # unknown/zero

    bucket = market_cap_bucket(mc, mid_floor=mid_floor, high_floor=high_floor)
    return {"High": "🟢 High", "Mid": "🟡 Mid", "Low": "🔴 Low"}[bucket]

# --- Category normalization helper (canonical names used by trading/eligibility) ---
def _normalize_categories(tok: dict) -> None:
    """
    Ensure categories contain canonical names used by trading/eligibility:
      - low_cap, mid_cap, large_cap, newly_launched
    Accept common aliases and map them to canonical names in-place.
    """
    if tok is None:
        return
    cats = tok.get("categories") or []
    if isinstance(cats, str):
        cats = [cats]

    out = set()
    for c in cats:
        if not c:
            continue
        cs = str(c).strip().lower()
        if cs in {"high", "high_cap", "large", "large_cap"}:
            out.add("large_cap")
        elif cs in {"mid", "mid_cap"}:
            out.add("mid_cap")
        elif cs in {"low", "low_cap"}:
            out.add("low_cap")
        elif cs in {"new", "newly_launched", "newlylaunched"}:
            out.add("newly_launched")
        elif cs in {"shortlist", "fallback", "unknown_cap"}:
            out.add(cs)
        else:
            out.add(cs)

    # honor/mirror any explicit bucket hint
    try:
        bk = str(tok.get("_bucket") or "").strip().lower()
        if bk:
            if bk in {"high", "high_cap", "large", "large_cap"}:
                out.add("large_cap")
            elif bk in {"mid", "mid_cap"}:
                out.add("mid_cap")
            elif bk in {"low", "low_cap"}:
                out.add("low_cap")
            elif bk in {"new", "newly_launched"}:
                out.add("newly_launched")
    except Exception:
        pass

    tok["categories"] = list(out)

# ---------- UI state defaults ----------
if "disc_auto_refresh" not in st.session_state:
    st.session_state.disc_auto_refresh = False
if "disc_refresh_interval_ms" not in st.session_state:
    st.session_state.disc_refresh_interval_ms = 60_000
if "last_token_refresh" not in st.session_state:
    st.session_state.last_token_refresh = 0.0

def _pct_or_none(v):
    """Return a float percent if numeric, else None.
    Accepts either a percent (e.g. 3.12 or 0.77 meaning 0.77%) or a fraction
    (e.g. 0.0312 meaning 3.12%). Only normalize fractions when the original
    textual form clearly indicates a fraction (e.g. starts with "0." or ".")."""
    try:
        s = str(v).strip()
        if not s:
            return None
        has_percent_sign = s.endswith("%")
        if has_percent_sign:
            s = s[:-1].strip()
        x = float(s)
        if x != x:  # NaN
            return None
    except Exception:
        return None

    s_l = str(v).strip()
    if s_l.startswith(".") or s_l.startswith("0."):
        if abs(x) > 0.0:
            return x * 100.0
    return x

def bind_discovery_autorefresh(label: str = "Enable Auto-Refresh (every 60 seconds)") -> None:
    st.checkbox(label, key="disc_auto_refresh")
    if st.session_state.disc_auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore
            st_autorefresh(interval=st.session_state.disc_refresh_interval_ms, key="disc_auto_tick")
        except Exception:
            if not st.session_state.get("_disc_tick_warned"):
                st.info("Auto-refresh active. For precise intervals, install: pip install streamlit-autorefresh==1.0.1")
                st.session_state["_disc_tick_warned"] = True

# ---------- DataFrame styling + numeric right alignment + uniform buttons ----------
# IMPORTANT: delete any older CSS blocks; keep ONLY this one.
st.markdown(
    """
<style>
/* 1) Stable table layout */
div[data-testid="stDataFrame"] table { table-layout: fixed; width: 100%; }

/* Truncation + allow inner wrappers to shrink */
div[data-testid="stDataFrame"] thead th,
div[data-testid="stDataFrame"] tbody td {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  box-sizing: border-box;
}
div[data-testid="stDataFrame"] thead th > div,
div[data-testid="stDataFrame"] tbody td > div { min-width: 0 !important; }

/* Header look */
div[data-testid="stDataFrame"] thead th { text-align: center; font-weight: 800; }
div[data-testid="stDataFrame"] thead th > div { display: flex; justify-content: center; }

/* Default body alignment (NO !important) so per-column rules can win */
div[data-testid="stDataFrame"] tbody td { text-align: left; }

/* 2) Right-align numeric columns.
   st.dataframe wraps each cell in a flex container; we right-align by pushing
   content to the end of that flex container.
   Column order (no index): 1 Name | 2 Token Address | 3 Dex | 4 Safety | 5 Price | 6 Liquidity | 7 Market Cap | 8 Volume (24h) | 9 1H | 10 6H | 11 24H
*/
div[data-testid="stDataFrame"] tbody tr td:nth-child(5)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(6)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(7)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(8)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(9)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(10) > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(11) > div {
  justify-content: flex-end !important;
  text-align: right !important;
}

/* With an (accidental) index column, numeric columns shift by +1 (6..12) */
div[data-testid="stDataFrame"] tbody tr td:nth-child(6)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(7)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(8)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(9)  > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(10) > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(11) > div,
div[data-testid="stDataFrame"] tbody tr td:nth-child(12) > div {
  justify-content: flex-end !important;
  text-align: right !important;
}

/* Optional: tighten padding on numeric columns a bit */
div[data-testid="stDataFrame"] tbody tr td:nth-child(5),
div[data-testid="stDataFrame"] tbody tr td:nth-child(6),
div[data-testid="stDataFrame"] tbody tr td:nth-child(7),
div[data-testid="stDataFrame"] tbody tr td:nth-child(8),
div[data-testid="stDataFrame"] tbody tr td:nth-child(9),
div[data-testid="stDataFrame"] tbody tr td:nth-child(10),
div[data-testid="stDataFrame"] tbody tr td:nth-child(11) { padding-right: 8px; }

/* 3) Buttons: Start/Stop same size, no wrapping */
div.stButton > button {
  min-width: 120px;
  height: 36px;
  border-radius: 12px;
  white-space: nowrap;
}
div.stButton > button:disabled { opacity: 0.7; }
</style>
""",
    unsafe_allow_html=True,
)

# Prevent inner DataFrame scrollbar: helper that returns an appropriate height (int) for st.dataframe
def _df_height(num_rows: int | None = None, *, row_height: int = 36, header_height: int = 40, max_height: int = 800) -> int:
    """
    Compute an integer height for st.dataframe that prevents an inner scrollbar
    for moderately sized tables. If num_rows is None, return max_height.
    """
    try:
        if num_rows is None or num_rows <= 0:
            return max_height
        h = header_height + int(num_rows) * row_height
        # clamp to sensible bounds
        if h < 120:
            h = 120
        if h > max_height:
            h = max_height
        return h
    except Exception:
        return max_height

# --- Wrapper that auto-applies sensible money/percent formatting to known columns ---
def st_dataframe_fmt(
    df: pd.DataFrame | None,
    *,
    money_columns: Iterable[str] = ("Price", "Market Cap", "Volume (24h)", "Liquidity"),
    percent_columns: Iterable[str] = ("1H", "6H", "24H"),
    height_rows: int | None = None,
    use_styler: bool = True,
    **st_data_frame_kwargs,
) -> None:

    if df is None:
        st.info("No data available")
        return

    try:
        df_to_show = df.copy()
    except Exception:
        # if df isn't a DataFrame but is displayable, just show it
        st.dataframe(df, **st_data_frame_kwargs)
        return

    # compute height
    rows = len(df_to_show) if height_rows is None else int(height_rows)
    height = _df_height(rows)

    if use_styler:
        try:
            fmt_map = {}
            for col in money_columns:
                if col in df_to_show.columns:
                    fmt_map[col] = _fmt_money_for_styler
            for col in percent_columns:
                if col in df_to_show.columns:
                    fmt_map[col] = _fmt_pct_for_styler

            if fmt_map:
                styler = df_to_show.style.format(fmt_map)
                # Optionally tweak additional styler options here in future
                st.dataframe(styler, height=height, **st_data_frame_kwargs)
                return
        except Exception:
            # fall through to plain display on any styler issue
            pass

    # fallback: coerce numeric columns to nicely rounded strings for display
    df_fallback = df_to_show.copy()
    for col in money_columns:
        if col in df_fallback.columns:
            try:
                df_fallback[col] = df_fallback[col].apply(lambda x: _fmt_money_for_styler(x))
            except Exception:
                pass
    for col in percent_columns:
        if col in df_fallback.columns:
            try:
                df_fallback[col] = df_fallback[col].apply(lambda x: _fmt_pct_for_styler(x))
            except Exception:
                pass

    st.dataframe(df_fallback, height=height, **st_data_frame_kwargs)

# --- Helper: merge your custom Styler.format map with default money/percent columns ---
def style_with_defaults(df: pd.DataFrame, extra_fmt: Optional[Dict[str, object]] = None):
    try:
        cols_l = {c.lower(): c for c in df.columns}
    except Exception:
        return df.style if hasattr(df, "style") else df

    default_money_keys = {
        "price", "liquidity", "market cap", "market_cap", "volume (24h)", "volume_24h"
    }
    default_pct_keys = {
        "1h", "6h", "24h", "pc1h", "pc6h", "pc24h", "change 1h", "change 6h", "change 24h"
    }

    fmt_map: Dict[str, object] = {}
    for k in default_money_keys:
        if k in cols_l:
            fmt_map[cols_l[k]] = _fmt_money_for_styler
    for k in default_pct_keys:
        if k in cols_l:
            fmt_map[cols_l[k]] = _fmt_pct_for_styler

    if extra_fmt:
        # caller wins for overlapping keys
        fmt_map.update(extra_fmt)

    try:
        return df.style.format(fmt_map) if fmt_map else df.style
    except Exception:
        return df.style if hasattr(df, "style") else df


def st_dataframe_with_defaults(
    df: pd.DataFrame | pd.io.formats.style.Styler | None,
    *,
    height: Optional[int] = None,
    key: Optional[str] = None,
):
    # If it's already a Styler, forward directly
    try:
        # pandas.Styler has attribute .render / is instance of Styler, but we avoid an import check
        if hasattr(df, "render") and not isinstance(df, pd.DataFrame):
            return st_dataframe_fmt(df, height=height, key=key)
    except Exception:
        # fall back to normal handling below
        pass

    if df is None:
        return st_dataframe_fmt(df, height=height, key=key)

    # Try to create a styler with defaults
    try:
        cols = {c.lower(): c for c in df.columns}
    except Exception:
        return st_dataframe_fmt(df, height=height, key=key)

    price_cols = [
        c for k, c in cols.items()
        if k in {"price", "liquidity", "market cap", "market_cap", "volume (24h)", "volume_24h"}
    ]
    pct_cols = [
        c for k, c in cols.items()
        if k in {"1h", "6h", "24h", "pc1h", "pc6h", "pc24h", "change 1h", "change 6h", "change 24h"}
    ]

    try:
        styler = df.style
        fmt_map: Dict[str, object] = {}
        for c in price_cols:
            fmt_map[c] = _fmt_money_for_styler
        for c in pct_cols:
            fmt_map[c] = _fmt_pct_for_styler

        if fmt_map:
            styler = styler.format(fmt_map)
            return st_dataframe_fmt(styler, height=height, key=key)
    except Exception:
        # any styling error -> fall back
        pass

    return st_dataframe_fmt(df, height=height, key=key)


def calc_df_height(n_rows: int, row_px: int = 34, header_px: int = 38, pad_px: int = 16, max_px: int = 520) -> int:
    try:
        n = int(n_rows)
    except Exception:
        n = 0
    return min(header_px + n * row_px + pad_px, max_px)

# --- One-time live refresh bootstrap for the Discovered Tokens tab (deduplicated) ---
def _bootstrap_discovery_once() -> None:
    ss = st.session_state
    if not ss.get("discovery_bootstrapped"):
        ss["discovery_bootstrapped"] = True
        # Clear any cached DB fetch so the next call performs a live pull
        try:
            # This will exist by the time the tab calls us.
            fetch_tokens_from_db.clear()
        except Exception:
            pass
        # Force the auto-refresh window to be 'expired' so a pull happens now
        ss["last_token_refresh"] = 0.0

# --- Ensure rugcheck status/failures paths and helper exist -----------------
# Resolve a sane APP_DIR if one wasn't defined earlier in the module
try:
    # If APP_DIR exists and is a Path-like object, coerce to Path; otherwise fall back to a safe per-user folder
    _APPDIR = Path(APP_DIR) if ("APP_DIR" in globals() and APP_DIR) else (Path.home() / ".local" / "share" / "SOLOTradingBot")
except Exception:
    _APPDIR = Path.home() / ".local" / "share" / "SOLOTradingBot"

# Ensure the directory exists (non-fatal)
try:
    _APPDIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# Normalize any preexisting symbolic names to Path; provide defaults if they were not defined
try:
    STATUS_FILE = Path(STATUS_FILE)
except Exception:
    STATUS_FILE = _APPDIR / "rugcheck_status.json"

try:
    FAILURES_FILE = Path(FAILURES_FILE)
except Exception:
    FAILURES_FILE = _APPDIR / "rugcheck_failures.json"

def ensure_rugcheck_status_file() -> bool:
    """
    Create a minimal rugcheck status JSON file if it doesn't already exist.
    Non-fatal: returns True on success or if file already exists, False on error.
    """
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not STATUS_FILE.exists():
            default = {
                "enabled": False,
                "available": False,
                "message": "",
                "timestamp": int(time.time()),
            }
            STATUS_FILE.write_text(json.dumps(default, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception as _e:
        # logger may not be configured yet; try to log, otherwise fallback to print
        try:
            logger.warning("ensure_rugcheck_status_file failed: %s", _e)
        except Exception:
            print("ensure_rugcheck_status_file failed:", _e)
        return False

# Seed the Rugcheck banner once at app boot (safe no-op if already written)
try:
    ensure_rugcheck_status_file()
except Exception:
    pass

# --- Enforce non-migrating mode for .env handling (GUI should never migrate env on users) ---
os.environ.setdefault("SOLANABOT_NO_ENV_MIGRATE", "1")

# ---- Stdlib / third-party ----
import subprocess
import logging

# --- Unified logger name ---
LOGGER_NAME = "SOLOTradingBotGUI"
logger = logging.getLogger(LOGGER_NAME)
import yaml
import time
import threading
import atexit
import asyncio
from logging.handlers import RotatingFileHandler

# ---- Third-party
import psutil
import aiosqlite
import aiohttp
from pandas import DataFrame
from dotenv import load_dotenv

# ---- Currency/time formatters for Status tab ----
# Try short aliases first (preferred), then long names, then package import.
try:
    from .utils_exec import fmt_usd, fmt_ts  # preferred short aliases
except Exception:
    try:
        # fallback to long names (older utils_exec.py)
        from utils_exec import format_usd as fmt_usd
        from utils_exec import format_ts_human as fmt_ts
    except Exception:
        try:
            # final fallback if running as a package
            from solana_trading_bot_bundle.trading_bot.utils_exec import (
                format_usd as fmt_usd,
                format_ts_human as fmt_ts,
            )
        except Exception:
            # last-resort stubs so the GUI still renders
            def fmt_usd(v, *, compact=False, decimals=2):
                try:
                    n = float(v)
                except Exception:
                    return "$0.00"
                return f"${n:,.{decimals}f}"

            def fmt_ts(v, *, with_time=True):
                from datetime import datetime
                try:
                    x = float(v)
                    if 0 < x < 10_000_000_000:
                        dt = datetime.fromtimestamp(x)
                    else:
                        dt = datetime.fromtimestamp(x / 1000.0)
                    return dt.strftime("%Y-%m-%d %H:%M:%S") if with_time else dt.strftime("%Y-%m-%d")
                except Exception:
                    return "-"

# --- Rugcheck import shim (unchanged behavior) ---
import types as _types
def _install_rugcheck_shim():
    # Try utils.rugcheck_auth
    _fn = None
    try:
        import rugcheck_auth as _rcmod  # utils on sys.path above
        _fn = getattr(_rcmod, "get_rugcheck_token", None)
    except Exception:
        try:
            from utils import rugcheck_auth as _rcmod  # if package-style
            _fn = getattr(_rcmod, "get_rugcheck_token", None)
        except Exception:
            _fn = None
    mname = "solana_trading_bot_bundle.rugcheck_auth"
    if mname not in sys.modules:
        _shim = _types.ModuleType(mname)
        if callable(_fn):
            _shim.get_rugcheck_token = _fn
        else:
            async def get_rugcheck_token():
                # Fallback: read JWT/API key from env; empty string is OK (public endpoints)
                return os.getenv("RUGCHECK_JWT", "") or os.getenv("RUGCHECK_API_TOKEN", "")
            _shim.get_rugcheck_token = get_rugcheck_token  # async fallback
        sys.modules[mname] = _shim

_install_rugcheck_shim()

# ---- Local package
from solana_trading_bot_bundle.common.constants import (
    APP_NAME, appdata_dir, config_path, env_path, token_cache_path, prefer_appdata_file,
    display_cap,  
)


from solana_trading_bot_bundle.trading_bot.trading import main as trading_main

# ---- In-process trading globals (place them here)
import threading, asyncio

# ---- Resolve paths (force GUI to use the SAME AppData as the launcher) ----
import sys as _sys, os as _os, shutil as _shutil
from pathlib import Path as _Path

_APPNAME = "SOLOTradingBot"

if _os.name == "nt":
    _base = _os.getenv("LOCALAPPDATA") or _os.getenv("APPDATA") or str(_Path.home() / "AppData" / "Local")
    APP_DIR: _Path = _Path(_base) / _APPNAME
elif _sys.platform == "darwin":
    APP_DIR: _Path = _Path.home() / "Library" / "Application Support" / _APPNAME
else:
    APP_DIR: _Path = _Path.home() / ".local" / "share" / _APPNAME

APP_DIR.mkdir(parents=True, exist_ok=True)

# Primary (canonical) paths
CONFIG_PATH: _Path    = APP_DIR / "config.yaml"
ENV_PATH: _Path       = APP_DIR / ".env"
DB_PATH: _Path        = APP_DIR / "tokens.sqlite3"   
LOG_PATH: _Path       = APP_DIR / "logs" / "bot.log"

# --- Streamlit-safe logging initialization (idempotent) ---
def _init_gui_logging() -> None:
    try:
        level_name = (os.getenv("LOG_LEVEL") or "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        # Create logs directory
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        logger.setLevel(level)
        logger.propagate = False  # avoid duplicate logs in Streamlit

        # Prevent duplicate handlers on reruns
        has_file = any(getattr(h, 'baseFilename', None) == str(LOG_PATH) for h in logger.handlers)
        if not has_file:
            from logging.handlers import RotatingFileHandler
            fh = RotatingFileHandler(str(LOG_PATH), maxBytes=2_000_000, backupCount=3, encoding='utf-8')
            fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
            fh.setFormatter(fmt)
            fh.setLevel(level)
            logger.addHandler(fh)

        # Also ensure a console handler for dev
        has_console = any(isinstance(h, logging.StreamHandler) and not hasattr(h, 'baseFilename') for h in logger.handlers)
        if not has_console:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(levelname)s %(name)s: %(message)s'))
            ch.setLevel(level)
            logger.addHandler(ch)
    except Exception:
        # Avoid crashing the GUI if logging fails
        pass

STOP_FLAG_PATH: _Path = APP_DIR / "bot_stop_flag.txt"
# Persisted process metadata
PID_FILE: _Path     = APP_DIR / "bot.pid"
SUBPROC_LOG: _Path  = APP_DIR / "bot_subprocess.log"

try:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)                    
        os.environ["SOLBOT_ENV"] = str(ENV_PATH) 
except Exception as _e:
    print(f"[env] failed to load {ENV_PATH}: {_e}", flush=True)

# --- Ensure config.yaml exists in the canonical APP_DIR; no XDG/mac fallback ---
try:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        # Seed from best available local candidates (project / bundled)
        _candidates = [
            _Path(__file__).with_name("config.yaml"),
            _Path.cwd() / "config.yaml",
            _Path(getattr(_sys, "_MEIPASS", _Path(__file__).parent)) / "config.yaml",
        ]
        for _src in _candidates:
            try:
                if _src.exists():
                    _shutil.copy2(_src, CONFIG_PATH)
                    break
            except Exception:
                pass
except Exception:
    # non-fatal — downstream loader will surface a clear error if still missing
    pass


# (dirs for logs/stop flag created later)

# =============================
# Bot spawn/stop helpers (idempotent)  — with dynamic launcher resolution
# =============================

if "start_bot" not in globals() or "_bot_running" not in globals():

    # --- Persisted process metadata (fallback if not defined earlier; no bare expressions) ---
    if "PID_FILE" not in globals():
        PID_FILE: _Path = APP_DIR / "bot.pid"
    if "SUBPROC_LOG" not in globals():
        SUBPROC_LOG: _Path = APP_DIR / "bot_subprocess.log"
            

    # =============================
    # Bot spawn/stop helpers (PID-aware)
    # =============================

    def _build_pythonpath_for_spawn() -> str:
        """Compose a PYTHONPATH that lets the child import the bundle as a package."""
        parts: list[str] = []
        parts.extend([str(_PKG_ROOT)])  # repo/package root
        existing = os.environ.get("PYTHONPATH", "")
        if existing:
            parts.extend([p for p in existing.split(os.pathsep) if p])
        seen, out = set(), []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return os.pathsep.join(out)

    # ---- small pidfile utilities ----
    def _read_pidfile() -> int | None:
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            return pid if pid > 0 else None
        except Exception:
            return None

    def _write_pidfile(pid: int) -> None:
        try:
            PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            PID_FILE.write_text(f"{int(pid)}\n", encoding="utf-8")
        except Exception:
            pass

    def _clean_pidfile() -> None:
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    def _is_alive(pid: int) -> bool:
        try:
            p = psutil.Process(int(pid))
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    def _bot_running() -> bool:
        """Return True if a bot process is alive; restores PID from file across reruns."""
        pid = st.session_state.get("bot_pid")
        if pid and _is_alive(pid):
            return True
        st.session_state.pop("bot_pid", None)  # drop bad session PID

        pid = _read_pidfile()
        if pid and _is_alive(pid):
            st.session_state["bot_pid"] = int(pid)
            return True

        _clean_pidfile()  # stale pidfile
        return False

def _running_bool() -> bool:
    try:
        return bool(_bot_running()) if callable(_bot_running) else False
    except Exception:
        return False
    # ---- Resolve a runnable bot command (module-first, then script) ----
def _resolve_bot_cmd() -> tuple[list[str], str]:
    """
    Return (cmd_list, human_label). Prefers `python -m solana_trading_bot`
    (your working top-level module), then a package-level __main__ at
    `solana_trading_bot_bundle.trading_bot`, and finally a direct script path.
    """
    import importlib.util as iu

    # 1) Top-level module: solana_trading_bot.py in repo root
    if iu.find_spec("solana_trading_bot"):
        return ([sys.executable, "-m", "solana_trading_bot"], "module: solana_trading_bot")

    # 2) Package __main__: solana_trading_bot_bundle/trading_bot/__main__.py (optional)
    if iu.find_spec("solana_trading_bot_bundle.trading_bot"):
        # Only works if that package has a __main__.py; harmless if not chosen.
        return ([sys.executable, "-m", "solana_trading_bot_bundle.trading_bot"],
                "module: solana_trading_bot_bundle.trading_bot")

    # 3) Direct script path fallbacks
    for candidate in (
        _PKG_ROOT / "solana_trading_bot.py",
        Path(__file__).with_name("solana_trading_bot.py"),
    ):
        try:
            if candidate.exists():
                return ([sys.executable, str(candidate)], f"script: {candidate}")
        except Exception:
            pass

    # Nothing found; keep the old (broken) target so we can surface a clear error in the log
    return ([sys.executable, "-m", "solana_trading_bot_bundle.trading_bot.solana_trading_bot"],
            "module: solana_trading_bot_bundle.trading_bot.solana_trading_bot (legacy/missing)")

def start_bot() -> None:
    """Spawn the trading bot as a subprocess and persist its PID (session + pidfile)."""
    if _bot_running():
        return  # already running

    # Respect stop flag: if user requested stop, don't start (only "1" is treated as explicit stop)
    try:
        if STOP_FLAG_PATH.exists():
            try:
                if STOP_FLAG_PATH.read_text(encoding="utf-8").strip() == "1":
                    return
            except Exception:
                # If we can't read it for some reason, be conservative and refuse to start
                return
    except Exception:
        pass

    # clear any previous error since we're attempting a fresh start
    st.session_state.pop("_bot_last_error", None)

    # Prepare environment for the child process
    env = dict(os.environ)
    try:
        env["PYTHONPATH"] = _build_pythonpath_for_spawn()
    except Exception:
        # fallback: keep parent's PYTHONPATH
        pass
    try:
        if "SOLBOT_ENV" not in env and 'ENV_PATH' in globals() and ENV_PATH:
            env["SOLBOT_ENV"] = str(ENV_PATH)
    except Exception:
        pass

    # Decide how to launch the bot (no hard-coded legacy -m target)
    cmd, label = _resolve_bot_cmd()

    SUBPROC_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBPROC_LOG, "a", encoding="utf-8") as f:
        f.write("\n" + ("=" * 78) + "\n")
        import time as _t, shlex as _shlex
        f.write(_t.strftime("[%Y-%m-%d %H:%M:%S] ") + "Starting bot…\n")
        f.write(f"launcher={label}\n")
        f.write(f"cwd={os.getcwd()}\n")
        try:
            try:
                f.write(f"cmd={_shlex.join(cmd)}\n")
            except Exception:
                f.write(f"cmd={cmd}\n")
            f.flush()

             # Platform-specific detached spawn so Streamlit reruns won't kill the child
            if os.name == "nt":
                # Windows: create hidden process (no console window) + new process group
                # CREATE_NO_WINDOW prevents a visible console; NEW_PROCESS_GROUP keeps it separate from the GUI.
                CREATE_NO_WINDOW = 0x08000000
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                creation = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
                try:
                    si = subprocess.STARTUPINFO()
                    # Ask Windows not to show a window (defensive; may be ignored on some Python builds)
                    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                except Exception:
                    si = None

                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    creationflags=creation,
                    close_fds=True,
                    cwd=os.getcwd(),
                    text=True,
                    startupinfo=si if si is not None else None,
                )
            else:
                # POSIX: start in a new session so SIGHUP to the GUI won't kill the child.
                # Prefer start_new_session=True (Python 3.2+) over preexec_fn=os.setsid.
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                    cwd=os.getcwd(),
                    text=True,
                )

            pid = int(proc.pid)
            st.session_state["bot_pid"] = pid
            _write_pidfile(pid)

            # Clear any autostart locks/flags to avoid races
            for k in ("_autostart_done", "_autostart_pending", "_autostart_error", "_autostart_last_try_ts"):
                st.session_state.pop(k, None)
            try:
                _release_autostart_lock()
            except Exception:
                pass

            f.write(f"spawned pid={pid}\n")
            f.flush()

            # Detect instant crash and surface the tail to the UI
            try:
                time.sleep(0.8)
            except Exception:
                pass
            if not _is_alive(pid):
                tail = "(no subprocess output available)"
                try:
                    tail = "".join(SUBPROC_LOG.read_text(encoding="utf-8", errors="ignore").splitlines(True)[-200:])
                except Exception:
                    pass
                st.session_state["_bot_last_error"] = f"Bot exited immediately after spawn. See subprocess log:\n{tail}"
                _clean_pidfile()
                st.session_state.pop("bot_pid", None)
        except Exception as e:
            # Surface spawn-time exceptions to the subprocess log and the GUI
            try:
                f.write(f"Spawn failed: {e}\n")
                f.flush()
            except Exception:
                pass
            st.session_state["_bot_last_error"] = f"Spawn failed: {e}"
            return

def stop_bot() -> None:
    """Signal the bot to stop, wait briefly for exit, then force-kill if needed. Cleans all state."""
    ss = st.session_state

    # 1) Ask the bot to exit gracefully (your loop should watch this file)
    try:
        STOP_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Use canonical "1" so trading.py (which checks for "1") will detect it
        STOP_FLAG_PATH.write_text("1\n", encoding="utf-8")
    except Exception:
        pass

    # 2) Find the PID (prefer session, fall back to pidfile)
    pid = ss.get("bot_pid") or _read_pidfile()

    # 3) Try to terminate the process cleanly; escalate if it won't exit
    if pid:
        try:
            import psutil, time as _time, subprocess as _subp
            proc = psutil.Process(int(pid))
            if proc.is_running():
                try:
                    proc.terminate()             # gentle
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)         # give it up to ~5s to honor STOP flag
                except psutil.TimeoutExpired:
                    # Terminate children first, then process
                    try:
                        for child in proc.children(recursive=True):
                            try:
                                child.terminate()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        proc.kill()              # force if still alive
                    except Exception:
                        pass
        except ImportError:
            # psutil not available — best-effort using taskkill on Windows or kill on POSIX
            try:
                if os.name == "nt":
                    try:
                        import subprocess as _subp  # ensure subprocess is available here
                        _subp.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
                    except Exception:
                        pass
                else:
                    try:
                        os.kill(int(pid), 15)  # SIGTERM
                    except Exception:
                        pass
            except Exception:
                pass
        except psutil.NoSuchProcess:
            pass
        except Exception:
            pass

    # 4) Clean up files (so UI can't misread stale state)
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        HEARTBEAT_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        STARTED_AT_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    # NOTE: keep STOP_FLAG_PATH present so the bot process itself can remove it after a clean shutdown.
    # If you prefer the GUI to remove the stop flag immediately, uncomment the lines below.
    # try:
    #     STOP_FLAG_PATH.unlink(missing_ok=True)
    # except Exception:
    #     pass

    # 5) Clear session flags so Start is immediately available and autostart can re-run if enabled
    for k in ("bot_pid", "_autostart_done", "_autostart_pending", "_autostart_error", "_autostart_last_try_ts"):
        ss.pop(k, None)

    # 6) Release multi-tab autostart lock (must be defined earlier in the file)
    try:
        _release_autostart_lock()
    except Exception:
        pass
    
# --- Auto-start guard (call once per rerun; respects STOP_FLAG_PATH; double-spawn safe) ---
AUTO_START_BOT = os.getenv("AUTO_START_BOT", "0").strip().lower() in ("1", "true", "yes", "y")
_AUTOSTART_BACKOFF_SEC = 2.0

def _maybe_autostart_bot() -> None:
    """
    Safe autostart: respects STOP_FLAG_PATH, runs once per session,
    and confirms the bot is actually running before marking done.
    Includes a tiny backoff and a multi-tab lock.
    """
    ss = st.session_state
    now = time.time()

    # If autostart disabled, clear pending/error and bail
    if not AUTO_START_BOT:
        ss.pop("_autostart_pending", None)
        ss.pop("_autostart_error", None)
        return

    # Backoff throttle to avoid thrash across rapid reruns
    last_try = ss.get("_autostart_last_try_ts")
    if last_try and (now - last_try) < _AUTOSTART_BACKOFF_SEC:
        return

    # One-shot per session
    if ss.get("_autostart_done", False):
        return

    # Respect STOP flag: ONLY treat an explicit "1" as a blocking stop instruction.
    # If the file exists with any other content we do not treat it as an authoritative stop
    # (this prevents stale or unrelated contents from permanently blocking autostart).
    try:
        if STOP_FLAG_PATH.exists():
            try:
                content = STOP_FLAG_PATH.read_text(encoding="utf-8").strip()
                if content == "1":
                    ss["_autostart_pending"] = False
                    return
                # If content is present but not "1", ignore and continue with autostart.
            except Exception:
                # If we can't read the file, be conservative and don't autostart now.
                ss["_autostart_pending"] = False
                return
    except Exception:
        # If checking .exists() failed for some reason, continue conservatively.
        pass

    # Ensure helpers exist
    sb = globals().get("start_bot")
    br = globals().get("_bot_running")
    if not callable(sb) or not callable(br):
        ss["_autostart_pending"] = True
        return

    # If already running, mark done and release any lock
    if br():
        ss["_autostart_done"] = True
        ss["_autostart_pending"] = False
        ss.pop("_autostart_error", None)
        try:
            _release_autostart_lock()
        except Exception:
            pass
        return

    # Acquire multi-tab lock; if another tab is handling autostart, skip
    if not _has_autostart_lock():
        return

    # Try starting
    ss["_autostart_last_try_ts"] = now
    ss["_autostart_pending"] = True
    try:
        sb()
        # Let PID/heartbeat land
        time.sleep(0.6)
        if br():
            ss["_autostart_done"] = True
            ss["_autostart_pending"] = False
            ss.pop("_autostart_error", None)
            try:
                _release_autostart_lock()
            except Exception:
                pass
        else:
            # leave done=False; next rerun will retry after backoff
            ss["_autostart_done"] = False
            # keep the lock so another tab doesn't race; it will be released on success/stop
    except Exception as e:
        ss["_autostart_error"] = str(e)
        ss["_autostart_pending"] = False   # allow manual Start
        try:
            _release_autostart_lock()
        except Exception:
            pass

# ... define start_bot / _bot_running first 
_maybe_autostart_bot()

# =============================
# Config Helpers
# =============================

@st.cache_data(show_spinner=False)
def load_config(path: Path | None = None) -> dict:
    """Load YAML config with caching. If path is None, uses current CONFIG_PATH."""
    cfg_path = Path(path) if path is not None else Path(CONFIG_PATH)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except Exception as e:
        logging.getLogger("SolanaMemeBotGUI").error("Failed to load config: %s", e)
        try:
            st.error(f"❌ Failed to load configuration: {e}")
        except Exception:
            pass
        return {}  # allow app to continue

def save_config(config: dict, path: Path = CONFIG_PATH) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return True
    except Exception as e:
        logging.getLogger("SolanaMemeBotGUI").error("Failed to save config: %s", e)
        try:
            st.error(f"❌ Failed to save configuration: {e}")
        except Exception:
            pass
        return False


def _resolve_db_and_logs_from_config(cfg: dict[str, Any]) -> None:
    global DB_PATH, LOG_PATH
    lg = logging.getLogger("SolanaMemeBotGUI")


    if not isinstance(cfg, dict):
        lg.debug("_resolve_db_and_logs_from_config: cfg is not a dict; skipping")
        return

    # Accept multiple common keys from config
    
# --- Helpers ---------------------------------------------------------------
# You can pass your logger in; if not, we fall back to a tiny stub.
class _NullLogger:
    def debug(self, *a, **k): ...
    def info(self, *a, **k): ...
    def warning(self, *a, **k): ...
    def error(self, *a, **k): ...
    def exception(self, *a, **k): ...

def _looks_windows_path(s: str) -> bool:
    """Heuristic: drive-letter or lots of backslashes ⇒ looks like Windows."""
    s = str(s or "")
    return (len(s) >= 2 and s[1] == ":" and s[0].isalpha()) or ("\\" in s and "/" not in s)


def _sanitize_win_user_profile_path(raw_path: str) -> str:
    r"""
    If raw_path looks like a Windows per-user AppData path for a DIFFERENT user,
    rewrite it to use the current process' LOCALAPPDATA base. Otherwise return raw_path unchanged.

    Examples rewritten:
      C:\Users\johnk\AppData\Local\SOLOTradingBot -> <LOCALAPPDATA>\SOLOTradingBot

    This helps when a config was authored on another machine/account.
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

def _expand_user_env(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(str(p)))

def _first_str(cfg: dict, keys: tuple[str, ...]) -> Optional[str]:
    """Return the first non-empty string value for any of the keys."""
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _default_app_dir(app_name: str = "SOLOTradingBot") -> Path:
    """Cross-platform per-user app dir."""
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / app_name
    # macOS
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / app_name
    # Linux/other
    return Path.home() / ".local" / "share" / app_name

# --- Core resolution -------------------------------------------------------

def resolve_paths_from_config(
    cfg: Dict[str, Any],
    *,
    app_dir: Optional[Path] = None,
    logger=None,
) -> tuple[Path, Path]:
    """
    Resolve DB_PATH and LOG_PATH from a config dict safely.
    Returns (DB_PATH, LOG_PATH). Creates parent dirs when needed.
    """
    lg = logger or _NullLogger()
    cfg = cfg or {}

    # Where we keep defaults
    APP_DIR = app_dir or _default_app_dir()
    APP_DIR.mkdir(parents=True, exist_ok=True)

    # -------- Determine default DB via the database module (if available) -----
    try:
        try:
            from solana_trading_bot_bundle.trading_bot import database as _dbmod  # type: ignore
        except Exception:
            import trading_bot.database as _dbmod  # type: ignore

        if hasattr(_dbmod, "_resolve_db_path"):
            default_db = Path(_dbmod._resolve_db_path())
        else:
            default_db = APP_DIR / "tokens.sqlite3"
    except Exception:
        default_db = APP_DIR / "tokens.sqlite3"

    default_log = APP_DIR / "logs" / "bot.log"

    # ----- Early string-only resolution (flat keys) ---------------------------
    db_raw  = _first_str(cfg, ("db_path", "DB_PATH", "database_path"))
    log_raw = _first_str(cfg, ("log_path", "LOG_PATH", "log", "logfile"))

    DB_PATH: Path = default_db
    LOG_PATH: Path = default_log

    # Resolve DB_PATH from flat keys if provided
    if db_raw:
        try:
            raw = _expand_user_env(db_raw)
            raw = _sanitize_win_user_profile_path(raw)
            if os.name != "nt" and _looks_windows_path(raw):
                # Don't mis-normalize Windows paths on POSIX; make a best-effort slash swap
                db_path = Path(raw.replace("\\", "/"))
            else:
                db_path = Path(raw)
            # If it's a directory or suffixless, choose a sensible filename
            if (db_path.suffix == "") and (db_path.is_dir() or db_path.name.find(".") == -1):
                db_path = db_path / "tokens.sqlite3"
            db_path = db_path.resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            DB_PATH = db_path
            lg.debug("Resolved DB_PATH from flat keys: %s", DB_PATH)
        except Exception:
            lg.exception("Failed to resolve DB_PATH from flat keys; keeping default.")

    # Resolve LOG_PATH from flat keys if provided
    if log_raw:
        try:
            raw = _expand_user_env(log_raw)
            raw = _sanitize_win_user_profile_path(raw)
            if os.name != "nt" and _looks_windows_path(raw):
                log_path = Path(raw.replace("\\", "/"))
            else:
                log_path = Path(raw)
            if (log_path.suffix == "") and (log_path.is_dir() or log_path.name.find(".") == -1):
                log_path = log_path / "bot.log"
            log_path = log_path.resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            LOG_PATH = log_path
            lg.debug("Resolved LOG_PATH from flat keys: %s", LOG_PATH)
        except Exception:
            lg.exception("Failed to resolve LOG_PATH from flat keys; keeping default.")

    # -------- Structured sections fallbacks ----------------------------------
    # Database section
    try:
        db_section = cfg.get("database", {}) if isinstance(cfg, dict) else {}
        cand = db_section.get("token_cache_path") or db_section.get("path")
        if cand and isinstance(cand, str) and cand.strip():
            raw = _expand_user_env(cand.strip())
            raw = _sanitize_win_user_profile_path(raw)
            if os.name != "nt" and _looks_windows_path(raw):
                lg.warning(
                    "Configured token_cache_path looks Windows-style on this OS (%s). "
                    "Using default DB path instead.", raw
                )
                db_path = Path(default_db)
            else:
                db_path = Path(raw)
                if (db_path.suffix == "") and (db_path.is_dir() or db_path.name.find(".") == -1):
                    db_path = db_path / "tokens.sqlite3"
                db_path = db_path.resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            DB_PATH = db_path
        else:
            # Probe common locations only if DB_PATH is still default
            if DB_PATH == default_db:
                candidates = [
                    Path(default_db),
                    APP_DIR / "tokens.sqlite3",
                    APP_DIR / "tokens.db",  # legacy
                ]
                la = os.getenv("LOCALAPPDATA", "")
                if la:
                    candidates += [
                        Path(la) / "SOLOTradingBot" / "tokens.sqlite3",
                        Path(la) / "SOLOTradingBot" / "tokens.db",
                    ]
                existing = [p for p in candidates if p.exists()]
                DB_PATH = existing[0] if existing else candidates[0]
                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        lg.info("Using token DB at %s", DB_PATH)
    except Exception as e:
        lg.warning("Could not resolve DB path from config (structured); falling back: %s", e)
        DB_PATH = Path(default_db)
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Logging section
    try:
        log_section = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
        lf = log_section.get("log_file_path")
        if isinstance(lf, str) and lf.strip():
            raw = _expand_user_env(lf.strip())
            raw = _sanitize_win_user_profile_path(raw)
            if os.name != "nt" and _looks_windows_path(raw):
                lg.warning(
                    "Configured log_file_path looks Windows-style on this OS (%s). "
                    "Using default log path instead.", raw
                )
                log_path = default_log
            else:
                log_path = Path(raw)
                if (log_path.suffix == "") and (log_path.is_dir() or log_path.name.find(".") == -1):
                    log_path = log_path / "bot.log"
                log_path = log_path.resolve()
        else:
            # Prefer existing parent dir if present
            candidates = [default_log]
            la = os.getenv("LOCALAPPDATA", "")
            if la:
                candidates.insert(0, Path(la) / "SOLOTradingBot" / "logs" / "bot.log")
            existing_parent = [p for p in candidates if p.parent.exists()]
            log_path = existing_parent[0] if existing_parent else candidates[0]

        log_path.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH = log_path
        lg.info("Using log file at %s", LOG_PATH)
    except Exception as e:
        lg.warning("Could not resolve LOG path from config; using default: %s", e)
        LOG_PATH = default_log
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    return DB_PATH, LOG_PATH


# --- Optional: Environment loading (kept robust & non-overwriting) ----------

try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
except Exception:
    load_dotenv = None
    find_dotenv = None

def _safe_load_env_file(path: Path, *, override: bool = False) -> bool:
    """Load a .env file if it exists; return True if loaded."""
    try:
        if not load_dotenv or not path or not path.exists():
            return False
        return bool(load_dotenv(dotenv_path=str(path), override=override))
    except Exception:
        return False

def load_env_chain(*, app_dir: Optional[Path] = None) -> list[str]:
    """
    Load .env files in priority order (no overwrites). Returns list of files used.
    Priority:
      0) SOLBOT_ENV (explicit path)
      1) canonical AppData .env (ENV_PATH or <APP_DIR>/.env)
      2) project-local .env (same folder as this script)
      3) auto-find in CWD via python-dotenv
    """
    APP_DIR = app_dir or _default_app_dir()
    env_path_default = APP_DIR / ".env"

    loaded_from: list[str] = []
    seen: set[str] = set()

    try:
        # 0) explicit override
        solbot_env = (os.getenv("SOLBOT_ENV") or "").strip()
        if solbot_env:
            p = Path(solbot_env)
            if p.exists():
                key = str(p.resolve())
                if key not in seen and _safe_load_env_file(p, override=False):
                    loaded_from.append(key)
                seen.add(key)

        # 1) canonical AppData .env
        ENV_PATH = os.getenv("ENV_PATH")  # optional external definition
        p = Path(ENV_PATH) if ENV_PATH else env_path_default
        if p.exists():
            key = str(p.resolve())
            if key not in seen and _safe_load_env_file(p, override=False):
                loaded_from.append(key)
            seen.add(key)

        # 2) project-local .env
        proj_env = Path(__file__).with_name(".env")
        if proj_env.exists():
            key = str(proj_env.resolve())
            if key not in seen and _safe_load_env_file(proj_env, override=False):
                loaded_from.append(key)
            seen.add(key)

        # 3) auto-find in CWD
        if find_dotenv:
            auto_env_str = find_dotenv(usecwd=True)
            if auto_env_str:
                p = Path(auto_env_str)
                key = str(p.resolve())
                if key not in seen and p.exists() and _safe_load_env_file(p, override=False):
                    loaded_from.append(key)
                seen.add(key)
    except Exception:
        # intentionally swallow — env loading must never crash boot
        pass

    return loaded_from

# --- Example usage ----------------------------------------------------------
# lg = logging.getLogger("SoloTradingBot")  # recommended unified name
# DB_PATH, LOG_PATH = resolve_paths_from_config(cfg, logger=lg)
# env_files = load_env_chain()

    # Never crash the GUI due to .env loading
    pass

# Belt-and-suspenders: export SOLBOT_ENV for child processes (pick the most canonical)
# Belt-and-suspenders: export SOLBOT_ENV for child processes (pick the most canonical)
from pathlib import Path
import os

try:
    # Safely read optional globals
    env_path = globals().get("ENV_PATH")
    env_loaded_from = globals().get("_env_loaded_from") or []
    if not isinstance(env_loaded_from, list):
        # In case another symbol type is used (e.g., tuple)
        env_loaded_from = list(env_loaded_from)

    if isinstance(env_path, str) and env_path.strip() and Path(env_path).exists():
        os.environ["SOLBOT_ENV"] = str(Path(env_path).resolve())
    elif os.getenv("SOLBOT_ENV"):  # leave as-is if already set
        pass
    elif env_loaded_from:
        os.environ["SOLBOT_ENV"] = str(env_loaded_from[0])
except Exception:
    # Never let this crash boot
    pass

# GUI should never migrate env on users
os.environ.setdefault("SOLANABOT_NO_ENV_MIGRATE", "1")

# Optional: stash for diagnostics (defensive: _env_loaded_from may be absent or not a list)
try:
    _env_loaded_from_val = globals().get("_env_loaded_from", [])
    # coerce to list for consistent consumption in the UI
    if not isinstance(_env_loaded_from_val, list):
        try:
            _env_loaded_from_val = list(_env_loaded_from_val)
        except Exception:
            _env_loaded_from_val = [_env_loaded_from_val]
    st.session_state["_env_loaded_from"] = _env_loaded_from_val
except Exception:
    # keep GUI resilient if session_state isn't writable (rare) or other issues occur
    pass


# ---- Bootstrap config-driven paths before logger init ----
try:
    cfg_boot = load_config()  # cached; respects current CONFIG_PATH
except Exception:
    cfg_boot = {}

if isinstance(cfg_boot, dict):
    _resolve_db_and_logs_from_config(cfg_boot)

# Ensure directories now that LOG_PATH may have changed
try:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
try:
    STOP_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# --- Bot metadata files (GUI-side), ensure these exist as Path objects ---
HEARTBEAT_FILE: Path   = APP_DIR / "heartbeat"
STARTED_AT_FILE: Path  = APP_DIR / "started_at.txt"
AUTOSTART_LOCK_FILE: Path = APP_DIR / "autostart.lock"

# --- Multi-tab autostart lock (prevents two browser tabs from starting the bot) ---
def _has_autostart_lock() -> bool:
    try:
        # Try to create exclusively; if it exists, another tab/session holds it
        fd = os.open(str(AUTOSTART_LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        st.session_state["_autostart_lock_owned"] = True
        return True
    except FileExistsError:
        return False
    except Exception:
        return False

def _release_autostart_lock() -> None:
    if st.session_state.get("_autostart_lock_owned"):
        try:
            os.remove(str(AUTOSTART_LOCK_FILE))
        except Exception:
            pass
        st.session_state.pop("_autostart_lock_owned", None)

# ---- Logging (configure a dedicated app logger once)
logger = logging.getLogger("SolanaMemeBotGUI")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(str(LOG_PATH), maxBytes=2_000_000, backupCount=3,
                                 encoding="utf-8", delay=True)
        fh.setFormatter(_fmt)
        logger.addHandler(fh)
    except Exception as e:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logger.warning("Failed to attach RotatingFileHandler at %s: %s", LOG_PATH, e)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_fmt)
    logger.addHandler(sh)

# Prevent duplicate lines via the root logger
logger.propagate = False
logger.info("Logging initialized at %s", LOG_PATH)

def _rebind_logger_file_handler() -> None:
    """Point the existing logger's RotatingFileHandler at the (possibly updated) LOG_PATH."""
    for h in list(logger.handlers):
        if isinstance(h, RotatingFileHandler):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(str(LOG_PATH), maxBytes=2_000_000, backupCount=3,
                                 encoding="utf-8", delay=True)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
        logger.info("Rebound file handler to %s", LOG_PATH)
    except Exception as e:
        logger.warning("Failed to rebind file handler to %s: %s", LOG_PATH, e)


# ---- App/env flags
BOT_ENTRY_MODULE = "solana_trading_bot_bundle.trading_bot.solana_trading_bot"
DEV_MODE = str(os.getenv("GUI_DEV_MODE", "0")).lower() in ("1", "true", "yes")
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY", "")

# Load settings from the already-cached config (avoid re-reading file)
_cfg = cfg_boot if isinstance(cfg_boot, dict) else {}

SAFETY_CFG = _cfg.get("safety", {})
NEW_TOKEN_BADGE_WINDOW_MIN = int(SAFETY_CFG.get("new_badge_minutes", 180))
NEW_TOKEN_SAFETY_WINDOW_MIN = int(SAFETY_CFG.get("new_safety_minutes", 1440))
SAFE_MIN_LIQUIDITY_USD = float(SAFETY_CFG.get("safe_min_liquidity_usd", 5000.0))
STRONG_SCORE = float(SAFETY_CFG.get("strong_score", 85.0))

# --- UI/Discovery knobs (shared with bot) ---
DISCOVERY_CFG = _cfg.get("discovery", {}) if isinstance(_cfg, dict) else {}
TOP_N_PER_CATEGORY = int(
    os.getenv("TOP_N_PER_CATEGORY", DISCOVERY_CFG.get("shortlist_per_bucket", 5))
)

# Fallback Prices (unchanged)
FALLBACK_PRICES = {
    "So11111111111111111111111111111111111111112": {
        "price": 175.81,
        "price_change_1h": -0.77,
        "price_change_6h": -0.70,
        "price_change_24h": -3.12,
        "volume_24h": 25000.00,
    }
}

# === DB SCHEMA BOOTSTRAP (run once per Streamlit session) ====================
try:
    # Keep this import EXACTLY on its own line
    try:
        from solana_trading_bot_bundle.trading_bot.database import (
            connect_db, init_db
        )
    except Exception:
        from trading_bot.database import (
            connect_db, init_db  # flat layout fallback
        )

    async def _maybe_await(x):
        import inspect
        return (await x) if inspect.isawaitable(x) else x

    async def _call_init_db_maybe_with_conn(conn):
        """
        init_db may be:
          - async def init_db(conn)
          - def init_db(conn)
          - async def init_db()
          - def init_db()
        Call it safely regardless of signature.
        """
        import inspect
        try:
            sig = inspect.signature(init_db)
        except Exception:
            sig = None

        if sig:
            params = [
                p for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
            ]
            if len(params) >= 1:
                return await _maybe_await(init_db(conn))  # expects a connection
            else:
                return await _maybe_await(init_db())      # expects no args
        # If introspection failed, try with conn first, then fallback to no-arg
        try:
            return await _maybe_await(init_db(conn))
        except TypeError:
            return await _maybe_await(init_db())

    async def _bootstrap_db_once() -> None:
        
        import inspect
        obj = connect_db()
        if hasattr(obj, "__aenter__") and hasattr(obj, "__aexit__"):
            async with obj as db:
                await _call_init_db_maybe_with_conn(db)
            return

        conn = await obj if inspect.isawaitable(obj) else obj
        try:
            await _call_init_db_maybe_with_conn(conn)
        finally:
            close = getattr(conn, "close", None)
            if callable(close):
                try:
                    import inspect as _ins
                    if _ins.iscoroutinefunction(close):
                        await close()
                    else:
                        close()
                except Exception:
                    pass

    @st.cache_resource(show_spinner=False)
    def _ensure_db_schema_ready() -> bool:
        asyncio.run(_bootstrap_db_once())
        return True

    _ = _ensure_db_schema_ready()
except Exception as _db_e:
    logger.warning("DB bootstrap failed: %s", _db_e)

# =============================
# Asyncio Event Loop Management
# =============================
# --- NO Streamlit decorator here ---
_LOOP = None
_LOOP_THREAD = None
_LOOP_LOCK = threading.Lock()

def _shutdown_loop():
    global _LOOP, _LOOP_THREAD
    try:
        if _LOOP and _LOOP.is_running():
            _LOOP.call_soon_threadsafe(_LOOP.stop)
        if _LOOP_THREAD and _LOOP_THREAD.is_alive():
            _LOOP_THREAD.join(timeout=1.5)
    except Exception:
        pass
    try:
        if _LOOP and not _LOOP.is_closed():
            _LOOP.close()
    except Exception:
        pass
    _LOOP = None
    _LOOP_THREAD = None

atexit.register(_shutdown_loop)

def get_event_loop():
    """Create (once) a dedicated asyncio event loop in a background thread (thread-safe, no Streamlit cache)."""
    global _LOOP, _LOOP_THREAD
    with _LOOP_LOCK:
        if _LOOP and _LOOP.is_running() and not _LOOP.is_closed():
            return _LOOP
        loop = asyncio.new_event_loop()
        def _run():
            try:
                asyncio.set_event_loop(loop)
                loop.run_forever()
            except Exception:
                pass
        t = threading.Thread(target=_run, daemon=True, name="gui-asyncio-loop")
        t.start()
        _LOOP, _LOOP_THREAD = loop, t
        return loop


def run_async_task(task, timeout: float | None = None):
    """
    Runs an async task safely from Streamlit (which does not run an event loop).

    Accepts:
      - a coroutine OBJECT,
      - a coroutine FUNCTION (async def),
      - a plain callable (lambda/def) which may return either a value or a coroutine.

    Always awaits the underlying coroutine on the dedicated background loop
    returned by get_event_loop(), preventing 'coroutine was never awaited' issues.
    """
    import inspect
    import concurrent.futures

    # 1) Normalize to a coroutine object (coro)
    if inspect.iscoroutine(task):
        coro = task
    elif inspect.iscoroutinefunction(task):
        coro = task()  # call to get the coroutine object
    elif callable(task):
        # Call the callable. It might return a value OR a coroutine.
        result = task()
        if inspect.iscoroutine(result):
            coro = result
        else:
            async def _wrap(val=result):
                return val
            coro = _wrap()
    else:  # bad input
        raise TypeError(f"run_async_task expected coroutine/corofunc/callable, got {type(task)!r}")

    # 2) Always schedule on our dedicated loop
    loop = get_event_loop()
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        try:
            fut.cancel()
        except Exception:
            pass
        raise
    except Exception:
        raise

# -- minimal dependencies for live-first path --
import asyncio as _lf_asyncio
from typing import List as _lf_List, Dict as _lf_Dict, Any as _lf_Any, Optional as _lf_Optional

try:
    import aiohttp as _lf_aiohttp
except Exception:
    _lf_aiohttp = None

# Use the already-installed rugcheck shim.
try:
    from solana_trading_bot_bundle.rugcheck_auth import get_rugcheck_token as _lf_get_rugcheck_token
except Exception:
    async def _lf_get_rugcheck_token():
        import os as _os
        return _os.getenv("RUGCHECK_JWT", "") or _os.getenv("RUGCHECK_API_TOKEN", "")

# Stable/WSOL hide set (fallback if not defined)
_LF_HIDE = set()

# Prefer an existing ALWAYS_HIDE_IN_CATEGORIES if present, else DEFAULT_ALWAYS_HIDE_IN_CATEGORIES, else literal fallback.
_src = globals().get("ALWAYS_HIDE_IN_CATEGORIES") or globals().get("DEFAULT_ALWAYS_HIDE_IN_CATEGORIES")
if _src:
    try:
        _LF_HIDE = set(_src)
    except Exception:
        _LF_HIDE = set(map(str, _src)) if _src is not None else set()

# Final literal fallback (hard-coded canonical addresses)
if not _LF_HIDE:
    _LF_HIDE = {
        "So11111111111111111111111111111111111111112",  # WSOL
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
        "jupSoLaHXQiZZTSfEWMTRRgpnyFm8f6sZdosWBjx93v",  # JITOSOL
    }

# --- Also hide by symbol/name to catch non-canonical clones (WSOL/USDC/etc.) ---
# You can extend/override via ENV: ALWAYS_HIDE_SYMBOLS="SOL,WSOL,USDC,USDT,Wrapped Solana"
def _csv_env(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {s.strip().upper() for s in raw.split(",") if s.strip()}

ALWAYS_HIDE_SYMBOLS = _csv_env(
    "ALWAYS_HIDE_SYMBOLS",
    "SOL,WSOL,USDC,USDT,WRAPPED SOLANA,WRAPPED USD COIN,USD COIN, JITOSOL"
)

def _hidden_token(t: dict) -> bool:
    addr = (t.get("address") or t.get("token_address") or "").strip()
    if addr and addr in _LF_HIDE:
        return True
    sym_or_name = (t.get("symbol") or t.get("name") or "").strip().upper()
    return bool(sym_or_name and sym_or_name in ALWAYS_HIDE_SYMBOLS)


# Config fallbacks
try:
    _LF_NEW_BADGE_MIN = int(NEW_TOKEN_BADGE_WINDOW_MIN)
except Exception:
    _LF_NEW_BADGE_MIN = 180

try:
    _LF_BIRDEYE_API_KEY = BIRDEYE_API_KEY
except Exception:
    import os as _os
    _LF_BIRDEYE_API_KEY = _os.getenv("BIRDEYE_API_KEY", "")

async def _lf_best_solana_pair(pairs: _lf_List[_lf_Dict[str, _lf_Any]], token: str) -> _lf_Optional[str]:
    best, best_liq = None, -1.0
    for p in pairs or []:
        try:
            if p.get("chainId") != "solana":
                continue
            base = (p.get("baseToken") or {}).get("address")
            quote = (p.get("quoteToken") or {}).get("address")
            if token not in {base, quote}:
                continue
            liq = p.get("liquidity")
            if isinstance(liq, dict):
                liq = float(liq.get("usd", 0) or 0)
            else:
                liq = float(liq or 0)
            if liq > best_liq:
                best_liq = liq
                best = p
        except Exception:
            continue
    return best and best.get("pairAddress")

async def _lf_fetch_price_for_token(addr: str, session) -> _lf_Dict[str, _lf_Any]:
    # 1) Dexscreener tokens endpoint
    try:
        async with session.get(f"https://api.dexscreener.com/latest/dex/tokens/{addr}", timeout=8) as r:
            if r.status == 200:
                j = await r.json()
                pairs = j.get("pairs") or []
                best = await _lf_best_solana_pair(pairs, addr)
                if best:
                    chosen = next((p for p in pairs if p.get("pairAddress") == best), None)
                    if chosen:
                        pc = chosen.get("priceChange") or {}
                        vol = chosen.get("volume") or {}

                        def _maybe_pct_pc(dct, k):
                            if k in dct:
                                v = dct.get(k)
                                if v in (None, ""):
                                    return None
                                try:
                                    return float(v)
                                except Exception:
                                    try:
                                        return float(str(v).replace("%", "").strip())
                                    except Exception:
                                        return None
                            return None

                        return {
                            "price": float(chosen.get("priceUsd", 0) or 0),
                            "price_change_1h": _maybe_pct_pc(pc, "h1"),
                            "price_change_6h": _maybe_pct_pc(pc, "h6"),
                            "price_change_24h": _maybe_pct_pc(pc, "h24"),
                            "volume_24h": float(vol.get("h24", 0) or 0),
                        }
    except Exception:
        pass
    # 2) Birdeye fallback
    try:
        if _LF_BIRDEYE_API_KEY:
            async with session.get(
                f"https://public-api.birdeye.so/defi/price?address={addr}",
                headers={"X-API-KEY": _LF_BIRDEYE_API_KEY},
                timeout=8
            ) as r:
                if r.status == 200:
                    j = await r.json()
                    d = j.get("data") or {}
                    return {
                        "price": float(d.get("value", 0) or 0),
                        "price_change_1h": None,
                        "price_change_6h": None,
                        "price_change_24h": None,
                        "volume_24h": float(d.get("v24hUSD", 0) or 0),
                    }
    except Exception:
        pass
    return {}

async def _lf_discover_pairs(limit: int = 200) -> _lf_List[_lf_Dict[str, _lf_Any]]:
    if not _lf_aiohttp:
        return []
    out: _lf_List[_lf_Dict[str, _lf_Any]] = []
    try:
        async with _lf_aiohttp.ClientSession() as sess:
            async with sess.get("https://api.dexscreener.com/latest/dex/pairs/solana", timeout=10) as r:
                if r.status != 200:
                    return out
                j = await r.json()
                for p in (j.get("pairs") or [])[:max(0, int(limit))]:
                    try:
                        base = p.get("baseToken") or {}
                        addr = base.get("address") or ""
                        if not addr or addr in _LF_HIDE:
                            continue
                        out.append({
                            "address": addr,
                            "name": base.get("name") or base.get("symbol") or addr[:6]+"…"+addr[-4:],
                            "symbol": base.get("symbol") or "",
                            "pairAddress": p.get("pairAddress"),
                            "liquidity": float((p.get("liquidity") or {}).get("usd", 0) or 0),
                            "market_cap": float(p.get("fdv", 0) or 0),
                            "volume_24h": float((p.get("volume") or {}).get("h24", 0) or 0),
                            "dex": p.get("dexId") or "",
                            "createdAt": int((p.get("pairCreatedAt") or 0) / 1000),
                        })
                    except Exception:
                        continue
    except Exception:
        pass
    # dedupe by address
    seen = {}
    for t in out:
        a = t.get("address")
        if a and a not in seen:
            seen[a] = t
    return list(seen.values())

async def _lf_enrich_prices(tokens: _lf_List[_lf_Dict[str, _lf_Any]]) -> _lf_List[_lf_Dict[str, _lf_Any]]:
    if not _lf_aiohttp or not tokens:
        return tokens
    async with _lf_aiohttp.ClientSession() as sess:
        tasks = [_lf_fetch_price_for_token(t.get("address", ""), session=sess) for t in tokens]
        res = await _lf_asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for t, r in zip(tokens, res):
        if isinstance(r, dict):
            t.update(r)
        out.append(t)
    return out

async def _lf_enrich_rugcheck(tokens: _lf_List[_lf_Dict[str, _lf_Any]]) -> _lf_List[_lf_Dict[str, _lf_Any]]:
    if not _lf_aiohttp or not tokens:
        return tokens
    jwt = ""
    try:
        jwt = await _lf_get_rugcheck_token()
    except Exception:
        jwt = ""
    headers = {"Authorization": f"Bearer {jwt}"} if jwt else {}
    async with _lf_aiohttp.ClientSession() as sess:
        tasks = []
        for t in tokens:
            addr = t.get("address", "")
            if not addr:
                tasks.append(_lf_asyncio.sleep(0, result=None))
                continue
            url = f"https://api.rugcheck.xyz/v1/tokens/{addr}"
            async def _one(u=url):
                try:
                    async with sess.get(u, headers=headers, timeout=8) as r:
                        if r.status != 200:
                            return None
                        j = await r.json()
                        return {
                            "rugcheck_label": (j.get("risk", {}) or {}).get("label") or j.get("label") or "",
                            "rugcheck_score": float((j.get("risk", {}) or {}).get("score", 0) or j.get("score", 0) or 0),
                        }
                except Exception:
                    return None
            tasks.append(_one())
        res = await _lf_asyncio.gather(*tasks, return_exceptions=True)
    for t, rc in zip(tokens, res):
        if isinstance(rc, dict) and rc:
            t.update(rc)
        else:
            t.setdefault("rugcheck_label", "")
            t.setdefault("rugcheck_score", 0.0)
    return tokens

def _lf_mcap_bucket(x: float) -> str:
    try:
        x = float(x or 0)
    except Exception:
        x = 0.0
    if x >= 500_000_000: return "High"
    if x >= 50_000_000: return "Mid"
    return "Low"

def _flatten_shortlist_to_list(shortlist, tokens_fallback: list[dict], limit_total: int) -> list[dict]:
    """Normalize shortlist (dict or list) to a flat list while tagging bucket when available."""
    flat: list[dict] = []
    if isinstance(shortlist, dict):
        for k in ("high", "mid", "low", "new", "large", "small"):
            for t in shortlist.get(k, []) or []:
                t["_bucket"] = k
                flat.append(t)
    elif isinstance(shortlist, list):
        flat = shortlist
    else:
        flat = tokens_fallback or []
    return flat[:limit_total]

# --- LIVE-FIRST IMPL ---------------------------------------------------------

# NEW: normalize field names coming from different sources (Dexscreener/Birdeye/etc.)
def _normalize_price_change_keys(t: dict) -> None:
    """Make sure GUI sees price_change_1h / _6h / _24h even if sources use other keys."""
    def _first_existing(keys: list[str]) -> float | None:
        for k in keys:
            v = t.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                pass
        return None

    # only set if missing (don't clobber a proper value)
    if t.get("price_change_1h") is None:
        v = _first_existing(["pc1h", "h1", "1h", "change1h", "priceChange1h"])
        if v is not None:
            t["price_change_1h"] = v

    if t.get("price_change_6h") is None:
        v = _first_existing(["pc6h", "h6", "6h", "change6h", "priceChange6h"])
        if v is not None:
            t["price_change_6h"] = v

    if t.get("price_change_24h") is None:
        v = _first_existing(["pc24h", "h24", "24h", "change24h", "priceChange24h"])
        if v is not None:
            t["price_change_24h"] = v


async def _lf_pull_live_tokens_impl(limit_total: int = 90) -> _lf_List[_lf_Dict[str, _lf_Any]]:
    base = await _lf_discover_pairs(limit=250)
    if not base:
        return []
    base = await _lf_enrich_prices(base)
    base = await _lf_enrich_rugcheck(base)

    BAD_SYMBOLS = {"USDC", "USDT", "JUP", "SOL", "MSOL", "BSOL", "JITOSOL", "STSOL"}
    BAD_NAME_SNIPPETS = ("LP", "Perps", "Staked", "Jupiter", "USDC/USDT", "USDT/USDC")
    # Allow all caps through live-first by default to avoid missing pct data for large caps.
    MAX_CAP_FOR_LIVE = float("inf")  # adjust if you want

    filtered: list[dict] = []
    for t in base:
        sym   = (t.get("symbol") or "").upper()
        name  = t.get("name") or ""
        mc    = float(t.get("market_cap") or t.get("mc") or 0)
        price = float(t.get("price") or 0)
        liq   = float(t.get("liquidity") or 0)
        vol   = float(t.get("volume_24h") or 0)

        # normalize core numeric keys
        t["market_cap"] = mc
        t["liquidity"]  = float(t.get("liquidity")  or t.get("liq")    or liq)
        t["volume_24h"] = float(t.get("volume_24h") or t.get("vol24")  or vol)

        if sym in BAD_SYMBOLS:
            continue
        if any(snip.lower() in name.lower() for snip in BAD_NAME_SNIPPETS):
            continue
        if mc <= 0 or mc > MAX_CAP_FOR_LIVE:
            continue
        if not (price > 0 or liq > 1_000 or vol > 10_000):
            continue

        # ----- attach cap bucket + categories (mirror async path) -----
        HIGH_MIN = float(MC_THRESHOLDS.get("high_min", 50_000_000))
        MID_MIN  = float(MC_THRESHOLDS.get("mid_min",      500_000))

        if mc >= HIGH_MIN:
            cap = "large_cap"   # canonical name
        elif mc >= MID_MIN:
            cap = "mid_cap"
        else:
            cap = "low_cap"

        t["_bucket"] = cap

        existing_cats = t.get("categories") or []
        if isinstance(existing_cats, str):
            existing_cats = [existing_cats]
        existing_cats = [str(c).strip().lower() for c in existing_cats if c]
        if cap not in existing_cats:
            existing_cats.append(cap)
        t["categories"] = existing_cats

        # Canonicalize category aliases if helper is present
        try:
            _normalize_categories(t)
        except Exception:
            pass
        # ----- /cap tagging -----

        # normalize change keys so GUI won’t render 0.00% when they exist under pct_*
        t["price_change_1h"]  = t.get("price_change_1h")  or t.get("pct_change_1h")
        t["price_change_6h"]  = t.get("price_change_6h")  or t.get("pct_change_6h")
        t["price_change_24h"] = t.get("price_change_24h") or t.get("pct_change_24h")

        filtered.append(t)

    out = sorted(
        filtered,
        key=lambda x: float(x.get("liquidity") or 0) * 2
                    + float(x.get("volume_24h") or 0) * 1.5
                    + float(x.get("market_cap") or 0),
        reverse=True,
    )
    return out[:limit_total]

# Thin sync wrapper so the GUI can call the async impl by name expected elsewhere.
def _pull_live_tokens(limit_total: int = 90) -> list[dict]:
    """
    Synchronous entrypoint used by the GUI. Calls the async live-first implementation.
    """
    try:
        # If you already use run_async_task elsewhere (preferred)
        return run_async_task(_lf_pull_live_tokens_impl, limit_total)
    except NameError:
        # If run_async_task isn't in scope here, fall back to a local event loop.
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(_lf_pull_live_tokens_impl(limit_total))


# =========================== END LIVE-FIRST INJECTION ===========================

def _pull_live_tokens_legacy(
    limit_total: int = 90,
    *,
    max_age_hours: int | None = 24,
) -> list[dict]:
    """
    Legacy/DB-backed path. Optionally restricts DB fallback to rows newer than `max_age_hours`.
    If `max_age_hours` is 0 or None, returns the most recent rows without an age filter.

    NOTE: This block assumes `run_async_task` accepts either:
      - a coroutine function (callable) which it will call/schedule, or
      - a coroutine object.
    """
    try:
        # Optional refresh if available (pass CALLABLES, not already-created coroutines)
        try:
            from solana_trading_bot_bundle.trading_bot.fetching import refresh_token_cache as _rtc
            _ = run_async_task(_rtc, timeout=120)  # callable, no parentheses
        except Exception:
            try:
                from solana_trading_bot_bundle.trading_bot.market_data import refresh_token_cache as _rtc
                _ = run_async_task(_rtc, timeout=120)  # callable, no parentheses
            except Exception:
                pass

        # Latest tokens via network (best effort)
        tokens = None
        try:
            from solana_trading_bot_bundle.trading_bot.fetching import get_latest_tokens as _gl
            tokens = run_async_task(_gl, timeout=120)  # callable, no parentheses
        except Exception:
            try:
                from solana_trading_bot_bundle.trading_bot.market_data import get_latest_tokens as _gl
                tokens = run_async_task(_gl, timeout=120)  # callable, no parentheses
            except Exception:
                tokens = None

        # Safe DB fallback (list interface, not single-token getter)
        if not tokens:
            try:
                from solana_trading_bot_bundle.trading_bot.database import list_eligible_tokens
                newer_than = None
                if isinstance(max_age_hours, (int, float)) and max_age_hours > 0:
                    newer_than = int(time.time()) - int(float(max_age_hours) * 3600)
                tokens = run_async_task(
                lambda: list_eligible_tokens(
                    limit=limit_total,
                    newer_than=newer_than,
                    order_by_score_desc=True,
                )
            ) or []
            except Exception:
                tokens = []

        # Normalize for legacy path too
        for _t in (tokens or []):
            _normalize_price_change_keys(_t)

        # --- Bucket & flatten ---
        try:
            from solana_trading_bot_bundle.trading_bot.eligibility import select_top_five_per_category as _sel5
            shortlist = run_async_task(lambda: _sel5(tokens or [], blacklist=set()), timeout=120)
        except Exception:
            shortlist = None

        return _flatten_shortlist_to_list(shortlist, tokens_fallback=tokens or [], limit_total=limit_total)

    except Exception as e:
        try:
            st.error(f"Live scan (legacy) failed: {e}")
        except Exception:
            pass
        return []

def _pull_tokens_with_fallback(
    limit_total: int = 90,
    *,
    legacy_hours: int | None = None,
) -> list[dict]:
    """
    Try live-first; optionally fall back to DB-backed legacy if empty.
    - If `legacy_hours` is provided, it controls the DB window.
    - Otherwise, if `st.session_state['fallback_hours']` exists, that value is used.
    - A value of 0 disables the fallback.
    """
    tokens = _pull_live_tokens(limit_total)
    if tokens:
        return tokens

    # Decide the window (caller arg wins; else UI knob; else 0 = disabled)
    hours = legacy_hours
    if hours is None:
        try:
            hours = int(st.session_state.get("fallback_hours", 0) or 0)
        except Exception:
            hours = 0

    if isinstance(hours, int) and hours > 0:
        return _pull_live_tokens_legacy(limit_total, max_age_hours=hours)

    return []

# =============================
# SQLite Helpers (hybrid)
# =============================
class _ConnectDB:
    def __init__(self, path: Path):
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    def __await__(self):
        # Allow: conn = await connect_db()
        return aiosqlite.connect(str(self._path)).__await__()

    async def __aenter__(self) -> aiosqlite.Connection:
        # Ensure directory exists (avoids failures when DB dir is missing)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create DB directory %s: %s", self._path.parent, e)

        # Small diagnostic if we're on POSIX but got a Windows-looking path
        try:
            if os.name != "nt" and ("\\" in str(self._path)) and (":" in str(self._path)):
                logger.warning(
                    "DB path looks Windows-style on POSIX: %s (will be created as a literal file name).",
                    self._path,
                )
        except Exception:
            pass

        self._conn = await aiosqlite.connect(str(self._path))
        try:
            await self._conn.execute("PRAGMA journal_mode=WAL;")
            await self._conn.execute("PRAGMA busy_timeout=30000;")
            await self._conn.execute("PRAGMA synchronous=NORMAL;")
            await self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.row_factory = aiosqlite.Row
        except Exception as e:
            logger.warning("DB PRAGMA init failed: %s", e, exc_info=True)
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if self._conn is not None:
                await self._conn.close()
        finally:
            self._conn = None


def connect_db() -> _ConnectDB:
    return _ConnectDB(DB_PATH)


# --- Minimal core tables safety net -------------------------------------------
# Guarantees the tables/columns the GUI queries depend on always exist,
# even if init_db() didn't run for some reason.
async def _ensure_core_tables(db: aiosqlite.Connection) -> None:
    # trade_history must support the P&L query in Tab 5
    await db.execute("""
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_address TEXT,
            symbol TEXT,
            buy_amount REAL DEFAULT 0,
            sell_amount REAL DEFAULT 0,
            buy_price REAL DEFAULT 0,
            sell_price REAL DEFAULT 0,
            profit REAL DEFAULT 0,
            buy_time INTEGER,
            sell_time INTEGER
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_history_sell_time
        ON trade_history (sell_time)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_history_token_addr
        ON trade_history (token_address)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_trade_history_symbol
        ON trade_history (symbol)
    """)

    # tokens table is read by the GUI status snapshot
    await db.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            address TEXT PRIMARY KEY,
            symbol TEXT,
            name   TEXT,
            is_trading INTEGER DEFAULT 0,
            sell_time INTEGER
        )
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_tokens_is_trading
        ON tokens (is_trading)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_tokens_sell_time
        ON tokens (sell_time)
    """)

    await db.commit()


# =============================
# eligible_tokens schema
# =============================
async def ensure_eligible_tokens_schema():
    """
    Create/upgrade the `eligible_tokens` table. This function is async and intended to be
    executed on a background event loop. It only logs errors to avoid Streamlit context
    warnings in background threads.

    Notes / fixes based on logs:
      - Add a `data` TEXT column because the GUI reads: SELECT data FROM eligible_tokens
      - Add a `created_at` INTEGER column for lightweight insertion timestamps.
      - Keep all numeric columns the GUI and bot expect.
      - Create indices on creation_timestamp, created_at, and timestamp to speed GUI filters.
    """
    try:
        async with connect_db() as db:
            # Ensure the two core tables exist first (prevents "no such table" in GUI queries)
            await _ensure_core_tables(db)

            # Base table (idempotent)
            await db.execute(
                """
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
                    creation_timestamp INTEGER
                )
                """
            )

            # Snapshot existing columns
            cols = set()
            async with db.execute("PRAGMA table_info(eligible_tokens)") as cur:
                async for row in cur:
                    cols.add(row[1])

            async def add_col(name: str, decl: str):
                """Add a column if it does not already exist."""
                if name not in cols:
                    await db.execute(f"ALTER TABLE eligible_tokens ADD COLUMN {name} {decl}")
                    cols.add(name)

            # Ensure all expected numeric/text columns exist (backfill for older DBs)
            for name, decl in (
                ("name", "TEXT"),
                ("symbol", "TEXT"),
                ("volume_24h", "REAL"),
                ("liquidity", "REAL"),
                ("market_cap", "REAL"),
                ("price", "REAL"),
                ("price_change_1h", "REAL"),
                ("price_change_6h", "REAL"),
                ("price_change_24h", "REAL"),
                ("score", "REAL"),
                ("categories", "TEXT"),
                ("timestamp", "INTEGER"),
                ("creation_timestamp", "INTEGER"),
            ):
                await add_col(name, decl)

            # >>> NEW: columns required by GUI reads <<<
            await add_col("data", "TEXT")  # canonical JSON blob the GUI selects directly

            if "created_at" not in cols:
                await db.execute(
                    "ALTER TABLE eligible_tokens ADD COLUMN created_at INTEGER DEFAULT (strftime('%s','now'))"
                )
                cols.add("created_at")

            # Helpful indices for GUI filtering / sorting
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_eligible_tokens_creation_ts ON eligible_tokens(creation_timestamp)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_eligible_tokens_created_at ON eligible_tokens(created_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_eligible_tokens_timestamp ON eligible_tokens(timestamp)"
            )

            await db.commit()

    except Exception as e:
        logger.error("Failed to ensure eligible_tokens schema: %s", e, exc_info=True)
        # Do NOT call st.* here.


# =============================
# shortlist_tokens schema
# =============================
async def ensure_shortlist_tokens_schema():
    """
    Create/upgrade the `shortlist_tokens` table (address, data, created_at).
    """
    try:
        async with connect_db() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS shortlist_tokens (
                    address TEXT PRIMARY KEY,
                    data    TEXT,
                    created_at INTEGER DEFAULT (strftime('%s','now'))
                )
                """
            )
            cols = set()
            async with db.execute("PRAGMA table_info(shortlist_tokens)") as cur:
                async for row in cur:
                    cols.add(row[1])
            if "data" not in cols:
                await db.execute("ALTER TABLE shortlist_tokens ADD COLUMN data TEXT")
                cols.add("data")
            if "created_at" not in cols:
                await db.execute("ALTER TABLE shortlist_tokens ADD COLUMN created_at INTEGER")
                cols.add("created_at")
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_shortlist_tokens_created_at ON shortlist_tokens(created_at)"
            )
            await db.commit()
    except Exception as e:
        logger.error("Failed to ensure shortlist_tokens schema: %s", e, exc_info=True)


# =====================================================
# price_cache schema (corrected & hardened)
# =====================================================
async def ensure_price_cache_schema():
    """
    Create/upgrade the `price_cache` table. This function is async and intended to be
    executed on a background event loop. It only logs errors.

    Fixes & notes:
      - Guarantees presence of price, % change, volume, and timestamp columns.
      - Adds/refreshes indices (timestamp; address) for faster lookups.
      - Updates the local `cols` snapshot when adding columns to avoid duplicate ALTERs.
      - IMPORTANT: Avoid DEFAULT 0 for pct columns so NULL can represent missing data.
    """
    try:
        async with connect_db() as db:
            # Base table (idempotent) — do not force defaults of 0 so we can store NULL for missing
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS price_cache (
                    address TEXT PRIMARY KEY,
                    price REAL,
                    price_change_1h REAL,
                    price_change_6h REAL,
                    price_change_24h REAL,
                    volume_24h REAL,
                    timestamp INTEGER DEFAULT (strftime('%s','now'))
                )
                """
            )

            # Snapshot existing columns
            cols = set()
            async with db.execute("PRAGMA table_info(price_cache)") as cur:
                async for row in cur:
                    cols.add(row[1])

            async def add_if_missing(col: str, sqltype: str, default_clause: str = ""):
                if col not in cols:
                    clause = f" {default_clause}" if default_clause else ""
                    await db.execute(f"ALTER TABLE price_cache ADD COLUMN {col} {sqltype}{clause}")
                    cols.add(col)

            # Ensure all price-related columns exist (no DEFAULT 0 so NULL allowed)
            await add_if_missing("price", "REAL", "")
            await add_if_missing("price_change_1h", "REAL", "")
            await add_if_missing("price_change_6h", "REAL", "")
            await add_if_missing("price_change_24h", "REAL", "")
            await add_if_missing("volume_24h", "REAL", "")

            # Ensure timestamp column with default now()
            if "timestamp" not in cols:
                await db.execute(
                    "ALTER TABLE price_cache ADD COLUMN timestamp INTEGER DEFAULT (strftime('%s','now'))"
                )
                cols.add("timestamp")

            # Helpful indices
            await db.execute("CREATE INDEX IF NOT EXISTS idx_price_cache_timestamp ON price_cache(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_price_cache_address ON price_cache(address)")

            await db.commit()
    except Exception as e:
        logger.error("Failed to initialize/migrate price_cache: %s", e, exc_info=True)
        # Do NOT call st.* here.


# >>> Normalize scan results so UI code can safely .get(...) on category buckets
def _normalize_scan_result(obj):
    if isinstance(obj, list):
        return {"unknown": obj}
    if isinstance(obj, dict):
        return obj
    return {"unknown": []}


# Ensure GUI tables exist up-front so early SELECTs work (matches bot writer)
async def _ensure_gui_tables():
    await ensure_eligible_tokens_schema()
    await ensure_price_cache_schema()
    await ensure_shortlist_tokens_schema()

# Call once at import time (via the background loop) to avoid "no such column: data"
try:
    run_async_task(_ensure_gui_tables, timeout=20)
except Exception:
    # If run_async_task isn't ready yet, the first UI action will invoke them again (idempotent).
    pass

# =============================
# Price & API Helpers
# =============================
# mirror Part-3's optional import behavior
try:
    import aiohttp as _aiohttp_local
except Exception:
    _aiohttp_local = None

async def get_cached_price(address: str, cache_seconds=300) -> dict | None:
    try:
        async with connect_db() as db:
            async with db.execute(
                "SELECT price, price_change_1h, price_change_6h, price_change_24h, volume_24h, timestamp "
                "FROM price_cache WHERE address = ? AND timestamp >= ?",
                (address, int(time.time()) - cache_seconds),
            ) as cur:
                row = await cur.fetchone()
                if row:
                    return {
                        "price": row["price"],
                        "price_change_1h": row["price_change_1h"],
                        "price_change_6h": row["price_change_6h"],
                        "price_change_24h": row["price_change_24h"],
                        "volume_24h": row["volume_24h"],
                    }
        return None
    except Exception as e:
        logger.warning("Failed to retrieve cached price for %s: %s", address, e)
        return None


async def cache_price(address: str, price_data: dict):
    try:
        async with connect_db() as db:
            # Write NULL for missing pct fields rather than 0
            price_val = price_data.get("price", None)
            pc1 = price_data.get("price_change_1h", None)
            pc6 = price_data.get("price_change_6h", None)
            pc24 = price_data.get("price_change_24h", None)
            vol = price_data.get("volume_24h", 0) or 0
            await db.execute(
                """
                INSERT OR REPLACE INTO price_cache
                (address, price, price_change_1h, price_change_6h, price_change_24h, volume_24h, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    address,
                    None if price_val is None else float(price_val or 0.0),
                    None if pc1 is None else float(pc1),
                    None if pc6 is None else float(pc6),
                    None if pc24 is None else float(pc24),
                    float(vol),
                    int(time.time()),
                ),
            )
            await db.commit()
    except Exception as e:
        logger.warning("Failed to cache price for %s: %s", address, e)


# Prefer de-duplicated best-pair logic from Part 3 when available
async def _best_solana_pair_from_list(pairs: list, token_address: str):
    try:
        # Use Part-3 helper if present to avoid duplication
        return await _lf_best_solana_pair(pairs, token_address)  # type: ignore[name-defined]
    except Exception:
        # Fallback local implementation (kept for resilience)
        best = None
        best_liq = -1.0
        for p in pairs or []:
            try:
                if p.get("chainId") != "solana":
                    continue
                base = (p.get("baseToken") or {}).get("address")
                quote = (p.get("quoteToken") or {}).get("address")
                if token_address not in {base, quote}:
                    continue
                liq = p.get("liquidity")
                if isinstance(liq, dict):
                    liq = float(liq.get("usd", 0) or 0)
                else:
                    liq = float(liq or 0)
                if liq > best_liq:
                    best_liq = liq
                    best = p
            except Exception:
                continue
        return best and best.get("pairAddress")


async def fetch_price_by_token_endpoint(token_address: str, session):
    if not _aiohttp_local:
        return None
    try:
        async with session.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{token_address}",
            timeout=8,
        ) as resp:
            if resp.status != 200:
                logger.warning("Dexscreener tokens endpoint failed for %s: HTTP %s", token_address, resp.status)
                return None
            data = await resp.json()
            pairs = data.get("pairs") or []
            best_pair_addr = await _best_solana_pair_from_list(pairs, token_address)
            if not best_pair_addr:
                logger.warning("No SOL pair for %s via tokens endpoint", token_address)
                return None
            chosen = next((p for p in pairs if p.get("pairAddress") == best_pair_addr), None)
            if not chosen:
                return None
            pc = chosen.get("priceChange") or {}
            vol = chosen.get("volume") or {}

            def _maybe_pct_pc(dct, k):
                if k in dct:
                    v = dct.get(k)
                    if v in (None, ""):
                        return None
                    try:
                        return float(v)
                    except Exception:
                        try:
                            return float(str(v).replace("%", "").strip())
                        except Exception:
                            return None
                return None

            return {
                "price": float(chosen.get("priceUsd", 0) or 0),
                "price_change_1h": _maybe_pct_pc(pc, "h1"),
                "price_change_6h": _maybe_pct_pc(pc, "h6"),
                "price_change_24h": _maybe_pct_pc(pc, "h24"),
                "volume_24h": float(vol.get("h24", 0) or 0),
            }
    except Exception as e:
        logger.warning("Token endpoint fetch failed for %s: %s", token_address, e)
        return None


async def fetch_pair_address(token_address: str, session) -> str | None:
    if not _aiohttp_local:
        return None
    try:
        async with session.get(
            f"https://api.dexscreener.com/latest/dex/search?q={token_address}",
            timeout=8,
        ) as response:
            if response.status != 200:
                logger.warning("Dexscreener pair search failed for %s: HTTP %s", token_address, response.status)
                return None
            data = await response.json()
            pairs = data.get("pairs", [])
            best = await _best_solana_pair_from_list(pairs, token_address)
            return best
    except Exception as e:
        logger.warning("Failed to fetch pair address for %s: %s", token_address, e)
        return None


async def fetch_price_changes_dexscreener(pair_address: str, session) -> dict | None:
    if not _aiohttp_local:
        return None
    try:
        async with session.get(
            f"https://api.dexscreener.com/latest/dex/pairs/solana/{pair_address}",
            timeout=8,
        ) as response:
            if response.status != 200:
                logger.warning("Dexscreener failed for pair %s: HTTP %s", pair_address, response.status)
                return None
            data = await response.json()
            pair = data.get("pair", {})

            pc = (pair.get("priceChange") or {})
            vol = (pair.get("volume") or {})

            def _maybe_pct_pc(dct, k):
                if k in dct:
                    v = dct.get(k)
                    if v in (None, ""):
                        return None
                    try:
                        return float(v)
                    except Exception:
                        try:
                            return float(str(v).replace("%", "").strip())
                        except Exception:
                            return None
                return None

            return {
                "price": float(pair.get("priceUsd", 0) or 0),
                "price_change_1h": _maybe_pct_pc(pc, "h1"),
                "price_change_6h": _maybe_pct_pc(pc, "h6"),
                "price_change_24h": _maybe_pct_pc(pc, "h24"),
                "volume_24h": float((pair.get("volume") or {}).get("h24", 0) or 0),
            }
    except Exception as e:
        logger.warning("Failed to fetch Dexscreener data for pair %s: %s", pair_address, e)
        return None


async def fetch_price_data(
    token_address: str,
    pair_address: str | None = None,
    retries: int = 2,
    backoff: int = 5,
    session: any | None = None,
) -> dict:
    cached_price = await get_cached_price(token_address)
    if cached_price:
        return cached_price

    # Without aiohttp, fall back quickly to cache/seed
    if _aiohttp_local is None and session is None:
        cached_fallback = await get_cached_price(token_address, cache_seconds=86400)
        if cached_fallback:
            return cached_fallback
        fd = FALLBACK_PRICES.get(
            token_address,
            {"price": 0.0, "price_change_1h": 0.0, "price_change_6h": 0.0, "price_change_24h": 0.0, "volume_24h": 0.0},
        )
        await cache_price(token_address, fd)
        return fd

    own = False
    if session is None:
        session = _aiohttp_local.ClientSession()  # type: ignore[union-attr]
        own = True
    try:
        tok_data = await fetch_price_by_token_endpoint(token_address, session=session)
        if tok_data:
            await cache_price(token_address, tok_data)
            return tok_data

        if pair_address is None:
            pair_address = await fetch_pair_address(token_address, session=session)
        if pair_address:
            pair_data = await fetch_price_changes_dexscreener(pair_address, session=session)
            if pair_data:
                await cache_price(token_address, pair_data)
                return pair_data

        if BIRDEYE_API_KEY:
            endpoint = f"https://public-api.birdeye.so/defi/price?address={token_address}"
            for attempt in range(retries):
                try:
                    async with session.get(endpoint, headers={"X-API-KEY": BIRDEYE_API_KEY}, timeout=8) as response:
                        if response.status == 429:
                            if attempt < retries - 1:
                                logger.warning("Birdeye 429 for %s; retrying in %ss", token_address, backoff)
                                await asyncio.sleep(backoff)
                                backoff *= 2
                                continue
                            logger.warning("Rate limit exceeded for %s on Birdeye after %s attempts", token_address, retries)
                            break
                        if response.status != 200:
                            logger.warning("Birdeye failed for %s: HTTP %s", token_address, response.status)
                            break
                        data = await response.json()
                        bd = {
                            "price": float((data.get("data") or {}).get("value") or 0.0),
                            "price_change_1h": None,
                            "price_change_6h": None,
                            "price_change_24h": None,
                            "volume_24h": 0.0,
                        }
                        await cache_price(token_address, bd)
                        return bd
                except Exception as e:
                    logger.warning("Failed to fetch Birdeye data for %s: %s", token_address, e)
                    break
        else:
            logger.info("BIRDEYE_API_KEY not set; skipping Birdeye fallback for %s", token_address)
    finally:
        if own:
            try:
                await session.close()
            except Exception:
                pass

    cached_fallback = await get_cached_price(token_address, cache_seconds=86400)
    if cached_fallback:
        return cached_fallback

    fd = FALLBACK_PRICES.get(
        token_address,
        {"price": 0.0, "price_change_1h": 0.0, "price_change_6h": 0.0, "price_change_24h": 0.0, "volume_24h": 0.0},
    )
    await cache_price(token_address, fd)
    return fd

# ==========================
# Data Load/Save & Maintenance
# ==========================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_tokens_from_db(db_mtime: int, fallback_hours: int, refresh_tick: float) -> list[dict]:
    """
    GUI-facing shortlist reader that tolerates both layouts:
      - eligible_tokens (scalar columns and/or JSON 'data')
      - shortlist_tokens (JSON 'json' or 'data')

    Cache keys:
      - db_mtime:       auto-busts when the DB file changes
      - fallback_hours: affects cutoff window
      - refresh_tick:   busts when you hit "Refresh Tokens"
    """
    try:
        import json, time

        # --- window / cutoff ---
        if not isinstance(fallback_hours, int) or fallback_hours < 0:
            fallback_hours = 24
        cutoff_ts = int(time.time()) - fallback_hours * 3600 if fallback_hours > 0 else None

        # --- helpers ---
        def _f(x, default=0.0):
            if x is None:
                return float(default)
            if isinstance(x, (int, float)):
                return float(x)
            try:
                xs = str(x).replace(",", "").strip()
                if xs in ("", "-", "—", "None", "null"):
                    return float(default)
                return float(xs)
            except Exception:
                return float(default)

        _STABLE_SYMBOLS = {"USDC", "USDT", "USD", "USDE", "DAI", "USDS", "PYUSD"}
        def _looks_stable(tok: dict) -> bool:
            sym = str(tok.get("symbol") or "").upper()
            nm  = str(tok.get("name") or "").upper()
            if sym in _STABLE_SYMBOLS:
                return True
            for kw in ("USDC", "USDT", "STABLE", "DOLLAR", "USD "):
                if kw in sym or kw in nm:
                    return True
            return False

        NEW_TOKEN_BADGE_WINDOW_MIN = 60 * 24  # <= 24h
        def _assign_bucket(tok: dict) -> str:
            try:
                now = int(time.time())
                cts = tok.get("creation_timestamp")
                if cts is not None and int(cts) >= now - NEW_TOKEN_BADGE_WINDOW_MIN * 60:
                    return "newly_launched"
            except Exception:
                pass
            mc = _f(tok.get("market_cap") or tok.get("mc"))
            if mc >= 500_000:
                return "high_cap"     # NOTE: GUI expects "high_cap"
            if 100_000 <= mc < 500_000:
                return "mid_cap"
            return "low_cap"

        def _normalize(tok: dict) -> dict:
            out = dict(tok) if isinstance(tok, dict) else {}

            out["market_cap"] = _f(tok.get("market_cap", tok.get("mc")))
            out["liquidity"]  = _f(tok.get("liquidity"))
            out["volume_24h"] = _f(tok.get("volume_24h", tok.get("vol24h", tok.get("vol24", tok.get("volume24h")))))
            out["price"]      = _f(tok.get("price"))

            pc1h  = tok.get("price_change_1h",  tok.get("pc1h",  tok.get("change_1h",  tok.get("pct_change_1h"))))
            pc6h  = tok.get("price_change_6h",  tok.get("pc6h",  tok.get("change_6h",  tok.get("pct_change_6h"))))
            pc24h = tok.get("price_change_24h", tok.get("pc24h", tok.get("change_24h", tok.get("pct_change_24h"))))

            def _maybe_none(v):
                return None if v in (None, "", "null") else _f(v, 0.0)

            p1, p6, p24 = _maybe_none(pc1h), _maybe_none(pc6h), _maybe_none(pc24h)
            if (p1 or 0.0) == 0.0 and (p6 or 0.0) == 0.0 and (p24 or 0.0) == 0.0:
                p1 = p6 = p24 = None

            out["price_change_1h"]  = p1
            out["price_change_6h"]  = p6
            out["price_change_24h"] = p24

            out["address"] = tok.get("address") or tok.get("mint") or tok.get("token_address")
            out["symbol"]  = tok.get("symbol") or ""
            out["name"]    = tok.get("name")   or out["symbol"] or ""

            if out.get("creation_timestamp") is None:
                maybe_cts = tok.get("creation_timestamp") or tok.get("created_at") or tok.get("timestamp")
                try:
                    out["creation_timestamp"] = int(maybe_cts) if maybe_cts is not None else None
                except Exception:
                    out["creation_timestamp"] = None

            out["_bucket"] = tok.get("_bucket") or _assign_bucket(out)

            if "score" in tok:
                try:
                    out["score"] = float(tok["score"])
                except Exception:
                    pass

            return out

        # --- DB read (both tables), tolerant to JSON/legacy layouts ---
        async def inner() -> list[dict]:
            # Guard optional schema helpers (avoid NameError if not imported)
            try:
                await ensure_eligible_tokens_schema()
            except Exception:
                pass
            try:
                await ensure_price_cache_schema()
            except Exception:
                pass
            try:
                await ensure_shortlist_tokens_schema()
            except Exception:
                pass

            rows: list[dict] = []
            async with connect_db() as db:
                # -------- eligible_tokens --------
                et_cols = set()
                try:
                    async with db.execute("PRAGMA table_info(eligible_tokens)") as cur:
                        async for r in cur:
                            # r[1] is the column name for PRAGMA table_info
                            try:
                                et_cols.add(r[1])
                            except Exception:
                                pass
                except Exception:
                    et_cols = set()

                et_where, et_params = "1=1", []
                if cutoff_ts is not None and (("created_at" in et_cols) or ("timestamp" in et_cols)):
                    if "created_at" in et_cols:
                        et_where += " AND (created_at IS NULL OR created_at >= ?)"
                    else:
                        et_where += " AND (timestamp IS NULL OR timestamp >= ?)"
                    et_params.append(cutoff_ts)

                created_alias = "created_at" if "created_at" in et_cols else ("timestamp" if "timestamp" in et_cols else "NULL")
                et_select = f"SELECT *, {created_alias} AS _row_created FROM eligible_tokens WHERE {et_where} ORDER BY _row_created DESC LIMIT 1000"

                try:
                    async with db.execute(et_select, tuple(et_params)) as cur:
                        async for row in cur:
                            r = dict(row)
                            raw_json = r.get("data")
                            if raw_json:
                                try:
                                    tok = json.loads(raw_json)
                                    if isinstance(tok, dict):
                                        t = _normalize(tok)
                                        t["_created_at_row"] = int(r.get("_row_created") or 0)
                                        rows.append(t)
                                        continue
                                except Exception:
                                    pass
                            # legacy columns reconstruction
                            legacy = {
                                "address": r.get("address") or r.get("mint") or r.get("token_address"),
                                "symbol": r.get("symbol"),
                                "name": r.get("name"),
                                "market_cap": r.get("market_cap") or r.get("mc"),
                                "liquidity": r.get("liquidity"),
                                "volume_24h": r.get("volume_24h"),
                                "price": r.get("price"),
                                "price_change_1h": r.get("price_change_1h"),
                                "price_change_6h": r.get("price_change_6h"),
                                "price_change_24h": r.get("price_change_24h"),
                                "score": r.get("score"),
                                "categories": r.get("categories"),
                                "creation_timestamp": r.get("creation_timestamp") or r.get("created_at") or r.get("timestamp"),
                            }
                            t = _normalize(legacy)
                            t["_created_at_row"] = int(r.get("_row_created") or 0)
                            rows.append(t)
                except Exception:
                    pass

                # -------- shortlist_tokens --------
                sl_cols = set()
                try:
                    async with db.execute("PRAGMA table_info(shortlist_tokens)") as cur:
                        async for r in cur:
                            try:
                                sl_cols.add(r[1])
                            except Exception:
                                pass
                except Exception:
                    sl_cols = set()

                if sl_cols:
                    col_json = "json" if "json" in sl_cols else ("data" if "data" in sl_cols else None)
                    col_created = "created_at" if "created_at" in sl_cols else None
                    if col_json:
                        sl_where, sl_params = "1=1", []
                        if cutoff_ts is not None and col_created:
                            sl_where += f" AND {col_created} >= ?"
                            sl_params.append(cutoff_ts)
                        sl_select = (
                            f"SELECT {col_json} AS blob, {col_created or 'NULL'} AS _row_created "
                            f"FROM shortlist_tokens WHERE {sl_where} ORDER BY _row_created DESC LIMIT 1000"
                        )
                        try:
                            async with db.execute(sl_select, tuple(sl_params)) as cur:
                                async for row in cur:
                                    r = dict(row)
                                    blob = r.get("blob")
                                    if not blob:
                                        continue
                                    try:
                                        tok = json.loads(blob)
                                        if isinstance(tok, dict):
                                            t = _normalize(tok)
                                            t["_created_at_row"] = int(r.get("_row_created") or 0)
                                            rows.append(t)
                                    except Exception:
                                        continue
                        except Exception:
                            pass

            return rows  # <-- ensure rows are returned

        raw = run_async_task(inner) or []

        # --- de-dupe by address, keep best ---
        def _key_for_best(t: dict):
            return (
                int(t.get("_created_at_row") or 0),
                float(t.get("score") or 0.0),
                float(t.get("liquidity") or 0.0),
                float(t.get("volume_24h") or 0.0),
                float(t.get("market_cap") or 0.0),
            )

        best: dict[str, dict] = {}
        for t in raw:
            addr = str(t.get("address") or "").strip()
            if not addr:
                continue
            prev = best.get(addr)
            if prev is None or _key_for_best(t) > _key_for_best(prev):
                best[addr] = t

        tokens = list(best.values())

        # --- final GUI filters (hide stables/whitelist in category lists only) ---
        hidden = set()
        try:
            hidden |= set(ALWAYS_HIDE_IN_CATEGORIES)
        except Exception:
            pass
        try:
            hidden |= set(WHITELISTED_TOKENS)
        except Exception:
            pass

        filtered: list[dict] = []
        for t in tokens:
            addr = str(t.get("address") or "")
            # NOTE: keep the stable filter here if you want them gone everywhere.
            # If you want them visible in details but hidden only in category tables,
            # move the `_looks_stable(t)` check into the per-table filter instead.
            if addr in hidden or _looks_stable(t):
                continue
            filtered.append(t)

        logger.info("Fetched %d tokens from database (post-filter, de-duped)", len(filtered))
        return filtered

    except Exception as e:
        logger.error("Failed to fetch tokens from database: %s", e, exc_info=True)
        try:
            st.error(f"❌ Failed to fetch tokens from database: {e}")
        except Exception:
            pass
        return []


async def prune_old_tokens(days: int = 7) -> bool:
    try:
        cutoff = int(time.time()) - days * 24 * 3600
        async with connect_db() as db:
            await db.execute("DELETE FROM eligible_tokens WHERE timestamp < ?", (cutoff,))
            await db.commit()
        logger.info("Pruned tokens older than %d days", days)
        return True
    except Exception as e:
        logger.error("Failed to prune old tokens: %s", e)
        # Avoid calling st.* from background threads
        return False

# =============================
# Safety, Formatting & Grouping
# =============================

def emoji_safety(score):
    if score is None:
        return "❓"
    try:
        s = float(score)
        if s >= 70:
            return "🛡️"
        elif s >= 40:
            return "⚠️"
        else:
            return "💀"
    except Exception:
        return "❓"


def _normalize_labels(labels_raw: list | str | None) -> set:
    if not labels_raw:
        return set()
    if isinstance(labels_raw, str):
        try:
            labels_raw = json.loads(labels_raw)
        except Exception:
            labels_raw = [labels_raw]
    if not isinstance(labels_raw, (list, set, tuple)):
        return set()
    return {str(l).lower().strip() for l in labels_raw if l}

DANGER_LABELS = {
    "rugpull", "scam", "honeypot", "blacklisted", "malicious", "high_risk",
    "locked_liquidity_low", "no_liquidity", "suspicious_contract"
}

WARNING_LABELS = {
    "new_token", "low_liquidity", "unverified_contract", "recently_deployed",
    "moderate_risk", "unknown_team"
}

# Keep a single canonical _normalize_categories defined earlier; do not redefine.

# ---- Whitelists / UI hide lists (typed + import-friendly) ----
from typing import Set

# Local defaults (safe to ship)
DEFAULT_WHITELISTED_TOKENS: Set[str] = {
    "So11111111111111111111111111111111111111112",  # WSOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
    "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",   # JUP
    "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",   # mSOL
    "7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj",  # stSOL
    "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn",  # JitoSOL
    "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1",   # bSOL
    "HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3",  # PYTH
}
DEFAULT_ALWAYS_HIDE_IN_CATEGORIES: Set[str] = {
    "So11111111111111111111111111111111111111112",  # WSOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}

# Start with defaults
WHITELISTED_TOKENS: Set[str] = set(DEFAULT_WHITELISTED_TOKENS)
ALWAYS_HIDE_IN_CATEGORIES: Set[str] = set(DEFAULT_ALWAYS_HIDE_IN_CATEGORIES)

# Try to merge in package-level definitions if available
try:  # bundle path
    from solana_trading_bot_bundle.trading_bot.eligibility import (  # type: ignore[reportMissingImports]
        WHITELISTED_TOKENS as _PKG_WHITELIST,
        ALWAYS_HIDE_IN_CATEGORIES as _PKG_HIDE,
    )
    if _PKG_WHITELIST:
        WHITELISTED_TOKENS |= set(_PKG_WHITELIST)
    if _PKG_HIDE:
        ALWAYS_HIDE_IN_CATEGORIES |= set(_PKG_HIDE)
except Exception:
    pass


def determine_safety(token: dict) -> tuple[str, str]:
    addr = (token.get("address") or "").strip()
    liq_raw = token.get("liquidity") or 0
    try:
        liq = float(liq_raw)
    except Exception:
        liq = 0.0
    cts = int(token.get("creation_timestamp") or 0)
    is_new = (cts > 0 and (time.time() - cts) <= NEW_TOKEN_SAFETY_WINDOW_MIN * 60)

    if addr in WHITELISTED_TOKENS:
        base, reason = "safe", "whitelist"
    else:
        labels_raw = token.get("labels") or token.get("rugcheck_labels") or []
        labels = _normalize_labels(labels_raw)
        if not labels:
            base, reason = "warning", "no_labels"
        else:
            lset = set(labels)
            if lset & DANGER_LABELS:
                base, reason = "dangerous", "labels"
            elif lset & WARNING_LABELS:
                base, reason = "warning", "labels"
            else:
                base, reason = "safe", "labels_ok"

    if base == "safe" and (liq < SAFE_MIN_LIQUIDITY_USD or is_new):
        base = "warning"; reason = "new_or_low_liquidity"
    return base, reason


def new_coin_badge(token: dict) -> str:
    try:
        cts = int(token.get("creation_timestamp") or 0)  # epoch seconds
    except Exception:
        cts = 0
    addr = (token.get("address") or "").strip()
    if addr in WHITELISTED_TOKENS:
        return ""
    if cts and (time.time() - cts) <= NEW_TOKEN_BADGE_WINDOW_MIN * 60:
        return "🔵 New"
    return ""

# --- Numeric helpers and deduplication ---
import re as _re
_NUM_RE = _re.compile(r"[^0-9.\-]")

def _num(x, default=0.0) -> float:
    if x is None:
        return float(default)
    try:
        if isinstance(x, str):
            x = _NUM_RE.sub("", x)
        return float(x)
    except Exception:
        return float(default)

def enrich_mc_in_place(t: dict) -> None:
    mc = t.get("mc")
    if mc in (None, "", 0, "0"):
        mc = t.get("marketCap") or t.get("fdv") or t.get("fdvUsd") or t.get("market_cap")
    t["mc"] = _num(mc, 0.0)

def normalize_token_fields_in_place(t: dict) -> None:
    """
    Make live-scan tokens consistent with DB schema:
    - creation_timestamp: derive from createdAt if missing
    - liquidity: accept dicts (e.g., {'usd': ...})
    - labels: if a single 'rugcheck_label' exists, map to ['...']
    """
    # creation_timestamp from createdAt (already sec in our live path; guard if ms elsewhere)
    if not t.get("creation_timestamp"):
        created = t.get("createdAt") or t.get("created_at") or 0
        try:
            created = int(created or 0)
            if created > 0:
                t["creation_timestamp"] = created if created < 10**12 else created // 1000
        except Exception:
            pass

    # liquidity normalization
    liq = t.get("liquidity")
    if isinstance(liq, dict):
        t["liquidity"] = _num(liq.get("usd", 0), 0.0)
    else:
        t["liquidity"] = _num(liq, 0.0)

    # labels normalization (single -> list)
    if "labels" not in t:
        rl = t.get("rugcheck_label") or t.get("rugcheck_labels")
        if rl:
            t["labels"] = [rl] if isinstance(rl, str) else rl

def deduplicate_tokens(tokens: list[dict]) -> list[dict]:
    by_addr = {}
    for t in tokens or []:
        addr = (t.get("address") or "").strip()
        if not addr:
            continue
        prev = by_addr.get(addr)
        if not prev:
            by_addr[addr] = t
            continue
        def _f(k):
            try:
                return float(t.get(k) or 0), float(prev.get(k) or 0)
            except:
                return 0.0, 0.0
        for k in ("timestamp", "score", "liquidity", "mc"):
            cur, old = _f(k)
            if cur > old:
                prev[k] = t.get(k)
        for k in ("symbol","name"):
            if (not prev.get(k)) and t.get(k):
                prev[k] = t.get(k)
        if t.get("labels"):
            prev["labels"] = t["labels"]
    return list(by_addr.values())

# --- Rugcheck verification (live/cache/whitelist/failed) ---
from typing import Tuple, Dict, Any, List, Optional

# === Canonical .env loader (single source) ===============================
import os as _os
from pathlib import Path as _Path
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
except Exception:
    _load_dotenv = None

_APPDATA_ENV = _Path(_os.getenv("LOCALAPPDATA", _Path.home() / "AppData" / "Local")) / "SOLOTradingBot" / ".env"
if _load_dotenv:
    try:
        _load_dotenv(_APPDATA_ENV, override=False)
    except Exception:
        pass
# ========================================================================

_RC_CACHE: Dict[str, Dict[str, Any]] = {}

def _now() -> float: 
    return time.time()

def _rc_age_sec(meta: Optional[Dict[str, Any]]) -> float:
    if not meta or "fetched_at" not in meta:
        return float("inf")
    return max(0.0, _now() - float(meta["fetched_at"]))

# --- Hide helpers --------------------------------------------------------
import os, time

# Fallback to whatever your file defined; otherwise empty
try:
    _ADDR_HIDE = set(ALWAYS_HIDE_IN_CATEGORIES)
except Exception:
    _ADDR_HIDE = set()

def _csv_env(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {s.strip().upper() for s in str(raw).split(",") if s.strip()}

# Symbols/names to hide even if the address is different (WSOL/USDC clones, etc.)
ALWAYS_HIDE_SYMBOLS = _csv_env(
    "ALWAYS_HIDE_SYMBOLS",
    "SOL,WSOL,USDC,USDT,WRAPPED SOLANA,WRAPPED USD COIN,USD COIN,USDC.E"
)


# ------------------------------------------------------------------------

async def _fetch_rugcheck_labels(session, jwt: str, addr: str) -> Tuple[List[str], int]:
    base = "https://api.rugcheck.xyz/v1"
    headers = {"Authorization": f"Bearer {jwt}"} if jwt else {}
    timeout = aiohttp.ClientTimeout(total=12, sock_connect=5, sock_read=8)
    urls = [
        f"{base}/tokens/scan/solana/{addr}",
        f"{base}/tokens/{addr}",
        f"{base}/tokens/{addr}/labels",
    ]
    last_status = 0
    for url in urls:
        try:
            async with session.get(url, headers=headers, timeout=timeout) as resp:
                last_status = resp.status
                if resp.status != 200:
                    continue
                data = await resp.json()
                payload = data.get("data", data) if isinstance(data, dict) else {}
                labels = payload.get("labels") or payload.get("riskLabels") or payload.get("flags") or []
                return labels, 200
        except Exception:
            continue
    return [], last_status

async def verify_rugcheck_for_tokens(tokens: List[Dict[str, Any]], minutes_ttl: int = 15) -> Dict[str, int]:
    from solana_trading_bot_bundle.rugcheck_auth import get_rugcheck_token
    import inspect
    ttl_sec = max(60, minutes_ttl * 60)
    counters = dict(total=0, live=0, cache=0, skipped_whitelist=0, failed=0)
    whitelist = WHITELISTED_TOKENS

    try:
        jwt = (await get_rugcheck_token() if inspect.iscoroutinefunction(get_rugcheck_token) else get_rugcheck_token())
    except Exception:
        jwt = ""

    async with aiohttp.ClientSession() as session:
        for t in tokens or []:
            counters["total"] += 1
            addr = (t.get("address") or "").strip()
            if not addr:
                t["rugcheck_meta"] = {"status":"failed","source":"n/a","http_status":0,"fetched_at":_now(),"age_sec":0,"labels_sample":[]}
                counters["failed"] += 1
                continue
            if addr in whitelist:
                t["rugcheck_meta"] = {"status":"skipped_whitelist","source":"whitelist","http_status":200,"fetched_at":_now(),"age_sec":0,"labels_sample":[]}
                counters["skipped_whitelist"] += 1
                continue
            cached = _RC_CACHE.get(addr)
            if cached and _rc_age_sec(cached) <= ttl_sec:
                t["rugcheck_meta"] = {
                    "status":"ok_cache","source":"cache","http_status":cached.get("http_status",200),
                    "fetched_at":cached.get("fetched_at",_now()),"age_sec":_rc_age_sec(cached),
                    "labels_sample": list(cached.get("labels", []))[:5],
                }
                t.setdefault("labels", cached.get("labels", []))
                counters["cache"] += 1
                continue
            labels, http_status = ([], 0)
            if jwt:
                labels, http_status = await _fetch_rugcheck_labels(session, jwt, addr)
            status = "ok_live" if (http_status == 200 and isinstance(labels, list)) else "failed"
            meta = {"status":status,"source":"live" if status=="ok_live" else "live_error",
                    "http_status":http_status,"fetched_at":_now(),"age_sec":0,"labels_sample": list(labels)[:5]}
            t["rugcheck_meta"] = meta
            if status == "ok_live":
                t.setdefault("labels", labels)
                _RC_CACHE[addr] = {"labels":labels,"fetched_at":meta["fetched_at"],"http_status":http_status,"source":"live"}
                counters["live"] += 1
            else:
                counters["failed"] += 1
    return counters

def format_percent(value):
    try:
        if value is None:
            return "N/A"
        value = float(value)
        return f"🟢 {value:.2f}%" if value > 0 else f"🔴 {value:.2f}%" if value < 0 else f"{value:.2f}%"
    except Exception:
        return "N/A"

DEX_BASE = "https://dexscreener.com/solana"

def to_display_row(t: dict) -> dict:
    addr = (t.get("address") or "").strip()
    name = str(t.get('symbol') or t.get('name') or 'N/A')
    safety_label, _ = determine_safety(t)

    price = _num(t.get('price'), 0.0)
    liq   = _num(t.get('liquidity'), 0.0)
    # Robust MC fallback so categories work even if upstream uses other keys
    mc    = _num(t.get('mc') or t.get('market_cap') or t.get('fdv'), 0.0)
    vol   = _num(t.get('volume_24h') or (t.get('volume') or {}).get('h24'), 0.0)

    dex_url = f"{DEX_BASE}?q={addr}" if addr else ""
    row = {
        "Name": name,
        "Token Address": addr,
        "Dex": dex_url,
        "Safety": {"safe": "🛡️ Safe", "warning": "⚠️ Warning", "dangerous": "💀 Dangerous"}.get(safety_label, "❓ Unknown"),
        "Price": f"${price:,.6f}",
        "Liquidity": f"${liq:,.0f}",
        "Market Cap": f"${mc:,.0f}",
        "MC Tier": market_cap_badge(mc),
        "New": new_coin_badge(t),
        "Volume (24h)": f"${vol:,.0f}",
        "1H": format_percent(t.get("price_change_1h", None)),
        "6H": format_percent(t.get("price_change_6h", None)),
        "24H": format_percent(t.get("price_change_24h", None)),
    }
    return row


def partition_by_category(tokens: list[dict]):
    """
    Return dict with keys: new, large, mid, low.
    - New tokens are exclusive.
    - Hide core assets by address AND symbol/name (to filter WSOL/USDC clones).
    - Deduplicate by address across all groups.
    """
    import time
    now = int(time.time())
    new_cutoff = now - NEW_TOKEN_BADGE_WINDOW_MIN * 60 if 'NEW_TOKEN_BADGE_WINDOW_MIN' in globals() else now - 180*60

    groups = {"new": [], "large": [], "mid": [], "low": []}
    seen: set[str] = set()

    for t in tokens or []:
        addr = (t.get("address") or t.get("token_address") or "").strip()
        if not addr:
            continue
        if addr in seen:
            continue
        if _hidden_token(t):
            continue

        try:
            cts = int(t.get("creation_timestamp") or t.get("created_at") or 0)
        except Exception:
            cts = 0

        mc_val = _num(t.get("mc") or t.get("market_cap") or t.get("fdv") or 0, 0.0)

        # New is exclusive (but don't consider whitelisted core assets as "new")
        if cts >= new_cutoff and addr not in WHITELISTED_TOKENS:
            groups["new"].append(t)
            seen.add(addr)
            continue

        if mc_val >= 500_000:
            groups["large"].append(t)
        elif mc_val >= 100_000:
            groups["mid"].append(t)
        else:
            groups["low"].append(t)
        seen.add(addr)

    # sort each group by score desc (fallback to market cap)
    def _score(tok):
        try:
            return float(tok.get("score") or tok.get("market_cap") or tok.get("mc") or tok.get("fdv") or 0)
        except Exception:
            return 0.0

    for k in groups:
        groups[k].sort(key=_score, reverse=True)
    return groups

# =============================
# Bot Control
# =============================
def _log_rc(proc: subprocess.Popen):
    rc = proc.poll()
    logger.info("Bot subprocess terminated with return code: %s", rc)
    if rc not in (0, None):
        logger.error("Bot subprocess exited abnormally (code: %s)", rc)

def _stop_proc_tree(proc: subprocess.Popen, *, write_stop_flag: bool, grace: float = 2.0, term_wait: float = 5.0):
    """Gracefully stop a subprocess; escalate to terminate/kill with child cleanup."""
    if not proc or proc.poll() is not None:
        return

    # Optional cooperative stop
    if write_stop_flag:
        try:
            STOP_FLAG_PATH.write_text("1", encoding="utf-8")
            logger.info("Stop flag written to %s", STOP_FLAG_PATH)
        except Exception:
            pass

    # Give the bot a moment to exit on its own
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass

    # Then terminate process tree if still running
    try:
        ps_proc = psutil.Process(proc.pid)
    except psutil.Error:
        ps_proc = None

    if ps_proc and ps_proc.is_running():
        for child in ps_proc.children(recursive=True):
            try:
                child.terminate()
            except psutil.Error:
                pass
        try:
            ps_proc.terminate()
        except psutil.Error:
            pass

        try:
            proc.wait(timeout=term_wait)
            logger.info("Stopped bot (PID: %s)", proc.pid)
        except subprocess.TimeoutExpired:
            try:
                ps_proc.kill()
            except psutil.Error:
                pass
            logger.warning("Forced termination of bot (PID: %s)", proc.pid)

# =============================
# Bot spawn/stop helpers
# =============================

def _build_pythonpath_for_spawn() -> str:
    """Compose a PYTHONPATH that lets the child import the bundle as a package."""
    parts: list[str] = []
    # Prefer repo dir and bundle dir first
    parts.extend([str(_PKG_ROOT), str(_PKG_DIR)])
    # Keep any existing PYTHONPATH entries (dedup)
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.extend([p for p in existing.split(os.pathsep) if p])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return os.pathsep.join(out)

# =============================
# UI Helpers
# =============================
from collections import deque

@st.cache_data(ttl=10, show_spinner=False)
def read_logs(_k: int, log_file: Path = LOG_PATH, max_lines: int = 300) -> str:
    """Tail logs efficiently; cache invalidates when _k (mtime) changes."""
    try:
        if log_file.exists():
            dq = deque(maxlen=int(max_lines))
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    dq.append(line)
            return "".join(dq) if dq else "(empty log file)"
        return "⚠️ No logs found."
    except Exception as e:
        # (keep this as-is)
        logger.error("Failed to read logs: %s", e)
        return f"❌ Failed to read logs: {e}"

def _mtime(p: Path) -> int:
    try:
        return int(p.stat().st_mtime)
    except Exception:
        return 0

# ---- DRY JSON file reader (cached via mtime key) ----
def _read_json_file(path: Path) -> dict | list | None:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def _read_rugcheck_status(_k: int) -> dict:
    """Read Rugcheck status JSON; cache invalidates when file mtime changes."""
    return _read_json_file(STATUS_FILE) or {}

def _fmt_age(ts) -> str:
    """Return a compact 'X ago' string for a unix timestamp, or ''."""
    try:
        if ts is None:
            return ""
        age = max(0, int(time.time() - int(ts)))
        if age < 60:
            return f"{age}s"
        m, s = divmod(age, 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        if h < 24:
            return f"{h}h {m}m"
        d, h = divmod(h, 24)
        return f"{d}d {h}h"
    except Exception:
        return ""


def show_rugcheck_banner():
    try:
        status = _read_rugcheck_status(_mtime(Path(os.getenv("STATUS_FILE", str(APP_DIR / "rugcheck_status.json"))))) or {}
    except Exception:
        status = {}
    enabled = bool(status.get("enabled"))
    available = bool(status.get("available"))
    msg_raw = status.get("message") or ""
    msg = " ".join(str(msg_raw).split()).strip()
    age_text = _fmt_age(status.get("timestamp"))
    suffix = f" (updated {age_text} ago)" if age_text else ""
    if not enabled:
        st.info(f"🛈 Rugcheck disabled — tokens allowed{suffix}" + (f". {msg}" if msg else ""))
        return
    if not available:
        st.warning(f"⚠️ Rugcheck unavailable{suffix}" + (f". {msg}" if msg else ""))
        return
    st.success(f"✅ Rugcheck active{suffix}" + (f". {msg}" if msg else ""))

@st.cache_data(show_spinner=False)
def _read_rugcheck_failures(_k: int) -> list:
    data = _read_json_file(Path(os.getenv("FAILURES_FILE", str(APP_DIR / "rugcheck_failures.json"))))
    return data if isinstance(data, list) else []

def display_rugcheck_failures_table():
    failures = _read_rugcheck_failures(_mtime(Path(os.getenv("FAILURES_FILE", str(APP_DIR / "rugcheck_failures.json")))))
    if not failures:
        st.info("No Rugcheck validation failures recorded.")
        return
    st.warning("Some tokens failed Rugcheck validation and were excluded (unless whitelisted):")
    st.dataframe([{"Token Address": f.get("address", ""), "Reason": f.get("reason", "")} for f in failures],
                 width="stretch", hide_index=True, height=_df_height(len(failures)))

# -------------------------
# UI layout: title + initial banner
# -------------------------
st.title("📈 SOLO Meme Coin Trading Bot")
st.markdown("<style>.stAlert { margin-top: .25rem; }</style>", unsafe_allow_html=True)
show_rugcheck_banner()

# -------------------------
# Initialize async tasks in background thread (schema ensure etc.)
# -------------------------
INIT_ERROR_MSG = None
def initialize_async_tasks():
    global INIT_ERROR_MSG
    try:
        run_async_task(ensure_eligible_tokens_schema)
        run_async_task(ensure_price_cache_schema)
    except Exception as e:
        logger.error("Initialization failed", exc_info=True)
        INIT_ERROR_MSG = str(e)

threading.Thread(target=initialize_async_tasks, daemon=True).start()
if INIT_ERROR_MSG:
    st.error(f"❌ Initialization failed: {INIT_ERROR_MSG}")

# -------------------------
# Tabs
# -------------------------
BASE_TAB_LABELS = [
    "🔧 Configuration",
    "⚙️ Bot Control",
    "🪙 Discovered Tokens",
    "💹 P&L",
    "📊 Status",
    "🛡 Rugcheck",
]
DEV_EXTRA_LABELS = ["🧪 Diagnostics", "📜 Trade Logs"]

DEV_MODE = str(os.getenv("GUI_DEV_MODE", "0")).lower() in ("1", "true", "yes")
if DEV_MODE:
    TAB_LABELS = BASE_TAB_LABELS[:-1] + DEV_EXTRA_LABELS + BASE_TAB_LABELS[-1:]
    (tab_config, tab_control, tab_tokens, tab_pl, tab_status, tab_diag, tab_tradelogs, tab_rc) = st.tabs(TAB_LABELS)
else:
    TAB_LABELS = BASE_TAB_LABELS[:]
    tab_config, tab_control, tab_tokens, tab_pl, tab_status, tab_rc = st.tabs(TAB_LABELS)

# Sticky tab shim (persist across reruns)
DEFAULT_TAB = "⚙️ Bot Control"
if "active_tab" not in st.session_state:
    st.session_state.active_tab = st.query_params.get("tab") or DEFAULT_TAB

def goto_tab(tab_label: str) -> None:
    st.session_state.active_tab = tab_label
    st.query_params["tab"] = tab_label
    wanted_js = json.dumps(tab_label)
    st.markdown(
        f"""
        <script>
          (function(){{
            const wanted = {wanted_js};
            function tryClick(){{
              const tabs = window.parent.document.querySelectorAll('button[role="tab"]');
              for (const t of tabs) {{ if (t.innerText.trim() === wanted) {{ t.click(); return; }} }}
              setTimeout(tryClick, 80);
            }}
            setTimeout(tryClick, 0);
          }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

desired_label = st.session_state.pop("_next_tab", None) or st.session_state.get("active_tab", DEFAULT_TAB)
if desired_label not in TAB_LABELS:
    desired_label = DEFAULT_TAB
if desired_label != st.session_state.get("active_tab"):
    try:
        goto_tab(desired_label)
    except Exception:
        st.session_state["active_tab"] = desired_label

# -------------------------
# Rugcheck tab
# -------------------------
with tab_rc:
    st.header("🛡 Rugcheck Validation Failures")
    display_rugcheck_failures_table()

# -------------------------
# Configuration tab
# -------------------------
with tab_config:
    st.header("🔧 Bot Configuration")
    st.subheader("Bot Settings")
    cfg = load_config()
    if cfg is None:
        st.error("❌ Failed to load configuration. Please check your setup.")
        st.stop()
    bot_cfg = cfg.setdefault("bot", {})
    disc_cfg = cfg.setdefault("discovery", {})
    for sec in ("low_cap", "mid_cap", "large_cap", "newly_launched"):
        disc_cfg.setdefault(sec, {})

    def _default_token_db_path() -> str:
        if os.name == "nt":
            base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            return str(base / "SOLOTradingBot" / "tokens.sqlite3")
        return str(Path.home() / ".cache" / "SOLOTradingBot" / "tokens.sqlite3")
    def _default_log_dir() -> str:
        if os.name == "nt":
            base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            return str(base / "SOLOTradingBot" / "logs")
        return str(Path.home() / ".cache" / "SOLOTradingBot" / "logs")
    def _looks_windows_path(p: str) -> bool:
        return isinstance(p, str) and (":\\" in p or p.startswith("\\\\"))
    def _normalize_path_for_os(raw: str, is_dir: bool = False) -> tuple[str, bool]:
        if not raw:
            return (_default_log_dir() if is_dir else _default_token_db_path(), False)
        if os.name != "nt" and _looks_windows_path(raw):
            return (_default_log_dir() if is_dir else _default_token_db_path(), True)
        return (raw, False)

    bot_cfg["minimum_wallet_balance_usd"] = st.number_input("Minimum Wallet Balance (USD)", key="bot_min_wallet_usd", min_value=0.0, value=float(bot_cfg.get("minimum_wallet_balance_usd", 0.20)), step=0.10, format="%.2f")
    bot_cfg["cycle_interval"] = st.number_input("Cycle Interval (seconds)", key="bot_cycle_interval", min_value=30, value=int(bot_cfg.get("cycle_interval", 60)), step=10)
    bot_cfg["profit_target"] = st.number_input("Profit Target Ratio", key="bot_profit_target", min_value=1.0, value=float(bot_cfg.get("profit_target", 1.5)), step=0.1, format="%.2f")
    bot_cfg["stop_loss"] = st.number_input("Stop Loss Ratio", key="bot_stop_loss", min_value=0.1, max_value=1.0, value=float(bot_cfg.get("stop_loss", 0.7)), step=0.1, format="%.2f")

    st.subheader("Discovery Settings")
    with st.expander("Pre-Enrichment Work-Set & Lane Caps", expanded=False):
        disc_cfg.setdefault("workset_cap", 120)
        disc_cfg.setdefault("quality_cap", 120)
        disc_cfg.setdefault("new_coin_cap", 40)
        disc_cfg.setdefault("spike_cap", 0)
        disc_cfg["workset_cap"] = st.number_input("WORKSET_CAP (max tokens enriched per cycle)", key="disc_workset_cap", min_value=20, max_value=500, step=10, value=int(disc_cfg.get("workset_cap", 120)))
        cols_caps = st.columns(3)
        with cols_caps[0]:
            disc_cfg["quality_cap"] = st.number_input("QUALITY_CAP", key="disc_quality_cap", min_value=0, max_value=500, step=10, value=int(disc_cfg.get("quality_cap", 120)))
        with cols_caps[1]:
            disc_cfg["new_coin_cap"] = st.number_input("NEW_COIN_CAP", key="disc_new_cap", min_value=0, max_value=500, step=5, value=int(disc_cfg.get("new_coin_cap", 40)))
        with cols_caps[2]:
            disc_cfg["spike_cap"] = st.number_input("SPIKE_CAP", key="disc_spike_cap", min_value=0, max_value=500, step=5, value=int(disc_cfg.get("spike_cap", 0)))
        st.caption("The work-set used for enrichment is built from lane caps (quality/new/spike) and then clamped by WORKSET_CAP.")

    with st.expander("Newly Launched Tokens"):
        new_cfg = disc_cfg["newly_launched"]
        new_cfg["max_token_age_minutes"] = st.number_input("Max Token Age (minutes)", key="new_max_token_age_minutes", min_value=1, value=int(new_cfg.get("max_token_age_minutes", 180)), step=1)
        new_cfg["liquidity_threshold"] = st.number_input("Min Liquidity (USD)", key="new_liquidity_threshold", min_value=10, value=int(new_cfg.get("liquidity_threshold", 100)), step=10)
        new_cfg["volume_threshold"] = st.number_input("Min 24H Volume (USD)", key="new_volume_threshold", min_value=10, value=int(new_cfg.get("volume_threshold", 50)), step=10)
        new_cfg["max_rugcheck_score"] = st.number_input("Max Rugcheck Score", key="new_max_rugcheck_score", min_value=100, value=int(new_cfg.get("max_rugcheck_score", 2000)), step=100)

    with st.expander("Low Cap Tokens"):
        low_cfg = disc_cfg["low_cap"]
        low_cfg["max_market_cap"] = st.number_input("Max Market Cap (USD)", key="low_max_market_cap", min_value=1000, value=int(low_cfg.get("max_market_cap", 100000)), step=1000)
        low_cfg["liquidity_threshold"] = st.number_input("Min Liquidity (USD)", key="low_liquidity_threshold", min_value=10, value=int(low_cfg.get("liquidity_threshold", 100)), step=10)
        low_cfg["volume_threshold"] = st.number_input("Min 24H Volume (USD)", key="low_volume_threshold", min_value=10, value=int(low_cfg.get("volume_threshold", 50)), step=10)

    with st.expander("Mid Cap Tokens"):
        mid_cfg = disc_cfg["mid_cap"]
        mid_cfg["max_market_cap"] = st.number_input("Max Market Cap (USD)", key="mid_max_market_cap", min_value=1000, value=int(mid_cfg.get("max_market_cap", 500000)), step=1000)
        mid_cfg["liquidity_threshold"] = st.number_input("Min Liquidity (USD)", key="mid_liquidity_threshold", min_value=10, value=int(mid_cfg.get("liquidity_threshold", 500)), step=10)
        mid_cfg["volume_threshold"] = st.number_input("Min 24H Volume (USD)", key="mid_volume_threshold", min_value=10, value=int(mid_cfg.get("volume_threshold", 250)), step=10)

    with st.expander("Large Cap Tokens"):
        lg_cfg = disc_cfg["large_cap"]
        lg_cfg["liquidity_threshold"] = st.number_input("Min Liquidity (USD)", key="large_liquidity_threshold", min_value=10, value=int(lg_cfg.get("liquidity_threshold", 1000)), step=10)
        lg_cfg["volume_threshold"] = st.number_input("Min 24H Volume (USD)", key="large_volume_threshold", min_value=10, value=int(lg_cfg.get("volume_threshold", 500)), step=10)

    st.subheader("Global Discovery Limits")
    disc_cfg["max_price_change"] = st.number_input("Max 24H Price Change (%)", key="global_max_price_change", min_value=0.0, value=float(disc_cfg.get("max_price_change", 100.0)), step=1.0, format="%.2f")
    disc_cfg["min_holder_count"] = st.number_input("Min Holder Count", key="global_min_holder_count", min_value=0, value=int(disc_cfg.get("min_holder_count", 50)), step=1)

    _env_topn = os.getenv("TOP_N_PER_CATEGORY")
    try:
        _env_topn_val = int(_env_topn) if _env_topn not in (None, "") else None
    except Exception:
        _env_topn_val = None
    _config_topn_val = int(disc_cfg.get("shortlist_per_bucket", 5))
    _initial_topn = _env_topn_val if _env_topn_val is not None else _config_topn_val
    st.subheader("Shortlist")
    disc_cfg["shortlist_per_bucket"] = st.number_input("Top N per category (GUI & Bot)", key="global_shortlist_per_bucket", min_value=1, max_value=50, value=_initial_topn, step=1)
    if _env_topn_val is not None:
        st.caption(f"🔧 Environment override active: TOP_N_PER_CATEGORY={_env_topn_val} (will take precedence over config).")

    st.subheader("Paths & Storage")
    paths_cfg = cfg.setdefault("paths", {})
    current_db = paths_cfg.get("token_cache_path") or cfg.get("token_cache_path") or _default_token_db_path()
    current_logs = paths_cfg.get("log_dir") or cfg.get("log_dir") or _default_log_dir()
    db_input_raw = st.text_input("Token DB path", key="paths_token_db", value=str(current_db))
    logs_input_raw = st.text_input("Log directory", key="paths_log_dir", value=str(current_logs))
    db_input_norm, db_rewritten = _normalize_path_for_os(db_input_raw, is_dir=False)
    logs_input_norm, logs_rewritten = _normalize_path_for_os(logs_input_raw, is_dir=True)
    paths_cfg["token_cache_path"] = db_input_norm
    paths_cfg["log_dir"] = logs_input_norm
    if db_rewritten:
        st.caption(f"⚠️ Provided DB path looked Windows-style on this OS; using '{db_input_norm}'.")
    if logs_rewritten:
        st.caption(f"⚠️ Provided log directory looked Windows-style on this OS; using '{logs_input_norm}'.")
    if st.button("💾 Save Config"):
        if save_config(cfg):
            st.success("✅ Configuration saved.")
            load_config.clear()
            _rebind_logger_file_handler()

# -------------------------
# Diagnostics & other tabs left intact (DEV only)
# -------------------------
if DEV_MODE:
    with tab_diag:
        st.header("🧪 Diagnostics")
        try:
            proc = psutil.Process(os.getpid())
        except Exception:
            proc = None
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Python", sys.version.split()[0])
            st.metric("Streamlit", st.__version__)
        with c2:
            st.metric("PID (GUI)", os.getpid())
            st.metric("Bot Running", "Yes" if st.session_state.get("bot_running") else "No")
        with c3:
            if proc:
                st.metric("Memory (GUI)", f"{proc.memory_info().rss/1e6:,.0f} MB")
            else:
                st.write("Install `psutil` for memory stats")
        st.subheader("Paths / Config")
        _paths = cfg.get("paths", {})
        st.json({"token_db": _paths.get("token_cache_path"), "log_dir": _paths.get("log_dir"), "CONFIG_PATH": str(CONFIG_PATH), "DB_PATH": str(DB_PATH), "LOG_PATH": str(LOG_PATH)})
        async def _diag_prepare_schema():
            try:
                await ensure_eligible_tokens_schema()
            except Exception:
                pass
            try:
                await ensure_price_cache_schema()
            except Exception:
                pass
            try:
                await ensure_shortlist_tokens_schema()
            except Exception:
                pass
            try:
                async with connect_db() as _db:
                    try:
                        await _ensure_core_tables(_db)
                        await _db.commit()
                    except Exception:
                        pass
            except Exception:
                pass
        async def _diag_inspect_db():
            info = {"tables": [], "counts": {}}
            try:
                await _diag_prepare_schema()
                async with connect_db() as db:
                    async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
                        rows = await cur.fetchall()
                        info["tables"] = [r[0] for r in rows] if rows else []
                    for tname in ["eligible_tokens", "shortlist_tokens", "price_cache", "tokens", "trade_history"]:
                        try:
                            cur = await db.execute(f"SELECT COUNT(*) FROM {tname}")
                            row = await cur.fetchone()
                            info["counts"][tname] = int(row[0]) if row else 0
                        except Exception:
                            info["counts"][tname] = "n/a"
            except Exception as e:
                info = {"error": str(e)}
            return info
        def _diag_tail_logs(lines: int = 300) -> str:
            try:
                p = Path(str(LOG_PATH))
                if p.exists():
                    dq = deque(maxlen=lines)
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            dq.append(line)
                    return "".join(dq) if dq else "(empty log file)"
                return "No log file yet. Trigger some actions first."
            except Exception as e:
                return f"Failed to read logs: {e}"
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔍 Inspect DB", key="diag_btn_inspect_db"):
                res = run_async_task(_diag_inspect_db, timeout=40) or {}
                st.json(res or {"error": "inspection failed"})
        with c2:
            if st.button("📜 Tail Logs", key="diag_btn_tail_logs"):
                st.text_area("Logs (last 300 lines)", value=_diag_tail_logs(), height=320, key="diag_txt_logs")
        with c3:
            if st.button("🧩 Runtime Info", key="diag_btn_env"):
                st.json({"python": sys.version, "executable": sys.executable, "cwd": os.getcwd(), "frozen": bool(getattr(sys, "frozen", False)), "PYTHONPATH": os.environ.get("PYTHONPATH", "")})
                try:
                    _tok = fetch_tokens_from_db(_mtime(DB_PATH), int(st.session_state.get("fallback_hours", 24)), st.session_state.get("last_token_refresh", 0.0))
                    if isinstance(_tok, list) and len(_tok) == 0:
                        st.warning("⚠️ No tokens found in database. Run the bot to populate token data.")
                except Exception:
                    pass

if DEV_MODE:
    with tab_tradelogs:
        st.header("📜 Trade Logs")
        if st.button("🔄 Refresh Logs"):
            read_logs.clear()
        logs = read_logs(_mtime(LOG_PATH))
        st.text_area("Recent Logs", value=logs, height=500)

# -------------------------
# Bot Control tab
# -------------------------
with tab_control:
    st.header("⚙️ Bot Control")

    # detect running state
    running = False
    try:
        bot_pid = st.session_state.get("bot_pid")
        running = bool(bot_pid) and psutil.Process(int(bot_pid)).is_running()
    except Exception:
        running = False

    pid = st.session_state.get("bot_pid") or None
    if running and "bot_started_at" not in st.session_state:
        try:
            if pid:
                st.session_state.bot_started_at = psutil.Process(int(pid)).create_time()
            else:
                st.session_state.bot_started_at = time.time()
        except Exception:
            st.session_state.bot_started_at = time.time()

    def _fmt_tdelta(seconds: float) -> str:
        try:
            seconds = max(0, int(seconds))
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        except Exception:
            return "—"

    def _get_db_path() -> str:
        for candidate in (
            lambda: str(globals().get("DB_PATH", "")),
            lambda: st.session_state.get("DB_PATH"),
            lambda: os.path.join(os.path.expanduser("~"), "Library/Application Support/SOLOTradingBot", "tokens.sqlite3"),
        ):
            try:
                pth = candidate()
                if pth:
                    return pth
            except Exception:
                pass
        return "tokens.sqlite3"

    def _get_log_path() -> str:
        for candidate in (
            lambda: str(globals().get("LOG_PATH", "")),
            lambda: st.session_state.get("LOG_PATH"),
            lambda: os.path.join(os.path.expanduser("~"), "Library/Application Support/SOLOTradingBot", "logs", "bot.log"),
        ):
            try:
                pth = candidate()
                if pth:
                    return pth
            except Exception:
                pass
        return "bot.log"

    def _safe_int(x):
        try:
            return int(x)
        except Exception:
            return 0

    def _db_count(table: str) -> int:
        try:
            conn = __import__("sqlite3").connect(_get_db_path(), timeout=1)
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if _safe_int(cur.fetchone()[0]) == 0:
                    return 0
                cur.execute(f"SELECT COUNT(1) FROM {table}")
                row = cur.fetchone()
                return _safe_int(row[0]) if row else 0
            finally:
                conn.close()
        except Exception:
            return 0

    def _eligible_latest_ts() -> int:
        try:
            conn = __import__("sqlite3").connect(_get_db_path(), timeout=1)
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='eligible_tokens'")
                if _safe_int(cur.fetchone()[0]) == 0:
                    return 0
                cur.execute("PRAGMA table_info(eligible_tokens)")
                cols = [r[1] for r in cur.fetchall()]
                if "timestamp" in cols:
                    cur.execute("SELECT MAX(COALESCE(timestamp,0)) FROM eligible_tokens")
                    v = cur.fetchone()
                    return _safe_int(v[0] or 0)
                return 0
            finally:
                conn.close()
        except Exception:
            return 0

    def _tail_last_timestamp_from_log() -> float | None:
        try:
            path = _get_log_path()
            if not Path(path).exists():
                return None
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 12_000))
                data = f.read().decode("utf-8", "ignore")
            for line in data.splitlines()[::-1]:
                if not line:
                    continue
                try:
                    from datetime import datetime
                    ts = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
                    return ts.timestamp()
                except Exception:
                    continue
            return None
        except Exception:
            return None

    # backfill helper (runs once per session)
    def _backfill_json_data():
        try:
            conn = __import__("sqlite3").connect(_get_db_path(), timeout=2)
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='eligible_tokens'")
                if _safe_int(cur.fetchone()[0]) == 0:
                    return
                cur.execute("PRAGMA table_info(eligible_tokens)")
                cols = {r[1] for r in cur.fetchall()}
                if "data" not in cols:
                    return
                cur.execute("SELECT COUNT(1) FROM eligible_tokens WHERE data IS NOT NULL")
                if _safe_int(cur.fetchone()[0]) > 0:
                    return
                mapping = [
                    ("address", "address"), ("name", "name"), ("symbol", "symbol"),
                    ("volume_24h", "volume_24h"), ("liquidity", "liquidity"), ("mc", "market_cap"),
                    ("price", "price"), ("price_change_1h", "price_change_1h"), ("price_change_6h", "price_change_6h"),
                    ("price_change_24h", "price_change_24h"), ("score", "score"), ("categories", "categories"),
                    ("timestamp", "timestamp"), ("creation_timestamp", "creation_timestamp"),
                ]
                kv = []
                for key, col in mapping:
                    if col in cols:
                        kv.append(f"'{key}', {col}")
                if not kv:
                    return
                sql = f"UPDATE eligible_tokens SET data = json_object({', '.join(kv)}) WHERE data IS NULL"
                cur.execute(sql)
                conn.commit()
                st.caption("✅ Backfilled JSON in eligible_tokens.data for display.")
            finally:
                conn.close()
        except Exception:
            pass

    if not st.session_state.get("_did_json_backfill", False):
        _backfill_json_data()
        st.session_state["_did_json_backfill"] = True

    uptime_s = (time.time() - st.session_state.get("bot_started_at", time.time())) if running else 0
    last_hb_ts = _tail_last_timestamp_from_log()
    last_hb_ago = (time.time() - last_hb_ts) if (last_hb_ts) else None
    eligible_count = _db_count("eligible_tokens")
    shortlisted = eligible_count
    db_records = _db_count("tokens") + _db_count("token_cache")
    latest_ts = _eligible_latest_ts()
    last_update_age = (time.time() - latest_ts) if latest_ts else None
    prev = st.session_state.get("_bot_control_prev", {})
    deltas = {
        "eligible": eligible_count - int(prev.get("eligible", 0)),
        "short": shortlisted - int(prev.get("short", 0)),
        "db": db_records - int(prev.get("db", 0)),
    }
    st.session_state["_bot_control_prev"] = {"eligible": eligible_count, "short": shortlisted, "db": db_records}

    col_status, col_start, col_stop, col_spacer = st.columns([2.2, 0.9, 0.9, 7.0], gap="small")
    with col_status:
        st.markdown(
            f"""
            <div class="status-card">
              <div class="status-badge {'ok' if running else 'stop'}">
                Status: {'Running' if running else 'Stopped'}
              </div>
              <div class="muted">
                PID: {pid or '—'} &nbsp;&nbsp; Uptime: { _fmt_tdelta(uptime_s) if running else '—'}<br/>
                Last heartbeat: { (str(int(last_hb_ago)) + 's ago') if last_hb_ago is not None else '—' }
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Start button column
    with col_start:
        if st.button("▶ Start Bot", disabled=running, key="btn_start_bot_compact"):
            st.session_state["_next_tab"] = "⚙️ Bot Control"
            if not running:
                # Remove stop flag if present
                stop_flag = globals().get("STOP_FLAG_PATH")
                if stop_flag is not None:
                    try:
                        # Path.unlink may or may not support missing_ok depending on Python version
                        try:
                            stop_flag.unlink(missing_ok=True)
                        except TypeError:
                            stop_flag.unlink()
                    except Exception:
                        logger.debug("Couldn't unlink STOP_FLAG_PATH (ignored).", exc_info=True)

                # Attempt to start the bot if a start function is available
                start_fn = globals().get("start_bot")
                if callable(start_fn):
                    try:
                        start_fn()
                    except Exception:
                        logger.exception("start_bot() raised an exception")
                else:
                    logger.warning("start_bot() not available to call.")

                st.session_state.bot_started_at = time.time()
                # Try to refresh the app so status updates immediately
                try:
                    st.experimental_rerun()
                except Exception:
                    try:
                        st.rerun()
                    except Exception:
                        pass

    # Stop button column
    with col_stop:
        if st.button("🛑 Stop Bot", disabled=not running, key="btn_stop_bot_compact"):
            st.session_state["_next_tab"] = "⚙️ Bot Control"
            if running:
                stop_fn = globals().get("stop_bot")
                if callable(stop_fn):
                    try:
                        stop_fn()
                    except Exception:
                        logger.exception("stop_bot() raised an exception")
                else:
                    logger.warning("stop_bot() not available to call.")

                # Ensure stop-flag is present if module expects it
                stop_flag = globals().get("STOP_FLAG_PATH")
                if stop_flag is not None:
                    try:
                        # create the file if not exists
                        stop_flag.parent.mkdir(parents=True, exist_ok=True)
                        stop_flag.write_text("stop", encoding="utf-8")
                    except Exception:
                        logger.debug("Couldn't create STOP_FLAG_PATH (ignored).", exc_info=True)

                try:
                    st.experimental_rerun()
                except Exception:
                    try:
                        st.rerun()
                    except Exception:
                        pass

    st.markdown('<hr class="thin-rule" />', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4, gap="large")
    with m1:
        st.metric("Eligible tokens", eligible_count, f"{deltas['eligible']:+d}")
    with m2:
        st.metric("Shortlisted", shortlisted, f"{deltas['short']:+d}")
    with m3:
        st.metric("Last update age", (f"{int(last_update_age)} s" if last_update_age is not None else "—"))
    with m4:
        st.metric("DB records", db_records, f"{deltas['db']:+d}")
# -------------------------
# Discovered Tokens tab (updated to redraw every rerun)
# -------------------------
with tab_tokens:
    st.header("🪙 Discovered Tokens")
    # one-time bootstrap to make first landing refresh live
    if not st.session_state.get("discovery_bootstrapped"):
        st.session_state["discovery_bootstrapped"] = True
        try:
            fetch_tokens_from_db.clear()
        except Exception:
            pass
        st.session_state["last_token_refresh"] = 0.0

    AUTO_SECS = int(os.getenv("GUI_AUTO_REFRESH_SECS", "60"))
    st.session_state.setdefault("disc_auto_refresh", False)
    st.session_state.setdefault("disc_refresh_interval_ms", AUTO_SECS * 1000)
    st.session_state.setdefault("last_token_refresh", 0.0)
    st.session_state.setdefault("fallback_hours", 24)

    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore
    except Exception:
        st_autorefresh = None

    enabled = st.checkbox(f"Enable Auto-Refresh (every {AUTO_SECS} seconds)", key="disc_auto_refresh")
    if enabled and st_autorefresh:
        st_autorefresh(interval=st.session_state.get("disc_refresh_interval_ms", AUTO_SECS * 1000), key="disc_autorefresh_loop")
    elif enabled and not st_autorefresh:
        st.info("Auto-refresh requires the optional `streamlit-autorefresh` package. Install it with: `pip install streamlit-autorefresh`.")

    def _do_refresh():
        try:
            fetch_tokens_from_db.clear()
        except Exception:
            pass
        st.session_state["last_token_refresh"] = time.time()
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    if st.button("🔄 Refresh Tokens", key="manual_refresh_tokens"):
        _do_refresh()

    _ = st.number_input("Include DB fallback if live scan is empty (hours)", min_value=0, max_value=168, step=1, key="fallback_hours")

    # Use a safe lookup for DB_PATH so this snippet doesn't raise if DB_PATH isn't defined
    db_path_candidate = globals().get("DB_PATH", None)
    db_mtime = _mtime(db_path_candidate)
    refresh_tick = float(st.session_state.get("last_token_refresh", 0.0))
    fallback_hours = int(st.session_state.get("fallback_hours", 24))
    tokens = fetch_tokens_from_db(db_mtime, fallback_hours, refresh_tick)

    # compile once (ensure `re` is imported at module scope)
    _NUM_ONLY = re.compile(r"[^0-9eE\.\-]+")

    def _to_float_ui(v):
        if v is None:
            return math.nan
        if isinstance(v, (int, float)):
            return float(v)
        s = _NUM_ONLY.sub("", str(v))
        try:
            return float(s) if s else math.nan
        except Exception:
            return math.nan

    def _safety_str(t):
        lvl, _ = determine_safety(t)
        return {"safe": "🛡️ Safe", "warning": "⚠️ Warning", "dangerous": "⛔ Dangerous"}.get(lvl, "❔ Unknown")

    def _dex_url_for(token):
        pair = token.get("pair_address") or token.get("pair") or token.get("pairAddress") or token.get("pairAddressBase58") or ""
        addr = token.get("address") or token.get("token_address") or token.get("mint") or ""
        pair = str(pair).strip()
        addr = str(addr).strip()
        base = "https://dexscreener.com/solana"
        if pair:
            return f"{base}/{pair}"
        if addr:
            return f"{base}/{addr}"
        return base

    # mirror pct aliases
    for _t in tokens:
        _t.setdefault("price_change_1h", _t.get("pct_change_1h"))
        _t.setdefault("price_change_6h", _t.get("pct_change_6h"))
        _t.setdefault("price_change_24h", _t.get("pct_change_24h"))

    # Partition
    try:
        groups = partition_by_category(tokens)
        high = groups.get("large", []) or groups.get("high", [])
        mid = groups.get("mid", [])
        low = groups.get("low", [])
        new = groups.get("new", [])
    except Exception:
        HIGH_CAP_USD = 500_000
        MID_CAP_USD = 100_000
        high, mid, low, new = [], [], [], []
        now = int(time.time())
        cutoff = now - (NEW_TOKEN_BADGE_WINDOW_MIN * 60)
        for t in tokens:
            try:
                cts = int(t.get("creation_timestamp") or 0)
            except Exception:
                cts = 0
            if cts and cts >= cutoff:
                new.append(t)
                continue
            mc_val = _to_float_ui(t.get("mc") or t.get("market_cap"))
            mc_val = 0.0 if (mc_val != mc_val) else mc_val
            if mc_val >= HIGH_CAP_USD:
                high.append(t)
            elif mc_val >= MID_CAP_USD:
                mid.append(t)
            else:
                low.append(t)

    env_topn = os.getenv("TOP_N_PER_CATEGORY")
    try:
        _env_topn_val = int(env_topn) if env_topn not in (None, "") else None
    except Exception:
        _env_topn_val = None
    try:
        cfg_topn = int(cfg.get("discovery", {}).get("shortlist_per_bucket", 5) or 5)
    except Exception:
        cfg_topn = 5
    _topn_cfg = _env_topn_val if _env_topn_val is not None else cfg_topn
    _topn_cfg = max(1, min(int(_topn_cfg), 5))

    def _score_tuple(t):
        mc = _to_float_ui(t.get("mc") or t.get("market_cap"))
        liq = _to_float_ui(t.get("liquidity"))
        vol = _to_float_ui(t.get("volume_24h"))
        mc = 0.0 if (mc != mc) else mc
        liq = 0.0 if (liq != liq) else liq
        vol = 0.0 if (vol != vol) else vol
        return (mc, liq, vol)

    def _top_n(items, n=5):
        return sorted(items, key=_score_tuple, reverse=True)[:n]

    high = _top_n(high, _topn_cfg)
    mid = _top_n(mid, _topn_cfg)
    low = _top_n(low, _topn_cfg)
    new = _top_n(new, _topn_cfg)

    def _render_bucket(label, items):
        if not items:
            return
        import pandas as pd
        # build rows
        rows = []
        for t in items:
            addr = (t.get("address") or t.get("token_address") or "").strip()
            name = t.get("name") or t.get("symbol") or (addr[:6] + "…" + addr[-4:] if addr else "N/A")
            price = None if t.get("price") is None else float(t.get("price") or 0.0)
            liq = None if t.get("liquidity") is None else float(t.get("liquidity") or 0.0)
            mc = None if (t.get("mc") is None and t.get("market_cap") is None) else float(t.get("mc") or t.get("market_cap") or 0.0)
            vol24 = None if t.get("volume_24h") is None else float(t.get("volume_24h") or 0.0)
            p1 = None if t.get("price_change_1h") is None else float(t.get("price_change_1h"))
            p6 = None if t.get("price_change_6h") is None else float(t.get("price_change_6h"))
            p24 = None if t.get("price_change_24h") is None else float(t.get("price_change_24h"))
            rows.append({
                "Name": name,
                "Token Address": addr,
                "Dex": _dex_url_for(t),
                "Safety": _safety_str(t),
                "Price": price,
                "Liquidity": liq,
                "Market Cap": mc,
                "Volume (24h)": vol24,
                "1H": p1, "6H": p6, "24H": p24,
            })
        df = pd.DataFrame(rows, columns=["Name","Token Address","Dex","Safety","Price","Liquidity","Market Cap","Volume (24h)","1H","6H","24H"])
        money_cols = ["Price","Liquidity","Market Cap","Volume (24h)"]
        pct_cols = ["1H","6H","24H"]

        def _color_pct(v):
            try:
                if v is None:
                    return ""
                v = float(v)
                if v > 0:
                    return "color: green"
                if v < 0:
                    return "color: red"
            except Exception:
                pass
            return ""

        styler = (
            style_with_defaults(
                df,
                {**{c: _fmt_money_for_styler for c in money_cols}, **{c: _fmt_pct_for_styler for c in pct_cols}}
            )
            .map(_color_pct, subset=pct_cols)
            .hide(axis="index")
            .set_properties(subset=money_cols + pct_cols, **{"text-align": "right", "padding-right": "6px"})
            .set_table_styles(
                [{"selector": "td", "props": [("font-variant-numeric", "tabular-nums"), ("font-feature-settings", '"tnum" 1')]}],
                overwrite=False,
            )
        )

        st.subheader(label)
        st_dataframe_fmt(
            styler,
            width="stretch",
            height=_df_height(len(df)),
            hide_index=True,
            column_order=["Name","Token Address","Dex","Safety","Price","Liquidity","Market Cap","Volume (24h)","1H","6H","24H"],
            column_config={
                "Name": st.column_config.TextColumn("Name", width=220),
                "Token Address": st.column_config.TextColumn("Token Address", width=340),
                "Dex": st.column_config.LinkColumn("Dex", display_text="Link", width=20),
                "Safety": st.column_config.TextColumn("Safety", width=30),
                "Price": st.column_config.TextColumn("Price", width=30),
                "Liquidity": st.column_config.TextColumn("Liquidity", width=30),
                "Market Cap": st.column_config.TextColumn("Market Cap", width=30),
                "Volume (24h)": st.column_config.TextColumn("Volume (24h)", width=30),
                "1H": st.column_config.TextColumn("1H", width=20),
                "6H": st.column_config.TextColumn("6H", width=20),
                "24H": st.column_config.TextColumn("24H", width=20),
            },
        )

    # always redraw (no one-time-only guard)
    labels_and_data = [
        (f"🟢 {display_cap('High Cap')}", high),
        ("🟡 Mid Cap", mid),
        ("🔴 Low Cap", low),
        ("🔵 New", new),
    ]
    for label, df_bucket in labels_and_data:
        _render_bucket(label, df_bucket)

with tab_status:
    st.subheader("📊 Token Status")

    def _safe_row_int(row: Any, key: str = "c") -> int:
        """Extract an integer count from a DB row that may be a mapping, sqlite Row, or tuple."""
        try:
            if row is None:
                return 0
            if isinstance(row, dict):
                return int(row.get(key, 0) or 0)
            # sqlite3.Row behaves like a mapping but may also support indexing
            try:
                return int(row[key]) if key in getattr(row, "keys", lambda: {})() else int(row[0])
            except Exception:
                # fallback to index 0 for tuple-like rows
                return int(row[0])
        except Exception:
            return 0

    async def _status_snapshot():
        """
        Asynchronously gather top-level counts and recent trade rows.
        Returns (counts: dict, rows: list[dict]).
        """
        counts = {"holding": 0, "sold": 0, "watchlist": 0}
        rows = []
        try:
            async with connect_db() as db:
                # holding/open
                try:
                    cur1 = await db.execute("SELECT COUNT(*) AS c FROM tokens WHERE sell_time IS NULL OR is_trading = 1")
                    row1 = await cur1.fetchone()
                    counts["holding"] = _safe_row_int(row1)
                except Exception:
                    counts["holding"] = 0

                # sold/closed
                try:
                    cur2 = await db.execute("SELECT COUNT(*) AS c FROM tokens WHERE sell_time IS NOT NULL")
                    row2 = await cur2.fetchone()
                    counts["sold"] = _safe_row_int(row2)
                except Exception:
                    counts["sold"] = 0

                # watchlist (try eligible_tokens then shortlist_tokens)
                try:
                    curw = await db.execute("SELECT COUNT(1) AS c FROM eligible_tokens")
                    roww = await curw.fetchone()
                    counts["watchlist"] = _safe_row_int(roww)
                except Exception:
                    try:
                        curw2 = await db.execute("SELECT COUNT(1) AS c FROM shortlist_tokens")
                        roww2 = await curw2.fetchone()
                        counts["watchlist"] = _safe_row_int(roww2)
                    except Exception:
                        counts["watchlist"] = 0

                # determine columns for trade_history if present
                cols = set()
                try:
                    curti = await db.execute("PRAGMA table_info(trade_history)")
                    ti_rows = await curti.fetchall()
                    for r in ti_rows:
                        # r may be tuple like (cid,name,type,...)
                        if isinstance(r, dict):
                            cols.add(r.get("name"))
                        else:
                            # second column is name
                            if len(r) >= 2:
                                cols.add(r[1])
                except Exception:
                    cols = set()

                base_cols = ["symbol", "token_address", "buy_amount", "sell_amount", "buy_price", "sell_price", "profit", "buy_time", "sell_time"]
                if "action" in cols:
                    base_cols.insert(2, "action")
                select_list = ", ".join(base_cols)

                try:
                    cur = await db.execute(f"SELECT {select_list} FROM trade_history ORDER BY COALESCE(sell_time, buy_time) DESC LIMIT 50")
                    fetched = await cur.fetchall()
                    # Normalize rows into list[dict]
                    rows = []
                    for r in fetched:
                        if r is None:
                            continue
                        if isinstance(r, dict):
                            rows.append(r)
                        else:
                            try:
                                rows.append(dict(r))
                            except Exception:
                                # fallback: map by column order
                                rows.append({col: (r[i] if i < len(r) else None) for i, col in enumerate(base_cols)})
                except Exception:
                    rows = []
        except Exception as e:
            logger.error("Status snapshot failed: %s", e, exc_info=True)
        return counts, rows

    async def _watchlist_preview(limit: int = 25):
        """
        Return a list of lightweight watchlist entries (dicts).
        Tries eligible_tokens first, then shortlist_tokens as fallback.
        """
        preview = []
        try:
            async with connect_db() as db:
                try:
                    cur = await db.execute(
                        "SELECT address, data, created_at FROM eligible_tokens ORDER BY created_at DESC LIMIT ?",
                        (limit,),
                    )
                    rows = await cur.fetchall()
                except Exception:
                    # fallback to shortlist_tokens
                    try:
                        cur = await db.execute(
                            "SELECT address, data, created_at FROM shortlist_tokens ORDER BY created_at DESC LIMIT ?",
                            (limit,),
                        )
                        rows = await cur.fetchall()
                    except Exception:
                        return []

                for r in rows:
                    if r is None:
                        continue
                    # r may be mapping or sequence
                    if isinstance(r, dict):
                        addr = r.get("address")
                        data_json = r.get("data")
                        db_created_at = r.get("created_at")
                    else:
                        addr = r[0] if len(r) > 0 else None
                        data_json = r[1] if len(r) > 1 else None
                        db_created_at = r[2] if len(r) > 2 else None

                    try:
                        t = json.loads(data_json) if isinstance(data_json, str) else (data_json or {})
                    except Exception:
                        t = {}

                    created_at = db_created_at or t.get("pairCreatedAt") or t.get("createdAt") or t.get("created_at") or t.get("listedAt") or t.get("timestamp")

                    # symbol may be top-level or nested under baseToken
                    symbol = (t.get("symbol") or (isinstance(t.get("baseToken"), dict) and t["baseToken"].get("symbol")) or "—")
                    preview.append({
                        "symbol": symbol,
                        "address": addr,
                        "bucket": t.get("_bucket") or "—",
                        "market_cap": t.get("market_cap") or t.get("mc"),
                        "liquidity": t.get("liquidity") or t.get("liquidity_usd"),
                        "volume_24h": t.get("volume_24h") or t.get("v24hUSD") or t.get("volume"),
                        "created_at": created_at,
                    })
                return preview
        except Exception:
            return []

    # Run the async snapshot and watchlist tasks using run_async_task with coroutines
    snap = run_async_task(_status_snapshot(), timeout=90)
    counts, recent = snap if isinstance(snap, tuple) and len(snap) == 2 else ({'holding': 0, 'sold': 0, 'watchlist': 0}, [])

    c1, c2, c3 = st.columns(3)
    c1.metric("Holding / Open", counts.get("holding", 0))
    c2.metric("Sold / Closed", counts.get("sold", 0))
    c3.metric("On Watchlist (this cycle)", counts.get("watchlist", 0))

    st.divider()
    st.markdown("**Recent Trades**")
    if recent:
        # Ensure recent is a DataFrame or Styler acceptable to st_dataframe_fmt
        try:
            df_recent = recent if isinstance(recent, pd.DataFrame) else pd.DataFrame(recent)
            st_dataframe_fmt(df_recent, width="stretch", hide_index=True)
        except Exception:
            # last-resort: show raw list
            st.write(recent)
    else:
        st.info("No trade history yet.")

    st.divider()
    st.markdown("**Active Watchlist (current snapshot)**")
    wl = run_async_task(_watchlist_preview(), timeout=60)
    if wl:
        formatted_rows = []
        for r in wl:
            formatted_rows.append({
                "symbol": r.get("symbol", "—"),
                "address": r.get("address", "—"),
                "bucket": r.get("bucket", "—"),
                "Market Cap (USD)": fmt_usd(r.get("market_cap")),
                "Liquidity (USD)": fmt_usd(r.get("liquidity")),
                "Volume 24H (USD)": fmt_usd(r.get("volume_24h")),
                "Created At (local)": fmt_ts(r.get("created_at")),
            })
        try:
            df_wl = pd.DataFrame(formatted_rows)
            st_dataframe_fmt(df_wl, width="stretch", hide_index=True)
        except Exception:
            st.write(formatted_rows)
    else:
        st.info("No shortlist found yet. Run discovery from Bot Control to populate this.")
        
# ---------------------------
# P&L tab 
# ---------------------------
with tab_pl:
    st.header("💹 P&L Dashboard")

    st.markdown(
        "Comprehensive profit & loss analysis with realized/unrealized P&L, performance metrics, and risk insights."
    )

    # Import pnl_engine
    try:
        from solana_trading_bot_bundle.trading_bot.pnl_engine import (
            calculate_comprehensive_metrics,
            format_duration,
        )
    except ImportError:
        st.error("❌ Failed to import pnl_engine. Please ensure the module is available.")
        calculate_comprehensive_metrics = None
        format_duration = None

    # Try to use existing snapshot helper if it exists; otherwise fall back to a local DB query.
    try:
        if "_status_snapshot" in globals() and callable(globals().get("_status_snapshot")):
            snap = run_async_task(globals()["_status_snapshot"](), timeout=60)
        else:
            async def _pl_local_snapshot():
                """
                Enhanced local fallback for P&L data retrieval.
                Returns (counts, closed_trades, open_positions) tuple.
                """
                counts = {"holding": 0, "sold": 0, "watchlist": 0}
                closed_trades = []
                open_positions = []
                try:
                    # Ensure GUI helper schemas exist (idempotent)
                    try:
                        await ensure_eligible_tokens_schema()
                    except Exception:
                        pass
                    try:
                        await ensure_price_cache_schema()
                    except Exception:
                        pass
                    try:
                        await ensure_shortlist_tokens_schema()
                    except Exception:
                        pass

                    async with connect_db() as db:
                        # holding / open
                        try:
                            cur = await db.execute("SELECT COUNT(1) FROM tokens WHERE sell_time IS NULL OR is_trading = 1")
                            r = await cur.fetchone()
                            counts["holding"] = int((r[0] if isinstance(r, (list, tuple)) else r.get("c")) or 0)
                        except Exception:
                            counts["holding"] = 0

                        # sold / closed
                        try:
                            cur = await db.execute("SELECT COUNT(1) FROM tokens WHERE sell_time IS NOT NULL")
                            r = await cur.fetchone()
                            counts["sold"] = int((r[0] if isinstance(r, (list, tuple)) else r.get("c")) or 0)
                        except Exception:
                            counts["sold"] = 0

                        # watchlist / eligible_tokens
                        try:
                            cur = await db.execute("SELECT COUNT(1) FROM eligible_tokens")
                            r = await cur.fetchone()
                            counts["watchlist"] = int((r[0] if isinstance(r, (list, tuple)) else r.get("c")) or 0)
                        except Exception:
                            counts["watchlist"] = 0

                        # Closed trades from trade_history
                        try:
                            q = (
                                "SELECT symbol, token_address, buy_amount, sell_amount, "
                                "buy_price, sell_price, profit, buy_time, sell_time "
                                "FROM trade_history "
                                "WHERE buy_price IS NOT NULL AND sell_price IS NOT NULL "
                                "ORDER BY COALESCE(sell_time, buy_time) DESC LIMIT 100"
                            )
                            cur = await db.execute(q)
                            fetched = await cur.fetchall()
                            for r in fetched:
                                if isinstance(r, dict):
                                    closed_trades.append(r)
                                else:
                                    try:
                                        closed_trades.append(dict(r))
                                    except Exception:
                                        cols = ["symbol","token_address","buy_amount","sell_amount","buy_price","sell_price","profit","buy_time","sell_time"]
                                        closed_trades.append({cols[i]: (r[i] if i < len(r) else None) for i in range(len(cols))})
                        except Exception:
                            closed_trades = []

                        # Open positions from tokens table
                        try:
                            q = (
                                "SELECT symbol, address as token_address, buy_amount, "
                                "buy_price, price, buy_time "
                                "FROM tokens "
                                "WHERE sell_time IS NULL AND buy_price IS NOT NULL "
                                "ORDER BY buy_time DESC"
                            )
                            cur = await db.execute(q)
                            fetched = await cur.fetchall()
                            for r in fetched:
                                if isinstance(r, dict):
                                    open_positions.append(r)
                                else:
                                    try:
                                        open_positions.append(dict(r))
                                    except Exception:
                                        cols = ["symbol","token_address","buy_amount","buy_price","price","buy_time"]
                                        open_positions.append({cols[i]: (r[i] if i < len(r) else None) for i in range(len(cols))})
                        except Exception:
                            open_positions = []
                except Exception:
                    pass
                return counts, closed_trades, open_positions

            snap = run_async_task(_pl_local_snapshot(), timeout=60)
    except Exception as e:
        logger.exception("Failed to load P&L snapshot: %s", e)
        st.error(f"Failed to load P&L data: {e}")
        snap = ({'holding': 0, 'sold': 0, 'watchlist': 0}, [], [])

    counts, closed_trades, open_positions = snap if isinstance(snap, tuple) and len(snap) == 3 else ({'holding': 0, 'sold': 0, 'watchlist': 0}, [], [])

    # Calculate comprehensive metrics using pnl_engine
    metrics = None
    if calculate_comprehensive_metrics:
        try:
            metrics = calculate_comprehensive_metrics(closed_trades, open_positions, wallet_balance=None)
        except Exception as e:
            logger.exception("Failed to calculate P&L metrics: %s", e)
            st.warning(f"⚠️ Metrics calculation failed: {e}")

    # ==================== TOP SUMMARY BAR ====================
    st.subheader("📊 Performance Summary")
    
    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Realized P&L",
                fmt_usd(metrics.realized_pnl, compact=False),
                delta=None,
                help="Total profit/loss from closed trades"
            )
        
        with col2:
            st.metric(
                "Unrealized P&L",
                fmt_usd(metrics.unrealized_pnl, compact=False),
                delta=None,
                help="Current profit/loss from open positions"
            )
        
        with col3:
            st.metric(
                "Total P&L",
                fmt_usd(metrics.total_pnl, compact=False),
                delta=None,
                help="Combined realized + unrealized P&L"
            )
        
        with col4:
            win_rate_color = "🟢" if metrics.win_rate >= 50 else "🔴"
            st.metric(
                "Win Rate",
                f"{win_rate_color} {metrics.win_rate:.1f}%",
                delta=None,
                help="Percentage of profitable closed trades"
            )
        
        with col5:
            st.metric(
                "Max Drawdown",
                f"{metrics.max_drawdown_pct:.1f}%",
                delta=None,
                help="Maximum peak-to-trough decline in P&L"
            )
    else:
        st.info("📊 Calculating metrics...")

    st.markdown("---")

    # ==================== EXPANDED PERFORMANCE STATS PANEL ====================
    with st.expander("📈 Detailed Performance Statistics", expanded=True):
        if metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Trade Statistics**")
                st.write(f"Total Trades: **{metrics.total_trades}**")
                st.write(f"Winning Trades: 🟢 **{metrics.winning_trades}**")
                st.write(f"Losing Trades: 🔴 **{metrics.losing_trades}**")
                st.write(f"Break-even Trades: ⚪ **{metrics.breakeven_trades}**")
            
            with col2:
                st.markdown("**Profitability Metrics**")
                st.write(f"Average Win: **{fmt_usd(metrics.avg_win, compact=False)}**")
                st.write(f"Average Loss: **{fmt_usd(metrics.avg_loss, compact=False)}**")
                
                # Format profit factor
                if metrics.profit_factor == float('inf'):
                    pf_str = "∞ (no losses)"
                else:
                    pf_str = f"{metrics.profit_factor:.2f}"
                st.write(f"Profit Factor: **{pf_str}**")
                
                st.write(f"Expectancy/Trade: **{fmt_usd(metrics.expectancy, compact=False)}**")
            
            with col3:
                st.markdown("**Time & Fees**")
                if metrics.avg_hold_time_seconds > 0 and format_duration:
                    avg_hold_str = format_duration(metrics.avg_hold_time_seconds)
                    st.write(f"Avg Hold Time: **{avg_hold_str}**")
                else:
                    st.write(f"Avg Hold Time: **N/A**")
                
                st.write(f"Total Fees (USD): **{fmt_usd(metrics.total_fees_usd, compact=False)}**")
                st.write(f"Total Fees (SOL): **{metrics.total_fees_sol:.4f}**")
                
                # Add tooltip about fees
                st.caption("💡 Fee tracking may be limited based on available data")
        else:
            st.info("No performance data available yet")

    st.markdown("---")

    # ==================== RISK METRICS PANEL ====================
    with st.expander("🛡️ Risk Metrics", expanded=False):
        if metrics and open_positions:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Exposure & Positions**")
                if metrics.current_exposure_pct > 0:
                    st.write(f"Current Exposure: **{metrics.current_exposure_pct:.1f}%** of wallet")
                else:
                    st.write(f"Current Exposure: **N/A** (wallet balance needed)")
                
                st.write(f"Open Positions: **{len(open_positions)}**")
                st.write(f"Positions at Loss: 🔴 **{metrics.open_positions_at_loss}**")
            
            with col2:
                st.markdown("**Worst Performers**")
                st.write(f"Largest Open Drawdown: **{fmt_usd(metrics.largest_open_drawdown, compact=False)}**")
                
                if metrics.worst_performing_token:
                    st.write(f"Worst Token: **{metrics.worst_performing_token}**")
                    st.write(f"Loss: **{fmt_usd(metrics.worst_performing_pnl, compact=False)}**")
                else:
                    st.write("No positions currently at loss")
        else:
            st.info("No open positions to analyze")

    st.markdown("---")

    # ==================== POSITION COUNTS ====================
    c1, c2, c3 = st.columns(3)
    c1.metric("Holding / Open", counts.get("holding", 0))
    c2.metric("Sold / Closed", counts.get("sold", 0))
    c3.metric("On Watchlist", counts.get("watchlist", 0))

    st.markdown("---")

    # ==================== ENHANCED TRADE TABLE ====================
    if not closed_trades:
        st.info(
            "No closed trades found in the database. Trades will appear here once you complete your first sale."
        )
    else:
        try:
            import pandas as _pd
            df_trades = _pd.DataFrame(closed_trades)

            # Filter out rows with null buy_price or sell_price
            df_trades = df_trades[
                (df_trades["buy_price"].notna()) & 
                (df_trades["sell_price"].notna())
            ]

            if len(df_trades) == 0:
                st.info("No complete trades with both buy and sell prices")
            else:
                # Ensure columns exist
                for col in ("symbol", "token_address", "buy_amount", "sell_amount", "buy_price", "sell_price", "profit", "buy_time", "sell_time"):
                    if col not in df_trades.columns:
                        df_trades[col] = None

                # Calculate additional columns
                # P&L percentage
                def calc_pnl_pct(row):
                    try:
                        if row["buy_price"] and row["sell_price"]:
                            buy_p = float(row["buy_price"])
                            sell_p = float(row["sell_price"])
                            if buy_p > 0:
                                return ((sell_p - buy_p) / buy_p) * 100.0
                    except Exception:
                        pass
                    return None

                df_trades["pnl_pct"] = df_trades.apply(calc_pnl_pct, axis=1)

                # Trade side (simplified - could be enhanced with actual buy/sell tracking)
                df_trades["side"] = "BUY→SELL"

                # Duration held
                def calc_duration(row):
                    try:
                        if row["buy_time"] and row["sell_time"]:
                            duration_sec = float(row["sell_time"]) - float(row["buy_time"])
                            if duration_sec > 0 and format_duration:
                                return format_duration(duration_sec)
                    except Exception:
                        pass
                    return "N/A"

                df_trades["duration"] = df_trades.apply(calc_duration, axis=1)

                # Format timestamps to human readable strings
                def _fmt_ts_col(v):
                    try:
                        return fmt_ts(v, with_time=True)
                    except Exception:
                        return ""

                if "buy_time" in df_trades.columns:
                    df_trades["buy_time"] = df_trades["buy_time"].apply(_fmt_ts_col)
                if "sell_time" in df_trades.columns:
                    df_trades["sell_time"] = df_trades["sell_time"].apply(_fmt_ts_col)

                # Prepare display dataframe with selected columns
                display_cols = [
                    "symbol", "side", "buy_price", "sell_price", 
                    "profit", "pnl_pct", "buy_amount", "duration", 
                    "buy_time", "sell_time"
                ]
                
                # Only include columns that exist
                display_cols = [c for c in display_cols if c in df_trades.columns]
                df_display = df_trades[display_cols].copy()

                # Rename columns for better display
                column_renames = {
                    "symbol": "Symbol",
                    "side": "Side",
                    "buy_price": "Entry Price",
                    "sell_price": "Exit Price",
                    "profit": "P&L (USD)",
                    "pnl_pct": "P&L %",
                    "buy_amount": "Amount",
                    "duration": "Duration",
                    "buy_time": "Entry Time",
                    "sell_time": "Exit Time",
                }
                df_display = df_display.rename(columns=column_renames)

                st.subheader(f"📋 Recent Trades (latest {len(df_display)})")

                # Apply styling
                def style_pnl_row(row):
                    """Apply green/red color to profitable/losing trades"""
                    styles = [''] * len(row)
                    
                    # Check P&L column
                    pnl_col = "P&L (USD)"
                    if pnl_col in row.index:
                        try:
                            pnl_val = float(row[pnl_col]) if row[pnl_col] is not None else 0
                            if pnl_val > 0.01:
                                # Green for profit, bold
                                styles = ['color: #00cc66; font-weight: bold'] * len(row)
                            elif pnl_val < -0.01:
                                # Red for loss
                                styles = ['color: #ff4444'] * len(row)
                            else:
                                # Gray for break-even
                                styles = ['color: #888888; opacity: 0.7'] * len(row)
                        except Exception:
                            pass
                    
                    return styles

                # Format numeric columns
                styler = df_display.style.apply(style_pnl_row, axis=1)
                
                # Format specific columns
                if "Entry Price" in df_display.columns:
                    styler = styler.format({"Entry Price": _fmt_money_for_styler})
                if "Exit Price" in df_display.columns:
                    styler = styler.format({"Exit Price": _fmt_money_for_styler})
                if "P&L (USD)" in df_display.columns:
                    styler = styler.format({"P&L (USD)": _fmt_money_for_styler})
                if "P&L %" in df_display.columns:
                    styler = styler.format({"P&L %": _fmt_pct_for_styler})

                st_dataframe_fmt(
                    styler,
                    height_rows=min(len(df_display), 15),
                    use_styler=True,
                )

                # Aggregate total profit display
                try:
                    total_profit = float(df_trades["profit"].dropna().astype(float).sum())
                    st.markdown(f"**Total Realized Profit:** {fmt_usd(total_profit, compact=False)}")
                except Exception:
                    pass

        except Exception as e:
            logger.exception("Failed to render P&L table: %s", e)
            st.error(f"Error rendering trade table: {e}")
            st.write(closed_trades[:10])  # Show first 10 as fallback


