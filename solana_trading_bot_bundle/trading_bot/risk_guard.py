# risk_guard.py
"""
SOLO Meme Coin Bot — Risk Guard
- Reads cfg["risk_management"] (all keys optional)
- Integrates with metrics_engine.MetricsStore to enforce daily drawdown and pause windows
- Provides simple pre-trade checks: max position size, max open positions, min equity
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import os, time, json

from . import metrics_engine as MET

DEFAULTS = {
    "max_daily_drawdown_usd": None,   # e.g., 200.0
    "max_position_size_sol": None,    # e.g., 0.2
    "max_open_positions": None,       # e.g., 4
    "min_equity_usd": None,           # e.g., 500.0
    "pause_minutes_on_breach": 60,    # cooldown after breach
}

def _app_state_path() -> str:
    base = os.getenv("SOLO_APPDATA_DIR")
    if not base:
        base = os.path.join(os.path.expanduser("~"), "AppData", "Local", "SOLOTradingBot")
        if not os.path.isdir(base):
            base = os.path.join(os.path.expanduser("~"), ".solotradingbot")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "risk_state.json")

def _load_state() -> Dict[str, Any]:
    p = _app_state_path()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_state(state: Dict[str, Any]) -> None:
    p = _app_state_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass

class RiskGuard:
    def __init__(self, cfg: Dict[str, Any], metrics: MET.MetricsStore, logger=None):
        self.cfg = cfg or {}
        self.metrics = metrics
        self.logger = logger

    def _rg(self) -> Dict[str, Any]:
        rg = (self.cfg.get("risk_management") or {}).copy()
        out = DEFAULTS.copy()
        out.update({k: rg.get(k, v) for k, v in DEFAULTS.items()})
        return out

    def should_pause(self) -> Tuple[bool, str]:
        """Returns (paused, reason). True means block trading."""
        st = _load_state()
        now = int(time.time())
        paused_until = int(st.get("paused_until_ts", 0))
        if paused_until and now < paused_until:
            return True, f"paused_until_ts={paused_until}"

        rg = self._rg()
        max_dd = rg.get("max_daily_drawdown_usd")
        if max_dd is not None:
            pnl_today = float(self.metrics.realized_pnl_for_day())
            if pnl_today <= -abs(float(max_dd)):
                pause_min = int(rg.get("pause_minutes_on_breach") or 60)
                st["paused_until_ts"] = now + pause_min * 60
                _save_state(st)
                return True, f"daily_drawdown_breached({pnl_today} <= -{max_dd})"
        return False, ""

    def pretrade_checks(self, *, open_positions: int, planned_size_sol: float, equity_usd: Optional[float]) -> Tuple[bool, str]:
        """
        Lightweight pre-trade risk checks. Returns (ok, reason_if_blocked).
        """
        rg = self._rg()
        if rg.get("max_open_positions") is not None and open_positions >= int(rg["max_open_positions"]):
            return False, f"max_open_positions_reached({open_positions})"
        if rg.get("max_position_size_sol") is not None and planned_size_sol > float(rg["max_position_size_sol"]):
            return False, f"position_size_exceeds({planned_size_sol} > {rg['max_position_size_sol']})"
        if rg.get("min_equity_usd") is not None and (equity_usd is None or equity_usd < float(rg["min_equity_usd"])):
            return False, f"min_equity_breach({equity_usd} < {rg['min_equity_usd']})"
        paused, why = self.should_pause()
        if paused:
            return False, why
        return True, "ok"
