"""
SOLO Meme Coin Bot — Metrics Engine (Spot, long/flat friendly)

Changes:
- Optional fee proration across matched lots (prorate_fees flag).
  When prorate_fees=True, a single fill matching multiple opposite-side lots
  will have its fee allocated proportionally to each matched piece and
  produce per-piece TradeResult entries. When prorate_fees=False (default),
  the whole fee is subtracted once and a single aggregated TradeResult is emitted.
- Supports shorting, dry-run handling, JSONL persistence and replay.
- Backwards-compatible aliases and make_store factory (reads SOLO_METRICS_PRORATE_FEES).
- Robust handling of dry-run vs real attribution when prorating fees:
  - Persisted aggregated realize event now includes realized_net_real computed
    as the sum of piece-level nets that are "real" (neither fill nor lot dry).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time
import math
import threading
import datetime
import errno

# ---------- Paths & IO ----------
def _app_dir() -> str:
    env = os.getenv("SOLO_APPDATA_DIR")
    if env:
        base = env
    else:
        home = os.path.expanduser("~") or "."
        winpath = os.path.join(home, "AppData", "Local", "SOLOTradingBot")
        if os.path.isdir(winpath):
            base = winpath
        else:
            base = os.path.join(home, ".solotradingbot")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError as e:
        # fallback to cwd if permission issues occur
        if e.errno in (errno.EACCES, errno.ENOENT):
            base = os.path.join(os.getcwd(), ".solotradingbot")
            try:
                os.makedirs(base, exist_ok=True)
            except Exception:
                pass
    return base

def _metrics_path() -> str:
    return os.path.join(_app_dir(), "metrics.jsonl")

def _safe_write_jsonl(path: str, obj: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")
    except Exception:
        # metrics writes must not crash the trading loop
        pass

def _utc_now() -> int:
    return int(time.time())

def _day_bucket(ts: int) -> str:
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

# ---------- Data ----------
@dataclass
class Fill:
    ts: int
    token: str
    side: str       # "BUY" or "SELL"
    qty: float      # base token amount (positive)
    price_usd: float
    fee_usd: float
    dry_run: bool

@dataclass
class TradeResult:
    token: str
    open_ts: int
    close_ts: int
    hold_seconds: int
    qty: float
    avg_entry: float
    avg_exit: float
    gross_pnl_usd: float
    net_pnl_usd: float
    ret_pct: float           # net PnL / entry_cost * 100
    dry_run: bool

# ---------- Engine ----------
class MetricsStore:
    """
    Maintains per-token FIFO lots and realizes PnL when opposite-side fills close inventory.
    Supports both long and short.

    Parameters:
    - prorate_fees: bool
        If True, when a single fill matches multiple opposite-side lots, the fee for the fill
        is pro-rated across matched lots and a TradeResult is created per matched lot (with its
        fee share applied). If False (default), the whole fee is subtracted once from the aggregate
        realized PnL and a single TradeResult is created for the combined matched quantity.
    """
    def __init__(self, logger: Optional[Any] = None, metrics_path: Optional[str] = None,
                 replay_on_init: bool = True, prorate_fees: bool = False):
        self.logger = logger
        self.path = metrics_path or _metrics_path()
        self.lock = threading.Lock()
        self.prorate_fees = bool(prorate_fees)

        # per token: list of lots: (qty_remaining, price_usd, ts_open, dry_run, side)
        # qty_remaining always positive; side is "BUY" (long lot) or "SELL" (short lot)
        self._lots: Dict[str, List[Tuple[float, float, int, bool, str]]] = {}

        # equity curve tracks ONLY real (non-dry) realized cumulative PnL
        self._equity_curve: List[Tuple[int, float]] = []
        self._cum_net_pnl_real: float = 0.0
        self._cum_net_pnl_all: float = 0.0

        self._last_activity_ts: Optional[int] = None

        self._wins: List[float] = []
        self._losses: List[float] = []
        self._trades: List[TradeResult] = []

        if replay_on_init:
            try:
                self._replay_events_from_file()
            except Exception:
                # replay failure should not stop the bot
                pass

    # ---- Back-compat method aliases ----
    def record_fill(self, **kw) -> None:
        """Back-compat alias for on_fill(...)."""
        return self.on_fill(**kw)

    def realized_pnl_today(self, include_dry_runs: bool = False) -> float:
        """Back-compat alias: realized PnL for the current UTC day."""
        return self.realized_pnl_for_day(None, include_dry_runs=include_dry_runs)

    def get_trades(self, include_dry_runs: bool = True) -> List[Dict[str, Any]]:
        """Back-compat alias for trades(...)."""
        return self.trades(include_dry_runs=include_dry_runs)

    def get_equity_curve(self) -> List[Tuple[int, float]]:
        """Back-compat alias for equity_curve(...)."""
        return self.equity_curve()

    def get_summary(self, include_dry_runs: bool = False) -> Dict[str, Any]:
        """Back-compat alias for summary(...)."""
        return self.summary(include_dry_runs=include_dry_runs)

    # ---- public API ----
    def on_fill(self, *, token_addr: str, side: str, qty: float, price_usd: float,
                fee_usd: float = 0.0, dry_run: bool = False, ts: Optional[int] = None) -> None:
        """
        Record a fill.
        qty must be positive. side must be 'BUY' or 'SELL'.
        """
        ts = int(ts or _utc_now())
        side = (side or "").upper().strip()
        if side not in ("BUY", "SELL"):
            return
        try:
            qty = float(qty)
            price_usd = float(price_usd)
            fee_usd = float(fee_usd or 0.0)
        except Exception:
            return
        if qty <= 0 or price_usd <= 0:
            return

        with self.lock:
            self._last_activity_ts = ts
            _safe_write_jsonl(self.path, {
                "type": "fill",
                "ts": ts,
                "token": token_addr,
                "side": side,
                "qty": qty,
                "price_usd": price_usd,
                "fee_usd": fee_usd,
                "dry_run": bool(dry_run),
                "day": _day_bucket(ts),
            })
            self._apply_fill(token_addr=token_addr, side=side, qty=qty, price_usd=price_usd, fee_usd=fee_usd, dry_run=bool(dry_run), ts=ts)

    def snapshot_equity_point(self) -> None:
        ts = _utc_now()
        with self.lock:
            self._equity_curve.append((ts, self._cum_net_pnl_real))
            _safe_write_jsonl(self.path, {
                "type": "equity",
                "ts": ts,
                "equity_real": self._cum_net_pnl_real,
                "equity_all": self._cum_net_pnl_all,
                "day": _day_bucket(ts)
            })

    def summary(self, include_dry_runs: bool = False) -> Dict[str, Any]:
        with self.lock:
            trades = [t for t in self._trades if include_dry_runs or not t.dry_run]
            curve = list(self._equity_curve)
            wins = [t.net_pnl_usd for t in trades if t.net_pnl_usd is not None and t.net_pnl_usd > 0]
            losses = [t.net_pnl_usd for t in trades if t.net_pnl_usd is not None and t.net_pnl_usd < 0]
            n = len(trades)
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses)) if losses else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
            win_rate = (len(wins) / n * 100.0) if n > 0 else 0.0
            avg_win = (sum(wins) / len(wins)) if wins else 0.0
            avg_loss = (sum(losses) / len(losses)) if losses else 0.0

            rets = [t.ret_pct / 100.0 for t in trades if t.ret_pct is not None]
            sharpe = 0.0
            if len(rets) >= 2:
                mu = sum(rets) / len(rets)
                var = sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)
                sd = math.sqrt(var) if var > 0 else 0.0
                if sd > 0.0:
                    sharpe = (mu / sd) * math.sqrt(len(rets))

            max_dd = 0.0
            if len(curve) >= 1:
                peak = curve[0][1]
                for _, eq in curve:
                    if eq > peak: peak = eq
                    dd = peak - eq
                    if dd > max_dd: max_dd = dd

            avg_hold = (sum(t.hold_seconds for t in trades) / n) if n else 0.0

            return {
                "trades": n,
                "win_rate_pct": win_rate,
                "profit_factor": profit_factor,
                "avg_win_usd": avg_win,
                "avg_loss_usd": avg_loss,
                "max_drawdown_usd": max_dd,
                "sharpe_like": sharpe,
                "avg_hold_seconds": avg_hold,
                "realized_pnl_usd_real": self._cum_net_pnl_real,
                "realized_pnl_usd_all": self._cum_net_pnl_all,
                "equity_points": len(curve),
                "prorate_fees": self.prorate_fees,
            }

    def equity_curve(self) -> List[Tuple[int, float]]:
        with self.lock:
            return self._equity_curve[:]

    def trades(self, include_dry_runs: bool = True) -> List[Dict[str, Any]]:
        with self.lock:
            if include_dry_runs:
                return [asdict(t) for t in self._trades]
            else:
                return [asdict(t) for t in self._trades if not t.dry_run]

    def realized_pnl_for_day(self, day_str_yyyy_mm_dd: Optional[str] = None, include_dry_runs: bool = False) -> float:
        day = day_str_yyyy_mm_dd or datetime.datetime.utcfromtimestamp(_utc_now()).strftime("%Y-%m-%d")
        total = 0.0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("type") == "realize" and obj.get("day") == day:
                        if obj.get("dry_run") and not include_dry_runs:
                            continue
                        val = obj.get("realized_net_real")
                        if val is None:
                            val = obj.get("realized_net", 0.0)
                        try:
                            total += float(val)
                        except Exception:
                            continue
        except FileNotFoundError:
            pass
        return total

    # ---- internals ----
    def _apply_fill(self, token_addr: str, side: str, qty: float, price_usd: float,
                    fee_usd: float, dry_run: bool, ts: int) -> None:
        """
        Core FIFO matching:
        - If there are opposite-side lots, consume them FIFO and realize PnL per matched piece.
        - If remaining qty remains, append as a new lot with side.
        - Fee handling:
            * If prorate_fees is False: subtract the full fee from the aggregate realized amount,
              create a single aggregated TradeResult for all matched qty.
            * If prorate_fees is True: distribute the fee proportionally to each matched piece and
              create per-piece TradeResult entries.
        """
        lots = self._lots.setdefault(token_addr, [])
        want = float(qty)

        matched_pieces: List[Dict[str, Any]] = []  # each: {qty, lot_price, lot_ts, lot_dry, lot_side, piece_gross}
        i = 0
        while want > 0 and i < len(lots):
            lot_qty, lot_price, lot_ts, lot_dry, lot_side = lots[i]
            if lot_side == side:
                i += 1
                continue
            take = lot_qty if lot_qty <= want else want

            if lot_side == "BUY" and side == "SELL":
                # closing a long
                piece_gross = (price_usd - lot_price) * take
            elif lot_side == "SELL" and side == "BUY":
                # closing a short
                piece_gross = (lot_price - price_usd) * take
            else:
                piece_gross = 0.0

            matched_pieces.append({
                "qty": take,
                "lot_price": lot_price,
                "lot_ts": lot_ts,
                "lot_dry": lot_dry,
                "lot_side": lot_side,
                "piece_gross": piece_gross,
            })

            remaining = lot_qty - take
            if remaining <= 0:
                lots.pop(i)
            else:
                lots[i] = (remaining, lot_price, lot_ts, lot_dry, lot_side)
                i += 1
            want -= take

        # If leftover becomes opening lot of this side
        if want > 0:
            lots.append((want, price_usd, ts, dry_run, side))
            want = 0.0

        qty_closed_total = sum(p["qty"] for p in matched_pieces)
        realized_gross_total = sum(p["piece_gross"] for p in matched_pieces)
        total_entry_cost = sum(p["lot_price"] * p["qty"] for p in matched_pieces)
        open_ts_first = matched_pieces[0]["lot_ts"] if matched_pieces else None

        # No matched pieces -> create a realize event with zero closed qty for observability
        if qty_closed_total == 0:
            _safe_write_jsonl(self.path, {
                "type": "realize",
                "ts": ts,
                "token": token_addr,
                "qty_closed": 0.0,
                "avg_exit": price_usd,
                "realized_gross": 0.0,
                "realized_net": 0.0,
                "realized_net_real": 0.0,
                "cum_net_pnl_real": self._cum_net_pnl_real,
                "cum_net_pnl_all": self._cum_net_pnl_all,
                "dry_run": dry_run,
                "day": _day_bucket(ts),
                "note": "sell_or_buy_without_inventory_opened"
            })
            return

        # Fee handling and trade emission
        if self.prorate_fees and qty_closed_total > 0:
            # Allocate fee per piece and create per-piece trades
            piece_nets: List[float] = []
            piece_nets_real_sum = 0.0  # sum of nets that are considered "real" (neither fill nor lot dry)
            for p in matched_pieces:
                take_qty = p["qty"]
                piece_gross = p["piece_gross"]
                fee_share = (abs(fee_usd) * (take_qty / qty_closed_total)) if qty_closed_total > 0 else 0.0
                piece_net = piece_gross - fee_share
                piece_nets.append(piece_net)

                # Determine if piece is real: both fill and lot are non-dry
                piece_is_real = (not dry_run) and (not bool(p.get("lot_dry", False)))
                if piece_is_real:
                    piece_nets_real_sum += piece_net

                # Update cumulatives per piece
                if dry_run or p.get("lot_dry", False):
                    # If either the fill or the lot was dry-run, treat as dry-run trade
                    self._cum_net_pnl_all += piece_net
                else:
                    self._cum_net_pnl_real += piece_net
                    self._cum_net_pnl_all += piece_net
                    self._equity_curve.append((ts, self._cum_net_pnl_real))

                # Build TradeResult for this piece
                if p["lot_price"] * take_qty > 0:
                    avg_entry = p["lot_price"]
                    ret_pct = (piece_net / (avg_entry * take_qty)) * 100.0 if (avg_entry * take_qty) != 0 else 0.0
                else:
                    avg_entry = 0.0
                    ret_pct = 0.0

                tr = TradeResult(
                    token=token_addr,
                    open_ts=int(p["lot_ts"] or ts),
                    close_ts=ts,
                    hold_seconds=int(ts - int(p["lot_ts"] or ts)),
                    qty=take_qty,
                    avg_entry=avg_entry,
                    avg_exit=price_usd,
                    gross_pnl_usd=piece_gross,
                    net_pnl_usd=piece_net,
                    ret_pct=ret_pct,
                    dry_run=bool(dry_run or p.get("lot_dry", False)),
                )
                self._trades.append(tr)
                if tr.net_pnl_usd >= 0:
                    self._wins.append(tr.net_pnl_usd)
                else:
                    self._losses.append(tr.net_pnl_usd)

            # Persist aggregated realize event for observability (include totals and real-subset)
            aggregated_realized_net_all = sum(piece_nets)
            # realized_net_real is the sum of piece nets that are considered real
            aggregated_realized_net_real = piece_nets_real_sum
            _safe_write_jsonl(self.path, {
                "type": "realize",
                "ts": ts,
                "token": token_addr,
                "qty_closed": qty_closed_total,
                "avg_exit": price_usd,
                "realized_gross": realized_gross_total,
                "realized_net": aggregated_realized_net_all,
                "realized_net_real": aggregated_realized_net_real,
                "cum_net_pnl_real": self._cum_net_pnl_real,
                "cum_net_pnl_all": self._cum_net_pnl_all,
                "dry_run": dry_run,
                "day": _day_bucket(ts),
                "prorated_fee": True,
            })
        else:
            # Non-prorated: subtract whole fee once and create a single aggregated TradeResult
            realized_net_total = realized_gross_total - abs(fee_usd)

            # Update cumulatives once
            if dry_run:
                self._cum_net_pnl_all += realized_net_total
            else:
                self._cum_net_pnl_real += realized_net_total
                self._cum_net_pnl_all += realized_net_total
                self._equity_curve.append((ts, self._cum_net_pnl_real))

            if total_entry_cost > 0:
                avg_entry = total_entry_cost / qty_closed_total
                ret_pct = (realized_net_total / total_entry_cost) * 100.0 if total_entry_cost != 0 else 0.0
            else:
                avg_entry = 0.0
                ret_pct = 0.0

            tr = TradeResult(
                token=token_addr,
                open_ts=int(open_ts_first or ts),
                close_ts=ts,
                hold_seconds=int(ts - int(open_ts_first or ts)) if open_ts_first else 0,
                qty=qty_closed_total,
                avg_entry=avg_entry,
                avg_exit=price_usd,
                gross_pnl_usd=realized_gross_total,
                net_pnl_usd=realized_net_total,
                ret_pct=ret_pct,
                dry_run=bool(dry_run),
            )
            self._trades.append(tr)
            if tr.net_pnl_usd >= 0:
                self._wins.append(tr.net_pnl_usd)
            else:
                self._losses.append(tr.net_pnl_usd)

            _safe_write_jsonl(self.path, {
                "type": "realize",
                "ts": ts,
                "token": token_addr,
                "qty_closed": qty_closed_total,
                "avg_exit": price_usd,
                "realized_gross": realized_gross_total,
                "realized_net": realized_net_total,
                # In non-prorated mode we mark realized_net_real using the fill-level dry flag:
                "realized_net_real": realized_net_total if not dry_run else 0.0,
                "cum_net_pnl_real": self._cum_net_pnl_real,
                "cum_net_pnl_all": self._cum_net_pnl_all,
                "dry_run": dry_run,
                "day": _day_bucket(ts),
                "prorated_fee": False,
            })

    def _replay_events_from_file(self) -> None:
        """
        Replay the JSONL file to rebuild internal state.
        Uses 'fill' events to reconstruct lots and 'realize' events to rebuild cumulatives.
        """
        if not os.path.exists(self.path):
            return

        lots: Dict[str, List[Tuple[float, float, int, bool, str]]] = {}
        trades: List[TradeResult] = []
        cum_real = 0.0
        cum_all = 0.0
        equity_curve: List[Tuple[int, float]] = []

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    typ = obj.get("type")
                    if typ == "fill":
                        token = obj.get("token")
                        side = (obj.get("side") or "").upper()
                        qty = float(obj.get("qty") or 0.0)
                        px = float(obj.get("price_usd") or 0.0)
                        ts = int(obj.get("ts") or _utc_now())
                        dry = bool(obj.get("dry_run"))
                        if qty > 0 and px > 0 and token:
                            lots.setdefault(token, []).append((qty, px, ts, dry, side))
                    elif typ == "realize":
                        token = obj.get("token")
                        qty_closed = float(obj.get("qty_closed") or 0.0)
                        avg_exit = float(obj.get("avg_exit") or 0.0)
                        realized_gross = float(obj.get("realized_gross") or 0.0)
                        realized_net = float(obj.get("realized_net") or 0.0)
                        realized_net_real = float(obj.get("realized_net_real") or realized_net)
                        ts = int(obj.get("ts") or _utc_now())
                        dry = bool(obj.get("dry_run"))
                        cum_all += realized_net
                        if not dry:
                            cum_real += realized_net
                            equity_curve.append((ts, cum_real))
                        # best-effort reconstruct trade
                        if avg_exit and qty_closed:
                            avg_entry = (avg_exit - (realized_gross / qty_closed)) if qty_closed != 0 else 0.0
                            total_entry_cost = avg_entry * qty_closed
                            ret_pct = (realized_net / total_entry_cost) * 100.0 if total_entry_cost else 0.0
                            tr = TradeResult(
                                token=token or "",
                                open_ts=int(obj.get("ts") or ts),
                                close_ts=ts,
                                hold_seconds=0,
                                qty=qty_closed,
                                avg_entry=avg_entry,
                                avg_exit=avg_exit,
                                gross_pnl_usd=realized_gross,
                                net_pnl_usd=realized_net,
                                ret_pct=ret_pct,
                                dry_run=dry,
                            )
                            trades.append(tr)
        except Exception:
            # If replay fails, leave the store empty; not critical
            return

        with self.lock:
            self._lots = lots
            self._trades = trades
            self._cum_net_pnl_real = cum_real
            self._cum_net_pnl_all = cum_all
            self._equity_curve = equity_curve
            self._wins = [t.net_pnl_usd for t in trades if t.net_pnl_usd is not None and t.net_pnl_usd >= 0]
            self._losses = [t.net_pnl_usd for t in trades if t.net_pnl_usd is not None and t.net_pnl_usd < 0]

# ---------- Back-compat factories & aliases ----------
def make_store(logger: Optional[Any] = None,
               metrics_path: Optional[str] = None,
               replay_on_init: bool = True,
               prorate_fees: Optional[bool] = None) -> MetricsStore:
    """
    Back-compat factory. If prorate_fees is None, read SOLO_METRICS_PRORATE_FEES (default '0').
    """
    if prorate_fees is None:
        env_flag = os.getenv("SOLO_METRICS_PRORATE_FEES", "0").strip()
        prorate_fees = (env_flag == "1" or env_flag.lower() in ("true", "yes", "on"))
    return MetricsStore(logger=logger, metrics_path=metrics_path,
                        replay_on_init=replay_on_init, prorate_fees=bool(prorate_fees))

# historical alias some projects used
def new_metrics_store(*args, **kwargs) -> MetricsStore:
    return make_store(*args, **kwargs)

# class alias (older code sometimes imported Metrics instead of MetricsStore)
class Metrics(MetricsStore):
    pass


if __name__ == "__main__":
    # Default remains non-prorated unless you set SOLO_METRICS_PRORATE_FEES=1
    store = make_store(replay_on_init=True)
    now = _utc_now()
    store.on_fill(token_addr="TOKEN-A", side="BUY", qty=100, price_usd=1.0, fee_usd=0.0, dry_run=False, ts=now - 3600)
    store.on_fill(token_addr="TOKEN-A", side="SELL", qty=50, price_usd=1.5, fee_usd=0.0, dry_run=False, ts=now - 1800)
    store.on_fill(token_addr="TOKEN-A", side="SELL", qty=50, price_usd=2.0, fee_usd=0.0, dry_run=False, ts=now)
    store.snapshot_equity_point()
    print("Summary:", store.summary())