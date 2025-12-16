from __future__ import annotations

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

COLUMNS = ["open", "high", "low", "close", "volume"]


def _get_candle(df: pd.DataFrame, idx: int) -> pd.Series:
    return df.iloc[idx][["open", "high", "low", "close"]]


def _body_size(c: pd.Series) -> float:
    return abs(c["close"] - c["open"])


def _is_bullish(c: pd.Series) -> bool:
    return c["close"] > c["open"]


def _is_bearish(c: pd.Series) -> bool:
    return c["close"] < c["open"]


def is_doji(df: pd.DataFrame, idx: Optional[int] = None, max_ratio: float = 0.1) -> bool:
    if idx is None:
        idx = len(df) - 1
    c = _get_candle(df, idx)
    body = _body_size(c)
    total = c["high"] - c["low"]
    if total == 0:
        return False
    return (body / total) <= max_ratio


def is_hammer(df: pd.DataFrame, idx: Optional[int] = None, body_to_tail: float = 2.0) -> bool:
    if idx is None:
        idx = len(df) - 1
    c = _get_candle(df, idx)
    body = _body_size(c)
    lower_tail = min(c["open"], c["close"]) - c["low"]
    upper_tail = c["high"] - max(c["open"], c["close"])
    # hammer: small body, long lower tail, little or no upper tail
    if body == 0:
        return False
    return (lower_tail >= body * body_to_tail) and (upper_tail <= body * 0.5)


def is_inverted_hammer(df: pd.DataFrame, idx: Optional[int] = None, body_to_tail: float = 2.0) -> bool:
    if idx is None:
        idx = len(df) - 1
    c = _get_candle(df, idx)
    body = _body_size(c)
    lower_tail = min(c["open"], c["close"]) - c["low"]
    upper_tail = c["high"] - max(c["open"], c["close"])
    # inverted hammer: small body, long upper tail, little lower tail
    if body == 0:
        return False
    return (upper_tail >= body * body_to_tail) and (lower_tail <= body * 0.5)


def is_bullish_engulfing(df: pd.DataFrame, idx: Optional[int] = None) -> bool:
    if idx is None:
        idx = len(df) - 1
    if idx < 1:
        return False
    prev = _get_candle(df, idx - 1)
    cur = _get_candle(df, idx)
    return _is_bullish(cur) and _is_bearish(prev) and (cur["open"] < prev["close"]) and (cur["close"] > prev["open"])


def is_bearish_engulfing(df: pd.DataFrame, idx: Optional[int] = None) -> bool:
    if idx is None:
        idx = len(df) - 1
    if idx < 1:
        return False
    prev = _get_candle(df, idx - 1)
    cur = _get_candle(df, idx)
    return _is_bearish(cur) and _is_bullish(prev) and (cur["open"] > prev["close"]) and (cur["close"] < prev["open"])


def is_three_white_soldiers(df: pd.DataFrame, idx: Optional[int] = None) -> bool:
    """Three consecutive strong bullish candles with progressively higher opens/closes."""
    if idx is None:
        idx = len(df) - 1
    if idx < 2:
        return False
    c1 = _get_candle(df, idx - 2)
    c2 = _get_candle(df, idx - 1)
    c3 = _get_candle(df, idx)
    cond = all(_is_bullish(c) for c in (c1, c2, c3))
    cond = cond and (c2["open"] > c1["open"]) and (c2["close"] > c1["close"])
    cond = cond and (c3["open"] > c2["open"]) and (c3["close"] > c2["close"])
    return bool(cond)


def is_three_black_crows(df: pd.DataFrame, idx: Optional[int] = None) -> bool:
    """Three consecutive bearish candles with progressively lower opens/closes."""
    if idx is None:
        idx = len(df) - 1
    if idx < 2:
        return False
    c1 = _get_candle(df, idx - 2)
    c2 = _get_candle(df, idx - 1)
    c3 = _get_candle(df, idx)
    cond = all(_is_bearish(c) for c in (c1, c2, c3))
    cond = cond and (c2["open"] < c1["open"]) and (c2["close"] < c1["close"])
    cond = cond and (c3["open"] < c2["open"]) and (c3["close"] < c2["close"])
    return bool(cond)


def is_morning_star(df: pd.DataFrame, idx: Optional[int] = None) -> bool:
    """Mirror of evening star: bearish -> small -> bullish closing into or above first body midpoint."""
    if idx is None:
        idx = len(df) - 1
    if idx < 2:
        return False
    c1 = _get_candle(df, idx - 2)  # bearish
    c2 = _get_candle(df, idx - 1)  # small indecision
    c3 = _get_candle(df, idx)      # bullish
    if not _is_bearish(c1):
        return False
    if abs(c2["close"] - c2["open"]) > _body_size(c1) * 0.5:
        # c2 should be a small body relative to c1
        return False
    midpoint = (c1["open"] + c1["close"]) / 2.0
    return _is_bullish(c3) and (c3["close"] > midpoint)


def is_evening_star(df: pd.DataFrame, idx: Optional[int] = None) -> bool:
    """
    Simple evening star test:
    - first candle bullish with significant body
    - second candle small body (gap up or indecision)
    - third candle bearish and closes well into the body of the first (below midpoint)
    """
    if idx is None:
        idx = len(df) - 1
    if idx < 2:
        return False
    c1 = _get_candle(df, idx - 2)  # bullish
    c2 = _get_candle(df, idx - 1)  # small
    c3 = _get_candle(df, idx)      # bearish
    if not _is_bullish(c1):
        return False
    # require c1 body to be reasonably large relative to candle range
    if _body_size(c1) < (c1["high"] - c1["low"]) * 0.2:
        return False
    # c2 should be small
    if _body_size(c2) > _body_size(c1) * 0.5:
        return False
    # c3 should be bearish and close below midpoint of c1 body
    midpoint = (c1["open"] + c1["close"]) / 2.0
    return _is_bearish(c3) and (c3["close"] < midpoint)


# ---------------------------------------------------------------------------
# classify_patterns(...) entrypoint
#
# Accepts either:
#  - a pandas DataFrame with columns at least ["open","high","low","close"] (volume optional)
#  - OR an OHLC dict: {"open": [...], "high":[...], "low":[...], "close":[...], "volume":[...]} (lists)
#
# Returns a dict mapping pattern_name -> list[bool] (one value per bar). Defensive:
# on any internal error returns {} rather than raising.
# ---------------------------------------------------------------------------
def classify_patterns(ohlcv: Dict[str, Any] | pd.DataFrame) -> Dict[str, Any]:
    """
    Produce per-bar pattern hit lists for supported candlestick patterns.
    """
    try:
        # Normalize to DataFrame
        if isinstance(ohlcv, dict):
            try:
                df = pd.DataFrame({
                    "open":  list(ohlcv.get("open", [])),
                    "high":  list(ohlcv.get("high", [])),
                    "low":   list(ohlcv.get("low", [])),
                    "close": list(ohlcv.get("close", [])),
                })
                if "volume" in ohlcv:
                    df["volume"] = list(ohlcv.get("volume", []))
                else:
                    df["volume"] = [0] * len(df)
            except Exception:
                return {}
        elif isinstance(ohlcv, pd.DataFrame):
            df = ohlcv.copy()
            for c in ("open", "high", "low", "close"):
                if c not in df.columns:
                    return {}
            if "volume" not in df.columns:
                df["volume"] = 0
        else:
            return {}

        n = len(df)
        if n == 0:
            return {}

        results: Dict[str, list] = {}
        pattern_funcs = [
            ("doji", is_doji),
            ("hammer", is_hammer),
            ("inverted_hammer", is_inverted_hammer),
            ("bullish_engulfing", is_bullish_engulfing),
            ("bearish_engulfing", is_bearish_engulfing),
            ("three_white_soldiers", is_three_white_soldiers),
            ("three_black_crows", is_three_black_crows),
            ("morning_star", is_morning_star),
            ("evening_star", is_evening_star),
        ]

        for name, fn in pattern_funcs:
            hits = []
            for idx in range(n):
                try:
                    hits.append(bool(fn(df, idx)))
                except Exception:
                    hits.append(False)
            results[name] = hits

        return results
    except Exception:
        return {}