"""
Candlestick pattern detectors (small, testable helpers).
Each detector expects a pandas.DataFrame with columns:
['open', 'high', 'low', 'close', 'volume'] and chronological ordering (oldest first).

Functions:
- is_evening_star(df, idx=None)  # tests last 3 candles by default
- is_morning_star(df, idx=None)
- is_three_black_crows(df, idx=None)
- is_three_white_soldiers(df, idx=None)
- is_hammer(df, idx=None)
- is_inverted_hammer(df, idx=None)
- is_bullish_engulfing(df, idx=None)
- is_bearish_engulfing(df, idx=None)
- is_doji(df, idx=None)
"""
from __future__ import annotations
from typing import Optional

import pandas as pd
import numpy as np

COLUMNS = ["open", "high", "low", "close", "volume"]

def _get_candle(df: pd.DataFrame, idx: int) -> pd.Series:
    return df.iloc[idx][["open", "high", "low", "close"]]

def _body_size(c):
    return abs(c["close"] - c["open"])

def _is_bullish(c):
    return c["close"] > c["open"]

def _is_bearish(c):
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
    # c3 should be bullish and close into/above the midpoint of c1 body
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


# --- convenience aggregator used by fetching (non-invasive) -----------------
def classify_patterns(df) -> list:
    """
    Lightweight aggregator returning a list of pattern names detected on the latest
    candle window. Non-exhaustive and conservative — it simply calls the present
    detector helpers and returns their names when True.

    This function exists so callers that expect `classify_patterns` can import it.
    """
    detected = []
    try:
        # call each detector defensively; they expect chronological df (oldest first)
        try:
            if is_evening_star(df):
                detected.append("evening_star")
        except Exception:
            pass
        try:
            if is_morning_star(df):
                detected.append("morning_star")
        except Exception:
            pass
        try:
            if is_three_black_crows(df):
                detected.append("three_black_crows")
        except Exception:
            pass
        try:
            if is_three_white_soldiers(df):
                detected.append("three_white_soldiers")
        except Exception:
            pass
        try:
            if is_hammer(df):
                detected.append("hammer")
        except Exception:
            pass
        try:
            if is_inverted_hammer(df):
                detected.append("inverted_hammer")
        except Exception:
            pass
        try:
            if is_bullish_engulfing(df):
                detected.append("bullish_engulfing")
        except Exception:
            pass
        try:
            if is_bearish_engulfing(df):
                detected.append("bearish_engulfing")
        except Exception:
            pass
        try:
            if is_doji(df):
                detected.append("doji")
        except Exception:
            pass
    except Exception:
        # total failure -> return empty list (do not raise during import)
        return []
    return detected

# Backwards-compat alias some importers may expect
classify_patterns_arrays = classify_patterns