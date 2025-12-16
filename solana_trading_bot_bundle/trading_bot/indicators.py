"""
SOLO Trading Bot — Indicators
=============================

Fast, vectorized NumPy indicators plus pure‑Python helpers (no NumPy) for portability.

Exports
-------
- rsi(close, period=14) -> np.ndarray
- ema(arr, period) -> np.ndarray
- macd(close, fast=12, slow=26, signal=9) -> {"macd","signal","hist"} arrays
- bollinger(close, period=20, stddev=2.0) -> {"mid","upper","lower","width","percent_b"} arrays
- td9(close, lookback=30) -> {"up","down","setup","td9_up","td9_down"} arrays
- bollinger_bands(close, length=34, mult=2.0) -> (upper, basis, lower) lists (pure‑Python, warmup=None)
- atr(high, low, close, length=14) -> ATR list (pure‑Python, warmup=None)
"""
from __future__ import annotations
import math
from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np

__all__ = [
    "rsi", "ema", "macd", "bollinger", "td9",
    "bollinger_bands", "atr"
]


def _to_np(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def rsi(close: Sequence[float], period: int = 14) -> np.ndarray:
    """
    Wilder RSI (14 by default).
    Returns an ndarray same length as `close` with np.nan for warmup indices.

    Note: when both avg_gain and avg_loss are 0 (flat-price region), this implementation
    returns 50.0 (neutral) rather than 100 or 0.
    """
    c = _to_np(close)
    n = c.size
    if n < period + 1:
        return np.full(n, np.nan)

    diff = np.diff(c)  # length n-1
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)

    # initial Wilder averages are mean of first `period` diffs
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    out = np.full(n, np.nan)
    # first RSI is placed at index `period` (differences start at index 1)
    if avg_gain == 0.0 and avg_loss == 0.0:
        out[period] = 50.0
    elif avg_loss == 0.0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))

    ag = avg_gain
    al = avg_loss
    # iterate diffs from index `period` .. `n-2` -> price indices `period+1` .. `n-1`
    for i in range(period, n - 1):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        price_idx = i + 1
        if ag == 0.0 and al == 0.0:
            out[price_idx] = 50.0
        elif al == 0.0:
            out[price_idx] = 100.0
        else:
            rs = ag / al
            out[price_idx] = 100.0 - (100.0 / (1.0 + rs))
    return out


def ema(arr: Sequence[float], period: int) -> np.ndarray:
    """
    Exponential Moving Average (EMA), seeded with SMA of first `period` values.
    Returns an ndarray same length as input, with np.nan for indices < period-1.

    If the seed window contains NaNs, this implementation returns all-NaN (can't seed).
    """
    x = _to_np(arr)
    n = x.size
    if period < 1:
        raise ValueError("period must be >= 1")
    if n < period:
        return np.full(n, np.nan)

    k = 2.0 / (period + 1.0)
    out = np.full(n, np.nan)
    seed_slice = x[:period]
    if np.isnan(seed_slice).any():
        # Ambiguous seed: don't attempt to seed through NaNs here.
        return out
    seed = seed_slice.mean()
    out[period - 1] = seed
    for i in range(period, n):
        out[i] = x[i] * k + out[i - 1] * (1.0 - k)
    return out


def macd(close: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    """
    MACD: macd line = EMA(fast) - EMA(slow); signal = EMA(macd_line, signal); hist = macd - signal
    """
    c = _to_np(close)
    ema_fast = ema(c, fast)
    ema_slow = ema(c, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist}


def bollinger(close: Sequence[float], period: int = 20, stddev: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Vectorized Bollinger bands using convolution for O(n) performance.
    Returns dict with mid, upper, lower, width, percent_b arrays (np.ndarrays).
    Warmup values are np.nan for indices < period - 1.

    Robustness:
      - small negative variance rounding errors are clipped to 0.
      - width is set to NaN where mid (basis) == 0 to avoid infinities.
      - percent_b is computed only where band width (denom) != 0.
    """
    c = _to_np(close)
    n = c.size
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    width = np.full(n, np.nan)
    percent_b = np.full(n, np.nan)

    if n < period:
        return {"mid": mid, "upper": upper, "lower": lower, "width": width, "percent_b": percent_b}

    # rolling sum and sum of squares via convolution (valid returns length n - period + 1)
    window = np.ones(period, dtype=float)
    sum_ = np.convolve(c, window, mode="valid")
    sumsq = np.convolve(c * c, window, mode="valid")
    m = sum_ / period
    var = (sumsq / period) - (m * m)
    # numerical safety: clip tiny negative variance due to rounding
    var = np.where(var < 0, 0.0, var)
    s = np.sqrt(var)

    mid[period - 1 :] = m
    upper_seg = m + stddev * s
    lower_seg = m - stddev * s
    upper[period - 1 :] = upper_seg
    lower[period - 1 :] = lower_seg

    # width and percent_b (robust handling for mid == 0 and denom == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # width = (upper - lower) / mid, only where mid != 0
        width_vals = np.full_like(m, np.nan)
        nonzero_mid = m != 0.0
        width_vals[nonzero_mid] = (upper_seg[nonzero_mid] - lower_seg[nonzero_mid]) / m[nonzero_mid]
        width[period - 1 :] = width_vals

        denom = upper_seg - lower_seg
        percent_b_vals = np.full_like(m, np.nan)
        nonzero_denom = denom != 0.0
        if nonzero_denom.any():
            # price segment aligned with m: c[period-1:]
            price_seg = c[period - 1 :]
            idxs = nonzero_denom
            percent_b_vals[idxs] = (price_seg[idxs] - lower_seg[idxs]) / denom[idxs]
        percent_b[period - 1 :] = percent_b_vals

    return {"mid": mid, "upper": upper, "lower": lower, "width": width, "percent_b": percent_b}


def td9(close: Sequence[float], lookback: int = 30) -> Dict[str, np.ndarray]:
    """
    Basic TD Sequential setup counts:
    - Bullish setup (down count): close[n] < close[n-4]
    - Bearish setup (up count)  : close[n] > close[n-4]

    Returns counts, textual setup, and booleans at 9 for up/down.
    """
    c = _to_np(close)
    n = c.size
    up = np.zeros(n, dtype=int)
    down = np.zeros(n, dtype=int)
    setup = np.full(n, "", dtype=object)
    td9_up = np.zeros(n, dtype=bool)
    td9_down = np.zeros(n, dtype=bool)

    for i in range(4, n):
        if c[i] > c[i - 4]:
            up[i] = (up[i - 1] + 1) if up[i - 1] > 0 else 1
            down[i] = 0
        elif c[i] < c[i - 4]:
            down[i] = (down[i - 1] + 1) if down[i - 1] > 0 else 1
            up[i] = 0
        else:
            up[i] = 0
            down[i] = 0
        if up[i] == 9:
            setup[i] = "td9_up"
            td9_up[i] = True
        elif down[i] == 9:
            setup[i] = "td9_down"
            td9_down[i] = True

    if lookback and n > lookback:
        start = n - lookback
        td9_up[:start] = False
        td9_down[:start] = False

    return {"up": up, "down": down, "setup": setup, "td9_up": td9_up, "td9_down": td9_down}


# ===================== Pure-Python helpers (optimized O(n)) =====================
def _prefix_sums(values: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Return prefix sums and prefix sums of squares for a numeric sequence."""
    n = len(values)
    ps = [0.0] * (n + 1)
    pssq = [0.0] * (n + 1)
    for i, v in enumerate(values):
        fv = float(v)
        ps[i + 1] = ps[i] + fv
        pssq[i + 1] = pssq[i] + fv * fv
    return ps, pssq


def _sma(values: Sequence[float], length: int) -> List[Optional[float]]:
    if length <= 0:
        raise ValueError("length must be > 0")
    n = len(values)
    out: List[Optional[float]] = [None] * n
    if n < length:
        return out
    ps, _ = _prefix_sums(values)
    for i in range(length - 1, n):
        s = ps[i + 1] - ps[i + 1 - length]
        out[i] = s / length
    return out


def _stdev(values: Sequence[float], length: int) -> List[Optional[float]]:
    """
    Rolling population standard deviation (ddof=0) using prefix sums for O(n).
    """
    if length <= 1:
        raise ValueError("length must be > 1")
    n = len(values)
    out: List[Optional[float]] = [None] * n
    if n < length:
        return out
    ps, pssq = _prefix_sums(values)
    for i in range(length - 1, n):
        s = ps[i + 1] - ps[i + 1 - length]
        ssq = pssq[i + 1] - pssq[i + 1 - length]
        mean = s / length
        var = (ssq / length) - (mean * mean)
        if var < 0 and var > -1e-12:
            var = 0.0
        out[i] = math.sqrt(var) if var >= 0.0 else None
    return out


def bollinger_bands(close: Sequence[float], length: int = 34, mult: float = 2.0) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    Pure-Python Bollinger bands (upper, basis, lower). Warmup entries are None.
    Uses prefix-sums to compute rolling mean and std in O(n).
    """
    basis = _sma(close, length)
    dev = _stdev(close, length)
    n = len(close)
    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    for i in range(n):
        if basis[i] is not None and dev[i] is not None:
            b = float(basis[i])
            d = float(dev[i])
            upper[i] = b + mult * d
            lower[i] = b - mult * d
    return upper, basis, lower


def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float], length: int = 14) -> List[Optional[float]]:
    """
    Wilder ATR (pure Python). Returns list with leading None for warmup bars.
    Uses Wilder smoothing (first value is simple average of the first `length` TRs).
    """
    n = len(close)
    if not (len(high) == len(low) == n):
        raise ValueError("high/low/close must be same length")
    if length <= 0:
        raise ValueError("length must be > 0")

    tr: List[float] = [0.0] * n
    for i in range(n):
        h = float(high[i])
        l = float(low[i])
        if i == 0:
            tr[i] = h - l
        else:
            pc = float(close[i - 1])
            tr[i] = max(h - l, abs(h - pc), abs(l - pc))

    out: List[Optional[float]] = [None] * n
    if n < length:
        return out

    s = sum(tr[:length])
    out[length - 1] = s / length

    for i in range(length, n):
        prev = out[i - 1]
        assert prev is not None
        out[i] = ((prev * (length - 1)) + tr[i]) / length

    return out

