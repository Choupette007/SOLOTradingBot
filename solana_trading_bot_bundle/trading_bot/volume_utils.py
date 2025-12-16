from __future__ import annotations

from typing import Optional, Dict, Any

import pandas as pd
import numpy as np


def rolling_median(series: pd.Series, window: int = 20) -> pd.Series:
    """Return rolling median for the series (right-aligned, min_periods=1)."""
    return series.rolling(window=window, min_periods=1).median()


def rolling_mad(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling MAD (median absolute deviation) defined as median(|x - median(x)|)
    over the rolling window.
    """
    med = rolling_median(series, window=window)
    abs_dev = (series - med).abs()
    return abs_dev.rolling(window=window, min_periods=1).median()


def is_volume_spike(series: pd.Series,
                    idx: Optional[int] = None,
                    window: int = 20,
                    k: float = 3.0,
                    factor: float = 2.0,
                    min_samples: int = 5) -> Dict[str, Any]:
    """
    Decide whether a volume point is a spike using median + k*MAD (robust)
    and also allow a multiplicative fallback (median * factor).

    Parameters
    - series: pd.Series of volume values, chronological order (oldest first)
    - idx: index position (integer) to test; defaults to last point (-1)
    - window: rolling window length used to form baseline (default 20)
    - k: multiplier for MAD threshold (default 3.0)
    - factor: multiplicative fallback threshold on median (default 2.0)
    - min_samples: minimum prior points required to make a robust judgment

    Returns dict:
    {
      "spike": bool,
      "value": float,           # tested value
      "median": float,          # reference median
      "mad": float,             # reference mad
      "mad_threshold": float,   # median + k * mad
      "factor_threshold": float,# median * factor
      "reason": "mad"|"factor"|"insufficient_data"|"not_spike"
    }
    """
    if series is None or len(series) == 0:
        return {"spike": False, "reason": "insufficient_data", "value": None}

    if idx is None:
        idx = len(series) - 1

    # allow negative indexing
    if idx < 0:
        idx = len(series) + idx
    if idx < 0 or idx >= len(series):
        raise IndexError("idx out of bounds")

    # reference window: use prior `window` values excluding the tested point
    start = max(0, idx - window)
    ref = series.iloc[start:idx] if idx > start else series.iloc[start:idx]

    # If not enough prior samples, fallback to multiplicative factor check (but still return helpful metadata)
    if len(ref) < min_samples:
        median_ref = float(ref.median()) if len(ref) > 0 else float(series.median())
        mad_ref = float(((ref - median_ref).abs()).median()) if len(ref) > 0 else 0.0
        value = float(series.iloc[idx])
        factor_threshold = median_ref * factor
        is_spike = value > factor_threshold
        reason = "factor" if is_spike else "insufficient_data"
        return {
            "spike": is_spike,
            "value": value,
            "median": median_ref,
            "mad": mad_ref,
            "mad_threshold": median_ref + k * mad_ref,
            "factor_threshold": factor_threshold,
            "reason": reason,
        }

    median_ref = float(ref.median())
    mad_ref = float(((ref - median_ref).abs()).median())
    mad_threshold = median_ref + k * mad_ref
    factor_threshold = median_ref * factor
    value = float(series.iloc[idx])

    # Primary (robust) decision
    if mad_ref > 0 and value > mad_threshold:
        return {
            "spike": True,
            "value": value,
            "median": median_ref,
            "mad": mad_ref,
            "mad_threshold": mad_threshold,
            "factor_threshold": factor_threshold,
            "reason": "mad",
        }

    # Secondary: multiplicative threshold (for very small MAD or stable medians)
    if value > factor_threshold:
        return {
            "spike": True,
            "value": value,
            "median": median_ref,
            "mad": mad_ref,
            "mad_threshold": mad_threshold,
            "factor_threshold": factor_threshold,
            "reason": "factor",
        }

    return {
        "spike": False,
        "value": value,
        "median": median_ref,
        "mad": mad_ref,
        "mad_threshold": mad_threshold,
        "factor_threshold": factor_threshold,
        "reason": "not_spike",
    }