# https://github.com/your/repo/placeholder-path
"""Small canonical helpers for trading.py to deduplicate repeated small coercers."""

from typing import Any

def _to_float(x: Any, default: float = 0.0) -> float:
    """
    Robust float coercer:
      - Accepts None, numeric strings with commas, dicts like {"usd": ...}
      - Returns `default` on parse error.
    """
    try:
        if x is None:
            return float(default)
        # Accept dicts like {"usd": ...}
        if isinstance(x, dict):
            # prefer "usd" key if present
            v = x.get("usd", None)
            if v is None and x:
                # fallback to first numeric-like value
                for vv in x.values():
                    if isinstance(vv, (int, float, str)):
                        v = vv
                        break
            if v is None:
                return float(default)
            return float(v)
        # Strings: strip commas and currency symbols
        if isinstance(x, str):
            s = x.strip().replace(",", "").replace("$", "")
            if s == "":
                return float(default)
            return float(s)
        # numeric-ish
        return float(x)
    except Exception:
        try:
            return float(default)
        except Exception:
            return 0.0


def _to_int(x: Any, default: int = 0) -> int:
    """
    Robust int coercer:
      - Accepts numeric strings, floats and ints, dicts with numeric values.
      - Returns `default` on parse error.
    """
    try:
        if x is None:
            return int(default)
        if isinstance(x, dict):
            v = x.get("usd", None)
            if v is None and x:
                for vv in x.values():
                    if isinstance(vv, (int, float, str)):
                        v = vv
                        break
            if v is None:
                return int(default)
            return int(float(v))
        if isinstance(x, str):
            s = x.strip().replace(",", "").replace("$", "")
            if s == "":
                return int(default)
            return int(float(s))
        return int(x)
    except Exception:
        try:
            return int(default)
        except Exception:
            return 0


def truthy_env(name: str, default: str = "0") -> bool:
    """
    Canonical truthy env parser used across the module.
    Returns True for values like "1","true","yes","on","y" (case-insensitive).
    """
    try:
        v = (__import__("os").getenv(name, default) or "").strip().lower()
        return v in ("1", "true", "yes", "on", "y")
    except Exception:
        return False