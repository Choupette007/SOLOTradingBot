from __future__ import annotations

import base64
import json
import logging
import os
import platform
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Optional

# Optional dotenv loader
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore

logger = logging.getLogger("TradingBot.utils.rugcheck_auth")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Try to use bundle constants if available (for appdata/env paths)
try:
    from solana_trading_bot_bundle.common.constants import appdata_dir, env_path as _env_path  # type: ignore
except Exception:
    appdata_dir = None  # type: ignore
    _env_path = None    # type: ignore

# Optionally import the bundle's async login (for fallback refresh)
try:
    from solana_trading_bot_bundle.trading_bot import rugcheck_auth as bundle_auth  # type: ignore
except Exception:
    bundle_auth = None  # type: ignore


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _strip_quotes(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


# ---------------------------------------------------------------------
# .env discovery & loading (robust, OS-aware)
# ---------------------------------------------------------------------
def _candidate_env_paths() -> List[Path]:
    cand: List[Path] = []

    # Highest-priority override
    p = os.getenv("SOLBOT_ENV") or os.getenv("SOLBOT_DOTENV")
    if p:
        cand.append(Path(p).expanduser())

    # Project local
    cand.append(Path.cwd() / ".env")

    # Bundle-provided paths (if any)
    try:
        if _env_path is not None:
            pth = _env_path() if callable(_env_path) else _env_path  # type: ignore
            cand.append(Path(pth))
    except Exception:
        pass

    try:
        if appdata_dir is not None:
            base = appdata_dir() if callable(appdata_dir) else Path(appdata_dir)  # type: ignore
            cand.append(Path(base) / ".env")
    except Exception:
        pass

    # OS fallbacks
    system = platform.system()
    if system == "Darwin":
        for folder in ("SolanaBot", "SolanaTradingBot", "SOLOTradingBot"):
            cand.append(Path.home() / "Library" / "Application Support" / folder / ".env")
    elif system == "Windows":
        appdata = os.getenv("APPDATA")
        local = os.getenv("LOCALAPPDATA")
        if appdata:
            cand.append(Path(appdata) / "SOLOTradingBot" / ".env")
        if local:
            for folder in ("SolanaBot", "SolanaTradingBot", "SOLOTradingBot"):
                cand.append(Path(local) / folder / ".env")
    else:  # Linux/other
        cand.append(Path.home() / ".local" / "share" / "SOLOTradingBot" / ".env")

    # Deduplicate while preserving order
    seen = set()
    out: List[Path] = []
    for c in cand:
        try:
            key = c.resolve()
        except Exception:
            key = c
        if key not in seen:
            out.append(c)
            seen.add(key)
    return out


def _load_dotenv_if_possible() -> Optional[Path]:
    """
    Load the first existing .env so os.getenv(...) sees values (no overrides).
    """
    for envp in _candidate_env_paths():
        try:
            if not envp.exists():
                continue
            if load_dotenv:
                load_dotenv(dotenv_path=envp, override=False)
            else:
                for raw in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), _strip_quotes(v.strip()))
            return envp
        except Exception:
            continue
    return None


def _reload_env_override() -> Optional[Path]:
    """
    Reload the first existing .env and override os.environ values.
    Use after a refresh so the new JWT is visible immediately.
    """
    for envp in _candidate_env_paths():
        try:
            if not envp.exists():
                continue
            if load_dotenv:
                load_dotenv(dotenv_path=envp, override=True)
            else:
                for raw in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = _strip_quotes(v.strip())
            return envp
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------
def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _decode_jwt_expiry(token: str) -> Optional[int]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None


def _token_expired(token: str, skew_seconds: int = 60) -> bool:
    if not token:
        return True
    exp = _decode_jwt_expiry(token)
    if exp is None:  # treat no-exp as valid (non-expiring dev token)
        return False
    return (time.time() + max(0, skew_seconds)) >= exp


def _get_env_jwt() -> str:
    for k in ("RUGCHECK_JWT_TOKEN", "RUGCHECK_JWT", "RUGCHECK_API_TOKEN"):
        v = os.getenv(k)
        if v:
            v = _strip_quotes(v)
            if v:
                return v
    return ""


# ---------------------------------------------------------------------
# Persist helpers (write new JWT to .env and process env)
# ---------------------------------------------------------------------
def _load_env_map(envp: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if envp.exists():
        for raw in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = _strip_quotes(v.strip())
    return out


def _write_env_file(envp: Path, kv: Dict[str, str]) -> None:
    lines = ["# SOLOTradingBot .env file"]
    for k, v in kv.items():
        s = str(v or "").replace("\r", "").replace("\n", "")
        lines.append(f'{k}="{s}"')
    tmp = envp.with_suffix(envp.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(envp)
    if os.name != "nt":
        try:
            os.chmod(envp, 0o600)
        except Exception:
            pass


def _persist_token(token: str) -> None:
    """
    Persist token to all candidate .env files (best-effort) and update current process env.
    Writes both RUGCHECK_JWT and RUGCHECK_JWT_TOKEN for compatibility.
    """
    token = _strip_quotes(token)
    if not token:
        return

    cands = _candidate_env_paths()
    written = []
    first_written = None

    for envp in cands:
        try:
            envp.parent.mkdir(parents=True, exist_ok=True)
            kv = _load_env_map(envp)
            kv["RUGCHECK_JWT"] = token
            kv["RUGCHECK_JWT_TOKEN"] = token
            _write_env_file(envp, kv)
            written.append(str(envp))
            if first_written is None:
                first_written = envp
        except Exception as e:
            logger.debug("Failed to persist Rugcheck token to %s: %s", envp, e, exc_info=True)
            continue

    # Update current process environment variables immediately
    os.environ["RUGCHECK_JWT"] = token
    os.environ["RUGCHECK_JWT_TOKEN"] = token

    if written:
        logger.info("Persisted Rugcheck JWT to %d candidate .env files; first=%s", len(written), str(first_written))
        logger.debug("Persisted paths: %s", written)
    else:
        logger.warning("Persisting Rugcheck JWT failed for all candidate paths.")


# ---------------------------------------------------------------------
# Login helper discovery & invocation (non-blocking)
# ---------------------------------------------------------------------
def _find_login_helper() -> Optional[Path]:
    p = os.getenv("RUGCHECK_LOGIN_HELPER", "").strip()
    if p:
        cand = Path(p).expanduser()
        if cand.exists():
            return cand

    try:
        spec = find_spec("solana_trading_bot_bundle.trading_bot.rugcheck_login")
        if spec and spec.origin:
            return Path(spec.origin)
    except Exception:
        pass

    cand = Path.cwd() / "rugcheck_login.py"
    if cand.exists():
        return cand

    this_dir = Path(__file__).resolve().parent
    cand = (this_dir.parent / "trading_bot" / "rugcheck_login.py").resolve()
    if cand.exists():
        return cand

    return None


def _run_silent_login_helper(timeout_seconds: Optional[float] = None) -> None:
    if not _env_bool("RUGCHECK_ENABLE", True):
        return
    if not _env_bool("RUGCHECK_LOGIN_ENABLE", True):
        return

    helper = _find_login_helper()
    if not helper:
        return

    if timeout_seconds is None:
        try:
            timeout_seconds = float(os.getenv("RUGCHECK_LOGIN_TIMEOUT_SECONDS", "8"))
        except Exception:
            timeout_seconds = 8.0

    try:
        subprocess.run(
            [sys.executable, str(helper), "--try-http"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "RUGCHECK_LOGIN_SILENT": "1"},
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        logger.debug("Silent login helper invocation failed", exc_info=True)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def get_rugcheck_token() -> str:
    """
    Return a JWT from env (first of RUGCHECK_JWT_TOKEN / RUGCHECK_JWT / RUGCHECK_API_TOKEN),
    provided it is not expired. If expired/missing, returns ''.
    """
    _load_dotenv_if_possible()
    tok = _get_env_jwt()
    if tok and _token_expired(tok):
        return ""
    return tok


def ensure_valid_rugcheck_headers(session=None, force_refresh: bool = False) -> Dict[str, str]:
    """
    Return Authorization headers for Rugcheck.

    Flow:
      1) If a valid token exists and force_refresh=False => return it (no subprocess).
      2) Else try helper refresh (short timeout) and reload .env into process.
      3) If still no token AND bundle async login is available, call it and persist the new token.
      4) Return {'Authorization': 'Bearer ...'} if we have a token; else {}.

    Notes:
      - Respects RUGCHECK_ENABLE / RUGCHECK_LOGIN_ENABLE.
      - Persists new tokens to .env + os.environ (no restart needed).
    """
    if not _env_bool("RUGCHECK_ENABLE", True):
        return {}

    tok = get_rugcheck_token()
    if tok and not force_refresh:
        return {"Authorization": f"Bearer {tok}"}

    # 2) Attempt helper-based refresh (non-blocking)
    _run_silent_login_helper()
    _reload_env_override()

    tok = get_rugcheck_token()
    if tok:
        return {"Authorization": f"Bearer {tok}"}

    # 3) Fallback: bundle's async login (if available)
    if bundle_auth is not None:
        try:
            import asyncio

            async def _call():
                # Request a true refresh regardless of current env token
                return await bundle_auth.ensure_valid_rugcheck_headers(force_refresh=True)  # type: ignore

            def _run_coro(coro):
                # Safe runner even if an event loop is already running
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    # Run in a background thread with its own loop
                    import threading, queue
                    q: "queue.Queue[object]" = queue.Queue()

                    def worker():
                        try:
                            q.put(asyncio.run(coro))
                        except Exception as e:
                            q.put(e)

                    t = threading.Thread(target=worker, daemon=True)
                    t.start()
                    t.join(float(os.getenv("RUGCHECK_LOGIN_TIMEOUT_TOTAL", "15")))
                    return None if q.empty() else q.get()
                else:
                    return asyncio.run(coro)

            res = _run_coro(_call())
            if isinstance(res, dict):
                # Extract token from header, persist for future calls
                bearer = res.get("Authorization", "")
                if bearer.startswith("Bearer "):
                    _persist_token(bearer[len("Bearer "):].strip())
                # Reload so process env reflects persisted value
                _reload_env_override()
                tok = _get_env_jwt()
                if tok:
                    return {"Authorization": f"Bearer {tok}"}
        except Exception:
            pass

    # 4) Give up (fail-open for caller)
    return {}


def get_rugcheck_headers(force_refresh: bool = False) -> Dict[str, str]:
    """Convenience wrapper used across the codebase."""
    return ensure_valid_rugcheck_headers(session=None, force_refresh=force_refresh)


# Back-compat placeholder (do not populate at import time)
HEADERS: Dict[str, str] = {}