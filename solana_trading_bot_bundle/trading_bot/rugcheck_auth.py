# solana_trading_bot_bundle/trading_bot/rugcheck_auth.py
# Auto-verify Rugcheck JWT and auto-refresh via an external login script.
# On success, persist JWT to multiple .env candidate locations and update process env.
from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
import threading
import queue
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from importlib.util import find_spec

import aiohttp

# For signing the login message
from solders.keypair import Keypair

logger = logging.getLogger("TradingBot")
if not logger.handlers:
    # If running standalone during debugging, ensure minimal configuration.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --------------------------
# .env resolution & I/O
# --------------------------
def _appdata_base() -> Path:
    try:
        from solana_trading_bot_bundle.common.constants import appdata_dir as _appdata_dir  # type: ignore
        base = _appdata_dir() if callable(_appdata_dir) else Path(_appdata_dir)
    except Exception:
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / ".local" / "share")) / "SOLOTradingBot"
    base.mkdir(parents=True, exist_ok=True)
    return base

def _env_file_path() -> Path:
    """
    Legacy single-path accessor (kept for compatibility). Returns the primary env file path
    the bundle historically used (fallback to _appdata_base()).
    """
    try:
        from solana_trading_bot_bundle.common.constants import env_path as _env_path  # type: ignore
        envp = _env_path() if callable(_env_path) else Path(_env_path)
    except Exception:
        envp = _appdata_base() / ".env"
    return Path(envp)

def _candidate_env_paths() -> List[Path]:
    """
    Return ordered candidate .env paths that the bot and helpers may use.
    Order: explicit override, project .env, bundle env_path, Roaming APPDATA, LocalAppData, fallback appdata base.
    """
    cand: List[Path] = []

    # Highest-priority override
    p = os.getenv("SOLBOT_ENV") or os.getenv("SOLBOT_DOTENV")
    if p:
        cand.append(Path(p).expanduser())

    # Project local
    cand.append(Path.cwd() / ".env")

    # Bundle-provided env_path if available
    try:
        from solana_trading_bot_bundle.common.constants import env_path as _env_path  # type: ignore
        pth = _env_path() if callable(_env_path) else Path(_env_path)
        cand.append(Path(pth))
    except Exception:
        pass

    # Windows APPDATA/LOCALAPPDATA
    appdata = os.getenv("APPDATA")
    localapp = os.getenv("LOCALAPPDATA")
    if appdata:
        cand.append(Path(appdata) / "SOLOTradingBot" / ".env")
    if localapp:
        cand.append(Path(localapp) / "SOLOTradingBot" / ".env")

    # Fallback (matches previous behavior)
    cand.append(_appdata_base() / ".env")

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

def _load_env_map(envp: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if envp.exists():
        for raw in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v.startswith('"') and v.endswith('"') and len(v) >= 2:
                v = v[1:-1]
            out[k] = v
    return out

def _write_env_file(envp: Path, kv: Dict[str, str]) -> None:
    """
    Atomic-write a .env file from kv mapping. Values will be quoted.
    """
    lines = ["# SOLOTradingBot .env file"]
    # keep original order stable-ish
    for k, v in kv.items():
        s = "" if v is None else str(v).replace("\r", "").replace("\n", "").replace('"', r'\"')
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
    This ensures both Roaming and LocalAppData .env (and project .env) are updated.
    """
    token = (token or "").strip()
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

def _load_dotenv_into_env() -> None:
    """
    Load first existing .env into process env without overriding existing variables.
    """
    for envp in _candidate_env_paths():
        try:
            if not envp.exists():
                continue
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(dotenv_path=envp, override=False)
            except Exception:
                # fallback simple parser
                for raw in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    v = v.strip()
                    if v.startswith('"') and v.endswith('"') and len(v) >= 2:
                        v = v[1:-1]
                    os.environ.setdefault(k.strip(), v)
            return
        except Exception:
            continue

def _reload_env_override() -> Optional[Path]:
    """
    Reload the first existing .env and override os.environ values.
    Use this after a refresh so the new JWT is visible immediately.
    Returns the path reloaded or None.
    """
    for envp in _candidate_env_paths():
        try:
            if not envp.exists():
                continue
            try:
                from dotenv import load_dotenv  # type: ignore
                load_dotenv(dotenv_path=envp, override=True)
            except Exception:
                for raw in envp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    v = v.strip()
                    if v.startswith('"') and v.endswith('"') and len(v) >= 2:
                        v = v[1:-1]
                    os.environ[k.strip()] = v
            return envp
        except Exception:
            continue
    return None

# --------------------------
# JWT helpers
# --------------------------
def _decode_jwt_noverify(token: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        import base64
        parts = token.split(".")
        if len(parts) != 3:
            return None, None
        # Add padding safely
        payload_b64 = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8", errors="ignore")
        data = json.loads(payload_json)
        exp = data.get("exp")
        iat = data.get("iat")
        return (int(exp) if exp is not None else None, int(iat) if iat is not None else None)
    except Exception:
        return None, None

def _is_token_expired(token: str, skew_seconds: int = 60) -> bool:
    exp, _ = _decode_jwt_noverify(token)
    if exp is None:
        return True
    return int(time.time()) >= (exp - max(0, skew_seconds))

def get_rugcheck_headers(force_refresh: bool = False) -> Dict[str, str]:
    """
    Sync, no-network version (for quick callers). Returns whatever is in env.
    Network refresh is done by ensure_valid_rugcheck_headers(...).
    """
    _load_dotenv_into_env()
    tok = (os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT") or "").strip()
    if not tok or (force_refresh and _is_token_expired(tok)):
        return {}
    return {"Authorization": f"Bearer {tok}"}

# --------------------------
# Real login: /auth/login/solana
# --------------------------
def _build_login_payload(kp: Keypair) -> Dict:
    """
    Builds the canonical message + signature required by RugCheck.
    """
    msg = {
        "message": "Sign-in to Rugcheck.xyz",
        "timestamp": int(time.time() * 1000),  # ms
        "publicKey": str(kp.pubkey()),
    }
    # Deterministic JSON for signing
    msg_json = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
    sig = kp.sign_message(msg_json.encode("utf-8"))

    # Signature bytes -> list[int]
    try:
        sig_bytes = bytes(sig)  # works on recent solders
    except Exception:
        import base58  # pip install base58
        sig_bytes = base58.b58decode(str(sig))

    return {
        "message": msg,  # object (same fields we signed)
        "signature": {"data": list(sig_bytes), "type": "ed25519"},
        "wallet": str(kp.pubkey()),
    }

async def _login_via_solana(session: aiohttp.ClientSession, login_url: str) -> str:
    """
    POST /auth/login/solana with signed message; return the raw JWT or ''.
    """
    priv = (os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("WALLET_PRIVATE_KEY") or "").strip()
    if not priv:
        logger.error("SOLANA_PRIVATE_KEY (or WALLET_PRIVATE_KEY) not set; cannot login.")
        return ""
    try:
        kp = Keypair.from_base58_string(priv)
    except Exception:
        logger.error("SOLANA_PRIVATE_KEY is not a valid base58-encoded secret key.")
        return ""

    payload = _build_login_payload(kp)
    try:
        async with session.post(
            login_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=20, sock_connect=6, sock_read=12),
        ) as resp:
            body = await resp.text()
            if resp.status != 200:
                logger.error("RugCheck login failed: HTTP %s, body: %s", resp.status, body)
                return ""
            try:
                data = json.loads(body)
            except Exception:
                logger.error("Unexpected RugCheck login response: %s", body[:200])
                return ""
            # Handle common field names robustly
            token = (
                data.get("token")
                or data.get("jwt")
                or data.get("jwt_token")
                or data.get("accessToken")
                or data.get("access_token")
                or ""
            ).strip()
            if not token:
                logger.error("RugCheck login response missing token field.")
            return token
    except aiohttp.ClientError as e:
        logger.error("RugCheck login network error: %s", e)
        return ""

# ---------------------------------------------------------------------
# Login helper discovery & invocation (non-blocking) - subprocess-based
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
    """
    Run the external rugcheck_login helper (if present) in a subprocess with a
    short timeout and silent output; after the helper returns attempt to reload
    the .env so this process picks up new token values.
    """
    try:
        def _env_bool(name: str, default: bool = True) -> bool:
            v = os.getenv(name)
            if v is None:
                return default
            return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
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

        # Run helper silently. Use current env but set silent flag.
        subprocess.run(
            [sys.executable, str(helper), "--try-http"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "RUGCHECK_LOGIN_SILENT": "1"},
            timeout=timeout_seconds,
        )
        # After helper completes, reload env with override so new values are visible.
        _reload_env_override()
    except subprocess.TimeoutExpired:
        # helper timed out — ignore silently
        pass
    except Exception:
        # guard against any failure — don't raise to callers
        logger.debug("Silent login helper invocation failed", exc_info=True)
        try:
            _reload_env_override()
        except Exception:
            pass

# --------------------------
# Main entry the fetcher uses
# --------------------------
async def ensure_valid_rugcheck_headers(
    session: Optional[aiohttp.ClientSession] = None,
    force_refresh: bool = False,
) -> Dict[str, str]:
    """
    Returns headers; refreshes token via subprocess helper or via in-process login if needed.

    Flow (short summary):
      1) Load .env candidates into env (non-overriding).
      2) If a valid token exists and force_refresh=False -> return it.
      3) Attempt to run the external helper (short timeout) and reload .env override.
      4) If token now present -> return it.
      5) Else, if an async in-process login path is available, call it to obtain and persist a token.
      6) Return headers if token available; else {}.
    """
    # Ensure we loaded any candidate .env values (without overwriting shell env)
    _load_dotenv_into_env()

    tok = (os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT") or "").strip()
    if tok and not force_refresh and not _is_token_expired(tok):
        return {"Authorization": f"Bearer {tok}"}

    # 2) Attempt helper-based refresh (subprocess) and reload env override
    _run_silent_login_helper()
    # reload env override (in case helper wrote to a different candidate than first loaded)
    _reload_env_override()

    tok = (os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT") or "").strip()
    if tok and not _is_token_expired(tok):
        return {"Authorization": f"Bearer {tok}"}

    # 3) Fallback: attempt in-process async login (useful if private key present)
    login_url = os.getenv("RUGCHECK_LOGIN_URL", "https://api.rugcheck.xyz/auth/login/solana")
    close_me = False
    if session is None:
        session = aiohttp.ClientSession()
        close_me = True
    try:
        # Force-refresh via in-process login if available (this will persist via _persist_token)
        fresh = await _login_via_solana(session, login_url)
        if fresh:
            try:
                _persist_token(fresh)
            except Exception:
                logger.debug("Persist token failed after in-process login", exc_info=True)
            return {"Authorization": f"Bearer {fresh}"}
        return {}
    finally:
        if close_me:
            try:
                await session.close()
            except Exception:
                pass