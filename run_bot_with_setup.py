# run_bot_with_setup.py  — Windows-only launcher: safe .env seeding + AppData-first env loading + Rugcheck token compat
#
# Notes:
# - This variant is intentionally **Windows-only**. It exits immediately on non-Windows platforms.
# - Accepts constants.* as either Path objects OR callables returning Path.
# - Skips .env "migration": we keep the canonical .env in AppData (~\AppData\Local\SOLOTradingBot).
# - Sanitizes .env before loading (strips inline comments outside quotes).

from __future__ import annotations

import os
import sys
import subprocess
import logging
import re
import yaml
import json
import textwrap
import time
import shutil
import base64
from pathlib import Path
from typing import Sequence, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # Provide Keypair to the type checker (no runtime import)
    # solders may be unavailable at import-time in some environments, so guard it.
    from solders.keypair import Keypair

import signal
import psutil
from logging.handlers import RotatingFileHandler
try:
    from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
except Exception:  # pragma: no cover
    def retry(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap
    def stop_after_attempt(n):  # noqa: N802
        return None
    def wait_exponential(**kw):  # noqa: N802
        return None


# --- Teach PYTHONPATH (so local repo imports work out-of-the-box) ---
# Place this block BEFORE any 'from solana_trading_bot_bundle ...' imports.

# lightweight, idempotent launcher logger (safe even if you set one later)
logger = logging.getLogger("TradingBotLauncher")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(os.environ.get("LAUNCHER_LOGLEVEL", "INFO").upper())

try:
    here = Path(__file__).resolve().parent
    # Search for the project root that actually contains the package folder
    candidates = [
        here,               # repo root == this folder
        here.parent,        # repo root is one level up
    ]
    pkg_root: Optional[Path] = None
    for c in candidates:
        if (c / "solana_trading_bot_bundle").is_dir():
            pkg_root = c
            break

    if pkg_root:
        # Make imports work in THIS process…
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))

        # …and in ALL child processes (Streamlit, bot subprocess, etc.)
        existing = os.environ.get("PYTHONPATH", "")
        parts = [str(pkg_root)] + ([p for p in existing.split(os.pathsep) if p] if existing else [])
        # de-duplicate while preserving order
        dedup = []
        for p in parts:
            if p and p not in dedup:
                dedup.append(p)
        os.environ["PYTHONPATH"] = os.pathsep.join(dedup)

        logger.info("Bootstrapped PYTHONPATH with project root: %s", pkg_root)
        logger.debug("Effective PYTHONPATH: %s", os.environ.get("PYTHONPATH", ""))

        # sanity check: confirm the package is available without importing it.
        # Use find_spec to avoid executing package/module-level code prematurely.
        try:
            import importlib.util as _ilutil
            spec = _ilutil.find_spec("solana_trading_bot_bundle")
            if spec:
                logger.info("solana_trading_bot_bundle package available (spec.origin=%s)", getattr(spec, "origin", "?"))
            else:
                logger.warning(
                    "solana_trading_bot_bundle not found under %s or %s; imports may fail until PYTHONPATH is set.",
                    here, here.parent
                )
        except Exception as ie:
            logger.warning("Import availability check failed: %s", ie)
    else:
        logger.warning(
            "Could not find 'solana_trading_bot_bundle' under %s or %s; "
            "imports may fail until PYTHONPATH is set.",
            here, here.parent
        )
except Exception as e:
    logger.warning("PYTHONPATH bootstrap skipped due to error: %s", e)


# --------------------------------------------------------------------------------------
# Constants import (Path OR callable(Path) tolerant)
# --------------------------------------------------------------------------------------
from solana_trading_bot_bundle.common.constants import (
    APP_NAME, local_appdata_dir, appdata_dir, logs_dir, data_dir,
    config_path, env_path, db_path, token_cache_path, ensure_app_dirs, prefer_appdata_file
)

def _path_of(x) -> Path:
    """Return a Path for either a Path or a function that returns a Path/string."""
    try:
        if callable(x):
            v = x()
        else:
            v = x
        return v if isinstance(v, Path) else Path(str(v))
    except Exception:
        # very defensive fallback
        return Path(str(x))

# Normalize names once so we never call .name on functions:
_CONFIG_PATH = _path_of(config_path)
_ENV_PATH    = _path_of(env_path)
_DB_PATH     = _path_of(db_path)
_CACHE_PATH  = _path_of(token_cache_path)

CONFIG_BASENAME = _CONFIG_PATH.name if _CONFIG_PATH.name else "config.yaml"
ENV_BASENAME    = _ENV_PATH.name if _ENV_PATH.name else ".env"
DB_BASENAME     = _DB_PATH.name if _DB_PATH.name else "tokens.db"
CACHE_BASENAME  = _CACHE_PATH.name if _CACHE_PATH.name else "token_cache.json"

# =====================================================================================
# App constants and globals
# =====================================================================================

# Streamlit / Python env defaults (safe no-ops if already set)
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "0")
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")  # backstop; CLI also sets this
os.environ.setdefault("PYTHONWARNINGS", "ignore")

APP_NAME = os.environ.get("APP_NAME", "SOLOTradingBot")

IS_WINDOWS = sys.platform.startswith("win")
CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0

# ---- Windows-only guard (this launcher is for Windows variants only)
if not IS_WINDOWS:
    logger.error("This launcher is Windows-only. Detected platform: %s", sys.platform)
    print("This launcher is Windows-only. Please run the macOS/Linux launcher script on this OS.", file=sys.stderr)
    sys.exit(1)

# Allow override while keeping an int; used when we pick a port for Streamlit
DEFAULT_STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8085"))

# Single-instance lock handle (keep alive on POSIX; assigned in _setup_single_instance)
_LOCK_FD = None

# Resolve base directory: frozen EXE vs. script
BASE_DIR = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent
)
HERE = BASE_DIR

# Prefer a single cross-platform requirements.txt (allow override via env)
REQ_FILE = Path(os.environ.get("REQUIREMENTS_FILE") or (HERE / "requirements.txt")).resolve()

# -----------------------------
# Config.yaml resolution (NEW)
# -----------------------------
CONFIG_SOURCE = "unknown"

def _expand_path(p: str) -> str:
    p = os.path.expandvars(p)
    p = os.path.expanduser(p)
    return str(Path(p))

def _package_dir_for(mod_name: str) -> Path | None:
    """Locate the installed package directory for `mod_name`."""
    try:
        import importlib.util as _ilu
        spec = _ilu.find_spec(mod_name)
        if not spec or not spec.origin:
            return None
        return Path(spec.origin).resolve().parent
    except Exception:
        return None

def _get_roaming_dir() -> Path:
    """Secondary per-user roaming dir – used for seeding config/env for GUI convenience."""
    # Windows-only code path will always take this branch
    return Path(os.getenv("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / APP_NAME

def _get_appdata_dir() -> Path:
    """Resolve Windows per-user appdata dir and set global paths."""
    global APPDATA_DIR, ENV_PATH, STOP_FLAG_PATH, LOG_PATH
    APPDATA_DIR = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / APP_NAME
    APPDATA_DIR.mkdir(parents=True, exist_ok=True)

    ENV_PATH = APPDATA_DIR / ENV_BASENAME
    STOP_FLAG_PATH = APPDATA_DIR / "bot_stop_flag.txt"
    LOG_PATH = APPDATA_DIR / "launcher.log"

    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.touch(exist_ok=True)
    except Exception as e:
        print(f"Failed to create or write to {LOG_PATH}: {e}", file=sys.stderr)
        raise

    # No chmod on Windows
    return APPDATA_DIR

def _resolve_config_yaml() -> Path:
    """
    Prefer config.yaml from the *package folder* (solana_trading_bot_bundle) so
    beta testers get the shipped defaults. Fallbacks:
      1) env override CONFIG_FILE / SOLO_CONFIG
      2) package_dir/config.yaml or package_dir/config/config.yaml
      3) next to this script or PyInstaller _MEIPASS
      4) roaming/appdata copies (mainly for the GUI)
    """
    global CONFIG_SOURCE

    # 0) explicit env override
    env_override = os.getenv("CONFIG_FILE") or os.getenv("SOLO_CONFIG")
    if env_override:
        p = Path(_expand_path(env_override))
        if p.exists():
            CONFIG_SOURCE = "env"
            return p

    # 1) package folder
    pkg_dir = _package_dir_for("solana_trading_bot_bundle")
    if pkg_dir:
        for cand in (
            pkg_dir / CONFIG_BASENAME,
            pkg_dir / "config" / CONFIG_BASENAME,
        ):
            if cand.exists():
                CONFIG_SOURCE = "package"
                return cand

    # 2) next to script / frozen dir
    for cand, src in (
        (HERE / CONFIG_BASENAME, "here"),
        (Path(__file__).with_name(CONFIG_BASENAME), "__file__"),
        (Path(getattr(sys, "_MEIPASS", HERE)) / CONFIG_BASENAME, "_MEIPASS"),
    ):
        try:
            if cand.exists():
                CONFIG_SOURCE = src
                return cand
        except Exception:
            pass

    # 3) roaming / appdata (GUI-friendly copies)
    try:
        roaming = _get_roaming_dir() / CONFIG_BASENAME
        if roaming.exists():
            CONFIG_SOURCE = "roaming"
            return roaming
    except Exception:
        pass

    try:
        appd = _get_appdata_dir() / CONFIG_BASENAME
        if appd.exists():
            CONFIG_SOURCE = "appdata"
            return appd
    except Exception:
        pass

    # 4) final fallback — where we are
    CONFIG_SOURCE = "fallback(here)"
    return HERE / CONFIG_BASENAME

# Where we expect to read config.yaml from (prefer package)
CONFIG_YAML = _resolve_config_yaml()
print(f"[CFG] Using config.yaml at: {CONFIG_YAML} (source={CONFIG_SOURCE})")

APPDATA_DIR: Path | None = None
ENV_PATH: Path | None = None
STOP_FLAG_PATH: Path | None = None
LOG_PATH: Path | None = None

logger = logging.getLogger("TradingBotLauncher")
SELECTED_PORT: int | None = None

# Whether to nuke ALL python processes on exit (0/1)
KILL_ALL_PY_FALLBACK = str(os.getenv("KILL_ALL_PY_ON_EXIT", "0")).lower() in ("1", "true", "yes")

# =====================================================================================
# Child-process registry (for graceful shutdown)
# =====================================================================================
_CHILD_PROCS: list[subprocess.Popen] = []

def _register_child(proc: subprocess.Popen | None):
    if proc and proc.pid:
        _CHILD_PROCS.append(proc)
        logger.info("Registered child process PID %s", proc.pid)
    return proc

def _terminate_children(reason: str = "shutdown"):
    for proc in list(_CHILD_PROCS):
        if not proc:
            continue
        try:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    logger.info("Sent terminate to PID %s (%s)", proc.pid, reason)
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        proc.kill()
                        logger.warning("Forced kill of PID %s (%s)", proc.pid, reason)
                    except Exception:
                        pass
        except Exception as e:
            logger.error("Failed to terminate PID %s: %s", getattr(proc, "pid", "?"), e)
        finally:
            try:
                _CHILD_PROCS.remove(proc)
            except ValueError:
                pass

# =====================================================================================
# Paths / AppData bootstrap
# =====================================================================================
# Helper that returns existing candidate files (de-duplicated & resolved)
def _unique_existing_candidates(paths: list[Path]) -> list[Path]:
    seen = set()
    out: list[Path] = []
    for p in paths:
        try:
            if not p:
                continue
            p = Path(p)
            if p.exists() and p.is_file():
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(p)
        except Exception:
            # If any candidate is unreadable/invalid, skip it quietly
            continue
    return out

def _seed_file_if_missing(dest: Path, candidates: list[Path], label: str) -> bool:
    """
    Ensure `dest` exists by copying the first existing file found in `candidates`.
    Returns True if dest now exists (either already did or was seeded successfully).

    - dest: Path to create/seed (file)
    - candidates: list[Path] possible source files to copy from (in preferred order)
    - label: human-friendly label used in logging (e.g. "config.yaml" or ".env")
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            logger.debug("Seed target already exists: %s", dest)
            return True

        for src in _unique_existing_candidates(candidates):
            try:
                shutil.copy2(src, dest)
                logger.info("Seeded %s to %s from %s", label, dest, src)
                return True
            except Exception as e:
                logger.warning("Failed to seed %s from %s -> %s: %s", label, src, dest, e)

        logger.warning("No candidate found to seed %s. Expected at: %s", label, dest)
        return dest.exists()
    except Exception as e:
        logger.warning("Seeding %s failed for %s: %s", label, dest, e)
        return dest.exists()

def _ensure_gui_config_and_env_seed():
    """
    Place a copy of config.yaml in a roaming location for the GUI to pick up.
    For .env: NEVER overwrite an existing file unless ENV_FORCE_OVERWRITE=1.
    Also skip entirely when SOLANABOT_NO_ENV_MIGRATE=1.
    """
    roaming_dir = _get_roaming_dir()
    try:
        roaming_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Failed to create roaming dir %s: %s", roaming_dir, e)

    # --- config.yaml ---
    dest_config = roaming_dir / CONFIG_BASENAME
    # Seed from the same path the launcher *actually* uses (package-preferred)
    candidates = [
        CONFIG_YAML,                          # ← primary (package-preferred)
        HERE / CONFIG_BASENAME,               # local
        Path(__file__).with_name(CONFIG_BASENAME),
        Path(getattr(sys, "_MEIPASS", HERE)) / CONFIG_BASENAME,
    ]
    _seed_file_if_missing(dest_config, candidates, "config.yaml")

    # --- .env (safe seeding) ---
    dest_env = roaming_dir / ENV_BASENAME
    if os.getenv('SOLANABOT_NO_ENV_MIGRATE', '0') == '1':
        logger.info("Skipping .env seeding to roaming (non-migrating mode)")
        return

    if dest_env.exists() and os.getenv("ENV_FORCE_OVERWRITE", "0") != "1":
        logger.info("Roaming .env already exists; not overwriting.")
        return

    env_candidates: list[Path] = []
    if ENV_PATH:
        env_candidates.append(ENV_PATH)
    env_candidates.extend([
        HERE / ENV_BASENAME,
        Path(__file__).with_name(ENV_BASENAME),
        Path(getattr(sys, "_MEIPASS", HERE)) / ENV_BASENAME,
    ])
    _seed_file_if_missing(dest_env, env_candidates, ".env")

# =====================================================================================
# NEW: .env sanitizer (inline fallback)
# =====================================================================================
def _sanitize_env_file(env_file: Path):
    """
    Sanitize the .env by stripping inline comments while preserving quoted values.
    If tools/sanitize_env.py exists, use it; otherwise do it inline.
    """
    try:
        if not env_file or not env_file.exists():
            logger.info("Sanitizer skipped: %s does not exist", env_file)
            return

        script = HERE / "tools" / "sanitize_env.py"
        if script.exists():
            res = subprocess.run(
                [sys.executable, str(script), str(env_file)],
                capture_output=True, text=True, check=False, creationflags=CREATE_NO_WINDOW
            )
            if res.returncode == 0:
                logger.info("Sanitized .env via script: %s", env_file)
            else:
                logger.warning("Sanitizer script exited with %s: %s",
                               res.returncode, (res.stderr or res.stdout or "").strip())
            return

        def _strip_inline_comment(line: str) -> str:
            s = line.strip()
            if not s or s.startswith("#"):
                return line
            out = []
            in_str = False
            q = None
            for ch in line:
                if ch in ("'", '"'):
                    if not in_str:
                        in_str, q = True, ch
                    elif q == ch:
                        in_str, q = False, None
                    out.append(ch)
                elif ch == "#" and not in_str:
                    break
                else:
                    out.append(ch)
            return "".join(out).rstrip()

        raw = env_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        cleaned = [_strip_inline_comment(l) for l in raw]
        env_file.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
        logger.info("Sanitized .env inline: %s", env_file)

    except Exception as e:
        logger.warning("Sanitizer failed (non-fatal): %s", e)

# =====================================================================================
# One-time credentials bootstrap (persists to AppData .env; prompts only once)
# =====================================================================================
from getpass import getpass

def _canonical_env_path_forced() -> Path:
    """Canonical AppData .env path without relying on globals."""
    # Windows-only variant
    base = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    return base / APP_NAME / ENV_BASENAME

def _read_env_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition("=")
                    out[k.strip()] = v.strip()
    except Exception:
        pass
    return out

def _write_env_map_file(path: Path, kv: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = _read_env_map(path)
    merged = dict(current)
    for k, v in kv.items():
        if k not in merged or merged[k] == "":
            merged[k] = v
    content = "\n".join(f"{k}={merged[k]}" for k in sorted(merged)) + "\n"
    if (not path.exists()) or path.read_text(encoding="utf-8", errors="ignore") != content:
        path.write_text(content, encoding="utf-8")

def ensure_one_time_credentials(required_keys: list[str] | None = None) -> Path:
    """
    Ensures the canonical AppData .env contains the required secrets.
    Prompts ONLY for missing/empty values on FIRST run.
    """
    from dotenv import load_dotenv

    target_env = ENV_PATH if ENV_PATH else _canonical_env_path_forced()
    target_env.parent.mkdir(parents=True, exist_ok=True)

    try:
        load_dotenv(dotenv_path=target_env, override=False)
    except Exception:
        pass

    required_keys = required_keys or [
        "SOLANA_PRIVATE_KEY",
        "BIRDEYE_API_KEY",
        "RUGCHECK_JWT_TOKEN",
    ]

    existing = _read_env_map(target_env)
    to_save: dict[str, str] = {}
    missing: list[str] = []

    for key in required_keys:
        val = os.getenv(key) or existing.get(key, "")
        if val:
            to_save[key] = val
        else:
            missing.append(key)

    if missing:
        print("First-run setup: please enter the following credentials (saved to AppData):")
        print(f"  {target_env}")
        for key in missing:
            prompt = key.replace("_", " ").title() + ": "
            val = getpass(prompt)  # masked input for secrets
            to_save[key] = (val or "").strip()

    _write_env_map_file(target_env, to_save)

    try:
        load_dotenv(dotenv_path=target_env, override=True)
    except Exception:
        pass

    return target_env

# =====================================================================================
# Logging setup
# =====================================================================================
logger.setLevel(logging.INFO)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(stream_h)

def _setup_logging():
    global logger, LOG_PATH
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
    try:
        file_h = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        file_h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_h)
        logger.info("Logging initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize RotatingFileHandler for %s: %s", LOG_PATH, e)
        raise

def _check_appdata_writable():
    try:
        test_file = APPDATA_DIR / ".test_write"
        test_file.write_text("test", encoding="utf-8")
        test_file.unlink()
    except Exception as e:
        logger.error("APPDATA_DIR %s is not writable: %s", APPDATA_DIR, e)
        raise PermissionError(f"APPDATA_DIR {APPDATA_DIR} is not writable")

# =====================================================================================
# Single-Instance Guard
# =====================================================================================
def _setup_single_instance():
    # Windows Mutex
    try:
        import ctypes
        from ctypes import wintypes
        _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _CreateMutexW = _kernel32.CreateMutexW
        _CreateMutexW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
        _CreateMutexW.restype = wintypes.HANDLE
        _ = _CreateMutexW(None, False, f"Global\\{APP_NAME}SingletonMutex")
        if ctypes.get_last_error() == 183:
            try:
                ctypes.windll.user32.MessageBoxW(0, f"{APP_NAME} is already running.", APP_NAME, 0x00000040)
            except Exception:
                logger.warning("Failed to show MessageBox, exiting silently.")
            sys.exit(0)
    except Exception as e:
        logger.warning("Windows mutex setup failed: %s, continuing without single-instance lock.", e)

# =====================================================================================
# Dependency Management
# =====================================================================================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _upgrade_build_tools():
    if getattr(sys, "frozen", False):
        logger.info("Skipping pip upgrade in frozen environment.")
        return
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Failed to upgrade build tools: %s", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _pip_install_requirements():
    if getattr(sys, "frozen", False):
        logger.info("Skipping pip install in frozen environment.")
        return
    if not REQ_FILE.exists():
        logger.warning("Requirements file not found at %s, skipping installation.", REQ_FILE)
        return
    logger.info("Installing dependencies from %s", REQ_FILE.name)
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE), "--default-timeout", "180"]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Failed to install requirements: %s", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)

# =====================================================================================
# Misc helpers
# =====================================================================================
def _ensure_parent_dir(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)

def _parse_dotenv_file(path: Path) -> dict:
    data: dict[str, str] = {}
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip().strip('"')
    return data

def _write_env_map(path: Path, mapping: dict, header_comment: str | None = None):
    """Write mapping as KEY="VALUE" pairs, overwriting the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if header_comment:
        for ln in header_comment.strip("\n").splitlines():
            lines.append(f"# {ln}")
        lines.append("")
    for k, v in mapping.items():
        v_clean = str(v).replace("\n", "").replace("\r", "").replace('"', '\\"')
        lines.append(f'{k}="{v_clean}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # No chmod on Windows

def _has_tty() -> bool:
    try:
        return bool(sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty())
    except Exception:
        return False

def _powershell_prompt(prompt: str) -> str:
    cmd = [
        "powershell", "-NoProfile", "-Command",
        f"$v = Read-Host '{prompt}'; "
        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
        "Write-Output $v"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return (res.stdout or "").strip()
    except Exception:
        pass
    return ""

def _open_notepad_for_env(env_path: Path, missing_keys: list[str]):
    header = textwrap.dedent(f"""
    First-run setup — fill in the missing values below, then save and close Notepad.
    Required keys (leave no blanks): {', '.join(missing_keys)}
    Tips:
      • RUGCHECK_JWT_TOKEN — paste your Rugcheck JWT (starts with eyJ...)
      • RUGCHECK_JWT       — supported too (alias)
      • SOLANA_PRIVATE_KEY — base58, or a JSON array of 64 bytes
      • BIRDEYE_API_KEY    — your Birdeye API key
    Keep KEY="VALUE" format, one per line. Example:
      RUGCHECK_JWT_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    """).strip("\n")
    current = _parse_dotenv_file(env_path)
    for k in missing_keys:
        current.setdefault(k, "")
    _write_env_map(env_path, current, header_comment=header)
    try:
        subprocess.run(["notepad.exe", str(env_path)], check=False)
        time.sleep(0.2)
    except Exception:
        pass

# =====================================================================================
# JWT / Rugcheck helpers (optional auto-login)
# =====================================================================================
def _base64url_decode_noverify(seg: str) -> bytes:
    seg_b = seg.encode() if isinstance(seg, str) else seg
    pad = b'=' * ((4 - (len(seg_b) % 4)) % 4)
    return base64.urlsafe_b64decode(seg_b + pad)

def _jwt_exp_unix(jwt_token: str) -> int | None:
    try:
        parts = jwt_token.split(".")
        if len(parts) != 3:
            return None
        payload = json.loads(_base64url_decode_noverify(parts[1]).decode("utf-8"))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None

def _is_token_still_valid(jwt_token: str, leeway_seconds: int = 300) -> bool:
    exp = _jwt_exp_unix(jwt_token)
    if not exp:
        return False
    now = int(time.time())
    return (exp - now) > leeway_seconds

def _load_wallet_keypair_from_env() -> "Keypair":
    pk = os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("WALLET_PRIVATE_KEY") or ""
    if not pk:
        raise ValueError("SOLANA_PRIVATE_KEY (or WALLET_PRIVATE_KEY) is required to auto-login Rugcheck.")
    pk = pk.strip()
    try:
        from solders.keypair import Keypair  # local import for runtime
        return Keypair.from_base58_string(pk)
    except Exception:
        pass
    try:
        arr = json.loads(pk)
        if isinstance(arr, list) and len(arr) == 64 and all(isinstance(x, int) for x in arr):
            from solders.keypair import Keypair
            return Keypair.from_bytes(bytes(arr))
    except Exception:
        pass
    raise ValueError("Private key format not recognized (expect base58 string or JSON array of 64 bytes).")

def _rugcheck_auto_login() -> str | None:
    try:
        from solders.keypair import Keypair  # noqa: F401
        import base58  # type: ignore
        import requests  # type: ignore
    except ImportError:
        logger.warning("Missing dependencies for Rugcheck auto-login (install: requests, solders, base58). Manual RUGCHECK_JWT_TOKEN entry required.")
        return None

    try:
        wallet = _load_wallet_keypair_from_env()
        message_data = {
            "message": "Sign-in to Rugcheck.xyz",
            "timestamp": int(time.time() * 1000),
            "publicKey": str(wallet.pubkey()),
        }
        message_json = json.dumps(message_data, separators=(',', ':')).encode("utf-8")
        signature = wallet.sign_message(message_json)
        signature_base58 = str(signature)

        import base58 as _b58
        signature_data = list(_b58.b58decode(signature_base58))

        payload = {
            "signature": {"data": signature_data, "type": "ed25519"},
            "wallet": str(wallet.pubkey()),
            "message": message_data,
        }

        import requests as _requests
        resp = _requests.post(
            "https://api.rugcheck.xyz/auth/login/solana",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("token") or data.get("jwt") or data.get("access_token")
            if token:
                return token
            logger.warning("Rugcheck login succeeded but token field not found in response.")
            return None
        else:
            logger.warning("Rugcheck login failed: %s %s", resp.status_code, resp.text[:200])
            return None
    except Exception as e:
        logger.warning("Rugcheck auto-login error: %s", e)
        return None

def _write_env_key(key: str, value: str):
    cur = _parse_dotenv_file(ENV_PATH)
    cur[key] = value
    _write_env_map(ENV_PATH, cur, header_comment="Application configuration")
    os.environ[key] = value

# --- Keyring helpers (optional) ---
try:
    import keyring as __kr  # optional
    __HAVE_KEYRING = True
except Exception:
    __kr = None
    __HAVE_KEYRING = False

def _store_sensitive_key(key: str, value: str):
    """Persist secret both to .env and (if available) the OS keyring."""
    try:
        _write_env_key(key, value)
    except Exception:
        pass
    try:
        if __HAVE_KEYRING:
            __kr.set_password(APP_NAME, key, value)
    except Exception:
        pass

def _retrieve_sensitive_key(key: str) -> str | None:
    if __HAVE_KEYRING:
        try:
            return __kr.get_password(APP_NAME, key)
        except Exception:
            return None
    return None

# =====================================================================================
# Config Validation
# =====================================================================================
def _validate_config_yaml():
    if not CONFIG_YAML.exists():
        logger.error("Configuration file not found at %s", CONFIG_YAML)
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_YAML}")
    try:
        with CONFIG_YAML.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        required_keys = ["wallet", "solana", "bot", "discovery"]
        missing = [k for k in required_keys if k not in config]
        if missing:
            logger.error("Config file %s missing required keys: %s", CONFIG_YAML, missing)
            raise ValueError(f"Config file missing required keys: {missing}")

        if not config.get("wallet", {}).get("private_key_env"):
            logger.error("Config file %s missing wallet.private_key_env", CONFIG_YAML)
            raise ValueError("Config file missing wallet.private_key_env")

        if not config.get("solana", {}).get("rpc_endpoint"):
            logger.error("Config file %s missing solana.rpc_endpoint", CONFIG_YAML)
            raise ValueError("Config file missing solana.rpc_endpoint")

        # Expand any relative paths into AppData
        try:
            if "database" in config:
                if isinstance(config["database"].get("path"), str):
                    db_path_s = _expand_path(config["database"]["path"])
                    if not Path(db_path_s).is_absolute():
                        db_path_s = str(APPDATA_DIR / db_path_s)
                    _ensure_parent_dir(db_path_s)
                    config["database"]["path"] = db_path_s

                if isinstance(config["database"].get("token_cache_path"), str):
                    cache_path_s = _expand_path(config["database"]["token_cache_path"])
                    if not Path(cache_path_s).is_absolute():
                        cache_path_s = str(APPDATA_DIR / cache_path_s)
                    _ensure_parent_dir(cache_path_s)
                    config["database"]["token_cache_path"] = cache_path_s

            if "logging" in config and isinstance(config["logging"].get("log_file_path"), str):
                log_path_s = _expand_path(config["logging"]["log_file_path"])
                if not Path(log_path_s).is_absolute():
                    log_path_s = str(APPDATA_DIR / log_path_s)
                _ensure_parent_dir(log_path_s)
                config["logging"]["log_file_path"] = log_path_s

        except Exception as e:
            logger.warning("Path expansion failed: %s", e)

        logger.info("Validated config.yaml at %s (source=%s)", CONFIG_YAML, CONFIG_SOURCE)
        return config

    except Exception as e:
        logger.error("Failed to validate config.yaml: %s", e)
        raise

# =====================================================================================
# First-run Migration (move legacy DB/cache/logs)
# =====================================================================================
def _first_run_migrate_data(config: dict):
    try:
        db_target = Path(config.get("database", {}).get("path", "") or "")
        cache_target = Path(config.get("database", {}).get("token_cache_path", "") or "")

        # Look for legacy files by filename in both HERE and APPDATA_DIR
        legacy_db_candidates = [HERE / DB_BASENAME, APPDATA_DIR / DB_BASENAME]
        legacy_cache_candidates = [HERE / CACHE_BASENAME, APPDATA_DIR / CACHE_BASENAME]

        if db_target and not db_target.exists():
            for cand in legacy_db_candidates:
                if cand.exists() and cand.is_file():
                    try:
                        _ensure_parent_dir(str(db_target))
                        shutil.move(str(cand), str(db_target))
                        logger.info("Migrated legacy database from %s -> %s", cand, db_target)
                        break
                    except Exception as e:
                        logger.warning("Failed to migrate DB %s -> %s: %s", cand, db_target, e)

        if cache_target and not cache_target.exists():
            for cand in legacy_cache_candidates:
                if cand.exists() and cand.is_file():
                    try:
                        _ensure_parent_dir(str(cache_target))
                        shutil.move(str(cand), str(cache_target))
                        logger.info("Migrated legacy token cache from %s -> %s", cand, cache_target)
                        break
                    except Exception as e:
                        logger.warning("Failed to migrate cache %s -> %s: %s", cand, cache_target, e)

        if db_target and not db_target.exists():
            try:
                _ensure_parent_dir(str(db_target))
                Path(db_target).touch()
                logger.info("Initialized new database file at %s", db_target)
            except Exception as e:
                logger.warning("Couldn't create DB file at %s: %s", db_target, e)

        if cache_target and not cache_target.exists():
            try:
                _ensure_parent_dir(str(cache_target))
                Path(cache_target).touch()
                logger.info("Initialized new token cache at %s", cache_target)
            except Exception as e:
                logger.warning("Couldn't create token cache at %s: %s", cache_target, e)
    except Exception as e:
        logger.warning("First-run migration skipped due to error: %s", e)

# =====================================================================================
# .env Handling
# =====================================================================================
def _ensure_dotenv_available() -> bool:
    try:
        from dotenv import load_dotenv, set_key  # noqa: F401
        return True
    except ImportError:
        if not getattr(sys, "frozen", False):
            _upgrade_build_tools()
            subprocess.run([sys.executable, "-m", "pip", "install", "python-dotenv"], check=False)
            try:
                from dotenv import load_dotenv, set_key  # noqa: F401
                return True
            except ImportError:
                logger.warning("Failed to install python-dotenv.")
                return False
        logger.warning("python-dotenv not available in frozen environment.")
        return False

def _validate_env_syntax():
    if not ENV_PATH.exists():
        return
    try:
        with ENV_PATH.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    logger.warning("Invalid .env syntax at line %d: %s", line_num, line)
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    logger.warning("Empty key in .env at line %d: %s", line_num, line)
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                if not value:
                    logger.warning("Empty value for %s in .env at line %d", key, line_num)
    except Exception as e:
        logger.error("Failed to validate .env syntax: %s", e)

def _load_env_values() -> dict:
    ok = _ensure_dotenv_available()
    if ok:
        try:
            from dotenv import load_dotenv
            if ENV_PATH.exists():
                _validate_env_syntax()
                load_dotenv(dotenv_path=ENV_PATH, override=False)
            else:
                load_dotenv(override=False)
        except Exception as e:
            logger.warning("Failed to load .env: %s", e)
    sol_pk = os.getenv("SOLANA_PRIVATE_KEY") or ""
    wal_pk = os.getenv("WALLET_PRIVATE_KEY") or ""
    active = sol_pk or wal_pk
    return {
        "SOLANA_PRIVATE_KEY": sol_pk,
        "WALLET_PRIVATE_KEY": wal_pk,
        "ACTIVE_PRIVATE_KEY": active,
        "BIRDEYE_API_KEY": os.getenv("BIRDEYE_API_KEY") or "",
        "RUGCHECK_JWT_TOKEN": os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT") or "",
        "SOLANA_RPC_URL": os.getenv("SOLANA_RPC_URL") or "https://api.mainnet-beta.solana.com",
    }

def _validate_key_format(key: str, key_name: str) -> bool:
    if key_name in ["SOLANA_PRIVATE_KEY", "WALLET_PRIVATE_KEY"]:
        if not key:
            return False
        try:
            if re.match(r"^[1-9A-HJ-NP-Za-km-z]{44,88}$", key):
                return True
            arr = json.loads(key)
            return isinstance(arr, list) and len(arr) == 64 and all(isinstance(x, int) and 0 <= x <= 255 for x in arr)
        except json.JSONDecodeError:
            return False
    elif key_name in ["BIRDEYE_API_KEY", "RUGCHECK_JWT_TOKEN"]:
        return bool(key and len(key.strip()) > 10)
    elif key_name == "SOLANA_RPC_URL":
        return bool(key and key.startswith("https://"))
    return True

def _write_env(values: dict):
    """Merge provided values into .env (valid values only) and write file."""
    existing = _parse_dotenv_file(ENV_PATH)
    for k, v in values.items():
        if v is None or not _validate_key_format(v, k):
            logger.warning("Invalid or missing value for %s, skipping.", k)
            continue
        existing[k] = v
    _write_env_map(ENV_PATH, existing, header_comment="Application configuration")

def _ensure_keys():
    """
    Ensure required env keys exist. Prompt if missing.
    Accepts RUGCHECK_JWT or RUGCHECK_JWT_TOKEN; persists as RUGCHECK_JWT_TOKEN.
    """
    try:
        ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        ENV_PATH.touch(exist_ok=True)
    except Exception as e:
        logger.warning("Unable to create canonical .env at %s: %s", ENV_PATH, e)

    exe_dir_env = HERE / ENV_BASENAME
    if exe_dir_env.exists():
        logger.info("Found local .env at %s; non-migrating mode active — using canonical %s", exe_dir_env, ENV_PATH)

    file_map = _parse_dotenv_file(ENV_PATH)
    rug_env = file_map.get("RUGCHECK_JWT_TOKEN") or file_map.get("RUGCHECK_JWT") \
              or os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT") or ""

    env_vals = {
        "SOLANA_PRIVATE_KEY": file_map.get("SOLANA_PRIVATE_KEY") or os.getenv("SOLANA_PRIVATE_KEY") or "",
        "WALLET_PRIVATE_KEY": file_map.get("WALLET_PRIVATE_KEY") or os.getenv("WALLET_PRIVATE_KEY") or "",
        "BIRDEYE_API_KEY": file_map.get("BIRDEYE_API_KEY") or os.getenv("BIRDEYE_API_KEY") or "",
        "RUGCHECK_JWT_TOKEN": rug_env,
        "SOLANA_RPC_URL": file_map.get("SOLANA_RPC_URL") or os.getenv("SOLANA_RPC_URL") or "https://api.mainnet-beta.solana.com",
    }
    try:
        __jwt_from_kr = _retrieve_sensitive_key("RUGCHECK_JWT_TOKEN")
        if __jwt_from_kr:
            env_vals["RUGCHECK_JWT_TOKEN"] = __jwt_from_kr
            os.environ["RUGCHECK_JWT_TOKEN"] = __jwt_from_kr
    except Exception:
        pass

    try:
        tok = (env_vals.get("RUGCHECK_JWT_TOKEN") or "").strip()
        if not tok or not _is_token_still_valid(tok, leeway_seconds=300):
            fresh = _rugcheck_auto_login()
            if fresh:
                _store_sensitive_key("RUGCHECK_JWT_TOKEN", fresh)
                env_vals["RUGCHECK_JWT_TOKEN"] = fresh
    except Exception as e:
        logger.warning("Auto-refresh for Rugcheck token failed (will prompt if required): %s", e)

    need: list[str] = []
    active_pk = env_vals.get("SOLANA_PRIVATE_KEY") or env_vals.get("WALLET_PRIVATE_KEY")
    if not active_pk:
        need.append("SOLANA_PRIVATE_KEY")
    if not env_vals.get("BIRDEYE_API_KEY"):
        need.append("BIRDEYE_API_KEY")
    if not env_vals.get("RUGCHECK_JWT_TOKEN"):
        need.append("RUGCHECK_JWT_TOKEN")
    if not env_vals.get("SOLANA_RPC_URL") or not _validate_key_format(env_vals["SOLANA_RPC_URL"], "SOLANA_RPC_URL"):
        need.append("SOLANA_RPC_URL")

    if not need:
        os.environ.update({k: v for k, v in env_vals.items() if v})
        return

    if _has_tty():
        logger.info("First-run setup: missing keys %s. Prompting in console...", need)
        for key in need:
            label = key.replace("_", " ")
            is_secret = any(x in key.upper() for x in ("KEY", "TOKEN", "SECRET"))
            val = ""
            while not val:
                try:
                    if is_secret:
                        import getpass
                        val = getpass.getpass(f"Enter {label}: ").strip()
                    else:
                        val = input(f"{label}: ").strip()
                except Exception as e:
                    logger.warning("Console prompt failed for %s: %s", key, e)
                    break
            if val:
                env_vals[key] = val

        _write_env_map(ENV_PATH, env_vals, header_comment="Application configuration")
        os.environ.update({k: v for k, v in env_vals.items() if v})
        still = [k for k in need if not env_vals.get(k)]
        if still:
            raise RuntimeError(f"Missing required environment variables: {still}. Edit {ENV_PATH} and re-run.")
        return

    _open_notepad_for_env(ENV_PATH, need)
    env_vals.update(_parse_dotenv_file(ENV_PATH))
    os.environ.update({k: v for k, v in env_vals.items() if v})
    still = [k for k in need if not os.getenv(k)]
    if not still:
        return

    # PowerShell fallback (Windows)
    for key in still:
        entered = ""
        tries = 3
        while tries > 0 and not entered:
            entered = _powershell_prompt(f"Enter {key}")
            if not entered:
                tries -= 1
        if entered:
            env_vals[key] = entered

    _write_env_map(ENV_PATH, env_vals, header_comment="Application configuration")
    os.environ.update({k: v for k, v in env_vals.items() if v})
    final_missing = [k for k in need if not os.getenv(k)]
    if final_missing:
        raise RuntimeError(
            f"Missing required environment variables: {final_missing}. Please edit your .env at:\n{ENV_PATH}"
        )

def _reload_env_into_process():
    try:
        from dotenv import load_dotenv
        # Reload from AppData .env and allow it to override
        if ENV_PATH.exists():
            load_dotenv(dotenv_path=ENV_PATH, override=True)
        else:
            load_dotenv(override=True)
    except Exception as e:
        logger.warning("Failed to reload .env: %s", e)

def get_active_private_key() -> str:
    key = os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("WALLET_PRIVATE_KEY") or ""
    if not key or not _validate_key_format(key, "SOLANA_PRIVATE_KEY"):
        raise ValueError("No valid private key found. Set SOLANA_PRIVATE_KEY or WALLET_PRIVATE_KEY in your .env.")
    return key

# =====================================================================================
# Networking / Port utils
# =====================================================================================
def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) == 0

def _find_free_port(start_port: int, max_tries: int = 15) -> int:
    """Find the next available port starting at start_port (inclusive)."""
    port = start_port
    for _ in range(max_tries):
        if not _is_port_in_use(port):
            return port
        port += 1
    import socket as _s
    with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _pids_listening_on_port(port: int) -> set[int]:
    pids: set[int] = set()
    if IS_WINDOWS:
        try:
            out = subprocess.check_output(
                f'netstat -ano | findstr :{port}',
                shell=True,
                creationflags=CREATE_NO_WINDOW
            ).decode(errors="ignore")
        except subprocess.CalledProcessError:
            return pids

        self_pid = os.getpid()
        ancestor_pids = {self_pid}
        try:
            proc = psutil.Process(self_pid)
            while proc and proc.ppid() and proc.ppid() not in ancestor_pids:
                ancestor_pids.add(proc.ppid())
                proc = psutil.Process(proc.ppid())
        except Exception:
            pass

        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[-1].isdigit():
                if any(p.endswith(f":{port}") for p in parts):
                    pid = int(parts[-1])
                    if pid not in ancestor_pids:
                        pids.add(pid)
    return pids

def _kill_port_listeners(port: int) -> None:
    pids = _pids_listening_on_port(port)
    for pid in pids:
        try:
            subprocess.run(f'taskkill /PID {pid} /T /F', shell=True, check=True)
            logger.info("Killed PID %s on port %s", pid, port)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to kill PID %s on port %s: %s", pid, port, e)
            if IS_WINDOWS:
                try:
                    subprocess.run(f'powershell -Command "Start-Process taskkill -ArgumentList \'/PID {pid} /T /F\' -Verb RunAs"', shell=True)
                    logger.info("Admin taskkill succeeded for PID %s on port %s", pid, port)
                except subprocess.CalledProcessError as e2:
                    logger.error("Admin taskkill failed for PID %s on port %s: %s", pid, port, e2)

def _kill_processes_on_port(port: int):
    try:
        for p in psutil.process_iter(['pid', 'name']):
            try:
                conns = p.net_connections(kind='inet')
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            for c in conns:
                if getattr(c.laddr, "port", None) == port:
                    try:
                        p.terminate()
                        p.wait(timeout=3)
                        logger.info("Terminated lingering process (PID: %s) on port %s", p.pid, port)
                    except Exception:
                        try:
                            p.kill()
                            logger.warning("Killed lingering process (PID: %s) on port %s", p.pid, port)
                        except Exception:
                            pass
                    break
    except Exception as e:
        logger.debug("_kill_processes_on_port(%s) failed: %s", port, e)

def _graceful_stop(proc: subprocess.Popen) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            logger.info("Sent terminate to PID %s", proc.pid)
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
                logger.warning("Forced kill of PID %s", proc.pid)
            except Exception:
                pass
    except Exception as e:
        logger.error("Failed to stop PID %s: %s", proc.pid if proc else "?", e)

# =====================================================================================
# Cleanup Handler (registered at exit and for signals)
# =====================================================================================
streamlit_process: subprocess.Popen | None = None

def _safe_unlink(p: Path | None) -> None:
    if not p:
        return
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass

def _release_single_instance_lock():
    # Optional: only if your script defined LOCK_FILE / _LOCK_FD elsewhere
    try:
        lf = globals().get("LOCK_FILE", None)
        fd = globals().get("_LOCK_FD", None)
        if fd:
            try:
                os.close(fd)
            except Exception:
                pass
            globals()["_LOCK_FD"] = None
        if isinstance(lf, Path):
            _safe_unlink(lf)
    except Exception:
        pass

def cleanup():
    """Best-effort, idempotent shutdown."""
    global streamlit_process, SELECTED_PORT
    try:
        # 1) Clear any stop-flag file
        try:
            STOP_FLAG_PATH = globals().get("STOP_FLAG_PATH", None)
            if isinstance(STOP_FLAG_PATH, Path) and STOP_FLAG_PATH.exists():
                STOP_FLAG_PATH.unlink()
                logger.info("Removed stop flag at %s", STOP_FLAG_PATH)
        except Exception:
            pass

        # 2) Stop Streamlit subprocess if still running
        if streamlit_process and streamlit_process.poll() is None:
            try:
                logger.info("Terminating Streamlit process (PID: %s)", streamlit_process.pid)
                _graceful_stop(streamlit_process)  # your helper: terminate → wait → kill
            except Exception as e:
                logger.warning("Failed graceful stop for Streamlit: %s", e, exc_info=True)
        streamlit_process = None  # don’t reuse the handle

        # 3) Kill anything still listening on the selected port (only after stopping proc)
        try:
            if globals().get("SELECTED_PORT"):
                _kill_port_listeners(SELECTED_PORT)
                _kill_processes_on_port(SELECTED_PORT)
        except Exception as e:
            logger.debug("Port cleanup skipped/failed: %s", e)

        # 4) Reap any orphan children of this launcher (psutil)
        try:
            _terminate_children("finalize")
        except Exception as e:
            logger.debug("Child process cleanup failed: %s", e)

        # 5) Optional Windows-only nuclear fallback (disabled by default)
        try:
            if IS_WINDOWS and globals().get("KILL_ALL_PY_FALLBACK", False):
                logger.warning("Fallback enabled: terminating all python* processes (dangerous).")
                subprocess.run('taskkill /IM python.exe /F', shell=True, check=False)
                subprocess.run('taskkill /IM pythonw.exe /F', shell=True, check=False)
        except Exception as e:
            logger.debug("Fallback taskkill failed: %s", e)

    except Exception as e:
        logger.error("Cleanup failed: %s", e, exc_info=True)
    finally:
        # 6) Release single-instance lock if you created one
        _release_single_instance_lock()

def _signal_handler(sig, frame):
    try:
        logger.info("Received signal %s, initiating cleanup", sig)
    except Exception:
        pass
    print(f"Received signal {sig}, initiating cleanup", flush=True)
    try:
        cleanup()
    finally:
        # Exit code: 130 for SIGINT, 143 for SIGTERM, else 0
        if sig == getattr(signal, "SIGINT", None):
            sys.exit(130)
        if sig == getattr(signal, "SIGTERM", None):
            sys.exit(143)
        sys.exit(0)

# Register handlers
signal.signal(signal.SIGINT, _signal_handler)
try:
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    # Some environments (Windows / embedded) may not expose SIGTERM
    pass

# =====================================================================================
# Streamlit runner
# =====================================================================================
def _ensure_streamlit_available() -> None:
    try:
        import streamlit  # noqa: F401
        import streamlit as st  # for version
        logger.info("Streamlit version: %s", st.__version__)
    except Exception as e:
        logger.error("Streamlit is not installed in this environment: %s", e)
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=False)
        try:
            import streamlit as st  # noqa: F401
            logger.info("Streamlit installed, version: %s", st.__version__)
        except ImportError:
            logger.error("Failed to install Streamlit. Please install it manually: 'pip install streamlit'")
            sys.exit(1)

def _verify_gui_syntax(path: Path) -> None:
    try:
        import py_compile
        py_compile.compile(str(path), doraise=True)
        logger.info("Verified syntax of %s", path)
    except Exception as e:
        # fix formatting + include the exception
        logger.error("Syntax check failed for %s: %s", path, e, exc_info=True)
        sys.exit(1)

def _build_streamlit_cmd(extra_args: Sequence[str] | None = None) -> list[str]:
    args = [sys.executable, "-m", "streamlit", "run", str(HERE / "solana_bot_gui_corrected_working.py")]
    if extra_args:
        args.extend(extra_args)
    return args

def launch_gui():
    global streamlit_process, SELECTED_PORT
    GUI_FILE = HERE / "solana_bot_gui_corrected_working.py"
    if not GUI_FILE.exists():
        logger.error("GUI file not found at %s", GUI_FILE)
        print(f"Error: {GUI_FILE} not found")
        sys.exit(1)

    _ensure_streamlit_available()
    _verify_gui_syntax(GUI_FILE)

    port_env = os.getenv("STREAMLIT_PORT")
    SELECTED_PORT = int(port_env) if (port_env and port_env.isdigit()) else DEFAULT_STREAMLIT_PORT

    if _is_port_in_use(SELECTED_PORT):
        logger.info("Port %s in use; attempting to free it.", SELECTED_PORT)
        _kill_port_listeners(SELECTED_PORT)
        time.sleep(0.5)
        if _is_port_in_use(SELECTED_PORT):
            _kill_processes_on_port(SELECTED_PORT)
            time.sleep(0.3)

    if _is_port_in_use(SELECTED_PORT):
        try:
            new_port = _find_free_port(SELECTED_PORT + 1)
            logger.info("Port %s still busy; switching to free port %s", SELECTED_PORT, new_port)
            SELECTED_PORT = new_port
        except Exception as _e:
            logger.warning("Could not switch to a new port: %s", _e)

    extra = [
        "--browser.gatherUsageStats", "false",
        "--server.headless", "true",
        "--server.address", "localhost",
        "--server.fileWatcherType", "none",
    ]
    if SELECTED_PORT:
        extra += ["--server.port", str(SELECTED_PORT)]

    logger.info("Starting Streamlit app")
    print("Starting Streamlit app")

    try:
        cmd = _build_streamlit_cmd(extra)
        logger.info("Spawning: %s", " ".join(cmd))

        # >>> Ensure PYTHONPATH/env from this process is preserved in Streamlit (and its children)
        env = os.environ.copy()
        logger.info("PYTHONPATH for Streamlit: %s", env.get("PYTHONPATH", ""))

        streamlit_process = _register_child(subprocess.Popen(
            cmd,
            cwd=str(HERE),
            env=env,  # <<< pass through the environment
            stdout=sys.stdout,
            stderr=sys.stderr,
            stdin=None,
            creationflags=CREATE_NO_WINDOW if IS_WINDOWS else 0,
        ))

        logger.info(
            "Streamlit started (PID: %s) on http://localhost:%s",
            streamlit_process.pid,
            SELECTED_PORT if SELECTED_PORT else "<auto>"
        )
        print(f"Streamlit URL: http://localhost:{SELECTED_PORT if SELECTED_PORT else '<auto>'}")

        try:
            return_code = streamlit_process.wait()
            logger.info("Streamlit exit code: %s", return_code)
            if return_code != 0:
                print("Streamlit exited with errors. See logs for details.")
            sys.exit(return_code)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt: terminating Streamlit")
            _graceful_stop(streamlit_process)
            sys.exit(130)

    except Exception as e:
        logger.error("Failed to start Streamlit: %s", e, exc_info=True)
        print(f"Error: Failed to start Streamlit: {e}")
        cleanup()
        sys.exit(1)

# =====================================================================================
# Main
# =====================================================================================
def main():
    try:
        # Non-migrating mode: disable any .env moves/copies by default
        os.environ.setdefault('SOLANABOT_NO_ENV_MIGRATE', '1')

        _get_appdata_dir()
        ensure_app_dirs()
        _setup_logging()
        _check_appdata_writable()
        _setup_single_instance()

        os.chdir(HERE)

        config = _validate_config_yaml()
        _ensure_gui_config_and_env_seed()   # safe: never overwrites existing .env

        # Sanitize the canonical AppData .env BEFORE any dotenv loads
        try:
            _sanitize_env_file(ENV_PATH)
        except Exception as _e:
            logger.warning("Sanitizer step failed (continuing): %s", _e)

        _first_run_migrate_data(config)

        _upgrade_build_tools()
        _pip_install_requirements()

        # -------------------------------------------------------------------------
        # One-time credentials bootstrap:
        # Prompts ONLY if keys are missing; writes to AppData .env; never prompts again.
        # Also: sanitize & load the canonical .env, then reload with override so new keys win.
        # -------------------------------------------------------------------------
        try:
            from dotenv import load_dotenv

            # 0) Sanitize the canonical AppData .env BEFORE any loads (never logs secrets)
            try:
                _sanitize_env_file(ENV_PATH)
            except Exception as _e:
                logger.warning("Sanitizer step failed (continuing): %s", _e)

            # 1) Initial load (no override) so we can read any existing flags
            load_dotenv(dotenv_path=ENV_PATH, override=False)

            # persists credentials in the right place. Use globals().get() to avoid
            # direct-name references that static analyzers flag.
            kwargs: dict = {}
            _appdata_env_val = globals().get("APPDATA_ENV_PATH", None)
            if _appdata_env_val is not None:
                try:
                    # Support either a Path/string or a callable that returns one
                    candidate = _appdata_env_val() if callable(_appdata_env_val) else _appdata_env_val
                    kwargs["dest_path"] = str(candidate)
                except Exception:
                    # be conservative: fall back to str() of the raw value
                    try:
                        kwargs["dest_path"] = str(_appdata_env_val)
                    except Exception:
                        logger.debug("Could not coerce APPDATA_ENV_PATH to string; ignoring.", exc_info=True)

            # Optional: allow non-interactive mode via env (skip prompts in CI/packaged runs)
            if os.getenv("LAUNCHER_NONINTERACTIVE", ""):
                kwargs["noninteractive"] = True

            # Optional: allow non-interactive mode via env (skip prompts in CI/packaged runs)
            if os.environ.get("LAUNCHER_NONINTERACTIVE"):
                kwargs["noninteractive"] = True

            # 2) Ensure secrets exist (writes to AppData .env if missing)
            env_used = ensure_one_time_credentials(
                required_keys=[
                    "SOLANA_PRIVATE_KEY",
                    "BIRDEYE_API_KEY",
                    "RUGCHECK_JWT_TOKEN",
                ],
                **kwargs,  # safe: only adds if present
            )

            # Avoid logging secrets; only the path to the .env file
            logger.info("Using credentials file: %s", str(env_used))

            # 3) Reload with override so freshly written values DEFINITIVELY land in process env
            load_dotenv(dotenv_path=env_used, override=True)

            # 4) Safe defaults for Birdeye; .env can still override these
            os.environ.setdefault("BIRDEYE_ENABLE", "1")
            os.environ.setdefault("FORCE_DISABLE_BIRDEYE", "0")

            # (Optional but useful) sensible defaults for Dex fetch if user hasn’t set them
            os.environ.setdefault("DEX_PAGES", "30")
            os.environ.setdefault("DEX_PER_PAGE", "100")
            os.environ.setdefault("DEX_MAX", "2000")
            # Windows friendliness (the codepath checks this)
            os.environ.setdefault("DEX_FORCE_IPV4", "auto")

            # 5) Diagnostics that also make mismatches obvious in logs
            logger.info(
                "ENV loaded from %s | Birdeye: ENABLE=%s FORCE_DISABLE=%s KEY_PRESENT=%s | "
                "RPS=%s cycle_cap=%s run_cap=%s | Dex: DEX_PAGES=%s DEX_PER_PAGE=%s DEX_MAX=%s DEX_FORCE_IPV4=%s",
                str(env_used),
                os.getenv("BIRDEYE_ENABLE"),
                os.getenv("FORCE_DISABLE_BIRDEYE"),
                "yes" if os.getenv("BIRDEYE_API_KEY") else "no",
                os.getenv("BIRDEYE_RPS", "?"),
                os.getenv("BIRDEYE_MAX_CALLS_PER_CYCLE", "?"),
                os.getenv("BIRDEYE_MAX_CALLS_PER_RUN", "?"),
                os.getenv("DEX_PAGES", "?"),
                os.getenv("DEX_PER_PAGE", "?"),
                os.getenv("DEX_MAX", "?"),
                os.getenv("DEX_FORCE_IPV4", "?"),
            )
        except Exception as e:
            # Do not block the GUI if bootstrap fails; we’ll still reload whatever is available.
            logger.warning("Credentials bootstrap skipped/failed: %s", e)

        # Keep normalization/compat layer (should not prompt anymore)
        try:
            _ensure_keys()
        except Exception as e:
            logger.debug("Key normalization skipped: %s", e)

        # Optional: if your helper performs extra merging from file → process env
        # (safe to keep; it will be a no-op if already loaded by load_dotenv)
        try:
            _reload_env_into_process()
        except Exception as e:
            logger.debug("Reloading env into process failed (continuing): %s", e)

        # Soft-start Rugcheck auth (skip failures silently for non-technical testers)
        try:
            _ = get_active_private_key()  # validates and/or warms any wallet-dependent flows
        except Exception as e:
            logger.info("Skipping Rugcheck auto-login (no/invalid wallet key): %s", e)

        # NOTE: ensure launch_gui() (or any subprocess spawns) passes os.environ to children.
        # e.g., subprocess.Popen(cmd, env=os.environ.copy(), ...)
        launch_gui()
    except Exception as e:
        logger.error("Fatal error: %s", e)
        print(f"Fatal error: {e}")
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as e:
        logger.error("Launcher failed: %s", e)
        print(f"Launcher failed: {e}")
        cleanup()
        sys.exit(1)
