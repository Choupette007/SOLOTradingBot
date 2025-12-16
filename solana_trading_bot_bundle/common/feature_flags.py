# solana_trading_bot_bundle/common/feature_flags.py
import os

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def is_enabled_birdeye(cfg: dict) -> bool:
    # Hard env kill-switch wins
    if _env_bool("FORCE_DISABLE_BIRDEYE", False):
        return False
    # Primary env flag (your .env uses BIRDEYE_ENABLE)
    env_ok = _env_bool("BIRDEYE_ENABLE", False)
    # Config mirrors (both the explicit toggle and discovery cap)
    cfg_toggle = bool(cfg.get("sources", {}).get("birdeye_enabled", False))
    cap_ok = (cfg.get("discovery", {}).get("birdeye_max_tokens", 0) or 0) > 0
    # Enabled only if BOTH env and config want it (safer)
    return env_ok and cfg_toggle and cap_ok

def is_enabled_raydium(cfg: dict) -> bool:
    if _env_bool("FORCE_DISABLE_RAYDIUM", False):
        return False
    env_ok = _env_bool("RAYDIUM_ENABLE", False)
    cfg_toggle = bool(cfg.get("sources", {}).get("raydium_enabled", False))
    pairs_ok = (cfg.get("discovery", {}).get("raydium_max_pairs", 0) or 0) > 0
    page_ok  = (cfg.get("discovery", {}).get("raydium_page_size", 0) or 0) > 0
    return env_ok and cfg_toggle and pairs_ok and page_ok

def is_enabled_rugcheck(cfg: dict) -> bool:
    if _env_bool("FORCE_DISABLE_RUGCHECK", False):
        return False
    env_ok = _env_bool("RUGCHECK_ENABLE", False)
    cfg_toggle = bool(cfg.get("rugcheck", {}).get("enabled", False))
    return env_ok and cfg_toggle

def resolved_run_flags() -> dict:
    # unify & truthify what you print in logs
    def b(name, default=False):
        return _env_bool(name, default)
    DRY_RUN = b("DRY_RUN", True)
    DISABLE_SEND_TX = b("DISABLE_SEND_TX", True)
    JUPITER_EXECUTE = b("JUPITER_EXECUTE", False)
    JUPITER_QUOTE_ONLY = b("JUPITER_QUOTE_ONLY", True)
    SEND_TX = (not DISABLE_SEND_TX) and JUPITER_EXECUTE and (not JUPITER_QUOTE_ONLY)
    return {
        "dry_run": DRY_RUN,
        "simulate": True,      # your config default; keep if you have a different source
        "send_tx": SEND_TX,
        "env": {
            "DRY_RUN": DRY_RUN,
            "DISABLE_SEND_TX": DISABLE_SEND_TX,
            "JUPITER_EXECUTE": JUPITER_EXECUTE,
            "JUPITER_QUOTE_ONLY": JUPITER_QUOTE_ONLY,
        }
    }
