"""
Market/chain data utilities for the Solana trading bot.

This file contains defensive improvements to make mint/account validation robust
when running against a cluster that may not contain the same token mints
(e.g. running against devnet while discovery sources provide mainnet mints).

"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import socket
from typing import Any, Dict, Optional, Tuple, Union, Iterable

import aiohttp
from aiohttp import TCPConnector
from aiohttp.client_exceptions import (
    ClientConnectorError,
    ClientOSError,
    ServerDisconnectedError,
)
from cachetools import TTLCache

# solana-py
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment
from solana.rpc.types import TxOpts
from solana.rpc.core import RPCException as SolanaRPCException

# solders
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import Message
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from solana_trading_bot_bundle.common.constants import (
    APP_NAME,
    local_appdata_dir,
    appdata_dir,
    logs_dir,
    data_dir,
    config_path,
    env_path,
    db_path,
    token_cache_path,
    ensure_app_dirs,
    prefer_appdata_file,
)

from .utils_exec import (
    WHITELISTED_TOKENS,
    load_config,
    price_cache,
    token_account_existence_cache,
    token_balance_cache,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Fallback logger if helper not present
try:
    from .utils_exec import log_error_with_stacktrace  # type: ignore
except Exception:  # pragma: no cover
    def log_error_with_stacktrace(message: str, error: Exception) -> None:
        logging.getLogger("TradingBot").error("%s: %s", message, error, exc_info=True)

logger = logging.getLogger("TradingBot")

# ---------------------------------------------------------------------------
# Constants / caches
# ---------------------------------------------------------------------------

# Token-2022 program id (used by many new mints / pump.fun style tokens)
TOKEN_2022_PROGRAM_ID = Pubkey.from_string("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb")

async def get_token_program_id(mint: str, solana_client: AsyncClient) -> Pubkey:
    """
    Fetch the owner (program id) of a mint account. If unavailable/fails, return
    the legacy TOKEN_PROGRAM_ID as a conservative fallback.
    """
    try:
        if not _is_valid_pubkey_str(mint):
            return TOKEN_PROGRAM_ID
        mint_pub = Pubkey.from_string(mint)
        resp = await solana_client.get_account_info(mint_pub, commitment=Commitment("confirmed"))
        val = getattr(resp, "value", None)
        if not val:
            return TOKEN_PROGRAM_ID
        # Many RPC wrappers return 'owner' as a base58 string on the account value.
        owner = getattr(val, "owner", None)
        if owner:
            try:
                return Pubkey.from_string(str(owner))
            except Exception:
                # owner might already be a Pubkey-like, try bytes
                try:
                    return Pubkey.from_string(str(owner))
                except Exception:
                    return TOKEN_PROGRAM_ID
        # Fallback: try to inspect raw dicts (some clients return dict)
        if isinstance(val, dict):
            owner_str = val.get("owner")
            if owner_str:
                try:
                    return Pubkey.from_string(owner_str)
                except Exception:
                    pass
    except Exception:
        # Be conservative: don't escalate import-time or RPC parse errors
        try:
            logger.debug("get_token_program_id: could not determine owner for %s", mint, exc_info=True)
        except Exception:
            pass
    return TOKEN_PROGRAM_ID

# Program IDs (solders Pubkeys)
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
SYSTEM_PROGRAM_ID = Pubkey.from_string("11111111111111111111111111111111")
RENT_SYSVAR = Pubkey.from_string("SysvarRent111111111111111111111111111111111")  # added for ATA fallback

HTTP_HEADERS = {"Accept": "application/json", "User-Agent": "TradingBot/1.0"}
SHORT_TIMEOUT = aiohttp.ClientTimeout(total=15, sock_connect=5, sock_read=10)
MED_TIMEOUT = aiohttp.ClientTimeout(total=30, sock_connect=10, sock_read=20)

SOL_PRICE_CACHE_KEY = "sol_usd_price"

# Cache results of mint validation to avoid repeated RPC churn.
# TTL 10 minutes is a reasonable compromise.
mint_validation_cache: TTLCache = TTLCache(maxsize=2048, ttl=600)

# ---------------------------------------------------------------------------
# Networking helpers (dual-stack with IPv4 fallback)
# ---------------------------------------------------------------------------

def _make_connector(ipv4: bool = False) -> TCPConnector:
    """
    Dual-stack by default. If ipv4=True, force IPv4 (useful when IPv6 path is flaky).
    """
    family = socket.AF_INET if ipv4 else socket.AF_UNSPEC  # AF_UNSPEC lets OS pick v6+v4 (Happy Eyeballs)
    return TCPConnector(limit=100, ttl_dns_cache=15, family=family)

# broader "networky" exceptions that warrant trying IPv4-only
_NETWORKY_EXC = (
    ClientConnectorError,
    ClientOSError,
    ServerDisconnectedError,
    asyncio.TimeoutError,
    socket.gaierror,
    socket.timeout,
    ConnectionResetError,
    BrokenPipeError,
)

async def _fetch_text_with_ipv4_fallback(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    data: str | bytes | None = None,
    headers: dict | None = None,
    timeout: aiohttp.ClientTimeout = SHORT_TIMEOUT,
    session: aiohttp.ClientSession | None = None,
) -> tuple[int, str]:
    """
    Try on the provided session (dual-stack). If a networky error occurs,
    retry once with a temporary IPv4-only session. Returns (status, text).
    If JUP_FORCE_IPV4=1 is set, use IPv4-only immediately.
    """
    force_ipv4 = _env_truthy("JUP_FORCE_IPV4", "0")

    # If forced IPv4, skip straight to the IPv4 connector path.
    if force_ipv4:
        async with aiohttp.ClientSession(connector=_make_connector(ipv4=True), timeout=timeout) as s4:
            async with s4.request(method, url, params=params, data=data, headers=headers) as r4:
                return r4.status, await r4.text()

    # 1) Try using the caller's session (or a temp dual-stack one)
    try:
        if session is None:
            async with aiohttp.ClientSession(connector=_make_connector(), timeout=timeout) as s:
                async with s.request(method, url, params=params, data=data, headers=headers) as r:
                    return r.status, await r.text()
        else:
            async with session.request(method, url, params=params, data=data, headers=headers, timeout=timeout) as r:
                return r.status, await r.text()
    except _NETWORKY_EXC:
        # 2) Fallback: IPv4-only once
        async with aiohttp.ClientSession(connector=_make_connector(ipv4=True), timeout=timeout) as s4:
            async with s4.request(method, url, params=params, data=data, headers=headers) as r4:
                return r4.status, await r4.text()

# ---------------------------------------------------------------------------
# Token / account utilities
# ---------------------------------------------------------------------------
def _is_transient_rpc_error_text(msg: str) -> bool:
    """
    Heuristic to detect transient/simulation/preflight RPC errors that should NOT
    count toward persistent blacklisting. Covers common patterns returned by
    solana RPC libraries and simulation failures.
    """
    if not msg:
        return False
    m = str(msg).lower()
    # Common indicators of simulation / preflight / instruction/compute transient issues
    transient_indicators = (
        "transaction simulation failed",
        "preflight failure",
        "preflight",
        "simulate",
        "simulation failed",
        "notenoughaccountkeys",         # compacted variants
        "insufficient account keys",
        "not enough account keys",
        "instructionerror",
        "failed: insufficient account keys",
        "sendtransactionpreflightfailuremessage",
        # RPC providers sometimes include "Program <...> failed" with logs showing compute limit; treat as transient
        "program .* failed",           # keep generic — will be substring matched
    )
    for tok in transient_indicators:
        try:
            if tok in m:
                return True
        except Exception:
            continue
    return False

# ---------------------------------------------------------------------------
# Token / account utilities
# ---------------------------------------------------------------------------

def is_sufficient_liquidity(
    token_data: Optional[Dict],
    *,
    min_liquidity_usd: float = 1000.0,
    min_volume_24h_usd: float = 250.0,
) -> bool:
    """
    Best-effort check whether token_data indicates sufficient market depth to attempt a swap.

    Returns True when liquidity >= min_liquidity_usd OR volume_24h >= min_volume_24h_usd.
    Accepts token_data shapes from discovery/db (uses common keys and tries a few fallbacks).
    """
    if not token_data or not isinstance(token_data, dict):
        return False
    def _num_val(keys: Iterable[str]) -> float:
        for k in keys:
            v = token_data.get(k)
            try:
                if v is None:
                    continue
                fv = float(v)
                if fv and fv > 0:
                    return float(fv)
            except Exception:
                # try safe parsing for strings with commas / scientific notation
                try:
                    s = str(v).replace(",", "").replace("$", "").strip()
                    fv = float(s)
                    if math.isfinite(fv):
                        return float(fv)
                except Exception:
                    continue
        return 0.0

    liq = _num_val(("liquidity", "liquidity_usd", "dex_liquidity", "ray_liquidity_usd", "reserve_usd"))
    vol = _num_val(("volume_24h", "v24hUSD", "volume", "v24hUsd"))
    return (liq >= float(min_liquidity_usd)) or (vol >= float(min_volume_24h_usd))

async def get_associated_token_address(
    owner: Pubkey,
    mint: Pubkey,
    token_program_id: Optional[Pubkey] = None,
    solana_client: Optional[AsyncClient] = None,
) -> Pubkey:
    """
    Derive the associated token account address (ATA).
    Dynamically uses the correct token program ID (legacy or Token-2022).
    If token_program_id is not provided, fetches it via RPC (requires solana_client).
    """
    if token_program_id is None:
        if solana_client is None:
            raise ValueError("solana_client is required when token_program_id is not provided")
        token_program_id = await get_token_program_id(str(mint), solana_client)

    ata, _bump = Pubkey.find_program_address(
        [bytes(owner), bytes(token_program_id), bytes(mint)],
        ASSOCIATED_TOKEN_PROGRAM_ID,
    )
    return ata


def get_associated_token_address_sync(
    owner: Pubkey,
    mint: Pubkey,
    token_program_id: Pubkey,
) -> Pubkey:
    """
    Synchronous version of ATA derivation.
    Use when you already know the token_program_id (legacy or Token-2022) and don't want to await RPC.
    """
    ata, _bump = Pubkey.find_program_address(
        [bytes(owner), bytes(token_program_id), bytes(mint)],
        ASSOCIATED_TOKEN_PROGRAM_ID,
    )
    return ata


def _is_valid_pubkey_str(s: Any) -> bool:
    """
    Lightweight syntactic check: attempt to construct a solders.Pubkey.
    Returns True if the string is a valid base58 pubkey accepted by solders.
    """
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s:
        return False
    try:
        Pubkey.from_string(s)
        return True
    except Exception:
        return False


async def is_valid_mint(token_mint: str, solana_client: AsyncClient) -> Tuple[bool, str]:
    """
    Centralized mint validation helper.

    Returns (is_valid, reason). Reason is a short string for logging.
    Logic:
      - Syntactic check (fast). If malformed -> invalid.
      - If cached -> use cache.
      - If DRY_RUN -> treat syntactically-valid mints as valid so dry-run flows can proceed.
      - Otherwise perform on-chain get_token_supply:
         • if supply info returned -> valid
         • if RPC says "could not find account" -> invalid on this cluster
         • if RPC says "not a Token mint" -> invalid (wrong account type)
         • other RPC errors -> treat as invalid (but log stacktrace)
    Caches positive and negative results to reduce RPC volume.
    """
    token_mint = (token_mint or "").strip()
    if not _is_valid_pubkey_str(token_mint):
        return False, "malformed_pubkey"

    # Check cache
    cached = mint_validation_cache.get(token_mint)
    if cached is not None:
        return bool(cached), "cached_valid" if cached else "cached_invalid"

    # If DRY_RUN, accept syntactically-valid mints (avoid blacklisting mainnet mints when testing on devnet)
    if _env_truthy("DRY_RUN", "0"):
        mint_validation_cache[token_mint] = True
        return True, "dry_run_syntactic_accept"

    try:
        mint = Pubkey.from_string(token_mint)
        resp = await solana_client.get_token_supply(mint, commitment=Commitment("confirmed"))
        if getattr(resp, "value", None) is not None:
            # Confirmed SPL mint present on the connected cluster
            mint_validation_cache[token_mint] = True
            return True, "onchain_supply"
        # No value -> considered invalid
        mint_validation_cache[token_mint] = False
        return False, "no_supply_value"
    except SolanaRPCException as e:
        # Inspect message when possible to provide finer reasons
        msg = str(e)
        lower = msg.lower()
        if "could not find account" in lower or "account not found" in lower:
            mint_validation_cache[token_mint] = False
            return False, "account_not_found_on_cluster"
        if "not a token mint" in lower or "not a Token mint".lower() in lower:
            mint_validation_cache[token_mint] = False
            return False, "account_not_a_token_mint"
        # Other RPC exceptions: treat as invalid but don't crash
        log_error_with_stacktrace(f"RPC error while validating mint {token_mint}", e)
        mint_validation_cache[token_mint] = False
        return False, "rpc_exception"
    except Exception as e:
        # Unexpected errors
        log_error_with_stacktrace(f"Unexpected error validating mint {token_mint}", e)
        mint_validation_cache[token_mint] = False
        return False, "unexpected_exception"


async def validate_token_mint(token_mint: str, solana_client: AsyncClient) -> bool:
    """
    Backwards-compatible wrapper that returns True only when the mint is valid
    (or accepted because DRY_RUN=1). Logs the reason for easier debugging.
    """
    ok, reason = await is_valid_mint(token_mint, solana_client)
    if ok:
        logger.debug("Validated token mint %s (%s)", token_mint, reason)
    else:
        # Less noisy log lines for common 'not found' cases
        if reason == "account_not_found_on_cluster":
            logger.warning("Invalid token mint %s: %s", token_mint, "account not found on connected cluster")
        elif reason == "account_not_a_token_mint":
            logger.warning("Invalid token mint %s: %s", token_mint, "account exists but is not an SPL token mint")
        elif reason == "malformed_pubkey":
            logger.warning("Malformed token mint %s", token_mint)
        else:
            logger.warning("Invalid token mint %s: %s", token_mint, reason)
    return ok


async def check_token_account(
    user_pubkey: str,
    token_mint: str,
    solana_client: AsyncClient,
    blacklist: set,
    failure_count: Dict,
    token_data: Optional[Dict] = None,
) -> bool:
    """
    Check whether the user's ATA exists for a given mint.

    Improvements:
      - If token_mint is missing, skip (do not increment failure_count/blacklist).
      - Derive ATA using the mint's actual token program id (Tokenkeg vs Token-2022).
      - Treat transient RPC simulation/preflight errors as non-blacklistable.
    """
    symbol = (token_data or {}).get("symbol", "UNKNOWN")
    try:
        if not token_mint:
            logger.debug("Missing token_mint for token account check for %s (owner=%s); skipping", symbol, user_pubkey)
            return False

        cache_key = f"account:{user_pubkey}:{token_mint}"
        cached = token_account_existence_cache.get(cache_key)
        if cached is not None:
            logger.debug("Using cached token account existence for %s (%s): %s", symbol, token_mint, cached)
            return bool(cached)

        # Basic syntactic validation
        if not _is_valid_pubkey_str(user_pubkey) or not _is_valid_pubkey_str(token_mint):
            logger.warning("Malformed owner or mint while checking token account for %s: owner=%s mint=%s",
                           symbol, user_pubkey, token_mint)
            # Count malformed as failures (client-side issue)
            failure_count[token_mint] = failure_count.get(token_mint, 0) + 1
            if failure_count[token_mint] >= 5 and token_mint not in WHITELISTED_TOKENS:
                logger.warning("Blacklisting %s (%s) due to repeated failures", symbol, token_mint)
                blacklist.add(token_mint)
            return False

        # DRY_RUN short-circuit
        if _env_truthy("DRY_RUN", "0"):
            logger.info("DRY_RUN: assuming token account exists for %s (%s)", symbol, token_mint)
            token_account_existence_cache[cache_key] = True
            return True

        owner = Pubkey.from_string(user_pubkey)
        mint = Pubkey.from_string(token_mint)

        # Try to detect the mint's token program id (Tokenkeg vs Token-2022)
        token_program_id = await get_token_program_id(token_mint, solana_client)

        # Derive ATA using the detected token program id
        ata, _bump = Pubkey.find_program_address(
            [bytes(owner), bytes(token_program_id), bytes(mint)],
            ASSOCIATED_TOKEN_PROGRAM_ID,
        )

        account_info = await solana_client.get_account_info(ata, commitment=Commitment("confirmed"))
        exists = account_info.value is not None
        token_account_existence_cache[cache_key] = exists

        if exists:
            logger.debug("Token account exists for %s (%s): %s", symbol, token_mint, ata)
        else:
            logger.debug("No token account found for %s (%s)", symbol, token_mint)

        return exists

    except SolanaRPCException as e:
        msg = str(e)
        logger.warning("RPC error while checking token account for %s (%s): %s", symbol, token_mint, msg)

        # transient simulation/preflight errors should NOT increment persistent failure_count
        if _is_transient_rpc_error_text(msg):
            logger.debug("Transient RPC error detected for token account check: %s", msg)
            return False

        # non-transient: increment and possibly blacklist
        failure_count[token_mint] = failure_count.get(token_mint, 0) + 1
        if failure_count[token_mint] >= 5 and token_mint not in WHITELISTED_TOKENS:
            logger.warning("Blacklisting %s (%s) due to repeated RPC failures", symbol, token_mint)
            blacklist.add(token_mint)
        return False

    except Exception as e:
        log_error_with_stacktrace(f"Error checking token account for {symbol} ({token_mint})", e)
        failure_count[token_mint] = failure_count.get(token_mint, 0) + 1
        if failure_count[token_mint] >= 5 and token_mint not in WHITELISTED_TOKENS:
            logger.warning("Blacklisting %s (%s) due to repeated failures", symbol, token_mint)
            blacklist.add(token_mint)
        return False


# Replace create_token_account implementation (use mint's token program id for ATA/instruction)
async def create_token_account(
    wallet: Keypair,
    token_mint: Optional[str] = None,
    solana_client: Optional[AsyncClient] = None,
    token_data: Optional[Dict] = None,
    *,
    mint_address: Optional[str] = None,
    token: Optional[Union[Dict, str]] = None,
) -> Optional[Pubkey]:
    """
    Ensure the user's Associated Token Account (ATA) exists for `token_mint`.

    Defensive behavior:
      - If token_mint cannot be derived, return None (do not raise) to avoid noisy blacklisting.
      - Detect and use the mint's token program id (Tokenkeg vs Token-2022) when deriving
        the ATA and constructing the create instruction.
    """
    # Accept alias names / legacy shapes
    if token_mint is None and mint_address is not None:
        token_mint = mint_address
    if token_mint is None and token is not None:
        try:
            if isinstance(token, dict):
                token_mint = token.get("address") or token.get("mint") or token.get("token_address")
                if token_data is None:
                    token_data = token
            elif isinstance(token, str):
                token_mint = token
        except Exception:
            pass
    if token_mint is None and token_data is not None and isinstance(token_data, dict):
        token_mint = token_data.get("address") or token_data.get("mint") or token_data.get("token_address")

    symbol = (token_data or {}).get("symbol", "UNKNOWN")
    try:
        # If token_mint still missing, treat as missing metadata (do not raise)
        if not token_mint:
            logger.debug("create_token_account: token_mint not provided for %s; skipping ATA creation", symbol)
            return None

        if not _is_valid_pubkey_str(token_mint):
            logger.warning("Malformed token mint string for %s: %s", symbol, token_mint)
            return None

        mint = Pubkey.from_string(token_mint)
        owner = wallet.pubkey()

        # Determine the mint's token program id (Tokenkeg or Token-2022)
        token_program_id = await get_token_program_id(token_mint, solana_client) if solana_client is not None else TOKEN_PROGRAM_ID

        # Derive ATA address using the correct token program id
        ata, _bump = Pubkey.find_program_address(
            [bytes(owner), bytes(token_program_id), bytes(mint)],
            ASSOCIATED_TOKEN_PROGRAM_ID,
        )

        # Early exit if ATA exists
        if solana_client is None:
            logger.debug("create_token_account: solana_client not provided for %s (%s)", symbol, token_mint)
            return None

        info = await solana_client.get_account_info(ata, commitment=Commitment("confirmed"))
        if info.value is not None:
            logger.debug("Token account already exists for %s (%s): %s", symbol, token_mint, ata)
            return ata

        # DRY RUN guard
        if _env_truthy("DRY_RUN", "0"):
            logger.info("DRY_RUN: skipping ATA creation for %s (%s); pretending it exists.", symbol, token_mint)
            token_account_existence_cache[f"account:{str(owner)}:{token_mint}"] = True
            return ata

        # Prefer SPL idempotent instruction when available; when constructing a raw fallback
        # ensure we use token_program_id instead of hard-coded TOKEN_PROGRAM_ID.
        try:
            from spl.token.instructions import (  # type: ignore
                CreateAssociatedTokenAccountParams,
                create_associated_token_account_idempotent,
            )
            ix = create_associated_token_account_idempotent(
                CreateAssociatedTokenAccountParams(
                    funder=owner,
                    owner=owner,
                    mint=mint,
                    associated_token_program_id=ASSOCIATED_TOKEN_PROGRAM_ID,
                    program_id=token_program_id,
                )
            )
        except Exception:
            logger.debug("Falling back to raw ATA create instruction for %s (%s)", symbol, token_mint)
            ix = Instruction(
                program_id=ASSOCIATED_TOKEN_PROGRAM_ID,
                accounts=[
                    AccountMeta(pubkey=owner, is_signer=True, is_writable=True),    # funder
                    AccountMeta(pubkey=ata, is_signer=False, is_writable=True),     # ATA
                    AccountMeta(pubkey=owner, is_signer=False, is_writable=False),  # owner
                    AccountMeta(pubkey=mint, is_signer=False, is_writable=False),   # mint
                    AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
                    AccountMeta(pubkey=token_program_id, is_signer=False, is_writable=False),
                    AccountMeta(pubkey=RENT_SYSVAR, is_signer=False, is_writable=False),
                ],
                data=b"",
            )

        # Build, sign and submit raw bytes
        recent = await solana_client.get_latest_blockhash()
        message = Message.new_with_blockhash([ix], owner, recent.value.blockhash)
        tx = VersionedTransaction(message, [wallet])  # sign locally

        try:
            raw_tx = bytes(tx)
        except Exception:
            raw_tx = tx.to_bytes()

        send_opts = TxOpts(skip_preflight=False, skip_confirmation=False)
        resp = await solana_client.send_raw_transaction(raw_tx, opts=send_opts)
        sig = _extract_signature(resp) or str(resp)

        logger.info("Created token account for %s (%s): %s, txid: %s", symbol, token_mint, ata, sig)
        token_account_existence_cache[f"account:{str(owner)}:{token_mint}"] = True
        return ata

    except SolanaRPCException as e:
        msg = str(e)
        logger.warning("RPC error while creating token account for %s (%s): %s", symbol, token_mint, msg)
        if _is_transient_rpc_error_text(msg):
            logger.debug("Transient RPC error during ATA creation for %s: %s", token_mint, msg)
            return None
        log_error_with_stacktrace(f"Non-transient RPC error while creating token account for {symbol} ({token_mint})", e)
        return None
    except Exception as e:
        log_error_with_stacktrace(f"Failed to create token account for {symbol} ({token_mint})", e)
        return None

def _env_truthy(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "on", "y")


def _extract_signature(resp: Any) -> str:
    """
    Best-effort extraction of signature from solana-py responses across versions.
    """
    try:
        if isinstance(resp, str):
            return resp
        if hasattr(resp, "value") and isinstance(resp.value, str):
            return resp.value
        if isinstance(resp, dict):
            return resp.get("value") or resp.get("result") or ""
    except Exception:
        pass
    return ""

async def get_token_balance(
    wallet: Keypair,
    token_mint: str,
    solana_client: AsyncClient,
    token_data: Optional[Dict] = None,
) -> float:
    symbol = (token_data or {}).get("symbol", "UNKNOWN")
    try:
        cache_key = f"balance:{wallet.pubkey()}:{token_mint}"
        cached = token_balance_cache.get(cache_key)
        if cached is not None:
            logger.debug("Using cached balance for %s (%s): %s", symbol, token_mint, cached)
            return float(cached)

        if not _is_valid_pubkey_str(token_mint):
            logger.warning("Malformed token mint while fetching balance for %s: %s", symbol, token_mint)
            return 0.0

        # Get the ATA address using the proper token program ID
        mint_pubkey = Pubkey.from_string(token_mint)
        ata = await get_associated_token_address(wallet.pubkey(), mint_pubkey, solana_client=solana_client)

        resp = await solana_client.get_token_account_balance(ata, commitment=Commitment("confirmed"))
        balance = float(resp.value.ui_amount) if getattr(resp, "value", None) else 0.0

        token_balance_cache[cache_key] = balance
        logger.debug("Token balance for %s (%s): %s", symbol, token_mint, balance)
        return balance
    except SolanaRPCException as e:
        logger.warning("RPC error fetching token balance for %s (%s): %s", symbol, token_mint, e)
        return 0.0
    except Exception as e:
        log_error_with_stacktrace(f"Error fetching token balance for %s (%s)" % (symbol, token_mint), e)
        return 0.0

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    # jittered exponential backoff helps avoid thundering herds
    wait=wait_random_exponential(multiplier=1, max=180),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
)
async def get_token_market_cap(
    token_address: str,
    session: aiohttp.ClientSession,
    solana_client: AsyncClient,
    token_data: Optional[Dict] = None,
) -> float:
    symbol = (token_data or {}).get("symbol", "UNKNOWN")
    cache_key = f"market_cap:{token_address}"

    cached = price_cache.get(cache_key)
    if cached is not None:
        logger.debug("Using cached market cap for %s (%s): %s", symbol, token_address, cached)
        return float(cached)

    # If token metadata already comes with a market cap, use it.
    if token_data and token_data.get("market_cap", 0):
        mc = float(token_data.get("market_cap", 0))
        if mc > 0:
            price_cache[cache_key] = mc
            logger.debug("Using provided token data market cap for %s (%s): %s", symbol, token_address, mc)
            return mc

    # Birdeye
    try:
        api_key = os.getenv("BIRDEYE_API_KEY")
        if api_key:
            headers = {**HTTP_HEADERS, "X-API-KEY": api_key}
            url = "https://public-api.birdeye.so/defi/token/overview"
            params = {"address": token_address}

            status, text = await _fetch_text_with_ipv4_fallback(
                "GET", url, params=params, headers=headers, timeout=MED_TIMEOUT, session=session
            )
            if status == 429:
                logger.info("Birdeye market cap rate limit for %s (%s); backing off 60s", symbol, token_address)
                await asyncio.sleep(60)
                raise aiohttp.ClientError("Rate limit")

            if status == 200:
                data = json.loads(text)
                mc = float((data.get("data") or {}).get("marketCap", 0) or 0)
                if mc > 0:
                    price_cache[cache_key] = mc
                    logger.debug("Birdeye market cap for %s (%s): %s", symbol, token_address, mc)
                    return mc
    except Exception as e:
        log_error_with_stacktrace(f"Birdeye market cap fetch failed for {symbol} ({token_address})", e)

    # Dexscreener
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        status, text = await _fetch_text_with_ipv4_fallback(
            "GET", url, headers=HTTP_HEADERS, timeout=SHORT_TIMEOUT, session=session
        )

        if status == 429:
            logger.info("Dexscreener market cap rate limit for %s (%s); backing off 60s", symbol, token_address)
            await asyncio.sleep(60)
            raise aiohttp.ClientError("Rate limit")

        if status == 200:
            data = json.loads(text)
            pairs = data.get("pairs", []) or []
            if pairs and pairs[0].get("fdv"):
                mc = float(pairs[0].get("fdv", 0) or 0)
                if mc > 0:
                    price_cache[cache_key] = mc
                    logger.debug("Dexscreener market cap for %s (%s): %s", symbol, token_address, mc)
                    return mc
            # Optional SOL special-case
            if token_address == "So11111111111111111111111111111111111111112":
                price_usd = float(pairs[0].get("priceUsd", 0) if pairs else 0.0)
                mc = price_usd * 582_000_000
                if mc > 0:
                    price_cache[cache_key] = mc
                    logger.debug("SOL market cap calculated as: %s", mc)
                    return mc
    except Exception as e:
        log_error_with_stacktrace(f"Dexscreener market cap fetch failed for {symbol} ({token_address})", e)

    # On-chain last-resort estimate
    try:
        if not _is_valid_pubkey_str(token_address):
            logger.debug("Skipping on-chain market cap check for malformed token address: %s", token_address)
        else:
            # Wrap on-chain supply retrieval to avoid noisy warnings for devnet/mainnet mismatch.
            try:
                mint = Pubkey.from_string(token_address)
                supply_resp = await solana_client.get_token_supply(mint, commitment=Commitment("confirmed"))
            except SolanaRPCException as e:
                # If DRY_RUN: don't warn; just skip on-chain estimate (we already tried APIs)
                if _env_truthy("DRY_RUN", "0"):
                    logger.debug("DRY_RUN: on-chain supply unavailable for %s: %s", token_address, e)
                else:
                    logger.warning("On-chain market cap: token %s not found or not a mint: %s", token_address, e)
                supply_resp = None
            except Exception as e:
                log_error_with_stacktrace(f"On-chain market cap fetch failed for {symbol} ({token_address})", e)
                supply_resp = None

            if supply_resp and getattr(supply_resp, "value", None):
                supply_ui = float(supply_resp.value.ui_amount or 0.0)
                price_in_sol = await get_token_price_in_sol(token_address, session, price_cache, token_data)
                if price_in_sol and supply_ui:
                    sol_price = await get_sol_price(session)
                    mc = supply_ui * price_in_sol * sol_price
                    if mc > 0:
                        price_cache[cache_key] = mc
                        logger.debug("On-chain market cap for %s (%s): %s", symbol, token_address, mc)
                        return mc
    except Exception as e:
        log_error_with_stacktrace(f"On-chain market cap fetch failed for {symbol} ({token_address})", e)

    logger.warning("No market cap found for %s (%s); defaulting to 0", symbol, token_address)
    price_cache[cache_key] = 0.0
    return 0.0


async def get_sol_balance(wallet_or_pubkey: Union[Keypair, Pubkey, str], solana_client: AsyncClient) -> float:
    """
    Return SOL balance (in SOL) for a wallet/key. Accepts:
      • Keypair (has .pubkey())
      • Pubkey
      • str (base58 pubkey)
    """
    try:
        if hasattr(wallet_or_pubkey, "pubkey"):
            pubkey = wallet_or_pubkey.pubkey()  # Keypair-like
        elif isinstance(wallet_or_pubkey, Pubkey):
            pubkey = wallet_or_pubkey
        elif isinstance(wallet_or_pubkey, str):
            pubkey = Pubkey.from_string(wallet_or_pubkey)
        else:
            raise TypeError(f"Unsupported wallet type: {type(wallet_or_pubkey)}")

        account_info = await solana_client.get_balance(pubkey, commitment=Commitment("confirmed"))
        balance = float(account_info.value) / 1e9
        try:
            logger.debug("Wallet %s SOL balance: %.8f SOL", str(pubkey), balance)
        except Exception:
            pass
        return balance
    except Exception as e:
        try:
            who = str(pubkey)  # may be set above
        except Exception:
            try:
                who = str(wallet_or_pubkey)
            except Exception:
                who = "<unknown>"
        log_error_with_stacktrace(f"Error fetching SOL balance for {who}", e)
        return 0.0


async def get_sol_price(session: aiohttp.ClientSession, *_ignore, **_kw) -> float:
    """
    Get current SOL price in USD (cached). Compatible with older call sites.
    """
    cfg = load_config()
    fallback = (cfg.get("bot", {}) or {}).get("sol_fallback_price", 180.0)

    cached = price_cache.get(SOL_PRICE_CACHE_KEY)
    if cached is not None:
        return float(cached)

    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "solana", "vs_currencies": "usd"}

        status, text = await _fetch_text_with_ipv4_fallback(
            "GET", url, params=params, headers=HTTP_HEADERS, timeout=SHORT_TIMEOUT, session=session
        )
        if status != 200:
            logger.warning("Failed to fetch SOL price: HTTP %s, %s", status, text[:300])
            price_cache[SOL_PRICE_CACHE_KEY] = fallback
            return float(fallback)

        data = json.loads(text)
        price = float((data.get("solana") or {}).get("usd", fallback))
        price_cache[SOL_PRICE_CACHE_KEY] = price
        logger.debug("SOL price: $%.2f", price)
        return price
    except Exception as e:
        log_error_with_stacktrace("Error fetching SOL price", e)
        price_cache[SOL_PRICE_CACHE_KEY] = fallback
        return float(fallback)


async def get_token_price_in_sol(
    token_address: str,
    session: aiohttp.ClientSession,
    price_cache: TTLCache,
    token_data: Optional[Dict] = None,
) -> float:
    symbol = (token_data or {}).get("symbol", "UNKNOWN")
    cache_key = f"price_in_sol:{token_address}"

    cached = price_cache.get(cache_key)
    if cached is not None:
        logger.debug("Using cached price for %s (%s): %s SOL", symbol, token_address, cached)
        return float(cached)

    # Birdeye (USD → convert to SOL)
    try:
        api_key = os.getenv("BIRDEYE_API_KEY")
        if api_key:
            headers = {**HTTP_HEADERS, "X-API-KEY": api_key}
            url = "https://public-api.birdeye.so/defi/price"
            params = {"address": token_address}

            status, text = await _fetch_text_with_ipv4_fallback(
                "GET", url, params=params, headers=headers, timeout=SHORT_TIMEOUT, session=session
            )
            if status == 429:
                logger.warning("Birdeye price 429 for %s (%s); backoff 60s", symbol, token_address)
                await asyncio.sleep(60)
                raise aiohttp.ClientError("Rate limit")

            if status == 200:
                data = json.loads(text)
                price_usd = float((data.get("data") or {}).get("value", 0) or 0)
                if price_usd > 0:
                    sol_price = await get_sol_price(session)
                    price_in_sol = (price_usd / sol_price) if sol_price > 0 else 0.0
                    if price_in_sol > 0:
                        price_cache[cache_key] = price_in_sol
                        logger.debug("Birdeye price for %s (%s): %s SOL", symbol, token_address, price_in_sol)
                        return float(price_in_sol)
    except Exception as e:
        log_error_with_stacktrace(f"Birdeye price fetch failed for {symbol} ({token_address})", e)

    # Dexscreener (has priceUsd directly)
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
        status, text = await _fetch_text_with_ipv4_fallback(
            "GET", url, headers=HTTP_HEADERS, timeout=SHORT_TIMEOUT, session=session
        )
        if status == 200:
            data = json.loads(text)
            pairs = data.get("pairs", []) or []
            chosen = None
            for p in pairs:
                try:
                    if p.get("chainId") != "solana":
                        continue
                    if p.get("priceUsd"):
                        chosen = p
                        break
                except Exception:
                    continue
            if chosen:
                price_usd = float(chosen.get("priceUsd", 0) or 0)
                if price_usd > 0:
                    sol_price = await get_sol_price(session)
                    price_in_sol = (price_usd / sol_price) if sol_price > 0 else 0.0
                    if price_in_sol > 0:
                        price_cache[cache_key] = price_in_sol
                        logger.debug("Dexscreener price for %s (%s): %s SOL", symbol, token_address, price_in_sol)
                        return float(price_in_sol)
    except Exception as e:
        log_error_with_stacktrace(f"Dexscreener price fetch failed for {symbol} ({token_address})", e)

    logger.warning("No price found for %s (%s); defaulting to 0", symbol, token_address)
    price_cache[cache_key] = 0.0
    return 0.0


async def get_token_price(token: Dict, session: aiohttp.ClientSession) -> float:
    token_address = token.get("address")
    if not token_address:
        logger.warning("Missing address for token: %s", token)
        return 0.0
    try:
        cache_key = f"price_usd:{token_address}"
        cached = price_cache.get(cache_key)
        if cached is not None:
            logger.debug(
                "Using cached USD price for %s (%s): $%.6f",
                token.get("symbol", "UNKNOWN"), token_address, cached
            )
            return float(cached)

        price_in_sol = await get_token_price_in_sol(token_address, session, price_cache, token)
        if price_in_sol > 0:
            sol_price = await get_sol_price(session)
            price_usd = price_in_sol * sol_price
            price_cache[cache_key] = price_usd
            logger.debug(
                "USD price for %s (%s): $%.6f",
                token.get("symbol", "UNKNOWN"), token_address, price_usd
            )
            return float(price_usd)

        logger.warning("No USD price found for %s (%s); using fallback", token.get("symbol", "UNKNOWN"), token_address)
        return float(token.get("price", 0.0))
    except Exception as e:
        log_error_with_stacktrace(
            f"Error fetching USD price for {token.get('symbol', 'UNKNOWN')} ({token_address})", e
        )
        return float(token.get("price", 0.0))


async def get_jupiter_quote(
    input_mint: str,
    output_mint: str,
    amount: int,
    user_pubkey: str,  # kept for signature compatibility; not used by quote
    session: aiohttp.ClientSession,
    slippage_bps: int = 50,
    *,
    token_data: Optional[Dict] = None,
    min_liquidity_usd: float | None = None,
    min_volume_24h_usd: float | None = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Jupiter Lite quote: GET /swap/v1/quote with dual-stack → IPv4 fallback.

    Defensive addition:
      - If token_data provided, run a fast pre-quote liquidity check (is_sufficient_liquidity).
        If liquidity/volume are below thresholds, avoid calling Jupiter (returns (None, "insufficient_liquidity")).

    Backwards-compatible: token_data and min_* thresholds are optional and do not change prior behavior.
    """
    try:
        # If caller provided token_data, enforce a pre-check to avoid NO_ROUTE / wasted API calls
        if token_data is not None:
            min_liq = float(min_liquidity_usd) if min_liquidity_usd is not None else float(os.getenv("JUP_MIN_QUOTE_LIQ", "500.0"))
            min_vol = float(min_volume_24h_usd) if min_volume_24h_usd is not None else float(os.getenv("JUP_MIN_QUOTE_VOL", "100.0"))
            try:
                if not is_sufficient_liquidity(token_data, min_liquidity_usd=min_liq, min_volume_24h_usd=min_vol):
                    msg = f"insufficient_liquidity (liq<{min_liq} & vol24<{min_vol})"
                    logger.info("Skipping Jupiter quote for %s -> %s: %s", input_mint, output_mint, msg)
                    return None, msg
            except Exception:
                # If the helper errors, continue to attempt the quote (be conservative)
                logger.debug("is_sufficient_liquidity check failed — attempting quote anyway", exc_info=True)

        url = "https://lite-api.jup.ag/swap/v1/quote"
        params = {
            "inputMint":   input_mint,
            "outputMint":  output_mint,
            "amount":      str(amount),         # strings match Jupiter examples
            "slippageBps": str(slippage_bps),
        }

        status, text = await _fetch_text_with_ipv4_fallback(
            "GET", url, params=params, headers=HTTP_HEADERS, timeout=SHORT_TIMEOUT, session=session
        )

        if status != 200:
            logger.error(
                "Jupiter Lite quote failed for %s -> %s: HTTP %s, %s",
                input_mint, output_mint, status, (text or "")[:300]
            )
            return None, text

        quote = json.loads(text)
        logger.debug("Jupiter Lite quote %s -> %s: %s", input_mint, output_mint, quote)
        return quote, None

    except Exception as e:
        log_error_with_stacktrace(
            f"Error fetching Jupiter Lite quote for {input_mint} -> {output_mint}", e
        )
        return None, str(e)


async def build_jupiter_swap(
    quote_json: Dict,
    user_pubkey: str,
    session: aiohttp.ClientSession,
    compute_unit_price_micro_lamports: int = 1500,
    wrap_and_unwrap_sol: bool = True,
    use_shared_accounts: bool = True,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Jupiter Lite swap: POST /swap/v1/swap with dual-stack → IPv4 fallback.
    Returns JSON containing 'swapTransaction' (base64) and metadata.
    """
    try:
        url = "https://lite-api.jup.ag/swap/v1/swap"
        body = {
            "quoteResponse": quote_json,
            "userPublicKey": user_pubkey,
            "wrapAndUnwrapSol": wrap_and_unwrap_sol,
            "useSharedAccounts": use_shared_accounts,
            "computeUnitPriceMicroLamports": int(compute_unit_price_micro_lamports),
        }

        status, text = await _fetch_text_with_ipv4_fallback(
            "POST", url,
            data=json.dumps(body),
            headers={**HTTP_HEADERS, "Content-Type": "application/json"},
            timeout=SHORT_TIMEOUT,
            session=session,
        )

        if status != 200:
            logger.error("Jupiter Lite swap build failed (HTTP %s): %s", status, text[:300])
            return None, text

        data = json.loads(text)
        if "swapTransaction" not in data:
            logger.error("Jupiter Lite swap build missing 'swapTransaction': %s", text[:300])
            return None, "missing swapTransaction"

        logger.debug("Jupiter Lite swap built OK (lastValidBlockHeight=%s)", data.get("lastValidBlockHeight"))
        return data, None

    except Exception as e:
        log_error_with_stacktrace("Error building Jupiter Lite swap", e)
        return None, str(e)


def format_market_cap(market_cap: float) -> str:
    try:
        if market_cap >= 1_000_000_000:
            return f"${market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            return f"${market_cap / 1_000_000:.2f}M"
        elif market_cap >= 1_000:
            return f"${market_cap / 1_000:.2f}K"
        else:
            return f"${market_cap:.2f}"
    except Exception as e:
        logger.error("Error formatting market cap %s: %s", market_cap, e)
        return "$0.00"