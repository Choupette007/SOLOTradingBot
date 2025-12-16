# solana_trading_bot_bundle/trading_bot/rugcheck_client.py
import os
import time
import random
import asyncio
import logging
import inspect
from typing import Any, Dict, Optional

import aiohttp

# Reuse your packaged helper for JWT refresh/persistence
try:
    from .rugcheck_auth import ensure_valid_rugcheck_headers, get_rugcheck_headers  # type: ignore
except Exception:
    # fallback import path if called outside package context
    try:
        from solana_trading_bot_bundle.trading_bot.rugcheck_auth import ensure_valid_rugcheck_headers, get_rugcheck_headers  # type: ignore
    except Exception:
        ensure_valid_rugcheck_headers = None  # type: ignore
        get_rugcheck_headers = None  # type: ignore

logger = logging.getLogger(__name__)


class RugcheckClient:
    """
    Async Rugcheck client with rate spacing, concurrency limit, Retry-After honoring,
    and exponential backoff with jitter. Uses either API key (X-API-KEY) or JWT
    (Authorization: Bearer <jwt>) depending on environment and caller preference.

    Notes:
      - By default callers can request JWT-based auth (use_jwt=True) but the client
        will only attempt wallet signing if a likely private-key is present.
      - Prefer a static JWT in env (RUGCHECK_JWT_TOKEN / RUGCHECK_JWT) when available.
      - If no JWT is available, fall back to RUGCHECK_API_KEY (X-API-KEY) or the
        legacy get_rugcheck_headers() helper.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.rugcheck.xyz",
        max_concurrency: int = 1,
        min_interval: float = 1.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
        session_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key or os.getenv("RUGCHECK_API_KEY")
        self.base_url = base_url.rstrip("/")
        self._sem = asyncio.Semaphore(max_concurrency)
        self._min_interval = float(min_interval)
        self._max_retries = int(max_retries)
        self._backoff_base = float(backoff_base)
        self._backoff_max = float(backoff_max)
        self._last_call = 0.0
        self._session_kwargs = session_kwargs or {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_lock = asyncio.Lock()

        # Log (non-secret) availability of common auth methods
        try:
            has_jwt_env = bool(os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT"))
            has_sk = bool(os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("SOLANA_PRIVATE_KEY_BASE58") or os.getenv("SOLANA_PRIVATE_KEY_JSON"))
            has_api_key = bool(self.api_key)
            logger.info("Rugcheck auth availability: jwt_env=%s private_key=%s api_key=%s", has_jwt_env, has_sk, has_api_key)
        except Exception:
            pass

    async def start(self) -> None:
        if self._session is None:
            headers = {"Accept": "application/json"}
            self._session = aiohttp.ClientSession(headers=headers, **self._session_kwargs)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def _wait_rate_slot(self) -> None:
        # Ensure at least min_interval between calls
        async with self._rate_lock:
            now = time.time()
            elapsed = now - self._last_call
            to_wait = self._min_interval - elapsed
            if to_wait > 0:
                await asyncio.sleep(to_wait)

    def _backoff(self, attempt: int) -> float:
        base = self._backoff_base * (2 ** (attempt - 1))
        capped = min(base, self._backoff_max)
        jitter = random.uniform(0, base * 0.1)
        return min(capped + jitter, self._backoff_max)

    def _redact(self, s: Optional[str]) -> str:
        if not s:
            return "<missing>"
        s = str(s)
        if len(s) <= 10:
            return s[:2] + "..." + s[-2:]
        return s[:4] + "..." + s[-4:]

    async def _auth_headers(self, use_jwt: bool = True) -> Dict[str, str]:
        """
        Compose auth headers for Rugcheck requests.

        Priority:
         1) Static JWT: RUGCHECK_JWT_TOKEN or RUGCHECK_JWT (env)
         2) If use_jwt=True and a SOLANA private key appears available, attempt
            to obtain a wallet-signed JWT via ensure_valid_rugcheck_headers().
         3) RUGCHECK_API_KEY -> X-API-KEY
         4) get_rugcheck_headers() legacy helper
         5) Empty dict (caller must handle missing auth)
        """
        hdrs: Dict[str, str] = {}

        # 1) Static JWT env
        jwt_token = os.getenv("RUGCHECK_JWT_TOKEN") or os.getenv("RUGCHECK_JWT")
        if jwt_token:
            hdrs["Authorization"] = f"Bearer {jwt_token}"
            logger.debug("Using static RUGCHECK_JWT_TOKEN (redacted=%s)", self._redact(jwt_token))
            return hdrs

        # 2) Wallet-signed JWT (only if requested and private key present)
        if use_jwt:
            sk = os.getenv("SOLANA_PRIVATE_KEY") or os.getenv("SOLANA_PRIVATE_KEY_BASE58") or os.getenv("SOLANA_PRIVATE_KEY_JSON")
            if sk and ensure_valid_rugcheck_headers is not None:
                try:
                    # Call ensure_valid_rugcheck_headers in a flexible way:
                    # - It may accept (session, force_refresh) or (session) or no args.
                    # - It may be sync or async.
                    fn = ensure_valid_rugcheck_headers
                    hdr_obj = None
                    try:
                        sig = inspect.signature(fn)
                        params = sig.parameters
                        # prefer passing our session if available
                        if self._session is not None and len(params) >= 1:
                            maybe = fn(self._session)
                        elif len(params) == 0:
                            maybe = fn()
                        else:
                            # try calling with no args as last resort
                            maybe = fn()
                    except (ValueError, TypeError):
                        # signature introspection failed: call without args
                        maybe = fn()

                    if asyncio.iscoroutine(maybe):
                        hdr_obj = await maybe
                    else:
                        hdr_obj = maybe

                    if isinstance(hdr_obj, dict) and hdr_obj.get("Authorization"):
                        logger.debug("Using wallet-signed JWT (private key present, redacted=%s)", self._redact(sk))
                        hdrs.update(hdr_obj)
                        return hdrs
                    else:
                        logger.debug("ensure_valid_rugcheck_headers returned no Authorization; falling back.")
                except Exception as e:
                    logger.warning("Wallet-signed Rugcheck JWT attempt failed: %s. Falling back to other auth.", e)

        # 3) API key fallback
        api_key = self.api_key or os.getenv("RUGCHECK_API_KEY") or os.getenv("RUGCHECK_KEY")
        if api_key:
            hdrs["X-API-KEY"] = str(api_key)
            logger.debug("Using RUGCHECK_API_KEY (redacted=%s)", self._redact(api_key))
            return hdrs

        # 4) Legacy helper fallback
        if get_rugcheck_headers is not None:
            try:
                maybe = get_rugcheck_headers()
                if asyncio.iscoroutine(maybe):
                    maybe = await maybe
                if isinstance(maybe, dict) and maybe:
                    logger.debug("Using get_rugcheck_headers() fallback (headers present)")
                    return dict(maybe)
            except Exception:
                logger.debug("get_rugcheck_headers() fallback failed", exc_info=True)

        # Nothing available
        logger.debug("No Rugcheck auth headers available (use_jwt=%s)", bool(use_jwt))
        return {}

    async def request(
            self,
            method: str,
            path: str,
            *,
            params: Optional[Dict[str, Any]] = None,
            json: Optional[Any] = None,
            use_jwt: bool = True,
            timeout: float = 20.0,
            allow_404: bool = False,
        ) -> Any:
            """
            Generic HTTP request with retry, rate limiting, and JWT / API-key auth.

            If allow_404=True, return None when server responds 404 instead of raising.
            """
            if self._session is None:
                await self.start()

            url = path if path.startswith("http") else f"{self.base_url}{path if path.startswith('/') else '/' + path}"

            last_exc: Optional[Exception] = None

            for attempt in range(1, self._max_retries + 1):
                try:
                    async with self._sem:
                        await self._wait_rate_slot()
                        if self._session is None:
                            await self.start()

                        headers = await self._auth_headers(use_jwt=use_jwt)

                        # redact header preview for logs (do not print secrets)
                        hdr_preview = {k: (v[:6] + "..." + v[-4:] if len(v) > 12 else v) for k, v in headers.items()}

                        logger.info(
                            "RugCheck → %s %s | Params: %s | Headers: %s",
                            method.upper(),
                            url,
                            params or {},
                            hdr_preview
                        )

                        async with self._session.request(
                            method, url,
                            params=params,
                            json=json,
                            headers=headers,
                            timeout=timeout
                        ) as resp:
                            self._last_call = time.time()
                            status = resp.status
                            text = await resp.text()

                            if status == 200:
                                try:
                                    return await resp.json()
                                except:
                                    return text

                            # Treat 404 specially when caller prefers it
                            if status == 404:
                                if allow_404:
                                    logger.debug("RugCheck 404 (not found) for %s; returning None", url)
                                    return None
                                logger.warning("RugCheck 404 (not found) for %s", url)
                                resp.raise_for_status()

                            if status == 401:
                                # Auth failed: clear any cached jwt helper state by attempting ensure_valid_rugcheck_headers(force_refresh) if available
                                logger.warning("Rugcheck HTTP 401 on %s", url)
                                try:
                                    # best-effort forced refresh
                                    if ensure_valid_rugcheck_headers is not None:
                                        maybe = ensure_valid_rugcheck_headers
                                        try:
                                            sig = inspect.signature(maybe)
                                            if len(sig.parameters) >= 1:
                                                # try to force refresh if function accepts (session, force_refresh)
                                                try:
                                                    maybe2 = maybe(self._session, True)
                                                except TypeError:
                                                    maybe2 = maybe(self._session)
                                            else:
                                                try:
                                                    maybe2 = maybe(True)
                                                except TypeError:
                                                    maybe2 = maybe()
                                        except Exception:
                                            try:
                                                maybe2 = maybe(True)
                                            except Exception:
                                                maybe2 = maybe()
                                        if asyncio.iscoroutine(maybe2):
                                            await maybe2
                                except Exception:
                                    pass
                                # Treat 401 as final for this attempt (caller can retry)
                                resp.raise_for_status()

                            if status == 429:
                                wait = self._backoff(attempt)
                                ra = resp.headers.get("Retry-After")
                                if ra:
                                    try:
                                        wait = max(wait, float(ra))
                                    except:
                                        pass
                                logger.warning("RugCheck 429 → retry in %.1fs", wait)
                                await asyncio.sleep(wait)
                                continue

                            if 500 <= status < 600:
                                wait = self._backoff(attempt)
                                logger.warning("RugCheck %d → retry in %.1fs", status, wait)
                                await asyncio.sleep(wait)
                                continue

                            logger.error("RugCheck failed %d → %s", status, text[:400])
                            resp.raise_for_status()

                except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                    last_exc = e
                    wait = self._backoff(attempt)
                    logger.debug("Network error → retry in %.1fs", wait, exc_info=True)
                    await asyncio.sleep(wait)
                    continue

                except Exception as e:
                    last_exc = e
                    logger.exception("Unexpected error (attempt %d)", attempt)
                    await asyncio.sleep(self._backoff(attempt))
                    continue

            raise last_exc or RuntimeError("Max retries exceeded")

    # convenience methods
    async def get(self, path: str, **kwargs) -> Any:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, json: Any = None, **kwargs) -> Any:
        return await self.request("POST", path, json=json, **kwargs)

    # ---- Convenience wrappers for common RugCheck endpoints ----
    async def get_report(self, token_address: str, *, use_jwt: bool = True, timeout: float = 20.0) -> Optional[dict]:
        """
        Fetch the detailed RugCheck report for `token_address`.
        - GET /v1/tokens/{mint}/report
        - Returns parsed JSON (dict) on 200, None on 404 (not found).
        """
        path = f"/v1/tokens/{token_address}/report"
        try:
            data = await self.get(path, use_jwt=use_jwt, timeout=timeout, allow_404=True)
        except Exception as e:
            logger.debug("Rugcheck get_report error for %s: %s", token_address, e, exc_info=True)
            raise
        return data if isinstance(data, dict) else None

    async def scan_token(self, token_address: str, *, chain: str = "solana", use_jwt: bool = True, timeout: float = 20.0) -> Optional[dict]:
        """
        Call the RugCheck scan endpoint:
          GET /v1/tokens/scan/{chain}/{mint}
        Returns parsed JSON (dict) on 200, None on 404.
        """
        path = f"/v1/tokens/scan/{chain}/{token_address}"
        try:
            data = await self.get(path, use_jwt=use_jwt, timeout=timeout, allow_404=True)
        except Exception as e:
            logger.debug("Rugcheck scan_token error for %s (chain=%s): %s", token_address, chain, e, exc_info=True)
            raise
        return data if isinstance(data, dict) else None

    async def get_token_report(self, mint: str, *, use_jwt: bool = True, timeout: float = 30.0) -> Optional[dict]:
        """
        Convenience wrapper that prefers the rich /report endpoint then falls back
        to the summary endpoint (/v1/tokens/{mint}) if needed. Returns dict or None.
        """
        # Try canonical rich report first
        try:
            res = await self.get_report(mint, use_jwt=use_jwt, timeout=timeout)
            if res:
                return res
        except Exception:
            logger.debug("get_token_report: report endpoint failed for %s; trying summary", mint, exc_info=True)

        # Fallback to summary
        try:
            res2 = await self.get(f"/v1/tokens/{mint}", use_jwt=use_jwt, timeout=timeout, allow_404=True)
            return res2 if isinstance(res2, dict) else None
        except Exception as e:
            logger.debug("get_token_report: summary endpoint failed for %s: %s", mint, e, exc_info=True)
            return None


def make_rugcheck_client_from_env() -> RugcheckClient:
    api_key = os.getenv("RUGCHECK_API_KEY") or os.getenv("RUGCHECK_KEY")
    client = RugcheckClient(api_key=api_key, max_concurrency=1, min_interval=1.0)
    return client