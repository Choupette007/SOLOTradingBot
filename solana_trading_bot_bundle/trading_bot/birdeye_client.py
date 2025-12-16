# SOLOTradingBot\solana_trading_bot_bundle\trading_bot\birdeye_client.py

import asyncio
import time
import logging
import os
import threading
from typing import Optional, Dict, Any, Iterable, List, Tuple

import requests
from email.utils import parsedate_to_datetime

logger = logging.getLogger("birdeye_gate")

# Read env overrides (use whichever you prefer in your .env)
_env_rps = os.environ.get("BIRDEYE_RPS")
_env_rpm = os.environ.get("BIRDEYE_RPM")  # e.g., 900
_DEFAULT_RPM = 900
_DEFAULT_RPS = 15

if _env_rps:
    try:
        DEFAULT_RPS = max(1, int(_env_rps))
    except Exception:
        DEFAULT_RPS = _DEFAULT_RPS
elif _env_rpm:
    try:
        DEFAULT_RPS = max(1, int(int(_env_rpm) / 60))
    except Exception:
        DEFAULT_RPS = _DEFAULT_RPS
else:
    DEFAULT_RPS = _DEFAULT_RPS

# Sanity cap to avoid runaway configs
DEFAULT_RPS = min(DEFAULT_RPS, 1000)

# Feature-flag: allow multi_price endpoint.
# Default: true (existing behavior). In tester/free envs set BIRDEYE_ALLOW_MULTI_PRICE=false
_ENV_ALLOW_MULTI = os.environ.get("BIRDEYE_ALLOW_MULTI_PRICE", "true").lower()
_ALLOW_MULTI_PRICE_DEFAULT = _ENV_ALLOW_MULTI in ("1", "true", "yes")


def parse_retry_after_seconds(headers: Optional[Dict[str, Any]]) -> float:
    """
    Parse Retry-After header (seconds or HTTP-date) into seconds remaining.
    Returns 0.0 if not present or not parseable.
    """
    if not headers:
        return 0.0
    ra = headers.get("Retry-After") or headers.get("retry-after")
    if not ra:
        return 0.0
    try:
        # numeric seconds
        return float(ra)
    except Exception:
        try:
            # HTTP-date
            dt = parsedate_to_datetime(ra)
            if dt:
                remaining = dt.timestamp() - time.time()
                return max(0.0, remaining)
        except Exception:
            pass
    return 0.0


class BirdeyeGate:
    """
    Cooperative rate gate that supports both:
      - async usage via `async with birdeye_gate.request():`
      - sync/blocking usage via `birdeye_gate.sync_acquire()` (for thread-based callers)

    Implementation details:
      - Uses a monotonic sliding window (timestamps in self._recent)
      - Uses a threading.Lock for internal state so sync callers and sync_acquire
        are straightforward. Async request() offloads blocking acquire to a thread.
      - Provides sync_mark_down/sync_note_429 to allow synchronous callers to
        cooperatively mark the gate down when a 429 is observed.
    """

    def __init__(self, rps: int = DEFAULT_RPS):
        self._rps = max(1, int(rps))
        # Protect shared state with a threading.Lock so sync callers can safely mutate it.
        self._state_lock = threading.Lock()
        # timestamps (monotonic) of recent requests within last 1s
        self._recent: list[float] = []
        # monotonic timestamp until which gate is down (cooperative backoff)
        self._down_until: float = 0.0

        # Telemetry counters
        self.requests_total = 0
        self._429_total = 0

    # --------------------
    # Sync (blocking) API
    # --------------------
    def _prune_recent(self, now: float) -> None:
        cutoff = now - 1.0
        # keep only timestamps within last 1s
        self._recent = [t for t in self._recent if t >= cutoff]

    def sync_acquire(self) -> None:
        """
        Blocking acquire for synchronous callers (e.g., requests in a thread).
        Will sleep as needed to respect rps and cooperative down_until.
        """
        while True:
            with self._state_lock:
                now = time.monotonic()
                # if gate marked down, compute wait and do a bounded sleep outside lock
                if now < self._down_until:
                    wait = self._down_until - now
                    to_sleep = min(wait, 1.0)
                    # release lock, then sleep
                else:
                    # refill window by pruning stale entries and allow if capacity
                    self._prune_recent(now)
                    if len(self._recent) < self._rps:
                        self._recent.append(now)
                        self.requests_total += 1
                        return
                    # not enough capacity, compute small wait
                    to_sleep = 0.02
            # sleep outside lock to allow other threads to make progress
            time.sleep(to_sleep)

    def sync_mark_down(self, seconds: float) -> None:
        """
        Mark the gate down for `seconds` (cooperative backoff). Uses monotonic clock.
        Safe to call from any thread.
        """
        with self._state_lock:
            prev = self._down_until
            now = time.monotonic()
            self._down_until = max(self._down_until, now + max(0.0, float(seconds)))
        logger.warning("Birdeye gate (sync): marked down for %.1f s (prev_down_until=%.3f)", seconds, prev)

    def sync_note_429(self) -> None:
        with self._state_lock:
            self._429_total += 1

    # ---------------------
    # Async (non-blocking) API
    # ---------------------
    async def request(self):
        """
        Async context manager you can 'async with' before making an HTTP call.
        It will offload a blocking acquire to a thread so the event loop isn't blocked.
        Usage:
          async with birdeye_gate.request():
              resp = await session.get(...)
        """
        gate = self

        class _Ctx:
            def __init__(self, outer: "BirdeyeGate"):
                self._outer = outer
                self._entered = False

            async def __aenter__(self):
                # Offload the blocking acquire to a thread to avoid blocking the loop.
                await asyncio.to_thread(self._outer.sync_acquire)
                self._entered = True
                return self

            async def __aexit__(self, exc_type, exc, tb):
                # nothing to do on exit for now
                return False

        return _Ctx(gate)

    async def mark_down(self, seconds: float) -> None:
        """
        Async-friendly wrapper around sync_mark_down.
        """
        await asyncio.to_thread(self.sync_mark_down, float(seconds))

    async def note_429(self) -> None:
        """
        Async-friendly wrapper around sync_note_429.
        """
        await asyncio.to_thread(self.sync_note_429)

    # ---------------------
    # Diagnostics
    # ---------------------
    def stats(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "requests_total": int(self.requests_total),
                "429_total": int(self._429_total),
                "rps": int(self._rps),
                "recent_len": len(self._recent),
                "down_until_monotonic": float(self._down_until),
            }


# module-level gate instance (import and reuse)
birdeye_gate = BirdeyeGate()


# ---------------------------
# BirdeyeClient (singleton)
# ---------------------------
MASK = "REDACTED"
_DEFAULT_BASE_URL = "https://public-api.birdeye.so"


def _redact_headers(headers: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in headers.items():
        if "key" in k.lower() or "authorization" in k.lower() or "token" in k.lower():
            out[k] = MASK
        else:
            out[k] = v
    return out


class BirdeyeClient:
    """
    Birdeye client - v3 endpoints (token list, price, multi_price) with simple rate-limiting,
    Retry-After handling, cooperative "down until" backoff and a tiny in-memory TTL cache.

    This client cooperates with the module-level birdeye_gate singleton.
    """

    def __init__(self, api_key: str, base_url: str = _DEFAULT_BASE_URL, timeout: int = 10, rps_limit: int = DEFAULT_RPS, cache_ttl: float = 30.0):
        if not api_key:
            raise ValueError("Birdeye API key is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # NOTE: We no longer instantiate an internal RateLimiter here by default.
        # Sync coordination with other modules is achieved via the shared birdeye_gate.
        self._cache_ttl = float(cache_ttl)
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_lock = threading.Lock()

        # Feature flags / state
        # If set false via env, multi_price calls will be short-circuited.
        self.allow_multi_price = _ALLOW_MULTI_PRICE_DEFAULT
        # If we observe a 401 for multi_price we set this to avoid repeated calls/log floods.
        self._multi_price_disabled = False

        # Last response status (helpful for callers that want to react to 401/403)
        self._last_response_status: Optional[int] = None
        self._last_response_body: Optional[str] = None

    def _headers(self, chain: str) -> Dict[str, str]:
        return {
            "User-Agent": "SolanaTradingBot/1.0 birdeye-client",
            "X-API-KEY": self.api_key,
            "x-chain": chain,
        }

    def _safe_json(self, resp: requests.Response) -> Optional[Dict[str, Any]]:
        ctype = resp.headers.get("Content-Type", "")
        if "application/json" not in ctype:
            logger.error("Birdeye returned non-JSON content-type: %s (url=%s)", ctype, resp.url)
            return None
        try:
            return resp.json()
        except Exception as e:
            logger.exception("Failed to parse Birdeye JSON for %s: %s", resp.url, e)
            return None

    def _log_resp(self, method: str, url: str, req_headers: Dict[str, Any], resp: requests.Response):
        try:
            body_preview = resp.text[:4096]
        except Exception:
            body_preview = "<unreadable>"
        logger.info("Birdeye %s %s -> %s (rate remaining=%s/%s)",
                    method, url, resp.status_code, resp.headers.get("x-ratelimit-remaining"), resp.headers.get("x-ratelimit-limit"))
        if resp.status_code == 429:
            logger.warning("Birdeye 429 for %s %s; redacted request headers=%s; body=%s",
                           method, url, _redact_headers(req_headers), body_preview)
        elif resp.status_code != 200:
            logger.warning("Birdeye non-200 response for %s %s; redacted request headers=%s; body=%s",
                           method, url, _redact_headers(req_headers), body_preview)
        else:
            logger.debug("Birdeye response for %s %s headers=%s body_preview=%s",
                         method, url, dict(resp.headers), body_preview)

    def _do_request_with_rate(self, func, url: str, headers: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Common wrapper used by _get and _post to centralize 429 handling / mark_down.
        This implementation will use the shared birdeye_gate if available to cooperatively
        respect RPS and mark-down behavior across the process.
        """
        # Cooperative acquire (sync/blocking)
        try:
            if birdeye_gate is not None:
                birdeye_gate.sync_acquire()
            else:
                # No shared gate available: do a conservative sleep to avoid bursts
                time.sleep(0.02)
        except Exception:
            # Best-effort; don't crash callers if gate fails
            logger.debug("birdeye_gate.sync_acquire failed (continuing without cooperative gate)", exc_info=True)

        try:
            resp: requests.Response = func(url, headers=headers, timeout=self.timeout, **kwargs)
            # capture last response info so callers can inspect status/body if needed
            try:
                self._last_response_status = resp.status_code
                self._last_response_body = resp.text[:4096]
            except Exception:
                self._last_response_body = None

            # Log + handle 429
            self._log_resp("GET/POST", resp.url, headers, resp)
            if resp.status_code == 429:
                # Inform shared gate if present
                if birdeye_gate is not None:
                    birdeye_gate.sync_note_429()
                # parse Retry-After if present; fallback to short pause
                ra = parse_retry_after_seconds(resp.headers) or 1.0
                ra = max(1.0, min(ra, 300.0))
                # cooperatively mark the gate down for RA seconds (if available)
                try:
                    if birdeye_gate is not None:
                        birdeye_gate.sync_mark_down(ra)
                except Exception:
                    logger.debug("birdeye_gate.sync_mark_down failed", exc_info=True)
                return None

            return self._safe_json(resp)
        except requests.RequestException as e:
            logger.warning("Birdeye request failed: %s %s", url, e, exc_info=True)
            # clear last response status on exception
            self._last_response_status = None
            self._last_response_body = None
            return None

    def _get(self, path: str, params: Dict[str, Any], chain: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        headers = self._headers(chain)
        logger.debug("Birdeye GET %s params=%s headers=%s", url, params, _redact_headers(headers))
        return self._do_request_with_rate(requests.get, url, headers, params=params)

    def _post(self, path: str, body: Dict[str, Any], chain: str) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        headers = self._headers(chain)
        headers["Content-Type"] = "application/json"
        logger.debug("Birdeye POST %s body=%s headers=%s", url, body, _redact_headers(headers))
        return self._do_request_with_rate(requests.post, url, headers, json=body)

    # -----------------------
    # Cache helpers
    # -----------------------
    def _get_cached(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._cache_lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            expires_at, value = entry
            if now >= expires_at:
                del self._cache[key]
                return None
            return value

    def _set_cached(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = self._cache_ttl if ttl is None else float(ttl)
        with self._cache_lock:
            self._cache[key] = (time.time() + ttl, value)

    # ---------------------------------------------------------------------
    # Public API methods (v3) - unchanged semantics
    # ---------------------------------------------------------------------
    def token_list(self, q: str = "", chain: str = "solana", limit: int = 100, page: int = 1) -> List[Dict[str, Any]]:
        path = "/defi/v3/token/list"
        params = {"q": q, "limit": limit, "page": page}
        j = self._get(path, params=params, chain=chain)
        if not j:
            return []
        data = j.get("data") or j
        items = data.get("items") if isinstance(data, dict) else None
        out: List[Dict[str, Any]] = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it.get("result"), list):
                    out.extend(it.get("result", []))
        if not out:
            for key in ("tokens", "results", "items"):
                cand = data.get(key)
                if isinstance(cand, list):
                    out.extend(cand)
        if not out and isinstance(j.get("data"), list):
            out.extend(j.get("data"))
        logger.info("Birdeye token_list parsed %d entries for q=%s", len(out), q)
        return out

    def price(self, address: str, chain: str = "solana", cache_ttl: Optional[float] = None) -> Optional[Dict[str, Any]]:
        cache_key = f"price:{chain}:{address}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        path = "/defi/price"
        params = {"address": address}
        j = self._get(path, params=params, chain=chain)
        if j and isinstance(j, dict):
            data = j.get("data") or j
            if isinstance(data, dict):
                self._set_cached(cache_key, data, ttl=cache_ttl)
                return data
        body = {"address": address}
        j2 = self._post(path, body=body, chain=chain)
        if j2:
            data2 = j2.get("data") or j2
            self._set_cached(cache_key, data2, ttl=cache_ttl)
            return data2
        return None

    def multi_price(self, addresses: Iterable[str], chain: str = "solana", cache_ttl: Optional[float] = None) -> Dict[str, Optional[Dict[str, Any]]]:
        # Runtime env check: allow operators to disable multi_price dynamically via env var
        try:
            env_val = os.getenv("BIRDEYE_ALLOW_MULTI_PRICE")
            if env_val is not None and str(env_val).strip().lower() in ("0", "false", "no"):
                logger.info("Birdeye multi_price disabled by BIRDEYE_ALLOW_MULTI_PRICE env (runtime). Returning cached-only results.")
                result = {}
                for a in addresses:
                    result[a] = self._get_cached(f"price:{chain}:{a}") or None
                return result
        except Exception:
            # if env check fails for any reason, continue to existing logic
            pass

        # Existing instance-level guards (env-set at init and 401-based _multi_price_disabled)
        if not self.allow_multi_price:
            logger.info("Birdeye multi_price disabled by client config (allow_multi_price=False). Returning cached-only results.")
            result = {}
            for a in addresses:
                result[a] = self._get_cached(f"price:{chain}:{a}") or None
            return result
        if self._multi_price_disabled:
            logger.info("Birdeye multi_price disabled due to prior 401. Returning cached-only results.")
            result = {}
            for a in addresses:
                result[a] = self._get_cached(f"price:{chain}:{a}") or None
            return result

        addrs = list(addresses)
        result: Dict[str, Optional[Dict[str, Any]]] = {}
        missing = []
        for a in addrs:
            cached = self._get_cached(f"price:{chain}:{a}")
            if cached:
                result[a] = cached
            else:
                missing.append(a)
        if not missing:
            return result

        path = "/defi/multi_price"
        comma = ",".join(missing)

        def _apply_data_to_result(data_obj: Any) -> None:
            # Accept either dict mapping addr->entry or list of entries
            if isinstance(data_obj, dict):
                # mapping address -> details?
                mapped = True
                for k, v in data_obj.items():
                    if not isinstance(v, dict):
                        mapped = False
                        break
                if mapped:
                    for k, v in data_obj.items():
                        result[k] = v
                        self._set_cached(f"price:{chain}:{k}", v, ttl=cache_ttl)
                    return
                prices_list = data_obj.get("prices") if isinstance(data_obj, dict) else None
                if isinstance(prices_list, list):
                    for entry in prices_list:
                        if not isinstance(entry, dict):
                            continue
                        addr = entry.get("address") or entry.get("token") or entry.get("id")
                        if addr:
                            result[addr] = entry
                            self._set_cached(f"price:{chain}:{addr}", entry, ttl=cache_ttl)
                    return
            if isinstance(data_obj, list):
                for entry in data_obj:
                    if not isinstance(entry, dict):
                        continue
                    addr = entry.get("address") or entry.get("token") or entry.get("id")
                    if addr:
                        result[addr] = entry
                        self._set_cached(f"price:{chain}:{addr}", entry, ttl=cache_ttl)

        # 1) Try GET with list_address param first (matches public gateway / curl)
        try:
            j = self._get(path, params={"list_address": comma}, chain=chain)
        except Exception:
            j = None

        # If the HTTP layer recorded a 401, mark multi_price disabled to avoid loops.
        if self._last_response_status == 401:
            logger.warning("Birdeye multi_price returned 401 (plan/auth). Disabling multi_price for process until operator intervention.")
            self._multi_price_disabled = True
            # return cached-only results
            for a in missing:
                result[a] = None
            return result

        if j:
            data = j.get("data") or j
            _apply_data_to_result(data)
        else:
            # 2) Try POST with list_address string (some gateways accept this)
            try:
                j = self._post(path, body={"list_address": comma}, chain=chain)
            except Exception:
                j = None

            if self._last_response_status == 401:
                logger.warning("Birdeye multi_price returned 401 (plan/auth) on POST. Disabling multi_price for process until operator intervention.")
                self._multi_price_disabled = True
                for a in missing:
                    result[a] = None
                return result

            if j:
                data = j.get("data") or j
                _apply_data_to_result(data)
            else:
                # 3) Fallback: POST with addresses array (legacy)
                try:
                    j = self._post(path, body={"addresses": missing}, chain=chain)
                except Exception:
                    j = None

                if self._last_response_status == 401:
                    logger.warning("Birdeye multi_price returned 401 (plan/auth) on POST addresses. Disabling multi_price for process until operator intervention.")
                    self._multi_price_disabled = True
                    for a in missing:
                        result[a] = None
                    return result

                if j:
                    data = j.get("data") or j
                    _apply_data_to_result(data)

        # Ensure every requested address has an entry (cached or None)
        for a in addrs:
            result.setdefault(a, self._get_cached(f"price:{chain}:{a}") or None)

        return result

    # ---------------------------------------------------------------------
    # Optional helpers / diagnostics
    # ---------------------------------------------------------------------
    def ping(self) -> bool:
        path = "/"
        try:
            resp = requests.get(self.base_url + path, timeout=self.timeout)
            logger.info("Birdeye ping %s -> %s", self.base_url + path, resp.status_code)
            return resp.status_code == 200
        except requests.RequestException as e:
            logger.warning("Birdeye ping failed: %s", e, exc_info=True)
            return False

    def stats(self) -> Dict[str, Any]:
        """Expose cache stats and shared gate stats when available"""
        s: Dict[str, Any] = {}
        with self._cache_lock:
            s["cache_size"] = len(self._cache)
        if birdeye_gate is not None:
            try:
                s["gate"] = birdeye_gate.stats()
            except Exception:
                s["gate"] = {}
        # expose multi_price state
        s["multi_price_allowed_by_env"] = bool(self.allow_multi_price)
        s["multi_price_disabled_due_to_401"] = bool(self._multi_price_disabled)
        s["last_response_status"] = int(self._last_response_status) if self._last_response_status is not None else None
        return s

    def enable_multi_price_after_operator(self) -> None:
        """
        Call this after upgrading Birdeye plan or rotating API key to re-enable
        multi_price usage in this process.
        """
        self._multi_price_disabled = False
        logger.info("Birdeye multi_price re-enabled by operator intervention.")


# --- Module-level helpers to programmatically toggle multi_price at runtime ---
def set_multi_price_enabled(enabled: bool) -> None:
    """
    Programmatically enable/disable multi_price for the module-level client.
    Also updates the BIRDEYE_ALLOW_MULTI_PRICE env var for consistency.
    """
    try:
        os.environ["BIRDEYE_ALLOW_MULTI_PRICE"] = "1" if enabled else "0"
    except Exception:
        # best-effort; ignore env write failures
        pass
    global _birdeye_client
    try:
        if _birdeye_client is not None:
            _birdeye_client.allow_multi_price = bool(enabled)
    except Exception:
        logger.debug("Failed to set allow_multi_price on module client", exc_info=True)


def disable_multi_price() -> None:
    """Convenience: disable multi_price at runtime."""
    set_multi_price_enabled(False)


def enable_multi_price() -> None:
    """Convenience: enable multi_price at runtime."""
    set_multi_price_enabled(True)


# Module-level singleton client management
_birdeye_client: Optional[BirdeyeClient] = None
_birdeye_client_lock = threading.Lock()


def get_birdeye_client(api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: int = 10, rps_limit: int = DEFAULT_RPS, cache_ttl: float = 30.0) -> BirdeyeClient:
    """
    Return the process-wide singleton BirdeyeClient. If not yet initialized, create it.

    - If a singleton already exists, it is returned and provided arguments are ignored.
    - If no singleton exists, `api_key` must be provided (or environment variable BIRDEYE_API_KEY).
    - base_url falls back to BIRDEYE_BASE_URL env var (if present) then to module default.
    - If rps_limit is provided, apply it to the shared birdeye_gate.
    """
    global _birdeye_client
    with _birdeye_client_lock:
        if _birdeye_client is not None:
            return _birdeye_client

        key = api_key or os.environ.get("BIRDEYE_API_KEY")
        if not key:
            raise ValueError("Birdeye API key is required to initialize client. Provide api_key or set BIRDEYE_API_KEY env var.")

        # Prefer explicit arg, then env override, then default constant
        env_base = os.environ.get("BIRDEYE_BASE_URL")
        base = base_url or (env_base.rstrip("/") if env_base else None) or _DEFAULT_BASE_URL

        # Instantiate client
        _birdeye_client = BirdeyeClient(api_key=key, base_url=base, timeout=timeout, rps_limit=rps_limit, cache_ttl=cache_ttl)
        logger.info("Initialized module-level birdeye client (base_url=%s)", base)

        # Apply provided rps_limit to the shared gate (best-effort)
        try:
            if isinstance(rps_limit, int) and rps_limit > 0:
                birdeye_gate._rps = max(1, int(rps_limit))
        except Exception:
            logger.debug("Failed to apply rps_limit to birdeye_gate (ignored)", exc_info=True)

        return _birdeye_client


def set_birdeye_client(client: Optional[BirdeyeClient]) -> None:
    """
    Replace or clear the module-level BirdeyeClient (for tests or manual wiring).
    """
    global _birdeye_client
    with _birdeye_client_lock:
        _birdeye_client = client
        logger.info("Module-level birdeye client set to %s", "None" if client is None else "provided instance")