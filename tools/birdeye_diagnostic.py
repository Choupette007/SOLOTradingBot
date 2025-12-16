#!/usr/bin/env python3
"""
Birdeye diagnostic helper (improved)

Usage (examples):
  # pass URL and key on the command line
  python tools/birdeye_diagnostic.py --url "https://<REAL_BIRDEYE_HOST>/v1/search" --key "YOUR_KEY" --query "solana"

  # or set env vars then run
  set BIRDEYE_API_URL=https://<REAL_BIRDEYE_HOST>/v1/search
  set BIRDEYE_API_KEY=YOUR_KEY
  python tools/birdeye_diagnostic.py --query "solana"

Notes:
- This script will fail fast if no URL is provided (to avoid the placeholder domain).
- Prints request (redacting keys), response status, headers and body for diagnosis.
"""
from __future__ import annotations

import os
import sys
import argparse
import json
from typing import Optional

try:
    import requests
except Exception:
    print("Please install requests: pip install requests", file=sys.stderr)
    sys.exit(2)


REDACT = "REDACTED"


def redact_headers(h: dict) -> dict:
    out = dict(h)
    for k in list(out.keys()):
        if "key" in k.lower() or "authorization" in k.lower() or "token" in k.lower():
            out[k] = REDACT
    return out


def pretty_print_json(text: str):
    try:
        parsed = json.loads(text)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except Exception:
        print(text)


def run_once(url: str, key: Optional[str], query: str, method: str = "post", body_as_json: bool = True, timeout: int = 20):
    headers = {
        "User-Agent": "birdeye-diagnostic/1.0",
    }
    if key:
        headers["Authorization"] = f"Bearer {key}"
        headers["X-API-KEY"] = key

    print("=== Birdeye diagnostic ===")
    print("Request:")
    print("  URL:", url)
    print("  Method:", method.upper())
    print("  Headers (redacted):")
    for k, v in redact_headers(headers).items():
        print("    ", k, ":", v)

    params = {"q": query}
    data = {"query": query, "limit": 100}

    try:
        if method.lower() == "get":
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
        else:
            if body_as_json:
                r = requests.post(url, headers=headers, json=data, timeout=timeout)
            else:
                r = requests.post(url, headers=headers, data=data, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        # Provide clearer troubleshooting advice for DNS/network issues
        msg = str(e)
        print("\nRequest failed: connection error.", file=sys.stderr)
        print("  Exception:", msg, file=sys.stderr)
        print("\nPossible causes:", file=sys.stderr)
        print("  - Invalid/placeholder URL (ensure --url or BIRDEYE_API_URL is set to the real Birdeye endpoint).", file=sys.stderr)
        print("  - Local network/DNS issues (try pinging the hostname or use curl)." , file=sys.stderr)
        print("  - Proxy/firewall blocking outbound traffic.", file=sys.stderr)
        return 3
    except requests.exceptions.Timeout as e:
        print("\nRequest failed: timeout.", file=sys.stderr)
        print("  Exception:", e, file=sys.stderr)
        return 3
    except Exception as e:
        print("\nRequest failed with exception:", e, file=sys.stderr)
        return 3

    print("\nResponse:")
    print("  HTTP", r.status_code)
    print("  Response headers:")
    for k, v in r.headers.items():
        print("    ", k, ":", v)
    print("  Response body:")
    pretty_print_json(r.text)

    # Heuristic inspection
    try:
        j = r.json()
        if isinstance(j, dict):
            arrays = {k: len(v) for k, v in j.items() if isinstance(v, list)}
            if arrays:
                print("\nDetected arrays in JSON response (key -> length):")
                for k, ln in arrays.items():
                    print("  ", k, "->", ln)
    except Exception:
        pass

    if r.status_code != 200:
        print("\nNon-200 response; exit code 4 (diagnostic failed).", file=sys.stderr)
        return 4

    print("\nDiagnostic succeeded (HTTP 200). If the response looks empty, check the API contract, request params, and any account restrictions.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Birdeye diagnostic helper (improved)")
    p.add_argument("--url", "-u", default=os.environ.get("BIRDEYE_API_URL"), help="Birdeye API URL (or set BIRDEYE_API_URL)")
    p.add_argument("--key", "-k", default=os.environ.get("BIRDEYE_API_KEY"), help="Birdeye API key (or set BIRDEYE_API_KEY)")
    p.add_argument("--query", "-q", default="solana", help="Query to send (depends on API)")
    p.add_argument("--method", "-m", choices=("get", "post"), default="post", help="HTTP method to use")
    p.add_argument("--no-json", dest="json_body", action="store_false", help="Send form-encoded body instead of JSON")
    args = p.parse_args(argv)

    if not args.url:
        print("ERROR: No Birdeye URL provided. Please pass --url or set BIRDEYE_API_URL to the real endpoint.", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print('  python tools/birdeye_diagnostic.py --url "https://api.birdeye.your-domain/v1/search" --key "<API_KEY>" --query "solana"', file=sys.stderr)
        sys.exit(2)

    code = run_once(args.url, args.key, args.query, method=args.method, body_as_json=args.json_body)
    sys.exit(code)


if __name__ == "__main__":
    main()