#!/usr/bin/env python3
"""
tools/check_jupiter_endpoint.py

Quick utility to verify the Jupiter endpoint configured in config.yaml (or via env)
and to exercise the /quote (swap) route to confirm the correct API is reachable.

Usage:
  python tools/check_jupiter_endpoint.py               # reads config.yaml in cwd
  python tools/check_jupiter_endpoint.py --config /path/to/config.yaml
  python tools/check_jupiter_endpoint.py --mint <MINT> --out <OUT_MINT>

Notes:
 - The script only performs safe HTTP GETs and prints diagnostics.
 - It does not mutate your DB or config.
"""
from __future__ import annotations
import argparse
import os
import sys
import json
import requests
from typing import Optional

try:
    import yaml
except Exception:
    yaml = None  # fallback: try to parse minimally if needed

# Common Solana tokens we can use for a smoke quote
DEFAULT_USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
DEFAULT_WSOL = "So11111111111111111111111111111111111111112"  # WSOL

def load_config(path: Optional[str]) -> dict:
    # Try YAML first
    if path and os.path.isfile(path):
        if yaml:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        else:
            # Minimal parse: look for jupiter.base_url manually
            txt = open(path, "r", encoding="utf-8").read()
            return {"_raw": txt}
    # fallback: try environment variables
    return {}

def find_jupiter_base(cfg: dict) -> Optional[str]:
    # Prefer config value if present
    try:
        j = cfg.get("jupiter")
        if isinstance(j, dict):
            base = j.get("base_url") or j.get("api_base") or j.get("base")
            if base:
                return str(base).strip()
    except Exception:
        pass
    # Try top-level env
    env = os.getenv("JUPITER_API_BASE") or os.getenv("JUP_API") or os.getenv("JUP_BASE")
    if env:
        return env
    # No config found
    return None

def try_get(url: str, headers=None, timeout=8):
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        return r
    except Exception as e:
        return e

def main():
    p = argparse.ArgumentParser(description="Check Jupiter endpoint reachability and quote route.")
    p.add_argument("--config", "-c", help="Path to config.yaml (default: ./config.yaml)")
    p.add_argument("--mint", help="Input mint to test (default: USDC)", default=DEFAULT_USDC)
    p.add_argument("--out", help="Output mint to test (default: WSOL)", default=DEFAULT_WSOL)
    p.add_argument("--amount", help="Amount (raw units) to request for quote (default: 1000)", default="1000")
    p.add_argument("--show-json", action="store_true", help="Show returned JSON if any")
    args = p.parse_args()

    cfg_path = args.config or os.path.join(os.getcwd(), "config.yaml")
    cfg = load_config(cfg_path)

    base = find_jupiter_base(cfg)
    if not base:
        print("No Jupiter base URL found in config.yaml or env. Please set jupiter.base_url or JUPITER_API_BASE.")
        print("Common correct value for the swap API is: https://lite-api.jup.ag/swap/v1")
        sys.exit(2)

    base = base.rstrip("/")
    print("Using Jupiter base_url:", base)

    # 1) Try a GET on the base url
    print("\n1) GET base URL")
    r = try_get(base)
    if isinstance(r, Exception):
        print("  ERROR reaching base URL:", r)
    else:
        print(f"  HTTP {r.status_code} from {base}")
        if r.headers.get("content-type","").lower().startswith("application/json"):
            try:
                js = r.json()
                print("  JSON keys:", list(js.keys())[:10])
                if args.show_json:
                    print(json.dumps(js, indent=2))
            except Exception:
                print("  Response body (first 400 chars):")
                print(r.text[:400])
        else:
            print("  Response body (first 400 chars):")
            print((r.text or "")[:400])

    # 2) Try the quote route commonly used by Jupiter swap API:
    #    {base}/quote?inputMint=...&outputMint=...&amount=...&slippage=...
    quote_path = f"{base}/quote"
    quote_url = f"{quote_path}?inputMint={args.mint}&outputMint={args.out}&amount={args.amount}&slippage=1"
    print("\n2) GET quote endpoint (smoke test)")
    print("  Requesting:", quote_url)
    r2 = try_get(quote_url)
    if isinstance(r2, Exception):
        print("  ERROR reaching quote endpoint:", r2)
    else:
        print(f"  HTTP {r2.status_code} from quote endpoint")
        ct = r2.headers.get("content-type","")
        if "application/json" in ct:
            try:
                js2 = r2.json()
                print("  Returned JSON top-level keys:", list(js2.keys())[:10])
                # quick heuristic: check for 'data' or 'routes' presence
                if isinstance(js2, dict) and ("data" in js2 or "routes" in js2 or "price" in js2):
                    print("  Looks like a Jupiter-style JSON response.")
                else:
                    print("  JSON returned but structure not obviously Jupiter 'quote' shape.")
                if args.show_json:
                    print(json.dumps(js2, indent=2)[:2000])
            except Exception as e:
                print("  Failed to parse JSON:", e)
                print("  Response text (first 400 chars):")
                print(r2.text[:400])
        else:
            print("  Non-JSON response; first 800 chars:")
            print((r2.text or "")[:800])

    # 3) If base contained 'quote-api.jup.ag' recommend swap API
    if "quote-api.jup.ag" in base or "quote-api" in base:
        print("\nNOTE: The configured base_url contains 'quote-api.jup.ag' which may be deprecated for swap operations.")
        print("Recommended swap/base endpoint: https://lite-api.jup.ag/swap/v1")
        print("If your code expects a /quote route under a different host, update jupiter.base_url in config.yaml.")

    print("\nRepository scan suggestion:")
    print("  Run these commands in your repo to find other references to jupiter endpoints:")
    print("    grep -R \"jup.ag\" . || true")
    print("    grep -R \"quote-api.jup.ag\" . || true")
    print("    rg \"jup.ag\" || true   # if ripgrep installed")

    print("\nIf the quote endpoint returned 200 and JSON with 'data' or 'routes', the configured base is likely correct.")
    print("If you got 404/405 or non-JSON or an unexpected HTML response, switch base_url to https://lite-api.jup.ag/swap/v1 and re-run this check.")

if __name__ == "__main__":
    main()