"""
monitor_route.py
Polls Jupiter (lite-api or configured base) for a route and optionally auto-whitelists when route appears.
Usage: python monitor_route.py <mint> [--auto-whitelist]
"""
import os
import time
import requests
import sqlite3
import sys
import json

# Prefer an env override, then try to reuse package token_cache_path when available,
# otherwise fall back to the previous hard-coded Windows path.
try:
    DB_PATH = os.getenv("SOLO_TOKENS_DB") or os.getenv("JUP_MONITOR_DB")
    if not DB_PATH:
        try:
            from solana_trading_bot_bundle.common.constants import token_cache_path  # type: ignore
            DB_PATH = str(token_cache_path())
        except Exception:
            DB_PATH = os.path.expanduser(r"C:\Users\Admin\AppData\Local\SOLOTradingBot\tokens.sqlite3")
except Exception:
    DB_PATH = os.path.expanduser(r"C:\Users\Admin\AppData\Local\SOLOTradingBot\tokens.sqlite3")

JUP_BASE = os.getenv("JUPITER_API_BASE", "https://lite-api.jup.ag")
HEADERS = {"User-Agent": "SOLOTradingBot/monitor/1.0", "Accept": "application/json"}
if os.getenv("JUPITER_API_KEY"):
    HEADERS["x-api-key"] = os.getenv("JUPITER_API_KEY")


def check_route(mint, out='So11111111111111111111111111111111111111112', amount=1000):
    uri = f"{JUP_BASE}/quote?inputMint={mint}&outputMint={out}&amount={amount}&slippage=100"
    try:
        r = requests.get(uri, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            js = r.json()
            # Jupiter lite API returns data key (list of routes) under 'data' or 'data'->'data' in some versions.
            if js.get("data"):
                return True, js
        return False, None
    except Exception:
        return False, None


def whitelist_mint(mint):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    # tolerant schema modifications: add is_trading column if missing
    try:
        cur.execute("PRAGMA table_info(tokens);")
        cols = [r[1] for r in cur.fetchall()]
        if 'is_trading' not in cols:
            try:
                cur.execute("ALTER TABLE tokens ADD COLUMN is_trading INTEGER DEFAULT 0;")
                con.commit()
            except Exception:
                # ignore alter failures on locked DBs/schema mismatch
                pass
    except Exception:
        pass

    try:
        cur.execute("SELECT 1 FROM tokens WHERE address=?", (mint,))
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO tokens (address,name,symbol,volume_24h,liquidity,market_cap,price,categories,timestamp,is_trading) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (mint, 'AUTO-MONITOR-WHITELIST', 'AUTO', 0.0, 0.0, 0.0, 0.0, '[]', int(time.time()), 1)
            )
            con.commit()
        else:
            cur.execute("UPDATE tokens SET is_trading=1 WHERE address=?", (mint,))
            con.commit()
    except Exception:
        # Best-effort: if table or schema differs, try a simple upsert into a generic tokens table.
        try:
            cur.execute("INSERT OR REPLACE INTO tokens (address, is_trading) VALUES (?, ?);", (mint, 1))
            con.commit()
        except Exception:
            pass
    finally:
        try:
            con.close()
        except Exception:
            pass


def run(mint, auto_whitelist=False, interval=30):
    print("Monitoring", mint, "JUP_BASE=", JUP_BASE, "DB_PATH=", DB_PATH)
    while True:
        ok, js = check_route(mint)
        if ok:
            print("ROUTE FOUND at", time.strftime("%Y-%m-%d %H:%M:%S"), "for", mint)
            # save route json for inspection (use JSON formatting)
            try:
                with open("route_found.json", "w", encoding="utf-8") as f:
                    json.dump(js, f, indent=2)
            except Exception:
                try:
                    open("route_found.json", "w").write(str(js))
                except Exception:
                    pass
            if auto_whitelist:
                print("Auto-whitelist enabled: inserting tokens row")
                whitelist_mint(mint)
            break
        else:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "no route yet")
        time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_route.py <mint> [--auto-whitelist]")
        sys.exit(2)
    mint = sys.argv[1]
    auto = ("--auto-whitelist" in sys.argv)
    run(mint, auto_whitelist=auto)