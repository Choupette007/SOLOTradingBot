# tools/sanitize_env.py
# Strips inline comments from a .env file while preserving quotes and full-line comments.
# Usage: python tools/sanitize_env.py [path_to_env]; defaults to AppData\SOLOTradingBot\.env

import sys, pathlib

def strip_inline_comment(line: str) -> str:
    s = line.strip()
    if not s or s.startswith("#"):
        return line  # keep blank lines and full-line comments
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
            break  # remove anything after unquoted '#'
        else:
            out.append(ch)
    return "".join(out).rstrip()

def main():
    default_env = pathlib.Path.home() / "AppData" / "Local" / "SOLOTradingBot" / ".env"
    p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else default_env
    if not p.exists():
        print(f"[sanitize_env] No .env at {p}; nothing to sanitize.")
        return
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    cleaned = [strip_inline_comment(l) for l in lines]
    p.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    print(f"[sanitize_env] Sanitized: {p}")

if __name__ == "__main__":
    main()





