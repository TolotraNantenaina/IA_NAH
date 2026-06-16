import argparse
import lzma
import os
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MG_CC100_URL = "https://data.statmt.org/cc-100/mg.txt.xz"
_URL_RE = re.compile(r"https?://", re.IGNORECASE)

def line_has_url(s: str) -> bool:
    return bool(_URL_RE.search(s))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "max_rows",
        nargs="?",
        type=int,
        default=50000,
        help="Nombre maximum de lignes à écrire (défaut: 50000)",
    )
    args = parser.parse_args()
    max_rows = args.max_rows

    from config import SCRAPPING_OUTPUT

    out_dir = SCRAPPING_OUTPUT
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cc100_malagasy.txt")

    req = urllib.request.Request(
        MG_CC100_URL,
        headers={"User-Agent": "IA_NAH-fetch-cc100/1.0 (corpus mg; +https://github.com/IA_NAH)"},
    )
    written = 0
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            with lzma.open(resp, "rt", encoding="utf-8", errors="replace") as f:
                with open(out_path, "w", encoding="utf-8") as out:
                    for line in f:
                        line = line.rstrip("\n")
                        if line_has_url(line):
                            continue
                        out.write(line + "\n")
                        written += 1
                        if written % 10000 == 0:
                            print(written)
                        if written >= max_rows:
                            break
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.reason, MG_CC100_URL)
        raise SystemExit(1) from e
    except urllib.error.URLError as e:
        print("URL", e.reason)
        raise SystemExit(1) from e

    print("Écrit:", out_path, "lignes:", written)

if __name__ == "__main__":
    main()
