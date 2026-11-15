"""Scan the files git would publish for private data before a commit or push.

    python tools/checks/private_scan.py

Scans every tracked file and every untracked file git does not ignore (text files, notebook sources and outputs) for local
paths, scratch folders, kernel names, e-mail domains and the author's account name outside the lines where it belongs (author,
copyright, licence, repository address). Exit status 1 and a list of hits if anything is found.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ["k" + "Drive", "C:" + chr(92) + "Users", "C:/" + "Users", "c:" + chr(92) + "Users", "c:/" + "Users", "App" + "Data",
           "ANTIMA" + "~1", "jupyter" + "-venv", "@" + "gmail", "scratch" + "pad", "antimajo" + "rana"]   # in pieces: no self-hit
AUTHOR_LINE = re.compile(r"author|copyright|licen[sc]e|github\.com/", re.I)
TEXT_EXT = {".py", ".json", ".txt", ".md", ".toml", ".yml", ".yaml", ".ipynb", ".cfg", ".ini", ".gitignore", ""}


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from strings(x)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("image/png", "image/jpeg"):
                continue
            yield from strings(v)


def files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    return [ROOT / f for f in out if (ROOT / f).is_file()]


def main():
    bad = []
    for f in files():
        if f.suffix not in TEXT_EXT and f.name != "LICENSE":
            continue
        text = f.read_text(encoding="utf8", errors="replace")
        if f.suffix == ".ipynb":
            text = "\n".join(strings(json.loads(text)))
        for n, line in enumerate(text.splitlines(), 1):
            for s in PRIVATE:
                if s in line and not (s == PRIVATE[-1] and AUTHOR_LINE.search(line)):
                    bad.append(f"{f.relative_to(ROOT).as_posix()}:{n}: {s}: {line.strip()[:110]}")
    print(f"scanned {len(files())} files")
    if bad:
        print(f"PRIVATE-DATA SCAN FAILED: {len(bad)} hit(s)")
        print("\n".join(bad[:60]))
        return 1
    print("private-data scan: no hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
