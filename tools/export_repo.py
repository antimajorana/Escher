"""Copy a clean, publishable tree of papertilling to another folder (the working copy is never a git repository).

    python tools/export_repo.py [TARGET]        default TARGET: ~/Documents/GitHub/papertilling

Re-runnable: first deletes what TARGET/.export_manifest lists (the previous run), never touches TARGET/.git. Included: papertilling/ (.py .json .txt), tests/, the three notebooks, tools/ (without
build_colab.py, vendor_legacy.py, bench/, *.out), pyproject.toml, README.md, LICENSE, CHANGELOG.md, .gitignore, .github/, docs/.
legacy/SOURCES.txt is rewritten without local paths. At the end every text file is scanned for private strings; a hit in a
notebook OUTPUT drops that output only, any other hit is listed and the exit status is 1 (the tree is still written).
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = Path.home() / "Documents" / "GitHub" / "papertilling"
MANIFEST = ".export_manifest"
TOP_FILES = ["pyproject.toml", "README.md", "LICENSE", "CHANGELOG.md", ".gitignore"]
TOP_DIRS = [".github", "docs"]
NOTEBOOKS = ["1. Lattices of a wallpaper group.ipynb", "2. Defects of a lattice.ipynb", "3. Defect configurations.ipynb"]
TOOLS_SKIP_NAMES = {"build_colab.py", "vendor_legacy.py"}
PRIVATE = ["k" + "Drive", "C:" + chr(92) + "Users", "C:/" + "Users", "c:" + chr(92) + "Users", "App" + "Data", "jupyter" + "-venv",
           "@" + "gmail", "antimajo" + "rana"]       # written in pieces so that this file does not trip its own scan
AUTHOR_LINE = re.compile(r"author|copyright|licen[sc]e|github.com/", re.I)      # lines where the author name may stand (also the repository address)
AUTHOR_FILES = {"pyproject.toml", "README.md", "LICENSE", "CHANGELOG.md"}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
TEXT_EXT = {".py", ".json", ".txt", ".md", ".toml", ".yml", ".yaml", ".ipynb", ".cfg", ".ini", ".gitignore", ""}


def keep(p, root):
    rel = p.relative_to(root)
    return p.is_file() and not (set(rel.parts) & SKIP_PARTS) and p.suffix != ".pyc" and not any(
        part.endswith(".egg-info") for part in rel.parts)


def collect():
    """(source path, destination relative path) for everything that is exported."""
    out = []
    for p in sorted((ROOT / "papertilling").rglob("*")):
        if keep(p, ROOT) and p.suffix in {".py", ".json", ".txt"}:
            out.append((p, p.relative_to(ROOT)))
    for p in sorted((ROOT / "tests").rglob("*")):
        if keep(p, ROOT) and p.suffix in {".py", ".json", ".txt"}:
            out.append((p, p.relative_to(ROOT)))
    for n in NOTEBOOKS:
        out.append((ROOT / "notebooks" / n, Path("notebooks") / n))
    for p in sorted((ROOT / "tools").rglob("*")):
        rel = p.relative_to(ROOT / "tools")
        if keep(p, ROOT) and rel.parts[0] != "bench" and p.name not in TOOLS_SKIP_NAMES and p.suffix != ".out":
            out.append((p, p.relative_to(ROOT)))
    for n in TOP_FILES:
        out.append((ROOT / n, Path(n)))
    for d in TOP_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if keep(p, ROOT):
                out.append((p, p.relative_to(ROOT)))
    return out


def clean_previous(target):
    mf = target / MANIFEST
    if mf.exists():
        for line in mf.read_text(encoding="utf8").splitlines():
            f = target / line
            if line and ".git" not in Path(line).parts and f.is_file():
                f.unlink()
        for d in sorted((p for p in target.rglob("*") if p.is_dir() and ".git" not in p.relative_to(target).parts),
                        key=lambda q: -len(q.parts)):
            if not any(d.iterdir()):
                d.rmdir()


def hits(text):
    return [s for s in PRIVATE if s in text]


def strings(o):
    """All decoded strings inside a JSON value (so that backslashes are not seen in their escaped form)."""
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from strings(k)
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def scrub_notebook(path, report):
    nb = json.loads(path.read_text(encoding="utf8"))
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        keep_out = []
        for o in c.get("outputs", []):
            h = hits(chr(10).join(strings(o)))
            if h:
                report.append(f"dropped an output of cell {i} in {path.name} (contained {h})")
            else:
                keep_out.append(o)
        c["outputs"] = keep_out
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf8")


def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    target.mkdir(parents=True, exist_ok=True)
    clean_previous(target)
    files = collect()
    for src, rel in files:
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    sources = target / "papertilling" / "legacy" / "SOURCES.txt"
    if sources.exists():
        lines = [re.sub(r"\s+from .*$", "", ln) for ln in sources.read_text(encoding="utf8").splitlines()]
        sources.write_text("\n".join(lines) + "\n", encoding="utf8")
    (target / MANIFEST).write_text("\n".join(sorted(r.as_posix() for _, r in files) + [MANIFEST]) + "\n", encoding="utf8")
    notes, bad = [], []
    for _, rel in files:
        f = target / rel
        if f.suffix == ".ipynb":
            scrub_notebook(f, notes)
        if f.suffix not in TEXT_EXT and f.name != "LICENSE":
            continue
        text = f.read_text(encoding="utf8", errors="replace")
        if f.suffix == ".ipynb":
            text = chr(10).join(strings(json.loads(text)))
        for n, line in enumerate(text.splitlines(), 1):
            for s in hits(line):
                if s == PRIVATE[-1] and f.name in AUTHOR_FILES and AUTHOR_LINE.search(line):
                    continue
                bad.append(f"{rel.as_posix()}:{n}: {s}: {line.strip()[:100]}")
    print(f"exported {len(files)} files to {target}")
    for n in notes:
        print("NOTE", n)
    print("top level:", ", ".join(sorted(p.name for p in target.iterdir())))
    if bad:
        print(f"PRIVATE-DATA SCAN FAILED: {len(bad)} hit(s)")
        print("\n".join(bad))
        sys.exit(1)
    print("private-data scan: no hits")


if __name__ == "__main__":
    main()
