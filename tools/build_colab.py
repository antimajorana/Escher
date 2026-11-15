"""Make Google Colab versions of the three notebooks (they are not part of the package or the repository).

    python tools/build_colab.py OUTPUT_FOLDER

Each notebook in notebooks/ is copied to OUTPUT_FOLDER with its first setup cell replaced by a Colab setup cell. That cell
unpacks the whole papertilling package, embedded as a base64 zip (no upload or GitHub access needed), and installs spglib
and ipywidgets only if Colab lacks them. The payload is written in short lines inside a hidden form cell (Colab shows only
its title): one very long line in an open code cell makes the Colab editor slow. Rebuild after every change to the package.
Check a build without Colab: python tools/run_notebooks.py --dir OUTPUT_FOLDER --no-save, run from a Python that does not
have papertilling installed.
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
LOCAL_SETUP = "# Setup: find the papertilling package"


def package_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted((ROOT / "papertilling").rglob("*")):
            if p.is_file() and not (set(p.parts) & SKIP_DIRS) and p.suffix in (".py", ".json", ".txt"):
                data = p.read_bytes()
                if p.suffix == ".json":                         # compact JSON: same content, fewer bytes
                    data = json.dumps(json.loads(data.decode("utf8")), separators=(",", ":")).encode("utf8")
                z.writestr(p.relative_to(ROOT).as_posix(), data)
        z.write(ROOT / "pyproject.toml", "pyproject.toml")
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    return "\n".join(b[i:i + 100] for i in range(0, len(b), 100))


SETUP = '''#@title Setup: run this cell first (unpacks the papertilling package; a few seconds) { display-mode: "form" }
import base64, io, os, sys, zipfile, subprocess, importlib.util, tempfile, time
_t = time.time()
_dir = "/content/papertilling_pkg" if os.path.isdir("/content") else os.path.join(tempfile.gettempdir(), "papertilling_pkg")
_PKG = """
{b64}
"""
zipfile.ZipFile(io.BytesIO(base64.b64decode(_PKG))).extractall(_dir)
sys.path.insert(0, _dir)
_need = [m for m in ("spglib", "ipywidgets") if importlib.util.find_spec(m) is None]
if _need:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_need], check=False)
try:
    from google.colab import output; output.enable_custom_widget_manager()
except Exception:
    pass
import papertilling
print("papertilling", papertilling.__version__, "ready in %.1f s" % (time.time() - _t), "| Python", sys.version.split()[0],
      "| numba:", importlib.util.find_spec("numba") is not None)
'''


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    setup = SETUP.replace("{b64}", package_zip())
    for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf8"))
        cell = dict(cell_type="code", metadata={"id": "papertilling-setup", "cellView": "form"}, execution_count=None,
                    outputs=[], source=setup.splitlines(keepends=True))
        cells = [c for c in nb["cells"] if not "".join(c["source"]).startswith(LOCAL_SETUP)]
        if len(cells) != len(nb["cells"]) - 1:
            raise SystemExit(f"{nb_path.name}: the local setup cell was not found")
        for c in cells:
            c.setdefault("metadata", {}).pop("execution", None)
            if c["cell_type"] == "code":
                c["outputs"], c["execution_count"] = [], None
        nb["cells"] = cells[:1] + [cell] + cells[1:]             # title, then setup
        nb.setdefault("metadata", {})["colab"] = {"provenance": []}
        nb["metadata"].pop("widgets", None)                      # stale slider state from the local run
        dst = out / nb_path.name.replace(".ipynb", " (Colab).ipynb")
        dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf8")
        print(f"{dst.name}: {dst.stat().st_size / 1e3:.0f} kB, longest line {max(len(l) for c in nb['cells'] for l in c['source'])} characters")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
