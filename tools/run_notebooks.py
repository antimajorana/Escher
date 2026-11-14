"""Execute the notebooks top to bottom (a smoke test of every cell) and save them with their outputs.

    python tools/run_notebooks.py                         # the three local notebooks, saved with outputs
    python tools/run_notebooks.py 2                       # only notebook 2
    python tools/run_notebooks.py --kernel <kernel-name> --no-save          # under another Jupyter kernel
    python tools/run_notebooks.py --dir <folder> --no-save                  # the notebooks of another folder

Kernel names: `jupyter kernelspec list`. --no-save runs without touching the files.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    args = sys.argv[1:]
    kernel = args[args.index("--kernel") + 1] if "--kernel" in args else "python3"
    save = "--no-save" not in args
    want = [a for a in args if a.isdigit()]             # 1, 2 or 3: only that notebook
    folder = (ROOT / args[args.index("--dir") + 1]) if "--dir" in args else ROOT / "notebooks"
    bad = 0
    for p in sorted(folder.glob("*.ipynb")):
        if want and p.name.split(".")[0] not in want:
            continue
        t = time.time()
        nb = nbformat.read(p, as_version=4)
        try:
            NotebookClient(nb, timeout=900, kernel_name=kernel, resources={"metadata": {"path": str(p.parent)}}).execute()
        except Exception as err:                                  # show the failing cell's error, keep going
            bad += 1
            print(f"{p.name}: FAILED under kernel {kernel} after {time.time() - t:.0f} s\n   " + str(err)[-600:].replace("\n", "\n   "))
            continue
        if save:
            nbformat.write(nb, p)
        print(f"{p.name}: OK in {time.time() - t:.0f} s (kernel {kernel}{'' if save else ', not saved'})")
    sys.exit(1 if bad else 0)
