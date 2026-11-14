"""Write the three notebooks of papertilling from their text (tools/nb_text.py).

    python tools/build_notebooks.py

The text of each notebook is in tools/nb_text.py (cells as a list of ("md" | "code", source)).
"""
import json
from pathlib import Path

from nb_text import NB1, NB2, NB3, BOOT

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, src):
    src = src.strip("\n") + "\n"
    lines = src.splitlines(keepends=True)
    lines[-1] = lines[-1].rstrip("\n")
    if kind == "md":
        return dict(cell_type="markdown", metadata={}, source=lines)
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=lines)


def write(name, cells):
    cells = [cells[0], ("code", BOOT)] + list(cells[1:])        # setup cell right after the title
    cs = [cell(k, s) for k, s in cells]
    for i, c in enumerate(cs):
        c["id"] = f"cell-{i:02d}"
    nb = dict(nbformat=4, nbformat_minor=5, cells=cs,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                            language_info=dict(name="python")))
    p = ROOT / "notebooks" / name
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf8")
    print(p.name, len(cells), "cells")


if __name__ == "__main__":
    write("1. Lattices of a wallpaper group.ipynb", NB1)
    write("2. Defects of a lattice.ipynb", NB2)
    write("3. Defect configurations.ipynb", NB3)
