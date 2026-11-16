"""Write wall_pictures.json: Python's entry.picture(R0) of every wall (Wa, Wb, Wd) of the 12 lattices of
tools/checks/drawing_check.py (the Stage 4 gate), measured with drawing_check.measure (the independent tool).

Reference for rust/papertilling-core/tests/wall_pictures.rs (computed by the Python package, not by the Rust
code under test):

    PAPERTILLING_NO_RUST=1 PYTHONPATH=<repo> python rust/papertilling-core/tests/fixtures/dump_wall_pictures.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PAPERTILLING_NO_RUST"] = "1"
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "checks"))

import papertilling as pt  # noqa: E402
from papertilling.defects.catalogue import DefectCatalogue  # noqa: E402
from drawing_check import measure  # noqa: E402

NAMED = ("square", "kagome", "honeycomb", "shastry_sutherland", "chiral_square", "chiral_kagome", "zigzag",
         "sutherlands_brick", "brick_pg", "straight_p1")


def main():
    warnings.simplefilter("ignore")
    cats = pt.load_catalogue()
    lats = [("p4g#1", cats["p4g"].lattices[0]), ("p6m#4", cats["p6m"].lattices[3])]
    lats += [(k, pt.named(k)) for k in NAMED]
    out = {"how": "PAPERTILLING_NO_RUST=1 python rust/papertilling-core/tests/fixtures/dump_wall_pictures.py "
                  "(DefectEntry.picture(R0), drawing_check.measure)", "lattices": {}}
    T = time.time()
    for tag, L in lats:
        t = time.time()
        cat = DefectCatalogue(L)
        res = {}
        for e in list(cat.domain_walls) + list(cat.twin_walls) + list(cat.glide_breaking):
            d, info = e.picture(R=cat.R0)
            m = measure(d)
            res[e["id"]] = dict(stage=info["stage"], note=info["note"], sites=m["sites"], bonds=m["bonds"],
                                crossings=m["plain_crossings"], overlaps=m["overlaps"], crowded=m["crowded"])
        out["lattices"][tag] = dict(R0=cat.R0, walls=res)
        stages = {}
        for v in res.values():
            stages[v["stage"]] = stages.get(v["stage"], 0) + 1
        print(f"{tag:20s} {len(res)} walls {stages}  # time {time.time() - t:.1f} s", flush=True)
    path = Path(__file__).with_name("wall_pictures.json")
    body = ",\n".join(f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'), ensure_ascii=False)}"
                      for k, v in out["lattices"].items())
    path.write_bytes(('{"how":' + json.dumps(out["how"]) + ',\n"lattices":{\n' + body + "\n}}\n").encode("utf8"))
    print(f"wrote {path}  # time {time.time() - T:.1f} s")


if __name__ == "__main__":
    main()
