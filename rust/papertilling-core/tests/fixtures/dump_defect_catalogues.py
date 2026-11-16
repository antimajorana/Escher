"""Write defect_catalogues.json: Python's DefectCatalogue for the lattices the app's Defects tab offers first.

The reference for rust/papertilling-core/tests/defect_catalogue_fixture.rs (computed by the Python package, never
by the Rust code under test). Lattices: the first entry of every group in papertilling/data/points.json (the
Gallery's default rule) and the 21 named lattices of papertilling/data/named.json, each built the way the app
builds it: Family.from_recipe(recipe) with the stored frame (identity for points.json).

    PAPERTILLING_NO_RUST=1 PYTHONPATH=<repo> python rust/papertilling-core/tests/fixtures/dump_defect_catalogues.py

PAPERTILLING_NO_RUST=1 makes sure the optional Rust extension is not used.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["PAPERTILLING_NO_RUST"] = "1"
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from papertilling.defects.catalogue import DefectCatalogue  # noqa: E402
from papertilling.legacy.lattice_families import Family  # noqa: E402

GROUPS = ["p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm", "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m"]


def fl(x, nd=9):
    return [round(float(v), nd) for v in np.asarray(x, float).ravel()]


def opt(x):
    return None if x is None else str(x)


def dump(cat):
    nrm = cat.normaliser
    return {
        "group": cat.group,
        "chiral": bool(cat.chiral),
        "R0": round(float(cat.R0), 9),
        "b_min": round(float(cat.b_min), 9),
        "dislocations": [dict(id=d["id"], b_frac=[int(v) for v in d["b_frac"]], b=fl(d["b"]), length=round(float(d["length"]), 9),
                              angle=round(float(d["angle"]), 6), size=int(d["size"]), self_inverse=bool(d["self_inverse"]),
                              inverse_id=d["inverse_id"], mirror_id=opt(d["mirror_id"])) for d in cat.dislocations],
        "centre_classes": [dict(id=c["id"], order=int(c["order"]), gclass=c["gclass"], label=c["label"], location=c["location"],
                                mirror_of=opt(c["mirror_of"]), point=fl(c["point"]), size=int(c["size"]), split=bool(c["split"]),
                                angles=[float(a) for a in c["angles"]]) for c in cat.centre_classes],
        "disclinations": [dict(id=d["id"], centre_class=d["centre_class"], wyckoff=d["wyckoff"], gclass=d["gclass"],
                               order=int(d["order"]), angle=float(d["angle"]), location=d["location"], point=fl(d["centre"]),
                               inverse_id=d["inverse_id"], mirror_id=opt(d["mirror_id"])) for d in cat.disclinations],
        "mirrors": [dict(id=m["id"], kind=m["kind"], glide=round(float(m["glide"]), 9), period=round(float(m["period"]), 9),
                         offset=round(float(m["offset"]), 9), spacing=round(float(m["spacing"]), 9),
                         through_site=bool(m["through_site"]), size=int(m["size"]), direction=fl(m["direction"]))
                    for m in cat.mirrors],
        "normaliser": dict(index=int(nrm["index"]), continuous=int(nrm["continuous"]), complete=bool(nrm["complete"]),
                           ita_generic_index=nrm["ita_generic_index"], note=nrm["note"],
                           holohedry_order=int(nrm["holohedry_order"]), point_group_order=int(nrm["point_group_order"]),
                           cosets=[dict(id=r["id"], kind=r["kind"], centre_map=r["centre_map"], sites_to=r["sites_to"],
                                        check_group=bool(r["check_group"]), check_geometry=bool(r["check_geometry"]))
                                   for r in nrm["cosets"]]),
        "domain_walls": [dict(id=w["id"], coset=w["coset"], type=w["type"], centre_map=w["centre_map"], sites_to=w["sites_to"],
                              complete=bool(w["complete"])) for w in cat.domain_walls],
        "twin_walls": [dict(id=w["id"], direction_frac=[int(v) for v in w["direction_frac"]], through_site=bool(w["through_site"]),
                            sigma_in_G=bool(w["sigma_in_G"]), network_invisible=bool(w["network_invisible"]),
                            n_irregular=int(w["n_irregular"]), visible=bool(w["visible"]), domain_coset=opt(w["domain_coset"]),
                            class_size=int(w["class_size"]), offset=round(float(w["offset"]), 9)) for w in cat.twin_walls],
        "mobius_seams": [dict(id=w["id"], line=w["line"], line_kind=w["line_kind"]) for w in cat.mobius_seams],
        "glide_breaking": [dict(id=w["id"], line=w["line"], line_kind=w["line_kind"], slip=round(float(w["slip"]), 9),
                                slip_over_period=round(float(w["slip_over_period"]), 9),
                                same_domains_as=opt(w["same_domains_as"]), same_as_twin=opt(w["same_as_twin"]))
                           for w in cat.glide_breaking],
    }


def dumps(res):
    """Compact JSON, one lattice per line."""
    body = ",\n".join(f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'), ensure_ascii=False)}"
                      for k, v in res["lattices"].items())
    return '{"how":' + json.dumps(res["how"]) + ',\n"lattices":{\n' + body + "\n}}\n"


def main(out=None):
    data = ROOT / "papertilling" / "data"
    points = json.loads((data / "points.json").read_text(encoding="utf8"))["groups"]
    named = json.loads((data / "named.json").read_text(encoding="utf8"))["named"]
    items = [(f"{g}#1", points[g]["entries"][0]["recipe"], None) for g in GROUPS]
    items += [(nl["key"], nl["recipe"], nl["frame"]) for nl in named]
    res = {"how": "PAPERTILLING_NO_RUST=1 python rust/papertilling-core/tests/fixtures/dump_defect_catalogues.py "
                  "(Python DefectCatalogue, default bmax=2)", "lattices": {}}
    T = time.time()
    for tag, recipe, frame in items:
        t = time.time()
        fam = Family.from_recipe(recipe)
        if frame is not None:
            fam.frame = (np.asarray(frame["q"], float), np.asarray(frame["origin"], float))
        entry = dump(DefectCatalogue(fam))
        entry["recipe_group"] = recipe["group"]
        res["lattices"][tag] = entry
        print(f"{tag:24s} {entry['group']:5s} T={len(entry['dislocations'])} C={len(entry['centre_classes'])} "
              f"D={len(entry['disclinations'])} Wa={len(entry['domain_walls'])} Wb={len(entry['twin_walls'])} "
              f"Wc={len(entry['mobius_seams'])} Wd={len(entry['glide_breaking'])}  # time {time.time() - t:.1f} s")
    path = Path(out) if out else Path(__file__).with_name("defect_catalogues.json")
    path.write_bytes(dumps(res).encode("utf8"))
    print(f"wrote {path} ({path.stat().st_size} bytes)  # time {time.time() - T:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
