"""Dump Python reference answers for the Rust Combine tab (Stage 5) parity tests.

    PAPERTILLING_NO_RUST=1 python tools/checks/combine_parity_dump.py

Writes, under tests/fixtures/:
  combine_parity_py.json   total charge (label, kind, angle, b, centre class), core graph size and degrees, local cores,
                           and the equivalent_single verdict (equivalent, matches, verdict) of every configuration below,
                           or the refusal message when build() refuses it (refusal rule: build.validate_specs / _run);
                           the G+ rotation-centre classes of a few lattices (label, order, members, angles) and
                           nearest_centre answers for fixed query points (the snapping rule of build(snap=True)).
  config_py_*.json         configurations written by papertilling.defects.config.save_config (portable
                           papertilling.config.v1 files) for the Rust import tests.
Read by rust/papertilling-core/tests/combine_python_parity.rs and the papertilling-app parity tests. Every value here
is computed by the pure-Python package (run with PAPERTILLING_NO_RUST=1 so papertilling_rs is not used).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PAPERTILLING_NO_RUST", "1")

from papertilling import named  # noqa: E402
from papertilling.defects.build import build, lattice_data  # noqa: E402
from papertilling.defects.config import equivalent_single, save_config  # noqa: E402
from papertilling.lattice import load_catalogue  # noqa: E402

OUT = ROOT / "tests" / "fixtures"


def fl(x):
    return [float(v) + 0.0 for v in np.asarray(x, float).ravel()]


def lattice_ref(key):
    """('named', name) or ('points', group, index) -> (Lattice, json description)."""
    if key[0] == "named":
        L = named(key[1])
        desc = dict(source="named", name=key[1])
    else:
        L = load_catalogue(group=key[1], rule=key[0]).lattices[key[2]]
        desc = dict(source=key[0], group=key[1], index=key[2])
    fam = L.family
    raw = fam.recipe()
    recipe = dict(group=str(raw["group"]), sites=[fl(p) for p in raw["sites"]],
                  bonds=[[fl(p), fl(q)] for p, q in raw["bonds"]], a=float(raw.get("a", 1.0)))
    for k in ("b", "gamma"):
        if raw.get(k) is not None:
            recipe[k] = float(raw[k])
    Q, o0 = fam.frame
    desc.update(group=str(fam.group), recipe=recipe, frame=dict(q=np.asarray(Q, float).tolist(), origin=fl(o0)))
    return L, desc


def spec_json(s):
    out = dict(kind=s["kind"])
    if s["kind"] == "dislocation":
        out["b"] = fl(s["b"])
        if s.get("at") is not None:
            out["at"] = fl(s["at"])
    else:
        out["angle"] = float(s["angle"])
        out["at"] = fl(s["at"])
    if s.get("cut") is not None:
        out["cut"] = fl(s["cut"])
    return out


def run_case(name, key, specs_fn, grid):
    L, desc = lattice_ref(key)
    lat = lattice_data(L)
    specs = specs_fn(lat)
    rec = dict(name=name, lattice=desc, specs=[spec_json(s) for s in specs], grid=grid, radius=2)
    t0 = time.time()
    try:
        conf = build(lat, *specs, picture="raw")
    except ValueError as e:
        rec["python"] = dict(ok=False, error=str(e))
        print(f"{name}: refused ({e})")
        return rec
    q = conf.total_charge()
    charge = dict(kind=q["kind"], angle=float(q["angle"]), label=q["label"])
    if "b" in q:
        charge["b"] = fl(q["b"])
        charge["lattice_vector"] = bool(q["lattice_vector"])
    if q.get("centre") is not None:
        charge["centre"] = fl(q["centre"])
        charge["centre_class"] = q["centre_class"]
    py = dict(ok=True, charge=charge, cores=[c["label"] for c in conf.cores()])
    try:
        H = conf.core_graph(2)
    except ValueError as e:                 # the core ball reaches the patch rim (graph_core refuses)
        py["core_error"] = str(e)
        rec["python"] = py
        print(f"{name}: {q['label']} | core refused ({e})")
        return rec
    py["core"] = dict(nodes=H.number_of_nodes(), edges=H.number_of_edges(),
                      degrees=sorted(int(d) for _, d in H.nodes(data="deg")))
    res = equivalent_single(conf, radius=2, grid=grid)
    py.update(equivalent=bool(res["equivalent"]), matches=list(res["matches"]), verdict=res["verdict"],
              candidates=[c["name"] for c in res["candidates"]])
    rec["python"] = py
    print(f"{name}: {q['label']} | {res['verdict']} ({time.time() - t0:.1f} s)")
    return rec


def centre(lat, label, shift=(0, 0)):
    r = next(r for r in lat.centres if r["label"] == label)
    return np.asarray(r["point"], float) + lat.A @ np.asarray(shift, float)


def top_labels(lat):
    top = max(r["order"] for r in lat.centres)
    return top, [r["label"] for r in lat.centres if r["order"] == top]


def nb3_next_cell(lat):
    top, labels = top_labels(lat)
    a = 360.0 / top
    return [dict(kind="disclination", angle=a, at=centre(lat, labels[0])),
            dict(kind="disclination", angle=-a, at=centre(lat, labels[0], (1, 0)))]


def nb3_first_last(lat):
    top, labels = top_labels(lat)
    a = 360.0 / top
    return [dict(kind="disclination", angle=a, at=centre(lat, labels[0])),
            dict(kind="disclination", angle=-a, at=centre(lat, labels[-1]))]


def nb3_shortest(lat):
    from papertilling.defects.catalogue import DefectCatalogue
    cat = DefectCatalogue(lat.cell)
    return [dict(kind="dislocation", b=cat.dislocations[0]["b"], at=lat.A @ np.array([0.371, 0.413]))]


def disc(angle, at):
    return dict(kind="disclination", angle=float(angle), at=tuple(float(x) for x in at))


def disl(b, at, cut=None):
    d = dict(kind="dislocation", b=tuple(float(x) for x in b), at=tuple(float(x) for x in at))
    if cut is not None:
        d["cut"] = tuple(float(x) for x in cut)
    return d


SQ = ("named", "square")
HC = ("named", "honeycomb")
SS = ("named", "shastry_sutherland")
P4G = ("points", "p4g", 0)
P3 = ("points", "p3", 0)
P31M = ("points", "p31m", 0)

CASES = [
    # Notebook 3, section 0 (control) and sections 2-3 on its default lattice (p4g #1, points rule)
    ("nb3 square +90 site / -90 face", SQ, lambda lat: [disc(90, (0, 0)), disc(-90, (0.5, 0.5))], 1),
    ("nb3 square unit dislocation", SQ, lambda lat: [disl((1, 0), (0.371, 0.413))], 1),
    ("nb3 p4g +90/-90 next cell", P4G, nb3_next_cell, 2),
    ("nb3 p4g +90 2a(i) / -90 2a(ii)", P4G, nb3_first_last, 2),
    ("nb3 p4g shortest dislocation", P4G, nb3_shortest, 2),
    # tests/test_config.py controls
    ("honeycomb 5-7 pair", HC, lambda lat: [disc(60, centre(lat, top_labels(lat)[1][0])),
                                            disc(-60, centre(lat, top_labels(lat)[1][0], (1, 0)))], 2),
    ("square +90 site", SQ, lambda lat: [disc(90, (0, 0))], 1),
    ("square +90 face", SQ, lambda lat: [disc(90, (0.5, 0.5))], 1),
    ("square +180 at 2-fold", SQ, lambda lat: [disc(180, (0, 0.5))], 1),
    ("square -180 at 2-fold", SQ, lambda lat: [disc(-180, (0, 0.5))], 1),
    ("square +90 +90 far apart", SQ, lambda lat: [disc(90, (0, 0)), disc(90, (12, 0))], 1),
    ("square dislocation dipole", SQ, lambda lat: [disl((1, 0), (-3.63, 0.41), (0, 1)), disl((-1, 0), (3.37, 0.41), (0, -1))], 1),
    ("square b=(0,1) elsewhere", SQ, lambda lat: [disl((0, 1), (2.29, -1.37))], 1),
    ("shastry_sutherland +90 2a(i)", SS, lambda lat: [disc(90, (0.5, 1.5))], 1),
    # split G+ classes
    ("p3 +120 1b / -120 1c", P3, lambda lat: [disc(120, centre(lat, "1b")), disc(-120, centre(lat, "1c"))], 1),
    ("p31m +120 2b(i) / -120 2b(ii)", P31M, lambda lat: [disc(120, centre(lat, "2b(i)")), disc(-120, centre(lat, "2b(ii)"))], 1),
    ("p3 +120 1b / -120 1c next cells", P3, lambda lat: [disc(120, centre(lat, "1b")), disc(-120, centre(lat, "1c", (1, 1)))], 1),
    ("p3 +120 1a / -120 1a next cell", P3, lambda lat: [disc(120, centre(lat, "1a")), disc(-120, centre(lat, "1a", (1, 0)))], 1),
    # surgery edge cases (refusal rule = validate_specs: sum of Frank angles >= 360 refused; negative sums allowed)
    ("square +90 x3", SQ, lambda lat: [disc(90, (3 * i, 0)) for i in range(3)], 1),
    ("square +90 x4 refused", SQ, lambda lat: [disc(90, (3 * i, 0)) for i in range(4)], 1),
    ("square +180 +180 refused", SQ, lambda lat: [disc(180, (0, 0.5)), disc(180, (4, 0.5))], 1),
    ("square -90 x4 whole turn", SQ, lambda lat: [disc(-90, (3 * i + 0.5, 0.5)) for i in range(4)], 1),
    ("square -180 -180 whole turn", SQ, lambda lat: [disc(-180, (0, 0.5)), disc(-180, (4, 0.5))], 1),
    ("honeycomb +120 x3 refused", HC, lambda lat: [disc(120, centre(lat, top_labels(lat)[1][0], (3 * i, 0))) for i in range(3)], 1),
    ("square coincident +90/-90 site", SQ, lambda lat: [disc(90, (0, 0)), disc(-90, (0, 0))], 1),
    ("square coincident +90/-90 face", SQ, lambda lat: [disc(90, (0.5, 0.5)), disc(-90, (0.5, 0.5))], 1),
    ("square coincident +90/+90 site", SQ, lambda lat: [disc(90, (0, 0)), disc(90, (0, 0))], 1),
    ("square +90 / -90 neighbouring sites", SQ, lambda lat: [disc(90, (0, 0)), disc(-90, (1, 0))], 1),
    ("square dislocation at a disclination", SQ, lambda lat: [disc(90, (0, 0)), disl((1, 0), (0, 0))], 1),
    ("square two dislocations same end", SQ, lambda lat: [disl((1, 0), (0.371, 0.413)), disl((0, 1), (0.371, 0.413))], 1),
    ("square non-lattice b refused", SQ, lambda lat: [disl((0.5, 0), (0.371, 0.413))], 1),
    ("square zero b refused", SQ, lambda lat: [disl((0, 0), (0.371, 0.413))], 1),
]

SNAP_LATTICES = [SQ, HC, SS, ("named", "kagome"), P4G, P3, P31M, ("points", "p6", 0), ("points", "p2", 0),
                 ("points", "cmm", 0), ("points", "pgg", 0), ("points", "pmg", 0)]
QUERY_FRAC = [(0.12, -0.08), (0.37, 0.41), (0.71, 0.23), (0.49, 0.52), (-0.33, 0.66), (1.27, 0.88), (0.05, 0.95)]


def snap_reference(key):
    L, desc = lattice_ref(key)
    lat = lattice_data(L)
    centres = [dict(label=r["label"], order=int(r["order"]), point=fl(r["point"]),
                    members=[fl(m) for m in r["members"]], angles=[float(a) for a in r["angles"]]) for r in lat.centres]
    angles = sorted({a for r in lat.centres for a in r["angles"]})
    queries = []
    for f in QUERY_FRAC:
        near = lat.A @ np.asarray(f, float)
        for a in angles:
            p = lat.nearest_centre(a, near)
            queries.append(dict(angle=float(a), near=fl(near), point=fl(p), label=lat.centre_class(p)["label"]))
    return dict(lattice=desc, a1=fl(lat.cell.a1), a2=fl(lat.cell.a2), centres=centres, nearest=queries)


def save_fixtures():
    """Python-written papertilling.config.v1 files (save_config with its default summary)."""
    out = []
    sq = named("square")
    pair = build(sq, disc(90, (0, 0)), disc(-90, (0.5, 0.5)), picture="raw")
    out.append(save_config(pair, OUT / "config_py_square_pair.json", grid=1))
    ss = named("shastry_sutherland")
    one = build(ss, disc(90, (0.5, 1.5)), picture="raw")
    out.append(save_config(one, OUT / "config_py_shastry_sutherland.json", grid=1))
    hc = named("honeycomb")
    b = np.asarray(lattice_data(hc).cell.a1, float)
    u = b / np.linalg.norm(b)
    cut = np.array([-u[1], u[0]]) + 0.35 * u          # cross2(cut, b) < 0, as build() requires
    d = build(hc, disl(b, (0.4, 0.3), cut=cut), picture="relaxed")
    out.append(save_config(d, OUT / "config_py_honeycomb_cut.json", grid=1))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = dict(
        note="generated by tools/checks/combine_parity_dump.py (pure-Python papertilling, PAPERTILLING_NO_RUST=1)",
        cases=[run_case(*c) for c in CASES],
        snap=[snap_reference(k) for k in SNAP_LATTICES],
    )
    (OUT / "combine_parity_py.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf8")
    for p in save_fixtures():
        print("wrote", p.relative_to(ROOT))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
