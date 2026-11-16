"""Dump the Python reference answers for the Rust surface audit to surface_table_py.json (next to this script).

Run from the repository root with the pure-Python package (no Rust extension), about 15 minutes:
    PAPERTILLING_NO_RUST=1 PYTHONPATH=. python rust/papertilling-core/tests/fixtures/dump_surface_table.py
(`--quick` skips the flat rows; names of groups / named lattices as arguments write a partial file instead.)

Cases: the first `points` lattice of each of the 17 groups (pt.load_catalogue()[g].lattices[0]) and the named lattices
square, honeycomb, kagome, triangular, chiral_square, shastry_sutherland, zigzag, brick_pg. For each case:
  cell       the Python cell (a1, a2, basis, bonds [i, j, n1, n2, kind]) the Rust tests rebuild from
  table      surface_table(L) rows: status, V, E, F, chi, orientable, closed, cone angles, face histogram, params
  poly       sphere / projective_plane at explicit sizes and centre modes (None, "face centre", "site")
  flat       torus(T1=(h, 0), T2=(0, k)), cylinder(wrap=(0, k), length=h), Klein bottle / Moebius band at (along, across)
Every value here is computed by the Python package (the reference); nothing is copied from the Rust code under test.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["PAPERTILLING_NO_RUST"] = "1"

import papertilling as pt  # noqa: E402
from papertilling.surfaces import (cylinder, genus_surface, klein_bottle, mobius_band, projective_plane,  # noqa: E402
                                   sphere, surface_table, torus)
from papertilling.surfaces.refusal import SurfaceImpossible, SurfaceNotFound  # noqa: E402

GROUPS = ["p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm", "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m"]
NAMED = ["square", "honeycomb", "kagome", "triangular", "chiral_square", "shastry_sutherland", "zigzag", "brick_pg"]
POLY_SIZES = [(1, 0), (1, 1), (2, 0), (2, 1)]
CENTRES = [None, "face centre", "site"]
FLAT_HK = [(4, 4), (1, 1), (3, 0), (2, 5)]


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return jsonable(x.tolist())
    if isinstance(x, float):
        return round(x, 9)
    return x


def summary(S):
    hist = {}
    for f in S.faces:
        hist[str(len(f))] = hist.get(str(len(f)), 0) + 1
    angles = sorted(round(float(c["angle"]), 6) for c in (S.cones or []))
    ch = S.checks(show=False)
    params = {k: v for k, v in (S.params or {}).items() if k in ("T1", "T2", "wrap", "length", "along", "across", "width",
                                                                 "glide", "size", "centre", "centre_location", "torus_size")}
    return dict(status="OK", V=len(S.sites), E=len(S.bonds), F=len(S.faces), chi=int(S.chi), orientable=bool(S.orientable),
                closed=bool(S.closed), cone_angles=angles, face_hist=hist, checks_ok=bool(ch["ok"]), params=jsonable(params))


def attempt(fn):
    try:
        S = fn()
    except SurfaceImpossible as e:
        return dict(status="IMPOSSIBLE", reason=str(e.reason)[:160])
    except SurfaceNotFound as e:
        return dict(status="NOT FOUND", reason=str(e.reason)[:160])
    except ValueError as e:
        return dict(status="INVALID", reason=str(e)[:160])
    out = summary(S)
    if not out["checks_ok"]:
        out["status"] = "CHECKS FAILED"
    return out


def cell_of(L):
    c = L.cell()
    A = c.A
    return dict(a1=[float(A[0, 0]), float(A[1, 0])], a2=[float(A[0, 1]), float(A[1, 1])],
                basis=[[float(x) for x in p] for p in c.basis],
                bonds=[[int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4]) if len(b) > 4 else 0] for b in c.bonds])


def one_case(label, L, quick):
    t0 = time.perf_counter()
    rows = {}
    for r in surface_table(L, show=False):
        rows[r["surface"]] = summary(r["S"]) if r["status"] == "OK" else dict(status=r["status"], reason=r["reason"][:160])
    poly = []
    for proj in (False, True):
        for size in POLY_SIZES:
            for centre in CENTRES:
                ctor = projective_plane if proj else sphere
                res = attempt(lambda: ctor(L, size=size, centre=centre))
                poly.append(dict(projective=proj, size=list(size), centre=centre, **res))
    flat = []
    if not quick:
        for h, k in FLAT_HK:
            flat.append(dict(kind="torus", h=h, k=k, **attempt(lambda: torus(L, T1=(h, 0), T2=(0, max(k, 1))))))
            flat.append(dict(kind="cylinder", h=h, k=k, **attempt(lambda: cylinder(L, wrap=(0, max(k, 3)), length=h))))
            flat.append(dict(kind="Klein bottle", h=h, k=k,
                             **attempt(lambda: klein_bottle(L, along=2 * max(k, 2) + 1, across=max(h, 2), which=0))))
            flat.append(dict(kind="Moebius band", h=h, k=k,
                             **attempt(lambda: mobius_band(L, along=2 * max(k, 2) + 1, width=h, which=0))))
        for g in (2, 3):
            flat.append(dict(kind=f"genus {g}", h=0, k=0, **attempt(lambda: genus_surface(L, g=g))))
    print(f"{label:28s} {L.group:5s} {time.perf_counter() - t0:7.1f} s", flush=True)
    return dict(label=label, group=L.group, cell=cell_of(L), table=rows, poly=poly, flat=flat)


def compact(case):
    """Keep the fixture small: the explicit-size rows keep their verdict and counts only (no reason text, no params)."""
    for key in ("poly", "flat"):
        for row in case[key]:
            row.pop("reason", None)
            row.pop("params", None)
    return case


def main():
    if sys.argv[1:2] == ["--recompact"]:                       # re-apply compact() to an existing dump
        path = sys.argv[2]
        with open(path, encoding="utf8") as f:
            data = json.load(f)
        data["cases"] = [compact(c) for c in data["cases"]]
        with open(path, "w", encoding="utf8") as f:
            json.dump(data, f, separators=(",", ":"))
            f.write("\n")
        return
    quick = "--quick" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    cat = pt.load_catalogue()
    cases = []
    for g in GROUPS:
        if only and g not in only:
            continue
        cases.append(one_case(f"{g}#1", cat[g].lattices[0], quick))
    for key in NAMED:
        if only and key not in only:
            continue
        cases.append(one_case(key, pt.named(key), quick))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surface_table_py.json" if not only else "surface_table_py_partial.json")
    with open(out, "w", encoding="utf8") as f:
        json.dump(dict(generated_by="rust/papertilling-core/tests/fixtures/dump_surface_table.py", pure_python=True,
                       cases=[compact(c) for c in cases]), f, separators=(",", ":"))
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
