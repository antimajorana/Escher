"""Dump the Python reference answers for the Rust boundary-condition tests (C1 WP-B1) to surface_bc_py.json.

Run from the repository root with the pure-Python package (no Rust extension), under a minute:
    PAPERTILLING_NO_RUST=1 PYTHONPATH=. python rust/papertilling-core/tests/fixtures/dump_surface_bc.py

Cases (each with the Python cell [a1, a2, basis, bonds [i, j, n1, n2, kind]] that the Rust test rebuilds from):
  oblique_tuned   the named oblique lattice (label p2) tuned to b = 1, gamma = 90 deg: surface_table statuses (bug C1)
  <bc>:<lattice>  V, E, F of one boundary condition built by papertilling.surfaces._flat_bc.boundary_surface
Every value is computed by the Python package; nothing is copied from the Rust code under test.
"""
from __future__ import annotations

import json
import os
import time
import warnings

os.environ["PAPERTILLING_NO_RUST"] = "1"

import papertilling as pt  # noqa: E402
from papertilling.surfaces import surface_table  # noqa: E402
from papertilling.surfaces._flat_bc import boundary_surface  # noqa: E402
from papertilling.surfaces.flat import _cell_and_ops  # noqa: E402

ROWS = ["torus", "cylinder", "Moebius band", "Klein bottle", "sphere", "projective plane"]
COUNTS = [("torus_hex", "triangular", dict(kind="torus_hex", t1=(3, 0), t2=(0, 3))),
          ("cylinder_6_0", "honeycomb", dict(kind="cylinder", wrap=(6, 0), rows=3)),
          ("cylinder_3_1", "square", dict(kind="cylinder", wrap=(3, 1), rows=8)),
          ("mobius_even", "square", dict(kind="mobius", glide=0, steps=4, width=2)),
          ("klein_even", "square", dict(kind="klein", glide=0, steps=4, across=4))]


def cell_json(L):
    _, cell, A, *_ = _cell_and_ops(L)
    return [A[:, 0].tolist(), A[:, 1].tolist(), [list(map(float, p)) for p in cell.basis],
            [[int(x) for x in b[:5]] for b in cell.bonds]]


def main():
    t0 = time.perf_counter()
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = pt.named("oblique")
        L = L.set(b=1.0, gamma=90.0) or L
    T = surface_table(L, show=False, only=ROWS)
    out.append(dict(label="oblique_tuned", group=L.group, cell=cell_json(L), statuses={r["surface"]: r["status"] for r in T}))
    for label, key, kw in COUNTS:
        L = pt.named(key)
        S = boundary_surface(L, **kw)
        out.append(dict(label=f"{label}:{key}", cell=cell_json(L), counts=[len(S.sites), len(S.bonds), len(S.faces)]))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surface_bc_py.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"# time dump_surface_bc {time.perf_counter() - t0:.1f}s -> {path}")


if __name__ == "__main__":
    main()
