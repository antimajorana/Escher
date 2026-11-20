"""Build papertilling/data/layer_sites_80.json: one exact-group site set per layer group (C1 WP-L1, plan Q12).

For each of the 80 layer groups, try k = 1, 2, 3 general-position orbits (seeds drawn with numpy seed = group
number: x, y uniform in [0, 1), z = +-uniform(0.1, 0.4), a = 1, metric `_layer_slab.METRICS`) and keep the first
draw whose site set
  - has every orbit at the general multiplicity and no two sites closer than 0.3 sqrt(cell area / sites),
  - is exactly G for spglib's layer-group check (`get_symmetry_layerdataset`, symprec 1e-5, aperiodic axis z):
    the same number and the same operation count (point order x centring) on the conventional cell,
  - gets a connected 2-periodic net from the shells-until-connected rule.
Offline only: spglib is not used at run time. Run: python tools/build_layer_sites.py  (about 10 s)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import spglib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from papertilling import _layer_slab as LS  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "papertilling" / "data" / "layer_sites_80.json"
C = 20.0


def spglib_layer(g, F):
    L = LS.lattice(g)
    A = np.vstack([L, [0.0, 0.0, C]])
    X = np.column_stack([F[:, 0], F[:, 1], F[:, 2] / C + 0.5])
    ds = spglib.get_symmetry_layerdataset((A, X, [1] * len(F)), aperiodic_dir=2, symprec=1e-5)
    return (None, 0) if ds is None else (int(ds.number), len(ds.rotations))


def min_distance(P, L):
    best = np.inf
    for n0 in (-1, 0, 1):
        for n1 in (-1, 0, 1):
            D = P[None, :, :] + n0 * L[0] + n1 * L[1] - P[:, None, :]
            dist = np.linalg.norm(D, axis=2)
            if n0 == 0 and n1 == 0:
                dist += np.eye(len(P)) * 9
            best = min(best, float(dist.min()))
    return best


def build(g):
    rng = np.random.default_rng(g["number"])
    gen = g["point_order"] * (2 if g["centred"] else 1)
    L = LS.lattice(g)
    for k in (1, 2, 3):
        for _ in range(40):
            seeds = [[float(rng.random()), float(rng.random()),
                      float(rng.choice([-1, 1]) * rng.uniform(0.1, 0.4))] for _ in range(k)]
            seeds = [[round(x, 4) for x in s] for s in seeds]
            try:
                F, lab = LS.sites_from_orbits(g, seeds)
            except LS.SlabRefusal:
                continue
            if any((lab == o).sum() != gen for o in range(k)):
                continue
            P = LS.cartesian(F, L)
            if min_distance(P, L) < 0.3 * np.sqrt(abs(np.linalg.det(L[:, :2])) / len(P)):
                continue
            num, nops = spglib_layer(g, F)
            if num != g["number"] or nops != gen:
                continue
            try:
                bonds, _ = LS.shell_bonds(P, L, lab)
            except LS.SlabRefusal:
                continue
            return dict(number=g["number"], symbol=g["symbol"], orbits=seeds, sites=len(F), exact=True,
                        spglib_number=num, spglib_ops=nops, bonds_per_cell=len(bonds))
    return dict(number=g["number"], symbol=g["symbol"], orbits=[[0.17, 0.29, 0.18]], sites=gen, exact=False)


def main():
    t0 = time.perf_counter()
    groups = [LS.group(n) for n in range(1, 81)]
    out = [build(g) for g in groups]
    OUT.write_text(json.dumps(dict(version=1, source="tools/build_layer_sites.py (spglib 2.8.0 layer check)",
                                   groups=out), indent=1) + "\n", encoding="utf-8")
    bad = [o["number"] for o in out if not o["exact"]]
    print(f"{len(out)} groups, {len(out) - len(bad)} exact, not exact: {bad}")
    print("orbits used:", {k: sum(len(o['orbits']) == k for o in out) for k in (1, 2, 3)})
    print(f"# time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
