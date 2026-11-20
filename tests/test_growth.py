"""C1 WP-G1: touching-rule growth (papertilling.growth), against hand and independent answers.

Sources of the pinned numbers: hand derivations (boxes of whole squares / cubes, polyhedron and
incidence counts, closed forms of plan §2.4), and `_indep_e3` below, a float grower built on
scipy's Voronoi diagram that shares no code with `papertilling._growth_e3` ([COMPUTATIONAL
EVIDENCE]). The Rust tests (`growth/tests.rs`) pin the same values; `test_parity_with_rust` compares
the two implementations when the compiled extension exposes the growth bindings.
"""
from __future__ import annotations

import itertools
import json
import time
from functools import lru_cache

import numpy as np
import pytest

from papertilling import _rust, growth
from papertilling.growth import E3Complex, FiniteComplex, GrowthError

KINDS = ("cubic", "bitruncated_cubic", "rhombic_dodecahedral", "alternated_cubic")
SEEDS = ("vertex", "edge", "face", "cell")


def cells(g):
    return [s["cells"] for s in g["stats"]]


# ------------------------------------------------------------------ independent E3 grower
_BASIS = {"SC": np.eye(3), "BCC": np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1.]]),
          "FCC": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0.]])}


def _key(x):
    return tuple(int(t) for t in np.round(np.asarray(x, float) * 1e6))


@lru_cache(maxsize=None)
def _cell_vertices(lat):
    from scipy.spatial import Voronoi
    pts = np.array([np.array(n) @ _BASIS[lat] for n in itertools.product(range(-3, 4), repeat=3)])
    vor = Voronoi(pts)
    i0 = int(np.argmin(np.linalg.norm(pts, axis=1)))
    return vor.vertices[vor.regions[vor.point_region[i0]]]


def _nearest(x, lat):
    b = _BASIS[lat]
    n0 = np.round(np.asarray(x) @ np.linalg.inv(b))
    cand = (n0 + np.array(list(itertools.product(range(-2, 3), repeat=3)))) @ b
    d = np.linalg.norm(cand - x, axis=1)
    return [_key(c) for c in cand[d < d.min() + 1e-7]]


def _indep_e3(kind, seed, n):
    """Voronoi kinds: cells = lattice points, vertices = Voronoi vertices, cells at v = nearest
    lattice points. Octet: its dual, vertices = FCC + (1,0,0), cells = the FCC Voronoi vertices."""
    lat = {"cubic": "SC", "bitruncated_cubic": "BCC"}.get(kind, "FCC")
    vv, sh = _cell_vertices(lat), np.array([1.0, 0, 0])
    p = lambda k: np.array(k) / 1e6
    if kind != "alternated_cubic":
        verts_of = lambda c: [_key(p(c) + v) for v in vv]
        cells_at = lambda v: _nearest(p(v), lat)
    else:
        verts_of = lambda h: [_key(p(q) + sh) for q in _nearest(p(h) - sh, lat)]
        cells_at = lambda v: [_key(p(v) + w) for w in vv]
    pts = {"cubic": [(.5, .5, .5), (.5, .5, -.5), (.5, -.5, .5), (.5, -.5, -.5)],
           "bitruncated_cubic": [(0, .5, 1), (.5, 1, 0), (1, .5, 0), (1, 0, .5), (1, 0, -.5), (1, -.5, 0)],
           "rhombic_dodecahedral": [(1, 0, 0), (.5, .5, .5), (0, 1, 0), (.5, .5, -.5)],
           "alternated_cubic": [(1, 0, 0), (0, 1, 0), (0, 0, 1)]}[kind]
    # vertex / edge / face seeds as point sets (the same positions as e3_complex.rs documents)
    sets = {"cubic": (pts[:1], pts[:2], pts), "bitruncated_cubic": (pts[:1], pts[1:3], pts[3:]),
            "rhombic_dodecahedral": (pts[:1], pts[:2], pts), "alternated_cubic": (pts[:1], pts[:2], pts)}[kind]
    if seed == "cell":
        S = {_key((0, 0, 0))}
    else:
        chosen = sets[SEEDS.index(seed)]
        S = set.intersection(*[set(cells_at(_key(q))) for q in chosen])
    out, star = [len(S)], {}
    for _ in range(n):
        new = set(S)
        for c in S:
            for v in verts_of(c):
                if v not in star:
                    star[v] = cells_at(v)
                new.update(star[v])
        S = new
        out.append(len(S))
    return out


def test_e3_other_kinds_match_independent_grower():
    t = time.time()
    for kind in KINDS:
        for seed in SEEDS:
            assert cells(growth.grow_e3(kind, seed, 2)) == _indep_e3(kind, seed, 2), (kind, seed)
    print(f"# time indep E3 n=0..2: {time.time() - t:.1f}s")


@pytest.mark.slow
def test_e3_independent_grower_to_layer_4():
    # The source of the n = 3, 4 pins in growth/tests.rs (Kelvin, rhombic, octet rows).
    t = time.time()
    for kind in KINDS[1:]:
        for seed in SEEDS:
            assert cells(growth.grow_e3(kind, seed, 4)) == _indep_e3(kind, seed, 4), (kind, seed)
    print(f"# time indep E3 n=0..4: {time.time() - t:.1f}s")


def test_e3_hand_counts():
    # Layer 0 by incidence; cell-seed layer 1: Kelvin 1 + 14; rhombic 1 + 12 + 6;
    # octahedron 1 + 18 octahedra + 32 tetrahedra; tetrahedron 16 + 23 = 39 (inclusion-exclusion).
    want0 = {"cubic": (8, 4, 2), "bitruncated_cubic": (4, 3, 2), "rhombic_dodecahedral": (6, 3, 2),
             "alternated_cubic": (14, 4, 2)}
    s1 = {"cubic": 27, "bitruncated_cubic": 15, "rhombic_dodecahedral": 19, "alternated_cubic": 51}
    for kind in KINDS:
        assert tuple(cells(growth.grow_e3(kind, s, 0))[0] for s in SEEDS[:3]) == want0[kind]
        assert cells(growth.grow_e3(kind, "cell", 1)) == [1, s1[kind]]
    x = E3Complex("alternated_cubic")
    assert cells(growth.grow(x, [(1, 1, 1)], 1)) == [1, 39]
    assert len(E3Complex("rhombic_dodecahedral").cells_at((1, 1, 1))) == 4


def test_e3_cubic_boxes():
    # Hand: a x b x c box of cubes; chi = 1; surface 2(ab+bc+ca)+2 vertices, 4(ab+bc+ca) edges.
    for seed, dims in zip(SEEDS, ((2, 2, 2), (1, 2, 2), (1, 1, 2), (1, 1, 1))):
        g = growth.grow_e3("cubic", seed, 4)
        assert g["stopped"] is None and g["scale"] == 2
        for n, s in enumerate(g["stats"]):
            a, b, c = (2 * n + d for d in dims)
            p = a * b + b * c + c * a
            assert (s["cells"], s["euler"]) == (a * b * c, 1)
            assert (s["boundary_vertices"], s["boundary_edges"]) == (2 * p + 2, 4 * p)


def test_e3_cell_tables():
    # Hand: (V, E, F) of cube, truncated octahedron, rhombic dodecahedron, octahedron, tetrahedra.
    want = {"cubic": [(8, 12, 6)], "bitruncated_cubic": [(24, 36, 14)], "rhombic_dodecahedral": [(14, 24, 12)],
            "alternated_cubic": [(6, 12, 8), (4, 6, 4), (4, 6, 4)]}
    for kind, rows in want.items():
        x = E3Complex(kind)
        assert [(len(c["offsets"]), len(c["edges"]), len(c["faces"])) for c in x.classes] == rows


# ------------------------------------------------------------------ planar tables (built here)
def _tri(a, b, s):
    return [(a, b), (a + 1, b), (a, b + 1)] if s == 0 else [(a + 1, b), (a + 1, b + 1), (a, b + 1)]


def _hex_around(x, y):
    star = [(x, y, 0), (x - 1, y, 0), (x, y - 1, 0), (x - 1, y, 1), (x, y - 1, 1), (x - 1, y - 1, 1)]
    cyc = [star[0]]
    while len(cyc) < 6:
        last = _tri(*cyc[-1])
        cyc.append(next(t for t in star if t not in cyc and len(set(_tri(*t)) & set(last)) == 2))
    return cyc


def _planar(name, r):
    rng = [(a, b) for a in range(-r, r) for b in range(-r, r)]
    if name == "44":
        polys, q = [[(a, b), (a + 1, b), (a + 1, b + 1), (a, b + 1)] for a, b in rng], 4
    elif name == "36":
        polys, q = [_tri(a, b, s) for a, b in rng for s in (0, 1)], 6
    else:
        polys, q = [_hex_around(a, b) for a, b in rng], 3
    ids: dict = {}
    rows = [[ids.setdefault(v, len(ids)) for v in p] for p in polys]
    deg: dict = {}
    for row in rows:
        for v in row:
            deg[v] = deg.get(v, 0) + 1
    return {"polygons": rows, "rim": [v for v, d in deg.items() if d < q]}, ids


FORMS = {"44": (lambda n: (2 * n + 1) ** 2, lambda n: (2 * n + 2) ** 2, lambda n: (2 * n + 1) * (2 * n + 2)),
         "36": (lambda n: 6 * n * n + 6 * n + 1, lambda n: 6 * (n + 1) ** 2, lambda n: 2 * (3 * n + 1) * (n + 1)),
         "63": (lambda n: 3 * n * n + 3 * n + 1, lambda n: 3 * (n + 1) ** 2, lambda n: (3 * n + 2) * (n + 1))}


@pytest.mark.parametrize("name", sorted(FORMS))
def test_planar_closed_forms(name):
    # Plan §2.4 (hand for {4,4}, {6,3} face, {3,6} vertex; the rest verified to n = 8 by
    # independent growers). n = 0..6 here; the Rust test runs n = 0..8.
    cx, ids = _planar(name, 9)
    x = FiniteComplex.from_dict(cx)
    v0 = ids[(0, 0) if name != "63" else (0, 0, 0)]
    tile = x.cells_at(v0)[0]
    nb = next(w for w in x.vertices_of(tile) if w != v0 and sorted((v0, w)) in [sorted(e) for e in x.ev])
    face, vertex, edge = FORMS[name]
    for seed, form in (({"kind": "cell", "at": tile}, face), ({"kind": "vertex", "at": v0}, vertex),
                       ({"kind": "edge", "at": [v0, nb]}, edge)):
        g = growth.grow_finite(cx, seed, 6)
        assert cells(g) == [form(n) for n in range(7)], (name, seed)
        assert g["stopped"] is None
        assert all(s["euler"] == 1 and s["boundary_vertices"] == s["boundary_edges"] for s in g["stats"])


def test_rim_closed_and_hosohedron():
    cx, ids = _planar("44", 2)
    x = FiniteComplex.from_dict(cx)
    tile = next(c for c in x.cells_at(ids[(0, 0)]) if ids[(1, 1)] in x.vertices_of(c))
    g = growth.grow_finite(cx, {"kind": "cell", "at": tile}, 8)
    assert cells(g) == [1, 9] and g["stopped"] == {"kind": "rim", "layer": 2}
    # Hand: cube surface, face seed 1, 5, 6 with chi 1, 1, 2, then closed.
    cube = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    g = growth.grow_finite({"polygons": cube}, {"kind": "cell", "at": 0}, 10)
    assert cells(g) == [1, 5, 6] and [s["euler"] for s in g["stats"]] == [1, 1, 2]
    assert g["stopped"] == {"kind": "closed", "layer": 2}
    # Hosohedron {2,3}: a vertex seed is the whole sphere (chi 2); an edge seed is 2 digons.
    hoso = {"cell_vertices": [[0, 1]] * 3, "cell_edges": [[0, 1], [1, 2], [2, 0]], "edge_vertices": [[0, 1]] * 3}
    g = growth.grow_finite(hoso, {"kind": "vertex", "at": 0}, 5)
    assert (cells(g), g["stats"][0]["euler"], g["stopped"]) == ([3], 2, {"kind": "closed", "layer": 0})
    assert cells(growth.grow_finite(hoso, {"kind": "edge", "at": 0}, 5)) == [2, 3]
    with pytest.raises(ValueError, match="names 3 edges"):
        growth.grow_finite(hoso, {"kind": "edge", "at": [0, 1]}, 5)


class _Ideal:
    def vertices_of(self, c): return [0, 1]
    def edges_of(self, c): return []
    def faces_of(self, c): return []
    def on_rim(self, v): return False
    def cells_on_edge(self, e): return []
    def cells_on_face(self, f): return []

    def cells_at(self, v):
        if v == 1:
            raise GrowthError("infinite", "ideal vertex")
        return [0]


def test_errors_caps_and_refusals():
    g = growth.grow(_Ideal(), [0, 0], 3)
    assert g["layers"] == [[0]] and g["stats"][0]["boundary_vertices"] == 1
    assert g["stopped"] == {"kind": "infinite", "layer": 1, "reason": "ideal vertex"}
    assert growth.grow(_Ideal(), [0], 0)["stopped"] is None
    assert growth.grow(_Ideal(), [], 3) == {"layers": [], "stats": [], "stopped": None}
    x = E3Complex("cubic")
    assert cells(growth.grow(x, [(0, 0, 0)], 1, max_cells=27)) == [1, 27]
    g = growth.grow(x, [(0, 0, 0)], 5, max_cells=26)
    assert cells(g) == [1] and g["stopped"] == {"kind": "cell_cap", "layer": 1, "partial": 27}
    assert growth.grow(x, [(0, 0, 0), (2, 0, 0)], 5, max_cells=1)["stopped"] == {"kind": "cell_cap", "layer": 0, "partial": 2}
    with pytest.raises(ValueError, match="Dirichlet"):
        growth.grow_e3("cubic", "dirichlet", 1)
    with pytest.raises(ValueError, match="coordinates"):
        growth.grow_finite({"polygons": [[0, 1, 2]]}, {"kind": "dirichlet"}, 1)
    g = growth.grow(x, [(0, 0, 0)], 0)
    assert json.dumps(g["stats"]) == ('[{"cells": 1, "new_cells": 1, "boundary_vertices": 8, '
                                      '"boundary_edges": 12, "euler": 1}]')


def _rust_growth():
    rs = _rust.get()
    if rs is None or not hasattr(rs, "grow_e3_json"):
        return None
    try:
        rs.grow_e3_json("cubic", "cell", 0, 10)
    except NotImplementedError:
        return None
    return rs


def test_parity_with_rust(monkeypatch):
    rs = _rust_growth()
    if rs is None:
        pytest.skip("papertilling_rs without the WP-G1 growth bindings (rebuilt at integration)")
    for kind in KINDS:
        for seed in SEEDS:
            monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
            py = growth.grow_e3(kind, seed, 3, 2000)
            monkeypatch.delenv("PAPERTILLING_NO_RUST")
            assert json.loads(rs.grow_e3_json(kind, seed, 3, 2000)) == py, (kind, seed)
    cx, _ = _planar("63", 3)
    for seed in ({"kind": "vertex", "at": 0}, {"kind": "cell", "at": 5}):
        monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
        py = growth.grow_finite(cx, seed, 9)
        monkeypatch.delenv("PAPERTILLING_NO_RUST")
        assert json.loads(rs.grow_finite_json(json.dumps(cx), json.dumps(seed), 9, 50_000)) == py
