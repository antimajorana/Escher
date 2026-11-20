"""C1 WP-G2: seeded growth of the E³ / H³ / S³ honeycombs, the face-shell rule, paracompact refusals and balls.

Provenance of the numbers (plan §2.4): [hand] cell shapes, vertex stars, the layer-1 formula, boxes of cubes,
the small polychora; [COMPUTATIONAL EVIDENCE, same coset method] the 600-cell layers and the H³ face shells,
from the critique's independent scripts `cox_grow.py` and `face_shells.py`.
"""
from __future__ import annotations

import math
import time

import pytest

from papertilling import _rust
from papertilling import honeycomb3d as hc
from papertilling._honeycomb3d_ball import _py_ball
from papertilling._honeycomb3d_growth import MAX_GROWTH_CELLS
from papertilling._honeycomb3d_growth_wiring import _py_grow

COMPACT = [(4, 3, 5), (5, 3, 4), (5, 3, 5), (3, 5, 3)]
PARACOMPACT = [(3, 3, 6), (4, 3, 6), (5, 3, 6), (3, 4, 4), (4, 4, 3), (4, 4, 4), (3, 6, 3), (6, 3, 3), (6, 3, 4),
               (6, 3, 5), (6, 3, 6)]


def _sym(pqr):
    return "{%d,%d,%d}" % pqr


def _totals(g):
    return [s["cells"] for s in g["stats"]]


def test_layer_one_hand_formula_and_vertex_stars():
    # [hand] cell (V, E, F), c_v = cells of the vertex figure; new cells at layer 1 =
    # F + E(r-3) + V(c_v - 1 - q - q(r-3)) = 110, 62, 272, 92.
    shapes = {(4, 3, 5): (8, 12, 6, 20), (5, 3, 4): (20, 30, 12, 8), (5, 3, 5): (20, 30, 12, 20), (3, 5, 3): (12, 30, 20, 12)}
    t = time.time()
    for (p, q, r), (v, e, f, cv) in shapes.items():
        g = hc.grow_honeycomb(_sym((p, q, r)), "cell", 1, draw=False)
        assert g["stats"][1]["new_cells"] == f + e * (r - 3) + v * (cv - 1 - q - q * (r - 3))
        assert hc.grow_honeycomb(_sym((p, q, r)), "vertex", 0, draw=False)["stats"][0]["cells"] == cv
        assert hc.grow_honeycomb(_sym((p, q, r)), "face", 0, draw=False)["stats"][0]["cells"] == 2
    print("# time layer-1 formula, 4 honeycombs: %.2fs" % (time.time() - t))


def test_s3_closes():
    g = hc.grow_honeycomb("{3,3,5}", "vertex", 10)
    assert _totals(g) == [20, 130, 330, 530, 600]   # cox_grow.py
    assert [s["euler"] for s in g["stats"]] == [1, 1, 1, 1, 0]
    assert g["stopped"] == {"kind": "closed", "layer": 4}
    assert hc.counts(g["complex"]) == (120, 720, 1200, 600)   # Coxeter, Regular Polytopes, Table I(ii)
    for pqr, want in [((3, 3, 3), [4, 5]), ((3, 3, 4), [8, 16]), ((4, 3, 3), [4, 8])]:   # [hand]
        g = hc.grow_honeycomb(_sym(pqr), "vertex", 10, draw=False)
        assert _totals(g) == want and g["stopped"] == {"kind": "closed", "layer": 1} and g["stats"][1]["euler"] == 0


def test_face_shells_and_e3_boxes():
    cum = [1, 7, 37, 163]   # face_shells.py: 1, 6, 30, 126 new
    assert _totals(hc.grow_honeycomb("{4,3,5}", "cell", 3, rule="face_shells", draw=False)) == cum
    octa = [(2 * n + 1) * (2 * n * n + 2 * n + 3) // 3 for n in range(4)]   # [hand] L1 balls of cubes
    assert _totals(hc.grow_honeycomb("cubic", "cell", 3, rule="face_shells", draw=False)) == octa
    assert _totals(hc.grow_honeycomb("{4,3,4}", "vertex", 3, draw=False)) == [(2 * n + 2) ** 3 for n in range(4)]
    g = hc.grow_honeycomb("cubic", "cell", 1)
    assert hc.counts(g["complex"]) == (64, 144, 108, 27)   # [hand] a 3x3x3 box of cubes


def test_paracompact_refusals_and_hyperideal():
    for pqr in PARACOMPACT:
        cls = hc.classify_schlafli(*pqr)
        for seed in ("vertex", "cell"):
            st = hc.grow_honeycomb(_sym(pqr), seed, 3, draw=False)["stopped"]
            want = 0 if cls["has_infinite_euclidean_cells"] or seed == "vertex" else 1
            assert st["kind"] == "infinite" and st["layer"] == want and st["reason"].startswith("[PROVED]"), (pqr, seed)
    with pytest.raises(ValueError, match="PROVED"):
        hc.grow_honeycomb("{7,3,3}", "vertex", 1)


def test_paracompact_ball_hand_cases():
    assert (lambda b: (b["faces"], b["cells"]))(hc.paracompact_ball(6, 3, 3, 0.01)) == (1, 2)
    # {3,3,6}: the central tetrahedron's faces sit at the inradius rho = acosh(w2_0).
    rho = math.acosh(hc._Coxeter(3, 3, 6).w[2][0])
    assert hc.paracompact_ball(3, 3, 6, rho - 1e-6)["faces"] == 0
    b = hc.paracompact_ball(3, 3, 6, rho + 1e-6)
    assert (b["faces"], b["cells"]) == (4, 5)
    for pqr in [(4, 4, 4), (5, 3, 6), (6, 3, 3)]:   # (6,3,3): the old chi = -163 case (shells 2, patch_layers 2)
        a, bb = _py_ball(*pqr, 1.0, 1), _py_ball(*pqr, 1.0, 2)
        assert (a["faces"], a["cells"]) == (bb["faces"], bb["cells"])
        assert hc.paracompact_ball(*pqr, 0.5)["faces"] <= a["faces"] <= hc.paracompact_ball(*pqr, 1.5)["faces"]


def test_build_honeycomb_seed_keyword():
    cx = hc.build_honeycomb(5, 3, 4, seed="cell", layers=1)
    assert len(cx["cells"]) == 63 and hc.euler(cx) == 1
    assert hc.counts(hc.build_honeycomb(4, 3, 5, shells=1)) == (32, 60, 36, 7)   # old face-adjacency patch kept


def test_cap_bounds_interned_cells():
    # {5,3,5}: vertex S0 = c_v = 20 [hand], S1 = 3 550 (cox_grow.py) is over a cap of 100, which must stop
    # the build after about `max_cells` interned cells (cap guard), with partial = cap + 1.
    t = time.time()
    g = _py_grow("{5,3,5}", "vertex", 10, 100, "touching")
    print("# time python {5,3,5} vertex over cap 100: %.2fs" % (time.time() - t))
    assert _totals(g) == [20]
    assert g["stopped"] == {"kind": "cell_cap", "layer": 1, "partial": 101}
    assert g["precision"]["keys"][0] <= 100 + 20


@pytest.mark.slow
def test_vertex_s2_python_fallback():
    t = time.time()
    g = _py_grow("{4,3,5}", "vertex", 2, 250_000, "touching")
    print("# time python {4,3,5} vertex S2: %.2fs" % (time.time() - t))
    assert _totals(g) == [20, 760, 23100]   # cox_grow.py
    assert g["boundary_only"] and g["precision"]["min_gap"] is None


def _has_rust_growth():
    rs = _rust.get() if _rust.enabled() else None
    return rs is not None and hasattr(rs, "grow_honeycomb_json") and hasattr(rs, "honeycomb3d_paracompact_ball_json")


@pytest.mark.skipif(not _has_rust_growth(), reason="needs a papertilling_rs extension built with WP-G2")
def test_parity_with_rust():
    keys = ("name", "regime", "seed", "rule", "stats", "stopped", "boundary_only")
    cases = [("{4,3,5}", "cell", 1, "touching"), ("{5,3,4}", "vertex", 1, "touching"), ("{3,5,3}", "edge", 1, "touching"),
             ("{3,3,5}", "vertex", 10, "touching"), ("{4,3,5}", "cell", 2, "face_shells"), ("{4,3,6}", "cell", 2, "touching"),
             ("{6,3,3}", "vertex", 2, "touching"), ("bitruncated_cubic", "face", 2, "touching"),
             ("alternated_cubic", "cell", 2, "face_shells"), ("{4,3,5}", "vertex", 2, "touching")]
    for kind, seed, n, rule in cases:
        rs = hc.grow_honeycomb(kind, seed, n, rule=rule)
        py = _py_grow(kind, seed, n, MAX_GROWTH_CELLS, rule)
        assert {k: rs[k] for k in keys} == {k: py[k] for k in keys}, (kind, seed, rule)
        assert hc.counts(rs["complex"]) == hc.counts(py["complex"]), kind
        if py["precision"]:
            assert rs["precision"]["keys"] == py["precision"]["keys"]
    for pqr in [(6, 3, 3), (3, 3, 6), (4, 4, 4)]:
        rs, py = hc.paracompact_ball(*pqr, 1.2), _py_ball(*pqr, 1.2)
        assert (rs["faces"], rs["cells"], rs["reason"]) == (py["faces"], py["cells"], py["reason"])
        assert hc.counts(rs["complex"]) == hc.counts(py["complex"])
