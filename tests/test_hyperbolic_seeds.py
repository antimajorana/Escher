"""C1 WP-G3: vertex, edge and tile seeds for {p, q} growth under the touching rule (pure-Python path).

Sources of the pinned numbers (none is read from the code under test):
* hand: seed letters (boundary_counts.rs doc), the layer-1 formulas, the {5,4} tile shells 1, 11, 51, 201, 761, the
  cube 1, 5, 6 and the hosohedron / dihedron closures;
* hand-derived or fitted E^2 closed forms of plan §2.4 (fits checked n = 0..8 by independent growers);
* [COMPUTATIONAL EVIDENCE] `cox_grow.py` (the C1 critique's independent script, outside the repository): a Coxeter-group grower in root coordinates that
  shares no code with papertilling.
The parity test runs only when the compiled extension has the seeded functions (built after C1 WP-G3).
"""
from __future__ import annotations

import json
import math
from itertools import accumulate

import pytest

from papertilling import _hyperbolic_seeds as seeds
from papertilling import _noneuclid_pq as pq
from papertilling.hyperbolic import build_hyperbolic_tiling

# cumulative |S_0..S_3| per seed (cell, vertex, edge), from cox_grow.py
COX = {
    (5, 4): ([1, 11, 51, 201], [4, 24, 100, 384], [2, 16, 70, 272]),
    (7, 3): ([1, 8, 29, 85], [3, 15, 48, 135], [2, 12, 40, 114]),
    (4, 5): ([1, 13, 61, 241], [5, 30, 125, 480], [2, 18, 80, 312]),
    (3, 7): ([1, 16, 61, 181], [7, 35, 112, 315], [2, 20, 72, 210]),
    (8, 3): ([1, 9, 41, 161], [3, 18, 75, 288], [2, 14, 60, 232]),
}
SEEDS = ("cell", "vertex", "edge")


@pytest.fixture
def no_rust(monkeypatch):
    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")


@pytest.mark.parametrize("pq_pair", sorted(COX))
def test_counters_and_geometry_match_cox_grow(no_rust, pq_pair):
    p, q = pq_pair
    for seed, want in zip(SEEDS, COX[pq_pair]):
        letters = seeds.vertex_completion_counts_seeded(p, q, seed, 3)
        a0, word = seeds.seed_word(p, q, seed)
        assert seeds.boundary_word_counts_from(p, q, a0, word, 3) == letters
        assert list(accumulate(letters)) == want, (p, q, seed)
        if (p, q) in ((5, 4), (7, 3)):  # the geometric grower, on two pairs (the others cost more in Python)
            vc = seeds.grow_vertex_completion_seeded(p, q, seed, 3, 10**6)
            assert vc.stopped is None and vc.shell_sizes == letters


def test_layer_one_hand_formulas(no_rust):
    # vertex seed: q (q-3) + q (p-3)(q-2) for q > 3, 3 (p-3) for q = 3; edge seed: 2 (q-3) + 2 (p-2)(q-2), 2 (p-2).
    for p, q in [(5, 4), (4, 5), (6, 4), (7, 3), (8, 3), (5, 5), (3, 7)]:
        v = q * (q - 3) + q * (p - 3) * (q - 2) if q > 3 else 3 * (p - 3)
        e = 2 * (q - 3) + 2 * (p - 2) * (q - 2) if q > 3 else 2 * (p - 2)
        assert seeds.vertex_completion_counts_seeded(p, q, "vertex", 1)[1] == v
        assert seeds.vertex_completion_counts_seeded(p, q, "edge", 1)[1] == e


def test_counters_on_euclidean_closed_forms(no_rust):
    forms = {
        (4, 4): (lambda n: (2 * n + 1) ** 2, lambda n: (2 * n + 2) ** 2, lambda n: (2 * n + 1) * (2 * n + 2)),
        (3, 6): (lambda n: 6 * n * n + 6 * n + 1, lambda n: 6 * (n + 1) ** 2, lambda n: 2 * (3 * n + 1) * (n + 1)),
        (6, 3): (lambda n: 3 * n * n + 3 * n + 1, lambda n: 3 * (n + 1) ** 2, lambda n: (3 * n + 2) * (n + 1)),
    }
    for (p, q), fs in forms.items():
        for seed, f in zip(SEEDS, fs):
            a0, cnt = seeds.seed_letters(p, q, seed)
            assert list(accumulate(seeds.letter_counts_from(p, q, a0, cnt, 8))) == [f(n) for n in range(9)], (p, q, seed)
            a0, word = seeds.seed_word(p, q, seed)
            assert list(accumulate(seeds.boundary_word_counts_from(p, q, a0, word, 8))) == [f(n) for n in range(9)]


def _euler(t) -> int:
    used = {v for f in t["faces"] for v in f}
    return len(used) - len(t["edges"]) + len(t["faces"])


def test_seeded_patches_zero_based(no_rust):
    for seed, totals in zip(SEEDS, COX[(5, 4)]):
        for n, want in enumerate(totals):
            t = build_hyperbolic_tiling(5, 4, "RegularPQ", n, 0, seed=seed)
            assert (len(t["faces"]), t["complete_through_shell"], t["layers"], t["seed"]) == (want, n, n, seed)
            assert _euler(t) == 1
    assert len(build_hyperbolic_tiling(5, 4, "DualQP", 0, 0, seed="vertex")["faces"]) == 5
    for style, delta in (("RectifiedKagome", 0), ("RegularPQ", 1)):
        t = build_hyperbolic_tiling(7, 3, style, 0, delta)
        assert t["faces"] and t["layers"] == 0


def test_seed_refusals(no_rust):
    with pytest.raises(ValueError, match="kagome cluster"):
        build_hyperbolic_tiling(7, 3, "RectifiedKagome", 2, 0, seed="vertex")
    with pytest.raises(ValueError, match="wedge core"):
        build_hyperbolic_tiling(7, 3, "RegularPQ", 2, 1, seed="edge")
    with pytest.raises(ValueError, match="E² lattices only"):
        build_hyperbolic_tiling(5, 4, "RegularPQ", 1, 0, seed="dirichlet")
    with pytest.raises(ValueError, match="unknown seed"):
        build_hyperbolic_tiling(5, 4, "RegularPQ", 1, 0, seed="corner")
    with pytest.raises(ValueError, match="plane inflation"):
        pq.build_unified_pq(4, 4, 2, seed="vertex")


def test_s2_closure(no_rust):
    cube = pq.build_unified_pq(4, 3, 5, seed="cell")["growth"]
    assert (cube["layer_counts"], cube["layer_euler"]) == ([1, 5, 6], [1, 1, 2])
    assert cube["stopped"].startswith("closed at layer 2")
    ico = pq.build_unified_pq(3, 5, 5)["growth"]
    assert (ico["layer_counts"], ico["layer_euler"]) == ([1, 10, 19, 20], [1, 1, 1, 2])
    for (p, q), seed, want in [((3, 5), "vertex", [5, 15, 20]), ((5, 3), "edge", [2, 8, 12]), ((3, 4), "edge", [2, 8])]:
        g = pq.build_unified_pq(p, q, 5, seed=seed)["growth"]
        assert g["layer_counts"] == want and g["layer_euler"][-1] == 2
    for q in (2, 3, 6):
        v = pq.build_unified_pq(2, q, 3, seed="vertex")["growth"]
        assert (v["layer_counts"], v["stopped"][:17]) == ([q], "closed at layer 0")
        assert pq.build_unified_pq(2, q, 3)["growth"]["layer_counts"] == [1, q]
    assert pq.build_unified_pq(5, 2, 3)["growth"]["layer_counts"] == [1, 2]
    assert pq.build_unified_pq(4, 3, 1)["growth"]["stopped"] is None


def test_unified_h2_growth_and_wythoff_stop(no_rust):
    h = pq.build_unified_pq(5, 4, 1, seed="vertex")
    assert (h["n_faces"], h["growth"]["layer_counts"], h["growth"]["stopped"]) == (24, [4, 24], None)
    assert pq.build_unified_pq(5, 4, 0)["n_faces"] == 1
    assert pq.build_unified_pq(5, 4, 9)["growth"]["stopped"]
    assert pq.build_unified_pq(4, 4, 0)["n_faces"] == 1
    deep = pq.build_wythoff(7, 3, "Regular", layers=20)
    assert deep["stopped"] and "face budget" in deep["stopped"]
    assert pq.build_wythoff(7, 3, "Regular", layers=1)["stopped"] is None
    assert pq.build_wythoff(4, 3, "Truncated", layers=1)["stopped"] is None


def _seeded_extension():
    rs = pytest.importorskip("papertilling_rs")
    if not hasattr(rs, "hyperbolic_tiling_seeded_json") or not hasattr(rs, "unified_pq_seeded_json"):
        pytest.skip("installed papertilling_rs predates C1 WP-G3 (seeded layers); rebuild it")
    return rs


@pytest.mark.parametrize("p,q,layers", [(5, 4, 3), (7, 3, 4), (4, 5, 2)])
@pytest.mark.parametrize("seed", SEEDS)
def test_parity_with_rust_extension(monkeypatch, p, q, layers, seed):
    rs = _seeded_extension()
    r = json.loads(rs.hyperbolic_tiling_seeded_json(p, q, "RegularPQ", layers, 0, seed))
    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
    f = build_hyperbolic_tiling(p, q, "RegularPQ", layers, 0, seed=seed)
    assert f.keys() == r.keys()
    for key in ("layers", "complete_through_shell", "growth_rule", "face_shell", "faces", "seed"):
        assert f[key] == r[key], key
    assert max(math.dist(a, b) for a, b in zip(f["poincare_pts"], r["poincare_pts"])) < 1e-9
    for pp, qq in ((4, 3), (3, 5), (2, 4), (5, 4)):
        ru = json.loads(rs.unified_pq_seeded_json(pp, qq, layers, seed))
        fu = pq.build_unified_pq(pp, qq, layers, seed=seed)
        assert fu.keys() == ru.keys()
        for key in ("layer_counts", "layer_euler", "stopped", "seed"):
            assert fu["growth"][key] == ru["growth"][key], (pp, qq, key)


def test_budget_estimate_is_exact(no_rust):
    # {12,12} vertex seed: a_1 = q (q-3) + q (p-3)(q-2) = 1188 (hand); the stop quotes the exact next shell.
    vc = seeds.grow_vertex_completion_seeded(12, 12, "vertex", 64, 1000)
    assert vc.stopped == "face budget 1000: shell 1 (~1188 faces) not built" and vc.shell_sizes == [12]


def test_seeded_drawings_centred_and_unreached_faces(no_rust):
    v = build_hyperbolic_tiling(5, 4, "RegularPQ", 0, 0, seed="vertex")
    assert any(math.hypot(*z) < 1e-12 for z in v["poincare_pts"])
    e = build_hyperbolic_tiling(5, 4, "RegularPQ", 0, 0, seed="edge")
    pts = e["poincare_pts"]
    assert any(math.hypot(a[0] + b[0], a[1] + b[1]) < 1e-12 and math.hypot(*a) > 1e-3 for a in pts for b in pts)
    short = pq.build_unified_pq(4, 3, 1)["growth"]
    assert short["stopped"] is None and short["face_layer"].count(None) == 1


def test_deep_requests_stop_at_caps(no_rust):
    # Pilot (2026-11-20): each of these builds takes 0.2-0.7 s in Python.
    for args in [(7, 3, "RectifiedKagome", 64, 0), (7, 3, "RegularPQ", 64, 1), (4, 5, "RegularPQ", 64, -1)]:
        t = build_hyperbolic_tiling(*args)
        assert len(t["faces"]) <= 1000 and t["complete_through_shell"] < 64 and "omitted" in t["growth_rule"]
    # E^2 from one face at 12 layers: (2n+1)^2 [hand], 6n^2+6n+1 [fit], 3n^2+3n+1 [hand]; a disk.
    for (p, q), want in (((4, 4), 625), ((3, 6), 937), ((6, 3), 469)):
        u = pq.build_unified_pq(p, q, 12)
        assert (u["n_faces"], u["euler_chi"]) == (want, 1)
