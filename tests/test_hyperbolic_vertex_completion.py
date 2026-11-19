"""Pure-Python hyperbolic fallback: BDF vertex completion against hand counts and published formulas.

Vertex completion (Boyle, Dickens, Flicker, PRX 10, 011009 (2020), arXiv:1805.02665 v2, Sec. IV):
S_0 = one p-gon, S_{n+1} = S_n + every tile sharing a vertex with S_n. Every known-answer test forces
the pure-Python fallback (PAPERTILLING_NO_RUST=1), so the counts test this module in both run modes;
the parity test compares it with the compiled extension when a build with `face_shell` is importable.
"""
from __future__ import annotations

import json
import math
from collections import Counter

import pytest

from papertilling.hyperbolic import build_hyperbolic_tiling


@pytest.fixture
def fallback(monkeypatch):
    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
    return build_hyperbolic_tiling


def _shell_sizes(t) -> list[int]:
    c = Counter(t["face_shell"])
    return [c[n] for n in range(max(c) + 1)]


def _poincare_dist(z, w) -> float:
    """Poincare-disk distance 2 atanh(|z - w| / |1 - conj(z) w|)."""
    a, b = complex(*z), complex(*w)
    return 2.0 * math.atanh(abs(a - b) / abs(1.0 - a.conjugate() * b))


def _edge_length(p: int, q: int) -> float:
    """Edge length l of {p, q} from the right triangle (centre, edge midpoint, vertex): cos(pi/p) = cosh(l/2) sin(pi/q)."""
    return 2.0 * math.acosh(math.cos(math.pi / p) / math.sin(math.pi / q))


def test_73_two_layers_hand_count(fallback):
    # Hand count, {7,3}. S_1: each seed vertex has q - 1 = 2 other faces, both across seed edges
    # (q = 3), so S_1 adds the 7 edge neighbours. Each edge neighbour A has 2 seed vertices, 2 vertices
    # w shared with the neighbouring edge faces (k = 2 faces of S_1, o = q - k = 1 outside face) and
    # 3 private vertices (k = 1, o = 2). Boundary of S_1: 7 * (1 + 3) = 28 vertices and 28 edges. S_2:
    # one face across each boundary edge, but at the 7 vertices with o = 1 the faces across both edges
    # coincide, and no corner faces (o - 2 = 0): 28 - 7 = 21. |S_2| = 1 + 7 + 21 = 29.
    t = fallback(7, 3, style="RegularPQ", layers=2, wedge_delta=0)
    assert len(t["faces"]) == 29
    assert _shell_sizes(t) == [1, 7, 21]
    assert len(t["faces"][0]) == 7 and t["face_shell"][0] == 0
    assert all(len(f) == 7 for f in t["faces"])
    assert t["complete_through_shell"] == 2
    assert t["p"] == 7 and t["q"] == 3 and t["layers"] == 2 and t["symbol_label"] == "{7,3}"
    assert t["geometry_kind"] == "Hyperbolic" and t["curvature_kappa"] < 0
    assert isinstance(t["growth_rule"], str) and "vertex completion" in t["growth_rule"]


def test_54_shells_1_10_40_hand_count(fallback):
    # Hand count, {5,4}. S_1: around each of the 5 seed vertices there are q - 1 = 3 other faces: the
    # two faces across the seed edges at it and one corner face. Edge faces: 5; corner faces: 5; total
    # p(q - 2) = 10. Boundary of S_1: each edge face has 1 private vertex (k = 1) and shares one vertex
    # with each adjacent corner face (k = 2); each corner face has 2 private vertices (k = 1). Cycle
    # length 5 * (1 + 2 + 2) = 25: 15 vertices with k = 1 (o = 3) and 10 with k = 2 (o = 2). S_2: one
    # face across each of the 25 boundary edges (distinct, as no o = 1), plus o - 2 = 1 corner face at
    # each of the 15 vertices with o = 3: 25 + 15 = 40.
    t = fallback(5, 4, style="RegularPQ", layers=2, wedge_delta=0)
    assert _shell_sizes(t) == [1, 10, 40]
    assert len(t["faces"]) == 51 and t["complete_through_shell"] == 2


@pytest.mark.parametrize("p,q", [(7, 3), (8, 3), (5, 4), (4, 5), (3, 7), (3, 8), (5, 5), (6, 4), (4, 6)])
def test_first_shell_is_p_times_q_minus_2(fallback, p, q):
    # Around each seed vertex there are q - 1 other faces; the p edge faces are each counted at two
    # vertices and the p(q - 3) corner faces at one: p + p(q - 3) = p(q - 2).
    t = fallback(p, q, style="RegularPQ", layers=1, wedge_delta=0)
    assert t["complete_through_shell"] >= 1, t["growth_rule"]
    assert _shell_sizes(t) == [1, p * (q - 2)]


def test_dual_style_swaps_p_and_q(fallback):
    # DualQP of (7, 3) draws {3, 7}: triangles, first shell 3 (7 - 2) = 15.
    t = fallback(7, 3, style="DualQP", layers=1, wedge_delta=0)
    assert t["p"] == 3 and t["q"] == 7 and t["symbol_label"] == "{3,7} dual"
    assert all(len(f) == 3 for f in t["faces"])
    assert _shell_sizes(t) == [1, 15]


@pytest.mark.parametrize("p,q,layers", [(7, 3, 3), (5, 4, 2), (3, 7, 2), (4, 5, 2)])
def test_complete_patch_is_a_disk_with_exact_edge_lengths(fallback, p, q, layers):
    # A complete S_n is a topological disk (BDF Sec. IV: vertex completion keeps the patch convex), so
    # V - E + F = 1; every edge has the {p, q} edge length (published formula above) and no two
    # distinct vertices are closer than that (hyperbolic-distance dedup did not split or merge points).
    t = fallback(p, q, style="RegularPQ", layers=layers, wedge_delta=0)
    assert t["complete_through_shell"] == layers == max(t["face_shell"]), t["growth_rule"]
    pts = t["poincare_pts"]
    assert len(pts) - len(t["edges"]) + len(t["faces"]) == 1
    ell = _edge_length(p, q)
    for u, v in t["edges"]:
        assert abs(_poincare_dist(pts[u], pts[v]) - ell) < 1e-6 * max(1.0, ell)
    for i in range(len(pts)):
        for j in range(i):
            assert _poincare_dist(pts[i], pts[j]) > 0.99 * ell


def test_points_lie_in_disk_and_on_upper_hyperboloid(fallback):
    for style in ("RegularPQ", "RectifiedKagome", "DualQP"):
        t = fallback(7, 3, style=style, layers=4, wedge_delta=0)
        assert all(z[0] ** 2 + z[1] ** 2 < 1.0 for z in t["poincare_pts"])
        assert all(k[0] ** 2 + k[1] ** 2 < 1.0 for k in t["klein_pts"])
        for x, y, tt in t["hyperboloid_pts"]:
            assert tt >= 1.0
            assert abs(tt * tt - x * x - y * y - 1.0) < 1e-6 * tt * tt


def test_kagome_cluster_rule_hand_counts(fallback):
    # H^2 rectified (p.q.p.q) kagome grown by the cluster rule:
    # * (5, 4), p > q: P_0..P_2 has 1 + 5 + 15 = 21 pentagons; vertices = 5 + 15 + 40 = 60 squares -> 81 faces.
    # * (7, 3), p > q: P_0..P_2 has 1 + 7 + 21 = 29 heptagons; vertices = 7 + 28 + 77 = 112 triangles -> 141 faces [8, 35, 98].
    # * (3, 7), p < q: P_0 = 3 heptagons around F_0, 16 triangles -> 19 at S_0; 15 + 61 = 76 at S_1.
    # * (5, 5), p == q: {5, 4} rotated by pi / 5 -> 1 + 10 + 40 = 51 pentagons.
    k54 = fallback(5, 4, style="RectifiedKagome", layers=2, wedge_delta=0)
    assert k54["symbol_label"] == "(5.4.5.4) kagome"
    assert "cluster rule" in k54["growth_rule"]
    assert len(k54["faces"]) == 81
    assert sum(len(f) == 5 for f in k54["faces"]) == 21
    assert sum(len(f) == 4 for f in k54["faces"]) == 60
    assert len(k54["poincare_pts"]) - len(k54["edges"]) + len(k54["faces"]) == 1

    k73 = fallback(7, 3, style="RectifiedKagome", layers=2, wedge_delta=0)
    assert _shell_sizes(k73) == [8, 35, 98]
    assert sum(len(f) == 7 for f in k73["faces"]) == 29
    assert sum(len(f) == 3 for f in k73["faces"]) == 112

    k37 = fallback(3, 7, style="RectifiedKagome", layers=1, wedge_delta=0)
    assert _shell_sizes(k37) == [19, 57]
    assert len(k37["faces"][0]) == 3
    cen0 = [sum(k37["poincare_pts"][i][ax] for i in k37["faces"][0]) / 3.0 for ax in (0, 1)]
    assert abs(cen0[0]) < 1e-12 and abs(cen0[1]) < 1e-12

    k55 = fallback(5, 5, style="RectifiedKagome", layers=2, wedge_delta=0)
    assert _shell_sizes(k55) == [1, 10, 40]


def test_face_cap_keeps_whole_shells(fallback):
    # {10, 7}: a_0 = 1, a_1 = 10*(7 - 2) = 50 (|S_1| = 51 <= 1000). S_1 boundary has 10*(10 - 4) + 10*4*(10 - 3)
    # = 340 vertices with o = 6 (count 5) and 50 with o = 5 (count 4), so a_2 = 340*5 + 50*4 = 1900 (> 1000).
    t = fallback(10, 7, style="RegularPQ", layers=2, wedge_delta=0)
    assert len(t["faces"]) == 51
    assert _shell_sizes(t) == [1, 50]
    assert t["complete_through_shell"] == 1
    assert "omitted" in t["growth_rule"]


def test_radius_cap_0998_truncates_and_combines_with_face_cap(fallback):
    # {5, 5}: a_0 = 1, a_1 = 15, a_2 = 105 (|S_2| = 121), a_3 = 720 (|S_3| = 841 <= 1000), a_4 = 4935 (> 1000).
    t3 = fallback(5, 5, style="RegularPQ", layers=3, wedge_delta=0)
    assert 121 < len(t3["faces"]) < 841
    assert t3["complete_through_shell"] == 2
    assert "truncated" in t3["growth_rule"] and "omitted" not in t3["growth_rule"]
    for f in t3["faces"]:
        zc = [sum(t3["poincare_pts"][i][ax] for i in f) / 5.0 for ax in (0, 1)]
        # RegularPQ caps the hyperbolic centre at 0.998; {5,5}'s Euclidean vertex centroid can exceed it by ~2.1e-4 near the rim.
        assert math.hypot(*zc) <= 0.998 + 5e-4
    t4 = fallback(5, 5, style="RegularPQ", layers=4, wedge_delta=0)
    assert len(t4["faces"]) == len(t3["faces"]) and t4["complete_through_shell"] == 2
    assert "truncated" in t4["growth_rule"] and "omitted" in t4["growth_rule"]


def test_wedge_q4_and_odd_q_gt3_hand_counts(fallback):
    # {5, 4} delta = -1 (p_core = 4, q = 4): matched edge length cosh(l/2) = cos(pi/5) / sin(pi/4),
    # alpha ~ 76.345 deg (neither 4*alpha nor 5*alpha is 2*pi). walk = (4-1)//2 = 1 edge reflection each
    # side -> 4 edge faces, plus 1 centred half-turn opposite per corner -> |S_1| = 1 + 4 + 4 = 9.
    half_edge = math.acosh(math.cos(math.pi / 5) / math.sin(math.pi / 4))
    rc = math.asinh(math.sinh(half_edge) / math.sin(math.pi / 4))
    alpha = 2.0 * math.atan(1.0 / (math.cosh(rc) * math.tan(math.pi / 4)))
    assert abs(4 * alpha - 2 * math.pi) > 0.1 and abs(5 * alpha - 2 * math.pi) > 0.1

    w_neg = fallback(5, 4, style="RegularPQ", layers=1, wedge_delta=-1)
    assert len(w_neg["faces"]) == 9 and len(w_neg["faces"][0]) == 4
    cens = [[sum(w_neg["poincare_pts"][i][ax] for i in f) / 4.0 for ax in (0, 1)] for f in w_neg["faces"]]
    for cx, cy in cens:
        rot = (-cy, cx)
        assert any(math.dist(rot, d) < 1e-9 for d in cens)
        for k in range(4):
            th = math.pi * k / 4.0
            s = 2.0 * (cx * math.cos(th) + cy * math.sin(th))
            ref = (s * math.cos(th) - cx, s * math.sin(th) - cy)
            assert any(math.dist(ref, d) < 1e-9 for d in cens)

    # Equal gaps (pi - 2*alpha ~ 27.309 deg) beside the centred opposite copy at vertex 0 (delta = -1).
    v0_id = w_neg["faces"][0][0]
    r0 = complex(*w_neg["poincare_pts"][v0_id])
    spans = []
    for f in w_neg["faces"]:
        if v0_id in f:
            pos = f.index(v0_id)
            p1 = (complex(*w_neg["poincare_pts"][f[(pos + 1) % 4]]) - r0) / (1.0 - r0.conjugate() * complex(*w_neg["poincare_pts"][f[(pos + 1) % 4]]))
            p2 = (complex(*w_neg["poincare_pts"][f[(pos + 3) % 4]]) - r0) / (1.0 - r0.conjugate() * complex(*w_neg["poincare_pts"][f[(pos + 3) % 4]]))
            t1, t2 = math.atan2(p1.imag, p1.real), math.atan2(p2.imag, p2.real)
            mid = (t1 + 0.5 * ((t2 - t1 + math.pi) % (2 * math.pi) - math.pi)) % (2 * math.pi)
            if mid > math.pi:
                mid -= 2 * math.pi
            spans.append((mid - 0.5 * alpha, mid + 0.5 * alpha))
    opp = next(sp for sp in spans if abs(sp[0] + sp[1]) < 1e-6)
    above = next(sp for sp in spans if 0.0 < sp[0] - opp[1] < 1.0)
    below = next(sp for sp in spans if 0.0 < opp[0] - sp[1] < 1.0)
    assert abs((above[0] - opp[1]) - (math.pi - 2 * alpha)) < 1e-9
    assert abs((opp[0] - below[1]) - (math.pi - 2 * alpha)) < 1e-9

    assert len(fallback(5, 4, style="RegularPQ", layers=1, wedge_delta=1)["faces"]) == 13
    assert len(fallback(4, 6, style="RegularPQ", layers=1, wedge_delta=1)["faces"]) == 21
    assert len(fallback(3, 7, style="RegularPQ", layers=1, wedge_delta=1)["faces"]) == 21


def test_wedge_kagome_rectifies_wedge_copies(fallback):
    for p, q, lay in [(7, 3, 2), (5, 4, 1)]:
        kag = fallback(p, q, style="RectifiedKagome", layers=lay, wedge_delta=-1)
        reg = fallback(p, q, style="RegularPQ", layers=lay, wedge_delta=-1)
        assert kag["symbol_label"] == f"({p}.{q}.{p}.{q}) kagome"
        assert len(kag["faces"]) == len(reg["faces"])
        assert kag["face_shell"] == reg["face_shell"]
        assert len(kag["poincare_pts"]) == len(reg["edges"])
        assert kag["poincare_pts"] != reg["poincare_pts"]


def test_layers_clamped_and_non_hyperbolic_rejected(fallback):
    assert fallback(7, 3, layers=0)["layers"] == 1
    assert fallback(7, 3, layers=9)["layers"] == 4
    for p, q in [(4, 4), (6, 3), (3, 6), (5, 3)]:
        with pytest.raises(ValueError):
            fallback(p, q)


@pytest.mark.parametrize("p,delta", [(7, -1), (7, 1), (8, -2)])
def test_wedge_core_q3_first_shell_is_one_face_per_core_edge(fallback, p, delta):
    # q = 3: at each core vertex the rule takes one reflection across each of the two edges there, so
    # S_1 is the p_core faces across the core's edges (each found from both of its ends, merged by the
    # hyperbolic-distance dedup): |S_1| = p_core = p + delta. Every face is a copy of the core.
    p_core = p + delta
    t = fallback(p, 3, style="RegularPQ", layers=1, wedge_delta=delta)
    assert t["complete_through_shell"] == 1, t["growth_rule"]
    assert _shell_sizes(t) == [1, p_core]
    assert all(len(f) == p_core for f in t["faces"])
    # Vertices: each copy is the reflection of the core across one edge, so it shares exactly that
    # edge's 2 endpoints with the core and adds p_core - 2 vertices. Two copies across the two edges at
    # a core vertex would share a further edge only if three core angles closed up to 2 pi, which is
    # exactly what a wedge defect breaks; so they share only that core vertex, and
    # V = p_core + p_core (p_core - 2) = p_core (p_core - 1).
    assert len(t["poincare_pts"]) == p_core * (p_core - 1)
    # Rotational holonomy of |delta| sectors around a p_core-gon cone has angle theta = 2*pi*|delta|/p_core;
    # in PSU(1, 1), diag(exp(i*theta/2), exp(-i*theta/2)) has unsigned half-trace |cos(theta / 2)|.
    assert abs(t["psu11_half_trace"] - abs(math.cos(math.pi * abs(delta) / p_core))) < 1e-12
    assert all(z[0] ** 2 + z[1] ** 2 < 1.0 for z in t["poincare_pts"])


def _extension_with_face_shell():
    rs = pytest.importorskip("papertilling_rs")
    if not hasattr(rs, "hyperbolic_tiling_json"):
        pytest.skip("papertilling_rs has no hyperbolic_tiling_json")
    probe = json.loads(rs.hyperbolic_tiling_json(7, 3, "RectifiedKagome", 1, 0))
    if "face_shell" not in probe or "cluster rule" not in probe.get("growth_rule", ""):
        pytest.skip("installed papertilling_rs predates Phase 2 hyperbolic cluster/wedge updates; rebuild it")
    return rs


def _assert_parity(fallback, rs, style, p, q, layers, delta):
    r = json.loads(rs.hyperbolic_tiling_json(p, q, style, layers, delta))
    f = fallback(p, q, style=style, layers=layers, wedge_delta=delta)
    assert f.keys() == r.keys()
    for key in ("p", "q", "style", "layers", "wedge_delta", "symbol_label", "geometry_kind", "complete_through_shell", "growth_rule"):
        assert f[key] == r[key], key
    assert abs(f["psu11_half_trace"] - r["psu11_half_trace"]) < 1e-12
    assert f["face_shell"] == r["face_shell"]
    assert f["faces"] == r["faces"]
    assert [list(e) for e in f["edges"]] == [list(e) for e in r["edges"]]
    assert len(f["poincare_pts"]) == len(r["poincare_pts"])
    assert max(math.dist(a, b) for a, b in zip(f["poincare_pts"], r["poincare_pts"])) < 1e-9
    assert max(math.dist(a, b) for a, b in zip(f["klein_pts"], r["klein_pts"])) < 1e-9
    assert max(max(abs(x - y) for x, y in zip(a, b)) for a, b in zip(f["hyperboloid_pts"], r["hyperboloid_pts"])) < 1e-8


@pytest.mark.parametrize("style", ["RegularPQ", "RectifiedKagome", "DualQP"])
@pytest.mark.parametrize("p,q,layers", [(7, 3, 3), (5, 4, 3), (4, 5, 2), (3, 8, 2), (8, 8, 2), (5, 5, 3)])
@pytest.mark.parametrize("delta", [0, -1, 1])
def test_parity_with_rust_extension(fallback, style, p, q, layers, delta):
    _assert_parity(fallback, _extension_with_face_shell(), style, p, q, layers, delta)


@pytest.mark.parametrize("style", ["RegularPQ", "RectifiedKagome", "DualQP"])
def test_parity_10_7_delta0_with_rust_extension(fallback, style):
    _assert_parity(fallback, _extension_with_face_shell(), style, 10, 7, 2, 0)
