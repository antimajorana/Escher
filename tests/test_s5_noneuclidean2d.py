"""Stage S5 non-Euclidean 2D: known-answer tests for papertilling.noneuclidean2d.

Every expected value is a published fact or a hand count, with the source next to it; none is copied
from the code under test. The `ne` fixture forces the pure-Python fallback (PAPERTILLING_NO_RUST=1), so
these tests run identically with or without the compiled extension; `test_rust_parity` compares the
two paths and skips cleanly when `papertilling_rs` (with the Stage 5 functions) is not importable.
"""
from __future__ import annotations

import math
from fractions import Fraction

import pytest

import papertilling.noneuclidean2d as ne_mod


@pytest.fixture
def ne(monkeypatch):
    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
    return ne_mod


def _spread(pts, edges, dist) -> float:
    lens = [dist(pts[u], pts[v]) for u, v in edges]
    return max(lens) - min(lens)


def _chord(a, b) -> float:
    return math.dist(a, b)


# Coxeter, Regular Polytopes (3rd ed. 1973), Table I(i): (p, q, V, E, F).
PLATONIC = [(3, 3, 4, 6, 4), (3, 4, 6, 12, 8), (4, 3, 8, 12, 6), (3, 5, 12, 30, 20), (5, 3, 20, 30, 12)]


def test_curvature_sign_of_pq(ne):
    # sign(1/p + 1/q - 1/2): S^2 for the Platonic five and {2,q}, E^2 for {4,4}, {3,6}, {6,3}, else H^2.
    for p, q, *_ in PLATONIC:
        assert ne.geometry_kind_pq(p, q) == "Spherical"
    for p, q in [(4, 4), (3, 6), (6, 3)]:
        assert ne.delta_pq(p, q) == 0 and ne.geometry_kind_pq(p, q) == "Euclidean"
    for p, q in [(7, 3), (3, 7), (5, 4), (4, 5), (8, 8)]:
        assert ne.geometry_kind_pq(p, q) == "Hyperbolic"
    assert ne.geometry_kind_pq(2, 9) == "Spherical"
    with pytest.raises(ValueError):
        ne.delta_pq(1, 4)


@pytest.mark.parametrize("p,q,v,e,f", PLATONIC)
def test_platonic_counts(ne, p, q, v, e, f):
    t = ne.build_unified_pq(p, q, 1)
    assert (t["n_vertices"], t["n_edges"], t["n_faces"], t["euler_chi"]) == (v, e, f, 2)
    assert all(abs(math.hypot(*x) - 1.0) < 1e-11 for x in t["coords_3d"])
    assert all(len(face) == p for face in t["faces"])


def test_hosohedra_dihedra(ne):
    # {2,q}: 2 vertices, q edges, q lunes; {p,2}: p vertices, p edges, 2 hemispheres (Coxeter 1973, 1.7).
    for q in range(2, 7):
        t = ne.build_unified_pq(2, q, 1)
        assert (t["n_vertices"], t["n_edges"], t["n_faces"]) == (2, q, q)
    for p in range(3, 7):
        t = ne.build_unified_pq(p, 2, 1)
        assert (t["n_vertices"], t["n_edges"], t["n_faces"]) == (p, p, 2)


def test_euclidean_patches_and_hyperbolic_shells(ne):
    # Hand counts of vertex completion: {4,4} (2n+1)^2; {6,3} 3n(n+1)+1; {3,6} 13 then 50 - 13 = 37
    # (12 vertices x 6 triangles, Euler on a disk). {7,3}: shells 1 + 7 + 21 = 29.
    for (p, q), (f1, f2) in {(4, 4): (9, 25), (6, 3): (7, 19), (3, 6): (13, 37)}.items():
        assert ne.build_unified_pq(p, q, 1)["n_faces"] == f1
        t = ne.build_unified_pq(p, q, 2)
        assert t["n_faces"] == f2 and t["geometry_kind"] == "Euclidean"
        assert _spread(t["coords_2d"], t["edges"], _chord) < 1e-9
        assert abs(_chord(*[t["coords_2d"][i] for i in t["edges"][0]]) - 1.0) < 1e-9
    h = ne.build_unified_pq(7, 3, 2)
    assert h["geometry_kind"] == "Hyperbolic" and h["n_faces"] == 29 and h["euler_chi"] == 1


# Conway, Burgiel & Goodman-Strauss, The Symmetries of Things (2008), Part I: the 17 plane groups.
WALLPAPER = {"p6m": "*632", "p6": "632", "p4m": "*442", "p4": "442", "p4g": "4*2", "p3m1": "*333",
             "p31m": "3*3", "p3": "333", "pmm": "*2222", "pmg": "22*", "pgg": "22x", "cmm": "2*22",
             "pm": "**", "pg": "xx", "cm": "*x", "p2": "2222", "p1": "o"}


def test_wallpaper_orbifolds(ne):
    assert dict(ne.wallpaper_orbifold_table()) == WALLPAPER
    for wg, orb in WALLPAPER.items():
        assert ne.wallpaper_to_orbifold(wg) == orb
        assert ne.orbifold_to_wallpaper(orb) == wg
        info = ne.parse_orbifold(orb)
        assert (info["chi_num"], info["geometry_kind"], info["canonical"]) == (0, "Euclidean", orb)
    assert ne.wallpaper_to_orbifold("p2mm") == "*2222"
    assert ne.orbifold_to_wallpaper("2 * 2 2") == "cmm"
    with pytest.raises(ValueError):
        ne.orbifold_to_wallpaper("*237")


@pytest.mark.parametrize("sym,chi", [
    # Spherical: chi = 2 / |G| (|*532| = 120, |532| = 60, |*432| = 48, |*332| = 24, |*22n| = 4n).
    ("*532", Fraction(1, 60)), ("532", Fraction(1, 30)), ("*432", Fraction(1, 24)), ("*332", Fraction(1, 12)),
    ("*2.2.10", Fraction(1, 20)), ("2 2 10", Fraction(1, 10)), ("*(12)(12)", Fraction(1, 12)),
    # Hyperbolic triangle groups: chi(*pqr) = (1/p + 1/q + 1/r - 1) / 2, chi(pqr) = 2 chi(*pqr).
    ("*237", Fraction(-1, 84)), ("237", Fraction(-1, 42)), ("*(2,4,5)", Fraction(-1, 40)), ("*334", Fraction(-1, 24)),
    # Closed surfaces: genus g has chi = 2 - 2g; xxx (3 crosscaps) has chi = -1.
    ("oo", Fraction(-2)), ("xxx", Fraction(-1)), ("1", Fraction(2)),
])
def test_orbifold_euler_characteristic(ne, sym, chi):
    info = ne.parse_orbifold(sym)
    assert Fraction(info["chi_num"], info["chi_den"]) == chi
    want = "Spherical" if chi > 0 else ("Euclidean" if chi == 0 else "Hyperbolic")
    assert info["geometry_kind"] == want and info["provenance"].startswith("[PROVED]")


@pytest.mark.parametrize("bad", ["3", "*3", "35", "*35", "(10)", "*(10)", "2 7", "*2.7"])
def test_bad_orbifolds_refused(ne, bad):
    # Thurston's bad orbifolds: teardrop n, spindle mn (m != n), and their mirrored versions.
    with pytest.raises(ValueError, match=r"\[PROVED: bad orbifold"):
        ne.parse_orbifold(bad)


def test_orbifold_per_boundary_cycles_and_orientation(ne):
    assert ne.parse_orbifold("o*234 x")["canonical"] == "*432xxx"
    assert ne.parse_orbifold("*( 12 )(12)")["canonical"] == "*(12,12)"
    # Two cornered boundaries (*23*45) vs one cornered + one empty (*5432*):
    # chi = 2 - 2 - (1/4 + 1/3 + 3/8 + 2/5) = -163/120 (hand calculation).
    for sym in ("*23*45", "*45*23", "*32*54"):
        info = ne.parse_orbifold(sym)
        assert (info["canonical"], info["boundary_components"], info["wallpaper_group"]) == ("*54*32", 2, None)
        assert (info["chi_num"], info["chi_den"], info["geometry_kind"]) == (-163, 120, "Hyperbolic")
    for sym in ("*5432*", "**2345"):
        info = ne.parse_orbifold(sym)
        assert (info["canonical"], info["boundary_components"], info["chi_num"], info["chi_den"]) == ("*5432*", 2, -163, 120)
    two_c = ne.parse_orbifold("*2*2")  # chi = 2 - 2 - 2*(1/4) = -1/2; must not match pm ("**")
    assert (two_c["canonical"], two_c["wallpaper_group"], two_c["chi_num"], two_c["chi_den"]) == ("*2*2", None, -1, 2)
    # Orientable base (crosscaps == 0): global reflection reverses all boundary cycles simultaneously.
    assert ne.parse_orbifold("*234*234")["canonical"] == "*432*432"
    assert ne.parse_orbifold("*432*432")["canonical"] == "*432*432"
    assert ne.parse_orbifold("*234*432")["canonical"] == "*432*423"
    assert ne.parse_orbifold("*432*234")["canonical"] == "*432*423"
    assert (ne.parse_orbifold("*234*234")["chi_num"], ne.parse_orbifold("*234*234")["chi_den"]) == (-23, 12)
    assert ne.parse_orbifold("o*234*234")["canonical"] == "o*432*432"
    assert ne.parse_orbifold("o*234*432")["canonical"] == "o*432*423"
    # Nonorientable base (crosscaps > 0): crosscap slide reverses one boundary independently; chi = -35/12.
    assert ne.parse_orbifold("*234*234x")["canonical"] == "*432*432x"
    assert ne.parse_orbifold("*234*432x")["canonical"] == "*432*432x"
    assert (ne.parse_orbifold("*234*432x")["chi_num"], ne.parse_orbifold("*234*432x")["chi_den"]) == (-35, 12)


# The 21 vertex types with angle sum 2 pi; exactly 11 extend to vertex-transitive tilings
# (Grunbaum & Shephard, Tilings and Patterns, 1987, section 2.1).
EUCLID_21 = ["3.7.42", "3.8.24", "3.9.18", "3.10.15", "3.12.12", "4.5.20", "4.6.12", "4.8.8", "5.5.10",
             "6.6.6", "3.3.4.12", "3.4.3.12", "3.3.6.6", "3.6.3.6", "3.4.4.6", "3.4.6.4", "4.4.4.4",
             "3.3.3.4.4", "3.3.4.3.4", "3.3.3.3.6", "3.3.3.3.3.3"]
ARCHIMEDEAN_11 = {"3.12.12", "4.6.12", "4.8.8", "6.6.6", "3.6.3.6", "3.4.6.4", "4.4.4.4", "3.3.3.4.4",
                  "3.3.4.3.4", "3.3.3.3.6", "3.3.3.3.3.3"}


def test_vertex_configurations(ne):
    ok = set()
    for s in EUCLID_21:
        try:
            info = ne.parse_vertex_config(s)
        except ValueError as exc:
            assert "[PROVED: odd-gon parity obstruction]" in str(exc)
            continue
        assert info["geometry_kind"] == "Euclidean" and info["angle_defect_pi_num"] == 0
        ok.add(s)
    assert ok == ARCHIMEDEAN_11
    rid = ne.parse_vertex_config("4.5.4.3")  # 3.4.5.4: Delta = pi/15, V = 4 pi / Delta = 60
    assert (rid["canonical"], rid["angle_defect_pi_num"], rid["angle_defect_pi_den"]) == ("3.4.5.4", 1, 15)
    assert rid["geometry_kind"] == "Spherical"
    assert ne.parse_vertex_config("3.3.3.3.3.3.3")["geometry_kind"] == "Hyperbolic"  # 7 pi / 3 > 2 pi
    assert ne.parse_vertex_config("3.3.3.3.4")["is_snub_chiral"]  # snub cube is chiral
    assert not ne.parse_vertex_config("3.3.4.3.4")["is_snub_chiral"]  # snub square tiling: p4g
    assert ne.parse_vertex_config("3.3.4.3.5")["is_chiral"]
    with pytest.raises(ValueError):
        ne.parse_vertex_config("3.4")


def test_spherical_wythoff_archimedean(ne):
    # Coxeter, Longuet-Higgins & Miller (1954); Cromwell, Polyhedra (1997): (symbol, V, E, F).
    cases = [((3, 5, "Truncated"), "5.6.6", (60, 90, 32), {5: 12, 6: 20}),
             ((5, 3, "Cantellated"), "3.4.5.4", (60, 120, 62), {3: 20, 4: 30, 5: 12}),
             ((3, 5, "Omnitruncated"), "4.6.10", (120, 180, 62), {4: 30, 6: 20, 10: 12}),
             ((4, 3, "Rectified"), "3.4.3.4", (12, 24, 14), {3: 8, 4: 6}),
             ((4, 3, "Snub"), "3.3.3.3.4", (24, 60, 38), {3: 32, 4: 6}),
             ((5, 3, "Snub"), "3.3.3.3.5", (60, 150, 92), {3: 80, 5: 12})]
    for (p, q, op), sym, counts, ftc in cases:
        w = ne.build_wythoff(p, q, op)
        assert w["vertex_symbol"] == sym
        assert (w["n_vertices"], w["n_edges"], w["n_faces"]) == counts
        assert {int(k): v for k, v in w["face_type_counts"].items()} == ftc
        # Uniform polyhedra have all edges equal.
        assert _spread(w["coords_3d"], w["edges"], _chord) < 1e-6, sym
    left = ne.build_wythoff(4, 3, "snub", "left")
    right = ne.build_wythoff(4, 3, "Snub", "Right")
    assert left["chirality"] == "Left" and right["chirality"] == "Right"
    assert left["provenance"].startswith("[COMPUTATIONAL EVIDENCE]")
    assert sorted(map(tuple, left["coords_3d"])) != sorted(map(tuple, right["coords_3d"]))


def test_prisms_and_antiprisms(ne):
    # 5-prism (10, 15, 7) and 5-antiprism (10, 20, 12); k = 3 antiprism = octahedron, z = +-1/sqrt 3.
    prism = ne.build_wythoff(2, 5, "Truncated")
    assert (prism["n_vertices"], prism["n_edges"], prism["n_faces"]) == (10, 15, 7)
    assert _spread(prism["coords_3d"], prism["edges"], _chord) < 1e-9
    for hand in ("Left", "Right"):
        anti = ne.build_wythoff(2, 5, "Snub", hand)
        assert (anti["n_vertices"], anti["n_edges"], anti["n_faces"]) == (10, 20, 12)
        assert _spread(anti["coords_3d"], anti["edges"], _chord) < 1e-9
    octa = ne.build_wythoff(2, 3, "Snub")
    assert all(abs(abs(x[2]) - 1.0 / math.sqrt(3.0)) < 1e-12 for x in octa["coords_3d"])


def test_fallback_refuses_unsupported_wythoff(ne):
    with pytest.raises(ValueError):
        ne.build_wythoff(7, 3, "Truncated")
    with pytest.raises(ValueError):
        ne.build_wythoff(4, 3, "Elongated")


def test_models_and_geodesics(ne):
    z1, z2 = [0.25, -0.35], [-0.45, 0.30]
    w1, w2 = ne.poincare_to_uhp(z1), ne.poincare_to_uhp(z2)
    assert math.dist(ne.uhp_to_poincare(w1), z1) < 1e-13
    assert ne.poincare_to_uhp([0.0, 0.0]) == pytest.approx([0.0, 1.0])  # Cayley map sends 0 to i
    d = ne.hyp_distance_poincare(z1, z2)
    assert abs(d - ne.hyp_distance_uhp(w1, w2)) < 1e-12
    assert abs(ne.hyp_distance_poincare([0.0, 0.0], [0.5, 0.0]) - 2.0 * math.atanh(0.5)) < 1e-14
    assert abs(ne.hyp_distance_uhp([0.0, 1.0], [0.0, math.e]) - 1.0) < 1e-14  # d(i, e i) = ln e
    k = ne.poincare_to_klein(z1)
    assert math.dist(ne.klein_to_poincare(k), z1) < 1e-13
    x, y, t = ne.poincare_to_hyperboloid(z1)
    assert abs(x * x + y * y - t * t + 1.0) < 1e-12
    assert math.dist(ne.hyperboloid_to_poincare([x, y, t]), z1) < 1e-13
    arc = ne.geodesic_arc_poincare(z1, z2, 8)
    assert len(arc) == 9 and abs(ne.hyp_distance_poincare(z1, arc[4]) - 0.5 * d) < 1e-11
    arc_u = ne.geodesic_arc_uhp(w1, w2, 4)
    assert abs(ne.hyp_distance_uhp(w1, arc_u[2]) - 0.5 * d) < 1e-10
    with pytest.raises(ValueError):
        ne.poincare_to_uhp([1.2, 0.0])


def _gauss_curvature(f, u: float, v: float, h: float = 1e-4) -> float:
    def sub(a, b):
        return [a[i] - b[i] for i in range(3)]

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(3))

    p0 = f(u, v)
    ru = [c / (2 * h) for c in sub(f(u + h, v), f(u - h, v))]
    rv = [c / (2 * h) for c in sub(f(u, v + h), f(u, v - h))]
    ruu = [(f(u + h, v)[i] - 2 * p0[i] + f(u - h, v)[i]) / h ** 2 for i in range(3)]
    rvv = [(f(u, v + h)[i] - 2 * p0[i] + f(u, v - h)[i]) / h ** 2 for i in range(3)]
    ruv = [(f(u + h, v + h)[i] - f(u + h, v - h)[i] - f(u - h, v + h)[i] + f(u - h, v - h)[i]) / (4 * h * h) for i in range(3)]
    n = [ru[1] * rv[2] - ru[2] * rv[1], ru[2] * rv[0] - ru[0] * rv[2], ru[0] * rv[1] - ru[1] * rv[0]]
    nl = math.sqrt(dot(n, n))
    n = [c / nl for c in n]
    e, ff, g = dot(ru, ru), dot(ru, rv), dot(rv, rv)
    return (dot(ruu, n) * dot(rvv, n) - dot(ruv, n) ** 2) / (e * g - ff * ff)


def test_surfaces_of_curvature_minus_one(ne):
    # Pseudosphere, Dini (a^2 + b^2 = 1) and Kuen surfaces have K = -1 (classical differential geometry).
    for kind in ("Pseudosphere", "Dini", "Kuen"):
        mesh = ne.build_h2_surface_mesh(kind, 0.4, 8, 8)
        assert mesh["kind_name"] == kind and mesh["expected_gaussian_curvature"] == -1.0
        assert len(mesh["faces"]) == 64 and len(mesh["positions_3d"]) == 81
    from papertilling._noneuclid_models import _surf_pt
    for kind in ("pseudosphere", "dini", "kuen"):
        k = _gauss_curvature(lambda u, v: _surf_pt(kind, 0.4, u, v), 0.7, 1.1)
        assert abs(k + 1.0) < 1e-3, (kind, k)
    with pytest.raises(ValueError):
        ne.build_h2_surface_mesh("klein_bottle")
    emb = ne.embed_tiling_on_surface([[0.0, 0.0], [0.3, 0.1]], [[0, 1]], "bowl")
    assert emb["positions_3d"][0] == pytest.approx([0.0, 0.0, 1.0])  # origin -> hyperboloid apex


def test_papercraft_cube(ne):
    # A cube net: 6 squares, spanning tree of 5 glued edges, 12 - 5 = 7 cut edges, exact lengths.
    cube = ne.build_unified_pq(4, 3, 1)
    net = ne.unfold_papercraft_net(cube["coords_3d"], cube["faces"])
    assert len(net["pieces"]) == 6 and len(net["tree_edges"]) == 5 and len(net["cut_pairings"]) == 7
    assert net["max_edge_length_error"] < 1e-11


# (name, p, q, genus, V, E, F): E = |G|/2, V = |G|/q, F = |G|/p. Bolza: |G+| = 48 (GL(2,3));
# Klein quartic: PSL(2,7), order 168, 24 heptagons, 84 edges, 56 vertices (Levy (ed.), The Eightfold
# Way, MSRI Publ. 35, 1999); Bring's surface {5,5}: A_5, 12 pentagons (Weber, Pacific J. Math. 2005).
TGCELLS = [("bolza_8_8", 8, 8, 2, 1, 4, 1), ("bolza_8_3", 8, 3, 2, 16, 24, 6), ("klein_7_3", 7, 3, 3, 56, 84, 24),
           ("bring_5_5", 5, 5, 4, 12, 30, 12), ("genus2_6_4", 6, 4, 2, 6, 12, 4),
           ("genus2_10_5", 10, 5, 2, 2, 5, 1), ("genus2_6_6", 6, 6, 2, 2, 6, 2)]


@pytest.mark.parametrize("name,p,q,g,v,e,f", TGCELLS)
def test_tgcell_quotients(ne, name, p, q, g, v, e, f):
    t = ne.tgcell_quotient(name)
    assert (t["p"], t["q"], t["genus"], t["chi"]) == (p, q, g, 2 - 2 * g)
    assert (t["n_vertices"], t["n_edges"], t["n_faces"]) == (v, e, f)
    assert t["group_order"] == 2 * e
    assert len(t["side_pairings"]) == e - f + 1  # non-tree edges of a spanning tree of faces
    assert len(t["fundamental_polygon_poincare"]) == 2 * (e - f + 1)
    assert all(len(face) == p for face in t["faces"])


def test_tgcell_aliases_and_refusal(ne):
    assert ne.tgcell_quotient("Klein")["n_faces"] == 24
    assert set(ne.list_tgcell_quotients()) == {row[0] for row in TGCELLS}
    with pytest.raises(ValueError):
        ne.tgcell_quotient("no_such_surface")


def test_h2_large_patch_fallback(ne):
    patch = ne.build_hyperbolic_large(7, 3, 29)
    # 29 = 1 + 7 + 21 faces of {7,3} vertex completion; a disk, so V - E + F = 1.
    assert patch.n_active_faces == 29
    assert patch.n_vertices - patch.n_edges + patch.n_active_faces == 1
    assert sorted(patch.edge_neighbors(0)) == list(range(1, 8))
    assert len(patch.vertex_neighbors(0)) == 7
    assert 0 in patch.hyperbolic_ball_query([0.0, 0.0], 0.2)
    assert patch.remove_face(0) and patch.n_active_faces == 28
    assert 0 not in patch.hyperbolic_ball_query([0.0, 0.0], 0.2)
    assert patch.restore_face(0) and patch.n_active_faces == 29
    rim = patch.total_faces - 1
    side = next(s for s, nb in enumerate(patch.face_neighbors[rim]) if nb is None)
    new = patch.add_face_across_edge(rim, side)
    # Regular heptagon of {7,3}: cosh(l / 2) = cos(pi/7) / sin(pi/3) (Coxeter 1973, 4.5).
    edge = 2.0 * math.acosh(math.cos(math.pi / 7) / math.sin(math.pi / 3))
    cyc = patch.face_vertices(new)
    pts = patch.poincare_pts
    for k in range(7):
        assert abs(ne.hyp_distance_poincare(pts[cyc[k]], pts[cyc[(k + 1) % 7]]) - edge) < 1e-9
    assert patch.filter_by_region("shellrange", 0, 1) == 8
    with pytest.raises(ValueError):
        ne.build_hyperbolic_large(4, 4, 10)  # {4,4} is Euclidean: refused, not approximated
    with pytest.raises(ValueError):
        ne.build_hyperbolic_large(7, 3, 10, "no_such_kernel")


def test_rust_parity(monkeypatch):
    rs = pytest.importorskip("papertilling_rs")
    if not hasattr(rs, "unified_pq_tiling_json") or not hasattr(rs, "tgcell_quotient_json"):
        pytest.skip("papertilling_rs build predates Stage 5 noneuclidean2d bindings")
    monkeypatch.delenv("PAPERTILLING_NO_RUST", raising=False)
    for p, q, v, e, f in PLATONIC:
        t = ne_mod.build_unified_pq(p, q, 1)
        assert (t["n_vertices"], t["n_edges"], t["n_faces"]) == (v, e, f)
    for name, *_rest, v, e, f in TGCELLS:
        t = ne_mod.tgcell_quotient(name)
        assert (t["n_vertices"], t["n_edges"], t["n_faces"]) == (v, e, f)
    assert ne_mod.parse_orbifold("*237")["chi_den"] == 84
    assert ne_mod.parse_orbifold("*23*45")["canonical"] == "*54*32"
    assert ne_mod.parse_orbifold("*234*234")["canonical"] == "*432*432"
    assert ne_mod.parse_orbifold("*234*432")["canonical"] == "*432*423"
    assert ne_mod.parse_orbifold("*234*432x")["canonical"] == "*432*432x"
    orders = [4294967291, 4294967279, 4294967231, 4294967197]
    symbol = ",".join(map(str, orders))
    with pytest.raises(ValueError, match="i128"):
        rs.parse_orbifold_json(symbol)
    value = ne_mod.parse_orbifold(symbol)
    assert Fraction(value["chi_num"], value["chi_den"]) == 2 - sum(Fraction(n - 1, n) for n in orders)
    w = ne_mod.build_wythoff(5, 3, "Cantellated")
    assert (w["vertex_symbol"], w["n_vertices"], w["n_edges"], w["n_faces"]) == ("3.4.5.4", 60, 120, 62)
    snub = ne_mod.build_wythoff(7, 3, "Snub")
    assert snub["geometry_kind"] == "Hyperbolic" and snub["n_faces"] > 10
    patch = ne_mod.build_hyperbolic_large(7, 3, 29)
    assert patch.n_vertices - patch.n_edges + patch.n_active_faces == 1
