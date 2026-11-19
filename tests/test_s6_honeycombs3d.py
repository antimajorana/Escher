"""Stage 6: {p,q,r} honeycombs (S³ / E³ / H³), 3D Voronoi cells, Dirichlet domains, P / D / G nets.

Every expected value is a hand count or a classical theorem, stated next to the assertion. Each test
runs on the pure-Python fallback (`PAPERTILLING_NO_RUST=1`) and, when `papertilling_rs` is installed
with the Stage 6 functions, on the Rust core.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from papertilling import _rust
from papertilling import honeycomb3d as hc


def _rust_ready() -> bool:
    mod = _rust._load()
    return mod is not None and hasattr(mod, "honeycomb3d_build_json")


@pytest.fixture(params=["py", "rust"])
def backend(request, monkeypatch):
    if request.param == "rust":
        if not _rust_ready():
            pytest.skip("papertilling_rs with Stage 6 functions not installed")
        monkeypatch.setenv("PAPERTILLING_NO_RUST", "0")
    else:
        monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
    return request.param


COMPACT = [(3, 5, 3), (4, 3, 5), (5, 3, 4), (5, 3, 5)]
PARACOMPACT = [(3, 3, 6), (4, 3, 6), (5, 3, 6), (3, 4, 4), (4, 4, 3), (4, 4, 4), (3, 6, 3), (6, 3, 3), (6, 3, 4),
               (6, 3, 5), (6, 3, 6)]


def test_classification(backend):
    for s in [(3, 3, 3), (4, 3, 3), (3, 3, 4), (3, 4, 3), (5, 3, 3), (3, 3, 5)]:
        c = hc.classify_schlafli(*s)
        assert c["regime"] == "SphericalS3" and c["det_g"] > 0
    assert hc.classify_schlafli(4, 3, 4)["regime"] == "EuclideanE3"
    assert all(hc.classify_schlafli(*s)["regime"] == "HyperbolicCompactH3" for s in COMPACT)
    # Euclidean vertex figure {3,6},{4,4},{6,3} -> ideal vertices; Euclidean cell -> infinite cells ({3,6,3} has both).
    for p, q, r in PARACOMPACT:
        c = hc.classify_schlafli(p, q, r)
        assert c["regime"] == "HyperbolicParacompactH3"
        assert c["has_ideal_vertices"] == ((q - 2) * (r - 2) == 4)
        assert c["has_infinite_euclidean_cells"] == ((p - 2) * (q - 2) == 4)
        assert c["det_g"] <= -1 / 16 + 1e-12
    assert sum(hc.classify_schlafli(*s)["has_ideal_vertices"] for s in PARACOMPACT) == 7
    assert sum(hc.classify_schlafli(*s)["has_infinite_euclidean_cells"] for s in PARACOMPACT) == 7
    h = hc.classify_schlafli(7, 3, 3)
    assert h["regime"] == "HyperidealRefused" and h["reason"].startswith("[PROVED")
    d = hc.classify_schlafli(4, 4, 2)
    assert d["regime"] == "DegenerateRefused" and "reducible" in d["reason"]
    with pytest.raises(ValueError, match="PROVED"):
        hc.build_honeycomb(3, 3, 7)


@pytest.mark.parametrize("sym,want", [
    ((3, 3, 3), (5, 10, 10, 5)), ((4, 3, 3), (16, 32, 24, 8)), ((3, 3, 4), (8, 24, 32, 16)),
    ((3, 4, 3), (24, 96, 96, 24)), ((5, 3, 3), (600, 1200, 720, 120)), ((3, 3, 5), (120, 720, 1200, 600)),
])
def test_six_polychora(backend, sym, want):
    cx = hc.build_honeycomb(*sym)
    assert hc.counts(cx) == want  # Coxeter, Regular Polytopes (3rd ed., 1973), Table I(ii)
    assert hc.euler(cx) == 0  # chi(S^3) = 0
    assert np.allclose(np.linalg.norm(cx["coords4"], axis=1), 1.0)
    assert np.all(np.isfinite(cx["points"]))


def test_h3_honeycombs(backend):
    # Shell 1 for r >= 4: V + F(V - p), E + F(E - p), F + F(F - 1), F + 1.
    assert hc.counts(hc.build_honeycomb(4, 3, 5, 0)) == (8, 12, 6, 1)
    assert hc.counts(hc.build_honeycomb(4, 3, 5, 1)) == (32, 60, 36, 7)
    assert hc.counts(hc.build_honeycomb(5, 3, 4, 1)) == (200, 330, 144, 13)
    assert len(hc.build_honeycomb(3, 5, 3, 1)["cells"]) == 21
    for s in COMPACT:
        cx = hc.build_honeycomb(*s, shells=1)
        assert hc.euler(cx) == 1  # a ball of cells
        assert np.all(np.linalg.norm(cx["points"], axis=1) < 1)
    t = hc.build_honeycomb(3, 3, 6, 0)
    assert hc.counts(t) == (4, 6, 4, 1) and all(t["is_ideal"])
    assert np.allclose(np.linalg.norm(t["points"], axis=1), 1.0)
    # {6,3} cell cut to the seed hexagon + 1 ring: 7 hexagons, 24 vertices, 30 edges (coronene).
    assert hc.counts(hc.build_honeycomb(6, 3, 3, 0, 1)) == (24, 30, 7, 1)
    for p, q, r in PARACOMPACT:
        cx = hc.build_honeycomb(p, q, r, shells=1)
        ideal = (q - 2) * (r - 2) == 4
        norms = np.linalg.norm(cx["points"], axis=1)
        assert all(b == ideal for b in cx["is_ideal"])
        assert np.allclose(norms, 1.0) if ideal else np.all(norms < 1)


def test_voronoi_and_euclidean(backend):
    for lat, want in [("SC", (8, 12, 6)), ("FCC", (14, 24, 12)), ("BCC", (24, 36, 14))]:
        c = hc.lattice_voronoi_cell(lat)
        assert (len(c["vertices"]), len(c["edges"]), len(c["faces"])) == want
    bcc = hc.lattice_voronoi_cell("BCC")
    assert sorted(len(f) for f in bcc["faces"]) == [4] * 6 + [6] * 8  # Kelvin truncated octahedron
    nn = [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    oct_ = hc.voronoi_cell_3d([0, 0, 0], nn)  # first BCC shell only: octahedron
    assert (len(oct_["vertices"]), len(oct_["faces"])) == (6, 8)
    assert hc.counts(hc.build_euclidean_honeycomb("cubic", 1)) == (32, 60, 36, 7)
    assert hc.counts(hc.build_euclidean_honeycomb("alternated_cubic", 1)) == (14, 36, 32, 9)
    assert len(hc.build_euclidean_honeycomb("bitruncated_cubic", 1)["cells"]) == 15
    assert len(hc.build_euclidean_honeycomb("rhombic_dodecahedral", 1)["cells"]) == 13


def test_dirichlet_domains(backend):
    sw = hc.dirichlet_manifold("seifert_weber")
    q = sw["quotient"]
    assert sw["domain"]["geometry"] == "H3"
    assert (q["cells"], q["faces"], q["edges"], q["vertices"], q["euler"]) == (1, 6, 6, 1, 0)
    # Weber & Seifert, Math. Z. 37 (1933): Seifert-Weber space has H_1 = (Z/5)^3; edge cycles of 5 (2π/5 dihedral).
    assert q["edge_class_sizes"] == [5] * 6 and q["h1_label"] == "Z_5^3"
    assert q["evidence"].startswith("[PROVED")
    phs = hc.dirichlet_manifold("poincare_homology_sphere")
    q = phs["quotient"]
    assert phs["domain"]["geometry"] == "S3"
    assert (q["faces"], q["edges"], q["vertices"], q["euler"], q["h1_label"]) == (6, 10, 5, 0, "0")
    for name, want in [("cubic_3_torus", (3, 3, 1)), ("bcc_3_torus", (7, 12, 6)), ("fcc_3_torus", (6, 8, 3))]:
        q = hc.dirichlet_manifold(name)["quotient"]
        assert (q["faces"], q["edges"], q["vertices"], q["h1_label"]) == (*want, "Z^3")


def test_labyrinth_nets(backend):
    for kind, n, z, bond, sg1, sg2 in [("P", 1, 6, 1.0, 221, 229), ("D", 8, 4, math.sqrt(3) / 4, 227, 224),
                                       ("G", 8, 3, math.sqrt(2) / 4, 214, 230)]:
        one = hc.tpms_net(kind, 1)
        assert len(one["positions"]) == n and set(one["coordination"]) == {z}
        assert one["bond_length"] == pytest.approx(bond) and one["space_group_number"] == sg1
        two = hc.tpms_net(kind, 2)
        assert two["space_group_number"] == sg2
        f = hc.nodal(kind, np.array(two["positions"]))
        lab = np.array(two["labyrinth_of"])
        assert np.all(f[lab == 0] > 0.5) and np.all(f[lab == 1] < -0.5)  # opposite sides of the surface


def test_periodic_surfaces_genus_three(backend):
    p = hc.petrie_coxeter_surface("P")
    assert (len(p["vertices"]), len(p["edges"]), len(p["faces"])) == (12, 24, 8)
    d = hc.petrie_coxeter_surface("D")
    assert (len(d["vertices"]), len(d["edges"]), len(d["faces"])) == (16, 48, 16)
    for s in (p, d):
        assert s["closed_manifold"] and s["genus_primitive"] == 3 and s["evidence"].startswith("[PROVED")
    for kind, chi in [("P", -4), ("D", -16), ("G", -8)]:
        m = hc.nodal_surface_mesh(kind, 12)
        assert m["closed_manifold"] and m["connected_components"] == 1
        assert (m["euler_conventional"], m["genus_primitive"]) == (chi, 3)
        assert m["evidence"].startswith("[COMPUTATIONAL EVIDENCE")


def test_rust_python_parity(monkeypatch):
    if not _rust_ready():
        pytest.skip("papertilling_rs with Stage 6 functions not installed")

    def run():
        return (
            [hc.counts(hc.build_honeycomb(*s, shells=1)) for s in [(5, 3, 3), (4, 3, 5), (3, 3, 6), (6, 3, 3)]],
            hc.counts(hc.build_euclidean_honeycomb("bitruncated_cubic", 1)),
            {k: v for k, v in hc.dirichlet_manifold("seifert_weber")["quotient"].items()},
            hc.classify_schlafli(6, 3, 6),
            hc.nodal_surface_mesh("G", 12)["euler_conventional"],
        )

    monkeypatch.setenv("PAPERTILLING_NO_RUST", "0")
    a = run()
    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
    b = run()
    assert a[:3] == b[:3] and a[4] == b[4]
    assert {k: v for k, v in a[3].items() if k != "det_g"} == {k: v for k, v in b[3].items() if k != "det_g"}
    assert a[3]["det_g"] == pytest.approx(b[3]["det_g"])
