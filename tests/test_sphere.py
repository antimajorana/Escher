"""A lattice on a sphere or a projective plane (papertilling/surfaces/polyhedra.py, pieces.py).

Expected values, with their sources (none is read from the code under test):
  * Goldberg polyhedra (honeycomb on the icosahedron, size (h, k), T = h^2 + h k + k^2): V = 20 T, E = 30 T, 12 pentagons
    and 10 (T - 1) hexagons (Goldberg 1937; Coxeter 1971; the fullerene counting rule). (1, 0): dodecahedron 20, 30, 12
    pentagons; (1, 1): C60 60, 90, 12 + 20; (2, 1): T = 7, chiral, 140, 210, 12 + 60.
  * Triangular lattice on the icosahedron (geodesic polyhedra, the duals of the Goldberg ones): V = 10 T + 2, E = 30 T,
    F = 20 T triangles. (1, 0): icosahedron 12, 30, 20; (1, 1): 32, 90, 60.
  * Square lattice on the cube with its corners at sites: V = 6 (h^2 + k^2) + 2, E = 12 (h^2 + k^2), F = 6 (h^2 + k^2)
    squares. (1, 0): cube 8, 12, 6; (2, 0): 26, 48, 24.
  * Square lattice, default corners (the face-centre 4-fold class): by hand at (1, 0), every face of the cube holds one
    site at its centre, joined to the 4 sites of the neighbouring faces: the octahedron graph, 6 sites, 12 bonds, 8
    triangles (one per cube corner).
  * Kagome on the icosahedron (1, 0), corners at hexagon centres: the icosidodecahedron, 30 vertices, 60 edges, 20
    triangles + 12 pentagons (classical).
  * Oblique (p2) on the pillow, corners at sites, U = h b1, W = k b2: area 2 h k cells, so 2 h k + 2 sites (each of the 4
    cones of 180 deg is a site of degree 2 and counts half in the area), E = (4 (2hk - 2) + 4 x 2) / 2 = 4 h k, F = 2hk.
    (1, 1): 4, 4, 2; (2, 2): 10, 16, 8.
  * Projective plane = sphere / antipodal map: half of everything, chi = 1, non-orientable. Honeycomb (1, 0): the
    hemi-dodecahedron = Petersen graph, 10 vertices, 15 edges, 6 pentagons; (1, 1): 30 sites, 45 bonds, 6 pentagons +
    10 hexagons. Square lattice (1, 0) with corners at sites: the hemi-cube = K4, 4, 6, 3 squares.
  * Hemi-pillow of the rectangular (pmm) lattice, corners at sites, size (6, 6) = rectangle of 3 x 3 cells with opposite
    sides glued by glides: counted by hand, 4 interior sites + 4 pairs of edge sites + 2 pairs of corners = 10 sites,
    12 interior bonds + 3 bottom (= top) + 3 left (= right) = 18 bonds, 9 squares; V - E + F = 1.
  * Gauss-Bonnet (proved): the Frank angles add up to 360 x chi.
  * Negative controls: no rotation (p1, pm, pg) -> no disclination allowed -> refused; the icosahedron needs 60 deg
    rotations, absent from p3 -> refused by the group test while the octahedron (120 deg) passes it; a projective plane
    needs orientation-reversing gluings, absent from chiral groups -> refused.
"""
import functools
import time

import numpy as np
import pytest

import papertilling

from papertilling.surfaces import sphere, projective_plane, solid_net, frame_control
from papertilling.surfaces.pieces import ConeRefused, pairing_test


class pt:                                    # each named lattice built once per test run (only read afterwards)
    named = staticmethod(functools.lru_cache(maxsize=None)(papertilling.named))


def counts(S):
    c = S.checks(show=False)
    return c["V"], c["E"], c["F"], c["face_sizes"], c


@pytest.mark.parametrize("hk,V,E,faces", [((1, 0), 20, 30, {5: 12}), ((1, 1), 60, 90, {5: 12, 6: 20})])
def test_goldberg_honeycomb(hk, V, E, faces):
    S = sphere(pt.named("honeycomb"), hk)
    v, e, f, fs, c = counts(S)
    assert (v, e, fs) == (V, E, faces) and c["ok"] and S.chi == 2 and S.orientable
    assert len(S.cones) == 12 and all(abs(k["angle"] - 60) < 1e-9 for k in S.cones)
    assert all(k["label"] == "1a" and k["location"] == "face centre" for k in S.cones)


def test_goldberg_chiral_2_1():
    S = sphere(pt.named("honeycomb"), (2, 1))
    v, e, f, fs, c = counts(S)
    assert (v, e, fs) == (140, 210, {5: 12, 6: 60}) and c["ok"]


def test_c60_in_under_5_seconds():
    L = pt.named("honeycomb")
    t0 = time.time()
    S = sphere(L, (1, 1))
    assert time.time() - t0 < 5.0 and len(S.sites) == 60


@pytest.mark.parametrize("hk,V,E,F", [((1, 0), 12, 30, 20), ((1, 1), 32, 90, 60)])
def test_geodesic_triangular(hk, V, E, F):
    S = sphere(pt.named("triangular"), hk)
    v, e, f, fs, c = counts(S)
    assert (v, e, f, fs) == (V, E, F, {3: F}) and c["ok"]
    deg = c["degrees"]
    assert deg == ({5: 12} if V == 12 else {5: 12, 6: V - 12})        # the 12 cone sites keep 5 of their 6 bonds


@pytest.mark.parametrize("hk,V,E,F", [((1, 0), 8, 12, 6), ((2, 0), 26, 48, 24)])
def test_square_on_cube_site_corners(hk, V, E, F):
    S = sphere(pt.named("square"), hk, centre="site")
    v, e, f, fs, c = counts(S)
    assert (v, e, f, fs) == (V, E, F, {4: F}) and c["ok"]
    assert len(S.cones) == 8 and all(abs(k["angle"] - 90) < 1e-9 and k["site"] is not None for k in S.cones)


def test_square_default_is_face_centred():
    S = sphere(pt.named("square"))
    v, e, f, fs, c = counts(S)
    assert (v, e, f, fs) == (6, 12, 8, {3: 8}) and c["ok"] and S.params["centre_location"] == "face centre"


def test_kagome_icosidodecahedron():
    v, e, f, fs, c = counts(sphere(pt.named("kagome")))
    assert (v, e, fs) == (30, 60, {3: 20, 5: 12}) and c["ok"]


@pytest.mark.parametrize("hk,V,E,F", [((1, 1), 4, 4, 2), ((2, 2), 10, 16, 8)])
def test_oblique_pillow(hk, V, E, F):
    S = sphere(pt.named("oblique"), hk)
    v, e, f, fs, c = counts(S)
    assert (v, e, f) == (V, E, F) and c["euler"] and c["gauss_bonnet"]
    assert len(S.cones) == 4 and all(abs(k["angle"] - 180) < 1e-9 for k in S.cones)
    assert len(c["low_degree"]) == 4                                   # the 4 cone sites keep 2 bonds: reported
    assert S.params["centre_location"] == "site"                       # the face-centre class gives 2-gons: skipped


@pytest.mark.parametrize("hk,V,E,faces", [((1, 0), 10, 15, {5: 6}), ((1, 1), 30, 45, {5: 6, 6: 10})])
def test_projective_honeycomb(hk, V, E, faces):
    P = projective_plane(pt.named("honeycomb"), hk)
    v, e, f, fs, c = counts(P)
    assert (v, e, fs) == (V, E, faces) and c["ok"] and P.chi == 1 and not P.orientable
    assert len(P.cones) == 6 and abs(sum(k["angle"] for k in P.cones) - 360) < 1e-9
    S = sphere(pt.named("honeycomb"), hk)
    assert 2 * v == len(S.sites) and 2 * e == len(S.bonds)          # the double cover


def test_projective_square_hemicube():
    v, e, f, fs, c = counts(projective_plane(pt.named("square"), (1, 0), centre="site"))
    assert (v, e, f, fs) == (4, 6, 3, {4: 3}) and c["ok"]


def test_hemi_pillow_rectangular():
    P = projective_plane(pt.named("rectangular"), (6, 6), centre="site")
    v, e, f, fs, c = counts(P)
    assert (v, e, f, fs) == (10, 18, 9, {4: 9}) and c["ok"] and not P.orientable
    assert len(P.cones) == 2 and all(abs(k["angle"] - 180) < 1e-9 for k in P.cones)


def test_gauss_bonnet_and_3d_shape():
    for S in (sphere(pt.named("kagome")), projective_plane(pt.named("triangular"), (1, 0))):
        c = S.checks(show=False)
        assert abs(c["frank_sum"] - 360 * S.chi) < 1e-9 and c["ok"]
        assert np.allclose(np.linalg.norm(S.xyz, axis=1), 1.0)
    X = S.cover.xyz
    n = len(S.sites)
    assert X.shape == (2 * n, 3) and np.allclose(X[:n], -X[n:])          # double cover: antipodal pairs


def test_frame_control():
    """Rotation by 360/n about every catalogue centre passes the group test on the cell (frames agree)."""
    for name in ("honeycomb", "chiral_triangular", "oblique", "shastry_sutherland"):
        assert all(ok for _, ok in frame_control(pt.named(name)))


# ----------------------------------------------------------------------------------------------------------------
# negative controls
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["straight_p1", "shastrys_ladder_pm", "brick_pg"])
def test_no_rotation_refused(name):
    with pytest.raises(ConeRefused, match="no rotation"):
        sphere(pt.named(name))
    with pytest.raises(ConeRefused, match="no rotation"):
        projective_plane(pt.named(name))


def test_p3_octahedron_passes_icosahedron_refused_by_group_test():
    L = pt.named("chiral_triangular")
    octa = pairing_test(L, *solid_net(L, "octahedron", (1, 0))[:2])
    ico = pairing_test(L, *solid_net(L, "icosahedron", (1, 0))[:2])
    assert all(r["ok"] for r in octa)
    assert not all(r["ok"] for r in ico)
    assert any("rotation by" in r["isometry"] and ("60.0" in r["isometry"] or "-60.0" in r["isometry"]) for r in ico if not r["ok"])


def test_p3_sphere_is_an_octahedron():
    S = sphere(pt.named("chiral_triangular"))
    c = S.checks(show=False)
    assert S.params["solid"] == "octahedron" and len(S.cones) == 6 and c["ok"]
    assert all(abs(k["angle"] - 120) < 1e-9 for k in S.cones)


@pytest.mark.parametrize("name", ["chiral_triangular", "chiral_square", "chiral_kagome"])
def test_projective_plane_of_chiral_lattice_refused(name):
    with pytest.raises(ConeRefused, match="not symmetries"):
        projective_plane(pt.named(name))


def test_bond_midpoint_centres_refused():
    with pytest.raises(ConeRefused, match="bond midpoint"):
        sphere(pt.named("zigzag"))                                      # pmg: every 2-fold centre is a bond midpoint


def test_explicit_bond_midpoint_centre_refused_by_the_engine():
    """Oblique (p2): a cone of 180 deg on a bond midpoint folds that bond into a loop (proved: the rotation about the
    midpoint swaps the bond's two ends)."""
    with pytest.raises(ConeRefused, match="midpoint of a bond"):
        sphere(pt.named("oblique"), (1, 1), centre="bond midpoint")


def test_explicit_face_centre_with_too_few_sides_refused():
    """Oblique (p2): a cone of 180 deg at the centre of a 4-sided face leaves it 4 x (1 - 1/2) = 2 sides (a double bond)."""
    with pytest.raises(ConeRefused, match="faces_at_least_3"):
        sphere(pt.named("oblique"), (1, 1), centre="face centre")
