"""The points-only rule (papertilling/points.py; author, 2026-11-14): points with exactly the symmetry of G,
bonds = Voronoi neighbours.

Expected values are hand-known facts, not outputs of the code:
  * Voronoi neighbours of Bravais lattices: square 4 (the diagonals are not bonds: the four corners of a square
    lie on one circle), rectangular 4, hexagonal 6, a generic oblique lattice 6 (the two cell edges and the
    SHORTER diagonal), honeycomb points 3;
  * the uniform (Archimedean) tilings with symmetry p6m and their sites per primitive cell: 3^6 (1), 6^3 (2),
    3.6.3.6 (3), 3.4.6.4 (6), 3.12^2 (6), 4.6.12 (12); with p4g: the snub square tiling 3^2.4.3.4, which is the
    Shastry-Sutherland network (cited result: Gruenbaum & Shephard, Tilings and Patterns, ch. 2, from memory);
  * one or two points per primitive cell, with nothing but translations, always have a 2-fold rotation (about a
    point, or about the midpoint of the pair), so a lattice of p1 needs at least 3 sites per cell (hand proof);
  * the symmetry of the triangular network is p6m (12 operations per cell): drawn as an oblique Bravais lattice
    (p2, 2 operations) its symmetry index is 6, as a centred rectangular one (cmm, 4 operations) 3.
"""
import math
import warnings

import numpy as np
import pytest
from scipy.spatial import Voronoi

from papertilling.dsymbol import dsymbol, network_key
from papertilling.lattice import load_catalogue
from papertilling.legacy.wallpaper_lattices import make_cell, _canon_bond
from papertilling.points import voronoi_bonds, is_voronoi, points_lattices

SQUARE_NET, TRI_NET, HONEY_NET = ((4, 4, 0, 0, 0),), ((3, 6, 0, 0, 0),), ((6, 3, 0, 0, 0),)
own_net = lambda name: network_key(dsymbol(make_cell(name))["key"])


def cell(b, gamma):
    g = math.radians(gamma)
    return np.array([[1.0, b * math.cos(g)], [0.0, b * math.sin(g)]])


def test_voronoi_bonds_of_bravais_lattices():
    one = [[0.0, 0.0]]
    c = lambda *ns: sorted(_canon_bond(0, 0, n1, n2, 0)[:4] for n1, n2 in ns)                 # a bond and its reverse are one bond
    assert voronoi_bonds(one, cell(1, 90)) == c((1, 0), (0, 1))                               # square: no diagonal
    assert voronoi_bonds(one, cell(1.4, 90)) == c((1, 0), (0, 1))                             # rectangular
    assert len(voronoi_bonds(one, cell(1, 60))) == 3                                          # hexagonal: 6 neighbours
    got = voronoi_bonds(one, cell(1.2, 75))                                                   # oblique, gamma < 90:
    assert got == c((1, 0), (0, 1), (1, -1))                                                  # a1 - a2 is the short diagonal
    got = voronoi_bonds(one, cell(1.2, 105))                                                  # gamma > 90: a1 + a2 is shorter
    assert got == c((1, 0), (0, 1), (1, 1))
    honey = voronoi_bonds([[1 / 3, 2 / 3], [2 / 3, 1 / 3]], cell(1, 120))
    assert len(honey) == 3 and all(i != j for (i, j, _, _) in honey)                          # 3 bonds, all between the two sites


def test_voronoi_bonds_second_route_random_points():
    """Independent route: scipy's Voronoi diagram of the tiled points; two sites are neighbours when their cells
    share a ridge of non-zero length."""
    rng = np.random.default_rng(5)
    for m in (2, 3, 5):
        F = rng.uniform(0, 1, (m, 2)); A = cell(1.15, 72)
        R = 3
        sh = [(a, b) for a in range(-R, R + 1) for b in range(-R, R + 1)]
        P = np.concatenate([(F + np.array(s, float)) @ A.T for s in sh])
        lab = [(i, s) for s in sh for i in range(m)]
        vor = Voronoi(P)
        want = set()
        for (p, q), verts in zip(vor.ridge_points, vor.ridge_vertices):
            if -1 in verts or np.hypot(*(vor.vertices[verts[0]] - vor.vertices[verts[1]])) < 1e-9:
                continue
            (i, s), (j, t) = lab[p], lab[q]
            if s == (0, 0) or t == (0, 0):
                want.add(_canon_bond(i, j, t[0] - s[0], t[1] - s[1], 0)[:4])
        assert set(voronoi_bonds(F, A)) == want


@pytest.fixture(scope="module")
def cat():
    return load_catalogue()                               # the points-only lists (default)


def test_bravais_lattices_come_first(cat):
    for g, vt, net, index in [("p4m", "4.4.4.4", SQUARE_NET, 1), ("pmm", "4.4.4.4", SQUARE_NET, 2),
                              ("p6m", "3.3.3.3.3.3", TRI_NET, 1), ("p2", "3.3.3.3.3.3", TRI_NET, 6), ("cmm", "3.3.3.3.3.3", TRI_NET, 3)]:
        L = cat[g].lattices[0]
        assert L.family.m == 1 and L.vertex_types() == [vt] and L.network == net and L.symmetry_index == index, g


def test_p6m_single_orbit_lattices_are_the_uniform_tilings(cat):
    Ls = [L for L in cat["p6m"].lattices if L.rank[0] == 1]
    assert [(L.family.m, L.vertex_types()) for L in Ls[:3]] == [(1, ["3.3.3.3.3.3"]), (2, ["6.6.6"]), (3, ["3.3.6.6"])]
    assert sorted((L.family.m, L.vertex_types()[0]) for L in Ls[3:]) == [(6, "3.12.12"), (6, "3.4.4.6"), (12, "4.6.12")]
    assert all(L.symmetry_index == 1 for L in Ls)                       # p6m is each network's own symmetry
    assert [L.network for L in Ls[:3]] == [TRI_NET, HONEY_NET, own_net("kagome")]


def test_p4g_first_is_shastry_sutherland(cat):
    L = cat["p4g"].lattices[0]
    assert L.family.m == 4 and L.vertex_types() == ["3.3.3.4.4"] and L.symmetry_index == 1
    assert L.network == own_net("shastry_sutherland")


def test_p6_first_is_the_snub_hexagonal_tiling(cat):
    """3^4.6 is the uniform tiling whose symmetry is p6 (chiral); its 6 sites per cell are the general position."""
    L = cat["p6"].lattices[0]
    assert L.family.m == 6 and L.vertex_types() == ["3.3.3.3.6"] and L.symmetry_index == 1


def test_groups_without_a_single_orbit_lattice(cat):
    assert cat["p1"].lattices[0].rank[0] == 3 and cat["p1"].lattices[0].family.m == 3      # hand proof in the docstring
    for g in ("pm", "pg", "cm"):
        assert cat[g].lattices[0].rank[0] >= 2, g
    for g in ("p2", "pmm", "pmg", "pgg", "cmm", "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m"):
        assert cat[g].lattices[0].rank[0] == 1, g
    assert cat["p4"].lattices[0].family.m == 4            # negative control: 1a, 1b, 2c of p4 are square lattices (p4m)


def test_every_listed_lattice(cat):
    assert set(cat) == {"p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm", "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m"}
    for g, res in cat.items():
        nets = [L.network for L in res.all]
        assert len(res.lattices) == 7 and len(set(nets)) == len(nets), g
        for L in res.lattices:
            assert L.meta["routes_agree"] and is_voronoi(L.family), (g, L.rank)
            assert min(len(v.split(".")) for v in L.vertex_types()) >= 3          # every site has >= 3 bonds


def test_negative_control_voronoi_lost_when_a_parameter_moves(cat):
    """The 3.4.6.4 and 3.12.12 entries of p6m are the same points (6e) at different parameter values: at the
    other entry's value the first entry's bonds are no longer the Voronoi ones."""
    a, b = [L for L in cat["p6m"].lattices if L.vertex_types()[0] in ("3.4.4.6", "3.12.12")]
    assert a.meta["wyckoff_search"] == b.meta["wyckoff_search"]
    assert is_voronoi(a.family) and not is_voronoi(a.family, b.family.values)


@pytest.mark.slow
def test_live_search_p6m_equals_catalogue(cat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = points_lattices("p6m")
    assert [L.key for L in res.lattices] == [L.key for L in cat["p6m"].lattices]


def test_take_fewer_and_refuse_more(cat):
    r = cat["p6m"].take(3)
    assert [L.vertex_types() for L in r.lattices] == [["3.3.3.3.3.3"], ["6.6.6"], ["3.3.6.6"]] and r.n == 3
    assert len(cat["p6m"].lattices) == 7                      # the stored result is not changed
    with pytest.raises(ValueError):
        cat["p6m"].take(8)


@pytest.mark.slow
def test_longer_list_extends_the_stored_one(cat):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = points_lattices("p6m", n=10)
    assert len(res.lattices) == 10 and len({L.network for L in res.lattices}) == 10
    assert [L.key for L in res.lattices[:7]] == [L.key for L in cat["p6m"].lattices]     # the first 7 are the same
    assert all(L.meta["routes_agree"] and is_voronoi(L.family) for L in res.lattices)
