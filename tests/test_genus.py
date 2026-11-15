"""A lattice on a closed surface of genus g >= 2: a double cover of a flat torus branched at 2(g - 1) face centres
(papertilling/surfaces/genus.py).

Expected values are counted by hand or come from published formulas:
  * a torus of k primitive cells carries k x (sites, bonds, faces per cell): square 1, 2, 1; honeycomb 2, 3, 1;
    kagome 3, 6, 3 (two triangles, one hexagon). The 6 x 6 torus (k = 36): square V0 36, E0 72, F0 36;
    honeycomb 72, 108, 36; kagome 108, 216, 108.
  * Riemann-Hurwitz (cited result) for a double cover of the torus (chi 0) branched at b points: chi = 2 x 0 - b, so b = 2g - 2.
    Two sheets double the sites and bonds: V = 2 V0, E = 2 E0. A face holding a branch point lifts to ONE face with twice
    the sides, every other face to two faces: F = 2 F0 - b.
      square g = 2:     V 72,  E 144, F 72 - 2 = 70  (two octagons instead of four squares),  V - E + F = -2
      honeycomb g = 2:  V 144, E 216, F 72 - 2 = 70  (two 12-gons),                            V - E + F = -2
      kagome g = 2:     V 216, E 432, F 216 - 2 = 214,                                         V - E + F = -2
      genus 3 (b = 4):  F = 2 F0 - 4, V - E + F = -4 (square 72, 144, 68; honeycomb 144, 216, 68; kagome 216, 432, 212).
  * Gauss-Bonnet: each branch point has 720 deg around it, a Frank angle of 360 - 720 = -360 deg; b of them add up to
    -360 b = 360 chi.
  * A horizontal slit from the centre of square (1.5, 2.5) to that of square (4.5, 2.5) crosses the three vertical bonds
    at x = 2, 3, 4: 3 torus bonds, each lifting to 2 bonds that change sheet, so 6 sheet-changing bonds.
  * Negative controls: a slit from (0.5, 0.5) to (2.5, 2.5) on the square lattice passes through the sites (1, 1) and (2, 2)
    and is refused; two branch points in one face are refused; g = 1 returns the flat torus; g = 0 is refused.
"""
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

import papertilling as pt
from papertilling.surfaces.genus import GenusSurface, genus_surface


@pytest.mark.parametrize("name,counts,sides", [("square", (72, 144, 70), 4), ("honeycomb", (144, 216, 70), 6),
                                               ("kagome", (216, 432, 214), None)])
def test_genus_two_counts(name, counts, sides):
    S = genus_surface(pt.named(name), g=2, torus_size=(6, 6))
    c = S.checks(show=False)
    assert (c["V"], c["E"], c["F"]) == counts and c["V_minus_E_plus_F"] == -2 and S.chi == -2 and c["ok"]
    assert S.orientable and S.closed and S.xyz is None and S.xyz_note
    assert len(S.cones) == 2 and all(x["angle"] == -360 and x["location"] == "face centre" for x in S.cones)
    assert c["frank_sum"] == -720
    for x in S.cones:                                              # each branch face: twice the sides of its plane face
        assert len(S.faces[x["face"]]) == 2 * x["plane_sides"]
        if sides:
            assert x["plane_sides"] == sides


@pytest.mark.parametrize("name,counts", [("square", (72, 144, 68)), ("honeycomb", (144, 216, 68)), ("kagome", (216, 432, 212))])
def test_genus_three_counts(name, counts):
    S = genus_surface(pt.named(name), g=3)
    c = S.checks(show=False)
    assert (c["V"], c["E"], c["F"]) == counts and c["V_minus_E_plus_F"] == -4 and c["ok"]
    assert len(S.cones) == 4 and c["frank_sum"] == -1440


def test_straight_p1_by_riemann_hurwitz():
    """p1: no symmetry but translations; the -360 deg branch points are still allowed (rotation by 360 = identity)."""
    S = genus_surface(pt.named("straight_p1"), g=2)
    assert S.lattice.group == "p1"
    c, t = S.checks(show=False), S.torus.checks(show=False)
    assert (c["V"], c["E"], c["F"]) == (2 * t["V"], 2 * t["E"], 2 * t["F"] - 2) and c["ok"]


def test_sheet_changing_bonds_by_hand():
    S = genus_surface(pt.named("square"), g=2, branch=[((1.5, 2.5), (4.5, 2.5))])
    assert S.checks(show=False)["ok"]
    changing = [(a, b) for a, b, _ in S.bonds if S.sheet[a] != S.sheet[b]]
    assert len(changing) == 6
    for a, b, v in S.bonds:                                        # the sheet-changing bonds are the vertical ones at x = 2, 3, 4
        if S.sheet[a] != S.sheet[b]:
            assert abs(v[0]) < 1e-9 and round(S.plane_uv[a][0]) in (2, 3, 4)


def test_slit_across_the_torus_edge():
    """On the 6 x 6 square torus the shortest slit from square (5.5, 2.5) to square (1.5, 2.5) runs through x = 6 = 0 and
    x = 7 = 1: it crosses 2 vertical bonds, so 4 bonds change sheet. A skew torus (6, 2), (-2, 5) of the triangular lattice
    has |det| = 34 cells, 1 site, 3 bonds, 2 triangles each: genus 2 gives V 68, E 204, F 136 - 2 = 134."""
    S = genus_surface(pt.named("square"), g=2, branch=[((5.5, 2.5), (1.5, 2.5))])
    assert S.checks(show=False)["ok"] and sum(S.sheet[a] != S.sheet[b] for a, b, _ in S.bonds) == 4
    c = genus_surface(pt.named("triangular"), g=2, torus_size=((6, 2), (-2, 5))).checks(show=False)
    assert (c["V"], c["E"], c["F"]) == (68, 204, 134) and c["ok"]


def test_refusals():
    sq = pt.named("square")
    with pytest.raises(ValueError, match="site"):
        genus_surface(sq, g=2, branch=[((0.5, 0.5), (2.5, 2.5))])
    with pytest.raises(ValueError, match="same face"):
        genus_surface(sq, g=2, branch=[((0.5, 0.5), (0.6, 0.6))])
    with pytest.raises(ValueError):
        genus_surface(sq, g=0)
    T = genus_surface(sq, g=1)
    assert not isinstance(T, GenusSurface) and T.kind == "torus" and T.chi == 0 and not T.cones


def test_checks_catch_a_broken_surface():
    S = genus_surface(pt.named("honeycomb"), g=2)
    S.bonds = S.bonds[1:]
    c = S.checks(show=False)
    assert not c["ok"] and not c["two_faces_per_bond"]


def test_plot_net_and_no_3d():
    import matplotlib.pyplot as plt
    S = genus_surface(pt.named("kagome"), g=2, torus_size=(5, 5))
    ax = S.plot_net()
    assert ax is not None
    with pytest.raises(ValueError):
        S.plot()
    assert S.show() is not None                                    # Surface.show calls plot_net(rotate=True)
    plt.close("all")


@pytest.mark.slow
def test_every_named_lattice_genus_two():
    from papertilling.sheet import named_lattices
    for row in named_lattices(show=False):
        S = genus_surface(pt.named(row["key"]), g=2)
        c, t = S.checks(show=False), S.torus.checks(show=False)
        assert c["ok"], row["key"]
        assert (c["V"], c["E"], c["F"]) == (2 * t["V"], 2 * t["E"], 2 * t["F"] - 2), row["key"]
