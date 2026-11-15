"""A lattice on the flat surfaces: cylinder, torus, Moebius band, Klein bottle (papertilling/surfaces/flat.py).

Expected values are counted by hand:
  * a torus of k primitive cells carries k x (sites, bonds, faces per cell): square 1, 2, 1; honeycomb 2, 3, 1;
    kagome 3, 6, 3 (two triangles and one hexagon); so V - E + F = 0;
  * a Klein bottle is a torus divided by a free involution: half as many sites, bonds and faces as that torus;
  * every site keeps its plane coordination (4 square, 3 honeycomb, 4 kagome, 5 Shastry-Sutherland) away from an open edge;
  * a non-orientable surface needs a mirror or glide line: refused for p1, p2, p3, p4, p6 (chiral lattices);
  * rolling a sheet into a cylinder of circumference |C| puts every site at distance |C| / 2 pi from the axis.
"""
import numpy as np
import pytest

import papertilling as pt
from papertilling.surfaces import torus, cylinder, klein_bottle, mobius_band, glides


@pytest.mark.parametrize("name,per_cell", [("square", (1, 2, 1)), ("honeycomb", (2, 3, 1)), ("kagome", (3, 6, 3))])
def test_torus_counts(name, per_cell):
    S = torus(pt.named(name), (4, 0), (0, 3))
    c = S.checks(show=False)
    assert (c["V"], c["E"], c["F"]) == tuple(12 * x for x in per_cell) and c["ok"]


def test_skew_torus_and_its_cell_count():
    S = torus(pt.named("triangular"), (3, 1), (-1, 2))               # |det| = 7 cells
    c = S.checks(show=False)
    assert (c["V"], c["E"], c["F"]) == (7, 21, 14) and c["ok"]


@pytest.mark.parametrize("name", ["square", "kagome", "honeycomb", "shastry_sutherland", "zigzag", "brick_pg"])
def test_klein_bottle_is_half_a_torus(name):
    L = pt.named(name)
    K = klein_bottle(L)
    c = K.checks(show=False)
    assert c["ok"] and not K.orientable and K.chi == 0
    g = glides(L)[0]
    T = torus(L, tuple(K.params["along"] * g["tau"]), tuple(K.params["across"] * g["across"]))
    t = T.checks(show=False)
    assert (2 * c["V"], 2 * c["E"], 2 * c["F"]) == (t["V"], t["E"], t["F"])


@pytest.mark.parametrize("name", ["square", "kagome", "honeycomb", "shastry_sutherland"])
def test_open_surfaces(name):
    for S in (cylinder(pt.named(name)), mobius_band(pt.named(name))):
        c = S.checks(show=False)
        assert c["ok"] and c["V_minus_E_plus_F"] == 0 and not S.closed and S.edge_sites


def test_chiral_lattices_refused():
    for name in ("chiral_square", "chiral_kagome", "chiral_triangular", "oblique"):
        L = pt.named(name)
        assert glides(L) == []
        with pytest.raises(ValueError):
            klein_bottle(L)
        with pytest.raises(ValueError):
            mobius_band(L)


def test_cylinder_is_isometric():
    L = pt.named("square")
    S = cylinder(L, (6, 0), 5)
    r = np.hypot(S.xyz[:, 0], S.xyz[:, 1])
    assert np.allclose(r, 6 / (2 * np.pi))
    for a, b, v in S.bonds:                                        # every drawn bond is a plane bond (length 1)
        assert abs(np.linalg.norm(v) - 1) < 1e-9


def test_negative_control_too_small_klein_bottle_is_caught():
    """A Klein bottle one glide step long and one translation wide: a site is glued to its own neighbour, so a bond becomes
    a loop. The checks must say so (no silent wrong answer)."""
    c = klein_bottle(pt.named("square"), along=1, across=1).checks(show=False)
    assert not c["ok"]


def test_bonds_are_lattice_bonds_everywhere():
    L = pt.named("kagome")
    lengths = sorted({round(float(np.linalg.norm(v)), 6) for S in (torus(L), klein_bottle(L), mobius_band(L), cylinder(L))
                      for _, _, v in S.bonds})
    cell = L.cell()
    plane = sorted({round(float(np.linalg.norm(cell.basis[j] + cell.A @ np.array([n1, n2]) - cell.basis[i])), 6)
                    for i, j, n1, n2, *_ in cell.bonds})
    assert lengths == plane
