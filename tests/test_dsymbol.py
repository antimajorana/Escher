"""D-symbol controls with expected values derived independently of the code.

Size: |D| = 4 E / |P| (E bonds per primitive cell, |P| point-group order): square 4*2/8 = 1,
triangular 4*3/12 = 1, honeycomb 4*3/12 = 1, kagome 4*6/12 = 2, Shastry-Sutherland 4*10/8 = 5.
Labels: square (m01, m12) = (4, 4); triangular (3, 6); honeycomb (6, 3). [hand]
Curvature sum_D (1/m01 + 1/m12 - 1/2) = 0 for every planar periodic tiling. [formula]
"""
import time
from fractions import Fraction

import numpy as np
import pytest

from papertilling.dsymbol import dsymbol, DSymbolError
from papertilling.legacy.wallpaper_lattices import make_cell, primitive_cell, UnitCell
from papertilling.legacy.lattice_families import Family


def ds(name, **kw):
    return dsymbol(primitive_cell(make_cell(name, **kw), use_types=False))


@pytest.mark.parametrize("name,size,labels", [
    ("square", 1, {(4, 4)}), ("triangular", 1, {(3, 6)}), ("honeycomb", 1, {(6, 3)}),
    ("kagome", 2, {(3, 4), (6, 4)}), ("shastry_sutherland", 5, None)])
def test_known_sizes(name, size, labels):
    d = ds(name)
    assert d["size"] == size == d["checks"]["size_formula"]
    assert d["checks"]["curvature"] == 0 and d["checks"]["euler"] == 0
    if labels is not None:
        assert set(zip(d["m01"], d["m12"])) == labels


def test_distinct_tilings_differ():
    keys = {n: ds(n)["key"] for n in ("square", "triangular", "honeycomb", "kagome", "shastry_sutherland", "chiral_square")}
    assert len(set(keys.values())) == len(keys)


def test_same_family_two_parameter_points():
    f = Family.from_cell("chiral_square")
    k1 = dsymbol(f.cell(), f._ops)["key"]
    k2 = dsymbol(f.cell(x1=f.values["x1"] + 0.02, y1=f.values["y1"] - 0.01), f._ops)["key"]
    assert k1 == k2


def test_brick_vs_honeycomb_net_same_tiling_different_group():
    # brick wall = honeycomb network (same tiling) but group cmm, not p6m: rule B separates them
    assert ds("brick")["key"] != ds("honeycomb")["key"]


def test_subgroup_action_differs():
    # square lattice with the full group p4m versus the same lattice seen through the subgroup p4
    c = primitive_cell(make_cell("square"), use_types=False)
    full = dsymbol(c)
    from papertilling.legacy.wallpaper_lattices import _symmetry_ops
    rot = [(M, t) for M, t in _symmetry_ops(c, use_types=False) if round(np.linalg.det(M)) == 1]
    sub = dsymbol(c, rot)
    assert sub["size"] == 2 * full["size"] and sub["key"] != full["key"]


def test_negative_crossing_bonds_refused():
    c = primitive_cell(make_cell("square"), use_types=False)
    crossed = UnitCell("x", "x", "?", c.a1, c.a2, c.basis, list(c.bonds) + [(0, 0, 1, 1, 0), (0, 0, 1, -1, 0)])
    with pytest.raises(DSymbolError):
        dsymbol(crossed)


def test_timing():
    t = time.perf_counter()
    for n in ("square", "kagome", "shastry_sutherland", "chiral_kagome"):
        ds(n)
    assert time.perf_counter() - t < 2.0
