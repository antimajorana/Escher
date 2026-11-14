"""Tests of papertilling.groups (ITA Wyckoff tables of the 17 plane groups).

Where the expected values come from (never from the code under test):
* [ITA]  ITA Vol. A, Part 6, plane-group tables (memory; locator = plane-group number).
* [hand] a hand count written in the comment.
* [spglib] spglib 2.8 layer-group database (ITE Vol. E), through the polar layer group.
"""
import time

import numpy as np
import pytest

from papertilling import groups as GR
from papertilling.groups import (wyckoff_table, label_point, label_family, check_table, wyckoff_markdown,
                                 proper_centre_classes, spglib_layer_check, POLAR_LAYER, ITA_TABLES)
from papertilling.legacy.lattice_families import Family
from papertilling.legacy.wallpaper_lattices import WALLPAPER_GROUPS, available_lattices

spglib = pytest.importorskip("spglib")


def _lab(fam, o=0):
    d = label_family(fam)[o]
    return d["name"], d["site_symmetry"]


# ---------------------------------------------------------------------------------------------
# positive controls
# ---------------------------------------------------------------------------------------------
def test_known_positions():
    # [hand + ITA No. 11] square lattice, site at the origin: 4-fold centre on 4 mirrors -> 1a 4mm
    sq = Family("p4m", sites=[(0, 0)], bonds=[((0, 0), (1, 0))])
    assert _lab(sq) == ("1a", "4mm")
    # [hand + ITA No. 17] triangular lattice, site at the origin -> 1a 6mm
    tri = Family("p6m", sites=[(0, 0)], bonds=[((0, 0), (1, 0))])
    assert _lab(tri) == ("1a", "6mm")
    # [hand] honeycomb: 2 sites/cell on 3-fold centres with mirrors (the only 3m. position of p6mm: 2b);
    # kagome: 3 sites/cell on 2-fold centres with 2 mirrors (3c 2mm); Shastry-Sutherland: 4 sites/cell
    # on the diagonal mirrors of p4gm (4c ..m). [spglib] pilot agrees (honeycomb b 3m., kagome c 2mm).
    assert _lab(Family.from_cell("honeycomb")) == ("2b", "3m.")
    assert _lab(Family.from_cell("kagome")) == ("3c", "2mm")
    assert _lab(Family.from_cell("shastry_sutherland")) == ("4c", "..m")
    # [hand] direct points: p4m (1/2,1/2) is the other 4mm centre (1b); p6m (1/2,0) 3c;
    # p4g (0.2, 0.7) lies on the line (x, x+1/2) -> 4c; p4g (1/2, 0) lies on two mirrors but is a
    # 2-fold centre -> 2b, not the line 4c (rotation centres must win over lines)
    assert label_point("p4m", (0.5, 0.5))["letter"] == "b"
    assert label_point("p6m", (0.5, 0.0))["letter"] == "c"
    assert label_point("p4g", (0.2, 0.7))["letter"] == "c"
    assert label_point("p4g", (0.5, 0.0))["letter"] == "b"
    assert label_point("cmm", (0.75, 0.25))["name"] == "4c"
    assert label_point("p31m", (2 / 3, 1 / 3))["name"] == "2b"


# [hand] |G| per conventional cell = point-group order x centring; row counts from ITA Part 6
GEN_MULT = {"p1": 1, "p2": 2, "pm": 2, "pg": 2, "cm": 4, "pmm": 4, "pmg": 4, "pgg": 4, "cmm": 8,
            "p4": 4, "p4m": 8, "p4g": 8, "p3": 3, "p3m1": 6, "p31m": 6, "p6": 6, "p6m": 12}
N_ROWS = {"p1": 1, "p2": 5, "pm": 3, "pg": 1, "cm": 2, "pmm": 9, "pmg": 4, "pgg": 3, "cmm": 6,
          "p4": 4, "p4m": 7, "p4g": 4, "p3": 4, "p3m1": 5, "p31m": 4, "p6": 4, "p6m": 6}


def _order_from_symbol(s):
    """Site-group order read off the symbol alone: rotation order n, doubled if any mirror."""
    head = s[0]
    n = 1 if head in ".m" else int(head)
    return 2 * n if "m" in s else n


@pytest.mark.parametrize("g", WALLPAPER_GROUPS)
def test_multiplicity_sums(g):
    rows = wyckoff_table(g)
    assert len(rows) == N_ROWS[g]
    assert rows[-1]["multiplicity"] == GEN_MULT[g] == len(GR._G(g).ops)
    for r in rows:   # orbit-stabiliser, from the table alone: multiplicity x |site group| = |G|
        assert r["multiplicity"] * _order_from_symbol(r["site_symmetry"]) == GEN_MULT[g], r


def test_total_rows():
    # [hand] sum of N_ROWS = 72 Wyckoff positions over the 17 plane groups
    assert sum(len(ITA_TABLES[g]) for g in WALLPAPER_GROUPS) == 72


# ---------------------------------------------------------------------------------------------
# route 2: every table against the group operations
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("g", WALLPAPER_GROUPS)
def test_route2_table_matches_ops(g):
    assert check_table(g) == []


# ---------------------------------------------------------------------------------------------
# route 3: spglib polar layer groups
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("g", WALLPAPER_GROUPS)
def test_route3_every_row(g):
    res = spglib_layer_check(g, seed=1)
    for r in res:
        assert r["layer_number"] == POLAR_LAYER[g][0], r      # [ITE] polar layer group of g
        assert r["agree"], r
        if g not in ("p1", "p2"):     # oblique: spglib re-chooses the cell; elsewhere it keeps ITA's
            assert r["same_setting"], r
            assert r["spg_letter"] == r["letter"]


def _layer_check(cell, group):
    """Layer from a UnitCell (sites z=0, bond markers at 30%/70% of each bond, z=0.1): spglib's letter
    of every site atom = our label of its position in spglib's standard frame. Returns spglib's
    (letter, raw site symbol) of the site atoms."""
    A = cell.A
    Ai = np.linalg.inv(A)
    pos, num = [], []
    for p in cell.basis:
        f = Ai @ p
        pos.append([f[0] % 1, f[1] % 1, 0.0]); num.append(1)
    for (i, j, n1, n2, t) in cell.bonds:
        p = cell.basis[i]; q = cell.basis[j] + A @ np.array([n1, n2])
        for s in (0.3, 0.7):
            f = Ai @ (p + s * (q - p))
            pos.append([f[0] % 1, f[1] % 1, 0.1]); num.append(2)
    lat = np.array([[A[0, 0], A[1, 0], 0], [A[0, 1], A[1, 1], 0], [0, 0, 10.0]])
    d = spglib.get_layergroup((lat, np.array(pos), num), aperiodic_dir=2, symprec=1e-5)
    assert d.number == POLAR_LAYER[group][0]
    P, sh = np.array(d.transformation_matrix), np.array(d.origin_shift)
    for k in range(len(cell.basis)):
        xs = (P @ np.array(pos[k]) + sh)[:2]
        lab = label_point(group, xs)
        assert lab["letter"] == d.wyckoffs[k]
        assert lab["site_symmetry"] == GR.plane_symbol_from_layer(d.site_symmetry_symbols[k], GR._G(group).system)
    return [(d.wyckoffs[k], d.site_symmetry_symbols[k]) for k in range(len(cell.basis))]


@pytest.mark.parametrize("name", available_lattices())
def test_route3_catalogue(name):
    fam = Family.from_cell(name)
    got = _layer_check(fam.cell(), fam.group)
    # validation of the route itself on known answers [hand + ITA No. 17]: spglib alone must say
    # honeycomb -> b 3m. (2 sites), kagome -> c 2mm (3 sites)
    if name == "honeycomb":
        assert got == [("b", "3m.")] * 2
    if name == "kagome":
        assert got == [("c", "2mm")] * 3


@pytest.mark.parametrize("g", ["p2", "p31m", "p4g"])
def test_route3_random_cells(g):
    # random_symmetric_cell reaches only the general position (the per-row probes above cover the
    # special ones); cmm takes ~17 s there, so only fast groups are run (# time measured 2026-11-13)
    from papertilling.legacy.wallpaper_lattices import random_symmetric_cell
    cell = random_symmetric_cell(g, seed=1)
    assert {w for w, _ in _layer_check(cell, g)} == {wyckoff_table(g)[-1]["letter"]}


# ---------------------------------------------------------------------------------------------
# rotation centres under G+
# ---------------------------------------------------------------------------------------------
# [hand, orbit-stabiliser] a G-orbit of centres splits into two G+-classes iff its site group has no
# mirror and G has an improper element; expected class names per group:
CLASSES = {"p1": [], "pm": [], "pg": [], "cm": [],
           "p2": ["1a", "1b", "1c", "1d"], "p3": ["1a", "1b", "1c"], "p4": ["1a", "1b", "2c"],
           "p6": ["1a", "2b", "3c"], "pmm": ["1a", "1b", "1c", "1d"],
           "pmg": ["2a(i)", "2a(ii)", "2b(i)", "2b(ii)"], "pgg": ["2a(i)", "2a(ii)", "2b(i)", "2b(ii)"],
           "cmm": ["2a", "2b", "4c(i)", "4c(ii)"], "p4m": ["1a", "1b", "2c"],
           "p4g": ["2a(i)", "2a(ii)", "2b"], "p3m1": ["1a", "1b", "1c"], "p31m": ["1a", "2b(i)", "2b(ii)"],
           "p6m": ["1a", "2b", "3c"]}


@pytest.mark.parametrize("g", WALLPAPER_GROUPS)
def test_proper_centre_classes(g):
    cl = proper_centre_classes(g)
    assert [c["name"] for c in cl] == CLASSES[g]
    names = {c["name"]: c for c in cl}
    for c in cl:
        if c["g_orbit_splits"]:
            partner = c["name"][:-3] + "(ii)" if c["name"].endswith("(i)") else c["name"][:-4] + "(i)"
            assert c["mirror_partner"] == partner
        elif g in ("p2", "p3", "p4", "p6"):
            assert c["mirror_partner"] is None
        else:
            assert c["mirror_partner"] == c["name"]
    for name, pts in SPLIT_POINTS.get(g, {}).items():
        assert {tuple(np.round(q, 6)) for q in names[name]["points"]} == set(pts), name
    if g == "p4g":
        assert names["2a(i)"]["order"] == 4 and names["2b"]["order"] == 2


# [hand] split classes in one cell: G+ orbits of the centres under the proper operations
# (pmg, pgg: x,y and -x,-y only, so each centre is its own class; cmm: also the centring (1/2,1/2);
# p4g: the 4-fold rotations about the origin; p31m: the 3-fold rotations about the origin)
SPLIT_POINTS = {
    "p4g": {"2a(i)": [(0.0, 0.0)], "2a(ii)": [(0.5, 0.5)]},
    "pmg": {"2a(i)": [(0.0, 0.0)], "2a(ii)": [(0.5, 0.0)], "2b(i)": [(0.0, 0.5)], "2b(ii)": [(0.5, 0.5)]},
    "pgg": {"2a(i)": [(0.0, 0.0)], "2a(ii)": [(0.5, 0.5)], "2b(i)": [(0.0, 0.5)], "2b(ii)": [(0.5, 0.0)]},
    "cmm": {"4c(i)": [(0.25, 0.25), (0.75, 0.75)], "4c(ii)": [(0.75, 0.25), (0.25, 0.75)]},
    "p31m": {"2b(i)": [(round(1 / 3, 6), round(2 / 3, 6))], "2b(ii)": [(round(2 / 3, 6), round(1 / 3, 6))]},
}


def test_markdown():
    s = wyckoff_markdown("p6m", occupied=Family.from_cell("kagome"))
    assert "| 3c | `2mm` | point | (1/2,0) (0,1/2) (1/2,1/2) | **yes** (1) |" in s
    assert s.count("**yes**") == 1


# ---------------------------------------------------------------------------------------------
# negative controls: corrupted tables must be caught
# ---------------------------------------------------------------------------------------------
def _corrupt(g, letter, **change):
    out = []
    for row in ITA_TABLES[g]:
        row = list(row)
        if row[0] == letter:
            for k, v in change.items():
                row[{"mult": 1, "sym": 2, "coords": 3}[k]] = v
        out.append(tuple(row))
    return out


@pytest.mark.parametrize("g,table,needle", [
    ("p4m", _corrupt("p4m", "d", sym=".m."), "site symmetry"),            # wrong orientation
    ("pmm", _corrupt("pmm", "e", coords=["0,y", "0,-y"]), "site symmetry"),  # e and g swapped coordinates
    ("cmm", _corrupt("cmm", "c", mult=2), "multiplicity"),
    ("p4g", _corrupt("p4g", "c", coords=["x,x", "-x,-x", "-x,x", "x,-x"]), ""),   # p4m's line in p4g
    ("p6m", [r for r in ITA_TABLES["p6m"] if r[0] != "e"][:-1] + [("e", 12, "1", ITA_TABLES["p6m"][-1][3])],
     "special line"),                                                        # row 6e missing
])
def test_route2_catches_corruption(g, table, needle):
    err = check_table(g, table)
    assert err and any(needle in e for e in err), err


def test_route3_catches_swapped_letters(monkeypatch):
    # p4m with 1a and 1b swapped: route 2 cannot see it (same multiplicity and site symmetry) ...
    rows = list(ITA_TABLES["p4m"])
    rows[0], rows[1] = ("a",) + rows[1][1:], ("b",) + rows[0][1:]
    assert check_table("p4m", rows) == []
    # ... but the spglib route does
    monkeypatch.setitem(ITA_TABLES, "p4m", rows)
    res = spglib_layer_check("p4m", seed=1)
    assert [r["row"] for r in res if not r["agree"]] == ["1a", "1b"]


def test_runtime():
    t = time.time()
    for g in WALLPAPER_GROUPS:
        check_table(g); spglib_layer_check(g)
    dt = time.time() - t
    print(f"# time route 2 + route 3, 17 groups: {dt:.2f} s")
    assert dt < 10
