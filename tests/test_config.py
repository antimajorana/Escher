"""Tests of papertilling.defects.config: total charge, core graph, equivalence of configurations.

Expected values are derived by hand in the comments (holonomy algebra with the author's conventions),
or cited (C01, C04, C05 of "Defects on wallpaper lattices"), never read off the code's output.

Conventions (author): outer CCW holonomy of a disclination alpha at c is (R_alpha, (I - R_alpha) c); of
a dislocation (I, -b) with b the K5 Burgers vector. Charge class = conjugacy class in G~+.
"""
import copy
import math
import time

import networkx as nx
import numpy as np
import pytest

from papertilling.defects.build import Configuration, build, classify_charge, lattice_data, same_charge
from papertilling.defects.config import classify, compare, equivalent_single
from papertilling.legacy.lab_defects import frame_faces


def R(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


@pytest.fixture(scope="module")
def sq():
    return lattice_data("square")


@pytest.fixture(scope="module")
def pair(sq):
    return build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)),
                 picture="raw")


def test_user_example_square(sq, pair):
    """+90 at the site c1 = (0,0) and -90 at the face centre c2 = (1/2,1/2).
    Hand derivation. H1 = (R90, (I - R90) c1) = (R90, 0); H2 = (R-90, (I - R-90) c2).
    The outer CCW loop is the product in one of the two orders (depending on the base point):
      H1 H2 = (I, R90 (I - R-90) c2) = (I, (R90 - I) c2) = (I, (-1, 0))  -> b = -t = (I - R90) c2 = (1, 0);
      H2 H1 = (I, (I - R-90) c2) = (I, (0, 1))                          -> b = (0, -1).
    The two are conjugate by R-90 in P+ (both are unit lattice vectors: the class T1 of the unit
    dislocation). Frank angle 90 - 90 = 0. Core: C05 gives a 3-bond site (+90 at a site) and a
    pentagon (-90 at a face centre); here they share the site. C01: the unit dislocation of the
    square lattice has one core, a pentagon with one 3-bond site."""
    q = pair.total_charge()
    assert q["kind"] == "dislocation"
    assert any(np.allclose(q["b"], v) for v in ((1, 0), (0, -1)))
    unit = classify_charge(sq, 0.0, (np.eye(2), -np.array([1.0, 0.0])))
    assert same_charge(sq, q, unit)
    cs = pair.cores()
    site = [c for c in cs if c["kind"] == "disclination" and c.get("site") is not None]
    face = [c for c in cs if c["kind"] == "disclination" and c.get("site") is None]
    assert len(site) == 1 and site[0]["angle"] == 90 and site[0]["coordination"] == [3]
    assert len(face) == 1 and face[0]["angle"] == -90 and face[0]["ring"] == 5 and site[0]["site"] in face[0]["sites"]
    res = equivalent_single(pair, grid=1, show=True)
    assert res["equivalent"] and any(n.startswith("Volterra") for n in res["matches"])
    single = build(sq, dict(kind="dislocation", b=(1, 0), at=(0.371, 0.413)), picture="raw")
    cmp = compare(pair, single)
    assert cmp["same_charge"] and cmp["same_core"], cmp


def test_honeycomb_57(capsys):
    """+60 and -60 at neighbouring hexagon centres c1, c2 = c1 + a1 (|a1| = sqrt3 in bond units).
    Hand: b = (I - R60)(c2 - c1) (or its R60-conjugate); |I - R60| = 2 sin 30 = 1, so |b| = |a1| = sqrt3:
    the shortest dislocation. Cores: pentagon (+60 at a hexagon centre) next to a heptagon (-60).
    C04 (proved result there): every cut-and-reglue core of the honeycomb is bipartite; a 5-ring is an
    odd cycle, so this core is not isomorphic to any Volterra core. C04 result 4 (computational
    evidence there): core K2 plus one bond between its two 2-bond sites is the 5-7 core; asserted here
    as the completion with rings (5, 7)."""
    lat = lattice_data("honeycomb")
    c1 = next(r["point"] for r in lat.centres if r["order"] == 6)
    t0 = time.time()
    conf = build(lat, dict(kind="disclination", angle=60, at=c1), dict(kind="disclination", angle=-60, at=c1 + lat.cell.a1),
                 picture="raw")
    q = conf.total_charge()
    assert q["kind"] == "dislocation" and abs(np.linalg.norm(q["b"]) - math.sqrt(3)) < 1e-9
    assert sorted(c["ring"] for c in conf.cores()) == [5, 7]
    assert not nx.is_bipartite(conf.core_graph(2))
    res = equivalent_single(conf, grid=2, show=True)
    vol = [c for c in res["candidates"] if c["volterra"]]
    assert vol and all(c["bipartite"] for c in vol)                    # C04: Volterra cores bipartite
    assert not any(n.startswith("Volterra") for n in res["matches"])
    assert any(c["name"] in res["matches"] and tuple(c["rings"]) == (5, 7) for c in res["candidates"])
    print(f"# time honeycomb 5-7 control {time.time() - t0:.1f} s")


def test_same_and_different_charges(sq):
    """Hand: the translation by (1, 2) is in G+, so +90 at (0,0) and +90 at (1,2) are conjugate: same
    charge, and the crystals are congruent: same core. Sites (0,0) and face centres (1/2,1/2) are
    different Wyckoff positions of p4mm (ITA Vol. A, plane group 11: 1a and 1b), so no element of G
    maps one to the other: different charge (also up to mirror). +180 and -180 at the 2-fold centre
    (0, 1/2): the angle is part of the class in G~+ (unreduced), and a mirror keeps the angle."""
    a = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), picture="raw")
    b = build(sq, dict(kind="disclination", angle=90, at=(1, 2)), picture="raw")
    f = build(sq, dict(kind="disclination", angle=90, at=(0.5, 0.5)), picture="raw")
    r = compare(a, b)
    assert r["same_charge"] and r["same_core"]
    r = compare(a, f)
    assert not r["same_charge"] and not r["same_charge_up_to_mirror"] and not r["same_core"]
    p = build(sq, dict(kind="disclination", angle=180, at=(0, 0.5)), picture="raw")
    m = build(sq, dict(kind="disclination", angle=-180, at=(0, 0.5)), picture="raw")
    r = compare(p, m)
    assert not r["same_charge"] and not r["same_charge_up_to_mirror"]


def test_mirror_partners_p4g():
    """Shastry-Sutherland (p4g): the two 4-fold centres (ITA Vol. A plane group 12, Wyckoff 2a: one
    G-orbit) are exchanged only by mirrors/glides, so G+ = p4 splits them into two classes (author
    convention example). +90 at one vs +90 at the other: different charge, same up to mirror."""
    ss = lattice_data("shastry_sutherland")
    four = [r for r in ss.centres if r["order"] == 4]
    assert len(four) == 2
    a = build(ss, dict(kind="disclination", angle=90, at=four[0]["point"]), picture="raw")
    b = build(ss, dict(kind="disclination", angle=90, at=four[1]["point"]), picture="raw")
    r = compare(a, b)
    assert not r["same_charge"] and r["same_charge_up_to_mirror"]


def test_classify_table(sq, pair):
    """Expected classes by hand: the +90/-90 pair, the unit dislocation b = (1,0) and the unit
    dislocation b = (0,1) elsewhere are one class (same charge T1; same core: C01, one core);
    +90 at two different sites are one class; +90 at a face centre is a third."""
    d1 = build(sq, dict(kind="dislocation", b=(1, 0), at=(0.371, 0.413)), picture="raw")
    d2 = build(sq, dict(kind="dislocation", b=(0, 1), at=(2.29, -1.37)), picture="raw")
    s1 = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), picture="raw")
    s2 = build(sq, dict(kind="disclination", angle=90, at=(3, -1)), picture="raw")
    f1 = build(sq, dict(kind="disclination", angle=90, at=(0.5, 0.5)), picture="raw")
    labels = classify([pair, d1, d2, s1, s2, f1], names=["+90 site & -90 face", "b=(1,0)", "b=(0,1)",
                                                        "+90 site", "+90 other site", "+90 face"])
    assert labels == [0, 0, 0, 1, 1, 2]


def test_negative_controls(sq, pair):
    # different lattices cannot be compared
    tri = build("triangular", dict(kind="disclination", angle=60, at=(0, 0)), picture="raw")
    with pytest.raises(ValueError):
        compare(pair, tri)
    # Finding (not a negative control): deleting one pentagon bond of this core gives the 7-ring core
    # that the oblique Volterra cuts produce ("Volterra core 2") - a genuine dislocation core.
    # Corrupted core: a diagonal across a square next to the pentagon gives two 5-bond sites; no
    # cut-and-reglue core or completion has a site with more bonds than in the perfect lattice
    # (lab_defects.dislocation_cores step 2), so nothing may match.
    d = copy.deepcopy(pair.defect)
    pent = set(next(c for c in pair.cores() if c.get("ring") == 5)["sites"])
    deg = d.degrees()
    sq4 = next(f for f in frame_faces(d) if len(f) == 4 and set(f) & pent and all(deg[x] == 4 for x in f)
               and all("g" not in d.edge[(f[k], f[(k + 1) % 4])] for k in range(4)))
    u, w, v = sq4[0], sq4[1], sq4[2]
    vec = d.edge[(u, w)]["vec"] + d.edge[(w, v)]["vec"]
    d.edge[(u, v)] = dict(vec=vec, type=0)
    d.edge[(v, u)] = dict(vec=-vec, type=0)
    d.edges = sorted({(min(a, b), max(a, b)) for (a, b) in d.edge})
    d.info.pop("_charge", None)
    bad = Configuration(d, pair.specs, sq, pair.report)
    assert max(dd for _, dd in bad.core_graph(2).nodes(data="deg")) == 5
    assert not equivalent_single(bad, grid=1)["equivalent"]
    # a far-apart dislocation dipole (b and -b, cuts pointing away from each other): total charge is
    # the vacuum (hand: (I, -b)(I, b) = I), but the core graph is not empty: not the perfect lattice
    dip = build(sq, dict(kind="dislocation", b=(1, 0), at=(-3.63, 0.41), cut=(0, 1)),
                dict(kind="dislocation", b=(-1, 0), at=(3.37, 0.41), cut=(0, -1)), picture="raw")
    assert dip.total_charge()["kind"] == "vacuum"
    res = equivalent_single(dip)
    assert not res["equivalent"] and dip.core_graph(2).number_of_nodes() > 0


def test_disclination_branch(sq):
    """Positive: a single +90 at a site is equivalent to itself (the single defect of its charge).
    Negative: +90 at (0,0) and +90 at (12,0). Hand: (R90, 0)(R90, (I - R90)(12,0)) = (R180, (12, 12)),
    a +180 disclination about c = (I - R180)^-1 (12, 12) = (6, 6), a site (ITA p4m 1a, renamed from the old "4a" when labels were unified). Its single core is
    a 2-bond site (C05); the configuration has two separate 3-bond sites: not equivalent."""
    one = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), picture="raw")
    assert equivalent_single(one)["equivalent"]
    two = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=90, at=(12, 0)),
                picture="raw")
    q = two.total_charge()
    assert q["kind"] == "disclination" and q["angle"] == 180 and q["centre_class"] == "1a"
    assert sq.same_mod_lattice(q["centre"], (6, 6))
    res = equivalent_single(two)
    assert not res["equivalent"] and nx.number_connected_components(two.core_graph(2)) == 2


# ---------------------------------------------------------------------------------------------
# Fast path == legacy path for the lab engine (expected values: the legacy functions themselves)
# ---------------------------------------------------------------------------------------------
import warnings

from papertilling.defects import fast as fastmod
from papertilling.legacy import lab_defects as ld


def _same_defect(x, y):
    """Identical lab_defects.Defect: same sites, positions, bond keys in the same order, same data."""
    assert list(x.sites) == list(y.sites)
    assert np.array_equal(x.positions, y.positions)
    assert list(x.edge) == list(y.edge)
    for k in x.edge:
        assert x.edge[k].keys() == y.edge[k].keys()
        for f in x.edge[k]:
            assert np.array_equal(np.asarray(x.edge[k][f]), np.asarray(y.edge[k][f])), (k, f)
    assert x.info.keys() == y.info.keys()
    for k in x.info:
        assert str(x.info[k]) == str(y.info[k]), k


_LATS = ["square", "honeycomb", "zigzag", "kagome", "shastry_sutherland", "chiral_triangular"]
_LATS_P = [pytest.param(n, marks=() if k < 3 else pytest.mark.slow) for k, n in enumerate(_LATS)]   # 3 in the default run


@pytest.mark.parametrize("name", _LATS_P)
def test_dislocation_fast_equals_legacy(name):
    """fast.dislocation_fast (re-implemented Volterra cut and reglue) against lab_defects.dislocation:
    identical output for b = a1 and b = a2, perpendicular, oblique and lattice-direction cuts, a
    generic core point and a lattice point (sites exactly on the cut lines: the borderline branch).
    Negative control: a b that is not a lattice translation is refused by both."""
    lat = lattice_data(name)
    core = 0.371 * lat.cell.a1 + 0.413 * lat.cell.a2
    for b, other in ((lat.cell.a1, lat.cell.a2), (lat.cell.a2, lat.cell.a1)):
        for p, cut in ((core, None), (core, -(lat.cell.a1 + lat.cell.a2)), (lat.cell.basis[0], other)):
            if cut is not None and ld.cross2(cut, b) >= 0:
                cut = -cut
            x = ld.dislocation(lat.cell, b, core=p, cut=cut, R=7.0)
            y = fastmod.dislocation_fast(lat.cell, b, core=p, cut=cut, R=7.0)
            _same_defect(x, y)
    for f in (ld.dislocation, fastmod.dislocation_fast):
        with pytest.raises(AssertionError):
            f(lat.cell, 0.5 * lat.cell.a1, core=core, R=5.0)


@pytest.mark.parametrize("name", _LATS_P)
def test_disclination_fast_equals_legacy(name):
    """fast.disclination_fast (legacy code re-bound to the fast primitives) against
    lab_defects.disclination: identical output at every rotation-centre class, largest angle."""
    lat = lattice_data(name)
    for rec in lat.centres:
        a = math.radians(rec["angles"][-1])
        _same_defect(ld.disclination(lat.cell, rec["point"], a, R=6.0),
                     fastmod.disclination_fast(lat.cell, rec["point"], a, R=6.0))


def _same_scan(name, grid):
    lat = lattice_data(name)
    b = lat.cell.a1
    R = (2 * lat.r_det + 5) * 0.5 * (lat.lmean + lat.lmax) + 0.5 * lat.scale + np.linalg.norm(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t0 = time.time()
        F1, n1 = ld.dislocation_cores(lat.cell, b, grid=grid, R=R, ball=2, max_candidates=8)
        t1 = time.time()
        F2, n2 = fastmod.dislocation_cores_fast(lat.cell, b, grid=grid, R=R, ball=2, max_candidates=8)
        t2 = time.time()
    assert n1 == n2 and len(F1) == len(F2) and F1
    for f, g in zip(F1, F2):                                   # same records, in the same order
        assert f.keys() == g.keys()
        for k in f:
            if k == "defect":
                _same_defect(f[k], g[k])
            elif k == "graph":                                 # the same core graph (same nodes and bonds)
                assert set(f[k].nodes) == set(g[k].nodes)
                assert {frozenset(e) for e in f[k].edges} == {frozenset(e) for e in g[k].edges}
                assert nx.is_isomorphic(f[k], g[k])
            elif isinstance(f[k], np.ndarray) or isinstance(g[k], np.ndarray):
                assert np.array_equal(f[k], g[k]), k
            else:
                assert f[k] == g[k], (k, f[k], g[k])
    print(f"# time dislocation_cores {name} grid {grid}: legacy {t1 - t0:.2f} s, fast {t2 - t1:.2f} s, {len(F1)} cores")


def test_dislocation_cores_fast_equals_legacy():
    """fast.dislocation_cores_fast against lab_defects.dislocation_cores (square, grid 1): the same
    list of cores (names, rings, counts, hashes, builds) and the same scan notes."""
    _same_scan("square", 1)


@pytest.mark.slow
@pytest.mark.parametrize("name,grid", [("honeycomb", 2), ("shastry_sutherland", 1), ("zigzag", 1), ("kagome", 1)])
def test_dislocation_cores_fast_equals_legacy_slow(name, grid):
    _same_scan(name, grid)


@pytest.mark.slow
def test_equivalent_single_fast_equals_legacy(pair, monkeypatch):
    """equivalent_single on the fast path and on the full legacy path (PAPERTILLING_LEGACY=1, which
    also switches graph_core and the readers): the same verdict, matches, candidates, scan."""
    r1 = equivalent_single(pair, grid=1, fast=True)
    monkeypatch.setenv("PAPERTILLING_LEGACY", "1")
    r0 = equivalent_single(pair, grid=1, fast=False)
    assert r1["matches"] == r0["matches"] and r1["equivalent"] == r0["equivalent"] and r1["verdict"] == r0["verdict"]
    assert r1["scan"] == r0["scan"]
    strip = lambda c: {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
    assert [strip(c) for c in r1["candidates"]] == [strip(c) for c in r0["candidates"]]
