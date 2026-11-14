"""Controls for papertilling.enumerate, kernels, kernels_dsymbol and lattice.

Expected values come from outside the search (derivations in the comments):
* square lattice (p4m): one site per cell at 1a (site symmetry 4mm), bonds to the four nearest neighbours
  = one bond orbit, 2 bonds per primitive cell, rank (1, 1, 1, 2). Key: dsymbol.py applied to the
  catalogue cell make_cell('square') with its own symmetry (independent of the search code).
* p6m tier (1, 1): triangular = 1a with 6 neighbours (3 bonds/cell); honeycomb = 2b (2 sites, 3 bonds/cell);
  kagome = 3c (3 sites, 6 bonds/cell). Keys: dsymbol.py on the catalogue cells.
* p1, one site orbit: one site p per cell; the point reflection x -> 2p - x maps the bond p -> p + n onto
  p -> p - n, the same bond translated by -n, so every 1-site lattice contains p2 (hand proof, as in the
  old notebook 'Minimal lattices.ipynb'). Planarity on the torus gives bonds <= 3 sites = 3, so tiers
  (1, 1..3) hold every 1-site-orbit lattice of p1: they must be empty.
* rule B: legacy enumerate_families('pmg', 2, network='zigzag', max_bond_orbits=3) returns two families
  with the zigzag network; their keys must differ.
* negative control: the square lattice in p4 has 8 operations mod translations (p4m), not 4.
"""
import itertools
from collections import Counter
import json
import os
import time

import numpy as np
import pytest

from papertilling import kernels as KN
from papertilling import kernels_dsymbol as KD
from papertilling import enumerate as EN
from papertilling.dsymbol import dsymbol
from papertilling.lattice import (Lattice, LatticeWarning, family_key, ops_in_cell, minimal_lattices, load_catalogue,
                                  DATA)
from papertilling.legacy.lattice_families import Family, enumerate_families
from papertilling.legacy.wallpaper_lattices import make_cell, primitive_cell, periodic_connected, WallpaperGroup
from papertilling.legacy import minimal_lattices as ML


def own_key(name):
    return dsymbol(primitive_cell(make_cell(name), use_types=False))["key"]


def config_data(group, names, samples=24, seed=3):
    G = WallpaperGroup(group)
    wy = ML.wyckoff_positions(G)
    idx = tuple(sorted(next(k for k, w in enumerate(wy) if w["name"] == nm) for nm in names))
    return EN.ConfigData(G, wy, idx, 1, samples, np.random.default_rng(seed))


# ----------------------------------------------------------------------------------------------
# kernels: numba == Python reference
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("group,names", [("p4m", ["4 m line (1/2, t)"]), ("p2", ["2 1 (x, y)"]),
                                         ("p6m", ["3 2mm (0, 1/2)", "2 3m (1/3, 2/3)"])])
def test_kernels_match_reference(group, names):
    if not KN.HAVE_NUMBA:
        pytest.skip("numba missing")
    cd = config_data(group, names)
    s_nb, p_nb = KN.sample_masks(cd.Fs, cd.orbits, cd.valid, use_numba=True)
    s_py, p_py = KN.sample_masks(cd.Fs, cd.orbits, cd.valid, use_numba=False)
    assert np.array_equal(s_nb, s_py) and np.array_equal(p_nb, p_py)
    for n in (1, 2, 3):
        for first in range(len(cd.orbits) - n + 1):
            a, ma = KN.subsets(n, cd.orbits, cd.degv, p_nb, s_nb, 3, 3 * cd.m, first, use_numba=True)
            b, mb = KN.subsets(n, cd.orbits, cd.degv, p_nb, s_nb, 3, 3 * cd.m, first, use_numba=False)
            assert a == b and [list(x) for x in ma] == [list(x) for x in mb]


def test_connectivity_kernel():
    rng = np.random.default_rng(1)
    for _ in range(300):
        m = int(rng.integers(1, 5))
        bonds = [(int(rng.integers(m)), int(rng.integers(m)), int(rng.integers(-2, 3)), int(rng.integers(-2, 3)))
                 for _ in range(int(rng.integers(1, 7)))]
        bonds = [b for b in bonds if b[0] != b[1] or b[2:] != (0, 0)]
        if not bonds:
            continue
        ref = periodic_connected(m, bonds)              # legacy wallpaper_lattices
        assert KN.periodic_connected(m, bonds, use_numba=False) == ref
        if KN.HAVE_NUMBA:
            assert KN.periodic_connected(m, bonds, use_numba=True) == ref
    # known answers: chain along a1 is not connected in 2D, square net is
    assert not KN.periodic_connected(1, [(0, 0, 1, 0)])
    assert KN.periodic_connected(1, [(0, 0, 1, 0), (0, 0, 0, 1)])
    assert not KN.periodic_connected(1, [(0, 0, 1, 1), (0, 0, 1, -1)])   # index-2 sublattice


@pytest.mark.parametrize("group,names", [("p4g", ["4 m line (t+3/4, t+1/4)"]), ("p2", ["2 1 (x, y)"]),
                                         ("p31m", ["1 3m (0, 0)", "3 m line (t+1/2, t+1/2)"])])
def test_dsymbol_kernel_equals_dsymbol_py(group, names):
    if not KN.HAVE_NUMBA:
        pytest.skip("numba missing")
    cd = config_data(group, names, samples=64)
    pair, selfm = cd.masks(2)
    checked = 0
    for n in (1, 2, 3):
        for first in range(len(cd.orbits) - n + 1):
            subs, masks = KN.subsets(n, cd.orbits, cd.degv, pair, selfm, 3, 3 * cd.m, first)
            for sub, mask in list(zip(subs, masks))[:15]:
                bi, bj, bn, ptr = cd.flat
                P, Sh, M, fl = cd.action
                karr, nk, which, st = KD.keys_of_structure(np.asarray(sub, np.int64), mask, cd.Fs, bi, bj, bn, ptr,
                                                           cd.m, P, Sh, M, fl)
                kn = {KD.key_to_tuple(karr[q]): sorted(np.nonzero(which == q)[0].tolist()) for q in range(nk)}
                bonds = [b for a in sub for b in cd.orbits[a]]
                kp = {k: sorted(v) for k, v in EN._keys_python(cd, sub, mask, bonds, Counter(), []).items()}
                assert st == 0 and kn == kp
                checked += 1
    assert checked > 0


# ----------------------------------------------------------------------------------------------
# exactness: operation count == legacy is_exact
# ----------------------------------------------------------------------------------------------
def test_exact_count_vs_legacy_is_exact():
    rng = np.random.default_rng(5)
    agree = 0
    for group, names in [("p4m", ["4 m line (1/2, t)"]), ("p2", ["1 2 (0, 0)"]), ("pmg", ["2 m line (3/4, t)"])]:
        cd = config_data(group, names, samples=32)
        pair, selfm = cd.masks(2)
        for n in (1, 2, 3):
            for first in range(len(cd.orbits) - n + 1):
                subs, masks = KN.subsets(n, cd.orbits, cd.degv, pair, selfm, 3, 3 * cd.m, first)
                for sub, mask in list(zip(subs, masks))[:6]:
                    bonds = [b for a in sub for b in cd.orbits[a]]
                    s = KN.bits_to_list(mask)[0]
                    cnt = EN.exact_count(cd.base, bonds, cd.Fs[s], EN.random_cell_params(cd.base, rng))
                    fam = cd.base._with_orbits([cd.orbits[a] for a in sub])
                    assert (cnt == cd.nops) == fam.is_exact(n=2, seed=int(rng.integers(1000)))
                    agree += 1
    assert agree >= 10


def test_negative_square_offered_to_p4():
    fam = Family.from_cell("square", group="p4")
    F = np.array([np.linalg.solve(fam.cell().A, p) for p in fam.cell().basis])
    assert EN.exact_count(fam, fam.bonds(), F, {}) == 8 != len(fam._ops) == 4
    # and the p4 search never returns the square tiling under p4's operations
    r = EN.search("p4", only_tier=(1, 1))
    assert family_key(fam) not in {e.key for e in r["entries"]}


def test_ops_in_cell_handles_origin_shift():
    fam = Family.from_cell("kagome")                  # standardised origin is not the input origin
    assert np.linalg.norm(fam.frame[1]) > 0.1
    with pytest.raises((RuntimeError, ValueError)):
        dsymbol(fam.cell(), fam._ops)                 # negative control: raw primitive-origin ops
    assert family_key(fam) == own_key("kagome")


# ----------------------------------------------------------------------------------------------
# known answers of the search
# ----------------------------------------------------------------------------------------------
def test_p4m_first_is_square():
    r = EN.search("p4m", n=1)
    e = r["entries"][0]
    assert e.rank == (1, 1, 1, 2)
    assert e.cd.wyckoff == ["1 4mm (0, 0)"]
    assert e.key == own_key("square")


def test_p6m_tier_11_contains_tri_honey_kagome():
    r = EN.search("p6m", only_tier=(1, 1))
    keys = {e.key: e for e in r["entries"]}
    for name, rank in [("triangular", (1, 1, 1, 3)), ("honeycomb", (1, 1, 2, 3)), ("kagome", (1, 1, 3, 6))]:
        k = own_key(name)
        assert k in keys and keys[k].rank == rank


def test_p1_no_single_site_orbit():
    for n in (1, 2, 3):
        r = EN.search("p1", only_tier=(1, n))
        assert r["entries"] == [] and r["tiers"][0]["status"] == "complete"


@pytest.mark.parametrize("group,tier", [("pmg", (1, 2)), ("cmm", (1, 2))])
def test_search_python_route_equals_numba_route(group, tier):
    """End to end: the whole search with Python kernels and dsymbol.py gives the same keys and ranks."""
    if not KN.HAVE_NUMBA:
        pytest.skip("numba missing")
    a = EN.search(group, only_tier=tier, samples=32, use_numba=True)
    b = EN.search(group, only_tier=tier, samples=32, use_numba=False)
    assert [(e.key, e.rank) for e in a["entries"]] == [(e.key, e.rank) for e in b["entries"]]
    assert len(a["entries"]) > 0


def test_rule_b_separates_zigzag_pmg_families():
    r = enumerate_families("pmg", 2, network="zigzag", max_bond_orbits=3)
    assert len(r) == 2
    assert family_key(r[0]) != family_key(r[1])


# ----------------------------------------------------------------------------------------------
# Lattice object
# ----------------------------------------------------------------------------------------------
def test_lattice_set_warns_on_special_locus_and_crossing():
    L = Lattice.from_cell("chiral_square")            # p4, free x1, y1
    loci = L.family.supergroup_loci()
    assert loci, "chiral square should have a p4m locus"
    ex = loci[0].example
    with pytest.warns(LatticeWarning, match="special locus"):
        L.set(**ex)
    L2 = Lattice.from_recipe(L.recipe())
    assert L2.key == L.key
    with pytest.warns(LatticeWarning, match="special locus"):
        L3 = L.set(**ex)
    assert L3.key == L3.dsymbol()["key"]            # the new object carries the key at its own values
    # a crossing point found by the legacy crossing test (independent of Lattice.set)
    from papertilling.legacy.lattice_families import has_crossings
    pts = [dict(x1=x, y1=y) for x in np.linspace(-0.9, 0.9, 13) for y in np.linspace(-0.9, 0.9, 13)]
    bad = next(v for v in pts if has_crossings(L.family.cell(**v)))
    with pytest.warns(LatticeWarning, match="bonds now cross"):
        L.set(**bad)
    assert "Lattice.from_recipe" in L.code()


# ----------------------------------------------------------------------------------------------
# catalogue (built by tools/build_catalogue.py): coverage of old work, recomputed keys
# ----------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(DATA), reason="run tools/build_catalogue.py first")
def test_catalogue_coverage_of_old_work():
    """Every old planar lattice that fits the new definition and lies in a COMPLETE searched tier of its
    group must be in the catalogue with the same key (key recomputed here from the old recipe)."""
    with open(DATA, encoding="utf8") as f:
        data = json.load(f)
    old = json.load(open(os.path.join(os.path.dirname(DATA), "..", "legacy", "data", "minimal_lattices.json"), encoding="utf8"))
    cands = [(f"legacy {g} #{i}", Family.from_recipe(e["recipe"])) for g, r in old["groups"].items()
             for i, e in enumerate(r["planar"])]
    from papertilling.legacy.wallpaper_lattices import available_lattices
    cands += [(f"catalogue {n}", Family.from_cell(n)) for n in available_lattices()]
    checked = 0
    for label, fam in cands:
        st = EN_check(fam)
        if st is None:
            continue
        g = data["groups"][fam.group]
        mb = g["bounds"]["max_bond_orbits"]
        tier = [len(fam.gens), len(fam.orbits)]
        done = [[t["k"], t["n"]] for t in g["tiers"] if t["status"] == "complete"]
        from papertilling.legacy.lattice_families import in_coverage
        if tier not in done or not in_coverage(fam, mb, g["bounds"]["reach"])[0]:
            continue
        keys = {tuple(map(tuple, e["key"])) for e in g["entries"]}
        assert family_key(fam) in keys, f"coverage hole: {label} {fam.code()}"
        checked += 1
    assert checked >= 5


def EN_check(fam):
    """None unless the lattice satisfies the definition at its own values (degree >= 3, exact, planar)."""
    from papertilling.lattice import check_family
    from papertilling.legacy.lattice_families import degrees_of
    if min(degrees_of(fam.m, fam.bonds())) < 3:
        return None
    st = check_family(fam)
    if not (st["exact"] and st["planar"] and not st["bond_through_site"]):
        return None
    return st


@pytest.mark.skipif(not os.path.exists(DATA), reason="run tools/build_catalogue.py first")
def test_catalogue_loads_and_rebuilds():
    t = time.perf_counter()
    cat = load_catalogue(rule="bonds")
    assert set(cat) == {"p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm", "p4", "p4m", "p4g", "p3", "p3m1",
                        "p31m", "p6", "p6m"}
    L = cat["p4m"].lattices[0]
    assert L.key == own_key("square") and tuple(L.rank) == (1, 1, 1, 2)
    for g in ("p6m", "p4g", "p2"):
        for L in cat[g].lattices[:3]:
            assert family_key(L.family) == L.key            # stored key == recomputed key
    assert time.perf_counter() - t < 10
