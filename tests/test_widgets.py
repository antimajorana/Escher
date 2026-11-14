"""Tests of papertilling.tuning (allowed slider intervals, special values) and of the sliders of widgets.explore.

Expected values come from outside the code under test:
  * hand-known fact: p4g lattice #1 (rotating squares, one parameter t1) is the square lattice, group p4m, at
    t1 = -1/2 (also printed by Lattice.special(); the test re-checks the group with the slow check_family);
  * the slow trusted check lattice.check_family (legacy group route + spglib, legacy crossing test, legacy
    clearance, D-symbol key) at points inside / outside the intervals found by the vectorised scan.
Probe offset outside a finite end: two slider steps (2 x 0.002), fixed before running.
"""
import time
import warnings

import numpy as np
import pytest

from papertilling.lattice import load_catalogue, check_family, Lattice
from papertilling import kernels as KN
from papertilling.tuning import allowed_interval, special_points, classify

STEP = 0.002
PROBE = 2 * STEP


@pytest.fixture(scope="module")
def L4():
    return load_catalogue(group="p4g", rule="bonds").lattices[0]


@pytest.fixture(scope="module")
def L1():
    return load_catalogue(group="p1", rule="bonds").lattices[0]


def _good(fam, key, **v):
    st = check_family(fam, key=key, **v)
    return (not st["sites_coincide"]) and st["planar"] and not st["bond_through_site"] and st["same_key"] is True


# ---------------------------------------------------------------------------------------------- known answer
def test_p4g_special_point_is_square_lattice(L4):
    name = L4.family.site_names[0]
    assert L4.family.site_names == ["t1"]
    sp = special_points(L4, name, -1.0, 0.5)
    hit = [d for d in sp if abs(d["value"] + 0.5) < 1e-6]
    assert len(hit) == 1 and hit[0]["group"] == "p4m"
    assert check_family(L4.family, key=L4.key, t1=-0.5)["group"] == "p4m"      # independent route
    assert special_points(L4, name, -0.4, 0.5) == []                            # locus not reached: nothing


def test_cell_parameter_special_point(L1):
    """The minimiser path (cell parameter b). Hand-known fact: a p1 cell with |a2| = |a1| (b = 1) is rhombic and
    has a mirror along a diagonal; with the site parameters on the family's cm locus the group is cm there.
    The site parameters are put on the locus with Locus.project, b is put back off it; the group at b = 1 is
    re-checked by check_family."""
    f = L1.family
    loc = [Lc for Lc in f.supergroup_loci() if Lc.group == "cm"]
    assert loc, "this p1 entry has no cm locus: the catalogue changed"
    v = dict(loc[0].project(f.values), b=f.values["b"])                         # sites onto the locus, b off it
    assert abs(v["b"] - 1.0) > 0.05
    sp = special_points(L1, "b", 0.4, 3.0, values=v)
    assert [round(d["value"], 6) for d in sp] == [1.0] and sp[0]["group"] == "cm"
    assert check_family(f, **dict(v, b=1.0))["group"] == "cm"
    assert special_points(L1, "b", 0.4, 3.0) == []                              # relation not satisfied: nothing


def test_cell_parameter_special_points_oblique_net():
    """Hand-known: the oblique net (one site, bonds along a1 and a2) at b = 2.5 becomes rectangular at
    gamma = 90 (mirrors appear), and at gamma = 70 it becomes rhombic (cmm) at b = |a2| / |a1| = 1.
    The group at each found value is re-checked by check_family (it must be larger than p2)."""
    L = Lattice.from_recipe(dict(group="p2", sites=[(0, 0)], bonds=[((0, 0), (1, 0)), ((0, 0), (0, 1))], b=2.5, gamma=70))
    f = L.family
    assert f.names == ["b", "gamma"]
    for name, want, lo, hi in (("gamma", 90.0, 45.0, 135.0), ("b", 1.0, 0.4, 3.0)):
        sp = special_points(L, name, lo, hi)
        hit = [d for d in sp if abs(d["value"] - want) < 1e-6]
        assert len(hit) == 1, (name, sp)
        g = check_family(f, **{name: want})["group"]
        assert g != "p2" and hit[0]["group"] == g
    assert special_points(L, "gamma", 45.0, 85.0) == []                         # 90 outside the range: nothing


# ---------------------------------------------------------------------------------------------- independent route
@pytest.mark.parametrize("which", ["p4g", "p1"])
def test_intervals_against_check_family(which, L4, L1):
    L = L4 if which == "p4g" else L1
    fam = L.family
    assert _good(fam, L.key)                                                   # the start point is a lattice
    n_finite = 0
    for name in fam.site_names:
        x0 = fam.values[name]
        r = allowed_interval(L, name, x0 - 0.5, x0 + 0.5)
        a, b = r["interval"]
        assert r["current_ok"] and a <= x0 <= b
        # the legacy group route inside check_family raises ValueError within ~1e-6 of a supergroup locus
        # (seen at t1 = -0.49999976 for p4g: primitive_cell finds the smaller cell, _locate then fails at
        # tol 1e-6), so grid points within 1e-4 of a special value are replaced by the special value itself
        sp = [d["value"] for d in special_points(L, name, a, b)]
        for x in np.linspace(a, b, 15):
            x = next((s for s in sp if abs(s - x) < 1e-4), x)
            assert _good(fam, L.key, **{name: float(x)}), (name, x, "inside the interval but check_family fails")
        for e, sgn in zip(r["edges"], (-1, 1)):
            if e is not None:
                n_finite += 1
                assert not _good(fam, L.key, **{name: e + sgn * PROBE}), (name, e, "just outside but check_family passes")
        for s in r["segments"]:
            mid = 0.5 * (s["lo"] + s["hi"])
            assert not _good(fam, L.key, **{name: mid}), (name, s, "inside a forbidden segment but check_family passes")
    assert n_finite >= 1
    for name in fam.cell_names:                                                 # cell parameters: whole range (proved)
        lo, hi = (0.4, 3.0) if name == "b" else (45.0, 135.0)
        r = allowed_interval(L, name, lo, hi)
        assert r["interval"] == (lo, hi) and r["segments"] == []
        for x in np.linspace(lo, hi, 5):
            assert _good(fam, L.key, **{name: float(x)})


def test_python_kernels_agree_with_numba(L4):
    """use_numba=False (the path without numba) gives the same interval (to the refinement resolution)."""
    x0 = L4.values["t1"]
    r0 = allowed_interval(L4, "t1", x0 - 0.5, x0 + 0.5, n=101, use_numba=False)
    r1 = allowed_interval(L4, "t1", x0 - 0.5, x0 + 0.5, n=101, use_numba=True)
    assert np.allclose(r0["interval"], r1["interval"], atol=1e-5)
    assert len(r0["segments"]) == len(r1["segments"])


# ---------------------------------------------------------------------------------------------- negative controls
def test_corrupt_value_is_reported(L4):
    """A value where check_family finds a non-planar lattice: the scan must say the current value is not allowed
    and put it inside a forbidden segment."""
    fam = L4.family
    bad = next(float(x) for x in np.linspace(-0.2, 0.15, 36) if not check_family(fam, t1=float(x))["planar"])
    r = allowed_interval(L4, "t1", -0.824, 0.176, values=dict(t1=bad))
    assert not r["current_ok"]
    assert any(s["lo"] <= bad <= s["hi"] for s in r["segments"])


def test_extra_crossing_bond_orbit_is_reported(L4):
    """Add one bond orbit (site 0 to its translate by a1 + a2): check_family says the network is no longer
    planar at the current value, and the scan must report forbidden regions there."""
    fam = L4.family
    f2 = fam._with_orbits(fam.orbits + [fam._orbit_of((0, 0, 1, 1))])
    assert not check_family(f2)["planar"]                                       # independent route first
    r = allowed_interval(f2, "t1", -0.824, 0.176)
    assert not r["current_ok"] and r["segments"]
    code, _ = classify(f2, [[fam.values["t1"]]])
    assert code[0] != 0


# ---------------------------------------------------------------------------------------------- widget (headless)
def _explore(L):
    pytest.importorskip("ipywidgets")
    import matplotlib
    matplotlib.use("Agg")
    from papertilling.widgets import explore
    return explore(L, n=1)


def test_slider_callback(L4):
    """Moved from tests/test_review.py (2026-11-14); lookups adapted to the new box (bars are HTML too)."""
    box = _explore(L4)
    name = L4.family.names[-1]
    slider = next(c for c in box.children if getattr(c, "description", None) == name)
    status = box.status
    slider.value = round(L4.values[name] + 0.02, 3)                       # generic: still the same lattice
    assert "same lattice" in status.value and "#1b5e20" in status.value
    assert box.values[name] == pytest.approx(slider.value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert box.lattice().values[name] == pytest.approx(slider.value)
    slider.value = -0.5                                                   # special locus: the square lattice
    assert "p4m" in status.value and "#b00020" in status.value
    assert box.values[name] == pytest.approx(-0.5)


def test_widget_limits_bar_and_snap(L4):
    box = _explore(L4)
    s, bar = box.sliders["t1"], box.bars["t1"]
    a, b = box.limits["t1"]
    assert (s.min, s.max) == (a, b) and a <= s.value <= b
    assert _good(L4.family, L4.key, t1=a) and _good(L4.family, L4.key, t1=b)   # slider ends are lattices
    assert not _good(L4.family, L4.key, t1=a - PROBE) and not _good(L4.family, L4.key, t1=b + PROBE)
    assert "p4m" in bar.value and "repeating-linear-gradient" in bar.value
    assert [d["group"] for d in box.special["t1"]] == ["p4m"]
    before = bar.value
    s.value = s.max + 0.3                                                 # beyond the limit: clamped, no error
    assert s.value == pytest.approx(b) and bar.value != before
    s.value = -0.5 - 0.4 * STEP                                           # within half a step: snapped on
    assert box.values["t1"] == pytest.approx(-0.5, abs=1e-9) and "special value t1 = -0.5" in box.status.value


def test_widget_limits_depend_on_other_parameters(L1):
    box = _explore(L1)
    assert set(box.limits) == set(L1.family.names)
    for nm in L1.family.cell_names:
        assert box.limits[nm] == (box.sliders[nm].min, box.sliders[nm].max)
    before = dict(box.limits)
    s = box.sliders["x1"]
    s.value = round(0.5 * (s.value + s.min), 3)                           # move x1 halfway to its lower limit
    assert "same lattice" in box.status.value
    assert any(box.limits[nm] != before[nm] for nm in ("y1", "x2", "y2"))
    v = box.values
    for nm in L1.family.site_names:                                       # every slider end is still a lattice
        for x in box.limits[nm]:
            assert _good(L1.family, L1.key, **dict(v, **{nm: x}))


def test_rigid_lattice():
    sq = Lattice.from_recipe(dict(group="p4m", sites=[(0, 0)], bonds=[((0, 0), (1, 0))]))   # square lattice
    box = _explore(sq)
    assert "no free parameter" in box.status.value and box.limits == {}


def test_time_one_move(L4, L1):
    out = {}
    for tag, L, name in (("p4g #1", L4, "t1"), ("p1 #1", L1, "x1")):
        box = _explore(L)
        s = box.sliders[name]
        t = time.time()
        s.value = round(s.value + 2 * s.step, 6)
        out[tag] = time.time() - t
        print(f"# time one slider move on {tag} (scan of every slider + check_family + plot): {out[tag]:.3f} s"
              f" (numba {KN.HAVE_NUMBA})")
    assert out["p4g #1"] < 3.0 and out["p1 #1"] < 6.0                          # loose: only catches a blow-up


def test_widget_without_numba(L4, L1, monkeypatch):
    """As on a machine without numba (KN.HAVE_NUMBA false): the pure-Python kernels, a coarser scan grid."""
    monkeypatch.setattr(KN, "HAVE_NUMBA", False)
    for tag, L, name in (("p4g #1", L4, "t1"), ("p1 #1", L1, "x1")):
        box = _explore(L)
        s = box.sliders[name]
        t = time.time()
        s.value = round(s.value + 2 * s.step, 6)
        print(f"# time one slider move on {tag} WITHOUT numba: {time.time() - t:.3f} s")
        assert "same lattice" in box.status.value
        for nm in L.family.site_names:
            for x in box.limits[nm]:
                assert _good(L.family, L.key, **dict(box.values, **{nm: x}))
