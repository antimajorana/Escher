"""Refusals told apart: IMPOSSIBLE (proved from the lattice's symmetry) vs NOT FOUND (this construction, these settings).

Expected values and their sources (none is read from the code under test):
  * Proof (a), refusal.py: an orientation-reversing loop has an orientation-reversing holonomy in the lattice's group, so a
    group with no mirror or glide line (p1 p2 p3 p4 p6) carries no Moebius band, Klein bottle or projective plane.
  * Proof (b): a cone's holonomy is the rotation by its total angle about it; with no rotation (p1 pm pg cm) every Frank
    angle is <= 0, and Gauss-Bonnet needs +720 (sphere) or +360 (projective plane). So both are impossible.
  * The flat surfaces exist for every group (torus, cylinder; Moebius band and Klein bottle from any glide), the genus-g
    surfaces for every lattice (branch points of -360 deg are allowed by every group), the sphere for honeycomb (p6m) and
    chiral_square (p4: the cube, 8 x +90), the projective plane for honeycomb (the hemi-dodecahedron): test_sphere.py.
  * Proof (c), _obstruction.py: every rotation centre of the lattice fails local cone graph checks (e.g. zigzag has
    every 2-fold centre on a bond midpoint; p3#1 and p31m#1 have every 3-fold centre inside a triangle), so sphere and
    projective plane are IMPOSSIBLE (scope="lattice").
  * Shastry-Sutherland (p4g) realises the projective plane by alternating between its two split 4-fold classes 2a(i) and
    2a(ii) (V=6, E=15, F=10 at size (1, 1)); single-class and non-diagonal sizes remain NOT FOUND.
  * Positive control of symmetry_facts: on the first three lattices of each of the 17 groups, the rotation orders and the
    existence of an orientation-reversing symmetry found on the cell equal those of the group's label (International
    Tables). Negative control: the reflections of the square cell are found NOT to be symmetries of chiral_square.
"""
import functools

import pytest

import papertilling as pt
from papertilling.surfaces import (sphere, projective_plane, klein_bottle, mobius_band, genus_surface, surface_table, SurfaceImpossible, SurfaceNotFound, SurfaceRefused,
                                   symmetry_facts, torus)
from papertilling.surfaces.pieces import ConeRefused


@functools.lru_cache(maxsize=None)
def table(name):
    return surface_table(pt.named(name), show=False)


def statuses(name):
    return table(name).statuses()


# ----------------------------------------------------------------------------------------------------------------
# the classes
# ----------------------------------------------------------------------------------------------------------------
def test_exception_hierarchy():
    for cls in (SurfaceImpossible, SurfaceNotFound):
        assert issubclass(cls, SurfaceRefused) and issubclass(cls, ConeRefused) and issubclass(cls, ValueError)
    assert not issubclass(SurfaceImpossible, SurfaceNotFound) and not issubclass(SurfaceNotFound, SurfaceImpossible)


def test_constructors_raise_the_right_class():
    with pytest.raises(SurfaceImpossible, match="no mirror or glide line"):
        klein_bottle(pt.named("chiral_square"))
    with pytest.raises(SurfaceImpossible, match="no mirror or glide line"):
        mobius_band(pt.named("chiral_square"))
    with pytest.raises(SurfaceImpossible, match="no mirror or glide line"):
        projective_plane(pt.named("chiral_square"))
    with pytest.raises(SurfaceImpossible, match="no rotation"):
        sphere(pt.named("brick_pg"))
    with pytest.raises(SurfaceImpossible, match="bond midpoint") as e:   # Proof (c): every 2-fold centre on a bond
        sphere(pt.named("zigzag"))
    assert e.value.retry is False and e.value.scope == "lattice"
    with pytest.raises(SurfaceNotFound) as e:                          # p4g: non-diagonal size (2, 1) fails pairing_test
        projective_plane(pt.named("shastry_sutherland"), (2, 1))
    assert "try size=" in e.value.hint
    with pytest.raises(SurfaceNotFound, match="faces_at_least_3") as e:
        sphere(pt.named("oblique"), (1, 1), centre="face centre")
    with pytest.raises(SurfaceNotFound) as e:
        genus_surface(pt.named("square"), g=2, torus_size=(1, 1))
    assert "torus_size=(3, 3)" in e.value.hint
    with pytest.raises(ValueError) as e:                               # a bad argument is neither: a glide class with
        klein_bottle(pt.named("brick_pg"), along=2)                    # no mirror closes only after odd steps (C1 WP-B1)
    assert not isinstance(e.value, SurfaceRefused)


def test_built_surfaces_have_status_ok():
    assert torus(pt.named("square")).status == "OK"
    assert sphere(pt.named("honeycomb")).status == "OK"


# ----------------------------------------------------------------------------------------------------------------
# the table: known answers
# ----------------------------------------------------------------------------------------------------------------
def test_chiral_square_p4():
    s = statuses("chiral_square")
    assert s["Moebius band"] == s["Klein bottle"] == s["projective plane"] == "IMPOSSIBLE"
    assert s["sphere"] == s["torus"] == "OK"
    rows = {r["surface"]: r for r in table("chiral_square")}
    assert "p4 has no mirror or glide line" in rows["Moebius band"]["reason"]
    assert rows["sphere"]["cones"] == "8 x +90" and rows["sphere"]["S"].checks(show=False)["ok"]


def test_straight_p1():
    s = statuses("straight_p1")
    for name in ("sphere", "projective plane", "Moebius band", "Klein bottle"):
        assert s[name] == "IMPOSSIBLE", name
    for name in ("torus", "cylinder", "genus 2", "genus 3"):
        assert s[name] == "OK", name


def test_honeycomb_everything():
    assert set(statuses("honeycomb").values()) == {"OK"}


def test_brick_pg():
    s = statuses("brick_pg")
    assert s["sphere"] == s["projective plane"] == "IMPOSSIBLE"
    assert s["Moebius band"] == s["Klein bottle"] == "OK"


def test_shastry_sutherland_projective_plane_ok():
    rows = {r["surface"]: r for r in table("shastry_sutherland")}
    r = rows["projective plane"]
    assert r["status"] == "OK" and r["S"] is not None and r["S"].checks(show=False)["ok"]
    assert r["S"].params["centre"] == "2a(i)+2a(ii)" and len(r["S"].sites) == 6
    assert rows["sphere"]["status"] == "OK"


def test_ok_rows_pass_their_checks_and_impossible_rows_have_no_surface():
    for name in ("chiral_square", "straight_p1", "honeycomb", "brick_pg", "shastry_sutherland", "zigzag", "rectangular"):
        for r in table(name):
            assert r["status"] in ("OK", "IMPOSSIBLE"), (name, r)
            if r["status"] == "OK":
                assert r["S"].checks(show=False)["ok"] and r["sites"] == len(r["S"].sites)
            else:
                assert r["S"] is None and r["reason"] and r["scope"] in ("group", "lattice")


def test_default_only_and_override():
    T = surface_table(pt.named("square"), sizes={"sphere": [{"size": (2, 0)}]}, only=["sphere"], show=False)
    assert T[0]["status"] == "OK" and T[0]["S"].params["size"] == (2, 0)
    T = surface_table(pt.named("rectangular"), sizes="default", only=["projective plane"], show=False)
    assert T[0]["status"] == "NOT FOUND" and len(T[0]["tried"]) == 1 and T[0]["hint"]


def test_17x8_polyhedra_and_scopes():
    cat = pt.load_catalogue()
    for g in cat:
        T = surface_table(cat[g].lattices[0], only=["sphere", "projective plane"], show=False)
        for r in T:
            assert r["status"] in ("OK", "IMPOSSIBLE"), (g, r)
            if r["status"] == "IMPOSSIBLE":
                assert r["scope"] in ("group", "lattice")
    T_zz = surface_table(pt.named("zigzag"), only=["sphere", "projective plane"], show=False)
    assert all(r["status"] == "IMPOSSIBLE" and r["scope"] == "lattice" and len(r["tried"]) == 1 for r in T_zz)
    T_cs = surface_table(pt.named("chiral_square"), only=["projective plane"], show=False)
    assert T_cs[0]["status"] == "IMPOSSIBLE" and T_cs[0]["scope"] == "group"


# ----------------------------------------------------------------------------------------------------------------
# the symmetry finder behind IMPOSSIBLE
# ----------------------------------------------------------------------------------------------------------------
def test_symmetry_facts_agree_with_the_group_labels():
    cat = pt.load_catalogue()
    for g in cat:
        for L in cat[g].lattices[:3]:
            f = symmetry_facts(L)
            assert f["agrees"], (g, f["rotation_orders"], f["reversing"])


def test_symmetry_facts_negative_control():
    f = symmetry_facts(pt.named("chiral_square"))
    assert f["rotation_orders"] == [2, 4] and not f["reversing"]
    f = symmetry_facts(pt.named("square"))
    assert f["rotation_orders"] == [2, 4] and f["reversing"]
