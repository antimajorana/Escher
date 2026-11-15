"""DefectCatalogue controls. Every expected value is derived by hand below (or cited), not read off the code.

HAND DERIVATIONS
[D1] Square, primitive Burgers vectors (gcd 1) with |b| <= 2: (+-1,0), (0,+-1) of length 1 and (+-1,+-1)
     of length sqrt2; the next primitive ones, (2,+-1) etc., have length sqrt5 > 2, and (2,0) has gcd 2.
     8 vectors. P+ = C4 (4 rotations) acts freely on nonzero vectors, so 2 classes of size 4.
[D2] Square disclinations: G+-classes of rotation centres = site (0,0) order 4, face centre (1/2,1/2)
     order 4, bond midpoints (1/2,0) ~ (0,1/2) (one class: the 90-degree rotation swaps them) order 2.
     Angles k 360/n in [-180, 180], k != 0, +180 != -180: 4 + 4 + 2 = 10 (C05 minus its two +270).
[D3] p3 (chiral triangular): P+ = C3 = {0, 120, 240 degrees}; -b = R(180) b is not R(0, 120, 240) b for
     b != 0, so b and -b are in different classes; each class has 3 members.
[D4] p4g (Shastry-Sutherland): the 4-fold centres (0,0), (1/2,1/2) of the conventional cell form one
     G-orbit (ITA 2a) because a glide maps one onto the other, but G+ = p4 keeps them apart (a proper
     element maps a 4-fold centre of p4 to a lattice translate of itself here); an improper op swaps them,
     so the two G+-classes are mirror partners.
[D5] Euclidean normaliser of p4m: p4m with basis (a-b)/2, (a+b)/2 (ITA Vol. A, Part 15,
     Table 15.2.1.3 -- locator from the brief, table not opened in this session; re-derived by hand:
     the translation (1/2,1/2) maps every rotation centre, mirror and glide of p4m onto one of the same
     kind, and is not in p4m). Index 16 / 8 = 2, the extra coset is that translation; it maps sites
     (0,0) to face centres (1/2,1/2) and back, and bond midpoints to bond midpoints.
[D6] p4 (chiral square): normaliser p4m with basis (a-b)/2, (a+b)/2, index 4 (point group 4mm / 4 x 2
     translations), two of the cosets have det -1 (mirror-image domains).
[D7] Reflection walls on the chiral square (r = 0.3, twist 20 deg; sites at 0.3 (cos, sin)(20 + 90k)):
     the wall graph depends only on which sites lie on which side (C02, proved), so the classes in one
     direction are the critical offsets (a line through a site) and the gaps between them. Heights
     along (0,1) mod 1: 0.3 sin20 = 0.10261, 0.3 sin110 = 0.28191, and 1 minus these: 4 critical + 4
     gaps = 8. p4 has no improper op and its only proper op keeping an oriented line direction is the
     identity, so nothing merges: 8 classes per direction. C02 R5/R6: exactly one of the 8 is invisible
     in the bond network per direction (Wx6 = gap 0.2819 < y < 0.7181; Wd8 = the diagonal gap
     containing offset 0): 2 of 16 in total. (The brief's "2 invisible" per direction reads C02's
     total of 2; this file tests C02's statement.)
"""
import math
import time

import numpy as np
import pytest

from papertilling.defects import DefectCatalogue
from papertilling.legacy.lattice_families import Family
from papertilling.legacy.lab_defects import is_orientable


@pytest.fixture(scope="module")
def square():
    return DefectCatalogue("square", bmax=2.0)


@pytest.fixture(scope="module")
def chiral_square():
    return DefectCatalogue("chiral_square", bmax=2.0)


def test_square_dislocations(square):                                   # [D1]
    ds = square.dislocations
    assert len(ds) == 2 and sum(d["size"] for d in ds) == 8
    assert ds[0]["b_frac"] == (1, 0) and ds[0]["size"] == 4 and abs(ds[0]["length"] - 1) < 1e-12
    assert ds[1]["b_frac"] == (1, 1) and ds[1]["size"] == 4 and abs(ds[1]["length"] - math.sqrt(2)) < 1e-12
    assert all(d["self_inverse"] for d in ds)
    assert ds[0]["spec"] == dict(kind="dislocation", b=(1.0, 0.0), at=ds[0]["spec"]["at"])


def test_square_disclinations(square):                                  # [D2]
    dl = square.disclinations
    assert len(dl) == 10
    by_loc = {}
    for d in dl:
        by_loc.setdefault(d["location"], []).append(d["angle"])
    assert sorted(by_loc["site"]) == [-180, -90, 90, 180]
    assert sorted(by_loc["face centre"]) == [-180, -90, 90, 180]
    assert sorted(by_loc["bond midpoint"]) == [-180, 180]
    d = next(x for x in dl if x["location"] == "face centre" and x["angle"] == 90)
    R, t = d["H"]                                                       # H = (R_alpha, (I - R_alpha) c)
    assert np.allclose(t, (np.eye(2) - R) @ d["centre"])


def test_input_routes_agree(square):
    """Positive control: Family and recipe-dict inputs give the same classes as the catalogue name."""
    fam = Family.from_cell("square")
    for lat in (fam, fam.recipe()):
        c = DefectCatalogue(lat)
        assert c.group == "p4m"
        assert [d["size"] for d in c.dislocations] == [d["size"] for d in square.dislocations]
        assert len(c.disclinations) == 10


def test_chiral_triangular_b_not_minus_b():                             # [D3]
    c = DefectCatalogue("chiral_triangular", bmax=2.0)
    assert c.group == "p3"
    assert c.dislocations and all(not d["self_inverse"] and d["inverse_id"] != d["id"] and d["size"] == 3
                                  for d in c.dislocations)


def test_shastry_sutherland_split_centres():                           # [D4]
    c = DefectCatalogue("shastry_sutherland")
    assert c.group == "p4g"
    four = [r for r in c.centre_classes if r["order"] == 4]
    assert len(four) == 2 and four[0]["gclass"] == four[1]["gclass"]   # one G-class (2a), two G+-classes
    assert four[0]["mirror_of"] == four[1]["id"] and four[1]["mirror_of"] == four[0]["id"]
    d = next(x for x in c.disclinations if x["centre_class"] == four[0]["id"] and x["angle"] == 90)
    assert d["mirror_id"] == f"D({four[1]['label']}, +90)"


def test_square_normaliser(square):                                     # [D5]
    N = square.normaliser
    assert N["index"] == 2 and N["complete"] and N["ita_generic_index"] == 2
    (cos,) = N["cosets"]
    assert cos["kind"] == "translation" and cos["check_group"] and cos["check_geometry"]
    M, t = cos["sigma"]
    f = t - np.round(t)
    assert np.allclose(M, np.eye(2)) and np.allclose(np.abs(f), [0.5, 0.5])
    w = square.domain_walls[0]
    assert w["sites_to"] == "face centre"
    labels = {r["location"]: r["label"] for r in square.centre_classes}
    assert f"{labels['site']}->{labels['face centre']}" in w["centre_map"]
    assert f"{labels['face centre']}->{labels['site']}" in w["centre_map"]
    assert w["spec"]["kind"] == "wall" and w["spec"]["wall_type"] == "a"
    x = w.build(R=3.0)                                                  # legacy interface builds
    assert len(x.sites) > 10


def test_chiral_square_domains_and_twin_walls(chiral_square):           # [D6], [D7]
    t0 = time.time()
    N = chiral_square.normaliser
    assert N["index"] == 4 and any(c["kind"] == "mirror" and np.linalg.det(c["sigma"][0]) < 0 for c in N["cosets"])
    tw = chiral_square.twin_walls
    per_dir = {}
    for w in tw:
        per_dir.setdefault(tuple(abs(x) for x in w["direction_frac"]), []).append(w)
    assert sorted(per_dir) == [(1, 0), (1, 1)]
    assert all(len(v) == 8 for v in per_dir.values())
    inv = [w for w in tw if not w["visible"]]
    assert len(inv) == 2 and all(w["network_invisible"] and not w["sigma_in_G"] for w in inv)
    (wx,) = [w for w in inv if tuple(abs(x) for x in w["direction_frac"]) == (1, 0)]
    h = 0.3 * math.sin(math.radians(110))
    assert wx["interval"][0] == "gap" and abs(wx["interval"][1] - h) < 1e-6 and abs(wx["interval"][2] - (1 - h)) < 1e-6
    (wd,) = [w for w in inv if tuple(abs(x) for x in w["direction_frac"]) == (1, 1)]
    lo, hi = wd["interval"][1], wd["interval"][2]
    assert wd["interval"][0] == "gap" and (lo < 1e-9 < hi or lo < wd["spacing"] < hi)   # contains offset 0
    print(f"# time chiral square normaliser + 16 walls: {time.time() - t0:.1f} s")


def test_negative_mirror_wall_invisible(square):
    """Negative control: a wall on a mirror line of the square lattice is invisible."""
    w = next(x for x in square.twin_walls if x["direction_frac"] == (1, 0) and x["through_site"])
    assert w["sigma_in_G"] and not w["visible"]
    assert w.build(R=3.0).info["invisible"] is True


def test_negative_disallowed_disclination(square):
    """Negative control: 90 deg at a 2-fold centre, and any angle off a rotation centre, are refused."""
    with pytest.raises(ValueError):
        square.check_disclination(90.0, (0.5, 0.0))
    with pytest.raises(ValueError):
        square.build_disclination(90.0, (0.5, 0.0))
    with pytest.raises(ValueError):
        square.check_disclination(90.0, (0.3, 0.1))
    assert square.check_disclination(180.0, (0.5, 0.0))["order"] == 2


def test_mobius_and_glide_breaking(square):
    """(c) a Moebius seam is non-orientable (C02 R1); (d) a pure reflection on the p4m glide line
    x - y = 1/2 gives side B = L shifted by the full glide length tau = |(1/2,1/2)| = 0.7071 along the
    line (sigma = t(-tau u) o g), which is the (1/2,1/2) antiphase domain [D5] and the twin wall on that line."""
    seam = square.mobius_seams[0]
    assert seam["non_orientable"] and not is_orientable(seam.build())
    gb = next(w for w in square.glide_breaking if w["line_kind"] == "glide")
    assert abs(gb["slip"] - math.sqrt(0.5)) < 1e-9
    assert gb["same_domains_as"] == square.normaliser["cosets"][0]["id"]
    assert gb["same_as_twin"] is not None


def test_show_smoke(square):
    import matplotlib
    matplotlib.use("Agg")
    fig = square.show("disclinations", entries=square.disclinations[:2], R=3.0)
    assert fig is not None


def test_fallback_labels_without_groups(monkeypatch):
    """Without papertilling.groups the labels fall back to 'order-n class X' and [D4] still holds."""
    import papertilling.defects.catalogue as C
    monkeypatch.setattr(C, "_groups_attr", lambda name: None)
    c = DefectCatalogue("shastry_sutherland")
    four = [r for r in c.centre_classes if r["order"] == 4]
    assert [r["gclass"] for r in four] == ["order-4 class A"] * 2
    assert {r["label"] for r in four} == {"order-4 class A/1", "order-4 class A/2"}
    assert four[0]["mirror_of"] == four[1]["id"]


@pytest.mark.parametrize("name", ["square", "honeycomb"])
def test_glide_reflection_twin_is_not_in_G(name, square):
    """A pure reflection on a glide line is never in G (a mirror and a glide on one line would make the
    glide part a lattice translation shorter than the period). So the twin wall (b) that a glide-breaking
    wall (d) coincides with must have sigma not in G (regression: a gap representative once snapped to a
    mirror sitting on the gap's end point)."""
    c = square if name == "square" else DefectCatalogue(name)
    tw = {w["id"]: w for w in c.twin_walls}
    for w in c.glide_breaking:
        if w["line_kind"] == "glide" and w["same_as_twin"] is not None:
            assert not tw[w["same_as_twin"]]["sigma_in_G"]
    if name == "honeycomb":
        # [hand] zigzag direction (1,0): site heights 0 and 1 mod 1.5 -> 2 lines through sites + 2 gaps;
        # no element of p6m maps (y=0, side above) to (y=1, side above): the ops keeping the line set
        # either fix y=0 (x-mirror keeps the side, 2-fold/y-mirror swap it) or exchange y=0 and y=1
        # (y=1/2 mirror, 2-fold at a bond midpoint) and then swap the side. Armchair direction: all
        # sites on one line mod 0.866 -> 1 line + 1 gap. Total 6. sigma in G: the zigzag gap holds the
        # mirror y = 1/2, the armchair line through the sites is a mirror: 2.
        assert len(tw) == 6 and sum(w["sigma_in_G"] for w in tw.values()) == 2


def test_twin_walls_wraparound_orbit_stabiliser():
    """Regression for 1-ULP dl - x < 1e-9 wrap-around in _twin_setup: hs must stay strictly sorted so gap
    intervals have hi > lo and twin-wall classes satisfy orbit-stabiliser on zigzag (8) and brick_pg (16)."""
    zz = DefectCatalogue("zigzag")
    assert len(zz.twin_walls) == 8
    assert all(w["interval"][2] > w["interval"][1] for w in zz.twin_walls if not w["through_site"])

    bp = DefectCatalogue("brick_pg")
    assert len(bp.twin_walls) == 16
    assert sum(w["class_size"] for w in bp.twin_walls) == 32
    assert all(w["interval"][2] > w["interval"][1] for w in bp.twin_walls if not w["through_site"])
    gb = {w["id"]: w["same_as_twin"] for w in bp.glide_breaking}
    assert gb["Wd1"] != gb["Wd2"] and gb["Wd1"] is not None and gb["Wd2"] is not None

