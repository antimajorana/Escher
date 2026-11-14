"""The named lattices (papertilling/sheet.py) and the "sheet" rule (points alone exactly G, fewest kinds of bond).

Expected values come from the hand-made sheet "Lattices of the 17 wallpaper groups" and from hand arguments:
  * the group written on the sheet for each lattice; three drawings have MORE symmetry than their label:
      Straight   (sheet p1): each diagonal is unchanged by a half-turn about its own midpoint          -> p2
      Brick      (sheet pg): rows offset by exactly half a brick have 2-fold rotations and mirrors     -> cmm
      Shastry's Ladder (sheet pm): the zigzag has a glide and 2-fold rotations                         -> pmg
  * the sheet's lattices with one kind of site are what the sheet rule is meant to produce: each must be in
    the sheet-rule list of its (measured) group, and the ones listed in FIRST below must be its first entry;
  * one or two points per cell with only translations always have a 2-fold rotation, so p1 needs 3 sites.
"""
import warnings

import pytest

import papertilling as pt
from papertilling.lattice import load_catalogue
from papertilling.legacy.lattice_families import Family
from papertilling.sheet import SHEET, VARIANTS

MEASURED = dict(straight="p2", brick="cmm", shastrys_ladder="pmg")           # the three whose label is too low
ONE_KIND = ["oblique", "elongated_triangular", "square", "triangular", "kagome", "honeycomb", "rectangular", "fractal_triangular",
            "breathing_kagome", "shastry_sutherland", "brick", "zigzag", "sutherlands_brick", "chiral_square", "chiral_kagome",
            "chiral_triangular"]
FIRST = ["oblique", "square", "triangular", "rectangular", "chiral_square", "chiral_kagome", "chiral_triangular"]


def test_every_named_lattice_loads_with_its_group():
    assert len(SHEET) == 18 and len(VARIANTS) == 3
    for name, key, sheet_group in SHEET:
        L = pt.named(name)
        assert L.group == MEASURED.get(key, sheet_group), name
        assert L.meta["name"] == name and L.meta["sheet_group"] == sheet_group
        assert L.key is not None                                    # a planar tiling with a D-symbol
    for name, key, group, parent in VARIANTS:
        assert pt.named(key).group == group, key                    # the deformed versions have the labelled group


def test_names_are_matched_loosely():
    assert pt.named("Kagomé").key == pt.named("kagome").key == pt.named("KAGOME").key
    assert pt.named("Zig Zag").key == pt.named("zigzag").key
    assert pt.named("Sutherland's Brick").group == "pgg" and pt.named("shastry-sutherland").group == "p4g"
    with pytest.raises(KeyError):
        pt.named("no such lattice")


def test_table_and_kinds_of_site():
    rows = {r["key"]: r for r in pt.named_lattices(show=False)}
    assert len(rows) == 21
    assert [k for k in (key for _, key, _ in SHEET) if rows[k]["kinds_of_site"] == 1] == [k for _, k, _ in SHEET if k in ONE_KIND]
    assert rows["straight"]["kinds_of_site"] == 2 and rows["shastrys_ladder"]["kinds_of_site"] == 2
    assert rows["kagome"]["faces"] == ["3.3.6.6"] and rows["chiral_square"]["faces"] == ["4.8.8"] and rows["chiral_kagome"]["faces"] == ["3.12.12"]
    # four drawings of one network (the kagome network), told apart only by their groups
    kag = {pt.named(k).network for k in ("kagome", "fractal_triangular", "breathing_kagome", "chiral_triangular")}
    assert len(kag) == 1


def test_named_lattices_work_with_the_defect_tools():
    L = pt.named("Shastry-Sutherland")
    cat = pt.DefectCatalogue(L)
    assert len(cat.dislocations) >= 1 and len(cat.disclinations) >= 1
    c = pt.build(L, dict(kind="dislocation", b=cat.dislocations[0]["b"], at=(0.371, 0.413)), picture="raw")
    assert c.total_charge()["kind"] == "dislocation"


@pytest.fixture(scope="module")
def sheet():
    return load_catalogue(rule="sheet")


def test_sheet_rule_lists_contain_the_sheet(sheet):
    for key in ONE_KIND:
        L = pt.named(key)
        nets = [x.network for x in sheet[L.group].all]
        assert L.network in nets, f"{key} ({L.group}) is not in the sheet-rule list"
    for key in FIRST:
        L = pt.named(key)
        assert sheet[L.group].lattices[0].network == L.network, key


def test_sheet_rule_points_alone_have_the_group(sheet):
    """Second route for the condition of the rule: the sites of every listed lattice, with no bonds."""
    for g, res in sheet.items():
        assert len(res.lattices) == 7 and len({L.network for L in res.all}) == len(res.all), g
        for L in res.lattices:
            r = L.family.recipe()
            bare = Family(g, [tuple(p) for p in r["sites"]], **{k: r[k] for k in ("a", "b", "gamma") if k in r})
            assert bare.group_at() == (g, L.family.m), (g, L.rank)
            assert L.meta["routes_agree"]


def test_sheet_rule_groups_without_one_kind_of_site(sheet):
    assert sheet["p1"].lattices[0].rank[0] == 3 and sheet["p1"].lattices[0].family.m == 3
    for g in ("pm", "pg", "cm"):
        assert sheet[g].lattices[0].rank[0] >= 2, g


@pytest.mark.slow
def test_sheet_rule_live_search_equals_catalogue(sheet):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pt.minimal_lattices("p4", rule="sheet")
    assert [L.key for L in res.lattices] == [L.key for L in sheet["p4"].lattices]
