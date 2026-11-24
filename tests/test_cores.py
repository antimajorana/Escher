"""Tests of papertilling.defects.cores: the gallery of the cores of one dislocation.

Expected values are derived by hand in the docstrings, or cited (C04 of "Defects on wallpaper lattices"),
never read off the code's output. Definitions: cores.py module docstring (Volterra = cut and reglue only;
completion = a Volterra core plus non-crossing new bonds between under-coordinated ring sites).
"""
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import papertilling  # noqa: E402
from papertilling.defects.build import build, graph_core, isomorphic_cores, lattice_data  # noqa: E402
from papertilling.defects.config import dislocation_core_list, equivalent_single  # noqa: E402
from papertilling.defects.cores import config_entry, core_gallery, describe, plot_core_gallery  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "checks"))
from drawing_check import crossings  # noqa: E402


@pytest.fixture(scope="module")
def square_gallery():
    lat = lattice_data("square")
    t0 = time.time()
    g = core_gallery(lat, (1.0, 0.0), grid=1)
    print(f"# time square gallery {time.time() - t0:.1f} s")
    return lat, g


@pytest.fixture(scope="module")
def honeycomb_gallery():
    lat = lattice_data("honeycomb")
    t0 = time.time()
    g = core_gallery(lat, lat.cell.a1, grid=2)
    print(f"# time honeycomb gallery {time.time() - t0:.1f} s")
    return lat, g


def test_equivalence_gallery_uses_requested_radius():
    conf = build("square", dict(kind="dislocation", b=(1, 0), at=(0.371, 0.413)), R=6, picture="raw")
    with pytest.raises(ValueError, match="patch rim"):
        conf.core_graph(2)
    res = equivalent_single(conf, radius=0, grid=1, show=True)
    fig = res["figure"]
    assert "this configuration" in fig.axes[0].get_title()
    assert any("Volterra core" in ax.get_title() for ax in fig.axes[1:])
    plt.close(fig)


def test_scan_reports_last_attempt_when_a_larger_patch_is_still_too_small():
    _, scan = dislocation_core_list("square", (1, 0), R=5.5, radius=2, grid=1)
    assert scan["skipped_rim"] > 0
    assert scan["R"] == pytest.approx(5.5 * 1.4)


def test_gallery_can_draw_a_cancelled_pair_without_core_sites():
    at = (0.371, 0.413)
    conf = build("square", dict(kind="dislocation", b=(1, 0), at=at),
                 dict(kind="dislocation", b=(-1, 0), at=at), R=7, picture="raw")
    entry = config_entry(conf)
    assert not entry["core_sites"] and not entry["faces"]
    fig = plot_core_gallery([], extra=[entry])
    assert "no irregular sites" in fig.axes[0].get_title()
    assert len(fig.axes[0].collections[0].get_segments()) > 0
    plt.close(fig)


def _degs(H):
    return sorted(d for _, d in H.nodes(data="deg"))


def test_square_cores_by_hand(square_gallery):
    """Square lattice (perfect coordination 4, only 4-rings), b = (1, 0).
    Perpendicular cut: core point c in the square [i, i+1] x [j, j+1], cut down (direction (0, -1)); the
    removed half-strip {x in [cx, cx + 1), y < cy} holds the sites (i+1, y), y <= j. Regluing bonds (i, y) to
    (i+2, y). The face round the cut end is (i,j) -> (i+2,j) -> (i+2,j+1) -> (i+1,j+1) -> (i,j+1): a
    pentagon; (i+1, j+1) lost its bond to (i+1, j): one 3-bond site, every other site keeps 4 bonds.
    So: a Volterra core with rings (5,) whose core graph has exactly one site of degree 3.
    Oblique cut: deleting a pentagon bond (p, q) next to the 3-bond site a (a-p adjacent on the pentagon
    a, p, q, r, t) joins the pentagon to the square p, s1, s2, q: a 7-ring a, p, s1, s2, q, r, t with three
    3-bond sites a, p, q, two of them (a, p) bonded. Its completions (one new bond: each of a, p, q may get
    one): (p, q) restores the pentagon core; (a, q) cuts the 7-ring into the pentagon a, p, s1, s2, q (3-bond
    site p) and the square q, r, t, a: again a pentagon with one 3-bond site among squares; (a, p) is
    already a bond; two bonds would give a, p or q a fifth bond. The (p, q) completion is the pentagon core
    by construction (hand); that the (a, q) one is isomorphic to it too (same ring sizes and degrees, which
    alone do not prove it) is computational evidence: the scan lists no completion core. So the gallery is
    the two Volterra cores, and the pentagon core is also reached as a completion of the 7-ring (the 7-ring
    comes from the oblique cuts: test_config.test_negative_controls, computational evidence)."""
    lat, g = square_gallery
    assert [e["kind"] for e in g] == ["Volterra", "Volterra"]
    pent = [e for e in g if e["rings"] == (5,)]
    hept = [e for e in g if e["rings"] == (7,)]
    assert len(pent) == 1 and len(hept) == 1
    H = pent[0]["graph"]
    assert _degs(H).count(3) == 1 and set(_degs(H)) == {3, 4}
    d = hept[0]["defect"]
    deg = d.degrees()
    three = [v for v in hept[0]["core_sites"] if deg[v] == 3]
    assert len(three) == 3
    assert sum((min(u, v), max(u, v)) in set(d.edges) for i, u in enumerate(three) for v in three[i + 1:]) == 1
    assert any(hept[0]["name"] in note for note in pent[0]["also"])
    assert "pentagon" in pent[0]["description"] and "one under-coordinated" in pent[0]["description"]


def test_honeycomb_57_completion(honeycomb_gallery):
    """Honeycomb, b = a1 (|b| = sqrt3 bond lengths: the shortest dislocation). The 5-7 pair (+60 next to
    -60) has every site 3-bonded and one odd ring, so it is not bipartite, while every cut-and-reglue core is
    bipartite (C04, proved result there): it can only be a completion core. Expected: a completion entry
    whose rings, hexagons left out, are exactly (5, 7), with completion bonds, every site of its core graph
    3-bonded, described as a 5-7 pair; every Volterra entry bipartite."""
    lat, g = honeycomb_gallery
    hits = [e for e in g if e["kind"] == "completion" and tuple(r for r in e["rings"] if r != 6) == (5, 7)]
    assert hits
    e = hits[0]
    assert e["chords"] and set(_degs(e["graph"])) == {3}
    assert e["description"].startswith("5-7 pair")
    assert not nx.is_bipartite(e["graph"])
    assert all(nx.is_bipartite(v["graph"]) for v in g if v["kind"] == "Volterra")


@pytest.mark.parametrize("which", ["square", "honeycomb"])
def test_drawn_pictures(which, square_gallery, honeycomb_gallery):
    """Every entry: the picture has no crossing bond, completion bonds included (drawing_check.crossings
    on all bonds at full length); the completion bonds are bonds of the drawn crystal; the faces after
    adding them are the expected ones (core ring replaced by its split rings)."""
    lat, g = square_gallery if which == "square" else honeycomb_gallery
    for e in g:
        d = e["defect"]
        assert e["drawing"] is not None
        E = np.array(d.edges)
        assert crossings(d.positions, E) == 0, e["name"]
        assert all((min(u, v), max(u, v)) in set(d.edges) for u, v in e["chords"])
        assert e["faces_ok"], e["name"]
        assert sorted(len(f) for f in e["faces"]) == sorted(e["rings"])


@pytest.mark.parametrize("which", ["square", "honeycomb"])
def test_gallery_matches_core_list(which, square_gallery, honeycomb_gallery):
    """The gallery lists the cores of dislocation_core_list (same names, order, rings), and the core graph
    read again from each drawn crystal (graph_core on its own graph) is isomorphic to that candidate's."""
    lat, g = square_gallery if which == "square" else honeycomb_gallery
    b = (1.0, 0.0) if which == "square" else lat.cell.a1
    cands, _ = dislocation_core_list(lat, b, grid=1 if which == "square" else 2)
    assert [e["name"] for e in g] == [c["name"] for c in cands]
    for e, c in zip(g, cands):
        assert tuple(e["rings"]) == tuple(c["rings"])
        H = graph_core(e["defect"].to_networkx(), e["rim"], lat, 2)[1]
        assert isomorphic_cores(H, c["graph"]), e["name"]


def test_negative_controls(honeycomb_gallery):
    """A wrong core graph is not isomorphic to the entry's (one completion bond removed: the 5-7 graph
    changes); a disclination charge has no gallery; describe() on hand cases."""
    lat, g = honeycomb_gallery
    e = next(x for x in g if x["kind"] == "completion" and x["chords"])
    G = e["defect"].to_networkx()
    G.remove_edge(*e["chords"][0])
    assert not isomorphic_cores(graph_core(G, e["rim"], lat, 2)[1], e["graph"])
    sq = lattice_data("square")
    one = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), picture="raw")
    assert equivalent_single(one)["gallery"] == []
    assert describe((5, 6, 7), {6}).startswith("5-7 pair, plus one hexagon")
    assert describe((5,), {4}, [1]) == "one pentagon; one under-coordinated site on the ring"


def _p4g_pair():
    lat = lattice_data(papertilling.load_catalogue()["p4g"].lattices[0])
    four = sorted((r for r in lat.centres if r["order"] == 4), key=lambda r: r["label"])
    assert [r["label"] for r in four] == ["2a(i)", "2a(ii)"]
    return lat, build(lat, dict(kind="disclination", angle=90, at=four[0]["point"]),
                      dict(kind="disclination", angle=-90, at=four[1]["point"]), picture="raw")


def _check_p4g(grid):
    """Shastry-Sutherland (p4g), +90 at 2a(i) and -90 at 2a(ii). Hand (same algebra as
    test_config.test_user_example_square): b = (I - R90)(c2 - c1) with c2 - c1 = (1/2, 1/2) a: |b| = a,
    a unit dislocation. That its core matches a completion core and no Volterra core is the known case of
    the task statement (computational evidence of the scan, not derived here); the figure must then show
    the Volterra cores too, and every entry must pass the chord-face check and the core-graph check."""
    lat, conf = _p4g_pair()
    t0 = time.time()
    res = equivalent_single(conf, grid=grid, show=True)
    print(f"# time p4g pair equivalent_single(grid={grid}, show=True) {time.time() - t0:.1f} s")
    q = conf.total_charge()
    assert q["kind"] == "dislocation" and abs(np.linalg.norm(q["b"]) - np.linalg.norm(lat.cell.a1)) < 1e-9
    assert res["equivalent"] and res["matches"] and all(m.startswith("completion") for m in res["matches"])
    gal = res["gallery"]
    assert any(e["kind"] == "Volterra" for e in gal)
    assert [e["name"] for e in gal] == [c["name"] for c in res["candidates"]]
    titles = [ax.get_title() for ax in res["figure"].axes]
    assert sum("Volterra core" in t for t in titles) == sum(e["kind"] == "Volterra" for e in gal)
    assert any(t.startswith("this configuration") for t in titles)
    assert all(any(m in t for t in titles) for m in res["matches"])
    assert all(e["faces_ok"] is True and e["iso_ok"] is True for e in gal),         [e["name"] for e in gal if not (e["faces_ok"] is True and e["iso_ok"] is True)]
    plt.close("all")


def test_volterra_match_still_shows_completions(honeycomb_gallery):
    """A single honeycomb dislocation built by cut and reglue (b = a1, generic cut end) has a Volterra
    core; the charge also has completion cores (the 5-7 pair, test above), so the figure shows both kinds."""
    lat, g = honeycomb_gallery
    one = build(lat, dict(kind="dislocation", b=lat.cell.a1, at=0.371 * lat.cell.a1 + 0.413 * lat.cell.a2), picture="raw")
    res = equivalent_single(one, grid=2, show=True)
    titles = [ax.get_title() for ax in res["figure"].axes]
    assert any("Volterra core" in t for t in titles) and any("completion core" in t for t in titles)
    plt.close("all")


def test_p4g_pair_grid2():
    """grid=2 (grid=1 finds no matching core: its scan is too coarse, the verdict says so)."""
    _check_p4g(2)


@pytest.mark.slow
def test_p4g_pair_grid3():
    _check_p4g(3)


def test_show_false_unchanged(square_gallery):
    """show=False: the old keys and values, plus 'gallery' (the same entries as core_gallery's)."""
    sq = lattice_data("square")
    pair = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)),
                 picture="raw")
    res = equivalent_single(pair, grid=1)
    assert set(res) == {
        "charge", "kind", "single", "candidates", "matches",
        "volterra_matches", "completion_matches", "match_tier",
        "equivalent", "scan", "verdict", "gallery",
    }
    assert all("defect" not in c and "graph" not in c for c in res["candidates"])
    assert [e["name"] for e in res["gallery"]] == [e["name"] for e in square_gallery[1]]
    fig = plot_core_gallery(res["gallery"], highlight=res["matches"])
    assert len([a for a in fig.axes if a.get_title()]) == len(res["gallery"])
    plt.close("all")


@pytest.mark.slow
def test_kagome_gallery_drawn():
    """Kagome, b = a1: every entry drawn without crossings and isomorphic to its candidate."""
    lat = lattice_data("kagome")
    g = core_gallery(lat, lat.cell.a1)
    cands, _ = dislocation_core_list(lat, lat.cell.a1)
    assert any(e["kind"] == "Volterra" for e in g) and any(e["kind"] == "completion" for e in g)
    for e, c in zip(g, cands):
        assert crossings(e["defect"].positions, np.array(e["defect"].edges)) == 0, e["name"]
        assert isomorphic_cores(graph_core(e["defect"].to_networkx(), e["rim"], lat, 2)[1], c["graph"])
