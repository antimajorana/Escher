"""Drawn pictures are faithful planar pictures (papertilling.defects.draw, planarity).

Measured with tools/checks/drawing_check.measure (the independent tool): plain_crossings (bonds drawn in full
that cross), folded (faces of the bond-frame face structure with the wrong orientation, tiny area or a
self-crossing boundary; None when bonds carry no frame), overlaps (collinear bonds, sites on bonds).

  A. planarity.check (the package copy of the definitions) gives measure()'s numbers.
  B. picture="raw" is unchanged relative to the frozen legacy builder (fast=False: identical graph/sites and
     positions within cross-OS float64 tolerance, with no drawing stage metadata).
  C. a relaxed picture that had no crossing and no fold is unchanged (equal to fast_relax.relax_picture on
     the raw build, array_equal) and is marked stage 'given'.
  D. the relaxed pictures that were not embedded (diagnosis of 2026-11-14) now have 0 crossings, 0 folds,
     0 overlaps, no stubs, and the same graph.
  E. Notebook 2: a wall entry drawn by entry.picture() is embedded and keeps the lattice of its two domains
     (bond lengths inside a domain); a Moebius seam keeps its stub picture.
  F. crowded sites (2026-11-15): p4g lattices[0] Wd1 was drawn with wall sites stacked (closest pair 0.047 x the
     bond length, 0 crossings, 0 overlaps); it is now opened (side B moved rigidly off the wall line) with 0
     crowded sites; a wall that cannot be drawn cleanly returns its rigid picture with a note in the title;
     a configuration keeping an overlap says so in its title.
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import papertilling as pt
from papertilling.defects import draw, fast_relax, planarity
from papertilling.defects.catalogue import DefectCatalogue

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "checks"))
from drawing_check import measure  # noqa: E402


def D(a, at):
    return dict(kind="disclination", angle=float(a), at=tuple(float(x) for x in at))


def _edges(d):
    return sorted(tuple(sorted(e)) for e in d.edges)


@pytest.fixture(scope="module")
def p4g():
    return pt.load_catalogue()["p4g"].lattices[0]


SQ_PAIR = [D(90, (0, 0)), D(-90, (0.5, 0.5))]
P4G_PAIR = [D(90, (0, 0)), D(-90, (0.5, 0.5))]      # 2a(i), 2a(ii) of p4g lattices[0]


# --- A ----------------------------------------------------------------------------------------------------
def test_check_equals_measure():
    """Raw picture with stubs (control 2 of drawing_check), the same with two bonded sites swapped (3b: a
    fold), a rigid dislocation with crossings and folds (kagome T2), a wall with overlaps and no frames."""
    t0 = time.time()
    raw = pt.build("square", *SQ_PAIR, picture="raw")
    rel = pt.build("square", *SQ_PAIR, picture="relaxed")
    sw = copy.deepcopy(rel.defect)
    X = np.array(sw.positions, float)
    i = int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))
    j = next(v for (u, v), e in sw.edge.items() if u == i and e.get("ghost") is None)
    X[[i, j]] = X[[j, i]]
    sw.positions, sw._nbr = X, None
    kag = DefectCatalogue(pt.named("kagome"))
    sq = DefectCatalogue(pt.named("square"))
    pics = [raw.defect, sw, next(e for e in kag.dislocations if e["id"] == "T2").build(R=kag.R0),
            next(e for e in sq.glide_breaking if e["id"] == "Wd1").build(R=sq.R0)]
    got = []
    for d in pics:
        m, c = measure(d), planarity.check(d)
        assert (c["crossings"], c["overlaps"], c["folded"]) == (m["plain_crossings"], m["overlaps"], m["folded"])
        assert (c["crowded_pairs"], c["crowded_on"]) == (m["crowded_pairs"], m["crowded_sites_on_bonds"])
        got.append((c["crossings"], c["overlaps"], c["folded"]))
    assert got[0][0] > 0 and got[1][2] > 0 and got[2][0] > 0 and got[3][1] > 0 and got[3][2] is None
    print(f"# time {time.time() - t0:.1f} s; (crossings, overlaps, folded) = {got}")


# --- B, C -------------------------------------------------------------------------------------------------
def test_raw_unchanged(p4g):
    for lat, pair in [("square", SQ_PAIR), (p4g, P4G_PAIR)]:
        c = pt.build(lat, *pair, picture="raw")
        ref = pt.build(lat, *pair, picture="raw", fast=False)
        assert "drawing" not in c.defect.info
        assert len(c.defect.sites) == len(ref.defect.sites)
        assert _edges(c.defect) == _edges(ref.defect)
        assert np.allclose(c.defect.positions, ref.defect.positions, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("which", ["square", "p4g", "dislocation"])
def test_clean_relaxed_unchanged(which, p4g):
    L, specs = {"square": ("square", SQ_PAIR), "p4g": (p4g, P4G_PAIR),
                "dislocation": ("square", [dict(kind="dislocation", b=(1, 0), at=(0.37, 0.41))])}[which]
    c = pt.build(L, *specs, picture="relaxed")
    ref = copy.deepcopy(pt.build(L, *specs, picture="raw").defect)
    fast_relax.relax_picture(ref)
    assert np.array_equal(c.defect.positions, ref.positions)
    assert c.defect.info["drawing"]["stage"] == "given"
    m = measure(c)
    assert m["plain_crossings"] == 0 and m["folded"] == 0


# --- D ----------------------------------------------------------------------------------------------------
FAILING = {
    "chiral_square +90/+90": ("chiral_square", [D(90, (0, 0)), D(90, (1, 0))], None),
    "honeycomb +180/-180 3c": ("honeycomb", [D(180, (-1.453727223886262e-16, 0.49999999999999983)),
                                             D(-180, (1.732050807568877, 0.49999999999999983))], None),
    "zigzag +180 with b=(2,0)": ("zigzag", [D(180, (0.0, 0.5)), dict(kind="dislocation", b=(2.0, 0.0), at=(-0.826, 3.64))], None),
    "chiral_square +90/+90 at 2R0": ("chiral_square", [D(90, (0, 0)), D(90, (1, 0))], 13.051163197408322),
    "square -180 x3": ("square", [D(-180, (0, 0.5)), D(-180, (3, 0.5)), D(-180, (0, 3.5))], None),
}
SLOW = {"chiral_square +90/+90 at 2R0", "square -180 x3"}


@pytest.mark.parametrize("name", [pytest.param(k, marks=pytest.mark.slow) if k in SLOW else k for k in FAILING])
def test_failing_relaxed_now_embedded(name):
    L, specs, R = FAILING[name]
    t0 = time.time()
    c = pt.build(L, *specs, R=R, picture="relaxed")
    t1 = time.time()
    m = measure(c)
    info = c.defect.info["drawing"]
    assert info["stage"] != "given"                       # it was redrawn (it was not embedded)
    assert m["plain_crossings"] == 0 and m["folded"] == 0 and m["overlaps"] == 0
    assert m["crossings"] == 0 and m["ghost_bonds"] == 0  # no bond hidden as a stub
    assert _edges(c.defect) == _edges(pt.build(L, *specs, R=R, picture="raw").defect)
    print(f"# time {name}: build+draw {t1 - t0:.1f} s, stage {info['stage']} "
          f"{ {k: info.get(k) for k in ('radius', 'free', 'steps')} }, {len(c.defect.sites)} sites")


# --- E ----------------------------------------------------------------------------------------------------
def test_wall_entry_embedded():
    """p6m lattices[3] Wd2 (glide-breaking wall): the rigid picture crosses near the wall; entry.picture()
    is embedded, has the same bonds, and away from the wall the sites keep their rigid places (the two
    domains stay visible); the entry's stored data and a fresh build are unchanged."""
    t0 = time.time()
    cat = DefectCatalogue(pt.load_catalogue()["p6m"].lattices[3])
    e = next(x for x in cat.glide_breaking if x["id"] == "Wd2")
    rigid = e.build(R=cat.R0)
    m0 = measure(rigid)
    assert m0["plain_crossings"] > 0                      # negative control: the rigid picture crosses
    d, info = e.picture(R=cat.R0)
    m = measure(d)
    assert m["plain_crossings"] == 0 and m["overlaps"] == 0 and m["stub_crossings"] == 0
    # without sliding the two sides (which would misstate the slip), the local repair leaves at most one crowded site,
    # and the picture must then say so
    assert m["crowded"] == 0 or (m["crowded"] <= 1 and "crowded" in (info or {}).get("note", ""))
    assert _edges(d) == _edges(rigid) and len(d.sites) == len(rigid.sites)
    E = np.array([k for k in rigid.edges if not rigid.edge[k].get("across")])   # bonds inside a domain
    ln = lambda X: np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    rel = np.abs(ln(d.positions) / ln(rigid.positions) - 1)
    assert np.median(rel) < 0.05                          # the domains keep their lattice (measured 0.038 with the local repair, no slide)
    assert np.array_equal(e.build(R=cat.R0).positions, rigid.positions)
    print(f"# time {time.time() - t0:.1f} s; rigid crossings {m0['plain_crossings']}, overlaps {m0['overlaps']}; "
          f"stage {info['stage']}; in-domain bond length change: median {np.median(rel):.3g}, max {rel.max():.3g}")


def test_mobius_keeps_stubs_and_plot_smoke():
    import matplotlib
    matplotlib.use("Agg")
    cat = DefectCatalogue(pt.named("square"))
    e = cat.mobius_seams[0]
    d, info = e.picture()
    assert info is None and np.array_equal(d.positions, e.build().positions)
    assert any(x.get("ghost") is not None for x in d.edge.values())
    w = next(x for x in cat.glide_breaking if x["id"] == "Wd1")
    assert measure(w.picture()[0])["overlaps"] == 0 and measure(w.build())["overlaps"] > 0
    fig = cat.show("glide_breaking", R=3.0)
    assert fig is not None


def _clean_or_rigid_with_note(e, R):
    """entry.picture(R) is clean (no crossing, overlap, crowded site), or it is the rigid picture itself with a
    note in the plot title saying what it keeps. Returns (stage, measure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, info = e.picture(R=R)
    m = measure(d)
    if info["stage"] == "rigid":
        assert np.array_equal(d.positions, e.build(R=R).positions)
        assert info["note"].startswith("rigid picture") and str(info["check"]["crowded"] or info["check"]["crossings"]) in info["note"]
        assert info["check"]["crossings"] == m["plain_crossings"] and info["check"]["crowded"] == m["crowded"]
        ax = e.plot(ax=plt.subplots()[1], R=R)
        assert ax.get_title().split("\n")[-1] == info["note"]
        plt.close("all")
    else:
        assert info["note"] == ""
        assert m["plain_crossings"] == 0 and m["stub_crossings"] == 0 and m["overlaps"] == 0 and m["crowded"] == 0
    return info["stage"], m


def test_straight_p1_one_wall():
    """straight_p1 Wb16 (its local repair was crushed on 2026-11-14: 23 overlaps, closest pair 0.048): now clean
    or rigid with a note in the title (measured 2026-11-15: rigid, 42 crossings, 28 crowded sites)."""
    t0 = time.time()
    cat = DefectCatalogue(pt.named("straight_p1"))
    e = next(x for x in cat.twin_walls if x["id"] == "Wb16")
    stage, m = _clean_or_rigid_with_note(e, cat.R0)
    print(f"# time {time.time() - t0:.1f} s; Wb16 stage {stage}, crossings {m['plain_crossings']}, crowded {m['crowded']}")


@pytest.mark.slow
def test_straight_p1_twin_walls():
    """Every straight_p1 twin wall at R0: clean, or the rigid picture with a note in the title."""
    cat = DefectCatalogue(pt.named("straight_p1"))
    got = {e["id"]: _clean_or_rigid_with_note(e, cat.R0)[0] for e in cat.twin_walls}
    print("# stages:", got)


# --- F ----------------------------------------------------------------------------------------------------
def test_p4g_wd1_not_crowded(p4g):
    """The artefact of 2026-11-14 (wall sites stacked along the wall line, invisible to crossings and overlaps):
    the rigid picture counts crowded sites (negative control); entry.picture() has none, no overlap, no
    crossing, and both domains keep their lattice exactly (side B is moved rigidly)."""
    t0 = time.time()
    cat = DefectCatalogue(p4g)
    e = next(x for x in cat.glide_breaking if x["id"] == "Wd1")
    rigid = e.build(R=cat.R0)
    m0 = measure(rigid)
    assert m0["crowded"] > 0 and m0["overlaps"] > 0
    d, info = e.picture(R=cat.R0)
    m = measure(d)
    assert info["stage"] == "opened" and info["note"] == ""
    assert m["crowded"] == 0 and m["crowded_pairs"] == 0 and m["overlaps"] == 0 and m["plain_crossings"] == 0
    X, Y = np.asarray(rigid.positions), np.asarray(d.positions)
    E = np.array([k for k in rigid.edges if not rigid.edge[k].get("across")])
    assert np.allclose(np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1), np.linalg.norm(Y[E[:, 0]] - Y[E[:, 1]], axis=1))
    unit = m["crowd_tol"] / 0.35
    dd = np.linalg.norm(Y[:, None] - Y[None], axis=2) + 1e9 * np.eye(len(Y))
    print(f"# time {time.time() - t0:.1f} s; rigid crowded {m0['crowded']}, overlaps {m0['overlaps']}; "
          f"picture gap {info.get('gap')}, slide {info.get('slide')}, closest pair {dd.min() / unit:.3f} bond units")
    assert dd.min() > 0.35 * unit


def test_configuration_title_note():
    """A configuration whose drawing keeps an overlap (made here: one site moved onto the middle of a bond it is
    not an end of, then ensure_embedded(overlaps=False), the rule of build) says so in its plot title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pt.build("square", dict(kind="dislocation", b=(1, 0), at=(0.37, 0.41)), picture="relaxed")
    ax = c.plot(ax=plt.subplots()[1])
    assert "\n" not in ax.get_title()                     # clean picture: no note
    d = c.defect
    X = np.array(d.positions, float)
    k = int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))
    u, v = next((u, v) for (u, v) in d.edges if k not in (u, v) and np.linalg.norm(X[u] - X[k]) < 1.5
                and np.linalg.norm(X[v] - X[k]) < 1.5)
    X[k] = 0.5 * (X[u] + X[v])
    d.positions, d._nbr = X, None
    info = draw.ensure_embedded(d, overlaps=False)
    assert info["stage"] == "given" and d.info["drawing"]["check"]["overlaps"] > 0
    ax = c.plot(ax=plt.subplots()[1])
    note = ax.get_title().split("\n")[-1]
    assert "overlaps" in note and str(d.info["drawing"]["check"]["overlaps"]) in note
    plt.close("all")


def test_show_titles_wrapped():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cat = DefectCatalogue(pt.named("square"))
    fig = cat.show("glide_breaking")
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    labels = [e.get("label", e["id"]) for e in cat.glide_breaking]
    assert max(len(x) for x in labels) > 45                   # something to wrap
    assert len(titles) == len(labels) and all(len(line) <= 45 for t in titles for line in t.split("\n"))
    plt.close(fig)


def test_nonplanar_lattice_kept(capsys):
    """Negative control: the square lattice with both diagonals (bonds cross in the perfect lattice; graph not
    planar) keeps its given picture: there is no planar picture to draw."""
    import dataclasses
    from papertilling.legacy.lab_defects import dislocation
    from papertilling.legacy.wallpaper_lattices import make_cell
    sq = make_cell("square")
    x = dataclasses.replace(sq, bonds=list(sq.bonds) + [(0, 0, 1, 1, 1), (0, 0, 1, -1, 1)])
    d = dislocation(x, (1.0, 0.0), core=(0.37, 0.41), R=4.0)
    assert planarity.check(d)["crossings"] > 0
    e, info = draw.embedded_copy(d)
    assert info["stage"] == "given" and info["three_connected"] == "no: not planar"
    assert np.array_equal(e.positions, d.positions)
    c = pt.build(x, dict(kind="dislocation", b=(1, 0), at=(0.37, 0.41)), R=7, picture="raw")
    assert c.inspect()["total_frank"] is None
    assert "unavailable from planar faces" in capsys.readouterr().out
