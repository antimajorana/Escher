"""Equivalence of defect configurations (author rule, 2026-11-13): two configurations are the same
when they have the same total charge (class in G~+) AND isomorphic core graphs, "in the same way a
+pi/2 vertex disclination and a -pi/2 face disclination equal a dislocation on the square lattice".

  equivalent_single(config)  the single defect with the same total charge, and whether the
                             configuration's core graph is a core of that single defect
  compare(c1, c2)            same charge / same charge up to a mirror / same core graph
  classify(configs)          equivalence classes, printed as a table

Core graph: Configuration.core_graph (graph-only rule, see build.py). Charges: build.classify_charge.
"""
from __future__ import annotations

import string
import warnings

import networkx as nx
import numpy as np

from ..legacy import fusion_framework as ff
from . import fast as _fast
from .build import (Configuration, build, graph_core, isomorphic_cores, lattice_data, patch_rim, same_charge,
                    _norm)

_DISLOC_CACHE: dict = {}
_DISLOC_BUILDS: dict = {}               # same keys: the legacy builds behind each core


def dislocation_core_list(lattice, b, radius: int = 2, grid: int = 3, max_candidates: int = 8,
                          fast: bool | None = None, R: float | None = None) -> tuple[list[dict], dict]:
    """Every core of lab_defects.dislocation_cores for Burgers vector b (Volterra cut-and-reglue builds
    over grid x grid cut ends and up to six cut directions, plus every non-crossing completion of the
    core ring), re-read with the graph-only core rule of build.py. Only builds whose measured Burgers
    vector is in the P+ class of b are kept (dislocation_cores scans both cut senses, giving -b too,
    which is another class on p3, p3m1, ...). Returns (cores, scan info). The scan is finite
    (grid, six cuts, at most max_candidates completion bonds): a core it misses is not excluded.
    If a core ball reaches the patch rim, the whole scan is redone once with a 1.4x larger patch;
    candidates still too close are counted in info["skipped_rim"], never silently dropped.
    fast=None/True: fast.dislocation_cores (legacy code, faster primitives, identical output); False: legacy.
    R: patch radius of the builds (default None: the radius above). The legacy builds behind each core
    (Defect, completion bonds) are kept apart for cores.core_gallery (_disloc_builds), not in the output."""
    fast = _fast.enabled(fast)
    lat = lattice_data(lattice, fast)
    b = np.asarray(b, float)
    key = _disloc_key(lat, b, radius, grid, max_candidates, fast, R)
    if key in _DISLOC_CACHE:
        return _DISLOC_CACHE[key]
    if R is None:
        R = (2 * lat.r_det + radius + 3) * 0.5 * (lat.lmean + lat.lmax) + 0.5 * lat.scale + _norm(b)
    I = np.eye(2)
    for attempt in range(2):
        with warnings.catch_warnings():          # networkx 3.5 notice about unlabelled WL hashes (legacy code)
            warnings.simplefilter("ignore", UserWarning)
            found, notes = _fast.dislocation_cores(lat.cell, b, fast=fast, grid=grid, R=R, ball=radius, complete=True,
                                                   reconstruct=True, max_candidates=max_candidates)
        out, skipped, other_class = [], 0, 0
        builds, merged = [], []                  # aligned with out; (dropped lab record, index in out it equals)
        for f in found:
            bm = f["b_measured"]
            if bm is None or not ff.same_proper_holonomy((I, -np.asarray(bm, float)), (I, -b), lat.ops_plus, lat.A, lat.Ainv):
                other_class += 1
                continue
            x = f["defect"]
            G = nx.Graph()
            G.add_nodes_from(range(len(x.sites)))
            G.add_edges_from(x.edges)
            G.add_edges_from(f["chords"])
            keys = list(x.sites)
            rim = patch_rim(lat, [lat.L.pos(k) for k in keys], [k[2] for k in keys], [G.degree(i) for i in G],
                            f["core_point"], R)
            try:
                _, H = graph_core(G, rim, lat, radius)
            except ValueError:
                skipped += 1
                continue
            same = [k for k, o in enumerate(out) if isomorphic_cores(H, o["graph"])]
            if same:
                merged.append(dict(name=f"{'Volterra' if f['volterra'] else 'completion'} {f['name']}",
                                   volterra=f["volterra"], equals=same[0], chords=list(f["chords"])))
                continue
            builds.append(dict(defect=x, chords=list(f["chords"]), lab_name=f["name"], count=f.get("count"), R=R))
            out.append(dict(name=f"{'Volterra' if f['volterra'] else 'completion'} {f['name']}", volterra=f["volterra"],
                            rings=f["rings"], under=f["under"], added=f["added"], lattice_bonds=f["lattice_bonds"],
                            reconstruction_bonds=f["reconstruction_bonds"], bipartite=nx.is_bipartite(H), graph=H,
                            core_point=f["core_point"], cut=f["cut"]))
        if skipped == 0:
            break
        R *= 1.4
    info = dict(grid=grid, R=R, builds=notes["builds"], anomalies=notes["anomalies"], capped=notes["capped"],
                max_candidates=max_candidates, lab_cores=len(found), other_class=other_class, skipped_rim=skipped,
                distinct=len(out))
    _DISLOC_CACHE[key] = (out, info)
    _DISLOC_BUILDS[key] = dict(builds=builds, merged=merged, perfect_ring_sizes=notes.get("perfect_ring_sizes", []),
                               R=R, b=b)
    return out, info


def _disloc_key(lat, b, radius, grid, max_candidates, fast, R=None):
    k = (id(lat), tuple(np.round(np.asarray(b, float), 6)), radius, grid, max_candidates, fast)
    return k if R is None else k + (round(float(R), 6),)


def _disloc_builds(lattice, b, radius: int = 2, grid: int = 3, max_candidates: int = 8, fast: bool | None = None,
                   R: float | None = None) -> tuple[list[dict], dict, dict]:
    """(cores, info, store) of dislocation_core_list; store: the legacy builds aligned with cores
    (defect: the cut-and-reglue Defect, chords: its completion bonds), the scan records dropped as
    isomorphic to an earlier core ('merged'), the perfect ring sizes and the patch radius."""
    fast = _fast.enabled(fast)
    lat = lattice_data(lattice, fast)
    cands, info = dislocation_core_list(lat, b, radius=radius, grid=grid, max_candidates=max_candidates, fast=fast, R=R)
    return cands, info, _DISLOC_BUILDS[_disloc_key(lat, b, radius, grid, max_candidates, fast, R)]


def equivalent_single(config: Configuration, radius: int = 2, grid: int = 3, show: bool = False, fast: bool | None = None) -> dict:
    """The single defect with the same total charge, and which of its cores (if any) has a core graph
    isomorphic to the configuration's.
    Dislocation charge: every core of lab_defects.dislocation_cores (Volterra + completions).
    Disclination charge: the single disclination at the measured centre (its core does not depend on
    the seam direction, C05 step 3). fast: as in build.build (identical results, tests/test_config.py).
    res['gallery']: dislocation charge, cores.core_gallery of the charge (the same cores as the
    candidates, undrawn until plotted); otherwise []. show=True also draws the configuration's own core
    next to the gallery, the matching cores framed (res['figure']): every Volterra core, and, when the
    charge has completion (healed) cores too, the matching ones and others up to COMPLETION_PANELS (all of
    them, at most 30 panels, when nothing matches); a charge with Volterra cores only shows those only."""
    lat = config.lattice
    q = config.total_charge()
    H = config.core_graph(radius)
    res = dict(charge=q["label"], kind=q["kind"], single=q["single"], candidates=[], matches=[], equivalent=False,
               gallery=[])
    if q["kind"] == "dislocation":
        cands, info = dislocation_core_list(lat, q["b"], radius=radius, grid=grid, fast=fast)
        from .cores import core_gallery
        res["gallery"] = core_gallery(lat, q["b"], radius=radius, grid=grid, draw=False, fast=fast)
        res["scan"] = info
        res["candidates"] = [dict((k, v) for k, v in c.items() if k != "graph") for c in cands]
        res["matches"] = [c["name"] for c in cands if isomorphic_cores(H, c["graph"])]
        if not res["matches"] and len(config.specs) == 1 and config.specs[0].get("kind") == "dislocation":
            # one dislocation IS a single defect; its cut end is simply not one of the scanned positions
            res["matches"] = ["its own core (the configuration is one dislocation; its cut end is not among the scanned ones)"]
    elif q["kind"] == "disclination" and q["single"] is not None:
        s = q["single"]
        single = build(lat, dict(kind="disclination", angle=s["angle"], at=s["at"]), picture="raw", fast=fast)
        ok = isomorphic_cores(H, single.core_graph(radius))
        res["candidates"] = [dict(name=f"single {s['angle']:+g} deg at {np.round(s['at'], 4).tolist()}")]
        res["matches"] = [res["candidates"][0]["name"]] if ok else []
    elif q["kind"] == "vacuum":
        res["matches"] = ["perfect lattice"] if H.number_of_nodes() == 0 else []
    res["equivalent"] = bool(res["matches"])
    if res["equivalent"]:
        res["verdict"] = f"equivalent to the single defect (core: {', '.join(res['matches'])})"
    elif "scan" in res:
        sc = res["scan"]
        res["verdict"] = (f"no match among {sc['distinct']} distinct cores scanned (grid={sc['grid']}, {sc['builds']} "
                          f"Volterra builds, completions capped {sc['capped']} times, {sc['skipped_rim']} skipped at the rim); "
                          f"a larger grid may find more cores")
    else:
        res["verdict"] = "not equivalent to the single defect"
    if show:
        print(f"total charge: {q['label']}")
        for c in res["candidates"]:
            tag = "MATCH" if c["name"] in res["matches"] else "     "
            extra = f" rings {c['rings']}, bipartite {c['bipartite']}" if "rings" in c else ""
            print(f"  {tag} {c['name']}{extra}")
        print(res["verdict"])
        if res["gallery"]:
            res["figure"] = _show_gallery(config, res, radius)
    return res


COMPLETION_PANELS = 5                    # completion cores drawn by equivalent_single(show=True) when a core matches


def _show_gallery(config: Configuration, res: dict, radius: int):
    """The configuration's own core, then every Volterra core, then the completion cores (when the charge has
    any): the matching ones first, filled up to 5 completion panels (all of them, up to 30 panels in all,
    when nothing matches); matching cores framed."""
    import matplotlib
    import matplotlib.pyplot as plt

    from .cores import config_entry, plot_core_gallery
    gal = res["gallery"]
    vol = [e for e in gal if e["kind"] == "Volterra"]
    comp = [e for e in gal if e["kind"] != "Volterra"]
    hit = [e for e in comp if e["name"] in res["matches"]]
    cap = max(len(hit), COMPLETION_PANELS) if res["matches"] else len(comp)
    shown = vol + (hit + [e for e in comp if e not in hit])[:cap]
    fig = plot_core_gallery(shown, highlight=res["matches"], extra=[config_entry(config, radius)], max_panels=30,
                            title=f"{res['charge']}: the configuration's core and the cores of the single dislocation")
    if len(shown) < len(gal):
        print(f"(figure: {len(vol)} Volterra core(s) and {len(shown) - len(vol)} of {len(comp)} completion core(s); "
              f"res['gallery'] has them all, cores.plot_core_gallery draws any of them)")
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()
    return fig


def _same_lattice(c1: Configuration, c2: Configuration):
    if c1.lattice is not c2.lattice:
        raise ValueError("the two configurations are on different lattices")


def compare(c1: Configuration, c2: Configuration, radius: int = 2) -> dict:
    """dict(same_charge, same_charge_up_to_mirror, same_core, details)."""
    _same_lattice(c1, c2)
    lat = c1.lattice
    q1, q2 = c1.total_charge(), c2.total_charge()
    H1, H2 = c1.core_graph(radius), c2.core_graph(radius)
    return dict(same_charge=same_charge(lat, q1, q2), same_charge_up_to_mirror=same_charge(lat, q1, q2, mirror=True),
                same_core=isomorphic_cores(H1, H2),
                details=dict(charge1=q1["label"], charge2=q2["label"],
                             core1=(H1.number_of_nodes(), H1.number_of_edges()),
                             core2=(H2.number_of_nodes(), H2.number_of_edges())))


def classify(configs, radius: int = 2, names=None, show: bool = True) -> list[int]:
    """Group configurations into classes (same charge AND isomorphic core graphs). Both relations
    are equivalence relations, so comparing with one representative per class is enough."""
    configs = list(configs)
    if not configs:
        return []
    for c in configs[1:]:
        _same_lattice(configs[0], c)
    names = list(names) if names is not None else [f"#{k + 1}" for k in range(len(configs))]
    lat = configs[0].lattice
    reps, labels = [], []
    for i, c in enumerate(configs):
        for k, j in enumerate(reps):
            if same_charge(lat, c.total_charge(), configs[j].total_charge()) and \
                    isomorphic_cores(c.core_graph(radius), configs[j].core_graph(radius)):
                labels.append(k)
                break
        else:
            reps.append(i)
            labels.append(len(reps) - 1)
    if show:
        w = max(len(n) for n in names + ["configuration"])
        print(f"{'configuration':{w}s}  class  core (nodes, bonds)  total charge")
        for i, c in enumerate(configs):
            H = c.core_graph(radius)
            print(f"{names[i]:{w}s}  {string.ascii_uppercase[labels[i] % 26]:5s}  "
                  f"{str((H.number_of_nodes(), H.number_of_edges())):19s}  {c.total_charge()['label']}")
    return labels
