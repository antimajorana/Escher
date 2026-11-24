"""Gallery of the cores of one dislocation: every distinct core of config.dislocation_core_list, drawn.

  core_gallery(lattice, b)        one entry per distinct core (the same list, names and order as
                                  dislocation_core_list), each with a drawable crystal
  plot_core_gallery(gallery)      one figure: Volterra cores in the first row(s), completion cores in the
                                  next; completion bonds in a distinct colour and style
  config_entry(config)            the same kind of entry for a configuration's own core (equivalent_single)

Two routes to a core (definitions, lab_defects.dislocation_cores):
  Volterra    cut and reglue only; the core depends on where the cut ends in the cell and on its direction.
  completion  a Volterra core plus new bonds ('chords') between under-coordinated sites of its core ring,
              not crossing each other and giving no site more bonds than in the perfect lattice.
A completion whose core graph equals a Volterra core's is the same core: it is listed once, as the Volterra
core, and entry['also'] says by which completions it was reached as well. That list is a lower bound: the
scan merges such records silently (it keeps a count only), so they are re-found here by enumerating the
completions of each listed Volterra build (the same rules as the scan), plus the records that
dislocation_core_list dropped as isomorphic to a Volterra core.

Each entry's crystal is a crop of the legacy build (sites within D = 2 r_det + radius + 3 bonds of the core
ring; deep copies, the cached builds are never changed) with the chords added as real bonds. A chord u-v gets
the bond vector 'vec' (which fixes its place in the order of bonds round u and v, hence the faces) of one of
the two arcs of the core ring from u to v (turned by up to 30 degrees when needed; when no straight
direction fits, separate angles 'ang' at its two ends), the first for which the faces of the crop are the
expected ones (the core ring replaced by the rings the chords cut it into: checked for every chord and
again on all faces of every entry, entry['faces_ok']); its length is kept when it is a lattice bond length,
else set to the mean bond length (the rest length of the drawing springs). The picture (draw_entry, on first plot) is the standard relaxation (fast_relax)
followed by draw.ensure_embedded; entry['drawing'] keeps draw's info.
"""
from __future__ import annotations

import copy
import itertools
import math
import warnings
from collections import Counter

import networkx as nx
import numpy as np

from ..legacy import lab_defects as _ld
from . import fast as _fast
from .build import graph_core, isomorphic_cores, lattice_data, patch_rim

RING_NAMES = {3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon", 9: "nonagon",
              10: "decagon", 11: "hendecagon", 12: "dodecagon"}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
_GALLERY_CACHE: dict = {}             # (id of the dislocation_core_list store, both_routes) -> entries
CHORD_COLOUR, CHORD_STYLE = "#d81b60", (0, (3.0, 1.6))
ODD_FACE, SPLIT_FACE, CONFIG_FACE, FRAME = "#f2a33a", "#f7d9a8", "#8fb8e8", "#2e7d32"


# --- words -----------------------------------------------------------------------------------------------------
def _ring_word(n: int, count: int) -> str:
    name = RING_NAMES.get(n, f"{n}-ring")
    return f"{WORDS.get(count, count)} {name}{'s' if count > 1 else ''}"


def describe(rings, perfect, under=()) -> str:
    """Short words for a core: its rings whose size does not occur in the perfect lattice first ("5-7 pair"
    for one pentagon and one heptagon), then the perfect-size rings, then the coordination."""
    perfect = set(perfect)
    odd = Counter(r for r in rings if r not in perfect)
    rest = Counter(r for r in rings if r in perfect)
    if sorted(odd.elements()) == [5, 7]:
        head = "5-7 pair"
    elif odd:
        head = " and ".join(_ring_word(n, c) for n, c in sorted(odd.items()))
    else:
        head = " and ".join(_ring_word(n, c) for n, c in sorted(rest.items())) + " (ring sizes of the perfect lattice)"
    if rest and odd:
        head += ", plus " + " and ".join(_ring_word(n, c) for n, c in sorted(rest.items()))
    under = list(under)
    if under:
        k = len(under)
        head += f"; {WORDS.get(k, k)} under-coordinated site{'s' if k > 1 else ''} on the ring"
    else:
        head += "; every ring site fully bonded"
    return head


# --- crops of legacy Defects --------------------------------------------------------------------------------
def _crop(x, keep):
    """A new legacy Defect with the sites `keep` (old indices, sorted) of x: deep-copied bond data, info arrays
    of per-site length cut down, drawing and charge caches dropped. Returns (defect, old -> new index)."""
    keep = sorted(keep)
    new = {o: i for i, o in enumerate(keep)}
    edge = {(new[u], new[v]): {k: (w.copy() if isinstance(w, np.ndarray) else w) for k, w in e.items()}
            for (u, v), e in x.edge.items() if u in new and v in new}             # arrays copied; scalars, tuples shared
    n = len(x.sites)
    info = {}
    for k, val in x.info.items():
        if k in ("_charge", "drawing", "picture", "strain"):
            continue
        if isinstance(val, (list, np.ndarray)) and len(val) == n and k != "b":
            val = [val[o] for o in keep] if isinstance(val, list) else np.asarray(val)[keep]
        info[k] = copy.deepcopy(val)
    for e in edge.values():
        e.pop("ghost", None)
    y = _ld.Defect(x.kind, [x.sites[o] for o in keep], np.asarray(x.positions, float)[keep], edge, info)
    return y, new


def _graph_ball(G, sources, D):
    return nx.multi_source_dijkstra_path_length(G, set(sources), cutoff=D)


# --- chords ---------------------------------------------------------------------------------------------------
def _first_pos(ring):
    pos = {}
    for k, v in enumerate(ring):
        pos.setdefault(v, k)
    return pos


def _arc_vec(d, ring, pos, u, v):
    """Sum of the bond vectors along the ring walk from (the first visit of) u forward to v."""
    n = len(ring)
    k, s = pos[u], np.zeros(2)
    while ring[k % n] != v or k == pos[u]:
        s = s + np.asarray(d.edge[(ring[k % n], ring[(k + 1) % n])]["vec"], float)
        k += 1
        if k - pos[u] > n:
            raise ValueError("chord end not on the ring")
    return s


TURNS = (0.0, 15.0, -15.0, 30.0, -30.0)       # degrees: chord directions tried round each arc direction
TWO_PI = 2 * math.pi


def _corner_angle(e):
    """The angle that orders a bond round its site (as fast._corners_fast reads it)."""
    if "ang" in e:
        return e["ang"]
    return math.atan2(float(e["vec"][1]), float(e["vec"][0])) % TWO_PI


def _chord_options(d, nb, ring, u, v, lat):
    """Candidate bond data (forward, backward) of a chord u -> v. First 'vec' only: the two arc vectors of
    the ring (they differ by the ring's Burgers vector), each also turned by TURNS; a direction that ties
    with a bond already at u (or, reversed, at v) is skipped (the order round the site would rest on a tie
    and the face between them would have zero area: kagome, a chord along a straight lattice line). Then,
    for a ring whose developed shape admits no straight chord, separate angles at the two ends ('ang': the
    middle of each gap between the bonds at u and at v; the order of bonds round a site is read from 'ang'
    when present, fast._corners_fast), 'vec' along the angle at u, of the length above."""
    pos = _first_pos(ring)
    arc = _arc_vec(d, ring, pos, u, v)
    tot = sum(np.asarray(d.edge[(ring[k], ring[(k + 1) % len(ring)])]["vec"], float) for k in range(len(ring)))
    at_u = [_corner_angle(d.edge[(u, w)]) for w in nb[u]]
    at_v = [_corner_angle(d.edge[(v, w)]) for w in nb[v]]
    tie = lambda t, angs: any(abs(math.remainder(t - x, TWO_PI)) < 1e-3 for x in angs)
    out, first = [], None
    for c in (arc, arc - tot):
        L = float(np.linalg.norm(c))
        if L < 1e-9:
            continue
        L2 = L if L <= 1.05 * lat.lmax else lat.lmean
        for turn in TURNS:
            t = math.atan2(c[1], c[0]) + math.radians(turn)
            vec = L2 * np.array([math.cos(t), math.sin(t)])
            first = vec if first is None else first
            if not (tie(t, at_u) or tie(t + math.pi, at_v)):
                out.append((dict(vec=vec), dict(vec=-vec)))
    first = first if first is not None else arc

    def gaps(angs):
        A = sorted(x % TWO_PI for x in angs)
        return [((A[k] + ((A[(k + 1) % len(A)] - A[k]) % TWO_PI or TWO_PI) / 2) % TWO_PI) for k in range(len(A))] if A else [0.0]
    L2 = float(np.linalg.norm(first))
    for gu in gaps(at_u):
        vec = L2 * np.array([math.cos(gu), math.sin(gu)])     # along the chord's angle at u: the drawing's corner
        for gv in gaps(at_v):                                  # areas at u then open a face that is flat in 'vec'
            out.append((dict(vec=vec, ang=gu), dict(vec=-vec, ang=gv)))
    return out


def _face_from(d, nb, a, b):
    """The face walk of fast.frame_faces through the half-edge a -> b (local: only the sites it visits)."""
    order = {}

    def nbrs(w):
        if w not in order:
            order[w] = [x for _, x in sorted((_corner_angle(d.edge[(w, x)]), x) for x in nb[w])]
        return order[w]
    f, a0, b0 = [], a, b
    while True:
        f.append(a)
        lst = nbrs(b)
        a, b = b, lst[lst.index(a) - 1]
        if (a, b) == (a0, b0) or len(f) > 4 * len(d.sites):
            return f


def _face_keys(d):
    return Counter(tuple(sorted(f)) for f in _fast.frame_faces(d))


def _add_chords(y, ring, chords, lat):
    """Add the chords (new indices) to y one at a time, each with the first candidate (_chord_options) for
    which the two faces through the new bond are the two rings it cuts off (lab_defects._split of the ring
    by the chords added so far). Adding a bond to a rotation system changes only the face it splits, so the
    faces of y are then those of the crop with the ring replaced by the split rings; this is checked once
    more on all faces at the end. Returns (ok, split rings)."""
    pos = _first_pos(ring)
    sub2 = [tuple(sorted(c, key=lambda w: pos[w])) for c in chords]
    base = _face_keys(y)
    nb = {}
    for (p, q) in y.edge:
        nb.setdefault(p, []).append(q)
    ok = True
    for i, (u, v) in enumerate(sub2):
        split = _ld._split(ring, sub2[:i + 1], sizes_only=False)
        adj = lambda r: any({r[k], r[(k + 1) % len(r)]} == {u, v} for k in range(len(r)))
        want = sorted(tuple(sorted(r)) for r in split if adj(r))
        opts = _chord_options(y, nb, ring, u, v, lat)
        nb[u].append(v); nb[v].append(u)
        for fw, bw in opts:
            y.edge[(u, v)] = dict(fw, type=0, chord=True)
            y.edge[(v, u)] = dict(bw, type=0, chord=True)
            if sorted(tuple(sorted(_face_from(y, nb, *h))) for h in ((u, v), (v, u))) == want:
                break
        else:
            ok = False
            y.edge[(u, v)] = dict(opts[0][0], type=0, chord=True)
            y.edge[(v, u)] = dict(opts[0][1], type=0, chord=True)
    y.edges = sorted({(min(p, q), max(p, q)) for (p, q) in y.edge})
    y._nbr = None
    split = _ld._split(ring, sub2, sizes_only=False)
    want = base.copy()
    want[tuple(sorted(ring))] -= 1
    for r in split:
        want[tuple(sorted(r))] += 1
    return ok and _face_keys(y) == +want, split


# --- the gallery ------------------------------------------------------------------------------------------------
def _build_data(x, lat, R, core_point, memo):
    """Per legacy build (shared by the cores found on it): ring walks, graph, degrees, patch rim."""
    if id(x) not in memo:
        keys = list(x.sites)
        deg = x.degrees()
        rim = patch_rim(lat, [lat.L.pos(k) for k in keys], [k[2] for k in keys], [int(v) for v in deg], core_point, R)
        memo[id(x)] = dict(walks=[f for f, _ in x.charged_faces("vec")], G=x.to_networkx(), deg=deg, rim=rim, x=x)
    return memo[id(x)]


def _crop_build(x, ring_sites, chords, lat, radius, data):
    """Crop of the build x round its core ring, its rim (sites near the crop edge or the patch rim)."""
    G = data["G"].copy() if chords else data["G"]
    G.add_edges_from(chords)
    D = 2 * lat.r_det + radius + 3
    dist = _graph_ball(G, ring_sites, D)
    y, new = _crop(x, dist)
    deg, old_rim = data["deg"], data["rim"]
    ydeg = y.degrees()
    rim = {new[o] for o in dist if dist[o] >= D - 1 or o in old_rim or ydeg[new[o]] < deg[o]}
    return y, new, rim


def core_gallery(lattice, b, radius: int = 2, grid: int = 3, R: float | None = None, draw: bool = True,
                 both_routes: bool = True, max_candidates: int = 8, fast: bool | None = None) -> list[dict]:
    """Every distinct core of the single dislocation of Burgers vector b (config.dislocation_core_list, same
    names and order), each with a drawable crystal. Returns a list of dicts:
      name, kind ('Volterra' / 'completion'), defect (legacy Defect: a crop of the build round the core,
      completion chords added as real bonds, edge data chord=True), core_sites (sites of the core ring(s),
      crop indices), chords (crop index pairs; [] for Volterra), rings (face sizes at the core, as
      dislocation_core_list), faces (the core faces, lists of crop indices), description, under, added,
      core_point, cut, graph (the core graph of dislocation_core_list), also (Volterra: completions found
      to give the same core), faces_ok (the chords give the expected faces), iso_ok (the crop's graph core
      is isomorphic to `graph`; None if it could not be read), drawing (draw's info once drawn).
    draw=True: draw every entry now (draw_entry); False: on first plot (drawing costs ~0.05-0.4 s an entry).
    both_routes: look for completions equal to Volterra cores (see the module docstring). Cached: the same
    arguments return the same entry dicts (drawn in place once)."""
    from .config import _disloc_builds
    lat = lattice_data(lattice, fast)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cands, info, store = _disloc_builds(lat, b, radius=radius, grid=grid, max_candidates=max_candidates, fast=fast, R=R)
    key = (id(store), both_routes)
    if key in _GALLERY_CACHE:
        out = _GALLERY_CACHE[key]
        if draw:
            for e in out:
                draw_entry(e)
        return out
    perfect = store["perfect_ring_sizes"]
    out, memo = [], {}
    for c, src in zip(cands, store["builds"]):
        x = src["defect"]
        data = _build_data(x, lat, src["R"], c["core_point"], memo)
        rings_walks = data["walks"]
        ring_sites = {v for f in rings_walks for v in f}
        y, new, rim = _crop_build(x, ring_sites, src["chords"], lat, radius, data)
        walks = [[new[v] for v in f] for f in rings_walks]
        chords = [(new[u], new[v]) for u, v in src["chords"]]
        faces_ok, faces = True, walks
        if chords:
            faces_ok, faces = _add_chords(y, walks[0], chords, lat)
        try:
            iso = isomorphic_cores(graph_core(y.to_networkx(), rim, lat, radius)[1], c["graph"])
        except ValueError:
            iso = None
        kind = "Volterra" if c["volterra"] else "completion"
        out.append(dict(name=c["name"], kind=kind, defect=y, core_sites=sorted({v for f in walks for v in f}),
                        chords=chords, rings=tuple(c["rings"]), faces=faces, under=list(c["under"]), added=c["added"],
                        description=describe(c["rings"], perfect, c["under"]), core_point=c["core_point"], cut=c["cut"],
                        graph=c["graph"], also=[], faces_ok=faces_ok, iso_ok=iso, rim=rim, perfect_rings=list(perfect),
                        drawing=None, lattice=lat))
    for m in store["merged"]:                                   # records dropped as equal to an earlier core
        if not m["volterra"] and out[m["equals"]]["kind"] == "Volterra":
            out[m["equals"]]["also"].append(f"{m['name']} of the scan ({len(m['chords'])} bond(s) added)")
    if both_routes:
        _find_both_routes(out, lat, radius, max_candidates)
    _GALLERY_CACHE[key] = out
    if draw:
        for e in out:
            draw_entry(e)
    return out


def _completions(G, ring, under, max_candidates):
    """The scan's step 2 on one ring: every set of non-crossing new bonds between under-coordinated sites."""
    pos = _first_pos(ring)
    cand = [(u, v) for u, v in itertools.combinations(sorted(under), 2) if not G.has_edge(u, v)][:max_candidates]
    crosses = lambda p, q: len({p[0], p[1], q[0], q[1]}) == 4 and \
        ((pos[p[0]] < pos[q[0]] < pos[p[1]]) != (pos[p[0]] < pos[q[1]] < pos[p[1]]))
    for r in range(1, len(cand) + 1):
        for sub in itertools.combinations(cand, r):
            use = Counter(w for c in sub for w in c)
            if any(use[w] > under[w] for w in use):
                continue
            sub2 = [tuple(sorted(c, key=lambda w: pos[w])) for c in sub]
            if any(crosses(p, q) for p, q in itertools.combinations(sub2, 2)):
                continue
            yield sub


def _find_both_routes(out, lat, radius, max_candidates):
    vol = [e for e in out if e["kind"] == "Volterra"]
    for e in vol:
        y = e["defect"]
        if len(e["faces"]) != 1:
            continue
        ring = e["faces"][0]
        deg = y.degrees()
        under = {v: lat.perfect_degree[y.sites[v][2]] - int(deg[v]) for v in set(ring)
                 if deg[v] < lat.perfect_degree[y.sites[v][2]]}
        G = y.to_networkx()
        for sub in _completions(G, ring, under, max_candidates):
            G2 = G.copy()
            G2.add_edges_from(sub)
            try:
                H = graph_core(G2, e["rim"], lat, radius)[1]
            except ValueError:
                continue
            for f in vol:
                if isomorphic_cores(H, f["graph"]):
                    note = f"a completion of {e['name']} ({len(sub)} bond(s) added)"
                    if note not in f["also"]:
                        f["also"].append(note)


def draw_entry(e: dict) -> dict:
    """Draw an entry's crystal once (in place on its own crop): standard relaxation, then
    draw.ensure_embedded. Returns draw's info."""
    if e.get("drawing") is None:
        from . import draw as _draw
        from .fast_relax import relax_picture
        d = e["defect"]
        relax_picture(d)
        e["drawing"] = _draw.ensure_embedded(d)
    return e["drawing"]


# --- a configuration's own core ---------------------------------------------------------------------------------
def config_entry(config, radius: int = 2, title: str = "this configuration") -> dict:
    """A gallery-like entry for a Configuration: crop round its graph-only core sites, core faces = the
    faces of Configuration.cores()."""
    lat = config.lattice
    d = config.defect
    K, _ = config._get_core(radius)
    D = 2 * lat.r_det + radius + 3
    keep = _graph_ball(d.to_networkx(), K, D) if K else range(len(d.sites))
    y, new = _crop(d, keep)
    cores = config.cores()
    faces = [[new[v] for v in c["sites"]] for c in cores if len(c["sites"]) > 2 and all(v in new for v in c["sites"])]
    rings = tuple(sorted(c["ring"] for c in cores if c.get("ring")))
    words = ", ".join(c["label"] for c in cores) or ("no irregular sites" if not K else "no charged faces")
    return dict(name=title, kind="configuration", defect=y, core_sites=sorted(new[v] for v in K if v in new), chords=[],
                rings=rings, faces=faces, description=words, drawing=None, lattice=lat,
                perfect_rings=[], also=[])


# --- plotting -----------------------------------------------------------------------------------------------------
def _sublattice(label):
    """Sublattice index of a site label: (i, j, s) for the lab builds, ((i, j, s), chart) for build.build."""
    return label[0][2] if isinstance(label[0], tuple) else label[2]


def _draw_panel(ax, e, framed=False, zoom=True, zoom_bonds=2.5):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon, Rectangle
    draw_entry(e)
    d, lat = e["defect"], e["lattice"]
    P = np.asarray(d.positions, float)
    chord = {(min(u, v), max(u, v)) for u, v in e["chords"]}
    plain, ghost = [], []
    for u, v in d.edges:
        if (u, v) in chord:
            continue
        (ghost if d.edge[(u, v)].get("ghost") is not None else plain).append((u, v))
    cols = [_ld.PALETTE[int(d.edge[(u, v)].get("type", 0)) % len(_ld.PALETTE)] for u, v in plain]
    ax.add_collection(LineCollection([[P[u], P[v]] for u, v in plain], colors=cols, linewidths=1.0, zorder=2))
    for u, v in ghost:
        for a, q in ((u, d.edge[(u, v)]["ghost"]), (v, d.edge[(v, u)]["ghost"])):
            ax.plot([P[a][0], q[0]], [P[a][1], q[1]], color="#888", lw=0.9, ls=":", zorder=2)
    if chord:
        ax.add_collection(LineCollection([[P[u], P[v]] for u, v in chord], colors=CHORD_COLOUR, linewidths=2.3,
                                         linestyles=[CHORD_STYLE], zorder=3))
    perfect = set(e.get("perfect_rings") or [])
    for f in e["faces"]:
        col = CONFIG_FACE if e["kind"] == "configuration" else (SPLIT_FACE if len(f) in perfect else ODD_FACE)
        ax.add_patch(Polygon(P[f], closed=True, facecolor=col, edgecolor="none", alpha=0.75, zorder=1))
        c = P[f].mean(axis=0)
        ax.text(c[0], c[1], str(len(f)), fontsize=7, ha="center", va="center", zorder=4, color="#333")
    deg = d.degrees()
    low = [v for v in range(len(d.sites)) if deg[v] < lat.perfect_degree[_sublattice(d.sites[v])]]
    ax.scatter(P[:, 0], P[:, 1], s=5, c="k", zorder=5)
    if zoom and e["core_sites"]:
        Q = P[e["core_sites"]]
        c = Q.mean(axis=0)
        lens = [np.linalg.norm(P[u] - P[v]) for u, v in plain]
        unit = float(np.median(lens)) if lens else 1.0
        half = float(np.max(np.linalg.norm(Q - c, axis=1))) + zoom_bonds * unit
        low = [v for v in low if np.all(np.abs(P[v] - c) < half)]
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)
    if low:
        ax.scatter(P[low, 0], P[low, 1], s=34, facecolors="none", edgecolors="#c62828", linewidths=1.1, zorder=6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if framed:
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, lw=3.0, ec=FRAME, clip_on=False, zorder=10))


def _title(e):
    rings = ",".join(map(str, e["rings"])) if e["rings"] else "-"
    t = f"{e['name']}\nrings {rings}: {e['description']}"
    if e.get("also"):
        t += "\n(also reached as a completion)"
    c = (e.get("drawing") or {}).get("check") or {}
    bad = [f"{c[k]} {k}" for k in ("crossings", "folded", "overlaps", "crowded") if c.get(k)]
    if bad:
        t += f"\n[picture: {', '.join(bad)}]"
    return t


def _wrap(s, width=34):
    import textwrap
    return "\n".join(textwrap.fill(line, width) for line in s.split("\n"))


def plot_core_gallery(gallery, highlight=None, ax_size: float = 3.2, zoom: bool = True, extra=None,
                      max_panels: int = 30, ncols: int | None = None, title: str | None = None):
    """One figure: optional extra panels (e.g. config_entry) in the first row, then the Volterra cores, then
    the completion cores (each group starts a new row). Each panel is zoomed on the core, core faces shaded
    (orange: ring size not in the perfect lattice; light: perfect size), under-coordinated sites circled in
    red, completion bonds dashed in magenta. highlight: names to frame (the matches). At most max_panels
    gallery panels: highlighted ones first, then Volterra, then completion cores; the number left out is
    printed in the figure title. Returns the figure."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    highlight = set(highlight or [])
    extra = list(extra or [])
    keep = [e for e in gallery if e["name"] in highlight]
    for e in [e for e in gallery if e["kind"] == "Volterra"] + [e for e in gallery if e["kind"] != "Volterra"]:
        if len(keep) >= max_panels:
            break
        if e not in keep:
            keep.append(e)
    left = len(gallery) - len(keep)
    groups = [g for g in (extra, [e for e in gallery if e in keep and e["kind"] == "Volterra"],
                          [e for e in gallery if e in keep and e["kind"] != "Volterra"]) if g]
    ncols = ncols or max(1, min(5, max(len(g) for g in groups) if groups else 1))
    rows = sum(math.ceil(len(g) / ncols) for g in groups) or 1
    fig, axes = plt.subplots(rows, ncols, figsize=(max(ax_size * ncols, 7.0), (ax_size + 1.25) * rows + 0.9), squeeze=False)
    r = 0
    for g in groups:
        for k, e in enumerate(g):
            ax = axes[r + k // ncols][k % ncols]
            _draw_panel(ax, e, framed=e["name"] in highlight, zoom=zoom)
            ax.set_title(_wrap(_title(e)), fontsize=7.5)
        for k in range(len(g), math.ceil(len(g) / ncols) * ncols):
            axes[r + k // ncols][k % ncols].axis("off")
        r += math.ceil(len(g) / ncols)
    handles = [Line2D([], [], color=_ld.PALETTE[0], lw=1.2, label="bond of the crystal (colour: bond type)"),
               Line2D([], [], color=CHORD_COLOUR, lw=2.3, ls=CHORD_STYLE, label="completion bond (chord)"),
               Patch(color=ODD_FACE, label="core ring (size not in the perfect lattice)"),
               Patch(color=SPLIT_FACE, label="core ring of perfect size"),
               Line2D([], [], marker="o", ls="", mfc="none", mec="#c62828", label="under-coordinated site")]
    if extra:
        handles.append(Patch(color=CONFIG_FACE, label="core face of the configuration"))
    if highlight:
        handles.append(Patch(facecolor="none", edgecolor=FRAME, lw=2.5, label="same core as the configuration"))
    head = title or "Cores of the dislocation"
    if left:
        head += f"  ({left} more core(s) not shown: pass max_panels or select entries)"
    fig.suptitle(_wrap(head, int(fig.get_figwidth() * 12)), fontsize=10)
    fig.legend(handles=handles, loc="lower center", ncol=min(2 if ncols < 3 else 4, len(handles)), fontsize=7.5,
               frameon=False)
    fig.tight_layout(rect=(0, 0.75 / fig.get_figheight(), 1, 1 - 0.35 / fig.get_figheight()), h_pad=1.2)
    return fig
