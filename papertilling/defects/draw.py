"""Faithful planar pictures of a built crystal (drawing layer only: no graph datum is read from positions).

embed(d) returns (positions, info) and tries, in this order (planarity.check decides "embedded": no crossing,
fold, overlap or crowded site; with overlaps=False, the rule of build(picture="relaxed"), the given picture may
keep overlaps and crowded sites, and crowded sites are only counted (info['check']), never a reason to redraw
or to refuse a step: gating them made the hard builds 3-6 x slower, 2026-11-15, e.g. zigzag +180 with b=(2,0)
5.2 s -> 30.3 s, still crowded):
  'given'    the given positions, kept if they have no crossing, no fold (and, with overlaps=True, no overlap and
             no crowded site);
  'opened'   (walls of lab_defects.wall only: info 'sigma', sites labelled 'A' / 'B') the given picture with side
             B moved rigidly: away from side A along the wall normal by gap x the bond unit, gap = 0.5, 0.75, 1,
             and along the wall by slide x the median along-wall part of the bonds across the wall
             (u - sigma(u)), slide = 0 first (the slip of a glide wall stays visible), then 0.5, 1 (the across
             bonds turn towards the normal). Both domains stay perfect lattices; sites on or near the wall
             line and their images no longer stack. (Slides are skipped when the across bonds have no
             along-wall part, a twin wall.) If no (slide, gap) gives an embedded picture, a wall gets no further
             stage (wall_local=True adds 'relaxed' and the local repair, from the least bad opened picture),
             never the global Tutte stages (they do not keep the two domains recognisable): embedded_copy then
             returns the rigid picture. Measured 2026-11-15 on straight_p1, the local repair made 8 of 22
             twin walls clean at R0 (5-16 s each) and 3 of 22 at 2 R0 (44-72 s each), against 0.1-0.5 s for
             the rigid picture;
  'relaxed'  (relax_given=True, used for the rigid catalogue pictures) the standard relaxation (fast_relax
             energy: bond and corner springs, corner areas, weak anchor) from the given positions;
  'local'    local repair: Tutte (barycentric) positions inside the graph ball of radius k = 1, 2, 4, 8, 16
             about the bad sites (ends of crossing bonds, overlaps, folded faces), all other sites kept (free
             rim sites tied to their place by a spring), then the standard relaxation, a step being taken only
             if the picture stays embedded (relax_guarded); used while the ball holds at most 90% of the
             sites (half, when no rim is known);
  'tutte+relax'  global Tutte embedding: outer face (the longest frame_faces walk; for bonds without 'ang' /
             'vec', a face of a networkx planar embedding or the drawn rim) on a circle in its walk order by
             arc length, then the standard relaxation started there (anchor: the Tutte positions);
  'pinned'   the same relaxation with the outer face held at its Tutte positions;
  'guarded'  the relaxation from the Tutte embedding, outer face free, a step taken only if the picture gets no
             worse (relax_guarded); with 0 steps (info['steps']) it IS the Tutte embedding.
Bonds marked 'across' (a wall's u - sigma(u)) get the rest length max(drawn length, median length of their
type): a short one is pushed out to a normal bond, a long one (a glide wall, where u - sigma(u) runs along the
wall) is not shortened, which would slide the two sides along the wall onto each other (it stacked the sites
of p4g Wd1 and square Wd1, 2026-11-14). embedded_copy (the catalogue pictures) returns the rigid picture,
info['stage'] = 'rigid', when no stage is embedded; drawing_note(info) is the one-line title note saying what
a picture keeps. A graph that is not planar keeps the given picture. Each stage after 'given' is accepted only if embedded; if
none is, the least bad picture tried is returned (crossings and folds first, then overlaps) and info['check']
counts what remains. By Tutte's theorem the
Tutte embedding is planar with convex faces when the graph is 3-connected; info['three_connected'] says what
is known. Pieces Tutte would collapse are handled apart (tutte_solve): trees and rings hanging from one cut
site, groups held by only two fixed sites. A cone apex is an ordinary site here (frame_faces reads its
unrolled 'ang'). A picture made here draws every bond in full (no 'ghost' stubs: nothing crosses).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from . import fast_relax as _fr
from .planarity import (Faces, _components, _cross, _planar_outer_faces, bond_unit, check, edge_types,
                        frame_faces, has_frames)

# --- Tutte embeddings ------------------------------------------------------------------------------------------
def _neighbours(d, n):
    nbr = {v: set() for v in range(n)}
    for (u, v) in d.edge:
        nbr[u].add(v)
    return nbr


def _rest_length(d, u, v, X0):
    e = d.edge[(u, v)]
    return float(np.hypot(*e["vec"])) if "vec" in e else float(np.linalg.norm(X0[v] - X0[u]))


def _hanging_pieces(d, fixed):
    """Maximal sets of sites cut off from every fixed site by one site (pendant trees, bubbles at a cut site):
    [(cut site, sites)]. Tutte would put all of them on the cut site."""
    import networkx as nx
    G = d.to_networkx()
    fx = set(np.nonzero(fixed)[0].tolist())
    pieces = []
    for a in nx.articulation_points(G):
        H = G.subgraph(set(G) - {a})
        for comp in nx.connected_components(H):
            if not comp & fx and any(G.has_edge(a, v) for v in comp) and len(comp) < len(G) // 2 + 1:
                pieces.append((a, comp))
    pieces.sort(key=lambda p: -len(p[1]))
    taken, out = set(), []
    for a, comp in pieces:
        if a not in taken and not comp & taken:
            out.append((a, comp))
            taken |= comp
    return out


def tutte_solve(d, P, fixed, anchor=None):
    """Sites not in `fixed` (bool array) at the mean of their neighbours (uniform weights), the fixed ones at P.
    Pieces hanging from one cut site with no fixed site (pendant trees, small rings) are taken out first and put
    back afterwards: their shape in P, turned into the widest angular gap at the cut site and scaled to 0.45 x
    its shortest placed bond. Groups of free sites whose only fixed neighbours are two sites a, b (Tutte would
    put them on the segment a-b) keep their shape in P, moved by the similarity that takes a, b to their
    places. anchor: {site: weight} adds a spring of that weight from a free site to its place in P (a rim
    site of a local region, which has no fixed site beyond it). Returns new positions."""
    n = len(P)
    P0 = np.asarray(P, float)
    P = np.array(P0, copy=True)
    nbr = _neighbours(d, n)
    pieces = _hanging_pieces(d, fixed) if not fixed.all() else []
    gone = set().union(*[c for _, c in pieces]) if pieces else set()
    flaps = []                                                 # free groups held by two fixed sites only
    seen = set(gone)
    for v0 in range(n):
        if fixed[v0] or v0 in seen:
            continue
        comp, stack, att = {v0}, [v0], set()
        seen.add(v0)
        while stack:
            u = stack.pop()
            for w in nbr[u]:
                if fixed[w]:
                    att.add(w)
                elif w not in seen and w not in gone:
                    seen.add(w); comp.add(w); stack.append(w)
        if len(att) == 2:
            flaps.append((sorted(att), comp))
            gone |= comp
        elif not att and not (anchor and comp & set(anchor)):  # nothing holds it: it keeps its place
            gone |= comp
    free = [v for v in range(n) if not fixed[v] and v not in gone and nbr[v] - gone]
    idx = {v: k for k, v in enumerate(free)}
    if free:
        rows, cols, vals = [], [], []
        rhs = np.zeros((len(free), 2))
        for v in free:
            k = idx[v]
            live = [w for w in nbr[v] if w not in gone]
            lam = anchor.get(v, 0.0) if anchor else 0.0
            rows.append(k); cols.append(k); vals.append(float(len(live)) + lam)
            rhs[k] += lam * P0[v]
            for w in live:
                if w in idx:
                    rows.append(k); cols.append(idx[w]); vals.append(-1.0)
                else:
                    rhs[k] += P[w]
        lu = splu(sp.csc_matrix((vals, (rows, cols)), shape=(len(free), len(free))))
        P[free] = np.column_stack([lu.solve(rhs[:, 0]), lu.solve(rhs[:, 1])])
    for (a, b), comp in flaps:                                 # the given shape, by the similarity a, b -> a, b
        S = sorted(comp)
        u0, u1 = P0[b] - P0[a], P[b] - P[a]
        z = complex(*u1) / complex(*u0) if np.hypot(*u0) > 1e-12 else 1.0
        Q = (P0[S] - P0[a]) @ np.array([[z.real, z.imag], [-z.imag, z.real]])
        ref = P[free].mean(0) if free else P[fixed].mean(0)    # put it on the side of the interior
        if _cross(u1, Q.mean(0)) * _cross(u1, ref - P[a]) < 0:
            e = u1 / max(float(np.hypot(*u1)), 1e-300)
            Q = 2 * np.outer(Q @ e, e) - Q
        P[S] = P[a] + Q
    for a, comp in pieces:
        placed = [w for w in nbr[a] if w not in gone]
        lens = [float(np.linalg.norm(P[w] - P[a])) for w in placed]
        S = sorted(comp)
        Q = P0[S] - P0[a]
        ext = float(np.max(np.linalg.norm(Q, axis=1))) or 1.0
        ell = 0.45 * min(lens) if lens and min(lens) > 0 else ext
        ang = 0.0
        if placed:
            th = sorted(math.atan2(*(P[w] - P[a])[::-1]) for w in placed)
            gaps = [((th[(i + 1) % len(th)] - th[i]) % (2 * math.pi)) or 2 * math.pi for i in range(len(th))]
            i = int(np.argmax(gaps))
            ang = th[i] + 0.5 * gaps[i]
        m = Q.mean(0)
        rot = ang - (math.atan2(m[1], m[0]) if np.hypot(*m) > 1e-12 else 0.0)
        R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        P[S] = P[a] + (ell / ext) * Q @ R.T
    return P


def tutte(d, rims, X0, sign=1.0):
    """Global Tutte embedding: each rim walk (sites of degree 1 dropped, first occurrence of each site kept) on
    a circle of its own length (bond lengths) about the centroid of its component in X0, by arc length
    (sign: orientation); the rest by tutte_solve. Returns (positions, pinned sites)."""
    n = len(X0)
    nbr = _neighbours(d, n)
    root = _components(n, d.edges)
    P = np.array(X0, float, copy=True)
    fixed = np.zeros(n, bool)
    for walk in rims:
        w = [v for v in walk if len(nbr[v]) > 1]
        w = [v for i, v in enumerate(w) if v != w[i - 1]] if len(w) > 1 else w
        if len(w) < 3:
            continue
        r0 = root(w[0])
        c = X0[[v for v in range(n) if root(v) == r0]].mean(0)
        L = np.array([_rest_length(d, w[i], w[(i + 1) % len(w)], X0) for i in range(len(w))])
        s = np.concatenate([[0.0], np.cumsum(L)])[:-1] / L.sum()
        rad = L.sum() / (2 * math.pi)
        for v, si in zip(w, s):
            if not fixed[v]:
                a = sign * 2 * math.pi * si
                P[v] = c + rad * np.array([math.cos(a), math.sin(a)])
                fixed[v] = True
    return tutte_solve(d, P, fixed), np.nonzero(fixed)[0]


def local_tutte(d, X, seeds, k, rim=()):
    """Tutte inside the graph ball of radius k about the seed sites; every other site stays at X; free sites
    of the rim are tied to their place in X by a spring of weight 1. Returns (positions, number of free sites)."""
    import networkx as nx
    ball = nx.multi_source_dijkstra_path_length(d.to_networkx(), set(seeds), cutoff=k)
    fixed = np.ones(len(X), bool)
    fixed[list(ball)] = False
    return tutte_solve(d, X, fixed, anchor={v: 1.0 for v in rim if v in ball}), len(ball)


def connectivity_note(d, rim_sites) -> str:
    import networkx as nx
    G = d.to_networkx()
    cut = sum(1 for _ in nx.articulation_points(G))
    deg = d.degrees()
    rim = set(rim_sites)
    low = int(sum(1 for v in range(len(deg)) if deg[v] <= 2 and v not in rim))
    comps = nx.number_connected_components(G)
    if cut == 0 and low == 0 and comps == 1:
        return "not excluded (no cut site, no interior site of degree <= 2; 2-cuts not checked)"
    return f"no: {comps} component(s), {cut} cut site(s), {low} interior site(s) of degree <= 2"


# --- relaxations ------------------------------------------------------------------------------------------------
def _work_copy(d, X0):
    """A copy whose bonds all carry 'vec' (frame-less bonds: the given drawn vector; a bond marked 'across' gets
    the length max(drawn, median length of the other bonds of its type): never shortened), for
    fast_relax.relax_setup."""
    w = copy.copy(d)
    w.edge = {k: dict(e) for k, e in d.edge.items()}
    w.info = dict(d.info)
    w.info.pop("guess", None)
    w.positions = np.asarray(X0, float)
    if has_frames(d):
        return w
    typ = {}
    for (u, v), e in w.edge.items():
        if not e.get("across"):
            typ.setdefault(e.get("type", 0), []).append(float(np.linalg.norm(X0[v] - X0[u])))
    allmed = float(np.median([x for l in typ.values() for x in l])) if typ else 1.0
    for (u, v), e in w.edge.items():
        vec = X0[v] - X0[u]
        ell = float(np.linalg.norm(vec))
        if e.get("across") or ell < 1e-9:
            want = float(np.median(typ[e.get("type", 0)])) if typ.get(e.get("type", 0)) else allmed
            if e.get("across"):
                want = max(want, ell)
            vec = (vec / ell if ell > 1e-9 else np.array([1.0, 0.0])) * want
        e["vec"] = vec
    return w


def _setup(w, Z0):
    S = _fr.relax_setup(w)
    S["X0"] = np.asarray(Z0, float).copy()
    return S


def _step(S, z, free):
    J = _fr.jacobian(S, z)
    if free is not None:
        J = J[:, free]
    r = _fr.residual(S, z)
    JT = J.T.tocsr()
    return splu((JT @ J).tocsc(), permc_spec="MMD_AT_PLUS_A").solve(-(JT @ r)), 0.5 * float(r @ r)


def relax_pinned(S, z0, free, max_iter=200, rtol=1e-9):
    """Gauss-Newton with backtracking on the free coordinates only (the others stay at z0)."""
    z = z0.copy()
    for it in range(1, max_iter + 1):
        dz, F = _step(S, z, free)
        a = 1.0
        while a > 1e-3:
            z2 = z.copy()
            z2[free] += a * dz
            r2 = _fr.residual(S, z2)
            F2 = 0.5 * float(r2 @ r2)
            if F2 < F:
                break
            a *= 0.5
        else:
            return z, it
        z = z2
        if F - F2 <= rtol * F2:
            break
    return z, it


def relax_guarded(S, z0, badness, max_iter=150, rtol=1e-8, halvings=16, max_checks=120):
    """Gauss-Newton steps from z0, each taken only if the cost drops and badness(z) does not grow (step halved
    up to `halvings` times, else stop; stop also after `max_checks` evaluations of badness, so that the
    result does not depend on the machine's speed). Returns (z, steps taken)."""
    z, b = z0.copy(), badness(z0)
    steps, checks = 0, 0
    for _ in range(max_iter):
        dz, F = _step(S, z, None)
        a = 1.0
        for _ in range(halvings + 1):
            z2 = z + a * dz
            r2 = _fr.residual(S, z2)
            F2 = 0.5 * float(r2 @ r2)
            if F2 < F and checks < max_checks:
                checks += 1
                b2 = badness(z2)
                if b2 <= b:
                    break
            a *= 0.5
        else:
            break
        z, b, steps = z2, b2, steps + 1
        if F - F2 <= rtol * F2:
            break
    return z, steps


# --- walls: open the wall line ------------------------------------------------------------------------------
OPEN_GAPS = (0.5, 0.75, 1.0)
OPEN_SLIDES = (0.0,)          # never slide: a slide would change the slip the picture shows (a glide wall's defining feature)


def wall_sides(d):
    """(unit normal pointing from side B to side A, bool mask of side B) for a wall of lab_defects.wall (info
    'sigma' = (M, t) with M a reflection, sites (key, 'A' / 'B')); None for anything else."""
    sig = d.info.get("sigma")
    if sig is None or d.info.get("invisible"):
        return None
    try:
        side = np.array([s[1] if isinstance(s, tuple) and len(s) == 2 else None for s in d.sites], object)
    except Exception:                                          # noqa: BLE001 - not a wall
        return None
    A, B = side == "A", side == "B"
    if not A.any() or not B.any() or not (A | B).all():
        return None
    M = np.asarray(sig[0], float)
    if M.shape != (2, 2) or np.linalg.det(M) > 0:
        return None
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    nu = V[:, int(np.argmin(w))]
    X = np.asarray(d.positions, float)
    if (X[A] @ nu).mean() < (X[B] @ nu).mean():
        nu = -nu
    return nu, B


def _across_slide(d, X, sides):
    """Median along-wall component of the across bonds from side A to side B (0 when there are none)."""
    nu, B = sides
    tau = np.array([-nu[1], nu[0]])
    acr = [(u, v) for (u, v) in d.edges if d.edge[(u, v)].get("across") and B[v] and not B[u]]
    return float(np.median([(X[v] - X[u]) @ tau for u, v in acr])) if acr else 0.0


def open_wall(d, X, gap, sides=None, slide=0.0):
    """Positions X with side B moved rigidly by -gap x bond unit (bonds not marked 'across') along the wall
    normal and by -slide x (median along-wall component of the across bonds, from side A to side B)."""
    nu, B = sides or wall_sides(d)
    X = np.asarray(X, float)
    keep = [k for k, (u, v) in enumerate(d.edges) if not d.edge[(u, v)].get("across")]
    E = np.array(d.edges, np.int64).reshape(-1, 2)[keep]
    unit = bond_unit(np.linalg.norm(X[E[:, 1]] - X[E[:, 0]], axis=1), edge_types(d)[keep])
    Y = X.copy()
    Y[B] -= gap * unit * nu
    if slide:
        Y[B] -= slide * _across_slide(d, X, (nu, B)) * np.array([-nu[1], nu[0]])
    return Y


# --- the driver ---------------------------------------------------------------------------------------------
LOCAL_RADII = (1, 2, 4, 8, 16)


def _key(c, crowd=True):
    """Badness, lexicographic: crossings and folds first, then overlaps and (crowd=True) crowded sites."""
    return 1_000_000 * (c["crossings"] + (c["folded"] or 0)) + c["overlaps"] + (c.get("crowded", 0) if crowd else 0)


def _counts(c):
    return dict(crossings=c["crossings"], overlaps=c["overlaps"], crowded=c.get("crowded", 0), folded=c["folded"])


def clean(c) -> bool:
    """No crossing, overlap, crowded site or fold left in the counts c (an info['check'])."""
    return not (c.get("crossings") or c.get("overlaps") or c.get("crowded") or c.get("folded"))


def drawing_note(info) -> str:
    """One line for a plot title: what a picture still has (crossings, overlaps, crowded sites, folds); '' when
    nothing (or no info). A rigid fallback says so."""
    c = (info or {}).get("check") or {}
    parts = [f"{c[k]} {w}" for k, w in (("crossings", "crossings"), ("overlaps", "overlaps"),
                                        ("crowded", "crowded sites"), ("folded", "folds")) if c.get(k)]
    if not parts:
        return ""
    return ("rigid picture: " if info.get("stage") == "rigid" else "picture keeps ") + ", ".join(parts)


def embed(d, relax_given=False, overlaps=True, max_iter=200, max_checks=120, verbose=False, wall_local=False):
    """(positions, info) for the Defect d (or a Configuration): see the module docstring."""
    d = getattr(d, "defect", d)
    X0 = np.asarray(d.positions, float)
    n = len(X0)
    info = dict(tried=[])
    T = time.time()
    frames = has_frames(d)
    faces, outer = frame_faces(d) if frames else (None, None)
    faces = Faces(faces) if faces is not None else None
    types = edge_types(d)
    full = lambda Z: check(d, Z, plain_only=False, faces=faces, types=types)
    key = lambda c: _key(c, crowd=overlaps)
    ok = lambda c: key(c) == 0 and not c["folded"]
    bad = lambda z: key(full(z.reshape(-1, 2)))
    cands = []                                                 # (key, order, stage, positions, check, extra)

    def consider(stage, Z, c, t, **kw):
        cands.append((key(c), len(cands), stage, np.asarray(Z, float).reshape(-1, 2), c, kw))
        info["tried"].append((stage, dict(_counts(c), **kw), round(time.time() - t, 3)))
        if verbose:
            print(f"   {stage}: {info['tried'][-1][1]}  # time {time.time() - t:.2f} s")
        return ok(c)

    def done(force=None):
        key, _, stage, Z, c, kw = cands[force] if force is not None else min(cands, key=lambda x: (x[0], -x[1]))
        info.update(stage=stage, check=_counts(c), seconds=round(time.time() - T, 3), **kw)
        return Z, info

    t = time.time()
    c = check(d, X0, plain_only=True, faces=faces, types=types)
    consider("given", X0, c, t)
    if c["crossings"] == 0 and not c["folded"] and ((c["overlaps"] == 0 and c["crowded"] == 0) or not overlaps):
        info["tried"][-1][1]["accepted"] = True
        return done(force=0)
    import networkx as nx
    if not nx.check_planarity(d.to_networkx())[0]:            # no planar picture exists: keep the given one
        info["three_connected"] = "no: not planar"
        return done(force=0)
    sides = wall_sides(d)
    if sides is not None:                                      # a wall: open the wall line
        opened = []
        X = np.asarray(X0, float)
        E = np.array(d.edges, np.int64).reshape(-1, 2)
        unit = bond_unit(np.linalg.norm(X[E[:, 1]] - X[E[:, 0]], axis=1), types)
        slides = OPEN_SLIDES if abs(_across_slide(d, X, sides)) > 1e-6 * unit else OPEN_SLIDES[:1]
        for sl in slides:
            for g in OPEN_GAPS:
                t = time.time()
                Y = open_wall(d, X0, g, sides, sl)
                c = full(Y)
                if consider("opened", Y, c, t, gap=g, slide=sl):
                    return done()
                opened.append((key(c), Y))
        X0 = min(opened, key=lambda x: x[0])[1]                # the later stages start from the least bad one
    w = _work_copy(d, X0)
    if relax_given and (sides is None or wall_local):         # the standard relaxation from the given picture
        t = time.time()
        Z = _fr.gauss_newton(_setup(w, X0), X0.ravel(), max_iter=max_iter)[0]
        if consider("relaxed", Z, full(Z.reshape(-1, 2)), t):
            return done()
    if sides is not None and not wall_local:                   # a wall: no local repair unless asked
        return done()
    c0 = full(X0)                                              # local repair round the bad sites
    seeds = c0["sites"] | (c0["crowded_sites"] if overlaps else set())
    try:
        rim = set(v for f in outer for v in f) if frames else set(d.boundary_loop())
    except Exception:                                          # noqa: BLE001 - a rim is only a help here
        rim = set()
    for k in LOCAL_RADII:
        t = time.time()
        P, nfree = local_tutte(d, X0, seeds, k, rim)
        if nfree > (0.9 if rim else 0.5) * n:                 # free rim sites are held by their springs
            break
        c = full(P)
        if not ok(c) and rim:                                  # the same without the rim springs
            P2 = local_tutte(d, X0, seeds, k)[0]
            c2 = full(P2)
            if key(c2) < key(c):
                P, c = P2, c2
        if c["crossings"] or c["folded"]:
            consider(f"local k={k}", P, c, t, radius=k, free=nfree)
            continue
        Z, steps = relax_guarded(_setup(w, P), P.ravel(), bad, max_checks=max_checks)
        if consider("local", Z, full(Z.reshape(-1, 2)), t, radius=k, free=nfree, steps=steps):
            return done()
    if sides is not None:                                      # a wall: no global Tutte picture
        return done()
    t = time.time()                                            # (b) global Tutte, then the relaxation
    if frames:
        P, pinned = tutte(d, [list(f) for f in outer], X0)
        if len(faces):                                         # keep the handedness of the given picture
            s0 = np.sign(np.sum(np.sign(faces.areas(X0))))
            if s0 and np.sign(np.sum(np.sign(faces.areas(P)))) == -s0:
                P, pinned = tutte(d, [list(f) for f in outer], X0, sign=-1.0)
        cT = full(P)
    else:                                                      # rim: a face of a planar embedding, or the drawn rim
        rims = _planar_outer_faces(d)
        if rims is None:
            info["three_connected"] = "no: not planar"
            return done()
        tries = [rims] + ([[d.boundary_loop()]] if len(rims) == 1 else [])
        best = None
        for r in tries:
            Q, pq = tutte(d, r, X0)
            cq = full(Q)
            if best is None or key(cq) < key(best[2]):
                best = (Q, pq, cq)
        P, pinned, cT = best
    info["three_connected"] = connectivity_note(d, pinned)
    consider("tutte", P, cT, t)
    t = time.time()
    S = _setup(w, P)
    Z = _fr.gauss_newton(S, P.ravel(), max_iter=max_iter)[0]
    if consider("tutte+relax", Z, full(Z.reshape(-1, 2)), t):
        return done()
    t = time.time()                                            # (c) the same, rim held at its Tutte positions
    fixed = np.zeros(2 * n, bool)
    fixed[2 * pinned] = fixed[2 * pinned + 1] = True
    Z = relax_pinned(S, P.ravel(), np.nonzero(~fixed)[0], max_iter=max_iter)[0]
    if consider("pinned", Z, full(Z.reshape(-1, 2)), t):
        return done()
    t = time.time()                                            # (d) Tutte, relaxed only while it gets no worse
    Z, steps = relax_guarded(S, P.ravel(), bad, max_checks=max_checks)
    consider("guarded", Z, full(Z.reshape(-1, 2)), t, steps=steps)
    return done()


def apply(d, P, info):
    """Write positions P (from embed) on d; a picture made here draws every bond in full (no 'ghost' stubs:
    nothing crosses), and info['strain'] keeps the largest relative bond-length change as mark_long_bonds."""
    d.positions = np.asarray(P, float)
    d._nbr = None
    worst = 0.0
    for (u, v), e in d.edge.items():
        e.pop("ghost", None)
        if "vec" in e:
            ell = float(np.hypot(*e["vec"]))
            worst = max(worst, abs(float(np.linalg.norm(P[v] - P[u])) - ell) / max(ell, 1e-12))
    d.info["strain"] = worst
    d.info["drawing"] = dict(info)
    return d


def ensure_embedded(d, relax_given=False, overlaps=True, **kw):
    """In place: keep d's picture if it is embedded (overlaps=False: if it has no crossing and no fold, the
    rule of build(picture="relaxed")), else replace it by embed()'s. Returns info."""
    P, info = embed(d, relax_given=relax_given, overlaps=overlaps, **kw)
    if info["stage"] == "given":
        d.info["drawing"] = dict(stage="given", check=info["check"])
    else:
        apply(d, P, info)
    return info


def _worse(c):
    return c["crossings"] + (c["folded"] or 0)


def embedded_copy(d, rigid_fallback=True, **kw):
    """A copy of the legacy Defect d with an embedded picture (relax_given=True: the standard relaxation from
    the rigid picture is tried first); d itself is not changed. Returns (copy, info). When no stage gives an
    embedded picture and rigid_fallback: the rigid picture (a copy of d), info stage 'rigid', 'check' its
    counts, 'redraw' the best redraw's counts, unless the best redraw has fewer crossings + folds than the
    rigid picture (it keeps only overlaps or crowded sites; e.g. kagome T2: 0 crossings and 2 crowded sites
    against 2 crossings and 2 folds), which is then returned. info['note'] = drawing_note(info) either way."""
    P, info = embed(d, relax_given=True, **kw)                # embed reads d only
    e = copy.deepcopy(d)
    if info["stage"] == "given":
        e.info["drawing"] = dict(stage="given", check=info["check"])
    elif rigid_fallback and not clean(info["check"]) and _worse(info["tried"][0][1]) <= _worse(info["check"]):
        given = info["tried"][0][1]                            # the counts of the rigid picture
        best = dict(stage=info["stage"], check=info["check"])
        info = dict(info, stage="rigid", check={k: given[k] for k in ("crossings", "overlaps", "crowded", "folded")},
                    redraw=best)
        e.info["drawing"] = info
    else:
        apply(e, P, info)
    info["note"] = drawing_note(info)
    return e, info
