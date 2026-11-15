"""Is a drawn crystal a faithful planar picture? Measures of what Defect.plot actually draws.

What is drawn (mirrors lab_defects.Defect.plot, which Configuration.plot and DefectEntry.plot call):
  a bond (u, v) of d.edges whose d.edge[(u, v)] has no 'ghost' key is drawn as the full segment u-v (a
  "plain" bond); a bond with a 'ghost' key (stretched > 1.8 x its length, fusion_framework.mark_long_bonds,
  or closing through a seam, lab_defects.set_picture / mobius_strip) is a "ghost" bond, drawn as two dotted
  stubs u -> edge[(u, v)]['ghost'] and v -> edge[(v, u)]['ghost'].

Measures (definitions; geometry on the drawn positions, except the face structure, see 'folded'):
  plain_crossings   pairs of plain bonds that cross in their interiors (no common end site).
  stub_crossings    pairs of bonds that cross when every ghost bond is taken at full length, at least one of
                    the pair a ghost bond: what the dotted stubs stand in for (the diagnosis' 'ghost_involved').
  stub_drawn_crossings  information: pairs of drawn segments (plain bonds and stubs as drawn) that cross, at
                    least one a stub.
  crossings         information: all pairs of bonds at full length that cross (the old tool's single number)
                    = plain_crossings + stub_crossings.
  overlaps          overlap_pairs + sites_on_bonds, over plain bonds:
                      overlap_pairs  pairs of collinear plain bonds whose common part is longer than tol
                                     (two bonds leaving one site in the same direction count);
                      sites_on_bonds (site, plain bond) with the site not an end of the bond, within tol of
                                     the bond and more than tol from both of its ends.
                    tol = rel_tol x the median drawn plain-bond length (default rel_tol = 0.01).
  stub_overlaps     information: the same two tests where a stub (as drawn) is involved.
  crowded           crowded_pairs + crowded_sites_on_bonds (sites stacked or nearly so; counted like overlaps):
                      crowded_pairs  pairs of sites (bonded or not) closer than crowd_tol;
                      crowded_sites_on_bonds (site, plain bond) with the site not an end of the bond, closer
                                     than crowd_tol to the bond's line, its foot on the bond (0 <= along <= length),
                                     and at least crowd_tol from both ends (else it is a crowded pair).
                    crowd_tol = crowd_rel x the bond unit (default crowd_rel = 0.35); the bond unit is the
                    smallest, over the bond types (edge 'type', one type when missing), of the median drawn
                    plain-bond length: the median bond length on a lattice with one bond length. (The per-type
                    unit keeps perfect patches at 0: perfect brick_pg has bonds 1/3 of the median, perfect
                    straight_p1 sites 0.34 median from bonds; control 7.)
  folded            number of faces of the COMBINATORIAL face structure that are folded: signed area (on the
                    drawn positions) of the minority sign (folded_minority), or |area| < area_tol x median
                    |area| (folded_small, default 1e-3), or a boundary that crosses itself (folded_selfcross).
                    The faces are the walks of fast.frame_faces: the order of the bonds round a site is read
                    from the bonds' 'ang' / 'vec' (perfect-lattice directions in the site's frame), never from
                    the drawing, so a fold or a swap of two sites cannot hide itself. The outer face of each
                    connected component (its longest walk) is skipped; 'faces' counts the others.
                    folded is None (and faces None) when there is no such structure: some bond carries
                    neither 'ang' nor 'vec' (the legacy catalogue walls, disclinations and Moebius strips), or
                    the lattice is not planar; faces_source says which. Test 'folded' with `or 0` only if a
                    None may be read as "not measured".
  folded_drawn      information, ALWAYS an int, position-dependent (can be fooled): face walks of the DRAWN
                    rotation system (Defect._rotation_system) with signed area < 0, minus one outer walk per
                    connected component. Defect.faces() drops exactly these walks. It can be negative (a
                    component whose outer walk has zero area, e.g. a tree) and it misses folds the drawn
                    rotation system re-sorts away (it stays 0 on controls 2 and 3b below).
  rim_turn          information: unwrapped polar angle (deg) about the centroid of the drawn sites,
                    accumulated along the outer face over plain bonds; rim_turn_ghost over ghost bonds. Their
                    sum is 360 x (winding number of the rim); ghost bonds carrying a large share mean a far
                    field covering more or less than 360 deg.

    python tools/checks/drawing_check.py      # known-answer controls (time printed)

Use from code:
    from tools.checks.drawing_check import measure
    measure(configuration_or_defect)   # dict, keys listed in KEYS
"""
from __future__ import annotations

import math

import numpy as np

KEYS = ("sites", "bonds", "faces", "plain_bonds", "ghost_bonds", "plain_crossings", "stub_crossings",
        "stub_drawn_crossings", "crossings", "overlaps", "overlap_pairs", "sites_on_bonds", "stub_overlaps",
        "crowded", "crowded_pairs", "crowded_sites_on_bonds", "crowd_tol", "folded", "folded_minority", "folded_small", "folded_selfcross", "folded_drawn", "faces_source",
        "rim_turn", "rim_turn_ghost", "rim_source", "tol")


# --- what is drawn -----------------------------------------------------------------------------------
def _defect(config):
    return getattr(config, "defect", config)


def drawn(d):
    """(X, plain, ghost, stubs): positions; plain and ghost bonds as lists of (u, v) in d.edges order; the
    stub segments as a list of (site, end point, ghost bond index). Same rule as Defect.plot."""
    X = np.asarray(d.positions, float)
    plain, ghost, stubs = [], [], []
    for (u, v) in d.edges:
        e = d.edge[(u, v)]
        if e.get("ghost") is None:
            plain.append((u, v))
            continue
        k = len(ghost)
        ghost.append((u, v))
        back = d.edge.get((v, u), {}).get("ghost")
        stubs.append((u, np.asarray(e["ghost"], float), k))
        stubs.append((v, np.asarray(back if back is not None else 0.5 * (X[u] + X[v]), float), k))
    return X, plain, ghost, stubs


# --- segment geometry ----------------------------------------------------------------------------------
def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _boxes(A, B, tol=0.0):
    return np.minimum(A, B) - tol, np.maximum(A, B) + tol


def crossing_pairs(A, B, ea, eb, eps=1e-9, chunk=256):
    """Pairs (k, j), k < j, of segments A[k]-B[k] that cross in both interiors and share no end id
    (ea, eb: end ids; a stub's free end gets an id no site has)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ea, eb = np.asarray(ea), np.asarray(eb)
    n = len(A)
    out = []
    if n < 2:
        return out
    lo, hi = _boxes(A, B)
    R = B - A
    for s in range(0, n, chunk):
        k = np.arange(s, min(s + chunk, n))[:, None]
        j = np.arange(n)[None, :]
        cand = (j > k) & (lo[j, 0] <= hi[k, 0]) & (lo[k, 0] <= hi[j, 0]) & (lo[j, 1] <= hi[k, 1]) & (lo[k, 1] <= hi[j, 1])
        cand &= (ea[j] != ea[k]) & (ea[j] != eb[k]) & (eb[j] != ea[k]) & (eb[j] != eb[k])
        kk, jj = np.nonzero(cand)
        if not len(kk):
            continue
        kk = kk + s
        r, q = R[kk], R[jj]
        den = _cross(r, q)
        w = A[jj] - A[kk]
        ok = np.abs(den) > 1e-14 * np.linalg.norm(r, axis=1) * np.linalg.norm(q, axis=1)
        den = np.where(ok, den, 1.0)
        t = _cross(w, q) / den
        u = _cross(w, r) / den
        hit = ok & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)
        out.extend(zip(kk[hit].tolist(), jj[hit].tolist()))
    return out


def crossings(X, E) -> int:
    """Pairs of bonds X[E[k,0]]-X[E[k,1]] (full length) that cross in their interiors, no common end site.
    Kept for the diagnosis scripts that import it."""
    X = np.asarray(X, float)
    E = np.asarray(E, int).reshape(-1, 2)
    return len(crossing_pairs(X[E[:, 0]], X[E[:, 1]], E[:, 0], E[:, 1]))


def collinear_pairs(A, B, tol, chunk=256):
    """Pairs (k, j), k < j, of segments on a common line (both ends of j within tol of k's line) whose
    common part is longer than tol."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    n = len(A)
    out = []
    if n < 2:
        return out
    R = B - A
    L = np.maximum(np.linalg.norm(R, axis=1), 1e-300)
    D = R / L[:, None]
    lo, hi = _boxes(A, B, tol)
    for s in range(0, n, chunk):
        k = np.arange(s, min(s + chunk, n))[:, None]
        j = np.arange(n)[None, :]
        cand = (j > k) & (lo[j, 0] <= hi[k, 0]) & (lo[k, 0] <= hi[j, 0]) & (lo[j, 1] <= hi[k, 1]) & (lo[k, 1] <= hi[j, 1])
        kk, jj = np.nonzero(cand)
        if not len(kk):
            continue
        kk = kk + s
        dk = D[kk]
        par = (np.abs(_cross(dk, A[jj] - A[kk])) <= tol) & (np.abs(_cross(dk, B[jj] - A[kk])) <= tol)
        pa = np.einsum("ij,ij->i", A[jj] - A[kk], dk)      # j's ends projected on k's line
        pb = np.einsum("ij,ij->i", B[jj] - A[kk], dk)
        common = np.minimum(L[kk], np.maximum(pa, pb)) - np.maximum(0.0, np.minimum(pa, pb))
        hit = par & (common > tol)
        out.extend(zip(kk[hit].tolist(), jj[hit].tolist()))
    return out


def points_on_segments(P, pid, A, B, ea, eb, tol, chunk=256):
    """(point index, segment index): point within tol of the segment, more than tol from both of its
    ends, and not one of its ends (pid[i] not in {ea[k], eb[k]})."""
    P, A, B = np.asarray(P, float), np.asarray(A, float), np.asarray(B, float)
    pid, ea, eb = np.asarray(pid), np.asarray(ea), np.asarray(eb)
    out = []
    if not len(P) or not len(A):
        return out
    R = B - A
    L = np.maximum(np.linalg.norm(R, axis=1), 1e-300)
    D = R / L[:, None]
    lo, hi = _boxes(A, B, tol)
    for s in range(0, len(P), chunk):
        i = np.arange(s, min(s + chunk, len(P)))[:, None]
        k = np.arange(len(A))[None, :]
        px, py = P[i[:, 0], 0][:, None], P[i[:, 0], 1][:, None]
        cand = (px >= lo[k, 0]) & (px <= hi[k, 0]) & (py >= lo[k, 1]) & (py <= hi[k, 1])
        cand &= (pid[i] != ea[k]) & (pid[i] != eb[k])
        ii, kk = np.nonzero(cand)
        if not len(ii):
            continue
        ii = ii + s
        w = P[ii] - A[kk]
        along = np.einsum("ij,ij->i", w, D[kk])
        off = np.abs(_cross(D[kk], w))
        hit = (off <= tol) & (along > tol) & (along < L[kk] - tol)
        out.extend(zip(ii[hit].tolist(), kk[hit].tolist()))
    return out


def close_pairs(P, tol, chunk=256):
    """Pairs (i, j), i < j, of points closer than tol (brute force by chunks)."""
    P = np.asarray(P, float)
    out = []
    for s in range(0, len(P), chunk):
        i = np.arange(s, min(s + chunk, len(P)))
        D = np.linalg.norm(P[i][:, None, :] - P[None, :, :], axis=2)
        ii, jj = np.nonzero((D < tol) & (np.arange(len(P))[None, :] > i[:, None]))
        out.extend(zip((ii + s).tolist(), jj.tolist()))
    return out


def points_near_segments(P, pid, A, B, ea, eb, tol, chunk=256):
    """(point index, segment index): point not an end, closer than tol to the segment's line with its foot on
    the segment (0 <= along <= length), and at least tol from both ends."""
    P, A, B = np.asarray(P, float), np.asarray(A, float), np.asarray(B, float)
    pid, ea, eb = np.asarray(pid), np.asarray(ea), np.asarray(eb)
    out = []
    if not len(P) or not len(A):
        return out
    R = B - A
    L = np.maximum(np.linalg.norm(R, axis=1), 1e-300)
    D = R / L[:, None]
    lo, hi = _boxes(A, B, tol)
    for s in range(0, len(P), chunk):
        i = np.arange(s, min(s + chunk, len(P)))[:, None]
        k = np.arange(len(A))[None, :]
        px, py = P[i[:, 0], 0][:, None], P[i[:, 0], 1][:, None]
        cand = (px >= lo[k, 0]) & (px <= hi[k, 0]) & (py >= lo[k, 1]) & (py <= hi[k, 1])
        cand &= (pid[i] != ea[k]) & (pid[i] != eb[k])
        ii, kk = np.nonzero(cand)
        if not len(ii):
            continue
        ii = ii + s
        w = P[ii] - A[kk]
        along = np.einsum("ij,ij->i", w, D[kk])
        off = np.abs(_cross(D[kk], w))
        da, db = np.linalg.norm(w, axis=1), np.linalg.norm(P[ii] - B[kk], axis=1)
        hit = (off < tol) & (along >= 0) & (along <= L[kk]) & (da >= tol) & (db >= tol)
        out.extend(zip(ii[hit].tolist(), kk[hit].tolist()))
    return out


def bond_unit(d, plain, lens):
    """Smallest median drawn length over the bond types of the plain bonds."""
    if not len(lens):
        return 1.0
    typ = np.array([str(d.edge[(u, v)].get("type", 0)) for u, v in plain])
    return min(float(np.median(lens[typ == t])) for t in set(typ.tolist()))


def signed_areas(X, faces):
    X = np.asarray(X, float)
    out = []
    for f in faces:
        P = X[list(f)]
        x, y = P[:, 0], P[:, 1]
        out.append(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return np.array(out)


# --- faces ---------------------------------------------------------------------------------------------
def _components(n, edges):
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (u, v) in edges:
        parent[root(u)] = root(v)
    return root


def combinatorial_faces(config):
    """(inner faces, outer faces, source). The face walks of fast.frame_faces (rotation system from the
    bonds' 'ang' / 'vec', independent of the positions); the longest walk of each connected component is
    its outer face. (None, None, reason) when the structure does not exist."""
    d = _defect(config)
    lat = getattr(config, "lattice", None)
    if lat is not None and getattr(lat, "planar", True) is False:
        return None, None, "none: the lattice is not planar"
    if not d.edge:
        return [], [], "none: no bonds"
    if any("ang" not in e and "vec" not in e for e in d.edge.values()):
        return None, None, "none: some bonds carry neither 'ang' nor 'vec' (no drawing-free rotation system)"
    from papertilling.defects import fast as _fast
    F = _fast.frame_faces(d)
    root = _components(len(d.positions), d.edge)
    outer = {}
    for i, f in enumerate(F):
        c = root(f[0])
        if c not in outer or len(f) > len(F[outer[c]]):
            outer[c] = i
    skip = set(outer.values())
    return [f for i, f in enumerate(F) if i not in skip], [F[i] for i in sorted(skip)], "fast.frame_faces ('ang'/'vec')"


def self_crossing(X, f) -> bool:
    """Does the closed walk f cross itself (two edges of the walk sharing no site cross)?"""
    if len(f) < 4:
        return False
    f = list(f)
    g = f[1:] + f[:1]
    return bool(crossing_pairs(X[f], X[g], np.array(f), np.array(g)))


def folded_faces(X, faces, area_tol=1e-3):
    """(minority sign, small |area|, self-crossing, folded = any of the three) counts over the faces."""
    if not faces:
        return 0, 0, 0, 0
    X = np.asarray(X, float)
    a = signed_areas(X, faces)
    s = np.sign(a)
    sign = 1.0 if np.sum(s > 0) >= np.sum(s < 0) else -1.0
    minority = s != sign
    small = np.abs(a) < area_tol * np.median(np.abs(a))
    selfx = np.array([self_crossing(X, f) for f in faces])
    return int(minority.sum()), int(small.sum()), int(selfx.sum()), int((minority | small | selfx).sum())


def drawn_negative_walks(d) -> int:
    """Face walks of the drawn rotation system with signed area < 0, minus one outer walk per connected
    component (position-dependent)."""
    nbr, P = d._rotation_system(), np.asarray(d.positions, float)
    seen, neg = set(), 0
    for u in nbr:
        for v in nbr[u]:
            if (u, v) in seen:
                continue
            f, a, b = [], u, v
            while (a, b) not in seen:
                seen.add((a, b)); f.append(a)
                k = nbr[b].index(a)
                a, b = b, nbr[b][k - 1]
            neg += signed_areas(P, [f])[0] < -1e-12
    root = _components(len(P), d.edges)
    comps = {root(u) for e in d.edges for u in e}
    return int(neg) - len(comps)


def rim_turn(X, loop, is_ghost):
    """(deg over plain bonds, deg over ghost bonds): polar angle about the centroid of X, unwrapped
    step by step along the closed loop of sites."""
    c = np.asarray(X, float).mean(0)
    plain = gh = 0.0
    for i in range(len(loop)):
        u, v = loop[i], loop[(i + 1) % len(loop)]
        du, dv = X[u] - c, X[v] - c
        da = math.degrees(math.atan2(du[0] * dv[1] - du[1] * dv[0], du @ dv))
        if is_ghost(u, v):
            gh += da
        else:
            plain += da
    return round(plain, 3), round(gh, 3)


# --- the measure -----------------------------------------------------------------------------------------
def measure(config, rel_tol=0.01, area_tol=1e-3, crowd_rel=0.35) -> dict:
    """All measures for a papertilling Configuration (pt.build(...)) or a legacy lab_defects.Defect
    (DefectEntry.build(R)). Keys: KEYS."""
    d = _defect(config)
    X, plain, ghost, stubs = drawn(d)
    n = len(X)
    P = np.array(plain, int).reshape(-1, 2)
    G = np.array(ghost, int).reshape(-1, 2)
    m = len(P)
    lens = np.linalg.norm(X[P[:, 1]] - X[P[:, 0]], axis=1) if m else np.zeros(0)
    tol = rel_tol * (float(np.median(lens)) if m else 1.0)

    # full-length bonds: plain first, then ghost
    FA = np.vstack([X[P[:, 0]], X[G[:, 0]]]); FB = np.vstack([X[P[:, 1]], X[G[:, 1]]])
    fea = np.concatenate([P[:, 0], G[:, 0]]); feb = np.concatenate([P[:, 1], G[:, 1]])
    full = crossing_pairs(FA, FB, fea, feb)
    plain_x = sum(1 for k, j in full if k < m and j < m)

    # drawn segments: plain bonds, then stubs (free end id -1 - stub index, never a site)
    SA = np.vstack([X[P[:, 0]], np.array([X[s] for s, _, _ in stubs]).reshape(-1, 2)])
    SB = np.vstack([X[P[:, 1]], np.array([q for _, q, _ in stubs]).reshape(-1, 2)])
    ea = np.concatenate([P[:, 0], np.array([s for s, _, _ in stubs], int)])
    eb = np.concatenate([P[:, 1], -1 - np.arange(len(stubs))])
    stub_drawn = sum(1 for k, j in crossing_pairs(SA, SB, ea, eb) if k >= m or j >= m)

    col = collinear_pairs(SA, SB, tol)
    on = points_on_segments(X, np.arange(n), SA, SB, ea, eb, tol)
    ov_pairs = sum(1 for k, j in col if k < m and j < m)
    ov_sites = sum(1 for i, k in on if k < m)
    stub_ov = (len(col) - ov_pairs) + (len(on) - ov_sites)

    ctol = crowd_rel * bond_unit(d, plain, lens)
    c_pairs = len(close_pairs(X, ctol))
    c_on = len(points_near_segments(X, np.arange(n), X[P[:, 0]], X[P[:, 1]], P[:, 0], P[:, 1], ctol))

    inner, outer, source = combinatorial_faces(config)
    gset = {(u, v) for u, v in ghost} | {(v, u) for u, v in ghost}
    if inner is None:
        nf, fm, fs, fx, folded = None, None, None, None, None
        try:
            loop, rim_src = d.boundary_loop(), "Defect.boundary_loop (drawn rotation system)"
        except Exception as ex:                                    # noqa: BLE001 - information only
            loop, rim_src = None, f"none: {type(ex).__name__}"
    else:
        nf = len(inner)
        fm, fs, fx, folded = folded_faces(X, inner, area_tol)
        loop = max(outer, key=len) if outer else None
        rim_src = "outer face of fast.frame_faces"
    turn = rim_turn(X, loop, lambda u, v: (u, v) in gset) if loop else (None, None)

    return dict(sites=n, bonds=len(d.edges), faces=nf, plain_bonds=m, ghost_bonds=len(G),
                plain_crossings=int(plain_x), stub_crossings=int(len(full) - plain_x),
                stub_drawn_crossings=int(stub_drawn), crossings=len(full),
                overlaps=int(ov_pairs + ov_sites), overlap_pairs=int(ov_pairs), sites_on_bonds=int(ov_sites),
                stub_overlaps=int(stub_ov), crowded=int(c_pairs + c_on), crowded_pairs=int(c_pairs),
                crowded_sites_on_bonds=int(c_on), crowd_tol=round(ctol, 6), folded=folded, folded_minority=fm, folded_small=fs,
                folded_selfcross=fx, folded_drawn=drawn_negative_walks(d), faces_source=source,
                rim_turn=turn[0], rim_turn_ghost=turn[1], rim_source=rim_src, tol=round(tol, 6))


# --- known-answer controls ---------------------------------------------------------------------------------
def _controls():
    import copy
    import sys
    import time
    import warnings
    from pathlib import Path
    warnings.simplefilter("ignore")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import papertilling as pt
    from papertilling.defects.catalogue import DefectCatalogue

    show = ("plain_crossings", "stub_crossings", "stub_drawn_crossings", "overlaps", "overlap_pairs",
            "sites_on_bonds", "stub_overlaps", "crowded", "crowded_pairs", "crowded_sites_on_bonds", "folded", "folded_minority", "folded_small", "folded_selfcross",
            "folded_drawn", "faces", "sites", "bonds", "ghost_bonds", "rim_turn", "rim_turn_ghost")
    results = []

    def report(name, m, ok, t):
        results.append(ok)
        print(f"[{'ok' if ok else 'FAIL'}] {name}  # time {time.time() - t:.1f} s")
        print("      " + ", ".join(f"{k}={m.get(k)}" for k in show))
        if m.get("folded") is None:
            print("      faces_source:", m["faces_source"])

    T = time.time()
    t = time.time()
    m = measure(pt.build("square", dict(kind="dislocation", b=(1, 0), at=(0.37, 0.41)), picture="relaxed"))
    report("1 relaxed square dislocation b=(1,0) at (0.37,0.41): expect all zero", m,
           m["plain_crossings"] == 0 and m["stub_crossings"] == 0 and m["overlaps"] == 0 and m["folded"] == 0, t)

    pair = (dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)))
    t = time.time()
    m = measure(pt.build("square", *pair, picture="raw"))
    report("2 raw square +90 at (0,0), -90 at (0.5,0.5): expect plain_crossings > 0, folded > 0", m,
           m["plain_crossings"] > 0 and (m["folded"] or 0) > 0, t)

    t = time.time()
    c = pt.build("square", *pair, picture="relaxed")
    m0 = measure(c)
    report("3a the same, relaxed (baseline for 3b)", m0, m0["plain_crossings"] == 0 and m0["folded"] == 0, t)
    t = time.time()
    d = copy.deepcopy(c.defect)
    X = np.array(d.positions, float)
    i = int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))
    j = next(v for (u, v), e in d.edge.items() if u == i and e.get("ghost") is None)
    X[[i, j]] = X[[j, i]]
    d.positions, d._nbr = X, None
    m = measure(d)
    report(f"3b the same relaxed picture, bonded sites {i} and {j} (nearest the centroid) swapped: expect folded > 0",
           m, (m["folded"] or 0) > 0, t)

    t = time.time()
    cat = DefectCatalogue(pt.named("square"))
    e = cat.mobius_seams[0]
    m = measure(e.build(R=cat.R0))
    report(f"4 square Moebius seam {e['id']} (R ignored, strip): expect plain_crossings 0, stub_crossings > 0", m,
           m["plain_crossings"] == 0 and m["stub_crossings"] > 0, t)

    t = time.time()
    e = next(x for x in cat.glide_breaking if x["id"] == "Wd1")
    m = measure(e.build(R=cat.R0))
    report(f"5 square glide-breaking wall Wd1, rigid picture, R0={cat.R0}: expect overlaps > 0", m, m["overlaps"] > 0, t)

    t = time.time()
    p4g = DefectCatalogue(pt.load_catalogue()["p4g"].lattices[0])
    e = next(x for x in p4g.glide_breaking if x["id"] == "Wd1")
    rigid = e.build(R=p4g.R0)
    m = measure(rigid)
    report(f"6a p4g lattices[0] glide-breaking wall Wd1, rigid picture (sites on the wall line): expect crowded > 0",
           m, m["crowded"] > 0 and m["overlaps"] > 0, t)
    t = time.time()
    X = np.array(rigid.positions, float)                  # the picture of 2026-11-14: wall sites stacked
    dd = np.linalg.norm(X[:, None] - X[None], axis=2) + np.eye(len(X)) * 1e9
    i, j = np.unravel_index(np.argmin(dd), dd.shape)
    st = copy.deepcopy(rigid)
    X[j] = X[i] + 0.047 * m["crowd_tol"] / 0.35 * (X[j] - X[i]) / np.linalg.norm(X[j] - X[i])
    st.positions, st._nbr = X, None
    m = measure(st)
    report(f"6b the same, sites {i} and {j} (closest pair) moved to 0.047 bond units apart: expect crowded_pairs > 0",
           m, m["crowded_pairs"] > 0, t)
    t = time.time()
    d, info = e.picture(R=p4g.R0)
    m = measure(d)
    report(f"6c the same wall, entry.picture() (stage {info['stage']}): expect crossings, overlaps, crowded all 0",
           m, m["plain_crossings"] == 0 and m["stub_crossings"] == 0 and m["overlaps"] == 0 and m["crowded"] == 0, t)

    t = time.time()
    from papertilling.legacy.lab_defects import Defect, Perfect
    lats = [("p4g lattices[0]", p4g), ("p6m lattices[3]", DefectCatalogue(pt.load_catalogue()["p6m"].lattices[3]))]
    lats += [(nm, DefectCatalogue(pt.named(nm))) for nm in ("square", "kagome", "honeycomb", "shastry_sutherland",
             "chiral_square", "chiral_kagome", "zigzag", "sutherlands_brick", "brick_pg", "straight_p1")]
    worst = {}
    for nm, c in lats:
        L = Perfect(c.cell)
        keys = L.sites_in_disk(np.zeros(2), c.R0)
        index = {k: i for i, k in enumerate(keys)}
        edge = {(i, index[k2]): dict(type=typ) for k, i in index.items() for k2, typ in L.nbrs(k) if k2 in index}
        pd = Defect("perfect", keys, [L.pos(k) for k in keys], edge, {})
        worst[nm] = measure(pd)
    bad = {k: (v["crowded"], v["overlaps"], v["plain_crossings"]) for k, v in worst.items()
           if v["crowded"] or v["overlaps"] or v["plain_crossings"]}
    m = worst["square"]
    report(f"7a perfect patches (radius R0) of 12 lattices: expect crowded = overlaps = crossings = 0; nonzero: {bad}",
           m, not bad, t)
    t = time.time()
    X = np.array(pd.positions, float)                     # straight_p1 perfect patch, one site moved
    k0 = int(np.argmin(np.linalg.norm(X - X.mean(0), axis=1)))
    u, v = next((u, v) for (u, v) in pd.edges if u == k0)
    X[k0] = X[v] + 0.2 * (X[k0] - X[v])
    bump = copy.deepcopy(pd)
    bump.positions, bump._nbr = X, None
    m = measure(bump)
    report(f"7b perfect straight_p1, site {k0} moved to 0.2 of its bond to {v}: expect crowded > 0", m, m["crowded"] > 0, t)

    ok = all(results)
    print("CONTROLS", "PASSED" if ok else "FAILED", f"({sum(results)}/{len(results)})  # time {time.time() - T:.1f} s")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _controls() else 1)
