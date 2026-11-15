"""Is a drawn crystal a faithful planar picture? The checks used by draw.py (package copy of the definitions
of tools/checks/drawing_check.py, which the package cannot import; tests/test_drawing.py compares the two).

A picture is *embedded* when it has no
  crossings  two bonds crossing in their interiors (eps 1e-9; bonds sharing a site are not compared);
  folds      face of the combinatorial face structure (fast.frame_faces: bond order round a site read from the
             bonds' 'ang' / 'vec', never from the drawing; the longest walk of each connected component is its
             outer face and is skipped) with signed area of the minority sign, |area| < 1e-3 median |area|, or a
             boundary crossing itself; None (not measured) when some bond has neither 'ang' nor 'vec';
  overlaps   two collinear bonds sharing a part longer than tol, or a site within tol of a bond it is not an end
             of (more than tol from both ends); tol = 0.01 x median bond length;
  crowded    two sites closer than ctol (bonded or not), or a site closer than ctol to a bond it is not an end of,
             its foot on the bond and the site at least ctol from both ends (else it is a crowded pair);
             ctol = 0.35 x the bond unit, the smallest over bond types ('type', one type when missing) of the
             median drawn length. On a lattice with one bond length the unit is the median bond length; the
             per-type unit keeps perfect patches at 0 where short bonds are common (brick_pg: 1/3 of the
             median) or sites sit 0.34 median from bonds (straight_p1). Counted like the overlaps: a crowded
             picture (sites stacked at a wall) is not embedded.
check(d, X, plain_only=True) counts only the bonds drawn as full segments (no 'ghost' key), as
drawing_check.measure does; plain_only=False counts every bond at full length.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import fast as _fast

EPS = 1e-9
REL_TOL = 0.01
CROWD_REL = 0.35
AREA_TOL = 1e-3


# --- segment geometry (definitions of drawing_check) ---------------------------------------------------
def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _near_pairs(A, B, pad=0.0):
    """Pairs (k, j), k < j, of segments whose bounding boxes (grown by pad) meet: a superset of the pairs that
    cross or come within pad. Grid of cell size = median length; a segment is entered in every cell its box
    covers; pairs sharing a cell, de-duplicated."""
    n = len(A)
    if n < 2:
        return np.zeros((0, 2), np.int64)
    lo, hi = np.minimum(A, B) - pad, np.maximum(A, B) + pad
    h = max(float(np.median(np.linalg.norm(B - A, axis=1))), 1e-12)
    i0, i1 = np.floor(lo / h).astype(np.int64), np.floor(hi / h).astype(np.int64)
    nxs, nys = i1[:, 0] - i0[:, 0] + 1, i1[:, 1] - i0[:, 1] + 1
    cnt = nxs * nys
    seg = np.repeat(np.arange(n), cnt)
    off = np.arange(len(seg)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    cx, cy = i0[seg, 0] + off % nxs[seg], i0[seg, 1] + off // nxs[seg]
    key = (cx - cx.min()) * (int(cy.max() - cy.min()) + 1) + (cy - cy.min())
    o = np.argsort(key, kind="stable")
    key, seg = key[o], seg[o]
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    size = np.diff(np.r_[start, len(key)])
    end = np.repeat(start + size, size)                       # end of each element's group
    num = end - np.arange(len(key)) - 1
    left = np.repeat(np.arange(len(key)), num)
    right = left + 1 + (np.arange(len(left)) - np.repeat(np.cumsum(num) - num, num))
    a, b = seg[left], seg[right]
    k, j = np.minimum(a, b), np.maximum(a, b)
    keep = k != j
    code = np.unique(k[keep] * n + j[keep])
    k, j = code // n, code % n
    meet = (lo[j, 0] <= hi[k, 0]) & (lo[k, 0] <= hi[j, 0]) & (lo[j, 1] <= hi[k, 1]) & (lo[k, 1] <= hi[j, 1])
    return np.column_stack([k[meet], j[meet]])


def crossing_pairs(A, B, ea, eb, pairs=None):
    """Pairs (k, j) of segments A-B crossing in both interiors, no common end id (drawing_check.crossing_pairs)."""
    A, B, ea, eb = np.asarray(A, float), np.asarray(B, float), np.asarray(ea), np.asarray(eb)
    P = _near_pairs(A, B) if pairs is None else pairs
    if not len(P):
        return P
    k, j = P[:, 0], P[:, 1]
    keep = (ea[j] != ea[k]) & (ea[j] != eb[k]) & (eb[j] != ea[k]) & (eb[j] != eb[k])
    k, j = k[keep], j[keep]
    r, q, w = B[k] - A[k], B[j] - A[j], A[j] - A[k]
    den = _cross(r, q)
    ok = np.abs(den) > 1e-14 * np.linalg.norm(r, axis=1) * np.linalg.norm(q, axis=1)
    den = np.where(ok, den, 1.0)
    t, u = _cross(w, q) / den, _cross(w, r) / den
    hit = ok & (t > EPS) & (t < 1 - EPS) & (u > EPS) & (u < 1 - EPS)
    return np.column_stack([k[hit], j[hit]])


def _site_bond_candidates(X, E, A, B, L, radius, tree=None):
    """(site, bond) index arrays: sites within 0.5 L + radius of a bond's midpoint, the bond's ends removed."""
    tree = cKDTree(X) if tree is None else tree
    lists = tree.query_ball_point(0.5 * (A + B), 0.5 * L + radius)
    kk = np.repeat(np.arange(len(E)), [len(x) for x in lists])
    ii = np.fromiter((i for x in lists for i in x), np.int64, count=len(kk))
    keep = (ii != E[kk, 0]) & (ii != E[kk, 1])
    return ii[keep], kk[keep]


def overlap_count(X, E, tol, pairs=None, where=None, cand=None):
    """(collinear pairs with a common part > tol, (site, bond) with the site within tol of the bond's interior);
    the sites involved are added to the set `where` if given. cand: (site, bond) candidates, a superset (from
    _site_bond_candidates with a radius >= tol)."""
    where = set() if where is None else where
    A, B = X[E[:, 0]], X[E[:, 1]]
    R = B - A
    L = np.maximum(np.linalg.norm(R, axis=1), 1e-300)
    D = R / L[:, None]
    P = _near_pairs(A, B, tol) if pairs is None else pairs
    col = 0
    if len(P):
        k, j = P[:, 0], P[:, 1]
        dk = D[k]
        par = (np.abs(_cross(dk, A[j] - A[k])) <= tol) & (np.abs(_cross(dk, B[j] - A[k])) <= tol)
        pa, pb = np.einsum("ij,ij->i", A[j] - A[k], dk), np.einsum("ij,ij->i", B[j] - A[k], dk)
        common = np.minimum(L[k], np.maximum(pa, pb)) - np.maximum(0.0, np.minimum(pa, pb))
        hit = par & (common > tol)
        col = int(np.sum(hit))
        where.update(E[k[hit]].ravel().tolist()); where.update(E[j[hit]].ravel().tolist())
    ii, kk = _site_bond_candidates(X, E, A, B, L, tol) if cand is None else cand
    w = X[ii] - A[kk]
    along, off = np.einsum("ij,ij->i", w, D[kk]), np.abs(_cross(D[kk], w))
    hit = (off <= tol) & (along > tol) & (along < L[kk] - tol)
    where.update(ii[hit].tolist()); where.update(E[kk[hit]].ravel().tolist())
    return col, int(np.sum(hit))


def edge_types(d) -> np.ndarray:
    """Integer code of each bond's 'type' (0 when missing), bonds in d.edges order."""
    code = {}
    return np.array([code.setdefault(d.edge[(u, v)].get("type", 0), len(code)) for u, v in d.edges], np.int64)


def bond_unit(lens, types=None) -> float:
    """The smallest median length over bond types (the median length when types is None)."""
    lens = np.asarray(lens, float)
    if not len(lens):
        return 1.0
    if types is None:
        return float(np.median(lens))
    types = np.asarray(types)
    return min(float(np.median(lens[types == t])) for t in np.unique(types))


def crowd_count(X, E, ctol, where=None, cand=None, tree=None):
    """(pairs of sites closer than ctol, (site, bond) with the site closer than ctol to the bond's line, its
    foot on the bond, at least ctol from both ends, not an end); the sites involved are added to `where`."""
    where = set() if where is None else where
    tree = cKDTree(X) if tree is None else tree
    pr = tree.query_pairs(ctol, output_type="ndarray")
    where.update(pr.ravel().tolist())
    if not len(E):
        return len(pr), 0
    A, B = X[E[:, 0]], X[E[:, 1]]
    R = B - A
    L = np.maximum(np.linalg.norm(R, axis=1), 1e-300)
    ii, kk = _site_bond_candidates(X, E, A, B, L, ctol, tree) if cand is None else cand
    D = R / L[:, None]
    w = X[ii] - A[kk]
    along, off = np.einsum("ij,ij->i", w, D[kk]), np.abs(_cross(D[kk], w))
    da, db = np.linalg.norm(w, axis=1), np.linalg.norm(X[ii] - B[kk], axis=1)
    hit = (off < ctol) & (along >= 0) & (along <= L[kk]) & (da >= ctol) & (db >= ctol)
    where.update(ii[hit].tolist()); where.update(E[kk[hit]].ravel().tolist())
    return len(pr), int(np.sum(hit))


class Faces(list):
    """A list of face walks with flat index arrays for vectorised signed areas."""

    def __init__(self, faces):
        super().__init__(list(f) for f in faces)
        size = np.array([len(f) for f in self], np.int64)
        self.idx = np.fromiter((v for f in self for v in f), np.int64, count=int(size.sum()))
        self.nxt = np.fromiter((v for f in self for v in f[1:] + f[:1]), np.int64, count=int(size.sum()))
        self.start = np.cumsum(size) - size

    def areas(self, X):
        if not len(self):
            return np.zeros(0)
        return 0.5 * np.add.reduceat(_cross(X[self.idx], X[self.nxt]), self.start)


def signed_areas(X, faces):
    return (faces if isinstance(faces, Faces) else Faces(faces)).areas(np.asarray(X, float))


# --- faces and rim ----------------------------------------------------------------------------------------
def has_frames(d) -> bool:
    return bool(d.edge) and all("ang" in e or "vec" in e for e in d.edge.values())


def _components(n, pairs):
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for u, v in pairs:
        parent[root(u)] = root(v)
    return root


def frame_faces(d):
    """(inner faces, outer faces) of fast.frame_faces; outer = the longest walk of each connected component."""
    F = _fast.frame_faces(d)
    root = _components(len(d.positions), d.edge)
    outer = {}
    for i, f in enumerate(F):
        c = root(f[0])
        if c not in outer or len(f) > len(F[outer[c]]):
            outer[c] = i
    skip = set(outer.values())
    return [f for i, f in enumerate(F) if i not in skip], [F[i] for i in sorted(skip)]


def _planar_outer_faces(d):
    """Frame-less graphs: outer faces of a combinatorial planar embedding (networkx), the longest face walk of
    each component, oriented as the given drawing has it. None if the graph is not planar."""
    import networkx as nx
    G = d.to_networkx()
    ok, emb = nx.check_planarity(G)
    if not ok:
        return None
    X = np.asarray(d.positions, float)
    seen, best = set(), {}
    root = _components(len(X), d.edges)
    for u, v in emb.edges():
        if (u, v) in seen:
            continue
        f = emb.traverse_face(u, v, mark_half_edges=seen)
        c = root(f[0])
        if c not in best or len(f) > len(best[c]):
            best[c] = f
    out = []
    for f in best.values():
        if len(f) > 2 and signed_areas(X, [f])[0] < 0:
            f = f[::-1]                                        # counter-clockwise, like a drawn outer boundary
        out.append(f)
    return out


# --- the check ----------------------------------------------------------------------------------------------
def check(d, X=None, plain_only=True, faces=None, types=None) -> dict:
    """Embeddedness of positions X (default d.positions). plain_only: only bonds without 'ghost' (the given
    picture, as drawing_check.measure counts); else all bonds at full length. faces: the inner faces
    (default: frame_faces(d)[0] when d has frames). types: edge_types(d) (computed when None). Returns
    crossings, overlaps, crowded (= crowded_pairs + crowded_on), folded (None when not measurable),
    bad = crossings + overlaps + crowded + (folded or 0), embedded = bad == 0, 'sites': the sites of the
    crossing bonds, overlaps and folded faces, and 'crowded_sites': those of the crowded pairs and sites."""
    X = np.asarray(d.positions if X is None else X, float)
    Eall = np.array(d.edges, np.int64).reshape(-1, 2)
    plain = np.array([not (plain_only and d.edge[(u, v)].get("ghost") is not None) for u, v in d.edges], bool)
    E = Eall[plain]
    A, B = X[E[:, 0]], X[E[:, 1]]
    lens = np.linalg.norm(B - A, axis=1)
    tol = REL_TOL * (float(np.median(lens)) if len(E) else 1.0)
    ctol = CROWD_REL * bond_unit(lens, (edge_types(d) if types is None else np.asarray(types))[plain])
    pairs = _near_pairs(A, B, tol)
    hits = crossing_pairs(A, B, E[:, 0], E[:, 1], pairs)
    where = set(E[hits.ravel()].ravel().tolist())
    tree = cKDTree(X)
    cand = _site_bond_candidates(X, E, A, B, np.maximum(lens, 1e-300), max(tol, ctol), tree) if len(E) else None
    col, on = overlap_count(X, E, tol, pairs, where, cand) if len(E) else (0, 0)
    csites = set()
    cp, con = crowd_count(X, E, ctol, csites, cand, tree)
    if faces is None and has_frames(d):
        faces = frame_faces(d)[0]
    folded = None
    if faces is not None:
        faces = faces if isinstance(faces, Faces) else Faces(faces)
        folded = 0
        if len(faces):
            a = faces.areas(X)
            s = np.sign(a)
            sign = 1.0 if np.sum(s > 0) >= np.sum(s < 0) else -1.0
            bad = (s != sign) | (np.abs(a) < AREA_TOL * np.median(np.abs(a)))
            if plain_only and not plain.all():
                h2 = crossing_pairs(X[Eall[:, 0]], X[Eall[:, 1]], Eall[:, 0], Eall[:, 1])
                ends = set(Eall[h2.ravel()].ravel().tolist())
            else:
                ends = set(E[hits.ravel()].ravel().tolist())
            for i, f in enumerate(faces):                      # a face crossing itself has two crossing bonds
                if not bad[i] and len(f) >= 4 and ends.intersection(f):
                    g = f[1:] + f[:1]
                    bad[i] = len(crossing_pairs(X[f], X[g], np.array(f), np.array(g))) > 0
            folded = int(bad.sum())
            for i in np.nonzero(bad)[0]:
                where.update(faces[i])
    tot = len(hits) + col + on + cp + con + (folded or 0)
    return dict(crossings=len(hits), overlaps=col + on, crowded=cp + con, crowded_pairs=cp, crowded_on=con,
                folded=folded, bad=tot, embedded=tot == 0, sites=where, crowded_sites=csites)
