"""Fast, result-identical versions of the legacy defect engines used by build.py and config.py.

No algorithm is changed. Legacy functions are re-bound (types.FunctionType on their own, unedited
code objects) to a globals dict in which only slow primitives are swapped for equivalent ones:
_add (the same np.allclose test on Python floats); Perfect -> FastPerfect (memoised pos returning
read-only arrays, pure-Python locate with _locate's strict 1e-6 test, vectorised sites_in_disk with
the same keys in the same order and the legacy test for sites within 1e-7 of the radius, memoised
is_symmetry); memoised primitive_cell / has_crossings; Defect -> FastDefect (faces and charged_faces
on Python floats, to_networkx -> FastGraph with a direct copy()); _ball materialised, not a view.
Re-implemented with the same operations: dislocation (dislocation_fast), frame_faces, _corners,
loop_charge, check_frames; new: irregular (sparse-matrix layer signatures, build.graph_core).

Reference path (definition, technical choice): PAPERTILLING_LEGACY=1 (read at call time) switches
every call to the legacy functions; fast=False switches only the call it is passed to (graph_core's
irregular sites and Configuration's readers follow the environment variable). Equality of the two paths: tests/test_build.py (Part C) and
tests/test_config.py (computational evidence).
"""
from __future__ import annotations

import math
import os
import types

import networkx as nx
import numpy as np

from ..legacy import fusion_framework as _ff
from ..legacy import lab_defects as _ld
from ..legacy import wallpaper_lattices as _wl


def enabled(fast: bool | None = None) -> bool:
    """fast=None: fast unless the environment variable PAPERTILLING_LEGACY is set to 1/true/yes."""
    if fast is not None:
        return bool(fast)
    return os.environ.get("PAPERTILLING_LEGACY", "").strip().lower() not in ("1", "true", "yes")


# --- _add: identical bookkeeping, scalar closeness test ---------------------------------------
def _close(a, b) -> bool:
    """np.allclose(a, b) (rtol 1e-5, atol 1e-8). Two 2-vectors: the same elementwise test in Python
    floats; a pass with finite b is final (then a is finite too and numpy agrees), anything else is
    decided by np.allclose itself."""
    if type(a) is np.ndarray and type(b) is np.ndarray and a.shape == (2,) and b.shape == (2,):
        a0, a1 = a.tolist()
        b0, b1 = b.tolist()
        if abs(a0 - b0) <= 1e-8 + 1e-5 * abs(b0) and abs(a1 - b1) <= 1e-8 + 1e-5 * abs(b1) and abs(b0) + abs(b1) < math.inf:
            return True
    elif type(a) is int and type(b) is int and abs(a) < 2 ** 52 and abs(b) < 2 ** 52:   # exact in float64
        return abs(a - b) <= 1e-8 + 1e-5 * abs(b)
    return bool(np.allclose(a, b))


def _add(edge, u, v, multi=None, **data):
    """lab_defects._add with the np.allclose consistency assert replaced by _close (same test)."""
    if multi is not None and (u, v) in edge and edge[(u, v)].get("j") != data.get("j"):
        multi.add((min(u, v), max(u, v))); return
    back = dict(data)
    for k in ("vec", "j"):
        if k in back:
            back[k] = -back[k]
    for key, val in (((u, v), data), ((v, u), back)):
        old = edge.get(key)
        if old is not None:
            for k in ("vec", "j"):
                if k in val:
                    assert _close(old[k], val[k]), f"two charts disagree on bond {key}"
        edge[key] = val


# --- Perfect lattice with memoised / vectorised primitives ------------------------------------
class FastPerfect(_ld.Perfect):
    def __init__(self, cell):
        super().__init__(cell)
        self._pos: dict = {}
        self._sym: dict = {}
        self._Ai = [[float(x) for x in row] for row in self.Ainv]
        self._B = [(float(b[0]), float(b[1])) for b in np.asarray(cell.basis, float)]

    def pos(self, key):
        p = self._pos.get(key)
        if p is None:
            p = self.A @ np.array(key[:2], float) + self.cell.basis[key[2]]     # the legacy expression
            p.setflags(write=False)
            self._pos[key] = p
        return p

    def locate(self, p):
        """Same test as wallpaper_lattices._locate (first sublattice s with |f - round(f)| < 1e-6)."""
        px, py = float(p[0]), float(p[1])
        (a00, a01), (a10, a11) = self._Ai
        for s, (bx, by) in enumerate(self._B):
            dx, dy = px - bx, py - by
            f0 = a00 * dx + a01 * dy
            f1 = a10 * dx + a11 * dy
            r0, r1 = round(f0), round(f1)
            if abs(f0 - r0) < 1e-6 and abs(f1 - r1) < 1e-6:
                return (int(r0), int(r1), s)
        return None

    def sites_in_disk(self, centre, R):
        c = np.asarray(centre, float)
        n0 = np.round(self.Ainv @ c).astype(int)
        h = abs(np.linalg.det(self.A)) / max(np.linalg.norm(self.A[:, 0]), np.linalg.norm(self.A[:, 1]))
        K = int(math.ceil(R / h)) + 2
        n1 = np.arange(n0[0] - K, n0[0] + K + 1)
        n2 = np.arange(n0[1] - K, n0[1] + K + 1)
        ns = len(self.cell.basis)
        g1, g2, gs = np.meshgrid(n1, n2, np.arange(ns), indexing="ij")         # legacy loop order n1, n2, s
        g1, g2, gs = g1.ravel(), g2.ravel(), gs.ravel()
        basis = np.asarray(self.cell.basis, float)
        P = np.outer(g1, self.A[:, 0]) + np.outer(g2, self.A[:, 1]) + basis[gs] - c
        r = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
        lim = R + 1e-9
        out = []
        for idx in np.nonzero(r <= lim + 1e-7)[0]:
            key = (int(g1[idx]), int(g2[idx]), int(gs[idx]))
            if r[idx] < lim - 1e-7 or np.linalg.norm(self.pos(key) - c) <= lim:   # borderline: legacy test
                out.append(key)
        return out

    def is_symmetry(self, g, probe=3.0):
        M, t = g
        key = (np.asarray(M, float).tobytes(), np.asarray(t, float).tobytes(), float(probe))
        if key not in self._sym:
            self._sym[key] = super().is_symmetry(g, probe)
        return self._sym[key]


_MEMO: dict = {}


def _memo(fn, cell, *args):
    """fn(cell, *args), computed once per cell object (a strong reference to the cell is kept)."""
    k = (fn, id(cell), args)
    hit = _MEMO.get(k)
    if hit is None or hit[0] is not cell:
        hit = _MEMO[k] = (cell, fn(cell, *args))
    return hit[1]


def perfect(cell) -> FastPerfect:
    """One FastPerfect per cell object."""
    return _memo(FastPerfect, cell)


def _primitive_cell(cell, use_types=True):
    return _memo(_wl.primitive_cell, cell, use_types)


def _has_crossings(cell):
    return _memo(_wl.has_crossings, cell)


def has_crossings_fast(cell) -> bool:
    """wallpaper_lattices.has_crossings: the same _segments_cross test (tol 1e-9) on every bond pair
    and every shift n in [-5, 5]^2 (shift vectors by the legacy expression A @ n), elementwise in numpy."""
    A = cell.A
    segs = [(cell.basis[i], cell.basis[j] + A @ np.array([n1, n2])) for (i, j, n1, n2, _) in cell.bonds]
    if not segs:
        return False
    P = np.array([p for p, _ in segs]); Q = np.array([q for _, q in segs])
    shifts = [(n, A @ np.array(n)) for n in __import__("itertools").product(range(-5, 6), repeat=2)]
    m = len(segs)
    k, l = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    k, l = k.ravel(), l.ravel()
    tol = 1e-9
    for n, sh in shifts:
        kk, ll = (k[l > k], l[l > k]) if n == (0, 0) else (k, l)
        p, q, r, s = P[kk], Q[kk], P[ll] + sh, Q[ll] + sh
        a, b_ = q - p, s - r
        den = a[:, 0] * b_[:, 1] - a[:, 1] * b_[:, 0]
        ok = ~(np.abs(den) < tol)
        rp = r - p
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (rp[:, 0] * b_[:, 1] - rp[:, 1] * b_[:, 0]) / den
            v = (rp[:, 0] * a[:, 1] - rp[:, 1] * a[:, 0]) / den
        if np.any(ok & (tol < u) & (u < 1 - tol) & (tol < v) & (v < 1 - tol)):
            return True
    return False


# --- networkx: a Graph whose copy() builds the same dicts directly; materialised balls --------
class FastGraph(nx.Graph):
    """nx.Graph whose copy() builds the dicts of nx.Graph.copy (networkx 3.x: same node order,
    neighbour order, one data dict per edge) directly, without the per-edge add_edge overhead."""

    def copy(self, as_view=False):
        if as_view:
            return super().copy(as_view=True)
        G = self.__class__()
        G.graph.update(self.graph)
        G._node = {n: dict(d) for n, d in self._node.items()}
        adj = {n: {} for n in G._node}
        for u, nbrs in self._adj.items():
            au = adj[u]
            for v, dd in nbrs.items():
                x = au.get(v)
                if x is None:
                    x = dict(dd)
                    au[v] = x
                    adj[v][u] = x
                else:
                    x.update(dd)
        G._adj = adj
        return G


def _ball(G, core, radius):
    """lab_defects._ball (subgraph on the sites within `radius` bonds of `core`) materialised, not a
    view: same nodes and edges; callers only read it (hash, isomorphism, bipartite, copy: order-free)."""
    nodes = set()
    for v in core:
        nodes |= set(nx.single_source_shortest_path_length(G, v, cutoff=radius))
    if not isinstance(G, FastGraph):
        return G.subgraph(nodes)
    H = FastGraph()
    H.graph.update(G.graph)
    H._node = {n: dict(G._node[n]) for n in G._node if n in nodes}
    adj = {n: {} for n in H._node}
    for u in H._node:
        au = adj[u]
        for v, dd in G._adj[u].items():
            if v in nodes:
                x = adj[v].get(u)
                au[v] = x if x is not None else dict(dd)
    H._adj = adj
    return H


# --- Defect with a faster charged_faces (same faces, same sequential sums) --------------------
class FastDefect(_ld.Defect):
    """lab_defects.Defect; rotation system, faces and charged_faces in Python floats (positions via
    tolist(): the same IEEE operations as on numpy float64 scalars, so the same faces in the same order)."""

    def _rotation_system(self):
        if self._nbr is None:
            P = self.positions.tolist()
            nbr = {k: [] for k in range(len(self.sites))}
            for u, v in self.edges:
                nbr[u].append(v); nbr[v].append(u)
            for u in nbr:
                pu = P[u]
                nbr[u].sort(key=lambda v: math.atan2(P[v][1] - pu[1], P[v][0] - pu[0]))
            self._nbr = nbr
        return self._nbr

    def faces(self):
        nbr, P = self._rotation_system(), self.positions.tolist()
        seen, out = set(), []
        for u in nbr:
            for v in nbr[u]:
                if (u, v) in seen:
                    continue
                f, a, b = [], u, v
                while (a, b) not in seen:
                    seen.add((a, b)); f.append(a)
                    k = nbr[b].index(a)
                    a, b = b, nbr[b][k - 1]
                n = len(f)
                area = 0.5 * sum(P[f[i]][0] * P[f[(i + 1) % n]][1] - P[f[i]][1] * P[f[(i + 1) % n]][0] for i in range(n))
                if area > 1e-12:
                    out.append(f)
        return out

    def to_networkx(self):
        G = FastGraph()
        G.add_nodes_from(range(len(self.sites)))
        G.add_edges_from(self.edges)
        return G

    def charged_faces(self, field):
        out = []
        E = self.edge
        for f in self.faces():
            n = len(f)
            if field == "vec":
                sx = sy = 0.0
                for i in range(n):
                    vx, vy = E[(f[i], f[(i + 1) % n])]["vec"].tolist()
                    sx += vx; sy += vy
                h = math.hypot(sx, sy)          # legacy: np.linalg.norm > 1e-9 (asked of numpy near 1e-9)
                if h > 1e-8 or (h > 1e-10 and np.linalg.norm(np.array([sx, sy])) > 1e-9):
                    out.append((f, np.array([sx, sy])))
            else:
                v = self.loop_sum(f, field)
                if np.linalg.norm(np.atleast_1d(v)) > 1e-9:
                    out.append((f, v))
        return out


# --- Re-bound legacy functions ----------------------------------------------------------------
_G = dict(vars(_ld))
_G.update(_add=_add, Perfect=perfect, primitive_cell=_primitive_cell, has_crossings=_has_crossings, Defect=FastDefect,
          _ball=_ball)


def _rebind(f, G=None):
    G = _G if G is None else G
    g = types.FunctionType(f.__code__, G, f.__name__, f.__defaults__, f.__closure__)
    g.__kwdefaults__ = f.__kwdefaults__
    g.__doc__ = f"[fast re-binding of legacy {f.__module__}.{f.__name__}]\n" + (f.__doc__ or "")
    G[f.__name__] = g
    return g


def dislocation_fast(cell, b, core=(0.0, 0.0), cut=None, R=6.0):
    """lab_defects.dislocation re-implemented with the same operations in the same order (Volterra
    cut and reglue, three charts). Differences in form only: (t, s) of all sites at once (rows within
    1e-7 of a chart boundary are recomputed with the legacy per-site expression); chart images by
    integer key arithmetic (key - m, b = A m: what L.locate returns for a lattice translation b);
    _add inlined on Python floats with the same consistency test; vec arrays made at the end.
    Output: the same sites, positions, bond dict (key order, values) as legacy (tests/test_config.py)."""
    L = perfect(cell)
    b = np.asarray(b, float); core = np.asarray(core, float)
    assert L.is_symmetry((np.eye(2), b)), "b is not a lattice translation"
    d = np.array([b[1], -b[0]]) / np.linalg.norm(b) if cut is None else np.asarray(cut, float)
    B = np.column_stack([b, d]); Binv = np.linalg.inv(B)
    allk = L.sites_in_disk(core, R)
    TS = (np.array([L.pos(k) for k in allk]).reshape(-1, 2) - core) @ Binv.T
    near = np.nonzero((np.abs(TS[:, 0] + 1e-9) < 1e-7) | (np.abs(TS[:, 0] - (1 - 1e-9)) < 1e-7)
                      | (np.abs(TS[:, 1] - 1e-9) < 1e-7))[0]
    TS = TS.tolist()
    for i in near:
        TS[i] = (Binv @ (L.pos(allk[i]) - core)).tolist()
    keys, ts = [], []
    for k, (t, s_) in zip(allk, TS):
        if s_ > 1e-9 and -1e-9 <= t < 1 - 1e-9:
            continue                                            # removed half-strip
        keys.append(k); ts.append((t, s_))
    m1, m2 = (int(x) for x in np.round(L.Ainv @ b))
    posl, E = {}, {}
    for c in range(3):
        where = {}
        for u, (k, (t, s_)) in enumerate(zip(keys, ts)):
            if c == 0:
                ok, sh = s_ <= 1e-9 or t < -1e-9, 0
            elif c == 1:
                ok, sh = t >= 1 - 1e-9, -1
            else:
                ok, sh = s_ > 1e-9, (-1 if t >= 1 - 1e-9 else 0)
            if ok:
                where[(k[0] + sh * m1, k[1] + sh * m2, k[2]) if sh else k] = u
        for img, u in where.items():
            pi = posl.get(img)
            if pi is None:
                pi = posl[img] = L.pos(img).tolist()
            for k2, typ in L.nbrs(img):
                v = where.get(k2)
                if v is None or v == u:
                    continue
                p2 = posl.get(k2)
                if p2 is None:
                    p2 = posl[k2] = L.pos(k2).tolist()
                x, y = p2[0] - pi[0], p2[1] - pi[1]
                for key, vx, vy in (((u, v), x, y), ((v, u), -x, -y)):
                    old = E.get(key)                    # legacy _add: assert np.allclose(old, new)
                    if old is not None and not (abs(old[0] - vx) <= 1e-8 + 1e-5 * abs(vx)
                                                and abs(old[1] - vy) <= 1e-8 + 1e-5 * abs(vy)):
                        if not np.allclose(np.array(old[:2]), np.array([vx, vy])):
                            raise AssertionError(f"two charts disagree on bond {key}")
                    E[key] = (vx, vy, typ)
    edge = {key: dict(vec=np.array([x, y]), type=typ) for key, (x, y, typ) in E.items()}
    P = np.array([L.pos(k) for k in keys])                      # drawing: the legacy expressions
    sgn = 1.0 if _ld.cross2(d, b) > 0 else -1.0
    r = P - (core + 0.5 * b)
    dh = d / np.linalg.norm(d)
    ang = np.arctan2(sgn * (dh[0] * r[:, 1] - dh[1] * r[:, 0]), r @ dh) % _ld.TWO_PI
    P = P - np.outer(1 - ang / _ld.TWO_PI, b)
    return FastDefect("dislocation", keys, P, edge, dict(b=b, core=core, cut=d))


_G["dislocation"] = dislocation_fast
perfect_patch_fast = _rebind(_ld.perfect_patch)
disclination_fast = _rebind(_ld.disclination)            # used by the gate of tests/test_build.py
dislocation_cores_fast = _rebind(_ld.dislocation_cores)


def dislocation_cores(*args, fast: bool | None = None, **kw):
    return (dislocation_cores_fast if enabled(fast) else _ld.dislocation_cores)(*args, **kw)


# --- Reading a crystal: faces, corners, loop holonomy (same operations, fewer numpy calls) ----
def _corners_fast(d):
    """fusion_framework._corners: per site, (bond angle, neighbour) sorted; 'vec' read via tolist()."""
    nbr: dict = {}
    for (u, v), e in d.edge.items():
        if "ang" in e:
            a = e["ang"]
        else:
            vec = e["vec"]
            vx, vy = vec.tolist() if type(vec) is np.ndarray else (float(vec[0]), float(vec[1]))
            a = math.atan2(vy, vx) % _ld.TWO_PI
        nbr.setdefault(u, []).append((a, v))
    return {u: sorted(lst) for u, lst in nbr.items()}


def frame_faces_fast(d):
    """lab_defects.frame_faces: the same rotation system (from _corners_fast) and the same walk; the
    position of a in nbr[b] is looked up in a dict (neighbours of a site are distinct)."""
    nbr = {u: [v for _, v in lst] for u, lst in _corners_fast(d).items()}
    where = {u: {v: k for k, v in enumerate(lst)} for u, lst in nbr.items()}
    seen, faces = set(), []
    for u in nbr:
        for v in nbr[u]:
            if (u, v) in seen:
                continue
            f, a, b = [], u, v
            while (a, b) not in seen:
                seen.add((a, b)); f.append(a)
                k = where[b][a]
                a, b = b, nbr[b][k - 1]
            faces.append(f)
    return faces


def loop_charge_fast(d, loop):
    """lab_defects.loop_charge; a bond without 'g' (frame change = identity) skips the two
    compositions with the identity, which leave Dm and H unchanged (exact up to the sign of zeros)."""
    from ..legacy.lab_defects import iso_compose, iso_inverse
    Dm = (np.eye(2), np.zeros(2)); H = (np.eye(2), np.zeros(2)); S = np.zeros(2)
    n = len(loop)
    for k in range(n):
        e = d.edge[(loop[k], loop[(k + 1) % n])]
        S = S + Dm[0] @ e["vec"]
        g = e.get("g")
        if g is not None:
            Dm = iso_compose(Dm, iso_inverse(g)); H = iso_compose(g, H)
    return H, S


_FG = dict(vars(_ff))
_FG.update(frame_faces=frame_faces_fast, _corners=_corners_fast, loop_charge=loop_charge_fast)
for _f in (_ff.frank_angles, _ff.total_frank, _ff.total_charge, _ff.cores):
    _rebind(_f, _FG)


def frame_faces(d, fast: bool | None = None):
    return frame_faces_fast(d) if enabled(fast) else _ld.frame_faces(d)


def total_frank(d, fast: bool | None = None) -> float:
    return (_FG["total_frank"] if enabled(fast) else _ff.total_frank)(d)


def total_charge(d, fast: bool | None = None):
    return (_FG["total_charge"] if enabled(fast) else _ff.total_charge)(d)


def cores(d, fast: bool | None = None) -> list:
    return (_FG["cores"] if enabled(fast) else _ff.cores)(d)


# --- Builder, frame check, relaxation switches used by build.py -------------------------------
_BUILDER = []


def builder_class():
    """build._Builder with one faster method, snapshot (created on first use: import cycle).
    snapshot: the legacy one copies every position array; none of them is ever written in place
    (checked: legacy _ConeSafeBuilder and build._Builder only rebind self.X[i] / self.ref[i]), so the
    lists are copied and the arrays shared; they are made read-only, so an in-place write would raise."""
    if not _BUILDER:
        from collections import Counter
        from .build import _Builder

        class FastBuilder(_Builder):
            def snapshot(self):
                for x in self.ref:
                    x.flags.writeable = False
                for x in self.X:
                    x.flags.writeable = False
                return (list(self.label), list(self.ref), list(self.alive),
                        [{v: dict(e) for v, e in n.items()} for n in self.nb], list(self.X),
                        Counter(self.layers), set(self.touched), set(self.cone_sites), list(self.records))
        _BUILDER.append(FastBuilder)
    return _BUILDER[0]


def check_frames_fast(d) -> int:
    """fusion_framework.check_frames with the identity-frame bonds (no 'g' either way) done in floats:
    there iso_apply(I, x) = x and iso_compose(I, I) = I exactly, so the test is the same two
    math.hypot (= _norm2) comparisons; bonds with a frame change 'g' use the legacy expressions."""
    ff = _ff
    from ..legacy.lab_defects import iso_apply, iso_compose
    X = d.info["reference"]
    Xl = np.asarray(X, float).tolist()
    wr = d.info.get("wraps") or {}
    ident = ff._ident()
    bad = 0
    for (u, v), e in d.edge.items():
        back = d.edge[(v, u)]
        if "g" not in e and "g" not in back:
            v0, v1 = e["vec"].tolist()
            w0, w1 = back["vec"].tolist()
            wv = wr.get((u, v), 0.0)
            xu, xv = Xl[u], Xl[v]
            if isinstance(wv, float):
                b0, b1 = xv[0] + wv, xv[1] + wv
            else:
                wv0, wv1 = np.asarray(wv, float).tolist()
                b0, b1 = xv[0] + wv0, xv[1] + wv1
            ok = math.hypot((xu[0] + v0) - b0, (xu[1] + v1) - b1) < 1e-5 and math.hypot(w0 + v0, w1 + v1) < 1e-5
        else:
            g = e.get("g", ident)
            ok = ff._norm2(iso_apply(g, X[u] + e["vec"]) - (X[v] + wr.get((u, v), 0.0))) < 1e-5
            ok = ok and ff._same_iso(iso_compose(back.get("g", ident), g), ident, tol=1e-5)
            ok = ok and ff._norm2(back["vec"] + g[0] @ e["vec"]) < 1e-5
        if not ok:
            bad += 1
    return bad


def check_frames(d, fast: bool | None = None) -> int:
    return check_frames_fast(d) if enabled(fast) else _ff.check_frames(d)


def relax_picture(d, fast: bool | None = None):
    from .fast_relax import relax_picture as rp
    return rp(d, fast=fast)


# --- Irregular sites (build.graph_core): layer_signature(v) not in perfect_sigs, all v at once -----
_PCODES: dict = {}


def _encode(sig) -> list:
    """(depth, degree, a, b, c) -> one integer, base 256, most significant first: injective and
    order-preserving while every entry is < 256, so sorted tuples <-> sorted codes."""
    return sorted((((d * 256 + g) * 256 + a) * 256 + b) * 256 + c for d, g, a, b, c in sig)


def _irregular_sparse(adj, deg, cand, r, pcodes, plen) -> np.ndarray:
    """Boolean mask over cand. Sparse-matrix form of build.layer_signature for all candidates at once:
    D_k[v, u] = 1 iff dist(v, u) = k (k <= r, breadth-first layers from the candidate v); then
    S_k = D_k A counts, for each u, its neighbours at distance k from v, so the entries
    (k, deg u, a, b, c) of v's signature are a = S_{k-1}[v, u], b = S_k[v, u], c = S_{k+1}[v, u]
    (c = 0 for k = r: neighbours outside the ball are not counted, as in layer_signature)."""
    import scipy.sparse as sp
    n, m = len(adj), len(cand)
    indptr = np.zeros(n + 1, np.int64)
    indptr[1:] = np.cumsum([len(a) for a in adj])
    indices = np.fromiter((w for a in adj for w in a), np.int64, count=int(indptr[-1]))
    A = sp.csr_matrix((np.ones(len(indices), np.int64), indices, indptr), shape=(n, n))
    D = [sp.csr_matrix((np.ones(m, np.int64), (np.arange(m), np.asarray(cand, np.int64))), shape=(m, n))]
    seen = D[0]
    for _ in range(r):
        nxt = (D[-1] @ A).tocsr()
        nxt.data[:] = 1
        new = (nxt - nxt.multiply(seen)).tocsr()
        new.eliminate_zeros()
        D.append(new)
        seen = (seen + new).tocsr()
    S = [(Dk @ A).tocsr() for Dk in D]
    degs = np.asarray(deg, np.int64)
    R, C = [], []
    for k in range(r + 1):
        rv, cu = D[k].nonzero()
        z = np.zeros(len(rv), np.int64)
        a = np.asarray(S[k - 1][rv, cu], np.int64).ravel() if k else z
        b = np.asarray(S[k][rv, cu], np.int64).ravel()
        c = np.asarray(S[k + 1][rv, cu], np.int64).ravel() if k < r else z
        R.append(rv)
        C.append((((k * 256 + degs[cu]) * 256 + a) * 256 + b) * 256 + c)
    R, C = np.concatenate(R), np.concatenate(C)
    o = np.lexsort((C, R))
    R, C = R[o], C[o]
    cnt = np.bincount(R, minlength=m)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    match = np.zeros(m, bool)
    for p in range(len(plen)):
        L = int(plen[p])
        rows = np.nonzero(cnt == L)[0]
        if len(rows):
            match[rows] |= np.all(C[start[rows][:, None] + np.arange(L)] == pcodes[p, :L], axis=1)
    return ~match


def irregular(adj, deg, cand, r, perfect_sigs, fast: bool | None = None) -> list:
    """[v in cand with build.layer_signature(adj, deg, v, r) not in perfect_sigs] (definition of an
    irregular site, build.py). fast: _irregular_sparse (exact: the same multiset, encoded by _encode);
    fast=False / PAPERTILLING_LEGACY=1 / any degree or r >= 256: build.layer_signature itself."""
    from .build import layer_signature
    if not enabled(fast) or not len(cand) or max(deg) >= 256 or r >= 256:
        return [v for v in cand if layer_signature(adj, deg, v, r) not in perfect_sigs]
    key = frozenset(perfect_sigs)
    if key not in _PCODES:
        codes = [_encode(s) for s in perfect_sigs]
        pc = np.zeros((len(codes), max([len(c) for c in codes] + [1])), np.int64)
        for i, c in enumerate(codes):
            pc[i, :len(c)] = c
        _PCODES[key] = (pc, np.array([len(c) for c in codes], np.int64))
    mask = _irregular_sparse(adj, deg, cand, int(r), *_PCODES[key])
    return [v for v, m in zip(cand, mask) if m]
