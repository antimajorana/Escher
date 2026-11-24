"""Seeded growth of the regular honeycombs of E³, H³ and S³ (C1 WP-G2; mirrors
`papertilling-core/src/growth/h3_complex.rs` and `honeycomb3d/growth_build.rs`).

    from papertilling import honeycomb3d as hc
    g = hc.grow_honeycomb("{4,3,5}", "cell", layers=1)
    [s["cells"] for s in g["stats"]]          # [1, 111]: 1 + 110 (hand formula, plan §2.4)

Rules: "touching" (every cell at a vertex of S_k) and "face_shells" (every cell sharing a face).
Refusals: hyperideal or degenerate symbols raise ValueError with the [PROVED] reason; infinite cells
stop at layer 0, a vertex seed at an ideal vertex at layer 0, other seeds at ideal vertices at layer 1.

Parity with the Rust core is on invariant output only: per-layer `stats`, `stopped`, `name`,
`regime`, `seed`, `rule`, `boundary_only`, the drawn complex's counts, and the interned key counts.
The two realisations of the Coxeter group are the same LDLᵀ construction, but ids, vertex
coordinates and the measured merge error may differ by round-off.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np

from . import _rust
from ._growth_engine import GrowthError, grow

MAX_GROWTH_CELLS = 120_000   # Rust growth_build.rs: Q13 cap lowered by the round-7 timing rule
MAX_GROWTH_LAYERS = 10
DRAW_ALL_FACES_MAX = 10_000
KEY_TOL = 1e-4
LOOKAHEAD = 2 ** 32 - 1
FACE_TAG = 1 << 31
IDEAL_VERTEX_REASON = ("[PROVED] an ideal vertex lies in infinitely many cells: its vertex figure {q,r} "
                       "is a Euclidean tiling ((q-2)(r-2) = 4)")
INFINITE_CELL_REASON = ("[PROVED] every cell is an infinite horospherical tiling "
                        "((p-2)(q-2) = 4), so a cell has infinitely many vertices and no layer can be built whole")


class KeyIndex:
    """Tolerance index with a measured margin (mirrors `honeycomb3d/key_index.rs`)."""

    def __init__(self, dims, tol):
        self.dims, self.tol, self.probe, self.side = dims, tol, 100 * tol, 1000 * tol
        self.buckets: dict = {}
        self.pts: list = []
        self.max_merge, self.min_gap = 0.0, math.inf

    def _nearest(self, x):
        lo = [math.floor((x[d] - self.probe) / self.side) for d in range(self.dims)]
        hi = [math.floor((x[d] + self.probe) / self.side) for d in range(self.dims)]
        best = None
        for k in itertools.product(*[range(a, b + 1) for a, b in zip(lo, hi)]):
            for i in self.buckets.get(k, ()):
                p = self.pts[i]
                dist = math.sqrt(sum((p[d] - x[d]) ** 2 for d in range(self.dims)))
                if dist < self.probe and (best is None or dist < best[1]):
                    best = (i, dist)
        return best

    def insert(self, x):
        nb = self._nearest(x)
        if nb is not None:
            if nb[1] <= self.tol:
                self.max_merge = max(self.max_merge, nb[1])
                return nb[0], False
            self.min_gap = min(self.min_gap, nb[1])
        i = len(self.pts)
        self.buckets.setdefault(tuple(math.floor(x[d] / self.side) for d in range(self.dims)), []).append(i)
        self.pts.append(tuple(float(t) for t in x[:self.dims]))
        return i, True

    def lookup(self, x):
        nb = self._nearest(x)
        if nb is None:
            return None
        if nb[1] <= self.tol:
            self.max_merge = max(self.max_merge, nb[1])
            return nb[0]
        self.min_gap = min(self.min_gap, nb[1])
        return None

    def find(self, x):
        nb = self._nearest(x)
        return nb[0] if nb is not None and nb[1] <= self.tol else None


def ldl_realisation(p, q, r):
    """[p,q,r] from G = L D Lᵀ, coordinates (a3, a0, a1, a2); mirrors `coxeter_ldl.rs`."""
    c = [-math.cos(math.pi / m) for m in (p, q, r)]
    d, l = [1.0] * 4, [0.0] * 3
    for i in range(3):
        l[i] = c[i] / d[i]
        d[i + 1] = 1 - l[i] * l[i] * d[i]
    if min(d[:3]) <= 1e-12 or abs(d[3]) <= 1e-12:
        raise ValueError(f"[{p},{q},{r}] has no LDL realisation with a finite cell group")
    lorentz = d[3] < 0
    s = [math.sqrt(abs(x)) for x in d]
    normals = np.zeros((4, 4))
    for j in range(4):
        a = [0.0] * 4
        a[j] = s[j]
        if j:
            a[j - 1] = l[j - 1] * s[j - 1]
        normals[j] = [a[3], a[0], a[1], a[2]]
    eta = np.array([-1.0, 1, 1, 1]) if lorentz else np.ones(4)
    refl = [np.eye(4) - 2 * np.outer(n, eta * n) for n in normals]
    w = np.linalg.inv(normals * eta).T.copy()
    for j in range(4):
        n2 = float(np.sum(eta * w[j] * w[j]))
        if not lorentz:
            w[j] /= math.sqrt(n2)
        elif n2 < -1e-9 * w[j][0] ** 2:
            w[j] /= math.sqrt(-n2)
        else:
            w[j] /= abs(w[j][0])
    if lorentz and w[3][0] < 0:
        w = -w
    w[3] = [1.0, 0, 0, 0] if lorentz or w[3][0] > 0 else [-1.0, 0, 0, 0]
    return eta, normals, refl, w


def _finite_group(gens, xi):
    idx = KeyIndex(4, 1e-7)
    idx.insert(xi)
    out, i = [np.eye(4)], 0
    while i < len(out):
        for s in gens:
            h = out[i] @ s
            if idx.insert(h @ xi)[1]:
                out.append(h)
        i += 1
        if len(out) > 20000:
            raise RuntimeError("stabiliser is not finite")
    return out


def _three_smallest(it):
    return tuple(sorted(it)[:3])


class CoxeterComplex:
    """Coset adapter for S³ and H³ {p,q,r} (mirrors `growth::h3_complex::CoxeterComplex`)."""

    def __init__(self, p, q, r):
        from .honeycomb3d import _py_classify
        cls = _py_classify(p, q, r)
        reg = cls["regime"]
        if reg == "EuclideanE3":
            raise ValueError("{4,3,4} is Euclidean: grow it with the E3 adapter (cubic)")
        if reg not in ("SphericalS3", "HyperbolicCompactH3", "HyperbolicParacompactH3"):
            raise ValueError(cls["reason"])
        if cls["has_infinite_euclidean_cells"]:
            raise ValueError(INFINITE_CELL_REASON)
        self.pqr, self.spherical, self.ideal = (p, q, r), reg == "SphericalS3", cls["has_ideal_vertices"]
        eta, _, rr, w = ldl_realisation(p, q, r)
        self.w = w
        xi = w.sum(axis=0)

        def key(x, vertex=False):
            if vertex and self.ideal and not self.spherical:
                s = x[1:] / x[0]
                return np.append(s / np.linalg.norm(s), 0.0)
            return self._key(x)

        def reps(gens, base, vertex=False):
            idx, mats, pts = KeyIndex(4, 1e-7), [], []
            for g in _finite_group(gens, xi):
                x = g @ base
                if idx.insert(key(x, vertex))[1]:
                    mats.append(g)
                    pts.append(x)
            return mats, np.array(pts).T

        cg = [rr[0], rr[1], rr[2]]
        self.cv, self.cv_pts = reps(cg, w[0], True)
        self.ce, _ = reps(cg, w[1])
        self.cf, _ = reps(cg, w[2])
        local = KeyIndex(4, 1e-7)
        for x in self.cv_pts.T:
            local.insert(key(x, True))
        rot = rr[0] @ rr[1]
        self.face_cycles = []
        for h in self.cf:
            g, cyc = h, []
            for _ in range(p):
                cyc.append(local.find(key(g @ w[0], True)))
                g = g @ rot
            self.face_cycles.append(cyc)
        self.edge_ends = [(local.find(key(h @ w[0], True)), local.find(key(h @ rr[0] @ w[0], True))) for h in self.ce]
        self.vc = None if self.ideal else reps([rr[1], rr[2], rr[3]], w[3])
        self.ec = reps([rr[0], rr[2], rr[3]], w[3])
        self.fc = reps([rr[0], rr[1], rr[3]], w[3])
        dims = 4 if self.spherical else 3
        self.cells, self.verts = KeyIndex(dims, KEY_TOL), KeyIndex(dims, KEY_TOL)
        self.cell_src, self.vert_src, self.edge_src, self.face_src = [], [], [], []
        self.mats: dict = {}
        self.edge_ids: dict = {}
        self.face_ids: dict = {}
        self.step, self.querying, self.final_step, self.force = 0, False, None, False
        self.cap = None   # cap guard: star queries answer by lookup once more cells are interned
        self._intern_cell(w[3], ("root", 0), 0)
        self._materialise(0)

    def _key(self, x):
        return x / np.linalg.norm(x) if self.spherical else np.array([x[1], x[2], x[3], 0.0])

    def _vkey(self, x):
        if self.ideal and not self.spherical:
            s = x[1:] / x[0]
            return np.append(s / np.linalg.norm(s), 0.0)
        return self._key(x)

    def _intern_cell(self, x, src, j):
        i, new = self.cells.insert(self._key(x))
        if new:
            self.cell_src.append((src, j))
        return i

    def _cell_mat(self, c):
        if c in self.mats:
            return self.mats[c][0]
        (kind, ref), j = self.cell_src[c]
        if kind == "root":
            return np.eye(4)
        if kind == "v":
            return self._vert_mat(ref) @ self.vc[0][j]
        if kind == "e":
            c2, i = self.edge_src[ref]
            return self._cell_mat(c2) @ self.ce[i] @ self.ec[0][j]
        c2, i = self.face_src[ref]
        return self._cell_mat(c2) @ self.cf[i] @ self.fc[0][j]

    def _vert_mat(self, v):
        c, i = self.vert_src[v]
        return self._cell_mat(c) @ self.cv[i]

    def _materialise(self, c):
        if c in self.mats:
            return
        m = self._cell_mat(c)
        pts = (m @ self.cv_pts).T
        v = []
        for i, x in enumerate(pts):
            vid, new = self.verts.insert(self._vkey(x))
            if new:
                self.vert_src.append((c, i))
            v.append(vid)
        e = []
        for i, (a, b) in enumerate(self.edge_ends):
            k = (min(v[a], v[b]), max(v[a], v[b]))
            if k not in self.edge_ids:
                self.edge_ids[k] = len(self.edge_src)
                self.edge_src.append((c, i))
            e.append(self.edge_ids[k])
        f = []
        for i, cyc in enumerate(self.face_cycles):
            k = _three_smallest(v[t] for t in cyc)
            if k not in self.face_ids:
                self.face_ids[k] = len(self.face_src)
                self.face_src.append((c, i))
            f.append(self.face_ids[k])
        self.mats[c] = (m, v, e, f)

    def _note(self):
        if self.querying:
            self.step += 1
            self.querying = False

    def _check(self):
        gap = min(self.cells.min_gap, self.verts.min_gap)
        if math.isfinite(gap):
            raise GrowthError("precision", gap=gap)

    def _cells_from(self, src, m, reps):
        self._check()
        self.querying = True
        pts = (m @ reps[1]).T
        over_cap = self.cap is not None and len(self.cells.pts) > self.cap
        if self.force or over_cap or self.final_step == self.step:
            out = [self.cells.lookup(self._key(x)) for x in pts]
            out = [LOOKAHEAD if c is None else c for c in out]
        else:
            out = [self._intern_cell(x, src, j) for j, x in enumerate(pts)]
        self._check()
        return out

    # -- the CellComplex interface used by `_growth_engine.grow`
    def vertices_of(self, c):
        self._note()
        self._materialise(c)
        return list(self.mats[c][1])

    def edges_of(self, c):
        self._note()
        self._materialise(c)
        return list(self.mats[c][2])

    def faces_of(self, c):
        self._note()
        self._materialise(c)
        return [f | FACE_TAG for f in self.mats[c][3]]

    def on_rim(self, v):
        return False

    def cells_at(self, v):
        if self.vc is None:
            raise GrowthError("infinite", IDEAL_VERTEX_REASON)
        return self._cells_from(("v", v), self._vert_mat(v), self.vc)

    def cells_on_edge(self, e):
        if e & FACE_TAG or e >= len(self.edge_src):
            raise GrowthError("out_of_range", "not an interned edge key")
        c, i = self.edge_src[e]
        return self._cells_from(("e", e), self._cell_mat(c) @ self.ce[i], self.ec)

    def cells_on_face_key(self, f):
        fid = f & ~FACE_TAG
        if not f & FACE_TAG or fid >= len(self.face_src):
            raise GrowthError("out_of_range", "not a face key")
        c, i = self.face_src[fid]
        return self._cells_from(("f", fid), self._cell_mat(c) @ self.cf[i], self.fc)

    def cells_on_face(self, f):
        fid = self.face_ids.get(_three_smallest(f))
        if fid is None:
            raise GrowthError("out_of_range", "no interned face on these vertices")
        return self.cells_on_face_key(fid | FACE_TAG)

    def lookup_query(self, fn, arg):
        self.force = True
        try:
            return fn(arg)
        finally:
            self.force = False

    def default_seed(self, seed):
        _, v, e, _ = self.mats[0]
        if seed == "vertex":
            return self.cells_at(v[0])
        if seed == "edge":
            return self.cells_on_edge(e[0])
        if seed == "face":
            return self.cells_on_face([v[t] for t in self.face_cycles[0]])
        if seed == "cell":
            return [0]
        raise ValueError(f"seed {seed!r} is not offered for {{p,q,r}} (vertex, edge, face, cell)")

    def precision(self):
        gap = min(self.cells.min_gap, self.verts.min_gap)
        return {"tol": KEY_TOL, "max_merge": max(self.cells.max_merge, self.verts.max_merge),
                "min_gap": gap if math.isfinite(gap) else None,
                "keys": [len(self.cells.pts), len(self.verts.pts), len(self.edge_src), len(self.face_src)]}

    def vertex_coords4(self, v):
        k = self.verts.pts[v]
        if self.spherical:
            return list(k)
        if self.ideal:
            return [1.0, k[0], k[1], k[2]]
        return [math.sqrt(1 + k[0] ** 2 + k[1] ** 2 + k[2] ** 2), k[0], k[1], k[2]]

    def face_cycles_of(self, c):
        self._materialise(c)
        v = self.mats[c][1]
        return [[v[t] for t in cyc] for cyc in self.face_cycles]


from ._honeycomb3d_growth_wiring import face_shells, grow_honeycomb  # noqa: E402,F401  (re-export)
