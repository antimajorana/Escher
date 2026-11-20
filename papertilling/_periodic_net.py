"""Dirichlet (Voronoi) cell complex of a 2- or 3-periodic point set, for the touching-rule grower.

Mirrors `papertilling-core/src/growth/periodic_net.rs` (+ `periodic_net/delaunay.rs`); C1 WP-L1, plan Q2:
a cell is the Dirichlet cell of a site, and two cells touch when they share a Voronoi vertex.

Sites: m points `sites` (Cartesian, R^3) of one cell of the lattice spanned by `lattice` (d = 2 or 3 rows;
for d = 2 the rows lie in the plane z = 0). A cell is (site, n0, n1, n2): site + n . lattice (n2 = 0 when d = 2).

Vertices. A Voronoi vertex is keyed by the maximal set T of sites lying on an empty sphere (no site strictly
inside): the Delaunay polytope dual to it. Three regimes:
  flat   d = 2, all sites at one height: the minimal Voronoi faces are vertical lines over the 2D Voronoi
         vertices, so T = the cocircular sets of the 2D Delaunay subdivision (circles through 3 sites);
  slab   d = 2, sites at several heights: finite spheres through 4 non-coplanar sites (every minimal face is a
         vertex because the sites span R^3); the cells are unbounded above and below;
  3d     d = 3: as slab, cells bounded.
A sphere is accepted only when its footprint (its intersection with the slab, or the whole ball) lies inside
the searched neighbourhood, so emptiness is checked against every site that could be inside it. Search radius
R starts at 2.5 h (h = (cell measure / m)^(1/D)); the spheres tried pass through site i and D others among
the N = 20 nearest (3D distance, ties by order) in that neighbourhood. R and N grow by 1.5 until the Delaunay
polytopes found (one per
translation class) fill the cell: area (flat), area x thickness (slab: the hull of a 2-periodic slab is the
slab between its lowest and highest sites), volume (3d), to relative 1e-6. The Delaunay polytopes of a point
set tile its convex hull, so a matching total means none is missing [PROVED: interior-disjoint polytopes whose
measures add up to the hull's]. Otherwise the build is refused with a reason.

Edges and faces (for the boundary counts and chi):
  flat   edge = two cells sharing 2 vertices (a Voronoi edge); no faces; chi = V - E + F is that of the union;
  3d     edge = two vertices whose polytopes share >= 3 sites (a Voronoi edge); face = two cells sharing >= 3
         vertices (a Voronoi facet); chi = V - E + F - C;
  slab   the same as 3d, counted over bounded faces only. The cells are unbounded, so chi of their union is not
         defined: `euler_defined` is False and callers must not show it.
Cospherical tolerance: relative 1e-7 (a site within 1e-7 rho of the sphere is on it; strictly inside below).
"""
from __future__ import annotations

import itertools

import numpy as np

RTOL = 1e-7
MEASURE_RTOL = 1e-6
MAX_ROUNDS = 6
N_NEAREST = 20


class DirichletError(ValueError):
    """The Delaunay polytopes found never filled the cell (refusal with a reason)."""


def _det(m):
    return float(np.linalg.det(np.asarray(m, float)))


def _polygon_area(pts, centre):
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    q = pts[np.argsort(ang, kind="stable")]
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()))


def hull_volume(pts):
    """Volume of the convex hull of a few points in general 3D position (brute-force facets)."""
    pts = np.asarray(pts, float)
    k = len(pts)
    g = pts.mean(0)
    scale = float(np.abs(pts - g).max())
    tol = 1e-9 * scale
    seen, vol = set(), 0.0
    for a, b, c in itertools.combinations(range(k), 3):
        n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
        nn = float(np.linalg.norm(n))
        if nn <= 1e-12 * scale * scale:
            continue
        n = n / nn
        s = (pts - pts[a]) @ n
        if s.max() > tol:
            if s.min() < -tol:
                continue
            n, s = -n, -s
        on = tuple(int(i) for i in np.nonzero(np.abs(s) <= tol)[0])
        if on in seen:
            continue
        seen.add(on)
        f = pts[list(on)]
        fc = f.mean(0)
        u = f[0] - fc
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        q = np.stack([(f - fc) @ u, (f - fc) @ v], 1)
        vol += _polygon_area(q, np.zeros(2)) * float((pts[a] - g) @ n) / 3.0
    return vol


def _canonical(T):
    """Translation-normalised sorted tuple of (site, n0, n1, n2): the smallest over the members of least site."""
    lo = min(t[0] for t in T)
    best = None
    for e in T:
        if e[0] != lo:
            continue
        rep = tuple(sorted((s, a - e[1], b - e[2], c - e[3]) for s, a, b, c in T))
        if best is None or rep < best:
            best = rep
    return best


class PeriodicNet:
    """Dirichlet complex of a periodic point set (module docstring); implements the grower's interface."""

    def __init__(self, lattice, sites):
        L = np.asarray(lattice, float)
        P = np.asarray(sites, float)
        self.d, self.m = len(L), len(P)
        if self.d not in (2, 3) or self.m == 0:
            raise DirichletError("need 2 or 3 lattice vectors and at least one site")
        self.L3 = np.vstack([L, np.zeros((3 - self.d, 3))])
        self.P = P
        z = P[:, 2]
        self.zmin, self.zmax = float(z.min()), float(z.max())
        if self.d == 3:
            self.mode = "3d"
        elif self.zmax - self.zmin <= 1e-9 * max(1.0, float(np.abs(L).max())):
            self.mode = "flat"
        else:
            self.mode = "slab"
        self.euler_defined = self.mode != "slab"
        self.verts = self._delaunay()
        self._templates()

    # ------------------------------------------------------------------ Delaunay polytopes
    def _target(self):
        if self.d == 3:
            return abs(_det(self.L3))
        area = float(np.linalg.norm(np.cross(self.L3[0], self.L3[1])))
        return area if self.mode == "flat" else area * (self.zmax - self.zmin)

    def _work(self, X):
        return X[..., :2] if self.mode == "flat" else X

    def _neighbourhood(self, i, R):
        """Every (site, n) whose (in-plane for slab) distance from site i is <= R, with its position."""
        Lp = self.L3[: self.d, : (3 if self.d == 3 else 2)]
        g = np.linalg.norm(np.linalg.inv(Lp), axis=0)
        rng = [range(-(int(np.ceil(R * gk)) + 1), int(np.ceil(R * gk)) + 2) for gk in g]
        out, pos = [], []
        for n in itertools.product(*rng):
            n3 = tuple(n) + (0,) * (3 - self.d)
            shift = np.array(n3, float) @ self.L3
            for j in range(self.m):
                q = self.P[j] + shift
                dv = q - self.P[i]
                dist = np.linalg.norm(dv[:2]) if self.mode == "slab" else np.linalg.norm(dv)
                if dist <= R:
                    out.append((j,) + n3)
                    pos.append(q)
        return out, np.array(pos)

    def _footprint_ok(self, c, rho, pi, R):
        if self.mode != "slab":
            return float(np.linalg.norm(c - self._work(pi))) + rho <= R
        dz = 0.0 if self.zmin <= c[2] <= self.zmax else min(abs(c[2] - self.zmin), abs(c[2] - self.zmax))
        rf = float(np.sqrt(max(0.0, rho * rho - dz * dz)))
        return float(np.linalg.norm(c[:2] - pi[:2])) + rf <= R

    def _measure(self, pts, c):
        return _polygon_area(pts, c) if self.mode == "flat" else hull_volume(pts)

    def _delaunay(self):
        target = self._target()
        dim = 2 if self.mode == "flat" else 3
        meas = self._target() if self.mode != "slab" else target / (self.zmax - self.zmin)
        R = 2.5 * (meas / self.m) ** (1.0 / (dim if self.mode != "slab" else 2))
        nc = N_NEAREST
        for _ in range(MAX_ROUNDS):
            found, uncertified = {}, False
            for i in range(self.m):
                ids, pos = self._neighbourhood(i, R)
                me = ids.index((i, 0, 0, 0))
                W = self._work(pos)
                d3 = np.linalg.norm(pos - pos[me], axis=1)
                others = [k for k in np.argsort(d3, kind="stable").tolist() if k != me][:nc]
                others.sort()
                combos = np.array(list(itertools.combinations(others, dim)), int).reshape(-1, dim)
                Q0, Qs = W[me], W[combos]
                A = 2.0 * (Qs - Q0)
                b = (Qs ** 2).sum(2) - (Q0 ** 2).sum()
                scale = np.maximum(np.abs(Qs - Q0).max(axis=(1, 2)), 1e-300)
                ok = np.abs(np.linalg.det(A)) > 1e-12 * (2 * scale) ** dim
                combos, A, b = combos[ok], A[ok], b[ok]
                if not len(combos):
                    continue
                C = np.linalg.solve(A, b[..., None])[..., 0]
                rho = np.linalg.norm(Q0 - C, axis=1)
                dist = np.linalg.norm(W[None, :, :] - C[:, None, :], axis=2)
                empty = ~(dist < (rho * (1 - RTOL))[:, None]).any(1)
                for x in np.nonzero(empty)[0]:
                    c, r = C[x], float(rho[x])
                    if not self._footprint_ok(c, r, pos[me], R):
                        uncertified = True
                        continue
                    on = np.nonzero(np.abs(dist[x] - r) <= RTOL * r)[0]
                    key = _canonical([ids[k] for k in on])
                    if key not in found:
                        found[key] = self._measure(W[on], c)
            total = sum(found.values())
            if not uncertified and abs(total - target) <= MEASURE_RTOL * target:
                return sorted(found)
            R *= 1.5
            nc = int(nc * 1.5)
        raise DirichletError(f"Dirichlet complex not certified: Delaunay polytopes fill {total:.6g} of {target:.6g}")

    # ------------------------------------------------------------------ incidence templates
    def _templates(self):
        self.inc = [[] for _ in range(self.m)]
        for u, T in enumerate(self.verts):
            for s, a, b, c in T:
                self.inc[s].append((u, -a, -b, -c))
        for row in self.inc:
            row.sort()
        self.edge_t, self.face_t = [], []
        for i in range(self.m):
            vs = self.vertices_of((i, 0, 0, 0))
            stars = {v: set(self.cells_at(v)) for v in vs}
            if self.mode == "flat":
                cnt = {}
                for v in vs:
                    for c in stars[v]:
                        if c != (i, 0, 0, 0):
                            cnt[c] = cnt.get(c, 0) + 1
                self.edge_t.append(sorted(c for c, k in cnt.items() if k >= 2))
                self.face_t.append([])
                continue
            self.edge_t.append(sorted((a, b) for a, b in itertools.combinations(vs, 2)
                                      if len(stars[a] & stars[b]) >= 3))
            cnt = {}
            for v in vs:
                for c in stars[v]:
                    if c != (i, 0, 0, 0):
                        cnt[c] = cnt.get(c, 0) + 1
            self.face_t.append(sorted(c for c, k in cnt.items() if k >= 3))

    @staticmethod
    def _add(x, n):
        return (x[0], x[1] + n[0], x[2] + n[1], x[3] + n[2])

    def vertices_of(self, c):
        return sorted(self._add(v, c[1:]) for v in self.inc[c[0]])

    def cells_at(self, v):
        return sorted(self._add(t, v[1:]) for t in self.verts[v[0]])

    def edges_of(self, c):
        n = c[1:]
        if self.mode == "flat":
            return sorted(tuple(sorted((c, self._add(o, n)))) for o in self.edge_t[c[0]])
        return sorted((self._add(a, n), self._add(b, n)) for a, b in self.edge_t[c[0]])

    def faces_of(self, c):
        if self.mode == "flat":
            return []
        return sorted(tuple(sorted((c, self._add(o, c[1:])))) for o in self.face_t[c[0]])

    def on_rim(self, v):
        return False

    def cells_on_edge(self, e):
        if self.mode == "flat":
            return sorted(e)
        return sorted(set(self.cells_at(e[0])) & set(self.cells_at(e[1])))

    def cells_on_face(self, f):
        s = set(self.cells_at(f[0]))
        for v in f[1:]:
            s &= set(self.cells_at(v))
        return sorted(s)

    # ------------------------------------------------------------------ seeds (plan §2.1)
    def seed_cells(self, kind):
        """The seed set: vertex = the vertex in most cells at site 0 (lowest index on ties), edge / face = the
        first of cell (0, 0, 0, 0), cell = that cell; dirichlet = that cell (a site's Dirichlet cell)."""
        c0 = (0, 0, 0, 0)
        if kind in ("cell", "dirichlet"):
            return [c0]
        if kind == "vertex":
            vs = self.vertices_of(c0)
            best = max(vs, key=lambda v: (len(self.verts[v[0]]), [-x for x in v]))
            return self.cells_at(best)
        if kind == "edge":
            return self.cells_on_edge(self.edges_of(c0)[0])
        if kind == "face":
            if self.mode == "flat" or not self.face_t[0]:
                raise ValueError("a face seed needs a 3D Dirichlet complex")
            return sorted(self.faces_of(c0)[0])
        raise ValueError(f"unknown seed kind {kind!r}")
