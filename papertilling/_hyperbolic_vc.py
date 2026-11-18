"""Pure-Python BDF vertex completion on a regular hyperbolic {p, q} tiling (fallback engine).

A port of `rust/papertilling-core/src/hyperbolic/inflation.rs` (`BaseFace`, `HypPointSet`,
`grow_vertex_completion`), used by `papertilling.hyperbolic` when the `papertilling_rs` extension is
absent or disabled. Reference: L. Boyle, M. Dickens, F. Flicker, "Conformal Quasicrystals and
Holography", Phys. Rev. X 10, 011009 (2020), arXiv:1805.02665 (v2), Sec. IV-V: with S_0 one p-gon,
S_{n+1} = S_n + every tile not in S_n sharing a vertex with S_n.

Geometry lives on the hyperboloid t^2 - x^2 - y^2 = 1 (points (x, y, t), Minkowski form
<a, b> = a0 b0 + a1 b1 - a2 b2). A face is a Lorentz matrix g applied to the base face centred at
(0, 0, 1). Points are deduplicated by hyperbolic distance d = 2 asinh(sqrt(<a - b, a - b>) / 2)
below half the minimum separation, with the same suspect-band precision guard as the Rust engine.
The iteration order (face, vertex j, star face m) is the Rust one, so the face order and the shell
index of each face match the Rust engine.
"""
from __future__ import annotations

import math

# Mirrored from rust/papertilling-core/src/hyperbolic/inflation.rs (T_PRECISION_LIMIT).
T_PRECISION_LIMIT = 1.0e6

_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_ORIGIN = (0.0, 0.0, 1.0)


def mink(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] - a[2] * b[2]


def mat_vec(m, v):
    return tuple(m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] for i in range(3))


def mat_mul(a, b):
    return tuple(
        tuple(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] for j in range(3)) for i in range(3)
    )


def hyp_dist(a, b) -> float:
    """Hyperbolic distance 2 asinh(sqrt(<a-b, a-b>) / 2); a finite negative square is clamped to 0."""
    d = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    m = mink(d, d)
    if not math.isfinite(m):
        return math.nan
    if m < 0.0:
        m = 0.0
    return 2.0 * math.asinh(0.5 * math.sqrt(m))


def _max_or_nan(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return max(a, b)


def hyperboloid_to_poincare(x) -> list[float]:
    return [x[0] / (1.0 + x[2]), x[1] / (1.0 + x[2])]


def is_hyperbolic(p: int, q: int) -> bool:
    return p >= 3 and q >= 3 and (p - 2) * (q - 2) > 4


def gamma_pq(p: int, q: int) -> float:
    """gamma_{p,q} = ((p - 2)(q - 2) - 2) / 2, BDF Eq. (11)."""
    return ((p - 2.0) * (q - 2.0) - 2.0) / 2.0


class BaseFace:
    """The base p-gon centred at (0, 0, 1), its edge reflections and its vertex stars."""

    def __init__(self, p: int, q: int):
        if not is_hyperbolic(p, q):
            raise ValueError(f"{{{p}, {q}}} is not hyperbolic (requires (p-2)(q-2) > 4)")
        self.p, self.q = p, q
        ap, aq = math.pi / p, math.pi / q
        self.inradius = math.acosh(math.cos(aq) / math.sin(ap))
        self.circumradius = math.acosh(1.0 / (math.tan(ap) * math.tan(aq)))
        sr, cr = math.sinh(self.circumradius), math.cosh(self.circumradius)
        self.vertices = [
            (sr * math.cos(2.0 * math.pi * j / p), sr * math.sin(2.0 * math.pi * j / p), cr) for j in range(p)
        ]
        self.edge_midpoints = []
        for k in range(p):
            s = tuple(self.vertices[k][i] + self.vertices[(k + 1) % p][i] for i in range(3))
            nrm = math.sqrt(-mink(s, s))
            self.edge_midpoints.append(tuple(c / nrm for c in s))
        shr, chr_ = math.sinh(self.inradius), math.cosh(self.inradius)
        self.reflections = []
        for k in range(p):
            th = 2.0 * math.pi * (k + 0.5) / p
            n = (chr_ * math.cos(th), chr_ * math.sin(th), shr)
            jn = (n[0], n[1], -n[2])
            self.reflections.append(
                tuple(tuple(_IDENTITY[i][j] - 2.0 * n[i] * jn[j] for j in range(3)) for i in range(3))
            )
        edge = hyp_dist(self.vertices[0], self.vertices[1 % p])
        tol_c, tol_v = 0.25 * self.inradius, 0.125 * edge
        self.star = []
        for j in range(p):
            a, b = j, (j + p - 1) % p
            w = _IDENTITY
            ring = []
            drift = 0.0
            for m in range(q):
                w = mat_mul(w, self.reflections[a if m % 2 == 0 else b])
                drift = _max_or_nan(drift, hyp_dist(mat_vec(w, self.vertices[j]), self.vertices[j]))
                if m + 1 < q:
                    ring.append(w)
            back = hyp_dist(mat_vec(w, _ORIGIN), _ORIGIN)
            vert_err = drift
            for k in range(p):
                k_img = k if q % 2 == 0 else (2 * j + p - k) % p
                vert_err = _max_or_nan(vert_err, hyp_dist(mat_vec(w, self.vertices[k]), self.vertices[k_img]))
            if math.isnan(back) or back >= tol_c or math.isnan(vert_err) or vert_err >= tol_v:
                raise ValueError(
                    f"{{{p},{q}}}: vertex star does not close within f64 precision (centre off by {back:.3e}, "
                    f"tolerance r/4 = {tol_c:.3e}; vertices off by {vert_err:.3e}, tolerance l/8 = {tol_v:.3e}); "
                    "q is too large for this construction"
                )
            self.star.append(ring)

    def edge_length(self) -> float:
        return hyp_dist(self.vertices[0], self.vertices[1 % self.p])


class HypPointSet:
    """Hyperboloid points deduplicated by hyperbolic distance (grid hash on ambient R^3).

    Points closer than min_sep / 2 merge. An insert whose nearest grid neighbour lies in the band
    [min_sep / 4, 3 min_sep / 4], or whose point is not finite, is counted in `suspect`.
    """

    def __init__(self, min_sep: float):
        self.cell = 2.0 * math.sinh(0.5 * min_sep)
        self.tol = 0.5 * min_sep
        self.pts: list[tuple[float, float, float]] = []
        self.grid: dict[tuple[int, int, int], list[int]] = {}
        self.max_merged = 0.0
        self.min_distinct = math.inf
        self.suspect = 0

    def insert(self, x) -> tuple[int, bool]:
        """Index of the point within tolerance of x, or insert it; returns (index, inserted)."""
        x = tuple(x)
        if not all(math.isfinite(c) for c in x):
            self.suspect += 1
            self.pts.append(x)
            return len(self.pts) - 1, True
        k = tuple(math.floor(c / self.cell) for c in x)
        best_i, best_d = -1, math.inf
        found = False
        for o in range(27):
            cell = (k[0] + o % 3 - 1, k[1] + (o // 3) % 3 - 1, k[2] + o // 9 - 1)
            for i in self.grid.get(cell, ()):
                d = hyp_dist(self.pts[i], x)
                if not found or d < best_d:
                    best_i, best_d, found = i, d, True
        if found:
            if math.isnan(best_d) or 0.5 * self.tol <= best_d <= 1.5 * self.tol:
                self.suspect += 1
            if best_d < self.tol:
                self.max_merged = max(self.max_merged, best_d)
                return best_i, False
            self.min_distinct = min(self.min_distinct, best_d)
        idx = len(self.pts)
        self.pts.append(x)
        self.grid.setdefault(k, []).append(idx)
        return idx, True


class VcPatch:
    """A vertex-completion patch: faces as Lorentz matrices, in shell order."""

    def __init__(self, base, faces, shell, shell_sizes, stopped):
        self.base, self.faces, self.shell, self.shell_sizes, self.stopped = base, faces, shell, shell_sizes, stopped

    def centre(self, i: int):
        return mat_vec(self.faces[i], _ORIGIN)

    def face_vertices(self, i: int):
        return [mat_vec(self.faces[i], v) for v in self.base.vertices]

    def face_edge_midpoints(self, i: int):
        return [mat_vec(self.faces[i], v) for v in self.base.edge_midpoints]


def grow_vertex_completion(p: int, q: int, n_max: int, face_budget: int) -> VcPatch:
    """Grow S_0 .. S_{n_max} from one p-gon, with the Rust engine's budget and precision stops."""
    base = BaseFace(p, q)
    centres = HypPointSet(2.0 * base.inradius)
    verts = HypPointSet(base.edge_length())
    face_verts: list[int] = []
    completed: list[bool] = []

    def add_vertices(g):
        for v in base.vertices:
            vi, new = verts.insert(mat_vec(g, v))
            if new:
                completed.append(False)
            face_verts.append(vi)

    centres.insert(_ORIGIN)
    add_vertices(_IDENTITY)
    faces, shell, shell_sizes, stopped = [_IDENTITY], [0], [1], None
    two_gamma = 2.0 * gamma_pq(p, q)
    star_centres = [[(w[0][2], w[1][2], w[2][2]) for w in ring] for ring in base.star]
    for n in range(n_max):
        a_n = shell_sizes[n]
        estimate = p * (q - 2) if n == 0 else math.ceil(two_gamma * a_n)
        if len(faces) + estimate > face_budget:
            stopped = f"face budget {face_budget}: shell {n + 1} (~{estimate} faces) not built"
            break
        start, end = len(faces) - a_n, len(faces)
        precision_hit = None
        for fi in range(start, end):
            g = faces[fi]
            for j in range(p):
                vi = face_verts[fi * p + j]
                if completed[vi]:
                    continue
                completed[vi] = True
                for m, c0 in enumerate(star_centres[j]):
                    c = mat_vec(g, c0)
                    if not all(math.isfinite(x) for x in c) or c[2] > T_PRECISION_LIMIT:
                        precision_hit = "f64 precision limit"
                        break
                    if centres.insert(c)[1]:
                        h = mat_mul(g, base.star[j][m])
                        add_vertices(h)
                        faces.append(h)
                        shell.append(n + 1)
                if precision_hit:
                    break
            if precision_hit:
                break
        if precision_hit is None and centres.suspect + verts.suspect > 0:
            precision_hit = "f64 precision: ambiguous dedup (a distance within [sep/4, 3 sep/4])"
        if precision_hit is not None:
            del faces[end:]
            del shell[end:]
            stopped = f"{precision_hit}: shell {n + 1} not built"
            break
        shell_sizes.append(len(faces) - end)
    return VcPatch(base, faces, shell, shell_sizes, stopped)


# Wedge-disclination patch (wedge_delta != 0): port of rust/papertilling-core/src/hyperbolic/wedge.rs.
# Copies of a regular n-gon under reflections across their edges (not a tiling: the interior angle is
# not 2 pi / q in general), grown by vertex completion: at every vertex j of every face of the last
# shell, floor(q/2) alternating reflections across the two edges at j, starting on each side. For an
# angle 2 pi / q this is BDF vertex completion; for q = 3 it is exactly edge-BFS.


class Polygon:
    """A regular n-gon centred at (0, 0, 1) on the hyperboloid, with its edge reflections."""

    def __init__(self, n: int, circumradius: float):
        # Right triangle centre / edge midpoint / vertex: tanh r = tanh R cos(pi / n).
        inradius = math.atanh(math.tanh(circumradius) * math.cos(math.pi / n))
        sr, cr = math.sinh(circumradius), math.cosh(circumradius)
        self.n = n
        self.vertices = [
            (sr * math.cos(2.0 * math.pi * j / n), sr * math.sin(2.0 * math.pi * j / n), cr) for j in range(n)
        ]
        shr, chr_ = math.sinh(inradius), math.cosh(inradius)
        self.reflections = []
        for k in range(n):
            th = 2.0 * math.pi * (k + 0.5) / n
            nv = (chr_ * math.cos(th), chr_ * math.sin(th), shr)
            jn = (nv[0], nv[1], -nv[2])
            self.reflections.append(
                tuple(tuple(_IDENTITY[i][j] - 2.0 * nv[i] * jn[j] for j in range(3)) for i in range(3))
            )


def grow_polygon_vertex_completion(poly: Polygon, q: int, n_max: int, max_faces: int, max_centre_radius: float,
                                   merge_dist: float):
    """S_0 .. S_{n_max} of copies of `poly`; returns (faces, shell, complete_through, notes, suspect).

    Faces whose Poincare vertex centroid lies beyond radius max_centre_radius are dropped (truncating
    their shell); a shell that would bring the total above max_faces is dropped whole; centres closer than
    merge_dist (hyperbolic) are one face; `suspect` counts centre-dedup band hits (near-coincident
    copies, kept).
    """
    n, walk = poly.n, q // 2
    centres = HypPointSet(2.0 * merge_dist)
    centres.insert(_ORIGIN)
    faces, shell, notes = [_IDENTITY], [0], []
    complete_through, truncated = 0, False
    r2_max = max_centre_radius * max_centre_radius
    for s in range(n_max):
        start = shell.index(s) if s in shell else len(faces)
        end = len(faces)
        over_cap = False
        for fi in range(start, end):
            g = faces[fi]
            for j in range(n):
                for first in (j, (j + n - 1) % n):
                    second = (j + n - 1) % n if first == j else j
                    w = g
                    for step in range(walk):
                        w = mat_mul(w, poly.reflections[first if step % 2 == 0 else second])
                        c = mat_vec(w, _ORIGIN)
                        # Radius cap on the Euclidean centroid of the face's Poincare vertices (the
                        # pre-upgrade criterion, kept by the Rust engine so q = 3 reproduces edge-BFS).
                        zx = zy = 0.0
                        for v in poly.vertices:
                            x = hyperboloid_to_poincare(mat_vec(w, v))
                            zx, zy = zx + x[0] / n, zy + x[1] / n
                        if not (zx * zx + zy * zy <= r2_max):
                            truncated = True
                            continue
                        if centres.insert(c)[1]:
                            faces.append(w)
                            shell.append(s + 1)
                            if len(faces) > max_faces:
                                over_cap = True
                                break
                    if over_cap:
                        break
                if over_cap:
                    break
            if over_cap:
                break
        if over_cap:
            del faces[end:]
            del shell[end:]
            notes.append(f"shell {s + 1} omitted by the {max_faces}-face cap")
            break
        if len(faces) == end:
            break
        if not truncated:
            complete_through = s + 1
    if truncated:
        notes.append(f"shell {complete_through + 1} truncated by the centre radius cap {max_centre_radius}")
    return faces, shell, complete_through, notes, centres.suspect


def polygon_faces_to_poincare(poly: Polygon, faces, kagome: bool, merge_dist: float):
    """Vertices (edge midpoints for kagome) of each face, deduplicated below merge_dist: (pts, faces)."""
    n = poly.n
    if kagome:
        base = []
        for k in range(n):
            a, b = poly.vertices[k], poly.vertices[(k + 1) % n]
            s = tuple(a[i] + b[i] for i in range(3))
            nrm = math.sqrt(s[2] * s[2] - s[0] * s[0] - s[1] * s[1])
            base.append(tuple(c / nrm for c in s))
    else:
        base = poly.vertices
    pset = HypPointSet(2.0 * merge_dist)
    idx = [[pset.insert(mat_vec(g, v))[0] for v in base] for g in faces]
    return [hyperboloid_to_poincare(x) for x in pset.pts], idx
