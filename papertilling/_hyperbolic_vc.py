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
    """Grow S_0 .. S_{n_max} from one p-gon, with the Rust engine's budget and precision stops
    (other seeds: `_hyperbolic_seeds.grow_vertex_completion_seeded`)."""
    from ._hyperbolic_seeds import grow_vertex_completion_seeded

    return grow_vertex_completion_seeded(p, q, "cell", n_max, face_budget)


# Wedge-disclination patch (wedge_delta != 0): port of rust/papertilling-core/src/hyperbolic/wedge.rs.
# Copies of a regular p_core-gon whose edge length matches {p, q} (cosh(l/2) = cos(pi/p)/sin(pi/q))
# grown by vertex completion: two star walks of (q - 1) // 2 alternating reflections (1 step each side
# for q = 3, 4; edge-BFS for q = 3) plus, for even q, the centred opposite hyperboloid half-turn H_{v_j}.


def _half_turn(v):
    jv = (v[0], v[1], -v[2])
    return tuple(tuple((-1.0 if i == j else 0.0) - 2.0 * v[i] * jv[j] for j in range(3)) for i in range(3))


class Polygon:
    """A regular n-gon centred at (0, 0, 1) on the hyperboloid, with reflections and vertex half-turns."""

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
        self.half_turns = [_half_turn(v) for v in self.vertices]


def grow_polygon_vertex_completion(poly: Polygon, q: int, n_max: int, max_faces: int, max_centre_radius: float,
                                   merge_dist: float):
    """S_0 .. S_{n_max} of copies of `poly`; returns (faces, shell, complete_through, notes, merged, kept).

    Faces whose Poincare vertex centroid lies beyond radius max_centre_radius are dropped (truncating
    their shell); a shell that would bring the total above max_faces is dropped whole; centres closer than
    merge_dist (hyperbolic) are one face; near-coincident guard-band hits split into (merged, kept).
    """
    n, walk, even_q = poly.n, (q - 1) // 2, (q % 2 == 0)
    centres = HypPointSet(2.0 * merge_dist)
    centres.insert(_ORIGIN)
    faces, shell, notes = [_IDENTITY], [0], []
    complete_through, truncated = 0, False
    merged, kept = 0, 0
    r2_max = max_centre_radius * max_centre_radius
    for s in range(n_max):
        start = shell.index(s) if s in shell else len(faces)
        end = len(faces)
        over_cap = False
        shell_trunc = False

        def _try_push(w):
            nonlocal shell_trunc, over_cap, merged, kept
            c = mat_vec(w, _ORIGIN)
            zx = zy = 0.0
            for v in poly.vertices:
                x = hyperboloid_to_poincare(mat_vec(w, v))
                zx, zy = zx + x[0] / n, zy + x[1] / n
            if not (zx * zx + zy * zy <= r2_max):
                shell_trunc = True
                return False
            prev_suspect = centres.suspect
            _, inserted = centres.insert(c)
            if centres.suspect > prev_suspect:
                if inserted:
                    kept += 1
                else:
                    merged += 1
            if inserted:
                faces.append(w)
                shell.append(s + 1)
                if len(faces) > max_faces:
                    over_cap = True
                    return True
            return False

        for fi in range(start, end):
            g = faces[fi]
            for j in range(n):
                for first in (j, (j + n - 1) % n):
                    second = (j + n - 1) % n if first == j else j
                    w = g
                    for step in range(walk):
                        w = mat_mul(w, poly.reflections[first if step % 2 == 0 else second])
                        if _try_push(w):
                            break
                    if over_cap:
                        break
                if over_cap:
                    break
                if even_q and _try_push(mat_mul(g, poly.half_turns[j])):
                    break
            if over_cap:
                break
        if over_cap:
            del faces[end:]
            del shell[end:]
            notes.append(f"shell {s + 1} omitted by the {max_faces}-face cap")
            break
        truncated = truncated or shell_trunc
        if len(faces) == end:
            break
        if not truncated:
            complete_through = s + 1
    if truncated:
        notes.append(f"shell {complete_through + 1} truncated by the centre radius cap {max_centre_radius}")
    return faces, shell, complete_through, notes, merged, kept


def keep_whole_shells(shell_sizes, stopped, max_faces: int):
    """Whole-shell face-cap selection matching rust/papertilling-core/src/hyperbolic/mod.rs."""
    keep, total = 0, 0
    for a in shell_sizes:
        if total + a > max_faces:
            break
        total += a
        keep += 1
    notes = []
    if keep < len(shell_sizes):
        notes.append(f"shell {keep} ({shell_sizes[keep]} faces) omitted by the {max_faces}-face cap")
    if stopped is not None and stopped.startswith("face budget"):
        if keep >= len(shell_sizes):
            notes.append(f"shell {len(shell_sizes)} omitted by the {max_faces}-face cap (estimated larger)")
    elif stopped is not None:
        notes.append(stopped)
    return keep, notes


def _bx_rz_pi(r: float):
    ch, sh = math.cosh(r), math.sinh(r)
    return ((-ch, 0.0, sh), (0.0, -1.0, 0.0), (-sh, 0.0, ch))


def _qgon_midpoints_at_vertex(base: BaseFace, j: int):
    p, q = base.p, base.q
    a, b = j, (j + p - 1) % p
    pts = [base.edge_midpoints[b], base.edge_midpoints[a]]
    for k in range(2, q):
        side = b if (k % 2 == 0) else a
        pts.append(mat_vec(base.star[j][k - 2], base.edge_midpoints[side]))
    pts.reverse()
    return pts


def grow_kagome_cluster(p: int, q: int, lay: int, max_faces: int, max_centre_radius: float):
    """Rectified (p.q.p.q) kagome grown by the cluster rule (mirrors hyperbolic/kagome.rs)."""
    r2_max = max_centre_radius * max_centre_radius
    if p == q:
        vc4 = grow_vertex_completion(p, 4, lay, 2 * max_faces)
        keep, notes = keep_whole_shells(vc4.shell_sizes, vc4.stopped, max_faces)
        ang = math.pi / p
        ca, sa = math.cos(ang), math.sin(ang)
        pset = HypPointSet(hyp_dist(vc4.base.vertices[0], vc4.base.vertices[1]))
        complete = keep - 1
        faces, face_shell = [], []
        for i, s in enumerate(vc4.shell):
            if s >= keep:
                break
            if i > 0:
                z = hyperboloid_to_poincare(vc4.centre(i))
                if z[0] * z[0] + z[1] * z[1] > r2_max:
                    complete = min(complete, s - 1)
                    continue
            faces.append([pset.insert((ca * v[0] - sa * v[1], sa * v[0] + ca * v[1], v[2]))[0] for v in vc4.face_vertices(i)])
            face_shell.append(s)
        if pset.suspect > 0:
            raise ValueError(f"{{{p},{p}}}: f64 precision: {pset.suspect} drawn points have an ambiguous dedup distance")
        if complete + 1 < keep:
            notes.append(f"shell {complete + 1} truncated by the centre radius cap {max_centre_radius}")
        pts = [hyperboloid_to_poincare(x) for x in pset.pts]
        return pts, faces, face_shell, complete, notes

    dual_mode = p < q
    base = BaseFace(q, p) if dual_mode else BaseFace(p, q)
    bp = base.p
    t_mat = _bx_rz_pi(base.circumradius) if dual_mode else _IDENTITY
    l_k = hyp_dist(base.edge_midpoints[0], base.edge_midpoints[1 % bp])
    centres = HypPointSet(2.0 * base.inradius)
    verts = HypPointSet(base.edge_length())
    p_mats = [_IDENTITY] + list(base.star[0]) if dual_mode else [_IDENTITY]
    for g in p_mats:
        centres.insert(mat_vec(g, _ORIGIN))

    def _face_pair(g):
        wg = mat_mul(t_mat, g)
        return mat_vec(wg, _ORIGIN), [mat_vec(wg, x) for x in base.edge_midpoints]

    def _dec_pair(g, j):
        wg = mat_mul(t_mat, g)
        return mat_vec(wg, base.vertices[j]), [mat_vec(wg, x) for x in _qgon_midpoints_at_vertex(base, j)]

    dec0 = []
    for g in p_mats:
        for j in range(bp):
            if verts.insert(mat_vec(g, base.vertices[j]))[1]:
                dec0.append(_dec_pair(g, j))
    p0_faces = [_face_pair(g) for g in p_mats]
    s0 = ([dec0[0]] + p0_faces + dec0[1:]) if dual_mode else (p0_faces + dec0)
    total = len(s0)
    raw_shells, notes, p_start = [s0], [], 0
    for s in range(lay):
        p_end = len(p_mats)
        next_p, next_dec, aborted = [], [], None
        for fi in range(p_start, p_end):
            g = p_mats[fi]
            for k in range(bp):
                h = mat_mul(g, base.reflections[k])
                c = mat_vec(h, _ORIGIN)
                if not all(math.isfinite(x) for x in c) or c[2] > T_PRECISION_LIMIT:
                    aborted = f"f64 precision limit: shell {s + 1} not built"
                    break
                if centres.insert(c)[1]:
                    p_mats.append(h)
                    next_p.append(_face_pair(h))
                    for j in range(bp):
                        if verts.insert(mat_vec(h, base.vertices[j]))[1]:
                            next_dec.append(_dec_pair(h, j))
                    if total + len(next_p) + len(next_dec) > 2 * max_faces:
                        aborted = f"shell {s + 1} omitted by the {max_faces}-face cap (estimated larger)"
                        break
            if aborted is not None:
                break
        if aborted is not None:
            notes.append(aborted)
            break
        if centres.suspect + verts.suspect > 0:
            notes.append(f"f64 precision: ambiguous dedup: shell {s + 1} not built")
            break
        a = len(next_p) + len(next_dec)
        if total + a > max_faces:
            notes.append(f"shell {s + 1} ({a} faces) omitted by the {max_faces}-face cap")
            break
        raw_shells.append(next_p + next_dec)
        total += a
        p_start = p_end
    keep = len(raw_shells)
    complete = keep - 1
    kag_set = HypPointSet(l_k)
    faces, face_shell = [], []
    for s, sh in enumerate(raw_shells):
        for cen, vpts in sh:
            if s > 0:
                z = hyperboloid_to_poincare(cen)
                if z[0] * z[0] + z[1] * z[1] > r2_max:
                    complete = min(complete, s - 1)
                    continue
            faces.append([kag_set.insert(x)[0] for x in vpts])
            face_shell.append(s)
    if kag_set.suspect > 0:
        raise ValueError(f"{{{p},{q}}}: f64 precision: {kag_set.suspect} drawn points have an ambiguous dedup distance")
    if complete + 1 < keep:
        notes.append(f"shell {complete + 1} truncated by the centre radius cap {max_centre_radius}")
    pts = [hyperboloid_to_poincare(x) for x in kag_set.pts]
    return pts, faces, face_shell, complete, notes


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
