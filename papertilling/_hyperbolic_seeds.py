"""Seeds and layers for {p, q} vertex completion (C1 WP-G3): pure-Python mirror of
rust/papertilling-core/src/hyperbolic/{inflation_seeds,boundary_counts}.rs and noneuclidean2d/s2_seeds.rs.

Touching rule: S_{k+1} = S_k plus every tile sharing a vertex with S_k; layer 0 is the seed set.
Seeds: "cell" (or "face": one tile), "vertex" (the q tiles at a vertex), "edge" (the 2 tiles on an edge).
"dirichlet" is an E^2 lattice seed and is refused here.
"""
from __future__ import annotations

import math

from . import _hyperbolic_vc as vc

SEEDS = ("vertex", "edge", "face", "cell", "dirichlet")
_DIRICHLET_H2 = "the Dirichlet seed is defined for E² lattices only; use a vertex, edge or tile seed"


def norm_seed(seed) -> str:
    s = str(seed).lower()
    if s not in SEEDS:
        raise ValueError(f"unknown seed {seed!r}; expected one of {SEEDS}")
    return s


def one_tile(seed: str) -> bool:
    return seed in ("cell", "face")


def seed_tiles(base, seed: str):
    if one_tile(seed):
        return [vc._IDENTITY]
    if seed == "vertex":
        return [vc._IDENTITY] + list(base.star[0])
    if seed == "edge":
        return [vc._IDENTITY, base.reflections[0]]
    raise ValueError(_DIRICHLET_H2)


def seed_centre(base, seed: str):
    """Point the drawing is centred on: tile centre, base vertex 0 (vertex seed) or edge-0 midpoint (edge seed)."""
    if seed == "vertex":
        return base.vertices[0]
    if seed == "edge":
        return base.edge_midpoints[0]
    return vc._ORIGIN


def recentre(x):
    """Lorentz boost taking hyperboloid point x to (0, 0, 1): [[I + s s^T / (1 + t), -s], [-s^T, t]]."""
    a, b, t = x
    k = 1.0 / (1.0 + t)
    return ((1.0 + a * a * k, a * b * k, -a), (a * b * k, 1.0 + b * b * k, -b), (-a, -b, t))


def seed_label(p: int, q: int, seed: str) -> str:
    if one_tile(seed):
        return f"one {p}-gon"
    return {"vertex": f"the {q} tiles at a vertex", "edge": "the 2 tiles on an edge"}.get(seed, "a Dirichlet seed (refused in H²)")


# ---- exact boundary counters (port of boundary_counts.rs) ----

def seed_letters(p: int, q: int, seed: str):
    """(a_0, cnt) with cnt[k] = boundary vertices lying in exactly k seed tiles; None for dirichlet or p, q < 3."""
    if p < 3 or q < 3 or seed == "dirichlet":
        return None
    cnt = [0] * (q + 2)
    if one_tile(seed):
        cnt[1] = p
        return 1, cnt
    if seed == "vertex":
        cnt[2], cnt[1] = q, q * (p - 3)
        return q, cnt
    cnt[2], cnt[1] = 2, 2 * (p - 2)
    return 2, cnt


def _step_q_gt3(p, q, cnt):
    nxt, faces = [0] * (q + 2), 0
    for k, n in enumerate(cnt):
        if n == 0:
            continue
        o = q - k
        if o < 2 or (p == 3 and o < 3):
            return None
        faces += n * (o - 1)
        if p >= 4:
            nxt[2] += n * (o - 1)
            nxt[1] += n * ((o - 2) * (p - 3) + (p - 4))
        else:
            nxt[2] += n * (o - 3)
            nxt[3] += n
    return faces, nxt


def _step_q3(p, cnt):
    if any(n > 0 for n in cnt[3:]) or cnt[1] < cnt[2] or cnt[1] == 0:
        return None
    n1, n2 = cnt[1], cnt[2]
    c1, c2 = p - 2, p - 3
    if (n1 > n2 and c1 < 3) or (n2 > 0 and c2 < 3):
        return None
    nxt = [0] * 5
    nxt[2] = n1
    nxt[1] = (n1 - n2) * (c1 - 2) + n2 * max(c2 - 2, 0)
    return n1, nxt


def letter_counts_from(p: int, q: int, a0: int, cnt, n_max: int):
    if p < 3 or q < 3:
        return None
    cnt = list(cnt) + [0] * max(0, q + 2 - len(cnt))
    out = [a0]
    for _ in range(n_max):
        r = _step_q3(p, cnt) if q == 3 else _step_q_gt3(p, q, cnt)
        if r is None:
            return None
        out.append(r[0])
        cnt = r[1]
    return out


def vertex_completion_counts_seeded(p: int, q: int, seed: str, n_max: int):
    """Exact shell sizes [a_0, .., a_n] from the seed's boundary letters; None off the hyperbolic range."""
    if not vc.is_hyperbolic(p, q):
        return None
    s = seed_letters(p, q, norm_seed(seed))
    return None if s is None else letter_counts_from(p, q, s[0], s[1], n_max)


def seed_word(p: int, q: int, seed: str):
    if p < 3 or q < 3 or seed == "dirichlet":
        return None
    if one_tile(seed):
        return 1, [1] * p
    if seed == "vertex":
        return q, [x for _ in range(q) for x in [2] + [1] * (p - 3)]
    return 2, [x for _ in range(2) for x in [2] + [1] * (p - 2)]


def boundary_word_counts_from(p: int, q: int, a0: int, boundary, n_max: int):
    if p < 3 or q < 3 or not boundary:
        return None
    boundary, out = list(boundary), [a0]
    for _ in range(n_max):
        n = len(boundary)
        if any(q - k < 1 for k in boundary):
            return None
        o = [q - k for k in boundary]
        convex = all(x >= 2 for x in o) if q > 3 else all(o[i] >= 2 or o[(i + 1) % n] >= 2 for i in range(n))
        i0 = next((i for i in range(n) if o[i] >= 2), None)
        if not convex or i0 is None:
            return None
        c_list, open_m = [], 0
        for t in range(n):
            i = (i0 + t) % n
            if o[i] == 1:
                open_m += 1
                continue
            if open_m > 0:
                if p - open_m - 1 < 1:
                    return None
                c_list.append(p - open_m - 1)
            c_list.extend([p - 1] * (o[i] - 2))
            open_m = 1
        if p - open_m - 1 < 1:
            return None
        c_list.append(p - open_m - 1)
        out.append(len(c_list))
        f0 = next((j for j in range(len(c_list)) if c_list[j] >= 2), None)
        if f0 is None:
            return None
        nxt, pending = [], 0
        for t in range(len(c_list)):
            c = c_list[(f0 + t) % len(c_list)]
            if c == 1:
                pending += 1
                continue
            if t > 0:
                nxt.append(pending)
            nxt.extend([1] * (c - 2))
            pending = 2
        nxt.append(pending)
        boundary = nxt
    return out


# ---- seeded geometric grower (port of inflation_seeds.rs) ----

def grow_vertex_completion_seeded(p: int, q: int, seed: str, n_max: int, face_budget: int) -> vc.VcPatch:
    """Grow S_0 .. S_{n_max} from `seed`, with the Rust engine's budget estimate and precision stops."""
    seed = norm_seed(seed)
    base = vc.BaseFace(p, q)
    seed_mats = seed_tiles(base, seed)
    exact = None if one_tile(seed) else vertex_completion_counts_seeded(p, q, seed, n_max)
    centres, verts = vc.HypPointSet(2.0 * base.inradius), vc.HypPointSet(base.edge_length())
    face_verts: list[int] = []
    completed: list[bool] = []

    def add_vertices(g):
        for v in base.vertices:
            vi, new = verts.insert(vc.mat_vec(g, v))
            if new:
                completed.append(False)
            face_verts.append(vi)

    faces = []
    for g in seed_mats:
        if centres.insert(vc.mat_vec(g, vc._ORIGIN))[1]:
            add_vertices(g)
            faces.append(g)
    shell, shell_sizes, stopped = [0] * len(faces), [len(faces)], None
    two_gamma = 2.0 * vc.gamma_pq(p, q)
    star_centres = [[(w[0][2], w[1][2], w[2][2]) for w in ring] for ring in base.star]
    for n in range(n_max):
        a_n = shell_sizes[n]
        if exact is not None and len(exact) > n + 1:
            estimate = exact[n + 1]
        elif n == 0:
            estimate = p * (q - 2) if one_tile(seed) else a_n * p * (q - 1)
        else:
            estimate = math.ceil(two_gamma * a_n)
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
                    c = vc.mat_vec(g, c0)
                    if not all(math.isfinite(x) for x in c) or c[2] > vc.T_PRECISION_LIMIT:
                        precision_hit = "f64 precision limit"
                        break
                    if centres.insert(c)[1]:
                        h = vc.mat_mul(g, base.star[j][m])
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
    return vc.VcPatch(base, faces, shell, shell_sizes, stopped)


# ---- S^2 growth (port of s2_seeds.rs) ----

def grow_s2(p: int, faces, seed: str, max_layers: int) -> dict:
    """Touching-rule layers on a spherical {p, q} face list; closes when S_k is the whole sphere."""
    seed = norm_seed(seed)
    if not faces:
        raise ValueError("no faces to grow on")
    if seed == "dirichlet":
        raise ValueError("the Dirichlet seed is defined for E² lattices only; use a vertex, edge or face seed")
    if p == 2:
        nq = len(faces)
        edges = [[k, (k + 1) % nq] for k in range(nq)]
    else:
        ids: dict = {}
        edges = [[ids.setdefault((min(f[k], f[(k + 1) % len(f)]), max(f[k], f[(k + 1) % len(f)])), len(ids)) for k in range(len(f))] for f in faces]
    if one_tile(seed):
        s = {0}
    elif seed == "vertex":
        s = {f for f in range(len(faces)) if faces[0][0] in faces[f]}
    else:
        s = {f for f in range(len(faces)) if edges[0][0] in edges[f]}

    def euler(sel):
        vs = {v for f in sel for v in faces[f]}
        es = {e for f in sel for e in edges[f]}
        return len(vs) - len(es) + len(sel)

    face_layer: list = [None] * len(faces)
    for f in s:
        face_layer[f] = 0
    counts, chis, stopped = [len(s)], [euler(s)], None
    for k in range(max_layers + 1):
        if len(s) == len(faces):
            stopped = f"closed at layer {k}: S_{k} is the whole sphere (χ = 2)"
            break
        if k == max_layers:
            break
        verts = {v for f in s for v in faces[f]}
        grown = {f for f in range(len(faces)) if f not in s and any(v in verts for v in faces[f])}
        for f in grown:
            face_layer[f] = k + 1
        s |= grown
        counts.append(len(s))
        chis.append(euler(s))
    return {"seed": seed, "face_layer": face_layer, "layer_counts": counts,
            "layer_euler": chis, "stopped": stopped}
