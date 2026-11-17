"""Exact Burgers-circuit parallel-transport engine and automatic circuit finder."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..legacy.fusion_framework import is_orientable_surface
from ..legacy.lab_defects import iso_apply, iso_compose, iso_inverse
from . import fast as _fast
from .build import LatticeData, classify_charge, lattice_data

TWO_PI = 2.0 * math.pi


def _norm(v) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y, inside, n = float(pt[0]), float(pt[1]), False, len(poly)
    for i in range(n):
        x1, y1, x2, y2 = float(poly[i, 0]), float(poly[i, 1]), float(poly[(i + 1) % n, 0]), float(poly[(i + 1) % n, 1])
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _winding(pt: np.ndarray, poly: np.ndarray) -> int:
    px, py, ang, n = float(pt[0]), float(pt[1]), 0.0, len(poly)
    for i in range(n):
        ax, ay = float(poly[i, 0]) - px, float(poly[i, 1]) - py
        bx, by = float(poly[(i + 1) % n, 0]) - px, float(poly[(i + 1) % n, 1]) - py
        if math.hypot(ax, ay) < 1e-9 or math.hypot(bx, by) < 1e-9:
            return 0
        ang += math.atan2(ax * by - ay * bx, ax * bx + ay * by)
    return int(round(ang / TWO_PI))


def shift_circuit_basepoint(path, new_v0: int) -> list[int]:
    """Return an equivalent closed circuit with basepoint shifted to `new_v0`."""
    verts = [int(v) for v in path]
    core = verts[:-1] if (len(verts) >= 4 and verts[0] == verts[-1]) else verts
    if len(core) < 3:
        raise ValueError("circuit path must have at least 3 distinct edges")
    v0 = int(new_v0)
    if v0 not in core:
        raise ValueError(f"vertex {v0} does not lie on the circuit path")
    idx = core.index(v0)
    rot_c = core[idx:] + core[:idx]
    return rot_c + [rot_c[0]]


def _edge_vec_and_g(d, u: int, v: int):
    e = d.edge.get((u, v))
    if e is None:
        raise ValueError(f"circuit traverses missing directed bond ({u}, {v})")
    g = e.get("g") or (e.get("seam") if isinstance(e.get("seam"), (tuple, list)) else None)
    if g is not None:
        g = (np.asarray(g[0], float), np.asarray(g[1], float))
    if "vec" in e or "v" in e:
        return np.asarray(e.get("vec", e.get("v")), float), g
    if getattr(d, "kind", "") == "mobius" and "h" in d.info:
        j = int(e.get("j", 0))
        h = (np.asarray(d.info["h"][0], float), np.asarray(d.info["h"][1], float))
        hinv = iso_inverse(h)
        pu, pv = np.asarray(d.positions[u], float), np.asarray(d.positions[v], float)
        if j == 0:
            return pv - pu, None
        step_g, out_step = (h, hinv) if j > 0 else (hinv, h)
        mapped, out_g = pv, out_step
        for _ in range(abs(j)):
            mapped = iso_apply(step_g, mapped)
        for _ in range(abs(j) - 1):
            out_g = iso_compose(out_g, out_step)
        return mapped - pu, out_g
    return np.asarray(d.positions[v], float) - np.asarray(d.positions[u], float), g


def _frank_from_walk(d, verts: list[int], vecs: list[np.ndarray], gs: list, H_rot: np.ndarray) -> float:
    m, turns, P = len(vecs), 0.0, np.asarray(d.positions, float)
    for k in range(m):
        vp, gp, vo = vecs[(k - 1) % m], gs[(k - 1) % m], vecs[k]
        vi = vp if gp is None else (gp[0] @ vp)
        turns += math.degrees(math.atan2(float(vi[0] * vo[1] - vi[1] * vo[0]), float(vi[0] * vo[0] + vi[1] * vo[1])))
    area = sum(float(P[verts[k], 0] * P[verts[k + 1], 1] - P[verts[k], 1] * P[verts[k + 1], 0]) for k in range(m))
    omega_gb = 360.0 * (1.0 if area >= 0.0 else -1.0) - turns
    rot_deg = math.degrees(math.atan2(float(H_rot[1, 0]), float(H_rot[0, 0])))
    return float(round(rot_deg + 360.0 * round((omega_gb - rot_deg) / 360.0), 6))


def burgers_circuit(config_or_defect, path, lattice=None) -> dict:
    """Develop a closed bond walk `path = [v0, ..., v0]` and return exact holonomy and closure data."""
    d, lat = (config_or_defect.defect, config_or_defect.lattice if lattice is None else lattice) \
        if hasattr(config_or_defect, "defect") else (config_or_defect, lattice)
    if lat is not None and not isinstance(lat, LatticeData):
        lat = lattice_data(lat)
    verts = [int(v) for v in path]
    if len(verts) < 4 or verts[0] != verts[-1]:
        raise ValueError("path must be a closed walk [v0, v1, ..., v0] with at least 3 bonds")
    F = (np.eye(2, dtype=float), np.zeros(2, float))
    H = (np.eye(2, dtype=float), np.zeros(2, float))
    ref_path, frames, cum_angle_deg, det_signs, vecs, gs = [[0.0, 0.0]], [np.eye(2, dtype=float)], [0.0], [1], [], []
    for k in range(len(verts) - 1):
        vec, g = _edge_vec_and_g(d, verts[k], verts[k + 1])
        vecs.append(vec); gs.append(g)
        step = F[0] @ vec
        ref_path.append([float(ref_path[-1][0] + step[0]), float(ref_path[-1][1] + step[1])])
        d_ang = 0.0
        if g is not None:
            F, H = iso_compose(F, iso_inverse(g)), iso_compose(g, H)
            if float(np.linalg.det(g[0])) > 0.0 and det_signs[-1] > 0:
                d_ang = math.degrees(math.atan2(float(g[0][1, 0]), float(g[0][0, 0])))
        frames.append(F[0].copy())
        det_s = 1 if float(np.linalg.det(F[0])) > 0.0 else -1
        det_signs.append(det_s)
        cum_angle_deg.append(cum_angle_deg[-1] + (d_ang if det_s > 0 else 0.0))
    orientation_preserving = det_signs[-1] > 0
    if orientation_preserving:
        frank_deg = _frank_from_walk(d, verts, vecs, gs, H[0])
        shift = frank_deg - cum_angle_deg[-1]
        if abs(shift) > 1.0:
            last_seam = next((i for i in range(len(gs) - 1, -1, -1) if gs[i] is not None), len(gs) - 1)
            for j in range(last_seam + 1, len(cum_angle_deg)):
                cum_angle_deg[j] += shift
    else:
        frank_deg = None
    closure = np.asarray(ref_path[-1], float) - np.asarray(ref_path[0], float)
    closure[np.abs(closure) < 1e-11] = 0.0
    charge = classify_charge(lat, frank_deg, H) if lat is not None else None
    return {
        "path": verts, "ref_path": ref_path, "frames": frames, "cum_angle_deg": cum_angle_deg,
        "det_signs": det_signs, "orientation_preserving": orientation_preserving, "H": H,
        "closure_vector": closure, "burgers_norm": float(_norm(closure)),
        "frank_angle_deg": frank_deg, "charge": charge,
    }


def _mobius_seam_circuit(config_or_defect, lattice=None) -> dict:
    d = config_or_defect.defect if hasattr(config_or_defect, "defect") else config_or_defect
    adj: dict[int, list[tuple[int, int]]] = {}
    seeds = []
    for (u, v), _ in d.edge.items():
        _, g = _edge_vec_and_g(d, u, v)
        sgn = 1 if (g is None or float(np.linalg.det(g[0])) > 0.0) else -1
        adj.setdefault(u, []).append((v, sgn))
        if sgn == -1:
            seeds.append(u)
    if not seeds:
        raise ValueError("non-orientable defect has no orientation-reversing seam edge")
    start = seeds[0]
    parent, dq, goal = {(start, 1): None}, deque([(start, 1)]), (start, -1)
    while dq and goal not in parent:
        u, s = dq.popleft()
        for v, es in adj.get(u, ()):
            nxt = (v, s * es)
            if nxt not in parent:
                parent[nxt] = (u, s); dq.append(nxt)
    if goal not in parent:
        raise ValueError("could not find an orientation-reversing closed loop")
    walk, cur = [], goal
    while cur is not None:
        walk.append(cur[0]); cur = parent[cur]
    return burgers_circuit(config_or_defect, walk[::-1], lattice=lattice)


def _match_point_to_face(pt: np.ndarray, config, int_faces: list[list[int]], f_charges: list[tuple], P: np.ndarray):
    d = config.defect if hasattr(config, "defect") else config
    X_ref = d.info.get("reference")
    lmax = config.lattice.lmax if hasattr(config, "lattice") else 1.5
    cores = config.cores() if hasattr(config, "cores") else _fast.cores(d)
    best_core = None
    for c in cores:
        dm = min(_norm(P[c["sites"]].mean(axis=0) - pt), _norm(X_ref[c["sites"]].mean(axis=0) - pt) if X_ref is not None else float("inf"))
        if dm <= 1.2 * lmax and (best_core is None or dm < best_core[0]):
            best_core = (dm, c)
    if best_core is not None:
        c = best_core[1]
        if c.get("site") is not None:
            s_v = int(c["site"])
            inc = [fi for fi, f in enumerate(int_faces) if s_v in f]
            rot_inc = [fi for fi in inc if not np.allclose(f_charges[fi][0][0], np.eye(2), atol=1e-5)]
            return inc, (rot_inc[0] if rot_inc else inc[0]), P[s_v]
        tset = set(c["sites"])
        for fi, f in enumerate(int_faces):
            if set(f) == tset:
                return [fi], fi, P[f].mean(axis=0)
    for coords, use_p in ((P, True), (X_ref, False)):
        if coords is not None:
            for fi, f in enumerate(int_faces):
                if _point_in_poly(pt, coords[f]):
                    return [fi], fi, (pt if use_p else P[f].mean(axis=0))
    fi = min(range(len(int_faces)), key=lambda i: _norm(P[int_faces[i]].mean(axis=0) - pt))
    return [fi], fi, P[int_faces[fi]].mean(axis=0)


def _extract_boundary_cycle(int_faces: list[list[int]], region: set[int]) -> list[int] | None:
    dir_edges = {(f[k], f[(k + 1) % len(f)]) for fi in region for f in (int_faces[fi],) for k in range(len(f))}
    bnd = [(u, v) for (u, v) in dir_edges if (v, u) not in dir_edges]
    if not bnd:
        return None
    out_map = {}
    for u, v in bnd:
        if u in out_map:
            return None
        out_map[u] = v
    start = bnd[0][0]
    cycle, cur = [start], out_map[start]
    while cur != start:
        if cur in cycle or cur not in out_map:
            return None
        cycle.append(cur); cur = out_map[cur]
    return cycle + [start] if len(cycle) == len(bnd) else None


def _connect_seed(seed: set[int], dual_adj: list[set[int]], forbidden: set[int]) -> set[int] | None:
    if len(seed) <= 1:
        return set(seed)
    rem, first = set(seed), next(iter(seed))
    comp, dq = {first}, deque([first])
    while dq:
        for nb in dual_adj[dq.popleft()]:
            if nb in rem and nb not in comp:
                comp.add(nb); dq.append(nb)
    rem -= comp
    connected = set(comp)
    while rem:
        parent, dq, hit = {fi: None for fi in connected}, deque(connected), None
        while dq and hit is None:
            fi = dq.popleft()
            for nb in dual_adj[fi]:
                if nb not in forbidden and nb not in parent:
                    parent[nb] = fi
                    if nb in rem:
                        hit = nb; break
                    dq.append(nb)
        if hit is None:
            return None
        cur = hit
        while cur is not None:
            connected.add(cur); rem.discard(cur); cur = parent[cur]
    return connected


def _fill_holes(n_faces: int, dual_adj: list[set[int]], outer_adj: set[int], reg: set[int], forbidden: set[int]) -> set[int] | None:
    outside = {fi for fi in outer_adj if fi not in reg}
    dq = deque(outside)
    while dq:
        for nb in dual_adj[dq.popleft()]:
            if nb not in reg and nb not in outside:
                outside.add(nb); dq.append(nb)
    holes = set(range(n_faces)) - reg - outside
    return None if (holes & forbidden) else (reg | holes)


def _normalize_points(pts) -> list[np.ndarray]:
    if pts is None:
        return []
    arr = np.asarray(pts, float)
    return [arr] if (arr.ndim == 1 and arr.shape == (2,)) else [np.asarray(p, float) for p in pts]


def _dual_graph(int_faces: list[list[int]]):
    edge_owner = {(f[k], f[(k + 1) % len(f)]): fi for fi, f in enumerate(int_faces) for k in range(len(f))}
    dual_adj: list[set[int]] = [set() for _ in int_faces]
    v2f: dict[int, set[int]] = {}
    outer_adj: set[int] = set()
    for fi, f in enumerate(int_faces):
        for k in range(len(f)):
            v2f.setdefault(f[k], set()).add(fi)
            rev = (f[(k + 1) % len(f)], f[k])
            (dual_adj[fi].add(edge_owner[rev]) if rev in edge_owner else outer_adj.add(fi))
    return dual_adj, v2f, outer_adj


def auto_burgers_circuit(config, target=None, exclude=None, radius_clearance: int = 2, lattice=None) -> dict:
    """Find a clean simple CCW crystal circuit enclosing `target` (excluding `exclude`) and evaluate it."""
    d = config.defect if hasattr(config, "defect") else config
    if not is_orientable_surface(d):
        return _mobius_seam_circuit(config, lattice=lattice)

    P = np.asarray(d.positions, float)
    faces = _fast.frame_faces(d)
    k_outer = max(range(len(faces)), key=lambda i: len(faces[i]))
    int_faces = [f for i, f in enumerate(faces) if i != k_outer]
    f_charges = [_fast.loop_charge_fast(d, f) for f in int_faces]
    dual_adj, v2f, outer_adj = _dual_graph(int_faces)
    rim = config.rim() if (hasattr(config, "rim") and "reference" in d.info and "R" in d.info) else set(faces[k_outer])
    rim_faces = {fi for v in rim for fi in v2f.get(v, ())}
    has_wall = any(e.get("across") for e in d.edge.values())
    if has_wall:
        ctr = np.asarray(d.info.get("centre", P.mean(axis=0)), float)
        lmax = config.lattice.lmax if hasattr(config, "lattice") else 1.5
        irreg = {v for c in (config.cores() if hasattr(config, "cores") else _fast.cores(d))
                 for v in c["sites"] if _norm(P[v] - ctr) <= 3.0 * lmax} or {
            i for i in range(len(P)) if _norm(P[i] - ctr) <= 1.2 * lmax and i not in rim}
    elif hasattr(config, "core_sites") and "reference" in d.info and "R" in d.info:
        irreg = set(config.core_sites())
    else:
        irreg = {v for c in (config.cores() if hasattr(config, "cores") else _fast.cores(d)) for v in c["sites"]}

    tgt_pts, exc_pts = _normalize_points(target), _normalize_points(exclude)
    if not tgt_pts:
        t_faces = {fi for v in irreg for fi in v2f.get(v, ())} or {
            min(range(len(int_faces)), key=lambda i: _norm(P[int_faces[i]].mean(axis=0) - P.mean(axis=0)))}
        t_strict, t_chk = set(t_faces), ([P[v] for v in irreg] if irreg else [P[int_faces[next(iter(t_faces))]].mean(axis=0)])
    else:
        t_faces, t_strict, t_chk = set(), set(), []
        for pt in tgt_pts:
            inc, sing_fi, p_chk = _match_point_to_face(pt, config, int_faces, f_charges, P)
            t_faces.update(inc); t_strict.add(sing_fi); t_chk.append(p_chk)

    x_faces, x_strict, x_chk = set(), set(), []
    for pt in exc_pts:
        inc, sing_fi, p_chk = _match_point_to_face(pt, config, int_faces, f_charges, P)
        x_faces.update(inc); x_strict.add(sing_fi); x_chk.append(p_chk)

    for clr in range(max(0, int(radius_clearance)), -1, -1):
        seed = set(t_faces) if clr > 0 else set(t_strict)
        forb = set(rim_faces) | (set(x_faces) if clr > 0 else set(x_strict))
        if clr > 0 and x_faces:
            xv = {v for fi in x_faces for v in int_faces[fi]}
            for _ in range(clr - 1):
                xv = {w for v in xv for fi in v2f.get(v, ()) for w in int_faces[fi]}
            forb |= {fi for v in xv for fi in v2f.get(v, ())}
        if seed & forb:
            continue
        reg = _connect_seed(seed, dual_adj, forb)
        if reg is None:
            continue
        for _ in range(clr):
            reg |= {fi for v in {v for fi in reg for v in int_faces[fi]} for fi in v2f.get(v, ())} - forb
        filled = _fill_holes(len(int_faces), dual_adj, outer_adj, reg, set(x_strict) if clr == 0 else x_faces)
        cyc = None if filled is None else _extract_boundary_cycle(int_faces, filled)
        if cyc is None:
            continue
        poly = P[cyc[:-1]]
        if not (clr == 0 and any(v in cyc[:-1] for v in irreg if any(_norm(P[v] - p) < 1e-6 for p in t_chk))):
            if not all(_winding(p, poly) == 1 for p in t_chk):
                continue
        if any(_winding(p, poly) != 0 for p in x_chk):
            continue
        return burgers_circuit(config, cyc, lattice=lattice)
    raise ValueError("could not find a clean Burgers circuit separating target from exclude without hitting rim")
