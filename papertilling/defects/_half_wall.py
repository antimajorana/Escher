"""Surgical half-wall builder on a planar disk patch (`Configuration` with exact fractional holonomy)."""
from __future__ import annotations

import math
from collections import Counter

import networkx as nx
import numpy as np

from ..legacy import fusion_framework as ff
from ..legacy.lab_defects import Defect
from ._wall_endings import nearest_ending_holonomy
from .catalogue import DefectCatalogue

TWO_PI = 2.0 * math.pi


def _norm(v) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _rot(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s], [s, c]], float)


def _polar_phi(r: np.ndarray, u_hat: np.ndarray) -> float:
    if _norm(r) < 1e-9:
        return 0.0
    return math.atan2(float(u_hat[0] * r[1] - u_hat[1] * r[0]), float(r @ u_hat)) % TWO_PI


def _seg_cross(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray, tol: float = 1e-7) -> bool:
    a, b = p2 - p1, p4 - p3
    den = float(a[0] * b[1] - a[1] * b[0])
    if abs(den) < 1e-12:
        return False
    rp = p3 - p1
    u, v = float(rp[0] * b[1] - rp[1] * b[0]) / den, float(rp[0] * a[1] - rp[1] * a[0]) / den
    return (tol < u < 1.0 - tol) and (tol < v < 1.0 - tol)


def _in_corner_sector(edge: dict, face: list[int], idx_pos: int, vec_new: np.ndarray) -> bool:
    n = len(face)
    u, prev_v, next_v = face[idx_pos], face[(idx_pos - 1) % n], face[(idx_pos + 1) % n]
    ep, en = edge[(u, prev_v)]["vec"], edge[(u, next_v)]["vec"]
    a_prev, a_next = math.atan2(float(ep[1]), float(ep[0])) % TWO_PI, math.atan2(float(en[1]), float(en[0])) % TWO_PI
    a_new = math.atan2(float(vec_new[1]), float(vec_new[0])) % TWO_PI
    gap = (a_prev - a_next) % TWO_PI or TWO_PI
    return 1e-6 < ((a_new - a_next) % TWO_PI) < gap - 1e-6


def _lookup_wall_entry(cat: DefectCatalogue, wall_id: str):
    pools = {"Wa": (cat.domain_walls,), "Wc": (cat.mobius_seams,), "Wd": (cat.glide_breaking,),
             "Wb": (cat.twin_walls,)}.get(wall_id[:2], (cat.domain_walls, cat.glide_breaking, cat.mobius_seams, cat.twin_walls))
    for pool in pools:
        for entry in pool:
            if entry["id"] == wall_id:
                return entry
    raise ValueError(f"unknown wall id {wall_id!r} for {cat.group}")


def resolve_half_wall_spec(lat, wall_entry_or_spec, at=None, direction=None) -> dict:
    """Normalise a DefectEntry, wall id string, or spec dict into a validated half_wall spec."""
    wall_kind, wall_id, sigma, default_dir, default_at = "wall", None, None, None, (0.0, 0.0)
    if isinstance(wall_entry_or_spec, str):
        entry = _lookup_wall_entry(DefectCatalogue(lat.cell), wall_entry_or_spec)
        wall_id, wall_kind, sigma = entry["id"], entry["kind"], entry["sigma"]
        default_dir = entry.get("direction") or (entry.get("spec") or {}).get("direction")
    elif isinstance(wall_entry_or_spec, dict):
        w = wall_entry_or_spec
        if w.get("kind") in ("domain_wall", "twin_wall", "mobius_seam", "glide_breaking"):
            wall_id, wall_kind, sigma = w.get("id"), w["kind"], w["sigma"]
            default_dir = w.get("direction") or (w.get("spec") or {}).get("direction")
        else:
            wall_id = w.get("wall_id")
            default_at = w.get("at") if w.get("at") is not None else default_at
            default_dir = w.get("direction")
            if w.get("sigma") is not None:
                sigma = (np.asarray(w["sigma"][0], float), np.asarray(w["sigma"][1], float))
            if wall_id is not None and (sigma is None or default_dir is None):
                entry = _lookup_wall_entry(DefectCatalogue(lat.cell), wall_id)
                wall_kind, sigma = entry["kind"], (entry["sigma"] if sigma is None else sigma)
                default_dir = (entry.get("direction") or (entry.get("spec") or {}).get("direction")) if default_dir is None else default_dir
    else:
        raise TypeError("wall_entry_or_spec must be a wall DefectEntry, wall id str, or spec dict")
    if sigma is None:
        raise ValueError("half_wall spec must provide either wall_id or sigma")
    at_arr = np.asarray(default_at if at is None else at, float)
    if at_arr.shape != (2,):
        raise ValueError("at must be a 2D point (x, y)")
    u = np.asarray((1.0, 0.0) if (default_dir if direction is None else direction) is None else (default_dir if direction is None else direction), float)
    if _norm(u) < 1e-9:
        raise ValueError("half_wall direction must be non-zero")
    u_hat = u / _norm(u)
    ending, hol = nearest_ending_holonomy(lat, sigma, at_arr, wall_kind=wall_kind)
    out = {"kind": "half_wall", "at": at_arr, "direction": u_hat,
           "sigma": (np.asarray(sigma[0], float), np.asarray(sigma[1], float)), "ending": ending, "holonomy": hol}
    if wall_id is not None:
        out["wall_id"] = wall_id
    return out


def _add_chart_edges(edge: dict, charts: list[dict], L, pos_all: dict) -> None:
    for here in charts:
        for kl, i in here.items():
            pi = pos_all[kl]
            for k2, typ in L.nbrs(kl):
                j = here.get(k2)
                if j is not None and j != i:
                    v = pos_all[k2] - pi
                    edge[(i, j)] = dict(vec=v, type=typ)
                    edge[(j, i)] = dict(vec=-v, type=typ)


def _cut_pure_slip(lat, allk, at, ctr, u_hat, n_hat, t_rel, R):
    L, pos_all = lat.L, {k: lat.L.pos(k) for k in allk}
    sites, X_ref, phi_list, P_list, edge, idx_map = [], [], [], [], {}, {}
    has_shift = _norm(t_rel) > 1e-7
    for k in allk:
        p = pos_all[k]; r = p - at
        phi = 0.0 if (abs(float(r @ n_hat)) < 1e-9 and float(r @ u_hat) >= -1e-9) else _polar_phi(r, u_hat)
        P = p + (phi / TWO_PI) * t_rel
        if _norm(P - ctr) <= R:
            idx_map[k] = len(sites); sites.append((k, 0)); X_ref.append(p); phi_list.append(phi); P_list.append(P)
    for k, i in idx_map.items():
        p1 = X_ref[i]
        for k2, typ in L.nbrs(k):
            j = idx_map.get(k2)
            if j is None or j <= i:
                continue
            p2 = X_ref[j]
            if has_shift:
                d1, d2 = float((p1 - at) @ n_hat), float((p2 - at) @ n_hat)
                if (d1 >= -1e-9 and d2 < -1e-9) or (d2 >= -1e-9 and d1 < -1e-9):
                    s1, s2 = float((p1 - at) @ u_hat), float((p2 - at) @ u_hat)
                    if s1 + (d1 / (d1 - d2)) * (s2 - s1) > 1e-9:
                        continue
            v = p2 - p1
            edge[(i, j)] = dict(vec=v, type=typ); edge[(j, i)] = dict(vec=-v, type=typ)
    return sites, X_ref, phi_list, P_list, edge


def _cut_strip(lat, allk, at, ctr, u_hat, t_rel, tn, R):
    L, pos_all = lat.L, {k: lat.L.pos(k) for k in allk}
    Binv = np.linalg.inv(np.column_stack([-t_rel, u_hat]))
    ts_all = {k: Binv @ (pos_all[k] - at) for k in allk}
    sites, X_ref, phi_list, P_list, edge = [], [], [], [], {}
    if tn > 0.0:
        for k in allk:
            t, s = ts_all[k]
            if s > 1e-9 and -1e-9 <= t < 1.0 - 1e-9:
                continue
            p = pos_all[k]; phi = _polar_phi(p - at, u_hat)
            phi += TWO_PI if (s > 1e-9 and t >= 1.0 - 1e-9 and phi < math.pi) else 0.0
            P = p + (phi / TWO_PI) * t_rel
            if _norm(P - ctr) <= R:
                sites.append((k, 0)); X_ref.append(p); phi_list.append(phi); P_list.append(P)
        charts = [{k: i for i, (k, _) in enumerate(sites) if ts_all[k][1] <= 1e-9 or ts_all[k][0] < -1e-9},
                  {k: i for i, (k, _) in enumerate(sites) if ts_all[k][0] >= 1.0 - 1e-9}]
    else:
        orig, dup = {}, {}
        for k in allk:
            t, s = ts_all[k]; p = pos_all[k]; phi = _polar_phi(p - at, u_hat)
            if s > 1e-9 and -1e-9 <= t < 1.0 - 1e-9:
                if _norm(p + t_rel - ctr) <= R:
                    orig[k] = len(sites); sites.append((k, 0)); X_ref.append(p); phi_list.append(TWO_PI); P_list.append(p + t_rel)
                if _norm(p - ctr) <= R:
                    dup[k] = len(sites); sites.append((k, 1)); X_ref.append(p); phi_list.append(0.0); P_list.append(p)
            else:
                if s > 1e-9 and t < -1e-9 and phi < math.pi: phi += TWO_PI
                if s > 1e-9 and t >= 1.0 - 1e-9 and phi > math.pi: phi -= TWO_PI
                P = p + (phi / TWO_PI) * t_rel
                if _norm(P - ctr) <= R:
                    orig[k] = len(sites); sites.append((k, 0)); X_ref.append(p); phi_list.append(phi); P_list.append(P)
        cA = {k: i for k, i in orig.items() if ts_all[k][1] <= 1e-9 or ts_all[k][0] >= 1.0 - 1e-9}; cA.update(dup)
        cB = {k: i for k, i in orig.items() if ts_all[k][0] < -1e-9 or (ts_all[k][1] > 1e-9 and ts_all[k][0] < 1.0 - 1e-9)}
        charts = [cA, cB]
    _add_chart_edges(edge, charts, L, pos_all)
    return sites, X_ref, phi_list, P_list, edge


def _cut_rot(lat, allk, at, ctr, u_hat, alpha, t_rel, R):
    L, Theta, eps = lat.L, TWO_PI - alpha, 1e-7
    th0_cut = math.atan2(float(u_hat[1]), float(u_hat[0])) - 1e-6
    sites, X_ref, phi_list, P_list, edge, key_sites, core_i = [], [], [], [], {}, {}, None
    for k in allk:
        p = L.pos(k); r = p - at
        if _norm(r) < 1e-9:
            if _norm(at - ctr) <= R:
                core_i = len(sites); key_sites[(k, 0)] = core_i
                sites.append((k, 0)); X_ref.append(p); phi_list.append(0.0); P_list.append(at.copy())
            continue
        th_a = (math.atan2(float(r[1]), float(r[0])) - th0_cut) % TWO_PI
        m = 0
        while th_a + TWO_PI * m < Theta - eps:
            phi = th_a + TWO_PI * m
            P = at + _rot(phi * (TWO_PI / Theta) - phi) @ r + (phi / Theta) * t_rel
            if _norm(P - ctr) <= R:
                key_sites[(k, m)] = len(sites); sites.append((k, m)); X_ref.append(p); phi_list.append(phi); P_list.append(P)
            m += 1
    for (k, _), u in key_sites.items():
        p = X_ref[u]
        for k2, typ in L.nbrs(k):
            q = L.pos(k2)
            if _norm(q - at) < 1e-9:
                if core_i is not None:
                    v_vec = q - p; edge[(u, core_i)] = dict(vec=v_vec, type=typ); edge[(core_i, u)] = dict(vec=-v_vec, type=typ)
                continue
            if u == core_i:
                continue
            a, bb = p - at, q - at
            if abs(float(a[0] * bb[1] - a[1] * bb[0])) < 1e-9 and float(a @ bb) < 0.0:
                continue
            phq = phi_list[u] + math.atan2(float(a[0] * bb[1] - a[1] * bb[0]), float(a @ bb))
            if -eps < phq < Theta - eps:
                v = key_sites.get((k2, int(math.floor((phq + eps) / TWO_PI))))
                if v is not None and v != u:
                    v_vec = q - p; edge[(u, v)] = dict(vec=v_vec, type=typ); edge[(v, u)] = dict(vec=-v_vec, type=typ)
    return sites, X_ref, phi_list, P_list, edge


def _keep_main_component(sites, X_ref, phi_list, P_list, edge):
    G = nx.Graph(); G.add_nodes_from(range(len(sites))); G.add_edges_from([(u, v) for u, v in edge if u < v])
    main = max(nx.connected_components(G), key=len)
    if len(main) == len(sites):
        return sites, X_ref, phi_list, P_list, edge
    keep = sorted(main); remap = {old: new for new, old in enumerate(keep)}
    return ([sites[i] for i in keep], [X_ref[i] for i in keep], [phi_list[i] for i in keep],
            [P_list[i] for i in keep], {(remap[u], remap[v]): val for (u, v), val in edge.items() if u in remap and v in remap})


def _zip_half_wall(lat, sites, X_ref, phi_list, P_arr, edge, Rh, th, at, u_hat, n_hat, Theta, R, fast):
    from . import fast as _fast
    deg = [0] * len(sites)
    for u, v in edge:
        if u < v: deg[u] += 1; deg[v] += 1
    d = Defect("half_wall", sites, P_arr, edge, dict(reference=np.array(X_ref, float), centre=at, R=R))
    if lat.planar:
        f_outer = max(_fast.frame_faces(d, fast), key=len)
        active_faces, outer_set = [f_outer], set(f_outer)
    else:
        active_faces, outer_set = [], set(range(len(sites)))
    near = lambda i: (_norm(P_arr[i] - at) > 0.2 * lat.lmax and float((P_arr[i] - at) @ u_hat) > -0.2 * lat.lmax
                      and abs(float((P_arr[i] - at) @ n_hat)) < 1.35 * lat.lmax)
    bank_A = [i for i in outer_set if phi_list[i] < 0.65 * math.pi and near(i)]
    bank_B = [i for i in outer_set if phi_list[i] > Theta - 0.65 * math.pi and near(i)]
    cands = []
    for u in bank_B:
        qu = Rh @ X_ref[u] + th
        for v in bank_A:
            if u == v or (u, v) in edge: continue
            d_phys, d_draw = _norm(qu - X_ref[v]), _norm(P_arr[u] - P_arr[v])
            if d_phys <= 1.75 * lat.lmax or d_draw <= 1.75 * lat.lmax:
                cands.append((d_phys + 0.2 * d_draw, u, v))
    cands.sort()
    existing_segs = [(P_arr[a], P_arr[b]) for a, b in edge if a < b]
    Rh_T = Rh.T
    g_fwd, g_rev = (Rh.copy(), th.copy()), (Rh_T.copy(), -Rh_T @ th)
    for _, u, v in cands:
        if (u, v) in edge or (deg[u] >= lat.perfect_degree[sites[u][0][2]] and deg[v] >= lat.perfect_degree[sites[v][0][2]]):
            continue
        pu, pv = P_arr[u], P_arr[v]
        if any(_seg_cross(pu, pv, s1, s2) for s1, s2 in existing_segs): continue
        vec_uv, vec_vu = Rh_T @ (X_ref[v] - th) - X_ref[u], (Rh @ X_ref[u] + th) - X_ref[v]
        if lat.planar:
            hit = next((fi for fi, f in enumerate(active_faces)
                        if u in f and v in f and _in_corner_sector(edge, f, f.index(u), vec_uv)
                        and _in_corner_sector(edge, f, f.index(v), vec_vu)), None)
            if hit is None: continue
            f = active_faces[hit]; iu, iv = sorted((f.index(u), f.index(v)))
            active_faces[hit:hit + 1] = [f[iu:iv + 1], f[iv:] + f[:iu + 1]]
        edge[(u, v)] = dict(vec=vec_uv, type=0, g=g_fwd, seam=True, across=True)
        edge[(v, u)] = dict(vec=vec_vu, type=0, g=g_rev, seam=True, across=True)
        existing_segs.append((pu, pv)); deg[u] += 1; deg[v] += 1
    return Defect("half_wall", sites, P_arr, edge, dict(reference=np.array(X_ref, float), centre=at, R=R))


def build_half_wall_defect(lat, spec: dict, R: float | None = None, centre=None, check: bool = True, fast: bool | None = None):
    """Build a raw Defect (and report dict) for a single half_wall spec."""
    from . import fast as _fast
    at = np.asarray(spec["at"], float)
    u_hat = np.asarray(spec["direction"], float); u_hat /= _norm(u_hat)
    n_hat = np.array([-u_hat[1], u_hat[0]])
    Rh, th = np.asarray(spec["holonomy"][0], float), np.asarray(spec["holonomy"][1], float)
    alpha = math.atan2(float(Rh[1, 0]), float(Rh[0, 0]))
    alpha = math.pi if alpha <= -math.pi + 1e-9 else alpha
    Theta, t_rel = TWO_PI - alpha, (Rh - np.eye(2)) @ at + th
    R = max(8.5 * lat.lmax + _norm(t_rel), 4.0 * lat.scale) if R is None else float(R)
    ctr = at if centre is None else np.asarray(centre, float)
    allk = lat.L.sites_in_disk(ctr, R + _norm(ctr - at) + _norm(t_rel) + 2.5 * lat.lmax)
    if abs(alpha) <= 1e-7:
        tn = float(th @ n_hat)
        raw = _cut_pure_slip(lat, allk, at, ctr, u_hat, n_hat, t_rel, R) if abs(tn) < 1e-7 else _cut_strip(lat, allk, at, ctr, u_hat, t_rel, tn, R)
    else:
        raw = _cut_rot(lat, allk, at, ctr, u_hat, alpha, t_rel, R)
    sites, X_ref, phi_list, P_list, edge = _keep_main_component(*raw)
    P_arr = np.array(P_list, float)
    d = _zip_half_wall(lat, sites, X_ref, phi_list, P_arr, edge, Rh, th, at, u_hat, n_hat, Theta, R, fast) \
        if (abs(alpha) > 1e-7 or _norm(th) > 1e-7) else Defect("half_wall", sites, P_arr, edge, dict(reference=np.array(X_ref, float), centre=ctr, R=R))
    d.info.update(centre=ctr, R=R, picture="raw")
    expected_frank = float(math.degrees(alpha)) if abs(alpha) > 1e-7 else 0.0
    rep = dict(R=R, centre=ctr, expected_frank=expected_frank)
    if lat.planar:
        rep.update(frames_bad=_fast.check_frames(d, fast), euler=len(d.sites) - len(d.edges) + len(_fast.frame_faces(d, fast)),
                   frank=_fast.total_frank(d, fast), min_degree=int(d.degrees().min()))
        if check and (rep["frames_bad"] != 0 or rep["euler"] != 2 or abs(rep["frank"] - expected_frank) > 1e-2 or rep["min_degree"] < 1):
            raise ValueError(f"half-wall combinatorial check failed ({rep})")
    return d, rep


def seed_builder_from_defect(builder_cls, lat, d: Defect, centre: np.ndarray, R: float):
    """Initialise a _Builder instance from an existing half-wall Defect so point defects can follow."""
    B = builder_cls(lat.cell, lat.L, centre, float(R))
    B.label, B.alive = list(d.sites), [True] * len(d.sites)
    B.ref = [np.asarray(x, float).copy() for x in d.info["reference"]]
    B.X = [np.asarray(x, float).copy() for x in d.positions]
    B.nb = [{} for _ in d.sites]
    for (u, v), e in d.edge.items():
        B.nb[u][v] = {k: (val.copy() if isinstance(val, np.ndarray) else val) for k, val in e.items()}
    B.layers, B.touched, B.cone_sites, B.records, B.perfect_sites = Counter(k for k, _ in d.sites), set(), set(), [], len(d.sites)
    return B


def build_half_wall(lattice, wall_entry_or_spec, at=None, direction=None, R=None, picture: str = "relaxed", fast: bool | None = None):
    """Build a half-wall configuration terminating at `at` along `direction`."""
    from . import fast as _fast
    from .build import Configuration, lattice_data
    if picture not in ("relaxed", "raw"):
        raise ValueError("picture must be 'relaxed' or 'raw'")
    lat = lattice_data(lattice, fast)
    spec = resolve_half_wall_spec(lat, wall_entry_or_spec, at=at, direction=direction)
    d, rep = build_half_wall_defect(lat, spec, R=R, fast=fast)
    if picture == "relaxed":
        _fast.relax_picture(d, fast=fast)
        if lat.planar:
            from .draw import ensure_embedded
            ensure_embedded(d, overlaps=False)
    else:
        ff.mark_long_bonds(d)
    return Configuration(d, [spec], lat, rep)
