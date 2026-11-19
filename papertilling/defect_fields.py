"""Defect fields (Stage 3): find defects from bare coordinates, order parameters, strain / Vitek / Nye fields,
homogenised moduli with the Stroh dislocation field, and motion GIFs.

    import papertilling as pt
    det = pt.defect_fields.detect_defects_coords(pos, "honeycomb")     # pos: (N, 2) array, nothing else
    det["defects"][0]["uv"], det["defects"][0]["weber"]                 # [1, 0], '1/3[2 -1 -1 0]'

Conventions (same as the Rust core `papertilling_core::defect_fields`):
- faces are the bounded faces of the planar bond graph, counter-clockwise; vertices on the outer face (the rim) carry
  no charge;
- Frank charges in degrees from snapped corner angles: vertex `360 - sum corners`, face `sum corners - (p - 2) 180`
  (pentagon in honeycomb `+60`, heptagon `-60`);
- Burgers vector `burgers` = sum of snapped ideal bond vectors round a counter-clockwise loop (`closure_vector` of
  `burgers_circuit`, the convention of the defect builder), each undirected bond snapped once so
  `G(j->i) = -G(i->j)`, in a frame parallel-transported along the loop from `loop_path[0]` (whose chart `b` is in); `uv` in the cell basis (`b = u a1 + v a2`) of `lattice_data(lattice).cell` (or of the dict
  cell given). `burgers_fs_rh = -burgers` is the FS/RH Burgers vector of Hirth & Lothe (1982, ch. 1) for line
  direction +z; a Volterra / Stroh field jumping by +b counter-clockwise gives `burgers = -b`, `burgers_fs_rh = b`;
- `F` maps reference to current bonds, `eps = (F + F^T)/2 - I`, `E = (F^T F - I)/2`; Vitek arrows wrap `u_j - u_i`
  along `b` into `(-|b|/2, |b|/2]`; Nye `alpha(f) = (1/A_f) sum_{boundary f} G` (integrates to `burgers`; the FS/RH
  Nye tensor is `-alpha`);
- coordination charge (`voronoi_coordination`): `z_ideal - z` from Voronoi neighbours (`6 - z` for a triangular
  crystal, so a 5-7 pair is neutral); anti-plane screw field `u_z = (b / 2 pi) arg(x + p y)` (`screw_dislocation_field`).
Diagnostics carry `[PROVED]` / `[COMPUTATIONAL EVIDENCE]`; impossible inputs are refused (`REFUSED: ...`).

Every function uses the compiled `papertilling_rs` when it exports the binding, else the pure-Python code here and in
`_defect_fields_extra` (`PAPERTILLING_NO_RUST=1` forces it).
"""
from __future__ import annotations

import json
import math

import numpy as np

from ._defect_fields_extra import (barnett_lothe_stroh, export_motion_gif, interpolate_crystal_frames,
                                   screw_dislocation_field, voronoi_coordination)
from ._defect_fields_util import TWO_PI, _IDENT_FRAME, _cell_dict, _edge_list, _pos, _rs

__all__ = ["detect_defects_coords", "wigner_seitz_occupancy", "compute_nye_and_strain", "psi_n_order",
           "cna_signatures", "barnett_lothe_stroh", "screw_dislocation_field", "voronoi_coordination",
           "export_motion_gif", "interpolate_crystal_frames"]


# ----------------------------------------------------------------------------------------------- pure-Python geometry
class _Ref:
    def __init__(self, cell, frame):
        A = np.column_stack([cell["a1"], cell["a2"]])
        basis = np.asarray(cell["basis"], float)
        per_site, star = [[] for _ in basis], []

        def push(lst, v):
            if np.linalg.norm(v) > 1e-12 and not any(np.linalg.norm(w - v) < 1e-9 for w in lst):
                lst.append(v)
        for b in cell["bonds"]:
            d = basis[b["j"]] + A @ np.array([b["n1"], b["n2"]], float) - basis[b["i"]]
            push(per_site[b["i"]], d), push(per_site[b["j"]], -d), push(star, d), push(star, -d)
        corners, degrees = [], sorted({len(lst) for lst in per_site})
        for lst in per_site:
            # every run of consecutive ideal corners (a site that lost bonds has merged corners, 90 + 90 = 180)
            angs = sorted(math.atan2(v[1], v[0]) % TWO_PI for v in lst)
            m = len(angs)
            for k, a in enumerate(angs):
                for ln in range(1, max(m, 2)):
                    c = math.degrees((angs[k + ln] if k + ln < m else angs[k + ln - m] + TWO_PI) - a)
                    if c < 360.0 - 1e-6 and not any(abs(x - c) < 1e-6 for x in corners):
                        corners.append(c)
        self.A, self.basis, self.star = A, basis, np.array(star)
        self.corners, self.degrees = np.array(sorted(corners)), degrees
        self.lmax = float(np.max(np.linalg.norm(self.star, axis=1)))
        q = np.asarray(frame["q"], float)
        self.frame_angle = math.atan2(q[1, 0], q[0, 0])

    def snap_corner(self, deg):
        return float(self.corners[np.argmin(np.abs(self.corners - deg))])

    def default_cutoff(self):
        n = np.arange(-3, 4)
        T = np.stack(np.meshgrid(n, n), -1).reshape(-1, 2) @ self.A.T
        d = np.linalg.norm(self.basis[None, :, None] + T[:, None, None] - self.basis[None, None, :], axis=-1).ravel()
        d = d[d > self.lmax + 1e-6]
        return 0.5 * (self.lmax + float(d.min())) if d.size else 1.2 * self.lmax

    def nearest(self, d, th):
        c, s = math.cos(th), math.sin(th)
        rs = self.star @ np.array([[c, s], [-s, c]])
        return int(np.argmin(np.linalg.norm(rs - d, axis=1)))


def _graph(pos, edges, cutoff):
    if edges is None:
        from scipy.spatial import cKDTree
        prs = cKDTree(pos).query_pairs(cutoff, output_type="ndarray") if len(pos) else np.zeros((0, 2), int)
        edges = [(int(u), int(v)) for u, v in prs if np.linalg.norm(pos[v] - pos[u]) < cutoff]
    es = sorted({(min(u, v), max(u, v)) for u, v in edges if u != v})
    nbrs = [[] for _ in pos]
    for u, v in es:
        nbrs[u].append(v), nbrs[v].append(u)
    for u, lst in enumerate(nbrs):
        lst.sort(key=lambda w: math.atan2(pos[w][1] - pos[u][1], pos[w][0] - pos[u][0]) % TWO_PI)
    return es, nbrs


def _faces(pos, nbrs):
    idx = [{v: k for k, v in enumerate(lst)} for lst in nbrs]
    seen, faces, areas, rim, left = set(), [], [], [not lst for lst in nbrs], {}
    for u, lst in enumerate(nbrs):
        for v in lst:
            if (u, v) in seen:
                continue
            f, a, b = [], u, v
            while (a, b) not in seen:
                seen.add((a, b)), f.append(a)
                bl, k = nbrs[b], idx[b][a]
                a, b = b, bl[k - 1]
            p = pos[f]
            area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))
            if area > 1e-12:
                for k in range(len(f)):
                    left[(f[k], f[(k + 1) % len(f)])] = len(faces)
                faces.append(f), areas.append(area)
            else:
                for x in f:
                    rim[x] = True
    return faces, areas, rim, left


def _corner(pos, a, b, c):
    d = (math.atan2(*(pos[a] - pos[b])[::-1]) - math.atan2(*(pos[c] - pos[b])[::-1])) % TWO_PI
    return TWO_PI if d <= 1e-12 else d


def _face_nbrs(faces, left, fi):
    f = faces[fi]
    return [left[(f[(k + 1) % len(f)], f[k])] for k in range(len(f)) if (f[(k + 1) % len(f)], f[k]) in left]


def _charges(pos, es, nbrs, faces, rim, left, ref):
    r6 = lambda x: round(x * 1e6) / 1e6 + 0.0  # noqa: E731
    vq, bad_v = [0.0] * len(pos), [False] * len(pos)
    for v, lst in enumerate(nbrs):
        if rim[v] or not lst:
            continue
        m = len(lst)
        vq[v] = r6(360.0 - sum(ref.snap_corner(math.degrees(_corner(pos, lst[(k + 1) % m], v, lst[k]))) for k in range(m)))
        bad_v[v] = vq[v] != 0.0 or m not in ref.degrees
    fq = [r6(sum(ref.snap_corner(math.degrees(_corner(pos, f[k - 1], f[k], f[(k + 1) % len(f)]))) for k in range(len(f)))
             - (len(f) - 2) * 180.0) for f in faces]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nv = len(pos)
    pairs = [(u, v) for u, v in es if bad_v[u] and bad_v[v]]
    for fi, f in enumerate(faces):
        if fq[fi] != 0.0:
            pairs += [(v, nv + fi) for v in f if bad_v[v]]
            pairs += [(nv + fi, nv + nb) for nb in _face_nbrs(faces, left, fi) if fq[nb] != 0.0]
    pr, n = np.array(pairs, int).reshape(-1, 2), nv + len(faces)
    lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)), directed=False)[1]
    groups = {}  # insertion order = order of each cluster's smallest node, as in the Rust core
    for x in [v for v in range(nv) if bad_v[v]] + [nv + f for f in range(len(faces)) if fq[f] != 0.0]:
        groups.setdefault(lab[x], ([], []))[int(x >= nv)].append(x if x < nv else x - nv)
    clusters = []
    for vs, fs in groups.values():
        pts = [pos[v] for v in vs] + [pos[faces[f]].mean(axis=0) for f in fs]
        on_rim = any(rim[v] for f in fs for v in faces[f]) or any(rim[w] for v in vs for w in nbrs[v])
        clusters.append(dict(vertices=vs, faces=fs, frank_deg=r6(sum(vq[v] for v in vs) + sum(fq[f] for f in fs)),
                             centre=np.mean(pts, axis=0).tolist(), touches_rim=bool(on_rim)))
    return vq, fq, clusters


def _touches_rim(faces, left, rim, fi):
    f = faces[fi]
    return any((f[(k + 1) % len(f)], f[k]) not in left for k in range(len(f))) or any(rim[v] for v in f)


def _grow(faces, left, rim, v2f, seed, clr):
    reg = set(seed)
    for _ in range(clr):
        reg |= {g for fi in list(reg) for v in faces[fi] for g in v2f[v] if not _touches_rim(faces, left, rim, g)}
    outside = {f for f in range(len(faces)) if f not in reg and _touches_rim(faces, left, rim, f)}
    stack = list(outside)
    while stack:
        for nb in _face_nbrs(faces, left, stack.pop()):
            if nb not in outside and nb not in reg:
                outside.add(nb), stack.append(nb)
    return reg | {f for f in range(len(faces)) if f not in outside}


def _boundary_walk(faces, left, reg):
    out = {}
    for f in (faces[fi] for fi in reg):
        for u, v in zip(f, f[1:] + f[:1]):
            if left.get((v, u)) not in reg:
                out.setdefault(u, []).append(v)
    for lst in out.values():
        lst.sort(reverse=True)
    stack, walk = [min(out)] if out else [], []
    while stack:  # Hierholzer on the boundary darts (counter-clockwise)
        walk.append(stack.pop()) if not out.get(stack[-1]) else stack.append(out[stack[-1]].pop())
    return walk[::-1]


def _region_frank(faces, reg, walk, vq, fq):
    on = set(walk)
    inner = {v for f in reg for v in faces[f] if v not in on}
    return round((sum(fq[f] for f in reg) + sum(vq[v] for v in inner)) * 1e6) / 1e6 + 0.0


def _fit_angle(pos, ref, edges, th):
    for _ in range(3 if edges else 0):
        d = np.array([pos[v] - pos[u] for u, v in edges])
        s = ref.star[[ref.nearest(x, th) for x in d]]
        th += float(np.mean((np.arctan2(d[:, 1], d[:, 0]) - np.arctan2(s[:, 1], s[:, 0]) - th + math.pi) % TWO_PI - math.pi))
    return th


def _wrap(x):
    return (x + math.pi) % TWO_PI - math.pi


def _local_frame(pos, nbrs, ref, v, th):
    res = []
    for w in nbrs[v]:
        d = pos[w] - pos[v]
        s = ref.star[ref.nearest(d, th)]
        res.append(_wrap(math.atan2(d[1], d[0]) - math.atan2(s[1], s[0]) - th))
    return th + (float(np.mean(res)) if res else 0.0)


def _transported(pos, nbrs, ref, walk):
    """Closure round `walk` with a parallel-transported frame (Rust `transported_closure`): start at the vertex whose
    local frame is nearest `ref.frame_angle`, carry the frame vertex to vertex, snap each undirected edge once in the
    mean frame of its ends. Returns `(walk, b, th0)` or a REFUSED string."""
    n = len(walk) - 1
    if n < 3 or walk[0] != walk[n]:
        return "REFUSED: the loop is not a closed walk"
    a = np.sort(np.arctan2(ref.star[:, 1], ref.star[:, 0]) % TWO_PI)
    gaps = np.diff(np.append(a, a[0] + TWO_PI))
    tol = 0.45 * float(gaps[gaps > 1e-9].min())
    k0 = min(range(n), key=lambda k: abs(_wrap(_local_frame(pos, nbrs, ref, walk[k], ref.frame_angle) - ref.frame_angle)))
    w = [walk[(k0 + k) % n] for k in range(n + 1)]
    th0 = th = _local_frame(pos, nbrs, ref, w[0], ref.frame_angle)
    snap, b = {}, np.zeros(2)
    for k in range(n):
        if k:
            th = _local_frame(pos, nbrs, ref, w[k], th)
        u, v = w[k], w[k + 1]
        key = (min(u, v), max(u, v))
        if key not in snap:
            te = 0.5 * (th + _local_frame(pos, nbrs, ref, v, th))
            d = pos[key[1]] - pos[key[0]]
            sv = ref.star[ref.nearest(d, te)]
            res = abs(_wrap(math.atan2(d[1], d[0]) - math.atan2(sv[1], sv[0]) - te))
            if res > tol:
                return (f"REFUSED: bond ({u}, {v}) is {math.degrees(res):.1f} deg from every star direction in the "
                        f"transported frame (limit {math.degrees(tol):.1f})")
            snap[key] = sv
        b = b + (snap[(u, v)] if u < v else -snap[(v, u)])
    back = abs(_wrap(_local_frame(pos, nbrs, ref, w[0], th) - th0))
    if back > tol:
        return f"REFUSED: the transported frame turned by {math.degrees(back):.1f} deg round the loop (it must close)"
    return w, np.where(np.abs(b) < 1e-11, 0.0, b), th0


def _weber(group, cell, uv):
    a1, a2 = np.array(cell["a1"]), np.array(cell["a2"])
    l1, l2 = np.linalg.norm(a1), np.linalg.norm(a2)
    cg = float(a1 @ a2) / (l1 * l2)
    if group not in ("p3", "p3m1", "p31m", "p6", "p6m") or abs(l1 - l2) > 1e-5 * l1 or min(abs(cg + 0.5), abs(cg - 0.5)) > 1e-4:
        return None
    U, V = (uv[0], uv[1]) if abs(cg + 0.5) <= 1e-4 else (uv[0] + uv[1], uv[1])
    u3, v3, t3 = 2 * U - V, 2 * V - U, -(U + V)
    return f"[{u3 // 3} {v3 // 3} {t3 // 3} 0]" if u3 % 3 == 0 and v3 % 3 == 0 else f"1/3[{u3} {v3} {t3} 0]"


def _py_detect(pos, group, cell, frame, edges, cutoff, clearance):
    ref = _Ref(cell, frame)
    cut = float(cutoff) if cutoff is not None else ref.default_cutoff()
    es, nbrs = _graph(pos, edges, cut)
    faces, areas, rim, left = _faces(pos, nbrs)
    vq, fq, clusters = _charges(pos, es, nbrs, faces, rim, left, ref)
    v2f = [[] for _ in pos]
    for fi, f in enumerate(faces):
        for v in f:
            v2f[v].append(fi)
    seeds = [set(c["faces"]) | {g for v in c["vertices"] for g in v2f[v]} for c in clusters]
    charged = [i for i, c in enumerate(clusters) if c["frank_deg"] != 0.0 and not c["touches_rim"]]
    cand = sorted((float(np.linalg.norm(np.subtract(clusters[i]["centre"], clusters[j]["centre"]))), i, j)
                  for a, i in enumerate(charged) for j in charged[a + 1:]
                  if abs(clusters[i]["frank_deg"] + clusters[j]["frank_deg"]) < 1e-6)
    used, groups = set(), []
    for _, i, j in cand:
        if i not in used and j not in used:
            used |= {i, j}
            groups.append([i, j])
    groups += [[i] for i, c in enumerate(clusters) if i not in used and not c["touches_rim"]]
    defects = []
    for grp in sorted(groups):
        seed = set().union(*(seeds[i] for i in grp))
        if len(grp) == 2:
            seed |= _dual_path(faces, left, seeds[grp[0]], seeds[grp[1]])
        foreign = set().union(set(), *(seeds[i] for i in range(len(clusters)) if i not in grp))
        regs = ((_grow(faces, left, rim, v2f, seed, c), c) for c in range(clearance, -1, -1))
        reg, clean = next((r, not (r & foreign)) for r, c in regs if c == 0 or not (r & foreign))
        walk = _boundary_walk(faces, left, reg)
        frank = _region_frank(faces, reg, walk, vq, fq)
        d = dict(kind="disclination", clusters=grp, frank_deg=frank, loop_path=walk, burgers=None, burgers_fs_rh=None,
                 uv=None, weber=None,
                 centre=np.mean([clusters[i]["centre"] for i in grp], axis=0).tolist(),
                 frame_angle_deg=math.degrees(ref.frame_angle), reason="")
        if len(walk) < 4:
            d["reason"] = "REFUSED: no closed loop could be formed round the cluster (it touches the rim)"
        elif frank != 0.0:
            d["reason"] = "[PROVED] net Frank angle != 0 inside the loop: its closure depends on the frame; no Burgers vector"
        elif isinstance(tr := _transported(pos, nbrs, ref, walk), str):
            d["reason"] = tr
        else:
            walk, b, th = tr
            fr = np.linalg.solve(ref.A, b)
            d.update(loop_path=walk, burgers=b.tolist(), burgers_fs_rh=(0.0 - b).tolist(), frame_angle_deg=math.degrees(th))
            if np.all(np.abs(fr - np.round(fr)) < 1e-6):
                d["uv"] = [int(x) for x in np.round(fr)]
                d["weber"] = _weber(group, cell, d["uv"])
            d["kind"] = "neutral" if np.linalg.norm(b) < 1e-9 else ("dipole" if len(grp) == 2 else "dislocation")
            d["reason"] = ("[PROVED] exact closure of snapped bonds round a loop enclosing zero net Frank angle" if clean
                           else "[PROVED] loop also encloses other defective clusters (zero net Frank angle): the closure "
                           "is their combined Burgers vector")
        defects.append(d)
    return dict(edges=np.array(es, int).reshape(-1, 2), faces=faces, face_areas=areas, rim=rim, vertex_frank=vq,
                face_frank=fq, clusters=clusters, defects=defects, cutoff=cut)


def _dual_path(faces, left, src, dst):
    prev, frontier, seen = {}, list(src), set(src)
    while frontier:
        hit = next((f for f in frontier if f in dst), None)
        if hit is not None:
            path = [hit]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return set(path)
        nxt = []
        for f in frontier:
            for nb in (n for n in _face_nbrs(faces, left, f) if n not in seen):
                seen.add(nb), nxt.append(nb), prev.__setitem__(nb, f)
        frontier = nxt
    return set()


def detect_defects_coords(pos, lattice, *, group=None, frame=None, edges=None, cutoff=None, clearance=1) -> dict:
    """Find disclinations, 5-7 dislocations and +- dipoles from coordinates alone, with exact Burgers vectors.

    `lattice`: the reference lattice (name, Lattice, UnitCell, or a dict cell with a1, a2, basis, bonds). Bonds are
    every pair closer than `cutoff` (default: midway between the longest ideal bond and the nearest non-bonded
    distance) unless `edges` is given. Returns the bond graph, faces, Frank charges, clusters, and `defects` (kind,
    frank_deg, loop_path, burgers, uv, weber, reason)."""
    cell, g0 = _cell_dict(lattice)
    group, frame, P = group or g0 or "p1", frame or _IDENT_FRAME, _pos(pos)
    rs = _rs("detect_defects_coords_json")
    if rs is None:
        return _py_detect(P, group, cell, frame, _edge_list(edges), cutoff, int(clearance))
    out = json.loads(rs.detect_defects_coords_json(P.tolist(), group, json.dumps(cell), json.dumps(frame),
                                                   _edge_list(edges), cutoff, int(clearance)))
    out["edges"] = np.array(out["edges"], int).reshape(-1, 2)
    for d in out["defects"]:
        idx = d.pop("index", None)
        d["weber"] = idx.get("weber_str") if idx else None
    return out


def wigner_seitz_occupancy(ref_sites, actual_sites) -> dict:
    """Occupancy of each reference site's Wigner-Seitz cell: `occupancy`, `assignment`, `vacancies` (occupancy 0)
    and `interstitial_hosts` (occupancy >= 2)."""
    R, X = _pos(ref_sites), _pos(actual_sites)
    rs = _rs("wigner_seitz_occupancy_json")
    if rs is not None:
        return json.loads(rs.wigner_seitz_occupancy_json(R.tolist(), X.tolist()))
    from scipy.spatial import cKDTree
    assign = cKDTree(R).query(X)[1].astype(int) if len(X) else np.zeros(0, int)
    occ = np.bincount(assign, minlength=len(R))
    return dict(occupancy=occ.tolist(), assignment=assign.tolist(), vacancies=np.flatnonzero(occ == 0).tolist(),
                interstitial_hosts=np.flatnonzero(occ >= 2).tolist())


def _py_strain(R, X, es):
    n = len(X)
    A, M = np.zeros((n, 2, 2)), np.zeros((n, 2, 2))
    if len(es):
        e = np.asarray(es, int)
        dx, dX = X[e[:, 1]] - X[e[:, 0]], R[e[:, 1]] - R[e[:, 0]]
        for s in (e[:, 0], e[:, 1]):
            np.add.at(A, s, np.einsum("ki,kj->kij", dx, dX))
            np.add.at(M, s, np.einsum("ki,kj->kij", dX, dX))
    F = np.full((n, 2, 2), np.nan)
    det = np.linalg.det(M)
    ok = np.abs(det) > 1e-10 * np.trace(M, axis1=1, axis2=2) ** 2
    F[ok] = A[ok] @ np.linalg.inv(M[ok])
    eps = 0.5 * (F + F.transpose(0, 2, 1)) - np.eye(2)
    green = 0.5 * (F.transpose(0, 2, 1) @ F - np.eye(2))
    return dict(u=X - R, F=F, eps=eps, green=green)


def compute_nye_and_strain(pos, lattice, *, ref_pos=None, edges=None, cutoff=None, frame=None, burgers=None) -> dict:
    """Discrete Nye tensor per face (one far-field frame fitted on the outer ring) and, with `ref_pos`, the local
    deformation gradient `F`, strains `eps` / `green`, displacement `u` per site, and (with `burgers`) Vitek arrows.

    `nye_total = sum_f alpha(f) A_f` equals `boundary_closure` (the outer loop's Burgers vector) whenever the patch
    encloses zero net Frank angle; otherwise `nye_total` is None and `reason` says why."""
    cell, _ = _cell_dict(lattice)
    frame, P = frame or _IDENT_FRAME, _pos(pos)
    rs = _rs("nye_strain_json")
    if rs is not None:
        out = json.loads(rs.nye_strain_json(P.tolist(), json.dumps(cell), json.dumps(frame),
                                            None if ref_pos is None else _pos(ref_pos).tolist(), _edge_list(edges),
                                            cutoff, None if burgers is None else [float(x) for x in burgers]))
        st = out.pop("strain", None) or []
        for key, k in (("F", "f"), ("eps", "eps"), ("green", "green")) if st else ():
            out[key] = np.array([np.full((2, 2), np.nan) if s[k] is None else s[k] for s in st], float).reshape(-1, 2, 2)
        if st:
            out["u"] = np.array([s["u"] for s in st], float).reshape(-1, 2)
        out["edges"] = np.array(out["edges"], int).reshape(-1, 2)
        out["alpha"] = np.array(out["alpha"], float).reshape(-1, 2)
        if "vitek" in out:
            v = out["vitek"]
            out["vitek"] = dict(edges=np.array(v["edges"], int).reshape(-1, 2), du=np.array(v["du"], float).reshape(-1, 2),
                                along_b=np.array(v["along_b"], float), wrapped=np.array(v["wrapped"], float))
        return out
    ref = _Ref(cell, frame)
    es, nbrs = _graph(P, _edge_list(edges), float(cutoff) if cutoff is not None else ref.default_cutoff())
    faces, areas, rim, left = _faces(P, nbrs)
    th = _fit_angle(P, ref, [(u, v) for u, v in es if rim[u] and rim[v]], ref.frame_angle)
    snap = {(u, v): ref.star[ref.nearest(P[v] - P[u], th)] for u, v in es}
    gdir = lambda u, v: snap[(u, v)] if u < v else -snap[(v, u)]  # noqa: E731
    alpha = [sum((gdir(f[k], f[(k + 1) % len(f)]) for k in range(len(f))), np.zeros(2)) / a for f, a in zip(faces, areas)]
    reg = set(range(len(faces)))
    walk = _boundary_walk(faces, left, reg)
    vq, fq, _ = _charges(P, es, nbrs, faces, rim, left, ref)
    net = _region_frank(faces, reg, walk, vq, fq)
    closure = sum((gdir(u, v) for u, v in zip(walk[:-1], walk[1:])), np.zeros(2))
    total = sum((al * a for al, a in zip(alpha, areas)), np.zeros(2)).tolist() if net == 0.0 else None
    out = dict(edges=np.array(es, int).reshape(-1, 2), faces=faces, areas=areas, alpha=np.array(alpha).reshape(-1, 2),
               frame_angle_deg=math.degrees(th), boundary_walk=walk, net_frank_deg=net, nye_total=total,
               boundary_closure=closure.tolist(),
               reason=("[PROVED] antisymmetric snapping telescopes: sum_f alpha A_f = boundary closure" if total is not None
                       else f"[PROVED] region encloses net Frank angle {net} deg: the Burgers vector depends on the frame"))
    if ref_pos is not None:
        R = _pos(ref_pos)
        out.update(_py_strain(R, P, es))
        if burgers is not None:
            b = np.asarray(burgers, float)
            lb = float(np.linalg.norm(b))
            e = np.asarray(es, int).reshape(-1, 2)
            du = out["u"][e[:, 1]] - out["u"][e[:, 0]]
            along = du @ (b / lb)
            out["vitek"] = dict(edges=e, du=du, along_b=along, wrapped=along - lb * np.ceil(along / lb - 0.5))
    return out


def psi_n_order(pos, n, *, edges=None, cutoff=None) -> np.ndarray:
    """Bond-orientational order `psi_n(i) = mean_j exp(i n theta_ij)` (complex, 0 for isolated sites). Bonds: `edges`,
    or all pairs closer than `cutoff`."""
    P = _pos(pos)
    if edges is None and cutoff is None:
        raise ValueError("give edges or cutoff")
    rs = _rs("psi_n_order_list")
    if rs is not None:
        z = np.array(rs.psi_n_order_list(P.tolist(), int(n), _edge_list(edges), cutoff), float).reshape(-1, 2)
        return z[:, 0] + 1j * z[:, 1]
    es, nbrs = _graph(P, _edge_list(edges), float(cutoff or 0.0))
    out = np.zeros(len(P), complex)
    for i, lst in enumerate(nbrs):
        if lst:
            d = P[lst] - P[i]
            out[i] = np.mean(np.exp(1j * n * np.arctan2(d[:, 1], d[:, 0])))
    return out


def cna_signatures(pos, *, edges=None, cutoff=None) -> dict:
    """2D common-neighbour analysis: per bond `(n_common, n_bonds, l_chain)` and per site the signature counts."""
    P = _pos(pos)
    if edges is None and cutoff is None:
        raise ValueError("give edges or cutoff")
    rs = _rs("cna_signatures_json")
    if rs is not None:
        out = json.loads(rs.cna_signatures_json(P.tolist(), _edge_list(edges), cutoff))
        return dict(edges=[tuple(e) for e in out["edges"]], signatures=[tuple(s) for s in out["signatures"]],
                    per_site=[{tuple(int(x) for x in k.split(",")): v for k, v in d.items()} for d in out["per_site"]])
    es, nbrs = _graph(P, _edge_list(edges), float(cutoff or 0.0))
    sets, eset = [set(lst) for lst in nbrs], set(es)
    sigs, per_site = [], [dict() for _ in P]
    for i, j in es:
        com = sorted(sets[i] & sets[j])
        bonds = [(x, y) for a, x in enumerate(com) for y in com[a + 1:] if (min(x, y), max(x, y)) in eset]
        sig = (len(com), len(bonds), _longest_chain(com, bonds))
        sigs.append(sig)
        for s in (i, j):
            per_site[s][sig] = per_site[s].get(sig, 0) + 1
    return dict(edges=es, signatures=sigs, per_site=per_site)


def _longest_chain(nodes, bonds):
    def dfs(cur, used):
        best = 0
        for k, (a, b) in enumerate(bonds):
            if k not in used and cur in (a, b):
                best = max(best, 1 + dfs(b if a == cur else a, used | {k}))
        return best
    return max((dfs(s, frozenset()) for s in nodes), default=0)


