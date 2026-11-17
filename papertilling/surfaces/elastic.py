"""3D elastic relaxation (stretching + orientation-safe hinge bending) for surfaces and defect sheets."""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import minimize

from .. import _rust
from .pieces import LatticeData


class ElasticResult(dict):
    """Result of 3D elastic relaxation supporting both dict-key and attribute access."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


def _canonical_gauge(X: np.ndarray, is_proj: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fix rigid-body gauge deterministically; return `(X_aligned, com, R)`."""
    pts = np.asarray(X, float).copy()
    if len(pts) == 0:
        return pts, np.zeros(3, float), np.eye(3, float)
    com = np.zeros(3, float) if is_proj else pts.mean(axis=0)
    pts -= com
    if len(pts) < 3:
        return pts, com, np.eye(3, float)
    sample = np.vstack([pts, -pts]) if is_proj else pts
    evals, evecs = np.linalg.eigh((sample.T @ sample) / len(sample))
    order = np.argsort(evals)[::-1]
    v0, v1 = evecs[:, order[0]].copy(), evecs[:, order[1]].copy()
    for v in (v0, v1):
        vmax = float(np.max(np.abs(v)))
        if vmax > 1e-15 and v[next(j for j in range(3) if abs(v[j]) >= vmax - 1e-12)] < 0.0:
            v *= -1.0
    v1 -= float(v1 @ v0) * v0
    n1 = float(np.linalg.norm(v1))
    if n1 <= 1e-12:
        e_min = int(np.argmin(np.abs(v0)))
        v1 = np.eye(3, float)[e_min] - float(v0[e_min]) * v0
        n1 = float(np.linalg.norm(v1))
    v1 /= n1
    v2 = np.cross(v0, v1)
    v2 /= max(float(np.linalg.norm(v2)), 1e-15)
    return pts @ np.column_stack([v0, v1, v2]), com, np.column_stack([v0, v1, v2])


def _cyclic_polygon_radii(ls: list[float]) -> list[float]:
    """Centroid-to-corner distances of the flat cyclic polygon with edge lengths `ls`."""
    d, a = len(ls), np.asarray(ls, float)
    if float(a.max() - a.min()) <= 1e-7 * max(float(a.mean()), 1e-12):
        return [float(a.mean()) / (2.0 * math.sin(math.pi / d))] * d
    imax = int(np.argmax(a))
    lmax, others = float(a[imax]), [float(a[m]) for m in range(d) if m != imax]
    if lmax >= sum(others) - 1e-9:
        return [float(a.mean()) / (2.0 * math.sin(math.pi / d))] * d
    r_min = 0.5 * lmax + 1e-9
    outside = sum(2.0 * math.asin(min(1.0, l / (2.0 * r_min))) for l in others) < math.pi
    lo, hi = r_min, max(float(a.sum()), 1.0)
    for _ in range(72):
        mid = 0.5 * (lo + hi)
        s_oth = sum(2.0 * math.asin(min(1.0, l / (2.0 * mid))) for l in others)
        s_max = 2.0 * math.asin(min(1.0, lmax / (2.0 * mid)))
        lo, hi = (mid, hi) if ((s_max - s_oth > 0.0) if outside else (s_max + s_oth > 2.0 * math.pi)) else (lo, mid)
    R, th, pts = 0.5 * (lo + hi), 0.0, []
    for m, l in enumerate(a):
        da = 2.0 * math.asin(min(1.0, float(l) / (2.0 * R)))
        pts.append([R * math.cos(th), R * math.sin(th)])
        th += -da if (outside and m == imax) else da
    P = np.asarray(pts, float)
    return [float(np.linalg.norm(p - P.mean(axis=0))) for p in P]


def _match_template(ls: list[float], tpls: list[tuple[list[float], list[float]]]) -> list[float] | None:
    d = len(ls)
    for t_ls, t_rs in tpls:
        if len(t_ls) != d:
            continue
        for sl, sr in ((t_ls, t_rs), (t_ls[::-1], [t_rs[0]] + t_rs[1:][::-1])):
            for sh in range(d):
                if max(abs(a - b) for a, b in zip(ls, sl[sh:] + sl[:sh])) < 1e-5:
                    return sr[sh:] + sr[:sh]
    return None


def _lattice_face_templates(lattice) -> list[tuple[list[float], list[float]]]:
    if lattice is None:
        return []
    from ..lattice import Lattice
    lat = lattice if isinstance(lattice, LatticeData) else LatticeData(
        lattice if isinstance(lattice, Lattice) else Lattice.from_cell(lattice))
    tpls: list[tuple[list[float], list[float]]] = []
    for labs, _ in lat.faces:
        P = np.array([lat.pos(lb) for lb in labs], float)
        c, k = P.mean(axis=0), len(P)
        tpls.append(([float(np.linalg.norm(P[(m + 1) % k] - P[m])) for m in range(k)],
                     [float(np.linalg.norm(P[m] - c)) for m in range(k)]))
    return tpls


def _face_spoke_lengths(faces, bmap, bvec, uv, cone_faces, lattice) -> dict[int, list[float]]:
    """Flat-template centroid-to-corner spoke rest lengths for every face with `d >= 4`."""
    tpls: list[tuple[list[float], list[float]]] | None = None
    out: dict[int, list[float]] = {}
    for fi, f in enumerate(faces):
        d = len(f)
        if d < 4:
            continue
        ls = [bmap[(min(f[m], f[(m + 1) % d]), max(f[m], f[(m + 1) % d]))] for m in range(d)]
        if bvec is not None and all((f[m], f[(m + 1) % d]) in bvec for m in range(d)):
            vecs = [bvec[(f[m], f[(m + 1) % d])] for m in range(d)]
            if float(np.linalg.norm(np.sum(vecs, axis=0))) < 1e-5:
                pts = np.vstack([np.zeros((1, 2)), np.cumsum(vecs[:-1], axis=0)])
                out[fi] = [float(np.linalg.norm(p - pts.mean(axis=0))) for p in pts]
                continue
        if uv is not None and all(abs(float(np.linalg.norm(uv[f[(m + 1) % d]] - uv[f[m]])) - ls[m]) < 1e-5 for m in range(d)):
            pts = uv[f]
            out[fi] = [float(np.linalg.norm(p - pts.mean(axis=0))) for p in pts]
            continue
        if fi not in cone_faces and lattice is not None:
            if tpls is None:
                tpls = _lattice_face_templates(lattice)
            hit = _match_template(ls, tpls) if tpls else None
            if hit is not None:
                out[fi] = hit
                continue
        out[fi] = _cyclic_polygon_radii(ls)
    return out


def _build_hinges(faces, vc_map: dict[int, int]):
    """Build inter-face dihedral hinges and intra-face spoke fan hinges."""
    edge_inc: dict[tuple[int, int], list[tuple[int, int]]] = {}
    hp, hq, ha, hb, hw = [], [], [], [], []
    for fi, f in enumerate(faces):
        d = len(f)
        if d < 3:
            continue
        for m in range(d):
            u, v = f[m], f[(m + 1) % d]
            edge_inc.setdefault((min(u, v), max(u, v)), []).append((fi, f[(m + 2) % 3] if d == 3 else vc_map[fi]))
        if d >= 4:
            vc = vc_map[fi]
            for m in range(d):
                hp.append(vc); hq.append(f[m]); ha.append(f[(m - 1) % d]); hb.append(f[(m + 1) % d]); hw.append(0.5)
    for (u, v), inc in sorted(edge_inc.items()):
        if len(inc) == 2 and inc[0][0] != inc[1][0]:
            hp.append(u); hq.append(v); ha.append(inc[0][1]); hb.append(inc[1][1]); hw.append(1.0)
    return tuple(np.array(arr, int) for arr in (hp, hq, ha, hb)) + (np.array(hw, float),)


def _eval_energy_grad(P, su, sv, sl0, sw, hp, hq, ha, hb, hw, kappa_eff: float):
    """Compute `(E_stretch, E_bend, grad_P)` for full coordinate array `P`."""
    gP = np.zeros_like(P)
    e_s = 0.0
    if len(su):
        dvec = P[su] - P[sv]
        lens = np.sqrt((dvec * dvec).sum(axis=1) + 1e-24)
        diff = lens - sl0
        e_s = float(0.5 * np.sum(sw * diff * diff))
        coeff = (sw * diff / lens)[:, None] * dvec
        np.add.at(gP, su, coeff)
        np.add.at(gP, sv, -coeff)
    e_b = 0.0
    if kappa_eff > 0.0 and len(hp):
        u_v, va, vb = P[hq] - P[hp], P[ha] - P[hp], P[hb] - P[hp]
        na_r, nb_r = np.cross(u_v, va), np.cross(vb, u_v)
        la = np.sqrt((na_r * na_r).sum(axis=1) + 1e-20)
        lb = np.sqrt((nb_r * nb_r).sum(axis=1) + 1e-20)
        na, nb = na_r / la[:, None], nb_r / lb[:, None]
        wh = 0.5 * kappa_eff * hw
        e_b = float(np.sum(wh * (1.0 - np.einsum("ij,ij->i", na, nb))))
        wa, wb = -wh[:, None] * nb, -wh[:, None] * na
        ga = (wa - np.einsum("ij,ij->i", wa, na)[:, None] * na) / la[:, None]
        gb = (wb - np.einsum("ij,ij->i", wb, nb)[:, None] * nb) / lb[:, None]
        gq = np.cross(va, ga) + np.cross(gb, vb)
        gao, gbo = np.cross(ga, u_v), np.cross(u_v, gb)
        gp = -(gq + gao + gbo)
        for idx, g in ((hp, gp), (hq, gq), (ha, gao), (hb, gbo)):
            np.add.at(gP, idx, g)
    return e_s, e_b, gP


def _run_elastic(V: int, X0: np.ndarray, edges: list[tuple[int, int]], edge_l0: np.ndarray,
                 bond_edge_idx: list[int], faces: list[list[int]], spoke_lengths: dict[int, list[float]],
                 kappa_b: float, max_iter: int, is_proj: bool = False, is_open_sheet: bool = False) -> ElasticResult:
    """Run L-BFGS-B relaxation and assemble ElasticResult."""
    n_pts = 2 * V if is_proj else V
    l_mean = float(np.mean(edge_l0)) if len(edge_l0) else 1.0
    kappa_eff = float(kappa_b) * l_mean * l_mean
    vc_map: dict[int, int] = {}
    spokes: list[tuple[int, int, float]] = []
    C0: list[np.ndarray] = []
    for fi, f in enumerate(faces):
        if len(f) >= 4:
            vc = n_pts + len(vc_map)
            vc_map[fi] = vc
            C0.append(X0[f].mean(axis=0))
            spokes.extend((vc, v_idx, spoke_lengths[fi][m]) for m, v_idx in enumerate(f))
    n_e = len(edges)
    su = np.array([e[0] for e in edges] + [s[0] for s in spokes], int)
    sv = np.array([e[1] for e in edges] + [s[1] for s in spokes], int)
    sl0 = np.concatenate([edge_l0, np.array([s[2] for s in spokes], float)]) if spokes else edge_l0.copy()
    sw = np.ones(len(su), float) * (0.5 if is_proj else 1.0)
    hp, hq, ha, hb, hw = _build_hinges(faces, vc_map)
    if is_proj and len(hw):
        hw = 0.5 * hw

    def unpack(x: np.ndarray) -> np.ndarray:
        pts = x.reshape(-1, 3)
        if not is_proj:
            return pts
        return np.vstack([pts[:V], -pts[:V], pts[V:]]) if len(pts) > V else np.vstack([pts[:V], -pts[:V]])

    def fg(x: np.ndarray):
        P = unpack(x)
        es, eb, gP = _eval_energy_grad(P, su, sv, sl0, sw, hp, hq, ha, hb, hw, kappa_eff)
        if is_proj:
            g_out = np.vstack([gP[:V] - gP[V : 2 * V], gP[2 * V :]]) if len(C0) else (gP[:V] - gP[V : 2 * V])
        else:
            g_out = gP
        return es + eb, g_out.ravel()

    x0 = (np.vstack([X0[:V], np.array(C0, float)]) if C0 else X0[:V]).ravel()
    if max_iter <= 0 or V == 0:
        x_opt, n_iter, sol_ok = x0, 0, True
    else:
        sol = minimize(fg, x0, method="L-BFGS-B", jac=True,
                       options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-7})
        x_opt, n_iter, sol_ok = sol.x, int(sol.nit), bool(sol.success)
    _, g_fin = fg(x_opt)
    rms_g = float(np.sqrt(np.mean(g_fin * g_fin))) if len(g_fin) else 0.0
    converged = bool(sol_ok or rms_g <= 1e-6)

    P_fin = unpack(x_opt)
    X, com, R = _canonical_gauge(P_fin[:V], is_proj=is_proj)
    cover_xyz = np.vstack([X, -X]) if is_proj else None
    base_pts = cover_xyz if is_proj else X
    P_gauge = np.vstack([base_pts, (P_fin[n_pts:] - com) @ R]) if C0 else base_pts
    es, eb, _ = _eval_energy_grad(P_gauge, su, sv, sl0, sw, hp, hq, ha, hb, hw, kappa_eff)

    r_norms = np.linalg.norm(X, axis=1)
    r_mean = float(np.mean(r_norms)) if V else 0.0
    asph = float(np.var(r_norms) / (r_mean * r_mean)) if (V and r_mean > 1e-15) else 0.0
    fvk_gamma = float((r_mean * r_mean) / kappa_eff) if (kappa_eff > 0.0 and r_mean > 0.0) else float("inf")

    if n_e:
        elens = np.linalg.norm(P_gauge[su[:n_e]] - P_gauge[sv[:n_e]], axis=1)
        edge_strain = (elens - edge_l0) / np.maximum(edge_l0, 1e-12)
        bond_strain = np.asarray([edge_strain[idx] for idx in bond_edge_idx], float)
    else:
        edge_strain, bond_strain = np.zeros(0, float), np.zeros(0, float)

    ssum, scnt = np.zeros(V, float), np.zeros(V, int)
    for idx, (u, v) in enumerate(edges):
        for w in (u, v):
            wi = w % V if is_proj else w
            ssum[wi] += edge_strain[idx]
            scnt[wi] += 1
    site_strain = np.where(scnt > 0, ssum / np.maximum(scnt, 1), 0.0)
    radial_buckling = (
        (X[:, 2] - float(np.mean(X[:, 2]))) / max(l_mean, 1e-12)
        if is_open_sheet
        else (r_norms - r_mean) / max(r_mean, 1e-12)
    )
    return ElasticResult(
        xyz=X, cover_xyz=cover_xyz, stretch_energy=float(es), bend_energy=float(eb),
        total_energy=float(es + eb), r_mean=r_mean, fvk_gamma=fvk_gamma, asphericity=asph,
        bond_strain=bond_strain, site_strain=site_strain, radial_buckling=radial_buckling,
        converged=converged, n_iter=n_iter,
    )


def _result_from_json(raw: str) -> ElasticResult:
    d = json.loads(raw)
    for k in ("xyz", "cover_xyz", "bond_strain", "site_strain", "radial_buckling"):
        if d.get(k) is not None:
            d[k] = np.asarray(d[k], float)
    return ElasticResult(d)


def _apply_surface_result(surface, res: ElasticResult) -> ElasticResult:
    surface.xyz, surface._elastic_relaxed = res["xyz"], True
    if getattr(surface, "cover", None) is not None and res.get("cover_xyz") is not None:
        surface.cover.xyz = res["cover_xyz"]
    return res


def relax_surface_3d(surface, kappa_b: float = 0.02, max_iter: int = 400) -> ElasticResult:
    """Relax a `Surface` in 3D under spring stretching and orientation-safe hinge bending."""
    if kappa_b < 0.0 or not math.isfinite(kappa_b):
        raise ValueError("kappa_b must be finite and non-negative")
    rs = _rust.get()
    if rs is not None and hasattr(rs, "relax_surface_3d_json"):
        cov = getattr(surface, "cover", None)
        cone_dicts = lambda cs: [{k: v for k, v in c.items() if k in ("site", "sites", "face", "faces", "angle")} for c in cs]
        payload = json.dumps(dict(
            kind=surface.kind, uv=np.asarray(surface.uv, float).tolist(),
            xyz=None if surface.xyz is None else np.asarray(surface.xyz, float).tolist(),
            bonds=[(int(a), int(b), [float(v[0]), float(v[1])]) for a, b, v in surface.bonds],
            faces=[[int(x) for x in f] for f in surface.faces],
            cones=cone_dicts(surface.cones),
            chi=int(surface.chi), orientable=bool(surface.orientable), closed=bool(surface.closed),
            cover=None if cov is None else dict(
                xyz=np.asarray(cov.xyz, float).tolist(),
                bonds=[(int(a), int(b), [float(v[0]), float(v[1])]) for a, b, v in cov.bonds],
                faces=[[int(x) for x in f] for f in cov.faces],
                cones=cone_dicts(getattr(cov, "cones", [])),
            ),
            kappa_b=float(kappa_b), max_iter=int(max_iter),
        ))
        return _apply_surface_result(surface, _result_from_json(rs.relax_surface_3d_json(payload)))

    V = len(surface.sites)
    is_proj = bool(getattr(surface, "projective", False)
                   or (getattr(surface, "cover", None) is not None and not surface.orientable and surface.chi == 1))
    use_surf = surface.cover if is_proj else surface
    bmap, bvec = {}, {}
    for a, b, v in use_surf.bonds:
        vv = np.asarray(v, float)
        bmap.setdefault((min(int(a), int(b)), max(int(a), int(b))), math.hypot(float(vv[0]), float(vv[1])))
        bvec[(int(a), int(b))], bvec[(int(b), int(a))] = vv, -vv
    edges = sorted(bmap.keys())
    e_idx, edge_l0 = {e: i for i, e in enumerate(edges)}, np.array([bmap[e] for e in edges], float)
    if is_proj:
        cbonds, bond_edge_idx = surface.cover.bonds, []
        for i, (a, b, _) in enumerate(surface.bonds):
            ca, cb = (cbonds[2 * i][0], cbonds[2 * i][1]) if 2 * i < len(cbonds) else (a, b)
            if {ca % V, cb % V} != {a % V, b % V}:
                ca, cb = next((u, w) for u, w, _ in cbonds if {u % V, w % V} == {a % V, b % V})
            bond_edge_idx.append(e_idx[(min(ca, cb), max(ca, cb))])
    else:
        bond_edge_idx = [e_idx[(min(int(a), int(b)), max(int(a), int(b)))] for a, b, _ in surface.bonds]

    cone_faces = {int(x) for c in use_surf.cones for k in ("face", "faces") if c.get(k) is not None
                  for x in ([c[k]] if k == "face" else c[k])}
    uv_use = None if is_proj else np.asarray(use_surf.uv, float)
    spoke_lengths = _face_spoke_lengths(use_surf.faces, bmap, bvec, uv_use, cone_faces, surface.lattice)
    if is_proj:
        X0 = np.asarray(surface.cover.xyz, float).copy()
    elif surface.xyz is not None:
        X0 = np.asarray(surface.xyz, float).copy()
    else:
        uv = np.asarray(surface.uv, float)
        X0 = np.column_stack([uv, 0.05 * np.sin(uv[:, 0]) * np.cos(uv[:, 1])])
    if edges and len(X0):
        m_init = float(np.mean(np.linalg.norm(X0[[e[0] for e in edges]] - X0[[e[1] for e in edges]], axis=1)))
        if m_init > 1e-12 and abs(m_init - float(np.mean(edge_l0))) / float(np.mean(edge_l0)) > 1e-3:
            X0 *= float(np.mean(edge_l0)) / m_init

    res = _run_elastic(V, X0, edges, edge_l0, bond_edge_idx, use_surf.faces,
                       spoke_lengths, kappa_b, max_iter, is_proj=is_proj, is_open_sheet=False)
    return _apply_surface_result(surface, res)


def _interior_faces(d, planar: bool) -> list[list[int]]:
    """Extract interior face cycles of a defect `Defect`, excluding the outer boundary walk."""
    if not planar or not d.edges:
        return []
    from ..defects import fast as _fast
    walks, P, n = _fast.frame_faces(d), np.asarray(d.positions, float), len(d.sites)
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in d.edges:
        adj[u].append(v); adj[v].append(u)
    comp, cid = [-1] * n, 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack, comp[s] = [s], cid
        while stack:
            for w in adj[stack.pop()]:
                if comp[w] < 0:
                    comp[w] = cid; stack.append(w)
        cid += 1
    by_comp: dict[int, list[int]] = {}
    for fi, f in enumerate(walks):
        if f:
            by_comp.setdefault(comp[f[0]], []).append(fi)
    drop: set[int] = set()
    for cs in by_comp.values():
        if len(cs) <= 1:
            drop.update(cs)
        else:
            drop.add(max(cs, key=lambda i: (len(walks[i]), -0.5 * float(np.sum(
                P[walks[i], 0] * np.roll(P[walks[i], 1], -1) - np.roll(P[walks[i], 0], -1) * P[walks[i], 1])))))
    return [f for i, f in enumerate(walks) if i not in drop and len(f) >= 3]


def _defect_sheet_inputs(conf):
    """Extract defect edges, bond rest lengths, interior faces, and spoke lengths."""
    d, cell = conf.defect, conf.lattice.cell
    type_lens: dict[int, list[float]] = {}
    for i, j, n1, n2, typ in cell.bonds:
        dv = np.asarray(cell.basis[j], float) + cell.A @ np.array([n1, n2], float) - np.asarray(cell.basis[i], float)
        type_lens.setdefault(int(typ), []).append(math.hypot(float(dv[0]), float(dv[1])))
    kind_len = {k: float(np.mean(vs)) for k, vs in type_lens.items()}
    edges = [(int(u), int(v)) for u, v in d.edges]
    bmap, bvec, edge_l0 = {}, {}, np.zeros(len(edges), float)
    for idx, (u, v) in enumerate(edges):
        ed = d.edge[(u, v)]
        if "vec" in ed and not ed.get("across"):
            vv = np.asarray(ed["vec"], float)
            l0 = max(math.hypot(float(vv[0]), float(vv[1])), 1e-6)
            bvec[(u, v)], bvec[(v, u)] = vv, -vv
        else:
            l0 = kind_len[ed["type"]] if ed.get("type") in kind_len else float(conf.lattice.lmean)
        edge_l0[idx] = bmap[(min(u, v), max(u, v))] = l0
    faces = _interior_faces(d, conf.lattice.planar)
    return edges, edge_l0, faces, _face_spoke_lengths(faces, bmap, bvec, None, set(), cell)


def relax_configuration_3d(conf, kappa_b: float = 0.02, max_iter: int = 400) -> ElasticResult:
    """Relax a defect `Configuration` in 3D under stretching + hinge bending."""
    if kappa_b < 0.0 or not math.isfinite(kappa_b):
        raise ValueError("kappa_b must be finite and non-negative")
    edges, edge_l0, faces, spoke_lengths = _defect_sheet_inputs(conf)
    rs = _rust.get()
    if rs is not None and hasattr(rs, "relax_defect_sheet_3d_json"):
        d = conf.defect
        payload = json.dumps(dict(
            positions=np.asarray(d.positions, float).tolist(),
            edges=edges,
            faces=faces,
            edge_l0=edge_l0.tolist(),
            spoke_lengths={str(fi): np.asarray(sp, float).tolist() for fi, sp in spoke_lengths.items()},
            kappa_b=float(kappa_b), max_iter=int(max_iter),
        ))
        res = _result_from_json(rs.relax_defect_sheet_3d_json(payload))
        conf.xyz = res["xyz"]
        return res

    from ..defects import fast as _fast
    d = conf.defect
    V, P2 = len(d.sites), np.asarray(d.positions, float)
    X0 = np.zeros((V, 3), float)
    if V:
        X0[:, :2] = P2
        dx, dy = P2[:, 0] - P2[:, 0].mean(), P2[:, 1] - P2[:, 1].mean()
        r, th = np.hypot(dx, dy), np.arctan2(dy, dx)
        rmax = max(float(r.max()), float(conf.lattice.lmean), 1.0)
        frank = _fast.total_frank(d) if (conf.lattice.planar and _fast.check_frames(d) == 0) else 0.0
        a0 = float(np.mean(edge_l0)) if len(edge_l0) else float(conf.lattice.lmean)
        if frank > 3.0:
            X0[:, 2] = 0.55 * min(max(math.sqrt(2.0 * min(max(frank / 360.0, 0.02), 0.45)), 0.25), 0.95) * (rmax - r) + 0.02 * a0 * np.cos(2.0 * th)
        elif frank < -3.0:
            X0[:, 2] = 0.5 * min(max(math.sqrt(min(max(-frank / 360.0, 0.02), 0.65)), 0.25), 0.75) * r * np.cos(2.0 * th) + 0.02 * a0 * np.sin(3.0 * th)
        else:
            env = np.exp(-(r * r) / max(12.0 * a0 * a0, 0.25 * rmax * rmax))
            X0[:, 2] = 0.35 * a0 * env * (1.0 + 0.35 * np.cos(th)) + 0.1 * a0 * (r / rmax) * np.cos(2.0 * th)
        X0[:, 2] += 1e-3 * a0 * np.sin(np.arange(V) * 1.61803398875)

    res = _run_elastic(V, X0, edges, edge_l0, list(range(len(edges))), faces,
                       spoke_lengths, kappa_b, max_iter, is_proj=False, is_open_sheet=True)
    conf.xyz = res["xyz"]
    return res
