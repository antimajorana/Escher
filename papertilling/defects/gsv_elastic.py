"""Layer 2: Gauge-invariant Grioli-Sander-Varga (GSV) discrete metric elasticity and continuum homogenization."""
from __future__ import annotations

import json
import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from .. import _rust
from ..legacy import fusion_framework as ff
from .gauge import _to_defect, _to_lat, ribbon_faces

_K3 = np.array([[2.0, -1.0, -1.0], [-1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]], float)


def _edge_vec(d, u: int, v: int) -> np.ndarray:
    e = d.edge[(u, v)]
    if "vec" in e:
        return np.asarray(e["vec"], float)
    from .circuit import _edge_vec_and_g
    return _edge_vec_and_g(d, u, v)[0]


def gsv_pairs(d, *, face_corners_only: bool = False) -> dict:
    """Extract G-gauge-invariant stretch and angle/shear pairs from crystal `d`."""
    d = _to_defect(d)
    V = len(d.sites)
    cone_sites = sorted(int(s) for s in d.info.get("cone_sites", ()))
    cone_set = set(cone_sites)

    edges = list(d.edges)
    su = np.array([u for u, _ in edges], dtype=int)
    sv = np.array([v for _, v in edges], dtype=int)
    sl0 = np.array([max(math.hypot(*_edge_vec(d, u, v)[:2]), 1e-12) for u, v in edges], dtype=float)

    au, av, aw, ac0, ak = [], [], [], [], []
    if not face_corners_only:
        nbr: dict[int, list[tuple[int, np.ndarray]]] = {u: [] for u in range(V)}
        for (u, v) in d.edge:
            nbr[u].append((v, _edge_vec(d, u, v)))
        for u in range(V):
            if u in cone_set:
                continue
            lst = sorted(nbr[u], key=lambda item: item[0])
            for i in range(len(lst)):
                v, dv = lst[i]
                lv = max(math.hypot(float(dv[0]), float(dv[1])), 1e-12)
                for j in range(i + 1, len(lst)):
                    w, dw = lst[j]
                    lw = max(math.hypot(float(dw[0]), float(dw[1])), 1e-12)
                    au.append(u); av.append(v); aw.append(w)
                    ac0.append(float(dv @ dw))
                    ak.append(1.0 / (lv * lw))

    if cone_set or face_corners_only:
        for f in ribbon_faces(d, include_boundary=False):
            verts = list(f)
            m = len(verts)
            if m < 3:
                continue
            for idx, u in enumerate(verts):
                if not face_corners_only and u not in cone_set:
                    continue
                f_rot = verts[idx:] + verts[:idx]
                vp, vm = f_rot[1], f_rot[-1]
                if vp == vm:
                    continue
                dp = _edge_vec(d, u, vp)
                if u in cone_set:
                    F = np.eye(2, dtype=float)
                    for step in range(m - 1):
                        g = d.edge[(f_rot[step], f_rot[step + 1])].get("g")
                        if g is not None:
                            F = F @ np.asarray(g[0], float).T
                    dm = -F @ _edge_vec(d, vm, u)
                else:
                    dm = _edge_vec(d, u, vm)
                lp = max(float(np.linalg.norm(dp)), 1e-12)
                lm = max(float(np.linalg.norm(dm)), 1e-12)
                au.append(u); av.append(vp); aw.append(vm)
                ac0.append(float(dp @ dm))
                ak.append(1.0 / (lp * lm))

    ak_arr = np.asarray(ak, dtype=float)
    return {
        "V": V, "E": len(edges), "su": su, "sv": sv, "sl0": sl0,
        "au": np.asarray(au, dtype=int), "av": np.asarray(av, dtype=int), "aw": np.asarray(aw, dtype=int),
        "ac0": np.asarray(ac0, dtype=float), "ak": ak_arr, "ask": np.sqrt(ak_arr), "cone_sites": cone_sites,
    }


def gsv_breakdown(positions, pairs_data: dict) -> dict:
    """Evaluate total, stretch, shear, per-bond, and per-site GSV energies and strains."""
    V = pairs_data["V"]
    P = np.asarray(positions, dtype=float).reshape(V, -1)
    su, sv, sl0 = pairs_data["su"], pairs_data["sv"], pairs_data["sl0"]
    au, av, aw, ac0, ask = pairs_data["au"], pairs_data["av"], pairs_data["aw"], pairs_data["ac0"], pairs_data["ask"]

    if len(su):
        dy = P[sv] - P[su]
        d2 = np.sum(dy * dy, axis=1)
        rs = (d2 - sl0 * sl0) / (2.0 * sl0)
        bond_e = 0.5 * rs * rs
        bond_strain = (np.sqrt(np.maximum(d2, 0.0)) - sl0) / sl0
    else:
        rs, bond_e, bond_strain = np.zeros(0), np.zeros(0), np.zeros(0)

    if len(au):
        a, b = P[av] - P[au], P[aw] - P[au]
        ra = 0.5 * ask * (np.sum(a * b, axis=1) - ac0)
        ang_e = 0.5 * ra * ra
    else:
        ra, ang_e = np.zeros(0), np.zeros(0)

    site_shear = np.zeros(V, dtype=float)
    if len(au):
        np.add.at(site_shear, au, ang_e)
    site_e = site_shear.copy()
    if len(su):
        np.add.at(site_e, su, 0.5 * bond_e)
        np.add.at(site_e, sv, 0.5 * bond_e)

    es, ea = float(np.sum(bond_e)), float(np.sum(ang_e))
    return {
        "total_energy": es + ea, "stretch_energy": es, "shear_energy": ea,
        "bond_strain": bond_strain, "bond_gsv_energy": bond_e,
        "site_gsv_energy": site_e, "site_shear": site_shear, "rs": rs, "ra": ra,
    }


def eval_gsv_energy(positions, pairs_data: dict) -> float:
    """Evaluate scalar gauge-invariant GSV energy E_GSV = 1/2 ||r(y)||^2 (2D or 3D positions)."""
    return float(gsv_breakdown(positions, pairs_data)["total_energy"])


def gsv_residual_and_jacobian(positions, pairs_data: dict):
    """Return `(r, J_csr, E_stretch, E_shear)` for 2D positions."""
    V = pairs_data["V"]
    P = np.asarray(positions, dtype=float).reshape(V, 2)
    su, sv, sl0 = pairs_data["su"], pairs_data["sv"], pairs_data["sl0"]
    au, av, aw, ac0, ask = pairs_data["au"], pairs_data["av"], pairs_data["aw"], pairs_data["ac0"], pairs_data["ask"]
    m_s, m_a = len(su), len(au)
    blocks = []

    if m_s:
        dy = P[sv] - P[su]
        rs = (np.sum(dy * dy, axis=1) - sl0 * sl0) / (2.0 * sl0)
        gx = dy / sl0[:, None]
        scols = np.column_stack([2 * sv, 2 * sv + 1, 2 * su, 2 * su + 1]).ravel()
        svals = np.column_stack([gx[:, 0], gx[:, 1], -gx[:, 0], -gx[:, 1]]).ravel()
        blocks.append(sp.coo_matrix((svals, (np.repeat(np.arange(m_s), 4), scols)), shape=(m_s, 2 * V)))
    else:
        rs = np.zeros(0, dtype=float)

    if m_a:
        a, b = P[av] - P[au], P[aw] - P[au]
        ra = 0.5 * ask * (np.sum(a * b, axis=1) - ac0)
        gv, gw = 0.5 * ask[:, None] * b, 0.5 * ask[:, None] * a
        gu = -(gv + gw)
        acols = np.column_stack([2 * au, 2 * au + 1, 2 * av, 2 * av + 1, 2 * aw, 2 * aw + 1]).ravel()
        avals = np.column_stack([gu[:, 0], gu[:, 1], gv[:, 0], gv[:, 1], gw[:, 0], gw[:, 1]]).ravel()
        blocks.append(sp.coo_matrix((avals, (np.repeat(np.arange(m_a), 6), acols)), shape=(m_a, 2 * V)))
    else:
        ra = np.zeros(0, dtype=float)

    J = sp.vstack(blocks).tocsr() if blocks else sp.csr_matrix((0, 2 * V), dtype=float)
    return np.concatenate([rs, ra]), J, 0.5 * float(rs @ rs), 0.5 * float(ra @ ra)


def gsv_hessian_2d(positions, pairs_data: dict, *, sparse: bool = False):
    """Compute exact analytical 2V x 2V 2D Hessian d^2 E_GSV / dy^2."""
    V = pairs_data["V"]
    P = np.asarray(positions, dtype=float).reshape(V, 2)
    su, sv, sl0 = pairs_data["su"], pairs_data["sv"], pairs_data["sl0"]
    au, av, aw, ac0, ask = pairs_data["au"], pairs_data["av"], pairs_data["aw"], pairs_data["ac0"], pairs_data["ask"]
    I2 = np.eye(2, dtype=float)
    rows, cols, vals = [], [], []

    if len(su):
        dy = P[sv] - P[su]
        rs = (np.sum(dy * dy, axis=1) - sl0 * sl0) / (2.0 * sl0)
        gx = dy / sl0[:, None]
        H_loc = gx[:, :, None] * gx[:, None, :] + (rs / sl0)[:, None, None] * I2[None, :, :]
        block = np.empty((len(su), 2, 2, 2, 2), dtype=float)
        block[:, 0, :, 0, :] = H_loc
        block[:, 1, :, 1, :] = H_loc
        block[:, 0, :, 1, :] = -H_loc
        block[:, 1, :, 0, :] = -H_loc
        dofs = np.column_stack([2 * su, 2 * su + 1, 2 * sv, 2 * sv + 1])
        rr = np.broadcast_to(dofs[:, :, None], (len(su), 4, 4)).ravel()
        cc = np.broadcast_to(dofs[:, None, :], (len(su), 4, 4)).ravel()
        rows.append(rr); cols.append(cc); vals.append(block.reshape(len(su), 4, 4).ravel())

    if len(au):
        a, b = P[av] - P[au], P[aw] - P[au]
        ra = 0.5 * ask * (np.sum(a * b, axis=1) - ac0)
        gv, gw = 0.5 * ask[:, None] * b, 0.5 * ask[:, None] * a
        gu = -(gv + gw)
        G = np.stack([gu, gv, gw], axis=1)
        ch = 0.5 * ask * ra
        blk = np.einsum("mia,mjb->miajb", G, G) + ch[:, None, None, None, None] * (
            _K3[None, :, None, :, None] * I2[None, None, :, None, :]
        )
        dofs = np.column_stack([2 * au, 2 * au + 1, 2 * av, 2 * av + 1, 2 * aw, 2 * aw + 1])
        rr = np.broadcast_to(dofs[:, :, None], (len(au), 6, 6)).ravel()
        cc = np.broadcast_to(dofs[:, None, :], (len(au), 6, 6)).ravel()
        rows.append(rr); cols.append(cc); vals.append(blk.reshape(len(au), 6, 6).ravel())

    if not rows:
        H_csr = sp.csr_matrix((2 * V, 2 * V), dtype=float)
    else:
        H_csr = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(2 * V, 2 * V)).tocsr()
    return H_csr if sparse else H_csr.toarray()


def _initial_planar_guess(d) -> np.ndarray:
    X0 = np.asarray(d.positions, float)
    if not d.edge or d.info.get("mobius") or d.info.get("klein"):
        return X0.copy()
    from .planarity import check, has_frames
    if has_frames(d):
        c = check(d, X0, plain_only=False)
        if c["crossings"] > 0 or (c["folded"] or 0) > 0:
            from .draw import embed
            return embed(d, relax_given=True, overlaps=False)[0]
    return X0.copy()


def _fix_e2_gauge(P: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    P_c = P - P.mean(axis=0)
    X_c = X_ref - X_ref.mean(axis=0)
    M = P_c.T @ X_c
    if float(np.linalg.norm(M)) < 1e-12:
        return P_c
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if float(np.linalg.det(R)) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return P_c @ R


def _safe_splu(mat):
    csc = mat.tocsc(copy=True)
    csc.sum_duplicates()
    csc.sort_indices()
    if csc.indices.dtype != np.int32:
        csc.indices = csc.indices.astype(np.int32)
    if csc.indptr.dtype != np.int32:
        csc.indptr = csc.indptr.astype(np.int32)
    return splu(csc, permc_spec="MMD_AT_PLUS_A")


def _hessian_min_eigs(P: np.ndarray, pairs: dict, k: int = 8) -> np.ndarray:
    V = pairs["V"]
    if V == 0:
        return np.zeros(0, dtype=float)
    n_dof = 2 * V
    if n_dof <= 600:
        return np.sort(np.linalg.eigvalsh(gsv_hessian_2d(P, pairs, sparse=False)))[:min(k, n_dof)]
    H_sp = gsv_hessian_2d(P, pairs, sparse=True).tocsc()
    kk = min(k, n_dof - 2)
    if kk <= 0:
        return np.sort(np.linalg.eigvalsh(H_sp.toarray()))
    try:
        lu_shift = _safe_splu(H_sp + 1e-4 * sp.identity(n_dof, format="csc"))
        op_inv = LinearOperator((n_dof, n_dof), matvec=lu_shift.solve, dtype=float)
        vals = eigsh(H_sp, k=kk, sigma=-1e-4, OPinv=op_inv, which="LM", return_eigenvectors=False)
    except Exception:
        vals = eigsh(H_sp, k=kk, which="SM", return_eigenvectors=False)
    return np.sort(np.real(vals))


def _defect_to_built_crystal_json(d_raw) -> str:
    from .circuit import _edge_vec_and_g

    pos = np.asarray(d_raw.positions, float)
    v_cnt = len(pos)
    ref = np.asarray(d_raw.info.get("reference", pos), float)
    non_ori = bool(d_raw.info.get("mobius") or d_raw.info.get("klein") or d_raw.info.get("_orientable") is False)
    basis_idx, side, adj = [], [], [[] for _ in range(v_cnt)]
    for s in d_raw.sites:
        key = s[0] if isinstance(s, (tuple, list)) and len(s) >= 2 else s
        b_i = int(key[2]) if isinstance(key, (tuple, list)) and len(key) >= 3 else 0
        basis_idx.append(b_i)
        s_tag = s[1] if (not non_ori and isinstance(s, (tuple, list)) and len(s) >= 2) else None
        side.append("A" if s_tag in ("A", 0) else ("B" if s_tag in ("B", 1) else None))
    for (u, v), e in sorted(d_raw.edge.items()):
        vec, g = _edge_vec_and_g(d_raw, int(u), int(v))
        seam = {"m": [[float(g[0][0, 0]), float(g[0][0, 1])], [float(g[0][1, 0]), float(g[0][1, 1])]],
                "t": [float(g[1][0]), float(g[1][1])]} if g is not None else None
        ghost = e.get("ghost")
        adj[int(u)].append({
            "to": int(v),
            "vec": [float(vec[0]), float(vec[1])],
            "kind": int(e.get("kind", e.get("type", 0)) or 0),
            "seam": seam,
            "ang": float(e["ang"]) if e.get("ang") is not None else None,
            "across": bool(e.get("across", False)),
            "ghost_stub": [float(ghost[0]), float(ghost[1])] if ghost is not None else None,
        })
    payload = {
        "pos": pos.tolist(),
        "ref_pos": ref.tolist(),
        "basis_idx": basis_idx,
        "side": side,
        "cone_sites": sorted(int(s) for s in d_raw.info.get("cone_sites", ())),
        "edges": [[int(u), int(v)] for u, v in d_raw.edges],
        "adj": adj,
        "wall_sigma": None,
        "is_non_orientable": non_ori,
    }
    return json.dumps(payload)


def _try_rust_relax_2d(d_raw, X_init: np.ndarray):
    rs = _rust.get()
    if rs is None or not hasattr(rs, "relax_elastic_2d_patch_json"):
        return None
    try:
        res = json.loads(rs.relax_elastic_2d_patch_json(_defect_to_built_crystal_json(d_raw)))
        return _fix_e2_gauge(np.asarray(res["pos"], float), X_init), int(res.get("iterations", 0))
    except Exception:
        return None


def relax_elastic_2d(d, *, lattice=None, max_iter: int = 150, gtol: float = 1e-12):
    """Minimize zero-anchor GSV energy E_GSV(y) using sparse LM Gauss-Newton + exact full-Newton polish."""
    _ = lattice
    d_raw = _to_defect(d)
    pairs = gsv_pairs(d_raw)
    V = pairs["V"]
    if V == 0:
        return d
    X_init = _initial_planar_guess(d_raw)
    if (rs_out := _try_rust_relax_2d(d_raw, X_init)) is not None:
        P_fin, it_used = rs_out
    else:
        y = X_init.ravel().copy()
        I_sp = sp.identity(2 * V, format="csc")
        lam = 1e-6
        r, J, es, ea = gsv_residual_and_jacobian(y, pairs)
        E, it_used = es + ea, 0

        for it in range(1, max_iter + 1):
            it_used = it
            g = J.T @ r
            gnorm = float(np.max(np.abs(g))) if len(g) else 0.0
            if gnorm <= gtol:
                break
            if gnorm < 1e-2 and lam <= 1e-10:
                H = gsv_hessian_2d(y, pairs, sparse=True).tocsc()
                mu, stepped = 1e-12, False
                for _ in range(8):
                    try:
                        dy = _safe_splu(H + mu * I_sp).solve(-g)
                    except Exception:
                        dy = None
                    if dy is not None and np.all(np.isfinite(dy)) and float(g @ dy) < 0.0:
                        alpha = 1.0
                        for _ in range(12):
                            y2 = y + alpha * dy
                            r2, J2, es2, ea2 = gsv_residual_and_jacobian(y2, pairs)
                            E2 = es2 + ea2
                            if E2 <= E:
                                y, r, J, E, stepped = y2, r2, J2, E2, True
                                break
                            alpha *= 0.5
                        if stepped:
                            break
                    mu *= 100.0
                if stepped:
                    continue

            JTJ = (J.T @ J).tocsc()
            accepted = False
            for _ in range(20):
                try:
                    dy = _safe_splu(JTJ + lam * I_sp).solve(-g)
                except Exception:
                    dy = np.full_like(g, np.nan)
                if np.all(np.isfinite(dy)):
                    y2 = y + dy
                    r2, J2, es2, ea2 = gsv_residual_and_jacobian(y2, pairs)
                    E2 = es2 + ea2
                    if E2 < E or (E2 <= E + 1e-15 and float(np.max(np.abs(J2.T @ r2))) < gnorm):
                        y, r, J, E, accepted = y2, r2, J2, E2, True
                        lam = max(lam * 0.1, 1e-13)
                        break
                lam *= 10.0
                if lam > 1e12:
                    break
            if not accepted:
                break

        P_fin = _fix_e2_gauge(y.reshape(V, 2), X_init)

    r_fin, J_fin, _, _ = gsv_residual_and_jacobian(P_fin, pairs)
    grad_inf = float(np.max(np.abs(J_fin.T @ r_fin))) if V else 0.0
    bd = gsv_breakdown(P_fin, pairs)

    d_raw.positions = P_fin
    d_raw._nbr = None
    ff.mark_long_bonds(d_raw)
    d_raw.info["picture"] = "elastic"
    d_raw.info["gsv"] = {
        "total_energy": bd["total_energy"], "stretch_energy": bd["stretch_energy"], "shear_energy": bd["shear_energy"],
        "bond_strain": bd["bond_strain"], "bond_gsv_energy": bd["bond_gsv_energy"],
        "site_gsv_energy": bd["site_gsv_energy"], "site_shear": bd["site_shear"],
        "hessian_min_eigs": _hessian_min_eigs(P_fin, pairs), "pairs": pairs,
        "grad_inf": grad_inf, "n_iter": it_used,
    }
    return d


def cauchy_born_tensor(lattice) -> np.ndarray:
    """Compute exact 2x2x2x2 continuum stiffness tensor C_{iajb} with Born-Huang sublattice relaxation."""
    lat = _to_lat(lattice)
    cell, A = lat.cell, lat.A
    area, Ns = abs(float(np.linalg.det(A))), len(cell.basis)
    bonds_cell, seen_b = [], set()
    nbrs: dict[int, list[tuple[int, tuple[int, int], np.ndarray]]] = {s: [] for s in range(Ns)}
    for (i, j, n1, n2, _) in cell.bonds:
        canon = min(((0, 0, i), (n1, n2, j)), ((0, 0, j), (-n1, -n2, i)))
        dv = np.asarray(cell.basis[j], float) + A @ np.array([n1, n2], float) - np.asarray(cell.basis[i], float)
        for s_from, s_to, sh, vec in ((i, j, (n1, n2), dv), (j, i, (-n1, -n2), -dv)):
            if not any(jj == s_to and ss == sh for jj, ss, _ in nbrs[s_from]):
                nbrs[s_from].append((s_to, sh, vec))
        if canon not in seen_b:
            seen_b.add(canon)
            bonds_cell.append((i, j, dv))

    n_eta = 2 * max(Ns - 1, 0)
    rows_G, rows_eta = [], []
    for i, j, dv in bonds_cell:
        uhat = dv / max(float(np.linalg.norm(dv)), 1e-12)
        rows_G.append([uhat[0] * dv[0], uhat[0] * dv[1], uhat[1] * dv[0], uhat[1] * dv[1]])
        me = np.zeros(n_eta, float)
        if j > 0:
            me[2 * (j - 1):2 * j] += uhat
        if i > 0:
            me[2 * (i - 1):2 * i] -= uhat
        rows_eta.append(me)

    for s in range(Ns):
        lst = nbrs[s]
        for ia in range(len(lst)):
            ja, _, da = lst[ia]
            la = max(float(np.linalg.norm(da)), 1e-12)
            for ib in range(ia + 1, len(lst)):
                jb, _, db = lst[ib]
                lb = max(float(np.linalg.norm(db)), 1e-12)
                c = 0.5 / math.sqrt(la * lb)
                rows_G.append([c * (da[0] * db[0] + db[0] * da[0]), c * (da[0] * db[1] + db[0] * da[1]),
                               c * (da[1] * db[0] + db[1] * da[0]), c * (da[1] * db[1] + db[1] * da[1])])
                me = np.zeros(n_eta, float)
                if jb > 0:
                    me[2 * (jb - 1):2 * jb] += c * da
                if ja > 0:
                    me[2 * (ja - 1):2 * ja] += c * db
                if s > 0:
                    me[2 * (s - 1):2 * s] -= c * (da + db)
                rows_eta.append(me)

    JG = np.asarray(rows_G, float)
    if n_eta > 0:
        Jeta = np.asarray(rows_eta, float)
        eta_G = -np.linalg.lstsq(Jeta.T @ Jeta, Jeta.T @ JG, rcond=None)[0]
        Jeff = JG + Jeta @ eta_G
    else:
        Jeff = JG
    K4 = (Jeff.T @ Jeff) / area
    idx = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    C = np.zeros((2, 2, 2, 2), dtype=float)
    for i in range(2):
        for a in range(2):
            for j in range(2):
                for b in range(2):
                    C[i, a, j, b] = K4[idx[(i, a)], idx[(j, b)]]
    return C


def _as_C(lattice_or_C) -> np.ndarray:
    if isinstance(lattice_or_C, np.ndarray) and lattice_or_C.shape == (2, 2, 2, 2):
        return lattice_or_C
    return cauchy_born_tensor(lattice_or_C)


def acoustic_tensor(lattice_or_C, theta: float) -> np.ndarray:
    """Return 2x2 symmetric acoustic modulus tensor Gamma_ij(theta) = C_{iajb} n_a n_b."""
    C = _as_C(lattice_or_C)
    n = np.array([math.cos(theta), math.sin(theta)], float)
    return np.einsum("iajb,a,b->ij", C, n, n)


def barnett_lothe_tensor(lattice_or_C, *, n_quad: int = 128) -> np.ndarray:
    """Compute 2x2 Barnett-Lothe prelogarithmic energy tensor B from angular quadrature of Gamma(theta)."""
    C = _as_C(lattice_or_C)
    B = np.zeros((2, 2), dtype=float)
    for th in np.linspace(0.0, 2.0 * math.pi, n_quad, endpoint=False):
        m = np.array([math.cos(th), math.sin(th)], float)
        n = np.array([-math.sin(th), math.cos(th)], float)
        mm = np.einsum("iajb,a,b->ij", C, m, m)
        nn = np.einsum("iajb,a,b->ij", C, n, n)
        mn = np.einsum("iajb,a,b->ij", C, m, n)
        B += mm - mn @ np.linalg.solve(nn, mn.T)
    return B / float(n_quad)


def dipole_prelog_coeff(lattice_or_C, b, *, n_quad: int = 128) -> float:
    """Return exact Stroh/Barnett-Lothe dipole prelogarithmic slope K_dipole = (b^T B b) / (2 pi)."""
    B = barnett_lothe_tensor(lattice_or_C, n_quad=n_quad)
    bv = np.asarray(b, float)
    return float(bv @ B @ bv) / (2.0 * math.pi)


def plot_elastic(config_or_defect, ax=None, title: str | None = None, mode: str = "strain"):
    """Plot elastic strain, shear, or GSV energy heatmap on Configuration or Defect."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    d = _to_defect(config_or_defect)
    if "gsv" not in d.info:
        relax_elastic_2d(d, lattice=getattr(config_or_defect, "lattice", None))
    gsv, P = d.info["gsv"], d.positions
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    segs = [[P[u], P[v]] for u, v in d.edges]
    mode_l = str(mode).lower()
    if mode_l in ("strain", "bond", "energy"):
        vals = gsv["bond_gsv_energy"] if mode_l == "energy" else np.abs(gsv["bond_strain"])
        lc = LineCollection(segs, array=vals, cmap="magma", linewidths=1.5)
        ax.add_collection(lc)
        ax.scatter(P[:, 0], P[:, 1], s=8, c="k", zorder=3)
        plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04, label="bond energy" if mode_l == "energy" else "|bond strain|")
    else:
        ax.add_collection(LineCollection(segs, colors="#888888", linewidths=1.0))
        svals = gsv["site_shear"] if mode_l == "shear" else gsv["site_gsv_energy"]
        sc = ax.scatter(P[:, 0], P[:, 1], c=svals, cmap="viridis", s=26, zorder=4)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="site shear" if mode_l == "shear" else "site GSV energy")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or f"GSV elastic (E={gsv['total_energy']:.4g})", fontsize=9)
    return ax
