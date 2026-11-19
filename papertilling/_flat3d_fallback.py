"""Pure-Python fallbacks for `papertilling.flat3d` (used when `papertilling_rs` is absent).

Krivy-Gruber Niggli reduction, lattice holohedry `[PROVED]` obstructions, platycosm quotients and
z-preserving in-plane glides. Mirrors `rust/papertilling-core/src/flat3d/` (`niggli_supercell.rs`,
`lattice_symmetry.rs`, `platycosms.rs`, `platycosm_data.rs`, `slabs3d_nonorient.rs`).
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from .bieberbach import _hnf_compress_rows
from .space3d import _raw_space_230

# Krivy & Gruber, Acta Cryst. A32 (1976) 297; eps-tolerant tests of Grosse-Kunstleve, Sauter &
# Adams, Acta Cryst. A60 (2004) 1. Same steps as `niggli_reduce_3d` in `niggli_supercell.rs`.
def py_niggli(lattice: Any) -> dict[str, Any]:
    v, p = np.asarray(lattice, float).copy(), np.eye(3, dtype=int)
    vol0 = abs(float(np.linalg.det(v))) if np.all(np.isfinite(v)) else 0.0
    if vol0 < 1e-12 * max(1e-12, float(np.max(np.sum(v * v, axis=1))) ** 1.5):
        raise ValueError("degenerate, singular or non-finite 3D lattice basis")
    eps = 1e-6 * vol0 ** (2.0 / 3.0)
    sgn = lambda x: 1 if x > eps else (-1 if x < -eps else 0)
    met = lambda: (float(v[0] @ v[0]), float(v[1] @ v[1]), float(v[2] @ v[2]), 2.0 * float(v[1] @ v[2]), 2.0 * float(v[0] @ v[2]), 2.0 * float(v[0] @ v[1]))
    it, converged = 0, False
    for it in range(10000):
        a, b, c, xi, eta, zeta = met()
        if a > b + eps or (abs(a - b) <= eps and abs(xi) > abs(eta) + eps):
            v[[0, 1]], p[[0, 1]] = v[[1, 0]], p[[1, 0]]
            continue
        if b > c + eps or (abs(b - c) <= eps and abs(eta) > abs(zeta) + eps):
            v[[1, 2]], p[[1, 2]] = v[[2, 1]], p[[2, 1]]
            continue
        lmn = [sgn(xi), sgn(eta), sgn(zeta)]
        if lmn[0] * lmn[1] * lmn[2] == 1:
            signs = lmn  # A3
        else:  # A4: flip vectors with positive products; fix handedness on the last zero product
            signs = [-1 if s > 0 else 1 for s in lmn]
            zeros = [k for k, s in enumerate(lmn) if s == 0]
            if signs[0] * signs[1] * signs[2] < 0 and zeros:
                signs[zeros[-1]] = -1
        for i in range(3):
            if signs[i] < 0:
                v[i], p[i] = -v[i], -p[i]
        a, b, c, xi, eta, zeta = met()
        for val, ref, k1, k2, dst, src in ((xi, b, 2 * eta - zeta, zeta, 2, 1), (eta, a, 2 * xi - zeta, zeta, 2, 0), (zeta, a, 2 * xi - eta, eta, 1, 0)):
            if abs(val) > ref + eps or (abs(val - ref) <= eps and k1 < -eps) or (abs(val + ref) <= eps and k2 < -eps):
                s = 1 if val > 0 else -1
                v[dst] -= s * v[src]
                p[dst] -= s * p[src]
                break
        else:
            s_all = xi + eta + zeta + a + b
            if s_all < -eps or (abs(s_all) <= eps and 2 * (a + eta) + zeta > eps):
                v[2] += v[0] + v[1]
                p[2] += p[0] + p[1]
                continue
            converged = True
            break
    if not converged:
        raise ValueError("Niggli reduction did not converge in 10000 steps")
    if round(float(np.linalg.det(p))) == -1:
        v, p = -v, -p
    _, _, _, xi, eta, zeta = met()
    lens = [float(np.linalg.norm(v[i])) for i in range(3)]
    ang = lambda i, j: math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v[i], v[j]) / (lens[i] * lens[j]))))))
    return {"reduced": v.tolist(), "p_mat": p.tolist(), "lengths": lens, "angles_deg": [ang(1, 2), ang(0, 2), ang(0, 1)], "niggli_type": 1 if min(xi, eta, zeta) > eps else 2, "volume": abs(float(np.linalg.det(v))), "iterations": it}


def _darts(bonds: list[Any]) -> set[tuple[int, int, tuple[int, int, int]]]:
    out = set()
    for bd in bonds:
        i, j, n = int(bd[0]), int(bd[1]), tuple(int(x) for x in bd[2:5])
        out.add((i, j, n))
        out.add((j, i, tuple(-x for x in n)))
    return out


def _site_map(fb: np.ndarray, kinds: list[int], image) -> tuple[list[int], list[tuple[int, ...]]] | None:
    perm, shift = [], []
    for s in range(len(fb)):
        img = image(s)
        hit = next(((s2, tuple(int(x) for x in np.round(img - fb[s2]))) for s2 in range(len(fb)) if kinds[s2] == kinds[s] and np.all(np.abs(img - fb[s2] - np.round(img - fb[s2])) < 1e-4)), None)
        if hit is None:
            return None
        perm.append(hit[0])
        shift.append(hit[1])
    return perm, shift


def full_translation_lattice(cell: dict[str, Any], fb: np.ndarray, ignore_kinds: bool) -> np.ndarray | None:
    """Row basis of the crystal's full translation lattice (cell lattice refined by pure translations)."""
    lat = np.asarray(cell["lattice"], float)
    kinds = [0] * len(fb) if ignore_kinds else [int(k) for k in (cell.get("site_kinds") or [0] * len(fb))]
    darts, ts = _darts(cell["bonds"]), []
    near = lambda x: bool(np.all(np.abs(x - np.round(x)) < 1e-4))
    for j in range(1, len(fb)):
        t = (fb[j] - fb[0]) % 1.0
        if kinds[j] != kinds[0] or near(t) or any(near(o - t) for o in ts):
            continue
        sm = _site_map(fb, kinds, lambda s: fb[s] + t)
        if sm is not None and all((sm[0][int(bd[0])], sm[0][int(bd[1])], tuple(int(bd[2 + k]) + sm[1][int(bd[1])][k] - sm[1][int(bd[0])][k] for k in range(3))) in darts for bd in cell["bonds"]):
            ts.append(t)
    if not ts:
        return lat
    den = next((d for d in range(2, 49) if all(np.all(np.abs(t * d - np.round(t * d)) < 1e-3) for t in ts)), None)
    if den is None:
        return None
    h = _hnf_compress_rows([[den, 0, 0], [0, den, 0], [0, 0, den]] + [[int(round(x * den)) for x in t] for t in ts], 3)
    h = [r for r in h if any(r)]
    return (np.asarray(h, float) / den) @ lat if len(h) == 3 else None


def lattice_holohedry(lat: Any) -> dict[str, Any] | None:
    """Holohedry of a 3D lattice: order, proper rotation orders, and whether it contains 222."""
    b = np.asarray(py_niggli(lat)["reduced"], float)
    g = b @ b.T
    gmax = float(np.max(np.diag(g)))
    tol, binv = 1e-5 * gmax, np.linalg.inv(b)
    bound = [int(math.floor(math.sqrt(gmax) * float(np.linalg.norm(binv[:, i])) + 1e-6)) for i in range(3)]
    if max(bound) > 6:
        return None
    vecs = np.array(list(itertools.product(*[range(-k, k + 1) for k in bound])), dtype=int)
    n2 = np.einsum("ij,jk,ik->i", vecs, g, vecs)
    cands = [vecs[np.abs(n2 - g[j, j]) <= tol] for j in range(3)]
    proper, order = [], 0
    for m0 in cands[0]:
        for m1 in cands[1]:
            if abs(m0 @ g @ m1 - g[0, 1]) > tol:
                continue
            for m2 in cands[2]:
                if abs(m0 @ g @ m2 - g[0, 2]) > tol or abs(m1 @ g @ m2 - g[1, 2]) > tol:
                    continue
                m = np.array([m0, m1, m2])
                d = round(float(np.linalg.det(m)))
                if d not in (1, -1):
                    return None
                order += 1
                if d == 1:
                    proper.append(m)
    if order == 0 or order > 48:
        return None
    trace_to_order = {3: 1, -1: 2, 0: 3, 1: 4, 2: 6}  # trace = 1 + 2 cos(2 pi / n)
    orders, twofolds = set(), []
    for m in proper:
        o = trace_to_order.get(int(np.trace(m)))
        if o is None:
            return None
        orders.add(o)
        if o == 2:
            twofolds.append(m)
    has_222 = any(np.array_equal(x @ y, y @ x) for i, x in enumerate(twofolds) for y in twofolds[i + 1:])
    return {"order": order, "proper_orders": sorted(orders), "has_222": has_222}


def holohedry_obstruction(code: str, h: dict[str, Any]) -> str | None:
    need = {"c2": (2, "a 2-fold axis (point group 2, or m = -1 * 2)"), "+a1": (2, "a 2-fold axis (point group 2, or m = -1 * 2)"), "-a1": (2, "a 2-fold axis (point group 2, or m = -1 * 2)"),
            "c3": (3, "a 3-fold axis (point group 3)"), "c4": (4, "a 4-fold axis (point group 4)"), "c6": (6, "a 6-fold axis (point group 6)"),
            "c22": (222, "two perpendicular 2-fold axes (point group 222, or mm2 = -1 * 222)"), "+a2": (222, "two perpendicular 2-fold axes (point group 222, or mm2 = -1 * 222)"), "-a2": (222, "two perpendicular 2-fold axes (point group 222, or mm2 = -1 * 222)")}
    if code not in need:
        return None
    req, text = need[code]
    if (h["has_222"] if req == 222 else req in h["proper_orders"]):
        return None
    orders = "{" + ", ".join(str(o) for o in h["proper_orders"]) + "}"
    return f"IMPOSSIBLE [PROVED]: {code} needs {text}, but the holohedry of the crystal's full translation lattice (order {h['order']}, proper rotation orders {orders}) has none; the point group of any symmetry subgroup lies in that holohedry"


def find_inplane_glide(cell: dict[str, Any], fb: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[int, ...]]] | None:
    """Search M (entries in [-2, 2], a3 fixed, M^2 = I, det M = -1) and t for a z-preserving glide."""
    lat = np.asarray(cell["lattice"], float)
    g = lat @ lat.T
    tol = 1e-5 * max(1e-12, float(np.max(np.diag(g))))
    kinds = [int(k) for k in (cell.get("site_kinds") or [0] * len(fb))]
    darts, cands = _darts(cell["bonds"]), []
    for a, b, c, d in itertools.product(range(-2, 3), repeat=4):
        r = np.array([[a, b, 0], [c, d, 0], [0, 0, 1]])
        if a * d - b * c == -1 and a * a + b * c == 1 and b * (a + d) == 0 and c * (a + d) == 0 and b * c + d * d == 1 and np.allclose(r.T @ g @ r, g, atol=tol, rtol=0.0):
            cands.append(np.array([[a, b], [c, d]]))
    cands.sort(key=lambda m: 0 if np.array_equal(m, [[1, 0], [0, -1]]) else (1 if np.array_equal(m, [[-1, 0], [0, 1]]) else 2))
    for m2 in cands:
        for b0 in range(len(fb)):
            if kinds[b0] != kinds[0] or abs(fb[b0, 2] - fb[0, 2]) > 1e-4:
                continue
            t = (fb[b0, :2] - m2 @ fb[0, :2]) % 1.0
            sm = _site_map(fb, kinds, lambda s: np.concatenate([m2 @ fb[s, :2] + t, fb[s, 2:3]]))
            if sm is None:
                continue
            perm, shift = sm
            if all((perm[int(bd[0])], perm[int(bd[1])], (int(m2[0, 0] * int(bd[2]) + m2[0, 1] * int(bd[3])) + shift[int(bd[1])][0] - shift[int(bd[0])][0], int(m2[1, 0] * int(bd[2]) + m2[1, 1] * int(bd[3])) + shift[int(bd[1])][1] - shift[int(bd[0])][1], int(bd[4]) + shift[int(bd[1])][2] - shift[int(bd[0])][2])) in darts for bd in cell["bonds"]):
                return m2, t, perm, shift
    return None


def _ops_for_ita_py(ita: int, ax: int = 0):
    if ita == 4 and ax > 0:
        r2, tn = (np.diag([-1, -1, 1]), (0, 0, 1)) if ax == 1 else (np.diag([1, -1, -1]), (1, 0, 0))
        return [(np.eye(3, dtype=int), np.zeros(3), (0, 0, 0), 1), (r2, np.array(tn) * 0.5, tn, 2)]
    e = next(g for g in _raw_space_230() if g["number"] == ita)
    out = [(np.asarray(op["r"], int), np.array([n / d for n, d in op["t"]], float), tuple(int(n) * (math.lcm(*[int(d) for _, d in op["t"]]) // int(d)) for n, d in op["t"]), math.lcm(*[int(d) for _, d in op["t"]])) for op in e["ops"]]
    if e["centring"] == "C":
        for r, t, tn, den in list(out):
            l2 = math.lcm(den, 2); s, h = l2 // den, l2 // 2
            out.append((r, (t + [0.5, 0.5, 0.0]) % 1.0, ((tn[0] * s + h) % l2, (tn[1] * s + h) % l2, (tn[2] * s) % l2), l2))
    return out


def _platy_cw_chi(code: str, ops: list[Any], l_sup: tuple[int, int, int]) -> tuple[int, int, int, int, int]:
    hex_m, mult = code in ("c3", "c6"), 6 if code in ("c3", "c6") else 2
    d = (6 if code == "c6" else 3) if hex_m else (1 if code == "c1" else (4 if code == "c4" else 2))
    k1, k2, k3 = (max(1, l_sup[0]), max(1, l_sup[1]), ((max(l_sup[2], d) + d - 1) // d) * d) if hex_m else [((max(x, d) + d - 1) // d) * d for x in l_sup]
    m1, m2, m3 = k1 * mult, k2 * mult, k3 * mult
    dims: list[set[tuple[int, int, int]]] = [set(), set(), set(), set()]
    for bx, by, bz in ((i * mult, j * mult, k * mult) for i, j, k in np.ndindex(k1, k2, k3)):
        if hex_m:
            dims[0].add((bx, by, bz))
            dims[1].update([(bx + 3, by, bz), (bx, by + 3, bz), (bx + 3, by + 3, bz), (bx, by, bz + 3)])
            dims[2].update([(bx + 4, by + 2, bz), (bx + 2, by + 4, bz), (bx + 3, by, bz + 3), (bx, by + 3, bz + 3), (bx + 3, by + 3, bz + 3)])
            dims[3].update([(bx + 4, by + 2, bz + 3), (bx + 2, by + 4, bz + 3)])
        else:
            for dx, dy, dz in np.ndindex(2, 2, 2):
                dims[dx + dy + dz].add((bx + dx, by + dy, bz + dz))
    cnts = [len({min({tuple((int(r[q, 0] * c[0] + r[q, 1] * c[1] + r[q, 2] * c[2]) + (tn[q] * [m1, m2, m3][q]) // den) % [m1, m2, m3][q] for q in range(3)) for r, _, tn, den in ops}) for c in cells}) for cells in dims]
    return cnts[0], cnts[1], cnts[2], cnts[3], cnts[0] - cnts[1] + cnts[2] - cnts[3]


def py_platycosm(cell: dict[str, Any], info: dict[str, Any], supercell: tuple[int, int, int] | None, ignore_site_kinds: bool) -> dict[str, Any]:
    """Quotient-shaped result: status OK, IMPOSSIBLE ([PROVED] holohedry) or NOT_FOUND ([COMPUTATIONAL EVIDENCE])."""
    lat = np.asarray(cell["lattice"], float)
    fb = np.linalg.solve(lat.T, np.asarray(cell["basis"], float).T).T
    kinds = cell.get("site_kinds") or [0] * len(fb)
    full = full_translation_lattice(cell, fb, ignore_site_kinds)
    holo = lattice_holohedry(full) if full is not None else None
    obs = holohedry_obstruction(info["code"], holo) if holo is not None else None
    if obs is not None:
        return _refusal(info, "IMPOSSIBLE", obs, supercell)
    cands = [tuple(int(x) for x in supercell)] if supercell is not None else [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, 6), (2, 2, 1), (2, 2, 2), (2, 2, 4), (2, 2, 6)]
    shifts = [[sx, sy, sz] for sz in (0.0, 0.5, 0.25, 0.75) for sx in (0.0, 0.5, 0.25, 0.75) for sy in (0.0, 0.5, 0.25, 0.75)]
    any_m = False
    for l_sup in cands:
        slat = lat * np.asarray(l_sup, float)[:, None]
        g = slat @ slat.T
        tol = 1e-5 * max(1e-12, float(np.max(np.diag(g))))
        l1, l2, l3 = l_sup
        n_sites = len(fb) * l1 * l2 * l3
        sid = lambda b, x, y, z: ((b * l1 + x) * l2 + y) * l3 + z
        xc = np.array([[(ix + fb[b][0]) / l1, (iy + fb[b][1]) / l2, (iz + fb[b][2]) / l3] for b in range(len(fb)) for ix in range(l1) for iy in range(l2) for iz in range(l3)])
        all_b = [(sid(int(bd[0]), ix, iy, iz), sid(int(bd[1]), (ix + int(bd[2])) % l1, (iy + int(bd[3])) % l2, (iz + int(bd[4])) % l3), ((ix + int(bd[2])) // l1, (iy + int(bd[3])) // l2, (iz + int(bd[4])) // l3), int(bd[5]) if len(bd) > 5 else 0)
                 for ix in range(l1) for iy in range(l2) for iz in range(l3) for bd in cell["bonds"]]
        darts = {(u, v, sh) for u, v, sh, _ in all_b} | {(v, u, (-sh[0], -sh[1], -sh[2])) for u, v, sh, _ in all_b}
        for ita, sym in zip(info["ita_numbers"], info["ita_symbols"]):
            for ax in ((0, 1, 2) if info["code"] == "c2" else (0,)):
                ops = _ops_for_ita_py(ita, ax)
                if not all(np.allclose(r.T @ g @ r, g, atol=tol, rtol=0.0) for r, _, _, _ in ops):
                    continue
                any_m = True
                for sh_v in shifts:
                    sh_a, perms, csh, ok = np.asarray(sh_v, float), [], [], True
                    for r, t, _, _ in ops:
                        teff = t + sh_a - r @ sh_a
                        is_id = np.array_equal(r, np.eye(3, dtype=int)) and np.allclose(teff, np.round(teff), atol=1e-5)
                        p_op, cs_op = [0] * n_sites, [(0, 0, 0)] * n_sites
                        for s_i, xp in enumerate(xc):
                            k_src, trg, hit = kinds[s_i // (l1 * l2 * l3)], r @ xp + teff, None
                            for b2 in range(len(fb)):
                                if ignore_site_kinds or kinds[b2] == k_src:
                                    u = trg * [l1, l2, l3] - fb[b2]; ru = np.round(u)
                                    if np.all(np.abs(u - ru) < 1e-4):
                                        iv = [int(x) for x in ru]; tid = sid(b2, iv[0] % l1, iv[1] % l2, iv[2] % l3)
                                        hit = (tid, (iv[0] // l1, iv[1] // l2, iv[2] // l3)) if (is_id or tid != s_i) else None; break
                            if hit is None:
                                ok = False; break
                            p_op[s_i], cs_op[s_i] = hit
                        if not ok:
                            break
                        perms.append(p_op); csh.append(cs_op)
                    if not ok or any((p_op[u], p_op[v], tuple(int(r[k, 0] * sh[0] + r[k, 1] * sh[1] + r[k, 2] * sh[2]) + cs_op[v][k] - cs_op[u][k] for k in range(3))) not in darts for (r, _, _, _), p_op, cs_op in zip(ops, perms, csh) for u, v, sh, _ in all_b):
                        continue
                    sorb, reps = [-1] * n_sites, []
                    for s_i in range(n_sites):
                        if sorb[s_i] < 0:
                            imgs = sorted({p[s_i] for p in perms})
                            if len(imgs) != len(ops):
                                ok = False; break
                            for x in imgs:
                                sorb[x] = len(reps)
                            reps.append(imgs)
                    if not ok:
                        continue
                    seen_b, q_bonds = set(), []
                    for u, v, sh, bk in all_b:
                        can = min([((p_op[u], p_op[v], rs) if (p_op[u], rs) <= (p_op[v], tuple(-x for x in rs)) else (p_op[v], p_op[u], tuple(-x for x in rs))) for (r, _, _, _), p_op, cs_op in zip(ops, perms, csh) for rs in [tuple(int(r[k, 0] * sh[0] + r[k, 1] * sh[1] + r[k, 2] * sh[2]) + cs_op[v][k] - cs_op[u][k] for k in range(3))]])
                        if can not in seen_b:
                            seen_b.add(can); q_bonds.append({"u": sorb[u], "v": sorb[v], "vec": ((xc[v] + sh - xc[u]) @ slat).tolist(), "kind": bk, "across_seam": sorb[u] == sorb[v] or any(sh)})
                    cw_v, cw_e, cw_f, cw_c, chi = _platy_cw_chi(info["code"], ops, l_sup)
                    verts = [{"id": vid, "basis_idx": (b_i := orb[0] // (l1 * l2 * l3)), "site_kind": kinds[b_i], "frac": (fr := [float((((orb[0] % (l1 * l2 * l3)) // (l2 * l3)) + fb[b_i][0]) / l1), float(((((orb[0] % (l2 * l3)) // l3)) + fb[b_i][1]) / l2), float(((orb[0] % l3) + fb[b_i][2]) / l3)]), "pos": (np.asarray(fr) @ slat).tolist(), "orbit_size": len(orb)} for vid, orb in enumerate(reps)]
                    return {"platycosm_code": info["code"], "platycosm_name": info["name"], "status": "OK", "compatible": True, "reason": f"OK: {info['code']} ({sym}) quotient with {len(verts)} sites, {len(q_bonds)} bonds, chi_3={chi}", "orientable": info["orientable"], "ita_number": ita, "ita_symbol": sym, "h1_label": info["h1_label"], "b1": info["b1"], "supercell": list(l_sup), "origin_shift": sh_v, "vertices": verts, "bonds": q_bonds, "cw_v": cw_v, "cw_e": cw_e, "cw_f": cw_f, "cw_c": cw_c, "euler_chi_3": chi}
    if not any_m:
        l0 = cands[0]
        msg = f"NOT_FOUND [COMPUTATIONAL EVIDENCE]: {info['code']} ({info['ita_symbols'][0]}) point operations in the ITA standard setting do not preserve the metric of any searched supercell of the given cell axes (e.g. {list(l0)}); other settings are not excluded"
    else:
        msg = f"NOT_FOUND [COMPUTATIONAL EVIDENCE]: {info['code']} ({info['ita_symbols'][0]}) metric holds for some searched supercell of {[list(c) for c in cands]}, but no origin shift on the 1/4 grid gives a free {info['ita_symbols'][0]}-action by crystal symmetries; the lattice holohedry does not exclude {info['code']}"
    return _refusal(info, "NOT_FOUND", msg, supercell)


def _refusal(info: dict[str, Any], status: str, reason: str, supercell: Any) -> dict[str, Any]:
    return {"platycosm_code": info["code"], "platycosm_name": info["name"], "status": status, "compatible": False, "reason": reason, "orientable": info["orientable"], "ita_number": None, "ita_symbol": None, "h1_label": info["h1_label"], "b1": info["b1"], "supercell": list(supercell) if supercell is not None else None, "origin_shift": None, "vertices": [], "bonds": [], "cw_v": 0, "cw_e": 0, "cw_f": 0, "cw_c": 0, "euler_chi_3": 0}
