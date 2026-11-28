"""Layer 1: Nonsymmorphic space-group extension, Wyckoff rotocenters, signed ribbon faces, and twisted coboundary."""
from __future__ import annotations

import itertools
import math

import numpy as np

from ..legacy.lab_defects import iso_compose, iso_inverse, rotation_about
from ..legacy.wallpaper_lattices import UnitCell, _symmetry_ops, identify_wallpaper_group
from .build import LatticeData, classify_charge, lattice_data, same_charge
from .planarity import _components

TWO_PI = 2.0 * math.pi


def _to_lat(lattice) -> LatticeData:
    if isinstance(lattice, LatticeData):
        return lattice
    return lattice_data(lattice.cell if hasattr(lattice, "cell") and isinstance(lattice.cell, UnitCell) else lattice)


def _to_defect(x):
    return x.defect if hasattr(x, "defect") else x


def _clean_zero(arr: np.ndarray, tol: float = 1e-11) -> np.ndarray:
    out = np.asarray(arr, float).copy()
    out[np.abs(out) < tol] = 0.0
    return out


def _mod1(v: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    f = np.asarray(v, float) - np.floor(np.asarray(v, float) + tol)
    f[(np.abs(f) < tol) | (np.abs(f - 1.0) < tol)] = 0.0
    return np.where(np.abs(f - np.round(f * 12.0) / 12.0) < 1e-11, np.round(f * 12.0) / 12.0, f)


def _mat_key(Mf: np.ndarray) -> tuple[int, int, int, int]:
    return tuple(int(x) for x in np.round(Mf).astype(int).ravel())


def _mat_order(Mf: np.ndarray) -> int:
    cur, m = np.eye(2, dtype=int), np.round(Mf).astype(int)
    for k in range(1, 13):
        cur = cur @ m
        if np.array_equal(cur, np.eye(2, dtype=int)):
            return k
    return 0


def _is_mirror_frac(Mf: np.ndarray, tau_f: np.ndarray) -> bool:
    I2 = np.eye(2, dtype=int)
    return any(np.all(np.abs((I2 + Mf) @ (tau_f + np.array([n1, n2], float))) < 1e-6)
               for n1, n2 in itertools.product(range(-3, 4), repeat=2))


def _glide_vector(A: np.ndarray, R: np.ndarray, tau: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (R + R.T))
    u = vecs[:, int(np.argmax(vals))]
    u = u / max(math.hypot(float(u[0]), float(u[1])), 1e-15)
    n = np.array([-u[1], u[0]], float)
    per = min((abs(float((A @ np.array([m1, m2], float)) @ u))
               for m1, m2 in itertools.product(range(-6, 7), repeat=2)
               if (m1 or m2) and abs(float((A @ np.array([m1, m2], float)) @ n)) < 1e-5
               and abs(float((A @ np.array([m1, m2], float)) @ u)) > 1e-6), default=None)
    if per is None:
        return np.zeros(2, float)
    g_along = float(tau @ u) % per
    if g_along < 1e-6 or abs(g_along - per) < 1e-6:
        return np.zeros(2, float)
    return _clean_zero((g_along - per if g_along > 0.5 * per + 1e-6 else g_along) * u)


def _has_trivial_cocycle(opsf: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    I2 = np.eye(2, dtype=float)
    rots = [(Mf, tf) for Mf, tf in opsf if round(np.linalg.det(Mf)) == 1 and not np.array_equal(Mf, np.eye(2, dtype=int))]
    if rots:
        inv_im = np.linalg.inv(I2 - rots[0][0])
        for n1, n2 in itertools.product(range(-2, 3), repeat=2):
            x0 = _mod1(inv_im @ (rots[0][1] + np.array([n1, n2], float)))
            if all(np.all(np.abs(((Mf - I2) @ x0 + tf) - np.round((Mf - I2) @ x0 + tf)) < 1e-5) for Mf, tf in opsf):
                return True
        return False
    return all(_is_mirror_frac(Mf, tf) for Mf, tf in opsf if round(np.linalg.det(Mf)) == -1)


def space_group_extension(lattice) -> dict:
    """Extract the exact wallpaper group extension 1 -> Lambda -> G -> P -> 1 and 2-cocycle table."""
    lat = _to_lat(lattice)
    A, Ainv = lat.A.copy(), lat.Ainv.copy()
    group = identify_wallpaper_group(lat.cell, use_types=False)
    opsf = [(np.round(Mf).astype(int), _mod1(tf)) for Mf, tf in _symmetry_ops(lat.cell, use_types=False)]
    opsf.sort(key=lambda it: (0 if round(np.linalg.det(A @ it[0] @ Ainv)) > 0 else 1,
                              round(math.degrees(math.atan2(float((A @ it[0] @ Ainv)[1, 0]),
                                                            float((A @ it[0] @ Ainv)[0, 0]))) % 360.0, 6)))
    key_to_idx, n_p, elements = {_mat_key(Mf): i for i, (Mf, _) in enumerate(opsf)}, len(opsf), []
    for i, (Mf, tf) in enumerate(opsf):
        R, tau = _clean_zero(A @ Mf @ Ainv), _clean_zero(A @ tf)
        det, ord_r = int(round(np.linalg.det(R))), _mat_order(Mf)
        if det > 0:
            ang = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
            ang = 180.0 if ang <= -180.0 + 1e-6 else ang
            kind, is_mir, is_gl, gvec = ("identity" if ord_r == 1 else "rotation"), False, False, np.zeros(2, float)
        else:
            ang, is_mir = None, _is_mirror_frac(Mf, tf)
            is_gl, kind = not is_mir, ("mirror" if is_mir else "glide")
            gvec = _glide_vector(A, R, tau) if is_gl else np.zeros(2, float)
        elements.append({"index": i, "R": R, "R_frac": Mf.copy(), "det": det, "order": ord_r,
                         "angle_deg": None if ang is None else float(round(ang, 9)),
                         "kind": kind, "is_mirror": is_mir, "is_glide": is_gl,
                         "tau": tau, "tau_frac": tf.copy(), "glide_vec": gvec})

    mul_table = np.zeros((n_p, n_p), dtype=int)
    cocycle = np.zeros((n_p, n_p, 2), dtype=float)
    cocycle_frac = np.zeros((n_p, n_p, 2), dtype=int)
    for i, (M1, t1) in enumerate(opsf):
        for j, (M2, t2) in enumerate(opsf):
            k = key_to_idx[_mat_key(M1 @ M2)]
            mul_table[i, j] = k
            nu_f = t1 + M1 @ t2 - opsf[k][1]
            nu_int = np.round(nu_f).astype(int)
            if float(np.max(np.abs(nu_f - nu_int))) > 1e-5:
                raise ValueError(f"2-cocycle failed lattice closure at ({i}, {j}): {nu_f}")
            cocycle_frac[i, j], cocycle[i, j] = nu_int, _clean_zero(A @ nu_int.astype(float))

    max_id_err = max((float(np.linalg.norm(elements[i]["R"] @ cocycle[j, k] - cocycle[mul_table[i, j], k]
                                           + cocycle[i, mul_table[j, k]] - cocycle[i, j]))
                      for i in range(n_p) for j in range(n_p) for k in range(n_p)), default=0.0)
    return {"group": group, "A": A, "Ainv": Ainv, "nonsymmorphic": not _has_trivial_cocycle(opsf),
            "elements": elements, "point_group": [e["R"] for e in elements],
            "point_group_frac": [e["R_frac"] for e in elements],
            "taus": [e["tau"] for e in elements], "taus_frac": [e["tau_frac"] for e in elements],
            "mul_table": mul_table, "cocycle": cocycle, "cocycle_frac": cocycle_frac,
            "cocycle_identity_max_err": max_id_err,
            "mirrors": [e for e in elements if e["is_mirror"]], "glides": [e for e in elements if e["is_glide"]]}


def wyckoff_rotocentres(lattice) -> dict:
    """Compute distinct Wyckoff rotocenter classes Lambda / (I - R) Lambda and improper pairings."""
    lat = _to_lat(lattice)
    ext = space_group_extension(lat)
    A, Ainv, m0 = lat.A, lat.Ainv, lat.m0
    classes, by_label = [], {}
    for rec in lat.centres:
        pt = _clean_zero(np.asarray(rec["point"], float))
        hit = None if m0 is None else lat.centre_class(m0[0] @ pt + m0[1])
        partner = None if hit is None else hit["label"]
        entry = {"label": rec["label"], "gclass": rec.get("gclass", rec["label"]), "order": int(rec["order"]),
                 "point": pt, "frac": _clean_zero(_mod1(Ainv @ pt)),
                 "members": [_clean_zero(np.asarray(m, float)) for m in rec["members"]],
                 "members_frac": [_clean_zero(_mod1(Ainv @ m)) for m in rec["members"]],
                 "on_site": bool(rec["on_site"]), "angles": list(rec["angles"]), "mirror_partner": partner,
                 "splits_under_Gplus": partner is not None and partner != rec["label"]}
        classes.append(entry); by_label[entry["label"]] = entry

    by_rotation = {}
    for el in ext["elements"]:
        if el["det"] != 1 or el["order"] <= 1:
            continue
        I_M = np.eye(2, dtype=float) - el["R_frac"]
        det_imr, inv_im = int(round(abs(np.linalg.det(I_M)))), np.linalg.inv(I_M)
        cosets, seen_cf = [], []
        for n1, n2 in itertools.product(range(-3, 4), repeat=2):
            m_vec = np.array([n1, n2], dtype=int)
            cf = _mod1(inv_im @ (el["tau_frac"] + m_vec.astype(float)))
            if any(np.all(np.abs(_mod1(cf - q)) < 1e-5) for q in seen_cf):
                continue
            seen_cf.append(cf); c_cart = _clean_zero(A @ cf); hit = lat.centre_class(c_cart)
            cosets.append({"coset_shift_frac": m_vec, "coset_shift": _clean_zero(A @ m_vec.astype(float)),
                           "centre": c_cart, "centre_frac": _clean_zero(cf),
                           "class_label": None if hit is None else hit["label"]})
            if len(cosets) == det_imr:
                break
        by_rotation[int(round(el["angle_deg"]))] = {"R": el["R"], "R_frac": el["R_frac"], "det_I_minus_R": det_imr, "cosets": cosets}

    improper_pairings = []
    for el in ext["elements"]:
        if el["det"] != -1:
            continue
        g = (el["R"], el["tau"])
        g2 = iso_compose(g, g)
        rot_maps = []
        for rec in classes:
            conj = iso_compose(g, iso_compose(rotation_about(rec["point"], math.radians(360.0 / rec["order"])), iso_inverse(g)))
            c_im = np.linalg.solve(np.eye(2) - conj[0], conj[1])
            hit = lat.centre_class(c_im)
            rot_maps.append({"from_label": rec["label"], "from_centre": rec["point"],
                             "to_label": None if hit is None else hit["label"], "to_centre": _clean_zero(c_im),
                             "conj_H": (_clean_zero(conj[0]), _clean_zero(conj[1]))})
        improper_pairings.append({"kind": el["kind"], "R": el["R"], "tau": el["tau"], "tau_frac": el["tau_frac"],
                                  "square_trans": _clean_zero(g2[1]), "square_trans_frac": _clean_zero(Ainv @ g2[1]),
                                  "is_involutive_mod_lattice": float(np.linalg.norm(g2[1])) < 1e-7,
                                  "rotocentre_maps": rot_maps})

    return {"group": ext["group"], "nonsymmorphic": ext["nonsymmorphic"], "classes": classes,
            "by_label": by_label, "by_rotation": by_rotation, "improper_pairings": improper_pairings}


def _resolve_fusion_item(wmap: dict[str, dict], item) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    if isinstance(item, dict):
        if "H" in item:
            H = (np.asarray(item["H"][0], float), np.asarray(item["H"][1], float))
            ang = item.get("angle")
            if ang is None:
                ang = 0.0 if np.allclose(H[0], np.eye(2), atol=1e-6) else math.degrees(math.atan2(H[0][1, 0], H[0][0, 0]))
            return float(ang), H
        if item.get("kind") == "dislocation":
            return 0.0, (np.eye(2, float), -np.asarray(item["b"], float))
        if item.get("kind") == "disclination":
            return float(item["angle"]), rotation_about(np.asarray(item["at"], float), math.radians(float(item["angle"])))
        if "label" in item and "angle" in item:
            return float(item["angle"]), rotation_about(wmap[item["label"]]["point"], math.radians(float(item["angle"])))
    if isinstance(item, (tuple, list)) and len(item) == 2:
        if isinstance(item[0], str):
            return float(item[1]), rotation_about(wmap[item[0]]["point"], math.radians(float(item[1])))
        M, t = np.asarray(item[0], float), np.asarray(item[1], float)
        if M.shape == (2, 2) and t.shape == (2,):
            ang = 0.0 if np.allclose(M, np.eye(2), atol=1e-6) else math.degrees(math.atan2(M[1, 0], M[0, 0]))
            return float(ang), (M, t)
    raise TypeError(f"unsupported defect/charge item for fusion_channels: {item!r}")


def fusion_channels(lattice, specs_or_charges, *, max_shells: int = 3, non_orientable: bool = False) -> dict:
    """Compute basepoint-free conjugate fusion channels {[g1 h g2 h^-1] : h in G+} ordered by elastic energy."""
    lat, ext, wyck = _to_lat(lattice), space_group_extension(lattice), wyckoff_rotocentres(lattice)
    if len(specs_or_charges) != 2:
        raise ValueError("specs_or_charges must contain exactly two defect specs or charges")
    a1, H1 = _resolve_fusion_item(wyck["by_label"], specs_or_charges[0])
    a2, H2 = _resolve_fusion_item(wyck["by_label"], specs_or_charges[1])
    proper_ops = [(e["R"], e["tau"]) for e in ext["elements"] if e["det"] == 1]
    improper_ops = [(e["R"], e["tau"]) for e in ext["elements"] if e["det"] == -1] if non_orientable else []
    candidates: list[tuple[float, dict, list[dict] | None]] = []
    is_turn = False
    if abs(a1) < 1e-6 and abs(a2) < 1e-6:
        law, b1 = "T×T", -H1[1]
        for R_h, _ in proper_ops:
            H_tot = (np.eye(2, float), -(b1 + R_h @ (-H2[1])))
            candidates.append((float(np.dot(H_tot[1], H_tot[1])), classify_charge(lat, 0.0, H_tot), None))
    elif abs(a1) < 1e-6 or abs(a2) < 1e-6:
        law, is_d1 = "D×T", abs(a1) >= 1e-6
        ang, H_d, b_base = (a1, H1, -H2[1]) if is_d1 else (a2, H2, -H1[1])
        c0 = np.linalg.solve(np.eye(2) - H_d[0], H_d[1])
        for R_h, _ in proper_ops:
            t_op = (np.eye(2, float), -(R_h @ b_base))
            H_tot = iso_compose(H_d, t_op) if is_d1 else iso_compose(t_op, H_d)
            candidates.append((float(np.linalg.norm(np.linalg.solve(np.eye(2) - H_tot[0], H_tot[1]) - c0)),
                               classify_charge(lat, ang, H_tot), None))
    else:
        ang_sum = a1 + a2
        c1, c2 = np.linalg.solve(np.eye(2) - H1[0], H1[1]), np.linalg.solve(np.eye(2) - H2[0], H2[1])
        # A whole number of turns (Frank angle +-360, ...) has the identity rotation as holonomy, so the outcome is
        # a Burgers vector (translation), exactly as for an annihilating pair; the Frank angle itself is not trivial.
        turns = round(ang_sum / 360.0)
        is_turn = abs(ang_sum - 360.0 * turns) < 1e-6 and turns != 0
        if not is_turn and ang_sum >= 360.0 - 1e-6:       # cone-closing angle with a non-trivial rotation: not handled
            return {"law": "D×D (turn)", "channels": [], "contains_vacuum": False,
                    "infinite_tower": False, "frank_angle_deg": ang_sum}
        is_pair = abs(ang_sum) < 1e-6 or is_turn          # identity rotation: the charge is a translation
        law = "D×D (turn)" if is_turn else ("D×D->T" if is_pair else "D×D->D")
        lim = 5 if is_pair else max(1, int(round(abs(np.linalg.det(np.eye(2) - H1[0] @ H2[0]))))) + 2
        bases = [(R_h @ c2 + t_h) for R_h, t_h in proper_ops]
        if abs(ang_sum) < 1e-6 and improper_ops:
            H2m = iso_compose(improper_ops[0], iso_compose(iso_inverse(H2), iso_inverse(improper_ops[0])))
            c2m = np.linalg.solve(np.eye(2) - H2m[0], H2m[1])
            bases.extend(R_h @ c2m + t_h for R_h, t_h in proper_ops)
        for c2_0 in bases:
            for n1, n2 in itertools.product(range(-lim, lim + 1), repeat=2):
                c2_p = c2_0 + lat.A @ np.array([n1, n2], float)
                H_tot = iso_compose(H1, rotation_about(c2_p, math.radians(a2)))
                e_p = float(np.dot(H_tot[1], H_tot[1])) if is_pair else float(np.linalg.norm(c2_p - c1))
                wit = ([] if (is_pair and not is_turn and e_p < 1e-12) else
                       [{"kind": "disclination", "angle": a1, "at": c1.copy()},
                        {"kind": "disclination", "angle": a2, "at": c2_p.copy()}])
                candidates.append((e_p, classify_charge(lat, 0.0 if is_pair else ang_sum, H_tot), wit))
    candidates.sort(key=lambda item: (round(item[0], 9), item[1]["label"]))
    dedup = []
    for e_p, q, wit in candidates:
        if any(same_charge(lat, q, prev["charge"]) for prev in dedup):
            continue
        b = q.get("b")
        dedup.append({"kind": q["kind"], "label": q["label"], "charge": q, "H": q["H"],
                      "b": None if b is None else b.copy(), "b_frac": None if b is None else np.round(lat.Ainv @ b, 6),
                      "burgers_norm": None if b is None else float(np.linalg.norm(b)),
                      "centre": q.get("centre"), "centre_class": q.get("centre_class"),
                      "energy_proxy": float(e_p), "witness_specs": wit})
    if is_turn:                                         # kind/label describe the holonomy; the Frank angle is the turn
        for c in dedup:
            c["frank_angle_deg"] = float(ang_sum)
            c["label"] = f"{360 * turns:+d} deg turn + {c['label']}"
    if law in ("D×D->T", "D×D (turn)"):
        shells = sorted({round(c["energy_proxy"], 6) for c in dedup if c["energy_proxy"] > 1e-9})[:max_shells]
        dedup = [c for c in dedup if c["energy_proxy"] < 1e-9 or round(c["energy_proxy"], 6) in shells]
    min_e = min((c["energy_proxy"] for c in dedup), default=0.0)
    return {"law": law, "channels": dedup,
            "minimal_channels": [c for c in dedup if abs(c["energy_proxy"] - min_e) < 1e-6],
            "contains_vacuum": any(c["kind"] == "vacuum" for c in dedup) and not is_turn,
            "infinite_tower": law in ("D×D->T", "D×D (turn)")}


class RibbonFace(list):
    """Face cycle from signed dart tracing (list[int] with dart sign metadata)."""

    def __init__(self, verts: list[int], signs: list[int], crosses_seam: bool):
        super().__init__(verts)
        self.verts, self.signs = list(verts), list(signs)
        self.darts = [(verts[k], verts[(k + 1) % len(verts)], signs[k]) for k in range(len(verts))]
        self.crosses_seam = bool(crosses_seam)


def _edge_data(d, u: int, v: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None, float]:
    from .circuit import _edge_vec_and_g

    vec, g = _edge_vec_and_g(d, u, v)
    e = d.edge[(u, v)]
    return vec, g, (float(e["ang"]) if "ang" in e else math.atan2(float(vec[1]), float(vec[0])) % TWO_PI)


def _rotation_orders(d):
    nbr, edata = {}, {}
    for (u, v) in d.edge:
        vec, g, ang = _edge_data(d, u, v)
        nbr.setdefault(u, []).append((ang, v))
        edata[(u, v)] = (vec, g, 1 if (g is None or float(np.linalg.det(g[0])) > 0.0) else -1)
    sorted_nbr = {u: sorted(lst) for u, lst in nbr.items()}
    order = {u: [v for _, v in lst] for u, lst in sorted_nbr.items()}
    idx_map = {u: {v: k for k, v in enumerate(lst)} for u, lst in order.items()}
    return sorted_nbr, order, idx_map, edata


def ribbon_faces(crystal_or_config, *, include_boundary: bool = False, with_signs: bool = False, _orders=None) -> list:
    """Trace bounded polygon faces via signed darts (u, v, sign) across det(g) < 0 seams."""
    d = _to_defect(crystal_or_config)
    if not d.edge:
        return []
    _, order, idx_map, edata = _orders or _rotation_orders(d)
    seen, all_walks = set(), []
    for u in sorted(order):
        for v in order[u]:
            for s0 in (1, -1):
                if (u, v, s0) in seen:
                    continue
                states, verts, signs, a_cur, b_cur, s_cur, seam_hit = [], [], [], u, v, s0, False
                while (a_cur, b_cur, s_cur) not in seen:
                    states.append((a_cur, b_cur, s_cur)); seen.add((a_cur, b_cur, s_cur))
                    verts.append(a_cur); signs.append(s_cur)
                    gs = edata[(a_cur, b_cur)][2]; seam_hit = seam_hit or (gs < 0)
                    s_next, lst, k = s_cur * gs, order[b_cur], idx_map[b_cur][a_cur]
                    a_cur, b_cur, s_cur = b_cur, lst[(k - 1) % len(lst)] if s_next == 1 else lst[(k + 1) % len(lst)], s_next
                for aa, bb, ss in states:
                    seen.add((bb, aa, -(ss * edata[(aa, bb)][2])))
                all_walks.append((verts, signs, seam_hit))
    if not include_boundary and not d.info.get("torus") and not d.info.get("klein") and all_walks:
        root, outer = _components(len(d.sites), d.edges), {}
        for i, (verts, _, _) in enumerate(all_walks):
            c = root(verts[0])
            if c not in outer or len(verts) > len(all_walks[outer[c]][0]):
                outer[c] = i
        all_walks = [w for i, w in enumerate(all_walks) if i not in set(outer.values())]
    return [(verts, signs) for verts, signs, _ in all_walks] if with_signs else [RibbonFace(*w) for w in all_walks]


def plaquette_holonomies(config_or_defect, *, lattice=None) -> list[dict]:
    """Compute W(df) = (R_f, t_f), Frank angle, det(R_f), Burgers/torsion alpha_f, and curvature/dispiration flags."""
    d = _to_defect(config_or_defect)
    lat = _to_lat(lattice) if lattice is not None else getattr(config_or_defect, "lattice", None)
    orders = _rotation_orders(d)
    sorted_nbr, _, idx_map, edata = orders
    I2, ident, out = np.eye(2, dtype=float), (np.eye(2, dtype=float), np.zeros(2, float)), []
    for fi, face in enumerate(ribbon_faces(d, include_boundary=False, _orders=orders)):
        verts, signs = list(face), face.signs
        loop = verts if signs[0] == 1 else ([verts[0]] + verts[1:][::-1])
        Dm, H, alpha, crosses = ident, ident, np.zeros(2, float), face.crosses_seam
        for k in range(len(loop)):
            vec, g, _ = edata[(loop[k], loop[(k + 1) % len(loop)])]
            alpha = alpha + Dm[0] @ vec
            if g is not None:
                crosses = True; Dm, H = iso_compose(Dm, iso_inverse(g)), iso_compose(g, H)
        R_f, t_f, alpha = _clean_zero(H[0]), _clean_zero(H[1]), _clean_zero(alpha)
        det = int(round(np.linalg.det(R_f)))
        omega_rad = math.atan2(float(R_f[1, 0]), float(R_f[0, 0])) if det > 0 else None
        tot, n = 0.0, len(verts)
        for k in range(n):
            lst, i = sorted_nbr[verts[k]], idx_map[verts[k]][verts[k - 1]]
            j = (i + 1) % len(lst) if signs[k] != 1 else i
            tot += ((lst[i][0] - lst[i - 1][0]) if i > 0 else (lst[0][0] + TWO_PI - lst[-1][0])) if signs[k] == 1 \
                else ((lst[j][0] - lst[i][0]) if j > 0 else (lst[0][0] + TWO_PI - lst[-1][0]))
        frank_deg = float(round(math.degrees(tot - (n - 2) * math.pi), 6))
        has_curv = (det < 0) or (not np.allclose(R_f, I2, atol=1e-6)) or abs(frank_deg) > 1e-5
        alpha_norm = float(np.linalg.norm(alpha))
        rotocentre = None if (det < 0 or np.allclose(R_f, I2, atol=1e-6)) else np.linalg.solve(I2 - R_f, t_f)
        nonsym = (not lat.same_mod_lattice(alpha, np.zeros(2)) if (not has_curv and alpha_norm > 1e-6)
                  else (lat.centre_class(rotocentre) is None if (has_curv and rotocentre is not None) else False)) if lat else False
        out.append({"face_index": fi, "verts": verts, "signs": signs, "ring": n, "H": (R_f, t_f), "R": R_f, "t": t_f,
                    "det": det, "omega_rad": omega_rad,
                    "omega_deg": None if omega_rad is None else float(round(math.degrees(omega_rad), 6)),
                    "frank_deg": frank_deg, "alpha": alpha, "alpha_norm": alpha_norm, "rotocentre": rotocentre,
                    "has_curvature": has_curv, "has_torsion": (not has_curv) and (alpha_norm > 1e-6),
                    "is_flat": not has_curv, "is_dispiration": bool(nonsym or (has_curv and det < 0)), "crosses_seam": crosses})
    return out


def twisted_coboundary(crystal_or_defect, cochain, *, faces=None):
    """Apply twisted coboundary d_R: 0-cochain -> 1-cochain (R_e m_{t(e)} - m_{s(e)}), or 1-cochain -> 2-cochain."""
    d = _to_defect(crystal_or_defect)
    V = len(d.sites)
    edge_R = {(u, v): (np.eye(2, dtype=float) if g is None else np.asarray(g[0], float).T)
              for (u, v) in d.edge for _, g, _ in (_edge_data(d, u, v),)}
    if not (isinstance(cochain, dict) and bool(cochain) and isinstance(next(iter(cochain)), tuple)):
        if isinstance(cochain, dict):
            m = {int(k): np.asarray(val, float) for k, val in cochain.items()}
        else:
            arr = np.asarray(cochain, float)
            if arr.shape[0] != V:
                raise ValueError(f"0-cochain length {arr.shape[0]} != number of sites {V}")
            m = {i: arr[i] for i in range(V)}
        return {(u, v): _clean_zero(edge_R[(u, v)] @ m[v] - m[u]) for (u, v) in d.edge}
    out = []
    for face in (ribbon_faces(d, include_boundary=False) if faces is None else faces):
        verts, signs = list(face), getattr(face, "signs", [1] * len(face))
        loop = verts if signs[0] == 1 else ([verts[0]] + verts[1:][::-1])
        F, acc = np.eye(2, dtype=float), np.zeros(2, float)
        for k in range(len(loop)):
            u, v = loop[k], loop[(k + 1) % len(loop)]
            acc, F = acc + F @ np.asarray(cochain[(u, v)], float), F @ edge_R[(u, v)]
        out.append(_clean_zero(acc))
    return out
