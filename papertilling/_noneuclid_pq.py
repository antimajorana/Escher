"""Unified {p,q} realiser and spherical / digon Wythoff constructions (pure-Python fallback)."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from . import _hyperbolic_seeds as seeds
from . import hyperbolic as hyp_py
from ._noneuclid_common import _dihedral_canon, _norm3, _rs_fn, _stereo_safe
from ._noneuclid_models import poincare_to_hyperboloid

def delta_pq(p: int, q: int) -> int:
    if (fn := _rs_fn("delta_pq")) is not None:
        return int(fn(int(p), int(q)))
    if p < 2 or q < 2:
        raise ValueError(f"invalid Schlafli symbol {{{p}, {q}}}: requires p >= 2 and q >= 2")
    return 4 - (p - 2) * (q - 2)


def geometry_kind_pq(p: int, q: int) -> str:
    if (fn := _rs_fn("geometry_kind_pq")) is not None:
        return str(fn(int(p), int(q)))
    d = delta_pq(p, q)
    return "Spherical" if d > 0 else ("Euclidean" if d == 0 else "Hyperbolic")


def _slerp(a, b, segs=12):
    om = math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))
    if om < 1e-9 or abs(math.pi - om) < 1e-9:
        return [list(a), list(b)]
    so = math.sin(om)
    return [_norm3(math.sin((1 - k / segs) * om) / so * np.array(a) + math.sin((k / segs) * om) / so * np.array(b)) for k in range(segs + 1)]


# Mirrored from noneuclidean2d/unified_pq.rs: E2_MAX_LAYERS (0-based layers) and the empty growth readout.
E2_MAX_LAYERS = 40
WYTHOFF_TILE_BUDGET = 600


def _no_growth() -> dict[str, Any]:
    return {"seed": None, "face_layer": [], "layer_counts": [], "layer_euler": [], "stopped": None}


def _build_e2_pq(p: int, q: int, layers: int, kappa: float) -> dict[str, Any]:
    lay = max(0, min(E2_MAX_LAYERS, int(layers)))
    rad = 0.5 / math.sin(math.pi / p)
    bv = [rad * complex(math.cos(2 * math.pi * k / p + math.pi / p), math.sin(2 * math.pi * k / p + math.pi / p)) for k in range(p)]
    bm = [0.5 * (bv[k] + bv[(k + 1) % p]) for k in range(p)]
    qk = lambda z: (round(z.real * 1e6), round(z.imag * 1e6))
    vmap, c2, f_cen, seen_c, faces = {}, [], [], {}, []

    def add_f(c: complex, s: float):
        if (ck := qk(c)) in seen_c:
            return None
        fid = seen_c[ck] = len(faces)
        f_cen.append((c, s))
        cyc = []
        for v in bv:
            z = c + s * v
            k = qk(z)
            if (vi := vmap.get(k)) is None:
                vi = vmap[k] = len(c2)
                c2.append([z.real, z.imag])
            cyc.append(vi)
        faces.append(list(reversed(cyc)) if s < 0 else cyc)
        return fid

    add_f(0j, 1.0)
    bounds, act_v = [(0, 1)], set(faces[0])
    for _ in range(lay):
        st, en = bounds[-1]
        que, vis, idx = list(range(st, en)), set(range(st, en)), 0
        while idx < len(que):
            c, s = f_cen[que[idx]]
            idx += 1
            for m in bm:
                nc, ns = c + 2 * s * m, -s
                if any(vmap.get(qk(nc + ns * v)) in act_v for v in bv):
                    nid = nid if (nid := add_f(nc, ns)) is not None else seen_c[qk(nc)]
                    if nid not in vis:
                        vis.add(nid); que.append(nid)
        bounds.append((en, len(faces)))
        act_v.update(*(faces[fi] for fi in range(en, len(faces))))
    edges = [list(e) for e in sorted({(min(f[k], f[(k + 1) % p]), max(f[k], f[(k + 1) % p])) for f in faces for k in range(p)})]
    c3 = [[x, y, 0.0] for x, y in c2]
    return {"p": p, "q": q, "layers": lay, "delta": 0, "geometry_kind": "Euclidean", "curvature_kappa": kappa,
            "euler_chi": len(c2) - len(edges) + len(faces), "n_vertices": len(c2), "n_edges": len(edges), "n_faces": len(faces),
            "coords_3d": c3, "coords_2d": c2, "edges": edges, "edge_arcs_2d": [[c2[u], c2[v]] for u, v in edges],
            "edge_arcs_3d": [[c3[u], c3[v]] for u, v in edges], "faces": faces, "provenance": "[PROVED]",
            "growth": _no_growth()}


def build_unified_pq(p: int, q: int, layers: int = 2, seed: str = "cell") -> dict[str, Any]:
    """{p,q} in S^2 / E^2 / H^2 grown from `seed` (0-based layers; see unified_pq.rs `build_unified_pq_seeded`)."""
    seed = seeds.norm_seed(seed)
    # An extension built before C1 WP-G3 has only the tile-seed function; other seeds then use the Python path.
    if (fn := _rs_fn("unified_pq_seeded_json")) is not None:
        return json.loads(fn(int(p), int(q), int(layers), seed))
    if (fn := _rs_fn("unified_pq_tiling_json")) is not None and seed == "cell":
        return json.loads(fn(int(p), int(q), int(layers)))
    delta, kappa = delta_pq(p, q), delta_pq(p, q) / (2.0 * p * q)
    growth = _no_growth()
    if delta == 0:
        if not seeds.one_tile(seed):
            raise ValueError(f"{seed.capitalize()} seed: the E² {{p,q}} preview grows from one face; vertex and edge seeds "
                             "of E² lattices are built by the plane inflation (Gallery)")
        return _build_e2_pq(p, q, layers, kappa)
    if delta > 0:
        gk, out_lay = "Spherical", layers
        if p == 2:
            c3, pa = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], math.pi / q
            e3 = [[[math.sin(math.pi * s / 16) * math.cos(2 * math.pi * k / q), math.sin(math.pi * s / 16) * math.sin(2 * math.pi * k / q), math.cos(math.pi * s / 16)] for s in range(17)] for k in range(q)]
            pole, e1, e2 = np.array([math.cos(pa), math.sin(pa), 0.0]), np.array([-math.sin(pa), math.cos(pa), 0.0]), np.array([0.0, 0.0, 1.0])
            proj = lambda pts: [[float(np.dot(x, e1) / max(1e-9, 1.0 - np.dot(x, pole))), float(np.dot(x, e2) / max(1e-9, 1.0 - np.dot(x, pole)))] for x in pts]
            c2, edges, faces, e2_a = proj(c3), [[0, 1]] * q, [[0, 1]] * q, [proj(a) for a in e3]
        elif q == 2:
            c3 = [[math.cos(2 * math.pi * k / p), math.sin(2 * math.pi * k / p), 0.0] for k in range(p)]
            c2, edges, faces = [[x[0], x[1]] for x in c3], [[k, (k + 1) % p] for k in range(p)], [list(range(p)), list(reversed(range(p)))]
            e3 = [[[math.cos(2 * math.pi * (k + s / 12) / p), math.sin(2 * math.pi * (k + s / 12) / p), 0.0] for s in range(13)] for k in range(p)]
            e2_a = [[[x[0], x[1]] for x in a] for a in e3]
        else:
            w = _wythoff_s2(p, q, "Regular", "Left")
            c3, c2, edges, faces = w["coords_3d"], w["coords_2d"], w["edges"], w["faces"]
            e3 = [_slerp(c3[u], c3[v], 12) for u, v in edges]
            e2_a = [_stereo_safe(a) for a in e3]
        growth = seeds.grow_s2(p, faces, seed, int(layers))
    else:
        h = hyp_py.build_hyperbolic_tiling(p, q, "RegularPQ", int(layers), 0, seed=seed)
        gk, out_lay, kappa = "Hyperbolic", h["layers"], h["curvature_kappa"]
        cpl = h["complete_through_shell"]
        growth = {"seed": h["seed"], "face_layer": list(h["face_shell"]),
                  "layer_counts": [sum(1 for s in h["face_shell"] if s <= k) for k in range(cpl + 1)], "layer_euler": [],
                  "stopped": h["growth_rule"] if cpl < h["layers"] else None}
        c3, c2, edges, faces, e2_a = h["hyperboloid_pts"], h["poincare_pts"], h["edges"], h["faces"], h["edge_arcs"]
        e3 = [[poincare_to_hyperboloid(z) for z in a] for a in e2_a]
    return {"p": p, "q": q, "layers": out_lay, "delta": delta, "geometry_kind": gk, "curvature_kappa": kappa,
            "euler_chi": len(c3) - len(edges) + len(faces), "n_vertices": len(c3), "n_edges": len(edges), "n_faces": len(faces),
            "coords_3d": c3, "coords_2d": c2, "edges": edges, "edge_arcs_2d": e2_a, "edge_arcs_3d": e3, "faces": faces, "provenance": "[PROVED]",
            "growth": growth}


_OPS = ("Regular", "Dual", "Rectified", "Truncated", "Bitruncated", "Cantellated", "Omnitruncated", "Snub")


def _norm_op(op: str, chirality: str) -> tuple[str, str]:
    name = {o.lower(): o for o in _OPS}.get(str(op).strip().lower())
    if name is None:
        raise ValueError(f"unknown Wythoff operation {op!r}; choose from {_OPS}")
    c = str(chirality).strip().lower()
    if c not in ("left", "l", "", "right", "r"):
        raise ValueError(f"unknown snub chirality {chirality!r}; use 'Left' or 'Right'")
    return name, ("Right" if c in ("right", "r") else "Left")


def _assemble_wythoff(p, q, op, chir, gk, c3, c2, faces):
    k = max(p, q)
    sym_map = ({"Regular": [p, q], "Dual": [q, p], "Rectified": [k, k], "Truncated": [4, 4, q] if p == 2 else [2 * p, 2 * p],
                "Bitruncated": [4, 4, p] if q == 2 else [2 * q, 2 * q], "Cantellated": [4, 4, k], "Omnitruncated": [4, 4, 2 * k], "Snub": [3, 3, 3, k]}
               if (p == 2 or q == 2) else
               {"Regular": [p] * q, "Dual": [q] * p, "Rectified": [p, q, p, q], "Truncated": [q, 2 * p, 2 * p],
                "Bitruncated": [p, 2 * q, 2 * q], "Cantellated": [p, 4, q, 4], "Omnitruncated": [4, 2 * p, 2 * q], "Snub": [3, 3, p, 3, q]})
    edges = [list(e) for e in sorted({(min(f[i], f[(i + 1) % len(f)]), max(f[i], f[(i + 1) % len(f)])) for f in faces for i in range(len(f))})]
    ftc: dict[int, int] = {}
    for f in faces: ftc[len(f)] = ftc.get(len(f), 0) + 1
    return {"p": p, "q": q, "op_name": op, "chirality": chir if op == "Snub" else None, "vertex_symbol": ".".join(map(str, _dihedral_canon(sym_map[op])[0])),
            "geometry_kind": gk, "n_vertices": len(c3), "n_edges": len(edges), "n_faces": len(faces), "euler_chi": len(c3) - len(edges) + len(faces),
            "face_type_counts": ftc, "stopped": None, "coords_3d": c3, "coords_2d": c2, "edges": edges, "faces": faces,
            "provenance": ("[COMPUTATIONAL EVIDENCE] snub generating point from a Newton solve (residual < 1e-9)" if op == "Snub"
                           else "[PROVED] Wythoff construction (Coxeter, Longuet-Higgins & Miller 1954)")}


def _wythoff_s2(p: int, q: int, op: str, chir: str) -> dict[str, Any]:
    ap, aq = math.pi / p, math.pi / q
    n0, n1 = np.array([0.0, 1.0, 0.0]), np.array([math.sin(ap), -math.cos(ap), 0.0])
    n2x = -math.cos(aq) / math.sin(ap)
    n2 = np.array([n2x, 0.0, math.sqrt(max(0.0, 1.0 - n2x * n2x))])
    A = np.vstack([n0, n1, n2])
    s = [np.eye(3) - 2.0 * np.outer(n, n) for n in (n0, n1, n2)]
    rf, rv, re, rv_inv = s[1] @ s[0], s[2] @ s[1], s[0] @ s[2], s[1] @ s[2]
    seed_of = lambda m: np.array(_norm3(np.linalg.solve(A, m)))
    cf, ce, cv = np.array([0.0, 0.0, 1.0]), seed_of([0.0, 1.0, 0.0]), seed_of([1.0, 0.0, 0.0])
    masks = {"Regular": [1, 0, 0], "Dual": [0, 0, 1], "Rectified": [0, 1, 0], "Truncated": [1, 1, 0], "Bitruncated": [0, 1, 1], "Cantellated": [1, 0, 1], "Omnitruncated": [1, 1, 1]}
    if op == "Snub":
        u = v = 0.0
        pt = lambda uu, vv: np.array(_norm3(cf + math.exp(uu) * ce + math.exp(vv) * cv))
        res = lambda uu, vv: (lambda q0: np.array([np.linalg.norm(q0 - rf @ q0) - np.linalg.norm(q0 - re @ q0), np.linalg.norm(q0 - rv @ q0) - np.linalg.norm(q0 - re @ q0)]))(pt(uu, vv))
        for _ in range(35):
            r = res(u, v)
            if float(r @ r) < 1e-24:
                break
            h = 1e-6
            J = np.column_stack([(res(u + h, v) - res(u - h, v)) / (2 * h), (res(u, v + h) - res(u, v - h)) / (2 * h)])
            du, dv = np.linalg.solve(J, r)
            u, v = u - float(du), v - float(dv)
        if not float(np.linalg.norm(res(u, v))) < 1e-9:
            raise ValueError(f"snub {{{p}, {q}}}: equal-edge generating point not found; refusing")
        seed, is_snub = pt(u, v), True
    else:
        seed, is_snub = seed_of(masks[op]), False
    qm = lambda m: tuple(np.round(m * 1e5).astype(int).ravel().tolist())

    def gen_sub(gens):
        res_l, sg, k = [np.eye(3)], {qm(np.eye(3))}, 0
        while k < len(res_l):
            c = res_l[k]; k += 1
            for gm in gens:
                nx = c @ gm
                if (kk := qm(nx)) not in sg:
                    sg.add(kk)
                    res_l.append(nx)
        return res_l

    grp = gen_sub(s)
    act = [g for g in grp if np.linalg.det(g) > 0] if is_snub else grp
    sf, sv, se = (gen_sub([rf]), gen_sub([rv]), []) if is_snub else (gen_sub([s[0], s[1]]), gen_sub([s[1], s[2]]), gen_sub([s[0], s[2]]))
    vmap, c3, faces, seen_f = {}, [], [], set()

    def gv(pt):
        lst = pt.tolist()
        k = (round(lst[0] * 1e5), round(lst[1] * 1e5), round(lst[2] * 1e5))
        if (vi := vmap.get(k)) is None:
            vi = vmap[k] = len(c3)
            c3.append(lst)
        return vi

    for sb, cen0 in ((sf, cf), (sv, cv), (se, ce)):
        for g in act:
            cyc = list(dict.fromkeys(gv(g @ (h @ seed)) for h in sb))
            if len(cyc) >= 3 and (fk := tuple(sorted(cyc))) not in seen_f:
                seen_f.add(fk)
                c = g @ cen0
                u0 = np.array(_norm3(np.cross(c, [1.0, 0.0, 0.0] if abs(c[0]) < 0.9 else [0.0, 1.0, 0.0])))
                w0 = np.cross(c, u0)
                cyc.sort(key=lambda a: math.atan2(float(np.dot(c3[a], w0)), float(np.dot(c3[a], u0))))
                faces.append(cyc)
    if is_snub:
        for g in act:
            tri = [gv(g @ seed), gv(g @ (rf @ seed)), gv(g @ (rv_inv @ seed))]
            if (fk := tuple(sorted(tri))) not in seen_f:
                seen_f.add(fk)
                a, b, c = np.array(c3[tri[0]]), np.array(c3[tri[1]]), np.array(c3[tri[2]])
                faces.append(tri if float(np.dot(np.cross(b - a, c - a), a + b + c)) >= 0 else [tri[0], tri[2], tri[1]])
    if is_snub and chir.lower().startswith("r"):
        c3, faces = [[x, -y, z] for x, y, z in c3], [list(reversed(f)) for f in faces]
    return _assemble_wythoff(p, q, op, chir, "Spherical", c3, _stereo_safe(c3), faces)


def build_wythoff(p: int, q: int, op: str = "Regular", chirality: str = "Left", layers: int = 2) -> dict[str, Any]:
    if (fn := _rs_fn("build_wythoff_json")) is not None:
        res = json.loads(fn(int(p), int(q), op, chirality, int(layers)))
        res["face_type_counts"] = {int(k): int(v) for k, v in res["face_type_counts"].items()}
        return res
    op, chirality = _norm_op(op, chirality)
    delta = delta_pq(p, q)
    if p == 2 or q == 2:
        k = max(p, q)
        if op in ("Regular", "Dual", "Rectified") or (op == "Truncated" and q == 2) or (op == "Bitruncated" and p == 2):
            pp, qq = (p, q) if op == "Regular" else ((q, p) if op == "Dual" else (k if op == "Rectified" else 2 * k, 2))
            u = build_unified_pq(pp, qq, layers)
            return _assemble_wythoff(p, q, op, chirality, "Spherical", u["coords_3d"], u["coords_2d"], u["faces"])
        snub, kk = (op == "Snub"), (2 * k if op == "Omnitruncated" else k)
        # Uniform antiprism: h^2 = A / (4 + A), A = 2 (1 + c - 2 c^2), c = cos(pi/k) (equal edges on S^2);
        # prism: h = sin(pi/k) / sqrt(1 + sin^2(pi/k)) (square sides).
        c_k = math.cos(math.pi / k)
        a_k = 2.0 * (1.0 + c_k - 2.0 * c_k * c_k)
        z = math.sqrt(a_k / (4.0 + a_k)) if snub else math.sin(math.pi / kk) / math.sqrt(1.0 + math.sin(math.pi / kk) ** 2)
        r, sh = math.sqrt(1.0 - z * z), (math.pi / k if snub else 0.0)
        n_ring = k if snub else kk
        c3 = [[r * math.cos(2 * math.pi * j / n_ring + (sh if zz < 0 else 0.0)), r * math.sin(2 * math.pi * j / n_ring + (sh if zz < 0 else 0.0)), zz] for zz in (z, -z) for j in range(n_ring)]
        sides = [f for j in range(n_ring) for f in (([j, n_ring + j, (j + 1) % n_ring], [(j + 1) % n_ring, n_ring + j, n_ring + (j + 1) % n_ring]) if snub else ([j, n_ring + j, n_ring + (j + 1) % n_ring, (j + 1) % n_ring],))]
        faces = [list(range(n_ring)), [n_ring + j for j in reversed(range(n_ring))]] + sides
        if snub and chirality == "Right":
            c3, faces = [[x, -y, zz] for x, y, zz in c3], [list(reversed(f)) for f in faces]
        return _assemble_wythoff(p, q, op, chirality, "Spherical", c3, _stereo_safe(c3), faces)
    if delta > 0:
        return _wythoff_s2(p, q, op, chirality)
    if op in ("Regular", "Dual"):
        u = build_unified_pq(p if op == "Regular" else q, q if op == "Regular" else p, layers)
        res = _assemble_wythoff(p, q, op, chirality, u["geometry_kind"], u["coords_3d"], u["coords_2d"], u["faces"])
        if delta < 0:  # the Rust H^2 Wythoff grows WYTHOFF_TILE_BUDGET tiles and reports its stop (wythoff_h2.rs)
            lay = max(0, min(hyp_py.MAX_H2_LAYERS, int(layers)))
            res["stopped"] = seeds.grow_vertex_completion_seeded(p, q, "cell", lay, WYTHOFF_TILE_BUDGET).stopped
        return res
    raise ValueError(f"pure-Python Wythoff fallback supports Spherical/Regular/Dual; build papertilling_rs for {op} on {{{p},{q}}}")
