"""H^2 model maps, distances, geodesics, 3D surfaces and papercraft nets (pure-Python fallback).

Cayley w = i (1 + z) / (1 - z); Klein k = 2 z / (1 + |z|^2); hyperboloid (2 z, 1 + |z|^2) / (1 - |z|^2);
sinh^2(d/2) = |z1 - z2|^2 / ((1 - |z1|^2)(1 - |z2|^2)) (Beardon, The Geometry of Discrete Groups, 1983,
ch. 7). Pseudosphere, Dini (a^2 + b^2 = 1) and Kuen surfaces have Euclidean Gaussian curvature -1;
only the HyperboloidBowl placement is an isometry (Minkowski metric), the others are visual charts.
"""
from __future__ import annotations

from collections import deque
import json
import math
from typing import Any

import numpy as np

from . import _hyperbolic_vc as vc
from . import hyperbolic as hyp_py
from ._noneuclid_common import _check_p, _rs_fn

def poincare_to_uhp(z) -> list[float]:
    if (fn := _rs_fn("poincare_to_uhp")) is not None: return list(fn(list(z)))
    r2 = _check_p(z); d = (1.0 - z[0]) ** 2 + z[1] ** 2
    return [(-2.0 * z[1]) / d, (1.0 - r2) / d]


def uhp_to_poincare(w) -> list[float]:
    if (fn := _rs_fn("uhp_to_poincare")) is not None: return list(fn(list(w)))
    if w[1] <= 0.0: raise ValueError("point outside UHP")
    d = w[0] ** 2 + (w[1] + 1.0) ** 2
    return [(w[0] ** 2 + w[1] ** 2 - 1.0) / d, (-2.0 * w[0]) / d]


def poincare_to_klein(z) -> list[float]:
    return list(fn(list(z))) if (fn := _rs_fn("poincare_to_klein")) is not None else hyp_py.to_klein(z)


def klein_to_poincare(k) -> list[float]:
    if (fn := _rs_fn("klein_to_poincare")) is not None: return list(fn(list(k)))
    if (r2 := k[0] * k[0] + k[1] * k[1]) >= 1.0: raise ValueError("point outside Klein disk")
    s = 1.0 + math.sqrt(1.0 - r2)
    return [k[0] / s, k[1] / s]


def poincare_to_hyperboloid(z) -> list[float]:
    if (fn := _rs_fn("poincare_to_hyperboloid_exact")) is not None: return list(fn(list(z)))
    d = 1.0 - _check_p(z)
    return [2.0 * z[0] / d, 2.0 * z[1] / d, (2.0 - d) / d]


def hyperboloid_to_poincare(x) -> list[float]:
    return list(fn(list(x))) if (fn := _rs_fn("hyperboloid_to_poincare_exact")) is not None else vc.hyperboloid_to_poincare(x)


def hyp_distance_poincare(z1, z2) -> float:
    if (fn := _rs_fn("hyp_distance_poincare")) is not None: return float(fn(list(z1), list(z2)))
    return 2.0 * math.asinh(math.sqrt(((z1[0] - z2[0]) ** 2 + (z1[1] - z2[1]) ** 2) / ((1.0 - _check_p(z1)) * (1.0 - _check_p(z2)))))


def hyp_distance_uhp(w1, w2) -> float:
    if (fn := _rs_fn("hyp_distance_uhp")) is not None: return float(fn(list(w1), list(w2)))
    if w1[1] <= 0.0 or w2[1] <= 0.0: raise ValueError("point outside UHP")
    return 2.0 * math.asinh(math.hypot(w1[0] - w2[0], w1[1] - w2[1]) / (2.0 * math.sqrt(w1[1] * w2[1])))


def geodesic_arc_poincare(z1, z2, steps: int = 8) -> list[list[float]]:
    if (fn := _rs_fn("geodesic_arc_poincare")) is not None: return [list(p) for p in fn(list(z1), list(z2), int(steps))]
    _check_p(z1); _check_p(z2)
    n, a, b = max(1, int(steps)), complex(*z1), complex(*z2)
    w2 = (b - a) / (1.0 - a.conjugate() * b)
    if (r2 := abs(w2)) < 1e-14:
        return [list(z1) for _ in range(n + 1)]
    d_tot, u = 2.0 * math.atanh(r2), w2 / r2
    pts = [(math.tanh(0.5 * d_tot * k / n) * u + a) / (1.0 + a.conjugate() * math.tanh(0.5 * d_tot * k / n) * u) for k in range(n + 1)]
    out = [[p.real, p.imag] for p in pts]
    out[0], out[-1] = list(z1), list(z2)
    return out


def geodesic_arc_uhp(w1, w2, steps: int = 8) -> list[list[float]]:
    if (fn := _rs_fn("geodesic_arc_uhp")) is not None: return [list(p) for p in fn(list(w1), list(w2), int(steps))]
    return [poincare_to_uhp(z) for z in geodesic_arc_poincare(uhp_to_poincare(w1), uhp_to_poincare(w2), steps)]


_SURF_KINDS = {"hyperboloidbowl": "HyperboloidBowl", "bowl": "HyperboloidBowl", "hemisphere": "Hemisphere",
               "pseudosphere": "Pseudosphere", "dini": "Dini", "kuen": "Kuen"}


def _canon_kind(kind: str) -> str:
    if (name := _SURF_KINDS.get(str(kind).strip().lower())) is None:
        raise ValueError(f"unknown H2 surface kind {kind!r}; choose from {sorted(set(_SURF_KINDS.values()))}")
    return name


def _surf_pt(kind: str, twist: float, u: float, v: float) -> list[float]:
    kl = _canon_kind(kind).lower()
    if kl in ("hyperboloidbowl", "bowl"): s = max(0.0, v); return [math.sinh(s) * math.cos(u), math.sinh(s) * math.sin(u), math.cosh(s)]
    if kl == "hemisphere": r = max(0.0, min(0.999, v)); return [r * math.cos(u), r * math.sin(u), math.sqrt(1.0 - r * r)]
    if kl == "pseudosphere": t = max(0.0, v); r = 1.0 / math.cosh(t); return [r * math.cos(u), r * math.sin(u), t - math.tanh(t)]
    if kl == "dini":
        t, nm = max(0.0, v), math.sqrt(1.0 + twist * twist); a, b = 1.0 / nm, twist / nm
        return [(a / math.cosh(t)) * math.cos(u), (a / math.cosh(t)) * math.sin(u), a * (t - math.tanh(t)) + b * u]
    vv = max(0.05, min(math.pi - 0.05, v)); w = 1.0 + u * u * math.sin(vv) ** 2
    return [2 * (math.cos(u) + u * math.sin(u)) * math.sin(vv) / w, 2 * (math.sin(u) - u * math.cos(u)) * math.sin(vv) / w, math.log(math.tan(0.5 * vv)) + 2 * math.cos(vv) / w]


def map_poincare_to_surface(z, kind: str = "HyperboloidBowl", twist: float = 0.35) -> list[float]:
    if (fn := _rs_fn("map_poincare_to_surface")) is not None: return list(fn(list(z), kind, float(twist)))
    r2, kl = _check_p(z), _canon_kind(kind).lower()
    if kl in ("hyperboloidbowl", "bowl"): return poincare_to_hyperboloid(z)
    if kl == "hemisphere": d = 1.0 + r2; return [2.0 * z[0] / d, 2.0 * z[1] / d, (1.0 - r2) / d]
    if kl in ("pseudosphere", "dini"): return _surf_pt(kind, twist, math.atan2(z[1], z[0]), 2.0 * math.atanh(math.sqrt(r2)))
    return _surf_pt("kuen", twist, 3.2 * z[0], 0.5 * math.pi + 1.1 * z[1])


def build_h2_surface_mesh(kind: str = "Pseudosphere", twist: float = 0.35, u_steps: int = 24, v_steps: int = 16) -> dict[str, Any]:
    if (fn := _rs_fn("build_h2_surface_mesh_json")) is not None: return json.loads(fn(kind, float(twist), int(u_steps), int(v_steps)))
    nu, nv = max(4, min(128, u_steps)), max(4, min(128, v_steps))
    rng = {"hyperboloidbowl": ((0, 2 * math.pi), (0, 2.2), -1.0), "hemisphere": ((0, 2 * math.pi), (0, 0.98), 1.0),
           "pseudosphere": ((0, 2 * math.pi), (0, 2.4), -1.0), "dini": ((0, 4 * math.pi), (0, 2.2), -1.0), "kuen": ((-3.5, 3.5), (0.25, math.pi - 0.25), -1.0)}[_canon_kind(kind).lower()]
    pos = [_surf_pt(kind, twist, rng[0][0] + (rng[0][1] - rng[0][0]) * iu / nu, rng[1][0] + (rng[1][1] - rng[1][0]) * iv / nv) for iv in range(nv + 1) for iu in range(nu + 1)]
    faces = [[iv * (nu + 1) + iu, iv * (nu + 1) + iu + 1, (iv + 1) * (nu + 1) + iu + 1, (iv + 1) * (nu + 1) + iu] for iv in range(nv) for iu in range(nu)]
    return {"kind_name": _canon_kind(kind), "expected_gaussian_curvature": rng[2], "positions_3d": pos, "faces": faces}


def embed_tiling_on_surface(coords_poincare, faces, kind: str = "HyperboloidBowl", twist: float = 0.35) -> dict[str, Any]:
    if (fn := _rs_fn("embed_tiling_on_surface_json")) is not None: return json.loads(fn(json.dumps(coords_poincare), json.dumps(faces), kind, float(twist)))
    return {"kind_name": _canon_kind(kind), "expected_gaussian_curvature": 1.0 if _canon_kind(kind) == "Hemisphere" else -1.0,
            "positions_3d": [map_poincare_to_surface(z, kind, twist) for z in coords_poincare], "faces": faces}


def unfold_papercraft_net(coords_3d, faces) -> dict[str, Any]:
    if (fn := _rs_fn("unfold_papercraft_net_json")) is not None: return json.loads(fn(json.dumps(coords_3d), json.dumps(faces)))
    pts3 = np.asarray(coords_3d, dtype=float)
    d3 = lambda i, j: float(np.linalg.norm(pts3[i] - pts3[j]))
    local = []
    for f in faces:
        pts = [np.array([0.0, 0.0]), np.array([d3(f[0], f[1]), 0.0])]
        for k in range(2, len(f)):
            a, b, c = d3(f[0], f[k - 1]), d3(f[k - 1], f[k]), d3(f[0], f[k])
            u = (pts[-1] - pts[0]) / max(1e-12, a)
            ca = max(-1.0, min(1.0, (a * a + c * c - b * b) / max(1e-12, 2 * a * c)))
            sa = math.sqrt(max(0.0, 1.0 - ca * ca))
            pts.append(pts[0] + c * np.array([u[0] * ca - u[1] * sa, u[0] * sa + u[1] * ca]))
        local.append(np.array(pts))
    einc = {}
    for fi, f in enumerate(faces):
        for s in range(len(f)): einc.setdefault((min(f[s], f[(s + 1) % len(f)]), max(f[s], f[(s + 1) % len(f)])), []).append((fi, s))
    placed, dq, trees, tset = [None] * len(faces), deque([0]), [], set()
    placed[0] = local[0]
    while dq:
        fa = dq.popleft(); pa = placed[fa]; fc = faces[fa]
        for sa in range(len(fc)):
            u, v = fc[sa], fc[(sa + 1) % len(fc)]
            for fb, sb in einc[(min(u, v), max(u, v))]:
                if fb != fa and placed[fb] is None:
                    loc = local[fb]; su, sv = loc[sb], loc[(sb + 1) % len(loc)]
                    dst0, dst1 = (pa[sa], pa[(sa + 1) % len(fc)]) if fc[sa] == faces[fb][sb] else (pa[(sa + 1) % len(fc)], pa[sa])
                    ang = math.atan2(dst1[1] - dst0[1], dst1[0] - dst0[0]) - math.atan2(sv[1] - su[1], sv[0] - su[0])
                    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
                    placed[fb] = dst0 + (loc - su) @ rot.T
                    trees.append([fa, sa, fb, sb]); tset.add((min(fa, fb), max(fa, fb))); dq.append(fb)
    cuts = [[a, b, c, d] for inc in einc.values() if len(inc) == 2 for (a, b), (c, d) in (inc,) if (min(a, c), max(a, c)) not in tset]
    merr = max(abs(d3(f[s], f[(s + 1) % len(f)]) - float(np.linalg.norm(placed[fi][s] - placed[fi][(s + 1) % len(f)]))) for fi, f in enumerate(faces) for s in range(len(f)))
    return {"pieces": [p.tolist() for p in placed], "tree_edges": trees, "cut_pairings": cuts, "max_edge_length_error": merr}
