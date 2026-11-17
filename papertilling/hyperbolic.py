"""Hyperbolic H^2 (p, q) tilings, rectified (p.q.p.q) kagome, Poincare geodesics, Klein chords, and 3D hyperboloid."""
from __future__ import annotations

import json
import math
from typing import Any

from . import _rust
from .dsymbol import orbifold_curvature

__all__ = ["build_hyperbolic_tiling", "to_klein", "to_hyperboloid"]


def to_klein(z: list[float] | tuple[float, float]) -> list[float]:
    d = 1.0 + z[0] * z[0] + z[1] * z[1]
    return [2.0 * z[0] / d, 2.0 * z[1] / d]


def to_hyperboloid(z: list[float] | tuple[float, float]) -> list[float]:
    n2 = z[0] * z[0] + z[1] * z[1]
    if n2 > 0.9025:
        sc = math.sqrt(0.9025 / n2)
        x, y = z[0] * sc, z[1] * sc
    else:
        x, y = z[0], z[1]
    d = max(0.05, 1.0 - (x * x + y * y))
    return [2.0 * x / d, 2.0 * y / d, (1.0 + x * x + y * y) / d]


def _orth_circle(a, b):
    det = a[0] * b[1] - a[1] * b[0]
    if abs(det) < 1e-10:
        return None
    r0, r1 = 0.5 * (1.0 + a[0] ** 2 + a[1] ** 2), 0.5 * (1.0 + b[0] ** 2 + b[1] ** 2)
    cx, cy = (r0 * b[1] - a[1] * r1) / det, (a[0] * r1 - r0 * b[0]) / det
    r2 = cx * cx + cy * cy - 1.0
    return (cx, cy, r2) if r2 > 0.0 else None


def _reflect(z, a, b):
    oc = _orth_circle(a, b)
    if oc is None:
        ln = max(1e-18, math.hypot(a[0], a[1]))
        ux, uy = a[0] / ln, a[1] / ln
        u2x, u2y = ux * ux - uy * uy, 2.0 * ux * uy
        return [u2x * z[0] + u2y * z[1], u2y * z[0] - u2x * z[1]]
    cx, cy, r2 = oc
    dx, dy = z[0] - cx, -(z[1] - cy)
    den = max(1e-18, dx * dx + dy * dy)
    return [cx + r2 * dx / den, cy - r2 * dy / den]


def build_hyperbolic_tiling(
    p: int = 7,
    q: int = 3,
    style: str = "RegularPQ",
    layers: int = 2,
    wedge_delta: int = 0,
) -> dict[str, Any]:
    """Generate a hyperbolic {p, q} / rectified (p.q.p.q) / dual {q, p} tiling patch in D and H^2."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "hyperbolic_tiling_json"):
        return json.loads(rs.hyperbolic_tiling_json(int(p), int(q), style, int(layers), int(wedge_delta)))
    pe, qe = (q, p) if style == "DualQP" else (p, q)
    if pe < 3 or qe < 3 or (pe - 2) * (qe - 2) <= 4:
        raise ValueError(f"{{{pe}, {qe}}} is not hyperbolic (requires (p-2)(q-2) > 4)")
    r0 = math.sqrt(max(0.0, math.cos(math.pi / pe + math.pi / qe) / math.cos(math.pi / pe - math.pi / qe)))
    p_core = max(3, min(12, pe + int(wedge_delta)))
    polys = [[[r0 * math.cos(2 * math.pi * k / p_core), r0 * math.sin(2 * math.pi * k / p_core)] for k in range(p_core)]]
    centers = [[0.0, 0.0]]
    queue = [(0, 0)]
    lay = max(1, min(4, int(layers)))
    while queue and len(polys) < 120:
        pid, depth = queue.pop(0)
        if depth >= lay:
            continue
        cur = polys[pid]
        for e in range(len(cur)):
            nxt = [_reflect(z, cur[e], cur[(e + 1) % len(cur)]) for z in cur]
            cen = [sum(v[0] for v in nxt) / len(nxt), sum(v[1] for v in nxt) / len(nxt)]
            if cen[0] ** 2 + cen[1] ** 2 <= 0.97 and not any(math.hypot(c[0] - cen[0], c[1] - cen[1]) < 1e-3 for c in centers):
                centers.append(cen)
                polys.append(nxt)
                queue.append((len(polys) - 1, depth + 1))
    pts, faces, eset = [], [], set()
    for poly in polys:
        fv = []
        for z in poly:
            hit = next((idx for idx, u in enumerate(pts) if math.hypot(u[0] - z[0], u[1] - z[1]) < 2e-4), None)
            if hit is None:
                hit = len(pts)
                pts.append(z)
            fv.append(hit)
        for k in range(len(fv)):
            u, v = fv[k], fv[(k + 1) % len(fv)]
            if u != v:
                eset.add((min(u, v), max(u, v)))
        faces.append(fv)
    kappa = float(orbifold_curvature([(pe, qe, 0, 0, 0)]))
    return {
        "p": pe,
        "q": qe,
        "style": style,
        "layers": lay,
        "wedge_delta": int(wedge_delta),
        "symbol_label": f"{{{pe},{qe}}}",
        "curvature_kappa": kappa,
        "geometry_kind": "Hyperbolic",
        "poincare_pts": pts,
        "klein_pts": [to_klein(z) for z in pts],
        "hyperboloid_pts": [to_hyperboloid(z) for z in pts],
        "edges": sorted(eset),
        "faces": faces,
    }
