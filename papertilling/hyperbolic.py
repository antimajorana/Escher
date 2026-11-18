"""Hyperbolic H^2 (p, q) tilings, rectified (p.q.p.q) kagome, Poincare geodesics, Klein chords, and 3D hyperboloid.

`build_hyperbolic_tiling` returns the Rust engine's JSON (`papertilling_rs.hyperbolic_tiling_json`)
when the extension is available, else the pure-Python fallback below, which mirrors the Rust drawn
patch (`rust/papertilling-core/src/hyperbolic/mod.rs`): BDF vertex completion S_0 .. S_layers of one
p-gon, whole shells within the face cap, hyperbolic-distance dedup, same output keys; for
wedge_delta != 0, vertex completion of copies of the wedge core (`hyperbolic/wedge.rs`).
"""
from __future__ import annotations

import json
import math
from typing import Any

from . import _rust
from . import _hyperbolic_vc as vc
from .dsymbol import orbifold_curvature

__all__ = ["build_hyperbolic_tiling", "to_klein", "to_hyperboloid"]

# Mirrored from rust/papertilling-core/src/hyperbolic/mod.rs: `DRAW_CAPS` (face cap and Poincare-radius
# cap of face centres of the drawn patch) and `WEDGE_MERGE_DIST` (hyperbolic merge distance of the wedge
# path). The vertex-completion compute budget is 2 * MAX_DRAWN_FACES there (next-shell estimates are
# upper bounds, so it never stops a shell that would fit).
MAX_DRAWN_FACES = 1000
MAX_CENTRE_RADIUS = 0.998
WEDGE_MERGE_DIST = 1e-3
_STYLES = ("RegularPQ", "RectifiedKagome", "DualQP")


def to_klein(z: list[float] | tuple[float, float]) -> list[float]:
    d = 1.0 + z[0] * z[0] + z[1] * z[1]
    return [2.0 * z[0] / d, 2.0 * z[1] / d]


def to_hyperboloid(z: list[float] | tuple[float, float]) -> list[float]:
    """Upper-sheet hyperboloid image of a Poincare point, radius clamped to 0.95 for display."""
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


def _geodesic_arc(a, b, segs: int = 8):
    """Points along the Poincare geodesic from a to b (the chord ends for a diameter)."""
    oc = _orth_circle(a, b)
    if oc is None:
        return [list(a), list(b)]
    cx, cy, r2 = oc
    rad = math.sqrt(r2)
    th0, th1 = math.atan2(a[1] - cy, a[0] - cx), math.atan2(b[1] - cy, b[0] - cx)
    dth = (th1 - th0 + math.pi) % (2.0 * math.pi) - math.pi
    if abs(dth) < 1e-6:
        dth = 0.0
    return [[cx + rad * math.cos(th0 + dth * k / segs), cy + rad * math.sin(th0 + dth * k / segs)] for k in range(segs + 1)]


def _vertex_completion_patch(p: int, q: int, style: str, lay: int):
    """Drawn patch for wedge_delta == 0: whole vertex-completion shells within the caps (Rust mirror)."""
    patch = vc.grow_vertex_completion(p, q, lay, 2 * MAX_DRAWN_FACES)
    keep, total = 0, 0
    for a in patch.shell_sizes:
        if total + a > MAX_DRAWN_FACES:
            break
        total += a
        keep += 1
    notes = []
    if keep < len(patch.shell_sizes):
        notes.append(f"shell {keep} ({patch.shell_sizes[keep]} faces) omitted by the {MAX_DRAWN_FACES}-face cap")
    stopped = patch.stopped
    if stopped is not None and stopped.startswith("face budget"):
        # The compute budget (twice the cap) only stops shells that cannot fit the cap (Rust mirror).
        if keep >= len(patch.shell_sizes):
            notes.append(f"shell {len(patch.shell_sizes)} omitted by the {MAX_DRAWN_FACES}-face cap (estimated larger)")
    elif stopped is not None:
        notes.append(stopped)
    kagome = style == "RectifiedKagome"
    base_pts = patch.base.edge_midpoints if kagome else patch.base.vertices
    pset = vc.HypPointSet(vc.hyp_dist(base_pts[0], base_pts[1]))
    complete = keep - 1
    faces, face_shell = [], []
    for i, s in enumerate(patch.shell):
        if s >= keep:
            break
        if i > 0:
            z = vc.hyperboloid_to_poincare(patch.centre(i))
            if z[0] * z[0] + z[1] * z[1] > MAX_CENTRE_RADIUS**2:
                complete = min(complete, s - 1)
                continue
        pts3 = patch.face_edge_midpoints(i) if kagome else patch.face_vertices(i)
        faces.append([pset.insert(x)[0] for x in pts3])
        face_shell.append(s)
    if pset.suspect > 0:
        raise ValueError(f"{{{p},{q}}}: f64 precision: {pset.suspect} drawn points have an ambiguous dedup distance")
    if complete + 1 < keep:
        notes.append(f"shell {complete + 1} truncated by the centre radius cap {MAX_CENTRE_RADIUS}")
    # Kagome: the (p.q.p.q) p-gons of the {p, q} vertex-completion patch; the q-gons are the gaps.
    what = "kagome on the p-gons of BDF vertex completion" if kagome else "BDF vertex completion"
    growth_rule = (
        f"{what} from one {p}-gon: S_0..S_{complete} complete, {len(faces)} faces drawn "
        f"(requested S_{lay})" + "".join(f"; {n}" for n in notes)
    )
    pts = [vc.hyperboloid_to_poincare(x) for x in pset.pts]
    return pts, faces, face_shell, complete, growth_rule


def _rust_e(x: float) -> str:
    """Rust's `{:.1e}` format (e.g. 5.0e-4), for growth_rule texts identical to the Rust engine."""
    mant, ex = f"{x:.1e}".split("e")
    return f"{mant}e{int(ex)}"


def _wedge_patch(p: int, q: int, style: str, lay: int, p_core: int):
    """Drawn patch for wedge_delta != 0 (Rust mirror): vertex completion of wedge-core copies.

    The core is a regular p_core-gon of circumradius asinh(sinh(r) / sin(pi / p_core)), r the inradius
    of {p, q}, capped at Poincare radius 0.96 (the original core geometry); dedup is hyperbolic.
    """
    inradius_pq = math.acosh(max(1.0, math.cos(math.pi / q) / math.sin(math.pi / p)))
    rc_hyp = math.asinh(math.sinh(inradius_pq) / math.sin(math.pi / p_core))
    r_core = min(math.tanh(0.5 * rc_hyp), 0.96)
    poly = vc.Polygon(p_core, 2.0 * math.atanh(r_core))
    faces3, shell, complete, notes, suspect = vc.grow_polygon_vertex_completion(
        poly, q, lay, MAX_DRAWN_FACES, MAX_CENTRE_RADIUS, WEDGE_MERGE_DIST
    )
    pts, faces = vc.polygon_faces_to_poincare(poly, faces3, style == "RectifiedKagome", WEDGE_MERGE_DIST)
    if suspect > 0:
        # Copies of a non-closing polygon can nearly coincide; they are kept, and counted here.
        lo, hi = _rust_e(0.5 * WEDGE_MERGE_DIST), _rust_e(1.5 * WEDGE_MERGE_DIST)
        notes = notes + [f"{suspect} near-coincident copies (centres {lo}..{hi} apart) kept"]
    walk = "= edge-BFS for q = 3" if q == 3 else f"star walks of {q // 2} steps each side"
    growth_rule = (
        f"vertex completion of {p_core}-gon wedge-core copies ({walk}): S_0..S_{complete} complete, "
        f"{len(faces)} faces drawn (requested S_{lay})" + "".join(f"; {n}" for n in notes)
    )
    return pts, faces, shell, complete, growth_rule


def build_hyperbolic_tiling(
    p: int = 7,
    q: int = 3,
    style: str = "RegularPQ",
    layers: int = 2,
    wedge_delta: int = 0,
) -> dict[str, Any]:
    """Generate a hyperbolic {p, q} / rectified (p.q.p.q) / dual {q, p} tiling patch in D and H^2.

    For wedge_delta == 0 the patch is BDF vertex completion S_layers of one p-gon (layers clamped to
    1..4), trimmed to whole shells within MAX_DRAWN_FACES faces and to face centres within Poincare
    radius MAX_CENTRE_RADIUS; `face_shell`, `complete_through_shell` and `growth_rule` say which. For
    wedge_delta != 0 it is vertex completion of copies of a p_core-gon wedge core, with the same caps.
    """
    rs = _rust.get()
    if rs is not None and hasattr(rs, "hyperbolic_tiling_json"):
        return json.loads(rs.hyperbolic_tiling_json(int(p), int(q), style, int(layers), int(wedge_delta)))
    st = style if style in _STYLES else "RegularPQ"
    p_in, q_in, wd = int(p), int(q), int(wedge_delta)
    pe, qe = (q_in, p_in) if st == "DualQP" else (p_in, q_in)
    if not vc.is_hyperbolic(pe, qe):
        raise ValueError(f"{{{pe}, {qe}}} is not hyperbolic (requires (p-2)(q-2) > 4)")
    lay = max(1, min(4, int(layers)))
    p_core = max(3, min(12, pe + wd))
    if wd == 0:
        pts, faces, face_shell, complete, growth_rule = _vertex_completion_patch(pe, qe, st, lay)
    else:
        pts, faces, face_shell, complete, growth_rule = _wedge_patch(pe, qe, st, lay, p_core)
    eset = set()
    for fv in faces:
        for k in range(len(fv)):
            u, v = fv[k], fv[(k + 1) % len(fv)]
            if u != v:
                eset.add((min(u, v), max(u, v)))
    edges = sorted(eset)
    labels = {
        "RegularPQ": f"{{{p_in},{q_in}}}",
        "RectifiedKagome": f"({p_in}.{q_in}.{p_in}.{q_in}) kagome",
        "DualQP": f"{{{q_in},{p_in}}} dual",
    }
    return {
        "p": pe,
        "q": qe,
        "style": st,
        "layers": lay,
        "wedge_delta": wd,
        "symbol_label": labels[st],
        "curvature_kappa": float(orbifold_curvature([(pe, qe, 0, 0, 0)])),
        "geometry_kind": "Hyperbolic",
        "psu11_half_trace": 1.0 if wd == 0 else abs(math.cos(math.pi * abs(wd) / p_core)),
        "poincare_pts": pts,
        "klein_pts": [to_klein(z) for z in pts],
        "hyperboloid_pts": [to_hyperboloid(z) for z in pts],
        "edges": [list(e) for e in edges],
        "edge_arcs": [_geodesic_arc(pts[u], pts[v], 8) for u, v in edges],
        "faces": faces,
        "face_shell": face_shell,
        "complete_through_shell": complete,
        "growth_rule": growth_rule,
    }
