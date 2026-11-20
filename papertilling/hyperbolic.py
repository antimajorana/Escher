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
from . import _hyperbolic_seeds as seeds
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
# `MAX_H2_LAYERS` there: layers are 0-based and only bound the loop; the face cap sets the drawn depth.
MAX_H2_LAYERS = 64
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


def _vertex_completion_patch(p: int, q: int, lay: int, seed: str = "cell"):
    """Drawn patch for wedge_delta == 0 (RegularPQ/DualQP): whole vertex-completion shells within the caps."""
    patch = seeds.grow_vertex_completion_seeded(p, q, seed, lay, 2 * MAX_DRAWN_FACES)
    keep, notes = vc.keep_whole_shells(patch.shell_sizes, patch.stopped, MAX_DRAWN_FACES)
    if keep == 0:
        raise ValueError(
            f"{{{p},{q}}}: the seed alone ({patch.shell_sizes[0]} tiles) exceeds the {MAX_DRAWN_FACES}-face drawing cap"
        )
    pset = vc.HypPointSet(vc.hyp_dist(patch.base.vertices[0], patch.base.vertices[1]))
    complete = keep - 1
    faces, face_shell = [], []
    for i, s in enumerate(patch.shell):
        if s >= keep:
            break
        if s > 0:
            z = vc.hyperboloid_to_poincare(patch.centre(i))
            if z[0] * z[0] + z[1] * z[1] > MAX_CENTRE_RADIUS**2:
                complete = min(complete, s - 1)
                continue
        faces.append([pset.insert(x)[0] for x in patch.face_vertices(i)])
        face_shell.append(s)
    if pset.suspect > 0:
        raise ValueError(f"{{{p},{q}}}: f64 precision: {pset.suspect} drawn points have an ambiguous dedup distance")
    if complete + 1 < keep:
        notes.append(f"shell {complete + 1} truncated by the centre radius cap {MAX_CENTRE_RADIUS}")
    growth_rule = (
        f"BDF vertex completion from {seeds.seed_label(p, q, seed)}: S_0..S_{complete} complete, {len(faces)} faces drawn "
        f"(requested S_{lay})" + "".join(f"; {n}" for n in notes)
    )
    pts = [vc.hyperboloid_to_poincare(x) for x in pset.pts]
    return pts, faces, face_shell, complete, growth_rule


def _kagome_cluster_patch(p: int, q: int, lay: int):
    """Drawn patch for wedge_delta == 0, RectifiedKagome: cluster rule on (p.q.p.q) including q-gons."""
    pts, faces, face_shell, complete, notes = vc.grow_kagome_cluster(
        p, q, lay, MAX_DRAWN_FACES, MAX_CENTRE_RADIUS
    )
    suffix = "".join(f"; {n}" for n in notes)
    if p == q:
        growth_rule = (
            f"cluster rule on ({p}.{p}.{p}.{p}) kagome (all {p}-gons are centres): "
            f"S_0..S_{complete} complete, {len(faces)} faces drawn (requested S_{lay}){suffix}"
        )
    else:
        c_sides = max(p, q)
        growth_rule = (
            f"cluster rule on ({p}.{q}.{p}.{q}) kagome (centres: {c_sides}-gons; {p}-gons and {q}-gons): "
            f"S_0..S_{complete} complete, {len(faces)} faces drawn (requested S_{lay}){suffix}"
        )
    return pts, faces, face_shell, complete, growth_rule


def _rust_e(x: float) -> str:
    """Rust's `{:.1e}` format (e.g. 5.0e-4), for growth_rule texts identical to the Rust engine."""
    mant, ex = f"{x:.1e}".split("e")
    return f"{mant}e{int(ex)}"


def _wedge_patch(p: int, q: int, style: str, lay: int, p_core: int):
    """Drawn patch for wedge_delta != 0 (Rust mirror): vertex completion of wedge-core copies.

    The core is a regular p_core-gon matching {p, q} edge length: cosh(l/2) = cos(pi/p)/sin(pi/q),
    sinh(R_core) = sinh(l/2)/sin(pi/p_core); dedup is hyperbolic.
    """
    half_edge_pq = math.acosh(max(1.0, math.cos(math.pi / p) / math.sin(math.pi / q)))
    rc_hyp = math.asinh(math.sinh(half_edge_pq) / math.sin(math.pi / p_core))
    poly = vc.Polygon(p_core, rc_hyp)
    faces3, shell, complete, notes, merged, kept = vc.grow_polygon_vertex_completion(
        poly, q, lay, MAX_DRAWN_FACES, MAX_CENTRE_RADIUS, WEDGE_MERGE_DIST
    )
    pts, faces = vc.polygon_faces_to_poincare(poly, faces3, style == "RectifiedKagome", WEDGE_MERGE_DIST)
    notes = list(notes)
    if merged > 0:
        notes.append(f"{merged} near-coincident copies (centres {_rust_e(0.5 * WEDGE_MERGE_DIST)}..{_rust_e(WEDGE_MERGE_DIST)} apart) merged")
    if kept > 0:
        notes.append(f"{kept} near-coincident copies (centres {_rust_e(WEDGE_MERGE_DIST)}..{_rust_e(1.5 * WEDGE_MERGE_DIST)} apart) kept")
    if q == 3:
        walk = "= edge-BFS for q = 3"
    else:
        steps = (q - 1) // 2
        unit = "step" if steps == 1 else "steps"
        opp = " + centred opposite copy" if q % 2 == 0 else ""
        walk = f"star walks of {steps} {unit} each side{opp}"
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
    seed: str = "cell",
) -> dict[str, Any]:
    """Generate a hyperbolic {p, q} / rectified (p.q.p.q) / dual {q, p} tiling patch in D and H^2.

    For wedge_delta == 0 the patch is BDF vertex completion S_layers of one p-gon (or the cluster
    rule on (p.q.p.q) kagome; layers 0-based, at most MAX_H2_LAYERS), trimmed to whole shells within MAX_DRAWN_FACES
    faces and to face centres within Poincare radius MAX_CENTRE_RADIUS; `face_shell`,
    `complete_through_shell` and `growth_rule` say which. For wedge_delta != 0 it is vertex
    completion of copies of a p_core-gon wedge core, with the same caps. `seed` ("cell", "vertex" or
    "edge"; C1 WP-G3) applies to the regular and dual paths only; kagome and wedge refuse other seeds.
    """
    seed = seeds.norm_seed(seed)
    rs = _rust.get()
    args = (int(p), int(q), style, int(layers), int(wedge_delta))
    # An extension built before C1 WP-G3 has only the tile-seed function; other seeds then use the Python path.
    if rs is not None and hasattr(rs, "hyperbolic_tiling_seeded_json"):
        return json.loads(rs.hyperbolic_tiling_seeded_json(*args, seed))
    if rs is not None and hasattr(rs, "hyperbolic_tiling_json") and seed == "cell":
        return json.loads(rs.hyperbolic_tiling_json(*args))
    one = seeds.one_tile(seed)
    st = style if style in _STYLES else "RegularPQ"
    p_in, q_in, wd = int(p), int(q), int(wedge_delta)
    pe, qe = (q_in, p_in) if st == "DualQP" else (p_in, q_in)
    if not vc.is_hyperbolic(pe, qe):
        raise ValueError(f"{{{pe}, {qe}}} is not hyperbolic (requires (p-2)(q-2) > 4)")
    lay = max(0, min(MAX_H2_LAYERS, int(layers)))
    if not one and (wd != 0 or st == "RectifiedKagome"):
        raise ValueError(
            f"{seed.capitalize()} seed: only the regular and dual {{p,q}} paths take vertex or edge seeds; the kagome "
            "cluster rule and the wedge core grow from their own core (use the tile seed)"
        )
    p_core = max(3, min(12, pe + wd))
    if wd == 0 and st == "RectifiedKagome":
        pts, faces, face_shell, complete, growth_rule = _kagome_cluster_patch(pe, qe, lay)
    elif wd == 0:
        pts, faces, face_shell, complete, growth_rule = _vertex_completion_patch(pe, qe, lay, seed)
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
        "seed": "cell" if one else seed,
    }
