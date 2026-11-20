"""Honeycomb growth wiring: rules, refusals and the drawn complex (mirrors `honeycomb3d/growth_build.rs`).

Public names are re-exported by `papertilling._honeycomb3d_growth` and `papertilling.honeycomb3d`.
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import _rust
from ._growth_e3 import E3_SCALE, E3Complex
from ._growth_engine import GrowthError, grow


def _parse(kind: str):
    t = kind.strip().lstrip("{").rstrip("}")
    parts = [x.strip() for x in t.split(",")]
    if len(parts) == 3:
        try:
            return tuple(int(x) for x in parts)
        except ValueError:
            raise ValueError(f"bad Schläfli symbol {kind!r}") from None
    from .honeycomb3d import EUCLIDEAN_KINDS
    if kind not in EUCLIDEAN_KINDS:
        raise ValueError(f"unknown Euclidean honeycomb {kind!r} (cubic, bitruncated_cubic, rhombic_dodecahedral, "
                         "alternated_cubic)")
    return kind


def face_shells(x, seed, max_layers, max_cells, on_face, star, on_edge):
    """The face-shell rule with the engine's readout and stops (mirrors `growth_build::face_shells`)."""
    seed = sorted(set(seed))
    g = {"layers": [], "stats": [], "stopped": None}
    if not seed:
        return g
    if len(seed) > max_cells:
        g["stopped"] = {"kind": "cell_cap", "layer": 0, "partial": len(seed)}
        return g
    in_s, verts, edges, faces = set(seed), set(), set(), set()
    open_v, open_e = [], []
    g["layers"].append(seed)
    k = 0
    while True:
        layer = g["layers"][k]
        fv, fe = set(), set()
        for c in layer:
            fv.update(v for v in x.vertices_of(c) if v not in verts)
            fe.update(e for e in x.edges_of(c) if e not in edges)
            faces.update(x.faces_of(c))
        verts |= fv
        edges |= fe
        err = None

        def still_open(q, a):
            nonlocal err
            try:
                return any(c not in in_s for c in q(a))
            except GrowthError as e:
                if e.kind != "infinite":
                    err = err or e
                return True
        open_v = [v for v in open_v + sorted(fv) if still_open(star, v)]
        open_e = [e for e in open_e + sorted(fe) if still_open(on_edge, e)]
        n = len(in_s)
        euler = len(verts) - len(edges) + (len(faces) - n if faces else n)
        g["stats"].append({"cells": n, "new_cells": len(layer), "boundary_vertices": len(open_v),
                           "boundary_edges": len(open_e), "euler": euler})
        if k >= max_layers:
            break
        nxt, capped = [], False
        for c in layer:
            for f in x.faces_of(c):
                try:
                    cs = on_face(f)
                except GrowthError as e:
                    err = err or e
                    continue
                for c2 in cs:
                    if c2 not in in_s:
                        in_s.add(c2)
                        nxt.append(c2)
                        if n + len(nxt) > max_cells:
                            capped = True
                            break
                if capped:
                    break
            if capped:
                break
        if err is not None:
            g["stopped"] = err.stopped_at(k + 1)
            break
        if capped:
            g["stopped"] = {"kind": "cell_cap", "layer": k + 1, "partial": n + len(nxt)}
            break
        if not nxt:
            g["stopped"] = {"kind": "closed", "layer": k}
            break
        g["layers"].append(sorted(nxt))
        k += 1
    return g


class _Builder:
    def __init__(self):
        self.faces, self.edges, self.fid, self.eid = [], [], {}, set()

    def add_face(self, cyc):
        key = tuple(sorted(cyc))
        if key in self.fid:
            return self.fid[key]
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            e = (min(a, b), max(a, b))
            if e not in self.eid:
                self.eid.add(e)
                self.edges.append(list(e))
        self.fid[key] = len(self.faces)
        self.faces.append(list(cyc))
        return self.fid[key]


def _draw(cells, faces_of, cycles_of, coords_of, name, regime):
    """Faces of S_K (only boundary faces above DRAW_ALL_FACES_MAX cells) as a complex dict."""
    from ._honeycomb3d_growth import DRAW_ALL_FACES_MAX
    boundary_only = len(cells) > DRAW_ALL_FACES_MAX
    lists = [faces_of(c) for c, _ in cells]
    count: dict = {}
    for fs in lists:
        for f in fs:
            count[f] = count.get(f, 0) + 1
    b, remap, coords = _Builder(), {}, []
    cx = {"name": name, "regime": regime, "cells": [], "cell_shell": [], "cell_truncated": []}
    for (c, layer), fs in zip(cells, lists):
        out = []
        for f, cyc in zip(fs, cycles_of(c, fs)):
            if boundary_only and count[f] != 1:
                continue
            ids = []
            for v in cyc:
                if v not in remap:
                    remap[v] = len(coords)
                    coords.append(coords_of(v))
                ids.append(remap[v])
            out.append(b.add_face(ids))
        if not boundary_only:
            cx["cells"].append(out)
            cx["cell_shell"].append(layer)
            cx["cell_truncated"].append(False)
    cx.update(edges=b.edges, faces=b.faces)
    return cx, coords, boundary_only


def _layers(g):
    return [(c, k) for k, layer in enumerate(g["layers"][:len(g["stats"])]) for c in layer]


def _draw_coxeter(x, g, name, regime):
    cx, coords, bo = _draw(_layers(g), x.faces_of, lambda c, fs: x.face_cycles_of(c), x.vertex_coords4, name, regime)
    c4 = np.array(coords) if coords else np.zeros((0, 4))
    cx["coords4"] = c4.tolist()
    if x.spherical:
        for w in (x.w[3], x.w[2], x.w[1], np.array([0.3, -0.5, 0.7, 0.4])):
            pole = -w / np.linalg.norm(w)
            if np.all(c4 @ pole < 1 - 1e-3):
                break
        basis = []
        for e in np.eye(4):
            v = e - (e @ pole) * pole - sum((e @ bb) * bb for bb in basis)
            if v @ v > 1e-6 and len(basis) < 3:
                basis.append(v / np.linalg.norm(v))
        cx["points"] = (c4 @ np.array(basis).T / (1 - c4 @ pole)[:, None]).tolist() if coords else []
        cx["klein"], cx["is_ideal"] = [], [False] * len(coords)
    else:
        s = c4[:, 1:]
        cx["points"] = (s if x.ideal else s / (1 + c4[:, :1])).tolist()
        cx["klein"] = (s / c4[:, :1]).tolist()
        cx["is_ideal"] = [x.ideal] * len(coords)
    return cx, bo


def _draw_e3(x, g, name):
    def cycles(c, fs):
        out = []
        verts = x.vertices_of(c)
        for n, ssum in fs:
            normal = [ssum[d] - n * c[d] for d in range(3)]
            on = [v for v in verts if sum((n * v[d] - ssum[d]) * normal[d] for d in range(3)) == 0]
            ctr = np.array(ssum, float) / n
            u = np.array(on[0], float) - ctr
            w = np.cross(np.array(normal, float), u)
            out.append(sorted(on, key=lambda v: math.atan2((np.array(v) - ctr) @ w, (np.array(v) - ctr) @ u)))
        return out
    cx, coords, bo = _draw(_layers(g), x.faces_of, cycles, lambda v: [t / E3_SCALE for t in v], name, "EuclideanE3")
    cx["points"] = coords
    cx["coords4"] = [[*p, 1.0] for p in coords]
    cx["klein"], cx["is_ideal"] = [], [False] * len(coords)
    return cx, bo


def _py_grow(kind, seed, layers, max_cells, rule):
    from ._honeycomb3d_growth import INFINITE_CELL_REASON, CoxeterComplex
    from .honeycomb3d import EUCLIDEAN_KINDS, _py_classify
    k = _parse(kind)
    if isinstance(k, tuple):
        cls = _py_classify(*k)
        reg, name = cls["regime"], "{%d,%d,%d}" % k
        if reg == "EuclideanE3":
            k = "cubic"
        elif reg not in ("SphericalS3", "HyperbolicCompactH3", "HyperbolicParacompactH3"):
            raise ValueError(cls["reason"])
    if isinstance(k, str):
        x = E3Complex(k)
        s0 = x.default_seed(seed)
        name, reg = EUCLIDEAN_KINDS[k][1], "EuclideanE3"
        if rule == "touching":
            g = grow(x, s0, layers, max_cells)
        else:
            g = face_shells(x, s0, layers, max_cells, lambda f: x._cells_with(f[0], f[1], "fsum"), x.cells_at,
                            x.cells_on_edge)
        cx, bo = _draw_e3(x, g, name)
        return {"name": name, "regime": reg, "seed": seed, "rule": rule, "stats": g["stats"],
                "stopped": g["stopped"], "precision": None, "boundary_only": bo, "complex": cx}
    out = {"name": name, "regime": reg, "seed": seed, "rule": rule, "stats": [], "stopped": None,
           "precision": None, "boundary_only": False,
           "complex": {"name": name, "regime": reg, "coords4": [], "points": [], "klein": [], "is_ideal": [],
                       "edges": [], "faces": [], "cells": [], "cell_shell": [], "cell_truncated": []}}
    if cls["has_infinite_euclidean_cells"]:
        out["stopped"] = {"kind": "infinite", "layer": 0, "reason": INFINITE_CELL_REASON}
        return out
    x = CoxeterComplex(*k)
    try:
        s0 = x.default_seed(seed)
    except GrowthError as e:
        out["stopped"] = e.stopped_at(0)
        return out
    if rule == "touching":
        x.step, x.querying, x.final_step, x.cap = 0, False, layers, max_cells
        g = grow(x, s0, layers, max_cells)
        x.final_step = x.cap = None
    else:
        g = face_shells(x, s0, layers, max_cells, x.cells_on_face_key, lambda v: x.lookup_query(x.cells_at, v),
                        lambda e: x.lookup_query(x.cells_on_edge, e))
    cx, bo = _draw_coxeter(x, g, name, reg)
    out.update(stats=g["stats"], stopped=g["stopped"], precision=x.precision(), boundary_only=bo, complex=cx)
    return out


def grow_honeycomb(kind: str, seed: str = "vertex", layers: int = 2, max_cells: int | None = None,
                   rule: str = "touching", draw: bool = True) -> dict:
    """Grow a honeycomb from a seed; same JSON as `papertilling_rs.grow_honeycomb_json`.

    `kind` is a Schläfli symbol ("{4,3,5}") or an E³ kind ("cubic", ...); `seed` vertex, edge, face
    or cell; `rule` "touching" or "face_shells"; `max_cells` defaults to MAX_GROWTH_CELLS (120 000)."""
    from ._honeycomb3d_growth import MAX_GROWTH_CELLS
    if rule not in ("touching", "face_shells"):
        raise ValueError(f"unknown rule {rule!r} (touching, face_shells)")
    if seed not in ("vertex", "edge", "face", "cell", "dirichlet"):
        raise ValueError(f"unknown seed kind {seed!r} (vertex, edge, face, cell)")
    cap = MAX_GROWTH_CELLS if max_cells is None else max_cells
    rs = _rust.get()
    if rs is not None and hasattr(rs, "grow_honeycomb_json"):
        try:
            return json.loads(rs.grow_honeycomb_json(kind, seed, layers, cap, rule, draw))
        except NotImplementedError:
            pass
    out = _py_grow(kind, seed, layers, cap, rule)
    if not draw:
        out.pop("complex")
    return out
