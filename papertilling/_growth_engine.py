"""The touching-rule grower and its error type (mirrors `growth/engine.rs`, `growth/stats.rs`).

Public names are re-exported by `papertilling.growth`, whose docstring states the rules.
"""
from __future__ import annotations

from typing import Any


class GrowthError(Exception):
    """An incidence query failed: kind is 'infinite', 'precision' or 'out_of_range'."""

    def __init__(self, kind: str, reason: str = "", gap: float | None = None):
        super().__init__(f"{kind}: {reason or gap}")
        self.kind, self.reason, self.gap = kind, reason, gap

    def stopped_at(self, layer: int) -> dict:
        if self.kind == "infinite":
            return {"kind": "infinite", "layer": layer, "reason": self.reason}
        if self.kind == "precision":
            return {"kind": "precision", "layer": layer, "gap": self.gap}
        return {"kind": "rim", "layer": layer}


def grow(x, seed, max_layers: int, max_cells: int = 50_000) -> dict:
    """The touching-rule grower; mirrors `growth::engine::grow` step for step."""
    seed = sorted(set(seed))
    g: dict[str, Any] = {"layers": [], "stats": [], "stopped": None}
    if not seed:
        return g
    if len(seed) > max_cells:
        g["stopped"] = {"kind": "cell_cap", "layer": 0, "partial": len(seed)}
        return g
    in_s = set(seed)
    verts, edges, faces = set(), set(), set()
    g["layers"].append(seed)
    k = 0
    while True:
        fresh_v, fresh_e = set(), set()
        for c in g["layers"][k]:
            fresh_v.update(v for v in x.vertices_of(c) if v not in verts)
            fresh_e.update(e for e in x.edges_of(c) if e not in edges)
            faces.update(x.faces_of(c))
        verts |= fresh_v
        edges |= fresh_e
        fresh_v, fresh_e = sorted(fresh_v), sorted(fresh_e)
        err, rim, stars, bv, be = None, False, [], 0, 0
        for v in fresh_v:
            rim = rim or x.on_rim(v)
            try:
                cs = x.cells_at(v)
                stars.append(cs)
                bv += any(c not in in_s for c in cs)
            except GrowthError as e:
                bv += 1
                err = err or e
        for e in fresh_e:
            try:
                be += any(c not in in_s for c in x.cells_on_edge(e))
            except GrowthError as e2:
                be += 1
                err = err or e2
        n = len(in_s)
        euler = len(verts) - len(edges) + (len(faces) - n if faces else n)
        g["stats"].append({"cells": n, "new_cells": len(g["layers"][k]), "boundary_vertices": bv,
                           "boundary_edges": be, "euler": euler})
        if k >= max_layers:
            break
        if rim:
            g["stopped"] = {"kind": "rim", "layer": k + 1}
            break
        if err is not None:
            g["stopped"] = err.stopped_at(k + 1)
            break
        nxt: list = []
        seen: set = set()
        capped = False
        for cs in stars:
            for c in cs:
                if c not in in_s and c not in seen:
                    seen.add(c)
                    nxt.append(c)
                    if n + len(nxt) > max_cells:
                        capped = True
                        break
            if capped:
                break
        if capped:
            g["stopped"] = {"kind": "cell_cap", "layer": k + 1, "partial": n + len(nxt)}
            break
        if not nxt:
            g["stopped"] = {"kind": "closed", "layer": k}
            break
        nxt.sort()
        in_s.update(nxt)
        g["layers"].append(nxt)
        k += 1
    return g
