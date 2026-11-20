"""Seeded layer growth on a cell complex (C1 track T2; mirrors `papertilling-core/src/growth`).

    from papertilling import growth
    growth.grow_e3("cubic", "cell", layers=2)["stats"][2]["cells"]          # 125 = 5^3
    growth.grow_finite({"polygons": [[0, 1, 2, 3]]}, {"kind": "cell", "at": 0}, 1)["stopped"]

Touching rule (BDF vertex completion): S_{k+1} = union of cells_at(v) over the vertices v of S_k;
layer 0 is the seed. A layer is built whole or not at all, and `stopped` says why growth ended:
`cell_cap` (partial = max_cells + 1, the count at which the build stopped), `infinite`,
`precision`, `closed` (S_{k+1} = S_k) or `rim` (a vertex on the rim of a finite table).
Boundary vertices / edges of S_k are those whose cells are not all in S_k (the growth frontier).
Euler characteristic: V - E + F in 2D (F = tiles), V - E + F - C in 3D.

The compiled `papertilling_rs` is used when present (unless `PAPERTILLING_NO_RUST=1`); the pure-Python
code below gives the same JSON (layers sorted, cells as lists).
"""
from __future__ import annotations

import json

from . import _rust
from ._growth_engine import GrowthError, grow

__all__ = ["GrowthError", "grow", "FiniteComplex", "E3Complex", "grow_finite", "grow_e3", "E3_SCALE"]


# ---------------------------------------------------------------- finite tables (mirrors finite.rs)
class FiniteComplex:
    """A complex from explicit tables (cells, edges with endpoints, optional faces, rim vertices)."""

    def __init__(self, cell_vertices, cell_edges, edge_vertices, cell_faces=None, face_vertices=None, rim=()):
        self.cv = [list(c) for c in cell_vertices]
        self.ce = [list(c) for c in cell_edges]
        self.ev = [tuple(e) for e in edge_vertices]
        self.cf = [list(c) for c in (cell_faces or [[] for _ in self.cv])]
        self.fv = [list(f) for f in (face_vertices or [])]
        self.rim = set(rim)
        if not (len(self.cv) == len(self.ce) == len(self.cf)):
            raise ValueError("cell_vertices, cell_edges and cell_faces need one row per cell")
        self.vc: dict[int, list[int]] = {}
        self.ec = [[] for _ in self.ev]
        self.fc = [[] for _ in self.fv]
        for c, vs in enumerate(self.cv):
            for v in vs:
                self.vc.setdefault(v, []).append(c)
            for e in self.ce[c]:
                if e >= len(self.ev) or not set(self.ev[e]) <= set(vs):
                    raise ValueError(f"edge {e} of cell {c} is not an edge between vertices of that cell")
                self.ec[e].append(c)
            for f in self.cf[c]:
                if f >= len(self.fv) or not set(self.fv[f]) <= set(vs):
                    raise ValueError(f"face {f} of cell {c} is not a face on vertices of that cell")
                self.fc[f].append(c)
        if any(v not in self.vc for v in self.rim):
            raise ValueError("a rim vertex is in no cell")
        self.face_index = {tuple(sorted(f)): i for i, f in enumerate(self.fv)}

    @classmethod
    def from_polygons(cls, polys, rim=()):
        return cls._from_cycles([[p] for p in polys], rim, False)

    @classmethod
    def from_polyhedra(cls, cells, rim=()):
        return cls._from_cycles(cells, rim, True)

    @classmethod
    def _from_cycles(cls, cells, rim, three_d):
        eid: dict = {}
        fid: dict = {}
        cv, ce, cf, fv = [], [], [], []
        for cell in cells:
            vs, es, fs = [], [], []
            for cyc in cell:
                if len(cyc) < 3 or len(set(cyc)) != len(cyc):
                    raise ValueError(f"face cycle {cyc} needs at least 3 distinct vertices")
                vs.extend(v for v in cyc if v not in vs)
                for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                    e = eid.setdefault((min(a, b), max(a, b)), len(eid))
                    if e not in es:
                        es.append(e)
                if three_d:
                    f = fid.setdefault(tuple(sorted(cyc)), len(fid))
                    if f == len(fv):
                        fv.append(list(cyc))
                    fs.append(f)
            cv.append(vs); ce.append(es); cf.append(fs)
        ev = [None] * len(eid)
        for pair, i in eid.items():
            ev[i] = pair
        return cls(cv, ce, ev, cf, fv, rim)

    @classmethod
    def from_dict(cls, d: dict):
        if "polygons" in d:
            return cls.from_polygons(d["polygons"], d.get("rim", ()))
        if "polyhedra" in d:
            return cls.from_polyhedra(d["polyhedra"], d.get("rim", ()))
        return cls(d["cell_vertices"], d["cell_edges"], d["edge_vertices"], d.get("cell_faces"),
                   d.get("face_vertices"), d.get("rim", ()))

    def vertices_of(self, c): return self.cv[c]
    def edges_of(self, c): return self.ce[c]
    def faces_of(self, c): return self.cf[c]
    def on_rim(self, v): return v in self.rim

    def cells_at(self, v):
        if v not in self.vc:
            raise GrowthError("out_of_range", "vertex not in the table")
        return list(self.vc[v])

    def cells_on_edge(self, e):
        if not 0 <= e < len(self.ec):
            raise GrowthError("out_of_range", "edge not in the table")
        return list(self.ec[e])

    def cells_on_face(self, f):
        i = self.face_index.get(tuple(sorted(f)))
        if i is None:
            raise GrowthError("out_of_range", "face not in the table")
        return list(self.fc[i])

    def seed_cells(self, seed: dict) -> list:
        kind, at = seed.get("kind"), seed.get("at")
        if kind == "vertex":
            return self.cells_at(at)
        if kind == "cell":
            if not (isinstance(at, int) and 0 <= at < len(self.cv)):
                raise ValueError(f"cell {at} is not in the table")
            return [at]
        if kind == "edge":
            if isinstance(at, int):
                return self.cells_on_edge(at)
            hits = [i for i, p in enumerate(self.ev) if sorted(p) == sorted(at)]
            if len(hits) != 1:
                raise ValueError(f"vertex pair {at} names {len(hits)} edges; give the edge id")
            return self.cells_on_edge(hits[0])
        if kind == "face":
            return self.cells_on_face(at)
        if kind == "dirichlet":
            raise ValueError("a Dirichlet seed needs coordinates; a finite table has none (use vertex, edge, face or cell)")
        raise ValueError(f"unknown seed kind {kind!r}")


def grow_finite(complex_: dict, seed: dict, max_layers: int, max_cells: int = 50_000) -> dict:
    """Grow on a finite table; same JSON as `papertilling_rs.grow_finite_json`."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "grow_finite_json"):
        try:
            return json.loads(rs.grow_finite_json(json.dumps(complex_), json.dumps(seed), max_layers, max_cells))
        except NotImplementedError:
            pass
    x = FiniteComplex.from_dict(complex_)
    try:
        cells = x.seed_cells(seed)
    except GrowthError as e:
        raise ValueError(f"seed: {e}") from None
    return grow(x, cells, max_layers, max_cells)


from ._growth_e3 import E3_SCALE, E3Complex, grow_e3  # noqa: E402  (re-export)
