"""The four E3 honeycombs as exact integer complexes (mirrors `growth/e3_complex.rs`).

Coordinates are integers at scale E3_SCALE = 2 (real = integer / 2), the geometry of
`honeycomb3d.build_euclidean_honeycomb`: cubic cells at integer points; Kelvin cells at the BCC
points (all even or all odd); rhombic dodecahedra at the FCC points (even coordinate sum); octet
vertices at the odd-sum integers, octahedra at the even-sum integers, tetrahedra at the
half-integer points. A cell is its scaled centre; an edge or face is keyed by (vertex count,
sum of its scaled vertices), which is exact and unique because distinct faces of a face-to-face
convex tiling have distinct centroids.
"""
from __future__ import annotations

import itertools
import json

from . import _rust
from .growth import GrowthError, grow

E3_SCALE = 2
KINDS = ("cubic", "bitruncated_cubic", "rhombic_dodecahedral", "alternated_cubic")
_SIGNS = list(itertools.product((-1, 1), repeat=3))


def _perms_of(abs_sorted):
    out = set()
    for p in itertools.product(range(-2, 3), repeat=3):
        if sorted(abs(t) for t in p) == list(abs_sorted):
            out.add(p)
    return sorted(out)


def _class(offsets, normals):
    edges, faces = [], []
    d2 = lambda a, b: sum((s - t) ** 2 for s, t in zip(a, b))
    m = min(d2(a, b) for a, b in itertools.combinations(offsets, 2))
    edges = [(i, j) for i, j in itertools.combinations(range(len(offsets)), 2) if d2(offsets[i], offsets[j]) == m]
    for n in normals:
        dots = [sum(s * t for s, t in zip(n, o)) for o in offsets]
        top = max(dots)
        faces.append([i for i, d in enumerate(dots) if d == top])
    esum = {_add(offsets[i], offsets[j]) for i, j in edges}
    fsum = {_face_key([offsets[i] for i in f]) for f in faces}
    return {"offsets": offsets, "edges": edges, "faces": faces, "esum2": {(2, t) for t in esum}, "fsum": fsum}


def _add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


class E3Complex:
    """Cells, vertices and keys of one E3 honeycomb kind, all exact integers."""

    def __init__(self, kind: str):
        if kind not in KINDS:
            raise ValueError(f"unknown Euclidean honeycomb {kind!r} ({', '.join(KINDS)})")
        self.kind = kind
        corners = list(_SIGNS)
        axes = _perms_of([0, 0, 1])
        if kind == "cubic":
            self.classes = [_class(corners, axes)]
        elif kind == "bitruncated_cubic":
            self.classes = [_class(_perms_of([0, 1, 2]), axes + corners)]
        elif kind == "rhombic_dodecahedral":
            self.classes = [_class(_perms_of([0, 0, 2]) + corners, _perms_of([0, 1, 1]))]
        else:
            tet_a = [o for o in corners if sum(o) % 4 == 1]
            tet_b = [o for o in corners if sum(o) % 4 == 3]
            self.classes = [_class(_perms_of([0, 0, 2]), corners),
                            _class(tet_a, [tuple(-t for t in o) for o in tet_a]),
                            _class(tet_b, [tuple(-t for t in o) for o in tet_b])]

    def class_of(self, c):
        ev = all(t % 2 == 0 for t in c)
        if self.kind == "cubic":
            return 0 if ev else None
        if self.kind == "bitruncated_cubic":
            return 0 if len({t % 4 for t in c}) == 1 and c[0] % 2 == 0 else None
        if ev:
            return 0 if sum(c) % 4 == 0 else None
        if self.kind == "alternated_cubic" and all(t % 2 for t in c):
            return 1 if sum(c) % 4 == 1 else 2
        return None

    def is_vertex(self, v) -> bool:
        odd = [t % 2 for t in v]
        if self.kind == "cubic":
            return all(odd)
        if self.kind == "bitruncated_cubic":
            ev = [t % 4 for t in v if t % 2 == 0]
            return sum(odd) == 1 and ev[0] != ev[1]
        even2 = not any(odd) and sum(v) % 4 == 2
        return even2 or (self.kind == "rhombic_dodecahedral" and all(odd))

    def _verts(self, c):
        ci = self.class_of(c)
        return [] if ci is None else [_add(c, o) for o in self.classes[ci]["offsets"]]

    def vertices_of(self, c): return self._verts(c)

    def edges_of(self, c):
        ci, vs = self.class_of(c), self._verts(c)
        return [] if ci is None else [(2, _add(vs[i], vs[j])) for i, j in self.classes[ci]["edges"]]

    def faces_of(self, c):
        ci, vs = self.class_of(c), self._verts(c)
        return [] if ci is None else [_face_key([vs[i] for i in f]) for f in self.classes[ci]["faces"]]

    def on_rim(self, v): return False

    def cells_at(self, v):
        v = tuple(v)
        if not self.is_vertex(v):
            raise GrowthError("out_of_range", "not a vertex of this honeycomb")
        out = set()
        for ci, cl in enumerate(self.classes):
            for o in cl["offsets"]:
                c = _sub(v, o)
                if self.class_of(c) == ci:
                    out.add(c)
        return sorted(out)

    def _cells_with(self, n, s, table):
        """Cells c with (n, s - n*c) among their class's relative edge or face keys."""
        out = []
        for ci, cl in enumerate(self.classes):
            for m, t in cl[table]:
                d = _sub(s, t)
                if m == n and all(q % n == 0 for q in d):
                    c = tuple(q // n for q in d)
                    if self.class_of(c) == ci:
                        out.append(c)
        return sorted(set(out))

    def cells_on_edge(self, e):
        if e[0] != 2:
            raise GrowthError("out_of_range", "not an edge key")
        return self._cells_with(2, e[1], "esum2")

    def cells_on_face(self, f):
        if len(f) < 3:
            raise GrowthError("out_of_range", "a face needs at least 3 vertices")
        n, s = _face_key([tuple(v) for v in f])
        return self._cells_with(n, s, "fsum")

    def default_seed(self, seed: str) -> list:
        """The seed cells for a seed kind; representatives are fixed per kind (see e3_complex.rs)."""
        v, e, f = _SEEDS[self.kind]
        if seed == "vertex":
            return self.cells_at(v)
        if seed == "edge":
            return self.cells_on_edge((2, _add(*e)))
        if seed == "face":
            return self.cells_on_face(f)
        if seed == "cell":
            return [(0, 0, 0)]
        if seed == "dirichlet":
            raise ValueError("Dirichlet seed is not offered for E3 honeycombs: the Dirichlet cell of a lattice "
                             "point is the cell itself (use cell)")
        raise ValueError(f"unknown seed kind {seed!r} (vertex, edge, face, cell)")


def _face_key(vs):
    s = (sum(v[0] for v in vs), sum(v[1] for v in vs), sum(v[2] for v in vs))
    return (len(vs), s)


# (vertex, edge endpoints, face vertices), scaled; the most symmetric choices (e3_complex.rs docs).
_SEEDS = {
    "cubic": ((1, 1, 1), ((1, 1, 1), (1, 1, -1)), [(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]),
    "bitruncated_cubic": ((0, 1, 2), ((1, 2, 0), (2, 1, 0)), [(2, 0, 1), (2, 0, -1), (2, 1, 0), (2, -1, 0)]),
    "rhombic_dodecahedral": ((2, 0, 0), ((2, 0, 0), (1, 1, 1)), [(2, 0, 0), (0, 2, 0), (1, 1, 1), (1, 1, -1)]),
    "alternated_cubic": ((2, 0, 0), ((2, 0, 0), (0, 2, 0)), [(2, 0, 0), (0, 2, 0), (0, 0, 2)]),
}


def _to_json_ready(g: dict) -> dict:
    g["layers"] = [[list(c) for c in layer] for layer in g["layers"]]
    return g


def grow_e3(kind: str, seed: str = "vertex", layers: int = 2, max_cells: int = 50_000) -> dict:
    """Grow an E3 honeycomb from a seed kind; same JSON as `papertilling_rs.grow_e3_json`."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "grow_e3_json"):
        try:
            return json.loads(rs.grow_e3_json(kind, seed, layers, max_cells))
        except NotImplementedError:
            pass
    x = E3Complex(kind)
    g = _to_json_ready(grow(x, x.default_seed(seed), layers, max_cells))
    return {"kind": kind, "seed": seed, "scale": E3_SCALE, **g}
