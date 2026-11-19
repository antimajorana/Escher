"""Shared helpers of `papertilling.defect_fields` (cell / position / edge normalisation, Rust lookup)."""
from __future__ import annotations

import math

import numpy as np

from . import _rust

TWO_PI = 2.0 * math.pi
_IDENT_FRAME = {"q": [[1.0, 0.0], [0.0, 1.0]], "origin": [0.0, 0.0]}


def _rs(name: str):
    """The compiled module when it is enabled and exports `name` (an older build without the Stage 3 bindings falls
    back to the pure-Python code), else None."""
    rs = _rust.get()
    return rs if rs is not None and hasattr(rs, name) else None


def _cell_dict(lattice):
    """(cell dict in the Rust `Cell2D` JSON shape, group or None)."""
    if isinstance(lattice, dict) and "a1" in lattice:
        bonds = [b if isinstance(b, dict) else dict(i=int(b[0]), j=int(b[1]), n1=int(b[2]), n2=int(b[3]),
                                                     kind=int(b[4]) if len(b) > 4 else 0) for b in lattice["bonds"]]
        return dict(a1=[float(x) for x in lattice["a1"]], a2=[float(x) for x in lattice["a2"]],
                    basis=[[float(x) for x in p] for p in lattice["basis"]], bonds=bonds), lattice.get("group")
    from .defects.gauge import _to_lat
    c = _to_lat(lattice).cell
    return dict(a1=[float(x) for x in c.a1], a2=[float(x) for x in c.a2], basis=np.asarray(c.basis, float).tolist(),
                bonds=[dict(i=int(b[0]), j=int(b[1]), n1=int(b[2]), n2=int(b[3]), kind=int(b[4])) for b in c.bonds]), c.group


def _pos(p):
    return np.asarray(p, float).reshape(-1, 2)


def _edge_list(edges):
    return None if edges is None else [(int(u), int(v)) for u, v in np.asarray(edges, int).reshape(-1, 2)]
