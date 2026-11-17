"""0D rosettes (C_n, D_n) and 1D frieze groups (all 7: 5 symmorphic + 2 nonsymmorphic)."""
from __future__ import annotations

import json
import math
from typing import Any

from . import _rust
from .bieberbach import fmt_abelian

__all__ = ["FRIEZE_SYMBOLS", "build_rosette", "analyze_frieze", "build_frieze_patch"]

FRIEZE_SYMBOLS = ["p1", "p11m", "p1m1", "p11g", "p2", "p2mm", "p2mg"]

_FRIEZE_META: dict[str, dict[str, Any]] = {
    "p1": {"number": 1, "iucr": "p 1 (f1)", "point_order": 1, "h2": [], "is_split": True, "ab_plus": ([], 1), "ab_full": ([], 1)},
    "p11m": {"number": 2, "iucr": "p 1 1 m (f2)", "point_order": 2, "h2": [2], "is_split": True, "ab_plus": ([], 1), "ab_full": ([2], 1)},
    "p11g": {"number": 3, "iucr": "p 1 1 g (f3)", "point_order": 2, "h2": [2], "is_split": False, "ab_plus": ([], 1), "ab_full": ([], 1)},
    "p1m1": {"number": 4, "iucr": "p 1 m 1 (f4)", "point_order": 2, "h2": [], "is_split": True, "ab_plus": ([], 1), "ab_full": ([2, 2], 0)},
    "p2": {"number": 5, "iucr": "p 2 1 1 (f5)", "point_order": 2, "h2": [], "is_split": True, "ab_plus": ([2, 2], 0), "ab_full": ([2, 2], 0)},
    "p2mm": {"number": 6, "iucr": "p 2 m m (f6)", "point_order": 4, "h2": [2], "is_split": True, "ab_plus": ([2, 2], 0), "ab_full": ([2, 2, 2], 0)},
    "p2mg": {"number": 7, "iucr": "p 2 m g (f7)", "point_order": 4, "h2": [2], "is_split": False, "ab_plus": ([2, 2], 0), "ab_full": ([2, 2], 0)},
}


def build_rosette(dihedral: bool = False, n: int = 5, wedge_delta: int = 0, mirror_wall: bool = False) -> dict[str, Any]:
    """Build a 0D rosette C_n or D_n graph with optional wedge disclination or mirror wall."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "subperiodic_rosette_json"):
        return json.loads(rs.subperiodic_rosette_json(bool(dihedral), int(n), int(wedge_delta), bool(mirror_wall)))
    n_orig = max(2, min(16, int(n)))
    eff_n = max(2, min(20, n_orig + int(wedge_delta)))
    symbol = f"D{n_orig}" if dihedral else f"C{n_orig}"
    ab_label = (f"Z_{n_orig}" if not dihedral else ("Z_2^2" if n_orig % 2 == 0 else "Z_2"))
    positions = [[0.0, 0.0]]
    for k in range(eff_n):
        th = 2.0 * math.pi * k / eff_n
        asym = 0.0 if dihedral else 0.22
        positions.append([math.cos(th), math.sin(th)])
        positions.append([2.0 * math.cos(th + asym), 2.0 * math.sin(th + asym)])
    edges = []
    for k in range(eff_n):
        kn = (k + 1) % eff_n
        i_in, i_out, j_in, j_out = 1 + 2 * k, 2 + 2 * k, 1 + 2 * kn, 2 + 2 * kn
        kr = 1 if (wedge_delta != 0 and k + 1 == eff_n) else 0
        edges.extend([(0, i_in, 0), (i_in, i_out, 0), (i_in, j_in, kr), (i_out, j_out, kr)])
        if dihedral and not mirror_wall:
            edges.append((i_out, j_in, 0))
    if mirror_wall and dihedral:
        edges.append((2, 1 + 2 * (eff_n // 2), 2))
    return {
        "symbol": symbol,
        "dihedral": bool(dihedral),
        "n": n_orig,
        "wedge_delta": int(wedge_delta),
        "mirror_wall": bool(mirror_wall),
        "point_order": (2 * n_orig if dihedral else n_orig),
        "ab_label": ab_label,
        "positions": positions,
        "edges": edges,
    }


def analyze_frieze(symbol: str) -> dict[str, Any]:
    """Return Bieberbach cohomology, splitness, and abelianization for one of the 7 frieze groups."""
    if symbol not in _FRIEZE_META:
        raise ValueError(f"unknown frieze group {symbol!r}; choose from {FRIEZE_SYMBOLS}")
    m = _FRIEZE_META[symbol]
    return {
        "dim": 1,
        "number": m["number"],
        "symbol": symbol,
        "iucr_name": m["iucr"],
        "point_order": m["point_order"],
        "h2_torsion": list(m["h2"]),
        "h2_label": fmt_abelian(m["h2"]),
        "is_split": m["is_split"],
        "ab_plus_label": fmt_abelian(*m["ab_plus"]),
        "ab_full_label": fmt_abelian(*m["ab_full"]),
    }


def build_frieze_patch(
    symbol: str,
    cells: int = 10,
    topology: str = "OpenStrip",
    defect: str = "None",
) -> dict[str, Any]:
    """Build 2D/3D strip patch for a frieze group with topology OpenStrip/Cylinder/Mobius."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "subperiodic_frieze_json"):
        return json.loads(rs.subperiodic_frieze_json(symbol, int(cells), topology, defect))
    summary = analyze_frieze(symbol)
    n_cells = max(4, min(24, int(cells)))
    nx, ny = n_cells * 2 + 1, 3
    radius, width, length = n_cells / math.pi, 1.4, float(n_cells)
    pos2d, pos3d = [], []
    for ix in range(nx):
        u = ix / (nx - 1)
        x2 = (u - 0.5) * length
        for iy in range(ny):
            v = iy / (ny - 1) - 0.5
            d_sh = 0.5 if (defect == "GlideDispiration" and ix > nx // 2) else 0.0
            y2 = v * width * (-1.0 if (defect == "HalfTurn180" and ix > nx // 2) else 1.0)
            pos2d.append([x2, y2])
            if topology == "Cylinder":
                th = 2.0 * math.pi * u
                r_eff = radius + d_sh * 0.15
                pos3d.append([r_eff * math.cos(th), v * width, r_eff * math.sin(th)])
            elif topology == "Mobius":
                th = 2.0 * math.pi * u
                tw = 0.5 * th
                r = radius + v * width * math.cos(tw)
                pos3d.append([r * math.cos(th), v * width * math.sin(tw), r * math.sin(th)])
            else:
                pos3d.append([x2, y2, d_sh * 0.25])
    edges, quads = [], []
    vid = lambda ix, iy: ix * ny + iy
    for ix in range(nx - 1):
        seam = defect in ("HalfTurn180", "GlideDispiration") and ix == nx // 2
        for iy in range(ny):
            edges.append((vid(ix, iy), vid(ix + 1, iy), 1 if seam else 0))
        for iy in range(ny - 1):
            edges.append((vid(ix, iy), vid(ix, iy + 1), 0))
            if symbol in ("p11g", "p2mg") and ix % 2 == 0:
                edges.append((vid(ix, iy), vid(ix + 1, iy + 1), 2))
            if symbol in ("p1m1", "p2mm", "p2") and iy == 0:
                edges.append((vid(ix, 0), vid(ix, ny - 1), 2))
            quads.append([vid(ix, iy), vid(ix + 1, iy), vid(ix + 1, iy + 1), vid(ix, iy + 1)])
    for iy in range(ny - 1):
        edges.append((vid(nx - 1, iy), vid(nx - 1, iy + 1), 0))
    return {
        "symbol": symbol,
        "iucr_name": summary["iucr_name"],
        "cells": n_cells,
        "topology": topology,
        "defect": defect,
        "summary": summary,
        "pos2d": pos2d,
        "pos3d": pos3d,
        "edges": edges,
        "quads": quads,
    }
