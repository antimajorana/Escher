"""0D rosettes (C_n, D_n) and 1D frieze groups (all 7: 5 symmorphic + 2 nonsymmorphic)."""
from __future__ import annotations

import json
import math
from typing import Any

from . import _rust
from .bieberbach import fmt_abelian

__all__ = [
    "FRIEZE_SYMBOLS", "NOT_ISOMETRIC_NOTE", "build_rosette", "analyze_frieze", "build_frieze_patch",
    "frieze_decoration", "mobius_columns",
]

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


NOT_ISOMETRIC_NOTE = "topological picture, not isometric"

_T_UP, _T_DN, _B_UP, _B_DN = (1, 2), (2, 1), (0, 1), (1, 0)
# Decoration diagonals (left_row, right_row) per quad column parity (even, odd); mirrors
# `frieze_decoration` in `subperiodic/frieze_patch.rs`. Each realises exactly its frieze group.
_DECOR: dict[str, tuple[list[tuple[int, int]], list[tuple[int, int]]]] = {
    "p1": ([_T_UP], []),
    "p11m": ([_T_UP, _B_DN], []),
    "p1m1": ([_T_UP], [_T_DN]),
    "p11g": ([_T_UP], [_B_DN]),
    "p2": ([_T_UP], [_B_UP]),
    "p2mm": ([_T_UP, _B_DN], [_T_DN, _B_UP]),
    "p2mg": ([_T_UP, _B_UP], [_T_DN, _B_DN]),
}


def frieze_decoration(symbol: str, c: int) -> list[tuple[int, int]]:
    """Decoration diagonals of quad column `c` as (left_row, right_row); rows 0..2 bottom to top."""
    even, odd = _DECOR.get(symbol, ([], []))
    return list(even if c % 2 == 0 else odd)


def mobius_columns(symbol: str, cells: int) -> int:
    """Glide columns of a Möbius strip of `cells` periods, or ValueError with the [PROVED] refusal.

    A Möbius strip needs g(x, y) = (x + L, -y), L != 0, in the group: a horizontal mirror (L whole
    periods: p11m, p2mm) or a glide (L an odd number of glides: p11g, p2mg). p1, p2, p1m1 have none.
    """
    if symbol in ("p11m", "p2mm"):
        return 2 * cells
    if symbol in ("p11g", "p2mg"):
        return 2 * cells + 1
    if symbol in ("p1", "p2", "p1m1"):
        raise ValueError(
            f"[PROVED] {symbol} cannot be closed into a Möbius strip: that needs a symmetry that flips the "
            f"strip across its axis without a fixed point (a horizontal mirror or a glide), and {symbol} has none"
        )
    raise ValueError(f"unknown frieze group {symbol!r}")


def build_frieze_patch(
    symbol: str,
    cells: int = 10,
    topology: str = "OpenStrip",
    defect: str = "None",
) -> dict[str, Any]:
    """Open strip, cylinder or Möbius strip for a frieze group (glued seam; [PROVED] Möbius refusals)."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "subperiodic_frieze_json"):
        return json.loads(rs.subperiodic_frieze_json(symbol, int(cells), topology, defect))
    summary = analyze_frieze(symbol)
    if topology not in ("Cylinder", "Mobius"):
        topology = "OpenStrip"
    if defect not in ("HalfTurn180", "GlideDispiration"):
        defect = "None"
    n_cells = max(4, min(24, int(cells)))
    ny, width = 3, 1.4
    closed, flip = topology != "OpenStrip", topology == "Mobius"
    nx = {"OpenStrip": 2 * n_cells + 1, "Cylinder": 2 * n_cells}.get(topology) or mobius_columns(symbol, n_cells)
    n_qc = nx if closed else nx - 1
    length = n_qc / 2.0
    r_cyl = length / (2.0 * math.pi)
    r_mob = max(r_cyl, width)
    pos2d, pos3d = [], []
    for ix in range(nx):
        x2 = ix * 0.5 - 0.5 * length
        th = 2.0 * math.pi * ix / nx
        right = ix > nx // 2
        d_sh = 0.5 if (defect == "GlideDispiration" and right) else 0.0
        for iy in range(ny):
            v = iy / (ny - 1) - 0.5
            y2 = v * width * (-1.0 if (defect == "HalfTurn180" and right) else 1.0)
            pos2d.append([x2, y2])
            if topology == "Cylinder":
                r = r_cyl + d_sh * 0.15
                pos3d.append([r * math.cos(th), v * width, r * math.sin(th)])
            elif topology == "Mobius":
                tw = 0.5 * th
                r = r_mob + v * width * math.cos(tw)
                pos3d.append([r * math.cos(th), v * width * math.sin(tw), r * math.sin(th)])
            else:
                pos3d.append([x2, y2, d_sh * 0.25])
    vid = lambda ix, iy: ix * ny + iy  # noqa: E731
    rho = (lambda iy: ny - 1 - iy) if flip else (lambda iy: iy)
    edges: list[tuple[int, int, int]] = [(vid(ix, iy), vid(ix, iy + 1), 0) for ix in range(nx) for iy in range(ny - 1)]
    quads: list[list[int]] = []
    wrap_edges: list[int] = []
    wrap_quads: list[int] = []
    seam_defect = defect in ("HalfTurn180", "GlideDispiration")
    for c in range(n_qc):
        wrap = c + 1 == nx
        right_v = (lambda iy: vid(0, rho(iy))) if wrap else (lambda iy, c=c: vid(c + 1, iy))
        kind = 1 if (seam_defect and c == nx // 2) else 0
        for iy in range(ny):
            if wrap:
                wrap_edges.append(len(edges))
            edges.append((vid(c, iy), right_v(iy), kind))
        for l_row, r_row in frieze_decoration(symbol, c):
            if wrap:
                wrap_edges.append(len(edges))
            edges.append((vid(c, l_row), right_v(r_row), 2))
        for iy in range(ny - 1):
            if wrap:
                wrap_quads.append(len(quads))
            quads.append([vid(c, iy), right_v(iy), right_v(iy + 1), vid(c, iy + 1)])
    if topology == "OpenStrip":
        length_periods, glue_note = float(n_cells), "open strip"
    elif topology == "Cylinder":
        length_periods, glue_note = float(n_cells), f"cylinder: x ~ x + {n_cells} a"
    elif nx % 2 == 1:
        length_periods = n_cells + 0.5
        glue_note = f"Möbius: (x, y) ~ (x + {n_cells}.5 a, -y), an odd number ({nx}) of glides"
    else:
        length_periods = float(n_cells)
        glue_note = f"Möbius: (x, y) ~ (x + {n_cells} a, -y), the horizontal mirror after {n_cells} periods"
    return {
        "symbol": symbol,
        "iucr_name": summary["iucr_name"],
        "cells": n_cells,
        "topology": topology,
        "defect": defect,
        "summary": summary,
        "columns": nx,
        "length_periods": length_periods,
        "pos2d": pos2d,
        "pos3d": pos3d,
        "edges": edges,
        "quads": quads,
        "wrap_edges": wrap_edges,
        "wrap_quads": wrap_quads,
        "picture_note": NOT_ISOMETRIC_NOTE if flip else None,
        "glue_note": glue_note,
    }
