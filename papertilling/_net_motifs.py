"""The one Python table of the 10 canonical 3D net motifs (mirror of `rust/papertilling-core/src/space3d/motifs.rs`).

Same sites in the same order as Rust. A label names the coloured structure: BCC, Diamond, Pyrochlore, FCC, SC, HCP
and beta-quartz carry one site kind; Tetragonal123 keeps two (one kind would be I4/mmm, #139); Rutile is Ti (0) at
2a + O (1) at 4f with x = 0.3; Perovskite is kind 0 at (0,0,0), kind 1 at (1/2,1/2,1/2), kind 2 = O at 3d.
Pyrochlore is the 16c orbit of Fd-3m (the diamond bond midpoints); beta-quartz (Si only) is the 3c orbit of
P6_222 (#180). `build_net_patch` mirrors Rust `space3d::nets3d::build_net_3d` (block, bonds and line-defect warp).
"""
from __future__ import annotations

import math
from typing import Any

HEX_C = 1.35
_SC = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_HEX = [[1.0, 0.0, 0.0], [-0.5, math.sqrt(0.75), 0.0], [0.0, 0.0, HEX_C]]
_F_CENTRING = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]


def _cubic(c: float) -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, c]]


def _f_centred(base: list[tuple[float, float, float]], kind: int) -> list[tuple[list[float], int]]:
    """F-centred copies of `base`, centring-major order, reduced into [0, 1) (same order as Rust `f_centred`)."""
    return [([(b[k] + t[k]) % 1.0 for k in range(3)], kind) for t in _F_CENTRING for b in base]


_DIAMOND = _f_centred([(0.0, 0.0, 0.0)], 0) + [
    ([0.25, 0.25, 0.25], 0), ([0.75, 0.75, 0.25], 0), ([0.75, 0.25, 0.75], 0), ([0.25, 0.75, 0.75], 0)]
_PYRO_BASE = [(0.125, 0.125, 0.125), (0.125, 0.875, 0.875), (0.875, 0.125, 0.875), (0.875, 0.875, 0.125)]

# name -> (space-group number, lattice rows, [(fractional site, kind)], bond cutoff)
NET_MOTIFS: dict[str, tuple[int, list[list[float]], list[tuple[list[float], int]], float]] = {
    "SimpleCubic": (221, _SC, [([0.0, 0.0, 0.0], 0)], 1.05),
    "BCC": (229, _SC, [([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.5], 0)], 0.92),
    "FCC": (225, _SC, _f_centred([(0.0, 0.0, 0.0)], 0), 0.76),
    "Diamond": (227, _SC, _DIAMOND, 0.48),
    "HCP": (194, _HEX, [([1 / 3, 2 / 3, 0.25], 0), ([2 / 3, 1 / 3, 0.75], 0)], 1.02),
    "Tetragonal123": (123, _cubic(1.15), [([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.5], 1)], 0.96),
    "Rutile136": (136, _cubic(0.72), [([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.5], 0), ([0.3, 0.3, 0.0], 1),
                                      ([0.7, 0.7, 0.0], 1), ([0.8, 0.2, 0.5], 1), ([0.2, 0.8, 0.5], 1)], 0.48),
    "Perovskite": (221, _SC, [([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.5], 1), ([0.5, 0.0, 0.0], 2),
                              ([0.0, 0.5, 0.0], 2), ([0.0, 0.0, 0.5], 2)], 0.55),
    "Pyrochlore": (227, _SC, _f_centred(_PYRO_BASE, 0), 0.42),
    "BetaQuartz": (180, _HEX, [([0.5, 0.0, 0.0], 0), ([0.0, 0.5, 2 / 3], 0), ([0.5, 0.5, 1 / 3], 0)], 0.72),
}

NET_LABELS = {
    "SimpleCubic": "Simple Cubic (#221)", "BCC": "BCC (#229 Im-3m)", "FCC": "FCC (#225 Fm-3m)",
    "Diamond": "Diamond (#227 Fd-3m)", "HCP": "HCP (#194 P6_3/mmc)", "Tetragonal123": "Tetragonal (#123 P4/mmm)",
    "Rutile136": "Rutile (#136 P4_2/mnm)", "Perovskite": "Perovskite (#221 Pm-3m)",
    "Pyrochlore": "Pyrochlore (#227 Fd-3m)", "BetaQuartz": "beta-Quartz (#180 P6_222)",
}

DEFECT_LABELS = {
    "None": "Pristine 3D Net",
    "ScrewDislocationZ": "Screw Dislocation (b = c e_z)",
    "PureDisclination90_123": "Pure 90° Wedge (#123, Δz = 0)",
    "ScrewDispiration42_136": "Forced 4_2 Screw-Dispiration (#136, Δz = c/2)",
}


def _warp(p: list[float], defect: str, c_len: float) -> list[float] | None:
    """Rust `build_net_3d` line-defect map of one site (None = removed by the wedge cut)."""
    th = math.atan2(p[1], p[0])
    if defect == "None":
        return p
    if defect == "ScrewDislocationZ":
        return [p[0], p[1], p[2] + c_len * th / (2.0 * math.pi)]
    if abs(th) < 0.24 * math.pi:
        return None
    r = math.hypot(p[0], p[1])
    alpha = th - 0.25 * math.pi if th >= 0.0 else th + 2.0 * math.pi - 0.25 * math.pi
    u = min(max(alpha / (1.5 * math.pi), 0.0), 1.0)
    th_new = 2.0 * math.pi * u
    dz = 0.5 * c_len * u if defect == "ScrewDispiration42_136" else 0.0
    return [r * math.cos(th_new), r * math.sin(th_new), p[2] + dz]


def build_net_patch(net: str, defect: str = "None") -> dict[str, Any]:
    """Pure-Python `build_net_3d`: a 2x2x2 block (3x3x2 with a line defect) with the same keys as the Rust JSON."""
    if net not in NET_MOTIFS:
        raise ValueError(f"unknown 3D net {net!r}; choose from {list(NET_MOTIFS)}")
    if defect not in DEFECT_LABELS:
        raise ValueError(f"unknown 3D line defect {defect!r}; choose from {list(DEFECT_LABELS)}")
    sg, lat, motif, cutoff = NET_MOTIFS[net]
    nxy, nz = (2 if defect == "None" else 3), 2
    mid, mid_z = 0.5 * nxy, 0.5 * nz * lat[2][2]
    raw_pos, raw_kinds = [], []
    for ix in range(nxy):
        for iy in range(nxy):
            for iz in range(nz):
                for f, k in motif:
                    fx, fy, fz = ix + f[0], iy + f[1], iz + f[2]
                    raw_pos.append([fx * lat[0][0] + fy * lat[1][0] + fz * lat[2][0] - mid * lat[0][0],
                                    fx * lat[0][1] + fy * lat[1][1] + fz * lat[2][1] - mid * lat[1][1],
                                    fx * lat[0][2] + fy * lat[1][2] + fz * lat[2][2] - mid_z])
                    raw_kinds.append(k)
    c2 = cutoff * cutoff
    raw_bonds = [(i, j) for i in range(len(raw_pos)) for j in range(i + 1, len(raw_pos))
                 if 1e-4 < sum((raw_pos[j][k] - raw_pos[i][k]) ** 2 for k in range(3)) <= c2]
    positions, kinds, idx = [], [], [None] * len(raw_pos)
    for n, p in enumerate(raw_pos):
        q = _warp(p, defect, lat[2][2])
        if q is not None:
            idx[n] = len(positions)
            positions.append(q)
            kinds.append(raw_kinds[n])
    bonds = []
    for i, j in raw_bonds:
        u, v = idx[i], idx[j]
        if u is None or v is None:
            continue
        length = math.sqrt(sum((positions[v][k] - positions[u][k]) ** 2 for k in range(3)))
        if length < 1.85 * cutoff:
            bonds.append([u, v, 1 if defect != "None" and length > 1.28 * cutoff else 0])
    axis = None if defect == "None" else [[0.0, 0.0, -mid_z - 0.4], [0.0, 0.0, mid_z + 0.6]]
    label, dlabel = NET_LABELS[net], DEFECT_LABELS[defect]
    return {"name": label, "space_group_num": sg, "defect_label": dlabel, "positions": positions, "bonds": bonds,
            "site_kinds": kinds, "defect_axis": axis,
            "info": f"{label} | {dlabel} | {len(positions)} sites, {len(bonds)} bonds"}
