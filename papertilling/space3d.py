"""All 80 crystallographic layer groups and 230 3D space groups, plus canonical 3D crystal nets."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from . import _rust
from .bieberbach import SpaceGroupAnalyzer, fmt_abelian

__all__ = [
    "CANONICAL_NETS_3D",
    "list_layer_groups",
    "analyze_layer_group",
    "list_space_groups_3d",
    "analyze_space_group_3d",
    "build_net_3d",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"

CANONICAL_NETS_3D = [
    "SimpleCubic",
    "BCC",
    "FCC",
    "Diamond",
    "HCP",
    "Tetragonal123",
    "Rutile136",
    "Perovskite",
    "Pyrochlore",
    "BetaQuartz",
]


@lru_cache(maxsize=1)
def _raw_layer_80() -> list[dict[str, Any]]:
    return json.loads((_DATA_DIR / "layer_groups_80.json").read_text(encoding="utf-8"))["groups"]


@lru_cache(maxsize=1)
def _raw_space_230() -> list[dict[str, Any]]:
    return json.loads((_DATA_DIR / "space_groups_230.json").read_text(encoding="utf-8"))["groups"]


def list_layer_groups() -> list[dict[str, Any]]:
    """Return metadata for all 80 ITE Vol. E layer groups."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "layer_groups_json"):
        return json.loads(rs.layer_groups_json())
    return list(_raw_layer_80())


def list_space_groups_3d() -> list[dict[str, Any]]:
    """Return metadata (number, HM, Schoenflies, splitness, H^2, G^ab) for all 230 3D space groups."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "space_groups_3d_json"):
        return json.loads(rs.space_groups_3d_json())
    return list(_raw_space_230())


def analyze_layer_group(number: int) -> dict[str, Any]:
    """Compute Bieberbach invariants (H^2, splitness, G^ab) for layer group `1..=80`."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "analyze_layer_group_json"):
        return json.loads(rs.analyze_layer_group_json(int(number)))
    entry = next((g for g in _raw_layer_80() if g["number"] == int(number)), None)
    if entry is None:
        raise ValueError(f"unknown layer group #{number} (expected 1..=80)")
    # Analyze in-plane + z=0 embedding using 2D plane-lattice reduction + transverse sign:
    ops = entry["ops"]
    tb2 = [[Fraction(1, 2), Fraction(1, 2)], [Fraction(-1, 2), Fraction(1, 2)]] if entry["centred"] else [[1, 0], [0, 1]]
    gens = [( [[int(r[0][0]), int(r[0][1])], [int(r[1][0]), int(r[1][1])]], [Fraction(n, d) for n, d in op["t2"]] ) for op in ops for r in [op["r"]]]
    for row in tb2:
        gens.append(([[1, 0], [0, 1]], row))
    # Return full entry plus exact in-plane point count when in pure-Python subperiodic fallback
    sg_inplane = SpaceGroupAnalyzer(2, entry["number"], entry["symbol"], gens[:len(ops)] + [([[1,0],[0,1]], r) for r in tb2], tb2)
    h2_t, is_split = sg_inplane.compute_cohomology_h2()
    ab_ft, ab_ff, _ = sg_inplane.compute_abelianization()
    return {
        "dim": 2,
        "number": entry["number"],
        "symbol": entry["symbol"],
        "point_order": entry["point_order"],
        "h2_torsion": h2_t,
        "h2_label": fmt_abelian(h2_t),
        "is_split": is_split,
        "ab_full_label": fmt_abelian(ab_ft, ab_ff),
        "system": entry["system"],
        "centred": entry["centred"],
    }


def analyze_space_group_3d(number: int, torus_ns: tuple[int, ...] = (2,)) -> dict[str, Any]:
    """Compute exact 3D Bieberbach cohomology H^2, splitness, (G^+)^ab, G^ab for space group `1..=230`."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "analyze_space_group_3d_json"):
        return json.loads(rs.analyze_space_group_3d_json(int(number)))
    entry = next((g for g in _raw_space_230() if g["number"] == int(number)), None)
    if entry is None:
        raise ValueError(f"unknown 3D space group #{number} (expected 1..=230)")
    tb = [[Fraction(n, d) for n, d in row] for row in entry["tb_rows"]]
    gens = [
        (op["r"], [Fraction(n, d) for n, d in op["t"]]) for op in entry["ops"]
    ] + [([[1, 0, 0], [0, 1, 0], [0, 0, 1]], row) for row in tb]
    sg = SpaceGroupAnalyzer(3, entry["number"], entry["symbol"], gens, tb)
    return sg.summarize(torus_ns=torus_ns)


def build_net_3d(net: str = "SimpleCubic", defect: str = "None") -> dict[str, Any]:
    """Build a 3D crystal patch with optional topological line defect (ScrewDislocationZ / PureDisclination90_123 / ScrewDispiration42_136)."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "build_net_3d_json"):
        return json.loads(rs.build_net_3d_json(net, defect))
    if net not in CANONICAL_NETS_3D:
        raise ValueError(f"unknown 3D net {net!r}; choose from {CANONICAL_NETS_3D}")
    # Pure-Python minimal 2x2x2 patch fallback for the 10 canonical nets:
    basis_map = {
        "SimpleCubic": (221, [([0, 0, 0], 0)], 1.05, 1.0),
        "BCC": (229, [([0, 0, 0], 0), ([0.5, 0.5, 0.5], 1)], 0.92, 1.0),
        "FCC": (225, [([0, 0, 0], 0), ([0.5, 0.5, 0], 0), ([0.5, 0, 0.5], 0), ([0, 0.5, 0.5], 0)], 0.76, 1.0),
        "Tetragonal123": (123, [([0, 0, 0], 0), ([0.5, 0.5, 0.5], 1)], 0.96, 1.15),
        "Rutile136": (136, [([0, 0, 0], 0), ([0.5, 0.5, 0.5], 0), ([0.3, 0.3, 0], 1), ([0.7, 0.7, 0], 1)], 0.48, 0.72),
    }
    sg_num, motif, cutoff, c_len = basis_map.get(net, basis_map["SimpleCubic"])
    nxy = 2 if defect == "None" else 3
    mid = 0.5 * nxy
    raw_pos, raw_kinds = [], []
    for ix in range(nxy):
        for iy in range(nxy):
            for iz in range(2):
                for f, k in motif:
                    raw_pos.append([ix + f[0] - mid, iy + f[1] - mid, (iz + f[2] - 1.0) * c_len])
                    raw_kinds.append(k)
    bonds = []
    for i in range(len(raw_pos)):
        for j in range(i + 1, len(raw_pos)):
            d = math.sqrt(sum((raw_pos[j][k] - raw_pos[i][k]) ** 2 for k in range(3)))
            if 0.01 < d <= cutoff:
                bonds.append((i, j, 0))
    return {"name": net, "space_group_num": sg_num, "defect_label": defect, "positions": raw_pos, "bonds": bonds, "site_kinds": raw_kinds}
