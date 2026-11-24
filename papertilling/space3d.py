"""All 80 crystallographic layer groups and 230 3D space groups, plus canonical 3D crystal nets."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from . import _rust
from ._net_motifs import NET_MOTIFS, build_net_patch
from .bieberbach import SpaceGroupAnalyzer

__all__ = [
    "CANONICAL_NETS_3D",
    "list_layer_groups",
    "analyze_layer_group",
    "list_space_groups_3d",
    "analyze_space_group_3d",
    "build_net_3d",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"

CANONICAL_NETS_3D = list(NET_MOTIFS)


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
    return deepcopy(_raw_layer_80())


def list_space_groups_3d() -> list[dict[str, Any]]:
    """Return metadata (number, HM, Schoenflies, splitness, H^2, G^ab) for all 230 3D space groups."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "space_groups_3d_json"):
        return json.loads(rs.space_groups_3d_json())
    return deepcopy(_raw_space_230())


def analyze_layer_group(number: int) -> dict[str, Any]:
    """Compute Bieberbach invariants (H^2, splitness, G^ab) for layer group `1..=80`."""
    entry = next((g for g in _raw_layer_80() if g["number"] == int(number)), None)
    if entry is None:
        raise ValueError(f"unknown layer group #{number} (expected 1..=80)")
    rs = _rust.get()
    if rs is not None and hasattr(rs, "analyze_layer_group_json"):
        summary = json.loads(rs.analyze_layer_group_json(entry["number"]))
    else:
        half = Fraction(1, 2)
        basis = [[half, half], [-half, half]] if entry["centred"] else [[1, 0], [0, 1]]
        ops = [(op["r"], [Fraction(n, d) for n, d in op["t2"]]) for op in entry["ops"]]
        analyzer = SpaceGroupAnalyzer._from_subperiodic(2, 3, entry["number"], entry["symbol"], ops, basis)
        summary = analyzer.summarize(torus_ns=())
    return dict(summary, system=entry["system"], centred=entry["centred"])


def analyze_space_group_3d(number: int, torus_ns: tuple[int, ...] = (2,)) -> dict[str, Any]:
    """Compute exact 3D Bieberbach cohomology H^2, splitness, (G^+)^ab, G^ab for space group `1..=230`."""
    rs = _rust.get()
    summary = None
    if rs is not None and hasattr(rs, "analyze_space_group_3d_json"):
        summary = json.loads(rs.analyze_space_group_3d_json(int(number)))
        if torus_ns == (2,):
            return summary
    entry = next((g for g in _raw_space_230() if g["number"] == int(number)), None)
    if entry is None:
        raise ValueError(f"unknown 3D space group #{number} (expected 1..=230)")
    tb = [[Fraction(n, d) for n, d in row] for row in entry["tb_rows"]]
    gens = [
        (op["r"], [Fraction(n, d) for n, d in op["t"]]) for op in entry["ops"]
    ] + [([[1, 0, 0], [0, 1, 0], [0, 0, 1]], row) for row in tb]
    sg = SpaceGroupAnalyzer(3, entry["number"], entry["symbol"], gens, tb)
    if summary is not None:
        summary["torus_counts"] = [list(row) for row in sg._torus_counts(torus_ns)]
        return summary
    return sg.summarize(torus_ns=torus_ns)


def build_net_3d(net: str = "SimpleCubic", defect: str = "None") -> dict[str, Any]:
    """Build a 3D crystal patch with optional topological line defect (ScrewDislocationZ / PureDisclination90_123 / ScrewDispiration42_136)."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "build_net_3d_json"):
        return json.loads(rs.build_net_3d_json(net, defect))
    return build_net_patch(net, defect)
