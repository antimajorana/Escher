"""Unified S^2/E^2/H^2 realiser, Conway orbifolds, Wythoff & vertex configs, UHP/3D H^2 surfaces,
O(N) 10^5+ hyperbolic kernels, and compact TGCell quotients (Rust-backed with pure-Python fallback).

Every public function calls the compiled `papertilling_rs` extension when it is available (and
`PAPERTILLING_NO_RUST` is unset) and otherwise a pure-Python implementation with the same JSON shapes.
The implementations live in the `_noneuclid_*` helper modules; this module is the public API.

Provenance: results carry a `provenance` string stamped `[PROVED]` (exact combinatorics / closed
formulas, sources cited in the helpers) or `[COMPUTATIONAL EVIDENCE]` (f64 root finding, e.g. the snub
generating point). Impossible inputs (bad orbifolds, odd-gon parity, Descartes obstructions, non-
hyperbolic {p,q} for H^2 kernels) raise `ValueError` with the reason.
"""
from __future__ import annotations

from ._noneuclid_h2large import H2LargePatch, build_hyperbolic_large
from ._noneuclid_models import (
    build_h2_surface_mesh,
    embed_tiling_on_surface,
    geodesic_arc_poincare,
    geodesic_arc_uhp,
    hyp_distance_poincare,
    hyp_distance_uhp,
    hyperboloid_to_poincare,
    klein_to_poincare,
    map_poincare_to_surface,
    poincare_to_hyperboloid,
    poincare_to_klein,
    poincare_to_uhp,
    uhp_to_poincare,
    unfold_papercraft_net,
)
from ._noneuclid_pq import build_unified_pq, build_wythoff, delta_pq, geometry_kind_pq
from ._noneuclid_quotients import TGCELL_NAMES, list_tgcell_quotients, tgcell_quotient
from ._noneuclid_symbols import (
    WALLPAPER_ORBIFOLD_TABLE,
    orbifold_to_wallpaper,
    parse_orbifold,
    parse_vertex_config,
    wallpaper_orbifold_table,
    wallpaper_to_orbifold,
)

__all__ = [
    "WALLPAPER_ORBIFOLD_TABLE", "TGCELL_NAMES", "delta_pq", "geometry_kind_pq",
    "build_unified_pq", "wallpaper_orbifold_table", "wallpaper_to_orbifold",
    "orbifold_to_wallpaper", "parse_orbifold", "build_wythoff", "parse_vertex_config",
    "poincare_to_uhp", "uhp_to_poincare", "poincare_to_klein", "klein_to_poincare",
    "poincare_to_hyperboloid", "hyperboloid_to_poincare", "hyp_distance_poincare",
    "hyp_distance_uhp", "geodesic_arc_poincare", "geodesic_arc_uhp",
    "map_poincare_to_surface", "build_h2_surface_mesh", "embed_tiling_on_surface",
    "unfold_papercraft_net", "list_tgcell_quotients", "tgcell_quotient",
    "H2LargePatch", "build_hyperbolic_large",
]
