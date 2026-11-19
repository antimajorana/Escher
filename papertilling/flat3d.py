"""Stage 4 geometries: torsion-free groups, 10 closed flat 3D platycosms, flat 3D
non-compact slabs/chimneys, chiral `(n, m)` nanotubes/ribbons, symmetry-centred
flakes, proper `det=+1` 3D Niggli, `CELLTRAN` supercells, strain tensors, and BCs.
"""
from __future__ import annotations

from fractions import Fraction
import json
import math
from typing import Any

import numpy as np

from . import _rust
from ._flat3d_fallback import find_inplane_glide, py_niggli, py_platycosm
from .bieberbach import SpaceGroupAnalyzer, _det, _snf, _solve_int_linear, fmt_abelian, from_wallpaper_group
from .groups import ITA_NUMBER
from .legacy.wallpaper_lattices import UnitCell, _symmetry_ops
from .space3d import _raw_space_230

# Wolf's names G1..G6 / B1..B4 (Spaces of Constant Curvature, Sec. 3.5; theorem locator to verify) for the Conway-Rossetti codes.
WOLF_ALIASES = {"g1": "c1", "g2": "c2", "g3": "c3", "g4": "c4", "g5": "c6", "g6": "c22", "b1": "+a1", "b2": "-a1", "b3": "+a2", "b4": "-a2"}

__all__ = ["PLATYCOSMS", "list_platycosms", "lookup_platycosm", "canonical_cell_3d",
           "torsion_free_wallpaper_groups", "torsion_free_space_groups", "is_torsion_free",
           "check_platycosm_compatibility", "build_platycosm_quotient", "build_slab_3d",
           "build_nanotube", "build_nanoribbon", "build_symmetric_flake", "build_wigner_seitz_flake",
           "niggli_reduce_3d", "build_supercell_3d", "build_supercell_2d", "cell_strain_tensor",
           "apply_boundary_conditions_3d", "apply_boundary_conditions_2d"]

# Published table (J. H. Conway & J. P. Rossetti, "Describing the platycosms", arXiv:math/0311476, 2003; ITA numbers
# from International Tables for Crystallography Vol. A). H_1 is re-derived as G^ab in tests/test_s4_flat3d_geometries.py.
PLATYCOSMS: list[dict[str, Any]] = [
    {"code": "c1", "name": "torocosm", "orientable": True, "ita_numbers": [1], "ita_symbols": ["P1"], "point_order": 1, "crystal_system": "triclinic", "h1_label": "Z^3", "b1": 3, "h1_torsion": []},
    {"code": "c2", "name": "dicosm", "orientable": True, "ita_numbers": [4], "ita_symbols": ["P2_1"], "point_order": 2, "crystal_system": "monoclinic", "h1_label": "Z_2^2 x Z", "b1": 1, "h1_torsion": [2, 2]},
    {"code": "c3", "name": "tricosm", "orientable": True, "ita_numbers": [144, 145], "ita_symbols": ["P3_1", "P3_2"], "point_order": 3, "crystal_system": "trigonal", "h1_label": "Z_3 x Z", "b1": 1, "h1_torsion": [3]},
    {"code": "c4", "name": "tetracosm", "orientable": True, "ita_numbers": [76, 78], "ita_symbols": ["P4_1", "P4_3"], "point_order": 4, "crystal_system": "tetragonal", "h1_label": "Z_2 x Z", "b1": 1, "h1_torsion": [2]},
    {"code": "c6", "name": "hexacosm", "orientable": True, "ita_numbers": [169, 170], "ita_symbols": ["P6_1", "P6_5"], "point_order": 6, "crystal_system": "hexagonal", "h1_label": "Z", "b1": 1, "h1_torsion": []},
    {"code": "c22", "name": "didicosm", "orientable": True, "ita_numbers": [19], "ita_symbols": ["P2_12_12_1"], "point_order": 4, "crystal_system": "orthorhombic", "h1_label": "Z_4^2", "b1": 0, "h1_torsion": [4, 4]},
    {"code": "+a1", "name": "first amphicosm", "orientable": False, "ita_numbers": [7], "ita_symbols": ["Pc"], "point_order": 2, "crystal_system": "monoclinic", "h1_label": "Z_2 x Z^2", "b1": 2, "h1_torsion": [2]},
    {"code": "-a1", "name": "second amphicosm", "orientable": False, "ita_numbers": [9], "ita_symbols": ["Cc"], "point_order": 2, "crystal_system": "monoclinic", "h1_label": "Z^2", "b1": 2, "h1_torsion": []},
    {"code": "+a2", "name": "first amphidicosm", "orientable": False, "ita_numbers": [29], "ita_symbols": ["Pca2_1"], "point_order": 4, "crystal_system": "orthorhombic", "h1_label": "Z_2^2 x Z", "b1": 1, "h1_torsion": [2, 2]},
    {"code": "-a2", "name": "second amphidicosm", "orientable": False, "ita_numbers": [33], "ita_symbols": ["Pna2_1"], "point_order": 4, "crystal_system": "orthorhombic", "h1_label": "Z_4 x Z", "b1": 1, "h1_torsion": [4]},
]

_SC = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_HEX = [[1.0, 0.0, 0.0], [-0.5, math.sqrt(0.75), 0.0], [0.0, 0.0, 1.35]]
_NET_MOTIFS = {
    "SimpleCubic": (221, _SC, [([0.0, 0.0, 0.0], 0)], 1.05),
    "BCC": (229, _SC, [([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.5], 1)], 0.92),
    "FCC": (225, _SC, [([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.0], 0), ([0.5, 0.0, 0.5], 0), ([0.0, 0.5, 0.5], 0)], 0.76),
    "Diamond": (227, _SC, [([0, 0, 0], 0), ([0.5, 0.5, 0], 0), ([0.5, 0, 0.5], 0), ([0, 0.5, 0.5], 0), ([0.25, 0.25, 0.25], 1), ([0.75, 0.75, 0.25], 1), ([0.75, 0.25, 0.75], 1), ([0.25, 0.75, 0.75], 1)], 0.48),
    "HCP": (194, _HEX, [([1 / 3, 2 / 3, 0.25], 0), ([2 / 3, 1 / 3, 0.75], 0)], 1.02),
    "Tetragonal123": (123, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.15]], [([0.0, 0.0, 0.0], 0), ([0.5, 0.5, 0.5], 1)], 0.96),
    "Rutile136": (136, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.72]], [([0, 0, 0], 0), ([0.5, 0.5, 0.5], 0), ([0.3, 0.3, 0], 1), ([0.7, 0.7, 0], 1), ([0.8, 0.2, 0.5], 1), ([0.2, 0.8, 0.5], 1)], 0.48),
    "Perovskite": (221, _SC, [([0, 0, 0], 0), ([0.5, 0.5, 0.5], 1), ([0.5, 0, 0], 2), ([0, 0.5, 0], 2), ([0, 0, 0.5], 2)], 0.55),
    "Pyrochlore": (227, _SC, [([0, 0, 0], 0), ([0.25, 0.25, 0], 0), ([0.25, 0, 0.25], 0), ([0, 0.25, 0.25], 0), ([0.5, 0.5, 0.5], 1)], 0.42),
    "BetaQuartz": (180, _HEX, [([0.5, 0.0, 0.0], 0), ([0.0, 0.5, 1 / 3], 0), ([0.5, 0.5, 2 / 3], 0)], 0.72),
}


def _coerce_cell2d(cell: Any) -> dict[str, Any]:
    if hasattr(cell, "cell") and callable(cell.cell):
        cell = cell.cell()
    if hasattr(cell, "a1") and hasattr(cell, "a2"):
        raw = {"a1": cell.a1, "a2": cell.a2, "basis": cell.basis, "bonds": cell.bonds}
    elif isinstance(cell, dict):
        raw = cell
    else:
        raise TypeError(f"cannot coerce {type(cell).__name__} to Cell2D")
    bonds = [[int(b["i"]), int(b["j"]), int(b["n1"]), int(b["n2"]), int(b.get("kind", 0))] if isinstance(b, dict)
             else [int(x) for x in b] + ([0] if len(b) == 4 else []) for b in raw.get("bonds", [])]
    return {"a1": [float(x) for x in raw["a1"]], "a2": [float(x) for x in raw["a2"]], "basis": [[float(x) for x in p] for p in raw["basis"]], "bonds": bonds}


def canonical_cell_3d(net: str) -> dict[str, Any]:
    key = next((k for k in _NET_MOTIFS if k.lower() == net.strip().lower()), None)
    if key is None:
        raise ValueError(f"unknown canonical 3D net {net!r}")
    sg, lat, motif, cutoff = _NET_MOTIFS[key]
    lat_a = np.asarray(lat, float)
    basis = [(np.asarray(f, float) @ lat_a).tolist() for f, _ in motif]
    kinds, c2, bonds_set = [int(k) for _, k in motif], cutoff * cutoff, set()
    for i, (fi, _) in enumerate(motif):
        pi = np.asarray(fi, float) @ lat_a
        for j in range(i, len(motif)):
            fj = np.asarray(motif[j][0], float)
            for s1, s2, s3 in np.ndindex(5, 5, 5):
                n1, n2, n3 = s1 - 2, s2 - 2, s3 - 2
                if (i != j or n1 or n2 or n3) and 1e-4 < float(np.sum(((fj + [n1, n2, n3]) @ lat_a - pi) ** 2)) <= c2:
                    bonds_set.add((i, j, n1, n2, n3, 0) if (i, j, n1, n2, n3) <= (j, i, -n1, -n2, -n3) else (j, i, -n1, -n2, -n3, 0))
    return {"lattice": lat, "basis": basis, "bonds": [list(b) for b in sorted(bonds_set)], "space_group_num": sg, "site_kinds": kinds}


def _coerce_cell3d(net_or_cell: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(net_or_cell, str):
        return net_or_cell, canonical_cell_3d(net_or_cell)
    c = dict(net_or_cell)
    c["bonds"] = _bond_rows(c.get("bonds", []), 3)
    if not c.get("site_kinds"):
        c["site_kinds"] = [0] * len(c["basis"])
    return json.dumps(c), c


def _bond_rows(bonds: list[Any], dim: int) -> list[list[int]]:
    """Bonds as rows `[i, j, n1, .., n_dim, kind]` (Rust serializes `Bond2D`/`Bond3D` as objects)."""
    keys = ("i", "j", "n1", "n2", "n3")[: 2 + dim]
    return [[int(b[k]) for k in keys] + [int(b.get("kind", 0))] if isinstance(b, dict) else [int(x) for x in b] + ([0] if len(b) == 2 + dim else []) for b in bonds]


def _rs_json(fn_name: str, *args: Any) -> Any:
    rs = _rust.get()
    return json.loads(getattr(rs, fn_name)(*args)) if rs is not None and hasattr(rs, fn_name) else None


def list_platycosms() -> list[dict[str, Any]]:
    return res if (res := _rs_json("flat3d_list_platycosms_json")) is not None else [dict(p) for p in PLATYCOSMS]


def lookup_platycosm(query: str) -> dict[str, Any]:
    q = str(query).strip()
    ql = WOLF_ALIASES.get(q.lower(), q.lower())
    for p in PLATYCOSMS:
        if p["code"] == ql or p["name"] == ql or (q.isdigit() and int(q) in p["ita_numbers"]) or any(s.lower().replace("_", "") == ql.replace("_", "") for s in p["ita_symbols"]):
            return dict(p)
    raise ValueError(f"unknown platycosm {query!r}")


def _py_analyze_tf(sg: SpaceGroupAnalyzer, full: bool = True) -> dict[str, Any]:
    """Exact torsion test: coset (R | tau + Z^n) holds a finite-order element iff S_R tau in S_R Z^n, S_R = sum R^k."""
    n, obs = sg.dim, []
    for r_idx in range(1, len(sg.p_ops)):
        ord_r, cur, s_mat = 1, r_idx, [[0] * n for _ in range(n)]
        while cur != 0:
            cur, ord_r = sg.mul_table[cur][r_idx], ord_r + 1
        cur = 0
        for _ in range(ord_r):
            s_mat = [[s_mat[i][j] + sg.p_ops[cur][i][j] for j in range(n)] for i in range(n)]
            cur = sg.mul_table[cur][r_idx]
        rhs = [sum(Fraction(s_mat[i][j]) * sg.tau[r_idx][j] for j in range(n)) for i in range(n)]
        if all(x.denominator == 1 for x in rhs) and _solve_int_linear(s_mat, [-x.numerator for x in rhs]) is not None:
            obs.append({"r_idx": r_idx, "order": ord_r, "power_k": 1, "description": f"[PROVED] coset #{r_idx} (order {ord_r}) contains an element with a fixed point"})
    out = {"dim": sg.dim, "number": sg.number, "symbol": sg.symbol, "is_torsion_free": not obs, "point_order": len(sg.p_ops), "is_split": None, "h2_label": None, "ab_full_label": None, "obstructions": obs}
    if full or not obs:  # the pure-Python 230-group sweep skips H^2 / G^ab of non-Bieberbach groups (too slow)
        h2_t, out["is_split"] = sg.compute_cohomology_h2()
        ab_t, ab_f, _ = sg.compute_abelianization()
        out["h2_label"], out["ab_full_label"] = fmt_abelian(h2_t), fmt_abelian(ab_t, ab_f)
    return out


def _sg3_analyzer(e: dict[str, Any]) -> SpaceGroupAnalyzer:
    tb = [[Fraction(n, d) for n, d in row] for row in e["tb_rows"]]
    gens = [(op["r"], [Fraction(n, d) for n, d in op["t"]]) for op in e["ops"]] + [([[1, 0, 0], [0, 1, 0], [0, 0, 1]], r) for r in tb]
    return SpaceGroupAnalyzer(3, e["number"], e["symbol"], gens, tb)

def torsion_free_wallpaper_groups() -> list[dict[str, Any]]:
    return res if (res := _rs_json("flat3d_torsion_free_wallpaper_json")) is not None else [_py_analyze_tf(from_wallpaper_group(g)) for g in ITA_NUMBER]

def torsion_free_space_groups() -> list[dict[str, Any]]:
    return res if (res := _rs_json("flat3d_torsion_free_space_groups_json")) is not None else [_py_analyze_tf(_sg3_analyzer(e), full=False) for e in _raw_space_230()]

def is_torsion_free(spec: int | str | dict[str, Any]) -> dict[str, Any]:
    arg = json.dumps(spec) if isinstance(spec, dict) else str(spec)
    if (res := _rs_json("flat3d_is_torsion_free_json", arg)) is not None:
        return res
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        e = next((g for g in _raw_space_230() if g["number"] == int(spec)), None)
        if e is None:
            raise ValueError(f"unknown 3D space group #{spec}")
        return _py_analyze_tf(_sg3_analyzer(e))
    if isinstance(spec, str):
        return _py_analyze_tf(from_wallpaper_group(spec))
    tb = [[Fraction(*x) if isinstance(x, (list, tuple)) else Fraction(x) for x in row] for row in spec["tb_rows"]]
    gens = [(op["r"], [Fraction(*x) if isinstance(x, (list, tuple)) else Fraction(x) for x in op["t"]]) for op in spec["ops"]]
    return _py_analyze_tf(SpaceGroupAnalyzer(int(spec["dim"]), int(spec.get("number", 0)), str(spec.get("symbol", "custom")), gens, tb))

_COMPAT_KEYS = ("platycosm_code", "status", "compatible", "ita_number", "ita_symbol", "supercell", "origin_shift", "reason")


def check_platycosm_compatibility(net_or_cell: str | dict[str, Any], platycosm: str, supercell: Any = None, ignore_site_kinds: bool = False) -> dict[str, Any]:
    arg, cell = _coerce_cell3d(net_or_cell)
    sup = [int(x) for x in supercell] if supercell is not None else None
    if (res := _rs_json("flat3d_check_platycosm_json", arg, platycosm, sup, bool(ignore_site_kinds))) is not None:
        return res
    res = py_platycosm(cell, lookup_platycosm(platycosm), tuple(sup) if sup else None, ignore_site_kinds)
    if res["status"] == "OK":
        res["reason"] = f"compatible with {res['platycosm_code']} ({res['platycosm_name']}) on supercell {res['supercell']}"
    return {k: res[k] for k in _COMPAT_KEYS}


def build_platycosm_quotient(net_or_cell: str | dict[str, Any], platycosm: str, supercell: Any = None, ignore_site_kinds: bool = False) -> dict[str, Any]:
    arg, cell = _coerce_cell3d(net_or_cell)
    sup = [int(x) for x in supercell] if supercell is not None else None
    if (res := _rs_json("flat3d_build_platycosm_quotient_json", arg, platycosm, sup, bool(ignore_site_kinds))) is not None:
        return res
    res = py_platycosm(cell, lookup_platycosm(platycosm), tuple(sup) if sup else None, ignore_site_kinds)
    res["supercell"] = res["supercell"] or [1, 1, 1]
    return {k: v for k, v in res.items() if k not in ("compatible", "origin_shift")}


def build_slab_3d(net_or_cell: str | dict[str, Any], kind: str = "TorusSlab", repeats: Any = (3, 3, 3)) -> dict[str, Any]:
    arg, cell = _coerce_cell3d(net_or_cell)
    rep = [int(x) for x in repeats]
    if (res := _rs_json("flat3d_build_slab_3d_json", arg, kind, rep)) is not None:
        return res
    meta_map = {"torusslab": ("TorusSlab", 2, 1, True, "Z^2", 2), "kleinslab": ("KleinSlab", 2, 1, False, "Z_2 x Z", 1), "moebiustube": ("MoebiusTube", 2, 1, False, "Z^2", 2), "mobiustube": ("MoebiusTube", 2, 1, False, "Z^2", 2), "cylindercrosss1": ("CylinderCrossS1", 2, 1, True, "Z^2", 2), "cylinderchimney": ("CylinderChimney", 1, 2, True, "Z", 1)}
    cname, cd, nd, ori, h1, b1 = meta_map["".join(c.lower() for c in kind if c.isalnum())]
    lat = np.asarray(cell["lattice"], float)
    fb = np.linalg.solve(lat.T, np.asarray(cell["basis"], float).T).T
    l1, l2, l3 = rep
    if cname in ("KleinSlab", "MoebiusTube"):
        glide = find_inplane_glide(cell, fb)
        if glide is None:
            return {"kind": cname, "status": "IMPOSSIBLE", "reason": f"IMPOSSIBLE [PROVED]: {cname} is glued by a z-preserving in-plane mirror/glide (x, y, z) -> (M(x, y) + t, z) of the crystal; exhaustive search over M with entries in [-2, 2] in the given a1, a2 basis (a3 fixed) finds none for this cell setting", "manifold_dim": 3, "compact_dims": cd, "noncompact_dims": nd, "orientable": ori, "h1_label": h1, "b1": b1, "repeats": rep, "positions": [], "flat_positions": [], "bonds": [], "site_kinds": [], "boundary_sites": [], "periodic_vectors": []}
        m2, t_g, perm, _ = glide
        if not (np.array_equal(m2, [[1, 0], [0, -1]]) and np.allclose(t_g, 0.0, atol=1e-6) and perm == list(range(len(fb))) and np.allclose(fb[:, 1], np.round(fb[:, 1]), atol=1e-6)):
            raise ValueError(f"REFUSED: the pure-Python {cname} fallback only glues by the mirror y -> -y fixing every basis site; this cell needs papertilling_rs")
    sid = lambda b, x, y, z: ((b * l1 + x) * l2 + y) * l3 + z
    flat = [(np.array([fb[b, 0] + ix, fb[b, 1] + iy - (0.5 * l2 if cname in ("CylinderCrossS1", "CylinderChimney", "MoebiusTube") else 0.0), fb[b, 2] + iz - (0.5 * l3 if cname in ("TorusSlab", "KleinSlab", "CylinderChimney") else 0.0)]) @ lat).tolist() for b in range(len(fb)) for ix in range(l1) for iy in range(l2) for iz in range(l3)]
    sk = [int(cell.get("site_kinds", [0] * len(fb))[b]) for b in range(len(fb)) for _ in range(l1 * l2 * l3)]
    px, py, pz = cname in ("TorusSlab", "KleinSlab", "MoebiusTube", "CylinderCrossS1", "CylinderChimney"), cname in ("TorusSlab", "KleinSlab"), cname in ("MoebiusTube", "CylinderCrossS1")
    bonds_s, seen_p, deg = [], set(), [0] * len(flat)
    for ix, iy, iz in np.ndindex(l1, l2, l3):
        for bd in cell["bonds"]:
            bi, bj, n1, n2, n3 = [int(x) for x in bd[:5]]
            u, tx, ty, tz = sid(bi, ix, iy, iz), ix + n1, iy + n2, iz + n3
            if (not px and not 0 <= tx < l1) or (not py and not 0 <= ty < l2) or (not pz and not 0 <= tz < l3):
                continue
            y_wrap = (l2 - 1 - (ty % l2)) if (cname in ("KleinSlab", "MoebiusTube") and (tx // l1) % 2 != 0) else (ty % l2)
            v = sid(bj, tx % l1, y_wrap, tz % l3)
            if u != v and (min(u, v), max(u, v)) not in seen_p:
                seen_p.add((min(u, v), max(u, v)))
                bonds_s.append([min(u, v), max(u, v), tx // l1 != 0 or ty // l2 != 0 or tz // l3 != 0, int(bd[5]) if len(bd) > 5 else 0])
                deg[u] += 1; deg[v] += 1
    md = max(deg, default=0)
    return {"kind": cname, "status": "OK", "reason": f"OK: {cname}", "manifold_dim": 3, "compact_dims": cd, "noncompact_dims": nd, "orientable": ori, "h1_label": h1, "b1": b1, "repeats": rep, "positions": flat, "flat_positions": flat, "bonds": bonds_s, "site_kinds": sk, "boundary_sites": [i for i, d in enumerate(deg) if d < md], "periodic_vectors": []}


def build_nanotube(cell: Any, n: int, m: int, periods_z: int = 1, periodic_z: bool = False, standard_hex_60: bool = False) -> dict[str, Any]:
    c = _coerce_cell2d(cell)
    if (res := _rs_json("flat3d_build_nanotube_json", json.dumps(c), int(n), int(m), int(periods_z), bool(periodic_z), bool(standard_hex_60))) is not None:
        return res
    a1, a2 = np.asarray(c["a1"], float), np.asarray(c["a2"], float)
    if standard_hex_60 and abs(np.linalg.norm(a1) - np.linalg.norm(a2)) <= 1e-5 * np.linalg.norm(a1) and abs(float(np.dot(a1, a2)) / (np.linalg.norm(a1) * np.linalg.norm(a2)) + 0.5) <= 1e-5:
        c = build_supercell_2d(c, [[1, 1], [0, 1]])
        a1, a2 = np.asarray(c["a1"], float), np.asarray(c["a2"], float)
    ch = n * a1 + m * a2
    ch_len, alpha, beta, best = float(np.linalg.norm(ch)), float(np.dot(ch, a1)), float(np.dot(ch, a2)), None
    if abs(alpha) <= 1e-8 * ch_len * ch_len:
        best = (1, 0, float(np.linalg.norm(a1)))
    elif abs(beta) <= 1e-8 * ch_len * ch_len:
        best = (0, 1, float(np.linalg.norm(a2)))
    else:
        r = -alpha / beta
        for t1 in [s * x for x in range(1, 129) for s in (1, -1)]:
            t2 = round(r * t1)
            if math.gcd(abs(t1), abs(t2)) == 1:
                tv = t1 * a1 + t2 * a2
                tl = float(np.linalg.norm(tv))
                if tl > 1e-12 and abs(float(np.dot(ch, tv)) / (ch_len * tl)) <= 1e-6 and (best is None or tl < best[2] - 1e-9):
                    best = (t1, t2, tl)
    if best is None:
        raise ValueError(f"REFUSED [COMPUTATIONAL EVIDENCE]: no primitive lattice period T = t1*a1 + t2*a2 with |t1| <= 128 is orthogonal to Ch = ({n}, {m}) within cos 1e-6; a 1D-periodic tube exists iff (Ch.a1)/(Ch.a2) is rational, which floating-point input cannot certify")
    t1, t2 = (-best[0], -best[1]) if n * best[1] - m * best[0] < 0 else (best[0], best[1])
    det_m, tr = n * t2 - m * t1, t1 * a1 + t2 * a2
    per_len, rad = float(np.linalg.norm(tr)), ch_len / (2.0 * math.pi)
    c_hat, t_hat = ch / ch_len, tr / per_len
    cosets = {((t2 * k1 - t1 * k2) % det_m, (-m * k1 + n * k2) % det_m): ((n * ((t2 * k1 - t1 * k2) % det_m) + t1 * ((-m * k1 + n * k2) % det_m)) // det_m, (m * ((t2 * k1 - t1 * k2) % det_m) + t2 * ((-m * k1 + n * k2) % det_m)) // det_m)
              for k1 in range(min(0, n, t1, n + t1) - 2, max(0, n, t1, n + t1) + 3) for k2 in range(min(0, m, t2, m + t2) - 2, max(0, m, t2, m + t2) + 3)}
    usites = [(i, *cosets[ck], ck) for ck in sorted(cosets) for i in range(len(c["basis"]))]
    umap = {(i, ck[0], ck[1]): idx for idx, (i, _, _, ck) in enumerate(usites)}
    apu = len(usites)
    p2, p3, sb = [], [], []
    for p in range(periods_z):
        for i, r1, r2, _ in usites:
            rv = r1 * a1 + r2 * a2 + np.asarray(c["basis"][i], float) + p * tr
            s, z = float(np.dot(rv, c_hat)), float(np.dot(rv, t_hat))
            th = 2.0 * math.pi * s / ch_len
            p2.append([s, z]); p3.append([rad * math.cos(th), rad * math.sin(th), z]); sb.append(i)
    seen_b, bonds, deg = set(), [], [0] * (apu * periods_z)
    for p in range(periods_z):
        for uu, (ib, k1, k2, _) in enumerate(usites):
            uid = p * apu + uu
            for bi, bj, n1, n2, bk in c["bonds"]:
                for src, dst, dn1, dn2 in ((bi, bj, n1, n2), (bj, bi, -n1, -n2)):
                    if src == ib:
                        kt1, kt2 = k1 + dn1, k2 + dn2
                        vu = umap[(dst, (t2 * kt1 - t1 * kt2) % det_m, (-m * kt1 + n * kt2) % det_m)]
                        _, kw1, kw2, _ = usites[vu]
                        dv, du = (-m * (kt1 - kw1) + n * (kt2 - kw2)) // det_m, (t2 * (kt1 - kw1) - t1 * (kt2 - kw2)) // det_m
                        tp_raw = p + dv
                        if periodic_z or 0 <= tp_raw < periods_z:
                            tp = tp_raw % periods_z if periodic_z else tp_raw
                            vid, seam = tp * apu + vu, du != 0 or (periodic_z and tp_raw != tp)
                            key = (uid, vid, du, dv, bk) if (uid, du, dv) <= (vid, -du, -dv) else (vid, uid, -du, -dv, bk)
                            if key not in seen_b:
                                seen_b.add(key); bonds.append([key[0], key[1], seam, bk]); deg[key[0]] += 1; deg[key[1]] += 1
    return {"n": n, "m": m, "t_vec": [t1, t2], "chiral_vec": ch.tolist(), "trans_vec": tr.tolist(), "circumference": ch_len, "radius": rad, "period_length": per_len, "chiral_angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ch, a1)) / (ch_len * float(np.linalg.norm(a1))))))), "atoms_per_unit": apu, "periods_z": periods_z, "periodic_z": periodic_z, "positions_2d": p2, "positions_3d": p3, "bonds": bonds, "site_basis_idx": sb, "coordination_numbers": deg}

def build_nanoribbon(cell: Any, chiral_dir: list[int] | tuple[int, int], width_cells: int, length_cells: int, periodic_along: bool = False) -> dict[str, Any]:
    c = _coerce_cell2d(cell)
    if (res := _rs_json("flat3d_build_nanoribbon_json", json.dumps(c), [int(x) for x in chiral_dir], int(width_cells), int(length_cells), bool(periodic_along))) is not None:
        return res
    c1, c2 = int(chiral_dir[0]), int(chiral_dir[1]); g = math.gcd(abs(c1), abs(c2)); l1, l2 = c1 // g, c2 // g
    w1, w2 = next((a, b) for a in range(-20, 21) for b in range(-20, 21) if l1 * b - l2 * a == 1)
    rib = build_supercell_2d(c, [[c1, w1], [c2, w2]])
    m, a1, a2 = len(rib["basis"]), np.asarray(rib["a1"], float), np.asarray(rib["a2"], float)
    sid = lambda b, u, v: (u * width_cells + v) * m + b
    p2 = [(np.asarray(pt, float) + u * a1 + v * a2).tolist() for u in range(length_cells) for v in range(width_cells) for pt in rib["basis"]]
    seen_b, bonds, deg = set(), [], [0] * len(p2)
    for u, v in np.ndindex(length_cells, width_cells):
        for bi, bj, n1, n2, bk in rib["bonds"]:
            tv, tu_raw = v + n2, u + n1
            if 0 <= tv < width_cells and (periodic_along or 0 <= tu_raw < length_cells):
                tu, iu, iv = (tu_raw % length_cells if periodic_along else tu_raw), sid(bi, u, v), sid(bj, tu_raw % length_cells if periodic_along else tu_raw, tv)
                du = tu_raw // length_cells
                key = (iu, iv, du, bk) if (iu, du) <= (iv, -du) else (iv, iu, -du, bk)
                if key not in seen_b:
                    seen_b.add(key); bonds.append([key[0], key[1], periodic_along and tu_raw != tu, bk]); deg[key[0]] += 1; deg[key[1]] += 1
    md = max(deg, default=0)
    return {"chiral_dir": [c1, c2], "width_dir": [w1, w2], "width_cells": width_cells, "length_cells": length_cells, "periodic_along": periodic_along, "period_vec": (length_cells * a1).tolist(), "positions_2d": p2, "positions_3d": [[q[0], q[1], 0.0] for q in p2], "bonds": bonds, "site_basis_idx": [b % len(c["basis"]) for _ in range(length_cells * width_cells) for b in range(m)], "edge_sites": [i for i, d in enumerate(deg) if d < md]}


_MIRRORLESS_2D = {"p1", "p2", "pg", "pgg", "p3", "p4", "p6"}
_MAX_ROT = {"p1": 1, "pm": 1, "pg": 1, "cm": 1, "p2": 2, "pmm": 2, "pmg": 2, "pgg": 2, "cmm": 2, "p3": 3, "p3m1": 3, "p31m": 3, "p4": 4, "p4m": 4, "p4g": 4, "p6": 6, "p6m": 6}


def _py_flake(group: str, cell_d: dict[str, Any], centre: list[float], radius: float, n_poly: int | None, hps: list[tuple[np.ndarray, float]] | None) -> dict[str, Any]:
    uc = UnitCell("c", "c", group or "p1", np.asarray(cell_d["a1"]), np.asarray(cell_d["a2"]), np.asarray(cell_d["basis"]), [tuple(b) for b in cell_d["bonds"]])
    a, a_inv, cen = uc.A, np.linalg.inv(uc.A), np.asarray(centre, float)
    no_mirrors, max_rot = (group in _MIRRORLESS_2D, _MAX_ROT.get(group, 6)) if group else (False, 6)
    rots, refls = [np.eye(2)], []
    for mf, tf in _symmetry_ops(uc, use_types=False):
        m, t = a @ np.asarray(mf, float) @ a_inv, a @ np.asarray(tf, float)
        det, f = float(np.linalg.det(m)), a_inv @ (cen - (m @ cen + t))
        if (no_mirrors and det < 0) or not np.all(np.abs(f - np.round(f)) < 1e-4):
            continue
        if det > 0:
            ord_r = 1 if abs(math.atan2(m[1, 0], m[0, 0])) < 1e-5 else round(2 * math.pi / abs(math.atan2(m[1, 0], m[0, 0])))
            if (ord_r == 1 or max_rot % ord_r == 0) and not any(np.allclose(m, k, atol=1e-5) for k in rots):
                rots.append(m)
        elif not any(np.allclose(m, k, atol=1e-5) for k in refls):
            refls.append(m)
    mats = rots + ([m for m in refls if any(np.allclose(m @ refls[0], r, atol=1e-4) for r in rots)] if refls else [])
    if n_poly is not None and hps is None:
        m_ref = next((m for m in mats if np.linalg.det(m) < 0), None)
        th0 = math.atan2(m_ref[1, 0] if abs(m_ref[0, 0] + 1) + abs(m_ref[1, 0]) > 0.5 else m_ref[1, 1] + 1, m_ref[0, 0] + 1 if abs(m_ref[0, 0] + 1) + abs(m_ref[1, 0]) > 0.5 else m_ref[0, 1]) if m_ref is not None else 0.0
        hps = [(np.array([math.cos(th0 + 2 * math.pi * k / n_poly), math.sin(th0 + 2 * math.pi * k / n_poly)]), radius) for k in range(n_poly)]
    if hps is not None:
        mats = [m for m in mats if all(any(np.linalg.norm(m @ u - u2) < 1e-4 and abs(d - d2) < 1e-4 for u2, d2 in hps) for u, d in hps)]
    inside = (lambda rel: all(float(np.dot(rel, u)) <= d + 1e-6 for u, d in hps)) if hps is not None else (lambda rel: float(np.linalg.norm(rel)) <= radius + 1e-6)
    n_max, cf = int(math.ceil((radius + float(np.linalg.norm(uc.a1)) + float(np.linalg.norm(uc.a2)) + 2.0) * float(np.linalg.norm(a_inv)))) + 2, np.round(a_inv @ cen).astype(int)
    kept = {(i, n1, n2): bi + n1 * uc.a1 + n2 * uc.a2 for n1 in range(int(cf[0]) - n_max, int(cf[0]) + n_max + 1) for n2 in range(int(cf[1]) - n_max, int(cf[1]) + n_max + 1) for i, bi in enumerate(uc.basis) if inside(bi + n1 * uc.a1 + n2 * uc.a2 - cen)}
    keys = sorted(kept)
    idx, pos = {k: u for u, k in enumerate(keys)}, [kept[k].tolist() for k in keys]
    edges, adj = set(), [[] for _ in keys]
    for u, (i, n1, n2) in enumerate(keys):
        for bi, bj, d1, d2, bk in uc.bonds:
            for src, dst, s1, s2 in ((bi, bj, d1, d2), (bj, bi, -d1, -d2)):
                if src == i and (dst, n1 + s1, n2 + s2) in idx and (v := idx[(dst, n1 + s1, n2 + s2)]) != u:
                    adj[u].append({"to": v, "vec": (np.asarray(pos[v]) - pos[u]).tolist(), "kind": bk, "seam": None, "ang": None, "across": False, "ghost_stub": None})
                    edges.add((min(u, v), max(u, v)))
    perms = [[next(w for w, pw in enumerate(pos) if np.linalg.norm((cen + m @ (np.asarray(pu) - cen)) - pw) < 1e-4) for pu in pos] for m in mats]
    orbs, seen_o = [], set()
    for u in range(len(pos)):
        if u not in seen_o:
            orb = sorted({p[u] for p in perms}); seen_o.update(orb); orbs.append(orb)
    md = max((len(a_u) for a_u in adj), default=0)
    return {"group": group, "centre": list(centre), "radius": radius, "n_fold_polygon": n_poly, "point_group_order": len(mats), "is_dihedral": any(np.linalg.det(m) < 0 for m in mats), "crystal": {"pos": pos, "ref_pos": pos, "basis_idx": [k[0] for k in keys], "side": [None] * len(keys), "cone_sites": [], "edges": sorted(edges), "adj": adj, "wall_sigma": None, "is_non_orientable": False}, "site_orbits": orbs, "vertex_perms": perms, "boundary_sites": [u for u, a_u in enumerate(adj) if len(a_u) < md]}


def build_symmetric_flake(group: str, cell: Any, centre: Any, n_fold_polygon: int | None = None, radius: float = 2.5) -> dict[str, Any]:
    c = _coerce_cell2d(cell)
    if (res := _rs_json("flat3d_build_symmetric_flake_json", group, json.dumps(c), [float(centre[0]), float(centre[1])], n_fold_polygon, float(radius))) is not None:
        return res
    return _py_flake(group, c, [float(centre[0]), float(centre[1])], float(radius), n_fold_polygon, None)


def build_wigner_seitz_flake(group: str, cell: Any, centre: Any, scale: float = 1.5) -> dict[str, Any]:
    c = _coerce_cell2d(cell)
    if (res := _rs_json("flat3d_build_wigner_seitz_flake_json", group, json.dumps(c), [float(centre[0]), float(centre[1])], float(scale))) is not None:
        return res
    a1, a2 = np.asarray(c["a1"], float), np.asarray(c["a2"], float)
    vecs = sorted([n1 * a1 + n2 * a2 for n1 in range(-2, 3) for n2 in range(-2, 3) if n1 or n2], key=lambda v: float(np.linalg.norm(v)))
    hps = [(v / float(np.linalg.norm(v)), 0.5 * float(np.linalg.norm(v)) * scale) for v in vecs[:12]]
    return _py_flake(group, c, [float(centre[0]), float(centre[1])], max(d for _, d in hps), None, hps)


def niggli_reduce_3d(lattice: Any) -> dict[str, Any]:
    lat = np.asarray(lattice, float).tolist()
    if (res := _rs_json("flat3d_niggli_reduce_3d_json", json.dumps(lat))) is not None:
        return res
    return py_niggli(lat)


def _supercell_cosets(m_int: np.ndarray, d: int) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], int], Any]:
    diag, u_m, _ = _snf(m_int.tolist(), True)
    u_inv = np.round(np.linalg.inv(np.asarray(u_m, float))).astype(int)
    adj = np.round(np.linalg.inv(m_int.astype(float)) * d).astype(int)
    norm_c = lambda o: (tuple(((m_int @ ((adj @ np.asarray(o, int)) % d)) // d).tolist()), ((adj @ np.asarray(o, int)) // d).tolist())
    ranges = [range(x) for x in diag]
    offs = [norm_c((u_inv @ np.array(c)).tolist())[0] for c in np.ndindex(*[r.stop for r in ranges])]
    return offs, {o: i for i, o in enumerate(offs)}, norm_c


def build_supercell_3d(net_or_cell: str | dict[str, Any], celltran: list[list[int]]) -> dict[str, Any]:
    arg, c = _coerce_cell3d(net_or_cell)
    mat = [[int(x) for x in r] for r in celltran]
    if (res := _rs_json("flat3d_build_supercell_3d_json", arg, json.dumps(mat))) is not None:
        return {**res, "bonds": _bond_rows(res["bonds"], 3)}
    d = _det(mat)
    if d <= 0:
        raise ValueError("3D CELLTRAN must have det > 0")
    offs, omap, norm_c = _supercell_cosets(np.asarray(mat, int).T, d)
    lat, m = np.asarray(c["lattice"], float), len(c["basis"])
    bs = {(u, v, q[0], q[1], q[2], bk) if (u, v, q[0], q[1], q[2]) <= (v, u, -q[0], -q[1], -q[2]) else (v, u, -q[0], -q[1], -q[2], bk)
          for oi, o in enumerate(offs) for bd in c["bonds"]
          for tr, q in [norm_c([o[0] + int(bd[2]), o[1] + int(bd[3]), o[2] + int(bd[4])])]
          for u, v, bk in [(oi * m + int(bd[0]), omap[tr] * m + int(bd[1]), int(bd[5]) if len(bd) > 5 else 0)]}
    return {"lattice": (np.asarray(mat, float) @ lat).tolist(), "basis": [(np.asarray(p) + np.asarray(o) @ lat).tolist() for o in offs for p in c["basis"]], "bonds": [list(x) for x in sorted(bs)], "space_group_num": c.get("space_group_num") if d == 1 else None, "site_kinds": [int(c.get("site_kinds", [0] * m)[i]) for _ in offs for i in range(m)]}


def build_supercell_2d(cell: Any, celltran: list[list[int]]) -> dict[str, Any]:
    c, mat = _coerce_cell2d(cell), [[int(x) for x in r] for r in celltran]
    if (res := _rs_json("flat3d_build_supercell_2d_json", json.dumps(c), json.dumps(mat))) is not None:
        return {**res, "bonds": _bond_rows(res["bonds"], 2)}
    if (d := _det(mat)) <= 0:
        raise ValueError("2D CELLTRAN must have det > 0")
    offs, omap, norm_c = _supercell_cosets(np.asarray(mat, int), d)
    a1, a2, m = np.asarray(c["a1"], float), np.asarray(c["a2"], float), len(c["basis"])
    bs = {(u, v, q[0], q[1], bk) if (u, q[0], q[1]) <= (v, -q[0], -q[1]) else (v, u, -q[0], -q[1], bk)
          for oi, o in enumerate(offs) for bi, bj, n1, n2, bk in c["bonds"]
          for tr, q in [norm_c([o[0] + n1, o[1] + n2])] for u, v in [(oi * m + bi, omap[tr] * m + bj)]}
    return {"a1": (mat[0][0] * a1 + mat[1][0] * a2).tolist(), "a2": (mat[0][1] * a1 + mat[1][1] * a2).tolist(), "basis": [(np.asarray(p) + o[0] * a1 + o[1] * a2).tolist() for o in offs for p in c["basis"]], "bonds": [list(x) for x in sorted(bs)]}

def cell_strain_tensor(reference: Any, deformed: Any) -> dict[str, Any]:
    r, d = np.asarray(reference, float).tolist(), np.asarray(deformed, float).tolist()
    if (res := _rs_json("flat3d_cell_strain_tensor_json", json.dumps(r), json.dumps(d))) is not None:
        return res
    f = np.asarray(d, float).T @ np.linalg.inv(np.asarray(r, float).T)
    e_gl = 0.5 * (f.T @ f - np.eye(3))
    dev = e_gl - (np.trace(e_gl) / 3.0) * np.eye(3)
    return {"deformation_gradient": f.tolist(), "green_lagrange": e_gl.tolist(), "infinitesimal_strain": (0.5 * (f + f.T) - np.eye(3)).tolist(), "volumetric_strain": float(np.linalg.det(f) - 1.0), "von_mises_equivalent": math.sqrt(2.0 * float(np.sum(dev * dev)) / 3.0)}

def _norm_bcs(bcs: list[Any]) -> list[dict[str, Any]]:
    return [{"kind": bc, "phase_rad": 0.0} if isinstance(bc, str) else ({"kind": str(bc[0]), "phase_rad": float(bc[1])} if isinstance(bc, (tuple, list)) else {"kind": str(bc.get("kind", next(iter(bc)))), "phase_rad": float(bc.get("phase_rad", bc.get("Twisted", {}).get("phase_rad", 0.0)))}) for bc in bcs]

def _apply_bcs_nd(basis: list[Any], vectors: np.ndarray, bonds_in: list[Any], repeats: list[int], bcs: list[Any], site_kinds: list[int] | None) -> dict[str, Any]:
    dim, m, nbcs = len(repeats), len(basis), _norm_bcs(bcs)
    step = lambda bc, w: None if (bc["kind"].lower() == "open" and w != 0) else (w * math.pi if bc["kind"].lower() == "antiperiodic" else (w * bc["phase_rad"] if bc["kind"].lower() == "twisted" else 0.0))
    cells = list(np.ndindex(*repeats))
    sid = lambda b, c_idx: b * len(cells) + (c_idx[0] * repeats[1] + c_idx[1] if dim == 2 else (c_idx[0] * repeats[1] + c_idx[1]) * repeats[2] + c_idx[2])
    pos = [None] * (m * len(cells)); sk = [0] * len(pos)
    for c_idx in cells:
        sh = np.asarray(c_idx, float) @ vectors
        for b, p in enumerate(basis):
            pos[sid(b, c_idx)] = (np.asarray(p, float) + sh).tolist()
            if site_kinds is not None:
                sk[sid(b, c_idx)] = int(site_kinds[b])
    seen, bonds = set(), []
    for c_idx in cells:
        for bd in bonds_in:
            bi, bj, ns, bk = int(bd[0]), int(bd[1]), [int(bd[2 + k]) for k in range(dim)], (int(bd[2 + dim]) if len(bd) > 2 + dim else 0)
            w = [(c_idx[k] + ns[k]) // repeats[k] for k in range(dim)]
            ps = [step(nbcs[k], w[k]) for k in range(dim)]
            if any(x is None for x in ps):
                continue
            u, v, ph = sid(bi, c_idx), sid(bj, tuple((c_idx[k] + ns[k]) % repeats[k] for k in range(dim))), float(sum(ps))
            a, b, wr, ph = (u, v, tuple(w), ph) if (u, tuple(w)) <= (v, tuple(-x for x in w)) else (v, u, tuple(-x for x in w), -ph)
            if (a, b, wr) not in seen:
                seen.add((a, b, wr)); bonds.append({"u": a, "v": b, "kind": bk, "wrap": list(wr), "phase_rad": ph, "weight_real": math.cos(ph)})
    fmt_bcs = [{"Twisted": {"phase_rad": bc["phase_rad"]}} if bc["kind"].lower() == "twisted" else bc["kind"].capitalize() for bc in nbcs]
    return {"positions": pos, "bonds": bonds, "bcs": fmt_bcs, **({"site_kinds": sk} if site_kinds is not None else {})}

def apply_boundary_conditions_3d(net_or_cell: str | dict[str, Any], repeats: Any, bcs: list[Any]) -> dict[str, Any]:
    arg, c = _coerce_cell3d(net_or_cell); rep = [int(x) for x in repeats]
    return res if (res := _rs_json("flat3d_apply_boundary_conditions_3d_json", arg, rep, json.dumps(_norm_bcs(bcs)))) is not None else _apply_bcs_nd(c["basis"], np.asarray(c["lattice"], float), c["bonds"], rep, bcs, c.get("site_kinds", [0] * len(c["basis"])))

def apply_boundary_conditions_2d(cell: Any, repeats: Any, bcs: list[Any]) -> dict[str, Any]:
    c, rep = _coerce_cell2d(cell), [int(x) for x in repeats]
    return res if (res := _rs_json("flat3d_apply_boundary_conditions_2d_json", json.dumps(c), rep, json.dumps(_norm_bcs(bcs)))) is not None else _apply_bcs_nd(c["basis"], np.asarray([c["a1"], c["a2"]], float), c["bonds"], rep, bcs, None)
