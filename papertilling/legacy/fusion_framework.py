"""
fusion_framework.py — Minimal lattices, proper-symmetry defect catalogues, cone-safe multi-defect
combination, constructive defect fusion rules for all 17 wallpaper groups (C10), and non-orientable
Möbius-strip & closed Klein-bottle braiding and cross-seam fusion for the 12 achiral groups (C11).

Built on top of:
  - wallpaper_lattices.py  (UnitCell, WallpaperLattice, make_cell, primitive_cell, identify_wallpaper_group)
  - minimal_lattices.py    (load_catalogue, enumerate_lattices, rebuild, recipe_code)
  - lab_defects.py         (Perfect, Defect, dislocation, disclination, mobius_strip, interface,
                            catalogue, dislocation_cores, dislocation_dipole, disclination_pair,
                            disclination_quadrupole, frame_faces, loop_charge, euler_characteristic)

Uses only Python stdlib + numpy, scipy, networkx, matplotlib.

Conventions (fixed in Problems/Defects on wallpaper lattices):
  - An isometry is (M, t): x -> M x + t in Cartesian coordinates.
  - Bond colours are ignored when determining symmetry group G and proper subgroup G+ (det M = +1).
  - Burgers vector (K5): sum of perfect-lattice bond vectors round a counter-clockwise loop.
    Outer CCW holonomy of a pure dislocation with K5 Burgers vector b is H = (I, -b).
  - Disclination Frank angle alpha (degrees): alpha > 0 removes a wedge of angle alpha;
    alpha < 0 inserts a wedge of angle |alpha|. Outer CCW holonomy about c is
    H = (R_alpha, (I - R_alpha) c).
  - Proper topological charge: conjugacy class in the universal cover \\widetilde{G}_+ < R |x R^2
    (unreduced real Frank angle alpha in (-inf, 360) together with G+-conjugacy class of H).
    Improper mirror/glide m0 = (M0, s0) (det M0 = -1) induces the exact involution
    mu(alpha, H) = (+alpha, m0 o H^-1 o m0^-1), mapping c -> m0(c) and b -> -M0 b.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import os
import warnings
from collections import Counter, deque
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

try:
    from . import wallpaper_lattices as _wl_probe  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    for _cand in (
        Path(__file__).resolve().parent,
    ):
        if (_cand / "wallpaper_lattices.py").is_file() and str(_cand) not in _sys.path:
            _sys.path.insert(0, str(_cand))
            break

from .wallpaper_lattices import (
    WALLPAPER_GROUPS,
    PALETTE,
    UnitCell,
    WallpaperLattice,
    available_lattices,
    has_crossings,
    identify_wallpaper_group,
    make_cell,
    primitive_cell,
    _symmetry_ops,
)
from .minimal_lattices import (
    Entry,
    enumerate_lattices,
    load_catalogue,
    plot_entries,
    rebuild,
    recipe_code,
)
from .lab_defects import (
    Defect,
    Perfect,
    catalogue as full_group_catalogue,
    disclination,
    disclination_pair,
    disclination_quadrupole,
    dislocation,
    dislocation_cores,
    dislocation_dipole,
    euler_characteristic,
    frame_faces,
    interface,
    iso_apply,
    iso_compose,
    iso_inverse,
    loop_charge,
    mobius_strip,
    reflection_across,
    rot,
    rotation_about,
    TWO_PI,
)

_HERE = Path(__file__).resolve().parent
_CONSTRUCTING_DIR = _HERE


def _find_default_min_json() -> Path:
    for _p in (
        _HERE / "data" / "minimal_lattices.json",
        _HERE / "Lab reports" / "minimal_lattices.json",
        _CONSTRUCTING_DIR / "minimal_lattices.json",
        _CONSTRUCTING_DIR / "Lab reports" / "minimal_lattices.json",
        _CONSTRUCTING_DIR.parent / "minimal_lattices.json",
    ):
        if _p.exists():
            return _p
    return _HERE / "data" / "minimal_lattices.json"


_DEFAULT_MIN_JSON = _find_default_min_json()
_MIN_CAT_CACHE: dict[str, dict] | None = None
_MINIMAL_LAB_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


class ConfigError(ValueError):
    """Raised when a requested defect or multi-defect combination violates geometric or topological rules."""


class _SeamHit(Exception):
    pass


# ==============================================================================================
# Fast 2D Euclidean primitives (avoids numpy dispatch overhead in inner graph-development loops)
# ==============================================================================================
def _cross2(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _cross2v(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def _norm2(v: np.ndarray) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _ident() -> tuple[np.ndarray, np.ndarray]:
    return np.eye(2), np.zeros(2)


def _same_iso(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], tol: float = 1e-6) -> bool:
    A0, at = a[0], a[1]
    B0, bt = b[0], b[1]
    return (
        abs(float(A0[0, 0]) - float(B0[0, 0])) <= tol
        and abs(float(A0[0, 1]) - float(B0[0, 1])) <= tol
        and abs(float(A0[1, 0]) - float(B0[1, 0])) <= tol
        and abs(float(A0[1, 1]) - float(B0[1, 1])) <= tol
        and abs(float(at[0]) - float(bt[0])) <= tol
        and abs(float(at[1]) - float(bt[1])) <= tol
    )


def _ray_distance(y: np.ndarray, p: np.ndarray, u: np.ndarray) -> float:
    rx = float(y[0]) - float(p[0])
    ry = float(y[1]) - float(p[1])
    a = rx * float(u[0]) + ry * float(u[1])
    if a < 0.0:
        return math.hypot(rx, ry)
    return abs(float(u[0]) * ry - float(u[1]) * rx)


def _fast_locate(p: np.ndarray, Ainv: np.ndarray, basis: np.ndarray, tol: float = 1e-6) -> tuple[int, int, int] | None:
    px, py = float(p[0]), float(p[1])
    a00, a01 = float(Ainv[0, 0]), float(Ainv[0, 1])
    a10, a11 = float(Ainv[1, 0]), float(Ainv[1, 1])
    for s in range(len(basis)):
        dx = px - float(basis[s, 0])
        dy = py - float(basis[s, 1])
        f0 = a00 * dx + a01 * dy
        f1 = a10 * dx + a11 * dy
        r0 = round(f0)
        if abs(f0 - r0) <= tol:
            r1 = round(f1)
            if abs(f1 - r1) <= tol:
                return (int(r0), int(r1), s)
    return None


def _r(x: Any, n: int = 3) -> Any:
    arr = np.round(np.asarray(x, float), n) + 0.0
    return arr.tolist() if arr.ndim else float(arr)


# ==============================================================================================
# 1. Minimal-lattice loader for all 17 wallpaper groups
# ==============================================================================================
def load_minimal_catalogue(json_path: str | Path | None = None) -> dict[str, dict]:
    """Load precomputed minimal lattices for all 17 wallpaper groups (cached in memory)."""
    global _MIN_CAT_CACHE
    if json_path is None and _MIN_CAT_CACHE is not None:
        return _MIN_CAT_CACHE
    path = Path(json_path) if json_path is not None else _DEFAULT_MIN_JSON
    if not path.exists():
        raise FileNotFoundError(f"minimal_lattices.json not found at {path}")
    cat, _ = load_catalogue(str(path))
    if json_path is None:
        _MIN_CAT_CACHE = cat
    return cat


def minimal_lattice_entries(group: str, kind: str = "planar", n: int = 5, json_path: str | Path | None = None) -> list[Entry]:
    """Return ranked minimal-lattice Entry objects for wallpaper `group` ('planar', 'crossing', or 'all')."""
    if group not in WALLPAPER_GROUPS:
        raise ValueError(f"Unknown wallpaper group {group!r}; choose from {list(WALLPAPER_GROUPS)}")
    try:
        cat = load_minimal_catalogue(json_path)
        rec = cat[group]
        if kind == "planar":
            entries = list(rec["planar"])
        elif kind == "crossing":
            entries = list(rec["crossing"])
        else:
            entries = list(rec["planar"]) + list(rec["crossing"])
        if entries:
            return entries[:n]
    except FileNotFoundError:
        pass
    res = enumerate_lattices(group, n=n, samples=25, time_limit=60)
    if kind == "planar":
        return list(res["planar"])[:n]
    if kind == "crossing":
        return list(res["crossing"])[:n]
    return (list(res["planar"]) + list(res["crossing"]))[:n]


def minimal_cell(group: str, rank: int = 1, planar_only: bool = True, json_path: str | Path | None = None) -> tuple[UnitCell, Entry]:
    """Return `(primitive_unit_cell, entry)` for the `rank`-th minimal lattice of wallpaper `group`."""
    kind = "planar" if planar_only else "all"
    entries = minimal_lattice_entries(group, kind=kind, n=max(rank, 5), json_path=json_path)
    if not (1 <= rank <= len(entries)):
        raise ValueError(f"Requested rank={rank}, but {group} has {len(entries)} {kind} minimal entries")
    entry = entries[rank - 1]
    cell = rebuild(entry)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prim = primitive_cell(cell, use_types=False)
    # Deduplicate parallel bond colours on the bare graph
    seen, bonds = set(), []
    for bd in prim.bonds:
        if bd[:4] not in seen:
            seen.add(bd[:4])
            bonds.append(bd)
    prim = dataclasses.replace(prim, key=f"min_{group}_{rank}", label=f"Minimal {group} #{rank}", group=group, bonds=bonds)
    return prim, entry


# ==============================================================================================
# 2. Proper-group (G+) classification & mirror-partner involution
# ==============================================================================================
def _same_point_mod_lattice(p: np.ndarray, q: np.ndarray, Ainv: np.ndarray, tol: float = 1e-4) -> bool:
    f = Ainv @ (p - q)
    return bool(abs(float(f[0]) - round(float(f[0]))) < tol and abs(float(f[1]) - round(float(f[1]))) < tol)


def _reduce_point(p: np.ndarray, A: np.ndarray, Ainv: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    f = Ainv @ p
    f = f - np.floor(f + tol)
    f[np.abs(f - 1.0) < tol] = 0.0
    f[np.abs(f) < tol] = 0.0
    return A @ f


def _orbits(items: list[Any], ops: list[tuple[np.ndarray, np.ndarray]], image_fn, same_fn) -> list[list[Any]]:
    cls = list(range(len(items)))
    for i, it in enumerate(items):
        for g in ops:
            im = image_fn(it, g)
            for j, other in enumerate(items):
                if cls[j] != cls[i] and same_fn(im, other):
                    old, new = cls[j], cls[i]
                    cls = [new if c == old else c for c in cls]
    out: dict[int, list[Any]] = {}
    for i, c in enumerate(cls):
        out.setdefault(c, []).append(items[i])
    return list(out.values())


def same_proper_holonomy(
    H1: tuple[np.ndarray, np.ndarray],
    H2: tuple[np.ndarray, np.ndarray],
    ops_plus: list[tuple[np.ndarray, np.ndarray]],
    A: np.ndarray,
    Ainv: np.ndarray | None = None,
    tol: float = 1e-4,
) -> bool:
    """Are Euclidean holonomies H1 and H2 conjugate in the PROPER space group G+ (det M = +1)?"""
    if Ainv is None:
        Ainv = np.linalg.inv(A)
    for g in ops_plus:
        Hc = iso_compose(g, iso_compose(H1, iso_inverse(g)))
        if not np.allclose(Hc[0], H2[0], atol=tol):
            continue
        dlt = H2[1] - Hc[1]
        if np.allclose(Hc[0], np.eye(2), atol=tol):
            if _norm2(dlt) < tol:
                return True
            continue
        f = Ainv @ np.linalg.solve(np.eye(2) - Hc[0], dlt)
        if abs(float(f[0]) - round(float(f[0]))) < tol and abs(float(f[1]) - round(float(f[1]))) < tol:
            return True
    return False


def mirror_holonomy(
    alpha_deg: float,
    H: tuple[np.ndarray, np.ndarray],
    m0: tuple[np.ndarray, np.ndarray],
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """Apply the orientation-reversing mirror/glide involution mu_{m0}(alpha, H) = (+alpha, m0 o H^-1 o m0^-1).
    Preserves Frank angle (+alpha -> +alpha), maps centre c -> m0(c), and maps Burgers vector b -> -M0 b."""
    Hm = iso_compose(m0, iso_compose(iso_inverse(H), iso_inverse(m0)))
    return float(alpha_deg), Hm


class ProperCatalogue:
    """Complete proper-symmetry (G+) defect catalogue for a primitive UnitCell, with mirror-partner annotations."""

    def __init__(self, cell: UnitCell, bmax: float = 1.05, bmax_fusion: float = 3.6, angle_range: tuple[float, float] = (-180.0, 180.0)):
        self.base_cat = full_group_catalogue(cell, bmax=bmax, angle_range=angle_range)
        self.cell: UnitCell = self.base_cat.cell
        self.A: np.ndarray = self.cell.A
        self.Ainv: np.ndarray = np.linalg.inv(self.A)
        self.lattice: Perfect = self.base_cat.lattice
        self.ops: list[tuple[np.ndarray, np.ndarray]] = self.base_cat.ops
        self.ops_plus: list[tuple[np.ndarray, np.ndarray]] = [(M, t) for M, t in self.ops if np.linalg.det(M) > 0]
        self.ops_minus: list[tuple[np.ndarray, np.ndarray]] = [(M, t) for M, t in self.ops if np.linalg.det(M) < 0]
        self.chiral: bool = len(self.ops_minus) == 0
        self.m0: tuple[np.ndarray, np.ndarray] | None = self.ops_minus[0] if self.ops_minus else None

        # --- 1. Proper Burgers-vector shells & classes under P+ ---
        P_plus = [(M, np.zeros(2)) for M, _ in self.ops_plus]
        vecs = [
            self.A @ np.array([i, j], float)
            for i in range(-6, 7)
            for j in range(-6, 7)
            if (i, j) != (0, 0)
        ]
        self.bmin: float = min(_norm2(v) for v in vecs)
        vecs_fus = [v for v in vecs if _norm2(v) <= bmax_fusion * self.bmin + 1e-6]
        raw_b_orbits = _orbits(
            vecs_fus,
            P_plus,
            lambda v, g: g[0] @ v,
            lambda p, q: _norm2(p - q) < 1e-4,
        )
        # Canonicalize representative of each P+ orbit
        b_recs = []
        for cl in raw_b_orbits:
            # Prefer positive x, then positive y
            cl_sorted = sorted(cl, key=lambda v: (-round(float(v[0]), 5), -round(float(v[1]), 5)))
            rep = cl_sorted[0]
            length = _norm2(rep)
            b_recs.append(
                dict(
                    b=rep,
                    H=(np.eye(2), -rep),
                    length=length,
                    size=len(cl),
                    members=cl,
                    elementary=(length <= bmax * self.bmin + 1e-5),
                )
            )
        b_recs.sort(key=lambda c: (round(c["length"], 5), -round(float(c["b"][0]), 4), -round(float(c["b"][1]), 4)))
        for idx, rec in enumerate(b_recs, 1):
            rec["id"] = f"T{idx}"
            rec["kind"] = "dislocation"
            rec["angle"] = 0.0

        # Annotate inverse class [-b]_{P+} and mirror partner [-M0 b]_{P+}
        for rec in b_recs:
            inv_b = -rec["b"]
            rec["inverse_id"] = next(
                r2["id"] for r2 in b_recs if any(_norm2(m - inv_b) < 1e-4 for m in r2["members"])
            )
            if self.m0 is not None:
                mir_b = -self.m0[0] @ rec["b"]
                rec["mirror_of"] = next(
                    (r2["id"] for r2 in b_recs if any(_norm2(m - mir_b) < 1e-4 for m in r2["members"])),
                    None,
                )
            else:
                rec["mirror_of"] = None

        self.all_translations: list[dict] = b_recs
        self.translations: list[dict] = [r for r in b_recs if r["elementary"]]

        # --- 2. Proper rotation-centre Wyckoff classes under G+ ---
        all_pts: list[np.ndarray] = []
        for M, t in self.ops_plus:
            if np.allclose(M, np.eye(2), atol=1e-6):
                continue
            for m in itertools.product(range(-2, 3), repeat=2):
                c = _reduce_point(np.linalg.solve(np.eye(2) - M, t + self.A @ np.array(m, float)), self.A, self.Ainv)
                if not any(_same_point_mod_lattice(c, q, self.Ainv) for q in all_pts):
                    all_pts.append(c)

        def stab_order(c: np.ndarray) -> int:
            return sum(1 for M, t in self.ops_plus if _same_point_mod_lattice(M @ c + t, c, self.Ainv))

        raw_c_orbits = _orbits(
            all_pts,
            self.ops_plus,
            lambda c, g: _reduce_point(g[0] @ c + g[1], self.A, self.Ainv),
            lambda p, q: _same_point_mod_lattice(p, q, self.Ainv),
        )
        c_recs = []
        for cl in raw_c_orbits:
            cl_red = [_reduce_point(q, self.A, self.Ainv) for q in cl]
            rep = min(cl_red, key=lambda q: (round(float(q[0]), 4), round(float(q[1]), 4)))
            n = stab_order(rep)
            angs = [
                k * 360.0 / n
                for k in range(-4 * n, n)
                if k != 0 and angle_range[0] - 1e-6 <= k * 360.0 / n <= angle_range[1] + 1e-6
            ]
            c_recs.append(
                dict(
                    point=rep,
                    frac=np.round(self.Ainv @ rep, 6) + 0.0,
                    order=n,
                    size=len(cl_red),
                    members=cl_red,
                    angles=angs,
                    on_site=(_fast_locate(rep, self.Ainv, self.cell.basis) is not None),
                )
            )
        c_recs.sort(key=lambda c: (-c["order"], round(float(c["frac"][0]), 4), round(float(c["frac"][1]), 4)))
        letters = "abcdefghijklmnopqrstuvwxyz"
        for idx, rec in enumerate(c_recs):
            rec["wyckoff"] = f"{rec['order']}{letters[idx]}"

        for rec in c_recs:
            if self.m0 is not None:
                mc = _reduce_point(iso_apply(self.m0, rec["point"]), self.A, self.Ainv)
                rec["mirror_of"] = next(
                    r2["wyckoff"]
                    for r2 in c_recs
                    if any(_same_point_mod_lattice(mc, q, self.Ainv) for q in r2["members"])
                )
            else:
                rec["mirror_of"] = None

        self.centres: list[dict] = c_recs
        self.all_centres: list[tuple[np.ndarray, int, str]] = [
            (q, rec["order"], rec["wyckoff"]) for rec in c_recs for q in rec["members"]
        ]

        # --- 3. Proper disclination classes D(wyckoff, angle) ---
        self.disclinations: list[dict] = []
        for rec in self.centres:
            w = rec["wyckoff"]
            mw = rec["mirror_of"]
            for ang in rec["angles"]:
                did = f"D({w}, {ang:+.0f}°)"
                mid = f"D({mw}, {ang:+.0f}°)" if mw is not None else None
                H = rotation_about(rec["point"], math.radians(ang))
                self.disclinations.append(
                    dict(
                        id=did,
                        kind="disclination",
                        wyckoff=w,
                        centre=rec["point"],
                        order=rec["order"],
                        on_site=rec["on_site"],
                        angle=float(ang),
                        H=H,
                        mirror_of=mid,
                        inverse_id=f"D({w}, {-ang:+.0f}°)" if (-ang in rec["angles"]) else None,
                    )
                )

        # --- 4. Lines (Moebius strips) & Domains (Walls) from C06 ---
        self.lines: list[dict] = []
        for idx, ln in enumerate(self.base_cat.lines, 1):
            r = dict(ln)
            r["id"] = f"M{idx}"
            self.lines.append(r)

        self.domains: list[dict] = []
        for idx, dm in enumerate(self.base_cat.domains, 1):
            r = dict(dm)
            r["id"] = f"W{idx}"
            self.domains.append(r)

        # Lookup table of all point-defect classes (including vacuum I and fusion Burgers shells)
        self.vacuum = dict(id="I", kind="vacuum", angle=0.0, b=np.zeros(2), H=_ident(), mirror_of="I" if self.m0 else None, inverse_id="I")
        self.by_id: dict[str, dict] = {"I": self.vacuum}
        for r in self.all_translations:
            self.by_id[r["id"]] = r
        for r in self.disclinations:
            self.by_id[r["id"]] = r
        for r in self.lines:
            self.by_id[r["id"]] = r
        for r in self.domains:
            self.by_id[r["id"]] = r

    def identify(self, alpha_deg: float, H: tuple[np.ndarray, np.ndarray], tol: float = 1e-3) -> dict:
        """Identify a measured (alpha_deg, H) pair in the proper-group catalogue."""
        if abs(alpha_deg) < tol and np.allclose(H[0], np.eye(2), atol=1e-4):
            b = -H[1]
            lb = _norm2(b)
            if lb < 1e-4:
                return dict(class_id="I", kind="vacuum", angle=0.0, b=np.zeros(2), mirror_of=self.vacuum["mirror_of"])
            for r in self.all_translations:
                if abs(lb - r["length"]) < 1e-3 and same_proper_holonomy(H, r["H"], self.ops_plus, self.A, self.Ainv):
                    return dict(class_id=r["id"], kind="dislocation", angle=0.0, b=b, length=lb, mirror_of=r["mirror_of"])
            return dict(class_id=f"T(|b|={lb:.3f})", kind="dislocation", angle=0.0, b=b, length=lb, mirror_of=None)

        if np.allclose(H[0], np.eye(2), atol=1e-4):
            # Whole-turn negative disclination (e.g. -360 deg) + Burgers shift
            b = -H[1]
            return dict(class_id=f"Turn({alpha_deg:+.0f}°, |b|={_norm2(b):.3f})", kind="whole_turn", angle=alpha_deg, b=b, mirror_of=None)

        # Disclination class: match both unreduced alpha_deg AND proper centre class
        for r in self.disclinations:
            if abs(alpha_deg - r["angle"]) < tol and same_proper_holonomy(H, r["H"], self.ops_plus, self.A, self.Ainv):
                return dict(
                    class_id=r["id"],
                    kind="disclination",
                    angle=r["angle"],
                    wyckoff=r["wyckoff"],
                    centre=np.linalg.solve(np.eye(2) - H[0], H[1]),
                    mirror_of=r["mirror_of"],
                )
        # Composite angle outside [-180, 180] (e.g. -240, -270, +240, +270): match Wyckoff centre directly
        c_meas = _reduce_point(np.linalg.solve(np.eye(2) - H[0], H[1]), self.A, self.Ainv)
        for rec in self.centres:
            if any(_same_point_mod_lattice(c_meas, q, self.Ainv) for q in rec["members"]):
                w = rec["wyckoff"]
                mw = rec["mirror_of"]
                return dict(
                    class_id=f"D({w}, {alpha_deg:+.0f}°)",
                    kind="disclination",
                    angle=float(round(alpha_deg, 4)),
                    wyckoff=w,
                    centre=c_meas,
                    mirror_of=f"D({mw}, {alpha_deg:+.0f}°)" if mw else None,
                )
        return dict(class_id=f"D(?, {alpha_deg:+.0f}°)", kind="disclination", angle=alpha_deg, centre=c_meas, mirror_of=None)

    def mirror_class_id(self, class_id: str) -> str | None:
        """Return the mirror-partner class ID \\mu(C) for any class ID (including composite angles)."""
        if self.m0 is None:
            return None
        if class_id in self.by_id:
            return self.by_id[class_id]["mirror_of"]
        for c in self.centres:
            pref = f"D({c['wyckoff']},"
            if class_id.startswith(pref):
                return f"D({c['mirror_of']}," + class_id[len(pref) :]
        return class_id


# ==============================================================================================
# 3. Cone-safe multi-defect builder (fixes all C08 mixed-list refusals and all C09 audit bugs)
# ==============================================================================================
class _ConeSafeBuilder:
    def __init__(self, cell: UnitCell, L: Perfect, centre: np.ndarray, R: float):
        self.cell = cell
        self.L = L
        self.A = cell.A
        self.Ainv = L.Ainv
        self.basis = cell.basis
        self.centre = np.asarray(centre, float)
        self.R = float(R)
        self.lmax = max(_norm2(L.pos(k2) - L.pos((0, 0, s))) for s in L.rule for k2, _ in L.nbrs((0, 0, s)))
        self.scale = max(_norm2(cell.a1), _norm2(cell.a2))
        keys = L.sites_in_disk(self.centre, self.R)
        self.label = [(k, 0) for k in keys]
        self.ref = [L.pos(k) for k in keys]
        self.alive = [True] * len(keys)
        self.nb: list[dict[int, dict]] = [dict() for _ in keys]
        index = {k: i for i, k in enumerate(keys)}
        for k, i in index.items():
            pk = self.ref[i]
            for k2, typ in L.nbrs(k):
                j = index.get(k2)
                if j is not None:
                    self.nb[i][j] = dict(vec=self.ref[j] - pk, type=typ)
        self.X = [x.copy() for x in self.ref]
        self.layers = Counter(keys)
        self.touched: set[int] = set()
        self.cone_sites: set[int] = set()   # live vertex indices sitting on site-centred disclination apices
        self.records: list[dict] = []
        self.perfect_sites = len(keys)

    def snapshot(self):
        return (
            list(self.label),
            [x.copy() for x in self.ref],
            list(self.alive),
            [{v: dict(e) for v, e in n.items()} for n in self.nb],
            [x.copy() for x in self.X],
            Counter(self.layers),
            set(self.touched),
            set(self.cone_sites),
            list(self.records),
        )

    def restore(self, snap):
        (
            self.label,
            self.ref,
            self.alive,
            self.nb,
            self.X,
            self.layers,
            self.touched,
            self.cone_sites,
            self.records,
        ) = snap

    def kill(self, i: int):
        if i in self.cone_sites:
            raise ConfigError("a later cut would remove an existing site-centred disclination core; try another cut direction")
        for v in list(self.nb[i]):
            if v in self.cone_sites:
                raise ConfigError("a later cut removes a neighbour of a site-centred disclination core; try another cut direction")
            self.nb[v].pop(i, None)
        self.nb[i] = {}
        self.alive[i] = False

    def new_copy(self, i: int) -> int:
        if i in self.cone_sites:
            raise ConfigError("a later cut duplicates an existing site-centred disclination core; try another cut direction")
        key = self.label[i][0]
        self.label.append((key, self.layers[key]))
        self.layers[key] += 1
        self.ref.append(self.ref[i].copy())
        self.alive.append(True)
        self.nb.append({})
        self.X.append(self.X[i].copy())
        return len(self.label) - 1

    def clear(self, p: np.ndarray) -> float:
        return self.R - _norm2(np.asarray(p, float) - self.centre)

    def base_site(self, p: np.ndarray, inside) -> int:
        best = None
        for i in range(len(self.label)):
            if self.alive[i] and i not in self.cone_sites and inside(self.ref[i]):
                key = (round(_norm2(self.ref[i] - p), 7), self.label[i][1], i)
                if best is None or key < best:
                    best = key
        if best is None or best[0] > self.lmax + self.scale:
            raise ConfigError(
                f"no crystal site near {np.round(p, 4).tolist()}: point is outside patch or inside a previously removed wedge/strip"
            )
        return best[2]

    def develop(self, base: int, inside):
        ident = _ident()
        D = {base: ident}
        Y = {base: self.ref[base].copy()}
        dq = deque([base])
        while dq:
            u = dq.popleft()
            Du = D[u]
            Yu = Y[u]
            for v, e in self.nb[u].items():
                g = e.get("g")
                Dv = Du if g is None else iso_compose(Du, iso_inverse(g))
                if v in D:
                    if not _same_iso(Dv, D[v]):
                        raise ConfigError(
                            "developed region encloses an earlier defect core (two walks disagree); use another cut direction or defect order"
                        )
                    continue
                yv = iso_apply(Dv, self.ref[v])
                if inside(yv):
                    D[v] = Dv
                    Y[v] = yv
                    # Never walk THROUGH an existing cone singularity (flat chart is multi-valued round it),
                    # though recording its endpoint position allows an adjacent site-core to keep its bond to v.
                    if v not in self.cone_sites:
                        dq.append(v)
        where = {}
        for i, y in Y.items():
            k = _fast_locate(y, self.Ainv, self.basis)
            if k is None:
                raise ConfigError("developed position fell off the lattice")
            if k in where:
                raise ConfigError("developed region overlaps itself round an earlier disclination; try another cut direction")
            where[k] = i
        return D, Y, where

    def local_rotation(self, u: int) -> np.ndarray:
        s = c = 0.0
        Xu = self.X[u]
        for v, e in self.nb[u].items():
            dv = self.X[v] - Xu
            vec = e["vec"]
            s += _cross2v(vec, dv)
            c += float(vec[0]) * float(dv[0]) + float(vec[1]) * float(dv[1])
        return rot(math.atan2(s, c)) if abs(s) + abs(c) > 1e-12 else np.eye(2)

    def _set(self, u: int, v: int, vec_chart: np.ndarray, T: tuple[np.ndarray, np.ndarray], typ: int, D: dict, allow_cone: int | None = None, **extra):
        if (u in self.cone_sites and u != allow_cone) or (v in self.cone_sites and v != allow_cone):
            raise ConfigError("a later cut would overwrite bonds of an earlier site-centred disclination; try another cut direction")
        g = iso_compose(iso_inverse(D[v]), iso_compose(T, D[u]))
        e = dict(vec=D[u][0].T @ vec_chart, type=typ)
        if not _same_iso(g, _ident()):
            e["g"] = g
            e["seam"] = True
        e.update(extra)
        self.nb[u][v] = e

    def _apply(self, rule: dict, kept: list[int], D: dict, T_of, new_sites):
        fresh = set(new_sites)
        # Protect existing site-centred cone apices so their unrolled 'ang' attributes are never erased
        for (u, v), (_, j, _) in rule.items():
            if (u in self.cone_sites or v in self.cone_sites) and j != 0:
                raise ConfigError("seam crosses a bond incident on an earlier site-centred disclination; try another cut direction")
        keptset = set(kept) - self.cone_sites
        for u in keptset:
            for v in list(self.nb[u]):
                if v in keptset and (u, v) not in rule:
                    del self.nb[u][v]
                    self.nb[v].pop(u, None)
        for (u, v), (vec, j, typ) in rule.items():
            if (v, u) not in rule:
                continue
            if u in self.cone_sites or v in self.cone_sites:
                # j == 0 already checked above; ensure the existing bond to the cone site is intact
                if v not in self.nb[u]:
                    raise ConfigError("cut runs into a missing bond of an earlier cone site; try another cut direction")
                continue
            if j == 0 and u not in fresh and v not in fresh:
                e = self.nb[u].get(v)
                if e is None or _norm2(D[u][0] @ e["vec"] - vec) > 1e-5:
                    raise ConfigError("cut runs into the core of an earlier defect (missing/altered bond); try another cut direction")
                continue
            self._set(u, v, vec, T_of(j), typ, D)

    def add_dislocation(self, b: np.ndarray, p: np.ndarray, cut: np.ndarray | None = None):
        L = self.L
        b = np.asarray(b, float)
        p = np.asarray(p, float)
        lb = _norm2(b)
        fb = self.Ainv @ b
        if lb < 1e-9 or abs(float(fb[0]) - round(float(fb[0]))) > 1e-5 or abs(float(fb[1]) - round(float(fb[1]))) > 1e-5:
            raise ConfigError(f"b = {np.round(b, 4).tolist()} is not a non-zero lattice translation")
        c = np.array([-b[1], b[0]]) if cut is None else np.asarray(cut, float)
        lc = _norm2(c)
        if lc < 1e-9:
            raise ConfigError("cut direction must be non-zero")
        if _cross2v(c, b) > -1e-7 * lc * lb:
            raise ConfigError(
                "for Volterra removal, b must point to the right of the cut (cross2(cut, b) < 0); reverse cut -> -cut"
            )
        ch = c / lc
        nh = np.array([-ch[1], ch[0]])
        bn = float(b @ nh)
        ba = float(b @ ch)
        m = 1.05 * self.lmax
        if self.clear(p) < lb + 2.5 * self.lmax:
            raise ConfigError(f"point {np.round(p, 4).tolist()} is too close to patch rim; increase R")

        def inside(y: np.ndarray) -> bool:
            rx = float(y[0]) - float(p[0])
            ry = float(y[1]) - float(p[1])
            a = rx * float(ch[0]) + ry * float(ch[1])
            n = rx * float(nh[0]) + ry * float(nh[1])
            return (bn - m <= n <= m) and (a >= min(0.0, ba) - m)

        base = self.base_site(p, inside)
        D, Y, _ = self.develop(base, inside)
        N = set(D)
        Binv = np.linalg.inv(np.column_stack([b, c]))
        ts = {i: tuple(Binv @ (Y[i] - p)) for i in N}
        gone = [i for i in N if ts[i][1] > 1e-9 and -1e-9 <= ts[i][0] < 1.0 - 1e-9]
        gone_set = set(gone)
        kept = [i for i in N if i not in gone_set]
        charts = [
            (lambda t, s: s <= 1e-9 or t < -1e-9, lambda t, s: 0),
            (lambda t, s: t >= 1.0 - 1e-9, lambda t, s: -1),
            (lambda t, s: s > 1e-9, lambda t, s: -1 if t >= 1.0 - 1e-9 else 0),
        ]
        rule = {}
        for region, shift in charts:
            here = {}
            for i in kept:
                t, s = ts[i]
                if region(t, s):
                    sh = shift(t, s)
                    k_loc = _fast_locate(Y[i] + sh * b, self.Ainv, self.basis)
                    if k_loc is not None:
                        here[k_loc] = (i, sh)
            for img, (u, su) in here.items():
                p_img = L.pos(img)
                for k2, typ in L.nbrs(img):
                    hit = here.get(k2)
                    if hit is None or hit[0] == u:
                        continue
                    v, sv = hit
                    new_entry = (L.pos(k2) - p_img, su - sv, typ)
                    rule[(u, v)] = new_entry
        Q = self.local_rotation(base)
        xb = self.X[base].copy()
        yb = Y[base].copy()
        for i in gone:
            self.kill(i)
        self._apply(rule, kept, D, lambda j: (np.eye(2), j * b), new_sites=())
        core = xb + Q @ (p + 0.5 * b - yb)
        bq = Q @ b
        dq_ = Q @ ch
        sgn = 1.0 if _cross2v(c, b) > 0 else -1.0
        for i in range(len(self.X)):
            if self.alive[i]:
                r = self.X[i] - core
                ang = math.atan2(sgn * _cross2v(dq_, r), float(r @ dq_)) % TWO_PI
                self.X[i] = self.X[i] - (1.0 - ang / TWO_PI) * bq
        self.touched |= set(kept)
        self.records.append(
            dict(
                kind="dislocation",
                b=b,
                at=p,
                cut=ch,
                removed=len(gone),
                inserted=0,
                frame_of=self.label[base],
                core_site=None,
                double_bonds=0,
                self_bonds=0,
            )
        )

    def add_disclination(self, angle: float, p: np.ndarray, cut: np.ndarray | None = None):
        p = np.asarray(p, float)
        if abs(angle) < 1e-9 or angle >= 360.0 - 1e-9 or angle <= -360.0 + 1e-9:
            raise ConfigError(
                f"elementary disclination angle must satisfy 0 < |angle| < 360 degrees (got {angle:g})"
            )
        if not self.L.is_symmetry(rotation_about(p, math.radians(angle)), probe=2.2):
            raise ConfigError(
                f"rotation by {angle:g} degrees about {np.round(p, 4).tolist()} is not a symmetry of this lattice"
            )
        if self.clear(p) < 3.5 * self.lmax:
            raise ConfigError(f"point {np.round(p, 4).tolist()} is too close to patch rim; increase R")
        c = np.array([math.cos(0.123), math.sin(0.123)]) if cut is None else np.asarray(cut, float)
        if _norm2(c) < 1e-9:
            raise ConfigError("cut direction must be non-zero")
        th_cut = math.atan2(float(c[1]), float(c[0]))
        for nudge in (0.0, 1e-5, -1e-5, 3e-5, -3e-5, 1e-4):
            snap = self.snapshot()
            try:
                return self._disclination(angle, p, th_cut + nudge)
            except _SeamHit:
                self.restore(snap)
                continue
        raise ConfigError("lattice sites lie on the cut ray for all nearby nudges; choose another cut direction")

    def _disclination(self, angle: float, p: np.ndarray, th0: float):
        L = self.L
        alpha = math.radians(angle)
        Theta = TWO_PI - alpha
        eps = 1e-7
        m = 1.05 * self.lmax
        u0 = np.array([math.cos(th0), math.sin(th0)])
        u1 = np.array([math.cos(th0 - alpha), math.sin(th0 - alpha)])

        def inside(y: np.ndarray) -> bool:
            rx = float(y[0]) - float(p[0])
            ry = float(y[1]) - float(p[1])
            if math.hypot(rx, ry) < 1e-9:
                return True
            th = (math.atan2(ry, rx) - th0) % TWO_PI
            if (alpha > 0.0 and th >= Theta) or (alpha < 0.0 and th <= -alpha):
                return True
            return min(_ray_distance(y, p, u0), _ray_distance(y, p, u1)) <= m

        base = self.base_site(p, inside)
        D, Y, where = self.develop(base, inside)
        keyof = {i: k for k, i in where.items()}
        lift: dict[int, float] = {}
        core: int | None = None
        for i, y in Y.items():
            r = y - p
            if _norm2(r) < 1e-9:
                core = i
                continue
            th = (math.atan2(float(r[1]), float(r[0])) - th0) % TWO_PI
            if min(th, TWO_PI - th) < 1e-6:
                raise _SeamHit()
            lift[i] = th

        if core is not None and core in self.cone_sites:
            raise ConfigError("two disclinations cannot sit on the same lattice site")

        Q = self.local_rotation(base)
        xb = self.X[base].copy()
        yb = Y[base].copy()
        rep: dict[tuple[int, int], int] = {}
        phi: dict[int, float] = {}
        gone: list[int] = []
        fresh: list[int] = []
        for i, th in sorted(lift.items()):
            k = 0
            while th + TWO_PI * k < Theta - eps:
                if k == 0:
                    v = i
                else:
                    v = self.new_copy(i)
                    fresh.append(v)
                    D[v] = D[i]
                rep[(i, k)] = v
                phi[v] = th + TWO_PI * k
                k += 1
            if k == 0:
                gone.append(i)

        rule = {}
        selfs, multi, through = set(), set(), set()
        for (i, k), u in rep.items():
            x = Y[i]
            for k2, typ in L.nbrs(keyof[i]):
                q = L.pos(k2)
                if _norm2(q - p) < 1e-9:
                    continue
                a, bb = x - p, q - p
                if abs(_cross2v(a, bb)) < 1e-9 and float(a @ bb) < 0.0:
                    through.add(frozenset((keyof[i], k2)))
                    continue
                phq = phi[u] + math.atan2(_cross2v(a, bb), float(a @ bb))
                j = math.floor((phq + eps) / Theta)
                Rj = rot(-j * Theta)
                q2 = p + Rj @ (q - p)
                ph2 = phq - j * Theta
                kloc = _fast_locate(q2, self.Ainv, self.basis)
                i2 = where.get(kloc) if kloc is not None else None
                if i2 is None:
                    continue
                v = rep.get((i2, int(math.floor((ph2 + eps) / TWO_PI))))
                if v is None:
                    continue
                if v == u:
                    selfs.add(u)
                    continue
                if (u, v) in rule:
                    if rule[(u, v)][1] != j:
                        multi.add((min(u, v), max(u, v)))
                    continue
                rule[(u, v)] = (q - x, j, typ)
                rule.setdefault((v, u), (Rj @ (x - q), -j, typ))

        gone_set = set(gone)
        for i in gone:
            self.kill(i)
        kept = [i for i in lift if i not in gone_set]
        self._apply(rule, kept + fresh, D, lambda j: (rot(-j * Theta), p - rot(-j * Theta) @ p), new_sites=fresh)

        if core is not None:
            self.cone_sites.add(core)
            core_key = _fast_locate(p, self.Ainv, self.basis)
            for k2, typ in L.nbrs(core_key):
                i2 = where.get(k2)
                if i2 is None:
                    continue
                q = L.pos(k2)
                for k in range(3):
                    v = rep.get((i2, k))
                    if v is None:
                        continue
                    if k > 0 or v not in self.nb[core]:
                        self._set(core, v, q - p, _ident(), typ, D, allow_cone=core)
                        self._set(v, core, p - q, _ident(), typ, D, allow_cone=core)
                    self.nb[core][v]["ang"] = phi[v]

        xc = xb + Q @ (p - yb)
        thd = th0 + math.atan2(float(Q[1, 0]), float(Q[0, 0]))
        for i in range(len(self.X)):
            if not self.alive[i] or i == core:
                continue
            r = self.X[i] - xc
            rad = _norm2(r)
            if rad < 1e-12:
                continue
            th = (math.atan2(float(r[1]), float(r[0])) - thd) % TWO_PI
            if i in phi:
                dlt = (th - phi[i] + math.pi) % TWO_PI - math.pi
                th = phi[i] + dlt
            ang = th * TWO_PI / Theta + thd
            self.X[i] = xc + rad * np.array([math.cos(ang), math.sin(ang)])

        self.touched |= set(kept) | set(fresh) | ({core} if core is not None else set())
        self.records.append(
            dict(
                kind="disclination",
                angle=float(angle),
                at=p,
                cut=u0,
                removed=len(gone),
                inserted=len(fresh),
                frame_of=self.label[base],
                core_site=core,
                double_bonds=len(multi),
                self_bonds=len(selfs),
                bonds_through_centre=len(through),
            )
        )

    def prune_rim_spurs(self):
        """Remove 0-bond isolated sites and outer-rim degree-1 spurs (never touching protected core sites),
        and keep the main connected component so V - E + F = 2 holds cleanly."""
        protected = set(self.cone_sites)
        # Iteratively peel degree <= 1 vertices that are not protected cone sites
        changed = True
        while changed:
            changed = False
            for i in range(len(self.label)):
                if self.alive[i] and i not in protected and len(self.nb[i]) <= 1:
                    for v in list(self.nb[i]):
                        self.nb[v].pop(i, None)
                    self.nb[i] = {}
                    self.alive[i] = False
                    changed = True

        # Keep largest connected component
        live_set = {i for i in range(len(self.label)) if self.alive[i]}
        if not live_set:
            raise ConfigError("all crystal sites were removed")
        comps = []
        seen = set()
        for start in live_set:
            if start in seen:
                continue
            comp = set()
            dq = deque([start])
            seen.add(start)
            while dq:
                u = dq.popleft()
                comp.add(u)
                for v in self.nb[u]:
                    if v in live_set and v not in seen:
                        seen.add(v)
                        dq.append(v)
            comps.append(comp)
        if len(comps) > 1:
            main = max(comps, key=len)
            for i in live_set - main:
                if i in protected:
                    raise ConfigError("cut disconnected an earlier disclination core from the crystal")
                for v in list(self.nb[i]):
                    self.nb[v].pop(i, None)
                self.nb[i] = {}
                self.alive[i] = False

    def finish(self, kind: str = "configuration") -> Defect:
        self.prune_rim_spurs()
        live = [i for i in range(len(self.label)) if self.alive[i]]
        new = {i: n for n, i in enumerate(live)}
        edge = {}
        for i in live:
            ui = new[i]
            for v, e in self.nb[i].items():
                edge[(ui, new[v])] = dict(e)
        ref = np.array([self.ref[i] for i in live]).reshape(-1, 2)
        X = np.array([self.X[i] for i in live]).reshape(-1, 2)
        recs = []
        for r in self.records:
            r2 = dict(r)
            if r2["core_site"] is not None:
                r2["core_site"] = new.get(r2["core_site"])
            recs.append(r2)
        rim = [new[i] for i in live if _norm2(self.ref[i] - self.centre) > self.R - 1.5 * self.lmax]
        info = dict(
            reference=ref,
            torus=False,
            wraps={},
            T=None,
            lmax=self.lmax,
            pinned=rim,
            rim=rim,
            touched=sorted(new[i] for i in self.touched if i in new),
            cone_sites=sorted(new[i] for i in self.cone_sites if i in new),
            defects=recs,
            R=self.R,
            centre=self.centre,
            perfect_sites=self.perfect_sites,
            guess=X.copy(),
            removed=sum(r["removed"] for r in recs),
            inserted=sum(r["inserted"] for r in recs),
        )
        return Defect(kind, [self.label[i] for i in live], X, edge, info)


def _candidate_cuts(d: dict, others: list[dict], disloc: bool) -> list[np.ndarray]:
    """Order candidate cut rays by maximum clearance from ALL other defect positions (not just their centroid)."""
    at = np.asarray(d["at"], float)
    other_pts = [np.asarray(o["at"], float) for o in others if _norm2(np.asarray(o["at"], float) - at) > 1e-6]

    if disloc:
        b = np.asarray(d["b"], float)
        c0 = np.array([-b[1], b[0]]) / max(_norm2(b), 1e-30)
        cand = [rot(math.radians(a)) @ c0 for a in (0, 25, -25, 50, -50, 72, -72)]
        if other_pts:
            cand.sort(key=lambda c: -min(_ray_distance(pt, at, c / _norm2(c)) for pt in other_pts))
        return cand

    ang_deg = float(d.get("angle", 0.0))
    alpha = math.radians(ang_deg)
    cand = [np.array([math.cos(0.117 + math.radians(30 * k)), math.sin(0.117 + math.radians(30 * k))]) for k in range(12)]
    if not other_pts:
        return cand

    other_angs = [math.atan2(float(pt[1] - at[1]), float(pt[0] - at[0])) for pt in other_pts]

    def wedge_clearance(u: np.ndarray) -> float:
        th0 = math.atan2(float(u[1]), float(u[0]))
        worst = math.pi
        for phi in other_angs:
            if alpha > 0.0:
                # Removed wedge is clockwise of th0: (th0 - phi) mod 2pi in [0, alpha]
                cw = (th0 - phi) % TWO_PI
                if cw <= alpha:
                    return -1.0  # pt sits directly inside the removed wedge!
                gap = min(cw - alpha, TWO_PI - cw)
            else:
                # Inserted wedge: seam ray at th0
                gap = abs((phi - th0 + math.pi) % TWO_PI - math.pi)
            if gap < worst:
                worst = gap
        return worst

    cand.sort(key=lambda u: -wedge_clearance(u))
    return cand


# ==============================================================================================
# 4. Graph-only invariants, relaxation, and 3-level comparison
# ==============================================================================================
def check_frames(d: Defect) -> int:
    """Count directed bonds failing local-frame compatibility; 0 means a globally consistent crystal."""
    X = d.info["reference"]
    wr = d.info.get("wraps") or {}
    ident = _ident()
    bad = 0
    for (u, v), e in d.edge.items():
        g = e.get("g", ident)
        back = d.edge[(v, u)]
        ok = _norm2(iso_apply(g, X[u] + e["vec"]) - (X[v] + wr.get((u, v), 0.0))) < 1e-5
        ok = ok and _same_iso(iso_compose(back.get("g", ident), g), ident, tol=1e-5)
        ok = ok and _norm2(back["vec"] + g[0] @ e["vec"]) < 1e-5
        if not ok:
            bad += 1
    return bad


def _corners(d: Defect) -> dict[int, list[tuple[float, int]]]:
    nbr: dict[int, list[tuple[float, int]]] = {}
    for (u, v), e in d.edge.items():
        a = e["ang"] if "ang" in e else math.atan2(float(e["vec"][1]), float(e["vec"][0])) % TWO_PI
        nbr.setdefault(u, []).append((a, v))
    return {u: sorted(lst) for u, lst in nbr.items()}


def _outer_face_index(faces: list[list[int]]) -> int:
    return max(range(len(faces)), key=lambda i: len(faces[i]))


def frank_angles(d: Defect, faces: list[list[int]]) -> list[float]:
    order = _corners(d)
    index = {u: {v: k for k, (_, v) in enumerate(lst)} for u, lst in order.items()}
    out = []
    for f in faces:
        tot = 0.0
        n = len(f)
        for k in range(n):
            a, b_ = f[k - 1], f[k]
            lst = order[b_]
            i = index[b_][a]
            gap = (lst[i][0] - lst[i - 1][0]) if i > 0 else (lst[0][0] + TWO_PI - lst[-1][0])
            tot += gap
        out.append(math.degrees(tot - (n - 2) * math.pi))
    return out


def _det_sign_iso(g: tuple[np.ndarray, np.ndarray] | None) -> int:
    if g is None:
        return 1
    M = g[0]
    return 1 if (float(M[0, 0]) * float(M[1, 1]) - float(M[0, 1]) * float(M[1, 0])) > 0.0 else -1


def is_orientable_surface(d: Defect) -> bool:
    """Return True iff the combinatorial crystal surface is orientable (2-colourable under sign(det g_uv))."""
    if "_orientable" in d.info:
        return bool(d.info["_orientable"])
    has_g = any("g" in e for e in d.edge.values())
    if not has_g and any(abs(int(e.get("j", 0))) % 2 == 1 for e in d.edge.values()):
        from .lab_defects import is_orientable as _ld_is_orientable
        res = bool(_ld_is_orientable(d))
        d.info["_orientable"] = res
        return res
    nbr: dict[int, list[tuple[int, int]]] = {}
    for (u, v), e in d.edge.items():
        nbr.setdefault(u, []).append((v, _det_sign_iso(e.get("g"))))
    color: dict[int, int] = {}
    for start in range(len(d.sites)):
        if start in color:
            continue
        color[start] = 1
        dq = deque([start])
        while dq:
            u = dq.popleft()
            cu = color[u]
            for v, sgn in nbr.get(u, ()):
                need = cu * sgn
                if v in color:
                    if color[v] != need:
                        d.info["_orientable"] = False
                        return False
                else:
                    color[v] = need
                    dq.append(v)
    d.info["_orientable"] = True
    return True


def ribbon_faces(d: Defect) -> list[tuple[list[int], list[int]]]:
    """Signed rotation-system 2-cell face walk `(verts, local_signs)` for both orientable and
    non-orientable crystal surfaces (Möbius strips, Klein bottles). Reduces identically to
    `frame_faces(d)` when `is_orientable_surface(d)` is True."""
    if is_orientable_surface(d):
        return [(f, [1] * len(f)) for f in frame_faces(d)]
    nbr: dict[int, list[tuple[float, int]]] = {}
    for (u, v), e in d.edge.items():
        a = e["ang"] if "ang" in e else math.atan2(float(e["vec"][1]), float(e["vec"][0])) % TWO_PI
        nbr.setdefault(u, []).append((a, v))
    order = {u: [v for _, v in sorted(lst)] for u, lst in nbr.items()}
    idx_map = {u: {v: k for k, v in enumerate(lst)} for u, lst in order.items()}

    seen: set[tuple[int, int, int]] = set()
    faces: list[tuple[list[int], list[int]]] = []
    for u in order:
        for v in order[u]:
            for s0 in (1, -1):
                if (u, v, s0) in seen:
                    continue
                walk_states: list[tuple[int, int, int]] = []
                verts: list[int] = []
                signs: list[int] = []
                a_cur, b_cur, s_cur = u, v, s0
                while (a_cur, b_cur, s_cur) not in seen:
                    walk_states.append((a_cur, b_cur, s_cur))
                    seen.add((a_cur, b_cur, s_cur))
                    verts.append(a_cur)
                    signs.append(s_cur)
                    s_next = s_cur * _det_sign_iso(d.edge[(a_cur, b_cur)].get("g"))
                    lst = order[b_cur]
                    k = idx_map[b_cur][a_cur]
                    nxt = lst[(k - 1) % len(lst)] if s_next == 1 else lst[(k + 1) % len(lst)]
                    a_cur, b_cur, s_cur = b_cur, nxt, s_next
                for aa, bb, ss in walk_states:
                    sb = ss * _det_sign_iso(d.edge[(aa, bb)].get("g"))
                    seen.add((bb, aa, -sb))
                faces.append((verts, signs))
    return faces


def ribbon_frank_angles(d: Defect, rfaces: list[tuple[list[int], list[int]]]) -> list[float]:
    """Intrinsic discrete Gauss-Bonnet curvature (in degrees) of each signed ribbon face."""
    order = _corners(d)
    idx_map = {u: {v: k for k, (_, v) in enumerate(lst)} for u, lst in order.items()}
    out: list[float] = []
    for verts, signs in rfaces:
        tot = 0.0
        n = len(verts)
        for k in range(n):
            a = verts[k - 1]
            b_ = verts[k]
            sb = signs[k]
            lst = order[b_]
            i = idx_map[b_][a]
            if sb == 1:
                gap = (lst[i][0] - lst[i - 1][0]) if i > 0 else (lst[0][0] + TWO_PI - lst[-1][0])
            else:
                j = (i + 1) % len(lst)
                gap = (lst[j][0] - lst[i][0]) if j > 0 else (lst[0][0] + TWO_PI - lst[-1][0])
            tot += gap
        out.append(math.degrees(tot - (n - 2) * math.pi))
    return out


def _interior_ribbon_faces(d: Defect) -> list[tuple[list[int], list[int]]]:
    rfaces = ribbon_faces(d)
    if not rfaces or d.info.get("torus") or d.info.get("klein"):
        return rfaces
    k = max(range(len(rfaces)), key=lambda i: len(rfaces[i][0]))
    return rfaces[:k] + rfaces[k + 1 :]


def surface_euler_characteristic(d: Defect) -> dict:
    """Topological surface counts for disks, tori, Möbius strips, and Klein bottles."""
    orient = is_orientable_surface(d)
    rfaces = ribbon_faces(d)
    closed = bool(d.info.get("torus") or d.info.get("klein"))
    b_comp = 0 if closed else (1 if rfaces else 0)
    f_int = len(rfaces) - b_comp
    V, E = len(d.sites), len(d.edges)
    return dict(
        V=V,
        E=E,
        F_interior=f_int,
        F_capped=len(rfaces),
        boundary_components=b_comp,
        euler_surface=V - E + f_int,
        euler_capped=V - E + len(rfaces),
        orientable=orient,
    )


def graph_face_histogram(d: Defect) -> dict[int, int]:
    """Purely combinatorial interior-face ring-size histogram (supports orientable & non-orientable surfaces)."""
    if is_orientable_surface(d):
        faces = frame_faces(d)
        if not d.info.get("torus") and faces:
            k = _outer_face_index(faces)
            faces = faces[:k] + faces[k + 1 :]
        return dict(sorted(Counter(len(f) for f in faces).items()))
    ifaces = _interior_ribbon_faces(d)
    return dict(sorted(Counter(len(verts) for verts, _ in ifaces).items()))


def total_frank(d: Defect) -> float:
    if is_orientable_surface(d):
        faces = frame_faces(d)
        if d.info.get("torus"):
            return float(sum(frank_angles(d, faces)))
        k = _outer_face_index(faces)
        return float(sum(a for i, a in enumerate(frank_angles(d, faces)) if i != k))
    ifaces = _interior_ribbon_faces(d)
    return float(sum(ribbon_frank_angles(d, ifaces)))


def total_charge(d: Defect) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    if is_orientable_surface(d):
        faces = frame_faces(d)
        return loop_charge(d, faces[_outer_face_index(faces)][::-1])
    rfaces = ribbon_faces(d)
    k = max(range(len(rfaces)), key=lambda i: len(rfaces[i][0]))
    verts, signs = rfaces[k]
    loop = verts[::-1] if signs[0] == 1 else [verts[0]] + verts[1:]
    return loop_charge(d, loop)


def cores(d: Defect) -> list[dict]:
    if is_orientable_surface(d):
        faces = frame_faces(d)
        if not d.info.get("torus") and faces:
            k = _outer_face_index(faces)
            faces = faces[:k] + faces[k + 1 :]
        face_pairs = [(f, frank) for f, frank in zip(faces, frank_angles(d, faces))]
    else:
        ifaces = _interior_ribbon_faces(d)
        franks = ribbon_frank_angles(d, ifaces)
        face_pairs = []
        for (verts, signs), frank in zip(ifaces, franks):
            if sum(signs) >= 0 and (1 in signs):
                idx0 = signs.index(1)
                loop = verts[idx0:] + verts[:idx0]
            else:
                idx0 = signs.index(-1) if (-1 in signs) else 0
                v_rot = verts[idx0:] + verts[:idx0]
                loop = [v_rot[0]] + v_rot[1:][::-1]
            face_pairs.append((loop, frank))
    X = d.info["reference"]
    deg = d.degrees()
    ident = _ident()
    out = []
    for f, frank in face_pairs:
        H, S = loop_charge(d, f)
        if np.allclose(H[0], np.eye(2), atol=1e-6):
            if _norm2(S) > 1e-6:
                out.append(dict(kind="dislocation", b=S, ring=len(f), site=None, sites=list(f)))
            continue
        ang = float(round(frank, 6))
        c = np.linalg.solve(np.eye(2) - H[0], H[1])
        Dm = ident
        x = X[f[0]].copy()
        site = None
        for k in range(len(f)):
            if _norm2(x - c) < 1e-5:
                site = f[k]
                break
            e = d.edge[(f[k], f[(k + 1) % len(f)])]
            x = x + Dm[0] @ e["vec"]
            g = e.get("g")
            if g is not None:
                Dm = iso_compose(Dm, iso_inverse(g))
        rec = dict(kind="disclination", angle=ang, ring=None if site is not None else len(f), site=site, sites=list(f))
        if site is not None:
            rec["bonds"] = int(deg[site])
            rec["sites"] = [site]
        out.append(rec)
    return out


def relax_picture(d: Defect, steps: int = 80) -> Defect:
    """Fold-free least-squares drawing relaxation (purely visual; no graph observable uses positions)."""
    import scipy.sparse as sp
    from scipy.optimize import least_squares

    n = len(d.sites)
    X0 = np.asarray(d.info.get("guess", d.positions), float)
    non_orient = bool(d.info.get("mobius") or d.info.get("klein"))

    def _is_wrap_edge(u: int, v: int) -> bool:
        if not non_orient:
            return False
        e = d.edge[(u, v)]
        return (_det_sign_iso(e.get("g")) < 0) or (_norm2(X0[v] - X0[u]) > 3.2 * max(_norm2(e["vec"]), 1e-6))

    springs = {}
    for u, v in d.edges:
        if not _is_wrap_edge(u, v):
            springs[(u, v)] = (_norm2(d.edge[(u, v)]["vec"]), 1.0)
    corners = _corners(d)
    tri = []
    for u, lst in corners.items():
        unrolled = any("ang" in d.edge[(u, v)] for _, v in lst)
        pairs = list(zip(lst[:-1], lst[1:])) + ([] if unrolled or len(lst) < 3 else [(lst[-1], lst[0])])
        for (a1, v), (a2, w) in pairs:
            if _is_wrap_edge(u, v) or _is_wrap_edge(u, w):
                continue
            gap = (a2 - a1) if unrolled else (a2 - a1) % TWO_PI
            if v == w or not (1e-6 < gap < math.pi - 1e-6):
                continue
            key = (min(v, w), max(v, w))
            if key not in springs:
                springs[key] = (_norm2(d.edge[(u, w)]["vec"] - d.edge[(u, v)]["vec"]), 0.5)
            a0 = _cross2v(d.edge[(u, v)]["vec"], d.edge[(u, w)]["vec"])
            if a0 > 1e-9:
                tri.append((u, v, w, a0))
    tu = np.array([t[0] for t in tri], int)
    tv = np.array([t[1] for t in tri], int)
    tw = np.array([t[2] for t in tri], int)
    ta = np.array([t[3] for t in tri], float)
    tk = len(tri)
    tsc = 0.7 / np.sqrt(ta) if tk else ta
    trow = np.repeat(np.arange(tk), 6)
    uu = np.array([k[0] for k in springs], int)
    vv = np.array([k[1] for k in springs], int)
    rest = np.array([s[0] for s in springs.values()], float)
    wt = np.array([s[1] for s in springs.values()], float)
    m = len(rest)
    anchor = 0.05 if non_orient else 1e-3
    rows = np.repeat(np.arange(m), 4)

    def fun(z):
        P = z.reshape(-1, 2)
        dv = P[uu] - P[vv]
        e1 = P[tv] - P[tu]
        e2 = P[tw] - P[tu]
        area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        return np.concatenate([wt * (np.linalg.norm(dv, axis=1) - rest), tsc * (area - ta), anchor * (z - X0.ravel())])

    def jac(z):
        P = z.reshape(-1, 2)
        dv = P[uu] - P[vv]
        ln = np.maximum(np.linalg.norm(dv, axis=1), 1e-12)
        gx = wt[:, None] * dv / ln[:, None]
        cols = np.column_stack([2 * uu, 2 * uu + 1, 2 * vv, 2 * vv + 1]).ravel()
        vals = np.column_stack([gx[:, 0], gx[:, 1], -gx[:, 0], -gx[:, 1]]).ravel()
        J = sp.coo_matrix((vals, (rows, cols)), shape=(m, 2 * n))
        e1 = P[tv] - P[tu]
        e2 = P[tw] - P[tu]
        dvx, dvy = e2[:, 1], -e2[:, 0]
        dwx, dwy = -e1[:, 1], e1[:, 0]
        tcols = np.column_stack([2 * tv, 2 * tv + 1, 2 * tw, 2 * tw + 1, 2 * tu, 2 * tu + 1]).ravel()
        tvals = (tsc[:, None] * np.column_stack([dvx, dvy, dwx, dwy, -dvx - dwx, -dvy - dwy])).ravel() if tk else np.zeros(0)
        Jt = sp.coo_matrix((tvals, (trow, tcols)), shape=(tk, 2 * n))
        return sp.vstack([J, Jt, anchor * sp.identity(2 * n)]).tocsr()

    if m:
        sol = least_squares(fun, X0.ravel(), jac=jac, method="trf", tr_solver="lsmr", max_nfev=steps)
        d.positions = sol.x.reshape(-1, 2)
    d._nbr = None
    d.info["picture"] = "relaxed"
    return mark_long_bonds(d)


def mark_long_bonds(d: Defect) -> Defect:
    P = d.positions
    worst = 0.0
    for (u, v), e in d.edge.items():
        e.pop("ghost", None)
        ell = _norm2(e["vec"])
        now = _norm2(P[v] - P[u])
        worst = max(worst, abs(now - ell) / max(ell, 1e-12))
        if now > 1.8 * ell:
            e["ghost"] = P[u] + 0.5 * ell * (P[v] - P[u]) / now
    d.info["strain"] = worst
    return d


def build_configuration(
    cell: UnitCell,
    defects: list[dict],
    L: Perfect | None = None,
    R: float | None = None,
    centre: np.ndarray | None = None,
    picture: str = "relaxed",
    verify_planar: bool = True,
) -> Defect:
    """Build any list of dislocations and disclinations on an open patch with cone-apex protection
    and combinatorial map verification."""
    if not defects:
        raise ConfigError("provide at least one defect")
    defects = [dict(d) for d in defects]
    expected_frank = sum(float(d.get("angle", 0.0)) for d in defects if d.get("kind") in ("disclination", "rotation"))
    if expected_frank >= 360.0 - 1e-6:
        raise ConfigError(
            f"total Frank angle {expected_frank:+.1f} deg >= 360 deg closes or inverts the planar cone; open-patch disk requires sum(angle) < 360 deg"
        )
    L = L or Perfect(cell)
    pts = np.array([np.asarray(d["at"], float) for d in defects]).reshape(-1, 2)
    ctr = pts.mean(axis=0) if centre is None else np.asarray(centre, float)
    probe = _ConeSafeBuilder(cell, L, ctr, 1.0)
    if R is None:
        max_b = max((_norm2(np.asarray(d["b"], float)) for d in defects if "b" in d), default=0.0)
        R = float(np.max(np.linalg.norm(pts - ctr, axis=1))) + max(12.5 * probe.lmax + max_b, 5.0 * probe.scale)
    B = _ConeSafeBuilder(cell, L, ctr, R)

    for idx, d in enumerate(defects, 1):
        kind = d.get("kind")
        if kind not in ("dislocation", "translation", "disclination", "rotation"):
            raise ConfigError(f"defect {idx}: kind must be 'dislocation' or 'disclination', got {kind!r}")
        disloc = kind in ("dislocation", "translation")
        cuts = (
            [np.asarray(d["cut"], float)]
            if d.get("cut") is not None
            else _candidate_cuts(d, [o for o in defects if o is not d], disloc)
        )
        err = None
        for cut in cuts:
            snap = B.snapshot()
            try:
                if disloc:
                    B.add_dislocation(d["b"], d["at"], cut)
                else:
                    B.add_disclination(float(d["angle"]), d["at"], cut)
                err = None
                break
            except ConfigError as e:
                B.restore(snap)
                err = e
        if err is not None:
            raise ConfigError(f"defect {idx} ({kind}): {err}") from None

    out = B.finish()
    if verify_planar and not has_crossings(cell):
        bad_f = check_frames(out)
        eu = euler_characteristic(out)
        tf = total_frank(out)
        min_deg = int(out.degrees().min()) if len(out.sites) else 0
        if bad_f != 0 or eu != 2 or abs(tf - expected_frank) > 1e-2 or min_deg < 2:
            raise ConfigError(
                f"combinatorial map check failed after cut (frames_bad={bad_f}, euler={eu}, frank={tf:.2f} vs {expected_frank:.2f}, min_deg={min_deg}); increase separation or adjust cut"
            )
    if picture == "relaxed":
        relax_picture(out)
    else:
        mark_long_bonds(out)
    return out


# ==============================================================================================
# 3b. Non-orientable Möbius-strip & Klein-bottle surface builders (C11)
# ==============================================================================================
def _canon_axis(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, float)
    u = u / _norm2(u)
    return -u if (u[1] < -1e-7 or (abs(u[1]) < 1e-7 and u[0] < 0.0)) else u


def _period_and_spacing_along(A: np.ndarray, u_ax: np.ndarray) -> tuple[float, float]:
    u_ax = _canon_axis(u_ax)
    n_ax = np.array([-u_ax[1], u_ax[0]])
    per_u: float | None = None
    per_n: float | None = None
    delta_n: float | None = None
    for m1 in range(-8, 9):
        for m2 in range(-8, 9):
            if m1 == 0 and m2 == 0:
                continue
            v = A @ np.array([m1, m2], float)
            vu = float(v @ u_ax)
            vn = float(v @ n_ax)
            if abs(vn) < 1e-5 and vu > 1e-5 and (per_u is None or vu < per_u - 1e-7):
                per_u = vu
            if abs(vu) < 1e-5 and vn > 1e-5 and (per_n is None or vn < per_n - 1e-7):
                per_n = vn
            if abs(vn) > 1e-5 and (delta_n is None or abs(vn) < delta_n - 1e-7):
                delta_n = abs(vn)
    if per_u is None or per_n is None or delta_n is None:
        raise ConfigError("axis direction is not crystallographic on this lattice")
    return per_u, per_n


def _h_pow_iso(h: tuple[np.ndarray, np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    res = _ident()
    if k == 0:
        return res
    step = h if k > 0 else iso_inverse(h)
    for _ in range(abs(int(k))):
        res = iso_compose(step, res)
    return res


def _line_deck_data(cell: UnitCell, line_rec: dict, length: int) -> dict:
    if int(length) < 2:
        raise ConfigError(f"strip length must be at least 2 lattice periods (got {length})")
    u_ax = _canon_axis(line_rec["direction"])
    n_ax = np.array([-u_ax[1], u_ax[0]])
    per_u, per_n = _period_and_spacing_along(cell.A, u_ax)
    glide = float(line_rec.get("glide", 0.0))
    g = line_rec.get("g") or reflection_across(line_rec["point"], u_ax, glide)
    M, t = g
    if float(t @ u_ax) < -1e-9:
        u_ax = -u_ax
        n_ax = -n_ax
    glide_pos = abs(float(t @ u_ax)) % per_u
    if glide_pos < 1e-5 or abs(per_u - glide_pos) < 1e-5:
        glide_pos = 0.0
    x0 = n_ax * float(t @ n_ax) / 2.0
    ell_u = glide_pos + (int(length) - 1) * per_u if glide_pos > 1e-6 else int(length) * per_u
    h = (M, x0 - M @ x0 + ell_u * u_ax)
    hinv = iso_inverse(h)
    return dict(
        g=g,
        M=M,
        u_ax=u_ax,
        n_ax=n_ax,
        per_u=per_u,
        per_n=per_n,
        glide=glide_pos,
        x0=x0,
        ell_u=ell_u,
        n_periods=int(length),
        h=h,
        hinv=hinv,
    )


def build_mobius_surface(
    cell: UnitCell,
    L: Perfect | None,
    line_rec: dict,
    length: int = 8,
    width: float | None = None,
    lobes: list[tuple[float, float, float, float]] | None = None,
    picture: str = "raw",
) -> Defect:
    """Build a crystal Möbius strip for mirror/glide class `line_rec` with full local frames (`reference`,
    `vec`, `g = hinv` for `j = +1` and `h` for `j = -1`) so all graph-only ribbon invariants apply."""
    L = L or Perfect(cell)
    lmax = max(_norm2(L.pos(k2) - L.pos((0, 0, s))) for s in L.rule for k2, _ in L.nbrs((0, 0, s)))
    scale = max(_norm2(cell.a1), _norm2(cell.a2))
    dd = _line_deck_data(cell, line_rec, int(length))
    u_ax, n_ax, x0, ell_u, h, hinv = dd["u_ax"], dd["n_ax"], dd["x0"], dd["ell_u"], dd["h"], dd["hinv"]
    w_band = float(max(3.4 * lmax, 2.4 * scale) if width is None else width)

    if not lobes:
        d_raw = mobius_strip(cell, dd["g"], length=int(length), width=w_band)
        keys = list(d_raw.sites)
        X_raw = np.array([L.pos(k) for k in keys], float).reshape(-1, 2)
        nb: list[dict[int, dict]] = [dict() for _ in keys]
        for (u, v), ed in d_raw.edge.items():
            j = int(ed.get("j", 0))
            typ = ed.get("type", 0)
            if j == 0:
                nb[u][v] = dict(vec=X_raw[v] - X_raw[u], type=typ)
            elif j == 1:
                q = iso_apply(h, X_raw[v])
                nb[u][v] = dict(vec=q - X_raw[u], type=typ, g=hinv, seam=True, j=1)
            elif j == -1:
                q = iso_apply(hinv, X_raw[v])
                nb[u][v] = dict(vec=q - X_raw[u], type=typ, g=h, seam=True, j=-1)
    else:
        max_r_lobe = max(r_lb + abs(d_lb) for _, d_lb, r_lb, _ in lobes)

        def _in_sheet(pos: np.ndarray) -> bool:
            a = float((pos - x0) @ u_ax)
            d = float((pos - x0) @ n_ax)
            if not (-1e-9 <= a < ell_u - 1e-9):
                return False
            if abs(d) <= w_band + 1e-9:
                return True
            for u_lb, d_lb, r_lb, sgn_lb in lobes:
                if sgn_lb * d > w_band and abs(a - u_lb) <= r_lb:
                    over = max(0.0, sgn_lb * (d - d_lb))
                    if math.hypot(a - u_lb, over) <= r_lb:
                        return True
            return False

        keys = [
            k
            for k in L.sites_in_disk(x0 + 0.5 * ell_u * u_ax, ell_u + w_band + max_r_lobe + 4.0 * lmax)
            if _in_sheet(L.pos(k))
        ]
        idx_map = {k: i for i, k in enumerate(keys)}
        X_raw = np.array([L.pos(k) for k in keys], float).reshape(-1, 2)
        nb = [dict() for _ in keys]
        for k, i in idx_map.items():
            pk = X_raw[i]
            for k2, typ in L.nbrs(k):
                q = L.pos(k2)
                a = float((q - x0) @ u_ax)
                d = float((q - x0) @ n_ax)
                j, img, g_edge = 0, k2, None
                if a >= ell_u - 1e-9 and abs(d) <= w_band + 1e-9:
                    j, img, g_edge = 1, _fast_locate(iso_apply(hinv, q), L.Ainv, cell.basis), hinv
                elif a < -1e-9 and abs(d) <= w_band + 1e-9:
                    j, img, g_edge = -1, _fast_locate(iso_apply(h, q), L.Ainv, cell.basis), h
                v = idx_map.get(img) if img is not None else None
                if v is None or v == i:
                    continue
                e = dict(vec=q - pk, type=typ)
                if g_edge is not None:
                    e["g"] = g_edge
                    e["seam"] = True
                    e["j"] = j
                nb[i][v] = e

    # Peel degree <= 1 boundary spurs and keep largest component so V - E + F_int == 0 strictly
    alive = [True] * len(keys)
    changed = True
    while changed:
        changed = False
        for i in range(len(keys)):
            if alive[i] and len(nb[i]) <= 1:
                for v in list(nb[i]):
                    nb[v].pop(i, None)
                nb[i] = {}
                alive[i] = False
                changed = True
    live_set = {i for i in range(len(keys)) if alive[i]}
    if not live_set:
        raise ConfigError("Möbius strip width/length left no connected crystal sites")
    comps: list[set[int]] = []
    seen_c: set[int] = set()
    for st in live_set:
        if st in seen_c:
            continue
        comp: set[int] = set()
        dq = deque([st])
        seen_c.add(st)
        while dq:
            u = dq.popleft()
            comp.add(u)
            for v in nb[u]:
                if v in live_set and v not in seen_c:
                    seen_c.add(v)
                    dq.append(v)
        comps.append(comp)
    main = max(comps, key=len)
    live = sorted(main)
    remap = {i: n for n, i in enumerate(live)}
    edge_out: dict[tuple[int, int], dict] = {}
    for i in live:
        ui = remap[i]
        for v, ed in nb[i].items():
            if v in remap:
                edge_out[(ui, remap[v])] = dict(ed)
    ref = X_raw[live].copy()
    info = dict(
        reference=ref,
        mobius=True,
        klein=False,
        torus=False,
        wraps={},
        T=None,
        lmax=lmax,
        axis=u_ax,
        normal=n_ax,
        point=x0,
        length=ell_u,
        ell_u=ell_u,
        n_periods=int(length),
        width=w_band,
        lobes=list(lobes) if lobes else [],
        glide=dd["glide"],
        h=h,
        seam_id=line_rec.get("id"),
        perfect_sites=len(live),
        guess=ref.copy(),
        removed=0,
        inserted=0,
    )
    out = Defect("mobius", [(keys[i], 0) for i in live], ref.copy(), edge_out, info)
    if picture == "relaxed":
        relax_picture(out)
    else:
        mark_long_bonds(out)
    return out


def build_klein_surface(
    cell: UnitCell,
    L: Perfect | None,
    Ainv: np.ndarray | None,
    line_rec: dict,
    n_u: int = 10,
    n_perp: int = 10,
    picture: str = "raw",
) -> Defect:
    """Build a closed boundary-free crystal Klein bottle along mirror/glide axis `line_rec` (glued by
    improper `h = (M, x0 - M x0 + ell_u u_ax)`) and transverse Bravais lattice period `T_perp = ell_n n_ax`
    (satisfying `h(q + T_perp) = h(q) - T_perp`)."""
    if int(n_u) < 4 or int(n_perp) < 4:
        raise ConfigError(f"Klein bottle periods (n_u={n_u}, n_perp={n_perp}) must both be >= 4")
    L = L or Perfect(cell)
    Ainv = L.Ainv if Ainv is None else Ainv
    lmax = max(_norm2(L.pos(k2) - L.pos((0, 0, s))) for s in L.rule for k2, _ in L.nbrs((0, 0, s)))
    dd = _line_deck_data(cell, line_rec, int(n_u))
    u_ax, n_ax, per_u, per_n, x0, ell_u, h = (
        dd["u_ax"],
        dd["n_ax"],
        dd["per_u"],
        dd["per_n"],
        dd["x0"],
        dd["ell_u"],
        dd["h"],
    )
    ell_n = int(n_perp) * per_n
    T_perp = ell_n * n_ax
    # Generic irrational-like fundamental-domain offsets so no lattice site sits on a domain boundary
    a0 = -0.13731 * per_u
    d0 = -0.5 * ell_n - 0.17319 * per_n

    ctr = x0 + (a0 + 0.5 * ell_u) * u_ax + (d0 + 0.5 * ell_n) * n_ax
    rad = 0.5 * math.hypot(ell_u, ell_n) + 3.0 * lmax
    keys = []
    for k in L.sites_in_disk(ctr, rad):
        p = L.pos(k)
        a = float((p - x0) @ u_ax)
        d = float((p - x0) @ n_ax)
        if (a0 <= a < a0 + ell_u) and (d0 <= d < d0 + ell_n):
            keys.append(k)
    if not keys:
        raise ConfigError("Klein bottle fundamental domain contains no crystal sites")
    idx_map = {k: i for i, k in enumerate(keys)}
    X = np.array([L.pos(k) for k in keys], float).reshape(-1, 2)
    edge: dict[tuple[int, int], dict] = {}
    for k, i in idx_map.items():
        pk = X[i]
        for k2, typ in L.nbrs(k):
            q = L.pos(k2)
            ku = int(math.floor((float((q - x0) @ u_ax) - a0) / ell_u))
            g_u = _h_pow_iso(h, -ku)
            q1 = iso_apply(g_u, q)
            mp = int(math.floor((float((q1 - x0) @ n_ax) - d0) / ell_n))
            q2 = q1 - mp * T_perp
            kt = _fast_locate(q2, Ainv, cell.basis)
            if kt is None or kt not in idx_map:
                raise ConfigError("Klein bottle periodic gluing missed a target site; check lattice symmetry")
            v = idx_map[kt]
            if v == i:
                raise ConfigError("Klein bottle periods are too small (self-loop bond); increase n_u or n_perp")
            if (i, v) in edge:
                raise ConfigError("Klein bottle periods are too small (double bond); increase n_u or n_perp")
            g_uv = iso_compose((np.eye(2), -mp * T_perp), g_u)
            e = dict(vec=q - pk, type=typ)
            if ku != 0 or mp != 0:
                e["g"] = g_uv
                e["seam"] = True
                e["ku"] = ku
                e["mp"] = mp
            edge[(i, v)] = e

    info = dict(
        reference=X.copy(),
        klein=True,
        torus=True,
        mobius=False,
        wraps={},
        T=None,
        lmax=lmax,
        axis=u_ax,
        normal=n_ax,
        point=x0,
        length=ell_u,
        ell_u=ell_u,
        ell_n=ell_n,
        per_u=per_u,
        per_n=per_n,
        n_u=int(n_u),
        n_perp=int(n_perp),
        a0=a0,
        d0=d0,
        glide=dd["glide"],
        h=h,
        T_perp=T_perp,
        seam_id=line_rec.get("id"),
        perfect_sites=len(keys),
        guess=X.copy(),
        removed=0,
        inserted=0,
    )
    out = Defect("klein", [(k, 0) for k in keys], X.copy(), edge, info)
    if picture == "relaxed":
        relax_picture(out)
    else:
        mark_long_bonds(out)
    return out


class _SurfaceBuilder(_ConeSafeBuilder):
    """Cone-safe Volterra defect and cross-seam dipole builder on Möbius strips and Klein bottles."""

    def __init__(self, cell: UnitCell, L: Perfect, surface: Defect):
        self.cell = cell
        self.L = L
        self.A = cell.A
        self.Ainv = L.Ainv
        self.basis = cell.basis
        self.surface_kind = surface.kind
        self.surf_info = dict(surface.info)
        self.u_ax = np.asarray(surface.info["axis"], float)
        self.n_ax = np.asarray(surface.info["normal"], float)
        self.x0 = np.asarray(surface.info["point"], float)
        self.ell_u = float(surface.info["ell_u"])
        self.ell_n = float(surface.info.get("ell_n", 0.0))
        self.width = float(surface.info.get("width", 0.0))
        self.lobes = list(surface.info.get("lobes", ()))
        self.is_klein = bool(surface.info.get("klein"))
        self.centre = self.x0 + 0.5 * self.ell_u * self.u_ax
        self.R = self.ell_u + max(self.width, self.ell_n)
        self.lmax = float(surface.info["lmax"])
        self.scale = max(_norm2(cell.a1), _norm2(cell.a2))
        self._max_dev_u = 0.46 * self.ell_u
        self._max_dev_n = 0.46 * self.ell_n if self.is_klein else 1e9

        n = len(surface.sites)
        self.label = [
            s if (isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], tuple)) else (s, 0)
            for s in surface.sites
        ]
        ref_arr = np.asarray(surface.info["reference"], float)
        self.ref = [ref_arr[i].copy() for i in range(n)]
        self.alive = [True] * n
        self.nb = [dict() for _ in range(n)]
        for (u, v), ed in surface.edge.items():
            e_clean = {k: val for k, val in ed.items() if k != "ghost"}
            self.nb[u][v] = e_clean
        self.X = [surface.positions[i].copy() for i in range(n)]
        self.layers = Counter(lbl[0] for lbl in self.label)
        self.touched: set[int] = set()
        self.cone_sites: set[int] = set(surface.info.get("cone_sites", ()))
        self.records: list[dict] = list(surface.info.get("defects", ()))
        self.perfect_sites = int(surface.info.get("perfect_sites", n))

    def clear(self, p: np.ndarray) -> float:
        if self.is_klein:
            return 0.45 * min(self.ell_u, self.ell_n)
        pos = np.asarray(p, float)
        a = float((pos - self.x0) @ self.u_ax)
        d = float((pos - self.x0) @ self.n_ax)
        for u_lb, d_lb, r_lb, sgn_lb in self.lobes:
            if abs(a - u_lb) <= r_lb and abs(d - d_lb) <= r_lb:
                return min(r_lb - abs(a - u_lb), r_lb - max(0.0, sgn_lb * (d - d_lb)), abs(d) + self.width)
        return self.width - abs(d)

    def base_site(self, p: np.ndarray, inside) -> int:
        # Pick an alive non-cone site directly in F0 so D[base] = (I, 0) has det = +1
        best = None
        for i in range(len(self.label)):
            if self.alive[i] and i not in self.cone_sites and inside(self.ref[i]):
                key = (round(_norm2(self.ref[i] - p), 7), self.label[i][1], i)
                if best is None or key < best:
                    best = key
        if best is None or best[0] > self.lmax + self.scale:
            raise ConfigError(
                f"no F0 crystal site near {np.round(p, 4).tolist()}: choose a point inside the fundamental domain away from earlier cuts"
            )
        return best[2]

    def develop(self, base: int, inside):
        y_anchor = getattr(self, "_dev_anchor", None)
        if y_anchor is None:
            y_anchor = self.ref[base]

        def bounded_inside(yv: np.ndarray) -> bool:
            if abs(float((yv - y_anchor) @ self.u_ax)) > self._max_dev_u:
                return False
            if self.is_klein and abs(float((yv - y_anchor) @ self.n_ax)) > self._max_dev_n:
                return False
            return bool(inside(yv))

        return super().develop(base, bounded_inside)

    def add_cross_seam_dipole(self, b: np.ndarray, p1: np.ndarray, p2_lifted: np.ndarray) -> str:
        """Cut a Volterra dislocation dipole between `p1` (in F0) and `p2_lifted` (lifted across a seam,
        e.g. `h(p2)` in F1), supporting BOTH removal (`cross2(p2_lifted - p1, b) < 0`) and insertion
        (`cross2(p2_lifted - p1, b) > 0`). In F0, Core 1 at `p1` acquires K5 CCW Burgers vector `+b`
        and Core 2 at `p2 = h^-1(p2_lifted)` acquires `+M_h @ b` (`-b` if `p2_lifted == p2` in F0)."""
        L = self.L
        b = np.asarray(b, float)
        p1 = np.asarray(p1, float)
        p2_lifted = np.asarray(p2_lifted, float)
        lb = _norm2(b)
        fb = self.Ainv @ b
        if lb < 1e-9 or abs(float(fb[0]) - round(float(fb[0]))) > 1e-5 or abs(float(fb[1]) - round(float(fb[1]))) > 1e-5:
            raise ConfigError(f"b = {np.round(b, 4).tolist()} is not a non-zero lattice translation")
        d_vec = p2_lifted - p1
        lw = _norm2(d_vec)
        if lw < 1e-9:
            raise ConfigError("dipole endpoints p1 and p2_lifted must be distinct")
        cr = _cross2v(d_vec, b) / (lw * lb)
        if abs(cr) < 1e-6:
            raise ConfigError("cross-seam dipole line p2_lifted - p1 is collinear with b; offset p1/p2 transversely")
        how = "removal" if cr < 0.0 else "insertion"
        w_vec = -b if how == "removal" else b  # points to the LEFT of d_vec (cross2(d_vec, w_vec) > 0)

        dh = d_vec / lw
        nh = np.array([-dh[1], dh[0]])
        wn = float(w_vec @ nh)
        wa = float(w_vec @ dh)
        m = 1.25 * self.lmax
        a_lo = min(0.0, wa) - m
        a_hi = lw + max(0.0, wa) + m
        n_lo = min(0.0, wn) - m
        n_hi = max(0.0, wn) + m

        # Verify chart will not wrap all the way around ell_u or ell_n and meet itself
        corners = [
            p1 + a_lo * dh + n_lo * nh,
            p1 + a_lo * dh + n_hi * nh,
            p1 + a_hi * dh + n_lo * nh,
            p1 + a_hi * dh + n_hi * nh,
        ]
        u_vals = [float((c - self.x0) @ self.u_ax) for c in corners]
        if max(u_vals) - min(u_vals) >= self.ell_u - 1.2 * self.lmax:
            raise ConfigError("cross-seam dipole span is too wide for strip/bottle length; increase length / n_u")
        if self.is_klein:
            n_vals = [float((c - self.x0) @ self.n_ax) for c in corners]
            if max(n_vals) - min(n_vals) >= self.ell_n - 1.2 * self.lmax:
                raise ConfigError("cross-seam dipole span is too wide for Klein transverse period; increase n_perp")
        else:
            # Check transverse boundary clearance on Möbius strip
            if max(abs(float((c - self.x0) @ self.n_ax)) for c in corners) > self.width - 0.5 * self.lmax:
                raise ConfigError("cross-seam dipole approaches Möbius transverse boundary; increase strip width")

        def inside(y: np.ndarray) -> bool:
            r = y - p1
            a = float(r @ dh)
            n = float(r @ nh)
            return (a_lo <= a <= a_hi) and (n_lo <= n <= n_hi)

        mid = 0.5 * (p1 + p2_lifted) + 0.5 * w_vec
        old_u, old_n = self._max_dev_u, self._max_dev_n
        self._dev_anchor = mid
        self._max_dev_u = 0.52 * (max(u_vals) - min(u_vals)) + 1.5 * self.lmax
        if self.is_klein:
            self._max_dev_n = 0.52 * (max(n_vals) - min(n_vals)) + 1.5 * self.lmax
        try:
            base = self.base_site(p1, inside)
            D, Y, _ = self.develop(base, inside)
        finally:
            self._dev_anchor = None
            self._max_dev_u, self._max_dev_n = old_u, old_n

        N = sorted(D)
        Binv = np.linalg.inv(np.column_stack([d_vec, w_vec]))
        st = {i: tuple(Binv @ (Y[i] - p1)) for i in N}
        in_p = [i for i in N if (1e-9 < st[i][0] < 1.0 - 1e-9) and (-1e-9 <= st[i][1] < 1.0 - 1e-9)]
        in_p_set = set(in_p)

        rule: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}

        def add_chart(entries: list[tuple[int, np.ndarray, int]]):
            here: dict[tuple, tuple[int, int]] = {}
            for idx_site, y_pos, sh in entries:
                kloc = _fast_locate(y_pos, self.Ainv, self.basis)
                if kloc is not None:
                    here[kloc] = (idx_site, sh)
            for img, (u, su) in here.items():
                p_img = L.pos(img)
                for k2, typ in L.nbrs(img):
                    hit = here.get(k2)
                    if hit is None or hit[0] == u:
                        continue
                    v, sv = hit
                    rule[(u, v)] = (L.pos(k2) - p_img, su - sv, typ)

        if how == "removal":
            kept = [i for i in N if i not in in_p_set]
            fresh: list[int] = []
            # Chart 1 & 2: end-caps at s <= 0 and s >= 1 with right side t < 0
            add_chart([(i, Y[i], 0) for i in kept if st[i][0] <= 1e-9 or st[i][1] < -1e-9])
            add_chart([(i, Y[i], 0) for i in kept if st[i][0] >= 1.0 - 1e-9 or st[i][1] < -1e-9])
            # Chart 3: full left side t >= 1 across ALL s so s=0 and s=1 end-caps close without slits
            add_chart([(i, Y[i], 0) for i in kept if st[i][1] >= 1.0 - 1e-9])
            # Chart 4: glue across removed parallelogram (1e-9 < s < 1 - 1e-9)
            add_chart(
                [(i, Y[i], 0) for i in kept if (1e-9 < st[i][0] < 1.0 - 1e-9) and st[i][1] < -1e-9]
                + [(i, Y[i] - w_vec, -1) for i in kept if (1e-9 < st[i][0] < 1.0 - 1e-9) and st[i][1] >= 1.0 - 1e-9]
            )
            for i in in_p:
                self.kill(i)
            self._apply(rule, kept, D, lambda j: (np.eye(2), j * w_vec), new_sites=())
            n_rem, n_ins = len(in_p), 0
        else:
            kept = list(N)
            fresh = []
            q_entries: list[tuple[int, np.ndarray, int]] = []
            for i in in_p:
                qi = self.new_copy(i)
                fresh.append(qi)
                D[qi] = D[i]
                self.X[qi] = self.X[i] - 0.35 * (D[i][0].T @ w_vec)
                q_entries.append((qi, Y[i], 0))
            add_chart([(i, Y[i], 0) for i in kept if st[i][0] <= 1e-9 or st[i][1] < -1e-9])
            add_chart([(i, Y[i], 0) for i in kept if st[i][0] >= 1.0 - 1e-9 or st[i][1] < -1e-9])
            add_chart([(i, Y[i], 0) for i in kept if st[i][1] >= -1e-9])
            add_chart(q_entries + [(i, Y[i], 0) for i in kept if st[i][1] < -1e-9])
            add_chart(
                q_entries
                + [(i, Y[i] + w_vec, +1) for i in kept if (1e-9 < st[i][0] < 1.0 - 1e-9) and st[i][1] >= -1e-9]
            )
            self._apply(rule, kept + fresh, D, lambda j: (np.eye(2), j * w_vec), new_sites=fresh)
            n_rem, n_ins = 0, len(fresh)

        self.touched |= set(kept) | set(fresh)
        self.records.append(
            dict(
                kind="cross_seam_dipole",
                b=b,
                at=p1,
                to_lifted=p2_lifted,
                how=how,
                removed=n_rem,
                inserted=n_ins,
                core_site=None,
            )
        )
        return how

    def finish(self, kind: str | None = None) -> Defect:
        self.prune_rim_spurs()
        live = [i for i in range(len(self.label)) if self.alive[i]]
        new = {i: n for n, i in enumerate(live)}
        edge = {}
        for i in live:
            ui = new[i]
            for v, e in self.nb[i].items():
                edge[(ui, new[v])] = dict(e)
        ref = np.array([self.ref[i] for i in live]).reshape(-1, 2)
        X = np.array([self.X[i] for i in live]).reshape(-1, 2)
        recs = []
        for r in self.records:
            r2 = dict(r)
            if r2.get("core_site") is not None:
                r2["core_site"] = new.get(r2["core_site"])
            recs.append(r2)
        info = dict(self.surf_info)
        info.update(
            reference=ref,
            wraps={},
            T=None,
            lmax=self.lmax,
            touched=sorted(new[i] for i in self.touched if i in new),
            cone_sites=sorted(new[i] for i in self.cone_sites if i in new),
            defects=recs,
            perfect_sites=self.perfect_sites,
            guess=X.copy(),
            removed=sum(r.get("removed", 0) for r in recs),
            inserted=sum(r.get("inserted", 0) for r in recs),
        )
        info.pop("_orientable", None)
        info.pop("_rim", None)
        info.pop("_cmp_prep", None)
        return Defect(kind or self.surface_kind, [self.label[i] for i in live], X, edge, info)


# --- Graph comparison helpers ---
_PERF_HASH_CACHE: dict[tuple[int, int], set[str]] = {}


def _deg_graph(d: Defect) -> nx.Graph:
    G = d.to_networkx()
    deg = dict(G.degree())
    for v in G:
        G.nodes[v]["deg"] = str(deg[v])
    return G


def _ball_wl_hash(G: nx.Graph, v: int, radius: int = 2) -> str:
    H = G.subgraph(nx.single_source_shortest_path_length(G, v, cutoff=radius)).copy()
    return nx.weisfeiler_lehman_graph_hash(H, node_attr="deg", iterations=3)


def _perfect_ball_hashes(cell: UnitCell, L: Perfect, radius: int = 2) -> set[str]:
    key = (id(cell), radius)
    if key not in _PERF_HASH_CACHE:
        p0 = _ConeSafeBuilder(cell, L, np.zeros(2), 1.0)
        B = _ConeSafeBuilder(cell, L, np.zeros(2), (2 * radius + 3.5) * p0.lmax + 2.2 * p0.scale)
        d = B.finish()
        G = _deg_graph(d)
        X = d.info["reference"]
        inner = [i for i in range(len(d.sites)) if _norm2(X[i]) < 1.5 * p0.scale + 1e-9]
        _PERF_HASH_CACHE[key] = {_ball_wl_hash(G, i, radius) for i in inner}
    return _PERF_HASH_CACHE[key]


def rim_sites(d: Defect) -> set[int]:
    if d.info.get("torus") or d.info.get("klein"):
        return set()
    if "_rim" not in d.info:
        if is_orientable_surface(d):
            faces = frame_faces(d)
            d.info["_rim"] = set(faces[_outer_face_index(faces)]) if faces else set()
        else:
            rfaces = ribbon_faces(d)
            if rfaces:
                k = max(range(len(rfaces)), key=lambda i: len(rfaces[i][0]))
                d.info["_rim"] = set(rfaces[k][0])
            else:
                d.info["_rim"] = set()
    return d.info["_rim"]


def irregular_sites(d: Defect, cell: UnitCell, L: Perfect, radius: int = 2) -> tuple[set[int], nx.Graph]:
    G = _deg_graph(d)
    good = _perfect_ball_hashes(cell, L, radius)
    rim = rim_sites(d)
    near_rim = set(nx.multi_source_dijkstra_path_length(G, rim, cutoff=radius)) if rim else set()
    todo = d.info.get("touched")
    if todo is None:
        cand = range(len(d.sites))
    elif todo:
        cand = set(nx.multi_source_dijkstra_path_length(G, set(todo), cutoff=radius))
    else:
        cand = set()
    K = {v for v in cand if v not in near_rim and _ball_wl_hash(G, v, radius) not in good}
    for c in cores(d):
        K |= set(c["sites"])
    return K, G


def subgraph_ball(G: nx.Graph, K: set[int], r: int) -> nx.Graph:
    if not K:
        return nx.Graph()
    return G.subgraph(nx.multi_source_dijkstra_path_length(G, K, cutoff=r)).copy()


def patch_room(d: Defect, K: set[int], G: nx.Graph) -> int:
    rim = rim_sites(d)
    if not K or not rim:
        return 10**6
    dist = nx.multi_source_dijkstra_path_length(G, K)
    return min((dist[v] for v in rim if v in dist), default=10**6)


def isomorphic_deg_graphs(G1: nx.Graph, G2: nx.Graph) -> bool:
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False
    if G1.number_of_nodes() == 0:
        return True
    if nx.weisfeiler_lehman_graph_hash(G1, node_attr="deg", iterations=4) != nx.weisfeiler_lehman_graph_hash(
        G2, node_attr="deg", iterations=4
    ):
        return False
    return bool(nx.vf2pp_is_isomorphic(G1, G2, node_label="deg"))


# ==============================================================================================
# 5. Exact Algebraic + Constructive Wallpaper Fusion Engine (FusionLab)
# ==============================================================================================
class FusionLab:
    """Unified interface for minimal lattices, proper-symmetry defect catalogues, multi-defect
    combination, 3-level equivalence, and constructive defect fusion rules of a wallpaper group."""

    def __init__(self, group_or_cell: str | UnitCell = "p4g", rank: int = 1, **params):
        self.minimal_entry: Entry | None = None
        if isinstance(group_or_cell, UnitCell):
            given = group_or_cell
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.group = identify_wallpaper_group(given, use_types=False)
                prim = primitive_cell(given, use_types=False)
            seen, bonds = set(), []
            for bd in prim.bonds:
                if bd[:4] not in seen:
                    seen.add(bd[:4])
                    bonds.append(bd)
            self.cell = dataclasses.replace(prim, bonds=bonds)
            self.name = given.label
        elif isinstance(group_or_cell, str) and group_or_cell in WALLPAPER_GROUPS and not params:
            ckey = (group_or_cell, int(rank))
            if ckey in _MINIMAL_LAB_CACHE:
                cached = _MINIMAL_LAB_CACHE[ckey]
                self.cell = cached["cell"]
                self.minimal_entry = cached["minimal_entry"]
                self.group = cached["group"]
                self.name = cached["name"]
                self.cat = cached["cat"]
                self.lattice = cached["lattice"]
                self.planar = cached["planar"]
                self.lmax = cached["lmax"]
                self.scale = cached["scale"]
                return
            self.cell, self.minimal_entry = minimal_cell(group_or_cell, rank=rank, planar_only=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.group = identify_wallpaper_group(self.cell, use_types=False)
            self.name = f"Minimal {self.group} #{rank} ({len(self.cell.basis)} site{'s' if len(self.cell.basis) > 1 else ''}/cell)"
            self.cat = ProperCatalogue(self.cell)
            self.cell = self.cat.cell
            self.lattice = self.cat.lattice
            self.planar = not has_crossings(self.cell)
            self.lmax = max(
                _norm2(self.lattice.pos(k2) - self.lattice.pos((0, 0, s)))
                for s in self.lattice.rule
                for k2, _ in self.lattice.nbrs((0, 0, s))
            )
            self.scale = max(_norm2(self.cell.a1), _norm2(self.cell.a2))
            _MINIMAL_LAB_CACHE[ckey] = dict(
                cell=self.cell,
                minimal_entry=self.minimal_entry,
                group=self.group,
                name=self.name,
                cat=self.cat,
                lattice=self.lattice,
                planar=self.planar,
                lmax=self.lmax,
                scale=self.scale,
            )
            return
        else:
            given = make_cell(group_or_cell, **params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.group = identify_wallpaper_group(given, use_types=False)
                prim = primitive_cell(given, use_types=False)
            seen, bonds = set(), []
            for bd in prim.bonds:
                if bd[:4] not in seen:
                    seen.add(bd[:4])
                    bonds.append(bd)
            self.cell = dataclasses.replace(prim, bonds=bonds)
            self.name = given.label

        self.cat = ProperCatalogue(self.cell)
        self.cell = self.cat.cell
        self.lattice = self.cat.lattice
        self.planar = not has_crossings(self.cell)
        self.lmax = max(
            _norm2(self.lattice.pos(k2) - self.lattice.pos((0, 0, s)))
            for s in self.lattice.rule
            for k2, _ in self.lattice.nbrs((0, 0, s))
        )
        self.scale = max(_norm2(self.cell.a1), _norm2(self.cell.a2))

    # ------------------------------------------------------------------------------------------
    # Stage 1: Minimal lattices & defect catalogue
    # ------------------------------------------------------------------------------------------
    def minimal_candidates(self, n: int = 5, kind: str = "planar") -> list[Entry]:
        """Return up to `n` ranked minimal-lattice candidates for this wallpaper group."""
        return minimal_lattice_entries(self.group, kind=kind, n=n)

    def plot_lattice(self, n: int = 3, symmetry: bool = True, ax=None, title: str | None = None):
        """Plot a periodic patch of the active minimal lattice with optional symmetry elements."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots(figsize=(5.5, 5.5))[1]
        lat = WallpaperLattice(self.cell, n=n)
        lat.plot(ax=ax, title=False, show_sites=True, site_size=14, symmetry=symmetry)
        wy = "; ".join(self.minimal_entry.wyckoff) if self.minimal_entry else ""
        ax.set_title(
            title
            or f"{self.name} — group {self.group}\n{len(self.cell.basis)} site(s), {len(self.cell.bonds)} bond(s)/cell"
            + (f"\nWyckoff: {wy}" if wy else ""),
            fontsize=9,
        )
        return ax

    def summary(self, show: bool = True) -> dict:
        info = dict(
            name=self.name,
            group=self.group,
            chiral=self.cat.chiral,
            sites_per_cell=len(self.cell.basis),
            bonds_per_cell=len(self.cell.bonds),
            proper_point_order=len(self.cat.ops_plus),
            full_point_order=len(self.cat.ops),
            elementary_dislocations=[r["id"] for r in self.cat.translations],
            rotation_centres=[
                dict(wyckoff=c["wyckoff"], order=c["order"], point=_r(c["point"]), frac=_r(c["frac"]), on_site=c["on_site"], mirror_of=c["mirror_of"])
                for c in self.cat.centres
            ],
            disclination_classes=[r["id"] for r in self.cat.disclinations],
            moebius_classes=[r["id"] for r in self.cat.lines],
            domain_walls=[r["id"] for r in self.cat.domains],
            recipe=recipe_code(self.minimal_entry) if self.minimal_entry else None,
        )
        if show:
            orient = "proper/chiral (G = G+)" if self.cat.chiral else f"achiral ([G : G+] = 2, |P+| = {len(self.cat.ops_plus)}, |P| = {len(self.cat.ops)})"
            print(f"=== {self.name} | group {self.group} ({orient}) ===")
            print(f"  Primitive cell: a1 = {_r(self.cell.a1)}, a2 = {_r(self.cell.a2)} | {len(self.cell.basis)} site(s), {len(self.cell.bonds)} bond(s)")
            if self.minimal_entry:
                print(f"  Wyckoff orbits: {'; '.join(self.minimal_entry.wyckoff)}")
                print(f"  Recipe:         {recipe_code(self.minimal_entry)}")
            print("  Proper rotation-centre Wyckoff classes (G+):")
            if not self.cat.centres:
                print("    (none — pure translation group p1)")
            for c in self.cat.centres:
                mir = f", mirror partner -> {c['mirror_of']}" if c["mirror_of"] and c["mirror_of"] != c["wyckoff"] else (
                    ", self-mirror" if c["mirror_of"] else ""
                )
                print(
                    f"    {c['wyckoff']:3s} : order {c['order']}, rep = {_r(c['point'])} (frac {_r(c['frac'])}), "
                    f"orbit size {c['size']}/cell" + (" [on site]" if c["on_site"] else " [face/bond]") + mir
                )
        return info

    def list_defects(self, include_nonpoint: bool = True, show: bool = True) -> list[dict]:
        """List all elementary defect classes under G+ (with mirror-partner annotations)."""
        items: list[dict] = []
        for r in self.cat.translations:
            mir = (
                f" [mirror partner: {r['mirror_of']}]"
                if r["mirror_of"] and r["mirror_of"] != r["id"]
                else (" [self-mirror]" if r["mirror_of"] else "")
            )
            items.append(
                dict(
                    num=len(items) + 1,
                    id=r["id"],
                    kind="dislocation",
                    b=r["b"].copy(),
                    length=r["length"],
                    size=r["size"],
                    inverse_id=r["inverse_id"],
                    mirror_of=r["mirror_of"],
                    combinable=True,
                    description=f"Dislocation {r['id']}: b = {_r(r['b'])} (|b| = {r['length']:.3f}, {r['size']} vector(s) in P+){mir}",
                )
            )
        for r in self.cat.disclinations:
            mir = (
                f" [mirror partner: {r['mirror_of']}]"
                if r["mirror_of"] and r["mirror_of"] != r["id"]
                else (" [self-mirror]" if r["mirror_of"] else "")
            )
            items.append(
                dict(
                    num=len(items) + 1,
                    id=r["id"],
                    kind="disclination",
                    wyckoff=r["wyckoff"],
                    angle=r["angle"],
                    centre=r["centre"].copy(),
                    order=r["order"],
                    on_site=r["on_site"],
                    inverse_id=r["inverse_id"],
                    mirror_of=r["mirror_of"],
                    combinable=True,
                    description=f"Disclination {r['id']}: {r['angle']:+.0f}° at {r['wyckoff']} {_r(r['centre'])}"
                    + (" (site)" if r["on_site"] else " (plaquette/bond)")
                    + mir,
                )
            )
        if include_nonpoint:
            for r in self.cat.lines:
                items.append(
                    dict(
                        num=len(items) + 1,
                        id=r["id"],
                        kind="moebius",
                        line_kind=r["kind"],
                        point=r["point"].copy(),
                        direction=r["direction"].copy(),
                        glide=float(r["glide"]),
                        combinable=False,
                        description=f"Möbius strip {r['id']}: {r['kind']} along {_r(r['direction'])} through {_r(r['point'])}"
                        + (f", glide = {r['glide']:.3f}" if r["glide"] else ""),
                    )
                )
            for r in self.cat.domains:
                items.append(
                    dict(
                        num=len(items) + 1,
                        id=r["id"],
                        kind="domain_wall",
                        sigma=r["sigma"],
                        combinable=False,
                        description=f"Domain wall {r['id']}: {r['what']}",
                    )
                )
        if show:
            print(f"Elementary defects of {self.name} (group {self.group}):")
            for it in items:
                tag = "point" if it["combinable"] else "non-point"
                print(f"  {it['num']:2d}. [{it['id']:14s}] ({tag:9s}) {it['description']}")
        return items

    def nearest_centre(self, angle: float, near: tuple[float, float] | np.ndarray = (0.0, 0.0), wyckoff: str | None = None) -> np.ndarray:
        """Snap `near` to the closest rotation centre supporting `angle` (and optional `wyckoff` class)."""
        near_arr = np.asarray(near, float)
        n0 = np.round(self.cat.Ainv @ near_arr).astype(int)
        best = None
        for q, _, w in self.cat.all_centres:
            if wyckoff is not None and w != wyckoff:
                continue
            if not self.lattice.is_symmetry(rotation_about(q, math.radians(angle)), probe=2.2):
                continue
            for i in range(-2, 3):
                for j in range(-2, 3):
                    p = q + self.cell.A @ (n0 + np.array([i, j], float))
                    dist = _norm2(p - near_arr)
                    if best is None or dist < best[0] - 1e-9:
                        best = (dist, p)
        if best is None:
            msg = f"no rotation centre supporting {angle:g}°" + (f" in class {wyckoff}" if wyckoff else "")
            raise ConfigError(msg)
        return best[1]

    def build_elementary(self, spec: str | int, R: float | None = None, picture: str = "relaxed") -> Defect:
        """Build any single elementary defect from `list_defects()` (including Möbius strips and domain walls)."""
        items = self.list_defects(include_nonpoint=True, show=False)
        if isinstance(spec, (int, np.integer)):
            if not (1 <= int(spec) <= len(items)):
                raise ConfigError(f"defect number must be in 1..{len(items)}")
            it = items[int(spec) - 1]
        elif isinstance(spec, str):
            hit = next((x for x in items if x["id"] == spec), None)
            if hit is None and spec in self.cat.by_id:
                r = self.cat.by_id[spec]
                if r["kind"] == "dislocation":
                    it = dict(id=r["id"], kind="dislocation", b=r["b"])
                elif r["kind"] == "disclination":
                    it = dict(id=r["id"], kind="disclination", angle=r["angle"], centre=r["centre"], wyckoff=r["wyckoff"])
                else:
                    raise ConfigError(f"cannot build {spec!r} as elementary defect")
            elif hit is not None:
                it = hit
            else:
                raise ConfigError(f"unknown defect id {spec!r}")
        else:
            raise ConfigError("spec must be a defect id string or 1-based integer")

        if it["kind"] == "dislocation":
            d = self.combine((it["id"], dict()), R=R, picture=picture, name=it["id"])
            return d
        if it["kind"] == "disclination":
            d = self.combine((it["id"], dict()), R=R, picture=picture, name=it["id"])
            return d
        if it["kind"] == "moebius":
            g = reflection_across(it["point"], it["direction"], it["glide"])
            length = 4 if it["line_kind"] == "mirror" else 3
            d = mobius_strip(self.cell, g, length=length, width=max(2.2 * self.scale, 3.0 * self.lmax))
            d.info["name"] = it["id"]
            return d
        if it["kind"] == "domain_wall":
            R_wall = R or max(4.0 * self.scale, 8.0 * self.lmax)
            d = interface(
                self.cell,
                it["sigma"],
                point=self.cell.basis.mean(axis=0) + 0.137 * self.cell.a2,
                direction=self.cell.a1,
                R=R_wall,
            )
            d.info["name"] = it["id"]
            return d
        raise ConfigError(f"unsupported defect kind {it['kind']!r}")

    def plot_catalogue(self, include_nonpoint: bool = True, max_angles: tuple[float, ...] | None = None, ncols: int = 4, size: float = 4.2):
        """Build and plot a visual gallery of elementary defects of this wallpaper group."""
        import matplotlib.pyplot as plt

        items = self.list_defects(include_nonpoint=include_nonpoint, show=False)
        if max_angles is not None:
            allowed = {round(float(a), 3) for a in max_angles}
            items = [it for it in items if it["kind"] != "disclination" or round(float(it["angle"]), 3) in allowed]
        k = len(items)
        if k == 0:
            return None
        ncols = min(ncols, k)
        nrows = math.ceil(k / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows + 0.4), squeeze=False)
        for idx, ax in enumerate(axs.flat):
            if idx >= k:
                ax.axis("off")
                continue
            it = items[idx]
            d = self.build_elementary(it["id"])
            if it["kind"] in ("dislocation", "disclination"):
                self.plot(d, ax=ax, title=f"{it['id']}\n{it['description'].split(':', 1)[1].strip()}")
            else:
                d.plot(ax=ax, site_size=6, title=f"{it['id']}: {it['kind']}")
        fig.suptitle(f"Elementary Defect Catalogue of {self.name} (group {self.group})", fontsize=11, y=1.01)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------------------------------
    # Stage 2: Combining arbitrary lists of dislocations & disclinations, and equivalence
    # ------------------------------------------------------------------------------------------
    def _parse_spec(self, choice: Any, idx: int) -> dict:
        over: dict[str, Any] = {}
        if isinstance(choice, tuple) and len(choice) == 2 and isinstance(choice[1], dict):
            choice, over = choice[0], dict(choice[1])
        if isinstance(choice, (int, np.integer)):
            items = self.list_defects(include_nonpoint=True, show=False)
            if not (1 <= int(choice) <= len(items)):
                raise ConfigError(f"defect index {choice} out of range 1..{len(items)}")
            base = items[int(choice) - 1]
            if not base["combinable"]:
                raise ConfigError(f"{base['id']} ({base['kind']}) is a non-point defect; build it alone with build_elementary()")
            spec = dict(kind=base["kind"])
            if base["kind"] == "dislocation":
                spec["b"] = base["b"].copy()
            else:
                spec["angle"] = base["angle"]
                spec["at"] = base["centre"].copy()
                spec["wyckoff"] = base["wyckoff"]
        elif isinstance(choice, str):
            if choice not in self.cat.by_id:
                raise ConfigError(f"unknown defect id {choice!r}")
            r = self.cat.by_id[choice]
            if r["kind"] == "vacuum":
                raise ConfigError("vacuum 'I' adds no defect; omit it or inspect the perfect patch")
            if r["kind"] not in ("dislocation", "disclination"):
                raise ConfigError(f"{choice!r} is not a point defect; use build_elementary({choice!r})")
            spec = dict(kind=r["kind"])
            if r["kind"] == "dislocation":
                spec["b"] = r["b"].copy()
            else:
                spec["angle"] = r["angle"]
                spec["at"] = r["centre"].copy()
                spec["wyckoff"] = r["wyckoff"]
        elif isinstance(choice, dict):
            spec = dict(choice)
        else:
            raise ConfigError("each defect spec must be an ID ('T1', 'D(4a, +90°)'), list number, (id, overrides), or dict")

        shift = over.pop("shift", spec.pop("shift", None))
        strict = bool(over.pop("strict", spec.pop("strict", False)))
        wyckoff = over.pop("wyckoff", spec.pop("wyckoff", None))
        spec.update(over)
        kind = {"translation": "dislocation", "rotation": "disclination"}.get(spec.get("kind"), spec.get("kind"))
        spec["kind"] = kind

        if kind == "dislocation":
            spec["b"] = np.asarray(spec["b"], float)
            if "at" not in spec:
                # Default generic interior cell point, staggered by index if multiple dislocations are given without positions
                base_pt = 0.371 * self.cell.a1 + 0.413 * self.cell.a2
                if idx > 0:
                    base_pt = base_pt + (3 * idx) * self.cell.a1
                spec["at"] = base_pt
            else:
                spec["at"] = np.asarray(spec["at"], float)
            if shift is not None:
                spec["at"] = spec["at"] + self.cell.A @ np.asarray(shift, float)
        elif kind == "disclination":
            ang = float(spec["angle"])
            spec["angle"] = ang
            if "at" not in spec:
                spec["at"] = self.nearest_centre(ang, (3.0 * idx * self.scale, 0.0), wyckoff=wyckoff)
            else:
                spec["at"] = np.asarray(spec["at"], float)
            if shift is not None:
                spec["at"] = spec["at"] + self.cell.A @ np.asarray(shift, float)
            asked = spec["at"].copy()
            if not strict:
                spec["at"] = self.nearest_centre(ang, asked, wyckoff=None)
                if _norm2(spec["at"] - asked) > 1e-5:
                    spec["moved_from"] = asked
        else:
            raise ConfigError(f"unsupported point-defect kind {kind!r}")
        return spec

    def combine(
        self,
        *choices: Any,
        R: float | None = None,
        centre: tuple[float, float] | np.ndarray | None = None,
        picture: str = "relaxed",
        name: str | None = None,
    ) -> Defect:
        """Combine any mixed list of dislocations and disclinations on the minimal lattice."""
        if len(choices) == 1 and isinstance(choices[0], list):
            choices = tuple(choices[0])
        specs = [self._parse_spec(c, i) for i, c in enumerate(choices)]
        d = build_configuration(self.cell, specs, L=self.lattice, R=R, centre=centre, picture=picture, verify_planar=self.planar)
        for rec, sp in zip(d.info["defects"], specs):
            if "moved_from" in sp:
                rec["moved_from"] = sp["moved_from"]
        d.info["name"] = name or " + ".join(
            (f"b={_r(r['b'], 2)}@{_r(r['at'], 2)}" if r["kind"] == "dislocation" else f"{r['angle']:+.0f}°@{_r(r['at'], 2)}")
            for r in d.info["defects"]
        )
        return d

    def neutral(self, kind: str, torus: Any = None, R: float | None = None, name: str | None = None, **kw) -> Defect:
        """Build neutral configurations on an open patch or torus ('dipole', 'pair', 'quadrupole').
        Pass `torus=True` (or `'auto'`) to choose a sufficiently wide periodic supercell automatically,
        or pass an integer / tuple / 2x2 matrix for an explicit supercell."""
        at = np.asarray(kw.pop("at", (0.0, 0.0)), float)
        to = np.asarray(kw.pop("to"), float)
        if torus is True or torus == "auto":
            span = _norm2(to - at) + (_norm2(np.asarray(kw["strip"], float)) if "strip" in kw else 0.0)
            torus = max(14, int(math.ceil((2.0 * span + 8.0 * self.lmax + 4.0 * self.scale) / max(min(_norm2(self.cell.a1), _norm2(self.cell.a2)), 1e-6))))
        if kind == "dipole":
            b = kw.pop("b")
            if isinstance(b, str) and b in self.cat.by_id:
                b = self.cat.by_id[b]["b"]
            d = dislocation_dipole(self.cell, b, at, to, R=R, supercell=torus, **kw)
        elif kind == "pair":
            if torus is not None:
                raise ConfigError("a disclination pair has a non-zero Burgers vector in general; use 'quadrupole' on a torus")
            a = float(kw.pop("angle"))
            d = disclination_pair(self.cell, a, self.nearest_centre(a, at), self.nearest_centre(-a, to), R=R, **kw)
        elif kind == "quadrupole":
            a = float(kw.pop("angle"))
            strip = kw.pop("strip", None)
            if strip is None:
                raise ConfigError("provide strip= (a non-zero lattice vector leaving c2-c2' away from c1)")
            d = disclination_quadrupole(self.cell, a, self.nearest_centre(a, at), self.nearest_centre(-a, to), strip, R=R, supercell=torus, **kw)
        else:
            raise ConfigError("kind must be 'dipole', 'pair', or 'quadrupole'")
        d.info["name"] = name or f"{kind}" + (" (torus)" if torus is not None else "")
        return d

    def dislocation_cores(self, b_or_id: str | np.ndarray = "T1", grid: int = 6, complete: bool = True):
        """Scan distinct dislocation core graphs for Burgers vector `b_or_id`."""
        b = self.cat.by_id[b_or_id]["b"] if isinstance(b_or_id, str) else np.asarray(b_or_id, float)
        return dislocation_cores(self.cell, b, grid=grid, complete=complete)

    def identify_charge(self, d: Defect) -> dict:
        """Read total topological charge purely from `d`'s combinatorial graph and map to catalogue ID."""
        if d.info.get("klein"):
            seam_info = self.measure_mobius_seam(d)
            return dict(
                class_id=f"Klein({seam_info['seam_id']})",
                kind="klein_neutral",
                angle=total_frank(d),
                frank=total_frank(d),
                H=seam_info["h"],
                seam_id=seam_info["seam_id"],
                orientable=False,
                mirror_of=None,
            )
        if d.info.get("mobius"):
            seam_info = self.measure_mobius_seam(d)
            tf = total_frank(d)
            return dict(
                class_id=f"Mobius({seam_info['seam_id']}, α={tf:+.0f}°)",
                kind="mobius_surface",
                angle=tf,
                frank=tf,
                H=seam_info["h"],
                seam_id=seam_info["seam_id"],
                orientable=False,
                mirror_of=None,
            )
        if d.info.get("torus"):
            return dict(class_id="I", kind="torus_neutral", angle=0.0, frank=0.0, H=_ident(), orientable=True, mirror_of=self.cat.vacuum["mirror_of"])
        H, S = total_charge(d)
        frank = total_frank(d)
        ident_rec = self.cat.identify(frank, H)
        ident_rec["H"] = H
        ident_rec["S"] = S
        ident_rec["frank"] = frank
        ident_rec["orientable"] = True
        return ident_rec

    def inspect(self, d: Defect, show: bool = True) -> dict:
        """Report how `d` changed the lattice: vertices, bonds, Euler characteristic, core rings/sites, and total charge."""
        cs = cores(d)
        core_desc = []
        for c in cs:
            if c["kind"] == "dislocation":
                core_desc.append(f"ring of {c['ring']} sites, Burgers vector b = {_r(c['b'])} (|b| = {_norm2(c['b']):.3f})")
            elif c["site"] is not None:
                core_desc.append(f"site #{c['site']} with {c['bonds']} bonds, Frank angle {c['angle']:+.0f}°")
            else:
                core_desc.append(f"ring of {c['ring']} sites, Frank angle {c['angle']:+.0f}°")
        ch = self.identify_charge(d)
        surf = surface_euler_characteristic(d)
        out = dict(
            name=d.info.get("name", d.kind),
            sites=len(d.sites),
            bonds=len(d.edges),
            faces=surf["F_capped"],
            interior_faces=surf["F_interior"],
            boundary_components=surf["boundary_components"],
            euler=surf["euler_capped"] if surf["orientable"] else surf["euler_surface"],
            euler_surface=surf["euler_surface"],
            orientable=surf["orientable"],
            frames_consistent=(check_frames(d) == 0) if "reference" in d.info else None,
            sites_removed=d.info.get("removed", 0),
            sites_inserted=d.info.get("inserted", 0),
            face_histogram=graph_face_histogram(d) if self.planar else {},
            cores=core_desc,
            charge_id=ch["class_id"],
            mirror_partner=ch.get("mirror_of"),
            total_frank=round(float(ch.get("frank", 0.0)), 4),
            charge_details=ch,
        )
        if show:
            mir = f" (mirror partner: {ch['mirror_of']})" if ch.get("mirror_of") and ch["mirror_of"] != ch["class_id"] else ""
            ori_tag = "orientable" if out["orientable"] else "non-orientable"
            print(f"Crystal '{out['name']}' ({ori_tag}):")
            if out["orientable"]:
                print(f"  Graph:   V = {out['sites']}, E = {out['bonds']}, F = {out['faces']}, V - E + F = {out['euler']}, frames_ok = {out['frames_consistent']}")
            else:
                print(
                    f"  Graph:   V = {out['sites']}, E = {out['bonds']}, F_int = {out['interior_faces']} "
                    f"(bdy={out['boundary_components']}), V - E + F_int = {out['euler_surface']}, frames_ok = {out['frames_consistent']}"
                )
            print(f"  Cut:     removed = {out['sites_removed']} site(s), inserted = {out['sites_inserted']} site(s)")
            print(f"  Charge:  {out['charge_id']}{mir} | total Frank angle = {out['total_frank']:+.1f}°")
            print(f"  Cores ({len(core_desc)}):")
            for line in core_desc:
                print(f"    - {line}")
        return out

    def _prep_compare(self, d: Defect) -> dict:
        if "_cmp_prep" not in d.info:
            K, G = irregular_sites(d, self.cell, self.lattice)
            ch = self.identify_charge(d)
            d.info["_cmp_prep"] = dict(K=K, G=G, room=patch_room(d, K, G), charge=ch, torus=bool(d.info.get("torus")))
        return d.info["_cmp_prep"]

    def _same_proper(self, p: dict, q: dict) -> bool:
        if p["torus"] or q["torus"]:
            return (p["charge"]["class_id"] == "I") and (q["charge"]["class_id"] == "I")
        c1, c2 = p["charge"], q["charge"]
        return bool(
            abs(c1["frank"] - c2["frank"]) < 1e-3
            and same_proper_holonomy(c1["H"], c2["H"], self.cat.ops_plus, self.cell.A, self.cat.Ainv)
        )

    def _same_up_to_mirror(self, p: dict, q: dict) -> bool:
        if self._same_proper(p, q):
            return True
        if self.cat.m0 is None or p["torus"] or q["torus"]:
            return False
        ang_m, Hm = mirror_holonomy(p["charge"]["frank"], p["charge"]["H"], self.cat.m0)
        return bool(
            abs(ang_m - q["charge"]["frank"]) < 1e-3
            and same_proper_holonomy(Hm, q["charge"]["H"], self.cat.ops_plus, self.cell.A, self.cat.Ainv)
        )

    def compare(
        self,
        crystals: list[Defect],
        names: list[str] | None = None,
        radius: int | None = None,
        core_radius: int = 3,
        max_radius: int = 8,
        show: bool = True,
    ) -> dict:
        """Compare defect configurations at 3 levels:
        1. proper_charge (G+ conjugacy class + Frank angle) [plus mirror_charge under full G],
        2. cores (degree-labelled graph isomorphism within `core_radius` bonds of irregular sites),
        3. lattice (degree-labelled graph isomorphism out to `r` bonds)."""
        if not self.planar:
            raise ConfigError("comparison requires a planar lattice")
        ds = list(crystals)
        names = list(names) if names is not None else [str(d.info.get("name", f"#{k+1}")) for k, d in enumerate(ds)]
        prep = [self._prep_compare(d) for d in ds]
        fit = min(p["room"] for p in prep) - 1
        r = min(fit, max_radius) if radius is None else radius

        def group_labels(eq_fn):
            cls, reps = [], []
            for i in range(len(ds)):
                for k, j in enumerate(reps):
                    if eq_fn(i, j):
                        cls.append(k)
                        break
                else:
                    reps.append(i)
                    cls.append(len(reps) - 1)
            return cls

        balls: dict[tuple[int, int], nx.Graph] = {}

        def get_ball(i: int, rad: int) -> nx.Graph:
            if (i, rad) not in balls:
                balls[(i, rad)] = subgraph_ball(prep[i]["G"], prep[i]["K"], rad)
            return balls[(i, rad)]

        cores_ok = fit >= core_radius
        lat_ok = r >= core_radius and fit >= r
        out = dict(
            names=names,
            radius=r,
            room_fit=fit,
            proper_charge=group_labels(lambda i, j: self._same_proper(prep[i], prep[j])),
            mirror_charge=group_labels(lambda i, j: self._same_up_to_mirror(prep[i], prep[j])),
            cores=group_labels(lambda i, j: isomorphic_deg_graphs(get_ball(i, core_radius), get_ball(j, core_radius))) if cores_ok else None,
            lattice=group_labels(lambda i, j: isomorphic_deg_graphs(get_ball(i, r), get_ball(j, r))) if lat_ok else None,
            charge_ids=[p["charge"]["class_id"] for p in prep],
        )
        if show:
            let = lambda k: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[k % 26]
            w = max(len(nm) for nm in names + ["crystal"])
            print(f"{'crystal':{w}s}  G+ charge  G(mirror)  cores(r={core_radius})  lattice(r={r})  identified class")
            for i, nm in enumerate(names):
                c_str = "-" if out["cores"] is None else let(out["cores"][i])
                l_str = "-" if out["lattice"] is None else let(out["lattice"][i])
                print(
                    f"{nm:{w}s}  {let(out['proper_charge'][i]):9s}  {let(out['mirror_charge'][i]):9s}  "
                    f"{c_str:10s}  {l_str:12s}  {out['charge_ids'][i]}"
                )
            if not cores_ok:
                print(f"  [note: patch leaves only {fit} bond(s) around cores < core_radius={core_radius}; rebuild with larger R]")
        return out

    def same(self, d1: Defect, d2: Defect, **kw) -> dict:
        res = self.compare([d1, d2], show=False, **kw)
        return dict(
            proper_charge=(res["proper_charge"][0] == res["proper_charge"][1]),
            mirror_charge=(res["mirror_charge"][0] == res["mirror_charge"][1]),
            cores=None if res["cores"] is None else (res["cores"][0] == res["cores"][1]),
            lattice=None if res["lattice"] is None else (res["lattice"][0] == res["lattice"][1]),
            radius=res["radius"],
        )

    def plot(self, d: Defect, ax=None, title: str | None = None, mark: bool = True):
        """Plot a built Defect crystal, highlighting dislocation core rings (amber) and disclinations (red/blue)."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots(figsize=(6.0, 6.0))[1]
        d.plot(ax=ax, site_size=6)
        sub = ""
        if mark and d.kind not in ("interface", "wall") and (d.kind != "mobius" or "reference" in d.info) and self.planar:
            P = d.positions
            for c in cores(d):
                Q = P[c["sites"]]
                if c["kind"] == "dislocation":
                    if not any(
                        d.edge[(c["sites"][k - 1], c["sites"][k])].get("ghost") is not None for k in range(len(c["sites"]))
                    ):
                        ax.fill(Q[:, 0], Q[:, 1], color="#e09a1a", alpha=0.45, zorder=1)
                    else:
                        ax.scatter(Q[:, 0].mean(), Q[:, 1].mean(), s=130, marker="s", facecolors="none", edgecolors="#e09a1a", linewidths=1.6, zorder=5)
                else:
                    ax.scatter(
                        Q[:, 0].mean(),
                        Q[:, 1].mean(),
                        s=165,
                        facecolors="none",
                        linewidths=1.8,
                        edgecolors="#d9432f" if c["angle"] > 0 else "#1f6fc5",
                        zorder=5,
                    )
            ch = self.identify_charge(d)
            sub = f"\ncharge: {ch['class_id']}"
        ax.set_title(title or (str(d.info.get("name", d.kind)) + sub), fontsize=8.5)
        return ax

    def show(self, *crystals: Defect, names: list[str] | None = None, ncols: int = 3, size: float = 5.2):
        import matplotlib.pyplot as plt

        if len(crystals) == 1 and isinstance(crystals[0], list):
            crystals = tuple(crystals[0])
        n = len(crystals)
        ncols = min(ncols, max(1, n))
        nrows = math.ceil(n / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows + 0.4), squeeze=False)
        for k, ax in enumerate(axs.flat):
            if k < n:
                self.plot(crystals[k], ax=ax, title=None if names is None else names[k])
            else:
                ax.axis("off")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------------------------------
    # Stage 3: Exact Algebraic Fusion Rules & Constructive Lattice Verification
    # ------------------------------------------------------------------------------------------
    def fuse(self, id1: str, id2: str, max_burgers_shells: int = 3) -> dict:
        """Compute the exact fusion rule C1 x C2 -> (+) C^(m) in Conj(\\widetilde{G}_+), together with
        explicit lattice witness specs that realize every channel constructively."""
        if id1 not in self.cat.by_id or id2 not in self.cat.by_id:
            raise ConfigError(f"unknown class id(s): {id1!r}, {id2!r}")
        r1 = self.cat.by_id[id1]
        r2 = self.cat.by_id[id2]

        # Trivial vacuum unit I
        if r1["kind"] == "vacuum":
            return dict(c1=id1, c2=id2, law="I×C", channels=[id2], infinite_tower=False, contains_vacuum=(id2 == "I"), witnesses={id2: []})
        if r2["kind"] == "vacuum":
            return dict(c1=id1, c2=id2, law="C×I", channels=[id1], infinite_tower=False, contains_vacuum=(id1 == "I"), witnesses={id1: []})

        # Law 1: Dislocation x Dislocation
        if r1["kind"] == "dislocation" and r2["kind"] == "dislocation":
            b1 = r1["b"]
            channels: list[str] = []
            witnesses: dict[str, list] = {}
            p1 = 0.371 * self.cell.a1 + 0.413 * self.cell.a2
            p2 = p1 + 3.0 * self.cell.a1 + 1.0 * self.cell.a2
            for M, _ in self.cat.ops_plus:
                b2 = M @ r2["b"]
                bsum = b1 + b2
                ch = self.cat.identify(0.0, (np.eye(2), -bsum))["class_id"]
                if ch not in witnesses:
                    channels.append(ch)
                    witnesses[ch] = [
                        dict(kind="dislocation", b=b1, at=p1),
                        dict(kind="dislocation", b=b2, at=p2),
                    ]
            return dict(
                c1=id1,
                c2=id2,
                law="T×T",
                channels=channels,
                infinite_tower=False,
                contains_vacuum=("I" in channels),
                formula=f"{id1} × {id2} = " + " ⊕ ".join(channels),
                witnesses=witnesses,
            )

        # Law 2: Disclination x Dislocation (or Dislocation x Disclination — commutative!)
        if (r1["kind"] == "disclination" and r2["kind"] == "dislocation") or (
            r1["kind"] == "dislocation" and r2["kind"] == "disclination"
        ):
            disc = r1 if r1["kind"] == "disclination" else r2
            disl = r2 if r1["kind"] == "disclination" else r1
            ang = disc["angle"]
            c0 = disc["centre"]
            H_disc = rotation_about(c0, math.radians(ang))
            det_idx = int(round(abs(np.linalg.det(np.eye(2) - H_disc[0]))))
            channels = []
            witnesses = {}
            p_disl = c0 + 3.371 * self.cell.a1 + 1.413 * self.cell.a2
            for M, _ in self.cat.ops_plus:
                b_vec = M @ disl["b"]
                H_tot = iso_compose(H_disc, (np.eye(2), -b_vec))
                ch = self.cat.identify(ang, H_tot)["class_id"]
                if ch not in witnesses:
                    channels.append(ch)
                    if r1["kind"] == "disclination":
                        witnesses[ch] = [
                            dict(kind="disclination", angle=ang, at=c0, strict=True),
                            dict(kind="dislocation", b=b_vec, at=p_disl),
                        ]
                    else:
                        witnesses[ch] = [
                            dict(kind="dislocation", b=b_vec, at=p_disl),
                            dict(kind="disclination", angle=ang, at=c0, strict=True),
                        ]
            return dict(
                c1=id1,
                c2=id2,
                law="D×T",
                index_det=det_idx,
                channels=channels,
                infinite_tower=False,
                contains_vacuum=False,
                formula=f"{id1} × {id2} = " + " ⊕ ".join(channels),
                witnesses=witnesses,
            )

        # Disclination x Disclination
        ang1, ang2 = r1["angle"], r2["angle"]
        ang_sum = ang1 + ang2
        c1 = r1["centre"]
        H1 = rotation_about(c1, math.radians(ang1))
        w2_rec = next(c for c in self.cat.centres if c["wyckoff"] == r2["wyckoff"])

        # Law 3: Opposite-angle disclination pair (alpha1 + alpha2 == 0)
        if abs(ang_sum) < 1e-6:
            same_wyckoff = r1["wyckoff"] == r2["wyckoff"]
            found_shells: list[tuple[float, str, np.ndarray]] = []
            seen_ch = set()
            for q2 in w2_rec["members"]:
                for n1 in range(-4, 5):
                    for n2 in range(-4, 5):
                        c2 = q2 + self.cell.A @ np.array([n1, n2], float)
                        H2 = rotation_about(c2, math.radians(ang2))
                        H_tot = iso_compose(H1, H2)
                        b_res = -H_tot[1]
                        lb = _norm2(b_res)
                        ch = self.cat.identify(0.0, H_tot)["class_id"]
                        if ch not in seen_ch:
                            seen_ch.add(ch)
                            found_shells.append((round(lb, 5), ch, c2))
            found_shells.sort(key=lambda t: (t[0], t[1]))
            # Keep vacuum (if present) + first `max_burgers_shells` distinct non-zero lengths
            distinct_lengths = sorted({t[0] for t in found_shells if t[0] > 1e-5})[:max_burgers_shells]
            kept = [t for t in found_shells if t[0] < 1e-5 or t[0] in distinct_lengths]
            channels = [t[1] for t in kept]
            witnesses = {}
            for lb, ch, c2 in kept:
                if lb < 1e-5:
                    # Coincident c2 == c1 annihilates algebraically to I; on the lattice a dipole isn't placed on one point
                    witnesses[ch] = []
                else:
                    # Pick a well-separated representative in the same class if c2 is very close to c1
                    witnesses[ch] = [
                        dict(kind="disclination", angle=ang1, at=c1, strict=True),
                        dict(kind="disclination", angle=ang2, at=c2, strict=True),
                    ]
            min_channels = [t[1] for t in kept if abs(t[0] - (0.0 if same_wyckoff else distinct_lengths[0])) < 1e-4]
            return dict(
                c1=id1,
                c2=id2,
                law="D×D->T",
                channels=channels,
                minimal_channels=min_channels,
                infinite_tower=True,
                contains_vacuum=same_wyckoff,
                formula=f"{id1} × {id2} = " + " ⊕ ".join(channels) + " ⊕ ⋯ (coset tower)",
                witnesses=witnesses,
            )

        # Whole-turn case (e.g. -180 + -180 = -360) or cone-closing >= +360
        if abs(ang_sum) % 360.0 < 1e-6 or ang_sum >= 360.0 - 1e-6:
            return dict(
                c1=id1,
                c2=id2,
                law="D×D (boundary/turn)",
                channels=[f"Turn({ang_sum:+.0f}°)"],
                infinite_tower=(abs(ang_sum) % 360.0 < 1e-6),
                contains_vacuum=False,
                formula=f"{id1} × {id2} -> Frank angle {ang_sum:+.0f}° ("
                + ("closes planar cone >= +360°" if ang_sum > 0 else "full inserted turn -360° + Burgers coset")
                + ")",
                witnesses={},
            )

        # Law 4: Composite disclination (alpha1 + alpha2 != 0 mod 360, < +360)
        # By the Finite Coset Exhaustion Lemma, testing n1, n2 in range(det_12) across q2 in w2_rec["members"]
        # provably exhausts 100% of output proper conjugacy classes!
        R12 = rot(math.radians(ang_sum))
        det_12 = int(round(abs(np.linalg.det(np.eye(2) - R12))))
        channels = []
        witnesses = {}
        # Prefer separations of ~2..5 cells for clean non-overlapping lattice witnesses
        shifts_ordered = sorted(
            [(n1, n2) for n1 in range(0, max(det_12, 1) + 4) for n2 in range(0, max(det_12, 1) + 4)],
            key=lambda nm: abs(_norm2(self.cell.A @ np.array(nm, float)) - 2.8 * self.scale),
        )
        for q2 in w2_rec["members"]:
            for n1, n2 in shifts_ordered:
                c2 = q2 + self.cell.A @ np.array([n1, n2], float)
                if _norm2(c2 - c1) < 1.2 * self.lmax:
                    continue
                H2 = rotation_about(c2, math.radians(ang2))
                H_tot = iso_compose(H1, H2)
                ch = self.cat.identify(ang_sum, H_tot)["class_id"]
                if ch not in witnesses:
                    channels.append(ch)
                    witnesses[ch] = [
                        dict(kind="disclination", angle=ang1, at=c1, strict=True),
                        dict(kind="disclination", angle=ang2, at=c2, strict=True),
                    ]
        channels.sort()
        return dict(
            c1=id1,
            c2=id2,
            law="D×D->D",
            index_det=det_12,
            channels=channels,
            infinite_tower=False,
            contains_vacuum=False,
            formula=f"{id1} × {id2} = " + " ⊕ ".join(channels),
            witnesses=witnesses,
        )

    def verify_fusion_on_lattice(self, id1: str, id2: str, max_burgers_shells: int = 1) -> dict:
        """Compute algebraic fusion of (id1, id2) AND constructively build lattice witnesses for every
        realizable channel, verifying that the graph-measured charge equals the predicted channel."""
        res = self.fuse(id1, id2, max_burgers_shells=max_burgers_shells)
        verified = []
        for ch, specs in res["witnesses"].items():
            if not specs:
                # Vacuum annihilation channel (coincident opposite cores = perfect patch)
                verified.append(dict(channel=ch, built=False, match=True, note="coincident annihilation -> perfect lattice"))
                continue
            # Try witness (or a lattice-period-shifted copy if cores sit closer than 2 bonds)
            built_ok = False
            last_err = None
            R_det = 1
            if specs[-1]["kind"] == "disclination":
                ang_sum = sum(s.get("angle", 0.0) for s in specs)
                if abs(ang_sum) > 1e-6 and abs(ang_sum) % 360.0 > 1e-6:
                    R_det = int(round(abs(np.linalg.det(np.eye(2) - rot(math.radians(ang_sum))))))
                else:
                    # Opposite-angle pair: shifting c2 changes b! Keep exact c2 and increase R if needed,
                    # or shift BOTH c1 and c2? No, c2 - c1 fixes b; if ||c2 - c1|| >= lmax, default cuts away from each other work!
                    R_det = 0
            candidates = [specs]
            if R_det > 0:
                for mult in (1, 2):
                    sp_copy = [dict(s) for s in specs]
                    sp_copy[-1]["at"] = np.asarray(sp_copy[-1]["at"], float) + (mult * R_det) * self.cell.a1
                    candidates.append(sp_copy)
            for cand in candidates:
                try:
                    d = self.combine(*cand, picture="raw")
                    meas = self.identify_charge(d)["class_id"]
                    verified.append(dict(channel=ch, measured=meas, built=True, method="combine", match=(meas == ch), defect=d, specs=cand))
                    built_ok = True
                    break
                except ConfigError as e:
                    last_err = str(e)
            # If two opposite-angle cores sit 1 bond apart and a wide wedge (e.g. +-180 deg) pinches a shared bond
            # in sequential cutting, fall back to the simultaneous polygon-gluing construction of C07 (disclination_pair)
            if not built_ok and len(specs) == 2 and specs[0]["kind"] == "disclination" and specs[1]["kind"] == "disclination":
                a1, a2 = float(specs[0]["angle"]), float(specs[1]["angle"])
                if abs(a1 + a2) < 1e-6 and 0.0 < abs(a1) <= 180.0 + 1e-6:
                    try:
                        if a1 > 0:
                            dp = disclination_pair(self.cell, a1, specs[0]["at"], specs[1]["at"], picture="raw")
                        else:
                            dp = disclination_pair(self.cell, -a1, specs[1]["at"], specs[0]["at"], picture="raw")
                        meas = self.identify_charge(dp)["class_id"]
                        verified.append(dict(channel=ch, measured=meas, built=True, method="disclination_pair", match=(meas == ch), defect=dp, specs=specs))
                        built_ok = True
                    except Exception as e:
                        last_err = str(e)
            if not built_ok:
                verified.append(dict(channel=ch, measured=None, built=False, match=False, error=last_err))
        res["verified"] = verified
        res["all_matched"] = all(v["match"] for v in verified)
        return res

    def fusion_summary_tables(self, positive_angles_only: bool = False) -> dict:
        """Compute structured fusion tables for this wallpaper group:
        - disclination_x_dislocation: how elementary dislocations permute/absorb into disclination classes
        - disclination_x_disclination: pairwise disclination fusion channels"""
        t_ids = [r["id"] for r in self.cat.translations]
        d_recs = [r for r in self.cat.disclinations if (not positive_angles_only or r["angle"] > 0)]
        d_ids = [r["id"] for r in d_recs]

        dt_rows = []
        for did in d_ids:
            row = {"disclination": did}
            for tid in t_ids:
                f = self.fuse(did, tid)
                row[tid] = " ⊕ ".join(f["channels"])
            dt_rows.append(row)

        dd_matrix: dict[tuple[str, str], dict] = {}
        for d1 in d_ids:
            for d2 in d_ids:
                dd_matrix[(d1, d2)] = self.fuse(d1, d2, max_burgers_shells=1)

        return dict(
            group=self.group,
            dislocations=t_ids,
            disclinations=d_ids,
            disclination_x_dislocation=dt_rows,
            disclination_x_disclination=dd_matrix,
        )

    def plot_fusion_table(self, max_abs_angle: float = 180.0, figsize: tuple[float, float] | None = None):
        """Plot a colour-coded visual fusion table of elementary point defects (T and D classes)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        t_ids = [r["id"] for r in self.cat.translations[:2]]
        d_ids = [r["id"] for r in self.cat.disclinations if abs(r["angle"]) <= max_abs_angle + 1e-6]
        labels = t_ids + d_ids
        n = len(labels)
        if n == 0:
            return None
        figsize = figsize or (max(7.5, 1.15 * n + 2.0), max(6.5, 0.85 * n + 1.8))
        fig, ax = plt.subplots(figsize=figsize)

        bg_colors = {
            "vacuum": "#c8eed3",      # green: can annihilate to vacuum I
            "dipole": "#fde9b8",      # gold: opposite angles, non-zero residual Burgers coset (no I)
            "disclination": "#d8e8fa",# light blue: composite or Wyckoff-shifted disclination
            "dislocation": "#ececec", # grey: T x T dislocation sum
            "boundary": "#f8d4d4",    # coral: >= 360 cone-closing or -360 whole turn
        }

        for i, r_id in enumerate(labels):
            for j, c_id in enumerate(labels):
                f = self.fuse(r_id, c_id, max_burgers_shells=1)
                law = f["law"]
                if f["contains_vacuum"]:
                    kind_key = "vacuum"
                elif law == "D×D->T":
                    kind_key = "dipole"
                elif law in ("D×T", "D×D->D"):
                    kind_key = "disclination"
                elif law == "T×T":
                    kind_key = "dislocation"
                else:
                    kind_key = "boundary"

                rect = plt.Rectangle((j, n - 1 - i), 1, 1, facecolor=bg_colors[kind_key], edgecolor="#666666", lw=0.8)
                ax.add_patch(rect)
                chs = f["channels"]
                txt = " ⊕ ".join(chs[:3]) + (" ⊕ …" if len(chs) > 3 or f["infinite_tower"] else "")
                txt = txt.replace(", ", ",")
                ax.text(j + 0.5, n - 0.5 - i, txt, ha="center", va="center", fontsize=7.0, wrap=True)

        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([k + 0.5 for k in range(n)])
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks([n - 0.5 - k for k in range(n)])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(
            f"Proper-Group ($G_+$) Defect Fusion Rules for {self.group} ({self.name})",
            fontsize=10.5,
            pad=12,
        )
        legend_handles = [
            Patch(facecolor=bg_colors["vacuum"], edgecolor="#666", label="Contains vacuum I (annihilation allowed)"),
            Patch(facecolor=bg_colors["dipole"], edgecolor="#666", label="Forbidden annihilation (residual Burgers coset)"),
            Patch(facecolor=bg_colors["disclination"], edgecolor="#666", label="Disclination channel(s)"),
            Patch(facecolor=bg_colors["dislocation"], edgecolor="#666", label="Dislocation × Dislocation"),
            Patch(facecolor=bg_colors["boundary"], edgecolor="#666", label="±360° whole-turn / cone-closing"),
        ]
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=7.5, frameon=True)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------------------------------
    # Stage 4: Non-orientable Möbius-strip & Klein-bottle braiding and fusion (C11)
    # ------------------------------------------------------------------------------------------
    def _require_achiral(self) -> None:
        if self.cat.chiral or not self.cat.lines:
            raise ConfigError(
                f"wallpaper group {self.group!r} is chiral/proper (G = G+); non-orientable Möbius strips and Klein bottles require an improper symmetry"
            )

    def _get_line_rec(self, line: str | int | dict = "M1") -> dict:
        self._require_achiral()
        if isinstance(line, dict):
            return line
        if isinstance(line, (int, np.integer)):
            idx = int(line) - 1
            if 0 <= idx < len(self.cat.lines):
                return self.cat.lines[idx]
            raise ConfigError(f"Möbius line index {line} out of range 1..{len(self.cat.lines)}")
        if isinstance(line, str):
            for rec in self.cat.lines:
                if rec["id"] == line:
                    return rec
        raise ConfigError(
            f"unknown Möbius seam class {line!r} for group {self.group}; available: {[r['id'] for r in self.cat.lines]}"
        )

    def _resolve_point_class(self, defect_id: str) -> tuple[float, tuple[np.ndarray, np.ndarray], dict]:
        if defect_id in self.cat.by_id:
            r = self.cat.by_id[defect_id]
            if r["kind"] in ("vacuum", "dislocation", "disclination"):
                return float(r["angle"]), r["H"], r
            raise ConfigError(f"{defect_id!r} ({r['kind']}) is not a point-defect class")
        # Support composite disclinations D(w, +ang°) outside [-180, 180]
        if defect_id.startswith("D(") and defect_id.endswith("°)") and "," in defect_id:
            inside = defect_id[2:-2]
            wyk, ang_s = [s.strip() for s in inside.split(",", 1)]
            crec = next((c for c in self.cat.centres if c["wyckoff"] == wyk), None)
            if crec is not None:
                ang = float(ang_s)
                if abs(ang % (360.0 / crec["order"])) < 1e-4 or abs((ang % (360.0 / crec["order"])) - (360.0 / crec["order"])) < 1e-4:
                    H = rotation_about(crec["point"], math.radians(ang))
                    ident = self.cat.identify(ang, H)
                    return ang, H, dict(id=ident["class_id"], kind="disclination", wyckoff=wyk, centre=crec["point"], angle=ang, H=H)
        raise ConfigError(f"unknown point-defect class {defect_id!r}")

    def identify_improper(self, h: tuple[np.ndarray, np.ndarray], tol: float = 1e-3) -> dict:
        """Classify any improper lattice symmetry `h = (M_h, t_h)` (`det M_h = -1`) into the group's
        mirror/glide seam classes `M1, M2, ...`, reporting its invariant axis, glide modulo the axis
        lattice period, and intrinsic mirror vs glide character."""
        self._require_achiral()
        M_h = np.asarray(h[0], float)
        t_h = np.asarray(h[1], float)
        if M_h.shape != (2, 2) or abs(float(np.linalg.det(M_h)) + 1.0) > 1e-3 or not np.allclose(M_h.T @ M_h, np.eye(2), atol=1e-3):
            raise ConfigError("isometry h is not an orientation-reversing orthogonal map (det M != -1)")
        if not self.lattice.is_symmetry((M_h, t_h), probe=2.2):
            raise ConfigError(f"improper isometry (M, t={np.round(t_h, 4).tolist()}) is not a symmetry of lattice {self.group}")

        w, V = np.linalg.eigh(0.5 * (M_h + M_h.T))
        u_ax = _canon_axis(V[:, int(np.argmax(w))])
        n_ax = np.array([-u_ax[1], u_ax[0]])
        x0 = 0.5 * float(t_h @ n_ax) * n_ax
        shift_u = float(t_h @ u_ax)
        per_u, _ = _period_and_spacing_along(self.cell.A, u_ax)
        gl = abs(shift_u) % per_u
        if gl < tol or abs(per_u - gl) < tol:
            gl = 0.0
        intrinsic_kind = "mirror" if gl == 0.0 else "glide"

        for rec in self.cat.lines:
            for M_u, t_u in self.cat.ops:
                u_im = _canon_axis(M_u @ rec["direction"])
                if abs(_cross2v(u_im, u_ax)) > tol:
                    continue
                if abs(float(rec["glide"]) - gl) > tol:
                    continue
                p_im = M_u @ rec["point"] + t_u
                dn = float((p_im - x0) @ n_ax) / float(rec["spacing"])
                if abs(dn - round(dn)) < tol:
                    return dict(
                        seam_id=rec["id"],
                        kind=rec["kind"],
                        intrinsic_kind=intrinsic_kind,
                        axis=u_ax,
                        point=x0,
                        shift_along_axis=shift_u,
                        axis_period=per_u,
                        glide_mod_period=gl,
                        h=(M_h, t_h),
                    )
        raise ConfigError(f"could not match improper symmetry (axis={np.round(u_ax, 3).tolist()}, glide={gl:.4f}) to any line class of {self.group}")

    def measure_mobius_seam(self, d: Defect) -> dict:
        """Find a shortest closed walk `beta` on `d` crossing the orientation-reversing seam once
        (`winding_u = +1`, `winding_perp = 0`, avoiding irregular defect core sites when possible)
        and read its exact deck holonomy `h = iso_inverse(loop_charge(d, beta)[0])` (`det = -1`)."""
        self._require_achiral()
        if is_orientable_surface(d):
            raise ConfigError("surface is orientable; no orientation-reversing Möbius seam exists")
        if "reference" not in d.info:
            if "h" in d.info:
                ident = self.identify_improper(d.info["h"])
                ident["loop"] = []
                return ident
            raise ConfigError("Defect lacks local reference frames; build with FusionLab.mobius_strip() or klein_bottle()")

        K, _ = irregular_sites(d, self.cell, self.lattice)
        cone_set = set(d.info.get("cone_sites", ()))
        avoid = K | cone_set
        X = np.asarray(d.info["reference"], float)
        u_ref = np.asarray(d.info["axis"], float) if "axis" in d.info else None
        n_ref = np.asarray(d.info["normal"], float) if "normal" in d.info else None
        x0 = np.asarray(d.info.get("point", np.zeros(2)), float)
        is_klein = bool(d.info.get("klein"))
        ell_n = float(d.info.get("ell_n", 0.0))
        h_nom = d.info.get("h")

        nbr_list: dict[int, list[int]] = {}
        for u, v in d.edge:
            nbr_list.setdefault(u, []).append(v)

        def edge_windings(u: int, v: int) -> tuple[int, int]:
            e = d.edge[(u, v)]
            if "ku" in e and "mp" in e:
                return int(e["ku"]), int(e["mp"])
            if "j" in e and not is_klein:
                return int(e["j"]), 0
            g = e.get("g")
            if g is None:
                return 0, 0
            sgn = _det_sign_iso(g)
            if sgn > 0:
                if is_klein and n_ref is not None and ell_n > 1e-6:
                    mp = int(round(-float(g[1] @ n_ref) / ell_n))
                    return 0, mp
                return 0, 0
            # Orientation-reversing transition: determine forward (+1) vs backward (-1)
            if u_ref is not None:
                ku = 1 if float(g[1] @ u_ref) < 0.0 else -1
            else:
                w_e, V_e = np.linalg.eigh(0.5 * (g[0] + g[0].T))
                u_e = _canon_axis(V_e[:, int(np.argmax(w_e))])
                ku = 1 if float(g[1] @ u_e) < 0.0 else -1
            mp = 0
            if is_klein and n_ref is not None and ell_n > 1e-6 and h_nom is not None:
                rem = iso_compose(g, h_nom if ku == 1 else iso_inverse(h_nom))
                mp = int(round(-float(rem[1] @ n_ref) / ell_n))
            return ku, mp

        # Candidate start vertices near the invariant axis (where h maps x0 -> x0 + ell_u u_ax directly)
        candidates = [i for i in range(len(d.sites)) if i not in avoid]
        if not candidates:
            candidates = [i for i in range(len(d.sites)) if i not in cone_set]
        if n_ref is not None:
            candidates.sort(key=lambda i: abs(float((X[i] - x0) @ n_ref)))

        def find_loop(blocked: set[int]) -> list[int] | None:
            for start in candidates[:12]:
                if start in blocked:
                    continue
                parent: dict[tuple[int, int, int], tuple[tuple[int, int, int] | None, int]] = {(start, 0, 0): (None, start)}
                dq = deque([(start, 0, 0)])
                while dq:
                    u, W, M = dq.popleft()
                    if abs(W) > 2 or abs(M) > 1:
                        continue
                    for v in nbr_list.get(u, ()):
                        if v in blocked:
                            continue
                        ku, mp = edge_windings(u, v)
                        nW = W + ku
                        nM = ((-1 if (ku % 2) else 1) * M) + mp
                        state = (v, nW, nM)
                        if v == start and nW == 1 and nM == 0:
                            path = [start]
                            cur = (u, W, M)
                            while cur is not None:
                                path.append(cur[0])
                                cur = parent[cur][0]
                            path.reverse()
                            return path[:-1]
                        if state not in parent:
                            parent[state] = ((u, W, M), v)
                            dq.append(state)
            return None

        beta = find_loop(avoid)
        if beta is None:
            beta = find_loop(cone_set)
        if beta is None:
            raise ConfigError("no closed orientation-reversing seam walk found on surface")

        H_loop, S_loop = loop_charge(d, beta)
        h_meas = iso_inverse(H_loop)
        ident = self.identify_improper(h_meas)
        ident["loop"] = beta
        ident["H_loop"] = H_loop
        ident["S_loop"] = S_loop
        return ident

    def braid_around_mobius(self, defect_id: str, line_id: str = "M1", length: int | None = None) -> dict:
        """Compute exact universal-cover monodromy when point defect `defect_id` is pushed once around
        the orientation-reversing loop of Möbius strip `line_id` (of `length` axis periods):
          - point-defect holonomy maps by `mu_h(alpha, H) = (+alpha, h o H^-1 o h^-1)` (scalar Frank
            angle `+alpha` is PRESERVED while Wyckoff centres reflect `c -> h(c)` and Burgers vectors
            map `b -> -M_h b`),
          - Möbius seam holonomy experiences topological back-reaction `h -> h' = h o H`."""
        line_rec = self._get_line_rec(line_id)
        n_per = 8 if length is None else int(length)
        dd = _line_deck_data(self.cell, line_rec, n_per)
        h = dd["h"]
        M_h = h[0]
        alpha, H, r = self._resolve_point_class(defect_id)
        alpha_after, H_after = mirror_holonomy(alpha, H, h)
        ident_after = self.cat.identify(alpha_after, H_after)
        h_prime = iso_compose(h, H)
        seam_after = self.identify_improper(h_prime)

        b_before = -H[1] if r["kind"] == "dislocation" else None
        b_after = (-M_h @ b_before) if b_before is not None else None
        return dict(
            defect_id=r["id"],
            transported_id=ident_after["class_id"],
            defect_changed=(ident_after["class_id"] != r["id"]),
            angle_before=alpha,
            angle_after=alpha_after,
            b_before=b_before,
            b_after=b_after,
            H_before=H,
            H_after=H_after,
            seam_before=line_rec["id"],
            seam_kind_before=line_rec["kind"],
            new_seam_id=seam_after["seam_id"],
            new_seam_kind=seam_after["kind"],
            seam_changed=(seam_after["seam_id"] != line_rec["id"]),
            length=n_per,
            length_parity="even" if (n_per % 2 == 0) else "odd",
            h=h,
            h_prime=h_prime,
        )

    def fuse_across_mobius(
        self,
        id1: str,
        id2: str,
        line_id: str = "M1",
        length: int | None = None,
        max_burgers_shells: int = 2,
    ) -> dict:
        """Compute cross-seam fusion `C1 x_{M_k} C2 = C1 x mu_h(C2)` when `id2` is transported once
        across Möbius seam `line_id` before fusing with `id1`, contrasting orientable vs Möbius vacuum
        annihilation."""
        self._require_achiral()
        braid2 = self.braid_around_mobius(id2, line_id=line_id, length=length)
        id2_tr = braid2["transported_id"]
        f_orient = self.fuse(id1, id2, max_burgers_shells=max_burgers_shells)
        f_mob = self.fuse(id1, id2_tr, max_burgers_shells=max_burgers_shells)
        vac_ori = bool(f_orient["contains_vacuum"])
        vac_mob = bool(f_mob["contains_vacuum"])
        return dict(
            c1=id1,
            c2=id2,
            line_id=braid2["seam_before"],
            transported_c2=id2_tr,
            seam_after_c2_transport=braid2["new_seam_id"],
            law=f_mob["law"],
            channels=f_mob["channels"],
            orientable_channels=f_orient["channels"],
            infinite_tower=f_mob["infinite_tower"],
            contains_vacuum_orientable=vac_ori,
            contains_vacuum_mobius=vac_mob,
            parity_flipped_vacuum=(vac_ori != vac_mob),
            formula=(
                f"{id1} ×_{{{braid2['seam_before']}}} {id2} = {id1} × {id2_tr} = "
                + " ⊕ ".join(f_mob["channels"])
                + (" ⊕ ⋯ (coset tower)" if f_mob["infinite_tower"] else "")
            ),
            witnesses=f_mob["witnesses"],
        )

    def mobius_strip(
        self,
        line: str = "M1",
        length: int = 8,
        width: float | None = None,
        picture: str = "raw",
        name: str | None = None,
    ) -> Defect:
        """Build a clean framed Möbius-strip crystal for mirror/glide class `line`."""
        line_rec = self._get_line_rec(line)
        d = build_mobius_surface(self.cell, self.lattice, line_rec, length=length, width=width, picture=picture)
        d.info["name"] = name or f"Möbius {line_rec['id']} ({line_rec['kind']}, L={int(length)})"
        return d

    def mobius_combine(
        self,
        *choices: Any,
        line: str = "M1",
        length: int = 8,
        width: float | None = None,
        picture: str = "raw",
        name: str | None = None,
    ) -> Defect:
        """Place any point defect or mixed list of dislocations and disclinations onto a Möbius strip
        of class `line` (automatically preserving `length % 2` parity if a wider ribbon is needed to
        accommodate multiple cores or wide disclination wedges)."""
        line_rec = self._get_line_rec(line)
        if len(choices) == 1 and isinstance(choices[0], list):
            choices = tuple(choices[0])
        if not choices:
            return self.mobius_strip(line=line_rec["id"], length=length, width=width, picture=picture, name=name)

        raw_specs = [self._parse_spec(c, i) for i, c in enumerate(choices)]
        expected_frank = sum(float(sp.get("angle", 0.0)) for sp in raw_specs if sp["kind"] == "disclination")
        if expected_frank >= 360.0 - 1e-6:
            raise ConfigError(f"total Frank angle {expected_frank:+.1f} deg >= 360 deg closes the surface boundary")

        n_def = len(raw_specs)
        dd0 = _line_deck_data(self.cell, line_rec, max(4, int(length)))
        u_ax, n_ax, per_u = dd0["u_ax"], dd0["n_ax"], dd0["per_u"]
        max_b = max((_norm2(sp["b"]) for sp in raw_specs if sp["kind"] == "dislocation"), default=0.0)
        w_base = float(max(3.8 * self.lmax + max_b, 2.6 * self.scale) if width is None else width)

        # Estimate required axial length while strictly preserving requested length parity (length % 2)
        slice_needs = []
        for sp in raw_specs:
            if sp["kind"] == "disclination":
                ang_abs = abs(float(sp["angle"]))
                if ang_abs >= 150.0:
                    slice_needs.append(2.3 * max(4.5 * self.lmax, 3.0 * self.scale))
                else:
                    tan_h = math.tan(0.5 * math.radians(ang_abs))
                    slice_needs.append(2.35 * ((w_base + 2.2 * self.lmax + self.scale) * tan_h + 2.5 * self.lmax))
            else:
                slice_needs.append(max(5.5 * self.lmax + 2.0 * _norm2(sp["b"]), 3.5 * self.scale))
        need_ell = sum(slice_needs) + 3.0 * self.lmax
        L_use = max(int(length), 4)
        while (dd0["glide"] + (L_use - 1) * per_u if dd0["glide"] > 1e-6 else L_use * per_u) < need_ell:
            L_use += 2  # preserve even/odd parity of length!

        dd = _line_deck_data(self.cell, line_rec, L_use)
        x0, ell_u = dd["x0"], dd["ell_u"]

        # Position each defect cleanly inside its axial slice of F0 and configure lobes for |angle| >= 150 deg
        placed_specs: list[dict] = []
        lobes: list[tuple[float, float, float, float]] = []
        w_band = w_base
        total_weight = sum(slice_needs)
        cum = 1.5 * self.lmax
        for idx, (sp_in, w_slice) in enumerate(zip(raw_specs, slice_needs)):
            sp = dict(sp_in)
            u_slot = cum + 0.5 * w_slice
            cum += w_slice
            user_gave_at = False
            orig_choice = choices[idx]
            if isinstance(orig_choice, dict) and "at" in orig_choice:
                user_gave_at = True
            elif isinstance(orig_choice, tuple) and len(orig_choice) == 2 and isinstance(orig_choice[1], dict) and "at" in orig_choice[1]:
                user_gave_at = True

            if sp["kind"] == "dislocation":
                if not user_gave_at:
                    d_off = (0.35 if idx % 2 == 0 else -0.35) * self.scale
                    sp["at"] = x0 + u_slot * u_ax + d_off * n_ax
                d_at = float((sp["at"] - x0) @ n_ax)
                w_band = max(w_band, abs(d_at) + _norm2(sp["b"]) + 3.2 * self.lmax)
            else:
                ang = float(sp["angle"])
                wyk = None
                if isinstance(orig_choice, str) and orig_choice in self.cat.by_id:
                    wyk = self.cat.by_id[orig_choice].get("wyckoff")
                elif isinstance(orig_choice, tuple) and isinstance(orig_choice[0], str) and orig_choice[0] in self.cat.by_id:
                    wyk = self.cat.by_id[orig_choice[0]].get("wyckoff")
                if abs(ang) >= 150.0 and not user_gave_at:
                    sgn_lb = 1.0 if idx % 2 == 0 else -1.0
                    r_lb = min(0.42 * ell_u, max(4.6 * self.lmax, 3.0 * self.scale))
                    target = x0 + u_slot * u_ax + sgn_lb * (w_base + 1.8 * self.lmax + 0.8 * self.scale) * n_ax
                    sp["at"] = self.nearest_centre(ang, target, wyckoff=wyk)
                    u_p = float((sp["at"] - x0) @ u_ax)
                    d_p = float((sp["at"] - x0) @ n_ax)
                    lobes.append((u_p, d_p, r_lb, sgn_lb))
                    sp["_in_lobe"] = sgn_lb
                else:
                    if not user_gave_at:
                        target = x0 + u_slot * u_ax + (0.15 if idx % 2 == 0 else -0.15) * self.scale * n_ax
                        sp["at"] = self.nearest_centre(ang, target, wyckoff=wyk)
                    d_p = float((sp["at"] - x0) @ n_ax)
                    w_band = max(w_band, abs(d_p) + 3.8 * self.lmax)
            placed_specs.append(sp)

        # Re-check whether grown w_band requires one more +2 length bump for non-lobe disclinations
        re_need = 3.0 * self.lmax
        for sp in placed_specs:
            if sp["kind"] == "disclination" and "_in_lobe" not in sp:
                d_p = float((sp["at"] - x0) @ n_ax)
                c_perp = w_band - abs(d_p) + 1.5 * self.lmax
                re_need += 2.35 * (c_perp * math.tan(0.5 * abs(math.radians(float(sp["angle"])))) + 2.5 * self.lmax)
            elif sp["kind"] == "disclination":
                re_need += 2.3 * max(4.6 * self.lmax, 3.0 * self.scale)
            else:
                re_need += max(5.5 * self.lmax + 2.0 * _norm2(sp["b"]), 3.5 * self.scale)
        if re_need > ell_u and not lobes:
            while (dd0["glide"] + (L_use - 1) * per_u if dd0["glide"] > 1e-6 else L_use * per_u) < re_need:
                L_use += 2
            dd = _line_deck_data(self.cell, line_rec, L_use)
            ell_u = dd["ell_u"]

        surf = build_mobius_surface(self.cell, self.lattice, line_rec, length=L_use, width=w_band, lobes=lobes or None, picture="raw")
        B = _SurfaceBuilder(self.cell, self.lattice, surf)

        for idx, sp in enumerate(placed_specs, 1):
            disloc = sp["kind"] == "dislocation"
            if sp.get("cut") is not None:
                cand_cuts = [np.asarray(sp["cut"], float)]
            elif disloc:
                b = sp["b"]
                cand_cuts = []
                for sgn_n in (1.0, -1.0):
                    base_n = sgn_n * n_ax
                    for tilt in (0.0, 22.0, -22.0, 38.0, -38.0, 55.0, -55.0):
                        c_try = rot(math.radians(tilt)) @ base_n
                        if _cross2v(c_try, b) < -1e-5 * _norm2(b):
                            cand_cuts.append(c_try)
            elif "_in_lobe" in sp:
                sgn_lb = float(sp["_in_lobe"])
                ang = float(sp["angle"])
                # Orient half-plane towards outer lobe cap (sgn_lb * n_ax)
                th_out = math.atan2(float(sgn_lb * n_ax[1]), float(sgn_lb * n_ax[0]))
                th0 = th_out + 0.5 * math.radians(ang)
                cand_cuts = [
                    np.array([math.cos(th0 + math.radians(da)), math.sin(th0 + math.radians(da))])
                    for da in (0.0, 8.0, -8.0, 16.0, -16.0)
                ]
            else:
                ang = float(sp["angle"])
                d_p = float((sp["at"] - x0) @ n_ax)
                order_sgn = (1.0, -1.0) if d_p >= 0.0 else (-1.0, 1.0)
                cand_cuts = []
                for sgn_n in order_sgn:
                    th_out = math.atan2(float(sgn_n * n_ax[1]), float(sgn_n * n_ax[0]))
                    th0 = th_out + 0.5 * math.radians(ang)
                    for da in (0.0, 10.0, -10.0, 20.0, -20.0):
                        cand_cuts.append(np.array([math.cos(th0 + math.radians(da)), math.sin(th0 + math.radians(da))]))

            err = None
            for cut in cand_cuts:
                snap = B.snapshot()
                try:
                    if disloc:
                        B.add_dislocation(sp["b"], sp["at"], cut)
                    else:
                        B.add_disclination(float(sp["angle"]), sp["at"], cut)
                    err = None
                    break
                except ConfigError as e:
                    B.restore(snap)
                    err = e
            if err is not None:
                raise ConfigError(f"Möbius defect {idx} ({sp['kind']}): {err}") from None

        out = B.finish("mobius")
        if self.planar:
            bad_f = check_frames(out)
            se = surface_euler_characteristic(out)
            tf = total_frank(out)
            if bad_f != 0 or se["euler_surface"] != 0 or se["orientable"] or abs(tf - expected_frank) > 1e-2:
                raise ConfigError(
                    f"Möbius combinatorial check failed (frames_bad={bad_f}, euler={se['euler_surface']}, orientable={se['orientable']}, frank={tf:.2f} vs {expected_frank:.2f})"
                )
        if picture == "relaxed":
            relax_picture(out)
        else:
            mark_long_bonds(out)
        desc = " + ".join(
            (f"b={_r(r['b'], 2)}" if r["kind"] == "dislocation" else f"{r['angle']:+.0f}°")
            for r in out.info["defects"]
        )
        out.info["name"] = name or f"Möbius {line_rec['id']} + [{desc}]"
        return out

    def mobius_transport(
        self,
        defect_id: str,
        line: str = "M1",
        length: int = 8,
        width: float | None = None,
        picture: str = "raw",
    ) -> dict:
        """Constructively realize universal-cover monodromy of `defect_id` pushed once around Möbius
        strip `line` on actual finite lattice graphs:
          - For dislocations (`b`): builds `before` (`+b` at `p0`, cut to `+n`), `after` (`mu_h(b) = -M_h b`
            at `q0`, cut to `-n`), AND `cross_seam_dipole` via `add_cross_seam_dipole(-b, p1, h(p2))`
            creating `-b` and `mu_h(b)` directly across the seam.
          - For disclinations (`D(w, alpha)`): builds `before` (`(alpha, H)` at `p0`, cut to `+n`) and
            `after` (`mu_h(alpha, H)` at `q0 = h^-1(p0 + ell_u u_ax)`, cut to `-n`).
          - Measures BOTH the local core holonomy in `F0` AND a fixed reference loop `beta_ref` on
            `before` vs `after`, confirming `h -> h' = h o H` directly on the lattice graph."""
        line_rec = self._get_line_rec(line)
        braid = self.braid_around_mobius(defect_id, line_id=line_rec["id"], length=length)
        alpha, H_before, r = self._resolve_point_class(defect_id)

        dd0 = _line_deck_data(self.cell, line_rec, max(8, int(length)))
        u_ax, n_ax, per_u, per_n = dd0["u_ax"], dd0["n_ax"], dd0["per_u"], dd0["per_n"]

        if r["kind"] == "dislocation":
            b = -H_before[1]
            b_after = braid["b_after"]
            L_use = max(int(length), 12, int(math.ceil((10.0 * self.lmax + 6.5 * per_u + 3.0 * _norm2(b)) / per_u)))
            if (L_use % 2) != (int(length) % 2):
                L_use += 1
            dd = _line_deck_data(self.cell, line_rec, L_use)
            x0, ell_u, h, hinv = dd["x0"], dd["ell_u"], dd["h"], dd["hinv"]
            w_use = float(max(4.6 * self.lmax + _norm2(b), 3.2 * self.scale, 2.5 * per_n) if width is None else width)
            tau_u = (L_use * per_u if dd["glide"] < 1e-6 else (L_use - 1) * per_u) * u_ax
            p0 = x0 + 0.48 * ell_u * u_ax + 0.15 * self.scale * n_ax
            q0 = iso_apply(h, p0 - tau_u)

            def _pick_disl_cut(b_vec: np.ndarray, out_dir: np.ndarray) -> np.ndarray:
                for tilt in (0.0, 22.0, -22.0, 38.0, -38.0, 55.0, -55.0):
                    c_try = rot(math.radians(tilt)) @ out_dir
                    if _cross2v(c_try, b_vec) < -1e-5 * _norm2(b_vec):
                        return c_try
                return np.array([-b_vec[1], b_vec[0]])

            surf0 = build_mobius_surface(self.cell, self.lattice, line_rec, length=L_use, width=w_use, picture="raw")
            B_bef = _SurfaceBuilder(self.cell, self.lattice, surf0)
            B_bef.add_dislocation(b, p0, _pick_disl_cut(b, n_ax))
            d_before = B_bef.finish("mobius")

            B_aft = _SurfaceBuilder(self.cell, self.lattice, surf0)
            B_aft.add_dislocation(b_after, q0, _pick_disl_cut(b_after, -n_ax))
            d_after = B_aft.finish("mobius")

            # Also build single-crystal cross-seam transport dipole (-b at p1, mu_h(b) = -M_h b at p2)
            B_dip = _SurfaceBuilder(self.cell, self.lattice, surf0)
            d_off = -1.25 * per_n
            p1 = x0 + (ell_u - 2.4 * per_u) * u_ax + d_off * n_ax
            p2 = x0 + (2.4 * per_u) * u_ax + d_off * n_ax
            if abs(_cross2v(iso_apply(h, p2) - p1, -b)) < 1e-4 * _norm2(b):
                p1 = p1 - 0.55 * per_n * n_ax
            B_dip.add_cross_seam_dipole(-b, p1, iso_apply(h, p2))
            d_cross = B_dip.finish("mobius")
        elif r["kind"] == "disclination":
            d_cross = None
            ang = float(r["angle"])
            # Use canonical catalogue centre r["centre"] and translate purely along u_ax so (p0 - r["centre"]) || u_ax
            c_base = np.asarray(r["centre"], float)
            if abs(ang) >= 150.0:
                L_use = max(int(length), int(math.ceil((14.0 * self.lmax + 6.0 * self.scale) / per_u)))
                if (L_use % 2) != (int(length) % 2):
                    L_use += 1
                dd = _line_deck_data(self.cell, line_rec, L_use)
                x0, ell_u, h, hinv = dd["x0"], dd["ell_u"], dd["h"], dd["hinv"]
                w_use = float(max(3.6 * self.lmax, 2.5 * self.scale) if width is None else width)
                r_lb = min(0.38 * ell_u, max(4.6 * self.lmax, 3.0 * self.scale))
                p0 = self.nearest_centre(ang, x0 + 0.46 * ell_u * u_ax + (w_use + 1.8 * self.lmax + 0.8 * self.scale) * n_ax, wyckoff=r["wyckoff"])
                tau_u = (L_use * per_u if dd["glide"] < 1e-6 else (L_use - 1) * per_u) * u_ax
                q0 = iso_apply(h, p0 - tau_u)
                u_p, d_p = float((p0 - x0) @ u_ax), float((p0 - x0) @ n_ax)
                u_q, d_q = float((q0 - x0) @ u_ax), float((q0 - x0) @ n_ax)
                surf_bef = build_mobius_surface(self.cell, self.lattice, line_rec, length=L_use, width=w_use, lobes=[(u_p, d_p, r_lb, +1.0)], picture="raw")
                surf_aft = build_mobius_surface(self.cell, self.lattice, line_rec, length=L_use, width=w_use, lobes=[(u_q, d_q, r_lb, -1.0)], picture="raw")
                B_bef = _SurfaceBuilder(self.cell, self.lattice, surf_bef)
                th_up = math.atan2(float(n_ax[1]), float(n_ax[0])) + 0.5 * math.radians(ang)
                B_bef.add_disclination(ang, p0, np.array([math.cos(th_up), math.sin(th_up)]))
                d_before = B_bef.finish("mobius")

                B_aft = _SurfaceBuilder(self.cell, self.lattice, surf_aft)
                th_dn = math.atan2(float(-n_ax[1]), float(-n_ax[0])) + 0.5 * math.radians(ang)
                B_aft.add_disclination(ang, q0, np.array([math.cos(th_dn), math.sin(th_dn)]))
                d_after = B_aft.finish("mobius")
            else:
                w_use = float(max(4.0 * self.lmax, 2.8 * self.scale) if width is None else width)
                tan_h = math.tan(0.5 * abs(math.radians(ang)))
                need_ell = 2.4 * ((w_use + 2.2 * self.lmax + self.scale) * tan_h + 2.5 * self.lmax)
                L_use = max(int(length), int(math.ceil(need_ell / per_u)))
                if (L_use % 2) != (int(length) % 2):
                    L_use += 1
                dd = _line_deck_data(self.cell, line_rec, L_use)
                x0, ell_u, h, hinv = dd["x0"], dd["ell_u"], dd["h"], dd["hinv"]
                k_step = int(round((0.46 * ell_u - float((c_base - x0) @ u_ax)) / per_u))
                p0 = c_base + (k_step * per_u) * u_ax
                tau_u = (L_use * per_u if dd["glide"] < 1e-6 else (L_use - 1) * per_u) * u_ax
                q0 = iso_apply(h, p0 - tau_u)
                w_use = max(w_use, abs(float((p0 - x0) @ n_ax)) + 3.8 * self.lmax, abs(float((q0 - x0) @ n_ax)) + 3.8 * self.lmax)
                need2 = 2.4 * ((w_use + 1.5 * self.lmax) * tan_h + 2.5 * self.lmax)
                while (dd["glide"] + (L_use - 1) * per_u if dd["glide"] > 1e-6 else L_use * per_u) < need2:
                    L_use += 2
                dd = _line_deck_data(self.cell, line_rec, L_use)
                x0, ell_u, h, hinv = dd["x0"], dd["ell_u"], dd["h"], dd["hinv"]
                k_step = int(round((0.46 * ell_u - float((c_base - x0) @ u_ax)) / per_u))
                p0 = c_base + (k_step * per_u) * u_ax
                tau_u = (L_use * per_u if dd["glide"] < 1e-6 else (L_use - 1) * per_u) * u_ax
                q0 = iso_apply(h, p0 - tau_u)

                surf0 = build_mobius_surface(self.cell, self.lattice, line_rec, length=L_use, width=w_use, picture="raw")
                B_bef = _SurfaceBuilder(self.cell, self.lattice, surf0)
                th_up = math.atan2(float(n_ax[1]), float(n_ax[0])) + 0.5 * math.radians(ang)
                B_bef.add_disclination(ang, p0, np.array([math.cos(th_up), math.sin(th_up)]))
                d_before = B_bef.finish("mobius")

                B_aft = _SurfaceBuilder(self.cell, self.lattice, surf0)
                th_dn = math.atan2(float(-n_ax[1]), float(-n_ax[0])) + 0.5 * math.radians(ang)
                B_aft.add_disclination(ang, q0, np.array([math.cos(th_dn), math.sin(th_dn)]))
                d_after = B_aft.finish("mobius")
        else:
            raise ConfigError("vacuum 'I' transport is trivial")

        if picture == "relaxed":
            relax_picture(d_before)
            relax_picture(d_after)
            if d_cross is not None:
                relax_picture(d_cross)
        else:
            mark_long_bonds(d_before)
            mark_long_bonds(d_after)
            if d_cross is not None:
                mark_long_bonds(d_cross)

        # Measure local core charge in F0 on `before` and `after`
        def _measure_core_class(d_surf: Defect) -> str:
            cs = cores(d_surf)
            if not cs:
                return "I"
            c0 = cs[0]
            if c0["kind"] == "dislocation":
                return self.cat.identify(0.0, (np.eye(2), -c0["b"]))["class_id"]
            # Reconstruct rotation holonomy about core centre in F0
            X_ref = d_surf.info["reference"]
            if c0["site"] is not None:
                ctr_pt = X_ref[c0["site"]]
            else:
                H_f, _ = loop_charge(d_surf, c0["sites"])
                ctr_pt = np.linalg.solve(np.eye(2) - H_f[0], H_f[1])
            return self.cat.identify(c0["angle"], rotation_about(ctr_pt, math.radians(c0["angle"])))["class_id"]

        core_before = _measure_core_class(d_before)
        core_after = _measure_core_class(d_after)

        # Measure seam holonomy along a reference loop routed through the bottom half (d < 0) across j=+1
        # and closing near a ~ per_u on the left of p0/q0, so `before` gives h and `after` gives h o H!
        def _measure_ref_seam(d_surf: Defect, route_bottom: bool = True) -> dict:
            K_s, _ = irregular_sites(d_surf, self.cell, self.lattice)
            cone_s = set(d_surf.info.get("cone_sites", ()))
            avoid_s = K_s | cone_s
            X_s = d_surf.info["reference"]
            nbr_s: dict[int, list[int]] = {}
            for u, v in d_surf.edge:
                nbr_s.setdefault(u, []).append(v)
            # Pick start near a = 1.1 * per_u, d = -0.65 * w_use
            target_st = x0 + 1.1 * per_u * u_ax - 0.65 * w_use * n_ax
            cands = [i for i in range(len(d_surf.sites)) if i not in avoid_s]
            cands.sort(key=lambda i: _norm2(X_s[i] - target_st))
            start = cands[0]
            # Weighted Dijkstra preferring d < 0 when W == 0 and a < 0.35 * ell_u when W == 1
            import heapq

            dist_map: dict[tuple[int, int], float] = {(start, 0): 0.0}
            prev_map: dict[tuple[int, int], tuple[int, int] | None] = {(start, 0): None}
            pq = [(0.0, start, 0)]
            loop_found = None
            while pq:
                cost, u, W = heapq.heappop(pq)
                if cost > dist_map.get((u, W), 1e18) + 1e-9:
                    continue
                for v in nbr_s.get(u, ()):
                    if v in cone_s:
                        continue
                    g_e = d_surf.edge[(u, v)].get("g")
                    dw = 0
                    if g_e is not None and _det_sign_iso(g_e) < 0:
                        dw = 1 if float(g_e[1] @ u_ax) < 0.0 else -1
                    nW = W + dw
                    if nW not in (0, 1):
                        continue
                    av = float((X_s[v] - x0) @ u_ax)
                    dv = float((X_s[v] - x0) @ n_ax)
                    step_pen = 1.0 + (20.0 if v in K_s else 0.0)
                    if route_bottom:
                        d_ceil = min(float((p0 - x0) @ n_ax), float((q0 - x0) @ n_ax)) - 0.25 * self.lmax
                        if nW == 0 and dv > d_ceil and abs(av - 0.5 * ell_u) < 0.28 * ell_u:
                            # Block walking above q0/p0 in W==0 so loop passes strictly BELOW both p0 and q0!
                            step_pen += 500.0
                        if nW == 1 and av > 0.28 * ell_u:
                            # Close on the left side a < 0.28 * ell_u after crossing seam!
                            step_pen += 500.0
                    ncost = cost + step_pen
                    if v == start and nW == 1:
                        path = [start]
                        cur: tuple[int, int] | None = (u, W)
                        while cur is not None:
                            path.append(cur[0])
                            cur = prev_map[cur]
                        path.reverse()
                        loop_found = path[:-1]
                        break
                    if ncost < dist_map.get((v, nW), 1e18):
                        dist_map[(v, nW)] = ncost
                        prev_map[(v, nW)] = (u, W)
                        heapq.heappush(pq, (ncost, v, nW))
                if loop_found is not None:
                    break
            if loop_found is None:
                return self.measure_mobius_seam(d_surf)
            H_lp, _ = loop_charge(d_surf, loop_found)
            ident = self.identify_improper(iso_inverse(H_lp))
            ident["loop"] = loop_found
            return ident

        seam_bef_meas = _measure_ref_seam(d_before, route_bottom=True)
        seam_aft_meas = _measure_ref_seam(d_after, route_bottom=True)
        braid_lat = self.braid_around_mobius(defect_id, line_id=line_rec["id"], length=L_use)

        return dict(
            braid=braid,
            strip_length_used=L_use,
            before=d_before,
            after=d_after,
            cross_seam_dipole=d_cross,
            measured_core_before=core_before,
            measured_core_after=core_after,
            measured_seam_before=seam_bef_meas["seam_id"],
            measured_seam_after=seam_aft_meas["seam_id"],
            expected_seam_after=braid_lat["new_seam_id"],
            core_matched=(core_after == braid["transported_id"]),
            seam_matched=(seam_aft_meas["seam_id"] == braid_lat["new_seam_id"]),
        )

    def klein_bottle(
        self,
        line: str = "M1",
        n_u: int = 10,
        n_perp: int = 10,
        picture: str = "raw",
        name: str | None = None,
    ) -> Defect:
        """Build a closed boundary-free crystal Klein bottle for improper line class `line`."""
        line_rec = self._get_line_rec(line)
        d = build_klein_surface(self.cell, self.lattice, self.cat.Ainv, line_rec, n_u=n_u, n_perp=n_perp, picture=picture)
        d.info["name"] = name or f"Klein bottle {line_rec['id']} ({n_u}×{n_perp})"
        return d

    def klein_neutral(
        self,
        kind: str = "cross_dipole",
        line: str = "M1",
        b: str | np.ndarray | None = None,
        mode: str = "removal",
        angle: float | None = None,
        angles: tuple[float, ...] | list[float] | None = None,
        n_u: int = 10,
        n_perp: int = 10,
        picture: str = "raw",
        name: str | None = None,
    ) -> Defect:
        """Build globally neutral defect configurations on a closed boundary-free Klein bottle:
          - `kind='cross_dipole'`: cuts a dislocation pair across the orientation-reversing seam `h`
            via `add_cross_seam_dipole`, producing Core 1 with `+b` and Core 2 with `+M_h @ b` in `F0`
            (so an axis-parallel Burgers vector `b || u_ax` yields TWO IDENTICAL `+b` dislocations on a
            closed surface with `V - E + F = 0`, impossible on an orientable torus!).
          - `kind='dipole'`: cuts a standard orientable-patch dislocation dipole inside `F0` (`+b` and `-b`).
          - Refuses non-zero total scalar Frank angle (e.g. a single `+90°` or two same-sign `(+90°, +90°)`
            disclinations), which is obstructed by discrete Gauss-Bonnet `sum(alpha) = 360° * chi(K) = 0°`."""
        line_rec = self._get_line_rec(line)

        if kind in ("disclination", "same_sign_pair") or angle is not None or angles is not None:
            tot_ang = float(angle) if angle is not None else sum(float(a) for a in (angles or ()))
            if abs(tot_ang) > 1e-6 or kind == "disclination":
                raise ConfigError(
                    f"Gauss-Bonnet scalar obstruction on closed Klein bottle: orientation reversal preserves signed Frank angle (mu_h(alpha) = +alpha), so sum(alpha) = {tot_ang:+.1f} deg != 0 deg = 360 deg * chi(K) cannot close without boundary"
                )
            raise ConfigError("unsupported disclination configuration on Klein bottle; use kind='cross_dipole' or 'dipole'")

        if kind not in ("cross_dipole", "dipole"):
            raise ConfigError(f"klein_neutral kind must be 'cross_dipole' or 'dipole' (got {kind!r})")

        dd0 = _line_deck_data(self.cell, line_rec, max(8, int(n_u)))
        u_ax, n_ax, per_u, per_n = dd0["u_ax"], dd0["n_ax"], dd0["per_u"], dd0["per_n"]

        if b is None or (isinstance(b, str) and b == "axis"):
            b_vec = per_u * u_ax
        elif isinstance(b, str) and b == "perp":
            b_vec = per_n * n_ax
        elif isinstance(b, str):
            if b not in self.cat.by_id or self.cat.by_id[b]["kind"] != "dislocation":
                raise ConfigError(f"unknown dislocation Burgers id {b!r}")
            b_vec = self.cat.by_id[b]["b"].copy()
        else:
            b_vec = np.asarray(b, float)

        lb = _norm2(b_vec)
        sep_u = max(2.5 * per_u, 1.5 * self.lmax + 1.2 * lb)
        nu_use = max(int(n_u), 12, int(math.ceil((2.0 * sep_u + 5.0 * self.lmax + 3.0 * lb) / per_u)))
        np_use = max(int(n_perp), int(math.ceil((6.5 * per_n + 4.0 * self.lmax + 2.5 * lb) / per_n)))
        if (nu_use % 2) != (int(n_u) % 2):
            nu_use += 1
        if (np_use % 2) != (int(n_perp) % 2):
            np_use += 1

        kb = build_klein_surface(self.cell, self.lattice, self.cat.Ainv, line_rec, n_u=nu_use, n_perp=np_use, picture="raw")
        B = _SurfaceBuilder(self.cell, self.lattice, kb)
        x0 = kb.info["point"]
        a0, ell_u, ell_n, h = kb.info["a0"], kb.info["ell_u"], kb.info["ell_n"], kb.info["h"]
        hinv = iso_inverse(h)

        want_Sign = -1.0 if mode == "removal" else 1.0
        # Generic irrational-like nudge avoids placing dipole ray/endpoints right on lattice sites
        jitter = 0.0713 * self.lmax * u_ax + 0.1137 * self.lmax * n_ax
        out = None
        last_err: Exception | None = None
        for j_scale in (1.0, -1.3, 2.1):
            jit = j_scale * jitter
            B = _SurfaceBuilder(self.cell, self.lattice, kb)
            try:
                if kind == "cross_dipole":
                    p1 = None
                    p2_lift = None
                    for forward_seam in (True, False):
                        for d_mult in (-1.25, 1.25, -1.85, 1.85, -0.85, 0.85):
                            if forward_seam:
                                cand_p1 = x0 + (a0 + ell_u - sep_u) * u_ax + (d_mult * per_n) * n_ax + jit
                                cand_p2 = x0 + (a0 + sep_u) * u_ax + (d_mult * per_n) * n_ax + jit
                                cand_lift = iso_apply(h, cand_p2)
                            else:
                                cand_p1 = x0 + (a0 + sep_u) * u_ax + (d_mult * per_n) * n_ax + jit
                                cand_p2 = x0 + (a0 + ell_u - sep_u) * u_ax + (d_mult * per_n) * n_ax + jit
                                cand_lift = iso_apply(hinv, cand_p2)
                            cr = _cross2v(cand_lift - cand_p1, b_vec)
                            if mode in ("removal", "insertion"):
                                if cr * want_Sign > 1e-4 * lb:
                                    p1, p2_lift = cand_p1, cand_lift
                                    break
                            elif abs(cr) > 1e-4 * lb:
                                p1, p2_lift = cand_p1, cand_lift
                                break
                        if p1 is not None:
                            break
                    if p1 is None or p2_lift is None:
                        raise ConfigError("could not orient cross-seam dipole non-collinearly with b; choose another b or mode")
                    how_used = B.add_cross_seam_dipole(b_vec, p1, p2_lift)
                else:
                    # Standard dipole completely inside F0 (yields +b and -b)
                    p1 = None
                    p2_f0 = None
                    for d_step in (-1.35, 1.35, 0.0, -2.0, 2.0):
                        cand_p1 = x0 + (a0 + 0.34 * ell_u) * u_ax + (0.2 * per_n) * n_ax + jit
                        cand_p2 = x0 + (a0 + 0.66 * ell_u) * u_ax + ((0.2 + d_step) * per_n) * n_ax + jit
                        cr = _cross2v(cand_p2 - cand_p1, b_vec)
                        if cr * want_Sign > 1e-4 * lb:
                            p1, p2_f0 = cand_p1, cand_p2
                            break
                    if p1 is None or p2_f0 is None:
                        cand_p1 = x0 + (a0 + 0.5 * ell_u) * u_ax - 1.5 * per_n * n_ax + jit
                        cand_p2 = x0 + (a0 + 0.5 * ell_u) * u_ax + 1.5 * per_n * n_ax + jit
                        if _cross2v(cand_p2 - cand_p1, b_vec) * want_Sign < 0:
                            cand_p1, cand_p2 = cand_p2, cand_p1
                        p1, p2_f0 = cand_p1, cand_p2
                    how_used = B.add_cross_seam_dipole(b_vec, p1, p2_f0)

                cand_out = B.finish("klein")
                cand_out.info["how"] = how_used
                cand_out.info["neutral_kind"] = kind
                cand_out.info["b_input"] = b_vec
                if self.planar:
                    bad_f = check_frames(cand_out)
                    se = surface_euler_characteristic(cand_out)
                    tf = total_frank(cand_out)
                    if bad_f != 0 or se["euler_surface"] != 0 or se["orientable"] or abs(tf) > 1e-2:
                        raise ConfigError(
                            f"Klein bottle neutral check failed (frames_bad={bad_f}, euler={se['euler_surface']}, orientable={se['orientable']}, frank={tf:.2f})"
                        )
                out = cand_out
                break
            except ConfigError as e:
                last_err = e
                continue
        if out is None:
            raise last_err if last_err is not None else ConfigError("Klein bottle neutral build failed")
        if picture == "relaxed":
            relax_picture(out)
        else:
            mark_long_bonds(out)
        out.info["name"] = name or f"Klein {line_rec['id']} {kind} (b={_r(b_vec, 2)}, {how_used})"
        return out

    def plot_mobius_fusion_table(
        self,
        line_id: str = "M1",
        length: int | None = None,
        max_abs_angle: float = 180.0,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot a colour-coded cross-seam fusion table `C1 x_{M_k} C2 = C1 x mu_h(C2)` highlighting
        parity-unlocked and parity-blocked vacuum annihilation channels."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        line_rec = self._get_line_rec(line_id)
        t_ids = [r["id"] for r in self.cat.translations[:2]]
        d_ids = [r["id"] for r in self.cat.disclinations if abs(r["angle"]) <= max_abs_angle + 1e-6]
        labels = t_ids + d_ids
        n = len(labels)
        if n == 0:
            return None
        figsize = figsize or (max(7.8, 1.18 * n + 2.2), max(6.8, 0.88 * n + 2.0))
        fig, ax = plt.subplots(figsize=figsize)

        bg_colors = {
            "parity_unlocked": "#99e2b4",  # vivid emerald: vacuum unlocked ONLY across Möbius seam!
            "vacuum_both": "#d8f3dc",      # pale green: vacuum in both orientable & Möbius
            "parity_blocked": "#ffcb9a",   # warm amber: orientable vacuum BLOCKED across Möbius seam!
            "dipole": "#fde9b8",           # pale gold: opposite angles without vacuum in either
            "disclination": "#d8e8fa",     # light blue: composite disclination channel(s)
            "dislocation": "#ececec",      # grey: T x T without vacuum
            "boundary": "#f8d4d4",         # coral: +-360 whole-turn / cone-closing
        }

        for i, r_id in enumerate(labels):
            for j, c_id in enumerate(labels):
                fm = self.fuse_across_mobius(r_id, c_id, line_id=line_rec["id"], length=length, max_burgers_shells=1)
                law = fm["law"]
                vo, vm = fm["contains_vacuum_orientable"], fm["contains_vacuum_mobius"]
                if vm and not vo:
                    kind_key = "parity_unlocked"
                elif vm and vo:
                    kind_key = "vacuum_both"
                elif vo and not vm:
                    kind_key = "parity_blocked"
                elif law == "D×D->T":
                    kind_key = "dipole"
                elif law in ("D×T", "D×D->D"):
                    kind_key = "disclination"
                elif law == "T×T":
                    kind_key = "dislocation"
                else:
                    kind_key = "boundary"

                rect = plt.Rectangle((j, n - 1 - i), 1, 1, facecolor=bg_colors[kind_key], edgecolor="#666666", lw=0.8)
                ax.add_patch(rect)
                chs = fm["channels"]
                txt = " ⊕ ".join(chs[:3]) + (" ⊕ …" if len(chs) > 3 or fm["infinite_tower"] else "")
                txt = txt.replace(", ", ",")
                ax.text(j + 0.5, n - 0.5 - i, txt, ha="center", va="center", fontsize=6.9, wrap=True)

        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([k + 0.5 for k in range(n)])
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks([n - 0.5 - k for k in range(n)])
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(
            f"Cross-Seam Möbius Fusion ($C_1 \\times_{{{line_rec['id']}}} C_2 = C_1 \\times \\mu_h(C_2)$) for {self.group}",
            fontsize=10.5,
            pad=12,
        )
        legend_handles = [
            Patch(facecolor=bg_colors["parity_unlocked"], edgecolor="#666", label="Vacuum I UNLOCKED across Möbius seam (forbidden in $G_+$)"),
            Patch(facecolor=bg_colors["parity_blocked"], edgecolor="#666", label="Vacuum I BLOCKED across Möbius seam (allowed in $G_+$)"),
            Patch(facecolor=bg_colors["vacuum_both"], edgecolor="#666", label="Vacuum I allowed in both orientable & Möbius"),
            Patch(facecolor=bg_colors["dipole"], edgecolor="#666", label="Residual Burgers coset (no vacuum)"),
            Patch(facecolor=bg_colors["disclination"], edgecolor="#666", label="Disclination channel(s)"),
            Patch(facecolor=bg_colors["boundary"], edgecolor="#666", label="±360° whole-turn / cone-closing"),
        ]
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7.4, frameon=True)
        fig.tight_layout()
        return fig
