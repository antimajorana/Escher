"""Point defects (dislocations, disclinations) and multi-defect configurations on a planar lattice.

One engine for everything: the cone-safe multi-defect builder of legacy fusion_framework, wrapped in
`_Builder` with two fixes (see its methods). Gated against the independent C01-C07 engine
legacy.lab_defects in tests/test_build.py. Coordinates are Cartesian, in the frame of `.lattice.cell`.

Conventions (definition, set by the author; docs/DEVELOPMENT.md): K5 Burgers vector = sum of perfect-lattice
bond vectors round a CCW loop (with the builder's cut rule cross2(cut, b) < 0 the measured b equals the
requested b). Frank angle alpha > 0 removes a wedge. Outer CCW holonomy: disclination at c
(R_alpha, (I - R_alpha) c), dislocation (I, -b). Charge class = conjugacy class in the lifted proper
group G~+ (unreduced total Frank angle + G+ class of the holonomy). Symmetries ignore bond colours (K3).

Core graph (definition, technical choice of this module): a site is *irregular* when its
degree-labelled ball of radius r_det = max(2, ceil(largest perfect ring / 2)) is not the ball of a
perfect-lattice site; sites within r_det bonds of the patch rim are never candidates. The core graph
of radius r = subgraph induced on the sites within r bonds of an irregular site, nodes labelled by
their degree in the whole crystal. Graph only (no positions, no cut data). The ball test compares
layered signatures (per node: depth, degree, neighbours one layer in / same layer / one layer out):
isomorphic balls have equal signatures (proved result, trivial); equal signatures of non-isomorphic
balls are not excluded (assumption; second route: exact VF2++ ball test in tests/test_build.py).
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import warnings

import networkx as nx
import numpy as np

from ..legacy import fusion_framework as ff
from . import fast as _fast
from ._graph import _multi_bfs, layer_signature
from ._surgery import _Builder, candidate_cuts_360, coalesce_coincident_specs
from ..legacy.lab_defects import Perfect, frame_faces, rotation_about, iso_compose, iso_inverse
from ..legacy.wallpaper_lattices import UnitCell, primitive_cell, _symmetry_ops, has_crossings, make_cell

_CACHE: dict = {}


def _norm(v) -> float:
    return math.hypot(float(v[0]), float(v[1]))


# ==============================================================================================
# Lattice data: cell, perfect lattice, proper group, rotation centres, perfect ball signatures
# ==============================================================================================
class LatticeData:
    """Everything the builder and the classifier need about one lattice (cached per cell)."""

    def __init__(self, cell: UnitCell, fast: bool = True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = primitive_cell(cell, use_types=False)
        seen, bonds = set(), []
        for bd in P.bonds:                      # bond colours are ignored (K3): drop parallel copies
            if tuple(bd[:4]) not in seen:
                seen.add(tuple(bd[:4]))
                bonds.append(bd)
        self.cell = dataclasses.replace(P, bonds=bonds)
        self.L = _fast.perfect(self.cell) if fast else Perfect(self.cell)     # fast: same values, memoised
        self.A = self.cell.A
        self.Ainv = np.linalg.inv(self.A)
        self.ops = [(self.A @ Mf @ self.Ainv, self.A @ np.asarray(t, float)) for Mf, t in _symmetry_ops(self.cell, use_types=False)]
        self.ops_plus = [g for g in self.ops if np.linalg.det(g[0]) > 0]
        self.ops_minus = [g for g in self.ops if np.linalg.det(g[0]) < 0]
        self.m0 = self.ops_minus[0] if self.ops_minus else None
        self.planar = not (_fast.has_crossings_fast(self.cell) if fast else has_crossings(self.cell))
        L = self.L
        lens = [_norm(L.pos(k2) - L.pos((0, 0, s))) for s in L.rule for k2, _ in L.nbrs((0, 0, s))]
        self.lmax, self.lmean = max(lens), float(np.mean(lens))
        self.scale = max(_norm(self.cell.a1), _norm(self.cell.a2))
        self._centres()
        self._perfect_balls()

    # --- rotation centres and their G+ classes ----------------------------------------------------
    def frac(self, p) -> np.ndarray:
        return self.Ainv @ np.asarray(p, float)

    def same_mod_lattice(self, p, q, tol=1e-5) -> bool:
        f = self.Ainv @ (np.asarray(p, float) - np.asarray(q, float))
        return bool(np.all(np.abs(f - np.round(f)) < tol))

    def reduce(self, p) -> np.ndarray:
        f = self.Ainv @ np.asarray(p, float)
        f = f - np.floor(f + 1e-6)
        return self.A @ f

    def _centres(self):
        pts = []
        for M, t in self.ops_plus:
            if np.allclose(M, np.eye(2), atol=1e-6):
                continue
            for m in itertools.product(range(-2, 3), repeat=2):
                c = self.reduce(np.linalg.solve(np.eye(2) - M, t + self.A @ np.array(m, float)))
                if not any(self.same_mod_lattice(c, q) for q in pts):
                    pts.append(c)
        order = lambda c: sum(1 for M, t in self.ops_plus if self.same_mod_lattice(M @ c + t, c))
        cls = list(range(len(pts)))
        for i, c in enumerate(pts):
            for M, t in self.ops_plus:
                im = M @ c + t
                for j, q in enumerate(pts):
                    if cls[j] != cls[i] and self.same_mod_lattice(im, q):
                        old = cls[j]
                        cls = [cls[i] if x == old else x for x in cls]
        groups: dict[int, list] = {}
        for i, c in enumerate(cls):
            groups.setdefault(c, []).append(pts[i])
        recs = []
        for members in groups.values():
            rep = min(members, key=lambda q: (round(float(q[0]), 4), round(float(q[1]), 4)))
            recs.append(dict(point=rep, order=order(rep), members=members, on_site=self.L.locate(rep) is not None))
        recs.sort(key=lambda r: (-r["order"], tuple(np.round(self.frac(r["point"]), 4))))
        for k, r in enumerate(recs):
            r["label"] = f"{r['order']}{'abcdefghijklmnop'[k]}"
            r["angles"] = [k2 * 360.0 / r["order"] for k2 in range(-(r["order"] // 2), r["order"] // 2 + 1) if k2 != 0]
        # One naming scheme in the whole package (lead, 2026-11-13): take the ITA G+-class labels
        # ('2a', '2a(i)', ...) of DefectCatalogue.centre_classes, matched point by point mod the lattice.
        # The order-letter labels above stay only as a fallback if the catalogue cannot be built.
        try:
            from .catalogue import DefectCatalogue
            crecs = DefectCatalogue(self.cell).centre_classes
            for r in recs:
                hit = [c for c in crecs if any(self.same_mod_lattice(r["point"], q) for q in c["members"])]
                if len(hit) == 1 and hit[0]["order"] == r["order"]:
                    r["label"], r["gclass"] = hit[0]["label"], hit[0]["gclass"]
        except Exception as exc:                        # pragma: no cover - fallback path
            warnings.warn(f"ITA centre labels unavailable ({exc}); using order-letter labels")
        self.centres = recs

    def centre_class(self, c) -> dict | None:
        """The G+ class record of a point, or None if it is not a rotation centre."""
        for r in self.centres:
            if any(self.same_mod_lattice(c, q) for q in r["members"]):
                return r
        return None

    def nearest_centre(self, angle: float, near) -> np.ndarray:
        near = np.asarray(near, float)
        n0 = np.floor(self.Ainv @ near)
        best = None
        for r in self.centres:
            if abs(angle * r["order"] / 360.0 - round(angle * r["order"] / 360.0)) > 1e-6:
                continue
            for q in r["members"]:
                for i, j in itertools.product(range(-1, 3), repeat=2):
                    p = q + self.A @ (n0 + np.array([i, j], float))
                    if best is None or _norm(p - near) < best[0] - 1e-9:
                        best = (_norm(p - near), p)
        if best is None:
            raise ValueError(f"no rotation centre of this lattice allows {angle:g} degrees")
        return best[1]

    # --- perfect-lattice ball signatures (for the graph-only core) --------------------------------
    def _perfect_balls(self):
        L = self.L
        keys = L.sites_in_disk((0.0, 0.0), 9.0 * self.lmax + 2.0 * self.scale)
        index = {k: i for i, k in enumerate(keys)}
        adj = [[index[k2] for k2, _ in L.nbrs(k) if k2 in index] for k in keys]
        ring = 0
        if self.planar:
            from ..legacy.lab_defects import perfect_patch
            pp = perfect_patch(self.cell, 3.0 * self.scale + 2.0 * self.lmax)
            faces = frame_faces(pp)
            k = max(range(len(faces)), key=lambda i: len(faces[i]))
            ring = max(len(f) for i, f in enumerate(faces) if i != k)
        self.max_ring = ring
        self.r_det = max(2, math.ceil(ring / 2)) if ring else 3
        deg = [len(a) for a in adj]
        self.perfect_sigs = {layer_signature(adj, deg, index[(0, 0, s)], self.r_det) for s in L.rule}
        self.perfect_degree = {s: len(L.rule[s]) for s in L.rule}


def lattice_data(lattice, fast: bool | None = None) -> LatticeData:
    """UnitCell, legacy Family, recipe dict, catalogue name, or LatticeData -> cached LatticeData."""
    if isinstance(lattice, LatticeData):
        return lattice
    if isinstance(lattice, str):
        cell = make_cell(lattice)
    elif isinstance(lattice, dict):
        from ..legacy.lattice_families import Family
        cell = Family.from_recipe(lattice).cell()
    elif isinstance(lattice, UnitCell):
        cell = lattice
    elif hasattr(lattice, "cell") and callable(lattice.cell):
        cell = lattice.cell()
    else:
        raise TypeError("lattice must be a UnitCell, a Family, a recipe dict or a catalogue name")
    key = (_fast.enabled(fast), np.round(cell.A, 9).tobytes(), np.round(np.asarray(cell.basis, float), 9).tobytes(),
           tuple(sorted(tuple(b[:4]) for b in cell.bonds)))
    return _CACHE[key] if key in _CACHE else _CACHE.setdefault(key, LatticeData(cell, fast=key[0]))


# ==============================================================================================
# Graph-only core
# ==============================================================================================
def graph_core(G: nx.Graph, rim: set, lat: LatticeData, radius: int = 2) -> tuple[set, nx.Graph]:
    """(irregular sites, degree-labelled core graph of `radius`) of any crystal graph G.
    rim: sites on or next to the patch boundary. Raises ValueError if the core ball meets the rim."""
    nodes = list(G.nodes)
    pos = {v: i for i, v in enumerate(nodes)}
    adj = [[pos[w] for w in G.adj[v]] for v in nodes]
    deg = [len(a) for a in adj]
    rim_i = {pos[v] for v in rim if v in pos}
    drim = _multi_bfs(adj, rim_i)
    cand = [i for i in range(len(nodes)) if drim.get(i, 10**9) > lat.r_det]   # irregular: signature not perfect
    K = {nodes[i] for i in _fast.irregular(adj, deg, cand, lat.r_det, lat.perfect_sigs)}
    ball = _multi_bfs(adj, {pos[v] for v in K}, cutoff=radius)
    if any(drim.get(i, 10**9) <= lat.r_det for i in ball):     # detection is off there: refuse, never truncate
        raise ValueError("the core ball reaches the patch rim: build with a larger R")
    H = G.subgraph([nodes[i] for i in ball]).copy()
    for v in H:
        H.nodes[v]["deg"] = G.degree(v)
    return K, H


def patch_rim(lat: LatticeData, refs, sublattice, degrees, centre, R) -> set:
    """Sites on or next to the boundary of a disk patch: reference position farther than R - 1.5 lmax
    from the patch centre, or a missing bond and farther than R - 3 lmax (technical choice)."""
    out = set()
    c = np.asarray(centre, float)
    for i, (x, s, k) in enumerate(zip(refs, sublattice, degrees)):
        r = _norm(np.asarray(x, float) - c)
        if r > R - 1.5 * lat.lmax or (k < lat.perfect_degree[s] and r > R - 3.0 * lat.lmax):
            out.add(i)
    return out


def isomorphic_cores(G1: nx.Graph, G2: nx.Graph) -> bool:
    """Exact degree-labelled graph isomorphism (networkx VF2++)."""
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False
    if G1.number_of_nodes() == 0:
        return True
    return bool(nx.vf2pp_is_isomorphic(G1, G2, node_label="deg"))


# ==============================================================================================
# Charge classes in G~+
# ==============================================================================================
def _orbit_rep(lat: LatticeData, b) -> np.ndarray:
    orb = [np.round(M @ b, 12) + 0.0 for M, _ in lat.ops_plus]
    return max(orb, key=lambda v: (round(float(v[0]), 5), round(float(v[1]), 5)))


def classify_charge(lat: LatticeData, frank: float | None, H) -> dict:
    """Class in G~+ of a measured (total Frank angle, holonomy H)."""
    M, t = H
    if float(np.linalg.det(M)) < 0.0:
        return dict(angle=None, H=(M.copy(), t.copy()), orientation_preserving=False,
                    kind="non_orientable", label="non-orientable loop (det H < 0)", single=None)
    rot_deg = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    if frank is None:
        frank = rot_deg
    if abs(((frank - rot_deg) + 180.0) % 360.0 - 180.0) > 1e-3:
        raise ValueError(f"Frank angle {frank:.4f} disagrees with the holonomy rotation {rot_deg:.4f}")
    out = dict(angle=float(round(frank, 6)), H=(M.copy(), t.copy()))
    if np.allclose(M, np.eye(2), atol=1e-6):
        b = -t + 0.0
        b[np.abs(b) < 1e-9] = 0.0
        f = lat.Ainv @ b
        out["b"] = b
        out["lattice_vector"] = bool(np.all(np.abs(f - np.round(f)) < 1e-5))
        if abs(frank) < 1e-6:
            if _norm(b) < 1e-6:
                out.update(kind="vacuum", label="I (no defect)", single=None)
                return out
            if not out["lattice_vector"]:
                out.update(
                    kind="partial_dislocation",
                    b=b.copy(),
                    lattice_vector=False,
                    single=None,
                    length=_norm(b),
                    label=f"partial dislocation b = {np.round(b, 4).tolist()} (|b| = {_norm(b):.4f}, fractional)",
                )
                return out
            rep = _orbit_rep(lat, b)
            out.update(kind="dislocation", b_rep=rep, length=_norm(b),
                       label=f"dislocation [b] = P+ orbit of {np.round(rep, 4).tolist()} (|b| = {_norm(b):.4f})",
                       single=dict(kind="dislocation", b=b.copy()))
            return out
        out.update(kind="whole_turn", label=f"whole-turn {frank:+.0f} deg with b = {np.round(b, 4).tolist()}", single=None)
        return out
    c = np.linalg.solve(np.eye(2) - M, t)
    rec = lat.centre_class(c)
    w = rec["label"] if rec else "?"
    # Author decision (2026-11-22): keep single = None for 180° < |angle| < 360°, matching docs/RUST_PLAN.md §9.
    allowed = (rec is not None and abs(frank) <= 180 + 1e-6
               and abs(frank * rec["order"] / 360.0 - round(frank * rec["order"] / 360.0)) < 1e-6)
    out.update(kind="disclination", centre=c, centre_class=w,
               label=f"disclination D({w}, {frank:+.0f} deg) about {np.round(c, 4).tolist()}",
               single=dict(kind="disclination", angle=float(round(frank, 6)), at=c.copy()) if allowed else None)
    return out


def same_charge(lat: LatticeData, q1: dict, q2: dict, mirror: bool = False) -> bool:
    """Same class in G~+ (mirror=True: also allow the image under an orientation-reversing element of G)."""
    if abs(q1["angle"] - q2["angle"]) > 1e-3:
        return False
    if ff.same_proper_holonomy(q1["H"], q2["H"], lat.ops_plus, lat.A, lat.Ainv):
        return True
    if mirror and lat.m0 is not None:
        Hm = iso_compose(lat.m0, iso_compose(iso_inverse(q1["H"]), iso_inverse(lat.m0)))
        return bool(ff.same_proper_holonomy(Hm, q2["H"], lat.ops_plus, lat.A, lat.Ainv))
    return False


# ==============================================================================================
# The builder (cone-safe builder in _surgery.py) and its driver
# ==============================================================================================
def validate_specs(lat: LatticeData, specs, snap: bool = False) -> list[dict]:
    """Normalise and check every spec; refuse impossible requests with ValueError."""
    out = []
    for idx, s in enumerate(specs):
        if not isinstance(s, dict):
            raise ValueError(f"spec {idx + 1}: give a dict(kind=..., ...)")
        s = dict(s)
        kind = {"translation": "dislocation", "rotation": "disclination"}.get(s.get("kind"), s.get("kind"))
        s["kind"] = kind
        if s.get("cut") is not None:
            s["cut"] = np.asarray(s["cut"], float)
        if kind == "dislocation":
            b = np.asarray(s["b"], float)
            f = lat.Ainv @ b
            if _norm(b) < 1e-9 or np.any(np.abs(f - np.round(f)) > 1e-6):
                raise ValueError(f"spec {idx + 1}: b = {b.tolist()} is not a non-zero lattice translation")
            s["b"] = b
            s["at"] = np.asarray(s["at"], float) if s.get("at") is not None else \
                0.371 * lat.cell.a1 + 0.413 * lat.cell.a2 + 3 * idx * lat.cell.a1
        elif kind == "disclination":
            ang = float(s["angle"])
            if abs(ang) < 1e-9 or abs(ang) >= 360 - 1e-9:
                raise ValueError(f"spec {idx + 1}: a single disclination needs 0 < |angle| < 360")
            if s.get("at") is None:
                raise ValueError(f"spec {idx + 1}: give the centre at=(x, y)")
            at = np.asarray(s["at"], float)
            rec = lat.centre_class(at)
            ok = rec is not None and abs(ang * rec["order"] / 360.0 - round(ang * rec["order"] / 360.0)) < 1e-6
            if not ok:
                if not snap:
                    why = "is not a rotation centre" if rec is None else f"is a {rec['order']}-fold centre ({rec['label']})"
                    raise ValueError(f"spec {idx + 1}: {np.round(at, 4).tolist()} {why}; {ang:g} deg is not allowed there "
                                     f"(nearest allowed centre: {np.round(lat.nearest_centre(ang, at), 4).tolist()}; or pass snap=True)")
                s["moved_from"] = at
                at = lat.nearest_centre(ang, at)
            if not lat.L.is_symmetry(rotation_about(at, math.radians(ang)), probe=2.2):   # second route
                raise ValueError(f"spec {idx + 1}: rotation by {ang:g} about {at.tolist()} is not a lattice symmetry")
            s["angle"], s["at"] = ang, at
        elif kind == "half_wall":
            from ._half_wall import resolve_half_wall_spec
            s = resolve_half_wall_spec(lat, s)
        else:
            raise ValueError(f"spec {idx + 1}: kind must be 'dislocation', 'disclination', or 'half_wall', got {kind!r}")
        out.append(s)
    hw_count = sum(1 for s in out if s["kind"] == "half_wall")
    if hw_count > 1:
        raise ValueError("at most one half_wall spec is supported per configuration")
    total = sum((s["ending"]["frank_angle_deg"] or 0.0) if s["kind"] == "half_wall" else s["angle"]
                for s in out if s["kind"] in ("disclination", "half_wall"))
    if total >= 360.0 - 1e-6:
        raise ValueError(f"total Frank angle {total:+g} deg >= 360 deg: an open patch cannot close into a cone")
    return out


def _run(lat: LatticeData, specs, R=None, centre=None, check=True, fast=None):
    """Build with the cone-safe builder. Returns (Defect, check dict). check=False skips refusal on
    failed invariants (used by the stress tests, which check them independently)."""
    hw_specs = [s for s in specs if s["kind"] == "half_wall"]
    pt_specs = [s for s in specs if s["kind"] != "half_wall"]
    if hw_specs and not pt_specs:
        from ._half_wall import build_half_wall_defect
        return build_half_wall_defect(lat, hw_specs[0], R=R, centre=centre, check=check, fast=fast)
    eff = coalesce_coincident_specs(pt_specs)
    pts = np.array([s["at"] for s in specs]).reshape(-1, 2)
    ctr = pts.mean(axis=0) if centre is None else np.asarray(centre, float)
    if R is None:
        max_b = max([_norm(s["b"]) for s in pt_specs if s["kind"] == "dislocation"]
                    + [_norm(s["holonomy"][1]) for s in hw_specs], default=0.0)
        R = float(np.max(np.linalg.norm(pts - ctr, axis=1))) + max(12.5 * lat.lmax + max_b, 5.0 * lat.scale)
    b_cls = _fast.builder_class() if _fast.enabled(fast) else _Builder
    if hw_specs:
        from ._half_wall import build_half_wall_defect, seed_builder_from_defect
        d_hw, _ = build_half_wall_defect(lat, hw_specs[0], R=float(R), centre=ctr, check=False, fast=fast)
        make_b = lambda: seed_builder_from_defect(b_cls, lat, d_hw, ctr, float(R))
    else:
        make_b = lambda: b_cls(lat.cell, lat.L, ctr, float(R))
    from ._surgery import run_eff_with_fallback
    B = run_eff_with_fallback(make_b, eff, lat)
    d = B.finish()
    expected = sum(s["angle"] for s in eff if s["kind"] == "disclination") + sum(
        (s["ending"]["frank_angle_deg"] or 0.0) for s in hw_specs
    )
    rep = dict(R=float(R), centre=ctr, expected_frank=expected)
    if lat.planar:
        rep.update(frames_bad=_fast.check_frames(d, fast), euler=len(d.sites) - len(d.edges) + len(_fast.frame_faces(d, fast)),
                   frank=_fast.total_frank(d, fast), min_degree=int(d.degrees().min()))
        bad = rep["frames_bad"] != 0 or rep["euler"] != 2 or abs(rep["frank"] - expected) > 1e-2 or rep["min_degree"] < 1
        if bad and check:
            raise ValueError(
                f"combinatorial check failed (frames_bad={rep['frames_bad']}, euler={rep['euler']}, "
                f"frank={rep['frank']:.3f} vs {expected:.3f}, min_deg={rep['min_degree']}); "
                "move the defects apart or give other cuts"
            )
    return d, rep


def build(lattice, *specs, R=None, picture="relaxed", centre=None, snap=False, check=True, fast=None):
    """Build any list of dislocations and disclinations on an open disk patch.

    spec: dict(kind="dislocation", b=(bx, by), at=(x, y), cut=None)
          dict(kind="disclination", angle=deg, at=(cx, cy), cut=None)
    Disclination centres must be rotation centres allowing the angle (snap=True moves them to the
    nearest such centre instead of refusing). A dislocation cut c can be any non-parallel vector
    (cross2(c, b) < 0 removes a strip, cross2(c, b) > 0 inserts one).
    fast: None = fast path (fast.py, fast_relax.py) unless PAPERTILLING_LEGACY=1; False = legacy engines.
    picture: "raw" = the builder's composed maps (bonds stretched > 1.8x drawn as dotted stubs; can cross);
    "relaxed" = the spring relaxation (fast_relax); if that picture has crossing bonds or folded faces it is
    replaced by a planar one from draw.ensure_embedded (local Tutte repair or Tutte embedding, then the same
    relaxation kept planar); defect.info['drawing'] says which stage drew it."""
    if picture not in ("relaxed", "raw", "elastic"):
        raise ValueError("picture must be 'relaxed', 'raw', or 'elastic'")
    if len(specs) == 1 and isinstance(specs[0], (list, tuple)):
        specs = tuple(specs[0])
    if not specs:
        raise ValueError("give at least one defect spec")
    lat = lattice_data(lattice, fast)
    sp = validate_specs(lat, specs, snap=snap)
    d, rep = _run(lat, sp, R=R, centre=centre, check=check, fast=fast)
    if picture == "relaxed":
        _fast.relax_picture(d, fast=fast)
        if lat.planar:                       # a picture with crossing bonds or folded faces is redrawn
            from .draw import ensure_embedded
            ensure_embedded(d, overlaps=False)
    elif picture == "elastic":
        from .gsv_elastic import relax_elastic_2d
        relax_elastic_2d(d, lattice=lat)
    else:
        ff.mark_long_bonds(d)
    return Configuration(d, sp, lat, rep)


class Configuration:
    """A built crystal: .defect (legacy Defect), .specs, .lattice (LatticeData), .report (build checks)."""

    def __init__(self, defect, specs, lat: LatticeData, report: dict):
        self.defect, self.specs, self.lattice, self.report = defect, specs, lat, report
        self._core = {}

    def total_charge(self) -> dict:
        """Class in G~+ read from the graph: holonomy round the outer boundary + total Frank angle."""
        if "_charge" not in self.defect.info:
            H, S = _fast.total_charge(self.defect)
            q = classify_charge(self.lattice, _fast.total_frank(self.defect), H)
            q["S"] = S
            self.defect.info["_charge"] = q
        return self.defect.info["_charge"]

    def cores(self) -> list[dict]:
        """One entry per face with a non-trivial holonomy: kind, local class, sites, ring, bonds."""
        d = self.defect
        deg = d.degrees()
        out = []
        for c in _fast.cores(d):
            rec = dict(kind=c["kind"], sites=list(c["sites"]), ring=c.get("ring"),
                       coordination=[int(deg[v]) for v in c["sites"]])
            if c["kind"] == "dislocation":
                rec["b"] = np.asarray(c["b"], float)
                rec["label"] = f"dislocation b = {np.round(c['b'], 4).tolist()} on a {c['ring']}-ring"
            else:
                rec["angle"] = c["angle"]
                rec["site"] = c.get("site")
                rec["label"] = (f"disclination {c['angle']:+.0f} deg on a site with {c['bonds']} bonds" if c.get("site") is not None
                                else f"disclination {c['angle']:+.0f} deg on a {c['ring']}-ring")
            out.append(rec)
        return out

    def rim(self) -> set:
        d = self.defect
        return patch_rim(self.lattice, d.info["reference"], [s[0][2] for s in d.sites], d.degrees(),
                         d.info["centre"], d.info["R"])

    def core_sites(self) -> set:
        return self._get_core(2)[0]

    def core_graph(self, radius: int = 2) -> nx.Graph:
        """Degree-labelled subgraph within `radius` bonds of the irregular sites (graph-only core)."""
        return self._get_core(radius)[1]

    def _get_core(self, radius):
        if radius not in self._core:
            self._core[radius] = graph_core(self.defect.to_networkx(), self.rim(), self.lattice, radius)
        return self._core[radius]

    def inspect(self, show: bool = True) -> dict:
        d = self.defect
        q = self.total_charge()
        faces = _fast.frame_faces(d) if self.lattice.planar else []
        out = dict(V=len(d.sites), E=len(d.edges), F=len(faces), euler=len(d.sites) - len(d.edges) + len(faces),
                   frames_consistent=_fast.check_frames(d) == 0, total_frank=_fast.total_frank(d) if faces else None,
                   requested_frank=self.report["expected_frank"], charge=q["label"], single=q["single"], cores=self.cores())
        if show:
            print(f"V = {out['V']}, E = {out['E']}, F = {out['F']} (with the outside): V - E + F = {out['euler']}; "
                  f"frames consistent: {out['frames_consistent']}")
            print(f"total Frank angle {out['total_frank']:+.3f} deg (requested sum {out['requested_frank']:+g})")
            print(f"total charge: {q['label']}" + (f"; equals the single defect {_fmt(q['single'])}" if q["single"] else ""))
            for c in out["cores"]:
                print(f"  core: {c['label']}; coordination of its sites {c['coordination']}")
        return out

    def plot(self, ax=None, title=None, mode=None):
        """Draw the crystal and its cores. When defect.info['drawing'] records crossings, overlaps, crowded
        sites or folds left in the picture, a line saying how many is added to the title."""
        if mode is not None:
            from .gsv_elastic import plot_elastic
            return plot_elastic(self, ax=ax, title=title, mode=mode)
        import matplotlib.pyplot as plt

        from .draw import drawing_note
        d = self.defect
        ax = ax or plt.subplots(figsize=(6, 6))[1]
        d.plot(ax=ax, site_size=6)
        P = d.positions
        for c in self.cores():
            Q = P[c["sites"]]
            col = "#e09a1a" if c["kind"] == "dislocation" else ("#d9432f" if c["angle"] > 0 else "#1f6fc5")
            if len(Q) > 2:
                ax.fill(Q[:, 0], Q[:, 1], color=col, alpha=0.4, zorder=1)
            ax.scatter(Q[:, 0].mean(), Q[:, 1].mean(), s=150, facecolors="none", edgecolors=col, linewidths=1.6, zorder=5)
        note = drawing_note(d.info.get("drawing"))
        title = title or self.total_charge()["label"]
        ax.set_title(f"{title}\n{note}" if note else title, fontsize=8.5)
        return ax

    def relax_3d(self, kappa_b: float = 0.02, max_iter: int = 400):
        """Relax this defect configuration in 3D under stretching and hinge bending."""
        from ..surfaces.elastic import relax_configuration_3d
        return relax_configuration_3d(self, kappa_b=kappa_b, max_iter=max_iter)


def _fmt(s: dict) -> str:
    if s["kind"] == "dislocation":
        return f"dislocation b = {np.round(s['b'], 4).tolist()}"
    return f"disclination {s['angle']:+g} deg at {np.round(s['at'], 4).tolist()}"
