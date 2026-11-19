"""Regular {p,q,r} honeycombs of S³ / E³ / H³, 3D Voronoi cells, Dirichlet domains of closed
3-manifolds, and clean-room P / D / G triply-periodic nets and surfaces.

    import papertilling as pt
    hc = pt.honeycomb3d
    hc.classify_schlafli(5, 3, 5)["regime"]          # 'HyperbolicCompactH3'
    cx = hc.build_honeycomb(3, 3, 5)                 # the 600-cell (stereographic image of S³)
    hc.counts(cx)                                     # (120, 720, 1200, 600)
    hc.build_honeycomb(4, 3, 6, shells=1)             # H³, ideal vertices on |p| = 1 (Poincaré ball)
    hc.dirichlet_manifold("seifert_weber")["quotient"]["h1_label"]   # 'Z_5^3'
    hc.nodal_surface_mesh("G", n=24)["genus_primitive"]               # 3

Every function returns plain dicts (the same keys as the Rust core). The compiled extension
`papertilling_rs` is used when present; otherwise (or with `PAPERTILLING_NO_RUST=1`) the pure-Python
implementation below runs. Nets are derived from classical geometry only (no external net database).
"""
from __future__ import annotations

import itertools
import json
import math
from collections import deque
from typing import Any

import numpy as np

from . import _rust

__all__ = [
    "classify_schlafli", "build_honeycomb", "build_euclidean_honeycomb", "voronoi_cell_3d",
    "lattice_voronoi_cell", "dirichlet_manifold", "tpms_net", "petrie_coxeter_surface",
    "nodal_surface_mesh", "nodal", "counts", "euler",
]

SPHERICAL = {(3, 3, 3): "5-cell {3,3,3}", (4, 3, 3): "8-cell (tesseract) {4,3,3}", (3, 3, 4): "16-cell {3,3,4}",
             (3, 4, 3): "24-cell {3,4,3}", (5, 3, 3): "120-cell {5,3,3}", (3, 3, 5): "600-cell {3,3,5}"}
EUCLIDEAN_KINDS = {"cubic": ("SC", "cubic {4,3,4} (SC Voronoi cells)"),
                   "bitruncated_cubic": ("BCC", "bitruncated cubic (Kelvin; BCC Voronoi truncated octahedra)"),
                   "rhombic_dodecahedral": ("FCC", "rhombic dodecahedral (FCC Voronoi cells)"),
                   "alternated_cubic": (None, "alternated cubic (octet: tetrahedra + octahedra)")}


def _call(name: str, *args):
    rs = _rust.get()
    if rs is not None and hasattr(rs, name):
        return json.loads(getattr(rs, name)(*args))
    return None


def counts(cx: dict) -> tuple[int, int, int, int]:
    """`(V, E, F, C)` of a cell complex dict."""
    return len(cx["points"]), len(cx["edges"]), len(cx["faces"]), len(cx["cells"])


def euler(cx: dict) -> int:
    v, e, f, c = counts(cx)
    return v - e + f - c


# ---------------------------------------------------------------- classification

def _kind(k: int) -> str:
    return "spherical" if k < 4 else ("Euclidean" if k == 4 else "hyperbolic")


def _gram_det(p, q, r) -> float:
    a, b, c = (math.cos(math.pi / m) for m in (p, q, r))
    return 1 - a * a - b * b - c * c + a * a * c * c


def classify_schlafli(p: int, q: int, r: int) -> dict[str, Any]:
    """Exact regime of {p,q,r} from cell_k = (p-2)(q-2), vfig_k = (q-2)(r-2) (det G is a diagnostic)."""
    out = _call("honeycomb3d_classify_json", p, q, r)
    return out if out is not None else _py_classify(p, q, r)


def _py_classify(p, q, r):
    out = dict(p=p, q=q, r=r, regime="DegenerateRefused", det_g=None, cell_k=0, vfig_k=0,
               has_ideal_vertices=False, has_infinite_euclidean_cells=False, reason="")
    if min(p, q, r) < 3:
        out["reason"] = (f"[PROVED: reducible or degenerate Schläfli symbol {{{p},{q},{r}}} (requires p, q, r >= 3; "
                         "an entry 2 splits the Coxeter diagram, e.g. {4,4,2} has det G = 0 but is a reducible dihedron)]")
        if min(p, q, r) >= 2:
            out["det_g"] = _gram_det(p, q, r)
        return out
    ck, vk = (p - 2) * (q - 2), (q - 2) * (r - 2)
    out.update(cell_k=ck, vfig_k=vk, det_g=_gram_det(p, q, r))
    if ck > 4 or vk > 4:
        why = []
        if ck > 4:
            why.append(f"the cell {{{p},{q}}} is hyperbolic ((p-2)(q-2) = {ck} > 4): its cells are ultra-infinite, "
                       "extending beyond the sphere at infinity S²_∞")
        if vk > 4:
            why.append(f"the vertex figure {{{q},{r}}} is hyperbolic ((q-2)(r-2) = {vk} > 4): its vertices are "
                       "ultra-ideal, outside the Klein ball")
        out.update(regime="HyperidealRefused", reason=f"[PROVED: hyperideal symbol {{{p},{q},{r}}}; {'; '.join(why)}]")
        return out
    out.update(has_ideal_vertices=vk == 4, has_infinite_euclidean_cells=ck == 4)
    if ck < 4 and vk < 4:
        if (p, q, r) in SPHERICAL:
            out.update(regime="SphericalS3", reason=f"[PROVED: {{{p},{q},{r}}} is a regular convex polychoron of S³ (det G > 0)]")
        elif (p, q, r) == (4, 3, 4):
            out.update(regime="EuclideanE3", reason="[PROVED: {4,3,4} is the cubic honeycomb of E³ (det G = 0)]")
        else:
            out.update(regime="HyperbolicCompactH3", reason=f"[PROVED: {{{p},{q},{r}}} is a compact regular honeycomb of H³ (det G < 0)]")
        return out
    out["regime"] = "HyperbolicParacompactH3"
    out["reason"] = (f"[PROVED: {{{p},{q},{r}}} is a paracompact regular honeycomb of H³ (det G < 0): cell {{{p},{q}}} is "
                     f"{_kind(ck)}, vertex figure {{{q},{r}}} is {_kind(vk)}"
                     + ("; vertices are ideal (on S²_∞), horosphere vertex figures" if vk == 4 else "")
                     + ("; cells are infinite horospherical tilings, truncated to finite patches" if ck == 4 else "") + "]")
    return out


# ---------------------------------------------------------------- shared helpers

class _Index:
    """Tolerance point index (rounded-key buckets, neighbouring buckets searched on a miss)."""

    def __init__(self, tol=1e-7):
        self.tol, self.buckets, self.pts = tol, {}, []

    def insert(self, x):
        x = tuple(float(c) for c in x)
        k = tuple(math.floor(c / (4 * self.tol)) for c in x)
        for off in itertools.product((0, -1, 1), repeat=len(x)):
            for i in self.buckets.get(tuple(a + b for a, b in zip(k, off)), ()):
                if sum((a - b) ** 2 for a, b in zip(self.pts[i], x)) <= self.tol ** 2:
                    return i, False
        self.buckets.setdefault(k, []).append(len(self.pts))
        self.pts.append(x)
        return len(self.pts) - 1, True


class _Builder:
    def __init__(self, name, regime):
        self.out = dict(name=name, regime=regime, coords4=[], points=[], klein=[], is_ideal=[], edges=[], faces=[],
                        cells=[], cell_shell=[], cell_truncated=[])
        self.face_ids, self.edge_ids = {}, set()

    def add_face(self, cyc):
        key = tuple(sorted(cyc))
        if key in self.face_ids:
            return self.face_ids[key]
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            e = (min(a, b), max(a, b))
            if e not in self.edge_ids:
                self.edge_ids.add(e)
                self.out["edges"].append(list(e))
        self.face_ids[key] = len(self.out["faces"])
        self.out["faces"].append(list(cyc))
        return self.face_ids[key]

    def add_cell(self, faces, shell, truncated):
        self.out["cells"].append(faces)
        self.out["cell_shell"].append(shell)
        self.out["cell_truncated"].append(truncated)


def _order_around(pts, normal):
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    u = pts[0] - c
    u = u / max(np.linalg.norm(u), 1e-300)
    v = np.cross(normal, u)
    ang = np.arctan2((pts - c) @ v, (pts - c) @ u)
    return [pts[i] for i in np.argsort(ang, kind="stable")]


# ---------------------------------------------------------------- Coxeter realisation and orbit engine

class _Coxeter:
    def __init__(self, p, q, r):
        g = np.eye(4)
        for i, m in enumerate((p, q, r)):
            g[i, i + 1] = g[i + 1, i] = -math.cos(math.pi / m)
        lam, qm = np.linalg.eigh(g)
        order = np.argsort(lam)  # a negative eigenvalue (H³) comes first = time coordinate
        self.lorentz = lam[order[0]] < -1e-12
        self.eta = np.array([-1.0, 1, 1, 1]) if self.lorentz else np.ones(4)
        normals = qm[:, order] * np.sqrt(np.abs(lam[order]))
        self._set(normals)
        if self.lorentz:
            c3 = self.w[3]
            centre = c3 if self.form(c3, c3) < -1e-9 else self.w[2]
            self._set(normals @ _boost(centre).T)

    def form(self, x, y):
        return float(np.sum(self.eta * x * y))

    def _set(self, normals):
        self.normals = normals
        self.refl = [np.eye(4) - 2 * np.outer(n, self.eta * n) for n in normals]
        w = np.linalg.inv(normals * self.eta).T  # rows w_j with <w_j, n_k> = delta_jk
        for j in range(4):
            n2 = self.form(w[j], w[j])
            if self.lorentz:
                s = -1.0 if w[j][0] < 0 else 1.0
                w[j] = w[j] * (s / math.sqrt(-n2) if n2 < -1e-9 * w[j][0] ** 2 else 1.0 / w[j][0])
            else:
                w[j] = w[j] / math.sqrt(n2)
        if not self.lorentz:
            for j in range(1, 4):
                if self.form(w[j], w[0]) < 0:
                    w[j] = -w[j]
        self.w = w

    def key(self, x):
        return x[1:] / x[0] if self.lorentz else x / math.sqrt(self.form(x, x))


def _boost(u):
    u = u / math.sqrt(u[0] ** 2 - u[1:] @ u[1:])
    b = np.eye(4)
    b[0, 0] = u[0]
    b[0, 1:] = b[1:, 0] = -u[1:]
    b[1:, 1:] += np.outer(u[1:], u[1:]) / (1 + u[0])
    return b


def _orbit_complex(cox, p, shells, patch_layers, infinite, ideal, name, regime):
    r = cox.refl
    rot = r[0] @ r[1]
    steps = [np.linalg.matrix_power(rot, k) @ r[2] for k in range(p)]
    fidx, patch, queue = _Index(), [np.eye(4)], deque([(np.eye(4), 0)])
    fidx.insert(cox.key(cox.w[2]))
    while queue:
        g, depth = queue.popleft()
        if infinite and depth >= patch_layers:
            continue
        for s in steps:
            h = g @ s
            if fidx.insert(cox.key(h @ cox.w[2]))[1]:
                patch.append(h)
                queue.append((h, depth + 1))
                if len(patch) > 2000:
                    raise ValueError("cell patch did not close")
    face0 = [np.linalg.matrix_power(rot, k) @ cox.w[0] for k in range(p)]
    b, vidx, cidx = _Builder(name, regime), _Index(), _Index()
    cidx.insert(cox.key(cox.w[3]))
    queue = deque([(np.eye(4), 0)])
    while queue:
        m, shell = queue.popleft()
        faces = []
        for g in patch:
            mg = m @ g
            cyc = []
            for v in face0:
                x = mg @ v
                vid, new = vidx.insert(cox.key(x))
                if new:
                    _push_vertex(b.out, cox, x, ideal)
                cyc.append(vid)
            faces.append(b.add_face(cyc))
            if shell < shells:
                nm = mg @ r[3]
                if cidx.insert(cox.key(nm @ cox.w[3]))[1]:
                    queue.append((nm, shell + 1))
                    if len(cidx.pts) > 50000:
                        raise ValueError("too many cells; reduce shells")
        b.add_cell(faces, shell, infinite)
    return b.out


def _push_vertex(out, cox, x, ideal):
    if not cox.lorentz:
        out["coords4"].append(list(x / math.sqrt(x @ x)))
        out["points"].append([0.0, 0.0, 0.0])
        out["is_ideal"].append(False)
    elif ideal:
        s = x[1:] / x[0]
        s = s / np.linalg.norm(s)
        out["coords4"].append([1.0, *s])
        out["points"].append(list(s))
        out["klein"].append(list(s))
        out["is_ideal"].append(True)
    else:
        y = x / math.sqrt(-cox.form(x, x))
        out["coords4"].append(list(y))
        out["points"].append(list(y[1:] / (1 + y[0])))
        out["klein"].append(list(y[1:] / y[0]))
        out["is_ideal"].append(False)


def build_honeycomb(p: int, q: int, r: int, shells: int = 1, patch_layers: int = 1) -> dict[str, Any]:
    """Polychoron (S³, whole), cubic patch (E³) or H³ patch of {p,q,r}; ValueError with the [PROVED] reason otherwise.

    `shells` (0..=2 in H³) face-adjacency shells of cells; infinite `{4,4}`/`{6,3}` cells keep the faces
    within `patch_layers` (1..=2) edge-adjacency rings of their seed face."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "honeycomb3d_build_json"):
        return json.loads(rs.honeycomb3d_build_json(p, q, r, shells, patch_layers))
    cls = _py_classify(p, q, r)
    reg = cls["regime"]
    if reg == "SphericalS3":
        cox = _Coxeter(p, q, r)
        cx = _orbit_complex(cox, p, 10 ** 9, 0, False, False, SPHERICAL[(p, q, r)], reg)
        c4 = np.array(cx["coords4"])
        for w in (cox.w[3], cox.w[2], cox.w[1], np.array([0.3, -0.5, 0.7, 0.4])):
            pole = -w / np.linalg.norm(w)
            if np.all(c4 @ pole < 1 - 1e-3):
                break
        basis = []
        for e in np.eye(4):
            v = e - (e @ pole) * pole - sum((e @ bb) * bb for bb in basis)
            if v @ v > 1e-6 and len(basis) < 3:
                basis.append(v / np.linalg.norm(v))
        cx["points"] = (c4 @ np.array(basis).T / (1 - c4 @ pole)[:, None]).tolist()
        return cx
    if reg == "EuclideanE3":
        return build_euclidean_honeycomb("cubic", shells)
    if reg not in ("HyperbolicCompactH3", "HyperbolicParacompactH3"):
        raise ValueError(cls["reason"])
    if shells > 2:
        raise ValueError(f"shells = {shells} exceeds the maximum 2")
    inf = cls["has_infinite_euclidean_cells"]
    if inf and not 1 <= patch_layers <= 2:
        raise ValueError(f"patch_layers = {patch_layers} must be in 1..=2 for infinite cells")
    return _orbit_complex(_Coxeter(p, q, r), p, shells, patch_layers, inf, cls["has_ideal_vertices"], f"{{{p},{q},{r}}}", reg)


# ---------------------------------------------------------------- Voronoi cells and Euclidean honeycombs

_BASES = {"SC": np.eye(3), "FCC": np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]),
          "BCC": np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])}


def voronoi_cell_3d(origin, neighbours) -> dict[str, Any]:
    """Voronoi cell of `origin` against `neighbours` (perpendicular-bisector half-space clipping)."""
    out = _call("honeycomb3d_voronoi_cell_json", list(map(float, origin)), [list(map(float, x)) for x in neighbours])
    return out if out is not None else _py_voronoi(np.asarray(origin, float), np.asarray(neighbours, float))


def lattice_voronoi_cell(lattice: str) -> dict[str, Any]:
    """Wigner–Seitz cell of 'SC' (cube), 'FCC' (rhombic dodecahedron) or 'BCC' (truncated octahedron)."""
    out = _call("honeycomb3d_lattice_voronoi_json", lattice)
    if out is not None:
        return out
    if lattice not in _BASES:
        raise ValueError(f"unknown lattice {lattice!r} (SC, FCC, BCC)")
    return _py_voronoi(np.zeros(3), _lattice_neighbours(_BASES[lattice]))


def _lattice_neighbours(basis):
    return np.array([np.array(n) @ basis for n in itertools.product(range(-2, 3), repeat=3) if any(n)], float)


def _py_voronoi(origin, neighbours):
    rel = [d for d in neighbours - origin if np.linalg.norm(d) > 1e-12]
    if not rel:
        raise ValueError("no neighbours")
    rel.sort(key=np.linalg.norm)
    big = 2 * np.linalg.norm(rel[-1])
    eps = 1e-10 * big
    polys = []
    for axis, s in itertools.product(range(3), (-1.0, 1.0)):
        n = np.zeros(3)
        n[axis] = s
        u, w = (axis + 1) % 3, (axis + 2) % 3
        pts = []
        for a, bb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x = np.zeros(3)
            x[axis], x[u], x[w] = s * big, a * big, bb * big
            pts.append(x)
        polys.append((_order_around(pts, n), None))
    for i, d in enumerate(rel):
        h = 0.5 * d @ d
        if all(d @ x - h <= eps for poly, _ in polys for x in poly):
            continue
        new, cap = [], _Index(1e-9)
        for poly, tag in polys:
            kept = []
            for a, bb in zip(poly, poly[1:] + poly[:1]):
                da, db = d @ a - h, d @ bb - h
                if da <= eps:
                    kept.append(a)
                    if abs(da) <= eps:
                        cap.insert(a)
                if (da <= eps) != (db <= eps) and abs(da) > eps and abs(db) > eps:
                    x = a + da / (da - db) * (bb - a)
                    kept.append(x)
                    cap.insert(x)
            if len(kept) >= 3:
                new.append((kept, tag))
        if len(cap.pts) >= 3:
            new.append((_order_around(cap.pts, d), i))
        polys = new
    if any(tag is None for _, tag in polys):
        raise ValueError("neighbours do not enclose the origin (unbounded cell)")
    idx, faces, fn, edges = _Index(1e-7), [], [], set()
    for poly, tag in polys:
        area = 0.5 * np.linalg.norm(sum(np.cross(a, bb) for a, bb in zip(poly, poly[1:] + poly[:1])))
        if area < 1e-9:
            continue
        cyc = []
        for x in poly:
            j = idx.insert(x)[0]
            if not cyc or (j != cyc[-1] and j != cyc[0]):
                cyc.append(j)
        if len(cyc) < 3:
            continue
        edges |= {(min(a, bb), max(a, bb)) for a, bb in zip(cyc, cyc[1:] + cyc[:1])}
        faces.append(cyc)
        fn.append(list(map(float, rel[tag])))
    return dict(vertices=[list(np.array(x) + origin) for x in idx.pts], faces=faces, face_neighbours=fn,
                edges=[list(e) for e in sorted(edges)])


def build_euclidean_honeycomb(kind: str = "cubic", shells: int = 1) -> dict[str, Any]:
    """'cubic', 'bitruncated_cubic', 'rhombic_dodecahedral' or 'alternated_cubic' patch (`shells` 0..=3)."""
    out = _call("honeycomb3d_euclidean_json", kind, shells)
    if out is not None:
        return out
    if kind not in EUCLIDEAN_KINDS:
        raise ValueError(f"unknown Euclidean honeycomb {kind!r} ({', '.join(EUCLIDEAN_KINDS)})")
    if shells > 3:
        raise ValueError(f"shells = {shells} exceeds the maximum 3")
    lat, label = EUCLIDEAN_KINDS[kind]
    k = shells + 1
    raw, centre = [], 0
    if lat is not None:
        cell = lattice_voronoi_cell(lat)
        vs = np.array(cell["vertices"])
        for n in itertools.product(range(-k, k + 1), repeat=3):
            if not any(n):
                centre = len(raw)
            t = np.array(n, float) @ _BASES[lat]
            raw.append([[vs[v] + t for v in f] for f in cell["faces"]])
    else:
        for o in itertools.product(range(-k, k + 1), repeat=3):
            o = np.array(o, float)
            if int(o.sum()) % 2 == 0:
                if not o.any():
                    centre = len(raw)
                raw.append([[o + [sx, 0, 0], o + [0, sy, 0], o + [0, 0, sz]] for sx, sy, sz in itertools.product((-1, 1), repeat=3)])
            corners = [o + 0.5 + 0.5 * np.array(s) for s in itertools.product((-1, 1), repeat=3)]
            corners = [x for x in corners if round(x.sum()) % 2 == 1]
            raw.append([[corners[i] for i in range(4) if i != skip] for skip in range(4)])
    idx = _Index()
    ids = [[[idx.insert(x)[0] for x in f] for f in c] for c in raw]
    face_cells = {}
    for ci, c in enumerate(ids):
        for f in c:
            face_cells.setdefault(tuple(sorted(f)), []).append(ci)
    shell_of, order, queue = {centre: 0}, [centre], deque([centre])
    while queue:
        ci = queue.popleft()
        if shell_of[ci] >= shells:
            continue
        for f in ids[ci]:
            for cj in face_cells[tuple(sorted(f))]:
                if cj not in shell_of:
                    shell_of[cj] = shell_of[ci] + 1
                    order.append(cj)
                    queue.append(cj)
    b, remap = _Builder(label, "EuclideanE3"), {}
    for ci in order:
        faces = []
        for f in ids[ci]:
            cyc = []
            for g in f:
                if g not in remap:
                    remap[g] = len(b.out["points"])
                    b.out["points"].append(list(idx.pts[g]))
                    b.out["coords4"].append([*idx.pts[g], 1.0])
                    b.out["is_ideal"].append(False)
                cyc.append(remap[g])
            faces.append(b.add_face(cyc))
        b.add_cell(faces, shell_of[ci], False)
    return b.out


from ._honeycomb3d_extra import dirichlet_manifold, nodal, nodal_surface_mesh, petrie_coxeter_surface, tpms_net  # noqa: E402
