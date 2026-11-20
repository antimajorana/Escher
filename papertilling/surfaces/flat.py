"""Flat surfaces: a lattice on a cylinder, a torus, a Moebius band or a Klein bottle.

WHY THESE FOUR. A surface carries a periodic lattice with no defect anywhere exactly when it is flat, and the complete flat
surfaces are the plane, the cylinder, the torus, the Moebius band and the Klein bottle (cited result: classical). Each is the
plane divided by a group Gamma of symmetries of the lattice that moves every point (no fixed points):
    torus          Gamma = two translations T1, T2 of the lattice              (closed, orientable,      chi = 0)
    cylinder       Gamma = one translation C (the "wrapping vector"), cut to a finite length  (two open ends)
    Klein bottle   Gamma = a glide g of the lattice and a translation T across its axis (closed, non-orientable, chi = 0)
    Moebius band   Gamma = a glide g, cut to a finite width around its axis    (one open edge)
Because Gamma consists of symmetries of the lattice, every site keeps exactly its neighbourhood in the plane (same bonds,
lengths, angles): the lattice is undistorted (proved: an isometry that maps the lattice to itself maps every bond to a bond).
A glide reverses orientation, so the Moebius band and the Klein bottle exist only for groups with a mirror or a glide line
(pm pg cm pmm pmg pgg cmm p4m p4g p3m1 p31m p6m), never for p1 p2 p3 p4 p6 (SurfaceImpossible, proof in refusal.py).

HOW. Everything is done on the torus R^2 / Lambda, Lambda = a sublattice of the lattice's translations given by an integer
matrix N (rows = its two vectors in units of the primitive cell): sites (i, n) with n mod N. A glide g with g^2 in Lambda acts
on that torus as an involution without fixed points; the Klein bottle and the Moebius band are its orbits. The cylinder and the
Moebius band keep only the sites of a strip and the bonds inside it.

SIZES AND PICTURES. With no size given, each surface is built with about SITES = 220 sites, in proportions that make its 3D
picture readable. The 3D shape is a map of the plane (Surface.xyz_map) invariant under Gamma, so bonds and faces are drawn as
curves on it: the cylinder is exact; the torus doughnut, the twisted Moebius strip and the figure-8 Klein bottle stretch the
bonds, and xyz_note gives the measured range of 3D bond length / flat bond length.

CHECKS (Surface.checks): no bond from a site to itself, no two bonds between the same two sites, every site away from an
open edge has its plane number of bonds, V - E + F = chi (faces traced from the plane's faces).
"""
from __future__ import annotations

import math

import numpy as np

from ..dsymbol import _darts, _faces, _rotation
from ..legacy.wallpaper_lattices import _apply_to_site
from .base import Surface
from ._glides import glide_power
from .refusal import SurfaceImpossible, SurfaceNotFound, no_glide_reason, no_reversing_reason, symmetry_facts


# ----------------------------------------------------------------------------------------------------------------
# the lattice data
# ----------------------------------------------------------------------------------------------------------------
def _cell_and_ops(lattice):
    from ..lattice import Lattice, ops_in_cell
    from ..sheet import named
    L = named(lattice) if isinstance(lattice, str) else lattice
    if not isinstance(L, Lattice):
        raise TypeError("give a papertilling Lattice or the name of a named lattice")
    cell = L.cell()
    A = cell.A
    F = np.asarray(cell.basis, float) @ np.linalg.inv(A).T
    bonds = [tuple(int(x) for x in b[:4]) for b in cell.bonds]
    return L, cell, A, F, bonds, ops_in_cell(L.family)


def glides(lattice):
    """The mirror and glide classes of the decorated lattice usable for a Klein bottle or a Moebius band, one per linear
    part: dicts M (fractional), t, tau = g^2 (integer), along / across (shortest lattice vectors on / across the axis),
    half = tau/2 (Cartesian), length, mirror (a mirror class: t is the mirror followed by `along`). Found by the complete
    search of _glides.py on the cell (bond kinds kept), not from the group label: empty exactly when the lattice has no
    mirror or glide line [PROVED]."""
    L, cell, A, F, bonds, ops = _cell_and_ops(lattice)
    from ._glides import glide_search
    return glide_search(A, cell.basis, cell.bonds)


# ----------------------------------------------------------------------------------------------------------------
# the torus R^2 / Lambda and its quotients
# ----------------------------------------------------------------------------------------------------------------
class _Torus:
    def __init__(self, L, A, F, bonds, N):
        self.N = np.asarray(N, int)
        if round(abs(np.linalg.det(self.N))) == 0:
            raise ValueError("the two vectors must not be parallel")
        self.Ninv = np.linalg.inv(self.N.astype(float))
        self.A, self.F, self.bonds, self.m = A, F, bonds, len(F)
        k = int(round(abs(np.linalg.det(self.N))))
        frac = []                                                  # cells n of the fundamental parallelogram
        lim = int(np.abs(self.N).sum()) + 1
        for a in range(-lim, lim + 1):
            for b in range(-lim, lim + 1):
                f = np.array([a, b]) @ self.Ninv
                if np.all(f >= -1e-9) and np.all(f < 1 - 1e-9):
                    frac.append((a, b))
        assert len(frac) == k, (len(frac), k)
        self.cells = frac
        self.index = {(i, a, b): s for s, (i, (a, b)) in enumerate((i, c) for c in frac for i in range(self.m))}
        self.labels = [(i, a, b) for (a, b) in frac for i in range(self.m)]

    def reduce(self, n):
        f = np.asarray(n, float) @ self.Ninv
        f = f - np.floor(f + 1e-9)
        r = np.rint(f @ self.N).astype(int)
        return int(r[0]), int(r[1])

    def site(self, i, n):
        return self.index[(i,) + self.reduce(n)]


def _plane_faces(cell, F):
    """Faces of the plane lattice: list of dart cycles; each dart (i, j, n1, n2)."""
    darts, index, rev = _darts(cell)[1:]
    nxt, prv, deg = _rotation(cell, F, darts)
    faces, _ = _faces(darts, rev, prv)
    return [[darts[d] for d in cyc] for cyc in faces], deg


def _canon(cyc):
    k = len(cyc)
    rots = [tuple(cyc[s:] + cyc[:s]) for s in range(k)]
    r = cyc[::-1]
    rots += [tuple(r[s:] + r[:s]) for s in range(k)]
    return min(rots)


def _build(kind, L, cell, A, F, bonds, N, glide=None, keep=None, rep=None):
    """Sites of the torus R^2/N (optionally divided by the glide involution), bonds, faces. keep(i, n) -> bool keeps a strip
    (open surfaces); rep(i, n) chooses the representative of a glide orbit."""
    T = _Torus(L, A, F, bonds, N)
    orbit = list(range(len(T.labels)))
    if glide is not None:
        M, t = glide["M"], glide["t"]
        for s, (i, a, b) in enumerate(T.labels):
            j, c = _apply_to_site(M, t, F, i)
            n2 = M @ np.array([a, b]) + c
            s2 = T.site(j, n2)
            if s2 == s:
                raise SurfaceNotFound("the glide fixes a site on this torus (a site would be glued to itself)",
                                      hint="choose other sizes")
            orbit[s] = min(s, s2)
    kept = [s for s in range(len(T.labels)) if keep is None or keep(*T.labels[s])]
    keptset = set(kept)
    classes = sorted({orbit[s] for s in kept})
    cid = {c: k for k, c in enumerate(classes)}
    # representative of each class
    reps = {}
    for s in kept:
        c = orbit[s]
        if c not in reps or (rep is not None and rep(*T.labels[s]) and not rep(*T.labels[reps[c]])):
            reps[c] = s
    sites, uv = [], []
    for c in classes:
        i, a, b = T.labels[reps[c]]
        sites.append((i, a, b))
        uv.append(cell.basis[i] + A @ np.array([a, b]))
    # bonds: a bond of the surface is an orbit of torus bonds under the glide; keep it once, drawn from a representative
    Mc = None if glide is None else A @ glide["M"] @ np.linalg.inv(A)

    def gimage(i, n):
        j, c = _apply_to_site(glide["M"], glide["t"], F, i)
        return j, glide["M"] @ np.asarray(n) + c

    out, seen = [], set()
    for s in kept:
        i, a, b = T.labels[s]
        n0 = np.array([a, b])
        for (p, q, n1, n2) in bonds:
            for (x, y, d) in ((p, q, np.array([n1, n2])), (q, p, -np.array([n1, n2]))):
                if x != i or (x == y and np.all(d == 0)):
                    continue
                n_end = n0 + d
                if keep is not None and not keep(y, *n_end):
                    continue
                s2 = T.site(y, n_end)
                if s2 not in keptset:
                    continue
                keys = [tuple(sorted((s, s2)))]
                if glide is not None:
                    keys.append(tuple(sorted((T.site(*gimage(i, n0)), T.site(*gimage(y, n_end))))))
                key = min(keys)
                if key in seen:
                    continue
                seen.add(key)
                v = cell.basis[y] + A @ n_end - (cell.basis[i] + A @ n0)
                if s == reps[orbit[s]]:
                    out.append((cid[orbit[s]], cid[orbit[s2]], v))
                elif s2 == reps[orbit[s2]]:
                    out.append((cid[orbit[s2]], cid[orbit[s]], -v))
                else:                                      # neither end is a representative: draw the glide image
                    out.append((cid[orbit[s]], cid[orbit[s2]], Mc @ v))
    # faces from the plane's faces placed at every kept cell
    pf, deg = _plane_faces(cell, F)
    bondset = {tuple(sorted((a, b))) for a, b, _ in out}
    faces, fseen = [], set()
    for s in kept:
        i, a, b = T.labels[s]
        for cyc in pf:
            if cyc[0][0] != i:
                continue
            n = np.array([a, b])
            ids, ok = [], True
            for (x, y, n1, n2) in cyc:
                if keep is not None and not keep(x, *n):
                    ok = False
                    break
                sx = T.site(x, n)
                if sx not in keptset:
                    ok = False
                    break
                ids.append(cid[orbit[sx]])
                n = n + np.array([n1, n2])
            if not ok:
                continue
            if any(tuple(sorted((ids[k], ids[(k + 1) % len(ids)]))) not in bondset for k in range(len(ids))):
                continue
            c = _canon(ids)
            if c not in fseen:
                fseen.add(c)
                faces.append(ids)
    edge = set()
    if keep is not None:
        dkept = np.zeros(len(classes), int)
        for a_, b_, _ in out:
            dkept[a_] += 1
            dkept[b_] += 1
        edge = {k for k in range(len(classes)) if dkept[k] != deg[sites[k][0]]}
    return sites, np.array(uv), out, faces, {int(i): int(d) for i, d in deg.items()}, edge, T


# ----------------------------------------------------------------------------------------------------------------
# the four surfaces
# ----------------------------------------------------------------------------------------------------------------
SITES = 220          # default number of sites aimed at when no size is given (notebook pictures: about 100-400)


def _site_area(cell, A):
    return abs(float(np.linalg.det(A))) / len(cell.basis)


def _odd(x):
    n = max(1, int(round(x)))
    return n if n % 2 else n + 1


def _note(S, what):
    lo, hi = S.stretch()
    return what.format(lo=lo, hi=hi)


def torus(lattice, T1=None, T2=None, sites=SITES):
    """The lattice on a torus: the plane with points T1 apart and T2 apart identified. T1, T2: integer vectors in units of
    the primitive cell vectors (a1, a2) of the lattice (L.cell()); when not given, T1 = n1 a1 and T2 = n2 T (T the lattice
    vector most perpendicular to a1), with about `sites` sites and |T1| about 2.2 |T2| (a doughnut that reads as one).
    Exact (flat); the 3D doughnut stretches the bonds."""
    L, cell, A, F, bonds, ops = _cell_and_ops(lattice)
    if T1 is None or T2 is None:
        Tc = _complement(A, (1, 0), 0)
        a1, tc = float(np.linalg.norm(A[:, 0])), float(np.linalg.norm(A @ Tc))
        n2 = max(2, int(round(math.sqrt(sites * a1 / (len(cell.basis) * 2.2 * tc)))))
        n1 = max(3, int(round(2.2 * n2 * tc / a1)))
        T1, T2 = (n1, 0), tuple(int(x) for x in n2 * Tc)
    N = np.array([T1, T2], int)
    sites_, uv, out, faces, deg, edge, T = _build("torus", L, cell, A, F, bonds, N)
    V1, V2 = A @ np.array(T1, float), A @ np.array(T2, float)
    o = np.zeros(2)
    dom = dict(polygon=[o, V1, V1 + V2, V2],
               edges=[(o, V1, "1", "#d9432f"), (V2, V1 + V2, "1", "#d9432f"), (o, V2, "2", "#2e7d32"), (V1, V1 + V2, "2", "#2e7d32")])
    Minv = np.linalg.inv(np.column_stack([V1, V2]))
    f = uv @ Minv.T                                                    # representatives inside the parallelogram
    uv = (f - np.floor(f + 1e-7)) @ np.column_stack([V1, V2]).T
    S = Surface("torus", L, sites_, uv, out, faces, 0, True, True, dom, dict(T1=tuple(T1), T2=tuple(T2)), deg)
    big = 0 if np.linalg.norm(V1) >= np.linalg.norm(V2) else 1           # the longer vector goes round the hole
    R, r = max(np.linalg.norm(V1), np.linalg.norm(V2)) / (2 * np.pi), min(np.linalg.norm(V1), np.linalg.norm(V2)) / (2 * np.pi)
    R = max(R, 1.8 * r)

    def xyz_map(p):
        f = np.asarray(p, float) @ Minv.T
        u, v = 2 * np.pi * f[:, big], 2 * np.pi * f[:, 1 - big]
        return np.column_stack([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)])

    S.xyz_map, S.xyz, S.view = xyz_map, xyz_map(S.uv), (38, -60)
    S.xyz_note = _note(S, "doughnut: bonds {lo:.2f}-{hi:.2f} x flat length (no flat torus fits in 3D undistorted)")
    return S


def _complement(A, C, length):
    """A lattice vector T with (C, T) a basis of a sublattice containing C as a primitive vector, as short as possible,
    as perpendicular to C as possible."""
    C = np.array(C, int)
    g = math.gcd(int(C[0]), int(C[1]))
    best = None
    for a in range(-6, 7):
        for b in range(-6, 7):
            if a * C[1] - b * C[0] == 0:
                continue
            if abs(a * C[1] - b * C[0]) != g:                     # (C/g, T) is a basis of the lattice
                continue
            v = A @ np.array([a, b])
            cosang = abs(np.dot(v, A @ C)) / (np.linalg.norm(v) * np.linalg.norm(A @ C))
            key = (round(cosang, 6), float(np.linalg.norm(v)))
            if best is None or key < best[0]:
                best = (key, np.array([a, b]))
    return best[1]


def cylinder(lattice, wrap=None, length=None, sites=SITES):
    """The lattice rolled into a cylinder: points `wrap` apart (integer vector in units of a1, a2; like the (n, m) of a
    nanotube) are identified; `length` rows of cells along the axis, with open ends. When not given, wrap = n a1 with n
    chosen so that about `sites` sites make a cylinder about 1.2 diameters long, and length = sites / (sites per row).
    Exact; the 3D shape is isometric."""
    L, cell, A, F, bonds, ops = _cell_and_ops(lattice)
    m = len(cell.basis)
    if wrap is None:
        Tc = _complement(A, (1, 0), 0)
        a1 = float(np.linalg.norm(A[:, 0]))
        hc = _site_area(cell, A) * m / a1                                # height of one row of cells
        wrap = (max(3, int(round(math.sqrt(sites * hc / (0.38 * m * a1))))), 0)
    C = np.array(wrap, int)
    from ._flat_bc import wrap_complement
    Tv = wrap_complement(A, C)                                     # exact (extended Euclid), any (n, m)
    if length is None:
        per_row = abs(int(round(np.linalg.det(np.array([C, Tv], float)))))
        length = max(2, int(round(sites / (m * per_row))))
    Cc = A @ C
    e1 = Cc / np.linalg.norm(Cc)
    e2 = np.array([-e1[1], e1[0]])
    if np.dot(A @ Tv, e2) < 0:                                     # rows go up the axis
        Tv = -Tv
    N = np.array([C, (length + 2) * Tv], int)
    h = abs(np.dot(A @ Tv, e2))

    def height(i, a, b):
        return float(np.dot(cell.basis[i] + A @ np.array([a, b]), e2))

    def keep(i, a, b):
        z = height(i, a, b)
        return -1e-9 <= z < length * h - 1e-9

    sites, uv, out, faces, deg, edge, T = _build("cylinder", L, cell, A, F, bonds, N, keep=keep)
    # representatives: unroll around the circumference into [0, |C|)
    w = np.linalg.norm(Cc)
    x = (uv @ e1) % w
    z = uv @ e2
    uv2 = np.column_stack([x, z]) @ np.array([e1, e2])
    o = np.zeros(2)
    dom = dict(polygon=[o, Cc, Cc + e2 * length * h, e2 * length * h],
               edges=[(o, e2 * length * h, "1", "#d9432f"), (Cc, Cc + e2 * length * h, "1", "#d9432f")])
    # bond vectors were measured from the old representatives; they do not change under the unrolling
    S = Surface("cylinder", L, sites, uv2, out, faces, 0, True, False, dom, dict(wrap=tuple(wrap), length=length), deg, edge)
    rad = w / (2 * np.pi)

    def xyz_map(p):
        p = np.asarray(p, float)
        th = 2 * np.pi * (p @ e1) / w
        return np.column_stack([rad * np.cos(th), rad * np.sin(th), p @ e2])

    S.xyz_map, S.xyz, S.view = xyz_map, xyz_map(S.uv), (18, -60)
    S.xyz_note = "exact: rolling a sheet into a cylinder keeps every length and angle"
    return S


def _axis_point(A, g):
    """A point on the glide axis (Cartesian): g(p) = p + tau/2."""
    M = A @ g["M"] @ np.linalg.inv(A)
    t = A @ g["t"]
    p, *_ = np.linalg.lstsq(M - np.eye(2), g["half"] - t, rcond=None)
    return p


def _bond_scale(cell, A):
    return float(np.median([np.linalg.norm(cell.basis[j] + A @ np.array([n1, n2]) - cell.basis[i]) for i, j, n1, n2, *_ in cell.bonds]))


def _glide_setup(lattice, which, along, auto=None, what="Moebius band or a Klein bottle"):
    """The glide g' (g followed by (along - 1)/2 translations tau) that closes the band; auto(cell, A, g) -> along when
    along is None (default: about 8 bond lengths around). No glide: SurfaceImpossible when the lattice has no
    orientation-reversing symmetry at all (refusal.py, proof (a)), else SurfaceNotFound."""
    L, cell, A, F, bonds, ops = _cell_and_ops(lattice)
    gl = glides(L)
    if not gl:                                                     # the search is complete: a proof (_glides.py)
        if not symmetry_facts(L)["reversing"]:                     # no reflection even without bond kinds: proof (a)
            raise SurfaceImpossible(no_reversing_reason(L.group, what), "so the lattice cannot sit on any non-orientable "
                                    "surface")
        raise SurfaceImpossible(no_glide_reason(what), "so the lattice cannot sit on any non-orientable surface",
                                scope="lattice")
    g = dict(gl[which % len(gl)])
    if along is None and auto is not None:
        along = auto(cell, A, g)
    if along is None:                                              # about 8 bond lengths around
        b0 = _bond_scale(cell, A)
        along = max(1, int(np.ceil(8 * b0 / np.linalg.norm(A @ g["tau"]))))
        along += 1 - along % 2
    g = glide_power(g, along)                                      # even lengths only for a mirror class
    g["half"] = A @ (along * g["tau"]) / 2.0
    g["along"] = along
    return L, cell, A, F, bonds, g


# the figure-8 cross-section (sin v, sin 2v): its perimeter and an arclength table, so that bonds round the tube keep about
# the same length (the reparametrisation is odd in v, so the glide's v -> -v is kept)
_V8 = np.linspace(0.0, 2 * np.pi, 2001)
_S8 = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(np.sin(_V8)), np.diff(np.sin(2 * _V8))))])
_P8 = float(_S8[-1])
_S8 = 2 * np.pi * _S8 / _P8
_K8 = 2.0                                                          # hole radius / cross-section size of the figure 8


def _arc8(x):
    """The figure-8 parameter v with arclength fraction x / 2 pi (x any real; v - x is 2 pi periodic)."""
    k = np.floor(x / (2 * np.pi))
    return np.interp(x - 2 * np.pi * k, _S8, _V8) + 2 * np.pi * k


def klein_bottle(lattice, along=None, across=None, which=0, sites=SITES):
    """The lattice on a Klein bottle: the plane divided by a glide g of the lattice (repeated `along` times: odd, or any number
    for a mirror class) and by
    `across` times the shortest translation T across its axis. Exact (flat). Needs a mirror or glide line (`which` picks
    one class, see glides(L)). When not given, along and across give about `sites` sites in proportions that suit the 3D
    picture, the figure-8 immersion (which passes through itself, as any Klein bottle in 3D must)."""
    kappa = 2 * np.pi * _K8 / _P8                                  # (length round the hole) / (length round the tube)
    size = {}

    def auto(cell, A, g):
        ac, tau = float(np.linalg.norm(A @ g["across"])), float(np.linalg.norm(A @ g["tau"]))
        n = across if across is not None else max(2, int(round(math.sqrt(sites * _site_area(cell, A) / kappa) / ac)))
        size["across"] = n
        return _odd(2 * kappa * n * ac / tau)

    L, cell, A, F, bonds, g = _glide_setup(lattice, which, along, auto, what="Klein bottle")
    along = g["along"]
    if across is None:
        across = size.get("across") or max(2, int(np.ceil(6 * _bond_scale(cell, A) / np.linalg.norm(A @ g["across"]))))
    N = np.array([along * g["tau"], across * g["across"]], int)
    p0 = _axis_point(A, g)
    U = np.column_stack([A @ (along * g["tau"]), A @ (across * g["across"])])
    Uinv = np.linalg.inv(U)

    def coords(i, a, b):
        return Uinv @ (cell.basis[i] + A @ np.array([a, b]) - p0)

    def rep(i, a, b):
        return ((coords(i, a, b)[0] + 1e-7) % 1.0) < 0.5 - 1e-7

    try:
        sites_, uv, out, faces, deg, edge, T = _build("Klein bottle", L, cell, A, F, bonds, N, glide=g, rep=rep)
    except SurfaceNotFound as e:
        raise SurfaceNotFound(e.reason, f"Klein bottle of {L.group} with along={along}, across={across}",
                              hint=f"try along={along + 2} or across={across + 1}") from None
    f = (uv - p0) @ Uinv.T
    f[:, 0] = (f[:, 0] + 1e-7) % 1.0 - 1e-7
    f[:, 1] = (f[:, 1] + 0.5) % 1.0 - 0.5
    o = p0
    a_half, b_full = U[:, 0] / 2, U[:, 1]
    lo = o - b_full / 2
    dom = dict(polygon=[lo, lo + a_half, lo + a_half + b_full, lo + b_full],
               edges=[(lo, lo + a_half, "1", "#d9432f"), (lo + b_full, lo + a_half + b_full, "1", "#d9432f"),
                      (lo, lo + b_full, "2", "#2e7d32"), (lo + a_half + b_full, lo + a_half, "2", "#2e7d32")])
    uv2 = p0 + f @ U.T
    S = Surface("Klein bottle", L, sites_, uv2, out, faces, 0, False, True, dom, dict(along=along, across=across, glide=which), deg)
    a8 = np.linalg.norm(U[:, 1]) / _P8                             # tube: perimeter = |across x T|
    R = max(np.linalg.norm(U[:, 0]) / (4 * np.pi), _K8 * a8)        # hole: circumference = half the glide's period

    def xyz_map(p):
        q = (np.asarray(p, float) - p0) @ Uinv.T
        u, v = 4 * np.pi * q[:, 0], _arc8(2 * np.pi * q[:, 1])
        w = R + a8 * (np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v))
        return np.column_stack([w * np.cos(u), w * np.sin(u), a8 * (np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v))])

    S.xyz_map, S.xyz, S.view = xyz_map, xyz_map(S.uv), (55, -60)
    S.xyz_note = _note(S, "figure-8 immersion, passes through itself (as any Klein bottle in 3D); "
                          "bonds {lo:.2f}-{hi:.2f} x flat length")
    return S


def mobius_band(lattice, along=None, width=None, which=0, sites=SITES):
    """The lattice on a Moebius band: a strip of `width` rows on each side of a glide line, closed up after `along` glide
    steps (odd), so that it comes back mirrored. Exact (flat); one open edge. Needs a mirror or glide line. When not given,
    width and along give about `sites` sites on a band whose radius is 1.6-4 times its half-width (2.4 when possible)."""
    size = {}

    def auto(cell, A, g):
        ac, tau, sa = float(np.linalg.norm(A @ g["across"])), float(np.linalg.norm(A @ g["tau"])), _site_area(cell, A)
        wd = width if width is not None else max(1, int(round(math.sqrt(sites * sa / (4 * np.pi * 2.4)) / ac + 0.25)))
        size["width"] = wd
        h = wd * ac                                                # half-width of the band
        core = float(np.clip(sites * sa / (2 * h), 2 * np.pi * 1.6 * h, 2 * np.pi * 4 * h))
        return _odd(2 * core / tau)

    L, cell, A, F, bonds, g = _glide_setup(lattice, which, along, auto, what="Moebius band")
    along = g["along"]
    if width is None:
        width = size.get("width") or max(1, int(round(2.5 * _bond_scale(cell, A) / np.linalg.norm(A @ g["across"]))))
    across = 2 * width + 4
    N = np.array([along * g["tau"], across * g["across"]], int)
    p0 = _axis_point(A, g)
    U = np.column_stack([A @ (along * g["tau"]), A @ (across * g["across"])])
    Uinv = np.linalg.inv(U)
    hcell = 1.0 / across

    def coords(i, a, b):
        return Uinv @ (cell.basis[i] + A @ np.array([a, b]) - p0)

    def keep(i, a, b):
        y = coords(i, a, b)[1]
        y = (y + 0.5) % 1.0 - 0.5
        return abs(y) <= width * hcell + 1e-9

    def rep(i, a, b):
        return ((coords(i, a, b)[0] + 1e-7) % 1.0) < 0.5 - 1e-7

    try:
        sites_, uv, out, faces, deg, edge, T = _build("Moebius band", L, cell, A, F, bonds, N, glide=g, keep=keep, rep=rep)
    except SurfaceNotFound as e:
        raise SurfaceNotFound(e.reason, f"Moebius band of {L.group} with along={along}, width={width}",
                              hint=f"try along={along + 2} or width={width + 1}") from None
    f = (uv - p0) @ Uinv.T
    f[:, 0] = (f[:, 0] + 1e-7) % 1.0 - 1e-7
    f[:, 1] = (f[:, 1] + 0.5) % 1.0 - 0.5
    a_half = U[:, 0] / 2
    wv = U[:, 1] * width * hcell
    dom = dict(polygon=[p0 - wv, p0 - wv + a_half, p0 + wv + a_half, p0 + wv],
               edges=[(p0 - wv, p0 + wv, "1", "#d9432f"), (p0 + wv + a_half, p0 - wv + a_half, "1", "#d9432f")])
    S = Surface("Moebius band", L, sites_, p0 + f @ U.T, out, faces, 0, False, False, dom, dict(along=along, width=width, glide=which), deg, edge)
    half = width * np.linalg.norm(U[:, 1]) * hcell
    R = max(np.linalg.norm(U[:, 0]) / (4 * np.pi), 1.6 * half)     # centre line: its flat length when possible

    def xyz_map(p):
        q = (np.asarray(p, float) - p0) @ Uinv.T
        t = 4 * np.pi * q[:, 0]
        s_ = ((q[:, 1] + 0.5) % 1.0 - 0.5) * np.linalg.norm(U[:, 1])
        return np.column_stack([(R + s_ * np.cos(t / 2)) * np.cos(t), (R + s_ * np.cos(t / 2)) * np.sin(t), s_ * np.sin(t / 2)])

    S.xyz_map, S.xyz, S.view = xyz_map, xyz_map(S.uv), (35, -60)
    S.xyz_note = _note(S, "twisted strip, exact across the band; bonds {lo:.2f}-{hi:.2f} x flat length")
    return S
