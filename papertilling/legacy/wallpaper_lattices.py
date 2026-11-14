"""
wallpaper_lattices: finite patches of the 2D lattices in
"Lattices of the 17 Wallpaper groups", in the spirit of the hypertiling package.

    lat = WallpaperLattice("kagome", n=4)            # 4 layers of unit cells around the centre
    lat = WallpaperLattice("square", shape="parallelogram", L=(8, 8), boundary="periodic")
    lat.plot(); lat.get_nbrs_list(); lat.adjacency_matrix(); lat.rigidity_matrix()

Every lattice is a periodic graph: a Bravais lattice (a1, a2), a basis of sites, and a list of
bonds (i, j, n1, n2, type) meaning "site i of cell (0,0) is bonded to site j of cell (n1,n2)".
Bond type 0 is drawn blue and type 1 red, as in the figure.

identify_wallpaper_group(cell) computes the wallpaper group of the infinite decorated graph
(sites + coloured bonds) directly from its symmetry operations, independent of the label.

Lattices with a chosen wallpaper group (any of the 17, ITA standard settings):

    cell = symmetric_cell("p4g", sites=[(0.1, 0.27)], n_shells=2)   # orbits of sites and bonds
    cell = random_symmetric_cell("p31m", seed=1)                   # random, exactly p31m
    WallpaperLattice(cell, n=3).plot(symmetry=True)                # mirrors, glides, rotation centres
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

BLUE = "#1f6fc5"
RED = "#c8361f"
PALETTE = (BLUE, RED, "#2a9d4b", "#e09a1a", "#7b4fb0", "#1aa3a3", "#a0522d", "#d4419a")
S3 = math.sqrt(3.0)
TOL = 1e-6


# ----------------------------------------------------------------------------------------------
# Unit cells
# ----------------------------------------------------------------------------------------------
@dataclass
class UnitCell:
    key: str
    label: str
    group: str                      # the wallpaper group the construction is meant to have
    a1: np.ndarray
    a2: np.ndarray
    basis: np.ndarray               # (m, 2) Cartesian positions of the sites in cell (0,0)
    bonds: list                     # [(i, j, n1, n2, type)]
    note: str = ""
    style: dict = field(default_factory=dict)   # bond type -> dict(color=..., lw=...)
    patch: str = "auto"             # shape of the layers: "auto" | "hex" | "cells" | "box"
    species: object = None          # optional site labels (len = sites); symmetries must preserve them

    @property
    def A(self):
        return np.column_stack([self.a1, self.a2])

    def __repr__(self):
        return (f"UnitCell({self.key!r}, group={self.group}, sites/cell={len(self.basis)}, "
                f"bonds/cell={len(self.bonds)})")


def _canon_bond(i, j, n1, n2, t):
    a = (i, j, n1, n2, t)
    b = (j, i, -n1, -n2, t)
    return min(a, b)


def _locate(p, A, Ainv, basis, tol=1e-6):
    """Return (site, (n1, n2)) for a Cartesian point p, or raise."""
    for s, b in enumerate(basis):
        f = Ainv @ (np.asarray(p, float) - b)
        r = np.round(f)
        if np.all(np.abs(f - r) < tol):
            return s, (int(r[0]), int(r[1]))
    raise ValueError(f"point {p} is not a lattice site")


def cell_from_segments(key, label, group, a1, a2, basis, segments, note="", style=None, patch="auto"):
    """Build a UnitCell from Cartesian bond segments [(p, q, type)]; endpoints must be sites."""
    a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
    basis = np.asarray(basis, float)
    A = np.column_stack([a1, a2])
    Ainv = np.linalg.inv(A)
    bonds = set()
    for p, q, t in segments:
        i, np_ = _locate(p, A, Ainv, basis)
        j, nq = _locate(q, A, Ainv, basis)
        d = (nq[0] - np_[0], nq[1] - np_[1])
        if i == j and d == (0, 0):
            raise ValueError("zero-length bond")
        bonds.add(_canon_bond(i, j, d[0], d[1], t))
    return UnitCell(key, label, group, a1, a2, basis, sorted(bonds), note, style or {}, patch)


def _grid_segments(a1, a2, basis_pts, t=0):
    """Bonds along a1 and a2 from every listed point (for grid-like lattices)."""
    a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
    segs = []
    for p in basis_pts:
        p = np.asarray(p, float)
        segs += [(p, p + a1, t), (p, p + a2, t)]
    return segs


# --- the catalogue ----------------------------------------------------------------------------
def oblique(a2=(0.32, 0.34)):
    a1, a2 = (1.0, 0.0), np.asarray(a2, float)
    return cell_from_segments("oblique", "Oblique", "p2", a1, a2, [(0, 0)],
                              _grid_segments(a1, a2, [(0, 0)]))


def straight(a2=(0.32, 0.34)):
    """As drawn: oblique grid, red long diagonals a1+a2 in one cell of every 2x2 block."""
    a1 = np.array([1.0, 0.0]); a2 = np.asarray(a2, float)
    A1, A2 = 2 * a1, 2 * a2
    pts = [i * a1 + j * a2 for j in range(2) for i in range(2)]
    segs = _grid_segments(a1, a2, pts) + [(a2, 2 * a2 + a1, 1)]
    return cell_from_segments("straight", "Straight", "p1", A1, A2, pts, segs,
                              note="as drawn in the figure")


def straight_p1(a2=(0.32, 0.34)):
    """Oblique grid with red long diagonals in an L of three cells per 3x3 block."""
    a1 = np.array([1.0, 0.0]); a2 = np.asarray(a2, float)
    pts = [i * a1 + j * a2 for j in range(3) for i in range(3)]
    segs = _grid_segments(a1, a2, pts)
    for (i, j) in [(0, 0), (1, 0), (0, 1)]:
        p = i * a1 + j * a2
        segs.append((p, p + a1 + a2, 1))
    return cell_from_segments("straight_p1", "Straight (L-decorated)", "p1", 3 * a1, 3 * a2, pts, segs,
                              note="variant: three red diagonals per 3x3 block, no inversion centre")


def square(a=1.0):
    a1, a2 = (a, 0.0), (0.0, a)
    return cell_from_segments("square", "Square", "p4m", a1, a2, [(0, 0)],
                              _grid_segments(a1, a2, [(0, 0)]))


def rectangular(w=2.0, h=1.0):
    a1, a2 = (w, 0.0), (0.0, h)
    return cell_from_segments("rectangular", "Rectangular", "pmm", a1, a2, [(0, 0)],
                              _grid_segments(a1, a2, [(0, 0)]))


def triangular(a=1.0):
    a1 = np.array([a, 0.0]); a2 = np.array([a / 2, a * S3 / 2])
    segs = [((0, 0), a1, 0), ((0, 0), a2, 0), ((0, 0), a2 - a1, 0)]
    return cell_from_segments("triangular", "Triangular", "p6m", a1, a2, [(0, 0)], segs)


def honeycomb(a=1.0):
    """Bond length a."""
    a1 = np.array([S3 * a, 0.0]); a2 = np.array([S3 * a / 2, 1.5 * a])
    A, B = np.array([0.0, 0.0]), np.array([0.0, a])
    segs = [(A, B, 0), (A, B - a2, 0), (A, B + a1 - a2, 0)]
    return cell_from_segments("honeycomb", "Honeycomb", "p6m", a1, a2, [A, B], segs)


def breathing_kagome(L=1.0, l=0.6):
    """Up triangles of side L (blue) corner-sharing with down triangles of side l (red)."""
    return _kagome_family(L, l, "breathing_kagome", "Breathing Kagomé", "p3m1")


def _kagome_family(L, l, key, label, group):
    s = L + l
    a1 = np.array([s, 0.0]); a2 = np.array([s / 2, s * S3 / 2])
    b0, b1, b2 = np.array([0, 0.0]), np.array([L, 0.0]), np.array([L / 2, L * S3 / 2])
    up = [(b0, b1, 0), (b1, b2, 0), (b2, b0, 0)]
    tdown = 0 if key == "kagome" else 1
    down = [(b1, b0 + a1, tdown), (b2, b0 + a2, tdown), (b2, b1 + a2 - a1, tdown)]
    return cell_from_segments(key, label, group, a1, a2, [b0, b1, b2], up + down)


def kagome(a=1.0):
    return _kagome_family(a, a, "kagome", "Kagomé", "p6m")


def fractal_triangular(a=1.0):
    """Triangular lattice (side a) with a sqrt3 x sqrt3 superstructure: two of every three
    up-triangles kept; the third, with all down triangles, merges into side-2a down holes."""
    t1 = np.array([a, 0.0]); t2 = np.array([a / 2, a * S3 / 2])
    A1, A2 = t1 + t2, 2 * t2 - t1
    pts = [np.zeros(2), t1, t2]
    segs = []
    for p in (t1, t2):            # up triangles at classes 1 and 2; class 0 removed
        segs += [(p, p + t1, 0), (p, p + t2, 0), (p + t1, p + t2, 0)]
    return cell_from_segments("fractal_triangular", "Fractal Triangular", "p31m", A1, A2, pts, segs)


def elongated_triangular(a=1.0):
    a1 = np.array([a, 0.0]); a2 = np.array([a / 2, a * (1 + S3 / 2)])
    p, q = np.array([0.0, 0.0]), np.array([0.0, a])
    segs = [(p, p + a1, 0), (q, q + a1, 0), (p, q, 0), (q, p + a2, 0), (q, p + a2 - a1, 0)]
    return cell_from_segments("elongated_triangular", "Elongated Triangular", "cmm", a1, a2, [p, q], segs,
                              patch="box")


def _brick_from_offsets(key, label, group, offsets, w, h, note=""):
    """Rows of w x h bricks; the vertical joints of row k sit at x = offsets[k]*w (mod w)."""
    R = len(offsets)
    a1, a2 = np.array([w, 0.0]), np.array([0.0, R * h])
    basis, segs = [], []
    for k in range(R):
        xs = sorted({round((offsets[k - 1] % 1) * w, 12), round((offsets[k] % 1) * w, 12)})
        basis += [(x, k * h) for x in xs]
        xs2 = xs + [xs[0] + w]
        segs += [((xs2[m], k * h), (xs2[m + 1], k * h), 0) for m in range(len(xs))]
        x = (offsets[k] % 1) * w
        segs.append(((x, k * h), (x, (k + 1) * h), 0))
    return cell_from_segments(key, label, group, a1, a2, basis, segs, note)


def brick(w=2.0, h=1.0):
    """As drawn: running bond, each row shifted by half a brick."""
    return _brick_from_offsets("brick", "Brick", "pg", [0.0, 0.5], w, h, note="as drawn in the figure")


def brick_pg(w=2.0, h=1.0):
    """Four-row brick wall with joints at 0, 1/6, 1/2, 1/3 of a brick: rows 2,3 are the mirror
    images of rows 0,1, so only a glide (reflect x, shift two rows) survives."""
    return _brick_from_offsets("brick_pg", "Brick (4-row bond)", "pg", [0.0, 1 / 6, 0.5, 1 / 3], w, h,
                               note="variant: rows shifted by 0, 1/6, 1/2, 1/3 of a brick")


def _rect_plaquette_dimers(key, label, group, w, h):
    a1, a2 = np.array([w, 0.0]), np.array([0.0, h])
    pts = [i * a1 + j * a2 for j in range(2) for i in range(2)]
    segs = _grid_segments(a1, a2, pts)
    segs += [((0, 0), a1 + a2, 1), (2 * a1 + a2, a1 + 2 * a2, 1)]   # orthogonal dimers, checkerboard
    return cell_from_segments(key, label, group, 2 * a1, 2 * a2, pts, segs)


def shastry_sutherland(a=1.0):
    return _rect_plaquette_dimers("shastry_sutherland", "Shastry-Sutherland", "p4g", a, a)


def sutherlands_brick(w=2.0, h=1.0):
    return _rect_plaquette_dimers("sutherlands_brick", "Sutherland's Brick", "pgg", w, h)


def zigzag(w=2.0, h=1.0):
    """Rectangular grid; a red chevron '<' from every other horizontal line, in every column."""
    a1, a2 = np.array([w, 0.0]), np.array([0.0, 2 * h])
    pts = [(0, 0), (0, h)]
    segs = _grid_segments(a1, (0, h), pts) + [((0, h), (w, 0), 1), ((0, h), (w, 2 * h), 1)]
    return cell_from_segments("zigzag", "Zig Zag", "pmg", a1, a2, pts, segs)


def shastrys_ladder(w=2.0, h=1.0):
    """As drawn: the zig-zag chevrons in every other column only."""
    a1, a2 = np.array([2 * w, 0.0]), np.array([0.0, 2 * h])
    pts = [(0, 0), (w, 0), (0, h), (w, h)]
    segs = _grid_segments((w, 0), (0, h), pts) + [((0, h), (w, 0), 1), ((0, h), (w, 2 * h), 1)]
    return cell_from_segments("shastrys_ladder", "Shastry's Ladder", "pm", a1, a2, pts, segs,
                              note="as drawn in the figure")


def shastrys_ladder_pm(w=2.0, h=1.0):
    """Isolated chevrons: one '<' per 2 columns x 4 rows, so the chevrons no longer chain up."""
    a1, a2 = np.array([2 * w, 0.0]), np.array([0.0, 4 * h])
    pts = [(i * w, j * h) for j in range(4) for i in range(2)]
    segs = _grid_segments((w, 0), (0, h), pts) + [((0, h), (w, 0), 1), ((0, h), (w, 2 * h), 1)]
    return cell_from_segments("shastrys_ladder_pm", "Shastry's Ladder (isolated chevrons)", "pm",
                              a1, a2, pts, segs,
                              note="variant: chevrons every 4 rows instead of every 2")


def chiral_square(r=0.3, twist=20.0):
    """Square lattice; each site replaced by a red square (corner radius r) rotated by `twist` deg."""
    th = math.radians(twist)
    c = [r * np.array([math.cos(th + k * math.pi / 2), math.sin(th + k * math.pi / 2)]) for k in range(4)]
    a1, a2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    segs = [(c[k], c[(k + 1) % 4], 1) for k in range(4)]
    segs += [(c[0], a1 + c[2], 0), (c[1], a2 + c[3], 0)]
    return cell_from_segments("chiral_square", "Chiral Square", "p4", a1, a2, c, segs,
                              style={1: dict(color=RED, lw=2.4)})


def chiral_kagome(r=0.25, twist=15.0):
    """Honeycomb (bond length 1) with each site replaced by a red triangle (corner radius r)
    rotated by `twist` deg; blue bonds join the corners that face each other."""
    th = math.radians(twist)
    a1 = np.array([S3, 0.0]); a2 = np.array([S3 / 2, 1.5])
    A, B = np.array([0.0, 0.0]), np.array([0.0, 1.0])
    ang = lambda d: np.array([math.cos(math.radians(d) + th), math.sin(math.radians(d) + th)])
    cA = [A + r * ang(90 + 120 * k) for k in range(3)]
    cB = [B + r * ang(270 + 120 * k) for k in range(3)]
    segs = [(cA[k], cA[(k + 1) % 3], 1) for k in range(3)] + [(cB[k], cB[(k + 1) % 3], 1) for k in range(3)]
    segs += [(cA[0], cB[0], 0), (cA[1], cB[1] - a2, 0), (cA[2], cB[2] + a1 - a2, 0)]
    return cell_from_segments("chiral_kagome", "Chiral Kagomé", "p6", a1, a2, cA + cB, segs,
                              style={1: dict(color=RED, lw=2.4)})


def chiral_triangular(L=1.0, l=0.5):
    """Blue up-triangles (side L) corner-sharing with red up-triangles (side l)."""
    b1, b2, b3 = np.array([0, 0.0]), np.array([L, 0.0]), np.array([L / 2, L * S3 / 2])
    a1 = np.array([L + l / 2, l * S3 / 2]); a2 = np.array([l + L / 2, -L * S3 / 2])
    segs = [(b1, b2, 0), (b2, b3, 0), (b3, b1, 0)]
    segs += [(b2, b1 + a1, 1), (b2, b3 + a2, 1), (b1 + a1, b3 + a2, 1)]
    return cell_from_segments("chiral_triangular", "Chiral Triangular", "p3", a1, a2, [b1, b2, b3], segs,
                              style={1: dict(color=RED, lw=2.4), 0: dict(color=BLUE, lw=2.4)})


CATALOGUE = {
    "oblique": oblique, "straight": straight, "elongated_triangular": elongated_triangular,
    "square": square, "triangular": triangular, "kagome": kagome, "honeycomb": honeycomb,
    "rectangular": rectangular, "fractal_triangular": fractal_triangular,
    "breathing_kagome": breathing_kagome, "shastrys_ladder": shastrys_ladder,
    "shastry_sutherland": shastry_sutherland, "brick": brick, "zigzag": zigzag,
    "sutherlands_brick": sutherlands_brick, "chiral_square": chiral_square,
    "chiral_kagome": chiral_kagome, "chiral_triangular": chiral_triangular,
    # variants that realise the labelled group where the drawing does not (see the notebook)
    "straight_p1": straight_p1, "brick_pg": brick_pg, "shastrys_ladder_pm": shastrys_ladder_pm,
}

FIGURE_ORDER = ["oblique", "elongated_triangular", "straight", "square", "triangular", "kagome",
                "honeycomb", "rectangular", "fractal_triangular", "breathing_kagome", "shastrys_ladder",
                "shastry_sutherland", "brick", "zigzag", "sutherlands_brick", "chiral_square",
                "chiral_kagome", "chiral_triangular"]


def available_lattices():
    return list(CATALOGUE)


def make_cell(name, **params) -> UnitCell:
    if isinstance(name, UnitCell):
        return name
    key = name.lower().replace(" ", "_").replace("'", "").replace("-", "_").replace("é", "e")
    if key not in CATALOGUE:
        raise KeyError(f"unknown lattice {name!r}; choose from {available_lattices()}")
    return CATALOGUE[key](**params)


# ----------------------------------------------------------------------------------------------
# Wallpaper-group identification of the infinite decorated graph
# ----------------------------------------------------------------------------------------------
def _frac_mod(f):
    f = np.asarray(f, float)
    g = f - np.floor(f + TOL)
    g[np.abs(g - 1) < TOL] = 0.0
    return g


def _pure_translations(cell, use_types=True):
    """Fractional translations t (mod 1) that map the decorated graph onto itself."""
    return [t for (M, t) in _symmetry_ops(cell, only_identity=True, use_types=use_types)]


def primitive_cell(cell: UnitCell, use_types=True) -> UnitCell:
    """Reduce a (possibly super-) cell to a primitive cell of the decorated graph."""
    ts = _pure_translations(cell, use_types)
    A = cell.A
    if len(ts) == 1:
        b1, b2 = _reduce_basis(cell.a1, cell.a2)
        if np.allclose(b1, cell.a1) and np.allclose(b2, cell.a2):
            return cell
        return _rebase(cell, b1, b2)
    # candidate lattice vectors (Cartesian) = A (t + n), small n
    cands = []
    for t in ts:
        for n in itertools.product(range(-2, 3), repeat=2):
            v = A @ (t + np.array(n))
            if np.linalg.norm(v) > TOL:
                cands.append(v)
    cands.sort(key=lambda v: (round(np.linalg.norm(v), 9), -v[0], -v[1]))
    area = abs(np.linalg.det(A)) / len(ts)
    b1 = cands[0]
    b2 = next(v for v in cands if abs(abs(b1[0] * v[1] - b1[1] * v[0]) - area) < 1e-6 * max(1, area))
    b1, b2 = _reduce_basis(b1, b2)
    return _rebase(cell, b1, b2)


def _reduce_basis(b1, b2):
    """Lagrange-Gauss reduction: the shortest basis of the same lattice, right-handed. In a reduced
    basis every mirror direction of the lattice is n1 b1 + n2 b2 with |n| <= 2, which is what
    _candidate_matrices relies on."""
    b1, b2 = np.array(b1, float), np.array(b2, float)
    for _ in range(200):
        if b2 @ b2 < b1 @ b1 - 1e-12:
            b1, b2 = b2, b1
        k = np.round((b1 @ b2) / (b1 @ b1))
        if k == 0:
            break
        b2 = b2 - k * b1
    if b1[0] * b2[1] - b1[1] * b2[0] < 0:
        b2 = -b2
    return b1, b2


def _rebase(cell, b1, b2):
    """The same decorated graph described in the cell (b1, b2), which must be a period lattice of it."""
    A = cell.A
    B = np.column_stack([b1, b2]); Binv = np.linalg.inv(B)
    new_basis, new_species = [], []
    for k, p in enumerate(cell.basis):
        f = _frac_mod(Binv @ p)
        q = B @ f
        if not any(np.linalg.norm(q - r) < 1e-6 for r in new_basis):
            new_basis.append(q)
            new_species.append(None if cell.species is None else cell.species[k])
    new_basis = np.array(new_basis)
    bonds = set()
    for (i, j, n1, n2, t) in cell.bonds:
        p = cell.basis[i]; q = cell.basis[j] + A @ np.array([n1, n2])
        si, ni = _locate(p, B, Binv, new_basis)
        sj, nj = _locate(q, B, Binv, new_basis)
        bonds.add(_canon_bond(si, sj, nj[0] - ni[0], nj[1] - ni[1], t))
    return UnitCell(cell.key, cell.label, cell.group, b1, b2, new_basis, sorted(bonds),
                    cell.note, cell.style, cell.patch,
                    None if cell.species is None else list(new_species))


def _bond_lookup(cell, use_types=True):
    S = set()
    for (i, j, n1, n2, t) in cell.bonds:
        t = t if use_types else 0
        S.add((i, j, n1, n2, t)); S.add((j, i, -n1, -n2, t))
    return S


_NGRID = np.array(list(itertools.product(range(-3, 4), repeat=2)), float)
_ROTS = [np.array([[math.cos(k * math.pi / 6), -math.sin(k * math.pi / 6)],
                   [math.sin(k * math.pi / 6), math.cos(k * math.pi / 6)]]) for k in range(12)]


def _candidate_matrices(cell):
    """Cartesian orthogonal matrices: rotations by k*30 deg, reflections across lattice directions."""
    V = _NGRID @ cell.A.T
    V = V[np.linalg.norm(V, axis=1) > TOL]
    a = np.arctan2(V[:, 1], V[:, 0]) % math.pi
    a = np.concatenate([a, (a + math.pi / 2) % math.pi])
    angles = np.unique(np.round(a, 9))
    mats = list(_ROTS)
    for x in angles:
        c, s = math.cos(2 * x), math.sin(2 * x)
        mats.append(np.array([[c, s], [s, -c]]))
    return mats


def _symmetry_ops(cell, only_identity=False, use_types=True):
    """All (Mf, t): x_frac -> Mf x_frac + t (t mod 1) mapping sites and coloured bonds onto themselves.
    use_types=False ignores bond colours (the bare graph). Site species, if set, are always respected."""
    A = cell.A; Ainv = np.linalg.inv(A)
    # unreduced fractional coordinates, so that cell shifts stay consistent with the bond list
    F = np.asarray(cell.basis, float) @ Ainv.T
    sp = np.array(cell.species if cell.species is not None else [0] * len(F), dtype=object)
    same_sp = sp[:, None] == sp[None, :]
    bonds = _bond_lookup(cell, use_types)
    BL = np.array([b[:4] for b in cell.bonds], int).reshape(-1, 4)
    BT = [(b[4] if use_types else 0) for b in cell.bonds]
    mats = [np.eye(2)] if only_identity else _candidate_matrices(cell)
    ops = []
    for M in mats:
        Mf = Ainv @ M @ A
        R = np.round(Mf)
        if np.max(np.abs(Mf - R)) > 1e-6:
            continue
        Mf = R.astype(int)
        MF = F @ Mf.T
        for j in range(len(F)):
            t = _frac_mod(F[j] - MF[0])
            D = (MF + t)[:, None, :] - F[None, :, :]
            RD = np.round(D)
            match = np.all(np.abs(D - RD) < 1e-6, axis=2) & same_sp
            if not np.all(match.any(axis=1)):
                continue
            perm = match.argmax(axis=1)
            shift = RD[np.arange(len(F)), perm].astype(int)
            if len(BL):
                dn = BL[:, 2:4] @ Mf.T + shift[BL[:, 1]] - shift[BL[:, 0]]
                pi, pk = perm[BL[:, 0]], perm[BL[:, 1]]
                ok = all((int(pi[b]), int(pk[b]), int(dn[b, 0]), int(dn[b, 1]), BT[b]) in bonds
                         for b in range(len(BL)))
            else:
                ok = True
            if ok and not any(np.array_equal(Mf, M2) and np.allclose(t, t2) for M2, t2 in ops):
                ops.append((Mf, t))
    return ops


def identify_wallpaper_group(cell, return_details=False, use_types=True):
    """Wallpaper group of the infinite decorated graph, computed from its symmetry operations.
    use_types=False: ignore bond colours and classify the bare graph (site species still count)."""
    cell = primitive_cell(make_cell(cell), use_types)
    A = cell.A; Ainv = np.linalg.inv(A)
    ops = _symmetry_ops(cell, use_types=use_types)
    rot_orders, refl = [], []
    for Mf, t in ops:
        M = A @ Mf @ Ainv
        if np.linalg.det(M) > 0:
            ang = math.atan2(M[1, 0], M[0, 0])
            n = 1 if abs(ang) < 1e-6 else int(round(2 * math.pi / abs(ang)))
            rot_orders.append((n, Mf, t))
        else:
            refl.append((Mf, t))
    n = max(o for o, _, _ in rot_orders)
    Ls = [np.array(v) for v in itertools.product(range(-3, 4), repeat=2)]
    I2 = np.eye(2, dtype=int)

    LA = np.array(Ls, float)

    def has_mirror(Mf, t):
        V = (t + LA) @ (Mf + I2).T
        return bool(np.any(np.all(np.abs(V) < 1e-6, axis=1)))

    def has_glide(Mf, t):
        g = (t + LA) @ (Mf + I2).T / 2
        return bool(np.any(np.max(np.abs(g - np.round(g)), axis=1) > 1e-6))

    def centres_on_mirrors(order):
        """Do all rotation centres of the given order lie on a mirror line?"""
        for o, Mf, t in rot_orders:
            if o != order:
                continue
            for L in Ls[:]:
                c = np.linalg.solve(np.eye(2) - Mf, t + L)
                on = False
                for Rf, s in refl:
                    d = Rf @ c + s - c
                    if np.all(np.abs(d - np.round(d)) < 1e-6):
                        on = True; break
                if not on:
                    return False
        return True

    mirrors = [has_mirror(M, t) for M, t in refl]
    glides = [has_glide(M, t) for M, t in refl]
    if n == 1:
        g = "p1" if not refl else ("pg" if not any(mirrors) else ("cm" if any(glides) else "pm"))
    elif n == 2:
        if not refl:
            g = "p2"
        elif all(mirrors):
            g = "cmm" if any(glides) else "pmm"
        elif any(mirrors):
            g = "pmg"
        else:
            g = "pgg"
    elif n == 3:
        g = "p3" if not refl else ("p3m1" if centres_on_mirrors(3) else "p31m")
    elif n == 4:
        g = "p4" if not refl else ("p4m" if centres_on_mirrors(4) else "p4g")
    elif n == 6:
        g = "p6" if not refl else "p6m"
    else:
        raise RuntimeError(f"rotation order {n}?")
    if return_details:
        return g, dict(point_group_order=len(ops), max_rotation=n, reflections=len(refl),
                       mirror_classes=sum(mirrors), primitive_sites=len(cell.basis),
                       primitive_bonds=len(cell.bonds))
    return g


# ----------------------------------------------------------------------------------------------
# Finite patches
# ----------------------------------------------------------------------------------------------
def _is_hexagonal(a1, a2):
    l1, l2 = np.linalg.norm(a1), np.linalg.norm(a2)
    if abs(l1 - l2) > 1e-6 * l1:
        return 0
    c = np.dot(a1, a2) / (l1 * l2)
    if abs(c - 0.5) < 1e-6:
        return +1          # 60 deg
    if abs(c + 0.5) < 1e-6:
        return -1          # 120 deg
    return 0


class WallpaperLattice:
    """A finite patch of a periodic lattice from the catalogue.

    Parameters
    ----------
    name : str or UnitCell     catalogue key (see available_lattices()) or a custom UnitCell
    n : int                    number of layers of unit cells around the central cell (shape="layers");
                               layers are hexagonal rings for hexagonal Bravais lattices, square rings
                               otherwise (in cell coordinates)
    shape : "layers" | "parallelogram" | "disk"
    L : (L1, L2)               cells along a1 and a2 (shape="parallelogram")
    radius : float             keep sites within this distance of the centre (shape="disk")
    boundary : "open" | "periodic"   periodic only with shape="parallelogram"
    prune : bool               open boundaries: repeatedly remove sites with at most one bond, so the
                               edge of the patch has no dangling bonds (as in the figure), and keep only
                               the largest connected piece
    **params                   passed to the lattice constructor (e.g. twist=25, l=0.4, w=3)
    """

    def __init__(self, name, n=3, shape="layers", L=None, radius=None, boundary="open", prune=True,
                 **params):
        self.cell = make_cell(name, **params)
        self.shape, self.boundary = shape, boundary
        if boundary not in ("open", "periodic"):
            raise ValueError("boundary must be 'open' or 'periodic'")
        if boundary == "periodic" and shape != "parallelogram":
            raise ValueError("periodic boundaries need shape='parallelogram'")
        a1, a2 = self.cell.a1, self.cell.a2
        cells, layer = [], []
        if shape == "layers":
            mode = self.cell.patch
            if mode == "auto":
                mode = "hex" if _is_hexagonal(a1, a2) else "cells"
            hexa = _is_hexagonal(a1, a2)
            ex = np.linalg.norm(a1); ey = abs(np.linalg.det(self.cell.A)) / ex
            ux = a1 / ex; uy = np.array([-ux[1], ux[0]])
            span = 3 * n + 3 if mode == "box" else n
            for i in range(-span, span + 1):
                for j in range(-span, span + 1):
                    if mode == "hex":
                        d = max(abs(i), abs(j), abs(i + j) if hexa == 1 else abs(i - j))
                    elif mode == "box":
                        c = self.cell.A @ np.array([i, j])
                        d = int(round(max(abs(c @ ux) / ex, abs(c @ uy) / ey)))
                    else:
                        d = max(abs(i), abs(j))
                    if d <= n:
                        cells.append((i, j)); layer.append(d)
            self.n = n
        elif shape == "parallelogram":
            L = L or (n, n)
            self.L = tuple(L)
            for i in range(L[0]):
                for j in range(L[1]):
                    cells.append((i, j)); layer.append(0)
        elif shape == "disk":
            if radius is None:
                raise ValueError("shape='disk' needs radius")
            R = radius
            span = int(np.ceil(R / min(np.linalg.norm(a1), np.linalg.norm(a2),
                                       abs(np.linalg.det(self.cell.A)) / max(np.linalg.norm(a1), np.linalg.norm(a2))))) + 2
            for i in range(-span, span + 1):
                for j in range(-span, span + 1):
                    cells.append((i, j)); layer.append(0)
        else:
            raise ValueError("shape must be 'layers', 'parallelogram' or 'disk'")

        m = len(self.cell.basis)
        A = self.cell.A
        pos, sub, cidx, lay = [], [], [], []
        index = {}
        for (i, j), l in zip(cells, layer):
            o = A @ np.array([i, j])
            for s in range(m):
                p = o + self.cell.basis[s]
                if shape == "disk" and np.linalg.norm(p) > radius + 1e-9:
                    continue
                index[(i, j, s)] = len(pos)
                pos.append(p); sub.append(s); cidx.append((i, j)); lay.append(l)
        self.positions = np.array(pos)
        self.sublattice = np.array(sub)
        self.cell_index = np.array(cidx)
        self.layer = np.array(lay)
        self._index = index

        edges, types, wraps, dvec = [], [], [], []
        seen = set()
        for (i, j, s), u in index.items():
            for (a, b, n1, n2, t) in self.cell.bonds:
                for (src, dst, d1, d2) in ((a, b, n1, n2), (b, a, -n1, -n2)):
                    if src != s:
                        continue
                    ti, tj = i + d1, j + d2
                    wrapped = False
                    if boundary == "periodic":
                        ti2, tj2 = ti % self.L[0], tj % self.L[1]
                        wrapped = (ti2, tj2) != (ti, tj)
                        ti, tj = ti2, tj2
                    v = index.get((ti, tj, dst))
                    if v is None or v == u:
                        continue
                    key = (min(u, v), max(u, v), t, (d1, d2) if u < v else (-d1, -d2))
                    if key in seen:
                        continue
                    seen.add(key)
                    vec = A @ np.array([d1, d2]) + self.cell.basis[dst] - self.cell.basis[src]
                    edges.append((u, v) if u < v else (v, u))
                    types.append(t); wraps.append(wrapped)
                    dvec.append(vec if u < v else -vec)
        self.edges = np.array(edges, dtype=int).reshape(-1, 2)
        self.edge_types = np.array(types, dtype=int)
        self.edge_wraps = np.array(wraps, dtype=bool)
        self.bond_vectors = np.array(dvec).reshape(-1, 2)   # from edges[:,0] to edges[:,1], unwrapped
        if prune and boundary == "open":
            self._prune()
        pairs = {(min(e), max(e)) for e in map(tuple, self.edges)}
        self.has_multi_edges = len(pairs) < len(self.edges)

    def _prune(self):
        keep = np.ones(len(self.positions), bool)
        alive = np.ones(len(self.edges), bool)
        while True:
            deg = np.bincount(self.edges[alive].ravel(), minlength=len(keep))
            bad = keep & (deg <= 1)
            if not bad.any():
                break
            keep &= ~bad
            alive &= keep[self.edges[:, 0]] & keep[self.edges[:, 1]]
        # keep only the largest connected piece (drops fragments cut off at the patch edge)
        if alive.any():
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            e = self.edges[alive]
            n = len(keep)
            _, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)),
                                          directed=False)
            big = np.bincount(lab[keep]).argmax()
            keep &= lab == big
            alive &= keep[self.edges[:, 0]] & keep[self.edges[:, 1]]
        new = -np.ones(len(keep), int); new[keep] = np.arange(keep.sum())
        for a in ("positions", "sublattice", "cell_index", "layer"):
            setattr(self, a, getattr(self, a)[keep])
        self._index = {k: int(new[v]) for k, v in self._index.items() if keep[v]}
        self.edges = new[self.edges[alive]]
        for a in ("edge_types", "edge_wraps", "bond_vectors"):
            setattr(self, a, getattr(self, a)[alive])

    # --- hypertiling-like accessors ---------------------------------------------------------
    def __len__(self):
        return len(self.positions)

    def __iter__(self):
        return iter(self.positions)

    def __repr__(self):
        return (f"WallpaperLattice({self.cell.label!r}, group={self.cell.group}, shape={self.shape}, "
                f"boundary={self.boundary}, sites={len(self)}, bonds={len(self.edges)})")

    def get_nbrs_list(self, types=None):
        """List of neighbour index arrays, one per site (like hypertiling's get_nbrs_list)."""
        nb = [[] for _ in range(len(self))]
        for (u, v), t in zip(self.edges, self.edge_types):
            if types is None or t in types:
                nb[u].append(v); nb[v].append(u)
        return [np.array(x, dtype=int) for x in nb]

    def get_nbrs(self, i, types=None):
        return self.get_nbrs_list(types)[i]

    def coordination(self):
        return np.array([len(x) for x in self.get_nbrs_list()])

    def bond_lengths(self):
        return np.linalg.norm(self.bond_vectors, axis=1)

    def adjacency_matrix(self, sparse=True, weights=None):
        """Symmetric adjacency matrix. weights: None (all 1) or {bond type: weight}."""
        import scipy.sparse as sp
        w = np.ones(len(self.edges)) if weights is None else np.array([weights.get(t, 0.0) for t in self.edge_types])
        u, v = self.edges[:, 0], self.edges[:, 1]
        M = sp.coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(len(self), len(self))).tocsr()
        return M if sparse else M.toarray()

    def rigidity_matrix(self, sparse=True):
        """Compatibility matrix C (bonds x 2N): bond extensions e = C u for site displacements u."""
        import scipy.sparse as sp
        nE = len(self.edges)
        e = self.bond_vectors / self.bond_lengths()[:, None]
        rows = np.repeat(np.arange(nE), 4)
        u, v = self.edges[:, 0], self.edges[:, 1]
        cols = np.stack([2 * u, 2 * u + 1, 2 * v, 2 * v + 1], axis=1).ravel()
        vals = np.stack([-e[:, 0], -e[:, 1], e[:, 0], e[:, 1]], axis=1).ravel()
        C = sp.coo_matrix((vals, (rows, cols)), shape=(nE, 2 * len(self))).tocsr()
        return C if sparse else C.toarray()

    def to_networkx(self):
        import networkx as nx
        G = nx.MultiGraph() if self.has_multi_edges else nx.Graph()   # small tori have doubled bonds
        for k, p in enumerate(self.positions):
            G.add_node(k, pos=tuple(p), sublattice=int(self.sublattice[k]), layer=int(self.layer[k]))
        for (u, v), t in zip(self.edges, self.edge_types):
            G.add_edge(int(u), int(v), type=int(t))
        return G

    def identify_symmetry(self):
        return identify_wallpaper_group(self.cell)

    # --- plotting -----------------------------------------------------------------------------
    def plot(self, ax=None, show_sites=False, site_size=8, color_by=None, lw=1.5, title=True,
             draw_wrapped=False, colors=None, symmetry=False):
        """Draw the patch. Bond type k uses colors[k] (default PALETTE: blue, red, green, ...).
        color_by: None | 'sublattice' | 'layer' | 'species' colours the sites.
        symmetry=True overlays the symmetry elements of the infinite lattice over the whole patch
        (mirrors solid, glides dashed, rotation centres as 2/3/4/6-gons); symmetry=(xmin, xmax,
        ymin, ymax) draws them only inside that window."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        colors = colors or PALETTE
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        for t in sorted(set(self.edge_types.tolist()), reverse=True):
            sel = (self.edge_types == t) & (draw_wrapped | ~self.edge_wraps)
            p0 = self.positions[self.edges[sel, 0]]
            p1 = p0 + self.bond_vectors[sel]
            st = dict(color=colors[t % len(colors)], lw=lw)
            st.update(self.cell.style.get(t, {}))
            ax.add_collection(LineCollection(np.stack([p0, p1], axis=1), capstyle="round",
                                             joinstyle="round", zorder=2 if t else 1, **st))
        if show_sites or color_by:
            if color_by == "species":
                sp = self.cell.species if self.cell.species is not None else [0] * len(self.cell.basis)
                c = np.array([sp[s] for s in self.sublattice])
            else:
                c = None if color_by is None else getattr(self, color_by)
            if c is None:
                ax.scatter(*self.positions.T, s=site_size, c="k", zorder=3)
            else:
                ax.scatter(*self.positions.T, s=site_size, c=c, cmap="viridis", zorder=3)
        if symmetry is not False and symmetry is not None:
            if symmetry is True:
                lo, hi = self.positions.min(axis=0), self.positions.max(axis=0)
                bounds = (lo[0], hi[0], lo[1], hi[1])
            else:
                bounds = tuple(symmetry)
            plot_symmetry_elements(self.cell, ax, bounds=bounds)
        ax.set_aspect("equal"); ax.autoscale_view(); ax.axis("off")
        if title:
            ax.set_title(f"{self.cell.label}\n({self.cell.group})", fontsize=10)
        return ax


# ----------------------------------------------------------------------------------------------
# Lattices with a chosen wallpaper group
# ----------------------------------------------------------------------------------------------
# General positions of the 17 plane groups in the standard settings of the International Tables
# for Crystallography, Vol. A (coordinates are fractional in the conventional cell; hexagonal axes
# have gamma = 120 deg; c-groups list the operations once and add the centring (1/2, 1/2)).
_GROUP_TABLE = {
    "p1":   ("oblique",     False, ["x,y"]),
    "p2":   ("oblique",     False, ["x,y", "-x,-y"]),
    "pm":   ("rectangular", False, ["x,y", "-x,y"]),
    "pg":   ("rectangular", False, ["x,y", "-x,y+1/2"]),
    "cm":   ("rectangular", True,  ["x,y", "-x,y"]),
    "pmm":  ("rectangular", False, ["x,y", "-x,-y", "-x,y", "x,-y"]),
    "pmg":  ("rectangular", False, ["x,y", "-x,-y", "-x+1/2,y", "x+1/2,-y"]),
    "pgg":  ("rectangular", False, ["x,y", "-x,-y", "-x+1/2,y+1/2", "x+1/2,-y+1/2"]),
    "cmm":  ("rectangular", True,  ["x,y", "-x,-y", "-x,y", "x,-y"]),
    "p4":   ("square",      False, ["x,y", "-x,-y", "-y,x", "y,-x"]),
    "p4m":  ("square",      False, ["x,y", "-x,-y", "-y,x", "y,-x", "-x,y", "x,-y", "y,x", "-y,-x"]),
    "p4g":  ("square",      False, ["x,y", "-x,-y", "-y,x", "y,-x", "-x+1/2,y+1/2", "x+1/2,-y+1/2",
                                    "y+1/2,x+1/2", "-y+1/2,-x+1/2"]),
    "p3":   ("hexagonal",   False, ["x,y", "-y,x-y", "-x+y,-x"]),
    "p3m1": ("hexagonal",   False, ["x,y", "-y,x-y", "-x+y,-x", "-y,-x", "-x+y,y", "x,x-y"]),
    "p31m": ("hexagonal",   False, ["x,y", "-y,x-y", "-x+y,-x", "y,x", "x-y,-y", "-x,-x+y"]),
    "p6":   ("hexagonal",   False, ["x,y", "-y,x-y", "-x+y,-x", "-x,-y", "y,-x+y", "x-y,x"]),
    "p6m":  ("hexagonal",   False, ["x,y", "-y,x-y", "-x+y,-x", "-x,-y", "y,-x+y", "x-y,x",
                                    "-y,-x", "-x+y,y", "x,x-y", "y,x", "x-y,-y", "-x,-x+y"]),
}
_GROUP_ALIASES = {"p2mm": "pmm", "p2mg": "pmg", "p2gg": "pgg", "c2mm": "cmm", "c1m1": "cm", "p1m1": "pm",
                  "p1g1": "pg", "p4mm": "p4m", "p4gm": "p4g", "p6mm": "p6m"}
WALLPAPER_GROUPS = list(_GROUP_TABLE)


def _parse_op(op):
    """'-x+1/2,y' -> (integer matrix Mf, translation t) acting on fractional coordinates."""
    comps = op.replace(" ", "").split(",")
    f = lambda c, x, y: eval(c, {"__builtins__": {}}, {"x": x, "y": y})
    M = np.zeros((2, 2)); t = np.zeros(2)
    for r, c in enumerate(comps):
        t[r] = f(c, 0.0, 0.0)
        M[r] = [f(c, 1.0, 0.0) - t[r], f(c, 0.0, 1.0) - t[r]]
    return np.round(M).astype(int), t


class WallpaperGroup:
    """One of the 17 plane groups, with its operations in the ITA standard setting.

    G = WallpaperGroup("p4g");  G.ops -> [(Mf, t)] including centring;  G.order -> multiplicity
    G.lattice(a=1, b=..., gamma=...) -> (a1, a2) conventional cell for that crystal system
    G.orbit((x, y)) -> fractional points of the orbit in the conventional cell
    """

    def __init__(self, symbol):
        key = symbol.lower().replace(" ", "")
        key = _GROUP_ALIASES.get(key, key)
        if key not in _GROUP_TABLE:
            raise KeyError(f"unknown wallpaper group {symbol!r}; choose from {WALLPAPER_GROUPS}")
        self.symbol = key
        self.system, self.centred, strs = _GROUP_TABLE[key]
        ops = [_parse_op(o) for o in strs]
        if self.centred:
            ops = ops + [(M, _frac_mod(t + 0.5)) for M, t in ops]
        self.ops = [(M, _frac_mod(t)) for M, t in ops]
        self.order = len(self.ops)         # multiplicity of the general position (conventional cell)

    def __repr__(self):
        return (f"WallpaperGroup({self.symbol}, {self.system}{', centred' if self.centred else ''}, "
                f"general position {self.order})")

    def lattice(self, a=1.0, b=None, gamma=None):
        if self.system == "oblique":
            b = 1.23 * a if b is None else b
            g = math.radians(76.0 if gamma is None else gamma)
            return np.array([a, 0.0]), b * np.array([math.cos(g), math.sin(g)])
        if self.system == "rectangular":
            b = (1.6 if self.centred else 1.37) * a if b is None else b
            return np.array([a, 0.0]), np.array([0.0, b])
        if self.system == "square":
            return np.array([a, 0.0]), np.array([0.0, a])
        return np.array([a, 0.0]), a * np.array([-0.5, S3 / 2])

    def orbit(self, p):
        p = np.asarray(p, float)
        pts = []
        for M, t in self.ops:
            q = _frac_mod(M @ p + t)
            if not any(np.all(np.abs((q - r) - np.round(q - r)) < 1e-7) for r in pts):
                pts.append(q)
        return pts


def _apply_to_site(Mf, t, F, k):
    """Image of site k under (Mf, t): (site index, integer cell shift)."""
    g = Mf @ F[k] + t
    for j in range(len(F)):
        d = g - F[j]
        r = np.round(d)
        if np.all(np.abs(d - r) < 1e-6):
            return j, r.astype(int)
    raise RuntimeError("site orbit is not closed under the group")


def _bond_orbits(bonds, ops, F):
    """Partition canonical bonds (i, j, n1, n2) into orbits under ops. Returns {bond: orbit id}."""
    bonds = list(dict.fromkeys(bonds))
    orbit = {}
    nid = 0
    for b in bonds:
        if b in orbit:
            continue
        i, j, n1, n2 = b
        for Mf, t in ops:
            si, ci = _apply_to_site(Mf, t, F, i)
            sj, cj = _apply_to_site(Mf, t, F, j)
            dn = Mf @ np.array([n1, n2]) + cj - ci
            img = _canon_bond(si, sj, int(dn[0]), int(dn[1]), 0)[:4]
            orbit.setdefault(img, nid)
        nid += 1
    return orbit


def _distance_shells(A, basis, n_shells=None, cutoffs=None, tol=1e-6):
    """All bonds (i, j, n1, n2) whose length is in the first n_shells distance shells (or <= cutoffs)."""
    m = len(basis)
    R = 4
    pairs = []
    for i in range(m):
        for j in range(m):
            for n in itertools.product(range(-R, R + 1), repeat=2):
                if i == j and n == (0, 0):
                    continue
                d = np.linalg.norm(basis[j] + A @ np.array(n) - basis[i])
                pairs.append((round(d, 7), _canon_bond(i, j, n[0], n[1], 0)[:4]))
    dists = sorted({d for d, _ in pairs})
    shells = []
    for d in dists:
        if not shells or d - shells[-1] > tol:
            shells.append(d)
    if cutoffs is not None:
        cut = sorted(cutoffs)
        shell_of = lambda d: next((k for k, c in enumerate(cut) if d <= c + tol), None)
    else:
        keep = shells[:n_shells]
        shell_of = lambda d: next((k for k, c in enumerate(keep) if abs(d - c) <= tol), None)
    out = {}
    for d, b in pairs:
        k = shell_of(d)
        if k is not None:
            out[b] = k
    return out, shells


def symmetric_cell(group, sites, bonds=None, n_shells=1, cutoffs=None, color_by="orbit",
                   species=None, a=1.0, b=None, gamma=None, label=None, check=True):
    """A lattice whose symmetry group contains `group`, built from orbits.

    group     : wallpaper group symbol, e.g. "p4g" (ITA standard setting; aliases like "p4gm" work)
    sites     : list of fractional points (x, y) in the conventional cell; each generates a site orbit
                (a general point gives G.order sites, a special Wyckoff position fewer)
    bonds     : None -> connect every pair of sites in the first n_shells distance shells
                (or with length <= each value in cutoffs); or a list of explicit generating bonds
                [(p, q), ...] / [(p, q, type), ...] with p, q fractional site positions (q may lie in
                a neighbouring cell, e.g. (1.1, 0.2)); each generates a bond orbit
    color_by  : "orbit" (each bond orbit its own type: the independent couplings the group allows),
                "shell" (one type per distance shell), or "single"
    species   : None (all sites alike) or "orbit" (sites of different orbits are distinguishable)
    a, b, gamma : lattice parameters; which are free depends on the crystal system
    check     : compute the actual group and store it in cell.found (with bond colours) and
                cell.found_uncoloured (bare graph); also set cell.connected and warn if the infinite
                lattice falls apart into pieces (then add shells or bonds)

    The result always has at least the symmetry of `group`. It has more when the sites or bonds are
    special; cell.found tells you.
    """
    G = group if isinstance(group, WallpaperGroup) else WallpaperGroup(group)
    a1, a2 = G.lattice(a, b, gamma)
    A = np.column_stack([a1, a2])
    F, sp = [], []
    for o, p in enumerate(sites):
        for q in G.orbit(p):
            if any(np.all(np.abs((q - r) - np.round(q - r)) < 1e-7) for r in F):
                raise ValueError(f"site {p} lies on the orbit of an earlier site")
            F.append(q); sp.append(o)
    F = np.array(F)
    basis = (A @ F.T).T

    if bonds is None:
        shell_bonds, _ = _distance_shells(A, basis, n_shells=n_shells, cutoffs=cutoffs)
        orbits = _bond_orbits(list(shell_bonds), G.ops, F)
        if color_by == "orbit":
            typed = {bd: orbits[bd] for bd in shell_bonds}
        elif color_by == "shell":
            typed = dict(shell_bonds)
        else:
            typed = {bd: 0 for bd in shell_bonds}
    else:
        Ainv = np.linalg.inv(A)
        typed = {}
        for k, bd in enumerate(bonds):
            p, q = np.asarray(bd[0], float), np.asarray(bd[1], float)
            i, ni = _locate(A @ p, A, Ainv, basis)
            j, nj = _locate(A @ q, A, Ainv, basis)
            gen = _canon_bond(i, j, nj[0] - ni[0], nj[1] - ni[1], 0)[:4]
            ty = bd[2] if len(bd) > 2 else (k if color_by == "orbit" else 0)
            for bnd in _bond_orbits([gen], G.ops, F):
                typed.setdefault(bnd, ty)
    # renumber automatic types 0, 1, 2, ... in order of increasing bond length (explicit types are kept)
    explicit = bonds is not None and any(len(bd) > 2 for bd in bonds)
    lengths = {}
    for (i, j, n1, n2), ty in typed.items():
        lengths.setdefault(ty, np.linalg.norm(basis[j] + A @ np.array([n1, n2]) - basis[i]))
    order = {ty: (ty if explicit else k)
             for k, ty in enumerate(sorted(lengths, key=lambda t: (round(lengths[t], 7), t)))}
    bond_list = sorted((i, j, n1, n2, order[ty]) for (i, j, n1, n2), ty in typed.items())
    cell = UnitCell(f"sym_{G.symbol}", label or f"{G.symbol} lattice", G.symbol, a1, a2, basis, bond_list,
                    note=f"built from {len(sites)} site orbit(s) of {G.symbol}",
                    species=None if species is None else sp)
    if check:
        cell.found = identify_wallpaper_group(cell)
        cell.found_uncoloured = identify_wallpaper_group(cell, use_types=False)
        cell.connected = is_connected(cell)
        if not cell.connected:
            import warnings
            warnings.warn(f"{cell.label}: the lattice is not connected; add distance shells or bonds",
                          stacklevel=2)
    return cell


def _segments_cross(p, q, r, s, tol=1e-9):
    d = lambda u, v: u[0] * v[1] - u[1] * v[0]
    a, b_ = q - p, s - r
    den = d(a, b_)
    if abs(den) < tol:
        return False
    u = d(r - p, b_) / den; v = d(r - p, a) / den
    return tol < u < 1 - tol and tol < v < 1 - tol


def has_crossings(cell):
    """True if two bonds cross (the lattice is not drawn as a planar network)."""
    A = cell.A
    segs0 = [(cell.basis[i], cell.basis[j] + A @ np.array([n1, n2])) for (i, j, n1, n2, _) in cell.bonds]
    for n in itertools.product(range(-5, 6), repeat=2):
        sh = A @ np.array(n)
        for k, (p, q) in enumerate(segs0):
            for l, (r, s) in enumerate(segs0):
                if n == (0, 0) and l <= k:
                    continue
                if _segments_cross(p, q, r + sh, s + sh):
                    return True
    return False


def periodic_connected(nsites, bonds):
    """Is the infinite periodic graph connected? bonds: (i, j, n1, n2).
    True iff the graph of one cell is connected and its closed loops, each with the lattice vector it
    winds by, generate the whole translation lattice (gcd of their 2x2 determinants equal to 1)."""
    from math import gcd
    adj = [[] for _ in range(nsites)]
    for (i, j, n1, n2) in bonds:
        adj[i].append((j, (n1, n2))); adj[j].append((i, (-n1, -n2)))
    off = {0: (0, 0)}
    stack = [0]
    cyc = []
    while stack:
        u = stack.pop()
        for v, (a, b) in adj[u]:
            c0 = (off[u][0] + a, off[u][1] + b)
            if v not in off:
                off[v] = c0; stack.append(v)
            else:
                c = (c0[0] - off[v][0], c0[1] - off[v][1])
                if c != (0, 0):
                    cyc.append(c)
    if len(off) < nsites:
        return False
    g = 0
    for p, q in itertools.combinations(cyc, 2):
        g = gcd(g, abs(p[0] * q[1] - p[1] * q[0]))
        if g == 1:
            return True
    return g == 1


def is_connected(cell):
    """Whether the infinite periodic graph is connected (exact)."""
    return periodic_connected(len(cell.basis), [tuple(int(x) for x in b[:4]) for b in cell.bonds])


def random_symmetric_cell(group, n_orbits="auto", max_shells=10, exact="uncoloured", planar=True,
                          min_separation=0.2, seed=None, max_tries=400, color_by="orbit", **lattice):
    """A random connected lattice with exactly the symmetry of `group`.

    Draws random general points for the site orbits, then adds distance shells until the lattice is
    connected (at most max_shells). A candidate is kept only if it has no crossing bonds (planar=True), no two sites closer
    than min_separation * (cell area / sites)^(1/2), and its computed group equals `group`:
    exact="uncoloured" requires this of the bare graph (strongest), "coloured" of the graph with bond
    colours, None skips the test. n_orbits="auto" starts with one site orbit and adds more when needed
    (p1, for example, needs three).
    """
    G = WallpaperGroup(group)
    rng = np.random.default_rng(seed)
    orbit_counts = [n_orbits] if n_orbits != "auto" else [1, 2, 3, 4]
    per = max(1, max_tries // len(orbit_counts))
    for no in orbit_counts:
        for _ in range(per):
            pts = [rng.uniform(0, 1, 2) for _ in range(no)]
            a1, a2 = G.lattice(**lattice)
            A = np.column_stack([a1, a2])
            allf = [q for p in pts for q in G.orbit(p)]
            if len(allf) != no * G.order:
                continue
            X = np.array([A @ f for f in allf])
            dmin = min(np.linalg.norm(X[i] - X[j] - A @ np.array(n))
                       for i in range(len(X)) for j in range(len(X))
                       for n in itertools.product(range(-1, 2), repeat=2) if (i, n) != (j, (0, 0)))
            if dmin < min_separation * math.sqrt(abs(np.linalg.det(A)) / len(X)):
                continue
            for ns in range(1, max_shells + 1):
                cell = symmetric_cell(G, pts, n_shells=ns, color_by=color_by, check=False, **lattice)
                if is_connected(cell):
                    break
            else:
                continue
            if planar and has_crossings(cell):
                continue
            cell.found = identify_wallpaper_group(cell)
            cell.found_uncoloured = identify_wallpaper_group(cell, use_types=False)
            target = {"uncoloured": cell.found_uncoloured, "coloured": cell.found, None: G.symbol}[exact]
            if target == G.symbol:
                cell.sites_frac = [tuple(float(x) for x in np.round(p, 4)) for p in pts]
                cell.note = (f"random {G.symbol} lattice: {no} site orbit(s) at {cell.sites_frac}, "
                             f"{ns} distance shell(s)")
                return cell
    raise RuntimeError(f"no lattice with exactly {G.symbol} found in {max_tries} tries; "
                       "try planar=False, more max_shells, or another seed")


def symmetry_elements(cell, bounds, use_types=True):
    """Rotation centres, mirror lines and glide lines of the infinite lattice inside
    bounds = (xmin, xmax, ymin, ymax). Returns dict(rotations=[(order, point)],
    mirrors=[(point, direction)], glides=[(point, direction, glide vector)])."""
    A = cell.A; Ainv = np.linalg.inv(A)
    ops = _symmetry_ops(cell, use_types=use_types)
    trans = [t for M, t in ops if np.array_equal(M, np.eye(2, dtype=int))]
    xmin, xmax, ymin, ymax = bounds
    span = max(xmax - xmin, ymax - ymin)
    hmin = abs(np.linalg.det(A)) / max(np.linalg.norm(cell.a1), np.linalg.norm(cell.a2))
    K = int(np.ceil(span / hmin)) + 2
    c0 = np.round(Ainv @ np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])).astype(int)

    def in_T(gf):
        return any(np.all(np.abs((gf - t) - np.round(gf - t)) < 1e-6) for t in trans)

    rots, mirrors, glides = {}, {}, {}
    pad = 1e-6
    for Mf, t in ops:
        M = A @ Mf @ Ainv
        det = np.linalg.det(M)
        if det > 0 and np.allclose(M, np.eye(2)):
            continue
        for L in itertools.product(range(-K, K + 1), repeat=2):
            # (I - Mf) c0 is an integer vector: same operation up to a lattice translation, with its
            # fixed point or axis moved by A c0, the centre of the window
            S = A @ (t + np.array(L) + (np.eye(2) - Mf) @ c0)
            if det > 0:
                c = np.linalg.solve(np.eye(2) - M, S)
                if xmin - pad <= c[0] <= xmax + pad and ymin - pad <= c[1] <= ymax + pad:
                    ang = math.atan2(M[1, 0], M[0, 0])
                    order = int(round(2 * math.pi / abs(ang)))
                    key = (round(c[0], 5), round(c[1], 5))
                    rots[key] = max(rots.get(key, 0), order)
            else:
                w, V = np.linalg.eigh(M)
                u = V[:, np.argmax(w)]
                g = (S @ u) * u
                p0 = (S - g) / 2
                nrm = np.array([-u[1], u[0]])
                ang = round(math.atan2(u[1], u[0]) % math.pi, 5)
                off = round(p0 @ nrm, 5) if ang < math.pi - 1e-5 else round(-(p0 @ nrm), 5)
                key = (ang % round(math.pi, 5), off)
                if in_T(Ainv @ g):
                    mirrors[key] = (p0, u)
                else:
                    glides.setdefault(key, (p0, u, g))
    for k in list(glides):
        if k in mirrors:
            del glides[k]
    return dict(rotations=[(o, np.array(k)) for k, o in rots.items()],
                mirrors=list(mirrors.values()), glides=list(glides.values()))


def plot_symmetry_elements(cell, ax, bounds, color="#444444", use_types=True):
    """Overlay mirrors (solid), glides (dashed) and rotation centres (2: circle, 3: triangle,
    4: square, 6: hexagon) of the infinite lattice on ax, inside bounds = (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = bounds
    el = symmetry_elements(cell, bounds, use_types)

    def clip(p0, u):
        ts = []
        for k, (lo, hi) in enumerate(((xmin, xmax), (ymin, ymax))):
            if abs(u[k]) < 1e-12:
                if not lo - 1e-9 <= p0[k] <= hi + 1e-9:
                    return None
                continue
            ts.append(sorted(((lo - p0[k]) / u[k], (hi - p0[k]) / u[k])))
        if not ts:
            return None
        t0 = max(t[0] for t in ts); t1 = min(t[1] for t in ts)
        return None if t0 >= t1 else (p0 + t0 * u, p0 + t1 * u)

    for p0, u in el["mirrors"]:
        seg = clip(p0, u)
        if seg:
            ax.plot(*np.array(seg).T, color=color, lw=0.9, alpha=0.6, zorder=0.5)
    for p0, u, g in el["glides"]:
        seg = clip(p0, u)
        if seg:
            ax.plot(*np.array(seg).T, color=color, lw=0.9, ls=(0, (4, 3)), alpha=0.6, zorder=0.5)
    marker = {2: dict(marker="o", s=22), 3: dict(marker="^", s=40), 4: dict(marker="s", s=34),
              6: dict(marker="h", s=50)}
    for o, c in el["rotations"]:
        ax.scatter(*c, color=color, zorder=5, edgecolors="white", linewidths=0.6, **marker[o])
    return el
