"""Wyckoff positions of the 230 space groups, derived from the stored operations (C1 WP-P0).

Source data: `data/space_groups_230.json` (one setting per group: `hall_number`, coset representatives `ops`
of G over its lattice of translations, and `tb_rows`, the primitive translations in conventional fractional
coordinates). Nothing here is copied from a Wyckoff table.

Method (exact up to a floating tolerance of 1e-7 on fractional coordinates):
  * the full operation set mod Z^3 is ops x centring translations;
  * a special position is the fixed-point set Fix(S) of a site-symmetry group S. Starting from the general
    position R^3, every subspace X is intersected with every fixed set Fix(g, n) = {x : R x + t + n = x}
    (g = (R, t) a full operation, n in Z^3); a proper intersection Y is a smaller special subspace, and every
    Fix(S) = Fix(g1) n Fix(g2) n ... is reached this way. Subspaces are merged by G-orbit (mod Z^3);
  * a generic point of Y has stabiliser S_Y, and Y = Fix(S_Y) (an operation fixing a generic point of Y fixes
    all of Y), so each orbit of subspaces is one Wyckoff position, with multiplicity |G_conv| / |S_Y|.
The positions are then labelled with spglib's Wyckoff letters (`label_with_spglib`, run in the same setting via
`hall_number`) and checked against it: one letter per position, all letters from 'a' up to the letter of the
general position, and |S| equal to the order of spglib's site-symmetry symbol (an independent table below).
"""
from __future__ import annotations

import itertools
import json
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

TOL = 1e-7
_DATA = Path(__file__).resolve().parent / "data" / "space_groups_230.json"
_N1 = np.array(list(itertools.product((-1, 0, 1), repeat=3)), float)
# generic coefficients for a "generic point" of a subspace (irrational-looking, fixed for reproducibility)
_GENERIC = np.array([0.1370982, 0.2928571, 0.4142136])


@lru_cache(maxsize=1)
def _raw():
    return {g["number"]: g for g in json.loads(_DATA.read_text(encoding="utf-8"))["groups"]}


def group_entry(number):
    try:
        return _raw()[int(number)]
    except KeyError:
        raise ValueError(f"unknown space group #{number} (expected 1..=230)") from None


def _frac(q):
    return Fraction(int(q[0]), int(q[1]))


def primitive_basis(g):
    """Rows: the primitive translations in conventional fractional coordinates (exact Fractions)."""
    return [[_frac(q) for q in row] for row in g["tb_rows"]]


def centrings(g):
    """The translations of the lattice in [0,1)^3 (conventional coordinates), as float rows."""
    B = np.array([[float(x) for x in row] for row in primitive_basis(g)])
    k = int(round(1 / abs(np.linalg.det(B))))
    out = []
    for c in itertools.product(range(-3, 4), repeat=3):
        v = np.array(c, float) @ B
        v = v - np.floor(v + 1e-9)
        if not any(np.allclose(v, w, atol=1e-9) for w in out):
            out.append(v)
    if len(out) != k:
        raise ValueError(f"centring search found {len(out)} translations, expected {k}")
    return np.array(sorted(out, key=tuple))


@lru_cache(maxsize=256)
def full_ops(number):
    """(R (n,3,3) int, t (n,3) float in [0,1)): every operation mod Z^3 in the conventional cell."""
    g = group_entry(number)
    Rs, ts = [], []
    for op in g["ops"]:
        R = np.array(op["r"], float)
        t = np.array([float(_frac(q)) for q in op["t"]])
        for c in centrings(g):
            Rs.append(R); ts.append((t + c) % 1.0)
    return np.array(Rs), np.array(ts) % 1.0


def orbit(number, p, tol=1e-6):
    """The G-orbit of the fractional point p in the conventional cell (rows in [0,1))."""
    R, t = full_ops(number)
    P = (np.einsum("nij,j->ni", R, np.asarray(p, float)) + t) % 1.0
    out = []
    for q in P:
        if not any(_same_mod1(q, w, tol) for w in out):
            out.append(q)
    return np.array(out)


def _same_mod1(a, b, tol=1e-6):
    d = np.asarray(a) - np.asarray(b)
    return bool(np.all(np.abs(d - np.round(d)) < tol))


def stabiliser(number, p, tol=1e-6):
    """Indices of the full operations fixing p mod Z^3."""
    R, t = full_ops(number)
    d = np.einsum("nij,j->ni", R, np.asarray(p, float)) + t - p
    return np.where(np.all(np.abs(d - np.round(d)) < tol, axis=1))[0]


def _orth(D):
    """Orthonormal basis (3, d) of the column span of D (d = 0 allowed)."""
    if D.shape[1] == 0:
        return D
    U, s, _ = np.linalg.svd(D, full_matrices=False)
    return U[:, s > 1e-9]


def _int_dir(D):
    """A basis of integer vectors (columns) spanning the rational subspace spanned by D."""
    D = _orth(D)
    d = D.shape[1]
    if d == 0:
        return D
    # reduced row echelon form of the projector's row space, rationalised
    P = D @ D.T
    best = []
    for v in itertools.product(range(-2, 3), repeat=3):
        v = np.array(v, float)
        if not v.any() or np.linalg.norm(P @ v - v) > 1e-9:
            continue
        cand = best + [v]
        if np.linalg.matrix_rank(np.array(cand).T, tol=1e-9) == len(cand):
            best = cand
        if len(best) == d:
            break
    if len(best) == d:
        return np.array(best).T
    # fallback: rationalise the projected unit vectors
    out = []
    for e in np.eye(3):
        w = P @ e
        if np.linalg.norm(w) < 1e-9:
            continue
        fr = [Fraction(x).limit_denominator(24) for x in w]
        den = int(np.lcm.reduce([f.denominator for f in fr]))
        out.append(np.array([float(f * den) for f in fr]))
        if np.linalg.matrix_rank(np.array(out).T, tol=1e-9) < len(out):
            out.pop()
        if len(out) == d:
            break
    if len(out) != d:
        raise ValueError("cannot find an integer basis of a special subspace")
    return np.array(out).T


def _point_on(p0, D):
    return p0 + D @ _GENERIC[: D.shape[1]]


def _split(R, t, p, D):
    """Sub-subspaces of X = p + span D fixed by (R, t + n), n in {-1,0,1}^3; [] if (R, t) fixes all of X."""
    M = (np.eye(3) - R) @ D
    rhs = t[None, :] + _N1 - ((np.eye(3) - R) @ p)[None, :]
    if D.shape[1] == 0:
        return []
    if np.linalg.norm(M) < 1e-12:
        return []
    Mp = np.linalg.pinv(M, rcond=1e-10)
    U = rhs @ Mp.T
    ok = np.linalg.norm(U @ M.T - rhs, axis=1) < 1e-8
    if not ok.any():
        return []
    # null space of M inside the coordinates of D
    _, s, Vt = np.linalg.svd(M)
    rank = int((s > 1e-9).sum())
    N = Vt[rank:].T                                    # (d, d - rank)
    out = []
    for u in U[ok]:
        q = p + D @ u
        E = _orth(D @ N) if N.shape[1] else np.zeros((3, 0))
        out.append((q % 1.0, E))
    return out


def _prim_int(v):
    v = np.array([int(round(x)) for x in v])
    g = int(np.gcd.reduce(np.abs(v)))
    v = v // g
    nz = v[np.nonzero(v)[0][0]]
    return v if nz > 0 else -v


@lru_cache(maxsize=64)
def _line_covectors(d):
    """Integer covectors n1, n2 with n.d = 0 and n1 x n2 = +-d: a basis of the integer covectors vanishing on
    the primitive direction d, so that x -> (n1.x, n2.x) maps Z^3 onto Z^2 (first pair in a fixed order)."""
    d = np.array(d)
    vs = [np.array(v) for v in itertools.product(range(-3, 4), repeat=3) if any(v) and np.dot(v, d) == 0]
    vs.sort(key=lambda v: (np.abs(v).sum(), tuple(v)))
    for a, b in itertools.combinations(vs, 2):
        c = np.cross(a, b)
        if np.array_equal(c, d) or np.array_equal(c, -d):
            return a, b
    raise ValueError(f"no covector basis for direction {d}")


def _r6(x):
    return round(float(x) % 1.0, 6) % 1.0


def _key(p, D):
    """A canonical key of the affine subspace p + span D mod Z^3 (D with rational span)."""
    d = D.shape[1]
    if d == 3:
        return ("general",)
    if d == 0:
        return (0,) + tuple(_r6(x) for x in p)
    B = _int_dir(D)
    if d == 2:
        n = _prim_int(np.cross(B[:, 0], B[:, 1]))
        return (2, tuple(n), _r6(n @ p))
    u = tuple(_prim_int(B[:, 0]))
    n1, n2 = _line_covectors(u)
    return (1, u, _r6(n1 @ p), _r6(n2 @ p))


@lru_cache(maxsize=256)
def _positions_raw(number):
    R, t = full_ops(number)
    n_full = len(R)
    seen, reps = set(), []

    def add(p, D):
        reps.append((p, D))
        for k in range(n_full):
            seen.add(_key(R[k] @ p + t[k], _orth(R[k] @ D) if D.shape[1] else D))

    add(np.zeros(3), np.eye(3))
    frontier = [reps[0]]
    while frontier:
        new = []
        for p, D in frontier:
            for k in range(n_full):
                for q, E in _split(R[k], t[k], p, D):
                    if E.shape[1] >= D.shape[1] or _key(q, E) in seen:
                        continue
                    add(q, E)
                    new.append((q, E))
        frontier = new
    out = []
    for p, D in reps:
        x = _point_on(p, D)
        S = stabiliser(number, x)
        out.append(dict(p0=p % 1.0, D=_int_dir(D) if D.shape[1] else np.zeros((3, 0)), dim=D.shape[1],
                        stab=len(S), mult_conv=n_full // len(S)))
    return out


KIND = {0: "point", 1: "line", 2: "plane", 3: "general"}


@lru_cache(maxsize=256)
def wyckoff_positions(number):
    """The Wyckoff positions of space group `number` in the stored setting, ordered by spglib letter (a first).
    Each is a dict: kind (point/line/plane/general), p0 (fractional), D (3 x dim integer directions),
    mult_conv (sites per conventional cell), mult (sites per primitive cell), stab (|site-symmetry group|),
    letter, site_symmetry (spglib's symbol), name ('4a', ...)."""
    raw = _positions_raw(number)
    n_c = len(centrings(group_entry(number)))
    lab = label_with_spglib(number, raw)
    out = []
    for w, (letter, sym) in zip(raw, lab):
        out.append(dict(kind=KIND[w["dim"]], p0=w["p0"], D=w["D"], dim=w["dim"], mult_conv=w["mult_conv"],
                        mult=w["mult_conv"] // n_c, stab=w["stab"], letter=letter, site_symmetry=sym,
                        name=f"{w['mult_conv']}{letter}"))
    out.sort(key=lambda w: _letter_rank(w["letter"]))
    return tuple(out)


def _letter_rank(c):
    return ord(c) - ord("a") if c.islower() else 26 + ord(c) - ord("A")


def system_metric(system, rng=None):
    """A cell (rows a, b, c, Cartesian) with the metric freedom of the crystal system, in the stored settings
    (monoclinic unique axis b; trigonal and hexagonal on hexagonal axes), with a = 1. Axial ratios b/a, c/a are
    log-uniform in [1/2, 2]; triclinic angles uniform in [70, 110] degrees, the monoclinic beta in [90, 125].
    With rng=None a fixed generic metric. Returns (A, free) with free the sampled metric parameters."""
    r = rng.random(5) if rng is not None else np.array([0.37, 0.61, 0.29, 0.53, 0.71])
    b, c = 2.0 ** (2 * r[0] - 1), 2.0 ** (2 * r[1] - 1)
    al, be, ga = 70 + 40 * r[2], 70 + 40 * r[3], 70 + 40 * r[4]
    if system == "triclinic":
        free = dict(b=b, c=c, alpha=al, beta=be, gamma=ga)
    elif system == "monoclinic":
        free = dict(b=b, c=c, beta=90 + 35 * r[3])
    elif system == "orthorhombic":
        free = dict(b=b, c=c)
    elif system in ("tetragonal", "trigonal", "hexagonal"):
        free = dict(c=c)
    elif system == "cubic":
        free = {}
    else:
        raise ValueError(f"unknown crystal system {system!r}")
    hexa = system in ("trigonal", "hexagonal")
    prm = dict(b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=120.0 if hexa else 90.0)
    prm.update(free)
    if system in ("tetragonal", "trigonal", "hexagonal"):
        prm["b"] = 1.0
    return cell_matrix(1.0, prm["b"], prm["c"], prm["alpha"], prm["beta"], prm["gamma"]), free


def cell_matrix(a, b, c, alpha, beta, gamma):
    al, be, ga = np.radians([alpha, beta, gamma])
    A = np.zeros((3, 3))
    A[0] = [a, 0, 0]
    A[1] = [b * np.cos(ga), b * np.sin(ga), 0]
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    A[2] = [cx, cy, np.sqrt(c * c - cx * cx - cy * cy)]
    return A


# Orders of the crystallographic point groups by Hermann-Mauguin symbol (dots of the oriented site-symmetry
# symbol removed). Independent of the code above: textbook orders.
_PG_ORDER = {
    "1": 1, "-1": 2, "2": 2, "m": 2, "2/m": 4, "222": 4, "mm2": 4, "m2m": 4, "2mm": 4, "mmm": 8,
    "4": 4, "-4": 4, "4/m": 8, "422": 8, "4mm": 8, "-42m": 8, "-4m2": 8, "4/mmm": 16,
    "3": 3, "-3": 6, "32": 6, "321": 6, "312": 6, "3m": 6, "3m1": 6, "31m": 6, "-3m": 12, "-3m1": 12, "-31m": 12,
    "6": 6, "-6": 6, "6/m": 12, "622": 12, "6mm": 12, "-6m2": 12, "-62m": 12, "6/mmm": 24,
    "23": 12, "m-3": 24, "432": 24, "-43m": 24, "m-3m": 48,
}


def pg_order(symbol):
    s = symbol.replace(".", "")
    if s not in _PG_ORDER:
        raise ValueError(f"unknown site-symmetry symbol {symbol!r}")
    return _PG_ORDER[s]


def label_with_spglib(number, raw):
    """spglib's (letter, site-symmetry symbol) for each raw position, with the checks of the module docstring.
    The cell holds the position's orbit (species 1) and two generic general-position orbits (species 2, 3),
    which pin the symmetry to G."""
    import spglib
    g = group_entry(number)
    A, _ = system_metric(g["system"])
    rng = np.random.default_rng(7)
    extra = [orbit(number, rng.random(3)) for _ in range(2)]
    out = []
    for w in raw:
        pts = orbit(number, _point_on(w["p0"], w["D"]))
        F = np.vstack([pts] + extra)
        types = [1] * len(pts) + [2] * len(extra[0]) + [3] * len(extra[1])
        ds = spglib.get_symmetry_dataset((A, F, types), symprec=1e-5, hall_number=g["hall_number"])
        if ds is None or ds.number != number:
            raise ValueError(f"#{number}: spglib does not find the group for a test cell")
        letters = {ds.wyckoffs[i] for i in range(len(pts))}
        syms = {ds.site_symmetry_symbols[i] for i in range(len(pts))}
        if len(letters) != 1 or len(syms) != 1:
            raise ValueError(f"#{number}: one orbit got several Wyckoff letters {letters}")
        letter, sym = letters.pop(), syms.pop()
        if pg_order(sym) != w["stab"]:
            raise ValueError(f"#{number} {letter}: |S| = {w['stab']}, spglib site symmetry {sym}")
        out.append((letter, sym))
    letters = [x[0] for x in out]
    gen = [x[0] for x, w in zip(out, raw) if w["dim"] == 3][0]
    if sorted(letters, key=_letter_rank) != [_rank_letter(i) for i in range(_letter_rank(gen) + 1)]:
        raise ValueError(f"#{number}: positions {sorted(letters)} do not match spglib's letters a..{gen}")
    return out


def _rank_letter(i):
    return chr(ord("a") + i) if i < 26 else chr(ord("A") + i - 26)


def sample_position(w, rng):
    """A random point of Wyckoff position w (fractional, conventional cell) and its free parameters."""
    if w["dim"] == 0:
        return np.array(w["p0"], float), ()
    u = rng.random(w["dim"])
    return (w["p0"] + w["D"] @ u) % 1.0, tuple(float(x) for x in u)
