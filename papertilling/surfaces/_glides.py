"""Mirror and glide classes of a decorated lattice, found by a complete search on the Gauss-reduced cell (C1 WP-B1).

Mirror of `rust/papertilling-core/src/surface/glides.rs` (same candidate order, translations and axis vectors).
Decorated: sites, bonds and bond kinds; a reflection that swaps two bond kinds is not a symmetry.

Hypothesis: the cell is primitive for its decoration, so every symmetry of the pattern keeps the cell lattice. On a
non-primitive cell a mirror can move the cell lattice and be missed, so an empty search there raises SurfaceNotFound
(NON_PRIMITIVE), never a proof.

Why the search is complete [PROVED] (the proof is written out in glides.rs): for a Gauss-reduced basis (|b1| <= |b2|,
2|b1.b2| <= |b1|^2) every linear part of a lattice symmetry has integer entries in [-1, 1] (the scan uses [-2, 2]); the
translation part is fixed modulo the lattice by the image of site 0, and every image is tried; the axis vectors have
reduced coefficients in [-2, 2] (scan [-4, 4]). So on a primitive cell an empty list proves that the decorated lattice has no mirror and no
glide line.
"""
from __future__ import annotations

import numpy as np

_HEX_PRIO = [((0, -1), (-1, 0)), ((-1, 1), (0, 1)), ((1, 0), (1, -1)),
             ((0, 1), (1, 0)), ((1, -1), (0, -1)), ((-1, 0), (-1, 1))]
_STD_PRIO = [((-1, 0), (0, 1)), ((1, 0), (0, -1)), ((0, 1), (1, 0)), ((0, -1), (-1, 0))]


def gauss_reduce(A):
    """Unimodular integer U (det +1; columns = reduced basis in cell units) with A @ U Gauss (Lagrange) reduced."""
    A = np.asarray(A, float)

    def dot(u, v):
        return float((A @ np.asarray(u, float)) @ (A @ np.asarray(v, float)))
    b1, b2 = [1, 0], [0, 1]
    for _ in range(200):
        if dot(b2, b2) < dot(b1, b1):
            b1, b2 = b2, b1
        mu = int(np.round(dot(b1, b2) / dot(b1, b1)))          # numpy rounds half to even, as Rust round_ties_even
        if mu == 0:
            break
        b2 = [b2[0] - mu * b1[0], b2[1] - mu * b1[1]]
    if dot(b2, b2) < dot(b1, b1) * (1 - 1e-9):
        b1, b2 = b2, b1
    U = np.array([[b1[0], b2[0]], [b1[1], b2[1]]], int)
    if round(np.linalg.det(U)) < 0:
        U[:, 1] = -U[:, 1]
    return U


def _wrap01(x):
    w = x - np.floor(x + 1e-7)
    return 0.0 if abs(w) < 1e-7 else float(w)


def _apply(M, t, F, k):
    g = M @ F[k] + t
    for j, fj in enumerate(F):
        d = g - fj
        r = np.round(d)
        if abs(d[0] - r[0]) < 1e-6 and abs(d[1] - r[1]) < 1e-6:
            return j, r.astype(int)
    return None


NON_PRIMITIVE = ("NOT FOUND: the cell is not primitive for its decoration (a translation by a fraction of the cell maps "
                 "the pattern onto itself), and the mirror/glide search is complete only for reflections that keep the "
                 "cell lattice; none was found, which proves nothing here. Use the primitive cell for a proof")


def _is_symmetry(M, t, F, bonds, darts):
    """Whether x -> M x + t (fractional) maps the decorated cell onto itself (sites, bonds and bond kinds)."""
    smap = [_apply(M, t, F, k) for k in range(len(F))]
    if any(s is None for s in smap):
        return False
    for i, j, n1, n2, kind in bonds:
        (ji, ci), (jj, cj) = smap[i], smap[j]
        mn = M @ np.array([n1, n2])
        if (ji, jj, int(mn[0] + cj[0] - ci[0]), int(mn[1] + cj[1] - ci[1]), kind) not in darts:
            return False
    return True


def glide_search(A, basis, bonds):
    """Every mirror/glide class (one per linear part) of the cell (A: columns a1, a2; basis: Cartesian sites; bonds:
    (i, j, n1, n2, kind)), as dicts M, t, tau, along, across, half, length, mirror. Raises ValueError when an axis
    vector is missing (never expected; never a silent drop), and SurfaceNotFound(NON_PRIMITIVE) when nothing is found on
    a cell that is not primitive for its decoration."""
    A = np.asarray(A, float)
    Ainv = np.linalg.inv(A)
    F = [Ainv @ np.asarray(p, float) for p in basis]
    bonds = [(int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(b[4]) if len(b) > 4 else 0) for b in bonds]
    darts = set()
    for i, j, n1, n2, k in bonds:
        darts.add((i, j, n1, n2, k))
        darts.add((j, i, -n1, -n2, k))
    a1, a2 = A[:, 0], A[:, 1]
    g00, g01, g11 = float(a1 @ a1), float(a1 @ a2), float(a2 @ a2)
    tol = 1e-5 * (g00 + g11)
    l1, l2 = float(np.hypot(*a1)), float(np.hypot(*a2))
    is_hex = abs(l1 - l2) <= 1e-5 * max(l1, l2) and abs(g01 / (l1 * l2) + 0.5) <= 1e-5
    prio = _HEX_PRIO if is_hex else _STD_PRIO
    U = gauss_reduce(A)
    Uinv = np.array([[U[1, 1], -U[0, 1]], [-U[1, 0], U[0, 0]]], int)
    cands, seen = [], set()
    rng = range(-2, 3)
    for m00 in rng:
        for m01 in rng:
            for m10 in rng:
                for m11 in rng:
                    if m00 * m11 - m01 * m10 != -1:
                        continue
                    M = U @ np.array([[m00, m01], [m10, m11]], int) @ Uinv
                    key = ((int(M[0, 0]), int(M[0, 1])), (int(M[1, 0]), int(M[1, 1])))
                    if key in seen:
                        continue
                    seen.add(key)
                    c0, c1 = A @ M[:, 0], A @ M[:, 1]
                    if abs(c0 @ c0 - g00) <= tol and abs(c0 @ c1 - g01) <= tol and abs(c1 @ c1 - g11) <= tol:
                        cands.append((prio.index(key) if key in prio else 100, key))
    cands.sort()
    vecs = [tuple(int(x) for x in U @ np.array([a, b])) for a in range(-4, 5) for b in range(-4, 5) if a or b]
    vecs.sort(key=lambda v: (float(np.hypot(*(A @ np.array(v, float)))), v))
    out = []
    for _, key in cands:
        M = np.array(key, int)
        mf0 = M @ F[0]
        t = None
        for j0 in range(len(F)):
            tt = np.array([_wrap01(F[j0][0] - mf0[0]), _wrap01(F[j0][1] - mf0[1])])
            if _is_symmetry(M, tt, F, bonds, darts):
                t = tt
                break
        if t is None:
            continue
        along = next((np.array(v) for v in vecs if tuple(M @ np.array(v)) == v), None)
        across = next((np.array(v) for v in vecs if tuple(M @ np.array(v)) == tuple(-x for x in v)), None)
        if along is None or across is None:
            raise ValueError(f"NOT FOUND: no axis vector found for the reflection {key} (please report this cell)")
        tau = M @ t + t
        ac = A @ along
        alpha = float((A @ tau) @ ac / (ac @ ac))
        if abs(alpha) >= 1.5:
            t = t - np.round(0.5 * alpha) * along
            tau = M @ t + t
        mirror = bool(abs(tau[0]) < 1e-6 and abs(tau[1]) < 1e-6)
        if mirror:
            t = t + along
            tau = M @ t + t
        tau = np.round(tau).astype(int)
        tc = A @ tau
        out.append(dict(M=M, t=np.asarray(t, float), tau=tau, along=along, across=across, half=tc / 2.0,
                        length=float(np.hypot(*tc)) / 2.0, mirror=mirror))
    if not out:
        I2 = np.eye(2, dtype=int)
        if any(_is_symmetry(I2, np.array([_wrap01(F[j][0] - F[0][0]), _wrap01(F[j][1] - F[0][1])]), F, bonds, darts)
               for j in range(1, len(F))):
            from .refusal import SurfaceNotFound
            raise SurfaceNotFound(NON_PRIMITIVE, retry=False)
    return out


def glide_power(g, n):
    """The class element closing a band after n steps (glides.rs::glide_power): a mirror class takes any n >= 1 (the
    mirror followed by n axis translations); a glide class with no mirror only odd n (g tau^k squares to (2k+1) tau).
    Returns a copy of g with t shifted; raises ValueError otherwise."""
    n = int(n)
    if n < 1:
        raise ValueError(f"the number of glide steps must be >= 1 (got {n})")
    if g["mirror"]:
        shift = (n - 1) * np.asarray(g["along"])
    elif n % 2 == 1:
        shift = (n - 1) // 2 * np.asarray(g["tau"])
    else:
        raise ValueError(f"this class is a glide with no mirror: it closes only after an odd number of steps (g followed "
                         f"by k periods squares to (2k+1) periods), so {n} steps are refused; pick an odd number or a "
                         "mirror class")
    out = dict(g)
    out["t"] = np.asarray(g["t"], float) + shift
    return out
