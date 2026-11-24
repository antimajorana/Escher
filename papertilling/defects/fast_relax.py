"""Faster drawing relaxation for build(picture="relaxed"). The drawing is a picture only: no graph
observable of the package reads positions (charges, cores, core graphs use bond data and the
reference positions). Definition: the energy is that of legacy fusion_framework.relax_picture,
F(z) = 1/2 |r(z)|^2 with r = (spring residuals, triangle-area residuals, anchor (z - X0)).

  relax_setup(d)       springs, triangles and anchor, built exactly as in legacy relax_picture
  residual(S, z), jacobian(S, z)   the legacy fun / jac
  relax_reference(d)   scipy least_squares(trf, lsmr, max_nfev=steps) on these: the legacy algorithm,
                       used by the tests to check that the setup copy reproduces legacy positions
  relax_picture(d)     fast: Gauss-Newton (Levenberg-Marquardt damping only when a step fails) with a
                       sparse direct solve (SuperLU) of the normal equations, from the same start X0;
                       the factor is reused once steps become small (chord steps)

Why another minimiser is acceptable (technical choice; checked in tests/test_build.py):
the positions are a drawing only, and the legacy run does not stop at the minimum either: scipy stops
on ftol (relative cost change < 1e-8 in one step) with first-order optimality ~1e-4 (computational
evidence, 4-disclination builds on square, shastry_sutherland, breathing_kagome). The energy is very
flat (anchor weight 1e-3): two stopping points with costs equal to 1e-7 relative can differ by ~0.1
in position, so positions cannot be compared; costs can. The tests assert the same graph and
|cost_fast - cost_legacy| <= 1e-6 cost_legacy. Measured with this gauss_newton (2026-11-13, 4-disclination
builds): (fast - legacy)/legacy = -8.2e-9 square (659 sites), -5.7e-8 shastry_sutherland (1472),
-3.6e-7 breathing_kagome (1301); largest position difference 0.009, 0.035, 0.10.
picture="raw" never calls this module.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ..legacy import fusion_framework as ff

TWO_PI = 2 * math.pi
_n2 = ff._norm2


def relax_setup(d) -> dict:
    """The data of legacy relax_picture (same loops, same order)."""
    n = len(d.sites)
    X0 = np.asarray(d.info.get("guess", d.positions), float)
    non_orient = bool(d.info.get("mobius") or d.info.get("klein"))

    def wrap(u, v):
        if not non_orient:
            return False
        e = d.edge[(u, v)]
        return (ff._det_sign_iso(e.get("g")) < 0) or (_n2(X0[v] - X0[u]) > 3.2 * max(_n2(e["vec"]), 1e-6))

    springs = {}
    for u, v in d.edges:
        if not wrap(u, v):
            springs[(u, v)] = (_n2(d.edge[(u, v)]["vec"]), 1.0)
    tri = []
    for u, lst in ff._corners(d).items():
        unrolled = any("ang" in d.edge[(u, v)] for _, v in lst)
        pairs = list(zip(lst[:-1], lst[1:])) + ([] if unrolled or len(lst) < 3 else [(lst[-1], lst[0])])
        for (a1, v), (a2, w) in pairs:
            if wrap(u, v) or wrap(u, w):
                continue
            gap = (a2 - a1) if unrolled else (a2 - a1) % TWO_PI
            if v == w or not (1e-6 < gap < math.pi - 1e-6):
                continue
            key = (min(v, w), max(v, w))
            if key not in springs:
                springs[key] = (_n2(d.edge[(u, w)]["vec"] - d.edge[(u, v)]["vec"]), 0.5)
            a0 = ff._cross2v(d.edge[(u, v)]["vec"], d.edge[(u, w)]["vec"])
            if a0 > 1e-9:
                tri.append((u, v, w, a0))
    S = dict(n=n, X0=X0, anchor=0.05 if non_orient else 1e-3)
    S["tu"] = np.array([t[0] for t in tri], int)
    S["tv"] = np.array([t[1] for t in tri], int)
    S["tw"] = np.array([t[2] for t in tri], int)
    S["ta"] = np.array([t[3] for t in tri], float)
    S["tk"] = len(tri)
    S["tsc"] = 0.7 / np.sqrt(S["ta"]) if S["tk"] else S["ta"]
    S["uu"] = np.array([k[0] for k in springs], int)
    S["vv"] = np.array([k[1] for k in springs], int)
    S["rest"] = np.array([s[0] for s in springs.values()], float)
    S["wt"] = np.array([s[1] for s in springs.values()], float)
    S["m"] = len(S["rest"])
    return S


def residual(S, z):
    P = z.reshape(-1, 2)
    dv = P[S["uu"]] - P[S["vv"]]
    e1 = P[S["tv"]] - P[S["tu"]]
    e2 = P[S["tw"]] - P[S["tu"]]
    area = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    return np.concatenate([S["wt"] * (np.linalg.norm(dv, axis=1) - S["rest"]), S["tsc"] * (area - S["ta"]),
                           S["anchor"] * (z - S["X0"].ravel())])


def jacobian(S, z):
    n, m, tk = S["n"], S["m"], S["tk"]
    uu, vv, tu, tv, tw = S["uu"], S["vv"], S["tu"], S["tv"], S["tw"]
    P = z.reshape(-1, 2)
    dv = P[uu] - P[vv]
    ln = np.maximum(np.linalg.norm(dv, axis=1), 1e-12)
    gx = S["wt"][:, None] * dv / ln[:, None]
    cols = np.column_stack([2 * uu, 2 * uu + 1, 2 * vv, 2 * vv + 1]).ravel()
    vals = np.column_stack([gx[:, 0], gx[:, 1], -gx[:, 0], -gx[:, 1]]).ravel()
    J = sp.coo_matrix((vals, (np.repeat(np.arange(m), 4), cols)), shape=(m, 2 * n))
    e1 = P[tv] - P[tu]
    e2 = P[tw] - P[tu]
    dvx, dvy = e2[:, 1], -e2[:, 0]
    dwx, dwy = -e1[:, 1], e1[:, 0]
    tcols = np.column_stack([2 * tv, 2 * tv + 1, 2 * tw, 2 * tw + 1, 2 * tu, 2 * tu + 1]).ravel()
    tvals = (S["tsc"][:, None] * np.column_stack([dvx, dvy, dwx, dwy, -dvx - dwx, -dvy - dwy])).ravel() if tk else np.zeros(0)
    Jt = sp.coo_matrix((tvals, (np.repeat(np.arange(tk), 6), tcols)), shape=(tk, 2 * n))
    return sp.vstack([J, Jt, S["anchor"] * sp.identity(2 * n)]).tocsr()


def cost(S, z) -> float:
    r = residual(S, z)
    return 0.5 * float(r @ r)


def relax_reference(d, steps: int = 80, **tol):
    """The legacy algorithm on relax_setup's data: scipy least_squares(trf, lsmr). Returns (z, result)."""
    from scipy.optimize import least_squares
    S = relax_setup(d)
    sol = least_squares(lambda z: residual(S, z), S["X0"].ravel(), jac=lambda z: jacobian(S, z), method="trf",
                        tr_solver="lsmr", max_nfev=steps, **tol)
    return sol.x, sol


def gauss_newton(S, z0, max_iter: int = 500, rtol: float = 1e-10, freeze: float = 1e-3):
    """Minimise 1/2 |r|^2 by Gauss-Newton steps (J^T J + mu D) dz = -J^T r, D = diag(J^T J), sparse LU.
    mu = 0 unless a step raises the cost (then mu = 1e-4, x 10 until accepted, x 0.1 after each accepted
    step, reset to 0 below 1e-10). Once a step lowers the cost by less than `freeze` (relative), the last
    LU factor is reused (chord steps; refactored whenever a chord step raises the cost). Stops when an
    accepted step lowers the cost by less than rtol (relative). Returns (z, cost, steps, factorisations)."""
    z = z0.copy()
    r = residual(S, z)
    F = 0.5 * float(r @ r)
    mu, lu, frozen, nfact, it = 0.0, None, False, 0, 0
    for it in range(1, max_iter + 1):
        J = jacobian(S, z)
        JT = J.T.tocsr()
        g = JT @ r
        if lu is None or not frozen:
            A = (JT @ J).tocsc()
            D = A.diagonal()
        while True:
            if lu is None or not frozen:
                M = A if mu == 0.0 else (A + sp.diags(mu * D)).tocsc()
                # Use the default COLAMD ordering to avoid a SuperLU crash in SciPy 1.10.
                lu = splu(M)
                nfact += 1
            dz = lu.solve(-g)
            r2 = residual(S, z + dz)
            F2 = 0.5 * float(r2 @ r2)
            if F2 <= F:
                break
            if frozen:                              # stale factor: refactor at the current point
                frozen = False
                A = (JT @ J).tocsc()
                D = A.diagonal()
                continue
            mu = 1e-4 if mu == 0.0 else 10.0 * mu
            if mu > 1e12:
                return z, F, it, nfact
        drop = F - F2
        z, r, F = z + dz, r2, F2
        mu = 0.0 if mu < 1e-10 else 0.1 * mu
        if drop <= rtol * F:
            break
        if drop <= freeze * F and mu == 0.0:
            frozen = True
    return z, F, it, nfact


def relax_picture(d, steps: int = 80, fast: bool | None = None):
    """Same effect as legacy fusion_framework.relax_picture (drawing positions, picture flag, long-bond
    marks); fast=False (or PAPERTILLING_LEGACY=1) runs the legacy function itself."""
    from .fast import enabled
    if not enabled(fast):
        return ff.relax_picture(d, steps=steps)
    S = relax_setup(d)
    if S["m"]:
        d.positions = gauss_newton(S, S["X0"].ravel())[0].reshape(-1, 2)
    d._nbr = None
    d.info["picture"] = "relaxed"
    return ff.mark_long_bonds(d)
