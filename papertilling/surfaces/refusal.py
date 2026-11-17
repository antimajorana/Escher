"""Why a lattice was not put on a surface: IMPOSSIBLE (proved from the lattice's symmetry) or NOT FOUND (this construction
refused at the settings tried; another size, centre or construction may work).

    SurfaceRefused(ValueError)          base class (pieces.ConeRefused is the same class, kept for old code)
      SurfaceImpossible                 proved: no surface of this kind carries the lattice (reason in words)
      SurfaceNotFound                   the construction refused at these settings; e.hint says what to try, e.tried
                                        lists what was tried

THE THREE PROOFS USED (proved results, under the model of every surface in this package: the surface is the plane lattice,
glued by symmetries of the lattice, away from finitely many cone points (disclinations)). Carry the lattice along a closed
loop that avoids the cones: it comes back moved by an isometry h of the plane (the holonomy), and h must map the lattice
onto itself, so h is in the lattice's symmetry group G.
  (a) [scope="group"] A loop along which the surface's orientation reverses has an orientation-reversing h (a reflection or a
      glide). If G has no mirror and no glide line (p1, p2, p3, p4, p6), no non-orientable surface carries the lattice: no
      Moebius band, no Klein bottle, no projective plane.
  (b) [scope="group"] Around a cone point of total angle alpha, h is the rotation by alpha about that point, so alpha is a
      multiple of 360/n for an n-fold rotation centre of G (of 360 if the point is no rotation centre). If G has no rotation
      (p1, pm, pg, cm), alpha is a positive multiple of 360 and the Frank angle 360 - alpha is <= 0. Gauss-Bonnet needs the
      Frank angles to add up to 360 x chi: 720 on the sphere, 360 on the projective plane. So no sphere and no projective
      plane carry the lattice.
  (c) [scope="lattice"] Local cone obstruction: on a closed surface with chi > 0, Gauss-Bonnet requires at least one cone
      (c, alpha > 0) at a rotation centre c of order n >= 2 with alpha = k * 360/n (1 <= k < n). If EVERY rotation centre c
      of the lattice fails local graph validity for every 1 <= k < n (bond midpoint/on a bond folds into a loop; face centre
      leaves < 3 or non-integer sides; site forces degree < 2, non-integer degree, or a loop/double bond in its local 1-ring),
      no sphere or projective plane carries the lattice.
G here is the lattice's full symmetry group, found on the cell by symmetry_facts (every isometry that maps the
lattice's sites onto sites and bonds onto bonds, modulo translations), not only the group's label.
"""
from __future__ import annotations

import itertools

import numpy as np


class SurfaceRefused(ValueError):
    """A surface was not built. reason: plain words; detail: the technical message; tried: what was tried (list);
    hint: what to try instead (or None); retry: False if no other size or centre of the same construction can help."""

    status = "REFUSED"

    def __init__(self, reason, detail="", tried=None, hint=None, retry=True):
        self.reason, self.detail, self.tried, self.hint, self.retry = reason, detail, list(tried or []), hint, retry
        msg = reason
        if detail:
            msg += ": " + detail
        if self.tried:
            msg += " [tried: " + "; ".join(str(t) for t in self.tried) + "]"
        if hint:
            msg += f" ({hint})"
        super().__init__(msg)


class SurfaceImpossible(SurfaceRefused):
    """Proved from the lattice's symmetry group or local cone obstruction: no surface of this kind carries the
    lattice (see the module notes). scope: 'group' (proofs a/b) or 'lattice' (proof c)."""

    status = "IMPOSSIBLE"

    def __init__(self, reason, detail="", tried=None, hint=None, retry=False, scope="group"):
        if scope not in ("group", "lattice"):
            raise ValueError(f"invalid SurfaceImpossible scope {scope!r}: expected 'group' or 'lattice'")
        self.scope = scope
        super().__init__(reason, detail, tried, hint, retry=False)


class SurfaceNotFound(SurfaceRefused):
    """The construction refused at the settings given (a gluing that is not a symmetry at this size, cones that fold a
    bond or a face, no room, ...). Not a proof that the surface does not exist."""

    status = "NOT FOUND"


# ----------------------------------------------------------------------------------------------------------------
# the lattice's full symmetry, on the cell
# ----------------------------------------------------------------------------------------------------------------
def _order(M):
    P, I = np.eye(2, dtype=int), np.eye(2, dtype=int)
    for k in range(1, 7):
        P = P @ M
        if np.array_equal(P, I):
            return k
    return 0


def symmetry_facts(lattice):
    """The lattice's symmetry found on the cell, modulo translations: every linear part M (integer in the cell basis,
    keeping the metric; entries -2..2 plus every linear part listed for the group) and every image j of site 0 give one
    candidate x -> M x + (F_j - M F_0); the candidate is a symmetry if every site lands on a site and every bond on a
    bond (pieces.LatticeData.test). Returns dict(group, rotation_orders (sorted orders > 1 found), reversing (an
    orientation-reversing symmetry found), listed_rotation_orders, listed_reversing (the same from the group's listed
    operations), agrees (found == listed), symmetries [(M, t, listed)])."""
    from .pieces import LatticeData
    lat = lattice if isinstance(lattice, LatticeData) else LatticeData(lattice)
    A, F = lat.A, np.asarray(lat.F, float)
    Ainv = np.linalg.inv(A)
    G = A.T @ A
    tol = 1e-6 * float(np.abs(G).max())
    cands = {tuple(np.rint(np.asarray(M, float)).astype(int).ravel()) for M, _ in lat.ops}
    cands |= set(itertools.product(range(-2, 3), repeat=4))
    found = []
    for e in sorted(cands):
        M = np.array(e, int).reshape(2, 2)
        if abs(round(float(np.linalg.det(M)))) != 1 or np.max(np.abs(M.T @ G @ M - G)) > tol:
            continue
        for j in range(lat.m):
            t = F[j] - M @ F[0]
            ok, _, listed = lat.test((A @ M @ Ainv, A @ t))
            if ok:
                found.append((M, t, listed))
                break                                        # one symmetry per linear part is enough for the facts
    rot = sorted({_order(M) for M, _, _ in found if round(np.linalg.det(M)) == 1 and _order(M) > 1})
    rev = any(round(np.linalg.det(M)) == -1 for M, _, _ in found)
    lops = [np.rint(np.asarray(M, float)).astype(int) for M, _ in lat.ops]
    lrot = sorted({_order(M) for M in lops if round(np.linalg.det(M)) == 1 and _order(M) > 1})
    lrev = any(round(np.linalg.det(M)) == -1 for M in lops)
    return dict(group=lat.L.group, rotation_orders=rot, reversing=rev, listed_rotation_orders=lrot, listed_reversing=lrev,
                agrees=(rot == lrot and rev == lrev), symmetries=found)


def no_reversing_reason(group, what):
    return (f"{group} has no mirror or glide line, and a {what} must be glued with a reflection or a glide: "
            "orientation-reversing gluings are not symmetries of this lattice")


def no_rotation_reason(group, what, chi):
    return (f"{group} has no rotation, so no disclination with a positive Frank angle is allowed, and a {what} needs "
            f"Frank angles adding up to 360 x chi = {360 * chi} deg (Gauss-Bonnet)")
