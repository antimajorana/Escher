"""Delaney-Dress symbol of a planar periodic lattice with a given symmetry group.

WHY. Two lattices are "the same" (rule B, the author's choice of 2026-11-13) when they are the same tiling
(same network, same faces, meeting the same way) AND the group G acts on it the same way. That is
exactly the equivariant equivalence of tilings, and the Delaney-Dress symbol decides it exactly
(Dress 1984; Delgado-Friedrichs 2003): two (tiling, G) pairs are equivalent iff their D-symbols are
isomorphic. cited result -- check the references before relying on it.

CONSTRUCTION (all from the lattice itself, no hashing).
  dart     (i, j, n): bond from site i of cell 0 to site j of cell n, in the cell's fractional basis.
           Every bond gives two darts, d and rev(d).
  rotation the darts leaving a site, sorted counter-clockwise by the angle of their drawn bond:
           next_ccw(d). Taken from the geometry, so it is the planar embedding when no bonds cross.
  chamber  (d, s), s = 0: the face on the LEFT of d (between d and next_ccw(d) at the tail of d),
           s = 1: the face on the right. A chamber is a flag (vertex, edge, face) of the tiling:
           vertex = tail of d, edge = bond of d, face = side s. 4 E chambers per cell (E bonds per cell).
  sigma0   other vertex, same edge and face:  (d, s) -> (rev d, 1 - s)
  sigma1   other edge, same vertex and face:  (d, 0) -> (next_ccw d, 1),  (d, 1) -> (prev_ccw d, 0)
  sigma2   other face, same vertex and edge:  (d, s) -> (d, 1 - s)
  group    an operation g = (Mf, t) sends dart (i, j, n) to (pi, pj, Mf n + c_j - c_i) and keeps s if
           det Mf = +1, flips it if det Mf = -1 (a reflection swaps left and right).
  D-set    chambers modulo G. G acts freely on chambers, so |D| = 4 E / |ops| (checked).
  labels   m01(D) = number of bonds of the face of D, m12(D) = number of bonds of the vertex of D.
CANONICAL FORM. For every start element, number the D-set breadth first (sigma0, sigma1, sigma2 in
order); write down the numbered sigma images and the labels; keep the smallest code. Two D-symbols are
isomorphic iff their canonical codes are equal (proved result for connected D-sets: an isomorphism is
fixed by the image of one element; formal verification not performed).
CHECKS returned with the symbol (exact, from the definitions):
  free action, every face closes with zero total cell shift (a disk), V - E + F = 0 per cell (torus),
  curvature  sum_D (1/m01 + 1/m12 - 1/2) = 0  (flat plane).
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from . import _rust
from .legacy.wallpaper_lattices import _symmetry_ops, _apply_to_site


class DSymbolError(ValueError):
    """The lattice is not a planar tiling on which the D-symbol is defined (crossing bonds, a face that
    is not a disk, or a group that does not act freely on flags)."""


def _darts(cell):
    A = cell.A
    F = np.asarray(cell.basis, float) @ np.linalg.inv(A).T
    darts = []
    for (i, j, n1, n2, _t) in cell.bonds:
        darts.append((i, j, n1, n2))
        darts.append((j, i, -n1, -n2))
    if len(set(darts)) != len(darts):
        raise DSymbolError("a bond is listed twice, or a bond joins a site to its own translate twice")
    index = {d: k for k, d in enumerate(darts)}
    rev = [index[(j, i, -n1, -n2)] for (i, j, n1, n2) in darts]
    return F, darts, index, rev


def _rotation(cell, F, darts):
    A = cell.A
    out = {}
    for k, (i, j, n1, n2) in enumerate(darts):
        v = A @ (F[j] + np.array([n1, n2]) - F[i])
        out.setdefault(i, []).append((math.atan2(v[1], v[0]), k))
    nxt, prv = {}, {}
    for i, lst in out.items():
        lst.sort()
        ks = [k for _, k in lst]
        for a in range(len(ks)):
            nxt[ks[a]] = ks[(a + 1) % len(ks)]
            prv[ks[a]] = ks[a - 1]
    return nxt, prv, {i: len(v) for i, v in out.items()}


def _faces(darts, rev, prv):
    """Faces as cycles of darts with the face on the left: after d = (u -> v), the next dart leaves v
    and is the one just clockwise of rev(d) at v (keeps the face on the left)."""
    face_of, faces = {}, []
    for d0 in range(len(darts)):
        if d0 in face_of:
            continue
        cyc, d = [], d0
        while d not in face_of:
            face_of[d] = len(faces); cyc.append(d)
            d = prv[rev[d]]
        if d != d0:
            raise DSymbolError("face walk did not close on its start (inconsistent rotation system)")
        faces.append(cyc)
    return faces, face_of


def dsymbol(cell, ops=None):
    """D-symbol of the lattice `cell` (a UnitCell) under the group `ops` (list of (Mf, t) in the cell's
    fractional basis; default: the lattice's own symmetry operations, bond colours ignored).
    Returns a dict: size, s0, s1, s2 (lists), m01, m12, key (canonical code, a tuple), checks."""
    F, darts, index, rev = _darts(cell)
    if ops is None:
        ops = _symmetry_ops(cell, use_types=False)
    nxt, prv, deg = _rotation(cell, F, darts)
    if len(deg) != len(F):
        raise DSymbolError("a site has no bonds")
    faces, face_of = _faces(darts, rev, prv)
    for cyc in faces:
        tot = np.sum([darts[d][2:4] for d in cyc], axis=0)
        if np.any(tot != 0):
            raise DSymbolError("a face winds around the torus (bonds cross, or the tiling is not planar)")
    V, E, Fn = len(F), len(darts) // 2, len(faces)
    if V - E + Fn != 0:
        raise DSymbolError(f"V - E + F = {V - E + Fn} per cell, not 0: not a planar tiling")
    nch = 2 * len(darts)                                  # chamber c = 2 d + s
    # group action on darts
    gact = []
    for Mf, t in ops:
        Mf = np.asarray(Mf); t = np.asarray(t, float)
        img = [_apply_to_site(Mf, t, F, k) for k in range(len(F))]
        perm = []
        for (i, j, n1, n2) in darts:
            (pi, ci), (pj, cj) = img[i], img[j]
            dn = Mf @ np.array([n1, n2]) + cj - ci
            key = (pi, pj, int(round(float(dn[0]))), int(round(float(dn[1]))))      # round: int() would truncate 0.9999999
            if key not in index:
                raise DSymbolError("an operation does not map the bonds onto bonds")
            perm.append(index[key])
        flip = int(round(np.linalg.det(Mf))) < 0
        gact.append((perm, flip))
    orbit = [-1] * nch
    reps = []
    for c in range(nch):
        if orbit[c] >= 0:
            continue
        o = len(reps); reps.append(c)
        d, s = divmod(c, 2)
        imgs = {2 * perm[d] + (1 - s if flip else s) for perm, flip in gact}
        if len(imgs) != len(gact):
            raise DSymbolError("the group does not act freely on flags")
        for x in imgs:
            orbit[x] = o
    n = len(reps)

    def s0(c):
        d, s = divmod(c, 2); return 2 * rev[d] + (1 - s)

    def s1(c):
        d, s = divmod(c, 2); return 2 * nxt[d] + 1 if s == 0 else 2 * prv[d]

    def s2(c):
        return c ^ 1

    def face_size(c):
        d, s = divmod(c, 2)
        return len(faces[face_of[d]]) if s == 0 else len(faces[face_of[rev[d]]])

    sig = [[orbit[f(c)] for c in reps] for f in (s0, s1, s2)]
    m01 = [face_size(c) for c in reps]
    m12 = [deg[darts[c // 2][0]] for c in reps]
    curv = sum(Fraction(1, a) + Fraction(1, b) - Fraction(1, 2) for a, b in zip(m01, m12))
    key = canonical_key(sig, m01, m12)
    return dict(size=n, s0=sig[0], s1=sig[1], s2=sig[2], m01=m01, m12=m12, key=key,
                checks=dict(V=V, E=E, F=Fn, euler=V - E + Fn, curvature=curv, free=True,
                            size_formula=4 * E / len(ops), order=len(ops)))


def canonical_key(sig, m01, m12):
    """Smallest breadth-first code over all start elements (see the module docstring)."""
    n = len(m01)
    if n and (rs := _rust.get()) is not None:
        try:
            return rs.canonical_key([[int(y) for y in s] for s in sig], [int(x) for x in m01], [int(x) for x in m12])
        except (ValueError, OverflowError) as err:
            raise DSymbolError(str(err)) from None
    best = None
    for start in range(n):
        num = {start: 0}; order = [start]; k = 0
        code = []
        while k < len(order):
            x = order[k]; k += 1
            row = [m01[x], m12[x]]
            for i in range(3):
                y = sig[i][x]
                if y not in num:
                    num[y] = len(order); order.append(y)
                row.append(num[y])
            code.append(tuple(row))
            if best is not None and tuple(code) > best[:len(code)]:
                break
        else:
            if len(order) != n:
                raise DSymbolError("D-set not connected")
            t = tuple(code)
            if best is None or t < best:
                best = t
    return best


def same_tiling(cell_a, cell_b, ops_a=None, ops_b=None):
    """Rule B: same tiling with the same symmetry action (equal canonical D-symbol keys)."""
    return dsymbol(cell_a, ops_a)["key"] == dsymbol(cell_b, ops_b)["key"]


def network_key(key):
    """The NETWORK of a lattice, with its symmetry forgotten: the canonical key of the smallest D-symbol
    that describes the same tiling (its "minimal image").

    WHY. The D-symbol of (tiling, G) changes with G: the square lattice seen through p4m has 1 element,
    seen through p2 with one site per cell 4, seen through p2 with two sites per cell 8. All of them
    unfold to the same infinite tiling. Merge elements that cannot be told apart by any walk: start with
    the classes (m01, m12) and split a class whenever two of its elements have sigma_0, sigma_1 or sigma_2
    images in different classes, until nothing splits (partition refinement). The classes are the elements
    of the D-symbol of the tiling under its FULL combinatorial symmetry group (cited result: Dress;
    Delgado-Friedrichs 2003, every D-symbol has a unique minimal image, and two tilings are
    combinatorially the same iff their minimal images are isomorphic -- check the reference).
    So: equal network_key <=> same network drawn in the plane (same sites, bonds and faces meeting the same
    way), whatever the group and the cell. A mirror image has the same key.
    Input: a canonical key (rows m01, m12, s0, s1, s2). Output: a canonical key of the same form."""
    if key and (rs := _rust.get()) is not None:
        try:
            return rs.network_key([[int(x) for x in r] for r in key])
        except (ValueError, OverflowError) as err:
            raise DSymbolError(str(err)) from None
    n = len(key)
    cls = {}
    cid = [cls.setdefault((r[0], r[1]), len(cls)) for r in key]
    while True:
        new = {}
        nid = [new.setdefault((cid[x], cid[key[x][2]], cid[key[x][3]], cid[key[x][4]]), len(new)) for x in range(n)]
        if len(new) == len(cls):
            break
        cls, cid = new, nid
    k = len(cls)
    rep = {}
    for x in range(n):
        rep.setdefault(cid[x], x)
    sig = [[cid[key[rep[c]][2 + i]] for c in range(k)] for i in range(3)]
    return canonical_key(sig, [key[rep[c]][0] for c in range(k)], [key[rep[c]][1] for c in range(k)])


def symmetry_index(key):
    """How many times larger the network's own symmetry is than the group the key was computed with:
    (elements of the key) / (elements of its minimal image). 1 = the group is the network's full symmetry."""
    if key and (rs := _rust.get()) is not None:
        try:
            return int(rs.symmetry_index([[int(x) for x in r] for r in key]))
        except (ValueError, OverflowError) as err:
            raise DSymbolError(str(err)) from None
    return len(key) // len(network_key(key))


def orbifold_curvature(key) -> Fraction:
    """Exact rational orbifold curvature sum_C (1/m01 + 1/m12 - 1/2)."""
    return sum((Fraction(1, int(r[0])) + Fraction(1, int(r[1])) - Fraction(1, 2) for r in key), Fraction(0))


def geometry_kind(key) -> str:
    """Return 'spherical' (kappa > 0), 'euclidean' (kappa == 0), or 'hyperbolic' (kappa < 0)."""
    k = orbifold_curvature(key)
    if k > 0:
        return "spherical"
    if k < 0:
        return "hyperbolic"
    return "euclidean"
