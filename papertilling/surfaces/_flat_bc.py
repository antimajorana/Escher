"""Boundary conditions of a 2D lattice (C1 WP-B1, plan §2.5): the cylinder on any non-zero circumference (n, m), the torus
drawn on the Wigner-Seitz cell of its superlattice (3 pairs of sides), the simplicity test of a quotient
(NOT FOUND (too small)), the twisted gluings with their proved refusals, and the view flags.

Mirror of `rust/papertilling-core/src/surface/{flat_bc,report_bc}.rs`. The open patch (a disk grown by the touching rule)
is Rust-only: E² inflation has no Python mirror (plan §0), so `boundary_surface(L, "open", ...)` refuses here.

    from papertilling.surfaces._flat_bc import boundary_surface
    S = boundary_surface(pt.named("triangular"), "torus_hex", t1=(3, 0), t2=(0, 3))
    S.picture_note, S.exact_view, S.readout
"""
from __future__ import annotations

import math

import numpy as np

from .refusal import SurfaceImpossible, SurfaceNotFound, no_glide_reason

TOO_SMALL = "NOT FOUND (too small)"
NOT_ISOMETRIC_NOTE = "topological picture, not isometric"


def _in_lattice(d, gens):
    d = [int(d[0]), int(d[1])]
    if len(gens) == 1:
        g = gens[0]
        gg = g[0] * g[0] + g[1] * g[1]
        return d[0] * g[1] - d[1] * g[0] == 0 and gg > 0 and (d[0] * g[0] + d[1] * g[1]) % gg == 0
    g0, g1 = gens
    det = g0[0] * g1[1] - g0[1] * g1[0]
    return det != 0 and (d[0] * g1[1] - d[1] * g1[0]) % det == 0 and (g0[0] * d[1] - g0[1] * d[0]) % det == 0


def _apply(M, t, F, k):
    g = M @ F[k] + t
    for j, fj in enumerate(F):
        d = g - fj
        r = np.round(d)
        if abs(d[0] - r[0]) < 1e-6 and abs(d[1] - r[1]) < 1e-6:
            return j, r.astype(int)
    return None


def simple_quotient(A, basis, bonds, gens, glide=None):
    """Raise SurfaceNotFound (TOO_SMALL) unless the quotient of the decorated plane by the lattice span(gens) (and the
    glide coset g span(gens), glide = (M, t) fractional) is a simple graph: no bond dart from a site ends in its own
    class, no two darts from a site end in the same class (exact integer tests; flat_bc.rs::simple_quotient)."""
    A = np.asarray(A, float)
    F = [np.linalg.solve(A, np.asarray(p, float)) for p in basis]
    gens = [[int(x) for x in g] for g in gens]

    def image(j, d):
        M, t = glide
        r = _apply(np.asarray(M, int), np.asarray(t, float), F, j)
        if r is None:
            raise SurfaceNotFound("the gluing is not a symmetry of the cell")
        md = np.asarray(M, int) @ np.asarray(d, int)
        return r[0], (int(md[0] + r[1][0]), int(md[1] + r[1][1]))

    def same(p, q):
        if p[0] == q[0] and _in_lattice((q[1][0] - p[1][0], q[1][1] - p[1][1]), gens):
            return True
        if glide is not None:
            gp = image(*p)
            return gp[0] == q[0] and _in_lattice((q[1][0] - gp[1][0], q[1][1] - gp[1][1]), gens)
        return False

    for i in range(len(basis)):
        ends = []
        for b in bonds:
            p, q, n1, n2 = (int(x) for x in b[:4])
            for x, y, d in ((p, q, (n1, n2)), (q, p, (-n1, -n2))):
                if x == i and not (x == y and d == (0, 0)):
                    ends.append((y, d))
        for k, e in enumerate(ends):
            if same((i, (0, 0)), e):
                raise SurfaceNotFound(f"{TOO_SMALL}: a bond of site {i} closes into a loop on this quotient; take a larger one",
                                      hint="take a larger one")
            for f in ends[k + 1:]:
                if same(e, f):
                    raise SurfaceNotFound(f"{TOO_SMALL}: two bonds of site {i} join the same pair of sites on this quotient; "
                                          "take a larger one", hint="take a larger one")


def _ext_gcd(a, b):
    if b == 0:
        return (-a, -1, 0) if a < 0 else (a, 1, 0)
    g, x, y = _ext_gcd(b, a % b)
    return g, y, x - (a // b) * y


def wrap_complement(A, c):
    """The lattice vector T with det(c/g, T) = 1 (g = gcd(c)), the most perpendicular to c in its coset T + Z c/g
    (exact; flat_bc.rs::wrap_complement)."""
    c = [int(c[0]), int(c[1])]
    if c == [0, 0]:
        raise ValueError("the circumference vector must not be zero")
    g = math.gcd(*c)
    p = (c[0] // g, c[1] // g)
    _, x, y = _ext_gcd(p[0], -p[1])
    t = np.array([y, x], int)
    A = np.asarray(A, float)
    pc = A @ np.array(p, float)
    k = int(np.round(float((A @ t) @ pc) / float(pc @ pc)))
    return t - k * np.array(p, int)


def _obtuse_reduced(v1, v2):
    """A Gauss-reduced basis w1, w2 of span(v1, v2) with w1 . w2 <= 0 (flat_bc.rs::obtuse_reduced)."""
    from ._glides import gauss_reduce
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    U = gauss_reduce(np.column_stack([v1, v2]))
    w1, w2 = U[0, 0] * v1 + U[1, 0] * v2, U[0, 1] * v1 + U[1, 1] * v2
    return (w1, -w2) if w1 @ w2 > 0 else (w1, w2)


def wigner_seitz(v1, v2):
    """Wigner-Seitz cell of span(v1, v2) (Cartesian): (vertices counter-clockwise, edges [(p, q, pair 1..3)], rectangle)."""
    w1, w2 = _obtuse_reduced(v1, v2)
    w12 = w1 + w2
    rect = abs(w1 @ w2) <= 1e-9 * max(w1 @ w1, w2 @ w2)
    facets = [(w1, 1), (w12, 3), (w2, 2), (-w1, 1), (-w12, 3), (-w2, 2)]
    if rect:
        facets = [f for f in facets if f[1] != 3]
    if w1[0] * w2[1] - w1[1] * w2[0] < 0:
        facets.reverse()
    n = len(facets)

    def cc(a, b):
        return np.linalg.solve(np.array([a, b]), np.array([a @ a / 2, b @ b / 2]))
    verts = [cc(facets[k][0], facets[(k + 1) % n][0]) for k in range(n)]
    edges = [(verts[(k - 1) % n], verts[k], facets[k][1]) for k in range(n)]
    return verts, edges, rect


def torus_hex_domain(lattice, t1, t2):
    """The torus R^2 / span(t1, t2) drawn on the Wigner-Seitz cell of the superlattice, three pairs of sides identified by
    translations: the same quotient as flat.torus (same sites, bonds, faces). Returns (Surface, rectangle)."""
    from .flat import _cell_and_ops, torus
    L, cell, A, F, bonds, ops = _cell_and_ops(lattice)
    simple_quotient(A, cell.basis, cell.bonds, [t1, t2])
    S = torus(L, tuple(t1), tuple(t2))
    v1, v2 = A @ np.array(t1, float), A @ np.array(t2, float)
    verts, edges, rect = wigner_seitz(v1, v2)
    v1, v2 = _obtuse_reduced(v1, v2)                               # the +-2 window is safe only in a reduced basis
    Vinv = np.linalg.inv(np.column_stack([v1, v2]))
    uv = []
    for p in S.uv:
        f = Vinv @ p
        best = None
        for da in range(-2, 3):
            for db in range(-2, 3):
                q = p - (np.round(f[0]) + da) * v1 - (np.round(f[1]) + db) * v2
                r = float(q @ q)
                if best is None or r < best[0] - 1e-9:
                    best = (r, q)
        uv.append(best[1])
    S.uv = np.array(uv)
    colours = {1: "#d9432f", 2: "#2e7d32", 3: "#1f5fbf"}
    S.domain = dict(polygon=verts, edges=[(p, q, str(k), colours[k]) for p, q, k in edges])
    S.kind = "torus (hexagonal domain)"
    if S.xyz_map is not None:
        S.xyz = S.xyz_map(S.uv)
    return S, rect


def picture_note(kind):
    """Caption of the 3D picture (report_bc.rs::picture_note): None for the open patch and the isometric cylinder."""
    return None if kind in ("open", "cylinder") else NOT_ISOMETRIC_NOTE


def boundary_surface(lattice, kind, **kw):
    """Build one boundary condition (report_bc.rs::build_boundary): kind 'cylinder' (wrap, rows), 'torus' (t1, t2),
    'torus_hex' (t1, t2), 'mobius' (glide, steps, width), 'klein' (glide, steps, across). The Surface gets picture_note,
    exact_view and readout. Refusals: SurfaceImpossible (scope 'lattice') for a twisted gluing with no mirror or glide,
    SurfaceNotFound (TOO_SMALL) for a quotient that is not simple, ValueError for bad arguments."""
    from . import flat
    from ._glides import glide_power
    L, cell, A, F, bonds, ops = flat._cell_and_ops(lattice)
    readout = []
    if kind == "open":
        raise ValueError("the open patch is Rust-only (E² inflation has no Python mirror)")
    if kind == "cylinder":
        wrap, rows = tuple(kw["wrap"]), int(kw["rows"])
        if rows < 1:
            raise ValueError(f"rows must be >= 1 (got {rows})")
        wrap_complement(A, wrap)
        simple_quotient(A, cell.basis, cell.bonds, [wrap])
        S = flat.cylinder(L, wrap=wrap, length=rows)
    elif kind == "torus":
        simple_quotient(A, cell.basis, cell.bonds, [kw["t1"], kw["t2"]])
        S = flat.torus(L, tuple(kw["t1"]), tuple(kw["t2"]))
    elif kind == "torus_hex":
        S, rect = torus_hex_domain(L, kw["t1"], kw["t2"])
        if rect:
            readout.append("3 pairs reduce to 2 for this supercell: its Wigner-Seitz cell is a rectangle; choose a "
                           "non-rectangular supercell for a hexagonal domain")
    elif kind in ("mobius", "klein"):
        gl = flat.glides(L)
        if not gl:
            raise SurfaceImpossible(no_glide_reason("Moebius band" if kind == "mobius" else "Klein bottle"), scope="lattice")
        steps = int(kw["steps"])
        g = glide_power(gl[kw.get("glide", 0) % len(gl)], steps)
        gens = [steps * g["tau"]] + ([int(kw["across"]) * g["across"]] if kind == "klein" else [])
        simple_quotient(A, cell.basis, cell.bonds, gens, glide=(g["M"], g["t"]))
        if kind == "mobius":
            S = flat.mobius_band(L, along=steps, width=int(kw["width"]), which=kw.get("glide", 0))
        else:
            S = flat.klein_bottle(L, along=steps, across=int(kw["across"]), which=kw.get("glide", 0))
    else:
        raise ValueError(f"unknown boundary condition {kind!r}")
    ch = S.checks(show=False)
    if not ch.get("ok"):
        raise SurfaceNotFound("the built surface fails its checks", str(ch))
    readout.insert(0, f"{S.kind}: V = {ch['V']}, E = {ch['E']}, F = {ch['F']}, chi = {ch['V'] - ch['E'] + ch['F']}")
    S.picture_note, S.exact_view, S.readout = picture_note(kind), True, readout
    return S
