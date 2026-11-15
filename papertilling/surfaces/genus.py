"""Closed orientable surfaces of genus g >= 2: the lattice on a branched double cover of a flat torus.

WHY THIS CONSTRUCTION. A genus-g surface has chi = 2 - 2g < 0, so (Gauss-Bonnet) the Frank angles of its disclinations add up
to 360 deg x chi < 0: it needs negative-curvature defects. The simplest ones that every lattice allows, whatever its group
(p1 included), are rotations by -360 deg about any point: they are the identity of the group, so the lattice closes up around
them with no misfit. They are the branch points of a double cover.

HOW. Take the lattice on a flat torus T (flat.torus). Choose 2(g - 1) branch points at the centres of distinct faces, well
separated, and join them in pairs by straight slits that pass through no site. Take two copies of T (sheets 0 and 1) and
glue them crosswise along the slits: a bond that crosses a slit goes from sheet s to sheet 1 - s; every other bond stays on
its sheet. The result Sigma is a double cover of T branched at the 2(g - 1) points, so (Riemann-Hurwitz, cited result)
    chi(Sigma) = 2 chi(T) - (number of branch points) = 0 - 2(g - 1) = 2 - 2g,
i.e. Sigma has genus g; it is orientable (a double cover of an orientable surface glued by translations only).
Going once around a branch point on Sigma you stay on the same sheet only after two turns: the angle around it is 720 deg,
a disclination of Frank angle 360 - 720 = -360 deg; 2(g - 1) of them add up to -720 (g - 1) = 360 chi (Gauss-Bonnet).

Faces: the boundary of a face crosses the slits an odd number of times exactly when the face contains one branch point
(proved: the slits start inside the face and leave it). Such a face lifts to ONE face with twice the sides (the corner around
the branch point, of 720 deg); every other face lifts to two copies, one on each sheet. Hence
    V = 2 V0,  E = 2 E0,  F = 2 F0 - 2(g - 1),   V - E + F = 2 (V0 - E0 + F0) - 2(g - 1) = 2 - 2g.
Every site keeps its plane neighbourhood (same bonds, lengths and angles): the branch points sit in faces, not at sites.

CHECKS (GenusSurface.checks, on top of Surface.checks): connected; every bond is a side of exactly two face corners;
V - E + F = chi; the Frank angles 360 (1 - sides / plane sides) of the branch faces add up to 360 chi; each branch face has
twice the sides of its plane face and every other plane face lifts to two faces.

    from papertilling import named
    from papertilling.surfaces.genus import genus_surface
    S = genus_surface(named("honeycomb"), g=2); S.checks(); S.plot_net()
"""
from __future__ import annotations

import numpy as np

from .base import Surface
from .flat import torus
from .refusal import SurfaceNotFound

_SLIT_COLOURS = ["#d9432f", "#2e7d32", "#7b3fb5", "#e08a00", "#00838f", "#ad1457", "#5d4037", "#3949ab"]


# ----------------------------------------------------------------------------------------------------------------
# plane geometry
# ----------------------------------------------------------------------------------------------------------------
def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _point_seg_dist(P, s0, s1):
    """Distance from each point of P (n x 2) to the segment s0 s1."""
    d = s1 - s0
    t = np.clip(((P - s0) @ d) / max(float(d @ d), 1e-300), 0.0, 1.0)
    return np.linalg.norm(P - (s0 + t[:, None] * d), axis=1)


def _seg_seg_dist(a0, a1, b0, b1):
    if _cross(a1 - a0, b0 - a0) * _cross(a1 - a0, b1 - a0) < 0 and _cross(b1 - b0, a0 - b0) * _cross(b1 - b0, a1 - b0) < 0:
        return 0.0
    return float(min(_point_seg_dist(np.array([a0, a1]), b0, b1).min(), _point_seg_dist(np.array([b0, b1]), a0, a1).min()))


def _inside(P, poly):
    """For each point of P (n x 2): inside the polygon? (ray casting, vectorised over the points)."""
    x, y = P[:, 0], P[:, 1]
    c = np.zeros(len(P), bool)
    n = len(poly)
    for k in range(n):
        (x1, y1), (x2, y2) = poly[k], poly[(k + 1) % n]
        if y1 == y2:
            continue
        c ^= ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return c


def _boundary_dist(P, poly):
    n = len(poly)
    return np.min([_point_seg_dist(P, poly[k], poly[(k + 1) % n]) for k in range(n)], axis=0)


def _face_centre(poly):
    """The area centroid if it lies well inside the face (it is the rotation centre of a symmetric face): inside, and at
    least half as far from the boundary as the innermost point of a fine grid; else that innermost point (non-convex or
    thin faces)."""
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    area = cr.sum() / 2
    c = np.array([((x + xn) * cr).sum(), ((y + yn) * cr).sum()]) / (6 * area) if abs(area) > 1e-12 else poly.mean(axis=0)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 61), np.linspace(lo[1], hi[1], 61))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    G = G[_inside(G, poly)]
    d = _boundary_dist(G, poly)
    best, dbest = G[int(np.argmax(d))], float(d.max())
    if _inside(c[None], poly)[0] and _boundary_dist(c[None], poly)[0] >= 0.5 * dbest:
        return c
    return best


# ----------------------------------------------------------------------------------------------------------------
# the surface
# ----------------------------------------------------------------------------------------------------------------
class GenusSurface(Surface):
    """A lattice on a closed orientable surface of genus g >= 2, built as a branched double cover of a flat torus.
    Extra attributes: g; sheet (0 or 1 for every site); plane_uv (the torus position of every site, the same on both
    sheets); torus (the flat torus Surface it covers); slits (list of (p, q) plane points, inside sheet 0's parallelogram);
    periods (V1, V2: the torus vectors, Cartesian); offset (the shift of sheet 1 in plot_net). sites are (i, n1, n2, sheet)."""

    def checks(self, show=True):
        out = super().checks(show=False)
        V = len(self.sites)
        adj = [[] for _ in range(V)]
        for a, b, _ in self.bonds:
            adj[a].append(b)
            adj[b].append(a)
        seen, stack = {0}, [0]
        while stack:
            for b in adj[stack.pop()]:
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        sides = {}
        for f in self.faces:
            for k in range(len(f)):
                e = tuple(sorted((f[k], f[(k + 1) % len(f)])))
                sides[e] = sides.get(e, 0) + 1
        pairs = [tuple(sorted((a, b))) for a, b, _ in self.bonds]
        frank = [360.0 * (1 - c["sides"] / c["plane_sides"]) for c in self.cones]
        out.update(connected=len(seen) == V,
                   two_faces_per_bond=all(sides.get(e, 0) == 2 for e in pairs) and len(sides) == len(pairs),
                   frank_angles=frank, frank_sum=float(sum(frank)),
                   gauss_bonnet=abs(sum(frank) - 360.0 * self.chi) < 1e-9 and all(abs(c["angle"] - fr) < 1e-9 for c, fr in zip(self.cones, frank)),
                   branch_faces_doubled=all(c["sides"] == 2 * c["plane_sides"] for c in self.cones),
                   other_faces_twice=len(self.faces) == 2 * len(self.torus.faces) - len(self.cones),
                   genus=self.g, branch_points=len(self.cones))
        keys = ("no_loops", "no_double_bonds", "plane_coordination", "euler", "faces_at_least_3", "connected",
                "two_faces_per_bond", "gauss_bonnet", "branch_faces_doubled", "other_faces_twice")
        out["ok"] = all(out[k] for k in keys)
        if show:
            print(f"{self.kind}: V {out['V']}, E {out['E']}, F {out['F']}, V - E + F = {out['V_minus_E_plus_F']} (chi {self.chi}); "
                  f"{len(self.cones)} branch points, Frank angles add to {out['frank_sum']:g} = 360 x {self.chi}: {out['gauss_bonnet']}; "
                  + "; ".join(f"{k.replace('_', ' ')} {out[k]}" for k in keys if k not in ("euler", "gauss_bonnet"))
                  + f"  ->  {'OK' if out['ok'] else 'FAILED'}")
        return out

    def plot_net(self, ax=None, title=None, size=6, show_faces=True, **kw):
        """The two sheets side by side (sheet 1 to the right). Each sheet is the torus's parallelogram, whose opposite edges
        (arrows 1, 2) are glued to each other on the SAME sheet. Each slit (thick line, letter) is drawn on both sheets: a
        bond crossing it (red) lands on the other sheet. Stars: branch points (Frank angle -360 deg); shaded: the faces that
        hold them (each becomes one face with twice the sides). Other keywords of Surface.plot_net (legend, rotate) are
        accepted and ignored."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.subplots(figsize=(2.2 * size, size))[1]
        V1, V2 = self.periods
        P = self.uv
        cross = {(a, b) for a, b, _ in self.bonds if self.sheet[a] != self.sheet[b]}
        for a, b, v in self.bonds:
            red = (a, b) in cross
            ax.plot([P[a, 0], P[a, 0] + v[0]], [P[a, 1], P[a, 1] + v[1]], "-", color="#d62728" if red else "#1f6fc5",
                    lw=1.6 if red else 0.9, zorder=2 if red else 1)
        ax.scatter(P[:, 0], P[:, 1], s=9, color="k", zorder=3)
        o = np.zeros(2)
        for s, D in enumerate((o, self.offset)):
            poly = np.array([D, D + V1, D + V1 + V2, D + V2])
            ax.fill(poly[:, 0], poly[:, 1], color="#f3e7b1", alpha=0.45, zorder=0)
            pr = "" if s == 0 else "'"
            for (p, q, lab, col) in ((D, D + V1, "1", "#555555"), (D + V2, D + V1 + V2, "1", "#555555"),
                                     (D, D + V2, "2", "#999999"), (D + V1, D + V1 + V2, "2", "#999999")):
                ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6), zorder=4)
                ax.text(*((p + q) / 2), lab + pr, color=col, fontsize=9, ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
            ax.text(*(D + 0.5 * V1 - 0.07 * V2), f"sheet {s}", ha="center", va="top", fontsize=11)
            if show_faces:
                for c in self.cones:
                    pp = c["polygon"] + D
                    ax.fill(pp[:, 0], pp[:, 1], color="#ffb36b", alpha=0.6, zorder=0.5)
            for k, (p, q) in enumerate(self.slits):
                col = _SLIT_COLOURS[k % len(_SLIT_COLOURS)]
                ax.plot([p[0] + D[0], q[0] + D[0]], [p[1] + D[1], q[1] + D[1]], "-", color=col, lw=3.2, zorder=4, alpha=0.85)
                m = (p + q) / 2 + D
                ax.text(*m, chr(65 + k), color=col, fontsize=11, ha="center", va="center", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, alpha=0.9))
                for x in (p, q):
                    ax.plot(*(x + D), marker="*", ms=13, color="k", zorder=6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title or (f"genus {self.g} surface of {self.lattice.group} ({len(self.cones)} branch points, -360 deg each): "
                               "two torus sheets.\nArrows glue edges of the same sheet; a bond crossing slit A, B, ... "
                               "(red) lands on the other sheet. Shaded faces become one face with twice the sides."), fontsize=9)
        return ax


def _torus_vectors(torus_size):
    t = np.asarray(torus_size, int)
    if t.shape == (2,):
        return (int(t[0]), 0), (0, int(t[1]))
    if t.shape == (2, 2):
        return tuple(int(x) for x in t[0]), tuple(int(x) for x in t[1])
    raise ValueError("torus_size: (m, n) for T1 = (m, 0), T2 = (0, n), or ((a, b), (c, d))")


def genus_surface(lattice, g=2, torus_size=(6, 6), branch=None):
    """The lattice on a closed orientable surface of genus g: a double cover of the flat torus `torus_size` (cells of the
    lattice: (m, n) or two integer vectors), branched at 2(g - 1) face centres joined in pairs by straight slits.
    branch: None (chosen automatically: g - 1 slits about a third of the torus long, in rows, through no site and at
    least one bond length apart), or a list of g - 1 pairs ((x1, y1), (x2, y2)) of plane points (Cartesian): each point
    moves to the nearest face centre and the slit is the shortest segment between the two (refused with SurfaceNotFound,
    a ValueError, if it passes through a site, or if two branch points fall in one face; also when the torus is too small
    or has no room for the slits: then e.hint names a larger torus_size). g = 1 returns the flat torus itself (no branch
    points); g < 1 is refused (use the sphere or projective plane constructions)."""
    if int(g) != g or g < 1:
        raise ValueError("g must be an integer >= 1 (genus 0 is the sphere: see polyhedra.py)")
    g = int(g)
    T1, T2 = _torus_vectors(torus_size)
    t = np.asarray(torus_size, int)
    bigger = f"try torus_size={(int(t[0]) + 2, int(t[1]) + 2)}" if t.shape == (2,) else "try a larger torus_size"
    T = torus(lattice, T1, T2)
    if g == 1:
        return T
    tc = T.checks(show=False)
    if not tc["ok"]:
        raise SurfaceNotFound(f"the torus {T1}, {T2} is too small for this lattice (loops or double bonds)", hint=bigger)
    L = T.lattice
    cell = L.cell()
    A = cell.A
    V1, V2 = A @ np.array(T1, float), A @ np.array(T2, float)
    U = np.column_stack([V1, V2])
    Uinv = np.linalg.inv(U)
    V0 = len(T.sites)
    f = T.uv @ Uinv.T
    uv = (f - np.floor(f + 1e-9)) @ U.T                            # wrapped into the parallelogram; bond vectors unchanged
    vec = {}
    for a, b, v in T.bonds:
        vec[(a, b)], vec[(b, a)] = np.asarray(v, float), -np.asarray(v, float)
    blen = float(np.median([np.linalg.norm(v) for _, _, v in T.bonds]))
    tol = 0.05 * blen
    # faces as plane polygons and their centres
    polys, centres, shape_centre = [], [], {}
    for fc in T.faces:
        X = [uv[fc[0]]]
        for k in range(len(fc)):
            X.append(X[-1] + vec[(fc[k], fc[(k + 1) % len(fc)])])
        if np.linalg.norm(X[-1] - X[0]) > 1e-6:
            raise RuntimeError("a face of the torus does not close in the plane")
        poly = np.array(X[:-1])
        key = tuple(np.round(poly - poly[0], 6).ravel())             # faces of one shape share their centre
        if key not in shape_centre:
            shape_centre[key] = _face_centre(poly) - poly[0]
        c = poly[0] + shape_centre[key]
        fr = Uinv @ c
        shift = U @ np.floor(fr + 1e-9)
        polys.append(poly - shift)
        centres.append(c - shift)
    centres = np.array(centres)
    shifts = [U @ np.array([i, j]) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)]
    sites_all = np.vstack([uv + s for s in shifts])

    def min_image(d):
        """The shortest plane vector equal to d modulo the torus periods (ties: the one closest to d itself)."""
        fr = Uinv @ np.asarray(d, float)
        cands = [(round(float(np.linalg.norm(U @ (fr + np.array([i, j])))), 9), abs(i) + abs(j), U @ (fr + np.array([i, j])))
                 for i in range(-3, 4) for j in range(-3, 4)]
        return min(cands, key=lambda x: x[:2])[2]

    def nearest_face(p):
        d = [np.linalg.norm(min_image(c - np.asarray(p, float))) for c in centres]
        return int(np.argmin(d))

    def problem(p, q, fp, fq, chosen):
        """Why the slit p -> q cannot be used (None if it can)."""
        if fp == fq:
            return "both branch points are in the same face"
        if any(fp in (x[2], x[3]) or fq in (x[2], x[3]) for x in chosen):
            return "a face already holds a branch point"
        dmin = float(_point_seg_dist(sites_all, p, q).min())
        if dmin < tol:
            return f"the slit passes through a site (distance {dmin:.3g})"
        for (p2, q2, _, _) in chosen:
            for s in shifts:
                if _seg_seg_dist(p, q, p2 + s, q2 + s) < blen:
                    return "the slit comes closer than one bond length to another slit"
        return None

    chosen = []
    if branch is not None:
        if len(branch) != g - 1:
            raise ValueError(f"branch: give g - 1 = {g - 1} pairs of points")
        for (x, y) in branch:
            fp, fq = nearest_face(x), nearest_face(y)
            p = centres[fp]
            q = p + min_image(centres[fq] - p)
            why = problem(p, q, fp, fq, chosen)
            if why:
                raise SurfaceNotFound(f"slit {x} -> {y} (moved to face centres {p.round(3).tolist()} -> {q.round(3).tolist()}): "
                                      f"{why}", hint="choose other branch points, or branch=None")
            chosen.append((p, q, fp, fq))
    else:
        n = g - 1
        fcen = centres @ Uinv.T
        for k in range(n):
            yk = (k + 0.5) / n
            tp, tq = U @ np.array([1 / 3, yk]), U @ np.array([2 / 3, yk])
            order_p = np.argsort(np.linalg.norm(centres - tp, axis=1))[:12]
            order_q = np.argsort(np.linalg.norm(centres - tq, axis=1))[:12]
            combos = sorted(((float(np.linalg.norm(centres[a] - tp) + np.linalg.norm(centres[b] - tq)), int(a), int(b))
                             for a in order_p for b in order_q), key=lambda x: x[0])
            for _, fp, fq in combos:
                if not (np.all(fcen[fp] > 0.02) and np.all(fcen[fp] < 0.98) and np.all(fcen[fq] > 0.02) and np.all(fcen[fq] < 0.98)):
                    continue
                p, q = centres[fp], centres[fq]
                used = [fp, fq] + [x[m] for x in chosen for m in (2, 3)]
                if any(set(T.faces[a]) & set(T.faces[b]) for a in used for b in used if a < b):
                    continue                                       # well separated: branch faces share no site
                if problem(p, q, fp, fq, chosen) is None:
                    chosen.append((p, q, fp, fq))
                    break
            else:
                raise SurfaceNotFound(f"no room for {n} slits on the torus {T1}, {T2} of {L.group} (genus {g})",
                                      "no slit pair through no site and a bond length from the others was found near its row",
                                      tried=[f"slit {k + 1} of {n}, 12 x 12 face pairs near 1/3 and 2/3 of the torus"],
                                      hint=bigger)
    # which bonds cross a slit (parity)
    P0 = np.array([uv[a] for a, b, v in T.bonds])
    P1 = P0 + np.array([v for a, b, v in T.bonds], float)
    par = np.zeros(len(T.bonds), int)
    for (p, q, _, _) in chosen:
        for s in shifts:
            s0, s1 = p + s, q + s
            o1, o2 = _cross(s1 - s0, P0 - s0), _cross(s1 - s0, P1 - s0)
            o3, o4 = _cross(P1 - P0, s0 - P0), _cross(P1 - P0, s1 - P0)
            par += ((o1 * o2 < 0) & (o3 * o4 < 0)).astype(int)
    par %= 2
    pmap = {}
    for k, (a, b, v) in enumerate(T.bonds):
        pmap[(a, b)] = pmap[(b, a)] = int(par[k])
    # the face parity must be odd exactly at the branch faces
    branch_faces = sorted({x[2] for x in chosen} | {x[3] for x in chosen})
    odd = [k for k, fc in enumerate(T.faces) if sum(pmap[(fc[j], fc[(j + 1) % len(fc)])] for j in range(len(fc))) % 2]
    if odd != branch_faces:
        raise RuntimeError(f"slit parity is odd at faces {odd}, expected the branch faces {branch_faces}")
    # the double cover
    sites = [tuple(T.sites[k]) + (s,) for s in (0, 1) for k in range(V0)]
    sheet = np.repeat([0, 1], V0)
    offset = 1.15 * V1
    uv2 = np.vstack([uv, uv + offset])
    bonds = [(a + s * V0, b + (s ^ pmap[(a, b)]) * V0, np.asarray(v, float)) for s in (0, 1) for (a, b, v) in T.bonds]
    faces, fseen, cones = [], set(), []
    for k, fc in enumerate(T.faces):
        for s0 in (0, 1):
            ids, s = [], s0
            for turn in (0, 1):
                for j in range(len(fc)):
                    ids.append(fc[j] + s * V0)
                    s ^= pmap[(fc[j], fc[(j + 1) % len(fc)])]
                if s == s0:
                    break
            key = frozenset(ids)
            if key in fseen:
                continue
            fseen.add(key)
            faces.append(ids)
            if k in branch_faces:
                cones.append(dict(kind="branch point", point=centres[k], face=len(faces) - 1, angle=360.0 * (1 - len(ids) / len(fc)),
                                  location="face centre", sides=len(ids), plane_sides=len(fc), polygon=polys[k]))
    slits = [(p, q) for (p, q, _, _) in chosen]
    dom = dict(polygon=[np.zeros(2), V1, V1 + V2, V2], edges=[], sheets=[np.zeros(2), offset], slits=slits)
    S = GenusSurface(f"genus {g} surface", L, sites, uv2, bonds, faces, 2 - 2 * g, True, True, dom,
                     dict(g=g, T1=T1, T2=T2, branch=[(p.tolist(), q.tolist()) for p, q in slits]), T.plane_degree)
    S.g, S.sheet, S.plane_uv, S.torus, S.slits, S.periods, S.offset = g, sheet, np.vstack([uv, uv]), T, slits, (V1, V2), offset
    S.cones = cones
    S.xyz = None
    S.xyz_note = ("no 3D shape: this genus-g surface is flat except at its branch points (each with 720 deg around it); "
                  "no simple 3D picture keeps that metric, so use plot_net() (the two torus sheets and the slits)")
    return S
