"""Pieces + pairings: a lattice on a surface glued from flat polygons (the engine behind polyhedra.py).

WHAT. A closed surface with cone points is glued from flat convex polygons ("pieces", e.g. the faces of an
icosahedron unfolded into the plane). Every side of a piece is glued to one other side (a "pairing"); the gluing is
the plane isometry g that carries the first side onto the second and the outside of the first piece onto the inside
of the second. The lattice is drawn in the plane under every piece; the surface carries the sites, bonds and faces
that lie in the pieces, identified across the glued sides.

THE GROUP TEST (proved: if every pairing g is a symmetry of the lattice, then g maps sites to sites, bonds to bonds
and faces to faces, so the copies match across every glued side and every point away from the corners keeps its
plane neighbourhood). Each pairing is tested on the cell itself: its linear part must be an integer matrix in the
cell's fractional basis, every site must land on a site and every bond on a bond. A construction with a pairing that
fails is refused (SurfaceNotFound), with the pairing named and described (rotation angle and centre, glide axis, ...).
Consequences: the icosahedron needs rotations by 60 deg, so it is refused for p3 (only 120 deg rotations); a
non-orientable gluing needs a reflection or glide in the group, so it is refused for chiral groups.

HOW (exact labels, floats only for point-in-piece tests). A site copy is (piece, (i, n1, n2)); a bond copy is
(piece, {two site labels}), kept in the piece that holds its midpoint; a face copy is (piece, {corner labels}), kept in
the piece that holds its centroid. Copies on a glued side are carried by g (its fractional form acts on labels) and
united with the copy on the partner side (union-find). A bond whose midpoint is a corner of the net would be folded
into a loop by the cone there: refused. The two ends of a bond, and the two faces beside it, are found by walking
straight from the bond's midpoint, crossing glued sides through their pairings. Corners of the pieces glued together
form the cone points; Frank angle = 360 deg - (sum of the corner angles at it).
"""
from __future__ import annotations

import math

import numpy as np

from ..legacy.wallpaper_lattices import _apply_to_site
from .base import Surface
from .flat import _cell_and_ops, _plane_faces
from .refusal import SurfaceNotFound, SurfaceRefused

TOL = 1e-7


# ConeRefused: the base class of both refusals (old name); every refusal below is a SurfaceNotFound (this net, at this
# size and centre, does not carry the lattice; another net may). The proved impossibilities are raised before any net is
# built (polyhedra._build, flat._glide_setup; see refusal.py).
ConeRefused = SurfaceRefused


# ----------------------------------------------------------------------------------------------------------------
# the lattice and its group
# ----------------------------------------------------------------------------------------------------------------
class LatticeData:
    """The lattice in the plane: cell, sites, darts, face templates; group test of plane isometries."""

    def __init__(self, lattice):
        self.L, self.cell, self.A, self.F, self.bonds, self.ops = _cell_and_ops(lattice)
        self.Ainv = np.linalg.inv(self.A)
        self.basis = np.asarray(self.cell.basis, float)
        self.m = len(self.F)
        self.darts = set()
        for (i, j, n1, n2) in self.bonds:
            self.darts.add((i, j, n1, n2))
            self.darts.add((j, i, -n1, -n2))
        cycles, deg = _plane_faces(self.cell, self.F)
        self.plane_degree = {int(i): int(d) for i, d in deg.items()}
        self.faces = []                     # templates: (corner labels relative to cell 0, centroid)
        self.dart_face = {}                 # dart -> (template, cell offset of its tail inside the template)
        for f, cyc in enumerate(cycles):
            n = np.zeros(2, int)
            labs = []
            for (i, j, n1, n2) in cyc:
                labs.append((i, int(n[0]), int(n[1])))
                self.dart_face[(i, j, n1, n2)] = (f, (int(n[0]), int(n[1])))
                n = n + np.array([n1, n2])
            P = np.array([self.pos(l) for l in labs])
            self.faces.append((labs, _centroid(P)))
        self.lmax = max(np.linalg.norm(self.pos((j, n1, n2)) - self.pos((i, 0, 0))) for (i, j, n1, n2) in self.bonds)
        self.fmax = max(max(np.linalg.norm(self.pos(l) - c) for l in labs) for labs, c in self.faces)

    def pos(self, lab):
        i, n1, n2 = lab
        return self.basis[i] + self.A @ np.array([n1, n2], float)

    # --- the group test --------------------------------------------------------------------------------------
    def frac(self, g):
        R, u = g
        Mf = self.Ainv @ R @ self.A
        return Mf, self.Ainv @ u

    def test(self, g):
        """(True, '') if the plane isometry g = (R, u) maps the lattice's sites onto sites and bonds onto bonds;
        else (False, reason). Also returns whether g is one of the group's listed operations (ops_in_cell)."""
        Mf, tf = self.frac(g)
        Mi = np.rint(Mf)
        if np.max(np.abs(Mf - Mi)) > 1e-6:
            return False, "its linear part is not a symmetry of the translation lattice", False
        Mi = Mi.astype(int)
        img = {}
        for i in range(self.m):
            try:
                img[i] = _apply_to_site(Mi, tf, self.F, i)
            except RuntimeError:
                return False, f"site {i} of the cell lands on no site", False
        for (i, j, n1, n2) in self.bonds:
            (pi, ci), (pj, cj) = img[i], img[j]
            d = Mi @ np.array([n1, n2]) + cj - ci
            if (pi, pj, int(d[0]), int(d[1])) not in self.darts:
                return False, f"bond ({i}, {j}, {n1}, {n2}) lands on no bond", False
        listed = any(np.allclose(np.asarray(M, float), Mi) and np.allclose(tf - t, np.rint(tf - t), atol=1e-6)
                     for M, t in self.ops)
        return True, "", listed

    def apply(self, gf, lab):
        """The label of the image of site `lab` under the fractional operation gf = (Mi, tf)."""
        Mi, tf = gf
        j, c = _apply_to_site(Mi, tf, self.F, lab[0])
        n = Mi @ np.array(lab[1:]) + c
        return (int(j), int(n[0]), int(n[1]))


def describe(g):
    """Words for a plane isometry g = (R, u)."""
    R, u = g
    det = np.linalg.det(R)
    if det > 0:
        ang = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        if abs(ang) < 1e-6:
            return "the identity" if np.linalg.norm(u) < 1e-9 else f"a translation by {np.round(u, 4).tolist()}"
        c = np.linalg.solve(np.eye(2) - R, u)
        return f"a rotation by {ang:.1f} deg about {np.round(c, 4).tolist()}"
    w, V = np.linalg.eigh((R + R.T) / 2)
    ax = V[:, np.argmax(w)]
    s = float(u @ ax)
    p = np.linalg.lstsq(np.eye(2) - R, u - s * ax, rcond=None)[0]
    kind = "a reflection" if abs(s) < 1e-9 else f"a glide by {s:.4f}"
    return f"{kind} in the line through {np.round(p, 4).tolist()} along {np.round(ax, 4).tolist()}"


def _centroid(P):
    x, y = P[:, 0], P[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cr = x * ys - xs * y
    a = cr.sum() / 2
    if abs(a) < 1e-12:
        return P.mean(axis=0)
    return np.array([((x + xs) * cr).sum(), ((y + ys) * cr).sum()]) / (6 * a)


def _rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


class _UF:
    """Union-find; the root of a class is its first-added member (keys may hold frozensets, which do not sort)."""

    def __init__(self):
        self.p, self.rank = {}, {}

    def add(self, x):
        if x not in self.p:
            self.p[x], self.rank[x] = x, len(self.rank)

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.rank[a] > self.rank[b]:
                a, b = b, a
            self.p[b] = a

    def roots(self):
        return sorted({self.find(x) for x in self.p}, key=self.rank.get)


# ----------------------------------------------------------------------------------------------------------------
# pieces and pairings
# ----------------------------------------------------------------------------------------------------------------
class Net:
    """Convex pieces (corner lists, counter-clockwise) and pairs ((k1, s1), (k2, s2), flip): side s1 of piece k1 (from
    corner s1 to s1 + 1) is glued to side s2 of piece k2, its start going to the start of s2 (flip False) or to its
    end (flip True). Computes every gluing isometry."""

    def __init__(self, pieces, pairs):
        self.P = [np.asarray(p, float) for p in pieces]
        self.N, self.c = [], []
        for k, P in enumerate(self.P):
            E = np.roll(P, -1, axis=0) - P
            if np.any(E[:, 0] * np.roll(E, -1, axis=0)[:, 1] - E[:, 1] * np.roll(E, -1, axis=0)[:, 0] <= 1e-12):
                raise ValueError(f"piece {k} is not convex and counter-clockwise")
            nrm = np.column_stack([E[:, 1], -E[:, 0]]) / np.linalg.norm(E, axis=1)[:, None]
            self.N.append(nrm)
            self.c.append(np.einsum("ij,ij->i", nrm, P))
        self.pairs = list(pairs)
        self.glue = {}                                   # (k, s) -> (l, s2, g=(R, u))
        for (k1, s1), (k2, s2), flip in self.pairs:
            for a, b, f in (((k1, s1), (k2, s2), flip), ((k2, s2), (k1, s1), flip)):
                if a in self.glue:
                    raise ValueError(f"side {a} is glued twice")
                self.glue[a] = (b[0], b[1], self._iso(a, b, f))
        for k, P in enumerate(self.P):
            for s in range(len(P)):
                if (k, s) not in self.glue:
                    raise ValueError(f"side {s} of piece {k} is not glued: the surface would have an open edge")

    def side(self, k, s):
        P = self.P[k]
        return P[s], P[(s + 1) % len(P)]

    def _iso(self, a, b, flip):
        p0, p1 = self.side(*a)
        q0, q1 = self.side(*b)
        if flip:
            q0, q1 = q1, q0
        if abs(np.linalg.norm(p1 - p0) - np.linalg.norm(q1 - q0)) > 1e-6:
            raise ValueError(f"sides {a} and {b} have different lengths")
        psi = math.atan2(*(p1 - p0)[::-1])
        phi = math.atan2(*(q1 - q0)[::-1])
        inner_a = self.P[a[0]].mean(axis=0)
        inner_b = self.P[b[0]].mean(axis=0)
        nb = self.N[b[0]][b[1]]
        for R in (_rot(phi - psi), _rot(phi) @ np.diag([1.0, -1.0]) @ _rot(-psi)):
            u = q0 - R @ p0
            if (R @ inner_a + u - q0) @ nb > 0 and (inner_b - q0) @ nb < 0:
                return R, u
        raise ValueError(f"no gluing of sides {a} and {b}")

    def inside(self, k, x, tol=TOL):
        return bool(np.all(self.N[k] @ x - self.c[k] <= tol))

    def on_side(self, k, x, s, tol=TOL):
        return abs(self.N[k][s] @ x - self.c[k][s]) <= tol and self.inside(k, x, tol)

    def corner_at(self, k, x, tol=1e-6):
        d = np.linalg.norm(self.P[k] - x, axis=1)
        j = int(np.argmin(d))
        return j if d[j] < tol else None

    def walk(self, k, x, target, labs, lat):
        """Walk straight from x (in piece k) to `target`, crossing glued sides; `labs` (site labels) are carried
        along. Returns (piece, target, labels, linear part of the composed gluings)."""
        Lin = np.eye(2)
        for _ in range(500):
            if self.inside(k, target):
                return k, target, labs, Lin
            d = target - x
            nd = self.N[k] @ d
            ts = np.where(nd > 1e-12, (self.c[k] - self.N[k] @ x) / np.where(nd > 1e-12, nd, 1.0), np.inf)
            ts = np.maximum(ts, 0.0)
            t = float(ts.min())
            exits = [s for s in range(len(ts)) if ts[s] <= t + 1e-9]
            y = x + t * d
            if len(exits) > 1 or self.corner_at(k, y, 1e-6) is not None:
                raise SurfaceNotFound("a bond or a face is cut by a cone point at this size",
                                      f"a straight path from {np.round(x, 4).tolist()} passes through a corner of the net "
                                      f"(a cone point) in piece {k}", hint="try another size")
            l, s2, g = self.glue[(k, exits[0])]
            R, u = g
            gf = self.gf[(k, exits[0])]
            x, target = R @ y + u, R @ target + u
            labs = [lat.apply(gf, a) for a in labs]
            Lin, k = R @ Lin, l
        raise RuntimeError("walk did not end")


# ----------------------------------------------------------------------------------------------------------------
# the surface
# ----------------------------------------------------------------------------------------------------------------
class ConeSurface(Surface):
    """A Surface glued from pieces, with cone points (disclinations). cones: list of dict(id, point (plane, in the
    net), angle (Frank angle, deg), corner_sum, site (surface site at the cone or None), face, label / location /
    order of its rotation-centre class). Extra checks in checks(); cover: the double cover's picture (projective
    plane), used by plot()."""

    def checks(self, show=True):
        V, E, Fc = len(self.sites), len(self.bonds), len(self.faces)
        pairs = [tuple(sorted((a, b))) for a, b, _ in self.bonds]
        deg = np.zeros(V, int)
        for a, b, _ in self.bonds:
            deg[a] += 1
            deg[b] += 1
        at_cone = {c["site"] for c in self.cones if c.get("site") is not None}
        frank = sum(c["angle"] for c in self.cones)
        out = dict(V=V, E=E, F=Fc, chi=self.chi, V_minus_E_plus_F=V - E + Fc, euler=(V - E + Fc) == self.chi,
                   no_loops=all(a != b for a, b in pairs), no_double_bonds=len(set(pairs)) == len(pairs),
                   faces_at_least_3=all(len(f) >= 3 for f in self.face_sides),
                   plane_coordination=all(deg[k] == self.plane_degree[self.sites[k][0]] for k in range(V) if k not in at_cone),
                   frank_sum=round(frank, 6), gauss_bonnet=abs(frank - 360.0 * self.chi) < 1e-6,
                   net_chi=self.net_chi, net_euler=self.net_chi == self.chi,
                   low_degree=[k for k in range(V) if deg[k] < 3], face_sizes=_hist([len(f) for f in self.face_sides]),
                   degrees=_hist(list(deg)))
        keys = ("no_loops", "no_double_bonds", "faces_at_least_3", "plane_coordination", "euler", "gauss_bonnet", "net_euler")
        out["ok"] = all(out[k] for k in keys)
        if show:
            print(f"{self.kind} of {self.lattice.group}: V {V}, E {E}, F {Fc}, V - E + F = {V - E + Fc} (chi {self.chi}); "
                  f"faces {out['face_sizes']}; cones {len(self.cones)} x Frank "
                  f"{sorted({round(c['angle'], 3) for c in self.cones})} deg, sum {out['frank_sum']:g} = 360 x chi "
                  f"{out['gauss_bonnet']}; no loops {out['no_loops']}, no double bonds {out['no_double_bonds']}, faces >= 3 "
                  f"{out['faces_at_least_3']}, plane coordination away from cones {out['plane_coordination']}"
                  + (f"; {len(out['low_degree'])} site(s) with fewer than 3 bonds" if out["low_degree"] else "")
                  + f"  ->  {'OK' if out['ok'] else 'FAILED'}")
        return out

    def plot_net(self, ax=None, title=None, size=6, legend=True, rotate=False):
        """The net (the pieces unfolded in the plane, exact); sides with the same colour and arrows are glued; thin
        lines: sides of neighbouring pieces, glued as they lie; red rings: the cone points (corners of the net)."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.subplots(figsize=(size, size))[1]
        for P in self.domain["pieces"]:
            P = np.vstack([P, P[:1]])
            ax.plot(P[:, 0], P[:, 1], "-", color="#c9b46a", lw=0.6, zorder=0.5)
        ax = Surface.plot_net(self, ax=ax, title=title or f"{self.name} ({self.lattice.group}) on a {self.kind}: the net "
                              "(exact)", size=size, legend=legend, rotate=False)
        for c in self.cones:
            ax.plot(*np.asarray(c["corners"]).T, "o", ms=8, mfc="none", mec="#d62728", mew=1.4, zorder=6)
        return ax

    def plot(self, ax=None, title=None, size=6, elev=None, azim=None, **kw):
        """3D: the sphere (faces of the solid pushed onto it). Projective plane: its double cover, the sphere, every site
        drawn twice (x and -x) in one colour per site."""
        if getattr(self, "cover", None) is None:
            return Surface.plot(self, ax=ax, title=title, size=size, elev=elev, azim=azim, **kw)
        C = self.cover
        ax = Surface.plot(C, ax=ax, title=title or f"{self.name} ({self.lattice.group}) on a {self.kind}: the double "
                          "cover" + chr(10) + self.xyz_note, size=size, elev=elev, azim=azim, sites=False, **kw)
        X = C.xyz
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=12, c=C.colour, cmap="hsv", depthshade=True, lw=0, zorder=5)
        return ax


def _hist(xs):
    out = {}
    for x in xs:
        out[int(x)] = out.get(int(x), 0) + 1
    return dict(sorted(out.items()))


def pairing_test(lattice, pieces, pairs):
    """The group test alone: for every pair, dict(pair, isometry (words), ok, reason, listed). Raises nothing."""
    lat = lattice if isinstance(lattice, LatticeData) else LatticeData(lattice)
    net = Net(pieces, pairs)
    out = []
    for (a, b, flip) in net.pairs:
        g = net.glue[a][2]
        ok, why, listed = lat.test(g)
        out.append(dict(pair=(a, b), isometry=describe(g), ok=ok, reason=why, listed=listed,
                        orientation_reversing=bool(np.linalg.det(g[0]) < 0)))
    return out


def glue(lattice, pieces, pairs, kind="surface", catalogue=None, params=None, strict=True):
    """The lattice on the surface glued from `pieces` by `pairs` (see Net). Returns a ConeSurface. Refuses
    (SurfaceNotFound) a pairing that is not a symmetry of the lattice, a bond folded at a cone, and, when strict, a result
    with a loop, a double bond or a face with fewer than 3 sides."""
    lat = lattice if isinstance(lattice, LatticeData) else LatticeData(lattice)
    net = Net(pieces, pairs)
    bad = [r for r in pairing_test(lat, pieces, pairs) if not r["ok"]]
    if bad:
        raise SurfaceNotFound(f"{kind} refused for {lat.L.group}: {len(bad)} of {len(net.pairs)} gluings of this net are not "
                              "symmetries of the lattice",
                              "; ".join(f"sides {r['pair'][0]} -> {r['pair'][1]}: {r['isometry']} ({r['reason']})"
                                        for r in bad[:4]) + (" ..." if len(bad) > 4 else ""), hint="try another size or centre")
    net.gf = {}
    for a, (l, s2, g) in net.glue.items():
        Mf, tf = lat.frac(g)
        net.gf[a] = (np.rint(Mf).astype(int), tf)
    # --- copies in every piece ---------------------------------------------------------------------------------
    copies = {"s": {}, "b": {}, "f": {}}          # (piece, key) -> reference point
    bond_ends = {}
    face_labs = {}
    margin = lat.lmax + lat.fmax + 1.0
    for k, P in enumerate(net.P):
        lo, hi = P.min(axis=0) - margin, P.max(axis=0) + margin
        fr = np.array([lat.Ainv @ np.array([x, y]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        r1 = range(int(np.floor(fr[:, 0].min())) - 1, int(np.ceil(fr[:, 0].max())) + 2)
        r2 = range(int(np.floor(fr[:, 1].min())) - 1, int(np.ceil(fr[:, 1].max())) + 2)
        for n1 in r1:
            for n2 in r2:
                for i in range(lat.m):
                    x = lat.pos((i, n1, n2))
                    if net.inside(k, x):
                        copies["s"][(k, (i, n1, n2))] = x
                for (i, j, d1, d2) in lat.bonds:
                    a, b = (i, n1, n2), (j, n1 + d1, n2 + d2)
                    x = (lat.pos(a) + lat.pos(b)) / 2
                    if net.inside(k, x):
                        if net.corner_at(k, x) is not None:
                            raise SurfaceNotFound(f"{kind} refused for {lat.L.group}: a cone point sits on the midpoint of a "
                                                  f"bond ({np.round(x, 4).tolist()}): the cone would fold that bond into a loop",
                                                  hint="try another centre")
                        key = (k, frozenset((a, b)))
                        copies["b"][key] = x
                        bond_ends[key] = (a, b)
                for labs, cen in lat.faces:
                    ls = tuple((i, l1 + n1, l2 + n2) for (i, l1, l2) in labs)
                    x = cen + lat.A @ np.array([n1, n2], float)
                    if net.inside(k, x):
                        copies["f"][(k, frozenset(ls))] = x
                        face_labs[(k, frozenset(ls))] = ls
    # --- identify copies across glued sides ---------------------------------------------------------------------
    uf = {t: _UF() for t in copies}
    links = {}                                     # site copy -> [(other copy, R)] for bond vectors
    for t in copies:
        for c in copies[t]:
            uf[t].add(c)
    for (k, s), (l, s2, g) in net.glue.items():
        R, u = g
        gf = net.gf[(k, s)]
        for t in copies:
            for (kk, key), x in copies[t].items():
                if kk != k or not net.on_side(k, x, s):
                    continue
                key2 = lat.apply(gf, key) if t == "s" else frozenset(lat.apply(gf, a) for a in key)
                c2 = (l, key2)
                if c2 not in copies[t]:
                    raise RuntimeError(f"glued side {(k, s)}: the image of {key} is missing in piece {l}")
                uf[t].union((k, key), c2)
                if t == "s":
                    links.setdefault((k, key), []).append((c2, R))
                    links.setdefault(c2, []).append(((k, key), np.linalg.inv(R)))
    # --- corners -> cone points ----------------------------------------------------------------------------------
    cuf = _UF()
    for k, P in enumerate(net.P):
        for j in range(len(P)):
            cuf.add((k, j))
    for (k, s), (l, s2, g) in net.glue.items():
        R, u = g
        for j in (s, (s + 1) % len(net.P[k])):
            j2 = net.corner_at(l, R @ net.P[k][j] + u)
            cuf.union((k, j), (l, j2))
    vclass = {}
    for c in cuf.p:
        vclass.setdefault(cuf.find(c), []).append(c)
    # --- sites -------------------------------------------------------------------------------------------------
    sroot = {}
    for c in copies["s"]:
        sroot.setdefault(uf["s"].find(c), []).append(c)
    roots = uf["s"].roots()
    sid = {r: n for n, r in enumerate(roots)}
    site_of = {c: sid[uf["s"].find(c)] for c in copies["s"]}
    sites = [r[1] for r in roots]
    uv = np.array([copies["s"][r] for r in roots])
    Lrep = {}
    for r in roots:                                # linear maps from each copy's frame to its representative's
        Lrep[r] = np.eye(2)
        todo = [r]
        while todo:
            c = todo.pop()
            for c2, R in links.get(c, []):         # R carries vectors from frame c to frame c2
                if c2 not in Lrep:
                    Lrep[c2] = Lrep[c] @ np.linalg.inv(R)
                    todo.append(c2)
    # --- bonds: ends by walking from the midpoint ----------------------------------------------------------------
    broot = uf["b"].roots()
    bonds, bond_mid = [], []
    for r in broot:
        k, key = r
        x = copies["b"][r]
        a, b = bond_ends[r]
        ends = []
        for lab in (a, b):
            m, y, labs, Lin = net.walk(k, x, lat.pos(lab), [lab], lat)
            ends.append(((m, labs[0]), Lin))
        (ca, La), (cb, Lb) = ends
        if ca not in site_of or cb not in site_of:
            raise RuntimeError(f"bond {key}: an end was not found among the sites")
        d = lat.pos(b) - lat.pos(a)
        if cb in sroot and ca not in sroot:          # draw the bond from an end whose copy is the drawn one (plot_net)
            bonds.append((site_of[cb], site_of[ca], -(Lrep.get(cb, np.eye(2)) @ Lb @ d)))
        else:
            bonds.append((site_of[ca], site_of[cb], Lrep.get(ca, np.eye(2)) @ La @ d))
        bond_mid.append((k, x, a, b))
    # --- faces: the two faces beside every bond --------------------------------------------------------------------
    froot = uf["f"].roots()
    fid = {r: n for n, r in enumerate(froot)}
    face_sides = [[] for _ in froot]
    for e, (k, x, a, b) in enumerate(bond_mid):
        for p, q in ((a, b), (b, a)):
            dart = (p[0], q[0], q[1] - p[1], q[2] - p[2])
            f, (o1, o2) = lat.dart_face[dart]
            labs, cen = lat.faces[f]
            sh = (p[1] - o1, p[2] - o2)
            ls = [(i, l1 + sh[0], l2 + sh[1]) for (i, l1, l2) in labs]
            c = cen + lat.A @ np.array(sh, float)
            m, y, ls2, _ = net.walk(k, x, c, ls, lat)
            fc = (m, frozenset(ls2))
            if fc not in copies["f"]:
                raise RuntimeError(f"face beside bond {e} not found after the walk")
            face_sides[fid[uf["f"].find(fc)]].append(e)
    faces = [_face_cycle(fs, bonds) for fs in face_sides]
    # --- cones ---------------------------------------------------------------------------------------------------
    cones = []
    for vr in cuf.roots():
        cs = vclass[vr]
        tot = 0.0
        for (k, j) in cs:
            P = net.P[k]
            u1, u2 = P[(j + 1) % len(P)] - P[j], P[j - 1] - P[j]
            tot += math.degrees(math.atan2(u1[0] * u2[1] - u1[1] * u2[0], u1 @ u2))
        frank = 360.0 - tot
        if abs(frank) < 1e-6:
            continue
        k0, j0 = cs[0]
        pt = net.P[k0][j0]
        site = next((site_of[c] for c in copies["s"] if c[0] == k0 and np.linalg.norm(copies["s"][c] - pt) < 1e-6), None)
        face = next((fid[uf["f"].find(c)] for c in copies["f"] if c[0] == k0 and np.linalg.norm(copies["f"][c] - pt) < 1e-6), None)
        cone = dict(point=pt, angle=round(frank, 9), corner_sum=tot, corners=[net.P[k][j] for k, j in cs], site=site, face=face)
        if catalogue is not None:
            rec = catalogue.centre_class_of(pt)
            cone.update(label=None if rec is None else rec["label"], centre_id=None if rec is None else rec["id"],
                        order=None if rec is None else rec["order"], location=None if rec is None else rec["location"])
        cones.append(cone)
    for n, c in enumerate(cones):
        c["id"] = f"K{n + 1}"
    # --- topology of the net -------------------------------------------------------------------------------------
    net_chi = len(vclass) - len(net.pairs) + len(net.P)
    sign = {0: 1}
    orientable, todo = True, [0]
    while todo:
        k = todo.pop()
        for (kk, s), (l, s2, g) in net.glue.items():
            if kk != k:
                continue
            want = sign[k] * (1 if np.linalg.det(g[0]) > 0 else -1)
            if l not in sign:
                sign[l] = want
                todo.append(l)
            elif sign[l] != want:
                orientable = False
    edges = []
    num = 0
    for (a, b, flip) in net.pairs:
        g = net.glue[a][2]
        if np.allclose(g[0], np.eye(2)) and np.linalg.norm(g[1]) < 1e-9:
            continue                                                      # unfolded side: drawn as the shared edge
        num += 1
        col = "#d9432f" if np.linalg.det(g[0]) > 0 else "#2e7d32"
        for (k, s), fl in ((a, False), (b, flip)):
            p, q = net.side(k, s)
            edges.append((q, p, str(num), col) if fl else (p, q, str(num), col))
    dom = dict(polygon=_outline(net), pieces=net.P, edges=edges)
    S = ConeSurface(kind, lat.L, sites, uv, bonds, faces, net_chi, orientable, True, dom, dict(params or {}), lat.plane_degree)
    S.face_sides, S.cones, S.net_chi, S.net, S.lat = face_sides, cones, net_chi, net, lat
    S.site_copies = {sid[r]: sroot[r] for r in roots}
    if strict:
        ch = S.checks(show=False)
        bad = [k for k in ("no_loops", "no_double_bonds", "faces_at_least_3") if not ch[k]]
        if bad:
            raise SurfaceNotFound(f"{kind} refused for {lat.L.group}: the cones fold the lattice ({', '.join(bad)} failed)",
                                  f"faces {ch['face_sizes']}, cones at {[c.get('location') for c in cones]}",
                                  hint="try a larger size or another centre")
    return S


def _face_cycle(sides, bonds):
    """The site cycle of a face from its sides (bond indices), chained end to end."""
    if not sides:
        return []
    rest = list(sides)
    e = rest.pop(0)
    a, b = bonds[e][0], bonds[e][1]
    cyc, cur = [a], b
    while rest:
        nxt = next((x for x in rest if cur in bonds[x][:2]), None)
        if nxt is None:
            return cyc + [cur] + [bonds[x][0] for x in rest]
        rest.remove(nxt)
        cyc.append(cur)
        x, y = bonds[nxt][0], bonds[nxt][1]
        cur = y if x == cur else x
    return cyc


def _outline(net):
    """The boundary of the net in the plane: the sides glued by something other than the identity, chained."""
    sides = []
    for (k, s), (l, s2, g) in net.glue.items():
        if not (np.allclose(g[0], np.eye(2)) and np.linalg.norm(g[1]) < 1e-9):
            sides.append(net.side(k, s))
    if not sides:
        return net.P[0]
    used, poly = [False] * len(sides), [sides[0][0]]
    used[0], cur = True, sides[0][1]
    while not np.allclose(cur, poly[0], atol=1e-6):
        nxt = [j for j in range(len(sides)) if not used[j] and np.allclose(sides[j][0], cur, atol=1e-6)]
        if not nxt:
            return net.P[0] if len(poly) < 3 else np.array(poly)
        used[nxt[0]] = True
        poly.append(cur)
        cur = sides[nxt[0]][1]
    return np.array(poly)
