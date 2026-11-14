"""The defect algorithm of the problem "Defects on wallpaper lattices" (C01-C06), in one file.

This file is a VERBATIM copy of three earlier modules of the author, so that the algorithm sits next to the
lattice code. Nothing in the algorithm was changed or simplified:
  part 1 = defect_engine.py     (md5 c6bb953a86157a538d3ca71bf0106d70)
  part 2 = defect_catalogue.py  (md5 b290b777cf936c76860adcb69bff5e56)
  part 3 = pair_engine.py       (md5 8dbb63c192d2396492f356434fd0d9c1)  [C07]
The only edit: the lines of parts 2 and 3 that import defect_engine are removed; e.g. part 2's line "from defect_engine import (...)" is removed, because part 1 is above it.
check_lattice_lab.py re-runs the engine checks (engine_checks.py) on this copy.

You normally use it through lattice_lab.py (the Lab class). Read it here to see how each defect is made.
"""

# ==============================================================================================
# PART 1: defect_engine.py
# ==============================================================================================
"""Defects of a periodic lattice, built from the lattice's own symmetry operations.

Four constructions, for any UnitCell of wallpaper_lattices.py:

  dislocation(...)    translation mismatch round a point  (Volterra cut and reglue, three charts)
  disclination(...)   rotation mismatch round a point     (wedge removed or inserted)
  mobius_strip(...)   reflection or glide mismatch round a strip (the sheet is not orientable)
  wall(...)           a line: one side is the image of the other under an isometry

Each returns a Defect: sites, drawing positions, bonds, and for every bond the data needed to
measure the mismatch round any loop. Nothing depends on an elastic model or a distance cut-off.

An isometry is a pair (M, t): x -> M x + t, in Cartesian coordinates.
"""
import math, itertools
import numpy as np
from .wallpaper_lattices import UnitCell, _locate, PALETTE

TWO_PI = 2 * math.pi


def rot(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def rotation_about(centre, angle):
    M = rot(angle); c = np.asarray(centre, float)
    return M, c - M @ c


def reflection_across(point, direction, glide=0.0):
    """Reflection across the line through `point` along `direction`, followed by a shift `glide`
    (a length) along the line."""
    u = np.asarray(direction, float); u = u / np.linalg.norm(u)
    M = 2 * np.outer(u, u) - np.eye(2)
    p = np.asarray(point, float)
    return M, p - M @ p + glide * u


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


# ==============================================================================================
# The perfect lattice
# ==============================================================================================
class Perfect:
    """Sites and bonds of the perfect lattice. A site is a key (n1, n2, s)."""

    def __init__(self, cell):
        self.cell = cell
        self.A = cell.A; self.Ainv = np.linalg.inv(cell.A)
        self.rule = {s: [] for s in range(len(cell.basis))}          # s -> [(d1, d2, s2, type)]
        for (i, j, d1, d2, t) in cell.bonds:
            self.rule[i].append((d1, d2, j, t)); self.rule[j].append((-d1, -d2, i, t))

    def pos(self, key):
        return self.A @ np.array(key[:2], float) + self.cell.basis[key[2]]

    def locate(self, p):
        """Key of the site at point p, or None."""
        try:
            s, n = _locate(p, self.A, self.Ainv, self.cell.basis, tol=1e-6)
        except ValueError:
            return None
        return (n[0], n[1], s)

    def nbrs(self, key):
        return [((key[0] + d1, key[1] + d2, s2), t) for (d1, d2, s2, t) in self.rule[key[2]]]

    def sites_in_disk(self, centre, R):
        c = np.asarray(centre, float)
        n0 = np.round(self.Ainv @ c).astype(int)
        h = abs(np.linalg.det(self.A)) / max(np.linalg.norm(self.A[:, 0]), np.linalg.norm(self.A[:, 1]))
        K = int(math.ceil(R / h)) + 2
        out = []
        for n1 in range(n0[0] - K, n0[0] + K + 1):
            for n2 in range(n0[1] - K, n0[1] + K + 1):
                for s in range(len(self.cell.basis)):
                    key = (n1, n2, s)
                    if np.linalg.norm(self.pos(key) - c) <= R + 1e-9:
                        out.append(key)
        return out

    def is_symmetry(self, g, probe=3.0):
        """Does the isometry g = (M, t) map the lattice (sites and bonds, types ignored) onto itself?"""
        M, t = g
        for key in self.sites_in_disk((0, 0), probe * np.linalg.norm(self.A[:, 0])):
            img = self.locate(M @ self.pos(key) + t)
            if img is None:
                return False
            mine = {self.locate(M @ self.pos(k2) + t) for k2, _ in self.nbrs(key)}
            if mine != {k2 for k2, _ in self.nbrs(img)}:
                return False
        return True


# ==============================================================================================
# A built defect
# ==============================================================================================
class Defect:
    """sites: labels; positions: for drawing; edge[(u, v)] for both directions holds the bond's data:
       'vec'  perfect-lattice bond vector from u to v (dislocations: summed round a loop = Burgers vector)
       'j'    integer: how many times the bond crosses the seam (disclinations, strips)
    """

    def __init__(self, kind, sites, positions, edge, info=None):
        self.kind, self.sites, self.positions, self.edge = kind, list(sites), np.asarray(positions, float), edge
        self.info = info or {}
        self.edges = sorted({(min(u, v), max(u, v)) for (u, v) in edge})
        self._nbr = None

    # --- graph ---------------------------------------------------------------------------------
    def degrees(self):
        d = np.zeros(len(self.sites), int)
        for u, v in self.edges:
            d[u] += 1; d[v] += 1
        return d

    def to_networkx(self):
        import networkx as nx
        G = nx.Graph(); G.add_nodes_from(range(len(self.sites))); G.add_edges_from(self.edges)
        return G

    def _rotation_system(self):
        if self._nbr is None:
            P = self.positions
            nbr = {k: [] for k in range(len(self.sites))}
            for u, v in self.edges:
                nbr[u].append(v); nbr[v].append(u)
            for u in nbr:
                nbr[u].sort(key=lambda v: math.atan2(P[v][1] - P[u][1], P[v][0] - P[u][0]))
            self._nbr = nbr
        return self._nbr

    def faces(self):
        """Bounded faces of the drawing, each a counter-clockwise list of sites. Needs a drawing
        without crossing bonds."""
        nbr, P = self._rotation_system(), self.positions
        seen, out = set(), []
        for u in nbr:
            for v in nbr[u]:
                if (u, v) in seen:
                    continue
                f, a, b = [], u, v
                while (a, b) not in seen:
                    seen.add((a, b)); f.append(a)
                    k = nbr[b].index(a)
                    a, b = b, nbr[b][k - 1]
                area = 0.5 * sum(cross2(P[f[i]], P[f[(i + 1) % len(f)]]) for i in range(len(f)))
                if area > 1e-12:
                    out.append(f)
        return out

    def boundary_loop(self):
        """Outer boundary of the drawing, counter-clockwise."""
        nbr, P = self._rotation_system(), self.positions
        start = min((k for k in nbr if nbr[k]), key=lambda k: (P[k][1], P[k][0]))
        a, b = start, nbr[start][0]
        first, loop = (a, b), [a]
        while True:
            k = nbr[b].index(a)
            a, b = b, nbr[b][(k + 1) % len(nbr[b])]
            if (a, b) == first:
                break
            loop.append(a)
        return loop

    def loop_sum(self, loop, field):
        """Sum of a bond quantity ('vec' or 'j') along a closed loop of sites."""
        tot = 0
        for i in range(len(loop)):
            tot = tot + self.edge[(loop[i], loop[(i + 1) % len(loop)])][field]
        return tot

    def charged_faces(self, field):
        """[(face, value)] for the faces whose loop sum of `field` is not zero."""
        out = []
        for f in self.faces():
            v = self.loop_sum(f, field)
            if np.linalg.norm(np.atleast_1d(v)) > 1e-9:
                out.append((f, v))
        return out

    def face_sizes(self):
        from collections import Counter
        return dict(sorted(Counter(len(f) for f in self.faces()).items()))

    def core_vertices(self, field):
        """Sites on the faces with a non-zero loop sum, or the core site of a disclination centred
        on a site."""
        if "core_site" in self.info:              # the defect sits on a site, not on a face
            return {self.info["core_site"]}
        return {v for f, _ in self.charged_faces(field) for v in f}

    def core_signature(self, field, radius=3):
        """A fingerprint of the graph around the core: the subgraph within `radius` bonds of the core,
        hashed. Equal cores give equal signatures; use is_isomorphic on the balls to confirm."""
        import networkx as nx
        G = self.to_networkx()
        ball = set()
        for v in self.core_vertices(field):
            ball |= set(nx.single_source_shortest_path_length(G, v, cutoff=radius))
        H = G.subgraph(ball).copy()
        return nx.weisfeiler_lehman_graph_hash(H, iterations=4), H

    # --- drawing -------------------------------------------------------------------------------
    def plot(self, ax=None, field=None, title=None, site_size=8, color=None, highlight="#d9432f"):
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        ax = ax or plt.subplots(figsize=(5.5, 5.5))[1]
        P = self.positions
        seam = [(u, v) for (u, v) in self.edges if self.edge[(u, v)].get("ghost") is not None]
        plain = [(u, v) for (u, v) in self.edges if self.edge[(u, v)].get("ghost") is None]
        # bonds keep the colours of the perfect lattice's bond types
        cols = [PALETTE[int(self.edge[(u, v)].get("type", 0)) % len(PALETTE)] if color is None else color for u, v in plain]
        ax.add_collection(LineCollection([[P[u], P[v]] for u, v in plain], colors=cols, linewidths=1.1))
        for (u, v) in seam:                                   # bonds that close through the seam: two stubs
            for a, q in ((u, self.edge[(u, v)]["ghost"]), (v, self.edge[(v, u)]["ghost"])):
                ax.plot([P[a][0], q[0]], [P[a][1], q[1]], color=highlight, lw=1.0, ls=":")
        ax.scatter(P[:, 0], P[:, 1], s=site_size, c="k", zorder=3)
        if field is not None and not seam and "core_site" not in self.info:
            for f, _ in self.charged_faces(field):
                Q = P[f]; ax.fill(Q[:, 0], Q[:, 1], color=highlight, alpha=0.35, zorder=1)
        if "core_site" in self.info:
            c = P[self.info["core_site"]]; ax.scatter([c[0]], [c[1]], s=60, c=highlight, zorder=4)
        ax.set_aspect("equal"); ax.axis("off")
        if title:
            ax.set_title(title, fontsize=9)
        return ax


def _add(edge, u, v, multi=None, **data):
    """Store a bond in both directions; 'vec' and 'j' change sign with direction.
    multi: a set. If given, a second, different bond between the same two sites is not an error: the
    pair is recorded there and the first bond is kept (the graph stays simple; the pair is a double bond)."""
    if multi is not None and (u, v) in edge and edge[(u, v)].get("j") != data.get("j"):
        multi.add((min(u, v), max(u, v))); return
    back = dict(data)
    for k in ("vec", "j"):
        if k in back:
            back[k] = -back[k]
    for key, val in (((u, v), data), ((v, u), back)):
        if key in edge:
            for k in ("vec", "j"):
                if k in val:
                    assert np.allclose(edge[key][k], val[k]), f"two charts disagree on bond {key}"
        edge[key] = val


# ==============================================================================================
# 1. Dislocation
# ==============================================================================================
def dislocation(cell, b, core=(0.0, 0.0), cut=None, R=6.0):
    """Volterra cut and reglue with the lattice translation b (Cartesian).

    core : Cartesian point where the cut starts.   cut : direction of the cut (default: perpendicular to b)
    The half-strip {core + s cut + t b : s > 0, 0 <= t < 1} is removed and its two sides are glued.
    Three charts: 'above and beside' (s <= 0 or t < 0), unchanged; 'far side' (t >= 1), shifted by -b;
    'below' (s > 0), shifted by -b where t >= 1. Two sites are bonded when a chart holds both and the
    perfect lattice bonds their images there.
    """
    L = Perfect(cell)
    b = np.asarray(b, float); core = np.asarray(core, float)
    assert L.is_symmetry((np.eye(2), b)), "b is not a lattice translation"
    d = np.array([b[1], -b[0]]) / np.linalg.norm(b) if cut is None else np.asarray(cut, float)
    B = np.column_stack([b, d]); Binv = np.linalg.inv(B)
    keys, ts = [], {}
    for key in L.sites_in_disk(core, R):
        t, s = Binv @ (L.pos(key) - core)
        if s > 1e-9 and -1e-9 <= t < 1 - 1e-9:
            continue                                            # removed half-strip
        keys.append(key); ts[key] = (t, s)
    index = {k: i for i, k in enumerate(keys)}
    charts = [
        (lambda t, s: s <= 1e-9 or t < -1e-9, lambda t, s: 0),
        (lambda t, s: t >= 1 - 1e-9, lambda t, s: -1),
        (lambda t, s: s > 1e-9, lambda t, s: -1 if t >= 1 - 1e-9 else 0),
    ]
    edge = {}
    for region, shift in charts:
        where = {}
        for key in keys:
            t, s = ts[key]
            if region(t, s):
                where[L.locate(L.pos(key) + shift(t, s) * b)] = index[key]
        for img, u in where.items():
            for k2, typ in L.nbrs(img):
                v = where.get(k2)
                if v is not None and v != u:
                    _add(edge, u, v, vec=L.pos(k2) - L.pos(img), type=typ)
    # drawing: slide every site along b by an amount going smoothly from -b (far side of the cut) to 0
    P = np.array([L.pos(k) for k in keys])
    sgn = 1.0 if cross2(d, b) > 0 else -1.0
    r = P - (core + 0.5 * b)
    dh = d / np.linalg.norm(d)
    ang = np.arctan2(sgn * (dh[0] * r[:, 1] - dh[1] * r[:, 0]), r @ dh) % TWO_PI
    P = P - np.outer(1 - ang / TWO_PI, b)
    return Defect("dislocation", keys, P, edge, dict(b=b, core=core, cut=d))


# ==============================================================================================
# 2. Disclination
# ==============================================================================================
def disclination(cell, centre, angle, R=6.0, theta0=0.123):
    """Wedge disclination about `centre`. angle > 0 removes a wedge of that angle; angle < 0 inserts one.
    The rotation by `angle` about `centre` must be a symmetry of the lattice.

    Construction: unroll the plane round the centre, so every site gets an angle that keeps growing
    as you go round. Identify the point at angle phi with the point at angle phi + Theta, where
    Theta = 2 pi - angle, rotating its position by Theta. Because the rotation by Theta is a symmetry,
    the bond rule is the same before and after the identification, so the result is perfect
    everywhere except at the centre. The result does not depend on theta0 (where the seam is drawn).
    """
    L = Perfect(cell)
    c = np.asarray(centre, float)
    Theta = TWO_PI - angle
    assert L.is_symmetry(rotation_about(c, angle)), "the rotation is not a symmetry about this centre"
    eps = 1e-7

    def theta(p):                                               # angle in [0, 2 pi) from the seam
        r = p - c
        return (math.atan2(r[1], r[0]) - theta0) % TWO_PI

    reps, lifted = [], {}
    core_key = None
    for key in L.sites_in_disk(c, R):
        p = L.pos(key)
        if np.linalg.norm(p - c) < 1e-9:
            core_key = key; continue
        th = theta(p)
        assert min(th, TWO_PI - th) > 1e-6, "a site lies on the seam: change theta0"
        k = 0
        while th + TWO_PI * k < Theta - eps:
            reps.append((key, k)); lifted[(key, k)] = th + TWO_PI * k
            k += 1
    index = {r_: i for i, r_ in enumerate(reps)}

    def reduce(p, phi):
        j = math.floor((phi + eps) / Theta)
        q = c + rot(-j * Theta) @ (p - c)
        phi2 = phi - j * Theta
        return (L.locate(q), int(math.floor((phi2 + eps) / TWO_PI))), j

    edge, skipped, multi, loops = {}, 0, set(), set()
    through = set()   # fix found in C03: distinct perfect bonds through the centre; skipped // 2 undercounts
                      # when only one end of the bond is a representative (e.g. +180 deg gives 0)
    for (key, k), u in index.items():
        p = L.pos(key)
        for k2, typ in L.nbrs(key):
            q = L.pos(k2)
            if np.linalg.norm(q - c) < 1e-9:
                continue                                        # bond to the core site: added below
            a, bb = p - c, q - c
            if abs(cross2(a, bb)) < 1e-9 and a @ bb < 0:
                skipped += 1                                    # the bond passes through the centre
                through.add(frozenset((key, k2)))               # fix found in C03: count distinct bonds
                continue
            dphi = math.atan2(cross2(a, bb), a @ bb)
            rep, j = reduce(q, lifted[(key, k)] + dphi)
            v = index.get(rep)
            if v is not None and v == u:
                loops.add(u)                                    # a site bonded to its own image
            if v is not None and v != u:
                _add(edge, u, v, multi=multi, j=j, type=typ)
    sites = list(reps)
    info = dict(centre=c, angle=angle, Theta=Theta, bonds_through_centre=len(through),   # fix found in C03: was skipped // 2
               
                double_bonds=sorted(multi), self_bonds=sorted(loops))
    if core_key is not None:
        w = len(sites); sites.append((core_key, 0)); info["core_site"] = w
        for k2, typ in L.nbrs(core_key):
            for k in range(3):
                v = index.get((k2, k))
                if v is not None:
                    _add(edge, w, v, j=0, type=typ)
    P = []
    for (key, k) in sites:
        p = L.pos(key) - c
        if np.linalg.norm(p) < 1e-9:
            P.append(c.copy()); continue
        phi = lifted[(key, k)] * TWO_PI / Theta + theta0
        P.append(c + np.linalg.norm(p) * np.array([math.cos(phi), math.sin(phi)]))
    return Defect("disclination", sites, P, edge, info)


# ==============================================================================================
# 3. Moebius strip
# ==============================================================================================
def mobius_strip(cell, g, length=1, width=3.0):
    """The lattice on a strip whose two ends are glued by h = g followed by (length - 1) further
    lattice periods along the axis, where g = (M, t) is a reflection or glide of the lattice.
    Returns a Defect whose seam-crossing bonds carry j = +1 or -1.
    """
    L = Perfect(cell)
    M, t = g
    assert np.linalg.det(M) < 0, "g must reverse orientation"
    assert L.is_symmetry(g), "g is not a symmetry of the lattice"
    w, V = np.linalg.eigh((M + M.T) / 2)
    u = V[:, int(np.argmax(w))]                                 # axis direction
    n = np.array([-u[1], u[0]])
    x0 = n * (t @ n) / 2                                        # a point of the axis
    glide = float(t @ u)
    # shortest lattice period along the axis
    per = None
    for m in itertools.product(range(-6, 7), repeat=2):
        v = L.A @ np.array(m, float)
        if abs(v @ n) < 1e-9 and v @ u > 1e-9 and (per is None or v @ u < per):
            per = float(v @ u)
    if glide < 0:
        u, glide = -u, -glide
    ell = glide + (length - 1) * per if glide > 1e-9 else length * per
    h = (M, x0 - M @ x0 + ell * u)
    hinv = (M.T, -M.T @ h[1])
    assert L.is_symmetry(h)
    coords = lambda p: (float((p - x0) @ u), float((p - x0) @ n))
    keys = []
    for key in L.sites_in_disk(x0 + 0.5 * ell * u, ell + width + 2):
        a, d = coords(L.pos(key))
        if -1e-9 <= a < ell - 1e-9 and abs(d) <= width + 1e-9:
            keys.append(key)
    index = {k: i for i, k in enumerate(keys)}
    edge = {}
    for key, i in index.items():
        for k2, typ in L.nbrs(key):
            q = L.pos(k2)
            a, d = coords(q)
            if abs(d) > width + 1e-9:
                continue
            j, img = 0, k2
            if a >= ell - 1e-9:
                j, img = 1, L.locate(hinv[0] @ q + hinv[1])
            elif a < -1e-9:
                j, img = -1, L.locate(h[0] @ q + h[1])
            v = index.get(img)
            if v is None or v == i:
                continue
            data = dict(j=j, type=typ)
            if j != 0:
                data["ghost"] = L.pos(key) + 0.5 * (q - L.pos(key))
            _add_strip(edge, i, v, data)
    P = np.array([L.pos(k) for k in keys])
    return Defect("mobius", keys, P, edge, dict(axis=u, point=x0, length=ell, glide=glide, h=h))


def _add_strip(edge, u, v, data):
    edge[(u, v)] = data
    if (v, u) not in edge:
        back = dict(data); back["j"] = -data["j"]
        edge[(v, u)] = back


def is_orientable(defect):
    """Give every site a handedness. A bond that crosses the seam (j odd) flips it. The sheet is
    orientable exactly when this can be done consistently."""
    import networkx as nx
    G = nx.Graph()
    for (u, v), dta in defect.edge.items():
        flip = abs(dta.get("j", 0)) % 2
        G.add_edge((u, 0), (v, flip)); G.add_edge((u, 1), (v, 1 - flip))
    base = defect.to_networkx()
    return nx.number_connected_components(G) == 2 * nx.number_connected_components(base)


# ==============================================================================================
# 4. Wall
# ==============================================================================================
def wall(cell, point, direction, glide=0.0, R=6.0):
    """A straight wall through `point` along `direction`. The side to the left of `direction` (side A)
    is the perfect lattice; side B is its image under sigma = reflection across the wall line,
    followed by a shift `glide` along it.

    If sigma is a symmetry of the lattice, side B is the same lattice: the wall is invisible and the
    perfect patch is returned (info['invisible'] = True).
    Otherwise (rule, a technical choice): sites of side A at distance >= 0 are kept, with their
    images; a site on the line is kept once if sigma fixes it; bonds inside side A are kept, with
    their images; a bond of side A that crosses the line from a site u is replaced by the bond from u
    to its own image sigma(u).
    """
    L = Perfect(cell)
    sig = reflection_across(point, direction, glide)
    u = np.asarray(direction, float); u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]]); x0 = np.asarray(point, float)
    dist = lambda p: float((p - x0) @ n)
    if L.is_symmetry(sig):
        keys = L.sites_in_disk(x0, R)
        index = {k: i for i, k in enumerate(keys)}
        edge = {}
        for key, i in index.items():
            for k2, typ in L.nbrs(key):
                if k2 in index:
                    edge[(i, index[k2])] = dict(type=typ)
        return Defect("wall", keys, [L.pos(k) for k in keys], edge, dict(invisible=True, sigma=sig))
    sites, P, index = [], [], {}
    for key in L.sites_in_disk(x0, R):
        p = L.pos(key)
        dd = dist(p)
        if dd < -1e-9:
            continue
        index[(key, "A")] = len(sites); sites.append((key, "A")); P.append(p)
        q = sig[0] @ p + sig[1]
        if np.linalg.norm(q - p) > 1e-9 and np.linalg.norm(q - x0) <= R + 1e-9:
            index[(key, "B")] = len(sites); sites.append((key, "B")); P.append(q)
    edge, redirected = {}, 0
    doubles = []                                                # fix found in C03: see below
    for key in {k for (k, side) in index}:
        p = L.pos(key)
        on_line = (key, "B") not in index and abs(dist(p)) < 1e-9
        for k2, typ in L.nbrs(key):
            if dist(L.pos(k2)) >= -1e-9:                         # the bond stays in side A
                for side in ("A", "B"):
                    a = index.get((key, side), index.get((key, "A")) if on_line else None)
                    k2_on = (k2, "B") not in index and abs(dist(L.pos(k2))) < 1e-9
                    bq = index.get((k2, side), index.get((k2, "A")) if k2_on else None)
                    if a is not None and bq is not None and a != bq:
                        edge[(a, bq)] = dict(type=typ); edge[(bq, a)] = dict(type=typ)
            else:                                               # the bond crosses the wall
                a, bq = index.get((key, "A")), index.get((key, "B"))
                if a is not None and bq is not None:
                    # fix found in C03: two crossing bonds of u (e.g. a square bond and a dimer) can both be
                    # re-attached to the same pair u - sigma(u). The old line overwrote the first with the
                    # second, so the stored type was whichever came last. Keep the first, record the pair
                    # and both types in info['double_bonds'].
                    if (a, bq) in edge:
                        doubles.append((min(a, bq), max(a, bq), edge[(a, bq)]["type"], typ))
                    else:
                        edge[(a, bq)] = dict(type=typ, across=True); edge[(bq, a)] = dict(type=typ, across=True)
                    redirected += 1
    return Defect("wall", sites, P, edge, dict(invisible=False, sigma=sig, redirected=redirected,
                                               double_bonds=[(u, v) for u, v, _, _ in doubles],   # C03 fix
                                               double_bond_types=doubles))                        # C03 fix


# ==============================================================================================
# PART 2: defect_catalogue.py
# ==============================================================================================
"""All defect types of a periodic lattice, listed from its symmetry group, and their construction.

    cat = catalogue(cell)                 # the types: fast, group theory only
    cores = dislocation_cores(cell, b)    # every core found for one Burgers vector
    built = build_all(cell, cat)          # one built Defect per type (uses defect_engine.py)

The rule (C06):
  translation in G            -> dislocation      one type per class of Burgers vectors
  rotation in G               -> disclination     one type per (class of rotation centre, total angle)
  reflection or glide in G    -> Moebius strip    one type per class of mirror or glide lines
  isometry not in G that maps G's symmetry elements onto themselves
                              -> domain, and walls between domains   one type per coset
An isometry is (M, t): x -> M x + t, Cartesian. G ignores bond colours.
"""
import math, itertools
import numpy as np
import networkx as nx
from .wallpaper_lattices import (UnitCell, primitive_cell, _symmetry_ops, _candidate_matrices, has_crossings)
# (Perfect, Defect, dislocation, ... are defined above, in part 1 of this file)

EPS = 1e-6


def compose(a, b):
    """a after b."""
    return a[0] @ b[0], a[0] @ b[1] + a[1]


def inverse(a):
    return a[0].T, -a[0].T @ a[1]


# ==============================================================================================
# 1. The types
# ==============================================================================================
class Catalogue:
    """cell: the primitive, reduced cell used everywhere; ops: one isometry per point-group element;
    translations, centres, lines, domains: the classes (see catalogue())."""

    def summary(self):
        t = ", ".join(f"|b|={c['length']:.3f} x{c['size']}" for c in self.translations)
        r = ", ".join(f"order {c['order']} at {np.round(c['point'], 3).tolist()}" for c in self.centres) or "none"
        m = sum(1 for c in self.lines if c["kind"] == "mirror"); g = sum(1 for c in self.lines if c["kind"] == "glide")
        return (f"point group of order {len(self.ops)}; Burgers classes: {t}; rotation centres: {r}; "
                f"mirror classes {m}, glide classes {g}; domains {len(self.domains)}")


def catalogue(cell, bmax=1.0, angle_range=(-180.0, 180.0)):
    """The defect types of the lattice.

    bmax        : Burgers vectors up to bmax times the shortest translation
    angle_range : total disclination angles (degrees) listed; a removed wedge is always < 360
    """
    P = primitive_cell(cell, use_types=False)
    A = P.A; Ainv = np.linalg.inv(A)
    L = Perfect(P)
    ops = [(A @ Mf @ Ainv, A @ t) for Mf, t in _symmetry_ops(P, use_types=False)]
    cat = Catalogue(); cat.cell, cat.ops, cat.lattice = P, ops, L

    TOL = 1e-4

    def same_point(p, q):                                       # equal modulo the lattice
        f = Ainv @ (p - q)
        return bool(np.all(np.abs(f - np.round(f)) < TOL))

    def reduce_point(p):
        f = Ainv @ p
        return A @ (f - np.floor(f + TOL))

    def classes(items, image, same):
        """Orbits of a list of items under the group. image(item, g) is the moved item."""
        cls = list(range(len(items)))
        for i, it in enumerate(items):
            for g in ops:
                im = image(it, g)
                for j, other in enumerate(items):
                    if cls[j] != cls[i] and same(im, other):
                        old, new = cls[j], cls[i]
                        cls = [new if c == old else c for c in cls]
        out = {}
        for i, c in enumerate(cls):
            out.setdefault(c, []).append(items[i])
        return list(out.values())

    # --- translations ---------------------------------------------------------------------------
    vecs = [A @ np.array(m, float) for m in itertools.product(range(-4, 5), repeat=2) if m != (0, 0)]
    bmin = min(np.linalg.norm(v) for v in vecs)
    vecs = [v for v in vecs if np.linalg.norm(v) <= bmax * bmin + 1e-6]
    cat.translations = []
    for cl in classes(vecs, lambda v, g: g[0] @ v, lambda p, q: bool(np.allclose(p, q, atol=TOL))):
        cat.translations.append(dict(b=cl[0], length=float(np.linalg.norm(cl[0])), size=len(cl), members=cl))
    cat.translations.sort(key=lambda c: (round(c["length"], 6), -c["size"]))

    # --- rotation centres -----------------------------------------------------------------------
    cen = []
    for M, t in ops:
        if np.linalg.det(M) < 0 or np.allclose(M, np.eye(2)):
            continue
        for m in itertools.product(range(-2, 3), repeat=2):
            c = reduce_point(np.linalg.solve(np.eye(2) - M, t + A @ np.array(m, float)))
            if not any(same_point(c, q) for q in cen):
                cen.append(c)

    def order(c):
        return sum(1 for M, t in ops if np.linalg.det(M) > 0 and same_point(M @ c + t, c))
    cat.centres = []
    for cl in classes(cen, lambda c, g: g[0] @ c + g[1], same_point):
        c = min(cl, key=lambda q: tuple(np.round(q, 4))); n = order(c)
        angs = [k * 360.0 / n for k in range(-4 * n, n) if k != 0 and angle_range[0] - 1e-6 <= k * 360.0 / n <= angle_range[1] + 1e-6]
        cat.centres.append(dict(point=c, order=n, size=len(cl), angles=angs, on_site=L.locate(c) is not None))
    cat.centres.sort(key=lambda c: (-c["order"], tuple(np.round(c["point"], 4))))
    cat.all_centres = [(q, order(q)) for q in cen]            # every centre in one cell, with its order

    # --- mirror and glide lines -----------------------------------------------------------------
    def direction_data(u):
        n = np.array([-u[1], u[0]])
        per = delta = None
        for m in itertools.product(range(-6, 7), repeat=2):
            v = A @ np.array(m, float)
            if abs(v @ n) < 1e-5 and v @ u > 1e-5 and (per is None or v @ u < per - 1e-7):
                per = float(v @ u)
            if abs(v @ n) > 1e-5 and (delta is None or abs(v @ n) < delta - 1e-7):
                delta = float(abs(v @ n))
        return n, per, delta

    def canon(u):
        u = u / np.linalg.norm(u)
        return -u if (u[1] < -1e-7 or (abs(u[1]) < 1e-7 and u[0] < 0)) else u

    def make_line(p, u, gl):                                    # point on the line, direction, glide length
        u = canon(u); n, per, delta = direction_data(u)
        g = abs(gl) % per
        g = 0.0 if (per - g < TOL or g < TOL) else g
        return dict(point=n * float(p @ n), direction=u, glide=g, period=per, spacing=delta)

    def same_line(a, b):
        if abs(cross2(a["direction"], b["direction"])) > TOL or abs(a["glide"] - b["glide"]) > TOL:
            return False
        n = np.array([-a["direction"][1], a["direction"][0]])
        d = float((a["point"] - b["point"]) @ n) / a["spacing"]
        return abs(d - round(d)) < TOL

    lines = []
    for M, t in ops:
        if np.linalg.det(M) > 0:
            continue
        w, V = np.linalg.eigh((M + M.T) / 2)
        u = V[:, int(np.argmax(w))]; n = np.array([-u[1], u[0]])
        for m in itertools.product(range(-2, 3), repeat=2):
            S = t + A @ np.array(m, float)
            ln = make_line(n * (S @ n) / 2, u, S @ u)
            if not any(same_line(ln, q) for q in lines):
                lines.append(ln)
    cat.lines = []
    for cl in classes(lines, lambda ln, g: make_line(g[0] @ ln["point"] + g[1], g[0] @ ln["direction"], ln["glide"]), same_line):
        rec = dict(min(cl, key=lambda q: (tuple(np.round(q["direction"], 4)), round(float(np.linalg.norm(q["point"])), 4))))
        rec["size"] = len(cl)
        rec["kind"] = "mirror" if rec["glide"] == 0.0 else "glide"
        nn = np.array([-rec["direction"][1], rec["direction"][0]])
        rec["through_site"] = any(abs((L.pos(k) - rec["point"]) @ nn) < TOL
                                  for k in L.sites_in_disk(rec["point"], 2.5 * max(np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1]))))
        cat.lines.append(rec)
    cat.all_lines = lines                                       # every mirror and glide line in one cell
    cat.lines.sort(key=lambda c: (c["kind"] != "mirror", tuple(np.round(c["direction"], 4)), round(float(np.linalg.norm(c["point"])), 4)))

    # --- domains: isometries outside G that carry G's symmetry elements onto themselves ---------
    hol = [M for M in _candidate_matrices(P) if np.allclose(Ainv @ M @ A, np.round(Ainv @ M @ A), atol=1e-6)]
    if cat.centres:
        top = max(c["order"] for c in cat.centres)
        anchors = sorted((c for c in cen if order(c) == top), key=lambda q: tuple(np.round(q, 4)))
    else:
        anchors = [P.basis[s] for s in range(len(P.basis))]
    found = []
    for M in hol:
        for c in anchors:
            s = (M, c - M @ anchors[0])
            if L.is_symmetry(s):
                continue
            sinv = inverse(s)
            if not all(L.is_symmetry(compose(compose(s, g), sinv)) for g in ops):
                continue
            if any(L.is_symmetry(compose(inverse(s2["sigma"]), s)) for s2 in found):
                continue
            found.append(dict(sigma=s, what=describe(s)))
    cat.domains = found
    return cat


def describe(s):
    M, t = s
    if np.linalg.det(M) > 0:
        if np.allclose(M, np.eye(2)):
            return f"translation by {np.round(t, 4).tolist()}"
        ang = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        c = np.linalg.solve(np.eye(2) - M, t)
        return f"rotation by {ang:.0f} degrees about {np.round(c, 4).tolist()}"
    w, V = np.linalg.eigh((M + M.T) / 2)
    u = V[:, int(np.argmax(w))]; n = np.array([-u[1], u[0]])
    if u[1] < -1e-7 or (abs(u[1]) < 1e-7 and u[0] < 0):
        u, n = -u, -n
    return (f"reflection across the line through {np.round(n * (t @ n) / 2, 4).tolist()} along "
            f"{np.round(u, 4).tolist()}" + (f", then a shift {t @ u:.4f} along it" if abs(t @ u) > 1e-6 else ""))


# ==============================================================================================
# 2. Dislocation cores: scan where the cut ends, then every way to complete the core
# ==============================================================================================
def _ball(G, core, radius):
    nodes = set()
    for v in core:
        nodes |= set(nx.single_source_shortest_path_length(G, v, cutoff=radius))
    return G.subgraph(nodes)


def perfect_patch(cell, R):
    """A disk of the perfect lattice as a Defect (for face sizes and controls)."""
    L = Perfect(cell)
    keys = L.sites_in_disk((0.0, 0.0), R)
    index = {k: i for i, k in enumerate(keys)}
    edge = {}
    for k, i in index.items():
        for k2, typ in L.nbrs(k):
            if k2 in index:
                edge[(i, index[k2])] = dict(type=typ, vec=L.pos(k2) - L.pos(k), j=0)
    return Defect("perfect", keys, [L.pos(k) for k in keys], edge)


def _split(ring, chords, sizes_only=True):
    """The rings that the non-crossing chords cut a ring into (their sizes, or the rings themselves)."""
    rings = [list(ring)]
    for a, b in chords:
        for i, r in enumerate(rings):
            if a in r and b in r:
                ia, ib = sorted((r.index(a), r.index(b)))
                rings[i:i + 1] = [r[ia:ib + 1], r[ib:] + r[:ia + 1]]
                break
    return tuple(sorted(len(r) for r in rings)) if sizes_only else rings


def dislocation_cores(cell, b, grid=6, R=None, ball=3, complete=True, reconstruct=True, max_candidates=10):
    """Every core found for the Burgers vector b.

    Step 1 (Volterra): the cut end is scanned over grid x grid points of the cell, for the cuts
    perpendicular to b in both senses and along the (up to four) shortest lattice directions not
    parallel to b. Each build is the three-chart cut and reglue of defect_engine.dislocation.
    Step 2 (completions): on the core ring, every set of new bonds between sites that have fewer
    bonds than in the perfect lattice is added, provided the new bonds do not cross each other and no
    site gets more bonds than in the perfect lattice. A new bond is 'lattice-type' when the two sites
    would be neighbours in the perfect lattice after shifting one of them by b; otherwise it is a
    'reconstruction' bond.
    reconstruct=False allows lattice-type bonds only.
    Cores are compared as graphs, within `ball` bonds of the core.
    Returns a list of dicts, simplest first.
    """
    P = primitive_cell(cell, use_types=False)
    L = Perfect(P)
    b = np.asarray(b, float)
    A = P.A
    lmax = max(np.linalg.norm(L.pos(k2) - L.pos((0, 0, s))) for s in L.rule for k2, _ in L.nbrs((0, 0, s)))
    R = R or max(4.5 * np.linalg.norm(b), 3.5 * max(np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1])),
                 (2 * ball + 7) * lmax + np.linalg.norm(b))
    planar = not has_crossings(P)
    perp = np.array([b[1], -b[0]]) / np.linalg.norm(b)
    cuts = [perp, -perp]
    others = sorted((A @ np.array(m, float) for m in itertools.product(range(-3, 4), repeat=2)
                     if m != (0, 0) and abs(cross2(A @ np.array(m, float), b)) > 1e-6), key=lambda v: round(np.linalg.norm(v), 6))
    for v in others:
        v = v / np.linalg.norm(v)
        if len(cuts) < 6 and not any(np.allclose(v, c, atol=1e-6) for c in cuts):
            cuts.append(v)
    perfect_deg = lambda key: len(L.rule[key[2]])
    step = {}
    for s in L.rule:
        for k2, _ in L.nbrs((0, 0, s)):
            step.setdefault((s, k2[2]), []).append(L.pos(k2) - L.pos((0, 0, s)))

    def lattice_type(ku, kv):
        d = L.pos(kv) - L.pos(ku)
        return any(np.allclose(d + m * b, w, atol=1e-6) for m in (-1, 0, 1) for w in step.get((ku[2], kv[2]), []))

    found, notes = [], dict(builds=0, anomalies=0, capped=0, planar=planar)
    perfect_sizes = set(perfect_patch(P, 3.0 * max(np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1]))).face_sizes()) if planar else set()
    notes["perfect_ring_sizes"] = sorted(perfect_sizes)

    def record(G, rings_now, rec, x):
        # the core, defined from the finished graph alone: sites with the wrong number of bonds, and
        # sites on rings whose size does not occur in the perfect lattice
        core = {v for r in rings_now for v in r if G.degree(v) != perfect_deg(x.sites[v])}
        core |= {v for r in rings_now if len(r) not in perfect_sizes for v in r}
        if not core:
            core = {v for r in rings_now for v in r}
        H = _ball(G, core, ball)
        h = nx.weisfeiler_lehman_graph_hash(H, iterations=4)
        for f in found:
            if f["hash"] == h and nx.is_isomorphic(f["graph"], H):
                n = f["count"] + 1
                if rec["volterra"] and not f["volterra"]:       # the same core, now reached by cut and reglue alone
                    f.update(rec)
                f["count"] = n
                return
        rec.update(hash=h, graph=H.copy(), count=1)
        found.append(rec)

    for i in range(grid):
        for j in range(grid):
            core_pt = A @ np.array([(i + 0.371) / grid, (j + 0.413) / grid])
            for cut in cuts:
                x = dislocation(P, b, core=core_pt, cut=cut, R=R)
                notes["builds"] += 1
                G = x.to_networkx()
                deg = x.degrees()
                near = lambda v: np.linalg.norm(L.pos(x.sites[v]) - core_pt) < R - 1.5 * lmax    # away from the patch edge
                if planar:
                    ch = x.charged_faces("vec")
                    if not ch:
                        notes["anomalies"] += 1; continue
                    bm = sum(v for _, v in ch)
                    rings = [f for f, _ in ch]
                    core = {v for f in rings for v in f}
                else:
                    bm = None; rings = []
                    core = {v for v in range(len(x.sites)) if deg[v] != perfect_deg(x.sites[v])
                            and np.linalg.norm(L.pos(x.sites[v]) - core_pt) < R - (ball + 1.5) * lmax}
                if not core or not all(near(v) for v in _ball(G, core, ball)):
                    notes["anomalies"] += 1; continue
                under = {v: int(perfect_deg(x.sites[v]) - deg[v]) for v in core if deg[v] < perfect_deg(x.sites[v])}
                base = dict(b_measured=bm, core_point=core_pt, cut=cut, rings=tuple(sorted(len(f) for f in rings)),
                            under=sorted(under.values(), reverse=True), added=0, lattice_bonds=0, reconstruction_bonds=0,
                            volterra=True, bipartite=nx.is_bipartite(_ball(G, core, ball)), defect=x, chords=[])
                record(G, rings if planar else [sorted(core)], base, x)
                if not (complete and planar and len(rings) == 1):
                    continue
                ring = rings[0]
                # a dangling site is met twice on the ring; its first position is used, and ring sizes
                # count the steps of the walk (a dangling bond is walked twice)
                pos = {}
                for k, v in enumerate(ring):
                    pos.setdefault(v, k)
                cand = [(u, v) for u, v in itertools.combinations(sorted(under), 2) if not G.has_edge(u, v)
                        and (reconstruct or lattice_type(x.sites[u], x.sites[v]))]
                if len(cand) > max_candidates:
                    notes["capped"] += 1; cand = cand[:max_candidates]
                crosses = lambda p, q: len({p[0], p[1], q[0], q[1]}) == 4 and \
                    ((pos[p[0]] < pos[q[0]] < pos[p[1]]) != (pos[p[0]] < pos[q[1]] < pos[p[1]]))
                for r in range(1, len(cand) + 1):
                    for sub in itertools.combinations(cand, r):
                        use = {}
                        for u, v in sub:
                            use[u] = use.get(u, 0) + 1; use[v] = use.get(v, 0) + 1
                        if any(use[v] > under[v] for v in use):
                            continue
                        sub2 = [tuple(sorted(c, key=lambda v: pos[v])) for c in sub]
                        if any(crosses(p, q) for p, q in itertools.combinations(sub2, 2)):
                            continue
                        G2 = G.copy(); G2.add_edges_from(sub)
                        nl = sum(lattice_type(x.sites[u], x.sites[v]) for u, v in sub)
                        left = sorted((under[v] - use.get(v, 0) for v in under if under[v] - use.get(v, 0) > 0), reverse=True)
                        rec = dict(base, rings=_split(ring, sub2), under=left, added=len(sub), lattice_bonds=nl,
                                   reconstruction_bonds=len(sub) - nl, volterra=False,
                                   bipartite=nx.is_bipartite(_ball(G2, core, ball)), chords=list(sub))
                        record(G2, _split(ring, sub2, sizes_only=False), rec, x)
    found.sort(key=lambda f: (not f["volterra"], sum(f["under"]), f["added"], f["rings"]))
    for k, f in enumerate(found, 1):
        f["name"] = f"core {k}"
    return found, notes


# ==============================================================================================
# 3. A bare wall between two domains
# ==============================================================================================
def interface(cell, sigma, point, direction, R=6.0):
    """Side A (left of `direction`): the lattice. Side B: its image under sigma. Each side keeps its own
    bonds; bonds that would cross the line are cut and nothing is added. What symmetry fixes is the
    two domains; the bonding across the wall is not fixed by it."""
    P = primitive_cell(cell, use_types=False)
    L = Perfect(P)
    u = np.asarray(direction, float); u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]]); x0 = np.asarray(point, float)
    M, t = sigma
    sinv = inverse(sigma)
    sites, pos, index = [], [], {}
    for key in L.sites_in_disk(x0, R):
        if (L.pos(key) - x0) @ n >= -1e-9:
            index[(key, "A")] = len(sites); sites.append((key, "A")); pos.append(L.pos(key))
    c2 = sinv[0] @ x0 + sinv[1]
    for key in L.sites_in_disk(c2, R):
        q = M @ L.pos(key) + t
        if (q - x0) @ n < -1e-9:
            index[(key, "B")] = len(sites); sites.append((key, "B")); pos.append(q)
    edge = {}
    for (key, side), i in index.items():
        for k2, typ in L.nbrs(key):
            j = index.get((k2, side))
            if j is not None:
                edge[(i, j)] = dict(type=typ)
    pos = np.array(pos)
    deg = np.zeros(len(sites), int)
    for (i, j) in edge:
        deg[i] += 1
    inner = [i for i in range(len(sites)) if np.linalg.norm(pos[i] - x0) < R - 2.0 * max(np.linalg.norm(P.a1), np.linalg.norm(P.a2))]
    cut_ends = int(sum(len(L.rule[sites[i][0][2]]) - deg[i] for i in inner))
    a = pos[[i for i in range(len(sites)) if sites[i][1] == "A"]]; bb = pos[[i for i in range(len(sites)) if sites[i][1] == "B"]]
    gap = float(np.min(np.linalg.norm(a[:, None, :] - bb[None, :, :], axis=2))) if len(a) and len(bb) else float("nan")
    return Defect("interface", sites, pos, edge, dict(sigma=sigma, cut_bond_ends=cut_ends, closest_approach=gap))


# ==============================================================================================
# 4. Build one of everything
# ==============================================================================================
def build_all(cell, cat=None, grid=6, R=None, strip_length=4, strip_width=None, complete=True):
    """One built object per type. Returns a list of records: kind, label, what, defect (a Defect) and,
    for dislocations, the list of cores."""
    cat = cat or catalogue(cell)
    P = cat.cell
    scale = max(np.linalg.norm(P.a1), np.linalg.norm(P.a2))
    R = R or 3.5 * scale
    out = []
    for c in cat.translations:
        cores, notes = dislocation_cores(P, c["b"], grid=grid, complete=complete)
        out.append(dict(kind="dislocation", label=f"b of length {c['length']:.3f} (class of {c['size']})", cls=c,
                        cores=cores, notes=notes, defect=cores[0]["defect"] if cores else None))
    for c in cat.centres:
        for a in c["angles"]:
            try:
                x = disclination(P, c["point"], math.radians(a), R=R)
            except AssertionError as e:
                out.append(dict(kind="disclination", label=f"{a:+.0f} at order-{c['order']} centre", cls=c, defect=None, error=str(e)))
                continue
            out.append(dict(kind="disclination", label=f"{a:+.0f} degrees at the order-{c['order']} centre "
                            f"{np.round(c['point'], 3).tolist()}" + (" (a site)" if c["on_site"] else ""), cls=c, angle=a, defect=x))
    for c in cat.lines:
        g = reflection_across(c["point"], c["direction"], c["glide"])
        length = strip_length if c["kind"] == "mirror" else max(1, strip_length - 1)
        x = mobius_strip(P, g, length=length, width=strip_width or 1.6 * scale)
        out.append(dict(kind="moebius", label=f"{c['kind']} line along {np.round(c['direction'], 3).tolist()} through "
                        f"{np.round(c['point'], 3).tolist()}" + (f", glide {c['glide']:.3f}" if c["glide"] else ""), cls=c, defect=x))
    for d in cat.domains:
        x = interface(P, d["sigma"], point=P.basis.mean(axis=0) + 0.137 * P.a2, direction=P.a1, R=R)
        out.append(dict(kind="domain wall", label=d["what"], cls=d, defect=x))
    return out


# ==============================================================================================
# PART 3: pair_engine.py
# ==============================================================================================
"""Neutral sets of defects, on an open patch or on a torus (C07).

    dislocation_dipole(cell, b, p1, p2, ...)            Burgers vector b at p1 and -b at p2
    disclination_pair(cell, angle, c1, c2, ...)         +angle at c1 and -angle at c2 (open patch only:
                                                        the pair is a dislocation)
    disclination_quadrupole(cell, angle, c1, c2, strip, ...)   +angle, -angle, -angle, +angle

Each returns a Defect of defect_engine.py. Every bond carries
  'vec'  the perfect-lattice bond vector it copies, in the frame of the site it starts from (K5)
  'g'    (only when it passes through a glued side) the isometry that turns coordinates of the frame
         of its first site into coordinates of the frame of its second site
so the mismatch round any loop can be read off the graph: loop_charge(), charges().

Why a torus needs neutral sets: on a torus glued by two translations T1 and T2, the mismatch round
all the defects together equals T1 T2 T1^-1 T2^-1 = identity. So the total Frank angle is zero and
the total Burgers vector is zero. A pair +angle at c1, -angle at c2 has the mismatch
R(c1, angle) R(c2, -angle), a translation by (R - 1)(c2 - c1) up to sign: the pair is a dislocation.

An isometry is (M, t): x -> M x + t, Cartesian. Angles in degrees; angle > 0 removes a wedge.
"""
import math, itertools
from collections import Counter, deque
import numpy as np
# (Perfect, Defect, rot, ... are defined above, in part 1 of this file)

_TINY = 1e-9


def iso_apply(g, x):
    return g[0] @ np.asarray(x, float) + g[1]


def iso_compose(a, b):
    """a after b."""
    return a[0] @ b[0], a[0] @ b[1] + a[1]


def iso_inverse(a):
    return a[0].T, -a[0].T @ a[1]


def _ident():
    return np.eye(2), np.zeros(2)


# ==============================================================================================
# The perfect lattice the defects are cut into: a disk, or a torus
# ==============================================================================================
class Sheet:
    """Open: the lattice sites within R of `centre`.
    Torus: supercell = n, (n1, n2) or two integer vectors [(p, q), (r, s)], meaning T1 = p a1 + q a2
    and T2 = r a1 + s a2 (a1, a2 of `cell`). Every torus site is represented by its copy in the
    supercell centred on `centre`, so that round the centre the sheet is an ordinary piece of plane.
    """

    def __init__(self, cell, centre, R=None, supercell=None):
        self.cell = cell; self.L = L = Perfect(cell)
        self.centre = np.asarray(centre, float)
        self.lmax = max(np.linalg.norm(L.pos(k2) - L.pos((0, 0, s))) for s in L.rule for k2, _ in L.nbrs((0, 0, s)))
        self.scale = max(np.linalg.norm(cell.a1), np.linalg.norm(cell.a2))
        A = cell.A
        if supercell is None:
            self.torus = False; self.R = float(R)
            self.keys = L.sites_in_disk(self.centre, self.R)
        else:
            self.torus = True
            if np.ndim(supercell) == 0:
                N = np.diag([int(supercell)] * 2)
            elif np.ndim(supercell) == 1:
                N = np.diag([int(x) for x in supercell])
            else:
                N = np.array(supercell, int).T
            if abs(round(np.linalg.det(N))) < 1:
                raise ValueError("the supercell vectors are parallel")
            self.N, self.Ninv = N, np.linalg.inv(N)
            self.n0 = np.round(np.linalg.solve(A, self.centre) - 0.5).astype(int)
            corners = np.array([N @ np.array(c) for c in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))])
            lo = np.floor(corners.min(axis=0)).astype(int) - 1 + self.n0
            hi = np.ceil(corners.max(axis=0)).astype(int) + 1 + self.n0
            cells = sorted({self.rep((i, j)) for i in range(lo[0], hi[0] + 1) for j in range(lo[1], hi[1] + 1)})
            assert len(cells) == abs(round(np.linalg.det(N))), "supercell enumeration failed"
            self.keys = [(i, j, s) for (i, j) in cells for s in range(len(cell.basis))]
            self.T = (A @ N[:, 0], A @ N[:, 1])
        self.index = {k: i for i, k in enumerate(self.keys)}

    def rep(self, n):
        f = self.Ninv @ (np.asarray(n, float) - self.n0)
        return tuple(int(x) for x in np.asarray(n) - self.N @ np.floor(f + 0.5 + 1e-9).astype(int))

    def find(self, key):
        """(site number or None, wrapped?) of the lattice site `key`."""
        if not self.torus:
            return self.index.get(key), False
        r = self.rep(key[:2])
        return self.index.get((r[0], r[1], key[2])), r != tuple(key[:2])

    def clear(self, p):
        """How far the point p is from the edge of the piece of plane the sheet is drawn on."""
        p = np.asarray(p, float)
        if not self.torus:
            return self.R - float(np.linalg.norm(p - self.centre))
        g = self.Ninv @ (np.linalg.solve(self.cell.A, p) - self.n0 - 0.5)
        T1, T2 = self.T; area = abs(cross2(T1, T2))
        h = np.array([area / np.linalg.norm(T2), area / np.linalg.norm(T1)])
        return float(np.min((0.5 - np.abs(g)) * h)) - self.scale

    def winding(self, W):
        """A wrap vector as integers (w1, w2): W = w1 T1 + w2 T2."""
        f = np.linalg.solve(np.column_stack(self.T), W)
        return tuple(int(round(x)) for x in f)


def _finish(kind, sheet, sites, ref, edge, wraps, info, picture):
    d = Defect(kind, sites, ref, edge, info)
    d.info.update(reference=np.asarray(ref, float).reshape(-1, 2), wraps=wraps, torus=sheet.torus,
                  T=getattr(sheet, "T", None), lmax=sheet.lmax)
    if not sheet.torus:
        d.info["pinned"] = [i for i in range(len(sites))
                            if np.linalg.norm(d.info["reference"][i] - sheet.centre) > sheet.R - 1.5 * sheet.lmax]
    set_picture(d, picture)
    return d


# ==============================================================================================
# 1. Dislocation dipole
# ==============================================================================================
def _chart_bonds(L, edge, entries):
    """One chart: entries = [(site number, position of the site in this chart)]. Two sites of the
    chart are bonded when the perfect lattice bonds their positions there (the rule of C01)."""
    where = {}
    for i, x in entries:
        k = L.locate(x)
        assert k is not None, "a chart position is not a lattice site"
        where[k] = i
    for k, u in where.items():
        for k2, typ in L.nbrs(k):
            v = where.get(k2)
            if v is not None and v != u:
                _add(edge, u, v, vec=L.pos(k2) - L.pos(k), type=typ)


def dislocation_dipole(cell, b, p1, p2, R=None, supercell=None, picture="relaxed"):
    """Burgers vector b at p1 and -b at p2 (K5: bond vectors summed counter-clockwise).

    Seen from the right of the line p1 -> p2, the left side is shifted by +b between the two points.
    With d = p2 - p1 this is one of three constructions (info['how']):
      'removal'    -b points to the left of d: the sites of the parallelogram spanned by d and -b,
                   which starts at p1 on the left of the line, are removed, and the two sides are
                   glued. Both ends follow the three-chart rule of C01.
      'insertion'  -b points to the right of d: a second copy of the sites of the parallelogram
                   spanned by d and b is put between the two sides (rule: assumption of C07).
      'glide'      b is along d: no site is removed or added. A bond that crosses the line between p1
                   and p2 is re-attached to the site one b further along the left side; over one b at
                   each end, one side has no partner and its bonds across the line are dropped (rule:
                   assumption of C07). The two cores then sit on these lips, between p - b and p, and
                   not exactly at p1 and p2.
    Sign: b is what the counter-clockwise sum round p1 gives. defect_engine.dislocation(cell, b)
    with its default cut gives the sum -b, so the end at p1 is its dislocation(cell, -b, cut=d).
    """
    L = Perfect(cell)
    b = np.asarray(b, float); p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    d = p2 - p1; ell = float(np.linalg.norm(d)); lb = float(np.linalg.norm(b))
    if lb < _TINY or ell < _TINY:
        raise ValueError("b and p2 - p1 must not be zero")
    if not L.is_symmetry((np.eye(2), b)):
        raise ValueError("b is not a lattice translation")
    beta = -b
    side = cross2(d, beta) / (ell * lb)
    how = "removal" if side > 1e-9 else ("insertion" if side < -1e-9 else "glide")
    mid = (p1 + p2) / 2
    probe = Sheet(cell, mid, R=1.0)
    lmax, scale = probe.lmax, probe.scale
    if supercell is None:
        sheet = Sheet(cell, mid, R=R or ell / 2 + max(8 * lmax + lb, 3.5 * scale))
    else:
        sheet = Sheet(cell, mid, supercell=supercell)
    w = beta if how == "removal" else -beta                       # points to the left of d (or along it)
    corners = [p1, p2] + ([p1 + w, p2 + w] if how != "glide" else [p1 - b, p2 - b])
    if min(sheet.clear(c) for c in corners) < lb + 1.5 * lmax:
        raise ValueError("the sheet is too small for this dipole: use a larger R or supercell")
    keys = sheet.keys
    X = np.array([L.pos(k) for k in keys]).reshape(-1, 2)
    edge, wraps = {}, {}
    if how == "glide":
        dh = d / ell; nh = np.array([-dh[1], dh[0]])
        s = (X - p1) @ dh / ell; n = (X - p1) @ nh
        left = n >= -_TINY
        beta0 = float(b @ dh) / ell
        cut = (min(0.0, -beta0), max(1.0, 1.0 - beta0))
        glued = {False: (0.0, 1.0), True: (-beta0, 1.0 - beta0)}       # right lip, left lip
        sites = [(k, 0) for k in keys]
        ref = X.copy()
        for i, k in enumerate(keys):
            for k2, typ in L.nbrs(k):
                q = L.pos(k2)
                sq = float((q - p1) @ dh) / ell; nq = float((q - p1) @ nh)
                target = q
                if left[i] != (nq >= -_TINY):
                    sc = s[i] + (0.0 - n[i]) / (nq - n[i]) * (sq - s[i])        # where the bond crosses the line
                    if cut[0] < sc < cut[1]:
                        lo, hi = glued[bool(left[i])]
                        if not lo < sc < hi:
                            continue                                         # a lip with no partner
                        target = q + b if left[i] else q - b
                kt = L.locate(target)
                j, wrapped = sheet.find(kt)
                if j is None or j == i:
                    continue
                _add(edge, i, j, vec=q - X[i], type=typ)
                if wrapped:
                    wraps[(i, j)] = L.pos(kt) - X[j]; wraps[(j, i)] = -wraps[(i, j)]
        nrem = nins = 0
    else:
        Binv = np.linalg.inv(np.column_stack([d, w]))
        ST = (X - p1) @ Binv.T
        s, t = ST[:, 0], ST[:, 1]
        band = (s > _TINY) & (s < 1 - _TINY)
        inP = band & (t >= -_TINY) & (t < 1 - _TINY)
        if how == "removal":
            keep = ~inP
            new = -np.ones(len(keys), int); new[keep] = np.arange(keep.sum())
            sites = [(keys[i], 0) for i in np.flatnonzero(keep)]
            ref = X[keep]
            ent = lambda mask, shift=0.0: [(int(new[i]), X[i] + shift) for i in np.flatnonzero(mask & keep)]
            _chart_bonds(L, edge, ent((s <= _TINY) | (t < -_TINY)))                  # behind p1, and the right side
            _chart_bonds(L, edge, ent((s >= 1 - _TINY) | (t < -_TINY)))              # beyond p2, and the right side
            _chart_bonds(L, edge, ent(t >= 1 - _TINY))                               # the left side
            _chart_bonds(L, edge, ent(band & (t < -_TINY)) + ent(band & (t >= 1 - _TINY), -w))   # across the strip
            nrem, nins = int(inP.sum()), 0
        else:
            new = np.arange(len(keys))
            qi = np.flatnonzero(inP)
            sites = [(k, 0) for k in keys] + [(keys[i], 1) for i in qi]
            Q = [(len(keys) + a, X[i]) for a, i in enumerate(qi)]
            ref = np.vstack([X, X[qi] - 0.5 * w]) if len(qi) else X.copy()
            keep = np.ones(len(keys), bool)
            ent = lambda mask, shift=0.0: [(int(i), X[i] + shift) for i in np.flatnonzero(mask)]
            _chart_bonds(L, edge, ent((s <= _TINY) | (t < -_TINY)))                  # behind p1, and the right side
            _chart_bonds(L, edge, ent((s >= 1 - _TINY) | (t < -_TINY)))              # beyond p2, and the right side
            _chart_bonds(L, edge, ent(t >= -_TINY))                                  # the left side
            _chart_bonds(L, edge, Q + ent(t < -_TINY))                               # new strip - right side
            _chart_bonds(L, edge, Q + ent(band & (t >= -_TINY), +w))                 # new strip - left side
            nrem, nins = 0, len(qi)
        if sheet.torus:                                               # bonds that leave the drawn supercell
            for i, k in enumerate(keys):
                if not keep[i]:
                    continue
                for k2, typ in L.nbrs(k):
                    if k2 in sheet.index:
                        continue
                    j, _ = sheet.find(k2)
                    if j is None or not keep[j]:
                        raise ValueError("the torus is too small for this dipole")
                    u, v = int(new[i]), int(new[j])
                    try:
                        _add(edge, u, v, vec=L.pos(k2) - X[i], type=typ)
                    except AssertionError:
                        raise ValueError("the torus is too small: two sites would be joined twice")
                    wraps[(u, v)] = L.pos(k2) - X[j]; wraps[(v, u)] = -wraps[(u, v)]
    off = -b / 2 if how == "glide" else 0.0                      # glide: the cores sit on the lips [p - b, p]
    info = dict(b=b, p1=p1, p2=p2, how=how, removed=nrem, inserted=nins, perfect_sites=len(keys),
                cores=[dict(point=p1 + off, charge=b), dict(point=p2 + off, charge=-b)])
    return _finish("dipole", sheet, sites, ref, edge, wraps, info, picture)


# ==============================================================================================
# 2. A polygon cut out of the sheet, with its sides glued by symmetries of the lattice
# ==============================================================================================
def _bent(a, b, bend):
    """The side a -> b as two segments, bulging by `bend` to the right of a -> b. The bulge decides
    on which side of the cut the lattice sites lying exactly on the line a - b fall."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = (b - a) / np.linalg.norm(b - a)
    return [a, (a + b) / 2 + bend * np.array([u[1], -u[0]]), b]


def _glue(sheet, sides, corners, apexes, merged, picture, kind, info, bend):
    """sides   : counter-clockwise list of dict(P=[points], g=isometry or None). Crossing a side from
                 outside applies g and continues from the partner side; g=None is a free side.
       corners : the corner points of the polygon (a bond through one of them is dropped)
       apexes  : corner points that stay sites if they are lattice sites (rotation centres)
       merged  : [(kept corner, removed corner, m, (A_k, B_k), (A_p, B_p))]: the two corners are one
                 point of the glued sheet; m maps coordinates round the removed corner to coordinates
                 round the kept one; A, B are the corners before and after it (counter-clockwise).
    Two sites are bonded when a perfect-lattice bond of one of them, continued through the glued
    sides, ends on the other."""
    from matplotlib.path import Path
    L = sheet.L; lmax = sheet.lmax
    poly = np.array([p for sd in sides for p in sd["P"][:-1]])
    path = Path(poly)
    segs = []                                                     # (a, b, side number)
    for k, sd in enumerate(sides):
        for a, b in zip(sd["P"][:-1], sd["P"][1:]):
            segs.append((np.asarray(a, float), np.asarray(b, float), k))
    near = lambda p, q: np.linalg.norm(np.asarray(p) - np.asarray(q)) < 1e-7
    kept_c = [np.asarray(c, float) for c in apexes] + [np.asarray(m[0], float) for m in merged]
    gone_c = [np.asarray(m[1], float) for m in merged]

    def removed(x):
        if any(near(x, c) for c in gone_c):
            return True
        if any(near(x, c) for c in kept_c):
            return False
        return bool(path.contains_point(x))

    lo, hi = poly.min(axis=0) - 1.5 * lmax, poly.max(axis=0) + 1.5 * lmax
    close = lambda x: bool(np.all(x >= lo) and np.all(x <= hi))

    def trace(pos, tgt):
        """Follow the straight bond pos -> tgt through the glued sides.
        Returns (end point, G, status): G maps the frame of the start to the frame of the end."""
        G = _ident(); skip = None
        for _ in range(8):
            best = None
            dv = tgt - pos
            tol = 1e-9 * lmax / max(float(np.linalg.norm(dv)), 1e-300)   # audit fix: an absolute length, not a fraction
            for a, b, k in segs:
                if k == skip:                                     # audit fix: we have just come out of this side
                    continue
                e = b - a
                den = cross2(dv, e)
                if abs(den) < 1e-14:
                    continue
                lam = cross2(a - pos, e) / den
                mu = cross2(a - pos, dv) / den
                if tol < lam < 1 - tol and -1e-10 <= mu <= 1 + 1e-10 and (best is None or lam < best[0]):
                    best = (lam, k)
            if best is None:
                return tgt, G, "ok"
            lam, k = best
            x = pos + lam * dv
            if any(np.linalg.norm(x - c) < 1e-7 for c in corners):
                return tgt, G, "corner"
            g = sides[k]["g"]
            if g is None:
                return tgt, G, "free"
            pos, tgt, G = iso_apply(g, x), iso_apply(g, tgt), iso_compose(g, G)
            skip = sides[k]["mate"]
        return tgt, G, "lost"

    keys = sheet.keys
    X = np.array([L.pos(k) for k in keys]).reshape(-1, 2)
    keep = np.array([not (close(x) and removed(x)) for x in X])
    new = -np.ones(len(keys), int); new[keep] = np.arange(keep.sum())
    sites = [(keys[i], 0) for i in np.flatnonzero(keep)]
    ref = X[keep]
    def ang_ccw(u, v):
        a = math.atan2(cross2(u, v), float(u @ v)) % TWO_PI
        return 0.0 if a > TWO_PI - 1e-7 else a                   # audit fix: rounding put a bond along the side at 2 pi
    found, notes = {}, Counter()
    wraps = {}
    for i in np.flatnonzero(keep):
        x = X[i]; k = keys[i]; u = int(new[i])
        if not close(x):                                          # far from the polygon: ordinary bonds
            for k2, typ in L.nbrs(k):
                j, wrapped = sheet.find(k2)
                if j is None or not keep[j]:
                    continue
                v = int(new[j]); vec = L.pos(k2) - x
                found.setdefault((u, v), []).append(dict(type=typ, vec=vec, g=_ident(), plain=True))
                if wrapped:
                    wraps[(u, v)] = L.pos(k2) - X[j]
            continue
        origins = [(x, _ident(), None)]                           # (start point, map to the site's own frame, merged record)
        mine = next((m for m in merged if near(x, m[0])), None)
        if mine is not None:
            origins.append((np.asarray(mine[1], float), mine[2], mine))
        at_corner = any(near(x, c) for c in kept_c)
        for o, m, rec in origins:
            for k2, typ in L.nbrs(L.locate(o)):
                wpos = L.pos(k2); delta = wpos - o
                if (at_corner or rec is not None) and path.contains_point(o + 1e-4 * lmax * delta / np.linalg.norm(delta)):
                    continue                                      # this direction points into the removed polygon
                z, G, status = trace(o, wpos)
                if status != "ok":
                    notes[status] += 1; continue
                for kc, gc, mm, _, _ in merged:                   # the bond ends on a removed corner: it is the kept one
                    if near(z, gc):
                        z, G = iso_apply(mm, z), iso_compose(mm, G); break
                kz = L.locate(z)
                if kz is None or (close(z) and removed(z)):
                    notes["ends in the polygon"] += 1; continue
                j, wrapped = sheet.find(kz)
                if j is None or not keep[j]:
                    continue
                v = int(new[j])
                vec = m[0] @ delta                                # in the frame of the site itself
                Gfull = iso_compose(G, iso_inverse(m))
                data = dict(type=typ, vec=vec, g=Gfull, plain=bool(np.allclose(Gfull[0], np.eye(2)) and np.allclose(Gfull[1], 0)))
                if mine is not None:                              # angle unrolled round a merged corner
                    kc, gc, mm, (Ak, Bk), (Ap, Bp) = mine
                    ext = ang_ccw(np.asarray(Ak) - kc, np.asarray(Bk) - kc)
                    if rec is None:
                        data["ang"] = min(ang_ccw(np.asarray(Ak) - kc, delta), ext)
                    else:
                        data["ang"] = ext + ang_ccw(np.asarray(Ap) - gc, delta)
                found.setdefault((u, v), []).append(data)
                if wrapped:
                    wraps[(u, v)] = L.pos(kz) - X[j]
    edge, doubles, lonely = {}, [], 0
    for (u, v), lst in found.items():
        if (u, v) in edge:                                        # already stored as the partner of (v, u)
            continue
        if u == v:
            notes["self-bond"] += 1; continue
        back = found.get((v, u))
        if not back:
            lonely += 1; continue
        if len(lst) > 1:
            doubles.append((min(u, v), max(u, v)))
        e = dict(lst[0])
        # the same bond seen from the other end: its frame map is the inverse
        e2 = next((c for c in back if np.allclose(iso_compose(c["g"], e["g"])[0], np.eye(2), atol=1e-7)
                   and np.allclose(iso_compose(c["g"], e["g"])[1], 0, atol=1e-7)), None)
        if e2 is None:
            lonely += 1; continue
        e2 = dict(e2)                                             # audit fix: store this partner, not the first of the list
        for x in (e, e2):
            if not x.pop("plain"):
                x["seam"] = True
        edge[(u, v)] = e; edge[(v, u)] = e2
    for key in list(edge):
        if (key[1], key[0]) not in edge:
            del edge[key]; lonely += 1
    wraps = {k: w for k, w in wraps.items() if k in edge}
    for (u, v) in list(wraps):
        wraps.setdefault((v, u), -wraps[(u, v)])
    info = dict(info)
    info.update(perfect_sites=len(keys), removed=int((~keep).sum()), inserted=0, double_bonds=sorted(set(doubles)),
                dropped=dict(notes), unmatched_bonds=lonely, polygon=poly, bend=bend)
    return _finish(kind, sheet, sites, ref, edge, wraps, info, picture)


def _pair_geometry(L, angle, c1, c2):
    alpha = math.radians(angle)
    if not 0 < angle <= 180 + 1e-9:
        raise ValueError("the angle of the pair must be between 0 and 180 degrees (it is removed at c1 and inserted at c2); "
                         "a wedge wider than a half turn is not the triangle this construction cuts out")
    c1 = np.asarray(c1, float); c2 = np.asarray(c2, float)
    if np.linalg.norm(c2 - c1) < 1e-9:
        raise ValueError("the two disclinations must be at different points")
    rho = rotation_about(c1, alpha)
    for c in (c1, c2):
        if not L.is_symmetry(rotation_about(c, alpha)):
            raise ValueError(f"the rotation by {angle:g} degrees about {np.round(c, 4).tolist()} is not a symmetry of the lattice")
    c2p = iso_apply(rho, c2)
    tau = c2p - c2
    assert L.is_symmetry((np.eye(2), tau)), "the mismatch of the pair is not a lattice translation"
    return alpha, c1, c2, rho, c2p, tau


def _away(c1, c2, c2p, tau):
    """Unit vector that leaves the base c2 - c2' on the side away from c1. For a half turn c1 lies on
    the base; then the side is the one that makes the cut polygon counter-clockwise."""
    out = (c2 + c2p) / 2 - c1
    if np.linalg.norm(out) > 1e-7 * np.linalg.norm(tau):
        return out / np.linalg.norm(out)
    th = tau / np.linalg.norm(tau)
    return np.array([th[1], -th[0]])               # c2 -> c3 turns left into c3 -> c3' -> c2' -> c1


def _check_side(c1, c2, tau, v, name):
    lt = np.linalg.norm(tau); lv = np.linalg.norm(v)
    sv = cross2(tau, v) / (lt * lv)
    if abs(sv) < 1e-7:
        raise ValueError(f"{name} must not be parallel to the base c2 - c2' of the cut")
    sc = cross2(tau, c1 - c2) / (lt * np.linalg.norm(c1 - c2))
    if abs(sc) < 1e-7:                             # a half turn: only the orientation matters
        if sv > 0:
            raise ValueError(f"{name} must leave the base c2 - c2' on its right, seen from c2 to c2' (try the opposite direction)")
    elif sv * sc > 0:
        raise ValueError(f"{name} must leave the base c2 - c2' on the side away from c1 (try the opposite direction)")


def disclination_pair(cell, angle, c1, c2, cut=None, R=None, picture="relaxed", bend=1e-3):
    """+angle at c1 (a wedge removed) and -angle at c2 (the same wedge inserted), on an open patch.

    The triangle c1, c2, c2' with c2' = R(c1, angle) c2 is cut out and its sides c1-c2 and c1-c2' are
    glued by the rotation: c2 and c2' become one point, where the -angle disclination sits. The two
    ends of the base c2-c2' differ by the lattice translation tau = c2' - c2, so the pair is a
    dislocation: the half-strip from the base to the edge of the patch, along `cut` (default: straight
    away from c1), is cut out too and its sides are glued by tau. loop_charge() of the outer boundary
    gives the Burgers vector of the pair.
    """
    L = Perfect(cell)
    alpha, c1, c2, rho, c2p, tau = _pair_geometry(L, angle, c1, c2)
    v = _away(c1, c2, c2p, tau) if cut is None else np.asarray(cut, float) / np.linalg.norm(cut)
    _check_side(c1, c2, tau, v, "the cut")
    mid = (c1 + c2 + c2p) / 3
    probe = Sheet(cell, mid, R=1.0)
    R = R or max(np.linalg.norm(c2 - c1), np.linalg.norm(tau)) + max(9 * probe.lmax, 3.5 * probe.scale)
    sheet = Sheet(cell, mid, R=R)
    if min(sheet.clear(q) for q in (c1, c2, c2p)) < 2.5 * probe.lmax:
        raise ValueError("the patch is too small for this pair: use a larger R")
    far = 6 * R
    c3, c3p = c2 + far * v, c2p + far * v
    eps = bend * sheet.lmax
    T = (np.eye(2), tau)
    S1 = _bent(c1, c2, eps); S2 = _bent(c2, c3, eps)
    S6 = [iso_apply(rho, p) for p in S1][::-1]                    # c2' -> c1
    S5 = [p + tau for p in S2][::-1]                              # c3' -> c2'
    sides = [dict(P=S1, g=rho, mate=4), dict(P=S2, g=T, mate=3), dict(P=[c3, c3p], g=None, mate=None),
             dict(P=S5, g=iso_inverse(T), mate=1), dict(P=S6, g=iso_inverse(rho), mate=0)]
    merged = [(c2p, c2, rho, (c3p, c1), (c1, c3))]
    info = dict(angle=angle, c1=c1, c2=c2, c2_kept=c2p, tau=tau, cut=v,
                cores=[dict(point=c1, charge=+angle), dict(point=c2p, charge=-angle, also=c2)])
    return _glue(sheet, sides, [c1, c2, c2p], [c1], merged, picture, "disclination pair", info, bend)


def disclination_quadrupole(cell, angle, c1, c2, strip, R=None, supercell=None, picture="relaxed", bend=1e-3):
    """+angle at c1, -angle at c2, -angle at c2 + strip, +angle at c4: total Frank angle and total
    Burgers vector are zero, so it fits on a torus.

    The hexagon c1, c2, c3, c4, c3', c2' is cut out, with c2' = R(c1, angle) c2, tau = c2' - c2,
    c3 = c2 + strip, c3' = c2' + strip, c4 = c2 + c2' - c1 + strip. Its sides are glued in pairs:
    c1-c2 with c1-c2' by the rotation about c1; c2-c3 with c2'-c3' by the translation tau; c3-c4
    with c3'-c4 by the rotation about c4. `strip` is a lattice translation that leaves the base
    c2 - c2' on the side away from c1. The corners c2, c2' become one point (a -angle disclination),
    and so do c3, c3'.
    """
    L = Perfect(cell)
    alpha, c1, c2, rho, c2p, tau = _pair_geometry(L, angle, c1, c2)
    v = np.asarray(strip, float)
    if np.linalg.norm(v) < 1e-9 or not L.is_symmetry((np.eye(2), v)):
        raise ValueError("strip must be a lattice translation, not zero")
    _check_side(c1, c2, tau, v, "strip")
    c3, c3p = c2 + v, c2p + v
    c4 = c2 + c2p - c1 + v
    sigma = rotation_about(c4, -alpha)                            # maps c3 to c3'
    assert np.allclose(iso_apply(sigma, c3), c3p, atol=1e-7)
    assert L.is_symmetry(sigma), "the rotation about c4 is not a symmetry"
    hexa = [c1, c2, c3, c4, c3p, c2p]
    area = 0.5 * sum(cross2(hexa[i], hexa[(i + 1) % 6]) for i in range(6))
    if area <= 0:
        raise ValueError("the cut polygon is not counter-clockwise: change strip")
    for i in range(6):
        for j in range(i + 2, 6):
            if (i, j) == (0, 5):
                continue
            a, b_, c, e = hexa[i], hexa[(i + 1) % 6], hexa[j], hexa[(j + 1) % 6]
            den = cross2(b_ - a, e - c)
            if abs(den) > 1e-12:
                lam = cross2(c - a, e - c) / den; mu = cross2(c - a, b_ - a) / den
                if 0 < lam < 1 and 0 < mu < 1:
                    raise ValueError("the cut polygon crosses itself: change strip")
    mid = np.mean(hexa, axis=0)
    probe = Sheet(cell, mid, R=1.0)
    span = max(np.linalg.norm(p - mid) for p in hexa)
    if supercell is None:
        sheet = Sheet(cell, mid, R=R or span + max(9 * probe.lmax, 3.5 * probe.scale))
    else:
        sheet = Sheet(cell, mid, supercell=supercell)
    if min(sheet.clear(p) for p in hexa) < 2.5 * probe.lmax:
        raise ValueError("the sheet is too small for this quadrupole: use a larger R or supercell")
    eps = bend * sheet.lmax
    T = (np.eye(2), tau)
    S1 = _bent(c1, c2, eps); S2 = _bent(c2, c3, eps); S3 = _bent(c3, c4, eps)
    S4 = [iso_apply(sigma, p) for p in S3][::-1]                  # c4 -> c3'
    S5 = [p + tau for p in S2][::-1]                              # c3' -> c2'
    S6 = [iso_apply(rho, p) for p in S1][::-1]                    # c2' -> c1
    sides = [dict(P=S1, g=rho, mate=5), dict(P=S2, g=T, mate=4), dict(P=S3, g=sigma, mate=3),
             dict(P=S4, g=iso_inverse(sigma), mate=2), dict(P=S5, g=iso_inverse(T), mate=1), dict(P=S6, g=iso_inverse(rho), mate=0)]
    merged = [(c2p, c2, rho, (c3p, c1), (c1, c3)), (c3p, c3, T, (c4, c2p), (c2, c4))]
    info = dict(angle=angle, c1=c1, c2=c2, c2_kept=c2p, c3=c3, c3_kept=c3p, c4=c4, tau=tau, strip=v,
                cores=[dict(point=c1, charge=+angle), dict(point=c2p, charge=-angle, also=c2),
                       dict(point=c3p, charge=-angle, also=c3), dict(point=c4, charge=+angle)])
    return _glue(sheet, sides, hexa, [c1, c4], merged, picture, "disclination quadrupole", info, bend)


# ==============================================================================================
# 3. Reading the result off the graph
# ==============================================================================================
def frame_faces(d):
    """The faces of the sheet: closed walks that always take the next bond round a site. The order of
    the bonds round a site comes from the directions of the perfect-lattice bonds they copy ('vec', in
    the frame of the site), not from the drawing. Needs a lattice without crossing bonds."""
    nbr = {}
    for (u, v), e in d.edge.items():
        a = e["ang"] if "ang" in e else math.atan2(e["vec"][1], e["vec"][0]) % TWO_PI
        nbr.setdefault(u, []).append((a, v))
    nbr = {u: [v for _, v in sorted(lst)] for u, lst in nbr.items()}
    seen, faces = set(), []
    for u in nbr:
        for v in nbr[u]:
            if (u, v) in seen:
                continue
            f, a, b = [], u, v
            while (a, b) not in seen:
                seen.add((a, b)); f.append(a)
                k = nbr[b].index(a)
                a, b = b, nbr[b][k - 1]
            faces.append(f)
    return faces


def loop_charge(d, loop):
    """Go once round a closed loop of sites. Returns (H, S):
    H  the isometry by which the frame has changed on return (identity if nothing is enclosed)
    S  the bond vectors summed along the loop, each carried back to the frame of the first site.
       For a loop that encloses no net rotation, S is the Burgers vector of what it encloses (K5,
       for a counter-clockwise loop)."""
    Dm = _ident(); H = _ident(); S = np.zeros(2)
    for k in range(len(loop)):
        e = d.edge[(loop[k], loop[(k + 1) % len(loop)])]
        S = S + Dm[0] @ e["vec"]
        g = e.get("g", _ident())
        Dm = iso_compose(Dm, iso_inverse(g)); H = iso_compose(g, H)
    return H, S


def _classify(H, S, base):
    M = H[0]
    if np.allclose(M, np.eye(2), atol=1e-7):
        if np.linalg.norm(S) < 1e-7:
            return None
        return dict(kind="translation", vector=S)
    ang = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    c = np.linalg.solve(np.eye(2) - M, H[1])
    return dict(kind="rotation", angle=ang, centre=c)


def charges(d):
    """Every face whose loop has a mismatch: [(face, dict)]. On an open patch the outer boundary (the
    longest face) is returned separately: (list, outer) with outer = (H, S) of the boundary taken
    counter-clockwise round the patch."""
    faces = frame_faces(d)
    outer = None
    if not d.info.get("torus"):
        k = max(range(len(faces)), key=lambda i: len(faces[i]))
        outer = loop_charge(d, faces[k][::-1])
        faces = faces[:k] + faces[k + 1:]
    out = []
    for f in faces:
        H, S = loop_charge(d, f)
        c = _classify(H, S, f[0])
        if c is not None:
            out.append((f, c))
    return out, outer


def euler_characteristic(d):
    """Sites - bonds + faces. 0 on a torus; 2 on an open patch (counting the outside as a face)."""
    return len(d.sites) - len(d.edges) + len(frame_faces(d))


def torus_loop(d, site, winding=(1, 0)):
    """The shortest closed walk through `site` that goes once round the torus in the direction
    `winding` = (1, 0) or (0, 1). Returns (walk, S) with S the summed bond vectors."""
    T = d.info["T"]; wr = d.info["wraps"]
    Tm = np.column_stack(T)
    wind = {k: tuple(int(round(x)) for x in np.linalg.solve(Tm, w)) for k, w in wr.items()}
    nbrs = {}
    for (u, v) in d.edge:
        nbrs.setdefault(u, []).append(v)
    start, goal = (site, (0, 0)), (site, tuple(winding))
    prev = {start: None}; dq = deque([start])
    while dq:
        cur = dq.popleft()
        if cur == goal:
            break
        u, w = cur
        for v in nbrs.get(u, []):
            dw = wind.get((u, v), (0, 0))
            nxt = (v, (w[0] + dw[0], w[1] + dw[1]))
            if max(abs(nxt[1][0]), abs(nxt[1][1])) > 1 or nxt in prev:
                continue
            prev[nxt] = cur; dq.append(nxt)
    if goal not in prev:
        return None, None
    walk = []; cur = goal
    while cur is not None:
        walk.append(cur[0]); cur = prev[cur]
    walk = walk[::-1][:-1]
    return walk, loop_charge(d, walk)[1]


# ==============================================================================================
# 4. The picture
# ==============================================================================================
def relaxed_positions(d):
    """A drawing: every bond is asked to be the perfect-lattice bond vector it copies, and the positions
    that do this best in the least-squares sense are taken (one linear solve). It closes the cuts. It
    is a picture, not an elastic calculation: no distance of the drawing is used anywhere."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    n = len(d.sites); X0 = d.info["reference"]; wr = d.info.get("wraps", {})
    rows, cols, vals = [], [], []
    B = np.zeros((n, 2))
    for (u, v) in d.edges:
        r = 0.5 * (d.edge[(u, v)]["vec"] - d.edge[(v, u)]["vec"]) - wr.get((u, v), 0.0)
        rows += [u, u, v, v]; cols += [u, v, v, u]; vals += [1.0, -1.0, 1.0, -1.0]
        B[u] -= r; B[v] += r
    a = np.full(n, 1e-6)
    for i in d.info.get("pinned", []):
        a[i] = 1e3
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr() + sp.diags(a)
    Y = np.column_stack([spsolve(A.tocsc(), B[:, k] + a * X0[:, k]) for k in range(2)])
    if d.info.get("torus"):
        Y = Y - Y.mean(axis=0) + X0.mean(axis=0)
    return Y


def set_picture(d, picture="relaxed"):
    """picture="relaxed": relaxed_positions(); only the bonds that leave the drawn supercell of a
    torus are drawn as dotted half-bonds. picture="cut": the sites where they sit in the perfect
    lattice, with the cut open; the bonds across the cut are dotted half-bonds too."""
    wr = d.info.get("wraps", {}); X0 = d.info["reference"]
    d.positions = relaxed_positions(d) if picture == "relaxed" else X0.copy()
    d._nbr = None
    P = d.positions
    for (u, v), e in d.edge.items():
        e.pop("ghost", None)
        r = 0.5 * (e["vec"] - d.edge[(v, u)]["vec"])
        long_ = np.linalg.norm(X0[v] + wr.get((u, v), 0.0) - X0[u] - r) > 1e-6
        stretched = (u, v) not in wr and np.linalg.norm(P[v] - P[u]) > 1.8 * np.linalg.norm(r)
        if (u, v) in wr or (picture == "cut" and long_) or stretched:
            e["ghost"] = P[u] + 0.5 * r
    d.info["picture"] = picture
    return d
