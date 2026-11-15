"""Every crystalline-symmetry-breaking defect a lattice can carry, and what each looks like.

    cat = DefectCatalogue("square")            # UnitCell, legacy Family, recipe dict, or catalogue name
    cat.summary()                              # printed tables, one per kind, with counts
    cat.dislocations[0].build().plot()         # any entry: build(R=...) -> legacy Defect (rigid picture)
    cat.dislocations[0].plot()                 # entry.plot(ax, R): a planar picture (entry.picture(R))
    cat.show("disclinations")                  # small multiples

CLASSIFICATION (author, 2026-11-13: "homotopy classification"; definitions, set by the author in docs/DEVELOPMENT.md)
  A point defect is labelled by its outer counter-clockwise holonomy H, an element of the lifted proper
  group G~+ (Frank angle kept unreduced), up to conjugation in G~+.
  * Dislocation: H = (I, -b). Conjugating by g = (R, t) in G+ gives (I, -R b), so the classes are the
    orbits of b under the proper point group P+ (proved result, one line; formal verification not performed).
  * Disclination at a rotation centre c: H = (R_alpha, (I - R_alpha) c). Conjugation by g in G+ keeps
    alpha (unreduced) and moves c to g(c), so the classes are (alpha, G+-orbit of c) (proved result, as
    above). +180 and -180 are different classes.
  * Mirror partner (fusion_framework's mu, for G with an improper op m0 = (M0, t0)):
    b -> -M0 b, (alpha, c) -> (alpha, m0(c)). For chiral G it is None.
  Walls and the other mirror/glide-breaking defects (the author's four kinds a-d) are in walls.py.

GROUP. G = isometries mapping sites to sites and bonds to bonds, bond colours ignored (K3), computed by
legacy wallpaper_lattices._symmetry_ops on the primitive cell (computational evidence).
Coordinates: Cartesian, plus fractional coordinates in self.cell (the reduced primitive cell, bond
colours ignored). Wyckoff labels are in the ITA standard setting (K1), via papertilling.groups when it
exists; otherwise fallback labels such as 'order-4 class A'.
"""
from __future__ import annotations

import itertools
import math
from functools import cached_property

import numpy as np

from ..legacy.wallpaper_lattices import (UnitCell, make_cell, primitive_cell, _symmetry_ops,
                                         identify_wallpaper_group, has_crossings)
from ..legacy.lab_defects import (Perfect, dislocation as _ld_dislocation, disclination as _ld_disclination,
                                  rotation_about, describe)

TOL = 1e-6
GENERIC_POINT = (0.371, 0.413)          # fractional point used as the default dislocation core (legacy grid)


# ----------------------------------------------------------------------------------------------
# Input and group core
# ----------------------------------------------------------------------------------------------
def _resolve(lattice):
    """(UnitCell, Family or None) from a UnitCell, Family, recipe dict, catalogue name, or an object
    with a .family or .cell attribute (e.g. papertilling.lattice.Lattice; assumption about its API)."""
    from ..legacy.lattice_families import Family
    if isinstance(lattice, dict):
        fam = Family.from_recipe(lattice)
        return fam.cell(), fam
    if isinstance(lattice, Family):
        return lattice.cell(), lattice
    if isinstance(lattice, (UnitCell, str)):
        return make_cell(lattice), None
    if getattr(lattice, "family", None) is not None:
        return lattice.family.cell(), lattice.family
    c = getattr(lattice, "cell", None)
    if c is not None:
        return (c() if callable(c) else c), None
    raise TypeError(f"cannot read a lattice from {type(lattice).__name__}")


def _clean(g):
    """An isometry with round-off below 1e-12 set to zero (display only)."""
    return tuple(np.where(np.abs(x) < 1e-12, 0.0, x) for x in g)


def _rot90(u):
    return np.array([-u[1], u[0]])


def _canon_dir(u):
    u = np.asarray(u, float) / np.linalg.norm(u)
    return -u if (u[1] < -1e-7 or (abs(u[1]) < 1e-7 and u[0] < 0)) else u


def _orbits(items, ops, image, same):
    """Orbits of items under ops (union-find). Returns lists of indices, in order of first member."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, it in enumerate(items):
        for g in ops:
            im = image(it, g)
            for j, other in enumerate(items):
                if same(im, other):
                    a, b = find(i), find(j)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
                    break
    out = {}
    for i in range(len(items)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


class _Group:
    """G of the lattice on its reduced primitive cell: ops = one (M, t) per point-group element."""

    def __init__(self, cell):
        P = primitive_cell(cell, use_types=False)
        self.cell, self.A = P, P.A
        self.Ainv = np.linalg.inv(self.A)
        self.opsf = [(np.asarray(Mf, int), np.asarray(t, float)) for Mf, t in _symmetry_ops(P, use_types=False)]
        self.ops = [(self.A @ Mf @ self.Ainv, self.A @ t) for Mf, t in self.opsf]
        self.plus = [g for g in self.ops if np.linalg.det(g[0]) > 0]
        self.minus = [g for g in self.ops if np.linalg.det(g[0]) < 0]
        self.chiral = not self.minus
        self.name = identify_wallpaper_group(P, use_types=False)
        self.L = Perfect(P)
        self.scale = max(np.linalg.norm(P.a1), np.linalg.norm(P.a2))
        self.planar = not has_crossings(P)
        self.lmax = max(np.linalg.norm(P.basis[j] + self.A @ np.array([n1, n2]) - P.basis[i])
                        for (i, j, n1, n2, _) in P.bonds)

    def is_lattice(self, v, tol=1e-5):
        f = self.Ainv @ np.asarray(v, float)
        return bool(np.all(np.abs(f - np.round(f)) < tol))

    def reduce(self, p):
        f = self.Ainv @ np.asarray(p, float)
        f = f - np.floor(f + 1e-6)
        f[np.abs(f) < 1e-9] = 0.0
        return self.A @ f

    def same_point(self, p, q):
        return self.is_lattice(np.asarray(p) - np.asarray(q), 1e-4)

    def contains(self, g, tol=1e-5):
        """Is the isometry g = (M, t) (Cartesian) an element of G?  Group-level test, no geometry."""
        for M, t in self.ops:
            if np.allclose(M, g[0], atol=1e-6):
                return self.is_lattice(np.asarray(g[1]) - t, tol)
        return False

    def bond_segments_near(self, c, reach):
        P, A = self.cell, self.A
        n0 = np.round(self.Ainv @ c).astype(int)
        K = int(math.ceil(reach / min(np.linalg.norm(P.a1), np.linalg.norm(P.a2)))) + 2
        for d1 in range(-K, K + 1):
            for d2 in range(-K, K + 1):
                o = A @ (n0 + np.array([d1, d2]))
                for (i, j, n1, n2, _) in P.bonds:
                    p = o + P.basis[i]; q = o + P.basis[j] + A @ np.array([n1, n2])
                    yield p, q


# ----------------------------------------------------------------------------------------------
# Entries
# ----------------------------------------------------------------------------------------------
class DefectEntry(dict):
    """One defect class: a dict of its data (always 'kind', 'id', 'spec'), with build(R) and plot(ax)."""

    def __init__(self, builder, field=None, **data):
        super().__init__(**data)
        self._builder, self._field = builder, field

    def build(self, R=None, **kw):
        """The defect, built by the legacy engine: a lab_defects.Defect."""
        return self._builder(R=R, **kw)

    def picture(self, R=None):
        """(Defect, info): a copy of build(R) drawn as a planar picture (draw.embedded_copy: the rigid legacy
        picture if it has no crossing, fold, overlap or crowded site; else, for a wall, the wall opened (side B
        moved rigidly off the wall line); else relaxed, or repaired locally round the bad sites, or a Tutte
        embedding; if none is clean, the rigid picture, info['stage'] = 'rigid'). info['note'] (draw.
        drawing_note) says what the picture keeps. A Moebius seam keeps its stub picture (info None)."""
        d = self.build(R=R)
        if self["kind"] == "mobius_seam":
            return d, None
        from .draw import embedded_copy
        # a glide-breaking wall that cannot be opened cleanly is repaired locally (a few per lattice, 1-3 s each);
        # other walls (often many, e.g. the twin walls of p1 lattices) fall back to the rigid picture with a note
        return embedded_copy(d, wall_local=self["kind"] == "glide_breaking")

    def plot(self, ax=None, R=None, title=None, embedded=True, wrap=None, **kw):
        """Draw the entry; embedded=False draws the rigid legacy picture as built. When the picture keeps
        crossings, overlaps, crowded sites or folds (or is the rigid fallback), a line saying how many is
        added to the title. wrap: wrap the title and the note at this many characters."""
        import textwrap
        if embedded:
            d, info = self.picture(R=R)
        else:
            d, info = self.build(R=R), None
        title = title or self.get("label", self["id"])
        note = (info or {}).get("note")
        if wrap:
            title = "\n".join(textwrap.fill(t, wrap) for t in str(title).split("\n"))
            note = textwrap.fill(note, wrap) if note else note
        return d.plot(ax=ax, field=self._field, title=f"{title}\n{note}" if note else title, **kw)

    def __repr__(self):
        return f"<{self['kind']} {self['id']}: {self.get('label', '')}>"


def _groups_attr(name):
    """papertilling.groups.<name> if it can be imported now, else None. Looked up at every call, so the
    labels switch to ITA Wyckoff labels automatically once groups.py exists. Signatures used (read from
    groups.py, 2026-11-13): label_point(group, (x, y) ITA conventional fractional) -> dict(name='2a', ...);
    proper_centre_classes(group) -> [dict(name='2a(i)', multiplicity, letter, order, points, ...)]."""
    try:
        from .. import groups as _g
    except Exception:
        return None
    return getattr(_g, name, None)


def _call_label(fn, group, xy):
    try:
        r = fn(group, tuple(float(x) for x in xy))
    except Exception:
        return None
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        for k in ("name", "label", "wyckoff", "letter"):
            if isinstance(r.get(k), str):
                return r[k]
    return None


def _ita_class_name(table, xy, order):
    """Name of the proper_centre_classes record whose points contain xy (mod 1), or None."""
    for rec in table or []:
        if rec.get("order") not in (None, order):
            continue
        for q in rec.get("points", []):
            d = np.asarray(xy, float) - np.asarray(q, float)
            if np.all(np.abs(d - np.round(d)) < 1e-5):
                return rec.get("name")
    return None


# ----------------------------------------------------------------------------------------------
# The catalogue
# ----------------------------------------------------------------------------------------------
class DefectCatalogue:
    """All crystalline-symmetry-breaking defect classes of one lattice (see the module docstring).

    DefectCatalogue(lattice, bmax=2.0, wall_bmax=None)
      bmax      : dislocations: primitive Burgers vectors with |b| <= bmax |b_min|
      wall_bmax : twin walls (kind b): line directions = primitive lattice vectors with
                  |v| <= wall_bmax |b_min| (default: bmax), up to the point group
    Attributes (lists of DefectEntry): dislocations, disclinations, mirrors, domain_walls, twin_walls,
    mobius_seams, glide_breaking; centre_classes (G+-classes of rotation centres); normaliser (dict).
    """

    KINDS = ("dislocations", "disclinations", "mirrors", "domain_walls", "twin_walls", "mobius_seams", "glide_breaking")

    def __init__(self, lattice, bmax=2.0, wall_bmax=None):
        self.input_cell, self._family = _resolve(lattice)
        self.bmax = float(bmax)
        self.wall_bmax = float(bmax if wall_bmax is None else wall_bmax)
        self.grp = _Group(self.input_cell)
        self.cell, self.group, self.chiral = self.grp.cell, self.grp.name, self.grp.chiral
        self.label = getattr(self.input_cell, "label", "lattice")
        self.R0 = 3.5 * self.grp.scale

    # --- ITA frame ----------------------------------------------------------------------------
    @cached_property
    def family(self):
        """The legacy Family (ITA standard setting, same Cartesian frame), or None if standardisation fails."""
        if self._family is not None:
            return self._family
        try:
            from ..legacy.lattice_families import Family
            return Family.from_cell(self.input_cell)
        except Exception:
            return None

    def ita_frac(self, x):
        """Fractional coordinates of the Cartesian point x in the ITA conventional cell, or None."""
        f = self.family
        if f is None:
            return None
        Q, o0 = f.frame
        return np.linalg.inv(f._Ac()) @ (Q.T @ (np.asarray(x, float) - o0))

    # --- 1. dislocations ----------------------------------------------------------------------
    @cached_property
    def b_min(self):
        A = self.grp.A
        return min(np.linalg.norm(A @ np.array(m, float)) for m in itertools.product(range(-3, 4), repeat=2) if m != (0, 0))

    def _primitive_vectors(self, factor):
        """Primitive lattice vectors (gcd 1) with |v| <= factor |b_min|. Bound: |n1| = |v x a2| / det
        <= |v| |a2| / det, so the integer box K below holds them all (proved, one line)."""
        A = self.grp.A
        lim = factor * self.b_min + 1e-6
        K = int(math.floor(lim * max(np.linalg.norm(A[:, 0]), np.linalg.norm(A[:, 1])) / abs(np.linalg.det(A)))) + 1
        return [m for m in itertools.product(range(-K, K + 1), repeat=2)
                if m != (0, 0) and math.gcd(*m) == 1 and np.linalg.norm(A @ np.array(m, float)) <= lim]

    @cached_property
    def dislocations(self):
        g = self.grp
        vecs = self._primitive_vectors(self.bmax)
        Pplus = [Mf for Mf, t in g.opsf if round(np.linalg.det(Mf)) == 1]
        orbs = _orbits(vecs, Pplus, lambda m, M: tuple(int(x) for x in M @ np.array(m)), lambda a, b: a == b)
        ang = lambda m: round(math.degrees(math.atan2(*(g.A @ np.array(m, float))[::-1])) % 360.0, 6) % 360.0
        recs = []
        for o in orbs:
            mem = sorted((vecs[i] for i in o), key=ang)
            rep = mem[0]
            b = g.A @ np.array(rep, float)
            recs.append(dict(b_frac=rep, b=b, length=float(np.linalg.norm(b)), angle=ang(rep), size=len(mem), members=mem))
        recs.sort(key=lambda r: (round(r["length"], 6), r["angle"]))
        where = {m: k for k, r in enumerate(recs) for m in r["members"]}
        M0 = next((Mf for Mf, t in g.opsf if round(np.linalg.det(Mf)) == -1), None)
        at = tuple(g.A @ np.array(GENERIC_POINT))
        out = []
        for k, r in enumerate(recs):
            m = r["b_frac"]
            inv = where.get((-m[0], -m[1]))
            mir = None if M0 is None else where.get(tuple(int(x) for x in -M0 @ np.array(m)))
            b = r["b"]
            spec = dict(kind="dislocation", b=(float(b[0]), float(b[1])), at=at)

            def build(R=None, b=b, at=at, **kw):
                return _ld_dislocation(self.cell, b, core=np.array(at), R=R or self.R0, **kw)
            out.append(DefectEntry(build, field="vec", kind="dislocation", id=f"T{k + 1}",
                                   label=f"T{k + 1}: b = {tuple(m)} cell, |b| = {r['length']:.4g}",
                                   b=b, b_frac=tuple(m), length=r["length"], angle=r["angle"], size=r["size"],
                                   members=r["members"], H=(np.eye(2), -b),
                                   inverse_id=f"T{inv + 1}", self_inverse=(inv == k),
                                   mirror_id=None if mir is None else f"T{mir + 1}", spec=spec))
        return out

    # --- 2. disclinations ---------------------------------------------------------------------
    def location(self, c):
        """'site', 'bond midpoint', 'on a bond', 'face centre' (a rotation centre inside a face of a
        planar lattice), 'inside a face', or 'elsewhere' (lattice with crossing bonds)."""
        g = self.grp
        c = np.asarray(c, float)
        if g.L.locate(c) is not None:
            return "site"
        for p, q in g.bond_segments_near(c, g.lmax):
            d = q - p
            s = float((c - p) @ d / (d @ d))
            if -1e-9 < s < 1 + 1e-9 and np.linalg.norm(p + s * d - c) < 1e-6:
                return "bond midpoint" if abs(s - 0.5) < 1e-6 else "on a bond"
        if not g.planar:
            return "elsewhere"
        return "face centre" if any(g.same_point(c, q) for q, _ in self._all_centres) else "inside a face"

    @cached_property
    def _all_centres(self):
        """Every rotation centre in one cell (mod the lattice), with its order n = |stabiliser in G+|."""
        g = self.grp
        pts = []
        for M, t in g.plus:
            if np.allclose(M, np.eye(2)):
                continue
            for m in itertools.product(range(-2, 3), repeat=2):
                c = g.reduce(np.linalg.solve(np.eye(2) - M, t + g.A @ np.array(m, float)))
                if not any(g.same_point(c, q) for q in pts):
                    pts.append(c)
        order = lambda c: sum(1 for M, t in g.plus if g.same_point(M @ c + t, c))
        return [(c, order(c)) for c in pts]

    @cached_property
    def centre_classes(self):
        """G+-classes of rotation centres. 'gclass' = the G-class (ITA Wyckoff label when groups.py is
        available); a G-class that splits into two G+-classes gets labels '<gclass>/1', '<gclass>/2'."""
        g = self.grp
        pts = [c for c, _ in self._all_centres]
        n_of = [n for _, n in self._all_centres]
        img = lambda c, h: g.reduce(h[0] @ c + h[1])
        plus = _orbits(pts, g.plus, img, g.same_point)
        full = _orbits(pts, g.ops, img, g.same_point)
        gid = {i: k for k, o in enumerate(full) for i in o}
        frac = lambda c: tuple(np.round(g.Ainv @ c, 4) % 1.0)
        recs = []
        for o in plus:
            rep = min((pts[i] for i in o), key=frac)
            recs.append(dict(point=rep, frac=np.array(frac(rep)), order=n_of[o[0]], size=len(o),
                             members=[pts[i] for i in o], g_index=gid[o[0]]))
        recs.sort(key=lambda r: (-r["order"], tuple(r["frac"])))
        # G-class labels: ITA Wyckoff name (groups.label_point) if available, else 'order-n class A, B, ...'.
        # G+-class labels: groups.proper_centre_classes names ('2a(i)', '2a(ii)') if available and
        # consistent, else '<G-class>/1', '/2'.
        fn, pcc = _groups_attr("label_point"), _groups_attr("proper_centre_classes")
        try:
            table = pcc(self.group) if pcc is not None else None
        except Exception:
            table = None
        glabel, letters = {}, {}
        for r in recs:
            xy = self.ita_frac(r["point"])
            r["ita_frac"] = xy
            k = r["g_index"]
            if k in glabel:
                continue
            lab = None if (fn is None or xy is None) else _call_label(fn, self.group, xy)
            if lab is None:
                n = r["order"]
                letters[n] = letters.get(n, -1) + 1
                lab = f"order-{n} class {'ABCDEFGH'[letters[n]]}"
            glabel[k] = lab
        count = {}
        for r in recs:
            count[r["g_index"]] = count.get(r["g_index"], 0) + 1
        names = [None if r["ita_frac"] is None else _ita_class_name(table, r["ita_frac"], r["order"]) for r in recs]
        use_ita = all(nm is not None for nm in names) and len(set(names)) == len(names)
        seen = {}
        for k, r in enumerate(recs):
            r["id"] = f"C{k + 1}"
            r["gclass"] = glabel[r["g_index"]]
            seen[r["g_index"]] = seen.get(r["g_index"], 0) + 1
            if use_ita:
                r["label"] = names[k]
            else:
                r["label"] = r["gclass"] if count[r["g_index"]] == 1 else f"{r['gclass']}/{seen[r['g_index']]}"
            r["split"] = count[r["g_index"]] > 1
            r["location"] = self.location(r["point"])
            r["angles"] = [k2 * 360.0 / r["order"] for k2 in range(-r["order"], r["order"] + 1)
                           if k2 != 0 and abs(k2 * 360.0 / r["order"]) <= 180.0 + 1e-9]
        m0 = g.minus[0] if g.minus else None
        for r in recs:
            r["mirror_of"] = None if m0 is None else self.centre_class_of(m0[0] @ r["point"] + m0[1], recs)["id"]
        return recs

    def centre_class_of(self, c, recs=None):
        """The G+-class record of the point c, or None if c is not a rotation centre."""
        for r in (recs if recs is not None else self.centre_classes):
            if any(self.grp.same_point(c, q) for q in r["members"]):
                return r
        return None

    def check_disclination(self, angle, at):
        """The centre class if `angle` (degrees, unreduced) is allowed at `at`; ValueError otherwise."""
        r = self.centre_class_of(at)
        if r is None:
            raise ValueError(f"{np.round(at, 4).tolist()} is not a rotation centre of {self.group}")
        k = angle * r["order"] / 360.0
        if abs(angle) < 1e-9 or abs(k - round(k)) > 1e-6:
            raise ValueError(f"angle {angle:g} is not allowed at an order-{r['order']} centre "
                             f"(multiples of {360.0 / r['order']:g} degrees only)")
        if not self.grp.contains(rotation_about(at, math.radians(angle))):
            raise ValueError("rotation not in G (internal inconsistency)")
        return r

    def build_disclination(self, angle, at, R=None):
        """Legacy wedge construction, refused (ValueError) when the angle is not allowed at `at`."""
        self.check_disclination(angle, at)
        last = None
        for th0 in (0.123, 0.311, 0.577):
            try:
                return _ld_disclination(self.cell, np.asarray(at, float), math.radians(angle), R=R or self.R0, theta0=th0)
            except AssertionError as e:
                last = e
        raise RuntimeError(f"legacy disclination failed: {last}")

    @cached_property
    def disclinations(self):
        out = []
        cls = {r["id"]: r for r in self.centre_classes}
        for r in self.centre_classes:
            c = r["point"]
            for a in r["angles"]:
                eid = f"D({r['label']}, {a:+.0f})"
                mir = None if r["mirror_of"] is None else f"D({cls[r['mirror_of']]['label']}, {a:+.0f})"
                spec = dict(kind="disclination", angle=float(a), at=(float(c[0]), float(c[1])))
                out.append(DefectEntry(lambda R=None, a=a, c=c, **kw: self.build_disclination(a, c, R=R),
                                       field="j", kind="disclination", id=eid,
                                       label=f"{a:+.0f} deg at {r['label']} ({r['location']})",
                                       angle=float(a), centre=c, centre_frac=r["frac"], centre_class=r["id"],
                                       wyckoff=r["label"], gclass=r["gclass"], order=r["order"], location=r["location"],
                                       H=_clean(rotation_about(c, math.radians(a))), inverse_id=f"D({r['label']}, {-a:+.0f})",
                                       mirror_id=mir, spec=spec))
        return out

    # --- 3. mirror and glide lines -------------------------------------------------------------
    def _line_data(self, u):
        """Shortest period along u, and spacing Delta of lattice lines parallel to u (= det / period)."""
        A = self.grp.A
        per = min(abs(float((A @ np.array(m, float)) @ u)) for m in itertools.product(range(-6, 7), repeat=2)
                  if m != (0, 0) and abs(float((A @ np.array(m, float)) @ _rot90(u))) < 1e-6)
        return per, abs(np.linalg.det(A)) / per

    @cached_property
    def all_lines(self):
        """Every mirror/glide line mod the lattice: dict(direction u, offset c in [0, Delta), glide, period, spacing)."""
        g = self.grp
        out = []
        for M, t in g.minus:
            w, V = np.linalg.eigh((M + M.T) / 2)
            u = _canon_dir(V[:, int(np.argmax(w))]); n = _rot90(u)
            per, dl = self._line_data(u)
            for m in itertools.product(range(-2, 3), repeat=2):
                S = t + g.A @ np.array(m, float)
                c = (float(S @ n) / 2) % dl
                c = 0.0 if dl - c < 1e-6 else c
                tau = float(S @ u) % per
                tau = 0.0 if (tau < 1e-6 or per - tau < 1e-6) else tau
                rec = dict(direction=u, offset=c, glide=tau, period=per, spacing=dl)
                if not any(self._same_line(rec, q) for q in out):
                    out.append(rec)
        return out

    @staticmethod
    def _same_line(a, b):
        if not np.allclose(a["direction"], b["direction"], atol=1e-6) or abs(a["glide"] - b["glide"]) > 1e-5:
            return False
        d = (a["offset"] - b["offset"]) / a["spacing"]
        return abs(d - round(d)) < 1e-5

    def _line_image(self, ln, h):
        u2 = _canon_dir(h[0] @ ln["direction"])
        p = h[0] @ (ln["offset"] * _rot90(ln["direction"])) + h[1]
        dl = ln["spacing"]
        c = float(p @ _rot90(u2)) % dl
        return dict(ln, direction=u2, offset=0.0 if dl - c < 1e-6 else c)

    @cached_property
    def mirrors(self):
        """Classes of mirror and glide lines under G (as legacy catalogue().lines)."""
        from ..legacy.lab_defects import reflection_across
        lines = self.all_lines
        orbs = _orbits(lines, self.grp.ops, self._line_image, self._same_line)
        recs = []
        for o in orbs:
            rep = min((lines[i] for i in o), key=lambda q: (tuple(np.round(-q["direction"], 4)), round(q["offset"], 4)))
            recs.append((rep, len(o)))
        recs.sort(key=lambda x: (x[0]["glide"] > 0, tuple(np.round(-x[0]["direction"], 4)), round(x[0]["offset"], 4)))
        out = []
        for k, (ln, size) in enumerate(recs):
            u, n = ln["direction"], _rot90(ln["direction"])
            p = ln["offset"] * n
            h = (self.cell.basis @ n - ln["offset"]) / ln["spacing"]
            on_site = bool(np.any(np.abs(h - np.round(h)) < 1e-6))
            kind = "mirror" if ln["glide"] == 0.0 else "glide"
            sig = reflection_across(p, u, ln["glide"])
            out.append(DefectEntry(lambda R=None, k=k, **kw: self.mobius_seams[k].build(R=R), kind=kind, id=f"M{k + 1}",
                                   label=f"{kind} along {np.round(u, 3).tolist()} through {np.round(p, 3).tolist()}"
                                         + (f", glide {ln['glide']:.4g}" if kind == "glide" else ""),
                                   direction=u, point=p, offset=ln["offset"], glide=ln["glide"], period=ln["period"],
                                   spacing=ln["spacing"], through_site=on_site, size=size, sigma=sig,
                                   spec=dict(kind="line", sigma=sig, point=tuple(p), direction=tuple(u))))
        return out

    # --- 4. walls (walls.py) -------------------------------------------------------------------
    @cached_property
    def normaliser(self):
        from .walls import normaliser_cosets
        return normaliser_cosets(self)

    @cached_property
    def domain_walls(self):
        from .walls import domain_walls
        return domain_walls(self)

    @cached_property
    def twin_walls(self):
        from .walls import twin_walls
        return twin_walls(self)

    @cached_property
    def mobius_seams(self):
        from .walls import mobius_seams
        return mobius_seams(self)

    @cached_property
    def glide_breaking(self):
        from .walls import glide_breaking
        return glide_breaking(self)

    def compare_legacy(self, show=True):
        from .walls import compare_legacy
        return compare_legacy(self, show=show)

    # --- 5. output ----------------------------------------------------------------------------
    def summary(self, show=True):
        from .walls import summary_text
        s = summary_text(self)
        if show:
            print(s)
        return s

    def show(self, kind="disclinations", entries=None, R=None, ncols=4, size=3.2, embedded=True):
        """Small multiples of the built defects of one kind (or of the given entries); embedded: as
        DefectEntry.plot. Titles and drawing notes are wrapped at 45 characters."""
        import matplotlib.pyplot as plt
        ents = list(entries if entries is not None else getattr(self, kind))
        if kind == "mirrors" and entries is None:
            ents = list(self.mobius_seams)
        if not ents:
            print(f"no {kind} for {self.label} ({self.group})")
            return None
        nr = (len(ents) + ncols - 1) // ncols
        fig, axs = plt.subplots(nr, min(ncols, len(ents)), figsize=(size * min(ncols, len(ents)), size * nr), squeeze=False)
        for ax in axs.ravel():
            ax.axis("off")
        for e, ax in zip(ents, axs.ravel()):
            try:
                e.plot(ax=ax, R=R, title=e.get("label", e["id"]), embedded=embedded, wrap=45)
            except Exception as ex:                      # show the refusal instead of stopping the figure
                ax.set_title(f"{e['id']}: {type(ex).__name__}", fontsize=8)
        fig.suptitle(f"{self.label} ({self.group}): {kind}", fontsize=10)
        fig.tight_layout()
        return fig
