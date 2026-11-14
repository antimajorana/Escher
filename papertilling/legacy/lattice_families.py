"""
lattice_families: lattices of a wallpaper group as FAMILIES with free parameters, and exhaustive
lists of families with a given feature.

    fam = Family.from_cell("chiral_square")        # any catalogue lattice, UnitCell or recipe
    fam.describe()                                 # group, Wyckoff positions, bond orbits, free parameters
    fam.set(x1=0.2, y1=0.1).plot()                 # move a free parameter: same network, same group
    fam.print_special()                            # where in parameter space the group jumps up
    fam.print_scan(fam.scan_rotation(lo=-30, hi=30))   # rotate a rigid unit, group along the way

    res = enumerate_families("p4", sites=4, degrees=3)   # every family within stated bounds
    res.show()

DEFINITIONS (mine, as code choices; each can be changed by a parameter or by editing this file)

family      A wallpaper group G in its ITA standard setting, one generating point per site orbit
            (each on a Wyckoff position of G), and one generating bond per bond orbit. The sites
            and bonds of the lattice are all images under G. The BOND NETWORK is fixed; the
            FREE PARAMETERS are the cell parameters G leaves free (b for rectangular cells, b and
            gamma for oblique ones; a is the overall scale) and the free coordinates of each
            generating point (none on a rotation centre, one along a mirror line, two in general
            position). Moving a free parameter can never lower the symmetry below G.
labels      Sites are labelled once, at construction: site k = (orbit, group operation), with a
            fixed integer cell shift. Its position is an affine function of the parameters, and a
            bond is a label (i, j, n1, n2): site i of cell (0,0) to site j of cell (n1, n2), in the
            PRIMITIVE cell of G. Labels do not change when parameters move.
generic     A property is generic if it holds at random parameter values (it then holds everywhere
            except on a set of measure zero). The generic group of a family contains G; the family
            is EXACT if its generic group is G with G's translations.
special locus  A set of parameter values on which the lattice has a symmetry it does not have
            generically. Method (see Family.loci): every extra symmetry h that maps the translation
            lattice of G to itself acts on the labels as an automorphism of the labelled periodic
            graph, (site k, cell n) -> (site pi(k), cell Mf n + s_k), with Mf an integer matrix.
            For a fixed automorphism the conditions "h is an isometry of the cell" and "h moves
            every site onto its image" are LINEAR in (b^2, b cos gamma) and in the site
            coordinates, so each locus is an affine subspace, given by explicit equations.
            All automorphisms are enumerated by backtracking. A supergroup with MORE translations
            contains a new pure translation, which is such an h (Mf = 1), so its locus is listed
            too. What is not resolved: finer strata inside a locus that has more translations.
Maxwell count   2 x (sites per primitive cell) - (bonds per primitive cell). Zero: isostatic.

LABELS OF THE CLAIMS THIS MODULE MAKES
* The list of special loci: complete under the assumptions above (integer matrices Mf with
  entries in -1..1, or -2..2 for oblique cells, in the primitive basis). The argument is in this
  docstring; formal verification not performed.
* Group names (generic, and on each locus): computational evidence, two routes
  (identify_wallpaper_group of wallpaper_lattices, and spglib through a 3D layer).
* enumerate_families: exhaustive over an explicitly stated finite set of structures (see its
  docstring); classification of each structure at random generic parameter values: computational
  evidence. Planarity is found by sampling parameter values: "no planar member found" is not a proof.
* same_network: isomorphism of finite balls of the two graphs: computational evidence.
"""
from __future__ import annotations

import copy
import itertools
import math
import time
from collections import Counter
from fractions import Fraction

import numpy as np

from . import wallpaper_lattices as WL
from . import minimal_lattices as ML
from .wallpaper_lattices import (UnitCell, WallpaperGroup, WallpaperLattice, WALLPAPER_GROUPS,
                                identify_wallpaper_group, primitive_cell, periodic_connected,
                                make_cell, _symmetry_ops, _canon_bond, _apply_to_site)

try:
    import spglib
except ImportError:                      # the second route is then skipped, and reported as None
    spglib = None

# wallpaper group -> number of the 3D space group  G x {1, m_z}  (same table as check_lattices.py)
LAYER = {"p1": 6, "p2": 10, "pm": 25, "pg": 26, "cm": 38, "pmm": 47, "pmg": 51, "pgg": 55, "cmm": 65,
         "p4": 83, "p4m": 123, "p4g": 127, "p3": 174, "p3m1": 187, "p31m": 189, "p6": 175, "p6m": 191}
_LAYER_INV = {v: k for k, v in LAYER.items()}
_UC = np.array([[0.5, -0.5], [0.5, 0.5]])      # primitive basis of a centred cell: (a+b)/2, (-a+b)/2
I2 = np.eye(2)


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------
def _is_int(v, tol=1e-6):
    v = np.asarray(v, float)
    return bool(np.all(np.abs(v - np.round(v)) < tol))


def _nice(x, max_den=24, tol=1e-9):
    """A float as a short string: simple fractions and square roots are recognised."""
    if abs(x) < tol:
        return "0"
    f = Fraction(x).limit_denominator(max_den)
    if abs(float(f) - x) < tol:
        return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
    f = Fraction(x * x).limit_denominator(max_den)
    if abs(float(f) - x * x) < tol and x > 0:
        return f"sqrt({f.numerator}{'' if f.denominator == 1 else '/' + str(f.denominator)})"
    return f"{x:.6g}"


def _rref(M, tol=1e-9):
    """Reduced row echelon form of an augmented matrix [C | c].
    Returns (rows kept, consistent?)."""
    M = np.array(M, float).reshape(-1, np.shape(M)[-1])
    rows, cols = M.shape
    r = 0
    for c in range(cols - 1):
        if r >= rows:
            break
        p = r + int(np.argmax(np.abs(M[r:, c])))
        if abs(M[p, c]) < tol:
            continue
        M[[r, p]] = M[[p, r]]
        M[r] = M[r] / M[r, c]
        for i in range(rows):
            if i != r:
                M[i] = M[i] - M[i, c] * M[r]
        r += 1
    consistent = not np.any(np.abs(M[r:, -1]) > 1e-7) if r < rows else True
    out = M[:r]
    out[np.abs(out) < 1e-10] = 0.0
    return out, consistent


def spg_number(cell, use_types=False, c=7.3):
    """Second route for the group. The 2D lattice becomes a 3D crystal: sites are atoms of species 1
    at z = 0, each bond is two atoms of species 2 at 30% and 70% of its length, and the layer is
    stacked with a long c axis. spglib returns the space-group number of G x {1, m_z}.
    What the two routes share: the unit cell and the bond list. (Same construction as in
    check_lattices.py.)"""
    if spglib is None or len(cell.basis) == 0:
        return None
    A = cell.A
    Ainv = np.linalg.inv(A)
    pos, num = [], []
    for p in cell.basis:
        f = Ainv @ p
        pos.append([f[0] % 1, f[1] % 1, 0.0]); num.append(1)
    for (i, j, n1, n2, t) in cell.bonds:
        p = cell.basis[i]; q = cell.basis[j] + A @ np.array([n1, n2])
        for s in (0.3, 0.7):
            f = Ainv @ (p + s * (q - p))
            pos.append([f[0] % 1, f[1] % 1, 0.0]); num.append(50 + (t if use_types else 0))
    lat = np.array([[A[0, 0], A[1, 0], 0], [A[0, 1], A[1, 1], 0], [0, 0, c * np.sqrt(abs(np.linalg.det(A)))]])
    d = spglib.get_symmetry_dataset((lat, np.array(pos), num), symprec=1e-5)
    return None if d is None else int(d.number)


def spg_group(cell, use_types=False):
    """Wallpaper group by the spglib route (None if spglib is missing or the number is not a layer)."""
    n = spg_number(cell, use_types)
    return _LAYER_INV.get(n, None if n is None else f"spacegroup {n}")


def supercell(cell, S):
    """The same lattice described in the larger cell (a1, a2) S, S an integer 2x2 matrix."""
    S = np.array(S, int)
    k = int(round(abs(np.linalg.det(S))))
    if k == 0:
        raise ValueError("singular supercell matrix")
    A = cell.A
    B = A @ S
    Binv = np.linalg.inv(B)
    offs = []
    for n in itertools.product(range(-4 * k, 4 * k + 1), repeat=2):
        f = np.linalg.solve(S, np.array(n, float))
        if np.all(f > -1e-9) and np.all(f < 1 - 1e-9):
            offs.append(np.array(n))
    assert len(offs) == k
    basis = np.array([p + A @ o for o in offs for p in cell.basis])
    bonds = set()
    for o in offs:
        for (i, j, n1, n2, t) in cell.bonds:
            p = cell.basis[i] + A @ o
            q = cell.basis[j] + A @ (o + np.array([n1, n2]))
            si, ni = WL._locate(p, B, Binv, basis)
            sj, nj = WL._locate(q, B, Binv, basis)
            bonds.add(_canon_bond(si, sj, nj[0] - ni[0], nj[1] - ni[1], t))
    return UnitCell(cell.key, cell.label, cell.group, B[:, 0], B[:, 1], basis, sorted(bonds))


# --- geometry diagnostics of a cell -------------------------------------------------------------
def _spacing(cell):
    return math.sqrt(abs(np.linalg.det(cell.A)) / max(1, len(cell.basis)))


_SH = np.array(list(itertools.product(range(-3, 4), repeat=2)), float)


def min_separation(cell):
    """Smallest distance between two sites, in units of the average spacing (cell area / sites)^(1/2)."""
    X = np.asarray(cell.basis, float)
    if len(X) == 0:
        return math.inf
    D = X[:, None, None, :] - X[None, :, None, :] - (_SH @ cell.A.T)[None, None, :, :]
    d = np.linalg.norm(D, axis=-1)
    zero = int(np.argmin(np.abs(_SH).sum(axis=1)))
    d[np.arange(len(X)), np.arange(len(X)), zero] = np.inf
    return float(d.min()) / _spacing(cell)


def clearance(cell):
    """Smallest distance from a site to a bond it is not an end of (average spacing = 1).
    Zero means a bond passes through a site."""
    X = np.asarray(cell.basis, float)
    A = cell.A
    best = math.inf
    for (i, j, n1, n2, _) in cell.bonds:
        r1 = range(min(0, n1) - 2, max(0, n1) + 3); r2 = range(min(0, n2) - 2, max(0, n2) + 3)
        sh = np.array(list(itertools.product(r1, r2)), float)
        S = X[:, None, :] + (sh @ A.T)[None, :, :]
        p = X[i]; q = X[j] + A @ np.array([n1, n2], float)
        v = q - p; L2 = v @ v
        s = np.clip(((S - p) @ v) / L2, 0.0, 1.0)
        d = np.linalg.norm(p + s[..., None] * v - S, axis=-1)
        d[i, np.all(sh == 0, axis=1)] = np.inf
        d[j, np.all(sh == np.array([n1, n2]), axis=1)] = np.inf
        best = min(best, float(d.min()))
    return best / _spacing(cell)


def has_crossings(cell):
    """Do two bonds cross? (vectorised version of wallpaper_lattices.has_crossings)"""
    b = [x[:4] for x in cell.bonds]
    return bool(b) and ML._orbits_cross(np.asarray(cell.basis, float), cell.A, b, b)


def degrees_of(nsites, bonds):
    d = [0] * nsites
    for b in bonds:
        d[b[0]] += 1; d[b[1]] += 1
    return d


# --- networks -----------------------------------------------------------------------------------
def _ball(nsites, bonds, root, depth):
    import networkx as nx
    adj = [[] for _ in range(nsites)]
    for (i, j, n1, n2) in bonds:
        adj[i].append((j, n1, n2)); adj[j].append((i, -n1, -n2))
    dist = {(root, 0, 0): 0}
    frontier = [(root, 0, 0)]
    for d in range(1, depth + 1):
        nxt = []
        for (u, x, y) in frontier:
            for (v, a, b) in adj[u]:
                k = (v, x + a, y + b)
                if k not in dist:
                    dist[k] = d; nxt.append(k)
        frontier = nxt
    G = nx.Graph()
    for k, d in dist.items():
        G.add_node(k, d=d)
    for (u, x, y) in dist:
        for (v, a, b) in adj[u]:
            k = (v, x + a, y + b)
            if k in dist:
                G.add_edge((u, x, y), k)
    return G


def _bare_bonds(cell):
    return sorted({_canon_bond(i, j, n1, n2, 0)[:4] for (i, j, n1, n2, _) in cell.bonds})


def _nkey(nsites, bonds, depth):
    c = Counter(ML.coordination_sequences(nsites, bonds, depth))
    g = 0
    for v in c.values():
        g = math.gcd(g, v)
    return tuple(sorted((k, v // g) for k, v in c.items()))


def _ball_hash(nsites, bonds, roots, depth=4):
    """Weisfeiler-Lehman hashes of the balls of graph radius `depth` around the given sites."""
    import networkx as nx
    return tuple(sorted(nx.weisfeiler_lehman_graph_hash(_ball(nsites, bonds, r, depth), node_attr="d", iterations=depth)
                        for r in roots))


def network_key(cell, depth=8):
    """Coordination sequences of the sites, as proportions: equal for isomorphic networks
    (a necessary condition, cheap)."""
    cs = ML.coordination_sequences(len(cell.basis), _bare_bonds(cell), depth)
    c = Counter(cs)
    g = 0
    for v in c.values():
        g = math.gcd(g, v)
    return tuple(sorted((k, v // g) for k, v in c.items()))


def same_network(cell_a, cell_b, depth=5):
    """Are the two bond networks the same graph? Test: equal coordination sequences, and the ball of
    graph radius `depth` around a site of A is isomorphic to the ball around some site of B, with
    the distance from the centre preserved. Coordinates, group and cell play no role.
    Label: computational evidence (balls of finite radius)."""
    import networkx as nx
    from networkx.algorithms.isomorphism import categorical_node_match
    if network_key(cell_a) != network_key(cell_b):
        return False
    ba, bb = _bare_bonds(cell_a), _bare_bonds(cell_b)
    csa = ML.coordination_sequences(len(cell_a.basis), ba, 8)
    csb = ML.coordination_sequences(len(cell_b.basis), bb, 8)
    Ga = _ball(len(cell_a.basis), ba, 0, depth)
    nm = categorical_node_match("d", None)
    seen = set()
    for t in range(len(cell_b.basis)):
        if csb[t] != csa[0]:
            continue
        Gb = _ball(len(cell_b.basis), bb, t, depth)
        key = (Gb.number_of_nodes(), Gb.number_of_edges())
        if key != (Ga.number_of_nodes(), Ga.number_of_edges()):
            continue
        if nx.is_isomorphic(Ga, Gb, node_match=nm):
            return True
        seen.add(t)
    return False


# ----------------------------------------------------------------------------------------------
# Group data in the primitive basis
# ----------------------------------------------------------------------------------------------
def _prim_U(G):
    return _UC if G.centred else I2


def _prim_ops(G):
    """One operation per element of G / (primitive translations), in the primitive basis."""
    U = _prim_U(G); Ui = np.linalg.inv(U)
    reps = G.ops[:len(G.ops) // 2] if G.centred else G.ops
    return [(np.round(Ui @ M @ U).astype(int), Ui @ t) for M, t in reps]


def _site_freedom(G, A, p):
    """Wyckoff type of the point p (conventional fractional coordinates): returns
    (kind, base, D, u, site symmetry) with p = base + D u and D spanning the directions in which p
    can move without changing its site symmetry."""
    p = np.asarray(p, float)
    Ainv = np.linalg.inv(A)
    rows, cart = [], []
    for M, t in G.ops:
        if _is_int(M @ p + t - p, 1e-7):
            rows.append(M - I2)
            cart.append(A @ M @ Ainv)
    K = np.vstack(rows)
    _, s, Vt = np.linalg.svd(K)
    rank = int(np.sum(s > 1e-9))
    sym = ML._site_symmetry_label(cart)
    if rank == 0:
        return "general", np.zeros(2), np.eye(2), p.copy(), sym
    if rank == 2:
        return "point", p.copy(), np.zeros((2, 0)), np.zeros(0), sym
    d = Vt[-1]
    nz = np.abs(d)[np.abs(d) > 1e-9]
    d = d / nz.min()
    d = np.round(d, 9)
    if d[np.abs(d) > 1e-9][0] < 0:
        d = -d
    t = float(p @ d) / float(d @ d)
    return "line", p - t * d, d.reshape(2, 1), np.array([t]), sym


# ----------------------------------------------------------------------------------------------
# Special loci
# ----------------------------------------------------------------------------------------------
class Locus:
    """An affine subspace of parameter space, C v = c, on which the lattice has extra symmetry.
    v = (metric variables, site parameters); see Family.loci."""

    def __init__(self, family, aug):
        self.family = family
        self.aug = aug                       # reduced row echelon form of [C | c]
        self.codim = len(aug)
        self.dim = family.n_free - self.codim
        self.group = None                    # group at a generic point of the locus
        self.spglib = None
        self.sites = None                    # sites per primitive cell there
        self.kind = None                     # "supergroup" | "collision" | "generic"
        self.example = None                  # parameter values of a generic point of the locus
        self.n_automorphisms = 0

    @property
    def key(self):
        return tuple(np.round(self.aug, 7).ravel().tolist()) + (self.aug.shape[0],)

    def residual(self, values=None):
        v = self.family._v(values)
        if self.codim == 0:
            return 0.0
        return float(np.max(np.abs(self.aug[:, :-1] @ v - self.aug[:, -1])))

    def project(self, values=None):
        """The point of the locus nearest to the given parameter values (as a dict of values)."""
        f = self.family
        v = f._v(values)
        if self.codim:
            C, c = self.aug[:, :-1], self.aug[:, -1]
            v = v - np.linalg.pinv(C) @ (C @ v - c)
        return f._values_from_v(v)

    def contains(self, values=None, tol=1e-7):
        return self.residual(values) < tol

    def equations(self):
        """The locus as equations between the family's parameters."""
        f = self.family
        names = f._v_names()
        out = []
        for row in self.aug:
            C, c = row[:-1], row[-1]
            idx = [k for k in range(len(C)) if abs(C[k]) > 1e-9]
            p = idx[0]
            terms = []
            for k in idx[1:]:
                co = -C[k]
                mag = "" if abs(abs(co) - 1) < 1e-9 else _nice(abs(co)) + " "
                terms.append(("+ " if co > 0 else "- ") + mag + names[k])
            if abs(c) > 1e-9 or not terms:
                terms.append(("+ " if c > 0 else "- ") + _nice(abs(c)) if terms else _nice(c))
            rhs = " ".join(terms)
            if rhs.startswith("+ "):
                rhs = rhs[2:]
            eq = f"{names[p]} = {rhs}"
            # friendlier forms for the cell
            if names[p] == "b^2" and len(idx) == 1:
                eq = f"b = {_nice(math.sqrt(c) * f.a) if c > 0 else '?'}" + (" (b = a)" if abs(c - 1) < 1e-9 else "")
            if names[p] == "b cos(gamma)" and len(idx) == 1 and abs(c) < 1e-9:
                eq = "gamma = 90"
            out.append(eq)
        return out

    def __repr__(self):
        eq = ", ".join(self.equations()) or "(all values)"
        g = self.group or "?"
        extra = "" if self.sites in (None, self.family.m) else f", {self.sites} site(s) per primitive cell"
        return f"Locus[{eq}] -> {g}{extra} ({self.kind}, {self.dim} free parameter(s) left)"

    def member(self):
        """The lattice at a generic point of the locus, as a Family of its own (larger) group."""
        return Family.from_cell(self.family.cell(**self.example))


# ----------------------------------------------------------------------------------------------
# Family
# ----------------------------------------------------------------------------------------------
class Family:
    """A lattice of a wallpaper group together with its free parameters (see the module docstring).

    Family(group, sites, bonds, a=1, b=None, gamma=None)
        group : wallpaper group symbol (ITA standard setting, as in wallpaper_lattices)
        sites : one point (x, y) per site orbit, fractional in the conventional cell
        bonds : one bond (p, q) or (p, q, type) per bond orbit; p and q are fractional positions of
                two sites (q may lie in another cell)
    The same arguments as wallpaper_lattices.symmetric_cell, and as the recipes of minimal_lattices.
    """

    def __init__(self, group, sites=(), bonds=(), a=1.0, b=None, gamma=None, frame=None, label=None,
                 skip_duplicates=False):
        G = group if isinstance(group, WallpaperGroup) else WallpaperGroup(group)
        self.G = G
        self.group = G.symbol
        self.a = float(a)
        a1, a2 = G.lattice(self.a, b, gamma)
        self.cellpar = {}
        if G.system in ("oblique", "rectangular"):
            self.cellpar["b"] = float(np.linalg.norm(a2))
        if G.system == "oblique":
            self.cellpar["gamma"] = float(np.degrees(math.atan2(a2[1], a2[0])))
        self.frame = (np.eye(2), np.zeros(2)) if frame is None else (np.array(frame[0], float), np.array(frame[1], float))
        self.label = label or f"{G.symbol} family"
        self._cache = {}
        self._U = _prim_U(G)
        self._Ui = np.linalg.inv(self._U)
        self._ops = _prim_ops(G)
        Ac = np.column_stack([a1, a2])
        # --- site orbits
        self.gens = []
        self.site_orbit, self._site_op, self._site_fl = [], [], []
        for p in sites:
            p = np.asarray(p, float)
            if self._find_site_at(p) is not None:
                if skip_duplicates:
                    continue
                raise ValueError(f"the point ({p[0]:.6g}, {p[1]:.6g}) lies on the orbit of a site that is already there")
            kind, base, D, u, sym = _site_freedom(G, Ac, p)
            o = len(self.gens)
            self.gens.append(dict(kind=kind, base=base, D=D, u=u, sym=sym))
            seen = []
            pp = self._Ui @ p
            for g, (M, t) in enumerate(self._ops):
                q = M @ pp + t
                if any(_is_int(q - r, 1e-7) for r in seen):
                    continue
                seen.append(q)
                self.site_orbit.append(o); self._site_op.append(g); self._site_fl.append(np.floor(q + 1e-9))
            self.gens[-1]["mult"] = len(seen)
        self._name_parameters()
        # --- bond orbits
        self.orbits, self.types = [], []
        for k, bd in enumerate(bonds):
            self._add_bond_orbit(bd[0], bd[1], bd[2] if len(bd) > 2 else None, quiet=skip_duplicates)
        self._cache = {}

    # --- construction helpers ---------------------------------------------------------------
    def _name_parameters(self):
        self.site_names = []
        self._slices = []
        k = 0
        for o, g in enumerate(self.gens):
            n = g["D"].shape[1]
            names = {0: [], 1: [f"t{o + 1}"], 2: [f"x{o + 1}", f"y{o + 1}"]}[n]
            g["names"] = names
            self.site_names += names
            self._slices.append(slice(k, k + n)); k += n
        self.cell_names = list(self.cellpar)
        self.names = self.cell_names + self.site_names

    def _find_site_at(self, p, tol=1e-6):
        """Label (site, cell shift) of the site at conventional fractional point p, or None."""
        if not self.gens:
            return None
        F = self._F()
        pp = self._Ui @ np.asarray(p, float)
        for k in range(len(F)):
            d = pp - F[k]
            if _is_int(d, tol):
                return k, np.round(d).astype(int)
        return None

    def _add_bond_orbit(self, p, q, ty=None, quiet=False):
        a = self._find_site_at(p); b = self._find_site_at(q)
        if a is None or b is None:
            raise ValueError(f"bond end {tuple(np.round(p if a is None else q, 6))} is not a site")
        (i, ni), (j, nj) = a, b
        n = nj - ni
        if i == j and not np.any(n):
            raise ValueError("zero-length bond")
        gen = _canon_bond(i, j, int(n[0]), int(n[1]), 0)[:4]
        if any(gen in orb for orb in self.orbits):
            if quiet:
                return None
            raise ValueError("this bond is already in the lattice")
        self.orbits.append(self._orbit_of(gen))
        self.types.append(len(self.orbits) - 1 if ty is None else ty)
        return len(self.orbits) - 1

    def _action(self):
        """perm[g][k], shift[g][k]: operation g sends site k to site perm (cell shift)."""
        if "action" not in self._cache:
            F = self._F()
            perm, shift = [], []
            for (M, t) in self._ops:
                pk, sk = [], []
                for k in range(len(F)):
                    j, c = _apply_to_site(M, t, F, k)
                    pk.append(j); sk.append(c)
                perm.append(pk); shift.append(sk)
            self._cache["action"] = (perm, shift)
        return self._cache["action"]

    def _orbit_of(self, bond):
        self._cache = getattr(self, "_cache", {})
        perm, shift = self._action()
        i, j, n1, n2 = bond
        out = set()
        for g, (M, t) in enumerate(self._ops):
            dn = M @ np.array([n1, n2]) + shift[g][j] - shift[g][i]
            out.add(_canon_bond(perm[g][i], perm[g][j], int(dn[0]), int(dn[1]), 0)[:4])
        return sorted(out)

    def _with_orbits(self, orbits, types=None):
        f = self.copy()
        f.orbits = [sorted(o) for o in orbits]
        f.types = list(range(len(orbits))) if types is None else list(types)
        f._cache = {k: v for k, v in self._cache.items() if k == "action"}
        return f

    def copy(self):
        f = copy.copy(self)
        f.gens = [dict(g, u=g["u"].copy()) for g in self.gens]
        f.cellpar = dict(self.cellpar)
        f.orbits = [list(o) for o in self.orbits]
        f.types = list(self.types)
        f._cache = {k: v for k, v in self._cache.items() if k in ("action", "auts")}
        return f

    @classmethod
    def from_recipe(cls, recipe, **kw):
        """From a recipe dict(group, sites, bonds, a, b, gamma), e.g. entry.recipe of minimal_lattices."""
        r = getattr(recipe, "recipe", recipe)
        lat = {k: r[k] for k in ("a", "b", "gamma") if k in r and r[k] is not None}
        return cls(r["group"], [tuple(p) for p in r["sites"]], [tuple(map(tuple, b[:2])) + tuple(b[2:]) for b in r["bonds"]],
                   **lat, **kw)

    @classmethod
    def from_cell(cls, cell, group=None, supercell_matrix=None, use_types=False, label=None):
        """The family of an existing lattice (catalogue name, UnitCell, or anything make_cell takes).

        group=None : the lattice's own group (bond colours ignored unless use_types=True).
        group="p4" : treat the lattice as a member of a family of a SUBGROUP with the same cell
                     (then it sits on a special locus of that family).
        supercell_matrix : integer 2x2 matrix S; the subgroup's cell is (a1, a2) S.
        The lattice is brought to the ITA standard setting of the group (found by trying the short
        cell vectors and the candidate origins; see _standardise). The Cartesian frame of the input
        is kept, so fam.cell() coincides with the input lattice."""
        cell = make_cell(cell)
        if group is None:
            pc = primitive_cell(cell, use_types)
            if abs(abs(np.linalg.det(pc.A)) - abs(np.linalg.det(cell.A))) < 1e-9 and len(pc.basis) == len(cell.basis):
                pc = cell                         # already primitive: keep the cell vectors as given
            group = identify_wallpaper_group(pc, use_types=use_types)
        else:
            pc = supercell(cell, supercell_matrix) if supercell_matrix is not None else cell
        G = WallpaperGroup(group)
        C, w = _standardise(pc, G, use_types)
        a = float(np.linalg.norm(C[:, 0])); b = float(np.linalg.norm(C[:, 1]))
        gamma = float(np.degrees(math.acos(np.clip(C[:, 0] @ C[:, 1] / (a * b), -1, 1))))
        Astd = np.column_stack(G.lattice(a, b, gamma))
        Q = C @ np.linalg.inv(Astd)
        if not np.allclose(Q.T @ Q, I2, atol=1e-6):
            raise RuntimeError("standardisation failed: cell shape does not fit the crystal system")
        Ci = np.linalg.inv(C)
        y = [Ci @ p - w for p in pc.basis]
        A = pc.A
        bonds = []
        for (i, j, n1, n2, t) in pc.bonds:
            bonds.append((y[i], Ci @ (pc.basis[j] + A @ np.array([n1, n2])) - w, t))
        kw = dict(a=a)
        if G.system in ("oblique", "rectangular"):
            kw["b"] = b
        if G.system == "oblique":
            kw["gamma"] = gamma
        fam = cls(G, y, [], frame=(Q, C @ w), label=label or cell.label, skip_duplicates=True, **kw)
        # all bonds are passed; each new orbit keeps the colour of its first bond
        for (p, q, t) in bonds:
            fam._add_bond_orbit(p, q, t, quiet=True)
        if not use_types:
            fam.types = list(range(len(fam.orbits)))
        return fam

    # --- parameters ---------------------------------------------------------------------------
    @property
    def m(self):
        """Sites per primitive cell."""
        return len(self.site_orbit)

    @property
    def n_free(self):
        return len(self.names)

    @property
    def values(self):
        v = dict(self.cellpar)
        for g in self.gens:
            v.update(zip(g["names"], (float(x) for x in g["u"])))
        return v

    def set(self, **values):
        """A copy with some parameters changed (the network and the labels are unchanged)."""
        f = self.copy()
        for k, x in values.items():
            if k == "a":
                f.a = float(x)
            elif k in f.cellpar:
                f.cellpar[k] = float(x)
            elif k in f.site_names:
                for g in f.gens:
                    if k in g["names"]:
                        g["u"][g["names"].index(k)] = float(x)
            else:
                raise KeyError(f"{k!r} is not a free parameter of this family; free: {self.names}")
        return f

    def parameters(self, show=True):
        """The free parameters: name, value, meaning."""
        rows = []
        if "b" in self.cellpar:
            rows.append(("b", self.cellpar["b"], f"cell: length of the second cell vector (a = {self.a:g} sets the scale)"))
        if "gamma" in self.cellpar:
            rows.append(("gamma", self.cellpar["gamma"], "cell: angle between the cell vectors, in degrees"))
        for o, g in enumerate(self.gens):
            if g["kind"] == "general":
                rows.append((g["names"][0], float(g["u"][0]), f"site orbit {o + 1}: x of its generating point (fraction of a)"))
                rows.append((g["names"][1], float(g["u"][1]), f"site orbit {o + 1}: y of its generating point (fraction of b)"))
            elif g["kind"] == "line":
                d = g["D"][:, 0]; bs = g["base"]
                rows.append((g["names"][0], float(g["u"][0]),
                             f"site orbit {o + 1}: position along its mirror line, point = "
                             f"({_nice(bs[0])}, {_nice(bs[1])}) + t ({_nice(d[0])}, {_nice(d[1])})"))
        if show:
            if not rows:
                print("no free parameter (apart from the overall scale a)")
            for n, v, what in rows:
                print(f"  {n:6s} = {v:<10.6g} {what}")
        return rows

    def gen_point(self, o):
        """Generating point of site orbit o (0-based), conventional fractional coordinates."""
        g = self.gens[o]
        return g["base"] + g["D"] @ g["u"]

    # --- geometry -----------------------------------------------------------------------------
    def _Ac(self):
        b = self.cellpar.get("b"); gamma = self.cellpar.get("gamma")
        return np.column_stack(self.G.lattice(self.a, b, gamma))

    def _F(self):
        """Primitive fractional coordinates of the labelled sites."""
        F = np.zeros((len(self.site_orbit), 2))
        pts = [self._Ui @ self.gen_point(o) for o in range(len(self.gens))]
        for k, (o, g) in enumerate(zip(self.site_orbit, self._site_op)):
            M, t = self._ops[g]
            F[k] = M @ pts[o] + t - self._site_fl[k]
        return F

    def cell(self, **values):
        """The UnitCell (primitive cell of the group) at the current or the given parameter values.
        Bond type = bond orbit, so a plot colours each orbit differently."""
        f = self.set(**values) if values else self
        B = f._Ac() @ f._U
        Q, o0 = f.frame
        F = f._F()
        basis = (Q @ (B @ F.T)).T + o0 if len(F) else np.zeros((0, 2))
        bonds = sorted(_canon_bond(i, j, n1, n2, t) for t, orb in zip(f.types, f.orbits) for (i, j, n1, n2) in orb)
        c = UnitCell(f"fam_{f.group}", f.label, f.group, Q @ B[:, 0], Q @ B[:, 1], basis, bonds,
                     note=f"family of {f.group}: {len(f.gens)} site orbit(s), {len(f.orbits)} bond orbit(s)")
        return c

    def bonds(self):
        return sorted(b for o in self.orbits for b in o)

    def bond_lengths(self, **values):
        """Length of each bond orbit."""
        c = self.cell(**values)
        A = c.A
        return [float(np.linalg.norm(c.basis[o[0][1]] + A @ np.array(o[0][2:4]) - c.basis[o[0][0]])) for o in self.orbits]

    def recipe(self):
        """dict(group, sites, bonds, a, b, gamma): rebuilds the family with Family.from_recipe, and
        the lattice with wallpaper_lattices.symmetric_cell / minimal_lattices.rebuild."""
        F = self._F()
        bonds = []
        for o in self.orbits:
            i, j, n1, n2 = o[0]
            p = self._U @ F[i]; q = self._U @ (F[j] + np.array([n1, n2]))
            sh = np.floor(p + 1e-9)
            bonds.append((tuple(float(x) for x in np.round(p - sh, 12)), tuple(float(x) for x in np.round(q - sh, 12))))
        r = dict(group=self.group, sites=[tuple(float(x) for x in np.round(self.gen_point(o), 12)) for o in range(len(self.gens))],
                 bonds=bonds, a=self.a)
        r.update(self.cellpar)
        return r

    def code(self):
        """One line of Python that rebuilds this family."""
        r = self.recipe()
        pt = lambda p: "(" + ", ".join(ML._num(x) for x in p) + ")"
        s = ", ".join(pt(p) for p in r["sites"])
        b = ", ".join(f"({pt(p)}, {pt(q)})" for p, q in r["bonds"])
        lat = "".join(f", {k}={ML._num(r[k])}" for k in ("a", "b", "gamma") if k in r and not (k == "a" and r[k] == 1))
        return f'Family("{r["group"]}", sites=[{s}], bonds=[{b}]{lat})'

    # --- what the lattice is at given parameter values -----------------------------------------
    def group_at(self, **values):
        """(group, sites per primitive cell) of the bare lattice at the given values, first route."""
        c = self.cell(**values)
        if len(c.basis) == 0:
            return None, 0
        return identify_wallpaper_group(c, use_types=False), len(primitive_cell(c, use_types=False).basis)

    def info(self):
        """Parameter-independent facts: sites, bonds, coordination numbers, Maxwell count."""
        b = self.bonds()
        deg = degrees_of(self.m, b)
        return dict(group=self.group, sites=self.m, site_orbits=len(self.gens), bond_orbits=len(self.orbits),
                    bonds=len(b), degrees=sorted(set(deg)), maxwell=2 * self.m - len(b),
                    connected=bool(self.m) and periodic_connected(self.m, b), free_parameters=self.n_free,
                    wyckoff=[f"{g['mult']} {g['sym']} ({g['kind']})" for g in self.gens])

    def status(self, **values):
        """Everything about the lattice at the given parameter values: the group by two routes and
        whether they agree, whether it is the family's group, connectivity, planarity, validity."""
        c = self.cell(**values)
        s = self.info()
        if self.m == 0:
            s.update(found=None, spglib=None, routes_agree=True, relation="empty", planar=True,
                     min_separation=math.inf, clearance=math.inf, valid=True)
            return s
        sep = min_separation(c)
        if sep < 1e-6:                         # two sites coincide: not a lattice of this network
            s.update(found=None, primitive_sites=None, spglib=None, routes_agree=True, relation="invalid",
                     planar=False, min_separation=sep, clearance=0.0, valid=False)
            return s
        g, mp = self.group_at(**values)
        sg = spg_group(c)
        cl = clearance(c)
        if g == self.group and mp == self.m:
            rel = "same"
        elif g == self.group:
            rel = "supergroup (same symbol, more translations)"
        else:
            rel = "supergroup"
        s.update(found=g, primitive_sites=mp, spglib=sg, routes_agree=(sg is None or sg == g), relation=rel,
                 planar=not has_crossings(c), min_separation=sep, clearance=cl,
                 valid=sep > 1e-6 and cl > 1e-6)
        return s

    def is_exact(self, n=2, seed=0):
        """Is the generic group exactly G, with G's translations? Tested at n random parameter points."""
        rng = np.random.default_rng(seed)
        for _ in range(n):
            g, mp = self.group_at(**self.random_values(rng))
            if g != self.group or mp != self.m:
                return False
        return True

    def random_values(self, rng):
        v = {}
        if "b" in self.cellpar:
            r = float(rng.uniform(0.62, 1.7))
            for bad in (1.0, math.sqrt(3), 1 / math.sqrt(3)):
                if abs(r - bad) < 0.07:
                    r += 0.16
            v["b"] = r * self.a
        if "gamma" in self.cellpar:
            v["gamma"] = float(rng.uniform(58, 86))
        # a site parameter and the same parameter + 1 put the site at equivalent points, but the bonds
        # (fixed labels) are then drawn differently: sample two periods
        for g in self.gens:
            for n in g["names"]:
                v[n] = float(rng.uniform(-0.97, 0.97))
        return v

    def fingerprint(self):
        """Orbit-structure fingerprint of minimal_lattices (the 'same entry' test used there)."""
        return ML.orbit_fingerprint(self.cell())[0]

    # --- describe / plot ------------------------------------------------------------------------
    def __repr__(self):
        i = self.info()
        return (f"Family({self.group}: {i['sites']} sites/cell in {i['site_orbits']} orbit(s), "
                f"{i['bond_orbits']} bond orbit(s), {i['bonds']} bonds/cell, {self.n_free} free parameter(s))")

    def describe(self, **values):
        f = self.set(**values) if values else self
        i = f.info(); s = f.status()
        print(f"{f.label}: family of {f.group}")
        print(f"  sites per primitive cell {i['sites']}, bonds per primitive cell {i['bonds']}, "
              f"coordination numbers {i['degrees']}, Maxwell count {i['maxwell']}")
        for o, g in enumerate(f.gens):
            p = f.gen_point(o)
            print(f"  site orbit {o + 1}: {g['mult']} site(s), site symmetry {g['sym']}, {g['kind']} position, "
                  f"generating point ({p[0]:.4g}, {p[1]:.4g})")
        L = f.bond_lengths()
        for k, o in enumerate(f.orbits):
            so = sorted({f.site_orbit[o[0][0]] + 1, f.site_orbit[o[0][1]] + 1})
            print(f"  bond orbit {k + 1}: {len(o)} bond(s) per cell, length {L[k]:.4g}, joins site orbit(s) {so}")
        print(f"  free parameters ({f.n_free}):")
        f.parameters()
        if f.m:
            print(f"  now: group {s['found']} (first route), {s['spglib']} (spglib); {s['relation']}; "
                  f"{'connected' if s['connected'] else 'NOT connected'}; {'planar' if s['planar'] else 'bonds cross'}"
                  + ("" if s["valid"] else "; INVALID (two sites coincide or a bond passes through a site)"))

    def plot(self, n=2, ax=None, symmetry=False, show_sites=True, title=True, **values):
        f = self.set(**values) if values else self
        lat = WallpaperLattice(f.cell(), n=n)
        ax = lat.plot(ax=ax, show_sites=show_sites, site_size=7, symmetry=symmetry, title=False)
        if title:
            s = ", ".join(f"{k}={v:.3g}" for k, v in f.values.items())
            ax.set_title(f"{f.label} ({f.group})\n{s}", fontsize=8)
        return ax

    # --- special loci ---------------------------------------------------------------------------
    def _metric_vars(self):
        return {"oblique": ["b cos(gamma)", "b^2"], "rectangular": ["b^2"]}.get(self.G.system, [])

    def _v_names(self):
        return self._metric_vars() + self.site_names

    def _v(self, values=None):
        """Parameter values as the vector in which the loci are affine: (metric variables in units
        of a, site parameters)."""
        f = self.set(**values) if values else self
        y = []
        if "b" in f.cellpar:
            r = f.cellpar["b"] / f.a
            if "gamma" in f.cellpar:
                y.append(r * math.cos(math.radians(f.cellpar["gamma"])))
            y.append(r * r)
        u = np.concatenate([g["u"] for g in f.gens]) if f.gens else np.zeros(0)
        return np.concatenate([np.array(y), u])

    def _values_from_v(self, v):
        v = np.asarray(v, float)
        out = {}
        k = 0
        if self.G.system == "oblique":
            y1, y2 = v[0], v[1]; k = 2
            b = math.sqrt(max(y2, 1e-300))
            out["b"] = b * self.a
            out["gamma"] = math.degrees(math.acos(np.clip(y1 / b, -1, 1)))
        elif self.G.system == "rectangular":
            out["b"] = math.sqrt(max(v[0], 1e-300)) * self.a; k = 1
        out.update(zip(self.site_names, (float(x) for x in v[k:])))
        return out

    def _affine_sites(self):
        """F_k = P_k u + q_k (primitive fractional coordinates as affine functions of the site parameters)."""
        d = len(self.site_names)
        P = np.zeros((self.m, 2, d)); q = np.zeros((self.m, 2))
        for k, (o, g) in enumerate(zip(self.site_orbit, self._site_op)):
            M, t = self._ops[g]
            gen = self.gens[o]
            P[k][:, self._slices[o]] = M @ self._Ui @ gen["D"]
            q[k] = M @ self._Ui @ gen["base"] + t - self._site_fl[k]
        return P, q

    def _metric_system(self, Mf):
        """Rows [C | c] on the metric variables expressing 'Mf is an isometry of the cell', or None
        if it never is."""
        U = self._U
        sysm = self.G.system
        if sysm == "oblique":
            G0 = np.array([[1.0, 0], [0, 0]]); Gs = [np.array([[0, 1.0], [1, 0]]), np.array([[0, 0], [0, 1.0]])]
        elif sysm == "rectangular":
            G0 = np.array([[1.0, 0], [0, 0]]); Gs = [np.array([[0, 0], [0, 1.0]])]
        elif sysm == "square":
            G0 = np.eye(2); Gs = []
        else:
            G0 = np.array([[1.0, -0.5], [-0.5, 1.0]]); Gs = []
        res = lambda Gm: (Mf.T @ (U.T @ Gm @ U) @ Mf - U.T @ Gm @ U)[[0, 0, 1], [0, 1, 1]]
        r0 = res(G0)
        if not Gs:
            return np.zeros((0, 1)) if np.max(np.abs(r0)) < 1e-9 else None
        rows = np.column_stack([res(Gm) for Gm in Gs] + [-r0])
        aug, ok = _rref(rows)
        return aug if ok else None

    def automorphisms(self):
        """All automorphisms (Mf, pi, s) of the labelled periodic graph, modulo lattice translations:
        (site k, cell n) -> (site pi[k], cell Mf n + s[k]). Only Mf that are isometries of the cell
        for some value of the cell parameters are tried. Needs a connected lattice."""
        if "auts" in self._cache:
            return self._cache["auts"]
        m = self.m
        bonds = self.bonds()
        if m == 0 or not periodic_connected(m, bonds):
            raise ValueError("automorphisms need a connected lattice")
        adj = [[] for _ in range(m)]
        for (i, j, n1, n2) in bonds:
            adj[i].append((j, (n1, n2))); adj[j].append((i, (-n1, -n2)))
        adjset = [set(a) for a in adj]
        deg = [len(a) for a in adj]
        E = 2 if self.G.system == "oblique" else 1
        out = []
        for ent in itertools.product(range(-E, E + 1), repeat=4):
            Mf = np.array(ent).reshape(2, 2)
            det = int(round(np.linalg.det(Mf)))
            if abs(det) != 1:
                continue
            P = np.eye(2, dtype=int); finite = False
            for _ in range(6):
                P = P @ Mf
                if np.array_equal(P, np.eye(2, dtype=int)):
                    finite = True; break
            if not finite:
                continue
            metric = self._metric_system(Mf)
            if metric is None:
                continue
            a, b, c, d = (int(x) for x in ent)
            ap = lambda n: (a * n[0] + b * n[1], c * n[0] + d * n[1])
            for j0 in range(m):
                if deg[j0] != deg[0]:
                    continue
                pi = {0: j0}; s = {0: (0, 0)}; used = {j0}

                def consistent(v):
                    w = pi[v]; sv = s[v]
                    for (x, n) in adj[v]:
                        if x in pi:
                            mn = ap(n); sx = s[x]
                            if (pi[x], (mn[0] + sx[0] - sv[0], mn[1] + sx[1] - sv[1])) not in adjset[w]:
                                return False
                    return True

                def extend():
                    pick = None
                    for u in pi:
                        for (v, n) in adj[u]:
                            if v not in pi:
                                pick = (u, v, n); break
                        if pick:
                            break
                    if pick is None:
                        out.append((Mf.copy(), [pi[k] for k in range(m)], np.array([s[k] for k in range(m)]), metric))
                        return
                    u, v, n = pick
                    mn = ap(n); su = s[u]
                    for (w, n2) in adj[pi[u]]:
                        if w in used or deg[w] != deg[v]:
                            continue
                        pi[v] = w; s[v] = (n2[0] - mn[0] + su[0], n2[1] - mn[1] + su[1]); used.add(w)
                        if consistent(v):
                            extend()
                        del pi[v]; del s[v]; used.discard(w)

                if consistent(0):
                    extend()
        self._cache["auts"] = out
        return out

    def _locus_of(self, aut, P, q):
        """Reduced equations of the locus of one automorphism, or None if it is empty."""
        Mf, pi, s, metric = aut
        ny = len(self._metric_vars()); d = len(self.site_names)
        rows = []
        if len(metric):
            rows += [np.concatenate([r[:-1], np.zeros(d), r[-1:]]) for r in metric]
        # Mf F_k + t = F_pi(k) + s_k for all k; subtract the k = 0 equation to eliminate t
        E = [Mf @ P[k] - P[pi[k]] for k in range(self.m)]
        r = [q[pi[k]] + s[k] - Mf @ q[k] for k in range(self.m)]
        for k in range(1, self.m):
            for c in range(2):
                rows.append(np.concatenate([np.zeros(ny), E[k][c] - E[0][c], [r[k][c] - r[0][c]]]))
        if not rows:
            return np.zeros((0, ny + d + 1))
        aug, ok = _rref(np.array(rows))
        return aug if ok else None

    def loci(self, max_loci=60, seed=0, classify=True):
        """The special loci of the family (see the module docstring), each with the group found at a
        generic point of it. Returns the list sorted by number of free parameters left.
        loci()[k].equations() are its equations; .group its group; .example a member."""
        key = ("loci", max_loci, classify)
        if key in self._cache:
            return self._cache[key]
        P, q = self._affine_sites()
        found = {}
        for aut in self.automorphisms():
            aug = self._locus_of(aut, P, q)
            if aug is None:
                continue
            L = Locus(self, aug)
            L = found.setdefault(L.key, L)
            L.n_automorphisms += 1
        self.generic_automorphisms = sum(L.n_automorphisms for L in found.values() if L.codim == 0)
        loci = [L for L in found.values() if L.codim > 0]
        # intersections of loci are loci too (with an even larger group)
        frontier = list(loci)
        while frontier and len(found) < max_loci:
            new = []
            for La in frontier:
                for Lb in loci:
                    if La is Lb:
                        continue
                    aug, ok = _rref(np.vstack([La.aug, Lb.aug]))
                    if not ok or len(aug) <= max(La.codim, Lb.codim):
                        continue
                    L = Locus(self, aug)
                    if L.key not in found and len(found) < max_loci:
                        found[L.key] = L; new.append(L)
            loci += new
            frontier = new
        self.loci_truncated = len(found) >= max_loci
        if classify:
            rng = np.random.default_rng(seed)
            for L in loci:
                self._classify(L, rng)
        loci.sort(key=lambda L: (-L.dim, L.key))
        self._cache[key] = loci
        return loci

    def _classify(self, L, rng):
        L.kind = "collision"
        for trial in range(9):
            # first the point of the locus nearest to the current lattice, then random points
            vals = L.project(None if trial == 0 else self.random_values(rng))
            if "b" in vals and not (0.2 * self.a < vals["b"] < 5 * self.a):
                continue
            if "gamma" in vals and not (20 < vals["gamma"] < 160):
                continue
            c = self.cell(**vals)
            if min_separation(c) < (0.05 if trial == 0 else 1e-3):
                continue
            g, mp = self.group_at(**vals)
            L.group, L.sites, L.example = g, mp, vals
            L.spglib = spg_group(c)
            L.kind = "supergroup" if (g != self.group or mp != self.m) else "generic"
            return

    def supergroup_loci(self):
        return [L for L in self.loci() if L.kind == "supergroup"]

    def special_now(self, tol=1e-7, **values):
        """The loci that contain the current (or given) parameter values."""
        return [L for L in self.loci() if L.contains(values or None, tol)]

    def print_special(self):
        loci = self.loci()
        sup = [L for L in loci if L.kind == "supergroup"]
        print(f"{self.label} ({self.group}), {self.n_free} free parameter(s): {len(sup)} special locus/loci where the group is larger")
        for L in sup:
            eq = ", ".join(L.equations())
            extra = "" if L.sites == self.m else f" with {L.sites} site(s) per primitive cell"
            agree = "" if L.spglib in (None, L.group) else f"  [spglib says {L.spglib}]"
            print(f"  {eq:45s} -> {L.group}{extra}; {L.dim} free parameter(s) left{agree}")
        col = [L for L in loci if L.kind == "collision"]
        if col:
            print(f"  ({len(col)} further locus/loci on which sites would coincide: not lattices of this network)")
        if getattr(self, "loci_truncated", False):
            print("  (list of intersections cut at max_loci)")
        return sup

    # --- rigid units and rotations ----------------------------------------------------------------
    def rotation_centres(self):
        """Rotation centres of G in the conventional cell: list of (point, order)."""
        Ac = self._Ac(); Ai = np.linalg.inv(Ac)
        best = {}
        for M, t in self.G.ops:
            Mc = Ac @ M @ Ai
            if np.linalg.det(Mc) < 0 or np.allclose(Mc, I2):
                continue
            order = int(round(2 * math.pi / abs(math.atan2(Mc[1, 0], Mc[0, 0]))))
            for L in itertools.product(range(-1, 2), repeat=2):
                c = np.linalg.solve(I2 - M, t + np.array(L))
                c = c - np.floor(c + 1e-9)
                k = (round(float(c[0]), 6) % 1.0, round(float(c[1]), 6) % 1.0)
                best[k] = max(best.get(k, 0), order)
        return sorted(((np.array(k), o) for k, o in best.items()), key=lambda x: (-x[1], x[0][0], x[0][1]))

    def _unit(self, o, centre, order):
        """Is the polygon made by rotating the generating point of orbit o about `centre` bonded?"""
        Ac = self._Ac(); Ai = np.linalg.inv(Ac)
        p = self.gen_point(o)
        ang = 2 * math.pi / order
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        q = centre + Ai @ R @ Ac @ (p - centre)
        a, b = self._find_site_at(p), self._find_site_at(q)
        if a is None or b is None:
            return False
        n = b[1] - a[1]
        bond = _canon_bond(a[0], b[0], int(n[0]), int(n[1]), 0)[:4]
        return any(bond in orb for orb in self.orbits)

    def rigid_units(self, show=True):
        """Polygons that can be rotated as a whole: for every site orbit in general position and every
        rotation centre of G, the polygon of the copies of its generating point around the centre.
        `bonded` says whether the sides of the polygon are bonds of the lattice (a rigid unit proper).
        Sorted: bonded first, then nearest centre."""
        Ac = self._Ac()
        out = []
        for o, g in enumerate(self.gens):
            if g["kind"] != "general":
                continue
            p = self.gen_point(o)
            for c0, order in self.rotation_centres():
                for L in itertools.product(range(-1, 2), repeat=2):
                    c = c0 + np.array(L)
                    v = Ac @ (p - c)
                    r = float(np.linalg.norm(v))
                    if r < 1e-9 or r > 0.75 * math.sqrt(abs(np.linalg.det(Ac))):
                        continue
                    out.append(dict(orbit=o + 1, centre=tuple(float(x) for x in c), order=order, radius=r,
                                    angle=float(np.degrees(math.atan2(v[1], v[0]))), bonded=self._unit(o, c, order)))
        out.sort(key=lambda d: (d["orbit"], not d["bonded"], -d["order"], d["radius"]))
        if show:
            for d in out:
                if d["bonded"] or d is next(x for x in out if x["orbit"] == d["orbit"]):
                    print(f"  site orbit {d['orbit']}: {d['order']}-gon about ({_nice(d['centre'][0])}, {_nice(d['centre'][1])}), "
                          f"radius {d['radius']:.4g}, angle {d['angle']:.4g} deg, sides are bonds: {d['bonded']}")
        return out

    def _centre_for(self, orbits, about):
        if about != "auto":
            return np.asarray(about, float)
        units = [d for d in self.rigid_units(show=False) if d["orbit"] == orbits[0]]
        if not units:
            raise ValueError("no rotation centre in this group: give about=(x, y)")
        return np.array(units[0]["centre"])

    def rotation_path(self, orbits=None, about="auto", absolute=True):
        """A function angle (degrees) -> parameter values that rotates the generating points of the
        given site orbits (1-based; default: the first one in general position) rigidly about a
        point. All copies follow by symmetry; copies related by a mirror or glide turn the other way.
        absolute=True: `angle` is the polar angle of the first orbit's generating point about the
        centre, measured from the a axis; False: the angle is added to the current one.
        Returns (path, centre, current angle)."""
        if orbits is None:
            orbits = [next(o + 1 for o, g in enumerate(self.gens) if g["kind"] == "general")]
        orbits = [orbits] if isinstance(orbits, int) else list(orbits)
        for o in orbits:
            if self.gens[o - 1]["kind"] != "general":
                raise ValueError(f"site orbit {o} is on a {self.gens[o - 1]['kind']} position and cannot be rotated")
        c = self._centre_for(orbits, about)
        Ac = self._Ac(); Ai = np.linalg.inv(Ac)
        p0 = [self.gen_point(o - 1) for o in orbits]
        v0 = Ac @ (p0[0] - c)
        ang0 = float(np.degrees(math.atan2(v0[1], v0[0])))

        def path(angle):
            th = math.radians(angle - ang0 if absolute else angle)
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
            vals = {}
            for o, p in zip(orbits, p0):
                pn = c + Ai @ R @ Ac @ (p - c)
                vals[f"x{o}"] = float(pn[0]); vals[f"y{o}"] = float(pn[1])
            return vals
        return path, c, ang0

    def rotate(self, angle, orbits=None, about="auto", absolute=False):
        """A copy with the rigid unit(s) rotated by `angle` degrees (see rotation_path)."""
        path, _, _ = self.rotation_path(orbits, about, absolute)
        return self.set(**path(angle))

    # --- scans --------------------------------------------------------------------------------
    def scan(self, name=None, lo=None, hi=None, n=181, path=None):
        """Move along a line in parameter space and report what happens.

        scan("b", 0.6, 1.8)      one parameter from lo to hi
        scan(path=f, lo=, hi=)   f(s) -> dict of parameter values (e.g. from rotation_path)

        Returns dict(s, events, intervals, lengths, planar):
        events    : values of s where the group is larger (kind "supergroup", with the group by both
                    routes and the equations of the locus), where a bond passes through a site
                    ("bond through site") or two sites coincide ("collision"). Found by locating the
                    zeros of the distance to each special locus / of the clearance on the grid and
                    refining them, so a jump at an isolated value is not missed by the grid.
        intervals : between events, the group (at the midpoint) and whether the drawing is planar.
        lengths   : bond-orbit lengths on the grid (a column that stays constant is a rigid bar).
        """
        from scipy.optimize import minimize_scalar
        if path is None:
            if name not in self.names:
                raise KeyError(f"{name!r} is not a free parameter; free: {self.names}")
            path = lambda s: {name: s}
        S = np.linspace(lo, hi, n)
        loci = self.supergroup_loci() if self.m and self.info()["connected"] else []
        fam = lambda s: self.set(**path(s))

        def zeros(f, kind, tol):
            vals = np.array([f(s) for s in S])
            ev = []
            if np.all(vals < tol):
                return [("all", kind)]
            for k in range(n):
                left = vals[k - 1] if k > 0 else math.inf
                right = vals[k + 1] if k < n - 1 else math.inf
                if vals[k] <= left and vals[k] <= right and vals[k] < 0.5 * max(vals.max(), tol):
                    a, b = S[max(k - 1, 0)], S[min(k + 1, n - 1)]
                    r = minimize_scalar(f, bounds=(a, b), method="bounded", options=dict(xatol=1e-13))
                    s0, f0 = (r.x, r.fun) if r.fun < vals[k] else (S[k], vals[k])
                    for cand in (float(Fraction(s0).limit_denominator(720)), round(s0, 6)):
                        if abs(cand - s0) < 1e-6 and lo - 1e-12 <= cand <= hi + 1e-12 and f(cand) <= max(f0, tol * 1e-3):
                            s0, f0 = cand, f(cand); break
                    if f0 < tol:
                        ev.append((float(s0), kind))
            return ev

        raw, whole = [], []
        for L in loci:
            for z in zeros(lambda s: L.residual(path(s)), "supergroup", 1e-8):
                (whole if z[0] == "all" else raw).append(z + (L,))
        if self.m:
            for z in zeros(lambda s: min_separation(fam(s).cell()), "collision", 1e-7):
                raw.append(z + (None,))
            for z in zeros(lambda s: clearance(fam(s).cell()), "bond through site", 1e-7):
                raw.append(z + (None,))
        raw = [z for z in raw if z[0] != "all"]
        raw.sort(key=lambda z: z[0])
        events = []
        for s0, kind, L in raw:
            if events and abs(events[-1]["s"] - s0) < 1e-6:
                e = events[-1]
                if kind != "supergroup":
                    e["kind"] = kind if e["kind"] == "supergroup" else e["kind"]
                elif L is not None:
                    e["equations"] += [q for q in L.equations() if q not in e["equations"]]
                continue
            events.append(dict(s=s0, kind=kind, equations=list(L.equations()) if L is not None else []))
        for e in events:
            f = fam(e["s"]); c = f.cell()
            if min_separation(c) < 1e-6:
                e["kind"] = "collision"; e["group"] = e["spglib"] = None
                continue
            g, mp = f.group_at()
            e["group"], e["sites"], e["spglib"] = g, mp, spg_group(c)
        cuts = [lo] + [e["s"] for e in events if lo < e["s"] < hi] + [hi]
        intervals = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 1e-9:
                continue
            f = fam(0.5 * (a + b)); c = f.cell()
            g, mp = f.group_at()
            intervals.append(dict(lo=a, hi=b, group=g, sites=mp, spglib=spg_group(c), planar=not has_crossings(c)))
        planar = np.array([not has_crossings(fam(s).cell()) for s in S])
        lengths = np.array([fam(s).bond_lengths() for s in S])
        # control: planarity may only change at an event
        unresolved = []
        for k in range(n - 1):
            if planar[k] != planar[k + 1] and not any(S[k] - 1e-9 <= e["s"] <= S[k + 1] + 1e-9 for e in events):
                unresolved.append((float(S[k]), float(S[k + 1])))
        return dict(s=S, events=events, intervals=intervals, lengths=lengths, planar=planar,
                    whole_path_in=[L for (_, _, L) in whole], unresolved_planarity_changes=unresolved)

    def scan_rotation(self, lo=None, hi=None, n=181, orbits=None, about="auto"):
        """scan() along the rotation of a rigid unit. s is the polar angle, in degrees, of the
        generating point about the centre of the unit, measured from the a axis (the current
        lattice is at res["current_angle"]). Default range: 45 degrees either side of the current angle."""
        path, c, ang0 = self.rotation_path(orbits, about, absolute=True)
        lo = ang0 - 45.0 if lo is None else lo
        hi = ang0 + 45.0 if hi is None else hi
        res = self.scan(path=path, lo=lo, hi=hi, n=n)
        res["centre"] = c; res["current_angle"] = ang0
        return res

    def print_scan(self, res, what="s"):
        print(f"{self.label} ({self.group}): scan of {what} from {res['s'][0]:g} to {res['s'][-1]:g}"
              + (f" (the lattice as it is now: {res['current_angle']:.6g})" if "current_angle" in res else ""))
        for L in res["whole_path_in"]:
            print(f"  the whole path lies in the special locus {L.equations()} ({L.group})")
        ev = {e["s"]: e for e in res["events"]}
        items = sorted([(i["lo"], 1, i) for i in res["intervals"]] + [(e["s"], 0, e) for e in res["events"]],
                       key=lambda x: (x[0], x[1]))
        for _, isint, x in items:
            if isint:
                ag = "" if x["spglib"] in (None, x["group"]) else f" [spglib: {x['spglib']}]"
                print(f"  {what} in ({x['lo']:.6g}, {x['hi']:.6g}): {x['group']}{ag}, {x['sites']} sites/cell, "
                      f"{'planar' if x['planar'] else 'bonds cross'}")
            elif x["kind"] == "supergroup":
                ag = "both routes" if x["spglib"] == x["group"] else f"spglib: {x['spglib']}"
                print(f"  {what} = {x['s']:.9g}: JUMP to {x['group']} ({ag}), {x['sites']} sites/cell"
                      + (f"; locus {', '.join(x['equations'])}" if x["equations"] else ""))
            else:
                print(f"  {what} = {x['s']:.9g}: not a lattice of this network ({x['kind']})"
                      + (f"; group there {x['group']}" if x.get("group") else ""))
        if res["unresolved_planarity_changes"]:
            print(f"  WARNING: planarity changes without a located event in {res['unresolved_planarity_changes']}")


# ----------------------------------------------------------------------------------------------
# Standard setting of an existing cell
# ----------------------------------------------------------------------------------------------
def _standardise(pc, G, use_types=False):
    """Find a conventional cell C (Cartesian columns) and an origin w (in units of C) such that every
    operation of G, written in its ITA standard setting in that cell and origin, is a symmetry of
    the lattice pc, and the translations of G are exactly the periods of pc. Returns (C, w)."""
    A = pc.A
    ops = _symmetry_ops(pc, use_types=use_types)
    cart = [(A @ Mf @ np.linalg.inv(A), A @ t) for Mf, t in ops]
    det = 2 if G.centred else 1
    cands = []
    for ent in itertools.product(range(-2, 3), repeat=4):
        N = np.array(ent).reshape(2, 2)
        if int(round(np.linalg.det(N))) != det:
            continue
        if G.centred and ((N[0, 0] + N[0, 1]) % 2 or (N[1, 0] + N[1, 1]) % 2):
            continue
        C = A @ N
        l1, l2 = np.linalg.norm(C[:, 0]), np.linalg.norm(C[:, 1])
        cosg = C[:, 0] @ C[:, 1] / (l1 * l2)
        if G.system == "rectangular" and abs(cosg) > 1e-7:
            continue
        if G.system == "square" and (abs(cosg) > 1e-7 or abs(l1 - l2) > 1e-7 * l1):
            continue
        if G.system == "hexagonal" and (abs(cosg + 0.5) > 1e-7 or abs(l1 - l2) > 1e-7 * l1):
            continue
        own = ent == (1, 0, 0, 1)                  # the cell as given is preferred when it fits
        if G.system == "oblique" and cosg < -1e-9 and not own:
            continue
        cands.append((not own, round(l1 + l2, 9), round(abs(math.atan2(C[1, 0], C[0, 0])), 9), tuple(ent), C))
    cands.sort(key=lambda x: x[:4])
    lat_shifts = [np.array(L, float) for L in itertools.product(range(-1, 2), repeat=2)]
    if G.centred:
        lat_shifts = lat_shifts + [L + 0.5 for L in lat_shifts]

    def in_lat(d):
        return _is_int(d, 1e-6) or (G.centred and _is_int(d - 0.5, 1e-6))

    for *_, C in cands:
        Ci = np.linalg.inv(C)
        conv = [(Ci @ Mc @ C, Ci @ tc) for Mc, tc in cart]          # found operations in the candidate cell
        table = G.ops[:len(G.ops) // 2] if G.centred else G.ops
        match = []
        for M, t in table:
            js = [j for j, (Mj, _) in enumerate(conv) if np.allclose(Mj, M, atol=1e-6)]
            if not js:
                match = None; break
            match.append(js)
        if match is None:
            continue
        # candidate origins from the first operation that is not the identity
        ws = [np.zeros(2)]
        for (M, t), js in zip(table, match):
            if np.array_equal(M, np.eye(2, dtype=int)):
                continue
            ws = []
            for j in js:
                for L in lat_shifts:
                    w, *_ = np.linalg.lstsq(M - I2, t - conv[j][1] + L, rcond=None)
                    ws.append(w)
            break
        ws.sort(key=lambda w: round(float(np.linalg.norm(C @ w)), 9))
        for w in ws:
            ok = True
            for (M, t), js in zip(table, match):
                if not any(in_lat(conv[j][1] + (M - I2) @ w - t) for j in js):
                    ok = False; break
            if ok:
                return C, w
    raise RuntimeError(f"could not bring the lattice to the standard setting of {G.symbol}: "
                       "is this group a subgroup of the lattice's group with the same cell?")


# ----------------------------------------------------------------------------------------------
# Exhaustive lists
# ----------------------------------------------------------------------------------------------
_REF_T = [(0.2137 + 0.3819 * k) % 1 for k in range(12)]
_REF_XY = [((0.1873 + 0.3819 * k) % 1, (0.3391 + 0.2764 * k) % 1) for k in range(12)]


class FamilyList:
    """Result of enumerate_families: .families (sorted), .complete, .coverage (what was enumerated),
    .counts, .log."""

    def __init__(self):
        self.families, self.log, self.counts = [], [], Counter()
        self.complete = True
        self.coverage = ""

    def __iter__(self):
        return iter(self.families)

    def __len__(self):
        return len(self.families)

    def __getitem__(self, k):
        return self.families[k]

    def show(self, limit=30, plot=False, special=False):
        print(self.coverage)
        print(f"{len(self.families)} family/families" + ("" if self.complete else "  [TIME LIMIT REACHED: the list is incomplete]"))
        for k, f in enumerate(self.families[:limit], 1):
            i = f.info()
            pl = {True: "planar members found", False: "no planar member found in the samples"}[f.planar]
            print(f"\n#{k}  {i['sites']} sites, {i['bond_orbits']} bond orbit(s), {i['bonds']} bonds/cell | coordination {i['degrees']} "
                  f"| Maxwell {i['maxwell']} | {f.n_free} free parameter(s): {', '.join(f.names) or '-'} | {pl}")
            print(f"    Wyckoff: {'; '.join(i['wyckoff'])}")
            print("    " + f.code())
            if special:
                for L in f.supergroup_loci():
                    print(f"    special: {', '.join(L.equations())} -> {L.group}")
        if plot and self.families:
            import matplotlib.pyplot as plt
            fams = self.families[:limit]
            ncols = min(5, len(fams)); rows = int(math.ceil(len(fams) / ncols))
            fig, axs = plt.subplots(rows, ncols, figsize=(3.3 * ncols, 3.6 * rows), squeeze=False)
            for k, (ax, f) in enumerate(zip(axs.flat, fams), 1):
                f.plot(ax=ax, title=False)
                ax.set_title(f"#{k} {f.group}: {f.m} sites, {len(f.orbits)} bond orbit(s)\n{f.n_free} free parameter(s)", fontsize=8)
            for ax in list(axs.flat)[len(fams):]:
                ax.axis("off")
            fig.tight_layout()
        return self


def _feature_ok(value, want):
    if want is None:
        return True
    if callable(want):
        return bool(want(value))
    return value == want


def enumerate_families(group, sites, max_bond_orbits=3, reach=1, min_degree=2, degrees=None, maxwell=None,
                       planar=None, network=None, n_samples=120, time_limit=300, seed=0, verbose=False,
                       trust_rejections=2):
    """Every family of `group` with `sites` sites per primitive cell and the requested features.

    WHAT IS ENUMERATED (exhaustively, no sampling):
      * every choice of Wyckoff positions of the group with `sites` sites per primitive cell in all
        (a rotation centre at most once; mirror lines and the general position any number of times);
      * for each choice: every set of 1..max_bond_orbits bond orbits, where the candidate orbits
        are all those containing a bond that can be drawn no longer than `reach` cells: its
        displacement, in units of the two primitive cell vectors, is at most `reach` in each
        direction for some position of the generating points (each free coordinate running over
        one period; the two directions are tested separately, which can only add candidates).
        This bound does not depend on where the sites happen to be when the list is made.
    A structure is kept if every site has at least min_degree bonds, the network is connected, no
    bond passes through a site identically, and the generic group is exactly `group` (tested at two
    random parameter points: computational evidence).
    WHAT IS NOT COVERED: bond orbits whose shortest possible bond is longer than `reach` cells,
    more bond orbits than max_bond_orbits, other numbers of sites.

    Features (None = no condition):
      degrees : the set of coordination numbers, e.g. 3 or {3, 4}, or a function of the sorted list
      maxwell : Maxwell count 2 sites - bonds, e.g. 0, or a function
      planar  : True keeps families in which a planar member was found among n_samples random
                parameter values; False keeps the others (sampling: evidence, not proof)
      network : a lattice (cell, catalogue name or Family); keeps the families with the same bond
                network (same_network)

    Two structures are the same family when they have the same orbit structure: the same Wyckoff
    positions with the same coordination sequences (5 terms), bond orbits of the same sizes between
    the same kinds of sites, turned end for end by an operation or not (the ingredients of the
    fingerprint of minimal_lattices), and the same Weisfeiler-Lehman hash of the ball of graph radius 4
    around a site of each orbit. This invariant is not proved complete: two different families with
    the same invariant would be merged (computational evidence).
    trust_rejections=2: once two structures with the same invariant were found to have a larger
    generic group, further ones with that invariant are not tested (None: test every one).
    Returns a FamilyList, sorted by (bond orbits, bonds per cell, free parameters).
    """
    G = WallpaperGroup(group)
    out = FamilyList()
    t0 = time.time()
    rng = np.random.default_rng(seed)
    if isinstance(degrees, int):
        degrees = {degrees}
    dmax = None                    # with explicit coordination numbers, larger ones are cut early
    if isinstance(degrees, (set, frozenset, list, tuple)):
        dset = sorted(set(degrees)); degrees = lambda d, dset=dset: d == dset
        dmax = dset[-1]
    nb_target = 2 * sites - maxwell if isinstance(maxwell, (int, np.integer)) else None
    target = None
    if network is not None:
        tcell = network.cell() if isinstance(network, Family) else primitive_cell(make_cell(network), use_types=False)
        tb = _bare_bonds(tcell)
        target = (tcell, network_key(tcell), _nkey(len(tcell.basis), tb, 3), _nkey(len(tcell.basis), tb, 5))
    wy = ML.wyckoff_positions(G)
    configs = ML.site_configurations(wy, sites)
    accepted = {}           # key -> list of families
    rejected = Counter()
    R = range(-reach - 3, reach + 4)
    timed_out = False
    for config in configs:
        pts, nl, ng = [], 0, 0
        for k in config:
            w = wy[k]
            if w["kind"] == "point":
                pts.append(np.array(w["rep"], float))
            elif w["kind"] == "line":
                pts.append(np.asarray(w["p0"], float) + _REF_T[nl] * np.asarray(w["d"], float)); nl += 1
            else:
                pts.append(np.array(_REF_XY[ng])); ng += 1
        try:
            base = Family(G, pts)
        except ValueError:
            continue
        if base.m != sites or min_separation(base.cell()) < 1e-6:
            continue
        m = base.m
        roots = [base.site_orbit.index(o) for o in range(len(base.gens))]
        # candidate bond orbits
        seen, orbits = set(), []
        Paff, qaff = base._affine_sites()
        for i in range(m):
            for j in range(i, m):
                coef = Paff[j] - Paff[i]
                span_lo = np.minimum(coef, 0).sum(axis=1); span_hi = np.maximum(coef, 0).sum(axis=1)
                for n in itertools.product(R, R):
                    if i == j and n == (0, 0):
                        continue
                    # displacement of the bond in cell units, as the generating points run over one period
                    const = qaff[j] - qaff[i] + np.array(n)
                    if np.any(const + span_lo > reach + 1e-9) or np.any(const + span_hi < -reach - 1e-9):
                        continue
                    bd = _canon_bond(i, j, n[0], n[1], 0)[:4]
                    if bd in seen:
                        continue
                    orb = base._orbit_of(bd)
                    seen.update(orb)
                    orbits.append(orb)
        # drop orbits whose bonds pass through a site at two generic points (then they always do)
        v1, v2 = base.random_values(rng), base.random_values(rng)
        good = []
        for orb in orbits:
            f = base._with_orbits([orb])
            if clearance(f.cell(**v1)) < 1e-7 and clearance(f.cell(**v2)) < 1e-7:
                out.counts["bond orbits dropped (pass through a site)"] += 1
                continue
            good.append(orb)
        orbits = good
        K = len(orbits)
        degv = np.zeros((K, m), int)
        for k, orb in enumerate(orbits):
            for (i, j, _, _) in orb:
                degv[k, i] += 1; degv[k, j] += 1
        nb = [len(o) for o in orbits]
        # is a bond of the orbit turned end for end by some operation? (part of the orbit structure)
        perm, shift = base._action()
        flip = []
        for orb in orbits:
            i, j, n1, n2 = orb[0]
            fl = False
            for g, (M, t) in enumerate(base._ops):
                dn = M @ np.array([n1, n2]) + shift[g][j] - shift[g][i]
                if perm[g][i] == j and perm[g][j] == i and dn[0] == -n1 and dn[1] == -n2:
                    fl = True; break
            flip.append(fl)
        out.log.append(f"Wyckoff choice {[wy[k]['name'] for k in config]}: {K} candidate bond orbits")
        if verbose:
            print(out.log[-1])
        def subsets():
            """All sets of 1..max_bond_orbits orbits, as (set, degrees, bonds). A partial set that already
            exceeds the largest coordination number or the bond count asked for is not extended: none
            of its supersets can have the feature."""
            stack = [((), np.zeros(m, int), 0, 0)]
            while stack:
                chosen, dg, nbd, start = stack.pop()
                for k in range(K - 1, start - 1, -1):
                    d2 = dg + degv[k]; n2 = nbd + nb[k]
                    if (dmax is not None and d2.max() > dmax) or (nb_target is not None and n2 > nb_target):
                        continue
                    c2 = chosen + (k,)
                    if len(c2) < max_bond_orbits:
                        stack.append((c2, d2, n2, k + 1))
                    yield c2, d2, n2

        for _one_pass in (0,):
            for sub, deg, nbonds in subsets():
                out.counts["bond-orbit sets enumerated"] += 1
                if deg.min() < min_degree:
                    continue
                if degrees is not None and not degrees(sorted(set(deg.tolist()))):
                    continue
                if not _feature_ok(2 * m - nbonds, maxwell):
                    continue
                bonds = [bd for k in sub for bd in orbits[k]]
                if not periodic_connected(m, bonds):
                    continue
                out.counts["connected, with the degree and Maxwell features"] += 1
                if (out.counts["bond-orbit sets enumerated"] & 255) == 0 and time.time() - t0 > time_limit:
                    timed_out = True; break
                if target is not None and _nkey(m, bonds, 3) != target[2]:
                    continue
                cs = ML.coordination_sequences(m, bonds, 5)
                c = Counter(cs); gg = 0
                for v in c.values():
                    gg = math.gcd(gg, v)
                nkey = tuple(sorted((k, v // gg) for k, v in c.items()))
                if target is not None and nkey != target[3]:
                    continue
                # orbit structure computed from the group operations (the fingerprint of minimal_lattices
                # if the generic group is G), plus hashes of the balls around one site of each orbit
                sdesc = tuple(sorted((g["mult"], g["sym"], cs[base.site_orbit.index(o)]) for o, g in enumerate(base.gens)))
                so = lambda k, e: base.site_orbit[orbits[k][0][e]]
                bdesc = tuple(sorted((len(orbits[k]), flip[k], tuple(sorted(((base.gens[so(k, 0)]["sym"], cs[base.site_orbit.index(so(k, 0))]),
                                                                             (base.gens[so(k, 1)]["sym"], cs[base.site_orbit.index(so(k, 1))])))))
                                     for k in sub))
                key = (sdesc, bdesc, nkey, _ball_hash(m, bonds, roots, 4))
                if key in accepted:
                    out.counts["duplicates of an accepted family"] += 1
                    continue
                if trust_rejections and rejected[key] >= trust_rejections:
                    out.counts["generic group larger than the group asked for"] += 1
                    continue
                fam = base._with_orbits([orbits[k] for k in sub])
                if not fam.is_exact(seed=int(rng.integers(1 << 30))):
                    out.counts["generic group larger than the group asked for"] += 1
                    rejected[key] += 1
                    continue
                if target is not None and not same_network(fam.cell(), target[0]):
                    continue
                accepted.setdefault(key, []).append(fam)
            if timed_out:
                break
        if timed_out:
            break
    out.complete = not timed_out
    fams = [f for fl in accepted.values() for f in fl]
    # choose good parameter values, and look for a planar member
    res = []
    for f in fams:
        best = None
        f.planar = False; nplanar = 0
        for s in range(n_samples):
            vals = f.random_values(rng)
            c = f.cell(**vals)
            sep = min_separation(c)
            if sep < 0.15:
                continue
            cl = clearance(c)
            pl = not has_crossings(c)
            nplanar += pl
            L = f.set(**vals).bond_lengths()
            score = (pl, cl > 0.08, min(L) / max(L))
            if best is None or score > best[0]:
                best = (score, vals)
        if best is not None:
            f2 = f.set(**{k: round(v, 3) for k, v in best[1].items()})
            if not (f2.status()["valid"] and f2.group_at() == (f.group, f.m)):
                f2 = f.set(**best[1])
            f2.planar = bool(best[0][0]); f2.planar_fraction = nplanar / n_samples
            f = f2
        if planar is not None and f.planar != planar:
            continue
        f.label = f"{G.symbol} family"
        res.append(f)
    res.sort(key=lambda f: (len(f.orbits), len(f.bonds()), f.n_free, f.code()))
    out.families = res
    out.counts["families"] = len(res)
    out.coverage = (f"{G.symbol}, {sites} site(s) per primitive cell: all {len(configs)} choice(s) of Wyckoff positions, "
                    f"all sets of up to {max_bond_orbits} bond orbit(s) that can be drawn at most {reach} cell(s) long "
                    f"({out.counts['bond-orbit sets enumerated']} sets examined"
                    + ("; sets already above the coordination or bond count asked for are cut early" if dmax is not None or nb_target is not None else "") + ")")
    out.seconds = time.time() - t0
    return out


def in_coverage(family, max_bond_orbits=3, reach=1):
    """Is this family inside the set that enumerate_families(family.group, family.m, max_bond_orbits,
    reach) enumerates? A direct test of the definition, without running the enumeration: at most
    max_bond_orbits bond orbits, and each orbit has a bond whose displacement (in units of the
    primitive cell vectors) can be made at most `reach` in each direction as the generating points
    run over one period. Returns (bool, reason)."""
    if len(family.orbits) > max_bond_orbits:
        return False, f"{len(family.orbits)} bond orbits, more than max_bond_orbits = {max_bond_orbits}"
    P, q = family._affine_sites()
    u = np.concatenate([g["u"] for g in family.gens]) if family.site_names else np.zeros(0)
    # shift every free coordinate into [0, 1): moving a generating point by a lattice vector relabels
    # the sites but does not change the lattice, so the displacements are evaluated over that period
    for k, orb in enumerate(family.orbits):
        ok = False
        for (i, j, n1, n2) in orb:
            coef = P[j] - P[i]
            lo = np.minimum(coef, 0).sum(axis=1); hi = np.maximum(coef, 0).sum(axis=1)
            const = q[j] - q[i] + np.array([n1, n2]) + coef @ np.floor(u)
            if np.all(const + lo <= reach + 1e-9) and np.all(const + hi >= -reach - 1e-9):
                ok = True; break
        if not ok:
            return False, f"bond orbit {k + 1} cannot be drawn within {reach} cell(s)"
    return True, "inside"


def find_network(cell, groups=None, sites=None, max_bond_orbits=4, reach=1, time_limit=120, verbose=True, **kw):
    """The families, in each wallpaper group, that carry the same bond network as `cell`
    (for example the chiral square lattice in p4 and the square-octagon lattice in p4m).
    sites: sites per primitive cell to try (default: as many as the lattice has). The network can
    also live in a group with a larger cell; pass sites=2 x ... to look there.
    Returns {group: FamilyList}; coverage and limits are those of enumerate_families."""
    c0 = cell.cell() if isinstance(cell, Family) else primitive_cell(make_cell(cell), use_types=False)
    m = len(c0.basis) if sites is None else sites
    b = len(_bare_bonds(c0)) * m // len(c0.basis)
    kw.setdefault("degrees", set(degrees_of(len(c0.basis), _bare_bonds(c0))))
    out = {}
    for g in (groups or WALLPAPER_GROUPS):
        r = enumerate_families(g, m, max_bond_orbits=max_bond_orbits, reach=reach, maxwell=2 * m - b, network=c0,
                               time_limit=time_limit, **kw)
        out[g] = r
        if verbose:
            print(f"{g:5s} {len(r)} family/families with this network"
                  + ("" if r.complete else "  [time limit: incomplete]") + f"  ({r.seconds:.0f} s)")
    return out


def link_families(lists):
    """Cross-links between lists of several groups. lists: {group: FamilyList or list of Family}.
    For every family, each special locus whose member is a family of another list is recorded:
    family.reaches = [(locus, group, index in that group's list)], and the supergroup family gets
    family.reached_from = [(group, index, equations)]. The match is by orbit-structure fingerprint."""
    fps = {}
    for g, fl in lists.items():
        for k, f in enumerate(fl):
            f.reaches = []; f.reached_from = getattr(f, "reached_from", [])
            fps[repr(f.fingerprint())] = (g, k)
    for g, fl in lists.items():
        for k, f in enumerate(fl):
            for L in f.supergroup_loci():
                fp = repr(ML.orbit_fingerprint(f.cell(**L.example))[0])
                if fp in fps:
                    g2, k2 = fps[fp]
                    f.reaches.append((L, g2, k2))
                    lists[g2][k2].reached_from.append((g, k, L.equations()))
    return lists
