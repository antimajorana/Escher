"""
minimal_lattices: for each wallpaper group, the lattices with the fewest sites per primitive cell.

    res = enumerate_lattices("p4g", n=5)        # res["planar"], res["crossing"]: ranked entries
    plot_entries(res["planar"])

A *lattice for group G* here is a periodic network whose geometry (sites and bonds, colours ignored)
has exactly the symmetry group G, including exactly G's translations, that is connected, and in which
every site has at least `min_degree` bonds and no bond passes through a site.

Two lattices are the *same entry* when they have the same orbit structure: the same kinds of site
orbits (Wyckoff positions, up to the normaliser) joined by the same kinds of bond orbits, whatever
the coordinates. This is decided with a fingerprint computed from the lattice itself (see
`orbit_fingerprint`): for every site orbit its multiplicity, site symmetry and coordination sequence;
for every bond orbit its size, stabiliser, and the descriptors of its two ends.

Ranking: (sites per primitive cell, number of bond orbits, bonds per primitive cell).
"planar" entries can be drawn without crossing bonds; "crossing" entries were only ever found with
crossing bonds.

Search: site orbits are chosen among the Wyckoff positions of G (computed from its operations),
by increasing sites per cell; for each choice, random geometries are sampled (free coordinates and
free cell parameters); in each geometry every set of up to `max_bond_orbits` of the shortest
`max_candidates` bond orbits with length <= max(max_bond_ratio x shortest site distance,
max_bond_spacing x average site spacing) is tried (average spacing = (cell area / sites)^(1/2)).
This is a sampled search inside those cut-offs: a lattice that needs longer bonds, more bond orbits,
or a narrow window of coordinates can be missed.
"""
from __future__ import annotations

import itertools
import math
import time
from math import gcd

import numpy as np

from .wallpaper_lattices import (WallpaperGroup, UnitCell, symmetric_cell, periodic_connected, WALLPAPER_GROUPS, WallpaperLattice, PALETTE,
                                _frac_mod, _bond_orbits, _canon_bond, _symmetry_ops, _apply_to_site,
                                primitive_cell, identify_wallpaper_group, _segments_cross)

# ----------------------------------------------------------------------------------------------
# Wyckoff positions, computed from the group operations
# ----------------------------------------------------------------------------------------------


def _same_mod1(p, q, tol=1e-7):
    d = np.asarray(p) - np.asarray(q)
    return np.all(np.abs(d - np.round(d)) < tol)


def _site_symmetry_label(Ms):
    """Point-group label from the Cartesian matrices of a stabiliser."""
    rot = 1
    mir = 0
    for M in Ms:
        if np.linalg.det(M) > 0:
            ang = math.atan2(M[1, 0], M[0, 0])
            if abs(ang) > 1e-6:
                rot = max(rot, int(round(2 * math.pi / abs(ang))))
        else:
            mir += 1
    if mir == 0:
        return str(rot)
    if rot == 1:
        return "m"
    return {2: "2mm", 3: "3m", 4: "4mm", 6: "6mm"}[rot]


def _stabiliser(G, A, p):
    Ainv = np.linalg.inv(A)
    Ms = []
    for Mf, t in G.ops:
        if _same_mod1(Mf @ p + t, p, 1e-7):
            Ms.append(A @ Mf @ Ainv)
    return Ms


def wyckoff_positions(group, a=1.0, b=None, gamma=None):
    """Wyckoff positions of a group as families: points (no free coordinate), mirror lines
    (one free coordinate) and the general position (two). Multiplicities are per primitive cell.
    Positions related by an origin shift are listed separately; the fingerprint merges them."""
    G = group if isinstance(group, WallpaperGroup) else WallpaperGroup(group)
    A = np.column_stack(G.lattice(a, b, gamma))
    prim = 2 if G.centred else 1
    fams = []
    # isolated points: fixed points of rotations
    pts = []
    for Mf, t in G.ops:
        if round(np.linalg.det(Mf)) != 1 or np.array_equal(Mf, np.eye(2, dtype=int)):
            continue
        for L in itertools.product(range(-1, 2), repeat=2):
            c = _frac_mod(np.linalg.solve(np.eye(2) - Mf, t + np.array(L)))
            if not any(_same_mod1(c, q) for q in pts):
                pts.append(c)
    seen = []
    for c in sorted(pts, key=lambda q: (round(q[0], 6), round(q[1], 6))):
        if any(any(_same_mod1(c, q) for q in orb) for orb in seen):
            continue
        orb = G.orbit(c)
        seen.append(orb)
        fams.append(dict(kind="point", rep=c, mult=len(orb) // prim,
                         sym=_site_symmetry_label(_stabiliser(G, A, c))))
    # mirror lines: fixed lines of pure reflections
    lines = []
    for Mf, t in G.ops:
        if round(np.linalg.det(Mf)) != -1:
            continue
        w, V = np.linalg.eig(Mf.astype(float))
        d = np.real(V[:, np.argmin(np.abs(w - 1))])
        d = d / np.max(np.abs(d))
        for L in itertools.product(range(-1, 2), repeat=2):
            rhs = t + np.array(L)
            x, *_ = np.linalg.lstsq(np.eye(2) - Mf, rhs, rcond=None)
            if np.allclose((np.eye(2) - Mf) @ x, rhs, atol=1e-9):
                lines.append((x, d))
    rng = np.random.default_rng(12345)
    fam_lines = []
    for p0, d in lines:
        q = p0 + 0.3171 * d
        # same family as an earlier line if some image of a generic point of it lies on that line
        dup = False
        for (p1, d1) in fam_lines:
            for img in G.orbit(q):
                for L in itertools.product(range(-2, 3), repeat=2):
                    v = A @ (img + np.array(L) - p1); u = A @ d1
                    if abs(v[0] * u[1] - v[1] * u[0]) / np.linalg.norm(u) < 1e-7:
                        dup = True; break
                if dup: break
            if dup: break
        if not dup:
            fam_lines.append((p0, d))
            fams.append(dict(kind="line", p0=_frac_mod(p0), d=d, mult=len(G.orbit(q)) // prim,
                             sym=_site_symmetry_label(_stabiliser(G, A, q))))
    fams.append(dict(kind="general", mult=G.order // prim, sym="1"))
    for f in fams:
        if f["kind"] == "point":
            f["name"] = f"{f['mult']} {f['sym']} ({_fmt(f['rep'][0])}, {_fmt(f['rep'][1])})"
        elif f["kind"] == "line":
            f["name"] = f"{f['mult']} {f['sym']} line {_fmt_line(f['p0'], f['d'])}"
        else:
            f["name"] = f"{f['mult']} 1 (x, y)"
    return fams


def _fmt(x):
    for den in (1, 2, 3, 4, 6):
        if abs(x * den - round(x * den)) < 1e-6:
            n = int(round(x * den)) % den if den > 1 else int(round(x))
            return f"{n}/{den}" if den > 1 and n else ("0" if n == 0 else str(n))
    return f"{x:.3f}"


def _fmt_line(p0, d):
    s = ["", ""]
    for k in range(2):
        if abs(d[k]) < 1e-9:
            s[k] = _fmt(p0[k])
        else:
            c = d[k]
            coef = "" if abs(c - 1) < 1e-9 else ("-" if abs(c + 1) < 1e-9 else f"{c:g}")
            off = _fmt(p0[k]) if abs(p0[k]) > 1e-9 else ""
            s[k] = f"{coef}t" + (f"+{off}" if off and off != "0" else "")
    return f"({s[0]}, {s[1]})"


def site_configurations(fams, m):
    """All multisets of Wyckoff families with total multiplicity m (points at most once each)."""
    out = []

    def rec(start, left, chosen):
        if left == 0:
            out.append(list(chosen)); return
        for k in range(start, len(fams)):
            f = fams[k]
            if f["mult"] > left:
                continue
            if f["kind"] == "point" and k in chosen:
                continue
            chosen.append(k)
            rec(k + (1 if f["kind"] == "point" else 0), left - f["mult"], chosen)
            chosen.pop()
    rec(0, m, [])
    # dedupe (multisets)
    uniq = []
    for c in out:
        s = sorted(c)
        if s not in uniq:
            uniq.append(s)
    return uniq


# ----------------------------------------------------------------------------------------------
# Graph checks
# ----------------------------------------------------------------------------------------------
def coordination_sequences(nsites, bonds, depth=8):
    adj = [[] for _ in range(nsites)]
    for (i, j, n1, n2) in bonds:
        adj[i].append((j, n1, n2)); adj[j].append((i, -n1, -n2))
    out = []
    for s in range(nsites):
        seen = {(s, 0, 0)}
        frontier = [(s, 0, 0)]
        seq = []
        for _ in range(depth):
            nxt = []
            for (u, x, y) in frontier:
                for (v, a, b) in adj[u]:
                    key = (v, x + a, y + b)
                    if key not in seen:
                        seen.add(key); nxt.append(key)
            seq.append(len(nxt)); frontier = nxt
        out.append(tuple(seq))
    return out


# ----------------------------------------------------------------------------------------------
# Fingerprint of the orbit structure, computed from the lattice itself
# ----------------------------------------------------------------------------------------------
def orbit_fingerprint(cell, depth=8):
    """Invariant of the orbit structure of a lattice, from its own (bare) symmetry group.
    Returns (fingerprint, info) with info = dict(group, sites, bond_orbits, bonds, site_orbits)."""
    group = identify_wallpaper_group(cell, use_types=False)
    pc = primitive_cell(cell, use_types=False)
    ops = _symmetry_ops(pc, use_types=False)
    A = pc.A; Ainv = np.linalg.inv(A)
    F = np.array([Ainv @ p for p in pc.basis])
    n = len(F)
    bonds = sorted({_canon_bond(i, j, n1, n2, 0)[:4] for (i, j, n1, n2, _) in pc.bonds})
    cs = coordination_sequences(n, bonds, depth)
    # site orbits and site symmetries
    site_orbit = [-1] * n
    site_desc = []
    for s in range(n):
        if site_orbit[s] >= 0:
            continue
        k = len(site_desc)
        stab = []
        for Mf, t in ops:
            j, c = _apply_to_site(Mf, t, F, s)
            site_orbit[j] = k
            if j == s:
                stab.append(A @ Mf @ Ainv)
        mult = sum(1 for x in site_orbit if x == k)
        site_desc.append((mult, _site_symmetry_label(stab), cs[s]))
    # bond orbits
    borb = _bond_orbits(bonds, ops, F)
    groups = {}
    for bd, o in borb.items():
        groups.setdefault(o, []).append(bd)
    bond_desc = []
    for o, members in groups.items():
        i, j, n1, n2 = members[0]
        flips = 0
        for Mf, t in ops:
            si, ci = _apply_to_site(Mf, t, F, i)
            sj, cj = _apply_to_site(Mf, t, F, j)
            dn = Mf @ np.array([n1, n2]) + cj - ci
            if si == j and sj == i and np.array_equal(dn, [-n1, -n2]):
                flips += 1
        stab = len(ops) // len(members)
        ends = tuple(sorted([(site_desc[site_orbit[i]][1], site_desc[site_orbit[i]][2]),
                             (site_desc[site_orbit[j]][1], site_desc[site_orbit[j]][2])]))
        bond_desc.append((len(members), stab, flips > 0, ends))
    fp = (group, tuple(sorted(site_desc)), tuple(sorted(bond_desc)))
    info = dict(group=group, sites=n, bond_orbits=len(bond_desc), bonds=len(bonds),
                site_orbits=sorted(site_desc), degrees=sorted({sum(1 for b in bonds for e in (b[0], b[1]) if e == s)
                                                              for s in range(n)}))
    return fp, info


# ----------------------------------------------------------------------------------------------
# The search
# ----------------------------------------------------------------------------------------------
class Entry:
    def __init__(self, fp, info, cell, planar, wyckoff, recipe=None):
        self.fingerprint, self.info, self.cell, self.planar, self.wyckoff = fp, info, cell, planar, wyckoff
        self.recipe = recipe
        self.quality = _quality(cell)

    @property
    def key(self):
        i = self.info
        return (i["sites"], i["bond_orbits"], i["bonds"], repr(self.fingerprint))

    def coordination(self, terms=4):
        """Site orbits as 'site symmetry: coordination sequence' strings."""
        return [f"{sym}: {','.join(map(str, cs[:terms]))}" for (_, sym, cs) in self.info["site_orbits"]]

    def __repr__(self):
        i = self.info
        return (f"{i['group']}: {i['sites']} sites/cell, {i['bond_orbits']} bond orbit(s), {i['bonds']} bonds/cell, "
                f"{'planar' if self.planar else 'crossing'}; Wyckoff {self.wyckoff}; "
                f"coordination sequences {self.coordination()}")

    def to_dict(self):
        c = self.cell
        return dict(group=self.info["group"], sites=self.info["sites"], bond_orbits=self.info["bond_orbits"],
                    bonds=self.info["bonds"], planar=self.planar, wyckoff=self.wyckoff,
                    site_orbits=[[m, s, list(cs)] for (m, s, cs) in self.info["site_orbits"]],
                    degrees=self.info["degrees"], fingerprint=repr(self.fingerprint),
                    a1=c.a1.tolist(), a2=c.a2.tolist(), basis=c.basis.tolist(),
                    bond_list=[list(map(int, b)) for b in c.bonds], recipe=self.recipe)

    @classmethod
    def from_dict(cls, d):
        e = cls.__new__(cls)
        e.cell = UnitCell(f"min_{d['group']}", d["group"], d["group"], np.array(d["a1"]), np.array(d["a2"]),
                          np.array(d["basis"]), [tuple(b) for b in d["bond_list"]])
        e.fingerprint = d["fingerprint"]; e.planar = d["planar"]; e.wyckoff = d["wyckoff"]
        e.recipe = d.get("recipe")
        e.info = dict(group=d["group"], sites=d["sites"], bond_orbits=d["bond_orbits"], bonds=d["bonds"],
                      site_orbits=[(m, s, tuple(cs)) for (m, s, cs) in d["site_orbits"]], degrees=d["degrees"])
        e.quality = _quality(e.cell)
        return e


def _quality(cell):
    L = [np.linalg.norm(cell.basis[j] + cell.A @ np.array([n1, n2]) - cell.basis[i])
         for (i, j, n1, n2, _) in cell.bonds]
    return min(L) / max(L)


def _sample_lattice(G, rng):
    if G.system == "oblique":
        return dict(a=1.0, b=round(float(rng.uniform(0.75, 1.45)), 2), gamma=float(rng.integers(62, 87)))
    if G.system == "rectangular":
        r = float(rng.uniform(0.6, 1.7))
        if abs(r - 1) < 0.08 or (G.centred and abs(r - math.sqrt(3)) < 0.08) or (G.centred and abs(r - 1 / math.sqrt(3)) < 0.05):
            r += 0.15
        return dict(a=1.0, b=round(r, 2))
    return dict(a=1.0)


def _sample_sites(fams, config, rng):
    pts = []
    for k in config:
        f = fams[k]
        if f["kind"] == "point":
            pts.append(np.array(f["rep"], float))
        elif f["kind"] == "line":
            pts.append(np.round((f["p0"] + rng.integers(1, 100) / 100 * f["d"]) % 1.0, 9) % 1.0)
        else:
            pts.append(rng.integers(1, 100, 2) / 100)
    return pts


def _candidate_bond_orbits(G, A, F, ratio, max_candidates, spacing_ratio=2.5, translation_ratio=2.7):
    X = (A @ F.T).T
    m = len(F)
    d = []
    R = 4
    for i in range(m):
        for j in range(m):
            for n in itertools.product(range(-R, R + 1), repeat=2):
                if i == j and n == (0, 0):
                    continue
                d.append((np.linalg.norm(X[j] + A @ np.array(n) - X[i]), i, j, n))
    dmin = min(x[0] for x in d)
    tmin = min(np.linalg.norm(A @ np.array(n)) for n in itertools.product(range(-2, 3), repeat=2) if n != (0, 0))
    cut = max(ratio * dmin, spacing_ratio * math.sqrt(abs(np.linalg.det(A)) / m), translation_ratio * tmin)
    cand = {_canon_bond(i, j, n[0], n[1], 0)[:4]: L for (L, i, j, n) in d if L <= cut + 1e-9}
    orb = _bond_orbits(list(cand), G.ops, F)
    # every image of a candidate bond belongs to its orbit, also images outside the enumerated range
    groups = {}
    for bd, o in orb.items():
        groups.setdefault(o, []).append(bd)
    length = lambda bs: min(cand[b] for b in bs if b in cand)
    orbits = sorted((sorted(bs) for bs in groups.values()), key=length)[:max_candidates]
    return orbits, dmin, X


def _bond_passes_site(X, A, bd, tol=1e-7):
    i, j, n1, n2 = bd
    p, q = X[i], X[j] + A @ np.array([n1, n2])
    v = q - p; L2 = v @ v
    Rr = (X[:, None, :] + (_SHIFTS @ A.T)[None, :, :]).reshape(-1, 2)
    s = (Rr - p) @ v / L2
    d = np.linalg.norm(p + s[:, None] * v - Rr, axis=1)
    return bool(np.any((s > 1e-6) & (s < 1 - 1e-6) & (d < tol * max(1.0, math.sqrt(L2)))))


def _orbit_unreduced(G, p):
    """Orbit of p without reducing into the unit cell, so that site labels (and the bond labels
    built on them) stay the same as the free coordinates of p vary."""
    pts = []
    for M, t in G.ops:
        q = M @ p + t
        if not any(_same_mod1(q, r) for r in pts):
            pts.append(q)
    return pts


_SHIFTS = np.array(list(itertools.product(range(-5, 6), repeat=2)), float)


def _segments(X, A, orbit):
    P = np.array([X[i] for (i, j, n1, n2) in orbit])
    Q = np.array([X[j] + A @ np.array([n1, n2]) for (i, j, n1, n2) in orbit])
    return P, Q


def _orbits_cross(X, A, oa, ob, tol=1e-9):
    """Does a bond of orbit oa properly cross a bond of orbit ob (or a lattice translate of one)?"""
    P, Q = _segments(X, A, oa)
    R, S = _segments(X, A, ob)
    sh = _SHIFTS @ A.T
    R = (R[:, None, :] + sh[None, :, :]).reshape(-1, 2)
    S = (S[:, None, :] + sh[None, :, :]).reshape(-1, 2)
    a = (Q - P)[:, None, :]; b = (S - R)[None, :, :]
    w = R[None, :, :] - P[:, None, :]
    cr = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    den = cr(a, b)
    good = np.abs(den) > tol
    den = np.where(good, den, 1.0)
    u = cr(w, b) / den; v = cr(w, a) / den
    return bool(np.any(good & (u > tol) & (u < 1 - tol) & (v > tol) & (v < 1 - tol)))


def _exact(cell, symbol, m):
    """Does this geometry (bond colours ignored) have exactly the group `symbol` and m sites per primitive cell?"""
    return (identify_wallpaper_group(cell, use_types=False) == symbol
            and len(primitive_cell(cell, use_types=False).basis) == m)


def enumerate_lattices(group, n=5, max_sites=12, min_sites=1, samples=40, max_bond_orbits=5, max_candidates=10,
                       max_bond_ratio=2.0, max_bond_spacing=2.5, max_bond_translation=2.7, min_degree=2, min_separation=0.15, seed=0,
                       time_limit=600,
                       verbose=False):
    """Lattices with exactly the symmetry of `group`, fewest sites per primitive cell first.

    Returns dict(planar=[Entry...], crossing=[Entry...], all=..., levels_done=m, complete=bool, log=[...]).
    The search goes level by level (sites per primitive cell = 1, 2, ...) and stops after the first
    level at which both lists hold at least n entries (or at max_sites / time_limit seconds).
    """
    G = WallpaperGroup(group)
    rng = np.random.default_rng(seed)
    fams = wyckoff_positions(G)
    found = {}
    memo = {}
    log = []
    t0 = time.time()
    prim = 2 if G.centred else 1
    m_done = min_sites - 1
    timed_out = False
    for m in range(min_sites, max_sites + 1):
        configs = site_configurations(fams, m)
        for config in configs:
            has_free = any(fams[k]["kind"] != "point" for k in config) or G.system in ("oblique", "rectangular")
            for _ in range(samples if has_free else 1):
                if time.time() - t0 > time_limit:
                    timed_out = True; break
                lat = _sample_lattice(G, rng)
                A = np.column_stack(G.lattice(**lat))
                pts = _sample_sites(fams, config, rng)
                F = [q for p in pts for q in _orbit_unreduced(G, p)]
                if len(F) != m * prim:
                    continue
                if any(_same_mod1(F[a], F[b], 1e-6) for a in range(len(F)) for b in range(a)):
                    continue
                F = np.array(F)
                orbits, dmin, X = _candidate_bond_orbits(G, A, F, max_bond_ratio, max_candidates, max_bond_spacing,
                                                        max_bond_translation)
                if dmin < min_separation * math.sqrt(abs(np.linalg.det(A)) / len(F)):
                    continue
                ok = [not any(_bond_passes_site(X, A, bd) for bd in o[:1]) for o in orbits]
                K = len(orbits)
                cross = np.zeros((K, K), bool)
                for a in range(K):
                    for b in range(a, K):
                        cross[a, b] = cross[b, a] = _orbits_cross(X, A, orbits[a], orbits[b])
                deg_of = []
                for o in orbits:
                    dgs = np.zeros(len(F), int)
                    for (i, j, _, _) in o:
                        dgs[i] += 1; dgs[j] += 1
                    deg_of.append(dgs)
                for size in range(1, max_bond_orbits + 1):
                    for sub in itertools.combinations([k for k in range(K) if ok[k]], size):
                        deg = sum(deg_of[k] for k in sub)
                        if deg.min() < min_degree:
                            continue
                        bonds = [bd for k in sub for bd in orbits[k]]
                        if not periodic_connected(len(F), bonds):
                            continue
                        planar = not any(cross[a, b] for a in sub for b in sub)
                        cell = UnitCell(f"min_{G.symbol}", f"{G.symbol}", G.symbol, A[:, 0], A[:, 1], X.copy(),
                                        sorted(bd + (t,) for t, k in enumerate(sub) for bd in orbits[k]))
                        # the structure (config + bond orbits in parameter-independent labels) decides
                        # the group and the fingerprint; classify each structure once
                        # A structure's verdict is cached. A rejection is trusted only after two geometries
                        # agree (one sampled geometry can be special and have extra symmetry), and every
                        # geometry that is stored is itself checked to have exactly the group G.
                        skey = (tuple(config), frozenset(frozenset(orbits[k]) for k in sub))
                        v = memo.get(skey)
                        checked = False
                        if v is None or (v[0] == "no" and v[1] < 2):
                            if _exact(cell, G.symbol, m):
                                memo[skey] = ("yes", orbit_fingerprint(cell)); checked = True
                            else:
                                memo[skey] = ("no", (v[1] if v else 0) + 1)
                                continue
                        elif v[0] == "no":
                            continue
                        fp, info = memo[skey][1]
                        old = found.get(fp)
                        if not (old is None or (planar and not old.planar)
                                or (planar == old.planar and _quality(cell) > old.quality)):
                            continue
                        if not checked and not _exact(cell, G.symbol, m):
                            continue
                        wy = [fams[k]["name"] for k in config]
                        e = Entry(fp, info, cell, planar, wy, _make_recipe(G, lat, pts, F, [orbits[k] for k in sub]))
                        if old is not None:
                            e.planar = e.planar or old.planar
                        found[fp] = e
            if timed_out:
                break
        if timed_out:
            log.append(f"time limit reached during level {m}")
            break
        m_done = m
        n_pl = sum(1 for e in found.values() if e.planar)
        n_cr = sum(1 for e in found.values() if not e.planar)
        log.append(f"level {m}: {len(configs)} site configuration(s); planar {n_pl}, crossing {n_cr} "
                   f"({time.time() - t0:.0f} s)")
        if verbose:
            print(log[-1])
        if n_pl >= n and n_cr >= n:
            break
    allx = sorted(found.values(), key=lambda e: e.key)
    return dict(group=G.symbol, planar=[e for e in allx if e.planar][:n],
                crossing=[e for e in allx if not e.planar][:n], all=allx, levels_done=m_done,
                complete=not timed_out, log=log)


def plot_entries(entries, ncols=5, n=2, size=3.3, symmetry=False, numbers=None):
    import matplotlib.pyplot as plt
    k = len(entries)
    if k == 0:
        return None
    rows = int(math.ceil(k / ncols))
    fig, axs = plt.subplots(rows, ncols, figsize=(size * ncols, size * rows * 1.1), squeeze=False)
    for k, (ax, e) in enumerate(zip(axs.flat, entries)):
        lat = WallpaperLattice(e.cell, n=n)
        lat.plot(ax=ax, title=False, show_sites=True, site_size=6, symmetry=symmetry)
        i = e.info
        head = f"#{numbers[k]}  " if numbers is not None else ""
        ax.set_title(f"{head}{i['group']}: {i['sites']} sites, {i['bond_orbits']} bond orbit(s), "
                     f"{'planar' if e.planar else 'crossing'}\n" + "; ".join(e.wyckoff), fontsize=7.5)
    for ax in list(axs.flat)[k:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------------------
# Recipes: rebuild a found lattice with symmetric_cell
# ----------------------------------------------------------------------------------------------
def _make_recipe(G, lat, pts, F, orbits):
    """Group, cell parameters, one generating point per site orbit and one generating bond per bond
    orbit, in fractional coordinates of the conventional cell. rebuild(recipe) rebuilds the lattice."""
    bonds = []
    for o in orbits:
        i, j, n1, n2 = o[0]
        p = np.asarray(F[i], float); q = np.asarray(F[j], float) + np.array([n1, n2])
        d = q - p
        if d[0] < -1e-9 or (abs(d[0]) < 1e-9 and d[1] < 0):     # write each bond pointing forward
            p, q = q, p
        sh = np.floor(p + 1e-9)
        bonds.append((tuple(float(x) for x in p - sh), tuple(float(x) for x in q - sh)))
    return dict(group=G.symbol, sites=[tuple(float(x) for x in p) for p in pts], bonds=bonds,
                **{k: float(v) for k, v in lat.items()})


def _num(x):
    from fractions import Fraction
    f = Fraction(x).limit_denominator(1200)
    if abs(float(f) - x) > 1e-9:
        return repr(round(x, 12))
    if f.denominator == 1:
        return str(f.numerator)
    if 100 % f.denominator == 0:
        return f"{float(f):g}"
    return f"{f.numerator}/{f.denominator}"


def recipe_code(entry_or_recipe):
    """The recipe as one line of Python that rebuilds the lattice (needs wallpaper_lattices)."""
    r = entry_or_recipe.recipe if isinstance(entry_or_recipe, Entry) else entry_or_recipe
    pt = lambda p: "(" + ", ".join(_num(x) for x in p) + ")"
    sites = ", ".join(pt(p) for p in r["sites"])
    bonds = ", ".join(f"({pt(p)}, {pt(q)})" for p, q in r["bonds"])
    lat = "".join(f", {k}={_num(r[k])}" for k in ("a", "b", "gamma") if k in r)
    return f'symmetric_cell("{r["group"]}", sites=[{sites}], bonds=[{bonds}]{lat})'


def rebuild(entry_or_recipe, **kwargs):
    """UnitCell rebuilt from a recipe (or an Entry's recipe) with symmetric_cell; each bond orbit
    gets its own bond type. Pass the result to WallpaperLattice(cell, n=...) to grow any size."""
    r = entry_or_recipe.recipe if isinstance(entry_or_recipe, Entry) else entry_or_recipe
    lat = {k: r[k] for k in ("a", "b", "gamma") if k in r}
    return symmetric_cell(r["group"], sites=[tuple(p) for p in r["sites"]],
                          bonds=[(tuple(p), tuple(q)) for p, q in r["bonds"]], **lat, **kwargs)


def list_lattices(group, sites=None, kind="all", show=30, max_sites=12, samples=40, time_limit=900, plot=True,
                  verbose=True, **search):
    """Every distinct lattice the search finds for `group` with the fewest sites per primitive cell
    (or with exactly `sites` sites per cell), numbered, with the one-line recipe that rebuilds each.

    kind : "all", "planar" or "crossing" (planar ones are listed first). With "planar" or "crossing"
           the fewest sites is taken over lattices of that kind only.
    show : how many to print and plot (the simplest first); all of them are returned.
    Returns the list of entries: entries[k-1] is #k. Rebuild one with rebuild(entries[k-1]),
    or paste its recipe line into any code that imports wallpaper_lattices.
    """
    levels = [sites] if sites is not None else range(1, max_sites + 1)
    for m in levels:
        r = enumerate_lattices(group, n=10**9, min_sites=m, max_sites=m, samples=samples,
                               time_limit=time_limit, **search)
        es = [e for e in r["all"] if e.planar] + [e for e in r["all"] if not e.planar]
        if kind != "all":
            es = [e for e in es if e.planar == (kind == "planar")]
        if es or not r["complete"]:
            break
    if verbose:
        print(f"{group}: {len(es)} lattice(s) with {m} site(s) per primitive cell"
              + ("" if r["complete"] else "  [time limit reached: the list may be incomplete]")
              + (f"; showing the first {show}" if len(es) > show else ""))
        for k, e in enumerate(es[:show], 1):
            i = e.info
            print(f"\n#{k}  {'planar  ' if e.planar else 'crossing'} | {i['bond_orbits']} bond orbit(s), "
                  f"{i['bonds']} bonds/cell | Wyckoff: {'; '.join(e.wyckoff)} | "
                  f"site symmetry: coordination sequence  {', '.join(e.coordination())}")
            print("    " + recipe_code(e))
    if plot and es:
        import matplotlib.pyplot as plt
        plot_entries(es[:show], numbers=list(range(1, min(show, len(es)) + 1)))
        plt.show()
    return es


def _catalogue_worker(args):
    g, n, kwargs = args
    t = time.time()
    r = enumerate_lattices(g, n=n, **kwargs)
    r["seconds"] = time.time() - t
    return g, r


def minimal_lattice_catalogue(groups=None, n=5, verbose=True, processes=1, **kwargs):
    """Run enumerate_lattices for several groups (default: all 17). Returns {group: result}.
    processes > 1 runs the groups in parallel (each group is searched exactly as in the serial run)."""
    groups = list(groups or WALLPAPER_GROUPS)
    jobs = [(g, n, kwargs) for g in groups]
    if processes > 1:
        import multiprocessing as mp
        with mp.Pool(processes) as pool:
            done = dict(pool.imap_unordered(_catalogue_worker, jobs))
    else:
        done = dict(_catalogue_worker(j) for j in jobs)
    out = {g: done[g] for g in groups}
    if verbose:
        for g, r in out.items():
            print(f"{g:5s} planar {len(r['planar'])}, crossing {len(r['crossing'])}, levels 1..{r['levels_done']}"
                  f"{'' if r['complete'] else ' (time limit)'}  {r['seconds']:.0f} s")
    return out


def save_catalogue(cat, path, settings=None):
    import json
    data = dict(settings=settings or {}, groups={
        g: dict(planar=[e.to_dict() for e in r["planar"]], crossing=[e.to_dict() for e in r["crossing"]],
                levels_done=r["levels_done"], complete=r["complete"], log=r["log"],
                found_total=len(r["all"]))
        for g, r in cat.items()})
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=1)


def load_catalogue(path):
    import json
    with open(path, encoding="utf8") as f:
        data = json.load(f)
    cat = {g: dict(group=g, planar=[Entry.from_dict(d) for d in r["planar"]],
                   crossing=[Entry.from_dict(d) for d in r["crossing"]], levels_done=r["levels_done"],
                   complete=r["complete"], log=r["log"], found_total=r["found_total"])
           for g, r in data["groups"].items()}
    return cat, data["settings"]
