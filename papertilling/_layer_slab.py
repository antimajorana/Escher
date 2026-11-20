"""Layer-group slabs: exact-group sites, shells-until-connected bonds, in-plane growth, AA stacking (C1 WP-L1).

Mirrors `papertilling-core/src/space3d/layer_slab.rs` (+ `layer_slab/shells.rs`); plan §5 T5, Q1, Q2, Q9, Q12.

  sites    the union of the general-position orbits of `data/layer_sites_80.json` (one, two or three orbits per
           group, chosen offline by `tools/build_layer_sites.py` so that spglib's layer-group check returns exactly
           G with G's operation count). A group whose entry is not verified reads "symmetry at least G".
           Fractional (x, y) in the conventional cell, Cartesian z (a = 1). Metric per crystal system (`METRICS`).
  bonds    shells until connected (plan Q1, rank 2): whole distance shells (ties at relative 1e-7) in increasing
           distance; a shell that changes none of (components, rank, index) of the quotient graph is skipped; stop
           at (1, 2, 1). Refused when a third site lies on a bond or one orbit gets two degrees.
  growth   the touching rule on the Dirichlet complex of the single layer (`_periodic_net`), then the grown patch
           is stacked.
  stack    m copies along the normal at spacing d, copy k shifted in-plane by k s (s fractional; AA is s = 0).
           Interlayer bonds: the shortest interlayer shell (the first shell already connects two connected layers,
           so the rule stops there). Refused when d <= the slab thickness (layers would interpenetrate) or when s
           breaks the group (`offset_keeps_group`). [PROVED] For m >= 2 copies, write lambda_g = R2 s - e s for an
           operation g = (R | t) with in-plane part R2 and normal sign e. (1) g maps the copy L + k c (c = (s, d))
           to L + k (e c + lambda_g), so copy 1 lands on a copy only if lambda_g is an in-plane translation.
           (2) Then the stack's operations with linear part R are exactly (R | t + delta_g), delta_g = (m - 1) s
           for e = -1 (copy k goes to copy m - 1 - k) and 0 for e = +1, since L has exactly G. They are G about a
           shifted origin w iff w - R2 w = delta_g mod translations for every g; with w = (m - 1) s / 2 + u this
           is u - R2 u = (m - 1) lambda_g / 2, which u = 0 solves for odd m. For even m it can fail: p11m with
           s = (1/2, 0) and m = 2 turns the mirror into a glide. u is searched on the 1/48 grid: the system is
           integral with right sides in (1/2) of the translations, so solutions have small denominators
           [COMPUTATIONAL EVIDENCE: a 1/240 grid gives the same verdicts on all 80 groups at 7 offsets, m = 2, 3].
           One layer (m = 1) has no offset to check. The test is "keeps G's operations": in a few cases (p11a with
           s = (0, 1/2), m = 2, say) the stack has a group of the same type in another setting (a glide's
           translation changes), and the refusal says so rather than "the type changed".
"""
from __future__ import annotations

import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent / "data"
SQ3_2 = 0.8660254037844386
METRICS = {"oblique": ((1.0, 0.0), (0.22, 1.24)), "rectangular": ((1.0, 0.0), (0.0, 1.32)),
           "square": ((1.0, 0.0), (0.0, 1.0)), "hexagonal": ((1.0, 0.0), (-0.5, SQ3_2))}
SHELL_RTOL = 1e-7
ON_BOND_TOL = 1e-6
MAX_SHELLS = 200
DEDUPE_TOL = 1e-6


class SlabRefusal(ValueError):
    """A request that cannot be built, with the reason."""


@lru_cache(maxsize=None)
def _groups():
    return {g["number"]: g for g in json.loads((_DATA / "layer_groups_80.json").read_text("utf-8"))["groups"]}


@lru_cache(maxsize=None)
def _site_data():
    p = _DATA / "layer_sites_80.json"
    if not p.exists():
        return {}
    return {g["number"]: g for g in json.loads(p.read_text("utf-8"))["groups"]}


def group(number):
    g = _groups().get(int(number))
    if g is None:
        raise SlabRefusal(f"unknown layer group #{number} (expected 1..=80)")
    return g


def operations(g):
    """Every operation (R 3x3, t 2) of the group modulo the conventional lattice, centring included."""
    shifts = [(0.0, 0.0), (0.5, 0.5)] if g["centred"] else [(0.0, 0.0)]
    out = []
    for op in g["ops"]:
        R = np.array(op["r"], float)
        t = np.array([a / b for a, b in op["t2"]], float)
        for s in shifts:
            out.append((R, t + np.array(s)))
    return out


def lattice(g):
    a1, a2 = METRICS[g["system"]]
    return np.array([[a1[0], a1[1], 0.0], [a2[0], a2[1], 0.0]])


def orbit(g, seed):
    """Orbit of (x, y, z) under the group: rows (x mod 1, y mod 1, z), first occurrence in operation order."""
    p = np.array(seed, float)
    out = []
    for R, t in operations(g):
        q = R @ p
        q = np.array([(q[0] + t[0]) % 1.0, (q[1] + t[1]) % 1.0, q[2]])
        if not any(_same(q, r) for r in out):
            out.append(q)
    return out


def _same(a, b):
    d = a[:2] - b[:2]
    d -= np.round(d)
    return abs(d[0]) < DEDUPE_TOL and abs(d[1]) < DEDUPE_TOL and abs(a[2] - b[2]) < DEDUPE_TOL


def sites_from_orbits(g, seeds):
    """(F (m, 3) fractional xy + z, labels) of the union of the seeds' orbits; refuses coincident orbits."""
    F, labels = [], []
    for k, s in enumerate(seeds):
        for q in orbit(g, s):
            if any(_same(q, r) for r in F):
                raise SlabRefusal(f"orbit {k} meets an earlier orbit")
            F.append(q)
            labels.append(k)
    return np.array(F), np.array(labels, int)


def cartesian(F, L):
    return F[:, :1] * L[0] + F[:, 1:2] * L[1] + np.outer(F[:, 2], [0.0, 0.0, 1.0])


# ---------------------------------------------------------------- quotient-graph state (rank 2)
def lattice_rank_index(V, d):
    """(rank, index) of the integer lattice spanned by V in Z^d; index 0 unless the rank is d."""
    rows = [list(map(int, v)) for v in V if any(v)]
    piv_rows = []
    for col in range(d):
        piv = [r for r in rows if r[col] != 0]
        rest = [r for r in rows if r[col] == 0]
        while len(piv) > 1:
            piv.sort(key=lambda r: abs(r[col]))
            p, nxt = piv[0], [piv[0]]
            for r in piv[1:]:
                q = r[col] // p[col]
                r2 = [a - q * b for a, b in zip(r, p)]
                (nxt if r2[col] != 0 else rest).append(r2)
            piv = nxt
        if piv:
            piv_rows.append(piv[0])
        rows = [r for r in rest if any(r)]
    rank = len(piv_rows)
    if rank < d:
        return rank, 0
    idx = 1
    for k, r in enumerate(piv_rows):
        idx *= r[k]
    return rank, abs(idx)


def quotient_state(m, bonds, d):
    """(components, rank, index) of the quotient graph with m sites and bonds (i, j, n)."""
    adj = [[] for _ in range(m)]
    for i, j, n in bonds:
        adj[i].append((j, tuple(n)))
        adj[j].append((i, tuple(-x for x in n)))
    pot = [None] * m
    comps, cyc = 0, []
    for s in range(m):
        if pot[s] is not None:
            continue
        comps += 1
        pot[s] = (0,) * d
        stack = [s]
        while stack:
            u = stack.pop()
            for v, n in adj[u]:
                w = tuple(a + b for a, b in zip(pot[u], n))
                if pot[v] is None:
                    pot[v] = w
                    stack.append(v)
                elif w != pot[v]:
                    cyc.append(tuple(a - b for a, b in zip(w, pot[v])))
    rank, index = lattice_rank_index(cyc, d)
    return comps, rank, index


# ---------------------------------------------------------------- shells until connected
def _pairs_within(P, L, r):
    """Directed pairs (i, j, n) with |P_j + n.L - P_i| <= r, n in Z^2, listed once (i < j, or i == j and n > 0)."""
    m = len(P)
    g = np.linalg.norm(np.linalg.inv(L[:, :2]), axis=0)
    R = [int(np.ceil(r * gk)) + 1 for gk in g]
    out = []
    for n in itertools.product(range(-R[0], R[0] + 1), range(-R[1], R[1] + 1)):
        shift = n[0] * L[0] + n[1] * L[1]
        for i in range(m):
            for j in range(i, m):
                if i == j and n <= (0, 0):
                    continue
                v = P[j] + shift - P[i]
                dist = float(np.linalg.norm(v))
                if 1e-9 < dist <= r:
                    out.append((dist, i, j, n, v))
    out.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return out


def shell_bonds(P, L, labels, max_shells=MAX_SHELLS):
    """The shells-until-connected bonds [(i, j, (n0, n1))] sorted, and the shell lengths used."""
    m = len(P)
    area = float(np.linalg.norm(np.cross(L[0], L[1])))
    r = 1.6 * np.sqrt(area / m)
    while True:
        pairs = _pairs_within(P, L, r)
        edges, lengths, cur, used, k = [], [], (m, 0, 0), 0, 0
        while k < len(pairs):
            d0, k0 = pairs[k][0], k
            if d0 * (1 + SHELL_RTOL) > r:
                break
            while k < len(pairs) and pairs[k][0] <= d0 * (1 + SHELL_RTOL):
                k += 1
            shell = [(p[1], p[2], p[3]) for p in pairs[k0:k]]
            used += 1
            if used > max_shells:
                raise SlabRefusal(f"no connected 2-periodic net within {max_shells} shells")
            new = quotient_state(m, edges + shell, 2)
            if new != cur:
                edges += shell
                lengths.append(d0)
                cur = new
                if cur == (1, 2, 1):
                    _validate(P, L, labels, edges, r)
                    return sorted(edges), lengths
        r *= 1.5


def _validate(P, L, labels, edges, r):
    deg = np.zeros(len(P), int)
    for i, j, _ in edges:
        deg[i] += 1
        deg[j] += 1
    for o in set(labels.tolist()):
        if len(set(deg[labels == o].tolist())) != 1:
            raise SlabRefusal("sites of one orbit have different degrees (tolerance failure)")
    near = [(p[1], p[2], p[3], p[4]) for p in _pairs_within(P, L, r)]
    disp = {}
    for i, j, n, v in near:
        disp.setdefault(i, []).append(v)
        disp.setdefault(j, []).append(-v)
    for i, j, n in edges:
        b = P[j] + n[0] * L[0] + n[1] * L[1] - P[i]
        bb = float(b @ b)
        for w in disp.get(i, []):
            t = float(w @ b) / bb
            perp = float(np.linalg.norm(np.cross(w, b))) / np.sqrt(bb)
            if 1e-9 < t < 1 - 1e-9 and perp < ON_BOND_TOL * np.sqrt(bb):
                raise SlabRefusal("a third site lies on a bond")


# ---------------------------------------------------------------- the slab
def layer(number):
    """dict(number, symbol, system, L (2, 3), F, P (m, 3), labels, bonds, shells, exact, note, thickness)."""
    g = group(number)
    d = _site_data().get(g["number"])
    exact = bool(d and d.get("exact"))
    seeds = d["orbits"] if d else [[0.17, 0.29, 0.18]]
    L = lattice(g)
    F, labels = sites_from_orbits(g, seeds)
    P = cartesian(F, L)
    bonds, shells = shell_bonds(P, L, labels)
    note = (f"exact layer group {g['symbol']} (spglib layer check)" if exact
            else f"symmetry at least {g['symbol']}")
    return dict(number=g["number"], symbol=g["symbol"], system=g["system"], L=L, F=F, P=P, labels=labels,
                bonds=bonds, shells=shells, exact=exact, note=note,
                thickness=float(P[:, 2].max() - P[:, 2].min()))


def offset_keeps_group(number, s, m=2):
    """True when m copies stacked with in-plane offset s (fractional) keep every operation of G (docstring)."""
    if m <= 1:
        return True
    g = group(number)
    s = np.array(s, float)
    lat = [np.zeros(2)] + ([np.array([0.5, 0.5])] if g["centred"] else [])

    def is_lat(v):
        return any(np.abs((v - c) - np.round(v - c)).max() < 1e-9 for c in lat)

    need = []
    for R, _ in operations(g):
        r2 = R[:2, :2]
        lam = r2 @ s - R[2, 2] * s
        if not is_lat(lam):
            return False
        need.append((r2, (m - 1) * lam / 2))
    if (m - 1) % 2 == 0:
        return True
    return bool(_grid_solutions(need, lat, 48).any())


def _grid_solutions(need, lat, den):
    """Mask of the grid points u (denominator den) with u - R2 u - h a translation for every (R2, h)."""
    a = np.arange(den) / den
    U = np.stack(np.meshgrid(a, a, indexing="ij"), -1).reshape(-1, 2)
    ok = np.ones(len(U), bool)
    for r2, h in need:
        V = U - U @ r2.T - h
        ok &= np.logical_or.reduce([np.abs((V - c) - np.round(V - c)).max(1) < 1e-9 for c in lat])
    return ok


def stack(lay, m, spacing, offset=(0.0, 0.0)):
    """Sites (k, i) for k < m and bonds of the stacked infinite layers: (in-layer bonds of each copy, the
    interlayer shell between consecutive copies [(i, j, n)])."""
    if m < 1:
        raise SlabRefusal("a stack needs at least one layer")
    if m > 1 and spacing <= lay["thickness"]:
        raise SlabRefusal(f"spacing {spacing:g} <= slab thickness {lay['thickness']:g}: layers would interpenetrate")
    if not offset_keeps_group(lay["number"], offset, m):
        raise SlabRefusal(f"offset {tuple(offset)} with {m} layers breaks {lay['symbol']}: some operation of the "
                          "group is not an operation of the stack (the type may survive in another setting)")
    L, P = lay["L"], lay["P"]
    c = offset[0] * L[0] + offset[1] * L[1] + np.array([0.0, 0.0, spacing])
    inter = []
    if m > 1:
        cand = []
        Q = P + c
        g = np.linalg.norm(np.linalg.inv(L[:, :2]), axis=0)
        r = float(np.linalg.norm(c))  # the copy of a site right above it is at |c|, so the shortest is <= r
        R = [int(np.ceil(r * gk)) + 1 for gk in g]
        for n in itertools.product(range(-R[0], R[0] + 1), range(-R[1], R[1] + 1)):
            for i in range(len(P)):
                for j in range(len(P)):
                    dist = float(np.linalg.norm(Q[j] + n[0] * L[0] + n[1] * L[1] - P[i]))
                    cand.append((dist, i, j, n))
        best = min(x[0] for x in cand)
        inter = sorted((i, j, n) for dist, i, j, n in cand if dist <= best * (1 + SHELL_RTOL))
    return dict(layers=m, spacing=spacing, offset=tuple(offset), stacking_vector=c, interlayer=inter)


def bond_set_invariant(lay):
    """True when every operation maps the site set and the bond set onto themselves."""
    g = group(lay["number"])
    F = lay["F"]
    bset = set((i, j, tuple(n)) for i, j, n in lay["bonds"])

    def find(q):
        for i, f in enumerate(F):
            if _same(q, f):
                return i, (int(round(q[0] - f[0])), int(round(q[1] - f[1])))
        return None

    for R, t in operations(g):
        for i, j, n in lay["bonds"]:
            qi = R @ F[i] + np.array([t[0], t[1], 0.0])
            qj = R @ (F[j] + np.array([n[0], n[1], 0.0])) + np.array([t[0], t[1], 0.0])
            a, b = find(qi), find(qj)
            if a is None or b is None:
                return False
            dn = (b[1][0] - a[1][0], b[1][1] - a[1][1])
            if (a[0], b[0], dn) not in bset and (b[0], a[0], (-dn[0], -dn[1])) not in bset:
                return False
    return True


def grow_slab(number, seed="vertex", layers=2, m=1, spacing=1.2, offset=(0.0, 0.0), max_cells=20_000):
    """Grow one layer in-plane (touching rule on its Dirichlet complex), then stack m copies; mirrors
    `layer_slab::grow_layer_slab`. Returns dict(stats, stopped, mode, euler_defined, cells, positions,
    bonds [(a, b, kind)], exact, note)."""
    from ._growth_engine import grow
    from ._periodic_net import PeriodicNet

    lay = layer(number)
    st = stack(lay, m, spacing, offset)
    L, P = lay["L"], lay["P"]
    net = PeriodicNet(L, P)
    g = grow(net, net.seed_cells(seed), max_layers=layers, max_cells=max_cells)
    cells = [c for layer_cells in g["layers"] for c in layer_cells]
    index = {c: k for k, c in enumerate(cells)}
    n = len(cells)
    pos = [P[c[0]] + c[1] * L[0] + c[2] * L[1] + k * st["stacking_vector"] for k in range(m) for c in cells]
    pos = np.array(pos) - np.mean(pos, axis=0) if pos else np.zeros((0, 3))
    bonds = []
    for lst, kind, dk in ((lay["bonds"], 0, 0), (st["interlayer"], 1, 1)):
        for k in range(m - dk):
            for ci, c in enumerate(cells):
                for i, j, nn in lst:
                    if i == c[0]:
                        cj = index.get((j, c[1] + nn[0], c[2] + nn[1], 0))
                        if cj is not None:
                            bonds.append((k * n + ci, (k + dk) * n + cj, kind))
    return dict(stats=g["stats"], stopped=g["stopped"], mode=net.mode, euler_defined=net.euler_defined,
                cells=n, positions=pos, bonds=bonds, exact=lay["exact"], note=lay["note"])
