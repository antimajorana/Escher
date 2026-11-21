"""Periodic 3D nets: connectivity state, coordination sequences and the duplicate key (C1 WP-P0, plan Q10).

A net is a quotient graph: m sites of a primitive cell and bonds (i, j, n), meaning site i in cell 0 joined to
site j in cell n (n in Z^3, primitive coordinates). Every bond is listed once.

  state(m, bonds)    (components, rank, index): the number of connected components of the quotient graph, the
                     rank of the lattice spanned by the cycle voltages, and (for rank 3) its index in Z^3. The
                     net is one connected 3-periodic net exactly when the state is (1, 3, 1); index k > 1
                     means k interpenetrating copies.
  coordination_sequence(m, bonds, site, depth)
  net_key_and_fingerprint(...)
                     plan Q10 key: (space group, sites per primitive cell, sorted (orbit size, CS to depth 12)
                     per site orbit). Two non-isomorphic nets can share it; the fingerprint adds the CS to
                     depth 14 and, per orbit, the sorted CS of the neighbouring orbits (one refinement step), so
                     the sweep can report "same key, different fingerprint" as a key collision.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _hnf_rows(V):
    """Integer row reduction of the integer vectors V (k, 3): the nonzero rows of an echelon form spanning
    the same lattice."""
    rows = [list(map(int, v)) for v in V if any(v)]
    out = []
    for col in range(3):
        piv = [r for r in rows if r[col] != 0]
        rest = [r for r in rows if r[col] == 0]
        while len(piv) > 1:                              # Euclid on column col
            piv.sort(key=lambda r: abs(r[col]))
            p = piv[0]
            nxt = [p]
            for r in piv[1:]:
                q = r[col] // p[col]
                r2 = [a - q * b for a, b in zip(r, p)]
                (nxt if r2[col] != 0 else rest).append(r2)
            piv = nxt
        if piv:
            out.append(piv[0])
        rows = [r for r in rest if any(r)]
    return out


def lattice_rank_index(V):
    """(rank, index) of the lattice spanned by integer vectors V in Z^3; index is 0 unless rank == 3."""
    E = _hnf_rows(V)
    rank = len(E)
    if rank < 3:
        return rank, 0
    return 3, abs(E[0][0] * E[1][1] * E[2][2])


def state(m, bonds):
    """(components, rank, index) of the quotient graph with m sites and bonds (i, j, n)."""
    adj = defaultdict(list)
    for i, j, n in bonds:
        n = tuple(int(x) for x in n)
        adj[i].append((j, n))
        adj[j].append((i, tuple(-x for x in n)))
    pot = [None] * m
    comps, cyc = 0, []
    for s in range(m):
        if pot[s] is not None:
            continue
        comps += 1
        pot[s] = (0, 0, 0)
        stack = [s]
        while stack:
            u = stack.pop()
            pu = pot[u]
            for v, n in adj[u]:
                w = (pu[0] + n[0], pu[1] + n[1], pu[2] + n[2])
                if pot[v] is None:
                    pot[v] = w
                    stack.append(v)
                elif w != pot[v]:
                    cyc.append((w[0] - pot[v][0], w[1] - pot[v][1], w[2] - pot[v][2]))
    rank, index = lattice_rank_index(cyc) if cyc else (0, 0)
    return comps, rank, index


def adjacency(bonds):
    adj = defaultdict(list)
    for i, j, n in bonds:
        n = tuple(int(x) for x in n)
        adj[i].append((j, n))
        adj[j].append((i, tuple(-x for x in n)))
    return adj


def coordination_sequence(adj, site, depth):
    """Numbers of vertices at graph distance 1..depth from (site, cell 0) in the periodic net."""
    start = (site, 0, 0, 0)
    seen = {start}
    frontier = [start]
    seq = []
    for _ in range(depth):
        nxt = []
        for v, a, b, c in frontier:
            for w, n in adj[v]:
                k = (w, a + n[0], b + n[1], c + n[2])
                if k not in seen:
                    seen.add(k)
                    nxt.append(k)
        frontier = nxt
        seq.append(len(frontier))
    return tuple(seq)


def net_key_and_fingerprint(group, m, orbit_of, orbit_reps, orbit_sizes, adj, depth=12, fp_depth=14):
    """The plan-Q10 key and the collision fingerprint, from one coordination sequence per site orbit.
    orbit_of: site -> orbit index; orbit_reps: one site per orbit; orbit_sizes: sites of each orbit per cell.
    key = (group, m, sorted (orbit size, CS to `depth`)); fingerprint = sorted (orbit size, CS to `fp_depth`,
    sorted CS-to-`depth` of the neighbours' orbits)."""
    cs = [coordination_sequence(adj, r, max(depth, fp_depth)) for r in orbit_reps]
    key = (int(group), int(m), tuple(sorted((int(s), c[:depth]) for s, c in zip(orbit_sizes, cs))))
    fp = tuple(sorted((int(s), c[:fp_depth], tuple(sorted(cs[orbit_of[w]][:depth] for w, _ in adj[r])))
                      for r, s, c in zip(orbit_reps, orbit_sizes, cs)))
    return key, fp


def degrees(m, bonds):
    deg = np.zeros(m, int)
    for i, j, _ in bonds:
        deg[i] += 1
        deg[j] += 1
    return deg
