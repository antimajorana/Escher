"""Hot loops of the exhaustive lattice search, each with a pure-Python reference.

All geometry here is in PRIMITIVE FRACTIONAL coordinates. Two segments cross, a point lies on a
segment, and the cyclic order of segments round a point are all unchanged by an orientation-
preserving linear map (proved result: these predicates are signs of 2x2 determinants, and
det(L u, L v) = det(L) det(u, v) with det L > 0). So planarity and the tiling do not depend on the
cell parameters b, gamma; only the site parameters matter.

Kernels (numba if importable, else the Python reference; `HAVE_NUMBA` tells which):
  sample_masks(Fs, bi, bj, bn, ptr, valid)      -> self_ok (K, W), pair_ok (K, K, W)   uint64 bitsets
      The numba kernel tests ONE bond of an orbit against all bonds of the other orbit (G maps
      crossings to crossings and sites onto sites: proved result); the Python reference tests every
      pair of bonds, so their agreement (tested) also checks that argument.
      bit s of self_ok[a]: at sample s no site lies inside a bond of orbit a, no two bonds of
      orbit a (or their translates) cross, and the sample is valid (no coincident sites).
      bit s of pair_ok[a, b]: no bond of orbit a crosses a bond of orbit b or a translate of it.
  subsets(n, ...)                                -> every n-set of orbits with a common planar
      sample, min degree >= min_degree, <= max_bonds bonds, periodic-connected.
  periodic_connected(m, bonds)                   -> bool (same definition as legacy
      wallpaper_lattices.periodic_connected: one-cell graph connected and its cycle vectors span Z^2).
Definitions above are mine (technical choices); the equality of numba and Python results is
tested in tests/test_enumerate.py (computational evidence).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

try:
    import numba
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                    # pragma: no cover - numba is optional
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if a and callable(a[0]) else deco

TOL_CROSS = 1e-9          # proper crossing: both parameters in (tol, 1 - tol)
TOL_ON = 1e-7             # a site lies on a bond: |cross| <= tol |q - p|^2, projection in (tol, 1 - tol)


def n_words(S):
    return (S + 63) // 64


def bits_to_list(mask):
    out = []
    for w, x in enumerate(np.asarray(mask, np.uint64)):
        x = int(x)
        while x:
            b = x & -x
            out.append(64 * w + b.bit_length() - 1)
            x ^= b
    return out


# ----------------------------------------------------------------------------------------------
# Python reference
# ----------------------------------------------------------------------------------------------
def _seg_cross_py(px, py, qx, qy, rx, ry, sx, sy):
    ax, ay = qx - px, qy - py
    bx, by = sx - rx, sy - ry
    den = ax * by - ay * bx
    if abs(den) < TOL_CROSS:
        return False
    wx, wy = rx - px, ry - py
    u = (wx * by - wy * bx) / den
    v = (wx * ay - wy * ax) / den
    return TOL_CROSS < u < 1 - TOL_CROSS and TOL_CROSS < v < 1 - TOL_CROSS


def _bonds_cross_py(P, Q, R, S_):
    """Does segment PQ cross some integer translate of segment RS?"""
    lo = [math.ceil(min(P[c], Q[c]) - max(R[c], S_[c]) - 1e-12) for c in range(2)]
    hi = [math.floor(max(P[c], Q[c]) - min(R[c], S_[c]) + 1e-12) for c in range(2)]
    for n1 in range(lo[0], hi[0] + 1):
        for n2 in range(lo[1], hi[1] + 1):
            if _seg_cross_py(P[0], P[1], Q[0], Q[1], R[0] + n1, R[1] + n2, S_[0] + n1, S_[1] + n2):
                return True
    return False


def _site_on_bond_py(F, P, Q):
    vx, vy = Q[0] - P[0], Q[1] - P[1]
    L2 = vx * vx + vy * vy
    for r in range(len(F)):
        lo = [math.ceil(min(P[c], Q[c]) - F[r][c] - 1e-9) for c in range(2)]
        hi = [math.floor(max(P[c], Q[c]) - F[r][c] + 1e-9) for c in range(2)]
        for n1 in range(lo[0], hi[0] + 1):
            for n2 in range(lo[1], hi[1] + 1):
                wx, wy = F[r][0] + n1 - P[0], F[r][1] + n2 - P[1]
                s = (wx * vx + wy * vy) / L2
                if TOL_ON < s < 1 - TOL_ON and abs(wx * vy - wy * vx) <= TOL_ON * L2:
                    return True
    return False


def sample_masks_py(Fs, bi, bj, bn, ptr, valid, do_pairs=True):
    S, m, _ = Fs.shape
    K = len(ptr) - 1
    W = n_words(S)
    self_ok = np.zeros((K, W), np.uint64)
    pair_ok = np.zeros((K, K, W), np.uint64)
    for s in range(S):
        if not valid[s]:
            continue
        F = Fs[s]
        P = [F[bi[e]] for e in range(len(bi))]
        Q = [F[bj[e]] + bn[e] for e in range(len(bi))]
        w, bit = divmod(s, 64)
        B = np.uint64(1 << bit)
        good = [not any(_site_on_bond_py(F, P[e], Q[e]) for e in range(ptr[a], ptr[a + 1])) for a in range(K)]
        for a in range(K):
            for b in range(a, K if do_pairs else a + 1):
                cr = False
                for e in range(ptr[a], ptr[a + 1]):
                    for f in range(ptr[b], ptr[b + 1]):
                        if _bonds_cross_py(P[e], Q[e], P[f], Q[f]):
                            cr = True; break
                    if cr:
                        break
                if not cr:
                    pair_ok[a, b, w] |= B; pair_ok[b, a, w] |= B
                    if a == b and good[a]:
                        self_ok[a, w] |= B
    return self_ok, pair_ok


def periodic_connected_py(m, bonds):
    adj = [[] for _ in range(m)]
    for (i, j, n1, n2) in bonds:
        adj[i].append((j, n1, n2)); adj[j].append((i, -n1, -n2))
    off = {0: (0, 0)}
    stack = [0]; cyc = []
    while stack:
        u = stack.pop()
        for v, a, b in adj[u]:
            c0 = (off[u][0] + a, off[u][1] + b)
            if v not in off:
                off[v] = c0; stack.append(v)
            else:
                c = (c0[0] - off[v][0], c0[1] - off[v][1])
                if c != (0, 0):
                    cyc.append(c)
    if len(off) < m:
        return False
    g = 0
    for p, q in itertools.combinations(cyc, 2):
        g = math.gcd(g, abs(p[0] * q[1] - p[1] * q[0]))
        if g == 1:
            return True
    return g == 1


def subsets_py(n, pair_ok, self_ok, degv, nb, bi, bj, bn, ptr, min_degree, max_bonds, first=None):
    """Reference: plain itertools.combinations with every test applied at the end."""
    K, m = degv.shape
    out, masks = [], []
    firsts = range(K) if first is None else [first]
    for f in firsts:
        for rest in itertools.combinations(range(f + 1, K), n - 1):
            sub = (f,) + rest
            mask = self_ok[sub[0]].copy()
            for a in sub:
                mask &= self_ok[a]
                for b in sub:
                    mask &= pair_ok[a, b]
            if not mask.any():
                continue
            deg = degv[list(sub)].sum(axis=0)
            if deg.min() < min_degree or nb[list(sub)].sum() > max_bonds:
                continue
            bonds = [(int(bi[e]), int(bj[e]), int(bn[e, 0]), int(bn[e, 1])) for a in sub for e in range(ptr[a], ptr[a + 1])]
            if not periodic_connected_py(m, bonds):
                continue
            out.append(sub); masks.append(mask)
    return out, masks


# ----------------------------------------------------------------------------------------------
# numba kernels (same algorithms)
# ----------------------------------------------------------------------------------------------
@njit(cache=True)
def _seg_cross_nb(px, py, qx, qy, rx, ry, sx, sy):
    ax = qx - px; ay = qy - py
    bx = sx - rx; by = sy - ry
    den = ax * by - ay * bx
    if abs(den) < TOL_CROSS:
        return False
    wx = rx - px; wy = ry - py
    u = (wx * by - wy * bx) / den
    v = (wx * ay - wy * ax) / den
    return TOL_CROSS < u < 1 - TOL_CROSS and TOL_CROSS < v < 1 - TOL_CROSS


@njit(cache=True)
def _bonds_cross_nb(P0, P1, Q0, Q1, R0, R1, S0, S1):
    lo0 = math.ceil(min(P0, Q0) - max(R0, S0) - 1e-12); hi0 = math.floor(max(P0, Q0) - min(R0, S0) + 1e-12)
    lo1 = math.ceil(min(P1, Q1) - max(R1, S1) - 1e-12); hi1 = math.floor(max(P1, Q1) - min(R1, S1) + 1e-12)
    for n1 in range(lo0, hi0 + 1):
        for n2 in range(lo1, hi1 + 1):
            if _seg_cross_nb(P0, P1, Q0, Q1, R0 + n1, R1 + n2, S0 + n1, S1 + n2):
                return True
    return False


@njit(cache=True)
def _site_on_bond_nb(F, P0, P1, Q0, Q1):
    vx = Q0 - P0; vy = Q1 - P1
    L2 = vx * vx + vy * vy
    for r in range(F.shape[0]):
        lo0 = math.ceil(min(P0, Q0) - F[r, 0] - 1e-9); hi0 = math.floor(max(P0, Q0) - F[r, 0] + 1e-9)
        lo1 = math.ceil(min(P1, Q1) - F[r, 1] - 1e-9); hi1 = math.floor(max(P1, Q1) - F[r, 1] + 1e-9)
        for n1 in range(lo0, hi0 + 1):
            for n2 in range(lo1, hi1 + 1):
                wx = F[r, 0] + n1 - P0; wy = F[r, 1] + n2 - P1
                s = (wx * vx + wy * vy) / L2
                if TOL_ON < s < 1 - TOL_ON and abs(wx * vy - wy * vx) <= TOL_ON * L2:
                    return True
    return False


@njit(cache=True)
def _sample_masks_nb(Fs, bi, bj, bn, ptr, valid, do_pairs):
    S = Fs.shape[0]
    K = ptr.shape[0] - 1
    E = bi.shape[0]
    W = (S + 63) // 64
    self_ok = np.zeros((K, W), np.uint64)
    pair_ok = np.zeros((K, K, W), np.uint64)
    P = np.empty((E, 2)); Q = np.empty((E, 2))
    good = np.empty(K, np.bool_)
    one = np.uint64(1)
    for s in range(S):
        if not valid[s]:
            continue
        F = Fs[s]
        for e in range(E):
            P[e, 0] = F[bi[e], 0]; P[e, 1] = F[bi[e], 1]
            Q[e, 0] = F[bj[e], 0] + bn[e, 0]; Q[e, 1] = F[bj[e], 1] + bn[e, 1]
        w = s // 64
        B = one << np.uint64(s % 64)
        for a in range(K):
            # G maps sites to sites and orbits to orbits, so it is enough to test ONE bond of orbit a
            e = ptr[a]
            good[a] = not _site_on_bond_nb(F, P[e, 0], P[e, 1], Q[e, 0], Q[e, 1])
        for a in range(K):
            for b in range(a, K if do_pairs else a + 1):
                # orbit a crosses orbit b iff one bond of the larger orbit crosses a bond of the other
                if ptr[a + 1] - ptr[a] >= ptr[b + 1] - ptr[b]:
                    r = a; o = b
                else:
                    r = b; o = a
                e = ptr[r]
                cr = False
                for f in range(ptr[o], ptr[o + 1]):
                    if _bonds_cross_nb(P[e, 0], P[e, 1], Q[e, 0], Q[e, 1], P[f, 0], P[f, 1], Q[f, 0], Q[f, 1]):
                        cr = True
                        break
                if not cr:
                    pair_ok[a, b, w] |= B
                    pair_ok[b, a, w] |= B
                    if a == b and good[a]:
                        self_ok[a, w] |= B
    return self_ok, pair_ok


@njit(cache=True)
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


@njit(cache=True)
def _connected_nb(m, ei, ej, en1, en2, ne):
    """Periodic connectivity of the graph with ne bonds (ei, ej, en1, en2)."""
    deg = np.zeros(m, np.int64)
    for e in range(ne):
        deg[ei[e]] += 1; deg[ej[e]] += 1
    start = np.zeros(m + 1, np.int64)
    for v in range(m):
        start[v + 1] = start[v] + deg[v]
    fill = start[:-1].copy()
    nbr = np.empty(2 * ne, np.int64); d1 = np.empty(2 * ne, np.int64); d2 = np.empty(2 * ne, np.int64)
    for e in range(ne):
        k = fill[ei[e]]; nbr[k] = ej[e]; d1[k] = en1[e]; d2[k] = en2[e]; fill[ei[e]] += 1
        k = fill[ej[e]]; nbr[k] = ei[e]; d1[k] = -en1[e]; d2[k] = -en2[e]; fill[ej[e]] += 1
    seen = np.zeros(m, np.bool_)
    o1 = np.zeros(m, np.int64); o2 = np.zeros(m, np.int64)
    stack = np.empty(m, np.int64)
    cy1 = np.empty(2 * ne, np.int64); cy2 = np.empty(2 * ne, np.int64)
    nc = 0
    sp = 0
    stack[0] = 0; sp = 1; seen[0] = True
    nseen = 1
    while sp > 0:
        sp -= 1
        u = stack[sp]
        for k in range(start[u], start[u + 1]):
            v = nbr[k]
            c1 = o1[u] + d1[k]; c2 = o2[u] + d2[k]
            if not seen[v]:
                seen[v] = True; o1[v] = c1; o2[v] = c2
                stack[sp] = v; sp += 1; nseen += 1
            else:
                x = c1 - o1[v]; y = c2 - o2[v]
                if x != 0 or y != 0:
                    cy1[nc] = x; cy2[nc] = y; nc += 1
    if nseen < m:
        return False
    g = 0
    for p in range(nc):
        for q in range(p + 1, nc):
            g = _gcd(g, abs(cy1[p] * cy2[q] - cy2[p] * cy1[q]))
            if g == 1:
                return True
    return g == 1


@njit(cache=True)
def _subsets_nb(n, pair_ok, self_ok, degv, nb, bi, bj, bn, ptr, min_degree, max_bonds, first):
    """Depth-first search over n-sets {first < k2 < ... < kn}: prune when no common planar sample
    is left, when the bond count exceeds max_bonds, or when some site can no longer reach min_degree."""
    K = degv.shape[0]; m = degv.shape[1]; W = self_ok.shape[1]
    maxc = np.zeros(m, np.int64)
    for k in range(K):
        for v in range(m):
            if degv[k, v] > maxc[v]:
                maxc[v] = degv[k, v]
    cap = 1024
    out = np.empty((cap, n), np.int64); outm = np.empty((cap, W), np.uint64)
    cnt = 0
    chosen = np.empty(n, np.int64)
    mask = np.empty((n + 1, W), np.uint64)
    deg = np.zeros((n + 1, m), np.int64)
    nbond = np.zeros(n + 1, np.int64)
    nxt = np.empty(n + 1, np.int64)
    ei = np.empty(bi.shape[0], np.int64); ej = np.empty(bi.shape[0], np.int64)
    en1 = np.empty(bi.shape[0], np.int64); en2 = np.empty(bi.shape[0], np.int64)
    # depth 1 = the forced first orbit
    any_bit = False
    for w in range(W):
        mask[1, w] = self_ok[first, w]
        if mask[1, w] != 0:
            any_bit = True
    if not any_bit or nb[first] > max_bonds:
        return out[:0], outm[:0]
    chosen[0] = first
    for v in range(m):
        deg[1, v] = degv[first, v]
    nbond[1] = nb[first]
    d = 1
    nxt[1] = first + 1
    while d >= 1:
        if d == n:
            ok = True
            for v in range(m):
                if deg[d, v] < min_degree:
                    ok = False
                    break
            if ok:
                ne = 0
                for t in range(n):
                    a = chosen[t]
                    for e in range(ptr[a], ptr[a + 1]):
                        ei[ne] = bi[e]; ej[ne] = bj[e]; en1[ne] = bn[e, 0]; en2[ne] = bn[e, 1]; ne += 1
                if _connected_nb(m, ei, ej, en1, en2, ne):
                    if cnt == cap:
                        cap *= 2
                        o2 = np.empty((cap, n), np.int64); o2[:cnt] = out[:cnt]; out = o2
                        m2 = np.empty((cap, W), np.uint64); m2[:cnt] = outm[:cnt]; outm = m2
                    for t in range(n):
                        out[cnt, t] = chosen[t]
                    for w in range(W):
                        outm[cnt, w] = mask[d, w]
                    cnt += 1
            d -= 1
            continue
        k = nxt[d]
        if k >= K or K - k < n - d:
            d -= 1
            continue
        nxt[d] = k + 1
        if nbond[d] + nb[k] > max_bonds:
            continue
        any_bit = False
        for w in range(W):
            x = mask[d, w] & self_ok[k, w]
            for t in range(d):
                x &= pair_ok[chosen[t], k, w]
            mask[d + 1, w] = x
            if x != 0:
                any_bit = True
        if not any_bit:
            continue
        left = n - d - 1
        ok = True
        for v in range(m):
            dv = deg[d, v] + degv[k, v]
            deg[d + 1, v] = dv
            if dv + left * maxc[v] < min_degree:
                ok = False
        if not ok:
            continue
        chosen[d] = k
        nbond[d + 1] = nbond[d] + nb[k]
        d += 1
        nxt[d] = k + 1
    return out[:cnt], outm[:cnt]


# ----------------------------------------------------------------------------------------------
# public entry points
# ----------------------------------------------------------------------------------------------
def _flat(orbits):
    E = sum(len(o) for o in orbits)
    bi = np.empty(E, np.int64); bj = np.empty(E, np.int64); bn = np.empty((E, 2), np.int64)
    ptr = np.zeros(len(orbits) + 1, np.int64)
    e = 0
    for a, o in enumerate(orbits):
        for (i, j, n1, n2) in o:
            bi[e], bj[e], bn[e, 0], bn[e, 1] = i, j, n1, n2; e += 1
        ptr[a + 1] = e
    return bi, bj, bn, ptr


def sample_masks(Fs, orbits, valid, use_numba=True, pairs=True):
    """pairs=False: only self_ok (and the diagonal of pair_ok) is filled."""
    bi, bj, bn, ptr = _flat(orbits)
    Fs = np.ascontiguousarray(Fs, float)
    valid = np.asarray(valid, np.bool_)
    if use_numba and HAVE_NUMBA:
        return _sample_masks_nb(Fs, bi, bj, bn, ptr, valid, bool(pairs))
    return sample_masks_py(Fs, bi, bj, bn, ptr, valid, pairs)


def subsets(n, orbits, degv, pair_ok, self_ok, min_degree, max_bonds, first, use_numba=True):
    """n-sets of orbit indices whose smallest element is `first` (see module docstring).
    Returns (list of tuples, array of sample masks)."""
    bi, bj, bn, ptr = _flat(orbits)
    nb = np.array([len(o) for o in orbits], np.int64)
    degv = np.ascontiguousarray(degv, np.int64)
    if use_numba and HAVE_NUMBA:
        out, masks = _subsets_nb(n, pair_ok, self_ok, degv, nb, bi, bj, bn, ptr, int(min_degree), int(max_bonds), int(first))
        return [tuple(int(x) for x in r) for r in out], masks
    out, masks = subsets_py(n, pair_ok, self_ok, degv, nb, bi, bj, bn, ptr, min_degree, max_bonds, first)
    return out, (np.array(masks, np.uint64) if masks else np.zeros((0, self_ok.shape[1]), np.uint64))


def periodic_connected(m, bonds, use_numba=True):
    if not use_numba or not HAVE_NUMBA:
        return periodic_connected_py(m, bonds)
    b = np.array(bonds, np.int64).reshape(-1, 4)
    return bool(_connected_nb(m, b[:, 0].copy(), b[:, 1].copy(), b[:, 2].copy(), b[:, 3].copy(), len(b)))
