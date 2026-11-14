"""numba port of papertilling.dsymbol (the pure-Python reference): canonical D-symbol keys of one
structure (bond set) at many planar samples, without building UnitCells.

Same definitions as dsymbol.py, line by line (darts 2e = (i, j, n), 2e + 1 = reverse; rotation =
counter-clockwise order of the darts at their tail; faces by d -> prev_ccw(rev d); chambers 2d + s;
sigma0, sigma1, sigma2, m01 = face size, m12 = vertex degree; canonical code = lexicographically
smallest breadth-first code over all start elements). The cyclic order is taken in fractional
coordinates: it equals the Cartesian one for any orientation-preserving cell (proved result: sign of a
2x2 determinant times det A > 0). tests/test_enumerate.py checks key equality with dsymbol.py
(computational evidence).

keys_of_structure(...) returns (keys, n_keys, which, status):
  keys[k]   canonical code (size x 5 int array) of the k-th distinct tiling met,
  which[p]  index of the key of planar sample p (-1 if the D-symbol is undefined there),
  status    0 ok, 1 an operation does not map the bonds onto bonds, 2 more than 8 tilings
            (the caller then uses the Python reference). which[p] = -1: D-symbol undefined there.
"""
from __future__ import annotations

import math

import numpy as np

from .kernels import njit, HAVE_NUMBA


@njit(cache=True)
def _rotation(F, di, dj, dn1, dn2, m):
    D = di.shape[0]
    ang = np.empty(D)
    for d in range(D):
        ang[d] = math.atan2(F[dj[d], 1] + dn2[d] - F[di[d], 1], F[dj[d], 0] + dn1[d] - F[di[d], 0])
    order = np.argsort(di * 8.0 + ang + np.pi)        # grouped by tail, then by angle
    nxt = np.empty(D, np.int64); prv = np.empty(D, np.int64)
    deg = np.zeros(m, np.int64)
    p = 0
    while p < D:
        q = p
        while q < D and di[order[q]] == di[order[p]]:
            q += 1
        for r in range(p, q):
            a = order[r]; b = order[r + 1] if r + 1 < q else order[p]
            nxt[a] = b; prv[b] = a
        deg[di[order[p]]] = q - p
        p = q
    return nxt, prv, deg


@njit(cache=True)
def _dsymbol_code(nxt, prv, deg, di, dj, dn1, dn2, m, gperm, flip):
    """Canonical code (n x 5) or an empty array if the D-symbol is undefined (faces not disks,
    Euler != 0, group not free)."""
    D = di.shape[0]
    rev = np.empty(D, np.int64)
    for d in range(D):
        rev[d] = d ^ 1
    face_of = -np.ones(D, np.int64)
    fsize = np.zeros(D, np.int64)
    nf = 0
    for d0 in range(D):
        if face_of[d0] >= 0:
            continue
        s1 = 0; s2 = 0; L = 0
        d = d0
        while face_of[d] < 0:
            face_of[d] = nf; s1 += dn1[d]; s2 += dn2[d]; L += 1
            d = prv[rev[d]]
        if d != d0 or s1 != 0 or s2 != 0:
            return np.empty((0, 5), np.int64)
        fsize[nf] = L
        nf += 1
    V = 0
    for v in range(m):
        if deg[v] > 0:
            V += 1
    if V != m or V - D // 2 + nf != 0:
        return np.empty((0, 5), np.int64)
    G = gperm.shape[0]
    nch = 2 * D
    orbit = -np.ones(nch, np.int64)
    reps = np.empty(nch, np.int64)
    n = 0
    for c in range(nch):
        if orbit[c] >= 0:
            continue
        d = c // 2; s = c % 2
        for g in range(G):
            img = gperm[g, d]
            x = 2 * img + ((1 - s) if flip[g] else s)
            if orbit[x] >= 0:
                return np.empty((0, 5), np.int64)       # not free (or inconsistent)
            orbit[x] = n
        reps[n] = c
        n += 1
    sig = np.empty((3, n), np.int64)
    m01 = np.empty(n, np.int64); m12 = np.empty(n, np.int64)
    for o in range(n):
        c = reps[o]; d = c // 2; s = c % 2
        sig[0, o] = orbit[2 * rev[d] + (1 - s)]
        sig[1, o] = orbit[2 * nxt[d] + 1] if s == 0 else orbit[2 * prv[d]]
        sig[2, o] = orbit[c ^ 1]
        m01[o] = fsize[face_of[d]] if s == 0 else fsize[face_of[rev[d]]]
        m12[o] = deg[di[d]]
    best = np.empty((n, 5), np.int64)
    have = False
    code = np.empty((n, 5), np.int64)
    num = np.empty(n, np.int64); order = np.empty(n, np.int64)
    for start in range(n):
        for x in range(n):
            num[x] = -1
        num[start] = 0; order[0] = start; cnt = 1; k = 0
        state = 0                                     # 0 equal so far, 1 smaller, 2 larger
        while k < cnt:
            x = order[k]
            code[k, 0] = m01[x]; code[k, 1] = m12[x]
            for i in range(3):
                y = sig[i, x]
                if num[y] < 0:
                    num[y] = cnt; order[cnt] = y; cnt += 1
                code[k, 2 + i] = num[y]
            if have and state == 0:
                for c5 in range(5):
                    if code[k, c5] < best[k, c5]:
                        state = 1; break
                    if code[k, c5] > best[k, c5]:
                        state = 2; break
            if state == 2:
                break
            k += 1
        if state == 2:
            continue
        if cnt != n:
            return np.empty((0, 5), np.int64)       # D-set not connected
        if not have or state == 1:
            best[:, :] = code
            have = True
    return best


@njit(cache=True)
def keys_of_structure(sub, mask, Fs, bi, bj, bn, ptr, m, perm, shift, Mf, flip):
    """All distinct keys of the bond set `sub` (orbit indices) over the planar samples in `mask`.
    perm[g, k], shift[g, k, :] : action of G's operation g on site labels; Mf[g] its integer matrix."""
    E = 0
    for t in range(sub.shape[0]):
        E += ptr[sub[t] + 1] - ptr[sub[t]]
    D = 2 * E
    di = np.empty(D, np.int64); dj = np.empty(D, np.int64); dn1 = np.empty(D, np.int64); dn2 = np.empty(D, np.int64)
    e = 0
    for t in range(sub.shape[0]):
        a = sub[t]
        for f in range(ptr[a], ptr[a + 1]):
            di[2 * e] = bi[f]; dj[2 * e] = bj[f]; dn1[2 * e] = bn[f, 0]; dn2[2 * e] = bn[f, 1]
            di[2 * e + 1] = bj[f]; dj[2 * e + 1] = bi[f]; dn1[2 * e + 1] = -bn[f, 0]; dn2[2 * e + 1] = -bn[f, 1]
            e += 1
    # group action on darts (by search; D is small)
    G = perm.shape[0]
    gperm = np.empty((G, D), np.int64)
    bad = False
    for g in range(G):
        for d in range(D):
            i = di[d]; j = dj[d]
            pi = perm[g, i]; pj = perm[g, j]
            x1 = Mf[g, 0, 0] * dn1[d] + Mf[g, 0, 1] * dn2[d] + shift[g, j, 0] - shift[g, i, 0]
            x2 = Mf[g, 1, 0] * dn1[d] + Mf[g, 1, 1] * dn2[d] + shift[g, j, 1] - shift[g, i, 1]
            hit = -1
            for d2 in range(D):
                if di[d2] == pi and dj[d2] == pj and dn1[d2] == x1 and dn2[d2] == x2:
                    hit = d2; break
            if hit < 0:
                bad = True
            gperm[g, d] = hit
    W = mask.shape[0]
    S = Fs.shape[0]
    which = -np.ones(S, np.int64)
    nk = 0
    keys = np.empty((8, 2 * D, 5), np.int64)          # at most 8 distinct tilings; size <= 2D chambers
    klen = 0
    seen_nxt = np.empty((8, D), np.int64)
    seen_key = np.empty(8, np.int64)
    nseen = 0
    if bad:
        return keys[:0, :0], 0, which, 1
    for s in range(S):
        w = s // 64
        if w >= W or (mask[w] >> np.uint64(s % 64)) & np.uint64(1) == 0:
            continue
        nxt, prv, deg = _rotation(Fs[s], di, dj, dn1, dn2, m)
        hit = -1
        for q in range(nseen):
            same = True
            for d in range(D):
                if seen_nxt[q, d] != nxt[d]:
                    same = False; break
            if same:
                hit = q; break
        if hit >= 0:
            which[s] = seen_key[hit]
            continue
        code = _dsymbol_code(nxt, prv, deg, di, dj, dn1, dn2, m, gperm, flip)
        kid = -1
        if code.shape[0] > 0:
            klen = code.shape[0]
            for q in range(nk):
                same = True
                for r in range(klen):
                    for c5 in range(5):
                        if keys[q, r, c5] != code[r, c5]:
                            same = False; break
                    if not same:
                        break
                if same:
                    kid = q; break
            if kid < 0:
                if nk == 8:
                    return keys[:nk, :klen], nk, which, 2   # too many tilings: caller falls back
                keys[nk, :klen] = code; kid = nk; nk += 1
        if nseen < 8:
            seen_nxt[nseen] = nxt; seen_key[nseen] = kid; nseen += 1
        which[s] = kid
    return keys[:nk, :klen], nk, which, 0


def action_arrays(base):
    """perm (G, m), shift (G, m, 2), Mf (G, 2, 2), flip (G) of the configuration's base Family."""
    perm, shift = base._action()
    P = np.array(perm, np.int64)
    Sh = np.rint(np.array(shift, float)).astype(np.int64).reshape(len(perm), -1, 2)      # rint: int() would truncate 0.9999999
    M = np.rint(np.array([np.asarray(Mf, float) for Mf, _ in base._ops])).astype(np.int64)
    if max(np.abs(np.array(shift, float).reshape(Sh.shape) - Sh).max(initial=0.0), max(np.abs(np.asarray(Mf, float) - M[g]).max() for g, (Mf, _) in enumerate(base._ops))) > 1e-9:
        raise ValueError("group operations are not integer matrices / shifts in the primitive basis")
    flip = np.array([int(round(np.linalg.det(Mf))) < 0 for Mf, _ in base._ops], np.bool_)
    return P, Sh, M, flip


_KEYS = {}


def key_to_tuple(arr):
    """Canonical code array -> the tuple-of-tuples key of dsymbol.py (memoised on the raw bytes)."""
    b = arr.tobytes() + bytes([arr.shape[0] % 256])
    k = _KEYS.get(b)
    if k is None:
        k = tuple(map(tuple, arr.tolist()))
        _KEYS[b] = k
    return k
