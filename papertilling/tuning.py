"""Where can one free parameter of a lattice move, and where does its symmetry jump? (slider support)

    from papertilling.tuning import allowed_interval, special_points
    allowed_interval(L, "t1", -0.8, 0.2)   # {'interval': (a, b), 'edges': ..., 'segments': [...], ...}
    special_points(L, "t1", -0.8, 0.2)     # [{'value': -0.5, 'group': 'p4m', 'equations': ['t1 = -1/2'], ...}]

Every other parameter is held at its current value (or at `values`).

allowed_interval
  DEFINITION (the author's, 2026-11-14: "forbid the regions where parameters cross"): a value is allowed iff
  no two sites coincide, no bond passes through a site, no two bonds cross, and the rotation system (cyclic
  order of the bonds round every site) equals the one at the current value. Equal rotation systems give
  equal D-symbols (enumerate.rotation_signatures), so the tiling (rule B) is unchanged on the interval.
  PROVED RESULT (kernels.py docstring): crossing, site-on-bond and the cyclic order are signs of 2x2
  determinants, unchanged by the orientation-preserving map fractional -> Cartesian; so the cell parameters
  b, gamma never make a lattice invalid: their interval is the whole range (they only have special values).
  METHOD (technical choices, mine): the site coordinates are affine in the site parameters,
  F = q + P u (Family._affine_sites); a grid of n values of `name` (plus the current value) is tested at once
  with the search kernels (kernels.sample_masks on the family's bond orbits; numba if present, else the
  pure-Python reference), then every valid/invalid transition is refined by two more vectorised scans of
  REFINE points inside its grid step (resolution (hi - lo) / (n - 1) / REFINE^2 ~ 1e-6). The interval
  returned is the connected valid run containing the current value, inset by `margin`, so both ends were
  tested valid. COMPUTATIONAL EVIDENCE only: a forbidden window narrower than one grid step can be missed.

special_points
  The values of `name` where a supergroup locus of the family (Family.supergroup_loci: affine equations
  C v = c in v = (b cos gamma, b^2 in units of a, site parameters)) is crossed. Along the line every row
  is a polynomial of degree <= 2 in t = the value (site parameter), cos(gamma) (gamma) or b/a (b), so the
  roots are solved exactly (no sampling); a root is kept if every row is < 1e-7 there (Locus.contains). A locus that the line never reaches gives nothing; a locus containing the whole
  line (the parameter does not enter its equations) is not listed either (the live group check shows it).
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import kernels as KN
from .enumerate import rotation_signatures

REFINE = 33            # points per refinement scan of one grid step (technical choice)
ROUNDS = 2             # refinement scans per transition
TOL_LOCUS = 1e-7       # Locus.contains default
REASONS = {1: "two sites coincide", 2: "bonds cross / a bond passes through a site", 3: "other tiling (rotation system)"}


def _family(lattice):
    return getattr(lattice, "family", lattice)


def _values(fam, values):
    v = dict(fam.values)
    if values:
        v.update(values)
    return v


def _site_vector(fam, values):
    return np.array([float(values[k]) for k in fam.site_names], float)


def classify(fam, U, ref_sig=None, use_numba=True):
    """Reason code per row of U (site-parameter vectors, shape (S, d)): 0 valid, 1 two sites coincide,
    2 bonds cross or a site on a bond, 3 rotation system differs from ref_sig. Also returns the signatures."""
    P, q = fam._affine_sites()
    U = np.atleast_2d(np.asarray(U, float))
    Fs = q[None, :, :] + np.einsum("kcd,sd->skc", P, U)
    S, m = len(U), fam.m
    code = np.zeros(S, np.int64)
    sep_ok = np.ones(S, bool)
    for i in range(m):                                   # as enumerate.ConfigData.__init__
        for j in range(i):
            D = Fs[:, i] - Fs[:, j]
            sep_ok &= np.max(np.abs(D - np.round(D)), axis=1) > 1e-5
    code[~sep_ok] = 1
    self_ok, pair_ok = KN.sample_masks(Fs, fam.orbits, sep_ok, use_numba and KN.HAVE_NUMBA, pairs=True)
    good = np.bitwise_and.reduce(self_ok, axis=0)
    K = len(fam.orbits)
    for a in range(K):
        for b in range(a + 1, K):
            good &= pair_ok[a, b]
    ok = np.unpackbits(np.ascontiguousarray(good).view(np.uint8), bitorder="little")[:S].astype(bool)
    code[(code == 0) & ~ok] = 2
    sig = rotation_signatures(Fs, fam.bonds())
    if ref_sig is not None:
        code[(code == 0) & np.any(sig != ref_sig[None, :], axis=1)] = 3
    return code, sig


def _rows(fam, values, col, xs):
    u = _site_vector(fam, values)
    U = np.repeat(u[None, :], len(xs), axis=0)
    U[:, col] = xs
    return U


def _refine(fam, values, col, brackets, ref_sig, use_numba):
    """brackets: list of (x_good, x_bad). Returns refined (x_good, x_bad) pairs (x_good tested valid)."""
    br = [tuple(map(float, b)) for b in brackets]
    if not br:
        return []
    t = np.linspace(0.0, 1.0, REFINE)
    for _ in range(ROUNDS):
        xs = np.concatenate([g + (b - g) * t for g, b in br])
        code, _ = classify(fam, _rows(fam, values, col, xs), ref_sig, use_numba)
        new = []
        for k, (g, b) in enumerate(br):
            c = code[k * REFINE:(k + 1) * REFINE]
            x = xs[k * REFINE:(k + 1) * REFINE]
            bad = np.nonzero(c != 0)[0]
            j = int(bad[0]) if len(bad) else REFINE - 1       # first invalid from the good side
            j = max(j, 1)
            new.append((float(x[j - 1]), float(x[j])))
        br = new
    return br


def allowed_interval(lattice, name, lo, hi, n=401, values=None, use_numba=True, margin=None):
    """dict(name, lo, hi, interval=(a, b), edges=(ea, eb), segments=[dict(lo, hi, reason)], current_ok, kind).
    interval: the valid run containing the current value, inset by margin (default 1e-4 (hi - lo)), always
    containing the current value; edges: the refined boundary estimates (None where the run reaches the end
    of [lo, hi]); segments: the forbidden pieces of [lo, hi]. If the current value itself is not valid,
    current_ok is False and interval is (lo, hi)."""
    fam = _family(lattice)
    vals = _values(fam, values)
    x0 = float(vals[name])
    lo, hi = min(float(lo), x0), max(float(hi), x0)
    out = dict(name=name, lo=lo, hi=hi, interval=(lo, hi), edges=(None, None), segments=[], current_ok=True)
    if name not in fam.site_names:
        out["kind"] = "cell"                              # proved: no forbidden region (module docstring)
        return out
    out["kind"] = "site"
    margin = 1e-4 * (hi - lo) if margin is None else margin
    col = fam.site_names.index(name)
    xs = np.unique(np.concatenate([np.linspace(lo, hi, n), [x0]]))
    i0 = int(np.searchsorted(xs, x0))
    code0, sig0 = classify(fam, _rows(fam, vals, col, [x0]), None, use_numba)
    if code0[0] != 0:
        out["current_ok"] = False
        ref = None
    else:
        ref = sig0[0]
    code, _ = classify(fam, _rows(fam, vals, col, xs), ref, use_numba)
    valid = code == 0
    trans = [k for k in range(len(xs) - 1) if valid[k] != valid[k + 1]]
    brackets = [(xs[k], xs[k + 1]) if valid[k] else (xs[k + 1], xs[k]) for k in trans]
    refined = dict(zip(trans, _refine(fam, vals, col, brackets, ref, use_numba)))
    edge = {k: 0.5 * (g + b) for k, (g, b) in refined.items()}
    # forbidden segments of [lo, hi]
    k = 0
    while k < len(xs):
        if valid[k]:
            k += 1; continue
        j = k
        while j + 1 < len(xs) and not valid[j + 1]:
            j += 1
        reasons = sorted({int(c) for c in code[k:j + 1]})
        out["segments"].append(dict(lo=edge.get(k - 1, lo) if k > 0 else lo, hi=edge.get(j, hi) if j < len(xs) - 1 else hi,
                                    reason="; ".join(REASONS[r] for r in reasons)))
        k = j + 1
    if not out["current_ok"]:
        return out
    kl = i0
    while kl > 0 and valid[kl - 1]:
        kl -= 1
    kr = i0
    while kr < len(xs) - 1 and valid[kr + 1]:
        kr += 1
    a, ea = (lo, None) if kl == 0 else (refined[kl - 1][0], edge[kl - 1])
    b, eb = (hi, None) if kr == len(xs) - 1 else (refined[kr][0], edge[kr])
    if ea is not None:
        a = min(a + margin, x0)
    if eb is not None:
        b = max(b - margin, x0)
    out["interval"], out["edges"] = (float(a), float(b)), (ea, eb)
    return out


def _v_of_t(fam, values, name, t, base):
    """v (Family._v) as a function of one scalar t, chosen so that v is POLYNOMIAL (degree <= 2) in t:
    t = the value for a site parameter (affine), t = cos(gamma) for gamma (v[0] = (b/a) t, affine),
    t = b/a for b (v = (t cos gamma, t^2), quadratic). Returns rows v(t) for the array t."""
    t = np.atleast_1d(np.asarray(t, float))
    V = np.repeat(base[None, :], len(t), axis=0)
    ny = len(fam._metric_vars())
    if name in fam.site_names:
        V[:, ny + fam.site_names.index(name)] = t
    elif name == "gamma":
        V[:, 0] = float(values["b"]) / fam.a * t
    else:                                               # b
        if "gamma" in fam.cellpar:
            V[:, 0] = t * math.cos(math.radians(float(values["gamma"])))
        V[:, ny - 1] = t * t
    return V


def _t_to_x(fam, name, t):
    if name in fam.site_names:
        return t
    if name == "gamma":
        return math.degrees(math.acos(t)) if -1 < t < 1 else None
    return t * fam.a if t > 0 else None


def special_points(lattice, name, lo, hi, values=None, loci=None):
    """[dict(value, group, equations, dim)] sorted by value; where several loci are crossed at one value
    the one with the fewest free parameters left (largest group) is kept. Exact: each locus row is a
    polynomial of degree <= 2 in t (_v_of_t), its coefficients are read off from the values at t = -1, 0, 1,
    and the candidate roots (numpy.roots of one non-constant row) are kept where EVERY row is < 1e-7."""
    fam = _family(lattice)
    vals = _values(fam, values)
    if loci is None:
        loci = fam.supergroup_loci()
    lo, hi = float(lo), float(hi)
    found = []
    base = fam._v(vals)
    for L in loci:
        C, c = L.aug[:, :-1], L.aug[:, -1]
        res = lambda t: _v_of_t(fam, vals, name, t, base) @ C.T - c              # (len(t), rows)
        Rm, R0, Rp = res([-1.0, 0.0, 1.0])
        A, B = 0.5 * (Rp + Rm) - R0, 0.5 * (Rp - Rm)      # row(t) = A t^2 + B t + R0, exactly
        live = (np.abs(A) > 1e-12) | (np.abs(B) > 1e-12)
        if not live.any() or np.any(np.abs(R0[~live]) >= TOL_LOCUS):
            continue                                      # never reached on this line (or the whole line)
        k = int(np.argmax(np.abs(A) + np.abs(B)))
        roots = []
        for t in np.roots([A[k], B[k], R0[k]] if abs(A[k]) > 1e-12 else [B[k], R0[k]]):
            if abs(t.imag) > 1e-6:
                continue
            t = float(t.real)
            if np.max(np.abs(res([t])[0])) < TOL_LOCUS:
                x = _t_to_x(fam, name, t)
                if x is not None:
                    roots.append(x)
        for x in roots:
            if lo - 1e-12 <= x <= hi + 1e-12:
                found.append(dict(value=x, group=L.group, equations=L.equations(), dim=L.dim))
    found.sort(key=lambda d: (d["value"], d["dim"]))
    out = []
    for d in found:
        if out and abs(d["value"] - out[-1]["value"]) < 1e-6:
            if d["dim"] < out[-1]["dim"]:
                out[-1] = d
            continue
        out.append(d)
    return out


if __name__ == "__main__":                                # pilot: p4g #1 and p1 #1
    from .lattice import load_catalogue
    for grp in ("p4g", "p1"):
        L = load_catalogue(group=grp).lattices[0]
        f = L.family
        loci = f.supergroup_loci()
        t = time.time()
        classify(f, [[f.values[k] for k in f.site_names]])            # numba: load / compile the kernels
        print(f"# time {grp} warm-up (numba kernels): {time.time() - t:.3f} s")
        for numba in (True, False):
            t = time.time()
            for nm in f.site_names:
                x = f.values[nm]
                r = allowed_interval(L, nm, x - 0.5, x + 0.5, n=401, use_numba=numba)
                sp = special_points(L, nm, x - 0.5, x + 0.5, loci=loci)
                if numba:
                    print(grp, nm, "interval", np.round(r["interval"], 6), "segments",
                          [(round(s["lo"], 5), round(s["hi"], 5), s["reason"]) for s in r["segments"]],
                          "special", [(round(d["value"], 8), d["group"]) for d in sp])
            print(f"# time {grp} all site parameters, numba={numba and KN.HAVE_NUMBA}: {time.time() - t:.3f} s")
