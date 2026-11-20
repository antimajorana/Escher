"""The points rule in 3D: the simplest lattices of a space group G (C1 track T4, WP-P0 pilot).

The 2D rule of `points.py` lifted to 3D, with the bond rule the user chose on 2026-11-19 (plan Q1).

  lattice   a union of k orbits of G (Wyckoff positions with their free parameters, `_wyckoff3d`), one
            species, whose points alone have exactly the symmetry G with G's translations. A fixed point
            position is used at most once; line, plane and general positions may repeat.
  free      the position parameters (uniform in [0,1)) and the cell metric allowed by G's crystal system
            (`_wyckoff3d.system_metric`: a = 1, axial ratios log-uniform in [1/2, 2], free angles), sampled
            `samples` times (default 48, seed 0).
  exact     spglib (symprec 1e-5) on the primitive cell returns G's number AND exactly |G/T| operations
            (an extra translation would show as extra operations). The first 3 samples are tested before any
            bond work and the choice is dropped if none is exact; a choice is kept when more than half of its
            valid samples are exact.
  bonds     shells until connected (plan Q1): whole distance shells (ties included) in increasing distance;
            a shell that changes none of (components, rank, index) of the quotient graph is skipped; stop at
            (1, 3, 1): one connected 3-periodic net. The index (of the cycle lattice in the translations) is
            a clarification of the plan's "connected and 3-periodic": index k > 1 would be k interpenetrating
            nets. A sample is refused, with the reason counted, when no connected net is reached within `max_shells`
            (200) shells, when a third site lies on a bond, or when sites of one orbit get different degrees.
  phases    each distinct key (`_net3d_key.net_key_and_fingerprint`, plan Q10) met in at least 4 % of a
            choice's exact samples is a candidate entry; rarer ones are ignored (counted). The entry's
            geometry is the sample nearest the centroid of the phase's parameters.
  order     fewest orbits k first, then fewest sites per primitive cell; each tier is searched completely and
            the search stops at the tier that brings the list to n entries (all entries of that tier are kept).
  same      equal key = the same lattice (listed once, with the other Wyckoff choices under `also`). A key met
            with two different `fingerprint`s is recorded as a key collision.
Status: computational evidence (sampling): a phase missed by every sample is not found.
"""
from __future__ import annotations

import itertools
import time
from collections import Counter

import numpy as np

from . import _net3d_key as NK
from . import _wyckoff3d as W

SHELL_RTOL = 1e-7
ON_BOND_TOL = 1e-6


def configurations(wy, k):
    """Every multiset of k Wyckoff positions (indices into wy); a fixed point at most once. Sorted by sites per
    primitive cell, then by the indices (Wyckoff order)."""
    out = [c for c in itertools.combinations_with_replacement(range(len(wy)), k)
           if not any(wy[i]["kind"] == "point" and c.count(i) > 1 for i in set(c))]
    out.sort(key=lambda c: (sum(wy[i]["mult"] for i in c), c))
    return out


def to_primitive(number, X, A):
    """Conventional fractional points X (rows) and cell A (rows, Cartesian) -> primitive (F, A_prim, keep)."""
    B = np.array([[float(x) for x in row] for row in W.primitive_basis(W.group_entry(number))])
    F = (X @ np.linalg.inv(B)) % 1.0
    keep = W.unique_mod1(F)
    return F[keep] % 1.0, B @ A, keep


def sample_lattice(number, wy, config, rng):
    """One random member of a choice: dict(F, A, labels, params) in the primitive cell, or None when the
    sample is degenerate (an orbit smaller than its position's, or two sites closer than 1e-3)."""
    g = W.group_entry(number)
    A, free = W.system_metric(g["system"], rng)
    pts, labels, params = [], [], [np.log2(free[x]) for x in ("b", "c") if x in free]
    params += [free[x] / 90.0 for x in ("alpha", "beta", "gamma") if x in free]
    for o, i in enumerate(config):
        p, u = W.sample_position(wy[i], rng)
        orb = W.orbit(number, p)
        if len(orb) != wy[i]["mult_conv"]:
            return None
        pts.append(orb); labels += [o] * len(orb); params += list(u)
    X = np.vstack(pts)
    F, Ap, keep = to_primitive(number, X, A)
    labels = np.array(labels)[keep]
    m = sum(wy[i]["mult"] for i in config)
    if len(F) != m:
        return None
    D = F[:, None, :] - F[None, :, :]
    D -= np.round(D)
    dist = np.linalg.norm(D @ Ap, axis=2) + np.eye(m) * 9
    if m > 1 and dist.min() < 1e-3:
        return None
    return dict(F=F, A=Ap, labels=labels, params=np.array(params, float))


def exact_group(number, F, A):
    """Do the points alone have exactly the symmetry G with G's translations? (spglib, symprec 1e-5)"""
    import spglib
    g = W.group_entry(number)
    ds = spglib.get_symmetry_dataset((A, F, [1] * len(F)), symprec=1e-5)
    return ds is not None and ds.number == number and len(ds.rotations) == len(g["ops"])


def _pairs_within(F, A, r):
    """Every directed pair (i, j, n) with |F_j + n - F_i| <= r (Cartesian), complete for the cutoff r."""
    m = len(F)
    g = np.linalg.norm(np.linalg.inv(A), axis=0)              # |f_k| <= |x| * g_k
    R = [int(np.ceil(r * gk)) + 1 for gk in g]
    N = np.array(list(itertools.product(*[range(-x, x + 1) for x in R])), float)
    D = F[None, :, None, :] - F[:, None, None, :] + N[None, None, :, :]    # (i, j, n, 3)
    L = np.linalg.norm(D @ A, axis=3)
    I, J, K = np.nonzero((L <= r) & (L > 1e-9))
    return I, J, N[K].astype(int), L[I, J, K], (D @ A)[I, J, K]


def shell_bonds(F, A, labels, max_shells=200):
    """The shells-until-connected bonds (i, j, n), each once, or (None, reason)."""
    m = len(F)
    r = 1.6 * (abs(np.linalg.det(A)) / m) ** (1 / 3)
    while True:
        I, J, N, L, V = _pairs_within(F, A, r)
        once = (I < J) | ((I == J) & np.array([tuple(n) > (0, 0, 0) for n in N], bool))
        order = np.argsort(L[once], kind="stable")
        Io, Jo, No, Lo = I[once][order], J[once][order], N[once][order], L[once][order]
        edges, cur, used, k = [], (m, 0, 0), 0, 0
        while k < len(Lo):
            d0, k0 = Lo[k], k
            if d0 * (1 + SHELL_RTOL) > r:
                break                                        # this shell may run past the cutoff
            while k < len(Lo) and Lo[k] <= d0 * (1 + SHELL_RTOL):
                k += 1
            shell = [(int(Io[x]), int(Jo[x]), tuple(No[x])) for x in range(k0, k)]
            new = NK.state(m, edges + shell)
            used += 1
            if used > max_shells:
                return None, f"no connected 3-periodic net within {max_shells} shells"
            if new != cur:
                edges += shell
                cur = new
                if cur == (1, 3, 1):
                    return _validate(edges, I, J, N, V, labels, m)
        r *= 1.5


def _validate(edges, I, J, N, V, labels, m):
    """Refuse a net whose orbits get unequal degrees or that has a site on a bond (sites within the cutoff,
    which is longer than every bond, are checked)."""
    deg = NK.degrees(m, edges)
    for o in set(labels.tolist()):
        if len(set(deg[labels == o].tolist())) != 1:
            return None, "sites of one orbit have different degrees (tolerance failure)"
    vec = {(int(i), int(j), tuple(n)): v for i, j, n, v in zip(I, J, N.tolist(), V)}
    disp = {}
    for i, v in zip(I, V):
        disp.setdefault(int(i), []).append(v)
    disp = {i: np.array(v) for i, v in disp.items()}
    for i, j, n in edges:
        b = vec[(i, j, tuple(n))]
        W_ = disp[i]
        bb = b @ b
        t = W_ @ b / bb
        perp = np.linalg.norm(np.cross(W_, b), axis=1) / np.sqrt(bb)
        if np.any((t > 1e-9) & (t < 1 - 1e-9) & (perp < ON_BOND_TOL * np.sqrt(bb))):
            return None, "a third site lies on a bond"
    return edges, "ok"


def net_of(number, smp, depth=12):
    """(bonds, key, fingerprint) of one sample, or (None, reason, None)."""
    bonds, why = shell_bonds(smp["F"], smp["A"], smp["labels"])
    if bonds is None:
        return None, why, None
    return net_of_bonds(number, smp, bonds, depth)


def _bond_key(bonds):
    return frozenset(bonds)


def evaluate_choice(number, wy, config, samples, rng, timing):
    """Sample one choice of Wyckoff positions. Returns dict(status, valid, exact, phases, refused, keys)."""
    g = W.group_entry(number)
    _, free = W.system_metric(g["system"])
    ns = samples if (free or any(wy[i]["dim"] for i in config)) else 1
    smps, valid, exact = [], 0, []
    for s in range(ns):
        smp = sample_lattice(number, wy, config, rng)
        if smp is None:
            continue
        valid += 1
        t0 = time.perf_counter()
        ok = exact_group(number, smp["F"], smp["A"])
        timing["spglib"] += time.perf_counter() - t0; timing["n_spglib"] += 1
        if ok:
            exact.append(smp)
        if valid == 3 and not exact:
            return dict(status="more_symmetry_than_G", valid=valid, exact=0, early=True)
    if valid == 0:
        return dict(status="impossible", valid=0, exact=0)
    if 2 * len(exact) <= valid:
        return dict(status="more_symmetry_than_G", valid=valid, exact=len(exact), early=False)
    cache, by_key, refused = {}, {}, Counter()
    m = len(exact[0]["F"])
    for smp in exact:
        t0 = time.perf_counter()
        bonds, why = shell_bonds(smp["F"], smp["A"], smp["labels"])
        t1 = time.perf_counter()
        timing["bonds"] += t1 - t0; timing["n_bonds"] += 1; timing.setdefault("by_m", Counter())[m] += 1
        timing.setdefault("bonds_by_m", Counter())[m] += t1 - t0
        if bonds is None:
            refused[why] += 1
            continue
        bk = _bond_key(bonds)
        if bk not in cache:
            _, key, fp = net_of_bonds(number, smp, bonds)
            cache[bk] = (key, fp)
            timing["key"] += time.perf_counter() - t1; timing["n_key"] += 1
        key, fp = cache[bk]
        d = by_key.setdefault(key, dict(count=0, samples=[], fps=set()))
        d["count"] += 1; d["samples"].append((smp, bonds)); d["fps"].add(fp)
    nets = sum(d["count"] for d in by_key.values())
    phases, rare = [], 0
    for key, d in sorted(by_key.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
        if d["count"] < 0.04 * len(exact):
            rare += 1
            continue
        P = np.array([s["params"] for s, _ in d["samples"]])
        best = int(np.argmin(np.linalg.norm(P - P.mean(axis=0), axis=1))) if P.size else 0
        smp, bonds = d["samples"][best]
        phases.append(dict(key=key, count=d["count"], share=round(d["count"] / len(exact), 3), sample=smp,
                           bonds=bonds, fingerprints=d["fps"]))
    return dict(status="ok" if phases else "no_net", valid=valid, exact=len(exact), nets=nets, phases=phases,
                refused=dict(refused), rare_phases=rare)


def net_of_bonds(number, smp, bonds, depth=12):
    adj = NK.adjacency(bonds)
    lab = smp["labels"]
    orbs = sorted(set(lab.tolist()))
    reps = [int(np.nonzero(lab == o)[0][0]) for o in orbs]
    sizes = [int((lab == o).sum()) for o in orbs]
    key, fp = NK.net_key_and_fingerprint(number, len(lab), {i: orbs.index(int(o)) for i, o in enumerate(lab)},
                                         reps, sizes, adj, depth)
    return bonds, key, fp


def points3d_lattices(number, n=7, samples=48, seed=0, time_limit=280.0, max_orbits=4):
    """The first n lattices of space group `number` under the 3D points rule (module docstring).
    Returns a dict: group, symbol, entries (each: wyckoff, key, degrees, cs, share, sample geometry, also),
    tiers (k, m, status, counts, choices, seconds), collisions, timing, seconds."""
    g = W.group_entry(number)
    wy = W.wyckoff_positions(number)
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()
    timing = Counter()
    entries, by_key, tiers, collisions, done = [], {}, [], [], False
    for k in range(1, max_orbits + 1):
        confs = configurations(wy, k)
        for m in sorted({sum(wy[i]["mult"] for i in c) for c in confs}):
            tier = [c for c in confs if sum(wy[i]["mult"] for i in c) == m]
            cnt, status, t_tier = Counter(), "complete", time.perf_counter()
            for config in tier:
                if time.perf_counter() - t_start > time_limit:
                    status = "OPEN (time limit)"; break
                r = evaluate_choice(number, wy, config, samples, rng, timing)
                cnt["choices"] += 1
                if r["status"] != "ok":
                    cnt[r["status"] + ("_early" if r.get("early") else "")] += 1
                    continue
                cnt["rare_phases"] += r["rare_phases"]
                for why, c in r["refused"].items():
                    cnt["refused: " + why] += c
                names = [wy[i]["name"] for i in config]
                for ph in r["phases"]:
                    key = ph["key"]
                    if key in by_key:
                        e = entries[by_key[key]]
                        if names not in e["also"] and names != e["wyckoff"]:
                            e["also"].append(names)
                        if not ph["fingerprints"] <= e["_fps"]:
                            collisions.append(dict(key=key, wyckoff=[e["wyckoff"], names]))
                            e["_fps"] |= ph["fingerprints"]
                        cnt["same_lattice"] += 1
                        continue
                    by_key[key] = len(entries)
                    entries.append(_entry(number, wy, config, ph))
                    cnt["new"] += 1
            tiers.append(dict(k=k, m=m, status=status, choices=len(tier), counts=dict(cnt),
                              seconds=round(time.perf_counter() - t_tier, 3)))
            if status != "complete" or len(entries) >= n:
                done = True; break
        if done:
            break
    for e in entries:
        if len(e.pop("_fps")) > 1:
            collisions.append(dict(key=e["key"], wyckoff=[e["wyckoff"]], note="one phase, several fingerprints"))
    return dict(group=number, symbol=g["symbol"], n=n, samples=samples, seed=seed, entries=entries, tiers=tiers,
                collisions=collisions, timing=dict(timing), seconds=round(time.perf_counter() - t_start, 3))


def _entry(number, wy, config, ph):
    smp, bonds, key = ph["sample"], ph["bonds"], ph["key"]
    deg = NK.degrees(len(smp["F"]), bonds)
    return dict(wyckoff=[wy[i]["name"] for i in config], key=key, sites=len(smp["F"]),
                degrees=sorted(set(deg.tolist())), cs=[list(c) for _, c in key[2]], share=ph["share"],
                count=ph["count"], A=smp["A"].tolist(), F=smp["F"].tolist(), labels=smp["labels"].tolist(),
                bonds=[[i, j, list(n)] for i, j, n in bonds], also=[], _fps=set(ph["fingerprints"]))
