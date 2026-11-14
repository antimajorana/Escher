"""Exhaustive search for the "most Bravais-like" lattices of a wallpaper group (ranking b).

DEFINITIONS (set by the author, 2026-11-13, see docs/DEVELOPMENT.md; min_degree = 3 confirmed by the author on
  2026-11-14: "Keep 3 bonds")
  lattice   planar, periodic-connected, exactly G with G's translations (bond colours ignored), no bond
            through a site, every site >= min_degree bonds (3 by default).
  same      (author, 2026-11-14) the same NETWORK: equal dsymbol.network_key (the tiling with its symmetry forgotten).
            Each network is listed once, by its simplest drawing (lowest rank). same="action" gives the earlier
            rule B (equal D-symbol key under G: same network AND G acting the same way), under which one
            network drawn with one or with two sites per cell counted twice.
  rank      (site orbits k, bond orbits n, sites per primitive cell, bonds per primitive cell).

WHAT IS ENUMERATED (technical choices, mine; all stated in the result's coverage text)
  tiers     (k, n) in lexicographic order, n = 1 .. max_bond_orbits for each k.
  sites     every multiset of k Wyckoff families of G (legacy minimal_lattices.wyckoff_positions):
            a rotation-centre point at most once, a line or the general position any number of times.
  bonds     candidate bond orbits = orbits containing a bond whose displacement, in primitive cell
            units, can be at most `reach` in each direction as the generating points run over one
            period (the rule of legacy enumerate_families). Every n-set of candidates is examined.
  planar    tested at `samples` random values of the site parameters u, each in (-1, 1) (two periods,
            as legacy Family.random_values), the SAME samples for every n-set of one configuration:
            an n-set is kept iff at one sample no two of its bonds cross, no site lies on one of its
            bonds and no two sites coincide. Crossing, site-on-bond and the cyclic order of bonds at a
            site are invariant under orientation-preserving linear maps (proved result), so the cell
            parameters b, gamma play no role in planarity or in the tiling, only in the group.
            Sampling: "no planar sample" is not a proof that no planar member exists.
  exact     the number of symmetry operations (modulo the primitive translations of G) of the bare
            lattice at random cell parameters and a planar sample equals |G / T|. Since G is a
            subgroup of the lattice's group by construction, equal counts mean equal groups.
            (computational evidence at one random point per new key; final entries are rechecked by
            legacy Family.group_at and by spglib through the layer route spg_group.)
PROVED BOUNDS used to skip work (each is an inequality every lattice in a tier satisfies):
  * connectivity: the quotient graph (site orbits, bond orbits) is connected, so n >= k - 1.
  * degree: min_degree * (sites per cell) <= 2 (bonds per cell) <= 2 n |G/T|.
  * planarity on the torus: V - E + F = 0 and 3F <= 2E give bonds per cell <= 3 sites per cell.
  RANK GAP (reported, not searched): for k below the stopping tier, a configuration whose n smallest
  candidate orbits fit under the planarity bound (bonds <= 3 sites) for some n > max_bond_orbits could
  hold lattices in tiers (k, n > max_bond_orbits), which rank BEFORE the stopping tier; these are listed
  in result["gaps"] with the largest such n.
  * adding a bond orbit only adds crossings, so a set with no common planar sample has no planar superset
    among the samples (the depth-first search is cut there).
"""
from __future__ import annotations

import itertools
import time
from collections import Counter, defaultdict

import numpy as np

from .dsymbol import dsymbol, DSymbolError, network_key
from .legacy import minimal_lattices as ML
from .legacy.lattice_families import Family, _REF_T, _REF_XY
from .legacy.wallpaper_lattices import WallpaperGroup, UnitCell, _canon_bond, _symmetry_ops
from . import kernels as KN
from . import kernels_dsymbol as KD


# ----------------------------------------------------------------------------------------------
# configurations of k site orbits
# ----------------------------------------------------------------------------------------------
def configurations(wy, k):
    """Every multiset of k Wyckoff families (indices into wy); points at most once. Sorted by sites per
    primitive cell, then by the indices."""
    out = []
    for c in itertools.combinations_with_replacement(range(len(wy)), k):
        if any(wy[i]["kind"] == "point" and c.count(i) > 1 for i in set(c)):
            continue
        out.append(c)
    out.sort(key=lambda c: (sum(wy[i]["mult"] for i in c), c))
    return out


def _ref_points(wy, config):
    pts, nl, ng = [], 0, 0
    for k in config:
        w = wy[k]
        if w["kind"] == "point":
            pts.append(np.array(w["rep"], float))
        elif w["kind"] == "line":
            pts.append(np.asarray(w["p0"], float) + _REF_T[nl] * np.asarray(w["d"], float)); nl += 1
        else:
            pts.append(np.array(_REF_XY[ng])); ng += 1
    return pts


def candidate_orbits(base, reach):
    """Bond orbits with a bond that can be drawn at most `reach` primitive cells long in each direction
    as the generating points run over one period (legacy enumerate_families rule, copied)."""
    m = base.m
    R = np.arange(-reach - 3, reach + 4)
    NN = np.array([(a, b) for a in R for b in R], float)
    seen, orbits = set(), []
    P, q = base._affine_sites()
    for i in range(m):
        for j in range(i, m):
            coef = P[j] - P[i]
            lo = np.minimum(coef, 0).sum(axis=1); hi = np.maximum(coef, 0).sum(axis=1)
            const = q[j] - q[i] + NN
            ok = ~(np.any(const + lo > reach + 1e-9, axis=1) | np.any(const + hi < -reach - 1e-9, axis=1))
            for n in NN[ok].astype(int):
                if i == j and not n.any():
                    continue
                bd = _canon_bond(i, j, int(n[0]), int(n[1]), 0)[:4]
                if bd in seen:
                    continue
                orb = base._orbit_of(bd)
                seen.update(orb)
                orbits.append(orb)
    return orbits


class ConfigData:
    """Everything about one configuration that does not depend on the chosen bond orbits."""

    def __init__(self, G, wy, config, reach, samples, rng, use_numba=True):
        self.config = config
        self.wyckoff = [wy[k]["name"] for k in config]
        self.ok = False
        try:
            base = Family(G, _ref_points(wy, config))
        except ValueError:
            return
        self.base = base
        self.m = base.m
        self.nops = len(base._ops)
        self.d = len(base.site_names)
        P, q = base._affine_sites()
        S = samples if self.d else 1
        self.U = rng.uniform(-1.0, 1.0, (S, self.d))
        self.Fs = q[None, :, :] + np.einsum("kcd,sd->skc", P, self.U)
        valid = np.ones(S, bool)
        for i in range(self.m):
            for j in range(i):
                D = self.Fs[:, i] - self.Fs[:, j]
                valid &= np.max(np.abs(D - np.round(D)), axis=1) > 1e-5
        self.valid = valid
        self.use_numba = use_numba
        orbits = candidate_orbits(base, reach)
        self.n_candidates = len(orbits)
        self_ok, _ = KN.sample_masks(self.Fs, orbits, valid, use_numba, pairs=False) if orbits else (np.zeros((0, 1), np.uint64), None)
        keep = [a for a in range(len(orbits)) if self_ok[a].any()]
        self.n_dropped = len(orbits) - len(keep)          # never drawable alone at any sample
        self.orbits = [orbits[a] for a in keep]
        self.self_ok = np.ascontiguousarray(self_ok[np.array(keep, int)]) if keep else self_ok[:0]
        self.pair_ok = None                               # filled on first need (n >= 2)
        degv = np.zeros((len(self.orbits), self.m), np.int64)
        for a, o in enumerate(self.orbits):
            for (i, j, _, _) in o:
                degv[a, i] += 1; degv[a, j] += 1
        self.degv = degv
        self.nb = np.array([len(o) for o in self.orbits], np.int64)
        self.flat = KN._flat(self.orbits)
        self.action = KD.action_arrays(base)
        self.ok = True

    def points_exact(self):
        """Do the points alone (no bonds) have exactly the symmetry G with G's translations? Tested with the legacy
        group finder at 3 random parameter values (two of three must agree); cached. computational evidence."""
        if not hasattr(self, "_points_exact"):
            rng = np.random.default_rng(12345)
            hits = 0
            for _ in range(3):
                try:
                    g, m = self.base.set(**self.base.random_values(rng)).group_at()
                except ValueError:
                    continue
                hits += (g == self.base.group and m == self.base.m)
            self._points_exact = hits >= 2
        return self._points_exact

    def masks(self, n):
        if n >= 2 and self.pair_ok is None:
            self.self_ok, self.pair_ok = KN.sample_masks(self.Fs, self.orbits, self.valid, self.use_numba)
        if self.pair_ok is None:
            K, W = self.self_ok.shape
            pair = np.zeros((K, K, W), np.uint64)
            pair[np.arange(K), np.arange(K)] = self.self_ok
            return pair, self.self_ok
        return self.pair_ok, self.self_ok

    def possible(self, n, min_degree):
        """Can an n-set of the candidates pass the degree bound at all?"""
        if not self.ok or len(self.orbits) < n:
            return False
        top = np.sort(self.nb)[::-1][:n].sum()
        return min_degree * self.m <= 2 * min(top, 3 * self.m)

    def values(self, u, cell=None):
        v = dict(zip(self.base.site_names, (float(x) for x in u)))
        if cell:
            v.update(cell)
        return v


# ----------------------------------------------------------------------------------------------
# tiling key of one structure
# ----------------------------------------------------------------------------------------------
def rotation_signatures(Fs, bonds):
    """For each sample row of Fs, the cyclic successor of every dart round its tail site (fractional
    coordinates; equal rows = equal rotation systems = equal D-symbols). Returns (S, 2E) int array."""
    b = np.array(bonds, np.int64)
    tail = np.concatenate([b[:, 0], b[:, 1]]); head = np.concatenate([b[:, 1], b[:, 0]])
    shift = np.concatenate([b[:, 2:4], -b[:, 2:4]])
    V = Fs[:, head] + shift[None] - Fs[:, tail]
    ang = np.arctan2(V[..., 1], V[..., 0]) + np.pi
    order = np.argsort(tail[None, :] * 8.0 + ang, axis=1, kind="stable")
    t_sorted = np.sort(tail, kind="stable")
    pos = np.arange(len(tail))
    starts = np.searchsorted(t_sorted, t_sorted, side="left")
    ends = np.searchsorted(t_sorted, t_sorted, side="right")
    nxtpos = np.where(pos + 1 < ends, pos + 1, starts)
    nxt = np.empty_like(order)
    np.put_along_axis(nxt, order, np.take_along_axis(order, np.broadcast_to(nxtpos, order.shape), axis=1), axis=1)
    return nxt


def cell_at(base, bonds, F, A=None):
    """UnitCell with the given primitive fractional site coordinates (no origin shift, frame identity)."""
    A = base._Ac() @ base._U if A is None else A
    basis = F @ A.T
    return UnitCell(f"fam_{base.group}", base.group, base.group, A[:, 0], A[:, 1], basis,
                    sorted(_canon_bond(i, j, n1, n2, 0) for (i, j, n1, n2) in bonds))


def random_cell_params(base, rng):
    v = {}
    vals = base.random_values(rng)
    for k in ("b", "gamma"):
        if k in vals:
            v[k] = vals[k]
    return v


def exact_count(base, bonds, F, cellpar):
    """Number of symmetry operations (mod primitive translations of G) of the bare lattice."""
    f = base.set(**cellpar) if cellpar else base
    A = f._Ac() @ f._U
    return len(_symmetry_ops(cell_at(base, bonds, F, A), use_types=False))


# ----------------------------------------------------------------------------------------------
# the search
# ----------------------------------------------------------------------------------------------
class RawEntry:
    """One rule-B class found by the search (lattice.Lattice wraps it)."""

    def __init__(self, key, cd, sub, rank):
        self.key = key
        self.net = network_key(key)
        self.cd = cd
        self.sub = sub
        self.orbits = [cd.orbits[a] for a in sub]
        self.rank = rank
        self.samples = []            # indices of planar samples of `sub` with this key
        self.structures = 1          # number of (configuration, bond-orbit set) giving this key
        self.tier = rank[:2]
        self.exact_cell = None

    def family(self, values=None):
        f = self.cd.base._with_orbits(self.orbits)
        return f.set(**values) if values else f


def search(group, n=7, min_degree=3, reach=1, max_bond_orbits=6, samples=256, time_limit=600, seed=0,
           use_numba=True, stop=True, only_tier=None, verbose=False, same="network", own_symmetry=False,
           points_exact=False):
    """Tier-by-tier search; returns a dict (see lattice.minimal_lattices for the user-facing wrapper).
    stop=False runs every tier inside the bounds; only_tier=(k, n) runs that tier alone.
    points_exact=True (the "sheet" rule, author 2026-11-14): only choices of Wyckoff positions whose POINTS ALONE have
    exactly the symmetry G are used (ConfigData.points_exact), so the bonds never have to break a symmetry."""
    G = WallpaperGroup(group)
    rng = np.random.default_rng(seed)
    wy = ML.wyckoff_positions(G)
    t0 = time.time()
    entries = {}                      # key -> RawEntry
    rejected = Counter()
    notes, tiers = [], []
    counts = Counter()
    nops = None
    prim = 2 if G.centred else 1
    P_order = G.order // prim
    timed_out = False
    k_stop = None
    kmax = max_bond_orbits + 1        # connectivity: n >= k - 1
    k_list = range(1, kmax + 1) if only_tier is None else [only_tier[0]]
    cdata = {}

    def found():                       # what the stopping rule counts: networks, or (rule B) drawings
        if own_symmetry:               # count only networks whose own symmetry is G (D-symbol already minimal)
            return len({e.net for e in entries.values() if len(e.key) == len(e.net)})
        return len({e.net for e in entries.values()}) if same == "network" else len(entries)

    for k in k_list:
        confs = configurations(wy, k)
        mmin = min(sum(wy[i]["mult"] for i in c) for c in confs)
        if min_degree * mmin > 2 * max_bond_orbits * P_order and only_tier is None:
            tiers.append(dict(k=k, n=None, status="empty (bound)", entries=0,
                              note=f"min_degree x sites >= {min_degree * mmin} > 2 x {max_bond_orbits} x {P_order}"))
            break
        n_range = range(1, max_bond_orbits + 1) if only_tier is None else [only_tier[1]]
        for nn in n_range:
            tk = time.time()
            if nn < k - 1:
                tiers.append(dict(k=k, n=nn, status="empty (proved)", entries=0, seconds=0.0,
                                  note="k site orbits need >= k - 1 bond orbits to be connected"))
                continue
            new = 0; partial = False
            tcount = Counter()
            for c in confs:
                if time.time() - t0 > time_limit:
                    partial = True; break
                if c not in cdata:
                    cdata[c] = ConfigData(G, wy, c, reach, samples, rng, use_numba)
                cd = cdata[c]
                if not cd.possible(nn, min_degree):
                    continue
                if points_exact and not cd.points_exact():
                    tcount["choices of sites whose points have more symmetry than G"] += 1
                    continue
                nops = cd.nops
                pair_ok, self_ok = cd.masks(nn)
                for first in range(len(cd.orbits) - nn + 1):
                    if time.time() - t0 > time_limit:
                        partial = True; break
                    subs, masks = KN.subsets(nn, cd.orbits, cd.degv, pair_ok, self_ok, min_degree,
                                             3 * cd.m, first, use_numba)
                    tcount["planar, connected, degree-ok sets"] += len(subs)
                    for sub, mask in zip(subs, masks):
                        new += _classify(cd, sub, mask, k, nn, entries, rejected, tcount, rng, notes)
                if partial:
                    break
            counts.update(tcount)
            st = "OPEN (time limit)" if partial else "complete"
            tiers.append(dict(k=k, n=nn, status=st, entries=new, seconds=round(time.time() - tk, 3),
                              configurations=len(confs), counts=dict(tcount)))
            if verbose:
                print(f"# time tier ({k},{nn}) {st}: {new} new entr(y/ies), {time.time() - tk:.2f} s, total {len(entries)}")
            if partial:
                timed_out = True; break
            if stop and only_tier is None and found() >= n:
                break
        if timed_out or (stop and only_tier is None and found() >= n):
            k_stop = k
            break
    else:
        if only_tier is None:
            tiers.append(dict(k=None, n=None, status="exhausted within bounds", entries=0,
                              note=f"every k <= {kmax} = max_bond_orbits + 1 done"))
    gaps = []
    for c, cd in cdata.items():
        if not cd.ok or (k_stop is not None and len(c) >= k_stop) or only_tier is not None:
            continue
        cs = np.cumsum(np.sort(cd.nb))
        ncap = int(np.sum(cs <= 3 * cd.m))
        if ncap > max_bond_orbits:
            gaps.append(dict(k=len(c), wyckoff=cd.wyckoff, n_max=ncap))
    ranked = sorted(entries.values(), key=lambda e: (e.rank, e.key))
    for e in ranked:
        e.open = (tuple(e.tier) == _last_open(tiers))
    return dict(group=G.symbol, entries=ranked, tiers=tiers, notes=notes, counts=dict(counts), gaps=gaps,
                rejected=sum(rejected.values()), seconds=time.time() - t0, timed_out=timed_out,
                bounds=dict(n=n, min_degree=min_degree, reach=reach, max_bond_orbits=max_bond_orbits,
                            samples=samples, seed=seed, time_limit=time_limit, same=same, own_symmetry=own_symmetry,
                            rule="sheet" if points_exact else "bonds"))


def _last_open(tiers):
    for t in tiers:
        if t["status"].startswith("OPEN"):
            return (t["k"], t["n"])
    return None


def _keys_python(cd, sub, mask, bonds, tcount, notes):
    """Reference route: dsymbol.py at one planar sample per distinct rotation system."""
    smp = np.array(KN.bits_to_list(mask))
    sig = rotation_signatures(cd.Fs[smp], bonds)
    _, first_ix, inv = np.unique(sig, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    keys = {}
    for g, fi in enumerate(first_ix):
        s = smp[fi]
        try:
            key = dsymbol(cell_at(cd.base, bonds, cd.Fs[s]), cd.base._ops)["key"]
        except DSymbolError:
            tcount["D-symbol refused a planar sample"] += 1
            continue
        except RuntimeError as err:
            tcount["errors"] += 1
            notes.append(f"{cd.wyckoff} {sub}: {err}")
            continue
        keys.setdefault(key, []).extend(smp[inv == g].tolist())
    return keys


def _classify(cd, sub, mask, k, nn, entries, rejected, tcount, rng, notes):
    """Keys of one structure at its planar samples; exactness for new keys. Returns # new entries."""
    bonds = [b for a in sub for b in cd.orbits[a]]
    keys = None
    if cd.use_numba and KN.HAVE_NUMBA:
        bi, bj, bn, ptr = cd.flat
        P, Sh, M, fl = cd.action
        karr, nk, which, status = KD.keys_of_structure(np.asarray(sub, np.int64), mask, cd.Fs, bi, bj, bn, ptr,
                                                       cd.m, P, Sh, M, fl)
        if status == 0:
            keys = {}
            for q in range(nk):
                keys[KD.key_to_tuple(karr[q])] = np.nonzero(which == q)[0].tolist()
            nund = sum(bin(int(x)).count("1") for x in mask) - int(np.sum(which >= 0))
            if nund:
                tcount["D-symbol refused a planar sample"] += nund
    if keys is None:
        keys = _keys_python(cd, sub, mask, bonds, tcount, notes)
    if len(keys) > 1:
        tcount["structures with several keys"] += 1
        notes.append(f"configuration {cd.wyckoff}, bond orbits {sub}: {len(keys)} different tilings at different "
                     "planar samples (the planar region has several pieces)")
    new = 0
    bpc = sum(len(cd.orbits[a]) for a in sub)
    for key, ss in keys.items():
        if key in entries:
            entries[key].structures += 1
            tcount["duplicates (same key)"] += 1
            continue
        cnt = exact_count(cd.base, bonds, cd.Fs[ss[0]], random_cell_params(cd.base, rng))
        if cnt < cd.nops:
            notes.append(f"{cd.wyckoff} {sub}: only {cnt} < {cd.nops} operations found (tolerance?)")
        if cnt != cd.nops:
            rejected[key] += 1
            tcount["generic group larger than G"] += 1
            continue
        e = RawEntry(key, cd, sub, (k, nn, cd.m, bpc))
        e.samples = ss
        entries[key] = e
        new += 1
    return new
