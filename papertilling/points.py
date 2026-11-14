"""The points-only rule for "the most Bravais-like lattices of a wallpaper group G" (author, 2026-11-14).

A Bravais lattice is ONE point copied by a group of translations: every site is equivalent, and nothing else
is chosen. The same idea for a wallpaper group G:

  DEFINITIONS (the author's, 2026-11-14)
    lattice    a set of points: the union of k orbits of G (k Wyckoff positions with their free parameters),
               whose symmetry, judged on the POINTS ALONE, is exactly G with G's translations.
    order      fewest site orbits k first (k = 1: every site equivalent, as in a Bravais lattice), then
               fewest sites per primitive cell.
    bonds      not chosen: two sites are joined when their Voronoi cells share an edge (the Delaunay rule;
               the author's choice "Voronoi neighbours"). Four or more sites on one empty circle (forced by
               symmetry) are left as one face.
    same       one entry per network per group (the author's rule of the same day, dsymbol.network_key): when a
               second set of points of the group gives a network already listed, it is kept in result.other.
  CONSEQUENCES (proved results, short arguments; formal verification not performed)
    * the Voronoi/Delaunay construction uses only distances, so the network has exactly the symmetry of the
      points: bonds can neither add nor remove symmetry;
    * the network is planar, connected, has no bond through a site, and every site has >= 3 bonds.
  TECHNICAL CHOICES (mine)
    * Tiers (k, sites per primitive cell) are searched in order, each one completely; the search stops after
      the tier that brings the list to n entries.
    * "exactly G" is tested on the lattice with its Voronoi bonds, by the number of symmetry operations
      (= |G / T|), once per different bond set met in `samples` random parameter values; a configuration is
      kept when more than half of its samples are exact. Final entries are re-checked by three routes
      (lattice.check_family). computational evidence.
    * The Voronoi network of one configuration changes when a parameter crosses a value where four sites become
      cocircular. Each network met in the samples is a "phase"; phases met in fewer than 4 % of the samples are
      ignored. A phase missed by every sample is not found.
    * Two configurations that share a phase (equal D-symbol key under G) are the same lattice up to a symmetry
      of the pattern of G (e.g. the positions 1a, 1b, 1c, 1d of p2): listed once.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import Delaunay

from .dsymbol import network_key
from .enumerate import configurations, _ref_points
from .legacy import minimal_lattices as ML
from .legacy.lattice_families import Family
from .legacy.wallpaper_lattices import WallpaperGroup, _canon_bond, _symmetry_ops


def voronoi_bonds(F, A, tol=1e-7):
    """Bonds (i, j, n1, n2) between Voronoi neighbours of the periodic point set with primitive fractional
    coordinates F (m, 2) and cell matrix A (columns a1, a2). An edge of the Delaunay triangulation whose two
    triangles have their four corners on one circle is dropped (the sites of that circle form one face).
    The points of (2R+1)^2 cells are triangulated; R grows until the bonds of the central cell stop changing."""
    F = np.asarray(F, float)
    m = len(F)
    last = None
    for R in (2, 3, 4):
        r = np.arange(-R, R + 1)
        sh = np.array([(a, b) for a in r for b in r])
        P = (F[None] + sh[:, None, :]).reshape(-1, 2) @ A.T                  # point index = cell * m + site
        site, cell = np.tile(np.arange(m), len(sh)), np.repeat(sh, m, axis=0)
        tri = Delaunay(P)
        S, N = tri.simplices, tri.neighbors
        a, b, c = P[S[:, 0]], P[S[:, 1]], P[S[:, 2]]
        d = 2 * ((a[:, 0] - c[:, 0]) * (b[:, 1] - c[:, 1]) - (b[:, 0] - c[:, 0]) * (a[:, 1] - c[:, 1]))
        ok = np.abs(d) > 1e-14
        d = np.where(ok, d, 1.0)
        a2, b2 = ((a - c) ** 2).sum(1), ((b - c) ** 2).sum(1)
        cx = c[:, 0] + (a2 * (b[:, 1] - c[:, 1]) - b2 * (a[:, 1] - c[:, 1])) / d
        cy = c[:, 1] + (b2 * (a[:, 0] - c[:, 0]) - a2 * (b[:, 0] - c[:, 0])) / d
        rad = np.hypot(a[:, 0] - cx, a[:, 1] - cy)
        bonds = set()
        for k in range(3):                                   # the edge opposite corner k of every triangle
            p, q, u = S[:, (k + 1) % 3], S[:, (k + 2) % 3], N[:, k]
            central = np.all(cell[p] == 0, axis=1) | np.all(cell[q] == 0, axis=1)
            far = np.where(u >= 0, S[u].sum(1) - p - q, 0)   # the corner of the neighbouring triangle off the edge
            on_circle = (u >= 0) & ok & (np.abs(np.hypot(P[far, 0] - cx, P[far, 1] - cy) - rad) < tol * np.maximum(rad, 1.0))
            for x, y in zip(p[central & ~on_circle], q[central & ~on_circle]):
                n = cell[y] - cell[x]
                bonds.add(_canon_bond(int(site[x]), int(site[y]), int(n[0]), int(n[1]), 0)[:4])
        if last is not None and bonds == last:
            return sorted(bonds)
        last = bonds
    return sorted(last)


def _orbits(base, bonds):
    """The bond list split into orbits of G (a sorted tuple of tuples), or None if it is not closed under G."""
    left, out, all_ = set(bonds), [], set(bonds)
    while left:
        orb = base._orbit_of(min(left))
        if not set(orb) <= all_:
            return None
        out.append(tuple(orb)); left -= set(orb)
    return tuple(sorted(out))


def with_voronoi_bonds(base, values):
    """The family with the Voronoi bonds of the points of `base` at `values` (None if the bond set found is not
    closed under G: a tolerance failure, since the exact Voronoi bonds always are)."""
    f = base.set(**values)
    orbits = _orbits(base, voronoi_bonds(f._F(), f._Ac() @ f._U))
    return None if orbits is None else base._with_orbits([list(o) for o in orbits]).set(**values)


def is_voronoi(family, values=None):
    """Are the bonds of `family` (at `values`) exactly the Voronoi-neighbour bonds of its points?"""
    f = family.set(**values) if values else family
    try:
        return sorted(voronoi_bonds(f._F(), f._Ac() @ f._U)) == sorted(tuple(b[:4]) for b in f.bonds())
    except Exception:
        return False


def _sites_distinct(f, tol=1e-4):
    F = f._F()
    D = F[:, None, :] - F[None, :, :]
    D = np.abs(D - np.round(D)).max(axis=2)
    return bool(np.all(D[~np.eye(len(F), dtype=bool)] > tol)) if len(F) > 1 else True


def phases(group, wy, config, samples, rng):
    """The Voronoi networks of one configuration of Wyckoff positions, met at `samples` random parameter values.
    Returns None if the configuration is impossible, else dict(base, exact, total, phases) with
    phases = {D-symbol key: dict(family, candidates [values with the family's bond labels], count)};
    exact / total = samples whose points have exactly the symmetry G / samples used."""
    from .lattice import family_key
    try:
        base = Family(group, _ref_points(wy, config))
    except ValueError:
        return None
    by_bonds, total = {}, 0
    for _ in range(samples if base.names else 1):
        vals = base.random_values(rng)
        f = base.set(**vals)
        if not _sites_distinct(f):
            continue
        orbits = _orbits(base, voronoi_bonds(f._F(), f._Ac() @ f._U))
        if orbits is None:
            continue
        total += 1
        d = by_bonds.setdefault(orbits, dict(count=0, candidates=[]))
        d["count"] += 1
        if len(d["candidates"]) < 24:
            d["candidates"].append(vals)
    out, exact = {}, 0
    for orbits, d in by_bonds.items():                       # one symmetry test and one key per bond set
        fam = base._with_orbits([list(o) for o in orbits]).set(**d["candidates"][0])
        if len(_symmetry_ops(fam.cell(), use_types=False)) != len(base._ops):
            continue                                         # here the points have more symmetry than G
        exact += d["count"]
        try:
            key = family_key(fam)
        except Exception:
            continue
        ph = out.setdefault(key, dict(family=fam, candidates=d["candidates"], count=0, largest=0))
        ph["count"] += d["count"]
        if d["count"] > ph["largest"]:                       # the bond labels met most often carry the candidates
            ph.update(family=fam, candidates=d["candidates"], largest=d["count"])
    return dict(base=base, exact=exact, total=total, phases=out)


def _lattice_of(ph, key, rng, meta, polish):
    from .lattice import Lattice, _best_values, check_family, family_rank
    fam = ph["family"]
    v = _best_values(fam, key, ph["candidates"], rng, polish=polish, accept=lambda w: is_voronoi(fam, w))
    fam = fam.set(**v)
    st = check_family(fam, key=key)
    cnt = len(_symmetry_ops(fam.cell(), use_types=False))
    routes = dict(count=fam.group if cnt == len(fam._ops) else f"{cnt} operations, not {len(fam._ops)}",
                  legacy=st["group"] if st["exact"] else f"{st['group']} ({st['primitive_sites']} sites)", spglib=st["spglib"])
    agree = cnt == len(fam._ops) and st["exact"] and st["spglib"] == fam.group and st["same_key"] and st["planar"]
    L = Lattice(fam, key=key, rank=family_rank(fam), planar_values=ph["candidates"][:20],
                meta=dict(meta, routes=routes, routes_agree=bool(agree), voronoi=bool(is_voronoi(fam))))
    L.status = st
    return L


def points_lattices(group, n=7, samples=48, seed=0, time_limit=600, max_orbits=9, polish=200):
    """The first n lattices of `group` under the points-only rule (see the module docstring).
    Returns a lattice.MinimalResult: .lattices (first n), .all, and .other[k] = up to 3 other sets of points of
    the group (or other parameter ranges) whose Voronoi network is the network of entry k."""
    from .lattice import MinimalResult
    G = WallpaperGroup(group)
    rng = np.random.default_rng(seed)
    wy = ML.wyckoff_positions(G)
    name = lambda config: [wy[i]["name"] for i in config]
    t0 = time.time()
    by_key, by_net = {}, {}                     # D-symbol key / network key -> index of the entry
    repeats = set()                             # keys of other points whose network is already listed
    entries, tiers = [], []
    done = False
    for k in range(1, max_orbits + 1):
        confs = configurations(wy, k)
        for m in sorted({sum(wy[i]["mult"] for i in c) for c in confs}):
            cnt = dict(new=0, more_symmetry_than_G=0, same_lattice=0, same_network=0, impossible=0)
            status = "complete"
            for config in [c for c in confs if sum(wy[i]["mult"] for i in c) == m]:
                if time.time() - t0 > time_limit:
                    status = "OPEN (time limit)"; break
                r = phases(G, wy, config, samples, rng)
                if r is None or r["total"] == 0:
                    cnt["impossible"] += 1; continue
                if 2 * r["exact"] <= r["total"]:
                    cnt["more_symmetry_than_G"] += 1; continue
                ranked = sorted(r["phases"].items(), key=lambda kv: (-kv[1]["count"], kv[0]))
                ranked = [kv for kv in ranked if kv[1]["count"] >= 0.04 * r["exact"]] or ranked[:1]
                for key, ph in ranked:
                    share = round(ph["count"] / r["exact"], 3)
                    if key in repeats:
                        cnt["same_network"] += 1
                        continue
                    if key in by_key:                         # the same lattice from an equivalent position
                        e = entries[by_key[key]]
                        if name(config) not in e["also"] and name(config) != name(e["config"]):
                            e["also"].append(name(config))
                        cnt["same_lattice"] += 1
                        continue
                    net = network_key(key)
                    if net in by_net:                         # other points, a network already listed
                        repeats.add(key)
                        entries[by_net[net]]["same_network"].append(dict(config=config, key=key, ph=ph, share=share, phases=len(ranked)))
                        cnt["same_network"] += 1
                        continue
                    by_key[key] = by_net[net] = len(entries)
                    entries.append(dict(config=config, key=key, ph=ph, share=share, phases=len(ranked), also=[], same_network=[]))
                    cnt["new"] += 1
            tiers.append(dict(k=k, n=f"{m} sites", status=status, entries=cnt.pop("new"), counts=cnt))
            if status != "complete" or len(entries) >= n:
                done = True; break
        if done:
            break
    lattices, other, notes = [], [], []
    for i, e in enumerate(entries):
        meta = dict(rule="points", wyckoff_search=name(e["config"]), equivalent_positions=e["also"], phases=e["phases"],
                    phase_share=e["share"], same_network=len(e["same_network"]))
        L = _lattice_of(e["ph"], e["key"], rng, meta, polish if i < n else 0)
        lattices.append(L)
        other.append([_lattice_of(o["ph"], o["key"], rng, dict(rule="points", wyckoff_search=name(o["config"]), phases=o["phases"],
                                                               phase_share=o["share"]), 0) for o in e["same_network"][:3]] if i < n else [])
        if not L.meta["routes_agree"]:
            notes.append(f"group routes disagree for {L.rank} {L.wyckoff()}: {L.meta['routes']}")
    bounds = dict(rule="points", n=n, samples=samples, seed=seed, time_limit=time_limit, max_orbits=max_orbits, same="action")
    ok = [f"({t['k']} kind(s) of site, {t['n']})" for t in tiers if t["status"] == "complete"]
    cov = (f"{G.symbol}: points-only rule. Tiers searched completely, in order: {', '.join(ok) or '-'}"
           + ("; the next tier ran out of time (OPEN)" if any(t["status"] != "complete" for t in tiers) else "") + ".\n"
           f"A choice of Wyckoff positions is kept when its points alone have exactly the symmetry {G.symbol} (tested at up to "
           f"{samples} random parameter values, seed {seed}); bonds = Voronoi neighbours; one entry per network. A network "
           "met in fewer than 4 % of the samples is ignored; one missed by every sample is not found.")
    return MinimalResult(G.symbol, lattices, tiers, cov, notes, n, time.time() - t0, bounds, {}, (), same="action", other=other)
