"""Thorough check of the symmetry and of the "same network" test, on every drawing of the stored catalogue.

    python tools/checks/symmetry_check.py            (about 1-2 minutes)

Run for the three stored catalogues (the points-only rule, the sheet rule and the bond-counting rule). For every group G and
every drawing stored for it (not only the 7 listed):

 A. SYMMETRY, at several parameter values. At up to 5 of the drawing's stored planar parameter values, each
    with a fresh random cell shape, the wallpaper group is found by three routes and must be G each time:
      count    number of symmetry operations of the bare lattice (legacy _symmetry_ops) = |G / translations|
      legacy   the group finder of lattice_families (group_at), with the number of sites per primitive cell
      spglib   spglib's layer-group finder (independent code)
    and the lattice must be planar, with no bond through a site, and with the stored D-symbol key.

 B. NO NETWORK TWICE. Inside one group's list (.all) every network appears once (dsymbol.network_key).

 C. THE NETWORK TEST ITSELF, by a second route that does not use D-symbols: the "fingerprint" of a lattice =
    for every site, its vertex type (face sizes round it) and its coordination sequence (how many sites are
    1, 2, ..., 6 bonds away, found by walking the periodic graph). Same network => same fingerprint (proved:
    both are read off the network). So
      - two drawings with EQUAL network keys must have equal fingerprints (a failure = network_key is wrong);
      - two drawings with DIFFERENT network keys and equal fingerprints are listed as "look-alikes": this
        can happen for genuinely different networks, so it is reported, not counted as a failure.

 E. (points-only rule) the stored bonds are exactly the Voronoi-neighbour bonds of the points.

 D. Information: for every listed network, the groups in whose list it appears, and how many times larger
    its own symmetry is than each group (Lattice.symmetry_index).
"""
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

warnings.simplefilter("ignore")
from papertilling.lattice import load_catalogue, check_family                    # noqa: E402
from papertilling.enumerate import random_cell_params                            # noqa: E402
from papertilling.points import is_voronoi                                       # noqa: E402
from papertilling.legacy.wallpaper_lattices import _symmetry_ops                 # noqa: E402

DEPTH, POINTS = 6, 5


def fingerprint(L):
    """(the different vertex types, the different (bonds at the site, coordination sequence to DEPTH)) of the lattice."""
    cell = L.cell()
    m = len(cell.basis)
    nb = defaultdict(list)
    for (i, j, n1, n2, _t) in cell.bonds:
        nb[i].append((j, n1, n2)); nb[j].append((i, -n1, -n2))
    vt = L.vertex_types()
    out = []
    for s in range(m):
        seen = {(s, 0, 0)}
        shell, seq = [(s, 0, 0)], []
        for _ in range(DEPTH):
            nxt = []
            for (i, a, b) in shell:
                for (j, n1, n2) in nb[i]:
                    k = (j, a + n1, b + n2)
                    if k not in seen:
                        seen.add(k); nxt.append(k)
            seq.append(len(nxt)); shell = nxt
        out.append((len(nb[s]), tuple(seq)))
    return (tuple(vt), tuple(sorted(set(out))))        # a SET: one network can be drawn with 1 or 2 sites per cell


def main(rule):
    t0 = time.time()
    rng = np.random.default_rng(7)
    cat = load_catalogue(rule=rule)
    print()
    print(f"===== lists of the rule '{rule}' =====")
    bad, npts, ndraw = [], 0, 0
    nets = defaultdict(list)            # network key -> [(group, list number or None, drawing index, symmetry index)]
    prints = {}
    for g, res in cat.items():
        listed = {id(L): k for k, L in enumerate(res.lattices, 1)}
        keys = [L.network for L in res.all]
        if len(set(keys)) != len(keys):
            bad.append((g, "B: a network appears twice in the list"))
        if len(res.lattices) < 7:
            bad.append((g, f"only {len(res.lattices)} networks"))
        for d, L in enumerate(res.drawings, 1):
            ndraw += 1
            f = L.family
            for v in L.planar_values[:POINTS]:
                vals = dict(v); vals.update(random_cell_params(f, rng))
                fam = f.set(**vals)
                st = check_family(fam, key=L.key)
                cnt = len(_symmetry_ops(fam.cell(), use_types=False))
                npts += 1
                why = [n for n, ok in dict(count=cnt == len(f._ops), legacy=st["exact"], spglib=st["spglib"] == g,
                                           planar=st["planar"], clear=not st["bond_through_site"], key=st["same_key"]).items() if not ok]
                if why:
                    bad.append((g, f"A: drawing {d} rank {L.rank} at {({k: round(x, 3) for k, x in vals.items()})}: {why} "
                                   f"(legacy {st['group']}, spglib {st['spglib']}, {cnt} operations)"))
            if rule == "points" and not is_voronoi(f):
                bad.append((g, f"drawing {d} rank {L.rank}: the stored bonds are not the Voronoi bonds"))
            net = L.network
            nets[net].append((g, listed.get(id(L)), d, L.symmetry_index))
            fp = fingerprint(L)
            if net in prints and prints[net][0] != fp:
                bad.append((g, f"C: drawing {d} has the network key of {prints[net][1]} but another fingerprint"))
            prints.setdefault(net, (fp, f"{g} drawing {d}"))
        print(f"{g:5s} {len(res.drawings):3d} drawings, {len(res.all):3d} networks; listed: "
              + " ".join(f"#{k}(x{L.symmetry_index})" for k, L in enumerate(res.lattices, 1)))
    by_fp = defaultdict(list)
    for net, (fp, where) in prints.items():
        by_fp[fp].append(where)
    alike = [w for w in by_fp.values() if len(w) > 1]
    print(f"\nA. {ndraw} drawings checked at {npts} parameter points by three routes.")
    print(f"C. {len(prints)} different networks; equal keys always had equal fingerprints: "
          f"{not any(b[1].startswith('C') for b in bad)}; look-alikes (different key, same fingerprint to depth {DEPTH}): {len(alike)}")
    for w in alike[:10]:
        print("     look-alike:", w)
    print("\nD. Networks listed (among the first 7) in more than one group  [x = how much larger the network's own symmetry is]:")
    for net, rows in sorted(nets.items(), key=lambda kv: (len(kv[0]), kv[0])):
        hit = [(g, k, x) for (g, k, d, x) in rows if k is not None]
        if len({g for g, _, _ in hit}) > 1:
            L = None
            print(f"   {len(net)}-element network (faces {sorted({r[0] for r in net})}, bonds per site {sorted({r[1] for r in net})}): "
                  + ", ".join(f"{g} #{k} (x{x})" for g, k, x in hit))
    print("\nSYMMETRY CHECK", "PASSED" if not bad else f"FAILED ({len(bad)})", f"  # time {time.time() - t0:.0f} s")
    for b in bad[:20]:
        print("  ", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(max(main("points"), main("sheet"), main("bonds")))
