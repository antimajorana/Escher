"""Re-check the shipped catalogue (papertilling/data/minimal_b.json) entry by entry, from the recipes alone.

For every group: 7 entries; bounds as documented (max_bond_orbits 6, no open tier, no rank gap); and for every
entry, rebuilt from its stored recipe: the D-symbol key recomputed by the Python reference route (dsymbol.py)
equals the stored key; the group is exactly G by three routes (operation count, legacy group finder, spglib);
planar; no bond through a site; every site has >= min_degree bonds; rank recomputed from the family.
Also: the operations of every family are exact integer matrices in the primitive basis (what the numba kernel assumes).

    python tools/checks/catalogue_check.py       (about 20 s)
"""
import sys, time, warnings
import numpy as np
warnings.simplefilter("ignore")
from papertilling.lattice import load_catalogue, check_family, family_rank
from papertilling.legacy.lattice_families import degrees_of
from papertilling.legacy.wallpaper_lattices import _symmetry_ops

t0 = time.time(); bad = []; nent = 0; worst = 0.0
cat = load_catalogue(rule="bonds")
for g, res in cat.items():
    b = res.bounds
    if len(res.lattices) != 7: bad.append((g, "networks", len(res.lattices)))
    nets = [L.network for L in res.all]
    if len(set(nets)) != len(nets): bad.append((g, "a network is listed twice"))
    if b.get("max_bond_orbits") != 6 or b.get("min_degree") != 3 or b.get("reach") != 1: bad.append((g, "bounds", b))
    if res.gaps: bad.append((g, "rank gap", res.gaps[:1]))
    if any(t["status"].startswith("OPEN") for t in res.tiers): bad.append((g, "open tier"))
    keys = set()
    for k, L in enumerate(res.drawings, 1):
        nent += 1
        f = L.family
        st = check_family(f, key=L.key)
        cnt = len(_symmetry_ops(f.cell(), use_types=False))
        deg = degrees_of(f.m, f.bonds())
        for Mf, _ in f._ops:
            worst = max(worst, float(np.abs(np.asarray(Mf, float) - np.rint(Mf)).max()))
        why = [n for n, ok in dict(key=st["same_key"], exact=st["exact"], spglib=st["spglib"] == g, count=cnt == len(f._ops),
                                   planar=st["planar"], clear=not st["bond_through_site"], degree=min(deg) >= b.get("min_degree", 3),
                                   rank=tuple(L.rank) == family_rank(f), distinct=L.key not in keys).items() if not ok]
        keys.add(L.key)
        if why: bad.append((g, k, why))
    ranks = [L.rank for L in res.drawings]
    if ranks != sorted(ranks): bad.append((g, "not sorted by rank"))
    print(f"{g:5s} {len(res.drawings):2d} drawings, {len(res.all):2d} networks, first 7 ranks {[L.rank for L in res.lattices]}")
print(f"{nent} entries re-checked; largest |M - round(M)| over all operations: {worst:.1e}")
print("CATALOGUE", "OK" if not bad else f"PROBLEMS: {bad}", f"  # time {time.time() - t0:.1f} s")
sys.exit(1 if bad else 0)
