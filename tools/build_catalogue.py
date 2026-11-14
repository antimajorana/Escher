"""Build papertilling/data/minimal_b.json: the first 7 lattices (rule B, ranking b) of all 17 groups,
plus a coverage audit against the old work (legacy/data/minimal_lattices.json and the catalogue
lattices of wallpaper_lattices).

    python -X utf8 tools/build_catalogue.py            # all groups
    python -X utf8 tools/build_catalogue.py p4m p6m    # some groups (merged into the existing file)
    python -X utf8 tools/build_catalogue.py --bench    # numba vs Python kernel timings only
    python -X utf8 tools/build_catalogue.py --stability [groups]   # rerun with seed 1 and with 512 samples,
                                                       # compare keys tier by tier with the stored file

Bounds used here (technical choices, stored in the file): reach 1, max_bond_orbits 6 (at 4, eleven groups
have a rank gap for k = 1, see the coverage text; 6 closes every gap and costs < 1 min per group),
min_degree 3, 256 planarity samples per configuration, seed 0, time limit 280 s per group.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from papertilling.lattice import minimal_lattices, family_key, check_family, DATA          # noqa: E402
from papertilling import enumerate as EN                                                  # noqa: E402
from papertilling.legacy.lattice_families import Family, in_coverage, degrees_of           # noqa: E402
from papertilling.legacy.wallpaper_lattices import WALLPAPER_GROUPS, available_lattices, WallpaperGroup  # noqa: E402
from papertilling.legacy import minimal_lattices as ML                                     # noqa: E402

BOUNDS = dict(n=7, min_degree=3, reach=1, max_bond_orbits=6, samples=256, time_limit=280, seed=0)
OLD = os.path.join(ROOT, "papertilling", "legacy", "data", "minimal_lattices.json")


def old_lattices():
    with open(OLD, encoding="utf8") as f:
        old = json.load(f)
    out = [(f"legacy {g} planar #{i + 1}", lambda e=e: Family.from_recipe(e["recipe"]))
           for g, r in old["groups"].items() for i, e in enumerate(r["planar"])]
    out += [(f"catalogue {n}", lambda n=n: Family.from_cell(n)) for n in available_lattices()]
    return out


def audit(results):
    """For every old lattice: does it satisfy the definition, is it inside the bounds, and is its key
    found? Lattices in a tier beyond the stopping tier are searched on their own (only_tier)."""
    rows = []
    extra = {}
    for label, make in old_lattices():
        fam = make()
        row = dict(lattice=label, group=fam.group, rank=[len(fam.gens), len(fam.orbits), fam.m, len(fam.bonds())])
        if min(degrees_of(fam.m, fam.bonds())) < BOUNDS["min_degree"]:
            row["verdict"] = "outside definition (a site has < 3 bonds)"; rows.append(row); continue
        st = check_family(fam)
        if not st["exact"]:
            row["verdict"] = f"outside definition (group {st['group']} with {st['primitive_sites']} sites, not exact)"
            rows.append(row); continue
        if not st["planar"] or st["bond_through_site"]:
            row["verdict"] = "outside definition at its own values (bonds cross or pass through a site)"
            rows.append(row); continue
        ok, why = in_coverage(fam, BOUNDS["max_bond_orbits"], BOUNDS["reach"])
        if not ok:
            row["verdict"] = f"outside bounds ({why})"; rows.append(row); continue
        key = family_key(fam)
        res = results.get(fam.group)
        tier = (len(fam.gens), len(fam.orbits))
        done = {(t["k"], t["n"]) for t in res["tiers"] if t["status"] == "complete"} if res else set()
        if tier in done:
            keys = {tuple(map(tuple, e["key"])) for e in res["entries"]}
            row["verdict"] = "found" if key in keys else "COVERAGE HOLE (in a complete tier, key not found)"
        else:
            if (fam.group, tier) not in extra:
                t0 = time.time()
                r = EN.search(fam.group, only_tier=tier, **{k: v for k, v in BOUNDS.items() if k not in ("n", "time_limit")},
                              time_limit=90)
                extra[(fam.group, tier)] = (r, time.time() - t0)
            r, secs = extra[(fam.group, tier)]
            found = key in {e.key for e in r["entries"]}
            stt = r["tiers"][0]["status"] if r["tiers"] else "?"
            row["verdict"] = (("found" if found else ("COVERAGE HOLE" if stt == "complete" else "not found, tier OPEN"))
                              + f" (beyond stopping tier; tier {tier} searched alone, {stt}, {secs:.1f} s)")
        rows.append(row)
    return rows


def bench():
    """numba vs Python reference on one configuration (p2, 2 general sites): same results, timings."""
    from papertilling import kernels as KN
    G = WallpaperGroup("p2")
    wy = ML.wyckoff_positions(G)
    gen = next(k for k, w in enumerate(wy) if w["kind"] == "general")
    out = {}
    for S in (64,):
        cd = EN.ConfigData(G, wy, (gen,), 1, S, np.random.default_rng(0))
        KN.sample_masks(cd.Fs[:2], cd.orbits[:2], cd.valid[:2], True)            # JIT warm-up
        t = time.perf_counter(); a = KN.sample_masks(cd.Fs, cd.orbits, cd.valid, True); t_nb = time.perf_counter() - t
        t = time.perf_counter(); b = KN.sample_masks(cd.Fs, cd.orbits, cd.valid, False); t_py = time.perf_counter() - t
        same = all(np.array_equal(x, y) for x, y in zip(a, b))
        out[f"crossing masks p2 2e, {len(cd.orbits)} orbits, {S} samples"] = dict(numba=t_nb, python=t_py, speedup=t_py / t_nb, same=same)
        selfm, pair = a
        for n in (3, 4):
            t = time.perf_counter()
            ra = [KN.subsets(n, cd.orbits, cd.degv, pair, selfm, 3, 3 * cd.m, f, True)[0] for f in range(len(cd.orbits) - n + 1)]
            t_nb = time.perf_counter() - t
            t = time.perf_counter()
            rb = [KN.subsets(n, cd.orbits, cd.degv, pair, selfm, 3, 3 * cd.m, f, False)[0] for f in range(len(cd.orbits) - n + 1)]
            t_py = time.perf_counter() - t
            out[f"subset search n={n}"] = dict(numba=t_nb, python=t_py, speedup=t_py / t_nb, same=ra == rb,
                                               sets=sum(len(x) for x in ra))
    # D-symbol keys: numba kernel vs dsymbol.py on the structures of one tier
    from papertilling import kernels_dsymbol as KD
    from collections import Counter
    cd = EN.ConfigData(G, wy, (gen,), 1, 256, np.random.default_rng(0))
    pair, selfm = cd.masks(3)
    structs = [(s, m) for f in range(len(cd.orbits) - 2) for s, m in zip(*KN.subsets(3, cd.orbits, cd.degv, pair, selfm, 3, 3 * cd.m, f))]
    bi, bj, bn, ptr = cd.flat; P, Sh, M, fl = cd.action
    t = time.perf_counter()
    for s, m in structs:
        KD.keys_of_structure(np.asarray(s, np.int64), m, cd.Fs, bi, bj, bn, ptr, cd.m, P, Sh, M, fl)
    t_nb = time.perf_counter() - t
    t = time.perf_counter()
    for s, m in structs:
        EN._keys_python(cd, s, m, [b for a in s for b in cd.orbits[a]], Counter(), [])
    t_py = time.perf_counter() - t
    out[f"D-symbol keys, {len(structs)} structures"] = dict(numba=t_nb, python=t_py, speedup=t_py / t_nb)
    for k, v in out.items():
        print(f"# time {k}: " + ", ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    return out


def stability(groups):
    """Second route for the lists: other random samples (seed 1) and twice as many samples (512).
    Any tier complete in both runs must hold the same set of keys."""
    with open(DATA, encoding="utf8") as f:
        data = json.load(f)
    bad = 0
    for g in groups:
        base = data["groups"][g]
        ref = {}
        for e in base["entries"]:
            ref.setdefault(tuple(e["meta"]["tier"]), set()).add(tuple(map(tuple, e["key"])))
        done0 = {(t["k"], t["n"]) for t in base["tiers"] if t["status"] == "complete"}
        first0 = [tuple(map(tuple, e["key"])) for e in base["entries"][:BOUNDS["n"]]]
        for label, kw in (("seed 1", dict(seed=1)), ("512 samples", dict(samples=512))):
            b = dict(BOUNDS); b.update(kw); b.pop("n")
            t0 = time.perf_counter()
            r = EN.search(g, n=BOUNDS["n"], **b)
            got = {}
            for e in r["entries"]:
                got.setdefault(tuple(e.tier), set()).add(e.key)
            done1 = {(t["k"], t["n"]) for t in r["tiers"] if t["status"] == "complete"}
            diffs = [(t, len(ref.get(t, ())), len(got.get(t, ()))) for t in sorted(done0 & done1)
                     if ref.get(t, set()) != got.get(t, set())]
            first1 = [e.key for e in r["entries"][:BOUNDS["n"]]]
            bad += bool(diffs)
            print(f"# time {g:5s} {label:12s} {time.perf_counter() - t0:6.1f} s  tiers compared {len(done0 & done1)}, "
                  f"differing {diffs or 'none'}; first {BOUNDS['n']} {'same' if first1 == first0 else 'DIFFERENT'}", flush=True)
    print("tiers with different key sets:", bad)


def main(argv):
    if "--bench" in argv:
        bench(); return
    if "--stability" in argv:
        stability([g for g in argv if not g.startswith("-")] or WALLPAPER_GROUPS); return
    groups = [g for g in argv if not g.startswith("-")] or WALLPAPER_GROUPS
    data = dict(settings=BOUNDS, groups={})
    if os.path.exists(DATA) and groups != WALLPAPER_GROUPS:
        with open(DATA, encoding="utf8") as f:
            data = json.load(f)
    for g in groups:
        t = time.perf_counter()
        res = minimal_lattices(g, **BOUNDS)
        d = res.to_dict()
        d["build_seconds"] = round(time.perf_counter() - t, 2)
        data["groups"][g] = d
        st = [t_ for t_ in res.tiers if t_["status"] not in ("empty (proved)",)]
        print(f"# time {g:5s} {d['build_seconds']:6.1f} s  {len(res.all)} entries, last tier "
              f"({st[-1]['k']},{st[-1]['n']}) {st[-1]['status']}, gaps {len(res.gaps)}", flush=True)
        for k, L in enumerate(res.lattices, 1):
            print(f"   #{k} {L.rank} {'; '.join(L.wyckoff())} bond orbits {[len(o) for o in L.family.orbits]} "
                  f"routes {L.meta['routes']} agree {L.meta['routes_agree']}")
        for nte in res.notes:
            print("   note:", nte)
    t = time.perf_counter()
    data["audit"] = audit(data["groups"])
    print(f"# time audit {time.perf_counter() - t:.1f} s")
    for row in data["audit"]:
        if row["verdict"] != "found" and not row["verdict"].startswith("outside"):
            print("  ", row)
    from collections import Counter
    print("audit verdicts:", dict(Counter(r["verdict"].split(" (")[0] for r in data["audit"])))
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", encoding="utf8") as f:
        json.dump(data, f, indent=0, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("written", DATA)


if __name__ == "__main__":
    main(sys.argv[1:])
