"""Build the stored lists of the two rules that judge the symmetry on the points alone.

    python -X utf8 tools/build_points.py                    # data/points.json: bonds = Voronoi neighbours (a few minutes)
    python -X utf8 tools/build_points.py p4g p6m            # some groups (merged into the existing file)
    python -X utf8 tools/build_points.py --sheet [groups]   # data/sheet.json: bonds = fewest kinds (the author's sheet rule)
"""
import json
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from papertilling.lattice import DATA_POINTS, DATA_SHEET, minimal_lattices     # noqa: E402
from papertilling.points import points_lattices                               # noqa: E402

GROUPS = "p1 p2 pm pg cm pmm pmg pgg cmm p4 p4m p4g p3 p3m1 p31m p6 p6m".split()
BOUNDS = dict(n=7, samples=48, seed=0, time_limit=280)

if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or GROUPS
    sheet = "--sheet" in sys.argv
    DATA_POINTS = DATA_SHEET if sheet else DATA_POINTS
    data = dict(settings=dict(n=7, rule="sheet", time_limit=280) if sheet else BOUNDS, groups={})
    if want != GROUPS and os.path.exists(DATA_POINTS):
        with open(DATA_POINTS, encoding="utf8") as f:
            data = json.load(f)
    for g in want:
        t = time.time()
        res = minimal_lattices(g, rule="sheet", n=7, time_limit=280) if sheet else points_lattices(g, **BOUNDS)
        d = res.to_dict(); d["build_seconds"] = round(time.time() - t, 1)
        data["groups"][g] = d
        last = res.tiers[-1]
        print(f"# time {g:5s} {d['build_seconds']:6.1f} s  {len(res.all)} lattices, last tier ({last['k']}, {last['n']}) {last['status']}")
        for k, L in enumerate(res.lattices, 1):
            print(f"   #{k} {'; '.join(L.wyckoff()):40s} {' | '.join(L.vertex_types()):40s} own symmetry x{L.symmetry_index}"
                  + (f"  rank {L.rank}" if sheet else f"  phases {L.meta['phases']} ({100 * L.meta['phase_share']:.0f} %)  voronoi {L.meta['voronoi']}")
                  + f"  routes agree {L.meta['routes_agree']}")
        for m in res.notes:
            print("   note:", m)
    with open(DATA_POINTS, "w", encoding="utf8") as f:
        json.dump(data, f, indent=0, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("written", DATA_POINTS)
