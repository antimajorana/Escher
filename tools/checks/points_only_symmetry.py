"""Pilot: judge the symmetry from the POINTS alone. For every wallpaper group G and every Wyckoff position,
take the orbit of one point (generic free parameters, generic cell shape allowed by G) with NO bonds, and find
the symmetry group of that set of points. "exactly G" = the orbit is a Bravais-like lattice of G in the strict
sense (all sites equivalent, and the points alone have symmetry G and no more)."""
import warnings, time; warnings.simplefilter("ignore")
import numpy as np
from papertilling.legacy import minimal_lattices as ML
from papertilling.legacy.lattice_families import Family
from papertilling.legacy.wallpaper_lattices import WallpaperGroup
from papertilling.enumerate import _ref_points
from papertilling.groups import label_family
t = time.time(); rng = np.random.default_rng(1)
GROUPS = "p1 p2 pm pg cm pmm pmg pgg cmm p4 p4m p4g p3 p3m1 p31m p6 p6m".split()
for g in GROUPS:
    G = WallpaperGroup(g); wy = ML.wyckoff_positions(G); rows = []
    for k in range(len(wy)):
        try:
            f = Family(G, _ref_points(wy, (k,)))
        except ValueError as err:
            rows.append(f"{wy[k]['name']}: (skipped: {err})"); continue
        got = set()
        for _ in range(3):                                   # three random parameter points
            got.add(f.set(**f.random_values(rng)).group_at())
        name = label_family(f)[0]["name"]
        (grp, m), = got if len(got) == 1 else [("several", 0)]
        rows.append(f"{name}:{'EXACT' if grp == g and m == f.m else grp}")
    print(f"{g:5s} " + "  ".join(rows))
print("# time %.1f s" % (time.time() - t))
