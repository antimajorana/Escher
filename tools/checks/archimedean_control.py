"""Known-answer control for the lattice search (independent of the code: the 11 Archimedean tilings,
cited result: Gruenbaum & Shephard, Tilings and Patterns, ch. 2 -- vertex types and symmetry groups
quoted from memory, check the book).

Every Archimedean tiling is a planar lattice with ONE site orbit whose symmetry group is exactly
the group listed below. So the search of that group, restricted to k = 1 site orbit, must find a
lattice with that vertex type (the face sizes round a site).  A missing one = a hole in the search.

    python tools/checks/archimedean_control.py            (about 1 minute)
"""
import sys, time
from collections import Counter
from fractions import Fraction
from papertilling.enumerate import search

EXPECTED = {   # group: vertex types (sorted face sizes round a site)
    "p6m": ["3.3.3.3.3.3", "6.6.6", "3.3.6.6", "3.4.4.6", "3.12.12", "4.6.12"],
    "p6": ["3.3.3.3.6"],
    "p4m": ["4.4.4.4", "4.8.8"],
    "p4g": ["3.3.3.4.4"],      # 3^2.4.3.4 (snub square)
    "cmm": ["3.3.3.4.4"],      # 3^3.4^2 (elongated triangular)
}


def vertex_type(key):
    """Face sizes round a site, from the D-symbol key (rows: m01, m12, s0, s1, s2) of a 1-site-orbit
    lattice: the chambers at one site are 2 x degree; D elements with face size f are a fixed fraction
    (1/|site stabiliser|) of the chambers with that face, so corners(f) = degree x count(f) / size."""
    deg = key[0][1]
    c = Counter(r[0] for r in key)
    out = []
    for f, m in sorted(c.items()):
        n = Fraction(deg * m, len(key))
        assert n.denominator == 1, (f, n)
        out += [f] * int(n)
    return ".".join(map(str, out))


bad = 0
for g, want in EXPECTED.items():
    t0 = time.time()
    found = Counter()
    for n in range(1, 4):                                   # the 11 tilings have at most 3 bond orbits
        r = search(g, only_tier=(1, n), stop=False, time_limit=300)
        for e in r["entries"]:
            found[vertex_type(e.key)] += 1
    miss = [w for w in want if w not in found]
    bad += len(miss)
    print(f"{g:4s} k=1, n<=3: {sum(found.values())} lattices; vertex types {dict(found)}")
    print(f"     expected {want}: {'ALL FOUND' if not miss else 'MISSING ' + str(miss)}   # time {time.time() - t0:.1f} s")
print("CONTROL", "PASSED" if not bad else "FAILED")
sys.exit(1 if bad else 0)
