"""groups: ITA Wyckoff positions of the 17 plane groups, and the ITA label of any point.

    wyckoff_table("p4g")            -> rows: letter, multiplicity, site_symmetry, kind, coords, base, D
    label_point("p6m", (1/3, 2/3))  -> {'letter': 'b', 'multiplicity': 2, 'site_symmetry': '3m.', ...}
    label_family(Family.from_cell("kagome"))   -> the position of each site orbit (3c 2mm)
    print(wyckoff_markdown("p4g", occupied=fam))
    proper_centre_classes("p4g")    -> rotation centres in classes under G+ (rotations + translations)

CONVENTIONS
* definition (author, K1): ITA standard settings, hexagonal gamma = 120 deg, cm and cmm in the centred
  rectangular cell. Fractional coordinates are those of the conventional cell; multiplicities are per
  conventional cell, as printed in ITA (centred groups count both centring copies).
* definition (ITA convention for oriented site-symmetry symbols): position 1 is the rotation about the
  normal; positions 2 and 3 are the secondary and tertiary symmetry directions, and a mirror is entered
  under the direction of its NORMAL. Directions: rectangular [10] | [01]; square {[10],[01]} | {[1-1],[11]};
  hexagonal {[10],[01],[11]} | {[1-1],[12],[21]}. One 'm' per class of mirrors of the site group in that
  set (so 2mm. and 2.mm in square groups, 4mm, 3m., 3.m). Oblique groups use the bare symbol 1 or 2.
* definition (my technical choice, documented in proper_centre_classes): when one G-orbit of rotation
  centres splits into two G+-classes, the class containing the FIRST coordinate of the row in
  ITA_TABLES is "(i)", the other "(ii)". (The order of coordinates within a row is from memory and is
  checked by no route; the names are fixed by the coordinates, which tests/test_groups.py pins:)
    p4g  2a(i) (0,0)                2a(ii) (1/2,1/2)          [4-fold]
    pmg  2a(i) (0,0)                2a(ii) (1/2,0)            2b(i) (0,1/2)   2b(ii) (1/2,1/2)
    pgg  2a(i) (0,0)                2a(ii) (1/2,1/2)          2b(i) (0,1/2)   2b(ii) (1/2,0)
    cmm  4c(i) (1/4,1/4),(3/4,3/4)  4c(ii) (3/4,1/4),(1/4,3/4)
    p31m 2b(i) (1/3,2/3)            2b(ii) (2/3,1/3)          [3-fold]

LABELS OF THE CLAIMS
* ITA_TABLES: cited result, ITA Vol. A, Part 6 (the 17 plane groups), one table per plane group,
  locator = plane-group number in the comment. Written from memory (no page checked), so treated as an
  input that must pass routes 2 and 3; nothing in it was changed to fit code output.
* check_table (route 2): computational evidence that every row has the stated multiplicity, oriented
  site symmetry, coordinates and kind, computed from the group operations of
  papertilling.legacy.wallpaper_lattices.WallpaperGroup, and that the table is complete (every fixed
  point of a rotation and every mirror line lies in exactly one row; row count and multiplicities agree
  with legacy minimal_lattices.wyckoff_positions). Route 2 cannot tell apart rows with equal
  multiplicity and site symmetry (e.g. p4m 1a/1b): their letters are checked only by route 3.
* spglib_layer_check (route 3): computational evidence from spglib's layer-group database (ITE Vol. E)
  via the polar layer group G x {1} (sites at z = 0, markers at z = 0.1). Independent of the table.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .legacy.wallpaper_lattices import WallpaperGroup

# ----------------------------------------------------------------------------------------------
# Route 1: the ITA tables (letter, multiplicity, oriented site symmetry, coordinates)
# ----------------------------------------------------------------------------------------------
ITA_NUMBER = {"p1": 1, "p2": 2, "pm": 3, "pg": 4, "cm": 5, "pmm": 6, "pmg": 7, "pgg": 8, "cmm": 9,
              "p4": 10, "p4m": 11, "p4g": 12, "p3": 13, "p3m1": 14, "p31m": 15, "p6": 16, "p6m": 17}
FULL_SYMBOL = {"p1": "p1", "p2": "p2", "pm": "p1m1", "pg": "p1g1", "cm": "c1m1", "pmm": "p2mm",
               "pmg": "p2mg", "pgg": "p2gg", "cmm": "c2mm", "p4": "p4", "p4m": "p4mm", "p4g": "p4gm",
               "p3": "p3", "p3m1": "p3m1", "p31m": "p31m", "p6": "p6", "p6m": "p6mm"}
_SQ8 = ["x,y", "-x,-y", "-y,x", "y,-x"]
_HEX6 = ["x,y", "-y,x-y", "-x+y,-x", "-x,-y", "y,-x+y", "x-y,x"]
_REFL6 = ["-y,-x", "-x+y,y", "x,x-y", "y,x", "x-y,-y", "-x,-x+y"]
ITA_TABLES = {
    # ITA Vol. A, Part 6, plane group No. 1 (p1)
    "p1": [("a", 1, "1", ["x,y"])],
    # No. 2 (p2)
    "p2": [("a", 1, "2", ["0,0"]), ("b", 1, "2", ["0,1/2"]), ("c", 1, "2", ["1/2,0"]),
           ("d", 1, "2", ["1/2,1/2"]), ("e", 2, "1", ["x,y", "-x,-y"])],
    # No. 3 (p1m1)
    "pm": [("a", 1, ".m.", ["0,y"]), ("b", 1, ".m.", ["1/2,y"]), ("c", 2, "1", ["x,y", "-x,y"])],
    # No. 4 (p1g1)
    "pg": [("a", 2, "1", ["x,y", "-x,y+1/2"])],
    # No. 5 (c1m1); coordinates (0,0)+ (1/2,1/2)+
    "cm": [("a", 2, ".m.", ["0,y"]), ("b", 4, "1", ["x,y", "-x,y"])],
    # No. 6 (p2mm)
    "pmm": [("a", 1, "2mm", ["0,0"]), ("b", 1, "2mm", ["0,1/2"]), ("c", 1, "2mm", ["1/2,0"]),
            ("d", 1, "2mm", ["1/2,1/2"]), ("e", 2, "..m", ["x,0", "-x,0"]), ("f", 2, "..m", ["x,1/2", "-x,1/2"]),
            ("g", 2, ".m.", ["0,y", "0,-y"]), ("h", 2, ".m.", ["1/2,y", "1/2,-y"]),
            ("i", 4, "1", ["x,y", "-x,-y", "-x,y", "x,-y"])],
    # No. 7 (p2mg)
    "pmg": [("a", 2, "2..", ["0,0", "1/2,0"]), ("b", 2, "2..", ["0,1/2", "1/2,1/2"]),
            ("c", 2, ".m.", ["1/4,y", "3/4,-y"]), ("d", 4, "1", ["x,y", "-x,-y", "-x+1/2,y", "x+1/2,-y"])],
    # No. 8 (p2gg)
    "pgg": [("a", 2, "2..", ["0,0", "1/2,1/2"]), ("b", 2, "2..", ["0,1/2", "1/2,0"]),
            ("c", 4, "1", ["x,y", "-x,-y", "-x+1/2,y+1/2", "x+1/2,-y+1/2"])],
    # No. 9 (c2mm); coordinates (0,0)+ (1/2,1/2)+
    "cmm": [("a", 2, "2mm", ["0,0"]), ("b", 2, "2mm", ["0,1/2"]), ("c", 4, "2..", ["1/4,1/4", "3/4,1/4"]),
            ("d", 4, "..m", ["x,0", "-x,0"]), ("e", 4, ".m.", ["0,y", "0,-y"]),
            ("f", 8, "1", ["x,y", "-x,-y", "-x,y", "x,-y"])],
    # No. 10 (p4)
    "p4": [("a", 1, "4..", ["0,0"]), ("b", 1, "4..", ["1/2,1/2"]), ("c", 2, "2..", ["1/2,0", "0,1/2"]),
           ("d", 4, "1", _SQ8)],
    # No. 11 (p4mm)
    "p4m": [("a", 1, "4mm", ["0,0"]), ("b", 1, "4mm", ["1/2,1/2"]), ("c", 2, "2mm.", ["1/2,0", "0,1/2"]),
            ("d", 4, "..m", ["x,x", "-x,-x", "-x,x", "x,-x"]), ("e", 4, ".m.", ["x,0", "-x,0", "0,x", "0,-x"]),
            ("f", 4, ".m.", ["x,1/2", "-x,1/2", "1/2,x", "1/2,-x"]),
            ("g", 8, "1", _SQ8 + ["-x,y", "x,-y", "y,x", "-y,-x"])],
    # No. 12 (p4gm)
    "p4g": [("a", 2, "4..", ["0,0", "1/2,1/2"]), ("b", 2, "2.mm", ["1/2,0", "0,1/2"]),
            ("c", 4, "..m", ["x,x+1/2", "-x,-x+1/2", "-x+1/2,x", "x+1/2,-x"]),
            ("d", 8, "1", _SQ8 + ["-x+1/2,y+1/2", "x+1/2,-y+1/2", "y+1/2,x+1/2", "-y+1/2,-x+1/2"])],
    # No. 13 (p3)
    "p3": [("a", 1, "3..", ["0,0"]), ("b", 1, "3..", ["1/3,2/3"]), ("c", 1, "3..", ["2/3,1/3"]),
           ("d", 3, "1", _HEX6[:3])],
    # No. 14 (p3m1)
    "p3m1": [("a", 1, "3m.", ["0,0"]), ("b", 1, "3m.", ["1/3,2/3"]), ("c", 1, "3m.", ["2/3,1/3"]),
             ("d", 3, ".m.", ["x,-x", "x,2x", "-2x,-x"]), ("e", 6, "1", _HEX6[:3] + _REFL6[:3])],
    # No. 15 (p31m)
    "p31m": [("a", 1, "3.m", ["0,0"]), ("b", 2, "3..", ["1/3,2/3", "2/3,1/3"]),
             ("c", 3, "..m", ["x,0", "0,x", "-x,-x"]), ("d", 6, "1", _HEX6[:3] + _REFL6[3:])],
    # No. 16 (p6)
    "p6": [("a", 1, "6..", ["0,0"]), ("b", 2, "3..", ["1/3,2/3", "2/3,1/3"]),
           ("c", 3, "2..", ["1/2,0", "0,1/2", "1/2,1/2"]), ("d", 6, "1", _HEX6)],
    # No. 17 (p6mm)
    "p6m": [("a", 1, "6mm", ["0,0"]), ("b", 2, "3m.", ["1/3,2/3", "2/3,1/3"]),
            ("c", 3, "2mm", ["1/2,0", "0,1/2", "1/2,1/2"]),
            ("d", 6, "..m", ["x,0", "0,x", "-x,-x", "-x,0", "0,-x", "x,x"]),
            ("e", 6, ".m.", ["x,-x", "x,2x", "-2x,-x", "-x,x", "-x,-2x", "2x,x"]),
            ("f", 12, "1", _HEX6 + _REFL6)],
}
_KIND = {0: "point", 1: "line", 2: "general"}
_DIRS = {"oblique": ((), ()), "rectangular": (((1, 0),), ((0, 1),)),
         "square": (((1, 0), (0, 1)), ((1, -1), (1, 1))),
         "hexagonal": (((1, 0), (0, 1), (1, 1)), ((1, -1), (1, 2), (2, 1)))}


def _G(group):
    return group if isinstance(group, WallpaperGroup) else WallpaperGroup(group)


def _A(G):
    return np.column_stack(G.lattice())


def _eqmod(p, q, tol=1e-6):
    d = np.asarray(p, float) - np.asarray(q, float)
    return bool(np.all(np.abs(d - np.round(d)) < tol))


def _affine(s):
    """'x,2x+1/2' -> (base, D, free variable names): the points are base + D u."""
    s = re.sub(r"(\d)([xy])", r"\1*\2", s.replace(" ", ""))
    comps = s.split(",")
    names = [v for v in "xy" if v in s]
    ev = lambda c, x, y: float(eval(c, {"__builtins__": {}}, {"x": x, "y": y}))
    base = np.array([ev(c, 0.0, 0.0) for c in comps])
    D = np.zeros((2, len(names)))
    for j, v in enumerate(names):
        val = {"x": 0.0, "y": 0.0, v: 1.0}
        D[:, j] = [ev(c, val["x"], val["y"]) - base[r] for r, c in enumerate(comps)]
    return base, D, names


def _eval_coord(s, u, names):
    base, D, nm = _affine(s)
    val = dict(zip(names, u))
    return base + D @ np.array([val[n] for n in nm])


def _rows(raw):
    out = []
    for letter, mult, sym, coords in raw:
        base, D, names = _affine(coords[0])
        out.append(dict(letter=letter, multiplicity=mult, site_symmetry=sym, kind=_KIND[D.shape[1]],
                        coords=list(coords), base=base, D=D, free=names))
    return out


def wyckoff_table(group, table=None):
    """ITA Wyckoff positions of `group` (cited result, ITA Vol. A Part 6; see ITA_TABLES).
    One dict per position: letter, multiplicity (conventional cell), site_symmetry (oriented),
    kind ('point' | 'line' | 'general'), coords (ITA strings), base + D u (machine form), free."""
    G = _G(group)
    return _rows(ITA_TABLES[G.symbol] if table is None else table)


# ----------------------------------------------------------------------------------------------
# Route 2 machinery: orbits, stabilisers and oriented site symmetry from the group operations
# ----------------------------------------------------------------------------------------------
def _orbit(G, p, tol=1e-6):
    pts = []
    for M, t in G.ops:
        q = M @ p + t
        q = q - np.floor(q + tol)
        if not any(_eqmod(q, r, tol) for r in pts):
            pts.append(q)
    return pts


def _stabiliser(G, p, tol=1e-6):
    return [(M, t) for M, t in G.ops if _eqmod(M @ p + t, p, tol)]


def site_symmetry(group, p, tol=1e-6):
    """Oriented site-symmetry symbol of the point p, computed from the operations (definition in the
    module docstring). Returns (symbol, number of rotations, number of mirrors)."""
    G = _G(group)
    A = _A(G); Ai = np.linalg.inv(A)
    Ms = [A @ M @ Ai for M, _ in _stabiliser(G, np.asarray(p, float), tol)]
    rots = [M for M in Ms if np.linalg.det(M) > 0]
    mirrors = [M for M in Ms if np.linalg.det(M) < 0]
    if G.system == "oblique":
        return str(len(rots)), len(rots), len(mirrors)
    used = 0
    pos = []
    for dset in _DIRS[G.system]:
        dirs = [A @ np.array(d, float) for d in dset]
        ms = []
        for M in mirrors:
            col = (np.eye(2) - M)[:, int(np.argmax(np.linalg.norm(np.eye(2) - M, axis=0)))]   # the normal
            if any(abs(col[0] * d[1] - col[1] * d[0]) < 1e-8 * np.linalg.norm(d) for d in dirs):
                ms.append(M)
        used += len(ms)
        classes = []                       # classes of mirrors under conjugation by the site rotations
        for M in ms:
            if not any(any(np.allclose(R @ M @ R.T, N, atol=1e-8) for R in rots) for N in classes):
                classes.append(M)
        pos.append("m" * len(classes) if classes else ".")
    if used != len(mirrors):
        raise RuntimeError("a mirror normal is not along a symmetry direction")
    s = (str(len(rots)) if len(rots) > 1 else ".") + "".join(pos)
    return ("1" if set(s) == {"."} else s), len(rots), len(mirrors)


def _in_row(q, row, tol=1e-6):
    if row["kind"] == "general":
        return True
    v = q - row["base"]
    if row["kind"] == "point":
        return _eqmod(v, 0.0, tol)
    d = row["D"][:, 0]
    c = v[0] * d[1] - v[1] * d[0]          # integer iff q lies on base + R d + lattice (d primitive)
    return abs(c - round(c)) < tol


def _rows_of(G, rows, p, tol=1e-6):
    orb = _orbit(G, np.asarray(p, float), tol)
    return [r for r in rows if r["multiplicity"] == len(orb) and any(_in_row(q, r, tol) for q in orb)], orb


def label_point(group, p_frac_conventional, tol=1e-6):
    """ITA Wyckoff position of a point (conventional fractional coordinates).
    Matching is by orbit size first, then by membership of an orbit image in base + D u."""
    G = _G(group)
    hits, orb = _rows_of(G, wyckoff_table(G), p_frac_conventional, tol)
    if len(hits) != 1:
        raise ValueError(f"{G.symbol}: point {tuple(p_frac_conventional)} matches {len(hits)} rows")
    r = hits[0]
    return dict(letter=r["letter"], multiplicity=r["multiplicity"], site_symmetry=r["site_symmetry"],
                kind=r["kind"], name=f"{r['multiplicity']}{r['letter']}")


def label_family(family):
    """ITA position of each site orbit's generating point of a legacy Family."""
    out = []
    for o in range(len(family.gens)):
        p = np.asarray(family.gen_point(o), float)
        out.append(dict(orbit=o, point=tuple(float(x) for x in p), **label_point(family.group, p)))
    return out


def _generic_u(k, shift=0.0):
    return np.array([0.1234 + shift, 0.3141 - shift][:k])


def check_table(group, table=None, tol=1e-6):
    """Route 2. Returns a list of error strings (empty: the table passed every check).
    For each row at a generic point base + D u: orbit size = multiplicity; computed oriented site
    symmetry = stated one; kind matches the number of rotations/mirrors; |stabiliser| x multiplicity =
    |G|; every listed coordinate is in the orbit, they are distinct, and their number x centring =
    multiplicity; the point lies in this row and no other. Completeness: every fixed point of a rotation
    and a generic point of every mirror line lie in exactly one row of the right kind; the number of rows
    and the (kind, multiplicity) multiset agree with legacy minimal_lattices.wyckoff_positions."""
    G = _G(group)
    rows = wyckoff_table(G, table)
    err = []
    nG, cen = len(G.ops), (2 if G.centred else 1)
    letters = [r["letter"] for r in rows]
    if letters != [chr(ord("a") + i) for i in range(len(rows))]:
        err.append(f"letters not a, b, c, ...: {letters}")
    gen = [r for r in rows if r["kind"] == "general"]
    if len(gen) != 1 or rows[-1]["kind"] != "general" or gen[0]["multiplicity"] != nG:
        err.append("there must be exactly one general position, last, of multiplicity |G|")
    for r in rows:
        tag = f"{G.symbol} {r['multiplicity']}{r['letter']}"
        k = r["D"].shape[1]
        if k == 1 and not np.allclose(r["D"], np.round(r["D"])) or (k == 1 and math.gcd(*map(int, np.round(r["D"][:, 0]))) != 1):
            err.append(f"{tag}: line direction not a primitive integer vector")
        u = _generic_u(k)
        p = r["base"] + r["D"] @ u
        orb = _orbit(G, p, tol)
        if len(orb) != r["multiplicity"]:
            err.append(f"{tag}: orbit size {len(orb)} != multiplicity {r['multiplicity']}")
        sym, nrot, nmir = site_symmetry(G, p, tol)
        if sym != r["site_symmetry"]:
            err.append(f"{tag}: site symmetry from ops {sym!r} != table {r['site_symmetry']!r}")
        want = "point" if nrot > 1 else ("line" if nmir else "general")
        if want != r["kind"]:
            err.append(f"{tag}: kind {r['kind']} but stabiliser says {want}")
        if (nrot + nmir) * r["multiplicity"] != nG:
            err.append(f"{tag}: |stabiliser| x multiplicity != |G| = {nG}")
        pts = [_eval_coord(c, u, r["free"]) for c in r["coords"]]
        if len(pts) * cen != r["multiplicity"]:
            err.append(f"{tag}: {len(pts)} coordinates x centring {cen} != multiplicity")
        for c, q in zip(r["coords"], pts):
            if not any(_eqmod(q, o, tol) for o in orb):
                err.append(f"{tag}: coordinate {c!r} is not in the orbit")
        cset = [np.asarray(q) + s for q in pts for s in ([0, 0], [0.5, 0.5])[:cen]]
        if any(_eqmod(cset[i], cset[j], tol) for i in range(len(cset)) for j in range(i)):
            err.append(f"{tag}: listed coordinates coincide (mod lattice and centring)")
        hits, _ = _rows_of(G, rows, p, tol)
        if [h["letter"] for h in hits] != [r["letter"]]:
            err.append(f"{tag}: generic point lies in rows {[h['letter'] for h in hits]}")
    # completeness from the operations
    for kind, p in _special_elements(G):
        hits, _ = _rows_of(G, rows, p, tol)
        if len(hits) != 1 or hits[0]["kind"] != kind:
            err.append(f"{G.symbol}: special {kind} at {np.round(p, 4)} lies in {[(h['letter'], h['kind']) for h in hits]}")
    rng = np.random.default_rng(7)
    for _ in range(3):
        hits, _ = _rows_of(G, rows, rng.uniform(0, 1, 2), tol)
        if [h["kind"] for h in hits] != ["general"]:
            err.append(f"{G.symbol}: random point not in exactly the general row")
    from .legacy.minimal_lattices import wyckoff_positions
    leg = sorted((f["kind"], f["mult"] * cen) for f in wyckoff_positions(G))
    mine = sorted((r["kind"], r["multiplicity"]) for r in rows)
    if leg != mine:
        err.append(f"{G.symbol}: (kind, multiplicity) {mine} != legacy wyckoff_positions {leg}")
    return err


def _special_elements(G):
    """Fixed points of every rotation, and a generic point (p0 + 0.3171 d) of every mirror line."""
    out = []
    for M, t in G.ops:
        det = round(np.linalg.det(M))
        if det == 1 and not np.array_equal(M, np.eye(2, dtype=int)):
            for L in np.ndindex(5, 5):
                c = np.linalg.solve(np.eye(2) - M, t + np.array(L) - 2)
                out.append(("point", c - np.floor(c + 1e-9)))
        elif det == -1:
            P = np.eye(2) + M
            d = P[:, int(np.argmax(np.abs(P).sum(axis=0)))]
            d = np.round(d / math.gcd(*map(int, np.round(d)))).astype(float)
            for L in np.ndindex(5, 5):
                rhs = t + np.array(L) - 2
                x, *_ = np.linalg.lstsq(np.eye(2) - M, rhs, rcond=None)
                if np.allclose((np.eye(2) - M) @ x, rhs, atol=1e-9):
                    out.append(("line", x + 0.3171 * d))
    return out


# ----------------------------------------------------------------------------------------------
# Rotation centres under G+ (rotations and translations only)
# ----------------------------------------------------------------------------------------------
def proper_centre_classes(group, tol=1e-6):
    """Rotation centres of one conventional cell, in classes under G+ (proper operations of G with
    all translations, centring included).

    Known answer (proved result, orbit-stabiliser): G+ has index 1 or 2 in G, and the G-orbit of p is
    one G+-orbit iff G = G+ Stab_G(p), i.e. iff the site group of p contains a mirror or G = G+.
    Otherwise it splits into two G+-classes, swapped by any improper element of G (mirror partners).
    Naming (my technical choice): '2a' when the G-orbit is one class; '2a(i)' = the class containing
    the first coordinate of the row in ITA_TABLES (order within a row unchecked against ITA; see the
    module docstring for the coordinates of every split class), '2a(ii)' = the other. mirror_partner is the name of the image class
    under an improper element of G (None when G has no improper element)."""
    G = _G(group)
    proper = [(M, t) for M, t in G.ops if round(np.linalg.det(M)) == 1]
    improper = [(M, t) for M, t in G.ops if round(np.linalg.det(M)) == -1]
    out = []
    for r in wyckoff_table(G):
        if r["kind"] != "point":
            continue
        pts = _orbit(G, r["base"], tol)
        classes = []
        for q in pts:
            if not any(any(_eqmod(q, c, tol) for c in cl) for cl in classes):
                classes.append([M @ q + t - np.floor(M @ q + t + tol) for M, t in proper])
        classes = [[c for i, c in enumerate(cl) if not any(_eqmod(c, d, tol) for d in cl[:i])] for cl in classes]
        nm = f"{r['multiplicity']}{r['letter']}"
        names = [nm] if len(classes) == 1 else [f"{nm}(i)", f"{nm}(ii)"]
        order = site_symmetry(G, r["base"], tol)[1]
        recs = []
        for cl, name in zip(classes, names):
            recs.append(dict(name=name, letter=r["letter"], multiplicity=r["multiplicity"], order=order,
                             site_symmetry=r["site_symmetry"], rep=tuple(float(x) for x in cl[0]),
                             points=[tuple(float(x) for x in c) for c in cl], g_orbit_splits=len(classes) == 2))
        for rec in recs:
            if not improper:
                rec["mirror_partner"] = None
                continue
            M, t = improper[0]
            img = M @ np.array(rec["rep"]) + t
            rec["mirror_partner"] = next(o["name"] for o in recs if any(_eqmod(img, c, tol) for c in o["points"]))
        out += recs
    return out


# ----------------------------------------------------------------------------------------------
# Notebook display
# ----------------------------------------------------------------------------------------------
def wyckoff_markdown(group, occupied=None):
    """Markdown table of the ITA Wyckoff positions; `occupied` = a legacy Family, a list of
    label dicts (from label_family/label_point) or of letters; occupied rows get the orbit numbers."""
    G = _G(group)
    occ = {}
    if occupied is not None:
        labels = label_family(occupied) if hasattr(occupied, "gens") else list(occupied)
        for i, lab in enumerate(labels):
            letter = lab if isinstance(lab, str) else lab["letter"]
            occ.setdefault(letter, []).append(str(lab.get("orbit", i) + 1 if isinstance(lab, dict) else i + 1))
    cen = " (coordinates (0,0)+ (1/2,1/2)+)" if G.centred else ""
    lines = [f"**{G.symbol}** = {FULL_SYMBOL[G.symbol]}, ITA plane group No. {ITA_NUMBER[G.symbol]}{cen}", "",
             "| Wyckoff | site symmetry | kind | coordinates | occupied (site orbit) |",
             "|---|---|---|---|---|"]
    for r in reversed(wyckoff_table(G)):
        mark = ("**yes** (" + ", ".join(occ[r["letter"]]) + ")") if r["letter"] in occ else ""
        lines.append(f"| {r['multiplicity']}{r['letter']} | `{r['site_symmetry']}` | {r['kind']} | "
                     f"{' '.join('(' + c + ')' for c in r['coords'])} | {mark} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Route 3: spglib polar layer groups (ITE Vol. E)
# ----------------------------------------------------------------------------------------------
# cited result (ITE Vol. E, from memory; confirmed when spglib returns these numbers on the probes):
# the polar layer group isomorphic to each plane group, in the same setting
POLAR_LAYER = {"p1": (1, "p1"), "p2": (3, "p112"), "pm": (11, "pm11"), "pg": (12, "pb11"), "cm": (13, "cm11"),
               "pmm": (23, "pmm2"), "pmg": (24, "pma2"), "pgg": (25, "pba2"), "cmm": (26, "cmm2"),
               "p4": (49, "p4"), "p4m": (55, "p4mm"), "p4g": (56, "p4bm"), "p3": (65, "p3"),
               "p3m1": (69, "p3m1"), "p31m": (70, "p31m"), "p6": (73, "p6"), "p6m": (77, "p6mm")}


def plane_symbol_from_layer(sym, system):
    """Layer site symbol -> plane-group order. Rectangular/oblique layer symbols are ordered
    [100][010][001] (rotation about z last): 'mm2' -> '2mm', 'm..' -> '.m.', '..2' -> '2..'; a bare 'm'
    (pm11, cm11: one mirror, normal [100]) -> '.m.'. Square and hexagonal symbols are already in
    plane-group order."""
    if system in ("square", "hexagonal") or sym in ("1", "2") and system == "oblique":
        return sym
    if sym == "1":
        return "1"
    if sym == "m":
        return ".m."
    if len(sym) == 3:
        return sym[2] + sym[0] + sym[1]
    return sym


def spglib_layer_check(group, seed=0, symprec=1e-5):
    """Route 3: for every row, a probe layer (species 1 on the row's orbit at a random point of the
    row, z = 0; species 2 on a random general orbit, z = 0.1) is passed to spglib.get_layergroup.
    Returns one dict per row with spglib's number, letter and site symbol, whether spglib kept the
    setting (P = 1, p = 0), and the label of the transformed point x_s = P x + p by our table."""
    import spglib
    G = _G(group)
    A = _A(G)
    lat = np.array([[A[0, 0], A[1, 0], 0], [A[0, 1], A[1, 1], 0], [0, 0, 10.0]])
    rng = np.random.default_rng(seed)
    out = []
    for r in wyckoff_table(G):
        u = rng.uniform(0.05, 0.45, r["D"].shape[1])
        p = r["base"] + r["D"] @ u
        pos = [[*q, 0.0] for q in _orbit(G, p)]
        num = [1] * len(pos)
        for q in _orbit(G, rng.uniform(0, 1, 2)):
            pos.append([*q, 0.1]); num.append(2)
        d = spglib.get_layergroup((lat, np.array(pos), num), aperiodic_dir=2, symprec=symprec)
        P, sh = np.array(d.transformation_matrix), np.array(d.origin_shift)
        same = bool(np.allclose(P, np.eye(3)) and _eqmod(sh[:2], 0.0))
        xs = (P @ np.array(pos[0]) + sh)[:2]
        try:
            lab = label_point(G, xs)
            mine_s, mine_sym = lab["letter"], lab["site_symmetry"]
        except ValueError:
            mine_s = mine_sym = None
        spg_sym = plane_symbol_from_layer(d.site_symmetry_symbols[0], G.system)
        out.append(dict(group=G.symbol, row=f"{r['multiplicity']}{r['letter']}", letter=r["letter"],
                        site_symmetry=r["site_symmetry"], layer_number=int(d.number),
                        layer_symbol=d.international, spg_letter=d.wyckoffs[0],
                        spg_site_symbol=d.site_symmetry_symbols[0], spg_site_symbol_plane=spg_sym,
                        same_setting=same, P=P[:2, :2].round(6).tolist(), shift=sh[:2].round(6).tolist(),
                        letter_in_spg_frame=mine_s,
                        agree=(mine_s == d.wyckoffs[0] and spg_sym == mine_sym
                               and int(d.number) == POLAR_LAYER[G.symbol][0])))
    return out
