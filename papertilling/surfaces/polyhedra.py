"""The lattice on a sphere or a projective plane: elementary disclinations at the lattice's own rotation centres.

WHY. On a closed surface the Frank angles of the disclinations add up to 360 deg x chi (Gauss-Bonnet), chi = 2 for the
sphere and 1 for the projective plane. A disclination at a point must be a rotation of the lattice's group about it, so
the smallest ones are +360/n at an n-fold centre (the author's choice of 2026-11-14): with n the highest rotation order,
    n = 6  icosahedron   12 x +60      (hemi-icosahedron  6 x +60  on the projective plane)
    n = 4  cube           8 x +90      (hemi-cube         4 x +90)
    n = 3  octahedron     6 x +120     (hemi-octahedron   3 x +120)
    n = 2  pillow         4 x +180     (hemi-pillow       2 x +180)
Each face of the solid is a flat polygon with its corners at rotation centres of one class; the size (h, k) sets its
side T = h b1 + k b2, b1 the shortest translation and b2 = b1 turned by 60 deg (90 deg for the cube), as for the
Goldberg-Coxeter construction of fullerenes (h, k) (cited result: Goldberg 1937; Coxeter 1971). The faces are unfolded
along a spanning tree into the plane and glued (pieces.py); every gluing must be a symmetry of the lattice, or the
construction is refused.

PILLOW. The four faces of a tetrahedron with equal opposite edges (a disphenoid), net = the triangle O, 2U, 2W cut at
the midpoints of its sides, U = h b1, W = k b2 (b1, b2 a reduced basis); size (1, 1) by default.
PROJECTIVE PLANE. The sphere divided by the antipodal map: one face of each antipodal pair, glued to the rest through
the antipodal map where needed (orientation-reversing gluings, so refused for chiral groups and wherever the reversing
gluing is not a reflection or glide of the lattice). The hemi-pillow is not a quotient of the disphenoid net (every
involution of a tetrahedron fixes the midpoint of an edge); it is the rectangle U x W with corners at 2-fold centres
and opposite sides glued by glides (the double cover is a pillowcase), U, W along two perpendicular mirror or glide
lines, |U| = h |u0| / 2, |W| = k |w0| / 2 with u0, w0 the shortest translations along them.

3D: every face is mapped affinely onto the face of the solid and every site pushed radially onto the unit sphere.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ._alt_polyhedra import alternating_matches_centre, build_alternating_rp2, find_split_order4_pair
from ._obstruction import check_local_cone_obstruction
from .pieces import ConeRefused, LatticeData, glue, pairing_test, _rot
from .refusal import (SurfaceImpossible, SurfaceNotFound, no_reversing_reason, no_rotation_reason,  # noqa: F401
                      symmetry_facts)

SOLIDS = {6: "icosahedron", 4: "cube", 3: "octahedron", 2: "pillow"}


# ----------------------------------------------------------------------------------------------------------------
# the solids: 3D vertices and faces (counter-clockwise seen from outside)
# ----------------------------------------------------------------------------------------------------------------
def _solid(name):
    phi = (1 + 5 ** 0.5) / 2
    if name == "icosahedron":
        V = [p for a in (-1, 1) for b in (-phi, phi) for p in ((0, a, b), (a, b, 0), (b, 0, a))]
        m = 3
    elif name == "octahedron":
        V = [tuple(s * (i == j) for j in range(3)) for i in range(3) for s in (1, -1)]
        m = 3
    elif name == "cube":
        V = list(itertools.product((-1, 1), repeat=3))
        m = 4
    elif name == "pillow":
        V = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        m = 3
    else:
        raise ValueError(name)
    V = np.array(V, float)
    D = np.linalg.norm(V[:, None] - V[None], axis=2)
    e = D[D > 1e-9].min()
    adj = (abs(D - e) < 1e-6)
    faces = []
    if m == 3:
        for a, b, c in itertools.combinations(range(len(V)), 3):
            if adj[a, b] and adj[b, c] and adj[a, c]:
                faces.append([a, b, c])
    else:
        for ax in range(3):
            for s in (-1, 1):
                faces.append([i for i in range(len(V)) if V[i, ax] == s])
    out = []
    for f in faces:
        c = V[f].mean(axis=0)
        u = V[f[0]] - c
        w = np.cross(c, u)
        f = sorted(f, key=lambda i: math.atan2((V[i] - c) @ w, (V[i] - c) @ u))
        out.append(f)
    anti = {i: int(np.argmin(np.linalg.norm(V + V[i], axis=1))) for i in range(len(V))} if name != "pillow" else None
    return V, out, anti


# ----------------------------------------------------------------------------------------------------------------
# lattice: rotation centres and the (h, k) basis
# ----------------------------------------------------------------------------------------------------------------
def _catalogue(L):
    from ..defects.catalogue import DefectCatalogue
    return DefectCatalogue(L)


def _short_vectors(lat, k=6):
    vs = [np.array([a, b]) for a in range(-k, k + 1) for b in range(-k, k + 1) if a or b]
    vs.sort(key=lambda v: (round(float(np.linalg.norm(lat.A @ v)), 9), math.atan2(*(lat.A @ v)[::-1]) % (2 * math.pi)))
    return [lat.A @ v for v in vs]


def _basis(lat, n):
    """b1 = a shortest translation, b2 = b1 turned by 60 deg (n = 3, 6) or 90 deg (n = 4); a reduced basis for n = 2."""
    vs = _short_vectors(lat)
    b1 = vs[0]
    if n == 2:
        b2 = next(v for v in vs if b1[0] * v[1] - b1[1] * v[0] > 1e-9)
        return b1, b2
    b2 = _rot(math.radians(60 if n in (3, 6) else 90)) @ b1
    f = lat.Ainv @ b2
    if np.max(np.abs(f - np.rint(f))) > 1e-6:
        raise SurfaceNotFound(f"{lat.L.group}: b1 turned by {60 if n != 4 else 90} deg is not a translation of the lattice, so "
                              f"the {'cube' if n == 4 else 'icosahedron or octahedron'} net does not fit it",
                              hint="use the solid that sphere() picks from the highest rotation order", retry=False)
    return b1, b2


def _order(cat):
    return max((r["order"] for r in cat.centre_classes), default=1)


def _centre_hint(recs):
    ok = [r["label"] for r in recs if r["location"] in ("face centre", "site")]
    if not ok:
        return "no class of this order is at a face centre or a site"
    return "try centre=None (every usable class), or centre=" + " or ".join(repr(x) for x in ok)


def _candidates(cat, n, centre, projective=False):
    """Rotation-centre classes of order n to try, in order: given one, or face centres then sites (default)."""
    recs = [r for r in cat.centre_classes if r["order"] == n]
    if centre is None:
        order = {"face centre": 0, "site": 1}
        return sorted([r for r in recs if r["location"] in order], key=lambda r: order[r["location"]]), \
            [r for r in recs if r["location"] not in order]
    if isinstance(centre, str):
        hit = [r for r in recs if centre in (r["id"], r["label"], r["location"], r.get("gclass"))]
        if not hit and projective and n == 4 and alternating_matches_centre(find_split_order4_pair(cat), centre):
            return [], []
    else:
        rec = cat.centre_class_of(np.asarray(centre, float))
        hit = [] if rec is None else [dict(rec, point=np.asarray(centre, float))]
        if rec is not None and rec["order"] != n:
            raise SurfaceNotFound(f"{np.round(centre, 4).tolist()} is an order-{rec['order']} centre; the {SOLIDS[n]} needs "
                                  f"order {n}", hint=_centre_hint(recs))
    if not hit:
        raise SurfaceNotFound(f"no order-{n} rotation centre class {centre!r} in {cat.group}: classes "
                              + ", ".join(f"{r['id']} {r['label']} ({r['location']})" for r in recs), hint=_centre_hint(recs))
    return hit, []


# ----------------------------------------------------------------------------------------------------------------
# nets
# ----------------------------------------------------------------------------------------------------------------
def _first_face(solid, c0, T):
    if solid == "cube":
        U = _rot(math.pi / 2) @ T
        return [c0, c0 + T, c0 + T + U, c0 + U]
    return [c0, c0 + T, c0 + _rot(math.pi / 3) @ T]


def _unfold(V, faces, first, keep=None):
    """Place faces in the plane along a breadth-first spanning tree. keep(face, placed) decides whether a reached face is
    placed (projective plane: one face of each antipodal pair). Returns {face: {vertex: plane point}}."""
    m = len(faces[0])
    pos = {0: dict(zip(faces[0], first))}
    todo = [0]
    while todo:
        f = todo.pop(0)
        cyc = faces[f]
        for s in range(m):
            u, v = cyc[s], cyc[(s + 1) % m]
            g = next(h for h in range(len(faces)) if h != f and u in faces[h] and v in faces[h])
            if g in pos or (keep is not None and not keep(g, pos)):
                continue
            cg = faces[g]
            j = cg.index(v)
            order = cg[j:] + cg[:j]                          # v, u, ...
            P = [pos[f][v], pos[f][u]]
            for _ in range(m - 2):
                P.append(P[-1] + _rot(2 * math.pi / m) @ (P[-1] - P[-2]))
            pos[g] = dict(zip(order, P))
            todo.append(g)
    return pos


def _pieces_and_pairs(faces, pos, anti=None):
    """Pieces (corner lists from each placed face's cycle) and pairs: sides with the same edge (or antipodal edges)."""
    ids = sorted(pos)
    pieces, sides = [], {}
    for k, f in enumerate(ids):
        cyc = faces[f]
        pieces.append([pos[f][v] for v in cyc])
        for s in range(len(cyc)):
            u, v = cyc[s], cyc[(s + 1) % len(cyc)]
            key = frozenset((u, v))
            if anti is not None:
                key = min(key, frozenset((anti[u], anti[v])), key=sorted)
            sides.setdefault(key, []).append((k, s, u, v))
    pairs = []
    for key, lst in sides.items():
        if len(lst) != 2:
            raise RuntimeError(f"edge {set(key)} is on {len(lst)} sides")
        (k1, s1, u1, v1), (k2, s2, u2, v2) = lst
        tgt = u1 if u1 in (u2, v2) else anti[u1]
        pairs.append(((k1, s1), (k2, s2), tgt == v2))
    return ids, pieces, pairs


def _xyz(S, faces3d, V3, cover=False):
    """Site positions: the affine map of each piece onto its 3D face, then pushed onto the unit sphere."""
    X = np.zeros((len(S.sites), 3))
    net = S.net
    for s, cps in S.site_copies.items():
        k, lab = cps[0]
        for c in cps:
            if c[0] < k:
                k, lab = c
        P = net.P[k]
        Q = V3[faces3d[k]]
        M = np.column_stack([P[1] - P[0], P[2] - P[0]])
        a, b = np.linalg.solve(M, S.lat.pos(lab) - P[0])
        x = Q[0] + a * (Q[1] - Q[0]) + b * (Q[2] - Q[0])
        X[s] = x / np.linalg.norm(x)
    S.xyz = X
    if cover:
        _set_cover(S)


def _set_cover(S):
    """The double cover (the sphere) for plot(): a Surface with every site twice (x and -x), every bond and face lifted
    twice (each lift follows the nearer copy of the next site); colour: one per site, shared by its two lifts."""
    from .base import Surface
    X = S.xyz
    N = len(X)
    XX = np.vstack([X, -X])

    def lift(a, b):                                     # the copy of b next to the copy a (a in 0 .. 2N - 1)
        xa = XX[a]
        return b if np.linalg.norm(xa - X[b]) <= np.linalg.norm(xa + X[b]) else b + N

    bonds = []
    for a, b, v in S.bonds:
        bonds += [(a, lift(a, b), v), (a + N, lift(a + N, b), v)]
    faces = []
    for f in S.faces:
        for start in (f[0], f[0] + N):
            cyc = [start]
            for b in f[1:]:
                cyc.append(lift(cyc[-1], b))
            faces.append(cyc)
    C = Surface("sphere (double cover)", S.lattice, list(S.sites) * 2, np.vstack([S.uv, S.uv]), bonds, faces, 2, True, True,
                S.domain, {}, S.plane_degree)
    C.xyz = XX
    C.colour = np.concatenate([np.arange(N), np.arange(N)]) / max(N, 1)
    C.cones = []
    for c in S.cones:
        if c.get("site") is not None:
            C.cones.append(dict(sites=[c["site"], c["site"] + N], angle=c["angle"]))
        if c.get("face") is not None:
            C.cones.append(dict(faces=[2 * c["face"], 2 * c["face"] + 1], angle=c["angle"]))
    S.cover = C


# ----------------------------------------------------------------------------------------------------------------
# public
# ----------------------------------------------------------------------------------------------------------------
def solid_net(lattice, solid, size=(1, 0), centre=None, projective=False):
    """Pieces and pairs of `solid` ('icosahedron', 'cube', 'octahedron', 'pillow') with corners at the rotation centre
    `centre` (a plane point; default: a centre of the highest order in the catalogue, whatever the solid). For the
    group test alone:
    pieces.pairing_test(lattice, *solid_net(...)[:2])."""
    lat = lattice if isinstance(lattice, LatticeData) else LatticeData(lattice)
    if centre is None:
        recs = _catalogue(lat.L).centre_classes                   # sorted by decreasing order
        if not recs:
            if not symmetry_facts(lat)["rotation_orders"]:
                raise SurfaceImpossible(f"{lat.L.group} has no rotation centre, so no corner (cone) of a solid can sit on one")
            raise SurfaceNotFound(f"the catalogue lists no rotation centre for {lat.L.group}, but the lattice has a rotation "
                                  "on the cell: please report this lattice")
        centre = recs[0]["point"]
    c0 = np.asarray(centre, float)
    h, k = (int(x) for x in size)
    if min(h, k) < 0 or (solid == "pillow" and min(h, k) < 1):
        raise ValueError(f"size {tuple(size)}: h, k >= 0" + (" and both >= 1 for the pillow" if solid == "pillow" else ""))
    if solid == "pillow":
        return _pillow_net(lat, c0, h, k, projective)
    V3, faces, anti = _solid(solid)
    b1, b2 = _basis(lat, {"icosahedron": 6, "octahedron": 3, "cube": 4}[solid])
    T = h * b1 + k * b2
    if np.linalg.norm(T) < 1e-9:
        raise ValueError("size (0, 0)")
    if projective:
        keep = lambda g, pos: anti is not None and all(anti_face(faces, anti, g) != f for f in pos)
        pos = _unfold(V3, faces, _first_face(solid, c0, T), keep)
        ids, pieces, pairs = _pieces_and_pairs(faces, pos, anti)
    else:
        pos = _unfold(V3, faces, _first_face(solid, c0, T))
        ids, pieces, pairs = _pieces_and_pairs(faces, pos)
    return pieces, pairs, [faces[f] for f in ids], V3


def anti_face(faces, anti, f):
    s = {anti[v] for v in faces[f]}
    return next(g for g in range(len(faces)) if set(faces[g]) == s)


def _pillow_net(lat, c0, h, k, projective):
    if not projective:
        b1, b2 = _basis(lat, 2)
        U, W = h * b1, k * b2
        O = c0
        X, Ui, Wi, Y = 0, 1, 2, 3                                  # tetrahedron vertices (see _solid('pillow'))
        pieces = [[O, O + U, O + W], [O + U, O + 2 * U, O + U + W], [O + W, O + U + W, O + 2 * W], [O + U, O + U + W, O + W]]
        ids = [[X, Ui, Wi], [Ui, X, Y], [Wi, Y, X], [Ui, Y, Wi]]
        V3, _, _ = _solid("pillow")
        sides = {}
        for p, cyc in enumerate(ids):
            for s in range(3):
                sides.setdefault(frozenset((cyc[s], cyc[(s + 1) % 3])), []).append((p, s, cyc[s], cyc[(s + 1) % 3]))
        pairs = [((a[0], a[1]), (b[0], b[1]), a[2] == b[3]) for a, b in sides.values()]
        return pieces, pairs, ids, V3
    from .flat import glides
    gl = glides(lat.L)
    pick = None
    for g1, g2 in itertools.combinations(gl, 2):
        a1, a2 = lat.A @ g1["along"], lat.A @ g2["along"]
        if abs(a1 @ a2) < 1e-9:
            pick = (a1, a2)
            break
    if pick is None:
        raise SurfaceNotFound(f"{lat.L.group}: the hemi-pillow needs two perpendicular mirror or glide lines (none found in "
                              "the listed operations)", retry=False)
    U, W = h * pick[0] / 2, k * pick[1] / 2
    if U[0] * W[1] - U[1] * W[0] < 0:
        W = -W
    pieces = [[c0, c0 + U, c0 + U + W, c0 + W]]
    pairs = [((0, 0), (0, 2), False), ((0, 1), (0, 3), False)]
    return pieces, pairs, None, None


def _failed_checks(S, projective):
    ch = S.checks(show=False)
    keys = ("no_loops", "no_double_bonds", "faces_at_least_3", "plane_coordination", "euler", "gauss_bonnet", "net_euler")
    bad = [k for k in keys if not ch[k]]
    if projective and S.orientable:
        bad.append("non_orientable")
    if S.chi != (1 if projective else 2):
        bad.append("chi")
    return bad


def _build(lattice, size, centre, projective):
    lat = lattice if isinstance(lattice, LatticeData) else LatticeData(lattice)
    cat = _catalogue(lat.L)
    n = _order(cat)
    what = ("projective plane" if projective else "sphere")
    facts = symmetry_facts(lat)                                    # the proved refusals first (refusal.py)
    why = []
    if projective and not facts["reversing"]:
        why.append(no_reversing_reason(lat.L.group, what))
    if not facts["rotation_orders"]:
        why.append(no_rotation_reason(lat.L.group, what, 1 if projective else 2))
    if why:
        raise SurfaceImpossible("; and ".join(why))
    if n == 1:
        raise SurfaceNotFound(f"the catalogue lists no rotation centre for {lat.L.group}, but the lattice has a rotation of "
                              f"order {facts['rotation_orders']} on the cell: please report this lattice", retry=False)
    if centre is None:
        check_local_cone_obstruction(lat, cat, what)
    solid = SOLIDS[n]
    if size is None:
        size = (1, 1) if (projective or n == 2) else (1, 0)
    cands, skipped = _candidates(cat, n, centre, projective=projective)
    can_alt = projective and solid == "cube" and alternating_matches_centre(find_split_order4_pair(cat), centre)
    if not cands and not can_alt:
        raise SurfaceNotFound(f"{what} not found for {lat.L.group}: every order-{n} centre is at a bond midpoint (a cone "
                              f"there folds the bond into a loop), so the {'hemi-' if projective else ''}{solid} construction has no corner to use at any "
                              "size", tried=[f"centres {', '.join(r['label'] for r in skipped)} (bond midpoints)"],
                              retry=False)
    errors = []
    for rec in cands:
        try:
            pieces, pairs, f3, V3 = solid_net(lat, solid, size, rec["point"], projective)
            kind = (("hemi-" + solid if solid != "pillow" else "hemi-pillow") + " (projective plane)") if projective else \
                f"{solid} (sphere)"
            S = glue(lat, pieces, pairs, kind=kind, catalogue=cat,
                     params=dict(size=tuple(size), solid=solid, centre=rec["label"], centre_id=rec["id"],
                                 centre_location=rec["location"], tried=[e[0] for e in errors],
                                 skipped=[r["label"] for r in skipped]))
        except SurfaceNotFound as e:
            errors.append((rec["label"], str(e)))
            continue
        bad = _failed_checks(S, projective)
        if bad:
            ch = S.checks(show=False)
            errors.append((rec["label"], f"checks failed ({', '.join(bad)}: V {ch['V']}, E {ch['E']}, F {ch['F']}, chi {S.chi})"))
            continue
        S.params["refused"] = errors
        if projective and solid == "pillow":
            _hemi_pillow_xyz(S, pieces)
        else:
            _xyz(S, f3, V3, cover=projective)
        if projective:
            S.xyz_note = ("double cover (a sphere): antipodal points are one site (same colour); "
                          + ("front and back of a pillowcase" if solid == "pillow" else f"{solid} pushed onto it"))
        else:
            S.xyz_note = f"faces of the {solid} pushed onto the unit sphere (bonds near the cones shortened)"
        return S
    if can_alt:
        alt = build_alternating_rp2(lat, cat, size, centre, errors, skipped, _failed_checks)
        if alt is not None:
            return alt
    ladder = [(1, 1), (1, 2), (2, 1), (2, 2)] if solid == "pillow" else [(1, 0), (1, 1), (2, 0), (2, 1)]
    other = [z for z in ladder if z != tuple(size)][:2]
    err = SurfaceNotFound(f"{what} of {lat.L.group} not found: the {'hemi-' if projective else ''}{solid} construction at "
                          f"size {tuple(size)} was refused at every order-{n} centre class tried",
                          tried=[f"size {tuple(size)}, centre {lab}: {msg}" for lab, msg in errors],
                          hint="try " + " or ".join(f"size={z}" for z in other)
                          + ("" if centre is None else ", or centre=None"))
    err.centres, err.causes = [lab for lab, _ in errors], [msg for _, msg in errors]
    raise err


def _hemi_pillow_xyz(S, pieces):
    P = np.asarray(pieces[0])
    U, W = P[1] - P[0], P[3] - P[0]
    M = np.column_stack([U, W])
    X = np.zeros((len(S.sites), 3))
    for s, cps in S.site_copies.items():
        k, lab = min(cps)
        u, v = np.linalg.solve(M, S.lat.pos(lab) - P[0])
        x, y = 2 * u - 1, 2 * v - 1
        px, py = x * math.sqrt(max(0.0, 1 - y * y / 2)), y * math.sqrt(max(0.0, 1 - x * x / 2))
        X[s] = (px, py, math.sqrt(max(0.0, 1 - px * px - py * py)))
    S.xyz = X
    _set_cover(S)


def sphere(lattice, size=None, centre=None):
    """The lattice on a sphere with the elementary disclinations of its highest rotation order n (icosahedron 12 x 60,
    cube 8 x 90, octahedron 6 x 120, pillow 4 x 180). size (h, k): default (1, 0) ((1, 1) for the pillow). centre: a
    class id ('C1'), label ('1a'), location ('site', 'face centre') or plane point; default: the order-n class at a
    face centre, else at a site (a class whose cones fold the lattice is skipped; see S.params). Returns a ConeSurface.
    Raises SurfaceImpossible for p1, pm, pg, cm (no rotation: proved, refusal.py) and SurfaceNotFound when this
    construction is refused at this size (a gluing that is not a symmetry, cones that fold the lattice); e.hint says
    what to try, and surface_table tries several sizes."""
    return _build(lattice, size, centre, projective=False)


def projective_plane(lattice, size=(1, 1), centre=None):
    """The sphere construction divided by the antipodal map (hemi-icosahedron 6 x 60, hemi-cube 4 x 90,
    hemi-octahedron 3 x 120, hemi-pillow 2 x 180): chi = 1, non-orientable. Accepted only if every gluing, the
    orientation-reversing ones included, is a symmetry of the lattice. xyz: one lift on the double cover; plot() shows
    the whole double cover (antipodal points same colour). Raises SurfaceImpossible for groups with no rotation or no
    mirror or glide line (proved, refusal.py), SurfaceNotFound when this construction is refused at this size."""
    return _build(lattice, size, centre, projective=True)


def frame_control(lattice):
    """Positive control of the frames: the rotation by 360/n about every order-n centre of the catalogue must pass
    the group test on the cell. Returns [(label, passed)]."""
    lat = LatticeData(lattice)
    cat = _catalogue(lat.L)
    out = []
    for r in cat.centre_classes:
        R = _rot(2 * math.pi / r["order"])
        c = np.asarray(r["point"], float)
        out.append((r["label"], lat.test((R, c - R @ c))[0]))
    return out


__all__ = ["sphere", "projective_plane", "solid_net", "frame_control", "ConeRefused", "SurfaceImpossible", "SurfaceNotFound",
           "pairing_test"]
