"""Alternating-centre hemi-cube construction for split order-4 G-orbits on RP2 (p4g)."""
from __future__ import annotations

import math

import numpy as np

from .pieces import glue, pairing_test, _rot
from .refusal import SurfaceNotFound


def find_split_order4_pair(cat):
    """Return (c1, c2) if the catalogue has a pair of order-4 G+-classes in the same G-orbit, else None."""
    recs = [r for r in cat.centre_classes if r["order"] == 4 and r.get("gclass")]
    for i, c1 in enumerate(recs):
        for c2 in recs[i + 1:]:
            if c1["gclass"] == c2["gclass"]:
                return c1, c2
    return None


def alternating_matches_centre(pair, centre):
    """True when `centre` allows combining the split order-4 pair `(c1, c2)` on the projective plane."""
    if pair is None:
        return False
    c1, c2 = pair
    if centre is None:
        return True
    if not isinstance(centre, str):
        return False
    pair_label = f"{c1['label']}+{c2['label']}"
    if centre in (c1.get("gclass"), pair_label):
        return True
    return centre == "face centre" and c1["location"] == "face centre" and c2["location"] == "face centre"


def _minimal_d0_candidates(c1_pt, c2_pt, b1, b2):
    """Shortest lattice-shifted vectors c2 - c1 + m*b1 + n*b2, with the b1+b2 quadrant first."""
    cands = []
    for m in range(-2, 3):
        for n in range(-2, 3):
            d = c2_pt - c1_pt + m * b1 + n * b2
            nm = float(np.linalg.norm(d))
            if nm > 1e-7:
                cands.append((nm, d))
    if not cands:
        return []
    min_norm = min(nm for nm, _ in cands)
    tol = 1e-6 * max(min_norm, 1.0)
    short = []
    for nm, d in cands:
        if abs(nm - min_norm) <= tol and not any(np.linalg.norm(d - q) < 1e-6 for q in short):
            short.append(d)
    short.sort(key=lambda d: (
        not (float(d @ b1) > 1e-9 and float(d @ b2) > 1e-9),
        -float(d @ (b1 + b2)),
        round(-float(d[0]), 6),
        round(-float(d[1]), 6),
    ))
    return short


def _hemi_cube_net(c1_pt, T):
    """Unfold the 3 hemi-cube faces starting from [c1, c1+T, c1+T+R90(T), c1+R90(T)]."""
    from .polyhedra import _pieces_and_pairs, _solid, _unfold, anti_face
    V3, faces, anti = _solid("cube")
    U = _rot(math.pi / 2) @ T
    first = [c1_pt, c1_pt + T, c1_pt + T + U, c1_pt + U]
    keep = lambda g, pos: anti is not None and all(anti_face(faces, anti, g) != f for f in pos)
    pos = _unfold(V3, faces, first, keep)
    ids, pieces, pairs = _pieces_and_pairs(faces, pos, anti)
    return pieces, pairs, [faces[f] for f in ids], V3


def build_alternating_rp2(lat, cat, size, centre, errors, skipped, failed_checks_fn):
    """Try building the alternating-centre hemi-cube RP2 surface when single-class candidates fail."""
    from .polyhedra import _basis, _xyz
    pair = find_split_order4_pair(cat)
    if not alternating_matches_centre(pair, centre):
        return None
    c1, c2 = pair
    pair_label = f"{c1['label']}+{c2['label']}"
    h, k = int(size[0]), int(size[1])
    if h < 1 or k < 1:
        errors.append((pair_label, "alternating-centre hemi-cube requires h >= 1 and k >= 1"))
        return None
    try:
        b1, b2 = _basis(lat, 4)
    except SurfaceNotFound as e:
        errors.append((pair_label, str(e)))
        return None
    c1_pt, c2_pt = np.asarray(c1["point"], float), np.asarray(c2["point"], float)
    d0_cands = _minimal_d0_candidates(c1_pt, c2_pt, b1, b2)
    if not d0_cands:
        errors.append((pair_label, "no offset vector between split order-4 classes"))
        return None
    # Select the base diagonal step d0 on size (1, 1) first so (2, 1) does not wrap back onto (1, 1).
    d0 = None
    first_bad = None
    for cand in d0_cands:
        _, pieces0, pairs0 = None, *_hemi_cube_net(c1_pt, cand)[:2]
        bad0 = [r for r in pairing_test(lat, pieces0, pairs0) if not r["ok"]]
        if not bad0:
            d0 = cand
            break
        if first_bad is None:
            first_bad = bad0
    if d0 is None:
        errors.append((pair_label, f"{len(first_bad)} gluings at (1, 1) are not symmetries of the lattice"))
        return None
    T = d0 + (h - 1) * b1 + (k - 1) * b2
    if float(np.linalg.norm(T)) < 1e-9:
        errors.append((pair_label, "zero edge vector T"))
        return None
    pieces, pairs, f3, V3 = _hemi_cube_net(c1_pt, T)
    loc = c1["location"] if c1["location"] == c2["location"] else f"{c1['location']}+{c2['location']}"
    params = dict(size=(h, k), solid="cube", centre=pair_label, centre_id=f"{c1['id']}+{c2['id']}",
                  centre_location=loc, tried=[e[0] for e in errors],
                  skipped=[r["label"] for r in skipped], alternating=True)
    try:
        S = glue(lat, pieces, pairs, kind="hemi-cube (projective plane)", catalogue=cat, params=params, strict=True)
    except SurfaceNotFound as e:
        errors.append((pair_label, str(e)))
        return None
    bad = failed_checks_fn(S, projective=True)
    if bad:
        errors.append((pair_label, f"internal check(s) {bad} failed"))
        return None
    S.params["refused"] = list(errors)
    _xyz(S, f3, V3, cover=True)
    S.xyz_note = "double cover (a sphere): antipodal points are one site (same colour); cube pushed onto it"
    return S
