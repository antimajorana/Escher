"""Proof (c): lattice-level local cone obstruction theorem for closed surfaces with chi > 0."""
from __future__ import annotations

import math

import numpy as np

from .pieces import _rot
from .refusal import SurfaceImpossible


def _point_in_poly(p, poly):
    """Ray-casting point-in-polygon test for an arbitrary simple polygon."""
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for k in range(n):
        x1, y1 = float(poly[k][0]), float(poly[k][1])
        x2, y2 = float(poly[(k + 1) % n][0]), float(poly[(k + 1) % n][1])
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def _containing_face_sides(lat, c):
    """Return the side count of the specific face in `lat.faces` containing `c`, or None."""
    c = np.asarray(c, float)
    min_a = min(float(np.linalg.norm(lat.A[:, 0])), float(np.linalg.norm(lat.A[:, 1])))
    box = max(3, int(math.ceil((lat.fmax + 1.0) / max(min_a, 1e-6))) + 1)
    for labs, cen in lat.faces:
        base_poly = np.array([lat.pos(l) for l in labs])
        n0 = np.rint(lat.Ainv @ (c - cen)).astype(int)
        for d1 in range(-box, box + 1):
            for d2 in range(-box, box + 1):
                sh = lat.A @ np.array([int(n0[0]) + d1, int(n0[1]) + d2], float)
                if _point_in_poly(c, base_poly + sh):
                    return len(labs)
    return None


def _locate_site(lat, c):
    c = np.asarray(c, float)
    for i in range(lat.m):
        f = lat.Ainv @ (c - lat.basis[i])
        r = np.rint(f).astype(int)
        if np.max(np.abs(f - r)) < 1e-5:
            return (i, int(r[0]), int(r[1]))
    return None


def _site_cone_usable(lat, c, n, k):
    """True iff an order-n cone with Frank angle k*360/n at site `c` passes local 1-ring graph validity."""
    s0 = _locate_site(lat, c)
    if s0 is None:
        return False, None
    i0, n01, n02 = s0
    d = lat.plane_degree[i0]
    if d * (n - k) < 2 * n or (d * (n - k)) % n != 0:
        return False, d
    d_surv = (d * (n - k)) // n
    p0 = lat.pos(s0)
    darts = [dt for dt in lat.darts if dt[0] == i0]
    darts.sort(key=lambda dt: math.atan2(*(lat.pos((dt[1], n01 + dt[2], n02 + dt[3])) - p0)[::-1]))
    faces = []
    for dt in darts:
        f_idx, (o1, o2) = lat.dart_face[dt]
        labs, _ = lat.faces[f_idx]
        sh = (n01 - o1, n02 - o2)
        faces.append([(ii, l1 + sh[0], l2 + sh[1]) for (ii, l1, l2) in labs])
    alpha = 2.0 * math.pi * k / n
    R = _rot(alpha)
    Mf, tf = lat.frac((R, p0 - R @ p0))
    gf = (np.rint(Mf).astype(int), tf)
    m = d // n
    for r0 in range(m):
        surv = [faces[(r0 + t) % d] for t in range(d_surv)]
        b_end = darts[(r0 + d_surv) % d]
        ray_dir = lat.pos((b_end[1], n01 + b_end[2], n02 + b_end[3])) - p0
        ray_u = ray_dir / float(np.linalg.norm(ray_dir))

        def on_cut_ray(lab):
            if lab == s0:
                return True
            v = lat.pos(lab) - p0
            proj = float(v @ ray_u)
            return proj > 1e-7 and float(np.linalg.norm(v - proj * ray_u)) < 1e-5

        def canon_v(lab):
            return s0 if lab == s0 else (lat.apply(gf, lab) if on_cut_ray(lab) else lab)

        ok_wedge = True
        bonds_seen = {}
        for cyc in surv:
            if len({canon_v(u) for u in cyc}) < 3:
                ok_wedge = False
                break
            lc = len(cyc)
            for idx in range(lc):
                u_raw, v_raw = cyc[idx], cyc[(idx + 1) % lc]
                if on_cut_ray(u_raw) and on_cut_ray(v_raw):
                    plane_key = frozenset((canon_v(u_raw), canon_v(v_raw)))
                else:
                    plane_key = frozenset((u_raw, v_raw))
                ua, vb = canon_v(u_raw), canon_v(v_raw)
                if ua == vb:
                    ok_wedge = False
                    break
                ep = tuple(sorted((ua, vb)))
                if ep in bonds_seen and bonds_seen[ep] != plane_key:
                    ok_wedge = False
                    break
                bonds_seen[ep] = plane_key
            if not ok_wedge:
                break
        if ok_wedge:
            return True, d
    return False, d


def check_local_cone_obstruction(lat, cat, what):
    """Proof (c): raise SurfaceImpossible(scope='lattice') if NO (centre, 1 <= k < n) is locally usable."""
    reasons = []
    for rec in cat.centre_classes:
        n, loc, pt = rec["order"], rec["location"], rec["point"]
        if n < 2:
            continue
        if loc in ("bond midpoint", "on a bond"):
            reasons.append(f"{rec['label']} ({loc}): folds bond into a loop")
        elif loc in ("face centre", "inside a face"):
            d = _containing_face_sides(lat, pt)
            if d is None:
                return
            if any(d * (n - k) >= 3 * n and (d * (n - k)) % n == 0 for k in range(1, n)):
                return
            reasons.append(f"{rec['label']} ({loc}, {d}-gon, order {n}): leaves < 3 or non-integer sides")
        elif loc == "site":
            d_site = None
            for k in range(1, n):
                ok, d_site = _site_cone_usable(lat, pt, n, k)
                if ok:
                    return
            deg_str = "?" if d_site is None else str(d_site)
            reasons.append(f"{rec['label']} (site, deg {deg_str}, order {n}): local 1-ring forces loop or double bond")
        else:
            return
    if reasons:
        raise SurfaceImpossible(
            f"{what} is impossible (proved for this lattice): every rotation centre fails local cone checks "
            f"({'; '.join(reasons)})",
            scope="lattice",
        )
