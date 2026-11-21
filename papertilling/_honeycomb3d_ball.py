"""Paracompact H³ honeycombs as a ball truncation (C1 WP-G2; mirrors `honeycomb3d/paracompact_ball.rs`).

Every face whose centre lies within hyperbolic distance R of the seed point (the central cell's centre, or
the central face's centre when cells are infinite). Faces are found through the faces consecutive around
each edge (g h R2 F0 and g h R3 R2 F0, h in <R0,R1>) of the whole honeycomb, searching to R + D (D the step
between consecutive face centres); that this reaches
every face within R is computational evidence (unchanged at R + 2D), not a proof.
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import _rust
from ._honeycomb3d_growth import IDEAL_VERTEX_REASON, INFINITE_CELL_REASON, KeyIndex

MAX_BALL_FACES = 20_000


def _group(gens, probe):
    idx = KeyIndex(4, 1e-9)
    idx.insert(probe)
    out, i = [np.eye(4)], 0
    while i < len(out):
        for s in gens:
            h = out[i] @ s
            if idx.insert(h @ probe)[1]:
                out.append(h)
        i += 1
    return out


def _spatial(x):
    return np.array([x[1], x[2], x[3], 0.0])


def _direction(x):
    s = x[1:] / np.linalg.norm(x[1:])
    return np.append(s, 0.0)


def _py_ball(p, q, r, radius, search_steps=1):
    from .honeycomb3d import _Coxeter, _py_classify
    cls = _py_classify(p, q, r)
    if cls["regime"] != "HyperbolicParacompactH3":
        raise ValueError(f"{{{p},{q},{r}}} is not paracompact: grow it by layers instead ({cls['reason']})")
    if not (math.isfinite(radius) and radius >= 0):
        raise ValueError(f"radius must be a finite number >= 0, got {radius}")
    cox = _Coxeter(p, q, r)
    rr, w = cox.refl, cox.w
    xi = w.sum(axis=0)
    steps = [h @ k for h in _group([rr[0], rr[1]], xi) for k in (rr[2], rr[3] @ rr[2])]
    eta = np.array([-1.0, 1, 1, 1])
    step = max(math.acosh(max(1.0, -float(np.sum(eta * (s @ w[2]) * w[2])))) for s in steps)
    search = radius + search_steps * step
    idx = KeyIndex(3, 1e-4)
    idx.insert(_spatial(w[2]))
    faces, i = [np.eye(4)], 0
    while i < len(faces):
        g = faces[i]
        i += 1
        for s in steps:
            h = g @ s
            x = h @ w[2]
            if x[0] <= math.cosh(search) and idx.insert(_spatial(x))[1]:
                faces.append(h)
                if len(faces) > MAX_BALL_FACES:
                    raise ValueError(f"a ball of radius {radius} explores more than {MAX_BALL_FACES} faces; lower R")
    if math.isfinite(idx.min_gap):
        raise ValueError("face keys came closer than 100 x tolerance (precision)")
    kept = [g for g in faces if (g @ w[2])[0] <= math.cosh(radius)]
    inf, ideal = cls["has_infinite_euclidean_cells"], cls["has_ideal_vertices"]
    vidx, cidx = KeyIndex(3, 1e-9 if ideal else 1e-4), KeyIndex(3, 1e-9 if inf else 1e-4)
    rot = rr[0] @ rr[1]
    coords, fcyc, fid, edges, eset, cell_faces = [], [], {}, [], set(), []
    for g in kept:
        cyc, h = [], g
        for _ in range(p):
            x = h @ w[0]
            vid, new = vidx.insert(_direction(x) if ideal else _spatial(x))
            if new:
                coords.append(x)
            cyc.append(vid)
            h = h @ rot
        key = tuple(sorted(cyc))
        if key not in fid:
            fid[key] = len(fcyc)
            fcyc.append(cyc)
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                e = (min(a, b), max(a, b))
                if e not in eset:
                    eset.add(e)
                    edges.append(list(e))
        f = fid[key]
        for m in (g, g @ rr[3]):
            c = m @ w[3]
            cid, new = cidx.insert(_direction(c) if inf else _spatial(c))
            if new:
                cell_faces.append([])
            cell_faces[cid].append(f)
    if math.isfinite(vidx.min_gap) or math.isfinite(cidx.min_gap):
        raise ValueError("vertex or cell keys came closer than 100 x tolerance (precision)")
    full = math.inf if inf else 4 * q // (2 * p + 2 * q - p * q)
    cx = {"name": f"{{{p},{q},{r}}}", "regime": "HyperbolicParacompactH3", "coords4": [], "points": [], "klein": [],
          "is_ideal": [], "edges": edges, "faces": fcyc, "cells": cell_faces, "cell_shell": [0] * len(cell_faces),
          "cell_truncated": [len(fs) < full for fs in cell_faces]}
    for x in coords:
        if ideal:
            s = list(_direction(x)[:3])
            cx["coords4"].append([1.0, *s])
            cx["points"].append(s)
            cx["klein"].append(s)
        else:
            y = x / math.sqrt(x[0] ** 2 - x[1:] @ x[1:])
            cx["coords4"].append(list(y))
            cx["points"].append(list(y[1:] / (1 + y[0])))
            cx["klein"].append(list(y[1:] / y[0]))
        cx["is_ideal"].append(ideal)
    return {"name": cx["name"], "radius": radius, "faces": len(kept), "cells": len(cell_faces), "step": step,
            "reason": INFINITE_CELL_REASON if inf else IDEAL_VERTEX_REASON, "complex": cx}


def paracompact_ball(p: int, q: int, r: int, radius: float) -> dict:
    """Ball truncation of a paracompact H³ honeycomb; same JSON as `honeycomb3d_paracompact_ball_json`."""
    rs = _rust.get()
    if rs is not None and hasattr(rs, "honeycomb3d_paracompact_ball_json"):
        return json.loads(rs.honeycomb3d_paracompact_ball_json(p, q, r, float(radius)))
    return _py_ball(p, q, r, float(radius))
