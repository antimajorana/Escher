"""Compact TGCell quotients as regular maps of finite groups (pure-Python fallback)."""
from __future__ import annotations

from collections import deque
import json
import math
from typing import Any

from . import _hyperbolic_vc as vc
from ._noneuclid_common import _rs_fn

TGCELL_NAMES = ("bolza_8_8", "bolza_8_3", "klein_7_3", "bring_5_5", "genus2_6_4", "genus2_10_5", "genus2_6_6")


def list_tgcell_quotients() -> list[str]:
    return json.loads(fn()) if (fn := _rs_fn("list_tgcell_quotients_json")) is not None else list(TGCELL_NAMES)


def _orbits(perm: list[int]) -> tuple[list[int], list[list[int]]]:
    comp, cycles = [-1] * len(perm), []
    for d0 in range(len(perm)):
        if comp[d0] == -1:
            cid, cyc, d = len(cycles), [], d0
            while comp[d] == -1:
                comp[d] = cid
                cyc.append(d)
                d = perm[d]
            cycles.append(cyc)
    return comp, cycles


def tgcell_quotient(name: str) -> dict[str, Any]:
    if (fn := _rs_fn("tgcell_quotient_json")) is not None: return json.loads(fn(name))
    key = name.strip().lower().replace("{", "_").replace("}", "_").replace(" ", "_").replace("-", "_")
    key = {"bolza": "bolza_8_8", "klein": "klein_7_3", "bring": "bring_5_5"}.get(key, key)
    m2 = lambda a, b, mod: ((a[0] * b[0] + a[1] * b[2]) % mod, (a[0] * b[1] + a[1] * b[3]) % mod, (a[2] * b[0] + a[3] * b[2]) % mod, (a[2] * b[1] + a[3] * b[3]) % mod)
    n7 = lambda a: a if next((x for x in a if x != 0), 0) <= 3 else tuple((7 - x) % 7 for x in a)
    mz3d4 = lambda e1, e2: ((e1[0] + (1 if e1[1] % 2 == 0 else 2) * e2[0]) % 3, (e1[1] + (e2[1] if e1[2] == 0 else (4 - e2[1]) % 4)) % 4, e1[2] ^ e2[2])
    iz3d4 = lambda e: (lambda ni: ((3 - ((1 if ni % 2 == 0 else 2) * e[0]) % 3) % 3, ni, e[2]))((4 - e[1]) % 4 if e[2] == 0 else e[1])
    specs = {
        "bolza_8_8": ("Z_8", 8, 8, 0, 4, 1, lambda a, b: (a + b) % 8, lambda a: (8 - a) % 8),
        "bolza_8_3": ("GL(2,3)", 8, 3, (1, 0, 0, 1), (1, 0, 1, 2), (1, 1, 0, 1), lambda a, b: m2(a, b, 3), lambda a: (lambda d: ((d * a[3]) % 3, (d * (3 - a[1])) % 3, (d * (3 - a[2])) % 3, (d * a[0]) % 3))((a[0] * a[3] + 9 - a[1] * a[2]) % 3)),
        "klein_7_3": ("PSL(2,7)", 7, 3, (1, 0, 0, 1), n7((0, 6, 1, 0)), n7((0, 6, 1, 1)), lambda a, b: n7(m2(a, b, 7)), lambda a: n7((a[3] % 7, (7 - a[1]) % 7, (7 - a[2]) % 7, a[0] % 7))),
        "bring_5_5": ("A_5", 5, 5, (0, 1, 2, 3, 4), (0, 3, 4, 1, 2), (1, 2, 3, 4, 0), lambda a, b: tuple(a[b[i]] for i in range(5)), lambda a: tuple(a.index(i) for i in range(5))),
        "genus2_6_4": ("Z_3 |> D_4", 6, 4, (0, 0, 0), (0, 1, 1), (1, 1, 0), mz3d4, iz3d4),
        "genus2_10_5": ("Z_10", 10, 5, 0, 5, 2, lambda a, b: (a + b) % 10, lambda a: (10 - a) % 10),
        "genus2_6_6": ("Z_2 x Z_6", 6, 6, (0, 0), (1, 0), (0, 1), lambda a, b: ((a[0] + b[0]) % 2, (a[1] + b[1]) % 6), lambda a: (a[0], (6 - a[1]) % 6)),
    }
    if key not in specs:
        raise ValueError(f"unknown TGCell quotient {name!r}")
    glab, p, q, id_el, x, y, mul, inv = specs[key]
    els, imap, dq, yi = [id_el], {id_el: 0}, deque([id_el]), inv(y)
    while dq:
        g = dq.popleft()
        for go in (x, y, yi):
            if (h := mul(g, go)) not in imap: imap[h] = len(els); els.append(h); dq.append(h)
    n = len(els); rev, nxt, prv = [imap[mul(e, x)] for e in els], [imap[mul(e, yi)] for e in els], [imap[mul(e, y)] for e in els]
    nf_d = [prv[rev[d]] for d in range(n)]
    _, f_darts = _orbits(nf_d); sd_of = [(-1, -1)] * n
    for fi, cyc in enumerate(f_darts):
        for s_i, d in enumerate(cyc): sd_of[d] = (fi, s_i)
    v_of, v_orbs = _orbits(nxt); nv = len(v_orbs)
    e_of, edges = [-1] * n, []
    for d in range(n):
        if e_of[d] == -1: r = rev[d]; e_of[d] = e_of[r] = len(edges); edges.append([v_of[d], v_of[r]])
    faces = [[v_of[d] for d in cyc] for cyc in f_darts]; base = vc.BaseFace(p, q)
    rot3 = lambda th: ((math.cos(th), -math.sin(th), 0.0), (math.sin(th), math.cos(th), 0.0), (0.0, 0.0, 1.0))
    refl3 = lambda th: ((math.cos(2 * th), math.sin(2 * th), 0.0), (math.sin(2 * th), -math.cos(2 * th), 0.0), (0.0, 0.0, 1.0))
    step = lambda k1, k2: vc.mat_mul(base.reflections[k1], vc.mat_mul(refl3(2 * math.pi * (k1 + 0.5) / p), rot3(2 * math.pi * (k1 - k2) / p)))
    placed, fq, tdarts = [None] * len(faces), deque([0]), set(); placed[0] = vc._IDENTITY
    while fq:
        f = fq.popleft()
        for k, d in enumerate(f_darts[f]):
            rd = rev[d]; f2, k2 = sd_of[rd]
            if placed[f2] is None: tdarts.update((d, rd)); placed[f2] = vc.mat_mul(placed[f], step(k, k2)); fq.append(f2)
    ds = next(d for d in range(n) if d not in tdarts); bd, bpos, d = [], {}, ds
    while True:
        bpos[d] = len(bd); bd.append(d); cur = nf_d[d]
        while cur in tdarts: cur = nf_d[rev[cur]]
        if (d := cur) == ds: break
    linv = lambda m: ((m[0][0], m[1][0], -m[2][0]), (m[0][1], m[1][1], -m[2][1]), (-m[0][2], -m[1][2], m[2][2]))
    pairs = [{"side_a": sa, "side_b": bpos[rev[da]], "edge_index": e_of[da], "lorentz_matrix": [list(r) for r in vc.mat_mul(vc.mat_mul(placed[sd_of[da][0]], step(sd_of[da][1], sd_of[rev[da]][1])), linv(placed[sd_of[rev[da]][0]]))]} for sa, da in enumerate(bd) if sa < bpos[rev[da]]]
    chi = nv - len(edges) + len(faces)
    return {"name": key, "group_label": glab, "group_order": n, "p": p, "q": q, "genus": (2 - chi) // 2, "chi": chi,
            "n_vertices": nv, "n_edges": len(edges), "n_faces": len(faces), "faces": faces, "edges": edges, "side_pairings": pairs,
            "fundamental_polygon_poincare": [vc.hyperboloid_to_poincare(vc.mat_vec(placed[sd_of[xd][0]], base.vertices[sd_of[xd][1]])) for xd in bd],
            "placed_faces_poincare": [[vc.hyperboloid_to_poincare(vc.mat_vec(m, v)) for v in base.vertices] for m in placed], "provenance": "[PROVED]"}
