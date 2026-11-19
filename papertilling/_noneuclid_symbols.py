"""Conway orbifold symbols, wallpaper bijection and vertex configurations (pure-Python fallback).

Orbifold Euler characteristic chi = 2 - cost with cost o = 2, * = x = 1, cone n = (n-1)/n, corner n =
(n-1)/(2n) (Conway 1992, LMS Lecture Notes 165; Conway, Burgiel & Goodman-Strauss, The Symmetries of
Things, 2008, Part I). The 17 wallpaper groups are the 17 orbifolds with chi = 0 (table below). Bad
orbifolds n, mn (m != n), *n, *mn (m != n) are refused (Thurston, 1980 notes, ch. 13).
Vertex configurations: angle defect pi (2 - m + 2 sum 1/n_i); necessary conditions refused with a
reason: odd-gon parity (Grunbaum & Shephard, Tilings and Patterns, 1987, section 2.1) and, on S^2,
Descartes V * Delta = 4 pi with V an integer.
"""
from __future__ import annotations

from collections import deque
from fractions import Fraction
import json
import math
import re
from typing import Any

from ._noneuclid_common import _dihedral_canon, _rs_fn
from .groups import ITA_NUMBER

WALLPAPER_ORBIFOLD_TABLE: tuple[tuple[str, str], ...] = (
    ("p6m", "*632"), ("p6", "632"), ("p4m", "*442"), ("p4", "442"), ("p4g", "4*2"),
    ("p3m1", "*333"), ("p31m", "3*3"), ("p3", "333"), ("pmm", "*2222"), ("pmg", "22*"),
    ("pgg", "22x"), ("cmm", "2*22"), ("pm", "**"), ("pg", "xx"), ("cm", "*x"),
    ("p2", "2222"), ("p1", "o"),
)


_WP_ALIASES = {"p1m1": "pm", "p1g1": "pg", "c1m1": "cm", "p2mm": "pmm", "p2mg": "pmg", "p2gg": "pgg", "c2mm": "cmm", "p4mm": "p4m", "p4gm": "p4g", "p6mm": "p6m"}


def wallpaper_orbifold_table() -> list[tuple[str, str]]:
    if (fn := _rs_fn("wallpaper_orbifold_table_json")) is not None:
        return [tuple(p) for p in json.loads(fn())]
    return list(WALLPAPER_ORBIFOLD_TABLE)


def wallpaper_to_orbifold(group: str) -> str:
    if (fn := _rs_fn("wallpaper_to_orbifold")) is not None:
        return str(fn(group))
    g = _WP_ALIASES.get(s := group.strip().lower().replace(" ", ""), s)
    if g not in ITA_NUMBER:
        raise ValueError(f"unknown wallpaper group {group!r}")
    return dict(WALLPAPER_ORBIFOLD_TABLE)[g]


def orbifold_to_wallpaper(symbol: str) -> str:
    if (fn := _rs_fn("orbifold_to_wallpaper")) is not None:
        return str(fn(symbol))
    if (wg := parse_orbifold(symbol)["wallpaper_group"]) is None:
        raise ValueError(f"orbifold {symbol!r} is not a Euclidean wallpaper group")
    return wg


_ORB_WP = {
    (0, 0, 1, (), (6, 3, 2)): "p6m", (0, 0, 0, (6, 3, 2), ()): "p6", (0, 0, 1, (), (4, 4, 2)): "p4m", (0, 0, 0, (4, 4, 2), ()): "p4",
    (0, 0, 1, (4,), (2,)): "p4g", (0, 0, 1, (), (3, 3, 3)): "p3m1", (0, 0, 1, (3,), (3,)): "p31m", (0, 0, 0, (3, 3, 3), ()): "p3",
    (0, 0, 1, (), (2, 2, 2, 2)): "pmm", (0, 0, 1, (2, 2), ()): "pmg", (0, 1, 0, (2, 2), ()): "pgg", (0, 0, 1, (2,), (2, 2)): "cmm",
    (0, 0, 2, (), ()): "pm", (0, 2, 0, (), ()): "pg", (0, 1, 1, (), ()): "cm", (0, 0, 0, (2, 2, 2, 2), ()): "p2", (1, 0, 0, (), ()): "p1",
}


def _canon_desc(vals: list[int]) -> tuple[int, ...]:
    n, r_s = len(vals), list(reversed(vals))
    return tuple(vals) if n <= 1 else max([tuple(vals[r:] + vals[:r]) for r in range(n)] + [tuple(r_s[r:] + r_s[:r]) for r in range(n)])


def parse_orbifold(symbol: str) -> dict[str, Any]:
    if (fn := _rs_fn("parse_orbifold_json")) is not None:
        return json.loads(fn(symbol))
    if not (t := symbol.strip().replace("×", "x")):
        raise ValueError("empty orbifold symbol")
    if t == "1":
        return {"input": symbol, "canonical": "1", "handles": 0, "crosscaps": 0, "boundary_components": 0, "cones": [], "corners": [],
                "chi_num": 2, "chi_den": 1, "chi_orb": 2.0, "geometry_kind": "Spherical", "wallpaper_group": None, "provenance": "[PROVED]"}
    handles = crosscaps = boundaries = 0
    cones, corners, in_mir = [], [], False
    toks = re.findall(r"\([^)]*\)|[*oxOX]|[0-9.,\s]+", t)
    if "".join(toks) != t:
        raise ValueError(f"unexpected syntax in orbifold symbol {symbol!r}")
    for tok in toks:
        if tok == "*": boundaries += 1; in_mir = True
        elif tok in "oO": handles += 1; in_mir = False
        elif tok in "xX": crosscaps += 1; in_mir = False
        else:
            inner = tok[1:-1] if tok.startswith("(") else tok
            vals = [int(p) for p in re.split(r"[.,\s]+", inner.strip()) if p] if any(c in inner for c in "., \t") else ([int(inner.strip())] if tok.startswith("(") else [int(c) for c in inner.strip()])
            if any(v < 2 for v in vals):
                raise ValueError(f"orbifold orders must be >= 2 in {symbol!r}")
            (corners if in_mir else cones).extend(vals)
    if handles == 0 and crosscaps == 0 and ((boundaries == 0 and len(cones) == 1) or (boundaries == 1 and not cones and len(corners) == 1) or (boundaries == 0 and len(cones) == 2 and cones[0] != cones[1]) or (boundaries == 1 and not cones and len(corners) == 2 and corners[0] != corners[1])):
        raise ValueError(f"[PROVED: bad orbifold — no manifold cover]: {symbol!r} is a Thurston bad orbifold")
    chi = Fraction(2 - 2 * handles - crosscaps - boundaries, 1) - sum((Fraction(n - 1, n) for n in cones), Fraction(0)) - sum((Fraction(m - 1, 2 * m) for m in corners), Fraction(0))
    cor_d = _canon_desc(corners)
    wg = _ORB_WP.get((handles, crosscaps, boundaries, tuple(sorted(cones, reverse=True)), cor_d))
    fmt = lambda seq: "".join(map(str, seq)) if all(x < 10 for x in seq) else "(" + ",".join(map(str, seq)) + ")"
    canon = dict(WALLPAPER_ORBIFOLD_TABLE)[wg] if wg else ("o" * handles + fmt(sorted(cones, reverse=True)) + ("*" + fmt(list(cor_d)) + "*" * (boundaries - 1) if boundaries else "") + "x" * crosscaps)
    return {"input": symbol, "canonical": canon, "handles": handles, "crosscaps": crosscaps, "boundary_components": boundaries,
            "cones": cones, "corners": corners, "chi_num": chi.numerator, "chi_den": chi.denominator, "chi_orb": float(chi),
            "geometry_kind": "Spherical" if chi > 0 else ("Euclidean" if chi == 0 else "Hyperbolic"), "wallpaper_group": wg, "provenance": "[PROVED]"}


def parse_vertex_config(symbol: str) -> dict[str, Any]:
    if (fn := _rs_fn("parse_vertex_config_json")) is not None:
        return json.loads(fn(symbol))
    try:
        seq = [int(x.strip()) for x in symbol.strip().strip("()[]").split(".")]
    except ValueError as exc:
        raise ValueError(f"invalid vertex configuration {symbol!r}: {exc}") from None
    if len(seq) < 3 or any(v < 3 for v in seq):
        raise ValueError(f"invalid vertex configuration {symbol!r}")
    m = len(seq)
    for k in sorted({v for v in seq if v % 2 == 1}):
        adj, loop = {}, False
        for i, v in enumerate(seq):
            if v == k:
                a, b = seq[(i - 1) % m], seq[(i + 1) % m]
                if a == b: loop = True; break
                adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
        if not loop:
            ok = False
            for s in adj:
                dist, q = {(s, 0): 0}, deque([(s, 0)])
                while q:
                    u, p = q.popleft()
                    if (d := dist[(u, p)]) < k:
                        for nb in adj.get(u, ()):
                            if (st := (nb, p ^ 1)) not in dist:
                                dist[st] = d + 1; q.append(st)
                if (s, 1) in dist:
                    ok = True; break
            if not ok:
                raise ValueError(f"[PROVED: odd-gon parity obstruction]: odd {k}-gon fails alternating rule in {symbol!r}")
    canon, is_chiral = _dihedral_canon(seq)
    # sr{p,q} = 3.3.p.3.q is chiral for p != q; the snub square tiling 3.3.4.3.4 is achiral (p4g).
    is_snub = is_chiral or (len(canon) == 5 and canon[:2] == [3, 3] and canon[3] == 3 and canon[4] >= 4
                            and (canon[2] == 3 or (canon[2] >= 4 and canon[2] != canon[4])))
    def_pi = Fraction(2 - m, 1) + sum(Fraction(2, v) for v in seq)
    if def_pi > 0 and (4 / def_pi).denominator != 1:
        raise ValueError(f"[PROVED: Descartes obstruction]: {'.'.join(map(str, canon))} has angle defect {def_pi} pi; "
                         "V = 4 pi / Delta is not an integer")
    return {"input": symbol, "canonical": ".".join(map(str, canon)), "sequence": canon, "is_chiral": is_chiral,
            "is_snub_chiral": is_snub, "angle_defect_rad": math.pi * float(def_pi), "angle_defect_pi_num": def_pi.numerator,
            "angle_defect_pi_den": def_pi.denominator, "geometry_kind": "Spherical" if def_pi > 0 else ("Euclidean" if def_pi == 0 else "Hyperbolic"),
            "realizable": True, "provenance": "[PROVED] exact angle defect; realisability: necessary local conditions only"}
