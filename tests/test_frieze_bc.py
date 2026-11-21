"""Frieze strips (C1 WP-B2): glued seams (C3) and [PROVED] Möbius refusals (C2).

Known answers are hand-derived: the standard frieze table (which isometry types each group has),
the Möbius rule (a horizontal mirror or a glide is needed), and V = 3N, E = 5N, F = 2N for an
N-column, 3-row closed quad strip. All checks measure the built patch.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import pytest

from papertilling import _rust
from papertilling import subperiodic as sp


@pytest.fixture(autouse=True)
def _python_only(monkeypatch):
    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")


def _complex(p):
    """(V, E, F, sorted boundary cycle lengths, orientable) from the quads."""
    sides = defaultdict(list)
    for i, q in enumerate(p["quads"]):
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            sides[(min(a, b), max(a, b))].append(i)
    verts = {v for q in p["quads"] for v in q}
    adj = defaultdict(list)
    for (a, b), qs in sides.items():
        if len(qs) == 1:
            adj[a].append(b)
            adj[b].append(a)
    assert all(len(n) == 2 for n in adj.values())
    seen, cycles = set(), []
    for s in adj:
        if s in seen:
            continue
        seen.add(s)
        prev, cur, n = s, adj[s][0], 1
        while cur != s:
            seen.add(cur)
            prev, cur = cur, (adj[cur][1] if adj[cur][0] == prev else adj[cur][0])
            n += 1
        cycles.append(n)
    # Orientability: propagate quad orientations across shared sides.
    def fwd(q, a, b):
        return any(q[k] == a and q[(k + 1) % 4] == b for k in range(4))
    sign = {}
    ok = True
    for root in range(len(p["quads"])):
        if root in sign:
            continue
        sign[root], stack = True, [root]
        while stack and ok:
            i = stack.pop()
            for (a, b), qs in sides.items():
                if i not in qs or len(qs) != 2:
                    continue
                j = qs[1] if qs[0] == i else qs[0]
                same = fwd(p["quads"][i], a, b) == fwd(p["quads"][j], a, b)
                want = (not sign[i]) if same else sign[i]
                if j not in sign:
                    sign[j] = want
                    stack.append(j)
                elif sign[j] != want:
                    ok = False
    return len(verts), len(sides), len(p["quads"]), sorted(cycles), ok


def test_closed_strips_hand_counts_and_topology():
    for sym in sp.FRIEZE_SYMBOLS:
        for cells in (4, 7):
            cyl = sp.build_frieze_patch(sym, cells, "Cylinder")
            n = cyl["columns"]
            assert n == 2 * cells
            assert _complex(cyl) == (3 * n, 5 * n, 2 * n, [n, n], True)
            try:
                mob = sp.build_frieze_patch(sym, cells, "Mobius")
            except ValueError as e:
                assert str(e).startswith("[PROVED]")
                continue
            n = mob["columns"]
            assert _complex(mob) == (3 * n, 5 * n, 2 * n, [2 * n], False)


def test_mobius_refusal_set_and_glide_length():
    refused = []
    for sym in sp.FRIEZE_SYMBOLS:
        try:
            sp.build_frieze_patch(sym, 8, "Mobius")
        except ValueError:
            refused.append(sym)
    assert refused == ["p1", "p1m1", "p2"]
    p = sp.build_frieze_patch("p11g", 10, "Mobius")
    assert (p["columns"], p["length_periods"]) == (21, 10.5)
    assert sp.build_frieze_patch("p2mm", 10, "Mobius")["columns"] == 20


def _decoration(sym):
    p = sp.build_frieze_patch(sym, 24, "OpenStrip")
    at = lambda v: (v // 3, v % 3)  # noqa: E731
    return {tuple(sorted((at(u), at(v)))) for u, v, k in p["edges"] if k == 2}, p["columns"]


def _is_sym(dec, cols, s, t, flip):
    m = lambda q: (s * q[0] + t, 2 - q[1] if flip else q[1])  # noqa: E731
    checked = 0
    for a, b in dec:
        ma, mb = m(a), m(b)
        if 0 <= ma[0] < cols and 0 <= mb[0] < cols:
            checked += 1
            if tuple(sorted((ma, mb))) not in dec:
                return False
    assert checked > 10
    return True


def test_decorations_realise_exactly_their_group():
    # Standard frieze table (hand): (horizontal mirror, glide, vertical mirror, half-turn).
    table = {
        "p1": (0, 0, 0, 0), "p11m": (1, 0, 0, 0), "p1m1": (0, 0, 1, 0), "p11g": (0, 1, 0, 0),
        "p2": (0, 0, 0, 1), "p2mm": (1, 0, 1, 1), "p2mg": (0, 1, 1, 1),
    }
    for sym, want in table.items():
        dec, cols = _decoration(sym)
        assert not _is_sym(dec, cols, 1, 1, False) and _is_sym(dec, cols, 1, 2, False)
        got = (
            any(_is_sym(dec, cols, 1, t, True) for t in (0, 2)),
            any(_is_sym(dec, cols, 1, t, True) for t in (1, 3)),
            any(_is_sym(dec, cols, -1, t, False) for t in range(40, 44)),
            any(_is_sym(dec, cols, -1, t, True) for t in range(40, 44)),
        )
        assert tuple(int(g) for g in got) == want, sym
        for cells in range(4, 11):
            for topo in ("Cylinder", "Mobius"):
                try:
                    p = sp.build_frieze_patch(sym, cells, topo)
                except ValueError:
                    continue
                assert _is_sym(dec, cols, 1, p["columns"], topo == "Mobius"), (sym, topo, cells)


def test_no_coincident_vertices_and_labels():
    for sym in sp.FRIEZE_SYMBOLS:
        for topo in ("OpenStrip", "Cylinder", "Mobius"):
            for defect in ("None", "HalfTurn180", "GlideDispiration"):
                for cells in (4, 13):
                    try:
                        p = sp.build_frieze_patch(sym, cells, topo, defect)
                    except ValueError:
                        continue
                    pts = p["pos3d"]
                    dmin = min(math.dist(pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts)))
                    assert dmin > 0.1, (sym, topo, defect, cells, dmin)
                    assert p["picture_note"] == ("topological picture, not isometric" if topo == "Mobius" else None)


def test_rust_matches_python(monkeypatch):
    monkeypatch.delenv("PAPERTILLING_NO_RUST", raising=False)
    rs = _rust.get()
    if rs is None or "wrap_edges" not in json.loads(rs.subperiodic_frieze_json("p11m", 6, "Cylinder", "None")):
        pytest.skip("papertilling_rs extension with the C1 frieze patch not built")
    for sym in sp.FRIEZE_SYMBOLS:
        for topo in ("OpenStrip", "Cylinder", "Mobius"):
            for defect in ("None", "GlideDispiration"):
                try:
                    r = json.loads(rs.subperiodic_frieze_json(sym, 9, topo, defect))
                except ValueError as e:
                    with pytest.raises(ValueError, match=r"^\[PROVED\]") as py_e:
                        monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
                        sp.build_frieze_patch(sym, 9, topo, defect)
                    monkeypatch.delenv("PAPERTILLING_NO_RUST")
                    assert str(py_e.value) == str(e)
                    continue
                monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
                p = sp.build_frieze_patch(sym, 9, topo, defect)
                monkeypatch.delenv("PAPERTILLING_NO_RUST")
                for key in ("columns", "length_periods", "quads", "wrap_edges", "wrap_quads", "picture_note", "glue_note"):
                    assert r[key] == p[key], (sym, topo, key)
                assert [list(e) for e in r["edges"]] == [list(e) for e in p["edges"]]
                for a, b in zip(r["pos3d"] + r["pos2d"], p["pos3d"] + p["pos2d"]):
                    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-12
