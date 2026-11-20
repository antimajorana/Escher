"""Boundary conditions of a 2D lattice (C1 WP-B1, plan §5 T3), pure Python (papertilling.surfaces._glides, _flat_bc).

Where the numbers come from: hand counts in the comments (the L x L torus of {4,4}; the reflections of p4m and p6m; a
Klein bottle is half of the torus R^2 / (along x tau, across x T)), and topological invariants (chi, boundary circles,
orientability) computed here from the faces, independently of the builders. The Rust tests read the same cases from
rust/papertilling-core/tests/fixtures/surface_bc_py.json; test_fixture_is_current checks that file against this package.
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pytest

import papertilling as pt
from papertilling.surfaces import glides, surface_table
from papertilling.surfaces._flat_bc import TOO_SMALL, boundary_surface, wrap_complement
from papertilling.surfaces._glides import glide_search
from papertilling.surfaces.refusal import SurfaceImpossible, SurfaceNotFound

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "rust", "papertilling-core", "tests", "fixtures", "surface_bc_py.json")


def topology(S):
    """(chi, boundary circles, orientable) from the faces only."""
    uses = {}
    for fi, f in enumerate(S.faces):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            uses.setdefault((min(a, b), max(a, b)), []).append((fi, a < b))
    chi = len(S.sites) - len(S.bonds) + len(S.faces)
    parent = list(range(len(S.sites)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    rim = set()
    for (a, b), u in uses.items():
        if len(u) == 1:
            rim |= {a, b}
            parent[find(a)] = find(b)
    circles = len({find(v) for v in rim})
    sign, orientable = [None] * len(S.faces), True
    for start in range(len(S.faces)):
        if sign[start] is not None:
            continue
        sign[start], stack = True, [start]
        while stack:
            f = stack.pop()
            for k in range(len(S.faces[f])):
                a, b = S.faces[f][k], S.faces[f][(k + 1) % len(S.faces[f])]
                for g, d in uses[(min(a, b), max(a, b))]:
                    if g == f:
                        continue
                    want = (a < b) ^ (not sign[f]) ^ d       # g runs the shared edge the other way
                    if sign[g] is None:
                        sign[g] = want
                        stack.append(g)
                    elif sign[g] != want:
                        orientable = False
    return chi, circles, orientable


def _sq(k1, k2, diag=None):
    bonds = [(0, 0, 1, 0, k1), (0, 0, 0, 1, k2)] + ([(0, 0, 1, 1, diag)] if diag is not None else [])
    return np.eye(2), [(0.0, 0.0)], bonds


def test_glide_search_is_kind_aware_and_basis_independent():
    # with kinds 0/1 on a1/a2 and kind 2 on a1 + a2, every reflection either swaps kinds 0 and 1 (the diagonals) or maps
    # a1 + a2 to a non-bond (the axes): none; all kinds equal: the 2 diagonals. Axes only: 2 with kinds, 4 without.
    assert glide_search(*_sq(0, 1, 2)) == []
    assert len(glide_search(*_sq(0, 0, 0))) == 2
    assert len(glide_search(*_sq(0, 1))) == 2 and len(glide_search(*_sq(0, 0))) == 4
    # p4m: 4 reflection linear parts, in the skewed basis a2' = 5 a1 + a2 too; p6m: 6
    A = np.array([[1.0, 5.0], [0.0, 1.0]])
    assert len(glide_search(A, [(0.0, 0.0)], [(0, 0, 1, 0, 0), (0, 0, -5, 1, 0)])) == 4
    assert len(glides(pt.named("triangular"))) == 6


def test_wrap_complement_is_a_basis_partner():
    A = pt.named("triangular").cell().A
    for c in [(1, 0), (6, 0), (3, 1), (4, -7), (0, 5), (-2, 9)]:
        t = wrap_complement(A, c)
        g = np.gcd(*c)
        assert (c[0] // g) * t[1] - (c[1] // g) * t[0] == 1, (c, t)
    with pytest.raises(ValueError):
        wrap_complement(A, (0, 0))


def test_square_torus_counts_and_too_small():
    # {4,4} on R^2 / (L Z)^2: V = L^2, E = 2 L^2, F = L^2; L = 1 gives a loop, L = 2 a double bond
    sq = pt.named("square")
    for n in (3, 4, 5):
        S = boundary_surface(sq, "torus", t1=(n, 0), t2=(0, n))
        assert (len(S.sites), len(S.bonds), len(S.faces)) == (n * n, 2 * n * n, n * n)
        assert topology(S) == (0, 0, True) and S.picture_note == "topological picture, not isometric" and S.exact_view
    for n in (1, 2):
        with pytest.raises(SurfaceNotFound, match=r"too small"):
            boundary_surface(sq, "torus", t1=(n, 0), t2=(0, n))
    assert TOO_SMALL == "NOT FOUND (too small)"


@pytest.mark.parametrize("key,wrap,rows", [("square", (4, 0), 3), ("square", (3, 2), 13), ("honeycomb", (6, 0), 3),
                                           ("triangular", (4, 1), 12)])
def test_cylinder_any_wrap_is_an_annulus(key, wrap, rows):
    S = boundary_surface(pt.named(key), "cylinder", wrap=wrap, rows=rows)
    assert topology(S) == (0, 2, True) and S.picture_note is None


def test_mobius_klein_even_lengths_and_refusals():
    sq = pt.named("square")
    assert glides(sq)[0]["mirror"]
    for steps in (3, 4):
        M = boundary_surface(sq, "mobius", glide=0, steps=steps, width=2)
        assert topology(M) == (0, 1, False)
        K = boundary_surface(sq, "klein", glide=0, steps=steps, across=4)
        assert topology(K) == (0, 0, False)
    # steps = 4: g'^2 = 4 tau = 8 a1, across 4 a2: the torus R^2 / (8 a1, 4 a2) has 32 sites; the Klein bottle half
    K = boundary_surface(sq, "klein", glide=0, steps=4, across=4)
    assert (len(K.sites), len(K.bonds), len(K.faces)) == (16, 32, 16)
    pg = pt.named("brick_pg")
    assert glides(pg) and not any(g["mirror"] for g in glides(pg))
    with pytest.raises(ValueError, match="odd number of steps"):
        boundary_surface(pg, "mobius", glide=0, steps=2, width=2)
    with pytest.raises(SurfaceImpossible, match=r"\[PROVED\]") as e:
        boundary_surface(pt.named("chiral_square"), "klein", glide=0, steps=3, across=3)
    assert e.value.scope == "lattice"


def test_hex_domain_same_torus_on_wigner_seitz_cell():
    tri = pt.named("triangular")
    A = tri.cell().A
    for t1, t2 in [((3, 0), (0, 3)), ((3, 1), (-1, 2))]:
        T = boundary_surface(tri, "torus", t1=t1, t2=t2)
        H = boundary_surface(tri, "torus_hex", t1=t1, t2=t2)
        assert (len(H.sites), len(H.bonds), len(H.faces)) == (len(T.sites), len(T.bonds), len(T.faces))
        poly = np.array(H.domain["polygon"])
        area = 0.5 * abs(sum(poly[k, 0] * poly[k - 5, 1] - poly[k - 5, 0] * poly[k, 1] for k in range(6)))
        v1, v2 = A @ np.array(t1, float), A @ np.array(t2, float)
        assert len(poly) == 6 and abs(area - abs(v1[0] * v2[1] - v1[1] * v2[0])) < 1e-9
        near = [a * v1 + b * v2 for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0)]
        assert all(p @ p <= (p - t) @ (p - t) + 1e-9 for p in H.uv for t in near)
    H = boundary_surface(pt.named("square"), "torus_hex", t1=(3, 0), t2=(0, 3))
    assert len(H.domain["edges"]) == 4 and any("3 pairs reduce to 2" in r for r in H.readout)


def test_c1_oblique_tuned_to_square_and_fixture_is_current():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = pt.named("oblique")
        L = L.set(b=1.0, gamma=90.0) or L
    assert L.group == "p2" and glides(L)                         # the label lists no mirror; the cell has two
    with open(FIXTURE, encoding="utf-8") as f:
        cases = json.load(f)
    want = next(c for c in cases if c["label"] == "oblique_tuned")["statuses"]
    T = surface_table(L, show=False, only=["Moebius band", "Klein bottle", "projective plane"])
    got = {r["surface"]: r["status"] for r in T}
    assert got == {k: want[k] for k in got} and all(v == "OK" for v in got.values())
    for c in cases:
        if "counts" in c:
            label, key = c["label"].split(":")
            kw = dict(torus_hex=dict(kind="torus_hex", t1=(3, 0), t2=(0, 3)), cylinder_6_0=dict(kind="cylinder", wrap=(6, 0), rows=3),
                      cylinder_3_1=dict(kind="cylinder", wrap=(3, 1), rows=8),
                      mobius_even=dict(kind="mobius", glide=0, steps=4, width=2),
                      klein_even=dict(kind="klein", glide=0, steps=4, across=4))[label]
            S = boundary_surface(pt.named(key), **kw)
            assert [len(S.sites), len(S.bonds), len(S.faces)] == c["counts"], label


def test_chiral_groups_refuse_on_every_catalogue_lattice():
    # [PROVED] no reflection in the decorated lattice -> no Moebius band, no Klein bottle; a reflection -> a glide class
    t0 = time.perf_counter()
    cat = pt.load_catalogue()
    chiral = 0
    for g in cat:
        for L in cat[g].lattices:
            gl = glides(L)
            is_chiral = g in ("p1", "p2", "p3", "p4", "p6")
            assert (gl == []) == is_chiral, (g, len(gl))
            if is_chiral:
                chiral += 1
                with pytest.raises(SurfaceImpossible):
                    boundary_surface(L, "mobius", glide=0, steps=3, width=2)
    print(f"# time {time.perf_counter() - t0:.2f}s")
    assert chiral == 35
