"""Independent mathematical known-answer tests for fusion review follow-up fixes (pure Python).

Every expected value is derived analytically in the docstring (holonomy algebra, Euler balance,
Author decision 2026-11-22 on |Frank| <= 180 deg), never copied from the code under test.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from papertilling.defects._surgery import _Builder, candidate_cuts_360, run_eff_with_fallback
from papertilling.defects.build import build, classify_charge, lattice_data
from papertilling.defects.config import equivalent_single
from papertilling.lattice import load_catalogue
from papertilling.legacy.lab_defects import frame_faces, iso_compose, rotation_about


def _disc(angle: float, at: tuple[float, float]) -> dict:
    return dict(kind="disclination", angle=float(angle), at=np.asarray(at, float))


def _disl(b: tuple[float, float], at: tuple[float, float]) -> dict:
    return dict(kind="dislocation", b=np.asarray(b, float), at=np.asarray(at, float))


def _centre(lat, label: str) -> np.ndarray:
    return np.asarray(next(r["point"] for r in lat.centres if r["label"] == label), float)


def _interior_faces_and_degrees(defect) -> tuple[dict[int, int], dict[int, int]]:
    faces = frame_faces(defect)
    k_out = max(range(len(faces)), key=lambda i: len(faces[i]))
    outer_verts = set(faces[k_out])
    face_hist = Counter(len(f) for i, f in enumerate(faces) if i != k_out)
    deg = defect.degrees()
    deg_hist = Counter(int(deg[v]) for v in range(len(defect.sites)) if v not in outer_verts)
    return dict(face_hist), dict(deg_hist)


def test_270_disclination_and_whole_turn_verdicts():
    """Author decision (2026-11-22): keep single = None for 180° < |angle| < 360°, matching docs/RUST_PLAN.md §9.

    Hand derivation: three +90 deg site disclinations on the square lattice carry unreduced Frank
    angle +270 deg (> 180 deg), so no single elementary disclination carries this charge.
    Two -180 deg disclinations carry -360 deg (whole_turn), also with no single defect.
    """
    sq = lattice_data("square")

    # Direct classify_charge check for +270 deg about a site (0, 0): R_270 = R_-90.
    h270 = rotation_about(np.array([0.0, 0.0]), math.radians(270.0))
    q_direct = classify_charge(sq, 270.0, h270)
    assert q_direct["kind"] == "disclination"
    assert abs(q_direct["angle"] - 270.0) < 1e-6
    assert q_direct["single"] is None

    # Built square +90 x3 configuration.
    triple = build(sq, _disc(90, (0, 0)), _disc(90, (3, 0)), _disc(90, (6, 0)), picture="raw")
    q = triple.total_charge()
    assert q["kind"] == "disclination"
    assert abs(q["angle"] - 270.0) < 1e-6
    assert q["single"] is None
    res = equivalent_single(triple, radius=2, grid=1)
    assert not res["equivalent"]
    assert res["verdict"] == "no single defect carries this charge (|Frank| > 180 deg)"
    assert triple.core_graph(2).number_of_nodes() > 0

    # Whole-turn (-360 deg) charge also reports honest no-single-defect verdict with non-empty core.
    wt = build(sq, _disc(-180, (0, 0.5)), _disc(-180, (4, 0.5)), picture="raw")
    q_wt = wt.total_charge()
    assert q_wt["kind"] == "whole_turn"
    assert q_wt["single"] is None
    res_wt = equivalent_single(wt, radius=2, grid=1)
    assert not res_wt["equivalent"]
    assert res_wt["verdict"] == "no single defect carries this charge"
    assert wt.core_graph(2).number_of_nodes() > 0


@pytest.mark.parametrize(
    ("group", "label_a", "label_b"),
    [("p3", "1b", "1c"), ("p31m", "2b(i)", "2b(ii)")],
)
def test_hex_split_dipole_burgers_length_algebraic(group: str, label_a: str, label_b: str):
    """For a ±theta disclination dipole at centres c1, c2 with d = c2 - c1:
      H = (R_theta, (I - R_theta) c1)(R_-theta, (I - R_-theta) c2) = (I, (R_theta - I) d),
    so |b| = |(I - R_theta) d| = 2 sin(theta / 2) * |d|.
    On unit hexagonal p3 / p31m lattices (|a1| = 1), the two non-origin 3-fold Wyckoff centres sit
    at distance |d| = 1 / sqrt(3), giving |b| = 2 sin(60 deg) * |d| = sqrt(3) / sqrt(3) = 1.0.
    """
    lat = lattice_data(load_catalogue(group=group, rule="points").lattices[0])
    c1 = _centre(lat, label_a)
    c2 = _centre(lat, label_b)
    d_len = float(np.linalg.norm(c2 - c1))
    b_expected = 2.0 * math.sin(math.radians(60.0)) * d_len
    assert abs(b_expected - 1.0) < 1e-6, f"{group}: expected 2*sin(60)*|d| = 1.0, got {b_expected}"

    conf = build(lat, _disc(120, tuple(c1)), _disc(-120, tuple(c2)), picture="raw")
    q = conf.total_charge()
    assert q["kind"] == "dislocation"
    assert q["lattice_vector"] is True
    assert abs(q["length"] - b_expected) < 1e-6
    assert abs(float(np.linalg.norm(q["b"])) - 1.0) < 1e-6


def test_run_eff_with_fallback_orders_dfs_and_refusal():
    """Verify run_eff_with_fallback on order fallback, DFS backtracking, and strict overlap refusal.

    Hand holonomy for opposite-corner triple [-90@(1,1), +90@(0.5,0.5), -90@(0,0)]:
      H = H(-90, (1,1)) H(+90, (0.5,0.5)) H(-90, (0,0)) has linear part R_-90 and fixed point
      c = (0.5, 0.5) (Wyckoff class 1b), so total charge is D(1b, -90 deg).
    """
    sq = lattice_data("square")

    # 1. [F_+90, V_-90]: fails in caller order, succeeds via fallback order [1, 0], |b| = 1.0.
    eff_fv = [_disc(90, (0.5, 0.5)), _disc(-90, (0.0, 0.0))]
    B_fv = run_eff_with_fallback(lambda: _Builder(sq.cell, sq.L, np.array([0.25, 0.25]), 14.0), eff_fv, sq)
    assert len(B_fv.finish().sites) > 0
    conf_fv = build(sq, _disc(90, (0.5, 0.5)), _disc(-90, (0.0, 0.0)), picture="raw")
    q_fv = conf_fv.total_charge()
    assert q_fv["kind"] == "dislocation"
    assert abs(q_fv["length"] - 2.0 * math.sin(math.radians(45.0)) * math.hypot(0.5, 0.5)) < 1e-6

    # 2. Opposite-corner triple [-90@(1,1), +90@(0.5,0.5), -90@(0,0)]: succeeds via DFS -> D(1b, -90 deg).
    h_hand = iso_compose(
        rotation_about(np.array([1.0, 1.0]), math.radians(-90.0)),
        iso_compose(
            rotation_about(np.array([0.5, 0.5]), math.radians(90.0)),
            rotation_about(np.array([0.0, 0.0]), math.radians(-90.0)),
        ),
    )
    c_hand = np.linalg.solve(np.eye(2) - h_hand[0], h_hand[1])
    assert sq.centre_class(c_hand)["label"] == "1b"

    eff_opp = [_disc(-90, (1.0, 1.0)), _disc(90, (0.5, 0.5)), _disc(-90, (0.0, 0.0))]
    B_opp = run_eff_with_fallback(lambda: _Builder(sq.cell, sq.L, np.array([0.5, 0.5]), 14.5), eff_opp, sq)
    assert len(B_opp.finish().sites) > 0
    conf_opp = build(sq, *eff_opp, picture="raw")
    q_opp = conf_opp.total_charge()
    assert q_opp["kind"] == "disclination"
    assert abs(q_opp["angle"] + 90.0) < 1e-6
    assert q_opp["centre_class"] == "1b"

    # 3. Adjacent-corner triple [-90@(0,0), -90@(1,0), +90@(0.5,0.5)]: strict overlap refusal.
    eff_adj = [_disc(-90, (0.0, 0.0)), _disc(-90, (1.0, 0.0)), _disc(90, (0.5, 0.5))]
    with pytest.raises(
        ValueError,
        match="a later cut would remove an existing site-centred disclination core; try another cut direction",
    ):
        run_eff_with_fallback(lambda: _Builder(sq.cell, sq.L, np.array([0.5, 0.1667]), 14.5), eff_adj, sq)


def test_wang_four_inner_dislocations_euler_balance():
    """Wang's 4 inner dislocations at r = 3 on the square lattice (b = ±x, ±y at (±3, 0), (0, ±3)).

    With candidate_cuts_360(..., lmax=sq.lmax) capping clearance at 2.2 * lmax, perpendicular cuts
    win once clear of other cores. Each unit dislocation core is a clean 5-gon with one 3-valent
    site, satisfying Euler balance: 4*(4 - 3) + 4*(4 - 5) = 0 (zero net Frank angle).
    """
    sq = lattice_data("square")
    specs = [
        _disl((1, 0), (-3.5, 0.0)),
        _disl((-1, 0), (3.5, 0.0)),
        _disl((0, 1), (0.0, 2.5)),
        _disl((0, -1), (0.0, -2.5)),
    ]
    # Check candidate_cuts_360 picks a cut perpendicular to b (dot(u, b) == 0) first for every defect.
    for i, d in enumerate(specs):
        others = [o for j, o in enumerate(specs) if j != i]
        cands = candidate_cuts_360(d, others, True, lmax=sq.lmax)
        assert abs(float(cands[0] @ d["b"])) < 1e-9, f"defect {i}: first cut {cands[0]} not perpendicular to {d['b']}"

    conf = build(sq, *specs, picture="raw")
    face_hist, deg_hist = _interior_faces_and_degrees(conf.defect)
    odd_faces = {k: v for k, v in face_hist.items() if k != 4}
    odd_degs = {k: v for k, v in deg_hist.items() if k != 4}
    assert odd_faces == {5: 4}, f"expected four 5-gons, got {odd_faces}"
    assert odd_degs == {3: 4}, f"expected four 3-valent interior vertices, got {odd_degs}"
