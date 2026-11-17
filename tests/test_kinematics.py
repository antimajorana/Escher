"""Tests for particle-conserving defect kinematics (1D glide and disclination-dislocation reactions)."""
import collections

import numpy as np
import pytest

from papertilling.defects import (
    absorb_dislocation,
    glide_dislocation,
    glide_line_spacing,
    project_glide,
    step_disclination_emit,
)
from papertilling.defects.build import build, lattice_data, same_charge


def _delta_n_square(conf, half_box: int = 5) -> int:
    """Exact site surplus/deficit inside a fixed material boundary [-L, L]^2 on a vacuum configuration."""
    d = conf.defect
    pos = d.positions
    adj = [[] for _ in range(len(pos))]
    for (u, v), ed in d.edge.items():
        adj[u].append((v, ed))
    s0 = min(range(len(pos)), key=lambda i: (pos[i, 0] + half_box) ** 2 + (pos[i, 1] + half_box) ** 2)
    frames, coords = {s0: np.eye(2)}, {s0: (0, 0)}
    dq = collections.deque([s0])
    while dq:
        u = dq.popleft()
        ux, uy = coords[u]
        fu = frames[u]
        for v, ed in adj[u]:
            if max(abs(pos[v, 0]), abs(pos[v, 1])) < 3.4:
                continue
            step = fu @ ed["vec"]
            nx_, ny_ = ux + int(round(step[0])), uy + int(round(step[1]))
            if abs(nx_ - half_box) <= half_box + 2 and abs(ny_ - half_box) <= half_box + 2:
                g = ed.get("g")
                fv = fu @ g[0].T if g is not None else fu
                if v not in coords:
                    coords[v] = (nx_, ny_)
                    frames[v] = fv
                    dq.append(v)
                else:
                    assert coords[v] == (nx_, ny_)
    bndry = {
        u
        for u, (ix, iy) in coords.items()
        if (0 <= ix <= 2 * half_box and iy in (0, 2 * half_box))
        or (0 <= iy <= 2 * half_box and ix in (0, 2 * half_box))
    }
    assert len(bndry) == 8 * half_box
    c_start = min(range(len(pos)), key=lambda i: pos[i, 0] ** 2 + pos[i, 1] ** 2)
    inside = set(bndry) | {c_start}
    qi = collections.deque([c_start])
    while qi:
        u = qi.popleft()
        for v, _ in adj[u]:
            if v not in inside:
                inside.add(v)
                qi.append(v)
    return len(inside) - (2 * half_box + 1) ** 2


def test_dislocation_glide_projection_forbids_climb_and_conserves_site_count():
    with pytest.raises(ValueError, match="immobile"):
        project_glide(dict(kind="disclination", angle=90, at=(0.0, 0.0)), (1.0, 0.0))

    d0 = dict(kind="dislocation", b=(1.0, 0.0), at=(-1.63, 0.41))
    glided = project_glide(d0, (1.42, 3.85))
    assert abs(glided["at"][1] - 0.41) < 1e-12
    assert abs(glided["at"][0] - 1.42) < 1e-12

    sq = lattice_data("square")
    dp = glide_line_spacing(sq, (1.0, 0.0))
    assert abs(dp - 1.0) < 1e-12
    partner = dict(kind="dislocation", b=(-1.0, 0.0), at=(1.37, 0.41))
    d_glide = project_glide(d0, (-2.63, 2.75))
    c_glide = build(sq, d_glide, partner, R=11.0, centre=(0.0, 0.0), picture="raw")
    assert _delta_n_square(c_glide) == 0

    d_climb_up = dict(kind="dislocation", b=(1.0, 0.0), at=(-1.63, 0.41 + dp))
    c_climb_up = build(sq, d_climb_up, partner, R=11.0, centre=(0.0, 0.0), picture="raw")
    d_climb_dn = dict(kind="dislocation", b=(1.0, 0.0), at=(-1.63, 0.41 - dp))
    c_climb_dn = build(sq, d_climb_dn, partner, R=11.0, centre=(0.0, 0.0), picture="raw")
    assert {_delta_n_square(c_climb_up), _delta_n_square(c_climb_dn)} == {-1, 1}

    glided_specs, rotated = glide_dislocation(sq, [d0, partner], 0, (-2.63, 2.75))
    assert not rotated and abs(glided_specs[0]["at"][1] - 0.41) < 1e-12


@pytest.mark.parametrize(
    ("name", "ang"),
    [("square", 90.0), ("honeycomb", 60.0), ("shastry_sutherland", 90.0), ("chiral_triangular", 120.0)],
)
def test_disclination_step_emit_and_annihilation_across_lattices(name, ang):
    lat = lattice_data(name)
    p0 = next(r["point"] for r in lat.centres if ang in r["angles"])
    p1 = p0 + lat.cell.a1
    p2 = p0 + 2.0 * lat.cell.a1
    specs0 = [dict(kind="disclination", angle=ang, at=tuple(p0)), dict(kind="disclination", angle=-ang, at=tuple(p2))]
    q0 = build(lat, *specs0, picture="raw").total_charge()

    specs1, em1 = step_disclination_emit(lat, specs0, 1, p1)
    assert em1 is not None
    q1 = build(lat, *specs1, picture="raw").total_charge()
    assert same_charge(lat, q0, q1)

    specs2, em2 = step_disclination_emit(lat, specs1, 1, p0)
    assert em2 is not None
    c2 = build(lat, *specs2, picture="raw")
    assert same_charge(lat, q0, c2.total_charge())
    assert all(c["kind"] == "dislocation" for c in c2.cores())


def test_180_disclination_wyckoff_step_emits_unit_dislocation():
    sq = lattice_data("square")
    specs0 = [dict(kind="disclination", angle=180.0, at=(0.0, 0.0))]
    q0 = build(sq, *specs0, picture="raw").total_charge()
    specs1, emitted = step_disclination_emit(sq, specs0, 0, (0.5, 0.0))
    assert emitted is not None and abs(np.linalg.norm(emitted["b"]) - 1.0) < 1e-6
    q1 = build(sq, *specs1, picture="raw").total_charge()
    assert same_charge(sq, q0, q1)


@pytest.mark.parametrize(
    ("name", "ang"),
    [("square", 90.0), ("honeycomb", 60.0), ("shastry_sutherland", 90.0), ("chiral_triangular", 120.0)],
)
def test_absorb_dislocation_exact_roundtrip_and_climb_refusal(name, ang):
    lat = lattice_data(name)
    p0 = next(r["point"] for r in lat.centres if ang in r["angles"])
    p1 = p0 + lat.cell.a1
    p2 = p0 + 2.0 * lat.cell.a1
    specs0 = [dict(kind="disclination", angle=ang, at=tuple(p0)), dict(kind="disclination", angle=-ang, at=tuple(p2))]
    q0 = build(lat, *specs0, picture="raw").total_charge()
    specs_em, _ = step_disclination_emit(lat, specs0, 1, p1)
    b_em = np.asarray(specs_em[2]["b"], float)
    ub = b_em / np.linalg.norm(b_em)
    specs_glided = None
    for sgn in (-1.0, 1.0):
        cand = [
            specs_em[0],
            specs_em[1],
            project_glide(specs_em[2], np.asarray(specs_em[2]["at"]) + sgn * 1.8 * lat.lmax * ub),
        ]
        try:
            if same_charge(lat, q0, build(lat, *cand, picture="raw").total_charge()):
                specs_glided = cand
                break
        except ValueError:
            continue
    assert specs_glided is not None

    rec, c_rec = absorb_dislocation(lat, specs_glided, 2, 1)
    assert len(rec) == 2 and np.allclose(c_rec, p2, atol=1e-5)

    dp = glide_line_spacing(lat, b_em)
    nb = np.array([-ub[1], ub[0]], float)
    climbed = [
        specs_em[0],
        specs_em[1],
        dict(specs_em[2], at=tuple((np.asarray(specs_em[2]["at"]) + dp * nb).tolist())),
    ]
    with pytest.raises(ValueError, match="climb"):
        absorb_dislocation(lat, climbed, 2, 1)
