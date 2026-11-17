"""Tests for Stage 3.1 (two-tier core equivalence) and Stage 3.2 (Burgers-circuit engine)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from papertilling.defects import DefectCatalogue, auto_burgers_circuit, burgers_circuit, shift_circuit_basepoint
from papertilling.defects import fast as _fast
from papertilling.defects.build import build, lattice_data
from papertilling.defects.config import config_to_dict, equivalent_single


def test_two_tier_equivalence_volterra_completion_exact_none():
    """Verify match_tier, volterra_matches, completion_matches across all four tiers."""
    sq = lattice_data("square")
    pair = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)), picture="raw")
    res_v = equivalent_single(pair, grid=1)
    assert res_v["equivalent"] is True
    assert res_v["match_tier"] == "volterra"
    assert res_v["volterra_matches"] and not res_v["completion_matches"]
    assert res_v["volterra_matches"] == res_v["matches"]

    # Point-defect JSON summary includes tier fields while keeping top-level format v1
    d_json = config_to_dict(pair, grid=1)
    assert d_json["format"] == "papertilling.config.v1"
    assert d_json["summary"]["match_tier"] == "volterra"
    assert d_json["summary"]["volterra_matches"] == res_v["volterra_matches"]
    assert d_json["summary"]["completion_matches"] == []

    # Honeycomb 5-7 core matches completion only (C04)
    hc = lattice_data("honeycomb")
    c6 = next(r["point"] for r in hc.centres if r["order"] == 6)
    hc_57 = build(hc, dict(kind="disclination", angle=60, at=c6), dict(kind="disclination", angle=-60, at=c6 + hc.cell.a1), picture="raw")
    res_c = equivalent_single(hc_57, grid=2)
    assert res_c["equivalent"] is True
    assert res_c["match_tier"] == "completion"
    assert res_c["volterra_matches"] == [] and res_c["completion_matches"] == res_c["matches"]

    # Disclination & vacuum -> exact tier
    one = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), picture="raw")
    res_e = equivalent_single(one)
    assert res_e["match_tier"] == "exact" and res_e["volterra_matches"] == [] and res_e["completion_matches"] == []

    vac = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0, 0)), picture="raw")
    res_vac = equivalent_single(vac)
    assert res_vac["match_tier"] == "exact" and res_vac["equivalent"] is True

    # Far dislocation dipole -> total charge vacuum, non-empty core -> none
    dip = build(sq, dict(kind="dislocation", b=(1, 0), at=(-3.63, 0.41), cut=(0, 1)),
                dict(kind="dislocation", b=(-1, 0), at=(3.37, 0.41), cut=(0, -1)), picture="raw")
    res_n = equivalent_single(dip)
    assert res_n["equivalent"] is False and res_n["match_tier"] == "none"


def test_square_pair_circuits_and_exact_loop_charge_agreement():
    """Square +90 at (0,0) + -90 at (0.5, 0.5): total circuit has frank=0, |b|=1;
    single-pin circuits recover +90 and -90; matches _fast.loop_charge_fast."""
    sq = lattice_data("square")
    pair = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)), picture="raw")

    c_all = auto_burgers_circuit(pair)
    assert abs(c_all["frank_angle_deg"]) < 1e-6
    assert abs(c_all["burgers_norm"] - 1.0) < 1e-6
    assert c_all["orientation_preserving"] is True
    assert c_all["charge"]["kind"] == "dislocation"
    H_ref, S_ref = _fast.loop_charge_fast(pair.defect, c_all["path"][:-1])
    assert np.allclose(c_all["H"][0], H_ref[0], atol=1e-9) and np.allclose(c_all["H"][1], H_ref[1], atol=1e-9)
    assert np.allclose(c_all["closure_vector"], S_ref, atol=1e-9)

    c_plus = auto_burgers_circuit(pair, target=(0, 0), exclude=(0.5, 0.5))
    c_minus = auto_burgers_circuit(pair, target=(0.5, 0.5), exclude=(0, 0))
    assert abs(c_plus["frank_angle_deg"] - 90.0) < 1e-6
    assert abs(c_minus["frank_angle_deg"] + 90.0) < 1e-6

    # Separated dipole allows radius_clearance=2 circuits strictly avoiding all irregular sites
    sep = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(5.5, 5.5)), picture="raw")
    irreg = sep.core_sites()
    cs_all = auto_burgers_circuit(sep, radius_clearance=2)
    cs_p = auto_burgers_circuit(sep, target=(0, 0), exclude=(5.5, 5.5), radius_clearance=2)
    cs_m = auto_burgers_circuit(sep, target=(5.5, 5.5), exclude=(0, 0), radius_clearance=2)
    assert abs(cs_all["frank_angle_deg"]) < 1e-6 and abs(cs_p["frank_angle_deg"] - 90.0) < 1e-6 and abs(cs_m["frank_angle_deg"] + 90.0) < 1e-6
    assert not (set(cs_all["path"]) & irreg)
    assert not (set(cs_p["path"]) & irreg) and not (set(cs_m["path"]) & irreg)


def test_unreduced_180_and_braiding_basepoint_rotation():
    """+180 != -180 in circuit Frank angle; shifting basepoint across a +alpha seam rotates b by R(alpha)."""
    sq = lattice_data("square")
    p180 = build(sq, dict(kind="disclination", angle=180, at=(0, 0.5)), picture="raw")
    m180 = build(sq, dict(kind="disclination", angle=-180, at=(0, 0.5)), picture="raw")
    cp = auto_burgers_circuit(p180)
    cm = auto_burgers_circuit(m180)
    assert abs(cp["frank_angle_deg"] - 180.0) < 1e-6 and abs(cp["cum_angle_deg"][-1] - 180.0) < 1e-6
    assert abs(cm["frank_angle_deg"] + 180.0) < 1e-6 and abs(cm["cum_angle_deg"][-1] + 180.0) < 1e-6

    # Braiding / basepoint control on separated +90 / -90 dipole: crossing the +90 seam rotates b by R(+/-90)
    sep = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(3.5, 3.5)), picture="raw")
    c0 = auto_burgers_circuit(sep)
    b0 = c0["closure_vector"]
    shifted_idx = next(i for i, F in enumerate(c0["frames"]) if not np.allclose(F, np.eye(2), atol=1e-5))
    F_cross = c0["frames"][shifted_idx]
    c1 = burgers_circuit(sep, shift_circuit_basepoint(c0["path"], c0["path"][shifted_idx]))
    b1 = c1["closure_vector"]
    assert abs(np.linalg.norm(b1) - np.linalg.norm(b0)) < 1e-6
    assert np.allclose(b1, F_cross.T @ b0, atol=1e-6)  # b0 = F_cross @ b1 => b1 = F_cross^T @ b0
    assert abs(float(b0 @ b1)) < 1e-6  # perpendicular: R(+-90) rotation

    # Chiral triangular +120 / -120 dipole: basepoint shift rotates Burgers vector by 120 deg
    tri = lattice_data("chiral_triangular")
    c3 = next(r["point"] for r in tri.centres if r["order"] == 3)
    tri_pair = build(tri, dict(kind="disclination", angle=120, at=c3),
                     dict(kind="disclination", angle=-120, at=c3 + 2.0 * tri.cell.a1), picture="raw")
    ct0 = auto_burgers_circuit(tri_pair)
    idx3 = next(i for i, F in enumerate(ct0["frames"]) if not np.allclose(F, np.eye(2), atol=1e-5))
    ct1 = burgers_circuit(tri_pair, shift_circuit_basepoint(ct0["path"], ct0["path"][idx3]))
    F120 = ct0["frames"][idx3]
    assert np.allclose(ct1["closure_vector"], F120.T @ ct0["closure_vector"], atol=1e-6)
    cos120 = float(ct0["closure_vector"] @ ct1["closure_vector"]) / (ct0["burgers_norm"] ** 2)
    assert abs(cos120 - (-0.5)) < 1e-6


def test_mobius_seam_circuit_and_validation_errors():
    """Wc1 Mobius seam circuit crosses orientation-reversing seam and sets orientation_preserving=False."""
    cat = DefectCatalogue("square")
    wc1 = cat.mobius_seams[0].build()
    res = auto_burgers_circuit(wc1, lattice=lattice_data("square"))
    assert res["orientation_preserving"] is False
    assert res["det_signs"][0] == 1 and res["det_signs"][-1] == -1
    assert res["frank_angle_deg"] is None
    assert res["charge"]["kind"] == "non_orientable"

    with pytest.raises(ValueError):
        burgers_circuit(wc1, [0, 1, 2])
    with pytest.raises(ValueError):
        shift_circuit_basepoint([0, 1, 2, 0], 999999)
