"""Tests for Stage 2.1 (algebraic wall endings) and Stage 2.2 (surgical half-wall builder)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from papertilling.defects import DefectCatalogue, auto_burgers_circuit, build_half_wall
from papertilling.defects.build import build, lattice_data
from papertilling.defects.config import config_to_dict, equivalent_single, load_config
from papertilling.defects import fast as _fast


def test_brick_pg_wall_endings():
    """brick_pg (pg): Wd1/Wd2 end as fractional partial dislocations (|b| = 0.5 * period),
    Wc1/Wc2 cannot end orientably, and Wa4..Wa7 end as 180-deg partial disclinations."""
    cat = DefectCatalogue("brick_pg")
    per = cat.mirrors[0]["period"]
    assert abs(per - 4.0) < 1e-9

    for wd in cat.glide_breaking:
        e = wd["ending"]
        assert e["can_end_orientable"] is True
        assert e["ending_kind"] == "partial_dislocation"
        assert e["is_full_lattice_burgers"] is False
        assert abs(e["burgers_norm"] - 0.5 * per) < 1e-6

    for wc in cat.mobius_seams:
        e = wc["ending"]
        assert e["can_end_orientable"] is False
        assert e["ending_kind"] == "non_orientable_twist"
        assert e["is_full_lattice_burgers"] is False
        assert e["burgers"] is None and e["frank_angle_deg"] is None

    wa4 = next(w for w in cat.domain_walls if w["id"] == "Wa4")
    assert wa4["ending"]["can_end_orientable"] is True
    assert wa4["ending"]["ending_kind"] == "partial_disclination"
    assert abs(wa4["ending"]["frank_angle_deg"] - 180.0) < 1e-6


def test_chiral_groups_wall_endings():
    """p3 (chiral_triangular) and p4 (chiral_square): mirror domain/twin walls are obstructed;
    proper rotation normaliser walls end as partial disclinations (+60 preferred over -60 on ties)."""
    tri = DefectCatalogue("chiral_triangular")
    assert tri.group == "p3" and tri.chiral
    proper_rot_wa = [w for w in tri.domain_walls if np.linalg.det(w["sigma"][0]) > 0 and w["type"] == "rotation"]
    mirror_wa = [w for w in tri.domain_walls if np.linalg.det(w["sigma"][0]) < 0]
    assert proper_rot_wa and mirror_wa
    for w in proper_rot_wa:
        e = w["ending"]
        assert e["can_end_orientable"] is True
        assert e["ending_kind"] == "partial_disclination"
        assert abs(e["frank_angle_deg"] - 60.0) < 1e-6
    for w in mirror_wa:
        assert w["ending"]["can_end_orientable"] is False
        assert w["ending"]["ending_kind"] == "chirality_obstruction"

    csq = DefectCatalogue("chiral_square")
    assert csq.group == "p4" and csq.chiral
    for w in csq.domain_walls:
        if np.linalg.det(w["sigma"][0]) < 0:
            assert w["ending"]["can_end_orientable"] is False
            assert w["ending"]["ending_kind"] == "chirality_obstruction"
        else:
            assert w["ending"]["can_end_orientable"] is True
            assert w["ending"]["ending_kind"] == "partial_dislocation"
    for w in csq.twin_walls[:4]:
        assert w["ending"]["can_end_orientable"] is False
        assert w["ending"]["ending_kind"] == "chirality_obstruction"


def test_square_twin_wall_ending_angle_and_trivial_mirrors():
    """Square (p4m): invisible mirror twin walls have full-lattice b=0; (2,1) twin walls have
    partial_disclination with alpha = 2 * angle(wall, nearest mirror) = -36.87 deg."""
    sq = DefectCatalogue("square", wall_bmax=2.5)
    for w in sq.twin_walls:
        e = w["ending"]
        assert e["can_end_orientable"] is True
        if w["sigma_in_G"]:
            assert e["ending_kind"] == "partial_dislocation"
            assert e["is_full_lattice_burgers"] is True
            assert e["burgers_norm"] < 1e-9
        elif w["direction_frac"] == (2, 1):
            assert e["ending_kind"] == "partial_disclination"
            expected = 2.0 * (math.degrees(math.atan2(1.0, 2.0)) - 45.0)
            assert abs(e["frank_angle_deg"] - expected) < 1e-6


@pytest.mark.parametrize("picture", ["raw", "relaxed"])
def test_half_wall_brick_pg_wd1(picture):
    """Surgical half-wall of brick_pg Wd1 produces exact fractional charge |b| = 0.5 * period."""
    cat = DefectCatalogue("brick_pg")
    wd1 = cat.glide_breaking[0]
    conf = build_half_wall("brick_pg", wd1, at=(0.5, 0.0), picture=picture)
    assert _fast.check_frames(conf.defect) == 0
    q = conf.total_charge()
    assert q["kind"] == "partial_dislocation"
    assert q["lattice_vector"] is False
    assert np.allclose(q["b"], wd1["ending"]["burgers"], atol=1e-6)
    assert abs(q["length"] - 0.5 * cat.mirrors[0]["period"]) < 1e-6
    circ = auto_burgers_circuit(conf)
    assert circ["charge"]["kind"] == "partial_dislocation"
    assert np.allclose(circ["closure_vector"], q["b"], atol=1e-6)


@pytest.mark.parametrize("wall_id", ["Wa1", "Wd2"])
def test_shastry_sutherland_half_wall_and_json_roundtrip(wall_id):
    """Shastry-Sutherland Wa1 and Wd2 via build(..., kind='half_wall') match sigma*g holonomy and survive JSON."""
    cat = DefectCatalogue("shastry_sutherland")
    entry = next(w for w in (cat.domain_walls + cat.glide_breaking) if w["id"] == wall_id)
    ending = entry["ending"]
    conf = build("shastry_sutherland", dict(kind="half_wall", wall_id=wall_id, at=(0.371, 0.413), direction=(1.0, 0.0)))
    assert _fast.check_frames(conf.defect) == 0
    q = conf.total_charge()
    assert q["kind"] == "partial_dislocation"
    assert np.allclose(q["b"], ending["burgers"], atol=1e-6)
    assert np.allclose(q["H"][1], ending["holonomy_trans"], atol=1e-6)
    assert equivalent_single(conf)["match_tier"] == "none"

    data = config_to_dict(conf, grid=1)
    assert data["defects"][0]["kind"] == "half_wall"
    assert data["summary"]["match_tier"] == "none"
    reloaded = load_config(data)
    assert np.allclose(reloaded.total_charge()["b"], q["b"], atol=1e-6)


def test_partial_disclination_half_wall_and_obstruction_refusals():
    """Partial disclination half-walls (chiral_triangular Wa3 +60, brick_pg Wa4 +180) build cleanly;
    non-orientable / chiral-obstructed walls are refused."""
    c60 = build_half_wall("chiral_triangular", "Wa3", at=(0.0, -0.2886751345948129), direction=(1.0, 0.0), picture="raw")
    q60 = c60.total_charge()
    assert q60["kind"] == "disclination" and abs(q60["angle"] - 60.0) < 1e-6 and q60["single"] is None
    assert equivalent_single(c60)["equivalent"] is False

    c180 = build_half_wall("brick_pg", "Wa4", at=(0.0, 0.0), direction=(1.0, 0.0), picture="relaxed")
    q180 = c180.total_charge()
    assert q180["kind"] == "disclination" and abs(q180["angle"] - 180.0) < 1e-6 and q180["single"] is None

    with pytest.raises(ValueError, match="Mobius|non-orientable"):
        build_half_wall("brick_pg", "Wc1")
    with pytest.raises(ValueError, match="chiral"):
        build_half_wall("chiral_triangular", "Wa6")
    with pytest.raises(ValueError, match="chiral"):
        build_half_wall("chiral_square", "Wa2")


def test_half_wall_combined_with_point_dislocation():
    """A half-wall combined with a point dislocation preserves frame consistency and sums holonomies."""
    lat = lattice_data("square")
    conf = build(
        lat,
        dict(kind="half_wall", wall_id="Wa1", at=(0.371, 0.413), direction=(1.0, 0.0)),
        dict(kind="dislocation", b=(1.0, 0.0), at=(-3.5, 0.413)),
        picture="raw",
    )
    assert _fast.check_frames(conf.defect) == 0
    q = conf.total_charge()
    assert q["kind"] == "partial_dislocation"
    assert np.allclose(q["b"], [1.5, 0.5], atol=1e-6)
