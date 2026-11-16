"""Tests for 360-degree Volterra dislocation insertion surgery and same-point defect superposition."""

import numpy as np
import pytest

from papertilling.defects.build import build, isomorphic_cores, lattice_data
from papertilling.defects.catalogue import DefectCatalogue
from papertilling.defects.config import equivalent_single


@pytest.mark.parametrize("name", ["square", "honeycomb", "kagome", "shastry_sutherland", "oblique"])
@pytest.mark.parametrize("picture", ["raw", "relaxed"])
def test_dislocation_insertion_matches_removal_across_lattices(name, picture):
    lat = lattice_data(name)
    cat = DefectCatalogue(name)
    at = tuple(lat.A @ np.array([0.371, 0.413], float))
    for d in cat.dislocations[:2]:
        bx, by = float(d["b"][0]), float(d["b"][1])
        c_rem = build(lat, dict(kind="dislocation", b=(bx, by), at=at, cut=(-by, bx)), picture=picture)
        c_ins = build(lat, dict(kind="dislocation", b=(bx, by), at=at, cut=(by, -bx)), picture=picture)
        q_ins = c_ins.total_charge()
        assert q_ins["kind"] == "dislocation"
        assert np.allclose(q_ins["b"], [bx, by], atol=1e-6)
        assert isomorphic_cores(c_ins.core_graph(2), c_rem.core_graph(2))


def test_symmetric_glide_and_climb_dislocation_dipoles():
    sq = lattice_data("square")
    cases = [
        (dict(kind="dislocation", b=(1, 0), at=(0.37, 0.41)), dict(kind="dislocation", b=(-1, 0), at=(3.37, 0.41))),
        (dict(kind="dislocation", b=(-1, 0), at=(1.37, 0.41)), dict(kind="dislocation", b=(1, 0), at=(2.37, 0.41))),
        (dict(kind="dislocation", b=(1, 0), at=(0.37, 0.41)), dict(kind="dislocation", b=(-1, 0), at=(1.37, 2.41))),
        (dict(kind="dislocation", b=(-1, 0), at=(1.37, 0.41)), dict(kind="dislocation", b=(1, 0), at=(0.37, 2.41))),
    ]
    for s0, s1 in cases:
        conf = build(sq, s0, s1)
        assert conf.total_charge()["kind"] == "vacuum"
        assert [c["ring"] for c in conf.cores()] == [5, 5]


def test_same_point_superposition_and_refusals():
    sq = lattice_data("square")

    c_vac_disc = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0, 0)))
    assert c_vac_disc.total_charge()["kind"] == "vacuum"
    assert c_vac_disc.cores() == []
    assert equivalent_single(c_vac_disc, grid=1)["equivalent"] is True

    c_180 = build(sq, dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=90, at=(0, 0)))
    assert c_180.total_charge()["kind"] == "disclination"
    assert abs(c_180.total_charge()["angle"] - 180.0) < 1e-6
    assert equivalent_single(c_180, grid=1)["equivalent"] is True

    c_vac_disl = build(
        sq,
        dict(kind="dislocation", b=(1, 0), at=(0.371, 0.413)),
        dict(kind="dislocation", b=(-1, 0), at=(0.371, 0.413)),
    )
    assert c_vac_disl.total_charge()["kind"] == "vacuum"
    assert c_vac_disl.cores() == []

    c_diag = build(
        sq,
        dict(kind="dislocation", b=(1, 0), at=(0.371, 0.413)),
        dict(kind="dislocation", b=(0, 1), at=(0.371, 0.413)),
    )
    assert c_diag.total_charge()["kind"] == "dislocation"
    assert np.allclose(c_diag.total_charge()["b"], [1.0, 1.0], atol=1e-6)
    assert equivalent_single(c_diag, grid=1)["equivalent"] is True

    c_mixed = build(
        sq,
        dict(kind="disclination", angle=90, at=(0, 0)),
        dict(kind="dislocation", b=(1, 0), at=(0, 0)),
        picture="raw",
    )
    assert c_mixed.total_charge()["kind"] == "disclination"
    assert abs(c_mixed.total_charge()["angle"] - 90.0) < 1e-6

    with pytest.raises(ValueError, match="360"):
        build(sq, dict(kind="disclination", angle=180, at=(0, 0.5)), dict(kind="disclination", angle=180, at=(0, 0.5)))
    with pytest.raises(ValueError, match="-360"):
        build(sq, dict(kind="disclination", angle=-180, at=(0, 0.5)), dict(kind="disclination", angle=-180, at=(0, 0.5)))
    with pytest.raises(ValueError, match="conflicting cut"):
        build(
            sq,
            dict(kind="disclination", angle=90, at=(0, 0), cut=(1, 0)),
            dict(kind="disclination", angle=90, at=(0, 0), cut=(0, 1)),
        )
