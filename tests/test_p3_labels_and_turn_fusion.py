"""Wyckoff labels agree between DefectCatalogue and wyckoff_rotocentres (p3 1b/1c), and a whole-turn fusion has channels."""

import io
import contextlib
import warnings

import numpy as np
import pytest

import papertilling
from papertilling.defects import DefectCatalogue, fusion_channels, wyckoff_rotocentres
from papertilling.defects.gauge import _to_lat

warnings.filterwarnings("ignore")


def _names():
    with contextlib.redirect_stdout(io.StringIO()):
        return [d["key"] for d in papertilling.named_lattices()]


@pytest.mark.parametrize("name", _names())
def test_catalogue_and_wyckoff_rotocentres_use_the_same_labels(name):
    lat = _to_lat(name)
    cat = DefectCatalogue(name)
    wr = wyckoff_rotocentres(name)
    assert {c["label"] for c in cat.centre_classes} == set(wr["by_label"])
    for rec in cat.centre_classes:
        assert lat.centre_class(rec["point"])["label"] == rec["label"]
        assert any(lat.same_mod_lattice(rec["point"], q) for q in wr["by_label"][rec["label"]]["members"])
    for d in cat.disclinations:                         # "<angle> deg at <label> (<where>)"
        label = d["label"].split(" at ")[1].split(" (")[0]
        assert lat.centre_class(d["spec"]["at"])["label"] == label


def test_p3_labels_follow_the_international_tables():
    # ITA p3 (hexagonal axes a, b at 120 degrees, right-handed): 1a (0,0), 1b (1/3,2/3), 1c (2/3,1/3).
    # The reduced primitive cell is right-handed, so a centre class is labelled by its position relative to 1a
    # in the fractional coordinates of that cell (hand computation, not read from the code under test).
    cat = DefectCatalogue("chiral_triangular")
    wr = wyckoff_rotocentres("chiral_triangular")
    A = _to_lat("chiral_triangular").A
    assert np.linalg.det(A) > 0
    assert wr["group"] == "p3"
    pts = {c["label"]: np.asarray(c["point"], float) for c in cat.centre_classes}
    assert set(pts) == {"1a", "1b", "1c"}
    for lab, shift in (("1b", (1 / 3, 2 / 3)), ("1c", (2 / 3, 1 / 3))):
        f = np.linalg.inv(A) @ (pts[lab] - pts["1a"]) - np.array(shift)
        assert np.allclose(f, np.round(f), atol=1e-6), lab


def _disc(angle, at):
    return dict(kind="disclination", angle=angle, at=at)


def test_whole_turn_fusion_reports_the_burgers_channels():
    # Two +180 degree disclinations: R1 R2 = 1, so the product is the translation b = (R1 - 1) D = -2 D (D = c2 - c1).
    f = fusion_channels("square", [_disc(180, (0, 0)), _disc(180, (0.5, 0.5))])
    assert f["law"] == "D×D (turn)"
    assert f["channels"], "a 360 degree total angle must not give an empty result"
    assert f["contains_vacuum"] is False                 # a full turn is not the vacuum, even with b = 0
    best = f["minimal_channels"][0]
    assert best["kind"] == "dislocation"
    assert np.allclose(best["b"], [-1.0, -1.0], atol=1e-9)
    assert abs(best["frank_angle_deg"] - 360.0) < 1e-9
    # Same centre: D = 0, so b = 0 (a pure turn) is a channel, next b = -2 (lattice translations of c2 give D = 1).
    g = fusion_channels("square", [_disc(180, (0, 0)), _disc(180, (0, 0))])
    norms = sorted(round(float(np.linalg.norm(c["b"])), 6) for c in g["channels"])
    assert norms[:2] == [0.0, 2.0]
    assert g["contains_vacuum"] is False
    # -360 behaves the same way.
    h = fusion_channels("square", [_disc(-180, (0, 0)), _disc(-180, (0.5, 0.5))])
    assert h["law"] == "D×D (turn)" and h["channels"]
    assert abs(h["channels"][0]["frank_angle_deg"] + 360.0) < 1e-9
