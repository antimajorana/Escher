"""Rust-written Notebook 3 configurations (papertilling.config.v1) open in Python.

Fixtures: tests/fixtures/config_rust_*.json, written by the Rust writer (NotebookConfigJson::to_json_pretty, the
format of the app's Export button) in rust/papertilling-core/tests/config_json_rust_fixtures.rs; their `summary`
is Rust's equivalent_single. Expected values: by hand in the docstrings, and that summary, recomputed here by
the Python package (the two implementations must agree). The Python-written direction is tested on the Rust side
(tests/fixtures/config_py_*.json, written by tools/checks/combine_parity_dump.py).
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from papertilling.defects.config import equivalent_single, load_config

FIX = Path(__file__).resolve().parent / "fixtures"
RUST_FILES = sorted(FIX.glob("config_rust_*.json"))


def test_rust_fixtures_present():
    assert [p.name for p in RUST_FILES] == ["config_rust_honeycomb_57.json", "config_rust_shastry_sutherland.json",
                                            "config_rust_square_pair.json"]


@pytest.mark.parametrize("path", RUST_FILES, ids=lambda p: p.stem)
def test_rust_written_config_reproduces_its_summary(path):
    raw = json.loads(path.read_text(encoding="utf8"))
    assert raw["format"] == "papertilling.config.v1"
    conf = load_config(path)
    opts = raw["options"]
    if opts.get("r_patch") is not None:
        assert conf.report["R"] == opts["r_patch"]
    q = conf.total_charge()
    res = equivalent_single(conf, radius=opts["radius"], grid=opts["grid"])
    s = raw["summary"]
    assert (q["label"], q["kind"]) == (s["charge"], s["kind"])
    assert (res["equivalent"], res["matches"], res["verdict"]) == (s["equivalent"], s["matches"], s["verdict"])


def test_square_pair_by_hand():
    """+90 at a site and -90 at a face centre: the unit dislocation |b| = 1 (test_config.test_user_example_square),
    with a pentagon core (C01)."""
    conf = load_config(FIX / "config_rust_square_pair.json")
    q = conf.total_charge()
    assert q["kind"] == "dislocation" and abs(np.linalg.norm(q["b"]) - 1.0) < 1e-9
    assert [c["ring"] for c in conf.cores() if c.get("site") is None] == [5]


def test_shastry_sutherland_frame_nulls_and_default_at():
    """Rust writes recipe b/gamma as null and omits a missing dislocation `at`; the frame origin (0.5, -0.5) is kept.
    Hand: +90 at c = (0.5, 1.5) (class 2a(i)) and b = (2, 0). The outer holonomy is (R, (I - R)c)(I, -b) or
    (I, -b)(R, (I - R)c), a +90 rotation about c - (I - R)^-1 R b = (1.5, 0.5) or c - (I - R)^-1 b = (-0.5, 0.5);
    both differ from c by (1, 1) mod the lattice 2Z^2, i.e. lie on the other 4-fold class 2a(ii) of p4g."""
    path = FIX / "config_rust_shastry_sutherland.json"
    raw = json.loads(path.read_text(encoding="utf8"))
    assert raw["recipe"]["b"] is None and raw["recipe"]["gamma"] is None
    assert raw["frame"]["origin"] == [0.5, -0.5]
    assert "at" not in raw["defects"][1] and raw["defects"][1]["cut"] == [0.3, 1.0]
    q = load_config(path).total_charge()
    assert q["kind"] == "disclination" and q["angle"] == 90 and q["centre_class"] == "2a(ii)"
    c = np.asarray(q["centre"], float)
    same_mod_2z2 = lambda p: np.allclose((c - p) / 2.0, np.round((c - p) / 2.0), atol=1e-6)
    assert same_mod_2z2((1.5, 0.5)) and same_mod_2z2((-0.5, 0.5))


def test_honeycomb_57_by_hand():
    """+60 / -60 at neighbouring hexagon centres: |b| = sqrt3 and a 5-7 core (test_config.test_honeycomb_57)."""
    conf = load_config(FIX / "config_rust_honeycomb_57.json")
    q = conf.total_charge()
    assert q["kind"] == "dislocation" and abs(np.linalg.norm(q["b"]) - math.sqrt(3)) < 1e-9
    assert sorted(c["ring"] for c in conf.cores()) == [5, 7]
