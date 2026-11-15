"""Stage 5 & 6 tests: optional papertilling_rs bridge, PAPERTILLING_NO_RUST toggle,
and portable Notebook 3 .json defect configuration import/export.

Hand-derived expected invariants:
  * square (p4m): 2 undirected bonds/cell -> 8 chambers/cell; |p4m / T| = 8 -> 1 D-chamber
    with m01 = 4, m12 = 4, canonical key ((4, 4, 0, 0, 0),), symmetry_index = 1.
    Seen through p2 (oblique): |p2 / T| = 2 -> 4 D-chambers, minimal image ((4, 4, 0, 0, 0),),
    symmetry_index = 8 / 2 = 4.
  * p4g #1 (rule="bonds", rotating squares, parameter t1): supergroup locus t1 = -1/2 gives p4m.
  * square +90 at site (0, 0) and -90 at face centre (1/2, 1/2): outer CCW holonomy has Frank
    angle 0 and |b| = 1 (unit dislocation T1), local core = pentagon (5-ring) sharing one
    3-coordinated site; by C01 this equals the Volterra core of a single unit dislocation.
  * shastry_sutherland (p4g): stored in frame with non-zero origin (0.5, -0.5); its two 4-fold
    centres (Wyckoff 2a) split in G+ = p4 into 2a(i) at (0.5, 1.5) and 2a(ii), so preserving
    frame on JSON reload is required to keep the G~+ charge class intact.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from papertilling import named
from papertilling.defects.build import build, isomorphic_cores, same_charge
from papertilling.dsymbol import dsymbol, network_key, symmetry_index
from papertilling.lattice import load_catalogue, ops_in_cell
from papertilling.tuning import allowed_interval, special_points

EXPECTED_CONFIG_KEYS = {
    "format",
    "lattice_label",
    "group",
    "recipe",
    "frame",
    "defects",
    "options",
    "summary",
}

SQUARE_KEY = ((4, 4, 0, 0, 0),)
OBLIQUE_P2_KEY = (
    (4, 4, 1, 2, 1),
    (4, 4, 0, 3, 0),
    (4, 4, 3, 0, 3),
    (4, 4, 2, 1, 2),
)


def _rs_iso_cores(rs, deg1, edges1, deg2, edges2):
    return rs.isomorphic_cores(len(deg1), deg1, edges1, len(deg2), deg2, edges2)


def test_no_rust_toggle_and_known_invariants(monkeypatch):
    """PAPERTILLING_NO_RUST=1 disables _rust dynamically; D-symbol and tuning invariants match hand values."""
    from papertilling import _rust

    sq = named("square")
    ob = named("oblique")
    l4 = load_catalogue(group="p4g", rule="bonds").lattices[0]

    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
    assert _rust.enabled() is False
    assert _rust.get() is None

    ds_py = dsymbol(sq.cell())
    ob_py = dsymbol(ob.cell(), ops_in_cell(ob.family))
    assert ds_py["key"] == SQUARE_KEY
    assert network_key(ds_py["key"]) == SQUARE_KEY
    assert symmetry_index(ds_py["key"]) == 1
    assert ob_py["key"] == OBLIQUE_P2_KEY
    assert network_key(ob_py["key"]) == SQUARE_KEY
    assert symmetry_index(ob_py["key"]) == 4

    iv_py = allowed_interval(l4, "t1", -1.0, 0.5)
    sp_py = special_points(l4, "t1", -1.0, 0.5)
    assert iv_py["current_ok"] is True
    assert iv_py["interval"][0] <= l4.values["t1"] <= iv_py["interval"][1]
    hit_py = [d for d in sp_py if abs(d["value"] + 0.5) < 1e-6]
    assert len(hit_py) == 1 and hit_py[0]["group"] == "p4m"

    monkeypatch.setenv("PAPERTILLING_NO_RUST", "0")
    assert _rust.enabled() is (_rust.get() is not None)
    assert dsymbol(sq.cell())["key"] == SQUARE_KEY
    assert network_key(OBLIQUE_P2_KEY) == SQUARE_KEY
    assert symmetry_index(OBLIQUE_P2_KEY) == 4
    iv_def = allowed_interval(l4, "t1", -1.0, 0.5)
    sp_def = special_points(l4, "t1", -1.0, 0.5)
    assert np.allclose(iv_def["interval"], iv_py["interval"], atol=1e-5)
    assert [(round(d["value"], 7), d["group"]) for d in sp_def] == [
        (round(d["value"], 7), d["group"]) for d in sp_py
    ]


def test_notebook3_config_json_roundtrip_and_rust_import(tmp_path):
    """Notebook 3 .json round-trip on square and shastry_sutherland + direct Rust NotebookConfigJson import."""
    from papertilling.defects.config import config_to_dict, load_config, save_config

    sq = named("square")
    pair = build(
        sq,
        dict(kind="disclination", angle=90, at=(0.0, 0.0)),
        dict(kind="disclination", angle=-90, at=(0.5, 0.5)),
        picture="raw",
    )
    d_pair = config_to_dict(pair, grid=1)
    assert set(d_pair.keys()) == EXPECTED_CONFIG_KEYS
    assert d_pair["format"] == "papertilling.config.v1"
    assert d_pair["group"] == "p4m"
    assert {"q", "origin"} <= set(d_pair["frame"].keys())
    assert len(d_pair["defects"]) == 2
    assert all(spec["kind"] == "disclination" and "angle" in spec and "at" in spec for spec in d_pair["defects"])

    pair_file = tmp_path / "pair.json"
    save_config(pair, pair_file, grid=1)
    on_disk = json.loads(pair_file.read_text(encoding="utf-8"))
    assert set(on_disk.keys()) == EXPECTED_CONFIG_KEYS

    pair2 = load_config(pair_file)
    q2 = pair2.total_charge()
    assert q2["kind"] == "dislocation" and abs(np.linalg.norm(q2["b"]) - 1.0) < 1e-9
    assert [c["ring"] for c in pair2.cores() if c.get("site") is None] == [5]
    assert same_charge(pair.lattice, pair.total_charge(), q2)
    assert isomorphic_cores(pair.core_graph(2), pair2.core_graph(2))

    # Non-trivial frame control: Shastry-Sutherland (p4g) has frame origin (0.5, -0.5) and split 2a centres.
    ss = named("shastry_sutherland")
    ss_conf = build(ss, dict(kind="disclination", angle=90, at=(0.5, 1.5)), picture="raw")
    ss_file = tmp_path / "ss.json"
    save_config(ss_conf, ss_file, grid=1)
    d_ss = json.loads(ss_file.read_text(encoding="utf-8"))
    assert set(d_ss.keys()) == EXPECTED_CONFIG_KEYS
    assert d_ss["group"] == "p4g"
    assert np.allclose(d_ss["frame"]["origin"], [0.5, -0.5])

    ss2 = load_config(ss_file)
    assert same_charge(ss_conf.lattice, ss_conf.total_charge(), ss2.total_charge())
    assert isomorphic_cores(ss_conf.core_graph(2), ss2.core_graph(2))

    # Direct Rust -> Python import using exact NotebookConfigJson raw JSON string (C01: unit dislocation == pair).
    rust_json = json.dumps(
        {
            "format": "papertilling.config.v1",
            "lattice_label": "Square",
            "group": "p4m",
            "recipe": {"group": "p4m", "sites": [[0.0, 0.0]], "bonds": [[[0.0, 0.0], [-1.0, 0.0]]], "a": 1.0},
            "frame": {"q": [[1.0, 0.0], [0.0, 1.0]], "origin": [0.0, 0.0]},
            "defects": [{"kind": "dislocation", "b": [1.0, 0.0], "at": [0.371, 0.413]}],
            "options": {"picture": "raw", "grid": 1, "radius": 2},
            "summary": {
                "charge": "T1",
                "kind": "dislocation",
                "equivalent": True,
                "matches": ["Volterra core 1"],
                "verdict": "equivalent to the single defect (core: Volterra core 1)",
            },
        },
        indent=2,
    )
    from_rust = load_config(rust_json)
    assert same_charge(pair.lattice, pair.total_charge(), from_rust.total_charge())
    assert isomorphic_cores(pair.core_graph(2), from_rust.core_graph(2))


def test_papertilling_rs_direct_and_bridge_parity(monkeypatch):
    """Direct PyO3 bindings check and Rust-vs-Python parity when papertilling_rs is installed."""
    rs = pytest.importorskip("papertilling_rs")
    from papertilling import _rust

    assert rs.version() == "0.2.0"

    sig = [[1, 0, 3, 2], [2, 3, 0, 1], [1, 0, 3, 2]]
    k = rs.canonical_key(sig, [4, 4, 4, 4], [4, 4, 4, 4])
    assert k == OBLIQUE_P2_KEY
    rows = [list(r) for r in k]
    assert rs.network_key(rows) == SQUARE_KEY
    assert rs.symmetry_index(rows) == 4

    # 4-node degree-labelled graph isomorphism: positive + two negative controls (label mismatch & non-isomorphic topology).
    c4_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    c4_relab = [(0, 3), (3, 2), (2, 1), (1, 0)]
    paw_edges = [(0, 1), (1, 2), (2, 0), (0, 3)]
    assert _rs_iso_cores(rs, [3, 4, 4, 4], c4_edges, [4, 4, 3, 4], c4_relab) is True
    assert _rs_iso_cores(rs, [3, 4, 4, 4], c4_edges, [3, 3, 4, 4], c4_edges) is False
    assert _rs_iso_cores(rs, [3, 4, 4, 4], c4_edges, [3, 4, 4, 4], paw_edges) is False

    sq_cell_json = json.dumps(
        {
            "a1": [1.0, 0.0],
            "a2": [0.0, 1.0],
            "basis": [[0.0, 0.0]],
            "bonds": [
                {"i": 0, "j": 0, "n1": 1, "n2": 0, "kind": 0},
                {"i": 0, "j": 0, "n1": 0, "n2": 1, "kind": 0},
            ],
        }
    )
    surf_rows = json.loads(rs.surface_table_json("p4m", sq_cell_json))
    assert len(surf_rows) == 8 and all(r["status"] == "OK" for r in surf_rows)

    cat = json.loads(rs.defect_catalogue_json("p4m", sq_cell_json))
    assert cat["group"] == "p4m" and cat["chiral"] is False
    assert len(cat["dislocations"]) == 2
    assert [c["label"] for c in cat["centre_classes"]] == ["1a", "1b", "2c"]
    assert len(cat["disclinations"]) == 10
    assert len(cat["domain_walls"]) == 1 and len(cat["mirrors"]) == 4

    # Stage 5 live PyO3 bindings: nearest Wyckoff centre, C01 equivalent_single, and Volterra core gallery.
    nc = json.loads(rs.nearest_allowed_centre_json("p4m", sq_cell_json, 90.0, [0.48, 0.52]))
    assert nc["label"] == "1b" and np.allclose(nc["point"], [0.5, 0.5])
    pair_specs_json = json.dumps(
        [{"kind": "disclination", "angle": 90.0, "at": [0.0, 0.0]}, {"kind": "disclination", "angle": -90.0, "at": [0.5, 0.5]}]
    )
    eq = json.loads(rs.equivalent_single_json("p4m", sq_cell_json, pair_specs_json, grid=1))
    assert eq["equivalent"] is True and eq["matches"] == ["Volterra core 1"]
    gal = json.loads(rs.core_gallery_json("p4m", sq_cell_json, [1.0, 0.0], grid=1))
    assert [tuple(g["rings"]) for g in gal["gallery"]] == [(5,), (7,)]

    l4 = load_catalogue(group="p4g", rule="bonds").lattices[0]
    sq = named("square")
    monkeypatch.setenv("PAPERTILLING_NO_RUST", "0")
    assert _rust.enabled() is True and _rust.get() is rs
    ds_on = dsymbol(sq.cell())["key"]
    iv_on = allowed_interval(l4, "t1", -1.0, 0.5)
    sp_on = special_points(l4, "t1", -1.0, 0.5)

    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
    assert _rust.enabled() is False and _rust.get() is None
    ds_off = dsymbol(sq.cell())["key"]
    iv_off = allowed_interval(l4, "t1", -1.0, 0.5)
    sp_off = special_points(l4, "t1", -1.0, 0.5)

    assert ds_on == ds_off == SQUARE_KEY
    assert np.allclose(iv_on["interval"], iv_off["interval"], atol=1e-5)
    assert [(round(d["value"], 7), d["group"]) for d in sp_on] == [
        (round(d["value"], 7), d["group"]) for d in sp_off
    ]
