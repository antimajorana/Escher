"""Stage 2 solver export & defect tests (both Rust-accelerated and pure-Python fallback paths).

Independent hand-derived controls:
  * Square (p4m): J1 = 4 (d=1), J2 = 4 (d=sqrt(2)), J3 = 4 (d=2), 1 orbit each;
    4x4 torus has 16 sites and symmetry order 16 * 8 = 128, with J3 collapsing +-2a1/+-2a2 to multiplicity 2.
  * Honeycomb (p6m): J1 = 3 (d=1), J2 = 6 (d=sqrt(3)), J3 = 3 (d=2), 1 orbit each;
    -60 deg disclination at a 6-fold hexagon centre has D5 symmetry (order 10).
  * Kagome (p6m): J1 = 4 (d=1, 1 orbit), J2 = 4 (d=sqrt(3), 1 orbit),
    J3 = 6 (d=2) split into 2 orbits with per-site coordinations 4 (straight chains) and 2 (hexagon diagonals).
  * Square +90 deg disclination at a 4-fold site (0, 0) has D3 symmetry (order 6).
  * CSL symmetric-tilt hand formulas: hex Sigma=7 -> 21.786789 / 38.213211 deg; sq Sigma=5 -> 36.869898 / 53.130102 deg.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

import papertilling as pt
from papertilling import _rust

ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")


def _cell_json(lat) -> str:
    c = lat.cell()
    return json.dumps({
        "a1": [float(v) for v in c.a1],
        "a2": [float(v) for v in c.a2],
        "basis": [[float(x) for x in p] for p in c.basis],
        "bonds": [{"i": int(b[0]), "j": int(b[1]), "n1": int(b[2]), "n2": int(b[3]), "kind": int(b[4])} for b in c.bonds],
    })


@pytest.mark.parametrize("no_rust", ["1", "0"])
def test_bond_shells_and_orbits_hand_counts(monkeypatch, no_rust):
    monkeypatch.setenv("PAPERTILLING_NO_RUST", no_rust)

    sq = pt.classify_bond_shells(pt.named("Square"), max_shells=3)
    d1 = sq["shells"][0]["distance"]
    for idx, (coord, ratio) in enumerate([(4, 1.0), (4, math.sqrt(2.0)), (4, 2.0)]):
        sh = sq["shells"][idx]
        assert sh["coordination_by_basis"] == [coord]
        assert abs(sh["distance"] / d1 - ratio) < 1e-9
        assert len(sh["orbits"]) == 1

    hc = pt.classify_bond_shells(pt.named("Honeycomb"), max_shells=3)
    d1 = hc["shells"][0]["distance"]
    for idx, (coord, ratio) in enumerate([(3, 1.0), (6, math.sqrt(3.0)), (3, 2.0)]):
        sh = hc["shells"][idx]
        assert sh["coordination_by_basis"] == [coord, coord]
        assert abs(sh["distance"] / d1 - ratio) < 1e-9
        assert len(sh["orbits"]) == 1

    kg = pt.classify_bond_shells(pt.named("Kagomé"), max_shells=3)
    d1 = kg["shells"][0]["distance"]
    for idx, (coord, ratio) in enumerate([(4, 1.0), (4, math.sqrt(3.0)), (6, 2.0)]):
        sh = kg["shells"][idx]
        assert sh["coordination_by_basis"] == [coord, coord, coord]
        assert abs(sh["distance"] / d1 - ratio) < 1e-9
    assert len(kg["shells"][0]["orbits"]) == 1
    assert len(kg["shells"][1]["orbits"]) == 1
    j3_orbits = sorted(o["coordination_by_basis"] for o in kg["shells"][2]["orbits"])
    assert j3_orbits == [[2, 2, 2], [4, 4, 4]]


@pytest.mark.parametrize("no_rust", ["1", "0"])
def test_extxyz_periodic_and_open_crystal(monkeypatch, no_rust):
    monkeypatch.setenv("PAPERTILLING_NO_RUST", no_rust)

    hc = pt.named("Honeycomb")
    txt_per = pt.export_extxyz(hc, supercell=(2, 3))
    assert "\r" not in txt_per
    lines = txt_per.splitlines()
    assert len(lines) == 2 * 3 * 2 + 2
    assert lines[0] == "12"
    assert 'pbc="T T F"' in lines[1]
    for row in lines[2:]:
        sp, xs, ys, zs, sub, coord = row.split()
        assert ELEMENT_RE.match(sp)
        assert len(xs.split(".")[1]) == 8 and len(ys.split(".")[1]) == 8 and zs == "0.00000000"
        assert int(sub) in (0, 1) and int(coord) == 3

    sq = pt.named("Square")
    cfg = pt.build(sq, {"kind": "disclination", "angle": 90.0, "at": (0.0, 0.0)}, R=3.5, picture="relaxed")
    n_sites = len(cfg.defect.positions)
    txt_open = pt.export_extxyz(cfg, species=["Si"] * n_sites, z=[0.25 * i for i in range(n_sites)])
    assert "\r" not in txt_open
    olines = txt_open.splitlines()
    assert len(olines) == n_sites + 2
    assert olines[0] == str(n_sites)
    assert 'pbc="F F F"' in olines[1]
    deg = cfg.defect.degrees()
    for idx, row in enumerate(olines[2:]):
        sp, xs, ys, zs, sub, coord = row.split()
        assert sp == "Si"
        assert abs(float(xs) - cfg.defect.positions[idx, 0]) < 1e-7
        assert abs(float(ys) - cfg.defect.positions[idx, 1]) < 1e-7
        assert abs(float(zs) - 0.25 * idx) < 1e-7
        assert int(sub) == 0 and int(coord) == int(deg[idx])


@pytest.mark.parametrize("no_rust", ["1", "0"])
def test_supercell_4x4_square_and_open_cone_permutations(monkeypatch, no_rust):
    monkeypatch.setenv("PAPERTILLING_NO_RUST", no_rust)

    sq = pt.named("Square")
    res = pt.cluster_site_permutations(sq, supercell=(4, 4), max_shells=3)
    n = 16
    assert res["n_sites"] == n
    assert res["order"] == 128
    assert res["group_closed"] is True
    assert res["permutations"][0] == list(range(n))

    for s_idx, (distinct, total) in enumerate([(4, 4), (4, 4), (2, 4)]):
        adj = np.zeros((n, n), dtype=int)
        for u, v, mult in res["shell_edges"][s_idx]:
            adj[u, v] = mult
            adj[v, u] = mult
        assert np.all(np.count_nonzero(adj, axis=1) == distinct)
        assert np.all(adj.sum(axis=1) == total)
        for pi in res["permutations"]:
            assert np.array_equal(adj[np.ix_(pi, pi)], adj)

    # Square +90 deg disclination at a site -> D3 cone (order 6) under raw / GSV elastic relaxation
    cone_sq = pt.build(sq, {"kind": "disclination", "angle": 90.0, "at": (0.0, 0.0)}, R=3.5, picture="elastic")
    rep_sq = pt.cluster_site_permutations(cone_sq)
    assert rep_sq["group_closed"] is True
    assert rep_sq["order"] == 6
    assert rep_sq["permutations"][0] == list(range(rep_sq["n_sites"]))
    adj_sq = np.zeros((rep_sq["n_sites"], rep_sq["n_sites"]), dtype=int)
    for u, v in cone_sq.defect.edges:
        adj_sq[u, v] = adj_sq[v, u] = 1
    for pi in rep_sq["permutations"]:
        assert np.array_equal(adj_sq[np.ix_(pi, pi)], adj_sq)

    # Honeycomb +60 deg disclination at a 6-fold hexagon centre -> D5 pentagon cone (order 10); -60 deg -> D7 (order 14)
    hc = pt.named("Honeycomb")
    cat = pt.DefectCatalogue(hc)
    c6 = next(c for c in cat.centre_classes if c["order"] == 6)
    cone_hc5 = pt.build(hc, {"kind": "disclination", "angle": 60.0, "at": tuple(c6["point"])}, R=4.0, picture="elastic")
    rep_hc5 = pt.cluster_site_permutations(cone_hc5)
    assert rep_hc5["group_closed"] is True
    assert rep_hc5["order"] == 10
    cone_hc7 = pt.build(hc, {"kind": "disclination", "angle": -60.0, "at": tuple(c6["point"])}, R=4.0, picture="elastic")
    rep_hc7 = pt.cluster_site_permutations(cone_hc7)
    assert rep_hc7["group_closed"] is True
    assert rep_hc7["order"] == 14


def test_rust_vs_python_exact_parity(monkeypatch):
    rs = pytest.importorskip("papertilling_rs")
    if not hasattr(rs, "periodic_supercell_solver_json"):
        pytest.skip("installed papertilling_rs predates Stage 2 solver exports")

    for name in ("Square", "Honeycomb", "Kagomé"):
        lat = pt.named(name)
        monkeypatch.setenv("PAPERTILLING_NO_RUST", "0")
        assert _rust.get() is rs
        sh_rs = pt.classify_bond_shells(lat, max_shells=3)
        xyz_rs = pt.export_extxyz(lat, supercell=(2, 2))
        sup_rs = pt.cluster_site_permutations(lat, supercell=(3, 3), max_shells=2)

        monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
        assert _rust.get() is None
        sh_py = pt.classify_bond_shells(lat, max_shells=3)
        xyz_py = pt.export_extxyz(lat, supercell=(2, 2))
        sup_py = pt.cluster_site_permutations(lat, supercell=(3, 3), max_shells=2)

        assert sh_rs == sh_py
        assert xyz_rs == xyz_py
        assert sup_rs["order"] == sup_py["order"]
        assert sup_rs["permutations"] == sup_py["permutations"]
        assert sup_rs["shell_edges"] == sup_py["shell_edges"]


def test_rust_defects_csl_stone_wales_and_point_defects():
    rs = pytest.importorskip("papertilling_rs")
    if not hasattr(rs, "list_csl_grain_boundaries_json"):
        pytest.skip("installed papertilling_rs predates Stage 2 defect bindings")

    hc = pt.named("Honeycomb")
    sq = pt.named("Square")
    csl_hc = json.loads(rs.list_csl_grain_boundaries_json(hc.group, _cell_json(hc), 19))
    s7 = [e["theta_deg"] for e in csl_hc if e["sigma"] == 7]
    assert len(s7) == 2
    assert abs(s7[0] - 21.786789) < 1e-6 and abs(s7[1] - 38.213211) < 1e-6

    csl_sq = json.loads(rs.list_csl_grain_boundaries_json(sq.group, _cell_json(sq), 17))
    s5 = [e["theta_deg"] for e in csl_sq if e["sigma"] == 5]
    assert len(s5) == 2
    assert abs(s5[0] - 36.869898) < 1e-6 and abs(s5[1] - 53.130102) < 1e-6

    # Build a clean finite honeycomb patch via a neutral +60/-60 pair? Easier: square/honeycomb via pt.build or crystal dict
    from papertilling.solver_export import _extract_crystal
    cfg_hc = pt.build(hc, {"kind": "dislocation", "b": (math.sqrt(3.0), 0.0), "at": (2.5, 2.5)}, R=9.0, picture="raw")
    crystal_dict, _, _, _ = _extract_crystal(cfg_hc)
    # Pick an interior bond far from the dislocation core near (-2.0, -2.0)
    pos = np.asarray(crystal_dict["pos"], float)
    edges = crystal_dict["edges"]
    target = np.array([-2.0, -2.0])
    eidx = min(range(len(edges)), key=lambda k: np.linalg.norm(0.5 * (pos[edges[k][0]] + pos[edges[k][1]]) - target))
    sw = json.loads(rs.stone_wales_flip_json(json.dumps(crystal_dict), eidx, None))
    assert sw["before_rings"] == [6, 6, 6, 6]
    assert sorted(sw["after_rings"]) == [5, 5, 7, 7]
    assert len(sw["crystal"]["pos"]) == len(crystal_dict["pos"])
    assert len(sw["crystal"]["edges"]) == len(crystal_dict["edges"])

    # Point defects on a square patch
    cfg_sq = pt.build(sq, {"kind": "dislocation", "b": (1.0, 0.0), "at": (2.0, 2.0)}, R=7.0, picture="raw")
    sq_dict, _, _, _ = _extract_crystal(cfg_sq)
    spos = np.asarray(sq_dict["pos"], float)
    host = int(np.argmin(np.linalg.norm(spos - np.array([-1.0, -1.0]), axis=1)))
    vac = int(np.argmin(np.linalg.norm(spos - np.array([0.0, 0.0]), axis=1)))
    ops = [
        {"Substitutional": {"site": host, "species": "Si"}},
        {"Adatom": {"host_site": host, "height": 1.5, "species": "O"}},
        {"Vacancy": {"site": vac, "reconstruct_ring": False}},
    ]
    pd = json.loads(rs.apply_point_defects_json(json.dumps(sq_dict), json.dumps(ops)))
    assert len(pd["crystal"]["pos"]) == len(sq_dict["pos"]) + 1 - 1
    new_host = int(np.argmin(np.linalg.norm(np.asarray(pd["crystal"]["pos"]) - spos[host], axis=1)))
    assert pd["species"][new_host] == "Si"
    assert pd["pos3d"][-1][2] == 1.5
    xyz_pd = pt.export_extxyz(pd)
    assert xyz_pd.startswith(f"{len(pd['crystal']['pos'])}\n")
