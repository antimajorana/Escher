"""C1 WP-N1: the canonical 3D net motifs (one table, `papertilling/_net_motifs.py` = core `space3d/motifs.rs`).

Known answers: space-group numbers and Wyckoff letters from spglib 2.8.0 (an independent tool) against the label
and the standard ITA settings (#227 origin choice 2, #180, #136, #221, #123, #194); hand-derived coordination numbers.
Wyckoff letters are fixed only up to the origin spglib picks (16c/16d, 3c/3d, 1a/1b): the group number is the real
check, the letters pin spglib's standard choice. spglib results are [COMPUTATIONAL EVIDENCE] at symprec 1e-5.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from papertilling import _rust
from papertilling._net_motifs import DEFECT_LABELS, NET_LABELS, NET_MOTIFS, build_net_patch
from papertilling.flat3d import canonical_cell_3d
from papertilling.space3d import CANONICAL_NETS_3D, build_net_3d

# ITA Wyckoff letters of each coloured motif (sorted, one per site).
ITA_WYCKOFF = {
    "SimpleCubic": "a", "BCC": "aa", "FCC": "aaaa", "Diamond": "aaaaaaaa", "HCP": "cc", "Tetragonal123": "ad",
    "Rutile136": "aaffff", "Perovskite": "abddd", "Pyrochlore": "c" * 16, "BetaQuartz": "ccc",
}


def _same_set_mod1(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        return False
    for p in b:
        d = np.abs((a - p + 0.5) % 1.0 - 0.5).max(axis=1)
        if int(np.sum(d < 1e-9)) != 1:
            return False
    return True


def test_space_group_and_wyckoff_equal_the_label_for_all_10_nets():
    spglib = pytest.importorskip("spglib")
    assert CANONICAL_NETS_3D == list(NET_MOTIFS) and len(NET_MOTIFS) == 10
    for name, (sg, lat, sites, _cut) in NET_MOTIFS.items():
        cell = (np.asarray(lat), [f for f, _ in sites], [k for _, k in sites])
        ds = spglib.get_symmetry_dataset(cell, symprec=1e-5)
        assert ds.number == sg, f"{name}: spglib {ds.number} {ds.international} != label #{sg}"
        assert f"#{sg}" in NET_LABELS[name]
        assert "".join(sorted(ds.wyckoffs)) == ITA_WYCKOFF[name], f"{name}: {ds.wyckoffs}"


def test_pyrochlore_and_beta_quartz_site_lists_match_ita():
    # #227 16c (origin choice 2) + F-centring, shifted by (1/8,1/8,1/8) to the diamond-at-origin setting.
    fc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    c16 = [(0, 0, 0), (0.75, 0.25, 0.5), (0.25, 0.5, 0.75), (0.5, 0.75, 0.25)]
    ita16c = [[s[k] + t[k] + 0.125 for k in range(3)] for t in fc for s in c16]
    assert _same_set_mod1([f for f, _ in NET_MOTIFS["Pyrochlore"][2]], ita16c)
    # #180 3c: 1/2,0,0; 0,1/2,2/3; 1/2,1/2,1/3 (the pre-N1 set with z = 1/3, 2/3 swapped is #181's 3c).
    assert _same_set_mod1([f for f, _ in NET_MOTIFS["BetaQuartz"][2]], [(0.5, 0, 0), (0, 0.5, 2 / 3), (0.5, 0.5, 1 / 3)])
    # Rutile has 6 sites: Ti 2a + O 4f (x = 0.3).
    assert [k for _, k in NET_MOTIFS["Rutile136"][2]] == [0, 0, 1, 1, 1, 1]


@pytest.mark.parametrize("net,want", [("SimpleCubic", 6), ("BCC", 8), ("FCC", 12), ("Diamond", 4), ("Pyrochlore", 6),
                                      ("HCP", 12), ("BetaQuartz", 4)])
def test_coordination_of_the_periodic_cell(net, want):
    # Hand values: sc 6, bcc 8, fcc 12, diamond 4, pyrochlore 6 (corner-sharing tetrahedra), hcp 12, Si in quartz 4.
    cell = canonical_cell_3d(net)
    deg = [0] * len(cell["basis"])
    for i, j, *_ in cell["bonds"]:
        deg[i] += 1
        deg[j] += 1
    assert deg == [want] * len(deg), deg


def test_fallback_patch_has_every_net_and_defect(monkeypatch):
    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
    for net, (_sg, _lat, sites, _cut) in NET_MOTIFS.items():
        patch = build_net_3d(net)
        assert len(patch["positions"]) == 8 * len(sites) and len(patch["site_kinds"]) == 8 * len(sites)
        assert patch["space_group_num"] == NET_MOTIFS[net][0]
        for d in DEFECT_LABELS:
            assert build_net_3d(net, d)["positions"], (net, d)
    # Pyrochlore bonds in the 2x2x2 block all have the hand length sqrt(2)/4.
    p = build_net_3d("Pyrochlore")
    pos = np.asarray(p["positions"])
    lens = [float(np.linalg.norm(pos[i] - pos[j])) for i, j, _ in p["bonds"]]
    assert lens and max(abs(x - math.sqrt(2) / 4) for x in lens) < 1e-12
    with pytest.raises(ValueError):
        build_net_3d("Kagome3D")
    with pytest.raises(ValueError):
        build_net_3d("BCC", "Twist")


def test_rust_equals_python_on_site_lists():
    rs = _rust.get()
    if rs is None or not hasattr(rs, "build_net_3d_json"):
        pytest.skip("papertilling_rs not available (PAPERTILLING_NO_RUST or not built)")
    if len(json.loads(rs.build_net_3d_json("Pyrochlore", "None"))["positions"]) == 40:
        pytest.skip("papertilling_rs predates C1 WP-N1 (old 5-site pyrochlore motif): rebuild the extension")
    for net in NET_MOTIFS:
        for d in DEFECT_LABELS:
            r, p = json.loads(rs.build_net_3d_json(net, d)), build_net_patch(net, d)
            assert r["site_kinds"] == p["site_kinds"], (net, d)
            assert np.allclose(r["positions"], p["positions"], atol=1e-9, rtol=0), (net, d)
            assert [list(b) for b in r["bonds"]] == p["bonds"], (net, d)
            for key in ("name", "space_group_num", "defect_label", "info"):
                assert r[key] == p[key], (net, d, key)
            assert (r["defect_axis"] is None) == (p["defect_axis"] is None)
