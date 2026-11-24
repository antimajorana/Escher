"""Integration tests for Bieberbach cohomology, subperiodic 0D/1D, 80 layer / 230 space groups,
hyperbolic H^2 tilings, and D-symbol orbifold curvature across Rust and pure-Python backends.
"""
from __future__ import annotations

import json
import os
from fractions import Fraction

import papertilling as pt
from papertilling import _rust


EXPECTED_WALLPAPER_17 = [
    ("p1", 1, [], True, "Z^2", "Z^2", [4, 9, 16, 36]),
    ("p2", 2, [], True, "Z_2^3", "Z_2^3", [8, 6, 14, 24]),
    ("pm", 2, [2], True, "Z^2", "Z_2^2 x Z", [8, 9, 20, 36]),
    ("pg", 2, [2], False, "Z^2", "Z_2 x Z", [8, 9, 20, 36]),
    ("cm", 2, [], True, "Z^2", "Z_2 x Z", [5, 9, 14, 27]),
    ("pmm", 4, [2, 2], True, "Z_2^3", "Z_2^4", [16, 9, 25, 36]),
    ("pmg", 4, [2, 2], False, "Z_2^3", "Z_2^3", [10, 9, 19, 30]),
    ("pgg", 4, [2, 2], False, "Z_2^3", "Z_2 x Z_4", [10, 9, 19, 30]),
    ("cmm", 4, [], True, "Z_2^3", "Z_2^3", [10, 9, 16, 24]),
    ("p4", 4, [], True, "Z_2 x Z_4", "Z_2 x Z_4", [10, 6, 13, 18]),
    ("p4m", 8, [2], True, "Z_2 x Z_4", "Z_2^3", [14, 9, 20, 27]),
    ("p4g", 8, [2], False, "Z_2 x Z_4", "Z_2 x Z_4", [11, 9, 17, 24]),
    ("p3", 3, [], True, "Z_3^2", "Z_3^2", [4, 11, 8, 20]),
    ("p3m1", 6, [], True, "Z_3^2", "Z_2", [5, 10, 10, 19]),
    ("p31m", 6, [], True, "Z_3^2", "Z_6", [5, 10, 10, 19]),
    ("p6", 6, [], True, "Z_6", "Z_6", [8, 10, 10, 16]),
    ("p6m", 12, [], True, "Z_6", "Z_2^2", [10, 11, 14, 20]),
]


def test_wallpaper_17_bieberbach_cohomology_and_68_torus_counts():
    for grp, p_ord, h2_t, split, ab_p, ab_f, tor in EXPECTED_WALLPAPER_17:
        s = pt.analyze_wallpaper_group(grp)
        assert s["point_order"] == p_ord
        assert s["h2_torsion"] == h2_t
        assert s["is_split"] is split
        assert s["ab_plus_label"] == ab_p
        assert s["ab_full_label"] == ab_f
        assert [c for _, c in s["torus_counts"]] == tor


def test_subperiodic_rosettes_and_7_friezes():
    r_c5 = pt.build_rosette(dihedral=False, n=5, wedge_delta=1)
    assert r_c5["point_order"] == 5 and r_c5["ab_label"] == "Z_5"
    assert len(r_c5["positions"]) == 1 + 2 * 6

    from papertilling import subperiodic as subp
    assert len(subp.FRIEZE_SYMBOLS) == 7
    nonsym = [sym for sym in subp.FRIEZE_SYMBOLS if not subp.analyze_frieze(sym)["is_split"]]
    assert nonsym == ["p11g", "p2mg"]

    mob = pt.build_frieze_patch("p2mg", cells=8, topology="Mobius", defect="GlideDispiration")
    assert len(mob["pos3d"]) > 10 and len(mob["quads"]) > 0


def test_80_layer_and_230_space_groups_with_3d_nets():
    layers = pt.list_layer_groups()
    assert len(layers) == 80
    lg1 = pt.analyze_layer_group(1)
    assert lg1["point_order"] == 1 and lg1["is_split"] is True

    sg230 = pt.list_space_groups_3d()
    assert len(sg230) == 230
    assert sum(1 for g in sg230 if g["is_split"]) == 73
    assert sum(1 for g in sg230 if not g["is_split"]) == 157

    s123 = pt.analyze_space_group_3d(123)
    s136 = pt.analyze_space_group_3d(136)
    assert s123["is_split"] is True and s123["ab_full_label"] == "Z_2^5"
    assert s136["is_split"] is False and s136["ab_full_label"] == "Z_2^2 x Z_4"

    net136 = pt.build_net_3d("Rutile136", "ScrewDispiration42_136")
    assert len(net136["positions"]) > 0 and len(net136["bonds"]) > 0


def test_layer_mirror_and_glide_keep_transverse_symmetry(monkeypatch):
    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")

    # The transverse mirror gives Z^2 x Z_2. In the glide group its square is
    # an in-plane primitive translation, leaving two free generators and no torsion.
    mirror = pt.analyze_layer_group(4)
    assert mirror["symbol"] == "p11m" and mirror["point_order"] == 2
    assert mirror["is_split"] and mirror["ab_full_label"] == "Z_2 x Z^2"
    assert mirror["h2_torsion"] == [2, 2] and mirror["ab_plus_label"] == "Z^2"

    glide = pt.analyze_layer_group(5)
    assert glide["symbol"] == "p11a" and glide["point_order"] == 2
    assert not glide["is_split"] and glide["ab_full_label"] == "Z^2"
    assert glide["h2_torsion"] == [2, 2] and glide["ab_plus_label"] == "Z^2"


def test_fallback_group_catalogue_entries_are_independent(monkeypatch):
    monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
    for list_groups in (pt.list_layer_groups, pt.list_space_groups_3d):
        groups = list_groups()
        groups[0]["ops"][0]["r"][0][0] = 0
        assert list_groups()[0]["ops"][0]["r"][0][0] == 1


def test_space_group_custom_torus_counts_with_native_backend_available(monkeypatch):
    class DefaultOnlyBackend:
        def analyze_space_group_3d_json(self, number):
            assert number == 1
            return '{"torus_counts": [[2, 8]], "sectors_full": []}'

    monkeypatch.setattr(_rust, "get", DefaultOnlyBackend)
    # P1 has only translations, so all n^3 elements of its finite quotient are separate classes.
    assert pt.analyze_space_group_3d(1)["torus_counts"] == [[2, 8]]
    summary = pt.analyze_space_group_3d(1, (3, 4))
    assert summary["torus_counts"] == [[3, 27], [4, 64]]
    assert summary["sectors_full"] == []


def test_custom_space_group_converts_exact_inputs_for_both_backends(monkeypatch):
    spec = {"dim": 1, "ops": [{"r": [[1]], "t": [Fraction(1, 2)]}], "tb_rows": [[[1, 2]]]}

    monkeypatch.setattr(_rust, "get", lambda: None)
    fallback = pt.analyze_space_group(spec)
    assert fallback["ab_full_label"] == "Z"
    # Pure translations have n distinct conjugacy classes on a rank-one torus of scale n.
    expected_counts = [(2, 2), (3, 3), (4, 4), (6, 6)]
    assert fallback["torus_counts"] == expected_counts

    class Backend:
        def bieberbach_analyze_json(self, payload):
            assert json.loads(payload) == {
                "dim": 1, "number": 0, "symbol": "custom",
                "ops": [{"r": [[1]], "t": [[1, 2]]}], "tb_rows": [[[1, 2]]],
            }
            return '{"ab_full_label": "Z", "torus_counts": [[2, 2]], "sectors_full": []}'

    monkeypatch.setattr(_rust, "get", Backend)
    native = pt.analyze_space_group(spec)
    assert native["ab_full_label"] == "Z"
    assert [tuple(row) for row in native["torus_counts"]] == expected_counts
    assert native["sectors_full"] == []


def test_hyperbolic_tilings_and_dsymbol_orbifold_curvature():
    cube_key = ((4, 3, 0, 0, 0),)
    sq_key = ((4, 4, 0, 0, 0),)
    hyp_key = ((7, 3, 0, 0, 0),)
    assert float(pt.orbifold_curvature(cube_key)) > 0
    assert pt.geometry_kind(cube_key) == "spherical"
    assert pt.orbifold_curvature(sq_key) == 0
    assert pt.geometry_kind(sq_key) == "euclidean"
    assert float(pt.orbifold_curvature(hyp_key)) < 0
    assert pt.geometry_kind(hyp_key) == "hyperbolic"

    t73 = pt.build_hyperbolic_tiling(7, 3, style="RegularPQ", layers=2, wedge_delta=0)
    assert t73["curvature_kappa"] < 0
    assert len(t73["faces"]) >= 8 and len(t73["faces"][0]) == 7
    assert all(p[2] >= 1.0 - 1e-9 for p in t73["hyperboloid_pts"])


def test_voronoi_and_live_n_gt_7_when_rust_enabled():
    rs = _rust.get()
    if os.environ.get("PAPERTILLING_NO_RUST", "") in ("1", "true", "yes") or rs is None:
        return
    import json
    found = json.loads(rs.search_minimal_lattices_json("p4m", 8, 12, 2, 10))
    assert len(found) > 0
    assert all(8 <= e["num_sites"] <= 12 for e in found)
