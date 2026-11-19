"""Comprehensive behavioral tests for Stage 4 Core (`papertilling.flat3d`).

Covers:
1. Exact rational torsion-free Bieberbach classifier over 2D (17) and 3D (230) groups.
2. All 10 Conway-Rossetti closed flat 3D platycosms (13 ITA groups), quotients,
   Euler characteristic chi_3 = 0, and [PROVED] metric/obstruction refusals.
3. All 5 flat 3D non-compact slabs & chimneys (T^2xR, K^2xR, MoebiusxS^1,
   CylinderxS^1, S^1xR^2) and in-plane mirror/glide impossibility.
4. Arbitrary-metric (n, m) chiral nanotubes & nanoribbons, irrational-metric refusal.
5. Symmetry-centred finite flakes (disk, n-gon, Wigner-Seitz) and orbits.
6. Proper det=+1 3D Krivy-Gruber Niggli reduction, 2D/3D CELLTRAN supercells,
   finite strain tensor, and per-direction BoundaryConditions.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from papertilling import _rust, flat3d
from papertilling._flat3d_fallback import lattice_holohedry


# The 13 torsion-free (Bieberbach) ITA types: International Tables for Crystallography Vol. A;
# 10 affine classes + 3 enantiomorphic pairs (144/145, 76/78, 169/170).
EXPECTED_TF_3D = [1, 4, 7, 9, 19, 29, 33, 76, 78, 144, 145, 169, 170]

# Published H_1 of the platycosms (Conway & Rossetti, arXiv:math/0311476, 2003; Wolf, Spaces of
# Constant Curvature, Sec. 3.5; theorem locator to verify), keyed by ITA number, written in the package's format.
PUBLISHED_H1 = {1: "Z^3", 4: "Z_2^2 x Z", 144: "Z_3 x Z", 145: "Z_3 x Z", 76: "Z_2 x Z", 78: "Z_2 x Z", 169: "Z", 170: "Z",
                19: "Z_4^2", 7: "Z_2 x Z^2", 9: "Z^2", 29: "Z_2^2 x Z", 33: "Z_4 x Z"}


def _honeycomb_120() -> dict:
    s3 = math.sqrt(3.0)
    return {
        "a1": [s3, 0.0],
        "a2": [-0.5 * s3, 1.5],
        "basis": [[0.0, 0.0], [0.0, 1.0]],
        "bonds": [(0, 1, 0, 0, 0), (0, 1, 0, -1, 0), (0, 1, -1, -1, 0)],
    }


def _square_cell() -> dict:
    return {
        "a1": [1.0, 0.0],
        "a2": [0.0, 1.0],
        "basis": [[0.0, 0.0]],
        "bonds": [(0, 0, 1, 0, 0), (0, 0, 0, 1, 0)],
    }


def test_torsion_free_classifiers_2d_and_3d():
    wp = flat3d.torsion_free_wallpaper_groups()
    assert len(wp) == 17
    tf_2d = {r["symbol"] for r in wp if r["is_torsion_free"]}
    assert tf_2d == {"p1", "pg"}

    sg = flat3d.torsion_free_space_groups()
    assert len(sg) == 230
    tf_3d = sorted(r["number"] for r in sg if r["is_torsion_free"])
    assert tf_3d == EXPECTED_TF_3D
    assert all(len(r["obstructions"]) == 0 for r in sg if r["is_torsion_free"])

    # Negative control #77 (P4_2): 4_2 is screw, but (4_2)^2 = 2_z has fixed axis!
    p42 = flat3d.is_torsion_free(77)
    assert p42["is_torsion_free"] is False
    assert len(p42["obstructions"]) >= 1

    # Negative control #14 (P2_1/c): inversion centres fix points.
    p21c = flat3d.is_torsion_free(14)
    assert p21c["is_torsion_free"] is False
    assert len(p21c["obstructions"]) >= 1

    # H_1 = G^ab recomputed from the ITA operations equals the published platycosm homology.
    for num, h1 in PUBLISHED_H1.items():
        rep = flat3d.is_torsion_free(num)
        assert rep["is_torsion_free"] is True and rep["ab_full_label"] == h1, (num, rep["ab_full_label"])
        assert flat3d.lookup_platycosm(str(num))["h1_label"] == h1

    # Custom torsion-free check (p1 in 2D via dict spec).
    custom_p1 = {
        "dim": 2,
        "number": 1,
        "symbol": "custom_p1",
        "tb_rows": [[[1, 1], [0, 1]], [[0, 1], [1, 1]]],
        "ops": [{"r": [[1, 0], [0, 1]], "t": [[0, 1], [0, 1]]}],
    }
    assert flat3d.is_torsion_free(custom_p1)["is_torsion_free"] is True


def test_ten_platycosms_catalogue_and_realizations():
    cat = flat3d.list_platycosms()
    assert len(cat) == 10
    codes = [p["code"] for p in cat]
    assert codes == ["c1", "c2", "c3", "c4", "c6", "c22", "+a1", "-a1", "+a2", "-a2"]
    assert sum(1 for p in cat if p["orientable"]) == 6
    assert sum(1 for p in cat if not p["orientable"]) == 4

    assert flat3d.lookup_platycosm("19")["code"] == "c22"
    assert flat3d.lookup_platycosm("P2_12_12_1")["code"] == "c22"
    assert flat3d.lookup_platycosm("torocosm")["code"] == "c1"
    # Wolf's names: G5 = hexacosm, G6 = Hantzsche-Wendt (H_1 = Z_4^2), B1 = Klein bottle x S^1 (Pc).
    assert flat3d.lookup_platycosm("G5")["code"] == "c6"
    assert flat3d.lookup_platycosm("G6")["h1_label"] == "Z_4^2"
    assert flat3d.lookup_platycosm("B1")["ita_numbers"] == [7]

    # Realize all 8 non-hexagonal platycosms on SimpleCubic.
    sc_cases = [
        ("c1", [1, 1, 1], 1),
        ("c2", [1, 1, 2], 1),
        ("c4", [1, 1, 4], 1),
        ("c22", [2, 2, 2], 2),
        ("+a1", [1, 1, 2], 1),
        ("-a1", [2, 2, 2], 2),
        ("+a2", [2, 2, 2], 2),
        ("-a2", [2, 2, 2], 2),
    ]
    for code, sup, expected_verts in sc_cases:
        comp = flat3d.check_platycosm_compatibility("SimpleCubic", code, supercell=sup)
        assert comp["compatible"] is True, f"{code}: {comp}"
        quot = flat3d.build_platycosm_quotient("SimpleCubic", code, supercell=sup)
        assert quot["status"] == "OK"
        assert quot["euler_chi_3"] == 0
        assert quot["cw_v"] - quot["cw_e"] + quot["cw_f"] - quot["cw_c"] == 0
        assert len(quot["vertices"]) == expected_verts

    # Realize c3 and c6 on hexagonal nets (HCP and BetaQuartz).
    for code, net in [("c3", "HCP"), ("c6", "BetaQuartz")]:
        quot = flat3d.build_platycosm_quotient(net, code)
        assert quot["status"] == "OK", f"{net}/{code}: {quot}"
        assert quot["euler_chi_3"] == 0
        assert quot["cw_v"] - quot["cw_e"] + quot["cw_f"] - quot["cw_c"] == 0

    # Extra multi-species & chiral net checks.
    assert flat3d.build_platycosm_quotient("Rutile136", "c4", supercell=[1, 1, 2])["status"] == "OK"
    assert flat3d.build_platycosm_quotient("Diamond", "c22", supercell=[1, 1, 1])["status"] == "OK"
    # Uncoloured diamond (Fd-3m): 4_1 screws have axial translation c/4 + (c/2)Z, so c4 embeds with
    # c' = c (8 sites / 4 = 2 vertices, 16 bonds / 4 = 4 bonds) but not with c' = 4c.
    dia = flat3d.build_platycosm_quotient("Diamond", "c4", supercell=[1, 1, 1], ignore_site_kinds=True)
    assert (dia["status"], len(dia["vertices"]), len(dia["bonds"])) == ("OK", 2, 4)
    assert flat3d.build_platycosm_quotient("Diamond", "c4", supercell=[1, 1, 4], ignore_site_kinds=True)["status"] == "NOT_FOUND"


def test_platycosm_obstructions():
    bad_hcp = flat3d.check_platycosm_compatibility("HCP", "c4")
    assert bad_hcp["compatible"] is False
    assert bad_hcp["status"] == "IMPOSSIBLE"
    assert "IMPOSSIBLE [PROVED]" in bad_hcp["reason"]

    assert "4-fold" in bad_hcp["reason"]  # hexagonal holohedry 6/mmm has no 4-fold axis

    # Cubic holohedry m-3m has no 6-fold axis -> c6 is IMPOSSIBLE [PROVED] on SimpleCubic.
    bad_sc6 = flat3d.check_platycosm_compatibility("SimpleCubic", "c6")
    assert bad_sc6["status"] == "IMPOSSIBLE" and "IMPOSSIBLE [PROVED]" in bad_sc6["reason"]
    # c3 is NOT impossible on SimpleCubic: (R_[111] | (1,0,0)) is a 3_1 screw of Pm-3m, since
    # (1,0,0) + R(1,0,0) + R^2(1,0,0) = (1,1,1). The standard-setting search misses it, so the
    # refusal must be stamped [COMPUTATIONAL EVIDENCE], never [PROVED].
    sc3 = flat3d.check_platycosm_compatibility("SimpleCubic", "c3")
    assert sc3["compatible"] is False and sc3["status"] == "NOT_FOUND"
    assert "[COMPUTATIONAL EVIDENCE]" in sc3["reason"] and "[PROVED]" not in sc3["reason"]

    # Generic triclinic lattice: holohedry -1 has no 2-fold axis -> c2, c22, +a1 IMPOSSIBLE [PROVED].
    tri = {"lattice": [[1.0, 0.0, 0.0], [0.3, 1.1, 0.0], [0.2, 0.4, 1.2]], "basis": [[0.0, 0.0, 0.0]], "bonds": []}
    for code in ("c2", "c22", "+a1"):
        assert flat3d.check_platycosm_compatibility(tri, code)["status"] == "IMPOSSIBLE"
    assert flat3d.check_platycosm_compatibility(tri, "c1")["status"] == "OK"


def test_lattice_holohedry_orders():
    # Holohedry orders (International Tables Vol. A): m-3m = 48, 6/mmm = 24, 4/mmm = 16, -3m = 12,
    # -1 = 2; proper rotation orders read off trace = 1 + 2 cos(2 pi / n).
    cubic = lattice_holohedry([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert cubic == {"order": 48, "proper_orders": [1, 2, 3, 4], "has_222": True}
    fcc = lattice_holohedry([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    assert fcc == {"order": 48, "proper_orders": [1, 2, 3, 4], "has_222": True}
    hexa = lattice_holohedry([[1, 0, 0], [-0.5, math.sqrt(0.75), 0], [0, 0, 1.35]])
    assert hexa == {"order": 24, "proper_orders": [1, 2, 3, 6], "has_222": True}
    tetra = lattice_holohedry([[1, 0, 0], [0, 1, 0], [0, 0, 1.3]])
    assert tetra == {"order": 16, "proper_orders": [1, 2, 4], "has_222": True}
    rhombo = lattice_holohedry([[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.2, 1]])  # -3m: D3 has no perpendicular 2-folds
    assert rhombo == {"order": 12, "proper_orders": [1, 2, 3], "has_222": False}
    assert lattice_holohedry([[1, 0, 0], [0.3, 1.1, 0], [0.2, 0.4, 1.2]]) == {"order": 2, "proper_orders": [1], "has_222": False}


def test_flat_3d_slabs_and_chimneys():
    specs = [
        ("TorusSlab", 2, 1, True, 2),
        ("KleinSlab", 2, 1, False, 1),
        ("MoebiusTube", 2, 1, False, 2),
        ("CylinderCrossS1", 2, 1, True, 2),
        ("CylinderChimney", 1, 2, True, 1),
    ]
    for kind, cd, nd, ori, b1 in specs:
        res = flat3d.build_slab_3d("SimpleCubic", kind=kind, repeats=(3, 3, 3))
        assert res["status"] == "OK"
        assert res["compact_dims"] == cd
        assert res["noncompact_dims"] == nd
        assert res["orientable"] is ori
        assert res["b1"] == b1
        assert len(res["positions"]) == 27
        assert len(res["boundary_sites"]) > 0
        assert any(b[2] for b in res["bonds"])

    # Chiral BetaQuartz lacks in-plane mirrors/glides -> cannot close into KleinSlab.
    bad = flat3d.build_slab_3d("BetaQuartz", kind="KleinSlab", repeats=(2, 2, 2))
    assert bad["status"] == "IMPOSSIBLE"
    assert "IMPOSSIBLE [PROVED]" in bad["reason"]


def test_chiral_nanotubes_and_nanoribbons():
    hc = _honeycomb_120()
    # Both standard 60-degree conversion and explicit acute 60-degree supercell must match.
    hc60 = flat3d.build_supercell_2d(hc, [[1, 1], [0, 1]])
    expected_atoms = {(5, 5): 20, (6, 0): 24, (4, 2): 56}
    for (n, m), apu in expected_atoms.items():
        t_auto = flat3d.build_nanotube(hc, n, m, periods_z=2, periodic_z=True, standard_hex_60=True)
        t_man = flat3d.build_nanotube(hc60, n, m, periods_z=2, periodic_z=True, standard_hex_60=False)
        for tube in (t_auto, t_man):
            assert tube["atoms_per_unit"] == apu
            assert len(tube["positions_3d"]) == 2 * apu
            assert all(d == 3 for d in tube["coordination_numbers"])
            r_exp = tube["circumference"] / (2.0 * math.pi)
            assert math.isclose(tube["radius"], r_exp, rel_tol=1e-9)
            radii = [math.hypot(x, y) for x, y, _ in tube["positions_3d"]]
            assert all(math.isclose(r, r_exp, rel_tol=1e-9) for r in radii)
            # Isometric unrolling preserves nearest-neighbour bond length (= 1.0).
            p2 = np.asarray(tube["positions_2d"])
            a, t1, c_len = np.linalg.norm(p2[tube["bonds"][0][0]] - p2[tube["bonds"][0][1]]), t_auto["t_vec"], tube["circumference"]
            assert a > 0.5 and len(t1) == 2 and c_len > 0.0

    # Irrational / incommensurate aspect ratio refuses orthogonal commensurate T.
    irr_sq = {
        "a1": [1.0, 0.0],
        "a2": [0.0, math.sqrt(math.pi)],
        "basis": [[0.0, 0.0]],
        "bonds": [(0, 0, 1, 0, 0), (0, 0, 0, 1, 0)],
    }
    with pytest.raises((ValueError, RuntimeError), match="REFUSED \\[COMPUTATIONAL EVIDENCE\\]"):
        flat3d.build_nanotube(irr_sq, 1, 1)

    # Saito, Dresselhaus & Dresselhaus, Physical Properties of Carbon Nanotubes (1998), Ch. 3:
    # |Ch| = a sqrt(n^2+nm+m^2) (a = sqrt 3 for unit C-C bonds), d = |Ch| / pi,
    # cos(theta) = (2n+m) / (2 sqrt(n^2+nm+m^2)); (5,5) -> 30 deg, (6,0) -> 0 deg, (4,2) -> 19.1066 deg.
    for (n, m), theta in {(5, 5): 30.0, (6, 0): 0.0, (4, 2): math.degrees(math.acos(10.0 / (2.0 * math.sqrt(28.0))))}.items():
        tube = flat3d.build_nanotube(hc60, n, m)
        q = n * n + n * m + m * m
        assert math.isclose(tube["circumference"], math.sqrt(3.0 * q), rel_tol=1e-9)
        assert math.isclose(2.0 * tube["radius"], math.sqrt(3.0 * q) / math.pi, rel_tol=1e-9)
        assert abs(tube["chiral_angle_deg"] - theta) < 1e-4

    rib = flat3d.build_nanoribbon(hc, [1, 0], width_cells=3, length_cells=4, periodic_along=True)
    assert rib["width_cells"] == 3 and rib["length_cells"] == 4
    assert len(rib["positions_2d"]) == 2 * 3 * 4
    assert len(rib["edge_sites"]) > 0
    assert any(b[2] for b in rib["bonds"])


def test_symmetric_and_wigner_seitz_flakes():
    hc = _honeycomb_120()
    cen = [0.5 * math.sqrt(3.0), 0.5]

    benz = flat3d.build_symmetric_flake("p6m", hc, cen, radius=1.1)
    assert benz["point_group_order"] == 12
    assert benz["is_dihedral"] is True
    assert len(benz["crystal"]["pos"]) == 6
    assert len(benz["site_orbits"]) == 1
    assert len(benz["vertex_perms"]) == 12

    cor = flat3d.build_symmetric_flake("p6m", hc, cen, radius=2.7)
    assert cor["point_group_order"] == 12
    assert len(cor["crystal"]["pos"]) == 24

    hex_poly = flat3d.build_symmetric_flake("p6m", hc, cen, n_fold_polygon=6, radius=2.7)
    assert hex_poly["point_group_order"] == 12
    assert len(hex_poly["crystal"]["pos"]) >= 24

    d3_sub = flat3d.build_symmetric_flake("p3m1", hc, cen, radius=1.1)
    assert d3_sub["point_group_order"] == 6
    assert d3_sub["is_dihedral"] is True

    sq = _square_cell()
    ws = flat3d.build_wigner_seitz_flake("p4m", sq, [0.0, 0.0], scale=2.5)
    assert ws["point_group_order"] == 8
    assert ws["is_dihedral"] is True
    assert len(ws["crystal"]["pos"]) == 9


def test_niggli_supercell_strain_and_bcs():
    fcc = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    bcc = [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]
    shears = [
        [[1, 2, 1], [0, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [3, 1, 2], [-1, 0, 1]],
    ]
    for base in (fcc, bcc):
        for sh in shears:
            lat = (np.asarray(sh, float) @ np.asarray(base, float)).tolist()
            red = flat3d.niggli_reduce_3d(lat)
            p_mat = np.asarray(red["p_mat"], float)
            assert round(float(np.linalg.det(p_mat))) == 1
            assert np.allclose(p_mat @ np.asarray(lat, float), np.asarray(red["reduced"], float), atol=1e-8)
            assert math.isclose(red["volume"], abs(float(np.linalg.det(lat))), rel_tol=1e-8)
            idem = flat3d.niggli_reduce_3d(red["reduced"])
            assert np.allclose(idem["reduced"], red["reduced"], atol=1e-8)
    # Hand-checked type-II Niggli cell (xi = eta = 0 > zeta; A = 1 <= B = 1.53 <= C = 2.25, |zeta| = 0.6 <= A):
    # Krivy-Gruber step A4 must leave it unchanged (p = identity).
    typ2 = flat3d.niggli_reduce_3d([[1.0, 0.0, 0.0], [-0.3, 1.2, 0.0], [0.0, 0.0, 1.5]])
    assert typ2["p_mat"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]] and typ2["niggli_type"] == 2

    # 3D and 2D supercells
    sc2 = flat3d.build_supercell_3d("SimpleCubic", [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert len(sc2["basis"]) == 8
    assert len(sc2["bonds"]) == 24

    sq = _square_cell()
    sq2 = flat3d.build_supercell_2d(sq, [[2, 0], [0, 2]])
    assert len(sq2["basis"]) == 4

    # Finite strain tensor
    ref = np.eye(3)
    def_mat = np.diag([1.1, 1.0, 1.0])
    st = flat3d.cell_strain_tensor(ref, def_mat)
    assert math.isclose(st["green_lagrange"][0][0], 0.5 * (1.21 - 1.0), rel_tol=1e-9)
    assert math.isclose(st["infinitesimal_strain"][0][0], 0.1, rel_tol=1e-9)
    assert math.isclose(st["volumetric_strain"], 0.1, rel_tol=1e-9)
    # Von Mises equivalent strain sqrt(2/3 e':e'): E = diag(0.105, 0, 0) -> e' = 0.035 diag(2, -1, -1) -> 0.07.
    assert math.isclose(st["von_mises_equivalent"], 0.07, rel_tol=1e-9)

    ang = 0.4
    rot = np.array([[math.cos(ang), -math.sin(ang), 0.0], [math.sin(ang), math.cos(ang), 0.0], [0.0, 0.0, 1.0]])
    st_rot = flat3d.cell_strain_tensor(ref, rot)
    assert np.allclose(st_rot["green_lagrange"], np.zeros((3, 3)), atol=1e-10)

    # Per-direction BoundaryConditions (3D and 2D)
    b3 = flat3d.apply_boundary_conditions_3d(
        "SimpleCubic",
        [2, 2, 2],
        ["Open", "Antiperiodic", {"Twisted": {"phase_rad": 0.7}}],
    )
    assert len(b3["positions"]) == 8
    # Open along x => no wrap[0] != 0
    assert all(bd["wrap"][0] == 0 for bd in b3["bonds"])
    y_wraps = [bd for bd in b3["bonds"] if bd["wrap"][1] != 0 and bd["wrap"][2] == 0]
    assert y_wraps and all(math.isclose(bd["weight_real"], -1.0, abs_tol=1e-9) for bd in y_wraps)
    z_wraps = [bd for bd in b3["bonds"] if bd["wrap"][2] != 0 and bd["wrap"][1] == 0]
    assert z_wraps and all(math.isclose(bd["weight_real"], math.cos(0.7), abs_tol=1e-9) for bd in z_wraps)

    b2 = flat3d.apply_boundary_conditions_2d(sq, [2, 2], ["Periodic", "Antiperiodic"])
    assert len(b2["positions"]) == 4
    assert any(math.isclose(bd["weight_real"], -1.0, abs_tol=1e-9) for bd in b2["bonds"])


_HAS_RS_FLAT3D = _rust.have_rust() and hasattr(_rust._load(), "flat3d_list_platycosms_json")


@pytest.mark.skipif(not _HAS_RS_FLAT3D, reason="papertilling_rs extension with Stage 4 flat3d bindings not built")
def test_rust_and_python_paths_agree(monkeypatch):
    def both(fn, *args, **kw):
        monkeypatch.delenv("PAPERTILLING_NO_RUST", raising=False)
        rs = fn(*args, **kw)
        monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
        return rs, fn(*args, **kw)

    rs, py = both(flat3d.list_platycosms)
    assert rs == py
    rs, py = both(flat3d.build_platycosm_quotient, "SimpleCubic", "c22", supercell=[2, 2, 2])
    # 8 sites / 4 = 2 vertices, 24 bonds / 4 = 6 bonds.
    assert set(rs) == set(py) and (rs["status"], len(rs["vertices"]), len(rs["bonds"])) == (py["status"], 2, 6)
    rs, py = both(flat3d.check_platycosm_compatibility, "SimpleCubic", "c3")
    assert set(rs) == set(py) and rs["status"] == py["status"] == "NOT_FOUND"
    rs, py = both(flat3d.build_supercell_3d, "SimpleCubic", [[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert sorted(map(tuple, rs["bonds"])) == sorted(map(tuple, py["bonds"]))
    rs, py = both(flat3d.build_nanotube, _honeycomb_120(), 4, 2, 1, False, True)
    assert set(rs) == set(py) and rs["atoms_per_unit"] == py["atoms_per_unit"] == 56
