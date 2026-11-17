"""Quantitative Ariza-Ortiz discrete gauge elasticity verification gates (Layers 1 & 2)."""

from __future__ import annotations

import copy
import time
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize

import papertilling as pt
from papertilling.defects import (
    DefectCatalogue,
    acoustic_tensor,
    barnett_lothe_tensor,
    cauchy_born_tensor,
    dipole_prelog_coeff,
    eval_gsv_energy,
    fusion_channels,
    gsv_hessian_2d,
    gsv_pairs,
    plaquette_holonomies,
    relax_elastic_2d,
    ribbon_faces,
    space_group_extension,
    twisted_coboundary,
    wyckoff_rotocentres,
)
from papertilling.defects._surgery import _Builder
from papertilling.defects.build import build, lattice_data
from papertilling.defects.gsv_elastic import gsv_residual_and_jacobian


def _ball_nodes(d, seeds, rad):
    adj = [[] for _ in range(len(d.sites))]
    for u, v in d.edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = set(seeds)
    q = deque((s, 0) for s in seeds)
    while q:
        u, dist = q.popleft()
        if dist >= rad:
            continue
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append((v, dist + 1))
    return sorted(seen)


def _crop_defect(d, keep_nodes):
    keep = set(keep_nodes)
    old_to_new = {u: i for i, u in enumerate(sorted(keep))}
    sites = [d.sites[u] for u in sorted(keep)]
    positions = np.array([d.positions[u] for u in sorted(keep)], dtype=float)
    edge = {
        (old_to_new[u], old_to_new[v]): copy.deepcopy(attr)
        for (u, v), attr in d.edge.items()
        if u in keep and v in keep
    }
    info = copy.deepcopy(d.info)
    if "cone_sites" in info and info["cone_sites"] is not None:
        info["cone_sites"] = [old_to_new[u] for u in info["cone_sites"] if u in keep]
    return type(d)(d.kind, sites, positions, edge, info=info)


def _kabsch_so2_max_diff(X1, X2_mapped):
    A = X1 - X1.mean(axis=0)
    B = X2_mapped - X2_mapped.mean(axis=0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return float(np.max(np.linalg.norm(A - B @ R, axis=1)))


def _best_cut_iso_diff(d1, d2):
    gm = nx.algorithms.isomorphism.GraphMatcher(d1.to_networkx(), d2.to_networkx())
    n = len(d1.sites)
    best_pos = np.inf
    best_iso = None
    for iso in gm.isomorphisms_iter():
        p2 = np.array([d2.positions[iso[u]] for u in range(n)])
        diff = _kabsch_so2_max_diff(d1.positions, p2)
        if diff < best_pos:
            best_pos = diff
            best_iso = iso
        if best_pos < 1e-10:
            break
    return best_pos, best_iso


def _fd_hessian(pos, pairs, eps=1e-6):
    x0 = pos.ravel().astype(float)
    dim = len(x0)
    r0, J0, _, _ = gsv_residual_and_jacobian(pos, pairs)
    g0 = np.asarray(J0.T @ r0, dtype=float)
    g_fd = np.zeros(dim, dtype=float)
    H = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        dx = np.zeros(dim, dtype=float)
        dx[i] = eps
        xp = (x0 + dx).reshape(pos.shape)
        xm = (x0 - dx).reshape(pos.shape)
        g_fd[i] = (eval_gsv_energy(xp, pairs) - eval_gsv_energy(xm, pairs)) / (2.0 * eps)
        rp, Jp, _, _ = gsv_residual_and_jacobian(xp, pairs)
        rm, Jm, _, _ = gsv_residual_and_jacobian(xm, pairs)
        H[:, i] = np.asarray((Jp.T @ rp) - (Jm.T @ rm), dtype=float) / (2.0 * eps)
    assert np.max(np.abs(g0 - g_fd)) < 1e-9
    return 0.5 * (H + H.T)


def test_public_exports_and_plot_elastic():
    t0 = time.perf_counter()
    for name in [
        "space_group_extension",
        "wyckoff_rotocentres",
        "fusion_channels",
        "ribbon_faces",
        "plaquette_holonomies",
        "twisted_coboundary",
        "relax_elastic_2d",
        "eval_gsv_energy",
        "gsv_hessian_2d",
        "gsv_pairs",
        "build",
    ]:
        assert hasattr(pt, name), f"missing top-level export {name}"

    c = build("square", dict(kind="dislocation", b=(1, 0), at=(0, 0)), R=6.0, picture="elastic")
    assert c.defect.info["picture"] == "elastic"
    gsv = c.defect.info["gsv"]
    for key in [
        "total_energy",
        "stretch_energy",
        "shear_energy",
        "bond_strain",
        "bond_gsv_energy",
        "site_gsv_energy",
        "site_shear",
        "hessian_min_eigs",
        "pairs",
    ]:
        assert key in gsv
    assert abs(gsv["total_energy"] - (gsv["stretch_energy"] + gsv["shear_energy"])) < 1e-12

    fig, axes = plt.subplots(1, 2)
    c.plot(ax=axes[0], mode="strain")
    c.plot(ax=axes[1], mode="shear")
    plt.close(fig)
    print(f"test_public_exports_and_plot_elastic # time: {time.perf_counter() - t0:.2f}s")


def test_gate1_square_disclination_cone_and_3d_cube():
    t0 = time.perf_counter()
    sq = lattice_data("square")
    B = _Builder(sq.cell, sq.L, np.zeros(2), 1.5)
    B._disclination(90.0, np.zeros(2), 0.123)
    d7 = B.finish()
    assert len(d7.sites) == 7
    assert len(d7.edges) == 9
    assert len(d7.info["cone_sites"]) == 1

    relax_elastic_2d(d7)
    gsv = d7.info["gsv"]
    assert d7.info["picture"] == "elastic"
    assert gsv["grad_inf"] < 1e-11
    assert gsv["total_energy"] > 0.05

    pairs = gsv["pairs"]
    H = gsv_hessian_2d(d7.positions, pairs)
    assert H.shape == (14, 14)
    H_fd = _fd_hessian(d7.positions, pairs)
    assert np.max(np.abs(H - H_fd)) < 1e-8

    eigs = np.linalg.eigvalsh(H)
    assert np.max(np.abs(eigs[:3])) < 1e-12
    assert eigs[3] > 0.1
    assert np.all(eigs[3:] > 0.0)

    c_cone = build("square", dict(kind="disclination", angle=90, at=(0, 0)), R=4.0, picture="elastic")
    assert c_cone.defect.info["gsv"]["grad_inf"] < 1e-11
    min_eigs = c_cone.defect.info["gsv"]["hessian_min_eigs"]
    assert np.max(np.abs(min_eigs[:3])) < 1e-10
    assert min_eigs[3] > 0.0

    # 3D cube corner isometric relaxation via face-corner GSV pairs
    d3 = B.finish()
    pairs3 = gsv_pairs(d3, face_corners_only=True)
    x0 = np.column_stack([d7.positions, np.zeros(7)])
    cs = d3.info["cone_sites"][0]
    x0[cs, 2] += 0.6
    res = minimize(lambda x: eval_gsv_energy(x.reshape(-1, 3), pairs3), x0.ravel(), method="BFGS", options={"gtol": 1e-11})
    assert res.fun < 1e-12
    X3 = res.x.reshape(-1, 3)
    for u, v in d3.edges:
        assert abs(np.linalg.norm(X3[v] - X3[u]) - 1.0) < 1e-5
    nbrs = [v for u, v in d3.edges if u == cs] + [u for u, v in d3.edges if v == cs]
    vecs = [X3[v] - X3[cs] for v in nbrs]
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            assert abs(float(np.dot(vecs[i], vecs[j]))) < 1e-5
    print(f"test_gate1_square_disclination_cone_and_3d_cube # time: {time.perf_counter() - t0:.2f}s")


def test_gate2_volterra_cut_routing_gauge_invariance():
    t0 = time.perf_counter()

    # 1. Square (1, 0) dislocation under rotated cut directions
    c1 = build("square", dict(kind="dislocation", b=(1, 0), at=(0, 0), cut=[1.0, 0.21]), R=15.0, picture="raw")
    c2 = build("square", dict(kind="dislocation", b=(1, 0), at=(0, 0), cut=[-0.43, 0.91]), R=15.0, picture="raw")
    d1 = _crop_defect(c1.defect, _ball_nodes(c1.defect, [v for c in c1.cores() for v in c["sites"]], 4))
    d2 = _crop_defect(c2.defect, _ball_nodes(c2.defect, [v for c in c2.cores() for v in c["sites"]], 4))
    relax_elastic_2d(d1)
    relax_elastic_2d(d2)
    pos_diff, iso = _best_cut_iso_diff(d1, d2)
    assert pos_diff < 1e-8
    se1 = d1.info["gsv"]["site_gsv_energy"]
    se2 = np.array([d2.info["gsv"]["site_gsv_energy"][iso[u]] for u in range(len(d1.sites))])
    assert np.max(np.abs(se1 - se2)) < 1e-10
    assert np.max(np.abs(np.sort(d1.info["gsv"]["bond_gsv_energy"]) - np.sort(d2.info["gsv"]["bond_gsv_energy"]))) < 1e-10

    # 2. +90/-90 disclination dipole under distinct cut vectors
    cd1 = build(
        "square",
        dict(kind="disclination", angle=90, at=(-2.0, 0.0), cut=[-1.0, 0.21]),
        dict(kind="disclination", angle=-90, at=(3.0, 0.0), cut=[1.0, 0.21]),
        R=15.0,
        picture="raw",
    )
    cd2 = build(
        "square",
        dict(kind="disclination", angle=90, at=(-2.0, 0.0), cut=[-0.21, -1.0]),
        dict(kind="disclination", angle=-90, at=(3.0, 0.0), cut=[0.21, 1.0]),
        R=15.0,
        picture="raw",
    )
    dd1 = _crop_defect(cd1.defect, _ball_nodes(cd1.defect, [v for c in cd1.cores() for v in c["sites"]], 4))
    dd2 = _crop_defect(cd2.defect, _ball_nodes(cd2.defect, [v for c in cd2.cores() for v in c["sites"]], 4))
    relax_elastic_2d(dd1)
    relax_elastic_2d(dd2)
    pos_diff_d, iso_d = _best_cut_iso_diff(dd1, dd2)
    assert pos_diff_d < 1e-8
    se_d2 = np.array([dd2.info["gsv"]["site_gsv_energy"][iso_d[u]] for u in range(len(dd1.sites))])
    assert np.max(np.abs(dd1.info["gsv"]["site_gsv_energy"] - se_d2)) < 1e-10

    # 3. Full-disk p4g (shastry_sutherland) disclinations: elastic is gauge-invariant; anchored relaxed breaks it
    ce1 = build("shastry_sutherland", dict(kind="disclination", angle=90, at=(0, 0), cut=[1.0, 0.21]), snap=True, R=8.0, picture="elastic")
    ce2 = build("shastry_sutherland", dict(kind="disclination", angle=90, at=(0, 0), cut=[-0.43, 0.91]), snap=True, R=8.0, picture="elastic")
    pos_diff_p4g, iso_p4g = _best_cut_iso_diff(ce1.defect, ce2.defect)
    assert pos_diff_p4g < 1e-8
    se_p2 = np.array([ce2.defect.info["gsv"]["site_gsv_energy"][iso_p4g[u]] for u in range(len(ce1.defect.sites))])
    assert np.max(np.abs(ce1.defect.info["gsv"]["site_gsv_energy"] - se_p2)) < 1e-10

    cr1 = build("shastry_sutherland", dict(kind="disclination", angle=90, at=(0, 0), cut=[1.0, 0.21]), snap=True, R=8.0, picture="relaxed")
    cr2 = build("shastry_sutherland", dict(kind="disclination", angle=90, at=(0, 0), cut=[-0.43, 0.91]), snap=True, R=8.0, picture="relaxed")
    diff_anchored, _ = _best_cut_iso_diff(cr1.defect, cr2.defect)
    assert diff_anchored > 1e-3
    print(f"test_gate2_volterra_cut_routing_gauge_invariance # time: {time.perf_counter() - t0:.2f}s")


def test_gate3_ariza_ortiz_linear_elasticity_and_scaling():
    t0 = time.perf_counter()

    # Pristine Hessians & acoustic tensors on square & honeycomb
    for lat in ["square", "honeycomb"]:
        ld = lattice_data(lat)
        d0 = _Builder(ld.cell, ld.L, np.zeros(2), 4.0).finish()
        pairs0 = gsv_pairs(d0)
        assert eval_gsv_energy(d0.positions, pairs0) < 1e-26
        H = gsv_hessian_2d(d0.positions, pairs0)
        H_fd = _fd_hessian(d0.positions, pairs0)
        assert np.max(np.abs(H - H_fd)) < 1e-9
        eigs = np.linalg.eigvalsh(H)
        assert np.max(np.abs(eigs[:3])) < 1e-11
        assert eigs[3] > 0.0
        for th in np.linspace(0.0, np.pi, 13, endpoint=False):
            G = acoustic_tensor(lat, th)
            assert np.allclose(G, G.T, atol=1e-12)
            assert np.min(np.linalg.eigvalsh(G)) > 0.3

    # Exact analytical Cauchy-Born moduli & prelogarithmic dipole coefficient
    C_sq = cauchy_born_tensor("square")
    assert abs(C_sq[0, 0, 0, 0] - 2.0) < 1e-12
    assert abs(C_sq[1, 1, 1, 1] - 2.0) < 1e-12
    assert abs(C_sq[0, 0, 1, 1]) < 1e-12
    assert abs(C_sq[0, 1, 0, 1] - 1.0) < 1e-12
    K_sq_th = 1.0 / (2.0 * np.pi)
    assert abs(dipole_prelog_coeff("square", (1.0, 0.0)) - K_sq_th) < 1e-11
    B_sq = barnett_lothe_tensor("square")
    assert np.allclose(B_sq, np.eye(2), atol=1e-11)

    A_hc = space_group_extension("honeycomb")["A"]
    b_hc = A_hc @ np.array([1.0, 0.0])
    K_hc_th = 3.0 * np.sqrt(3.0) / (8.0 * np.pi)
    assert abs(dipole_prelog_coeff("honeycomb", b_hc) - K_hc_th) < 1e-10

    # Square dislocation dipole energy vs ln(sep)
    seps_sq, E_dip_sq = [], []
    for k in [1.5, 2.0, 3.0, 4.0]:
        c = build(
            "square",
            dict(kind="dislocation", b=(1, 0), at=(-k, 0.0), cut=[0.01, 1.0]),
            dict(kind="dislocation", b=(-1, 0), at=(k, 0.0), cut=[0.01, 1.0]),
            R=22.0,
            picture="elastic",
        )
        pos = [np.mean(c.defect.positions[cr["sites"]], axis=0) for cr in c.cores()]
        seps_sq.append(float(np.linalg.norm(pos[1] - pos[0])))
        E_dip_sq.append(c.defect.info["gsv"]["total_energy"])
    assert np.all(np.diff(E_dip_sq) > 0.0)
    slope_dip, _ = np.polyfit(np.log(seps_sq), E_dip_sq, 1)
    assert abs(slope_dip - K_sq_th) / K_sq_th < 0.15

    # Single dislocation E(R) ~ 0.5 * K_th * ln(R) on square & honeycomb
    Rs_sq = [6.0, 9.0, 12.0, 15.0]
    E_s_sq = [build("square", dict(kind="dislocation", b=(1, 0), at=(0, 0)), R=R, picture="elastic").defect.info["gsv"]["total_energy"] for R in Rs_sq]
    s_sq, _ = np.polyfit(np.log(Rs_sq), E_s_sq, 1)
    assert abs(s_sq - 0.5 * K_sq_th) / (0.5 * K_sq_th) < 0.10

    Rs_hc = [7.0, 10.0, 14.0, 18.0]
    E_s_hc = [build("honeycomb", dict(kind="dislocation", b=b_hc, at=(0, 0), cut=[0.01, 1.0]), R=R, picture="elastic").defect.info["gsv"]["total_energy"] for R in Rs_hc]
    s_hc, _ = np.polyfit(np.log(Rs_hc), E_s_hc, 1)
    assert abs(s_hc - 0.5 * K_hc_th) / (0.5 * K_hc_th) < 0.10
    print(f"test_gate3_ariza_ortiz_linear_elasticity_and_scaling # time: {time.perf_counter() - t0:.2f}s")


def test_gate4_nonsymmorphic_p4m_vs_p4g_and_fusion():
    t0 = time.perf_counter()

    # Symmorphic p4 / p4m
    for name, grp in [("chiral_square", "p4"), ("square", "p4m")]:
        ext = space_group_extension(name)
        assert ext["group"] == grp
        assert ext["nonsymmorphic"] is False
        assert ext["cocycle_identity_max_err"] < 1e-12
        wy = wyckoff_rotocentres(name)
        assert "1a" in wy["by_label"] and "1b" in wy["by_label"]
        assert np.allclose(wy["by_label"]["1a"]["frac"], [0.0, 0.0], atol=1e-12)
        assert np.allclose(wy["by_label"]["1b"]["frac"], [0.5, 0.5], atol=1e-12)

    ext_sq = space_group_extension("square")
    assert len(ext_sq["glides"]) == 0 and len(ext_sq["mirrors"]) == 4
    for m in ext_sq["mirrors"]:
        t2 = m["tau_frac"] + m["R_frac"] @ m["tau_frac"]
        assert np.allclose(t2, 0.0, atol=1e-12)

    # Nonsymmorphic wallpaper groups & p4g algebraic relations
    for name, grp in [("brick_pg", "pg"), ("zigzag", "pmg"), ("sutherlands_brick", "pgg"), ("shastry_sutherland", "p4g")]:
        ext = space_group_extension(name)
        assert ext["group"] == grp
        assert ext["nonsymmorphic"] is True
        assert ext["cocycle_identity_max_err"] < 1e-12

    ext_ss = space_group_extension("shastry_sutherland")
    wy_ss = wyckoff_rotocentres("shastry_sutherland")
    assert len(ext_ss["glides"]) == 2 and len(ext_ss["mirrors"]) == 2
    assert "2a(i)" in wy_ss["by_label"] and "2a(ii)" in wy_ss["by_label"]
    assert wy_ss["by_label"]["2a(i)"]["splits_under_Gplus"] is True
    assert wy_ss["by_label"]["2a(i)"]["mirror_partner"] == "2a(ii)"

    gx = ext_ss["glides"][0]
    gy = ext_ss["glides"][1]
    assert np.allclose(gx["tau_frac"] + gx["R_frac"] @ gx["tau_frac"], [1.0, 0.0], atol=1e-12)
    assert np.allclose(gy["tau_frac"] + gy["R_frac"] @ gy["tau_frac"], [0.0, 1.0], atol=1e-12)
    C4 = np.array([[0.0, -1.0], [1.0, 0.0]])
    c_i = wy_ss["by_label"]["2a(i)"]["frac"]
    c_ii = wy_ss["by_label"]["2a(ii)"]["frac"]
    t_rot_i = (np.eye(2) - C4) @ c_i
    t_rot_ii_inv = (np.eye(2) - C4.T) @ c_ii
    Rg = gx["R_frac"]
    tg = gx["tau_frac"]
    t_conj = tg + Rg @ t_rot_i - (Rg @ C4 @ Rg.T) @ tg
    assert np.allclose(Rg @ C4 @ Rg.T, C4.T, atol=1e-12)
    assert np.allclose(t_conj - t_rot_ii_inv, [1.0, -1.0], atol=1e-12)

    # Basepoint-free conjugate fusion channels on square & shastry_sutherland
    f_1a1a = fusion_channels("square", [dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0, 0))])
    f_1b1a = fusion_channels("square", [dict(kind="disclination", angle=90, at=(0.5, 0.5)), dict(kind="disclination", angle=-90, at=(0, 0))])
    assert f_1a1a["contains_vacuum"] is True
    for ch in f_1a1a["channels"]:
        bf = np.rint(ch["b_frac"]).astype(int)
        assert (bf[0] + bf[1]) % 2 == 0
    assert f_1b1a["contains_vacuum"] is False
    min_1b1a = [ch for ch in f_1b1a["minimal_channels"] if ch["kind"] == "dislocation"]
    assert any(np.isclose(ch["burgers_norm"], 1.0) for ch in min_1b1a)

    f_2a_same = fusion_channels("shastry_sutherland", [dict(kind="disclination", angle=90, at=(0.5, 1.5)), dict(kind="disclination", angle=-90, at=(0.5, 1.5))])
    f_2a_opp = fusion_channels("shastry_sutherland", [dict(kind="disclination", angle=90, at=(0.5, 1.5)), dict(kind="disclination", angle=-90, at=(1.5, 0.5))])
    assert f_2a_same["contains_vacuum"] is True
    assert f_2a_opp["contains_vacuum"] is False
    assert np.isclose(f_2a_opp["minimal_channels"][0]["burgers_norm"], 2.0)

    # Elastic GSV energy ordering of fusion Burgers channels
    e10 = build("square", dict(kind="dislocation", b=(1, 0), at=(0, 0)), R=8.0, picture="elastic").defect.info["gsv"]["total_energy"]
    e11 = build("square", dict(kind="dislocation", b=(1, 1), at=(0, 0)), R=8.0, picture="elastic").defect.info["gsv"]["total_energy"]
    e20 = build("square", dict(kind="dislocation", b=(2, 0), at=(0, 0)), R=8.0, picture="elastic").defect.info["gsv"]["total_energy"]
    assert 0.0 < e10 < e11 < e20
    print(f"test_gate4_nonsymmorphic_p4m_vs_p4g_and_fusion # time: {time.perf_counter() - t0:.2f}s")


def test_gate5_wc1_mobius_ribbon_faces_and_bianchi():
    t0 = time.perf_counter()
    for lat, expected_faces, expected_ring, expected_seam in [("square", 8, 4, 2), ("brick_pg", 50, 6, 5)]:
        d_mob = DefectCatalogue(lat).mobius_seams[0].build()
        faces = ribbon_faces(d_mob)
        assert len(faces) == expected_faces
        assert all(len(f.verts) == expected_ring for f in faces)

        hol = plaquette_holonomies(d_mob)
        assert len(hol) == expected_faces
        assert sum(1 for h in hol if h["crosses_seam"]) == expected_seam
        assert all(h["is_flat"] for h in hol)
        assert all(not h["has_curvature"] for h in hol)

        rng = np.random.default_rng(7)
        m = rng.normal(size=(len(d_mob.sites), 2))
        d2m = twisted_coboundary(d_mob, twisted_coboundary(d_mob, m))
        assert max(np.linalg.norm(v) for v in d2m) < 1e-12

    # Discrete Bianchi identity d_R^2 = 0 holds on flat plaquettes, violated on curved plaquettes
    c_cone = build("square", dict(kind="disclination", angle=90, at=(0, 0)), R=4.0, picture="raw")
    hol_c = plaquette_holonomies(c_cone.defect)
    m_c = np.ones((len(c_cone.defect.sites), 2))
    d2_c = twisted_coboundary(c_cone.defect, twisted_coboundary(c_cone.defect, m_c))
    flat_vals = [np.linalg.norm(d2_c[h["face_index"]]) for h in hol_c if h["is_flat"]]
    curved_vals = [np.linalg.norm(d2_c[h["face_index"]]) for h in hol_c if not h["is_flat"]]
    assert len(flat_vals) > 0 and max(flat_vals) < 1e-12
    assert len(curved_vals) == 1 and curved_vals[0] > 0.5
    print(f"test_gate5_wc1_mobius_ribbon_faces_and_bianchi # time: {time.perf_counter() - t0:.2f}s")


def test_python_rust_ao_parity(monkeypatch):
    import json
    import pytest
    from papertilling import _rust

    monkeypatch.setenv("PAPERTILLING_NO_RUST", "0")
    rs = _rust.get()
    if rs is None or not hasattr(rs, "relax_elastic_2d_patch_json"):
        pytest.skip("papertilling_rs extension with AO hooks not installed")

    t0 = time.perf_counter()
    cases = [
        ("square", [dict(kind="disclination", angle=90, at=(0, 0))], 4.0),
        ("square", [dict(kind="dislocation", b=(1, 0), at=(0, 0))], 6.0),
        ("shastry_sutherland", [dict(kind="disclination", angle=90, at=(0.5, 1.5))], 8.0),
    ]
    for lat, specs, rad in cases:
        monkeypatch.setenv("PAPERTILLING_NO_RUST", "0")
        c_on = pt.build(lat, *specs, R=rad, picture="elastic")
        monkeypatch.setenv("PAPERTILLING_NO_RUST", "1")
        c_off = pt.build(lat, *specs, R=rad, picture="elastic")
        e_on = c_on.defect.info["gsv"]["total_energy"]
        e_off = c_off.defect.info["gsv"]["total_energy"]
        assert abs(e_on - e_off) < 1e-9, f"{lat} energy mismatch: {e_on} vs {e_off}"
        assert _kabsch_so2_max_diff(c_on.defect.positions, c_off.defect.positions) < 1e-7

    # Also cross-check Layer 1 gauge observables (extension, Wyckoff, fusion, faces, plaquettes).
    from papertilling.defects.build import lattice_data
    ss_lat = lattice_data("shastry_sutherland")
    ss_cell = ss_lat.cell
    cell_json = json.dumps({
        "a1": ss_cell.A[:, 0].tolist(),
        "a2": ss_cell.A[:, 1].tolist(),
        "basis": [list(map(float, p)) for p in ss_cell.basis],
        "bonds": [{"i": int(i), "j": int(j), "n1": int(n1), "n2": int(n2), "kind": int(k)} for i, j, n1, n2, k in ss_cell.bonds],
    })
    pair_specs = [{"kind": "disclination", "angle": 90.0, "at": [0.5, 1.5]},
                  {"kind": "disclination", "angle": -90.0, "at": [1.5, 0.5]}]
    py_ext = space_group_extension("shastry_sutherland")
    rs_obs = json.loads(rs.gauge_observables_json(py_ext["group"], cell_json, json.dumps(pair_specs), 6.0, None))
    py_wy = wyckoff_rotocentres("shastry_sutherland")
    py_fus = fusion_channels("shastry_sutherland", pair_specs)
    py_raw = pt.build("shastry_sutherland", *pair_specs, R=6.0, picture="raw").defect
    py_faces = ribbon_faces(py_raw, include_boundary=True)
    py_hols = plaquette_holonomies(py_raw)
    assert rs_obs["extension"]["is_nonsymmorphic"] == py_ext["nonsymmorphic"]
    assert [w["label"] for w in rs_obs["wyckoff_rotocentres"]] == [c["label"] for c in py_wy["classes"]]
    proper_fus = [ch for ch in rs_obs["fusion_channels"] if ch["reachable_proper"]]
    assert len(proper_fus) >= 1 and all(ch["burgers_coset_rep"] != [0, 0] for ch in proper_fus)
    assert py_fus["contains_vacuum"] is False
    assert len(rs_obs["ribbon_faces"]) == len(py_faces)
    assert len(rs_obs["plaquettes"]) == len(py_hols)
    assert sum(1 for p in rs_obs["plaquettes"] if p["has_curvature"]) == sum(1 for h in py_hols if h["has_curvature"])
    print(f"test_python_rust_ao_parity # time: {time.perf_counter() - t0:.2f}s")
