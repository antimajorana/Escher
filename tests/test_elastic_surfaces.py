"""Tests for Stage 4 3D elastic relaxation (`papertilling/surfaces/elastic.py`).

Includes analytical / physical controls:
  * Exact discrete isometric cylinder at kappa_b=0 (stretch_energy < 1e-8) and nearly isometric open
    honeycomb cylinder at kappa_b=0.05 (stretch_energy < 1e-3; finite free rims & curved hexagons
    induce ~4.3e-4 boundary/sublattice strain at kappa_b=0.05).
  * C60 (sphere('honeycomb', (1, 1))): total_energy and bond-length MSE vs l0 drop strictly vs unit-sphere
    projection, exact I_h symmetry gives asphericity < 1e-5; on (2, 0) bond-length variance also drops.
  * FvK buckling transition on triangulated sphere (4, 0) (V=162): asphericity(1e-3) > 3 * asphericity(0.2).
  * Projective plane double-cover exact antipodal symmetry: ||cover_xyz[i+V] + cover_xyz[i]|| < 1e-12.
  * Defect Configuration 3D buckling smoke control.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

import papertilling as pt
from papertilling.surfaces import projective_plane, sphere, cylinder
from papertilling.surfaces.elastic import ElasticResult


def _bond_lengths(xyz: np.ndarray, bonds) -> np.ndarray:
    return np.array([np.linalg.norm(xyz[b] - xyz[a]) for a, b, _ in bonds], float)


def _bond_rest_lengths(bonds) -> np.ndarray:
    return np.array([math.hypot(float(v[0]), float(v[1])) for _, _, v in bonds], float)


def test_cylinder_honeycomb_elastic():
    t0 = time.perf_counter()
    S0 = cylinder("honeycomb")
    res0 = S0.relax_3d(kappa_b=0.0, max_iter=500)
    assert isinstance(res0, ElasticResult)
    assert res0.stretch_energy < 1e-8
    assert res0["bend_energy"] == 0.0
    assert res0.converged

    S = cylinder("honeycomb")
    res = S.relax_3d(kappa_b=0.05, max_iter=400)
    # Free open cylinder ends + curved hexagons with kappa_b=0.05 induce ~4e-4 edge/sublattice relaxation.
    assert res.stretch_energy < 1e-3
    assert float(np.sqrt(np.mean(res.bond_strain ** 2))) < 2e-3
    assert res.converged
    print(f"# time cylinder_honeycomb_elastic: {time.perf_counter() - t0:.2f} s")


def test_c60_and_fullerene_relaxation():
    t0 = time.perf_counter()
    S = sphere("honeycomb", (1, 1))
    xyz_init = S.xyz.copy()
    l_init = _bond_lengths(xyz_init, S.bonds)
    l0 = _bond_rest_lengths(S.bonds)
    mse_init = float(np.mean((l_init - l0) ** 2))
    e_init_stretch = 0.5 * float(np.sum((l_init - l0) ** 2))

    res = S.relax_3d(kappa_b=0.05, max_iter=400)
    assert res.converged
    assert res.total_energy < e_init_stretch
    l_fin = _bond_lengths(res.xyz, S.bonds)
    assert float(np.mean((l_fin - l0) ** 2)) < mse_init
    assert res.asphericity < 1e-5
    assert np.allclose(res.xyz.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(S.xyz, res["xyz"])

    # On non-vertex-transitive (2, 0), raw bond-length variance also drops strictly.
    S20 = sphere("honeycomb", (2, 0))
    var_init = float(np.var(_bond_lengths(S20.xyz, S20.bonds)))
    res20 = S20.relax_3d(kappa_b=0.05, max_iter=400)
    var_fin = float(np.var(_bond_lengths(res20.xyz, S20.bonds)))
    assert var_fin < var_init
    print(f"# time c60_and_fullerene: {time.perf_counter() - t0:.2f} s")


def test_triangular_sphere_fvk_buckling():
    t0 = time.perf_counter()
    S_soft = sphere("triangular", (4, 0))
    S_stiff = sphere("triangular", (4, 0))
    assert len(S_soft.sites) == 162

    res_soft = S_soft.relax_3d(kappa_b=1e-3, max_iter=400)
    res_stiff = S_stiff.relax_3d(kappa_b=0.2, max_iter=400)
    assert res_soft.asphericity > 3.0 * res_stiff.asphericity
    assert res_soft.fvk_gamma > res_stiff.fvk_gamma
    print(f"# time triangular_fvk: {time.perf_counter() - t0:.2f} s")


def test_projective_plane_antipodal_symmetry():
    t0 = time.perf_counter()
    p4g_lat = pt.load_catalogue()["p4g"].lattices[0]
    for P in (
        projective_plane("square", (2, 2)),
        projective_plane("square", (2, 2), centre="site"),
        projective_plane(p4g_lat, (1, 1)),
    ):
        V = len(P.sites)
        res = P.relax_3d(kappa_b=0.05, max_iter=400)
        assert res.cover_xyz is not None and res.cover_xyz.shape == (2 * V, 3)
        err = np.linalg.norm(res.cover_xyz[:V] + res.cover_xyz[V:], axis=1)
        assert float(np.max(err)) < 1e-12
        assert np.allclose(P.xyz, res.xyz)
        assert np.allclose(P.cover.xyz, res.cover_xyz)
    print(f"# time projective_plane: {time.perf_counter() - t0:.2f} s")


def test_configuration_relax_3d_smoke():
    t0 = time.perf_counter()
    sq = pt.named("square")
    conf = pt.build(sq, dict(kind="disclination", angle=90, at=(0, 0)))
    d = conf.defect
    l2d = np.array([np.linalg.norm(d.positions[v] - d.positions[u]) for u, v in d.edges])
    l0 = np.array([math.hypot(*d.edge[e]["vec"]) for e in d.edges])
    e_2d = 0.5 * float(np.sum((l2d - l0) ** 2))

    res = conf.relax_3d(kappa_b=0.02, max_iter=400)
    assert res.xyz.shape == (len(d.sites), 3)
    assert np.all(np.isfinite(res.xyz))
    assert float(np.ptp(res.xyz[:, 2])) > 0.5
    assert res.stretch_energy < e_2d
    assert res.cover_xyz is None
    assert np.allclose(conf.xyz, res.xyz)
    print(f"# time configuration_relax_3d: {time.perf_counter() - t0:.2f} s")


def test_analytical_gradient_matches_finite_differences():
    from papertilling.surfaces.elastic import _build_hinges, _eval_energy_grad

    rng = np.random.default_rng(7)
    P = rng.normal(scale=0.3, size=(5, 3)) + np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [1.0, 1.0, -0.1], [0.0, 1.0, 0.2], [0.5, 0.5, 0.15],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]
    su = np.array([u for u, _ in edges], int)
    sv = np.array([v for _, v in edges], int)
    sl0 = np.array([1.0, 1.0, 1.0, 1.0] + [math.sqrt(0.5)] * 4, float)
    sw = np.ones(len(edges), float)
    hp, hq, ha, hb, hw = _build_hinges([[0, 1, 2, 3]], {0: 4})
    es, eb, grad = _eval_energy_grad(P, su, sv, sl0, sw, hp, hq, ha, hb, hw, 0.15)
    eps = 1e-6
    fd = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(3):
            Pp, Pm = P.copy(), P.copy()
            Pp[i, j] += eps
            Pm[i, j] -= eps
            ep = sum(_eval_energy_grad(Pp, su, sv, sl0, sw, hp, hq, ha, hb, hw, 0.15)[:2])
            em = sum(_eval_energy_grad(Pm, su, sv, sl0, sw, hp, hq, ha, hb, hw, 0.15)[:2])
            fd[i, j] = (ep - em) / (2.0 * eps)
    assert np.allclose(grad, fd, atol=1e-7, rtol=1e-6)


def test_concave_polygon_flat_sheet_python_fallback():
    from papertilling.surfaces.elastic import _concave_face_triangulation, _run_elastic

    pts = np.array([[0.0, 0.0], [1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [-0.1, 0.0]], float)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4)]
    edge_l0 = np.array([np.linalg.norm(pts[v] - pts[u]) for u, v in edges], float)
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 4, 1]]
    quad3_inv = [[pts[u, 0], -pts[u, 1], 0.0] for u in faces[2]]
    assert _concave_face_triangulation(quad3_inv) is not None

    x0 = np.column_stack([pts, np.array([0.22, -0.11, 0.08, -0.14, 0.19], float)])
    c_quad = pts[faces[2]].mean(axis=0)
    spokes = {2: [float(np.linalg.norm(pts[u] - c_quad)) for u in faces[2]]}
    res = _run_elastic(5, x0, edges, edge_l0, list(range(len(edges))), faces, spokes, 0.05, 400,
                       is_proj=False, is_open_sheet=True)
    assert res.converged
    assert res.stretch_energy < 1e-10
    assert res.bend_energy < 1e-10
