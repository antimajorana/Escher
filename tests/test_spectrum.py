"""Tests for Stage 5 tight-binding spectrum, zero modes, multiplet LDOS, and cone charges."""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

import papertilling as pt
from papertilling import DefectCatalogue, build, named, spectrum
from papertilling.surfaces import sphere, torus


def test_bipartite_theorem_and_nonbipartite_c60():
    t0 = time.perf_counter()
    T = torus("honeycomb", (4, 0), (0, 4))
    res_t = spectrum(T, beta=0.0, flux=0.0)
    ev = res_t.eigenvalues
    assert len(ev) == 32
    assert abs(ev[0] + 3.0) < 1e-11
    assert abs(ev[-1] - 3.0) < 1e-11
    assert float(np.max(np.abs(ev + ev[::-1]))) < 1e-11
    assert res_t["zero_mode_count"] == int(np.sum(np.abs(ev) < 1e-6))

    C60 = sphere("honeycomb", (1, 1))
    res_c = spectrum(C60, beta=0.0, flux=0.0)
    ev_c = res_c.eigenvalues
    assert len(ev_c) == 60
    assert abs(ev_c[0] + 3.0) < 1e-11
    assert ev_c[-1] < 2.9
    print(f"# time bipartite_and_c60: {time.perf_counter() - t0:.2f} s")


def test_kagome_exact_flat_band():
    t0 = time.perf_counter()
    K = torus(named("kagome"), (4, 0), (0, 4))
    assert len(K.sites) == 48
    res = spectrum(K, beta=0.0, flux=0.0)
    ev = res.eigenvalues
    assert int(np.sum(np.abs(ev - 2.0) < 1e-8)) == 17
    assert int(np.sum(np.abs(ev + 4.0) < 1e-8)) == 1
    assert abs(ev[0] + 4.0) < 1e-11

    # Passing a Lattice directly builds the finite periodic 4x4 torus.
    res_lat = pt.spectrum(named("kagome"))
    assert np.allclose(res_lat.eigenvalues, ev, atol=1e-11)
    print(f"# time kagome_flat_band: {time.perf_counter() - t0:.2f} s")


def test_c60_vertex_transitive_multiplet_invariance():
    t0 = time.perf_counter()
    C60 = sphere("honeycomb", (1, 1))
    res = spectrum(C60, multiplet=True)
    assert sum(dim for _, _, _, dim in res.multiplets) == 60
    for _, start, end, dim in res.multiplets:
        ldos = res.ldos_for_level(start, multiplet=True)
        assert np.allclose(ldos, dim / 60.0, atol=1e-11)
        if dim > 1:
            ldos_single = res.ldos_for_level(start, multiplet=False)
            assert float(np.ptp(ldos_single)) > 1e-4

    # level_idx selects the multiplet directly in spectrum().
    res_l = spectrum(C60, level_idx=10, multiplet=True)
    assert np.allclose(res_l.ldos, res.ldos_for_level(10, multiplet=True), atol=1e-12)
    print(f"# time c60_multiplet_invariance: {time.perf_counter() - t0:.2f} s")


def test_closed_surface_neutrality_and_cone_symmetry():
    t0 = time.perf_counter()
    C60 = sphere("honeycomb", (1, 1))
    for mu in (-1.2, 0.0, 0.8):
        res = spectrum(C60, mu=mu)
        assert abs(float(np.sum(res.delta_rho))) < 1e-11
        assert len(res.delta_Q) == 12
        assert float(np.ptp(res.delta_Q)) < 1e-9

    res_fill = spectrum(C60, filling=0.5)
    assert abs(float(np.sum(res_fill.delta_rho))) < 1e-11
    assert abs(res_fill.mu - 0.5 * (-0.6180339887498949 + 0.13856406460551018)) < 1e-3

    Sq = sphere("square", (2, 0), centre="site")
    assert len(Sq.cones) == 8
    res_sq0 = spectrum(Sq, mu=0.0)
    assert res_sq0.zero_mode_count == 8
    assert abs(float(np.sum(res_sq0.delta_rho))) < 1e-11
    assert float(np.ptp(res_sq0.cone_charges)) < 1e-9

    res_sq_gap = spectrum(Sq, mu=0.5)
    assert abs(float(np.sum(res_sq_gap.delta_rho))) < 1e-11
    assert float(np.ptp(res_sq_gap.delta_Q)) < 1e-9
    print(f"# time neutrality_cone_symmetry: {time.perf_counter() - t0:.2f} s")


def test_strain_flux_and_wc_seam_sign():
    t0 = time.perf_counter()
    # Unstrained torus is unaffected by beta; strained dislocation patch responds to beta > 0.
    T = torus("honeycomb", (4, 0), (0, 4))
    assert np.allclose(spectrum(T, beta=0.0).eigenvalues, spectrum(T, beta=2.0).eigenvalues, atol=1e-11)

    sq = named("square")
    disl = build(sq, dict(kind="dislocation", b=(1, 0), at=(0.371, 0.413)))
    r0 = spectrum(disl, beta=0.0, flux=0.0)
    r_strain = spectrum(disl, beta=2.0, flux=0.0)
    assert float(np.max(np.abs(r_strain.eigenvalues - r0.eigenvalues))) > 1e-2

    # Non-zero Peierls flux breaks time-reversal (complex eigenvectors, shifted spectrum, Hermitian evals).
    r_flux = spectrum(disl, beta=0.0, flux=0.25)
    assert np.iscomplexobj(r_flux.eigenvectors)
    assert float(np.max(np.abs(r_flux.eigenvalues - r0.eigenvalues))) > 1e-2

    # Wc Moebius seam: seam_sign = +1 vs -1 (pi-flux twist introduces exact zero modes).
    wc = DefectCatalogue(sq).mobius_seams[0]
    r_p = spectrum(wc, seam_sign=1)
    r_m = spectrum(wc, seam_sign=-1)
    assert r_p.zero_mode_count == 0
    assert r_m.zero_mode_count == 6
    assert float(np.max(np.abs(r_p.eigenvalues - r_m.eigenvalues))) > 0.5

    # Open non-bipartite disclination patch uses half-filling reference nu_ref = 0.5 at mu = 0.
    hc = named("honeycomb")
    c6 = next(r["point"] for r in DefectCatalogue(hc).centre_classes if r["order"] == 6)
    disc60 = build(hc, dict(kind="disclination", angle=60, at=c6))
    r_disc = spectrum(disc60, mu=0.0)
    assert len(r_disc.delta_Q) >= 1
    assert np.all(np.isfinite(r_disc.core_weight))
    print(f"# time strain_flux_wc: {time.perf_counter() - t0:.2f} s")
