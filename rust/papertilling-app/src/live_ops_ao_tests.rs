//! C1 known-answer tests of the 2D overlays kept after the AO elastic view left the UI (plan §4, T6). Every
//! expected value is derived by hand in the comment next to it, never read from the code under test.

use papertilling_core::defect::gsv_elastic::gsv_diagnostics_2d;
use papertilling_core::types::{Bond2D, BuiltCrystal, Cell2D, DirectedBondData, Frame2D};

use crate::canvas2d::gauge_overlay::CrystalOverlayMode;
use crate::defects_inspectors::overlay_shown;
use crate::live_ops_ao::{adjacent_angle_shear, build_ao_crystal_overlay};

/// An `n x n` patch of the unit square lattice (sites at integer points, bonds between nearest neighbours), with
/// rest bond vectors of length 1 and its positions mapped by `f`.
fn square_patch(n: usize, f: impl Fn([f64; 2]) -> [f64; 2]) -> BuiltCrystal {
    let id = |i: usize, j: usize| i * n + j;
    let rest: Vec<[f64; 2]> = (0..n * n).map(|k| [(k / n) as f64, (k % n) as f64]).collect();
    let mut adj = vec![Vec::new(); n * n];
    let mut edges = Vec::new();
    let mut bond = |u: usize, v: usize| {
        let d = [rest[v][0] - rest[u][0], rest[v][1] - rest[u][1]];
        let mk = |to, vec: [f64; 2]| DirectedBondData {
            to,
            vec,
            kind: 0,
            seam: None,
            ang: Some(vec[1].atan2(vec[0])),
            across: false,
            ghost_stub: None,
        };
        adj[u].push(mk(v, d));
        adj[v].push(mk(u, [-d[0], -d[1]]));
        edges.push((u, v));
    };
    for i in 0..n {
        for j in 0..n {
            if i + 1 < n {
                bond(id(i, j), id(i + 1, j));
            }
            if j + 1 < n {
                bond(id(i, j), id(i, j + 1));
            }
        }
    }
    let pos: Vec<[f64; 2]> = rest.iter().map(|&p| f(p)).collect();
    BuiltCrystal {
        ref_pos: rest,
        basis_idx: vec![0; n * n],
        side: vec![None; n * n],
        cone_sites: Vec::new(),
        edges,
        adj,
        wall_sigma: None,
        is_non_orientable: false,
        pos,
    }
}

fn square_cell() -> Cell2D {
    Cell2D {
        a1: [1.0, 0.0],
        a2: [0.0, 1.0],
        basis: vec![[0.0, 0.0]],
        bonds: vec![Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 }, Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 }],
    }
}

#[test]
fn bond_stretch_is_eps_on_every_bond_under_uniform_stretch() {
    // x -> (1+eps) x stretches every unit bond to length 1+eps: engineering strain (len - 1)/1 = eps exactly.
    let eps = 0.03;
    let c = square_patch(5, |p| [(1.0 + eps) * p[0], (1.0 + eps) * p[1]]);
    let d = gsv_diagnostics_2d(&c, &c.pos).expect("diagnostics");
    assert_eq!(d.bond_strain.len(), 40, "5x5 square patch: 2 * 5 * 4 = 40 bonds");
    for (k, s) in d.bond_strain.iter().enumerate() {
        assert!((s - eps).abs() < 1e-12, "bond {k}: strain {s} != {eps}");
    }
    let ov = build_ao_crystal_overlay(&c, CrystalOverlayMode::BondStretchStrain, "p4m", &square_cell(), Frame2D::default());
    assert!(ov.is_some(), "the kept bond-stretch overlay draws");
    assert!(overlay_shown(CrystalOverlayMode::BondStretchStrain));
}

#[test]
fn bond_stretch_is_zero_under_a_rigid_rotation() {
    // a rotation keeps every length: strain 0 on every bond.
    let (s, co) = 0.4_f64.sin_cos();
    let c = square_patch(4, |p| [co * p[0] - s * p[1], s * p[0] + co * p[1]]);
    let d = gsv_diagnostics_2d(&c, &c.pos).expect("diagnostics");
    assert!(d.bond_strain.iter().all(|e| e.abs() < 1e-12), "{:?}", d.bond_strain);
}

/// Every site value and every bond value of the adjacent-bond angle shear on `c`.
fn angle_shear_values(c: &BuiltCrystal) -> Vec<f64> {
    let (site, bond) = adjacent_angle_shear(c);
    assert_eq!((site.len(), bond.len()), (c.pos.len(), c.edges.len()));
    site.into_iter().chain(bond).collect()
}

#[test]
fn vertex_angle_shear_is_zero_under_a_uniform_stretch_and_a_rotation() {
    // C1 WP-U7 fix. A uniform stretch x -> (1+eps) x and a rotation keep every angle between bonds, so every
    // adjacent-pair angle change is 0 and so is every site and bond mean (the old overlay gave
    // (2 eps + eps^2)^2 / 4 at an interior site, because it also paired opposite bonds).
    let eps = 0.03;
    let c = square_patch(5, |p| [(1.0 + eps) * p[0], (1.0 + eps) * p[1]]);
    assert!(angle_shear_values(&c).iter().all(|v| v.abs() < 1e-12), "uniform stretch: {:?}", angle_shear_values(&c));
    let (s, co) = 0.4_f64.sin_cos();
    let r = square_patch(4, |p| [co * p[0] - s * p[1], s * p[0] + co * p[1]]);
    assert!(angle_shear_values(&r).iter().all(|v| v.abs() < 1e-12), "rotation");
    let ov = build_ao_crystal_overlay(&c, CrystalOverlayMode::VertexAngleShear, "p4m", &square_cell(), Frame2D::default());
    assert!(ov.is_some_and(|o| o.site_rgba.is_some() && o.bond_rgba.is_some()), "the angle-shear overlay draws");
    assert!(overlay_shown(CrystalOverlayMode::VertexAngleShear), "passing its known-answer tests shows it again");
    assert!(!overlay_shown(CrystalOverlayMode::GsvEnergyShear), "the GSV energy is part of the AO view");
}

#[test]
fn vertex_angle_shear_matches_the_hand_derived_pure_and_simple_shear() {
    // Pure shear F = [[1, h], [h, 1]], h = gamma/2: e1 -> (1, h) at angle atan h, e2 -> (h, 1) at angle pi/2 - atan h,
    // -e1 and -e2 opposite. Every adjacent pair of a square site (rest angle 90 deg) opens or closes by 2 atan h:
    // (e1, e2) closes to pi/2 - 2 atan h, (e2, -e1) opens to pi/2 + 2 atan h, and so on round the site. Rim and
    // corner sites have only such 90 deg pairs (their 180 deg gaps are skipped), so EVERY site and bond reads
    // 2 atan(gamma/2) ~ gamma.
    let gamma = 0.1_f64;
    let h = gamma / 2.0;
    let want = 2.0 * h.atan();
    let c = square_patch(5, |p| [p[0] + h * p[1], h * p[0] + p[1]]);
    for v in angle_shear_values(&c) {
        assert!((v - want).abs() < 1e-12, "pure shear: {v} vs 2 atan(gamma/2) = {want}");
    }
    // Simple shear x -> x + gamma y: e1 -> (1, 0), e2 -> (gamma, 1) at angle pi/2 - atan gamma, so every 90 deg pair
    // changes by atan gamma.
    let s = square_patch(5, |p| [p[0] + gamma * p[1], p[1]]);
    for v in angle_shear_values(&s) {
        assert!((v - gamma.atan()).abs() < 1e-12, "simple shear: {v} vs atan gamma = {}", gamma.atan());
    }
}

#[test]
fn plaquette_flux_is_zero_on_a_defect_free_patch() {
    // A perfect patch has trivial holonomy around every plaquette: no curvature, torsion or shift badge.
    let c = square_patch(5, |p| p);
    let ov = build_ao_crystal_overlay(&c, CrystalOverlayMode::PlaquetteFlux, "p4m", &square_cell(), Frame2D::default())
        .expect("plaquette overlay");
    assert!(ov.badges.is_empty(), "{} badges on a defect-free patch", ov.badges.len());
    assert!(overlay_shown(CrystalOverlayMode::PlaquetteFlux));
}
