//! Old (triplets + full `sp_cholesky` per factorisation) versus new (cached pattern and symbolic
//! analysis) Gauss-Newton solvers: bitwise-equal positions, costs, iteration and factorisation counts.

use faer::Mat;
use faer::prelude::Solve;

use crate::plane::planarity::check_positions;
use crate::plane::relax::{RelaxSetup, gauss_newton, relax_guarded, relax_picture, relax_pinned, work_copy_crystal, z_to_pos};
use crate::plane::relax_reference as old;
use crate::plane::test_fixtures::crystals;

fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn gauss_newton_cached_pattern_equals_reference_bitwise() {
    for (name, c) in crystals() {
        let w = work_copy_crystal(&c, &c.pos);
        let setup = RelaxSetup::from_crystal(&w, &c.pos);
        for (max_iter, freeze) in [(500, 1e-3), (7, 1e-3), (60, 1e-1)] {
            let (z_new, f_new, it_new, nf_new) = gauss_newton(&setup, &setup.x0, max_iter, 1e-10, freeze);
            let (z_old, f_old, it_old, nf_old) = old::gauss_newton(&setup, &setup.x0, max_iter, 1e-10, freeze);
            assert!(same_bits(&z_new, &z_old), "{name} ({max_iter}, {freeze}): positions differ");
            assert_eq!(f_new.to_bits(), f_old.to_bits(), "{name}: cost differs");
            assert_eq!((it_new, nf_new), (it_old, nf_old), "{name}: iteration/factorisation counts differ");
        }
    }
}

#[test]
fn relax_picture_equals_reference_bitwise() {
    for (name, c) in crystals() {
        let mut new = c.clone();
        relax_picture(&mut new);
        let mut reference = c.clone();
        let setup = RelaxSetup::from_crystal(&reference, &reference.pos);
        if !setup.springs.is_empty() {
            reference.pos = z_to_pos(&old::gauss_newton(&setup, &setup.x0, 500, 1e-10, 1e-3).0);
        }
        crate::defect::surgery::mark_long_bonds(&mut reference);
        assert_eq!(new, reference, "{name}: relaxed crystal differs");
    }
}

#[test]
fn pinned_and_guarded_steps_equal_reference_bitwise() {
    for (name, c) in crystals() {
        let w = work_copy_crystal(&c, &c.pos);
        let setup = RelaxSetup::from_crystal(&w, &c.pos);
        let n = c.pos.len();
        let free: Vec<usize> = (0..n).filter(|v| v % 5 != 0).flat_map(|v| [2 * v, 2 * v + 1]).collect();
        let (zp_new, ip_new) = relax_pinned(&setup, &setup.x0, &free, 40, 1e-9);
        let (zp_old, ip_old) = old::relax_pinned(&setup, &setup.x0, &free, 40, 1e-9);
        assert!(same_bits(&zp_new, &zp_old) && ip_new == ip_old, "{name}: relax_pinned differs");

        let bad = |z: &[f64]| {
            let r = check_positions(&c, &z_to_pos(z), false, None);
            1_000_000 * (r.crossings + r.folded.unwrap_or(0)) + r.overlaps
        };
        let (zg_new, sg_new) = relax_guarded(&setup, &setup.x0, bad, 12, 1e-8, 16, 40);
        let (zg_old, sg_old) = old::relax_guarded(&setup, &setup.x0, bad, 12, 1e-8, 16, 40);
        assert!(same_bits(&zg_new, &zg_old) && sg_new == sg_old, "{name}: relax_guarded differs");
    }
}

#[test]
fn levenberg_shifted_factor_equals_reference_bitwise() {
    // Direct check of the `mu > 0` (appended diagonal) and pinned-subset paths of the cached factor.
    for (name, c) in crystals().into_iter().take(6) {
        let w = work_copy_crystal(&c, &c.pos);
        let setup = RelaxSetup::from_crystal(&w, &c.pos);
        let ndof = 2 * setup.n;
        let rhs = Mat::from_fn(ndof, 1, |r, _| ((r * 7919) % 101) as f64 / 101.0 - 0.5);
        let mus = [0.0, 1e-8, 1e-4, 0.0, 1e-2, 10.0];
        let news = setup.factors_for_test(&setup.x0, &mus);
        for (mu, new) in mus.into_iter().zip(news) {
            let new = new.expect("new factor").solve(&rhs);
            let reference = old::factor_for_test(&setup, &setup.x0, mu).expect("old factor").solve(&rhs);
            let a: Vec<f64> = (0..ndof).map(|k| new[(k, 0)]).collect();
            let b: Vec<f64> = (0..ndof).map(|k| reference[(k, 0)]).collect();
            assert!(same_bits(&a, &b), "{name}: mu = {mu} solve differs");
        }
    }
}

#[test]
fn preview_lowers_the_picture_energy_without_changing_the_crystal_graph() {
    for (name, c) in crystals() {
        let setup = RelaxSetup::from_crystal(&c, &c.pos);
        for budget in [0.0, f64::INFINITY] {
            let mut preview = c.clone();
            let report = crate::plane::relax_preview::relax_preview(&mut preview, budget);
            assert!(preview.pos.iter().flatten().all(|x| x.is_finite()), "{name}");
            let z: Vec<f64> = preview.pos.iter().flatten().copied().collect();
            assert!(setup.cost(&z) <= setup.cost(&setup.x0) + 1e-10, "{name}");
            assert!(report.steps <= crate::plane::relax_preview::PREVIEW_STEPS);
            preview.pos.clone_from(&c.pos);
            // Only picture positions and display-only stretched-bond marks may change.
            for (out, input) in preview.adj.iter_mut().flatten().zip(c.adj.iter().flatten()) {
                out.ghost_stub = input.ghost_stub;
            }
            assert_eq!(preview, c, "{name}: preview changed mathematical data");
        }
    }
}
