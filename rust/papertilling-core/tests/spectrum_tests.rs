//! Stage 5 tests: tight-binding spectra, flat bands, C60 symmetry multiplets,
//! spectral neutrality, cone geodesic charges, and Peierls flux Hermiticity.

use std::path::PathBuf;
use std::time::Instant;

use papertilling_core::defect::DefectCatalogue;
use papertilling_core::plane::catalogue::load_named;
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::spectrum::{
    Eigenvectors, SpectrumOptions, crystal_spectrum, ldos_for_level, surface_spectrum,
};
use papertilling_core::surface::{sphere, torus};
use papertilling_core::types::Frame2D;

fn repo_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../papertilling/data")
}

fn named_cell(key: &str) -> (String, papertilling_core::types::Cell2D) {
    let named = load_named(&repo_data_dir()).expect("named.json");
    let e = named.get(key).unwrap_or_else(|| panic!("missing named lattice {key}"));
    let fam = LatticeFamily::from_recipe(&e.recipe, Some(e.frame)).expect("valid recipe");
    (fam.group.clone(), fam.cell())
}

#[test]
fn spectrum_honeycomb_bipartite_torus_vs_frustrated_c60() {
    let t0 = Instant::now();
    let (hc_grp, hc_cell) = named_cell("honeycomb");
    let tor = torus(&hc_cell, [4, 0], [0, 4]).expect("honeycomb 4x4 torus");
    let opts = SpectrumOptions::default();

    let res_tor = surface_spectrum(&tor, &opts).expect("torus spectrum");
    let n = res_tor.eigenvalues.len();
    assert_eq!(n, 32);
    assert!((res_tor.eigenvalues[0] + 3.0).abs() < 1e-11);
    assert!((res_tor.eigenvalues[n - 1] - 3.0).abs() < 1e-11);
    for k in 0..n {
        assert!(
            (res_tor.eigenvalues[k] + res_tor.eigenvalues[n - 1 - k]).abs() < 1e-11,
            "particle-hole asymmetry at {k}"
        );
    }

    let c60 = sphere(&hc_grp, &hc_cell, Some((1, 1)), None).expect("C60");
    let res_c60 = surface_spectrum(&c60, &opts).expect("C60 spectrum");
    assert_eq!(res_c60.eigenvalues.len(), 60);
    assert!((res_c60.eigenvalues[0] + 3.0).abs() < 1e-11);
    assert!(res_c60.eigenvalues[59] < 2.9, "expected frustrated top < 2.9, got {}", res_c60.eigenvalues[59]);
    println!("# time spectrum honeycomb vs C60: {:.4} s", t0.elapsed().as_secs_f64());
}

#[test]
fn spectrum_kagome_4x4_torus_flat_band_and_flux() {
    let t0 = Instant::now();
    let (_kag_grp, kag_cell) = named_cell("kagome");
    let tor = torus(&kag_cell, [4, 0], [0, 4]).expect("kagome 4x4 torus");
    assert_eq!(tor.sites.len(), 48);

    let res = surface_spectrum(&tor, &SpectrumOptions::default()).expect("kagome spectrum");
    assert!((res.eigenvalues[0] + 4.0).abs() < 1e-11);
    let m0 = res.multiplets.iter().find(|m| (m.energy + 4.0).abs() < 1e-9).expect("E=-4 multiplet");
    assert_eq!(m0.dim, 1);
    let flat = res.multiplets.iter().find(|m| (m.energy - 2.0).abs() < 1e-9).expect("E=+2 flat band");
    assert_eq!(flat.dim, 17, "expected multiplicity 17 at E=+2, got {}", flat.dim);

    // Non-zero symmetric-gauge flux triggers complex Hermitian eigensolver.
    let flux_opts = SpectrumOptions { flux: 0.25, ..SpectrumOptions::default() };
    let res_f = surface_spectrum(&tor, &flux_opts).expect("kagome flux spectrum");
    assert!(matches!(res_f.eigenvectors, Eigenvectors::Complex(_)));
    assert_eq!(res_f.eigenvalues.len(), 48);
    assert!(res_f.eigenvalues.windows(2).all(|w| w[0] <= w[1] + 1e-12));
    assert!((res_f.delta_rho.iter().sum::<f64>()).abs() < 1e-11);
    println!("# time spectrum kagome 4x4: {:.4} s", t0.elapsed().as_secs_f64());
}

#[test]
fn spectrum_c60_vertex_transitive_multiplet_invariance() {
    let t0 = Instant::now();
    let (hc_grp, hc_cell) = named_cell("honeycomb");
    let c60 = sphere(&hc_grp, &hc_cell, Some((1, 1)), None).expect("C60");
    let res = surface_spectrum(&c60, &SpectrumOptions::default()).expect("C60 spectrum");

    for (idx, m) in res.multiplets.iter().enumerate() {
        let ldos = ldos_for_level(&res.eigenvectors, &res.multiplets, m.start as isize, true).expect("ldos");
        assert_eq!(ldos.len(), 60);
        let target = (m.dim as f64) / 60.0;
        for (site, &val) in ldos.iter().enumerate() {
            assert!(
                (val - target).abs() < 1e-11,
                "multiplet {idx} (E={}, dim={}) site {site}: got {val} != {target}",
                m.energy,
                m.dim
            );
        }
    }
    println!("# time spectrum C60 multiplets: {:.4} s", t0.elapsed().as_secs_f64());
}

#[test]
fn spectrum_cone_neutrality_and_equivalent_cone_charges() {
    let t0 = Instant::now();
    let (hc_grp, hc_cell) = named_cell("honeycomb");
    let c60 = sphere(&hc_grp, &hc_cell, Some((1, 1)), None).expect("C60");
    let (sq_grp, sq_cell) = named_cell("square");
    let sq_sph = sphere(&sq_grp, &sq_cell, Some((2, 0)), Some("site")).expect("square (2,0) site sphere");
    assert_eq!(sq_sph.cones.len(), 8);

    for (label, mesh) in [("C60", &c60), ("sq(2,0)site", &sq_sph)] {
        for &mu in &[-1.5_f64, 0.0_f64, 1.2_f64] {
            let opts = SpectrumOptions { mu, ..SpectrumOptions::default() };
            let res = surface_spectrum(mesh, &opts).unwrap_or_else(|e| panic!("{label}: {e}"));
            let sum_rho: f64 = res.delta_rho.iter().sum();
            assert!(sum_rho.abs() < 1e-11, "{label} mu={mu}: neutrality residual {sum_rho}");
            assert_eq!(res.delta_q.len(), mesh.cones.len());
            let q0 = res.delta_q[0];
            for (ci, &q) in res.delta_q.iter().enumerate() {
                assert!(
                    (q - q0).abs() < 1e-9,
                    "{label} mu={mu} cone {ci}: {q} != {q0}"
                );
            }
        }
    }

    // Exercise seam_sign = +1 vs -1 on a square Wc1 Möbius crystal.
    let sq_cat = DefectCatalogue::new(&sq_grp, &sq_cell, Frame2D::default());
    let wc1 = sq_cat.build_wall("Wc1", None).expect("Wc1 crystal");
    let r_pos = crystal_spectrum(&wc1, &SpectrumOptions { seam_sign: 1, ..Default::default() }).expect("Wc1 +1");
    let r_neg = crystal_spectrum(&wc1, &SpectrumOptions { seam_sign: -1, ..Default::default() }).expect("Wc1 -1");
    assert_eq!(r_pos.eigenvalues.len(), r_neg.eigenvalues.len());
    let max_diff = r_pos.eigenvalues.iter().zip(&r_neg.eigenvalues).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
    assert!(max_diff > 1e-3, "expected antiperiodic seam flip to shift Wc1 spectrum, got max_diff={max_diff}");
    println!("# time spectrum cone charge + Wc1 seam: {:.4} s", t0.elapsed().as_secs_f64());
}
