//! Stage 4 tests: 3D elastic relaxation (`relax_surface_3d`) on cylinders,
//! buckyball/icosahedral spheres, and RP2 antipodal double covers.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use papertilling_core::plane::catalogue::{CatalogueRule, load_catalogue, load_named};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::surface::elastic3d::relax_surface_3d;
use papertilling_core::surface::{default_cylinder, projective_plane, sphere};
use papertilling_core::types::SurfaceMesh;

fn repo_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../papertilling/data")
}

fn named_cell(key: &str) -> (String, papertilling_core::types::Cell2D) {
    let named = load_named(&repo_data_dir()).expect("named.json");
    let e = named.get(key).unwrap_or_else(|| panic!("missing named lattice {key}"));
    let fam = LatticeFamily::from_recipe(&e.recipe, Some(e.frame)).expect("valid recipe");
    (fam.group.clone(), fam.cell())
}

fn raw_stretch_and_rmse(mesh: &SurfaceMesh) -> (f64, f64) {
    let mut bmap: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for &(a, b, v) in &mesh.bonds {
        bmap.entry((a.min(b), a.max(b))).or_insert_with(|| v[0].hypot(v[1]));
    }
    let (mut e_s, mut mse) = (0.0, 0.0);
    for (&(u, v), &l0) in &bmap {
        let p = mesh.xyz[u];
        let q = mesh.xyz[v];
        let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
        e_s += 0.5 * (d - l0).powi(2);
        mse += (d - l0).powi(2);
    }
    (e_s, if bmap.is_empty() { 0.0 } else { (mse / (bmap.len() as f64)).sqrt() })
}

#[test]
fn elastic3d_honeycomb_cylinder_is_isometric() {
    let t0 = Instant::now();
    let (_grp, hc_cell) = named_cell("honeycomb");
    let cyl = default_cylinder(&hc_cell).expect("honeycomb cylinder");

    let r0 = relax_surface_3d(&cyl, 0.0, 400).expect("relax kappa=0");
    assert!(r0.converged);
    assert!(r0.stretch_energy < 1e-8, "expected < 1e-8, got {}", r0.stretch_energy);

    let r1 = relax_surface_3d(&cyl, 0.05, 400).expect("relax kappa=0.05");
    assert!(r1.converged);
    assert!(r1.stretch_energy < 1e-3, "expected < 1e-3, got {}", r1.stretch_energy);
    assert!(r1.fvk_gamma.is_finite() && r1.fvk_gamma > 0.0);
    println!("# time elastic3d cylinder: {:.4} s", t0.elapsed().as_secs_f64());
}

#[test]
fn elastic3d_c60_sphere_relaxes_and_rounds() {
    let t0 = Instant::now();
    let (grp, hc_cell) = named_cell("honeycomb");
    let c60 = sphere(&grp, &hc_cell, Some((1, 1)), None).expect("C60 sphere");
    assert_eq!(c60.sites.len(), 60);

    let (e_raw, rmse_raw) = raw_stretch_and_rmse(&c60);
    let e_scaled = relax_surface_3d(&c60, 0.05, 0).expect("scaled init").total_energy;
    let res = relax_surface_3d(&c60, 0.05, 400).expect("relax C60");
    assert!(res.converged, "C60 did not converge");
    assert!(res.total_energy < e_raw && res.total_energy < e_scaled);
    assert!(res.asphericity < 1e-5, "C60 asphericity too large: {}", res.asphericity);

    let mut bmap: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for &(a, b, v) in &c60.bonds {
        bmap.entry((a.min(b), a.max(b))).or_insert_with(|| v[0].hypot(v[1]));
    }
    let rmse_fin = (bmap
        .iter()
        .map(|(&(u, v), &l0)| {
            let p = res.xyz[u];
            let q = res.xyz[v];
            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
            (d - l0).powi(2)
        })
        .sum::<f64>()
        / (bmap.len() as f64))
        .sqrt();
    assert!(rmse_fin < rmse_raw, "bond-length RMSE did not drop: {rmse_raw} -> {rmse_fin}");
    println!("# time elastic3d C60: {:.4} s (iters={})", t0.elapsed().as_secs_f64(), res.n_iter);
}

#[test]
fn elastic3d_triangular_sphere_buckles_at_low_kappa() {
    let t0 = Instant::now();
    let (grp, tri_cell) = named_cell("triangular");
    let sph = sphere(&grp, &tri_cell, Some((4, 0)), None).expect("triangular (4,0) sphere");
    assert_eq!(sph.sites.len(), 162);

    let soft = relax_surface_3d(&sph, 1e-3, 500).expect("soft relax");
    let stiff = relax_surface_3d(&sph, 0.2, 500).expect("stiff relax");
    assert!(soft.converged && stiff.converged);
    assert!(
        soft.asphericity > 3.0 * stiff.asphericity,
        "expected buckling ratio > 3: soft={} stiff={}",
        soft.asphericity,
        stiff.asphericity
    );
    println!(
        "# time elastic3d tri(4,0): {:.4} s (asph soft={:.3e}, stiff={:.3e})",
        t0.elapsed().as_secs_f64(),
        soft.asphericity,
        stiff.asphericity
    );
}

#[test]
fn elastic3d_rp2_preserves_exact_antipodal_double_cover() {
    let t0 = Instant::now();
    let (sq_grp, sq_cell) = named_cell("square");
    let rp2_sq = projective_plane(&sq_grp, &sq_cell, None, None).expect("square RP2");

    let pts_cat = load_catalogue(&repo_data_dir(), CatalogueRule::Points).expect("points.json");
    let p4g_entry = &pts_cat["p4g"].lattices[0];
    let rp2_p4g = projective_plane("p4g", &p4g_entry.family.cell(), None, None).expect("p4g RP2");

    for (label, mesh) in [("square", &rp2_sq), ("p4g#1", &rp2_p4g)] {
        let v_count = mesh.sites.len();
        let res = relax_surface_3d(mesh, 0.05, 400).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(res.converged, "{label} did not converge");
        let cov = res.cover_xyz.as_ref().expect("RP2 cover_xyz must be Some");
        assert_eq!(cov.len(), 2 * v_count);
        for i in 0..v_count {
            assert_eq!(res.xyz[i], cov[i]);
            let anti = [
                cov[i + v_count][0] + cov[i][0],
                cov[i + v_count][1] + cov[i][1],
                cov[i + v_count][2] + cov[i][2],
            ];
            let err = (anti[0] * anti[0] + anti[1] * anti[1] + anti[2] * anti[2]).sqrt();
            assert!(err < 1e-12, "{label} antipodal err at {i}: {err}");
        }
    }
    println!("# time elastic3d RP2: {:.4} s", t0.elapsed().as_secs_f64());
}
