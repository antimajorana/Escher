use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Instant;

use super::*;
use crate::defect::catalogue::DefectCatalogue;
use crate::defect::core_graph::LatticeCoreInfo;
use crate::defect::core_graph::TWO_PI;
use crate::defect::surgery::{DefectSpec, build_point_defects};
use crate::plane::catalogue::parse_named_str;
use crate::types::{DirectedBondData, Frame2D};

const NAMED_JSON: &str = include_str!("../../../../papertilling/data/named.json");

fn named_cell(key: &str) -> (String, Cell2D, Frame2D) {
    let cat = parse_named_str(NAMED_JSON).expect("valid named.json");
    let nl = cat
        .named
        .into_iter()
        .find(|n| n.key == key)
        .expect("named lattice exists");
    (nl.group, nl.cell, nl.frame)
}

fn pristine_disk(cell: &Cell2D, radius: f64) -> BuiltCrystal {
    let info = LatticeCoreInfo::new(cell);
    let keys = info.lat.sites_in_disk([0.0, 0.0], radius);
    let n = keys.len();
    let idx_map: HashMap<_, usize> = keys
        .iter()
        .copied()
        .enumerate()
        .map(|(i, k)| (k, i))
        .collect();
    let pos: Vec<[f64; 2]> = keys.iter().map(|&k| info.lat.pos(k)).collect();
    let basis_idx: Vec<usize> = keys.iter().map(|&k| k.2).collect();
    let mut adj = vec![Vec::new(); n];
    let mut edges = Vec::new();
    for (i, &k) in keys.iter().enumerate() {
        for (k2, kind) in info.lat.nbrs(k) {
            if let Some(&j) = idx_map.get(&k2) {
                if i < j {
                    edges.push((i, j));
                }
                adj[i].push(DirectedBondData {
                    to: j,
                    vec: [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]],
                    kind,
                    seam: None,
                    ang: None,
                    across: false,
                    ghost_stub: None,
                });
            }
        }
    }
    edges.sort_unstable();
    BuiltCrystal {
        ref_pos: pos.clone(),
        pos,
        basis_idx,
        side: vec![None; n],
        cone_sites: Vec::new(),
        edges,
        adj,
        wall_sigma: None,
        is_non_orientable: false,
    }
}

#[test]
fn gate1_pristine_flat_plane_control() {
    for key in ["square", "honeycomb", "kagome"] {
        let (_, cell, _) = named_cell(key);
        let disk = pristine_disk(&cell, 4.5);
        assert!(disk.pos.len() > 40, "{key}: nontrivial patch");
        let t0 = Instant::now();
        let res = relax_sheet_3d(&disk, &cell, &Sheet3DOptions::default(), None);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let max_z = res.xyz.iter().map(|p| p[2].abs()).fold(0.0_f64, f64::max);
        let tot_e = res.stretch_energy + res.bend_energy;
        println!(
            "# time pristine {key} ({} sites): {ms:.3} ms, max|z|={max_z:.2e}, E={tot_e:.2e}",
            res.xyz.len()
        );
        assert!(max_z < 1e-6, "{key}: max |z| = {max_z}");
        assert!(tot_e < 1e-10, "{key}: total energy = {tot_e}");
        assert!(
            res.vertex_omega.iter().all(|w| w.abs() < 1e-9),
            "{key}: zero Gaussian curvature"
        );
    }
}

#[test]
fn gate2_positive_disclination_cone_buckling() {
    // +60 deg disclination on honeycomb (6-fold hexagon centre) and +90 deg on square (4-fold site).
    for (key, ang, expected_omega_rad) in [
        ("honeycomb", 60.0, std::f64::consts::PI / 3.0),
        ("square", 90.0, std::f64::consts::FRAC_PI_2),
    ] {
        let (grp, cell, frame) = named_cell(key);
        let cat = DefectCatalogue::new(&grp, &cell, frame);
        let entry = cat
            .disclinations
            .iter()
            .find(|d| (d.angle - ang).abs() < 1e-6)
            .expect("positive disclination exists");
        let crystal = build_point_defects(
            &cell,
            &[DefectSpec::disclination(entry.angle, entry.point)],
            Some(5.5),
            None,
        )
        .unwrap();
        let opts = Sheet3DOptions {
            kappa_b: 0.005,
            max_iters: 260,
            grad_tol: 1e-6,
        };
        let t0 = Instant::now();
        let res = relax_sheet_3d(&crystal, &cell, &opts, None);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        let z_min = res.xyz.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
        let z_max = res
            .xyz
            .iter()
            .map(|p| p[2])
            .fold(f64::NEG_INFINITY, f64::max);
        let dz = z_max - z_min;
        let bond_red = 1.0 - res.bond_stretch_energy / res.planar_2d_bond_energy.max(1e-15);
        let stretch_red = 1.0 - res.stretch_energy / res.planar_2d_stretch_energy.max(1e-15);
        let total_omega: f64 = res.vertex_omega.iter().sum();

        println!(
            "# time +{ang:.0} cone on {key} ({} sites, {} iters): {ms:.2} ms | dz/a0={:.2} | bond_red={:.1}% | stretch_red={:.1}% | sum_omega={:.3} (want {:.3})",
            res.xyz.len(),
            res.iterations,
            dz / res.mean_bond_len,
            bond_red * 100.0,
            stretch_red * 100.0,
            total_omega,
            expected_omega_rad
        );

        assert!(
            dz > 0.5 * res.mean_bond_len,
            "{key}: cone height dz={dz} <= 0.5*a0"
        );
        assert!(
            bond_red > 0.80,
            "{key}: nearest-neighbour bond energy reduction {:.1}% <= 80%",
            bond_red * 100.0
        );
        assert!(
            stretch_red > 0.80,
            "{key}: total stretch energy reduction {:.1}% <= 80%",
            stretch_red * 100.0
        );
    }
}

#[test]
fn gate3_negative_disclination_saddle_buckling() {
    for (key, ang) in [("honeycomb", -60.0), ("square", -90.0)] {
        let (grp, cell, frame) = named_cell(key);
        let cat = DefectCatalogue::new(&grp, &cell, frame);
        let entry = cat
            .disclinations
            .iter()
            .find(|d| (d.angle - ang).abs() < 1e-6)
            .expect("negative disclination exists");
        let crystal = build_point_defects(
            &cell,
            &[DefectSpec::disclination(entry.angle, entry.point)],
            Some(5.5),
            None,
        )
        .unwrap();
        let opts = Sheet3DOptions {
            kappa_b: 0.008,
            max_iters: 260,
            grad_tol: 1e-6,
        };
        let t0 = Instant::now();
        let res = relax_sheet_3d(&crystal, &cell, &opts, None);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;

        let r_max = res
            .xyz
            .iter()
            .map(|p| p[0].hypot(p[1]))
            .fold(0.0_f64, f64::max);
        let z_span = res
            .xyz
            .iter()
            .map(|p| p[2])
            .fold(f64::NEG_INFINITY, f64::max)
            - res.xyz.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
        assert!(
            z_span > 0.5 * res.mean_bond_len,
            "{key}: saddle z_span={z_span}"
        );

        // Collect outer-ring vertices (`0.65 * r_max ..= 0.98 * r_max`) sorted by polar angle `atan2(y, x)`
        // and count sign changes in `z` (a saddle `cos(2*theta)` has >= 4 sign changes around the loop!).
        let mut ring: Vec<(f64, f64)> = res
            .xyz
            .iter()
            .filter_map(|p| {
                let r = p[0].hypot(p[1]);
                (r >= 0.65 * r_max && p[2].abs() > 0.03 * res.mean_bond_len)
                    .then(|| (p[1].atan2(p[0]), p[2]))
            })
            .collect();
        ring.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Smooth angularly into 24 angular bins to measure genuine macroscopic lobes, not lattice jitter.
        let mut bins = [0.0_f64; 24];
        let mut cnts = [0_usize; 24];
        for &(th, z) in &ring {
            let b = (((th + PI) / TWO_PI * 24.0).floor() as usize).min(23);
            bins[b] += z;
            cnts[b] += 1;
        }
        let signs: Vec<i32> = (0..24)
            .filter(|&b| cnts[b] > 0 && (bins[b] / cnts[b] as f64).abs() > 0.02 * res.mean_bond_len)
            .map(|b| if bins[b] > 0.0 { 1 } else { -1 })
            .collect();
        let mut flips = 0;
        for m in 0..signs.len() {
            if signs[m] != signs[(m + 1) % signs.len()] {
                flips += 1;
            }
        }
        println!(
            "# time {ang:.0} saddle on {key} ({} sites): {ms:.2} ms | z_span/a0={:.2} | angular sign changes={flips}",
            res.xyz.len(),
            z_span / res.mean_bond_len
        );
        assert!(
            flips >= 4,
            "{key}: expected saddle >= 4 sign changes around outer ring, got {flips}"
        );
    }
}

#[test]
fn gate4_dipole_dislocation_elastic_screening() {
    let (_, cell, _) = named_cell("square");
    let r_patch = 8.0;
    let isolated = build_point_defects(
        &cell,
        &[DefectSpec::disclination(90.0, [0.0, 0.0])],
        Some(r_patch),
        None,
    )
    .unwrap();
    // +90 at site (0,0) and -90 at nearest face centre (0.5, 0.5): equivalent to an elementary dislocation.
    let dipole = build_point_defects(
        &cell,
        &[
            DefectSpec::disclination(90.0, [0.0, 0.0]),
            DefectSpec::disclination(-90.0, [0.5, 0.5]),
        ],
        Some(r_patch),
        None,
    )
    .unwrap();

    let opts = Sheet3DOptions {
        kappa_b: 0.02,
        max_iters: 260,
        grad_tol: 1e-6,
    };
    let res_iso = relax_sheet_3d(&isolated, &cell, &opts, None);
    let res_dip = relax_sheet_3d(&dipole, &cell, &opts, None);

    let rms_z = |r: &Sheet3DResult| -> f64 {
        (r.xyz.iter().map(|p| p[2] * p[2]).sum::<f64>() / (r.xyz.len() as f64)).sqrt()
    };
    let (rms_iso, rms_dip) = (rms_z(&res_iso), rms_z(&res_dip));
    let z_span_dip = res_dip
        .xyz
        .iter()
        .map(|p| p[2])
        .fold(f64::NEG_INFINITY, f64::max)
        - res_dip
            .xyz
            .iter()
            .map(|p| p[2])
            .fold(f64::INFINITY, f64::min);

    println!(
        "# screening R={r_patch}: iso +90 rms|z|={rms_iso:.3} E_2d={:.4} E_stretch={:.4} E_bend={:.4} vs dip rms|z|={rms_dip:.3} E_2d={:.4} E_stretch={:.4} E_bend={:.4} (z_span={z_span_dip:.3})",
        res_iso.planar_2d_stretch_energy,
        res_iso.stretch_energy,
        res_iso.bend_energy,
        res_dip.planar_2d_stretch_energy,
        res_dip.stretch_energy,
        res_dip.bend_energy,
    );

    assert!(
        rms_dip < 0.5 * rms_iso,
        "dipole rms|z| ({rms_dip}) must be < 0.5 * isolated +90 ({rms_iso})"
    );
    assert!(
        res_dip.planar_2d_stretch_energy < 0.15 * res_iso.planar_2d_stretch_energy,
        "dipole planar stretch ({:.5}) must be << isolated +90 ({:.5})",
        res_dip.planar_2d_stretch_energy,
        res_iso.planar_2d_stretch_energy
    );
    let tot_iso = res_iso.stretch_energy + res_iso.bend_energy;
    let tot_dip = res_dip.stretch_energy + res_dip.bend_energy;
    assert!(
        tot_dip < tot_iso,
        "dipole total 3D energy ({tot_dip:.5}) must be < isolated +90 ({tot_iso:.5})"
    );
    assert!(
        res_dip.bend_energy < res_iso.bend_energy,
        "dipole bending energy ({:.5}) must be < isolated +90 ({:.5})",
        res_dip.bend_energy,
        res_iso.bend_energy
    );
    assert!(
        res_dip.stretch_energy < 0.25 * res_dip.planar_2d_stretch_energy,
        "dipole 3D buckling must relieve >75% of its 2D stretch ({:.5} vs {:.5})",
        res_dip.stretch_energy,
        res_dip.planar_2d_stretch_energy
    );
    assert!(
        z_span_dip < 4.0 * res_dip.mean_bond_len,
        "dipole buckle stays bounded"
    );
}

#[test]
fn analytic_gradient_matches_finite_differences_and_mobius_wall_is_safe() {
    let (grp, cell, frame) = named_cell("kagome");
    let cat = DefectCatalogue::new(&grp, &cell, frame);
    let d0 = &cat.disclinations[0];
    let crystal = build_point_defects(
        &cell,
        &[DefectSpec::disclination(d0.angle, d0.point)],
        None,
        None,
    )
    .unwrap();
    let (faces, _, _) = extract_interior_faces(&crystal, &crystal.pos, true);
    let mut springs = Vec::new();
    for &(u, v) in &crystal.edges {
        springs.push(Spring {
            u,
            v,
            l0: 0.5,
            weight: 1.0,
            is_primary_bond: true,
        });
    }
    add_bulk_tile_chord_springs(&crystal, &faces, true, &mut springs);
    let hinges = build_hinges(&faces);

    let n = crystal.pos.len();
    let mut x = vec![0.0; 3 * n];
    for i in 0..n {
        x[3 * i] = crystal.pos[i][0] + 0.03 * det_hash_unit(i as u64 + 1);
        x[3 * i + 1] = crystal.pos[i][1] + 0.03 * det_hash_unit(i as u64 + 2);
        x[3 * i + 2] = 0.25 * det_hash_unit(i as u64 + 3);
    }
    let mut g = vec![0.0; 3 * n];
    let mut c_buf = vec![[0.0; 3]; faces.len()];
    let mut cg_buf = vec![[0.0; 3]; faces.len()];
    let _ = eval_sheet_fg(
        &x,
        &mut g,
        &faces,
        &springs,
        &hinges,
        0.15,
        &mut c_buf,
        &mut cg_buf,
    );

    let eps = 1e-6;
    let mut g_tmp = vec![0.0; 3 * n];
    for k in (0..x.len()).step_by(7) {
        let orig = x[k];
        x[k] = orig + eps;
        let fp = eval_sheet_fg(
            &x,
            &mut g_tmp,
            &faces,
            &springs,
            &hinges,
            0.15,
            &mut c_buf,
            &mut cg_buf,
        );
        x[k] = orig - eps;
        let fm = eval_sheet_fg(
            &x,
            &mut g_tmp,
            &faces,
            &springs,
            &hinges,
            0.15,
            &mut c_buf,
            &mut cg_buf,
        );
        x[k] = orig;
        let fd = (fp - fm) / (2.0 * eps);
        assert!(
            (g[k] - fd).abs() < 1e-5,
            "coord {k}: analytic {} vs fd {fd}",
            g[k]
        );
    }

    // Non-orientable Möbius seam patch (`Wc1` on square): finite positions, zero NaNs, fast convergence.
    let (s_grp, s_cell, s_frame) = named_cell("square");
    let s_cat = DefectCatalogue::new(&s_grp, &s_cell, s_frame);
    let wc1 = s_cat.build_wall("Wc1", Some(4.0)).expect("square Wc1");
    assert!(wc1.is_non_orientable);
    let res_wc = relax_sheet_3d(&wc1, &s_cell, &Sheet3DOptions::default(), None);
    assert_eq!(res_wc.xyz.len(), wc1.pos.len());
    assert!(res_wc.xyz.iter().flatten().all(|v| v.is_finite()));
    assert!(res_wc.bend_energy.is_finite() && res_wc.stretch_energy.is_finite());
    assert!(
        res_wc
            .diagnostics
            .iter()
            .any(|s| s.contains("Möbius") && s.contains("not a physical"))
    );
}

#[path = "sheet3d_review_tests.rs"]
mod review;
