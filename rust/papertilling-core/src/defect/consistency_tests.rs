//! Mathematical consistency tests for Workstream A fixes:
//! A3 (±180° single-defect rule), perpendicular cut preference (Wang 4-dislocation loop),
//! build-order independence, symmetric dislocation removal, and symmetric disclination unrolling.

use std::collections::BTreeMap;

use crate::defect::charge::frame_faces;
use crate::defect::combinations::build_combinations_table;
use crate::defect::combinations_types::{CombinationMode, CombinationsOptions, SingleMatchStatus};
use crate::defect::combine_api::classify_crystal_charge;
use crate::defect::core_graph::outer_face_index;
use crate::defect::surgery::{DefectSpec, build_point_defects, build_point_defects_with_cuts};
use crate::types::{Bond2D, BuiltCrystal, Cell2D, Frame2D};

fn square_cell() -> Cell2D {
    Cell2D {
        a1: [1.0, 0.0],
        a2: [0.0, 1.0],
        basis: vec![[0.0, 0.0]],
        bonds: vec![
            Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 },
            Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 },
        ],
    }
}

/// Mean circular bulk bond-angle rotation (in degrees) of bonds whose both endpoints lie
/// at distance >= `r_min` from the origin, compared against their reference lattice vector `e.vec`.
fn mean_bulk_bond_rotation_deg(crystal: &BuiltCrystal, r_min: f64) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for (u, adj_u) in crystal.adj.iter().enumerate() {
        let pu = crystal.pos[u];
        if pu[0].hypot(pu[1]) < r_min {
            continue;
        }
        for e in adj_u {
            if e.to <= u || e.seam.is_some() {
                continue;
            }
            let pv = crystal.pos[e.to];
            if pv[0].hypot(pv[1]) < r_min {
                continue;
            }
            let d = [pv[0] - pu[0], pv[1] - pu[1]];
            let v = e.vec;
            // Angle from reference bond vector `v` to drawn bond vector `d`, folded mod 90 deg?
            // No: un-folded signed angle of d relative to v on non-seam bulk bonds.
            let len_d = d[0].hypot(d[1]);
            let len_v = v[0].hypot(v[1]);
            if len_d < 1e-9 || len_v < 1e-9 {
                continue;
            }
            let sin_a = (v[0] * d[1] - v[1] * d[0]) / (len_v * len_d);
            let cos_a = (v[0] * d[0] + v[1] * d[1]) / (len_v * len_d);
            s += sin_a;
            c += cos_a;
        }
    }
    s.atan2(c).to_degrees()
}

/// Interior face-size histogram (excluding the outer boundary face).
fn interior_face_counts(crystal: &BuiltCrystal) -> BTreeMap<usize, usize> {
    let faces = frame_faces(crystal);
    let k_out = outer_face_index(&faces);
    let mut counts = BTreeMap::new();
    for (k, f) in faces.iter().enumerate() {
        if k != k_out {
            *counts.entry(f.len()).or_insert(0) += 1;
        }
    }
    counts
}

#[test]
fn disclination_270_charge_has_no_single_defect() {
    let cell = square_cell();
    // D(1a, +180) + D(1b, +90) on square lattice has unreduced Frank angle +270 deg.
    // By Author Decision A3 (single elementary disclinations satisfy |angle| <= 180 deg),
    // the total charge keeps kind="disclination" and angle=270.0, with single=None.
    let specs = [
        DefectSpec::disclination(180.0, [0.0, 0.0]),
        DefectSpec::disclination(90.0, [0.5, 0.5]),
    ];
    let crystal = build_point_defects(&cell, &specs, None, None).expect("build +180/+90 pair");
    let q = classify_crystal_charge("p4m", &cell, Frame2D::default(), &crystal)
        .expect("classify +270 charge");
    assert_eq!(q.kind, "disclination");
    assert!((q.angle - 270.0).abs() < 1e-6);
    assert!(q.single.is_none(), "270 deg disclination must have single=None");

    let opts = CombinationsOptions {
        max_sep: 1.0,
        radius: 2,
        grid: 1,
        draw_thumbnails: false,
        include_single_gallery: false,
        bmax: 2.0,
    };
    let table = build_combinations_table(
        "p4m",
        &cell,
        Frame2D::default(),
        &CombinationMode::DisclinationPairs,
        &opts,
        None,
    )
    .expect("combinations table for square p4m");
    let groups_270: Vec<_> = table
        .charge_groups
        .iter()
        .filter(|cg| cg.charge.kind == "disclination" && (cg.charge.angle - 270.0).abs() < 1e-5)
        .collect();
    assert!(!groups_270.is_empty(), "expected +270 deg charge group in disclination pairs");
    for cg in groups_270 {
        assert!(!cg.has_single_defect, "+270 deg charge group must have has_single_defect=false");
        assert!(cg.charge.single.is_none());
        assert!(!cg.core_groups.is_empty());
        for core in &cg.core_groups {
            assert_eq!(core.status, SingleMatchStatus::NoSingleDefect);
        }
    }
}

#[test]
fn wang_four_inner_dislocations_produce_only_pentagons() {
    let cell = square_cell();
    // Wang's 4 inner dislocations at r = 3: b = ±x, ±y with strip midpoints at (±3, 0), (0, ±3).
    // With perpendicular cuts preferred once clear of other defects, each unit dislocation core
    // on the square lattice is a clean 5-gon (4 pentagons total, zero 7/9/11-gons).
    let specs = [
        DefectSpec::dislocation([1.0, 0.0], Some([-3.5, 0.0])),
        DefectSpec::dislocation([-1.0, 0.0], Some([3.5, 0.0])),
        DefectSpec::dislocation([0.0, 1.0], Some([0.0, 2.5])),
        DefectSpec::dislocation([0.0, -1.0], Some([0.0, -2.5])),
    ];
    let crystal = build_point_defects(&cell, &specs, None, None).expect("build Wang 4-dislocation loop");
    let counts = interior_face_counts(&crystal);
    let odd_faces: BTreeMap<usize, usize> =
        counts.into_iter().filter(|&(sides, _)| sides != 4).collect();
    assert_eq!(
        odd_faces,
        BTreeMap::from([(5, 4)]),
        "expected exactly four 5-gons and zero 7/9/11-gons, got {odd_faces:?}"
    );
}

#[test]
fn build_order_independence_face_then_vertex_dipole() {
    let cell = square_cell();
    // Both [V_{-90}, F_{+90}] and [F_{+90}, V_{-90}] at (0,0) and (0.5,0.5) must succeed directly
    // and carry the exact kinematic dipole Burgers length |b| = 2 sin(45 deg) * sqrt(0.5) = 1.
    let v_first = [
        DefectSpec::disclination(-90.0, [0.0, 0.0]),
        DefectSpec::disclination(90.0, [0.5, 0.5]),
    ];
    let f_first = [
        DefectSpec::disclination(90.0, [0.5, 0.5]),
        DefectSpec::disclination(-90.0, [0.0, 0.0]),
    ];
    let c_vf = build_point_defects(&cell, &v_first, None, None).expect("[V_-90, F_+90] builds");
    let (c_fv, cuts_fv, coalesced) =
        build_point_defects_with_cuts(&cell, &f_first, None, None).expect("[F_+90, V_-90] builds");
    assert_eq!(coalesced, 0);
    assert_eq!(cuts_fv.len(), 2);
    // Applied cuts must be mapped back to caller order: cuts_fv[0] at (0.5, 0.5), cuts_fv[1] at (0, 0).
    assert!((cuts_fv[0].origin[0] - 0.5).abs() < 1e-9 && (cuts_fv[0].origin[1] - 0.5).abs() < 1e-9);
    assert!(cuts_fv[1].origin[0].abs() < 1e-9 && cuts_fv[1].origin[1].abs() < 1e-9);

    for (label, crystal) in [("V then F", &c_vf), ("F then V", &c_fv)] {
        let q = classify_crystal_charge("p4m", &cell, Frame2D::default(), crystal)
            .expect("classify dipole");
        assert_eq!(q.kind, "dislocation", "{label}");
        let len = q.length.expect("dislocation length");
        assert!((len - 1.0).abs() < 1e-6, "{label}: expected |b|=1, got {len}");
    }
}

#[test]
fn single_removal_dislocation_has_zero_mean_bulk_shift() {
    let cell = square_cell();
    // Single dislocation with b = [1, 0] and default cut c0 = [0, 1] (cross(c, b) = -1 < 0 -> removal).
    // Symmetric displacement +(0.5 - frac) * b gives zero net bulk drift instead of -0.5 * b.
    let specs = [DefectSpec::Dislocation {
        b: [1.0, 0.0],
        at: Some([0.371, 0.413]),
        cut: Some([0.0, 1.0]),
    }];
    let crystal = build_point_defects(&cell, &specs, None, None).expect("build removal dislocation");
    let n = crystal.pos.len() as f64;
    let mean_dx: f64 = crystal.pos.iter().zip(&crystal.ref_pos).map(|(p, r)| p[0] - r[0]).sum::<f64>() / n;
    let mean_dy: f64 = crystal.pos.iter().zip(&crystal.ref_pos).map(|(p, r)| p[1] - r[1]).sum::<f64>() / n;
    let drift = mean_dx.hypot(mean_dy);
    assert!(
        drift < 0.15,
        "removal dislocation mean bulk shift ({mean_dx:.4}, {mean_dy:.4}), |shift|={drift:.4} >= 0.15"
    );
}

#[test]
fn square_disclination_dipole_has_small_bulk_rotation() {
    let cell = square_cell();
    let specs = [
        DefectSpec::disclination(90.0, [0.0, 0.0]),
        DefectSpec::disclination(-90.0, [0.5, 0.5]),
    ];
    let crystal = build_point_defects(&cell, &specs, None, None).expect("build square dipole");
    let rot_deg = mean_bulk_bond_rotation_deg(&crystal, 4.0);
    let c_pos = build_point_defects(&cell, &[DefectSpec::disclination(90.0, [0.0, 0.0])], None, None)
        .expect("build single +90");
    let c_neg = build_point_defects(&cell, &[DefectSpec::disclination(-90.0, [0.5, 0.5])], None, None)
        .expect("build single -90");
    let rot_pos = mean_bulk_bond_rotation_deg(&c_pos, 4.0);
    let rot_neg = mean_bulk_bond_rotation_deg(&c_neg, 4.0);
    assert!(rot_deg.abs() < 2.0, "dipole bulk bond-angle rotation {rot_deg:.3} deg exceeds 2 deg");
    assert!(rot_pos.abs() < 2.0, "single +90 bulk bond-angle rotation {rot_pos:.3} deg exceeds 2 deg");
    assert!(rot_neg.abs() < 2.0, "single -90 bulk bond-angle rotation {rot_neg:.3} deg exceeds 2 deg");
}
