//! Known-answer tests for the general 2D lattice defect cavity extractor (`fusion::cavity`).
//!
//! Covers overlapping cores refused by strict Volterra surgery (adjacent-corner F_{+90} V^2_{-90}),
//! vacuum multi-dislocation cancellation (Wang's 4 dislocations), non-square lattices (honeycomb p6m),
//! already-built crystals, and honest refusal of non-buildable far-field charges (+270 deg, +360 deg).

use crate::defect::surgery::{DefectSpec, build_point_defects};
use crate::types::{Bond2D, Cell2D, Frame2D};

use super::cavity::{enumerate_defect_cavity_reconnections, extract_defect_cavity};

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

fn honeycomb_cell() -> Cell2D {
    let s3 = 3.0_f64.sqrt();
    let a1 = [1.0, 0.0];
    let a2 = [0.5, 0.5 * s3];
    let b0 = [(a1[0] + a2[0]) / 3.0, (a1[1] + a2[1]) / 3.0];
    let b1 = [2.0 * b0[0], 2.0 * b0[1]];
    Cell2D {
        a1,
        a2,
        basis: vec![b0, b1],
        bonds: vec![
            Bond2D { i: 0, j: 1, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 0, j: 1, n1: -1, n2: 0, kind: 0 },
            Bond2D { i: 0, j: 1, n1: 0, n2: -1, kind: 0 },
        ],
    }
}

fn sorted(mut xs: Vec<usize>) -> Vec<usize> {
    xs.sort_unstable();
    xs
}

#[test]
fn adjacent_corner_f90_v2m90_extracts_d1b_m90_far_field_cavity_and_enumerates_cores() {
    let cell = square_cell();
    // Adjacent face centre (0.5, 0.5) with +90 deg and site (0, 0) with -180 deg (= V^2_{-90}):
    // Algebraic holonomy R((0.5,0.5), +90) * R((0,0), -180) = R((0.5, -0.5), -90), class 1b (-90 deg).
    let specs = [
        DefectSpec::disclination(90.0, [0.5, 0.5]),
        DefectSpec::disclination(-180.0, [0.0, 0.0]),
    ];
    let (ext, en) = enumerate_defect_cavity_reconnections(
        "p4m", &cell, Frame2D::default(), &specs, None, 0, None, 1, 256,
    )
    .expect("far-field D(1b, -90 deg) cavity extraction and enumeration must succeed");

    assert!(ext.algebraic_charge.contains("1b") && ext.algebraic_charge.contains("-90"), "{}", ext.algebraic_charge);
    assert!(ext.far_field_source.contains("far-field single defect"), "{}", ext.far_field_source);
    assert_eq!(ext.cavity_map.euler(), 2);
    assert_eq!(ext.reference_filled.euler(), 2);
    // Pentagon core + 5 edge-adjacent squares = 6 bounded faces, 15-gon boundary (D_5 symmetry order 10),
    // 0 interior vertices and 5 interior edges (the 5 edges of the central pentagon).
    assert_eq!(ext.cavity_cycle.len(), 15);
    assert_eq!(ext.orig_inner_verts, 0);
    assert_eq!(ext.orig_inner_edges, 5);
    assert_eq!(sorted(ext.reference_filled.census().face_sides), vec![4, 4, 4, 4, 4, 5]);
    assert_eq!(en.cavity_symmetry_order, 10);
    assert!(!en.cores.is_empty());

    // Every enumerated core must satisfy planar Euler = 2, Complete valence >= 3, bounded face sides >= 3.
    for core in &en.cores {
        assert_eq!(core.map.euler(), 2);
        assert!(core.census.valences.iter().all(|&(_, k)| k >= 3));
        assert!(core.census.face_sides.iter().all(|&s| s >= 3));
    }
    // Both the original pentagon core (V_inner = 0, faces {4,4,4,4,4,5}) and the 5-spoke central-vertex
    // star core (V_inner = 1, 5 edges -> Euler gives 5 faces, each a pentagon {5,5,5,5,5}) must be present!
    assert!(en.cores.iter().any(|c| c.inner_verts == 0 && sorted(c.census.face_sides.clone()) == vec![4, 4, 4, 4, 4, 5]));
    assert!(en.cores.iter().any(|c| c.inner_verts == 1 && sorted(c.census.face_sides.clone()) == vec![5, 5, 5, 5, 5]));
}

#[test]
fn noncommuting_cavity_specs_use_the_ordered_combine_charge() {
    let cell = square_cell();
    let specs = [
        DefectSpec::disclination(90.0, [0.0, 0.0]),
        DefectSpec::disclination(-90.0, [0.5, 0.5]),
        DefectSpec::dislocation([1.0, 0.0], Some([0.371, 0.413])),
    ];
    // The ordered rotation product translates by (-1, 0). The final dislocation contributes
    // another (-1, 0), so the far-field Burgers vector has length 2. Reversing the order would
    // give a translation by (-1, 1) and produce a different host.
    let ext = extract_defect_cavity("p4m", &cell, Frame2D::default(), &specs, None, 0)
        .expect("far-field dislocation of length two has a cavity");
    assert!(ext.algebraic_charge.contains("|b| = 2.0000"), "{}", ext.algebraic_charge);
    assert!(ext.title.contains("|b| = 2.0000"), "{}", ext.title);
    assert_eq!(
        Some(ext.algebraic_charge.clone()),
        crate::defect::combinations::algebraic_specs_charge_label("p4m", &cell, Frame2D::default(), &specs),
    );

    let crystal = build_point_defects(&cell, &specs, None, None).expect("Volterra configuration builds");
    let measured = crate::defect::charge::total_charge(&crystal).unwrap();
    // The outer walk visits the cuts in a different order, giving the diagonal translation.
    let [bx, by] = measured.burgers.unwrap();
    assert!((bx.abs() - 1.0).abs() < 1e-6 && (by.abs() - 1.0).abs() < 1e-6);
    let built = extract_defect_cavity("p4m", &cell, Frame2D::default(), &specs, Some(&crystal), 0)
        .expect("built configuration has a cavity");
    assert!(built.title.contains("|b| = 1.4142"), "{}", built.title);
    assert_eq!(built.algebraic_charge, ext.algebraic_charge);
    assert!(built.note.contains("different charges"), "{}", built.note);
}

#[test]
fn wang_four_dislocations_at_origin_extracts_vacuum_cavity_with_regular_square_filling() {
    let cell = square_cell();
    let specs = [
        DefectSpec::dislocation([1.0, 0.0], Some([0.0, 0.0])),
        DefectSpec::dislocation([-1.0, 0.0], Some([0.0, 0.0])),
        DefectSpec::dislocation([0.0, 1.0], Some([0.0, 0.0])),
        DefectSpec::dislocation([0.0, -1.0], Some([0.0, 0.0])),
    ];
    let (ext, en) = enumerate_defect_cavity_reconnections(
        "p4m", &cell, Frame2D::default(), &specs, None, 0, None, 1, 256,
    )
    .expect("vacuum cavity extraction and enumeration must succeed");

    assert!(ext.far_field_source.contains("vacuum"), "{}", ext.far_field_source);
    assert_eq!(ext.cavity_map.euler(), 2);
    assert_eq!(ext.reference_filled.euler(), 2);
    // 2x2 block of 4 squares around (0,0): 8-gon cavity (D_4 order 8), 1 interior vertex, 4 interior edges.
    assert_eq!(ext.cavity_cycle.len(), 8);
    assert_eq!(ext.orig_inner_verts, 1);
    assert_eq!(ext.orig_inner_edges, 4);
    assert_eq!(en.cavity_symmetry_order, 8);
    assert_eq!(en.regular_count, 1, "unique regular 2x2 square filling must be found");
    assert!(en.cores[0].is_regular_square);
    assert_eq!(en.cores[0].inner_verts, 1);
    assert_eq!(sorted(en.cores[0].census.face_sides.clone()), vec![4, 4, 4, 4]);
    for core in &en.cores {
        assert_eq!(core.map.euler(), 2);
        assert!(core.census.valences.iter().all(|&(_, k)| k >= 3));
        assert!(core.census.face_sides.iter().all(|&s| s >= 3));
    }
}

#[test]
fn honeycomb_p6m_dislocation_and_vacuum_cavities_work_beyond_square() {
    let cell = honeycomb_cell();

    // 1. Vacuum pair (b = +a1 and -a1) -> 3 hexagons around nearest site: 12-gon (D_3 order 6), 1 inner vert, 3 inner edges.
    let vac_specs = [
        DefectSpec::dislocation([1.0, 0.0], Some([0.0, 0.0])),
        DefectSpec::dislocation([-1.0, 0.0], Some([0.0, 0.0])),
    ];
    let (ext_v, en_v) = enumerate_defect_cavity_reconnections(
        "p6m", &cell, Frame2D::default(), &vac_specs, None, 0, None, 1, 256,
    )
    .expect("honeycomb vacuum cavity must succeed");
    assert_eq!(ext_v.cavity_map.euler(), 2);
    assert_eq!(ext_v.reference_filled.euler(), 2);
    assert_eq!(ext_v.cavity_cycle.len(), 12);
    assert_eq!(ext_v.orig_inner_verts, 1);
    assert_eq!(ext_v.orig_inner_edges, 3);
    assert_eq!(en_v.cavity_symmetry_order, 6);
    assert!(en_v.cores.iter().any(|c| c.inner_verts == 1 && sorted(c.census.face_sides.clone()) == vec![6, 6, 6]));
    for core in &en_v.cores {
        assert_eq!(core.map.euler(), 2);
        assert!(core.census.valences.iter().all(|&(_, k)| k >= 3));
        assert!(core.census.face_sides.iter().all(|&s| s >= 3));
    }

    // 2. Single dislocation b = a1 on honeycomb: cavity is a 10-gon with Euler-consistent reference filling
    //    (F_bounded = 1 + E_in - V_in) and valid non-empty lattice reconnections.
    let disloc_specs = [DefectSpec::dislocation([1.0, 0.0], Some([0.0, 0.0]))];
    let (ext_d, en_d) = enumerate_defect_cavity_reconnections(
        "p6m", &cell, Frame2D::default(), &disloc_specs, None, 0, None, 1, 256,
    )
    .expect("honeycomb dislocation cavity must succeed");
    assert_eq!(ext_d.cavity_map.euler(), 2);
    assert_eq!(ext_d.reference_filled.euler(), 2);
    assert_eq!(ext_d.cavity_cycle.len(), 10);
    assert_eq!(
        ext_d.reference_filled.bounded_faces().len(),
        1 + ext_d.orig_inner_edges - ext_d.orig_inner_verts
    );
    assert!(!en_d.cores.is_empty());
    for core in &en_d.cores {
        assert_eq!(core.map.euler(), 2);
        assert!(core.census.valences.iter().all(|&(_, k)| k >= 3));
        assert!(core.census.face_sides.iter().all(|&s| s >= 3));
    }
}

#[test]
fn existing_built_crystal_uses_built_configuration_directly() {
    let cell = square_cell();
    let specs = [DefectSpec::dislocation([1.0, 0.0], Some([0.371, 0.413]))];
    let built = build_point_defects(&cell, &specs, None, None).expect("single square dislocation builds");
    let (ext, en) = enumerate_defect_cavity_reconnections(
        "p4m", &cell, Frame2D::default(), &specs, Some(&built), 0, None, 1, 256,
    )
    .expect("cavity extraction from built crystal must succeed");
    assert_eq!(ext.far_field_source, "built configuration");
    assert_eq!(ext.cavity_map.euler(), 2);
    assert_eq!(ext.reference_filled.euler(), 2);
    assert!(!en.cores.is_empty());
    for core in &en.cores {
        assert_eq!(core.map.euler(), 2);
        assert!(core.census.valences.iter().all(|&(_, k)| k >= 3));
        assert!(core.census.face_sides.iter().all(|&s| s >= 3));
    }
}

#[test]
fn non_buildable_far_field_charges_are_honestly_refused() {
    let cell = square_cell();
    // +270 deg total Frank angle (|angle| > 180 deg): no single-defect far-field boundary.
    let specs_270 = [
        DefectSpec::disclination(90.0, [0.5, 0.5]),
        DefectSpec::disclination(90.0, [1.5, 0.5]),
        DefectSpec::disclination(90.0, [0.5, 1.5]),
    ];
    let err_270 = extract_defect_cavity("p4m", &cell, Frame2D::default(), &specs_270, None, 0)
        .expect_err("+270 deg total charge must be refused when no crystal is provided");
    assert!(err_270.contains("cannot construct a single-defect far-field boundary"), "{err_270}");

    // +360 deg whole turn: also refused cleanly.
    let specs_360 = [
        DefectSpec::disclination(180.0, [0.5, 0.5]),
        DefectSpec::disclination(180.0, [1.5, 0.5]),
    ];
    let err_360 = extract_defect_cavity("p4m", &cell, Frame2D::default(), &specs_360, None, 0)
        .expect_err("+360 deg whole turn must be refused when no crystal is provided");
    assert!(err_360.contains("cannot construct a single-defect far-field boundary"), "{err_360}");
}

#[test]
fn vacuum_far_field_cavity_rejects_unrepresentable_lattice_coordinates() {
    let at = Some([1e30, 0.0]);
    let specs = [DefectSpec::dislocation([1.0, 0.0], at), DefectSpec::dislocation([-1.0, 0.0], at)];
    let err = extract_defect_cavity("p4m", &square_cell(), Frame2D::default(), &specs, None, 0).unwrap_err();
    assert!(err.contains("patch extent exceeds supported finite lattice indices"), "{err}");
}
