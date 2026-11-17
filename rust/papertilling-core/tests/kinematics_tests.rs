//! Integration tests for particle-conserving defect kinematics in `papertilling_core`.

use papertilling_core::defect::combine_api::{
    classify_crystal_charge, inspect_crystal_cores, same_charge,
};
use papertilling_core::defect::core_graph::LatticeCoreInfo;
use papertilling_core::defect::kinematics::{
    absorb_dislocation, find_absorption_preview, glide_line_spacing,
    parallel_transport_glided_dislocation, project_glide_at, step_disclination_emit,
};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::plane::catalogue::parse_named_str;

const NAMED_JSON: &str = include_str!("../../../papertilling/data/named.json");

fn norm2(a: [f64; 2]) -> f64 {
    a[0].hypot(a[1])
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn first_centre_for_angle(lat: &LatticeCoreInfo, cell_item: &papertilling_core::plane::catalogue::NamedLattice, ang: f64) -> [f64; 2] {
    let cat = papertilling_core::defect::DefectCatalogue::new(&cell_item.group, &cell_item.cell, cell_item.frame);
    let c = cat
        .centre_classes
        .iter()
        .find(|r| r.angles.iter().any(|&a| (a - ang).abs() < 1e-4))
        .expect("rotation centre for angle");
    let _ = lat;
    c.point
}

#[test]
fn glide_projection_and_spacing_follow_exact_geometry() {
    let named = parse_named_str(NAMED_JSON).expect("parse named.json");
    let sq = named.get("square").expect("square");
    let dp = glide_line_spacing(&sq.cell, [1.0, 0.0]);
    assert!((dp - 1.0).abs() < 1e-12);
    let projected = project_glide_at([1.0, 0.0], [0.37, 0.41], [4.2, -3.5]);
    assert!((projected[0] - 4.2).abs() < 1e-12 && (projected[1] - 0.41).abs() < 1e-12);

    let hc = named.get("honeycomb").expect("honeycomb");
    let dp_hc = glide_line_spacing(&hc.cell, hc.cell.a1);
    let area = (hc.cell.a1[0] * hc.cell.a2[1] - hc.cell.a1[1] * hc.cell.a2[0]).abs();
    let expected = area / norm2(hc.cell.a1);
    assert!((dp_hc - expected).abs() < 1e-10);
}

#[test]
fn disclination_step_emit_annihilation_and_180_wyckoff_hop() {
    let named = parse_named_str(NAMED_JSON).expect("parse named.json");
    for (key, ang) in [
        ("square", 90.0),
        ("honeycomb", 60.0),
        ("shastry_sutherland", 90.0),
        ("chiral_triangular", 120.0),
    ] {
        let item = named.get(key).expect("named lattice");
        let lat = LatticeCoreInfo::new(&item.cell);
        let p0 = first_centre_for_angle(&lat, item, ang);
        let p1 = [p0[0] + item.cell.a1[0], p0[1] + item.cell.a1[1]];
        let p2 = [p0[0] + 2.0 * item.cell.a1[0], p0[1] + 2.0 * item.cell.a1[1]];
        let specs0 = vec![
            DefectSpec::disclination(ang, p0),
            DefectSpec::disclination(-ang, p2),
        ];
        let q0 = classify_crystal_charge(
            &item.group,
            &item.cell,
            item.frame,
            &build_point_defects(&item.cell, &specs0, None, None).unwrap(),
        )
        .unwrap();

        let step1 = step_disclination_emit(&item.group, &item.cell, item.frame, &specs0, 1, p1).expect(key);
        assert!(step1.emitted.is_some());
        let q1 = classify_crystal_charge(
            &item.group,
            &item.cell,
            item.frame,
            &build_point_defects(&item.cell, &step1.specs, None, None).unwrap(),
        )
        .unwrap();
        assert!(same_charge(&item.group, &item.cell, &q0, &q1, false), "{key}: step1 charge mismatch");

        let step2 = step_disclination_emit(&item.group, &item.cell, item.frame, &step1.specs, 1, p0).expect(key);
        let c2 = build_point_defects(&item.cell, &step2.specs, None, None).unwrap();
        let q2 = classify_crystal_charge(&item.group, &item.cell, item.frame, &c2).unwrap();
        assert!(same_charge(&item.group, &item.cell, &q0, &q2, false), "{key}: step2 charge mismatch");
        assert!(inspect_crystal_cores(&c2).iter().all(|c| c.kind == "dislocation"));
    }

    let sq = named.get("square").expect("square");
    let hop180 = step_disclination_emit(
        &sq.group,
        &sq.cell,
        sq.frame,
        &[DefectSpec::disclination(180.0, [0.0, 0.0])],
        0,
        [0.5, 0.0],
    )
    .expect("+180 hop 1a -> 2c");
    match hop180.emitted.expect("emitted") {
        DefectSpec::Dislocation { b, .. } => assert!((norm2(b) - 1.0).abs() < 1e-6),
        other => panic!("expected dislocation, got {other:?}"),
    }
}

#[test]
fn absorb_roundtrip_preview_and_climb_refusal_and_parallel_transport() {
    let named = parse_named_str(NAMED_JSON).expect("parse named.json");
    for (key, ang) in [
        ("square", 90.0),
        ("honeycomb", 60.0),
        ("shastry_sutherland", 90.0),
        ("chiral_triangular", 120.0),
    ] {
        let item = named.get(key).expect("named lattice");
        let lat = LatticeCoreInfo::new(&item.cell);
        let p0 = first_centre_for_angle(&lat, item, ang);
        let p1 = [p0[0] + item.cell.a1[0], p0[1] + item.cell.a1[1]];
        let p2 = [p0[0] + 2.0 * item.cell.a1[0], p0[1] + 2.0 * item.cell.a1[1]];
        let specs0 = vec![
            DefectSpec::disclination(ang, p0),
            DefectSpec::disclination(-ang, p2),
        ];
        let q0 = classify_crystal_charge(
            &item.group,
            &item.cell,
            item.frame,
            &build_point_defects(&item.cell, &specs0, None, None).unwrap(),
        )
        .unwrap();
        let step1 = step_disclination_emit(&item.group, &item.cell, item.frame, &specs0, 1, p1).unwrap();
        let DefectSpec::Dislocation { b, at: Some(at_em), .. } = step1.specs[2] else {
            panic!("missing emitted dislocation");
        };
        let ub = [b[0] / norm2(b), b[1] / norm2(b)];
        let glided = [-1.0_f64, 1.0]
            .into_iter()
            .find_map(|sgn| {
                let at_far = [at_em[0] + sgn * 1.8 * lat.lmax * ub[0], at_em[1] + sgn * 1.8 * lat.lmax * ub[1]];
                let cand = vec![
                    step1.specs[0].clone(),
                    step1.specs[1].clone(),
                    DefectSpec::dislocation(b, Some(at_far)),
                ];
                let cr = build_point_defects(&item.cell, &cand, None, None).ok()?;
                let qg = classify_crystal_charge(&item.group, &item.cell, item.frame, &cr).ok()?;
                same_charge(&item.group, &item.cell, &q0, &qg, false).then_some(cand)
            })
            .expect(key);
        let (rec, c_rec) = absorb_dislocation(&item.group, &item.cell, item.frame, &glided, 2, 1).expect(key);
        assert_eq!(rec.len(), 2);
        assert!(norm2(sub2(c_rec, p2)) < 1e-5, "{key}: recovered {c_rec:?} vs {p2:?}");

        let dp = glide_line_spacing(&item.cell, b);
        let nb = [-ub[1], ub[0]];
        let climbed = vec![
            step1.specs[0].clone(),
            step1.specs[1].clone(),
            DefectSpec::dislocation(b, Some([at_em[0] + dp * nb[0], at_em[1] + dp * nb[1]])),
        ];
        let err = absorb_dislocation(&item.group, &item.cell, item.frame, &climbed, 2, 1).unwrap_err();
        assert!(err.contains("climb"), "{key}: unexpected error {err}");

        // Preview detection triggers only once glided near the disclination along the valid glide line.
        let near_at = project_glide_at(b, at_em, [p1[0] - 0.5 * b[0], p1[1] - 0.5 * b[1]]);
        let near_specs = vec![
            step1.specs[0].clone(),
            step1.specs[1].clone(),
            DefectSpec::dislocation(b, Some(near_at)),
        ];
        assert!(find_absorption_preview(&item.cell, &near_specs, 2, near_at).is_some());
    }

    // Parallel transport check on a plain glided dislocation:
    let sq = named.get("square").expect("square");
    let before = vec![DefectSpec::dislocation([1.0, 0.0], Some([0.37, 0.41]))];
    let after = vec![DefectSpec::dislocation([1.0, 0.0], Some([2.37, 0.41]))];
    let (res, rotated) =
        parallel_transport_glided_dislocation(&sq.group, &sq.cell, sq.frame, &before, &after, 0).unwrap();
    assert!(!rotated && res == after);
}
