//! Integration tests for 360-degree Volterra dislocation insertion surgery and same-point defect superposition.

use papertilling_core::defect::combine_api::{
    classify_crystal_charge, equivalent_single, inspect_crystal_cores, same_charge,
};
use papertilling_core::defect::core_graph::{LatticeCoreInfo, core_graph, isomorphic_cores};
use papertilling_core::defect::surgery::{
    DefectSpec, build_point_defects, build_point_defects_with_cuts,
};
use papertilling_core::plane::catalogue::parse_named_str;

const NAMED_JSON: &str = include_str!("../../../papertilling/data/named.json");

#[test]
fn insertion_matches_removal_across_named_lattices() {
    let named = parse_named_str(NAMED_JSON).expect("parse named.json");
    for key in ["square", "honeycomb", "kagome", "shastry_sutherland"] {
        let item = named.get(key).expect("named lattice present");
        let cell = &item.cell;
        let a1 = cell.a1;
        let a2 = cell.a2;
        let at = [0.371 * a1[0] + 0.413 * a2[0], 0.371 * a1[1] + 0.413 * a2[1]];

        let rem = DefectSpec::Dislocation { b: a1, at: Some(at), cut: Some([-a1[1], a1[0]]) };
        let ins = DefectSpec::Dislocation { b: a1, at: Some(at), cut: Some([a1[1], -a1[0]]) };
        let (c_rem, cuts_rem, _) = build_point_defects_with_cuts(cell, &[rem], None, None).expect("removal builds");
        let (c_ins, cuts_ins, _) = build_point_defects_with_cuts(cell, &[ins], None, None).expect("insertion builds");

        assert_eq!(cuts_rem.len(), 1);
        assert!(!cuts_rem[0].is_insertion);
        assert_eq!(cuts_ins.len(), 1);
        assert!(cuts_ins[0].is_insertion);

        let q_rem = classify_crystal_charge(&item.group, cell, item.frame, &c_rem).expect("charge rem");
        let q_ins = classify_crystal_charge(&item.group, cell, item.frame, &c_ins).expect("charge ins");
        assert!(same_charge(&item.group, cell, &q_ins, &q_rem, false), "{key}: charge mismatch");
        let b_ins = q_ins.b.expect("dislocation b");
        assert!((b_ins[0] - a1[0]).abs() < 1e-6 && (b_ins[1] - a1[1]).abs() < 1e-6);

        let lat_info = LatticeCoreInfo::new(cell);
        let g_rem = core_graph(&c_rem, &lat_info, 2).expect("core rem");
        let g_ins = core_graph(&c_ins, &lat_info, 2).expect("core ins");
        assert!(isomorphic_cores(&g_ins, &g_rem), "{key}: core graph mismatch");

        let parallel = DefectSpec::Dislocation { b: a1, at: Some(at), cut: Some(a1) };
        let err = build_point_defects(cell, &[parallel], None, None).unwrap_err();
        assert!(err.contains("parallel"), "{key}: unexpected parallel cut error: {err}");
    }
}

#[test]
fn symmetric_glide_and_climb_dislocation_dipoles() {
    let named = parse_named_str(NAMED_JSON).expect("parse named.json");
    let sq = named.get("square").expect("square");

    for specs in [
        [
            DefectSpec::dislocation([1.0, 0.0], Some([0.37, 0.41])),
            DefectSpec::dislocation([-1.0, 0.0], Some([3.37, 0.41])),
        ],
        [
            DefectSpec::dislocation([-1.0, 0.0], Some([1.37, 0.41])),
            DefectSpec::dislocation([1.0, 0.0], Some([2.37, 0.41])),
        ],
        [
            DefectSpec::dislocation([1.0, 0.0], Some([0.37, 0.41])),
            DefectSpec::dislocation([-1.0, 0.0], Some([1.37, 2.41])),
        ],
        [
            DefectSpec::dislocation([-1.0, 0.0], Some([1.37, 0.41])),
            DefectSpec::dislocation([1.0, 0.0], Some([0.37, 2.41])),
        ],
    ] {
        let crystal = build_point_defects(&sq.cell, &specs, None, None).expect("dipole builds");
        let q = classify_crystal_charge(&sq.group, &sq.cell, sq.frame, &crystal).expect("charge");
        assert_eq!(q.kind, "vacuum");
        let rings: Vec<Option<usize>> = inspect_crystal_cores(&crystal).into_iter().map(|c| c.ring).collect();
        assert_eq!(rings, vec![Some(5), Some(5)]);
    }
}

#[test]
fn coincident_superposition_and_refusals() {
    let named = parse_named_str(NAMED_JSON).expect("parse named.json");
    let sq = named.get("square").expect("square");

    let (vac_disc, cuts_v, coal_v) = build_point_defects_with_cuts(
        &sq.cell,
        &[DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.0, 0.0])],
        None,
        None,
    )
    .expect("+90/-90 cancels");
    assert_eq!(coal_v, 2);
    assert!(cuts_v.is_empty());
    assert_eq!(classify_crystal_charge(&sq.group, &sq.cell, sq.frame, &vac_disc).unwrap().kind, "vacuum");

    let (d180, cuts_180, coal_180) = build_point_defects_with_cuts(
        &sq.cell,
        &[DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(90.0, [0.0, 0.0])],
        None,
        None,
    )
    .expect("+90/+90 merges to +180");
    assert_eq!(coal_180, 1);
    assert_eq!(cuts_180.len(), 1);
    let q180 = classify_crystal_charge(&sq.group, &sq.cell, sq.frame, &d180).unwrap();
    assert_eq!(q180.kind, "disclination");
    assert!((q180.angle - 180.0).abs() < 1e-6);

    let (vac_disl, cuts_dv, coal_dv) = build_point_defects_with_cuts(
        &sq.cell,
        &[
            DefectSpec::dislocation([1.0, 0.0], Some([0.371, 0.413])),
            DefectSpec::dislocation([-1.0, 0.0], Some([0.371, 0.413])),
        ],
        None,
        None,
    )
    .expect("(1,0)+(-1,0) cancels");
    assert_eq!(coal_dv, 2);
    assert!(cuts_dv.is_empty());
    assert_eq!(classify_crystal_charge(&sq.group, &sq.cell, sq.frame, &vac_disl).unwrap().kind, "vacuum");

    let sum_specs = [
        DefectSpec::dislocation([1.0, 0.0], Some([0.371, 0.413])),
        DefectSpec::dislocation([0.0, 1.0], Some([0.371, 0.413])),
    ];
    let (diag, _, _) = build_point_defects_with_cuts(&sq.cell, &sum_specs, None, None).expect("(1,0)+(0,1)");
    let qdiag = classify_crystal_charge(&sq.group, &sq.cell, sq.frame, &diag).unwrap();
    assert_eq!(qdiag.kind, "dislocation");
    assert_eq!(qdiag.b, Some([1.0, 1.0]));
    assert!(equivalent_single(&sq.group, &sq.cell, sq.frame, &sum_specs, &diag, 2, 1, false).unwrap().equivalent);

    let neg360 = build_point_defects(
        &sq.cell,
        &[DefectSpec::disclination(-180.0, [0.0, 0.5]), DefectSpec::disclination(-180.0, [0.0, 0.5])],
        None,
        None,
    );
    assert!(neg360.unwrap_err().contains("-360"));

    let (mixed_c, _, coal_m) = build_point_defects_with_cuts(
        &sq.cell,
        &[DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::dislocation([1.0, 0.0], Some([0.0, 0.0]))],
        None,
        None,
    )
    .expect("mixed disc+disl at (0,0) builds without coalescing");
    assert_eq!(coal_m, 0);
    let qm = classify_crystal_charge(&sq.group, &sq.cell, sq.frame, &mixed_c).unwrap();
    assert_eq!(qm.kind, "disclination");
    assert!((qm.angle - 90.0).abs() < 1e-6);

    let conflict = build_point_defects(
        &sq.cell,
        &[
            DefectSpec::Disclination { angle_deg: 90.0, at: [0.0, 0.0], cut: Some([1.0, 0.0]) },
            DefectSpec::Disclination { angle_deg: 90.0, at: [0.0, 0.0], cut: Some([0.0, 1.0]) },
        ],
        None,
        None,
    );
    assert!(conflict.unwrap_err().contains("conflicting cut"));
}
