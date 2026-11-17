//! Stage 3 tests: Burgers circuits, basepoint shifting / braiding, Moebius
//! orientation-reversing loops, and two-tier single-defect equivalence.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;

use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::combine_api::equivalent_single;
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::defect::{auto_burgers_circuit, burgers_circuit, shift_circuit_basepoint};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::{Bond2D, Cell2D, Frame2D, Recipe};

#[derive(Deserialize)]
struct NamedFile { named: Vec<NamedEntry> }
#[derive(Deserialize)]
struct NamedEntry { key: String, recipe: Recipe, frame: Frame2D }

fn load_named(key: &str) -> (String, Cell2D, Frame2D) {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../papertilling/data/named.json");
    let nf: NamedFile = serde_json::from_str(&std::fs::read_to_string(p).expect("named.json")).expect("parse");
    let e = nf.named.into_iter().find(|x| x.key == key).expect("named lattice exists");
    let fam = LatticeFamily::from_recipe(&e.recipe, Some(e.frame)).expect("recipe builds");
    let cell = fam.cell();
    (fam.group, cell, e.frame)
}

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

fn vec2_close(a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
    (a[0] - b[0]).hypot(a[1] - b[1]) <= tol
}

#[test]
fn square_pair_and_isolated_loops() {
    let sq = square_cell();
    let specs = [
        DefectSpec::disclination(90.0, [0.0, 0.0]),
        DefectSpec::disclination(-90.0, [0.5, 0.5]),
    ];
    let crystal = build_point_defects(&sq, &specs, None, None).unwrap();

    let whole = auto_burgers_circuit(&crystal, Some(("p4m", &sq, Frame2D::default())), &[], &[], 2, &specs).unwrap();
    assert!(whole.orientation_preserving);
    assert!((whole.frank_angle_deg.unwrap()).abs() < 1e-6);
    assert!((whole.burgers_norm - 1.0).abs() < 1e-6);
    assert_eq!(whole.charge.as_ref().unwrap().kind, "dislocation");

    let plus90 = auto_burgers_circuit(&crystal, Some(("p4m", &sq, Frame2D::default())), &[[0.0, 0.0]], &[[0.5, 0.5]], 2, &specs).unwrap();
    assert!((plus90.frank_angle_deg.unwrap() - 90.0).abs() < 1e-6);
    assert_eq!(plus90.charge.as_ref().unwrap().kind, "disclination");

    let minus90 = auto_burgers_circuit(&crystal, Some(("p4m", &sq, Frame2D::default())), &[[0.5, 0.5]], &[[0.0, 0.0]], 2, &specs).unwrap();
    assert!((minus90.frank_angle_deg.unwrap() + 90.0).abs() < 1e-6);
    assert_eq!(minus90.charge.as_ref().unwrap().kind, "disclination");
}

#[test]
fn basepoint_shift_and_braiding_in_chiral_triangular_and_square() {
    let sq = square_cell();
    let specs = [
        DefectSpec::disclination(90.0, [0.0, 0.0]),
        DefectSpec::disclination(-90.0, [0.5, 0.5]),
    ];
    let crystal = build_point_defects(&sq, &specs, None, None).unwrap();
    let base = auto_burgers_circuit(&crystal, Some(("p4m", &sq, Frame2D::default())), &[], &[], 2, &specs).unwrap();
    let mut rotated_vecs = BTreeMap::new();
    for (idx, &v) in base.path[..base.path.len() - 1].iter().enumerate() {
        let shifted = shift_circuit_basepoint(&base.path, v).unwrap();
        let res = burgers_circuit(&crystal, &shifted, Some(("p4m", &sq, Frame2D::default()))).unwrap();
        assert!((res.burgers_norm - base.burgers_norm).abs() < 1e-6);
        assert!((res.frank_angle_deg.unwrap() - base.frank_angle_deg.unwrap()).abs() < 1e-6);
        let ang_key = format!("{:.0}", res.frames[0][0][0]); // distinguish starting vertex
        let _ = ang_key;
        let bvec = res.closure_vector;
        rotated_vecs.entry((bvec[0].round() as i32, bvec[1].round() as i32)).or_insert(idx);
    }
    // Across a +90/-90 pair on the square lattice, shifting basepoint across one cut rotates b by 90 deg.
    assert!(rotated_vecs.len() >= 2, "expected distinct transported closure orientations, got {rotated_vecs:?}");

    // Order-3 disclination pair (+120 / -120) on chiral_triangular exhibits 120-deg braiding R(120) * b.
    let (grp_ct, cell_ct, f_ct) = load_named("chiral_triangular");
    let cat_ct = DefectCatalogue::new(&grp_ct, &cell_ct, f_ct);
    let mut cands = Vec::new();
    for cc in &cat_ct.centre_classes {
        if cc.order == 3 || cc.order == 6 {
            cands.push(cc.point);
        }
    }
    let p_a = cands[0];
    let p_b = [p_a[0] + cell_ct.a1[0], p_a[1] + cell_ct.a1[1]];
    let pair_ct = [DefectSpec::disclination(120.0, p_a), DefectSpec::disclination(-120.0, p_b)];
    let cry_ct = build_point_defects(&cell_ct, &pair_ct, None, None).unwrap();
    let cir_ct = auto_burgers_circuit(&cry_ct, Some((&grp_ct, &cell_ct, f_ct)), &[], &[], 2, &pair_ct).unwrap();
    assert!((cir_ct.frank_angle_deg.unwrap()).abs() < 1e-6);
    assert!(cir_ct.burgers_norm > 0.5);
    assert_eq!(cir_ct.charge.as_ref().unwrap().kind, "dislocation");
    let mut seen_dirs = Vec::new();
    for &v in &cir_ct.path[..cir_ct.path.len() - 1] {
        let s = shift_circuit_basepoint(&cir_ct.path, v).unwrap();
        let r = burgers_circuit(&cry_ct, &s, None).unwrap();
        assert!((r.burgers_norm - cir_ct.burgers_norm).abs() < 1e-5);
        if !seen_dirs.iter().any(|&q| vec2_close(q, r.closure_vector, 1e-4)) {
            seen_dirs.push(r.closure_vector);
        }
    }
    assert!(seen_dirs.len() >= 2, "expected rotated closure vectors under basepoint shift");

    // 180-deg +180/-180 square dipole + triple configuration target/exclude separation.
    let pair180 = [DefectSpec::disclination(180.0, [0.0, 0.0]), DefectSpec::disclination(-180.0, [1.5, 0.5])];
    let cry180 = build_point_defects(&sq, &pair180, None, None).unwrap();
    let c180 = auto_burgers_circuit(&cry180, Some(("p4m", &sq, Frame2D::default())), &[], &[], 2, &pair180).unwrap();
    assert!((c180.frank_angle_deg.unwrap()).abs() < 1e-6);
    assert!((c180.burgers_norm - 10.0_f64.sqrt()).abs() < 1e-5);

    let triple = [
        DefectSpec::disclination(90.0, [0.0, 0.0]),
        DefectSpec::disclination(-90.0, [0.5, 0.5]),
        DefectSpec::dislocation([1.0, 0.0], Some([2.5, -1.5])),
    ];
    let c_tri = build_point_defects(&sq, &triple, None, None).unwrap();
    let only_disloc = auto_burgers_circuit(&c_tri, Some(("p4m", &sq, Frame2D::default())), &[[2.5, -1.5]], &[[0.0, 0.0], [0.5, 0.5]], 2, &triple).unwrap();
    assert!((only_disloc.frank_angle_deg.unwrap()).abs() < 1e-6);
    assert!((only_disloc.burgers_norm - 1.0).abs() < 1e-5);
}

#[test]
fn mobius_seam_circuit_is_orientation_reversing() {
    let (grp, cell, frame) = load_named("brick_pg");
    let cat = DefectCatalogue::new(&grp, &cell, frame);
    let wc = cat.build_wall("Wc1", None).expect("Wc1 builds");
    assert!(wc.is_non_orientable);
    let cir = auto_burgers_circuit(&wc, Some((&grp, &cell, frame)), &[], &[], 2, &[]).unwrap();
    assert!(!cir.orientation_preserving);
    assert!(cir.frank_angle_deg.is_none());
    assert!(cir.det_signs.contains(&-1));
    assert_eq!(cir.charge.as_ref().unwrap().kind, "non_orientable");
}

#[test]
fn two_tier_equivalence_volterra_vs_completion_vs_exact_vs_none() {
    let sq = square_cell();
    let pair_sq = [DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.5, 0.5])];
    let cry_sq = build_point_defects(&sq, &pair_sq, None, None).unwrap();
    let rep_sq = equivalent_single("p4m", &sq, Frame2D::default(), &pair_sq, &cry_sq, 2, 2, false).unwrap();
    assert!(rep_sq.equivalent);
    assert_eq!(rep_sq.match_tier, "volterra");
    assert!(!rep_sq.volterra_matches.is_empty());
    assert!(rep_sq.completion_matches.is_empty());

    let (grp_hc, hc, frame_hc) = load_named("honeycomb");
    let s3 = 3.0_f64.sqrt();
    let pair_hc = [
        DefectSpec::disclination(60.0, [0.5 * s3, 0.5]),
        DefectSpec::disclination(-60.0, [0.0, -1.0]),
    ];
    let cry_hc = build_point_defects(&hc, &pair_hc, None, None).unwrap();
    let rep_hc = equivalent_single(&grp_hc, &hc, frame_hc, &pair_hc, &cry_hc, 2, 2, false).unwrap();
    assert!(rep_hc.equivalent);
    assert_eq!(rep_hc.match_tier, "completion");
    assert!(rep_hc.volterra_matches.is_empty());
    assert!(!rep_hc.completion_matches.is_empty());

    let disc = [DefectSpec::disclination(90.0, [0.0, 0.0])];
    let cry_d = build_point_defects(&sq, &disc, None, None).unwrap();
    let rep_d = equivalent_single("p4m", &sq, Frame2D::default(), &disc, &cry_d, 2, 1, false).unwrap();
    assert!(rep_d.equivalent && rep_d.match_tier == "exact");

    // Two-disclination dipole on square yielding b=[2,0] is scanned against b=[2,0]; a +90/+90 net +180
    // off-order / multi-site separation does not match any single defect -> match_tier == "none".
    let double90 = [DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(90.0, [2.0, 0.0])];
    let cry_2 = build_point_defects(&sq, &double90, None, None).unwrap();
    let rep_2 = equivalent_single("p4m", &sq, Frame2D::default(), &double90, &cry_2, 2, 1, false).unwrap();
    assert!(!rep_2.equivalent && rep_2.match_tier == "none");
}
