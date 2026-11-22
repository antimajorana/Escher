//! Independent mathematical known-answer tests for Phase 3A exact space-group algebra,
//! Smith normal form abelianizations, improper sector classification, and lifted `G~+` fusion rules.

use std::collections::BTreeMap;

use crate::defect::algebraic_fusion::{
    AlgebraicFusionOutcome, SpaceGroupAlgebra, SpaceGroupAlgebraReport, analyze_space_group_fusion,
    specs_abelian_charge,
};
use crate::defect::algebraic_fusion_snf::{
    LatticeCoset, format_abelian_group, invariant_factors_from_snf, smith_normal_form,
};
use crate::defect::catalogue::DefectCatalogue;
use crate::defect::surgery::DefectSpec;
use crate::plane::catalogue::{NamedLattice, parse_catalogue_str, parse_named_str};
use crate::plane::family::LatticeFamily;
use crate::types::{Cell2D, Frame2D};

const NAMED_JSON: &str = include_str!("../../../../papertilling/data/named.json");
const POINTS_JSON: &str = include_str!("../../../../papertilling/data/points.json");

fn load_named(key: &str) -> NamedLattice {
    parse_named_str(NAMED_JSON)
        .expect("parse named.json")
        .get(key)
        .unwrap_or_else(|| panic!("missing named lattice {key}"))
        .clone()
}

fn find_outcome<'a>(report: &'a SpaceGroupAlgebraReport, a: &str, b: &str) -> &'a AlgebraicFusionOutcome {
    let row = report
        .fusion_rows
        .iter()
        .find(|r| (r.defect_a == a && r.defect_b == b) || (r.defect_a == b && r.defect_b == a))
        .unwrap_or_else(|| panic!("missing fusion row ({a}, {b}) in {}", report.group));
    assert_eq!(row.outcomes.len(), 1);
    &row.outcomes[0]
}

fn mat_mul_rect(a: &[Vec<i64>], b: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let (m, k, n) = (a.len(), a[0].len(), b[0].len());
    let mut out = vec![vec![0i64; n]; m];
    for i in 0..m {
        for j in 0..n {
            out[i][j] = (0..k).map(|t| a[i][t] * b[t][j]).sum();
        }
    }
    out
}

fn det_square_i128(mat: &[Vec<i64>]) -> i128 {
    let n = mat.len();
    let mut a: Vec<Vec<i128>> = mat.iter().map(|r| r.iter().map(|&x| i128::from(x)).collect()).collect();
    let (mut sign, mut prev_piv) = (1i128, 1i128);
    for k in 0..n {
        let Some(p) = (k..n).find(|&i| a[i][k] != 0) else { return 0; };
        if p != k { a.swap(k, p); sign = -sign; }
        let piv = a[k][k];
        for i in (k + 1)..n {
            for j in (k + 1)..n {
                a[i][j] = (a[i][j] * piv - a[i][k] * a[k][j]) / prev_piv;
            }
            a[i][k] = 0;
        }
        prev_piv = piv;
    }
    sign * a[n - 1][n - 1]
}

#[test]
fn snf_exact_unimodular_and_formatting() {
    // Classic textbook 3x3 integer matrix whose exact Smith normal form is diag(2, 6, 12).
    let m = vec![vec![2, 4, 4], vec![-6, 6, 12], vec![10, -4, -16]];
    let snf = smith_normal_form(&m);
    assert_eq!(snf.diag, vec![2, 6, 12]);
    assert_eq!(det_square_i128(&snf.u).abs(), 1, "U must be unimodular");
    assert_eq!(det_square_i128(&snf.v).abs(), 1, "V must be unimodular");
    let prod = mat_mul_rect(&mat_mul_rect(&snf.u, &m), &snf.v);
    assert_eq!(prod, vec![vec![2, 0, 0], vec![0, 6, 0], vec![0, 0, 12]]);

    // Rectangular matrix with free rank + torsion: Z^3 / span((2,0,0), (0,4,0)) = Z x Z/2Z x Z/4Z.
    let rect = vec![vec![2, 0], vec![0, 4], vec![0, 0]];
    let snf_r = smith_normal_form(&rect);
    let (fr, tor) = invariant_factors_from_snf(3, &snf_r.diag);
    assert_eq!((fr, tor), (1, vec![2, 4]));
    assert_eq!(format_abelian_group(0, &[]), "0");
    assert_eq!(format_abelian_group(2, &[]), "Z^2");
    assert_eq!(format_abelian_group(0, &[2, 4]), "Z/2Z x Z/4Z");
    assert_eq!(format_abelian_group(0, &[2, 2, 2]), "(Z/2Z)^3");
    assert_eq!(format_abelian_group(1, &[2]), "Z x Z/2Z");

    // Canonical even/odd index-2 sublattice checks.
    let even = LatticeCoset::new([0, 0], &[[1, 1], [1, -1]]);
    assert_eq!(even.index(), Some(2));
    assert!(even.contains([0, 0]) && even.contains([1, 1]) && !even.contains([1, 0]));
    assert!(even.describe().contains("b1+b2 = 0 mod 2"));
    let odd = LatticeCoset::new([1, 0], &[[1, 1], [1, -1]]);
    assert!(odd.contains([1, 0]) && !odd.contains([0, 0]) && !odd.contains([1, 1]));
    assert!(odd.describe().contains("b1+b2 = 1 mod 2"));
}

#[test]
fn test1_abelianizations_across_eight_benchmark_groups() {
    // Independent published abelianizations (free_rank, invariant-factor torsion, label)
    // for G^ab and (G+)^ab. Note: for p31m, Z/2Z x Z/3Z == Z/6Z in SNF invariant-factor form.
    type AbSpec<'a> = (usize, &'a [i64], &'a str);
    type BenchmarkCase<'a> = (&'a str, &'a str, bool, AbSpec<'a>, AbSpec<'a>);
    let cases: [BenchmarkCase<'_>; 8] = [
        ("straight_p1", "p1", true, (2, &[], "Z^2"), (2, &[], "Z^2")),
        ("square", "p4m", true, (0, &[2, 2, 2], "(Z/2Z)^3"), (0, &[2, 4], "Z/2Z x Z/4Z")),
        ("shastry_sutherland", "p4g", false, (0, &[2, 4], "Z/2Z x Z/4Z"), (0, &[2, 4], "Z/2Z x Z/4Z")),
        ("chiral_triangular", "p3", true, (0, &[3, 3], "(Z/3Z)^2"), (0, &[3, 3], "(Z/3Z)^2")),
        ("breathing_kagome", "p3m1", true, (0, &[2], "Z/2Z"), (0, &[3, 3], "(Z/3Z)^2")),
        ("fractal_triangular", "p31m", true, (0, &[6], "Z/6Z"), (0, &[3, 3], "(Z/3Z)^2")),
        ("chiral_kagome", "p6", true, (0, &[6], "Z/6Z"), (0, &[6], "Z/6Z")),
        ("honeycomb", "p6m", true, (0, &[2, 2], "(Z/2Z)^2"), (0, &[6], "Z/6Z")),
    ];

    for (key, expected_grp, expected_symm, (ff, ft, fl), (pf, pt, pl)) in cases {
        let lat = load_named(key);
        let rep = analyze_space_group_fusion(&lat.group, &lat.cell, lat.frame, 2.0)
            .unwrap_or_else(|e| panic!("{key}: {e}"));
        assert_eq!(rep.group, expected_grp, "{key} group");
        assert_eq!(rep.symmorphic, expected_symm, "{key} symmorphic");
        assert_eq!((rep.ab_full.free_rank, rep.ab_full.torsion.as_slice(), rep.ab_full.label.as_str()), (ff, ft, fl), "{key} G^ab");
        assert_eq!((rep.ab_proper.free_rank, rep.ab_proper.torsion.as_slice(), rep.ab_proper.label.as_str()), (pf, pt, pl), "{key} (G+)^ab");
        assert!(rep.provenance.starts_with("[PROVED]"));
    }

    // Verify p4g axial glide has exact order 4 in G^ab.
    let ss = load_named("shastry_sutherland");
    let cat = DefectCatalogue::with_bounds(&ss.group, &ss.cell, ss.frame, 2.0, 0.0);
    let alg = SpaceGroupAlgebra::from_catalogue(&cat).expect("p4g algebra");
    let axial_idx = (alg.proper_order..alg.matrices.len())
        .find(|&i| matches!(alg.matrices[i], [[1, 0], [0, -1]] | [[-1, 0], [0, 1]]))
        .expect("p4g axial glide op");
    assert_eq!(alg.full_abelian_order([0, 0], axial_idx), Some(4), "p4g axial glide must have exact order 4 in G^ab");

    // Verify order-independent abelian charge conservation on square p4m:
    // D(1a, +90) + D(1b, -90) == T1 (b = (1, 0)) in both (G+)^ab and G^ab.
    let sq = load_named("square");
    let d_a = DefectSpec::disclination(90.0, [0.0, 0.0]);
    let d_b = DefectSpec::disclination(-90.0, [0.5, 0.5]);
    let t1 = DefectSpec::dislocation([1.0, 0.0], None);
    let ab_1 = specs_abelian_charge(&sq.group, &sq.cell, sq.frame, &[d_a.clone(), d_b.clone()]).expect("charge 1");
    let ab_2 = specs_abelian_charge(&sq.group, &sq.cell, sq.frame, &[d_b, d_a]).expect("charge 2");
    let ab_t1 = specs_abelian_charge(&sq.group, &sq.cell, sq.frame, &[t1]).expect("charge t1");
    assert_eq!(ab_1, ab_2, "abelian charge must be strictly order-independent");
    assert_eq!(ab_1, ab_t1, "D(1a,+90) + D(1b,-90) must carry the exact same abelian charge as T1");
}

#[test]
fn test2_improper_sectors_p4m_vs_p4g() {
    let sq = load_named("square");
    let p4m = analyze_space_group_fusion(&sq.group, &sq.cell, sq.frame, 2.0).expect("p4m");
    assert_eq!(p4m.improper_sectors.len(), 2, "p4m has 2 improper sectors (axial & diagonal)");
    assert_eq!(p4m.improper_sectors[0].quotient_label, "Z x Z/2Z");
    assert_eq!(p4m.improper_sectors[1].quotient_label, "Z");
    for s in &p4m.improper_sectors {
        assert!(s.has_pure_mirror, "p4m sector {} must have a pure mirror", s.label);
        assert!(!s.fixed_point_free_centralizer, "p4m sector {} centralizer must have fixed points", s.label);
        assert_eq!(s.square_proper_channel, "trivial (0)");
    }

    let ss = load_named("shastry_sutherland");
    let p4g = analyze_space_group_fusion(&ss.group, &ss.cell, ss.frame, 2.0).expect("p4g");
    assert_eq!(p4g.improper_sectors.len(), 2, "p4g has 2 improper sectors (axial glide & diagonal mirror)");
    let axial = p4g.improper_sectors.iter()
        .find(|s| matches!(s.parallel_axis, [1, 0] | [0, 1]))
        .expect("p4g axial sector");
    let diag = p4g.improper_sectors.iter()
        .find(|s| matches!(s.parallel_axis, [1, 1] | [1, -1]))
        .expect("p4g diagonal sector");

    assert!(!axial.has_pure_mirror, "p4g axial sector is a pure glide (no pure mirror)");
    assert!(axial.fixed_point_free_centralizer, "p4g axial sector Z_P(R) acts fixed-point-freely on Q_R");
    assert_eq!(axial.quotient_label, "Z x Z/2Z");
    assert!(axial.square_proper_channel.contains("b1+b2 = 1 mod 2"), "got {}", axial.square_proper_channel);

    assert!(diag.has_pure_mirror, "p4g diagonal sector has a pure mirror");
    assert!(!diag.fixed_point_free_centralizer, "p4g diagonal sector Z_P(R) fixes the diagonal mirror");
    assert_eq!(diag.quotient_label, "Z");
    assert_eq!(diag.square_proper_channel, "trivial (0)");
}

#[test]
fn test3_square_p4m_lifted_fusion_rules() {
    let sq = load_named("square");
    let rep = analyze_space_group_fusion(&sq.group, &sq.cell, sq.frame, 2.0).expect("p4m fusion");

    // D(1a, +90) x D(1b, -90): index-2 odd coset, contains (1,0), excludes (0,0) & (1,1).
    let o_ab = find_outcome(&rep, "D(1a, +90)", "D(1b, -90)");
    assert_eq!(o_ab.kind, "dislocation");
    assert_eq!(o_ab.burgers_sublattice_index, Some(2));
    assert_eq!(o_ab.burgers_coset_contains_zero, Some(false));
    assert!(o_ab.contains_burgers([1, 0]) && !o_ab.contains_burgers([0, 0]) && !o_ab.contains_burgers([1, 1]));
    assert!(o_ab.shortest_burgers_frac.contains(&[1, 0]));

    // D(1a, +90) x D(1a, -90) and D(1b, +90) x D(1b, -90): index-2 even sublattice.
    for pair in [("D(1a, +90)", "D(1a, -90)"), ("D(1b, +90)", "D(1b, -90)")] {
        let o = find_outcome(&rep, pair.0, pair.1);
        assert_eq!(o.kind, "dislocation");
        assert_eq!(o.burgers_sublattice_index, Some(2));
        assert_eq!(o.burgers_coset_contains_zero, Some(true));
        assert!(o.contains_burgers([0, 0]) && o.contains_burgers([1, 1]) && !o.contains_burgers([1, 0]));
    }

    // D(1b, -180) x D(1b, +180): index-4 2L sublattice, contains (2,0), excludes (1,1) & (1,0).
    let o_bb = find_outcome(&rep, "D(1b, -180)", "D(1b, +180)");
    assert_eq!(o_bb.kind, "dislocation");
    assert_eq!(o_bb.burgers_sublattice_index, Some(4));
    assert_eq!(o_bb.burgers_coset_contains_zero, Some(true));
    assert!(o_bb.contains_burgers([2, 0]) && !o_bb.contains_burgers([1, 1]) && !o_bb.contains_burgers([1, 0]));

    // D(1b, -180) x D(1a, +180): index-4 coset (1,1) + 2L, contains (1,1), excludes (1,0) & (2,0).
    let o_ba = find_outcome(&rep, "D(1b, -180)", "D(1a, +180)");
    assert_eq!(o_ba.kind, "dislocation");
    assert_eq!(o_ba.burgers_sublattice_index, Some(4));
    assert_eq!(o_ba.burgers_coset_contains_zero, Some(false));
    assert!(o_ba.contains_burgers([1, 1]) && !o_ba.contains_burgers([1, 0]) && !o_ba.contains_burgers([2, 0]));

    // D(1b, -180) x D(2c, +180): contains (1,0), excludes (0,0) & (1,1).
    let o_bc = find_outcome(&rep, "D(1b, -180)", "D(2c, +180)");
    assert_eq!(o_bc.kind, "dislocation");
    assert_eq!(o_bc.burgers_coset_contains_zero, Some(false));
    assert!(o_bc.contains_burgers([1, 0]) && !o_bc.contains_burgers([0, 0]) && !o_bc.contains_burgers([1, 1]));

    // Lifted G~+ non-reduction: D(1b, -180) x D(1b, -180) is a -360 deg whole turn, NOT a dislocation!
    let o_wt = find_outcome(&rep, "D(1b, -180)", "D(1b, -180)");
    assert_eq!(o_wt.kind, "whole_turn");
    assert_eq!(o_wt.frank_angle_deg, -360.0);
    assert!(!o_wt.allowed_single_defect);
    assert_eq!(o_wt.burgers_sublattice_index, Some(4));

    // Dislocation + disclination: T1 (b=(1,0)) shifts D(1a, +90) to 1b.
    let o_shift = find_outcome(&rep, "T1", "D(1a, +90)");
    assert_eq!(o_shift.kind, "disclination");
    assert_eq!(o_shift.allowed_centre_classes, vec!["1b"]);
    assert!(o_shift.allowed_single_defect);
}

#[test]
fn test4_hexagonal_p6m_lifted_fusion_rules() {
    let hc = load_named("honeycomb");
    let rep = analyze_space_group_fusion(&hc.group, &hc.cell, hc.frame, 2.0).expect("p6m fusion");

    // 1a(+120) x 1a(-120): index-3 sublattice (ker q), contains 0.
    let o_1a_1a = find_outcome(&rep, "D(1a, +120)", "D(1a, -120)");
    assert_eq!(o_1a_1a.kind, "dislocation");
    assert_eq!(o_1a_1a.burgers_sublattice_index, Some(3));
    assert_eq!(o_1a_1a.burgers_coset_contains_zero, Some(true));

    // 1a(+120) x 2b(-120): excludes 0 (nonzero q mod 3).
    let o_1a_2b = find_outcome(&rep, "D(1a, +120)", "D(2b, -120)");
    assert_eq!(o_1a_2b.kind, "dislocation");
    assert_eq!(o_1a_2b.burgers_sublattice_index, Some(3));
    assert_eq!(o_1a_2b.burgers_coset_contains_zero, Some(false));
    assert!(!o_1a_2b.contains_burgers([0, 0]));

    // 2b(+120) x 2b(-120): index-1 full lattice L.
    let o_2b_2b = find_outcome(&rep, "D(2b, +120)", "D(2b, -120)");
    assert_eq!(o_2b_2b.kind, "dislocation");
    assert_eq!(o_2b_2b.burgers_sublattice_index, Some(1));
    assert_eq!(o_2b_2b.burgers_coset_contains_zero, Some(true));
    assert!(o_2b_2b.contains_burgers([0, 0]) && o_2b_2b.contains_burgers([1, 0]) && o_2b_2b.contains_burgers([0, 1]));

    // 1a(+60) x 1a(+120) and 1a(+60) x 2b(+120): CRT gives BOTH 1a and 3c at +180 deg.
    for pair in [("D(1a, +60)", "D(1a, +120)"), ("D(1a, +60)", "D(2b, +120)")] {
        let o = find_outcome(&rep, pair.0, pair.1);
        assert_eq!(o.kind, "disclination");
        assert_eq!(o.frank_angle_deg, 180.0);
        assert!(o.allowed_single_defect);
        assert_eq!(o.allowed_centre_classes, vec!["1a", "3c"]);
    }

    // 1a(+180) x 1a(-180): index-4 2L, contains 0.
    let o_180_aa = find_outcome(&rep, "D(1a, +180)", "D(1a, -180)");
    assert_eq!(o_180_aa.kind, "dislocation");
    assert_eq!(o_180_aa.burgers_sublattice_index, Some(4));
    assert_eq!(o_180_aa.burgers_coset_contains_zero, Some(true));

    // 1a(+180) x 3c(-180): excludes 0, odd parity mod 2 (all 3 nonzero cosets of 2L).
    let o_180_ac = find_outcome(&rep, "D(1a, +180)", "D(3c, -180)");
    assert_eq!(o_180_ac.kind, "dislocation");
    assert_eq!(o_180_ac.burgers_sublattice_index, Some(4));
    assert_eq!(o_180_ac.burgers_coset_contains_zero, Some(false));
    assert!(o_180_ac.contains_burgers([1, 0]) && o_180_ac.contains_burgers([0, 1]) && o_180_ac.contains_burgers([1, 1]));
    assert!(!o_180_ac.contains_burgers([0, 0]) && !o_180_ac.contains_burgers([2, 0]));

    // 1a(+120) x 1a(+120): ONLY 1a at +240 deg, allowed_single_defect == false since |240| > 180.
    let o_240 = find_outcome(&rep, "D(1a, +120)", "D(1a, +120)");
    assert_eq!(o_240.kind, "disclination");
    assert_eq!(o_240.frank_angle_deg, 240.0);
    assert_eq!(o_240.allowed_centre_classes, vec!["1a"]);
    assert!(!o_240.allowed_single_defect);
}

#[test]
fn test5_all_17_wallpaper_groups_and_origin_shift_invariance() {
    let expected_ab: BTreeMap<&str, (bool, &str, &str)> = BTreeMap::from([
        ("p1", (true, "Z^2", "Z^2")),
        ("p2", (true, "(Z/2Z)^3", "(Z/2Z)^3")),
        ("pm", (true, "Z x (Z/2Z)^2", "Z^2")),
        ("pg", (false, "Z x Z/2Z", "Z^2")),
        ("cm", (true, "Z x Z/2Z", "Z^2")),
        ("pmm", (true, "(Z/2Z)^4", "(Z/2Z)^3")),
        ("pmg", (false, "(Z/2Z)^3", "(Z/2Z)^3")),
        ("pgg", (false, "Z/2Z x Z/4Z", "(Z/2Z)^3")),
        ("cmm", (true, "(Z/2Z)^3", "(Z/2Z)^3")),
        ("p4", (true, "Z/2Z x Z/4Z", "Z/2Z x Z/4Z")),
        ("p4m", (true, "(Z/2Z)^3", "Z/2Z x Z/4Z")),
        ("p4g", (false, "Z/2Z x Z/4Z", "Z/2Z x Z/4Z")),
        ("p3", (true, "(Z/3Z)^2", "(Z/3Z)^2")),
        ("p3m1", (true, "Z/2Z", "(Z/3Z)^2")),
        ("p31m", (true, "Z/6Z", "(Z/3Z)^2")),
        ("p6", (true, "Z/6Z", "Z/6Z")),
        ("p6m", (true, "(Z/2Z)^2", "Z/6Z")),
    ]);

    let cats = parse_catalogue_str(POINTS_JSON).expect("parse points.json");
    assert_eq!(cats.len(), 17, "points.json must contain all 17 wallpaper groups");
    for (grp, cat) in &cats {
        let ent = cat.drawings.first().unwrap_or_else(|| panic!("empty catalogue for {grp}"));
        let fam = LatticeFamily::from_recipe(&ent.recipe, Some(Frame2D::default())).expect("recipe");
        let rep = analyze_space_group_fusion(&fam.group, &fam.cell(), Frame2D::default(), 1.5)
            .unwrap_or_else(|e| panic!("group {grp} failed: {e}"));
        let &(exp_symm, exp_full, exp_prop) = expected_ab
            .get(rep.group.as_str())
            .unwrap_or_else(|| panic!("unexpected group {}", rep.group));
        assert_eq!(rep.symmorphic, exp_symm, "{grp} symmorphic mismatch");
        assert_eq!(rep.ab_full.label, exp_full, "{grp} G^ab mismatch");
        assert_eq!(rep.ab_proper.label, exp_prop, "{grp} (G+)^ab mismatch");
    }

    // Verify exact invariance under an irrational basis origin shift.
    for key in ["square", "shastry_sutherland"] {
        let lat = load_named(key);
        let base = analyze_space_group_fusion(&lat.group, &lat.cell, lat.frame, 1.5).expect("base");
        let shifted_cell = Cell2D {
            a1: lat.cell.a1,
            a2: lat.cell.a2,
            basis: lat.cell.basis.iter().map(|&p| [p[0] + 0.137, p[1] + 0.289]).collect(),
            bonds: lat.cell.bonds.clone(),
        };
        let shifted = analyze_space_group_fusion(&lat.group, &shifted_cell, Frame2D::default(), 1.5)
            .expect("shifted origin must still quantize exactly");
        assert_eq!(shifted.ab_full, base.ab_full, "{key} shifted G^ab");
        assert_eq!(shifted.ab_proper, base.ab_proper, "{key} shifted (G+)^ab");
        assert_eq!(shifted.symmorphic, base.symmorphic, "{key} shifted symmorphic");
        assert_eq!(shifted.improper_sectors.len(), base.improper_sectors.len());
    }
}
