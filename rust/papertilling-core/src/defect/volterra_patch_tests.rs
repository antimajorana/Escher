//! Focused known-answer tests for bidirectional Volterra dislocation surgery (`volterra_patch.rs`).

use crate::defect::charge::{check_frames, total_charge};
use crate::defect::circuit::{auto_burgers_circuit, burgers_circuit};
use crate::defect::core_gallery::core_gallery;
use crate::defect::core_graph::LatticeCoreInfo;
use crate::defect::equivalence::same_proper_holonomy;
use crate::defect::single_cache::SingleDefectCache;
use crate::defect::single_crystal::build_single_equivalence_bundle;
use crate::defect::surgery::{DefectSpec, build_point_defects};
use crate::defect::sym_helper::LatticeGroup;
use crate::defect::volterra_patch::build_volterra_dislocation;
use crate::defect::volterra_scan::scan_dislocation_cores;
use crate::plane::planarity::check_positions;
use crate::types::{Bond2D, Cell2D, Frame2D, Isometry2};

fn assert_vec2_close(got: [f64; 2], want: [f64; 2], tol: f64, ctx: &str) {
    assert!(
        (got[0] - want[0]).hypot(got[1] - want[1]) <= tol,
        "{ctx}: got {got:?}, want {want:?}"
    );
}

fn square_p4m_cell() -> Cell2D {
    Cell2D {
        a1: [1.0, 0.0],
        a2: [0.0, 1.0],
        basis: vec![[0.0, 0.0]],
        bonds: vec![
            Bond2D {
                i: 0,
                j: 0,
                n1: 1,
                n2: 0,
                kind: 0,
            },
            Bond2D {
                i: 0,
                j: 0,
                n1: 0,
                n2: 1,
                kind: 0,
            },
        ],
    }
}

/// Hand-derived `p3m1` breathing kagome (3-fold rotations + mirrors, NO C2 inversion):
/// equilateral small triangles of side 0.44 and large triangles of side 0.56.
fn breathing_kagome_p3m1_cell() -> Cell2D {
    let s3 = 3.0_f64.sqrt() / 2.0;
    let a1 = [1.0, 0.0];
    let a2 = [0.5, s3];
    let d = 0.44;
    let pt = |u: f64, v: f64| [u * a1[0] + v * a2[0], u * a1[1] + v * a2[1]];
    Cell2D {
        a1,
        a2,
        basis: vec![pt(0.0, 0.0), pt(d, 0.0), pt(0.0, d)],
        bonds: vec![
            Bond2D {
                i: 0,
                j: 1,
                n1: 0,
                n2: 0,
                kind: 0,
            },
            Bond2D {
                i: 1,
                j: 2,
                n1: 0,
                n2: 0,
                kind: 0,
            },
            Bond2D {
                i: 2,
                j: 0,
                n1: 0,
                n2: 0,
                kind: 0,
            },
            Bond2D {
                i: 1,
                j: 0,
                n1: 1,
                n2: 0,
                kind: 1,
            },
            Bond2D {
                i: 2,
                j: 0,
                n1: 0,
                n2: 1,
                kind: 1,
            },
            Bond2D {
                i: 1,
                j: 2,
                n1: 1,
                n2: -1,
                kind: 1,
            },
        ],
    }
}

/// Hand-derived `p1` planar asymmetric 3-site cell (no non-trivial rotations at all, NO C2).
fn asymmetric_p1_cell() -> Cell2D {
    let a1 = [1.0, 0.0];
    let a2 = [0.25, 1.15];
    let pt = |u: f64, v: f64| [u * a1[0] + v * a2[0], u * a1[1] + v * a2[1]];
    Cell2D {
        a1,
        a2,
        basis: vec![pt(0.0, 0.0), pt(0.41, 0.12), pt(0.17, 0.53)],
        bonds: vec![
            Bond2D {
                i: 0,
                j: 1,
                n1: 0,
                n2: 0,
                kind: 0,
            },
            Bond2D {
                i: 1,
                j: 2,
                n1: 0,
                n2: 0,
                kind: 0,
            },
            Bond2D {
                i: 2,
                j: 0,
                n1: 0,
                n2: 0,
                kind: 0,
            },
            Bond2D {
                i: 1,
                j: 0,
                n1: 1,
                n2: 0,
                kind: 0,
            },
            Bond2D {
                i: 2,
                j: 0,
                n1: 0,
                n2: 1,
                kind: 0,
            },
            Bond2D {
                i: 1,
                j: 2,
                n1: 1,
                n2: -1,
                kind: 0,
            },
        ],
    }
}

#[test]
fn p4m_square_volterra_default_left_and_right_cuts_all_produce_exact_plus_b() {
    let cell = square_p4m_cell();
    let lat_info = LatticeCoreInfo::new(&cell);
    let b = [1.0, 0.0];
    let neg_b = [-1.0, 0.0];
    let core_pt = [0.371, 0.413];

    let cuts: [(&str, Option<[f64; 2]>); 6] = [
        ("default (None -> left of b)", None),
        ("perp left (cross2(d,b) < 0)", Some([0.0, 1.0])),
        ("perp right (cross2(d,b) > 0)", Some([0.0, -1.0])),
        ("oblique left (cross2(d,b) < 0)", Some([1.0, 1.0])),
        ("oblique right (cross2(d,b) > 0)", Some([1.0, -1.0])),
        ("rescaled cut", Some([1e-18, -1e-18])),
    ];

    for (label, cut) in cuts {
        let crystal = build_volterra_dislocation(&lat_info, b, core_pt, cut, 7.0)
            .unwrap_or_else(|e| panic!("{label}: build failed: {e}"));
        assert_eq!(check_frames(&crystal), 0, "{label}: frame mismatch");
        let chk = check_positions(&crystal, &crystal.pos, false, None);
        assert_eq!(chk.crossings, 0, "{label}: raw crossings");
        assert_eq!(chk.overlaps, 0, "{label}: raw overlaps");

        let charge = total_charge(&crystal).expect("total_charge");
        assert_eq!(charge.kind, "dislocation", "{label}");
        assert!(
            charge.frank_deg.abs() < 1e-9,
            "{label}: frank={}",
            charge.frank_deg
        );
        assert_vec2_close(
            charge.burgers.unwrap(),
            b,
            1e-9,
            &format!("{label} total_charge.burgers"),
        );
        assert_vec2_close(
            charge.holonomy.t,
            neg_b,
            1e-9,
            &format!("{label} total_charge.holonomy.t"),
        );
        assert_vec2_close(
            charge.loop_sum,
            b,
            1e-9,
            &format!("{label} total_charge.loop_sum"),
        );

        let auto = auto_burgers_circuit(
            &crystal,
            Some(("p4m", &cell, Frame2D::default())),
            &[],
            &[],
            2,
            &[],
        )
        .unwrap_or_else(|e| panic!("{label}: auto_burgers_circuit failed: {e}"));
        assert!(auto.orientation_preserving, "{label}");
        assert!(auto.frank_angle_deg.unwrap().abs() < 1e-9, "{label}");
        assert_vec2_close(
            auto.closure_vector,
            b,
            1e-9,
            &format!("{label} auto_burgers_circuit closure"),
        );
        assert_vec2_close(
            auto.h.t,
            neg_b,
            1e-9,
            &format!("{label} auto_burgers_circuit H.t"),
        );

        let manual = burgers_circuit(
            &crystal,
            &auto.path,
            Some(("p4m", &cell, Frame2D::default())),
        )
        .unwrap_or_else(|e| panic!("{label}: burgers_circuit failed: {e}"));
        assert_vec2_close(
            manual.closure_vector,
            b,
            1e-9,
            &format!("{label} burgers_circuit closure"),
        );
    }

    for bad_cut in [[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [f64::NAN, 1.0]] {
        assert!(
            build_volterra_dislocation(&lat_info, b, core_pt, Some(bad_cut), 7.0).is_err(),
            "parallel/zero cut {bad_cut:?} must be refused"
        );
    }
}

#[test]
fn non_c2_groups_p1_and_p3m1_volterra_both_cut_sides_and_scan_preserve_plus_b() {
    let cases = [
        ("p1", asymmetric_p1_cell(), [1.0, 0.0]),
        ("p3m1", breathing_kagome_p3m1_cell(), [1.0, 0.0]),
    ];

    for (expected_group, cell, b) in cases {
        let lat_info = LatticeCoreInfo::new(&cell);
        let grp = LatticeGroup::new(&cell);
        assert_eq!(grp.name, expected_group, "hand-derived cell group");

        let h_plus = Isometry2 {
            t: [-b[0], -b[1]],
            ..Isometry2::IDENTITY
        };
        let h_minus = Isometry2 {
            t: [b[0], b[1]],
            ..Isometry2::IDENTITY
        };
        assert!(
            !same_proper_holonomy(h_minus, h_plus, &grp.plus, grp.a, grp.ainv, 1e-4),
            "{expected_group} must NOT have C2 conjugating -b to +b"
        );

        let core_pt = [
            0.371 * cell.a1[0] + 0.413 * cell.a2[0],
            0.371 * cell.a1[1] + 0.413 * cell.a2[1],
        ];
        let cuts: [(&str, Option<[f64; 2]>); 3] = [
            ("default", None),
            ("left_of_b", Some([0.0, 1.0])),
            ("right_of_b", Some([0.0, -1.0])),
        ];
        for (tag, cut) in cuts {
            let crystal = build_volterra_dislocation(&lat_info, b, core_pt, cut, 7.5)
                .unwrap_or_else(|e| panic!("{expected_group} {tag}: {e}"));
            assert_eq!(check_frames(&crystal), 0, "{expected_group} {tag}: frames");
            let chk = check_positions(&crystal, &crystal.pos, false, None);
            assert_eq!(chk.crossings, 0, "{expected_group} {tag}: crossings");
            assert_eq!(chk.overlaps, 0, "{expected_group} {tag}: overlaps");

            let charge = total_charge(&crystal).expect("total_charge");
            assert_eq!(charge.kind, "dislocation", "{expected_group} {tag}");
            assert_vec2_close(
                charge.burgers.unwrap(),
                b,
                1e-8,
                &format!("{expected_group} {tag} burgers"),
            );
            assert!(
                same_proper_holonomy(charge.holonomy, h_plus, &grp.plus, grp.a, grp.ainv, 1e-4),
                "{expected_group} {tag} must match +b class"
            );
            assert!(
                !same_proper_holonomy(charge.holonomy, h_minus, &grp.plus, grp.a, grp.ainv, 1e-4),
                "{expected_group} {tag} must NOT match -b class"
            );
        }

        // Volterra scan must accept both cut orientations without rejecting opposite-side cuts into other_class.
        let scan = scan_dislocation_cores(&cell, b, 2, 1, 8, None)
            .unwrap_or_else(|e| panic!("{expected_group} scan: {e}"));
        assert_eq!(
            scan.info.other_class, 0,
            "{expected_group}: spurious other_class rejection"
        );
        assert_eq!(scan.info.anomalies, 0, "{expected_group}: scan anomalies");
        assert!(
            !scan.cores.is_empty(),
            "{expected_group}: expected scanned cores"
        );
        for core in &scan.cores {
            let cp = core.info.core_point.expect("core_point");
            let built =
                build_volterra_dislocation(&lat_info, b, cp, core.info.cut, 7.5).expect("rebuild");
            let ch = total_charge(&built).expect("charge");
            assert_vec2_close(
                ch.burgers.unwrap(),
                b,
                1e-8,
                &format!("{expected_group} {}", core.info.name),
            );
        }
    }
}

#[test]
fn p3m1_dipole_single_equivalence_bundle_preserves_exact_plus_b_and_square_gallery_intact() {
    // 1. Planar square core gallery invariants remain undisturbed.
    let sq = square_p4m_cell();
    let (gal, scan) = core_gallery("p4m", &sq, Frame2D::default(), [1.0, 0.0], 2, 1, true).unwrap();
    assert_eq!(scan.other_class, 0);
    assert_eq!(scan.distinct, 2);
    assert_eq!(gal.len(), 2);
    assert!(
        gal.iter()
            .all(|e| e.kind == "Volterra" && e.faces_ok && e.iso_ok == Some(true))
    );
    let mut ring_sets: Vec<Vec<usize>> = gal.iter().map(|e| e.rings.clone()).collect();
    ring_sets.sort();
    assert_eq!(ring_sets, vec![vec![5], vec![7]]);

    // 2. Non-C2 p3m1 breathing kagome +120/-120 dipole -> single equivalence bundle carries exact +b.
    let bk = breathing_kagome_p3m1_cell();
    let d = 0.44;
    let c_small = [
        (d / 3.0) * bk.a1[0] + (d / 3.0) * bk.a2[0],
        (d / 3.0) * bk.a1[1] + (d / 3.0) * bk.a2[1],
    ];
    let c_next = [c_small[0] + bk.a1[0], c_small[1] + bk.a1[1]];
    let specs = [
        DefectSpec::disclination(120.0, c_small),
        DefectSpec::disclination(-120.0, c_next),
    ];
    let combined = build_point_defects(&bk, &specs, None, None).expect("p3m1 dipole builds");
    let comb_b = total_charge(&combined)
        .expect("combined charge")
        .burgers
        .expect("dislocation b");

    let mut cache = SingleDefectCache::default();
    let bundle = build_single_equivalence_bundle(
        "p3m1",
        &bk,
        Frame2D::default(),
        &specs,
        &combined,
        8.5,
        2,
        1,
        false,
        &mut cache,
    )
    .expect("p3m1 bundle");
    assert_eq!(bundle.report.kind, "dislocation");
    assert_eq!(bundle.report.scan.as_ref().expect("scan").other_class, 0);
    assert!(!bundle.candidates.is_empty());
    let rep_b = bundle.report.charge.b.expect("report b");
    assert_vec2_close(rep_b, comb_b, 1e-7, "p3m1 report b vs combined");
    for cand in &bundle.candidates {
        if cand.kind == "Volterra" {
            assert_eq!(check_frames(&cand.crystal), 0, "p3m1 {}", cand.name);
        }
        let ch = total_charge(&cand.crystal).expect("cand charge");
        assert_vec2_close(
            ch.burgers.unwrap(),
            comb_b,
            1e-7,
            &format!("p3m1 {}", cand.name),
        );
    }
}
