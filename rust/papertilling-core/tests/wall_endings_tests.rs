//! Stage 2 tests: wall endings `(sigma * G) ∩ E+(2)`, terminated half-walls,
//! fractional partial_dislocation charges, sheet3d compatibility, and 38-lattice parity.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;

use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::combine_api::{classify_crystal_charge, equivalent_single};
use papertilling_core::defect::combine_types::{ConfigOptionsJson, NotebookConfigJson};
use papertilling_core::defect::half_wall::build_half_wall;
use papertilling_core::defect::sheet3d::{Sheet3DOptions, relax_sheet_3d};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::defect::wall_endings::WallEndingInfo;
use papertilling_core::defect::{auto_burgers_circuit, classify_wall_ending};
use papertilling_core::plane::draw::{ensure_embedded, relax_picture};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::{Cell2D, Frame2D, Isometry2, Recipe};

#[derive(Deserialize)]
struct PointsFile { groups: BTreeMap<String, PointsGroup> }
#[derive(Deserialize)]
struct PointsGroup { entries: Vec<PointsEntry> }
#[derive(Deserialize)]
struct PointsEntry { recipe: Recipe }
#[derive(Deserialize)]
struct NamedFile { named: Vec<NamedEntry> }
#[derive(Deserialize)]
struct NamedEntry { key: String, recipe: Recipe, frame: Frame2D }

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../papertilling/data")
}

fn load_named(key: &str) -> (String, Cell2D, Frame2D, Recipe) {
    let raw = std::fs::read_to_string(data_dir().join("named.json")).expect("named.json");
    let nf: NamedFile = serde_json::from_str(&raw).expect("parse named.json");
    let e = nf.named.into_iter().find(|x| x.key == key).expect("named lattice exists");
    let fam = LatticeFamily::from_recipe(&e.recipe, Some(e.frame)).expect("recipe builds");
    (fam.group.clone(), fam.cell(), e.frame, e.recipe)
}

fn vec2_close(a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
    (a[0] - b[0]).hypot(a[1] - b[1]) <= tol
}

#[test]
fn all_38_gate_lattices_wall_ending_invariants() {
    let points: PointsFile = serde_json::from_str(&std::fs::read_to_string(data_dir().join("points.json")).unwrap()).unwrap();
    let named: NamedFile = serde_json::from_str(&std::fs::read_to_string(data_dir().join("named.json")).unwrap()).unwrap();

    const GROUPS: [&str; 17] = [
        "p1", "p2", "pm", "pg", "cm", "pmm", "pmg", "pgg", "cmm", "p4", "p4m", "p4g", "p3", "p3m1", "p31m", "p6", "p6m",
    ];
    let mut items: Vec<(String, Recipe, Option<Frame2D>)> = GROUPS
        .iter()
        .map(|g| (format!("{g}#1"), points.groups[*g].entries[0].recipe.clone(), None))
        .collect();
    items.extend(named.named.into_iter().map(|n| (n.key, n.recipe, Some(n.frame))));
    assert_eq!(items.len(), 38);

    let mut total_walls = 0usize;
    for (tag, recipe, frame) in items {
        let fam = LatticeFamily::from_recipe(&recipe, frame).unwrap();
        let cat = DefectCatalogue::new(&fam.group, &fam.cell(), frame.unwrap_or_default());
        for w in &cat.mobius_seams {
            assert!(!w.ending.can_end_orientable, "{tag}/{}", w.id);
            assert_eq!(w.ending.ending_kind, "non_orientable_twist", "{tag}/{}", w.id);
        }
        let check_ending = |id: &str, sigma: Isometry2, e: &WallEndingInfo| {
            if cat.chiral && sigma.det() < 0.0 {
                assert!(!e.can_end_orientable, "{tag}/{id}");
                assert_eq!(e.ending_kind, "chirality_obstruction", "{tag}/{id}");
            }
            if e.can_end_orientable {
                let r = e.holonomy_linear.expect("linear");
                assert!(r[0][0] * r[1][1] - r[0][1] * r[1][0] > 0.0, "{tag}/{id}");
                assert!(e.holonomy_trans.is_some(), "{tag}/{id}");
                assert!(matches!(e.ending_kind.as_str(), "partial_dislocation" | "partial_disclination"));
            }
        };
        for w in &cat.domain_walls { check_ending(&w.id, w.sigma, &w.ending); }
        for w in &cat.twin_walls { check_ending(&w.id, w.sigma, &w.ending); }
        for w in &cat.glide_breaking { check_ending(&w.id, w.sigma, &w.ending); }
        total_walls += cat.domain_walls.len() + cat.twin_walls.len() + cat.mobius_seams.len() + cat.glide_breaking.len();
    }
    assert_eq!(total_walls, 862);
}

#[test]
fn known_wall_ending_controls_and_obstructions() {
    let (_, cell_bp, f_bp, _) = load_named("brick_pg");
    let cat_bp = DefectCatalogue::new("pg", &cell_bp, f_bp);
    let wa1 = cat_bp.domain_walls.iter().find(|w| w.id == "Wa1").unwrap();
    assert!(wa1.ending.can_end_orientable);
    assert_eq!(wa1.ending.ending_kind, "partial_dislocation");
    assert!(!wa1.ending.is_full_lattice_burgers);
    assert!(vec2_close(wa1.ending.burgers.unwrap(), [1.0, 0.0], 1e-7));

    let wa4 = cat_bp.domain_walls.iter().find(|w| w.id == "Wa4").unwrap();
    assert_eq!(wa4.ending.ending_kind, "partial_disclination");
    assert!((wa4.ending.frank_angle_deg.unwrap() - 180.0).abs() < 1e-6);
    assert!(vec2_close(wa4.ending.centre.unwrap(), [0.0, 0.0], 1e-6));

    let wc1 = cat_bp.mobius_seams.iter().find(|w| w.id == "Wc1").unwrap();
    assert!(!wc1.ending.can_end_orientable);
    assert_eq!(wc1.ending.ending_kind, "non_orientable_twist");
    assert_eq!(classify_wall_ending(&cell_bp, wc1.sigma, "Wc").ending_kind, "non_orientable_twist");

    let wd1 = cat_bp.glide_breaking.iter().find(|w| w.id == "Wd1").unwrap();
    assert_eq!(wd1.ending.ending_kind, "partial_dislocation");
    assert!(vec2_close(wd1.ending.burgers.unwrap(), [0.0, 2.0], 1e-7));

    let (_, cell_ct, f_ct, _) = load_named("chiral_triangular");
    let cat_ct = DefectCatalogue::new("p3", &cell_ct, f_ct);
    let ct_wa3 = cat_ct.domain_walls.iter().find(|w| w.id == "Wa3").unwrap();
    assert_eq!(ct_wa3.ending.ending_kind, "partial_disclination");
    assert!((ct_wa3.ending.frank_angle_deg.unwrap() - 60.0).abs() < 1e-6);
    let ct_wa6 = cat_ct.domain_walls.iter().find(|w| w.id == "Wa6").unwrap();
    assert!(!ct_wa6.ending.can_end_orientable);
    assert_eq!(ct_wa6.ending.ending_kind, "chirality_obstruction");

    let (_, cell_cs, f_cs, _) = load_named("chiral_square");
    let cat_cs = DefectCatalogue::new("p4", &cell_cs, f_cs);
    assert!(cat_cs.domain_walls.iter().any(|w| !w.ending.can_end_orientable && w.ending.ending_kind == "chirality_obstruction"));

    let (_, cell_sq, f_sq, _) = load_named("square");
    let cat_sq = DefectCatalogue::new("p4m", &cell_sq, f_sq);
    let wb1 = cat_sq.twin_walls.iter().find(|w| w.id == "Wb1").unwrap();
    assert!(wb1.ending.can_end_orientable && wb1.ending.is_full_lattice_burgers);
    assert!(vec2_close(wb1.ending.burgers.unwrap(), [0.0, 0.0], 1e-7));
}

#[test]
fn half_wall_rim_holonomy_circuit_json_and_sheet3d() {
    let (grp, cell, frame, recipe) = load_named("shastry_sutherland");
    for (wid, expected_b) in [("Wa1", [1.0, 1.0]), ("Wd2", [1.0, 0.0])] {
        let specs = [DefectSpec::half_wall(Some(wid), [0.0, 0.0], None, None)];
        let mut crystal = build_point_defects(&cell, &specs, None, None).expect("shastry hw builds");
        let q = classify_crystal_charge(&grp, &cell, frame, &crystal).unwrap();
        assert_eq!(q.kind, "partial_dislocation", "{wid}");
        assert_eq!(q.lattice_vector, Some(false), "{wid}");
        assert!(vec2_close(q.b.unwrap(), expected_b, 1e-6), "{wid}");
        assert!(q.single.is_none(), "{wid}");

        let cir = auto_burgers_circuit(&crystal, Some((&grp, &cell, frame)), &[], &[], 2, &specs).unwrap();
        assert!((cir.frank_angle_deg.unwrap()).abs() < 1e-6);
        assert!((cir.burgers_norm - expected_b[0].hypot(expected_b[1])).abs() < 1e-5);

        let eq = equivalent_single(&grp, &cell, frame, &specs, &crystal, 2, 1, false).unwrap();
        assert!(!eq.equivalent && eq.match_tier == "none", "{wid}");

        relax_picture(&mut crystal);
        ensure_embedded(&mut crystal, false);
        let res3d = relax_sheet_3d(&crystal, &cell, &Sheet3DOptions { max_iters: 60, ..Default::default() }, None);
        assert!(res3d.has_finite_geometry(), "{wid} sheet3d");
    }

    // JSON roundtrip and build refusal for Mobius seam termination
    let doc = NotebookConfigJson {
        format: "papertilling.config.v1".to_string(),
        lattice_label: "shastry_sutherland".to_string(),
        group: grp,
        recipe,
        frame,
        defects: vec![DefectSpec::half_wall(Some("Wa1"), [0.0, 0.0], None, None)],
        options: ConfigOptionsJson::default(),
        summary: None,
    };
    let s = doc.to_json_pretty().unwrap();
    assert!(s.contains("\"half_wall\"") && s.contains("\"Wa1\""));
    assert_eq!(NotebookConfigJson::from_json_str(&s).unwrap(), doc);

    let (g_bp, cell_bp, _, _) = load_named("brick_pg");
    let err = build_half_wall(&cell_bp, Some("Wc1"), None, [0.0, 0.0], None, None, false).unwrap_err();
    assert!(err.contains("Mobius") || err.contains("non-orientable"), "{err}");

    // Partial disclination half-walls & superposition with point defect + 3D buckling
    for (lat_key, wid, exp_ang) in [("brick_pg", "Wa4", 180.0), ("chiral_triangular", "Wa3", 60.0)] {
        let (g, c, fr, _) = load_named(lat_key);
        let (_, crystal) = build_half_wall(&c, Some(wid), None, [0.0, 0.0], None, None, true).unwrap();
        let q = classify_crystal_charge(&g, &c, fr, &crystal).unwrap();
        assert_eq!(q.kind, "disclination", "{lat_key} {wid}");
        assert!((q.angle - exp_ang).abs() < 1e-5);
        assert!(q.single.is_none());
        let eq = equivalent_single(&g, &c, fr, &[DefectSpec::half_wall(Some(wid), [0.0, 0.0], None, None)], &crystal, 2, 1, false).unwrap();
        assert!(!eq.equivalent && eq.match_tier == "none");
        let res3d = relax_sheet_3d(&crystal, &c, &Sheet3DOptions { max_iters: 60, ..Default::default() }, None);
        assert!(res3d.has_finite_geometry());
    }

    // Half-wall + point dislocation superposition (brick_pg Wd1 b=[0,2] + b=[2,0])
    let sig_override: Option<([[f64; 2]; 2], [f64; 2])> = None;
    let combo = [
        DefectSpec::half_wall(Some("Wd1"), [0.0, 0.0], None, sig_override),
        DefectSpec::dislocation([2.0, 0.0], Some([2.3, -1.5])),
    ];
    let c_combo = build_point_defects(&cell_bp, &combo, None, None).expect("combo builds");
    let q_combo = classify_crystal_charge(&g_bp, &cell_bp, Frame2D::default(), &c_combo).unwrap();
    assert_eq!(q_combo.kind, "partial_dislocation");
    let b_sum = q_combo.b.unwrap();
    assert!(vec2_close(b_sum, [2.0, 2.0], 1e-5) || vec2_close(b_sum, [-2.0, 2.0], 1e-5), "got {b_sum:?}");
    let _ = Isometry2::IDENTITY;
}
