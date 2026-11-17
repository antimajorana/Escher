//! Stage 5 hand-derived unit tests, 500-site live-update latency benchmark,
//! and 100% parity gate against Python (`PT_GATE_STAGE5`).

use std::time::Instant;

use petgraph::algo::connected_components;
use serde::Deserialize;

use crate::defect::catalogue::DefectCatalogue;
use crate::defect::combine_api::{
    classify_configurations, classify_crystal_charge, compare_configurations, core_gallery,
    equivalent_single, inspect_crystal_cores, nearest_allowed_centre,
};
use crate::defect::combine_types::{
    ConfigOptionsJson, ConfigSummaryJson, NotebookConfigJson,
};
use crate::defect::core_gallery::describe_core;
use crate::defect::core_graph::{LatticeCoreInfo, core_graph};
use crate::defect::equivalence::is_bipartite;
use crate::defect::surgery::{DefectSpec, build_point_defects};
use crate::plane::draw::relax_picture;
use crate::plane::planarity::check_positions;
use crate::types::{Bond2D, Cell2D, Frame2D, Recipe};

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
    Cell2D {
        a1: [s3, 0.0],
        a2: [0.5 * s3, 1.5],
        basis: vec![[0.0, 0.0], [0.5 * s3, 0.5]],
        bonds: vec![
            Bond2D { i: 0, j: 1, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 1, j: 0, n1: 1, n2: 0, kind: 0 },
            Bond2D { i: 1, j: 0, n1: 0, n2: 1, kind: 0 },
        ],
    }
}

fn shastry_sutherland_cell() -> Cell2D {
    Cell2D {
        a1: [2.0, 0.0],
        a2: [0.0, 2.0],
        basis: vec![[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        bonds: vec![
            Bond2D { i: 0, j: 2, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 2, j: 0, n1: 1, n2: 0, kind: 0 },
            Bond2D { i: 0, j: 3, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 3, j: 0, n1: 0, n2: 1, kind: 0 },
            Bond2D { i: 1, j: 2, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 1, j: 2, n1: 0, n2: 1, kind: 0 },
            Bond2D { i: 1, j: 3, n1: 0, n2: 0, kind: 0 },
            Bond2D { i: 1, j: 3, n1: 1, n2: 0, kind: 0 },
            Bond2D { i: 0, j: 1, n1: 0, n2: 0, kind: 1 },
            Bond2D { i: 2, j: 3, n1: 1, n2: -1, kind: 1 },
        ],
    }
}

#[test]
fn stage5_square_pair_gallery_and_descriptions_by_hand() {
    // Hand derivation on square (p4m):
    // +90 at site (0,0) and -90 at face centre (0.5, 0.5) has Frank sum 0 and outer holonomy
    // translation of length 1 -> unit dislocation orbit [1.0, 0.0].
    // Perpendicular Volterra cut gives a 5-ring with one 3-bond site; oblique cut gives a 7-ring
    // with three 3-bond sites (one pair bonded), whose 1-bond completions recover the 5-ring core.
    assert!(describe_core(&[5, 6, 7], &[6], &[]).starts_with("5-7 pair, plus one hexagon"));
    assert_eq!(describe_core(&[5], &[4], &[1]), "one pentagon; one under-coordinated site on the ring");

    let sq = square_cell();
    let (pt, lbl) = nearest_allowed_centre("p4m", &sq, Frame2D::default(), 90.0, [0.12, -0.08]).unwrap();
    assert!((pt[0]).abs() < 1e-9 && (pt[1]).abs() < 1e-9 && lbl == "1a");
    assert!(nearest_allowed_centre("p4m", &sq, Frame2D::default(), 60.0, [0.0, 0.0]).is_err());

    let specs = [
        DefectSpec::disclination(90.0, [0.0, 0.0]),
        DefectSpec::disclination(-90.0, [0.5, 0.5]),
    ];
    let pair = build_point_defects(&sq, &specs, None, None).unwrap();
    let rep = equivalent_single("p4m", &sq, Frame2D::default(), &specs, &pair, 2, 1, true).unwrap();
    assert_eq!(rep.kind, "dislocation");
    assert!(rep.equivalent);
    assert_eq!(rep.matches, vec!["Volterra core 1"]);
    assert_eq!(rep.verdict, "equivalent to the single defect (core: Volterra core 1)");
    assert!(rep.config_core.is_some());

    let (gal, scan) = core_gallery("p4m", &sq, Frame2D::default(), [1.0, 0.0], 2, 1, true).unwrap();
    assert_eq!(scan.distinct, 2);
    assert_eq!(gal.len(), 2);
    assert!(gal.iter().all(|e| e.kind == "Volterra" && e.faces_ok && e.iso_ok == Some(true)));
    let pent = gal.iter().find(|e| e.rings == vec![5]).expect("pentagon core");
    let hept = gal.iter().find(|e| e.rings == vec![7]).expect("heptagon core");
    assert_eq!(pent.under_sites.len(), 1);
    assert_eq!(hept.under_sites.len(), 3);
    assert!(pent.also.iter().any(|s| s.contains(&hept.name)));
    for e in &gal {
        let chk = check_positions(&e.crystal, &e.crystal.pos, false, None);
        assert_eq!(chk.crossings, 0, "{} has drawn crossings", e.name);
    }
}

#[test]
fn stage5_honeycomb_57_completion_control() {
    // Hand + C04: on honeycomb, +60 and -60 at neighbouring 6-fold hexagon centres separated by a1
    // give a shortest dislocation (|b| = sqrt(3)) with adjacent 5- and 7-rings (non-bipartite).
    // Every Volterra core on honeycomb is bipartite, so the match must be a completion core.
    let hc = honeycomb_cell();
    let cat = DefectCatalogue::new("p6m", &hc, Frame2D::default());
    let c6 = cat.centre_classes.iter().find(|c| c.order == 6).expect("6-fold centre").point;
    let specs = [
        DefectSpec::disclination(60.0, c6),
        DefectSpec::disclination(-60.0, [c6[0] + hc.a1[0], c6[1] + hc.a1[1]]),
    ];
    let crys = build_point_defects(&hc, &specs, None, None).unwrap();
    let rep = equivalent_single("p6m", &hc, Frame2D::default(), &specs, &crys, 2, 2, true).unwrap();
    assert!(rep.equivalent);
    assert!(rep.candidates.iter().filter(|c| c.volterra == Some(true)).all(|c| c.bipartite == Some(true)));
    assert!(rep.matches.iter().all(|m| m.starts_with("completion")));

    let hit = rep
        .gallery
        .iter()
        .find(|e| e.kind == "completion" && e.rings.iter().copied().filter(|&r| r != 6).collect::<Vec<_>>() == vec![5, 7])
        .expect("5-7 completion gallery entry");
    assert!(!hit.chords.is_empty());
    assert!(hit.description.starts_with("5-7 pair"));
    assert!(hit.faces_ok && hit.iso_ok == Some(true));
    for e in &rep.gallery {
        let chk = check_positions(&e.crystal, &e.crystal.pos, false, None);
        assert_eq!(chk.crossings, 0, "honeycomb {} has drawn crossings", e.name);
    }
}

#[test]
fn stage5_p4g_mirror_partners_classify_and_negative_controls() {
    // 1. Shastry-Sutherland (p4g): two order-4 centres in Wyckoff 2a split in G+ = p4, swapped by mirrors.
    let ss = shastry_sutherland_cell();
    let cat = DefectCatalogue::new("p4g", &ss, Frame2D::default());
    let mut four: Vec<_> = cat.centre_classes.iter().filter(|c| c.order == 4).collect();
    four.sort_by(|a, b| a.label.cmp(&b.label));
    assert_eq!(four.len(), 2);
    let ca = build_point_defects(&ss, &[DefectSpec::disclination(90.0, four[0].point)], None, None).unwrap();
    let cb = build_point_defects(&ss, &[DefectSpec::disclination(90.0, four[1].point)], None, None).unwrap();
    let cmp = compare_configurations("p4g", &ss, Frame2D::default(), &ca, &cb, 2).unwrap();
    assert!(!cmp.same_charge && cmp.same_charge_up_to_mirror && cmp.same_core);

    // 2. Square equivalence classes: [pair, b=(1,0), b=(0,1), +90@(0,0), +90@(3,-1), +90@(0.5,0.5)] -> [0,0,0,1,1,2].
    let sq = square_cell();
    let cfgs = [
        build_point_defects(&sq, &[DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.5, 0.5])], None, None).unwrap(),
        build_point_defects(&sq, &[DefectSpec::dislocation([1.0, 0.0], Some([0.371, 0.413]))], None, None).unwrap(),
        build_point_defects(&sq, &[DefectSpec::dislocation([0.0, 1.0], Some([2.29, -1.37]))], None, None).unwrap(),
        build_point_defects(&sq, &[DefectSpec::disclination(90.0, [0.0, 0.0])], None, None).unwrap(),
        build_point_defects(&sq, &[DefectSpec::disclination(90.0, [3.0, -1.0])], None, None).unwrap(),
        build_point_defects(&sq, &[DefectSpec::disclination(90.0, [0.5, 0.5])], None, None).unwrap(),
    ];
    let cls = classify_configurations("p4m", &sq, Frame2D::default(), &cfgs, 2).unwrap();
    assert_eq!(cls, vec![0, 0, 0, 1, 1, 2]);

    // 3. Negative control A: separated dislocation dipole has vacuum total charge but non-empty core graph.
    let dip_specs = [
        DefectSpec::Dislocation { b: [1.0, 0.0], at: Some([-3.63, 0.41]), cut: Some([0.0, 1.0]) },
        DefectSpec::Dislocation { b: [-1.0, 0.0], at: Some([3.37, 0.41]), cut: Some([0.0, -1.0]) },
    ];
    let dip = build_point_defects(&sq, &dip_specs, None, None).unwrap();
    let dip_rep = equivalent_single("p4m", &sq, Frame2D::default(), &dip_specs, &dip, 2, 1, false).unwrap();
    assert_eq!(dip_rep.kind, "vacuum");
    assert!(!dip_rep.equivalent && dip_rep.core_nodes > 0);
    assert_eq!(dip_rep.verdict, "not equivalent to the single defect");

    // 4. Negative control B: two +90 disclinations far apart -> +180 about (6, 6), 2 components != single +180 core.
    let two_specs = [
        DefectSpec::disclination(90.0, [0.0, 0.0]),
        DefectSpec::disclination(90.0, [12.0, 0.0]),
    ];
    let two = build_point_defects(&sq, &two_specs, None, None).unwrap();
    let two_rep = equivalent_single("p4m", &sq, Frame2D::default(), &two_specs, &two, 2, 1, false).unwrap();
    assert_eq!(two_rep.kind, "disclination");
    assert!((two_rep.charge.angle - 180.0).abs() < 1e-6);
    assert!(!two_rep.equivalent);
    assert_eq!(two_rep.candidates[0].name, "single +180 deg at [6.0, 6.0]");
}

#[test]
fn stage5_notebook_config_json_roundtrip() {
    let doc = NotebookConfigJson {
        format: "papertilling.config.v1".to_string(),
        lattice_label: "square".to_string(),
        group: "p4m".to_string(),
        recipe: Recipe {
            group: "p4m".to_string(),
            a: 1.0,
            b: None,
            gamma: None,
            sites: vec![[0.0, 0.0]],
            bonds: vec![[[0.0, 0.0], [1.0, 0.0]]],
        },
        frame: Frame2D::default(),
        defects: vec![
            DefectSpec::disclination(90.0, [0.0, 0.0]),
            DefectSpec::dislocation([1.0, 0.0], Some([2.5, 0.5])),
        ],
        options: ConfigOptionsJson::default(),
        summary: Some(ConfigSummaryJson {
            charge: "disclination D(1a, +90 deg) about [0.0, 0.5]".to_string(),
            kind: "disclination".to_string(),
            equivalent: false,
            matches: Vec::new(),
            volterra_matches: Vec::new(),
            completion_matches: Vec::new(),
            match_tier: None,
            verdict: "not equivalent to the single defect".to_string(),
        }),
    };
    let s = doc.to_json_pretty().unwrap();
    assert!(s.contains("\"papertilling.config.v1\"") && s.contains("\"angle\": 90.0"));
    let parsed = NotebookConfigJson::from_json_str(&s).unwrap();
    assert_eq!(parsed, doc);
}

#[test]
fn stage5_live_update_500_sites_bench() {
    // R = 13.0 on unit square lattice encloses pi * 13^2 ~= 531 sites (450 <= N <= 650).
    let sq = square_cell();
    let lat_info = LatticeCoreInfo::new(&sq);
    let specs = [
        DefectSpec::disclination(90.0, [-1.0, 0.0]),
        DefectSpec::disclination(-90.0, [1.5, 0.5]),
        DefectSpec::dislocation([1.0, 0.0], Some([0.371, -2.587])),
    ];
    let mut best_ms = f64::INFINITY;
    let (mut n_sites, mut q_kind, mut n_core) = (0, String::new(), 0);
    for _ in 0..5 {
        let t0 = Instant::now();
        let mut crystal = build_point_defects(&sq, &specs, Some(12.4), Some([0.0, 0.0])).unwrap();
        relax_picture(&mut crystal);
        let q = classify_crystal_charge("p4m", &sq, Frame2D::default(), &crystal).unwrap();
        let h = core_graph(&crystal, &lat_info, 2).unwrap();
        best_ms = best_ms.min(t0.elapsed().as_secs_f64() * 1000.0);
        (n_sites, q_kind, n_core) = (crystal.pos.len(), q.kind, h.node_count());
    }
    println!("# time 500-site live update: {best_ms:.2} ms ({n_sites} sites)");
    assert!((450..=650).contains(&n_sites), "expected 450..=650 sites, got {n_sites}");
    assert_eq!(q_kind, "dislocation");
    assert!(n_core > 0);
    if cfg!(not(debug_assertions)) {
        assert!(best_ms < 100.0, "500-site live update took {best_ms:.2} ms (>= 100 ms target)");
    }
}

#[derive(Deserialize)]
struct GateCell {
    a1: [f64; 2],
    a2: [f64; 2],
    basis: Vec<[f64; 2]>,
    bonds: Vec<[i64; 5]>,
}

impl GateCell {
    fn to_cell2d(&self) -> Cell2D {
        Cell2D {
            a1: self.a1,
            a2: self.a2,
            basis: self.basis.clone(),
            bonds: self
                .bonds
                .iter()
                .map(|b| Bond2D {
                    i: b[0] as usize,
                    j: b[1] as usize,
                    n1: b[2] as i32,
                    n2: b[3] as i32,
                    kind: b[4] as usize,
                })
                .collect(),
        }
    }
}

#[derive(Deserialize)]
struct GateCharge {
    kind: String,
    angle: f64,
    label: String,
    b: Option<[f64; 2]>,
    lattice_vector: Option<bool>,
    b_rep: Option<[f64; 2]>,
    length: Option<f64>,
    centre: Option<[f64; 2]>,
    centre_class: Option<String>,
}

#[derive(Deserialize)]
struct GateGraphSummary {
    nodes: usize,
    edges: usize,
    degrees: Vec<u16>,
    components: usize,
    bipartite: bool,
}

#[derive(Deserialize)]
struct GateCoreFace {
    kind: String,
    ring: Option<usize>,
    coordination: Vec<usize>,
    label: String,
    angle: Option<f64>,
    b: Option<[f64; 2]>,
}

#[derive(Deserialize)]
struct GateScan {
    grid: usize,
    #[serde(rename = "R")]
    r_patch: f64,
    builds: usize,
    anomalies: usize,
    capped: usize,
    max_candidates: usize,
    lab_cores: usize,
    other_class: usize,
    skipped_rim: usize,
    distinct: usize,
}

#[derive(Deserialize)]
struct GateCandidate {
    name: String,
    volterra: Option<bool>,
    rings: Option<Vec<usize>>,
    under: Option<Vec<usize>>,
    added: Option<usize>,
    lattice_bonds: Option<usize>,
    reconstruction_bonds: Option<usize>,
    bipartite: Option<bool>,
    description: Option<String>,
    also: Option<Vec<String>>,
    faces_ok: Option<bool>,
    iso_ok: Option<bool>,
}

#[derive(Deserialize)]
struct GateCase {
    name: String,
    group: String,
    cell: GateCell,
    specs: Vec<DefectSpec>,
    grid: usize,
    charge: GateCharge,
    core_graph: GateGraphSummary,
    cores: Vec<GateCoreFace>,
    equivalent: bool,
    matches: Vec<String>,
    verdict: String,
    scan: Option<GateScan>,
    candidates: Vec<GateCandidate>,
}

#[derive(Deserialize)]
struct GateMirrorP4g {
    group: String,
    cell: GateCell,
    at_a: [f64; 2],
    at_b: [f64; 2],
    same_charge: bool,
    same_charge_up_to_mirror: bool,
    same_core: bool,
}

#[derive(Deserialize)]
struct GateStage5 {
    cases: Vec<GateCase>,
    classify_square: Vec<usize>,
    mirror_p4g: GateMirrorP4g,
}

fn assert_vec2_close(a: [f64; 2], b: [f64; 2], tol: f64, ctx: &str) {
    assert!((a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol, "{ctx}: {a:?} != {b:?}");
}

#[test]
fn stage5_python_gate() {
    let Ok(gate_path) = std::env::var("PT_GATE_STAGE5") else {
        println!("PT_GATE_STAGE5 not set; skipping external Stage 5 Python gate");
        return;
    };
    let raw = std::fs::read_to_string(&gate_path).expect("read PT_GATE_STAGE5");
    let gate: GateStage5 = serde_json::from_str(&raw).expect("parse PT_GATE_STAGE5");
    assert_eq!(gate.cases.len(), 10);

    let t0 = Instant::now();
    for tc in &gate.cases {
        let cell = tc.cell.to_cell2d();
        let lat_info = LatticeCoreInfo::new(&cell);
        let crystal = build_point_defects(&cell, &tc.specs, None, None)
            .unwrap_or_else(|e| panic!("{}: build failed: {e}", tc.name));
        let rep = equivalent_single(&tc.group, &cell, Frame2D::default(), &tc.specs, &crystal, 2, tc.grid, false)
            .unwrap_or_else(|e| panic!("{}: equivalent_single failed: {e}", tc.name));

        assert_eq!(rep.charge.kind, tc.charge.kind, "{}: charge.kind", tc.name);
        assert!((rep.charge.angle - tc.charge.angle).abs() < 1e-5, "{}: charge.angle", tc.name);
        assert_eq!(rep.charge.label, tc.charge.label, "{}: charge.label", tc.name);
        assert_eq!(rep.charge.lattice_vector, tc.charge.lattice_vector, "{}: lattice_vector", tc.name);
        if let (Some(gb), Some(eb)) = (rep.charge.b, tc.charge.b) {
            assert_vec2_close(gb, eb, 1e-5, &format!("{} charge.b", tc.name));
        }
        if let (Some(gr), Some(er)) = (rep.charge.b_rep, tc.charge.b_rep) {
            assert_vec2_close(gr, er, 1e-5, &format!("{} charge.b_rep", tc.name));
        }
        if let (Some(gl), Some(el)) = (rep.charge.length, tc.charge.length) {
            assert!((gl - el).abs() < 1e-5, "{}: charge.length", tc.name);
        }
        if let (Some(gc), Some(ec)) = (rep.charge.centre, tc.charge.centre) {
            assert_vec2_close(gc, ec, 1e-5, &format!("{} charge.centre", tc.name));
        }
        assert_eq!(rep.charge.centre_class, tc.charge.centre_class, "{}: centre_class", tc.name);

        let h = core_graph(&crystal, &lat_info, 2).unwrap();
        assert_eq!(h.node_count(), tc.core_graph.nodes, "{}: core_graph.nodes", tc.name);
        assert_eq!(h.edge_count(), tc.core_graph.edges, "{}: core_graph.edges", tc.name);
        let mut degs: Vec<u16> = h.node_weights().copied().collect();
        degs.sort_unstable();
        assert_eq!(degs, tc.core_graph.degrees, "{}: core_graph.degrees", tc.name);
        let comps = if h.node_count() == 0 { 0 } else { connected_components(&h) };
        assert_eq!(comps, tc.core_graph.components, "{}: core_graph.components", tc.name);
        assert_eq!(is_bipartite(&h), tc.core_graph.bipartite, "{}: core_graph.bipartite", tc.name);

        let cores = inspect_crystal_cores(&crystal);
        assert_eq!(cores.len(), tc.cores.len(), "{}: cores len", tc.name);
        for (gc, ec) in cores.iter().zip(&tc.cores) {
            assert_eq!(gc.kind, ec.kind, "{}: core kind", tc.name);
            assert_eq!(gc.ring, ec.ring, "{}: core ring", tc.name);
            assert_eq!(gc.coordination, ec.coordination, "{}: core coordination", tc.name);
            assert_eq!(gc.label, ec.label.replace("-0.0", "0.0"), "{}: core label", tc.name);
            assert_eq!(gc.angle, ec.angle, "{}: core angle", tc.name);
            if let (Some(gb), Some(eb)) = (gc.b, ec.b) {
                assert_vec2_close(gb, eb, 1e-5, &format!("{} core.b", tc.name));
            }
        }

        assert_eq!(rep.equivalent, tc.equivalent, "{}: equivalent", tc.name);
        assert_eq!(rep.matches, tc.matches, "{}: matches", tc.name);
        assert_eq!(rep.verdict, tc.verdict, "{}: verdict", tc.name);

        if let Some(ref esc) = tc.scan {
            let gsc = rep.scan.as_ref().expect("expected scan info");
            assert_eq!(gsc.grid, esc.grid, "{}: scan.grid", tc.name);
            assert!((gsc.r_patch - esc.r_patch).abs() < 1e-5, "{}: scan.r_patch", tc.name);
            assert_eq!(gsc.builds, esc.builds, "{}: scan.builds", tc.name);
            assert_eq!(gsc.anomalies, esc.anomalies, "{}: scan.anomalies", tc.name);
            assert_eq!(gsc.capped, esc.capped, "{}: scan.capped", tc.name);
            assert_eq!(gsc.max_candidates, esc.max_candidates, "{}: scan.max_candidates", tc.name);
            assert_eq!(gsc.lab_cores, esc.lab_cores, "{}: scan.lab_cores", tc.name);
            assert_eq!(gsc.other_class, esc.other_class, "{}: scan.other_class", tc.name);
            assert_eq!(gsc.skipped_rim, esc.skipped_rim, "{}: scan.skipped_rim", tc.name);
            assert_eq!(gsc.distinct, esc.distinct, "{}: scan.distinct", tc.name);
        }

        assert_eq!(rep.candidates.len(), tc.candidates.len(), "{}: candidates len", tc.name);
        for (idx, (gc, ec)) in rep.candidates.iter().zip(&tc.candidates).enumerate() {
            assert_eq!(gc.name, ec.name, "{} cand {idx} name", tc.name);
            assert_eq!(gc.volterra, ec.volterra, "{} cand {idx} volterra", tc.name);
            if let Some(ref er) = ec.rings {
                assert_eq!(&gc.rings, er, "{} cand {idx} rings", tc.name);
            }
            if let Some(ref eu) = ec.under {
                assert_eq!(&gc.under, eu, "{} cand {idx} under", tc.name);
            }
            if let Some(ea) = ec.added {
                assert_eq!(gc.added, ea, "{} cand {idx} added", tc.name);
                assert_eq!(gc.lattice_bonds, ec.lattice_bonds.unwrap(), "{} cand {idx} lattice_bonds", tc.name);
                assert_eq!(gc.reconstruction_bonds, ec.reconstruction_bonds.unwrap(), "{} cand {idx} recon", tc.name);
            }
            assert_eq!(gc.bipartite, ec.bipartite, "{} cand {idx} bipartite", tc.name);
            if let Some(ref edesc) = ec.description {
                let ge = &rep.gallery[idx];
                assert_eq!(&ge.description, edesc, "{} gal {idx} description", tc.name);
                assert_eq!(&ge.also, ec.also.as_ref().unwrap(), "{} gal {idx} also", tc.name);
                assert_eq!(Some(ge.faces_ok), ec.faces_ok, "{} gal {idx} faces_ok", tc.name);
                assert_eq!(ge.iso_ok, ec.iso_ok, "{} gal {idx} iso_ok", tc.name);
            }
        }
    }

    let sq = square_cell();
    let sq_cfgs = [
        build_point_defects(&sq, &[DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.5, 0.5])], None, None).unwrap(),
        build_point_defects(&sq, &[DefectSpec::dislocation([1.0, 0.0], Some([0.371, 0.413]))], None, None).unwrap(),
        build_point_defects(&sq, &[DefectSpec::dislocation([0.0, 1.0], Some([2.29, -1.37]))], None, None).unwrap(),
        build_point_defects(&sq, &[DefectSpec::disclination(90.0, [0.0, 0.0])], None, None).unwrap(),
        build_point_defects(&sq, &[DefectSpec::disclination(90.0, [3.0, -1.0])], None, None).unwrap(),
        build_point_defects(&sq, &[DefectSpec::disclination(90.0, [0.5, 0.5])], None, None).unwrap(),
    ];
    let got_cls = classify_configurations("p4m", &sq, Frame2D::default(), &sq_cfgs, 2).unwrap();
    assert_eq!(got_cls, gate.classify_square, "classify_square mismatch");

    let mp = &gate.mirror_p4g;
    let mp_cell = mp.cell.to_cell2d();
    let ma = build_point_defects(&mp_cell, &[DefectSpec::disclination(90.0, mp.at_a)], None, None).unwrap();
    let mb = build_point_defects(&mp_cell, &[DefectSpec::disclination(90.0, mp.at_b)], None, None).unwrap();
    let mcmp = compare_configurations(&mp.group, &mp_cell, Frame2D::default(), &ma, &mb, 2).unwrap();
    assert_eq!(mcmp.same_charge, mp.same_charge, "mirror_p4g same_charge");
    assert_eq!(mcmp.same_charge_up_to_mirror, mp.same_charge_up_to_mirror, "mirror_p4g mirror");
    assert_eq!(mcmp.same_core, mp.same_core, "mirror_p4g same_core");

    println!("# time stage5_python_gate (10 cases + classify + mirror): {:.3} s", t0.elapsed().as_secs_f64());
}
