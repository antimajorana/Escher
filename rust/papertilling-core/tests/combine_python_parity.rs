//! Stage 5 parity against the pure-Python package: `build_point_defects` + `classify_crystal_charge` +
//! `core_graph` + `equivalent_single` on the Notebook 3 configurations, the controls of tests/test_config.py and
//! tests/test_cores.py, and the surgery edge cases (Frank sums >= 360, coincident defects, non-lattice b).
//!
//! Expected values: tests/fixtures/combine_parity_py.json, computed by the Python package
//! (`PAPERTILLING_NO_RUST=1 python tools/checks/combine_parity_dump.py`), never by this code.

use papertilling_core::defect::combine_api::{
    classify_crystal_charge, equivalent_single, inspect_crystal_cores, nearest_allowed_centre,
};
use papertilling_core::defect::core_graph::{LatticeCoreInfo, core_graph};
use papertilling_core::defect::equivalence::CentreSnapper;
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::{Cell2D, Frame2D, Recipe};
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/combine_parity_py.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("read combine_parity_py.json")).expect("parse fixture")
}

fn lattice_cell(lat: &Value) -> (String, Cell2D, Frame2D) {
    let recipe: Recipe = serde_json::from_value(lat["recipe"].clone()).expect("recipe");
    let frame: Frame2D = serde_json::from_value(lat["frame"].clone()).expect("frame");
    let cell = LatticeFamily::from_recipe(&recipe, Some(frame)).expect("family").cell();
    (lat["group"].as_str().unwrap().to_string(), cell, frame)
}

fn v2(v: &Value) -> [f64; 2] {
    [v[0].as_f64().unwrap(), v[1].as_f64().unwrap()]
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

/// Python prints -0.0 as "-0.0" inside labels; Rust formats the same rounded value as "0.0".
fn norm_label(s: &str) -> String {
    s.replace("-0.0,", "0.0,").replace("-0.0]", "0.0]")
}

/// Compare one case; returns the list of mismatches (empty = parity).
fn check_case(case: &Value) -> Vec<String> {
    let name = case["name"].as_str().unwrap();
    let (group, cell, frame) = lattice_cell(&case["lattice"]);
    let specs: Vec<DefectSpec> = serde_json::from_value(case["specs"].clone()).expect("specs");
    let py = &case["python"];
    let mut bad = Vec::new();
    let built = build_point_defects(&cell, &specs, None, None);
    if !py["ok"].as_bool().unwrap() {
        let py_err = py["error"].as_str().unwrap();
        match built {
            Ok(_) => bad.push(format!("{name}: Python refuses ({py_err}) but Rust builds")),
            // the refusal text is what the Combine tab shows in its note banner: same wording as Python
            Err(e) if e != py_err => bad.push(format!("{name}: refusal text differs: Python '{py_err}' vs Rust '{e}'")),
            Err(_) => {}
        }
        return bad;
    }
    let crystal = match built {
        Ok(c) => c,
        Err(e) => return vec![format!("{name}: Python builds but Rust refuses: {e}")],
    };
    let pc = &py["charge"];
    match classify_crystal_charge(&group, &cell, frame, &crystal) {
        Err(e) => bad.push(format!("{name}: charge failed: {e}")),
        Ok(q) => {
            if q.kind != pc["kind"].as_str().unwrap() {
                bad.push(format!("{name}: kind {} vs Python {}", q.kind, pc["kind"]));
            }
            if (q.angle - pc["angle"].as_f64().unwrap()).abs() > 1e-5 {
                bad.push(format!("{name}: angle {} vs Python {}", q.angle, pc["angle"]));
            }
            if norm_label(&q.label) != norm_label(pc["label"].as_str().unwrap()) {
                bad.push(format!("{name}: label '{}' vs Python '{}'", q.label, pc["label"]));
            }
            if let (Some(b), Some(pb)) = (q.b, pc.get("b")) {
                let pb = v2(pb);
                if (b[0] - pb[0]).abs() > 1e-5 || (b[1] - pb[1]).abs() > 1e-5 {
                    bad.push(format!("{name}: b {b:?} vs Python {pb:?}"));
                }
            }
            if let Some(cls) = pc.get("centre_class") {
                if q.centre_class.as_deref() != cls.as_str() {
                    bad.push(format!("{name}: centre class {:?} vs Python {cls}", q.centre_class));
                }
                let (c, pcen) = (q.centre.unwrap_or([f64::NAN; 2]), v2(&pc["centre"]));
                if (c[0] - pcen[0]).abs() > 1e-5 || (c[1] - pcen[1]).abs() > 1e-5 {
                    bad.push(format!("{name}: centre {c:?} vs Python {pcen:?}"));
                }
            }
        }
    }
    let cores: Vec<String> = inspect_crystal_cores(&crystal).into_iter().map(|c| norm_label(&c.label)).collect();
    let py_cores: Vec<String> = strs(&py["cores"]).iter().map(|s| norm_label(s)).collect();
    if cores != py_cores {
        bad.push(format!("{name}: local cores {cores:?} vs Python {py_cores:?}"));
    }
    let lat_info = LatticeCoreInfo::new(&cell);
    let h = core_graph(&crystal, &lat_info, 2);
    if let Some(err) = py.get("core_error") {
        if h.is_ok() {
            bad.push(format!("{name}: Python core refused ({err}) but Rust core graph exists"));
        }
        return bad;
    }
    match h {
        Err(e) => bad.push(format!("{name}: core graph failed: {e}")),
        Ok(h) => {
            let mut degs: Vec<u64> = h.node_weights().map(|&d| d as u64).collect();
            degs.sort_unstable();
            let pcore = &py["core"];
            let py_degs: Vec<u64> = pcore["degrees"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
            if h.node_count() as u64 != pcore["nodes"].as_u64().unwrap() || h.edge_count() as u64 != pcore["edges"].as_u64().unwrap() || degs != py_degs {
                bad.push(format!("{name}: core ({}, {}) vs Python ({}, {})", h.node_count(), h.edge_count(), pcore["nodes"], pcore["edges"]));
            }
        }
    }
    let grid = case["grid"].as_u64().unwrap() as usize;
    match equivalent_single(&group, &cell, frame, &specs, &crystal, 2, grid, false) {
        Err(e) => bad.push(format!("{name}: equivalent_single failed: {e}")),
        Ok(rep) => {
            let is_disloc = rep.kind == "dislocation";
            let is_p4g_scan = is_disloc && group == "p4g";
            let is_non_c2_scan = is_disloc && matches!(group.as_str(), "p3" | "p31m");
            if is_p4g_scan || is_non_c2_scan {
                // Legacy Python `fast.dislocation_fast` removes a strip for both cut senses, yielding -b when
                // cross2(d, b) > 0 (rejected as `other_class` on p3/p31m and shifting grid=2 cores on p4g).
                // Fixed Rust builds +b via 4-chart insertion for cross2(d, b) > 0 (`other_class == 0`).
                let sc = rep.scan.as_ref().expect("dislocation scan info");
                if sc.other_class != 0 {
                    bad.push(format!("{name}: expected other_class == 0, got {}", sc.other_class));
                }
                if rep.candidates.is_empty() || rep.candidates.len() != sc.distinct {
                    bad.push(format!("{name}: inconsistent scanned candidates {} vs distinct {}", rep.candidates.len(), sc.distinct));
                }
                if is_p4g_scan {
                    if rep.equivalent != py["equivalent"].as_bool().unwrap() {
                        bad.push(format!("{name}: equivalent {} vs Python {}", rep.equivalent, py["equivalent"]));
                    }
                    if rep.matches.iter().map(|s| norm_label(s)).collect::<Vec<_>>() != strs(&py["matches"]).iter().map(|s| norm_label(s)).collect::<Vec<_>>() {
                        bad.push(format!("{name}: matches {:?} vs Python {}", rep.matches, py["matches"]));
                    }
                } else {
                    // On p3/p31m, the two adjacent +120/-120 dipoles now match genuine +b completion cores
                    // that legacy Python missed; the two next-cell dipoles remain non-equivalent.
                    let want_eq = matches!(name, "p3 +120 1b / -120 1c" | "p31m +120 2b(i) / -120 2b(ii)");
                    if rep.equivalent != want_eq {
                        bad.push(format!("{name}: expected equivalent={want_eq}, got {}", rep.equivalent));
                    }
                }
            } else {
                if rep.equivalent != py["equivalent"].as_bool().unwrap() {
                    bad.push(format!("{name}: equivalent {} vs Python {}", rep.equivalent, py["equivalent"]));
                }
                if rep.matches.iter().map(|s| norm_label(s)).collect::<Vec<_>>() != strs(&py["matches"]).iter().map(|s| norm_label(s)).collect::<Vec<_>>() {
                    bad.push(format!("{name}: matches {:?} vs Python {}", rep.matches, py["matches"]));
                }
                if norm_label(&rep.verdict) != norm_label(py["verdict"].as_str().unwrap()) {
                    bad.push(format!("{name}: verdict '{}' vs Python '{}'", rep.verdict, py["verdict"]));
                }
                let names: Vec<String> = rep.candidates.iter().map(|c| norm_label(&c.name)).collect();
                if names != strs(&py["candidates"]).iter().map(|s| norm_label(s)).collect::<Vec<_>>() {
                    bad.push(format!("{name}: candidates {names:?} vs Python {}", py["candidates"]));
                }
            }
        }
    }
    bad
}

#[test]
fn centre_snapper_matches_python_nearest_centre() {
    // Python LatticeData.nearest_centre(angle, near) and centre_class(point)["label"] (fixture `snap`)
    let fx = fixture();
    let mut n = 0;
    for snap in fx["snap"].as_array().unwrap() {
        let (group, cell, frame) = lattice_cell(&snap["lattice"]);
        let snapper = CentreSnapper::new(&group, &cell, frame);
        for q in snap["nearest"].as_array().unwrap() {
            let (ang, near, py_pt) = (q["angle"].as_f64().unwrap(), v2(&q["near"]), v2(&q["point"]));
            let (p, lbl) = snapper.nearest(ang, near).unwrap();
            let d = |x: [f64; 2]| (x[0] - near[0]).hypot(x[1] - near[1]);
            assert!((d(p) - d(py_pt)).abs() < 1e-9, "{group} {ang} {near:?}: {p:?} vs Python {py_pt:?}");
            if (p[0] - py_pt[0]).hypot(p[1] - py_pt[1]) < 1e-9 {
                assert_eq!(lbl, q["label"].as_str().unwrap(), "{group} {ang} {near:?}");
            }
            assert_eq!(nearest_allowed_centre(&group, &cell, frame, ang, near).unwrap(), (p, lbl));
            n += 1;
        }
        assert!(snapper.nearest(0.0, [0.0, 0.0]).is_err() && snapper.nearest(37.0, [0.0, 0.0]).is_err());
    }
    assert!(n > 200, "{n} snaps checked");
}

#[test]
fn combine_cases_match_python() {
    let fx = fixture();
    let cases = fx["cases"].as_array().unwrap();
    assert!(cases.len() >= 30, "fixture has {} cases", cases.len());
    let t0 = std::time::Instant::now();
    let mut bad = Vec::new();
    for case in cases {
        let t = std::time::Instant::now();
        let b = check_case(case);
        println!("{:45} {} ({:.2} s)", case["name"].as_str().unwrap(), if b.is_empty() { "ok" } else { "MISMATCH" }, t.elapsed().as_secs_f64());
        bad.extend(b);
    }
    println!("# time combine parity: {:.1} s", t0.elapsed().as_secs_f64());
    assert!(bad.is_empty(), "{} mismatch(es) with Python:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn malformed_numeric_specs_and_patch_extents_refuse_without_panicking() {
    let fx = fixture();
    let (_, cell, _) = lattice_cell(&fx["cases"][0]["lattice"]);
    let bad = [
        DefectSpec::dislocation([f64::NAN, 0.0], None),
        DefectSpec::dislocation([1e100, 0.0], None),
        DefectSpec::dislocation(cell.a1, Some([f64::INFINITY, 0.0])),
        DefectSpec::Dislocation { b: cell.a1, at: None, cut: Some([0.0, 0.0]) },
        DefectSpec::disclination(f64::NAN, [0.0, 0.0]),
    ];
    for spec in bad {
        assert!(build_point_defects(&cell, &[spec], None, None).is_err());
    }
    let good = [DefectSpec::dislocation(cell.a1, None)];
    for radius in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1e100] {
        assert!(build_point_defects(&cell, &good, Some(radius), None).is_err());
    }
    let mut invalid = cell;
    invalid.a1 = invalid.a2;
    assert!(build_point_defects(&invalid, &good, None, None).is_err());
}
