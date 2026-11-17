//! Rust-written `papertilling.config.v1` files for the Python import test (tests/test_config_rust_json.py).
//!
//! Regenerate (only when the writer or the configurations change):
//!     PT_WRITE_FIXTURES=1 cargo test --release -p papertilling-core --test config_json_rust_fixtures
//! Without the variable the test checks that the committed files equal what Rust writes now. The `summary` is
//! Rust's `equivalent_single` on the Rust-built crystal (as the app's Export does); the Python test recomputes it.

use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::combine_api::equivalent_single;
use papertilling_core::defect::combine_types::{ConfigOptionsJson, ConfigSummaryJson, NotebookConfigJson};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::types::{Frame2D, Recipe};

fn fixture_path(name: &str) -> String {
    format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Recipe and frame of a named lattice, as Python wrote them in a config_py_*.json file.
fn python_lattice(file: &str) -> (String, Recipe, Frame2D) {
    let cfg = NotebookConfigJson::from_json_str(&std::fs::read_to_string(fixture_path(file)).unwrap()).unwrap();
    (cfg.lattice_label, cfg.recipe, cfg.frame)
}

fn with_summary(mut cfg: NotebookConfigJson) -> NotebookConfigJson {
    let cell = LatticeFamily::from_recipe(&cfg.recipe, Some(cfg.frame)).unwrap().cell();
    let crystal = build_point_defects(&cell, &cfg.defects, cfg.options.r_patch, None).unwrap();
    let rep = equivalent_single(&cfg.group, &cell, cfg.frame, &cfg.defects, &crystal, cfg.options.radius, cfg.options.grid, false).unwrap();
    cfg.summary = Some(ConfigSummaryJson {
        charge: rep.charge.label.clone(),
        kind: rep.kind.clone(),
        equivalent: rep.equivalent,
        matches: rep.matches.clone(),
        volterra_matches: Vec::new(),
        completion_matches: Vec::new(),
        match_tier: None,
        verdict: rep.verdict.clone(),
    });
    cfg
}

fn config(label: String, recipe: Recipe, frame: Frame2D, defects: Vec<DefectSpec>, options: ConfigOptionsJson) -> NotebookConfigJson {
    with_summary(NotebookConfigJson {
        format: "papertilling.config.v1".to_string(),
        lattice_label: label,
        group: recipe.group.clone(),
        recipe,
        frame,
        defects,
        options,
        summary: None,
    })
}

fn rust_configs() -> Vec<(&'static str, NotebookConfigJson)> {
    let opts = |r_patch: Option<f64>, picture: &str, grid: usize| ConfigOptionsJson { r_patch, picture: picture.to_string(), grid, radius: 2 };
    let mut out = Vec::new();

    // square: +90 at a site and -90 at a face centre (C01 control), auto radius (no r_patch written)
    let (label, recipe, frame) = python_lattice("config_py_square_pair.json");
    let pair = vec![DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.5, 0.5])];
    out.push(("config_rust_square_pair.json", config(label, recipe, frame, pair, opts(None, "relaxed", 1))));

    // Shastry-Sutherland (frame origin (0.5, -0.5)): +90 at 2a(i) and a dislocation with no `at`, an explicit cut
    let (label, recipe, frame) = python_lattice("config_py_shastry_sutherland.json");
    let ss = vec![
        DefectSpec::disclination(90.0, [0.5, 1.5]),
        DefectSpec::Dislocation { b: [2.0, 0.0], at: None, cut: Some([0.3, 1.0]) },
    ];
    out.push(("config_rust_shastry_sutherland.json", config(label, recipe, frame, ss, opts(Some(14.0), "raw", 1))));

    // honeycomb (frame with origin (sqrt3/2, 1/2) and a 1e-16 entry in q): +60 / -60 at neighbouring hexagon centres
    let (label, recipe, frame) = python_lattice("config_py_honeycomb_cut.json");
    let cell = LatticeFamily::from_recipe(&recipe, Some(frame)).unwrap().cell();
    let cat = DefectCatalogue::new(&recipe.group, &cell, frame);
    let c6 = cat.centre_classes.iter().find(|c| c.order == 6).expect("6-fold centre").point;
    let a1 = cat.cell.a1;
    let hc = vec![DefectSpec::disclination(60.0, c6), DefectSpec::disclination(-60.0, [c6[0] + a1[0], c6[1] + a1[1]])];
    out.push(("config_rust_honeycomb_57.json", config(label, recipe, frame, hc, opts(None, "raw", 2))));
    out
}

#[test]
fn rust_written_config_fixtures_are_current() {
    let write = std::env::var("PT_WRITE_FIXTURES").is_ok_and(|v| v == "1");
    for (file, cfg) in rust_configs() {
        let path = fixture_path(file);
        if write {
            std::fs::write(&path, cfg.to_json_pretty().unwrap() + "\n").unwrap();
        }
        let committed = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{file} missing: run with PT_WRITE_FIXTURES=1"));
        assert_eq!(NotebookConfigJson::from_json_str(&committed).unwrap(), cfg, "{file} is stale: run with PT_WRITE_FIXTURES=1");
        // what Python must be able to read: recipe b/gamma as null, no r_patch key when automatic
        assert!(committed.contains("\"angle\":") && committed.contains("\"papertilling.config.v1\""));
    }
}
