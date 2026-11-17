//! Notebook 3 `.json` files written by Python (`papertilling.defects.config.save_config`) opened by the app:
//! identical recipe / frame / defects / options, the lattice found again, and the same charge and verdict as
//! the `summary` Python wrote. Fixtures: tests/fixtures/config_py_*.json (tools/checks/combine_parity_dump.py).

use serde_json::Value;

use super::*;
use crate::backend::AppBackend;
use papertilling_core::defect::combine_api::{classify_crystal_charge, equivalent_single};
use papertilling_core::defect::surgery::build_point_defects;
use papertilling_core::types::{Frame2D, Recipe};

const PY_FILES: [&str; 3] = ["config_py_square_pair.json", "config_py_shastry_sutherland.json", "config_py_honeycomb_cut.json"];

fn read(name: &str) -> String {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(path).expect("read fixture")
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn v2(v: &Value) -> [f64; 2] {
    [f(&v[0]), f(&v[1])]
}

#[test]
fn python_written_configs_load_with_identical_fields() {
    for name in PY_FILES {
        let raw_s = read(name);
        let raw: Value = serde_json::from_str(&raw_s).unwrap();
        let cfg = NotebookConfigJson::from_json_str(&raw_s).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(cfg.format, raw["format"].as_str().unwrap());
        assert_eq!(cfg.lattice_label, raw["lattice_label"].as_str().unwrap());
        assert_eq!(cfg.group, raw["group"].as_str().unwrap());
        let recipe: Recipe = serde_json::from_value(raw["recipe"].clone()).unwrap();
        assert_eq!(cfg.recipe, recipe, "{name}: recipe");
        assert_eq!(cfg.recipe.a, f(&raw["recipe"]["a"]));
        let fr = &raw["frame"];
        assert_eq!(cfg.frame, Frame2D { q: [v2(&fr["q"][0]), v2(&fr["q"][1])], origin: v2(&fr["origin"]) }, "{name}: frame");
        let defects = raw["defects"].as_array().unwrap();
        assert_eq!(cfg.defects.len(), defects.len());
        for (d, r) in cfg.defects.iter().zip(defects) {
            let cut_raw = r.get("cut").map(v2);
            match *d {
                DefectSpec::Disclination { angle_deg, at, cut } => {
                    assert_eq!(r["kind"], "disclination");
                    assert_eq!((angle_deg, at, cut), (f(&r["angle"]), v2(&r["at"]), cut_raw));
                }
                DefectSpec::Dislocation { b, at, cut } => {
                    assert_eq!(r["kind"], "dislocation");
                    assert_eq!((b, at, cut), (v2(&r["b"]), r.get("at").map(v2), cut_raw));
                }
                DefectSpec::HalfWall { at, .. } => {
                    assert_eq!(r["kind"], "half_wall");
                    assert_eq!(at, v2(&r["at"]));
                }
            }
        }
        let o = &raw["options"];
        assert_eq!(cfg.options.r_patch, o["r_patch"].as_f64());
        assert_eq!(cfg.options.picture, o["picture"].as_str().unwrap());
        assert_eq!(cfg.options.grid as u64, o["grid"].as_u64().unwrap());
        assert_eq!(cfg.options.radius as u64, o["radius"].as_u64().unwrap());
        let s = cfg.summary.as_ref().expect("Python writes a summary");
        assert_eq!(s.charge, raw["summary"]["charge"].as_str().unwrap());
        assert_eq!(s.equivalent, raw["summary"]["equivalent"].as_bool().unwrap());
        // Rust -> JSON -> Rust is the identity on what Python wrote
        assert_eq!(NotebookConfigJson::from_json_str(&cfg.to_json_pretty().unwrap()).unwrap(), cfg);
    }
}

#[test]
fn python_written_configs_rebuild_to_python_summary() {
    let backend = AppBackend::new();
    for name in PY_FILES {
        let cfg = NotebookConfigJson::from_json_str(&read(name)).unwrap();
        let item = item_from_notebook_json(&cfg, backend.named_lattices()).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((item.label.as_str(), item.group.as_str()), (cfg.lattice_label.as_str(), cfg.group.as_str()));
        assert_eq!((&item.recipe, item.frame), (&cfg.recipe, cfg.frame), "{name}: lattice kept verbatim");
        assert!(item.named_key.is_some(), "{name}: the named lattice is recognised");
        let info = combine_lattice_info(&item);
        let crystal = build_point_defects(&info.cell, &cfg.defects, cfg.options.r_patch, None).unwrap_or_else(|e| panic!("{name}: {e}"));
        let q = classify_crystal_charge(&item.group, &info.cell, item.frame, &crystal).unwrap();
        let s = cfg.summary.as_ref().unwrap();
        assert_eq!((q.label.as_str(), q.kind.as_str()), (s.charge.as_str(), s.kind.as_str()), "{name}: charge (Python summary)");
        let rep = equivalent_single(&item.group, &info.cell, item.frame, &cfg.defects, &crystal, cfg.options.radius, cfg.options.grid, false).unwrap();
        assert_eq!((rep.equivalent, &rep.matches, &rep.verdict), (s.equivalent, &s.matches, &s.verdict), "{name}: verdict (Python summary)");
        // markers sit where the defects are
        for (i, d) in cfg.defects.iter().enumerate() {
            let at = match *d {
                DefectSpec::Disclination { at, .. } | DefectSpec::HalfWall { at, .. } => at,
                DefectSpec::Dislocation { at, .. } => at.expect("Python always writes at"),
            };
            assert_eq!(spec_world_pos(&info, i, d), at);
        }
    }
}

#[test]
fn hand_written_config_with_defaults_and_unknown_fields_loads() {
    // Missing group/format/frame/options/summary, `angle_deg` instead of `angle`, `at: null`, unknown keys:
    // Python's load_config accepts all of these (it reads recipe, frame, defects, options only).
    let s = r#"{
      "recipe": {"group": "p4m", "sites": [[0.0, 0.0]], "bonds": [[[0.0, 0.0], [-1.0, 0.0]]], "a": 1.0, "b": null},
      "defects": [
        {"kind": "disclination", "angle_deg": 90, "at": [0, 0], "note": "hand"},
        {"kind": "dislocation", "b": [1, 0], "at": null}
      ],
      "options": null,
      "made_by": "someone"
    }"#;
    let cfg = NotebookConfigJson::from_json_str(s).unwrap();
    assert_eq!(cfg.format, "papertilling.config.v1");
    assert_eq!((cfg.group.as_str(), cfg.frame), ("", Frame2D::default()));
    assert_eq!(cfg.options, Default::default());
    assert_eq!(cfg.defects, vec![DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::dislocation([1.0, 0.0], None)]);
    let backend = AppBackend::new();
    let item = item_from_notebook_json(&cfg, backend.named_lattices()).unwrap();
    assert_eq!(item.group, "p4m");
    assert_eq!(item.named_key.as_deref(), Some("square"));
    // the dislocation without `at` is drawn where the surgery puts it: 0.371 a1 + 0.413 a2 + 3 a1 (idx 1)
    let info = combine_lattice_info(&item);
    assert_eq!(spec_world_pos(&info, 1, &cfg.defects[1]), [3.371, 0.413]);
}
