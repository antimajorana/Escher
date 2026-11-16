//! `papertilling.config.v1` reading rules, checked against what Python's `load_config` accepts
//! (papertilling/defects/config.py: format check, frame/options optional, `angle_deg` alias, kind aliases).

use super::*;

const SQUARE_RECIPE: &str = r#"{"group": "p4m", "sites": [[0.0, 0.0]], "bonds": [[[0.0, 0.0], [-1.0, 0.0]]], "a": 1.0}"#;

fn doc(extra: &str, defects: &str) -> String {
    format!(r#"{{"recipe": {SQUARE_RECIPE}, "defects": {defects}{extra}}}"#)
}

#[test]
fn format_is_checked_like_python() {
    let ok = NotebookConfigJson::from_json_str(&doc("", "[]")).unwrap();
    assert_eq!(ok.format, CONFIG_FORMAT);
    let bad = NotebookConfigJson::from_json_str(&doc(r#", "format": "papertilling.config.v2""#, "[]"));
    assert!(bad.unwrap_err().contains("unsupported configuration format"));
}

#[test]
fn optional_and_null_fields_take_defaults() {
    let cfg = NotebookConfigJson::from_json_str(&doc(
        r#", "group": null, "lattice_label": null, "frame": null, "options": null, "summary": null, "extra": {"x": 1}"#,
        "[]",
    ))
    .unwrap();
    assert_eq!((cfg.group.as_str(), cfg.lattice_label.as_str()), ("", ""));
    assert_eq!(cfg.frame, Frame2D::default());
    assert_eq!(cfg.options, ConfigOptionsJson::default());
    assert_eq!(cfg.summary, None);
    // Python's writer defaults (config_to_dict(grid=2, radius=2), load_config picture "relaxed")
    assert_eq!((cfg.options.grid, cfg.options.radius, cfg.options.picture.as_str()), (2, 2, "relaxed"));
    let partial = NotebookConfigJson::from_json_str(&doc(r#", "options": {"r_patch": null, "grid": 3}"#, "[]")).unwrap();
    assert_eq!((partial.options.r_patch, partial.options.grid, partial.options.radius), (None, 3, 2));
}

#[test]
fn spec_spellings_python_accepts() {
    let cfg = NotebookConfigJson::from_json_str(&doc(
        "",
        r#"[{"kind": "disclination", "angle_deg": -90, "at": [0.5, 0.5], "moved_from": [0.4, 0.4]},
            {"kind": "rotation", "angle": 180, "at": [0, 0.5], "cut": [1, 0]},
            {"kind": "translation", "b": [0, 1]},
            {"kind": "dislocation", "b": [1, 0], "at": null, "cut": null}]"#,
    ))
    .unwrap();
    assert_eq!(
        cfg.defects,
        vec![
            DefectSpec::disclination(-90.0, [0.5, 0.5]),
            DefectSpec::Disclination { angle_deg: 180.0, at: [0.0, 0.5], cut: Some([1.0, 0.0]) },
            DefectSpec::dislocation([0.0, 1.0], None),
            DefectSpec::dislocation([1.0, 0.0], None),
        ]
    );
    // written back with Python's key names only: "angle", no "at"/"cut" when absent
    let s = serde_json::to_string(&cfg.defects).unwrap();
    assert!(s.contains(r#""angle":-90.0"#) && !s.contains("angle_deg") && !s.contains("null") && !s.contains("translation"));
    let missing_kind = NotebookConfigJson::from_json_str(&doc("", r#"[{"angle": 90, "at": [0, 0]}]"#));
    assert!(missing_kind.is_err());
}

#[test]
fn floats_survive_the_round_trip_bit_for_bit() {
    // values Python wrote in tests/fixtures/config_py_honeycomb_cut.json, plus edge cases
    let xs: [f64; 8] = [1.2093841455807397e-16, 0.8660254037844386, 0.4999999999999999, 1.7320508075688772, -0.0, 1e21, 5e-324, 0.1];
    for &x in &xs {
        let cfg = NotebookConfigJson::from_json_str(&doc("", &format!(r#"[{{"kind": "dislocation", "b": [1, 0], "at": [{x:?}, 0.5]}}]"#))).unwrap();
        let back = NotebookConfigJson::from_json_str(&cfg.to_json_pretty().unwrap()).unwrap();
        let DefectSpec::Dislocation { at: Some(at), .. } = back.defects[0] else { panic!("dislocation expected") };
        assert_eq!(at[0].to_bits(), x.to_bits(), "{x:?}");
    }
}
