//! Session file format and persistence: round trip, a hand-written schema-1 document, graceful failures, atomic save.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use papertilling_core::defect::combine_types::{CONFIG_FORMAT, ConfigOptionsJson, NotebookConfigJson};
use papertilling_core::defect::surgery::DefectSpec;
use papertilling_core::types::{Frame2D, Recipe};

use super::*;

pub(crate) struct TempDir(pub(crate) PathBuf);

impl TempDir {
    pub(crate) fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("papertilling-session-{}-{id}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("create test directory: {e}"),
            }
        }
    }

    fn names(&self) -> Vec<String> {
        let mut v: Vec<String> =
            std::fs::read_dir(&self.0).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        v.sort();
        v
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn key(group: &str, rule: &str, idx: &str) -> LatticeAliasKey {
    LatticeAliasKey { group: group.into(), rule: rule.into(), dsymbol_key_or_index: idx.into() }
}

/// A state with every section away from its default.
pub(crate) fn sample_state() -> SessionState {
    SessionState {
        schema_version: SESSION_SCHEMA_VERSION,
        active_tab: Some("Defects".into()),
        theme: Some("Studio".into()),
        layout: LayoutSession {
            left_rail_width: Some(333.5),
            right_inspector_open: Some(false),
            right_inspector_width: Some(281.25),
            split_ratio: Some(0.35),
        },
        png_scale: Some(4),
        lattice: Some(LatticeSession { alias_key: key("p4g", "bonds", "#0"), group: "p4g".into(), named_key: None }),
        gallery: GallerySession {
            params: BTreeMap::from([(String::from("t1"), Some(-0.5))]),
            repeat: Some(5),
            show_unit_cell: Some(false),
            overlay_symmetry: Some(true),
            overlay_voronoi: Some(true),
            browser_view: Some("Table".into()),
            filter_query: Some("kag".into()),
            filter_system: Some("Hexagonal".into()),
            filter_chirality: Some("Achiral".into()),
        },
        recent: vec![key("p6m", "named", "#3"), key("p4g", "bonds", "#0")],
        surfaces: SurfacesSession {
            target: Some("Torus".into()),
            h: Some(3),
            k: Some(2),
            genus_g: Some(3),
            centre_mode: Some("Face".into()),
            auto_size: Some(false),
            show_cover: Some(true),
        },
        defects: DefectsSession {
            kind: Some("Disclinations".into()),
            dislocation_index: Some(1),
            disclination_index: Some(2),
            wall_index: Some(0),
            wall_half: Some(true),
            radius: Some(5.2),
            relaxed: Some(false),
            draw_mode: Some("Raw".into()),
            view_mode: Some("Sheet3D".into()),
            color_mode: Some("HeightZ".into()),
        },
        combine: Some(NotebookConfigJson {
            format: String::from(CONFIG_FORMAT),
            lattice_label: String::from("p4g #1"),
            group: String::from("p4g"),
            recipe: Recipe {
                group: String::from("p4g"),
                sites: vec![[0.25, 0.25]],
                bonds: vec![[[0.25, 0.25], [0.75, 0.25]]],
                a: 1.0,
                b: None,
                gamma: None,
            },
            frame: Frame2D::default(),
            defects: vec![DefectSpec::dislocation([1.0, 0.0], Some([0.25, 0.25]))],
            options: ConfigOptionsJson {
                r_patch: Some(4.5),
                picture: String::from("relaxed"),
                grid: 2,
                radius: 2,
            },
            summary: None,
        }),
        camera: CameraSession { yaw: Some(-1.25), pitch: Some(0.75), distance: Some(5.5), offset: Some([0.25, -0.5]), z_up: Some(true) },
        atlas_sub_view: Some("HyperbolicH2".into()),
        active_mode: Some("Defects".into()),
        lattices_view: Some("Plane2D".into()),
        defects_view: Some("Catalogue2D".into()),
        analysis_target: Some("SingleDefect".into()),
        analysis_view: Some("Split2D3D".into()),
    }
}

#[test]
fn non_default_state_round_trips_through_json_unchanged() {
    let st = sample_state();
    assert_ne!(st, SessionState::default());
    let raw = st.to_json_pretty().unwrap();
    let (back, dropped) = parse_session(&raw).unwrap();
    assert_eq!(back, st);
    assert_eq!(dropped, 0);
}

#[test]
fn hand_written_schema_1_document_restores_the_written_fields_and_defaults_the_rest() {
    let raw = r##"{
        "schema_version": 1,
        "active_tab": "Surfaces",
        "theme": "Studio",
        "png_scale": 1,
        "lattice": { "alias_key": { "group": "p6m", "rule": "named", "dsymbol_key_or_index": "#7" },
                     "group": "p6m", "named_key": "kagome" },
        "gallery": { "repeat": 4, "params": { "a": 1.5 } },
        "camera": { "yaw": 0.5, "z_up": true },
        "some_future_key": [1, 2, 3]
    }"##;
    let (st, dropped) = parse_session(raw).unwrap();
    assert_eq!(dropped, 0);
    assert_eq!(st.active_tab.as_deref(), Some("Surfaces"));
    assert_eq!(st.theme.as_deref(), Some("Studio"));
    assert_eq!(st.png_scale, Some(1));
    let l = st.lattice.unwrap();
    assert_eq!((l.alias_key.group.as_str(), l.alias_key.rule.as_str(), l.alias_key.dsymbol_key_or_index.as_str()), ("p6m", "named", "#7"));
    assert_eq!(l.named_key.as_deref(), Some("kagome"));
    assert_eq!(st.gallery.repeat, Some(4));
    assert_eq!(st.gallery.params.get("a"), Some(&Some(1.5)));
    assert_eq!(st.gallery.show_unit_cell, None);
    assert_eq!((st.camera.yaw, st.camera.pitch, st.camera.z_up), (Some(0.5), None, Some(true)));
    assert_eq!(st.layout, LayoutSession::default());
    assert!(st.recent.is_empty() && st.combine.is_none());
}

#[test]
fn corrupt_empty_and_foreign_schema_documents_are_refused_without_panic() {
    for raw in ["{not json", "", "[1, 2]", "{\"theme\": \"Studio\"}", "{\"schema_version\": \"1\"}"] {
        let err = parse_session(raw).unwrap_err();
        assert!(matches!(err, SessionError::Corrupt(_)), "{raw:?}: {err:?}");
        assert!(err.may_overwrite());
    }
    for v in [0_u32, 99] {
        let err = parse_session(&format!(r#"{{"schema_version": {v}, "theme": "Studio"}}"#)).unwrap_err();
        assert_eq!(err, SessionError::Unsupported(v));
        assert!(!err.may_overwrite(), "foreign schema_version {v} must never be overwritten");
        assert!(err.to_string().contains(&v.to_string()));
    }
    // a number JSON cannot hold (beyond f64) makes the document unreadable: wholesale fallback
    assert!(matches!(parse_session(r#"{"schema_version": 1, "camera": {"yaw": 1e999}}"#), Err(SessionError::Corrupt(_))));
}

#[test]
fn wrong_typed_fields_fall_back_one_by_one_and_are_counted() {
    let raw = r#"{"schema_version": 1, "theme": "Studio", "png_scale": "big", "layout": 7,
                  "camera": {"yaw": "left", "pitch": 0.25},
                  "gallery": {"repeat": 2.5, "show_unit_cell": false, "params": {"b": 1.2, "gamma": "ninety"}}}"#;
    let (st, dropped) = parse_session(raw).unwrap();
    // png_scale, layout, camera.yaw, gallery.repeat and gallery.params.gamma are dropped: 5 fields
    assert_eq!(dropped, 5);
    assert_eq!(st.theme.as_deref(), Some("Studio"));
    assert_eq!(st.png_scale, None);
    assert_eq!(st.layout, LayoutSession::default());
    assert_eq!((st.camera.yaw, st.camera.pitch), (None, Some(0.25)));
    assert_eq!((st.gallery.repeat, st.gallery.show_unit_cell), (None, Some(false)));
    assert_eq!(st.gallery.params.get("b"), Some(&Some(1.2)));
    assert!(!st.gallery.params.contains_key("gamma"));
}

#[test]
fn nan_and_infinite_numbers_are_written_as_null_and_read_back_as_absent() {
    let mut st = sample_state();
    st.camera.yaw = Some(f32::NAN);
    st.camera.distance = Some(f32::INFINITY);
    st.defects.radius = Some(f64::NEG_INFINITY);
    st.gallery.params.insert(String::from("t1"), Some(f64::NAN));
    let raw = st.to_json_pretty().unwrap();
    assert!(raw.contains("\"yaw\": null") && raw.contains("\"distance\": null"), "{raw}");
    let (back, dropped) = parse_session(&raw).unwrap();
    assert_eq!(dropped, 0);
    assert_eq!((back.camera.yaw, back.camera.distance, back.defects.radius), (None, None, None));
    assert_eq!(back.gallery.params.get("t1"), Some(&None));
    assert_eq!(back.camera.pitch, Some(0.75), "finite neighbours survive");
    // a NaN inside the offset pair cannot be stored as a pair: the whole offset falls back
    st.camera.offset = Some([f32::NAN, 1.0]);
    let (back, dropped) = parse_session(&st.to_json_pretty().unwrap()).unwrap();
    assert_eq!((back.camera.offset, dropped), (None, 1));
}

#[test]
fn missing_file_loads_as_none_and_enum_names_parse_back() {
    let dir = TempDir::new();
    assert_eq!(load_session_file(&dir.0.join(SESSION_FILENAME)).unwrap(), None);
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Abc {
        A,
        Bc,
    }
    assert_eq!(name_of(Abc::Bc), "Bc");
    assert_eq!(parse_named(&[Abc::A, Abc::Bc], Some("Bc")), Some(Abc::Bc));
    assert_eq!(parse_named(&[Abc::A, Abc::Bc], Some("bc")), None);
    assert_eq!(parse_named(&[Abc::A, Abc::Bc], None), None);
}

#[test]
fn atomic_save_replaces_the_file_and_leaves_only_it_and_its_lock_sidecar() {
    let dir = TempDir::new();
    let path = dir.0.join(SESSION_FILENAME);
    std::fs::write(&path, b"old content").unwrap();
    let st = sample_state();
    save_session_file(&path, &st).unwrap();
    assert_eq!(load_session_file(&path).unwrap(), Some((st.clone(), 0)));
    assert_eq!(dir.names(), ["session.json", "session.json.lock"]);
    // a second save replaces the first
    let mut st2 = st.clone();
    st2.theme = Some("Paper".into());
    save_session_file(&path, &st2).unwrap();
    assert_eq!(load_session_file(&path).unwrap().unwrap().0.theme.as_deref(), Some("Paper"));
    assert_eq!(dir.names(), ["session.json", "session.json.lock"]);
}

#[test]
fn failed_saves_leave_the_old_file_intact_and_create_nothing() {
    let dir = TempDir::new();
    let path = dir.0.join(SESSION_FILENAME);
    save_session_file(&path, &sample_state()).unwrap();
    let before = std::fs::read(&path).unwrap();

    // the write itself fails half-way: old file unchanged, staging file removed
    let failed = save_session_json_with(&path, true, |f| {
        f.write_all(b"{\"schema_version\": 1, \"the")?;
        Err(io::Error::other("injected write failure"))
    });
    assert!(failed.is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(dir.names(), ["session.json", "session.json.lock"]);

    // the lock is held elsewhere: refused without blocking, old file unchanged
    let held = lock_sidecar(&path, "test").unwrap();
    let busy = save_session_file(&path, &SessionState { schema_version: 1, ..Default::default() });
    assert!(busy.unwrap_err().contains("busy"));
    drop(held);
    assert_eq!(std::fs::read(&path).unwrap(), before);

    // the target folder is missing: an error, and the folder is not created
    let missing = dir.0.join("gone").join(SESSION_FILENAME);
    assert!(save_session_file(&missing, &sample_state()).is_err());
    assert!(!dir.0.join("gone").exists());
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn invalid_utf8_file_is_corrupt_and_unlocked_save_leaves_no_lock_sidecar() {
    let dir = TempDir::new();
    let bad = dir.0.join("bad.json");
    std::fs::write(&bad, b"\xff\xfe{\"schema_version\": 1}").unwrap();
    let err = load_session_file(&bad).unwrap_err();
    assert!(matches!(err, SessionError::Corrupt(_)), "{err:?}");
    assert!(err.may_overwrite());

    let exported = dir.0.join("exported.json");
    let st = sample_state();
    save_session_file_unlocked(&exported, &st).unwrap();
    assert_eq!(load_session_file(&exported).unwrap(), Some((st, 0)));
    assert_eq!(dir.names(), ["bad.json", "exported.json"]);
}
