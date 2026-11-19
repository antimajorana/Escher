//! Versioned session file (`session.json`, schema 1): the user-visible state needed to resume where the user left
//! off. Stored next to the alias file in the OS config folder and written with the same sidecar lock and atomic
//! temp-file replace (`crate::aliases`). Never read or written in `--smoke-test` mode or in tests (the app then
//! has no session path), mirroring the in-memory `AliasStore`.
//!
//! Loading is lenient: every field is read on its own (`lenient`), so a field of the wrong type falls back to its
//! default without losing the others; numbers are `Option`s because serde_json writes NaN / inf as `null`. Values
//! are only *stored* here; range checks (clamping, snapping, re-resolving the lattice from the catalogues) happen
//! when the app applies them (`app::session_hooks`).
//!
//! Deliberately not persisted: build results, meshes, caches and thumbnails, worker state, spectra and elastic
//! relaxations, comparison reports (the Combine `summary`), undo history, the Custom Builder's in-progress lattice,
//! inspector expansions, the 2D canvas pan / zoom and any rename in progress.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use papertilling_core::defect::combine_types::NotebookConfigJson;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};

use crate::aliases::{LatticeAliasKey, config_dir, lock_sidecar, replace_atomically};

pub const SESSION_FILENAME: &str = "session.json";
pub const SESSION_SCHEMA_VERSION: u32 = 2;

thread_local! {
    /// Fields dropped by [`lenient`] during the current [`parse_session`] call.
    static DROPPED_FIELDS: Cell<u32> = const { Cell::new(0) };
}

/// Deserialize a field, falling back to its default (and counting the drop) when its JSON has the wrong shape.
fn lenient<'de, D: Deserializer<'de>, T: DeserializeOwned + Default>(d: D) -> Result<T, D::Error> {
    let raw = serde_json::Value::deserialize(d)?;
    Ok(T::deserialize(raw).unwrap_or_else(|_| {
        DROPPED_FIELDS.with(|c| c.set(c.get() + 1));
        T::default()
    }))
}

/// Deserialize `gallery.params` entry by entry so one bad entry does not drop valid sibling parameters.
fn lenient_params<'de, D: Deserializer<'de>>(d: D) -> Result<BTreeMap<String, Option<f64>>, D::Error> {
    let raw = serde_json::Value::deserialize(d)?;
    let serde_json::Value::Object(map) = raw else {
        DROPPED_FIELDS.with(|c| c.set(c.get() + 1));
        return Ok(BTreeMap::new());
    };
    let mut out = BTreeMap::new();
    for (k, v) in map {
        match Option::<f64>::deserialize(v) {
            Ok(opt) => {
                out.insert(k, opt);
            }
            Err(_) => DROPPED_FIELDS.with(|c| c.set(c.get() + 1)),
        }
    }
    Ok(out)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionState {
    pub schema_version: u32,
    /// `NavTab` variant name (`"Gallery"`, `"Surfaces"`, ...).
    #[serde(default, deserialize_with = "lenient")]
    pub active_tab: Option<String>,
    /// `StudioTheme` variant name (`"Paper"`, `"Studio"`).
    #[serde(default, deserialize_with = "lenient")]
    pub theme: Option<String>,
    #[serde(default, deserialize_with = "lenient")]
    pub layout: LayoutSession,
    #[serde(default, deserialize_with = "lenient")]
    pub png_scale: Option<u8>,
    #[serde(default, deserialize_with = "lenient")]
    pub lattice: Option<LatticeSession>,
    #[serde(default, deserialize_with = "lenient")]
    pub gallery: GallerySession,
    /// Quick Lattice Switcher Recent list, most recent first.
    #[serde(default, deserialize_with = "lenient")]
    pub recent: Vec<LatticeAliasKey>,
    #[serde(default, deserialize_with = "lenient")]
    pub surfaces: SurfacesSession,
    #[serde(default, deserialize_with = "lenient")]
    pub defects: DefectsSession,
    /// Combine pins and options in the Notebook 3 format (`summary` dropped: it is a transient result).
    #[serde(default, deserialize_with = "lenient")]
    pub combine: Option<NotebookConfigJson>,
    #[serde(default, deserialize_with = "lenient")]
    pub camera: CameraSession,
    /// `AtlasSubTab` variant name.
    #[serde(default, deserialize_with = "lenient")]
    pub atlas_sub_view: Option<String>,
    /// `AppMode` variant name (`"Lattices"`, `"Defects"`, `"Analysis"`). Absent in schema v1.
    #[serde(default, deserialize_with = "lenient")]
    pub active_mode: Option<String>,
    /// `LatticesView` variant name.
    #[serde(default, deserialize_with = "lenient")]
    pub lattices_view: Option<String>,
    /// `DefectsView` variant name.
    #[serde(default, deserialize_with = "lenient")]
    pub defects_view: Option<String>,
    /// `AnalysisTarget` variant name.
    #[serde(default, deserialize_with = "lenient")]
    pub analysis_target: Option<String>,
    /// `DefectViewMode` variant name in Analysis mode.
    #[serde(default, deserialize_with = "lenient")]
    pub analysis_view: Option<String>,
    /// Workbench (stage S2..S6 panels): `{"view": "<WorkbenchView>", "s2": .., ...}`, each panel's own JSON.
    /// Optional and additive, so schema 2 is kept: older files load without it and older apps ignore it.
    #[serde(default, deserialize_with = "lenient")]
    pub workbench: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LayoutSession {
    #[serde(default, deserialize_with = "lenient")]
    pub left_rail_width: Option<f32>,
    #[serde(default, deserialize_with = "lenient")]
    pub right_inspector_open: Option<bool>,
    #[serde(default, deserialize_with = "lenient")]
    pub right_inspector_width: Option<f32>,
    #[serde(default, deserialize_with = "lenient")]
    pub split_ratio: Option<f32>,
}

/// The selected stored lattice: re-resolved from the catalogues on restore (its recipe is never stored).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LatticeSession {
    pub alias_key: LatticeAliasKey,
    #[serde(default)]
    pub group: String,
    #[serde(default, deserialize_with = "lenient")]
    pub named_key: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GallerySession {
    /// Free-parameter values by slider name (applied through `SliderState`, so clamped and snapped).
    #[serde(default, deserialize_with = "lenient_params")]
    pub params: BTreeMap<String, Option<f64>>,
    #[serde(default, deserialize_with = "lenient")]
    pub repeat: Option<i32>,
    #[serde(default, deserialize_with = "lenient")]
    pub show_unit_cell: Option<bool>,
    #[serde(default, deserialize_with = "lenient")]
    pub overlay_symmetry: Option<bool>,
    #[serde(default, deserialize_with = "lenient")]
    pub overlay_voronoi: Option<bool>,
    /// `BrowserView` variant name (`"Cards"`, `"Table"`).
    #[serde(default, deserialize_with = "lenient")]
    pub browser_view: Option<String>,
    #[serde(default, deserialize_with = "lenient")]
    pub filter_query: Option<String>,
    /// `SystemFacet` variant name; absent = all systems.
    #[serde(default, deserialize_with = "lenient")]
    pub filter_system: Option<String>,
    /// `ChiralFacet` variant name.
    #[serde(default, deserialize_with = "lenient")]
    pub filter_chirality: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SurfacesSession {
    /// `SurfaceTargetUi` variant name.
    #[serde(default, deserialize_with = "lenient")]
    pub target: Option<String>,
    #[serde(default, deserialize_with = "lenient")]
    pub h: Option<i32>,
    #[serde(default, deserialize_with = "lenient")]
    pub k: Option<i32>,
    #[serde(default, deserialize_with = "lenient")]
    pub genus_g: Option<u32>,
    /// `CentreModeUi` variant name.
    #[serde(default, deserialize_with = "lenient")]
    pub centre_mode: Option<String>,
    #[serde(default, deserialize_with = "lenient")]
    pub auto_size: Option<bool>,
    #[serde(default, deserialize_with = "lenient")]
    pub show_cover: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DefectsSession {
    /// `DefectSubTab` variant name: the selection kind.
    #[serde(default, deserialize_with = "lenient")]
    pub kind: Option<String>,
    #[serde(default, deserialize_with = "lenient")]
    pub dislocation_index: Option<usize>,
    #[serde(default, deserialize_with = "lenient")]
    pub disclination_index: Option<usize>,
    #[serde(default, deserialize_with = "lenient")]
    pub wall_index: Option<usize>,
    #[serde(default, deserialize_with = "lenient")]
    pub wall_half: Option<bool>,
    #[serde(default, deserialize_with = "lenient")]
    pub radius: Option<f64>,
    /// Informational (`draw_mode != Raw`); used only when `draw_mode` is absent.
    #[serde(default, deserialize_with = "lenient")]
    pub relaxed: Option<bool>,
    /// `Defect2dMode` variant name.
    #[serde(default, deserialize_with = "lenient")]
    pub draw_mode: Option<String>,
    /// `DefectViewMode` variant name.
    #[serde(default, deserialize_with = "lenient")]
    pub view_mode: Option<String>,
    /// `Sheet3dColorMode` variant name.
    #[serde(default, deserialize_with = "lenient")]
    pub color_mode: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CameraSession {
    #[serde(default, deserialize_with = "lenient")]
    pub yaw: Option<f32>,
    #[serde(default, deserialize_with = "lenient")]
    pub pitch: Option<f32>,
    #[serde(default, deserialize_with = "lenient")]
    pub distance: Option<f32>,
    #[serde(default, deserialize_with = "lenient")]
    pub offset: Option<[f32; 2]>,
    /// Turntable about world z (Front / Side presets) instead of world y.
    #[serde(default, deserialize_with = "lenient")]
    pub z_up: Option<bool>,
}

/// Why a session file was not used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionError {
    /// Not JSON or not a JSON object (empty file included): replaced by the next save.
    Corrupt(String),
    /// Another `schema_version` (e.g. a newer app): kept, never overwritten by the exit save.
    Unsupported(u32),
    /// Unreadable (permissions, ...): kept.
    Io(String),
}

impl SessionError {
    /// Whether the exit save may replace the file that failed to load.
    pub fn may_overwrite(&self) -> bool {
        matches!(self, Self::Corrupt(_))
    }
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corrupt(e) => write!(f, "invalid session file ({e})"),
            Self::Unsupported(v) => write!(f, "unsupported session schema_version {v} (expected {SESSION_SCHEMA_VERSION})"),
            Self::Io(e) => write!(f, "cannot read session file ({e})"),
        }
    }
}

/// Variant name of an enum value as stored in the session (its `Debug` name).
pub fn name_of<T: std::fmt::Debug>(v: T) -> String {
    format!("{v:?}")
}

/// The variant of `all` called `name` (`None` if absent or unknown).
pub fn parse_named<T: Copy + std::fmt::Debug>(all: &[T], name: Option<&str>) -> Option<T> {
    let name = name?;
    all.iter().copied().find(|v| format!("{v:?}") == name)
}

/// `session.json` in the per-user configuration folder (next to `lattice_names.json`).
pub fn default_session_path() -> Option<PathBuf> {
    config_dir().map(|d| d.join(SESSION_FILENAME))
}

impl SessionState {
    pub fn to_json_pretty(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }
}

/// Parse a session document: the state plus the number of fields that fell back to defaults (wrong JSON type).
pub fn parse_session(raw: &str) -> Result<(SessionState, u32), SessionError> {
    let value: serde_json::Value = serde_json::from_str(raw).map_err(|e| SessionError::Corrupt(e.to_string()))?;
    let Some(obj) = value.as_object() else {
        return Err(SessionError::Corrupt(String::from("not a JSON object")));
    };
    let version = obj.get("schema_version").and_then(serde_json::Value::as_u64);
    match version {
        Some(1 | 2) => {}
        Some(v) => return Err(SessionError::Unsupported(u32::try_from(v).unwrap_or(u32::MAX))),
        None => return Err(SessionError::Corrupt(String::from("missing schema_version"))),
    }
    DROPPED_FIELDS.with(|c| c.set(0));
    let state = SessionState::deserialize(value).map_err(|e| SessionError::Corrupt(e.to_string()));
    let dropped = DROPPED_FIELDS.with(|c| c.replace(0));
    Ok((state?, dropped))
}

/// Read `path`: `Ok(None)` when there is no file yet. Invalid UTF-8 (`InvalidData`) is treated as `Corrupt`
/// so a garbled session file falls back and may be replaced on exit rather than blocking exit saves forever.
pub fn load_session_file(path: &Path) -> Result<Option<(SessionState, u32)>, SessionError> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) if e.kind() == io::ErrorKind::InvalidData => return Err(SessionError::Corrupt(e.to_string())),
        Err(e) => return Err(SessionError::Io(e.to_string())),
    };
    parse_session(&raw).map(Some)
}

/// Write `state` to `path` under the `.lock` sidecar with an atomic same-directory replace. The parent folder must
/// exist (a missing folder is an error and nothing is created); an existing file is replaced only when the new
/// content is fully written and flushed.
pub fn save_session_file(path: &Path, state: &SessionState) -> Result<(), String> {
    save_session_with_lock(path, state, true)
}

/// Write `state` to `path` atomically without leaving a `.lock` sidecar (for "Save Session As..." exports).
pub fn save_session_file_unlocked(path: &Path, state: &SessionState) -> Result<(), String> {
    save_session_with_lock(path, state, false)
}

fn save_session_with_lock(path: &Path, state: &SessionState, use_lock: bool) -> Result<(), String> {
    let json = state.to_json_pretty()?;
    save_session_json_with(path, use_lock, |file| file.write_all(json.as_bytes()))
}

/// [`save_session_file`] with the writer supplied by the caller (tests inject a failing write).
pub(crate) fn save_session_json_with(
    path: &Path,
    use_lock: bool,
    write: impl FnOnce(&mut std::fs::File) -> io::Result<()>,
) -> Result<(), String> {
    let parent = crate::aliases::parent_dir(path);
    if !parent.is_dir() {
        return Err(format!("{}: folder does not exist", parent.display()));
    }
    let _lock = use_lock.then(|| lock_sidecar(path, "session")).transpose()?;
    replace_atomically(path, write).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
#[path = "session_tests.rs"]
pub(crate) mod tests;
