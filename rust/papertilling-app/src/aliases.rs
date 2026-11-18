//! Persistent user-defined custom names (aliases) for lattices, stored in the user's OS config directory
//! outside the git working tree. Degrades gracefully to in-memory storage in `--smoke-test`, tests, or
//! read-only environments. Saves merge one change under a non-blocking sidecar lock, then atomically
//! replace the JSON with a flushed sibling file. Invalid/unknown files are never overwritten.
//! Names are UTF-8; PNG export still uses the app's limited bitmap font (SVG/JSON preserve Unicode).

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

const ALIAS_FILENAME: &str = "lattice_names.json";
const ALIAS_VERSION: u32 = 1;
type AliasMap = BTreeMap<LatticeAliasKey, String>;

/// Stable lookup key `(group, rule, dsymbol_key_or_index)` identifying any catalogue or named lattice.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LatticeAliasKey {
    pub group: String,
    pub rule: String,
    pub dsymbol_key_or_index: String,
}

impl LatticeAliasKey {
    pub fn new(group: impl Into<String>, rule: impl Into<String>, index: usize, dsymbol_key: &str) -> Self {
        let dsymbol_key_or_index = if !dsymbol_key.is_empty() && dsymbol_key != "-" {
            format!("#{index}:{dsymbol_key}")
        } else {
            format!("#{index}")
        };
        Self {
            group: group.into(),
            rule: rule.into(),
            dsymbol_key_or_index,
        }
    }

    fn dsymbol_suffix(&self) -> Option<&str> {
        let (_, dsym) = self.dsymbol_key_or_index.split_once(':')?;
        (!dsym.is_empty() && dsym != "-").then_some(dsym)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct AliasFileJson {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    entries: Vec<AliasEntryJson>,
}

fn default_version() -> u32 {
    ALIAS_VERSION
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AliasEntryJson {
    group: String,
    rule: String,
    dsymbol_key_or_index: String,
    custom_name: String,
}

#[derive(Clone, Debug, Default)]
pub struct AliasStore {
    path: Option<PathBuf>,
    map: AliasMap,
}

impl AliasStore {
    /// Ephemeral store with no disk backing (used in `--smoke-test` and hermetic unit tests).
    pub fn in_memory() -> Self {
        Self { path: None, map: BTreeMap::new() }
    }

    /// Load aliases from the user's OS configuration folder (`lattice_names.json`), falling back cleanly
    /// if the file does not exist yet, is unreadable, or contains invalid JSON.
    pub fn load_default() -> Self {
        match default_config_file_path() {
            Some(path) => Self::load_from_path(path),
            None => Self::in_memory(),
        }
    }

    /// Load aliases from a specific JSON path (creates/updates that path on subsequent `set`/`reset`).
    pub fn load_from_path(path: PathBuf) -> Self {
        let map = read_aliases(&path).unwrap_or_default();
        Self { path: Some(path), map }
    }

    pub fn get(&self, key: &LatticeAliasKey) -> Option<&str> {
        if let Some(exact) = self.map.get(key) {
            return Some(exact.as_str());
        }
        let dsym = key.dsymbol_suffix()?;
        self.map
            .iter()
            .find(|(k, _)| k.group == key.group && k.dsymbol_suffix() == Some(dsym))
            .map(|(_, v)| v.as_str())
    }

    /// Set or clear a custom alias (`name` trimmed; empty or equal to `default_label` resets to default).
    /// Even if writing to disk fails (read-only environment), the in-memory update stays active.
    pub fn set(&mut self, key: LatticeAliasKey, name: &str, default_label: &str) -> Result<(), String> {
        let trimmed = name.trim();
        if trimmed.is_empty() || trimmed == default_label {
            return self.reset(&key);
        }
        apply_change(&mut self.map, &key, Some(trimmed));
        self.save_change(&key, Some(trimmed))
    }

    pub fn reset(&mut self, key: &LatticeAliasKey) -> Result<(), String> {
        apply_change(&mut self.map, key, None);
        self.save_change(key, None)
    }

    // The local snapshot (including failed, session-only edits) is not a disk write set. Never save
    // it wholesale: another independent store may have changed unrelated names since it was loaded.
    fn save_change(&self, key: &LatticeAliasKey, name: Option<&str>) -> Result<(), String> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        std::fs::create_dir_all(parent_dir(path)).map_err(|e| format!("config dir: {e}"))?;
        // Lock a stable sidecar, not the JSON inode that rename replaces. Keep the sidecar: unlinking
        // it on unlock could let a third process acquire a different lock during an existing save.
        let _lock = lock_sidecar(path, "alias")?;
        let mut latest = read_aliases(path)?;
        if std::fs::metadata(path).is_ok_and(|m| m.permissions().readonly()) {
            return Err(String::from("alias file is read-only"));
        }
        apply_change(&mut latest, key, name);
        let payload = AliasFileJson {
            version: ALIAS_VERSION,
            entries: latest
                .iter()
                .map(|(k, v)| AliasEntryJson {
                    group: k.group.clone(),
                    rule: k.rule.clone(),
                    dsymbol_key_or_index: k.dsymbol_key_or_index.clone(),
                    custom_name: v.clone(),
                })
                .collect(),
        };
        let json = serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())?;
        replace_atomically(path, |file| file.write_all(json.as_bytes()))
            .map_err(|e| format!("{}: {e}", path.display()))
        // Dropping `_lock` releases the OS lock, including on every error path above.
    }
}

fn apply_change(map: &mut AliasMap, key: &LatticeAliasKey, name: Option<&str>) {
    map.remove(key);
    if let Some(dsym) = key.dsymbol_suffix() {
        map.retain(|k, _| !(k.group == key.group && k.dsymbol_suffix() == Some(dsym)));
    }
    if let Some(name) = name {
        map.insert(key.clone(), name.to_owned());
    }
}

fn read_aliases(path: &Path) -> Result<AliasMap, String> {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(AliasMap::new()),
        Err(e) => return Err(format!("cannot read aliases: {e}")),
    };
    let parsed: AliasFileJson = serde_json::from_str(&raw).map_err(|e| format!("invalid alias file: {e}"))?;
    if parsed.version != ALIAS_VERSION {
        return Err(format!("unsupported alias file version {}", parsed.version));
    }
    let mut map = AliasMap::new();
    for e in parsed.entries {
        let trimmed = e.custom_name.trim();
        if !trimmed.is_empty() {
            map.insert(LatticeAliasKey {
                group: e.group, rule: e.rule, dsymbol_key_or_index: e.dsymbol_key_or_index,
            }, trimmed.to_owned());
        }
    }
    Ok(map)
}

pub(crate) fn parent_dir(path: &Path) -> &Path {
    path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."))
}

pub(crate) fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Take the non-blocking OS lock on `path`'s stable `.lock` sidecar (shared by the alias and session files).
/// Dropping the returned handle releases the lock; the sidecar itself is kept (see `save_change`).
pub(crate) fn lock_sidecar(path: &Path, what: &str) -> Result<File, String> {
    let lock = OpenOptions::new().read(true).write(true).create(true).truncate(false)
        .open(with_suffix(path, ".lock")).map_err(|e| format!("{what} lock: {e}"))?;
    lock.try_lock().map_err(|e| format!("{what} file busy or lock unavailable; retry saving: {e}"))?;
    Ok(lock)
}

// Same-directory rename replaces an existing file on Windows as well as Unix; never delete the
// destination first. `create_new` avoids following a stale/hostile staging-file symlink.
// The writer callback also gives tests a deterministic partial-write failure without filling a disk.
pub(crate) fn replace_atomically(path: &Path, write: impl FnOnce(&mut File) -> io::Result<()>) -> io::Result<()> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    #[cfg(unix)]
    let directory = File::open(parent_dir(path))?;
    let mut staged = None;
    for _ in 0..16 {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let tmp = with_suffix(path, &format!(".{}.{id}.tmp", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(file) => { staged = Some((tmp, file)); break; }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    let (tmp, mut file) = staged.ok_or_else(|| io::Error::new(io::ErrorKind::AlreadyExists, "alias staging files busy"))?;
    let written = write(&mut file).and_then(|()| file.sync_all());
    drop(file); // Close before rename/removal, particularly on Windows.
    let replaced = written.and_then(|()| std::fs::rename(&tmp, path));
    if replaced.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    replaced?;
    // Flush directory metadata where the standard library supports opening directories.
    #[cfg(unix)]
    directory.sync_all()?;
    Ok(())
}

/// Resolve the per-user OS configuration file path outside the repository working tree:
/// - Windows: `%APPDATA%/papertilling/lattice_names.json`
/// - macOS:   `$XDG_CONFIG_HOME/papertilling/...` or `~/Library/Application Support/papertilling/...`
/// - Linux:   `$XDG_CONFIG_HOME/papertilling/...` or `~/.config/papertilling/...`
pub fn default_config_file_path() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join(ALIAS_FILENAME))
}

/// The per-user `papertilling` configuration folder (see [`default_config_file_path`]); also holds `session.json`.
pub(crate) fn config_dir() -> Option<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        std::env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .filter(|v| !v.is_empty())
                    .map(|h| PathBuf::from(h).join("Library").join("Application Support"))
            })
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .filter(|v| !v.is_empty())
                    .map(|h| PathBuf::from(h).join(".config"))
            })
    };
    base.map(|dir| dir.join("papertilling"))
}

#[cfg(test)]
mod tests;
