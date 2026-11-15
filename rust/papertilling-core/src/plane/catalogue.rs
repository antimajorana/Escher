//! Catalogue loaders (`points.json`, `sheet.json`, `minimal_b.json`, `named.json`),
//! matching `papertilling.lattice.{MinimalResult, load_catalogue}` and `papertilling.sheet.named`.

use std::collections::BTreeMap;
use std::fs;
use std::ops::Deref;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::dsymbol::{self, DSymbol, DSymbolKey, KeyRow};
use crate::plane::family::LatticeFamily;
use crate::types::{Cell2D, Frame2D, Recipe};

/// Enumeration rule selecting one of the three stored plane-group catalogues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CatalogueRule {
    Points,
    Sheet,
    Bonds,
}

impl CatalogueRule {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Points => "points",
            Self::Sheet => "sheet",
            Self::Bonds => "bonds",
        }
    }

    pub const fn filename(self) -> &'static str {
        match self {
            Self::Points => "points.json",
            Self::Sheet => "sheet.json",
            Self::Bonds => "minimal_b.json",
        }
    }
}

/// Tier secondary index: integer bond-orbit count (`sheet`, `bonds`) or site count string (`points`, e.g. `"1 sites"`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TierN {
    Int(usize),
    Text(String),
}

/// Search status of one `(k, n)` enumeration tier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TierInfo {
    pub k: usize,
    pub n: TierN,
    pub status: String,
    pub entries: usize,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

fn default_bonds_rule() -> String {
    "bonds".to_string()
}

fn default_network_same() -> String {
    "network".to_string()
}

/// Search bounds metadata stored with a `GroupCatalogue`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogueBounds {
    #[serde(default = "default_bonds_rule")]
    pub rule: String,
    #[serde(default = "default_network_same")]
    pub same: String,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Default for CatalogueBounds {
    fn default() -> Self {
        Self {
            rule: default_bonds_rule(),
            same: default_network_same(),
            extra: BTreeMap::new(),
        }
    }
}

fn rows_to_key(rows: &[[u32; 5]]) -> DSymbolKey {
    rows.iter().map(|&[m01, m12, s0, s1, s2]| (m01, m12, s0, s1, s2)).collect()
}

#[derive(Deserialize)]
struct RawLatticeEntry {
    recipe: Recipe,
    #[serde(default)]
    key: Option<Vec<[u32; 5]>>,
    rank: [usize; 4],
    #[serde(default)]
    planar_values: Vec<BTreeMap<String, f64>>,
    #[serde(default)]
    meta: serde_json::Value,
}

/// One reconstructed lattice entry in a `GroupCatalogue`, with Rust-recomputed D-symbol keys.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatticeEntry {
    pub recipe: Recipe,
    pub family: LatticeFamily,
    pub dsymbol: DSymbol,
    pub canonical_key: DSymbolKey,
    pub stored_key: Option<DSymbolKey>,
    pub network_key: DSymbolKey,
    pub symmetry_index: usize,
    pub rank: [usize; 4],
    pub planar_values: Vec<BTreeMap<String, f64>>,
    pub meta: serde_json::Value,
}

impl LatticeEntry {
    fn from_raw(raw: RawLatticeEntry) -> Result<Self, String> {
        let family = LatticeFamily::from_recipe(&raw.recipe, None)?;
        let dsymbol = family.dsymbol()?;
        let canonical_key = dsymbol.canonical_key().map_err(|e| e.to_string())?;
        let network_key = dsymbol::network_key(&canonical_key).map_err(|e| e.to_string())?;
        let symmetry_index = dsymbol::symmetry_index(&canonical_key).map_err(|e| e.to_string())?;
        let stored_key = raw.key.as_deref().map(rows_to_key);
        Ok(Self {
            recipe: raw.recipe,
            family,
            dsymbol,
            canonical_key,
            stored_key,
            network_key,
            symmetry_index,
            rank: raw.rank,
            planar_values: raw.planar_values,
            meta: raw.meta,
        })
    }

    pub fn key(&self) -> &[KeyRow] {
        &self.canonical_key
    }
}

fn default_n() -> usize {
    7
}

#[derive(Deserialize)]
struct RawGroupCatalogue {
    group: String,
    #[serde(default = "default_n")]
    n: usize,
    coverage: String,
    #[serde(default)]
    tiers: Vec<TierInfo>,
    #[serde(default)]
    notes: Vec<String>,
    #[serde(default)]
    seconds: Option<f64>,
    #[serde(default)]
    bounds: CatalogueBounds,
    #[serde(default)]
    counts: serde_json::Value,
    #[serde(default)]
    gaps: Vec<serde_json::Value>,
    entries: Vec<RawLatticeEntry>,
    #[serde(default)]
    other: Option<Vec<Vec<RawLatticeEntry>>>,
}

#[derive(Deserialize)]
struct RawCatalogueFile {
    groups: BTreeMap<String, RawGroupCatalogue>,
}

/// Reconstructed per-group catalogue matching `papertilling.lattice.MinimalResult`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupCatalogue {
    pub group: String,
    pub rule: String,
    pub same: String,
    pub n: usize,
    pub coverage: String,
    pub tiers: Vec<TierInfo>,
    pub notes: Vec<String>,
    pub seconds: Option<f64>,
    pub bounds: CatalogueBounds,
    pub counts: serde_json::Value,
    pub gaps: Vec<serde_json::Value>,
    /// One entry per distinct network in entry order (`same == "network"`) or pre-grouped `points` entries.
    pub all: Vec<LatticeEntry>,
    /// First `n` entries of `all`.
    pub lattices: Vec<LatticeEntry>,
    /// Every drawing found (`all + other.flatten()` for `points`, or `entries` for `sheet`/`bonds`).
    pub drawings: Vec<LatticeEntry>,
    /// Secondary drawings grouped parallel to `all[k]`.
    pub other: Vec<Vec<LatticeEntry>>,
}

impl GroupCatalogue {
    fn from_raw(raw: RawGroupCatalogue) -> Result<Self, String> {
        let entries: Vec<LatticeEntry> = raw
            .entries
            .into_iter()
            .map(LatticeEntry::from_raw)
            .collect::<Result<_, _>>()?;
        let same = raw.bounds.same.clone();
        let rule = raw.bounds.rule.clone();

        let (all, other, drawings) = if let Some(raw_other) = raw.other {
            let mut other_built = Vec::with_capacity(raw_other.len());
            for bucket in raw_other {
                let b: Vec<LatticeEntry> = bucket
                    .into_iter()
                    .map(LatticeEntry::from_raw)
                    .collect::<Result<_, _>>()?;
                other_built.push(b);
            }
            let mut all_drawings = entries.clone();
            for bucket in &other_built {
                all_drawings.extend(bucket.iter().cloned());
            }
            (entries, other_built, all_drawings)
        } else {
            let mut all_vec: Vec<LatticeEntry> = Vec::new();
            let mut other_vec: Vec<Vec<LatticeEntry>> = Vec::new();
            let mut first: BTreeMap<DSymbolKey, usize> = BTreeMap::new();
            for entry in &entries {
                let dedup_key = if same == "network" {
                    &entry.network_key
                } else {
                    &entry.canonical_key
                };
                if let Some(&idx) = first.get(dedup_key) {
                    other_vec[idx].push(entry.clone());
                } else {
                    first.insert(dedup_key.clone(), all_vec.len());
                    all_vec.push(entry.clone());
                    other_vec.push(Vec::new());
                }
            }
            (all_vec, other_vec, entries)
        };

        let take = raw.n.min(all.len());
        let lattices = all[..take].to_vec();
        Ok(Self {
            group: raw.group,
            rule,
            same,
            n: raw.n,
            coverage: raw.coverage,
            tiers: raw.tiers,
            notes: raw.notes,
            seconds: raw.seconds,
            bounds: raw.bounds,
            counts: raw.counts,
            gaps: raw.gaps,
            all,
            lattices,
            drawings,
            other,
        })
    }
}

/// Parse one of the stored catalogues (`points.json`, `sheet.json`, `minimal_b.json`) from JSON text.
pub fn parse_catalogue_str(text: &str) -> Result<BTreeMap<String, GroupCatalogue>, String> {
    let parsed: RawCatalogueFile =
        serde_json::from_str(text).map_err(|e| format!("failed to parse catalogue JSON: {e}"))?;
    let mut out = BTreeMap::new();
    for (g, raw) in parsed.groups {
        out.insert(g, GroupCatalogue::from_raw(raw)?);
    }
    Ok(out)
}

/// Load one of the three stored catalogues (`points.json`, `sheet.json`, `minimal_b.json`) from `data_dir`.
pub fn load_catalogue(
    data_dir: &Path,
    rule: CatalogueRule,
) -> Result<BTreeMap<String, GroupCatalogue>, String> {
    let path = data_dir.join(rule.filename());
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    parse_catalogue_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

#[derive(Deserialize)]
struct RawNamedLattice {
    name: String,
    key: String,
    #[serde(default)]
    sheet_group: Option<String>,
    #[serde(default)]
    variant_of: Option<String>,
    group: String,
    recipe: Recipe,
    frame: Frame2D,
    rank: [usize; 4],
    dsymbol_key: Vec<[u32; 5]>,
    network_key: Vec<[u32; 5]>,
    symmetry_index: usize,
    cell: Cell2D,
    raw_cell: Cell2D,
}

#[derive(Deserialize)]
struct RawNamedFile {
    named: Vec<RawNamedLattice>,
    #[serde(default)]
    aliases: BTreeMap<String, String>,
}

/// One of the 21 named lattices (18 sheet drawings + 3 group-faithful variants) from `named.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedLattice {
    pub name: String,
    pub key: String,
    pub sheet_group: Option<String>,
    pub variant_of: Option<String>,
    pub group: String,
    pub recipe: Recipe,
    pub frame: Frame2D,
    pub family: LatticeFamily,
    pub dsymbol: DSymbol,
    pub rank: [usize; 4],
    pub dsymbol_key: DSymbolKey,
    pub stored_dsymbol_key: DSymbolKey,
    pub network_key: DSymbolKey,
    pub stored_network_key: DSymbolKey,
    pub symmetry_index: usize,
    pub stored_symmetry_index: usize,
    pub cell: Cell2D,
    pub stored_cell: Cell2D,
    pub raw_cell: Cell2D,
}

impl NamedLattice {
    fn from_raw(raw: RawNamedLattice) -> Result<Self, String> {
        let family = LatticeFamily::from_recipe(&raw.recipe, Some(raw.frame))?;
        let cell = family.cell();
        let dsymbol = family.dsymbol()?;
        let dsymbol_key = dsymbol.canonical_key().map_err(|e| e.to_string())?;
        let network_key = dsymbol::network_key(&dsymbol_key).map_err(|e| e.to_string())?;
        let symmetry_index = dsymbol::symmetry_index(&dsymbol_key).map_err(|e| e.to_string())?;
        Ok(Self {
            name: raw.name,
            key: raw.key,
            sheet_group: raw.sheet_group,
            variant_of: raw.variant_of,
            group: raw.group,
            recipe: raw.recipe,
            frame: raw.frame,
            family,
            dsymbol,
            rank: raw.rank,
            dsymbol_key,
            stored_dsymbol_key: rows_to_key(&raw.dsymbol_key),
            network_key,
            stored_network_key: rows_to_key(&raw.network_key),
            symmetry_index,
            stored_symmetry_index: raw.symmetry_index,
            cell,
            stored_cell: raw.cell,
            raw_cell: raw.raw_cell,
        })
    }
}

/// Normalise a loose human-readable lattice name into a lookup key (`papertilling.sheet._key`).
pub fn normalize_named_key(name: &str) -> String {
    let mut k = name.trim().to_lowercase();
    for (from, to) in [
        (" ", "_"),
        ("'", ""),
        ("\u{2019}", ""),
        ("-", "_"),
        ("\u{00e9}", "e"),
        ("(", ""),
        (")", ""),
    ] {
        k = k.replace(from, to);
    }
    k
}

/// Collection of the 21 named lattices and loose-name aliases loaded from `named.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedCatalogue {
    pub named: Vec<NamedLattice>,
    pub aliases: BTreeMap<String, String>,
}

impl NamedCatalogue {
    /// Look up a named lattice by loose name, canonical key, or alias (`"Kagomé"`, `"Snub Square"`, etc.).
    pub fn get(&self, name: &str) -> Option<&NamedLattice> {
        let norm = normalize_named_key(name);
        let target = self.aliases.get(&norm).map_or(norm.as_str(), String::as_str);
        self.named.iter().find(|item| item.key == target)
    }
}

impl Deref for NamedCatalogue {
    type Target = [NamedLattice];

    fn deref(&self) -> &Self::Target {
        &self.named
    }
}

impl<'a> IntoIterator for &'a NamedCatalogue {
    type Item = &'a NamedLattice;
    type IntoIter = std::slice::Iter<'a, NamedLattice>;

    fn into_iter(self) -> Self::IntoIter {
        self.named.iter()
    }
}

impl IntoIterator for NamedCatalogue {
    type Item = NamedLattice;
    type IntoIter = std::vec::IntoIter<NamedLattice>;

    fn into_iter(self) -> Self::IntoIter {
        self.named.into_iter()
    }
}

/// Parse `named.json` (18 sheet lattices + 3 variants + aliases) from JSON text.
pub fn parse_named_str(text: &str) -> Result<NamedCatalogue, String> {
    let parsed: RawNamedFile =
        serde_json::from_str(text).map_err(|e| format!("failed to parse named JSON: {e}"))?;
    let named = parsed
        .named
        .into_iter()
        .map(NamedLattice::from_raw)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NamedCatalogue { named, aliases: parsed.aliases })
}

/// Load `named.json` (18 sheet lattices + 3 variants + aliases) from `data_dir`.
pub fn load_named(data_dir: &Path) -> Result<NamedCatalogue, String> {
    let path = data_dir.join("named.json");
    let text = fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    parse_named_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}
