//! Cached single-defect comparison (`config.equivalent_single`) for live use.
//!
//! The Volterra + completion core gallery of a dislocation depends only on the lattice, its Burgers
//! vector `b`, the core radius, the scan grid and whether the cards are drawn: never on where the
//! defects of the configuration sit. Python caches it per `(lattice, round(b, 6), radius, grid, ...)`
//! (`config._DISLOC_CACHE`, `cores._GALLERY_CACHE`); `SingleDefectCache` is the same cache for Rust
//! callers that compare many configurations (the desktop app while defects are placed or dragged).
//! The single disclination's core graph is cached the same way, per `(angle, centre, radius)`.

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use petgraph::graph::UnGraph;

use crate::defect::combine_types::{
    CoreCandidateInfo, CoreGalleryEntry, DislocationScanInfo, EquivalentSingleReport,
};
use crate::defect::core_gallery::{config_core_entry, gallery_from_bundle, resolve_patch_rim};
use crate::defect::core_graph::{LatticeCoreInfo, core_graph_with_rim, isomorphic_cores, patch_rim};
use crate::defect::equivalence::{classify_crystal_charge, fmt_signed_g, fmt_vec2_round4};
use crate::defect::surgery::{DefectSpec, build_point_defects, patch_centre_and_radius};
use crate::defect::sym_helper::round_dec;
use crate::defect::volterra_patch::wl_hash_labelled;
use crate::defect::volterra_scan::scan_dislocation_cores;
use crate::types::{BuiltCrystal, Cell2D, Frame2D};

const MAX_CANDIDATES: usize = 8;

#[derive(Clone, Debug, PartialEq)]
struct DislocKey {
    cell: Cell2D,
    b: [u64; 2],
    radius: usize,
    grid: usize,
    draw: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct DisclKey {
    cell: Cell2D,
    angle: u64,
    at: [u64; 2],
    radius: usize,
}

/// One scanned single dislocation: its distinct cores (name, labelled WL hash, core graph),
/// the candidate list, the scan summary and the gallery cards.
struct DislocScan {
    cores: Vec<(String, u64, UnGraph<u16, ()>)>,
    candidates: Vec<CoreCandidateInfo>,
    info: DislocationScanInfo,
    gallery: Vec<CoreGalleryEntry>,
}

/// Least-recently-used cache of single-defect scans, keyed like Python's `config._DISLOC_CACHE`.
pub struct SingleDefectCache {
    disloc: VecDeque<(DislocKey, Arc<DislocScan>)>,
    discl: VecDeque<(DisclKey, Arc<UnGraph<u16, ()>>)>,
    capacity: usize,
    hits: usize,
    misses: usize,
    last_match_ms: f64,
}

impl Default for SingleDefectCache {
    fn default() -> Self {
        Self::new(24)
    }
}

/// Bits of Python's `np.round(x, 6)`; unlike an integer-scaled key, this cannot saturate at i64::MAX.
fn key6(x: f64) -> u64 {
    round_dec(x, 6).to_bits()
}

/// Fingerprint of a cell (lattice vectors, basis and bond list, bit for bit).
pub fn cell_fingerprint(cell: &Cell2D) -> u64 {
    let mut h = DefaultHasher::new();
    for x in cell.a1.iter().chain(cell.a2.iter()).chain(cell.basis.iter().flatten()) {
        (*x + 0.0).to_bits().hash(&mut h);
    }
    cell.basis.len().hash(&mut h);
    cell.bonds.hash(&mut h);
    h.finish()
}

fn touch<K: PartialEq, V>(list: &mut VecDeque<(K, Arc<V>)>, key: &K) -> Option<Arc<V>> {
    let pos = list.iter().position(|(k, _)| k == key)?;
    let entry = list.remove(pos)?;
    let val = Arc::clone(&entry.1);
    list.push_back(entry);
    Some(val)
}

fn insert<K, V>(list: &mut VecDeque<(K, Arc<V>)>, capacity: usize, key: K, val: Arc<V>) {
    while list.len() >= capacity.max(1) {
        list.pop_front();
    }
    list.push_back((key, val));
}

impl SingleDefectCache {
    pub fn new(capacity: usize) -> Self {
        Self { disloc: VecDeque::new(), discl: VecDeque::new(), capacity, hits: 0, misses: 0, last_match_ms: 0.0 }
    }

    /// `(hits, misses)` since creation.
    pub fn stats(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }

    /// Last dislocation candidate-match time (WL + exact VF2), excluding the scan, configuration-core
    /// extraction and picture drawing. Intended for profiling, not a real-time guarantee.
    pub fn last_match_ms(&self) -> f64 { self.last_match_ms }

    pub fn len(&self) -> usize {
        self.disloc.len() + self.discl.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.disloc.clear();
        self.discl.clear();
    }

    fn dislocation(
        &mut self, cell: &Cell2D, b: [f64; 2], radius: usize, grid: usize, draw: bool,
    ) -> Result<Arc<DislocScan>, String> {
        // Retain the exact cell, not only a hash: a collision must never reuse another lattice's cores.
        let key = DislocKey { cell: cell.clone(), b: [key6(b[0]), key6(b[1])], radius, grid, draw };
        if let Some(hit) = touch(&mut self.disloc, &key) {
            self.hits += 1;
            return Ok(hit);
        }
        self.misses += 1;
        let bundle = scan_dislocation_cores(cell, b, radius, grid, MAX_CANDIDATES, None)?;
        let gallery = gallery_from_bundle(&bundle, radius, true, MAX_CANDIDATES, draw);
        let cores = bundle.cores.iter().map(|c| (c.info.name.clone(), c.wl_hash, c.graph.clone())).collect();
        let candidates = bundle.cores.iter().map(|c| c.info.clone()).collect();
        let scan = Arc::new(DislocScan { cores, candidates, info: bundle.info, gallery });
        insert(&mut self.disloc, self.capacity, key, Arc::clone(&scan));
        Ok(scan)
    }

    fn disclination_core(
        &mut self, lat_info: &LatticeCoreInfo, angle_deg: f64, at: [f64; 2], radius: usize,
    ) -> Result<Arc<UnGraph<u16, ()>>, String> {
        let cell = &lat_info.lat.cell;
        let key = DisclKey { cell: cell.clone(), angle: key6(angle_deg), at: [key6(at[0]), key6(at[1])], radius };
        if let Some(hit) = touch(&mut self.discl, &key) {
            self.hits += 1;
            return Ok(hit);
        }
        self.misses += 1;
        let single_spec = [DefectSpec::Disclination { angle_deg, at, cut: None }];
        let single = build_point_defects(cell, &single_spec, None, None)?;
        let (s_ctr, s_r) = patch_centre_and_radius(lat_info, &single_spec, None, None)?;
        let rim = patch_rim(&single, lat_info, s_ctr, s_r);
        let h = Arc::new(core_graph_with_rim(&single, &rim, lat_info, radius)?);
        insert(&mut self.discl, self.capacity, key, Arc::clone(&h));
        Ok(h)
    }
}

/// `equivalent_single` reusing (and filling) `cache`: identical results, the single-defect scans
/// are computed once per lattice, Burgers vector / disclination, radius, grid and `draw_gallery`.
#[allow(clippy::too_many_arguments)]
pub fn equivalent_single_cached(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    specs: &[DefectSpec],
    crystal: &BuiltCrystal,
    radius: usize,
    grid: usize,
    draw_gallery: bool,
    cache: &mut SingleDefectCache,
) -> Result<EquivalentSingleReport, String> {
    cache.last_match_ms = 0.0;
    let lat_info = LatticeCoreInfo::new(cell);
    let charge = classify_crystal_charge(group, cell, frame, crystal)?;
    let rim = resolve_patch_rim(crystal, &lat_info, specs);
    let h_cfg: UnGraph<u16, ()> = core_graph_with_rim(crystal, &rim, &lat_info, radius)?;
    let config_core = config_core_entry(crystal, &rim, &lat_info, radius, draw_gallery);

    let (mut candidates, mut matches, mut scan, mut gallery) = (Vec::new(), Vec::new(), None, Vec::new());
    if charge.kind == "dislocation" {
        let b = charge.b.unwrap_or([0.0, 0.0]);
        let found = cache.dislocation(cell, b, radius, grid, draw_gallery)?;
        gallery = found.gallery.clone();
        let match_start = Instant::now();
        let h_wl = wl_hash_labelled(&h_cfg);
        for (name, wl, graph) in &found.cores {
            if *wl == h_wl && isomorphic_cores(&h_cfg, graph) {
                matches.push(name.clone());
            }
        }
        cache.last_match_ms = match_start.elapsed().as_secs_f64() * 1000.0;
        candidates = found.candidates.clone();
        if matches.is_empty() && specs.len() == 1 && matches!(specs[0], DefectSpec::Dislocation { .. }) {
            matches.push("its own core (the configuration is one dislocation; its cut end is not among the scanned ones)".to_string());
        }
        scan = Some(found.info.clone());
    } else if charge.kind == "disclination"
        && let Some(DefectSpec::Disclination { angle_deg, at, .. }) = charge.single
    {
        let h_single = cache.disclination_core(&lat_info, angle_deg, at, radius)?;
        let cand_name = format!("single {} deg at {}", fmt_signed_g(angle_deg), fmt_vec2_round4(at));
        if isomorphic_cores(&h_cfg, &h_single) {
            matches.push(cand_name.clone());
        }
        candidates.push(CoreCandidateInfo {
            name: cand_name, volterra: None, rings: Vec::new(), under: Vec::new(), added: 0,
            lattice_bonds: 0, reconstruction_bonds: 0, bipartite: None, core_point: None, cut: None,
        });
    } else if charge.kind == "vacuum" && h_cfg.node_count() == 0 {
        matches.push("perfect lattice".to_string());
    }

    let equivalent = !matches.is_empty();
    let verdict = if equivalent {
        format!("equivalent to the single defect (core: {})", matches.join(", "))
    } else if let Some(ref sc) = scan {
        format!(
            "no match among {} distinct cores scanned (grid={}, {} Volterra builds, completions capped {} times, {} skipped at the rim); a larger grid may find more cores",
            sc.distinct, sc.grid, sc.builds, sc.capped, sc.skipped_rim
        )
    } else {
        "not equivalent to the single defect".to_string()
    };

    Ok(EquivalentSingleReport {
        kind: charge.kind.clone(), single: charge.single.clone(), charge, candidates, matches,
        equivalent, verdict, scan, core_nodes: h_cfg.node_count(), core_edges: h_cfg.edge_count(),
        config_core, gallery,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defect::core_gallery::equivalent_single;

    fn square() -> Cell2D {
        // Square lattice by hand: one site per cell, bonds along a1 and a2.
        Cell2D {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            basis: vec![[0.0, 0.0]],
            bonds: vec![
                crate::types::Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 },
                crate::types::Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 },
            ],
        }
    }

    #[test]
    fn cached_equivalent_single_equals_uncached_and_hits_on_moved_defects() {
        let cell = square();
        let mut cache = SingleDefectCache::default();
        // +90 at a site and -90 at a face centre of the square lattice: one dislocation (RUST_PLAN §9).
        let pair_a = [DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.5, 0.5])];
        // The same pair translated by a lattice vector: same Burgers vector, so the scan must be reused.
        let pair_b = [DefectSpec::disclination(90.0, [2.0, 1.0]), DefectSpec::disclination(-90.0, [2.5, 1.5])];
        for (k, specs) in [pair_a, pair_b].iter().enumerate() {
            let crystal = build_point_defects(&cell, specs, None, None).expect("pair builds");
            let fresh = equivalent_single("p4m", &cell, Frame2D::default(), specs, &crystal, 2, 1, false).unwrap();
            let cached = equivalent_single_cached("p4m", &cell, Frame2D::default(), specs, &crystal, 2, 1, false, &mut cache).unwrap();
            if k == 0 {
                assert_eq!(fresh, cached, "a cold cache computes exactly what equivalent_single does");
            } else {
                // The translated pair measures b with other float noise (same key after rounding to
                // 6 decimals, as in Python): same verdict, candidates and scan, cards reused.
                assert_eq!(
                    (&fresh.kind, &fresh.matches, fresh.equivalent, &fresh.verdict, &fresh.candidates, &fresh.scan),
                    (&cached.kind, &cached.matches, cached.equivalent, &cached.verdict, &cached.candidates, &cached.scan),
                );
                let names = |r: &EquivalentSingleReport| r.gallery.iter().map(|e| e.name.clone()).collect::<Vec<_>>();
                assert_eq!(names(&fresh), names(&cached));
            }
            assert_eq!(cached.kind, "dislocation");
            assert!(cached.equivalent, "+90 site / -90 face pair is a single dislocation");
            assert_eq!(cache.stats(), (k, 1), "second configuration must reuse the scan");
        }
    }

    #[test]
    fn equivalent_single_is_deterministic() {
        let cell = square();
        let specs = [DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [0.5, 0.5])];
        let crystal = build_point_defects(&cell, &specs, None, None).expect("pair builds");
        for draw in [false, true] {
            let a = equivalent_single("p4m", &cell, Frame2D::default(), &specs, &crystal, 2, 2, draw).unwrap();
            let b = equivalent_single("p4m", &cell, Frame2D::default(), &specs, &crystal, 2, 2, draw).unwrap();
            assert_eq!(a, b, "draw={draw}");
        }
    }

    #[test]
    fn cache_is_bounded_and_keyed_by_lattice() {
        let mut cache = SingleDefectCache::new(2);
        let cell = square();
        let mut wide = square();
        wide.a1 = [1.5, 0.0];
        assert_ne!(cell_fingerprint(&cell), cell_fingerprint(&wide));
        assert_eq!(cell_fingerprint(&cell), cell_fingerprint(&square()));
        let lat = LatticeCoreInfo::new(&cell);
        for at in [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]] {
            cache.disclination_core(&lat, 90.0, at, 2).unwrap();
        }
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats(), (0, 3));
        cache.disclination_core(&lat, 90.0, [2.0, 0.0], 2).unwrap();
        assert_eq!(cache.stats(), (1, 3));
    }
}
