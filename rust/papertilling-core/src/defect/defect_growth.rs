//! Defected lattices grown by Dirichlet-seeded inflation (C1 WP-D1, plan §5 T7, Q8).
//!
//! The defected crystal is finite and not periodic, so growth runs on its own face graph
//! ([`crate::growth::face_graph`]) with the touching rule of plan §2.1.
//!
//! **Seed (Q8).** Take the perfect lattice's `DirichletAt(core)` faces (the faces whose centroids lie in the closed
//! Wigner-Seitz cell about the core, ties included, as `plane::inflation` defines it) and carry them through the
//! surgery's site map:
//! - a crystal face whose polygon, developed in the reference chart of one of its own sites
//!   ([`develop_face`]), is exactly a perfect seed face, is in the seed. A seed face copied by an inserted wedge has
//!   one such copy per sheet, and each copy is in the seed;
//! - a perfect seed face with no whole copy was cut (or removed) by the surgery. For it the seed takes the crystal's
//!   *defect faces* with a corner at one of its corners: faces that are not a perfect-lattice face in any of their
//!   corners' charts (the triangle or pentagon at a disclination apex, the odd ring at a dislocation core).
//!
//! **Complete layers.** A layer is complete when none of its cells has a vertex on the crystal's rim; only complete
//! layers are returned, and [`Stopped::Rim`] names the first incomplete one. This is stricter than the engine's own
//! rim stop, which keeps a layer that touches the rim and does not check the last requested layer.
//!
//! **Rebuild loop.** When fewer layers than asked are complete, the crystal is rebuilt at radius × 1.5, at most
//! [`MAX_REBUILDS`] times, or until the projected face count passes the cell cap.
//!
//! **Positions.** Faces are drawn at the positions of the crystal they were grown on; with `relax`, that crystal is
//! relaxed once as a whole ([`relax_picture`]) and S_n is not relaxed again on its own.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::defect::core_graph::{LatticeCoreInfo, PerfectLattice};
use crate::defect::surgery::{DefectSpec, build_point_defects, patch_centre_and_radius};
use crate::growth::face_graph::{CrystalComplex, crystal_complex, develop_face};
use crate::growth::{CellComplex, Growth, GrowthCaps, Stopped, grow};
use crate::plane::inflation::{InflationSeed, PlaneTiling};
use crate::plane::relax::relax_picture;
use crate::types::{BuiltCrystal, Cell2D};

/// Default cell cap for defected-crystal growth.
pub const DEFECT_MAX_CELLS: usize = 20_000;
/// At most this many rebuilds at a larger radius.
pub const MAX_REBUILDS: u32 = 3;
/// Radius factor per rebuild.
pub const REBUILD_FACTOR: f64 = 1.5;

/// Position key: coordinates rounded to 1e-6.
type PKey = (i64, i64);

fn pkey(p: [f64; 2]) -> PKey {
    ((p[0] * 1e6).round() as i64, (p[1] * 1e6).round() as i64)
}

fn poly_key(poly: &[[f64; 2]]) -> Vec<PKey> {
    let mut k: Vec<PKey> = poly.iter().map(|&p| pkey(p)).collect();
    k.sort_unstable();
    k
}

/// How the seed was carried through the surgery.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SeedReport {
    /// |DirichletAt(core)| in the perfect lattice.
    pub perfect_faces: usize,
    /// Perfect seed faces found whole in the crystal (at least one copy).
    pub carried: usize,
    /// Perfect seed faces with no whole copy (cut or removed by the surgery).
    pub missing: usize,
    /// Defect faces added for the missing ones.
    pub core_faces: usize,
    /// The seed: crystal face ids, sorted.
    pub cells: Vec<u32>,
}

/// Growth on one crystal: its complex, the seed, and the complete layers only.
#[derive(Clone, Debug)]
pub struct CrystalGrowth {
    pub complex: CrystalComplex,
    pub seed: SeedReport,
    pub growth: Growth<u32>,
}

impl CrystalGrowth {
    /// Number of complete layers (layers 0..complete-1 are shown).
    pub fn complete_layers(&self) -> usize {
        self.growth.layers.len()
    }
}

/// Is crystal face `walk` a perfect-lattice face in the chart of one of its corners?
fn is_perfect_face(crystal: &BuiltCrystal, walk: &[usize], lat: &PerfectLattice, tiling: &PlaneTiling) -> bool {
    (0..walk.len()).any(|b| {
        let Some(dev) = develop_face(crystal, walk, b) else { return false };
        let Some((n1, n2, s)) = lat.locate(dev[0]) else { return false };
        let key = poly_key(&dev);
        tiling.vertex_faces((s, [n1, n2])).into_iter().any(|f| poly_key(&tiling.polygon(f)) == key)
    })
}

/// The Q8 seed (module docs) on crystal `cc` about `core`. `lat` is the perfect uncoloured lattice the surgery used.
pub fn dirichlet_seed(crystal: &BuiltCrystal, cc: &CrystalComplex, lat: &PerfectLattice, core: [f64; 2]) -> Result<SeedReport, String> {
    let tiling = PlaneTiling::new(&lat.cell)?;
    let mut perfect: Vec<Vec<[f64; 2]>> =
        tiling.seed_faces(InflationSeed::DirichletAt(core))?.into_iter().map(|f| tiling.polygon(f)).collect();
    perfect.sort_by_key(|p| poly_key(p));
    let index: HashMap<Vec<PKey>, usize> = perfect.iter().enumerate().map(|(i, p)| (poly_key(p), i)).collect();
    let mut found = vec![false; perfect.len()];
    let mut cells: HashSet<u32> = HashSet::new();
    for (c, w) in cc.faces.iter().enumerate() {
        for b in 0..w.len() {
            if let Some(i) = develop_face(crystal, w, b).and_then(|d| index.get(&poly_key(&d)).copied()) {
                found[i] = true;
                cells.insert(c as u32);
            }
        }
    }
    let mut core_faces = 0;
    let missing: Vec<usize> = (0..perfect.len()).filter(|&i| !found[i]).collect();
    if !missing.is_empty() {
        let corners: HashSet<PKey> = missing.iter().flat_map(|&i| perfect[i].iter().map(|&p| pkey(p))).collect();
        for (c, w) in cc.faces.iter().enumerate() {
            let touches = w.iter().any(|&s| corners.contains(&pkey(crystal.ref_pos[s])));
            if touches && !cells.contains(&(c as u32)) && !is_perfect_face(crystal, w, lat, &tiling) {
                cells.insert(c as u32);
                core_faces += 1;
            }
        }
    }
    if cells.is_empty() {
        return Err(format!(
            "no face of the crystal carries the Dirichlet seed about ({:.3}, {:.3}): the core is outside the crystal or at its rim",
            core[0], core[1]
        ));
    }
    let mut cells: Vec<u32> = cells.into_iter().collect();
    cells.sort_unstable();
    Ok(SeedReport { perfect_faces: perfect.len(), carried: perfect.len() - missing.len(), missing: missing.len(), core_faces, cells })
}

/// Grow the Q8 seed on `crystal` for up to `max_layers` layers, keeping complete layers only (module docs).
pub fn grow_crystal(
    crystal: &BuiltCrystal,
    lat: &PerfectLattice,
    core: [f64; 2],
    max_layers: u32,
    max_cells: usize,
) -> Result<CrystalGrowth, String> {
    let mut cc = crystal_complex(crystal)?;
    let seed = dirichlet_seed(crystal, &cc, lat, core)?;
    let mut g = grow(&mut cc.complex, seed.cells.clone(), GrowthCaps { max_layers, max_cells });
    let rim: HashSet<u32> = cc.rim.iter().map(|&s| s as u32).collect();
    let first_bad = g
        .layers
        .iter()
        .position(|layer| layer.iter().any(|&c| cc.complex.vertices_of(c).iter().any(|v| rim.contains(v))));
    if let Some(k) = first_bad {
        g.layers.truncate(k);
        g.stats.truncate(k);
        g.stopped = Some(Stopped::Rim { layer: k as u32 });
    }
    Ok(CrystalGrowth { complex: cc, seed, growth: g })
}

/// Result of the rebuild loop.
#[derive(Clone, Debug)]
pub struct DefectGrowth {
    /// The crystal the layers live on (relaxed when asked).
    pub crystal: BuiltCrystal,
    /// Its radius.
    pub radius: f64,
    /// Rebuilds made (0 when the first crystal was large enough).
    pub rebuilds: u32,
    pub result: CrystalGrowth,
    /// Readout notes: why fewer layers than asked, and where the positions come from.
    pub note: String,
}

/// Build with `build(radius)` from `r0`, grow, and rebuild at radius × [`REBUILD_FACTOR`] while fewer than
/// `layers + 1` layers are complete (at most [`MAX_REBUILDS`] times, and not once the projected face count passes
/// `max_cells`). A failed rebuild keeps the last good result and says why.
pub fn grow_with_rebuild<F>(mut build: F, r0: f64, lat: &PerfectLattice, core: [f64; 2], layers: u32, max_cells: usize) -> Result<DefectGrowth, String>
where
    F: FnMut(f64) -> Result<BuiltCrystal, String>,
{
    if !(r0.is_finite() && r0 > 0.0) {
        return Err("the start radius must be finite and positive".into());
    }
    let mut r = r0;
    let mut best: Option<DefectGrowth> = None;
    let mut notes: Vec<String> = Vec::new();
    for attempt in 0..=MAX_REBUILDS {
        let crystal = match build(r) {
            Ok(c) => c,
            Err(e) if best.is_some() => {
                notes.push(format!("rebuild at radius {r:.2} failed: {e}"));
                break;
            }
            Err(e) => return Err(e),
        };
        let res = grow_crystal(&crystal, lat, core, layers, max_cells)?;
        let enough = res.complete_layers() > layers as usize || !matches!(res.growth.stopped, Some(Stopped::Rim { .. }));
        let faces = res.complex.faces.len();
        best = Some(DefectGrowth { crystal, radius: r, rebuilds: attempt, result: res, note: String::new() });
        if enough {
            break;
        }
        if (faces as f64) * REBUILD_FACTOR * REBUILD_FACTOR > max_cells as f64 {
            notes.push(format!("a larger crystal would pass the {max_cells}-cell cap"));
            break;
        }
        r *= REBUILD_FACTOR;
    }
    let mut out = best.expect("the first build either succeeded or returned");
    if let Some(Stopped::Rim { layer }) = out.result.growth.stopped
        && (layer as usize) <= layers as usize
    {
        let shown = if layer == 0 { "no complete layer".to_string() } else { format!("showing layers 0..{}", layer - 1) };
        notes.insert(
            0,
            format!("layer {layer} reaches the crystal's rim at radius {:.2} after {} rebuild(s); {shown}", out.radius, out.rebuilds),
        );
    }
    out.note = notes.join("; ");
    Ok(out)
}

/// The core of a set of point defects on `cell`: the mean of their positions, resolving `Dislocation { at: None, .. }`
/// at slot `idx` with the exact `(0.371 + 3·idx)·a₁ + 0.413·a₂` formula of [`crate::defect::surgery`].
pub fn defect_core_in_cell(cell: &Cell2D, specs: &[DefectSpec]) -> Result<[f64; 2], String> {
    if specs.is_empty() {
        return Err("give at least one defect spec".into());
    }
    let mut sum = [0.0, 0.0];
    for (idx, s) in specs.iter().enumerate() {
        let p = match s {
            DefectSpec::Dislocation { at, .. } => at.unwrap_or([
                (0.371 + 3.0 * idx as f64) * cell.a1[0] + 0.413 * cell.a2[0],
                (0.371 + 3.0 * idx as f64) * cell.a1[1] + 0.413 * cell.a2[1],
            ]),
            DefectSpec::Disclination { at, .. } => *at,
            DefectSpec::HalfWall { .. } => return Err("growth of walls and half-walls comes later (Q8: point defects first)".into()),
        };
        sum = [sum[0] + p[0], sum[1] + p[1]];
    }
    let n = specs.len() as f64;
    Ok([sum[0] / n, sum[1] / n])
}

/// The core of a set of point defects: the mean of their positions (a dislocation without `at` sits at `patch_centre`).
/// Prefer [`defect_core_in_cell`] when `cell` is available so multi-defect `at: None` specs match surgery's per-slot position.
pub fn defect_core(specs: &[DefectSpec], patch_centre: [f64; 2]) -> Result<[f64; 2], String> {
    if specs.is_empty() {
        return Err("give at least one defect spec".into());
    }
    let mut sum = [0.0, 0.0];
    for s in specs {
        let p = match s {
            DefectSpec::Dislocation { at, .. } => at.unwrap_or(patch_centre),
            DefectSpec::Disclination { at, .. } => *at,
            DefectSpec::HalfWall { .. } => return Err("growth of walls and half-walls comes later (Q8: point defects first)".into()),
        };
        sum = [sum[0] + p[0], sum[1] + p[1]];
    }
    let n = specs.len() as f64;
    Ok([sum[0] / n, sum[1] / n])
}

/// A start radius for `layers` layers about `core` in the perfect lattice: d₀ + (layers + 1)·D + ℓ, where d₀ is the
/// largest distance from `core` to a corner of the perfect `DirichletAt(core)` faces, D the largest face diameter of
/// the tiling and ℓ the longest bond. In the perfect lattice every face of S_{k+1} touches a vertex of S_k, so S_n lies
/// within d₀ + n·D of the core, and one more D keeps the faces at its boundary vertices inside the crystal. Near a
/// defect the faces and distances differ, so this is only a start; the rebuild loop corrects it.
pub fn start_radius(lat: &PerfectLattice, core: [f64; 2], layers: u32) -> Result<f64, String> {
    let tiling = PlaneTiling::new(&lat.cell)?;
    let dist = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).hypot(p[1] - q[1]);
    let d0 = tiling
        .seed_faces(InflationSeed::DirichletAt(core))?
        .into_iter()
        .flat_map(|f| tiling.polygon(f))
        .map(|p| dist(p, core))
        .fold(0.0_f64, f64::max);
    let diam = (0..tiling.orbit_count())
        .map(|o| {
            let p = tiling.polygon((o, [0, 0]));
            p.iter().flat_map(|&a| p.iter().map(move |&b| dist(a, b))).fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max);
    let bond = lat
        .rule
        .iter()
        .enumerate()
        .flat_map(|(s, r)| r.iter().map(move |&(d1, d2, s2, _)| (s, d1, d2, s2)))
        .map(|(s, d1, d2, s2)| dist(lat.pos((0, 0, s)), lat.pos((d1, d2, s2))))
        .fold(0.0_f64, f64::max);
    Ok(d0 + f64::from(layers + 1) * diam + bond)
}

/// Grow point defects `specs` on `cell` by `layers` layers from the Q8 seed, starting at radius `r_opt` (or the
/// larger of the surgery's own radius and [`start_radius`]) and rebuilding as needed. With `relax`, the final crystal
/// is relaxed once and the note says so.
pub fn grow_point_defects(
    cell: &Cell2D,
    specs: &[DefectSpec],
    layers: u32,
    r_opt: Option<f64>,
    relax: bool,
    max_cells: usize,
) -> Result<DefectGrowth, String> {
    if specs.iter().any(|s| matches!(s, DefectSpec::HalfWall { .. })) {
        return Err("growth of walls and half-walls comes later (Q8: point defects first)".into());
    }
    let info = LatticeCoreInfo::new(cell);
    let (ctr, r_min) = patch_centre_and_radius(&info, specs, r_opt, None)?;
    let core = defect_core_in_cell(cell, specs)?;
    let r0 = match r_opt {
        Some(r) => r,
        // the patch is centred at `ctr`, not at the core, so add the offset between them
        None => r_min.max(start_radius(&info.lat, core, layers)? + (ctr[0] - core[0]).hypot(ctr[1] - core[1])),
    };
    let mut out = grow_with_rebuild(|r| build_point_defects(cell, specs, Some(r), Some(ctr)), r0, &info.lat, core, layers, max_cells)?;
    if relax {
        relax_picture(&mut out.crystal);
        let msg = format!("positions from a radius-{:.2} relaxation", out.radius);
        out.note = if out.note.is_empty() { msg } else { format!("{}; {msg}", out.note) };
    }
    Ok(out)
}
