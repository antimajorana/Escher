//! Seeded growth of the regular honeycombs (C1 WP-G2): one entry point for E³, H³ and S³.
//!
//! Two rules (plan §1 Q13, user answer 2026-11-19):
//! - **touching** (default): S_{k+1} = every cell at a vertex of S_k, via [`crate::growth::grow`];
//! - **face shells**: S_{k+1} = S_k plus every cell sharing a face with S_k (the finer rule,
//!   the old face-adjacency shells with a seed and a cell cap).
//!
//! Adapters: [`E3Complex`] (exact integers) for the four E³ kinds and {4,3,4}; the coset
//! adapter [`CoxeterComplex`] for S³ and H³. Refusals, never approximations:
//! - hyperideal or degenerate symbols return `Err` with the classification's [PROVED] reason;
//! - honeycombs with infinite cells stop with `Infinite { layer: 0 }` and no layer;
//! - a vertex seed at an ideal vertex stops with `Infinite { layer: 0 }`; other seeds of a
//!   honeycomb with ideal vertices stop with `Infinite { layer: 1 }` under the touching rule.
//!   Paracompact honeycombs are drawn by [`super::paracompact_ball`] instead.
//!
//! Drawing: all faces of the cells of S_K up to [`DRAW_ALL_FACES_MAX`] cells; above it only
//! the boundary faces (faces lying in exactly one cell of S_K), with `boundary_only` set and
//! no cell list.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use super::coxeter_gram::{classify_schlafli, HoneycombRegime};
use super::polychora_s3::stereographic;
use super::{CellComplex3d, ComplexBuilder, EuclideanKind};
use crate::growth::e3_complex::{E3Complex, E3_SCALE};
use crate::growth::engine::seed_cells;
use crate::growth::h3_complex::{CoxeterComplex, KeyPrecision, INFINITE_CELL_REASON};
use crate::growth::stats::Tally;
use crate::growth::{grow, CellComplex, Growth, GrowthCaps, GrowthError, LayerStats, SeedKind, Stopped};

/// Default cell cap of honeycomb growth. Q13 asked for 250 000; the night rule (round 7, ship the
/// largest cap that builds within a few seconds with exact merging) lowers it. Measured 2026-11-20
/// (release, one test thread, `growth::tests_h3::pilot_cell_seed_depth` and
/// `tests_growth::pilot_default_cap_with_drawing`): cell-seed S3 {5,3,4} = 59 647 in 3.0 s and
/// {4,3,5} = 107 359 in 2.3 s (4.5 s and 3.2 s with drawing) fit; {3,5,3} cell S3 = 211 867 takes
/// 11.6 s, so any cap from 107 359 to 211 866 shows the same touching-rule layers. Every merge was
/// below 6e-12 (tol 1e-4) with no key gap under 100 × tol. The slider's worst over-cap requests
/// (10 layers) stop on this cap in 0.6-5.4 s with drawing, the cap guard keeping the interned cells
/// near the cap.
pub const MAX_GROWTH_CELLS: usize = 120_000;
/// Layers offered by the slider (E³ reaches 10 under the cap; H³ stops earlier on the cap).
pub const MAX_GROWTH_LAYERS: u32 = 10;
/// Above this many cells only the boundary faces are drawn (Q13 answer).
pub const DRAW_ALL_FACES_MAX: usize = 10_000;

/// Growth rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrowthRule {
    #[default]
    Touching,
    FaceShells,
}

/// Which honeycomb.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoneycombKind {
    Schlafli(u32, u32, u32),
    Euclid(EuclideanKind),
}

impl HoneycombKind {
    /// `"{4,3,5}"`, `"4,3,5"` or an E³ kind name (`"cubic"`, `"bitruncated_cubic"`, ...).
    pub fn parse(s: &str) -> Result<Self, String> {
        let t = s.trim().trim_start_matches('{').trim_end_matches('}');
        let parts: Vec<&str> = t.split(',').map(str::trim).collect();
        if parts.len() == 3 {
            let n: Result<Vec<u32>, _> = parts.iter().map(|x| x.parse::<u32>()).collect();
            let n = n.map_err(|_| format!("bad Schläfli symbol {s:?}"))?;
            return Ok(Self::Schlafli(n[0], n[1], n[2]));
        }
        EuclideanKind::parse(s).map(Self::Euclid)
    }
}

/// Result of [`grow_honeycomb`].
#[derive(Clone, Debug, Serialize)]
pub struct HoneycombGrowth {
    pub name: String,
    pub regime: String,
    pub seed: SeedKind,
    pub rule: GrowthRule,
    /// `stats[k]` describes S_k (cells, new, boundary V/E, χ).
    pub stats: Vec<LayerStats>,
    pub stopped: Option<Stopped>,
    /// Key precision of the coset adapter (`None` for the exact E³ adapter).
    pub precision: Option<KeyPrecision>,
    /// Only boundary faces are in `complex` (more than [`DRAW_ALL_FACES_MAX`] cells).
    pub boundary_only: bool,
    /// The last layer built, for drawing (`cell_shell` = layer of each cell).
    pub complex: CellComplex3d,
}

/// Grow a honeycomb from its default seed of kind `seed` under `rule` and `caps`.
pub fn grow_honeycomb(kind: HoneycombKind, seed: SeedKind, rule: GrowthRule, caps: GrowthCaps) -> Result<HoneycombGrowth, String> {
    let (p, q, r) = match kind {
        HoneycombKind::Euclid(k) => return grow_e3(k, seed, rule, caps),
        HoneycombKind::Schlafli(p, q, r) => (p, q, r),
    };
    let cls = classify_schlafli(p, q, r);
    let name = format!("{{{p},{q},{r}}}");
    let regime = format!("{:?}", cls.regime);
    match cls.regime {
        HoneycombRegime::EuclideanE3 => return grow_e3(EuclideanKind::Cubic, seed, rule, caps),
        HoneycombRegime::HyperidealRefused | HoneycombRegime::DegenerateRefused => return Err(cls.reason),
        _ => {}
    }
    let empty = |stopped: Stopped| HoneycombGrowth {
        name: name.clone(),
        regime: regime.clone(),
        seed,
        rule,
        stats: Vec::new(),
        stopped: Some(stopped),
        precision: None,
        boundary_only: false,
        complex: CellComplex3d { name: name.clone(), regime: regime.clone(), ..Default::default() },
    };
    if cls.has_infinite_euclidean_cells {
        return Ok(empty(Stopped::Infinite { layer: 0, reason: INFINITE_CELL_REASON }));
    }
    let mut x = CoxeterComplex::new(p, q, r)?;
    let at = x.default_seed(seed)?;
    let s0 = match seed_cells(&mut x, &at) {
        Ok(s) => s,
        Err(e) => return Ok(empty(e.stopped_at(0))),
    };
    let g = match rule {
        GrowthRule::Touching => {
            x.begin_growth(caps);
            let g = grow(&mut x, s0, caps);
            x.end_growth();
            g
        }
        GrowthRule::FaceShells => {
            let q = FaceQueries {
                on_face: |x: &mut CoxeterComplex, f| x.cells_on_face_key(f),
                star: |x: &mut CoxeterComplex, v| x.cells_at_lookup(v),
                on_edge: |x: &mut CoxeterComplex, e| x.cells_on_edge_lookup(e),
            };
            face_shells(&mut x, s0, caps, &q)
        }
    };
    let (complex, boundary_only) = draw_coxeter(&x, &g, &name, &regime);
    Ok(HoneycombGrowth {
        name,
        regime,
        seed,
        rule,
        stats: g.stats,
        stopped: g.stopped,
        precision: Some(x.precision()),
        boundary_only,
        complex,
    })
}

fn grow_e3(kind: EuclideanKind, seed: SeedKind, rule: GrowthRule, caps: GrowthCaps) -> Result<HoneycombGrowth, String> {
    let mut x = E3Complex::new(kind);
    let at = x.default_seed(seed)?;
    let s0 = seed_cells(&mut x, &at).map_err(|e| format!("seed: {e:?}"))?;
    let g = match rule {
        GrowthRule::Touching => grow(&mut x, s0, caps),
        GrowthRule::FaceShells => {
            let q = FaceQueries {
                on_face: |x: &mut E3Complex, f| Ok(x.cells_on_face_key(f)),
                star: |x: &mut E3Complex, v| x.cells_at(v),
                on_edge: |x: &mut E3Complex, e| x.cells_on_edge(e),
            };
            face_shells(&mut x, s0, caps, &q)
        }
    };
    let (complex, boundary_only) = draw_e3(&x, &g, kind.label());
    Ok(HoneycombGrowth {
        name: kind.label().into(),
        regime: "EuclideanE3".into(),
        seed,
        rule,
        stats: g.stats,
        stopped: g.stopped,
        precision: None,
        boundary_only,
        complex,
    })
}

/// The incidence queries of [`face_shells`]: the cells on a face key, and the star and edge
/// queries used only to test membership for the boundary counts.
pub struct FaceQueries<F, S, E> {
    pub on_face: F,
    pub star: S,
    pub on_edge: E,
}

type Answer<C> = Result<Vec<C>, GrowthError>;

/// The face-shell rule with the engine's readout and stop conventions (`growth::engine`).
///
/// Unlike the touching rule, a vertex older than the last layer can still miss cells, so the
/// boundary is re-checked on every vertex and edge still open (a closed one stays closed).
/// An ideal vertex (`Infinite` star) is boundary and does not stop the face rule.
pub fn face_shells<X, F, S, E>(x: &mut X, seed: Vec<X::Cell>, caps: GrowthCaps, q: &FaceQueries<F, S, E>) -> Growth<X::Cell>
where
    X: CellComplex,
    F: Fn(&mut X, X::Key) -> Answer<X::Cell>,
    S: Fn(&mut X, X::Vertex) -> Answer<X::Cell>,
    E: Fn(&mut X, X::Key) -> Answer<X::Cell>,
{
    let mut seed = seed;
    seed.sort_unstable();
    seed.dedup();
    let mut g = Growth { layers: Vec::new(), stats: Vec::new(), stopped: None };
    if seed.is_empty() {
        return g;
    }
    if seed.len() > caps.max_cells {
        g.stopped = Some(Stopped::CellCap { layer: 0, partial: seed.len() });
        return g;
    }
    let mut in_s: HashSet<X::Cell> = seed.iter().copied().collect();
    let mut tally = Tally::<X>::default();
    let (mut open_v, mut open_e): (Vec<X::Vertex>, Vec<X::Key>) = (Vec::new(), Vec::new());
    g.layers.push(seed);
    let mut k = 0u32;
    loop {
        let layer = g.layers[k as usize].clone();
        let fresh = tally.add_cells(x, &layer);
        let mut err: Option<GrowthError> = None;
        open_v.extend(fresh.vertices);
        open_v.retain(|&v| match (q.star)(x, v) {
            Ok(cs) => cs.iter().any(|c| !in_s.contains(c)),
            Err(GrowthError::Infinite { .. }) => true,
            Err(e) => {
                err.get_or_insert(e);
                true
            }
        });
        open_e.extend(fresh.edges);
        open_e.retain(|&e| match (q.on_edge)(x, e) {
            Ok(cs) => cs.iter().any(|c| !in_s.contains(c)),
            Err(e2) => {
                err.get_or_insert(e2);
                true
            }
        });
        let n = in_s.len();
        g.stats.push(tally.stats(n, layer.len(), open_v.len(), open_e.len()));
        if k >= caps.max_layers {
            break;
        }
        let mut next = Vec::new();
        let mut capped = false;
        'cells: for &c in &layer {
            for f in x.faces_of(c) {
                match (q.on_face)(x, f) {
                    Ok(cs) => {
                        for c2 in cs {
                            if in_s.insert(c2) {
                                next.push(c2);
                                if n + next.len() > caps.max_cells {
                                    capped = true;
                                    break 'cells;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        err.get_or_insert(e);
                    }
                }
            }
        }
        if let Some(e) = err {
            g.stopped = Some(e.stopped_at(k + 1));
            break;
        }
        if capped {
            g.stopped = Some(Stopped::CellCap { layer: k + 1, partial: n + next.len() });
            break;
        }
        if next.is_empty() {
            g.stopped = Some(Stopped::Closed { layer: k });
            break;
        }
        next.sort_unstable();
        g.layers.push(next);
        k += 1;
    }
    g
}

/// Faces to draw: every face of S_K, or those in exactly one cell when S_K is large.
fn faces_to_draw<C: Copy, K: Copy + Eq + std::hash::Hash>(cells: &[(C, u32)], faces_of: impl Fn(C) -> Vec<K>) -> (Vec<(usize, Vec<K>)>, bool) {
    let boundary_only = cells.len() > DRAW_ALL_FACES_MAX;
    let lists: Vec<(usize, Vec<K>)> = cells.iter().enumerate().map(|(i, &(c, _))| (i, faces_of(c))).collect();
    if !boundary_only {
        return (lists, false);
    }
    let mut count: HashMap<K, u32> = HashMap::new();
    for (_, fs) in &lists {
        for f in fs {
            *count.entry(*f).or_default() += 1;
        }
    }
    let kept = lists.into_iter().map(|(i, fs)| (i, fs.into_iter().filter(|f| count[f] == 1).collect())).collect();
    (kept, true)
}

fn cells_by_layer<C: Copy>(g: &Growth<C>) -> Vec<(C, u32)> {
    let last = g.stats.len();
    g.layers.iter().take(last).enumerate().flat_map(|(k, l)| l.iter().map(move |&c| (c, k as u32))).collect()
}

fn draw_coxeter(x: &CoxeterComplex, g: &Growth<u32>, name: &str, regime: &str) -> (CellComplex3d, bool) {
    let cells = cells_by_layer(g);
    let face_ids: Vec<Vec<u32>> = cells.iter().map(|&(c, _)| x.faces_of(c)).collect();
    let (kept, boundary_only) = faces_to_draw(&cells, |c| x.faces_of(c));
    let mut b = ComplexBuilder::new(name, regime);
    let mut remap: HashMap<u32, usize> = HashMap::new();
    let mut coords: Vec<[f64; 4]> = Vec::new();
    for (i, keep) in kept {
        if boundary_only && keep.is_empty() {
            continue;
        }
        let cycles = x.face_cycles_of(cells[i].0);
        let mut faces = Vec::new();
        for (cyc, fid) in cycles.iter().zip(&face_ids[i]) {
            if !keep.contains(fid) {
                continue;
            }
            let ids = cyc
                .iter()
                .map(|&v| {
                    *remap.entry(v).or_insert_with(|| {
                        coords.push(x.vertex_coords4(v));
                        coords.len() - 1
                    })
                })
                .collect();
            faces.push(b.add_face(ids));
        }
        if !boundary_only {
            b.add_cell(faces, cells[i].1, false);
        }
    }
    let mut out = b.finish();
    if x.spherical {
        let w = x.fundamental_points();
        let unit = |a: [f64; 4]| {
            let n = a.iter().map(|t| t * t).sum::<f64>().sqrt();
            a.map(|t| -t / n)
        };
        let pole = [w[3], w[2], w[1], [0.3, -0.5, 0.7, 0.4]]
            .map(unit)
            .into_iter()
            .find(|pl| coords.iter().all(|v| v.iter().zip(pl).map(|(a, b)| a * b).sum::<f64>() < 1.0 - 1e-3))
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        out.points = stereographic(&coords, &pole);
        out.is_ideal = vec![false; coords.len()];
    } else {
        for c in &coords {
            let s = [c[1], c[2], c[3]];
            out.points.push(if x.ideal_vertices { s } else { s.map(|t| t / (1.0 + c[0])) });
            out.klein.push(s.map(|t| t / c[0]));
            out.is_ideal.push(x.ideal_vertices);
        }
    }
    out.coords4 = coords;
    (out, boundary_only)
}

fn draw_e3(x: &E3Complex, g: &Growth<[i32; 3]>, name: &str) -> (CellComplex3d, bool) {
    let cells = cells_by_layer(g);
    let (kept, boundary_only) = faces_to_draw(&cells, |c| x.faces_of(c));
    let mut b = ComplexBuilder::new(name, "EuclideanE3");
    let mut remap: HashMap<[i32; 3], usize> = HashMap::new();
    let s = f64::from(E3_SCALE);
    for (i, keep) in kept {
        let c = cells[i].0;
        let verts = x.vertices_of(c);
        let mut faces = Vec::new();
        for key in keep {
            // Vertices v of the face: (n·v − sum)·(sum − n·c) = 0, exact (the vector from the
            // centre to the face centre is normal to the face for all five cell shapes).
            let n = i32::from(key.n);
            let normal = [0, 1, 2].map(|d| key.sum[d] - n * c[d]);
            let on: Vec<[i32; 3]> =
                verts.iter().copied().filter(|v| (0..3).map(|d| (n * v[d] - key.sum[d]) * normal[d]).sum::<i32>() == 0).collect();
            let cyc = order_cycle(&on, &key.sum.map(|t| f64::from(t) / f64::from(n)), &normal.map(f64::from));
            let ids = cyc
                .iter()
                .map(|v| {
                    *remap.entry(*v).or_insert_with(|| {
                        let p = v.map(|t| f64::from(t) / s);
                        b.out.points.push(p);
                        b.out.coords4.push([p[0], p[1], p[2], 1.0]);
                        b.out.is_ideal.push(false);
                        b.out.points.len() - 1
                    })
                })
                .collect();
            faces.push(b.add_face(ids));
        }
        if !boundary_only {
            b.add_cell(faces, cells[i].1, false);
        }
    }
    (b.finish(), boundary_only)
}

/// Order the vertices of a planar convex polygon by angle about its centre.
fn order_cycle(vs: &[[i32; 3]], centre: &[f64; 3], normal: &[f64; 3]) -> Vec<[i32; 3]> {
    let d = |v: &[i32; 3]| [0, 1, 2].map(|i| f64::from(v[i]) - centre[i]);
    let u = d(&vs[0]);
    let w = super::cross3(normal, &u);
    let mut out: Vec<([i32; 3], f64)> = vs.iter().map(|v| (*v, super::dot3(&d(v), &w).atan2(super::dot3(&d(v), &u)))).collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out.into_iter().map(|(v, _)| v).collect()
}
