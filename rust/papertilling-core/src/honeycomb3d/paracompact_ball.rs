//! Paracompact H³ honeycombs as a ball truncation (plan §1 Q4, user answer: an R slider).
//!
//! Layers cannot be grown here ([PROVED] reasons in `growth::h3_complex`: an ideal vertex lies
//! in infinitely many cells; an infinite {4,4} or {6,3} cell has infinitely many vertices).
//! Instead the ball keeps every face whose centre lies within hyperbolic distance R of the
//! seed point: the centre of the central cell (finite cells) or of the central face (infinite
//! cells), which [`CoxeterRealisation::new`] puts at the origin of the ball.
//!
//! Faces are the orbit of the face centre w2 (always a finite point: its stabiliser
//! ⟨R0,R1,R3⟩ is finite). Neighbour discovery uses the whole honeycomb, never a drawn patch:
//! the faces next to g·F0 around its edges are g·h·R2·F0 and g·h·R3R2·F0 with h ∈ ⟨R0,R1⟩
//! (consecutive faces around an edge; they connect all r faces around it). The search explores
//! every face within R + D, D the step between consecutive face centres, and keeps those within
//! R. That this
//! reaches every face within R is [COMPUTATIONAL EVIDENCE]: the counts are unchanged when the
//! search radius is R + 2D (tested); it is not proved.
//!
//! Keys: finite points by raw spatial hyperboloid coordinates (`key_index` docs), ideal points
//! (vertices, and the centres of infinite cells) by direction. A precision gap is an error.

use serde::Serialize;

use super::coxeter_gram::{classify_schlafli, mat_mul, mat_vec, CoxeterRealisation, HoneycombRegime, Mat4, Vec4};
use super::key_index::KeyIndex;
use super::coxeter_gram::SiteOrbit;
use super::{CellComplex3d, ComplexBuilder};
use crate::growth::h3_complex::{IDEAL_VERTEX_REASON, INFINITE_CELL_REASON};

/// Largest number of faces a ball may explore.
pub const MAX_BALL_FACES: usize = 20_000;

/// A ball truncation of a paracompact honeycomb.
#[derive(Clone, Debug, Serialize)]
pub struct ParacompactBall {
    pub name: String,
    pub radius: f64,
    pub faces: usize,
    pub cells: usize,
    /// D: the distance between the centres of two faces consecutive around an edge.
    pub step: f64,
    /// Why the layer slider is off for this honeycomb ([PROVED]).
    pub reason: &'static str,
    /// Cells are drawn with their faces inside the ball; `cell_truncated` marks a cell with
    /// faces outside it (every infinite cell).
    pub complex: CellComplex3d,
}

fn group(gens: &[Mat4], probe: &Vec4) -> Vec<Mat4> {
    let mut idx = KeyIndex::new(4, 1e-9);
    idx.insert(*probe);
    let mut out = vec![super::coxeter_gram::identity4()];
    let mut i = 0;
    while i < out.len() {
        for s in gens {
            let h = mat_mul(&out[i], s);
            if idx.insert(mat_vec(&h, probe)).1 {
                out.push(h);
            }
        }
        i += 1;
    }
    out
}

fn spatial(x: &Vec4) -> Vec4 {
    [x[1], x[2], x[3], 0.0]
}

fn direction(x: &Vec4) -> Vec4 {
    let n = (x[1] * x[1] + x[2] * x[2] + x[3] * x[3]).sqrt();
    [x[1] / n, x[2] / n, x[3] / n, 0.0]
}

/// Faces within `radius` of the seed point, and the faces explored within `search`.
fn faces_within(real: &CoxeterRealisation, steps: &[Mat4], radius: f64, search: f64) -> Result<Vec<Mat4>, String> {
    let w2 = real.w[2];
    let mut idx = KeyIndex::new(3, 1e-4);
    idx.insert(spatial(&w2));
    let mut all = vec![super::coxeter_gram::identity4()];
    let mut i = 0;
    while i < all.len() {
        let g = all[i];
        i += 1;
        for s in steps {
            let h = mat_mul(&g, s);
            let x = mat_vec(&h, &w2);
            if x[0] <= search.cosh() && idx.insert(spatial(&x)).1 {
                all.push(h);
                if all.len() > MAX_BALL_FACES {
                    return Err(format!("a ball of radius {radius} explores more than {MAX_BALL_FACES} faces; lower R"));
                }
            }
        }
    }
    if idx.min_gap.is_finite() {
        return Err(format!("face keys came {:.2e} apart (precision)", idx.min_gap));
    }
    // A face centred exactly on the bounding sphere is inside, including the seed at R=0.
    let limit = radius.cosh() * (1.0 + 8.0 * f64::EPSILON);
    Ok(all.into_iter().filter(|g| mat_vec(g, &w2)[0] <= limit).collect())
}

/// The ball truncation of the paracompact honeycomb {p,q,r} at radius `radius`; `search_steps`
/// (normally 1) is how many face steps D beyond R the search explores.
pub fn paracompact_ball_with(p: u32, q: u32, r: u32, radius: f64, search_steps: u32) -> Result<ParacompactBall, String> {
    build_ball(p, q, r, radius, search_steps, false)
}

/// Faces of the central infinite cell whose centres lie within hyperbolic radius R of the seed face.
pub fn single_cell_patch(p: u32, q: u32, r: u32, radius: f64) -> Result<ParacompactBall, String> {
    build_ball(p, q, r, radius, 1, true)
}

fn build_ball(p: u32, q: u32, r: u32, radius: f64, search_steps: u32, seed_only: bool) -> Result<ParacompactBall, String> {
    let cls = classify_schlafli(p, q, r);
    if cls.regime != HoneycombRegime::HyperbolicParacompactH3 {
        return Err(format!("{{{p},{q},{r}}} is not paracompact: grow it by layers instead ({})", cls.reason));
    }
    if seed_only && !cls.has_infinite_euclidean_cells {
        return Err("For a finite cell use the complete layer-zero construction.".into());
    }
    if !(radius.is_finite() && radius >= 0.0) {
        return Err(format!("radius must be a finite number >= 0, got {radius}"));
    }
    let real = CoxeterRealisation::new(p, q, r)?;
    let rr = &real.reflections;
    let xi: Vec4 = std::array::from_fn(|i| real.w.iter().map(|w| w[i]).sum::<f64>());
    // Around the edge e0 the faces are (R2 R3)^j F0 and R3 fixes F0, so the two faces next to F0
    // are R2 F0 and R3 R2 F0; h ∈ ⟨R0,R1⟩ moves them to the other edges of F0.
    let r3r2 = mat_mul(&rr[3], &rr[2]);
    let mut steps = Vec::new();
    for h in group(&[rr[0], rr[1]], &xi) {
        steps.push(mat_mul(&h, &rr[2]));
        // R0, R1, R2 preserve the seed cell. R3 crosses to the neighbouring cell.
        if !seed_only { steps.push(mat_mul(&h, &r3r2)); }
    }
    let w2 = real.w[2];
    let step = steps.iter().map(|s| mat_vec(s, &w2)).map(|x| (x[0] * w2[0] - x[1] * w2[1] - x[2] * w2[2] - x[3] * w2[3]).max(1.0).acosh()).fold(0.0, f64::max);
    let kept = faces_within(&real, &steps, radius, radius + f64::from(search_steps) * step)?;
    let infinite = cls.has_infinite_euclidean_cells;
    let ideal = cls.has_ideal_vertices;
    let name = format!("{{{p},{q},{r}}}");
    let mut b = ComplexBuilder::new(&name, "HyperbolicParacompactH3");
    let mut vidx = KeyIndex::new(3, if ideal { 1e-9 } else { 1e-4 });
    let mut cidx = KeyIndex::new(3, if infinite { 1e-9 } else { 1e-4 });
    let mut cell_faces: Vec<Vec<usize>> = Vec::new();
    let rot = mat_mul(&rr[0], &rr[1]);
    let mut coords: Vec<Vec4> = Vec::new();
    let (site, _) = cls.finite_site_orbit().expect("classified H³");
    let klein = |x: Vec4| [1, 2, 3].map(|i| x[i] / x[0]);
    let mut orbit_centres = Vec::new();
    let mut edge_centres = KeyIndex::new(3, 1e-4);
    for g in &kept {
        let mut cyc = Vec::with_capacity(p as usize);
        let mut h = *g;
        for _ in 0..p {
            if site == SiteOrbit::Edge {
                let centre = mat_vec(&h, &real.w[1]);
                if edge_centres.insert(spatial(&centre)).1 {
                    orbit_centres.push(klein(centre));
                }
            }
            let x = mat_vec(&h, &real.w[0]);
            let key = if ideal { direction(&x) } else { spatial(&x) };
            let (vid, new) = vidx.insert(key);
            if new {
                coords.push(x);
            }
            cyc.push(vid as usize);
            h = mat_mul(&h, &rot);
        }
        if site == SiteOrbit::Face {
            orbit_centres.push(klein(mat_vec(g, &real.w[2])));
        }
        let f = b.add_face(cyc);
        let neighbours = [*g, mat_mul(g, &rr[3])];
        for m in &neighbours[..if seed_only { 1 } else { 2 }] {
            let c = mat_vec(m, &real.w[3]);
            let (cid, new) = cidx.insert(if infinite { direction(&c) } else { spatial(&c) });
            if new {
                if site == SiteOrbit::Cell {
                    orbit_centres.push(klein(c));
                }
                cell_faces.push(Vec::new());
            }
            cell_faces[cid as usize].push(f);
        }
    }
    if vidx.min_gap.is_finite() || cidx.min_gap.is_finite() || edge_centres.min_gap.is_finite() {
        return Err("vertex or cell keys came closer than 100 × tolerance (precision)".into());
    }
    // Faces of a finite cell {p,q}: F = 4q / (2p + 2q − pq) (Euler's formula).
    let full = if infinite { usize::MAX } else { (4 * q / (2 * p + 2 * q - p * q)) as usize };
    let ncells = cell_faces.len();
    for fs in cell_faces {
        let truncated = fs.len() < full;
        b.add_cell(fs, 0, truncated);
    }
    let mut cx = b.finish();
    cx.orbit_centres = orbit_centres;
    for x in &coords {
        if ideal {
            let s = direction(x);
            let s = [s[0], s[1], s[2]];
            cx.coords4.push([1.0, s[0], s[1], s[2]]);
            cx.points.push(s);
            cx.klein.push(s);
        } else {
            let n = (x[0] * x[0] - x[1] * x[1] - x[2] * x[2] - x[3] * x[3]).sqrt();
            let y = x.map(|t| t / n);
            cx.coords4.push(y);
            cx.points.push([1, 2, 3].map(|i| y[i] / (1.0 + y[0])));
            cx.klein.push([1, 2, 3].map(|i| y[i] / y[0]));
        }
        cx.is_ideal.push(ideal);
    }
    Ok(ParacompactBall {
        name,
        radius,
        faces: kept.len(),
        cells: ncells,
        step,
        reason: if infinite { INFINITE_CELL_REASON } else { IDEAL_VERTEX_REASON },
        complex: cx,
    })
}

/// [`paracompact_ball_with`] searching one face step beyond R.
pub fn paracompact_ball(p: u32, q: u32, r: u32, radius: f64) -> Result<ParacompactBall, String> {
    paracompact_ball_with(p, q, r, radius, 1)
}
