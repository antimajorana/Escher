//! Hyperbolic `H^2` `(p, q)` tilings, rectified `(p.q.p.q)` kagome, dual `{q, p}`,
//! Poincare disk geodesics, Klein Beltrami chords, upper-sheet 3D hyperboloid bowl,
//! wedge disclinations, and `PSU(1,1)` loop holonomy.

use std::collections::BTreeSet;
use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::dsymbol::{DSymbol, GeometryKind};

pub mod boundary_counts;
#[cfg(test)]
mod draw_tests;
pub mod inflation;
#[cfg(test)]
mod inflation_tests;
mod kagome;
mod wedge;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HyperbolicStyle {
    RegularPQ,
    RectifiedKagome,
    DualQP,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HyperbolicTiling {
    pub p: usize,
    pub q: usize,
    pub style: HyperbolicStyle,
    pub layers: usize,
    pub wedge_delta: i32,
    pub symbol_label: String,
    pub curvature_kappa: f64,
    pub geometry_kind: GeometryKind,
    pub psu11_half_trace: f64,
    pub poincare_pts: Vec<[f64; 2]>,
    pub klein_pts: Vec<[f64; 2]>,
    pub hyperboloid_pts: Vec<[f64; 3]>,
    pub edges: Vec<(usize, usize)>,
    pub edge_arcs: Vec<Vec<[f64; 2]>>,
    pub faces: Vec<Vec<usize>>,
    /// Shell index of each face: `n` for faces in `S_n \ S_{n-1}` of vertex completion (0 = seed);
    /// on the wedge-core path, of the vertex completion of the core copies (module `wedge`).
    #[serde(default)]
    pub face_shell: Vec<usize>,
    /// Largest `n` with `S_n` fully drawn (on either path).
    #[serde(default)]
    pub complete_through_shell: usize,
    /// Which growth rule built the patch and which shells are complete or truncated.
    #[serde(default)]
    pub growth_rule: String,
}

#[inline]
pub(super) fn c_norm2(a: [f64; 2]) -> f64 { a[0] * a[0] + a[1] * a[1] }

/// Orthogonal circle center & R^2 for Poincare geodesic through A, B (or None if diameter).
fn orthogonal_circle(a: [f64; 2], b: [f64; 2]) -> Option<([f64; 2], f64)> {
    let det = a[0] * b[1] - a[1] * b[0];
    if det.abs() < 1e-10 {
        return None;
    }
    let rhs0 = 0.5 * (1.0 + c_norm2(a));
    let rhs1 = 0.5 * (1.0 + c_norm2(b));
    let cx = (rhs0 * b[1] - a[1] * rhs1) / det;
    let cy = (a[0] * rhs1 - rhs0 * b[0]) / det;
    let r2 = cx * cx + cy * cy - 1.0;
    (r2 > 0.0).then_some(([cx, cy], r2))
}

fn sample_geodesic_arc(a: [f64; 2], b: [f64; 2], segs: usize) -> Vec<[f64; 2]> {
    let Some((c, r2)) = orthogonal_circle(a, b) else {
        return vec![a, b];
    };
    let rad = r2.sqrt();
    let th0 = (a[1] - c[1]).atan2(a[0] - c[0]);
    let th1 = (b[1] - c[1]).atan2(b[0] - c[0]);
    let mut dth = (th1 - th0 + PI).rem_euclid(2.0 * PI) - PI;
    if dth.abs() < 1e-6 {
        dth = 0.0;
    }
    (0..=segs)
        .map(|k| {
            let th = th0 + dth * (k as f64) / (segs as f64);
            [c[0] + rad * th.cos(), c[1] + rad * th.sin()]
        })
        .collect()
}

pub fn to_klein(z: [f64; 2]) -> [f64; 2] {
    let d = 1.0 + c_norm2(z);
    [2.0 * z[0] / d, 2.0 * z[1] / d]
}

pub fn to_hyperboloid(z: [f64; 2]) -> [f64; 3] {
    let n2 = c_norm2(z).min(0.95 * 0.95);
    let scale = (n2 / c_norm2(z).max(1e-18)).sqrt().min(1.0);
    let x = z[0] * scale;
    let y = z[1] * scale;
    let d = (1.0 - (x * x + y * y)).max(0.05);
    [2.0 * x / d, 2.0 * y / d, (1.0 + x * x + y * y) / d]
}

/// Caps of the drawn patch: whole shells are kept while the face count stays within `max_faces`;
/// faces whose centre lies beyond Poincare radius `max_centre_radius` are dropped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DrawCaps {
    pub max_faces: usize,
    pub max_centre_radius: f64,
}

/// Default caps of [`build_hyperbolic_tiling`]. Pilot (2026-11-18, release, `draw_tests::pilot_draw_caps`):
/// the old 180 / 0.985 left {5,4}, {3,7}, {4,5} at S_2 for layers 2..4; the radius cap, not the face cap,
/// saturates the slider (1000 / 0.985 still stops {5,4} at S_2: its S_3 corner faces lie beyond 0.985).
/// 1000 / 0.998 draws S_1..S_4 whole for {5,4} (11, 51, 201, 761), {7,3}, {3,7}, {4,5} (913), and builds
/// all 92 pairs x 4 layers in ~1.5-1.8 s, worst single build ~21-25 ms. Layers 1..4 give >= 2 different
/// patches for 47 of the 92 pairs (4 different for 5), against 17 (0) under 180 / 0.985; the rest
/// grow too fast (S_2 alone > 1000 faces). 2000 / 0.998 gives 62 (7) at ~2x the time. At 0.998 the
/// drawn vertices drift off the hyperboloid by |<x, x> + 1| <= 4.4e-3 (vs 6.5e-5 at 0.985; at t ~ 500 that is a
/// relative error ~1e-8 in the Poincare point) with no dedup-band hit. Faces near radius 0.998 are
/// about a pixel wide; the 3D bowl (`to_hyperboloid`) clamps at radius 0.95, so they sit on its rim.
pub(crate) const DRAW_CAPS: DrawCaps = DrawCaps { max_faces: 1000, max_centre_radius: 0.998 };
/// Wedge path: face centres (and vertices) closer than this hyperbolic distance are one.
const WEDGE_MERGE_DIST: f64 = 1e-3;

pub(super) struct Patch {
    pts: Vec<[f64; 2]>,
    faces: Vec<Vec<usize>>,
    face_shell: Vec<usize>,
    complete_through_shell: usize,
    growth_rule: String,
}

pub(super) fn keep_whole_shells(shell_sizes: &[usize], stopped: &Option<String>, max_faces: usize) -> (usize, Vec<String>) {
    let mut keep = 0;
    let mut total = 0;
    for &a in shell_sizes {
        if total + a > max_faces {
            break;
        }
        total += a;
        keep += 1;
    }
    let mut notes = Vec::new();
    if let Some(&a) = shell_sizes.get(keep) {
        notes.push(format!("shell {keep} ({a} faces) omitted by the {max_faces}-face cap"));
    }
    match stopped {
        // The compute budget (twice the cap) only stops shells that cannot fit the cap: the estimate is
        // exact for shell 1 and exceeds shell n + 1 by at most a_{n-1} < cap (by the recurrence
        // a_{n+1} = 2 gamma a_n - a_{n-1}, GT-observed; a_2 = 2 gamma a_1 in the hand counts {7,3}, {5,4}, {3,7}).
        Some(s) if s.starts_with("face budget") && keep < shell_sizes.len() => {}
        Some(s) if s.starts_with("face budget") => {
            notes.push(format!("shell {} omitted by the {max_faces}-face cap (estimated larger)", shell_sizes.len()))
        }
        Some(s) => notes.push(s.clone()),
        None => {}
    }
    (keep, notes)
}

/// Regular core (`wedge_delta == 0`, non-kagome): BDF vertex completion `S_0 .. S_layers` from one `p`-gon.
/// Whole shells are kept while the total stays within `caps.max_faces`; faces whose centre lies
/// beyond Poincare radius `caps.max_centre_radius` are dropped (the seed never is), which truncates
/// that shell. `growth_rule` states which shells are complete.
fn vertex_completion_patch(p: usize, q: usize, lay: usize, caps: DrawCaps) -> Result<Patch, String> {
    let DrawCaps { max_faces, max_centre_radius } = caps;
    // The next-shell estimate of `grow_vertex_completion` is >= the true shell, so twice the cap
    // never stops a shell that would fit.
    let vc = inflation::grow_vertex_completion(p, q, lay, 2 * max_faces)?;
    let (keep, mut notes) = keep_whole_shells(&vc.shell_sizes, &vc.stopped, max_faces);
    let mut set = inflation::HypPointSet::new(inflation::hyp_dist(vc.base.vertices[0], vc.base.vertices[1]));
    let mut complete = keep - 1;
    let (mut faces, mut face_shell) = (Vec::new(), Vec::new());
    for (i, &s) in vc.shell.iter().enumerate().take_while(|&(_, &s)| s < keep) {
        if i > 0 && c_norm2(inflation::hyperboloid_to_poincare(vc.centre(i))) > max_centre_radius.powi(2) {
            complete = complete.min(s - 1);
            continue;
        }
        let pts3 = vc.face_vertices(i);
        faces.push(pts3.into_iter().map(|x| set.insert(x).0).collect());
        face_shell.push(s);
    }
    // The band guard sees `hyp_dist` only, not drift off the hyperboloid; for centres within the
    // default radius cap 0.998 (p, q <= 12) that vertex drift was measured <= 4.4e-3 in |<x, x> + 1|.
    if set.suspect > 0 {
        return Err(format!(
            "{{{p},{q}}}: f64 precision: {} drawn points have an ambiguous dedup distance",
            set.suspect
        ));
    }
    if complete + 1 < keep {
        notes.push(format!("shell {} truncated by the centre radius cap {max_centre_radius}", complete + 1));
    }
    let growth_rule = format!(
        "BDF vertex completion from one {p}-gon: S_0..S_{complete} complete, {} faces drawn (requested S_{lay}){}",
        faces.len(),
        notes.iter().map(|n| format!("; {n}")).collect::<String>()
    );
    let pts = set.pts.iter().map(|&x| inflation::hyperboloid_to_poincare(x)).collect();
    Ok(Patch { pts, faces, face_shell, complete_through_shell: complete, growth_rule })
}

/// Wedge-disclination core (`wedge_delta != 0`): vertex completion of copies of a regular
/// `p_core`-gon (see [`wedge`]), with hyperbolic-distance dedup. For `q = 3` this is the original
/// edge-BFS face set. The core edge length matches `{p, q}` (`cosh(l/2) = cos(pi/p) / sin(pi/q)`,
/// `sinh R_core = sinh(l/2) / sin(pi/p_core)`).
fn wedge_patch(p: usize, q: usize, style: HyperbolicStyle, lay: usize, p_core: usize, caps: DrawCaps) -> Patch {
    let half_edge_pq = ((PI / (p as f64)).cos() / (PI / (q as f64)).sin()).max(1.0).acosh();
    let rc_hyp = (half_edge_pq.sinh() / (PI / (p_core as f64)).sin()).asinh();
    let poly = wedge::Polygon::regular(p_core, rc_hyp);
    let g = wedge::grow_polygon_vertex_completion(&poly, q, lay, caps.max_faces, caps.max_centre_radius, WEDGE_MERGE_DIST);
    let kagome = style == HyperbolicStyle::RectifiedKagome;
    let (pts, faces) = wedge::polygon_faces_to_poincare(&poly, &g.faces, kagome, WEDGE_MERGE_DIST);
    let mut notes = g.notes;
    if g.near_coincident_merged > 0 {
        notes.push(format!(
            "{} near-coincident copies (centres {:.1e}..{:.1e} apart) merged",
            g.near_coincident_merged,
            0.5 * WEDGE_MERGE_DIST,
            WEDGE_MERGE_DIST
        ));
    }
    if g.near_coincident_kept > 0 {
        notes.push(format!(
            "{} near-coincident copies (centres {:.1e}..{:.1e} apart) kept",
            g.near_coincident_kept,
            WEDGE_MERGE_DIST,
            1.5 * WEDGE_MERGE_DIST
        ));
    }
    let walk = if q == 3 {
        "= edge-BFS for q = 3".to_string()
    } else {
        let steps = (q - 1) / 2;
        let unit = if steps == 1 { "step" } else { "steps" };
        let opp = if q.is_multiple_of(2) { " + centred opposite copy" } else { "" };
        format!("star walks of {steps} {unit} each side{opp}")
    };
    let growth_rule = format!(
        "vertex completion of {p_core}-gon wedge-core copies ({walk}): S_0..S_{} complete, {} faces drawn (requested S_{lay}){}",
        g.complete_through,
        faces.len(),
        notes.iter().map(|n| format!("; {n}")).collect::<String>()
    );
    Patch { pts, faces, face_shell: g.shell, complete_through_shell: g.complete_through, growth_rule }
}

/// Generate a finite patch of hyperbolic `{p, q}` / rectified `(p.q.p.q)` / dual `{q, p}` tiling.
///
/// For `wedge_delta == 0` the patch is BDF vertex completion `S_layers` of one `p`-gon (layers
/// clamped to 1..=4), trimmed to whole shells within [`DRAW_CAPS`] faces and to face centres within
/// its Poincare radius (see `growth_rule`). For `wedge_delta != 0` it is vertex completion of copies
/// of the wedge core (module [`wedge`]), with the same caps.
pub fn build_hyperbolic_tiling(
    p_in: usize,
    q_in: usize,
    style: HyperbolicStyle,
    layers: usize,
    wedge_delta: i32,
) -> Result<HyperbolicTiling, String> {
    build_hyperbolic_tiling_capped(p_in, q_in, style, layers, wedge_delta, DRAW_CAPS)
}

/// [`build_hyperbolic_tiling`] with explicit drawing caps (pilots and tests).
pub(crate) fn build_hyperbolic_tiling_capped(
    p_in: usize,
    q_in: usize,
    style: HyperbolicStyle,
    layers: usize,
    wedge_delta: i32,
    caps: DrawCaps,
) -> Result<HyperbolicTiling, String> {
    let (p, q) = match style {
        HyperbolicStyle::DualQP => (q_in, p_in),
        _ => (p_in, q_in),
    };
    if p < 3 || q < 3 || (p - 2) * (q - 2) <= 4 {
        return Err(format!("{{{p}, {q}}} is not hyperbolic (requires (p-2)(q-2) > 4)"));
    }
    let lay = layers.clamp(1, 4);
    let ds = DSymbol::new(vec![0], vec![0], vec![0], vec![p as u32], vec![q as u32])
        .map_err(|e| format!("D-symbol error: {e:?}"))?;
    let p_core = (p as i32 + wedge_delta).clamp(3, 12) as usize;
    let patch = match (wedge_delta, style) {
        (0, HyperbolicStyle::RectifiedKagome) => kagome::kagome_cluster_patch(p, q, lay, caps)?,
        (0, _) => vertex_completion_patch(p, q, lay, caps)?,
        (_, _) => wedge_patch(p, q, style, lay, p_core, caps),
    };
    let Patch { pts: poincare_pts, faces, face_shell, complete_through_shell, growth_rule } = patch;

    let mut edge_set: BTreeSet<(usize, usize)> = BTreeSet::new();
    for face_v in &faces {
        let m = face_v.len();
        for k in 0..m {
            let (u, v) = (face_v[k], face_v[(k + 1) % m]);
            if u != v {
                edge_set.insert((u.min(v), u.max(v)));
            }
        }
    }

    let edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    let edge_arcs: Vec<Vec<[f64; 2]>> = edges
        .iter()
        .map(|&(u, v)| sample_geodesic_arc(poincare_pts[u], poincare_pts[v], 8))
        .collect();
    let klein_pts: Vec<[f64; 2]> = poincare_pts.iter().copied().map(to_klein).collect();
    let hyperboloid_pts: Vec<[f64; 3]> = poincare_pts.iter().copied().map(to_hyperboloid).collect();

    let symbol_label = match style {
        HyperbolicStyle::RegularPQ => format!("{{{p_in},{q_in}}}"),
        HyperbolicStyle::RectifiedKagome => format!("({p_in}.{q_in}.{p_in}.{q_in}) kagome"),
        HyperbolicStyle::DualQP => format!("{{{q_in},{p_in}}} dual"),
    };
    let psu11_half_trace = if wedge_delta == 0 {
        1.0
    } else {
        (PI * (wedge_delta as f64).abs() / (p_core as f64)).cos().abs()
    };

    Ok(HyperbolicTiling {
        p,
        q,
        style,
        layers: lay,
        wedge_delta,
        symbol_label,
        curvature_kappa: ds.orbifold_curvature(),
        geometry_kind: ds.geometry_kind(),
        psu11_half_trace,
        poincare_pts,
        klein_pts,
        hyperboloid_pts,
        edges,
        edge_arcs,
        faces,
        face_shell,
        complete_through_shell,
        growth_rule,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyperbolic_tilings_and_dsymbol_curvatures() {
        let cube = DSymbol::new(vec![0], vec![0], vec![0], vec![4], vec![3]).unwrap();
        assert_eq!(cube.geometry_kind(), GeometryKind::Spherical);
        assert!(cube.orbifold_curvature() > 0.0);

        let sq = DSymbol::new(vec![0], vec![0], vec![0], vec![4], vec![4]).unwrap();
        assert_eq!(sq.geometry_kind(), GeometryKind::Euclidean);
        assert!(sq.is_flat());

        let t73 = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 2, 0).unwrap();
        assert_eq!(t73.geometry_kind, GeometryKind::Hyperbolic);
        assert!(t73.curvature_kappa < 0.0);
        assert!(t73.faces.len() >= 8);
        assert_eq!(t73.faces[0].len(), 7);
        assert!(t73.poincare_pts.iter().all(|z| z[0] * z[0] + z[1] * z[1] < 1.0));
        assert!(t73.hyperboloid_pts.iter().all(|p| p[2] >= 1.0 - 1e-9));

        let kag = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RectifiedKagome, 2, 0).unwrap();
        assert!(!kag.edges.is_empty());

        let cone6 = build_hyperbolic_tiling(7, 3, HyperbolicStyle::RegularPQ, 2, -1).unwrap();
        assert_eq!(cone6.faces[0].len(), 6);
    }
}
