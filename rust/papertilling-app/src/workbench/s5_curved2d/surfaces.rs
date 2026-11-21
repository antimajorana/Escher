//! An H² {p,q} patch placed on a surface in R³ (`embed_tiling_on_surface`) and the papercraft net of the placed
//! mesh (`unfold_papercraft_net`), retained for unit tests and session state compatibility (`SurfState`).

#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::f64::consts::PI;
#[cfg(test)]
use std::path::Path;

use papertilling_core::dsymbol::GeometryKind;
use papertilling_core::growth::SeedKind;
#[cfg(test)]
use papertilling_core::hyperbolic::{HyperbolicStyle, build_hyperbolic_tiling_seeded};
#[cfg(test)]
use papertilling_core::noneuclidean2d::{
    H2SurfaceKind, build_unified_pq_seeded, embed_tiling_on_surface, surface_point, unfold_papercraft_net,
};
use serde_json::{Value, json};

#[cfg(test)]
use super::draw::{Picture, mesh_with_arcs};
use super::growth_table::parse_seed_2d;
#[cfg(test)]
use super::growth_table::{GROWTH_HEADERS, non_disk_flags, patch_layer_stats, seed_label_2d, stats_to_rows};
use super::pq::layer_range;
use super::{MAX_PQ, get_usize};
#[cfg(test)]
use super::{S5Out, check_symbol, geom_label};
#[cfg(test)]
use crate::workbench::scene_helpers::ring_size_tint;

pub(crate) const SURFACES: [&str; 5] = ["Hyperboloid bowl", "Hemisphere", "Pseudosphere", "Dini surface", "Kuen surface"];

/// The surface for index `k` (Dini with the given twist b/a).
#[cfg(test)]
pub(crate) fn surface_kind(k: usize, twist: f64) -> Option<H2SurfaceKind> {
    Some(match k {
        0 => H2SurfaceKind::HyperboloidBowl,
        1 => H2SurfaceKind::Hemisphere,
        2 => H2SurfaceKind::Pseudosphere,
        3 => H2SurfaceKind::Dini { twist },
        4 => H2SurfaceKind::Kuen,
        _ => return None,
    })
}

/// What the placement preserves, per surface.
#[cfg(test)]
fn placement_note(k: usize) -> &'static str {
    match k {
        0 => "Isometric embedding of H² as the hyperboloid sheet in Minkowski space R^{2,1} [PROVED]; its Euclidean curvature is not constant.",
        1 => "Conformal hemisphere model: angles are kept, lengths are not; the hemisphere's Euclidean curvature is +1, so this is not an isometry [PROVED].",
        2 | 3 => "The surface has Gaussian curvature −1 [PROVED] (Gray, Abbena & Salamon 2006), but the tiling is placed by a polar chart: a visual placement that distorts lengths and angles, not an isometry.",
        _ => "Kuen's surface has Gaussian curvature −1 [PROVED] (Gray, Abbena & Salamon 2006), but the tiling is placed by an affine chart: a visual placement that distorts lengths and angles, not an isometry.",
    }
}

/// Curvature row, naming the metric it refers to: the bowl's value is for its Minkowski-induced metric (the H²
/// metric), every other value for the Euclidean-induced metric of the surface in R³.
#[cfg(test)]
pub(crate) fn curvature_row(k: usize, value: f64) -> (String, String) {
    let (label, extra) = match k {
        0 => ("Gaussian curvature of the Minkowski-induced metric (the H² metric)", ""),
        1 => ("Gaussian curvature of the Euclidean-induced metric", "; the H² metric it carries (hemisphere model) has K = −1"),
        _ => ("Gaussian curvature of the Euclidean-induced metric", ""),
    };
    (label.to_string(), format!("{value} [PROVED]{extra}"))
}

/// Branch offsets k_i (u_i = θ_i + 2π k_i) that unwrap a face's polar angles along its boundary, shifted so the mean
/// unwrapped angle lies in (−π, π]. `None` when the boundary winds around the origin (no single-valued branch).
#[cfg(test)]
pub(crate) fn unwrap_branches(theta: &[f64]) -> Option<Vec<i64>> {
    let tau = 2.0 * PI;
    let wrap = |d: f64| (d + PI).rem_euclid(tau) - PI;
    let (&first, &last) = (theta.first()?, theta.last()?);
    let mut acc = first;
    let mut u = vec![acc];
    for w in theta.windows(2) {
        acc += wrap(w[1] - w[0]);
        u.push(acc);
    }
    if (acc + wrap(first - last) - first).abs() > PI {
        return None;
    }
    let mean = u.iter().sum::<f64>() / u.len() as f64;
    let shift = tau * ((PI - mean) / tau).floor();
    Some(u.iter().zip(theta).map(|(&ui, &ti)| ((ui + shift - ti) / tau).round() as i64).collect())
}

/// Dini placement with the polar angle unwrapped per face: the height term b·u is not 2π-periodic, so a face that
/// straddles θ = ±π gets its vertices on one branch (vertex copies at u + 2πk), and a face that winds around the
/// origin, which no branch can place, is left out. Returns the faces (indices into the extended `pts`) and that count.
#[cfg(test)]
fn dini_unwrapped(coords: &[[f64; 2]], faces: &[Vec<usize>], pts: &mut Vec<[f64; 3]>, twist: f64) -> (Vec<Vec<usize>>, usize) {
    let mut copies: HashMap<(usize, i64), usize> = HashMap::new();
    let (mut out, mut winding) = (Vec::with_capacity(faces.len()), 0);
    for f in faces {
        let theta: Vec<f64> = f.iter().map(|&v| coords[v][1].atan2(coords[v][0])).collect();
        let Some(ks) = unwrap_branches(&theta) else {
            winding += 1;
            continue;
        };
        let ring: Vec<usize> = f
            .iter()
            .zip(&ks)
            .zip(&theta)
            .map(|((&v, &k), &th)| {
                if k == 0 {
                    return v;
                }
                *copies.entry((v, k)).or_insert_with(|| {
                    let z = coords[v];
                    let rho = 2.0 * (z[0] * z[0] + z[1] * z[1]).sqrt().atanh();
                    pts.push(surface_point(H2SurfaceKind::Dini { twist }, th + 2.0 * PI * k as f64, rho));
                    pts.len() - 1
                })
            })
            .collect();
        out.push(ring);
    }
    (out, winding)
}

#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SurfJob {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) layers: usize,
    pub(crate) seed: SeedKind,
    pub(crate) surface: usize,
    pub(crate) twist: f64,
}

pub(crate) struct SurfState {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) layers: usize,
    pub(crate) seed: SeedKind,
    pub(crate) surface: usize,
    pub(crate) twist: f64,
}

impl Default for SurfState {
    fn default() -> Self {
        Self { p: 7, q: 3, layers: 2, seed: SeedKind::Vertex, surface: 0, twist: 0.2 }
    }
}

impl SurfState {
    #[cfg(test)]
    pub(crate) fn job(&self) -> SurfJob {
        let (_, hi) = layer_range(GeometryKind::Hyperbolic);
        SurfJob {
            p: self.p,
            q: self.q,
            layers: self.layers.min(hi),
            seed: self.seed,
            surface: self.surface,
            twist: self.twist,
        }
    }

    pub(super) fn save(&self) -> Value {
        json!({
            "p": self.p,
            "q": self.q,
            "layers": self.layers,
            "seed": format!("{:?}", self.seed),
            "surface": self.surface,
            "twist": self.twist,
        })
    }

    pub(super) fn restore(&mut self, v: &Value) {
        if let Some(x) = get_usize(v, "p", 3, MAX_PQ) {
            self.p = x;
        }
        if let Some(x) = get_usize(v, "q", 3, MAX_PQ) {
            self.q = x;
        }
        let (_, hi) = layer_range(GeometryKind::Hyperbolic);
        if let Some(x) = get_usize(v, "layers", 0, hi) {
            self.layers = x;
        }
        if let Some(s) = v.get("seed").and_then(Value::as_str).and_then(parse_seed_2d) {
            self.seed = s;
        }
        if let Some(x) = get_usize(v, "surface", 0, SURFACES.len() - 1) {
            self.surface = x;
        }
        if let Some(t) = v.get("twist").and_then(Value::as_f64).filter(|t| t.is_finite()) {
            self.twist = t.clamp(0.0, 2.0);
        }
    }
}

/// Write `text` to `path`; returns a status line.
#[cfg(test)]
pub(crate) fn export_to(path: &Path, text: &str) -> Result<String, String> {
    std::fs::write(path, text).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(format!("wrote {} ({} bytes)", path.display(), text.len()))
}

/// The net pieces as an SVG document (y flipped, 1 unit = 100 px, 10 px margin).
#[cfg(test)]
pub(crate) fn net_svg(pieces: &[Vec<[f64; 2]>]) -> String {
    let pts = pieces.iter().flatten().filter(|p| p[0].is_finite() && p[1].is_finite());
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in pts {
        (0..2).for_each(|k| (lo[k], hi[k]) = (lo[k].min(p[k]), hi[k].max(p[k])));
    }
    if lo[0] > hi[0] {
        (lo, hi) = ([0.0; 2], [1.0; 2]);
    }
    let s = 100.0;
    let (w, h) = ((hi[0] - lo[0]) * s + 20.0, (hi[1] - lo[1]) * s + 10.0);
    let mut out = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.1}\" height=\"{h:.1}\" viewBox=\"0 0 {w:.1} {h:.1}\">\n");
    for piece in pieces {
        let coords: Vec<String> =
            piece.iter().map(|p| format!("{:.3},{:.3}", (p[0] - lo[0]) * s + 10.0, (hi[1] - p[1]) * s + 10.0)).collect();
        out.push_str(&format!("<polygon points=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.5\"/>\n", coords.join(" ")));
    }
    out.push_str("</svg>\n");
    out
}

#[cfg(test)]
pub(crate) fn build(j: &SurfJob) -> Result<S5Out, String> {
    let kind = check_symbol(j.p, j.q)?;
    if kind != GeometryKind::Hyperbolic {
        return Err(format!(
            "{{{},{}}} is {}: only H² tilings are placed on these surfaces (pick (p−2)(q−2) > 4)",
            j.p,
            j.q,
            geom_label(kind)
        ));
    }
    let (lo, hi) = layer_range(kind);
    if !(lo..=hi).contains(&j.layers) {
        return Err(format!("{} layers refused: allowed {lo}..={hi}", j.layers));
    }
    if !j.twist.is_finite() {
        return Err(String::from("Dini twist must be finite"));
    }
    let surf = surface_kind(j.surface, j.twist).ok_or_else(|| format!("unknown surface #{}", j.surface))?;
    let h = build_hyperbolic_tiling_seeded(j.p, j.q, HyperbolicStyle::RegularPQ, j.layers, 0, j.seed)?;
    let t = build_unified_pq_seeded(j.p, j.q, j.layers, j.seed)?;
    let stats = patch_layer_stats(&h.faces, &h.face_shell, h.complete_through_shell);
    let stopped = (h.complete_through_shell < j.layers).then(|| {
        format!(
            "Stopped at layer {} (requested {}): {} [COMPUTATIONAL EVIDENCE]",
            h.complete_through_shell, j.layers, h.growth_rule
        )
    });

    let m = embed_tiling_on_surface(&t.coords_2d, &t.faces, surf)?;
    let mut pts = m.positions_3d.clone();
    let unwrap = matches!(surf, H2SurfaceKind::Dini { .. }) && j.twist.abs() > 0.0;
    let (faces, winding) = if unwrap {
        dini_unwrapped(&t.coords_2d, &m.faces, &mut pts, j.twist)
    } else {
        (m.faces.clone(), 0)
    };
    if faces.is_empty() {
        return Err(String::from("no face can be placed on this surface without a tear"));
    }
    let mesh = mesh_with_arcs(&pts, &faces, None);
    let net = unfold_papercraft_net(&pts, &faces)?;
    let title = format!("S5 net of {{{},{}}} on the {}", j.p, j.q, SURFACES[j.surface]);
    let extent = net.pieces.iter().flatten().filter(|p| p[0].is_finite() && p[1].is_finite()).fold(0.0_f64, |m, p| m.max(p[0].abs()).max(p[1].abs()));
    let scale = if extent > 10.0 { 10.0 / extent } else { 1.0 };
    let note = format!("papercraft net (BFS spanning tree of the dual graph), drawn at scale {scale:.3}");
    let mut pic = Picture::new(&title, &note);
    for piece in &net.pieces {
        pic.face(Some(piece.iter().map(|p| [p[0] * scale, p[1] * scale]).collect()), ring_size_tint(piece.len()), true);
    }
    let (scene2d, dropped) = pic.finish(0);
    let rows = vec![
        (String::from("Surface"), m.kind_name.clone()),
        curvature_row(j.surface, m.expected_gaussian_curvature),
        (String::from("Seed / Layers"), format!("{} / {} (complete 0..={})", seed_label_2d(j.seed), j.layers, h.complete_through_shell)),
        (String::from("Faces placed"), faces.len().to_string()),
        (String::from("Net pieces / glued edges / cut edges"), format!("{} / {} / {}", net.pieces.len(), net.tree_edges.len(), net.cut_pairings.len())),
        (String::from("Max edge-length error of the net"), format!("{:.2e}", net.max_edge_length_error)),
    ];
    let mut notes = vec![
        placement_note(j.surface).to_string(),
        String::from("Faces are flattened by a fan from their first vertex; the residual above measures how far the placed faces are from planar [COMPUTATIONAL EVIDENCE]."),
    ];
    if let Some(s) = &stopped {
        notes.push(s.clone());
    }
    notes.extend(non_disk_flags(&stats, None));
    if winding > 0 {
        notes.push(format!(
            "{winding} face(s) wind around the origin of the disk: on the Dini surface the height b·u grows by 2πb per turn of u (b = twist/√(1 + twist²) ≠ 0), so no placement closes them; they are left out [PROVED]"
        ));
    }
    if unwrap {
        notes.push(String::from(
            "Dini: faces crossing the polar cut θ = ±π are placed on one branch of u, so the mesh is cut along a zigzag of edges there instead of torn [COMPUTATIONAL EVIDENCE]",
        ));
    }
    if dropped > 0 {
        notes.push(format!("{dropped} net piece(s) outside the 2D clip window were not drawn [COMPUTATIONAL EVIDENCE]."));
    }
    let table = (!stats.is_empty()).then(|| (GROWTH_HEADERS.to_vec(), stats_to_rows(&stats)));
    let caption = format!("{{{},{}}} placed on the {}", j.p, j.q, SURFACES[j.surface]);
    Ok(S5Out { scene2d, mesh, caption, rows, notes, stopped, table, extra_table: None, svg: Some(net_svg(&net.pieces)) })
}
