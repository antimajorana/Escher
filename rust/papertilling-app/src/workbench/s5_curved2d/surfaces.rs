//! An H² {p,q} patch placed on a surface in R³ (`embed_tiling_on_surface`) and the papercraft net of the placed
//! mesh (`unfold_papercraft_net`), with an SVG export of the net. Only the hyperboloid bowl is an isometric
//! embedding (for the Minkowski metric); the inspector says so for every other surface.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;

use papertilling_core::dsymbol::GeometryKind;
use papertilling_core::noneuclidean2d::{
    H2SurfaceKind, build_unified_pq, embed_tiling_on_surface, surface_point, unfold_papercraft_net,
};
use serde_json::{Value, json};

use super::draw::{Picture, mesh_with_arcs};
use super::pq::layer_range;
use super::{MAX_PQ, S5Out, check_symbol, delta_badge, geom_label, get_usize, pq_controls};
use crate::workbench::WbCtx;
use crate::workbench::scene_helpers::ring_size_tint;

pub(crate) const SURFACES: [&str; 5] = ["Hyperboloid bowl", "Hemisphere", "Pseudosphere", "Dini surface", "Kuen surface"];

/// The surface for index `k` (Dini with the given twist b/a).
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

/// What the placement preserves, per surface (shown in the inspector).
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SurfJob {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) layers: usize,
    pub(crate) surface: usize,
    pub(crate) twist: f64,
}

pub(crate) struct SurfState {
    pub(crate) p: usize,
    pub(crate) q: usize,
    pub(crate) layers: usize,
    pub(crate) surface: usize,
    pub(crate) twist: f64,
    pub(crate) status: String,
}

impl Default for SurfState {
    fn default() -> Self {
        Self { p: 7, q: 3, layers: 2, surface: 0, twist: 0.2, status: String::new() }
    }
}

impl SurfState {
    pub(crate) fn job(&self) -> SurfJob {
        SurfJob { p: self.p, q: self.q, layers: self.layers, surface: self.surface, twist: self.twist }
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = pq_controls(ui, &mut self.p, &mut self.q, 3);
        ui.label(delta_badge(self.p, self.q));
        ui.horizontal(|ui| {
            ui.label("Layers:");
            rebuild |= ui.add(egui::DragValue::new(&mut self.layers).range(1..=4)).changed();
        });
        ui.horizontal(|ui| {
            ui.label("Surface:");
            egui::ComboBox::from_id_salt("wb_s5_surface").selected_text(SURFACES[self.surface.min(4)]).show_ui(ui, |ui| {
                for (i, s) in SURFACES.iter().enumerate() {
                    rebuild |= ui.selectable_value(&mut self.surface, i, *s).changed();
                }
            });
        });
        if self.surface == 3 {
            ui.horizontal(|ui| {
                ui.label("Dini twist b/a:");
                rebuild |= ui.add(egui::DragValue::new(&mut self.twist).range(0.0..=2.0).speed(0.01)).changed();
            });
        }
        ui.small("2D: papercraft net of the placed mesh (BFS spanning tree; pieces may overlap).");
        rebuild
    }

    pub(super) fn show_export(&mut self, ui: &mut egui::Ui, cx: &WbCtx, svg: Option<&str>) {
        ui.separator();
        if ui.button("Save net as .svg").clicked() {
            self.status = match svg {
                None => String::from("refused: build a net first"),
                Some(_) if cx.smoke || cfg!(test) => String::from("papercraft_net.svg ready; file dialog skipped"),
                Some(text) => match rfd::FileDialog::new().set_file_name("papercraft_net.svg").save_file() {
                    Some(path) => export_to(&path, text).unwrap_or_else(|e| e),
                    None => String::from("export cancelled"),
                },
            };
        }
        if !self.status.is_empty() {
            ui.small(self.status.as_str());
        }
    }

    pub(super) fn save(&self) -> Value {
        json!({ "p": self.p, "q": self.q, "layers": self.layers, "surface": self.surface, "twist": self.twist })
    }

    pub(super) fn restore(&mut self, v: &Value) {
        if let Some(x) = get_usize(v, "p", 3, MAX_PQ) {
            self.p = x;
        }
        if let Some(x) = get_usize(v, "q", 3, MAX_PQ) {
            self.q = x;
        }
        if let Some(x) = get_usize(v, "layers", 1, 4) {
            self.layers = x;
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
pub(crate) fn export_to(path: &Path, text: &str) -> Result<String, String> {
    std::fs::write(path, text).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(format!("wrote {} ({} bytes)", path.display(), text.len()))
}

/// The net pieces as an SVG document (y flipped, 1 unit = 100 px, 10 px margin).
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
    let (w, h) = ((hi[0] - lo[0]) * s + 20.0, (hi[1] - lo[1]) * s + 20.0);
    let mut out = format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.1}\" height=\"{h:.1}\" viewBox=\"0 0 {w:.1} {h:.1}\">\n");
    for piece in pieces {
        let coords: Vec<String> =
            piece.iter().map(|p| format!("{:.3},{:.3}", (p[0] - lo[0]) * s + 10.0, (hi[1] - p[1]) * s + 10.0)).collect();
        out.push_str(&format!("<polygon points=\"{}\" fill=\"none\" stroke=\"black\" stroke-width=\"0.5\"/>\n", coords.join(" ")));
    }
    out.push_str("</svg>\n");
    out
}

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
    let t = build_unified_pq(j.p, j.q, j.layers)?;
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
    // the canvas clips at CLIP_R: draw the net uniformly scaled to fit (the SVG keeps true units)
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
        (String::from("Faces placed"), faces.len().to_string()),
        (String::from("Net pieces / glued edges / cut edges"), format!("{} / {} / {}", net.pieces.len(), net.tree_edges.len(), net.cut_pairings.len())),
        (String::from("Max edge-length error of the net"), format!("{:.2e}", net.max_edge_length_error)),
    ];
    let mut notes = vec![
        placement_note(j.surface).to_string(),
        String::from("Faces are flattened by a fan from their first vertex; the residual above measures how far the placed faces are from planar [COMPUTATIONAL EVIDENCE]."),
    ];
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
        notes.push(format!("{dropped} net piece(s) outside the 2D clip window were not drawn."));
    }
    let caption = format!("{{{},{}}} placed on the {}", j.p, j.q, SURFACES[j.surface]);
    Ok(S5Out { scene2d, mesh, caption, rows, notes, table: None, svg: Some(net_svg(&net.pieces)) })
}
