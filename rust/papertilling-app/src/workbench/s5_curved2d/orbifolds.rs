//! Conway orbifold symbols (`parse_orbifold`: χ_orb, geometry, wallpaper group), the 17 wallpaper groups, vertex
//! configurations (`parse_vertex_config`), and compact triangle-group quotients (`tgcell_quotient`): the fundamental
//! polygon and its faces in the Poincaré disk, genus, χ and the side pairings.
//!
//! The two parsers are exact and tiny, so they run on the UI thread (re-parsed only when the text changes); the TG
//! cell is built on the panel worker.

use papertilling_core::noneuclidean2d::{
    ConwayOrbifold, H2SurfaceKind, TGCELL_NAMES, VertexConfigInfo, geodesic_arc_poincare, map_poincare_to_surface,
    parse_orbifold, parse_vertex_config, tgcell_quotient, verify_tgcell_invariants, wallpaper_orbifold_table,
};
use serde_json::{Value, json};

use super::draw::{DiskModel, Picture};
use super::{S5Out, geom_label, get_usize};
use crate::mesh3d::wb_mesh::{mesh_from_polygons, wb_face_tint};
use crate::workbench::scene_helpers::{EDGE_RGBA, ring_size_tint};

const BOUNDARY_RGBA: [u8; 4] = [192, 57, 43, 255];
const MAX_SYMBOL_LEN: usize = 64;
/// Cap on the lcm L of the orders' denominators. The core sums the orders' costs in exact `i128` rationals reduced
/// by their gcd, so every intermediate is at most about 200 L² (|χ| <= 200 for 64 characters): L <= 10^15 keeps it
/// below 2.1e32, far from `i128::MAX` ≈ 1.7e38. Larger orders are refused here instead of overflowing.
pub(crate) const MAX_ORDER_LCM: u128 = 1_000_000_000_000_000;

pub(crate) struct OrbState {
    pub(crate) orb_text: String,
    pub(crate) vc_text: String,
    pub(crate) tg: usize,
    orb: Option<(String, Result<ConwayOrbifold, String>)>,
    vc: Option<(String, Result<VertexConfigInfo, String>)>,
}

impl Default for OrbState {
    /// p6m = *632, the 3.4.6.4 tiling, and the Klein quartic {7,3}.
    fn default() -> Self {
        Self { orb_text: String::from("*632"), vc_text: String::from("3.4.6.4"), tg: 2, orb: None, vc: None }
    }
}

fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Refuse when lcm(factor · n) over the orders exceeds [`MAX_ORDER_LCM`] (checked arithmetic, no overflow).
pub(crate) fn check_order_lcm(orders: &[u128], factor: u128) -> Result<(), String> {
    let mut l: u128 = 1;
    for &n in orders {
        let d = n.max(1).checked_mul(factor);
        let next = d.and_then(|d| (l / gcd_u128(l, d)).checked_mul(d));
        match next {
            Some(x) if x <= MAX_ORDER_LCM => l = x,
            _ => {
                return Err(format!(
                    "orders too large: the lcm of their denominators exceeds 10^15, beyond the exact i128 rational arithmetic (order {n})"
                ));
            }
        }
    }
    Ok(())
}

/// The orders the core's `parse_orbifold` reads from `s` (cones and corners alike), by its chunk rule: inside
/// parentheses, or in a run containing '.', ',' or whitespace, whole numbers; otherwise one order per digit. Tokens
/// that are not numbers are skipped (the core refuses them with its own reason).
pub(crate) fn orbifold_orders(s: &str) -> Vec<u128> {
    let is_sep = |c: char| c == '.' || c == ',' || c.is_whitespace();
    let chars: Vec<char> = s.trim().chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '(' {
            let end = chars[i + 1..].iter().position(|&x| x == ')').map_or(chars.len(), |k| i + 1 + k);
            let inner: String = chars[i + 1..end].iter().collect();
            out.extend(inner.split(is_sep).filter_map(|t| t.trim().parse::<u128>().ok()));
            i = end + 1;
        } else if c.is_ascii_digit() || is_sep(c) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || is_sep(chars[i])) {
                i += 1;
            }
            let run: String = chars[start..i].iter().collect();
            let run = run.trim();
            if run.chars().any(is_sep) {
                out.extend(run.split(is_sep).filter_map(|t| t.parse::<u128>().ok()));
            } else {
                out.extend(run.chars().filter_map(|d| d.to_digit(10)).map(u128::from));
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Parse with a length guard (the parsers are linear, but a pasted page of text is refused with a reason) and an
/// order-size guard (corner n costs (n−1)/(2n), so the denominators are 2n at most).
pub(crate) fn orbifold_card(s: &str) -> Result<ConwayOrbifold, String> {
    if s.chars().count() > MAX_SYMBOL_LEN {
        return Err(format!("symbol longer than {MAX_SYMBOL_LEN} characters refused"));
    }
    check_order_lcm(&orbifold_orders(s), 2)?;
    parse_orbifold(s)
}

/// Same guards for a vertex configuration (the angle defect has denominator dividing lcm(n_i)).
pub(crate) fn vertex_card(s: &str) -> Result<VertexConfigInfo, String> {
    if s.chars().count() > MAX_SYMBOL_LEN {
        return Err(format!("configuration longer than {MAX_SYMBOL_LEN} characters refused"));
    }
    let cleaned = s.trim().trim_matches(['(', ')', '[', ']']);
    let orders: Vec<u128> = cleaned.split('.').filter_map(|t| t.trim().parse::<u128>().ok()).collect();
    check_order_lcm(&orders, 1)?;
    parse_vertex_config(s)
}

fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

/// The two chirality fields of the core, each under its own label: `is_chiral` only says the cyclic sequence
/// differs from its reverse; `is_snub_chiral` is the core's chirality of the tiling (that, or a snub 3.3.p.3.q).
pub(crate) fn chirality_text(v: &VertexConfigInfo) -> String {
    format!(
        "vertex sequence equals its reverse (mirror-symmetric vertex figure): {}; chiral tiling (core rule: sequence differs from its reverse, or snub 3.3.p.3.q with p ≠ q): {}",
        yes_no(!v.is_chiral),
        yes_no(v.is_snub_chiral)
    )
}

fn refusal(ui: &mut egui::Ui, e: &str) {
    ui.colored_label(egui::Color32::from_rgb(192, 57, 43), format!("Refused: {e}"));
}

impl OrbState {
    pub(crate) fn tg_name(&self) -> &'static str {
        TGCELL_NAMES[self.tg.min(TGCELL_NAMES.len() - 1)]
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        ui.horizontal(|ui| {
            ui.label("Orbifold symbol:");
            ui.add(egui::TextEdit::singleline(&mut self.orb_text).desired_width(120.0).id_salt("wb_s5_orb"));
        });
        ui.horizontal(|ui| {
            ui.label("Vertex configuration:");
            ui.add(egui::TextEdit::singleline(&mut self.vc_text).desired_width(120.0).id_salt("wb_s5_vc"));
        });
        let mut rebuild = false;
        ui.horizontal(|ui| {
            ui.label("TG cell:");
            egui::ComboBox::from_id_salt("wb_s5_tg").selected_text(self.tg_name()).show_ui(ui, |ui| {
                for (i, n) in TGCELL_NAMES.iter().enumerate() {
                    rebuild |= ui.selectable_value(&mut self.tg, i, *n).changed();
                }
            });
        });
        ui.collapsing("17 wallpaper groups (orbifold notation)", |ui| {
            egui::Grid::new("wb_s5_wallpaper").striped(true).show(ui, |ui| {
                for (k, (g, o)) in wallpaper_orbifold_table().iter().enumerate() {
                    ui.label(*g);
                    ui.monospace(*o);
                    if k % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
        });
        rebuild
    }

    /// The parse cards (re-parsed only when a text changed).
    pub(super) fn show_cards(&mut self, ui: &mut egui::Ui) {
        if self.orb.as_ref().is_none_or(|(t, _)| *t != self.orb_text) {
            self.orb = Some((self.orb_text.clone(), orbifold_card(&self.orb_text)));
        }
        if self.vc.as_ref().is_none_or(|(t, _)| *t != self.vc_text) {
            self.vc = Some((self.vc_text.clone(), vertex_card(&self.vc_text)));
        }
        ui.strong("Orbifold card");
        match self.orb.as_ref().map(|(_, r)| r) {
            Some(Ok(o)) => {
                ui.label(format!("{} (canonical {})", o.input.trim(), o.canonical));
                ui.label(format!("χ_orb = {}/{} ≈ {:.6}: {}", o.chi_num, o.chi_den, o.chi_orb, geom_label(o.geometry_kind)));
                ui.label(format!(
                    "handles {}, crosscaps {}, mirror boundaries {}, cones {:?}, corners {:?}",
                    o.handles, o.crosscaps, o.boundary_components, o.cones, o.corners
                ));
                if let Some(w) = &o.wallpaper_group {
                    ui.label(format!("Wallpaper group {w}"));
                }
                ui.small(format!("Magic theorem χ = 2 − cost (Conway et al. 2008) {}", o.provenance));
            }
            Some(Err(e)) => refusal(ui, e),
            None => {}
        }
        ui.separator();
        ui.strong("Vertex configuration card");
        match self.vc.as_ref().map(|(_, r)| r) {
            Some(Ok(v)) => {
                ui.label(format!("{} (canonical {})", v.input.trim(), v.canonical));
                ui.label(chirality_text(v));
                ui.label(format!(
                    "angle defect {}/{} π: {}",
                    v.angle_defect_pi_num,
                    v.angle_defect_pi_den,
                    geom_label(v.geometry_kind)
                ));
                ui.small(v.provenance.as_str());
            }
            Some(Err(e)) => refusal(ui, e),
            None => {}
        }
    }

    pub(super) fn save(&self) -> Value {
        json!({ "orbifold": self.orb_text, "vertex_config": self.vc_text, "tg": self.tg })
    }

    pub(super) fn restore(&mut self, v: &Value) {
        if let Some(s) = v.get("orbifold").and_then(Value::as_str) {
            self.orb_text = s.chars().take(MAX_SYMBOL_LEN).collect();
        }
        if let Some(s) = v.get("vertex_config").and_then(Value::as_str) {
            self.vc_text = s.chars().take(MAX_SYMBOL_LEN).collect();
        }
        if let Some(x) = get_usize(v, "tg", 0, TGCELL_NAMES.len() - 1) {
            self.tg = x;
        }
    }
}

/// Geodesic ring through the given Poincaré points (closed, `steps` samples per side).
fn ring_arcs(ring: &[[f64; 2]], steps: usize) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    for k in 0..ring.len() {
        if let Ok(arc) = geodesic_arc_poincare(ring[k], ring[(k + 1) % ring.len()], steps) {
            out.extend(arc);
        }
    }
    out
}

pub(crate) fn build_tg(name: &str) -> Result<S5Out, String> {
    let qc = tgcell_quotient(name)?;
    let verified = verify_tgcell_invariants(&qc);
    let title = format!("S5 TG cell {}", qc.name);
    let note = format!("Poincaré disk: the {} faces of the fundamental polygon (red boundary)", qc.placed_faces_poincare.len());
    let mut pic = Picture::new(&title, &note);
    let model = DiskModel::Poincare;
    for f in &qc.placed_faces_poincare {
        pic.face(f.iter().map(|z| model.map(*z)).collect(), ring_size_tint(f.len()), false);
        pic.polyline(&ring_arcs(f, 8), EDGE_RGBA, 1.0);
    }
    pic.polyline(&ring_arcs(&qc.fundamental_polygon_poincare, 12), BOUNDARY_RGBA, 2.0);
    pic.add_unit_circle();
    let (scene2d, dropped) = pic.finish(0);

    let mut pts3 = Vec::new();
    let mut faces3 = Vec::new();
    for f in &qc.placed_faces_poincare {
        let base = pts3.len();
        pts3.extend(f.iter().map(|&z| map_poincare_to_surface(z, H2SurfaceKind::Hemisphere).unwrap_or([f64::NAN; 3])));
        faces3.push((base..base + f.len()).collect::<Vec<_>>());
    }
    let mesh = mesh_from_polygons(&pts3, &faces3, &super::draw::face_edges(&faces3), wb_face_tint);

    let rows = vec![
        (String::from("Surface"), qc.name.clone()),
        (String::from("Group G⁺, |G⁺|"), format!("{}, {}", qc.group_label, qc.group_order)),
        (String::from("Map type {p,q}"), format!("{{{},{}}}", qc.p, qc.q)),
        (String::from("Genus / χ"), format!("{} / {}", qc.genus, qc.chi)),
        (String::from("V / E / F"), format!("{} / {} / {}", qc.n_vertices, qc.n_edges, qc.n_faces)),
        (String::from("Side pairings"), qc.side_pairings.len().to_string()),
    ];
    let mut notes = vec![
        String::from("Counts E = |G|/2, V = |G|/q, F = |G|/p are exact integers (Coxeter & Moser, ch. 8) [PROVED]"),
        String::from("Fundamental polygon placement in the disk is floating point [COMPUTATIONAL EVIDENCE]; 3D: hemisphere model (conformal)."),
    ];
    notes.push(match verified {
        Ok(()) => String::from("Invariant check χ = 2 − 2g = V − E + F, uniform p-gons, vertex degree q: passed (exact integer check of the built map) [PROVED]"),
        Err(e) => format!("Invariant check FAILED: {e}"),
    });
    if dropped > 0 {
        notes.push(format!("{dropped} face(s) could not be drawn."));
    }
    let table = qc
        .side_pairings
        .iter()
        .map(|s| vec![s.side_a.to_string(), s.side_b.to_string(), s.edge_index.to_string()])
        .collect();
    let caption = format!("TG cell {} (genus {}) on the hemisphere model", qc.name, qc.genus);
    Ok(S5Out {
        scene2d,
        mesh,
        caption,
        rows,
        notes,
        stopped: None,
        table: Some((vec!["Side a", "Side b", "Edge"], table)),
        extra_table: None,
        svg: None,
    })
}
