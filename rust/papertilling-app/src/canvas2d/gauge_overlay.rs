//! 2D Ariza-Ortiz discrete gauge & metric elasticity overlays (GSV shear/energy,
//! bond stretch, vertex shear, plaquette flux `R_f`/`alpha_f`/Wyckoff, Volterra seams, and spectral coloring).

use crate::canvas2d::overlay_legend::{
    LegendChip, badge_category, colorbar_rgba, draw_categorical_legend, overlay_legend_rect, seam_category,
};
use crate::canvas2d::{Canvas2D, draw_dashed_egui};
use crate::mesh3d::sheet_colors::{diverging_rgba, rgba_u8, sequential_heat_rgba, strain_rgba};
use crate::scene2d::{CirclePrim, LinePrim, ScenePrimitives, TextPrim};
use crate::theme::ThemeTokens;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrystalOverlayMode {
    #[default]
    Standard,
    GsvEnergyShear,
    BondStretchStrain,
    VertexAngleShear,
    PlaquetteFlux,
    SeamConnections,
    SpectralLdos,
    SpectralCharge,
}

impl CrystalOverlayMode {
    pub const ALL: [Self; 8] = [
        Self::Standard,
        Self::GsvEnergyShear,
        Self::BondStretchStrain,
        Self::VertexAngleShear,
        Self::PlaquetteFlux,
        Self::SeamConnections,
        Self::SpectralLdos,
        Self::SpectralCharge,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Standard => "Standard sublattice",
            Self::GsvEnergyShear => "GSV elastic energy (AO)",
            Self::BondStretchStrain => "Signed bond stretch ε_e",
            Self::VertexAngleShear => "Vertex angle-shear energy",
            Self::PlaquetteFlux => "Plaquette flux (R_f, α_f, Wyckoff)",
            Self::SeamConnections => "Volterra seam g_e",
            Self::SpectralLdos => "LDOS |ψ|² (multiplet)",
            Self::SpectralCharge => "Spectral charge δρ(μ)",
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlaquetteBadge {
    pub center: [f64; 2],
    pub label: String,
    pub is_rotation: bool,
    pub improper: bool,
}

#[derive(Clone, Debug)]
pub struct SeamConnectionMark {
    pub a: [f64; 2],
    pub b: [f64; 2],
    pub label: String,
    pub det_neg: bool,
    /// The bond carries a seam group element g_e (false: a wall crossing without one).
    pub has_ge: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OverlayColorbar {
    pub title: String,
    pub min_val: f64,
    pub max_val: f64,
    pub diverging: bool,
    pub sqrt_scale: bool,
    /// Bar coloured with the bond-stretch map `strain_rgba` (saturating at its argument ±0.08) over
    /// `[min_val, max_val]` instead of `diverging_rgba` / `sequential_heat_rgba`.
    pub strain_map: bool,
}

/// Optional per-site, per-bond, plaquette-flux, seam, and colorbar overlays on 2D crystals + SVG/PNG.
#[derive(Clone, Debug, Default)]
pub struct CrystalOverlayData {
    pub site_rgba: Option<Vec<[u8; 4]>>,
    pub bond_rgba: Option<Vec<([u8; 4], f32)>>,
    pub badges: Vec<PlaquetteBadge>,
    pub seams: Vec<SeamConnectionMark>,
    pub legend: Option<OverlayColorbar>,
    /// Categorical legend (one chip per painted badge / seam category); drawn when `legend` is `None`.
    pub chips: Vec<LegendChip>,
    pub note_suffix: Option<String>,
}

/// Compute perceptual heatmap scale `max(P95, 0.20 * max, 1e-8)` so core singularities do not wash out 1/r bulk lobes.
pub fn perceptual_heat_scale(values: &[f64]) -> (f64, f64) {
    let mut pos: Vec<f64> = values.iter().copied().filter(|v| v.is_finite() && *v >= 0.0).collect();
    if pos.is_empty() {
        return (1e-6, 0.0);
    }
    pos.sort_by(f64::total_cmp);
    let vmax = *pos.last().unwrap_or(&0.0);
    let p95 = pos[((pos.len() - 1) * 95) / 100];
    let scale = p95.max(0.22 * vmax).max(1e-8);
    (scale, vmax)
}

pub fn color_heat_perceptual(values: &[f64]) -> (Vec<[u8; 4]>, f64) {
    let (scale, vmax) = perceptual_heat_scale(values);
    let cols = values
        .iter()
        .map(|&v| {
            let t = (v.max(0.0) / scale).clamp(0.0, 1.0).sqrt();
            rgba_u8(sequential_heat_rgba(t, 1.0, 1.0))
        })
        .collect();
    (cols, vmax.min(scale))
}

pub fn color_sites_scalar(values: &[f64], diverging: bool) -> Vec<[u8; 4]> {
    let vmax = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1e-6);
    values
        .iter()
        .map(|&v| {
            let c = if diverging {
                diverging_rgba(v, vmax, 1.0)
            } else {
                sequential_heat_rgba(v.max(0.0), vmax, 1.0)
            };
            rgba_u8(c)
        })
        .collect()
}

pub fn color_diverging_strain(values: &[f64], min_span: f64) -> (Vec<([u8; 4], f32)>, f64) {
    let vmax = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(min_span);
    let out = values
        .iter()
        .map(|&eps| {
            let frac = (eps.abs() / vmax).clamp(0.0, 1.0);
            let c = if min_span >= 0.01 {
                rgba_u8(diverging_rgba(eps, vmax, 1.0))
            } else {
                rgba_u8(strain_rgba(eps * (0.18 / vmax.max(1e-6)), 1.0))
            };
            (c, (1.4 + 1.8 * frac) as f32)
        })
        .collect();
    (out, vmax)
}

/// Apply per-site and per-bond colors using exact site/edge index maps (never heuristics on circle radii).
pub(crate) fn apply_site_and_bond_colors(prims: &mut ScenePrimitives, ov: &CrystalOverlayData) {
    if let Some(bcols) = &ov.bond_rgba {
        if prims.edge_line_indices.len() == bcols.len() {
            for (e_idx, &line_opt) in prims.edge_line_indices.iter().enumerate() {
                if let Some(li) = line_opt
                    && let Some(ln) = prims.lines.get_mut(li)
                {
                    let (c, w) = bcols[e_idx];
                    ln.color = c;
                    ln.width = w;
                }
            }
        } else {
            for (li, &(c, w)) in bcols.iter().enumerate().take(prims.lines.len()) {
                prims.lines[li].color = c;
                prims.lines[li].width = w;
            }
        }
    }
    if let Some(scols) = &ov.site_rgba
        && prims.site_circle_indices.len() == scols.len()
    {
        for (u, &circ_opt) in prims.site_circle_indices.iter().enumerate() {
            if let Some(ci) = circ_opt
                && let Some(circ) = prims.circles.get_mut(ci)
            {
                circ.fill = scols[u];
            }
        }
    }
    if let Some(suf) = &ov.note_suffix
        && !suf.is_empty()
    {
        if prims.note.is_empty() {
            prims.note = suf.clone();
        } else if !prims.note.contains(suf) {
            prims.note.push_str(" | ");
            prims.note.push_str(suf);
        }
    }
}

/// Append seams and plaquette flux badges once during SVG/PNG export (never called on live `self.compiled`).
pub(crate) fn append_gauge_export_prims(prims: &mut ScenePrimitives, ov: &CrystalOverlayData) {
    for s in &ov.seams {
        let col = seam_category(s).export_rgba();
        prims.lines.push(LinePrim {
            a: [s.a[0] as f32, s.a[1] as f32],
            b: [s.b[0] as f32, s.b[1] as f32],
            color: col,
            width: 3.1,
            dashed: true,
            arrow_mid: true,
        });
        if !s.label.is_empty() {
            prims.labels.push(TextPrim {
                pos: [0.5 * (s.a[0] + s.b[0]) as f32, 0.5 * (s.a[1] + s.b[1]) as f32],
                text: s.label.clone(),
                color: col,
                size_px: 9.0,
            });
        }
    }
    for b in &ov.badges {
        let col = badge_category(b).export_rgba();
        let p = [b.center[0] as f32, b.center[1] as f32];
        prims.circles.push(CirclePrim {
            center: p,
            radius_world: 0.24,
            min_px: 8.0,
            max_px: 15.0,
            fill: [col[0], col[1], col[2], 45],
            stroke: (col, 1.8),
        });
        prims.labels.push(TextPrim {
            pos: p,
            text: b.label.clone(),
            color: col,
            size_px: 9.5,
        });
    }
}

/// `box_r` unchanged if it misses `inset`, else lifted to end 8 px above it (`None` if it no longer fits).
pub(crate) fn legend_clear_of(box_r: egui::Rect, inset: egui::Rect, rect: egui::Rect) -> Option<egui::Rect> {
    if !box_r.intersects(inset) {
        return Some(box_r);
    }
    let lifted = box_r.translate(egui::vec2(0.0, inset.top() - 8.0 - box_r.bottom()));
    (lifted.top() >= rect.top() + 60.0).then_some(lifted)
}

pub(crate) fn draw_gauge_overlays(
    painter: &egui::Painter,
    canvas: &Canvas2D,
    rect: egui::Rect,
    ov: &CrystalOverlayData,
    tokens: &ThemeTokens,
) {
    let to_sc = |p: [f64; 2]| canvas.world_to_screen_at(rect, [p[0] as f32, p[1] as f32]);
    for s in &ov.seams {
        let col = seam_category(s).live_color(tokens);
        let (p0, p1) = (to_sc(s.a), to_sc(s.b));
        draw_dashed_egui(painter, p0, p1, egui::Stroke::new(2.8, col), 5.0, 3.0);
        if !s.label.is_empty() {
            let mid = p0.lerp(p1, 0.5);
            painter.text(mid, egui::Align2::CENTER_CENTER, &s.label, egui::FontId::proportional(9.5), col);
        }
    }
    for b in &ov.badges {
        let col = badge_category(b).live_color(tokens);
        let sp = to_sc(b.center);
        painter.circle_filled(sp, 11.5, egui::Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 38));
        painter.circle_stroke(sp, 11.5, egui::Stroke::new(1.8, col));
        painter.text(sp + egui::vec2(0.0, -14.5), egui::Align2::CENTER_CENTER, &b.label, egui::FontId::proportional(10.0), col);
    }
    let Some(mut box_r) = overlay_legend_rect(rect, ov) else { return };
    // the circuit reference inset is painted later at the bottom right: lift the legend above it if they meet
    if canvas.circuit_overlay.as_ref().is_some_and(|c| c.result.ref_path.len() >= 2)
        && let Some(inset) = super::circuit::reference_inset_rect(rect)
    {
        let Some(r) = legend_clear_of(box_r, inset, rect) else { return };
        box_r = r;
    }
    #[cfg(test)]
    painter.ctx().data_mut(|d| d.insert_temp(egui::Id::unique(LEGEND_RECT_TEST_ID), (rect, box_r)));
    match &ov.legend {
        Some(leg) => draw_colorbar_legend(painter, box_r, leg, tokens),
        None => draw_categorical_legend(painter, box_r, &ov.chips, tokens),
    }
}

/// egui temp-data key of `(canvas rect, legend rect)` of the last drawn 2D legend (tests only).
#[cfg(test)]
pub(crate) const LEGEND_RECT_TEST_ID: &str = "papertilling.canvas2d.legend_rect";

pub(crate) fn colorbar_legend_rect(rect: egui::Rect) -> Option<egui::Rect> {
    let stack_above_hud = rect.width() < 360.0;
    let min_h = if stack_above_hud { 140.0 } else { 110.0 };
    if rect.width() < 220.0 || rect.height() < min_h {
        return None;
    }
    let (y_top, y_bot) = if stack_above_hud { (-84.0, -40.0) } else { (-52.0, -8.0) };
    Some(egui::Rect::from_min_max(
        rect.left_bottom() + egui::vec2(8.0, y_top),
        rect.left_bottom() + egui::vec2(182.0, y_bot),
    ))
}

fn draw_colorbar_legend(
    painter: &egui::Painter,
    box_r: egui::Rect,
    leg: &OverlayColorbar,
    tokens: &ThemeTokens,
) {
    painter.rect_filled(box_r, 5.0, tokens.bg_hud);
    painter.rect_stroke(box_r, 5.0, egui::Stroke::new(1.0, tokens.border_strong), egui::StrokeKind::Inside);
    painter.text(
        box_r.left_top() + egui::vec2(7.0, 4.0),
        egui::Align2::LEFT_TOP,
        &leg.title,
        egui::FontId::proportional(9.5),
        tokens.text_primary,
    );
    let bar = egui::Rect::from_min_max(
        box_r.left_top() + egui::vec2(8.0, 19.0),
        box_r.right_top() + egui::vec2(-8.0, 27.0),
    );
    let steps = 24usize;
    let dx = bar.width() / (steps as f32);
    for i in 0..steps {
        let c = colorbar_rgba(leg, (i as f64 + 0.5) / (steps as f64));
        let seg = egui::Rect::from_min_max(
            egui::pos2(bar.left() + (i as f32) * dx, bar.top()),
            egui::pos2(bar.left() + ((i + 1) as f32) * dx + 0.5, bar.bottom()),
        );
        painter.rect_filled(seg, 0.0, egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], 255));
    }
    painter.rect_stroke(bar, 1.0, egui::Stroke::new(0.8, tokens.border_subtle), egui::StrokeKind::Inside);
    let fmt_v = |v: f64| if v.abs() >= 0.01 || v == 0.0 { format!("{v:.3}") } else { format!("{v:.1e}") };
    painter.text(box_r.left_bottom() + egui::vec2(7.0, -3.0), egui::Align2::LEFT_BOTTOM, fmt_v(leg.min_val), egui::FontId::proportional(9.0), tokens.text_secondary);
    painter.text(box_r.right_bottom() + egui::vec2(-7.0, -3.0), egui::Align2::RIGHT_BOTTOM, fmt_v(leg.max_val), egui::FontId::proportional(9.0), tokens.text_secondary);
}

#[cfg(test)]
mod tests {
    use super::*;
    use papertilling_core::defect::DefectCatalogue;
    use papertilling_core::defect::sheet3d::{Sheet3DOptions, relax_sheet_3d};
    use papertilling_core::defect::surgery::{DefectSpec, build_point_defects_with_cuts};
    use papertilling_core::plane::catalogue::parse_named_str;
    use papertilling_core::types::Frame2D;

    #[test]
    fn site_bond_recoloring_and_single_pass_overlays() {
        let named = parse_named_str(include_str!("../../../../papertilling/data/named.json")).unwrap();
        let sq = named.named.into_iter().find(|n| n.key == "square").unwrap();
        let mut crystal = papertilling_core::defect::surgery::build_point_defects(
            &sq.cell,
            &[DefectSpec::disclination(90.0, [0.5, 0.5])],
            None,
            None,
        ).unwrap();
        // Inject a ghost stub on edge 0 so !ln.dashed count != crystal.edges.len().
        let (u0, v0) = crystal.edges[0];
        if let Some(d) = crystal.adj[u0].iter_mut().find(|d| d.to == v0) {
            d.ghost_stub = Some(crystal.pos[u0]);
        }
        let mut cv = Canvas2D::new(crate::scene2d::Scene2D::CrystalPatch {
            crystal: crystal.clone(),
            title: String::from("test"),
            note: String::new(),
        });
        let ov = crate::live_ops_ao::build_ao_crystal_overlay(
            &crystal,
            CrystalOverlayMode::GsvEnergyShear,
            &sq.group,
            &sq.cell,
            sq.frame,
        ).unwrap();
        let expected_sc = ov.site_rgba.clone().unwrap();
        let expected_bc = ov.bond_rgba.clone().unwrap();
        cv.set_gauge_overlay(Some(ov));
        assert_eq!(cv.compiled.site_circle_indices.len(), crystal.pos.len());
        assert!(cv.compiled.edge_line_indices[0].is_none(), "ghost edge 0 must not shift later bond colors");
        for (u, &ci_opt) in cv.compiled.site_circle_indices.iter().enumerate() {
            let ci = ci_opt.unwrap();
            assert_eq!(cv.compiled.circles[ci].fill, expected_sc[u]);
        }
        for (ei, &li_opt) in cv.compiled.edge_line_indices.iter().enumerate() {
            if let Some(li) = li_opt {
                assert_eq!(cv.compiled.lines[li].color, expected_bc[ei].0);
            }
        }
        let base_lines = cv.compiled.lines.len();
        let base_labels = cv.compiled.labels.len();
        let ov_flux = crate::live_ops_ao::build_ao_crystal_overlay(
            &crystal,
            CrystalOverlayMode::PlaquetteFlux,
            &sq.group,
            &sq.cell,
            sq.frame,
        ).unwrap();
        assert!(ov_flux.badges.iter().any(|b| b.label.contains("+90") && (b.label.contains("1a") || b.label.contains("1b"))));
        cv.set_gauge_overlay(Some(ov_flux));
        assert_eq!(cv.compiled.lines.len(), base_lines, "seams must not be duplicated into live compiled prims");
        assert_eq!(cv.compiled.labels.len(), base_labels, "badges must not be duplicated into live compiled prims");
    }

    #[test]
    fn half_wall_ray_mobius_scale_and_hinge_bend_breakdown() {
        let named = parse_named_str(include_str!("../../../../papertilling/data/named.json")).unwrap();
        let pg = named.named.iter().find(|n| n.key == "brick_pg").unwrap();
        let (hw_crys, cuts, _) = build_point_defects_with_cuts(
            &pg.cell,
            &[DefectSpec::half_wall(Some("Wd1"), [0.0, 0.0], None, None)],
            Some(3.5),
            None,
        ).unwrap();
        assert!(!hw_crys.pos.is_empty());
        assert_eq!(cuts.len(), 1);
        assert!(cuts[0].is_half_wall);

        let pm = named.named.iter().find(|n| n.key == "square").unwrap();
        let cat = DefectCatalogue::new(&pm.group, &pm.cell, Frame2D::default());
        let w_def = cat.build_wall("Wc1", None).unwrap();
        let w_big = cat.build_wall_scaled("Wc1", 7.2).unwrap();
        assert!(w_big.pos.len() > w_def.pos.len());

        let cone = build_point_defects_with_cuts(
            &pm.cell,
            &[DefectSpec::disclination(90.0, [0.5, 0.5])],
            None,
            None,
        ).unwrap().0;
        let sheet = relax_sheet_3d(&cone, &pm.cell, &Sheet3DOptions { kappa_b: 0.05, max_iters: 40, grad_tol: 1e-5 }, None);
        let sum_site_bend: f64 = sheet.site_bend_energy.iter().sum();
        assert!((sum_site_bend - sheet.bend_energy).abs() < 1e-9, "site bend sum {sum_site_bend} vs {}", sheet.bend_energy);
    }

    #[test]
    fn colorbar_and_zoom_hud_never_overlap_across_widths() {
        use super::super::input::zoom_hud_rect;
        for (w, h, expect_legend) in [
            (300.0_f32, 300.0_f32, true),
            (220.0, 200.0, true),
            (346.0, 200.0, true),
            (360.0, 200.0, true),
            (600.0, 400.0, true),
            (250.0, 120.0, false),
            (180.0, 300.0, false),
        ] {
            let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(w, h));
            let leg = colorbar_legend_rect(rect);
            assert_eq!(leg.is_some(), expect_legend, "size ({w}, {h}) legend={leg:?}");
            if let Some(leg_r) = leg {
                assert!(rect.contains_rect(leg_r), "legend {leg_r:?} must stay inside {rect:?}");
                if let Some(hud_r) = zoom_hud_rect(rect) {
                    assert!(
                        !leg_r.intersects(hud_r),
                        "legend {leg_r:?} overlapped zoom HUD {hud_r:?} at ({w}, {h})"
                    );
                }
            }
        }
    }
}

