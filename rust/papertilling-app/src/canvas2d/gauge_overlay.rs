//! 2D Ariza-Ortiz discrete gauge & metric elasticity overlays (GSV shear/energy,
//! plaquette gauge flux `R_f` / `alpha_f`, Volterra seam connection `g_e`, and spectral coloring).

use crate::canvas2d::{Canvas2D, draw_dashed_egui};
use crate::mesh3d::sheet_colors::{diverging_rgba, rgba_u8, sequential_heat_rgba};
use crate::scene2d::{CirclePrim, LinePrim, ScenePrimitives, TextPrim};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrystalOverlayMode {
    #[default]
    Standard,
    GsvEnergyShear,
    PlaquetteFlux,
    SeamConnections,
    SpectralLdos,
    SpectralCharge,
}

impl CrystalOverlayMode {
    pub const ALL: [Self; 6] = [
        Self::Standard,
        Self::GsvEnergyShear,
        Self::PlaquetteFlux,
        Self::SeamConnections,
        Self::SpectralLdos,
        Self::SpectralCharge,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Standard => "Standard sublattice",
            Self::GsvEnergyShear => "GSV energy & shear",
            Self::PlaquetteFlux => "Plaquette flux (R_f, α_f)",
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
}

/// Optional per-site, per-bond, plaquette-flux, and seam overlays painted on 2D crystal scenes + SVG/PNG.
#[derive(Clone, Debug, Default)]
pub struct CrystalOverlayData {
    pub site_rgba: Option<Vec<[u8; 4]>>,
    pub bond_rgba: Option<Vec<([u8; 4], f32)>>,
    pub badges: Vec<PlaquetteBadge>,
    pub seams: Vec<SeamConnectionMark>,
}

/// Apply scalar per-site colormap (for LDOS or excess spectral charge `delta_rho`) cleanly.
pub fn color_sites_scalar(values: &[f64], diverging: bool) -> Vec<[u8; 4]> {
    let vmax = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1e-6);
    values
        .iter()
        .map(|&v| {
            let c = if diverging { diverging_rgba(v, vmax, 1.0) } else { sequential_heat_rgba(v.max(0.0), vmax, 1.0) };
            rgba_u8(c)
        })
        .collect()
}

pub(crate) fn apply_overlay_to_prims(prims: &mut ScenePrimitives, ov: &CrystalOverlayData) {
    if let Some(bcols) = &ov.bond_rgba {
        let mut bi = 0usize;
        for ln in &mut prims.lines {
            if !ln.dashed && bi < bcols.len() {
                let (c, w) = bcols[bi];
                ln.color = c;
                ln.width = w;
                bi += 1;
            }
        }
    }
    if let Some(scols) = &ov.site_rgba {
        let mut si = 0usize;
        for circ in &mut prims.circles {
            if circ.radius_world >= 0.08 && si < scols.len() {
                circ.fill = scols[si];
                si += 1;
            }
        }
    }
    for s in &ov.seams {
        let col = if s.det_neg { [186, 45, 118, 240] } else { [0, 131, 143, 235] };
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
        let col = if b.improper {
            [142, 36, 170, 235]
        } else if b.is_rotation {
            [198, 40, 40, 235]
        } else {
            [230, 81, 0, 235]
        };
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

pub(crate) fn draw_gauge_overlays(
    painter: &egui::Painter,
    canvas: &Canvas2D,
    rect: egui::Rect,
    ov: &CrystalOverlayData,
) {
    let to_sc = |p: [f64; 2]| canvas.world_to_screen_at(rect, [p[0] as f32, p[1] as f32]);
    for s in &ov.seams {
        let col = if s.det_neg {
            egui::Color32::from_rgb(186, 45, 118)
        } else {
            egui::Color32::from_rgb(0, 131, 143)
        };
        let (p0, p1) = (to_sc(s.a), to_sc(s.b));
        draw_dashed_egui(painter, p0, p1, egui::Stroke::new(2.8, col), 5.0, 3.0);
        if !s.label.is_empty() {
            let mid = p0.lerp(p1, 0.5);
            painter.text(mid, egui::Align2::CENTER_CENTER, &s.label, egui::FontId::proportional(9.5), col);
        }
    }
    for b in &ov.badges {
        let col = if b.improper {
            egui::Color32::from_rgb(142, 36, 170)
        } else if b.is_rotation {
            egui::Color32::from_rgb(198, 40, 40)
        } else {
            egui::Color32::from_rgb(230, 81, 0)
        };
        let sp = to_sc(b.center);
        painter.circle_filled(sp, 11.5, egui::Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 38));
        painter.circle_stroke(sp, 11.5, egui::Stroke::new(1.8, col));
        painter.text(sp + egui::vec2(0.0, -14.5), egui::Align2::CENTER_CENTER, &b.label, egui::FontId::proportional(10.0), col);
    }
}
