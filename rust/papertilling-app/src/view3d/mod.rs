use std::sync::Arc;

use eframe::egui_wgpu;

use crate::canvas2d::ZOOM_STEP_FACTOR;
use crate::mesh3d::GpuMeshData;
#[cfg(test)]
use crate::mesh3d::orbit_rotation;
use crate::theme::ThemeTokens;

mod camera;
#[cfg(test)]
mod camera_kittest;
mod gizmo;
#[cfg(test)]
mod gizmo_tests;
mod gpu;
#[cfg(test)]
mod interaction_tests;
mod legend;
mod export_png;
#[cfg(test)]
mod export_png_tests;
mod offscreen;
mod wire_quads;
#[cfg(test)]
mod orbit_tests;
#[cfg(test)]
mod legend_tests;

pub use camera::DIST_DEFAULT;
#[cfg(test)]
use camera::{DIST_MAX, DIST_MIN};
use gizmo::{button_label, draw_gizmo, gizmo_hit, gizmo_rect_3d, preset_button_rects, preset_for_axis, preset_tooltip};
pub use gpu::{DEPTH_BITS, GpuFrameCounters, init_view3d_resources};
use gpu::View3dCallback;
pub use export_png::Export3dInfo;
pub use legend::{ColorLegend3d, Legend3dMap};
pub use offscreen::OffscreenGpuContext;
use legend::{draw_legend_3d, legend_rect_3d, zoom_hud_rect_3d};

pub struct View3dState {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    /// Turntable axis of the orbit: world z (Front / Side presets) instead of world y (see `camera`).
    pub(crate) z_up: bool,
    /// Whether the current drag pans (`Some(true)`) or orbits, fixed when it starts; `None` between drags.
    drag_pans: Option<bool>,
    /// Translation in camera x/y coordinates; rendering only, never part of the mesh or its identity.
    pub(crate) camera_offset: egui::Vec2,
    pub counters: Arc<GpuFrameCounters>,
    mesh_generation: u64,
    mesh_data: Arc<GpuMeshData>,
    overlay_caption: String,
    legend: Option<ColorLegend3d>,
    /// Size (points) of the viewport the last `show` drew, its physical pixels per point, and the egui pass it ran
    /// in: the 3D PNG export's size is that of the on-screen pixels times the PNG scale.
    last_view: Option<(egui::Vec2, f32, u64)>,
    /// `(viewport, zoom HUD, legend)` rects of the last `show` (tests only).
    #[cfg(test)]
    pub(crate) last_layout: Option<(egui::Rect, Option<egui::Rect>, Option<egui::Rect>)>,
    /// `(gizmo, union of the drawn preset buttons, drawn hint)` rects of the last `show` (tests only).
    #[cfg(test)]
    pub(crate) last_overlays: Option<(Option<egui::Rect>, Option<egui::Rect>, Option<egui::Rect>)>,
    /// `(label, planned rect, drawn response rect)` of each preset button in the last `show` (tests only).
    #[cfg(test)]
    pub(crate) last_preset_buttons: Vec<(&'static str, egui::Rect, egui::Rect)>,
}

impl Default for View3dState {
    fn default() -> Self {
        Self {
            yaw: 0.55,
            pitch: 0.35,
            distance: DIST_DEFAULT,
            z_up: false,
            drag_pans: None,
            camera_offset: egui::Vec2::ZERO,
            counters: Arc::new(GpuFrameCounters::default()),
            mesh_generation: 1,
            mesh_data: Arc::new(GpuMeshData::default()),
            overlay_caption: String::new(),
            legend: None,
            last_view: None,
            #[cfg(test)]
            last_layout: None,
            #[cfg(test)]
            last_overlays: None,
            #[cfg(test)]
            last_preset_buttons: Vec::new(),
        }
    }
}

impl View3dState {
    /// Replace the mesh and caption; clears the colour legend (a tab that colours by a scalar sets it again with
    /// [`Self::set_legend`] right after, so no other tab's legend outlives its mesh).
    pub fn set_mesh(&mut self, mesh: GpuMeshData, caption: impl Into<String>) {
        self.mesh_generation = self.mesh_generation.wrapping_add(1);
        self.mesh_data = Arc::new(mesh);
        self.overlay_caption = caption.into();
        self.legend = None;
    }

    /// Colorbar of the current scalar colour mode (`None` for categorical colouring such as face ring size).
    pub fn set_legend(&mut self, legend: Option<ColorLegend3d>) {
        self.legend = legend;
    }

    pub fn legend(&self) -> Option<&ColorLegend3d> {
        self.legend.as_ref()
    }

    pub fn mesh_generation(&self) -> u64 {
        self.mesh_generation
    }

    pub fn has_geometry(&self) -> bool {
        !self.mesh_data.faces.is_empty() || !self.mesh_data.wires.is_empty()
    }

    pub fn is_current_mesh_painted(&self) -> bool {
        self.counters.painted_generation() == self.mesh_generation
            && self.counters.painted_primitives() > 0
    }

    pub fn mesh_data(&self) -> &GpuMeshData {
        &self.mesh_data
    }

    pub fn caption(&self) -> &str {
        &self.overlay_caption
    }

    pub fn show(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        let available = ui.available_size().max(egui::vec2(64.0, 64.0));
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());
        self.last_view = Some((rect.size(), ui.ctx().pixels_per_point(), ui.ctx().cumulative_pass_nr()));
        let hud_opt = zoom_hud_rect_3d(rect);
        let has_hud = hud_opt.is_some();
        let hud_rect = hud_opt.unwrap_or(egui::Rect::NOTHING);
        let gizmo_opt = gizmo_rect_3d(rect);
        let preset_buttons = preset_button_rects(ui, rect);
        let blockers: Vec<egui::Rect> =
            [hud_opt, gizmo_opt].into_iter().flatten().chain(preset_buttons.iter().map(|b| b.1)).collect();
        self.handle_camera_input(ui, rect, &response, &blockers);
        let gizmo_hover = gizmo_opt.and_then(|g| {
            let g_resp = ui.interact(g, response.id.with("axis_gizmo"), egui::Sense::click());
            let hit = g_resp.hover_pos().and_then(|p| gizmo_hit(&self.rotation(), g, p));
            if let Some(end) = g_resp.interact_pointer_pos().filter(|_| g_resp.clicked()).and_then(|p| gizmo_hit(&self.rotation(), g, p)) {
                self.apply_preset(preset_for_axis(end.axis));
            }
            if hit.is_some() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            hit
        });

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, tokens.bg_viewport_3d);
        let aspect = (rect.width() / rect.height().max(1.0)).max(0.1);
        let rot = self.rotation();
        let mvp = self.view_mvp(aspect, &rot);
        painter.add(egui_wgpu::Callback::new_paint_callback(
            rect,
            View3dCallback {
                mvp,
                rot,
                is_light: tokens.is_light,
                mesh_generation: self.mesh_generation,
                mesh_data: Arc::clone(&self.mesh_data),
                counters: Arc::clone(&self.counters),
            },
        ));
        if !self.overlay_caption.is_empty() {
            painter.text(
                rect.left_top() + egui::vec2(8.0, 6.0),
                egui::Align2::LEFT_TOP,
                &self.overlay_caption,
                egui::FontId::proportional(12.0),
                tokens.text_primary,
            );
        }
        let avail_hint_w = if has_hud {
            (hud_rect.left() - rect.left() - 14.0).max(0.0)
        } else {
            (rect.width() - 16.0).max(0.0)
        };
        // Longest hint whose measured width fits left of the zoom HUD.
        let hint_galley = [
            "Drag: orbit | Shift/right-drag: pan | Scroll/pinch/+/-: zoom | 0/dbl-click: fit",
            "Drag: orbit | Shift/right-drag: pan | Scroll: zoom",
            "Drag: orbit | Scroll: zoom",
        ]
        .into_iter()
        .map(|t| painter.layout_no_wrap(t.to_owned(), egui::FontId::proportional(11.0), tokens.text_muted))
        .find(|g| g.size().x <= avail_hint_w);
        let hint_rect = hint_galley.map(|galley| {
            let pos = rect.left_bottom() + egui::vec2(8.0, -6.0 - galley.size().y);
            let r = egui::Rect::from_min_size(pos, galley.size());
            painter.galley(pos, galley, tokens.text_muted);
            r
        });
        let legend_rect = self.legend().filter(|_| self.has_geometry()).and_then(|_| legend_rect_3d(rect));
        if let (Some(leg), Some(box_r)) = (self.legend(), legend_rect) {
            draw_legend_3d(&painter, box_r, leg, tokens);
        }
        if let Some(g) = gizmo_opt {
            draw_gizmo(&painter, g, &rot, gizmo_hover, tokens);
        }
        let mut row_drawn: Option<egui::Rect> = None;
        #[cfg(test)]
        self.last_preset_buttons.clear();
        for (button, planned) in preset_buttons {
            // `put` lays the button out justified, so it fills exactly the planned rect the blockers use.
            let label = egui::RichText::new(button_label(button)).size(11.0);
            let resp = ui.put(planned, egui::Button::new(label).small()).on_hover_text(preset_tooltip(button));
            row_drawn = Some(row_drawn.map_or(resp.rect, |r| r.union(resp.rect)));
            #[cfg(test)]
            self.last_preset_buttons.push((button_label(button), planned, resp.rect));
            if resp.clicked() {
                match button {
                    Some(preset) => self.apply_preset(preset),
                    None => self.reset(),
                }
            }
        }
        #[cfg(test)]
        {
            self.last_layout = Some((rect, hud_opt, legend_rect));
            self.last_overlays = Some((gizmo_opt, row_drawn, hint_rect));
        }
        #[cfg(not(test))]
        let _ = (row_drawn, hint_rect);
        painter.rect_stroke(
            rect,
            4.0,
            egui::Stroke::new(1.0, tokens.border_subtle),
            egui::StrokeKind::Inside,
        );
        if has_hud {
            ui.scope_builder(egui::UiBuilder::new().max_rect(hud_rect), |ui| {
                egui::Frame::NONE
                    .fill(tokens.bg_hud)
                    .stroke(egui::Stroke::new(1.0, tokens.border_strong))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::symmetric(5, 3))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            if ui.small_button("−").on_hover_text("Zoom out (-)").clicked() {
                                self.zoom_by(1.0 / ZOOM_STEP_FACTOR);
                            }
                            let pct = ((DIST_DEFAULT / self.distance.max(1e-3)) * 100.0).round() as i32;
                            ui.colored_label(
                                tokens.text_primary,
                                egui::RichText::new(format!("{pct}%")).size(11.0),
                            );
                            if ui.small_button("+").on_hover_text("Zoom in (+)").clicked() {
                                self.zoom_by(ZOOM_STEP_FACTOR);
                            }
                            if ui.small_button("Fit").on_hover_text("Reset 3D view (0 / double-click)").clicked() {
                                self.reset();
                            }
                        });
                    });
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::View3dState;
    use crate::mesh3d::GpuMeshData;

    #[test]
    fn orbit_wraps_yaw_clamps_pitch_and_reset_restores_the_default_view() {
        let mut v = View3dState::default();
        let start = (v.yaw, v.pitch, v.distance);
        for _ in 0..10_000 {
            v.orbit_by(0.37, 0.05);
        }
        assert!(v.yaw.abs() <= core::f32::consts::PI + 1e-6, "yaw {} not wrapped", v.yaw);
        assert_eq!(v.pitch, 1.50);
        v.orbit_by(0.0, -10.0);
        assert_eq!(v.pitch, -1.50);
        // wrapping keeps the angle: 10 000 x 0.37 rad
        let want = (start.0 + 10_000.0 * 0.37_f32).rem_euclid(core::f32::consts::TAU);
        let got = v.yaw.rem_euclid(core::f32::consts::TAU);
        assert!((want - got).abs() < 1e-2 || (want - got).abs() > core::f32::consts::TAU - 1e-2);
        v.distance = 7.0;
        v.reset();
        assert_eq!((v.yaw, v.pitch, v.distance), start);
    }

    #[test]
    fn set_mesh_bumps_the_generation_so_the_gpu_buffers_are_re_uploaded() {
        let mut v = View3dState::default();
        let g0 = v.mesh_generation();
        v.set_mesh(GpuMeshData::default(), "a");
        v.set_mesh(GpuMeshData::default(), "b");
        assert_eq!(v.mesh_generation(), g0 + 2);
        assert_eq!(v.caption(), "b");
    }
}
