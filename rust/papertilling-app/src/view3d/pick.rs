//! Screen-space picking support of the 3D viewport: the pointer of the last `show` (viewport rect, hover and click
//! positions outside the overlays and outside camera drags) and the projection of mesh-space points through the
//! renderer's MVP (`View3dState::view_mvp` with `View3dState::rotation`, so pan, presets and the z-up turntable are
//! respected exactly as drawn).

use super::View3dState;
use super::gizmo::{gizmo_rect_3d, preset_button_rects};
use super::legend::{legend_rect_3d, zoom_hud_rect_3d};

type Mat4 = [[f32; 4]; 4];

/// Pointer state of the last drawn 3D viewport. `hover` / `click` are `None` over the overlays (zoom HUD, gizmo,
/// preset buttons, legend) and during drags; `click` is a primary click (press and release without a drag).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View3dPointer {
    pub rect: egui::Rect,
    pub hover: Option<egui::Pos2>,
    pub click: Option<egui::Pos2>,
}

impl View3dState {
    /// The MVP `show` renders `rect` with (same aspect clamp, rotation and pan offset).
    pub fn mvp_for_rect(&self, rect: egui::Rect) -> Mat4 {
        let aspect = (rect.width() / rect.height().max(1.0)).max(0.1);
        self.view_mvp(aspect, &self.rotation())
    }

    /// Pointer of the last `show` (`None` before the first one).
    pub fn pointer(&self) -> Option<View3dPointer> {
        self.pointer
    }

    pub(super) fn record_pointer(&mut self, ui: &egui::Ui, rect: egui::Rect, response: &egui::Response) {
        let legend_blocker = (self.legend().is_some() && self.has_geometry())
            .then(|| legend_rect_3d(rect))
            .flatten();
        let mut blockers: Vec<egui::Rect> =
            [zoom_hud_rect_3d(rect), gizmo_rect_3d(rect), legend_blocker].into_iter().flatten().collect();
        blockers.extend(preset_button_rects(ui, rect).into_iter().map(|b| b.1));
        let free = |p: egui::Pos2| rect.contains(p) && !blockers.iter().any(|b| b.contains(p));
        let dragging = response.dragged();
        let hover = response.hover_pos().filter(|p| !dragging && free(*p));
        let click = response
            .interact_pointer_pos()
            .filter(|p| response.clicked_by(egui::PointerButton::Primary) && free(*p));
        self.pointer = Some(View3dPointer { rect, hover, click });
    }
}

/// Screen position (pixels, y down) and clip `w` (distance in front of the eye) of mesh-space point `p`, or `None`
/// when it is behind the eye.
pub fn project_point(mvp: &Mat4, rect: egui::Rect, p: [f32; 3]) -> Option<(egui::Pos2, f32)> {
    let clip: [f32; 4] = core::array::from_fn(|i| mvp[0][i] * p[0] + mvp[1][i] * p[1] + mvp[2][i] * p[2] + mvp[3][i]);
    if clip[3].is_nan() || clip[3] <= 1e-6 {
        return None;
    }
    let (nx, ny) = (clip[0] / clip[3], clip[1] / clip[3]);
    Some((
        egui::pos2(rect.center().x + 0.5 * rect.width() * nx, rect.center().y - 0.5 * rect.height() * ny),
        clip[3],
    ))
}

/// Index of the visible point whose projection lies within `radius_px` of `pos`, the one nearest the camera
/// (smallest clip `w`) when several do (points stacked along the view axis); `visible[i] == false` is never picked.
pub fn pick_nearest(
    points: &[[f32; 3]],
    visible: &[bool],
    mvp: &Mat4,
    rect: egui::Rect,
    pos: egui::Pos2,
    radius_px: f32,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &p) in points.iter().enumerate() {
        if !visible.get(i).copied().unwrap_or(false) {
            continue;
        }
        let Some((s, w)) = project_point(mvp, rect, p) else { continue };
        if s.distance(pos) <= radius_px && best.is_none_or(|(_, bw)| w < bw) {
            best = Some((i, w));
        }
    }
    best.map(|b| b.0)
}
