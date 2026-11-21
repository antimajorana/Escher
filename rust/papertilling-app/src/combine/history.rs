//! 32-deep undo/redo history, pin deletion/flip/auto-align helpers, and coalesced cut-angle edits.

use papertilling_core::defect::surgery::DefectSpec;

use super::CombineTabState;
use crate::backend::CatalogueItemUi;
use crate::canvas2d::Canvas2D;
use crate::live_combine::cuts::{
    default_cut_angle_deg, pin_explicit_cut_vec, set_pin_cut_angle,
};

pub(crate) const MAX_HISTORY: usize = 32;

/// Interaction flags returned by the shared cut-angle `DragValue` widget.
pub(crate) struct CutDragResponse {
    pub deg: f64,
    pub changed: bool,
    pub started: bool,
    pub finished: bool,
    pub active: bool,
}

/// Render a `-180..=180` degree `DragValue` and report coalescing lifecycle flags.
pub(crate) fn show_cut_angle_drag_value(
    ui: &mut egui::Ui,
    mut deg: f64,
    speed: f64,
) -> CutDragResponse {
    let resp = ui.add(
        egui::DragValue::new(&mut deg)
            .range(-180.0..=180.0)
            .speed(speed)
            .fixed_decimals(0)
            .suffix("°"),
    );
    CutDragResponse {
        deg,
        changed: resp.changed(),
        started: resp.drag_started() || resp.gained_focus(),
        finished: resp.drag_stopped() || resp.lost_focus(),
        active: resp.dragged() || resp.has_focus(),
    }
}

impl CombineTabState {
    pub(crate) fn clear_selection(&mut self) {
        self.selected_pin = None;
        self.popover_open = false;
    }

    pub(crate) fn record_undo_snapshot(&mut self, snapshot: Vec<DefectSpec>) {
        if self.undo_stack.last().is_some_and(|prev| prev == &snapshot) {
            self.redo_stack.clear();
            return;
        }
        if self.undo_stack.len() >= MAX_HISTORY {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(snapshot);
        self.redo_stack.clear();
    }

    pub fn push_undo(&mut self) {
        self.commit_cut_edit_anchor();
        self.record_undo_snapshot(self.specs.clone());
    }

    pub(crate) fn commit_cut_edit_anchor(&mut self) -> bool {
        if let Some((_, snap)) = self.cut_edit_anchor.take()
            && snap != self.specs
        {
            self.record_undo_snapshot(snap);
            return true;
        }
        false
    }

    /// Returns `Some(true)` for a preview-only build while dragging, `Some(false)` for a full build,
    /// or `None` when nothing changed.
    pub(crate) fn apply_cut_drag_response(&mut self, idx: usize, ev: CutDragResponse) -> Option<bool> {
        if idx >= self.specs.len() {
            return None;
        }
        if ev.started {
            self.commit_cut_edit_anchor();
            self.cut_edit_anchor = Some((idx, self.specs.clone()));
        }
        let mut build_preview: Option<bool> = None;
        if ev.changed {
            if self.cut_edit_anchor.is_none() {
                if ev.active {
                    self.cut_edit_anchor = Some((idx, self.specs.clone()));
                } else {
                    self.push_undo();
                }
            }
            set_pin_cut_angle(&mut self.specs[idx], Some(ev.deg));
            build_preview = Some(ev.active && !ev.finished);
        }
        if ev.finished
            && self.cut_edit_anchor.as_ref().is_some_and(|(i, _)| *i == idx)
            && self.commit_cut_edit_anchor()
        {
            build_preview = Some(false);
        }
        build_preview
    }

    fn preserve_or_clear_selection(&mut self, target: &[DefectSpec]) {
        let keep = self
            .selected_pin
            .is_some_and(|i| self.specs.get(i).is_some() && self.specs.get(i) == target.get(i));
        if !keep {
            self.clear_selection();
        }
    }

    fn reset_transient_drag_anchors(&mut self) {
        self.drag_anchor = None;
        self.pending_disc_step = None;
        self.pending_absorb = None;
        self.dragging = false;
        self.cut_drag_anchor = None;
        self.cut_edit_anchor = None;
        self.cut_dragging = false;
    }

    pub fn undo_last(&mut self, gallery_item: &CatalogueItemUi) {
        if self.lattice_unavailable() || self.dragging || self.cut_dragging {
            return;
        }
        self.commit_cut_edit_anchor();
        if let Some(prev) = self.undo_stack.pop() {
            self.preserve_or_clear_selection(&prev);
            let cur = std::mem::replace(&mut self.specs, prev);
            if self.redo_stack.len() >= MAX_HISTORY {
                self.redo_stack.remove(0);
            }
            self.redo_stack.push(cur);
            self.reset_transient_drag_anchors();
            self.kinematic_status = String::from("Undid last defect edit");
            self.request_build(gallery_item, false, self.auto_compare, false);
        }
    }

    pub fn redo_last(&mut self, gallery_item: &CatalogueItemUi) {
        if self.lattice_unavailable() || self.dragging || self.cut_dragging {
            return;
        }
        if let Some(next) = self.redo_stack.pop() {
            self.preserve_or_clear_selection(&next);
            let cur = std::mem::replace(&mut self.specs, next);
            if self.undo_stack.len() >= MAX_HISTORY {
                self.undo_stack.remove(0);
            }
            self.undo_stack.push(cur);
            self.reset_transient_drag_anchors();
            self.kinematic_status = String::from("Redid defect edit");
            self.request_build(gallery_item, false, self.auto_compare, false);
        }
    }

    pub fn delete_pin(
        &mut self,
        idx: usize,
        gallery_item: &CatalogueItemUi,
        canvas: Option<&mut Canvas2D>,
    ) -> bool {
        if self.lattice_unavailable() || self.dragging || self.cut_dragging || idx >= self.specs.len() {
            return false;
        }
        self.commit_cut_edit_anchor();
        self.push_undo();
        self.clear_kinematic_drag_state(false);
        self.specs.remove(idx);
        match self.selected_pin {
            Some(s) if s == idx => self.clear_selection(),
            Some(s) if s > idx => self.selected_pin = Some(s - 1),
            _ => {}
        }
        if let Some(cv) = canvas {
            self.sync_selection_to_canvas(cv);
        }
        self.request_build(gallery_item, false, self.auto_compare, false);
        true
    }

    pub(crate) fn flip_pin(
        &mut self,
        idx: usize,
        gallery_item: &CatalogueItemUi,
        canvas: Option<&mut Canvas2D>,
    ) -> bool {
        if self.lattice_unavailable() || self.dragging || self.cut_dragging || idx >= self.specs.len() {
            return false;
        }
        self.commit_cut_edit_anchor();
        self.push_undo();
        let hw_deg = if matches!(self.specs[idx], DefectSpec::HalfWall { direction: None, .. }) {
            Some(default_cut_angle_deg(
                &self.info,
                &self.specs,
                idx,
                &self.last_build.applied_cuts,
            ))
        } else {
            None
        };
        match &mut self.specs[idx] {
            DefectSpec::Disclination { angle_deg, .. } => *angle_deg = -*angle_deg,
            DefectSpec::Dislocation { b, .. } => *b = [-b[0], -b[1]],
            DefectSpec::HalfWall { direction, .. } => {
                let cur = direction.unwrap_or_else(|| {
                    let rad = hw_deg.unwrap_or(0.0).to_radians();
                    [rad.cos(), rad.sin()]
                });
                *direction = Some([-cur[0], -cur[1]]);
            }
        }
        if let Some(cv) = canvas {
            self.sync_selection_to_canvas(cv);
        }
        self.request_build(gallery_item, false, self.auto_compare, false);
        true
    }

    pub(crate) fn flip_pin_cut_180(
        &mut self,
        idx: usize,
        gallery_item: &CatalogueItemUi,
        canvas: Option<&mut Canvas2D>,
    ) -> bool {
        if self.lattice_unavailable() || self.dragging || self.cut_dragging || idx >= self.specs.len() {
            return false;
        }
        self.push_undo();
        let cur = match pin_explicit_cut_vec(&self.specs[idx]) {
            Some(c) => {
                let l = c[0].hypot(c[1]).max(1e-12);
                [c[0] / l, c[1] / l]
            }
            None => {
                let rad = default_cut_angle_deg(
                    &self.info,
                    &self.specs,
                    idx,
                    &self.last_build.applied_cuts,
                )
                .to_radians();
                [rad.cos(), rad.sin()]
            }
        };
        let opp_deg = (-cur[1]).atan2(-cur[0]).to_degrees();
        set_pin_cut_angle(&mut self.specs[idx], Some(opp_deg));
        if let Some(cv) = canvas {
            self.sync_selection_to_canvas(cv);
        }
        self.request_build(gallery_item, false, self.auto_compare, false);
        true
    }

    pub(crate) fn auto_align_pin_cut(
        &mut self,
        idx: usize,
        gallery_item: &CatalogueItemUi,
        canvas: Option<&mut Canvas2D>,
    ) -> bool {
        if self.lattice_unavailable() || self.dragging || self.cut_dragging || idx >= self.specs.len() {
            return false;
        }
        if pin_explicit_cut_vec(&self.specs[idx]).is_none() {
            return false;
        }
        self.commit_cut_edit_anchor();
        self.push_undo();
        set_pin_cut_angle(&mut self.specs[idx], None);
        if let Some(cv) = canvas {
            self.sync_selection_to_canvas(cv);
        }
        self.request_build(gallery_item, false, self.auto_compare, false);
        true
    }

    pub(crate) fn finish_cut_drag(
        &mut self,
        gallery_item: &CatalogueItemUi,
        canvas: Option<&mut Canvas2D>,
    ) {
        if !std::mem::take(&mut self.cut_dragging) {
            self.cut_drag_anchor = None;
            return;
        }
        if let Some((_, anchor_specs)) = self.cut_drag_anchor.take()
            && anchor_specs != self.specs
        {
            self.record_undo_snapshot(anchor_specs);
        }
        if let Some(cv) = canvas {
            self.sync_selection_to_canvas(cv);
        }
        self.request_build(gallery_item, false, self.auto_compare, false);
    }
}
