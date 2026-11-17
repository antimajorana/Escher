//! Interactive particle-conserving kinematics (`Delta_N = 0`), visual glide rails,
//! live reaction previews, and single-step Undo for the Combine tab.

use papertilling_core::defect::kinematics::{
    absorb_dislocation, find_absorption_preview, parallel_transport_glided_dislocation,
    preview_emitted_burgers, project_glide_at, step_disclination_emit,
};
use papertilling_core::defect::surgery::DefectSpec;

use super::CombineTabState;
use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::{Canvas2D, CanvasKinematicOverlay, CanvasReactionArrow};
use crate::live_combine::{mn_from_burgers, spec_world_pos};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionMode {
    Conserved,
    StepDisclinationEmit,
    FreeSetup,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PendingDiscStep {
    pub idx: usize,
    pub from: [f64; 2],
    pub to: [f64; 2],
    pub b_prev: [f64; 2],
}

impl CombineTabState {
    pub fn push_undo(&mut self) {
        if self.undo_stack.last().is_some_and(|prev| prev == &self.specs) {
            return;
        }
        if self.undo_stack.len() >= 32 {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(self.specs.clone());
    }

    pub fn undo_last(&mut self, gallery_item: &CatalogueItemUi) {
        if self.lattice_unavailable() {
            return;
        }
        if let Some(prev) = self.undo_stack.pop() {
            self.drag_anchor = None;
            self.pending_disc_step = None;
            self.pending_absorb = None;
            self.dragging = false;
            self.specs = prev;
            self.kinematic_status = String::from("Undid last defect edit");
            self.request_build(gallery_item, false, self.auto_compare, false);
        }
    }

    pub(crate) fn clear_kinematic_drag_state(&mut self, clear_undo: bool) {
        self.drag_anchor = None;
        self.pending_disc_step = None;
        self.pending_absorb = None;
        self.kinematic_status.clear();
        if clear_undo {
            self.undo_stack.clear();
        }
    }

    pub fn current_kinematic_overlay(&self) -> CanvasKinematicOverlay {
        let mut ov = CanvasKinematicOverlay::default();
        let reach = self.last_build.applied_radius.max(3.5);
        let dragged_idx = self.drag_anchor.as_ref().map(|(i, _)| *i);

        for (idx, sp) in self.specs.iter().enumerate() {
            match *sp {
                DefectSpec::Dislocation { b, .. } if self.show_glide_lines => {
                    let lb = b[0].hypot(b[1]);
                    if lb > 1e-12 {
                        let p = spec_world_pos(&self.info, idx, sp);
                        let u = [b[0] / lb, b[1] / lb];
                        let a0 = [p[0] - reach * u[0], p[1] - reach * u[1]];
                        let a1 = [p[0] + reach * u[0], p[1] + reach * u[1]];
                        ov.glide_lines.push((a0, a1, dragged_idx == Some(idx)));
                    }
                }
                DefectSpec::Disclination { .. } if self.motion_mode == MotionMode::Conserved => {
                    ov.locked_pins.push(idx);
                }
                _ => {}
            }
        }

        if let Some(step) = self.pending_disc_step {
            let b_lbl = mn_from_burgers(&self.info, step.b_prev)
                .map_or_else(|| format!("({:.2},{:.2})", step.b_prev[0], step.b_prev[1]), |(m, n)| format!("T({m},{n})"));
            ov.reaction = Some(CanvasReactionArrow {
                from: step.from,
                to: step.to,
                label: format!("emits {b_lbl}"),
                valid: true,
            });
        } else if let Some(abs) = self.pending_absorb
            && let Some((idx, _)) = self.drag_anchor.as_ref()
            && let Some(sp) = self.specs.get(*idx)
        {
            let pos = spec_world_pos(&self.info, *idx, sp);
            ov.reaction = Some(CanvasReactionArrow {
                from: pos,
                to: abs.new_at,
                label: format!("absorbs -> D#{} ({:+.2}, {:+.2})", abs.disc_idx + 1, abs.new_at[0], abs.new_at[1]),
                valid: true,
            });
        }
        ov
    }

    pub fn move_pin_with_mods(
        &mut self,
        idx: usize,
        world: [f64; 2],
        shift_held: bool,
        gallery_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) -> bool {
        if self.lattice_unavailable() || idx >= self.specs.len() {
            return false;
        }
        let orig_specs = self
            .drag_anchor
            .as_ref()
            .filter(|(i, _)| *i == idx)
            .map(|(_, s)| s.clone())
            .unwrap_or_else(|| self.specs.clone());
        let item = self.effective_item(gallery_item);

        if self.motion_mode == MotionMode::FreeSetup {
            let old_pos = spec_world_pos(&self.info, idx, &self.specs[idx]);
            let new_spec = match self.specs[idx] {
                DefectSpec::Disclination { angle_deg, cut, .. } => {
                    let (snapped, _) = backend.snap_disclination(item, &self.info, angle_deg, None, world);
                    if (snapped[0] - old_pos[0]).hypot(snapped[1] - old_pos[1]) < 1e-6 {
                        if !self.dragging {
                            self.drag_anchor = None;
                        }
                        return false;
                    }
                    DefectSpec::Disclination { angle_deg, at: snapped, cut }
                }
                DefectSpec::Dislocation { b, cut, .. } => {
                    if (world[0] - old_pos[0]).hypot(world[1] - old_pos[1]) < 1e-4 {
                        if !self.dragging {
                            self.drag_anchor = None;
                        }
                        return false;
                    }
                    DefectSpec::Dislocation { b, at: Some(world), cut }
                }
                DefectSpec::HalfWall { ref wall_id, direction, sigma, .. } => {
                    if (world[0] - old_pos[0]).hypot(world[1] - old_pos[1]) < 1e-4 {
                        if !self.dragging {
                            self.drag_anchor = None;
                        }
                        return false;
                    }
                    DefectSpec::HalfWall { wall_id: wall_id.clone(), at: world, direction, sigma }
                }
            };
            self.drag_anchor = Some((idx, orig_specs));
            self.specs[idx] = new_spec;
            self.dragging = true;
            canvas.set_kinematic_overlay(self.current_kinematic_overlay());
            self.request_build(gallery_item, true, false, false);
            return true;
        }

        match self.specs[idx] {
            DefectSpec::Dislocation { b, cut, .. } => {
                let anchor_at = orig_specs
                    .get(idx)
                    .map_or_else(|| spec_world_pos(&self.info, idx, &self.specs[idx]), |s| spec_world_pos(&self.info, idx, s));
                let old_pos = spec_world_pos(&self.info, idx, &self.specs[idx]);
                let mut new_at = project_glide_at(b, anchor_at, world);

                // Snap onto a same-line partner pin if dragged within 0.25*lmax for annihilation/fusion.
                let lb = b[0].hypot(b[1]).max(1e-12);
                let (ub, nb) = ([b[0] / lb, b[1] / lb], [-b[1] / lb, b[0] / lb]);
                let tol = 0.25 * self.info.lat_info.lmax;
                for (j, sj) in self.specs.iter().enumerate() {
                    if j != idx && matches!(sj, DefectSpec::Dislocation { .. }) {
                        let pj = spec_world_pos(&self.info, j, sj);
                        let da = [pj[0] - anchor_at[0], pj[1] - anchor_at[1]];
                        if (da[0] * nb[0] + da[1] * nb[1]).abs() <= 1e-4
                            && (pj[0] - new_at[0]).hypot(pj[1] - new_at[1]) <= tol
                        {
                            let s = da[0] * ub[0] + da[1] * ub[1];
                            new_at = [anchor_at[0] + s * ub[0], anchor_at[1] + s * ub[1]];
                            break;
                        }
                    }
                }
                self.pending_absorb = find_absorption_preview(&self.info.cell, &orig_specs, idx, new_at);
                if (new_at[0] - old_pos[0]).hypot(new_at[1] - old_pos[1]) < 1e-4 {
                    if !self.dragging {
                        self.drag_anchor = None;
                        self.pending_absorb = None;
                    }
                    canvas.set_kinematic_overlay(self.current_kinematic_overlay());
                    return false;
                }
                self.drag_anchor = Some((idx, orig_specs));
                self.specs[idx] = DefectSpec::Dislocation { b, at: Some(new_at), cut };
                self.dragging = true;
                canvas.set_kinematic_overlay(self.current_kinematic_overlay());
                self.request_build(gallery_item, true, false, false);
                true
            }
            DefectSpec::Disclination { angle_deg, at: c_old, .. } => {
                let allow_step = self.motion_mode == MotionMode::StepDisclinationEmit || shift_held;
                if !allow_step {
                    if !self.dragging {
                        self.drag_anchor = None;
                    }
                    self.kinematic_status = String::from(
                        "Disclinations are immobile alone (ΔN = 0). Hold Shift + drag or choose 'Step D (emit b)' to move by creating a dislocation.",
                    );
                    canvas.set_kinematic_overlay(self.current_kinematic_overlay());
                    return false;
                }
                let (snapped, _) = backend.snap_disclination(item, &self.info, angle_deg, None, world);
                if (snapped[0] - c_old[0]).hypot(snapped[1] - c_old[1]) < 1e-6 {
                    self.pending_disc_step = None;
                    if !self.dragging {
                        self.drag_anchor = None;
                    }
                    canvas.set_kinematic_overlay(self.current_kinematic_overlay());
                    return false;
                }
                self.drag_anchor = Some((idx, orig_specs));
                let b_prev = preview_emitted_burgers(angle_deg, c_old, snapped);
                self.pending_disc_step = Some(PendingDiscStep { idx, from: c_old, to: snapped, b_prev });
                self.dragging = true;
                canvas.set_kinematic_overlay(self.current_kinematic_overlay());
                true
            }
            DefectSpec::HalfWall { .. } => {
                self.kinematic_status = String::from("Half-wall pins reposition in Free Setup mode.");
                false
            }
        }
    }

    pub fn finish_drag_kinematics(&mut self, gallery_item: &CatalogueItemUi, mut canvas: Option<&mut Canvas2D>) {
        if !std::mem::take(&mut self.dragging) {
            self.drag_anchor = None;
            self.pending_disc_step = None;
            self.pending_absorb = None;
            return;
        }
        let anchor = self.drag_anchor.take();
        let disc_step = self.pending_disc_step.take();
        let absorb = self.pending_absorb.take();
        let item = self.effective_item(gallery_item).clone();

        if let (Some((_, anchor_specs)), Some(step)) = (&anchor, disc_step) {
            match step_disclination_emit(&item.group, &self.info.cell, item.frame, anchor_specs, step.idx, step.to) {
                Ok(res) => {
                    self.undo_stack.push(anchor_specs.clone());
                    if self.undo_stack.len() > 32 {
                        self.undo_stack.remove(0);
                    }
                    let em_lbl = res.emitted.as_ref().and_then(|s| match *s {
                        DefectSpec::Dislocation { b, .. } => mn_from_burgers(&self.info, b)
                            .map(|(m, n)| format!("T({m},{n})")),
                        _ => None,
                    }).unwrap_or_else(|| String::from("dislocation"));
                    self.specs = res.specs;
                    self.kinematic_status = format!(
                        "Stepped disclination ({:+.2}, {:+.2}) -> ({:+.2}, {:+.2}) and emitted {em_lbl} on ΔN = 0 glide line",
                        step.from[0], step.from[1], step.to[0], step.to[1]
                    );
                }
                Err(err) => {
                    self.specs = anchor_specs.clone();
                    self.kinematic_status = format!("Step refused: {err}");
                }
            }
            if let Some(cv) = canvas.as_deref_mut() {
                cv.set_kinematic_overlay(self.current_kinematic_overlay());
            }
            self.request_build(gallery_item, false, self.auto_compare, false);
            return;
        }

        if let (Some((idx, anchor_specs)), Some(abs_prev)) = (&anchor, absorb)
            && let Ok((new_specs, c_new)) = absorb_dislocation(
                &item.group, &self.info.cell, item.frame, anchor_specs, *idx, abs_prev.disc_idx,
            )
        {
            self.undo_stack.push(anchor_specs.clone());
            if self.undo_stack.len() > 32 {
                self.undo_stack.remove(0);
            }
            self.specs = new_specs;
            self.kinematic_status = format!(
                "Absorbed dislocation into disclination #{}: shifted centre to ({:+.2}, {:+.2}) (ΔN = 0)",
                abs_prev.disc_idx + 1, c_new[0], c_new[1]
            );
            if let Some(cv) = canvas.as_deref_mut() {
                cv.set_kinematic_overlay(self.current_kinematic_overlay());
            }
            self.request_build(gallery_item, false, self.auto_compare, false);
            return;
        }

        if self.motion_mode != MotionMode::FreeSetup
            && let Some((idx, anchor_specs)) = &anchor
        {
            match parallel_transport_glided_dislocation(
                &item.group, &self.info.cell, item.frame, anchor_specs, &self.specs, *idx,
            ) {
                Ok((final_specs, rotated)) => {
                    self.undo_stack.push(anchor_specs.clone());
                    if self.undo_stack.len() > 32 {
                        self.undo_stack.remove(0);
                    }
                    self.specs = final_specs;
                    self.kinematic_status = if rotated {
                        String::from("Glided across branch cut: Burgers vector parallel-transported to preserve G~+ holonomy")
                    } else {
                        String::from("Glided dislocation along b (ΔN = 0, climb = 0)")
                    };
                }
                Err(err) => {
                    self.specs = anchor_specs.clone();
                    self.kinematic_status = format!("Glide reverted: {err}");
                }
            }
        } else if let Some((_, anchor_specs)) = anchor {
            self.undo_stack.push(anchor_specs);
            if self.undo_stack.len() > 32 {
                self.undo_stack.remove(0);
            }
        }
        if let Some(cv) = canvas {
            cv.set_kinematic_overlay(self.current_kinematic_overlay());
        }
        self.request_build(gallery_item, false, self.auto_compare, false);
    }
}
