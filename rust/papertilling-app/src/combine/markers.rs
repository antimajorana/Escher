use papertilling_core::defect::surgery::DefectSpec;

use super::{CombineTabState, ToolMode};
use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::{Canvas2D, CanvasCutRay, CanvasMarker, MarkerKind, Scene2D};
use crate::live_combine::{
    burgers_from_mn,
    cuts::{
        build_cut_rays_for_specs, build_fallback_cut_ray_for_spec, is_pin_coincident,
        map_cut_ray_to_drawn,
    },
    map_drawn_to_ref, map_ref_to_drawn, preset_neighbour_pair, preset_shortest_dislocation,
    spec_core_ref_pos, spec_short_label, spec_world_pos,
};

impl CombineTabState {
    pub fn current_cut_rays(&self) -> Vec<CanvasCutRay> {
        if self.lattice_unavailable() {
            return Vec::new();
        }
        let mut rays = if self.last_build.refused.is_some() {
            Vec::new()
        } else {
            build_cut_rays_for_specs(
                &self.info,
                &self.specs,
                &self.last_build.applied_cuts,
                self.last_build.patch_centre,
                self.last_build.applied_radius,
            )
        };
        let drag_idx_opt = self.cut_dragging.then(|| self.cut_drag_anchor.as_ref().map(|(i, _)| *i)).flatten();
        if let Some(drag_idx) = drag_idx_opt
            && let Some(ray) = build_fallback_cut_ray_for_spec(
                &self.info,
                &self.specs,
                drag_idx,
                &self.last_build.applied_cuts,
                self.last_build.patch_centre,
                self.last_build.applied_radius,
            )
        {
            if let Some(existing) = rays.iter_mut().find(|r| r.spec_index == Some(drag_idx)) {
                *existing = ray;
            } else {
                rays.push(ray);
            }
        }
        if let Some(sel_idx) = self.selected_pin
            && !rays.iter().any(|r| r.spec_index == Some(sel_idx))
            && let Some(ray) = build_fallback_cut_ray_for_spec(
                &self.info,
                &self.specs,
                sel_idx,
                &self.last_build.applied_cuts,
                self.last_build.patch_centre,
                self.last_build.applied_radius,
            )
        {
            rays.push(ray);
        }
        if !self.show_branch_cuts {
            rays.retain(|r| {
                r.is_half_wall
                    || (self.selected_pin.is_some() && r.spec_index == self.selected_pin)
                    || (drag_idx_opt.is_some() && r.spec_index == drag_idx_opt)
            });
        }
        let core_pos = self.matched_dislocation_core_positions();
        rays.into_iter()
            .map(|mut r| {
                if let Some(idx) = r.spec_index
                    && let Some(Some(drawn_origin)) = core_pos.get(idx)
                {
                    let dx = drawn_origin[0] - r.origin[0];
                    let dy = drawn_origin[1] - r.origin[1];
                    r.origin = *drawn_origin;
                    r.end = [r.end[0] + dx, r.end[1] + dy];
                    if let Some((sa, sb)) = r.strip {
                        r.strip = Some(([sa[0] + dx, sa[1] + dy], [sb[0] + dx, sb[1] + dy]));
                    }
                    r
                } else {
                    map_cut_ray_to_drawn(&self.last_build.crystal, r)
                }
            })
            .collect()
    }

    pub(crate) fn sync_selection_to_canvas(&self, canvas: &mut Canvas2D) {
        canvas.set_selected_marker(self.selected_pin);
        canvas.set_cut_rays(&self.current_cut_rays());
    }

    pub fn sync_to_canvas(
        &self,
        gallery_item: &CatalogueItemUi,
        canvas: &mut Canvas2D,
        reset_view: bool,
    ) {
        let item = self.effective_item(gallery_item);
        let title = format!(
            "{} ({}) — {} defect(s) | {}",
            item.display_with_orig(),
            item.group,
            self.specs.len(),
            self.last_build.charge_label
        );
        canvas.set_scene(
            Scene2D::CrystalPatch {
                crystal: self.last_build.crystal.clone(),
                title,
                note: self.last_build.note.clone(),
            },
            reset_view,
        );
        canvas.set_export_markers(&self.build_markers());
        canvas.set_selected_marker(self.selected_pin);
        canvas.set_cut_rays(&self.current_cut_rays());
        canvas.set_kinematic_overlay(self.current_kinematic_overlay());
        self.sync_inspectors_to_canvas(gallery_item, canvas);
    }

    pub fn apply_preset_pair(
        &mut self,
        gallery_item: &CatalogueItemUi,
        _backend: &AppBackend,
        _canvas: &mut Canvas2D,
    ) {
        if self.lattice_unavailable() {
            return;
        }
        self.push_undo();
        self.clear_kinematic_drag_state(false);
        self.clear_selection();
        self.specs = preset_neighbour_pair(&self.info);
        self.request_build(gallery_item, false, self.auto_compare, true);
    }

    pub fn apply_preset_dislocation(
        &mut self,
        gallery_item: &CatalogueItemUi,
        _backend: &AppBackend,
        _canvas: &mut Canvas2D,
    ) {
        if self.lattice_unavailable() {
            return;
        }
        self.push_undo();
        self.clear_kinematic_drag_state(false);
        self.clear_selection();
        self.specs = preset_shortest_dislocation(&self.info);
        self.request_build(gallery_item, false, self.auto_compare, true);
    }

    pub fn clear_defects(
        &mut self,
        gallery_item: &CatalogueItemUi,
        _backend: &AppBackend,
        _canvas: &mut Canvas2D,
    ) {
        if self.lattice_unavailable() {
            return;
        }
        if !self.specs.is_empty() {
            self.push_undo();
        }
        self.clear_kinematic_drag_state(false);
        self.clear_selection();
        self.specs.clear();
        self.request_build(gallery_item, false, false, true);
    }

    pub fn place_at_world(
        &mut self,
        world: [f64; 2],
        gallery_item: &CatalogueItemUi,
        backend: &AppBackend,
        _canvas: &mut Canvas2D,
    ) {
        if self.lattice_unavailable() || self.tool_mode == ToolMode::PanAndDrag {
            return;
        }
        match self.tool_mode {
            ToolMode::PanAndDrag => return,
            ToolMode::PlaceDisclination => {
                if self.info.centres.is_empty() {
                    return;
                }
                self.push_undo();
                self.clear_kinematic_drag_state(false);
                let (cid, ang) = self.active_centre_and_angle();
                let item = self.effective_item(gallery_item);
                let (snapped, _) = backend.snap_disclination(item, &self.info, ang, cid, world);
                self.specs.push(DefectSpec::disclination(ang, snapped));
            }
            ToolMode::PlaceDislocation => {
                self.push_undo();
                self.clear_kinematic_drag_state(false);
                if self.custom_mn == (0, 0) {
                    self.custom_mn = (1, 0);
                }
                let b = burgers_from_mn(&self.info, self.custom_mn.0, self.custom_mn.1);
                self.specs.push(DefectSpec::dislocation(b, Some(world)));
            }
            ToolMode::PlaceHalfWall => {
                self.place_half_wall_at(world);
            }
        }
        self.request_build(gallery_item, false, self.auto_compare, false);
    }

    /// Move pin `idx` during a drag (respecting active `MotionMode` and `shift_held` overrides).
    #[cfg(test)]
    pub fn move_pin_to(
        &mut self,
        idx: usize,
        world: [f64; 2],
        gallery_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) -> bool {
        self.move_pin_with_mods(idx, world, false, gallery_item, backend, canvas)
    }

    /// Drag released: apply kinematic transition / parallel transport, then full build + compare.
    #[cfg(test)]
    pub fn finish_drag(&mut self, gallery_item: &CatalogueItemUi) {
        self.finish_drag_kinematics(gallery_item, None);
    }

    /// Match each non-coincident dislocation spec to its topological core ring centroid in
    /// `last_build.crystal.pos` (using `last_build.local_cores`), so pins sit directly on the
    /// actual drawn pentagon/core ring regardless of discrete strip snapping.
    fn matched_dislocation_core_positions(&self) -> Vec<Option<[f64; 2]>> {
        let mut out = vec![None; self.specs.len()];
        let crystal = &self.last_build.crystal;
        if self.last_build.refused.is_some()
            || self.last_build.local_cores.is_empty()
            || crystal.pos.is_empty()
        {
            return out;
        }
        let cores: Vec<([f64; 2], Option<[f64; 2]>)> = self
            .last_build
            .local_cores
            .iter()
            .filter(|c| {
                c.kind == "dislocation"
                    && !c.sites.is_empty()
                    && c.sites.iter().all(|&v| v < crystal.pos.len())
            })
            .map(|c| {
                let inv = 1.0 / (c.sites.len() as f64);
                let cx = c.sites.iter().map(|&v| crystal.pos[v][0]).sum::<f64>() * inv;
                let cy = c.sites.iter().map(|&v| crystal.pos[v][1]).sum::<f64>() * inv;
                ([cx, cy], c.b)
            })
            .collect();
        if cores.is_empty() {
            return out;
        }
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (idx, sp) in self.specs.iter().enumerate() {
            let DefectSpec::Dislocation { b, .. } = *sp else { continue };
            if is_pin_coincident(&self.info, &self.specs, idx) {
                continue;
            }
            let lb = b[0].hypot(b[1]);
            let tol = 1.6 * self.info.lat_info.lmax + lb;
            let approx = map_ref_to_drawn(crystal, spec_core_ref_pos(&self.info, idx, sp));
            let at_drawn = map_ref_to_drawn(crystal, spec_world_pos(&self.info, idx, sp));
            for (ci, &(cpos, cb_opt)) in cores.iter().enumerate() {
                let d = (cpos[0] - approx[0])
                    .hypot(cpos[1] - approx[1])
                    .min((cpos[0] - at_drawn[0]).hypot(cpos[1] - at_drawn[1]));
                if d > tol {
                    continue;
                }
                let b_penalty = cb_opt.map_or(0.0, |cb| {
                    let lcb = cb[0].hypot(cb[1]);
                    if lb > 1e-9 && lcb > 1e-9 {
                        let cross = (cb[0] * b[1] - cb[1] * b[0]).abs() / (lb * lcb);
                        if cross > 0.25 || (lcb - lb).abs() > 0.25 * lb {
                            0.75 * tol
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                });
                pairs.push((d + b_penalty, idx, ci));
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut used_core = vec![false; cores.len()];
        for (_, idx, ci) in pairs {
            if out[idx].is_none() && !used_core[ci] {
                out[idx] = Some(cores[ci].0);
                used_core[ci] = true;
            }
        }
        out
    }

    /// Drawn-space position of pin `idx` (snapped to the matched dislocation core centroid when
    /// available, otherwise mapping `spec_core_ref_pos` into `last_build.crystal.pos`).
    pub(crate) fn marker_world_pos(&self, idx: usize, spec: &DefectSpec) -> [f64; 2] {
        if let Some(Some(pos)) = self.matched_dislocation_core_positions().get(idx) {
            return *pos;
        }
        let core_ref = spec_core_ref_pos(&self.info, idx, spec);
        map_ref_to_drawn(&self.last_build.crystal, core_ref)
    }

    /// Convert a drawn-canvas drag position for existing pin `idx` back to its reference-lattice `at`
    /// without introducing a jump relative to the pin's current rendered position.
    pub(crate) fn drawn_to_pin_at(&self, idx: usize, drawn_world: [f64; 2]) -> [f64; 2] {
        let ref_drawn = map_drawn_to_ref(&self.last_build.crystal, drawn_world);
        if let Some(spec) = self.specs.get(idx) {
            let cur_at = spec_world_pos(&self.info, idx, spec);
            let cur_marker_ref = map_drawn_to_ref(&self.last_build.crystal, self.marker_world_pos(idx, spec));
            [
                cur_at[0] + (ref_drawn[0] - cur_marker_ref[0]),
                cur_at[1] + (ref_drawn[1] - cur_marker_ref[1]),
            ]
        } else {
            ref_drawn
        }
    }

    /// Convert a drawn-canvas click position for the active placement tool back to reference-lattice `at`.
    pub(crate) fn drawn_to_place_at(&self, drawn_world: [f64; 2]) -> [f64; 2] {
        let core_ref = map_drawn_to_ref(&self.last_build.crystal, drawn_world);
        if self.tool_mode == ToolMode::PlaceDislocation {
            let mn = if self.custom_mn == (0, 0) { (1, 0) } else { self.custom_mn };
            let b = burgers_from_mn(&self.info, mn.0, mn.1);
            [core_ref[0] - 0.5 * b[0], core_ref[1] - 0.5 * b[1]]
        } else {
            core_ref
        }
    }

    pub(crate) fn build_markers(&self) -> Vec<CanvasMarker> {
        let core_pos = self.matched_dislocation_core_positions();
        self.specs
            .iter()
            .enumerate()
            .map(|(idx, s)| {
                let pos = core_pos.get(idx).copied().flatten().unwrap_or_else(|| {
                    map_ref_to_drawn(&self.last_build.crystal, spec_core_ref_pos(&self.info, idx, s))
                });
                CanvasMarker {
                    index: idx,
                    pos,
                    kind: match *s {
                        DefectSpec::Disclination { angle_deg, .. } if angle_deg >= 0.0 => {
                            MarkerKind::DisclinationPlus
                        }
                        DefectSpec::Disclination { .. } => MarkerKind::DisclinationMinus,
                        DefectSpec::Dislocation { .. } | DefectSpec::HalfWall { .. } => {
                            MarkerKind::Dislocation
                        }
                    },
                    label: spec_short_label(&self.info, s),
                }
            })
            .collect()
    }
}
