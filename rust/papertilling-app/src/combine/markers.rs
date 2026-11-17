use papertilling_core::defect::surgery::DefectSpec;

use super::{CombineTabState, ToolMode};
use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::{Canvas2D, CanvasCutRay, CanvasMarker, MarkerKind, Scene2D};
use crate::live_combine::{
    burgers_from_mn, cuts::build_cut_rays, preset_neighbour_pair, preset_shortest_dislocation,
    spec_short_label, spec_world_pos,
};

impl CombineTabState {
    pub fn current_cut_rays(&self) -> Vec<CanvasCutRay> {
        if self.last_build.refused.is_some() {
            return Vec::new();
        }
        let cuts: Vec<_> = if self.show_branch_cuts {
            self.last_build.applied_cuts.clone()
        } else {
            self.last_build
                .applied_cuts
                .iter()
                .copied()
                .filter(|ac| ac.is_half_wall)
                .collect()
        };
        build_cut_rays(
            &cuts,
            self.last_build.patch_centre,
            self.last_build.applied_radius,
        )
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

    pub(crate) fn build_markers(&self) -> Vec<CanvasMarker> {
        self.specs
            .iter()
            .enumerate()
            .map(|(idx, s)| CanvasMarker {
                index: idx,
                pos: spec_world_pos(&self.info, idx, s),
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
            })
            .collect()
    }
}
