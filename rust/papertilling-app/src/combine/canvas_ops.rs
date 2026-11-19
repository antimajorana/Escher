use super::{CombineTabState, MotionMode, ToolMode};
use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::Canvas2D;
use crate::core_panel::show_core_panel;
use crate::live_combine::cuts::set_pin_cut_angle;
use crate::mesh3d::DefectViewMode;
use crate::theme::ThemeTokens;
use crate::ui::viewport_pane::{show_defect_viewport_header, show_split_2d_3d};
use crate::view3d::View3dState;

impl CombineTabState {
    #[allow(clippy::too_many_arguments)]
    pub fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        gallery_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
        split_ratio: &mut f32,
        tokens: &ThemeTokens,
    ) {
        let collapsed = self.core_panel.collapsed;
        let drawer = egui::Panel::bottom("combine_core_drawer")
            .resizable(!collapsed)
            .default_size(if collapsed { 28.0 } else { 236.0 })
            .size_range(if collapsed { 24.0..=38.0 } else { 140.0..=360.0 });
        drawer.show(ui, |ui| {
            if self.compare_stale && self.compare_result.is_some() {
                ui.small("Showing the previous comparison while the new one runs.");
            }
            if collapsed {
                show_core_panel(&mut self.core_panel, ui, self.compare_result.as_ref(), tokens);
            } else {
                egui::ScrollArea::vertical().id_salt("core_drawer_scroll").show(ui, |ui| {
                    show_core_panel(&mut self.core_panel, ui, self.compare_result.as_ref(), tokens);
                });
            }
        });

        let busy = self.is_building();
        let (kc, vc) = show_defect_viewport_header(
            ui,
            &mut self.view_mode,
            &mut self.sheet_opts,
            busy,
            &mut self.view3d_dirty,
            "comb_top_col",
        );
        if vc {
            self.on_combine_sheet_color_changed(ui.ctx());
        }
        if kc {
            self.request_sheet3d_only(gallery_item);
        }
        self.poll(Some(ui.ctx()), gallery_item, Some(canvas));
        if self.view3d_dirty && self.view_mode.uses_3d() {
            self.sync_view3d(gallery_item, view3d);
        }
        ui.add_space(2.0);

        match self.view_mode {
            DefectViewMode::Planar2D => {
                self.show_2d_pin_editor(ui, gallery_item, backend, canvas, tokens);
            }
            DefectViewMode::Sheet3D => {
                ui.strong("3D Live Elastic Sheet (orbit, scroll/pinch zoom; Clear All = flat z=0 plane)");
                ui.add_space(2.0);
                view3d.show(ui, tokens);
            }
            DefectViewMode::Split2D3D => {
                show_split_2d_3d(
                    ui,
                    split_ratio,
                    "combine_split",
                    tokens,
                    |ui| {
                        self.show_2d_pin_editor(ui, gallery_item, backend, canvas, tokens);
                    },
                    |ui| {
                        ui.strong("Live 3D Elastic Sheet (buckling & screening)");
                        ui.add_space(2.0);
                        view3d.show(ui, tokens);
                    },
                );
            }
        }
    }

    pub(super) fn show_2d_pin_editor(
        &mut self,
        ui: &mut egui::Ui,
        gallery_item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        tokens: &ThemeTokens,
    ) {
        if self.selected_pin.is_some_and(|i| i >= self.specs.len()) {
            self.clear_selection();
            self.sync_selection_to_canvas(canvas);
        }
        let header = match self.motion_mode {
            MotionMode::Conserved => "2D Pin Editor [Conserved ΔN=0: dislocations glide along b | Shift+drag disclination: emit b]",
            MotionMode::StepDisclinationEmit => "2D Pin Editor [Step D: dragging disclination emits b | dislocations glide along b]",
            MotionMode::FreeSetup => "2D Pin Editor [Free Setup: unconstrained drag | drag empty: pan | scroll/pinch: zoom]",
        };
        ui.strong(header);
        ui.add_space(2.0);
        let click_to_place = self.tool_mode != ToolMode::PanAndDrag && !self.is_initializing() && self.init_error.is_none();
        let preview_world = match (self.tool_mode, self.last_hover_world) {
            (ToolMode::PlaceDisclination, Some(hw)) if !self.info.centres.is_empty() => {
                let (cid, ang) = self.active_centre_and_angle();
                Some(backend.snap_disclination(self.effective_item(gallery_item), &self.info, ang, cid, hw).0)
            }
            (ToolMode::PlaceDislocation, Some(hw)) | (ToolMode::PlaceHalfWall, Some(hw)) => Some(hw),
            _ => None,
        };
        let markers = self.build_markers();
        let inter = canvas.show_interactive(ui, &markers, click_to_place, preview_world, tokens);
        self.last_hover_world = inter.hovered_world;
        if let Some((pin_idx, world)) = inter.dragged_marker {
            self.commit_cut_edit_anchor();
            self.move_pin_with_mods(pin_idx, world, inter.shift_held, gallery_item, backend, canvas);
        }
        let released_elsewhere = self.dragging && !ui.input(|i| i.pointer.any_down());
        if inter.drag_released || released_elsewhere {
            self.finish_drag_kinematics(gallery_item, Some(canvas));
        }
        if let Some((idx, deg)) = inter.dragged_cut_ray
            && !self.lattice_unavailable()
            && !self.dragging
            && idx < self.specs.len()
        {
            self.commit_cut_edit_anchor();
            self.cut_drag_anchor.get_or_insert_with(|| (idx, self.specs.clone()));
            self.cut_dragging = true;
            set_pin_cut_angle(&mut self.specs[idx], Some(deg));
            canvas.set_cut_rays(&self.current_cut_rays());
            self.request_build(gallery_item, true, false, false);
        }
        let cut_released_elsewhere = self.cut_dragging && !ui.input(|i| i.pointer.any_down());
        if inter.cut_drag_released || cut_released_elsewhere {
            self.finish_cut_drag(gallery_item, Some(canvas));
        }
        if let Some(idx) = inter.clicked_marker
            && idx < self.specs.len()
        {
            self.commit_cut_edit_anchor();
            self.selected_pin = Some(idx);
            self.popover_open = true;
            self.sync_selection_to_canvas(canvas);
        } else if inter.clicked_empty
            && self.tool_mode == ToolMode::PanAndDrag
            && (self.selected_pin.is_some() || self.popover_open)
        {
            self.commit_cut_edit_anchor();
            self.clear_selection();
            self.sync_selection_to_canvas(canvas);
        }
        if let Some(world) = inter.clicked_world {
            self.place_at_world(world, gallery_item, backend, canvas);
        }
        if let Some(v0) = inter.clicked_circuit_vertex {
            self.inspectors.circuit.base_v0 = Some(v0);
            self.on_combine_circuit_changed(gallery_item, canvas);
        }
        self.poll(Some(ui.ctx()), gallery_item, Some(canvas));
        self.show_pin_popover(ui, gallery_item, canvas, tokens);
    }
}
