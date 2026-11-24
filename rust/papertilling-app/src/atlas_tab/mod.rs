//! Group and defect atlas for Euclidean and regular hyperbolic models.

use papertilling_core::space3d::nets3d::LineDefect3d;

use crate::canvas2d::{Canvas2D, CanvasInteraction};
use crate::export_obj::write_gpu_mesh_obj;
use crate::theme::ThemeTokens;
use crate::ui::viewport_pane::show_split_2d_3d;
use crate::view3d::View3dState;

mod crystal_charge_ui;
mod hyperbolic3d_ui;
mod hyperbolic_charge_ui;
mod hyperbolic_pick;
pub mod hyperbolic_scene;
pub mod hyperbolic_shells;
pub mod hyperbolic_ui;
pub mod hyperbolic_view;
pub mod layers_growth;
pub mod layers_ui;
pub mod points3d_ui;
pub mod space3d_geom;
pub mod space3d_ui;
pub mod space3d_view;
pub mod subperiodic_ui;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_group_defects;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasSubTab {
    Subperiodic,
    Layers80,
    SpaceGroups3D,
    HyperbolicH2,
    HyperbolicH3,
}

impl AtlasSubTab {
    pub const ALL: [Self; 5] = [
        Self::Subperiodic,
        Self::Layers80,
        Self::SpaceGroups3D,
        Self::HyperbolicH2,
        Self::HyperbolicH3,
    ];

    pub fn is_hyperbolic(self) -> bool {
        matches!(self, Self::HyperbolicH2 | Self::HyperbolicH3)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Subperiodic => "0D/1D Subperiodic",
            Self::Layers80 => "80 Layer Groups",
            Self::SpaceGroups3D => "230 3D Space Groups",
            Self::HyperbolicH2 => "Hyperbolic H²",
            Self::HyperbolicH3 => "Hyperbolic H³",
        }
    }

    pub fn defect_label(self) -> &'static str {
        match self {
            Self::Subperiodic => "0D/1D Subperiodic Defects",
            Self::Layers80 => "Layer-group charges",
            Self::SpaceGroups3D => "Space-group charges",
            Self::HyperbolicH2 => "H² Wedge Disclinations",
            Self::HyperbolicH3 => "H³ Charges",
        }
    }
}

pub struct AtlasTabState {
    pub pristine_only: bool,
    pub sub_tab: AtlasSubTab,
    pub last_pristine_3d_subtab: AtlasSubTab,
    pub last_hyperbolic_subtab: AtlasSubTab,
    pub last_defect_subtab: AtlasSubTab,
    seeded_initial_defects: bool,
    pub subperiodic: subperiodic_ui::SubperiodicPanelState,
    pub layers: layers_ui::LayersPanelState,
    pub space3d: space3d_ui::Space3dPanelState,
    pub hyperbolic: hyperbolic_ui::HyperbolicPanelState,
    hyperbolic3d: hyperbolic3d_ui::Hyperbolic3dPanel,
    pub dirty: bool,
}

impl AtlasTabState {
    pub const PRISTINE_3D_SUBTABS: [AtlasSubTab; 3] = [
        AtlasSubTab::Subperiodic,
        AtlasSubTab::Layers80,
        AtlasSubTab::SpaceGroups3D,
    ];
    pub const HYPERBOLIC_SUBTABS: [AtlasSubTab; 2] = [AtlasSubTab::HyperbolicH2, AtlasSubTab::HyperbolicH3];
    pub const DEFECT_SUBTABS: [AtlasSubTab; 5] = AtlasSubTab::ALL;

    pub fn new() -> Self {
        let mut subperiodic = subperiodic_ui::SubperiodicPanelState::default();
        let mut space3d = space3d_ui::Space3dPanelState::default();
        let mut hyperbolic = hyperbolic_ui::HyperbolicPanelState::default();
        subperiodic.pristine_only = true;
        space3d.pristine_only = true;
        hyperbolic.pristine_only = true;
        Self {
            pristine_only: true,
            sub_tab: AtlasSubTab::SpaceGroups3D,
            last_pristine_3d_subtab: AtlasSubTab::SpaceGroups3D,
            last_hyperbolic_subtab: AtlasSubTab::HyperbolicH2,
            // Start the Defects atlas on the rosettes and friezes.
            last_defect_subtab: AtlasSubTab::Subperiodic,
            seeded_initial_defects: false,
            subperiodic,
            layers: layers_ui::LayersPanelState::default(),
            space3d,
            hyperbolic,
            hyperbolic3d: hyperbolic3d_ui::Hyperbolic3dPanel::default(),
            dirty: true,
        }
    }

    pub fn set_pristine_only(&mut self, pristine: bool) {
        if self.pristine_only != pristine {
            self.dirty = true;
        }
        self.pristine_only = pristine;
        self.subperiodic.pristine_only = pristine;
        self.space3d.pristine_only = pristine;
        self.hyperbolic.pristine_only = pristine;
    }

    /// Enter `DefectsView::AtlasDefects`: preserve existing user defect choices, seeding initial defaults only once.
    pub fn enter_defects_mode(&mut self, from_outside: bool) {
        self.set_pristine_only(false);
        if !self.seeded_initial_defects {
            self.seeded_initial_defects = true;
            if self.space3d.selected_defect == LineDefect3d::None {
                self.space3d.selected_defect = LineDefect3d::ScrewDislocationZ;
            }
            if self.hyperbolic.wedge_delta == 0 {
                self.hyperbolic.wedge_delta = -1;
            }
        }
        if from_outside || !Self::DEFECT_SUBTABS.contains(&self.sub_tab) {
            self.sub_tab = self.last_defect_subtab;
            self.dirty = true;
        }
    }

    /// Rebuild both scenes of the active sub-view and re-fit the 2D view; clears `dirty`.
    pub fn sync_scenes(&mut self, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        self.sync_inner(canvas, view3d, true);
    }

    /// Rebuild both scenes after a view-only change (cell / axes / slab / depth cue / pick / H² pan and zoom),
    /// keeping the 2D pan and zoom; clears `dirty`.
    pub fn refresh_scenes(&mut self, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        self.sync_inner(canvas, view3d, false);
    }

    fn sync_inner(&mut self, canvas: &mut Canvas2D, view3d: &mut View3dState, reset_view: bool) {
        self.subperiodic.pristine_only = self.pristine_only;
        self.space3d.pristine_only = self.pristine_only;
        self.hyperbolic.pristine_only = self.pristine_only;
        let rot = view3d.rotation();
        let (sc2, mesh3, cap) = match self.sub_tab {
            AtlasSubTab::Subperiodic => self.subperiodic.build_scenes(),
            AtlasSubTab::Layers80 => self.layers.build_scenes(),
            AtlasSubTab::SpaceGroups3D => self.space3d.build_scenes(&rot),
            AtlasSubTab::HyperbolicH2 => self.hyperbolic.build_scenes(),
            AtlasSubTab::HyperbolicH3 => self.hyperbolic3d.build_scenes(),
        };
        canvas.set_scene(sc2, reset_view);
        view3d.set_mesh(mesh3, cap);
        self.space3d.view.changed = false;
        self.hyperbolic.view_changed = false;
        self.dirty = false;
    }

    /// Whether a view-only change of the active sub-view awaits a refresh.
    fn view_changed(&self) -> bool {
        match self.sub_tab {
            AtlasSubTab::SpaceGroups3D => self.space3d.view.changed,
            AtlasSubTab::HyperbolicH2 => self.hyperbolic.view_changed,
            _ => false,
        }
    }

    pub fn show_left_panel(
        &mut self,
        ui: &mut egui::Ui,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
        offered_tabs: &[AtlasSubTab],
    ) {
        self.subperiodic.pristine_only = self.pristine_only;
        self.space3d.pristine_only = self.pristine_only;
        self.hyperbolic.pristine_only = self.pristine_only;
        let title = if !self.pristine_only {
            "Group defect atlas"
        } else if self.sub_tab.is_hyperbolic() {
            "Hyperbolic tilings & honeycombs"
        } else {
            "0D–3D Groups & Crystal Nets"
        };
        ui.strong(title);
        if offered_tabs.len() > 1 {
            ui.horizontal_wrapped(|ui| {
                for &st in offered_tabs {
                    let lbl = if self.pristine_only { st.label() } else { st.defect_label() };
                    if ui.selectable_label(self.sub_tab == st, lbl).clicked() && self.sub_tab != st {
                        self.sub_tab = st;
                        if !self.pristine_only {
                            self.last_defect_subtab = st;
                        } else if st.is_hyperbolic() {
                            self.last_hyperbolic_subtab = st;
                        } else {
                            self.last_pristine_3d_subtab = st;
                        }
                        self.space3d.view.hovered = None;
                        self.hyperbolic.hovered = None;
                        self.dirty = true;
                    }
                }
            });
        }
        ui.separator();

        let ctrl_changed = match self.sub_tab {
            AtlasSubTab::Subperiodic => self.subperiodic.show_controls(ui),
            AtlasSubTab::Layers80 => self.layers.show_controls(ui, self.pristine_only),
            AtlasSubTab::SpaceGroups3D => self.space3d.show_controls(ui),
            AtlasSubTab::HyperbolicH2 => self.hyperbolic.show_controls(ui),
            AtlasSubTab::HyperbolicH3 => self.hyperbolic3d.show_controls(ui),
        };
        if ctrl_changed || self.dirty {
            self.sync_scenes(canvas, view3d);
        } else if self.view_changed() {
            self.refresh_scenes(canvas, view3d);
        }
    }

    pub fn show_inspector_panel(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        self.subperiodic.pristine_only = self.pristine_only;
        self.space3d.pristine_only = self.pristine_only;
        self.hyperbolic.pristine_only = self.pristine_only;
        match self.sub_tab {
            AtlasSubTab::Subperiodic => self.subperiodic.show_inspector(ui, tokens),
            AtlasSubTab::Layers80 => self.layers.show_inspector(ui, tokens, self.pristine_only),
            AtlasSubTab::SpaceGroups3D => self.space3d.show_inspector(ui, tokens),
            AtlasSubTab::HyperbolicH2 => self.hyperbolic.show_inspector(ui, tokens),
            AtlasSubTab::HyperbolicH3 => self.hyperbolic3d.show_inspector(ui, tokens, self.pristine_only),
        }
    }

    pub fn show_central(
        &mut self,
        ui: &mut egui::Ui,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
        split_ratio: &mut f32,
        tokens: &ThemeTokens,
    ) {
        if self.dirty {
            self.sync_scenes(canvas, view3d);
        }
        let left_heading = match self.sub_tab {
            AtlasSubTab::HyperbolicH2 => "Hyperbolic disk",
            AtlasSubTab::HyperbolicH3 => "Honeycomb xy projection",
            AtlasSubTab::SpaceGroups3D => "3D Net xy-Projection",
            AtlasSubTab::Layers80 => "Layer Slab xy-Projection",
            AtlasSubTab::Subperiodic => "2D Rosette / Frieze Strip",
        };
        let right_heading = match self.sub_tab {
            AtlasSubTab::HyperbolicH2 => "Hyperboloid preview",
            AtlasSubTab::HyperbolicH3 => "Poincaré ball",
            AtlasSubTab::SpaceGroups3D => "3D crystal net",
            AtlasSubTab::Layers80 => "3D Layer Group Slab",
            AtlasSubTab::Subperiodic => "3D Strip Quotient (Cylinder / Mobius) or Planar Net",
        };
        let sub_tab = self.sub_tab;
        let interactive = matches!(sub_tab, AtlasSubTab::SpaceGroups3D | AtlasSubTab::HyperbolicH2);
        let mut inter2d: Option<CanvasInteraction> = None;
        show_split_2d_3d(
            ui,
            split_ratio,
            "atlas_split",
            tokens,
            |ui| {
                ui.strong(left_heading);
                ui.add_space(2.0);
                if interactive {
                    // H²: a background drag is a hyperbolic pan of the content, not a pan of the 2D view
                    canvas.drag_reports_world = sub_tab == AtlasSubTab::HyperbolicH2;
                    inter2d = Some(canvas.show_interactive(ui, &[], true, None, tokens));
                    canvas.drag_reports_world = false;
                } else {
                    canvas.show(ui, tokens);
                }
            },
            |ui| {
                ui.strong(right_heading);
                ui.add_space(2.0);
                view3d.show(ui, tokens);
            },
        );
        self.after_pointer(ui.ctx(), inter2d.as_ref(), canvas, view3d);
    }

    /// Picks, H² drag pan and the camera-dependent depth cue after this frame's input.
    fn after_pointer(
        &mut self,
        ctx: &egui::Context,
        inter2d: Option<&CanvasInteraction>,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
    ) {
        let mut hover_changed = false;
        match self.sub_tab {
            AtlasSubTab::SpaceGroups3D => {
                let prev = self.space3d.view.hovered;
                self.space3d.handle_pointer(inter2d, canvas.zoom, view3d);
                hover_changed = self.space3d.view.hovered != prev;
                if self.space3d.needs_depth_refresh(&view3d.rotation()) {
                    self.space3d.view.changed = true;
                }
            }
            AtlasSubTab::HyperbolicH2 => {
                let prev = self.hyperbolic.hovered;
                self.hyperbolic.handle_pointer(inter2d, canvas.zoom, view3d);
                hover_changed = self.hyperbolic.hovered != prev;
            }
            _ => {}
        }
        let refreshed = self.view_changed();
        if refreshed {
            self.refresh_scenes(canvas, view3d);
        }
        if refreshed || hover_changed {
            ctx.request_repaint();
        }
    }

    pub fn export_active_obj(&self, path: &std::path::Path, view3d: &View3dState) -> Result<String, String> {
        write_gpu_mesh_obj(path, view3d.mesh_data())
    }
}
