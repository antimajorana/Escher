//! 5th Desktop App Tab: Complete Atlas (0D–3D & Hyperbolic H²).

use crate::canvas2d::Canvas2D;
use crate::export_obj::write_gpu_mesh_obj;
use crate::theme::ThemeTokens;
use crate::ui::viewport_pane::show_split_2d_3d;
use crate::view3d::View3dState;

pub mod hyperbolic_ui;
pub mod layers_ui;
pub mod space3d_ui;
pub mod subperiodic_ui;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtlasSubTab {
    Subperiodic,
    Layers80,
    SpaceGroups3D,
    HyperbolicH2,
}

impl AtlasSubTab {
    pub const ALL: [Self; 4] = [
        Self::Subperiodic,
        Self::Layers80,
        Self::SpaceGroups3D,
        Self::HyperbolicH2,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Subperiodic => "0D/1D Subperiodic",
            Self::Layers80 => "80 Layer Groups",
            Self::SpaceGroups3D => "230 3D Space Groups",
            Self::HyperbolicH2 => "Hyperbolic H²",
        }
    }
}

pub struct AtlasTabState {
    pub sub_tab: AtlasSubTab,
    pub subperiodic: subperiodic_ui::SubperiodicPanelState,
    pub layers: layers_ui::LayersPanelState,
    pub space3d: space3d_ui::Space3dPanelState,
    pub hyperbolic: hyperbolic_ui::HyperbolicPanelState,
    pub dirty: bool,
}

impl AtlasTabState {
    pub fn new() -> Self {
        Self {
            sub_tab: AtlasSubTab::SpaceGroups3D,
            subperiodic: subperiodic_ui::SubperiodicPanelState::default(),
            layers: layers_ui::LayersPanelState::default(),
            space3d: space3d_ui::Space3dPanelState::default(),
            hyperbolic: hyperbolic_ui::HyperbolicPanelState::default(),
            dirty: true,
        }
    }

    pub fn sync_scenes(&mut self, canvas: &mut Canvas2D, view3d: &mut View3dState) {
        let (sc2, mesh3, cap) = match self.sub_tab {
            AtlasSubTab::Subperiodic => self.subperiodic.build_scenes(),
            AtlasSubTab::Layers80 => self.layers.build_scenes(),
            AtlasSubTab::SpaceGroups3D => self.space3d.build_scenes(),
            AtlasSubTab::HyperbolicH2 => self.hyperbolic.build_scenes(),
        };
        canvas.set_scene(sc2, true);
        view3d.set_mesh(mesh3, cap);
        self.dirty = false;
    }

    pub fn show_left_panel(
        &mut self,
        ui: &mut egui::Ui,
        canvas: &mut Canvas2D,
        view3d: &mut View3dState,
    ) {
        ui.strong("Generalized Group & Lattice Atlas (0D–3D & H²)");
        ui.horizontal_wrapped(|ui| {
            for st in AtlasSubTab::ALL {
                if ui.selectable_label(self.sub_tab == st, st.label()).clicked() && self.sub_tab != st {
                    self.sub_tab = st;
                    self.dirty = true;
                }
            }
        });
        ui.separator();

        let ctrl_changed = match self.sub_tab {
            AtlasSubTab::Subperiodic => self.subperiodic.show_controls(ui),
            AtlasSubTab::Layers80 => self.layers.show_controls(ui),
            AtlasSubTab::SpaceGroups3D => self.space3d.show_controls(ui),
            AtlasSubTab::HyperbolicH2 => self.hyperbolic.show_controls(ui),
        };
        if ctrl_changed || self.dirty {
            self.sync_scenes(canvas, view3d);
        }
    }

    pub fn show_inspector_panel(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        match self.sub_tab {
            AtlasSubTab::Subperiodic => self.subperiodic.show_inspector(ui, tokens),
            AtlasSubTab::Layers80 => self.layers.show_inspector(ui, tokens),
            AtlasSubTab::SpaceGroups3D => self.space3d.show_inspector(ui, tokens),
            AtlasSubTab::HyperbolicH2 => self.hyperbolic.show_inspector(ui, tokens),
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
            AtlasSubTab::HyperbolicH2 => "2D Hyperbolic Disk (Poincare Geodesics / Klein Chords)",
            AtlasSubTab::SpaceGroups3D => "3D Net xy-Projection",
            AtlasSubTab::Layers80 => "Layer Slab xy-Projection",
            AtlasSubTab::Subperiodic => "2D Rosette / Frieze Strip",
        };
        let right_heading = match self.sub_tab {
            AtlasSubTab::HyperbolicH2 => "3D Minkowski Hyperboloid Bowl (t = sqrt(1 + x^2 + y^2))",
            AtlasSubTab::SpaceGroups3D => "3D Crystal Net & Defect Axis (drag orbit, scroll zoom)",
            AtlasSubTab::Layers80 => "3D Layer Group Slab",
            AtlasSubTab::Subperiodic => "3D Strip Quotient (Cylinder / Mobius) or Planar Net",
        };
        show_split_2d_3d(
            ui,
            split_ratio,
            "atlas_split",
            tokens,
            |ui| {
                ui.strong(left_heading);
                ui.add_space(2.0);
                canvas.show(ui, tokens);
            },
            |ui| {
                ui.strong(right_heading);
                ui.add_space(2.0);
                view3d.show(ui, tokens);
            },
        );
    }

    pub fn export_active_obj(&self, path: &std::path::Path, view3d: &View3dState) -> Result<String, String> {
        write_gpu_mesh_obj(path, view3d.mesh_data())
    }
}
