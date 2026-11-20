//! All 230 3D Space Groups (`73 symmorphic + 157 nonsymmorphic`) and canonical 3D crystal nets.
//!
//! C1: the line-defect picker (screw, wedge, dispiration) is hidden and every net is drawn pristine, because the
//! current construction is geometrically wrong (off-centre block, a wedge that removes nothing, zero-length bonds,
//! a seam that is never re-glued). The selection and the core code stay until a real cut-and-reglue exists.

use papertilling_core::bieberbach::BieberbachSummary;
use papertilling_core::space3d::nets3d::{CanonicalNet3d, LineDefect3d};
use papertilling_core::space3d::{analyze_space_group_3d, list_space_groups_3d};

use super::space3d_view::Space3dView;
use crate::mesh3d::GpuMeshData;
use crate::scene2d::Scene2D;
use crate::theme::ThemeTokens;

pub struct Space3dPanelState {
    pub pristine_only: bool,
    pub filter: String,
    pub selected_sg: u16,
    pub selected_net: CanonicalNet3d,
    pub selected_defect: LineDefect3d,
    pub summary: Option<BieberbachSummary>,
    /// Cell / axes / slab / depth-cue options and the picked site.
    pub view: Space3dView,
    cached_sg: u16,
}

impl Default for Space3dPanelState {
    fn default() -> Self {
        Self {
            pristine_only: false,
            filter: String::new(),
            selected_sg: 136,
            selected_net: CanonicalNet3d::Rutile136,
            selected_defect: LineDefect3d::ScrewDispiration42_136,
            summary: analyze_space_group_3d(136).ok(),
            view: Space3dView::default(),
            cached_sg: 136,
        }
    }
}

impl Space3dPanelState {
    /// The line defect drawn: always none while the 3D line defects are hidden (C1; `selected_defect` is kept).
    pub fn effective_defect(&self) -> LineDefect3d {
        LineDefect3d::None
    }

    pub fn ensure_cached(&mut self) {
        if self.cached_sg != self.selected_sg || self.summary.is_none() {
            self.cached_sg = self.selected_sg;
            self.summary = analyze_space_group_3d(self.selected_sg).ok();
        }
    }

    pub fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        self.ensure_cached();
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("#123 P4/mmm").clicked() {
                self.selected_sg = 123;
                self.selected_net = CanonicalNet3d::Tetragonal123;
                if !self.pristine_only {
                    self.selected_defect = LineDefect3d::PureDisclination90_123;
                }
                dirty = true;
            }
            if ui.small_button("#136 P4_2/mnm (Rutile)").clicked() {
                self.selected_sg = 136;
                self.selected_net = CanonicalNet3d::Rutile136;
                if !self.pristine_only {
                    self.selected_defect = LineDefect3d::ScrewDispiration42_136;
                }
                dirty = true;
            }
        });
        ui.horizontal(|ui| {
            ui.small("Filter (230 Space Groups):");
            ui.add(egui::TextEdit::singleline(&mut self.filter).desired_width(120.0));
        });
        let q = self.filter.trim().to_lowercase();
        egui::ScrollArea::vertical().id_salt("atlas_sg230_scroll").max_height(155.0).show(ui, |ui| {
            for sg in list_space_groups_3d() {
                if !q.is_empty()
                    && !sg.symbol.to_lowercase().contains(&q)
                    && !sg.schoenflies.to_lowercase().contains(&q)
                    && !sg.number.to_string().contains(&q)
                {
                    continue;
                }
                let tag = if sg.is_split { "split" } else { "nonsym" };
                let lbl = format!("#{:>3} {} [{}] ({tag}, {})", sg.number, sg.symbol, sg.schoenflies, sg.ab_full_label);
                if ui.selectable_label(self.selected_sg == sg.number, lbl).clicked() {
                    self.selected_sg = sg.number;
                    dirty = true;
                }
            }
        });

        ui.separator();
        ui.strong("3D Crystal Net");
        egui::ComboBox::from_id_salt("atlas_3d_net")
            .selected_text(self.selected_net.label())
            .show_ui(ui, |ui| {
                for n in CanonicalNet3d::ALL {
                    if ui.selectable_value(&mut self.selected_net, n, n.label()).changed() {
                        self.set_catalogue(None); // C1 WP-U4: a canonical net replaces the catalogue entry
                        dirty = true;
                    }
                }
            });
        if !self.pristine_only {
            ui.small("3D line defects are hidden until a real cut-and-reglue exists; the net is drawn pristine.");
        }
        dirty |= super::points3d_ui::show(ui, self);
        self.show_view_controls(ui);
        dirty
    }

    pub fn show_inspector(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        self.ensure_cached();
        ui.strong("3D Space Group Cohomology");
        if let Some(s) = &self.summary {
            let (st, col) = if s.is_split {
                ("split ([ν]=0)", tokens.status_ok)
            } else {
                ("nonsymmorphic ([ν]≠0)", tokens.status_warn)
            };
            ui.horizontal_wrapped(|ui| {
                ui.strong(format!("#{} {} | |P|={} | ", s.number, s.symbol, s.point_order));
                ui.colored_label(col, egui::RichText::new(st).strong());
            });
            ui.small(format!(
                "H²(P, Λ) = {} | (G⁺)ᵃᵇ = {} | Gᵃᵇ = {}",
                s.h2_label, s.ab_plus_label, s.ab_full_label
            ));
            if !s.torus_counts.is_empty() {
                let tor: Vec<String> = s.torus_counts.iter().map(|(n, c)| format!("N={n}: {c}")).collect();
                ui.small(format!("#Conj(G / NΛ): {}", tor.join(", ")));
            }
        }
        ui.separator();
        ui.strong("Active 3D Crystal Net");
        self.show_net_line(ui);
        if !self.pristine_only {
            ui.small("Line defect: none (3D line defects are hidden)");
        }
        self.show_pick_readout(ui, tokens);
    }

    /// Scenes of the current net with the view options (`rot`: the 3D camera rotation, for the depth cue).
    pub fn build_scenes(&mut self, rot: &[[f32; 4]; 4]) -> (Scene2D, GpuMeshData, String) {
        self.ensure_cached();
        self.build_view_scenes(rot)
    }
}
