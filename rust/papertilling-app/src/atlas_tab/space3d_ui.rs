//! All 230 3D Space Groups (`73 symmorphic + 157 nonsymmorphic`) and canonical 3D crystal nets.
//!
//! The charge lab uses exact group operations. Screw previews are built from periodic-net bond addresses.
//! The original wedge and dispiration drawings stay disabled until their seams can be rejoined.

use papertilling_core::bieberbach::BieberbachSummary;
use papertilling_core::growth::SeedKind;
use papertilling_core::space3d::nets3d::{CanonicalNet3d, LineDefect3d};
use papertilling_core::space3d::{analyze_space_group_3d, list_space_groups_3d};

use super::crystal_charge_ui::{CrystalChargeUi, CrystalGroup};
use super::points3d_ui::CatalogueChoice;
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
    pub(super) screw_periods: i32,
    pub summary: Option<BieberbachSummary>,
    /// Cell / axes / slab / depth-cue options and the picked site.
    pub view: Space3dView,
    cached_sg: u16,
    charges: CrystalChargeUi,
}

impl Default for Space3dPanelState {
    fn default() -> Self {
        Self {
            pristine_only: false,
            filter: String::new(),
            selected_sg: 136,
            selected_net: CanonicalNet3d::Rutile136,
            selected_defect: LineDefect3d::ScrewDispiration42_136,
            screw_periods: 0,
            summary: analyze_space_group_3d(136).ok(),
            view: Space3dView::default(),
            cached_sg: 136,
            charges: CrystalChargeUi::default(),
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

    pub(crate) fn select_group(&mut self, number: u16) {
        self.selected_sg = number;
        self.set_catalogue(None);
        self.view.catalogue_error = None;
        if let Some(net) = CanonicalNet3d::ALL.into_iter().find(|net| net.space_group_num() == number) {
            self.selected_net = net;
        } else {
            self.choose_catalogue(CatalogueChoice { sg: number, idx: 0, seed: SeedKind::Cell, layers: 1 });
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
                self.set_catalogue(None);
                self.view.catalogue_error = None;
                dirty = true;
            }
            if ui.small_button("#136 P4_2/mnm (Rutile)").clicked() {
                self.selected_sg = 136;
                self.selected_net = CanonicalNet3d::Rutile136;
                if !self.pristine_only {
                    self.selected_defect = LineDefect3d::ScrewDispiration42_136;
                }
                self.set_catalogue(None);
                self.view.catalogue_error = None;
                dirty = true;
            }
        });
        if !self.pristine_only {
            ui.horizontal_wrapped(|ui| {
                for (number, label) in [(4, "P2₁ screw"), (6, "Pm mirror"), (7, "Pc glide")] {
                    if ui.small_button(label).clicked() {
                        self.select_group(number);
                        dirty = true;
                    }
                }
            });
        }
        ui.horizontal(|ui| {
            ui.small("Filter (230 Space Groups):");
            ui.add(egui::TextEdit::singleline(&mut self.filter).desired_width(120.0));
        });
        let q = self.filter.trim().to_lowercase();
        egui::ScrollArea::vertical().id_salt("atlas_sg230_scroll").max_height(120.0).show(ui, |ui| {
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
                if ui.selectable_label(self.selected_sg == sg.number, lbl).clicked() && self.selected_sg != sg.number {
                    self.select_group(sg.number);
                    dirty = true;
                }
            }
        });

        ui.separator();
        ui.strong("3D Crystal Net");
        let selected_text = if self.view.catalogue.is_some() {
            "Catalogue entry (pick canonical…)"
        } else {
            self.selected_net.label()
        };
        egui::ComboBox::from_id_salt("atlas_3d_net")
            .selected_text(selected_text)
            .show_ui(ui, |ui| {
                for n in CanonicalNet3d::ALL {
                    if ui.selectable_label(self.view.catalogue.is_none() && self.selected_net == n, n.label()).clicked() {
                        self.selected_sg = n.space_group_num();
                        self.selected_net = n;
                        self.set_catalogue(None);
                        self.view.catalogue_error = None;
                        dirty = true;
                    }
                }
            });
        if !self.pristine_only {
            ui.separator();
            ui.strong("Screw dislocation");
            ui.horizontal_wrapped(|ui| {
                dirty |= ui.selectable_value(&mut self.screw_periods, 0, "Reference").changed();
                for n in [-2, -1, 1, 2] {
                    dirty |= ui.selectable_value(&mut self.screw_periods, n, format!("{n:+} a₃")).changed();
                }
            });
            if self.screw_periods != 0 {
                ui.small("A finite cut-and-reglued block of the sampled net. Orange bonds cross the seam.");
            }
        }
        self.show_view_controls(ui);
        dirty |= super::points3d_ui::show(ui, self);
        dirty
    }

    pub fn show_inspector(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        self.ensure_cached();
        if !self.pristine_only {
            self.show_screw_readout(ui);
            self.charges.show(ui, CrystalGroup::Space(self.selected_sg));
            ui.separator();
        }
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
        self.show_pick_readout(ui, tokens);
    }

    /// Scenes of the current net with the view options (`rot`: the 3D camera rotation, for the depth cue).
    pub fn build_scenes(&mut self, rot: &[[f32; 4]; 4]) -> (Scene2D, GpuMeshData, String) {
        self.ensure_cached();
        self.build_view_scenes(rot)
    }
}
