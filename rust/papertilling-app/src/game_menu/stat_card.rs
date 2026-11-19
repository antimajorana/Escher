//! Stat card of the highlighted Command Deck card (right column of the deck): title, realm, subtitle, badge,
//! a few rows of facts the app already has, and the Equip / Open in Forge / Open in Lab buttons. Every row reads
//! existing state; nothing here computes a new property (no vertex symbols, no live mini-view: later stages).

use super::badges::{Badge, badge_widget};
use super::realms::{Card, CardId, LatticeRef, orbifold_of};
use crate::app::{DefectsView, LatticesView, PapertillingApp};
use crate::backend::{SurfaceTargetUi, VerdictKindUi};
use crate::gallery_tab::browser::rule_tag;
use crate::gallery_tab::filter::ita_number;
use crate::presets::PRESETS;
use crate::theme::ThemeTokens;

/// Side of the lattice thumbnail drawn on lattice cards.
const THUMB_PX: f32 = 120.0;

/// What a stat-card button asks the deck to do with its card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StatAction {
    Equip,
    OpenInForge,
    OpenInLab,
}

/// A non-selectable label (its text stays in the accessibility tree for the quarantine scan).
fn text(ui: &mut egui::Ui, rich: egui::RichText) {
    ui.add(egui::Label::new(rich).selectable(false).wrap());
}

/// One `key: value` row.
fn row(ui: &mut egui::Ui, t: &ThemeTokens, key: &str, value: &str) {
    ui.horizontal_wrapped(|ui| {
        ui.add(egui::Label::new(egui::RichText::new(format!("{key}:")).color(t.text_muted)).selectable(false));
        ui.add(egui::Label::new(egui::RichText::new(value).color(t.text_primary)).selectable(false).wrap());
    });
}

impl PapertillingApp {
    pub(crate) fn show_stat_card(&mut self, ui: &mut egui::Ui, card: &Card) -> Option<StatAction> {
        let t = self.theme.tokens();
        let mut action = None;
        ui.vertical(|ui| {
            text(ui, egui::RichText::new(card.title.as_str()).strong().size(15.0).color(t.text_primary));
            text(ui, egui::RichText::new(card.realm.title()).small().color(t.text_muted));
            if !card.subtitle.is_empty() {
                text(ui, egui::RichText::new(card.subtitle.as_str()).color(t.text_secondary));
            }
            if let Some(b) = card.badge {
                let _ = badge_widget(ui, b, &t);
            }
            ui.separator();
            self.stat_rows(ui, card, &t);
            ui.separator();
            if ui.button("⏎ Equip").clicked() {
                action = Some(StatAction::Equip);
            }
            if matches!(card.id, CardId::Lattice(_)) {
                if ui.button("⚒ Open in Defects Forge").clicked() {
                    action = Some(StatAction::OpenInForge);
                }
                if ui.button("ψ Open in Analysis Lab").clicked() {
                    action = Some(StatAction::OpenInLab);
                }
            }
        });
        action
    }

    fn stat_rows(&mut self, ui: &mut egui::Ui, card: &Card, t: &ThemeTokens) {
        match card.id {
            CardId::Lattice(r) => self.lattice_rows(ui, r, t),
            CardId::Surface(s) => self.surface_rows(ui, s, t),
            CardId::DefectCatalogue(s) => {
                row(ui, t, "Lattice", &self.gallery.active_item().display_with_orig());
                match self.catalogue_counts() {
                    Some(n) => row(ui, t, "Entries", &n[s as usize].to_string()),
                    None => row(ui, t, "Entries", "listed once the Forge has built this lattice"),
                }
            }
            CardId::Wb(v, i) => {
                let n = self.workbench.sub_labels(v).len();
                row(ui, t, "Row", &format!("{} of {n}", usize::from(i) + 1));
            }
            CardId::AtlasPristine(s) => row(ui, t, "Atlas sub-tab", s.label()),
            CardId::AtlasDefects(s) => row(ui, t, "Atlas sub-tab", s.defect_label()),
            CardId::Plane2D | CardId::WallpaperGroups | CardId::CustomBuilder => {
                row(ui, t, "View", LatticesView::Plane2D.label());
            }
            CardId::HyperbolicAtlas => row(ui, t, "View", LatticesView::NonEuclidean.label()),
            CardId::Combine => {
                row(ui, t, "View", DefectsView::CombineAndFuse.label());
                row(ui, t, "Lattice", &self.gallery.active_item().display_with_orig());
            }
            CardId::Analysis(_) => row(ui, t, "Lattice", &self.gallery.active_item().display_with_orig()),
            CardId::Preset(i) => {
                if let Some(p) = PRESETS.get(i) {
                    text(ui, egui::RichText::new(p.description).color(t.text_secondary));
                }
            }
            CardId::Io(_) => text(ui, egui::RichText::new("Equip opens a file dialog.").color(t.text_secondary)),
        }
    }

    /// Lattice cards: the cached 2×2 thumbnail plus the catalogue facts (group, orbifold, rule, vertex degrees).
    fn lattice_rows(&mut self, ui: &mut egui::Ui, r: LatticeRef, t: &ThemeTokens) {
        let Some(item) = self.lattice_ref_item(r).cloned() else {
            text(ui, egui::RichText::new("This lattice is no longer listed.").color(t.text_muted));
            return;
        };
        let (rect, _) = ui.allocate_exact_size(egui::vec2(THUMB_PX, THUMB_PX), egui::Sense::hover());
        if ui.is_rect_visible(rect) {
            let p = ui.painter_at(rect);
            p.rect_filled(rect, 4.0, t.bg_canvas_2d);
            if let Some(g) = self.gallery.thumbs.get_or_build(&item) {
                g.paint(&p, rect, t);
            }
        }
        let group = match ita_number(&item.group) {
            Some(n) => format!("{} (ITA {n})", item.group),
            None => item.group.clone(),
        };
        row(ui, t, "Group", &group);
        if let Some(orb) = orbifold_of(&item.group) {
            row(ui, t, "Orbifold", orb);
        }
        row(ui, t, "Rule", rule_tag(&item.alias_key));
        row(ui, t, "Vertex degrees", &item.coordination);
        row(ui, t, "Sites / cell", &item.sites_per_cell.to_string());
        row(ui, t, "Orbits", &format!("{} site, {} bond", item.site_orbits, item.bond_orbits));
        if item.chiral {
            row(ui, t, "Chirality", "chiral (±)");
        }
        if item.has_custom_name() {
            row(ui, t, "Original name", &item.default_label);
        }
        if item.alias_key == self.gallery.base_item().alias_key {
            text(ui, egui::RichText::new("Selected now").small().color(t.text_muted));
        }
    }

    /// Surface cards: the verdict of the selected lattice on this surface, worded as strongly as the evidence.
    fn surface_rows(&mut self, ui: &mut egui::Ui, s: SurfaceTargetUi, t: &ThemeTokens) {
        let item = self.gallery.active_item().clone();
        row(ui, t, "Lattice", &item.display_with_orig());
        let k = SurfaceTargetUi::ALL.iter().position(|&x| x == s);
        let cell = k.and_then(|k| self.surfaces.lattice_verdicts(&item).map(|cells| cells[k].clone()));
        let Some(cell) = cell else {
            text(ui, egui::RichText::new("No verdict for this lattice.").color(t.text_muted));
            return;
        };
        let badge = Badge::from_verdict(cell.verdict);
        let colour = badge.colour(t);
        match cell.verdict {
            VerdictKindUi::Ok => {
                text(ui, egui::RichText::new("Status: [PROVED OK] explicit cell built").color(colour));
                if let Some(n) = cell.sites {
                    row(ui, t, "Sites", &n.to_string());
                }
            }
            VerdictKindUi::Impossible => {
                let status = match cell.scope.as_deref() {
                    Some(sc) => format!("Status: [PROVED] impossible (scope: {sc})"),
                    None => "Status: [PROVED] impossible".to_owned(),
                };
                text(ui, egui::RichText::new(status).color(colour));
            }
            VerdictKindUi::NotFound => {
                text(ui, egui::RichText::new("Status: bounded search, no construction found").color(colour));
            }
        }
        if !cell.cones_summary.is_empty() {
            row(ui, t, "Cones", &cell.cones_summary);
        }
        if !cell.reason.is_empty() {
            text(ui, egui::RichText::new(cell.reason.as_str()).small().color(t.text_secondary));
        }
    }
}
