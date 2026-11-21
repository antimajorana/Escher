use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use papertilling_core::defect::combine_types::{CoreGalleryEntry, EquivalentSingleReport};

use crate::canvas2d::to_color32;
use crate::scene2d::ScenePrimitives;
use crate::theme::ThemeTokens;

mod card;
mod inspector;
#[cfg(test)]
mod tests;

pub(crate) use card::{compile_core_card, paint_card_primitives};
pub(crate) use inspector::CoreInspectorModal;

pub const CHORD_RGBA: [u8; 4] = [0xd8, 0x1b, 0x60, 255]; // #d81b60 magenta
pub const UNDER_SITE_RGBA: [u8; 4] = [0xc6, 0x28, 0x28, 255]; // #c62828 red
pub const ODD_FACE_RGBA: [u8; 4] = [0xf2, 0xa3, 0x3a, 195]; // #f2a33a warm orange
pub const SPLIT_FACE_RGBA: [u8; 4] = [0xf7, 0xd9, 0xa8, 195]; // #f7d9a8 light amber
pub const CONFIG_FACE_RGBA: [u8; 4] = [0x8f, 0xb8, 0xe8, 195]; // #8fb8e8 blue

/// Completion cards shown when a core matches (Python `config.COMPLETION_PANELS`).
pub const COMPLETION_PANELS: usize = 5;
/// Gallery cards shown at most when nothing matches (Python `plot_core_gallery(max_panels=30)`).
pub const MAX_PANELS: usize = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CoreFilterMode {
    #[default]
    All,
    VolterraOnly,
    MatchingOnly,
}

impl CoreFilterMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::VolterraOnly => "Volterra only",
            Self::MatchingOnly => "Matching",
        }
    }
}

#[derive(Default)]
pub struct CorePanelState {
    pub collapsed: bool,
    pub show_all_completions: bool,
    pub filter: CoreFilterMode,
    cards: CardCache,
    inspector: Option<CoreInspectorModal>,
}

impl CorePanelState {
    pub fn has_open_modal(&self) -> bool {
        self.inspector.is_some()
    }
}

/// Compiled card primitives of the report on display (compiled lazily, only for cards scrolled into view).
#[derive(Default)]
struct CardCache {
    report_key: u64,
    cards: HashMap<String, ScenePrimitives>,
}

impl CardCache {
    fn sync(&mut self, report: &EquivalentSingleReport) {
        let key = report_fingerprint(report);
        if key != self.report_key {
            self.report_key = key;
            self.cards.clear();
        }
    }

    fn get(&mut self, entry: &CoreGalleryEntry) -> &ScenePrimitives {
        self.cards.entry(entry.name.clone()).or_insert_with(|| compile_core_card(entry))
    }
}

fn report_fingerprint(report: &EquivalentSingleReport) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    report.verdict.hash(&mut h);
    report.charge.label.hash(&mut h);
    for e in report.config_core.iter().chain(&report.gallery) {
        e.name.hash(&mut h);
        e.kind.hash(&mut h);
        e.description.hash(&mut h);
        e.crystal.edges.hash(&mut h);
        e.chords.hash(&mut h);
        e.faces.hash(&mut h);
        e.core_sites.hash(&mut h);
        e.under_sites.hash(&mut h);
        e.perfect_rings.hash(&mut h);
        for p in &e.crystal.pos {
            (p[0].to_bits(), p[1].to_bits()).hash(&mut h);
        }
        for edge in e.crystal.adj.iter().flatten() {
            edge.kind.hash(&mut h);
            edge.ghost_stub.map(|p| [p[0].to_bits(), p[1].to_bits()]).hash(&mut h);
        }
    }
    h.finish().max(1)
}

/// The cards to draw, following Python's `config._show_gallery`: every Volterra core; completion cores with the
/// matching ones first, `max(matches, COMPLETION_PANELS)` of them when something matches (so a match is never
/// hidden), otherwise all of them up to `MAX_PANELS` cards in all. `show_all` lifts the caps.
pub(crate) struct CardSelection<'a> {
    pub volterra: Vec<&'a CoreGalleryEntry>,
    pub completions: Vec<&'a CoreGalleryEntry>,
    pub hidden: usize,
}

pub(crate) fn select_cards(report: &EquivalentSingleReport, show_all: bool) -> CardSelection<'_> {
    let match_set: HashSet<&str> = report.matches.iter().map(String::as_str).collect();
    let volterra: Vec<&CoreGalleryEntry> = report.gallery.iter().filter(|e| e.kind == "Volterra").collect();
    let (hits, rest): (Vec<&CoreGalleryEntry>, Vec<&CoreGalleryEntry>) =
        report.gallery.iter().filter(|e| e.kind != "Volterra").partition(|e| match_set.contains(e.name.as_str()));
    let total = hits.len() + rest.len();
    let cap = if show_all {
        total
    } else if !report.matches.is_empty() {
        hits.len().max(COMPLETION_PANELS)
    } else {
        MAX_PANELS.saturating_sub(volterra.len())
    };
    let completions: Vec<&CoreGalleryEntry> = hits.into_iter().chain(rest).take(cap).collect();
    CardSelection { hidden: total - completions.len(), volterra, completions }
}

pub fn show_core_panel(
    state: &mut CorePanelState,
    ui: &mut egui::Ui,
    compare_res: Option<&Result<EquivalentSingleReport, String>>,
    tokens: &ThemeTokens,
) {
    ui.horizontal_wrapped(|ui| {
        ui.strong("Core comparison");
        if ui.small_button(if state.collapsed { "Expand ▲" } else { "Collapse ▼" }).clicked() {
            state.collapsed = !state.collapsed;
        }
        if !state.collapsed {
            ui.separator();
            legend_chip(ui, to_color32(CONFIG_FACE_RGBA), "Config core");
            legend_chip(ui, to_color32(ODD_FACE_RGBA), "Odd ring");
            legend_chip(ui, to_color32(SPLIT_FACE_RGBA), "Perfect ring");
            legend_chip(ui, to_color32(CHORD_RGBA), "Chord (-- #d81b60)");
            legend_chip(ui, to_color32(UNDER_SITE_RGBA), "Under-coord (○)");
            legend_chip(ui, tokens.status_ok, "MATCH");
            ui.separator();
            ui.small("Show:");
            for fm in [CoreFilterMode::All, CoreFilterMode::VolterraOnly, CoreFilterMode::MatchingOnly] {
                ui.selectable_value(&mut state.filter, fm, fm.label());
            }
        }
    });

    if state.collapsed {
        return;
    }
    ui.add_space(3.0);

    let Some(res) = compare_res else {
        ui.small(
            "Click 'Compare with Single Defect' (or enable Auto-compare) to view Configuration, Volterra, and Completion cores side by side.",
        );
        return;
    };
    let report = match res {
        Ok(rep) => rep,
        Err(err) => {
            ui.colored_label(tokens.status_warn, format!("Core gallery status: {err}"));
            return;
        }
    };

    state.cards.sync(report);
    let match_set: HashSet<&str> = report.matches.iter().map(String::as_str).collect();
    let sel = select_cards(report, state.show_all_completions);

    ui.horizontal(|ui| {
        let v_col = if report.equivalent { tokens.status_ok } else { tokens.status_warn };
        ui.colored_label(v_col, egui::RichText::new(&report.verdict).strong());
        if sel.hidden > 0 || state.show_all_completions {
            let n_comp = sel.completions.len() + sel.hidden;
            ui.checkbox(&mut state.show_all_completions, format!("Show all {n_comp} completion cores ({} hidden)", sel.hidden));
        }
    });

    let cache = &mut state.cards;
    let mut clicked_card: Option<(&CoreGalleryEntry, &'static str, bool)> = None;
    egui::ScrollArea::horizontal().id_salt("core_gallery_scroll").show(ui, |ui| {
        ui.horizontal(|ui| {
            if let Some(cfg_entry) = &report.config_core {
                if draw_core_card(ui, cache, cfg_entry, "Configuration Core", false, tokens) {
                    clicked_card = Some((cfg_entry, "Configuration Core", false));
                }
                if !sel.volterra.is_empty() || !sel.completions.is_empty() {
                    ui.separator();
                }
            }
            for entry in &sel.volterra {
                let m = match_set.contains(entry.name.as_str());
                if state.filter == CoreFilterMode::MatchingOnly && !m {
                    continue;
                }
                if draw_core_card(ui, cache, entry, "Volterra", m, tokens) {
                    clicked_card = Some((entry, "Volterra", m));
                }
            }
            if state.filter != CoreFilterMode::VolterraOnly {
                if !sel.volterra.is_empty() && !sel.completions.is_empty() {
                    ui.separator();
                }
                for entry in &sel.completions {
                    let m = match_set.contains(entry.name.as_str());
                    if state.filter == CoreFilterMode::MatchingOnly && !m {
                        continue;
                    }
                    if draw_core_card(ui, cache, entry, "Completion", m, tokens) {
                        clicked_card = Some((entry, "Completion", m));
                    }
                }
            }
            if report.config_core.is_none() && report.gallery.is_empty() {
                ui.label(format!(
                    "Charge kind '{}': {} candidate(s) checked ({})",
                    report.kind,
                    report.candidates.len(),
                    if report.matches.is_empty() { "no match" } else { "matched" }
                ));
            }
        });
    });

    if let Some((entry, gtag, is_m)) = clicked_card {
        state.inspector = Some(CoreInspectorModal::from_entry(entry, gtag, is_m));
    }
    if let Some(mut modal) = state.inspector.take()
        && modal.show(ui.ctx(), tokens)
    {
        state.inspector = Some(modal);
    }
}

fn legend_chip(ui: &mut egui::Ui, color: egui::Color32, text: &str) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
    ui.small(text);
}

fn draw_core_card(
    ui: &mut egui::Ui,
    cache: &mut CardCache,
    entry: &CoreGalleryEntry,
    group_tag: &str,
    is_match: bool,
    tokens: &ThemeTokens,
) -> bool {
    let card_w = 184.0_f32;
    let card_h = 204.0_f32;
    let (card_rect, resp) = ui.allocate_exact_size(egui::vec2(card_w, card_h), egui::Sense::click());
    if !ui.is_rect_visible(card_rect) {
        return false; // scrolled out of view: neither compiled nor painted
    }
    let clicked = resp.clicked();
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    let mut tip = format!("{} ({})\n{}\n[Click to enlarge & zoom]", entry.name, group_tag, entry.description);
    if !entry.also.is_empty() {
        tip.push_str(&format!("\nAlso reached by: {}", entry.also.join(", ")));
    }
    if !entry.faces_ok || entry.iso_ok == Some(false) {
        tip.push_str("\n[picture check: completion faces or core graph of the drawn crop differ from the scan]");
    }
    resp.on_hover_text(tip);

    let bg = if is_match { tokens.bg_card_selected } else { tokens.bg_card };
    let painter = ui.painter_at(card_rect);
    painter.rect_filled(card_rect, 5.0, bg);

    painter.text(
        card_rect.left_top() + egui::vec2(7.0, 5.0),
        egui::Align2::LEFT_TOP,
        &entry.name,
        egui::FontId::proportional(11.0),
        tokens.text_primary,
    );
    if is_match {
        let badge = egui::Rect::from_min_size(egui::pos2(card_rect.right() - 52.0, card_rect.top() + 4.0), egui::vec2(46.0, 15.0));
        painter.rect_filled(badge, 3.0, tokens.status_ok);
        painter.text(badge.center(), egui::Align2::CENTER_CENTER, "MATCH", egui::FontId::proportional(9.5), tokens.text_on_accent);
    }

    let plot_rect = egui::Rect::from_min_max(
        egui::pos2(card_rect.left() + 6.0, card_rect.top() + 22.0),
        egui::pos2(card_rect.right() - 6.0, card_rect.bottom() - 28.0),
    );
    paint_card_primitives(&ui.painter_at(plot_rect), plot_rect, cache.get(entry), tokens);

    painter.text(
        egui::pos2(card_rect.left() + 6.0, card_rect.bottom() - 15.0),
        egui::Align2::LEFT_CENTER,
        card_subtitle(entry, group_tag),
        egui::FontId::proportional(9.5),
        tokens.text_secondary,
    );

    let border = if is_match {
        egui::Stroke::new(3.0, tokens.status_ok)
    } else {
        egui::Stroke::new(1.0, tokens.border_subtle)
    };
    painter.rect_stroke(card_rect, 5.0, border, egui::StrokeKind::Inside);
    clicked
}

/// `rings [..] · description`, cut to 30 characters (an ellipsis marks the cut).
pub(crate) fn card_subtitle(entry: &CoreGalleryEntry, group_tag: &str) -> String {
    let text = if entry.rings.is_empty() {
        group_tag.to_string()
    } else {
        let r_str: Vec<String> = entry.rings.iter().map(usize::to_string).collect();
        format!("rings [{}] · {}", r_str.join(","), entry.description)
    };
    if text.chars().count() > 30 {
        let s: String = text.chars().take(29).collect();
        format!("{s}…")
    } else {
        text
    }
}
