//! Global Quick Lattice Switcher: a searchable command palette (the HUD "Search (Ctrl+K)" button or Ctrl+K while
//! the Command Deck is closed) listing Recent (LRU, 10), the 21 named lattices and the full
//! stored catalogue of all three rules. Picking a lattice selects it in the Gallery and reloads the active tab
//! at once (`PapertillingApp::select_lattice_and_sync`).

use std::collections::VecDeque;
use std::sync::Arc;

use crate::app::PapertillingApp;
use crate::backend::{CatalogueItemUi, CatalogueRuleUi};
use crate::gallery_tab::browser::{rule_tag, tokens_of};
use crate::gallery_tab::filter::query_matches;
use crate::gallery_tab::thumb_cache::ThumbGeom;

pub const RECENT_MAX: usize = 10;
const ROW_H: f32 = 22.0;
const LIST_H: f32 = 420.0;

#[derive(Default)]
pub struct LatticeSwitcher {
    pub open: bool,
    pub query: String,
    focus_pending: bool,
    highlight: usize,
    /// Most recent first, unique by alias key, at most [`RECENT_MAX`].
    pub recent: VecDeque<CatalogueItemUi>,
}

impl LatticeSwitcher {
    pub fn open(&mut self) {
        self.open = true;
        self.query.clear();
        self.focus_pending = true;
        self.highlight = 0;
    }

    /// Refresh the labels of the Recent entries after a rename.
    pub fn refresh_labels(&mut self, backend: &crate::backend::AppBackend) {
        for it in &mut self.recent {
            it.label = backend.resolve_alias(it);
        }
    }

    /// Record `item` as the most recent lattice (moves it to the front if already listed).
    pub fn push_recent(&mut self, item: &CatalogueItemUi) {
        // custom-builder lattices cannot be reselected from a list (their builder state is not stored)
        if item.alias_key.rule == "custom" {
            return;
        }
        if self.recent.front().is_some_and(|f| f.alias_key == item.alias_key) {
            return;
        }
        self.recent.retain(|r| r.alias_key != item.alias_key);
        self.recent.push_front(item.clone());
        self.recent.truncate(RECENT_MAX);
    }
}

#[derive(Clone, Copy)]
enum Row {
    Header(&'static str),
    Recent(usize),
    Named(usize),
    Full(CatalogueRuleUi, usize),
}

impl PapertillingApp {
    pub(crate) fn toggle_switcher(&mut self) {
        if self.switcher.open {
            self.switcher.open = false;
        } else {
            self.gallery.cancel_rename();
            self.switcher.open();
        }
    }

    /// A `px` x `px` thumbnail of the lattice the HUD lattice segment names: `active_item()`, i.e. at the current
    /// slider values. At the stored values it is the shared Gallery preview (keyed by alias); a tuned lattice gets
    /// its own one-slot preview, rebuilt when its recipe changes (never added to the alias-keyed cache).
    pub(crate) fn paint_lattice_thumb(&mut self, ui: &mut egui::Ui, px: f32) {
        let t = tokens_of(ui);
        let (thumb, _) = ui.allocate_exact_size(egui::vec2(px, px), egui::Sense::hover());
        if ui.is_rect_visible(thumb) {
            let p = ui.painter_at(thumb);
            p.rect_filled(thumb, 3.0, t.bg_canvas_2d);
            if let Some(g) = self.hud_thumb_geom() {
                g.paint(&p.with_clip_rect(thumb), thumb, &t);
            }
        }
    }

    /// Preview geometry of `gallery.active_item()` (see [`Self::paint_lattice_thumb`]).
    pub(crate) fn hud_thumb_geom(&mut self) -> Option<Arc<ThumbGeom>> {
        let (base, active) = (self.gallery.base_item(), self.gallery.active_item());
        if active.recipe == base.recipe && active.frame == base.frame {
            let base = base.clone();
            return self.gallery.thumbs.get_or_build(&base);
        }
        let key = (active.alias_key.clone(), active.recipe.clone(), active.frame);
        if self.deck.hud_thumb.as_ref().is_none_or(|(k, _)| *k != key) {
            let g = Arc::new(ThumbGeom::build(active));
            self.deck.hud_thumb = Some((key, g));
        }
        self.deck.hud_thumb.as_ref().map(|(_, g)| Arc::clone(g))
    }

    /// The palette itself (a modal; Esc or a click outside closes it).
    pub(crate) fn show_switcher_modal(&mut self, ctx: &egui::Context) {
        if !self.switcher.open {
            return;
        }
        for r in CatalogueRuleUi::ALL {
            let _ = self.gallery.rule_index(r, &self.backend);
        }
        let id = egui::Id::unique("quick_lattice_switcher");
        let area = egui::Modal::default_area(id).anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 56.0));
        let mut picked: Option<CatalogueItemUi> = None;
        let resp = egui::Modal::new(id).area(area).show(ctx, |ui| {
            ui.set_width(520.0);
            picked = self.switcher_body(ui);
        });
        if let Some(item) = picked {
            self.switcher.open = false;
            self.select_lattice_and_sync(item);
        } else if resp.should_close() {
            self.switcher.open = false;
        }
    }

    fn switcher_body(&mut self, ui: &mut egui::Ui) -> Option<CatalogueItemUi> {
        let t = tokens_of(ui);
        ui.horizontal(|ui| {
            ui.strong("Switch lattice");
            ui.small("type to filter · ↑↓ · Enter · Esc");
        });
        let edit = ui.add(
            egui::TextEdit::singleline(&mut self.switcher.query)
                .hint_text("name, alias, group symbol or ITA number (e.g. kagome, p4g, 17)")
                .desired_width(f32::INFINITY),
        );
        if std::mem::take(&mut self.switcher.focus_pending) {
            edit.request_focus();
        }
        if edit.changed() {
            self.switcher.highlight = 0;
        }
        let rows = self.switcher_rows();
        let n_items = rows.iter().filter(|r| !matches!(r, Row::Header(_))).count();
        let (down, up, enter) = ui.input(|i| {
            (i.key_pressed(egui::Key::ArrowDown), i.key_pressed(egui::Key::ArrowUp), i.key_pressed(egui::Key::Enter))
        });
        if down {
            self.switcher.highlight = (self.switcher.highlight + 1).min(n_items.saturating_sub(1));
        }
        if up {
            self.switcher.highlight = self.switcher.highlight.saturating_sub(1);
        }
        let item_rows: Vec<Row> = rows.iter().copied().filter(|r| !matches!(r, Row::Header(_))).collect();
        if enter && let Some(r) = item_rows.get(self.switcher.highlight) {
            return self.row_item(*r).cloned();
        }
        if n_items == 0 {
            ui.colored_label(t.text_muted, "No stored lattice matches.");
            return None;
        }
        let hl_row = item_rows.get(self.switcher.highlight).copied();
        let mut picked = None;
        let mut area = egui::ScrollArea::vertical().id_salt("switcher_rows").max_height(LIST_H).auto_shrink([false, true]);
        if (up || down)
            && let Some(h) = hl_row
            && let Some(pos) = rows.iter().position(|r| same_row(*r, h))
        {
            // show_rows draws only the visible rows, so scroll_to_me cannot reach an undrawn one: place it ourselves
            let rh = ROW_H + ui.spacing().item_spacing.y;
            area = area.vertical_scroll_offset((pos as f32 * rh - 0.5 * (LIST_H - rh)).max(0.0));
        }
        area.show_rows(
            ui,
            ROW_H,
            rows.len(),
            |ui, range| {
                for r in &rows[range] {
                    match *r {
                        Row::Header(h) => {
                            ui.add_sized([ui.available_width(), ROW_H], egui::Label::new(egui::RichText::new(h).strong().color(t.text_secondary)));
                        }
                        row => {
                            let Some(it) = self.row_item(row) else { continue };
                            let is_hl = hl_row.is_some_and(|h| same_row(h, row));
                            let txt = format!(
                                "{}{}    {} · {}",
                                it.display_with_orig(),
                                if it.chiral { "  ±" } else { "" },
                                it.group,
                                rule_tag(&it.alias_key)
                            );
                            let resp = ui.add_sized([ui.available_width(), ROW_H], egui::Button::selectable(is_hl, txt));
                            if resp.clicked() {
                                picked = Some(it.clone());
                            }
                        }
                    }
                }
            },
        );
        picked
    }

    fn row_item(&self, r: Row) -> Option<&CatalogueItemUi> {
        match r {
            Row::Header(_) => None,
            Row::Recent(i) => self.switcher.recent.get(i),
            Row::Named(i) => self.backend.named_lattices().get(i),
            Row::Full(rule, i) => self.gallery.rule_index_ready(rule).get(i),
        }
    }

    /// Section headers and the matching rows of each section.
    fn switcher_rows(&self) -> Vec<Row> {
        let q = self.switcher.query.as_str();
        let hit = |it: &CatalogueItemUi| query_matches(q, it);
        let mut rows = Vec::new();
        let recent: Vec<Row> =
            (0..self.switcher.recent.len()).filter(|&i| hit(&self.switcher.recent[i])).map(Row::Recent).collect();
        if !recent.is_empty() {
            rows.push(Row::Header("Recent"));
            rows.extend(recent);
        }
        let named: Vec<Row> = self
            .backend
            .named_lattices()
            .iter()
            .enumerate()
            .filter(|(_, it)| hit(it))
            .map(|(i, _)| Row::Named(i))
            .collect();
        if !named.is_empty() {
            rows.push(Row::Header("Named 21"));
            rows.extend(named);
        }
        let mut full = Vec::new();
        for rule in CatalogueRuleUi::ALL {
            let list = self.gallery.rule_index_ready(rule);
            full.extend(list.iter().enumerate().filter(|(_, it)| hit(it)).map(|(i, _)| Row::Full(rule, i)));
        }
        if !full.is_empty() {
            rows.push(Row::Header("Full Catalogue (Standard, Atomic, Delaunay)"));
            rows.extend(full);
        }
        rows
    }
}

fn same_row(a: Row, b: Row) -> bool {
    match (a, b) {
        (Row::Recent(x), Row::Recent(y)) | (Row::Named(x), Row::Named(y)) => x == y,
        (Row::Full(r1, x), Row::Full(r2, y)) => r1 == r2 && x == y,
        _ => false,
    }
}
