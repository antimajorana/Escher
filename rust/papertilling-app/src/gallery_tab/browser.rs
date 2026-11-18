//! Visual lattice browser of the Gallery left rail: search + facet bar, a card grid with 2 x 2 unit-cell previews
//! (lazily built only for cards on screen) or a compact table, and selection of a lattice outside the current list.

use egui_extras::{Column, TableBuilder};

use super::filter::{BrowseFilter, ChiralFacet, SystemFacet, system_of};
use super::thumb_cache::LatticeThumbCache;
use super::{BrowseMode, GalleryTabState};
use crate::aliases::LatticeAliasKey;
use crate::backend::{AppBackend, CatalogueItemUi, CatalogueRuleUi};
use crate::canvas2d::Canvas2D;
use crate::theme::{StudioTheme, ThemeTokens};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrowserView {
    #[default]
    Cards,
    Table,
}

const CARD_MIN_W: f32 = 118.0;
const CARD_H: f32 = 128.0;
const CARD_GAP: f32 = 6.0;
const THUMB_H: f32 = 70.0;

/// Theme tokens of the visuals `ui` is drawn with (the left rail does not receive them explicitly).
pub(crate) fn tokens_of(ui: &egui::Ui) -> ThemeTokens {
    ThemeTokens::for_theme(if ui.visuals().dark_mode { StudioTheme::Studio } else { StudioTheme::Paper })
}

/// Short tag of the rule a catalogue / named item comes from (its alias-key rule slug).
pub(crate) fn rule_tag(key: &LatticeAliasKey) -> &'static str {
    CatalogueRuleUi::ALL
        .into_iter()
        .find(|r| r.slug() == key.rule)
        .map_or(if key.rule == "named" { "Named" } else { "Custom" }, CatalogueRuleUi::short_label)
}

impl GalleryTabState {
    /// Every lattice of `rule` in all 17 groups (built once per rule, labels kept current on rename).
    pub(crate) fn rule_index(&mut self, rule: CatalogueRuleUi, backend: &AppBackend) -> &[CatalogueItemUi] {
        self.rule_index[rule.index()].get_or_insert_with(|| {
            backend.groups().iter().flat_map(|g| backend.catalogue_for_group(g.symbol, rule)).collect()
        })
    }

    /// Select `item` (a stored catalogue or named lattice at its default values) wherever it is listed: switches
    /// mode, rule, group and list position together so the left rail shows it selected.
    pub fn select_external(&mut self, item: CatalogueItemUi, backend: &AppBackend, canvas: &mut Canvas2D) {
        let key = item.alias_key.clone();
        if key.rule == "named"
            && let Some(idx) = backend.named_lattices().iter().position(|n| n.alias_key == key)
        {
            self.select_named(idx, backend, canvas);
            return;
        }
        if let Some(rule) = CatalogueRuleUi::ALL.into_iter().find(|r| r.slug() == key.rule)
            && let Some(g_idx) = backend.groups().iter().position(|g| g.symbol == key.group)
        {
            let list = backend.catalogue_for_group(&key.group, rule);
            if let Some(idx) = list.iter().position(|it| it.alias_key == key) {
                self.rule = rule;
                self.mode = BrowseMode::ByGroup;
                self.selected_group_idx = g_idx;
                self.catalogue = list;
                self.select_catalogue_entry(idx, backend, canvas);
                return;
            }
        }
        self.activate_item(item, backend, canvas, true);
    }

    /// Search box, facet pickers and the Cards / Table toggle.
    pub(super) fn filter_bar(&mut self, ui: &mut egui::Ui) {
        // Search box alone on its row at the rail width: a wider row would widen the scroll area and clip the cards.
        ui.horizontal(|ui| {
            let clear_w = if self.filter.query.is_empty() { 0.0 } else { 24.0 };
            ui.add(
                egui::TextEdit::singleline(&mut self.filter.query)
                    .hint_text("Search name, alias, ITA no. or symbol")
                    .desired_width((ui.available_width() - clear_w).max(60.0)),
            );
            if !self.filter.query.is_empty() && ui.small_button("✕").on_hover_text("Clear search").clicked() {
                self.filter.query.clear();
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.view, BrowserView::Cards, "Cards");
            ui.selectable_value(&mut self.view, BrowserView::Table, "Table");
            ui.separator();
            egui::ComboBox::from_id_salt("facet_system")
                .width(96.0)
                .selected_text(self.filter.system.map_or("All systems", SystemFacet::label))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.filter.system, None, "All systems");
                    for s in SystemFacet::ALL {
                        ui.selectable_value(&mut self.filter.system, Some(s), s.label());
                    }
                });
            egui::ComboBox::from_id_salt("facet_chiral")
                .width(78.0)
                .selected_text(self.filter.chirality.label())
                .show_ui(ui, |ui| {
                    for c in ChiralFacet::ALL {
                        ui.selectable_value(&mut self.filter.chirality, c, c.label());
                    }
                });
            if !self.filter.is_default() && ui.small_button("Reset filters").clicked() {
                self.filter = BrowseFilter::default();
            }
        });
    }

    /// The lattice list of the current mode (filtered), as cards or table. Returns true when a lattice was picked.
    pub(super) fn lattice_list(&mut self, ui: &mut egui::Ui, backend: &AppBackend, canvas: &mut Canvas2D) -> bool {
        let groups = backend.groups();
        let named_mode = self.mode == BrowseMode::Named21;
        let cur_group = groups.get(self.selected_group_idx).map_or("", |g| g.symbol);
        let cur_visible = groups.get(self.selected_group_idx).is_some_and(|g| self.filter.group_visible(g));
        // Across all groups when text or chirality narrows the list, or when the system facet hides this group.
        let cross = !named_mode
            && (!self.filter.query.trim().is_empty() || self.filter.chirality != ChiralFacet::Any || !cur_visible);
        if cross {
            let _ = self.rule_index(self.rule, backend);
        }
        let src: &[CatalogueItemUi] = if named_mode {
            backend.named_lattices()
        } else if cross {
            self.rule_index[self.rule.index()].as_deref().unwrap_or_default()
        } else {
            &self.catalogue
        };
        let items: Vec<&CatalogueItemUi> =
            src.iter().filter(|it| self.filter.matches(it, system_of(groups, &it.group))).collect();
        let heading = if named_mode {
            format!("Named lattices ({} of {})", items.len(), src.len())
        } else if cross {
            format!("{} across 17 groups: {} match", self.rule.label(), items.len())
        } else {
            format!("Lattices in {cur_group} ({}, {} entries)", self.rule.label(), items.len())
        };
        ui.strong(heading);
        if items.is_empty() {
            ui.small("No lattice matches. Clear the search or reset the filters.");
            return false;
        }
        let selected = self.base_item.clone();
        let tokens = tokens_of(ui);
        self.thumbs.begin_frame(ui.ctx().cumulative_pass_nr());
        let picked = match self.view {
            BrowserView::Cards => card_grid(ui, &items, &selected, &mut self.thumbs, &tokens),
            BrowserView::Table => table(ui, &items, &selected, named_mode),
        };
        if self.thumbs.deferred {
            ui.ctx().request_repaint();
        }
        let Some(item) = picked.and_then(|i| items.get(i)).map(|it| (*it).clone()) else {
            return false;
        };
        self.select_external(item, backend, canvas);
        true
    }
}

/// `it` is the selected lattice: the same entry, or the catalogue entry of the selected named lattice (and vice
/// versa) — both carry the same named key in the same group.
pub(crate) fn is_selected(selected: &CatalogueItemUi, it: &CatalogueItemUi) -> bool {
    selected.alias_key == it.alias_key
        || (selected.named_key.is_some() && selected.named_key == it.named_key && selected.group == it.group)
}

fn card_grid(
    ui: &mut egui::Ui,
    items: &[&CatalogueItemUi],
    selected: &CatalogueItemUi,
    thumbs: &mut LatticeThumbCache,
    tokens: &ThemeTokens,
) -> Option<usize> {
    // never wider than the visible part of the rail (the scroll area may report more)
    let visible = ui.clip_rect().right() - ui.cursor().left() - 2.0;
    let avail = ui.available_width().min(visible).max(CARD_MIN_W);
    let cols = (((avail + CARD_GAP) / (CARD_MIN_W + CARD_GAP)).floor() as usize).max(1);
    let card_w = (avail - CARD_GAP * (cols as f32 - 1.0)) / cols as f32;
    let mut picked = None;
    ui.spacing_mut().item_spacing = egui::vec2(CARD_GAP, CARD_GAP);
    for (row_i, row) in items.chunks(cols).enumerate() {
        ui.horizontal(|ui| {
            for (j, it) in row.iter().enumerate() {
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(card_w, CARD_H), egui::Sense::click());
                if ui.is_rect_visible(rect) {
                    paint_card(ui, rect, it, is_selected(selected, it), resp.hovered(), thumbs, tokens);
                }
                let resp = resp.on_hover_text(card_tooltip(it));
                if resp.clicked() {
                    picked = Some(row_i * cols + j);
                }
            }
        });
    }
    picked
}

fn card_tooltip(it: &CatalogueItemUi) -> String {
    let chir = if it.topologically_chiral {
        "\n± chiral enantiomorph pair (chiral bare graph)"
    } else if it.chiral {
        "\n± chiral enantiomorph pair (merged under the Euclidean normalizer)"
    } else {
        ""
    };
    format!(
        "{}\n{} — {} | coordination {} | D-symbol {}{chir}",
        it.display_with_orig(),
        it.group,
        rule_tag(&it.alias_key),
        it.coordination,
        it.dsymbol_key
    )
}

fn paint_card(
    ui: &egui::Ui,
    rect: egui::Rect,
    it: &CatalogueItemUi,
    selected: bool,
    hovered: bool,
    thumbs: &mut LatticeThumbCache,
    t: &ThemeTokens,
) {
    let painter = ui.painter_at(rect);
    let bg = if selected {
        t.bg_card_selected
    } else if hovered {
        t.bg_card_hover
    } else {
        t.bg_card
    };
    let border = if selected { egui::Stroke::new(1.6, t.border_focus) } else { egui::Stroke::new(1.0, t.border_subtle) };
    painter.rect_filled(rect, 6.0, bg);
    painter.rect_stroke(rect, 6.0, border, egui::StrokeKind::Inside);
    let thumb = egui::Rect::from_min_size(rect.min + egui::vec2(5.0, 5.0), egui::vec2(rect.width() - 10.0, THUMB_H));
    painter.rect_filled(thumb, 4.0, t.bg_canvas_2d);
    match thumbs.get_or_build(it) {
        Some(g) => g.paint(&painter.with_clip_rect(thumb), thumb, t),
        None => {
            painter.text(thumb.center(), egui::Align2::CENTER_CENTER, "…", egui::FontId::proportional(14.0), t.text_muted);
        }
    }
    let x0 = rect.left() + 6.0;
    let w = rect.width() - 12.0;
    let mut y = thumb.bottom() + 4.0;
    let name = truncated(&painter, &it.label, egui::FontId::proportional(12.5), t.text_primary, w);
    painter.galley(egui::pos2(x0, y), name, t.text_primary);
    y += 17.0;
    // group pill, rule tag, chirality badge
    let mut x = x0;
    for (txt, fill, ink) in [
        (it.group.as_str(), t.accent_soft, t.accent_primary),
        (rule_tag(&it.alias_key), t.bg_card_hover, t.text_secondary),
    ] {
        let g = painter.layout_no_wrap(txt.to_owned(), egui::FontId::proportional(10.5), ink);
        let pill = egui::Rect::from_min_size(egui::pos2(x, y), g.size() + egui::vec2(10.0, 3.0));
        painter.rect_filled(pill, 7.0, fill);
        painter.galley(pill.min + egui::vec2(5.0, 1.5), g, ink);
        x = pill.right() + 4.0;
    }
    if it.chiral {
        let g = painter.layout_no_wrap(String::from("±"), egui::FontId::proportional(12.0), t.status_warn);
        painter.galley(egui::pos2(x + 1.0, y), g, t.status_warn);
    }
    y += 18.0;
    let counts = thumbs.peek(&it.alias_key).map_or_else(
        || format!("{} sites / cell", it.sites_per_cell),
        |g| format!("{} sites · {} bonds / cell", g.sites_per_cell, g.bonds_per_cell),
    );
    painter.text(egui::pos2(x0, y), egui::Align2::LEFT_TOP, counts, egui::FontId::proportional(10.5), t.text_muted);
}

fn truncated(painter: &egui::Painter, text: &str, font: egui::FontId, color: egui::Color32, w: f32) -> std::sync::Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::single_section(text.to_owned(), egui::TextFormat::simple(font, color));
    job.wrap = egui::text::TextWrapping::truncate_at_width(w);
    painter.layout_job(job)
}

fn table(ui: &mut egui::Ui, items: &[&CatalogueItemUi], selected: &CatalogueItemUi, named_mode: bool) -> Option<usize> {
    let mut picked = None;
    let cols: &[&str] = if named_mode { &["Lattice", "Grp", "Sheet", "Sites"] } else { &["Lattice", "Grp", "Sites", "Coord"] };
    ui.push_id(("gallery_list_tbl", named_mode), |ui| {
        TableBuilder::new(ui)
            .striped(true)
            .max_scroll_height(260.0)
            .column(Column::remainder())
            .column(Column::exact(42.0))
            .column(Column::exact(40.0))
            .column(Column::exact(44.0))
            .header(18.0, |mut h| {
                for t in cols {
                    h.col(|ui| {
                        ui.strong(*t);
                    });
                }
            })
            .body(|body| {
                body.rows(18.0, items.len(), |mut row| {
                    let i = row.index();
                    let it = items[i];
                    row.col(|ui| {
                        ui.horizontal(|ui| {
                            if ui.selectable_label(is_selected(selected, it), it.display_with_orig()).clicked() {
                                picked = Some(i);
                            }
                            if it.chiral {
                                ui.small("±").on_hover_text(card_tooltip(it));
                            }
                        });
                    });
                    row.col(|ui| {
                        ui.label(&it.group);
                    });
                    if named_mode {
                        row.col(|ui| {
                            ui.label(it.sheet_group.as_deref().unwrap_or("-"));
                        });
                        row.col(|ui| {
                            ui.label(it.sites_per_cell.to_string());
                        });
                    } else {
                        row.col(|ui| {
                            ui.label(it.sites_per_cell.to_string());
                        });
                        row.col(|ui| {
                            ui.label(&it.coordination);
                        });
                    }
                });
            });
    });
    picked
}
