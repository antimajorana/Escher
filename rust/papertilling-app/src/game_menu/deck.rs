//! The full-canvas Command Deck overlay: an `Order::Middle` area between the HUD and the status bar, covering both
//! drawers. Top to bottom: header (title, realm tabs, filter), pill row, body (card grid with a cursor ring on the
//! left, the highlighted card's stat card on the right) and a footer of key hints. Every colour is a theme token.
//!
//! The panels keep drawing and polling underneath; the opaque backdrop only swallows their clicks.

use super::badges::badge_widget;
use super::realms::{Card, CardId, LatticeRef, pills};
use super::stat_card::StatAction;
use super::Realm;
use crate::app::PapertillingApp;
use crate::theme::ThemeTokens;
use egui::{Align, Layout, Rect, RichText, Sense, Stroke, StrokeKind, UiBuilder, pos2, vec2};

/// Card size (lattice cards are taller: they carry a thumbnail).
const CARD_W: f32 = 168.0;
const CARD_H: f32 = 112.0;
const CARD_H_THUMB: f32 = 132.0;
const THUMB_H: f32 = 70.0;
const CARD_GAP: f32 = 8.0;
const CARD_PAD: f32 = 8.0;
const CARD_RADIUS: f32 = 6.0;
/// Width of the stat-card column (shrinks on narrow windows so the grid keeps at least one column).
const STAT_W: f32 = 300.0;
const COL_GAP: f32 = 12.0;
const FOOTER_H: f32 = 20.0;
const FOOTER: &str = "1–7 / Q E  realm    Arrows / WASD  move    Enter  equip    Esc  resume";

/// The deck's layer (also raised every frame above the Combine popover and the Core Inspector window).
fn deck_layer() -> egui::LayerId {
    egui::LayerId::new(egui::Order::Middle, egui::Id::unique("command_deck"))
}

/// Whether the cursor card should be scrolled into view: the cursor, realm or pill changed since the last drawn
/// pass, or the deck was not drawn on the previous pass (just opened). Kept in egui temp memory, since only the
/// deck needs it.
fn cursor_moved(ctx: &egui::Context, key: (usize, usize, usize)) -> bool {
    let id = egui::Id::unique("command_deck_cursor_seen");
    let pass = ctx.cumulative_pass_nr();
    let last: Option<(u64, (usize, usize, usize))> = ctx.data(|d| d.get_temp(id));
    ctx.data_mut(|d| d.insert_temp(id, (pass, key)));
    match last {
        Some((p, k)) => k != key || p.saturating_add(1) < pass,
        None => true,
    }
}

/// What the user asked of the deck this frame (applied after drawing, when no card is borrowed).
enum DeckAction {
    Equip(usize),
    Stat(StatAction, CardId),
}

impl PapertillingApp {
    pub(crate) fn show_command_deck(&mut self, ctx: &egui::Context) {
        let content = ctx.content_rect();
        let mut host = self.deck.host_rect;
        if !host.is_positive() {
            host = content;
        }
        let tokens = self.theme.tokens();
        egui::Area::new(deck_layer().id)
            .order(egui::Order::Middle)
            .fixed_pos(host.min)
            .show(ctx, |ui| {
                ui.set_clip_rect(host);
                // opaque backdrop that swallows every click meant for the panels underneath
                let _ = ui.allocate_rect(host, Sense::click_and_drag());
                ui.painter().rect_filled(host, 0.0, tokens.bg_popover);
                let inner = host.shrink(12.0);
                let mut body = ui.new_child(
                    UiBuilder::new().id_salt("deck_body").max_rect(inner).layout(Layout::top_down(Align::Min)),
                );
                body.set_clip_rect(host);
                self.deck_body(&mut body, ctx, &tokens);
            });
        ctx.move_to_top(deck_layer());
        if self.gallery.thumbs.deferred {
            ctx.request_repaint();
        }
    }

    fn deck_body(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, tokens: &ThemeTokens) {
        self.deck_header(ui, tokens);
        self.deck_pill_row(ui, tokens);
        ui.separator();

        let realm = self.deck.realm;
        let pill = self.deck.cur_pill();
        let filter = self.deck.filter.clone();
        let cards = self.deck_cards(realm, pill, &filter);
        self.deck.n_cards = cards.len();
        if self.deck.cur() >= cards.len() {
            self.deck.set_cur(cards.len().saturating_sub(1));
        }
        let scroll = cursor_moved(ctx, (realm as usize, pill, self.deck.cur()));
        let mut action = self.deck.equip_request.take().filter(|&i| i < cards.len()).map(DeckAction::Equip);

        // body = what is left above the footer: grid on the left, stat card on the right
        let rest = ui.available_rect_before_wrap();
        let bottom = (rest.max.y - FOOTER_H - 4.0).max(rest.min.y);
        let body = Rect::from_min_max(rest.min, pos2(rest.max.x, bottom));
        let stat_w = STAT_W.min((body.width() - CARD_W - COL_GAP).max(0.0));
        let grid_rect = Rect::from_min_max(body.min, pos2(body.max.x - stat_w - COL_GAP, body.max.y));
        let stat_rect = Rect::from_min_max(pos2(body.max.x - stat_w, body.min.y), body.max);

        let mut grid_ui = ui.new_child(
            UiBuilder::new().id_salt("deck_grid_col").max_rect(grid_rect).layout(Layout::top_down(Align::Min)),
        );
        if let Some(i) = self.deck_grid(&mut grid_ui, &cards, tokens, scroll) {
            action = Some(DeckAction::Equip(i));
        }

        if stat_w > 0.0 {
            let x = stat_rect.min.x - 0.5 * COL_GAP;
            ui.painter().vline(x, stat_rect.y_range(), Stroke::new(1.0, tokens.border_subtle));
            let mut stat_ui = ui.new_child(
                UiBuilder::new().id_salt("deck_stat_col").max_rect(stat_rect).layout(Layout::top_down(Align::Min)),
            );
            stat_ui.set_clip_rect(stat_rect.intersect(ui.clip_rect()));
            if let Some(card) = cards.get(self.deck.cur()) {
                let picked = egui::ScrollArea::vertical()
                    .id_salt("deck_stat")
                    .auto_shrink([false, false])
                    .show(&mut stat_ui, |ui| self.show_stat_card(ui, card))
                    .inner;
                if let Some(a) = picked {
                    action = Some(DeckAction::Stat(a, card.id));
                }
            }
        }

        let footer = Rect::from_min_max(pos2(rest.min.x, bottom + 4.0), rest.max);
        let mut foot_ui = ui.new_child(
            UiBuilder::new().id_salt("deck_footer").max_rect(footer).layout(Layout::left_to_right(Align::Center)),
        );
        foot_ui.add(egui::Label::new(RichText::new(FOOTER).small().color(tokens.text_muted)).selectable(false));

        match action {
            Some(DeckAction::Equip(i)) => self.deck_equip(cards[i].id, ctx),
            Some(DeckAction::Stat(a, id)) => self.deck_stat_action(a, id, ctx),
            None => {}
        }
    }

    /// Title, the seven realm tabs (wrapping when narrow) and the filter field on the right.
    fn deck_header(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        const FILTER_W: f32 = 200.0;
        ui.horizontal(|ui| {
            let tabs_w = (ui.available_width() - FILTER_W - 24.0).max(120.0);
            let wrap = Layout::left_to_right(Align::Center).with_main_wrap(true);
            ui.allocate_ui_with_layout(vec2(tabs_w, 0.0), wrap, |ui| {
                ui.label(RichText::new("❖ PAPERTILLING CODEX").strong().color(tokens.text_primary));
                ui.add_space(8.0);
                for r in Realm::ALL {
                    let on = self.deck.realm == r;
                    let colour = if on { tokens.text_primary } else { tokens.text_secondary };
                    if ui.selectable_label(on, RichText::new(r.tab_label()).color(colour)).clicked() {
                        self.deck_select_realm(r);
                    }
                }
            });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| self.deck_filter_field(ui));
        });
    }

    /// Pills of the current realm (only when it has more than "All"); a pill change puts the cursor on card 0.
    fn deck_pill_row(&mut self, ui: &mut egui::Ui, tokens: &ThemeTokens) {
        let pl = pills(self.deck.realm);
        if pl.len() <= 1 {
            return;
        }
        ui.horizontal_wrapped(|ui| {
            for (i, p) in pl.iter().enumerate() {
                let on = self.deck.cur_pill() == i;
                let colour = if on { tokens.text_primary } else { tokens.text_secondary };
                if ui.selectable_label(on, RichText::new(*p).color(colour)).clicked() && !on {
                    self.deck.pill[self.deck.realm.idx()] = i;
                    self.deck.set_cur(0);
                }
            }
        });
    }

    /// The card grid (fixed-size cards, as many columns as fit). A click moves the cursor; a double click asks for
    /// that card to be equipped (returned). Records `deck.cols`.
    fn deck_grid(&mut self, ui: &mut egui::Ui, cards: &[Card], tokens: &ThemeTokens, scroll: bool) -> Option<usize> {
        let mut equip = None;
        egui::ScrollArea::vertical().id_salt("deck_cards").auto_shrink([false, false]).show(ui, |ui| {
            let w = ui.available_width();
            let cols = (((w + CARD_GAP) / (CARD_W + CARD_GAP)).floor() as usize).max(1);
            self.deck.cols = cols;
            if cards.is_empty() {
                ui.label(RichText::new("No card matches the filter.").color(tokens.text_muted));
                return;
            }
            ui.spacing_mut().item_spacing = vec2(CARD_GAP, CARD_GAP);
            let cur = self.deck.cur();
            for (row_i, row) in cards.chunks(cols).enumerate() {
                ui.horizontal_top(|ui| {
                    for (j, card) in row.iter().enumerate() {
                        let i = row_i * cols + j;
                        let resp = self.deck_card(ui, i, card, i == cur, tokens);
                        if scroll && i == cur {
                            resp.scroll_to_me(None);
                        }
                        if resp.double_clicked() {
                            equip = Some(i);
                        } else if resp.clicked() {
                            self.deck.set_cur(i);
                        }
                    }
                });
            }
        });
        equip
    }

    /// One card: background by state, 2 px focus ring on the cursor, optional thumbnail, then real labels for the
    /// title, subtitle and badge (so their text is in the accessibility tree).
    fn deck_card(
        &mut self,
        ui: &mut egui::Ui,
        i: usize,
        card: &Card,
        is_cursor: bool,
        t: &ThemeTokens,
    ) -> egui::Response {
        let h = if card.thumb.is_some() { CARD_H_THUMB } else { CARD_H };
        let (rect, resp) = ui.allocate_exact_size(vec2(CARD_W, h), Sense::click());
        let label = card.a11y_label();
        resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::Button, true, is_cursor, &label));
        let visible = ui.is_rect_visible(rect);
        if visible {
            let fill = if is_cursor {
                t.bg_card_selected
            } else if resp.contains_pointer() {
                t.bg_card_hover
            } else {
                t.bg_card
            };
            let ring = if is_cursor { Stroke::new(2.0, t.border_focus) } else { Stroke::new(1.0, t.border_subtle) };
            ui.painter().rect_filled(rect, CARD_RADIUS, fill);
            ui.painter().rect_stroke(rect, CARD_RADIUS, ring, StrokeKind::Inside);
        }
        let inner = rect.shrink(CARD_PAD);
        let mut top = inner.min.y;
        if let Some(r) = card.thumb {
            let thumb = Rect::from_min_size(inner.min, vec2(inner.width(), THUMB_H));
            if visible {
                self.paint_card_thumb(ui, thumb, r, t);
            }
            top = thumb.max.y + 4.0;
        }
        let text_rect = Rect::from_min_max(pos2(inner.min.x, top), inner.max);
        let mut cu = ui.new_child(
            UiBuilder::new().id_salt(("deck_card", i)).max_rect(text_rect).layout(Layout::top_down(Align::Min)),
        );
        cu.set_clip_rect(rect.intersect(ui.clip_rect()));
        cu.spacing_mut().item_spacing.y = 2.0;
        let title = RichText::new(card.title.as_str()).strong().color(t.text_primary);
        cu.add(egui::Label::new(title).truncate().selectable(false));
        if !card.subtitle.is_empty() {
            let sub = RichText::new(card.subtitle.as_str()).small().color(t.text_secondary);
            cu.add(egui::Label::new(sub).truncate().selectable(false));
        }
        if let Some(b) = card.badge {
            badge_widget(&mut cu, b, t);
        }
        resp
    }

    /// The lattice preview of a realm-3 card, from the gallery's thumbnail cache (built within its frame budget;
    /// a deferred one shows the empty canvas colour and is painted on a later frame).
    fn paint_card_thumb(&mut self, ui: &egui::Ui, rect: Rect, r: LatticeRef, t: &ThemeTokens) {
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 3.0, t.bg_canvas_2d);
        let Some(item) = self.lattice_ref_item(r) else { return };
        if let Some(g) = self.gallery.thumbs.peek(&item.alias_key) {
            g.paint(&p, rect, t);
            return;
        }
        let item = item.clone();
        if let Some(g) = self.gallery.thumbs.get_or_build(&item) {
            g.paint(&p, rect, t);
        }
    }

    /// A stat-card button: Equip, or (lattice cards only) open the lattice in the Defects Forge / Analysis Lab.
    fn deck_stat_action(&mut self, a: StatAction, id: CardId, ctx: &egui::Context) {
        match (a, id) {
            (StatAction::Equip, _) => self.deck_equip(id, ctx),
            (StatAction::OpenInForge, CardId::Lattice(r)) => {
                self.deck_open_in_forge(r);
                self.deck_close();
            }
            (StatAction::OpenInLab, CardId::Lattice(r)) => {
                self.deck_open_in_lab(r);
                self.deck_close();
            }
            _ => {}
        }
    }
}
