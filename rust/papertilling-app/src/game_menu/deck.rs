//! The full-canvas Command Deck overlay (an `Order::Middle` area between the HUD and the status bar, covering
//! both drawers). WP0 version: opaque backdrop, realm tabs, filter, pills and a plain card list that a click
//! selects and a double click / Enter equips. WP-B replaces the list with the card grid and stat-card column.

use super::realms::pills;
use super::Realm;
use crate::app::PapertillingApp;

/// The deck's layer (also raised every frame above the Combine popover and the Core Inspector window).
fn deck_layer() -> egui::LayerId {
    egui::LayerId::new(egui::Order::Middle, egui::Id::unique("command_deck"))
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
                let _ = ui.allocate_rect(host, egui::Sense::click_and_drag());
                ui.painter().rect_filled(host, 0.0, tokens.bg_popover);
                let inner = host.shrink(12.0);
                ui.scope_builder(egui::UiBuilder::new().max_rect(inner), |ui| self.deck_body(ui, ctx));
            });
        ctx.move_to_top(deck_layer());
    }

    fn deck_body(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal_wrapped(|ui| {
            ui.strong("❖ PAPERTILLING CODEX");
            for r in Realm::ALL {
                if ui.selectable_label(self.deck.realm == r, r.tab_label()).clicked() {
                    self.deck_select_realm(r);
                }
            }
            self.deck_filter_field(ui);
        });
        let realm = self.deck.realm;
        let pl = pills(realm);
        if pl.len() > 1 {
            ui.horizontal_wrapped(|ui| {
                for (i, p) in pl.iter().enumerate() {
                    if ui.selectable_label(self.deck.cur_pill() == i, *p).clicked() {
                        self.deck.pill[realm as usize] = i;
                    }
                }
            });
        }
        ui.separator();
        let filter = self.deck.filter.clone();
        let cards = self.deck_cards(realm, self.deck.cur_pill(), &filter);
        self.deck.cols = 1;
        self.deck.n_cards = cards.len();
        if self.deck.cur() >= cards.len() {
            self.deck.set_cur(cards.len().saturating_sub(1));
        }
        let mut equip = self.deck.equip_request.take().filter(|&i| i < cards.len());
        egui::ScrollArea::vertical().id_salt("deck_cards").show(ui, |ui| {
            if cards.is_empty() {
                ui.weak("No card matches the filter.");
            }
            for (i, card) in cards.iter().enumerate() {
                let resp = ui.selectable_label(i == self.deck.cur(), card.a11y_label());
                if resp.clicked() {
                    self.deck.set_cur(i);
                }
                if resp.double_clicked() {
                    equip = Some(i);
                }
            }
        });
        ui.weak("1–7 / Q E  realm    Arrows / WASD  move    Enter  equip    Esc  resume");
        if let Some(i) = equip {
            self.deck_equip(cards[i].id, ctx);
        }
    }
}
