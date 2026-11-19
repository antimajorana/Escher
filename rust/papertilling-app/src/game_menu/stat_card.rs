//! Stat card of the highlighted Command Deck card (right column of the deck). WP0 stub with the final signature;
//! WP-C adds the rows, the verdict status lines and the Equip / Open in Forge / Open in Lab buttons.

use super::realms::Card;
use crate::app::PapertillingApp;

/// What a stat-card button asks the deck to do with its card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // M1 prewire: returned by the stat-card buttons (WP-C), handled by the deck (WP-B).
pub(crate) enum StatAction {
    Equip,
    OpenInForge,
    OpenInLab,
}

impl PapertillingApp {
    #[allow(dead_code)] // M1 prewire: hosted by the deck's right column (WP-B), filled by WP-C.
    pub(crate) fn show_stat_card(&mut self, ui: &mut egui::Ui, card: &Card) -> Option<StatAction> {
        ui.strong(card.title.as_str());
        None
    }
}
