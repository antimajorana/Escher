//! Command Deck filter: the filter field, card matching, and the realm-3 catalogue search. WP0 gives working
//! minimal bodies with the final signatures; WP-B owns the file afterwards.

use super::DECK_FILTER_ID;
use super::realms::{Card, LatticeRef, MAX_LATTICE_CARDS};
use crate::app::PapertillingApp;
use crate::backend::{CatalogueItemUi, CatalogueRuleUi};
use crate::gallery_tab::filter::query_matches;

/// Lattice cards match through `query_matches` (name, alias, ITA number, group symbol: the switcher's
/// semantics); every other card matches a case-insensitive substring of its title only.
pub(crate) fn filter_matches(card: &Card, item: Option<&CatalogueItemUi>, q: &str) -> bool {
    let q = q.trim();
    if q.is_empty() {
        return true;
    }
    match item {
        Some(it) => query_matches(q, it),
        None => card.title.to_lowercase().contains(&q.to_lowercase()),
    }
}

impl PapertillingApp {
    /// The filter TextEdit; records its real `Id` and takes focus when asked (`/`, Ctrl+K, the HUD Search).
    pub(crate) fn deck_filter_field(&mut self, ui: &mut egui::Ui) {
        let resp = ui.add(
            egui::TextEdit::singleline(&mut self.deck.filter)
                .id_salt(DECK_FILTER_ID)
                .hint_text("Filter… ( / )")
                .desired_width(200.0),
        );
        self.deck.filter_id = Some(resp.id);
        if self.deck.focus_filter {
            resp.request_focus();
            self.deck.focus_filter = false;
        }
    }

    /// Realm-3 search: Recent + Named + full catalogue (all rules) through `query_matches`, de-duplicated by
    /// alias key, at most [`MAX_LATTICE_CARDS`] results.
    pub(crate) fn deck_search_lattices(&mut self, q: &str) -> Vec<LatticeRef> {
        for rule in CatalogueRuleUi::ALL {
            let _ = self.gallery.rule_index(rule, &self.backend);
        }
        let mut seen = Vec::new();
        let mut out = Vec::new();
        let mut take = |r: LatticeRef, it: &CatalogueItemUi, out: &mut Vec<LatticeRef>| {
            if out.len() < MAX_LATTICE_CARDS && query_matches(q, it) && !seen.contains(&it.alias_key) {
                seen.push(it.alias_key.clone());
                out.push(r);
            }
        };
        for (i, it) in self.switcher.recent.iter().enumerate() {
            take(LatticeRef::Recent(i), it, &mut out);
        }
        for (i, it) in self.backend.named_lattices().iter().enumerate() {
            take(LatticeRef::Named(i), it, &mut out);
        }
        for rule in CatalogueRuleUi::ALL {
            for (i, it) in self.gallery.rule_index_ready(rule).iter().enumerate() {
                take(LatticeRef::Full(rule, i), it, &mut out);
            }
        }
        out
    }
}
