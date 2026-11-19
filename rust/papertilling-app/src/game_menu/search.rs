//! Command Deck filter: the filter field, card matching, and the realm-3 catalogue search (the only owner of the
//! catalogue search: `deck_cards` calls [`PapertillingApp::deck_search_lattices`] for a non-empty realm-3 filter).

use std::collections::HashSet;

use super::DECK_FILTER_ID;
use super::realms::{Card, LatticeRef};
use crate::aliases::LatticeAliasKey;
use crate::app::PapertillingApp;
use crate::backend::{CatalogueItemUi, CatalogueRuleUi};
use crate::gallery_tab::filter::query_matches;

/// Most realm-3 search results (the same cap as realm 3 "This group").
const MAX_HITS: usize = 200;

/// Lattice cards match through `query_matches` (name, alias, ITA number, group symbol: the switcher's
/// semantics); every other card matches a case-insensitive substring of its title only (subtitles repeat across
/// sibling cards, so they would match too much).
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

/// Collects search hits in order, de-duplicated by alias key, up to [`MAX_HITS`].
struct Hits<'q> {
    q: &'q str,
    seen: HashSet<LatticeAliasKey>,
    out: Vec<LatticeRef>,
}

impl Hits<'_> {
    fn full(&self) -> bool {
        self.out.len() >= MAX_HITS
    }

    fn offer(&mut self, r: LatticeRef, it: &CatalogueItemUi) {
        if !self.full() && query_matches(self.q, it) && self.seen.insert(it.alias_key.clone()) {
            self.out.push(r);
        }
    }
}

impl PapertillingApp {
    /// The filter TextEdit; records its real `Id`, takes focus when asked (`/`, Ctrl+K, the HUD Search), and puts
    /// the cursor back on the first card whenever the text changes (so Enter equips the best match).
    pub(crate) fn deck_filter_field(&mut self, ui: &mut egui::Ui) {
        let resp = ui.add(
            egui::TextEdit::singleline(&mut self.deck.filter)
                .id_salt(DECK_FILTER_ID)
                .hint_text("Filter… ( / )")
                .desired_width(200.0),
        );
        self.deck.filter_id = Some(resp.id);
        if resp.changed() {
            self.deck.set_cur(0);
        }
        if self.deck.focus_filter {
            resp.request_focus();
            self.deck.focus_filter = false;
        }
    }

    /// Realm-3 search: Recent, then the Named 21, then the full catalogue of every rule (built synchronously once
    /// per rule and cached, as the Quick Lattice Switcher does) through `query_matches`, de-duplicated by alias
    /// key, at most [`MAX_HITS`] results.
    pub(crate) fn deck_search_lattices(&mut self, q: &str) -> Vec<LatticeRef> {
        let q = q.trim();
        if q.is_empty() {
            return Vec::new();
        }
        for rule in CatalogueRuleUi::ALL {
            let _ = self.gallery.rule_index(rule, &self.backend);
        }
        let mut hits = Hits { q, seen: HashSet::new(), out: Vec::new() };
        for (i, it) in self.switcher.recent.iter().enumerate() {
            hits.offer(LatticeRef::Recent(i), it);
        }
        for (i, it) in self.backend.named_lattices().iter().enumerate() {
            hits.offer(LatticeRef::Named(i), it);
        }
        for rule in CatalogueRuleUi::ALL {
            for (i, it) in self.gallery.rule_index_ready(rule).iter().enumerate() {
                if hits.full() {
                    return hits.out;
                }
                hits.offer(LatticeRef::Full(rule, i), it);
            }
        }
        hits.out
    }
}
