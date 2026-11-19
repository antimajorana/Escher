//! The Codex: a one-row HUD (`hud_bar`) over the viewport and a full-canvas Command Deck overlay (`deck`) that
//! replace the old mode and sub-view strips. Seven realms group every view, sub-tab and stage sub-row as cards
//! (`realms`); equipping a card routes the app there (`equip`). Keyboard: Tab / M open and close the deck, 1-7 and
//! Q / E switch realms, arrows / WASD move, Enter equips, Esc resumes (`keys`). C1: realm 6 (Analysis Lab) is
//! hidden while Analysis is postponed: the deck shows [`Realm::SHOWN`], key 6 does nothing, Q / E skip it.
//!
//! The deck only overlays the panels: they keep drawing and polling underneath, so workers never stall.

use crate::app::{AppMode, DefectsView, LatticesView, PapertillingApp};
use crate::session::{name_of, parse_named};
use crate::workbench::WorkbenchView;

mod badges;
mod breadcrumb;
mod deck;
mod equip;
mod hud_bar;
mod keys;
mod realms;
mod search;
mod stat_card;
#[cfg(test)]
pub(crate) mod test_util;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_nav;
#[cfg(test)]
mod tests_nav2;
#[cfg(test)]
mod tests_session;
#[cfg(test)]
mod tests_review;
#[cfg(test)]
mod tests_c1;

pub(crate) use hud_bar::HUD_MIN_WINDOW_WIDTH;
pub(crate) use realms::{CardId, LatticeRef};

/// `id_salt` of the deck's filter TextEdit (its real `Id` is recorded in [`DeckState::filter_id`]).
pub(crate) const DECK_FILTER_ID: &str = "deck_filter";

/// Status note when a session saved in the postponed Analysis mode, tab or realm is restored (C1).
pub(crate) const ANALYSIS_POSTPONED_NOTE: &str = "Analysis is postponed; restored to Lattices";

/// The seven realms of the Command Deck, in tab order (keys 1-7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum Realm {
    Geometries,
    Groups,
    #[default]
    Tilings,
    Manifolds,
    Forge,
    AnalysisLab,
    PresetsIo,
}

impl Realm {
    pub(crate) const ALL: [Self; 7] = [
        Self::Geometries,
        Self::Groups,
        Self::Tilings,
        Self::Manifolds,
        Self::Forge,
        Self::AnalysisLab,
        Self::PresetsIo,
    ];

    /// The realms the deck shows (C1: Analysis Lab is postponed). `ALL` and the `[_; 7]` arrays keep their
    /// indices, so saved sessions and the number keys 1-5 and 7 are unchanged; key 6 does nothing.
    pub(crate) const SHOWN: [Self; 6] =
        [Self::Geometries, Self::Groups, Self::Tilings, Self::Manifolds, Self::Forge, Self::PresetsIo];

    /// The realm is one the deck shows.
    pub(crate) fn is_shown(self) -> bool {
        self != Self::AnalysisLab
    }

    /// 1..=7 (the realm's number key).
    pub(crate) fn number(self) -> usize {
        self as usize + 1
    }

    /// The shown realm of number key `n` (`None` for 6, the hidden Analysis Lab).
    pub(crate) fn from_number(n: usize) -> Option<Self> {
        n.checked_sub(1).and_then(|i| Self::ALL.get(i).copied()).filter(|r| r.is_shown())
    }

    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::Geometries => "Geometries",
            Self::Groups => "Groups & Symmetries",
            Self::Tilings => "Tilings & Nets",
            Self::Manifolds => "Manifolds & Surfaces",
            Self::Forge => "Defects Forge",
            Self::AnalysisLab => "Analysis Lab",
            Self::PresetsIo => "Presets & I/O",
        }
    }

    /// Label of the realm's tab in the deck header (number + short name).
    pub(crate) fn tab_label(self) -> &'static str {
        match self {
            Self::Geometries => "1 Geometries",
            Self::Groups => "2 Groups",
            Self::Tilings => "3 Tilings & Nets",
            Self::Manifolds => "4 Manifolds",
            Self::Forge => "5 Defects Forge",
            Self::AnalysisLab => "6 Analysis Lab",
            Self::PresetsIo => "7 Presets & I/O",
        }
    }

    /// The top-level mode the realm's cards lead to (`None`: Presets & I/O, which keeps the current mode).
    #[cfg(test)] // the realm table is checked by T1; the app routes through deck_equip instead
    pub(crate) fn app_mode(self) -> Option<AppMode> {
        match self {
            Self::Geometries | Self::Groups | Self::Tilings | Self::Manifolds => Some(AppMode::Lattices),
            Self::Forge => Some(AppMode::Defects),
            Self::AnalysisLab => Some(AppMode::Analysis),
            Self::PresetsIo => None,
        }
    }

    /// Position in [`Self::SHOWN`] (a hidden realm counts as the shown realm just before it).
    fn shown_pos(self) -> usize {
        Self::SHOWN.iter().rposition(|r| r.idx() <= self.idx()).unwrap_or(0)
    }

    /// Previous shown realm, wrapping (Q).
    pub(crate) fn prev(self) -> Self {
        let n = Self::SHOWN.len();
        let at = self.shown_pos();
        let back = if self.is_shown() { 1 } else { 0 };
        Self::SHOWN[(at + n - back) % n]
    }

    /// Next shown realm, wrapping (E).
    pub(crate) fn next(self) -> Self {
        Self::SHOWN[(self.shown_pos() + 1) % Self::SHOWN.len()]
    }

    /// `self`, or Tilings when `self` is hidden (the deck never opens on a hidden realm).
    pub(crate) fn shown_or_default(self) -> Self {
        if self.is_shown() { self } else { Self::Tilings }
    }

    fn idx(self) -> usize {
        self as usize
    }
}

/// Command Deck state. Only `realm` is saved in the session; the deck always loads closed.
pub(crate) struct DeckState {
    pub open: bool,
    pub realm: Realm,
    /// Highlighted card per realm (index into `deck_cards`).
    pub cursor: [usize; 7],
    /// Selected pill per realm (0 = All).
    pub pill: [usize; 7],
    /// Cleared on realm switch and on close.
    pub filter: String,
    /// Request focus on the filter TextEdit this frame.
    pub focus_filter: bool,
    /// Grid columns drawn last frame (>= 1).
    pub cols: usize,
    /// Cards drawn last frame for the current realm / pill / filter.
    pub n_cards: usize,
    /// +1 only when `deck_select_realm` changes the realm; `deck_open` never counts.
    pub realm_switches: u32,
    /// Real `Id` of the filter TextEdit (from its `Response`), set each frame it is drawn.
    pub filter_id: Option<egui::Id>,
    /// HUD bottom .. status-bar top, full window width (set each frame).
    pub host_rect: egui::Rect,
    /// Set by the keys (Enter / Space); consumed by the deck drawing (index into the cards).
    pub equip_request: Option<usize>,
    /// `/` focused the filter from its key event alone; its text event may still arrive in a later frame.
    pub slash_text_pending: bool,
    /// The realm came from a session file: the next Tab / Codex opens the deck there (once), not at the realm of
    /// the current view.
    pub resume_saved_realm: bool,
    /// How much of the HUD's right cluster is on the row (set each frame).
    pub hud_density: hud_bar::HudDensity,
    /// Width of the HUD rename controls last frame, with whether a rename was in progress then.
    pub hud_rename_w: Option<(bool, f32)>,
    /// HUD thumbnail of a tuned lattice (slider values off the stored recipe), keyed by what it was built from.
    pub hud_thumb: Option<(HudThumbKey, std::sync::Arc<crate::gallery_tab::thumb_cache::ThumbGeom>)>,
}

/// What a tuned HUD thumbnail was built from: alias key, recipe and frame.
pub(crate) type HudThumbKey =
    (crate::aliases::LatticeAliasKey, papertilling_core::types::Recipe, papertilling_core::types::Frame2D);

impl Default for DeckState {
    fn default() -> Self {
        Self {
            open: false,
            realm: Realm::default(),
            cursor: [0; 7],
            pill: [0; 7],
            filter: String::new(),
            focus_filter: false,
            cols: 3,
            n_cards: 0,
            realm_switches: 0,
            filter_id: None,
            host_rect: egui::Rect::NOTHING,
            equip_request: None,
            slash_text_pending: false,
            resume_saved_realm: false,
            hud_density: hud_bar::HudDensity::Full,
            hud_rename_w: None,
            hud_thumb: None,
        }
    }
}

impl DeckState {
    /// Session JSON: `{"realm": "<Realm>"}`.
    pub(crate) fn save(&self) -> serde_json::Value {
        serde_json::json!({ "realm": name_of(self.realm) })
    }

    /// Restore the realm; false (with a note when something was there) when absent or unreadable.
    pub(crate) fn restore(&mut self, v: Option<&serde_json::Value>, notes: &mut Vec<String>) -> bool {
        let Some(v) = v else { return false };
        let name = v.get("realm").and_then(serde_json::Value::as_str);
        match parse_named(&Realm::ALL, name) {
            Some(r) if !r.is_shown() => {
                if !notes.iter().any(|n| n == ANALYSIS_POSTPONED_NOTE) {
                    notes.push(String::from(ANALYSIS_POSTPONED_NOTE));
                }
                self.realm = r.shown_or_default();
                false
            }
            Some(r) => {
                self.realm = r;
                self.resume_saved_realm = true;
                true
            }
            None => {
                notes.push(String::from("unknown deck realm (kept current)"));
                false
            }
        }
    }

    /// Cursor of the current realm.
    pub(crate) fn cur(&self) -> usize {
        self.cursor[self.realm.idx()]
    }

    pub(crate) fn set_cur(&mut self, i: usize) {
        self.cursor[self.realm.idx()] = i;
    }

    /// Pill of the current realm.
    pub(crate) fn cur_pill(&self) -> usize {
        self.pill[self.realm.idx()]
    }
}

impl PapertillingApp {
    /// Open the deck at `realm` (not counted as a realm switch), with the cursor on the active view's card when it
    /// is listed there. Any focused widget gives up focus, so no hidden text field keeps receiving keys.
    pub(crate) fn deck_open(&mut self, realm: Realm, ctx: &egui::Context) {
        self.switcher.open = false;
        self.deck.open = true;
        let realm = realm.shown_or_default();
        self.deck.realm = realm;
        self.deck.resume_saved_realm = false;
        self.deck.filter.clear();
        self.deck.focus_filter = false;
        self.deck.equip_request = None;
        ctx.memory_mut(|m| {
            if let Some(id) = m.focused() {
                m.surrender_focus(id);
            }
        });
        if let Some(i) = self.active_card_index(realm) {
            self.deck.set_cur(i);
        }
    }

    /// Realm for Tab / the Codex button: the realm saved in the restored session the first time, else the realm of
    /// the current view.
    pub(crate) fn deck_entry_realm(&mut self) -> Realm {
        if std::mem::take(&mut self.deck.resume_saved_realm) { self.deck.realm } else { self.current_realm() }
    }

    pub(crate) fn deck_close(&mut self) {
        self.deck.open = false;
        self.deck.filter.clear();
        self.deck.focus_filter = false;
        self.deck.equip_request = None;
        self.deck.slash_text_pending = false;
    }

    pub(crate) fn deck_select_realm(&mut self, realm: Realm) {
        let realm = realm.shown_or_default();
        if realm != self.deck.realm {
            self.deck.realm_switches += 1;
        }
        self.deck.realm = realm;
        self.deck.filter.clear();
        self.deck.equip_request = None;
    }

    /// The realm of the view on screen (also the realm of the HUD mode segment).
    pub(crate) fn current_realm(&self) -> Realm {
        match self.active_mode {
            AppMode::Lattices => match self.lattices_view {
                LatticesView::Plane2D => Realm::Tilings,
                LatticesView::ThreeDAndSubperiodic => Realm::Groups,
                LatticesView::NonEuclidean | LatticesView::Honeycombs3D => Realm::Geometries,
                LatticesView::Manifolds | LatticesView::Flat3D => Realm::Manifolds,
                LatticesView::Curved2D if self.workbench.sub_index(WorkbenchView::S5Curved2d) <= 2 => {
                    Realm::Geometries
                }
                LatticesView::Curved2D => Realm::Manifolds,
            },
            AppMode::Defects => Realm::Forge,
            AppMode::Analysis => Realm::AnalysisLab,
        }
    }

    /// Index of the active view's card in `realm`'s current pill (unfiltered), if listed there.
    fn active_card_index(&mut self, realm: Realm) -> Option<usize> {
        let want = self.active_card_id()?;
        let pill = self.deck.pill[realm.idx()];
        let base = self.gallery.base_item().alias_key.clone();
        let cards = self.deck_cards(realm, pill, "");
        cards.iter().position(|c| match (c.id, want) {
            (CardId::Lattice(r), CardId::Lattice(_)) => {
                self.lattice_ref_item(r).is_some_and(|it| it.alias_key == base)
            }
            (a, b) => a == b,
        })
    }

    /// The card that leads to the view on screen (`Lattice(_)` stands for the selected lattice on the 2D Plane).
    fn active_card_id(&self) -> Option<CardId> {
        let wb = |v: WorkbenchView| CardId::Wb(v, u8::try_from(self.workbench.sub_index(v)).unwrap_or(0));
        Some(match self.active_mode {
            AppMode::Lattices => match self.lattices_view {
                LatticesView::Plane2D => CardId::Lattice(LatticeRef::Named(0)),
                LatticesView::ThreeDAndSubperiodic => CardId::AtlasPristine(self.atlas.sub_tab),
                LatticesView::NonEuclidean => CardId::HyperbolicAtlas,
                LatticesView::Manifolds => CardId::Surface(self.surfaces.target),
                LatticesView::Flat3D => wb(WorkbenchView::S4Flat3d),
                LatticesView::Curved2D => wb(WorkbenchView::S5Curved2d),
                LatticesView::Honeycombs3D => wb(WorkbenchView::S6Honeycombs),
            },
            AppMode::Defects => match self.defects_view {
                DefectsView::Catalogue2D => CardId::DefectCatalogue(self.defects.sub_tab),
                DefectsView::AtlasDefects => CardId::AtlasDefects(self.atlas.sub_tab),
                DefectsView::CombineAndFuse => CardId::Combine,
                DefectsView::PointAndBoundary => wb(WorkbenchView::S2Defects),
                DefectsView::DetectFields => wb(WorkbenchView::S3Detect),
            },
            AppMode::Analysis => CardId::Analysis(self.analysis.target),
        })
    }

    /// C1 check (smoke stage 9 and `analysis_hidden_everywhere`): every label source the user can see that names
    /// Analysis — the shown modes, the shown realm tabs and titles, the "All" cards of every shown realm (titles
    /// and subtitles), the shown presets and the HUD mode segment. Empty when Analysis is hidden.
    pub(crate) fn visible_analysis_labels(&mut self) -> Vec<String> {
        let mut seen: Vec<String> = AppMode::SHOWN.iter().map(|m| m.label().to_owned()).collect();
        for r in Realm::SHOWN {
            seen.push(r.title().to_owned());
            seen.push(r.tab_label().to_owned());
            for c in self.deck_cards(r, 0, "") {
                seen.push(c.title);
                seen.push(c.subtitle);
            }
        }
        seen.extend(crate::presets::PRESETS.iter().filter(|p| p.shown()).map(|p| p.name.to_owned()));
        seen.push(self.mode_segment_text());
        seen.retain(|s| s.contains("Analysis"));
        seen
    }
}
