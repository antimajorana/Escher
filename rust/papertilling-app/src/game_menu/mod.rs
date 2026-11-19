//! The Codex: a one-row HUD (`hud_bar`) over the viewport and a full-canvas Command Deck overlay (`deck`) that
//! replace the old mode and sub-view strips. Seven realms group every view, sub-tab and stage sub-row as cards
//! (`realms`); equipping a card routes the app there (`equip`). Keyboard: Tab / M open and close the deck, 1-7 and
//! Q / E switch realms, arrows / WASD move, Enter equips, Esc resumes (`keys`).
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

pub(crate) use realms::{CardId, LatticeRef};

/// `id_salt` of the deck's filter TextEdit (its real `Id` is recorded in [`DeckState::filter_id`]).
pub(crate) const DECK_FILTER_ID: &str = "deck_filter";

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

    /// 1..=7 (the realm's number key).
    pub(crate) fn number(self) -> usize {
        self as usize + 1
    }

    pub(crate) fn from_number(n: usize) -> Option<Self> {
        n.checked_sub(1).and_then(|i| Self::ALL.get(i).copied())
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

    /// Previous realm, wrapping (Q).
    pub(crate) fn prev(self) -> Self {
        Self::ALL[(self as usize + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    /// Next realm, wrapping (E).
    pub(crate) fn next(self) -> Self {
        Self::ALL[(self as usize + 1) % Self::ALL.len()]
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
}

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
            Some(r) => {
                self.realm = r;
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
        self.deck.realm = realm;
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

    pub(crate) fn deck_close(&mut self) {
        self.deck.open = false;
        self.deck.filter.clear();
        self.deck.focus_filter = false;
        self.deck.equip_request = None;
        self.deck.slash_text_pending = false;
    }

    pub(crate) fn deck_select_realm(&mut self, realm: Realm) {
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
}
