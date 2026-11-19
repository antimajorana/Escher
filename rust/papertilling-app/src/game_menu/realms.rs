//! Card inventory of the seven realms (the single source of truth for what the Command Deck lists). Cards exist
//! only for capabilities the app has now; every card names an existing view, sub-tab, sub-row, preset or export.

use super::Realm;
use super::badges::Badge;
use crate::analysis_mode::AnalysisTarget;
use crate::app::PapertillingApp;
use crate::atlas_tab::{AtlasSubTab, AtlasTabState};
use crate::backend::{CatalogueItemUi, CatalogueRuleUi, SurfaceTargetUi};
use crate::defects_tab::DefectSubTab;
use crate::presets::PRESETS;
use crate::workbench::WorkbenchView;

/// Most catalogue cards listed at once (realm 3 "This group" and search results).
pub(crate) const MAX_LATTICE_CARDS: usize = 200;

/// A stored lattice as the Quick Lattice Switcher lists it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LatticeRef {
    Recent(usize),
    Named(usize),
    Full(CatalogueRuleUi, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IoAction {
    SaveSession,
    OpenSession,
    ExportSvgPng,
    Export3dPng,
    ExportObj,
    ExportJson,
    ImportJson,
}

impl IoAction {
    pub(crate) const ALL: [Self; 7] = [
        Self::SaveSession,
        Self::OpenSession,
        Self::ExportSvgPng,
        Self::Export3dPng,
        Self::ExportObj,
        Self::ExportJson,
        Self::ImportJson,
    ];

    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::SaveSession => "Save session…",
            Self::OpenSession => "Open session…",
            Self::ExportSvgPng => "Export SVG / PNG…",
            Self::Export3dPng => "Export 3D PNG…",
            Self::ExportObj => "Export .obj (3D)…",
            Self::ExportJson => "Export .json (Notebook 3)…",
            Self::ImportJson => "Import .json…",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CardId {
    Plane2D,
    HyperbolicAtlas,
    WallpaperGroups,
    CustomBuilder,
    Combine,
    AtlasPristine(AtlasSubTab),
    AtlasDefects(AtlasSubTab),
    DefectCatalogue(DefectSubTab),
    Surface(SurfaceTargetUi),
    Wb(WorkbenchView, u8),
    Lattice(LatticeRef),
    Analysis(AnalysisTarget),
    Preset(usize),
    Io(IoAction),
}

pub(crate) struct Card {
    pub id: CardId,
    pub realm: Realm,
    pub title: String,
    #[allow(dead_code)] // M1 prewire: drawn by the card grid and stat card (WP-B / WP-C).
    pub subtitle: String,
    #[allow(dead_code)] // M1 prewire: drawn by the card grid and stat card (WP-B / WP-C).
    pub badge: Option<Badge>,
    #[allow(dead_code)] // M1 prewire: the lattice-card thumbnail is drawn by the deck grid (WP-B).
    pub thumb: Option<LatticeRef>,
}

impl Card {
    pub(crate) fn new(id: CardId, title: impl Into<String>) -> Self {
        let thumb = match id {
            CardId::Lattice(r) => Some(r),
            _ => None,
        };
        Self { id, realm: card_realm(id), title: title.into(), subtitle: String::new(), badge: None, thumb }
    }

    /// Accessible label of the card (realm number + title), unique within its realm.
    pub(crate) fn a11y_label(&self) -> String {
        format!("{} · {}", self.realm.number(), self.title)
    }
}

/// Pill row of `realm` (index 0 is "All" whenever there is more than one pill).
pub(crate) fn pills(realm: Realm) -> &'static [&'static str] {
    match realm {
        Realm::Tilings => &["All", "Named 21", "This group", "Custom"],
        Realm::Manifolds => &["All", "Surfaces", "Flat 3D"],
        Realm::Forge => &["All", "Catalogue", "Combine", "Point, SW & GB", "Detect & Fields"],
        Realm::PresetsIo => &["All", "Showcase presets", "Session & files"],
        Realm::Geometries | Realm::Groups | Realm::AnalysisLab => &["All"],
    }
}

/// The realm that lists card `id` (total).
pub(crate) fn card_realm(id: CardId) -> Realm {
    match id {
        CardId::Plane2D | CardId::HyperbolicAtlas => Realm::Geometries,
        CardId::WallpaperGroups | CardId::AtlasPristine(_) => Realm::Groups,
        CardId::CustomBuilder | CardId::Lattice(_) => Realm::Tilings,
        CardId::Surface(_) => Realm::Manifolds,
        CardId::Combine | CardId::AtlasDefects(_) | CardId::DefectCatalogue(_) => Realm::Forge,
        CardId::Wb(v, i) => match v {
            WorkbenchView::S2Defects | WorkbenchView::S3Detect => Realm::Forge,
            WorkbenchView::S4Flat3d => Realm::Manifolds,
            WorkbenchView::S5Curved2d if i <= 2 => Realm::Geometries,
            WorkbenchView::S5Curved2d => Realm::Manifolds,
            WorkbenchView::S6Honeycombs => Realm::Geometries,
        },
        CardId::Analysis(_) => Realm::AnalysisLab,
        CardId::Preset(_) | CardId::Io(_) => Realm::PresetsIo,
    }
}

/// `label()` with its first letter uppercased (`torus` -> `Torus`).
fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

impl PapertillingApp {
    /// Cards of `realm` under pill `pill`, filtered by `filter` (titles only; WP-C adds subtitles and badges).
    pub(crate) fn deck_cards(&mut self, realm: Realm, pill: usize, filter: &str) -> Vec<Card> {
        let q = filter.trim();
        if realm == Realm::Tilings && !q.is_empty() {
            let refs = self.deck_search_lattices(q);
            return refs.into_iter().filter_map(|r| self.lattice_card(r)).collect();
        }
        let cards = self.unfiltered_cards(realm, pill);
        if q.is_empty() {
            return cards;
        }
        cards
            .into_iter()
            .filter(|c| {
                let item = match c.id {
                    CardId::Lattice(r) => self.lattice_ref_item(r),
                    _ => None,
                };
                super::search::filter_matches(c, item, q)
            })
            .collect()
    }

    /// The stored lattice a [`LatticeRef`] points at.
    pub(crate) fn lattice_ref_item(&self, r: LatticeRef) -> Option<&CatalogueItemUi> {
        match r {
            LatticeRef::Recent(i) => self.switcher.recent.get(i),
            LatticeRef::Named(i) => self.backend.named_lattices().get(i),
            LatticeRef::Full(rule, i) => self.gallery.rule_index_ready(rule).get(i),
        }
    }

    fn lattice_card(&self, r: LatticeRef) -> Option<Card> {
        let item = self.lattice_ref_item(r)?;
        Some(Card::new(CardId::Lattice(r), item.label.clone()))
    }

    fn wb_cards(&self, view: WorkbenchView, rows: std::ops::RangeInclusive<usize>) -> Vec<Card> {
        let labels = self.workbench.sub_labels(view);
        rows.filter_map(|i| {
            let label = labels.get(i)?;
            Some(Card::new(CardId::Wb(view, u8::try_from(i).ok()?), *label))
        })
        .collect()
    }

    fn unfiltered_cards(&mut self, realm: Realm, pill: usize) -> Vec<Card> {
        let mut out = Vec::new();
        match realm {
            Realm::Geometries => {
                out.push(Card::new(CardId::Plane2D, "E² Euclidean plane"));
                out.push(Card::new(CardId::HyperbolicAtlas, "H² hyperbolic tiling (Atlas)"));
                out.extend(self.wb_cards(WorkbenchView::S5Curved2d, 0..=2));
                out.extend(self.wb_cards(WorkbenchView::S6Honeycombs, 0..=3));
            }
            Realm::Groups => {
                out.push(Card::new(CardId::WallpaperGroups, "17 wallpaper groups"));
                for s in AtlasTabState::PRISTINE_3D_SUBTABS {
                    out.push(Card::new(CardId::AtlasPristine(s), s.label()));
                }
            }
            Realm::Tilings => out = self.tiling_cards(pill),
            Realm::Manifolds => {
                if pill != 2 {
                    for t in SurfaceTargetUi::ALL {
                        out.push(Card::new(CardId::Surface(t), capitalised(t.label())));
                    }
                    out.extend(self.wb_cards(WorkbenchView::S5Curved2d, 3..=4));
                }
                if pill != 1 {
                    out.extend(self.wb_cards(WorkbenchView::S4Flat3d, 0..=5));
                }
            }
            Realm::Forge => {
                if matches!(pill, 0 | 1) {
                    for s in DefectSubTab::ALL {
                        out.push(Card::new(CardId::DefectCatalogue(s), s.label()));
                    }
                    for s in AtlasTabState::DEFECT_SUBTABS {
                        out.push(Card::new(CardId::AtlasDefects(s), s.defect_label()));
                    }
                }
                if matches!(pill, 0 | 2) {
                    out.push(Card::new(CardId::Combine, "Combine, Fuse & Move"));
                }
                if matches!(pill, 0 | 3) {
                    out.extend(self.wb_cards(WorkbenchView::S2Defects, 0..=4));
                }
                if matches!(pill, 0 | 4) {
                    out.extend(self.wb_cards(WorkbenchView::S3Detect, 0..=3));
                }
            }
            Realm::AnalysisLab => {
                for t in AnalysisTarget::ALL {
                    out.push(Card::new(CardId::Analysis(t), t.label()));
                }
            }
            Realm::PresetsIo => {
                if pill != 2 {
                    for (i, p) in PRESETS.iter().enumerate() {
                        out.push(Card::new(CardId::Preset(i), p.name));
                    }
                }
                if pill != 1 {
                    for a in IoAction::ALL {
                        out.push(Card::new(CardId::Io(a), a.title()));
                    }
                }
            }
        }
        out
    }

    /// Realm 3: All = Named 21, then Recent not already listed, then the builder; This group = the full catalogue
    /// (all rules) of the selected lattice's wallpaper group; Custom = the builder.
    fn tiling_cards(&mut self, pill: usize) -> Vec<Card> {
        let builder = || Card::new(CardId::CustomBuilder, "Custom lattice builder");
        let named = |app: &Self| -> Vec<Card> {
            (0..app.backend.named_lattices().len()).filter_map(|i| app.lattice_card(LatticeRef::Named(i))).collect()
        };
        match pill {
            1 => named(self),
            2 => {
                let group = self.gallery.base_item().group.clone();
                let mut out = Vec::new();
                for rule in CatalogueRuleUi::ALL {
                    let list = self.gallery.rule_index(rule, &self.backend);
                    let hits: Vec<usize> =
                        list.iter().enumerate().filter(|(_, it)| it.group == group).map(|(i, _)| i).collect();
                    for i in hits {
                        if out.len() >= MAX_LATTICE_CARDS {
                            return out;
                        }
                        if let Some(c) = self.lattice_card(LatticeRef::Full(rule, i)) {
                            out.push(c);
                        }
                    }
                }
                out
            }
            3 => vec![builder()],
            _ => {
                let mut out = named(self);
                let listed: Vec<_> = self.backend.named_lattices().iter().map(|it| it.alias_key.clone()).collect();
                for (i, it) in self.switcher.recent.iter().enumerate() {
                    if !listed.contains(&it.alias_key) {
                        out.push(Card::new(CardId::Lattice(LatticeRef::Recent(i)), it.label.clone()));
                    }
                }
                out.push(builder());
                out
            }
        }
    }
}
