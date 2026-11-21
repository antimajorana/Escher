//! Opening a card: route the app to what the card names, then close `All Views`. Every route goes through the
//! same functions the panels use (`select_lattices_view`, `select_defects_view`, `switch_mode`, ...).

use super::realms::{CardId, LatticeRef};
use crate::app::{AppMode, DefectsView, LatticesView, NavTab, PapertillingApp};
use crate::gallery_tab::BrowseMode;

impl PapertillingApp {
    /// Perform card `id`, then close `All Views`.
    pub(crate) fn deck_equip(&mut self, id: CardId, _ctx: &egui::Context) {
        match id {
            CardId::Plane2D => self.select_lattices_view(LatticesView::Plane2D),
            CardId::WallpaperGroups => self.equip_browse_mode(BrowseMode::ByGroup),
            CardId::CustomBuilder => self.equip_browse_mode(BrowseMode::CustomBuilder),
            CardId::HyperbolicAtlas => self.select_lattices_view(LatticesView::NonEuclidean),
            CardId::AtlasPristine(s) => {
                self.atlas.sub_tab = s;
                self.atlas.last_pristine_3d_subtab = s;
                self.atlas.dirty = true;
                self.select_lattices_view(LatticesView::ThreeDAndSubperiodic);
            }
            CardId::AtlasDefects(s) => {
                self.atlas.sub_tab = s;
                self.atlas.last_defect_subtab = s;
                self.atlas.dirty = true;
                self.select_defects_view(DefectsView::AtlasDefects);
            }
            CardId::DefectCatalogue(s) => {
                let was = (self.active_mode, self.active_tab) == (AppMode::Defects, NavTab::Defects);
                let changed = self.defects.sub_tab != s;
                self.defects.sub_tab = s;
                self.select_defects_view(DefectsView::Catalogue2D);
                if was && changed && s != crate::defects_tab::DefectSubTab::Combinations {
                    // activate_nav_tab returns early when nothing changed: rebuild as the left-panel click does
                    let item = self.gallery.active_item().clone();
                    self.defects.rebuild_crystal(&item, &self.backend, &mut self.canvas2d, true);
                }
            }
            CardId::Combine => self.select_defects_view(DefectsView::CombineAndFuse),
            CardId::Surface(t) => {
                self.surfaces.choose_target(t);
                self.surfaces.clamp_params(self.gallery.active_item());
                let builds = self.surfaces.syncs;
                self.select_lattices_view(LatticesView::Manifolds);
                if self.surfaces.syncs == builds {
                    // already on Manifolds (the view switch returned early): build the new target here, once
                    self.surfaces.activate_and_sync(
                        self.gallery.active_item(),
                        &self.backend,
                        &mut self.canvas2d,
                        &mut self.view3d,
                    );
                }
            }
            CardId::Wb(v, i) => self.enter_workbench(v, Some(usize::from(i))),
            CardId::Lattice(r) => {
                // a lattice cannot be shown in a view that does not draw the Gallery lattice
                let driven = self.view_is_lattice_driven();
                self.select_lattice_then(r, |app| {
                    if !driven {
                        app.select_lattices_view(LatticesView::Plane2D);
                    }
                });
            }
            // C1: Analysis is postponed; no shown section lists these cards, and opening one is refused
            CardId::Analysis(_) => self.export_status = String::from("Analysis is postponed"),
            CardId::Preset(i) => {
                if let Err(e) = self.apply_preset(i) {
                    self.export_status = e;
                }
            }
        }
        self.deck_close();
    }

    fn equip_browse_mode(&mut self, m: BrowseMode) {
        let lattice_changed = self.gallery.set_browse_mode(m, &self.backend, &mut self.canvas2d);
        self.select_lattices_view(LatticesView::Plane2D);
        if lattice_changed {
            self.resync_active_tab_for_lattice();
        }
    }

    /// Make the stored lattice `r` the Gallery selection (a no-op when it is already selected, which keeps its
    /// tuned parameters; a stale ref that no longer resolves is ignored), then `route` to the view that shows it.
    /// The tab being left is never reloaded for the new lattice: when `route` changes the view, the new view's
    /// activation loads the lattice once; only when it leaves the view as it was is the active tab resynced.
    fn select_lattice_then(&mut self, r: LatticeRef, route: impl FnOnce(&mut Self)) {
        let before = self.route_state();
        let changed = match self.lattice_ref_item(r).cloned() {
            Some(item) => self.select_lattice_quiet(item),
            None => false,
        };
        route(self);
        if changed && self.route_state() == before {
            self.resync_active_tab_for_lattice();
        }
    }

    fn route_state(&self) -> (AppMode, NavTab, LatticesView, DefectsView) {
        (self.active_mode, self.active_tab, self.lattices_view, self.defects_view)
    }

    /// Stat-card "Open in Defects": select the lattice, open the Defects catalogue on it, close `All Views`.
    pub(crate) fn deck_open_in_forge(&mut self, r: LatticeRef) {
        self.select_lattice_then(r, |app| app.select_defects_view(DefectsView::Catalogue2D));
        self.deck_close();
    }
}
