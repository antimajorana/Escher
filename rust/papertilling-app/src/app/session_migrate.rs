//! Session schema v1 -> v2 migration, Analysis restoration (C1: a saved Analysis view restores to Lattices with a
//! note, since Analysis is postponed), and Defects restoration.

use crate::analysis_mode::AnalysisTarget;
use crate::atlas_tab::AtlasSubTab;
use crate::defects_tab::DefectSubTab;
use crate::live_ops_ao::Defect2dMode;
use crate::mesh3d::sheet_modes::{DefectViewMode, Sheet3dColorMode};
use crate::session::{SessionState, parse_named};

use super::{AppMode, DefectsView, LatticesView, NavTab, PapertillingApp, item_key};

impl PapertillingApp {
    pub(super) fn apply_defects(&mut self, s: &crate::session::DefectsSession, notes: &mut Vec<String>) {
        let item = self.gallery.active_item().clone();
        if item_key(&item) != self.defects_seen {
            // As `switch_tab` would on opening Defects, reload now so the restored selection is kept.
            self.defects_seen = item_key(&item);
            self.defects
                .reload_for_item(&item, &self.backend, &mut self.canvas2d, false);
        }
        let d = &mut self.defects;
        match parse_named(&DefectSubTab::ALL, s.kind.as_deref()) {
            Some(k) => d.sub_tab = k,
            None if s.kind.is_some() => notes.push(String::from("unknown defect kind (kept current)")),
            None => {}
        }
        let wanted = [s.dislocation_index, s.disclination_index, s.wall_index];
        let cat_empty = d.catalogue.dislocations.is_empty()
            && d.catalogue.disclinations.is_empty()
            && d.catalogue.walls.is_empty();
        if cat_empty && wanted.iter().any(|i| i.is_some_and(|i| i > 0)) {
            // Only the lengths are needed: the build worker computes the same catalogue again.
            d.catalogue = self.backend.defect_catalogue(&item);
        }
        let lens = [
            d.catalogue.dislocations.len(),
            d.catalogue.disclinations.len(),
            d.catalogue.walls.len(),
        ];
        let slots = [&mut d.selected_disloc, &mut d.selected_disclin, &mut d.selected_wall];
        for ((want, len), slot) in wanted.into_iter().zip(lens).zip(slots) {
            match want {
                Some(i) if i == 0 || i < len => *slot = i,
                Some(i) => {
                    *slot = 0;
                    notes.push(format!("defect #{i} out of range (first one selected)"));
                }
                None => {}
            }
        }
        if let Some(h) = s.wall_half {
            d.wall_half = h;
        }
        if let Some(r) = s.radius.filter(|r| r.is_finite()) {
            d.radius = r.clamp(2.4, 8.0);
        }
        match parse_named(&Defect2dMode::DECODE, s.draw_mode.as_deref()) {
            // C1: the AO elastic view left the UI; a saved one opens as the relaxed drawing
            Some(Defect2dMode::ElasticAoMetric) => {
                d.draw_mode = Defect2dMode::RelaxedDrawing;
                notes.push(String::from("Elastic (AO metric) view removed; shown as Relaxed (drawing)"));
            }
            Some(m) => d.draw_mode = m,
            None if s.draw_mode.is_some() => {
                notes.push(String::from("unknown 2D drawing mode (kept current)"))
            }
            None => {
                if let Some(relaxed) = s.relaxed {
                    d.draw_mode = if relaxed { Defect2dMode::RelaxedDrawing } else { Defect2dMode::Raw };
                }
            }
        }
        d.relaxed = d.draw_mode != Defect2dMode::Raw;
        match parse_named(&DefectViewMode::ALL, s.view_mode.as_deref()) {
            Some(v) => d.view_mode = v,
            None if s.view_mode.is_some() => {
                notes.push(String::from("unknown 3D view mode (kept current)"))
            }
            None => {}
        }
        match parse_named(&Sheet3dColorMode::ALL, s.color_mode.as_deref()) {
            Some(c) => {
                let spectral = matches!(
                    c,
                    Sheet3dColorMode::SpectralLdos | Sheet3dColorMode::SpectralCharge
                );
                d.sheet_opts.color_mode = if spectral { Sheet3dColorMode::FaceRingSize } else { c };
                if spectral {
                    d.inspectors.overlay_mode = crate::canvas2d::gauge_overlay::CrystalOverlayMode::Standard;
                    notes.push(String::from(
                        "3D spectral colouring reset to ring size (spectral views are postponed)",
                    ));
                }
            }
            None if s.color_mode.is_some() => {
                notes.push(String::from("unknown 3D colour mode (kept current)"))
            }
            None => {}
        }
        d.view3d_dirty = true;
    }

    pub(super) fn apply_navigation_v1_or_v2(&mut self, st: &SessionState, notes: &mut Vec<String>) {
        match parse_named(&AtlasSubTab::ALL, st.atlas_sub_view.as_deref()) {
            Some(sub) => {
                self.atlas.sub_tab = sub;
                if sub != AtlasSubTab::HyperbolicH2 {
                    self.atlas.last_pristine_3d_subtab = sub;
                }
                if crate::atlas_tab::AtlasTabState::DEFECT_SUBTABS.contains(&sub) {
                    self.atlas.last_defect_subtab = sub;
                }
                self.atlas.dirty = true;
            }
            None if st.atlas_sub_view.is_some() => {
                notes.push(String::from("unknown Atlas sub-view (kept current)"))
            }
            None => {}
        }

        match parse_named(&AnalysisTarget::ALL, st.analysis_target.as_deref()) {
            Some(t) => self.analysis.target = t,
            None if st.analysis_target.is_some() => {
                notes.push(String::from("unknown analysis target (kept current)"))
            }
            None => {}
        }
        match parse_named(&DefectViewMode::ALL, st.analysis_view.as_deref()) {
            Some(v) => self.analysis.view_mode = v,
            None if st.analysis_view.is_some() => {
                notes.push(String::from("unknown analysis view mode (kept current)"))
            }
            None => {}
        }
        self.analysis.reset_transient_for_session();
        // before routing: the routed view wins over the JSON "view", and entering it uses the restored panel state
        self.workbench.restore(st.workbench.as_ref(), notes);

        let use_v2_mode = st.schema_version >= 2 && st.active_mode.is_some();
        if use_v2_mode {
            match parse_named(&LatticesView::ALL, st.lattices_view.as_deref()) {
                Some(v) => self.lattices_view = v,
                None if st.lattices_view.is_some() => {
                    notes.push(String::from("unknown Lattices view (kept current)"))
                }
                None => {}
            }
            match parse_named(&DefectsView::ALL, st.defects_view.as_deref()) {
                Some(v) => self.defects_view = v,
                None if st.defects_view.is_some() => {
                    notes.push(String::from("unknown Defects view (kept current)"))
                }
                None => {}
            }
            match parse_named(&AppMode::ALL, st.active_mode.as_deref()) {
                Some(AppMode::Lattices) => self.select_lattices_view(self.lattices_view),
                Some(AppMode::Defects) => self.select_defects_view(self.defects_view),
                Some(AppMode::Analysis) => self.restore_postponed_analysis(notes),
                None => {
                    notes.push(String::from("unknown mode (Gallery shown)"));
                    self.select_lattices_view(LatticesView::Plane2D);
                }
            }
        } else {
            // Migrate schema v1 `active_tab` + `atlas_sub_view` into the three-mode shell.
            match parse_named(&NavTab::ALL, st.active_tab.as_deref()) {
                Some(NavTab::Gallery) | None => {
                    if st.active_tab.is_some() {
                        notes.push(String::from("unknown tab (Gallery shown)"));
                    }
                    self.select_lattices_view(LatticesView::Plane2D);
                }
                Some(NavTab::Surfaces) => self.select_lattices_view(LatticesView::Manifolds),
                Some(NavTab::Defects) => self.select_defects_view(DefectsView::Catalogue2D),
                Some(NavTab::Combine) => self.select_defects_view(DefectsView::CombineAndFuse),
                Some(NavTab::Fuse) => self.select_defects_view(DefectsView::FuseAndCompare),
                Some(NavTab::Analysis) => self.restore_postponed_analysis(notes),
                // never written by a v1 app (the tab did not exist); listed because the match is exhaustive
                Some(NavTab::Workbench) => self.select_lattices_view(LatticesView::Plane2D),
                Some(NavTab::Atlas) => {
                    if self.atlas.sub_tab == AtlasSubTab::HyperbolicH2 {
                        self.select_lattices_view(LatticesView::NonEuclidean);
                    } else {
                        self.select_lattices_view(LatticesView::ThreeDAndSubperiodic);
                    }
                }
            }
        }
        // Command Deck (schema 3): its last realm, else the realm of the restored view; always loads closed. A saved
        // realm is where the next Tab / Codex opens the deck (`deck_entry_realm`, once).
        if !self.deck.restore(st.deck.as_ref(), notes) {
            self.deck.realm = self.current_realm();
            self.deck.resume_saved_realm = false;
        }
        self.deck.open = false;
        match st.advanced {
            Some(on) => self.set_advanced(on),
            None => {
                // Legacy v1-v3 session written before `advanced`: keep Advanced on iff the restored view requires it.
                self.advanced = match self.active_mode {
                    AppMode::Lattices => !self.lattices_view.is_default(),
                    AppMode::Defects => !self.defects_view.is_default(),
                    AppMode::Analysis => false,
                };
            }
        }
    }

    /// C1: a session saved in Analysis opens the Lattices 2D Plane, with a note (the internal
    /// `switch_mode(Analysis)` route still exists; no user input reaches it).
    fn restore_postponed_analysis(&mut self, notes: &mut Vec<String>) {
        notes.push(String::from(crate::game_menu::ANALYSIS_POSTPONED_NOTE));
        self.select_lattices_view(LatticesView::Plane2D);
    }
}
