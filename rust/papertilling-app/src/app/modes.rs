//! Three-mode shell state (Lattices, Defects, Analysis), sub-views, and tab transitions (routed by the Command
//! Deck, `crate::game_menu`).

use crate::analysis_mode::AnalysisTarget;
use crate::atlas_tab::{AtlasSubTab, AtlasTabState};
use crate::workbench::ModeView;

use super::{PapertillingApp, item_key};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavTab {
    Gallery,
    Surfaces,
    Defects,
    Combine,
    Atlas,
    Analysis,
    /// Internal route for the S2..S6 panels (`crate::workbench`), never shown: the user reaches them through
    /// their Command Deck cards.
    Workbench,
}

impl NavTab {
    pub(crate) const ALL: [Self; 7] = [
        Self::Gallery,
        Self::Surfaces,
        Self::Defects,
        Self::Combine,
        Self::Atlas,
        Self::Analysis,
        Self::Workbench,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AppMode {
    #[default]
    Lattices,
    Defects,
    Analysis,
}

impl AppMode {
    pub const ALL: [Self; 3] = [Self::Lattices, Self::Defects, Self::Analysis];
    /// The modes a user can reach (C1: Analysis is postponed; `ALL` keeps it for session decoding and the
    /// internal `switch_mode(Analysis)` route, which no user input calls any more).
    pub const SHOWN: [Self; 2] = [Self::Lattices, Self::Defects];

    pub fn label(self) -> &'static str {
        match self {
            Self::Lattices => "Lattice",
            Self::Defects => "Defects",
            Self::Analysis => "Analysis",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LatticesView {
    #[default]
    Plane2D,
    ThreeDAndSubperiodic,
    NonEuclidean,
    Manifolds,
    Flat3D,
    Curved2D,
    Honeycombs3D,
}

impl LatticesView {
    pub const ALL: [Self; 7] = [
        Self::Plane2D,
        Self::ThreeDAndSubperiodic,
        Self::NonEuclidean,
        Self::Manifolds,
        Self::Flat3D,
        Self::Curved2D,
        Self::Honeycombs3D,
    ];

    /// Views reachable in the two-mode Default view (`Advanced` off).
    pub fn is_default(self) -> bool {
        matches!(self, Self::Plane2D | Self::Manifolds)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Plane2D => "2D Plane",
            Self::ThreeDAndSubperiodic => "3D & Subperiodic",
            Self::NonEuclidean => "Non-Euclidean",
            Self::Manifolds => "Manifolds & Surfaces",
            Self::Flat3D => "Flat 3D: Tubes, Ribbons, Flakes",
            Self::Curved2D => "{p,q} on S² / E² / H²",
            Self::Honeycombs3D => "Honeycombs & TPMS",
        }
    }

    pub fn nav_tab(self) -> NavTab {
        match self {
            Self::Plane2D => NavTab::Gallery,
            Self::ThreeDAndSubperiodic | Self::NonEuclidean => NavTab::Atlas,
            Self::Manifolds => NavTab::Surfaces,
            Self::Flat3D | Self::Curved2D | Self::Honeycombs3D => NavTab::Workbench,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DefectsView {
    #[default]
    Catalogue2D,
    AtlasDefects,
    CombineAndFuse,
    PointAndBoundary,
    DetectFields,
    FusionSteps,
}

impl DefectsView {
    pub const ALL: [Self; 6] = [
        Self::Catalogue2D,
        Self::AtlasDefects,
        Self::CombineAndFuse,
        Self::PointAndBoundary,
        Self::DetectFields,
        Self::FusionSteps,
    ];

    /// Views reachable in the two-mode Default view (`Advanced` off).
    pub fn is_default(self) -> bool {
        matches!(self, Self::Catalogue2D | Self::CombineAndFuse | Self::FusionSteps)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Catalogue2D => "Catalogue (2D)",
            Self::AtlasDefects => "0D–1D & H² Defects",
            Self::CombineAndFuse => "Combine, Fuse & Move",
            Self::PointAndBoundary => "Point, Stone-Wales & GB",
            Self::DetectFields => "Detect & Fields",
            Self::FusionSteps => "Fusion steps (graphs)",
        }
    }

    pub fn nav_tab(self) -> NavTab {
        match self {
            Self::Catalogue2D => NavTab::Defects,
            Self::AtlasDefects => NavTab::Atlas,
            Self::CombineAndFuse => NavTab::Combine,
            Self::PointAndBoundary | Self::DetectFields | Self::FusionSteps => NavTab::Workbench,
        }
    }
}

impl PapertillingApp {
    /// Set the `Advanced` toggle; turning it off closes `All Views`, exits the Custom Wyckoff builder, and snaps any
    /// Advanced-only view back to the corresponding Default view (`Plane2D` or `Catalogue2D`).
    pub fn set_advanced(&mut self, on: bool) {
        self.advanced = on;
        if on {
            return;
        }
        self.deck_close();
        if self.gallery.exit_custom_builder(&self.backend, &mut self.canvas2d)
            && self.active_tab != NavTab::Gallery
        {
            self.resync_active_tab_for_lattice();
        }
        if !self.lattices_view.is_default() {
            self.lattices_view = LatticesView::Plane2D;
            if self.active_mode == AppMode::Lattices {
                self.select_lattices_view(LatticesView::Plane2D);
            }
        }
        if !self.defects_view.is_default() {
            self.defects_view = DefectsView::Catalogue2D;
            if self.active_mode == AppMode::Defects {
                self.select_defects_view(DefectsView::Catalogue2D);
            }
        }
        if self.active_mode == AppMode::Analysis {
            self.select_lattices_view(LatticesView::Plane2D);
        }
    }

    pub(crate) fn atlas_offered_tabs(&self) -> &'static [AtlasSubTab] {
        match (self.active_mode, self.lattices_view) {
            (AppMode::Lattices, LatticesView::NonEuclidean) => &[AtlasSubTab::HyperbolicH2],
            (AppMode::Lattices, _) => &AtlasTabState::PRISTINE_3D_SUBTABS,
            _ => &AtlasTabState::DEFECT_SUBTABS,
        }
    }

    pub(crate) fn switch_mode(&mut self, mode: AppMode) {
        match mode {
            AppMode::Lattices => self.select_lattices_view(self.lattices_view),
            AppMode::Defects => self.select_defects_view(self.defects_view),
            AppMode::Analysis => {
                let prev_tab = self.active_tab;
                self.active_mode = AppMode::Analysis;
                self.activate_nav_tab(NavTab::Analysis, prev_tab != NavTab::Analysis);
            }
        }
    }

    pub(crate) fn select_lattices_view(&mut self, view: LatticesView) {
        let prev_mode = self.active_mode;
        let prev_view = self.lattices_view;
        let prev_tab = self.active_tab;
        self.active_mode = AppMode::Lattices;
        self.lattices_view = view;
        let wb_changed = self.route_workbench_view(view.workbench_view());
        let view_changed =
            prev_mode != AppMode::Lattices || prev_view != view || prev_tab != view.nav_tab() || wb_changed;
        match view {
            LatticesView::ThreeDAndSubperiodic => {
                self.atlas.set_pristine_only(true);
                if self.atlas.sub_tab == AtlasSubTab::HyperbolicH2 {
                    self.atlas.sub_tab = self.atlas.last_pristine_3d_subtab;
                    self.atlas.dirty = true;
                } else if view_changed {
                    self.atlas.dirty = true;
                }
            }
            LatticesView::NonEuclidean => {
                self.atlas.set_pristine_only(true);
                if self.atlas.sub_tab != AtlasSubTab::HyperbolicH2 || view_changed {
                    self.atlas.sub_tab = AtlasSubTab::HyperbolicH2;
                    self.atlas.dirty = true;
                }
            }
            LatticesView::Plane2D
            | LatticesView::Manifolds
            | LatticesView::Flat3D
            | LatticesView::Curved2D
            | LatticesView::Honeycombs3D => {}
        }
        let target_tab = view.nav_tab();
        self.activate_nav_tab(target_tab, view_changed);
    }

    pub(crate) fn select_defects_view(&mut self, view: DefectsView) {
        let prev_mode = self.active_mode;
        let prev_view = self.defects_view;
        let prev_tab = self.active_tab;
        let from_outside = prev_mode != AppMode::Defects || prev_view != DefectsView::AtlasDefects;
        self.active_mode = AppMode::Defects;
        self.defects_view = view;
        if view == DefectsView::AtlasDefects {
            self.atlas.enter_defects_mode(from_outside);
        }
        let wb_changed = self.route_workbench_view(view.workbench_view());
        let target_tab = view.nav_tab();
        let changed = prev_mode != AppMode::Defects || prev_view != view || prev_tab != target_tab || wb_changed;
        self.activate_nav_tab(target_tab, changed);
    }

    pub(crate) fn drain_combine_fusion_cavity(&mut self) {
        if let Some(req) = self.combine.take_open_cavity_in_fusion() {
            self.workbench.load_fusion_cavity(req);
            self.select_defects_view(DefectsView::FusionSteps);
        }
    }

    pub(crate) fn switch_tab(&mut self, tab: NavTab) {
        match tab {
            NavTab::Gallery => self.select_lattices_view(LatticesView::Plane2D),
            NavTab::Surfaces => self.select_lattices_view(LatticesView::Manifolds),
            NavTab::Defects => self.select_defects_view(DefectsView::Catalogue2D),
            NavTab::Combine => self.select_defects_view(DefectsView::CombineAndFuse),
            NavTab::Analysis => self.switch_mode(AppMode::Analysis),
            NavTab::Workbench => match self.workbench.view.mode_view() {
                ModeView::Lattices(v) => self.select_lattices_view(v),
                ModeView::Defects(v) => self.select_defects_view(v),
            },
            NavTab::Atlas => {
                if self.atlas.pristine_only {
                    let view = if self.atlas.sub_tab == AtlasSubTab::HyperbolicH2 {
                        LatticesView::NonEuclidean
                    } else {
                        LatticesView::ThreeDAndSubperiodic
                    };
                    self.select_lattices_view(view);
                } else {
                    self.select_defects_view(DefectsView::AtlasDefects);
                }
            }
        }
    }

    pub(crate) fn activate_nav_tab(&mut self, tab: NavTab, changed: bool) {
        if !changed
            && (tab != NavTab::Atlas || !self.atlas.dirty)
            && (tab != NavTab::Workbench || !self.workbench.dirty)
        {
            return;
        }
        let prev_tab = self.active_tab;
        // A background parameter move still pending is applied now so the new tab gets the final lattice once.
        self.gallery.finish_pending_retune(&mut self.canvas2d);
        if prev_tab == NavTab::Analysis && tab != NavTab::Analysis {
            self.canvas2d.set_gauge_overlay(None);
            self.view3d.set_legend(None);
        }
        self.active_tab = tab;
        match tab {
            NavTab::Gallery => {
                self.view3d.set_legend(None);
                self.gallery.sync_canvas(&self.backend, &mut self.canvas2d, true);
            }
            NavTab::Surfaces => {
                self.surfaces.activate_and_sync(
                    self.gallery.active_item(),
                    &self.backend,
                    &mut self.canvas2d,
                    &mut self.view3d,
                );
            }
            NavTab::Defects => {
                let item = self.gallery.active_item().clone();
                if item_key(&item) == self.defects_seen {
                    self.defects
                        .rebuild_crystal(&item, &self.backend, &mut self.canvas2d, true);
                } else {
                    self.defects_seen = item_key(&item);
                    self.defects
                        .reload_for_item(&item, &self.backend, &mut self.canvas2d, true);
                }
                self.defects.sync_view3d(&mut self.view3d);
            }
            NavTab::Combine => {
                let item = self.gallery.active_item().clone();
                if !self.apply_pending_combine() {
                    if item_key(&item) == self.combine_seen {
                        self.combine.ensure_compared(&item, &self.backend);
                        self.combine.sync_to_canvas(&item, &mut self.canvas2d, true);
                    } else {
                        self.combine_seen = item_key(&item);
                        self.combine
                            .reload_for_item(&item, &self.backend, &mut self.canvas2d, true);
                    }
                    self.combine.sync_view3d(&item, &mut self.view3d);
                }
            }
            NavTab::Atlas => {
                self.canvas2d.set_gauge_overlay(None);
                self.view3d.set_legend(None);
                self.atlas.sync_scenes(&mut self.canvas2d, &mut self.view3d);
            }
            NavTab::Analysis => {
                self.ensure_analysis_sources_synced();
                self.analysis.on_enter();
                self.analysis.sync_canvas(&self.surfaces, &self.defects, &self.combine, &mut self.canvas2d);
                self.analysis.sync_view3d(&self.surfaces, &self.defects, &self.combine, &mut self.view3d);
            }
            NavTab::Workbench => {
                self.canvas2d.set_gauge_overlay(None);
                self.view3d.set_legend(None);
                self.workbench_defect_source_synced();
                self.workbench_call(|wb, cx, canvas, view3d| wb.on_enter(cx, canvas, view3d));
            }
        }
    }

    pub(crate) fn analysis_source_busy(&self) -> bool {
        match self.analysis.target {
            AnalysisTarget::PlaneLattice => false,
            AnalysisTarget::Surface => self.surfaces.is_building() || self.surfaces.elastic.is_busy(),
            AnalysisTarget::SingleDefect => self.defects.is_busy(),
            AnalysisTarget::CombinedDefects => {
                self.combine.is_building() || self.combine.dragging || self.combine.cut_dragging
            }
        }
    }

    pub(crate) fn ensure_analysis_sources_synced(&mut self) {
        let item = self.gallery.active_item().clone();
        let k = item_key(&item);
        match self.analysis.target {
            AnalysisTarget::PlaneLattice => {
                self.analysis.ensure_plane_source(&item);
            }
            AnalysisTarget::Surface => {
                if self.surfaces.needs_sync_for(&item) {
                    self.surfaces.build_quiet_for(&item);
                }
            }
            AnalysisTarget::SingleDefect if k != self.defects_seen => {
                self.defects_seen = k;
                self.defects.reload_for_item(&item, &self.backend, &mut self.canvas2d, false);
                self.defects.submit_rebuild(&item, true);
            }
            AnalysisTarget::CombinedDefects if k != self.combine_seen => {
                self.combine_seen = k;
                self.combine.reload_for_item(&item, &self.backend, &mut self.canvas2d, false);
            }
            _ => {}
        }
    }
}
