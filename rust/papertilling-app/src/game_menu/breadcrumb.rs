//! HUD breadcrumb (geometry ▸ group ▸ lattice ▸ mode): pure functions of the app state, no drawing (§4.4).
//!
//! The group and lattice segments appear only in lattice-driven views (`view_is_lattice_driven`), where the
//! Gallery lattice is what the viewport shows; elsewhere (Atlas, H², {p,q}, honeycombs, atlas defects) the
//! breadcrumb does not claim it. No vertex symbol: `coordination` holds vertex degrees only.

use std::fmt::Write as _;

use papertilling_core::noneuclidean2d::wallpaper_to_orbifold;

use super::Realm;
use crate::analysis_mode::AnalysisTarget;
use crate::app::{AppMode, DefectsView, LatticesView, PapertillingApp};
use crate::atlas_tab::AtlasSubTab;
use crate::gallery_tab::browser::rule_tag;
use crate::gallery_tab::filter::ita_number;
use crate::workbench::WorkbenchView;

/// One clickable breadcrumb segment: its text, hover text, and the realm a click opens.
pub(crate) struct Segment {
    pub text: String,
    pub hover: String,
    pub realm: Realm,
}

/// The four segments; `group` and `lattice` only when the view shows the Gallery lattice.
pub(crate) struct Breadcrumb {
    pub geometry: Segment,
    pub group: Option<Segment>,
    pub lattice: Option<Segment>,
    pub mode: Segment,
}

/// Geometry text of an Atlas sub-tab (Lattices / 3D & Subperiodic and Defects / Atlas Defects).
fn atlas_geometry(sub: AtlasSubTab) -> &'static str {
    match sub {
        AtlasSubTab::Subperiodic => "Subperiodic (0D/1D)",
        AtlasSubTab::Layers80 => "Layers (E³ slab)",
        AtlasSubTab::SpaceGroups3D => "E³ Space",
        AtlasSubTab::HyperbolicH2 => "H² Hyperbolic",
        AtlasSubTab::HyperbolicH3 => "H³ Hyperbolic",
    }
}

/// Hover line shared by every segment in Advanced mode: what a click does.
fn click_hint(realm: Realm) -> String {
    format!("Click: open All views at {} {}", realm.number(), realm.title())
}

impl PapertillingApp {
    /// Text of the mode segment (`Lattice: 2D Plane`, `Defects: …`, `Analysis: …`).
    pub(crate) fn mode_segment_text(&self) -> String {
        match self.active_mode {
            AppMode::Lattices => format!("Lattice: {}", self.lattices_view.label()),
            AppMode::Defects => format!("Defects: {}", self.defects_view.label()),
            AppMode::Analysis => format!("Analysis: {}", self.analysis.target.label()),
        }
    }

    /// The §4.4 geometry table (total).
    fn geometry_text(&self) -> &'static str {
        match self.active_mode {
            AppMode::Lattices => match self.lattices_view {
                LatticesView::Plane2D => "E² Plane",
                LatticesView::ThreeDAndSubperiodic | LatticesView::NonEuclidean => atlas_geometry(self.atlas.sub_tab),
                LatticesView::Manifolds => "Closed surfaces",
                LatticesView::Flat3D => "E³ Flat 3D",
                LatticesView::Curved2D => {
                    self.workbench.breadcrumb_geometry(WorkbenchView::S5Curved2d).unwrap_or("Constant curvature")
                }
                LatticesView::Honeycombs3D => "S³ / E³ / H³ Honeycombs",
            },
            AppMode::Defects => match self.defects_view {
                DefectsView::AtlasDefects => atlas_geometry(self.atlas.sub_tab),
                DefectsView::FusionSteps => "Lattice graph",
                _ => "E² Plane",
            },
            AppMode::Analysis => match self.analysis.target {
                AnalysisTarget::Surface => "Closed surfaces",
                _ => "E² Plane",
            },
        }
    }

    /// Group segment: `p6m (*632)`, hover with the ITA number and Conway orbifold.
    /// On a special locus of the sliders the tuned lattice has a larger group than its stored one: the segment
    /// then reads `p4g (4*2) → p4m` and the hover says why.
    fn group_segment(&self) -> Segment {
        let g = self.gallery.active_item().group.as_str();
        let orb = wallpaper_to_orbifold(g).ok();
        let mut text = match orb {
            Some(o) => format!("{g} ({o})"),
            None => g.to_string(),
        };
        let ro = &self.gallery.readout;
        let raised = (!ro.status_ok && !ro.active_group.is_empty() && ro.active_group != g)
            .then_some(ro.active_group.as_str());
        if let Some(big) = raised {
            let _ = write!(text, " → {big}");
        }
        let mut hover = format!("Wallpaper group {g}");
        if let Some(n) = ita_number(g) {
            let _ = write!(hover, " · ITA {n}");
        }
        if let Some(o) = orb {
            let _ = write!(hover, " · Conway orbifold {o}");
        }
        if let Some(big) = raised {
            let _ = write!(hover, "\nAt these slider values the lattice has the larger group {big} (special locus)");
        }
        let _ = write!(hover, "\n{}", click_hint(Realm::Groups));
        Segment { text, hover, realm: Realm::Groups }
    }

    /// Lattice segment: the Gallery label, hover with rule, vertex degrees and sites per cell. The `±` of a chiral
    /// lattice is drawn once, by the rename controls right after this segment (with its own hover).
    fn lattice_segment(&self) -> Segment {
        let it = self.gallery.active_item();
        let text = it.label.clone();
        let hover = format!(
            "{} · rule {} · vertex degrees {} · {} sites/cell{}\n{}",
            it.display_with_orig(),
            rule_tag(&it.alias_key),
            it.coordination,
            it.sites_per_cell,
            if it.chiral { " · chiral (±)" } else { "" },
            click_hint(Realm::Tilings),
        );
        Segment { text, hover, realm: Realm::Tilings }
    }

    /// Geometry ▸ group ▸ lattice ▸ mode for the current view (§4.4).
    pub(crate) fn breadcrumb(&self) -> Breadcrumb {
        let realm = self.current_realm();
        let geometry = Segment {
            text: self.geometry_text().to_string(),
            hover: format!("Geometry of the current view\n{}", click_hint(Realm::Geometries)),
            realm: Realm::Geometries,
        };
        let mode = Segment {
            text: self.mode_segment_text(),
            hover: format!("Current view\n{}", click_hint(realm)),
            realm,
        };
        let driven = self.view_is_lattice_driven();
        Breadcrumb {
            geometry,
            group: driven.then(|| self.group_segment()),
            lattice: driven.then(|| self.lattice_segment()),
            mode,
        }
    }

    /// True when the viewport shows the Gallery lattice (§4.4): Plane2D, Manifolds, Flat 3D on a lattice-based
    /// sub-row; every Defects view except Atlas Defects; every Analysis target.
    pub(crate) fn view_is_lattice_driven(&self) -> bool {
        match self.active_mode {
            AppMode::Lattices => match self.lattices_view {
                LatticesView::Plane2D | LatticesView::Manifolds => true,
                LatticesView::Flat3D => {
                    self.workbench.view == WorkbenchView::S4Flat3d && self.workbench.active().lattice_dependent()
                }
                LatticesView::ThreeDAndSubperiodic
                | LatticesView::NonEuclidean
                | LatticesView::Curved2D
                | LatticesView::Honeycombs3D => false,
            },
            AppMode::Defects => !matches!(self.defects_view, DefectsView::AtlasDefects | DefectsView::FusionSteps),
            AppMode::Analysis => true,
        }
    }
}
