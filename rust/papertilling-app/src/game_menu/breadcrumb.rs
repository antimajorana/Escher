//! HUD breadcrumb (geometry ▸ group ▸ lattice ▸ mode): pure functions of the app state, no drawing. WP0 stub
//! with the final signatures (only the mode segment is real); WP-A implements the §4.4 tables.

use super::Realm;
use crate::app::{AppMode, PapertillingApp};

/// One clickable breadcrumb segment: its text, hover text, and the realm a click opens.
#[allow(dead_code)] // M1 prewire: drawn by the HUD (WP-A).
pub(crate) struct Segment {
    pub text: String,
    pub hover: String,
    pub realm: Realm,
}

/// The four segments; `group` and `lattice` only when the view shows the Gallery lattice.
#[allow(dead_code)] // M1 prewire: drawn by the HUD (WP-A).
pub(crate) struct Breadcrumb {
    pub geometry: Segment,
    pub group: Option<Segment>,
    pub lattice: Option<Segment>,
    pub mode: Segment,
}

impl PapertillingApp {
    /// Text of the mode segment (`Lattices: 2D Plane`, `Defects: …`, `Analysis: …`).
    pub(crate) fn mode_segment_text(&self) -> String {
        match self.active_mode {
            AppMode::Lattices => format!("Lattices: {}", self.lattices_view.label()),
            AppMode::Defects => format!("Defects: {}", self.defects_view.label()),
            AppMode::Analysis => format!("Analysis: {}", self.analysis.target.label()),
        }
    }

    /// WP0 stub: only the mode segment is filled in (the geometry segment is empty, group and lattice absent).
    #[allow(dead_code)] // M1 prewire: replaced by WP-A.
    pub(crate) fn breadcrumb(&self) -> Breadcrumb {
        let realm = self.current_realm();
        Breadcrumb {
            geometry: Segment { text: String::new(), hover: String::new(), realm: Realm::Geometries },
            group: None,
            lattice: None,
            mode: Segment { text: self.mode_segment_text(), hover: String::new(), realm },
        }
    }

    /// WP0 stub (always false); WP-A implements the §4.4 lattice-driven rule.
    #[allow(dead_code)] // M1 prewire: replaced by WP-A, read by equip (WP-C).
    pub(crate) fn view_is_lattice_driven(&self) -> bool {
        false
    }
}
