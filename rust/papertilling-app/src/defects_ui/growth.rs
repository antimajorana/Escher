//! C1 prewire (WP-U7 fills it): controls for growing a defected lattice by Dirichlet-seeded inflation (plan §5 T7),
//! drawn at the end of the Defects catalogue's left panel. Empty until then.

use crate::backend::CatalogueItemUi;
use crate::defects_tab::DefectsTabState;

/// Draw the growth controls; true when the defect picture must be rebuilt. (`PapertillingApp` is not in scope at
/// this slot, which is a `DefectsTabState` method, so the stub takes the tab state and the Gallery lattice.)
pub(crate) fn show(_ui: &mut egui::Ui, _defects: &mut DefectsTabState, _item: &CatalogueItemUi) -> bool {
    false
}
