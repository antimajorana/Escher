//! C1 prewire (WP-U4 fills it): the 230-group points catalogue (up to 7 lattices per space group by the 2D "points"
//! rule, grown by layers with a seed selector; plan §5 T4), drawn at the end of the 3D space-group controls. Empty
//! until then.

use super::space3d_ui::Space3dPanelState;

/// Draw the catalogue entries of the selected space group; true when the scenes must be rebuilt.
pub(crate) fn show(_ui: &mut egui::Ui, _space3d: &mut Space3dPanelState) -> bool {
    false
}
