//! C1 prewire (WP-U4 fills it): grow a layer-group lattice in-plane and add layers (plan §5 T5), drawn at the end
//! of the 80 Layer Groups controls. Empty until then.

use super::layers_ui::LayersPanelState;

/// Draw the layer-growth controls; true when the scenes must be rebuilt.
pub(crate) fn show(_ui: &mut egui::Ui, _layers: &mut LayersPanelState) -> bool {
    false
}
