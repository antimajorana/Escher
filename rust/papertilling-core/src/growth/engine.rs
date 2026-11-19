//! The layer grower (`grow`): filled by WP-G1.

use super::{CellComplex, Growth, GrowthCaps};

/// Stub with the final signature. WP-G1 replaces the body with the touching-rule
/// grower of §2.1-§2.2; until then it refuses loudly instead of returning an empty
/// result that looks like success.
pub fn grow<X: CellComplex>(_x: &mut X, _seed: Vec<X::Cell>, _caps: GrowthCaps) -> Growth<X::Cell> {
    unimplemented!("growth engine not yet built (WP-G1)")
}
