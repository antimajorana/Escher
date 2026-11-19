//! C1 honeycomb growth binding (WP-G2): E³, H³ and S³ honeycombs from a seed.
//! WP-C0 stub: final signature, returns a "not yet built" error until WP-G2 fills it.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

/// Grow a honeycomb (`kind` a Schläfli symbol such as `"{4,3,5}"` or an E³ kind) from a
/// seed kind (`"vertex" | "edge" | "face" | "cell"`) for `layers` layers. `max_cells`
/// `None` uses the core default cap (§1 Q13). Returns the growth as JSON.
#[pyfunction]
#[pyo3(signature = (kind, seed="vertex", layers=2, max_cells=None))]
pub fn grow_honeycomb_json(
    kind: &str,
    seed: &str,
    layers: u32,
    max_cells: Option<usize>,
) -> PyResult<String> {
    let _ = (kind, seed, layers, max_cells);
    Err(PyNotImplementedError::new_err("grow_honeycomb_json: not yet built (WP-G2)"))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(grow_honeycomb_json, m)?)?;
    Ok(())
}
