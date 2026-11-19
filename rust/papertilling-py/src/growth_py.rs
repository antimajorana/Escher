//! C1 growth bindings (WP-G1): the generic grower on an explicit complex and the E³ adapter.
//! WP-C0 stubs: final signatures, each returns a "not yet built" error until WP-G1 fills it.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

/// Grow on a finite complex given as JSON tables. `seed_json` names the seed
/// (`{"kind": "vertex" | "edge" | "face" | "cell", "at": ...}`). Returns the growth as JSON.
#[pyfunction]
#[pyo3(signature = (complex_json, seed_json, max_layers, max_cells=50_000))]
pub fn grow_finite_json(
    complex_json: &str,
    seed_json: &str,
    max_layers: u32,
    max_cells: usize,
) -> PyResult<String> {
    let _ = (complex_json, seed_json, max_layers, max_cells);
    Err(PyNotImplementedError::new_err("grow_finite_json: not yet built (WP-G1)"))
}

/// Grow an E³ honeycomb (`kind` as in `honeycomb3d_euclidean_json`) from a seed kind
/// (`"vertex" | "edge" | "face" | "cell"`). Returns the growth as JSON.
#[pyfunction]
#[pyo3(signature = (kind, seed="vertex", layers=2, max_cells=50_000))]
pub fn grow_e3_json(kind: &str, seed: &str, layers: u32, max_cells: usize) -> PyResult<String> {
    let _ = (kind, seed, layers, max_cells);
    Err(PyNotImplementedError::new_err("grow_e3_json: not yet built (WP-G1)"))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(grow_finite_json, m)?)?;
    m.add_function(wrap_pyfunction!(grow_e3_json, m)?)?;
    Ok(())
}
