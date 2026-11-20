//! C1 growth bindings (WP-G1): the generic grower on an explicit complex and the E³ adapter.
//! The pure-Python mirror is `papertilling/growth.py` (+ `_growth_e3.py`); both emit the same JSON.

use papertilling_core::growth::e3_complex::{E3Complex, E3_SCALE};
use papertilling_core::growth::engine::seed_cells;
use papertilling_core::growth::finite::FiniteComplex;
use papertilling_core::growth::{grow, Growth, GrowthCaps, SeedKind};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;

fn py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Grow on a finite complex given as JSON tables (`growth::finite` docs list the forms).
/// `seed_json` names the seed (`{"kind": "vertex" | "edge" | "face" | "cell", "at": ...}`).
/// Returns the growth as JSON (`layers`, `stats`, `stopped`).
#[pyfunction]
#[pyo3(signature = (complex_json, seed_json, max_layers, max_cells=50_000))]
pub fn grow_finite_json(complex_json: &str, seed_json: &str, max_layers: u32, max_cells: usize) -> PyResult<String> {
    let mut x = FiniteComplex::from_json(complex_json).map_err(py_err)?;
    let seed: serde_json::Value = serde_json::from_str(seed_json).map_err(py_err)?;
    let cells = x.seed_from_json(&seed).map_err(py_err)?;
    let g = grow(&mut x, cells, GrowthCaps { max_layers, max_cells });
    serde_json::to_string(&g).map_err(py_err)
}

#[derive(Serialize)]
struct E3Out<'a> {
    kind: &'a str,
    seed: &'a str,
    scale: i32,
    #[serde(flatten)]
    growth: Growth<[i32; 3]>,
}

/// Grow an E³ honeycomb (`kind` as in `honeycomb3d_euclidean_json`) from a seed kind
/// (`"vertex" | "edge" | "face" | "cell"`; the representatives are in `growth::e3_complex`).
/// Returns the growth as JSON, cells as scaled integer centres (real = integer / `scale`).
#[pyfunction]
#[pyo3(signature = (kind, seed="vertex", layers=2, max_cells=50_000))]
pub fn grow_e3_json(kind: &str, seed: &str, layers: u32, max_cells: usize) -> PyResult<String> {
    let mut x = E3Complex::parse(kind).map_err(py_err)?;
    let sk: SeedKind = serde_json::from_value(serde_json::Value::String(seed.to_string()))
        .map_err(|_| py_err(format!("unknown seed kind {seed:?} (vertex, edge, face, cell)")))?;
    let at = x.default_seed(sk).map_err(py_err)?;
    let cells = seed_cells(&mut x, &at).map_err(|e| py_err(format!("seed: {e:?}")))?;
    let growth = grow(&mut x, cells, GrowthCaps { max_layers: layers, max_cells });
    serde_json::to_string(&E3Out { kind, seed, scale: E3_SCALE, growth }).map_err(py_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(grow_finite_json, m)?)?;
    m.add_function(wrap_pyfunction!(grow_e3_json, m)?)?;
    Ok(())
}
