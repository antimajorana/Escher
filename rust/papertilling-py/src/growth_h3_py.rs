//! C1 honeycomb growth binding (WP-G2): E³, H³ and S³ honeycombs from a seed.
//! The pure-Python mirror is `papertilling/_honeycomb3d_growth.py`; both emit the same JSON keys.

use papertilling_core::growth::{GrowthCaps, SeedKind};
use papertilling_core::honeycomb3d::growth_build::{MAX_GROWTH_CELLS, grow_honeycomb};
use papertilling_core::honeycomb3d::{GrowthRule, HoneycombKind};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Grow a honeycomb (`kind` a Schläfli symbol such as `"{4,3,5}"` or an E³ kind) from a
/// seed kind (`"vertex" | "edge" | "face" | "cell"`) for `layers` layers, by `rule`
/// (`"touching"` or `"face_shells"`). `max_cells` `None` uses the core default cap
/// (120 000, `MAX_GROWTH_CELLS`). Returns the growth as JSON (`stats`, `stopped`, `precision`,
/// `boundary_only`, and `complex` unless `draw` is false).
#[pyfunction]
#[pyo3(signature = (kind, seed="vertex", layers=2, max_cells=None, rule="touching", draw=true))]
pub fn grow_honeycomb_json(
    kind: &str,
    seed: &str,
    layers: u32,
    max_cells: Option<usize>,
    rule: &str,
    draw: bool,
) -> PyResult<String> {
    let k = HoneycombKind::parse(kind).map_err(py_err)?;
    let sk: SeedKind = serde_json::from_value(serde_json::Value::String(seed.to_string()))
        .map_err(|_| py_err(format!("unknown seed kind {seed:?} (vertex, edge, face, cell)")))?;
    let rl: GrowthRule = serde_json::from_value(serde_json::Value::String(rule.to_string()))
        .map_err(|_| py_err(format!("unknown rule {rule:?} (touching, face_shells)")))?;
    let caps = GrowthCaps { max_layers: layers, max_cells: max_cells.unwrap_or(MAX_GROWTH_CELLS) };
    let g = grow_honeycomb(k, sk, rl, caps).map_err(py_err)?;
    let mut v = serde_json::to_value(&g).map_err(py_err)?;
    if !draw && let Some(o) = v.as_object_mut() {
        o.remove("complex");
    }
    serde_json::to_string(&v).map_err(py_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(grow_honeycomb_json, m)?)?;
    Ok(())
}
