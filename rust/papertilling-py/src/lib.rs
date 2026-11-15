//! PyO3 bindings (`papertilling_rs`) exposing `papertilling-core` to Python.

use papertilling_core::dsymbol::{self, DSymbolKey, KeyRow};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

fn parse_key(rows: Vec<Vec<u32>>) -> Result<DSymbolKey, String> {
    let mut out = Vec::with_capacity(rows.len());
    for (idx, row) in rows.into_iter().enumerate() {
        let [m01, m12, s0, s1, s2]: [u32; 5] = row
            .try_into()
            .map_err(|v: Vec<u32>| format!("key row {idx} must have length 5, got {}", v.len()))?;
        out.push((m01, m12, s0, s1, s2));
    }
    Ok(out)
}

fn key_to_pytuple<'py>(py: Python<'py>, key: Vec<KeyRow>) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, key)
}

/// Return the `papertilling-core` crate version string.
#[pyfunction]
pub fn version() -> &'static str {
    papertilling_core::VERSION
}

/// Compute the canonical breadth-first D-symbol key as a tuple of 5-tuples.
#[pyfunction]
pub fn canonical_key<'py>(
    py: Python<'py>,
    sig: Vec<Vec<usize>>,
    m01: Vec<u32>,
    m12: Vec<u32>,
) -> PyResult<Bound<'py, PyTuple>> {
    let sig_arr: [Vec<usize>; 3] = sig.try_into().map_err(|v: Vec<Vec<usize>>| {
        PyValueError::new_err(format!("sig must contain 3 involutions, got {}", v.len()))
    })?;
    let key = dsymbol::canonical_key(&sig_arr, &m01, &m12)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    key_to_pytuple(py, key)
}

/// Compute the minimal-image network key (forgetting symmetry) as a tuple of 5-tuples.
#[pyfunction]
pub fn network_key<'py>(py: Python<'py>, key: Vec<Vec<u32>>) -> PyResult<Bound<'py, PyTuple>> {
    let parsed = parse_key(key).map_err(PyValueError::new_err)?;
    let min_key =
        dsymbol::network_key(&parsed).map_err(|e| PyValueError::new_err(e.to_string()))?;
    key_to_pytuple(py, min_key)
}

/// Compute `len(key) // len(network_key(key))`.
#[pyfunction]
pub fn symmetry_index(key: Vec<Vec<u32>>) -> PyResult<usize> {
    let parsed = parse_key(key).map_err(PyValueError::new_err)?;
    dsymbol::symmetry_index(&parsed).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn papertilling_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", papertilling_core::VERSION)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_key, m)?)?;
    m.add_function(wrap_pyfunction!(network_key, m)?)?;
    m.add_function(wrap_pyfunction!(symmetry_index, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use papertilling_core::dsymbol;

    use super::{parse_key, version};

    #[test]
    fn wrapper_version_and_key_parsing() {
        assert_eq!(version(), "0.2.0");
        let p2_sq = parse_key(vec![
            vec![4, 4, 1, 2, 1],
            vec![4, 4, 0, 3, 0],
            vec![4, 4, 3, 0, 3],
            vec![4, 4, 2, 1, 2],
        ])
        .unwrap();
        assert_eq!(dsymbol::symmetry_index(&p2_sq).unwrap(), 4);
        assert!(parse_key(vec![vec![4, 4, 0]]).is_err());
    }
}
