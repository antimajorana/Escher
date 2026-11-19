//! Stage 6 PyO3 JSON wrappers: `{p,q,r}` classification and honeycombs (S³ / E³ / H³), 3D Voronoi
//! cells, Dirichlet domains of closed 3-manifolds, and P / D / G labyrinth nets and surfaces.

use papertilling_core::honeycomb3d::{
    build_euclidean_honeycomb, build_honeycomb, classify_schlafli, dirichlet_manifold, lattice_voronoi_cell,
    nodal_surface_mesh, petrie_coxeter_surface, tpms_net, voronoi_cell_3d, EuclideanKind, TpmsKind,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;

fn py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn to_json<T: Serialize>(x: &T) -> PyResult<String> {
    serde_json::to_string(x).map_err(py_err)
}

#[pyfunction]
pub fn honeycomb3d_classify_json(p: u32, q: u32, r: u32) -> PyResult<String> {
    to_json(&classify_schlafli(p, q, r))
}

#[pyfunction]
#[pyo3(signature = (p, q, r, shells=1, patch_layers=1))]
pub fn honeycomb3d_build_json(p: u32, q: u32, r: u32, shells: u32, patch_layers: u32) -> PyResult<String> {
    to_json(&build_honeycomb(p, q, r, shells, patch_layers).map_err(py_err)?)
}

#[pyfunction]
#[pyo3(signature = (kind, shells=1))]
pub fn honeycomb3d_euclidean_json(kind: &str, shells: u32) -> PyResult<String> {
    let k = EuclideanKind::parse(kind).map_err(py_err)?;
    to_json(&build_euclidean_honeycomb(k, shells).map_err(py_err)?)
}

#[pyfunction]
pub fn honeycomb3d_voronoi_cell_json(origin: [f64; 3], neighbours: Vec<[f64; 3]>) -> PyResult<String> {
    to_json(&voronoi_cell_3d(origin, &neighbours).map_err(py_err)?)
}

#[pyfunction]
pub fn honeycomb3d_lattice_voronoi_json(lattice: &str) -> PyResult<String> {
    to_json(&lattice_voronoi_cell(lattice).map_err(py_err)?)
}

#[derive(Serialize)]
struct DirichletOut<'a> {
    domain: &'a papertilling_core::honeycomb3d::DirichletDomain3d,
    quotient: papertilling_core::honeycomb3d::DirichletQuotient,
}

#[pyfunction]
pub fn honeycomb3d_dirichlet_json(name: &str) -> PyResult<String> {
    let domain = dirichlet_manifold(name).map_err(py_err)?;
    let quotient = domain.quotient().map_err(py_err)?;
    to_json(&DirichletOut { domain: &domain, quotient })
}

#[pyfunction]
#[pyo3(signature = (kind, labyrinths=1))]
pub fn honeycomb3d_tpms_net_json(kind: &str, labyrinths: u32) -> PyResult<String> {
    let k = TpmsKind::parse(kind).map_err(py_err)?;
    to_json(&tpms_net(k, labyrinths).map_err(py_err)?)
}

#[pyfunction]
pub fn honeycomb3d_petrie_coxeter_json(kind: &str) -> PyResult<String> {
    let k = TpmsKind::parse(kind).map_err(py_err)?;
    to_json(&petrie_coxeter_surface(k).map_err(py_err)?)
}

#[pyfunction]
#[pyo3(signature = (kind, n=24))]
pub fn honeycomb3d_nodal_mesh_json(kind: &str, n: usize) -> PyResult<String> {
    let k = TpmsKind::parse(kind).map_err(py_err)?;
    to_json(&nodal_surface_mesh(k, n).map_err(py_err)?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(honeycomb3d_classify_json, m)?)?;
    m.add_function(wrap_pyfunction!(honeycomb3d_build_json, m)?)?;
    m.add_function(wrap_pyfunction!(honeycomb3d_euclidean_json, m)?)?;
    m.add_function(wrap_pyfunction!(honeycomb3d_voronoi_cell_json, m)?)?;
    m.add_function(wrap_pyfunction!(honeycomb3d_lattice_voronoi_json, m)?)?;
    m.add_function(wrap_pyfunction!(honeycomb3d_dirichlet_json, m)?)?;
    m.add_function(wrap_pyfunction!(honeycomb3d_tpms_net_json, m)?)?;
    m.add_function(wrap_pyfunction!(honeycomb3d_petrie_coxeter_json, m)?)?;
    m.add_function(wrap_pyfunction!(honeycomb3d_nodal_mesh_json, m)?)?;
    Ok(())
}
