//! PyO3 JSON wrappers exposing Bieberbach cohomology, periodic Voronoi, live N>7 search,
//! 0D/1D subperiodic groups, 80 layer groups, 230 3D space groups & nets, and hyperbolic tilings.

use papertilling_core::bieberbach::{analyze_wallpaper_group, ConvOp, Rat, SpaceGroupAnalyzer, summarize_analyzer};
use papertilling_core::hyperbolic::{build_hyperbolic_tiling, HyperbolicStyle};
use papertilling_core::plane::live_search::search_minimal_lattices;
use papertilling_core::plane::voronoi::voronoi_bonds;
use papertilling_core::space3d::nets3d::{build_net_3d, CanonicalNet3d, LineDefect3d};
use papertilling_core::space3d::{analyze_layer_group, analyze_space_group_3d, list_layer_groups, list_space_groups_3d};
use papertilling_core::subperiodic::{build_frieze_patch, build_rosette, FriezeDefect, FriezeTopology};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;

fn py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[derive(Deserialize)]
struct PyConvOp {
    r: Vec<Vec<i64>>,
    t: Vec<[i64; 2]>,
}

#[derive(Deserialize)]
struct PyAnalyzeSpec {
    dim: usize,
    number: i32,
    symbol: String,
    ops: Vec<PyConvOp>,
    tb_rows: Vec<Vec<[i64; 2]>>,
}

#[pyfunction]
pub fn bieberbach_wallpaper_json(group: &str) -> PyResult<String> {
    let sum = analyze_wallpaper_group(group).map_err(py_err)?;
    serde_json::to_string(&sum).map_err(py_err)
}

#[pyfunction]
pub fn bieberbach_analyze_json(spec_json: &str) -> PyResult<String> {
    let spec: PyAnalyzeSpec = serde_json::from_str(spec_json).map_err(py_err)?;
    let ops: Vec<ConvOp> = spec
        .ops
        .into_iter()
        .map(|op| ConvOp {
            r: op.r,
            t: op.t.into_iter().map(|[n, d]| Rat::new(n, d)).collect(),
        })
        .collect();
    let tb: Vec<Vec<Rat>> = spec
        .tb_rows
        .into_iter()
        .map(|row| row.into_iter().map(|[n, d]| Rat::new(n, d)).collect())
        .collect();
    let sg = SpaceGroupAnalyzer::new(spec.dim, spec.number, &spec.symbol, &ops, &tb).map_err(py_err)?;
    serde_json::to_string(&summarize_analyzer(&sg, &[2])).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (frac_pts_json, a1, a2, tol=1e-7))]
pub fn voronoi_bonds_json(
    frac_pts_json: &str,
    a1: [f64; 2],
    a2: [f64; 2],
    tol: f64,
) -> PyResult<String> {
    let pts: Vec<[f64; 2]> = serde_json::from_str(frac_pts_json).map_err(py_err)?;
    serde_json::to_string(&voronoi_bonds(&pts, a1, a2, tol)).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (group, n_min=7, n_max=18, min_degree=3, max_results=30))]
pub fn search_minimal_lattices_json(
    group: &str,
    n_min: usize,
    n_max: usize,
    min_degree: usize,
    max_results: usize,
) -> PyResult<String> {
    let res = search_minimal_lattices(group, n_min, n_max, min_degree, max_results).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (dihedral=false, n=5, wedge_delta=0, mirror_wall=false))]
pub fn subperiodic_rosette_json(
    dihedral: bool,
    n: usize,
    wedge_delta: i32,
    mirror_wall: bool,
) -> PyResult<String> {
    serde_json::to_string(&build_rosette(dihedral, n, wedge_delta, mirror_wall)).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (symbol, cells=10, topology="OpenStrip", defect="None"))]
pub fn subperiodic_frieze_json(
    symbol: &str,
    cells: usize,
    topology: &str,
    defect: &str,
) -> PyResult<String> {
    let topo = match topology {
        "Cylinder" => FriezeTopology::Cylinder,
        "Mobius" => FriezeTopology::Mobius,
        _ => FriezeTopology::OpenStrip,
    };
    let def = match defect {
        "HalfTurn180" => FriezeDefect::HalfTurn180,
        "GlideDispiration" => FriezeDefect::GlideDispiration,
        _ => FriezeDefect::None,
    };
    let patch = build_frieze_patch(symbol, cells, topo, def).map_err(py_err)?;
    serde_json::to_string(&patch).map_err(py_err)
}

#[pyfunction]
pub fn layer_groups_json() -> PyResult<String> {
    serde_json::to_string(list_layer_groups()).map_err(py_err)
}

#[pyfunction]
pub fn analyze_layer_group_json(number: u8) -> PyResult<String> {
    let res = analyze_layer_group(number).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
pub fn space_groups_3d_json() -> PyResult<String> {
    serde_json::to_string(list_space_groups_3d()).map_err(py_err)
}

#[pyfunction]
pub fn analyze_space_group_3d_json(number: u16) -> PyResult<String> {
    let res = analyze_space_group_3d(number).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (net="SimpleCubic", defect="None"))]
pub fn build_net_3d_json(net: &str, defect: &str) -> PyResult<String> {
    let n = match net {
        "BCC" => CanonicalNet3d::BCC,
        "FCC" => CanonicalNet3d::FCC,
        "Diamond" => CanonicalNet3d::Diamond,
        "HCP" => CanonicalNet3d::HCP,
        "Tetragonal123" => CanonicalNet3d::Tetragonal123,
        "Rutile136" => CanonicalNet3d::Rutile136,
        "Perovskite" => CanonicalNet3d::Perovskite,
        "Pyrochlore" => CanonicalNet3d::Pyrochlore,
        "BetaQuartz" => CanonicalNet3d::BetaQuartz,
        _ => CanonicalNet3d::SimpleCubic,
    };
    let d = match defect {
        "ScrewDislocationZ" => LineDefect3d::ScrewDislocationZ,
        "PureDisclination90_123" => LineDefect3d::PureDisclination90_123,
        "ScrewDispiration42_136" => LineDefect3d::ScrewDispiration42_136,
        _ => LineDefect3d::None,
    };
    serde_json::to_string(&build_net_3d(n, d)).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (p=7, q=3, style="RegularPQ", layers=2, wedge_delta=0))]
pub fn hyperbolic_tiling_json(
    p: usize,
    q: usize,
    style: &str,
    layers: usize,
    wedge_delta: i32,
) -> PyResult<String> {
    let st = match style {
        "RectifiedKagome" => HyperbolicStyle::RectifiedKagome,
        "DualQP" => HyperbolicStyle::DualQP,
        _ => HyperbolicStyle::RegularPQ,
    };
    let res = build_hyperbolic_tiling(p, q, st, layers, wedge_delta).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bieberbach_wallpaper_json, m)?)?;
    m.add_function(wrap_pyfunction!(bieberbach_analyze_json, m)?)?;
    m.add_function(wrap_pyfunction!(voronoi_bonds_json, m)?)?;
    m.add_function(wrap_pyfunction!(search_minimal_lattices_json, m)?)?;
    m.add_function(wrap_pyfunction!(subperiodic_rosette_json, m)?)?;
    m.add_function(wrap_pyfunction!(subperiodic_frieze_json, m)?)?;
    m.add_function(wrap_pyfunction!(layer_groups_json, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_layer_group_json, m)?)?;
    m.add_function(wrap_pyfunction!(space_groups_3d_json, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_space_group_3d_json, m)?)?;
    m.add_function(wrap_pyfunction!(build_net_3d_json, m)?)?;
    m.add_function(wrap_pyfunction!(hyperbolic_tiling_json, m)?)?;
    Ok(())
}
