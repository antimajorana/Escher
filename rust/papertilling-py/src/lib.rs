//! PyO3 bindings (`papertilling_rs`) exposing `papertilling-core` Stages 0–5 to Python.

mod atlas_py;
mod insights_py;
mod solver_py;
mod stage5_py;

use std::collections::BTreeMap;

use papertilling_core::defect::DefectCatalogue;
use papertilling_core::dsymbol::{self, DSymbolKey, KeyRow};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::tuning::{allowed_interval, special_points};
use papertilling_core::surface::surface_table;
use papertilling_core::types::{Cell2D, Frame2D, Recipe};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use serde::Serialize;

pub use insights_py::{
    auto_burgers_circuit_json, burgers_circuit_json, gauge_observables_json,
    relax_defect_sheet_3d_json, relax_elastic_2d_json, relax_elastic_2d_patch_json,
    relax_surface_3d_json, spectrum_json,
};
pub use stage5_py::{
    core_gallery_json, equivalent_single_json, isomorphic_cores, layer_signatures_irregular,
    nearest_allowed_centre_json,
};

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

fn pure_dsymbol_from_recipe(recipe_json: &str) -> Result<String, String> {
    let recipe: Recipe = serde_json::from_str(recipe_json).map_err(|e| e.to_string())?;
    let fam = LatticeFamily::from_recipe(&recipe, None)?;
    let ds = fam.dsymbol()?;
    ds.to_json().map_err(|e| e.to_string())
}

fn parse_values_override(values_json: Option<&str>) -> Result<Option<BTreeMap<String, f64>>, String> {
    match values_json {
        Some(s) => serde_json::from_str(s).map(Some).map_err(|e| e.to_string()),
        None => Ok(None),
    }
}

fn pure_allowed_interval(
    recipe_json: &str,
    param: &str,
    lo: f64,
    hi: f64,
    n: Option<usize>,
    values_json: Option<&str>,
    margin: Option<f64>,
) -> Result<String, String> {
    let recipe: Recipe = serde_json::from_str(recipe_json).map_err(|e| e.to_string())?;
    let vals_ov = parse_values_override(values_json)?;
    let mut res = allowed_interval(&recipe, param, lo, hi, vals_ov.as_ref(), n)?;
    if let Some(m) = margin
        && res.current_ok
        && (res.edges.0.is_some() || res.edges.1.is_some())
    {
        let x0 = match vals_ov.as_ref().and_then(|map| map.get(param).copied()) {
            Some(v) => v,
            None => {
                allowed_interval(
                    &recipe,
                    param,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    vals_ov.as_ref(),
                    Some(2),
                )?
                .interval
                .0
            }
        };
        let (lo_eff, hi_eff) = (lo.min(x0), hi.max(x0));
        let m0 = 1e-4 * (hi_eff - lo_eff);
        if let Some(ea) = res.edges.0 {
            let g_lo = if res.interval.0 < x0 { res.interval.0 - m0 } else { ea.min(x0) };
            res.interval.0 = (g_lo + m).min(x0);
        }
        if let Some(eb) = res.edges.1 {
            let g_hi = if res.interval.1 > x0 { res.interval.1 + m0 } else { eb.max(x0) };
            res.interval.1 = (g_hi - m).max(x0);
        }
    }
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

fn pure_special_points(
    recipe_json: &str,
    param: &str,
    lo: f64,
    hi: f64,
    values_json: Option<&str>,
) -> Result<String, String> {
    let recipe: Recipe = serde_json::from_str(recipe_json).map_err(|e| e.to_string())?;
    let vals_ov = parse_values_override(values_json)?;
    let res = special_points(&recipe, param, lo, hi, vals_ov.as_ref())?;
    serde_json::to_string(&res).map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct SurfaceRowSummary {
    surface: String,
    status: String,
    sites: Option<usize>,
    cones: String,
    reason: String,
}

fn pure_surface_table(group: &str, cell_json: &str) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let rows: Vec<SurfaceRowSummary> = surface_table(group, &cell)
        .into_iter()
        .map(|r| SurfaceRowSummary {
            surface: r.surface,
            status: r.status,
            sites: r.sites,
            cones: r.cones,
            reason: r.reason,
        })
        .collect();
    serde_json::to_string(&rows).map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct CatalogueJson<'a> {
    group: &'a str,
    chiral: bool,
    r0: f64,
    dislocations: &'a [papertilling_core::defect::DislocationEntry],
    centre_classes: &'a [papertilling_core::defect::CentreClassEntry],
    disclinations: &'a [papertilling_core::defect::DisclinationEntry],
    mirrors: &'a [papertilling_core::defect::MirrorLineEntry],
    domain_walls: &'a [papertilling_core::defect::DomainWallEntry],
    twin_walls: &'a [papertilling_core::defect::TwinWallEntry],
    mobius_seams: &'a [papertilling_core::defect::MobiusSeamEntry],
    glide_breaking: &'a [papertilling_core::defect::GlideBreakingEntry],
}

fn pure_defect_catalogue(
    group: &str,
    cell_json: &str,
    frame_json: Option<&str>,
) -> Result<String, String> {
    let cell: Cell2D = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let frame = match frame_json {
        Some(s) => serde_json::from_str(s).map_err(|e| e.to_string())?,
        None => Frame2D::default(),
    };
    let cat = DefectCatalogue::new(group, &cell, frame);
    let view = CatalogueJson {
        group: &cat.group,
        chiral: cat.chiral,
        r0: cat.r0,
        dislocations: &cat.dislocations,
        centre_classes: &cat.centre_classes,
        disclinations: &cat.disclinations,
        mirrors: &cat.mirrors,
        domain_walls: &cat.domain_walls,
        twin_walls: &cat.twin_walls,
        mobius_seams: &cat.mobius_seams,
        glide_breaking: &cat.glide_breaking,
    };
    serde_json::to_string(&view).map_err(|e| e.to_string())
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

#[pyfunction]
pub fn dsymbol_from_recipe_json(recipe_json: &str) -> PyResult<String> {
    pure_dsymbol_from_recipe(recipe_json).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (recipe_json, param, lo, hi, n=None, values_json=None, margin=None))]
pub fn allowed_interval_json(
    recipe_json: &str,
    param: &str,
    lo: f64,
    hi: f64,
    n: Option<usize>,
    values_json: Option<&str>,
    margin: Option<f64>,
) -> PyResult<String> {
    pure_allowed_interval(recipe_json, param, lo, hi, n, values_json, margin)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (recipe_json, param, lo, hi, values_json=None))]
pub fn special_points_json(
    recipe_json: &str,
    param: &str,
    lo: f64,
    hi: f64,
    values_json: Option<&str>,
) -> PyResult<String> {
    pure_special_points(recipe_json, param, lo, hi, values_json).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn surface_table_json(group: &str, cell_json: &str) -> PyResult<String> {
    pure_surface_table(group, cell_json).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (group, cell_json, frame_json=None))]
pub fn defect_catalogue_json(
    group: &str,
    cell_json: &str,
    frame_json: Option<&str>,
) -> PyResult<String> {
    pure_defect_catalogue(group, cell_json, frame_json).map_err(PyValueError::new_err)
}

#[pymodule]
fn papertilling_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", papertilling_core::VERSION)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_key, m)?)?;
    m.add_function(wrap_pyfunction!(network_key, m)?)?;
    m.add_function(wrap_pyfunction!(symmetry_index, m)?)?;
    m.add_function(wrap_pyfunction!(dsymbol_from_recipe_json, m)?)?;
    m.add_function(wrap_pyfunction!(allowed_interval_json, m)?)?;
    m.add_function(wrap_pyfunction!(special_points_json, m)?)?;
    m.add_function(wrap_pyfunction!(surface_table_json, m)?)?;
    m.add_function(wrap_pyfunction!(defect_catalogue_json, m)?)?;
    m.add_function(wrap_pyfunction!(isomorphic_cores, m)?)?;
    m.add_function(wrap_pyfunction!(layer_signatures_irregular, m)?)?;
    m.add_function(wrap_pyfunction!(equivalent_single_json, m)?)?;
    m.add_function(wrap_pyfunction!(core_gallery_json, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_allowed_centre_json, m)?)?;
    m.add_function(wrap_pyfunction!(burgers_circuit_json, m)?)?;
    m.add_function(wrap_pyfunction!(auto_burgers_circuit_json, m)?)?;
    m.add_function(wrap_pyfunction!(gauge_observables_json, m)?)?;
    m.add_function(wrap_pyfunction!(relax_elastic_2d_json, m)?)?;
    m.add_function(wrap_pyfunction!(relax_elastic_2d_patch_json, m)?)?;
    m.add_function(wrap_pyfunction!(relax_surface_3d_json, m)?)?;
    m.add_function(wrap_pyfunction!(relax_defect_sheet_3d_json, m)?)?;
    m.add_function(wrap_pyfunction!(spectrum_json, m)?)?;
    atlas_py::register(m)?;
    solver_py::register(m)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use papertilling_core::dsymbol;

    use super::{
        parse_key, pure_allowed_interval, pure_dsymbol_from_recipe, pure_special_points, version,
    };

    #[test]
    fn wrapper_version_and_stage12_helpers() {
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

        let kagome_json =
            r#"{"group":"p6m","sites":[[0.5,0.0]],"bonds":[[[0.5,0.0],[0.5,0.5]]],"a":1.0}"#;
        let ds_json = pure_dsymbol_from_recipe(kagome_json).unwrap();
        assert!(ds_json.contains("\"size\":2"));

        let p2_ob_json = r#"{"group":"p2","sites":[[0.0,0.0]],"bonds":[[[0.0,0.0],[1.0,0.0]],[[0.0,0.0],[0.0,1.0]]],"a":1.0,"b":2.5,"gamma":70.0}"#;
        let sp_json = pure_special_points(p2_ob_json, "gamma", 45.0, 135.0, None).unwrap();
        assert!(sp_json.contains("\"group\":\"pmm\""));

        // Overriding gamma=90 via values_json makes b=1 hit p4m instead of cmm!
        let sp_ov = pure_special_points(
            p2_ob_json,
            "b",
            0.4,
            3.0,
            Some(r#"{"gamma":90.0,"b":2.5}"#),
        )
        .unwrap();
        assert!(sp_ov.contains("\"group\":\"p4m\""));

        // Rotating-squares p4g family (t1 = (-0.324 + 0.176)/2 = -0.074) with custom n, values_json, and margin.
        let p4g_json = r#"{"group":"p4g","sites":[[-0.324,0.176]],"bonds":[[[-0.324,0.176],[0.324,0.824]],[[-0.324,0.176],[0.176,0.324]]],"a":1.0}"#;
        let ai_default =
            pure_allowed_interval(p4g_json, "t1", -0.574, 0.426, Some(101), None, None).unwrap();
        let ai_wider_margin = pure_allowed_interval(
            p4g_json,
            "t1",
            -0.574,
            0.426,
            Some(101),
            Some(r#"{"t1":-0.074}"#),
            Some(0.01),
        )
        .unwrap();
        let v_def: serde_json::Value = serde_json::from_str(&ai_default).unwrap();
        let v_wid: serde_json::Value = serde_json::from_str(&ai_wider_margin).unwrap();
        assert!(v_def["current_ok"].as_bool().unwrap());
        assert!(v_wid["interval"][0].as_f64().unwrap() > v_def["interval"][0].as_f64().unwrap());
        assert!(v_wid["interval"][1].as_f64().unwrap() < v_def["interval"][1].as_f64().unwrap());
    }
}
