//! Python bindings for Stage 3 defect fields (`papertilling_core::defect_fields`), used by
//! `papertilling/defect_fields.py`. Structured results are returned as JSON strings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Serialize;
use serde_json::{Value, json};

use papertilling_core::defect_fields::{
    BondGraph, LocalStrain, RefLattice, antiplane_screw_field, barnett_lothe_tensor, cauchy_born_tensor, classify_elastic, cna_2d,
    detect_defects_from_coords, dipole_prelog_coeff, export_motion_gif, local_deformation_and_strain, nye_tensor, psi_n,
    stroh_dislocation_field, vitek_differential, voronoi_coordination, wigner_seitz_occupancy,
};
use papertilling_core::defect_fields::nye_loops::default_cutoff;
use papertilling_core::types::{Cell2D, Frame2D};

/// Stamp of the numerical symmetry classification (same text as the Python fallback).
const CLASS_STATUS: &str =
    "[COMPUTATIONAL EVIDENCE] order-2 / order-4 harmonics of C_iajb e_i e_a e_j e_b compared with a 1e-8 relative tolerance";

fn err<E: ToString>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn to_json<T: Serialize>(v: &T) -> PyResult<String> {
    serde_json::to_string(v).map_err(err)
}

fn parse_cell(cell_json: &str) -> PyResult<Cell2D> {
    serde_json::from_str(cell_json).map_err(err)
}

fn parse_frame(frame_json: Option<&str>) -> PyResult<Frame2D> {
    frame_json.map_or(Ok(Frame2D::default()), |s| serde_json::from_str(s).map_err(err))
}

fn graph(pos: &[[f64; 2]], edges: Option<Vec<(usize, usize)>>, cutoff: Option<f64>) -> PyResult<BondGraph> {
    match (edges, cutoff) {
        (Some(e), _) => Ok(BondGraph::from_edges(pos, &e)),
        (None, Some(c)) => Ok(BondGraph::from_cutoff(pos, c)),
        (None, None) => Err(PyValueError::new_err("give edges or cutoff")),
    }
}

/// Coordinate-only defect detection (`detect_defects_from_coords`).
#[pyfunction]
#[pyo3(signature = (pos, group, cell_json, frame_json=None, edges=None, cutoff=None, clearance=1))]
pub fn detect_defects_coords_json(
    pos: Vec<[f64; 2]>, group: &str, cell_json: &str, frame_json: Option<&str>, edges: Option<Vec<(usize, usize)>>,
    cutoff: Option<f64>, clearance: usize,
) -> PyResult<String> {
    let cell = parse_cell(cell_json)?;
    let det = detect_defects_from_coords(&pos, group, &cell, parse_frame(frame_json)?, edges.as_deref(), cutoff, clearance).map_err(err)?;
    to_json(&det)
}

/// Wigner-Seitz occupancy of reference sites.
#[pyfunction]
pub fn wigner_seitz_occupancy_json(ref_sites: Vec<[f64; 2]>, actual_sites: Vec<[f64; 2]>) -> PyResult<String> {
    to_json(&wigner_seitz_occupancy(&ref_sites, &actual_sites).map_err(err)?)
}

/// Nye tensor per face plus (with `ref_pos`) local strain and (with `burgers`) Vitek arrows.
#[pyfunction]
#[pyo3(signature = (pos, cell_json, frame_json=None, ref_pos=None, edges=None, cutoff=None, burgers=None))]
pub fn nye_strain_json(
    pos: Vec<[f64; 2]>, cell_json: &str, frame_json: Option<&str>, ref_pos: Option<Vec<[f64; 2]>>,
    edges: Option<Vec<(usize, usize)>>, cutoff: Option<f64>, burgers: Option<[f64; 2]>,
) -> PyResult<String> {
    let (cell, frame) = (parse_cell(cell_json)?, parse_frame(frame_json)?);
    let all_edges = match &edges {
        Some(e) => BondGraph::from_edges(&pos, e).edges,
        None => {
            let rl = RefLattice::new(&cell, frame);
            BondGraph::from_cutoff(&pos, cutoff.unwrap_or_else(|| default_cutoff(&cell, &rl))).edges
        }
    };
    let nye = nye_tensor(&pos, &cell, frame, Some(&all_edges), None, None).map_err(err)?;
    let mut out = json!({
        "faces": nye.faces, "areas": nye.areas, "alpha": nye.alpha, "frame_angle_deg": nye.frame_angle_deg,
        "boundary_walk": nye.boundary_walk, "net_frank_deg": nye.net_frank_deg, "nye_total": nye.total,
        "boundary_closure": nye.boundary_closure, "reason": nye.reason, "edges": all_edges,
    });
    if let Some(r) = ref_pos {
        let st: Vec<LocalStrain> = local_deformation_and_strain(&r, &pos, &all_edges).map_err(err)?;
        out["strain"] = serde_json::to_value(&st).map_err(err)?;
        if let Some(b) = burgers {
            let v = vitek_differential(&r, &pos, &all_edges, b).map_err(err)?;
            out["vitek"] = json!({
                "edges": v.iter().map(|a| [a.i, a.j]).collect::<Vec<_>>(),
                "du": v.iter().map(|a| a.du).collect::<Vec<_>>(),
                "along_b": v.iter().map(|a| a.along_b).collect::<Vec<_>>(),
                "wrapped": v.iter().map(|a| a.wrapped).collect::<Vec<_>>(),
            });
        }
    }
    to_json(&out)
}

/// Bond-orientational order `psi_n` as `[re, im]` per site.
#[pyfunction]
#[pyo3(signature = (pos, n, edges=None, cutoff=None))]
pub fn psi_n_order_list(pos: Vec<[f64; 2]>, n: u32, edges: Option<Vec<(usize, usize)>>, cutoff: Option<f64>) -> PyResult<Vec<[f64; 2]>> {
    Ok(psi_n(&graph(&pos, edges, cutoff)?, n))
}

/// 2D common-neighbour analysis: per-bond signatures and per-site counts (keys `"n,b,l"`).
#[pyfunction]
#[pyo3(signature = (pos, edges=None, cutoff=None))]
pub fn cna_signatures_json(pos: Vec<[f64; 2]>, edges: Option<Vec<(usize, usize)>>, cutoff: Option<f64>) -> PyResult<String> {
    let g = graph(&pos, edges, cutoff)?;
    let (sigs, per_site) = cna_2d(&g);
    let per: Vec<Value> = per_site.iter().map(|m| {
        Value::Object(m.iter().map(|(k, v)| (format!("{},{},{}", k.0, k.1, k.2), json!(v))).collect())
    }).collect();
    to_json(&json!({ "edges": g.edges, "signatures": sigs, "per_site": per }))
}

/// Cauchy-Born moduli, symmetry class, Barnett-Lothe tensor, dipole slope and (at `points`) the Stroh field.
#[pyfunction]
#[pyo3(signature = (cell_json, c_flat, b, points=None, n_quad=128))]
pub fn barnett_lothe_stroh_json(
    cell_json: Option<&str>, c_flat: Option<Vec<f64>>, b: [f64; 2], points: Option<Vec<[f64; 2]>>, n_quad: usize,
) -> PyResult<String> {
    let c = match (cell_json, c_flat) {
        (Some(s), _) => cauchy_born_tensor(&parse_cell(s)?),
        (None, Some(v)) if v.len() == 16 => {
            let mut c = [[[[0.0; 2]; 2]; 2]; 2];
            for (k, x) in v.iter().enumerate() {
                c[k / 8][(k / 4) % 2][(k / 2) % 2][k % 2] = *x;
            }
            c
        }
        _ => return Err(PyValueError::new_err("give a cell or a 2x2x2x2 stiffness tensor")),
    };
    let field = match points {
        Some(p) => Some(stroh_dislocation_field(&c, b, &p).map_err(err)?),
        None => None,
    };
    to_json(&json!({
        "C": c, "elastic_class": classify_elastic(&c), "elastic_class_status": CLASS_STATUS, "B": barnett_lothe_tensor(&c, n_quad),
        "K": dipole_prelog_coeff(&c, b, n_quad), "field": field,
    }))
}

/// Anti-plane screw field `u_z = (b_z / 2 pi) arg(x + p y)` for anti-plane moduli `k` (2x2 SPD).
#[pyfunction]
pub fn screw_dislocation_json(k: [[f64; 2]; 2], bz: f64, points: Vec<[f64; 2]>) -> PyResult<String> {
    to_json(&antiplane_screw_field(k, bz, &points).map_err(err)?)
}

/// Voronoi coordination and the `z_ideal - z` coordination charge.
#[pyfunction]
#[pyo3(signature = (pos, z_ideal=6, tol=0.05))]
pub fn voronoi_coordination_json(pos: Vec<[f64; 2]>, z_ideal: usize, tol: f64) -> PyResult<String> {
    to_json(&voronoi_coordination(&pos, z_ideal, tol).map_err(err)?)
}

/// Render position frames and encode them as a looping GIF89a.
#[pyfunction]
pub fn export_motion_gif_bytes<'py>(
    py: Python<'py>, frames: Vec<Vec<[f64; 2]>>, edges: Vec<(usize, usize)>, width: u16, height: u16, delay_cs: u16,
) -> PyResult<Bound<'py, PyBytes>> {
    let data = export_motion_gif(&frames, &edges, width, height, delay_cs).map_err(err)?;
    Ok(PyBytes::new(py, &data))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_defects_coords_json, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_seitz_occupancy_json, m)?)?;
    m.add_function(wrap_pyfunction!(nye_strain_json, m)?)?;
    m.add_function(wrap_pyfunction!(psi_n_order_list, m)?)?;
    m.add_function(wrap_pyfunction!(cna_signatures_json, m)?)?;
    m.add_function(wrap_pyfunction!(barnett_lothe_stroh_json, m)?)?;
    m.add_function(wrap_pyfunction!(export_motion_gif_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(screw_dislocation_json, m)?)?;
    m.add_function(wrap_pyfunction!(voronoi_coordination_json, m)?)?;
    Ok(())
}
