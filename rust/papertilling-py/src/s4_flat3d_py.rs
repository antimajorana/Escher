//! Stage 4 PyO3 bindings exposing torsion-free classification, 10 platycosms,
//! flat 3D slabs/chimneys, chiral `(n, m)` nanotubes/ribbons, symmetry-centred
//! flakes, proper `det = +1` 3D Niggli reduction, supercells, strain, and BCs.

use papertilling_core::bieberbach::{ConvOp, Rat, from_wallpaper_group};
use papertilling_core::flat3d::{
    Bond3D, BoundaryCondition, Cell3D, FlatSlabKind, analyze_torsion_free,
    apply_boundary_conditions_2d, apply_boundary_conditions_3d, build_flat_slab_3d,
    build_nanoribbon, build_nanotube_opts, build_platycosm_quotient, build_supercell_2d,
    build_supercell_3d, build_symmetric_flake, build_wigner_seitz_flake, cell_strain_tensor,
    check_platycosm_compatibility, classify_all_2d_torsion_free, classify_all_3d_torsion_free,
    is_torsion_free_conv_ops, list_platycosms, niggli_reduce_3d, parse_canonical_net_3d,
};
use papertilling_core::space3d::build_analyzer_3d;
use papertilling_core::types::{Bond2D, Cell2D};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;

fn py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Bond2dInput {
    Struct(Bond2D),
    Tuple([i64; 5]),
    Tuple4([i64; 4]),
}

#[derive(Deserialize)]
struct Cell2dInput {
    a1: [f64; 2],
    a2: [f64; 2],
    basis: Vec<[f64; 2]>,
    bonds: Vec<Bond2dInput>,
}

fn parse_cell2d(cell_json: &str) -> Result<Cell2D, String> {
    let raw: Cell2dInput = serde_json::from_str(cell_json).map_err(|e| e.to_string())?;
    let bonds = raw
        .bonds
        .into_iter()
        .map(|b| match b {
            Bond2dInput::Struct(bd) => bd,
            Bond2dInput::Tuple([i, j, n1, n2, k]) => Bond2D {
                i: i as usize,
                j: j as usize,
                n1: n1 as i32,
                n2: n2 as i32,
                kind: k as usize,
            },
            Bond2dInput::Tuple4([i, j, n1, n2]) => Bond2D {
                i: i as usize,
                j: j as usize,
                n1: n1 as i32,
                n2: n2 as i32,
                kind: 0,
            },
        })
        .collect();
    Ok(Cell2D { a1: raw.a1, a2: raw.a2, basis: raw.basis, bonds })
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Bond3dInput {
    Struct(Bond3D),
    Tuple([i64; 6]),
    Tuple5([i64; 5]),
}

#[derive(Deserialize)]
struct Cell3dInput {
    lattice: [[f64; 3]; 3],
    basis: Vec<[f64; 3]>,
    bonds: Vec<Bond3dInput>,
    #[serde(default)]
    space_group_num: Option<u16>,
    #[serde(default)]
    site_kinds: Vec<u8>,
}

fn parse_cell3d_or_net(net_or_cell_json: &str) -> Result<Cell3D, String> {
    let s = net_or_cell_json.trim();
    if s.starts_with('{') {
        let raw: Cell3dInput = serde_json::from_str(s).map_err(|e| e.to_string())?;
        let bonds = raw
            .bonds
            .into_iter()
            .map(|b| match b {
                Bond3dInput::Struct(bd) => bd,
                Bond3dInput::Tuple([i, j, n1, n2, n3, k]) => Bond3D {
                    i: i as usize,
                    j: j as usize,
                    n1: n1 as i32,
                    n2: n2 as i32,
                    n3: n3 as i32,
                    kind: k as usize,
                },
                Bond3dInput::Tuple5([i, j, n1, n2, n3]) => Bond3D {
                    i: i as usize,
                    j: j as usize,
                    n1: n1 as i32,
                    n2: n2 as i32,
                    n3: n3 as i32,
                    kind: 0,
                },
            })
            .collect();
        let site_kinds = if raw.site_kinds.is_empty() {
            vec![0_u8; raw.basis.len()]
        } else {
            raw.site_kinds
        };
        Ok(Cell3D {
            lattice: raw.lattice,
            basis: raw.basis,
            bonds,
            space_group_num: raw.space_group_num,
            site_kinds,
        })
    } else {
        let name = s.trim_matches('"');
        Ok(Cell3D::from_canonical_net(parse_canonical_net_3d(name)?))
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum BcInput {
    Native(BoundaryCondition),
    Tagged { kind: String, #[serde(default)] phase_rad: f64 },
}

fn parse_bc_vec(bcs_json: &str, expected: usize) -> Result<Vec<BoundaryCondition>, String> {
    let raw: Vec<BcInput> = serde_json::from_str(bcs_json).map_err(|e| e.to_string())?;
    if raw.len() != expected {
        return Err(format!("expected {expected} boundary conditions, got {}", raw.len()));
    }
    raw.into_iter()
        .map(|b| match b {
            BcInput::Native(bc) => Ok(bc),
            BcInput::Tagged { kind, phase_rad } => match kind.to_lowercase().as_str() {
                "open" => Ok(BoundaryCondition::Open),
                "periodic" => Ok(BoundaryCondition::Periodic),
                "antiperiodic" => Ok(BoundaryCondition::Antiperiodic),
                "twisted" => Ok(BoundaryCondition::Twisted { phase_rad }),
                _ => Err(format!("unknown boundary condition {kind:?}")),
            },
        })
        .collect()
}

#[derive(Deserialize)]
struct PyConvOp {
    r: Vec<Vec<i64>>,
    t: Vec<[i64; 2]>,
}

#[derive(Deserialize)]
struct PySpaceSpec {
    dim: usize,
    #[serde(default)]
    number: i32,
    #[serde(default = "default_custom")]
    symbol: String,
    ops: Vec<PyConvOp>,
    tb_rows: Vec<Vec<[i64; 2]>>,
}

fn default_custom() -> String {
    "custom".to_string()
}

#[pyfunction]
pub fn flat3d_torsion_free_wallpaper_json() -> PyResult<String> {
    let rep = classify_all_2d_torsion_free().map_err(py_err)?;
    serde_json::to_string(&rep).map_err(py_err)
}

#[pyfunction]
pub fn flat3d_torsion_free_space_groups_json() -> PyResult<String> {
    let rep = classify_all_3d_torsion_free().map_err(py_err)?;
    serde_json::to_string(&rep).map_err(py_err)
}

#[pyfunction]
pub fn flat3d_is_torsion_free_json(spec_json: &str) -> PyResult<String> {
    let s = spec_json.trim();
    let rep = if s.starts_with('{') {
        let spec: PySpaceSpec = serde_json::from_str(s).map_err(py_err)?;
        let ops: Vec<ConvOp> = spec
            .ops
            .into_iter()
            .map(|op| ConvOp { r: op.r, t: op.t.into_iter().map(|[n, d]| Rat::new(n, d)).collect() })
            .collect();
        let tb: Vec<Vec<Rat>> = spec
            .tb_rows
            .into_iter()
            .map(|row| row.into_iter().map(|[n, d]| Rat::new(n, d)).collect())
            .collect();
        is_torsion_free_conv_ops(spec.dim, spec.number, &spec.symbol, &ops, &tb).map_err(py_err)?
    } else if let Ok(num) = s.trim_matches('"').parse::<u16>() {
        let sg = build_analyzer_3d(num).map_err(py_err)?;
        analyze_torsion_free(&sg)
    } else {
        let sym = s.trim_matches('"');
        let sg = from_wallpaper_group(sym).map_err(py_err)?;
        analyze_torsion_free(&sg)
    };
    serde_json::to_string(&rep).map_err(py_err)
}

#[pyfunction]
pub fn flat3d_list_platycosms_json() -> PyResult<String> {
    serde_json::to_string(list_platycosms()).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (net_or_cell_json, platycosm, supercell=None, ignore_site_kinds=false))]
pub fn flat3d_check_platycosm_json(
    net_or_cell_json: &str,
    platycosm: &str,
    supercell: Option<[usize; 3]>,
    ignore_site_kinds: bool,
) -> PyResult<String> {
    let cell = parse_cell3d_or_net(net_or_cell_json).map_err(py_err)?;
    let res = check_platycosm_compatibility(&cell, platycosm, supercell, ignore_site_kinds).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (net_or_cell_json, platycosm, supercell=None, ignore_site_kinds=false))]
pub fn flat3d_build_platycosm_quotient_json(
    net_or_cell_json: &str,
    platycosm: &str,
    supercell: Option<[usize; 3]>,
    ignore_site_kinds: bool,
) -> PyResult<String> {
    let cell = parse_cell3d_or_net(net_or_cell_json).map_err(py_err)?;
    let res = build_platycosm_quotient(&cell, platycosm, supercell, ignore_site_kinds).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (net_or_cell_json, kind, repeats))]
pub fn flat3d_build_slab_3d_json(
    net_or_cell_json: &str,
    kind: &str,
    repeats: [usize; 3],
) -> PyResult<String> {
    let cell = parse_cell3d_or_net(net_or_cell_json).map_err(py_err)?;
    let k = FlatSlabKind::from_name(kind).map_err(py_err)?;
    let res = build_flat_slab_3d(&cell, k, repeats).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (cell_json, n, m, periods_z=1, periodic_z=false, standard_hex_60=false))]
pub fn flat3d_build_nanotube_json(
    cell_json: &str,
    n: i32,
    m: i32,
    periods_z: usize,
    periodic_z: bool,
    standard_hex_60: bool,
) -> PyResult<String> {
    let cell = parse_cell2d(cell_json).map_err(py_err)?;
    let res = build_nanotube_opts(&cell, n, m, periods_z, periodic_z, standard_hex_60).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (cell_json, chiral_dir, width_cells, length_cells, periodic_along=false))]
pub fn flat3d_build_nanoribbon_json(
    cell_json: &str,
    chiral_dir: [i32; 2],
    width_cells: usize,
    length_cells: usize,
    periodic_along: bool,
) -> PyResult<String> {
    let cell = parse_cell2d(cell_json).map_err(py_err)?;
    let res = build_nanoribbon(&cell, chiral_dir, width_cells, length_cells, periodic_along).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (group, cell_json, centre, n_fold_polygon=None, radius=2.5))]
pub fn flat3d_build_symmetric_flake_json(
    group: &str,
    cell_json: &str,
    centre: [f64; 2],
    n_fold_polygon: Option<usize>,
    radius: f64,
) -> PyResult<String> {
    let cell = parse_cell2d(cell_json).map_err(py_err)?;
    let res = build_symmetric_flake(group, &cell, centre, n_fold_polygon, radius).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (group, cell_json, centre, scale=1.5))]
pub fn flat3d_build_wigner_seitz_flake_json(
    group: &str,
    cell_json: &str,
    centre: [f64; 2],
    scale: f64,
) -> PyResult<String> {
    let cell = parse_cell2d(cell_json).map_err(py_err)?;
    let res = build_wigner_seitz_flake(group, &cell, centre, scale).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
pub fn flat3d_niggli_reduce_3d_json(lattice_json: &str) -> PyResult<String> {
    let lat: [[f64; 3]; 3] = serde_json::from_str(lattice_json).map_err(py_err)?;
    let res = niggli_reduce_3d(lat).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
pub fn flat3d_build_supercell_3d_json(net_or_cell_json: &str, celltran_json: &str) -> PyResult<String> {
    let cell = parse_cell3d_or_net(net_or_cell_json).map_err(py_err)?;
    let mat: [[i32; 3]; 3] = serde_json::from_str(celltran_json).map_err(py_err)?;
    let res = build_supercell_3d(&cell, mat).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
pub fn flat3d_build_supercell_2d_json(cell_json: &str, celltran_json: &str) -> PyResult<String> {
    let cell = parse_cell2d(cell_json).map_err(py_err)?;
    let mat: [[i32; 2]; 2] = serde_json::from_str(celltran_json).map_err(py_err)?;
    let res = build_supercell_2d(&cell, mat).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
pub fn flat3d_cell_strain_tensor_json(ref_lat_json: &str, def_lat_json: &str) -> PyResult<String> {
    let r: [[f64; 3]; 3] = serde_json::from_str(ref_lat_json).map_err(py_err)?;
    let d: [[f64; 3]; 3] = serde_json::from_str(def_lat_json).map_err(py_err)?;
    let res = cell_strain_tensor(r, d).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
pub fn flat3d_apply_boundary_conditions_3d_json(
    net_or_cell_json: &str,
    repeats: [usize; 3],
    bcs_json: &str,
) -> PyResult<String> {
    let cell = parse_cell3d_or_net(net_or_cell_json).map_err(py_err)?;
    let bcs = parse_bc_vec(bcs_json, 3).map_err(py_err)?;
    let res = apply_boundary_conditions_3d(&cell, repeats, [bcs[0], bcs[1], bcs[2]]).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
pub fn flat3d_apply_boundary_conditions_2d_json(
    cell_json: &str,
    repeats: [usize; 2],
    bcs_json: &str,
) -> PyResult<String> {
    let cell = parse_cell2d(cell_json).map_err(py_err)?;
    let bcs = parse_bc_vec(bcs_json, 2).map_err(py_err)?;
    let res = apply_boundary_conditions_2d(&cell, repeats, [bcs[0], bcs[1]]).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

/// Register Stage 4 `flat3d` PyO3 functions onto `papertilling_rs`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(flat3d_torsion_free_wallpaper_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_torsion_free_space_groups_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_is_torsion_free_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_list_platycosms_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_check_platycosm_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_build_platycosm_quotient_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_build_slab_3d_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_build_nanotube_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_build_nanoribbon_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_build_symmetric_flake_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_build_wigner_seitz_flake_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_niggli_reduce_3d_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_build_supercell_3d_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_build_supercell_2d_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_cell_strain_tensor_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_apply_boundary_conditions_3d_json, m)?)?;
    m.add_function(wrap_pyfunction!(flat3d_apply_boundary_conditions_2d_json, m)?)?;
    Ok(())
}
