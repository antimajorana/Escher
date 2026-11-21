//! PyO3 bindings for `papertilling_core::noneuclidean2d` (Stage 5).

use papertilling_core::noneuclidean2d::{
    H2Kernel, H2LargePatch, H2SurfaceKind, RegionFilter, SnubHandedness, WythoffOp,
    build_h2_surface_mesh, build_hyperbolic_large, build_unified_pq, build_unified_pq_seeded, build_wythoff, delta_pq,
    embed_tiling_on_surface, geodesic_arc_poincare, geodesic_arc_uhp, geometry_kind_pq,
    hyp_distance_poincare, hyp_distance_uhp, hyperboloid_to_poincare_exact, klein_to_poincare,
    list_tgcell_quotients, map_poincare_to_surface, orbifold_to_wallpaper, parse_orbifold,
    parse_vertex_config, poincare_to_hyperboloid_exact, poincare_to_klein, poincare_to_uhp,
    tgcell_quotient, uhp_to_poincare, unfold_papercraft_net, wallpaper_orbifold_table,
    wallpaper_to_orbifold,
};
use papertilling_core::growth::SeedKind;
use papertilling_core::hyperbolic::{HyperbolicStyle, build_hyperbolic_tiling_seeded};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_wythoff_op(op: &str, chirality: &str) -> Result<WythoffOp, String> {
    match op.trim().to_ascii_lowercase().as_str() {
        "regular" => Ok(WythoffOp::Regular),
        "dual" => Ok(WythoffOp::Dual),
        "rectified" => Ok(WythoffOp::Rectified),
        "truncated" => Ok(WythoffOp::Truncated),
        "bitruncated" => Ok(WythoffOp::Bitruncated),
        "cantellated" => Ok(WythoffOp::Cantellated),
        "omnitruncated" => Ok(WythoffOp::Omnitruncated),
        "snub" => {
            let hand = match chirality.trim().to_ascii_lowercase().as_str() {
                "right" | "r" => SnubHandedness::Right,
                "left" | "l" | "" => SnubHandedness::Left,
                _ => return Err(format!("unknown snub chirality {chirality:?}; use 'Left' or 'Right'")),
            };
            Ok(WythoffOp::Snub { chirality: hand })
        }
        _ => Err(format!("unknown Wythoff operation {op:?}")),
    }
}

fn parse_surface_kind(kind: &str, twist: f64) -> Result<H2SurfaceKind, String> {
    match kind.trim().to_ascii_lowercase().as_str() {
        "hyperboloidbowl" | "bowl" => Ok(H2SurfaceKind::HyperboloidBowl),
        "hemisphere" => Ok(H2SurfaceKind::Hemisphere),
        "pseudosphere" => Ok(H2SurfaceKind::Pseudosphere),
        "dini" => Ok(H2SurfaceKind::Dini { twist }),
        "kuen" => Ok(H2SurfaceKind::Kuen),
        _ => Err(format!("unknown H2 surface kind {kind:?}")),
    }
}

fn parse_h2_kernel(kernel: &str) -> Result<H2Kernel, String> {
    match kernel.trim().to_ascii_lowercase().as_str() {
        "graphcombinatorial" | "graph" | "combinatorial" => Ok(H2Kernel::GraphCombinatorial),
        "dunhamhyperboloid" | "dunham" | "hyperboloid" => Ok(H2Kernel::DunhamHyperboloid),
        "spatialhashpoincare" | "spatialhash" | "poincare" => Ok(H2Kernel::SpatialHashPoincare),
        _ => Err(format!("unknown H2 kernel {kernel:?}")),
    }
}

#[pyfunction]
#[pyo3(name = "delta_pq")]
pub fn py_delta_pq(p: usize, q: usize) -> PyResult<i64> {
    delta_pq(p, q).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "geometry_kind_pq")]
pub fn py_geometry_kind_pq(p: usize, q: usize) -> PyResult<String> {
    let k = geometry_kind_pq(p, q).map_err(py_err)?;
    Ok(format!("{k:?}"))
}

#[pyfunction]
#[pyo3(signature = (p, q, layers=2))]
pub fn unified_pq_tiling_json(p: usize, q: usize, layers: usize) -> PyResult<String> {
    let res = build_unified_pq(p, q, layers).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

/// Parse a seed name ("vertex", "edge", "face", "cell", "dirichlet") into `SeedKind` (C1 WP-G3).
fn parse_seed(seed: &str) -> PyResult<SeedKind> {
    serde_json::from_value(serde_json::Value::String(seed.to_lowercase()))
        .map_err(|_| PyValueError::new_err(format!("unknown seed {seed:?}; expected vertex, edge, face, cell or dirichlet")))
}

/// `{p, q}` in S² / E² / H² grown from `seed` (0-based layers; `build_unified_pq_seeded`, C1 WP-G3).
#[pyfunction]
#[pyo3(signature = (p, q, layers=2, seed="cell"))]
pub fn unified_pq_seeded_json(p: usize, q: usize, layers: usize, seed: &str) -> PyResult<String> {
    let res = build_unified_pq_seeded(p, q, layers, parse_seed(seed)?).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

/// Hyperbolic `{p, q}` patch grown from `seed` (`build_hyperbolic_tiling_seeded`, C1 WP-G3).
#[pyfunction]
#[pyo3(signature = (p=7, q=3, style="RegularPQ", layers=2, wedge_delta=0, seed="cell"))]
pub fn hyperbolic_tiling_seeded_json(
    p: usize,
    q: usize,
    style: &str,
    layers: usize,
    wedge_delta: i32,
    seed: &str,
) -> PyResult<String> {
    let st = match style {
        "RectifiedKagome" => HyperbolicStyle::RectifiedKagome,
        "DualQP" => HyperbolicStyle::DualQP,
        _ => HyperbolicStyle::RegularPQ,
    };
    let res = build_hyperbolic_tiling_seeded(p, q, st, layers, wedge_delta, parse_seed(seed)?).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
pub fn wallpaper_orbifold_table_json() -> PyResult<String> {
    serde_json::to_string(wallpaper_orbifold_table()).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "wallpaper_to_orbifold")]
pub fn py_wallpaper_to_orbifold(group: &str) -> PyResult<&'static str> {
    wallpaper_to_orbifold(group).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "orbifold_to_wallpaper")]
pub fn py_orbifold_to_wallpaper(symbol: &str) -> PyResult<&'static str> {
    orbifold_to_wallpaper(symbol).map_err(py_err)
}

#[pyfunction]
pub fn parse_orbifold_json(symbol: &str) -> PyResult<String> {
    let res = parse_orbifold(symbol).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (p, q, op="Regular", chirality="Left", layers=2))]
pub fn build_wythoff_json(p: usize, q: usize, op: &str, chirality: &str, layers: usize) -> PyResult<String> {
    let parsed_op = parse_wythoff_op(op, chirality).map_err(py_err)?;
    let res = build_wythoff(p, q, parsed_op, layers).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
pub fn parse_vertex_config_json(symbol: &str) -> PyResult<String> {
    let res = parse_vertex_config(symbol).map_err(py_err)?;
    serde_json::to_string(&res).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "poincare_to_uhp")]
pub fn py_poincare_to_uhp(z: [f64; 2]) -> PyResult<[f64; 2]> {
    poincare_to_uhp(z).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "uhp_to_poincare")]
pub fn py_uhp_to_poincare(w: [f64; 2]) -> PyResult<[f64; 2]> {
    uhp_to_poincare(w).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "poincare_to_klein")]
pub fn py_poincare_to_klein(z: [f64; 2]) -> [f64; 2] {
    poincare_to_klein(z)
}

#[pyfunction]
#[pyo3(name = "klein_to_poincare")]
pub fn py_klein_to_poincare(k: [f64; 2]) -> PyResult<[f64; 2]> {
    klein_to_poincare(k).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "poincare_to_hyperboloid_exact")]
pub fn py_poincare_to_hyperboloid_exact(z: [f64; 2]) -> PyResult<[f64; 3]> {
    poincare_to_hyperboloid_exact(z).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "hyperboloid_to_poincare_exact")]
pub fn py_hyperboloid_to_poincare_exact(x: [f64; 3]) -> [f64; 2] {
    hyperboloid_to_poincare_exact(x)
}

#[pyfunction]
#[pyo3(name = "hyp_distance_poincare")]
pub fn py_hyp_distance_poincare(z1: [f64; 2], z2: [f64; 2]) -> PyResult<f64> {
    hyp_distance_poincare(z1, z2).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "hyp_distance_uhp")]
pub fn py_hyp_distance_uhp(w1: [f64; 2], w2: [f64; 2]) -> PyResult<f64> {
    hyp_distance_uhp(w1, w2).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "geodesic_arc_poincare", signature = (z1, z2, steps=8))]
pub fn py_geodesic_arc_poincare(z1: [f64; 2], z2: [f64; 2], steps: usize) -> PyResult<Vec<[f64; 2]>> {
    geodesic_arc_poincare(z1, z2, steps).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "geodesic_arc_uhp", signature = (w1, w2, steps=8))]
pub fn py_geodesic_arc_uhp(w1: [f64; 2], w2: [f64; 2], steps: usize) -> PyResult<Vec<[f64; 2]>> {
    geodesic_arc_uhp(w1, w2, steps).map_err(py_err)
}

#[pyfunction]
#[pyo3(name = "map_poincare_to_surface", signature = (z, kind="HyperboloidBowl", twist=0.35))]
pub fn py_map_poincare_to_surface(z: [f64; 2], kind: &str, twist: f64) -> PyResult<[f64; 3]> {
    let sk = parse_surface_kind(kind, twist).map_err(py_err)?;
    map_poincare_to_surface(z, sk).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (kind="Pseudosphere", twist=0.35, u_steps=24, v_steps=16))]
pub fn build_h2_surface_mesh_json(kind: &str, twist: f64, u_steps: usize, v_steps: usize) -> PyResult<String> {
    let sk = parse_surface_kind(kind, twist).map_err(py_err)?;
    let mesh = build_h2_surface_mesh(sk, u_steps, v_steps).map_err(py_err)?;
    serde_json::to_string(&mesh).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (coords_poincare_json, faces_json, kind="HyperboloidBowl", twist=0.35))]
pub fn embed_tiling_on_surface_json(
    coords_poincare_json: &str,
    faces_json: &str,
    kind: &str,
    twist: f64,
) -> PyResult<String> {
    let coords: Vec<[f64; 2]> = serde_json::from_str(coords_poincare_json).map_err(py_err)?;
    let faces: Vec<Vec<usize>> = serde_json::from_str(faces_json).map_err(py_err)?;
    let sk = parse_surface_kind(kind, twist).map_err(py_err)?;
    let mesh = embed_tiling_on_surface(&coords, &faces, sk).map_err(py_err)?;
    serde_json::to_string(&mesh).map_err(py_err)
}

#[pyfunction]
pub fn unfold_papercraft_net_json(coords_3d_json: &str, faces_json: &str) -> PyResult<String> {
    let coords: Vec<[f64; 3]> = serde_json::from_str(coords_3d_json).map_err(py_err)?;
    let faces: Vec<Vec<usize>> = serde_json::from_str(faces_json).map_err(py_err)?;
    let net = unfold_papercraft_net(&coords, &faces).map_err(py_err)?;
    serde_json::to_string(&net).map_err(py_err)
}

#[pyfunction]
pub fn list_tgcell_quotients_json() -> PyResult<String> {
    serde_json::to_string(list_tgcell_quotients()).map_err(py_err)
}

#[pyfunction]
pub fn tgcell_quotient_json(name: &str) -> PyResult<String> {
    let q = tgcell_quotient(name).map_err(py_err)?;
    serde_json::to_string(&q).map_err(py_err)
}

#[pyclass(name = "H2LargePatch")]
pub struct PyH2LargePatch {
    inner: H2LargePatch,
}

#[pymethods]
impl PyH2LargePatch {
    #[new]
    #[pyo3(signature = (p=7, q=3, target_faces=100, kernel="SpatialHashPoincare"))]
    pub fn new(p: usize, q: usize, target_faces: usize, kernel: &str) -> PyResult<Self> {
        let k = parse_h2_kernel(kernel).map_err(py_err)?;
        let inner = build_hyperbolic_large(p, q, target_faces, k).map_err(py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    pub fn p(&self) -> usize {
        self.inner.p
    }
    #[getter]
    pub fn q(&self) -> usize {
        self.inner.q
    }
    #[getter]
    pub fn kernel(&self) -> String {
        format!("{:?}", self.inner.kernel)
    }
    #[getter]
    pub fn n_active_faces(&self) -> usize {
        self.inner.n_active_faces
    }
    #[getter]
    pub fn total_faces(&self) -> usize {
        self.inner.faces.len()
    }
    #[getter]
    pub fn n_vertices(&self) -> usize {
        self.inner.n_vertices
    }
    #[getter]
    pub fn n_edges(&self) -> usize {
        self.inner.n_edges
    }

    pub fn summary_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.summary()).map_err(py_err)
    }

    pub fn hyperbolic_ball_query(&self, center: [f64; 2], radius: f64) -> PyResult<Vec<usize>> {
        self.inner.hyperbolic_ball_query(center, radius).map_err(py_err)
    }

    pub fn edge_neighbors(&self, face_id: usize) -> Vec<usize> {
        self.inner.edge_neighbors(face_id)
    }

    pub fn vertex_neighbors(&self, face_id: usize) -> Vec<usize> {
        self.inner.vertex_neighbors(face_id)
    }

    pub fn remove_face(&mut self, face_id: usize) -> bool {
        self.inner.remove_face(face_id)
    }

    pub fn restore_face(&mut self, face_id: usize) -> bool {
        self.inner.restore_face(face_id)
    }

    pub fn add_face_across_edge(&mut self, face_id: usize, side: usize) -> PyResult<usize> {
        self.inner.add_face_across_edge(face_id, side).map_err(py_err)
    }

    #[pyo3(signature = (kind, p1=0.0, p2=0.0, p3=0.0))]
    pub fn filter_by_region(&mut self, kind: &str, p1: f64, p2: f64, p3: f64) -> PyResult<usize> {
        let f = match kind.trim().to_ascii_lowercase().as_str() {
            "poincareradiusmax" => RegionFilter::PoincareRadiusMax(p1),
            "hyperbolicradiusmax" => RegionFilter::HyperbolicRadiusMax(p1),
            "shellrange" => RegionFilter::ShellRange(p1.max(0.0) as usize, p2.max(0.0) as usize),
            "diskball" => RegionFilter::DiskBall { center: [p1, p2], radius: p3 },
            "halfplane" => RegionFilter::HalfPlane { nx: p1, ny: p2, offset: p3 },
            _ => return Err(py_err(format!("unknown region filter {kind:?}"))),
        };
        Ok(self.inner.filter_by_region(f))
    }

    pub fn face_vertices(&self, face_id: usize) -> PyResult<Vec<usize>> {
        self.inner
            .faces
            .get(face_id)
            .cloned()
            .ok_or_else(|| py_err(format!("face_id {face_id} out of range")))
    }

    pub fn face_centre_poincare(&self, face_id: usize) -> PyResult<[f64; 2]> {
        self.inner
            .face_centres_poincare
            .get(face_id)
            .copied()
            .ok_or_else(|| py_err(format!("face_id {face_id} out of range")))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_delta_pq, m)?)?;
    m.add_function(wrap_pyfunction!(py_geometry_kind_pq, m)?)?;
    m.add_function(wrap_pyfunction!(unified_pq_tiling_json, m)?)?;
    m.add_function(wrap_pyfunction!(unified_pq_seeded_json, m)?)?;
    m.add_function(wrap_pyfunction!(hyperbolic_tiling_seeded_json, m)?)?;
    m.add_function(wrap_pyfunction!(wallpaper_orbifold_table_json, m)?)?;
    m.add_function(wrap_pyfunction!(py_wallpaper_to_orbifold, m)?)?;
    m.add_function(wrap_pyfunction!(py_orbifold_to_wallpaper, m)?)?;
    m.add_function(wrap_pyfunction!(parse_orbifold_json, m)?)?;
    m.add_function(wrap_pyfunction!(build_wythoff_json, m)?)?;
    m.add_function(wrap_pyfunction!(parse_vertex_config_json, m)?)?;
    m.add_function(wrap_pyfunction!(py_poincare_to_uhp, m)?)?;
    m.add_function(wrap_pyfunction!(py_uhp_to_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(py_poincare_to_klein, m)?)?;
    m.add_function(wrap_pyfunction!(py_klein_to_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(py_poincare_to_hyperboloid_exact, m)?)?;
    m.add_function(wrap_pyfunction!(py_hyperboloid_to_poincare_exact, m)?)?;
    m.add_function(wrap_pyfunction!(py_hyp_distance_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(py_hyp_distance_uhp, m)?)?;
    m.add_function(wrap_pyfunction!(py_geodesic_arc_poincare, m)?)?;
    m.add_function(wrap_pyfunction!(py_geodesic_arc_uhp, m)?)?;
    m.add_function(wrap_pyfunction!(py_map_poincare_to_surface, m)?)?;
    m.add_function(wrap_pyfunction!(build_h2_surface_mesh_json, m)?)?;
    m.add_function(wrap_pyfunction!(embed_tiling_on_surface_json, m)?)?;
    m.add_function(wrap_pyfunction!(unfold_papercraft_net_json, m)?)?;
    m.add_function(wrap_pyfunction!(list_tgcell_quotients_json, m)?)?;
    m.add_function(wrap_pyfunction!(tgcell_quotient_json, m)?)?;
    m.add_class::<PyH2LargePatch>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_wrappers_roundtrip() {
        assert_eq!(py_delta_pq(4, 4).unwrap(), 0);
        assert_eq!(py_geometry_kind_pq(7, 3).unwrap(), "Hyperbolic");
        let orb: serde_json::Value = serde_json::from_str(&parse_orbifold_json("*237").unwrap()).unwrap();
        assert_eq!(orb["chi_num"], serde_json::json!(-1));
        assert_eq!(orb["chi_den"], serde_json::json!(84));
        let vc: serde_json::Value = serde_json::from_str(&parse_vertex_config_json("6.4.3.4").unwrap()).unwrap();
        assert_eq!(vc["canonical"], serde_json::json!("3.4.6.4"));
        let klein: serde_json::Value = serde_json::from_str(&tgcell_quotient_json("klein_7_3").unwrap()).unwrap();
        assert_eq!(klein["n_faces"], serde_json::json!(24));
        let patch = PyH2LargePatch::new(7, 3, 29, "SpatialHashPoincare").unwrap();
        assert_eq!(patch.n_active_faces(), 29);
    }
}
