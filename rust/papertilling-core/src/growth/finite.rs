//! `FiniteComplex`: a complex built from explicit tables, for defected crystals (T7) and tests.
//!
//! JSON forms read by [`FiniteComplex::from_json`] (and by `papertilling.growth.grow_finite`):
//! - `{"polygons": [[v, ...], ...], "rim": [v, ...]}`: 2D tiles as vertex cycles; edges are
//!   the consecutive vertex pairs (so a digon or a repeated pair needs the table form);
//! - `{"polyhedra": [[[v, ...], ...], ...], "rim": [...]}`: 3D cells as lists of face cycles;
//!   faces are keyed by their vertex set, edges by consecutive pairs;
//! - the table form `{"cell_vertices": [[v..]..], "cell_edges": [[e..]..],
//!   "edge_vertices": [[u, w]..], "cell_faces": [[f..]..], "face_vertices": [[v..]..],
//!   "rim": [...]}` (`cell_faces`, `face_vertices` and `rim` optional). Edge ids index
//!   `edge_vertices`, face ids index `face_vertices`, cell ids are row numbers.
//!
//! Vertices are `u32` labels. `rim` lists vertices whose cells are cut off by the end of the
//! table: growth stops with `Stopped::Rim` before it uses one of them.
//! Seeds ([`FiniteComplex::seed_from_json`]): `{"kind": "vertex", "at": v}`,
//! `{"kind": "edge", "at": e}` (edge id) or `"at": [u, w]` (a vertex pair naming one edge),
//! `{"kind": "face", "at": [v, ...]}`, `{"kind": "cell", "at": c}`. A Dirichlet seed is refused:
//! a table has no coordinates.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;
use serde_json::Value;

use super::engine::{seed_cells, SeedAt};
use super::{CellComplex, GrowthError};

/// The table form (module docs).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct FiniteTables {
    pub cell_vertices: Vec<Vec<u32>>,
    pub cell_edges: Vec<Vec<u32>>,
    pub edge_vertices: Vec<[u32; 2]>,
    #[serde(default)]
    pub cell_faces: Vec<Vec<u32>>,
    #[serde(default)]
    pub face_vertices: Vec<Vec<u32>>,
    #[serde(default)]
    pub rim: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ComplexJson {
    Polygons {
        polygons: Vec<Vec<u32>>,
        #[serde(default)]
        rim: Vec<u32>,
    },
    Polyhedra {
        polyhedra: Vec<Vec<Vec<u32>>>,
        #[serde(default)]
        rim: Vec<u32>,
    },
    Tables(FiniteTables),
}

/// A finite complex with its incidence inverted for `cells_at`, `cells_on_edge`, `cells_on_face`.
#[derive(Clone, Debug)]
pub struct FiniteComplex {
    t: FiniteTables,
    vertex_cells: HashMap<u32, Vec<u32>>,
    edge_cells: Vec<Vec<u32>>,
    face_cells: Vec<Vec<u32>>,
    face_index: HashMap<Vec<u32>, u32>,
    rim: HashSet<u32>,
}

impl FiniteComplex {
    /// Check the tables and invert them.
    pub fn new(mut t: FiniteTables) -> Result<Self, String> {
        if t.cell_faces.is_empty() {
            t.cell_faces = vec![Vec::new(); t.cell_vertices.len()];
        }
        if t.cell_edges.len() != t.cell_vertices.len() || t.cell_faces.len() != t.cell_vertices.len() {
            return Err("cell_vertices, cell_edges and cell_faces need one row per cell".into());
        }
        let mut vertex_cells: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut edge_cells = vec![Vec::new(); t.edge_vertices.len()];
        let mut face_cells = vec![Vec::new(); t.face_vertices.len()];
        for (c, vs) in t.cell_vertices.iter().enumerate() {
            let c32 = c as u32;
            for &v in vs {
                vertex_cells.entry(v).or_default().push(c32);
            }
            for &e in &t.cell_edges[c] {
                let ok = t.edge_vertices.get(e as usize).is_some_and(|p| p.iter().all(|v| vs.contains(v)));
                if !ok {
                    return Err(format!("edge {e} of cell {c} is not an edge between vertices of that cell"));
                }
                edge_cells[e as usize].push(c32);
            }
            for &f in &t.cell_faces[c] {
                let ok = t.face_vertices.get(f as usize).is_some_and(|fv| fv.iter().all(|v| vs.contains(v)));
                if !ok {
                    return Err(format!("face {f} of cell {c} is not a face on vertices of that cell"));
                }
                face_cells[f as usize].push(c32);
            }
        }
        if t.rim.iter().any(|v| !vertex_cells.contains_key(v)) {
            return Err("a rim vertex is in no cell".into());
        }
        let face_index = t
            .face_vertices
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut k = f.clone();
                k.sort_unstable();
                (k, i as u32)
            })
            .collect();
        let rim = t.rim.iter().copied().collect();
        Ok(Self { t, vertex_cells, edge_cells, face_cells, face_index, rim })
    }

    /// 2D tiles given as vertex cycles.
    pub fn from_polygons(polys: &[Vec<u32>], rim: &[u32]) -> Result<Self, String> {
        let cells: Vec<Vec<Vec<u32>>> = polys.iter().map(|p| vec![p.clone()]).collect();
        Self::from_cycles(&cells, rim, false)
    }

    /// 3D cells given as lists of face cycles.
    pub fn from_polyhedra(cells: &[Vec<Vec<u32>>], rim: &[u32]) -> Result<Self, String> {
        Self::from_cycles(cells, rim, true)
    }

    fn from_cycles(cells: &[Vec<Vec<u32>>], rim: &[u32], three_d: bool) -> Result<Self, String> {
        let mut eid: HashMap<[u32; 2], u32> = HashMap::new();
        let mut fid: HashMap<Vec<u32>, u32> = HashMap::new();
        let mut t = FiniteTables { rim: rim.to_vec(), ..Default::default() };
        for cell in cells {
            let (mut vs, mut es, mut fs) = (Vec::new(), Vec::new(), Vec::new());
            for cyc in cell {
                let distinct: HashSet<&u32> = cyc.iter().collect();
                if cyc.len() < 3 || distinct.len() != cyc.len() {
                    return Err(format!("face cycle {cyc:?} needs at least 3 distinct vertices"));
                }
                for &v in cyc {
                    if !vs.contains(&v) {
                        vs.push(v);
                    }
                }
                for i in 0..cyc.len() {
                    let (a, b) = (cyc[i], cyc[(i + 1) % cyc.len()]);
                    let pair = [a.min(b), a.max(b)];
                    let next = eid.len() as u32;
                    let e = *eid.entry(pair).or_insert(next);
                    if e == next {
                        t.edge_vertices.push(pair);
                    }
                    if !es.contains(&e) {
                        es.push(e);
                    }
                }
                if three_d {
                    let mut key = cyc.clone();
                    key.sort_unstable();
                    let next = fid.len() as u32;
                    let f = *fid.entry(key).or_insert(next);
                    if f == next {
                        t.face_vertices.push(cyc.clone());
                    }
                    fs.push(f);
                }
            }
            t.cell_vertices.push(vs);
            t.cell_edges.push(es);
            t.cell_faces.push(fs);
        }
        Self::new(t)
    }

    /// Parse one of the JSON forms in the module docs.
    pub fn from_json(s: &str) -> Result<Self, String> {
        match serde_json::from_str::<ComplexJson>(s).map_err(|e| format!("complex JSON: {e}"))? {
            ComplexJson::Polygons { polygons, rim } => Self::from_polygons(&polygons, &rim),
            ComplexJson::Polyhedra { polyhedra, rim } => Self::from_polyhedra(&polyhedra, &rim),
            ComplexJson::Tables(t) => Self::new(t),
        }
    }

    pub fn num_cells(&self) -> usize {
        self.t.cell_vertices.len()
    }

    /// The one edge between `u` and `w`; an error when there is none or more than one.
    pub fn edge_between(&self, u: u32, w: u32) -> Result<u32, String> {
        let hits: Vec<u32> = (0..self.t.edge_vertices.len() as u32)
            .filter(|&e| {
                let p = self.t.edge_vertices[e as usize];
                (p[0], p[1]) == (u, w) || (p[0], p[1]) == (w, u)
            })
            .collect();
        match hits.as_slice() {
            [e] => Ok(*e),
            _ => Err(format!("vertex pair [{u}, {w}] names {} edges; give the edge id", hits.len())),
        }
    }

    /// The seed set for a JSON seed (module docs).
    pub fn seed_from_json(&mut self, seed: &Value) -> Result<Vec<u32>, String> {
        let kind = seed.get("kind").and_then(Value::as_str).unwrap_or("");
        let at = seed.get("at").cloned().unwrap_or(Value::Null);
        let as_u32 = |v: &Value| v.as_u64().and_then(|n| u32::try_from(n).ok());
        let list = |v: &Value| -> Option<Vec<u32>> { v.as_array()?.iter().map(as_u32).collect() };
        let pos: SeedAt<u32, u32, u32> = match kind {
            "vertex" => SeedAt::Vertex(as_u32(&at).ok_or("vertex seed needs \"at\": a vertex id")?),
            "cell" => {
                let c = as_u32(&at).ok_or("cell seed needs \"at\": a cell id")?;
                if c as usize >= self.num_cells() {
                    return Err(format!("cell {c} is not in the table"));
                }
                SeedAt::Cell(c)
            }
            "edge" => match (as_u32(&at), list(&at)) {
                (Some(e), _) => SeedAt::Edge(e),
                (None, Some(p)) if p.len() == 2 => SeedAt::Edge(self.edge_between(p[0], p[1])?),
                _ => return Err("edge seed needs \"at\": an edge id or a vertex pair".into()),
            },
            "face" => SeedAt::Face(list(&at).ok_or("face seed needs \"at\": a list of vertex ids")?),
            "dirichlet" => {
                return Err("a Dirichlet seed needs coordinates; a finite table has none (use vertex, edge, face or cell)".into())
            }
            other => return Err(format!("unknown seed kind {other:?}")),
        };
        seed_cells(self, &pos).map_err(|e| format!("seed: {e:?}"))
    }
}

impl CellComplex for FiniteComplex {
    type Cell = u32;
    type Vertex = u32;
    type Key = u32;

    fn vertices_of(&self, c: u32) -> Vec<u32> {
        self.t.cell_vertices[c as usize].clone()
    }
    fn edges_of(&self, c: u32) -> Vec<u32> {
        self.t.cell_edges[c as usize].clone()
    }
    fn faces_of(&self, c: u32) -> Vec<u32> {
        self.t.cell_faces[c as usize].clone()
    }
    fn on_rim(&self, v: u32) -> bool {
        self.rim.contains(&v)
    }
    fn cells_at(&mut self, v: u32) -> Result<Vec<u32>, GrowthError> {
        self.vertex_cells.get(&v).cloned().ok_or(GrowthError::OutOfRange { reason: "vertex not in the table" })
    }
    fn cells_on_edge(&mut self, e: u32) -> Result<Vec<u32>, GrowthError> {
        self.edge_cells.get(e as usize).cloned().ok_or(GrowthError::OutOfRange { reason: "edge not in the table" })
    }
    fn cells_on_face(&mut self, f: &[u32]) -> Result<Vec<u32>, GrowthError> {
        let mut k = f.to_vec();
        k.sort_unstable();
        match self.face_index.get(&k) {
            Some(&i) => Ok(self.face_cells[i as usize].clone()),
            None => Err(GrowthError::OutOfRange { reason: "face not in the table" }),
        }
    }
}
