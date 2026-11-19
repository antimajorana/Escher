//! Periodic surfaces on the P / D / G labyrinth graphs, as complexes on the 3-torus
//! (conventional cubic cell, vertices identified across the periodic boundary):
//!
//! * Petrie–Coxeter regular skew polyhedra: `{6,4|4}` (the hexagons of the Kelvin / bitruncated
//!   cubic honeycomb, on the P surface) and `{6,6|3}` (the hexagons shared by truncated tetrahedra
//!   centred on the diamond sites, on the D surface).
//! * Nodal level-set meshes (marching tetrahedra on the Kuhn subdivision of a periodic grid).
//!
//! The Euler characteristic per side-preserving primitive cell is `2 - 2g` with `g = 3` for all
//! three classical surfaces (A. H. Schoen, NASA TN D-5541, 1970). The regular skew polyhedra are
//! from H. S. M. Coxeter, "Regular skew polyhedra in three and four dimensions, and their
//! topological analogues", Proc. London Math. Soc. (2) 43 (1937) 33-62. Truncated-tetrahedron
//! vertices: all permutations of `(±1, ±1, ±3)` with an even (resp. odd) number of minus signs.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use super::tpms_nets::TpmsKind;
use super::voronoi3d_dirichlet::lattice_voronoi_cell;
use super::{cross3, sub3, PointIndex};

#[derive(Clone, Debug, Serialize)]
pub struct PeriodicSurface {
    pub name: String,
    pub kind: TpmsKind,
    /// Vertices in fractional coordinates of the conventional cell, in `[0, 1)`.
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
    pub edges: Vec<[usize; 2]>,
    pub euler_conventional: i64,
    pub side_preserving_cells: u32,
    pub euler_primitive: i64,
    pub genus_primitive: i64,
    pub connected_components: usize,
    /// Every edge lies on exactly two faces.
    pub closed_manifold: bool,
    /// `[PROVED]` (exact Petrie-Coxeter complex) or `[COMPUTATIONAL EVIDENCE]` (nodal mesh).
    pub evidence: String,
}

const PC_EVIDENCE: &str = "[PROVED: exact V, E, F of the finite complex on the 3-torus of the conventional cell; genus 3 per side-preserving primitive cell, as for Coxeter's regular skew polyhedra (Proc. London Math. Soc. 43, 1937)]";
const NODAL_EVIDENCE: &str = "[COMPUTATIONAL EVIDENCE: topology of a marching-tetrahedra mesh of the nodal approximation on an n^3 grid; resolution-dependent, not a proof about the minimal surface]";

struct TorusComplex {
    index: PointIndex<3>,
    verts: Vec<[f64; 3]>,
    faces: Vec<Vec<usize>>,
    face_keys: HashSet<Vec<usize>>,
}

fn wrap(x: &[f64; 3]) -> [f64; 3] {
    x.map(|c| {
        let w = c.rem_euclid(1.0);
        if w > 1.0 - 1e-9 { 0.0 } else { w }
    })
}

impl TorusComplex {
    fn new() -> Self {
        Self { index: PointIndex::new(1e-7), verts: Vec::new(), faces: Vec::new(), face_keys: HashSet::new() }
    }

    fn add_face(&mut self, pts: &[[f64; 3]]) {
        let mut cyc = Vec::with_capacity(pts.len());
        for x in pts {
            let (i, new) = self.index.insert(wrap(x));
            if new {
                self.verts.push(wrap(x));
            }
            cyc.push(i);
        }
        let mut key = cyc.clone();
        key.sort_unstable();
        if self.face_keys.insert(key) {
            self.faces.push(cyc);
        }
    }

    fn finish(self, name: &str, kind: TpmsKind, evidence: &str) -> PeriodicSurface {
        let mut edge_count: HashMap<[usize; 2], usize> = HashMap::new();
        let nv = self.verts.len();
        let mut parent: Vec<usize> = (0..nv).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for f in &self.faces {
            for k in 0..f.len() {
                let (a, b) = (f[k], f[(k + 1) % f.len()]);
                *edge_count.entry([a.min(b), a.max(b)]).or_default() += 1;
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                parent[ra] = rb;
            }
        }
        let comps = (0..nv).filter(|&i| find(&mut parent, i) == i).count();
        let closed = edge_count.values().all(|&c| c == 2);
        let mut edges: Vec<[usize; 2]> = edge_count.into_keys().collect();
        edges.sort_unstable();
        let chi = nv as i64 - edges.len() as i64 + self.faces.len() as i64;
        let cells = kind.side_preserving_cells();
        let chi_p = chi / cells as i64;
        PeriodicSurface {
            name: name.into(),
            kind,
            vertices: self.verts,
            faces: self.faces,
            edges,
            euler_conventional: chi,
            side_preserving_cells: cells,
            euler_primitive: chi_p,
            genus_primitive: 1 - chi_p / 2,
            connected_components: comps,
            closed_manifold: closed,
            evidence: evidence.into(),
        }
    }
}

/// Hexagon of the truncated tetrahedron `(1/8)·{perms of (±1,±1,±3)}` (sign parity `parity`)
/// facing the bond direction `s`, translated by `centre`.
fn tt_hexagon(centre: &[f64; 3], s: &[f64; 3], parity: usize) -> Vec<[f64; 3]> {
    let mut pts = Vec::new();
    for perm in [[0, 1, 2], [1, 2, 0], [2, 0, 1]] {
        for signs in 0..8u32 {
            let base = [1.0, 1.0, 3.0];
            let mut x = [0.0; 3];
            for k in 0..3 {
                let sg = if (signs >> k) & 1 == 1 { -1.0 } else { 1.0 };
                x[perm[k]] = sg * base[k];
            }
            if signs.count_ones() as usize % 2 == parity && (x[0] * s[0] + x[1] * s[1] + x[2] * s[2] - 3.0).abs() < 1e-9 {
                pts.push(x.map(|c| c / 8.0));
            }
        }
    }
    // Order around the face normal s.
    let m = pts.len() as f64;
    let c = pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / m, a[1] + p[1] / m, a[2] + p[2] / m]);
    let u = sub3(&pts[0], &c);
    let v = cross3(s, &u);
    pts.sort_by(|a, b| {
        let (da, db) = (sub3(a, &c), sub3(b, &c));
        let ta = (da[0] * v[0] + da[1] * v[1] + da[2] * v[2]).atan2(da[0] * u[0] + da[1] * u[1] + da[2] * u[2]);
        let tb = (db[0] * v[0] + db[1] * v[1] + db[2] * v[2]).atan2(db[0] * u[0] + db[1] * u[1] + db[2] * u[2]);
        ta.total_cmp(&tb)
    });
    pts.iter().map(|x| [x[0] + centre[0], x[1] + centre[1], x[2] + centre[2]]).collect()
}

/// Petrie–Coxeter polyhedron on the P (`{6,4|4}`) or D (`{6,6|3}`) surface.
pub fn petrie_coxeter_surface(kind: TpmsKind) -> Result<PeriodicSurface, String> {
    let mut tc = TorusComplex::new();
    match kind {
        TpmsKind::P => {
            // BCC Voronoi cell with conventional cube of side 2: rescale by 1/2.
            let cell = lattice_voronoi_cell("BCC")?;
            for t in [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]] {
                for f in cell.faces.iter().filter(|f| f.len() == 6) {
                    let pts: Vec<[f64; 3]> = f.iter().map(|&v| [0, 1, 2].map(|k| (cell.vertices[v][k] + t[k]) / 2.0)).collect();
                    tc.add_face(&pts);
                }
            }
            Ok(tc.finish("{6,4|4} Petrie-Coxeter polyhedron (hexagons of the Kelvin honeycomb, P surface)", kind, PC_EVIDENCE))
        }
        TpmsKind::D => {
            let fcc = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]];
            let even = [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]];
            for a in fcc {
                let b = a.map(|c| c + 0.25);
                for s in even {
                    tc.add_face(&tt_hexagon(&a, &s, 1));
                    tc.add_face(&tt_hexagon(&b, &s.map(|c| -c), 0));
                }
            }
            Ok(tc.finish("{6,6|3} Petrie-Coxeter polyhedron (hexagons between truncated tetrahedra on diamond sites, D surface)", kind, PC_EVIDENCE))
        }
        TpmsKind::G => Err("no regular Petrie-Coxeter polyhedron lies on the gyroid; use nodal_surface_mesh".into()),
    }
}

/// Marching-tetrahedra mesh of the nodal surface of `kind` on an `n³` periodic grid.
pub fn nodal_surface_mesh(kind: TpmsKind, n: usize) -> Result<PeriodicSurface, String> {
    if !(4..=96).contains(&n) {
        return Err("n must be in 4..=96".into());
    }
    let delta = 0.37;
    let nf = n as f64;
    let idx = |i: usize, j: usize, k: usize| (i % n) * n * n + (j % n) * n + (k % n);
    let mut vals = vec![0.0; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                vals[idx(i, j, k)] = kind.nodal(&[(i as f64 + delta) / nf, (j as f64 + delta) / nf, (k as f64 + delta) / nf]);
            }
        }
    }
    let mut vert_of: HashMap<(usize, u8), usize> = HashMap::new();
    let mut verts: Vec<[f64; 3]> = Vec::new();
    let mut tris: Vec<Vec<usize>> = Vec::new();
    let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                for perm in perms {
                    let mut corner = [0u8; 4];
                    for s in 1..4 {
                        corner[s] = corner[s - 1] | (1 << perm[s - 1]);
                    }
                    let pos = |b: u8| [i + (b & 1) as usize, j + ((b >> 1) & 1) as usize, k + ((b >> 2) & 1) as usize];
                    let val = |b: u8| {
                        let p = pos(b);
                        vals[idx(p[0], p[1], p[2])]
                    };
                    let mut crossing = |a: u8, b: u8| -> usize {
                        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                        let p = pos(lo);
                        let key = (idx(p[0], p[1], p[2]), hi ^ lo);
                        *vert_of.entry(key).or_insert_with(|| {
                            let (fa, fb) = (val(lo), val(hi));
                            let t = fa / (fa - fb);
                            let d = hi ^ lo;
                            let x = [0, 1, 2].map(|c| ((p[c] % n) as f64 + delta + t * ((d >> c) & 1) as f64) / nf);
                            verts.push(x);
                            verts.len() - 1
                        })
                    };
                    let inside: Vec<u8> = corner.iter().copied().filter(|&b| val(b) > 0.0).collect();
                    let outside: Vec<u8> = corner.iter().copied().filter(|&b| val(b) <= 0.0).collect();
                    match inside.len() {
                        1 => tris.push(outside.iter().map(|&o| crossing(inside[0], o)).collect()),
                        3 => tris.push(inside.iter().map(|&o| crossing(outside[0], o)).collect()),
                        2 => {
                            let (a, b, c, d) = (inside[0], inside[1], outside[0], outside[1]);
                            let (ac, ad, bd, bc) = (crossing(a, c), crossing(a, d), crossing(b, d), crossing(b, c));
                            tris.push(vec![ac, ad, bd]);
                            tris.push(vec![ac, bd, bc]);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    // Vertices are already identified exactly by (grid point, edge direction); no positional merge.
    let mut tc = TorusComplex::new();
    tc.verts = verts.iter().map(wrap).collect();
    tc.faces = tris;
    let name = match kind {
        TpmsKind::P => "Schwarz P nodal surface cos X + cos Y + cos Z = 0",
        TpmsKind::D => "Schwarz D nodal surface cos X cos Y cos Z + sin X sin Y sin Z = 0",
        TpmsKind::G => "Schoen gyroid nodal surface sin X cos Y + sin Y cos Z + sin Z cos X = 0",
    };
    Ok(tc.finish(name, kind, NODAL_EVIDENCE))
}
