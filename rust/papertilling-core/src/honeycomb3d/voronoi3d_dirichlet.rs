//! 3D Voronoi cells by half-space clipping, and Dirichlet fundamental domains of closed
//! 3-manifolds with face pairings: quotient cell counts, Euler characteristic and `H_1`.

use std::collections::HashSet;

use serde::Serialize;

use super::{cross3, dot3, norm3, sub3, PointIndex};

/// A convex polyhedron: ordered faces (counter-clockwise seen from outside).
#[derive(Clone, Debug, Default, Serialize)]
pub struct VoronoiPolyhedron3d {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
    /// The neighbour (relative to the origin) whose bisector carries each face.
    pub face_neighbours: Vec<[f64; 3]>,
    pub edges: Vec<[usize; 2]>,
}

impl VoronoiPolyhedron3d {
    pub fn counts(&self) -> (usize, usize, usize) {
        (self.vertices.len(), self.edges.len(), self.faces.len())
    }

    pub fn face_sizes(&self) -> Vec<usize> {
        let mut s: Vec<usize> = self.faces.iter().map(|f| f.len()).collect();
        s.sort_unstable();
        s
    }
}

struct Poly {
    pts: Vec<[f64; 3]>,
    tag: Option<usize>,
}

fn polygon_area_normal(pts: &[[f64; 3]]) -> [f64; 3] {
    let mut n = [0.0; 3];
    for i in 0..pts.len() {
        let c = cross3(&pts[i], &pts[(i + 1) % pts.len()]);
        for k in 0..3 {
            n[k] += 0.5 * c[k];
        }
    }
    n
}

/// Order coplanar points by angle around their centroid, counter-clockwise about `normal`.
fn order_around(mut pts: Vec<[f64; 3]>, normal: &[f64; 3]) -> Vec<[f64; 3]> {
    let m = pts.len() as f64;
    let c = pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / m, a[1] + p[1] / m, a[2] + p[2] / m]);
    let u = sub3(&pts[0], &c);
    let u = u.map(|x| x / norm3(&u).max(1e-300));
    let v = cross3(normal, &u);
    pts.sort_by(|a, b| {
        let (da, db) = (sub3(a, &c), sub3(b, &c));
        let ta = dot3(&da, &v).atan2(dot3(&da, &u));
        let tb = dot3(&db, &v).atan2(dot3(&db, &u));
        ta.total_cmp(&tb)
    });
    pts
}

fn clip(polys: Vec<Poly>, n: &[f64; 3], h: f64, tag: usize, eps: f64) -> Vec<Poly> {
    let dist = |x: &[f64; 3]| dot3(n, x) - h;
    if polys.iter().flat_map(|p| p.pts.iter()).all(|x| dist(x) <= eps) {
        return polys;
    }
    let mut out = Vec::new();
    let mut cap = PointIndex::<3>::new(1e-9);
    for poly in polys {
        let mut kept = Vec::new();
        let k = poly.pts.len();
        for i in 0..k {
            let (a, b) = (poly.pts[i], poly.pts[(i + 1) % k]);
            let (da, db) = (dist(&a), dist(&b));
            if da <= eps {
                kept.push(a);
                if da.abs() <= eps {
                    cap.insert(a);
                }
            }
            if (da <= eps) != (db <= eps) && (da.abs() > eps && db.abs() > eps) {
                let t = da / (da - db);
                let x = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
                kept.push(x);
                cap.insert(x);
            }
        }
        if kept.len() >= 3 {
            out.push(Poly { pts: kept, tag: poly.tag });
        }
    }
    if cap.len() >= 3 {
        out.push(Poly { pts: order_around(cap.points().to_vec(), n), tag: Some(tag) });
    }
    out
}

/// Voronoi cell of `origin` against `neighbours` (perpendicular-bisector half-spaces).
pub fn voronoi_cell_3d(origin: [f64; 3], neighbours: &[[f64; 3]]) -> Result<VoronoiPolyhedron3d, String> {
    let mut rel: Vec<[f64; 3]> = neighbours.iter().map(|x| sub3(x, &origin)).filter(|d| norm3(d) > 1e-12).collect();
    if rel.is_empty() {
        return Err("no neighbours".into());
    }
    rel.sort_by(|a, b| norm3(a).total_cmp(&norm3(b)));
    let big = 2.0 * norm3(rel.last().expect("nonempty"));
    let eps = 1e-10 * big;
    let mut polys: Vec<Poly> = Vec::new();
    for axis in 0..3 {
        for s in [-1.0, 1.0] {
            let mut n = [0.0; 3];
            n[axis] = s;
            let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
            let pts = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(a, b)| {
                let mut x = [0.0; 3];
                x[axis] = s * big;
                x[u] = a * big;
                x[w] = b * big;
                x
            });
            polys.push(Poly { pts: order_around(pts.to_vec(), &n), tag: None });
        }
    }
    for (i, d) in rel.iter().enumerate() {
        polys = clip(polys, d, 0.5 * dot3(d, d), i, eps);
    }
    if polys.iter().any(|p| p.tag.is_none()) {
        return Err("neighbours do not enclose the origin (unbounded cell)".into());
    }
    let mut index = PointIndex::<3>::new(1e-7);
    let mut out = VoronoiPolyhedron3d::default();
    let mut edge_set: HashSet<[usize; 2]> = HashSet::new();
    for poly in polys {
        let normal = polygon_area_normal(&poly.pts);
        if norm3(&normal) < 1e-9 {
            continue;
        }
        let mut cyc: Vec<usize> = Vec::new();
        for x in &poly.pts {
            let i = index.insert(*x).0;
            if cyc.last() != Some(&i) && cyc.first() != Some(&i) {
                cyc.push(i);
            }
        }
        if cyc.len() < 3 {
            continue;
        }
        for k in 0..cyc.len() {
            let (a, b) = (cyc[k], cyc[(k + 1) % cyc.len()]);
            edge_set.insert([a.min(b), a.max(b)]);
        }
        out.faces.push(cyc);
        out.face_neighbours.push(rel[poly.tag.expect("tagged")]);
    }
    out.vertices = index.points().iter().map(|x| [x[0] + origin[0], x[1] + origin[1], x[2] + origin[2]]).collect();
    let mut edges: Vec<[usize; 2]> = edge_set.into_iter().collect();
    edges.sort_unstable();
    out.edges = edges;
    Ok(out)
}

pub fn lattice_basis(lattice: &str) -> Result<[[f64; 3]; 3], String> {
    match lattice {
        "SC" => Ok([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        "FCC" => Ok([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        "BCC" => Ok([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
        _ => Err(format!("unknown lattice {lattice:?} (SC, FCC, BCC)")),
    }
}

/// Lattice vectors `n·basis` with `|n_i| <= 2` (first and second neighbour shells and more).
pub fn lattice_neighbours(basis: &[[f64; 3]; 3]) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for a in -2i32..=2 {
        for b in -2i32..=2 {
            for c in -2i32..=2 {
                if (a, b, c) != (0, 0, 0) {
                    let (a, b, c) = (a as f64, b as f64, c as f64);
                    out.push([0, 1, 2].map(|k| a * basis[0][k] + b * basis[1][k] + c * basis[2][k]));
                }
            }
        }
    }
    out
}

/// Voronoi (Wigner–Seitz) cell of a named lattice: SC cube, FCC rhombic dodecahedron, BCC truncated octahedron.
pub fn lattice_voronoi_cell(lattice: &str) -> Result<VoronoiPolyhedron3d, String> {
    let basis = lattice_basis(lattice)?;
    voronoi_cell_3d([0.0; 3], &lattice_neighbours(&basis))
}

pub use super::dirichlet_quotient::{dirichlet_manifold, DirichletDomain3d, DirichletQuotient, FacePairing};
