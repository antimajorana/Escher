//! Closed 3-manifolds from one polyhedron with paired faces (Poincaré polyhedron construction).
//!
//! Known answers used by the tests: Seifert-Weber space (C. Weber, H. Seifert, "Die beiden
//! Dodekaederräume", Math. Z. 37 (1933) 237-253): opposite faces of the regular {5,3,5} dodecahedron
//! glued with a 3π/5 twist, 1 vertex, 6 edges, 6 faces, 1 cell, `H_1 = (Z/5)^3`; the same paper's
//! π/5 twist gives the Poincaré homology sphere (5 vertices, 10 edges, `H_1 = 0`).
//!
//! The quotient has one 3-cell; faces, edges and vertices are counted as classes under the
//! pairings. `H_1` is the abelianised Poincaré presentation: one generator per face pair, one
//! relator per edge cycle, reduced by Smith normal form.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use super::honeycombs_h3::build_hyperbolic_honeycomb;
use super::voronoi3d_dirichlet::lattice_voronoi_cell;
use super::{cross3, dot3, norm3, sub3};
use crate::bieberbach::{smith_normal_form, IntMat};

#[derive(Clone, Debug, Serialize)]
pub struct FacePairing {
    pub face: usize,
    pub partner: usize,
    /// `(vertex of face, image vertex of partner)`.
    pub vertex_map: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DirichletDomain3d {
    pub name: String,
    /// `"H3"`, `"S3"` or `"E3"`.
    pub geometry: String,
    pub note: String,
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
    pub pairings: Vec<FacePairing>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DirichletQuotient {
    pub cells: usize,
    pub faces: usize,
    pub edges: usize,
    pub vertices: usize,
    pub euler: i64,
    pub edge_class_sizes: Vec<usize>,
    pub vertex_class_sizes: Vec<usize>,
    /// Torsion invariant factors `> 1` of `H_1`.
    pub h1_torsion: Vec<i64>,
    pub h1_free_rank: usize,
    pub h1_label: String,
    /// Evidence stamp of the counts and `H_1`.
    pub evidence: String,
}

/// Per face: `(generator index, exponent ±1, partner face, vertex map onto the partner)`.
type FaceInfo = (usize, i128, usize, HashMap<usize, usize>);

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn class_sizes(parent: &mut [usize]) -> Vec<usize> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for i in 0..parent.len() {
        *counts.entry(find(parent, i)).or_default() += 1;
    }
    let mut s: Vec<usize> = counts.into_values().collect();
    s.sort_unstable();
    s
}

impl DirichletDomain3d {
    /// Quotient complex counts and `H_1`; errors if the pairings are not a valid involution on faces.
    pub fn quotient(&self) -> Result<DirichletQuotient, String> {
        let nf = self.faces.len();
        // face -> (generator, sign, vertex map)
        let mut info: Vec<Option<FaceInfo>> = vec![None; nf];
        for (j, pr) in self.pairings.iter().enumerate() {
            let fwd: HashMap<usize, usize> = pr.vertex_map.iter().copied().collect();
            let bwd: HashMap<usize, usize> = pr.vertex_map.iter().map(|&(a, b)| (b, a)).collect();
            if info[pr.face].is_some() || info[pr.partner].is_some() || pr.face == pr.partner {
                return Err(format!("face {} or {} paired twice", pr.face, pr.partner));
            }
            info[pr.face] = Some((j, 1, pr.partner, fwd));
            info[pr.partner] = Some((j, -1, pr.face, bwd));
        }
        let info: Vec<_> = info.into_iter().map(|x| x.ok_or("unpaired face")).collect::<Result<_, _>>()?;
        let mut edge_id: HashMap<[usize; 2], usize> = HashMap::new();
        let mut edge_faces: Vec<Vec<usize>> = Vec::new();
        for (f, cyc) in self.faces.iter().enumerate() {
            for k in 0..cyc.len() {
                let (a, b) = (cyc[k], cyc[(k + 1) % cyc.len()]);
                let e = [a.min(b), a.max(b)];
                let id = *edge_id.entry(e).or_insert_with(|| {
                    edge_faces.push(Vec::new());
                    edge_faces.len() - 1
                });
                edge_faces[id].push(f);
            }
        }
        let ne = edge_faces.len();
        let mut vpar: Vec<usize> = (0..self.vertices.len()).collect();
        let mut epar: Vec<usize> = (0..ne).collect();
        let map_edge = |e: [usize; 2], m: &HashMap<usize, usize>| -> Result<usize, String> {
            let (a, b) = (m[&e[0]], m[&e[1]]);
            edge_id.get(&[a.min(b), a.max(b)]).copied().ok_or_else(|| "pairing does not map edges to edges".to_string())
        };
        let edges: Vec<[usize; 2]> = {
            let mut v = vec![[0, 0]; ne];
            for (e, &i) in &edge_id {
                v[i] = *e;
            }
            v
        };
        for pr in &self.pairings {
            for &(a, b) in &pr.vertex_map {
                let (ra, rb) = (find(&mut vpar, a), find(&mut vpar, b));
                vpar[ra] = rb;
            }
            let fwd = &info[pr.face].3;
            let cyc = &self.faces[pr.face];
            for k in 0..cyc.len() {
                let (a, b) = (cyc[k], cyc[(k + 1) % cyc.len()]);
                let e0 = edge_id[&[a.min(b), a.max(b)]];
                let e1 = map_edge(edges[e0], fwd)?;
                let (r0, r1) = (find(&mut epar, e0), find(&mut epar, e1));
                epar[r0] = r1;
            }
        }
        // Edge-cycle relators, one per edge class.
        let mut seen_class: HashSet<usize> = HashSet::new();
        let mut rows: Vec<Vec<i128>> = Vec::new();
        for e0 in 0..ne {
            let root = find(&mut epar, e0);
            if !seen_class.insert(root) {
                continue;
            }
            let f0 = edge_faces[e0][0];
            let mut row = vec![0i128; self.pairings.len()];
            let (mut e, mut f) = (e0, f0);
            for step in 0.. {
                if step > 10_000 {
                    return Err("edge cycle did not close".into());
                }
                let (j, s, partner, m) = &info[f];
                row[*j] += s;
                let e1 = map_edge(edges[e], m)?;
                let others: Vec<usize> = edge_faces[e1].iter().copied().filter(|g| g != partner).collect();
                if others.len() != 1 {
                    return Err("edge is not on exactly two faces".into());
                }
                (e, f) = (e1, others[0]);
                if (e, f) == (e0, f0) {
                    break;
                }
            }
            rows.push(row);
        }
        let ng = self.pairings.len();
        let snf = smith_normal_form(&IntMat::from_rows(&rows), false);
        let nonzero: Vec<i128> = snf.diag.iter().copied().filter(|d| *d != 0).map(|d| d.abs()).collect();
        let h1_free_rank = ng - nonzero.len();
        let h1_torsion: Vec<i64> = nonzero.iter().filter(|d| **d > 1).map(|d| *d as i64).collect();
        let mut parts: Vec<String> = Vec::new();
        if h1_free_rank > 0 {
            parts.push(if h1_free_rank == 1 { "Z".into() } else { format!("Z^{h1_free_rank}") });
        }
        let mut tors: Vec<(i64, usize)> = Vec::new();
        for &d in &h1_torsion {
            match tors.last_mut() {
                Some((x, c)) if *x == d => *c += 1,
                _ => tors.push((d, 1)),
            }
        }
        for (d, c) in tors {
            parts.push(if c == 1 { format!("Z_{d}") } else { format!("Z_{d}^{c}") });
        }
        let edge_class_sizes = class_sizes(&mut epar);
        let vertex_class_sizes = class_sizes(&mut vpar);
        let (v, e, f) = (vertex_class_sizes.len(), edge_class_sizes.len(), ng);
        Ok(DirichletQuotient {
            cells: 1,
            faces: f,
            edges: e,
            vertices: v,
            euler: v as i64 - e as i64 + f as i64 - 1,
            edge_class_sizes,
            vertex_class_sizes,
            h1_torsion,
            h1_free_rank,
            h1_label: if parts.is_empty() { "0".into() } else { parts.join(" + ") },
            evidence: "[PROVED: exact cell counts of the quotient complex; H_1 from the Poincare presentation (one generator per face pair, one relator per edge cycle) reduced by integer Smith normal form]".into(),
        })
    }
}

fn centroid(vs: &[[f64; 3]], cyc: &[usize]) -> [f64; 3] {
    let m = cyc.len() as f64;
    cyc.iter().fold([0.0; 3], |a, &i| [a[0] + vs[i][0] / m, a[1] + vs[i][1] / m, a[2] + vs[i][2] / m])
}

fn rotate(v: &[f64; 3], k: &[f64; 3], phi: f64) -> [f64; 3] {
    let (c, s) = (phi.cos(), phi.sin());
    let kx = cross3(k, v);
    let kd = dot3(k, v);
    [0, 1, 2].map(|i| v[i] * c + kx[i] * s + k[i] * kd * (1.0 - c))
}

fn nearest(vs: &[[f64; 3]], cyc: &[usize], x: &[f64; 3]) -> (usize, f64) {
    cyc.iter().map(|&i| (i, norm3(&sub3(&vs[i], x)))).min_by(|a, b| a.1.total_cmp(&b.1)).expect("nonempty")
}

/// Pair each face of a centrally symmetric polyhedron with the opposite face: translate across,
/// then turn by `phi` about the axis (right-handed about the direction of travel).
pub fn twisted_opposite_pairings(vs: &[[f64; 3]], faces: &[Vec<usize>], phi: f64) -> Result<Vec<FacePairing>, String> {
    let cents: Vec<[f64; 3]> = faces.iter().map(|c| centroid(vs, c)).collect();
    let edge = norm3(&sub3(&vs[faces[0][0]], &vs[faces[0][1]]));
    let mut out = Vec::new();
    for (f, c) in cents.iter().enumerate() {
        let anti = c.map(|x| -x);
        let partner = (0..faces.len()).min_by(|&a, &b| norm3(&sub3(&cents[a], &anti)).total_cmp(&norm3(&sub3(&cents[b], &anti)))).expect("faces");
        if partner < f {
            continue;
        }
        let cp = cents[partner];
        let d = sub3(&cp, c);
        let axis = d.map(|x| x / norm3(&d));
        let mut vmap = Vec::new();
        for &v in &faces[f] {
            let rel = sub3(&vs[v], c);
            let t = rotate(&rel, &axis, phi);
            let target = [cp[0] + t[0], cp[1] + t[1], cp[2] + t[2]];
            let (w, dist) = nearest(vs, &faces[partner], &target);
            if dist > 0.2 * edge {
                return Err(format!("twist {phi} does not map face {f} onto face {partner} vertex to vertex"));
            }
            vmap.push((v, w));
        }
        out.push(FacePairing { face: f, partner, vertex_map: vmap });
    }
    Ok(out)
}

/// Named closed 3-manifolds: `seifert_weber` (H³), `poincare_homology_sphere` (S³), and the
/// Euclidean 3-tori `cubic_3_torus`, `fcc_3_torus`, `bcc_3_torus` (Voronoi cells, translations).
pub fn dirichlet_manifold(name: &str) -> Result<DirichletDomain3d, String> {
    use std::f64::consts::PI;
    let lattice = match name {
        "cubic_3_torus" => Some("SC"),
        "fcc_3_torus" => Some("FCC"),
        "bcc_3_torus" => Some("BCC"),
        _ => None,
    };
    if let Some(lat) = lattice {
        let cell = lattice_voronoi_cell(lat)?;
        let mut pairings = Vec::new();
        for (f, b) in cell.face_neighbours.iter().enumerate() {
            let partner = cell.face_neighbours.iter().position(|c| norm3(&[c[0] + b[0], c[1] + b[1], c[2] + b[2]]) < 1e-9).ok_or("no opposite face")?;
            if partner < f {
                continue;
            }
            let vertex_map = cell.faces[f].iter().map(|&v| (v, nearest(&cell.vertices, &cell.faces[partner], &sub3(&cell.vertices[v], b)).0)).collect();
            pairings.push(FacePairing { face: f, partner, vertex_map });
        }
        return Ok(DirichletDomain3d {
            name: name.into(),
            geometry: "E3".into(),
            note: format!("Voronoi cell of the {lat} lattice, opposite faces identified by lattice translations"),
            vertices: cell.vertices,
            faces: cell.faces,
            pairings,
        });
    }
    let (phi, geometry, note) = match name {
        "seifert_weber" => (3.0 * PI / 5.0, "H3", "regular {5,3,5} dodecahedron (dihedral angle 2π/5) in the Poincaré ball; opposite faces glued with a 3π/5 twist"),
        "poincare_homology_sphere" => (PI / 5.0, "S3", "regular spherical dodecahedron of the 120-cell {5,3,3} (dihedral angle 2π/3); opposite faces glued with a π/5 twist; combinatorial model drawn with the same centred dodecahedron"),
        _ => return Err(format!("unknown manifold {name:?} (seifert_weber, poincare_homology_sphere, cubic_3_torus, fcc_3_torus, bcc_3_torus)")),
    };
    let cell = build_hyperbolic_honeycomb(5, 3, 5, 0, 1)?;
    let vs = cell.points.clone();
    let pairings = twisted_opposite_pairings(&vs, &cell.faces, phi)?;
    Ok(DirichletDomain3d { name: name.into(), geometry: geometry.into(), note: note.into(), vertices: vs, faces: cell.faces, pairings })
}
