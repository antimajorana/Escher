//! Pieces + pairings engine: glue a periodic planar lattice across flat convex polygons.

use std::collections::{HashMap, HashSet};

pub use super::{RefusalKind, SiteLabel, SurfaceError};
pub use super::pieces_glue::glue;
use super::{
    add2, apply_to_site, cross2, dot2, mat2_inv, mat2_mul, mat2_vec, norm2, rot2, sub2,
    symmetry_ops_on_cell,
};
use crate::types::{Cell2D, FracOp2D, Isometry2, SurfaceCheckReport, SurfaceMesh};

pub(super) const TOL: f64 = 1e-7;

pub type DartKey = (usize, usize, i32, i32);
pub type DartFaceMap = HashMap<DartKey, (usize, (i32, i32))>;

pub type PairingSpec = ((usize, usize), (usize, usize), bool);
pub type WalkResult = (usize, [f64; 2], Vec<SiteLabel>, [[f64; 2]; 2]);
pub type CentreLookup<'a> = &'a dyn Fn([f64; 2]) -> Option<(String, String)>;

fn polygon_centroid(pts: &[[f64; 2]]) -> [f64; 2] {
    let n = pts.len();
    let (mut area2, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (p, q) = (pts[i], pts[(i + 1) % n]);
        let cr = p[0] * q[1] - q[0] * p[1];
        area2 += cr;
        cx += (p[0] + q[0]) * cr;
        cy += (p[1] + q[1]) * cr;
    }
    if (0.5 * area2).abs() < 1e-12 {
        let s = pts.iter().fold([0.0, 0.0], |acc, &p| add2(acc, p));
        [s[0] / n as f64, s[1] / n as f64]
    } else {
        [cx / (3.0 * area2), cy / (3.0 * area2)]
    }
}

/// Planar lattice data prepared for pieces-and-pairings group tests and walks.
#[derive(Debug, Clone)]
pub struct LatticeData {
    pub group: String,
    pub a: [[f64; 2]; 2],
    pub a_inv: [[f64; 2]; 2],
    pub basis: Vec<[f64; 2]>,
    pub frac_basis: Vec<[f64; 2]>,
    pub bonds: Vec<DartKey>,
    pub darts: HashSet<DartKey>,
    pub plane_degree: Vec<usize>,
    pub faces: Vec<(Vec<SiteLabel>, [f64; 2])>,
    pub dart_face: DartFaceMap,
    pub ops: Vec<([[i32; 2]; 2], [f64; 2])>,
    pub lmax: f64,
    pub fmax: f64,
    pub bmax: f64,
}

impl LatticeData {
    pub fn new(group: &str, cell: &Cell2D) -> Result<Self, String> {
        let a = [[cell.a1[0], cell.a2[0]], [cell.a1[1], cell.a2[1]]];
        let a_inv = mat2_inv(a)?;
        let m = cell.basis.len();
        let frac_basis: Vec<[f64; 2]> = cell.basis.iter().map(|&p| mat2_vec(a_inv, p)).collect();
        let bonds: Vec<_> = cell.bonds.iter().map(|b| (b.i, b.j, b.n1, b.n2)).collect();
        let mut dart_list = Vec::with_capacity(bonds.len() * 2);
        let mut darts = HashSet::with_capacity(bonds.len() * 2);
        for &(i, j, n1, n2) in &bonds {
            for d in [(i, j, n1, n2), (j, i, -n1, -n2)] {
                dart_list.push(d);
                darts.insert(d);
            }
        }
        let dart_idx: HashMap<_, _> = dart_list.iter().copied().enumerate().map(|(k, d)| (d, k)).collect();
        let rev: Vec<usize> = dart_list.iter().map(|&(i, j, n1, n2)| dart_idx[&(j, i, -n1, -n2)]).collect();
        let mut by_tail: Vec<Vec<(f64, usize)>> = vec![Vec::new(); m];
        for (k, &(i, j, n1, n2)) in dart_list.iter().enumerate() {
            let v = sub2(add2(cell.basis[j], mat2_vec(a, [n1 as f64, n2 as f64])), cell.basis[i]);
            by_tail[i].push((v[1].atan2(v[0]), k));
        }
        let (mut prv, mut plane_degree) = (vec![0usize; dart_list.len()], vec![0usize; m]);
        for (i, lst) in by_tail.iter_mut().enumerate() {
            lst.sort_by(|x, y| x.0.total_cmp(&y.0));
            plane_degree[i] = lst.len();
            for idx in 0..lst.len() {
                prv[lst[idx].1] = lst[(idx + lst.len() - 1) % lst.len()].1;
            }
        }
        let (mut visited, mut faces, mut dart_face) =
            (vec![false; dart_list.len()], Vec::new(), HashMap::with_capacity(dart_list.len()));
        for d0 in 0..dart_list.len() {
            if visited[d0] {
                continue;
            }
            let (mut cyc, mut d) = (Vec::new(), d0);
            while !visited[d] {
                visited[d] = true;
                cyc.push(dart_list[d]);
                d = prv[rev[d]];
            }
            let (f_id, mut n, mut labs) = (faces.len(), [0i32, 0i32], Vec::with_capacity(cyc.len()));
            for &(i, j, n1, n2) in &cyc {
                labs.push((i, n[0], n[1]));
                dart_face.insert((i, j, n1, n2), (f_id, (n[0], n[1])));
                n = [n[0] + n1, n[1] + n2];
            }
            let pts: Vec<[f64; 2]> = labs
                .iter()
                .map(|&(i, s1, s2)| add2(cell.basis[i], mat2_vec(a, [s1 as f64, s2 as f64])))
                .collect();
            faces.push((labs, polygon_centroid(&pts)));
        }
        let ops = symmetry_ops_on_cell(a, a_inv, &frac_basis, &bonds, &darts, false);
        let lmax = bonds.iter().fold(0.0_f64, |acc, &(i, j, n1, n2)| {
            acc.max(norm2(sub2(add2(cell.basis[j], mat2_vec(a, [n1 as f64, n2 as f64])), cell.basis[i])))
        });
        let fmax = faces.iter().fold(0.0_f64, |acc, (labs, cen)| {
            labs.iter().fold(acc, |acc2, &(i, s1, s2)| {
                acc2.max(norm2(sub2(add2(cell.basis[i], mat2_vec(a, [s1 as f64, s2 as f64])), *cen)))
            })
        });
        let bmax = cell.basis.iter().fold(0.0_f64, |acc, &p| acc.max(norm2(p)));
        Ok(Self {
            group: group.to_string(),
            a,
            a_inv,
            basis: cell.basis.clone(),
            frac_basis,
            bonds,
            darts,
            plane_degree,
            faces,
            dart_face,
            ops,
            lmax,
            fmax,
            bmax,
        })
    }

    #[inline]
    pub fn pos(&self, lab: SiteLabel) -> [f64; 2] {
        add2(self.basis[lab.0], mat2_vec(self.a, [lab.1 as f64, lab.2 as f64]))
    }

    pub fn frac(&self, g: &Isometry2) -> ([[f64; 2]; 2], [f64; 2]) {
        (mat2_mul(mat2_mul(self.a_inv, g.m), self.a), mat2_vec(self.a_inv, g.t))
    }

    pub fn test_isometry(&self, g: &Isometry2) -> (bool, String, bool) {
        let (mf, tf) = self.frac(g);
        let mi = [
            [mf[0][0].round() as i32, mf[0][1].round() as i32],
            [mf[1][0].round() as i32, mf[1][1].round() as i32],
        ];
        if (mf[0][0] - mi[0][0] as f64).abs() > 1e-6
            || (mf[0][1] - mi[0][1] as f64).abs() > 1e-6
            || (mf[1][0] - mi[1][0] as f64).abs() > 1e-6
            || (mf[1][1] - mi[1][1] as f64).abs() > 1e-6
        {
            return (false, "its linear part is not a symmetry of the translation lattice".into(), false);
        }
        let mut imgs = Vec::with_capacity(self.frac_basis.len());
        for i in 0..self.frac_basis.len() {
            let Some(im) = apply_to_site(mi, tf, &self.frac_basis, i) else {
                return (false, format!("site {i} of the cell lands on no site"), false);
            };
            imgs.push(im);
        }
        for &(i, j, n1, n2) in &self.bonds {
            let ((pi, ci), (pj, cj)) = (imgs[i], imgs[j]);
            let d0 = mi[0][0] * n1 + mi[0][1] * n2 + cj[0] - ci[0];
            let d1 = mi[1][0] * n1 + mi[1][1] * n2 + cj[1] - ci[1];
            if !self.darts.contains(&(pi, pj, d0, d1)) {
                return (false, format!("bond ({i}, {j}, {n1}, {n2}) lands on no bond"), false);
            }
        }
        let listed = self.ops.iter().any(|&(m, t)| {
            m == mi
                && (tf[0] - t[0] - (tf[0] - t[0]).round()).abs() <= 1e-6
                && (tf[1] - t[1] - (tf[1] - t[1]).round()).abs() <= 1e-6
        });
        (true, String::new(), listed)
    }

    pub fn apply_frac(&self, gf: &FracOp2D, lab: SiteLabel) -> Result<SiteLabel, String> {
        let (j, c) = apply_to_site(gf.mf, gf.t, &self.frac_basis, lab.0)
            .ok_or_else(|| "site orbit is not closed under gluing".to_string())?;
        Ok((
            j,
            gf.mf[0][0] * lab.1 + gf.mf[0][1] * lab.2 + c[0],
            gf.mf[1][0] * lab.1 + gf.mf[1][1] * lab.2 + c[1],
        ))
    }
}

pub fn describe_isometry(g: &Isometry2) -> String {
    if g.det() > 0.0 {
        let ang = g.m[1][0].atan2(g.m[0][0]).to_degrees();
        if ang.abs() < 1e-6 {
            return if norm2(g.t) < 1e-9 {
                "the identity".into()
            } else {
                format!("a translation by [{:.4}, {:.4}]", g.t[0], g.t[1])
            };
        }
        let imr = [[1.0 - g.m[0][0], -g.m[0][1]], [-g.m[1][0], 1.0 - g.m[1][1]]];
        let c = mat2_inv(imr).map(|inv| mat2_vec(inv, g.t)).unwrap_or([0.0, 0.0]);
        return format!("a rotation by {ang:.1} deg about [{:.4}, {:.4}]", c[0], c[1]);
    }
    let (v0, v1) = ([1.0 + g.m[0][0], g.m[1][0]], [g.m[0][1], 1.0 + g.m[1][1]]);
    let raw = if dot2(v0, v0) >= dot2(v1, v1) { v0 } else { v1 };
    let len = norm2(raw).max(1e-12);
    let ax = [raw[0] / len, raw[1] / len];
    let s = dot2(g.t, ax);
    let p = [0.5 * (g.t[0] - s * ax[0]), 0.5 * (g.t[1] - s * ax[1])];
    let kind = if s.abs() < 1e-9 { "a reflection".to_string() } else { format!("a glide by {s:.4}") };
    format!("{kind} in the line through [{:.4}, {:.4}] along [{:.4}, {:.4}]", p[0], p[1], ax[0], ax[1])
}

/// Convex counter-clockwise polygonal chart piece in the Euclidean plane.
#[derive(Debug, Clone)]
pub struct Piece2D {
    pub corners: Vec<[f64; 2]>,
    normals: Vec<[f64; 2]>,
    offsets: Vec<f64>,
}

impl Piece2D {
    pub fn new(corners: Vec<[f64; 2]>) -> Result<Self, String> {
        let n = corners.len();
        if n < 3 {
            return Err("piece must have at least 3 corners".into());
        }
        let edges: Vec<[f64; 2]> = (0..n).map(|s| sub2(corners[(s + 1) % n], corners[s])).collect();
        if (0..n).any(|s| cross2(edges[s], edges[(s + 1) % n]) <= 1e-12) {
            return Err("piece is not convex and counter-clockwise".into());
        }
        let normals: Vec<[f64; 2]> = edges
            .iter()
            .map(|&e| {
                let l = norm2(e);
                [e[1] / l, -e[0] / l]
            })
            .collect();
        let offsets: Vec<f64> = (0..n).map(|s| dot2(normals[s], corners[s])).collect();
        Ok(Self { corners, normals, offsets })
    }

    pub fn side(&self, s: usize) -> ([f64; 2], [f64; 2]) {
        (self.corners[s], self.corners[(s + 1) % self.corners.len()])
    }
    pub fn inside(&self, x: [f64; 2], tol: f64) -> bool {
        (0..self.normals.len()).all(|s| dot2(self.normals[s], x) - self.offsets[s] <= tol)
    }
    pub fn on_side(&self, x: [f64; 2], s: usize, tol: f64) -> bool {
        (dot2(self.normals[s], x) - self.offsets[s]).abs() <= tol && self.inside(x, tol)
    }
    pub fn corner_at(&self, x: [f64; 2], tol: f64) -> Option<usize> {
        self.corners
            .iter()
            .enumerate()
            .map(|(j, &p)| (j, norm2(sub2(p, x))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .and_then(|(j, d)| (d < tol).then_some(j))
    }
}

/// Unfolded polygonal net with edge-to-edge Euclidean gluing isometries.
#[derive(Debug, Clone)]
pub struct Net2D {
    pub pieces: Vec<Piece2D>,
    pub pairs: Vec<PairingSpec>,
    pub glue_map: HashMap<(usize, usize), (usize, usize, Isometry2)>,
}

impl Net2D {
    pub fn new(raw_pieces: &[Vec<[f64; 2]>], pairs: &[PairingSpec]) -> Result<Self, String> {
        let pieces: Vec<Piece2D> = raw_pieces
            .iter()
            .enumerate()
            .map(|(k, p)| Piece2D::new(p.clone()).map_err(|e| format!("piece {k}: {e}")))
            .collect::<Result<_, _>>()?;
        let mut net = Self { pieces, pairs: pairs.to_vec(), glue_map: HashMap::with_capacity(pairs.len() * 2) };
        for &(a, b, flip) in pairs {
            for (src, dst) in [(a, b), (b, a)] {
                if net.glue_map.contains_key(&src) {
                    return Err(format!("side {src:?} is glued twice"));
                }
                let iso = net.side_isometry(src, dst, flip)?;
                net.glue_map.insert(src, (dst.0, dst.1, iso));
            }
        }
        for (k, p) in net.pieces.iter().enumerate() {
            for s in 0..p.corners.len() {
                if !net.glue_map.contains_key(&(k, s)) {
                    return Err(format!("side {s} of piece {k} is not glued"));
                }
            }
        }
        Ok(net)
    }

    fn side_isometry(&self, a: (usize, usize), b: (usize, usize), flip: bool) -> Result<Isometry2, String> {
        let (p0, p1) = self.pieces[a.0].side(a.1);
        let (mut q0, mut q1) = self.pieces[b.0].side(b.1);
        if flip {
            std::mem::swap(&mut q0, &mut q1);
        }
        if (norm2(sub2(p1, p0)) - norm2(sub2(q1, q0))).abs() > 1e-6 {
            return Err(format!("sides {a:?} and {b:?} have different lengths"));
        }
        let (psi, phi) = ((p1[1] - p0[1]).atan2(p1[0] - p0[0]), (q1[1] - q0[1]).atan2(q1[0] - q0[0]));
        let mean = |cs: &[[f64; 2]]| {
            let s = cs.iter().fold([0.0, 0.0], |acc, &p| add2(acc, p));
            [s[0] / cs.len() as f64, s[1] / cs.len() as f64]
        };
        let (inner_a, inner_b, nb) =
            (mean(&self.pieces[a.0].corners), mean(&self.pieces[b.0].corners), self.pieces[b.0].normals[b.1]);
        let (sp, cp) = (phi + psi).sin_cos();
        for m in [rot2(phi - psi), [[cp, sp], [sp, -cp]]] {
            let iso = Isometry2 { m, t: sub2(q0, mat2_vec(m, p0)) };
            if dot2(sub2(iso.apply(inner_a), q0), nb) > 0.0 && dot2(sub2(inner_b, q0), nb) < 0.0 {
                return Ok(iso);
            }
        }
        Err(format!("no gluing of sides {a:?} and {b:?}"))
    }

    pub fn walk(
        &self,
        gf_map: &HashMap<(usize, usize), FracOp2D>,
        mut k: usize,
        mut x: [f64; 2],
        mut target: [f64; 2],
        mut labs: Vec<SiteLabel>,
        lat: &LatticeData,
    ) -> Result<WalkResult, SurfaceError> {
        let mut lin = [[1.0, 0.0], [0.0, 1.0]];
        for _ in 0..500 {
            if self.pieces[k].inside(target, TOL) {
                return Ok((k, target, labs, lin));
            }
            let (d, piece) = (sub2(target, x), &self.pieces[k]);
            let ts: Vec<f64> = (0..piece.normals.len())
                .map(|s| {
                    let nd = dot2(piece.normals[s], d);
                    if nd > 1e-12 { ((piece.offsets[s] - dot2(piece.normals[s], x)) / nd).max(0.0) } else { f64::INFINITY }
                })
                .collect();
            let t = ts.iter().copied().fold(f64::INFINITY, f64::min);
            let exits: Vec<usize> = (0..ts.len()).filter(|&s| ts[s] <= t + 1e-9).collect();
            let y = [x[0] + t * d[0], x[1] + t * d[1]];
            if exits.len() != 1 || piece.corner_at(y, 1e-6).is_some() {
                return Err(SurfaceError::not_found(
                    "a bond or a face is cut by a cone point at this size",
                    format!("straight path in piece {k} hits a corner"),
                    true,
                )
                .with_hint("try another size"));
            }
            let (l, _, iso) = self.glue_map[&(k, exits[0])];
            let gf = &gf_map[&(k, exits[0])];
            (x, target) = (iso.apply(y), iso.apply(target));
            labs = labs.into_iter().map(|a| lat.apply_frac(gf, a)).collect::<Result<_, _>>().map_err(SurfaceError::invalid)?;
            (lin, k) = (mat2_mul(iso.m, lin), l);
        }
        Err(SurfaceError::invalid("walk did not end"))
    }
}

#[derive(Debug, Clone)]
pub struct PairingTestRow {
    pub pair: ((usize, usize), (usize, usize)),
    pub isometry: String,
    pub ok: bool,
    pub reason: String,
    pub listed: bool,
    pub orientation_reversing: bool,
}

pub fn pairing_test(
    group: &str,
    cell: &Cell2D,
    pieces: &[Vec<[f64; 2]>],
    pairs: &[PairingSpec],
) -> Result<Vec<PairingTestRow>, String> {
    let lat = LatticeData::new(group, cell)?;
    let net = Net2D::new(pieces, pairs)?;
    Ok(pairing_test_with(&lat, &net))
}

pub(crate) fn pairing_test_with(lat: &LatticeData, net: &Net2D) -> Vec<PairingTestRow> {
    net.pairs
        .iter()
        .map(|&(a, b, _)| {
            let g = &net.glue_map[&a].2;
            let (ok, reason, listed) = lat.test_isometry(g);
            PairingTestRow {
                pair: (a, b),
                isometry: describe_isometry(g),
                ok,
                reason,
                listed,
                orientation_reversing: g.det() < 0.0,
            }
        })
        .collect()
}

/// Glued polyhedral cone surface with full topological validation metadata.
#[derive(Debug, Clone)]
pub struct ConeSurface {
    pub mesh: SurfaceMesh,
    pub face_sides: Vec<Vec<usize>>,
    pub net_chi: i32,
    pub site_copies: Vec<Vec<(usize, SiteLabel)>>,
}

impl ConeSurface {
    pub fn checks(&self) -> SurfaceCheckReport {
        let mut rep = check_surface_mesh(&self.mesh);
        rep.faces_at_least_3 = self.face_sides.iter().all(|f| f.len() >= 3);
        let frank: f64 = self.mesh.cones.iter().map(|c| c.angle_deg).sum();
        let gauss_bonnet = (frank - 360.0 * self.mesh.chi as f64).abs() < 1e-6;
        let net_euler = self.net_chi == self.mesh.chi;
        rep.ok = rep.no_loops
            && rep.no_double_bonds
            && rep.faces_at_least_3
            && rep.plane_coordination
            && rep.euler
            && gauss_bonnet
            && net_euler;
        rep
    }
}

pub fn check_surface_mesh(mesh: &SurfaceMesh) -> SurfaceCheckReport {
    let (v, e, f) = (mesh.sites.len(), mesh.bonds.len(), mesh.faces.len());
    let (mut deg, mut no_loops, mut pair_set) = (vec![0usize; v], true, HashSet::with_capacity(e));
    for &(a, b, _) in &mesh.bonds {
        if a == b {
            no_loops = false;
        }
        deg[a] += 1;
        deg[b] += 1;
        pair_set.insert(if a < b { (a, b) } else { (b, a) });
    }
    let no_double_bonds = pair_set.len() == e;
    let mut exempt: HashSet<usize> = mesh.edge_sites.iter().copied().collect();
    exempt.extend(mesh.cones.iter().filter_map(|c| c.site));
    let plane_coordination = (0..v)
        .filter(|k| !exempt.contains(k))
        .all(|k| deg[k] == mesh.plane_degree[mesh.sites[k].0]);
    let euler = (v as i32 - e as i32 + f as i32) == mesh.chi;
    let faces_at_least_3 = mesh.faces.iter().all(|fc| fc.len() >= 3);
    let ok = no_loops && no_double_bonds && plane_coordination && euler && faces_at_least_3;
    SurfaceCheckReport { ok, no_loops, no_double_bonds, plane_coordination, euler, faces_at_least_3 }
}
