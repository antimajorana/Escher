//! Compact `H^2` orientable Fuchsian triangle-group quotients (`TGCell`)
//! via explicit coset / permutation action of `Delta^+(2, q, p)`.
//!
//! A regular map of type `{p, q}` is a finite group `G = <x, y>` with `x^2 = y^q = (xy)^p = 1`: darts
//! are elements of `G`, `x` reverses a dart, `y` turns it about its vertex, so `E = |G|/2`,
//! `V = |G|/q`, `F = |G|/p` and `chi = V - E + F` (Coxeter & Moser, *Generators and Relations for
//! Discrete Groups*, 4th ed. 1980, ch. 8). Published checks: Bolza surface, genus 2, `|G^+| = 48`
//! (`{3,8}` / `{8,3}`: 6 octagons, 16 vertices) and the one-octagon `{8,8}`; Klein quartic `{7,3}`,
//! genus 3, `PSL(2,7)` of order 168: 24 heptagons, 84 edges, 56 vertices (Klein 1878; Levy (ed.),
//! *The Eightfold Way*, MSRI Publ. 35, 1999); Bring's surface genus 4 as the regular map `{5,5}` with
//! group `A_5` (12 pentagons; Weber, "Kepler's small stellated dodecahedron as a Riemann surface",
//! Pacific J. Math. 220 (2005)). Counts are exact group computations ([PROVED]); the hyperbolic
//! placement of the fundamental polygon is f64.

use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

use crate::hyperbolic::inflation::{BaseFace, IDENTITY, Mat3, hyp_dist, hyperboloid_to_poincare, mat_mul, mat_vec};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SidePairing {
    pub side_a: usize,
    pub side_b: usize,
    pub edge_index: usize,
    pub lorentz_matrix: [[f64; 3]; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TgCellQuotient {
    pub name: String,
    pub group_label: String,
    pub group_order: usize,
    pub p: usize,
    pub q: usize,
    pub genus: usize,
    pub chi: i64,
    pub n_vertices: usize,
    pub n_edges: usize,
    pub n_faces: usize,
    pub faces: Vec<Vec<usize>>,
    pub edges: Vec<(usize, usize)>,
    pub side_pairings: Vec<SidePairing>,
    pub fundamental_polygon_poincare: Vec<[f64; 2]>,
    pub placed_faces_poincare: Vec<Vec<[f64; 2]>>,
    pub provenance: String,
}

pub const TGCELL_NAMES: [&str; 7] = [
    "bolza_8_8",
    "bolza_8_3",
    "klein_7_3",
    "bring_5_5",
    "genus2_6_4",
    "genus2_10_5",
    "genus2_6_6",
];

pub fn list_tgcell_quotients() -> &'static [&'static str] {
    &TGCELL_NAMES
}

pub fn tgcell_quotient(name: &str) -> Result<TgCellQuotient, String> {
    let key: String = name
        .trim()
        .to_lowercase()
        .replace(['{', '}', ' ', '-'], "_");
    match key.as_str() {
        "bolza_8_8" | "bolza_88" | "bolza" | "8_8" => build_bolza_8_8(),
        "bolza_8_3" | "bolza_83" | "8_3" => build_bolza_8_3(),
        "klein_7_3" | "klein_73" | "klein" | "7_3" => build_klein_7_3(),
        "bring_5_5" | "bring_55" | "bring" | "5_5" => build_bring_5_5(),
        "genus2_6_4" | "6_4" => build_genus2_6_4(),
        "genus2_10_5" | "10_5" => build_genus2_10_5(),
        "genus2_6_6" | "6_6" => build_genus2_6_6(),
        _ => Err(format!("unknown TGCell quotient {name:?}; choose from {TGCELL_NAMES:?}")),
    }
}

pub fn verify_tgcell_invariants(qcell: &TgCellQuotient) -> Result<(), String> {
    let expected_chi = 2 - 2 * (qcell.genus as i64);
    let actual_chi = qcell.n_vertices as i64 - qcell.n_edges as i64 + qcell.n_faces as i64;
    if qcell.chi != expected_chi || actual_chi != expected_chi {
        return Err(format!(
            "{}: Euler invariant mismatch: expected 2-2g={expected_chi}, got V-E+F={actual_chi}",
            qcell.name
        ));
    }
    if qcell.faces.len() != qcell.n_faces || qcell.edges.len() != qcell.n_edges {
        return Err(format!("{}: face/edge count mismatch", qcell.name));
    }
    if !qcell.faces.iter().all(|f| f.len() == qcell.p) {
        return Err(format!("{}: non-uniform face length (expected {})", qcell.name, qcell.p));
    }
    let mut deg = vec![0usize; qcell.n_vertices];
    for &(u, v) in &qcell.edges {
        if u >= qcell.n_vertices || v >= qcell.n_vertices {
            return Err(format!("{}: edge vertex out of bounds", qcell.name));
        }
        deg[u] += 1;
        deg[v] += 1;
    }
    if !deg.iter().all(|&d| d == qcell.q) {
        return Err(format!("{}: non-uniform vertex degree (expected {}): {deg:?}", qcell.name, qcell.q));
    }
    Ok(())
}

fn rot3(theta: f64) -> Mat3 {
    let (c, s) = (theta.cos(), theta.sin());
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

fn refl_line3(theta: f64) -> Mat3 {
    let (c2, s2) = ((2.0 * theta).cos(), (2.0 * theta).sin());
    [[c2, s2, 0.0], [s2, -c2, 0.0], [0.0, 0.0, 1.0]]
}

fn lorentz_inv(m: &Mat3) -> Mat3 {
    // M^{-1} = J * M^T * J for M in O(2, 1), J = diag(1, 1, -1).
    [
        [m[0][0], m[1][0], -m[2][0]],
        [m[0][1], m[1][1], -m[2][1]],
        [-m[0][2], -m[1][2], m[2][2]],
    ]
}

#[allow(clippy::too_many_arguments)]
fn build_regular_quotient<T, Mul, Inv>(
    name: &str,
    group_label: &str,
    p: usize,
    q: usize,
    id_el: T,
    x: T,
    y: T,
    mul: Mul,
    inv: Inv,
) -> Result<TgCellQuotient, String>
where
    T: Copy + Eq + Hash,
    Mul: Fn(T, T) -> T,
    Inv: Fn(T) -> T,
{
    let mut els = vec![id_el];
    let mut idx_map: HashMap<T, usize> = HashMap::from([(id_el, 0)]);
    let mut queue = VecDeque::from([id_el]);
    let yi = inv(y);
    while let Some(g) = queue.pop_front() {
        for g_op in [x, y, yi] {
            let h = mul(g, g_op);
            if let std::collections::hash_map::Entry::Vacant(e) = idx_map.entry(h) {
                e.insert(els.len());
                els.push(h);
                queue.push_back(h);
            }
        }
    }
    let n = els.len();
    let rev: Vec<usize> = els.iter().map(|&e| idx_map[&mul(e, x)]).collect();
    let nxt: Vec<usize> = els.iter().map(|&e| idx_map[&mul(e, yi)]).collect();
    let prv: Vec<usize> = els.iter().map(|&e| idx_map[&mul(e, y)]).collect();
    let nxt_face: Vec<usize> = (0..n).map(|d| prv[rev[d]]).collect();

    let mut face_of = vec![usize::MAX; n];
    let mut faces_darts = Vec::new();
    let mut side_of_dart = vec![(0usize, 0usize); n];
    for d0 in 0..n {
        if face_of[d0] != usize::MAX {
            continue;
        }
        let fi = faces_darts.len();
        let mut cyc = Vec::new();
        let mut d = d0;
        while face_of[d] == usize::MAX {
            face_of[d] = fi;
            side_of_dart[d] = (fi, cyc.len());
            cyc.push(d);
            d = nxt_face[d];
        }
        faces_darts.push(cyc);
    }

    let mut vert_of = vec![usize::MAX; n];
    let mut n_vertices = 0usize;
    for d0 in 0..n {
        if vert_of[d0] != usize::MAX {
            continue;
        }
        let vi = n_vertices;
        n_vertices += 1;
        let mut d = d0;
        while vert_of[d] == usize::MAX {
            vert_of[d] = vi;
            d = nxt[d];
        }
    }

    let mut edge_of = vec![usize::MAX; n];
    let mut edges = Vec::new();
    for d in 0..n {
        if edge_of[d] != usize::MAX {
            continue;
        }
        let r = rev[d];
        let ei = edges.len();
        edge_of[d] = ei;
        edge_of[r] = ei;
        edges.push((vert_of[d], vert_of[r]));
    }
    let faces: Vec<Vec<usize>> = faces_darts
        .iter()
        .map(|cyc| cyc.iter().map(|&d| vert_of[d]).collect())
        .collect();
    let n_edges = edges.len();
    let n_faces = faces.len();
    let chi = n_vertices as i64 - n_edges as i64 + n_faces as i64;
    if chi > 0 || (2 - chi) % 2 != 0 {
        return Err(format!("{name}: group action gives chi = {chi}, not a closed orientable hyperbolic surface"));
    }
    let genus = ((2 - chi) / 2) as usize;

    // Spanning tree of faces + exact Lorentz lift to H^2 and boundary polygon.
    let base = BaseFace::new(p, q)?;
    let side_step = |k1: usize, k2: usize| -> Mat3 {
        let rot = rot3(2.0 * PI * (k1 as isize - k2 as isize) as f64 / (p as f64));
        let r_perp = refl_line3(2.0 * PI * (k1 as f64 + 0.5) / (p as f64));
        mat_mul(&base.reflections[k1], &mat_mul(&r_perp, &rot))
    };

    let mut placed: Vec<Option<Mat3>> = vec![None; n_faces];
    placed[0] = Some(IDENTITY);
    let mut f_q = VecDeque::from([0usize]);
    let mut tree_darts = HashSet::new();
    while let Some(f) = f_q.pop_front() {
        let gf = placed[f].unwrap();
        for (k, &d) in faces_darts[f].iter().enumerate() {
            let rd = rev[d];
            let (f2, k2) = side_of_dart[rd];
            if placed[f2].is_none() {
                tree_darts.insert(d);
                tree_darts.insert(rd);
                placed[f2] = Some(mat_mul(&gf, &side_step(k, k2)));
                f_q.push_back(f2);
            }
        }
    }
    let placed: Vec<Mat3> = placed.into_iter().map(Option::unwrap).collect();
    let placed_faces_poincare: Vec<Vec<[f64; 2]>> = placed
        .iter()
        .map(|m| base.vertices.iter().map(|&v| hyperboloid_to_poincare(mat_vec(m, v))).collect())
        .collect();

    // Walk outer boundary cycle of the tree of faces.
    let d_start = (0..n).find(|d| !tree_darts.contains(d)).ok_or("no boundary dart")?;
    let mut b_darts = Vec::new();
    let mut b_pos = vec![usize::MAX; n];
    let mut d = d_start;
    loop {
        b_pos[d] = b_darts.len();
        b_darts.push(d);
        let mut cur = nxt_face[d];
        while tree_darts.contains(&cur) {
            cur = nxt_face[rev[cur]];
        }
        d = cur;
        if d == d_start {
            break;
        }
    }
    let fundamental_polygon_poincare: Vec<[f64; 2]> = b_darts
        .iter()
        .map(|&bd| {
            let (f, k) = side_of_dart[bd];
            hyperboloid_to_poincare(mat_vec(&placed[f], base.vertices[k]))
        })
        .collect();
    let mut side_pairings = Vec::new();
    for (sa, &da) in b_darts.iter().enumerate() {
        let db = rev[da];
        let sb = b_pos[db];
        if sa < sb {
            let (fa, ka) = side_of_dart[da];
            let (fb, kb) = side_of_dart[db];
            let g_target = mat_mul(&placed[fa], &side_step(ka, kb));
            let lorentz_matrix = mat_mul(&g_target, &lorentz_inv(&placed[fb]));
            let va = mat_vec(&lorentz_matrix, mat_vec(&placed[fb], base.vertices[kb]));
            let vb = mat_vec(&placed[fa], base.vertices[(ka + 1) % p]);
            if hyp_dist(va, vb) > 1e-7 {
                return Err(format!("{name}: side pairing Lorentz matrix failed endpoint check"));
            }
            side_pairings.push(SidePairing {
                side_a: sa,
                side_b: sb,
                edge_index: edge_of[da],
                lorentz_matrix,
            });
        }
    }

    Ok(TgCellQuotient {
        name: name.to_string(),
        group_label: group_label.to_string(),
        group_order: n,
        p,
        q,
        genus,
        chi,
        n_vertices,
        n_edges,
        n_faces,
        faces,
        edges,
        side_pairings,
        fundamental_polygon_poincare,
        placed_faces_poincare,
        provenance: "[PROVED]".to_string(),
    })
}

fn build_bolza_8_8() -> Result<TgCellQuotient, String> {
    build_regular_quotient("bolza_8_8", "Z_8", 8, 8, 0u8, 4u8, 1u8, |a, b| (a + b) % 8, |a| (8 - a) % 8)
}

fn build_genus2_10_5() -> Result<TgCellQuotient, String> {
    build_regular_quotient("genus2_10_5", "Z_10", 10, 5, 0u8, 5u8, 2u8, |a, b| (a + b) % 10, |a| (10 - a) % 10)
}

fn build_genus2_6_6() -> Result<TgCellQuotient, String> {
    build_regular_quotient(
        "genus2_6_6",
        "Z_2 x Z_6",
        6,
        6,
        (0u8, 0u8),
        (1u8, 0u8),
        (0u8, 1u8),
        |a, b| ((a.0 + b.0) % 2, (a.1 + b.1) % 6),
        |a| (a.0, (6 - a.1) % 6),
    )
}

fn build_bolza_8_3() -> Result<TgCellQuotient, String> {
    let mul = |a: [u8; 4], b: [u8; 4]| {
        [
            (a[0] * b[0] + a[1] * b[2]) % 3,
            (a[0] * b[1] + a[1] * b[3]) % 3,
            (a[2] * b[0] + a[3] * b[2]) % 3,
            (a[2] * b[1] + a[3] * b[3]) % 3,
        ]
    };
    let inv = |a: [u8; 4]| {
        let det = (a[0] * a[3] + 9 - a[1] * a[2]) % 3;
        [(det * a[3]) % 3, (det * (3 - a[1])) % 3, (det * (3 - a[2])) % 3, (det * a[0]) % 3]
    };
    build_regular_quotient("bolza_8_3", "GL(2,3)", 8, 3, [1, 0, 0, 1], [1, 0, 1, 2], [1, 1, 0, 1], mul, inv)
}

fn build_klein_7_3() -> Result<TgCellQuotient, String> {
    let norm = |a: [u8; 4]| -> [u8; 4] {
        for &c in &a {
            if c != 0 {
                return if c <= 3 { a } else { [(7 - a[0]) % 7, (7 - a[1]) % 7, (7 - a[2]) % 7, (7 - a[3]) % 7] };
            }
        }
        a
    };
    let mul = |a: [u8; 4], b: [u8; 4]| {
        norm([
            (a[0] * b[0] + a[1] * b[2]) % 7,
            (a[0] * b[1] + a[1] * b[3]) % 7,
            (a[2] * b[0] + a[3] * b[2]) % 7,
            (a[2] * b[1] + a[3] * b[3]) % 7,
        ])
    };
    let inv = |a: [u8; 4]| norm([a[3] % 7, (7 - a[1]) % 7, (7 - a[2]) % 7, a[0] % 7]);
    build_regular_quotient(
        "klein_7_3",
        "PSL(2,7)",
        7,
        3,
        [1, 0, 0, 1],
        norm([0, 6, 1, 0]),
        norm([0, 6, 1, 1]),
        mul,
        inv,
    )
}

fn build_bring_5_5() -> Result<TgCellQuotient, String> {
    let mul = |a: [u8; 5], b: [u8; 5]| [a[b[0] as usize], a[b[1] as usize], a[b[2] as usize], a[b[3] as usize], a[b[4] as usize]];
    let inv = |a: [u8; 5]| {
        let mut r = [0u8; 5];
        for (i, &v) in a.iter().enumerate() {
            r[v as usize] = i as u8;
        }
        r
    };
    build_regular_quotient("bring_5_5", "A_5", 5, 5, [0, 1, 2, 3, 4], [0, 3, 4, 1, 2], [1, 2, 3, 4, 0], mul, inv)
}

fn build_genus2_6_4() -> Result<TgCellQuotient, String> {
    let mul = |e1: (u8, u8, u8), e2: (u8, u8, u8)| -> (u8, u8, u8) {
        let (a1, i1, j1) = e1;
        let (a2, i2, j2) = e2;
        let s_d4 = if j1 == 0 { i2 } else { (4 - i2) % 4 };
        let ni = (i1 + s_d4) % 4;
        let nj = j1 ^ j2;
        let phi = if i1.is_multiple_of(2) { 1 } else { 2 };
        let na = (a1 + phi * a2) % 3;
        (na, ni, nj)
    };
    let inv = |e: (u8, u8, u8)| -> (u8, u8, u8) {
        let (a, i, j) = e;
        let nj = j;
        let ni = if j == 0 { (4 - i) % 4 } else { i };
        let phi_inv = if ni.is_multiple_of(2) { 1 } else { 2 };
        let na = (3 - ((phi_inv * a) % 3)) % 3;
        (na, ni, nj)
    };
    build_regular_quotient("genus2_6_4", "Z_3 |> D_4", 6, 4, (0, 0, 0), (0, 1, 1), (1, 1, 0), mul, inv)
}
