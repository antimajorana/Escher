//! Arbitrary-metric `(n, m)` chiral nanotube roller (supporting both 60° and 120°
//! hexagonal conventions via the cell's metric tensor `G_ij = a_i . a_j`) and
//! finite/1D-periodic nanoribbon `cut_piece` builder.

use std::collections::{BTreeSet, HashMap};
use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};

use crate::bieberbach::snf::ext_gcd;
use crate::plane::cell::{mat2_det, mat2_inv, mat2_vec};
use crate::types::{Bond2D, Cell2D};

use super::niggli_supercell::build_supercell_2d;

/// Rolled 3D nanotube mesh constructed from a 2D periodic `Cell2D`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanotubeMesh {
    pub n: i32,
    pub m: i32,
    pub t_vec: [i32; 2],
    pub chiral_vec: [f64; 2],
    pub trans_vec: [f64; 2],
    pub circumference: f64,
    pub radius: f64,
    pub period_length: f64,
    pub chiral_angle_deg: f64,
    pub atoms_per_unit: usize,
    pub periods_z: usize,
    pub periodic_z: bool,
    pub positions_2d: Vec<[f64; 2]>,
    pub positions_3d: Vec<[f64; 3]>,
    pub bonds: Vec<(usize, usize, bool, usize)>,
    pub site_basis_idx: Vec<usize>,
    pub coordination_numbers: Vec<usize>,
}

/// Finite or 1D-periodic planar/3D nanoribbon cut along `chiral_dir`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanoribbonPatch {
    pub chiral_dir: [i32; 2],
    pub width_dir: [i32; 2],
    pub width_cells: usize,
    pub length_cells: usize,
    pub periodic_along: bool,
    pub period_vec: [f64; 2],
    pub positions_2d: Vec<[f64; 2]>,
    pub positions_3d: Vec<[f64; 3]>,
    pub bonds: Vec<(usize, usize, bool, usize)>,
    pub site_basis_idx: Vec<usize>,
    pub edge_sites: Vec<usize>,
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn norm2(a: [f64; 2]) -> f64 {
    a[0].hypot(a[1])
}

fn gcd_i32(mut a: i32, mut b: i32) -> i32 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

fn normalize_cell_basis(cell: &Cell2D) -> Result<Cell2D, String> {
    let a = cell.mat_a();
    let a_inv = mat2_inv(a)?;
    let mut floors = Vec::with_capacity(cell.basis.len());
    let mut new_basis = Vec::with_capacity(cell.basis.len());
    for &p in &cell.basis {
        let f = mat2_vec(a_inv, p);
        let fl = [(f[0] + 1e-9).floor() as i32, (f[1] + 1e-9).floor() as i32];
        let np = [
            p[0] - f64::from(fl[0]) * cell.a1[0] - f64::from(fl[1]) * cell.a2[0],
            p[1] - f64::from(fl[0]) * cell.a1[1] - f64::from(fl[1]) * cell.a2[1],
        ];
        floors.push(fl);
        new_basis.push(np);
    }
    let bonds = cell
        .bonds
        .iter()
        .map(|bd| Bond2D {
            i: bd.i,
            j: bd.j,
            n1: bd.n1 + floors[bd.j][0] - floors[bd.i][0],
            n2: bd.n2 + floors[bd.j][1] - floors[bd.i][1],
            kind: bd.kind,
        })
        .collect();
    Ok(Cell2D { a1: cell.a1, a2: cell.a2, basis: new_basis, bonds })
}

/// Find the shortest non-zero primitive integer lattice vector `T = t1*a1 + t2*a2`
/// orthogonal to `C_h = n*a1 + m*a2` under the cell's actual metric tensor `G_ij = a_i . a_j`,
/// oriented so `det([[n, t1], [m, t2]]) > 0`.
///
/// For graphene in the 60-degree basis this reproduces the published translation vector
/// `T = (-(2m + n), 2n + m) / d_R`, `d_R = gcd(2n + m, 2m + n)`, so `det = 2(n^2 + nm + m^2) / d_R`
/// hexagons and `4(n^2 + nm + m^2) / d_R` atoms per tube unit cell, diameter
/// `|C_h| / pi = a sqrt(n^2 + nm + m^2) / pi` and chiral angle `cos(theta) = (2n + m) / (2 sqrt(n^2 + nm + m^2))`
/// (R. Saito, G. Dresselhaus, M. S. Dresselhaus, *Physical Properties of Carbon Nanotubes*,
/// Imperial College Press 1998, Ch. 3; sign convention for `T` flipped to make `det > 0`).
///
/// # Errors
/// Returns an error for `(0, 0)`, a singular cell, or a refused (non-certified) period search.
pub fn find_orthogonal_period(cell: &Cell2D, n: i32, m: i32) -> Result<[i32; 2], String> {
    if n == 0 && m == 0 {
        return Err("chiral vector (n, m) must be non-zero".to_string());
    }
    if mat2_det(cell.mat_a()).abs() < 1e-12 {
        return Err("singular 2D cell lattice".to_string());
    }
    let g11 = dot2(cell.a1, cell.a1);
    let g12 = dot2(cell.a1, cell.a2);
    let g22 = dot2(cell.a2, cell.a2);
    let ch = [
        f64::from(n) * cell.a1[0] + f64::from(m) * cell.a2[0],
        f64::from(n) * cell.a1[1] + f64::from(m) * cell.a2[1],
    ];
    let ch_len = norm2(ch);
    let alpha = f64::from(n) * g11 + f64::from(m) * g12;
    let beta = f64::from(n) * g12 + f64::from(m) * g22;
    let ch2 = ch_len * ch_len;

    let mut best: Option<([i32; 2], f64)> = None;
    if alpha.abs() <= 1e-8 * ch2 {
        let t = [1, 0];
        let tv = cell.a1;
        best = Some((t, norm2(tv)));
    } else if beta.abs() <= 1e-8 * ch2 {
        let t = [0, 1];
        let tv = cell.a2;
        best = Some((t, norm2(tv)));
    } else {
        // Need alpha * t1 + beta * t2 == 0 => t2 / t1 = -alpha / beta.
        let r = -alpha / beta;
        for t1_abs in 1_i32..=128 {
            for sign in [1_i32, -1] {
                let t1 = sign * t1_abs;
                let t2 = (r * f64::from(t1)).round() as i32;
                if gcd_i32(t1, t2) != 1 {
                    continue;
                }
                let tv = [
                    f64::from(t1) * cell.a1[0] + f64::from(t2) * cell.a2[0],
                    f64::from(t1) * cell.a1[1] + f64::from(t2) * cell.a2[1],
                ];
                let t_len = norm2(tv);
                if t_len < 1e-12 {
                    continue;
                }
                let cos_err = (dot2(ch, tv) / (ch_len * t_len)).abs();
                if cos_err <= 1e-6 && best.is_none_or(|(_, bl)| t_len < bl - 1e-9) {
                    best = Some(([t1, t2], t_len));
                }
            }
        }
    }

    let Some((mut t, _)) = best else {
        return Err(format!(
            "REFUSED [COMPUTATIONAL EVIDENCE]: no primitive lattice period T = t1*a1 + t2*a2 with |t1| <= 128 is orthogonal to Ch = ({n}, {m}) within cos 1e-6 (metric a1.a2/|a1|^2 = {:.6}, |a2|^2/|a1|^2 = {:.6}); a 1D-periodic tube exists iff the ratio (Ch.a1)/(Ch.a2) is rational, which floating-point input cannot certify",
            g12 / g11.max(1e-30),
            g22 / g11.max(1e-30)
        ));
    };
    if n * t[1] - m * t[0] < 0 {
        t = [-t[0], -t[1]];
    }
    Ok(t)
}

/// Roll a 2D `Cell2D` into an `(n, m)` nanotube of `periods_z` open-ended translational units
/// using the cell's own `(a1, a2)` basis.
///
/// # Errors
/// See [`build_nanotube_opts`].
pub fn build_nanotube(cell: &Cell2D, n: i32, m: i32, periods_z: usize) -> Result<NanotubeMesh, String> {
    build_nanotube_opts(cell, n, m, periods_z, false, false)
}

/// Roll a 2D `Cell2D` into an `(n, m)` nanotube with explicit `periodic_z` and
/// optional `standard_hex_60` conversion (converting 120° hexagonal bases to 60° acute basis).
///
/// # Errors
/// Returns an error for `periods_z == 0`, an empty basis, or a refused period search.
pub fn build_nanotube_opts(
    cell: &Cell2D,
    n: i32,
    m: i32,
    periods_z: usize,
    periodic_z: bool,
    standard_hex_60: bool,
) -> Result<NanotubeMesh, String> {
    if periods_z == 0 {
        return Err("periods_z must be >= 1".to_string());
    }
    if cell.basis.is_empty() {
        return Err("cell has no basis sites".to_string());
    }
    let work_cell = if standard_hex_60 {
        let l1 = norm2(cell.a1);
        let l2 = norm2(cell.a2);
        let cos_g = dot2(cell.a1, cell.a2) / (l1 * l2).max(1e-30);
        if (l1 - l2).abs() <= 1e-5 * l1.max(l2) && (cos_g + 0.5).abs() <= 1e-5 {
            build_supercell_2d(cell, [[1, 1], [0, 1]])?
        } else {
            cell.clone()
        }
    } else {
        cell.clone()
    };
    let norm_cell = normalize_cell_basis(&work_cell)?;
    let t_vec = find_orthogonal_period(&norm_cell, n, m)?;
    let [t1, t2] = t_vec;
    let d_det = n * t2 - m * t1;
    debug_assert!(d_det > 0);

    let ch = [f64::from(n) * norm_cell.a1[0] + f64::from(m) * norm_cell.a2[0], f64::from(n) * norm_cell.a1[1] + f64::from(m) * norm_cell.a2[1]];
    let tr = [f64::from(t1) * norm_cell.a1[0] + f64::from(t2) * norm_cell.a2[0], f64::from(t1) * norm_cell.a1[1] + f64::from(t2) * norm_cell.a2[1]];
    let (circ, period_len) = (norm2(ch), norm2(tr));
    let (radius, c_hat, t_hat) = (circ / TAU, [ch[0] / circ, ch[1] / circ], [tr[0] / period_len, tr[1] / period_len]);
    let chiral_angle_deg = (dot2(ch, norm_cell.a1) / (circ * norm2(norm_cell.a1))).clamp(-1.0, 1.0).acos().to_degrees();

    let (xs, ys) = ([0, n, t1, n + t1], [0, m, t2, m + t2]);
    let (xmin, xmax) = (xs.iter().min().copied().unwrap() - 2, xs.iter().max().copied().unwrap() + 2);
    let (ymin, ymax) = (ys.iter().min().copied().unwrap() - 2, ys.iter().max().copied().unwrap() + 2);
    let coset_key = |k1: i32, k2: i32| ((t2 * k1 - t1 * k2).rem_euclid(d_det), (-m * k1 + n * k2).rem_euclid(d_det));
    let mut cell_cosets: HashMap<(i32, i32), [i32; 2]> = HashMap::with_capacity(d_det as usize);
    for k1 in xmin..=xmax {
        for k2 in ymin..=ymax {
            let ck = coset_key(k1, k2);
            cell_cosets.entry(ck).or_insert([(n * ck.0 + t1 * ck.1) / d_det, (m * ck.0 + t2 * ck.1) / d_det]);
        }
    }
    let mut sorted_cosets: Vec<_> = cell_cosets.into_iter().collect();
    sorted_cosets.sort_by_key(|&(ck, _)| ck);

    let cap = (d_det as usize) * norm_cell.basis.len();
    let (mut unit_sites, mut unit_map) = (Vec::with_capacity(cap), HashMap::with_capacity(cap));
    for (ck, [r1, r2]) in sorted_cosets {
        for (i, &bi) in norm_cell.basis.iter().enumerate() {
            let r_raw = [f64::from(r1) * norm_cell.a1[0] + f64::from(r2) * norm_cell.a2[0] + bi[0], f64::from(r1) * norm_cell.a1[1] + f64::from(r2) * norm_cell.a2[1] + bi[1]];
            let k_t = (dot2(r_raw, t_hat) / period_len + 1e-9).floor() as i32;
            unit_map.insert((i, ck.0, ck.1), unit_sites.len());
            unit_sites.push((i, [r1 - k_t * t1, r2 - k_t * t2], ck));
        }
    }

    let atoms_per_unit = unit_sites.len();
    let total_sites = atoms_per_unit * periods_z;
    let (mut positions_2d, mut positions_3d, mut site_basis_idx) = (Vec::with_capacity(total_sites), Vec::with_capacity(total_sites), Vec::with_capacity(total_sites));
    for p in 0..periods_z {
        for &(i, [k1, k2], _) in &unit_sites {
            let bi = norm_cell.basis[i];
            let r = [f64::from(k1) * norm_cell.a1[0] + f64::from(k2) * norm_cell.a2[0] + bi[0] + (p as f64) * tr[0], f64::from(k1) * norm_cell.a1[1] + f64::from(k2) * norm_cell.a2[1] + bi[1] + (p as f64) * tr[1]];
            let (s, z) = (dot2(r, c_hat), dot2(r, t_hat));
            let th = TAU * (s / circ);
            positions_2d.push([s, z]);
            positions_3d.push([radius * th.cos(), radius * th.sin(), z]);
            site_basis_idx.push(i);
        }
    }

    let (mut seen_bonds, mut bonds, mut coordination_numbers) = (BTreeSet::new(), Vec::new(), vec![0usize; total_sites]);
    for p in 0..periods_z {
        for (u_unit, &(i_b, [k1, k2], _)) in unit_sites.iter().enumerate() {
            let u_idx = p * atoms_per_unit + u_unit;
            for bd in &norm_cell.bonds {
                for (src, dst, dn1, dn2) in [(bd.i, bd.j, bd.n1, bd.n2), (bd.j, bd.i, -bd.n1, -bd.n2)] {
                    if src != i_b {
                        continue;
                    }
                    let (kt1, kt2) = (k1 + dn1, k2 + dn2);
                    let ck_t = coset_key(kt1, kt2);
                    let v_unit = unit_map[&(dst, ck_t.0, ck_t.1)];
                    let [kw1, kw2] = unit_sites[v_unit].1;
                    let (dk1, dk2) = (kt1 - kw1, kt2 - kw2);
                    let (dv, du) = ((-m * dk1 + n * dk2) / d_det, (t2 * dk1 - t1 * dk2) / d_det);
                    let tgt_p_raw = p as i32 + dv;
                    let tgt_p = if periodic_z { tgt_p_raw.rem_euclid(periods_z as i32) as usize } else if tgt_p_raw < 0 || tgt_p_raw >= periods_z as i32 { continue } else { tgt_p_raw as usize };
                    let v_idx = tgt_p * atoms_per_unit + v_unit;
                    let seam = du != 0 || (periodic_z && tgt_p_raw != tgt_p as i32);
                    let key = if (u_idx, du, dv) <= (v_idx, -du, -dv) { (u_idx, v_idx, du, dv, bd.kind) } else { (v_idx, u_idx, -du, -dv, bd.kind) };
                    if seen_bonds.insert(key) {
                        bonds.push((key.0, key.1, seam, bd.kind));
                        coordination_numbers[key.0] += 1;
                        coordination_numbers[key.1] += 1;
                    }
                }
            }
        }
    }

    Ok(NanotubeMesh {
        n, m, t_vec, chiral_vec: ch, trans_vec: tr, circumference: circ, radius, period_length: period_len,
        chiral_angle_deg, atoms_per_unit, periods_z, periodic_z, positions_2d, positions_3d, bonds, site_basis_idx, coordination_numbers,
    })
}

/// Cut a finite or 1D-periodic nanoribbon along `chiral_dir` of width `width_cells`
/// and length `length_cells`.
///
/// # Errors
/// Returns an error for a zero direction or zero sizes, or a singular cell.
pub fn build_nanoribbon(
    cell: &Cell2D,
    chiral_dir: [i32; 2],
    width_cells: usize,
    length_cells: usize,
    periodic_along: bool,
) -> Result<NanoribbonPatch, String> {
    if chiral_dir == [0, 0] || width_cells == 0 || length_cells == 0 {
        return Err("chiral_dir must be non-zero and width_cells/length_cells >= 1".to_string());
    }
    let norm_cell = normalize_cell_basis(cell)?;
    let [c1, c2] = chiral_dir;
    let g = gcd_i32(c1, c2);
    let (l1, l2) = (c1 / g, c2 / g);
    let (_, x, y) = ext_gcd(i128::from(l1), i128::from(l2));
    // det([[l1, w1], [l2, w2]]) = l1*w2 - l2*w1 = 1 -> w1 = -y, w2 = x.
    let (mut w1, mut w2) = (-y as i32, x as i32);
    let l_vec = [f64::from(l1) * norm_cell.a1[0] + f64::from(l2) * norm_cell.a2[0], f64::from(l1) * norm_cell.a1[1] + f64::from(l2) * norm_cell.a2[1]];
    let w_raw = [f64::from(w1) * norm_cell.a1[0] + f64::from(w2) * norm_cell.a2[0], f64::from(w1) * norm_cell.a1[1] + f64::from(w2) * norm_cell.a2[1]];
    let k_red = (dot2(l_vec, w_raw) / dot2(l_vec, l_vec)).round() as i32;
    w1 -= k_red * l1;
    w2 -= k_red * l2;

    let step_along = [f64::from(c1) * norm_cell.a1[0] + f64::from(c2) * norm_cell.a2[0], f64::from(c1) * norm_cell.a1[1] + f64::from(c2) * norm_cell.a2[1]];
    let rib_cell = build_supercell_2d(&norm_cell, [[c1, w1], [c2, w2]])?;
    let m = rib_cell.basis.len();
    let sid = |b: usize, u: usize, v: usize| (u * width_cells + v) * m + b;
    let n_sites = length_cells * width_cells * m;
    let mut positions_2d = vec![[0.0_f64; 2]; n_sites];
    let mut positions_3d = vec![[0.0_f64; 3]; n_sites];
    let mut site_basis_idx = vec![0usize; n_sites];
    for u in 0..length_cells {
        for v in 0..width_cells {
            let sx = (u as f64) * rib_cell.a1[0] + (v as f64) * rib_cell.a2[0];
            let sy = (u as f64) * rib_cell.a1[1] + (v as f64) * rib_cell.a2[1];
            for (b, &p) in rib_cell.basis.iter().enumerate() {
                let id = sid(b, u, v);
                positions_2d[id] = [p[0] + sx, p[1] + sy];
                positions_3d[id] = [p[0] + sx, p[1] + sy, 0.0];
                site_basis_idx[id] = b % norm_cell.basis.len();
            }
        }
    }

    let mut seen = BTreeSet::new();
    let mut bonds = Vec::new();
    let mut deg = vec![0usize; n_sites];
    for u in 0..length_cells {
        for v in 0..width_cells {
            for bd in &rib_cell.bonds {
                let id_u = sid(bd.i, u, v);
                let tv = v as i32 + bd.n2;
                if tv < 0 || tv >= width_cells as i32 {
                    continue;
                }
                let tu_raw = u as i32 + bd.n1;
                let (tu, seam) = if periodic_along {
                    (tu_raw.rem_euclid(length_cells as i32) as usize, tu_raw < 0 || tu_raw >= length_cells as i32)
                } else if tu_raw < 0 || tu_raw >= length_cells as i32 {
                    continue;
                } else {
                    (tu_raw as usize, false)
                };
                let id_v = sid(bd.j, tu, tv as usize);
                let du = tu_raw.div_euclid(length_cells as i32);
                let key = if (id_u, du, 0) <= (id_v, -du, 0) {
                    (id_u, id_v, du, bd.kind)
                } else {
                    (id_v, id_u, -du, bd.kind)
                };
                if seen.insert(key) {
                    bonds.push((key.0, key.1, seam, bd.kind));
                    deg[key.0] += 1;
                    deg[key.1] += 1;
                }
            }
        }
    }
    let max_d = deg.iter().copied().max().unwrap_or(0);
    let edge_sites = (0..n_sites).filter(|&i| deg[i] < max_d).collect();
    Ok(NanoribbonPatch {
        chiral_dir,
        width_dir: [w1, w2],
        width_cells,
        length_cells,
        periodic_along,
        period_vec: [step_along[0] * length_cells as f64, step_along[1] * length_cells as f64],
        positions_2d,
        positions_3d,
        bonds,
        site_basis_idx,
        edge_sites,
    })
}
