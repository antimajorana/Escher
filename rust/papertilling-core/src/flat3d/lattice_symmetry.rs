//! Holohedry (lattice point symmetry group) of a crystal's full translation lattice, used for
//! `[PROVED]` platycosm obstructions.
//!
//! Theorem used: if a Bieberbach group `G` acts on a crystal by symmetries, the point group of `G`
//! is a subgroup of the crystal's point group, which is a subgroup of the holohedry of the crystal's
//! full (primitive) translation lattice (International Tables for Crystallography Vol. A,
//! "lattice point symmetry / holohedry"). The holohedry is enumerated exactly (up to the metric
//! tolerance `1e-5` of the floating-point input): every lattice automorphism maps each Niggli basis
//! vector `b_j` to a lattice vector of the same length, and such vectors `v = sum_i c_i b_i` obey the
//! Cauchy-Schwarz bound `|c_i| <= |v| * |column_i(B^-1)|`.

use std::collections::{BTreeSet, HashSet};

use crate::bieberbach::{IntMat, hnf_compress_rows};

use super::niggli_supercell::niggli_reduce_3d;
use super::{Cell3D, dot3, frac_to_cart_3d, inv3_f64};

/// Rotation content of a lattice holohedry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Holohedry {
    pub order: usize,
    pub proper_orders: BTreeSet<usize>,
    pub has_222: bool,
}

fn det3_i64(m: [[i64; 3]; 3]) -> i64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn mul3_i64(a: [[i64; 3]; 3], b: [[i64; 3]; 3]) -> [[i64; 3]; 3] {
    [0, 1, 2].map(|r| [0, 1, 2].map(|c| a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c]))
}

/// Fractional translations `t` (mod the cell lattice, `t != 0`) mapping the crystal onto itself.
fn pure_translations(cell: &Cell3D, frac: &[[f64; 3]], ignore_kinds: bool) -> Vec<[f64; 3]> {
    let kind = |b: usize| if ignore_kinds { 0 } else { cell.site_kinds.get(b).copied().unwrap_or(0) };
    let mut darts: HashSet<(usize, usize, [i32; 3])> = HashSet::new();
    for bd in &cell.bonds {
        darts.insert((bd.i, bd.j, [bd.n1, bd.n2, bd.n3]));
        darts.insert((bd.j, bd.i, [-bd.n1, -bd.n2, -bd.n3]));
    }
    let near_int = |x: f64| (x - x.round()).abs() < 1e-4;
    let mut out: Vec<[f64; 3]> = Vec::new();
    for (j, fj) in frac.iter().enumerate().skip(1) {
        if kind(j) != kind(0) {
            continue;
        }
        let t = [0, 1, 2].map(|k| (fj[k] - frac[0][k]).rem_euclid(1.0));
        if t.iter().all(|&x| near_int(x)) || out.iter().any(|o| (0..3).all(|k| near_int(o[k] - t[k]))) {
            continue;
        }
        let mut perm = Vec::with_capacity(frac.len());
        let mut shift = Vec::with_capacity(frac.len());
        for (s, fs) in frac.iter().enumerate() {
            let hit = frac.iter().enumerate().find_map(|(s2, f2)| {
                let d = [0, 1, 2].map(|k| fs[k] + t[k] - f2[k]);
                (kind(s2) == kind(s) && d.iter().all(|&x| near_int(x))).then_some((s2, d.map(|x| x.round() as i32)))
            });
            let Some((s2, sh)) = hit else { break };
            perm.push(s2);
            shift.push(sh);
        }
        if perm.len() != frac.len() {
            continue;
        }
        let bonds_ok = cell.bonds.iter().all(|bd| {
            let n = [bd.n1, bd.n2, bd.n3];
            let n2 = [0, 1, 2].map(|k| n[k] + shift[bd.j][k] - shift[bd.i][k]);
            darts.contains(&(perm[bd.i], perm[bd.j], n2))
        });
        if bonds_ok {
            out.push(t);
        }
    }
    out
}

/// Row basis of the crystal's full translation lattice (the cell lattice refined by any pure
/// fractional translations), or `None` if the refinement cannot be certified (denominator > 48).
pub(crate) fn full_translation_lattice(cell: &Cell3D, frac: &[[f64; 3]], ignore_kinds: bool) -> Option<[[f64; 3]; 3]> {
    let ts = pure_translations(cell, frac, ignore_kinds);
    if ts.is_empty() {
        return Some(cell.lattice);
    }
    let den = (2_i64..=48).find(|&d| ts.iter().all(|t| t.iter().all(|&x| (x * d as f64 - (x * d as f64).round()).abs() < 1e-3)))?;
    let d128 = i128::from(den);
    let mut gens: Vec<Vec<i128>> = vec![vec![d128, 0, 0], vec![0, d128, 0], vec![0, 0, d128]];
    gens.extend(ts.iter().map(|t| t.iter().map(|&x| (x * den as f64).round() as i128).collect()));
    let h = hnf_compress_rows(&IntMat::from_rows(&gens));
    if h.rows != 3 {
        return None;
    }
    let rows = [0, 1, 2].map(|r| [0, 1, 2].map(|c| h.get(r, c) as f64 / den as f64));
    Some(rows.map(|f| frac_to_cart_3d(&cell.lattice, f)))
}

/// Enumerate the holohedry of the lattice with row basis `lat`; `None` if not certifiable.
pub(crate) fn lattice_holohedry(lat: &[[f64; 3]; 3]) -> Option<Holohedry> {
    let b = niggli_reduce_3d(*lat).ok()?.reduced;
    let g = [0, 1, 2].map(|i| [0, 1, 2].map(|j| dot3(b[i], b[j])));
    let g_max = g[0][0].max(g[1][1]).max(g[2][2]);
    let tol = 1e-5 * g_max;
    let b_inv = inv3_f64(b).ok()?;
    let r_max = g_max.sqrt();
    let mut bound = [0_i64; 3];
    for (i, bd) in bound.iter_mut().enumerate() {
        let col = (b_inv[0][i] * b_inv[0][i] + b_inv[1][i] * b_inv[1][i] + b_inv[2][i] * b_inv[2][i]).sqrt();
        *bd = (r_max * col + 1e-6).floor() as i64;
        if *bd > 6 {
            return None;
        }
    }
    let quad = |x: [i64; 3], y: [i64; 3]| -> f64 {
        let gy = g.map(|row| row[0] * y[0] as f64 + row[1] * y[1] as f64 + row[2] * y[2] as f64);
        x[0] as f64 * gy[0] + x[1] as f64 * gy[1] + x[2] as f64 * gy[2]
    };
    let mut cands: [Vec<[i64; 3]>; 3] = Default::default();
    for c0 in -bound[0]..=bound[0] {
        for c1 in -bound[1]..=bound[1] {
            for c2 in -bound[2]..=bound[2] {
                let c = [c0, c1, c2];
                let n2 = quad(c, c);
                for (j, cj) in cands.iter_mut().enumerate() {
                    if (n2 - g[j][j]).abs() <= tol {
                        cj.push(c);
                    }
                }
            }
        }
    }
    let mut proper = Vec::new();
    let mut order = 0usize;
    for &m0 in &cands[0] {
        if (quad(m0, m0) - g[0][0]).abs() > tol {
            continue;
        }
        for &m1 in &cands[1] {
            if (quad(m0, m1) - g[0][1]).abs() > tol {
                continue;
            }
            for &m2 in &cands[2] {
                if (quad(m0, m2) - g[0][2]).abs() > tol || (quad(m1, m2) - g[1][2]).abs() > tol {
                    continue;
                }
                let m = [m0, m1, m2];
                match det3_i64(m) {
                    1 => proper.push(m),
                    -1 => {}
                    _ => return None,
                }
                order += 1;
            }
        }
    }
    if order == 0 || order > 48 {
        return None;
    }
    let mut proper_orders = BTreeSet::new();
    let mut twofolds = Vec::new();
    for &m in &proper {
        // Proper rotation by angle theta has trace 1 + 2 cos(theta) (crystallographic restriction).
        let ord = match m[0][0] + m[1][1] + m[2][2] {
            3 => 1,
            -1 => 2,
            0 => 3,
            1 => 4,
            2 => 6,
            _ => return None,
        };
        if ord == 2 {
            twofolds.push(m);
        }
        proper_orders.insert(ord);
    }
    let has_222 = twofolds
        .iter()
        .enumerate()
        .any(|(i, &a)| twofolds[i + 1..].iter().any(|&b| mul3_i64(a, b) == mul3_i64(b, a)));
    Some(Holohedry { order, proper_orders, has_222 })
}

/// `[PROVED]` holohedry obstruction for platycosm `code`, or `None` if the holohedry allows it.
pub(crate) fn holohedry_obstruction(code: &str, h: &Holohedry) -> Option<String> {
    let (ok, need) = match code {
        "c2" | "+a1" | "-a1" => (h.proper_orders.contains(&2), "a 2-fold axis (point group 2, or m = -1 * 2)"),
        "c3" => (h.proper_orders.contains(&3), "a 3-fold axis (point group 3)"),
        "c4" => (h.proper_orders.contains(&4), "a 4-fold axis (point group 4)"),
        "c6" => (h.proper_orders.contains(&6), "a 6-fold axis (point group 6)"),
        "c22" | "+a2" | "-a2" => (h.has_222, "two perpendicular 2-fold axes (point group 222, or mm2 = -1 * 222)"),
        _ => (true, ""),
    };
    (!ok).then(|| {
        format!(
            "IMPOSSIBLE [PROVED]: {code} needs {need}, but the holohedry of the crystal's full translation lattice (order {}, proper rotation orders {:?}) has none; the point group of any symmetry subgroup lies in that holohedry",
            h.order, h.proper_orders
        )
    })
}
