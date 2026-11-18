//! Cartesian symmetry elements (mirror lines, glide lines, rotation centres) of a wallpaper group
//! acting on a drawn `Cell2D`, clipped to the app's periodic repeat window.
//!
//! Work is done in primitive fractional coordinates `p` of the drawn cell (`x = A p`, `A = [a1 a2]`).
//! A conventional ITA op `f -> mf f + t` (conventional fractional `f = U (p - s)`, `s = A^{-1} origin`)
//! becomes `p -> Mp p + tp` with `Mp = U^{-1} mf U` and `tp = U^{-1} t + (I - Mp) s`.

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::plane::cell::{mat2_det, mat2_inv, mat2_mul, mat2_vec};
use crate::plane::groups::{canonical_symbol, group_ops, label_point, prim_u, prim_u_inv};
use crate::types::{Cell2D, Frame2D, Isometry2};

/// Kind of a reflection-type symmetry line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymLineKind {
    Mirror,
    Glide,
}

/// A mirror or glide line clipped to the window; `glide` is the shortest Cartesian glide translation
/// along the line (zero for mirrors).
#[derive(Debug, Clone, PartialEq)]
pub struct SymLine2D {
    pub kind: SymLineKind,
    pub a: [f64; 2],
    pub b: [f64; 2],
    pub glide: [f64; 2],
}

/// A rotation centre with the highest rotation order fixing it and its ITA Wyckoff name (e.g. `"2c"`).
#[derive(Debug, Clone, PartialEq)]
pub struct RotationCentre2D {
    pub pos: [f64; 2],
    pub order: u8,
    pub wyckoff: String,
}

/// Symmetry elements of a group inside the periodic repeat window.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetryElements2D {
    pub lines: Vec<SymLine2D>,
    pub centres: Vec<RotationCentre2D>,
}

impl SymLine2D {
    /// The reflection (or glide reflection) about this line as a Cartesian isometry.
    pub fn isometry(&self) -> Isometry2 {
        let d = [self.b[0] - self.a[0], self.b[1] - self.a[1]];
        let len = d[0].hypot(d[1]).max(1e-300);
        let (c, s) = (d[0] / len, d[1] / len);
        let m = [[2.0 * c * c - 1.0, 2.0 * c * s], [2.0 * c * s, 2.0 * s * s - 1.0]];
        let ma = mat2_vec(m, self.a);
        Isometry2 {
            m,
            t: [self.a[0] - ma[0] + self.glide[0], self.a[1] - ma[1] + self.glide[1]],
        }
    }
}

impl RotationCentre2D {
    /// Counter-clockwise rotation by `360 / order` degrees about this centre.
    pub fn isometry(&self) -> Isometry2 {
        let th = 2.0 * PI / f64::from(self.order.max(1));
        let m = [[th.cos(), -th.sin()], [th.sin(), th.cos()]];
        let mc = mat2_vec(m, self.pos);
        Isometry2 { m, t: [self.pos[0] - mc[0], self.pos[1] - mc[1]] }
    }
}

/// One group op in primitive fractional coordinates of the drawn cell.
#[derive(Clone, Copy)]
struct PrimOp {
    m: [[i32; 2]; 2],
    t: [f64; 2],
}

struct Setup {
    a: [[f64; 2]; 2],
    a_inv: [[f64; 2]; 2],
    u: [[f64; 2]; 2],
    s: [f64; 2],
    ops: Vec<PrimOp>,
}

fn to_f(m: [[i32; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [f64::from(m[0][0]), f64::from(m[0][1])],
        [f64::from(m[1][0]), f64::from(m[1][1])],
    ]
}

fn finite2(v: [f64; 2]) -> bool {
    v[0].is_finite() && v[1].is_finite()
}

/// `reduce_origin` replaces `s = A^{-1} origin` by `s - round(s)`: a primitive lattice shift of the origin
/// leaves the group unchanged and keeps every `tp` of order one (no overflow for far origins).
fn setup(group: &str, cell: &Cell2D, frame: Frame2D, reduce_origin: bool) -> Result<Setup, String> {
    let sym = canonical_symbol(group)?;
    let fin = finite2(cell.a1)
        && finite2(cell.a2)
        && finite2(frame.origin)
        && finite2(frame.q[0])
        && finite2(frame.q[1]);
    if !fin {
        return Err("non-finite cell or frame".to_string());
    }
    let a = cell.mat_a();
    let scale2 = (cell.a1[0].hypot(cell.a1[1]) * cell.a2[0].hypot(cell.a2[1])).max(1e-300);
    if mat2_det(a).abs() <= 1e-12 * scale2 {
        return Err("singular cell basis".to_string());
    }
    let a_inv = mat2_inv(a)?;
    let u = prim_u(sym)?;
    let ui = prim_u_inv(sym)?;
    let s0 = mat2_vec(a_inv, frame.origin);
    if !finite2(s0) {
        return Err("frame origin too far for this cell".to_string());
    }
    let s = if reduce_origin { [s0[0] - s0[0].round(), s0[1] - s0[1].round()] } else { s0 };
    let mut ops = Vec::new();
    for op in group_ops(sym)? {
        let mp_f = mat2_mul(mat2_mul(ui, to_f(op.mf)), u);
        let m = [
            [mp_f[0][0].round() as i32, mp_f[0][1].round() as i32],
            [mp_f[1][0].round() as i32, mp_f[1][1].round() as i32],
        ];
        let uit = mat2_vec(ui, op.t);
        let mf = to_f(m);
        let t = [
            uit[0] + s[0] - (mf[0][0] * s[0] + mf[0][1] * s[1]),
            uit[1] + s[1] - (mf[1][0] * s[0] + mf[1][1] * s[1]),
        ];
        ops.push(PrimOp { m, t });
    }
    Ok(Setup { a, a_inv, u, s, ops })
}

/// All ops of `group` (conventional ITA ops, centring included) as Cartesian isometries of the drawn `cell`
/// whose conventional origin is `frame.origin` (same convention as `LatticeFamily::ops_in_cell`).
pub fn group_isometries_in_cell(group: &str, cell: &Cell2D, frame: Frame2D) -> Result<Vec<Isometry2>, String> {
    let st = setup(group, cell, frame, false)?;
    Ok(st
        .ops
        .iter()
        .map(|op| Isometry2 {
            m: mat2_mul(mat2_mul(st.a, to_f(op.m)), st.a_inv),
            t: mat2_vec(st.a, op.t),
        })
        .collect())
}

struct LineRec {
    m: [[i32; 2]; 2],
    dir: [f64; 2],
    offset: f64,
    a: [f64; 2],
    b: [f64; 2],
    glide: f64,
    mirror: bool,
}

/// Clip `x0 + lam * dir` (fractional) to the box `[lo, hi]` (Liang-Barsky); `None` if it misses.
/// `dir[k]` is exactly zero for a line parallel to axis `k` (integer direction), which is kept when it lies
/// in the widened slab `[blo, bhi]` (edge lines); crossing axes are clipped to the exact box, so a line
/// that only touches a corner comes back with (near) zero length.
fn clip_line(x0: [f64; 2], dir: [f64; 2], (lo, hi): ([f64; 2], [f64; 2]), (blo, bhi): ([f64; 2], [f64; 2])) -> Option<(f64, f64)> {
    let (mut l0, mut l1) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..2 {
        if dir[k] == 0.0 {
            if x0[k] < blo[k] || x0[k] > bhi[k] {
                return None;
            }
        } else {
            let (ta, tb) = ((lo[k] - x0[k]) / dir[k], (hi[k] - x0[k]) / dir[k]);
            l0 = l0.max(ta.min(tb));
            l1 = l1.min(ta.max(tb));
        }
    }
    (l1 > l0).then_some((l0, l1))
}

fn rotation_order(m: [[i32; 2]; 2]) -> Option<u8> {
    match m[0][0] + m[1][1] {
        -2 => Some(2),
        -1 => Some(3),
        0 => Some(4),
        1 => Some(6),
        _ => None,
    }
}

/// Largest accepted repeat per axis (the app uses at most 9).
pub const MAX_REPEAT: i32 = 64;

/// Mirror/glide lines and rotation centres of `group` on the drawn `cell` inside the repeat window
/// `n_lo*(a1 + a2) + s*a1 + t*a2`, `s in [0, repeat0]`, `t in [0, repeat1]`, `n_lo = -(repeat/2)` per axis.
pub fn compute_symmetry_elements(
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
    repeat: [i32; 2],
) -> Result<SymmetryElements2D, String> {
    if repeat[0] < 1 || repeat[1] < 1 || repeat[0] > MAX_REPEAT || repeat[1] > MAX_REPEAT {
        return Err(format!("repeat must lie in 1..={MAX_REPEAT}, got {repeat:?}"));
    }
    let sym = canonical_symbol(group)?;
    let st = setup(sym, cell, frame, true)?;
    let lo = [f64::from(-(repeat[0] / 2)), f64::from(-(repeat[1] / 2))];
    let hi = [lo[0] + f64::from(repeat[0]), lo[1] + f64::from(repeat[1])];
    let ext = lo.iter().chain(hi.iter()).fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let ftol = 1e-9 * (1.0 + ext);
    let (blo, bhi) = ([lo[0] - ftol, lo[1] - ftol], [hi[0] + ftol, hi[1] + ftol]);
    let scale = mat2_det(st.a).abs().sqrt();
    let ltol = 1e-9 * scale * (1.0 + ext);
    let k = 3 * (ext as i32) + 4;

    let mut lines: Vec<LineRec> = Vec::new();
    let mut centres: Vec<RotationCentre2D> = Vec::new();
    let mut centre_frac: Vec<[f64; 2]> = Vec::new();
    // Rotation centres sit on the 1/12 grid of primitive coordinates relative to the conventional origin.
    let mut centre_key: HashMap<(i64, i64), usize> = HashMap::new();
    for op in &st.ops {
        let mf = to_f(op.m);
        let det = op.m[0][0] * op.m[1][1] - op.m[0][1] * op.m[1][0];
        let is_identity = op.m == [[1, 0], [0, 1]];
        if is_identity {
            continue;
        }
        let order = if det == 1 { rotation_order(op.m) } else { None };
        // Reflection axis (fractional): a nonzero column of I + Mp.
        let ipm = [[1.0 + mf[0][0], mf[0][1]], [mf[1][0], 1.0 + mf[1][1]]];
        let col = if ipm[0][0].abs() + ipm[1][0].abs() > 0.5 { [ipm[0][0], ipm[1][0]] } else { [ipm[0][1], ipm[1][1]] };
        let mut dcart = mat2_vec(st.a, col);
        let dl = dcart[0].hypot(dcart[1]);
        dcart = [dcart[0] / dl, dcart[1] / dl];
        let mut dfrac = [col[0] / dl, col[1] / dl]; // A dfrac = dcart; zero entries stay exactly zero
        if dcart[0] < -1e-12 || (dcart[0].abs() <= 1e-12 && dcart[1] < 0.0) {
            dcart = [-dcart[0], -dcart[1]];
            dfrac = [-dfrac[0], -dfrac[1]];
        }
        let nrm = [-dcart[1], dcart[0]];
        let imm = [[1.0 - mf[0][0], -mf[0][1]], [-mf[1][0], 1.0 - mf[1][1]]];
        let imm_inv = if order.is_some() { Some(mat2_inv(imm)?) } else { None };
        let c0 = [-op.t[0].round() as i32, -op.t[1].round() as i32];
        for n0 in (c0[0] - k)..=(c0[0] + k) {
            for n1 in (c0[1] - k)..=(c0[1] + k) {
                let tau = [op.t[0] + f64::from(n0), op.t[1] + f64::from(n1)];
                if det == -1 {
                    let tpar = [0.5 * (ipm[0][0] * tau[0] + ipm[0][1] * tau[1]), 0.5 * (ipm[1][0] * tau[0] + ipm[1][1] * tau[1])];
                    let x0 = [0.5 * (tau[0] - tpar[0]), 0.5 * (tau[1] - tpar[1])];
                    let Some((l0, l1)) = clip_line(x0, dfrac, (lo, hi), (blo, bhi)) else { continue };
                    if l1 - l0 <= ltol {
                        continue;
                    }
                    let xc = mat2_vec(st.a, x0);
                    let offset = nrm[0] * xc[0] + nrm[1] * xc[1];
                    let gc = mat2_vec(st.a, tpar);
                    let glide = dcart[0] * gc[0] + dcart[1] * gc[1];
                    let mirror = glide.abs() <= ltol;
                    if let Some(rec) = lines.iter_mut().find(|r| r.m == op.m && (r.offset - offset).abs() <= ltol) {
                        rec.mirror |= mirror;
                        let better = glide.abs() < rec.glide.abs() - ltol
                            || ((glide.abs() - rec.glide.abs()).abs() <= ltol && glide > rec.glide);
                        if better {
                            rec.glide = glide;
                        }
                    } else {
                        let pa = [x0[0] + l0 * dfrac[0], x0[1] + l0 * dfrac[1]];
                        let pb = [x0[0] + l1 * dfrac[0], x0[1] + l1 * dfrac[1]];
                        lines.push(LineRec {
                            m: op.m,
                            dir: dcart,
                            offset,
                            a: mat2_vec(st.a, pa),
                            b: mat2_vec(st.a, pb),
                            glide,
                            mirror,
                        });
                    }
                } else if let (Some(order), Some(imm_inv)) = (order, imm_inv) {
                    let c = mat2_vec(imm_inv, tau);
                    if c[0] < blo[0] || c[0] > bhi[0] || c[1] < blo[1] || c[1] > bhi[1] {
                        continue;
                    }
                    let g = [12.0 * (c[0] - st.s[0]), 12.0 * (c[1] - st.s[1])];
                    let key = (g[0].round() as i64, g[1].round() as i64);
                    if (g[0] - g[0].round()).abs() > 1e-6 || (g[1] - g[1].round()).abs() > 1e-6 {
                        return Err(format!("{sym}: rotation centre off the 1/12 grid"));
                    }
                    if let Some(&idx) = centre_key.get(&key) {
                        centres[idx].order = centres[idx].order.max(order);
                    } else {
                        centre_key.insert(key, centres.len());
                        centre_frac.push(c);
                        centres.push(RotationCentre2D { pos: mat2_vec(st.a, c), order, wyckoff: String::new() });
                    }
                }
            }
        }
    }
    for (rc, c) in centres.iter_mut().zip(&centre_frac) {
        let f = mat2_vec(st.u, [c[0] - st.s[0], c[1] - st.s[1]]);
        rc.wyckoff = label_point(sym, f)?.name;
    }
    centres.sort_by(|p, q| (p.pos[1], p.pos[0]).partial_cmp(&(q.pos[1], q.pos[0])).unwrap_or(std::cmp::Ordering::Equal));
    lines.sort_by(|p, q| {
        let ap = p.dir[1].atan2(p.dir[0]);
        let aq = q.dir[1].atan2(q.dir[0]);
        (ap, p.offset).partial_cmp(&(aq, q.offset)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let out_lines = lines
        .into_iter()
        .map(|r| {
            let g = if r.mirror { 0.0 } else { r.glide };
            SymLine2D {
                kind: if r.mirror { SymLineKind::Mirror } else { SymLineKind::Glide },
                a: r.a,
                b: r.b,
                glide: [g * r.dir[0], g * r.dir[1]],
            }
        })
        .collect();
    Ok(SymmetryElements2D { lines: out_lines, centres })
}
