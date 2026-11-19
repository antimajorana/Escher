//! Continuum displacement and stress of a straight dislocation (line along `z`, in-plane Burgers vector `b`) in a 2D
//! anisotropic medium `C_{iajb}` (`#041`).
//!
//! Main solver ([`stroh_dislocation_field`], Barnett-Lothe angular integral, also valid for degenerate roots): write
//! `u = c ln r + v(theta)`; with `m = (cos, sin)`, `n = (-sin, cos)` and `(xy)_ij = C_iajb x_a y_b`, equilibrium makes
//! the traction on radial lines constant, `T = (nm) c + (nn) v'`, so `v' = (nn)^-1 (T - (nm) c)`. `T` and `c` follow
//! from the jump `int v' = b` and zero net force `int [(mm) c + (mn) v'] = 0` (trapezoid sums of periodic integrands,
//! exact to rounding); `v(theta) = int_{-pi}^{theta} v'` by composite Gauss-Legendre. Then `du_j/dx_a = (c_j m_a +
//! v'_j n_a) / r`.
//!
//! Cross-checks: [`stroh_sextic_field`] (distinct roots `p`, Im p > 0, of `det(Q + p(R + R^T) + p^2 T) = 0`,
//! `Q_ik = C_i1k1`, `R_ik = C_i1k2`, `T_ik = C_i2k2`; `u = (1/pi) Im sum a ln(x + p y) (l.b)`, `l = (R^T + pT) a`,
//! `2 a.l = 1`) and [`isotropic_dislocation_field`] (classical plane strain, `lambda = C_1122`, `mu = C_1212`). All three
//! put the branch cut on the negative `x` axis with `u` jumping by `+b` counter-clockwise; `sigma_ia = C_iajb du_j/dx_b`.
//! [`antiplane_screw_field`] adds the decoupled anti-plane (screw) part for given anti-plane moduli.
//!
//! Sources: the integral formalism is Barnett & Lothe, Phys. Norv. 7 (1973) 13 and Hirth & Lothe, *Theory of
//! Dislocations*, 2nd ed. (1982) ch. 13; the sextic form and normalisation `2 a.l = 1`, `u = (1/pi) Im[A <ln z> L^T] b`
//! are Ting, *Anisotropic Elasticity* (1996) ch. 5 and 8. Sign: a field whose `u` jumps by `+b` counter-clockwise is the
//! FS/RH Burgers vector `b` of Hirth & Lothe (1982) ch. 1 for line direction `+z` (out of the page). A crystal
//! displaced by it has snapped-bond closure `sum G = -b` round a counter-clockwise loop (see `nye_loops`).

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use super::gsv_stroh::{ElasticClass, Tensor4, classify_elastic, contract};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Cx(f64, f64);

impl Cx {
    fn plus(self, o: Cx) -> Cx { Cx(self.0 + o.0, self.1 + o.1) }
    fn minus(self, o: Cx) -> Cx { Cx(self.0 - o.0, self.1 - o.1) }
    fn times(self, o: Cx) -> Cx { Cx(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0) }
    fn scale(self, s: f64) -> Cx { Cx(self.0 * s, self.1 * s) }
    fn over(self, o: Cx) -> Cx { let d = o.0 * o.0 + o.1 * o.1; Cx((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d) }
    fn abs(self) -> f64 { self.0.hypot(self.1) }
    fn ln(self) -> Cx { Cx(self.abs().ln(), self.1.atan2(self.0)) }
    fn sqrt(self) -> Cx { let r = self.abs().sqrt(); let t = 0.5 * self.1.atan2(self.0); Cx(r * t.cos(), r * t.sin()) }
}

/// Field values at one point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrohField {
    pub u: Vec<[f64; 2]>,
    /// `grad[j][b] = du_j / dx_b`.
    pub grad: Vec<[[f64; 2]; 2]>,
    pub stress: Vec<[[f64; 2]; 2]>,
    /// Logarithmic coefficient `c` and radial-line traction `T` (angular-integral solver; zero for the others).
    pub log_coeff: [f64; 2],
    pub traction: [f64; 2],
    pub method: String,
}

fn quad_coeffs(q: f64, r2: f64, t: f64) -> [f64; 3] {
    [q, r2, t]
}

fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

fn polyval(c: &[f64], z: Cx) -> Cx {
    c.iter().rev().fold(Cx(0.0, 0.0), |acc, &k| acc.times(z).plus(Cx(k, 0.0)))
}

/// Four roots of `c0 + c1 z + c2 z^2 + c3 z^3 + c4 z^4` (Durand-Kerner, then Newton polishing).
fn quartic_roots(c: &[f64]) -> Vec<Cx> {
    let lead = c[4];
    let mono: Vec<f64> = c.iter().map(|x| x / lead).collect();
    let seed = Cx(0.4, 0.9);
    let mut z: Vec<Cx> = (0..4).map(|k| (0..k).fold(Cx(1.0, 0.0), |p, _| p.times(seed))).collect();
    for _ in 0..500 {
        let mut delta = 0.0f64;
        for i in 0..4 {
            let mut den = Cx(1.0, 0.0);
            for j in 0..4 {
                if i != j { den = den.times(z[i].minus(z[j])); }
            }
            let step = polyval(&mono, z[i]).over(den);
            z[i] = z[i].minus(step);
            delta = delta.max(step.abs());
        }
        if delta < 1e-15 { break; }
    }
    let d: Vec<f64> = (1..mono.len()).map(|k| k as f64 * mono[k]).collect();
    for zi in &mut z {
        for _ in 0..3 {
            let dv = polyval(&d, *zi);
            if dv.abs() > 1e-300 { *zi = zi.minus(polyval(&mono, *zi).over(dv)); }
        }
    }
    z
}

struct StrohModes {
    p: [Cx; 2],
    a: [[Cx; 2]; 2],
    l: [[Cx; 2]; 2],
}

fn stroh_modes(c: &Tensor4) -> Result<StrohModes, String> {
    let (q, r, t) = (|i: usize, k: usize| c[i][0][k][0], |i: usize, k: usize| c[i][0][k][1], |i: usize, k: usize| c[i][1][k][1]);
    let m = |i: usize, k: usize| quad_coeffs(q(i, k), r(i, k) + r(k, i), t(i, k));
    let det: Vec<f64> = polymul(&m(0, 0), &m(1, 1)).iter().zip(polymul(&m(0, 1), &m(1, 0))).map(|(x, y)| x - y).collect();
    if det[4].abs() < 1e-14 {
        return Err("degenerate Stroh polynomial (T singular)".to_string());
    }
    let mut roots: Vec<Cx> = quartic_roots(&det).into_iter().filter(|z| z.1 > 0.0).collect();
    if roots.len() != 2 {
        return Err("Stroh roots are not two conjugate pairs (medium not strongly elliptic)".to_string());
    }
    roots.sort_by(|x, y| x.0.total_cmp(&y.0));
    if roots[0].minus(roots[1]).abs() < 1e-6 {
        return Err("degenerate Stroh roots (near-isotropic medium)".to_string());
    }
    let mut a = [[Cx(0.0, 0.0); 2]; 2];
    let mut l = [[Cx(0.0, 0.0); 2]; 2];
    for (al, &p) in roots.iter().enumerate() {
        let mm = |i: usize, k: usize| { let k3 = m(i, k); Cx(k3[0], 0.0).plus(p.scale(k3[1])).plus(p.times(p).scale(k3[2])) };
        let (r0, r1) = ([mm(0, 0), mm(0, 1)], [mm(1, 0), mm(1, 1)]);
        let row = if r0[0].abs() + r0[1].abs() >= r1[0].abs() + r1[1].abs() { r0 } else { r1 };
        let av = [row[1].scale(-1.0), row[0]];
        let lv: [Cx; 2] = std::array::from_fn(|i| (0..2).fold(Cx(0.0, 0.0), |s, k| s.plus(Cx(r(k, i), 0.0).plus(p.scale(t(i, k))).times(av[k]))));
        let norm = av[0].times(lv[0]).plus(av[1].times(lv[1])).scale(2.0).sqrt();
        a[al] = [av[0].over(norm), av[1].over(norm)];
        l[al] = [lv[0].over(norm), lv[1].over(norm)];
    }
    Ok(StrohModes { p: [roots[0], roots[1]], a, l })
}

fn with_stress(c: &Tensor4, mut f: StrohField) -> StrohField {
    f.stress = f.grad.iter().map(|g| {
        std::array::from_fn(|i| std::array::from_fn(|a| (0..4).map(|k| c[i][a][k / 2][k % 2] * g[k / 2][k % 2]).sum::<f64>()))
    }).collect();
    f
}

fn empty(method: &str) -> StrohField {
    StrohField { u: Vec::new(), grad: Vec::new(), stress: Vec::new(), log_coeff: [0.0; 2], traction: [0.0; 2], method: method.to_string() }
}

/// Stroh sextic solution; refuses (near-)degenerate roots.
pub fn stroh_sextic_field(c: &Tensor4, b: [f64; 2], points: &[[f64; 2]]) -> Result<StrohField, String> {
    let sm = stroh_modes(c)?;
    let mut f = empty("Stroh sextic (distinct roots)");
    for &x in points {
        let (mut u, mut g) = ([0.0; 2], [[0.0; 2]; 2]);
        for al in 0..2 {
            let z = Cx(x[0], 0.0).plus(sm.p[al].scale(x[1]));
            let lb = sm.l[al][0].scale(b[0]).plus(sm.l[al][1].scale(b[1]));
            let (lz, inv) = (z.ln(), Cx(1.0, 0.0).over(z));
            for j in 0..2 {
                let k = sm.a[al][j].times(lb);
                u[j] += k.times(lz).1 / PI;
                g[j][0] += k.times(inv).1 / PI;
                g[j][1] += k.times(inv).times(sm.p[al]).1 / PI;
            }
        }
        f.u.push(u);
        f.grad.push(g);
    }
    Ok(with_stress(c, f))
}

/// Classical isotropic solution with `lambda = C_1122`, `mu = C_1212`; REFUSES a `C` that [`classify_elastic`] does not
/// find isotropic (the closed form would silently be wrong there).
pub fn isotropic_dislocation_field(c: &Tensor4, b: [f64; 2], points: &[[f64; 2]]) -> Result<StrohField, String> {
    if classify_elastic(c) != ElasticClass::Isotropic {
        return Err("REFUSED: the isotropic closed form needs an isotropic C (use stroh_dislocation_field)".to_string());
    }
    let (lam, mu) = (c[0][0][1][1], c[0][1][0][1]);
    let mut f = empty("isotropic closed form (plane strain, Hirth & Lothe 1982 ch. 3)");
    for &x in points {
        if x[0] == 0.0 && x[1] == 0.0 {
            return Err("field is singular at the dislocation line".to_string());
        }
        let (u, g) = isotropic_point(lam, mu, b, x);
        f.u.push(u);
        f.grad.push(g);
    }
    Ok(with_stress(c, f))
}

/// Anti-plane (screw) field of a straight dislocation along `z` with Burgers vector `b_z`, for anti-plane moduli
/// `K_ab = C_{a3b3}` (symmetric positive definite). Equilibrium `K_ab d_a d_b u_z = 0` is solved by `f(x + p y)` with
/// `K_11 + 2 K_12 p + K_22 p^2 = 0`, `p = (-K_12 + i sqrt(det K)) / K_22` (Eshelby, Read & Shockley 1953; Hirth & Lothe
/// 1982 ch. 13), so `u_z = (b_z / 2 pi) arg(x + p y)`: cut on the negative `x` axis, jump `+b_z` counter-clockwise.
/// For `K = mu I` (`p = i`) this is the Volterra screw `u_z = b_z theta / (2 pi)` (Hirth & Lothe 1982 ch. 3).
/// Returns `(u_z, grad u_z, sigma_z.)` per point.
pub fn antiplane_screw_field(k: [[f64; 2]; 2], bz: f64, points: &[[f64; 2]]) -> Result<AntiplaneField, String> {
    let det = k[0][0] * k[1][1] - k[0][1] * k[1][0];
    if (k[0][1] - k[1][0]).abs() > 1e-12 * (k[0][0].abs() + k[1][1].abs()) || k[0][0] <= 0.0 || det <= 0.0 {
        return Err("REFUSED: anti-plane moduli K must be symmetric positive definite".to_string());
    }
    let p = Cx(-k[0][1] / k[1][1], det.sqrt() / k[1][1]);
    let s = bz / (2.0 * PI);
    let mut out = AntiplaneField { uz: Vec::new(), grad: Vec::new(), stress: Vec::new() };
    for &x in points {
        let z = Cx(x[0], 0.0).plus(p.scale(x[1]));
        if z.abs() == 0.0 {
            return Err("field is singular at the dislocation line".to_string());
        }
        let inv = Cx(1.0, 0.0).over(z);
        let g = [s * inv.1, s * inv.times(p).1];
        out.uz.push(s * z.1.atan2(z.0));
        out.grad.push(g);
        out.stress.push([k[0][0] * g[0] + k[0][1] * g[1], k[1][0] * g[0] + k[1][1] * g[1]]);
    }
    Ok(out)
}

/// Anti-plane screw field values: `u_z`, `d u_z / d x_a`, and `sigma_{za} = K_ab d u_z / d x_b`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AntiplaneField {
    pub uz: Vec<f64>,
    pub grad: Vec<[f64; 2]>,
    pub stress: Vec<[f64; 2]>,
}

fn inv2(m: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    [[m[1][1] / d, -m[0][1] / d], [-m[1][0] / d, m[0][0] / d]]
}

fn mv(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
}

fn mm(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j]))
}

fn vprime(c: &Tensor4, lc: [f64; 2], t: [f64; 2], th: f64) -> [f64; 2] {
    let (m, n) = ([th.cos(), th.sin()], [-th.sin(), th.cos()]);
    let r = mv(contract(c, n, m), lc);
    mv(inv2(contract(c, n, n)), [t[0] - r[0], t[1] - r[1]])
}

/// Radial traction `T` and log coefficient `c` from `[P, -Q; V, M - W] [T; c] = [b / 2pi; 0]` with the angular means
/// `P = <nn^-1>`, `Q = <nn^-1 nm>`, `V = <mn nn^-1>`, `M - W = <mm - mn nn^-1 nm>` (the Barnett-Lothe `B`).
fn angular_constants(c: &Tensor4, b: [f64; 2]) -> Result<([f64; 2], [f64; 2]), String> {
    let nq = 512;
    let w8 = 1.0 / nq as f64;
    let (mut p, mut q, mut v, mut w) = ([[0.0; 2]; 2], [[0.0; 2]; 2], [[0.0; 2]; 2], [[0.0; 2]; 2]);
    for k in 0..nq {
        let th = 2.0 * PI * k as f64 / nq as f64;
        let (m, n) = ([th.cos(), th.sin()], [-th.sin(), th.cos()]);
        let nni = inv2(contract(c, n, n));
        let (nm, mn, mmt) = (contract(c, n, m), contract(c, m, n), contract(c, m, m));
        let (a1, a2) = (mm(nni, nm), mm(mn, nni));
        let a3 = mm(a2, nm);
        for i in 0..2 {
            for j in 0..2 {
                p[i][j] += nni[i][j] * w8;
                q[i][j] += a1[i][j] * w8;
                v[i][j] += a2[i][j] * w8;
                w[i][j] += (mmt[i][j] - a3[i][j]) * w8;
            }
        }
    }
    let mut a = [[0.0; 5]; 4];
    for i in 0..2 {
        for j in 0..2 {
            a[i][j] = p[i][j];
            a[i][j + 2] = -q[i][j];
            a[i + 2][j] = v[i][j];
            a[i + 2][j + 2] = w[i][j];
        }
    }
    a[0][4] = b[0] / (2.0 * PI);
    a[1][4] = b[1] / (2.0 * PI);
    for col in 0..4 {
        let piv = (col..4).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs())).unwrap_or(col);
        a.swap(col, piv);
        if a[col][col].abs() < 1e-14 {
            return Err("singular angular system (medium not strongly elliptic)".to_string());
        }
        for r in 0..4 {
            if r != col {
                let f = a[r][col] / a[col][col];
                for k in col..5 {
                    a[r][k] -= f * a[col][k];
                }
            }
        }
    }
    Ok(([a[0][4] / a[0][0], a[1][4] / a[1][1]], [a[2][4] / a[2][2], a[3][4] / a[3][3]]))
}

const GL_X: [f64; 4] = [0.183_434_642_495_649_8, 0.525_532_409_916_329, 0.796_666_477_413_626_7, 0.960_289_856_497_536_3];
const GL_W: [f64; 4] = [0.362_683_783_378_362, 0.313_706_645_877_887_3, 0.222_381_034_453_374_5, 0.101_228_536_290_376_3];

/// Displacement, gradient and stress of a dislocation at the origin with Burgers vector `b` (angular-integral solver).
pub fn stroh_dislocation_field(c: &Tensor4, b: [f64; 2], points: &[[f64; 2]]) -> Result<StrohField, String> {
    let (t, lc) = angular_constants(c, b)?;
    let mut f = empty("Barnett-Lothe angular integral");
    f.log_coeff = lc;
    f.traction = t;
    for &x in points {
        let r = x[0].hypot(x[1]);
        if r == 0.0 {
            return Err("field is singular at the dislocation line".to_string());
        }
        let th = x[1].atan2(x[0]);
        let panels = (((th + PI) / (PI / 32.0)).ceil() as usize).max(1);
        let h = (th + PI) / panels as f64;
        let mut v = [0.0; 2];
        for k in 0..panels {
            let mid = -PI + (k as f64 + 0.5) * h;
            for (xg, wg) in GL_X.iter().zip(GL_W) {
                for s in [-1.0, 1.0] {
                    let vp = vprime(c, lc, t, mid + s * xg * 0.5 * h);
                    v[0] += wg * 0.5 * h * vp[0];
                    v[1] += wg * 0.5 * h * vp[1];
                }
            }
        }
        let lr = r.ln();
        f.u.push([lc[0] * lr + v[0], lc[1] * lr + v[1]]);
        let (m, n, vp) = ([x[0] / r, x[1] / r], [-x[1] / r, x[0] / r], vprime(c, lc, t, th));
        f.grad.push(std::array::from_fn(|j| std::array::from_fn(|a| (lc[j] * m[a] + vp[j] * n[a]) / r)));
    }
    Ok(with_stress(c, f))
}

/// Classical edge dislocation along `x` (Hirth & Lothe), rotated to the direction of `b`; with `k1 = 1/(2(1-nu))`
/// `= (lambda+mu)/(lambda+2mu)` and `k2 = (1-2nu)/(4(1-nu)) = mu/(2(lambda+2mu))`:
/// `u_x = b/(2pi) [theta + k1 x y / r^2]`, `u_y = -b/(2pi) [k2 ln r^2 + (k1/2)(x^2 - y^2)/r^2]`.
fn isotropic_point(lam: f64, mu: f64, b: [f64; 2], x: [f64; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
    let bl = b[0].hypot(b[1]);
    if bl == 0.0 {
        return ([0.0; 2], [[0.0; 2]; 2]);
    }
    let (k1, k2) = ((lam + mu) / (lam + 2.0 * mu), mu / (2.0 * (lam + 2.0 * mu)));
    let (cph, sph) = (b[0] / bl, b[1] / bl);
    let (xr, yr) = (cph * x[0] + sph * x[1], -sph * x[0] + cph * x[1]);
    let r2 = xr * xr + yr * yr;
    let s = bl / (2.0 * PI);
    // theta' = theta - phi with theta in (-pi, pi]: same gradient, cut kept on the global negative x axis
    let th = x[1].atan2(x[0]) - sph.atan2(cph);
    let ux = s * (th + k1 * xr * yr / r2);
    let uy = -s * (k2 * r2.ln() + 0.5 * k1 * (xr * xr - yr * yr) / r2);
    // derivatives in the rotated frame
    let r4 = r2 * r2;
    let dux = [s * (-yr / r2 + k1 * yr * (yr * yr - xr * xr) / r4), s * (xr / r2 + k1 * xr * (xr * xr - yr * yr) / r4)];
    let duy = [-s * (2.0 * k2 * xr / r2 + 2.0 * k1 * xr * yr * yr / r4), -s * (2.0 * k2 * yr / r2 - 2.0 * k1 * xr * xr * yr / r4)];
    let rm = [[cph, -sph], [sph, cph]];
    let u = mv(rm, [ux, uy]);
    // grad (global) = R G' R^T with G'[j][b] = d u'_j / d x'_b
    (u, mm(mm(rm, [dux, duy]), [[cph, sph], [-sph, cph]]))
}
