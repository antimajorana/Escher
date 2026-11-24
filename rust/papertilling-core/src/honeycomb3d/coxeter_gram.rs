//! Exact classification of Schläfli symbols `{p,q,r}` and realisation of the Coxeter group
//! `[p,q,r]` by reflections in R⁴ (S³) or Minkowski space R^{3,1} (H³).
//!
//! The regime is decided by integers only: `cell_k = (p-2)(q-2)` and `vfig_k = (q-2)(r-2)` are
//! `< 4`, `== 4`, `> 4` exactly when the cell `{p,q}` / vertex figure `{q,r}` is spherical,
//! Euclidean, hyperbolic. The Gram determinant is reported as a diagnostic only: `det G < 0` alone
//! does not give a finite-volume honeycomb (e.g. `{3,3,7}` has `det G < 0` but a hyperbolic vertex
//! figure), so hyperideal symbols are refused by the integer test.
//!
//! Sources (published mathematical facts, no copied tables):
//! * H. S. M. Coxeter, *Regular Polytopes*, 3rd ed. (Dover, 1973), Table I(ii): the six regular
//!   convex polychora `{3,3,3}, {4,3,3}, {3,3,4}, {3,4,3}, {5,3,3}, {3,3,5}` and their counts.
//! * H. S. M. Coxeter, "Regular honeycombs in hyperbolic space", Proc. ICM 1954, vol. III,
//!   pp. 155-169: the 4 compact and 11 paracompact regular honeycombs of H³.
//! * J. E. Humphreys, *Reflection Groups and Coxeter Groups* (CUP, 1990), §6.9: the compact and
//!   noncompact hyperbolic Coxeter simplex groups; §2.3 / §6.8 Gram (bilinear-form) criterion.
//! * The tridiagonal Gram determinant of the linear diagram `p-q-r` is
//!   `1 - cos²(π/p) - cos²(π/q) - cos²(π/r) + cos²(π/p) cos²(π/r)` (direct cofactor expansion).
//!
//! Exhaustiveness of the case table (hand enumeration, `[PROVED]`): with `p, q, r >= 3`, both
//! `{p,q}` and `{q,r}` spherical gives exactly 11 symbols (q = 3: p, r ∈ {3,4,5}; `{3,4,3}`;
//! `{3,5,3}`), namely the 6 polychora, `{4,3,4}` and the 4 compact H³ honeycombs; one of them
//! Euclidean and the other spherical or Euclidean gives exactly the 11 paracompact symbols.

use std::f64::consts::PI;

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum HoneycombRegime {
    SphericalS3,
    EuclideanE3,
    HyperbolicCompactH3,
    HyperbolicParacompactH3,
    HyperidealRefused,
    DegenerateRefused,
}

#[derive(Clone, Debug, Serialize)]
pub struct Classification {
    pub p: u32,
    pub q: u32,
    pub r: u32,
    pub regime: HoneycombRegime,
    /// det of the 4×4 Gram matrix `1 - a² - b² - c² + a²c²` (diagnostic; sign agrees with the regime).
    pub det_g: f64,
    pub cell_k: i64,
    pub vfig_k: i64,
    pub has_ideal_vertices: bool,
    pub has_infinite_euclidean_cells: bool,
    pub reason: String,
}

/// The sparsest single orbit of finite sites for a supported regular H³ reflection group.
/// The reciprocal stabilizer order is the number of sites per fundamental chamber.
/// Ties prefer existing vertices, then cell centres, then face centres.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteOrbit {
    Vertex,
    Cell,
    Face,
    Edge,
    FaceCell,
}

impl SiteOrbit {
    pub fn label(self) -> &'static str {
        match self {
            Self::Vertex => "vertices",
            Self::Cell => "cell centres",
            Self::Face => "face centres",
            Self::Edge => "edge centres",
            Self::FaceCell => "sites just inside each square face",
        }
    }
}

impl Classification {
    pub fn finite_site_orbit(&self) -> Option<(SiteOrbit, u32)> {
        if !matches!(self.regime, HoneycombRegime::HyperbolicCompactH3 | HoneycombRegime::HyperbolicParacompactH3) {
            return None;
        }
        // In [4,4,4], the two finite corner orbits each have regular octahedral
        // Dirichlet cells: their points alone gain extra mirrors. A site just
        // inside each face has D4 stabilizer, the largest remaining finite order.
        if (self.p, self.q, self.r) == (4, 4, 4) {
            return Some((SiteOrbit::FaceCell, 8));
        }
        // Number of flags of a spherical {a,b}. A Euclidean link is a cusp.
        let flags = |a: u32, b: u32| {
            let excess = 2 * (a + b) - a * b;
            (8_u32 * a * b).checked_div(excess).unwrap_or(0)
        };
        let (p,q,r) = (self.p,self.q,self.r);
        let options = [
            (SiteOrbit::Vertex, flags(q,r)), (SiteOrbit::Cell, flags(p,q)),
            (SiteOrbit::Face, 4*p), (SiteOrbit::Edge, 4*r),
        ];
        options.into_iter().rev().max_by_key(|&(_, order)| order)
    }

    pub fn is_buildable(&self) -> bool {
        !matches!(self.regime, HoneycombRegime::HyperidealRefused | HoneycombRegime::DegenerateRefused)
    }
}

fn kind(k: i64) -> &'static str {
    match k.cmp(&4) {
        std::cmp::Ordering::Less => "spherical",
        std::cmp::Ordering::Equal => "Euclidean",
        std::cmp::Ordering::Greater => "hyperbolic",
    }
}

pub fn gram_det(p: u32, q: u32, r: u32) -> f64 {
    let a = (PI / p as f64).cos();
    let b = (PI / q as f64).cos();
    let c = (PI / r as f64).cos();
    1.0 - a * a - b * b - c * c + a * a * c * c
}

/// Classify `{p,q,r}` exactly (integer case table; no floating-point test at zero).
pub fn classify_schlafli(p: u32, q: u32, r: u32) -> Classification {
    let mut out = Classification {
        p,
        q,
        r,
        regime: HoneycombRegime::DegenerateRefused,
        det_g: f64::NAN,
        cell_k: 0,
        vfig_k: 0,
        has_ideal_vertices: false,
        has_infinite_euclidean_cells: false,
        reason: String::new(),
    };
    if p < 3 || q < 3 || r < 3 {
        out.reason = format!(
            "[PROVED: reducible or degenerate Schläfli symbol {{{p},{q},{r}}} (requires p, q, r >= 3; \
             an entry 2 splits the Coxeter diagram, e.g. {{4,4,2}} has det G = 0 but is a reducible dihedron)]"
        );
        if p >= 2 && q >= 2 && r >= 2 {
            out.det_g = gram_det(p, q, r);
        }
        return out;
    }
    let cell_k = (i64::from(p) - 2).saturating_mul(i64::from(q) - 2);
    let vfig_k = (i64::from(q) - 2).saturating_mul(i64::from(r) - 2);
    out.cell_k = cell_k;
    out.vfig_k = vfig_k;
    out.det_g = gram_det(p, q, r);
    if cell_k > 4 || vfig_k > 4 {
        out.regime = HoneycombRegime::HyperidealRefused;
        let mut why = Vec::new();
        if cell_k > 4 {
            why.push(format!(
                "the cell {{{p},{q}}} is hyperbolic ((p-2)(q-2) = {cell_k} > 4): its cells are ultra-infinite, \
                 extending beyond the sphere at infinity S²_∞"
            ));
        }
        if vfig_k > 4 {
            why.push(format!(
                "the vertex figure {{{q},{r}}} is hyperbolic ((q-2)(r-2) = {vfig_k} > 4): its vertices are \
                 ultra-ideal, outside the Klein ball"
            ));
        }
        out.reason = format!("[PROVED: hyperideal symbol {{{p},{q},{r}}}; {}]", why.join("; "));
        return out;
    }
    out.has_ideal_vertices = vfig_k == 4;
    out.has_infinite_euclidean_cells = cell_k == 4;
    if cell_k < 4 && vfig_k < 4 {
        out.regime = match (p, q, r) {
            (3, 3, 3) | (4, 3, 3) | (3, 3, 4) | (3, 4, 3) | (5, 3, 3) | (3, 3, 5) => HoneycombRegime::SphericalS3,
            (4, 3, 4) => HoneycombRegime::EuclideanE3,
            _ => HoneycombRegime::HyperbolicCompactH3,
        };
        out.reason = match out.regime {
            HoneycombRegime::SphericalS3 => format!("[PROVED: {{{p},{q},{r}}} is a regular convex polychoron of S³ (det G > 0)]"),
            HoneycombRegime::EuclideanE3 => "[PROVED: {4,3,4} is the cubic honeycomb of E³ (det G = 0)]".into(),
            _ => format!("[PROVED: {{{p},{q},{r}}} is a compact regular honeycomb of H³ (det G < 0)]"),
        };
        return out;
    }
    out.regime = HoneycombRegime::HyperbolicParacompactH3;
    out.reason = format!(
        "[PROVED: {{{p},{q},{r}}} is a paracompact regular honeycomb of H³ (det G < 0): cell {{{p},{q}}} is {}, \
         vertex figure {{{q},{r}}} is {}{}{}]",
        kind(cell_k),
        kind(vfig_k),
        if out.has_ideal_vertices { "; vertices are ideal (on S²_∞), horosphere vertex figures" } else { "" },
        if out.has_infinite_euclidean_cells { "; cells are infinite horospherical tilings, truncated to finite patches" } else { "" },
    );
    out
}

pub type Mat4 = [[f64; 4]; 4];
pub type Vec4 = [f64; 4];

pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            c[i][j] = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

pub fn mat_vec(a: &Mat4, x: &Vec4) -> Vec4 {
    let mut y = [0.0; 4];
    for (yi, row) in y.iter_mut().zip(a) {
        *yi = (0..4).map(|k| row[k] * x[k]).sum();
    }
    y
}

pub fn identity4() -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

/// Gauss–Jordan inverse with partial pivoting.
pub fn inverse4(a: &Mat4) -> Option<Mat4> {
    let mut m = *a;
    let mut inv = identity4();
    for col in 0..4 {
        let piv = (col..4).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[piv][col].abs() < 1e-13 {
            return None;
        }
        m.swap(col, piv);
        inv.swap(col, piv);
        let d = m[col][col];
        for k in 0..4 {
            m[col][k] /= d;
            inv[col][k] /= d;
        }
        for row in 0..4 {
            if row != col {
                let f = m[row][col];
                for k in 0..4 {
                    m[row][k] -= f * m[col][k];
                    inv[row][k] -= f * inv[col][k];
                }
            }
        }
    }
    Some(inv)
}

/// Cyclic Jacobi eigen-decomposition of a symmetric 4×4 matrix: `(eigenvalues, eigenvectors as columns)`.
pub fn jacobi_eigen(a: &Mat4) -> (Vec4, Mat4) {
    let mut m = *a;
    let mut v = identity4();
    for _ in 0..100 {
        let off: f64 = (0..4).flat_map(|i| (0..4).map(move |j| (i, j))).filter(|(i, j)| i != j).map(|(i, j)| m[i][j] * m[i][j]).sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                if m[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in m.iter_mut() {
                    let (mkp, mkq) = (row[p], row[q]);
                    row[p] = c * mkp - s * mkq;
                    row[q] = s * mkp + c * mkq;
                }
                for k in 0..4 {
                    let (mpk, mqk) = (m[p][k], m[q][k]);
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    ([m[0][0], m[1][1], m[2][2], m[3][3]], v)
}

/// Reflection group `[p,q,r]` realised in R⁴ with metric `diag(eta)`; `eta[0] = -1` for H³.
#[derive(Clone, Debug)]
pub struct CoxeterRealisation {
    pub eta: Vec4,
    pub normals: [Vec4; 4],
    pub reflections: [Mat4; 4],
    /// Fundamental points: `w[0]` vertex, `w[1]` edge mid, `w[2]` face centre, `w[3]` cell centre.
    pub w: [Vec4; 4],
}

pub fn form(eta: &Vec4, x: &Vec4, y: &Vec4) -> f64 {
    (0..4).map(|i| eta[i] * x[i] * y[i]).sum()
}

fn reflection_matrix(eta: &Vec4, n: &Vec4) -> Mat4 {
    let mut m = identity4();
    for (a, row) in m.iter_mut().enumerate() {
        for (b, x) in row.iter_mut().enumerate() {
            *x -= 2.0 * n[a] * eta[b] * n[b];
        }
    }
    m
}

impl CoxeterRealisation {
    /// Realise `[p,q,r]` (spherical when det G > 0, Lorentzian when det G < 0).
    pub fn new(p: u32, q: u32, r: u32) -> Result<Self, String> {
        let ms = [p, q, r];
        let mut g = identity4();
        for (i, m) in ms.iter().enumerate() {
            let c = -(PI / *m as f64).cos();
            g[i][i + 1] = c;
            g[i + 1][i] = c;
        }
        let (lam, q_mat) = jacobi_eigen(&g);
        let negs: Vec<usize> = (0..4).filter(|&i| lam[i] < -1e-12).collect();
        if negs.len() > 1 || lam.iter().any(|l| l.abs() <= 1e-12) {
            return Err(format!("Gram matrix of [{p},{q},{r}] is not of signature (4,0) or (3,1): {lam:?}"));
        }
        // Coordinate order: the negative eigenvalue (if any) becomes coordinate 0 (time).
        let mut order: Vec<usize> = negs.clone();
        order.extend((0..4).filter(|i| !negs.contains(i)));
        let eta = if negs.is_empty() { [1.0; 4] } else { [-1.0, 1.0, 1.0, 1.0] };
        let mut normals = [[0.0; 4]; 4];
        for (k, nk) in normals.iter_mut().enumerate() {
            for (c, &e) in order.iter().enumerate() {
                nk[c] = q_mat[k][e] * lam[e].abs().sqrt();
            }
        }
        let mut real = Self::from_normals(eta, normals)?;
        if eta[0] < 0.0 {
            // Boost so that the cell centre (or the face centre when cells are infinite) sits at the origin.
            let c3 = real.w[3];
            let centre = if form(&eta, &c3, &c3) < -1e-9 { c3 } else { real.w[2] };
            let b = boost_to_origin(&centre);
            let nb = normals.map(|n| mat_vec(&b, &n));
            real = Self::from_normals(eta, nb)?;
        }
        Ok(real)
    }

    fn from_normals(eta: Vec4, normals: [Vec4; 4]) -> Result<Self, String> {
        let reflections = normals.map(|n| reflection_matrix(&eta, &n));
        let ne: Mat4 = normals.map(|n| [n[0] * eta[0], n[1] * eta[1], n[2] * eta[2], n[3] * eta[3]]);
        let inv = inverse4(&ne).ok_or("singular mirror system")?;
        let mut w = [[0.0; 4]; 4];
        for (j, wj) in w.iter_mut().enumerate() {
            for (x, inv_row) in wj.iter_mut().zip(&inv) {
                *x = inv_row[j];
            }
        }
        let lorentz = eta[0] < 0.0;
        // Normalise: timelike → <w,w> = -1, w0 > 0; lightlike → w0 = 1; spherical → unit length.
        for wj in w.iter_mut() {
            let n2 = form(&eta, wj, wj);
            let scale = if lorentz {
                let s = if wj[0] < 0.0 { -1.0 } else { 1.0 };
                if n2 < -1e-9 * wj[0] * wj[0] { s / (-n2).sqrt() } else { 1.0 / wj[0] }
            } else {
                1.0 / n2.sqrt()
            };
            for x in wj.iter_mut() {
                *x *= scale;
            }
        }
        if !lorentz {
            // Coxeter simplices are acute, so <w_j, w_0> >= 0 already; this only guards round-off.
            let w0 = w[0];
            for wj in w.iter_mut().skip(1) {
                if form(&eta, wj, &w0) < 0.0 {
                    *wj = wj.map(|x| -x);
                }
            }
        }
        Ok(Self { eta, normals, reflections, w })
    }
}

/// Lorentz boost mapping a future timelike vector `u` (any scale) to `(1, 0, 0, 0)`.
pub fn boost_to_origin(u: &Vec4) -> Mat4 {
    let eta = [-1.0, 1.0, 1.0, 1.0];
    let n = (-form(&eta, u, u)).sqrt();
    let u = u.map(|x| x / n);
    let g = u[0];
    let mut b = identity4();
    b[0][0] = g;
    for i in 1..4 {
        b[0][i] = -u[i];
        b[i][0] = -u[i];
        for j in 1..4 {
            b[i][j] += u[i] * u[j] / (1.0 + g);
        }
    }
    b
}
