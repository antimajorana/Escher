//! An accurate realisation of `[p,q,r]` from the LDLᵀ factorisation of its Gram matrix
//! (used by the growth adapter `growth::h3_complex`; plan §2.3).
//!
//! Why not [`CoxeterRealisation::new`]: it takes the normals from a Jacobi eigen-decomposition
//! and then applies a numerical boost, so the Coxeter relations (R_i R_j)^{m_ij} = 1 hold only
//! to round-off of those steps. In H³ two words for the same element then disagree by that
//! error times e^{2ρ} at distance ρ, which at touching-rule depth 3 reached the merge tolerance.
//!
//! Here the Gram matrix G (G_ii = 1, G_{i,i+1} = −cos(π/m_i)) is factored as G = L D Lᵀ with
//! L unit lower bidiagonal: d₀ = 1, l_{i+1,i} = G_{i,i+1}/d_i, d_{i+1} = 1 − l²_{i+1,i} d_i.
//! In an orthonormal basis a₀..a₃ with signs sign(d_k), the normals n_j = Σ_k L_jk √|d_k| a_k
//! satisfy ⟨n_i, n_j⟩ = (L D Lᵀ)_ij = G_ij [PROVED, by expansion]. The pivots d₀, d₁, d₂ are
//! those of the cell group [p,q], positive exactly when the cell {p,q} is spherical, so this
//! works for S³, compact H³ and H³ with ideal vertices; it fails (d₂ = 0) for infinite cells,
//! which the growth adapter refuses before calling it. d₃ has the sign of det G.
//!
//! n₀, n₁, n₂ have no a₃ component, so the cell centre w₃ (⊥ n₀, n₁, n₂) lies on the a₃ axis:
//! in coordinates (a₃, a₀, a₁, a₂) the central cell is centred at (1, 0, 0, 0) exactly, the
//! origin of the ball in H³, with no boost.

use super::coxeter_gram::{form, identity4, inverse4, CoxeterRealisation, Mat4, Vec4};

/// `[p,q,r]` with finite cell group (`(p-2)(q-2) < 4`), time coordinate 0 when det G < 0.
pub fn ldl_realisation(p: u32, q: u32, r: u32) -> Result<CoxeterRealisation, String> {
    let c = [p, q, r].map(|m| -(std::f64::consts::PI / f64::from(m)).cos());
    let mut d = [1.0f64; 4];
    let mut l = [0.0f64; 3];
    for i in 0..3 {
        l[i] = c[i] / d[i];
        d[i + 1] = 1.0 - l[i] * l[i] * d[i];
    }
    if d[..3].iter().any(|&x| x <= 1e-12) {
        return Err(format!("[{p},{q},{r}]: the cell group [{p},{q}] is not finite, no LDLᵀ realisation"));
    }
    if d[3].abs() <= 1e-12 {
        return Err(format!("[{p},{q},{r}] is Euclidean (det G = 0)"));
    }
    let lorentz = d[3] < 0.0;
    let s: [f64; 4] = d.map(|x| x.abs().sqrt());
    // coordinates (a3, a0, a1, a2)
    let mut normals = [[0.0f64; 4]; 4];
    for (j, nj) in normals.iter_mut().enumerate() {
        let mut a = [0.0f64; 4];
        a[j] = s[j];
        if j > 0 {
            a[j - 1] = l[j - 1] * s[j - 1];
        }
        *nj = [a[3], a[0], a[1], a[2]];
    }
    let eta: Vec4 = if lorentz { [-1.0, 1.0, 1.0, 1.0] } else { [1.0; 4] };
    let reflections = normals.map(|n| {
        let mut m = identity4();
        for (a, row) in m.iter_mut().enumerate() {
            for (b, x) in row.iter_mut().enumerate() {
                *x -= 2.0 * n[a] * eta[b] * n[b];
            }
        }
        m
    });
    let ne: Mat4 = normals.map(|n| [n[0] * eta[0], n[1] * eta[1], n[2] * eta[2], n[3] * eta[3]]);
    let inv = inverse4(&ne).ok_or("singular mirror system")?;
    let mut w = [[0.0f64; 4]; 4];
    for (j, wj) in w.iter_mut().enumerate() {
        for (x, row) in wj.iter_mut().zip(&inv) {
            *x = row[j];
        }
        // positive scale only: w_j stays in the chamber {⟨x, n_k⟩ ≥ 0}
        let n2 = form(&eta, wj, wj);
        let scale = if !lorentz {
            1.0 / n2.sqrt()
        } else if n2 < -1e-9 * wj[0] * wj[0] {
            1.0 / (-n2).sqrt()
        } else {
            1.0 / wj[0].abs()
        };
        *wj = wj.map(|x| x * scale);
    }
    if lorentz && w[3][0] < 0.0 {
        w = w.map(|wj| wj.map(|x| -x));
    }
    // w3 is the a3 axis exactly (n0, n1, n2 have no a3 component).
    w[3] = if lorentz || w[3][0] > 0.0 { [1.0, 0.0, 0.0, 0.0] } else { [-1.0, 0.0, 0.0, 0.0] };
    Ok(CoxeterRealisation { eta, normals, reflections, w })
}
