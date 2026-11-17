//! L-BFGS optimizer and 3x3 symmetric eigensolver for `defect::sheet3d`.

/// Two-loop L-BFGS minimization of `eval_fg(x, grad) -> energy`.
/// Returns `(final_energy, iterations_used)`.
pub(super) fn lbfgs_minimize<F>(
    x: &mut [f64],
    max_iters: usize,
    grad_tol: f64,
    mut eval_fg: F,
) -> (f64, usize)
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x.len();
    if n == 0 || max_iters == 0 {
        let mut g = vec![0.0; n];
        return (eval_fg(x, &mut g), 0);
    }

    const M: usize = 32;
    let mut grad = vec![0.0; n];
    let mut f = eval_fg(x, &mut grad);
    if !f.is_finite() || grad.iter().any(|g| !g.is_finite()) {
        return (f, 0);
    }
    if rms_norm(&grad) <= grad_tol {
        return (f, 0);
    }

    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(M);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(M);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(M);
    let mut alpha = [0.0; M];

    let mut dir = vec![0.0; n];
    let mut q = vec![0.0; n];
    let mut x_try = vec![0.0; n];
    let mut g_try = vec![0.0; n];

    let mut iters = 0;
    for it in 0..max_iters {
        if rms_norm(&grad) <= grad_tol {
            break;
        }
        iters = it + 1;

        // Two-loop L-BFGS recursion for search direction dir = -H_k * grad.
        q.copy_from_slice(&grad);
        let k_len = s_hist.len();
        for i in (0..k_len).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &q);
            alpha[i] = a;
            axpy(&mut q, -a, &y_hist[i]);
        }
        let gamma = if let (Some(s_last), Some(y_last)) = (s_hist.last(), y_hist.last()) {
            let yy = dot(y_last, y_last);
            let sy = dot(s_last, y_last);
            if yy > 1e-24 && sy > 1e-24 {
                (sy / yy).clamp(1e-4, 1e4)
            } else {
                1.0
            }
        } else {
            let gn = norm(&grad);
            if gn > 1.0 { 1.0 / gn } else { 1.0 }
        };
        for i in 0..n {
            dir[i] = q[i] * gamma;
        }
        for i in 0..k_len {
            let b = rho_hist[i] * dot(&y_hist[i], &dir);
            axpy(&mut dir, alpha[i] - b, &s_hist[i]);
        }
        for d in &mut dir {
            *d = -*d;
        }

        let mut dg0 = dot(&dir, &grad);
        if dg0 >= -1e-20 || !dg0.is_finite() {
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            let gn = norm(&grad).max(1.0);
            for i in 0..n {
                dir[i] = -grad[i] / gn;
            }
            dg0 = dot(&dir, &grad);
            if dg0 >= 0.0 {
                break;
            }
        }

        // Backtracking Armijo line search; curvature is checked when adding an L-BFGS pair below.
        let mut step = 1.0_f64;
        let c1 = 1e-4;
        let mut accepted = false;
        for _ls in 0..36 {
            for i in 0..n {
                x_try[i] = x[i] + step * dir[i];
            }
            let f_try = eval_fg(&x_try, &mut g_try);
            let near_floor = (-dg0) <= 1e-13 * (1.0 + f.abs())
                && f_try <= f + 1e-14 * (1.0 + f.abs())
                && rms_norm(&g_try) < rms_norm(&grad);
            if f_try.is_finite()
                && g_try.iter().all(|g| g.is_finite())
                && x_try.iter().all(|x| x.is_finite())
                && (f_try <= f + c1 * step * dg0 || near_floor)
            {
                f = f_try;
                accepted = true;
                break;
            }
            step *= 0.5;
        }

        if !accepted {
            // Retry without quasi-Newton history; a second failure is not evidence of convergence.
            if !s_hist.is_empty() {
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                continue;
            }
            break;
        }

        let mut s_vec = vec![0.0; n];
        let mut y_vec = vec![0.0; n];
        let mut step_max = 0.0_f64;
        for i in 0..n {
            let si = x_try[i] - x[i];
            let yi = g_try[i] - grad[i];
            step_max = step_max.max(si.abs());
            s_vec[i] = si;
            y_vec[i] = yi;
        }
        x.copy_from_slice(&x_try);
        grad.copy_from_slice(&g_try);

        let sy = dot(&s_vec, &y_vec);
        let sn = dot(&s_vec, &s_vec);
        let yn = dot(&y_vec, &y_vec);
        if sy.is_finite() && sy > 1e-18 * (sn * yn).sqrt().max(1e-18) {
            if s_hist.len() == M {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / sy);
        }

        let step_tol = (1e-3 * grad_tol).min(1e-13);
        if step_max < step_tol && rms_norm(&grad) <= grad_tol {
            break;
        }
    }

    (f, iters)
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[inline]
fn rms_norm(a: &[f64]) -> f64 {
    if a.is_empty() {
        0.0
    } else {
        (dot(a, a) / (a.len() as f64)).sqrt()
    }
}

#[inline]
fn axpy(dst: &mut [f64], alpha: f64, x: &[f64]) {
    for (d, &xi) in dst.iter_mut().zip(x) {
        *d += alpha * xi;
    }
}

/// Center `xyz` at COM = (0, 0, 0) and apply the minimal SO(3) tilt rotation mapping the smallest-variance
/// principal normal to `+z`, preserving in-plane `(x, y)` azimuth.
pub(crate) fn center_and_align_horizontal(xyz: &mut [[f64; 3]], prefer_apex_up: Option<usize>) {
    let n = xyz.len();
    if n == 0 {
        return;
    }
    let inv_n = 1.0 / (n as f64);
    let mut com = [0.0; 3];
    for p in xyz.iter() {
        com[0] += p[0] * inv_n;
        com[1] += p[1] * inv_n;
        com[2] += p[2] * inv_n;
    }
    for p in xyz.iter_mut() {
        p[0] -= com[0];
        p[1] -= com[1];
        p[2] -= com[2];
    }
    if n < 3 {
        return;
    }

    let mut cov = [[0.0; 3]; 3];
    for p in xyz.iter() {
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += p[r] * p[c] * inv_n;
            }
        }
    }
    let (evals, evecs) = jacobi_eigen_3x3(cov);
    let mut min_idx = 0;
    if evals[1] < evals[min_idx] {
        min_idx = 1;
    }
    if evals[2] < evals[min_idx] {
        min_idx = 2;
    }
    let mut nrm = [evecs[0][min_idx], evecs[1][min_idx], evecs[2][min_idx]];
    let nlen = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
    if !nlen.is_finite() || nlen < 1e-12 {
        return;
    }
    nrm = [nrm[0] / nlen, nrm[1] / nlen, nrm[2] / nlen];
    if nrm[2] < 0.0 {
        nrm = [-nrm[0], -nrm[1], -nrm[2]];
    }

    // Minimal SO(3) rotation taking unit vector `nrm` (with nrm[2] >= 0) to `[0, 0, 1]`.
    // Using Rodrigues around `k = nrm x z_hat = [nrm[1], -nrm[0], 0]`:
    // R = I + [k]_x + [k]_x^2 / (1 + c), valid and smooth for all c = nrm[2] >= 0!
    let c = nrm[2];
    let denom = 1.0 + c;
    let (kx, ky) = (nrm[1], -nrm[0]);
    let rot = [
        [1.0 - ky * ky / denom, kx * ky / denom, ky],
        [kx * ky / denom, 1.0 - kx * kx / denom, -kx],
        [-ky, kx, c],
    ];
    for p in xyz.iter_mut() {
        let q = *p;
        p[0] = rot[0][0] * q[0] + rot[0][1] * q[1] + rot[0][2] * q[2];
        p[1] = rot[1][0] * q[0] + rot[1][1] * q[1] + rot[1][2] * q[2];
        p[2] = rot[2][0] * q[0] + rot[2][1] * q[1] + rot[2][2] * q[2];
    }

    // No reflection or ad-hoc apex flip: preserve chirality with the proper rotation above.
    let _ = prefer_apex_up;
}

/// Exact cyclic Jacobi eigensolver for a 3x3 real symmetric matrix.
/// Returns `(eigenvalues, eigenvectors_as_columns)`.
pub fn jacobi_eigen_3x3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..24 {
        let mut pq = (0usize, 1usize, a[0][1].abs());
        for (p, q) in [(0, 2), (1, 2)] {
            if a[p][q].abs() > pq.2 {
                pq = (p, q, a[p][q].abs());
            }
        }
        if pq.2 < 1e-15 {
            break;
        }
        let (p, q, _) = pq;
        let diff = a[q][q] - a[p][p];
        let t = if diff.abs() < 1e-30 {
            1.0
        } else {
            let tau = diff / (2.0 * a[p][q]);
            tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let app = a[p][p] - t * a[p][q];
        let aqq = a[q][q] + t * a[p][q];
        a[p][p] = app;
        a[q][q] = aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        let r = 3 - p - q;
        let apr = c * a[p][r] - s * a[q][r];
        let aqr = s * a[p][r] + c * a[q][r];
        a[p][r] = apr;
        a[r][p] = apr;
        a[q][r] = aqr;
        a[r][q] = aqr;
        for row in &mut v {
            let vp = c * row[p] - s * row[q];
            let vq = s * row[p] + c * row[q];
            row[p] = vp;
            row[q] = vq;
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sheet3d_known_tilted_plane_aligns_without_changing_metric_and_is_idempotent() {
        // Rotate the rectangle [-2,2] x [-1,1] by +30 degrees about y, then translate.
        // The inverse tilt must recover these exact planar coordinates, not a 60-degree tilt.
        let c = 3.0_f64.sqrt() / 2.0;
        let original = [[-2.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-2.0, 1.0]];
        let mut xyz: Vec<_> = original
            .iter()
            .map(|p| [7.0 + c * p[0], -3.0 + p[1], 4.0 - 0.5 * p[0]])
            .collect();
        center_and_align_horizontal(&mut xyz, None);
        for (p, expected) in xyz.iter().zip(original) {
            assert!((p[0] - expected[0]).abs() < 1e-12);
            assert!((p[1] - expected[1]).abs() < 1e-12);
            assert!(p[2].abs() < 1e-12);
        }
        let distance =
            |a: [f64; 3], b: [f64; 3]| (a[0] - b[0]).hypot(a[1] - b[1]).hypot(a[2] - b[2]);
        assert!((distance(xyz[0], xyz[1]) - 4.0).abs() < 1e-12);
        assert!((distance(xyz[0], xyz[2]) - 20.0_f64.sqrt()).abs() < 1e-12);
        let once = xyz.clone();
        center_and_align_horizontal(&mut xyz, None);
        assert!(
            xyz.iter()
                .zip(once)
                .all(|(a, b)| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-12))
        );
    }

    #[test]
    fn sheet3d_quadratic_optimizer_known_minimum_and_zero_budget() {
        let fg = |x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * (x[0] - 3.0);
            (x[0] - 3.0).powi(2)
        };
        let mut x = [0.0];
        let (energy, iters) = lbfgs_minimize(&mut x, 20, 1e-10, fg);
        assert!((x[0] - 3.0).abs() < 1e-10 && energy < 1e-20 && iters <= 20);
        let mut x = [0.0];
        assert_eq!(lbfgs_minimize(&mut x, 0, 1e-10, fg), (9.0, 0));
        assert_eq!(x, [0.0]);
    }
}
