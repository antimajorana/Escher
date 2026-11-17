//! Color maps for 3D elastic sheets, closed FvK surfaces, and spectral LDOS / charge overlays.

/// Signed relative strain color map (`blue` compressed, `slate` unstrained, `red` tensile).
pub fn strain_rgba(eps: f64, alpha: f32) -> [f32; 4] {
    let t = (eps / 0.08).clamp(-1.0, 1.0) as f32;
    let mid = [0.85_f32, 0.88, 0.92];
    if t >= 0.0 {
        let hi = [0.89_f32, 0.22, 0.17];
        [
            mid[0] + t * (hi[0] - mid[0]),
            mid[1] + t * (hi[1] - mid[1]),
            mid[2] + t * (hi[2] - mid[2]),
            alpha,
        ]
    } else {
        let s = -t;
        let lo = [0.14_f32, 0.42, 0.87];
        [
            mid[0] + s * (lo[0] - mid[0]),
            mid[1] + s * (lo[1] - mid[1]),
            mid[2] + s * (lo[2] - mid[2]),
            alpha,
        ]
    }
}

/// Smooth diverging height/buckling colormap.
pub fn height_rgba(z: f64, half_span: f64, alpha: f32) -> [f32; 4] {
    let u = ((z / half_span.max(1e-12)).clamp(-1.0, 1.0) * 0.5 + 0.5) as f32;
    let lo = [0.16_f32, 0.46, 0.78];
    let mid = [0.46_f32, 0.75, 0.70];
    let hi = [0.95_f32, 0.56, 0.20];
    if u < 0.5 {
        let t = u * 2.0;
        [
            lo[0] + t * (mid[0] - lo[0]),
            lo[1] + t * (mid[1] - lo[1]),
            lo[2] + t * (mid[2] - lo[2]),
            alpha,
        ]
    } else {
        let t = (u - 0.5) * 2.0;
        [
            mid[0] + t * (hi[0] - mid[0]),
            mid[1] + t * (hi[1] - mid[1]),
            mid[2] + t * (hi[2] - mid[2]),
            alpha,
        ]
    }
}

/// Sequential plasma-like colormap for nonnegative elastic energy density and LDOS `|psi|^2`.
pub fn sequential_heat_rgba(val: f64, vmax: f64, alpha: f32) -> [f32; 4] {
    let u = (val / vmax.max(1e-12)).clamp(0.0, 1.0) as f32;
    let c0 = [0.89_f32, 0.91, 0.94];
    let c1 = [0.22_f32, 0.56, 0.76];
    let c2 = [0.91_f32, 0.34, 0.19];
    let c3 = [0.99_f32, 0.85, 0.22];
    let (a, b, t) = if u < 0.35 {
        (c0, c1, u / 0.35)
    } else if u < 0.75 {
        (c1, c2, (u - 0.35) / 0.40)
    } else {
        (c2, c3, (u - 0.75) / 0.25)
    };
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]), alpha]
}

/// Diverging colormap for discrete Gaussian curvature `Omega_v` and excess spectral charge `delta_rho`.
pub fn diverging_rgba(val: f64, half_span: f64, alpha: f32) -> [f32; 4] {
    let u = (val / half_span.max(1e-12)).clamp(-1.0, 1.0) as f32;
    let mid = [0.89_f32, 0.90, 0.93];
    if u >= 0.0 {
        let hi = [0.84_f32, 0.19, 0.17];
        [mid[0] + u * (hi[0] - mid[0]), mid[1] + u * (hi[1] - mid[1]), mid[2] + u * (hi[2] - mid[2]), alpha]
    } else {
        let s = -u;
        let lo = [0.13_f32, 0.44, 0.82];
        [mid[0] + s * (lo[0] - mid[0]), mid[1] + s * (lo[1] - mid[1]), mid[2] + s * (lo[2] - mid[2]), alpha]
    }
}

pub fn rgba_u8(c: [f32; 4]) -> [u8; 4] {
    [
        (c[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}
