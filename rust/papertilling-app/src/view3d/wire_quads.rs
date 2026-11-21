//! Thick wires for the 3D PNG export. The live view draws the wires as a `LineList`, 1 physical pixel wide; an
//! export at `s` times the on-screen pixel density keeps their on-screen coverage only if they are `s` pixels wide.
//! GPUs draw lines 1 px wide only, so each segment is expanded here (CPU, no GPU state) into a screen-space quad of
//! `width_px` pixels with square caps (half the width past each end, which closes the joints of a polyline), given
//! in NDC with the clip depth, to be drawn with an identity MVP through the wire fragment shader.

use crate::mesh3d::Vertex3d;

type Mat4 = [[f32; 4]; 4];

/// Smallest clip `w` kept: points on or behind the eye plane are cut off (with wgpu's near plane `z = 0`).
const W_EPS: f32 = 1e-6;

/// Column-major `m * (p, 1)`.
fn to_clip(m: &Mat4, p: [f32; 3]) -> [f32; 4] {
    core::array::from_fn(|i| m[0][i] * p[0] + m[1][i] * p[1] + m[2][i] * p[2] + m[3][i])
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    core::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// The part of the clip-space segment `a b` in front of wgpu's near plane (`z >= 0`) and the eye (`w >= W_EPS`),
/// or `None` when nothing of it is.
pub(crate) fn clip_near(a: [f32; 4], b: [f32; 4]) -> Option<([f32; 4], [f32; 4])> {
    let planes: [fn([f32; 4]) -> f32; 2] = [|c| c[2], |c| c[3] - W_EPS];
    let (mut t0, mut t1) = (0.0_f32, 1.0_f32);
    for plane in planes {
        let (da, db) = (plane(a), plane(b));
        if da < 0.0 && db < 0.0 {
            return None;
        }
        if da < 0.0 {
            t0 = t0.max(da / (da - db));
        } else if db < 0.0 {
            t1 = t1.min(da / (da - db));
        }
    }
    (t0 <= t1).then(|| (lerp4(a, b, t0), lerp4(a, b, t1)))
}

/// Two triangles (6 vertices, NDC x / y and depth z) per wire segment `wires[2i], wires[2i + 1]` of `mvp`, each a
/// quad `width_px` pixels wide on a `size` pixel target with square caps; segments behind the camera or of zero
/// screen length are dropped. Normals and colours are the endpoints' own (the wire shader reads only the colour).
pub(crate) fn wire_quads(wires: &[Vertex3d], mvp: &Mat4, size: [u32; 2], width_px: f32) -> Vec<Vertex3d> {
    let (sw, sh) = (size[0].max(1) as f32, size[1].max(1) as f32);
    let mut out = Vec::with_capacity(wires.len() / 2 * 6);
    for &[va, vb] in wires.as_chunks::<2>().0 {
        let Some((ca, cb)) = clip_near(to_clip(mvp, va.position), to_clip(mvp, vb.position)) else {
            continue;
        };
        let (a, b) = ([ca[0] / ca[3], ca[1] / ca[3], ca[2] / ca[3]], [cb[0] / cb[3], cb[1] / cb[3], cb[2] / cb[3]]);
        // direction in pixels (NDC spans 2 units over the size)
        let d = [(b[0] - a[0]) * 0.5 * sw, (b[1] - a[1]) * 0.5 * sh];
        let len = d[0].hypot(d[1]);
        if !(len > 1e-6 && len.is_finite()) {
            continue;
        }
        let u = [d[0] / len, d[1] / len];
        // half the width (width / 2 px = width / size NDC) across the segment and, as caps, along it
        let n = [-u[1] * width_px / sw, u[0] * width_px / sh];
        let e = [u[0] * width_px / sw, u[1] * width_px / sh];
        let corner = |p: [f32; 3], se: f32, sn: f32, v: Vertex3d| Vertex3d {
            position: [p[0] + se * e[0] + sn * n[0], p[1] + se * e[1] + sn * n[1], p[2]],
            normal: v.normal,
            color: v.color,
        };
        let (a0, a1) = (corner(a, -1.0, -1.0, va), corner(a, -1.0, 1.0, va));
        let (b0, b1) = (corner(b, 1.0, -1.0, vb), corner(b, 1.0, 1.0, vb));
        out.extend([a0, a1, b0, a1, b1, b0]);
    }
    out
}

/// Wire width of an export in pixels: export pixels per on-screen physical pixel (`export_w / screen_w_px`), so a
/// wire keeps the coverage of the live view's 1 px line; `None` at or below 1 px, where the export draws the same
/// 1 px `LineList` as the screen.
pub(crate) fn export_wire_width(export_w: u32, screen_w_px: f32) -> Option<f32> {
    let width = export_w as f32 / screen_w_px.max(1.0);
    (width > 1.0 + 1e-3).then_some(width)
}

#[cfg(test)]
#[path = "wire_quads_tests.rs"]
mod tests;
