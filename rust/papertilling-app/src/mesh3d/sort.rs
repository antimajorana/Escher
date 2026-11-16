//! Back-to-front order of the translucent face triangles for the current view rotation.

use super::Vertex3d;

/// View-space depth (z) of `p` under the column-major rotation `rot`; the camera looks down -z, so a smaller value
/// is farther away.
pub fn view_depth(rot: &[[f32; 4]; 4], p: [f32; 3]) -> f32 {
    rot[0][2] * p[0] + rot[1][2] * p[1] + rot[2][2] * p[2]
}

/// Vertex indices of the triangles `faces[start..]` (three consecutive vertices each), farthest triangle first, so
/// that alpha blending composes the translucent faces in the right order whatever their order in the buffer.
pub fn back_to_front_indices(faces: &[Vertex3d], start: usize, rot: &[[f32; 4]; 4]) -> Vec<u32> {
    let start = start.min(faces.len());
    let n_tri = (faces.len() - start) / 3;
    let mut keyed: Vec<(f32, u32)> = (0..n_tri)
        .map(|t| {
            let base = start + 3 * t;
            let z = (0..3).map(|k| view_depth(rot, faces[base + k].position)).sum::<f32>() / 3.0;
            (if z.is_finite() { z } else { f32::NEG_INFINITY }, base as u32)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut out = Vec::with_capacity(3 * n_tri);
    for (_, base) in keyed {
        out.extend([base, base + 1, base + 2]);
    }
    out
}
