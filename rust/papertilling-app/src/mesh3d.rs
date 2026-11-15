#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex3d {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

pub fn build_icosahedron_mesh() -> (Vec<Vertex3d>, Vec<Vertex3d>) {
    let phi = 0.5 * (1.0 + 5.0_f32.sqrt());
    let s = 1.0 / (1.0 + phi * phi).sqrt();
    let verts = [
        [-s, phi * s, 0.0],
        [s, phi * s, 0.0],
        [-s, -phi * s, 0.0],
        [s, -phi * s, 0.0],
        [0.0, -s, phi * s],
        [0.0, s, phi * s],
        [0.0, -s, -phi * s],
        [0.0, s, -phi * s],
        [phi * s, 0.0, -s],
        [phi * s, 0.0, s],
        [-phi * s, 0.0, -s],
        [-phi * s, 0.0, s],
    ];
    let tris: [[usize; 3]; 20] = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    let palette = [
        [0.29, 0.62, 0.85],
        [0.36, 0.74, 0.68],
        [0.86, 0.58, 0.34],
        [0.64, 0.56, 0.86],
    ];
    let mut face_vertices = Vec::with_capacity(60);
    let mut edges = Vec::with_capacity(30);
    for (idx, [i, j, k]) in tris.into_iter().enumerate() {
        let (a, b, c) = (verts[i], verts[j], verts[k]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let nx = u[1] * v[2] - u[2] * v[1];
        let ny = u[2] * v[0] - u[0] * v[2];
        let nz = u[0] * v[1] - u[1] * v[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-6);
        let normal = [nx / len, ny / len, nz / len];
        let color = palette[idx % palette.len()];
        for position in [a, b, c] {
            face_vertices.push(Vertex3d { position, normal, color });
        }
        for (p, q) in [(i, j), (j, k), (k, i)] {
            let edge = if p < q { (p, q) } else { (q, p) };
            if !edges.contains(&edge) {
                edges.push(edge);
            }
        }
    }
    let wire_color = [0.92, 0.95, 0.98];
    let mut wire_vertices = Vec::with_capacity(edges.len() * 2);
    for (p, q) in edges {
        for position in [verts[p], verts[q]] {
            wire_vertices.push(Vertex3d {
                position,
                normal: [0.0, 0.0, 1.0],
                color: wire_color,
            });
        }
    }
    (face_vertices, wire_vertices)
}

pub fn orbit_rotation(yaw: f32, pitch: f32) -> [[f32; 4]; 4] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let ry = [
        [cy, 0.0, -sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let rx = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cp, sp, 0.0],
        [0.0, -sp, cp, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    mat4_mul(&rx, &ry)
}

pub fn build_mvp(aspect: f32, distance: f32, rot: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let f = 1.0 / (0.5 * 45.0_f32.to_radians()).tan();
    let (near, far) = (0.2_f32, 25.0_f32);
    let proj = [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, (near * far) / (near - far), 0.0],
    ];
    let mut view_model = *rot;
    view_model[3][2] = -distance;
    mat4_mul(&proj, &view_model)
}

fn mat4_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosahedron_has_euler_characteristic_two() {
        let (faces, wires) = build_icosahedron_mesh();
        assert_eq!(faces.len(), 20 * 3);
        assert_eq!(wires.len(), 30 * 2);
        let (v, e, f) = (12_i32, (wires.len() / 2) as i32, (faces.len() / 3) as i32);
        assert_eq!(v - e + f, 2);
    }

    #[test]
    fn perspective_maps_origin_inside_wgpu_unit_depth() {
        let rot = orbit_rotation(0.4, -0.25);
        let mvp = build_mvp(1.5, 3.4, &rot);
        let ndc_z = mvp[3][2] / mvp[3][3];
        assert!((0.0..=1.0).contains(&ndc_z));
    }
}
