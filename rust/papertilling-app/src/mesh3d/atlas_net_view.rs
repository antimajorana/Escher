//! Interactive 3D crystal-net mesh of the Atlas space-group view: z-slice slab (hidden sites and their bonds left
//! out), unit-cell box and coordinate axes as extra wires, camera-dependent depth-cue alpha and a picked-site marker.
//!
//! The mesh is normalised on the *whole* net (`normalise(&net.positions)`, as `mesh_from_net3d`), so the slab and
//! the extra wires never move the picture and site `i` always sits at `(p_i - center) / radius`
//! ([`net_mesh_point`]), which is what 3D picking projects.

use papertilling_core::space3d::nets3d::Net3dPatch;

use super::geom::{append_icosahedron, append_octahedron, push_wire};
use super::sort::view_depth;
use super::{GpuMeshData, normalise};

type Mat4 = [[f32; 4]; 4];

/// Colour of the picked-site marker (3D) and ring (2D).
pub const PICK_RGBA: [f32; 4] = [1.0, 0.82, 0.08, 1.0];
/// Smallest depth-cue alpha (farthest site).
pub const DEPTH_CUE_MIN_ALPHA: f32 = 0.25;

/// What to draw on top of the net.
pub struct NetMeshView<'a> {
    /// `visible[i]`: site `i` is inside the slab (a bond is drawn only when both ends are).
    pub visible: &'a [bool],
    /// Extra wires in net (Cartesian) coordinates with their colours: unit-cell box edges, axes.
    pub extra_wires: Vec<([f64; 3], [f64; 3], [f32; 4])>,
    /// View rotation for the depth cue (`None`: depth cue off).
    pub depth_rot: Option<Mat4>,
    /// Picked site to mark.
    pub highlight: Option<usize>,
}

/// Mesh-space position of net point `p` for a mesh normalised with `(center, radius)`.
pub fn net_mesh_point(center: [f64; 3], radius: f64, p: [f64; 3]) -> [f32; 3] {
    core::array::from_fn(|k| ((p[k] - center[k]) / radius) as f32)
}

/// Mesh-space positions of all sites of `net` (the frame [`mesh_from_net_view`] draws them in).
pub fn net_mesh_points(net: &Net3dPatch) -> Vec<[f32; 3]> {
    let (center, radius) = normalise(&net.positions, &[]);
    net.positions.iter().map(|&p| net_mesh_point(center, radius, p)).collect()
}

/// Depth-cue alpha of a mesh-space point (the mesh lies in the unit ball): 1 at view depth +1 (nearest) down to
/// [`DEPTH_CUE_MIN_ALPHA`] at -1 (farthest).
pub fn depth_cue_alpha(rot: &Mat4, p: [f32; 3]) -> f32 {
    let t = (0.5 * (view_depth(rot, p) + 1.0)).clamp(0.0, 1.0);
    DEPTH_CUE_MIN_ALPHA + (1.0 - DEPTH_CUE_MIN_ALPHA) * t
}

fn site_rgba(kind: u8) -> [f32; 4] {
    match kind % 4 {
        0 => [0.22, 0.44, 0.78, 1.0],
        1 => [0.85, 0.36, 0.20, 1.0],
        2 => [0.18, 0.64, 0.44, 1.0],
        _ => [0.58, 0.36, 0.72, 1.0],
    }
}

pub fn mesh_from_net_view(net: &Net3dPatch, view: &NetMeshView) -> GpuMeshData {
    if net.positions.is_empty() {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }
    let (center, radius) = normalise(&net.positions, &[]);
    let pts: Vec<[f32; 3]> = net.positions.iter().map(|&p| net_mesh_point(center, radius, p)).collect();
    let shown = |i: usize| view.visible.get(i).copied().unwrap_or(true);
    let alpha = |p: [f32; 3]| view.depth_rot.as_ref().map_or(1.0, |r| depth_cue_alpha(r, p));

    let mut opaque = Vec::new();
    let mut faded = Vec::new();
    let mut wires = Vec::new();
    for (u, &p) in pts.iter().enumerate().filter(|(u, _)| shown(*u)) {
        let mut col = site_rgba(net.site_kinds.get(u).copied().unwrap_or(0));
        col[3] = alpha(p);
        // faded sites go after the opaque prefix, where the renderer sorts them back to front
        let out = if col[3] < 1.0 { &mut faded } else { &mut opaque };
        append_octahedron(out, p, 0.032, col);
    }
    if let Some(h) = view.highlight.filter(|&h| h < pts.len() && shown(h)) {
        append_icosahedron(&mut opaque, pts[h], 0.055, PICK_RGBA);
    }
    if let Some((a, b)) = net.defect_axis {
        let (pa, pb) = (net_mesh_point(center, radius, a), net_mesh_point(center, radius, b));
        append_icosahedron(&mut opaque, pa, 0.045, [0.95, 0.16, 0.16, 1.0]);
        append_icosahedron(&mut opaque, pb, 0.045, [0.95, 0.16, 0.16, 1.0]);
        push_wire(&mut wires, pa, pb, [0.95, 0.16, 0.16, 1.0]);
    }
    let mut drawn_bonds = 0;
    for &(u, v, kind) in &net.bonds {
        if u < pts.len() && v < pts.len() && shown(u) && shown(v) {
            let mut col = if kind == 1 { [0.95, 0.32, 0.14, 1.0] } else { [0.18, 0.24, 0.34, 0.92] };
            col[3] *= alpha(pts[u]).min(alpha(pts[v]));
            push_wire(&mut wires, pts[u], pts[v], col);
            drawn_bonds += 1;
        }
    }
    for &(a, b, col) in &view.extra_wires {
        push_wire(&mut wires, net_mesh_point(center, radius, a), net_mesh_point(center, radius, b), col);
    }
    let opaque_vertex_count = opaque.len();
    opaque.extend(faded);
    GpuMeshData {
        faces: opaque,
        wires,
        center: center.map(|x| x as f32),
        radius: radius as f32,
        vertex_count: (0..pts.len()).filter(|&i| shown(i)).count(),
        edge_count: drawn_bonds,
        face_count: 0,
        opaque_vertex_count,
    }
}
