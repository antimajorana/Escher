//! Projective plane drawn once (not its double cover): every face at its lift on the antipodal double cover whose
//! centre is in the upper half (z >= 0), i.e. the hemisphere model of RP2 with its rim glued antipodally. Faces are
//! continuous spherical patches; a site on the rim may be drawn at both of its lifts.

use std::collections::HashSet;

use papertilling_core::types::{SurfaceCover3D, SurfaceMesh};

use super::geom::{
    append_icosahedron, append_octahedron, arc_subdivisions, cone_rgba, dist3, push_wire, sph_arc, tint_towards,
    tri_normal,
};
use super::{GpuMeshData, Vertex3d, face_size_rgba};

fn centroid_dir(xyz: &[[f64; 3]], cyc: &[usize]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for &u in cyc {
        for k in 0..3 {
            c[k] += xyz[u][k];
        }
    }
    c
}

/// Which lift of each quotient face is drawn: index into `cov.faces` (lifts `2 f` and `2 f + 1` are antipodal).
pub(super) fn upper_lifts(surf: &SurfaceMesh, cov: &SurfaceCover3D) -> Option<Vec<usize>> {
    let n = surf.sites.len();
    if cov.xyz.len() != 2 * n || cov.faces.len() != 2 * surf.faces.len() {
        return None;
    }
    if cov.faces.iter().flatten().any(|&u| u >= cov.xyz.len()) {
        return None;
    }
    let key = |c: [f64; 3]| [c[2], c[1], c[0]];
    Some(
        (0..surf.faces.len())
            .map(|f| {
                let (a, b) = (key(centroid_dir(&cov.xyz, &cov.faces[2 * f])), key(centroid_dir(&cov.xyz, &cov.faces[2 * f + 1])));
                let pick_b = a
                    .iter()
                    .zip(&b)
                    .find(|(x, y)| (**x - **y).abs() > 1e-9)
                    .is_some_and(|(x, y)| y > x);
                2 * f + usize::from(pick_b)
            })
            .collect(),
    )
}

pub(super) fn rp2_hemisphere_mesh(surf: &SurfaceMesh, cov: &SurfaceCover3D) -> Option<GpuMeshData> {
    let n = surf.sites.len();
    let chosen = upper_lifts(surf, cov)?;
    let raw: Vec<[f32; 3]> = cov.xyz.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
    let mut used = vec![false; raw.len()];
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();
    for &lf in &chosen {
        let cyc = &cov.faces[lf];
        for m in 0..cyc.len() {
            let (a, b) = (cyc[m], cyc[(m + 1) % cyc.len()]);
            used[a] = true;
            if seen_edges.insert((a.min(b), a.max(b))) {
                edges.push((a, b));
            }
        }
    }
    for s in 0..n {
        if !used[s] && !used[s + n] {
            used[s] = true;
        }
    }

    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    let mut probe: Vec<[f32; 3]> = (0..raw.len()).filter(|&u| used[u]).map(|u| raw[u]).collect();
    probe.extend(edges.iter().map(|&(a, b)| sph_arc(raw[a], raw[b], 0.5)));
    for p in &probe {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let centre: [f32; 3] = core::array::from_fn(|k| 0.5 * (lo[k] + hi[k]));
    let radius = probe.iter().map(|&p| dist3(p, centre)).fold(1e-6_f32, f32::max);
    let norm = |p: [f32; 3]| -> [f32; 3] { core::array::from_fn(|k| (p[k] - centre[k]) / radius) };

    let chords: Vec<f32> = edges.iter().map(|&(a, b)| dist3(raw[a], raw[b]) / radius).collect();
    let avg = if chords.is_empty() { 0.22 } else { (chords.iter().sum::<f32>() / chords.len() as f32).clamp(0.06, 0.55) };
    let (site_r, cone_r) = ((avg * 0.095).clamp(0.013, 0.034), (avg * 0.19).clamp(0.028, 0.064));

    let mut site_cone: Vec<Option<f64>> = vec![None; n];
    let mut face_cone: Vec<Option<f64>> = vec![None; surf.faces.len()];
    for c in &surf.cones {
        if let Some(s) = c.site.filter(|&s| s < n) {
            site_cone[s] = Some(c.angle_deg);
        }
        if let Some(f) = c.face.filter(|&f| f < surf.faces.len()) {
            face_cone[f] = Some(c.angle_deg);
        }
    }

    let mut faces = Vec::new();
    for u in (0..raw.len()).filter(|&u| used[u]) {
        match site_cone[u % n] {
            Some(ang) => append_icosahedron(&mut faces, norm(raw[u]), cone_r, cone_rgba(ang)),
            None => append_octahedron(&mut faces, norm(raw[u]), site_r, [0.20, 0.25, 0.33, 1.0]),
        }
    }
    let face_centre = |cyc: &[usize]| {
        let c = centroid_dir(&cov.xyz, cyc);
        sph_arc([c[0] as f32, c[1] as f32, c[2] as f32], [c[0] as f32, c[1] as f32, c[2] as f32], 0.0)
    };
    for (f, &lf) in chosen.iter().enumerate() {
        if let Some(ang) = face_cone[f] {
            append_icosahedron(&mut faces, norm(face_centre(&cov.faces[lf])), cone_r, cone_rgba(ang));
        }
    }
    let opaque_vertex_count = faces.len();

    for (f, &lf) in chosen.iter().enumerate() {
        let cyc = &cov.faces[lf];
        if cyc.len() < 3 {
            continue;
        }
        let mut color = face_size_rgba(cyc.len(), false);
        if let Some(ang) = face_cone[f] {
            color = tint_towards(color, cone_rgba(ang), 0.22);
        }
        let c = norm(face_centre(cyc));
        for m in 0..cyc.len() {
            let (a, b) = (raw[cyc[m]], raw[cyc[(m + 1) % cyc.len()]]);
            let k = arc_subdivisions(dist3(a, b) / radius);
            for s in 0..k {
                let (p0, p1) = (norm(sph_arc(a, b, s as f32 / k as f32)), norm(sph_arc(a, b, (s + 1) as f32 / k as f32)));
                let nrm = tri_normal(c, p0, p1);
                for p in [c, p0, p1] {
                    faces.push(Vertex3d { position: p, normal: nrm, color });
                }
            }
        }
    }

    let mut wires = Vec::new();
    for &(a, b) in &edges {
        let k = arc_subdivisions(dist3(raw[a], raw[b]) / radius);
        for s in 0..k {
            let (p0, p1) = (sph_arc(raw[a], raw[b], s as f32 / k as f32), sph_arc(raw[a], raw[b], (s + 1) as f32 / k as f32));
            push_wire(&mut wires, norm(p0), norm(p1), [0.91, 0.95, 0.99, 0.96]);
        }
    }

    Some(GpuMeshData {
        faces,
        wires,
        center: centre,
        radius,
        vertex_count: n,
        edge_count: surf.bonds.len(),
        face_count: surf.faces.len(),
        opaque_vertex_count,
    })
}
