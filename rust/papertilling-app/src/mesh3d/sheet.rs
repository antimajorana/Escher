//! 3D GPU mesh builder for relaxed elastic defect sheets (`Defects` and `Combine` tabs).

use std::collections::HashSet;

use papertilling_core::defect::sheet3d::{Sheet3DResult, validate_sheet_crystal};
use papertilling_core::types::BuiltCrystal;

use crate::scene2d::polygon::{cross, triangulate};

use super::color_scales::SheetColorScales;
use super::geom::{
    append_icosahedron, append_octahedron, blend_normal, compute_smooth_vertex_normals, cone_rgba,
    mean_chord_len, polygon_centroid_idx, push_wire, tri_normal,
};
use super::sheet_colors::{diverging_rgba, height_rgba, sequential_heat_rgba, strain_rgba};
pub use super::sheet_modes::{
    DefectViewMode, Sheet3dColorMode, Sheet3dViewOpts, sheet3d_caption,
};
use super::{GpuMeshData, Vertex3d, face_size_rgba};

#[cfg(test)]
pub fn mesh_from_sheet3d(
    crystal: &BuiltCrystal,
    sheet: &Sheet3DResult,
    opts: &Sheet3dViewOpts,
) -> GpuMeshData {
    mesh_from_sheet3d_with_spectral(crystal, sheet, opts, None)
}

pub fn mesh_from_sheet3d_with_spectral(
    crystal: &BuiltCrystal,
    sheet: &Sheet3DResult,
    opts: &Sheet3dViewOpts,
    spectral_scalar: Option<(&[f64], bool)>,
) -> GpuMeshData {
    let n = sheet.xyz.len();
    if n == 0 || !renderable(crystal, sheet, opts) {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }
    let exag = f64::from(opts.z_exag.clamp(1.0, 3.0));
    let mapped: Vec<[f64; 3]> = sheet.xyz.iter().map(|p| [p[0], p[2] * exag, -p[1]]).collect();
    let r_horiz = sheet.xyz.iter().map(|p| p[0].hypot(p[1])).fold(0.0_f64, f64::max).max(1e-6);
    let r_3d = mapped.iter().map(|p| p[0].hypot(p[1]).hypot(p[2])).fold(r_horiz, f64::max);
    if !r_3d.is_finite() || r_3d > f64::from(f32::MAX) {
        return GpuMeshData { radius: 1.0, ..Default::default() };
    }

    let pts: Vec<[f32; 3]> = mapped
        .iter()
        .map(|p| [(p[0] / r_3d) as f32, (p[1] / r_3d) as f32, (p[2] / r_3d) as f32])
        .collect();
    let avg_bond = mean_chord_len(&pts, crystal.edges.iter().copied());
    let site_r = (avg_bond * 0.095).clamp(0.011, 0.032);
    let cone_r = (avg_bond * 0.185).clamp(0.024, 0.058);

    // One source for every colour normalisation (shared with the 3D colorbar legend).
    let SheetColorScales {
        site_tot_e,
        bond_tot_e,
        site_shear,
        omega_span,
        e_site_scale,
        e_bond_scale,
        sh_site_scale,
        sh_bond_scale,
        spec_vmax,
        z_half_span,
    } = SheetColorScales::new(crystal, sheet, spectral_scalar);

    let site_vis_rgba = |u: usize, fallback: [f32; 4]| -> [f32; 4] {
        match (opts.color_mode, spectral_scalar) {
            (Sheet3dColorMode::SpectralLdos | Sheet3dColorMode::SpectralCharge, Some((vals, div))) => {
                let x = vals.get(u).copied().unwrap_or(0.0);
                if div { diverging_rgba(x, spec_vmax, fallback[3]) } else { sequential_heat_rgba(x.max(0.0), spec_vmax, fallback[3]) }
            }
            (Sheet3dColorMode::ElasticEnergyGsv, _) if !sheet.site_gsv_energy.is_empty() => {
                let t = (site_tot_e[u].max(0.0) / e_site_scale).clamp(0.0, 1.0).sqrt();
                sequential_heat_rgba(t, 1.0, fallback[3])
            }
            (Sheet3dColorMode::BondShearStrain, _) if !sheet.bond_shear_strain.is_empty() => {
                let t = (site_shear[u].max(0.0) / sh_site_scale).clamp(0.0, 1.0).sqrt();
                sequential_heat_rgba(t, 1.0, fallback[3])
            }
            (Sheet3dColorMode::GaussianCurvature, _) => {
                diverging_rgba(sheet.vertex_omega.get(u).copied().unwrap_or(0.0), omega_span, fallback[3])
            }
            _ => fallback,
        }
    };

    let cone_set: HashSet<usize> = crystal.cone_sites.iter().copied().collect();
    let mut face_vertices = Vec::new();
    let mut wire_vertices = Vec::new();

    for (u, &pos) in pts.iter().enumerate() {
        let omega = sheet.vertex_omega.get(u).copied().unwrap_or(0.0);
        if (cone_set.contains(&u) || omega.abs() > 0.20)
            && !matches!(
                opts.color_mode,
                Sheet3dColorMode::SpectralLdos | Sheet3dColorMode::SpectralCharge | Sheet3dColorMode::GaussianCurvature
            )
        {
            let deg = if omega.abs() > 0.05 { omega.to_degrees() } else { 60.0 };
            append_icosahedron(&mut face_vertices, pos, cone_r, cone_rgba(deg));
        } else {
            let def_col = match crystal.side.get(u).copied().flatten() {
                Some('A') => [0.13, 0.42, 0.77, 1.0],
                Some('B') => [0.10, 0.62, 0.45, 1.0],
                _ => [0.20, 0.25, 0.33, 1.0],
            };
            append_octahedron(&mut face_vertices, pos, site_r, site_vis_rgba(u, def_col));
        }
    }
    let opaque_vertex_count = face_vertices.len();

    let orientable = !crystal.is_non_orientable;
    let alpha = if orientable { 0.88 } else { 0.74 };
    let smooth_normals = compute_smooth_vertex_normals(&pts, &sheet.faces);

    for (fi, face) in sheet.faces.iter().enumerate() {
        if face.len() < 3 || face.iter().any(|&u| u >= n) {
            continue;
        }
        let Some(centroid) = polygon_centroid_idx(&pts, face) else {
            continue;
        };
        let f_strain = sheet.face_strain.get(fi).copied().unwrap_or(0.0);
        let cz_sheet = face.iter().map(|&u| sheet.xyz[u][2]).sum::<f64>() / (face.len() as f64);

        let face_col = match opts.color_mode {
            Sheet3dColorMode::FaceRingSize => face_size_rgba(face.len(), orientable),
            Sheet3dColorMode::ElasticEnergyGsv if sheet.site_gsv_energy.is_empty() => {
                sequential_heat_rgba(f_strain * f_strain * 16.0, 0.15, alpha)
            }
            Sheet3dColorMode::BondStrain => strain_rgba(f_strain, alpha),
            Sheet3dColorMode::HeightZ => height_rgba(cz_sheet, z_half_span, alpha),
            _ => {
                let mut mean_c = [0.0_f32; 4];
                for &u in face {
                    let cu = site_vis_rgba(u, face_size_rgba(face.len(), orientable));
                    for ch in 0..4 { mean_c[ch] += cu[ch] / (face.len() as f32); }
                }
                mean_c[3] = alpha;
                mean_c
            }
        };

        let vertex = |u: usize, normal| {
            let color = match opts.color_mode {
                Sheet3dColorMode::HeightZ => height_rgba(sheet.xyz[u][2], z_half_span, alpha),
                Sheet3dColorMode::ElasticEnergyGsv if sheet.site_gsv_energy.is_empty() => face_col,
                Sheet3dColorMode::ElasticEnergyGsv
                | Sheet3dColorMode::BondShearStrain
                | Sheet3dColorMode::GaussianCurvature
                | Sheet3dColorMode::SpectralLdos
                | Sheet3dColorMode::SpectralCharge => site_vis_rgba(u, face_col),
                _ => face_col,
            };
            Vertex3d { position: pts[u], normal: blend_normal(normal, smooth_normals[u], 0.28), color }
        };

        match sheet_face_fill(&pts, face, centroid) {
            SheetFaceFill::CentroidFan => {
                for m in 0..face.len() {
                    let u0 = face[m];
                    let u1 = face[(m + 1) % face.len()];
                    let normal = tri_normal(centroid, pts[u0], pts[u1]);
                    face_vertices.extend([
                        Vertex3d { position: centroid, normal, color: face_col },
                        vertex(u0, normal),
                        vertex(u1, normal),
                    ]);
                }
            }
            SheetFaceFill::Triangles(triangles) => {
                for [u, v, w] in triangles {
                    let normal = tri_normal(pts[u], pts[v], pts[w]);
                    face_vertices.extend([vertex(u, normal), vertex(v, normal), vertex(w, normal)]);
                }
            }
            SheetFaceFill::OutlineOnly => {}
        }
    }

    for (e_idx, &(u, v)) in crystal.edges.iter().enumerate() {
        let across = crystal.adj.get(u).is_some_and(|r| r.iter().any(|d| d.to == v && d.across));
        let wc = if opts.color_mode == Sheet3dColorMode::BondStrain {
            strain_rgba(sheet.bond_strain.get(e_idx).copied().unwrap_or(0.0), 0.98)
        } else if opts.color_mode == Sheet3dColorMode::BondShearStrain && !sheet.bond_shear_strain.is_empty() {
            let t = (sheet.bond_shear_strain[e_idx].abs() / sh_bond_scale).clamp(0.0, 1.0).sqrt();
            sequential_heat_rgba(t, 1.0, 0.98)
        } else if opts.color_mode == Sheet3dColorMode::ElasticEnergyGsv && !bond_tot_e.is_empty() {
            let t = (bond_tot_e[e_idx].max(0.0) / e_bond_scale).clamp(0.0, 1.0).sqrt();
            sequential_heat_rgba(t, 1.0, 0.98)
        } else if across {
            [0.95, 0.45, 0.14, 0.98]
        } else {
            [0.91, 0.95, 0.99, 0.96]
        };
        push_wire(&mut wire_vertices, pts[u], pts[v], wc);
    }

    GpuMeshData {
        faces: face_vertices,
        wires: wire_vertices,
        center: [0.0; 3],
        radius: r_3d as f32,
        vertex_count: n,
        edge_count: crystal.edges.len(),
        face_count: sheet.faces.len(),
        opaque_vertex_count,
    }
}

enum SheetFaceFill {
    CentroidFan,
    Triangles(Vec<[usize; 3]>),
    OutlineOnly,
}

fn sheet_face_fill(pts: &[[f32; 3]], face: &[usize], centroid: [f32; 3]) -> SheetFaceFill {
    // Try the projection with the most area first, including for tilted and vertical faces.
    let project = |p: [f32; 3], axis| [f64::from(p[(axis + 1) % 3]), f64::from(p[(axis + 2) % 3])];
    let origin = pts[face[0]];
    let mut area = [0.0_f64; 3];
    for (&u, &v) in face.iter().zip(face.iter().cycle().skip(1)) {
        for (axis, a) in area.iter_mut().enumerate() {
            *a += cross(project(origin, axis), project(pts[u], axis), project(pts[v], axis));
        }
    }
    let mut axes = [0, 1, 2];
    axes.sort_by(|&a, &b| area[b].abs().total_cmp(&area[a].abs()));
    let Some((axis, outline)) = axes.into_iter().filter(|&axis| area[axis].abs() > 1e-12).find_map(|axis| {
        let outline: Vec<_> = face.iter().map(|&u| project(pts[u], axis)).collect();
        is_simple_projection(&outline).then_some((axis, outline))
    }) else {
        return SheetFaceFill::OutlineOnly;
    };
    let p = project(centroid, axis);
    let sense = area[axis].signum();
    if outline.iter().zip(outline.iter().cycle().skip(1)).all(|(&a, &b)| cross(p, a, b) * sense >= 0.0) {
        return SheetFaceFill::CentroidFan;
    }

    let triangles = triangulate(&outline);
    let mut result = Vec::with_capacity(triangles.len());
    for [a, b, c] in triangles {
        // The shared triangulator may fall back to a fan for degenerate input.
        // Reject reversed triangles and keep the original outline visible.
        if cross(a, b, c) < 0.0 {
            return SheetFaceFill::OutlineOnly;
        }
        let index = |p| outline.iter().position(|&q| q == p).map(|i| face[i]);
        let (Some(u), Some(v), Some(w)) = (index(a), index(b), index(c)) else {
            return SheetFaceFill::OutlineOnly;
        };
        result.push(if sense > 0.0 { [u, v, w] } else { [u, w, v] });
    }
    SheetFaceFill::Triangles(result)
}

fn is_simple_projection(outline: &[[f64; 2]]) -> bool {
    let n = outline.len();
    for i in 0..n {
        let after_i = (i + 1) % n;
        let (a, b) = (outline[i], outline[after_i]);
        if a == b {
            return false;
        }
        for j in i + 2..n {
            let after_j = (j + 1) % n;
            if after_j == i {
                continue;
            }
            let (c, d) = (outline[j], outline[after_j]);
            let (abc, abd, cda, cdb) = (cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b));
            let on = |p: [f64; 2], u: [f64; 2], v: [f64; 2]| {
                (0..2).all(|k| (u[k].min(v[k])..=u[k].max(v[k])).contains(&p[k]))
            };
            if (abc * abd < 0.0 && cda * cdb < 0.0)
                || (abc == 0.0 && on(c, a, b))
                || (abd == 0.0 && on(d, a, b))
                || (cda == 0.0 && on(a, c, d))
                || (cdb == 0.0 && on(b, c, d))
            {
                return false;
            }
        }
    }
    true
}

pub(super) fn renderable(crystal: &BuiltCrystal, sheet: &Sheet3DResult, opts: &Sheet3dViewOpts) -> bool {
    validate_sheet_crystal(crystal).is_ok()
        && sheet.has_finite_geometry()
        && sheet.xyz.len() == crystal.pos.len()
        && sheet.bond_strain.len() == crystal.edges.len()
        && sheet.bond_rest_len.len() == crystal.edges.len()
        && opts.z_exag.is_finite()
        && opts.kappa_b.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;
    use papertilling_core::defect::sheet3d::{Sheet3DOptions, relax_sheet_3d};
    use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
    use papertilling_core::plane::catalogue::parse_named_str;

    fn square_sheet() -> (BuiltCrystal, Sheet3DResult) {
        let named = parse_named_str(include_str!("../../../../papertilling/data/named.json")).unwrap();
        let cell = named.named.into_iter().find(|n| n.key == "square").unwrap().cell;
        let crystal = build_point_defects(&cell, &[DefectSpec::disclination(90.0, [0.0, 0.0])], Some(3.5), None).unwrap();
        let sheet = relax_sheet_3d(&crystal, &cell, &Sheet3DOptions { max_iters: 2, ..Default::default() }, None);
        (crystal, sheet)
    }

    #[test]
    fn concave_sheet_face_does_not_fill_its_notch() {
        let (crystal, mut sheet) = square_sheet();
        // The L consists of two 3 × 1 arms overlapping in one square. Tilt it far enough
        // to make its best projection lie outside the horizontal plane.
        let ring = [[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0], [1.0, 3.0], [0.0, 3.0]];
        for (p, [x, y]) in sheet.xyz.iter_mut().zip(ring) {
            *p = [x, y, 2.0 * x];
        }
        sheet.faces = vec![(0..ring.len()).collect()];
        sheet.face_strain = vec![0.0];

        for _ in 0..2 {
            let mesh = mesh_from_sheet3d(&crystal, &sheet, &Sheet3dViewOpts::default());
            let vertices = &mesh.faces[mesh.opaque_vertex_count..];
            assert_eq!(vertices.len(), 12, "a six-sided face uses four triangles");
            let mut area = 0.0;
            let (triangles, _) = vertices.as_chunks::<3>();
            for t in triangles {
                let original_xy = |i: usize| {
                    let v = t[i].position;
                    [f64::from(v[0] * mesh.radius), f64::from(-v[2] * mesh.radius)]
                };
                let (a, b, c) = (original_xy(0), original_xy(1), original_xy(2));
                let (x, y) = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0);
                assert!(x < 1.0 || y < 1.0, "triangle center {x}, {y} is in the notch");
                area += cross(a, b, c).abs() / 2.0;
            }
            assert!((area - 5.0).abs() < 1e-5, "expected five unit squares, got {area}");
            sheet.faces[0].reverse();
        }
    }

    #[test]
    fn sheet3d_mesh_keeps_display_ghost_bonds_and_guards_invalid_numerics() {
        let (mut crystal, sheet) = square_sheet();
        assert!(!sheet.xyz.is_empty(), "{:?}", sheet.diagnostics);
        let opts = Sheet3dViewOpts::default();
        let plain = mesh_from_sheet3d(&crystal, &sheet, &opts);
        for (u, row) in crystal.adj.iter_mut().enumerate() {
            for e in row {
                e.ghost_stub = Some(crystal.pos[u]);
            }
        }
        let marked = mesh_from_sheet3d(&crystal, &sheet, &opts);
        assert_eq!(marked.edge_count, crystal.edges.len());
        assert_eq!(marked.wires.len(), 2 * crystal.edges.len());
        assert_eq!(plain.wires, marked.wires);
        assert_eq!(plain.faces, marked.faces);
        assert!(marked.faces.iter().chain(&marked.wires).flat_map(|v| v.position.iter().chain(&v.normal).chain(&v.color)).all(|x| x.is_finite()));
        let mut bad = sheet.clone();
        bad.xyz[0][0] = f64::NAN;
        assert_eq!(mesh_from_sheet3d(&crystal, &bad, &opts).vertex_count, 0);
        assert!(sheet3d_caption("test", &bad, &opts).contains("unavailable"));
        bad = sheet.clone();
        bad.faces.push(vec![0, 1, usize::MAX]);
        bad.face_strain.push(0.0);
        assert_eq!(mesh_from_sheet3d(&crystal, &bad, &opts).vertex_count, 0);
        crystal.adj[0][0].to = usize::MAX;
        assert_eq!(mesh_from_sheet3d(&crystal, &sheet, &opts).vertex_count, 0);
        let (_, valid_sheet) = square_sheet();
        let invalid_opts = Sheet3dViewOpts { z_exag: f32::NAN, ..opts };
        assert!(sheet3d_caption("test", &valid_sheet, &invalid_opts).contains("unavailable"));
    }

    #[test]
    fn sheet3d_caption_reports_incomplete_or_unsupported_geometry_without_false_flat_claims() {
        let (_, mut sheet) = square_sheet();
        sheet.converged = false;
        sheet.planar_2d_converged = false;
        sheet.planar_2d_stretch_energy = 2.0;
        sheet.stretch_energy = 1.0;
        sheet.diagnostics = vec!["Möbius spring-only preview; not a physical sheet".into()];
        let opts = Sheet3dViewOpts::default();
        let caption = sheet3d_caption("test", &sheet, &opts);
        assert!(caption.contains("Möbius") && caption.contains("not a physical"));
        assert!(caption.contains("provisional") && !caption.contains("strain relief"));
        sheet.planar_2d_stretch_energy = 0.0;
        sheet.xyz[0][2] = 1.0;
        assert!(!sheet3d_caption("test", &sheet, &opts).contains("flat z=0"));
    }
}
