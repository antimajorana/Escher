//! Wavefront `.obj` 3D mesh export for `Surfaces` (`SurfaceMesh`) and 3D `Sheet` (`Sheet3DResult`).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use papertilling_core::defect::sheet3d::Sheet3DResult;
use papertilling_core::types::SurfaceMesh;

/// Export a closed or open `SurfaceMesh` (or its `S^2` antipodal cover) to a Wavefront `.obj` file.
pub fn export_surface_obj(mesh: &SurfaceMesh, show_cover: bool, path: &Path) -> Result<String, String> {
    let target = ensure_obj_ext(path);
    let (xyz, faces): (&[[f64; 3]], &[Vec<usize>]) = if show_cover && let Some(cov) = mesh.cover.as_deref() {
        (&cov.xyz, &cov.faces)
    } else {
        (&mesh.xyz, &mesh.faces)
    };
    if xyz.is_empty() {
        return Err(String::from("cannot export .obj: 3D coordinates are empty"));
    }
    let title = if show_cover && mesh.cover.is_some() {
        format!("{} [S2 cover]", mesh.name)
    } else {
        mesh.name.clone()
    };
    write_obj(&target, &title, xyz, faces)
}

/// Export a relaxed 3D elastic defect sheet to a Wavefront `.obj` file.
pub fn export_sheet_obj(title: &str, sheet: &Sheet3DResult, path: &Path) -> Result<String, String> {
    let target = ensure_obj_ext(path);
    if sheet.xyz.is_empty() || !sheet.has_finite_geometry() {
        return Err(String::from("cannot export .obj: 3D sheet geometry is empty or invalid"));
    }
    write_obj(&target, title, &sheet.xyz, &sheet.faces)
}

fn ensure_obj_ext(path: &Path) -> PathBuf {
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("obj")) {
        path.to_path_buf()
    } else {
        path.with_extension("obj")
    }
}

fn write_obj(target: &Path, object_name: &str, xyz: &[[f64; 3]], faces: &[Vec<usize>]) -> Result<String, String> {
    let mut out = String::with_capacity(xyz.len() * 42 + faces.len() * 24 + 128);
    let safe_name: String = object_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    let _ = writeln!(out, "# papertilling Wavefront OBJ export");
    let _ = writeln!(out, "o {}", if safe_name.is_empty() { "papertilling_mesh" } else { &safe_name });

    for p in xyz {
        if !p.iter().all(|v| v.is_finite()) {
            return Err(String::from("cannot export .obj: nonfinite vertex coordinate"));
        }
        let _ = writeln!(out, "v {:.8} {:.8} {:.8}", p[0], p[1], p[2]);
    }

    let normals = vertex_normals(xyz, faces);
    for nrm in &normals {
        let _ = writeln!(out, "vn {:.6} {:.6} {:.6}", nrm[0], nrm[1], nrm[2]);
    }

    let nv = xyz.len();
    for f in faces {
        if f.len() < 3 || f.iter().any(|&u| u >= nv) {
            continue;
        }
        let _ = write!(out, "f");
        for &u in f {
            let idx = u + 1;
            let _ = write!(out, " {idx}//{idx}");
        }
        out.push('\n');
    }

    std::fs::write(target, out).map_err(|e| format!("{}: {e}", target.display()))?;
    let fname = target.file_name().map_or_else(|| target.display().to_string(), |s| s.to_string_lossy().into_owned());
    Ok(format!("Exported .obj ({fname}, {} V, {} F)", xyz.len(), faces.len()))
}

fn vertex_normals(xyz: &[[f64; 3]], faces: &[Vec<usize>]) -> Vec<[f64; 3]> {
    let mut acc = vec![[0.0_f64; 3]; xyz.len()];
    for f in faces {
        if f.len() < 3 || f.iter().any(|&u| u >= xyz.len()) {
            continue;
        }
        let mut c = [0.0; 3];
        let inv = 1.0 / (f.len() as f64);
        for &u in f {
            for k in 0..3 {
                c[k] += xyz[u][k] * inv;
            }
        }
        for m in 0..f.len() {
            let a = xyz[f[m]];
            let b = xyz[f[(m + 1) % f.len()]];
            let u = [a[0] - c[0], a[1] - c[1], a[2] - c[2]];
            let v = [b[0] - c[0], b[1] - c[1], b[2] - c[2]];
            let cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            for &idx in &[f[m], f[(m + 1) % f.len()]] {
                for k in 0..3 {
                    acc[idx][k] += cr[k];
                }
            }
        }
    }
    for n in &mut acc {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l > 1e-12 {
            n[0] /= l;
            n[1] /= l;
            n[2] /= l;
        } else {
            *n = [0.0, 0.0, 1.0];
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use papertilling_core::plane::catalogue::parse_named_str;
    use papertilling_core::surface::sphere;

    #[test]
    fn obj_export_writes_valid_vertices_and_faces() {
        let named = parse_named_str(include_str!("../../../papertilling/data/named.json")).unwrap();
        let hc = named.named.into_iter().find(|n| n.key == "honeycomb").unwrap();
        let c60 = sphere(&hc.group, &hc.cell, Some((1, 1)), None).unwrap();
        let tmp = std::env::temp_dir().join("pt_obj_smoke_c60.obj");
        let msg = export_surface_obj(&c60, false, &tmp).unwrap();
        assert!(msg.contains("60 V") && msg.contains("32 F"));
        let txt = std::fs::read_to_string(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(txt.lines().filter(|l| l.starts_with("v ")).count(), 60);
        assert_eq!(txt.lines().filter(|l| l.starts_with("f ")).count(), 32);
    }
}
