//! "Strain / Nye": per-site least-squares strain from the builder's reference positions, the discrete Nye tensor per
//! face with its telescoping total, and Vitek differential displacements for the selected defect's Burgers vector.

use papertilling_core::defect_fields::strain_vitek_nye::VitekArrow;
use papertilling_core::defect_fields::{LocalStrain, NyeField, local_deformation_and_strain, nye_tensor, vitek_differential};
use papertilling_core::types::Frame2D;

use super::detect::{DetectOut, fmt_v, loop_lines, ref_lattice, resolve};
use super::{FieldKind, SCENE_TITLE, Settings, Source};
use crate::canvas2d::Scene2D;
use crate::scene2d::{LinePrim, PolyPrim};
use crate::workbench::scene_helpers::{EDGE_RGBA, heat_rgba, vector_scene, with_lines};

const VITEK_RGBA: [u8; 4] = [120, 20, 20, 255];

pub(crate) struct FieldsOut {
    pub(crate) det: DetectOut,
    /// Index of the defect the Vitek map uses (resolved from the settings).
    pub(crate) selected: usize,
    pub(crate) strain: Result<Vec<LocalStrain>, String>,
    /// Bonds whose reference vector is an ideal bond (used in the strain fit) and the ones left out (the cut).
    pub(crate) strain_bonds: (usize, usize),
    pub(crate) max_u: f64,
    pub(crate) nye: Result<NyeField, String>,
    pub(crate) vitek: Result<(Vec<VitekArrow>, [f64; 2]), String>,
}

/// The selected defect's Burgers vector (cell frame, chart of `loop_path[0]`), refused with the defect's reason.
pub(crate) fn selected_burgers(det: &DetectOut, selected: usize) -> Result<[f64; 2], String> {
    let n = det.det.defects.len();
    let d = det.det.defects.get(selected).ok_or_else(|| {
        if n == 0 { String::from("REFUSED: no defect was detected") } else { format!("REFUSED: defect #{selected} does not exist ({n} detected)") }
    })?;
    match d.burgers {
        Some(b) if b[0].hypot(b[1]) > 1e-9 => Ok(b),
        Some(_) => Err(format!("REFUSED: defect #{selected} ({}) has zero Burgers vector", d.kind)),
        None => Err(format!("REFUSED: defect #{selected} ({}) has no Burgers vector: {}", d.kind, d.reason)),
    }
}

pub(crate) fn build(src: &Source, set: &Settings, det: DetectOut) -> Result<FieldsOut, String> {
    let rl = ref_lattice(src);
    let tol = 1e-6 * rl.lmax;
    let has_ref = src.ref_pos.len() == src.pos.len();
    let is_ideal = |d: [f64; 2]| rl.star.iter().any(|s| (s[0] - d[0]).hypot(s[1] - d[1]) < tol);
    type Edges = Vec<(usize, usize)>;
    let (kept, cut): (Edges, Edges) = if has_ref {
        det.det.edges.iter().partition(|&&(u, v)| is_ideal([src.ref_pos[v][0] - src.ref_pos[u][0], src.ref_pos[v][1] - src.ref_pos[u][1]]))
    } else {
        (Vec::new(), Vec::new())
    };
    let strain = if has_ref {
        local_deformation_and_strain(&src.ref_pos, &src.pos, &kept)
    } else {
        Err(String::from("REFUSED: the source has no reference position for every site"))
    };
    let max_u = if has_ref {
        src.pos.iter().zip(&src.ref_pos).map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1])).fold(0.0, f64::max)
    } else {
        0.0
    };
    let nye = nye_tensor(&src.pos, &src.cell, Frame2D::default(), Some(&det.det.edges), None, None);
    let selected = resolve(&det.det, set.selected);
    let vitek = selected_burgers(&det, selected).and_then(|b| {
        if !has_ref {
            return Err(String::from("REFUSED: the source has no reference positions"));
        }
        vitek_differential(&src.ref_pos, &src.pos, &det.det.edges, b).map(|a| (a, b))
    });
    Ok(FieldsOut { det, selected, strain, strain_bonds: (kept.len(), cut.len()), max_u, nye, vitek })
}

/// The strain painted and summarised per site: Green-Lagrange `E = (F^T F - I) / 2`, which is zero under a rigid
/// rotation (the linear `(F + F^T) / 2 - I` reads `2 (cos θ - 1)` in its trace for a rotation by θ).
pub(crate) fn site_strain(l: &LocalStrain) -> Option<&[[f64; 2]; 2]> {
    l.green.as_ref()
}

pub(crate) fn trace(e: &[[f64; 2]; 2]) -> f64 {
    e[0][0] + e[1][1]
}

/// Largest in-plane shear `sqrt(((e11 - e22) / 2)^2 + e12^2)`.
pub(crate) fn max_shear(e: &[[f64; 2]; 2]) -> f64 {
    (0.5 * (e[0][0] - e[1][1])).hypot(0.5 * (e[0][1] + e[1][0]))
}

fn abs_max(v: impl Iterator<Item = f64>) -> f64 {
    v.filter(|x| x.is_finite()).fold(0.0_f64, |m, x| m.max(x.abs())).max(1e-12)
}

/// Recolour the site circles of a vector scene (one colour per site).
fn tint_sites(scene: Scene2D, rgba: &[[u8; 4]]) -> Scene2D {
    match scene {
        Scene2D::VectorScene { mut primitives } => {
            for (i, ci) in primitives.site_circle_indices.clone().into_iter().enumerate() {
                if let (Some(ci), Some(c)) = (ci, rgba.get(i)) {
                    primitives.circles[ci].fill = *c;
                }
            }
            Scene2D::VectorScene { primitives }
        }
        other => other,
    }
}

fn site_scene(o: &FieldsOut, vals: &[f64], what: &str) -> Scene2D {
    let m = abs_max(vals.iter().copied());
    let rgba: Vec<[u8; 4]> = vals.iter().map(|&v| heat_rgba(v, -m, m)).collect();
    let note = format!("{what}: colour range ±{m:.3e} (blue negative, white 0, red positive; grey = no fit)");
    let base = vector_scene(&o.det.pos, &o.det.det.edges, &[], |_| [0, 0, 0, 0], SCENE_TITLE, &note);
    tint_sites(base, &rgba)
}

pub(crate) fn scene(o: &FieldsOut, set: &Settings) -> Scene2D {
    let pos = &o.det.pos;
    let loops = loop_lines(&o.det, o.selected);
    let s = match set.field {
        FieldKind::NyeFace => match &o.nye {
            Ok(n) => {
                let mag: Vec<f64> = n.alpha.iter().map(|a| a[0].hypot(a[1])).collect();
                let m = abs_max(mag.iter().copied());
                let note = format!("Nye |α| per face, white 0 to red {m:.3e}");
                let mut s = vector_scene(pos, &o.det.det.edges, &n.faces, |f| heat_rgba(mag[f], -m, m), SCENE_TITLE, &note);
                if let Scene2D::VectorScene { primitives } = &mut s {
                    primitives.polygons.iter_mut().for_each(|p: &mut PolyPrim| p.stroke = None);
                }
                s
            }
            Err(e) => vector_scene(pos, &o.det.det.edges, &[], |_| [0, 0, 0, 0], SCENE_TITLE, e),
        },
        FieldKind::StrainTrace | FieldKind::MaxShear => {
            let f = if set.field == FieldKind::StrainTrace { trace } else { max_shear };
            let vals: Vec<f64> = match &o.strain {
                Ok(ls) => ls.iter().map(|l| site_strain(l).map_or(f64::NAN, f)).collect(),
                Err(_) => vec![f64::NAN; pos.len()],
            };
            site_scene(o, &vals, &format!("{} (Green-Lagrange E = (F^T F - I)/2)", set.field.label()))
        }
        FieldKind::Vitek => vitek_scene(o),
        FieldKind::PsiN => vector_scene(pos, &o.det.det.edges, &[], |_| [0, 0, 0, 0], SCENE_TITLE, ""),
    };
    with_lines(s, loops)
}

/// Vitek map: per bond, an arrow at the midpoint along the bond, length ∝ the wrapped component along b.
fn vitek_scene(o: &FieldsOut) -> Scene2D {
    let pos = &o.det.pos;
    let Ok((arrows, b)) = &o.vitek else {
        let why = o.vitek.as_ref().err().cloned().unwrap_or_default();
        return vector_scene(pos, &o.det.det.edges, &[], |_| [0, 0, 0, 0], SCENE_TITLE, &why);
    };
    let half = 0.5 * b[0].hypot(b[1]);
    let mut lines = Vec::new();
    for a in arrows {
        let (p, q) = (pos[a.i], pos[a.j]);
        let s = a.wrapped / half;
        if s.abs() < 0.02 {
            continue;
        }
        let mid = [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])];
        let d = [0.45 * s * (q[0] - p[0]), 0.45 * s * (q[1] - p[1])];
        let (a0, a1) = ([mid[0] - d[0], mid[1] - d[1]], [mid[0] + d[0], mid[1] + d[1]]);
        lines.push(LinePrim {
            a: [a0[0] as f32, a0[1] as f32],
            b: [a1[0] as f32, a1[1] as f32],
            color: VITEK_RGBA,
            width: 2.0,
            dashed: false,
            arrow_mid: true,
        });
    }
    let note = format!("Vitek map for b = {}: arrow length ∝ wrapped (u_j - u_i)·b̂, full bond = |b|/2", fmt_v(*b));
    let mut s = vector_scene(pos, &o.det.det.edges, &[], |_| [0, 0, 0, 0], SCENE_TITLE, &note);
    if let Scene2D::VectorScene { primitives } = &mut s {
        primitives.lines.iter_mut().for_each(|l| l.color = [EDGE_RGBA[0], EDGE_RGBA[1], EDGE_RGBA[2], 90]);
    }
    with_lines(s, lines)
}

pub(crate) fn show_inspector(ui: &mut egui::Ui, o: &FieldsOut, set: &Settings) {
    ui.label(format!("Shown: {}", set.field.label()));
    ui.label(format!("Largest site displacement |pos - ref_pos|: {:.4}", o.max_u));
    match &o.strain {
        Ok(ls) => {
            let eps: Vec<&[[f64; 2]; 2]> = ls.iter().filter_map(site_strain).collect();
            let tr = abs_max(eps.iter().map(|e| trace(e)));
            let sh = abs_max(eps.iter().map(|e| max_shear(e)));
            ui.label(format!(
                "[COMPUTATIONAL EVIDENCE] least-squares Green-Lagrange strain E = (F^T F - I)/2 at {} of {} sites; \
                 max |trace| {tr:.3e}, max shear {sh:.3e}",
                eps.len(),
                ls.len()
            ));
            ui.small(format!(
                "{} bonds fitted; {} bonds whose reference vector is not an ideal bond (they cross a cut) left out.",
                o.strain_bonds.0, o.strain_bonds.1
            ));
            if o.max_u == 0.0 {
                ui.small("Reference and current positions coincide: the strain is zero by construction, not by measurement.");
            }
        }
        Err(e) => {
            ui.label(format!("Strain: {e}"));
        }
    }
    ui.separator();
    match &o.nye {
        Ok(n) => {
            ui.label(format!("Nye tensor on {} faces (region: all bounded faces)", n.faces.len()));
            ui.label(format!("Net Frank angle of the region: {:+.0}°; frame angle {:.3}°", n.net_frank_deg, n.frame_angle_deg));
            match n.total {
                Some(t) => {
                    let c = n.boundary_closure;
                    let diff = (t[0] - c[0]).hypot(t[1] - c[1]);
                    ui.label(format!("Σ α A_f = {}, boundary closure = {}, |difference| = {diff:.2e}", fmt_v(t), fmt_v(c)));
                }
                None => {
                    ui.label(format!("Boundary closure = {} (no total: see the reason)", fmt_v(n.boundary_closure)));
                }
            }
            ui.small(n.reason.as_str());
        }
        Err(e) => {
            ui.label(format!("Nye tensor: {e}"));
        }
    }
    ui.separator();
    match &o.vitek {
        Ok((a, b)) => {
            let shown = a.iter().filter(|x| x.wrapped.abs() >= 0.01 * b[0].hypot(b[1])).count();
            ui.label(format!(
                "[COMPUTATIONAL EVIDENCE] Vitek map of defect #{} with b = {}: {shown} of {} bonds carry a visible component",
                o.selected,
                fmt_v(*b),
                a.len()
            ));
        }
        Err(e) => {
            ui.label(format!("Vitek map: {e}"));
        }
    }
    ui.small("Source: Vitek, Perrin & Bowen (1970) differential displacement maps.");
}
