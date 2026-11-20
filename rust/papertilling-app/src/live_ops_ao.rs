//! Ariza–Ortiz 2D metric relaxation modes and gauge/GSV overlay builders for `Defects` and `Combine`.

use papertilling_core::defect::DefectCatalogue;
use papertilling_core::defect::cone_builder::mark_long_bonds;
use papertilling_core::defect::gauge::{plaquette_holonomies, wyckoff_rotocentres_with_frame};
use papertilling_core::defect::gsv_elastic::{Elastic2DResult, gsv_diagnostics_2d, relax_elastic_2d};
use papertilling_core::plane::draw::relax_preview;
use papertilling_core::types::{BuiltCrystal, Cell2D, Frame2D};

use crate::canvas2d::gauge_overlay::{CrystalOverlayData, CrystalOverlayMode, PlaquetteBadge, perceptual_heat_scale};
use crate::canvas2d::overlay_scales::{bond_strain_overlay, perceptual_colorbar};
use crate::mesh3d::sheet_colors::{rgba_u8, sequential_heat_rgba};

#[cfg(test)]
#[path = "live_ops_ao_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Defect2dMode {
    Raw,
    #[default]
    RelaxedDrawing,
    ElasticAoMetric,
}

impl Defect2dMode {
    /// The modes the UI offers. C1: the AO elastic view left the UI (it solves a Green-St Venant metric energy
    /// with geometric weights, not Ariza-Ortiz elasticity, and a glued 2D wall has no isometric immersion in the
    /// plane); its variant stays for session decoding, where it is migrated to `RelaxedDrawing` with a note.
    pub const ALL: [Self; 2] = [Self::Raw, Self::RelaxedDrawing];
    /// Every variant a saved session may name.
    pub const DECODE: [Self; 3] = [Self::Raw, Self::RelaxedDrawing, Self::ElasticAoMetric];

    pub fn label(self) -> &'static str {
        match self {
            Self::Raw => "Raw surgery",
            Self::RelaxedDrawing => "Relaxed (drawing)",
            Self::ElasticAoMetric => "Elastic (AO metric)",
        }
    }
}

#[allow(dead_code)] // C1 hidden: the AO elastic view left the UI; the core solver is unchanged
pub fn relax_crystal_ao_2d(crystal: &mut BuiltCrystal) -> Result<Elastic2DResult, String> {
    let res = relax_elastic_2d(crystal)?;
    if res.pos.len() == crystal.pos.len() {
        crystal.pos = res.pos.clone();
        mark_long_bonds(crystal);
    }
    Ok(res)
}

#[allow(dead_code)] // C1 hidden: the AO elastic view left the UI
pub fn apply_combine_ao_mode(crystal: &mut BuiltCrystal, preview: bool, notes: &mut Vec<String>) {
    if preview {
        relax_preview(crystal, 8.0);
        return;
    }
    match relax_crystal_ao_2d(crystal) {
        Ok(res) => notes.push(format!("AO GSV E={:.4} (shear={:.4})", res.total_energy, res.shear_energy)),
        Err(err) => notes.push(format!("AO elastic fallback: {err}")),
    }
}

fn heat_bonds(values: &[f64]) -> (Vec<([u8; 4], f32)>, f64) {
    let (scale, vmax) = perceptual_heat_scale(values);
    let cols = values
        .iter()
        .map(|&e| {
            let t = (e.max(0.0) / scale).clamp(0.0, 1.0).sqrt();
            let c = rgba_u8(sequential_heat_rgba(t, 1.0, 1.0));
            (c, (1.3 + 2.0 * t) as f32)
        })
        .collect();
    (cols, vmax.min(scale))
}

fn match_wyckoff_label(
    cat: &DefectCatalogue,
    wyck: &[papertilling_core::defect::gauge::WyckoffRotocentre],
    ainv: [[f64; 2]; 2],
    centre: [f64; 2],
) -> Option<String> {
    if let Some(cc) = cat.centre_class_of(centre)
        && !cc.label.starts_with("order-")
    {
        return Some(cc.label.clone());
    }
    let fc = [
        ainv[0][0] * centre[0] + ainv[0][1] * centre[1],
        ainv[1][0] * centre[0] + ainv[1][1] * centre[1],
    ];
    wyck.iter()
        .find(|w| {
            let df = [fc[0] - w.frac[0], fc[1] - w.frac[1]];
            (df[0] - df[0].round()).hypot(df[1] - df[1].round()) < 1e-3
        })
        .map(|w| w.label.clone())
}

/// The C1 vertex angle shear: at each site, the bonds are ordered by the angle of their rest vectors, and each pair of
/// *adjacent* bonds (consecutive in that order, cyclically) whose rest angle is below 180° gives one angle change
/// |Δθ| = |θ_now − θ_rest| (wrapped to [0, π]), θ_now taken from the drawn positions. At a site-centred disclination
/// apex (`crystal.cone_sites`), `b.vec` is un-sheeted while `b.ang` orders the cone sheets, so adjacent arm pairs and
/// their intrinsic rest angles come from developing each bounded face corner around the apex via [`develop_face`].
/// Cut-open ghost bonds and non-orientable Möbius seam-crossing chords are skipped; opposite bonds and rim outer gaps
/// (≥ 180°) are never paired. Returns the mean |Δθ| per site and per bond (`crystal.edges` order).
pub fn adjacent_angle_shear(crystal: &BuiltCrystal) -> (Vec<f64>, Vec<f64>) {
    use papertilling_core::defect::charge::frame_faces;
    use papertilling_core::growth::face_graph::develop_face;
    use std::collections::{HashMap, HashSet};
    use std::f64::consts::{PI, TAU};
    let n = crystal.pos.len();
    let edge_of: HashMap<(usize, usize), usize> =
        crystal.edges.iter().enumerate().map(|(i, &(a, b))| ((a.min(b), a.max(b)), i)).collect();
    let cone_set: HashSet<usize> = crystal.cone_sites.iter().copied().collect();
    let cone_faces: Vec<Vec<usize>> = if !cone_set.is_empty() && !crystal.is_non_orientable {
        let walks = frame_faces(crystal);
        if walks.len() >= 2 {
            let outer = (0..walks.len()).fold(0, |best, i| if walks[i].len() > walks[best].len() { i } else { best });
            walks.into_iter().enumerate().filter(|&(i, _)| i != outer).map(|(_, w)| w).collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    let wrap = |a: f64| a.rem_euclid(TAU);
    let bond_ok = |b: &papertilling_core::types::DirectedBondData| {
        b.to < n && b.ghost_stub.is_none() && !(crystal.is_non_orientable && b.across) && b.vec[0].hypot(b.vec[1]) > 1e-12
    };
    let mut site = vec![0.0_f64; n];
    let (mut bond_sum, mut bond_cnt) = (vec![0.0_f64; crystal.edges.len()], vec![0_usize; crystal.edges.len()]);
    for u in 0..n.min(crystal.adj.len()) {
        let (mut sum, mut cnt) = (0.0, 0_usize);
        if cone_set.contains(&u) {
            for f in &cone_faces {
                if f.len() < 3 {
                    continue;
                }
                for m in 0..f.len() {
                    if f[m] != u {
                        continue;
                    }
                    let (w_next, w_prev) = (f[(m + 1) % f.len()], f[(m + f.len() - 1) % f.len()]);
                    if w_next == w_prev
                        || !crystal.adj[u].iter().any(|b| b.to == w_next && bond_ok(b))
                        || !crystal.adj[u].iter().any(|b| b.to == w_prev && bond_ok(b))
                    {
                        continue;
                    }
                    let Some(poly) = develop_face(crystal, f, m) else { continue };
                    let d_next = [poly[1][0] - poly[0][0], poly[1][1] - poly[0][1]];
                    let d_prev = [poly[poly.len() - 1][0] - poly[0][0], poly[poly.len() - 1][1] - poly[0][1]];
                    if d_next[0].hypot(d_next[1]) <= 1e-12 || d_prev[0].hypot(d_prev[1]) <= 1e-12 {
                        continue;
                    }
                    let rest = wrap(d_prev[1].atan2(d_prev[0]) - d_next[1].atan2(d_next[0]));
                    if rest >= PI - 1e-9 || rest <= 1e-12 {
                        continue;
                    }
                    let now_a = [crystal.pos[w_next][0] - crystal.pos[u][0], crystal.pos[w_next][1] - crystal.pos[u][1]];
                    let now_b = [crystal.pos[w_prev][0] - crystal.pos[u][0], crystal.pos[w_prev][1] - crystal.pos[u][1]];
                    if now_a[0].hypot(now_a[1]) <= 1e-12 || now_b[0].hypot(now_b[1]) <= 1e-12 {
                        continue;
                    }
                    let now = wrap(now_b[1].atan2(now_b[0]) - now_a[1].atan2(now_a[0]));
                    let d = ((now - rest + PI).rem_euclid(TAU) - PI).abs();
                    sum += d;
                    cnt += 1;
                    for w in [w_next, w_prev] {
                        if let Some(&e) = edge_of.get(&(u.min(w), u.max(w))) {
                            bond_sum[e] += d;
                            bond_cnt[e] += 1;
                        }
                    }
                }
            }
        } else {
            let mut arms: Vec<(f64, [f64; 2], Option<usize>)> = crystal.adj[u]
                .iter()
                .filter(|b| bond_ok(b))
                .map(|b| {
                    let now = [crystal.pos[b.to][0] - crystal.pos[u][0], crystal.pos[b.to][1] - crystal.pos[u][1]];
                    (b.vec[1].atan2(b.vec[0]), now, edge_of.get(&(u.min(b.to), u.max(b.to))).copied())
                })
                .collect();
            if arms.len() < 2 {
                continue;
            }
            arms.sort_by(|a, b| a.0.total_cmp(&b.0));
            for i in 0..arms.len() {
                let (a, b) = (arms[i], arms[(i + 1) % arms.len()]);
                let rest = wrap(b.0 - a.0);
                if rest >= PI - 1e-9 || rest <= 1e-12 {
                    continue;
                }
                let (now_a, now_b) = (a.1, b.1);
                if now_a[0].hypot(now_a[1]) <= 1e-12 || now_b[0].hypot(now_b[1]) <= 1e-12 {
                    continue;
                }
                let now = wrap(now_b[1].atan2(now_b[0]) - now_a[1].atan2(now_a[0]));
                let d = ((now - rest + PI).rem_euclid(TAU) - PI).abs();
                sum += d;
                cnt += 1;
                for e in [a.2, b.2].into_iter().flatten() {
                    bond_sum[e] += d;
                    bond_cnt[e] += 1;
                }
            }
        }
        if cnt > 0 {
            site[u] = sum / cnt as f64;
        }
    }
    let bond = bond_sum.iter().zip(&bond_cnt).map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 }).collect();
    (site, bond)
}

pub fn build_ao_crystal_overlay(
    crystal: &BuiltCrystal,
    mode: CrystalOverlayMode,
    group: &str,
    cell: &Cell2D,
    frame: Frame2D,
) -> Option<CrystalOverlayData> {
    if crystal.pos.is_empty() {
        return None;
    }
    match mode {
        CrystalOverlayMode::GsvEnergyShear => {
            let diag = gsv_diagnostics_2d(crystal, &crystal.pos).ok()?;
            let (site_cols, legend) = perceptual_colorbar(&diag.site_gsv_energy, "Site GSV E (√; bonds own scale)");
            let (bond_cols, _bmax) = heat_bonds(&diag.bond_gsv_energy);
            Some(CrystalOverlayData {
                site_rgba: Some(site_cols),
                bond_rgba: Some(bond_cols),
                legend: Some(legend),
                note_suffix: Some(format!("AO E_tot={:.4} (E_shear={:.4})", diag.total_energy, diag.shear_energy)),
                ..Default::default()
            })
        }
        CrystalOverlayMode::BondStretchStrain => {
            let diag = gsv_diagnostics_2d(crystal, &crystal.pos).ok()?;
            Some(bond_strain_overlay(crystal.pos.len(), &crystal.edges, &diag.bond_strain))
        }
        CrystalOverlayMode::VertexAngleShear => {
            let (site, bond) = adjacent_angle_shear(crystal);
            let (site_cols, legend) = perceptual_colorbar(&site, "Angle change |Δθ| between adjacent bonds (rad, √)");
            let (bond_cols, _bmax) = heat_bonds(&bond);
            Some(CrystalOverlayData {
                site_rgba: Some(site_cols),
                bond_rgba: Some(bond_cols),
                legend: Some(legend),
                ..Default::default()
            })
        }
        CrystalOverlayMode::PlaquetteFlux => {
            let hols = plaquette_holonomies(crystal, Some(group), Some(cell));
            let cat = DefectCatalogue::with_bounds(group, cell, frame, 0.0, 0.0);
            let wyck = wyckoff_rotocentres_with_frame(group, cell, frame).unwrap_or_default();
            let ainv = papertilling_core::defect::gauge::SpaceGroupExtension::from_cell(group, cell)
                .map_or([[1.0, 0.0], [0.0, 1.0]], |e| e.a_inv);
            let mut badges = Vec::new();
            for p in hols {
                if !(p.has_curvature || p.has_torsion || p.nonsymmorphic_shift) || p.vertices.is_empty() {
                    continue;
                }
                let mut sum = [0.0_f64; 2];
                let mut cnt = 0usize;
                for &u in &p.vertices {
                    if let Some(pt) = crystal.pos.get(u) {
                        sum[0] += pt[0];
                        sum[1] += pt[1];
                        cnt += 1;
                    }
                }
                if cnt == 0 {
                    continue;
                }
                let center = [sum[0] / (cnt as f64), sum[1] / (cnt as f64)];
                let wy_lbl = p.centre.and_then(|c| match_wyckoff_label(&cat, &wyck, ainv, c));
                let label = if p.det_sign < 0 {
                    if p.nonsymmorphic_shift { String::from("glide det=-1") } else { String::from("mirror det=-1") }
                } else if p.has_curvature {
                    let prefix = wy_lbl.map(|w| format!("{w} ")).unwrap_or_default();
                    if p.nonsymmorphic_shift || (p.alpha_f[0].hypot(p.alpha_f[1]) > 1e-4 && p.has_torsion) {
                        format!("{prefix}Ω={:+.0}°+τ", p.omega_deg)
                    } else {
                        format!("{prefix}Ω={:+.0}°", p.omega_deg)
                    }
                } else if p.has_torsion {
                    format!("α=({:+.2},{:+.2})", p.alpha_f[0], p.alpha_f[1])
                } else {
                    String::from("τ-shift")
                };
                badges.push(PlaquetteBadge {
                    center,
                    label,
                    is_rotation: p.has_curvature,
                    improper: p.det_sign < 0,
                });
            }
            let chips = crate::canvas2d::overlay_legend::flux_legend_chips(&badges);
            Some(CrystalOverlayData { badges, chips, ..Default::default() })
        }
        _ => None,
    }
}
