//! Cached Bieberbach space-group cohomology (`H^2`), abelianization (`G^ab`),
//! and affine-centralizer sector (`Q_R // Z_P`) UI readouts.

use std::collections::HashMap;
use std::sync::LazyLock;

use papertilling_core::bieberbach::{
    analyze_wallpaper_group, from_wallpaper_group, BieberbachSummary, SpaceGroupAnalyzer,
};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::groups::{canonical_symbol, prim_u, prim_u_inv, WALLPAPER_GROUPS};
use papertilling_core::types::{Isometry2, Recipe};

fn mat2_inv_loc(m: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    (d.abs() > 1e-12).then_some([[m[1][1] / d, -m[0][1] / d], [-m[1][0] / d, m[0][0] / d]])
}

fn mat2_mul_loc(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]
}

fn mat2_vec_loc(m: [[f64; 2]; 2], v: [f64; 2]) -> [f64; 2] {
    [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]]
}

static WALLPAPER_BIEBERBACH: LazyLock<HashMap<String, BieberbachSummary>> = LazyLock::new(|| {
    let mut map = HashMap::with_capacity(WALLPAPER_GROUPS.len());
    for &g in &WALLPAPER_GROUPS {
        if let Ok(sum) = analyze_wallpaper_group(g) {
            map.insert(g.to_string(), sum);
        }
    }
    map
});

static WALLPAPER_ANALYZERS: LazyLock<HashMap<String, SpaceGroupAnalyzer>> = LazyLock::new(|| {
    let mut map = HashMap::with_capacity(WALLPAPER_GROUPS.len());
    for &g in &WALLPAPER_GROUPS {
        if let Ok(sg) = from_wallpaper_group(g) {
            map.insert(g.to_string(), sg);
        }
    }
    map
});

/// Cached summary lookup for any wallpaper group symbol/alias.
pub fn wallpaper_bieberbach_summary(group: &str) -> Option<&'static BieberbachSummary> {
    let canon = canonical_symbol(group).ok()?;
    WALLPAPER_BIEBERBACH.get(canon)
}

/// Evaluate canonical `chi_G(g) in G^ab` for a planar crystal patch/circuit Cartesian holonomy.
pub fn format_holonomy_abelian_charge(
    group: &str,
    recipe: &Recipe,
    holonomy: &Isometry2,
) -> Option<String> {
    let canon = canonical_symbol(group).ok()?;
    let sg = WALLPAPER_ANALYZERS.get(canon)?;
    let sum = WALLPAPER_BIEBERBACH.get(canon)?;
    let cell = LatticeFamily::from_recipe(recipe, None).ok()?.cell();
    let a = [[cell.a1[0], cell.a2[0]], [cell.a1[1], cell.a2[1]]];
    let a_inv = mat2_inv_loc(a)?;
    let r_conv_f = mat2_mul_loc(a_inv, mat2_mul_loc(holonomy.m, a));
    let u_inv = prim_u_inv(canon).ok()?;
    let u = prim_u(canon).ok()?;
    let r_prim_f = mat2_mul_loc(u_inv, mat2_mul_loc(r_conv_f, u));
    let r_prim = vec![
        vec![r_prim_f[0][0].round() as i64, r_prim_f[0][1].round() as i64],
        vec![r_prim_f[1][0].round() as i64, r_prim_f[1][1].round() as i64],
    ];
    for i in 0..2 {
        for j in 0..2 {
            if (r_prim_f[i][j] - r_prim[i][j] as f64).abs() > 1e-2 {
                return None;
            }
        }
    }
    let r_idx = sg.point_index(&r_prim)?;
    let t_conv = mat2_vec_loc(a_inv, holonomy.t);
    let t_prim = mat2_vec_loc(u_inv, t_conv);
    let mut b_prim = [0_i64; 2];
    for k in 0..2 {
        let diff = t_prim[k] - sg.tau[r_idx][k].to_f64();
        let rnd = diff.round();
        if (diff - rnd).abs() > 5e-2 {
            return None;
        }
        b_prim[k] = rnd as i64;
    }
    let ab = sg.compute_abelianization(&sg.all_indices());
    let chi = ab.chi(&b_prim, r_idx);
    let chi_str = if chi.is_empty() {
        "0".to_string()
    } else {
        format!(
            "({})",
            chi.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(", ")
        )
    };
    Some(format!("chi_G = {chi_str} in {}", sum.ab_full_label))
}

/// Render a collapsible Bieberbach cohomology & topological sector readout.
pub fn show_wallpaper_bieberbach_collapsible(ui: &mut egui::Ui, group: &str, id_salt: &str) {
    let Some(sum) = wallpaper_bieberbach_summary(group) else {
        return;
    };
    egui::CollapsingHeader::new(format!(
        "Cohomology & Sectors: H²={}, Gᵃᵇ={}",
        sum.h2_label, sum.ab_full_label
    ))
    .id_salt(id_salt)
    .default_open(false)
    .show(ui, |ui| {
        let split_tag = if sum.is_split {
            "split ([ν]=0)"
        } else {
            "nonsymmorphic ([ν]≠0)"
        };
        let split_col = if sum.is_split {
            egui::Color32::from_rgb(24, 122, 54)
        } else {
            egui::Color32::from_rgb(162, 74, 14)
        };
        ui.horizontal_wrapped(|ui| {
            ui.small(format!("#{} {} | |P|={} | ", sum.number, sum.symbol, sum.point_order));
            ui.colored_label(split_col, egui::RichText::new(split_tag).small().strong());
        });
        ui.small(format!(
            "H²(P, Λ) = {} | (G⁺)ᵃᵇ = {} | Gᵃᵇ = {}",
            sum.h2_label, sum.ab_plus_label, sum.ab_full_label
        ));
        let tor_str: Vec<String> = sum
            .torus_counts
            .iter()
            .map(|(n, c)| format!("N={n}: {c}"))
            .collect();
        ui.small(format!("#Conj(G / NΛ): {}", tor_str.join(", ")));
        if sum.sectors_full.is_empty() {
            ui.small("Sectors: pure translations only (p1)");
        } else {
            for s in &sum.sectors_full {
                let kind = if s.det > 0 { "rot" } else { "mir/gl" };
                let detail = if s.r_rank == 0 {
                    format!("T_R={} -> {} Wyckoff class(es)", s.t_r_label, s.num_orbits)
                } else {
                    format!(
                        "T_R={}, α={}, pure={}, elem={}",
                        s.t_r_label,
                        s.alpha_r.as_deref().unwrap_or("0"),
                        s.num_pure_classes,
                        s.num_elem_glide_orbits
                    )
                };
                ui.small(format!(
                    "  {kind} ord={} (×{}): {detail}",
                    s.order, s.class_size_p
                ));
            }
        }
    });
}

/// Show abelianized loop holonomy charge `chi_G([W]) in G^ab` (when circuit active)
/// followed by the collapsible Bieberbach cohomology & sector summary.
pub fn show_bieberbach_inspector_section(
    ui: &mut egui::Ui,
    group: &str,
    recipe: &Recipe,
    inspectors: &crate::defects_inspectors::DefectInspectors,
    id_salt: &str,
) {
    if inspectors.circuit.enabled
        && let Some(ov) = &inspectors.circuit.overlay
        && let Some(chi_line) = format_holonomy_abelian_charge(group, recipe, &ov.result.h)
    {
        ui.small(format!("Abelianized loop charge: {chi_line}"));
    }
    show_wallpaper_bieberbach_collapsible(ui, group, id_salt);
}
