//! View-mode, color-mode, caption, and UI controls for 3D elastic defect sheets.

use papertilling_core::defect::sheet3d::Sheet3DResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefectViewMode {
    Planar2D,
    Sheet3D,
    Split2D3D,
}

impl DefectViewMode {
    pub const ALL: [Self; 3] = [Self::Planar2D, Self::Sheet3D, Self::Split2D3D];

    pub fn label(self) -> &'static str {
        match self {
            Self::Planar2D => "2D Planar",
            Self::Sheet3D => "3D Sheet",
            Self::Split2D3D => "Split (2D + 3D)",
        }
    }

    pub fn uses_3d(self) -> bool {
        !matches!(self, Self::Planar2D)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sheet3dColorMode {
    FaceRingSize,
    ElasticEnergyGsv,
    BondStrain,
    BondShearStrain,
    GaussianCurvature,
    HeightZ,
    SpectralLdos,
    SpectralCharge,
}

impl Sheet3dColorMode {
    pub const ALL: [Self; 8] = [
        Self::FaceRingSize,
        Self::ElasticEnergyGsv,
        Self::BondStrain,
        Self::BondShearStrain,
        Self::GaussianCurvature,
        Self::HeightZ,
        Self::SpectralLdos,
        Self::SpectralCharge,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::FaceRingSize => "Face ring size",
            Self::ElasticEnergyGsv => "Elastic energy (stretch+shear+bend)",
            Self::BondStrain => "Signed bond strain",
            Self::BondShearStrain => "Bond-angle shear strain",
            Self::GaussianCurvature => "Gaussian curvature Ω_v",
            Self::HeightZ => "Height z",
            Self::SpectralLdos => "LDOS |ψ|² (multiplet)",
            Self::SpectralCharge => "Spectral charge δρ(μ)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sheet3dViewOpts {
    pub kappa_b: f64,
    pub z_exag: f32,
    pub color_mode: Sheet3dColorMode,
    pub show_flat_baseline: bool,
}

impl Default for Sheet3dViewOpts {
    fn default() -> Self {
        Self {
            kappa_b: 0.02,
            z_exag: 1.0,
            color_mode: Sheet3dColorMode::FaceRingSize,
            show_flat_baseline: false,
        }
    }
}

/// Summary caption for the 3D elastic sheet viewport.
pub fn sheet3d_caption(title: &str, sheet: &Sheet3DResult, opts: &Sheet3dViewOpts) -> String {
    let title = if sheet.diagnostics.is_empty() {
        title.to_owned()
    } else {
        format!("{} | {title}", sheet.diagnostics.join("; "))
    };
    if !sheet.has_finite_geometry() || !opts.z_exag.is_finite() || !opts.kappa_b.is_finite() {
        return format!("3D unavailable: invalid numerical geometry | {title}");
    }
    if sheet.xyz.is_empty() {
        return if sheet.diagnostics.is_empty() {
            title
        } else {
            format!("3D unavailable: {title}")
        };
    }
    let z_min = sheet.xyz.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
    let z_max = sheet.xyz.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);
    let dz_a = (z_max - z_min) / sheet.mean_bond_len.max(1e-9);
    let max_eps = sheet.bond_strain.iter().map(|e| e.abs()).fold(0.0_f64, f64::max) * 100.0;
    let color_note = match opts.color_mode {
        Sheet3dColorMode::ElasticEnergyGsv => {
            format!(" | E_tot={:.4} (bend={:.4})", sheet.stretch_energy + sheet.bend_energy, sheet.bend_energy)
        }
        Sheet3dColorMode::BondShearStrain => {
            let sm = sheet.bond_shear_strain.iter().copied().fold(0.0_f64, f64::max) * 100.0;
            format!(" | max γ_shear={sm:.1}%")
        }
        Sheet3dColorMode::GaussianCurvature => {
            let om = sheet.vertex_omega.iter().map(|w| w.abs()).fold(0.0_f64, f64::max).to_degrees();
            format!(" | max|Ω_v|={om:.1}°")
        }
        _ => String::new(),
    };
    let tag = if opts.show_flat_baseline { " [Flat plane baseline]" } else { "" };
    if sheet.planar_2d_stretch_energy > 1e-8 && !opts.show_flat_baseline {
        let red = (1.0 - sheet.stretch_energy / sheet.planar_2d_stretch_energy).clamp(-1.0, 1.0) * 100.0;
        let comparison = if sheet.converged && sheet.planar_2d_converged && sheet.diagnostics.is_empty() {
            "strain relief"
        } else {
            "provisional stretch change"
        };
        format!(
            "{title}{tag} | Δz/a={dz_a:.2} (×{:.1}) | max|ε|={max_eps:.1}%{color_note} | {comparison} {red:.0}%",
            opts.z_exag
        )
    } else {
        let flat = if sheet.xyz.iter().all(|p| p[2].abs() < 1e-9 * sheet.mean_bond_len) {
            " (flat z=0)"
        } else {
            ""
        };
        format!("{title}{tag} | Δz/a={dz_a:.3}{flat} | max|ε|={max_eps:.2}%{color_note}")
    }
}

/// Compact inline UI strip for 3D sheet controls (shared by `Defects` and `Combine`).
pub fn show_sheet3d_controls(
    ui: &mut egui::Ui,
    opts: &mut Sheet3dViewOpts,
    id_salt: &str,
) -> (bool, bool) {
    let mut kappa_changed = false;
    let mut visual_changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.small("κ_b:");
        if ui
            .add(
                egui::Slider::new(&mut opts.kappa_b, 0.001..=1.5)
                    .logarithmic(true)
                    .custom_formatter(|x, _| format!("{x:.3}")),
            )
            .on_hover_text("Bending stiffness κ_b: small = sharp isometric cones/ridges; large = smooth stiff buckling")
            .changed()
        {
            kappa_changed = true;
        }
        ui.small("Z×:");
        if ui
            .add(egui::Slider::new(&mut opts.z_exag, 1.0..=3.0).step_by(0.1).suffix("×"))
            .on_hover_text("Vertical Z-exaggeration (1.0× = true metric geometry)")
            .changed()
        {
            visual_changed = true;
        }
        ui.small("Color:");
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text(opts.color_mode.label())
            .show_ui(ui, |ui| {
                for cm in Sheet3dColorMode::ALL {
                    if ui.selectable_value(&mut opts.color_mode, cm, cm.label()).changed() {
                        visual_changed = true;
                    }
                }
            });
        if ui
            .selectable_label(opts.show_flat_baseline, "Flat plane baseline")
            .on_hover_text("Compare directly against the defect-free flat z = 0 plane of this lattice")
            .clicked()
        {
            opts.show_flat_baseline = !opts.show_flat_baseline;
            visual_changed = true;
        }
    });
    (kappa_changed, visual_changed)
}
