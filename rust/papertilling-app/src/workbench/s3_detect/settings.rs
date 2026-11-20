//! S3 controls: defaults, validation (refusals with reasons), the left-rail widgets and session save / restore.

use serde_json::{Value, json};

use super::S3Sub;

/// What the "Strain / Nye" sub-row paints (display only: one build computes all of them).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum FieldKind {
    #[default]
    NyeFace,
    StrainTrace,
    MaxShear,
    Vitek,
    PsiN,
}

impl FieldKind {
    pub(crate) const ALL: [Self; 5] = [Self::NyeFace, Self::StrainTrace, Self::MaxShear, Self::Vitek, Self::PsiN];
    /// Shown fields (`|ψ_n|` is hidden from the UI).
    pub(crate) const SHOWN: [Self; 4] = [Self::NyeFace, Self::StrainTrace, Self::MaxShear, Self::Vitek];

    pub(crate) fn is_shown(self) -> bool {
        Self::SHOWN.contains(&self)
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::NyeFace => "Nye |α| per face",
            Self::StrainTrace => "Strain trace per site",
            Self::MaxShear => "Max shear per site",
            Self::Vitek => "Vitek differential map",
            Self::PsiN => "|ψ_n| per site",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Settings {
    /// Use the crystal's own bonds instead of a distance cutoff on the bare coordinates.
    pub(crate) own_bonds: bool,
    /// Cutoff midway between the longest ideal bond and the shortest ideal non-bond (`default_cutoff`).
    pub(crate) auto_cutoff: bool,
    /// Manual cutoff in units of the longest ideal bond.
    pub(crate) cutoff_rel: f64,
    /// Face rings added round each defect seed before its loop is walked.
    pub(crate) clearance: usize,
    /// Detected defect whose Burgers vector Fields / Elastic use; `None` = the first with a non-zero one.
    pub(crate) selected: Option<usize>,
    pub(crate) field: FieldKind,
    pub(crate) psi_n: u32,
    /// Stroh sampling grid: `grid x grid` cells (even, so no cell centre sits on the line).
    pub(crate) grid: usize,
    /// Half-width of the Stroh window in longest ideal bonds.
    pub(crate) extent: f64,
    pub(crate) manual_b: bool,
    /// Manual Burgers vector `u a1 + v a2` (FS/RH convention, as fed to the Stroh field).
    pub(crate) manual_uv: [i32; 2],
    pub(crate) gif_frames: usize,
    pub(crate) gif_px: u16,
    pub(crate) gif_delay_cs: u16,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            own_bonds: false,
            auto_cutoff: true,
            cutoff_rel: 1.3,
            clearance: 1,
            selected: None,
            field: FieldKind::default(),
            psi_n: 6,
            grid: 40,
            extent: 6.0,
            manual_b: false,
            manual_uv: [1, 0],
            gif_frames: 12,
            gif_px: 320,
            gif_delay_cs: 8,
        }
    }
}

/// Manual cutoff range in longest ideal bonds. The lower end sits just above 1: bonds are kept only when strictly
/// shorter than the cutoff, so at exactly 1.0 ideal bonds of the longest length are dropped by rounding. Above
/// sqrt 2 (square) or sqrt 3 (triangular, honeycomb) the cutoff graph is not plane and detection refuses it.
pub(crate) const CUTOFF_RANGE: std::ops::RangeInclusive<f64> = 1.001..=2.5;
pub(crate) const MAX_CLEARANCE: usize = 4;
pub(crate) const GRID_RANGE: std::ops::RangeInclusive<usize> = 10..=80;
pub(crate) const EXTENT_RANGE: std::ops::RangeInclusive<f64> = 2.0..=20.0;
pub(crate) const FRAMES_RANGE: std::ops::RangeInclusive<usize> = 2..=60;
pub(crate) const PX_RANGE: std::ops::RangeInclusive<u16> = 64..=800;

impl Settings {
    /// The inputs a sub-row's build depends on (display-only choices are left out).
    pub(crate) fn key(&self, sub: S3Sub) -> String {
        let det = format!("{}|{}|{}|{}", self.own_bonds, self.auto_cutoff, self.cutoff_rel, self.clearance);
        match sub {
            S3Sub::Detect => det,
            S3Sub::Fields => format!("{det}|{:?}|{}", self.selected, self.psi_n),
            S3Sub::Elastic => format!("{det}|{:?}|{}|{}|{}|{:?}", self.selected, self.grid, self.extent, self.manual_b, self.manual_uv),
            S3Sub::Gif => format!("{det}|{}|{}|{}", self.gif_frames, self.gif_px, self.gif_delay_cs),
        }
    }

    /// Refuse inputs outside the supported ranges (with the reason) instead of clamping silently.
    pub(crate) fn validate(&self, sub: S3Sub) -> Result<(), String> {
        if !self.own_bonds && !self.auto_cutoff && !(self.cutoff_rel.is_finite() && CUTOFF_RANGE.contains(&self.cutoff_rel)) {
            return Err(format!(
                "bond cutoff must lie in [{}, {}] longest ideal bonds (got {}); at 1.0 the longest ideal bonds sit on the cutoff \
                 and are dropped by rounding",
                CUTOFF_RANGE.start(),
                CUTOFF_RANGE.end(),
                self.cutoff_rel
            ));
        }
        if self.clearance > MAX_CLEARANCE {
            return Err(format!("loop clearance must be at most {MAX_CLEARANCE} face rings"));
        }
        match sub {
            S3Sub::Detect => {}
            S3Sub::Fields if !(1..=12).contains(&self.psi_n) => return Err(String::from("ψ_n order must lie in 1..=12")),
            S3Sub::Fields => {}
            S3Sub::Elastic => {
                if !GRID_RANGE.contains(&self.grid) || !self.grid.is_multiple_of(2) {
                    return Err(format!("Stroh grid must be even and in 10..=80 (got {})", self.grid));
                }
                if !(self.extent.is_finite() && EXTENT_RANGE.contains(&self.extent)) {
                    return Err(format!("Stroh window half-width must lie in [2, 20] bonds (got {})", self.extent));
                }
                if self.manual_b && self.manual_uv == [0, 0] {
                    return Err(String::from("manual Burgers vector (0, 0) is not a dislocation"));
                }
                if self.manual_b && self.manual_uv.iter().any(|x| x.abs() > 4) {
                    return Err(String::from("manual Burgers components must lie in -4..=4"));
                }
            }
            S3Sub::Gif => {
                if !FRAMES_RANGE.contains(&self.gif_frames) {
                    return Err(format!("GIF frame count must lie in 2..=60 (got {})", self.gif_frames));
                }
                if !PX_RANGE.contains(&self.gif_px) {
                    return Err(format!("GIF size must lie in 64..=800 px (got {})", self.gif_px));
                }
                if !(1..=100).contains(&self.gif_delay_cs) {
                    return Err(String::from("GIF frame delay must lie in 1..=100 cs"));
                }
            }
        }
        Ok(())
    }

    /// Bond source, cutoff, clearance and the selected defect (shared by every sub-row).
    pub(crate) fn show_common(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        ui.label("Bond graph read from:");
        ch |= ui.radio_value(&mut self.own_bonds, false, "bare coordinates + cutoff").changed();
        ch |= ui.radio_value(&mut self.own_bonds, true, "the crystal's own bonds").changed();
        if !self.own_bonds {
            ch |= ui.checkbox(&mut self.auto_cutoff, "Automatic cutoff (between bond and next distance)").changed();
            if !self.auto_cutoff {
                ui.horizontal(|ui| {
                    ui.label("Cutoff / longest bond:");
                    ch |= ui.add(egui::DragValue::new(&mut self.cutoff_rel).range(CUTOFF_RANGE).speed(0.01)).changed();
                });
            }
        }
        ui.horizontal(|ui| {
            ui.label("Loop clearance (face rings):");
            ch |= ui.add(egui::DragValue::new(&mut self.clearance).range(0..=MAX_CLEARANCE)).changed();
        });
        let mut auto = self.selected.is_none();
        if ui.checkbox(&mut auto, "Pick the defect automatically (first non-zero b)").changed() {
            self.selected = if auto { None } else { Some(0) };
            ch = true;
        }
        if let Some(k) = &mut self.selected {
            ui.horizontal(|ui| {
                ui.label("Selected defect #:");
                ch |= ui.add(egui::DragValue::new(k).range(0..=999)).changed();
            });
        }
        ui.small("Click near a defect on the canvas to select it.");
        ch
    }

    pub(crate) fn show_fields(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        ui.separator();
        egui::ComboBox::from_id_salt("wb_s3_field").selected_text(self.field.label()).show_ui(ui, |ui| {
            for f in FieldKind::SHOWN {
                ch |= ui.selectable_value(&mut self.field, f, f.label()).changed();
            }
        });
        ch
    }

    pub(crate) fn save(&self) -> Value {
        json!({
            "own_bonds": self.own_bonds, "auto_cutoff": self.auto_cutoff, "cutoff_rel": self.cutoff_rel,
            "clearance": self.clearance, "selected": self.selected, "field": format!("{:?}", self.field),
            "psi_n": self.psi_n, "grid": self.grid, "extent": self.extent, "manual_b": self.manual_b,
            "manual_uv": self.manual_uv, "gif_frames": self.gif_frames, "gif_px": self.gif_px,
            "gif_delay_cs": self.gif_delay_cs,
        })
    }

    /// Lenient restore: missing or junk fields keep the current value, numbers are clamped into their ranges.
    pub(crate) fn restore(&mut self, v: &Value) {
        let b = |k: &str| v.get(k).and_then(Value::as_bool);
        let u = |k: &str| v.get(k).and_then(Value::as_u64);
        let f = |k: &str| v.get(k).and_then(Value::as_f64).filter(|x| x.is_finite());
        if let Some(x) = b("own_bonds") {
            self.own_bonds = x;
        }
        if let Some(x) = b("auto_cutoff") {
            self.auto_cutoff = x;
        }
        if let Some(x) = f("cutoff_rel") {
            self.cutoff_rel = x.clamp(*CUTOFF_RANGE.start(), *CUTOFF_RANGE.end());
        }
        if let Some(x) = u("clearance") {
            self.clearance = (x as usize).min(MAX_CLEARANCE);
        }
        match v.get("selected") {
            Some(Value::Null) => self.selected = None,
            Some(x) => {
                if let Some(k) = x.as_u64() {
                    self.selected = Some((k as usize).min(999));
                }
            }
            None => {}
        }
        if let Some(s) = v.get("field").and_then(Value::as_str)
            && let Some(k) = FieldKind::ALL.into_iter().find(|k| format!("{k:?}") == s)
        {
            self.field = if k.is_shown() { k } else { FieldKind::NyeFace };
        }
        if let Some(x) = u("psi_n") {
            self.psi_n = x.clamp(1, 12) as u32;
        }
        if let Some(x) = u("grid") {
            let g = (x as usize).clamp(*GRID_RANGE.start(), *GRID_RANGE.end());
            self.grid = g - g % 2;
        }
        if let Some(x) = f("extent") {
            self.extent = x.clamp(*EXTENT_RANGE.start(), *EXTENT_RANGE.end());
        }
        if let Some(x) = b("manual_b") {
            self.manual_b = x;
        }
        if let Some(a) = v.get("manual_uv").and_then(Value::as_array)
            && let [Some(p), Some(q)] = [a.first().and_then(Value::as_i64), a.get(1).and_then(Value::as_i64)]
        {
            self.manual_uv = [p.clamp(-4, 4) as i32, q.clamp(-4, 4) as i32];
        }
        if let Some(x) = u("gif_frames") {
            self.gif_frames = (x as usize).clamp(*FRAMES_RANGE.start(), *FRAMES_RANGE.end());
        }
        if let Some(x) = u("gif_px") {
            self.gif_px = x.clamp(64, 800) as u16;
        }
        if let Some(x) = u("gif_delay_cs") {
            self.gif_delay_cs = x.clamp(1, 100) as u16;
        }
    }
}
