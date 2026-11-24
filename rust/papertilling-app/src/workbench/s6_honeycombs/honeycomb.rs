//! Regular `{p,q,r}` (S³ polychora, the cubic E³ honeycomb, compact H³ growth, and paracompact H³ ball truncations)
//! and the four Euclidean honeycombs: controls, live `classify_schlafli` badge, and core growth/ball dispatch.

use papertilling_core::growth::h3_complex::{IDEAL_VERTEX_REASON, INFINITE_CELL_REASON};
use papertilling_core::growth::{GrowthCaps, SeedKind};
use papertilling_core::honeycomb3d::growth_build::{MAX_GROWTH_CELLS, MAX_GROWTH_LAYERS};
use papertilling_core::honeycomb3d::{
    Classification, EuclideanKind, GrowthRule, HoneycombKind, HoneycombRegime, classify_schlafli, grow_honeycomb,
    paracompact_ball,
};
use serde_json::{Value, json};

use super::S6Job;
use super::honeycomb_present::{present_growth, present_paracompact_ball, rule_label, seed_label};
use super::present::Built;

/// Range offered for each Schläfli entry (wide enough that degenerate and hyperideal symbols are reachable).
pub(crate) const PQR_RANGE: std::ops::RangeInclusive<u32> = 2..=8;

pub(crate) const RADIUS_CENTI_MIN: u32 = 30;
pub(crate) const RADIUS_CENTI_MAX: u32 = 300;
pub(crate) const RADIUS_CENTI_DEFAULT: u32 = 100;

pub(crate) const EUCLID_KINDS: [EuclideanKind; 4] = [
    EuclideanKind::Cubic,
    EuclideanKind::BitruncatedCubic,
    EuclideanKind::RhombicDodecahedral,
    EuclideanKind::AlternatedCubic,
];

pub(crate) const SEEDS_3D: [SeedKind; 4] = [SeedKind::Vertex, SeedKind::Edge, SeedKind::Face, SeedKind::Cell];
pub(crate) const RULES_3D: [GrowthRule; 2] = [GrowthRule::Touching, GrowthRule::FaceShells];

/// Short badge of a regime (never equal to a sub-row label).
pub(crate) fn regime_badge(r: HoneycombRegime) -> &'static str {
    match r {
        HoneycombRegime::SphericalS3 => "regime: spherical (S³ polychoron)",
        HoneycombRegime::EuclideanE3 => "regime: flat (E³ honeycomb)",
        HoneycombRegime::HyperbolicCompactH3 => "regime: compact hyperbolic (H³)",
        HoneycombRegime::HyperbolicParacompactH3 => "regime: paracompact hyperbolic (H³)",
        HoneycombRegime::HyperidealRefused => "refused: hyperideal",
        HoneycombRegime::DegenerateRefused => "refused: degenerate",
    }
}

fn show_seed_rule_layers(ui: &mut egui::Ui, seed: &mut SeedKind, rule: &mut GrowthRule, layers: &mut u32) -> bool {
    let mut ch = false;
    ui.horizontal_wrapped(|ui| {
        ui.label("Seed:");
        for s in SEEDS_3D {
            ch |= ui.selectable_value(seed, s, seed_label(s)).changed();
        }
    });
    ui.horizontal_wrapped(|ui| {
        ui.label("Rule:");
        for r in RULES_3D {
            ch |= ui.selectable_value(rule, r, rule_label(r)).changed();
        }
    });
    let mut lay = (*layers).min(MAX_GROWTH_LAYERS);
    ui.horizontal(|ui| {
        ui.label("Layers:");
        if ui.add(egui::Slider::new(&mut lay, 0..=MAX_GROWTH_LAYERS)).changed() {
            *layers = lay;
            ch = true;
        }
    });
    ch
}

fn parse_seed(s: &str) -> Option<SeedKind> {
    SEEDS_3D.into_iter().find(|k| format!("{k:?}").eq_ignore_ascii_case(s))
}

fn parse_rule(s: &str) -> Option<GrowthRule> {
    match s {
        "Touching" | "touching" => Some(GrowthRule::Touching),
        "FaceShells" | "face_shells" => Some(GrowthRule::FaceShells),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SchlafliState {
    pub(crate) p: u32,
    pub(crate) q: u32,
    pub(crate) r: u32,
    pub(crate) seed: SeedKind,
    pub(crate) rule: GrowthRule,
    pub(crate) layers: u32,
    pub(crate) radius_centi: u32,
    pub(crate) surrounding_cells: bool,
}

impl Default for SchlafliState {
    fn default() -> Self {
        Self {
            p: 4,
            q: 3,
            r: 4,
            seed: SeedKind::Cell,
            rule: GrowthRule::Touching,
            layers: 1,
            radius_centi: RADIUS_CENTI_DEFAULT,
            surrounding_cells: false,
        }
    }
}

impl SchlafliState {
    pub(crate) fn classify(&self) -> Classification {
        classify_schlafli(self.p, self.q, self.r)
    }

    pub(crate) fn job(&self) -> S6Job {
        let c = self.classify();
        if c.regime == HoneycombRegime::HyperbolicParacompactH3 {
            if !self.surrounding_cells && !c.has_infinite_euclidean_cells {
                return S6Job::SchlafliGrowth {
                    p: self.p, q: self.q, r: self.r, seed: SeedKind::Cell, rule: GrowthRule::Touching, layers: 0,
                };
            }
            S6Job::ParacompactBall {
                seed_only: !self.surrounding_cells,
                p: self.p,
                q: self.q,
                r: self.r,
                radius_centi: self.radius_centi.clamp(RADIUS_CENTI_MIN, RADIUS_CENTI_MAX),
            }
        } else {
            S6Job::SchlafliGrowth {
                p: self.p,
                q: self.q,
                r: self.r,
                seed: self.seed,
                rule: self.rule,
                layers: self.layers.min(MAX_GROWTH_LAYERS),
            }
        }
    }

    pub(crate) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        ui.horizontal(|ui| {
            for (name, v) in [("p", &mut self.p), ("q", &mut self.q), ("r", &mut self.r)] {
                ui.label(name);
                ch |= ui.add(egui::DragValue::new(v).range(PQR_RANGE).speed(0.05)).changed();
            }
        });
        let c = self.classify();
        let col = if c.is_buildable() { egui::Color32::from_rgb(39, 174, 96) } else { egui::Color32::from_rgb(192, 57, 43) };
        ui.colored_label(col, regime_badge(c.regime));
        ui.add(egui::Label::new(egui::RichText::new(&c.reason).small()).wrap());
        if !c.is_buildable() {
            return ch;
        }
        if c.regime == HoneycombRegime::HyperbolicParacompactH3 {
            let why = if c.has_infinite_euclidean_cells { INFINITE_CELL_REASON } else { IDEAL_VERTEX_REASON };
            ui.colored_label(egui::Color32::from_rgb(180, 95, 20), egui::RichText::new(why).small());
            ch |= ui.checkbox(&mut self.surrounding_cells, "Include surrounding cells (radius preview)").changed();
            if !self.surrounding_cells && !c.has_infinite_euclidean_cells {
                ui.small("Layer 0 · one complete polyhedron");
                return ch;
            }
            if !self.surrounding_cells {
                ui.small("Layer 0 · one infinite cell, showing faces within the chosen radius.");
            }
            let mut rad = f64::from(self.radius_centi.clamp(RADIUS_CENTI_MIN, RADIUS_CENTI_MAX)) / 100.0;
            ui.horizontal(|ui| {
                ui.label("Ball radius R:");
                if ui.add(egui::Slider::new(&mut rad, 0.30..=3.00).fixed_decimals(2)).changed() {
                    let next = ((rad * 100.0).round() as u32).clamp(RADIUS_CENTI_MIN, RADIUS_CENTI_MAX);
                    if next != self.radius_centi {
                        self.radius_centi = next;
                        ch = true;
                    }
                }
            });
        } else {
            ch |= show_seed_rule_layers(ui, &mut self.seed, &mut self.rule, &mut self.layers);
        }
        ch
    }

    pub(crate) fn save(&self) -> Value {
        json!({
            "p": self.p,
            "q": self.q,
            "r": self.r,
            "seed": format!("{:?}", self.seed),
            "rule": format!("{:?}", self.rule),
            "layers": self.layers,
            "radius_centi": self.radius_centi,
            "surrounding_cells": self.surrounding_cells,
        })
    }

    pub(crate) fn restore(&mut self, v: &Value) {
        let get = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x.min(u64::from(u32::MAX)) as u32);
        let clamp_pqr = |x: u32| x.clamp(*PQR_RANGE.start(), *PQR_RANGE.end());
        if let Some(x) = get("p") {
            self.p = clamp_pqr(x);
        }
        if let Some(x) = get("q") {
            self.q = clamp_pqr(x);
        }
        if let Some(x) = get("r") {
            self.r = clamp_pqr(x);
        }
        if let Some(s) = v.get("seed").and_then(Value::as_str).and_then(parse_seed) {
            self.seed = s;
        }
        if let Some(r) = v.get("rule").and_then(Value::as_str).and_then(parse_rule) {
            self.rule = r;
        }
        if let Some(x) = get("layers").or_else(|| get("shells")) {
            self.layers = x.min(MAX_GROWTH_LAYERS);
        }
        // Older saved paracompact views always showed the surrounding ball.
        self.surrounding_cells = v.get("surrounding_cells").and_then(Value::as_bool)
            .unwrap_or(self.classify().regime == HoneycombRegime::HyperbolicParacompactH3);
        if let Some(x) = get("radius_centi") {
            self.radius_centi = x.clamp(RADIUS_CENTI_MIN, RADIUS_CENTI_MAX);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EuclidState {
    pub(crate) kind: EuclideanKind,
    pub(crate) seed: SeedKind,
    pub(crate) rule: GrowthRule,
    pub(crate) layers: u32,
}

impl Default for EuclidState {
    fn default() -> Self {
        Self {
            kind: EuclideanKind::Cubic,
            seed: SeedKind::Vertex,
            rule: GrowthRule::Touching,
            layers: 1,
        }
    }
}

impl EuclidState {
    pub(crate) fn job(&self) -> S6Job {
        S6Job::Euclid {
            kind: self.kind,
            seed: self.seed,
            rule: self.rule,
            layers: self.layers.min(MAX_GROWTH_LAYERS),
        }
    }

    pub(crate) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        egui::ComboBox::from_id_salt("wb_s6_euclid_kind").selected_text(self.kind.label()).show_ui(ui, |ui| {
            for k in EUCLID_KINDS {
                ch |= ui.selectable_value(&mut self.kind, k, k.label()).changed();
            }
        });
        ch |= show_seed_rule_layers(ui, &mut self.seed, &mut self.rule, &mut self.layers);
        ch
    }

    pub(crate) fn save(&self) -> Value {
        json!({
            "kind": format!("{:?}", self.kind),
            "seed": format!("{:?}", self.seed),
            "rule": format!("{:?}", self.rule),
            "layers": self.layers,
        })
    }

    pub(crate) fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        if let Some(s) = v.get("kind").and_then(Value::as_str) {
            match EUCLID_KINDS.into_iter().find(|k| format!("{k:?}") == s) {
                Some(k) => self.kind = k,
                None => notes.push(format!("S6 panel: unknown Euclidean honeycomb {s:?} (kept current)")),
            }
        }
        if let Some(s) = v.get("seed").and_then(Value::as_str).and_then(parse_seed) {
            self.seed = s;
        }
        if let Some(r) = v.get("rule").and_then(Value::as_str).and_then(parse_rule) {
            self.rule = r;
        }
        let get = |k: &str| v.get(k).and_then(Value::as_u64);
        if let Some(x) = get("layers").or_else(|| get("shells")) {
            self.layers = x.min(u64::from(MAX_GROWTH_LAYERS)) as u32;
        }
    }
}

/// Grow `{p,q,r}` (S³, E³ `{4,3,4}`, or compact H³); a refused symbol returns the core's `[PROVED]` reason.
pub(crate) fn build_schlafli_growth(
    p: u32,
    q: u32,
    r: u32,
    seed: SeedKind,
    rule: GrowthRule,
    layers: u32,
) -> Result<Built, String> {
    let c = classify_schlafli(p, q, r);
    if !c.is_buildable() {
        return Err(c.reason);
    }
    if layers > MAX_GROWTH_LAYERS {
        return Err(format!("layers = {layers} exceeds the maximum {MAX_GROWTH_LAYERS}"));
    }
    let caps = GrowthCaps { max_layers: layers, max_cells: MAX_GROWTH_CELLS };
    let g = grow_honeycomb(HoneycombKind::Schlafli(p, q, r), seed, rule, caps)?;
    Ok(present_growth(g, Some(&c), layers, MAX_GROWTH_CELLS))
}

/// Build the paracompact ball truncation of `{p,q,r}` at radius `radius_centi / 100.0`.
pub(crate) fn build_paracompact(p: u32, q: u32, r: u32, radius_centi: u32, seed_only: bool) -> Result<Built, String> {
    let c = classify_schlafli(p, q, r);
    if !c.is_buildable() {
        return Err(c.reason);
    }
    if !(RADIUS_CENTI_MIN..=RADIUS_CENTI_MAX).contains(&radius_centi) {
        return Err(format!(
            "radius_centi = {radius_centi} outside allowed {RADIUS_CENTI_MIN}..={RADIUS_CENTI_MAX}"
        ));
    }
    let radius = f64::from(radius_centi) / 100.0;
    let b = if seed_only {
        papertilling_core::honeycomb3d::paracompact_ball::single_cell_patch(p, q, r, radius)
    } else {
        paracompact_ball(p, q, r, radius)
    }?;
    let mut built = present_paracompact_ball(b, &c);
    if seed_only {
        built.rows.insert(0, ("Layer 0".into(), "One infinite cell (visible faces only)".into()));
    }
    Ok(built)
}

pub(crate) fn build_euclid(kind: EuclideanKind, seed: SeedKind, rule: GrowthRule, layers: u32) -> Result<Built, String> {
    if layers > MAX_GROWTH_LAYERS {
        return Err(format!("layers = {layers} exceeds the maximum {MAX_GROWTH_LAYERS}"));
    }
    let caps = GrowthCaps { max_layers: layers, max_cells: MAX_GROWTH_CELLS };
    let g = grow_honeycomb(HoneycombKind::Euclid(kind), seed, rule, caps)?;
    Ok(present_growth(g, None, layers, MAX_GROWTH_CELLS))
}
