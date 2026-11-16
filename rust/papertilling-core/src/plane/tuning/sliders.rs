//! Coupled constrained sliders for the free parameters of a lattice (`widgets.explore`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{AllowedInterval, SpecialPoint, allowed_interval_family, special_points_with_loci};
use crate::plane::loci::{Locus, supergroup_loci_for_family};
use crate::plane::tuning_eval::EvalFamily;
use crate::types::Recipe;

/// Nominal slider range `(lo, hi, step)` of a free parameter at its default value (`widgets._ranges`).
pub fn nominal_range(name: &str, default_value: f64) -> (f64, f64, f64) {
    match name {
        "b" => (0.4, 3.0, 0.01),
        "gamma" => (45.0, 135.0, 0.5),
        _ => (default_value - 0.5, default_value + 0.5, 0.002),
    }
}

/// Round `[a, b]` inward onto multiples of `step`, always preserving `x0` (`widgets._inward`).
pub fn inward_limits(a: f64, b: f64, x0: f64, step: f64) -> (f64, f64) {
    let a2 = ((a / step - 1e-9).ceil() * step * 1e10).round() / 1e10;
    let b2 = ((b / step + 1e-9).floor() * step * 1e10).round() / 1e10;
    (a2.min(x0), b2.max(x0))
}

/// Clamp `val` to `interval` and snap onto a `SpecialPoint` within `0.5 * step` (`widgets.explore`).
pub fn snap_to_special(val: f64, step: f64, special: &[SpecialPoint], interval: (f64, f64)) -> f64 {
    let v = val.clamp(interval.0, interval.1);
    for sp in special {
        if (v - sp.value).abs() <= 0.5 * step + 1e-12 && interval.0 <= sp.value && sp.value <= interval.1 {
            return sp.value;
        }
    }
    v
}

/// Value of a slider dragged to the raw position `raw`: clamped to `limits`, put on the step grid anchored at
/// `limits.0` (the ipywidgets slider grid; the ends themselves are always reachable), then snapped onto a special
/// value within half a step (`snap_to_special`). `NaN` goes to `limits.0`. The result always lies in `limits`.
pub fn slider_value(raw: f64, step: f64, special: &[SpecialPoint], limits: (f64, f64)) -> f64 {
    let (a, b) = limits;
    let q = if raw.is_nan() || raw <= a {
        a
    } else if raw >= b {
        b
    } else if step > 0.0 {
        let k = ((raw - a) / step).round();
        let mut q = ((a + k * step) * 1e12).round() / 1e12;
        if q > b {
            q = ((a + (k - 1.0) * step) * 1e12).round() / 1e12;
        }
        q.clamp(a, b)
    } else {
        raw
    };
    snap_to_special(q, step, special, limits)
}

/// Live state of one constrained parameter slider (`widgets.explore`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SliderSpec {
    pub name: String,
    pub nominal_range: (f64, f64),
    pub step: f64,
    pub limits: (f64, f64),
    pub allowed: AllowedInterval,
    pub special: Vec<SpecialPoint>,
}

/// Coupled multi-parameter constrained slider controller for a `Recipe`.
#[derive(Clone, Debug)]
pub struct SliderState {
    pub recipe: Recipe,
    pub values: BTreeMap<String, f64>,
    pub sliders: Vec<SliderSpec>,
    fam: EvalFamily,
    loci: Vec<Locus>,
}

impl SliderState {
    /// Initialise coupled sliders for all free parameters of `recipe` at its own values.
    pub fn new(recipe: Recipe) -> Result<Self, String> {
        Self::with_values(recipe, None)
    }

    /// As `new`, then move the parameters named in `values` there (other names are ignored). The nominal
    /// ranges stay those of the recipe's own values, as in the widget; the values are not clamped.
    pub fn with_values(recipe: Recipe, values: Option<&BTreeMap<String, f64>>) -> Result<Self, String> {
        let fam = EvalFamily::from_recipe(&recipe)?;
        let defaults = fam.values();
        let mut vals = defaults.clone();
        for (k, &v) in values.into_iter().flatten() {
            if let Some(slot) = vals.get_mut(k) {
                if !v.is_finite() {
                    return Err(format!("{k} = {v} is not a finite number"));
                }
                *slot = v;
            }
        }
        let loci = supergroup_loci_for_family(&fam).unwrap_or_default();
        let sliders = fam
            .names
            .iter()
            .map(|nm| {
                let (lo, hi, step) = nominal_range(nm, defaults[nm]);
                SliderSpec {
                    name: nm.clone(),
                    nominal_range: (lo, hi),
                    step,
                    limits: (lo, hi),
                    allowed: AllowedInterval { interval: (lo, hi), edges: (None, None), segments: Vec::new(), current_ok: true },
                    special: Vec::new(),
                }
            })
            .collect();
        let mut st = Self { recipe, values: vals, sliders, fam, loci };
        st.refresh_all()?;
        Ok(st)
    }

    fn refresh_all(&mut self) -> Result<(), String> {
        for s in &mut self.sliders {
            let (lo, hi) = s.nominal_range;
            let r = allowed_interval_family(&self.fam, &s.name, lo, hi, Some(&self.values), None)?;
            let sp = special_points_with_loci(&self.fam, &self.loci, &s.name, lo, hi, Some(&self.values))?;
            let (a, b) = if r.current_ok { r.interval } else { (lo, hi) };
            s.limits = if r.edges != (None, None) { inward_limits(a, b, self.values[&s.name], s.step) } else { (a, b) };
            s.allowed = r;
            s.special = sp;
        }
        Ok(())
    }

    /// The default values (those of the recipe).
    pub fn defaults(&self) -> BTreeMap<String, f64> {
        self.fam.values()
    }

    /// Look up the current slider specification for `name`.
    pub fn spec(&self, name: &str) -> Option<&SliderSpec> {
        self.sliders.iter().find(|s| s.name == name)
    }

    /// Move `name` to `raw_val` (clamping to current limits, snapping within `0.5 * step`, and refreshing all sliders).
    pub fn set_value(&mut self, name: &str, raw_val: f64) -> Result<f64, String> {
        let s = self.spec(name).ok_or_else(|| format!("unknown slider {name:?}"))?;
        if raw_val.is_nan() {
            return Err(format!("{name} = NaN is not a number"));
        }
        let snapped = snap_to_special(raw_val, s.step, &s.special, s.limits);
        self.values.insert(name.to_string(), snapped);
        self.refresh_all()?;
        Ok(snapped)
    }

    /// Move `name` as a slider dragged to `raw` would (`slider_value`: grid, limits, snap), refreshing all sliders.
    /// Returns the stored value; a move to the value already there changes nothing; `NaN` is an error.
    pub fn drag_to(&mut self, name: &str, raw: f64) -> Result<f64, String> {
        let s = self.spec(name).ok_or_else(|| format!("unknown slider {name:?}"))?;
        if raw.is_nan() {
            return Err(format!("{name} = NaN is not a number"));
        }
        let v = slider_value(raw, s.step, &s.special, s.limits);
        if self.values.get(name) == Some(&v) {
            return Ok(v);
        }
        self.values.insert(name.to_string(), v);
        self.refresh_all()?;
        Ok(v)
    }

    /// Back to the recipe's own values (the Reset button).
    pub fn reset(&mut self) -> Result<(), String> {
        self.values = self.fam.values();
        self.refresh_all()
    }
}
