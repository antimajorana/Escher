//! The Gallery lattice at the slider values: the item handed to the other tabs (its recipe rebuilt at the values,
//! `Family.recipe` / `box.lattice()`), its cell, and the status line (the checks of `widgets.explore`, i.e.
//! `lattice.check_family`: sites coincide, exact group, planar, bond through a site, same D-symbol key).

use std::collections::BTreeMap;

use papertilling_core::dsymbol;
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::tuning::SliderState;
use papertilling_core::types::Cell2D;

use super::{coordination_string, format_dsymbol_key, occupied_wyckoff_letters};
use crate::backend::{
    CatalogueItemUi, DSymbolReadoutUi, ForbiddenSpanUi, SliderParamUi, SpecialTickUi,
};

/// The lattice at the current values.
pub struct TunedLattice {
    /// The item at these values (the catalogue item itself at the default values). `None` when the values cannot
    /// be written as a recipe that rebuilds the same family (then the caller keeps the last exportable item).
    pub item: Option<CatalogueItemUi>,
    pub cell: Cell2D,
    pub readout: DSymbolReadoutUi,
}

/// UI rows of the sliders of `st` (limits, forbidden spans and special ticks at the current values).
pub fn slider_params_ui(st: &SliderState) -> Vec<SliderParamUi> {
    let defaults = st.defaults();
    st.sliders
        .iter()
        .map(|s| SliderParamUi {
            name: s.name.clone(),
            value: st.values[&s.name],
            default_value: defaults[&s.name],
            nominal_min: s.nominal_range.0.min(s.limits.0),
            nominal_max: s.nominal_range.1.max(s.limits.1),
            allowed_min: s.limits.0,
            allowed_max: s.limits.1,
            step: s.step,
            special_ticks: s
                .special
                .iter()
                .map(|sp| SpecialTickUi {
                    value: sp.value,
                    supergroup: sp.group.clone(),
                    equation: sp.equations.join(", "),
                })
                .collect(),
            forbidden: s
                .allowed
                .segments
                .iter()
                .map(|g| ForbiddenSpanUi { lo: g.lo, hi: g.hi, reason: g.reason.clone() })
                .collect(),
        })
        .collect()
}

/// Python's `{:.6g}` for the status line.
pub fn fmt_g6(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return format!("{x}");
    }
    let exp = x.abs().log10().floor() as i32;
    if !(-4..6).contains(&exp) {
        return format!("{x:e}");
    }
    let s = format!("{:.*}", (5 - exp).max(0) as usize, x);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// `"t1 = -0.5 (t1 = -1/2)"` for the first parameter sitting on one of its special values (`widgets.redraw`).
pub fn special_hit(st: &SliderState) -> Option<String> {
    st.sliders.iter().find_map(|s| {
        let v = st.values[&s.name];
        s.special.iter().find(|sp| (sp.value - v).abs() < 1e-9).map(|sp| {
            format!("{} = {} ({})", s.name, fmt_g6(sp.value), sp.equations.join(", "))
        })
    })
}

/// Group of the bare lattice and its sites per primitive cell (`Family.group_at`).
fn group_at(cell: &Cell2D) -> Result<(String, usize), String> {
    Ok((cell.try_identify_wallpaper_group(false)?, cell.try_primitive_cell(false)?.basis.len()))
}

/// Status of `fam` (the base family at some values) against the reference `base` (`widgets.redraw`).
pub fn readout(base: &CatalogueItemUi, fam: &LatticeFamily, cell: &Cell2D, on_special: Option<&str>) -> DSymbolReadoutUi {
    let sites_coincide = cell.basis.is_empty() || cell.min_separation() < 1e-6;
    let found = if sites_coincide { Ok(None) } else { group_at(cell).map(Some) };
    let key = fam.dsymbol().ok().and_then(|d| d.canonical_key().ok());
    let (dsymbol_key, network_key, symmetry_index) = match &key {
        Some(k) => (
            format_dsymbol_key(k),
            dsymbol::network_key(k).map_or_else(|_| base.network_key.clone(), |n| format_dsymbol_key(&n)),
            dsymbol::symmetry_index(k).unwrap_or(base.symmetry_index),
        ),
        None => (base.dsymbol_key.clone(), base.network_key.clone(), base.symmetry_index),
    };
    let mut out = DSymbolReadoutUi {
        dsymbol_key,
        network_key,
        symmetry_index,
        active_group: String::from("?"),
        status_ok: false,
        status_message: String::new(),
    };
    let group = match found {
        Ok(g) => g,
        Err(err) => {
            out.status_message = format!("check failed here ({err}); move the slider");
            return out;
        }
    };
    let exact = group.as_ref().is_some_and(|(g, mp)| *g == base.group && *mp == cell.basis.len());
    if let Some((g, _)) = &group {
        out.active_group.clone_from(g);
    }
    let planar = !cell.has_crossings();
    let bond_through_site = sites_coincide || cell.clearance() < 1e-6;
    let mut bad: Vec<String> = Vec::new();
    if sites_coincide {
        bad.push(String::from("two sites coincide"));
    } else if !exact {
        let g = out.active_group.as_str();
        let more = if g == base.group { " with more translations" } else { "" };
        let at = on_special.map(|s| format!(" at the special value {s}")).unwrap_or_default();
        bad.push(format!("group is now {g}{more}{at}: a lattice of a larger group (special locus)"));
    }
    if !planar {
        bad.push(String::from("bonds cross"));
    }
    if bond_through_site {
        bad.push(String::from("a bond passes through a site"));
    }
    if key.is_some() && out.dsymbol_key != base.dsymbol_key && planar {
        bad.push(String::from("different tiling (D-symbol changed)"));
    }
    out.status_ok = bad.is_empty();
    out.status_message = if bad.is_empty() {
        format!("same lattice: group {}, planar, same D-symbol", out.active_group)
    } else {
        bad.join("; ")
    };
    out
}

fn empty_cell() -> Cell2D {
    Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: Vec::new(), bonds: Vec::new() }
}

/// The base item's lattice at the values of `st` (`None`: the default values).
pub fn tune_item(base: &CatalogueItemUi, st: Option<&SliderState>) -> TunedLattice {
    let fam0 = match LatticeFamily::from_recipe(&base.recipe, Some(base.frame)) {
        Ok(f) => f,
        Err(err) => {
            let readout = DSymbolReadoutUi {
                dsymbol_key: base.dsymbol_key.clone(),
                network_key: base.network_key.clone(),
                symmetry_index: base.symmetry_index,
                active_group: String::from("?"),
                status_ok: false,
                status_message: format!("invalid recipe: {err}"),
            };
            return TunedLattice { item: Some(base.clone()), cell: empty_cell(), readout };
        }
    };
    let values: BTreeMap<String, f64> = st.map(|s| s.values.clone()).unwrap_or_default();
    let at_defaults = st.is_none_or(|s| s.values == s.defaults());
    let fam = fam0.with_values(values.iter().map(|(k, &v)| (k.as_str(), v))).unwrap_or(fam0);
    let cell = fam.cell();
    let hit = st.and_then(special_hit);
    let readout = readout(base, &fam, &cell, hit.as_deref());
    let item = if at_defaults { Some(base.clone()) } else { export_item(base, &fam, &cell, &readout) };
    TunedLattice { item, cell, readout }
}

/// `base` with the recipe of `fam` (rebuilt and checked to give the same family at the same values) and the
/// quantities that follow from it.
fn export_item(base: &CatalogueItemUi, fam: &LatticeFamily, cell: &Cell2D, ro: &DSymbolReadoutUi) -> Option<CatalogueItemUi> {
    let recipe = fam.to_recipe();
    let back = LatticeFamily::from_recipe(&recipe, Some(base.frame)).ok()?;
    if back.names() != fam.names() || back.rank() != fam.rank() {
        return None;
    }
    let vb = back.values();
    if fam.values().iter().any(|(k, v)| vb.get(k).is_none_or(|b| (b - v).abs() > 1e-9)) {
        return None;
    }
    Some(CatalogueItemUi {
        occupied_wyckoff: occupied_wyckoff_letters(&recipe),
        recipe,
        sites_per_cell: cell.basis.len(),
        coordination: coordination_string(cell),
        dsymbol_key: ro.dsymbol_key.clone(),
        network_key: ro.network_key.clone(),
        symmetry_index: ro.symmetry_index,
        ..base.clone()
    })
}
