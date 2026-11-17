use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use papertilling_core::dsymbol::{self, DSymbolKey};
use papertilling_core::plane::basis_reduce::is_geometrically_chiral;
use papertilling_core::plane::catalogue::{
    GroupCatalogue, NamedCatalogue, NamedLattice, parse_catalogue_str, parse_named_str,
};
use papertilling_core::plane::groups::{
    CrystalSystem, WALLPAPER_GROUPS, crystal_system, group_ops, is_chiral, label_point,
    wyckoff_table,
};
use papertilling_core::plane::tuning::SliderState;
use papertilling_core::types::{Cell2D, Frame2D, Recipe};

use crate::aliases::{AliasStore, LatticeAliasKey};
use crate::backend::{
    CatalogueItemUi, CatalogueRuleUi, DSymbolReadoutUi, GroupInfo, SliderParamUi, WyckoffRowUi,
};

mod tuned;
pub use tuned::{slider_params_ui, tune_item};

const POINTS_JSON: &str = include_str!("../../../papertilling/data/points.json");
const SHEET_JSON: &str = include_str!("../../../papertilling/data/sheet.json");
const MINIMAL_B_JSON: &str = include_str!("../../../papertilling/data/minimal_b.json");
const NAMED_JSON: &str = include_str!("../../../papertilling/data/named.json");

pub struct PlaneCatalogues {
    pub points: BTreeMap<String, GroupCatalogue>,
    pub sheet: BTreeMap<String, GroupCatalogue>,
    pub bonds: BTreeMap<String, GroupCatalogue>,
    pub named: NamedCatalogue,
}

pub fn shared_catalogues() -> &'static PlaneCatalogues {
    static CATALOGUES: OnceLock<PlaneCatalogues> = OnceLock::new();
    CATALOGUES.get_or_init(|| PlaneCatalogues {
        points: parse_catalogue_str(POINTS_JSON).expect("valid embedded points.json"),
        sheet: parse_catalogue_str(SHEET_JSON).expect("valid embedded sheet.json"),
        bonds: parse_catalogue_str(MINIMAL_B_JSON).expect("valid embedded minimal_b.json"),
        named: parse_named_str(NAMED_JSON).expect("valid embedded named.json"),
    })
}

pub fn build_groups_info(cats: &PlaneCatalogues) -> Vec<GroupInfo> {
    WALLPAPER_GROUPS
        .iter()
        .map(|&grp| {
            let sys_str = match crystal_system(grp).expect("valid wallpaper group") {
                CrystalSystem::Oblique => "oblique",
                CrystalSystem::Rectangular => "rectangular",
                CrystalSystem::Square => "square",
                CrystalSystem::Hexagonal => "hexagonal",
            };
            let ops = group_ops(grp).expect("valid group ops");
            let mut distinct_mf: BTreeSet<[[i32; 2]; 2]> = BTreeSet::new();
            let mut max_rot = 1_u8;
            for op in &ops {
                distinct_mf.insert(op.mf);
                let det = op.mf[0][0] * op.mf[1][1] - op.mf[0][1] * op.mf[1][0];
                if det == 1 {
                    let tr = op.mf[0][0] + op.mf[1][1];
                    let ord = match tr {
                        2 => 1,
                        -2 => 2,
                        -1 => 3,
                        0 => 4,
                        1 => 6,
                        _ => 1,
                    };
                    max_rot = max_rot.max(ord);
                }
            }
            let p_cnt = cats.points.get(grp).map_or(0, |c| c.all.len());
            let s_cnt = cats.sheet.get(grp).map_or(0, |c| c.all.len());
            let b_cnt = cats.bonds.get(grp).map_or(0, |c| c.all.len());
            GroupInfo {
                symbol: grp,
                crystal_system: sys_str,
                point_order: distinct_mf.len() as u8,
                max_rotation: max_rot,
                has_reflection_or_glide: !is_chiral(grp).expect("valid group chirality"),
                counts: [p_cnt, s_cnt, b_cnt],
            }
        })
        .collect()
}

pub fn format_dsymbol_key(key: &DSymbolKey) -> String {
    let rows: Vec<String> = key
        .iter()
        .map(|&(m01, m12, s0, s1, s2)| format!("({m01},{m12}:{s0},{s1},{s2})"))
        .collect();
    format!("[|D|={}] {}", key.len(), rows.join(" "))
}

pub fn coordination_string(cell: &Cell2D) -> String {
    if cell.basis.is_empty() {
        return String::from("-");
    }
    let mut darts: BTreeSet<(usize, usize, i32, i32)> = BTreeSet::new();
    for b in &cell.bonds {
        darts.insert((b.i, b.j, b.n1, b.n2));
        darts.insert((b.j, b.i, -b.n1, -b.n2));
    }
    let mut deg = vec![0_usize; cell.basis.len()];
    for &(i, _, _, _) in &darts {
        if i < deg.len() {
            deg[i] += 1;
        }
    }
    let uniq: BTreeSet<usize> = deg.into_iter().collect();
    uniq.into_iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub fn occupied_wyckoff_letters(recipe: &Recipe) -> Vec<char> {
    let mut out = Vec::new();
    for &pt in &recipe.sites {
        if let Ok(lbl) = label_point(&recipe.group, pt)
            && !out.contains(&lbl.letter)
        {
            out.push(lbl.letter);
        }
    }
    out
}

pub fn build_named_lattices(cats: &PlaneCatalogues, aliases: &AliasStore) -> Vec<CatalogueItemUi> {
    cats.named
        .named
        .iter()
        .enumerate()
        .map(|(idx, nl)| {
            let dsymbol_key = format_dsymbol_key(&nl.dsymbol_key);
            let default_label = nl.name.clone();
            let alias_key = LatticeAliasKey::new(&nl.group, "named", idx + 1, &dsymbol_key);
            let label = aliases.get(&alias_key).map_or_else(|| default_label.clone(), str::to_owned);
            CatalogueItemUi {
                index: idx + 1,
                label,
                default_label,
                alias_key,
                named_key: Some(nl.key.clone()),
                group: nl.group.clone(),
                sheet_group: nl.sheet_group.clone(),
                sites_per_cell: nl.cell.basis.len(),
                site_orbits: nl.rank[0],
                bond_orbits: nl.rank[1],
                coordination: coordination_string(&nl.cell),
                dsymbol_key,
                network_key: format_dsymbol_key(&nl.network_key),
                symmetry_index: nl.symmetry_index,
                chiral: is_geometrically_chiral(&nl.cell).unwrap_or(false),
                topologically_chiral: dsymbol::is_topologically_chiral(&nl.network_key).unwrap_or(false),
                occupied_wyckoff: occupied_wyckoff_letters(&nl.recipe),
                recipe: nl.recipe.clone(),
                frame: nl.frame,
            }
        })
        .collect()
}

fn match_named_in_group<'a>(
    named_list: &'a [NamedLattice],
    group: &str,
    canonical_key: &DSymbolKey,
    network_key: &DSymbolKey,
) -> Option<&'a NamedLattice> {
    named_list
        .iter()
        .find(|nl| nl.group == group && &nl.dsymbol_key == canonical_key)
        .or_else(|| {
            named_list
                .iter()
                .find(|nl| nl.group == group && &nl.network_key == network_key)
        })
}

pub fn catalogue_for_group(
    cats: &PlaneCatalogues,
    group: &str,
    rule: CatalogueRuleUi,
    aliases: &AliasStore,
) -> Vec<CatalogueItemUi> {
    let map = match rule {
        CatalogueRuleUi::Points => &cats.points,
        CatalogueRuleUi::Sheet => &cats.sheet,
        CatalogueRuleUi::Bonds => &cats.bonds,
    };
    let Some(cat) = map.get(group) else {
        return Vec::new();
    };
    let mut seen_named: BTreeSet<String> = BTreeSet::new();
    cat.all
        .iter()
        .enumerate()
        .map(|(idx, entry)| {
            let cell = entry.family.cell();
            let matched = match_named_in_group(
                &cats.named.named,
                group,
                &entry.canonical_key,
                &entry.network_key,
            );
            let (default_label, named_key, sheet_group) = if let Some(nl) = matched {
                let first_use = seen_named.insert(nl.key.clone());
                let lbl = if first_use {
                    nl.name.clone()
                } else {
                    format!("{} (#{})", nl.name, idx + 1)
                };
                (lbl, Some(nl.key.clone()), nl.sheet_group.clone())
            } else {
                (
                    format!("{group} #{} ({},{})", idx + 1, entry.rank[0], entry.rank[1]),
                    None,
                    None,
                )
            };
            let dsymbol_key = format_dsymbol_key(&entry.canonical_key);
            let alias_key = LatticeAliasKey::new(&entry.recipe.group, rule.slug(), idx + 1, &dsymbol_key);
            let label = aliases.get(&alias_key).map_or_else(|| default_label.clone(), str::to_owned);
            CatalogueItemUi {
                index: idx + 1,
                label,
                default_label,
                alias_key,
                named_key,
                group: entry.recipe.group.clone(),
                sheet_group,
                sites_per_cell: cell.basis.len(),
                site_orbits: entry.rank[0],
                bond_orbits: entry.rank[1],
                coordination: coordination_string(&cell),
                dsymbol_key,
                network_key: format_dsymbol_key(&entry.network_key),
                symmetry_index: entry.symmetry_index,
                chiral: is_geometrically_chiral(&cell).unwrap_or(false),
                topologically_chiral: dsymbol::is_topologically_chiral(&entry.network_key).unwrap_or(false),
                occupied_wyckoff: occupied_wyckoff_letters(&entry.recipe),
                recipe: entry.recipe.clone(),
                frame: Frame2D::default(),
            }
        })
        .collect()
}

pub fn wyckoff_rows_for_group(group: &str, occupied: &[char]) -> Vec<WyckoffRowUi> {
    let Ok(rows) = wyckoff_table(group) else {
        return Vec::new();
    };
    rows.into_iter()
        .map(|r| WyckoffRowUi {
            letter: r.letter,
            multiplicity: r.multiplicity,
            site_symmetry: r.site_symmetry,
            coords_expr: r.coords.into_iter().next().unwrap_or_default(),
            occupied: occupied.contains(&r.letter),
        })
        .collect()
}

fn values_map(params: &[(String, f64)]) -> BTreeMap<String, f64> {
    params.iter().map(|(k, v)| (k.clone(), *v)).collect()
}

/// Sliders of `item` at `current_vals` (names not free in `item` are ignored): Python's ranges, limits, forbidden
/// spans and special values (`SliderState`, the port of `widgets.explore`).
pub fn sliders_for_item(
    item: &CatalogueItemUi,
    current_vals: &[(String, f64)],
) -> Vec<SliderParamUi> {
    SliderState::with_values(item.recipe.clone(), Some(&values_map(current_vals)))
        .map(|st| slider_params_ui(&st))
        .unwrap_or_default()
}

/// Cell and status line of `item` at `params` (`widgets.redraw`); see `tuned::readout`.
pub fn build_cell_for_item(
    item: &CatalogueItemUi,
    params: &[(String, f64)],
) -> (Cell2D, DSymbolReadoutUi) {
    let st = (!params.is_empty())
        .then(|| SliderState::with_values(item.recipe.clone(), Some(&values_map(params))).ok())
        .flatten();
    let t = tune_item(item, st.as_ref());
    (t.cell, t.readout)
}

#[cfg(test)]
mod tests {
    use crate::backend::{AppBackend, CatalogueRuleUi};

    #[test]
    fn live_plane_groups_named_wyckoff_and_sliders() {
        let backend = AppBackend::new();
        let groups = backend.groups();
        assert_eq!(groups.len(), 17);
        let orders: Vec<u8> = groups.iter().map(|g| g.point_order).collect();
        assert_eq!(orders, vec![1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 8, 8, 3, 6, 6, 6, 12]);
        let rots: Vec<u8> = groups.iter().map(|g| g.max_rotation).collect();
        assert_eq!(rots, vec![1, 2, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 3, 3, 3, 6, 6]);
        assert!(groups.iter().all(|g| g.counts.iter().all(|&c| c >= 1)));

        assert_eq!(backend.named_lattices().len(), 21);
        let kagome = backend.find_named("kagome").expect("kagome present");
        assert_eq!(kagome.group, "p6m");
        assert_eq!(kagome.sites_per_cell, 3);
        assert_eq!(kagome.coordination, "4");
        assert_eq!(kagome.occupied_wyckoff, vec!['c']);
        let wyck = backend.wyckoff_table("p6m", &kagome.occupied_wyckoff);
        assert!(wyck.iter().any(|w| w.letter == 'c' && w.multiplicity == 3 && w.occupied));
        assert!(backend.sliders_for_item(kagome, &[]).is_empty());

        let p6m_sheet = backend.catalogue_for_group("p6m", CatalogueRuleUi::Sheet);
        assert!(p6m_sheet.iter().any(|it| it.named_key.as_deref() == Some("kagome")));

        let oblique = backend.find_named("oblique").expect("oblique present");
        let sliders = backend.sliders_for_item(oblique, &[]);
        assert_eq!(sliders.len(), 2); // b and gamma
        assert!(sliders.iter().any(|s| !s.special_ticks.is_empty()));
        let (cell, ro) = backend.build_cell(oblique, &[("b".into(), 1.0), ("gamma".into(), 90.0)]);
        assert_eq!(cell.basis.len(), 1);
        // a = b, gamma = 90: the square lattice. As widgets.explore (tests/test_widgets.py::test_slider_callback:
        // "p4m" in the status, in red) the status is not OK on a special locus.
        assert!(!ro.status_ok && ro.active_group == "p4m", "{}", ro.status_message);
        assert!(ro.status_message.starts_with("group is now p4m at the special value"), "{}", ro.status_message);
        let (_, ro0) = backend.build_cell(oblique, &[]);
        assert!(ro0.status_ok && ro0.active_group == "p2", "{}", ro0.status_message);
    }
}
