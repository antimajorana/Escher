use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use papertilling_core::dsymbol::{self, DSymbolKey};
use papertilling_core::plane::catalogue::{
    GroupCatalogue, NamedCatalogue, NamedLattice, parse_catalogue_str, parse_named_str,
};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::groups::{
    CrystalSystem, WALLPAPER_GROUPS, crystal_system, group_ops, is_chiral, label_point,
    wyckoff_table,
};
use papertilling_core::plane::tuning::{allowed_interval, inward_limits, special_points};
use papertilling_core::types::{Cell2D, Frame2D, Recipe};

use crate::backend::{
    CatalogueItemUi, CatalogueRuleUi, DSymbolReadoutUi, ForbiddenSpanUi, GroupInfo, SliderParamUi,
    SpecialTickUi, WyckoffRowUi,
};

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

pub fn build_named_lattices(cats: &PlaneCatalogues) -> Vec<CatalogueItemUi> {
    cats.named
        .named
        .iter()
        .enumerate()
        .map(|(idx, nl)| CatalogueItemUi {
            index: idx + 1,
            label: nl.name.clone(),
            named_key: Some(nl.key.clone()),
            group: nl.group.clone(),
            sheet_group: nl.sheet_group.clone(),
            sites_per_cell: nl.cell.basis.len(),
            site_orbits: nl.rank[0],
            bond_orbits: nl.rank[1],
            coordination: coordination_string(&nl.cell),
            dsymbol_key: format_dsymbol_key(&nl.dsymbol_key),
            network_key: format_dsymbol_key(&nl.network_key),
            symmetry_index: nl.symmetry_index,
            occupied_wyckoff: occupied_wyckoff_letters(&nl.recipe),
            recipe: nl.recipe.clone(),
            frame: nl.frame,
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
            let (label, named_key, sheet_group) = if let Some(nl) = matched {
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
            CatalogueItemUi {
                index: idx + 1,
                label,
                named_key,
                group: entry.recipe.group.clone(),
                sheet_group,
                sites_per_cell: cell.basis.len(),
                site_orbits: entry.rank[0],
                bond_orbits: entry.rank[1],
                coordination: coordination_string(&cell),
                dsymbol_key: format_dsymbol_key(&entry.canonical_key),
                network_key: format_dsymbol_key(&entry.network_key),
                symmetry_index: entry.symmetry_index,
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

pub fn sliders_for_item(
    item: &CatalogueItemUi,
    current_vals: &[(String, f64)],
) -> Vec<SliderParamUi> {
    let Ok(fam) = LatticeFamily::from_recipe(&item.recipe, Some(item.frame)) else {
        return Vec::new();
    };
    let defaults = fam.parameter_list();
    if defaults.is_empty() {
        return Vec::new();
    }
    let mut override_map: BTreeMap<String, f64> = defaults.iter().cloned().collect();
    for (k, v) in current_vals {
        if override_map.contains_key(k) {
            override_map.insert(k.clone(), *v);
        }
    }
    defaults
        .into_iter()
        .map(|(name, def_val)| {
            let val = override_map[&name];
            let (nom_lo, nom_hi, step) = match name.as_str() {
                "b" => (0.4, 2.6, 0.01),
                "gamma" => (40.0, 140.0, 0.5),
                _ => (def_val - 0.5, def_val + 0.5, 0.002),
            };
            let (allowed_min, allowed_max, forbidden) =
                match allowed_interval(&item.recipe, &name, nom_lo, nom_hi, Some(&override_map), None)
                {
                    Ok(r) => {
                        let (a, b) = if r.current_ok {
                            r.interval
                        } else {
                            (nom_lo, nom_hi)
                        };
                        let (lo_in, hi_in) = if r.edges != (None, None) {
                            inward_limits(a, b, val, step)
                        } else {
                            (a, b)
                        };
                        let segs = r
                            .segments
                            .into_iter()
                            .map(|s| ForbiddenSpanUi {
                                lo: s.lo,
                                hi: s.hi,
                                reason: s.reason,
                            })
                            .collect();
                        (lo_in, hi_in, segs)
                    }
                    Err(_) => (nom_lo, nom_hi, Vec::new()),
                };
            let special_ticks =
                special_points(&item.recipe, &name, nom_lo, nom_hi, Some(&override_map))
                    .unwrap_or_default()
                    .into_iter()
                    .map(|sp| SpecialTickUi {
                        value: sp.value,
                        supergroup: sp.group,
                        equation: sp.equations.join(", "),
                    })
                    .collect();
            SliderParamUi {
                name,
                value: val,
                default_value: def_val,
                nominal_min: nom_lo,
                nominal_max: nom_hi,
                allowed_min,
                allowed_max,
                step,
                special_ticks,
                forbidden,
            }
        })
        .collect()
}

pub fn build_cell_for_item(
    item: &CatalogueItemUi,
    params: &[(String, f64)],
) -> (Cell2D, DSymbolReadoutUi) {
    let base_fam = LatticeFamily::from_recipe(&item.recipe, Some(item.frame))
        .expect("valid recipe in CatalogueItemUi");
    let fam = if params.is_empty() {
        base_fam.clone()
    } else {
        base_fam
            .with_values(params.iter().map(|(k, v)| (k.as_str(), *v)))
            .unwrap_or(base_fam)
    };
    let cell = fam.cell();
    let active_group = cell.identify_wallpaper_group(false);
    let crossings = cell.has_crossings();
    let readout = match fam.dsymbol() {
        Ok(ds) => match ds.canonical_key() {
            Ok(ck) => {
                let nk = dsymbol::network_key(&ck).unwrap_or_else(|_| ck.clone());
                let si = dsymbol::symmetry_index(&ck).unwrap_or(item.symmetry_index);
                let dk_str = format_dsymbol_key(&ck);
                let nk_str = format_dsymbol_key(&nk);
                let status_ok = !crossings;
                let status_message = if crossings {
                    String::from("Bonds cross or pass through a site (non-planar drawing)")
                } else if active_group != item.group {
                    format!(
                        "Planar tiling OK — enhanced symmetry {} (base {})",
                        active_group, item.group
                    )
                } else if dk_str != item.dsymbol_key {
                    String::from("Planar tiling OK — combinatorially distinct D-symbol chamber")
                } else {
                    String::from("Planar tiling OK — canonical D-symbol preserved")
                };
                DSymbolReadoutUi {
                    dsymbol_key: dk_str,
                    network_key: nk_str,
                    symmetry_index: si,
                    active_group,
                    status_ok,
                    status_message,
                }
            }
            Err(err) => DSymbolReadoutUi {
                dsymbol_key: item.dsymbol_key.clone(),
                network_key: item.network_key.clone(),
                symmetry_index: item.symmetry_index,
                active_group,
                status_ok: false,
                status_message: format!("D-symbol key error: {err}"),
            },
        },
        Err(err) => DSymbolReadoutUi {
            dsymbol_key: item.dsymbol_key.clone(),
            network_key: item.network_key.clone(),
            symmetry_index: item.symmetry_index,
            active_group,
            status_ok: false,
            status_message: format!("Non-planar / degenerate drawing: {err}"),
        },
    };
    (cell, readout)
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
        assert!(ro.status_ok && ro.active_group == "p4m");
    }
}
