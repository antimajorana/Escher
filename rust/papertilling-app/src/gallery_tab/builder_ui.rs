//! Custom lattice builder (`BrowseMode::CustomBuilder`) and live `N = 7..=25` discovery UI
//! in the Gallery tab.

use papertilling_core::dsymbol::{self, DSymbolKey};
use papertilling_core::plane::basis_reduce::is_geometrically_chiral;
use papertilling_core::plane::custom_builder::{build_custom_lattice, BondSelection, CustomLatticeSpec};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::groups::{wyckoff_table, WALLPAPER_GROUPS};
use papertilling_core::plane::live_search::DiscoveredLattice;
use papertilling_core::types::{Frame2D, Recipe};

use super::search_worker::SearchWorker;
use super::{BrowseMode, GalleryTabState};
use crate::aliases::LatticeAliasKey;
use crate::backend::{AppBackend, CatalogueItemUi};
use crate::canvas2d::Canvas2D;
use crate::live_plane::{build_groups_info, format_dsymbol_key, occupied_wyckoff_letters};

pub struct CustomBuilderState {
    pub group: String,
    pub a: f64,
    pub b: f64,
    pub gamma: f64,
    pub seeds: Vec<[f64; 2]>,
    pub use_voronoi: bool,
    pub shell_mask: u32,
    pub min_degree: usize,
    pub shell_distances: Vec<f64>,
    pub build_status: String,
    pub build_counter: usize,
    pub search_n_min: usize,
    pub search_n_max: usize,
    pub search_min_deg: usize,
    pub search_results: Vec<DiscoveredLattice>,
    pub selected_search_idx: Option<usize>,
    pub search_status: String,
    pub worker: SearchWorker,
}

impl CustomBuilderState {
    pub fn new() -> Self {
        Self {
            group: "p4m".to_string(),
            a: 1.0,
            b: 1.37,
            gamma: 76.0,
            seeds: vec![[0.0, 0.0], [0.5, 0.0]],
            use_voronoi: false,
            shell_mask: 0b00001,
            min_degree: 2,
            shell_distances: Vec::new(),
            build_status: String::from("Ready (click Build & Activate)"),
            build_counter: 0,
            search_n_min: 7,
            search_n_max: 14,
            search_min_deg: 2,
            search_results: Vec::new(),
            selected_search_idx: None,
            search_status: String::new(),
            worker: SearchWorker::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn make_custom_catalogue_item(
    backend: &AppBackend,
    group: &str,
    default_label: String,
    idx: usize,
    recipe: Recipe,
    num_sites: usize,
    site_orbits: usize,
    bond_orbits: usize,
    coordination: String,
    ds_key: Option<&DSymbolKey>,
    net_key: Option<&DSymbolKey>,
    symmetry_index: usize,
    frame: Frame2D,
) -> CatalogueItemUi {
    let dsymbol_key = ds_key
        .map(format_dsymbol_key)
        .unwrap_or_else(|| format!("nonplanar-N{num_sites}"));
    let network_key = net_key
        .map(format_dsymbol_key)
        .unwrap_or_else(|| format!("nonplanar-{coordination}-N{num_sites}"));
    let alias_key = LatticeAliasKey::new(group, "custom", idx.max(1), &dsymbol_key);
    let chiral = LatticeFamily::from_recipe(&recipe, Some(frame))
        .ok()
        .and_then(|f| is_geometrically_chiral(&f.cell()).ok())
        .unwrap_or(false);
    let topologically_chiral = ds_key
        .and_then(|k| dsymbol::is_topologically_chiral(k).ok())
        .unwrap_or(false);
    let mut item = CatalogueItemUi {
        index: idx.max(1),
        label: default_label.clone(),
        default_label,
        alias_key,
        named_key: None,
        group: group.to_string(),
        sheet_group: None,
        sites_per_cell: num_sites,
        site_orbits,
        bond_orbits,
        coordination,
        dsymbol_key,
        network_key,
        symmetry_index,
        chiral,
        topologically_chiral,
        occupied_wyckoff: occupied_wyckoff_letters(&recipe),
        recipe,
        frame,
    };
    item.label = backend.resolve_alias(&item);
    item
}

impl GalleryTabState {
    pub(super) fn poll_live_search(
        &mut self,
        ui: &egui::Ui,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) -> bool {
        let Some(done) = self.custom_builder.worker.poll() else {
            if self.custom_builder.worker.busy {
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(50));
            }
            return false;
        };
        let _ = (backend, canvas);
        match done.result {
            Ok(list) => {
                self.custom_builder.search_status = format!(
                    "Found {} distinct {} lattices (N={}..={}, min_deg={})",
                    list.len(),
                    done.group,
                    self.custom_builder.search_n_min,
                    self.custom_builder.search_n_max,
                    self.custom_builder.search_min_deg
                );
                self.custom_builder.search_results = list;
                self.custom_builder.selected_search_idx = None;
            }
            Err(e) => {
                self.custom_builder.search_status = format!("Search error: {e}");
            }
        }
        false
    }

    pub(super) fn live_search_section(
        &mut self,
        ui: &mut egui::Ui,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        target_group: &str,
    ) -> bool {
        self.poll_live_search(ui, backend, canvas);
        let mut activated = false;
        egui::CollapsingHeader::new(format!("Live Search (N > 7, {target_group})"))
            .id_salt(format!("gal_live_search_{target_group}"))
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.small("N:");
                    ui.add(egui::DragValue::new(&mut self.custom_builder.search_n_min).range(3..=25).speed(0.2));
                    ui.small("..");
                    ui.add(egui::DragValue::new(&mut self.custom_builder.search_n_max).range(self.custom_builder.search_n_min..=25).speed(0.2));
                    ui.small("min deg:");
                    ui.selectable_value(&mut self.custom_builder.search_min_deg, 2, "2");
                    ui.selectable_value(&mut self.custom_builder.search_min_deg, 3, "3");
                    if ui.button("Search").clicked() {
                        self.custom_builder.search_status = format!("Searching {target_group}...");
                        self.custom_builder.worker.submit(
                            target_group.to_string(),
                            self.custom_builder.search_n_min,
                            self.custom_builder.search_n_max,
                            self.custom_builder.search_min_deg,
                            Some(ui.ctx().clone()),
                        );
                    }
                    if self.custom_builder.worker.busy {
                        ui.spinner();
                    }
                });
                if !self.custom_builder.search_status.is_empty() {
                    ui.small(&self.custom_builder.search_status);
                }
                if !self.custom_builder.search_results.is_empty() {
                    egui::ScrollArea::vertical().id_salt("live_search_scroll").max_height(135.0).show(ui, |ui| {
                        let mut pick = None;
                        for (idx, d) in self.custom_builder.search_results.iter().enumerate() {
                            let selected = self.custom_builder.selected_search_idx == Some(idx);
                            let tag = if d.planar { "planar" } else { "non-planar" };
                            let lbl = format!(
                                "#{} N={} B={} ({}) {} [{}]",
                                idx + 1, d.num_sites, d.num_bonds, d.coordination, d.bond_mode_label, tag
                            );
                            if ui.selectable_label(selected, lbl).clicked() {
                                pick = Some(idx);
                            }
                        }
                        if let Some(idx) = pick {
                            self.custom_builder.selected_search_idx = Some(idx);
                            let d = self.custom_builder.search_results[idx].clone();
                            self.custom_builder.build_counter += 1;
                            let grp = d.recipe.group.clone();
                            let lbl = format!("Search {grp} #{} (N={})", idx + 1, d.num_sites);
                            let item = make_custom_catalogue_item(
                                backend,
                                &grp,
                                lbl,
                                self.custom_builder.build_counter,
                                d.recipe,
                                d.num_sites,
                                d.site_orbits,
                                d.bond_orbits,
                                d.coordination,
                                d.dsymbol_key.as_ref(),
                                d.network_key.as_ref(),
                                d.symmetry_index,
                                d.frame,
                            );
                            self.activate_item(item, backend, canvas, true);
                            activated = true;
                        }
                    });
                }
            });
        activated
    }

    pub(super) fn custom_builder_panel(
        &mut self,
        ui: &mut egui::Ui,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
    ) -> bool {
        let _ = build_groups_info;
        let mut changed = false;
        ui.strong("Custom Periodic Lattice Builder");
        ui.horizontal_wrapped(|ui| {
            ui.small("Group:");
            egui::ComboBox::from_id_salt("cb_group_sel")
                .selected_text(&self.custom_builder.group)
                .show_ui(ui, |ui| {
                    for &g in &WALLPAPER_GROUPS {
                        ui.selectable_value(&mut self.custom_builder.group, g.to_string(), g);
                    }
                });
            if ui.small_button("Lieb (p4m)").clicked() {
                self.custom_builder.group = "p4m".to_string();
                self.custom_builder.seeds = vec![[0.0, 0.0], [0.5, 0.0]];
                self.custom_builder.use_voronoi = false;
                self.custom_builder.shell_mask = 0b00001;
                self.custom_builder.min_degree = 2;
            }
            if ui.small_button("Kagome (p6m)").clicked() {
                self.custom_builder.group = "p6m".to_string();
                self.custom_builder.seeds = vec![[0.5, 0.0]];
                self.custom_builder.use_voronoi = false;
                self.custom_builder.shell_mask = 0b00001;
                self.custom_builder.min_degree = 3;
            }
        });

        let grp = self.custom_builder.group.clone();
        if grp == "p1" || grp == "p2" {
            ui.horizontal(|ui| {
                ui.small("b:");
                ui.add(egui::DragValue::new(&mut self.custom_builder.b).range(0.3..=3.0).speed(0.01));
                ui.small("γ°:");
                ui.add(egui::DragValue::new(&mut self.custom_builder.gamma).range(35.0..=145.0).speed(0.5));
            });
        } else if matches!(grp.as_str(), "pm" | "pg" | "cm" | "pmm" | "pmg" | "pgg" | "cmm") {
            ui.horizontal(|ui| {
                ui.small("b:");
                ui.add(egui::DragValue::new(&mut self.custom_builder.b).range(0.3..=3.0).speed(0.01));
            });
        }

        if let Ok(rows) = wyckoff_table(&grp) {
            ui.horizontal_wrapped(|ui| {
                ui.small("+ Wyckoff:");
                for r in rows {
                    let lbl = format!("{}{}", r.multiplicity, r.letter);
                    if ui.small_button(lbl).on_hover_text(r.site_symmetry).clicked() {
                        let p = [
                            (r.base[0] + r.d_mat[0].first().copied().unwrap_or(0.0) * 0.19).rem_euclid(1.0),
                            (r.base[1] + r.d_mat[1].first().copied().unwrap_or(0.0) * 0.27).rem_euclid(1.0),
                        ];
                        self.custom_builder.seeds.push(p);
                    }
                }
            });
        }

        let mut remove_idx = None;
        for (i, s) in self.custom_builder.seeds.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.small(format!("Orbit {}:", i + 1));
                ui.add(egui::DragValue::new(&mut s[0]).range(0.0..=1.0).speed(0.005).fixed_decimals(3));
                ui.add(egui::DragValue::new(&mut s[1]).range(0.0..=1.0).speed(0.005).fixed_decimals(3));
                if ui.small_button("×").clicked() {
                    remove_idx = Some(i);
                }
            });
        }
        if let Some(i) = remove_idx {
            self.custom_builder.seeds.remove(i);
        }

        ui.horizontal_wrapped(|ui| {
            ui.small("Bonds:");
            ui.selectable_value(&mut self.custom_builder.use_voronoi, true, "Voronoi dual");
            ui.selectable_value(&mut self.custom_builder.use_voronoi, false, "Distance Shells");
            ui.small("min deg:");
            ui.selectable_value(&mut self.custom_builder.min_degree, 2, "2");
            ui.selectable_value(&mut self.custom_builder.min_degree, 3, "3");
        });
        if !self.custom_builder.use_voronoi {
            ui.horizontal_wrapped(|ui| {
                for bit in 0..5 {
                    let mut on = (self.custom_builder.shell_mask >> bit) & 1 == 1;
                    let d_lbl = self
                        .custom_builder
                        .shell_distances
                        .get(bit)
                        .map_or_else(|| format!("S{}", bit + 1), |d| format!("S{} ({d:.3})", bit + 1));
                    if ui.checkbox(&mut on, d_lbl).changed() {
                        if on {
                            self.custom_builder.shell_mask |= 1 << bit;
                        } else {
                            self.custom_builder.shell_mask &= !(1 << bit);
                        }
                    }
                }
            });
        }

        if ui.button("Build & Activate Custom Lattice").clicked() {
            let b_opt = if grp == "p1" || grp == "p2" || matches!(grp.as_str(), "pm" | "pg" | "cm" | "pmm" | "pmg" | "pgg" | "cmm") {
                Some(self.custom_builder.b)
            } else {
                None
            };
            let g_opt = if grp == "p1" || grp == "p2" {
                Some(self.custom_builder.gamma)
            } else {
                None
            };
            let spec = CustomLatticeSpec {
                group: grp.clone(),
                a: self.custom_builder.a,
                b: b_opt,
                gamma: g_opt,
                seeds: self.custom_builder.seeds.clone(),
                bond_mode: if self.custom_builder.use_voronoi {
                    BondSelection::Voronoi
                } else {
                    BondSelection::ShellMask(self.custom_builder.shell_mask)
                },
                min_degree: self.custom_builder.min_degree,
            };
            match build_custom_lattice(&spec) {
                Ok(res) => {
                    self.custom_builder.shell_distances = res.shell_distances.clone();
                    self.custom_builder.build_status = res.status.clone();
                    self.custom_builder.build_counter += 1;
                    let r_grp = res.recipe.group.clone();
                    let r_lbl = format!("Custom {r_grp} #{}", self.custom_builder.build_counter);
                    let item = make_custom_catalogue_item(
                        backend,
                        &r_grp,
                        r_lbl,
                        self.custom_builder.build_counter,
                        res.recipe,
                        res.num_sites,
                        res.site_orbits,
                        res.bond_orbits,
                        res.coordination,
                        res.dsymbol_key.as_ref(),
                        res.network_key.as_ref(),
                        res.symmetry_index.unwrap_or(1),
                        res.frame,
                    );
                    self.activate_item(item, backend, canvas, true);
                    self.mode = BrowseMode::CustomBuilder;
                    changed = true;
                }
                Err(e) => {
                    self.custom_builder.build_status = format!("Build error: {e}");
                }
            }
        }
        if !self.custom_builder.build_status.is_empty() {
            ui.small(&self.custom_builder.build_status);
        }
        ui.separator();
        changed |= self.live_search_section(ui, backend, canvas, &grp);
        changed
    }
}
