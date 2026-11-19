//! Point defects on the host disk: vacancy (optionally ring-reconstructed), substitution, interstitial and adatom.
//! Ops are picked on the host disk (clicks resolve to its nearest site) and applied in order by the core.

use std::collections::BTreeMap;

use papertilling_core::defect::{PointDefectOp, PointDefectedCrystal, apply_point_defects, validate_element_symbol};
use papertilling_core::types::Cell2D;

use super::{Host, make_host};
use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::canvas2d::Scene2D;
use crate::mesh3d::GpuMeshData;
use crate::mesh3d::wb_mesh::mesh_from_points_edges;

pub(super) const SCENE_TITLE: &str = "S2 point-defect patch";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(super) enum OpKind {
    #[default]
    Vacancy,
    Substitutional,
    Interstitial,
    Adatom,
}

impl OpKind {
    const ALL: [Self; 4] = [Self::Vacancy, Self::Substitutional, Self::Interstitial, Self::Adatom];

    fn label(self) -> &'static str {
        match self {
            Self::Vacancy => "Vacancy",
            Self::Substitutional => "Substitutional",
            Self::Interstitial => "Interstitial",
            Self::Adatom => "Adatom",
        }
    }
}

pub(crate) struct PointOut {
    pub(crate) host: Host,
    pub(crate) result: PointDefectedCrystal,
}

impl PointOut {
    pub(crate) fn has_adatom(&self) -> bool {
        self.result.pos3d.iter().any(|p| p[2].abs() > 1e-9)
    }

    /// Per-site height for the extended XYZ export (adatoms sit above their host).
    pub(crate) fn z(&self) -> Vec<f64> {
        self.result.pos3d.iter().map(|p| p[2]).collect()
    }
}

pub(crate) fn build(cell: &Cell2D, radius: f64, ops: Vec<PointDefectOp>) -> Result<PointOut, String> {
    let host = make_host(cell, radius)?;
    let result = apply_point_defects(&host.crystal, &ops)?;
    Ok(PointOut { host, result })
}

pub(crate) struct PointState {
    pub(super) kind: OpKind,
    species: String,
    reconstruct: bool,
    /// Adatom height and interstitial bond cutoff, in units of the host's mean bond length.
    height_bonds: f64,
    cutoff_bonds: f64,
    pub(crate) ops: Vec<PointDefectOp>,
    pub(crate) note: String,
    pub(crate) out: Option<PointOut>,
}

impl Default for PointState {
    fn default() -> Self {
        Self {
            kind: OpKind::Vacancy,
            species: String::from("N"),
            reconstruct: false,
            height_bonds: 1.0,
            cutoff_bonds: 1.05,
            ops: Vec::new(),
            note: String::new(),
            out: None,
        }
    }
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn describe(op: &PointDefectOp) -> String {
    match op {
        PointDefectOp::Vacancy { site, reconstruct_ring } => {
            format!("vacancy at site {site}{}", if *reconstruct_ring { " (ring reconstructed)" } else { "" })
        }
        PointDefectOp::Substitutional { site, species } => format!("{species} on site {site}"),
        PointDefectOp::Interstitial { pos, species, .. } => format!("{species} interstitial at ({:.2}, {:.2})", pos[0], pos[1]),
        PointDefectOp::Adatom { host_site, height, species } => format!("{species} adatom on site {host_site}, h = {height:.2}"),
    }
}

impl PointState {
    pub(super) fn clear_for_new_host(&mut self) {
        self.ops.clear();
        self.out = None;
        self.note.clear();
    }

    fn vacated(&self, site: usize) -> bool {
        self.ops.iter().any(|o| matches!(o, PointDefectOp::Vacancy { site: s, .. } if *s == site))
    }

    /// The op of the current kind at world point `w` of the host disk, or the reason it is refused.
    pub(crate) fn op_at(&self, w: [f64; 2]) -> Result<PointDefectOp, String> {
        let host = &self.out.as_ref().ok_or("the host disk is still building")?.host;
        let c = &host.crystal;
        let site = (0..c.pos.len()).min_by(|&a, &b| dist(c.pos[a], w).total_cmp(&dist(c.pos[b], w))).ok_or("empty host")?;
        let d = dist(c.pos[site], w);
        if d > 1.5 * host.bond {
            return Err(String::from("refused: the point lies outside the host disk"));
        }
        if self.kind != OpKind::Vacancy {
            validate_element_symbol(&self.species).map_err(|e| format!("refused: {e}"))?;
        }
        if self.kind != OpKind::Interstitial && self.vacated(site) {
            return Err(format!("refused: site {site} is already vacant"));
        }
        let species = self.species.clone();
        Ok(match self.kind {
            OpKind::Vacancy => {
                if host.rim.contains(&site) {
                    return Err(format!("refused: site {site} lies on the rim of the disk (a vacancy there opens the boundary)"));
                }
                PointDefectOp::Vacancy { site, reconstruct_ring: self.reconstruct }
            }
            OpKind::Substitutional => PointDefectOp::Substitutional { site, species },
            OpKind::Interstitial => {
                if d < 0.25 * host.bond {
                    return Err(format!("refused: too close to site {site} (atoms would overlap)"));
                }
                let near_added = self.ops.iter().any(|o| matches!(o, PointDefectOp::Interstitial { pos, .. } if dist(*pos, w) < 0.25 * host.bond));
                if near_added {
                    return Err(String::from("refused: an interstitial already sits there (atoms would overlap)"));
                }
                PointDefectOp::Interstitial { pos: w, species, bond_cutoff: self.cutoff_bonds * host.bond }
            }
            OpKind::Adatom if self.ops.iter().any(|o| matches!(o, PointDefectOp::Adatom { host_site, .. } if *host_site == site)) => {
                return Err(format!("refused: site {site} already carries an adatom (atoms would overlap)"));
            }
            OpKind::Adatom => PointDefectOp::Adatom { host_site: site, height: self.height_bonds * host.bond, species },
        })
    }

    /// A target near the disk centre: the central site, or (interstitial) the midpoint of its first bond.
    pub(crate) fn centre_target(&self) -> Option<[f64; 2]> {
        let c = &self.out.as_ref()?.host.crystal;
        let s = (0..c.pos.len()).min_by(|&a, &b| dist(c.pos[a], [0.0; 2]).total_cmp(&dist(c.pos[b], [0.0; 2])))?;
        if self.kind != OpKind::Interstitial {
            return Some(c.pos[s]);
        }
        let t = c.adj[s].first()?.to;
        Some([0.5 * (c.pos[s][0] + c.pos[t][0]), 0.5 * (c.pos[s][1] + c.pos[t][1])])
    }

    /// Canvas click: add an op (true = rebuild) or record why it was refused.
    pub(crate) fn click(&mut self, w: [f64; 2]) -> bool {
        match self.op_at(w) {
            Ok(op) => {
                self.note = format!("added: {}", describe(&op));
                self.ops.push(op);
                true
            }
            Err(e) => {
                self.note = e;
                false
            }
        }
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = false;
        egui::ComboBox::from_id_salt("wb_s2_point_kind").selected_text(self.kind.label()).show_ui(ui, |ui| {
            for k in OpKind::ALL {
                ui.selectable_value(&mut self.kind, k, k.label());
            }
        });
        match self.kind {
            OpKind::Vacancy => {
                ui.checkbox(&mut self.reconstruct, "Reconstruct the ring (one chord)");
            }
            _ => {
                ui.horizontal(|ui| {
                    ui.label("Species:");
                    ui.add(egui::TextEdit::singleline(&mut self.species).desired_width(40.0));
                });
                if let Err(e) = validate_element_symbol(&self.species) {
                    ui.colored_label(egui::Color32::from_rgb(192, 57, 43), e);
                }
            }
        }
        if self.kind == OpKind::Adatom {
            ui.horizontal(|ui| {
                ui.label("Height (bonds):");
                ui.add(egui::DragValue::new(&mut self.height_bonds).range(0.2..=3.0).speed(0.02));
            });
        }
        if self.kind == OpKind::Interstitial {
            ui.horizontal(|ui| {
                ui.label("Bond cutoff (bonds):");
                ui.add(egui::DragValue::new(&mut self.cutoff_bonds).range(0.5..=2.0).speed(0.01));
            });
        }
        ui.small("Click the 2D canvas to place the op at the nearest host site.");
        if ui.button("Add at the disk centre").clicked()
            && let Some(w) = self.centre_target()
        {
            rebuild |= self.click(w);
        }
        ui.separator();
        ui.label(format!("Ops ({}), applied in order:", self.ops.len()));
        let mut remove = None;
        for (k, op) in self.ops.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!("{}. {}", k + 1, describe(op)));
                if ui.small_button("remove").clicked() {
                    remove = Some(k);
                }
            });
        }
        if let Some(k) = remove {
            self.ops.remove(k);
            self.note.clear();
            rebuild = true;
        }
        if !self.ops.is_empty() && ui.button("Clear all ops").clicked() {
            self.ops.clear();
            self.note.clear();
            rebuild = true;
        }
        if !self.note.is_empty() {
            ui.small(self.note.as_str());
        }
        rebuild
    }

    pub(super) fn show_inspector(&self, ui: &mut egui::Ui) {
        let Some(o) = &self.out else {
            ui.label("Building the host disk ...");
            return;
        };
        let r = &o.result;
        ui.label(format!("Sites: {} host -> {} defected", o.host.crystal.pos.len(), r.crystal.pos.len()));
        ui.label(format!("Bonds: {} host -> {} defected", o.host.crystal.edges.len(), r.crystal.edges.len()));
        ui.label(r.summary.as_str());
        ui.label(format!("Holonomy {}, core rings {}", r.holonomy_badge, r.core_badge));
        let mut hist: BTreeMap<&str, usize> = BTreeMap::new();
        for s in &r.species {
            *hist.entry(s.as_str()).or_default() += 1;
        }
        let h: Vec<String> = hist.iter().map(|(s, n)| format!("{s}: {n}")).collect();
        ui.label(format!("Species: {}", h.join(", ")));
        if o.has_adatom() {
            ui.small("Adatoms sit above the plane: the 3D pane shows the patch with heights.");
        }
    }

    pub(super) fn scenes(&self) -> (Scene2D, Option<GpuMeshData>, String) {
        let Some(o) = &self.out else {
            return (empty_scene(SCENE_TITLE, "building ..."), None, String::new());
        };
        let c = &o.result.crystal;
        let scene = Scene2D::CrystalPatch { crystal: c.clone(), title: String::from(SCENE_TITLE), note: o.result.summary.clone() };
        if !o.has_adatom() {
            return (scene, None, String::new());
        }
        let raised = |i: usize| o.result.pos3d.get(i).is_some_and(|p| p[2].abs() > 1e-9);
        let edges: Vec<(usize, usize, u8)> = c.edges.iter().map(|&(u, v)| (u, v, u8::from(raised(u) || raised(v)))).collect();
        let kinds: Vec<u8> = c.basis_idx.iter().map(|&b| (b % 4) as u8).collect();
        let mesh = mesh_from_points_edges(&o.result.pos3d, &edges, &kinds);
        (scene, Some(mesh), format!("S2 adatom view: {} sites (adatom bonds in red)", c.pos.len()))
    }
}
