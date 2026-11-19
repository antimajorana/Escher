//! Stone-Wales bond rotation on the host disk: the bond nearest the centre by default, or one picked by clicking near
//! its midpoint. On the honeycomb four hexagons become the 5-7-7-5 core.

use papertilling_core::defect::{StoneWalesResult, stone_wales_flip};
use papertilling_core::types::Cell2D;

use super::{Host, make_host};
use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::canvas2d::Scene2D;

pub(super) const SCENE_TITLE: &str = "S2 bond-rotation patch";

pub(crate) struct SwOut {
    pub(crate) host: Host,
    pub(crate) result: StoneWalesResult,
    pub(crate) edge: usize,
}

fn midpoint(h: &Host, k: usize) -> [f64; 2] {
    let (u, v) = h.crystal.edges[k];
    let (p, q) = (h.crystal.pos[u], h.crystal.pos[v]);
    [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])]
}

/// The interior bond (both ends off the rim) whose midpoint is nearest `w`.
pub(crate) fn nearest_interior_edge(h: &Host, w: [f64; 2]) -> Option<usize> {
    let d = |k: usize| {
        let m = midpoint(h, k);
        (m[0] - w[0]).hypot(m[1] - w[1])
    };
    (0..h.crystal.edges.len())
        .filter(|&k| {
            let (u, v) = h.crystal.edges[k];
            !h.rim.contains(&u) && !h.rim.contains(&v)
        })
        .min_by(|&a, &b| d(a).total_cmp(&d(b)))
}

/// Interior bonds ordered by the distance of their midpoint from the disk centre.
fn interior_edges_by_centre(h: &Host) -> Vec<usize> {
    let d = |k: usize| {
        let m = midpoint(h, k);
        m[0].hypot(m[1])
    };
    let mut ks: Vec<usize> = (0..h.crystal.edges.len())
        .filter(|&k| {
            let (u, v) = h.crystal.edges[k];
            !h.rim.contains(&u) && !h.rim.contains(&v)
        })
        .collect();
    ks.sort_by(|&a, &b| d(a).total_cmp(&d(b)));
    ks
}

/// Bonds tried in auto mode before giving up (the nearest-centre bonds of the disk).
const AUTO_TRIES: usize = 48;

pub(crate) fn build(cell: &Cell2D, radius: f64, edge: Option<usize>, relax: bool) -> Result<SwOut, String> {
    let host = make_host(cell, radius)?;
    let relax_cell = relax.then_some(cell);
    let Some(k) = edge else {
        // auto: the interior bond nearest the centre that admits the rotation
        let mut first_err = None;
        for k in interior_edges_by_centre(&host).into_iter().take(AUTO_TRIES) {
            match stone_wales_flip(&host.crystal, k, relax_cell) {
                Ok(result) => return Ok(SwOut { host, result, edge: k }),
                Err(e) => {
                    first_err.get_or_insert(e);
                }
            }
        }
        return Err(match first_err {
            Some(e) => format!("no bond near the centre admits a bond rotation on this lattice ({e})"),
            None => String::from("the host disk has no interior bond: increase the radius"),
        });
    };
    let (u, v) = *host.crystal.edges.get(k).ok_or_else(|| format!("bond {k} is out of range: pick a bond on the canvas"))?;
    if host.rim.contains(&u) || host.rim.contains(&v) {
        return Err(format!("bond {k} touches the rim of the disk: pick an interior bond"));
    }
    let result = stone_wales_flip(&host.crystal, k, relax_cell)?;
    Ok(SwOut { host, result, edge: k })
}

fn rings(r: &[usize; 4]) -> String {
    r.iter().map(ToString::to_string).collect::<Vec<_>>().join("-")
}

#[derive(Default)]
pub(crate) struct SwState {
    /// `None` = the interior bond nearest the centre.
    pub(crate) edge: Option<usize>,
    pub(crate) relax: bool,
    pub(crate) out: Option<SwOut>,
    note: String,
}

impl SwState {
    pub(super) fn clear_for_new_host(&mut self) {
        self.edge = None;
        self.out = None;
        self.note.clear();
    }

    /// Canvas click: pick the interior bond nearest the click (true = rebuild).
    pub(crate) fn click(&mut self, w: [f64; 2]) -> bool {
        let Some(o) = &self.out else { return false };
        match nearest_interior_edge(&o.host, w) {
            Some(k) if Some(k) != self.edge && k != o.edge => {
                self.edge = Some(k);
                self.note.clear();
                true
            }
            Some(_) => false,
            None => {
                self.note = String::from("refused: no interior bond near the click");
                false
            }
        }
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut rebuild = false;
        ui.small("Rotates one bond by 90 deg about its midpoint. Click near a bond midpoint to pick it.");
        if let Some(o) = &self.out {
            let n = o.host.crystal.edges.len();
            let mut k = self.edge.unwrap_or(o.edge);
            ui.horizontal(|ui| {
                ui.label("Bond index:");
                if ui.add(egui::DragValue::new(&mut k).range(0..=n.saturating_sub(1))).changed() {
                    self.edge = Some(k);
                    rebuild = true;
                }
            });
        }
        if self.edge.is_some() && ui.button("Use the bond nearest the centre").clicked() {
            self.edge = None;
            rebuild = true;
        }
        rebuild |= ui.checkbox(&mut self.relax, "Relax (GSV elastic, uses the cell)").changed();
        if !self.note.is_empty() {
            ui.small(self.note.as_str());
        }
        rebuild
    }

    pub(super) fn show_inspector(&self, ui: &mut egui::Ui) {
        let Some(o) = &self.out else {
            ui.label("Building the bond rotation ...");
            return;
        };
        let r = &o.result;
        ui.label(format!("Rotated bond {} = sites {:?}", o.edge, r.flipped_edge));
        let mut sorted = r.after_rings;
        sorted.sort_unstable();
        ui.label(format!("Rings {} -> {} (sorted {})", rings(&r.before_rings), rings(&r.after_rings), rings(&sorted)));
        ui.small("Ring rule: (p1, p2, q1, q2) -> (p1-1, p2-1, q1+1, q2+1); no lattice cut, so the holonomy is trivial.");
        ui.label(format!("Holonomy {}, relaxation {}", r.holonomy_badge, r.relax_badge));
        ui.label(format!("Sites {}, bonds {} (unchanged by the rotation)", r.crystal.pos.len(), r.crystal.edges.len()));
    }

    pub(super) fn scene(&self) -> Scene2D {
        match &self.out {
            Some(o) => Scene2D::CrystalPatch {
                crystal: o.result.crystal.clone(),
                title: String::from(SCENE_TITLE),
                note: format!("rings {} -> {}", rings(&o.result.before_rings), rings(&o.result.after_rings)),
            },
            None => empty_scene(SCENE_TITLE, "building ..."),
        }
    }
}
