//! Stone-Wales bond rotation on the host disk: the bond nearest the centre by default, or one picked by clicking near
//! its midpoint. On the honeycomb four hexagons become the 5-7-7-5 core.

use papertilling_core::defect::core_graph::LatticeCoreInfo;
use papertilling_core::defect::defect_growth::{DEFECT_MAX_CELLS, grow_with_rebuild, start_radius};
use papertilling_core::defect::{StoneWalesResult, stone_wales_flip};
use papertilling_core::types::Cell2D;

use super::{Host, make_host};
use crate::atlas_tab::hyperbolic_scene::empty_scene;
use crate::canvas2d::Scene2D;
use crate::canvas2d::gauge_overlay::CrystalOverlayData;
use crate::defects_ui::growth::{DEFAULT_LAYERS, GrowthView, MAX_LAYERS, show_readout, stone_wales_seed_line};

pub(super) const SCENE_TITLE: &str = "S2 bond-rotation patch";

pub(crate) struct SwOut {
    pub(crate) host: Host,
    pub(crate) result: StoneWalesResult,
    pub(crate) edge: usize,
    /// C1 WP-U7: the bond-rotated disk grown from the Dirichlet cell about the rotated bond (`None` when `grow_on` is off).
    pub(crate) growth: Option<Result<GrowthView, String>>,
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
                Ok(result) => return Ok(SwOut { host, result, edge: k, growth: None }),
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
    Ok(SwOut { host, result, edge: k, growth: None })
}

/// Size and rebuild the host disk via [`grow_with_rebuild`] (which calls `grow_crystal` on each rebuilt disk) from
/// `r0 = radius.max(start_radius(&info.lat, target_w, layers)?)`, resolving the target bond by world midpoint
/// `target_w` and relaxing once at the end when `relax` is true.
pub(crate) fn build_grown(
    cell: &Cell2D,
    radius: f64,
    edge: Option<usize>,
    target_w: Option<[f64; 2]>,
    relax: bool,
    layers: u32,
) -> Result<SwOut, String> {
    let target_w = match target_w {
        Some(w) => w,
        None => {
            let init = build(cell, radius, edge, false)?;
            midpoint(&init.host, init.edge)
        }
    };
    let info = LatticeCoreInfo::new(cell);
    let r0 = radius.max(start_radius(&info.lat, target_w, layers)? + target_w[0].hypot(target_w[1]));
    let mut last: Option<(Host, usize, StoneWalesResult)> = None;
    let out = grow_with_rebuild(
        |r| {
            let host_r = make_host(cell, r)?;
            let k_r = nearest_interior_edge(&host_r, target_w)
                .ok_or_else(|| String::from("the host disk has no interior bond near the target"))?;
            let m = midpoint(&host_r, k_r);
            if (m[0] - target_w[0]).hypot(m[1] - target_w[1]) > 1e-4 {
                return Err(String::from("the target bond was not found in the rebuilt host disk"));
            }
            let sw = stone_wales_flip(&host_r.crystal, k_r, None)?;
            let crystal = sw.crystal.clone();
            last = Some((host_r, k_r, sw));
            Ok(crystal)
        },
        r0,
        &info.lat,
        target_w,
        layers,
        DEFECT_MAX_CELLS,
    )?;
    let (final_host, final_k, mut final_sw) = last.ok_or("Stone-Wales growth produced no host")?;
    let mut note = out.note;
    if relax {
        final_sw = stone_wales_flip(&final_host.crystal, final_k, Some(cell))?;
        let msg = format!("positions from a radius-{:.2} relaxation", out.radius);
        note = if note.is_empty() { msg } else { format!("{note}; {msg}") };
    }
    let line = stone_wales_seed_line(out.result.seed.cells.len());
    let view = GrowthView::from_growth(
        &final_sw.crystal,
        &out.result.complex,
        &out.result.growth,
        layers,
        out.radius,
        line,
        note,
    );
    Ok(SwOut { host: final_host, result: final_sw, edge: final_k, growth: Some(Ok(view)) })
}

pub(crate) fn run(
    cell: &Cell2D,
    radius: f64,
    edge: Option<usize>,
    target_w: Option<[f64; 2]>,
    relax: bool,
    grow_on: bool,
    layers: u32,
) -> Result<SwOut, String> {
    if grow_on {
        return build_grown(cell, radius, edge, target_w, relax, layers);
    }
    if let Some(tw) = target_w {
        let host = make_host(cell, radius)?;
        let k = nearest_interior_edge(&host, tw)
            .filter(|&k| {
                let m = midpoint(&host, k);
                (m[0] - tw[0]).hypot(m[1] - tw[1]) <= 1e-4
            })
            .ok_or_else(|| {
                String::from("the picked bond lies outside the host disk: increase the radius or use the centre bond")
            })?;
        return build(cell, radius, Some(k), relax);
    }
    build(cell, radius, edge, relax)
}

fn rings(r: &[usize; 4]) -> String {
    r.iter().map(ToString::to_string).collect::<Vec<_>>().join("-")
}

pub(crate) struct SwState {
    /// `None` = the interior bond nearest the centre.
    pub(crate) edge: Option<usize>,
    /// World midpoint of a user-picked bond, so switching `grow_on` resolves the same bond across host radii.
    pub(crate) target_w: Option<[f64; 2]>,
    pub(crate) relax: bool,
    pub(crate) grow_on: bool,
    /// Growth layers read out in the inspector when `grow_on` is ticked.
    pub(crate) layers: u32,
    pub(crate) out: Option<SwOut>,
    note: String,
}

impl Default for SwState {
    fn default() -> Self {
        Self {
            edge: None,
            target_w: None,
            relax: false,
            grow_on: false,
            layers: DEFAULT_LAYERS,
            out: None,
            note: String::new(),
        }
    }
}

impl SwState {
    pub(super) fn clear_for_new_host(&mut self) {
        self.edge = None;
        self.target_w = None;
        self.out = None;
        self.note.clear();
    }

    /// Canvas click: pick the interior bond nearest the click (true = rebuild).
    pub(crate) fn click(&mut self, w: [f64; 2]) -> bool {
        let Some(o) = &self.out else { return false };
        match nearest_interior_edge(&o.host, w) {
            Some(k) if Some(k) != self.edge && k != o.edge => {
                self.edge = Some(k);
                self.target_w = Some(midpoint(&o.host, k));
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
                    self.target_w = o.host.crystal.edges.get(k).map(|_| midpoint(&o.host, k));
                    rebuild = true;
                }
            });
        }
        if self.edge.is_some() && ui.button("Use the bond nearest the centre").clicked() {
            self.edge = None;
            self.target_w = None;
            rebuild = true;
        }
        rebuild |= ui.checkbox(&mut self.relax, "Relax (GSV elastic, uses the cell)").changed();
        ui.horizontal(|ui| {
            rebuild |= ui.checkbox(&mut self.grow_on, "Layers").changed();
            let slider = egui::Slider::new(&mut self.layers, 0..=MAX_LAYERS);
            rebuild |= ui.add_enabled(self.grow_on, slider).changed() && self.grow_on;
        });
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
        if self.grow_on {
            match &o.growth {
                Some(Ok(v)) => show_readout(ui, v, "sw_growth_tbl"),
                Some(Err(why)) => {
                    ui.small(format!("Not grown: {why}"));
                }
                None => {}
            }
        }
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

    pub(crate) fn overlay(&self) -> Option<CrystalOverlayData> {
        if !self.grow_on {
            return None;
        }
        self.out.as_ref()?.growth.as_ref()?.as_ref().ok().map(GrowthView::overlay)
    }
}
