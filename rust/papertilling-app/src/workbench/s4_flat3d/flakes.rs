//! Symmetry-centred finite flakes of the Gallery's current 2D lattice: a disk, a regular n-gon or a scaled
//! Wigner-Seitz cell around a chosen centre. Its point group is the subgroup of the centre's site-symmetry group
//! that preserves the cut. The core checks that each element permutes the flake's sites; [`check_bonds`] then
//! checks here that each permutation maps every bond onto a bond, and the flake is refused otherwise.

use std::collections::HashSet;

use papertilling_core::flat3d::{SymmetricFlake, build_symmetric_flake, build_wigner_seitz_flake};
use papertilling_core::types::Cell2D;
use serde_json::{Value, json};

use super::Scenes;
use crate::canvas2d::Scene2D;
use crate::mesh3d::wb_mesh::mesh_from_points_edges;

pub(super) const SCENE_TITLE: &str = "S4 symmetric flake";
pub(super) const MIN_SIZE: f64 = 0.5;
pub(super) const MAX_SIZE: f64 = 5.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(super) enum Shape {
    #[default]
    Disk,
    Polygon,
    WignerSeitz,
}

impl Shape {
    pub(super) const ALL: [Self; 3] = [Self::Disk, Self::Polygon, Self::WignerSeitz];

    fn label(self) -> &'static str {
        match self {
            Self::Disk => "Disk cut",
            Self::Polygon => "Regular n-gon cut",
            Self::WignerSeitz => "Scaled Wigner-Seitz cell",
        }
    }
}

/// Where the flake is centred (in the Gallery cell's coordinates).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub(super) enum Centre {
    /// The lattice frame's origin (its highest-symmetry point).
    #[default]
    FrameOrigin,
    /// The first basis site.
    Site0,
    /// A point clicked on the canvas (cleared on a lattice change).
    Clicked([f64; 2]),
}

impl Centre {
    pub(super) fn resolve(self, cell: &Cell2D, frame_origin: [f64; 2]) -> [f64; 2] {
        match self {
            Self::FrameOrigin => frame_origin,
            Self::Site0 => cell.basis.first().copied().unwrap_or([0.0, 0.0]),
            Self::Clicked(p) => p,
        }
    }

    fn label(self) -> String {
        match self {
            Self::FrameOrigin => String::from("Frame origin"),
            Self::Site0 => String::from("First basis site"),
            Self::Clicked(p) => format!("Clicked ({:.3}, {:.3})", p[0], p[1]),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct FlakeParams {
    pub(super) shape: Shape,
    pub(super) n_gon: usize,
    /// Disk radius / n-gon apothem in units of |a1|, or the Wigner-Seitz scale.
    pub(super) size: f64,
    pub(super) centre: Centre,
}

impl Default for FlakeParams {
    fn default() -> Self {
        Self { shape: Shape::Disk, n_gon: 6, size: 2.0, centre: Centre::FrameOrigin }
    }
}

pub(super) fn build(group: &str, cell: &Cell2D, centre: [f64; 2], p: &FlakeParams) -> Result<SymmetricFlake, String> {
    let a = cell.a1[0].hypot(cell.a1[1]).max(1e-12);
    let f = match p.shape {
        Shape::Disk => build_symmetric_flake(group, cell, centre, None, p.size * a)?,
        Shape::Polygon => build_symmetric_flake(group, cell, centre, Some(p.n_gon), p.size * a)?,
        Shape::WignerSeitz => build_wigner_seitz_flake(group, cell, centre, p.size)?,
    };
    if f.crystal.pos.is_empty() {
        return Err(String::from("the flake contains no site: increase its size or move its centre"));
    }
    check_bonds(f.crystal.pos.len(), &f.crystal.edges, &f.vertex_perms)?;
    Ok(f)
}

/// Each site permutation must be a bijection of `0..n` that maps every bond `{u, v}` onto a bond.
pub(super) fn check_bonds(n: usize, edges: &[(usize, usize)], perms: &[Vec<usize>]) -> Result<(), String> {
    let key = |u: usize, v: usize| if u <= v { (u, v) } else { (v, u) };
    if edges.iter().any(|&(u, v)| u >= n || v >= n) {
        return Err(format!("REFUSED: a flake bond names a site outside 0..{n} (nothing drawn)"));
    }
    let set: HashSet<(usize, usize)> = edges.iter().map(|&(u, v)| key(u, v)).collect();
    for (k, perm) in perms.iter().enumerate() {
        let mut seen = vec![false; n];
        let bijective = perm.len() == n && perm.iter().all(|&w| w < n && !std::mem::replace(&mut seen[w], true));
        if !bijective {
            return Err(format!("REFUSED: symmetry element {k} does not permute the {n} flake sites (nothing drawn)"));
        }
        if let Some(&(u, v)) = edges.iter().find(|&&(u, v)| !set.contains(&key(perm[u], perm[v]))) {
            return Err(format!(
                "REFUSED: symmetry element {k} maps bond ({u}, {v}) to ({}, {}), which is not a bond: the bonds break the site symmetry (nothing drawn)",
                perm[u], perm[v]
            ));
        }
    }
    Ok(())
}

/// Names the cut the flake was built with; `radius` is the core's: the apothem of an n-gon cut, the circumradius of a
/// Wigner-Seitz cut.
pub(super) fn cut_label(shape: Shape, n_fold: Option<usize>, radius: f64) -> String {
    match (shape, n_fold) {
        (Shape::Disk, _) => format!("disk cut, radius {radius:.4}"),
        (Shape::Polygon, Some(n)) => format!("regular {n}-gon cut, apothem {radius:.4}"),
        (Shape::Polygon, None) => format!("regular n-gon cut, apothem {radius:.4}"),
        (Shape::WignerSeitz, Some(n)) => format!("scaled Wigner-Seitz cell cut ({n} vertices, not regular in general), circumradius {radius:.4}"),
        (Shape::WignerSeitz, None) => format!("scaled Wigner-Seitz cell cut, circumradius {radius:.4}"),
    }
}

#[derive(Default)]
pub(super) struct FlakeState {
    pub(super) p: FlakeParams,
    pub(super) out: Option<SymmetricFlake>,
    /// The cut shape `out` was built with.
    pub(super) out_shape: Shape,
}

impl FlakeState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let p = &self.p;
        if !(p.size.is_finite() && (MIN_SIZE..=MAX_SIZE).contains(&p.size)) {
            return Err(format!("flake size must lie in {MIN_SIZE}..={MAX_SIZE}"));
        }
        if p.shape == Shape::Polygon && !(3..=12).contains(&p.n_gon) {
            return Err(String::from("the n-gon needs 3..=12 sides"));
        }
        Ok(())
    }

    pub(super) fn clear_for_new_lattice(&mut self) {
        self.out = None;
        if matches!(self.p.centre, Centre::Clicked(_)) {
            self.p.centre = Centre::FrameOrigin;
        }
    }

    pub(super) fn click(&mut self, w: [f64; 2]) -> bool {
        self.p.centre = Centre::Clicked(w);
        true
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let p = &mut self.p;
        let mut ch = false;
        egui::ComboBox::from_id_salt("wb_s4_flake_shape").selected_text(p.shape.label()).show_ui(ui, |ui| {
            for s in Shape::ALL {
                ch |= ui.selectable_value(&mut p.shape, s, s.label()).changed();
            }
        });
        if p.shape == Shape::Polygon {
            ch |= super::int_row(ui, "Polygon sides:", &mut p.n_gon, 3, 12);
        }
        ui.horizontal(|ui| {
            ui.label(if p.shape == Shape::WignerSeitz { "Cell scale:" } else { "Size (|a1| units):" });
            ch |= ui.add(egui::DragValue::new(&mut p.size).range(MIN_SIZE..=MAX_SIZE).speed(0.05)).changed();
        });
        egui::ComboBox::from_id_salt("wb_s4_flake_centre").selected_text(p.centre.label()).show_ui(ui, |ui| {
            for c in [Centre::FrameOrigin, Centre::Site0] {
                ch |= ui.selectable_value(&mut p.centre, c, c.label()).changed();
            }
        });
        ui.small("Click the 2D canvas to centre the flake there.");
        ch
    }

    pub(super) fn show_inspector(&self, ui: &mut egui::Ui) {
        let Some(f) = &self.out else {
            ui.label("No flake built for these settings.");
            return;
        };
        ui.strong(format!("Flake of {} at ({:.3}, {:.3})", f.group, f.centre[0], f.centre[1]));
        ui.label(cut_label(self.out_shape, f.n_fold_polygon, f.radius));
        ui.label(format!(
            "[COMPUTATIONAL EVIDENCE] point group of the cut flake: order {} ({}), the elements of the centre's site-symmetry group that preserve the cut; each permutes the sites (core, positions to 1e-4) and maps every bond onto a bond (panel check)",
            f.point_group_order,
            if f.is_dihedral { "dihedral" } else { "cyclic" }
        ));
        let sizes: Vec<String> = f.site_orbits.iter().map(|o| o.len().to_string()).collect();
        ui.label(format!(
            "{} sites, {} bonds, {} boundary sites (ringed); {} site orbits of sizes [{}]",
            f.crystal.pos.len(),
            f.crystal.edges.len(),
            f.boundary_sites.len(),
            f.site_orbits.len(),
            sizes.join(", ")
        ));
    }

    pub(super) fn scenes(&self) -> Option<Scenes> {
        let f = self.out.as_ref()?;
        let scene = Scene2D::CrystalPatch {
            crystal: f.crystal.clone(),
            title: String::from(SCENE_TITLE),
            note: format!("point group order {}, {} orbits", f.point_group_order, f.site_orbits.len()),
        };
        let mut kinds = vec![0_u8; f.crystal.pos.len()];
        for (k, orb) in f.site_orbits.iter().enumerate() {
            for &s in orb {
                if let Some(x) = kinds.get_mut(s) {
                    *x = (k % 4) as u8;
                }
            }
        }
        let pts: Vec<[f64; 3]> = f.crystal.pos.iter().map(|p| [p[0], p[1], 0.0]).collect();
        let edges: Vec<(usize, usize, u8)> = f.crystal.edges.iter().map(|&(u, v)| (u, v, 0)).collect();
        let mesh = mesh_from_points_edges(&pts, &edges, &kinds);
        Some((scene, Some(mesh), format!("Flake: {} sites coloured by orbit", f.crystal.pos.len())))
    }

    pub(super) fn save(&self) -> Value {
        let p = &self.p;
        let centre = match p.centre {
            Centre::FrameOrigin => json!("FrameOrigin"),
            Centre::Site0 => json!("Site0"),
            Centre::Clicked(_) => json!("FrameOrigin"), // a clicked point belongs to one lattice: not saved
        };
        json!({"shape": format!("{:?}", p.shape), "n_gon": p.n_gon, "size": p.size, "centre": centre})
    }

    pub(super) fn restore(&mut self, v: &Value, notes: &mut Vec<String>) {
        let p = &mut self.p;
        if let Some(s) = v.get("shape").and_then(Value::as_str) {
            match Shape::ALL.into_iter().find(|x| format!("{x:?}") == s) {
                Some(x) => p.shape = x,
                None => notes.push(format!("S4 flake: unknown shape {s:?} (kept current)")),
            }
        }
        if let Some(n) = v.get("n_gon").and_then(Value::as_u64) {
            p.n_gon = (n as usize).clamp(3, 12);
        }
        if let Some(s) = v.get("size").and_then(Value::as_f64).filter(|s| s.is_finite()) {
            p.size = s.clamp(MIN_SIZE, MAX_SIZE);
        }
        match v.get("centre").and_then(Value::as_str) {
            Some("Site0") => p.centre = Centre::Site0,
            Some("FrameOrigin") => p.centre = Centre::FrameOrigin,
            _ => {}
        }
    }
}
