//! C1 WP-U3: the boundary-condition options of the Surfaces tab. One entry point of the core
//! (`surface::build_boundary`) behind plain labels: open patch, cylinder (1 pair), torus (2 pairs), hexagonal torus
//! (3 pairs), Möbius band, Klein bottle, each with its verdict or refusal reason; the sphere and the projective plane
//! stay cone metrics and show their Gauss-Bonnet disclinations. Sizes always come from these options (no fixed
//! `[h,0],[0,k]` torus, no fixed glide 0).

use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::surface::report_bc::NOT_ISOMETRIC_NOTE;
use papertilling_core::surface::{BoundaryCondition, build_boundary, check_surface, glide_search};
use papertilling_core::types::{Cell2D, SurfaceCheckReport, SurfaceMesh};

use crate::backend::{CatalogueItemUi, SurfaceTargetUi, VerdictKindUi};

/// What a built boundary condition adds to the mesh: the exact-domain flag, the 3D caption and the plain-words read-out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BcNotes {
    /// True when the 2D chart is the exact flat isometric domain (`true` for the six flat boundary conditions,
    /// `false` for cone surfaces and when no surface is built).
    pub exact_view: bool,
    /// Caption required on the 3D picture (`None` for the flat open patch, the isometric cylinder and cone polyhedra).
    pub picture_note: Option<&'static str>,
    pub readout: Vec<String>,
}

/// The chosen options. `explicit` is set once the user edits a value here; until then the legacy `(h, k)` sizes of
/// `SurfaceBuildParamsUi` (presets, session, tests) are mapped onto the same options by [`BcControls::from_hk`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BcControls {
    pub explicit: bool,
    /// Growth layers of the open patch (0 = vertex-star seed S₀).
    pub layers: i32,
    /// Cylinder: circumference vector and rows, in cell vectors.
    pub wrap: [i32; 2],
    pub rows: i32,
    /// Torus and hexagonal torus: the supercell `Λ' = span(t1, t2)`, in cell vectors.
    pub t1: [i32; 2],
    pub t2: [i32; 2],
    /// Möbius and Klein: glide class, glide steps along the band, rows (Möbius) or cells across (Klein).
    pub glide: usize,
    pub steps: i32,
    pub width: i32,
    pub across: i32,
}

impl Default for BcControls {
    fn default() -> Self {
        Self {
            explicit: false,
            layers: 4,
            wrap: [4, 0],
            rows: 4,
            t1: [4, 0],
            t2: [0, 4],
            glide: 0,
            steps: 5,
            width: 3,
            across: 3,
        }
    }
}

impl BcControls {
    pub const MAX_LAYERS: i32 = 8;

    /// The legacy `(h, k)` sizes as options: torus `(h,0),(0,k)`, cylinder wrap `(0,k)` with `h` rows, Möbius and
    /// Klein with `2k+1` steps of glide class 0 and `h` rows or cells.
    pub fn from_hk(h: i32, k: i32) -> Self {
        let (h, k) = (h.max(1), k.max(1));
        Self {
            explicit: false,
            wrap: [0, k],
            rows: h,
            t1: [h, 0],
            t2: [0, k],
            steps: 2 * k + 1,
            width: h,
            across: h,
            ..Self::default()
        }
    }

    pub fn clamp(&mut self) {
        let v = |p: &mut [i32; 2]| p.iter_mut().for_each(|c| *c = (*c).clamp(-12, 12));
        v(&mut self.wrap);
        v(&mut self.t1);
        v(&mut self.t2);
        self.layers = self.layers.clamp(0, Self::MAX_LAYERS);
        self.rows = self.rows.clamp(1, 24);
        self.steps = self.steps.clamp(1, 25);
        self.width = self.width.clamp(1, 12);
        self.across = self.across.clamp(1, 12);
    }

    /// The core boundary condition of `target`, or `None` for a target that is not one (sphere, projective plane,
    /// genus 2 and 3 use their own builders).
    pub fn boundary(&self, target: SurfaceTargetUi) -> Option<BoundaryCondition> {
        Some(match target {
            SurfaceTargetUi::Open => BoundaryCondition::Open { layers: self.layers.clamp(0, Self::MAX_LAYERS) as usize },
            SurfaceTargetUi::Cylinder => BoundaryCondition::Cylinder { wrap: self.wrap, rows: self.rows },
            SurfaceTargetUi::Torus => BoundaryCondition::Torus { t1: self.t1, t2: self.t2 },
            SurfaceTargetUi::TorusHex => BoundaryCondition::TorusHex { t1: self.t1, t2: self.t2 },
            SurfaceTargetUi::MobiusBand => {
                BoundaryCondition::Mobius { glide: self.glide, steps: self.steps, width: self.width }
            }
            SurfaceTargetUi::KleinBottle => {
                BoundaryCondition::Klein { glide: self.glide, steps: self.steps, across: self.across }
            }
            _ => return None,
        })
    }
}

/// The six targets that are boundary conditions, in the order of the selector.
pub const BC_TARGETS: [SurfaceTargetUi; 6] = [
    SurfaceTargetUi::Open,
    SurfaceTargetUi::Cylinder,
    SurfaceTargetUi::Torus,
    SurfaceTargetUi::TorusHex,
    SurfaceTargetUi::MobiusBand,
    SurfaceTargetUi::KleinBottle,
];

/// Plain words for the selector (default view).
pub fn plain_label(t: SurfaceTargetUi) -> &'static str {
    match t {
        SurfaceTargetUi::Open => "Open patch",
        SurfaceTargetUi::Cylinder => "Cylinder (1 pair of sides)",
        SurfaceTargetUi::Torus => "Torus (2 pairs of sides)",
        SurfaceTargetUi::TorusHex => "Hexagonal torus (3 pairs)",
        SurfaceTargetUi::MobiusBand => "Möbius band",
        SurfaceTargetUi::KleinBottle => "Klein bottle",
        SurfaceTargetUi::Sphere => "Sphere (cone metric)",
        SurfaceTargetUi::ProjectivePlane => "Projective plane (cone metric)",
        SurfaceTargetUi::Genus2 => "Genus 2 surface (cone metric)",
        SurfaceTargetUi::Genus3 => "Genus 3 surface (cone metric)",
    }
}

/// Caption required on the 3D picture of `target`: [`NOT_ISOMETRIC_NOTE`] on every torus, Möbius, Klein or
/// higher-genus handlebody (`auto_size` and explicit options alike), `None` on isometric `Open | Cylinder` and
/// polyhedral cone metrics.
pub fn target_picture_note(target: SurfaceTargetUi) -> Option<&'static str> {
    match target {
        SurfaceTargetUi::Torus
        | SurfaceTargetUi::TorusHex
        | SurfaceTargetUi::MobiusBand
        | SurfaceTargetUi::KleinBottle
        | SurfaceTargetUi::Genus2
        | SurfaceTargetUi::Genus3 => Some(NOT_ISOMETRIC_NOTE),
        SurfaceTargetUi::Open
        | SurfaceTargetUi::Cylinder
        | SurfaceTargetUi::Sphere
        | SurfaceTargetUi::ProjectivePlane => None,
    }
}

/// Whether `target` is drawn on an exact flat 2D fundamental domain (`true` for the six flat boundary conditions,
/// `false` for cone metrics).
pub fn target_exact_view(target: SurfaceTargetUi) -> bool {
    BC_TARGETS.contains(&target)
}

/// [`BcNotes`] for a surface built outside `build_boundary` (`auto_size` table sizes, cone polyhedra, genus covers).
pub fn notes_for_built(mesh: &SurfaceMesh, target: SurfaceTargetUi) -> BcNotes {
    BcNotes {
        exact_view: target_exact_view(target),
        picture_note: target_picture_note(target),
        readout: cone_note(mesh, target).into_iter().collect(),
    }
}

/// Build `target` from the options (core refusals keep their `IMPOSSIBLE (...)` / `NOT FOUND` verdict prefix).
pub fn build_bc(
    item: &CatalogueItemUi,
    target: SurfaceTargetUi,
    ctl: &BcControls,
) -> Result<(SurfaceMesh, SurfaceCheckReport, BcNotes), String> {
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame))?.cell();
    build_bc_on_cell(&cell, target, ctl)
}

pub(crate) fn build_bc_on_cell(
    cell: &Cell2D,
    target: SurfaceTargetUi,
    ctl: &BcControls,
) -> Result<(SurfaceMesh, SurfaceCheckReport, BcNotes), String> {
    let bc = ctl.boundary(target).ok_or_else(|| format!("{} is not a boundary condition", target.label()))?;
    let built = build_boundary(cell, &bc).map_err(crate::live_ops::surface_error_message)?;
    let report = check_surface(&built.mesh);
    let notes = BcNotes {
        exact_view: built.exact_view,
        picture_note: built.picture_note,
        readout: built.readout,
    };
    Ok((built.mesh, report, notes))
}

/// Gauss-Bonnet read-out of a sphere, projective-plane or higher-genus cone metric: the disclination / branch-cone
/// angles add up to `360° × χ` (+720° on the sphere, +360° on the projective plane, −720° on genus 2, −1440° on
/// genus 3). `None` for any flat boundary condition.
pub fn cone_note(mesh: &SurfaceMesh, target: SurfaceTargetUi) -> Option<String> {
    let (expected, name) = match target {
        SurfaceTargetUi::Sphere => (720.0, "sphere"),
        SurfaceTargetUi::ProjectivePlane => (360.0, "projective plane"),
        SurfaceTargetUi::Genus2 => (-720.0, "genus 2 surface"),
        SurfaceTargetUi::Genus3 => (-1440.0, "genus 3 surface"),
        _ => return None,
    };
    let sum: f64 = mesh.cones.iter().map(|c| c.angle_deg).sum();
    Some(format!(
        "Cone metric, not flat: the {name} cannot carry a flat lattice, so the curvature sits at {} cone points \
         (disclinations). Their angles add up to {sum:+.0}°; Gauss-Bonnet requires {expected:+.0}° (360° × χ).",
        mesh.cones.len()
    ))
}

/// Verdict of one boundary condition on the lattice on screen, at its default options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BcVerdict {
    pub target: SurfaceTargetUi,
    pub kind: VerdictKindUi,
    /// "OK" detail (the read-out lines, joined) or the refusal reason.
    pub text: String,
}

/// The six verdicts and the glide classes of one lattice (cached by the tab per lattice).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BcSummary {
    pub verdicts: Vec<BcVerdict>,
    pub glides: Vec<String>,
}

impl BcSummary {
    pub fn verdict(&self, t: SurfaceTargetUi) -> Option<&BcVerdict> {
        self.verdicts.iter().find(|v| v.target == t)
    }
}

pub(crate) fn kind_of(msg: &str) -> VerdictKindUi {
    if msg.starts_with("IMPOSSIBLE") { VerdictKindUi::Impossible } else { VerdictKindUi::NotFound }
}

/// Run the six boundary conditions at their default options on `item` and list its glide classes.
pub fn summarize(item: &CatalogueItemUi) -> Option<BcSummary> {
    let cell = LatticeFamily::from_recipe(&item.recipe, Some(item.frame)).ok()?.cell();
    let ctl = BcControls::default();
    let verdicts = BC_TARGETS
        .iter()
        .map(|&target| match build_bc_on_cell(&cell, target, &ctl) {
            Ok((_, _, n)) => {
                BcVerdict { target, kind: VerdictKindUi::Ok, text: n.readout.join("; ") }
            }
            Err(e) => BcVerdict { target, kind: kind_of(&e), text: e },
        })
        .collect();
    let glides = glide_search(&cell)
        .unwrap_or_default()
        .iter()
        .map(|g| {
            let kind = if g.mirror { "mirror, then shift" } else { "glide" };
            format!("{kind} along ({}, {})", g.along[0], g.along[1])
        })
        .collect();
    Some(BcSummary { verdicts, glides })
}
