//! Worker jobs of the S4 panel: one variant per sub-row, run by [`run_job`] on the panel's `LatestWorker`.

use papertilling_core::flat3d::{
    FlatSlabKind, NanoribbonPatch, NanotubeMesh, PlatycosmQuotient3d, SlabPatch3d, SymmetricFlake, TorsionFreeReport,
};
use papertilling_core::space3d::nets3d::CanonicalNet3d;
use papertilling_core::types::Cell2D;

use super::flakes::{FlakeParams, Shape};
use super::slabs::GlideAxes;
use super::tubes::{RibbonParams, TubeParams};
use super::{flakes, platycosms, slabs, torsion_free, tubes};

/// The `Debug` form of a job is its cache key.
#[derive(Debug)]
pub(super) enum S4Job {
    Slab { net: CanonicalNet3d, kind: FlatSlabKind, repeats: [usize; 3] },
    Platy { net: CanonicalNet3d, code: String, supercell: Option<[usize; 3]>, ignore_kinds: bool },
    Tube { cell: Cell2D, p: TubeParams },
    Ribbon { cell: Cell2D, p: RibbonParams },
    Flake { group: String, cell: Cell2D, centre: [f64; 2], p: FlakeParams },
    Torsion,
}

pub(super) enum S4Out {
    /// The patch and, for the two glide-glued kinds, the glide's axes in the net's (a1, a2).
    Slab(Box<SlabPatch3d>, Option<GlideAxes>),
    Platy(Box<PlatycosmQuotient3d>),
    Tube(Box<NanotubeMesh>),
    Ribbon(Box<NanoribbonPatch>),
    /// The flake and the cut shape it was built with (the inspector names the cut from it).
    Flake(Box<SymmetricFlake>, Shape),
    Torsion(Vec<TorsionFreeReport>),
}

pub(super) fn run_job(job: S4Job) -> Result<S4Out, String> {
    Ok(match job {
        S4Job::Slab { net, kind, repeats } => {
            let p = slabs::build(net, kind, repeats)?;
            let glide = slabs::glide_axes(net, &p);
            S4Out::Slab(Box::new(p), glide)
        }
        S4Job::Platy { net, code, supercell, ignore_kinds } => {
            S4Out::Platy(Box::new(platycosms::build(net, &code, supercell, ignore_kinds)?))
        }
        S4Job::Tube { cell, p } => S4Out::Tube(Box::new(tubes::build_tube(&cell, &p)?)),
        S4Job::Ribbon { cell, p } => S4Out::Ribbon(Box::new(tubes::build_ribbon(&cell, &p)?)),
        S4Job::Flake { group, cell, centre, p } => S4Out::Flake(Box::new(flakes::build(&group, &cell, centre, &p)?), p.shape),
        S4Job::Torsion => S4Out::Torsion(torsion_free::build()?),
    })
}
