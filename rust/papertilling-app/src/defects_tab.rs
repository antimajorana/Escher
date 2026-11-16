use crate::backend::{
    AppBackend, CatalogueItemUi, DefectCatalogueUi, DefectReadoutUi, DefectSelectionUi,
};
use crate::canvas2d::{Canvas2D, Scene2D};
use crate::combine_worker::{LatestWorker, empty_crystal, empty_sheet3d};
use crate::mesh3d::{
    DefectViewMode, GpuMeshData, Sheet3dViewOpts, mesh_from_sheet3d, sheet3d_caption,
};
use crate::view3d::View3dState;
use papertilling_core::defect::sheet3d::Sheet3DResult;
use papertilling_core::types::{BuiltCrystal, Cell2D};

struct BuildJob {
    item: CatalogueItemUi,
    selection: DefectSelectionUi,
    radius: f64,
    relaxed: bool,
    kappa_b: f64,
    reuse_crystal: Option<(DefectCatalogueUi, BuiltCrystal, DefectReadoutUi)>,
}

type BuildOutput = (
    DefectCatalogueUi,
    BuiltCrystal,
    DefectReadoutUi,
    Sheet3DResult,
    BuiltCrystal,
    Sheet3DResult,
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefectSubTab {
    Dislocations,
    Disclinations,
    Walls,
}

impl DefectSubTab {
    pub const ALL: [Self; 3] = [Self::Dislocations, Self::Disclinations, Self::Walls];

    pub fn label(self) -> &'static str {
        match self {
            Self::Dislocations => "Dislocations",
            Self::Disclinations => "Disclinations",
            Self::Walls => "Walls (Wa-Wd)",
        }
    }
}

pub struct DefectsTabState {
    pub sub_tab: DefectSubTab,
    pub(crate) catalogue: DefectCatalogueUi,
    pub(crate) selected_disloc: usize,
    pub(crate) selected_disclin: usize,
    pub(crate) selected_wall: usize,
    pub radius: f64,
    pub relaxed: bool,
    pub view_mode: DefectViewMode,
    pub sheet_opts: Sheet3dViewOpts,
    pub readout: DefectReadoutUi,
    pub(crate) stale: bool,
    pub(crate) initializing: bool,
    worker: LatestWorker<BuildJob, BuildOutput>,
    last_scene: Option<Scene2D>,
    last_crystal: BuiltCrystal,
    pub(crate) last_sheet: Sheet3DResult,
    flat_crystal: BuiltCrystal,
    pub(crate) flat_sheet: Sheet3DResult,
    pub(crate) view3d_dirty: bool,
    reset_on_result: bool,
    pub(crate) error: Option<String>,
}

impl DefectsTabState {
    pub fn new(item: &CatalogueItemUi, backend: &AppBackend) -> Self {
        let catalogue = backend.defect_catalogue(item);
        let radius = 3.8;
        let relaxed = true;
        let sheet_opts = Sheet3dViewOpts::default();
        let (crystal, readout) =
            backend.build_defect(item, &DefectSelectionUi::Dislocation(0), radius, relaxed);
        let (last_sheet, flat_crystal, flat_sheet) =
            crate::live_ops::build_defect_sheets_3d(item, &crystal, radius, sheet_opts.kappa_b);
        let last_scene = Some(Scene2D::CrystalPatch {
            crystal: crystal.clone(),
            title: readout.title.clone(),
            note: readout.note.clone(),
        });
        Self {
            sub_tab: DefectSubTab::Dislocations,
            catalogue,
            selected_disloc: 0,
            selected_disclin: 0,
            selected_wall: 0,
            radius,
            relaxed,
            view_mode: DefectViewMode::Split2D3D,
            sheet_opts,
            readout,
            stale: false,
            initializing: false,
            worker: LatestWorker::spawn("defects-build", (), |_: &mut (), job: BuildJob| {
                let (catalogue, crystal, readout) = match job.reuse_crystal {
                    Some(prev) => prev,
                    None => {
                        let cat = crate::live_ops::catalogue_for_lattice(&job.item);
                        let (c, r) = crate::live_ops::build_defect_patch(
                            &job.item,
                            &job.selection,
                            job.radius,
                            job.relaxed,
                        );
                        (cat, c, r)
                    }
                };
                let (sheet, flat_c, flat_s) = crate::live_ops::build_defect_sheets_3d(
                    &job.item,
                    &crystal,
                    job.radius,
                    job.kappa_b,
                );
                (catalogue, crystal, readout, sheet, flat_c, flat_s)
            }),
            last_scene,
            last_crystal: crystal,
            last_sheet,
            flat_crystal,
            flat_sheet,
            view3d_dirty: true,
            reset_on_result: false,
            error: None,
        }
    }

    pub(crate) fn current_selection(&self) -> DefectSelectionUi {
        match self.sub_tab {
            DefectSubTab::Dislocations => DefectSelectionUi::Dislocation(self.selected_disloc),
            DefectSubTab::Disclinations => DefectSelectionUi::Disclination(self.selected_disclin),
            DefectSubTab::Walls => DefectSelectionUi::Wall(self.selected_wall),
        }
    }

    pub fn reload_for_item(
        &mut self,
        item: &CatalogueItemUi,
        backend: &AppBackend,
        canvas: &mut Canvas2D,
        sync_canvas: bool,
    ) {
        self.selected_disloc = 0;
        self.selected_disclin = 0;
        self.selected_wall = 0;
        self.worker.cancel();
        self.catalogue = DefectCatalogueUi::default();
        self.last_scene = None;
        self.last_crystal = empty_crystal();
        self.last_sheet = empty_sheet3d();
        self.flat_crystal = empty_crystal();
        self.flat_sheet = empty_sheet3d();
        self.readout = DefectReadoutUi::default();
        self.error = None;
        self.initializing = true;
        self.stale = true;
        self.view3d_dirty = true;
        if sync_canvas {
            self.rebuild_crystal(item, backend, canvas, true);
        }
    }

    pub fn rebuild_crystal(
        &mut self,
        item: &CatalogueItemUi,
        _backend: &AppBackend,
        canvas: &mut Canvas2D,
        reset_view: bool,
    ) {
        self.worker.submit(BuildJob {
            item: item.clone(),
            selection: self.current_selection(),
            radius: self.radius,
            relaxed: self.relaxed,
            kappa_b: self.sheet_opts.kappa_b,
            reuse_crystal: None,
        });
        self.stale = false;
        self.error = None;
        self.view3d_dirty = true;
        self.reset_on_result |= reset_view;
        let mut scene = self.last_scene.clone().unwrap_or_else(|| Scene2D::PeriodicCell {
            cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: Vec::new(), bonds: Vec::new() },
            repeat: 1,
            show_unit_cell: false,
        });
        if let Scene2D::CrystalPatch { note, .. } = &mut scene {
            *note = String::from("Previous result — stale while the new defect builds in the background.");
        }
        canvas.set_scene(scene, reset_view);
    }

    pub fn rebuild_sheet3d_only(&mut self, item: &CatalogueItemUi) {
        let reuse = (!self.is_busy() && !self.initializing && !self.last_crystal.pos.is_empty()).then(|| {
            (self.catalogue.clone(), self.last_crystal.clone(), self.readout.clone())
        });
        self.worker.submit(BuildJob {
            item: item.clone(),
            selection: self.current_selection(),
            radius: self.radius,
            relaxed: self.relaxed,
            kappa_b: self.sheet_opts.kappa_b,
            reuse_crystal: reuse,
        });
        self.error = None;
        self.view3d_dirty = true;
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.stale || self.worker.is_busy()
    }

    pub(crate) fn poll_worker(&mut self, ctx: &egui::Context, canvas: &mut Canvas2D) {
        self.worker.set_repaint_ctx(ctx);
        if let Some(result) = self.worker.poll() {
            self.take_result(result, canvas);
        }
    }

    pub fn sync_view3d(&mut self, view3d: &mut View3dState) {
        if self.initializing {
            view3d.set_mesh(GpuMeshData::default(), String::from("Preparing lattice…"));
            self.view3d_dirty = false;
            return;
        }
        if let Some(err) = &self.error
            && self.last_crystal.pos.is_empty()
        {
            view3d.set_mesh(GpuMeshData::default(), format!("3D unavailable: {err}"));
            self.view3d_dirty = false;
            return;
        }
        let (c, s) = if self.sheet_opts.show_flat_baseline {
            (&self.flat_crystal, &self.flat_sheet)
        } else {
            (&self.last_crystal, &self.last_sheet)
        };
        let gpu = mesh_from_sheet3d(c, s, &self.sheet_opts);
        let mut cap = sheet3d_caption(&self.readout.title, s, &self.sheet_opts);
        if self.is_busy() {
            cap.push_str(" | previous result — stale");
        } else if let Some(err) = &self.error {
            cap.push_str(&format!(" | previous result — failed: {err}"));
        }
        view3d.set_mesh(gpu, cap);
        self.view3d_dirty = false;
    }

    #[cfg(test)]
    pub(crate) fn fail_build_for_test(&mut self, err: &str, canvas: &mut Canvas2D) {
        self.worker.cancel();
        self.stale = false;
        self.take_result(Err(err.to_owned()), canvas);
    }

    pub(crate) fn take_result(&mut self, result: Result<BuildOutput, String>, canvas: &mut Canvas2D) {
        self.initializing = false;
        match result {
            Ok((catalogue, crystal, readout, sheet, flat_c, flat_s)) => {
                self.catalogue = catalogue;
                let scene = Scene2D::CrystalPatch {
                    crystal: crystal.clone(),
                    title: readout.title.clone(),
                    note: readout.note.clone(),
                };
                canvas.set_scene(scene.clone(), self.reset_on_result);
                self.last_scene = Some(scene);
                self.last_crystal = crystal;
                self.last_sheet = sheet;
                self.flat_crystal = flat_c;
                self.flat_sheet = flat_s;
                self.readout = readout;
                self.view3d_dirty = true;
            }
            Err(err) => {
                self.error = Some(err.clone());
                self.view3d_dirty = true;
                if let Some(mut scene) = self.last_scene.clone() {
                    if let Scene2D::CrystalPatch { note, .. } = &mut scene {
                        *note = format!("Build failed (showing previous result): {err}");
                    }
                    canvas.set_scene(scene, false);
                }
            }
        }
        self.reset_on_result = false;
    }

    #[cfg(test)]
    pub(crate) fn wait_idle(&mut self, canvas: &mut Canvas2D) {
        if let Some(result) = self.worker.wait(std::time::Duration::from_secs(120)) {
            self.take_result(result, canvas);
        }
        assert!(!self.is_busy(), "defect build did not complete");
    }
}

#[cfg(test)]
#[path = "defects_tab_tests.rs"]
pub(crate) mod tests;
