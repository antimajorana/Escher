//! "Export 3D PNG...": guards (the same busy / stale rules as the 2D and .obj exports), the save dialog, and the
//! smoke-test export that exercises the real GPU path on every `--smoke-test` run.

use std::path::Path;
use std::sync::mpsc;
use std::thread::JoinHandle;

use super::{NavTab, PapertillingApp};
use crate::view3d::{Export3dInfo, OffscreenGpuContext};

impl PapertillingApp {
    /// GPU handles for offscreen exports (`None`: headless or no wgpu; the 3D PNG export then reports an error).
    pub fn with_offscreen_gpu(mut self, gpu: Option<OffscreenGpuContext>) -> Self {
        self.offscreen_gpu = gpu;
        self
    }

    /// Why a 3D PNG export is refused in egui pass `pass_nr` (`None`: allowed): the state guards of
    /// [`Self::export_3d_png_state_refusal`], then a missing GPU context, all before the save dialog opens.
    pub(crate) fn export_3d_png_refusal(&self, pass_nr: u64) -> Option<String> {
        self.export_3d_png_state_refusal(pass_nr).or_else(|| {
            self.offscreen_gpu.is_none().then(|| String::from("3D PNG export needs the wgpu renderer, which is not available."))
        })
    }

    /// The tab / build / view guards of the 3D PNG export (no GPU check).
    pub(crate) fn export_3d_png_state_refusal(&self, pass_nr: u64) -> Option<String> {
        let refuse = |s: &str| Some(String::from(s));
        if self.export3d_job.is_some() {
            return refuse("A 3D PNG export is still being written.");
        }
        if self.active_tab == NavTab::Gallery {
            return refuse("Open Surfaces, Defects, Combine, Atlas, or Analysis to export a 3D PNG.");
        }
        // before the stale-view guards: a 2D-only view never clears view3d_dirty, so "wait" would never resolve
        if self.view3d.shown_view_size(pass_nr).is_none() {
            return refuse("Show the 3D view before exporting a 3D PNG.");
        }
        // a pending spectrum / relaxation would leave the previous result's colours and colorbar in the picture
        let pending_colours = match self.active_tab {
            NavTab::Surfaces => self.surfaces.elastic.is_busy(),
            NavTab::Analysis => self.analysis.is_busy() || self.analysis_source_busy(),
            _ => false,
        };
        if pending_colours {
            return refuse("Wait for the spectrum / relaxation to finish before exporting the 3D PNG.");
        }
        match self.active_tab {
            NavTab::Defects if self.defects.is_busy() => return refuse("Wait for the defect build before exporting the 3D PNG."),
            NavTab::Defects if self.defects.view3d_dirty => {
                return refuse("Wait for the 3D sheet to update before exporting the 3D PNG.");
            }
            NavTab::Surfaces if self.surfaces.is_building() => {
                return refuse("Wait for the surface build before exporting the 3D PNG.");
            }
            NavTab::Combine if self.combine.is_building() || self.combine.dragging || self.combine.view3d_dirty => {
                return refuse("Finish moving pins and wait for the crystal build before exporting the 3D PNG.");
            }
            NavTab::Atlas if self.atlas.dirty => return refuse("Wait for the Atlas 3D scene to update before exporting the 3D PNG."),
            NavTab::Workbench if self.workbench_busy() || self.workbench.dirty => {
                return refuse("Wait for the 3D scene to update before exporting the 3D PNG.");
            }
            NavTab::Analysis if self.analysis.view3d_dirty => {
                return refuse("Wait for the 3D spectral view to update before exporting the 3D PNG.");
            }
            _ => {}
        }
        if !self.view3d.has_geometry() {
            return refuse("No 3D mesh to export as PNG.");
        }
        None
    }

    /// Export the 3D view to `path` at the PNG scale (no dialog, no guards: callers check the refusal first).
    pub(crate) fn export_3d_png_to(&self, path: &Path) -> Result<Export3dInfo, String> {
        let gpu = self
            .offscreen_gpu
            .as_ref()
            .ok_or_else(|| String::from("3D PNG export needs the wgpu renderer, which is not available."))?;
        self.view3d.export_png(gpu, &self.theme.tokens(), u32::from(self.png_scale), path)
    }

    /// The menu action: guards, save dialog, then the GPU render here (UI thread, error scopes are per thread) and
    /// the overlay compositing, PNG encoding and file write on a worker thread ([`Self::poll_export_3d`]).
    pub(crate) fn export_3d_png_dialog(&mut self, ctx: &egui::Context) {
        if let Some(msg) = self.export_3d_png_refusal(ctx.cumulative_pass_nr()) {
            self.export_status = msg;
            return;
        }
        let wb_name = format!("{}_mesh.png", self.workbench.view.file_stem());
        let default_name = match self.active_tab {
            NavTab::Gallery => "mesh.png",
            NavTab::Surfaces => "surface_mesh.png",
            NavTab::Defects => "defect_sheet.png",
            NavTab::Combine => "combined_sheet.png",
            NavTab::Atlas => "atlas_mesh.png",
            NavTab::Analysis => "analysis_ldos.png",
            NavTab::Workbench => wb_name.as_str(),
        };
        let dialog = rfd::FileDialog::new()
            .set_title("Export 3D view (PNG)")
            .set_file_name(default_name)
            .add_filter("PNG image", &["png"]);
        if let Some(path) = dialog.save_file() {
            let path = match png_path(path) {
                Ok(p) => p,
                Err(msg) => {
                    self.export_status = msg;
                    return;
                }
            };
            let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
            let Some(gpu) = self.offscreen_gpu.as_ref() else {
                self.export_status = String::from("3D PNG export error: the wgpu renderer is not available.");
                return;
            };
            let rendered = match self.view3d.render_export(gpu, &self.theme.tokens(), u32::from(self.png_scale)) {
                Ok(r) => r,
                Err(err) => {
                    self.export_status = format!("3D PNG export error: {err}");
                    return;
                }
            };
            let scale = self.png_scale;
            let spawned = Export3dJob::spawn(Some(ctx.clone()), move || match rendered.finish(&path) {
                Ok(info) => export_3d_status(info, scale, &name),
                Err(err) => format!("3D PNG export error: {err}"),
            });
            match spawned {
                Ok(job) => {
                    self.export3d_job = Some(job);
                    self.export_status = String::from("Exporting 3D PNG...");
                }
                Err(err) => self.export_status = format!("3D PNG export error: cannot start the writer thread: {err}"),
            }
        }
    }

    /// Take the status of a finished background 3D PNG write (called every frame).
    pub(crate) fn poll_export_3d(&mut self) {
        if let Some(job) = &self.export3d_job {
            match job.rx.try_recv() {
                Ok(status) => {
                    self.export_status = status;
                    self.export3d_job = None;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.export_status = String::from("3D PNG export error: the writer thread stopped.");
                    self.export3d_job = None;
                }
            }
        }
    }

    /// Smoke run: export the current 3D view to a temp file and check that it decodes to the planned size.
    /// `Ok(None)` without a GPU context (the mock-paint unit tests).
    pub(super) fn smoke_export_3d(&self, pass_nr: u64) -> Result<Option<Export3dInfo>, String> {
        if self.offscreen_gpu.is_none() {
            return Ok(None);
        }
        if let Some(msg) = self.export_3d_png_refusal(pass_nr) {
            return Err(msg);
        }
        let path = std::env::temp_dir().join(format!("papertilling_smoke_3d_{}.png", std::process::id()));
        let info = self.export_3d_png_to(&path)?;
        let decoded = std::fs::read(&path).map_err(|e| e.to_string()).and_then(|bytes| {
            let reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().map_err(|e| e.to_string())?;
            Ok((reader.info().width, reader.info().height))
        });
        let _ = std::fs::remove_file(&path);
        // expected: the shown viewport's pixels x the PNG scale, rounded (or the clamped size, when it applied)
        let view = self.view3d.shown_view_pixels(pass_nr).unwrap_or_default() * f32::from(self.png_scale);
        let want = if info.clamped { (info.w, info.h) } else { (view.x.round() as u32, view.y.round() as u32) };
        match decoded? {
            (w, h) if (w, h) == want && want == (info.w, info.h) => Ok(Some(info)),
            (w, h) => Err(format!("3D PNG decodes to {w}x{h}, expected {}x{} (reported {}x{})", want.0, want.1, info.w, info.h)),
        }
    }
}

/// A 3D PNG being composited, encoded and written on a worker thread: its status arrives on `rx`; dropping the job
/// (the app closing mid-write) waits for the thread, so the process never exits halfway through the file.
pub(crate) struct Export3dJob {
    pub(crate) rx: mpsc::Receiver<String>,
    handle: Option<JoinHandle<()>>,
}

impl Export3dJob {
    /// Run `work` on a named thread, send its status and repaint `ctx` when done.
    pub(crate) fn spawn(ctx: Option<egui::Context>, work: impl FnOnce() -> String + Send + 'static) -> std::io::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::Builder::new().name(String::from("export3d_png")).spawn(move || {
            let _ = tx.send(work());
            if let Some(ctx) = ctx {
                ctx.request_repaint();
            }
        })?;
        Ok(Self { rx, handle: Some(handle) })
    }
}

impl Drop for Export3dJob {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Status line of a finished 3D PNG export.
pub(crate) fn export_3d_status(info: Export3dInfo, scale: u8, name: &str) -> String {
    let fit = if info.clamped { "; scaled down to fit the size limit" } else { "" };
    format!("Exported 3D {}x{} PNG at {scale}x ({name}{fit})", info.w, info.h)
}

/// The chosen file as a PNG path: no extension gets ".png", any other extension is refused (as the 2D export does).
pub(crate) fn png_path(path: std::path::PathBuf) -> Result<std::path::PathBuf, String> {
    match path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()) {
        None => Ok(path.with_extension("png")),
        Some(e) if e == "png" => Ok(path),
        Some(other) => Err(format!("cannot export the 3D view as a .{other} file: choose a .png file name")),
    }
}

