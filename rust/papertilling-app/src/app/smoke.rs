//! Real native GPU smoke coverage of five surface states and both new 3D sheet tabs.
use std::sync::atomic::Ordering;

use super::{NavTab, PapertillingApp, SurfaceTargetUi};
use crate::mesh3d::DefectViewMode;

const MAX_SMOKE_FRAMES: u32 = 1800;

impl PapertillingApp {
    fn smoke_stage_waiting(&self) -> bool {
        match self.active_tab {
            NavTab::Surfaces => {
                self.surfaces.is_building() || self.surfaces.elastic.is_busy() || self.surfaces.spectrum.is_busy()
            }
            NavTab::Defects => {
                self.defects.is_busy()
                    || self.defects.initializing
                    || self.defects.view3d_dirty
                    || self.defects.inspectors.spectrum.is_busy()
            }
            NavTab::Combine => {
                self.combine.is_building()
                    || self.combine.view3d_dirty
                    || self.combine.inspectors.spectrum.is_busy()
            }
            _ => false,
        }
    }

    fn smoke_stage_error(&self) -> bool {
        match self.active_tab {
            NavTab::Surfaces => {
                self.surfaces.has_refusal() || self.surfaces.elastic.last_error.is_some()
            }
            NavTab::Defects => {
                self.defects.error.is_some()
                    || self.defects.inspectors.spectrum.last_error.is_some()
            }
            NavTab::Combine => {
                self.combine.init_error.is_some()
                    || self.combine.last_build.refused.is_some()
                    || self.combine.inspectors.spectrum.last_error.is_some()
            }
            NavTab::Gallery | NavTab::Atlas => false,
        }
    }

    pub(super) fn smoke_step(&mut self, ctx: &egui::Context) -> bool {
        self.smoke_frames = self.smoke_frames.saturating_add(1);
        let (prepared, painted) = self.view3d.counters.counts();
        let waiting = self.smoke_stage_waiting();
        let stage_error = self.smoke_stage_error();
        let mesh_gen = self.view3d.mesh_generation();
        let cap = self.view3d.caption();

        let failed = (!waiting && stage_error)
            || (!waiting && mesh_gen > self.smoke_gen_at_switch && !self.view3d.has_geometry())
            || (!waiting && (cap.contains("unavailable") || cap.contains("Build failed")))
            || self.smoke_frames > MAX_SMOKE_FRAMES;
        if failed {
            self.smoke_outcome.store(2, Ordering::Release);
            if !self.smoke_reported {
                self.smoke_reported = true;
                eprintln!(
                    "PAPERTILLING_SMOKE_FAIL stage={} frames={} waiting={waiting} error={stage_error} geom={} caption={cap}",
                    self.last_smoke_stage, self.smoke_frames, self.view3d.has_geometry()
                );
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return true;
        }

        let ready = !waiting
            && !stage_error
            && prepared > 0
            && mesh_gen > self.smoke_gen_at_switch
            && painted > self.smoke_frame_at_switch
            && self.view3d.has_geometry()
            && self.view3d.is_current_mesh_painted()
            && !cap.contains("stale")
            && !cap.contains("unavailable");

        if ready {
            if self.last_smoke_stage == 11 {
                self.smoke_outcome.store(1, Ordering::Release);
                if !self.smoke_reported {
                    self.smoke_reported = true;
                    println!(
                        "PAPERTILLING_SMOKE_OK adapter={} backend={} frames={painted} uploads={} stages=12 (surfaces,defects-sheet,combine-sheet,fvk,ao-mobius,ldos,atlas-3d,atlas-h2)",
                        self.adapter_name, self.backend_name, self.view3d.counters.uploads()
                    );
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return true;
            }
            self.last_smoke_stage += 1;
            self.smoke_frame_at_switch = painted;
            self.smoke_gen_at_switch = mesh_gen;
            match self.last_smoke_stage {
                10 => {
                    self.atlas.sub_tab = crate::atlas_tab::AtlasSubTab::SpaceGroups3D;
                    self.atlas.dirty = true;
                    self.switch_tab(NavTab::Atlas);
                }
                11 => {
                    self.atlas.sub_tab = crate::atlas_tab::AtlasSubTab::HyperbolicH2;
                    self.atlas.sync_scenes(&mut self.canvas2d, &mut self.view3d);
                }
                5 => {
                    self.defects.view_mode = DefectViewMode::Sheet3D;
                    self.switch_tab(NavTab::Defects);
                }
                6 => {
                    self.combine.view_mode = DefectViewMode::Sheet3D;
                    self.switch_tab(NavTab::Combine);
                }
                7 => {
                    self.surfaces.target = SurfaceTargetUi::Sphere;
                    self.surfaces.show_cover = false;
                    self.surfaces.elastic.shape_mode = crate::surfaces_elastic::SurfaceShapeMode::ElasticFvK;
                    self.switch_tab(NavTab::Surfaces);
                }
                8 => {
                    self.surfaces.elastic.shape_mode = crate::surfaces_elastic::SurfaceShapeMode::AnalyticRadial;
                    let mob_idx = self.defects.catalogue.walls.iter()
                        .position(|w| w.kind == crate::backend::WallKindUi::MobiusWc)
                        .unwrap_or(0);
                    self.defects.sub_tab = crate::defects_tab::DefectSubTab::Walls;
                    self.defects.selected_wall = mob_idx;
                    self.defects.wall_half = false;
                    self.defects.draw_mode = crate::live_ops_ao::Defect2dMode::ElasticAoMetric;
                    self.defects.relaxed = true;
                    self.defects.view_mode = DefectViewMode::Sheet3D;
                    self.switch_tab(NavTab::Defects);
                }
                9 => {
                    self.defects.sheet_opts.color_mode = crate::mesh3d::Sheet3dColorMode::SpectralLdos;
                    self.defects.inspectors.on_sheet_color_changed(crate::mesh3d::Sheet3dColorMode::SpectralLdos);
                    self.defects.inspectors.spectrum.submit_crystal(
                        &self.defects.last_crystal,
                        Some(&self.defects.last_sheet),
                        Some(ctx),
                    );
                    self.defects.view3d_dirty = true;
                }
                stage => {
                    let (target, cover) = match stage {
                        1 => (SurfaceTargetUi::Torus, false),
                        2 => (SurfaceTargetUi::ProjectivePlane, true),
                        3 => (SurfaceTargetUi::KleinBottle, false),
                        _ => (SurfaceTargetUi::Genus2, false),
                    };
                    self.surfaces.target = target;
                    self.surfaces.show_cover = cover;
                    self.surfaces.activate_and_sync(
                        self.gallery.active_item(), &self.backend, &mut self.canvas2d, &mut self.view3d,
                    );
                }
            }
        }
        self.view3d.yaw += 0.14;
        ctx.request_repaint();
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::mesh3d::GpuMeshData;

    fn step_with_mock_paint(app: &mut PapertillingApp, ctx: &egui::Context, paint_nonempty: bool) -> bool {
        if app.view3d.has_geometry() && paint_nonempty {
            app.view3d.counters.record_mock_paint(app.view3d.mesh_generation(), 12);
        } else {
            app.view3d.counters.record_mock_paint(app.view3d.mesh_generation(), 0);
        }
        let mut closed = false;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            closed = app.frame_ui(ui);
        });
        closed
    }

    #[test]
    fn smoke_succeeds_only_after_all_seven_stages_paint_valid_geometry() {
        let ctx = egui::Context::default();
        let outcome = Arc::new(AtomicU8::new(0));
        let mut app = PapertillingApp::new(true, String::from("mock"), String::from("mock"))
            .with_smoke_outcome(Arc::clone(&outcome));
        for _ in 0..300 {
            if step_with_mock_paint(&mut app, &ctx, true) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(app.last_smoke_stage, 11);
        assert_eq!(outcome.load(Ordering::Acquire), 1);
    }

    #[test]
    fn smoke_fails_when_a_sheet_stage_has_an_error_or_empty_geometry() {
        let ctx = egui::Context::default();
        let outcome = Arc::new(AtomicU8::new(0));
        let mut app = PapertillingApp::new(true, String::from("mock"), String::from("mock"))
            .with_smoke_outcome(Arc::clone(&outcome));
        while app.last_smoke_stage < 5 {
            assert!(!step_with_mock_paint(&mut app, &ctx, true));
        }
        app.defects.wait_idle(&mut app.canvas2d);
        app.defects.take_result(Err(String::from("injected stage 5 error")), &mut app.canvas2d);
        app.defects.sync_view3d(&mut app.view3d);
        assert!(step_with_mock_paint(&mut app, &ctx, true));
        assert_eq!(outcome.load(Ordering::Acquire), 2);

        let outcome2 = Arc::new(AtomicU8::new(0));
        let mut app2 = PapertillingApp::new(true, String::from("mock"), String::from("mock"))
            .with_smoke_outcome(Arc::clone(&outcome2));
        while app2.last_smoke_stage < 6 {
            assert!(!step_with_mock_paint(&mut app2, &ctx, true));
            std::thread::sleep(Duration::from_millis(5));
        }
        let item = app2.gallery.active_item().clone();
        app2.combine.wait_idle(&item, Some(&mut app2.canvas2d));
        app2.combine.sync_view3d(&item, &mut app2.view3d);
        app2.view3d.set_mesh(GpuMeshData::default(), "empty stage 6");
        assert!(step_with_mock_paint(&mut app2, &ctx, false));
        assert_eq!(outcome2.load(Ordering::Acquire), 2);
    }
}
