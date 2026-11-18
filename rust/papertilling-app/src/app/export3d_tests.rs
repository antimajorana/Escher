use super::export3d::{Export3dJob, export_3d_status};
use super::{NavTab, PapertillingApp};
use crate::view3d::Export3dInfo;

fn frame(app: &mut PapertillingApp, ctx: &egui::Context) {
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        let _ = app.frame_ui(ui);
    });
}

#[test]
fn export_3d_png_is_refused_on_the_gallery_and_while_dirty_or_busy() {
    let t0 = std::time::Instant::now();
    let ctx = egui::Context::default();
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    assert_eq!(app.active_tab, NavTab::Gallery);
    let msg = app.export_3d_png_refusal(ctx.cumulative_pass_nr()).expect("Gallery has no 3D view");
    assert!(msg.contains("Open Surfaces"), "{msg}");

    app.switch_tab(NavTab::Defects);
    app.defects.wait_idle(&mut app.canvas2d);
    assert!(!app.defects.is_busy());
    app.defects.view3d_dirty = true;
    let msg = app.export_3d_png_refusal(0).expect("stale 3D sheet");
    assert!(msg.contains("3D sheet to update"), "{msg}");
    app.defects.stale = true;
    let msg = app.export_3d_png_refusal(0).expect("defect build running");
    assert!(msg.contains("defect build"), "{msg}");

    app.switch_tab(NavTab::Combine);
    app.combine.dragging = true;
    let msg = app.export_3d_png_refusal(0).expect("pins moving");
    assert!(msg.contains("Finish moving pins"), "{msg}");
    eprintln!("# time guards {:?}", t0.elapsed());
}

#[test]
fn export_3d_png_is_allowed_once_the_view_is_shown_and_reports_a_missing_gpu() {
    let t0 = std::time::Instant::now();
    let ctx = egui::Context::default();
    let mut app = PapertillingApp::new(false, String::from("test"), String::from("test"));
    app.switch_tab(NavTab::Atlas);
    // not drawn yet: refused until a frame shows the 3D view
    assert!(app.export_3d_png_refusal(ctx.cumulative_pass_nr()).is_some());
    for _ in 0..3 {
        frame(&mut app, &ctx);
    }
    assert!(app.view3d.has_geometry(), "Atlas 3D scene built");
    // the next pass (a menu click in the following frame) sees the view drawn in the previous one
    let now = ctx.cumulative_pass_nr();
    assert_eq!(app.export_3d_png_state_refusal(now), None);
    // no GPU context in this process: refused before the save dialog, with the wgpu message
    let msg = app.export_3d_png_refusal(now).expect("no offscreen GPU");
    assert!(msg.contains("wgpu renderer"), "{msg}");
    let msg = app.export_3d_png_refusal(now + 5).expect("view not drawn for several passes");
    assert!(msg.contains("Show the 3D view"), "{msg}");
    // the 3D viewport's size in pixels is points x pixels per point (1 in this context)
    assert_eq!(app.view3d.shown_view_pixels(now), app.view3d.shown_view_size(now));
    // at 150 % the export base is 1.5 pixels per point (show stores the context's pixels_per_point)
    ctx.set_pixels_per_point(1.5);
    for _ in 0..2 {
        frame(&mut app, &ctx);
    }
    let now = ctx.cumulative_pass_nr();
    assert_eq!(ctx.pixels_per_point(), 1.5);
    let pts = app.view3d.shown_view_size(now).expect("shown at 150 %");
    assert_eq!(app.view3d.shown_view_pixels(now), Some(pts * 1.5));
    // no offscreen GPU in this process: a clear error, no panic, no file
    let path = std::env::temp_dir().join(format!("papertilling_t3_nogpu_{}.png", std::process::id()));
    let err = app.export_3d_png_to(&path).expect_err("no GPU context");
    assert!(err.contains("wgpu"), "{err}");
    assert!(!path.exists());
    eprintln!("# time allowed {:?}", t0.elapsed());
}

#[test]
fn dropping_an_export_job_waits_for_its_writer_thread() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    let t0 = std::time::Instant::now();
    let written = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&written);
    let job = Export3dJob::spawn(None, move || {
        std::thread::sleep(std::time::Duration::from_millis(200));
        flag.store(true, Ordering::SeqCst);
        String::from("done")
    })
    .expect("thread");
    // the app closing drops the job before the 200 ms write is over: the drop waits for it
    drop(job);
    assert!(written.load(Ordering::SeqCst), "drop returned before the writer finished");
    assert!(t0.elapsed() >= std::time::Duration::from_millis(200));
    // a finished job hands its status over the channel
    let job = Export3dJob::spawn(None, || String::from("ok")).expect("thread");
    assert_eq!(job.rx.recv_timeout(std::time::Duration::from_secs(5)).as_deref(), Ok("ok"));
}

#[test]
fn export_3d_status_names_size_scale_and_file() {
    let info = Export3dInfo { w: 2400, h: 1600, clamped: false };
    assert_eq!(export_3d_status(info, 2, "atlas_mesh.png"), "Exported 3D 2400x1600 PNG at 2x (atlas_mesh.png)");
    let info = Export3dInfo { w: 4096, h: 2731, clamped: true };
    assert_eq!(
        export_3d_status(info, 4, "a.png"),
        "Exported 3D 4096x2731 PNG at 4x (a.png; scaled down to fit the size limit)"
    );
}
