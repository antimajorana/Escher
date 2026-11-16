#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod aliases;
mod app;
mod backend;
mod canvas2d;
#[cfg(test)]
mod combine_bench;
mod combine_tab;
mod combine_ui;
mod combine_worker;
mod core_panel;
mod defects_tab;
mod defects_ui;
mod export;
mod gallery_tab;
mod live_combine;
mod live_ops;
mod live_plane;
mod mesh3d;
mod scene2d;
mod scene_crystal;
mod surfaces_tab;
mod view3d;
#[cfg(test)]
mod features_gate_tests;

use std::io::Cursor;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use app::PapertillingApp;

fn load_app_icon() -> Arc<egui::IconData> {
    const ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");
    let mut decoder = png::Decoder::new(Cursor::new(ICON_PNG));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::ALPHA);
    let mut reader = decoder.read_info().expect("valid embedded icon.png header");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("icon buffer size")];
    let info = reader.next_frame(&mut buf).expect("valid embedded icon.png frame");
    buf.truncate(info.buffer_size());
    Arc::new(egui::IconData {
        rgba: buf,
        width: info.width,
        height: info.height,
    })
}

fn main() -> ExitCode {
    let mut smoke_test = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--smoke-test" => smoke_test = true,
            "--help" | "-h" => {
                println!("Usage: papertilling-app [--smoke-test]");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("papertilling — Desktop Explorer")
            .with_inner_size([1360.0, 840.0])
            .with_icon(load_app_icon()),
        depth_buffer: view3d::DEPTH_BITS,
        ..Default::default()
    };

    let smoke_outcome = Arc::new(AtomicU8::new(0));
    let smoke_for_app = Arc::clone(&smoke_outcome);

    let result = eframe::run_native(
        "papertilling",
        options,
        Box::new(move |cc| {
            let (adapter_name, backend_name) = if let Some(rs) = &cc.wgpu_render_state {
                view3d::init_view3d_resources(rs);
                let info = rs.adapter.get_info();
                (info.name, format!("{:?}", info.backend))
            } else {
                (String::from("none"), String::from("none"))
            };
            Ok(Box::new(
                PapertillingApp::new(smoke_test, adapter_name, backend_name)
                    .with_smoke_outcome(smoke_for_app),
            ))
        }),
    );

    match result {
        Ok(()) if !smoke_test || smoke_outcome.load(Ordering::Acquire) == 1 => ExitCode::SUCCESS,
        Ok(()) => {
            eprintln!("papertilling-app error: smoke test failed before finishing all stages");
            ExitCode::FAILURE
        }
        Err(err) => {
            eprintln!("papertilling-app error: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::load_app_icon;

    #[test]
    fn embedded_app_icon_decodes_256x256_rgba_with_transparent_corners() {
        let icon = load_app_icon();
        assert_eq!(icon.width, 256);
        assert_eq!(icon.height, 256);
        assert_eq!(icon.rgba.len(), 256 * 256 * 4);
        assert_eq!(icon.rgba[3], 0, "top-left corner outside squircle must be transparent");
        let mid = (128 * 256 + 128) * 4;
        assert_eq!(icon.rgba[mid + 3], 255, "icon centre must be opaque");
    }
}
