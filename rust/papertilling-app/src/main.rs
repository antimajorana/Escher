mod app;
mod canvas2d;
mod export;
mod mesh3d;
mod view3d;

use std::process::ExitCode;

use app::PapertillingApp;

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
            .with_title("papertilling — Stage 0")
            .with_inner_size([1280.0, 800.0]),
        depth_buffer: view3d::DEPTH_BITS,
        ..Default::default()
    };

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
            Ok(Box::new(PapertillingApp::new(
                smoke_test,
                adapter_name,
                backend_name,
            )))
        }),
    );

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("papertilling-app error: {err}");
            ExitCode::FAILURE
        }
    }
}
