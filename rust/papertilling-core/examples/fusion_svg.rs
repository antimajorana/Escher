//! Write the two routes of the four-dislocation fusion as SVG pictures.
//!
//! Run: `cargo run --release --manifest-path rust/Cargo.toml -p papertilling-core --example fusion_svg -- <folder>`

use papertilling_core::fusion::svg::steps_to_svg;
use papertilling_core::fusion::{gate_route, hamiltonian_route};

fn main() -> Result<(), String> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| String::from("."));
    for (name, steps) in [("fusion_gate_route", gate_route()?), ("fusion_hamiltonian_route", hamiltonian_route()?)] {
        let path = format!("{dir}/{name}.svg");
        std::fs::write(&path, steps_to_svg(&steps, 3)).map_err(|e| format!("{path}: {e}"))?;
        println!("wrote {path}");
    }
    Ok(())
}
