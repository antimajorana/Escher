//! "Motion GIF" (retained for unit tests): the crystal moving linearly from its reference positions to its current
//! ones, rendered and encoded as a looping GIF89a (`export_motion_gif`).

use std::path::Path;

use papertilling_core::defect_fields::{export_motion_gif, interpolate_crystal_frames};

use super::detect::DetectOut;
use super::{SCENE_TITLE, Settings, Source};
use crate::canvas2d::Scene2D;
use crate::scene2d::LinePrim;
use crate::workbench::scene_helpers::{vector_scene, with_lines};

const ARROW_RGBA: [u8; 4] = [39, 174, 96, 255];

pub(crate) struct GifOut {
    pub(crate) det: DetectOut,
    pub(crate) ref_pos: Vec<[f64; 2]>,
    pub(crate) edges: Vec<(usize, usize)>,
    pub(crate) max_u: f64,
    pub(crate) bytes: Result<Vec<u8>, String>,
}

pub(crate) fn build(src: &Source, set: &Settings, det: DetectOut) -> Result<GifOut, String> {
    let has_ref = src.ref_pos.len() == src.pos.len();
    let max_u = if has_ref {
        src.pos.iter().zip(&src.ref_pos).map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1])).fold(0.0, f64::max)
    } else {
        0.0
    };
    let bytes = if !has_ref {
        Err(String::from("REFUSED: the source has no reference position for every site"))
    } else if max_u == 0.0 {
        Err(String::from("REFUSED: reference and current positions coincide: there is no motion to animate"))
    } else {
        interpolate_crystal_frames(&src.ref_pos, &src.pos, set.gif_frames)
            .and_then(|f| export_motion_gif(&f, &src.edges, set.gif_px, set.gif_px, set.gif_delay_cs))
    };
    Ok(GifOut {
        det,
        ref_pos: if has_ref { src.ref_pos.clone() } else { Vec::new() },
        edges: src.edges.clone(),
        max_u,
        bytes,
    })
}

/// Write the GIF bytes (the dialog-free core of the export).
pub(crate) fn export_to(path: &Path, bytes: &[u8]) -> Result<String, String> {
    std::fs::write(path, bytes).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(format!("wrote {} ({} bytes)", path.display(), bytes.len()))
}

/// The current crystal with its displacement arrows from the reference positions.
pub(crate) fn scene(o: &GifOut) -> Scene2D {
    let pos = &o.det.pos;
    let tiny = 1e-3 * o.det.lmax;
    let arrows: Vec<LinePrim> = o
        .ref_pos
        .iter()
        .zip(pos)
        .filter(|(q, p)| (p[0] - q[0]).hypot(p[1] - q[1]) > tiny)
        .map(|(q, p)| LinePrim {
            a: [q[0] as f32, q[1] as f32],
            b: [p[0] as f32, p[1] as f32],
            color: ARROW_RGBA,
            width: 1.5,
            dashed: false,
            arrow_mid: true,
        })
        .collect();
    let note = format!("{} displacement arrows (reference -> current), largest {:.4}", arrows.len(), o.max_u);
    with_lines(vector_scene(pos, &o.edges, &[], |_| [0, 0, 0, 0], SCENE_TITLE, &note), arrows)
}
