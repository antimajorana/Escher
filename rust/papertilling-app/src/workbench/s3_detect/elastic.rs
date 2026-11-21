//! "Elastic & Stroh" (retained for unit tests): homogenised GSV moduli of the cell (Cauchy-Born with Born-Huang
//! relaxation), their symmetry class, the Barnett-Lothe tensor, and the anisotropic Stroh field of a straight
//! dislocation sampled on a grid round the selected defect (or a manual Burgers vector).

use papertilling_core::defect_fields::gsv_stroh::Tensor4;
use papertilling_core::defect_fields::{ElasticClass, barnett_lothe_tensor, cauchy_born_tensor, classify_elastic, stroh_dislocation_field};

use super::detect::{DetectOut, fmt_v, resolve};
use super::fields::selected_burgers;
use super::{SCENE_TITLE, Settings, Source};
use crate::canvas2d::Scene2D;
use crate::scene2d::PolyPrim;
use crate::workbench::scene_helpers::{heat_rgba, vector_scene, with_lines};

/// Angular quadrature nodes of the Barnett-Lothe tensor.
pub(crate) const N_QUAD: usize = 128;

/// `|sigma|` (Frobenius) of the Stroh field at the centres of a `n x n` grid of square cells.
pub(crate) struct StrohGrid {
    pub(crate) centre: [f64; 2],
    pub(crate) half: f64,
    pub(crate) n: usize,
    pub(crate) b: [f64; 2],
    pub(crate) sigma: Vec<f64>,
}

pub(crate) struct ElasticOut {
    pub(crate) det: DetectOut,
    /// Index of the defect the field is centred on (resolved from the settings).
    pub(crate) selected: usize,
    pub(crate) c: Tensor4,
    pub(crate) class: ElasticClass,
    pub(crate) bl: [[f64; 2]; 2],
    pub(crate) field: Result<StrohGrid, String>,
}

/// FS/RH Burgers vector fed to the Stroh field (whose displacement jumps by `+b` counter-clockwise) and its centre.
fn burgers_for_field(src: &Source, set: &Settings, det: &DetectOut, sel: usize) -> Result<([f64; 2], [f64; 2]), String> {
    let defect = det.det.defects.get(sel);
    if set.manual_b {
        let [u, v] = set.manual_uv.map(f64::from);
        let (a1, a2) = (src.cell.a1, src.cell.a2);
        let b = [u * a1[0] + v * a2[0], u * a1[1] + v * a2[1]];
        let n = src.pos.len() as f64;
        let centroid = [src.pos.iter().map(|p| p[0]).sum::<f64>() / n, src.pos.iter().map(|p| p[1]).sum::<f64>() / n];
        let c = defect.map_or(centroid, |d| d.centre);
        return Ok((b, c));
    }
    let b = selected_burgers(det, sel)?;
    let d = &det.det.defects[sel];
    Ok(([-b[0] + 0.0, -b[1] + 0.0], d.centre))
}

pub(crate) fn build(src: &Source, set: &Settings, det: DetectOut) -> Result<ElasticOut, String> {
    let c = cauchy_born_tensor(&src.cell);
    if c.iter().flatten().flatten().flatten().any(|x| !x.is_finite()) {
        return Err(String::from("REFUSED: the bond-model moduli of this cell are not finite"));
    }
    let class = classify_elastic(&c);
    let bl = barnett_lothe_tensor(&c, N_QUAD);
    let selected = resolve(&det.det, set.selected);
    let field = burgers_for_field(src, set, &det, selected).and_then(|(b, centre)| {
        let half = set.extent * det.lmax;
        let n = set.grid;
        let h = 2.0 * half / n as f64;
        // cell centres: with n even no centre sits on the line (the field is singular there)
        let rel: Vec<[f64; 2]> =
            (0..n * n).map(|k| [-half + (k % n) as f64 * h + 0.5 * h, -half + (k / n) as f64 * h + 0.5 * h]).collect();
        stroh_dislocation_field(&c, b, &rel).map(|f| StrohGrid {
            centre,
            half,
            n,
            b,
            sigma: f.stress.iter().map(|s| s.iter().flatten().map(|x| x * x).sum::<f64>().sqrt()).collect(),
        })
    });
    Ok(ElasticOut { det, selected, c, class, bl, field })
}

/// Stress magnitude cells (log colour scale, clamped to the 2-98 % range) under the crystal and the defect loops.
pub(crate) fn scene(o: &ElasticOut) -> Scene2D {
    let pos = &o.det.pos;
    let loops = super::detect::loop_lines(&o.det, o.selected);
    let Ok(g) = &o.field else {
        let why = o.field.as_ref().err().cloned().unwrap_or_default();
        return with_lines(vector_scene(pos, &o.det.det.edges, &[], |_| [0, 0, 0, 0], SCENE_TITLE, &why), loops);
    };
    let logs: Vec<f64> = g.sigma.iter().map(|s| s.max(1e-300).log10()).collect();
    let mut sorted: Vec<f64> = logs.iter().copied().filter(|x| x.is_finite()).collect();
    sorted.sort_by(f64::total_cmp);
    let q = |t: f64| sorted.get(((sorted.len().saturating_sub(1)) as f64 * t).round() as usize).copied().unwrap_or(0.0);
    let (lo, hi) = (q(0.02), q(0.98));
    let h = 2.0 * g.half / g.n as f64;
    let cells: Vec<PolyPrim> = (0..g.n * g.n)
        .map(|k| {
            let x0 = g.centre[0] - g.half + (k % g.n) as f64 * h;
            let y0 = g.centre[1] - g.half + (k / g.n) as f64 * h;
            let pts = [[x0, y0], [x0 + h, y0], [x0 + h, y0 + h], [x0, y0 + h]];
            PolyPrim { points: pts.iter().map(|p| [p[0] as f32, p[1] as f32]).collect(), fill: heat_rgba(logs[k], lo, hi), stroke: None }
        })
        .collect();
    let note = format!("log10 |σ| of the Stroh field (b = {}), colour clamped to its 2-98 % range [{lo:.2}, {hi:.2}]", fmt_v(g.b));
    let mut s = vector_scene(pos, &o.det.det.edges, &[], |_| [0, 0, 0, 0], SCENE_TITLE, &note);
    if let Scene2D::VectorScene { primitives } = &mut s {
        primitives.polygons = cells;
        let (a, b) = ([g.centre[0] - g.half, g.centre[1] - g.half], [g.centre[0] + g.half, g.centre[1] + g.half]);
        primitives.bbox = ([a[0] as f32, a[1] as f32], [b[0] as f32, b[1] as f32]);
    }
    with_lines(s, loops)
}
