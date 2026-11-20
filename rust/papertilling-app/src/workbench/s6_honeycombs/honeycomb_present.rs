//! Presentation of seeded [`HoneycombGrowth`] and [`ParacompactBall`] results into S6 [`Built`] scenes and readouts.

use std::collections::HashMap;

use papertilling_core::growth::{SeedKind, Stopped};
use papertilling_core::honeycomb3d::growth_build::{DRAW_ALL_FACES_MAX, MAX_GROWTH_CELLS};
use papertilling_core::honeycomb3d::{Classification, GrowthRule, HoneycombGrowth, ParacompactBall};

use super::present::{Built, COUNTS_EVIDENCE, CoreOut, GrowthMeta, palette_rgba, projection_scene};
use crate::mesh3d::wb_mesh::{mesh_from_polygons, placeholder_mesh};
use crate::workbench::scene_helpers::unit_circle_lines;

pub(crate) fn seed_label(seed: SeedKind) -> &'static str {
    match seed {
        SeedKind::Vertex => "Vertex",
        SeedKind::Edge => "Edge",
        SeedKind::Face => "Face",
        SeedKind::Cell => "Cell",
        SeedKind::Dirichlet => "Dirichlet",
    }
}

pub(crate) fn rule_label(rule: GrowthRule) -> &'static str {
    match rule {
        GrowthRule::Touching => "Touching",
        GrowthRule::FaceShells => "Face shells",
    }
}

/// Format [`Stopped`] for the UI. `Stopped::CellCap`'s `partial` counter is an incomplete lower bound and is NEVER
/// printed as a cell count (`c1-plan.md` §2.2, HANDOVER §3 item 5).
pub(crate) fn format_stopped(stopped: &Stopped, max_cells: usize) -> String {
    match *stopped {
        Stopped::CellCap { layer, .. } => {
            format!("Stopped: layer {layer} needs more than {max_cells} cells (cap {max_cells}).")
        }
        Stopped::Closed { layer } => {
            format!("Closed at layer {layer}: S_{layer} is the whole 3-sphere polychoron (χ = 0) [PROVED].")
        }
        Stopped::Infinite { layer, reason } => format!("Stopped at layer {layer}: {reason}"),
        Stopped::Precision { layer, gap } => {
            format!("Stopped at layer {layer}: duplicate-key precision margin reached (gap {gap:.2e}).")
        }
        Stopped::Rim { layer } => format!("Stopped at layer {layer}: finite rim reached."),
    }
}

/// Model note of a regime string (`CellComplex3d::regime`).
fn model_note(regime: &str) -> &'static str {
    match regime {
        "SphericalS3" => {
            "Drawn by stereographic projection of S³ from a pole that is not a vertex: conformal, not isometric (cells near the pole look large)."
        }
        "EuclideanE3" => "Drawn in Euclidean coordinates (isometric).",
        _ => {
            "Drawn in the Poincaré ball model of H³: conformal, not isometric; ideal vertices lie on the unit sphere."
        }
    }
}

/// Present a seeded [`HoneycombGrowth`] (S³, E³, or compact H³) as 3D mesh, 2D xy projection, banners, and layer table.
pub(crate) fn present_growth(g: HoneycombGrowth, cls: Option<&Classification>, requested_layers: u32) -> Built {
    let HoneycombGrowth { seed, rule, stats, stopped, boundary_only, complex: cx, .. } = g;
    let outer_layer = stats.len().saturating_sub(1) as u32;
    let default_shell = if boundary_only { outer_layer } else { u32::MAX };
    let mut face_shell = vec![default_shell; cx.faces.len()];
    for (ci, cell) in cx.cells.iter().enumerate() {
        let s = cx.cell_shell.get(ci).copied().unwrap_or(0);
        for &f in cell {
            if let Some(fs) = face_shell.get_mut(f) {
                *fs = (*fs).min(s);
            }
        }
    }
    let shell_of: HashMap<&[usize], u32> = cx.faces.iter().zip(&face_shell).map(|(f, &s)| (f.as_slice(), s)).collect();
    let spherical = cx.regime == "SphericalS3";
    let tint = |face: &[usize]| {
        let s = shell_of.get(face).copied().unwrap_or(0);
        if boundary_only {
            palette_rgba(s as usize, 0.65)
        } else if spherical {
            palette_rgba(s as usize, if s == 0 { 0.6 } else { 0.24 })
        } else {
            let alpha = [0.85, 0.42, 0.26, 0.18][s.min(3) as usize];
            palette_rgba(s as usize, alpha)
        }
    };
    let edges: Vec<(usize, usize)> = cx.edges.iter().map(|e| (e[0], e[1])).collect();
    let mesh = if cx.faces.is_empty() {
        placeholder_mesh()
    } else {
        mesh_from_polygons(&cx.points, &cx.faces, &edges, tint)
    };
    let hyperbolic = cx.regime.starts_with("Hyperbolic");
    let extra = if hyperbolic { unit_circle_lines(96, [120, 120, 120, 255]) } else { Vec::new() };
    let title = format!("{}: xy projection of the edges", cx.name);
    let scene2d = projection_scene(&cx.points, edges.iter().copied(), extra, &title, model_note(&cx.regime));

    let (v, e, f, _) = cx.counts();
    // Always read total cell count and Euler characteristic from `stats.last()` rather than `cx.cells`
    // (`cx.cells` is intentionally empty when `boundary_only` is true above `DRAW_ALL_FACES_MAX`).
    let total_cells = stats.last().map_or(0, |s| s.cells);
    let euler = stats.last().map_or(0, |s| s.euler);
    let built_range = if stats.is_empty() {
        String::from("none")
    } else {
        format!("0..={outer_layer}")
    };

    let mut rows = vec![
        (String::from("Complex"), cx.name.clone()),
        (String::from("Regime"), cx.regime.clone()),
        (String::from("Seed / Rule"), format!("{} / {}", seed_label(seed), rule_label(rule))),
        (String::from("Layers (built / requested)"), format!("{built_range} / {requested_layers}")),
    ];
    if boundary_only {
        rows.push((String::from("Cells in S_K"), total_cells.to_string()));
        rows.push((String::from("Drawn boundary V, E, F"), format!("{v}, {e}, {f} (boundary faces only)")));
    } else {
        rows.push((String::from("V, E, F, C"), format!("{v}, {e}, {f}, {total_cells}")));
    }
    rows.push((String::from("V − E + F − C"), euler.to_string()));

    let mut banners = Vec::new();
    let mut notes = Vec::new();
    if let Some(st) = &stopped {
        let msg = format_stopped(st, MAX_GROWTH_CELLS);
        banners.push(msg.clone());
        notes.push(msg);
    }
    if boundary_only {
        let msg = format!(
            "Boundary-only drawing: S_{outer_layer} has {total_cells} cells (> {DRAW_ALL_FACES_MAX}), so only its {f} outer boundary faces are drawn; cell count ({total_cells}) and χ ({euler}) come from the exact layer tally [COMPUTATIONAL EVIDENCE]."
        );
        banners.push(msg.clone());
        notes.push(msg);
    }
    for (k, s) in stats.iter().enumerate() {
        let closed_here = matches!(stopped, Some(Stopped::Closed { layer }) if layer as usize == k);
        if closed_here && s.euler != 0 {
            let flag = format!("Layer {k} closed with χ = {} != 0 [COMPUTATIONAL EVIDENCE].", s.euler);
            banners.push(flag.clone());
            notes.push(flag);
        } else if !closed_here && s.euler != 1 {
            let flag = format!("Layer {k} is not a 3-ball (χ = {} != 1) [COMPUTATIONAL EVIDENCE].", s.euler);
            banners.push(flag.clone());
            notes.push(flag);
        }
    }

    if let Some(c) = cls {
        notes.push(c.reason.clone());
    } else {
        notes.push(String::from(
            "[PROVED: the four Euclidean honeycombs here are the SC, BCC and FCC Voronoi tilings and the octet tiling, all face-to-face tilings of E³]",
        ));
    }
    notes.push(String::from(COUNTS_EVIDENCE));
    let closed_s3 = spherical && matches!(stopped, Some(Stopped::Closed { .. }));
    notes.push(String::from(if closed_s3 {
        "Expected V − E + F − C = 0 = χ(S³) at closure of a cell decomposition of S³ [PROVED: Euler characteristic of S³]."
    } else {
        "Expected V − E + F − C = 1 for a patch that is a 3-ball [PROVED: χ(B³) = 1]; any other value before S³ closure is flagged above."
    }));
    notes.push(String::from(model_note(&cx.regime)));

    let caption = if boundary_only {
        format!("{} — {total_cells} cells ({f} boundary faces drawn)", cx.name)
    } else {
        format!("{} — {v} V, {e} E, {f} F, {total_cells} cells", cx.name)
    };
    let meta = GrowthMeta { stats, stopped, boundary_only, banners };
    Built { core: CoreOut::Growth(Box::new(cx), meta), scene2d, mesh, caption, rows, notes }
}

/// Present a [`ParacompactBall`] truncation of a paracompact H³ honeycomb.
pub(crate) fn present_paracompact_ball(b: ParacompactBall, cls: &Classification) -> Built {
    let cx = &b.complex;
    let edges: Vec<(usize, usize)> = cx.edges.iter().map(|e| (e[0], e[1])).collect();
    let mesh = if cx.faces.is_empty() {
        placeholder_mesh()
    } else {
        mesh_from_polygons(&cx.points, &cx.faces, &edges, |face| palette_rgba(face.len(), 0.52))
    };
    let extra = unit_circle_lines(96, [120, 120, 120, 255]);
    let title = format!("{} (ball R = {:.2}): xy projection of the edges", b.name, b.radius);
    let scene2d = projection_scene(&cx.points, edges.iter().copied(), extra, &title, model_note(&cx.regime));

    let (v, e, f, c) = cx.counts();
    let ideal = cx.is_ideal.iter().filter(|&&x| x).count();
    let truncated = cx.cell_truncated.iter().filter(|&&x| x).count();
    let rows = vec![
        (String::from("Complex"), b.name.clone()),
        (String::from("Regime"), cx.regime.clone()),
        (String::from("Ball radius R"), format!("{:.2} (face step D = {:.3})", b.radius, b.step)),
        (String::from("V, E, F, C"), format!("{v}, {e}, {f}, {c}")),
        (String::from("V − E + F − C"), cx.euler().to_string()),
        (String::from("Ideal vertices"), ideal.to_string()),
        (String::from("Truncated cells"), truncated.to_string()),
    ];
    let mut notes = vec![
        String::from(b.reason),
        cls.reason.clone(),
        String::from(COUNTS_EVIDENCE),
        String::from(
            "No 3-ball test here: faces are cut by hyperbolic distance R from infinite cells or ideal-vertex stars, not whole compact 3-cells, so V − E + F − C need not equal 1.",
        ),
        String::from(
            "[COMPUTATIONAL EVIDENCE: neighbour discovery explores faces within R + D on the full honeycomb, independent of any patch ring cutoff]",
        ),
        String::from(model_note(&cx.regime)),
    ];
    if b.faces == 0 {
        notes.insert(0, format!("0 faces have centres within R = {:.2}; increase R to include faces.", b.radius));
    }
    let caption = format!("{} (R = {:.2}) — {v} V, {e} E, {f} F, {c} cells", b.name, b.radius);
    Built { core: CoreOut::Ball(Box::new(b)), scene2d, mesh, caption, rows, notes }
}
