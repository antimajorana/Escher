//! Symmetry impossibility proofs, retry schedules, and the 8-row `surface_table` runner.

use std::collections::BTreeMap;

use super::pieces::check_surface_mesh;
use super::polyhedra::build_polyhedron;
use super::{ConeDefect, RefusalKind, SurfaceError, SurfaceVerdict};
use crate::types::{Cell2D, SurfaceMesh};

/// Canonical 8 target surfaces evaluated by `surface_table` (matching `report.py`).
pub const SURFACES: [&str; 8] = [
    "torus", "cylinder", "Moebius band", "Klein bottle", "sphere", "projective plane", "genus 2", "genus 3",
];

/// Symmetry-group properties governing proved `IMPOSSIBLE` surface verdicts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymmetryFacts {
    pub group: String,
    pub rotation_orders: Vec<u32>,
    pub reversing: bool,
}

impl SymmetryFacts {
    pub fn max_rotation_order(&self) -> u32 {
        self.rotation_orders.iter().copied().max().unwrap_or(1)
    }

    pub fn impossible_reason(&self, surface: &str) -> Option<String> {
        let mut why = Vec::new();
        let non_orientable = matches!(surface, "Moebius band" | "Mobius band" | "Klein bottle" | "projective plane");
        if non_orientable && !self.reversing {
            why.push(format!(
                "{} has no mirror or glide line, and a {surface} must be glued with a reflection or a glide: orientation-reversing gluings are not symmetries of this lattice",
                self.group
            ));
        }
        if matches!(surface, "sphere" | "projective plane") && self.rotation_orders.is_empty() {
            let chi = if surface == "projective plane" { 1 } else { 2 };
            why.push(format!(
                "{} has no rotation, so no disclination with a positive Frank angle is allowed, and a {surface} needs Frank angles adding up to 360 x chi = {} deg (Gauss-Bonnet)",
                self.group,
                360 * chi
            ));
        }
        (!why.is_empty()).then(|| why.join("; and "))
    }
}

/// Wallpaper-group symmetry facts (`IMPOSSIBLE` iff chiral for non-orientable or rotationless for sphere/RP2).
pub fn symmetry_facts(group: &str) -> SymmetryFacts {
    let rotation_orders = match group {
        "p1" | "pm" | "pg" | "cm" => vec![],
        "p2" | "pmm" | "pmg" | "pgg" | "cmm" => vec![2],
        "p3" | "p3m1" | "p31m" => vec![3],
        "p4" | "p4m" | "p4g" => vec![2, 4],
        "p6" | "p6m" => vec![2, 3, 6],
        _ => vec![],
    };
    let reversing = !matches!(group, "p1" | "p2" | "p3" | "p4" | "p6");
    SymmetryFacts { group: group.to_string(), rotation_orders, reversing }
}

/// Cell-aware symmetry facts: identifies the realized wallpaper group of `cell`
/// (falling back to `fallback_group` on failure) so special-locus mirror/glide
/// symmetries on Sohncke recipes match `refusal.py`.
pub fn symmetry_facts_of_cell(fallback_group: &str, cell: &Cell2D) -> SymmetryFacts {
    let real_group = cell
        .try_identify_wallpaper_group(false)
        .unwrap_or_else(|_| fallback_group.to_string());
    symmetry_facts(&real_group)
}

/// Settings attempted for flat surfaces (`torus`, `cylinder`, `Moebius band`, `Klein bottle`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatAttempt {
    Default,
    WhichGlide(usize),
    TargetSites(usize),
}

/// One row returned by `surface_table_with`.
#[derive(Debug, Clone)]
pub struct SurfaceTableRow {
    pub surface: String,
    pub status: String,
    pub reason: String,
    pub scope: Option<String>,
    pub mesh: Option<SurfaceMesh>,
    pub sites: Option<usize>,
    pub cones: String,
    pub tried: Vec<String>,
    pub hint: Option<String>,
}

impl SurfaceTableRow {
    fn ok(surface: &str, reason: String, mesh: SurfaceMesh, tried: Vec<String>) -> Self {
        let (sites, cones) = (mesh.sites.len(), cone_summary(&mesh));
        Self { surface: surface.into(), status: "OK".into(), reason, scope: None, mesh: Some(mesh), sites: Some(sites), cones, tried, hint: None }
    }

    fn impossible(surface: &str, reason: String, scope: &str, tried: Vec<String>) -> Self {
        Self { surface: surface.into(), status: "IMPOSSIBLE".into(), reason, scope: Some(scope.into()), mesh: None, sites: None, cones: "-".into(), tried, hint: None }
    }

    fn not_found(surface: &str, reason: String, tried: Vec<String>, hint: Option<String>) -> Self {
        Self { surface: surface.into(), status: "NOT FOUND".into(), reason, scope: None, mesh: None, sites: None, cones: "-".into(), tried, hint }
    }

    pub fn to_verdict(&self) -> SurfaceVerdict {
        match self.status.as_str() {
            "OK" => {
                let m = self.mesh.as_ref().unwrap();
                let mut grouped: BTreeMap<(i32, String), usize> = BTreeMap::new();
                for c in &m.cones {
                    *grouped.entry((c.angle_deg.round() as i32, c.class_label.clone().unwrap_or_default())).or_insert(0) += 1;
                }
                let cones = grouped
                    .into_iter()
                    .map(|((frank_angle_deg, centre_class), multiplicity)| ConeDefect { frank_angle_deg, centre_class, multiplicity })
                    .collect();
                SurfaceVerdict::Ok { vertices: m.sites.len(), edges: m.bonds.len(), faces: m.faces.len(), orientable: m.orientable, cones }
            }
            "IMPOSSIBLE" => SurfaceVerdict::Impossible { reason: self.reason.clone(), scope: self.scope.clone().unwrap_or_else(|| "group".into()) },
            _ => SurfaceVerdict::NotFound { reason: self.reason.clone(), hint: self.hint.clone() },
        }
    }
}

fn pillow_schedule() -> Vec<(i32, i32)> {
    let mut zs: Vec<(i32, i32)> = (1..=8).flat_map(|h| (1..=8).map(move |k| (h, k))).collect();
    zs.sort_by_key(|&(h, k)| (h * k, h, k));
    zs
}

pub fn polyhedron_attempts(surface: &str, max_rot_order: u32) -> Vec<(i32, i32)> {
    if max_rot_order == 2 {
        return pillow_schedule();
    }
    if surface == "sphere" {
        vec![(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1)]
    } else {
        vec![(1, 1), (1, 0), (2, 0), (2, 1), (2, 2), (3, 0)]
    }
}

pub fn flat_attempts(surface: &str, glide_classes: usize) -> Vec<FlatAttempt> {
    if matches!(surface, "Moebius band" | "Mobius band" | "Klein bottle") {
        let mut out = vec![FlatAttempt::Default];
        for w in 1..glide_classes.max(1) {
            out.push(FlatAttempt::WhichGlide(w));
        }
        out.push(FlatAttempt::TargetSites(400));
        out
    } else {
        vec![FlatAttempt::Default, FlatAttempt::TargetSites(400)]
    }
}

pub fn genus_attempts() -> [(i32, i32); 3] {
    [(6, 6), (8, 8), (10, 10)]
}

fn cone_summary(mesh: &SurfaceMesh) -> String {
    let mut counts: BTreeMap<i64, usize> = BTreeMap::new();
    for c in &mesh.cones {
        *counts.entry((c.angle_deg * 1000.0).round() as i64).or_insert(0) += 1;
    }
    if counts.is_empty() {
        return "none".into();
    }
    counts
        .into_iter()
        .rev()
        .map(|(milli, k)| {
            let deg = milli as f64 / 1000.0;
            if (deg - deg.round()).abs() < 1e-6 {
                format!("{k} x {:+.0}", deg.round())
            } else {
                format!("{k} x {deg:+}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn run_poly_table(group: &str, cell: &Cell2D, projective: bool) -> SurfaceTableRow {
    try_poly_for_table_with_centre(group, cell, projective, None)
}

/// The `sphere` (`projective == false`) or `projective plane` row of `surface_table` with its retry schedule, with
/// the corners restricted to the centre classes matching `centre` (`None`: every usable class, as in the table).
pub fn try_poly_for_table_with_centre(group: &str, cell: &Cell2D, projective: bool, centre: Option<&str>) -> SurfaceTableRow {
    let name = if projective { "projective plane" } else { "sphere" };
    // C1: facts and group from the cell (the label can miss the symmetry of a tuned special locus).
    let facts = symmetry_facts_of_cell(group, cell);
    let group = facts.group.as_str();
    if let Some(reason) = facts.impossible_reason(name) {
        return SurfaceTableRow::impossible(name, reason, "group", vec!["default".into()]);
    }
    let (mut tried, mut whys, mut last_err): (Vec<String>, Vec<String>, Option<SurfaceError>) = (Vec::new(), Vec::new(), None);
    for sz in polyhedron_attempts(name, facts.max_rotation_order()) {
        let kw = format!("size={sz:?}");
        match build_polyhedron(group, cell, Some(sz), centre, projective) {
            Ok(surf) => {
                tried.push(kw.clone());
                if surf.checks().ok {
                    return SurfaceTableRow::ok(name, format!("{}: size {sz:?}", surf.mesh.name), surf.mesh, tried);
                }
                whys.push(format!("built at {kw} but checks failed"));
            }
            Err(e) => {
                tried.push(kw);
                if matches!(e.kind, RefusalKind::Impossible) {
                    return SurfaceTableRow::impossible(name, e.reason.clone(), e.scope.unwrap_or("group"), tried);
                }
                let retry = matches!(e.kind, RefusalKind::NotFound { retry: true });
                if !whys.contains(&e.reason) {
                    whys.push(e.reason.clone());
                }
                last_err = Some(e);
                if !retry {
                    break;
                }
            }
        }
    }
    let hint = last_err.as_ref().and_then(|e| (tried.len() == 1).then(|| e.hint.clone()).flatten());
    let reason = if let Some(e) = &last_err && matches!(e.kind, RefusalKind::NotFound { retry: false }) {
        e.to_string()
    } else {
        format!("refused at every setting tried ({} tried): {}", tried.len(), whys.join("; "))
    };
    SurfaceTableRow::not_found(name, reason, tried, hint)
}

/// Evaluate the `sphere` row of `surface_table` with the full retry schedule.
pub fn try_sphere_for_table(group: &str, cell: &Cell2D) -> SurfaceTableRow {
    run_poly_table(group, cell, false)
}

/// Evaluate the `projective plane` row of `surface_table` with the full retry schedule.
pub fn try_rp2_for_table(group: &str, cell: &Cell2D) -> SurfaceTableRow {
    run_poly_table(group, cell, true)
}

fn run_hook_row<A: std::fmt::Debug>(
    surface: &str,
    facts: &SymmetryFacts,
    attempts: &[A],
    mut f: impl FnMut(&A) -> Result<SurfaceMesh, SurfaceError>,
) -> SurfaceTableRow {
    if let Some(reason) = facts.impossible_reason(surface) {
        return SurfaceTableRow::impossible(surface, reason, "group", vec!["default".into()]);
    }
    let (mut tried, mut whys, mut last_hint) = (Vec::new(), Vec::new(), None);
    for att in attempts {
        tried.push(format!("{att:?}"));
        match f(att) {
            Ok(mesh) => {
                if check_surface_mesh(&mesh).ok {
                    return SurfaceTableRow::ok(surface, mesh.name.clone(), mesh, tried);
                }
                whys.push("checks failed".into());
            }
            Err(e) => {
                if matches!(e.kind, RefusalKind::Impossible) {
                    return SurfaceTableRow::impossible(surface, e.reason.clone(), e.scope.unwrap_or("group"), tried);
                }
                last_hint = e.hint.clone();
                let retry = matches!(e.kind, RefusalKind::NotFound { retry: true });
                if !whys.contains(&e.reason) {
                    whys.push(e.reason.clone());
                }
                if !retry {
                    break;
                }
            }
        }
    }
    let hint = if tried.len() == 1 { last_hint } else { None };
    SurfaceTableRow::not_found(surface, whys.join("; "), tried, hint)
}

/// Evaluate all 8 `SURFACES` rows using `try_sphere_for_table`/`try_rp2_for_table` plus hooks for flat & genus.
pub fn surface_table_with<FFlat, FGenus>(
    group: &str,
    cell: &Cell2D,
    glide_classes: usize,
    mut build_flat: FFlat,
    mut build_genus: FGenus,
) -> Vec<SurfaceTableRow>
where
    FFlat: FnMut(&str, FlatAttempt) -> Result<SurfaceMesh, SurfaceError>,
    FGenus: FnMut(u32, (i32, i32)) -> Result<SurfaceMesh, SurfaceError>,
{
    // C1: the verdicts come from the symmetry found on the cell, not from the label (a tuned special locus such as
    // an oblique cell at b = 1, gamma = 90 deg has mirrors its label p2 does not list).
    let facts = symmetry_facts_of_cell(group, cell);
    let group = facts.group.as_str();
    SURFACES
        .into_iter()
        .map(|name| match name {
            "sphere" => try_sphere_for_table(group, cell),
            "projective plane" => try_rp2_for_table(group, cell),
            "genus 2" => run_hook_row(name, &facts, &genus_attempts(), |&sz| build_genus(2, sz)),
            "genus 3" => run_hook_row(name, &facts, &genus_attempts(), |&sz| build_genus(3, sz)),
            _ => {
                let atts = flat_attempts(name, glide_classes);
                run_hook_row(name, &facts, &atts, |&att| build_flat(name, att))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Instant;

    use serde::Deserialize;

    use super::*;
    use crate::surface::polyhedra::{candidate_centres, projective_plane, sphere};
    use crate::types::Bond2D;

    fn square_cell() -> Cell2D {
        Cell2D {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            basis: vec![[0.0, 0.0]],
            bonds: vec![Bond2D { i: 0, j: 0, n1: -1, n2: 0, kind: 0 }, Bond2D { i: 0, j: 0, n1: 0, n2: -1, kind: 0 }],
        }
    }

    #[test]
    fn square_default_octahedron_vs_site_cube_and_symmetry_controls() {
        let sq = square_cell();
        let cands = candidate_centres("p4m", &sq);
        assert_eq!(cands[0].order, 4);
        assert_eq!(cands[0].location, "face centre");
        assert_eq!(cands[1].location, "site");

        let oct = sphere("p4m", &sq, None, None).unwrap();
        assert_eq!((oct.sites.len(), oct.bonds.len(), oct.faces.len(), oct.chi), (6, 12, 8, 2));
        assert!(check_surface_mesh(&oct).ok);

        let cube = sphere("p4m", &sq, Some((1, 0)), Some("site")).unwrap();
        assert_eq!((cube.sites.len(), cube.bonds.len(), cube.faces.len(), cube.chi), (8, 12, 6, 2));
        assert!(check_surface_mesh(&cube).ok);

        let rp2 = projective_plane("p4m", &sq, Some((1, 0)), Some("site")).unwrap();
        assert_eq!((rp2.sites.len(), rp2.bonds.len(), rp2.faces.len(), rp2.chi, rp2.orientable), (4, 6, 3, 1, false));
        let cov = rp2.cover.as_ref().unwrap();
        assert_eq!((cov.xyz.len(), cov.bonds.len(), cov.faces.len()), (8, 12, 6));
        assert_eq!(cov.site_rgba[0], cov.site_rgba[4]);

        for g in ["p1", "pm", "pg", "cm"] {
            assert!(symmetry_facts(g).impossible_reason("sphere").is_some());
            assert!(symmetry_facts(g).impossible_reason("projective plane").is_some());
        }
        for g in ["p1", "p2", "p3", "p4", "p6"] {
            assert!(symmetry_facts(g).impossible_reason("Moebius band").is_some());
            assert!(symmetry_facts(g).impossible_reason("Mobius band").is_some());
            assert!(symmetry_facts(g).impossible_reason("Klein bottle").is_some());
            assert!(symmetry_facts(g).impossible_reason("projective plane").is_some());
        }
        let rows = surface_table_with(
            "p4",
            &sq,
            0,
            |_, _| Err(SurfaceError::not_found("stub", "", false)),
            |_, _| Err(SurfaceError::not_found("stub", "", false)),
        );
        // C1: the verdicts follow the cell, not the label. A square cell labelled p4 has the mirrors of p4m, so the
        // Moebius row is not proved impossible (the stub builder refuses it: NOT FOUND) and neither is RP2.
        assert_eq!(rows.len(), 8);
        assert_eq!(rows[2].status, "NOT FOUND");
        assert_eq!(rows[4].status, "OK");
        assert_ne!(rows[5].status, "IMPOSSIBLE");
    }

    #[derive(Default, Deserialize)]
    #[serde(default)]
    struct GateSummary {
        status: Option<String>,
        #[serde(rename = "V")]
        v: usize,
        #[serde(rename = "E")]
        e: usize,
        #[serde(rename = "F")]
        f: usize,
        chi: i32,
        orientable: bool,
        closed: bool,
        face_hist: HashMap<String, usize>,
        cone_count: usize,
        cone_angles: Vec<f64>,
        checks_ok: bool,
    }
    #[derive(Deserialize)]
    struct GateInput {
        cell: Cell2D,
    }
    #[derive(Deserialize)]
    struct GateFile {
        known: HashMap<String, GateSummary>,
        inputs_17: HashMap<String, GateInput>,
        table_17x8: HashMap<String, HashMap<String, GateSummary>>,
    }
    #[derive(Deserialize)]
    struct NamedEntry {
        key: String,
        group: String,
        cell: Cell2D,
    }
    #[derive(Deserialize)]
    struct NamedFile {
        named: Vec<NamedEntry>,
    }

    fn assert_matches_gate(label: &str, mesh: &SurfaceMesh, exp: &GateSummary) {
        let mut hist: HashMap<String, usize> = HashMap::new();
        for f in &mesh.faces {
            *hist.entry(f.len().to_string()).or_insert(0) += 1;
        }
        let mut angles: Vec<f64> = mesh.cones.iter().map(|c| (c.angle_deg * 1e6).round() / 1e6).collect();
        angles.sort_by(|a, b| a.total_cmp(b));
        let ch = check_surface_mesh(mesh);
        assert_eq!(mesh.sites.len(), exp.v, "{label}: V");
        assert_eq!(mesh.bonds.len(), exp.e, "{label}: E");
        assert_eq!(mesh.faces.len(), exp.f, "{label}: F");
        assert_eq!(mesh.chi, exp.chi, "{label}: chi");
        assert_eq!(mesh.orientable, exp.orientable, "{label}: orientable");
        assert_eq!(mesh.closed, exp.closed, "{label}: closed");
        assert_eq!(hist, exp.face_hist, "{label}: face_hist");
        assert_eq!(mesh.cones.len(), exp.cone_count, "{label}: cone_count");
        assert_eq!(angles, exp.cone_angles, "{label}: cone_angles");
        assert_eq!(ch.ok, exp.checks_ok, "{label}: checks_ok");
        assert_eq!(mesh.xyz.len(), exp.v, "{label}: xyz len");
        for p in &mesh.xyz {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 1.0).abs() < 1e-6, "{label}: xyz on unit sphere");
        }
        if !mesh.orientable {
            let cov = mesh.cover.as_ref().expect("RP2 double cover");
            assert_eq!(cov.xyz.len(), 2 * exp.v, "{label}: cover V");
            for i in 0..exp.v {
                assert_eq!(cov.site_rgba[i], cov.site_rgba[i + exp.v], "{label}: antipodal colour");
            }
        }
    }

    #[test]
    fn stage3b_gate_known_16_and_17x2_table() {
        let (Ok(gate_path), Ok(named_path)) = (std::env::var("PT_GATE_STAGE3"), std::env::var("PT_NAMED_JSON")) else {
            return;
        };
        let t0 = Instant::now();
        let gate: GateFile = serde_json::from_str(&std::fs::read_to_string(gate_path).unwrap()).unwrap();
        let named_file: NamedFile = serde_json::from_str(&std::fs::read_to_string(named_path).unwrap()).unwrap();
        let named: HashMap<String, NamedEntry> = named_file.named.into_iter().map(|e| (e.key.clone(), e)).collect();

        type GateCase<'a> = (&'a str, &'a str, bool, Option<(i32, i32)>, Option<&'a str>);
        let cases: [GateCase<'_>; 16] = [
            ("honeycomb_sphere_1_0", "honeycomb", false, Some((1, 0)), None),
            ("honeycomb_sphere_1_1_c60", "honeycomb", false, Some((1, 1)), None),
            ("honeycomb_sphere_2_1", "honeycomb", false, Some((2, 1)), None),
            ("triangular_sphere_1_0", "triangular", false, Some((1, 0)), None),
            ("triangular_sphere_1_1", "triangular", false, Some((1, 1)), None),
            ("square_sphere_default", "square", false, Some((1, 0)), None),
            ("square_sphere_site_cube", "square", false, Some((1, 0)), Some("site")),
            ("square_sphere_site_2_0", "square", false, Some((2, 0)), Some("site")),
            ("kagome_sphere_1_0", "kagome", false, Some((1, 0)), None),
            ("oblique_pillow_1_1", "oblique", false, Some((1, 1)), Some("site")),
            ("oblique_pillow_2_2", "oblique", false, Some((2, 2)), Some("site")),
            ("chiral_triangular_sphere_1_0", "chiral_triangular", false, Some((1, 0)), None),
            ("honeycomb_rp2_1_0_petersen", "honeycomb", true, Some((1, 0)), None),
            ("honeycomb_rp2_1_1", "honeycomb", true, Some((1, 1)), None),
            ("square_rp2_site_1_0", "square", true, Some((1, 0)), Some("site")),
            ("rectangular_rp2_6_6", "rectangular", true, Some((6, 6)), Some("site")),
        ];
        for (label, key, proj, sz, cf) in cases {
            let ent = &named[key];
            let mesh = if proj {
                projective_plane(&ent.group, &ent.cell, sz, cf).unwrap_or_else(|e| panic!("{label}: {e}"))
            } else {
                sphere(&ent.group, &ent.cell, sz, cf).unwrap_or_else(|e| panic!("{label}: {e}"))
            };
            assert_matches_gate(label, &mesh, &gate.known[label]);
        }
        println!("# time 16 known sphere/RP2 controls: {:.4}s", t0.elapsed().as_secs_f64());

        let t1 = Instant::now();
        assert_eq!(gate.table_17x8.len(), 17);
        for (grp, surf_map) in &gate.table_17x8 {
            let cell = &gate.inputs_17[grp].cell;
            for (sname, row) in [("sphere", try_sphere_for_table(grp, cell)), ("projective plane", try_rp2_for_table(grp, cell))] {
                let exp = &surf_map[sname];
                assert_eq!(&row.status, exp.status.as_ref().unwrap(), "{grp} {sname}: status ({})", row.reason);
                if row.status == "OK" {
                    assert_matches_gate(&format!("{grp}:{sname}"), row.mesh.as_ref().unwrap(), exp);
                }
            }
        }
        println!("# time 17x2 sphere/RP2 table rows: {:.4}s", t1.elapsed().as_secs_f64());
    }
}
