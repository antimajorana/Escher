//! Stage 4b verification gate: point-defect surgery, VF2 core isomorphism, planarity, and drawing.

use std::collections::HashMap;
use std::time::Instant;

use petgraph::graph::UnGraph;
use serde::Deserialize;

use papertilling_core::defect::core_graph::{
    LatticeCoreInfo, PerfectLattice, core_graph, core_graph_on_patch, isomorphic_cores,
    total_charge,
};
use papertilling_core::defect::surgery::{
    DefectSpec, build_point_defects, patch_centre_and_radius,
};
use papertilling_core::plane::draw::{draw_wall_picture, ensure_embedded, relax_picture};
use papertilling_core::plane::planarity::check;
use papertilling_core::types::{Bond2D, BuiltCrystal, Cell2D, DirectedBondData, Isometry2};

#[derive(Deserialize)]
struct GateFile {
    sweep: HashMap<String, SweepEntry>,
    builds: Vec<BuildGateEntry>,
}

#[derive(Deserialize)]
struct SweepEntry {
    cell: RawCellJson,
    #[serde(rename = "R0")]
    r0: f64,
}

#[derive(Deserialize)]
struct RawCellJson {
    a1: [f64; 2],
    a2: [f64; 2],
    basis: Vec<[f64; 2]>,
    bonds: Vec<(usize, usize, i32, i32, usize)>,
}

impl RawCellJson {
    fn to_cell2d(&self) -> Cell2D {
        Cell2D {
            a1: self.a1,
            a2: self.a2,
            basis: self.basis.clone(),
            bonds: self
                .bonds
                .iter()
                .map(|&(i, j, n1, n2, kind)| Bond2D { i, j, n1, n2, kind })
                .collect(),
        }
    }
}

#[derive(Deserialize)]
struct BuildGateEntry {
    name: String,
    #[serde(rename = "R")]
    r_patch: f64,
    sites: usize,
    edges: usize,
    frank_deg: f64,
    charge_kind: String,
    core_reaches_rim: bool,
    #[serde(default)]
    core_nodes: usize,
    #[serde(default)]
    core_edges: usize,
    #[serde(default)]
    core_degs: Vec<u16>,
    #[serde(default)]
    core_node_degs: Vec<u16>,
    #[serde(default)]
    core_edge_pairs: Vec<[usize; 2]>,
}

fn gate_specs_for(name: &str) -> (&'static str, Vec<DefectSpec>, Option<f64>) {
    match name {
        "square_disloc_1_0" => (
            "square",
            vec![DefectSpec::dislocation([1.0, 0.0], Some([0.371, 0.413]))],
            None,
        ),
        "square_discl_p90_site" => ("square", vec![DefectSpec::disclination(90.0, [0.0, 0.0])], None),
        "square_discl_m90_face" => ("square", vec![DefectSpec::disclination(-90.0, [0.5, 0.5])], None),
        "square_pair_p90_m90" => (
            "square",
            vec![
                DefectSpec::disclination(90.0, [0.0, 0.0]),
                DefectSpec::disclination(-90.0, [0.5, 0.5]),
            ],
            None,
        ),
        "chiral_square_p90_p90" => (
            "chiral_square",
            vec![
                DefectSpec::disclination(90.0, [0.0, 0.0]),
                DefectSpec::disclination(90.0, [1.0, 0.0]),
            ],
            None,
        ),
        "honeycomb_p180_m180_3c" => (
            "honeycomb",
            vec![
                DefectSpec::disclination(180.0, [-1.453727223886262e-16, 0.49999999999999983]),
                DefectSpec::disclination(-180.0, [1.732050807568877, 0.49999999999999983]),
            ],
            None,
        ),
        "zigzag_p180_disloc_2_0" => (
            "zigzag",
            vec![
                DefectSpec::disclination(180.0, [0.0, 0.5]),
                DefectSpec::dislocation([2.0, 0.0], Some([-0.826, 3.64])),
            ],
            None,
        ),
        "chiral_square_p90_p90_2R0" => (
            "chiral_square",
            vec![
                DefectSpec::disclination(90.0, [0.0, 0.0]),
                DefectSpec::disclination(90.0, [1.0, 0.0]),
            ],
            Some(13.051163197408322),
        ),
        "square_m180_x3" => (
            "square",
            vec![
                DefectSpec::disclination(-180.0, [0.0, 0.5]),
                DefectSpec::disclination(-180.0, [3.0, 0.5]),
                DefectSpec::disclination(-180.0, [0.0, 3.5]),
            ],
            None,
        ),
        other => panic!("unexpected build case {other}"),
    }
}

#[test]
fn gate_stage4_builds_core_isomorphism_and_drawings() {
    let Ok(gate_path) = std::env::var("PT_GATE_STAGE4") else {
        eprintln!("PT_GATE_STAGE4 not set; skipping external JSON gate in bare test run");
        return;
    };
    let raw_json = std::fs::read_to_string(&gate_path).expect("read PT_GATE_STAGE4");
    let gate: GateFile = serde_json::from_str(&raw_json).expect("parse PT_GATE_STAGE4");
    assert_eq!(gate.builds.len(), 9);

    // 0. Perfect-patch crowding/overlap/crossing control across all 12 sweep lattices (control 7a)
    let t_perf = Instant::now();
    for (tag, sw) in &gate.sweep {
        let cell = sw.cell.to_cell2d();
        let lat = PerfectLattice::new(cell);
        let keys = lat.sites_in_disk([0.0, 0.0], sw.r0);
        let idx: HashMap<_, usize> = keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
        let pos: Vec<[f64; 2]> = keys.iter().map(|&k| lat.pos(k)).collect();
        let mut adj = vec![Vec::new(); keys.len()];
        let mut edges = Vec::new();
        for (i, &k) in keys.iter().enumerate() {
            for (k2, kind) in lat.nbrs(k) {
                if let Some(&j) = idx.get(&k2) {
                    if i < j {
                        edges.push((i, j));
                    }
                    adj[i].push(DirectedBondData {
                        to: j,
                        vec: [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]],
                        kind,
                        seam: None,
                        ang: None,
                        across: false,
                        ghost_stub: None,
                    });
                }
            }
        }
        edges.sort_unstable();
        let patch = BuiltCrystal {
            ref_pos: pos.clone(),
            pos,
            basis_idx: keys.iter().map(|k| k.2).collect(),
            side: vec![None; keys.len()],
            cone_sites: Vec::new(),
            edges,
            adj,
            wall_sigma: None,
            is_non_orientable: false,
        };
        let rep = check(&patch);
        assert_eq!(
            (rep.crossings, rep.overlaps, rep.crowded),
            (0, 0, 0),
            "perfect patch {tag} must have 0 crossings, 0 overlaps, 0 crowded"
        );
    }
    println!("# time 12 perfect-patch planarity controls: {:.3} s", t_perf.elapsed().as_secs_f64());

    // 1. All 9 builds + exact petgraph VF2 degree-labelled core isomorphism
    let t_builds = Instant::now();
    let mut raw_built: HashMap<String, BuiltCrystal> = HashMap::new();
    for b in &gate.builds {
        let t0 = Instant::now();
        let (lat_tag, specs, r_opt) = gate_specs_for(&b.name);
        let cell = gate.sweep[lat_tag].cell.to_cell2d();
        let lat_info = LatticeCoreInfo::new(&cell);
        let (ctr, r_calc) = patch_centre_and_radius(&lat_info, &specs, r_opt, None).unwrap();
        assert!((r_calc - b.r_patch).abs() < 1e-9, "{}: R mismatch {} vs {}", b.name, r_calc, b.r_patch);

        let crystal = build_point_defects(&cell, &specs, r_opt, None)
            .unwrap_or_else(|e| panic!("{} failed to build: {e}", b.name));
        let dir_edges: usize = crystal.adj.iter().map(Vec::len).sum();
        assert_eq!(crystal.pos.len(), b.sites, "{}: sites mismatch", b.name);
        assert_eq!(dir_edges, b.edges, "{}: directed edges mismatch", b.name);

        let chg = total_charge(&crystal).expect("total_charge");
        assert!((chg.frank_deg - b.frank_deg).abs() < 1e-6, "{}: frank_deg mismatch", b.name);
        assert_eq!(chg.kind, b.charge_kind, "{}: charge_kind mismatch", b.name);
        if b.name == "square_disloc_1_0" {
            let bv = chg.burgers.expect("dislocation burgers");
            assert!((bv[0] - 1.0).abs() < 1e-6 && bv[1].abs() < 1e-6);
        }

        let res_patch = core_graph_on_patch(&crystal, &lat_info, ctr, r_calc, 2);
        let res_auto = core_graph(&crystal, &lat_info, 2);
        if b.core_reaches_rim {
            assert!(res_patch.is_err(), "{}: expected core_graph_on_patch Err", b.name);
            assert!(res_auto.is_err(), "{}: expected core_graph Err", b.name);
        } else {
            let g_patch = res_patch.unwrap_or_else(|e| panic!("{}: {e}", b.name));
            let g_auto = res_auto.unwrap_or_else(|e| panic!("{} auto: {e}", b.name));
            assert_eq!(g_patch.node_count(), b.core_nodes, "{}: core_nodes", b.name);
            assert_eq!(g_patch.edge_count(), b.core_edges, "{}: core_edges", b.name);
            let mut degs: Vec<u16> = g_patch.node_weights().copied().collect();
            degs.sort_unstable();
            assert_eq!(degs, b.core_degs, "{}: core_degs", b.name);

            let mut expected = UnGraph::<u16, ()>::new_undirected();
            let nodes: Vec<_> = b.core_node_degs.iter().map(|&d| expected.add_node(d)).collect();
            for &[u, v] in &b.core_edge_pairs {
                expected.add_edge(nodes[u], nodes[v], ());
            }
            assert!(isomorphic_cores(&g_patch, &expected), "{}: VF2 isomorphism failed", b.name);
            assert!(isomorphic_cores(&g_auto, &expected), "{}: auto VF2 isomorphism failed", b.name);
        }
        println!(
            "# time build+core {}: {:.3} s (V={}, 2E={})",
            b.name,
            t0.elapsed().as_secs_f64(),
            b.sites,
            b.edges
        );
        raw_built.insert(b.name.clone(), crystal);
    }
    println!("# time all 9 gate builds + VF2 core checks: {:.3} s", t_builds.elapsed().as_secs_f64());

    // FIX 1 explicit check: chiral_square face-centred 4b disclination at (0.5, 0.5)
    let cs_cell = gate.sweep["chiral_square"].cell.to_cell2d();
    let cs_4b = build_point_defects(&cs_cell, &[DefectSpec::disclination(-90.0, [0.5, 0.5])], None, None)
        .expect("FIX 1 chiral_square 4b face disclination must succeed");
    assert!((total_charge(&cs_4b).unwrap().frank_deg - (-90.0)).abs() < 1e-5);

    // 2. All 5 FAILING multi-defect configurations: relax_picture + ensure_embedded
    let failing = [
        "chiral_square_p90_p90",
        "honeycomb_p180_m180_3c",
        "zigzag_p180_disloc_2_0",
        "chiral_square_p90_p90_2R0",
        "square_m180_x3",
    ];
    let t_draw = Instant::now();
    for name in failing {
        let t0 = Instant::now();
        let raw = &raw_built[name];
        let raw_edges = raw.edges.clone();
        let mut crystal = raw.clone();
        relax_picture(&mut crystal);
        let info = ensure_embedded(&mut crystal, false);
        let rep = check(&crystal);
        println!(
            "# time draw {}: {:.3} s, stage={}, crossings={}, folded={:?}, overlaps={}, ghosts={}",
            name,
            t0.elapsed().as_secs_f64(),
            info.stage,
            rep.crossings,
            rep.folded,
            rep.overlaps,
            rep.ghost_bonds
        );
        assert_ne!(info.stage, "given", "{name}: expected redraw stage != given");
        assert_eq!(rep.crossings, 0, "{name}: crossings != 0");
        assert_eq!(rep.stub_crossings, 0, "{name}: stub_crossings != 0");
        assert_eq!(rep.folded, Some(0), "{name}: folded != Some(0)");
        assert_eq!(rep.overlaps, 0, "{name}: overlaps != 0");
        assert_eq!(rep.ghost_bonds, 0, "{name}: ghost_bonds != 0");
        assert_eq!(crystal.edges, raw_edges, "{name}: edge set changed");
    }
    println!("# time all 5 FAILING multi-defect drawings: {:.3} s", t_draw.elapsed().as_secs_f64());
}

#[test]
fn wall_and_mobius_drawing_controls() {
    // 1. Non-orientable Moebius seam keeps rigid stub picture and returns None
    let mut mobius = BuiltCrystal {
        pos: vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        ref_pos: vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        basis_idx: vec![0, 0, 0],
        side: vec![None; 3],
        cone_sites: Vec::new(),
        edges: vec![(0, 1), (0, 2), (1, 2)],
        adj: vec![Vec::new(); 3],
        wall_sigma: None,
        is_non_orientable: true,
    };
    let add_undir = |c: &mut BuiltCrystal, u: usize, v: usize, across: bool, stub: Option<[f64; 2]>| {
        let dv = [c.pos[v][0] - c.pos[u][0], c.pos[v][1] - c.pos[u][1]];
        c.adj[u].push(DirectedBondData { to: v, vec: dv, kind: 0, seam: None, ang: None, across, ghost_stub: stub });
        c.adj[v].push(DirectedBondData {
            to: u, vec: [-dv[0], -dv[1]], kind: 0, seam: None, ang: None, across, ghost_stub: stub,
        });
    };
    add_undir(&mut mobius, 0, 1, false, None);
    add_undir(&mut mobius, 1, 2, false, None);
    add_undir(&mut mobius, 0, 2, true, Some([0.2, 0.0]));
    let (m_out, m_info) = draw_wall_picture(&mobius);
    assert!(m_info.is_none());
    assert_eq!(m_out.pos, mobius.pos);
    assert_eq!(check(&m_out).ghost_bonds, 1);

    // 2. Reflection wall with stacked wall-line sites opens cleanly via rigid side-B translation
    let mut wall = BuiltCrystal {
        // Side A at y >= 0, Side B at y <= 0, with sites 1 (A) and 3 (B) stacked near y = 0!
        pos: vec![[0.0, 1.0], [0.0, 0.02], [1.0, 1.0], [0.0, -0.02], [1.0, -0.02], [1.0, -1.02]],
        ref_pos: vec![[0.0, 1.0], [0.0, 0.02], [1.0, 1.0], [0.0, -0.02], [1.0, -0.02], [1.0, -1.02]],
        basis_idx: vec![0; 6],
        side: vec![Some('A'), Some('A'), Some('A'), Some('B'), Some('B'), Some('B')],
        cone_sites: Vec::new(),
        edges: vec![(0, 1), (0, 2), (1, 3), (3, 4), (4, 5)],
        adj: vec![Vec::new(); 6],
        wall_sigma: Some(Isometry2 { m: [[1.0, 0.0], [0.0, -1.0]], t: [0.0, 0.0] }),
        is_non_orientable: false,
    };
    add_undir(&mut wall, 0, 1, false, None);
    add_undir(&mut wall, 0, 2, false, None);
    add_undir(&mut wall, 1, 3, true, None);
    add_undir(&mut wall, 3, 4, false, None);
    add_undir(&mut wall, 4, 5, false, None);
    let r0 = check(&wall);
    assert_eq!(r0.folded, None);
    assert!(r0.crowded > 0);
    let (w_out, w_info) = draw_wall_picture(&wall);
    let info = w_info.expect("wall picture info");
    assert_eq!(info.stage, "opened");
    assert_eq!(info.note, "");
    let r1 = check(&w_out);
    assert_eq!((r1.crossings, r1.overlaps, r1.crowded, r1.folded), (0, 0, 0, None));
}
