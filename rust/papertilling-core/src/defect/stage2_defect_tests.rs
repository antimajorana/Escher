//! Hand-count tests for the Stage 2 defect modules: point defects, Stone-Wales flips, CSL grain boundaries.

use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use crate::defect::burgers_circuit;
use crate::defect::charge::{check_frames, frame_faces, outer_face_index};
use crate::defect::grain_boundary::{BicrystalOptions, build_symmetric_tilt_bicrystal, list_csl_grain_boundaries, perfect_disk};
use crate::defect::point_defects::{PointDefectOp, apply_point_defects, rim_sites};
use crate::defect::stone_wales::stone_wales_flip;
use crate::plane::catalogue::{NamedLattice, parse_named_str};
use crate::types::BuiltCrystal;

const NAMED_JSON: &str = include_str!("../../../../papertilling/data/named.json");

fn named(key: &str) -> NamedLattice {
    parse_named_str(NAMED_JSON).expect("named.json").get(key).expect("named lattice").clone()
}

fn interior_faces(c: &BuiltCrystal) -> Vec<Vec<usize>> {
    let f = frame_faces(c);
    let k = outer_face_index(&f);
    f.into_iter().enumerate().filter(|&(i, _)| i != k).map(|(_, x)| x).collect()
}

fn closed(face: &[usize]) -> Vec<usize> {
    let mut p = face.to_vec();
    p.push(face[0]);
    p
}

fn nearest(c: &BuiltCrystal, q: [f64; 2]) -> usize {
    (0..c.pos.len()).min_by(|&a, &b| {
        let da = (c.pos[a][0] - q[0]).hypot(c.pos[a][1] - q[1]);
        let db = (c.pos[b][0] - q[0]).hypot(c.pos[b][1] - q[1]);
        da.total_cmp(&db)
    }).unwrap()
}

#[test]
fn honeycomb_vacancy_counts_and_trivial_holonomy() {
    let hc = named("honeycomb");
    let disk = perfect_disk(&hc.cell, hc.cell.basis[0], 4.5);
    let (n, e) = (disk.pos.len(), disk.edges.len());
    let v = nearest(&disk, hc.cell.basis[0]);
    assert_eq!(disk.adj[v].len(), 3);
    let res = apply_point_defects(&disk, &[PointDefectOp::Vacancy { site: v, reconstruct_ring: false }]).unwrap();
    assert_eq!(res.crystal.pos.len(), n - 1);
    assert_eq!(res.crystal.edges.len(), e - 3);
    assert_eq!(check_frames(&res.crystal), 0);
    // removing a degree-3 site merges three hexagons into one 3 * (6 - 2) = 12-ring around the vacancy
    let faces = interior_faces(&res.crystal);
    let ring = faces.iter().find(|f| f.len() == 12).expect("12-ring around the vacancy").clone();
    let circ = burgers_circuit(&res.crystal, &closed(&ring), None).unwrap();
    assert_eq!(circ.closure_vector, [0.0, 0.0]);
    assert_eq!(circ.frank_angle_deg, Some(0.0));
    assert_eq!((res.holonomy_badge, res.core_badge), ("[PROVED]", "[PROVED]"));
    // reconstructed: one chord between two of the three neighbours -> 5-ring + 9-ring (5-9 vacancy)
    let rec = apply_point_defects(&disk, &[PointDefectOp::Vacancy { site: v, reconstruct_ring: true }]).unwrap();
    assert_eq!(rec.crystal.edges.len(), e - 3 + 1);
    let mut sizes: Vec<usize> = interior_faces(&rec.crystal).iter().map(Vec::len).filter(|&s| s != 6).collect();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![5, 9]);
    assert_eq!(check_frames(&rec.crystal), 0);
    let circ = burgers_circuit(&rec.crystal, &closed(&ring), None).unwrap();
    assert_eq!(circ.closure_vector, [0.0, 0.0]);
    assert_eq!(circ.frank_angle_deg, Some(0.0));
    assert_eq!(rec.core_badge, "[COMPUTATIONAL EVIDENCE]");
    // rim and out-of-range sites are refused
    let rim = rim_sites(&disk);
    let r0 = *rim.iter().next().unwrap();
    assert!(apply_point_defects(&disk, &[PointDefectOp::Vacancy { site: r0, reconstruct_ring: false }]).is_err());
    assert!(apply_point_defects(&disk, &[PointDefectOp::Vacancy { site: n, reconstruct_ring: false }]).is_err());
}

#[test]
fn square_substitution_interstitial_adatom() {
    let sq = named("square");
    let disk = perfect_disk(&sq.cell, [0.0, 0.0], 3.0);
    let (n, e) = (disk.pos.len(), disk.edges.len());
    let c = nearest(&disk, [0.0, 0.0]);
    let ops = vec![
        PointDefectOp::Substitutional { site: c, species: "Si".into() },
        // the centre of a square plaquette is sqrt(1/2) from its 4 corners and >= sqrt(5/2) from all other sites
        PointDefectOp::Interstitial { pos: [0.5, 0.5], species: "H".into(), bond_cutoff: 0.75 },
        PointDefectOp::Adatom { host_site: c, height: 1.5, species: "O".into() },
        PointDefectOp::Vacancy { site: nearest(&disk, [-1.0, -1.0]), reconstruct_ring: true },
    ];
    let res = apply_point_defects(&disk, &ops).unwrap();
    let cr = &res.crystal;
    assert_eq!(cr.pos.len(), n + 2 - 1);
    // +4 interstitial bonds, +1 adatom bond, -4 vacancy bonds, +2 reconstruction chords (square: 4 neighbours -> 2 pairs)
    assert_eq!(cr.edges.len(), e + 4 + 1 - 4 + 2);
    let idx = |q: [f64; 2]| nearest(cr, q);
    assert_eq!(res.species[idx([0.0, 0.0])], "Si");
    let inter = cr.pos.len() - 2;
    let ad = cr.pos.len() - 1;
    assert_eq!(cr.adj[inter].len(), 4);
    assert_eq!(res.species[inter], "H");
    assert_eq!(res.pos3d[ad], [0.0, 0.0, 1.5]);
    assert_eq!(cr.adj[ad].len(), 1);
    assert_eq!(res.pos3d[idx([1.0, 0.0])][2], 0.0);
    assert!(cr.adj.iter().enumerate().all(|(u, l)| l.iter().all(|b| cr.adj[b.to].iter().any(|x| x.to == u))));
    let bad = apply_point_defects(&disk, &[PointDefectOp::Substitutional { site: c, species: "si".into() }]);
    assert!(bad.is_err());
    let bad = apply_point_defects(&disk, &[PointDefectOp::Adatom { host_site: c, height: 0.0, species: "O".into() }]);
    assert!(bad.is_err());
}

#[test]
fn honeycomb_stone_wales_5775() {
    let t0 = Instant::now();
    let hc = named("honeycomb");
    let disk = perfect_disk(&hc.cell, [0.0, 0.0], 5.0);
    let (v0, e0) = (disk.pos.len(), disk.edges.len());
    let f0 = frame_faces(&disk).len();
    // the bond nearest the origin
    let k = (0..disk.edges.len()).min_by(|&a, &b| {
        let m = |i: usize| { let (u, v) = disk.edges[i]; (0.5 * (disk.pos[u][0] + disk.pos[v][0])).hypot(0.5 * (disk.pos[u][1] + disk.pos[v][1])) };
        m(a).total_cmp(&m(b))
    }).unwrap();
    let res = stone_wales_flip(&disk, k, None).unwrap();
    let c = &res.crystal;
    assert_eq!(res.before_rings, [6, 6, 6, 6]);
    let mut after = res.after_rings;
    after.sort_unstable();
    assert_eq!(after, [5, 5, 7, 7]);
    let faces = frame_faces(c);
    assert_eq!((c.pos.len(), c.edges.len(), faces.len()), (v0, e0, f0));
    assert_eq!(v0 as i64 - e0 as i64 + f0 as i64, 2);
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    for f in interior_faces(c) {
        *hist.entry(f.len()).or_default() += 1;
    }
    let hist0: usize = interior_faces(&disk).len();
    assert_eq!(hist.get(&5), Some(&2));
    assert_eq!(hist.get(&7), Some(&2));
    assert_eq!(hist.get(&6).copied().unwrap_or(0), hist0 - 4);
    assert_eq!(hist.len(), 3, "only 5-, 6- and 7-rings");
    // circuit round the four changed rings: the outer boundary walk of their union
    let (u, v) = res.flipped_edge;
    let core: Vec<Vec<usize>> = interior_faces(c).into_iter().filter(|f| f.len() != 6 && (f.contains(&u) || f.contains(&v))).collect();
    assert_eq!(core.len(), 4);
    let loop_walk = boundary_walk(&core);
    let circ = burgers_circuit(c, &closed(&loop_walk), None).unwrap();
    assert!(circ.burgers_norm < 1e-9, "Stone-Wales core has zero net Burgers vector");
    assert_eq!(circ.frank_angle_deg, Some(0.0));
    assert_eq!(res.holonomy_badge, "[PROVED]");
    // relaxed: the 5-7-7-5 core relaxes (energy drops) and stays the same graph
    let relaxed = stone_wales_flip(&disk, k, Some(&hc.cell)).unwrap();
    assert_eq!(relaxed.relax_badge, "[COMPUTATIONAL EVIDENCE]");
    assert_eq!(relaxed.crystal.edges, c.edges);
    let bl = |cr: &BuiltCrystal| { let (a, b) = (cr.pos[u], cr.pos[v]); (a[0] - b[0]).hypot(a[1] - b[1]) };
    assert!((bl(c) - 1.0).abs() < 1e-9);
    assert!(relaxed.crystal.pos.iter().all(|p| p[0].is_finite() && p[1].is_finite()));
    let rim: Vec<usize> = rim_sites(&disk).into_iter().collect();
    let drift = rim.iter().map(|&i| (relaxed.crystal.pos[i][0] - disk.pos[i][0]).hypot(relaxed.crystal.pos[i][1] - disk.pos[i][1])).fold(0.0, f64::max);
    println!("# SW relaxed: rotated bond {:.4}, max rim drift after Kabsch {drift:.4}", bl(&relaxed.crystal));
    assert!(stone_wales_flip(&disk, disk.edges.len(), None).is_err());
    println!("# time Stone-Wales: {:.4} s", t0.elapsed().as_secs_f64());
}

/// Boundary cycle of a union of faces: keep directed edges whose reverse is not in the union, then follow them.
fn boundary_walk(faces: &[Vec<usize>]) -> Vec<usize> {
    let mut dir = std::collections::HashSet::new();
    for f in faces {
        for k in 0..f.len() {
            dir.insert((f[k], f[(k + 1) % f.len()]));
        }
    }
    let bnd: BTreeMap<usize, usize> = dir.iter().filter(|(a, b)| !dir.contains(&(*b, *a))).map(|&(a, b)| (a, b)).collect();
    let start = *bnd.keys().next().unwrap();
    let mut walk = vec![start];
    let mut cur = bnd[&start];
    while cur != start {
        walk.push(cur);
        cur = bnd[&cur];
    }
    walk
}

#[test]
fn csl_hand_formulas() {
    let hc = named("honeycomb");
    let lst = list_csl_grain_boundaries(&hc.group, &hc.cell, 19);
    let ang = |s: u32| lst.iter().filter(|e| e.sigma == s).map(|e| e.theta_deg).collect::<Vec<_>>();
    let s7 = 2.0 * (3f64.sqrt() / 9.0).atan().to_degrees();
    assert_eq!(ang(7).len(), 2);
    assert!((ang(7)[0] - s7).abs() < 1e-9 && (ang(7)[0] - 21.786789).abs() < 1e-6);
    assert!((ang(7)[1] - (60.0 - s7)).abs() < 1e-9 && (ang(7)[1] - 38.213211).abs() < 1e-6);
    assert!(ang(13).iter().any(|t| (t - (11.0f64 / 13.0).acos().to_degrees()).abs() < 1e-9 && (t - 32.204228).abs() < 1e-6));
    assert!(ang(19).iter().any(|t| (t - (37.0f64 / 38.0).acos().to_degrees()).abs() < 1e-9 && (t - 13.173551).abs() < 1e-6));
    assert_eq!(lst.iter().map(|e| e.sigma).collect::<std::collections::BTreeSet<_>>().into_iter().collect::<Vec<_>>(), vec![7, 13, 19]);
    assert!(lst.windows(2).all(|w| (w[0].sigma, w[0].theta_deg) < (w[1].sigma, w[1].theta_deg)));
    let sq = named("square");
    let l = list_csl_grain_boundaries(&sq.group, &sq.cell, 17);
    let sa = |s: u32| l.iter().filter(|e| e.sigma == s).map(|e| e.theta_deg).collect::<Vec<_>>();
    assert!((sa(5)[0] - 36.869898).abs() < 1e-6 && (sa(5)[1] - 53.130102).abs() < 1e-6 && sa(5).len() == 2);
    assert!(sa(13).iter().any(|t| (t - 22.619865).abs() < 1e-6));
    assert!(sa(17).iter().any(|t| (t - 28.072487).abs() < 1e-6));
    assert!(l.iter().all(|e| e.provenance == "[PROVED]" && e.family == "square"));
    assert!(list_csl_grain_boundaries("p2", &sq.cell, 31).is_empty());
}

#[test]
fn honeycomb_sigma7_bicrystal_is_connected_two_sided() {
    let t0 = Instant::now();
    let hc = named("honeycomb");
    let s7 = list_csl_grain_boundaries(&hc.group, &hc.cell, 7)[0].clone();
    let res = build_symmetric_tilt_bicrystal(&hc.group, &hc.cell, s7.theta_deg, Some(s7.sigma), &BicrystalOptions::default()).unwrap();
    let c = &res.crystal;
    assert!(res.exact_csl && res.provenance.starts_with("[PROVED"));
    assert!(c.side.contains(&Some('A')) && c.side.contains(&Some('B')));
    let n_across = c.adj.iter().flatten().filter(|b| b.across).count();
    assert!(n_across > 0);
    assert_eq!(check_frames(c), 0);
    let mut seen = vec![false; c.pos.len()];
    let mut q = VecDeque::from([0usize]);
    seen[0] = true;
    while let Some(u) = q.pop_front() {
        for b in &c.adj[u] {
            if !seen[b.to] {
                seen[b.to] = true;
                q.push_back(b.to);
            }
        }
    }
    assert!(seen.iter().all(|&s| s), "bicrystal must be connected");
    let free = build_symmetric_tilt_bicrystal(&hc.group, &hc.cell, 10.0, None, &BicrystalOptions::default()).unwrap();
    assert!(!free.exact_csl && free.provenance.starts_with("[COMPUTATIONAL EVIDENCE"));
    println!("# time bicrystal: {:.4} s ({} sites, {} across bonds)", t0.elapsed().as_secs_f64(), c.pos.len(), n_across / 2);
}
