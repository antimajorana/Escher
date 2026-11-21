//! Pilot: which pairs of elementary defects of the square lattice combine to what.
//!
//! For every pair of elementary disclinations (and pairs of dislocations) up to a separation bound it
//! builds the pair, reads the total charge off the built lattice, and asks whether the pair's core is the
//! same graph as the core of a single defect with that charge.
//!
//! Run: `cargo run --release --manifest-path rust/Cargo.toml -p papertilling-core --example pair_table`
//! Every timing line starts with `# time`.

use std::collections::BTreeMap;
use std::time::Instant;

use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::combine_api::{classify_crystal_charge, equivalent_single};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects};
use papertilling_core::plane::catalogue::parse_named_str;

const NAMED_JSON: &str = include_str!("../../../papertilling/data/named.json");
const MAX_SEP: f64 = 1.5;

fn rot90(p: [f64; 2], k: usize) -> [f64; 2] {
    (0..k).fold(p, |q, _| [-q[1], q[0]])
}

/// Points of the symmetry class of `p` (4-fold rotations about the origin and lattice translations)
/// within `reach` of `from`, excluding `from` itself.
fn class_points(p: [f64; 2], from: [f64; 2], reach: f64) -> Vec<[f64; 2]> {
    let mut out: Vec<[f64; 2]> = Vec::new();
    for k in 0..4 {
        let q = rot90(p, k);
        for i in -3..=3 {
            for j in -3..=3 {
                let c = [q[0] + f64::from(i), q[1] + f64::from(j)];
                let d = (c[0] - from[0]).hypot(c[1] - from[1]);
                if d > 1e-9 && d <= reach && !out.iter().any(|o| (o[0] - c[0]).hypot(o[1] - c[1]) < 1e-9) {
                    out.push(c);
                }
            }
        }
    }
    out
}

fn main() {
    let named = parse_named_str(NAMED_JSON).expect("named.json");
    let lat = named.named.iter().find(|l| l.key == "square").expect("square lattice");
    let (group, cell, frame) = (lat.group.as_str(), &lat.cell, lat.frame);
    let cat = DefectCatalogue::new(group, cell, frame);
    println!("lattice: square, group {group}");
    for c in &cat.centre_classes {
        println!("  centre class {} ({}), order {}, at {:?}, angles {:?}", c.id, c.location, c.order, c.point, c.angles);
    }
    for d in &cat.dislocations {
        println!("  dislocation class {} b={:?} |b|={:.3} members {:?}", d.id, d.b, d.length, d.members);
    }

    let t_all = Instant::now();
    let mut rows = 0;
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    println!("\n{:<14} {:<14} {:>12}  {:<34} {:<10} {:>5}  single-defect core match", "A", "B", "B - A", "total charge", "equal?", "core");
    for a in &cat.disclinations {
        for b in &cat.disclinations {
            for pb in class_points(b.point, a.point, MAX_SEP) {
                let d = [pb[0] - a.point[0], pb[1] - a.point[1]];
                // Pairs related by a symmetry of the lattice that fixes A are the same pair.
                let (mut ax, mut ay) = (d[0].abs(), d[1].abs());
                if a.order == 4 && ax < ay {
                    std::mem::swap(&mut ax, &mut ay);
                }
                let key = format!("{}|{}|{ax:.3}|{ay:.3}", a.id, b.id);
                if seen.insert(key, ()).is_some() {
                    continue;
                }
                let specs = [DefectSpec::disclination(a.angle, a.point), DefectSpec::disclination(b.angle, pb)];
                let t = Instant::now();
                let line = match build_point_defects(cell, &specs, None, None) {
                    Err(e) => format!("REFUSED: {e}"),
                    Ok(crystal) => {
                        let charge = classify_crystal_charge(group, cell, frame, &crystal)
                            .map_or_else(|e| format!("? {e}"), |q| q.label);
                        match equivalent_single(group, cell, frame, &specs, &crystal, 2, 1, false) {
                            Ok(rep) => format!(
                                "{charge:<34} {:<10} {:>5}  {} [{}]",
                                if rep.equivalent { "yes" } else { "no" },
                                rep.core_nodes,
                                rep.matches.join(", "),
                                rep.match_tier
                            ),
                            Err(e) => format!("{charge:<34} {:<10} {:>5}  not compared: {e}", "-", "-"),
                        }
                    }
                };
                rows += 1;
                println!(
                    "{:<14} {:<14} ({:+.1},{:+.1})  {line}   # time {:.0} ms",
                    a.id, b.id, d[0], d[1], t.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
    }
    println!("\n# time total {:.1} s for {rows} rows", t_all.elapsed().as_secs_f64());
}
