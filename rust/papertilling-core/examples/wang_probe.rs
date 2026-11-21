//! Diagnosis probe (not a test): what the Combine view's surgery does on the configurations of the
//! four-dislocation fusion of Wang, Liu, Lu, Phys. Rev. B 109, 195165 (2024), Fig. 7(b)-(f).
//!
//! Run: `cargo run --release --manifest-path rust/Cargo.toml -p papertilling-core --example wang_probe`

use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

use papertilling_core::defect::charge::{frame_faces, total_charge};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects_with_cuts};
use papertilling_core::plane::catalogue::parse_named_str;
use papertilling_core::types::Cell2D;

const NAMED_JSON: &str = include_str!("../../../papertilling/data/named.json");

fn d(b: [f64; 2], at: [f64; 2]) -> DefectSpec {
    DefectSpec::dislocation(b, Some(at))
}

/// Interior census: vertex degrees and face sizes away from the rim (rim = the largest face).
fn report(name: &str, cell: &Cell2D, specs: &[DefectSpec], radius: Option<f64>) -> bool {
    let t = Instant::now();
    let res = build_point_defects_with_cuts(cell, specs, radius, None);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    match res {
        Err(e) => {
            println!("REFUSED  {name:<44} n={} | {e}", specs.len());
            false
        }
        Ok((c, cuts, coalesced)) => {
            let faces = frame_faces(&c);
            let k_out = (0..faces.len()).max_by_key(|&k| faces[k].len()).expect("faces");
            let rim: HashSet<usize> = faces[k_out].iter().copied().collect();
            let mut fh: BTreeMap<usize, usize> = BTreeMap::new();
            for (k, f) in faces.iter().enumerate() {
                if k != k_out && f.len() != 4 {
                    *fh.entry(f.len()).or_default() += 1;
                }
            }
            let mut vh: BTreeMap<usize, usize> = BTreeMap::new();
            for (v, a) in c.adj.iter().enumerate() {
                if !rim.contains(&v) && a.len() != 4 {
                    *vh.entry(a.len()).or_default() += 1;
                }
            }
            let ch = total_charge(&c).map_or_else(|e| e, |q| format!("{} b={:?}", q.kind, q.burgers));
            println!(
                "ok       {name:<44} n={} V={} E={} F={} cuts={} coalesced={coalesced} | odd faces {fh:?} odd vertices {vh:?} | {ch} | {ms:.0} ms",
                specs.len(), c.pos.len(), c.edges.len(), faces.len() - 1, cuts.len()
            );
            true
        }
    }
}

fn main() {
    let named = parse_named_str(NAMED_JSON).expect("named.json");
    let lat = named.named.iter().find(|l| l.key == "square").expect("square lattice");
    let cell = &lat.cell;
    println!("cell a1={:?} a2={:?} basis={:?}", cell.a1, cell.a2, cell.basis);
    let (x, y) = ([1.0, 0.0], [0.0, 1.0]);
    let (mx, my) = ([-1.0, 0.0], [0.0, -1.0]);

    println!("\n-- one dislocation, positive control");
    report("single b=x at origin", cell, &[d(x, [0.0, 0.0])], None);
    report("single b=x at (0.5,0.5)", cell, &[d(x, [0.5, 0.5])], None);

    println!("\n-- one dislocation: does the core depend on b's direction or on the cut?");
    for (nm, b) in [("x", x), ("y", y), ("-x", mx), ("-y", my)] {
        let at = [-0.5 * b[0], -0.5 * b[1]];
        report(&format!("single b={nm}, core on a vertex, auto cut"), cell, &[d(b, at)], None);
    }
    for deg in [0.0_f64, 25.0, -25.0, 50.0, -50.0, 72.0, -72.0, 180.0, 155.0, -155.0, 130.0, -130.0, 108.0, -108.0] {
        // the candidate list of surgery.rs: rotations of c0 = b turned by +90 deg
        let th = (90.0 + deg).to_radians();
        let spec = DefectSpec::Dislocation { b: x, at: Some([-0.5, 0.0]), cut: Some([th.cos(), th.sin()]) };
        report(&format!("single b=x, cut at {:+.0} deg from x", 90.0 + deg), cell, &[spec], None);
    }

    println!("\n-- one glide dipole (b, -b), cores s lattice constants apart along b (Wang's pair A, A')");
    for off in [0.0, 0.5] {
        for s in 1..=6 {
            let sf = f64::from(s);
            report(&format!("dipole x,-x  sep {s} along x, y-offset {off}"), cell, &[d(x, [0.0, off]), d(mx, [sf + 1.0, off])], None);
        }
    }
    println!("\n-- one climb dipole (b, -b), cores s apart perpendicular to b");
    for s in 1..=5 {
        report(&format!("dipole x,-x  sep {s} along y"), cell, &[d(x, [0.0, 0.0]), d(mx, [1.0, f64::from(s)])], None);
    }

    println!("\n-- four inner dislocations only (A, B, C, D of Fig. 7b), each r from the centre");
    for r in 1..=6 {
        let r = f64::from(r);
        let specs = [d(x, [-r - 0.5, 0.0]), d(mx, [r + 0.5, 0.0]), d(y, [0.0, r - 0.5]), d(my, [0.0, -r + 0.5])];
        report(&format!("four: x,-x,y,-y at radius {r}"), cell, &specs, None);
    }

    println!("\n-- Fig. 7(b): four pairs; inner at radius r, partner s further out");
    for (r, s) in [(2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (4, 3), (4, 4), (5, 4), (6, 5)] {
        let (r, s) = (f64::from(r), f64::from(s));
        let specs = [
            d(x, [-r - 0.5, 0.0]), d(mx, [-r - s + 0.5, 0.0]),
            d(mx, [r + 0.5, 0.0]), d(x, [r + s - 0.5, 0.0]),
            d(y, [0.0, r - 0.5]), d(my, [0.0, r + s + 0.5]),
            d(my, [0.0, -r + 0.5]), d(y, [0.0, -r - s - 0.5]),
        ];
        report(&format!("eight: inner radius {r}, pair separation {s}"), cell, &specs, None);
    }

    println!("\n-- Fig. 7(f): the four inner ones moved inside the 3x3 loop, partners left at radius 5");
    for r_in in [0.0, 1.0] {
        let specs = [
            d(x, [-r_in - 0.5, 0.0]), d(mx, [-5.5, 0.0]),
            d(mx, [r_in + 0.5, 0.0]), d(x, [5.5, 0.0]),
            d(y, [0.0, r_in - 0.5]), d(my, [0.0, 6.5]),
            d(my, [0.0, -r_in + 0.5]), d(y, [0.0, -6.5]),
        ];
        report(&format!("eight: inner radius {r_in}, partners at 5"), cell, &specs, None);
    }
    println!("\n-- the four inner ones at one point (the fused object)");
    let o = [0.0, 0.0];
    report("four coincident x,-x,y,-y", cell, &[d(x, o), d(mx, o), d(y, o), d(my, o)], None);
}
