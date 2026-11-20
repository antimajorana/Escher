//! Boundary-condition tests (C1 WP-B1, plan §5 T3). Where the numbers come from: every count is derived by hand in
//! the comment next to it (the L x L torus of {4,4}, the reflections of p4m and p6m, the vertex-star disk of
//! plan §2.4), or is a topological invariant (χ, boundary circles, orientability) computed here from the faces by
//! code independent of the builders. Rust = Python values come from `tests/fixtures/surface_bc_py.json`, written by
//! `tests/fixtures/dump_surface_bc.py` with the pure-Python package (`PAPERTILLING_NO_RUST=1`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use serde::Deserialize;

use super::flat_bc::{TOO_SMALL, wigner_seitz};
use super::report_bc::{NOT_ISOMETRIC_NOTE, is_proved};
use super::{
    BoundaryCondition as Bc, PairsGluing, build_boundary, glide_search, pairs_refusal, surface_table, torus_hex_domain,
};
use crate::plane::catalogue::{parse_catalogue_str, parse_named_str};
use crate::types::{Bond2D, Cell2D, SurfaceMesh};

const NAMED_JSON: &str = include_str!("../../../../papertilling/data/named.json");
const POINTS_JSON: &str = include_str!("../../../../papertilling/data/points.json");
const PY_FIXTURE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/surface_bc_py.json"));

fn named(key: &str) -> Cell2D {
    parse_named_str(NAMED_JSON).unwrap().get(key).unwrap().cell.clone()
}

fn bond(i: usize, j: usize, n1: i32, n2: i32, kind: usize) -> Bond2D {
    Bond2D { i, j, n1, n2, kind }
}

fn square_kinds(k1: usize, k2: usize, diag: Option<usize>) -> Cell2D {
    let mut bonds = vec![bond(0, 0, 1, 0, k1), bond(0, 0, 0, 1, k2)];
    if let Some(k) = diag {
        bonds.push(bond(0, 0, 1, 1, k));
    }
    Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds }
}

/// Topology of a face complex, computed from the faces only: (χ, boundary circles, orientable).
fn topology(m: &SurfaceMesh) -> (i64, usize, bool) {
    let mut uses: HashMap<(usize, usize), Vec<(usize, bool)>> = HashMap::new();
    for (fi, f) in m.faces.iter().enumerate() {
        for k in 0..f.len() {
            let (a, b) = (f[k], f[(k + 1) % f.len()]);
            uses.entry((a.min(b), a.max(b))).or_default().push((fi, a < b));
        }
    }
    let chi = m.sites.len() as i64 - m.bonds.len() as i64 + m.faces.len() as i64;
    // boundary circles: components of the edges lying on exactly one face
    let mut parent: Vec<usize> = (0..m.sites.len()).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        if p[x] == x { x } else { let r = find(p, p[x]); p[x] = r; r }
    }
    let mut rim = HashSet::new();
    for (&(a, b), u) in &uses {
        if u.len() == 1 {
            rim.extend([a, b]);
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            parent[ra] = rb;
        }
    }
    let circles = rim.iter().map(|&v| find(&mut parent, v)).collect::<HashSet<_>>().len();
    // orientation: two faces on one edge must run it in opposite directions
    let mut sign: Vec<Option<bool>> = vec![None; m.faces.len()];
    let mut orientable = true;
    for start in 0..m.faces.len() {
        if sign[start].is_some() {
            continue;
        }
        sign[start] = Some(true);
        let mut stack = vec![start];
        while let Some(f) = stack.pop() {
            let sf = sign[f].unwrap();
            for k in 0..m.faces[f].len() {
                let (a, b) = (m.faces[f][k], m.faces[f][(k + 1) % m.faces[f].len()]);
                for &(g, dir) in &uses[&(a.min(b), a.max(b))] {
                    if g == f {
                        continue;
                    }
                    // f runs the edge low -> high iff (a < b) == sf; g must run it the other way.
                    let want = (a < b) ^ !sf ^ dir;
                    match sign[g] {
                        None => {
                            sign[g] = Some(want);
                            stack.push(g);
                        }
                        Some(s) if s != want => orientable = false,
                        _ => {}
                    }
                }
            }
        }
    }
    (chi, circles, orientable)
}

#[test]
fn square_torus_counts_and_too_small_refusals() {
    // {4,4} on R^2 / (L Z)^2: L^2 sites, 2 bonds per site, one square per site: V = L^2, E = 2L^2, F = L^2.
    let sq = named("square");
    for l in 3..=5 {
        let s = build_boundary(&sq, &Bc::Torus { t1: [l, 0], t2: [0, l] }).unwrap();
        let (v, e, f) = (s.mesh.sites.len(), s.mesh.bonds.len(), s.mesh.faces.len());
        assert_eq!((v, e, f), ((l * l) as usize, (2 * l * l) as usize, (l * l) as usize), "L = {l}");
        assert_eq!(topology(&s.mesh), (0, 0, true));
        assert_eq!(s.picture_note, Some(NOT_ISOMETRIC_NOTE));
        assert!(s.exact_view);
    }
    // L = 1: the bond a1 closes into a loop; L = 2: the bonds +a1 and -a1 join the same two sites (a double bond).
    for l in 1..=2 {
        let e = build_boundary(&sq, &Bc::Torus { t1: [l, 0], t2: [0, l] }).unwrap_err();
        assert!(e.reason.starts_with(TOO_SMALL) && !is_proved(&e), "L = {l}: {e}");
    }
}

#[test]
fn cylinders_on_any_circumference_are_annuli() {
    // A cylinder is an annulus: χ = 0, two boundary circles, orientable; its rolled picture is isometric.
    let cases: [(&str, [i32; 2], i32); 4] =
        [("square", [4, 0], 3), ("square", [3, 2], 13), ("honeycomb", [6, 0], 3), ("triangular", [4, 1], 12)];
    for (key, wrap, rows) in cases {
        let s = build_boundary(&named(key), &Bc::Cylinder { wrap, rows }).unwrap_or_else(|e| panic!("{key} {wrap:?}: {e}"));
        assert_eq!(topology(&s.mesh), (0, 2, true), "{key} {wrap:?}");
        assert_eq!(s.picture_note, None);
    }
    let e = build_boundary(&named("square"), &Bc::Cylinder { wrap: [0, 0], rows: 2 }).unwrap_err();
    assert!(e.reason.contains("must not be zero"), "{e}");
    let e = build_boundary(&named("square"), &Bc::Cylinder { wrap: [2, 0], rows: 2 }).unwrap_err();
    assert!(e.reason.starts_with(TOO_SMALL), "{e}");
}

#[test]
fn mobius_and_klein_topology_and_even_lengths() {
    // Möbius band: χ = 0, one boundary circle, non-orientable (so its orientation double cover is connected).
    // Klein bottle: χ = 0, closed, non-orientable. Square (p4m): class 0 is a mirror, so even lengths are allowed.
    let sq = named("square");
    let gl = glide_search(&sq).unwrap();
    assert!(gl[0].mirror);
    for steps in [3, 4] {
        let m = build_boundary(&sq, &Bc::Mobius { glide: 0, steps, width: 2 }).unwrap_or_else(|e| panic!("{steps}: {e}"));
        assert_eq!(topology(&m.mesh), (0, 1, false), "Mobius steps {steps}");
        assert_eq!(m.picture_note, Some(NOT_ISOMETRIC_NOTE));
        let k = build_boundary(&sq, &Bc::Klein { glide: 0, steps, across: 4 }).unwrap_or_else(|e| panic!("{steps}: {e}"));
        assert_eq!(topology(&k.mesh), (0, 0, false), "Klein steps {steps}");
    }
    // brick_pg (pg): glides with no mirror close only after an odd number of steps.
    let pg = named("brick_pg");
    let gl = glide_search(&pg).unwrap();
    assert!(!gl.is_empty() && gl.iter().all(|g| !g.mirror), "{gl:?}");
    let e = build_boundary(&pg, &Bc::Mobius { glide: 0, steps: 2, width: 2 }).unwrap_err();
    assert!(e.reason.contains("odd number of steps") && !is_proved(&e), "{e}");
    // chiral_square (p4): refused, proved.
    let e = build_boundary(&named("chiral_square"), &Bc::Klein { glide: 0, steps: 3, across: 3 }).unwrap_err();
    assert!(is_proved(&e) && e.reason.contains("[PROVED]"), "{e}");
}

#[test]
fn hex_domain_is_the_same_torus_on_the_wigner_seitz_cell() {
    let tri = named("triangular");
    for (t1, t2) in [([3, 0], [0, 3]), ([4, 0], [0, 4]), ([3, 1], [-1, 2])] {
        let tor = build_boundary(&tri, &Bc::Torus { t1, t2 }).unwrap();
        let hex = build_boundary(&tri, &Bc::TorusHex { t1, t2 }).unwrap();
        let count = |m: &SurfaceMesh| (m.sites.len(), m.bonds.len(), m.faces.len());
        assert_eq!(count(&hex.mesh), count(&tor.mesh), "{t1:?} {t2:?}");
        let cart = |t: [i32; 2]| [t[0] as f64 * tri.a1[0] + t[1] as f64 * tri.a2[0], t[0] as f64 * tri.a1[1] + t[1] as f64 * tri.a2[1]];
        let (v1, v2) = (cart(t1), cart(t2));
        let poly = &hex.mesh.domain_polygons[0];
        // a triangular lattice is never rectangular: 6 sides, 3 pairs
        assert_eq!((poly.len(), hex.mesh.domain_edges.len()), (6, 6));
        assert!(!hex.readout.iter().any(|r| r.contains("reduce")));
        // area of the Wigner-Seitz cell = |det(v1, v2)|
        let area = 0.5 * (0..6).map(|k| poly[k][0] * poly[(k + 1) % 6][1] - poly[(k + 1) % 6][0] * poly[k][1]).sum::<f64>();
        assert!((area.abs() - (v1[0] * v2[1] - v1[1] * v2[0]).abs()).abs() < 1e-9, "area {area}");
        // every uv is no farther from 0 than from any other point of the superlattice (it lies in the cell)
        let near: Vec<[f64; 2]> = (-2..=2).flat_map(|a| (-2..=2).map(move |b| (a, b))).filter(|&(a, b)| (a, b) != (0, 0))
            .map(|(a, b)| [a as f64 * v1[0] + b as f64 * v2[0], a as f64 * v1[1] + b as f64 * v2[1]]).collect();
        for p in &hex.mesh.uv {
            let r0 = p[0] * p[0] + p[1] * p[1];
            assert!(near.iter().all(|t| r0 <= (p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2) + 1e-9), "{p:?}");
        }
        // opposite sides: midpoint m has a partner at -m and 2m is a superlattice vector
        let in_lat = |q: [f64; 2]| {
            let d = v1[0] * v2[1] - v1[1] * v2[0];
            let (a, b) = ((q[0] * v2[1] - q[1] * v2[0]) / d, (v1[0] * q[1] - v1[1] * q[0]) / d);
            (a - a.round()).abs() < 1e-9 && (b - b.round()).abs() < 1e-9
        };
        let mids: Vec<[f64; 2]> = hex.mesh.domain_edges.iter().map(|e| [0.5 * (e.p[0] + e.q[0]), 0.5 * (e.p[1] + e.q[1])]).collect();
        for m in &mids {
            assert!(in_lat([2.0 * m[0], 2.0 * m[1]]));
            assert!(mids.iter().any(|n| (n[0] + m[0]).abs() < 1e-9 && (n[1] + m[1]).abs() < 1e-9));
        }
    }
    // skewed generators (3,0), (9,3) of the same superlattice as (3,0), (0,3): the six shortest vectors of that
    // triangular superlattice are +-3 a1, +-3 a2, +-3 (a1 + a2) (hand), and they bound its Wigner-Seitz hexagon
    let skew = build_boundary(&tri, &Bc::TorusHex { t1: [3, 0], t2: [9, 3] }).unwrap();
    let six: Vec<[f64; 2]> = [[3, 0], [0, 3], [3, 3], [-3, 0], [0, -3], [-3, -3]]
        .iter()
        .map(|t: &[i32; 2]| [t[0] as f64 * tri.a1[0] + t[1] as f64 * tri.a2[0], t[0] as f64 * tri.a1[1] + t[1] as f64 * tri.a2[1]])
        .collect();
    assert_eq!((skew.mesh.sites.len(), skew.mesh.domain_polygons[0].len()), (9, 6));
    for p in &skew.mesh.uv {
        let r0 = p[0] * p[0] + p[1] * p[1];
        assert!(six.iter().all(|t| r0 <= (p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2) + 1e-9), "skewed {p:?}");
    }
    // square supercell (3,0),(0,3): its Wigner-Seitz cell is a square, so the 3 pairs reduce to 2
    let hex = build_boundary(&named("square"), &Bc::TorusHex { t1: [3, 0], t2: [0, 3] }).unwrap();
    assert_eq!(hex.mesh.domain_edges.len(), 4);
    assert!(hex.readout.iter().any(|r| r.contains("3 pairs reduce to 2")));
    let (_, _, rect) = wigner_seitz([2.0, 0.0], [1.0, 3.0]);
    assert!(!rect);
    assert!(torus_hex_domain(&named("square"), [2, 0], [0, 2]).is_err());
}

#[test]
fn glide_search_is_kind_aware_and_basis_independent() {
    // Square lattice with bonds a1 (kind 0), a2 (kind 1), a1 + a2 (kind 2). Ignoring kinds the diagonal mirrors
    // x <-> y and x <-> -y keep the bond set; they swap kinds 0 and 1, and every other reflection maps a1 + a2 to
    // a non-bond. So with kinds there is no mirror or glide; with all kinds equal there are 2 reflection classes.
    assert!(glide_search(&square_kinds(0, 1, Some(2))).unwrap().is_empty());
    // The verdict table follows the decorated cell, not the label: labelled p4m, this cell still has no mirror or
    // glide once kinds count, so the Möbius and Klein rows are refusals proved on the lattice.
    let rows = surface_table("p4m", &square_kinds(0, 1, Some(2)));
    assert_eq!((rows[2].status.as_str(), rows[3].status.as_str()), ("IMPOSSIBLE", "IMPOSSIBLE"), "{:?}", &rows[2..4]);
    assert_eq!(glide_search(&square_kinds(0, 0, Some(0))).unwrap().len(), 2);
    // Two kinds on the axes: the axis mirrors keep kinds, the diagonals swap them: 2 classes (4 without kinds).
    assert_eq!(glide_search(&square_kinds(0, 1, None)).unwrap().len(), 2);
    assert_eq!(glide_search(&square_kinds(0, 0, None)).unwrap().len(), 4);
    // p4m has 4 reflection linear parts and p6m 6, in any basis: the square lattice in the skewed basis
    // a1' = a1, a2' = 5 a1 + a2 (bond a2 = a2' - 5 a1'), and the triangular lattice in a1' = a1, a2' = 4 a1 + a2.
    let skew_sq = Cell2D {
        a1: [1.0, 0.0],
        a2: [5.0, 1.0],
        basis: vec![[0.0, 0.0]],
        bonds: vec![bond(0, 0, 1, 0, 0), bond(0, 0, -5, 1, 0)],
    };
    assert_eq!(glide_search(&skew_sq).unwrap().len(), 4);
    let tri = named("triangular");
    assert_eq!(glide_search(&tri).unwrap().len(), 6);
    let a2 = [4.0 * tri.a1[0] + tri.a2[0], 4.0 * tri.a1[1] + tri.a2[1]];
    let skew_tri = Cell2D {
        a1: tri.a1,
        a2,
        basis: vec![[0.0, 0.0]],
        bonds: vec![bond(0, 0, 1, 0, 0), bond(0, 0, -4, 1, 0), bond(0, 0, -3, 1, 0)],
    };
    assert_eq!(glide_search(&skew_tri).unwrap().len(), 6);
    // the skewed square still builds a Möbius band (its axis vectors have large coefficients in this basis)
    let m = build_boundary(&skew_sq, &Bc::Mobius { glide: 0, steps: 3, width: 2 }).unwrap();
    assert_eq!(topology(&m.mesh), (0, 1, false));
}

#[test]
fn non_primitive_cell_gives_not_found_not_a_proof() {
    // Rectangle a = 1, b = 1.7, one site, bonds a1 (kind 0) and a2 (kind 1): pattern group p2mm, whose reflections
    // have the linear parts x -> -x and y -> -y, so 2 classes on the primitive cell (hand count).
    let base = Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.7], basis: vec![[0.0, 0.0]], bonds: vec![bond(0, 0, 1, 0, 0), bond(0, 0, 0, 1, 1)] };
    assert_eq!(glide_search(&base).unwrap().len(), 2);
    // The same pattern on the supercell S = (2,1), (0,3) (index 6): its lattice is {(2p, p + 3q)}. Both mirrors move
    // (2,1) to (-2,1) or (2,-1), which are not in it, so no reflection of the pattern keeps the cell lattice and the
    // search finds none. The empty result must not be read as a proof.
    let s = [[2i32, 0], [1, 3]]; // columns (2,1), (0,3) in base coordinates
    let inv = |p: [i32; 2]| [(3 * p[0]) as f64 / 6.0, (-p[0] + 2 * p[1]) as f64 / 6.0]; // S^-1 p
    let mut pts: Vec<[i32; 2]> = Vec::new();
    for x in -6..=6 {
        for y in -6..=6 {
            let f = inv([x, y]);
            if f.iter().all(|&c| (-1e-9..1.0 - 1e-9).contains(&c)) {
                pts.push([x, y]);
            }
        }
    }
    assert_eq!(pts.len(), 6);
    let locate = |p: [i32; 2]| {
        let f = inv(p);
        let n = [(f[0] + 1e-9).floor() as i32, (f[1] + 1e-9).floor() as i32];
        let r = [p[0] - s[0][0] * n[0] - s[0][1] * n[1], p[1] - s[1][0] * n[0] - s[1][1] * n[1]];
        (pts.iter().position(|&q| q == r).unwrap(), n)
    };
    let mut bonds = Vec::new();
    for (i, p) in pts.iter().enumerate() {
        for (d, k) in [([1, 0], 0), ([0, 1], 1)] {
            let (j, n) = locate([p[0] + d[0], p[1] + d[1]]);
            bonds.push(bond(i, j, n[0], n[1], k));
        }
    }
    let sup = Cell2D {
        a1: [2.0, 1.7],
        a2: [0.0, 5.1],
        basis: pts.iter().map(|p| [p[0] as f64, 1.7 * p[1] as f64]).collect(),
        bonds,
    };
    let err = glide_search(&sup).unwrap_err();
    assert!(err.starts_with("NOT FOUND") && err.contains("not primitive"), "{err}");
    let refusal = build_boundary(&sup, &Bc::Mobius { glide: 0, steps: 3, width: 2 }).unwrap_err();
    assert!(!is_proved(&refusal), "{refusal}");
}

#[test]
fn chiral_groups_refuse_twisted_gluings_on_every_catalogue_lattice() {
    // [PROVED] (refusal.py proof (a), glides.rs search): a decorated lattice whose symmetry has no reflection
    // carries no Möbius band or Klein bottle; one with a mirror or glide line has a glide class.
    let t0 = Instant::now();
    let cats = parse_catalogue_str(POINTS_JSON).unwrap();
    let (mut n, mut chiral) = (0, 0);
    for (g, cat) in &cats {
        for e in &cat.lattices {
            let cell = e.family.cell();
            let ident = cell.try_identify_wallpaper_group(true).unwrap_or_else(|_| g.clone());
            let gl = glide_search(&cell).unwrap();
            let is_chiral = matches!(ident.as_str(), "p1" | "p2" | "p3" | "p4" | "p6");
            assert_eq!(gl.is_empty(), is_chiral, "{g} (identified {ident}): {} classes", gl.len());
            if is_chiral {
                chiral += 1;
                let err = build_boundary(&cell, &Bc::Mobius { glide: 0, steps: 3, width: 2 }).unwrap_err();
                assert!(is_proved(&err), "{g}: {err}");
            }
            n += 1;
        }
    }
    println!("# time glide refusal predicate on {n} catalogue lattices ({chiral} chiral): {:?}", t0.elapsed());
    assert_eq!(chiral, 35);
}

#[test]
fn pairs_refusals() {
    assert!(pairs_refusal(1, PairsGluing::Translations).is_none());
    assert!(pairs_refusal(3, PairsGluing::Translations).is_none());
    assert!(pairs_refusal(2, PairsGluing::OneTwisted).is_none());
    let rp2 = pairs_refusal(2, PairsGluing::AllTwisted).unwrap();
    assert!(is_proved(&rp2) && rp2.reason.contains("cone metric"));
    assert!(pairs_refusal(3, PairsGluing::OneTwisted).unwrap().reason.contains("Q7"));
    let four = pairs_refusal(4, PairsGluing::Translations).unwrap();
    assert!(!is_proved(&four) && four.reason.contains("parallelogram") && four.reason.contains("chi = 0"));
}

#[test]
fn open_patch_is_the_vertex_star_disk() {
    // {4,4} vertex seed, layer n: a (2n+2) x (2n+2) block of squares (plan §2.4, hand-derived), so
    // F = (2n+2)^2, V = (2n+3)^2, E = 2 (2n+2)(2n+3), χ = 1, one boundary circle; flat, no picture note.
    for n in 0..=3usize {
        let s = build_boundary(&named("square"), &Bc::Open { layers: n }).unwrap();
        let k = 2 * n + 2;
        assert_eq!((s.mesh.faces.len(), s.mesh.sites.len(), s.mesh.bonds.len()), (k * k, (k + 1) * (k + 1), 2 * k * (k + 1)));
        assert_eq!(topology(&s.mesh), (1, 1, true));
        assert_eq!(s.picture_note, None);
        assert!(!s.readout.iter().any(|r| r.contains("not a disk")));
    }
}

#[derive(Deserialize)]
struct PyCase {
    label: String,
    #[serde(default)]
    group: String,
    cell: Option<[serde_json::Value; 4]>,
    statuses: Option<BTreeMap<String, String>>,
    counts: Option<[usize; 3]>,
}

fn json_cell(v: &[serde_json::Value; 4]) -> Cell2D {
    let f2 = |x: &serde_json::Value| [x[0].as_f64().unwrap(), x[1].as_f64().unwrap()];
    Cell2D {
        a1: f2(&v[0]),
        a2: f2(&v[1]),
        basis: v[2].as_array().unwrap().iter().map(f2).collect(),
        bonds: v[3].as_array().unwrap().iter().map(|b| {
            let n = |k: usize| b[k].as_i64().unwrap();
            bond(n(0) as usize, n(1) as usize, n(2) as i32, n(3) as i32, n(4) as usize)
        }).collect(),
    }
}

#[test]
fn c1_regression_and_counts_match_python() {
    let cases: Vec<PyCase> = serde_json::from_str(PY_FIXTURE).unwrap();
    let mut seen = 0;
    for c in &cases {
        let cell = json_cell(c.cell.as_ref().unwrap());
        if let Some(want) = &c.statuses {
            // C1: the oblique cell tuned to b = 1, gamma = 90 deg (label p2) has mirrors, so the twisted surfaces and
            // the projective plane are not IMPOSSIBLE; and Rust = Python row by row.
            let rows = surface_table(&c.group, &cell);
            for (name, st) in want {
                let row = rows.iter().find(|r| &r.surface == name).unwrap();
                assert_eq!(&row.status, st, "{} {name}: {}", c.label, row.reason);
                if c.label.starts_with("oblique_tuned") && name != "sphere" {
                    assert_ne!(row.status, "IMPOSSIBLE");
                }
            }
            seen += 1;
        }
        if let Some(want) = c.counts {
            let bc: Bc = match c.label.split(':').next().unwrap() {
                "torus_hex" => Bc::TorusHex { t1: [3, 0], t2: [0, 3] },
                "cylinder_6_0" => Bc::Cylinder { wrap: [6, 0], rows: 3 },
                "cylinder_3_1" => Bc::Cylinder { wrap: [3, 1], rows: 8 },
                "mobius_even" => Bc::Mobius { glide: 0, steps: 4, width: 2 },
                "klein_even" => Bc::Klein { glide: 0, steps: 4, across: 4 },
                other => panic!("unknown case {other}"),
            };
            let s = build_boundary(&cell, &bc).unwrap_or_else(|e| panic!("{}: {e}", c.label));
            assert_eq!([s.mesh.sites.len(), s.mesh.bonds.len(), s.mesh.faces.len()], want, "{}", c.label);
            seen += 1;
        }
    }
    assert!(seen >= 6, "fixture has {seen} cases");
}
