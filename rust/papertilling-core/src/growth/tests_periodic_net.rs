//! Periodic-net (Dirichlet complex) growth tests, independent of any group data (C1 WP-L1).
//!
//! Every count below is hand-derived, not read from the code under test:
//! - square net: the Dirichlet cells are unit squares, so the touching rule grows a box of whole
//!   squares: cell seed (2n+1)², vertex seed (2n+2)², edge seed (2n+1)(2n+2) (plan §2.4 {4,4});
//! - triangular net (one site): hexagonal cells, the {6,3} forms 3n²+3n+1, 3(n+1)², (3n+2)(n+1);
//! - honeycomb net (two sites): triangular cells, the {3,6} forms 6n²+6n+1, 6(n+1)², 2(3n+1)(n+1);
//! - mirror bilayer of the square net, sites (0,0,±h): each cell is a square prism above or below
//!   z = 0 and each Voronoi vertex (a prism corner at z = 0) is in 8 cells, so the touching rule grows
//!   whole 2-site columns: cell seed 1 then 2(2n+1)², vertex seed 2(2n+2)²;
//! - simple cubic (d = 3): cubes, plan §2.4 E³ forms (2n+1)³, (2n+2)³, (2n+1)(2n+2)², (2n+1)²(2n+2).

use super::periodic_net::{NetMode, PeriodicNet};
use super::{grow, CellComplex, GrowthCaps, SeedKind, Stopped};

const SQ3_2: f64 = 0.866_025_403_784_438_6;

fn counts(net: &mut PeriodicNet, kind: SeedKind, layers: u32) -> (Vec<usize>, Vec<i64>) {
    let seed = net.seed_cells(kind).unwrap();
    let g = grow(net, seed, GrowthCaps { max_layers: layers, max_cells: 1_000_000 });
    assert!(g.stopped.is_none(), "{:?}", g.stopped);
    (g.stats.iter().map(|s| s.cells).collect(), g.stats.iter().map(|s| s.euler).collect())
}

fn check(net: &mut PeriodicNet, kind: SeedKind, layers: u32, f: impl Fn(usize) -> usize, chi: bool) {
    let (cells, euler) = counts(net, kind, layers);
    let want: Vec<usize> = (0..=layers as usize).map(f).collect();
    assert_eq!(cells, want, "{kind:?}");
    if chi {
        assert!(euler.iter().all(|&e| e == 1), "{kind:?} chi {euler:?}");
    }
}

#[test]
fn square_net_matches_the_44_forms() {
    let mut net = PeriodicNet::new(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], &[[0.0, 0.0, 0.0]]).unwrap();
    assert_eq!(net.mode(), NetMode::Flat);
    assert!(net.euler_defined());
    assert_eq!(net.polytopes().len(), 1); // one square of 4 cocircular sites per cell
    assert_eq!(net.polytopes()[0].len(), 4);
    check(&mut net, SeedKind::Cell, 8, |n| (2 * n + 1).pow(2), true);
    check(&mut net, SeedKind::Vertex, 8, |n| (2 * n + 2).pow(2), true);
    check(&mut net, SeedKind::Edge, 8, |n| (2 * n + 1) * (2 * n + 2), true);
    assert!(net.seed_cells(SeedKind::Face).is_err());
    // coordination 4 through the Dirichlet edges (one edge per neighbour)
    assert_eq!(net.edges_of((0, [0; 3])).len(), 4);
}

#[test]
fn triangular_and_honeycomb_nets_match_the_63_and_36_forms() {
    let tri = [[1.0, 0.0, 0.0], [-0.5, SQ3_2, 0.0]];
    let mut t = PeriodicNet::new(&tri, &[[0.0, 0.0, 0.0]]).unwrap();
    check(&mut t, SeedKind::Cell, 8, |n| 3 * n * n + 3 * n + 1, true);
    check(&mut t, SeedKind::Vertex, 8, |n| 3 * (n + 1).pow(2), true);
    check(&mut t, SeedKind::Edge, 8, |n| (3 * n + 2) * (n + 1), true);
    let honey = [[0.5, SQ3_2 / 3.0, 0.0], [0.0, 2.0 * SQ3_2 / 3.0, 0.0]];
    let mut h = PeriodicNet::new(&tri, &honey).unwrap();
    check(&mut h, SeedKind::Cell, 8, |n| 6 * n * n + 6 * n + 1, true);
    check(&mut h, SeedKind::Vertex, 8, |n| 6 * (n + 1).pow(2), true);
    check(&mut h, SeedKind::Edge, 8, |n| 2 * (3 * n + 1) * (n + 1), true);
}

#[test]
fn mirror_bilayer_grows_whole_columns_and_reports_no_chi() {
    let mut net = PeriodicNet::new(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], &[[0.0, 0.0, 0.3], [0.0, 0.0, -0.3]]).unwrap();
    assert_eq!(net.mode(), NetMode::Slab);
    assert!(!net.euler_defined());
    assert_eq!(net.polytopes().len(), 1); // the 8 corners of a 1 x 1 x 0.6 box
    assert_eq!(net.polytopes()[0].len(), 8);
    check(&mut net, SeedKind::Cell, 6, |n| if n == 0 { 1 } else { 2 * (2 * n + 1).pow(2) }, false);
    check(&mut net, SeedKind::Vertex, 6, |n| 2 * (2 * n + 2).pow(2), false);
}

#[test]
fn simple_cubic_matches_the_e3_forms() {
    let l = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut net = PeriodicNet::new(&l, &[[0.0, 0.0, 0.0]]).unwrap();
    assert_eq!(net.mode(), NetMode::Space);
    check(&mut net, SeedKind::Cell, 4, |n| (2 * n + 1).pow(3), true);
    check(&mut net, SeedKind::Vertex, 4, |n| (2 * n + 2).pow(3), true);
    check(&mut net, SeedKind::Edge, 4, |n| (2 * n + 1) * (2 * n + 2).pow(2), true);
    check(&mut net, SeedKind::Face, 4, |n| (2 * n + 1).pow(2) * (2 * n + 2), true);
    let c0 = (0, [0; 3]);
    assert_eq!((net.vertices_of(c0).len(), net.edges_of(c0).len(), net.faces_of(c0).len()), (8, 12, 6));
}

#[test]
fn cell_cap_stops_before_a_partial_layer() {
    let mut net = PeriodicNet::new(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], &[[0.0, 0.0, 0.0]]).unwrap();
    let seed = net.seed_cells(SeedKind::Cell).unwrap();
    // layer 2 is 25 cells, layer 3 would be 49 > 30
    let g = grow(&mut net, seed, GrowthCaps { max_layers: 8, max_cells: 30 });
    assert_eq!(g.stats.last().unwrap().cells, 25);
    assert_eq!(g.stopped, Some(Stopped::CellCap { layer: 3, partial: 31 }));
}

#[test]
fn shifted_and_oblique_cells_give_the_same_complex() {
    // an oblique basis of the square lattice, (1,0) and (1,1), and a site off the origin
    let mut net = PeriodicNet::new(&[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], &[[0.3, 0.2, 0.0]]).unwrap();
    check(&mut net, SeedKind::Cell, 5, |n| (2 * n + 1).pow(2), true);
}
