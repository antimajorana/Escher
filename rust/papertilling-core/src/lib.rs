//! Geometry-agnostic tiling, Delaney-Dress symbol, surface, and defect core for `papertilling`.

pub mod bieberbach;
pub mod defect;
pub mod dsymbol;
pub mod export;
pub mod hyperbolic;
pub mod plane;
pub mod space3d;
pub mod spectrum;
pub mod subperiodic;
pub mod surface;
pub mod types;

/// Crate version matching the workspace and Python package.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use faer::Mat;
    use faer::sparse::{SparseColMat, Triplet};
    use petgraph::graph::UnGraph;

    use crate::defect::cores_isomorphic;
    use crate::dsymbol::{DSymbol, DSymbolError, canonical_key, network_key, symmetry_index};
    use crate::plane::draw::{segments_to_svg, solve_tutte_spd};
    use crate::{VERSION, plane};

    #[test]
    fn version_and_wallpaper_group_count() {
        assert_eq!(VERSION, "0.2.0");
        assert_eq!(plane::groups::WALLPAPER_GROUPS.len(), 17);
        let svg = segments_to_svg(&[(0.0, 0.0, 1.0, 0.0)], (0.0, 0.0, 1.0, 1.0));
        assert!(svg.contains("<svg") && svg.contains("<line"));
    }

    #[test]
    fn dsymbol_one_chamber_platonic_tilings() {
        // Hand-computed 1-chamber symbols under full p4m / p6m symmetry:
        // square: (4, 4, 0, 0, 0), triangular: (3, 6, 0, 0, 0), honeycomb: (6, 3, 0, 0, 0).
        let cases = [
            (4u32, 4u32, (4, 4, 0, 0, 0)),
            (3u32, 6u32, (3, 6, 0, 0, 0)),
            (6u32, 3u32, (6, 3, 0, 0, 0)),
        ];
        for (m01, m12, expected_row) in cases {
            let ds = DSymbol::new(vec![0], vec![0], vec![0], vec![m01], vec![m12]).unwrap();
            assert!(ds.is_flat());
            let key = ds.canonical_key().unwrap();
            assert_eq!(key, vec![expected_row]);
            assert_eq!(network_key(&key).unwrap(), vec![expected_row]);
            assert_eq!(symmetry_index(&key).unwrap(), 1);
        }
    }

    #[test]
    fn dsymbol_kagome_two_chambers() {
        // Kagome under p6m: 2 chambers (triangle face m01=3, hexagon face m01=6, vertex m12=4).
        // Pass in non-canonical chamber order (hexagon first, triangle second) to verify canonicalisation.
        let ds = DSymbol::new(
            vec![0, 1],
            vec![0, 1],
            vec![1, 0],
            vec![6, 3],
            vec![4, 4],
        )
        .unwrap();
        assert!(ds.is_flat());
        let expected = vec![(3, 4, 0, 0, 1), (6, 4, 1, 1, 0)];
        let key = ds.canonical_key().unwrap();
        assert_eq!(key, expected);
        assert_eq!(network_key(&key).unwrap(), expected);
        assert_eq!(symmetry_index(&key).unwrap(), 1);

        let roundtrip = DSymbol::from_json(&ds.to_json().unwrap()).unwrap();
        assert_eq!(roundtrip, ds);
        assert_eq!(DSymbol::from_key(&key).unwrap().canonical_key().unwrap(), expected);
    }

    #[test]
    fn dsymbol_p2_square_collapses_to_minimal_network() {
        // 1-site square lattice viewed under subgroup p2 (order 2 vs p4m order 8 -> 4 chambers):
        // darts d0 (+x), d1 (+y) modulo -I, sides s in {0, 1}:
        // s0 = [1, 0, 3, 2], s1 = [3, 2, 1, 0], s2 = [1, 0, 3, 2], all (m01, m12) = (4, 4).
        let ds = DSymbol::new(
            vec![1, 0, 3, 2],
            vec![3, 2, 1, 0],
            vec![1, 0, 3, 2],
            vec![4, 4, 4, 4],
            vec![4, 4, 4, 4],
        )
        .unwrap();
        assert!(ds.is_flat());
        let key = ds.canonical_key().unwrap();
        assert_eq!(
            key,
            vec![
                (4, 4, 1, 2, 1),
                (4, 4, 0, 3, 0),
                (4, 4, 3, 0, 3),
                (4, 4, 2, 1, 2),
            ]
        );
        let net = network_key(&key).unwrap();
        assert_eq!(net.len(), 1);
        assert_eq!(net, vec![(4, 4, 0, 0, 0)]);
        assert_eq!(symmetry_index(&key).unwrap(), 4);
    }

    #[test]
    fn dsymbol_negative_disconnected_rejected() {
        let err = canonical_key(&[[0, 1], [0, 1], [0, 1]], &[4, 4], &[4, 4]).unwrap_err();
        assert_eq!(err, DSymbolError::Disconnected);
    }

    #[test]
    fn faer_sparse_cholesky_tutte_laplacian_smoke() {
        // Cube graph with 4 outer corners pinned to the unit square (0,0),(1,0),(1,1),(0,1)
        // and 4 interior vertices forming a 4-cycle: L = 3*I - A_{C4}.
        // Exact analytical equilibrium: interior square at {1/3, 2/3}^2.
        let triplets = [
            Triplet::new(0, 0, 3.0),
            Triplet::new(0, 1, -1.0),
            Triplet::new(0, 3, -1.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(1, 2, -1.0),
            Triplet::new(2, 1, -1.0),
            Triplet::new(2, 2, 3.0),
            Triplet::new(2, 3, -1.0),
            Triplet::new(3, 0, -1.0),
            Triplet::new(3, 2, -1.0),
            Triplet::new(3, 3, 3.0),
        ];
        let lap = SparseColMat::<usize, f64>::try_new_from_triplets(4, 4, &triplets).unwrap();

        let rhs = Mat::from_fn(4, 2, |row, col| match (row, col) {
            (1 | 2, 0) | (2 | 3, 1) => 1.0,
            _ => 0.0,
        });
        let xy = solve_tutte_spd(&lap, &rhs).unwrap();
        let expected = [
            (1.0 / 3.0, 1.0 / 3.0),
            (2.0 / 3.0, 1.0 / 3.0),
            (2.0 / 3.0, 2.0 / 3.0),
            (1.0 / 3.0, 2.0 / 3.0),
        ];
        for (i, &(ex, ey)) in expected.iter().enumerate() {
            assert!((xy[(i, 0)] - ex).abs() < 1e-12);
            assert!((xy[(i, 1)] - ey).abs() < 1e-12);
        }
    }

    #[test]
    fn petgraph_vf2_labelled_core_isomorphism_smoke() {
        // Positive control: relabelled 4-cycles with identical adjacent node/edge labels.
        let mut g1 = UnGraph::<(u8, u8), u8>::new_undirected();
        let a = [
            g1.add_node((3, 1)),
            g1.add_node((3, 1)),
            g1.add_node((4, 0)),
            g1.add_node((4, 0)),
        ];
        for (u, v, w) in [(0, 1, 2), (1, 2, 1), (2, 3, 1), (3, 0, 1)] {
            g1.add_edge(a[u], a[v], w);
        }

        let mut g2 = UnGraph::<(u8, u8), u8>::new_undirected();
        // Permute vertex creation order: (0,1,2,3) -> (2,3,0,1).
        let b = [
            g2.add_node((4, 0)),
            g2.add_node((4, 0)),
            g2.add_node((3, 1)),
            g2.add_node((3, 1)),
        ];
        for (u, v, w) in [(2, 3, 2), (3, 0, 1), (0, 1, 1), (1, 2, 1)] {
            g2.add_edge(b[u], b[v], w);
        }
        assert!(cores_isomorphic(&g1, &g2));

        // Negative control: same 4-cycle topology and same node-label multiset,
        // but (3, 1) nodes sit opposite across the cycle instead of adjacent.
        let mut g3 = UnGraph::<(u8, u8), u8>::new_undirected();
        let c = [
            g3.add_node((3, 1)),
            g3.add_node((4, 0)),
            g3.add_node((3, 1)),
            g3.add_node((4, 0)),
        ];
        for (u, v, w) in [(0, 1, 2), (1, 2, 1), (2, 3, 1), (3, 0, 1)] {
            g3.add_edge(c[u], c[v], w);
        }
        assert!(!cores_isomorphic(&g1, &g3));
    }
}
