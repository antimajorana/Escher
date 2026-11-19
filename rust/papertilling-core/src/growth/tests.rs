//! Growth tests: WP-G1 replaces this file. The WP-C0 placeholder below pins the frozen
//! API (signatures, defaults, stop mapping), not growth behaviour.

use super::{CellComplex, Growth, GrowthCaps, GrowthError, SeedKind, Stopped, grow};

/// One square tile (cell 0) with vertices 0..4 and edges 0..4; nothing else exists.
struct OneSquare;

impl CellComplex for OneSquare {
    type Cell = u32;
    type Vertex = u32;
    type Key = u32;

    fn vertices_of(&self, _c: u32) -> Vec<u32> {
        vec![0, 1, 2, 3]
    }
    fn edges_of(&self, _c: u32) -> Vec<u32> {
        vec![0, 1, 2, 3]
    }
    fn faces_of(&self, _c: u32) -> Vec<u32> {
        Vec::new()
    }
    fn cells_at(&mut self, _v: u32) -> Result<Vec<u32>, GrowthError> {
        Ok(vec![0])
    }
    fn cells_on_edge(&mut self, _e: u32) -> Result<Vec<u32>, GrowthError> {
        Ok(vec![0])
    }
    fn cells_on_face(&mut self, _f: &[u32]) -> Result<Vec<u32>, GrowthError> {
        Err(GrowthError::OutOfRange { reason: "2D complex has no faces" })
    }
}

#[test]
fn frozen_api_signatures_and_defaults() {
    // The plan's §4 signature for `grow`, checked at compile time.
    let _: fn(&mut OneSquare, Vec<u32>, GrowthCaps) -> Growth<u32> = grow::<OneSquare>;
    let x = OneSquare;
    assert!(!x.on_rim(0));
    assert_eq!(x.vertices_of(0).len(), 4);
    assert_eq!(x.edges_of(0).len(), 4);
    assert!(x.faces_of(0).is_empty());
    // D19: the vertex seed is the default; §2.1 lists five seed kinds.
    assert_eq!(SeedKind::default(), SeedKind::Vertex);
    assert_eq!(SeedKind::ALL.len(), 5);
    let caps = GrowthCaps { max_layers: 8, max_cells: 50_000 };
    assert_eq!(caps.max_cells, 50_000);
}

#[test]
fn errors_map_to_stops_at_the_layer() {
    let inf = GrowthError::Infinite { reason: "ideal vertex" };
    assert_eq!(inf.stopped_at(1), Stopped::Infinite { layer: 1, reason: "ideal vertex" });
    let gap = GrowthError::Precision { gap: 1e-9 };
    assert_eq!(gap.stopped_at(3), Stopped::Precision { layer: 3, gap: 1e-9 });
    let mut x = OneSquare;
    let err = x.cells_on_face(&[0, 1, 2]).unwrap_err();
    assert_eq!(err.stopped_at(2), Stopped::Rim { layer: 2 });
    for (s, k) in [
        (Stopped::CellCap { layer: 4, partial: 7 }, 4),
        (Stopped::Closed { layer: 2 }, 2),
        (Stopped::Rim { layer: 0 }, 0),
    ] {
        assert_eq!(s.layer(), k);
    }
}

#[test]
fn json_forms_of_seed_and_stop() {
    // The wire form the Python mirror reproduces (frozen with mod.rs).
    assert_eq!(serde_json::to_string(&SeedKind::Dirichlet).unwrap(), "\"dirichlet\"");
    let s = serde_json::to_string(&Stopped::CellCap { layer: 3, partial: 61_000 }).unwrap();
    assert_eq!(s, r#"{"kind":"cell_cap","layer":3,"partial":61000}"#);
    let back: SeedKind = serde_json::from_str("\"edge\"").unwrap();
    assert_eq!(back, SeedKind::Edge);
}
