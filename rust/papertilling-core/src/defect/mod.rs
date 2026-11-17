//! Geometry-agnostic holonomy classification, `DefectCatalogue`, wall classification,
//! and wall crystal builders (Stages 4–5).

pub mod catalogue;
pub mod charge;
pub mod circuit;
mod circuit_auto;
pub mod combine_api;
pub mod combine_types;
pub mod cone_builder;
pub mod cone_geometry;
pub mod core_gallery;
pub mod core_graph;
pub mod equivalence;
pub mod gauge;
mod gauge_holonomy;
pub mod gsv_elastic;
pub mod gsv_mobius;
mod gsv_model;
pub mod half_wall;
mod half_wall_cuts;
pub mod kinematics;
pub mod sheet3d;
mod sheet3d_energy;
mod sheet3d_opt;
pub(crate) mod sheet3d_topology;
pub mod single_cache;
#[cfg(test)]
mod stage5_gate_tests;
pub mod surgery;
mod sym_helper;
pub mod volterra_patch;
pub mod volterra_scan;
pub mod wall_build;
pub mod wall_endings;
pub mod walls;

use petgraph::algo::is_isomorphic_matching;
use petgraph::graph::UnGraph;
use serde::{Deserialize, Serialize};

pub use catalogue::{CentreClassEntry, DefectCatalogue, DisclinationEntry, DislocationEntry};
pub use circuit::{CircuitResult, auto_burgers_circuit, burgers_circuit, shift_circuit_basepoint};
pub use gauge::{
    FusionChannel, PlaquetteHolonomy, SpaceGroupElement, SpaceGroupExtension, WyckoffRotocentre,
    apply_gauge_transform, fusion_channels, plaquette_holonomies, ribbon_faces, twisted_coboundary,
    verify_d_r_squared, wyckoff_rotocentres,
};
pub use gsv_elastic::{
    Elastic2DOptions, Elastic2DResult, gsv_diagnostics_2d, gsv_eval_2d, gsv_hessian_2d,
    relax_elastic_2d, relax_elastic_2d_with_opts,
};
pub use half_wall::{ResolvedHalfWall, build_half_wall, resolve_half_wall_spec};
pub use wall_build::{build_interface, build_mobius_seam, build_wall};
pub use wall_endings::{WallEndingInfo, classify_wall_ending, nearest_ending_holonomy};
pub use walls::{
    DomainWallEntry, GlideBreakingEntry, MirrorLineEntry, MobiusSeamEntry, NormaliserCoset,
    NormaliserInfo, TwinWallEntry,
};

/// Crystalline defect families classified by holonomy and mirror/glide surgery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DefectKind {
    Dislocation,
    Disclination,
    DomainWall,
    TwinWall,
    MobiusSeam,
    GlideWall,
}

/// Defect core construction mode used for equivalence checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreKind {
    Volterra,
    Completion,
}

/// Abstract lifted proper symmetry group acting on bond holonomies across cuts.
pub trait HolonomyGroup {
    type Element: Clone + Eq;

    fn identity(&self) -> Self::Element;
    fn compose(&self, left: &Self::Element, right: &Self::Element) -> Self::Element;
    fn inverse(&self, elem: &Self::Element) -> Self::Element;

    fn conjugate(&self, elem: &Self::Element, by: &Self::Element) -> Self::Element {
        let by_inv = self.inverse(by);
        self.compose(&self.compose(by, elem), &by_inv)
    }
}

/// Directed bond in a surgically cut-and-reglued crystal graph, carrying accumulated holonomy `H`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurgicalBond<H> {
    pub tail: usize,
    pub head: usize,
    pub holonomy: H,
    pub bond_type: u32,
}

/// Geometry-free result of cut-and-reglue surgery on a crystal bond graph prior to spatial relaxation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BondGraphSurgery<H> {
    pub num_sites: usize,
    pub bonds: Vec<SurgicalBond<H>>,
}

impl<H> BondGraphSurgery<H> {
    pub fn new(num_sites: usize, bonds: Vec<SurgicalBond<H>>) -> Self {
        Self { num_sites, bonds }
    }
}

/// Labelled VF2 graph isomorphism check for comparing defect core graphs.
pub fn cores_isomorphic<N: PartialEq, E: PartialEq>(
    left: &UnGraph<N, E>,
    right: &UnGraph<N, E>,
) -> bool {
    is_isomorphic_matching(left, right, |a, b| a == b, |a, b| a == b)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Instant;

    use serde::Deserialize;

    use super::*;
    use crate::types::{Bond2D, BuiltCrystal, Cell2D, Frame2D};

    fn square_cell() -> Cell2D {
        Cell2D {
            a1: [1.0, 0.0],
            a2: [0.0, 1.0],
            basis: vec![[0.0, 0.0]],
            bonds: vec![
                Bond2D { i: 0, j: 0, n1: 1, n2: 0, kind: 0 },
                Bond2D { i: 0, j: 0, n1: 0, n2: 1, kind: 0 },
            ],
        }
    }

    #[test]
    fn square_known_answer_controls() {
        let cat = DefectCatalogue::new("p4m", &square_cell(), Frame2D::default());
        assert_eq!(cat.group, "p4m");
        assert!(!cat.chiral);
        assert!((cat.r0 - 3.5).abs() < 1e-12);

        // Known answers for square (p4m):
        // 2 primitive Burgers orbits with |b| <= 2: (1, 0) of size 4 and (1, 1) of size 4.
        assert_eq!(cat.dislocations.len(), 2);
        assert_eq!(cat.dislocations[0].b_frac, [1, 0]);
        assert_eq!(cat.dislocations[0].size, 4);
        assert!(cat.dislocations[0].self_inverse);
        assert_eq!(cat.dislocations[1].b_frac, [1, 1]);
        assert_eq!(cat.dislocations[1].size, 4);

        // 3 Wyckoff rotation-centre classes: 1a (site, order 4), 1b (face centre, order 4), 2c (bond midpoint, order 2).
        assert_eq!(cat.centre_classes.len(), 3);
        assert_eq!(cat.centre_classes[0].label, "1a");
        assert_eq!(cat.centre_classes[0].location, "site");
        assert_eq!(cat.centre_classes[1].label, "1b");
        assert_eq!(cat.centre_classes[1].location, "face centre");
        assert_eq!(cat.centre_classes[2].label, "2c");
        assert_eq!(cat.centre_classes[2].location, "bond midpoint");
        assert_eq!(cat.disclinations.len(), 10);

        // [N_{E(2)}(p4m) : p4m] = 2 (half-diagonal shift [0.5, 0.5] swapping 1a <-> 1b) -> 1 domain wall Wa1.
        // 4 mirror/glide G-classes (axial site, axial mid-bond, diagonal mirror, diagonal glide).
        assert_eq!(cat.normaliser.index, 2);
        assert_eq!(cat.domain_walls.len(), 1);
        assert_eq!(cat.mirrors.len(), 4);
        assert_eq!(cat.mobius_seams.len(), 4);
        assert_eq!(cat.glide_breaking.len(), 4);

        let wc1 = cat.build_wall("Wc1", None).expect("build square Wc1");
        assert!(wc1.is_non_orientable);
        assert!(wc1.adj.iter().flatten().any(|b| b.across && b.ghost_stub.is_some()));

        let wd1 = cat.build_wall("Wd1", None).expect("build square Wd1");
        assert!(!wd1.is_non_orientable);
        assert!(wd1.side.contains(&Some('A')) && wd1.side.contains(&Some('B')));
        assert!(wd1.adj.iter().flatten().any(|b| b.across));
    }

    #[derive(Deserialize)]
    struct GateCell {
        a1: [f64; 2],
        a2: [f64; 2],
        basis: Vec<[f64; 2]>,
        bonds: Vec<[i64; 5]>,
    }

    impl GateCell {
        fn to_cell2d(&self) -> Cell2D {
            Cell2D {
                a1: self.a1,
                a2: self.a2,
                basis: self.basis.clone(),
                bonds: self
                    .bonds
                    .iter()
                    .map(|b| Bond2D {
                        i: b[0] as usize,
                        j: b[1] as usize,
                        n1: b[2] as i32,
                        n2: b[3] as i32,
                        kind: b[4] as usize,
                    })
                    .collect(),
            }
        }
    }

    #[derive(Deserialize)]
    struct GateDisloc {
        id: String,
        b: [f64; 2],
        b_frac: [i32; 2],
        length: f64,
        angle: f64,
        size: usize,
        self_inverse: bool,
        inverse_id: String,
        mirror_id: Option<String>,
    }

    #[derive(Deserialize)]
    struct GateCentre {
        id: String,
        order: usize,
        gclass: String,
        label: String,
        location: String,
        mirror_of: Option<String>,
        point: [f64; 2],
        split: bool,
    }

    #[derive(Deserialize)]
    struct GateDiscl {
        id: String,
        centre_class: String,
        wyckoff: String,
        order: usize,
        angle: f64,
        location: String,
        mirror_id: Option<String>,
    }

    #[derive(Deserialize)]
    struct GateMirror {
        id: String,
        kind: String,
        glide: f64,
        period: f64,
        offset: f64,
        through_site: bool,
        size: usize,
    }

    #[derive(Deserialize)]
    struct GateDomainWall {
        id: String,
        #[serde(rename = "type")]
        kind: String,
    }

    #[derive(Deserialize)]
    struct GateTwinWall {
        id: String,
        through_site: bool,
        #[serde(rename = "sigma_in_G")]
        sigma_in_g: bool,
        network_invisible: bool,
        n_irregular: usize,
        visible: bool,
        domain_coset: Option<String>,
    }

    #[derive(Deserialize)]
    struct GateMobius {
        id: String,
        line: String,
        line_kind: String,
    }

    #[derive(Deserialize)]
    struct GateGlide {
        id: String,
        line: String,
        slip: f64,
        same_domains_as: Option<String>,
        same_as_twin: Option<String>,
    }

    #[derive(Deserialize)]
    struct GateSweepLattice {
        group: String,
        frame: Frame2D,
        cell: GateCell,
        #[serde(rename = "R0")]
        r0: f64,
        dislocations: Vec<GateDisloc>,
        centre_classes: Vec<GateCentre>,
        disclinations: Vec<GateDiscl>,
        mirrors: Vec<GateMirror>,
        domain_walls: Vec<GateDomainWall>,
        twin_walls: Vec<GateTwinWall>,
        mobius_seams: Vec<GateMobius>,
        glide_breaking: Vec<GateGlide>,
    }

    #[derive(Deserialize)]
    struct GateStage4 {
        sweep: BTreeMap<String, GateSweepLattice>,
    }

    fn assert_cross_wall_crystal(c: &BuiltCrystal, tag: &str) {
        assert!(!c.pos.is_empty(), "{tag}: empty crystal");
        assert_eq!(c.pos.len(), c.ref_pos.len(), "{tag}: ref_pos len");
        assert_eq!(c.pos.len(), c.basis_idx.len(), "{tag}: basis_idx len");
        assert_eq!(c.pos.len(), c.side.len(), "{tag}: side len");
        assert!(c.pos.iter().all(|p| p[0].is_finite() && p[1].is_finite()), "{tag}: non-finite pos");
        assert!(c.side.contains(&Some('A')), "{tag}: missing side A");
        assert!(c.side.contains(&Some('B')), "{tag}: missing side B");
        assert!(c.wall_sigma.is_some(), "{tag}: missing wall_sigma");
        assert!(!c.is_non_orientable, "{tag}: expected orientable wall");
        let n_across = c.adj.iter().flatten().filter(|b| b.across).count();
        assert!(n_across > 0, "{tag}: expected cross-wall bonds");
    }

    #[test]
    fn stage4_gate_12_lattices_and_wall_builds() {
        let Ok(gate_path) = std::env::var("PT_GATE_STAGE4") else {
            println!("PT_GATE_STAGE4 not set; skipping external JSON gate test");
            return;
        };
        let raw = std::fs::read_to_string(&gate_path).expect("read PT_GATE_STAGE4");
        let gate: GateStage4 = serde_json::from_str(&raw).expect("parse PT_GATE_STAGE4");

        let expected_tags = [
            "p4g_0",
            "p6m_3",
            "square",
            "kagome",
            "honeycomb",
            "shastry_sutherland",
            "chiral_square",
            "chiral_kagome",
            "zigzag",
            "sutherlands_brick",
            "brick_pg",
            "straight_p1",
        ];
        assert_eq!(gate.sweep.len(), expected_tags.len());

        let t0 = Instant::now();
        let mut built_cats: BTreeMap<String, DefectCatalogue> = BTreeMap::new();
        for tag in expected_tags {
            let exp = gate.sweep.get(tag).unwrap_or_else(|| panic!("missing sweep lattice {tag}"));
            let cell = exp.cell.to_cell2d();
            let cat = DefectCatalogue::new(&exp.group, &cell, exp.frame);

            assert!((cat.r0 - exp.r0).abs() < 1e-9, "{tag}: r0 {} != {}", cat.r0, exp.r0);

            assert_eq!(cat.dislocations.len(), exp.dislocations.len(), "{tag}: dislocations count");
            for (g, e) in cat.dislocations.iter().zip(&exp.dislocations) {
                assert_eq!(g.id, e.id, "{tag} disloc id");
                assert_eq!(g.b_frac, e.b_frac, "{tag} {} b_frac", e.id);
                assert!((g.b[0] - e.b[0]).abs() < 1e-6 && (g.b[1] - e.b[1]).abs() < 1e-6, "{tag} {} b", e.id);
                assert!((g.length - e.length).abs() < 1e-6, "{tag} {} length", e.id);
                assert!((g.angle - e.angle).abs() < 1e-5, "{tag} {} angle", e.id);
                assert_eq!(g.size, e.size, "{tag} {} size", e.id);
                assert_eq!(g.self_inverse, e.self_inverse, "{tag} {} self_inverse", e.id);
                assert_eq!(g.inverse_id, e.inverse_id, "{tag} {} inverse_id", e.id);
                assert_eq!(g.mirror_id, e.mirror_id, "{tag} {} mirror_id", e.id);
            }

            assert_eq!(cat.centre_classes.len(), exp.centre_classes.len(), "{tag}: centre_classes count");
            for (g, e) in cat.centre_classes.iter().zip(&exp.centre_classes) {
                assert_eq!(g.id, e.id, "{tag} centre id");
                assert_eq!(g.order, e.order, "{tag} {} order", e.id);
                assert_eq!(g.gclass, e.gclass, "{tag} {} gclass", e.id);
                assert_eq!(g.label, e.label, "{tag} {} label", e.id);
                assert_eq!(g.location, e.location, "{tag} {} location", e.id);
                assert_eq!(g.mirror_of, e.mirror_of, "{tag} {} mirror_of", e.id);
                assert!((g.point[0] - e.point[0]).abs() < 1e-6 && (g.point[1] - e.point[1]).abs() < 1e-6, "{tag} {} point", e.id);
                assert_eq!(g.split, e.split, "{tag} {} split", e.id);
            }

            assert_eq!(cat.disclinations.len(), exp.disclinations.len(), "{tag}: disclinations count");
            for (g, e) in cat.disclinations.iter().zip(&exp.disclinations) {
                assert_eq!(g.id, e.id, "{tag} discl id");
                assert_eq!(g.centre_class, e.centre_class, "{tag} {} centre_class", e.id);
                assert_eq!(g.wyckoff, e.wyckoff, "{tag} {} wyckoff", e.id);
                assert_eq!(g.order, e.order, "{tag} {} order", e.id);
                assert!((g.angle - e.angle).abs() < 1e-6, "{tag} {} angle", e.id);
                assert_eq!(g.location, e.location, "{tag} {} location", e.id);
                assert_eq!(g.mirror_id, e.mirror_id, "{tag} {} mirror_id", e.id);
            }

            assert_eq!(cat.mirrors.len(), exp.mirrors.len(), "{tag}: mirrors count");
            for (g, e) in cat.mirrors.iter().zip(&exp.mirrors) {
                assert_eq!(g.id, e.id, "{tag} mirror id");
                assert_eq!(g.kind, e.kind, "{tag} {} kind", e.id);
                assert!((g.glide - e.glide).abs() < 1e-6, "{tag} {} glide", e.id);
                assert!((g.period - e.period).abs() < 1e-6, "{tag} {} period", e.id);
                assert!((g.offset - e.offset).abs() < 1e-6, "{tag} {} offset", e.id);
                assert_eq!(g.through_site, e.through_site, "{tag} {} through_site", e.id);
                assert_eq!(g.size, e.size, "{tag} {} size", e.id);
            }

            assert_eq!(cat.domain_walls.len(), exp.domain_walls.len(), "{tag}: domain_walls count");
            for (g, e) in cat.domain_walls.iter().zip(&exp.domain_walls) {
                assert_eq!(g.id, e.id, "{tag} Wa id");
                assert_eq!(g.kind, e.kind, "{tag} {} kind", e.id);
            }

            assert_eq!(cat.twin_walls.len(), exp.twin_walls.len(), "{tag}: twin_walls count");
            for (g, e) in cat.twin_walls.iter().zip(&exp.twin_walls) {
                assert_eq!(g.id, e.id, "{tag} Wb id");
                assert_eq!(g.through_site, e.through_site, "{tag} {} through_site", e.id);
                assert_eq!(g.sigma_in_g, e.sigma_in_g, "{tag} {} sigma_in_G", e.id);
                assert_eq!(g.network_invisible, e.network_invisible, "{tag} {} network_invisible", e.id);
                assert_eq!(g.n_irregular, e.n_irregular, "{tag} {} n_irregular", e.id);
                assert_eq!(g.visible, e.visible, "{tag} {} visible", e.id);
                assert_eq!(g.domain_coset, e.domain_coset, "{tag} {} domain_coset", e.id);
            }

            assert_eq!(cat.mobius_seams.len(), exp.mobius_seams.len(), "{tag}: mobius_seams count");
            for (g, e) in cat.mobius_seams.iter().zip(&exp.mobius_seams) {
                assert_eq!(g.id, e.id, "{tag} Wc id");
                assert_eq!(g.line, e.line, "{tag} {} line", e.id);
                assert_eq!(g.line_kind, e.line_kind, "{tag} {} line_kind", e.id);
            }

            assert_eq!(cat.glide_breaking.len(), exp.glide_breaking.len(), "{tag}: glide_breaking count");
            for (g, e) in cat.glide_breaking.iter().zip(&exp.glide_breaking) {
                assert_eq!(g.id, e.id, "{tag} Wd id");
                assert_eq!(g.line, e.line, "{tag} {} line", e.id);
                assert!((g.slip - e.slip).abs() < 1e-6, "{tag} {} slip", e.id);
                assert_eq!(g.same_domains_as, e.same_domains_as, "{tag} {} same_domains_as", e.id);
                assert_eq!(g.same_as_twin, e.same_as_twin, "{tag} {} same_as_twin", e.id);
            }

            built_cats.insert(tag.to_string(), cat);
        }
        println!("# time 12-lattice DefectCatalogue sweep (Rust): {:.4} s", t0.elapsed().as_secs_f64());

        let t1 = Instant::now();
        // 1. square Wc1: Möbius seam with ghost stubs and non-orientable sheet.
        let sq_wc1 = built_cats["square"].build_wall("Wc1", None).expect("square Wc1");
        assert!(sq_wc1.is_non_orientable, "square Wc1 must be non-orientable");
        assert!(sq_wc1.wall_sigma.is_none());
        let n_ghosts = sq_wc1.adj.iter().flatten().filter(|b| b.across && b.ghost_stub.is_some() && b.seam.is_some()).count();
        assert!(n_ghosts > 0, "square Wc1 must carry seam ghost stubs");

        // 2. square Wd1, p4g_0 Wd1, p6m_3 Wd2, straight_p1 Wb16: cross-wall crystals with sides A & B.
        for (lat, wid) in [("square", "Wd1"), ("p4g_0", "Wd1"), ("p6m_3", "Wd2"), ("straight_p1", "Wb16")] {
            let c = built_cats[lat].build_wall(wid, None).unwrap_or_else(|| panic!("{lat} {wid}"));
            assert_cross_wall_crystal(&c, &format!("{lat} {wid}"));
        }

        // 3. p4g_0 Wa1: domain interface with sides A & B and cut bonds across interface.
        let p4g_wa1 = built_cats["p4g_0"].build_wall("Wa1", None).expect("p4g_0 Wa1");
        assert!(p4g_wa1.side.contains(&Some('A')) && p4g_wa1.side.contains(&Some('B')));
        assert!(p4g_wa1.wall_sigma.is_some());
        assert!(!p4g_wa1.is_non_orientable);
        assert!(p4g_wa1.adj.iter().flatten().all(|b| !b.across));
        println!("# time wall crystal builds (Rust): {:.4} s", t1.elapsed().as_secs_f64());
    }
}
