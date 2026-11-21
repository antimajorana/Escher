//! Regression tests for Place & fuse (Workstream B):
//! 1. Refused new pin preserves the previously built crystal and 3D sheet (same lattice only).
//! 2. Auto patch radius scales cleanly for 4+ defects (Wang's 4 inner dislocations at r=3).
//! 3. Post-relaxation 2D rigid alignment eliminates global rotation in relaxed drawings.
//! 4. Pin markers and cut rays align with the deformed crystal cores (x + y dislocations).

use papertilling_core::defect::charge::frame_faces;
use papertilling_core::defect::combine_api::{SingleDefectCache, equivalent_single_cached};
use papertilling_core::defect::surgery::DefectSpec;
use papertilling_core::types::Cell2D;

use super::*;
use crate::live_combine::{
    CombineBuildOptions, CombineDrawMode, build_combined_crystal, combine_lattice_info,
    map_drawn_to_ref, map_ref_to_drawn, preset_neighbour_pair,
};

fn test_canvas() -> Canvas2D {
    Canvas2D::new(Scene2D::PeriodicCell {
        cell: Cell2D { a1: [1.0, 0.0], a2: [0.0, 1.0], basis: vec![[0.0, 0.0]], bonds: Vec::new() },
        repeat: 2,
        show_unit_cell: false,
    })
}

fn mean_bulk_bond_rotation_deg(crystal: &papertilling_core::types::BuiltCrystal) -> f64 {
    let (mut c_sum, mut s_sum) = (0.0, 0.0);
    for (u, adj_u) in crystal.adj.iter().enumerate() {
        let pu = crystal.pos[u];
        for e in adj_u {
            if u >= e.to || e.across {
                continue;
            }
            let pure_trans = e.seam.is_none_or(|g| {
                (g.det() - 1.0).abs() < 1e-6 && ((g.m[0][0] + g.m[1][1]) - 2.0).abs() < 1e-6
            });
            if !pure_trans {
                continue;
            }
            let le = e.vec[0].hypot(e.vec[1]);
            let d = [crystal.pos[e.to][0] - pu[0], crystal.pos[e.to][1] - pu[1]];
            let ld = d[0].hypot(d[1]);
            if le < 1e-9 || ld < 1e-9 || ld > 3.0 * le {
                continue;
            }
            c_sum += e.vec[0] * d[0] + e.vec[1] * d[1];
            s_sum += e.vec[0] * d[1] - e.vec[1] * d[0];
        }
    }
    s_sum.atan2(c_sum).to_degrees()
}

#[test]
fn refused_new_pin_preserves_previously_built_crystal_and_sheet() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = test_canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.wait_idle(square, Some(&mut canvas));
    assert!(st.last_build.refused.is_none());
    let prev_sites = st.last_build.crystal.pos.len();
    let prev_pos = st.last_build.crystal.pos.clone();
    let prev_sheet_len = st.last_sheet3d.sheet.xyz.len();
    assert!(prev_sites > 50);
    assert_eq!(prev_sheet_len, prev_sites);

    // Add a new pin with an invalid/refused angle or Gauss-Bonnet violation (+360 total).
    st.specs = [
        [0.0, 0.0],
        [3.0, 0.0],
        [0.0, 3.0],
        [3.0, 3.0],
    ]
    .map(|p| DefectSpec::disclination(90.0, p))
    .to_vec();
    st.request_build(square, false, true, false);
    st.wait_idle(square, Some(&mut canvas));

    assert!(st.last_build.refused.is_some(), "expected refusal for +360° total");
    assert_eq!(
        st.last_build.crystal.pos.len(),
        prev_sites,
        "refused build must preserve previously built crystal site count"
    );
    assert_eq!(
        st.last_build.crystal.pos, prev_pos,
        "refused build must preserve previously built crystal geometry instead of blanking to pristine disk"
    );
    assert_eq!(
        st.last_sheet3d.sheet.xyz.len(),
        prev_sheet_len,
        "refused build must preserve previous 3D sheet"
    );
    assert!(matches!(st.compare_result, Some(Err(_))));
}

#[test]
fn wang_four_inner_dislocations_auto_radius_builds_and_compares_without_rim_error() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let info = combine_lattice_info(square);
    let specs = vec![
        DefectSpec::dislocation([1.0, 0.0], Some([-3.5, 0.0])),
        DefectSpec::dislocation([-1.0, 0.0], Some([3.5, 0.0])),
        DefectSpec::dislocation([0.0, 1.0], Some([0.0, 2.5])),
        DefectSpec::dislocation([0.0, -1.0], Some([0.0, -2.5])),
    ];
    let opts = CombineBuildOptions {
        draw_mode: CombineDrawMode::FastRelaxed,
        full_planar_repair: false,
        auto_radius: true,
        manual_radius: info.r_auto_base,
        core_radius: 2,
    };
    let out = build_combined_crystal(square, &info, &specs, &opts, false);
    assert!(out.refused.is_none(), "Wang 4-dislocation build refused: {:?}", out.refused);
    assert!(out.applied_radius >= 14.0, "expected auto radius >= 14.0, got {}", out.applied_radius);

    let mut cache = SingleDefectCache::default();
    let rep = equivalent_single_cached(
        &square.group,
        &info.cell,
        square.frame,
        &specs,
        &out.crystal,
        opts.core_radius,
        1,
        true,
        &mut cache,
    );
    assert!(
        rep.is_ok(),
        "equivalent_single_cached must succeed without rim error, got: {:?}",
        rep.err()
    );
}

#[test]
fn relaxed_default_square_pair_has_minimal_bulk_bond_rotation() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let info = combine_lattice_info(square);
    let specs = preset_neighbour_pair(&info);
    let opts = CombineBuildOptions {
        draw_mode: CombineDrawMode::FastRelaxed,
        full_planar_repair: false,
        auto_radius: true,
        manual_radius: info.r_auto_base,
        core_radius: 2,
    };
    let out = build_combined_crystal(square, &info, &specs, &opts, false);
    assert!(out.refused.is_none());
    let rot_deg = mean_bulk_bond_rotation_deg(&out.crystal).abs();
    assert!(
        rot_deg < 1.0,
        "relaxed default square pair bulk rotation {rot_deg:.4}° must be < 1.0°"
    );

    // Also verify sign convention directly on a crystal artificially rotated by +37°.
    let mut rotated = out.crystal.clone();
    let (s, c) = 37.0_f64.to_radians().sin_cos();
    for p in &mut rotated.pos {
        *p = [c * p[0] - s * p[1], s * p[0] + c * p[1]];
    }
    assert!((mean_bulk_bond_rotation_deg(&rotated) - 37.0).abs() < 1.0);
    crate::live_combine::align_crystal_orientation(&mut rotated);
    let after_deg = mean_bulk_bond_rotation_deg(&rotated).abs();
    assert!(after_deg < 1e-6, "re-aligned crystal rotation {after_deg:.6}° must be ~0°");
}

#[test]
fn x_and_y_dislocation_markers_align_with_drawn_pentagon_centroids() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut canvas = test_canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.specs = vec![
        DefectSpec::dislocation([1.0, 0.0], Some([-2.5, 0.0])),
        DefectSpec::dislocation([0.0, 1.0], Some([2.5, 0.0])),
    ];
    st.request_build(square, false, false, true);
    st.wait_idle(square, Some(&mut canvas));
    assert!(st.last_build.refused.is_none(), "refused: {:?}", st.last_build.refused);

    let crystal = &st.last_build.crystal;
    let faces = frame_faces(crystal);
    let k_out = faces
        .iter()
        .enumerate()
        .max_by_key(|(_, f)| f.len())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let pentagon_centroids: Vec<[f64; 2]> = faces
        .iter()
        .enumerate()
        .filter(|&(i, f)| i != k_out && f.len() == 5)
        .map(|(_, f)| {
            let inv = 1.0 / (f.len() as f64);
            let cx = f.iter().map(|&v| crystal.pos[v][0]).sum::<f64>() * inv;
            let cy = f.iter().map(|&v| crystal.pos[v][1]).sum::<f64>() * inv;
            [cx, cy]
        })
        .collect();
    assert_eq!(
        pentagon_centroids.len(),
        2,
        "expected 2 pentagons (one per dislocation), found {}",
        pentagon_centroids.len()
    );

    let markers = st.build_markers();
    assert_eq!(markers.len(), 2);
    let y_pin = markers[1].pos;
    let dist_y = pentagon_centroids
        .iter()
        .map(|c| (y_pin[0] - c[0]).hypot(y_pin[1] - c[1]))
        .fold(f64::INFINITY, f64::min);
    assert!(
        dist_y < 0.5,
        "rendered y-dislocation pin {y_pin:?} is {dist_y:.3} units from nearest pentagon centroid {pentagon_centroids:?} (want < 0.5)"
    );
    let x_pin = markers[0].pos;
    let dist_x = pentagon_centroids
        .iter()
        .map(|c| (x_pin[0] - c[0]).hypot(x_pin[1] - c[1]))
        .fold(f64::INFINITY, f64::min);
    assert!(
        dist_x < 0.5,
        "rendered x-dislocation pin {x_pin:?} is {dist_x:.3} units from nearest pentagon centroid {pentagon_centroids:?} (want < 0.5)"
    );

    // Cut rays sit on the exact same drawn core points as the markers.
    st.show_branch_cuts = true;
    let rays = st.current_cut_rays();
    assert_eq!(rays.len(), 2);
    for ray in &rays {
        let idx = ray.spec_index.expect("matched ray spec index");
        let d = (ray.origin[0] - markers[idx].pos[0]).hypot(ray.origin[1] - markers[idx].pos[1]);
        assert!(d < 1e-6, "cut ray origin {:?} vs marker {:?}: dist={d}", ray.origin, markers[idx].pos);
    }

    // Drawn <-> ref mapping round-trips smoothly for dragging.
    let probe = [1.2, -0.8];
    let round = map_drawn_to_ref(crystal, map_ref_to_drawn(crystal, probe));
    assert!(
        (round[0] - probe[0]).hypot(round[1] - probe[1]) < 0.15,
        "IDW ref->drawn->ref roundtrip drifted: {probe:?} -> {round:?}"
    );
}
