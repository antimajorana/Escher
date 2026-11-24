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
    assert!(
        st.last_build.note.contains("last valid lattice shown"),
        "expected preserved-crystal notice in refusal note, got: {}",
        st.last_build.note
    );
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
    for (marker, (a, b, _)) in markers.iter().zip(st.current_kinematic_overlay().glide_lines) {
        let midpoint = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
        assert!((midpoint[0] - marker.pos[0]).hypot(midpoint[1] - marker.pos[1]) < 1e-9);
    }
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

#[test]
fn oversized_pristine_disk_and_surgical_retry_are_refused() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let info = combine_lattice_info(square);
    let mut opts = super::default_build_opts(info.r_auto_base);
    opts.auto_radius = false;
    opts.manual_radius = 200.0;
    let pristine = build_combined_crystal(square, &info, &[], &opts, false);
    assert_eq!(pristine.charge_label, "Live view too large");
    assert!(pristine.applied_radius < 200.0);

    opts.manual_radius = 2.5;
    let pins = [
        DefectSpec::dislocation([1.0, 0.0], Some([-100.0, 0.0])),
        DefectSpec::dislocation([1.0, 0.0], Some([100.0, 0.0])),
    ];
    let separated = build_combined_crystal(square, &info, &pins, &opts, false);
    assert_eq!(separated.charge_label, "Live view too large");
    assert!(separated.applied_cuts.is_empty());
}

#[test]
fn cavity_inspector_waits_for_the_current_final_crystal() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut st = CombineTabState::new(square, &backend);
    st.wait_idle(square, None);
    assert!(st.build_cavity_request(square).is_some());

    st.specs = vec![DefectSpec::dislocation([1.0, 0.0], Some([2.371, 1.413]))];
    st.request_build(square, true, false, false);
    assert!(st.build_cavity_request(square).is_none());
    st.wait_idle(square, None);
    assert!(st.last_build.preview);
    assert!(st.build_cavity_request(square).is_none());

    st.request_build(square, false, false, false);
    st.wait_idle(square, None);
    let request = st.build_cavity_request(square).expect("current final build");
    assert_eq!(request.specs, st.specs);
    assert_eq!(request.existing_crystal.unwrap().pos, st.last_build.crystal.pos);
}

#[test]
fn changing_the_scan_grid_invalidates_the_exported_summary() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut st = CombineTabState::new(square, &backend);
    st.run_compare(square);
    st.wait_idle(square, None);
    assert!(st.to_notebook_json(square).summary.is_some());

    st.auto_compare = false;
    st.set_compare_grid(2);
    let cfg = st.to_notebook_json(square);
    assert_eq!(cfg.options.grid, 2);
    assert!(cfg.summary.is_none());
}

#[test]
fn importing_an_implicit_pin_preserves_its_position_when_another_pin_is_deleted() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut st = CombineTabState::new(square, &backend);
    let mut canvas = test_canvas();
    let mut cfg = st.to_notebook_json(square);
    cfg.defects = vec![
        DefectSpec::disclination(-90.0, [0.0, 0.0]),
        DefectSpec::dislocation([1.0, 0.0], None),
    ];
    st.auto_compare = false;
    st.load_notebook_json(&cfg, &backend, &mut canvas).unwrap();
    assert!(st.delete_pin(0, square, Some(&mut canvas)));
    assert_eq!(
        st.to_notebook_json(square).defects,
        vec![DefectSpec::dislocation([1.0, 0.0], Some([3.371, 0.413]))],
    );
}

#[test]
fn desktop_import_reports_when_it_replaces_a_saved_patch_centre() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let mut st = CombineTabState::new(square, &backend);
    let mut value = serde_json::to_value(st.to_notebook_json(square)).unwrap();
    value["options"]["centre"] = serde_json::json!([0.0, 0.0]);
    st.import_notebook_json(&value.to_string(), &backend, &mut test_canvas()).unwrap();
    assert!(st.json_status.contains("saved patch centre was replaced"));
}

#[test]
fn overlapping_pin_merge_keeps_terminated_walls_distinct_from_disclinations() {
    use crate::live_combine::cuts::{has_coincident_same_kind_pins, merge_coincident_pins};

    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let info = combine_lattice_info(square);
    let apex = [0.0, 0.0];
    let wall = DefectSpec::half_wall(Some("Wb"), apex, None, None);
    let mut pins = vec![wall.clone(), DefectSpec::disclination(90.0, apex)];
    assert!(!has_coincident_same_kind_pins(&info, &pins));
    pins.push(DefectSpec::disclination(-90.0, apex));
    assert!(has_coincident_same_kind_pins(&info, &pins));
    merge_coincident_pins(&info, &mut pins);
    assert_eq!(pins, vec![wall]);
}

#[test]
fn combine_discloses_a_different_pin_order_and_measured_charge() {
    use papertilling_core::defect::surgery::patch_centre_and_radius;

    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let info = combine_lattice_info(square);
    let pins = [
        DefectSpec::disclination(90.0, [0.0, 0.0]),
        DefectSpec::disclination(-90.0, [0.5, 0.5]),
        DefectSpec::dislocation([1.0, 0.0], Some([0.371, 0.413])),
    ];
    let (_, radius) = patch_centre_and_radius(&info.lat_info, &pins, None, None).unwrap();
    let opts = CombineBuildOptions {
        auto_radius: false,
        draw_mode: CombineDrawMode::Raw,
        ..super::default_build_opts(radius)
    };
    let out = build_combined_crystal(square, &info, &pins, &opts, false);
    assert!(out.refused.is_none(), "{:?}", out.refused);
    assert!(out.charge_label.contains("|b| = 1.4142"), "{}", out.charge_label);
    assert!(
        out.charge_detail.contains("Pin-order algebraic charge: dislocation [b] = P+ orbit of [2.0, 0.0] (|b| = 2.0000)"),
        "{}",
        out.charge_detail
    );
    assert!(out.charge_detail.contains(&format!("Measured boundary charge: {}", out.charge_label)));
    assert!(out.charge_detail.contains("Equivalence uses the measured charge and core graph."));
}

#[test]
fn first_build_refusal_notice_fallback_order_and_refused_algebraic_charge() {
    use papertilling_core::defect::combinations::build_combinations_table;
    use papertilling_core::defect::combinations_types::{
        CombinationCandidate, CombinationMode, CombinationsOptions,
    };

    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let info = combine_lattice_info(square);
    let opts = CombineBuildOptions {
        draw_mode: CombineDrawMode::FastRelaxed,
        full_planar_repair: false,
        auto_radius: true,
        manual_radius: info.r_auto_base,
        core_radius: 2,
    };

    // 1. Fallback-order build ([F_{+90}, V_{-90}]) reports "fallback order" in out.note
    //    and CombinationsTable records the exact surgical build_order [1, 0].
    let fv_specs = vec![
        DefectSpec::disclination(90.0, [0.5, 0.5]),
        DefectSpec::disclination(-90.0, [0.0, 0.0]),
    ];
    let out_fv = build_combined_crystal(square, &info, &fv_specs, &opts, false);
    assert!(out_fv.refused.is_none(), "expected [F_+90, V_-90] to succeed via fallback: {:?}", out_fv.refused);
    assert!(
        out_fv.note.contains("fallback order"),
        "fallback build must disclose fallback order in note, got: {:?}",
        out_fv.note
    );

    let table = build_combinations_table(
        &square.group,
        &info.cell,
        square.frame,
        &CombinationMode::Custom {
            candidates: vec![CombinationCandidate {
                label: String::from("F+90 + V-90"),
                defect_ids: vec![String::from("D(1b, +90)"), String::from("D(1a, -90)")],
                separation: 0.5_f64.sqrt(),
                displacements: vec![[-0.5, -0.5]],
                specs: fv_specs,
            }],
        },
        &CombinationsOptions { draw_thumbnails: false, ..CombinationsOptions::default() },
        None,
    )
    .expect("custom table build");
    let entry = &table.charge_groups[0].core_groups[0].members[0];
    assert_eq!(entry.build_order, vec![1, 0], "CombinationEntry.build_order must record exact surgical order");
    assert_eq!(entry.specs[0], DefectSpec::disclination(-90.0, [0.0, 0.0]));

    // 2. Refused adjacent-corner triple F_{+90} V_{-90} V_{-90} reports D(1b, -90 deg) in charge_detail.
    let adj_triple = vec![
        DefectSpec::disclination(90.0, [0.5, 0.5]),
        DefectSpec::disclination(-90.0, [0.0, 0.0]),
        DefectSpec::disclination(-90.0, [1.0, 0.0]),
    ];
    let out_ref = build_combined_crystal(square, &info, &adj_triple, &opts, false);
    assert!(out_ref.refused.is_some(), "expected adjacent-corner F+90 V-90 V-90 to be refused");
    assert_eq!(out_ref.charge_label, "Surgery refused");
    assert!(
        out_ref.charge_detail.contains("D(1b, -90 deg)"),
        "refused adjacent-corner triple must report exact algebraic charge D(1b, -90 deg), got: {}",
        out_ref.charge_detail
    );

    // 3. First-build refusal (no prior valid crystal) appends "defect-free disk shown" and stays honest on retry.
    let mut canvas = test_canvas();
    let mut st = CombineTabState::new(square, &backend);
    st.wait_idle(square, Some(&mut canvas));
    st.last_build.crystal = crate::combine_worker::empty_crystal();
    st.last_build.refused = None;
    st.last_build.note.clear();
    st.specs = adj_triple;
    st.request_build(square, false, false, false);
    st.wait_idle(square, Some(&mut canvas));
    assert!(st.last_build.refused.is_some());
    assert!(!st.last_build.crystal.pos.is_empty(), "stand-in disk should be drawn on first-build refusal");
    assert!(
        st.last_build.note.contains("defect-free disk shown"),
        "first-build refusal must disclose defect-free disk stand-in, got: {}",
        st.last_build.note
    );

    // A second refusal while still on the stand-in disk must keep saying "defect-free disk shown".
    st.request_build(square, false, false, false);
    st.wait_idle(square, Some(&mut canvas));
    assert!(
        st.last_build.note.contains("defect-free disk shown"),
        "repeated refusal on stand-in disk must not claim last valid lattice, got: {}",
        st.last_build.note
    );
}

#[test]
fn place_and_fuse_reports_proved_abelian_charge_on_built_and_refused_configs() {
    let backend = AppBackend::new();
    let square = backend.find_named("square").unwrap();
    let info = combine_lattice_info(square);
    let opts = CombineBuildOptions {
        draw_mode: CombineDrawMode::FastRelaxed,
        full_planar_repair: false,
        auto_radius: true,
        manual_radius: info.r_auto_base,
        core_radius: 2,
    };

    // Built configuration: square default neighbour pair.
    let pair_specs = preset_neighbour_pair(&info);
    let out_built = build_combined_crystal(square, &info, &pair_specs, &opts, false);
    assert!(out_built.refused.is_none(), "default square pair must build: {:?}", out_built.refused);
    for needle in ["Abelian charge:", "(G+)^ab =", "Z/2Z x Z/4Z", "G^ab =", "(Z/2Z)^3", "[PROVED]"] {
        assert!(
            out_built.charge_detail.contains(needle),
            "built square pair charge_detail missing '{needle}': {}",
            out_built.charge_detail
        );
    }

    // Refused configuration: adjacent-corner F_{+90} V^2_{-90}.
    let adj_triple = vec![
        DefectSpec::disclination(90.0, [0.5, 0.5]),
        DefectSpec::disclination(-90.0, [0.0, 0.0]),
        DefectSpec::disclination(-90.0, [1.0, 0.0]),
    ];
    let out_ref = build_combined_crystal(square, &info, &adj_triple, &opts, false);
    assert!(out_ref.refused.is_some(), "adjacent-corner F+90 V2-90 must be refused");
    for needle in ["Abelian charge:", "(G+)^ab =", "Z/2Z x Z/4Z", "G^ab =", "(Z/2Z)^3", "[PROVED]"] {
        assert!(
            out_ref.charge_detail.contains(needle),
            "refused adjacent-corner charge_detail missing '{needle}': {}",
            out_ref.charge_detail
        );
    }
}

