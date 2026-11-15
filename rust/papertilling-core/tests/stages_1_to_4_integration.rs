//! End-to-end Stages 1–4 integration tests running directly against `papertilling/data/*.json`
//! (zero external environment variables or local paths required; runs in standard `cargo test`).

use std::path::PathBuf;
use std::time::Instant;

use papertilling_core::defect::DefectCatalogue;
use papertilling_core::plane::catalogue::{CatalogueRule, load_catalogue, load_named};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::groups::WALLPAPER_GROUPS;
use papertilling_core::plane::planarity::check as check_planarity;
use papertilling_core::plane::tuning::{allowed_interval, special_points};
use papertilling_core::surface::{check_surface, genus_surface, surface_table};
use papertilling_core::types::Frame2D;

fn repo_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../papertilling/data")
}

#[test]
fn end_to_end_stage1_catalogues_and_named_lattices() {
    let t0 = Instant::now();
    let dir = repo_data_dir();
    assert!(dir.is_dir(), "expected papertilling/data directory at {dir:?}");

    let expected = [
        (CatalogueRule::Points, 181usize, 145usize),
        (CatalogueRule::Sheet, 262usize, 211usize),
        (CatalogueRule::Bonds, 221usize, 177usize),
    ];
    for (rule, want_drawings, want_all) in expected {
        let cat = load_catalogue(&dir, rule).unwrap();
        let got_drawings: usize = cat.values().map(|g| g.drawings.len()).sum();
        let got_all: usize = cat.values().map(|g| g.all.len()).sum();
        assert_eq!(got_drawings, want_drawings, "{rule:?} drawings count");
        assert_eq!(got_all, want_all, "{rule:?} .all deduplicated count");
        for grp_cat in cat.values() {
            for entry in &grp_cat.drawings {
                if let Some(ref stored) = entry.stored_key {
                    assert_eq!(&entry.canonical_key, stored);
                }
                let fam = LatticeFamily::from_recipe(&entry.recipe, Some(entry.family.frame)).unwrap();
                let ds = fam.dsymbol().unwrap();
                assert_eq!(ds.canonical_key().unwrap(), entry.canonical_key);
                assert_eq!(ds.network_key().unwrap(), entry.network_key);
                assert_eq!(ds.symmetry_index().unwrap(), entry.symmetry_index);
            }
        }
    }

    let named = load_named(&dir).unwrap();
    assert_eq!(named.named.len(), 21);
    for nl in &named.named {
        let fam = LatticeFamily::from_recipe(&nl.recipe, Some(nl.frame)).unwrap();
        let cell = fam.cell();
        assert!(!cell.has_crossings(), "{} has crossings", nl.key);
        assert_eq!(cell.identify_wallpaper_group(false), nl.group, "{} group", nl.key);
        let ds = fam.dsymbol().unwrap();
        assert_eq!(ds.canonical_key().unwrap(), nl.stored_dsymbol_key, "{} key", nl.key);
        assert_eq!(ds.network_key().unwrap(), nl.stored_network_key, "{} network", nl.key);
        assert_eq!(ds.symmetry_index().unwrap(), nl.stored_symmetry_index, "{} sym_idx", nl.key);
    }
    println!("# time e2e stage1 (664 drawings + 21 named): {:.3} s", t0.elapsed().as_secs_f64());
}

#[test]
fn end_to_end_stage2_tuning_on_stored_bonds_catalogue() {
    let t0 = Instant::now();
    let dir = repo_data_dir();
    let bonds_cat = load_catalogue(&dir, CatalogueRule::Bonds).unwrap();

    // Known answer from test_widgets.py: p4g #1 (rotating squares) hits square lattice (p4m) at t1 = -0.5.
    let p4g_0 = &bonds_cat["p4g"].lattices[0];
    let ai = allowed_interval(&p4g_0.recipe, "t1", -0.848, 0.152, None, None).unwrap();
    assert!(ai.current_ok);
    assert!(ai.interval.0 < -0.74 && ai.interval.1 > -0.26);
    let sp = special_points(&p4g_0.recipe, "t1", -1.0, 0.5, None).unwrap();
    assert_eq!(sp.len(), 1);
    assert!((sp[0].value + 0.5).abs() < 1e-9 && sp[0].group == "p4m");
    println!("# time e2e stage2: {:.4} s", t0.elapsed().as_secs_f64());
}

#[test]
fn end_to_end_stage3_unified_17x8_surface_table_and_3d_handlebody() {
    let t0 = Instant::now();
    let dir = repo_data_dir();
    let pts_cat = load_catalogue(&dir, CatalogueRule::Points).unwrap();
    let named = load_named(&dir).unwrap();

    // Verify 3D handlebody geometry on honeycomb genus 2 & genus 3.
    let hc = named.get("honeycomb").unwrap();
    let hc_cell = LatticeFamily::from_recipe(&hc.recipe, Some(hc.frame)).unwrap().cell();
    for g in [2usize, 3usize] {
        let m = genus_surface(&hc_cell, g).unwrap();
        assert!(check_surface(&m).ok);
        assert_eq!(m.chi, 2 - 2 * (g as i32));
        assert_eq!(m.cones.len(), 2 * (g - 1));
        assert_eq!(m.xyz.len(), m.sites.len());
        assert_eq!(m.bond_curves_3d.len(), m.bonds.len());
        let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
        for &[x, y, z] in &m.xyz {
            assert!(x.is_finite() && y.is_finite() && z.is_finite());
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
        assert!(max_z - min_z > 0.1, "genus {g} 3D handlebody must have positive Z thickness");
    }

    // Unified 17 x 8 surface_table across all 17 wallpaper groups (#0 of points.json).
    let mut ok_count = 0usize;
    let mut imp_count = 0usize;
    let mut nf_count = 0usize;
    for &grp in &WALLPAPER_GROUPS {
        let entry = &pts_cat[grp].lattices[0];
        let cell = entry.family.cell();
        let rows = surface_table(grp, &cell);
        assert_eq!(rows.len(), 8, "{grp}: expected 8 surface rows");
        for r in &rows {
            match r.status.as_str() {
                "OK" => {
                    ok_count += 1;
                    let mesh = r.mesh.as_ref().unwrap();
                    assert!(check_surface(mesh).ok, "{grp} {}: checks failed", r.surface);
                }
                "IMPOSSIBLE" => imp_count += 1,
                "NOT FOUND" => nf_count += 1,
                other => panic!("{grp} {}: unexpected status {other}", r.surface),
            }
        }
        // Frozen §9 check: p4g RP2 is single-class NOT FOUND; pmm and cmm RP2 succeed via retry schedule.
        if grp == "p4g" {
            let rp2 = rows.iter().find(|r| r.surface == "projective plane").unwrap();
            assert_eq!(rp2.status, "NOT FOUND");
        }
        if matches!(grp, "pmm" | "cmm") {
            let rp2 = rows.iter().find(|r| r.surface == "projective plane").unwrap();
            assert_eq!(rp2.status, "OK", "{grp} RP2 must succeed via (4,6) retry");
        }
    }
    assert_eq!(ok_count + imp_count + nf_count, 136);
    println!(
        "# time e2e stage3 unified 17x8 surface_table: {:.3} s (OK={ok_count}, IMP={imp_count}, NF={nf_count})",
        t0.elapsed().as_secs_f64()
    );
}

#[test]
fn end_to_end_stage4_joint_wall_catalogue_build_and_planar_drawing() {
    let t0 = Instant::now();
    let dir = repo_data_dir();
    let pts_cat = load_catalogue(&dir, CatalogueRule::Points).unwrap();
    let named = load_named(&dir).unwrap();

    // 1. square Wc1 (Möbius seam keeps stubs) & Wd1 (glide-breaking wall opens with 0 overlaps).
    let sq = named.get("square").unwrap();
    let sq_cell = LatticeFamily::from_recipe(&sq.recipe, Some(sq.frame)).unwrap().cell();
    let sq_cat = DefectCatalogue::new(&sq.group, &sq_cell, Frame2D::default());
    let wc1 = sq_cat.build_wall("Wc1", None).unwrap();
    let (wc1_pic, wc1_info) = sq_cat.wall_picture("Wc1", None).unwrap();
    assert!(wc1_info.is_none() && wc1_pic.pos == wc1.pos);
    let sq_wd1 = sq_cat.build_wall("Wd1", None).unwrap();
    assert!(check_planarity(&sq_wd1).overlaps > 0);
    let (sq_wd1_pic, sq_wd1_info) = sq_cat.wall_picture("Wd1", None).unwrap();
    let sq_m = check_planarity(&sq_wd1_pic);
    assert_eq!(sq_m.crossings, 0);
    assert_eq!(sq_m.overlaps, 0);
    assert_eq!(sq_m.crowded, 0);
    assert_eq!(sq_wd1_info.unwrap().note, "");

    // 2. p4g #0 Wd1 (test_drawing.py F: stacked wall sites opened rigidly with 0 crowded sites).
    let p4g_0 = &pts_cat["p4g"].lattices[0];
    let p4g_cell = p4g_0.family.cell();
    let p4g_cat = DefectCatalogue::new("p4g", &p4g_cell, p4g_0.family.frame);
    let p4g_wd1 = p4g_cat.build_wall("Wd1", None).unwrap();
    assert!(check_planarity(&p4g_wd1).crowded > 0);
    let (p4g_pic, p4g_info) = p4g_cat.wall_picture("Wd1", None).unwrap();
    let p4g_info = p4g_info.unwrap();
    let p4g_m = check_planarity(&p4g_pic);
    assert_eq!(p4g_info.stage, "opened");
    assert_eq!(p4g_info.note, "");
    assert_eq!((p4g_m.crossings, p4g_m.overlaps, p4g_m.crowded), (0, 0, 0));

    // 3. p6m #3 Wd2 (test_drawing.py E: rigid crosses -> embedded 0 crossings, 0 overlaps, crowded <= 1).
    let p6m_3 = &pts_cat["p6m"].lattices[3];
    let p6m_cell = p6m_3.family.cell();
    let p6m_cat = DefectCatalogue::new("p6m", &p6m_cell, p6m_3.family.frame);
    let p6m_wd2 = p6m_cat.build_wall("Wd2", None).unwrap();
    assert!(check_planarity(&p6m_wd2).crossings > 0);
    let (p6m_pic, p6m_info) = p6m_cat.wall_picture("Wd2", None).unwrap();
    let p6m_info = p6m_info.unwrap();
    let p6m_m = check_planarity(&p6m_pic);
    assert_eq!(p6m_m.crossings, 0);
    assert_eq!(p6m_m.overlaps, 0);
    assert_eq!(p6m_m.stub_crossings, 0);
    assert!(p6m_m.crowded == 0 || (p6m_m.crowded <= 1 && p6m_info.note.contains("crowded")));

    // 4. straight_p1 Wb16 (test_drawing.py E: clean or "rigid" fallback with explicit note).
    let sp1 = named.get("straight_p1").unwrap();
    let sp1_cell = LatticeFamily::from_recipe(&sp1.recipe, Some(sp1.frame)).unwrap().cell();
    let sp1_cat = DefectCatalogue::new(&sp1.group, &sp1_cell, sp1.frame);
    let wb16 = sp1_cat.build_wall("Wb16", None).unwrap();
    let (wb16_pic, wb16_info) = sp1_cat.wall_picture("Wb16", None).unwrap();
    let wb16_info = wb16_info.unwrap();
    let wb16_m = check_planarity(&wb16_pic);
    if wb16_info.stage == "rigid" {
        assert_eq!(wb16_pic.pos, wb16.pos);
        assert!(wb16_info.note.starts_with("rigid picture"));
        assert_eq!((wb16_m.crossings, wb16_m.crowded), (42, 28));
    } else {
        assert_eq!(wb16_info.note, "");
        assert_eq!((wb16_m.crossings, wb16_m.overlaps, wb16_m.crowded), (0, 0, 0));
    }
    println!("# time e2e stage4 joint wall catalogue+build+draw: {:.3} s", t0.elapsed().as_secs_f64());
}
