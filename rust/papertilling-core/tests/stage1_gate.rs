//! Stage 1 known-answer unit tests and external verification gate (`PT_GATE_STAGE1`, `PT_DATA_DIR`).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use serde::Deserialize;

use papertilling_core::dsymbol::{DSymbolKey, network_key, symmetry_index};
use papertilling_core::plane::catalogue::{CatalogueRule, load_catalogue, load_named};
use papertilling_core::plane::cell::{dsymbol_of_cell, reduce_basis_vectors};
use papertilling_core::plane::family::LatticeFamily;
use papertilling_core::plane::groups::{
    CrystalSystem, WALLPAPER_GROUPS, conventional_cell, crystal_system, has_rotations, is_centred,
    is_chiral, ita_number, label_point, prim_ops, proper_centre_classes, wyckoff_table,
};
use papertilling_core::types::{Frame2D, Recipe};

fn rows_to_key(rows: &[[u32; 5]]) -> DSymbolKey {
    rows.iter().map(|&[m01, m12, s0, s1, s2]| (m01, m12, s0, s1, s2)).collect()
}

fn assert_vec2_close(a: [f64; 2], b: [f64; 2], tol: f64, ctx: &str) {
    assert!(
        (a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol,
        "{ctx}: got {a:?}, expected {b:?}"
    );
}

#[test]
fn standalone_groups_wyckoff_and_proper_centres() {
    assert_eq!(WALLPAPER_GROUPS.len(), 17);
    let mut total_rows = 0usize;
    for (idx, &g) in WALLPAPER_GROUPS.iter().enumerate() {
        assert_eq!(ita_number(g).unwrap(), (idx + 1) as u8);
        let rows = wyckoff_table(g).unwrap();
        total_rows += rows.len();
    }
    // International Tables Vol. A, Part 6: 72 Wyckoff rows across the 17 plane groups.
    assert_eq!(total_rows, 72);

    assert_eq!(crystal_system("p2").unwrap(), CrystalSystem::Oblique);
    assert!(is_centred("cm").unwrap() && is_centred("cmm").unwrap());
    assert!(!is_centred("p4g").unwrap());
    assert!(is_chiral("p1").unwrap() && is_chiral("p4").unwrap() && is_chiral("p6").unwrap());
    assert!(!has_rotations("pm").unwrap() && !has_rotations("cm").unwrap());

    // Conventional defaults: oblique b=1.23*a, gamma=76°; centred rect b=1.6*a; prim rect b=1.37*a.
    let (a1_cm, a2_cm) = conventional_cell("cm", 2.0, None, None).unwrap();
    assert_vec2_close(a1_cm, [2.0, 0.0], 1e-12, "cm a1");
    assert_vec2_close(a2_cm, [0.0, 3.2], 1e-12, "cm a2");
    let (_, a2_pm) = conventional_cell("pm", 2.0, None, None).unwrap();
    assert_vec2_close(a2_pm, [0.0, 2.74], 1e-12, "pm a2");

    // Centred primitive order: cmm has 8 ops in conventional cell -> 4 ops in primitive cell in table order.
    let cmm_prim = prim_ops("cmm").unwrap();
    assert_eq!(cmm_prim.len(), 4);
    assert_eq!(cmm_prim[0].mf, [[1, 0], [0, 1]]);
    assert_eq!(cmm_prim[1].mf, [[-1, 0], [0, -1]]);

    // Wyckoff point labelling in p6m: (1/2, 0) -> 3c (2mm), (1/3, 2/3) -> 2b (3m.).
    assert_eq!(label_point("p6m", [0.5, 0.0]).unwrap().name, "3c");
    assert_eq!(label_point("p6m", [1.0 / 3.0, 2.0 / 3.0]).unwrap().name, "2b");

    // G+ proper-centre split order: (i) always holds the FIRST coordinate in ITA_TABLES.
    let p4g_cc = proper_centre_classes("p4g").unwrap();
    assert_eq!(p4g_cc.len(), 3);
    assert_eq!(p4g_cc[0].name, "2a(i)");
    assert_vec2_close(p4g_cc[0].points[0], [0.0, 0.0], 1e-12, "p4g 2a(i)");
    assert_eq!(p4g_cc[0].mirror_partner.as_deref(), Some("2a(ii)"));
    assert_eq!(p4g_cc[1].name, "2a(ii)");
    assert_vec2_close(p4g_cc[1].points[0], [0.5, 0.5], 1e-12, "p4g 2a(ii)");
}

#[test]
fn standalone_reduce_basis_ties_to_even_and_family_mirror_projection() {
    // Ties-to-even check matching NumPy np.round:
    // dot/denom = 0.5 -> round_ties_even(0.5) == 0.0 -> [0.5, 1.0] unchanged;
    // dot/denom = 1.5 -> round_ties_even(1.5) == 2.0 -> [1.5, 1.0] - 2*[1, 0] = [-0.5, 1.0].
    let (b1, b2) = reduce_basis_vectors([1.0, 0.0], [0.5, 1.0]);
    assert_vec2_close(b1, [1.0, 0.0], 1e-12, "ties-even 0.5 b1");
    assert_vec2_close(b2, [0.5, 1.0], 1e-12, "ties-even 0.5 b2");

    let (c1, c2) = reduce_basis_vectors([1.0, 0.0], [1.5, 1.0]);
    assert_vec2_close(c1, [1.0, 0.0], 1e-12, "ties-even 1.5 c1");
    assert_vec2_close(c2, [-0.5, 1.0], 1e-12, "ties-even 1.5 c2");

    // Kagome in p6m from recipe: 3 sites, 6 bonds, key [(3,4,0,0,1),(6,4,1,1,0)], group == "p6m".
    let kagome_recipe = Recipe {
        group: "p6m".to_string(),
        sites: vec![[0.5, 0.0]],
        bonds: vec![[[0.5, 0.0], [0.5, 0.5]]],
        a: 1.0,
        b: None,
        gamma: None,
    };
    let fam = LatticeFamily::from_recipe(&kagome_recipe, None).unwrap();
    assert!(fam.names().is_empty());
    let cell = fam.cell();
    assert_eq!(cell.basis.len(), 3);
    assert_eq!(cell.bonds.len(), 6);
    assert!(!cell.has_crossings());
    assert!(cell.clearance() > 0.4);
    assert!(cell.min_separation() > 0.5);
    assert_eq!(cell.identify_wallpaper_group(false), "p6m");
    let ds = fam.dsymbol().unwrap();
    assert_eq!(ds.canonical_key().unwrap(), vec![(3, 4, 0, 0, 1), (6, 4, 1, 1, 0)]);

    // Orthogonal mirror-line projection in p4g (4c mirror x, x+1/2 -> direction d=(1,1)):
    // p = (0.2, 0.7) lies on y = x + 0.5; base = p - t0 * d must satisfy base . d == 0 -> (-0.25, 0.25), t1 = 0.45.
    let p4g_mirror_recipe = Recipe {
        group: "p4g".to_string(),
        sites: vec![[0.2, 0.7]],
        bonds: vec![[[0.2, 0.7], [-0.2, 0.3]], [[0.2, 0.7], [0.3, 1.2]]],
        a: 1.0,
        b: None,
        gamma: None,
    };
    let sfam = LatticeFamily::from_recipe(
        &p4g_mirror_recipe,
        Some(Frame2D { q: [[0.0, -1.0], [1.0, 0.0]], origin: [0.1, -0.2] }),
    )
    .unwrap();
    assert_eq!(sfam.site_names, vec!["t1".to_string()]);
    let g0 = &sfam.gens[0];
    assert_eq!(g0.kind, "line");
    let dot_bd = g0.base[0] * g0.d_mat[0][0] + g0.base[1] * g0.d_mat[1][0];
    assert!(dot_bd.abs() < 1e-12, "mirror base must be orthogonal to d, got dot={dot_bd}");
    assert!((sfam.values()["t1"] - 0.45).abs() < 1e-12);

    // Verify frozen _site_fl keeps F_s(u) strictly affine across parameter shifts (even across cell walls)
    // and Frame2D invariance of dsymbol_of_cell.
    let s_shifted = sfam.set_param("t1", 0.42).unwrap();
    assert_eq!(s_shifted.cell().identify_wallpaper_group(false), "p4g");
    let key_a = sfam.dsymbol().unwrap().canonical_key().unwrap();
    let key_b = dsymbol_of_cell(&s_shifted.cell(), &s_shifted.ops_in_cell().unwrap())
        .unwrap()
        .canonical_key()
        .unwrap();
    assert_eq!(key_a, key_b);
    assert_eq!(symmetry_index(&key_a).unwrap(), 1);
    assert_eq!(network_key(&key_a).unwrap(), key_a);
}

#[derive(Deserialize)]
struct GateWyckoffRow {
    letter: char,
    multiplicity: u8,
    site_symmetry: String,
    kind: String,
    coords: Vec<String>,
    base: [f64; 2],
    #[serde(rename = "D")]
    d_mat: [Vec<f64>; 2],
}

#[derive(Deserialize)]
struct GateProperCentre {
    name: String,
    letter: char,
    order: usize,
    g_orbit_splits: bool,
    mirror_partner: Option<String>,
    points: Vec<[f64; 2]>,
}

#[derive(Deserialize)]
struct GateDrawing {
    key: Vec<[u32; 5]>,
    network: Vec<[u32; 5]>,
    symmetry_index: usize,
    rank: [usize; 4],
}

#[derive(Deserialize)]
struct GateGroupCat {
    all_count: usize,
    lattices_count: usize,
    drawings_count: usize,
    all_keys: Vec<Vec<[u32; 5]>>,
    all_networks: Vec<Vec<[u32; 5]>>,
    all_sym_idx: Vec<usize>,
    drawings: Vec<GateDrawing>,
}

#[derive(Deserialize)]
struct Stage1Gate {
    wyckoff: BTreeMap<String, Vec<GateWyckoffRow>>,
    proper_centres: BTreeMap<String, Vec<GateProperCentre>>,
    catalogues: BTreeMap<String, BTreeMap<String, GateGroupCat>>,
    named_count: usize,
}

#[test]
fn verify_stage1_gate_and_data_catalogues() {
    let Ok(gate_path_str) = std::env::var("PT_GATE_STAGE1") else {
        eprintln!("PT_GATE_STAGE1 not set; skipping external gate check");
        return;
    };
    let data_dir_str = std::env::var("PT_DATA_DIR")
        .expect("PT_DATA_DIR must be set whenever PT_GATE_STAGE1 is set");
    let t_start = Instant::now();
    let gate_text = fs::read_to_string(&gate_path_str).expect("failed to read PT_GATE_STAGE1");
    let gate: Stage1Gate = serde_json::from_str(&gate_text).expect("failed to parse PT_GATE_STAGE1");
    let data_dir = PathBuf::from(data_dir_str);

    // 1. All 17 groups' 72 Wyckoff rows and proper_centre_classes match gate_stage1.json 100%.
    let t_grp = Instant::now();
    let mut wyckoff_count = 0usize;
    for &g in &WALLPAPER_GROUPS {
        let got_w = wyckoff_table(g).unwrap();
        let exp_w = &gate.wyckoff[g];
        assert_eq!(got_w.len(), exp_w.len(), "wyckoff row count mismatch in {g}");
        for (r, e) in got_w.iter().zip(exp_w) {
            assert_eq!(r.letter, e.letter, "{g} letter");
            assert_eq!(r.multiplicity, e.multiplicity, "{g} {} mult", r.letter);
            assert_eq!(r.site_symmetry, e.site_symmetry, "{g} {} sym", r.letter);
            assert_eq!(r.kind, e.kind, "{g} {} kind", r.letter);
            assert_eq!(r.coords, e.coords, "{g} {} coords", r.letter);
            assert_vec2_close(r.base, e.base, 1e-12, &format!("{g} {} base", r.letter));
            assert_eq!(r.d_mat[0].len(), e.d_mat[0].len(), "{g} {} D[0] len", r.letter);
            assert_eq!(r.d_mat[1].len(), e.d_mat[1].len(), "{g} {} D[1] len", r.letter);
            for c in 0..r.d_mat[0].len() {
                assert!((r.d_mat[0][c] - e.d_mat[0][c]).abs() <= 1e-12);
                assert!((r.d_mat[1][c] - e.d_mat[1][c]).abs() <= 1e-12);
            }
            wyckoff_count += 1;
        }

        let got_pc = proper_centre_classes(g).unwrap();
        let exp_pc = &gate.proper_centres[g];
        assert_eq!(got_pc.len(), exp_pc.len(), "proper_centre_classes len mismatch in {g}");
        for (c, e) in got_pc.iter().zip(exp_pc) {
            assert_eq!(c.name, e.name, "{g} centre name");
            assert_eq!(c.letter, e.letter, "{g} {} letter", c.name);
            assert_eq!(c.order, e.order, "{g} {} order", c.name);
            assert_eq!(c.g_orbit_splits, e.g_orbit_splits, "{g} {} splits", c.name);
            assert_eq!(c.mirror_partner, e.mirror_partner, "{g} {} partner", c.name);
            assert_eq!(c.points.len(), e.points.len(), "{g} {} points len", c.name);
            for (&p, &q) in c.points.iter().zip(&e.points) {
                assert_vec2_close(p, q, 1e-12, &format!("{g} {} point", c.name));
            }
        }
    }
    assert_eq!(wyckoff_count, 72);
    println!("# time groups+wyckoff(72)+proper_centres: {:.3?}", t_grp.elapsed());

    // 2. All 664 stored drawings across points (181), sheet (262), bonds (221) + .all counts (145, 211, 177).
    let expected_totals = [
        (CatalogueRule::Points, 181usize, 145usize),
        (CatalogueRule::Sheet, 262usize, 211usize),
        (CatalogueRule::Bonds, 221usize, 177usize),
    ];
    let mut grand_drawings = 0usize;
    for (rule, exp_draw_total, exp_all_total) in expected_totals {
        let t_cat = Instant::now();
        let cat = load_catalogue(&data_dir, rule).unwrap();
        let exp_cat = &gate.catalogues[rule.as_str()];
        assert_eq!(cat.len(), 17);
        let (mut draw_sum, mut all_sum) = (0usize, 0usize);
        for &g in &WALLPAPER_GROUPS {
            let gc = &cat[g];
            let eg = &exp_cat[g];
            assert_eq!(gc.all.len(), eg.all_count, "{} {g} all_count", rule.as_str());
            assert_eq!(gc.lattices.len(), eg.lattices_count, "{} {g} lattices_count", rule.as_str());
            assert_eq!(gc.drawings.len(), eg.drawings_count, "{} {g} drawings_count", rule.as_str());
            draw_sum += gc.drawings.len();
            all_sum += gc.all.len();

            let got_all_keys: Vec<DSymbolKey> = gc.all.iter().map(|e| e.canonical_key.clone()).collect();
            let exp_all_keys: Vec<DSymbolKey> = eg.all_keys.iter().map(|k| rows_to_key(k)).collect();
            assert_eq!(got_all_keys, exp_all_keys, "{} {g} all_keys", rule.as_str());

            let got_all_nets: Vec<DSymbolKey> = gc.all.iter().map(|e| e.network_key.clone()).collect();
            let exp_all_nets: Vec<DSymbolKey> = eg.all_networks.iter().map(|k| rows_to_key(k)).collect();
            assert_eq!(got_all_nets, exp_all_nets, "{} {g} all_networks", rule.as_str());

            let got_all_sym: Vec<usize> = gc.all.iter().map(|e| e.symmetry_index).collect();
            assert_eq!(got_all_sym, eg.all_sym_idx, "{} {g} all_sym_idx", rule.as_str());

            for (idx, (d, ed)) in gc.drawings.iter().zip(&eg.drawings).enumerate() {
                assert_eq!(d.canonical_key, rows_to_key(&ed.key), "{} {g} drawing #{idx} key", rule.as_str());
                assert_eq!(Some(&d.canonical_key), d.stored_key.as_ref(), "{} {g} drawing #{idx} stored_key", rule.as_str());
                assert_eq!(d.network_key, rows_to_key(&ed.network), "{} {g} drawing #{idx} network", rule.as_str());
                assert_eq!(d.symmetry_index, ed.symmetry_index, "{} {g} drawing #{idx} sym_idx", rule.as_str());
                assert_eq!(d.rank, ed.rank, "{} {g} drawing #{idx} rank", rule.as_str());
            }
        }
        assert_eq!(draw_sum, exp_draw_total);
        assert_eq!(all_sum, exp_all_total);
        grand_drawings += draw_sum;
        println!(
            "# time catalogue {:6} ({draw_sum} drawings -> {all_sum} .all): {:.3?}",
            rule.as_str(),
            t_cat.elapsed()
        );
    }
    assert_eq!(grand_drawings, 664);

    // 3. All 21 named lattices in named.json rebuild from recipe + frame and match keys + wallpaper group.
    let t_named = Instant::now();
    let named = load_named(&data_dir).unwrap();
    assert_eq!(named.len(), gate.named_count);
    assert_eq!(named.len(), 21);
    for item in &named {
        assert_eq!(item.dsymbol_key, item.stored_dsymbol_key, "named {} dsymbol_key", item.key);
        assert_eq!(item.network_key, item.stored_network_key, "named {} network_key", item.key);
        assert_eq!(item.symmetry_index, item.stored_symmetry_index, "named {} sym_idx", item.key);
        assert_eq!(item.family.rank(), item.rank, "named {} rank", item.key);
        assert!(!item.cell.has_crossings(), "named {} should be crossing-free", item.key);
        let identified = item.cell.identify_wallpaper_group(false);
        assert_eq!(identified, item.group, "named {} identify_wallpaper_group", item.key);
    }
    // Loose lookup & alias checks ("Kagomé", "Snub Square" -> shastry_sutherland, "Zig-Zag" -> zigzag).
    assert_eq!(named.get("Kagom\u{00e9}").unwrap().key, "kagome");
    assert_eq!(named.get("Snub Square").unwrap().key, "shastry_sutherland");
    assert_eq!(named.get("Zig-Zag").unwrap().key, "zigzag");
    assert_eq!(named.get("Sutherland's Brick").unwrap().key, "sutherlands_brick");
    println!("# time named(21) rebuild+identify_wallpaper_group: {:.3?}", t_named.elapsed());
    println!("# time stage1_gate total: {:.3?}", t_start.elapsed());
}
