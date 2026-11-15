use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

use serde::Deserialize;

use super::loci::{supergroup_loci, supergroup_loci_for_family};
use super::tuning::{
    SliderState, allowed_interval, allowed_interval_family, classify_grid, classify_grid_family,
    clearance, has_crossings, min_separation, snap_to_special, special_points,
};
use super::tuning_eval::{
    EvalFamily, cell_mat, identify_wallpaper_group, imat2_ivec, mat2_inv, mat2_vec,
};
use crate::dsymbol::{DSymbolKey, canonical_key};
use crate::types::Recipe;

const STEP: f64 = 0.002;
const PROBE: f64 = 2.0 * STEP;

fn family_dsymbol_key(fam: &EvalFamily, vals: &BTreeMap<String, f64>) -> Result<DSymbolKey, String> {
    let cell = fam.cell_at(vals);
    let a = cell_mat(&cell);
    let ainv = mat2_inv(a).ok_or_else(|| "singular cell".to_string())?;
    let f: Vec<[f64; 2]> = cell.basis.iter().map(|&p| mat2_vec(ainv, p)).collect();
    let mut darts = Vec::with_capacity(cell.bonds.len() * 2);
    let mut dmap = HashMap::new();
    for bd in &cell.bonds {
        for dt in [(bd.i, bd.j, bd.n1, bd.n2), (bd.j, bd.i, -bd.n1, -bd.n2)] {
            if dmap.insert(dt, darts.len()).is_some() {
                return Err("duplicate dart".to_string());
            }
            darts.push(dt);
        }
    }
    let rev: Vec<usize> = darts.iter().map(|&(i, j, n1, n2)| dmap[&(j, i, -n1, -n2)]).collect();
    let mut by_site = vec![Vec::new(); f.len()];
    for (k, &(i, j, n1, n2)) in darts.iter().enumerate() {
        let v = mat2_vec(a, [f[j][0] + n1 as f64 - f[i][0], f[j][1] + n2 as f64 - f[i][1]]);
        by_site[i].push((v[1].atan2(v[0]), k));
    }
    if by_site.iter().any(Vec::is_empty) {
        return Err("isolated site".to_string());
    }
    let (mut nxt, mut prv) = (vec![0; darts.len()], vec![0; darts.len()]);
    for lst in &mut by_site {
        lst.sort_by(|x, y| x.0.total_cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        for idx in 0..lst.len() {
            nxt[lst[idx].1] = lst[(idx + 1) % lst.len()].1;
            prv[lst[idx].1] = lst[(idx + lst.len() - 1) % lst.len()].1;
        }
    }
    let (mut face_of, mut face_sizes) = (vec![usize::MAX; darts.len()], Vec::new());
    for d0 in 0..darts.len() {
        if face_of[d0] != usize::MAX {
            continue;
        }
        let fid = face_sizes.len();
        let (mut d, mut len, mut wind) = (d0, 0u32, [0i32; 2]);
        while face_of[d] == usize::MAX {
            face_of[d] = fid;
            len += 1;
            wind = [wind[0] + darts[d].2, wind[1] + darts[d].3];
            d = prv[rev[d]];
        }
        if d != d0 || wind != [0, 0] {
            return Err("non-planar face walk".to_string());
        }
        face_sizes.push(len);
    }
    if f.len() as isize - (darts.len() / 2) as isize + face_sizes.len() as isize != 0 {
        return Err("Euler characteristic != 0".to_string());
    }
    let gact: Vec<(Vec<usize>, bool)> = fam
        .prim_ops
        .iter()
        .enumerate()
        .map(|(g_idx, &(mf, _))| {
            let p_dart: Result<Vec<usize>, String> = darts
                .iter()
                .map(|&(i, j, n1, n2)| {
                    let mn = imat2_ivec(mf, [n1, n2]);
                    let (sj, si) = (fam.shift[g_idx][j], fam.shift[g_idx][i]);
                    dmap.get(&(fam.perm[g_idx][i], fam.perm[g_idx][j], mn[0] + sj[0] - si[0], mn[1] + sj[1] - si[1]))
                        .copied()
                        .ok_or_else(|| "dart orbit mismatch".to_string())
                })
                .collect();
            Ok((p_dart?, mf[0][0] * mf[1][1] - mf[0][1] * mf[1][0] < 0))
        })
        .collect::<Result<_, String>>()?;
    let nch = 2 * darts.len();
    let (mut orbit, mut reps) = (vec![usize::MAX; nch], Vec::new());
    for c in 0..nch {
        if orbit[c] != usize::MAX {
            continue;
        }
        let o = reps.len();
        reps.push(c);
        let (d, s) = (c / 2, c % 2);
        let imgs: HashSet<usize> = gact.iter().map(|(perm, flip)| 2 * perm[d] + if *flip { 1 - s } else { s }).collect();
        if imgs.len() != gact.len() {
            return Err("non-free chamber action".to_string());
        }
        for x in imgs {
            orbit[x] = o;
        }
    }
    let s0: Vec<usize> = reps.iter().map(|&c| orbit[2 * rev[c / 2] + (1 - c % 2)]).collect();
    let s1: Vec<usize> = reps.iter().map(|&c| orbit[if c % 2 == 0 { 2 * nxt[c / 2] + 1 } else { 2 * prv[c / 2] }]).collect();
    let s2: Vec<usize> = reps.iter().map(|&c| orbit[c ^ 1]).collect();
    let m01: Vec<u32> = reps.iter().map(|&c| face_sizes[if c % 2 == 0 { face_of[c / 2] } else { face_of[rev[c / 2]] }]).collect();
    let m12: Vec<u32> = reps.iter().map(|&c| by_site[darts[c / 2].0].len() as u32).collect();
    canonical_key(&[s0, s1, s2], &m01, &m12).map_err(|e| e.to_string())
}

fn check_good(fam: &EvalFamily, ref_key: &DSymbolKey, vals: &BTreeMap<String, f64>) -> bool {
    let cell = fam.cell_at(vals);
    if min_separation(&cell) < 1e-6 || has_crossings(&cell) || clearance(&cell) < 1e-6 {
        return false;
    }
    family_dsymbol_key(fam, vals).as_ref() == Ok(ref_key)
}

fn p4g_bonds_recipe() -> Recipe {
    Recipe {
        group: "p4g".to_string(),
        sites: vec![[-0.098, -0.598]],
        bonds: vec![[[0.902, 0.402], [0.598, -0.098]]],
        a: 1.0,
        b: None,
        gamma: None,
    }
}

fn p1_bonds_recipe() -> Recipe {
    Recipe {
        group: "p1".to_string(),
        sites: vec![[-0.994, -0.868], [0.779, 0.46]],
        bonds: vec![
            [[0.006, 0.132], [-0.994, -0.868]],
            [[0.006, 0.132], [-0.221, -0.54]],
            [[0.006, 0.132], [-0.221, 0.46]],
            [[0.006, 0.132], [0.779, 0.46]],
        ],
        a: 1.0,
        b: Some(1.154),
        gamma: Some(124.75600000000001),
    }
}

#[test]
fn p4g_special_point_and_independent_checks() {
    let rec = p4g_bonds_recipe();
    let fam = EvalFamily::from_recipe(&rec).unwrap();
    assert_eq!(fam.site_names, vec!["t1"]);
    let sp = special_points(&rec, "t1", -1.0, 0.5, None).unwrap();
    assert_eq!(sp.len(), 1);
    assert!((sp[0].value - (-0.5)).abs() < 1e-12);
    assert_eq!(sp[0].group, "p4m");

    let mut at_sq = fam.values();
    at_sq.insert("t1".to_string(), -0.5);
    let (grp, _) = identify_wallpaper_group(&fam.cell_at(&at_sq)).unwrap();
    assert_eq!(grp, "p4m");
    assert!(special_points(&rec, "t1", -0.4, 0.5, None).unwrap().is_empty());

    // Independent route across interior, edges +- PROBE, and forbidden segment midpoints
    let ref_key = family_dsymbol_key(&fam, &fam.values()).unwrap();
    let r = allowed_interval(&rec, "t1", -0.848, 0.152, None, None).unwrap();
    assert!(r.current_ok);
    for i in 0..15 {
        let mut x = r.interval.0 + (r.interval.1 - r.interval.0) * (i as f64 / 14.0);
        if (x - (-0.5)).abs() < 1e-4 {
            x = -0.5;
        }
        let mut v = fam.values();
        v.insert("t1".to_string(), x);
        assert!(check_good(&fam, &ref_key, &v), "interior point t1={x} failed independent check");
    }
    for (edge_opt, sgn) in [(r.edges.0, -1.0), (r.edges.1, 1.0)] {
        let e = edge_opt.unwrap();
        let mut v = fam.values();
        v.insert("t1".to_string(), e + sgn * PROBE);
        assert!(!check_good(&fam, &ref_key, &v));
    }
    for seg in &r.segments {
        let mut v = fam.values();
        v.insert("t1".to_string(), 0.5 * (seg.lo + seg.hi));
        assert!(!check_good(&fam, &ref_key, &v));
    }
}

#[test]
fn cell_parameter_special_points_cm_and_oblique() {
    // 1. p1_cm_locus: project onto cm locus, put b back off-locus, find b = 1 -> cm
    let p1 = p1_bonds_recipe();
    let fam1 = EvalFamily::from_recipe(&p1).unwrap();
    let cm_loc = supergroup_loci(&p1)
        .unwrap()
        .into_iter()
        .find(|l| l.group.as_deref() == Some("cm"))
        .expect("p1 #1 must have a cm locus");
    let mut proj = cm_loc.project(&p1, None).unwrap();
    proj.insert("b".to_string(), fam1.values()["b"]);
    assert!((proj["b"] - 1.0).abs() > 0.05);
    let sp_b = special_points(&p1, "b", 0.4, 3.0, Some(&proj)).unwrap();
    assert_eq!(sp_b.len(), 1);
    assert!((sp_b[0].value - 1.0).abs() < 1e-6);
    assert_eq!(sp_b[0].group, "cm");
    let mut at_cm = proj.clone();
    at_cm.insert("b".to_string(), 1.0);
    assert_eq!(identify_wallpaper_group(&fam1.cell_at(&at_cm)).unwrap().0, "cm");
    assert!(special_points(&p1, "b", 0.4, 3.0, None).unwrap().is_empty());

    // 2. p2_oblique: b = 1.0 -> cmm, gamma = 90.0 -> pmm
    let p2 = Recipe {
        group: "p2".to_string(),
        sites: vec![[0.0, 0.0]],
        bonds: vec![[[0.0, 0.0], [-1.0, 0.0]], [[0.0, 0.0], [0.0, -1.0]]],
        a: 1.0,
        b: Some(2.5),
        gamma: Some(70.0),
    };
    let sp_p2_b = special_points(&p2, "b", 0.4, 3.0, None).unwrap();
    assert_eq!(sp_p2_b.len(), 1);
    assert!((sp_p2_b[0].value - 1.0).abs() < 1e-6 && sp_p2_b[0].group == "cmm");
    let sp_p2_g = special_points(&p2, "gamma", 45.0, 135.0, None).unwrap();
    assert_eq!(sp_p2_g.len(), 1);
    assert!((sp_p2_g[0].value - 90.0).abs() < 1e-6 && sp_p2_g[0].group == "pmm");
    assert!(special_points(&p2, "gamma", 45.0, 85.0, None).unwrap().is_empty());
}

#[test]
fn negative_controls_corrupt_value_and_extra_orbit() {
    let rec = p4g_bonds_recipe();
    let fam = EvalFamily::from_recipe(&rec).unwrap();
    // Corrupt non-planar t1 (mirrors test_corrupt_value_is_reported in test_widgets.py)
    let bad = (0..36)
        .map(|i| -0.2 + 0.35 * (i as f64 / 35.0))
        .find(|&x| {
            let mut v = fam.values();
            v.insert("t1".to_string(), x);
            has_crossings(&fam.cell_at(&v))
        })
        .expect("must find non-planar t1 in [-0.2, 0.15]");
    let mut ov = BTreeMap::new();
    ov.insert("t1".to_string(), bad);
    let r = allowed_interval(&rec, "t1", -0.824, 0.176, Some(&ov), None).unwrap();
    assert!(!r.current_ok);
    assert!(r.segments.iter().any(|s| s.lo <= bad && bad <= s.hi));

    // Extra crossing bond orbit (0, 0, 1, 1) (mirrors test_extra_crossing_bond_orbit_is_reported)
    let mut fam2 = fam.clone();
    fam2.orbits.push(fam.orbit_of((0, 0, 1, 1)));
    assert!(has_crossings(&fam2.cell_at(&fam2.values())));
    let r2 = allowed_interval_family(&fam2, "t1", -0.824, 0.176, None, None).unwrap();
    assert!(!r2.current_ok && !r2.segments.is_empty());
    let (codes, _) = classify_grid_family(&fam2, &[vec![fam2.values()["t1"]]], None);
    assert_ne!(codes[0], 0);
    let (pub_codes, _) = classify_grid(&rec, &[vec![bad]], None).unwrap();
    assert_ne!(pub_codes[0], 0);
}

#[test]
fn slider_state_clamping_snapping_and_coupled_limits() {
    let mut st4 = SliderState::new(p4g_bonds_recipe()).unwrap();
    let lim = st4.spec("t1").unwrap().limits;
    assert_eq!(st4.set_value("t1", lim.1 + 0.3).unwrap(), lim.1);
    assert!((st4.set_value("t1", -0.5 - 0.4 * STEP).unwrap() - (-0.5)).abs() < 1e-12);
    let unsnapped = -0.5 - 0.6 * STEP;
    assert!(( snap_to_special(unsnapped, STEP, &st4.spec("t1").unwrap().special, lim) - unsnapped).abs() < 1e-12);

    // Coupled parameter limits on p1 #1
    let mut st1 = SliderState::new(p1_bonds_recipe()).unwrap();
    let before: BTreeMap<String, (f64, f64)> = st1.sliders.iter().map(|s| (s.name.clone(), s.limits)).collect();
    let x1_spec = st1.spec("x1").unwrap();
    let target = (0.5 * (st1.values["x1"] + x1_spec.limits.0) * 1000.0).round() / 1000.0;
    st1.set_value("x1", target).unwrap();
    assert!(["y1", "x2", "y2"].iter().any(|&nm| st1.spec(nm).unwrap().limits != before[nm]));

    // Rigid square lattice has zero free sliders
    let sq = Recipe {
        group: "p4m".to_string(),
        sites: vec![[0.0, 0.0]],
        bonds: vec![[[0.0, 0.0], [1.0, 0.0]]],
        a: 1.0,
        b: None,
        gamma: None,
    };
    assert!(SliderState::new(sq).unwrap().sliders.is_empty());
}

#[derive(Deserialize)]
struct GateSeg {
    lo: f64,
    hi: f64,
}

#[derive(Deserialize)]
struct GateSp {
    value: f64,
    group: String,
}

#[derive(Deserialize)]
struct GateParam {
    value: f64,
    range: [f64; 2],
    interval: [f64; 2],
    edges: [Option<f64>; 2],
    segments: Vec<GateSeg>,
    special_points: Vec<GateSp>,
}

#[derive(Deserialize)]
struct GateCase {
    rule: String,
    group: String,
    recipe: Recipe,
    params: BTreeMap<String, GateParam>,
}

#[derive(Deserialize)]
struct GateCmLocus {
    projected_values: BTreeMap<String, f64>,
    b_special: Vec<GateSp>,
}

#[derive(Deserialize)]
struct GateP2Oblique {
    recipe: Recipe,
    b_special: Vec<GateSp>,
    gamma_special: Vec<GateSp>,
}

#[derive(Deserialize)]
struct GateStage2 {
    cases: Vec<GateCase>,
    p1_cm_locus: GateCmLocus,
    p2_oblique: GateP2Oblique,
}

#[test]
fn gate_stage2_matches_python_reference() {
    let Ok(gate_path) = std::env::var("PT_GATE_STAGE2") else {
        eprintln!("PT_GATE_STAGE2 not set; skipping external JSON gate test");
        return;
    };
    let raw = std::fs::read_to_string(&gate_path).expect("failed to read PT_GATE_STAGE2 file");
    let gate: GateStage2 = serde_json::from_str(&raw).expect("failed to parse PT_GATE_STAGE2 JSON");

    let t_start = Instant::now();
    let mut max_interval_err = 0.0f64;
    let mut max_edge_err = 0.0f64;
    let mut max_sp_err = 0.0f64;

    for case in &gate.cases {
        let fam = EvalFamily::from_recipe(&case.recipe).unwrap();
        let vals = fam.values();
        let loci = supergroup_loci_for_family(&fam).unwrap();
        for (pname, exp) in &case.params {
            assert!(
                (vals[pname] - exp.value).abs() < 1e-9,
                "{} {}: value {} != {}",
                case.rule,
                case.group,
                vals[pname],
                exp.value
            );
            let r = allowed_interval_family(&fam, pname, exp.range[0], exp.range[1], None, Some(401)).unwrap();
            assert!(r.current_ok);
            let err_lo = (r.interval.0 - exp.interval[0]).abs();
            let err_hi = (r.interval.1 - exp.interval[1]).abs();
            max_interval_err = max_interval_err.max(err_lo).max(err_hi);
            assert!(err_lo < 1e-4 && err_hi < 1e-4, "{} {} {}: interval {:?}", case.rule, case.group, pname, r.interval);

            for (got_e, exp_e) in [(r.edges.0, exp.edges[0]), (r.edges.1, exp.edges[1])] {
                assert_eq!(got_e.is_some(), exp_e.is_some(), "{} {} {} edge presence", case.rule, case.group, pname);
                if let (Some(ge), Some(ee)) = (got_e, exp_e) {
                    let ee_err = (ge - ee).abs();
                    max_edge_err = max_edge_err.max(ee_err);
                    assert!(ee_err < 1e-4, "{} {} {} edge {ge} vs {ee}", case.rule, case.group, pname);
                }
            }
            assert_eq!(r.segments.len(), exp.segments.len(), "{} {} {} segment count", case.rule, case.group, pname);
            for (gs, es) in r.segments.iter().zip(&exp.segments) {
                assert!((gs.lo - es.lo).abs() < 1e-4 && (gs.hi - es.hi).abs() < 1e-4);
            }

            let sp = super::tuning::special_points_with_loci(&fam, &loci, pname, exp.range[0], exp.range[1], None).unwrap();
            assert_eq!(sp.len(), exp.special_points.len(), "{} {} {} special_points", case.rule, case.group, pname);
            for (gsp, esp) in sp.iter().zip(&exp.special_points) {
                let sp_err = (gsp.value - esp.value).abs();
                max_sp_err = max_sp_err.max(sp_err);
                assert!(sp_err < 1e-6 && gsp.group == esp.group);
            }
        }
        if case.rule == "bonds" && case.group == "p4g" {
            let sp = special_points(&case.recipe, "t1", -0.848, 0.152, None).unwrap();
            assert_eq!(sp.len(), 1);
            assert!((sp[0].value - (-0.5)).abs() < 1e-6 && sp[0].group == "p4m");
        }
    }

    // Verify p1_cm_locus against gate JSON
    let p1_rec = &gate.cases.iter().find(|c| c.rule == "bonds" && c.group == "p1").unwrap().recipe;
    let cm_loc = supergroup_loci(p1_rec).unwrap().into_iter().find(|l| l.group.as_deref() == Some("cm")).unwrap();
    let mut proj = cm_loc.project(p1_rec, None).unwrap();
    proj.insert("b".to_string(), p1_rec.b.unwrap());
    for (k, &ev) in &gate.p1_cm_locus.projected_values {
        assert!((proj[k] - ev).abs() < 1e-6, "p1_cm_locus projected {k}: {} vs {ev}", proj[k]);
    }
    let sp_cm = special_points(p1_rec, "b", 0.4, 3.0, Some(&proj)).unwrap();
    assert_eq!(sp_cm.len(), gate.p1_cm_locus.b_special.len());
    assert!((sp_cm[0].value - gate.p1_cm_locus.b_special[0].value).abs() < 1e-6);
    assert_eq!(sp_cm[0].group, gate.p1_cm_locus.b_special[0].group);
    assert!(special_points(p1_rec, "b", 0.4, 3.0, None).unwrap().is_empty());

    // Verify p2_oblique against gate JSON
    let sp_p2_b = special_points(&gate.p2_oblique.recipe, "b", 0.4, 3.0, None).unwrap();
    assert_eq!(sp_p2_b.len(), gate.p2_oblique.b_special.len());
    assert!((sp_p2_b[0].value - gate.p2_oblique.b_special[0].value).abs() < 1e-6);
    assert_eq!(sp_p2_b[0].group, gate.p2_oblique.b_special[0].group);

    let sp_p2_g = special_points(&gate.p2_oblique.recipe, "gamma", 45.0, 135.0, None).unwrap();
    assert_eq!(sp_p2_g.len(), gate.p2_oblique.gamma_special.len());
    assert!((sp_p2_g[0].value - gate.p2_oblique.gamma_special[0].value).abs() < 1e-6);
    assert_eq!(sp_p2_g[0].group, gate.p2_oblique.gamma_special[0].group);

    println!(
        "# time gate_stage2 ({} cases + p1_cm_locus + p2_oblique): {:.4} s | max_err interval={:.3e} edge={:.3e} special={:.3e}",
        gate.cases.len(),
        t_start.elapsed().as_secs_f64(),
        max_interval_err,
        max_edge_err,
        max_sp_err
    );
}
