//! Mirror/glide lines (`M`) and the four symmetry-breaking wall families (`Wa`, `Wb`, `Wc`, `Wd`)
//! plus Euclidean normaliser cosets `N_{E(2)}(G) / G`, matching `papertilling/defects/walls.py`.

use serde::{Deserialize, Serialize};

use crate::types::Isometry2;

use super::catalogue::CentreClassEntry;
use super::sym_helper::{
    LatticeGroup, canon_dir, describe_isometry, dot, eigvec_max_sym, imat2_vec, mat2_det,
    mat2_vec, norm, np_allclose_vec, orbits, py_mod, reflection_across, rot90, round_dec,
    round_even,
};
use super::wall_build::{build_wall, count_network_irregular, perfect_wl_sigs};

pub(crate) fn ita_generic_index(group: &str) -> Option<usize> {
    match group {
        "p6m" | "p4m" | "cmm" | "p31m" => Some(1),
        "p6" | "p4g" | "p3m1" => Some(2),
        "p4" | "pgg" | "pmg" | "pmm" => Some(4),
        "p3" => Some(6),
        "p2" => Some(16),
        _ => None, // p1, pm, pg, cm: infinite Euclidean normaliser
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MirrorLineEntry {
    pub id: String,
    pub kind: String,
    pub glide: f64,
    pub period: f64,
    pub offset: f64,
    pub spacing: f64,
    pub through_site: bool,
    pub size: usize,
    pub direction: [f64; 2],
    pub normal: [f64; 2],
    pub point: [f64; 2],
    pub sigma: Isometry2,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormaliserCoset {
    pub id: String,
    pub kind: String,
    pub sigma: Isometry2,
    pub sigma_frac_m: [[i32; 2]; 2],
    pub sigma_frac_t: [f64; 2],
    pub what: String,
    pub check_group: bool,
    pub check_geometry: bool,
    pub centre_map: String,
    pub sites_to: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormaliserInfo {
    pub index: usize,
    pub cosets: Vec<NormaliserCoset>,
    pub continuous: usize,
    pub complete: bool,
    pub ita_generic_index: Option<usize>,
    pub note: String,
    pub holohedry_order: usize,
    pub point_group_order: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainWallEntry {
    pub id: String,
    pub coset: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub sigma: Isometry2,
    pub what: String,
    pub centre_map: String,
    pub sites_to: String,
    pub complete: bool,
    pub point: [f64; 2],
    pub direction: [f64; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwinWallEntry {
    pub id: String,
    pub through_site: bool,
    #[serde(rename = "sigma_in_G")]
    pub sigma_in_g: bool,
    pub network_invisible: bool,
    pub n_irregular: usize,
    pub visible: bool,
    pub domain_coset: Option<String>,
    pub direction: [f64; 2],
    pub direction_frac: [i32; 2],
    pub point: [f64; 2],
    pub offset: f64,
    pub spacing: f64,
    pub class_size: usize,
    pub sigma: Isometry2,
    pub(crate) members: Vec<(usize, i32, usize)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MobiusSeamEntry {
    pub id: String,
    pub line: String,
    pub line_kind: String,
    pub sigma: Isometry2,
    pub point: [f64; 2],
    pub direction: [f64; 2],
    pub length: usize,
    pub width: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlideBreakingEntry {
    pub id: String,
    pub line: String,
    pub line_kind: String,
    pub glide: f64,
    pub slip: f64,
    pub slip_over_period: f64,
    pub same_domains_as: Option<String>,
    pub same_as_twin: Option<String>,
    pub point: [f64; 2],
    pub direction: [f64; 2],
    pub sigma: Isometry2,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawLine {
    pub direction: [f64; 2],
    pub offset: f64,
    pub glide: f64,
    pub period: f64,
    pub spacing: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct TwinDir {
    pub m: [i32; 2],
    pub u: [f64; 2],
    pub n: [f64; 2],
    pub per: f64,
    pub dl: f64,
    pub ivs: Vec<(bool, f64, f64)>,
    pub mirrors: Vec<f64>,
}

pub(crate) fn line_data(grp: &LatticeGroup, u: [f64; 2]) -> (f64, f64) {
    let nu = rot90(u);
    let mut per = f64::INFINITY;
    for m0 in -6..=6 {
        for m1 in -6..=6 {
            if (m0, m1) != (0, 0) {
                let v = mat2_vec(grp.a, [m0 as f64, m1 as f64]);
                if dot(v, nu).abs() < 1e-6 {
                    per = per.min(dot(v, u).abs());
                }
            }
        }
    }
    (per, mat2_det(grp.a).abs() / per)
}

fn same_line(a: &RawLine, b: &RawLine) -> bool {
    if !np_allclose_vec(a.direction, b.direction, 1e-5, 1e-6) || (a.glide - b.glide).abs() > 1e-5 {
        return false;
    }
    let d = (a.offset - b.offset) / a.spacing;
    (d - round_even(d)).abs() < 1e-5
}

fn line_image(ln: &RawLine, h: &Isometry2) -> RawLine {
    let u2 = canon_dir(h.apply_linear(ln.direction));
    let n1 = rot90(ln.direction);
    let c = py_mod(dot(h.apply([ln.offset * n1[0], ln.offset * n1[1]]), rot90(u2)), ln.spacing);
    RawLine { direction: u2, offset: if ln.spacing - c < 1e-6 { 0.0 } else { c }, ..*ln }
}

pub(crate) fn compute_all_lines(grp: &LatticeGroup) -> Vec<RawLine> {
    let mut out: Vec<RawLine> = Vec::new();
    for g in &grp.minus {
        let u = canon_dir(eigvec_max_sym(g.m));
        let n = rot90(u);
        let (per, dl) = line_data(grp, u);
        for m0 in -2..=2 {
            for m1 in -2..=2 {
                let sh = mat2_vec(grp.a, [m0 as f64, m1 as f64]);
                let s = [g.t[0] + sh[0], g.t[1] + sh[1]];
                let mut c = py_mod(0.5 * dot(s, n), dl);
                if dl - c < 1e-6 { c = 0.0; }
                let mut tau = py_mod(dot(s, u), per);
                if tau < 1e-6 || per - tau < 1e-6 { tau = 0.0; }
                let rec = RawLine { direction: u, offset: c, glide: tau, period: per, spacing: dl };
                if !out.iter().any(|q| same_line(&rec, q)) {
                    out.push(rec);
                }
            }
        }
    }
    out
}

pub(crate) fn compute_mirrors(grp: &LatticeGroup, lines: &[RawLine]) -> Vec<MirrorLineEntry> {
    let orbs = orbits(lines, &grp.ops, line_image, same_line);
    let key_of = |q: &RawLine| (round_dec(-q.direction[0], 4), round_dec(-q.direction[1], 4), round_dec(q.offset, 4));
    let cmp_key = |a: &RawLine, b: &RawLine| {
        let (ka, kb) = (key_of(a), key_of(b));
        ka.0.total_cmp(&kb.0).then_with(|| ka.1.total_cmp(&kb.1)).then_with(|| ka.2.total_cmp(&kb.2))
    };
    let mut recs: Vec<(RawLine, usize)> = orbs
        .into_iter()
        .map(|o| (o.iter().map(|&i| lines[i]).min_by(cmp_key).expect("mirror orbit"), o.len()))
        .collect();
    recs.sort_by(|(a, _), (b, _)| (a.glide > 0.0).cmp(&(b.glide > 0.0)).then_with(|| cmp_key(a, b)));
    let mut out = Vec::with_capacity(recs.len());
    for (k, (ln, size)) in recs.into_iter().enumerate() {
        let (u, n) = (ln.direction, rot90(ln.direction));
        let p = [ln.offset * n[0], ln.offset * n[1]];
        let through_site = grp.cell.basis.iter().any(|&b| {
            let h = (dot(b, n) - ln.offset) / ln.spacing;
            (h - round_even(h)).abs() < 1e-6
        });
        out.push(MirrorLineEntry {
            id: format!("M{}", k + 1),
            kind: if ln.glide == 0.0 { "mirror" } else { "glide" }.to_string(),
            glide: ln.glide,
            period: ln.period,
            offset: ln.offset,
            spacing: ln.spacing,
            through_site,
            size,
            direction: u,
            normal: n,
            point: p,
            sigma: reflection_across(p, u, ln.glide),
        });
    }
    out
}

pub(crate) fn compute_normaliser<FLoc>(grp: &LatticeGroup, centres: &[CentreClassEntry], mut loc_fn: FLoc) -> NormaliserInfo
where
    FLoc: FnMut([f64; 2]) -> &'static str,
{
    let (index, cont, hol_len, scored) = grp.solve_normaliser_cosets();
    let mut coset_entries = Vec::with_capacity(scored.len());
    for (k, (_, kind, sig, mf, sf)) in scored.into_iter().enumerate() {
        let sig_inv = sig.inverse();
        let conj: Vec<Isometry2> = grp.ops.iter().map(|h| sig.compose(h).compose(&sig_inv)).collect();
        let cm: Vec<String> = centres
            .iter()
            .map(|c| {
                let img = sig.apply(c.point);
                let tgt = centres.iter().find(|r| r.members.iter().any(|&q| grp.same_point(img, q))).map_or("?", |r| r.label.as_str());
                format!("{}->{tgt}", c.label)
            })
            .collect();
        coset_entries.push(NormaliserCoset {
            id: format!("N{}", k + 1),
            kind: kind.to_string(),
            what: describe_isometry(sig),
            sigma: sig,
            sigma_frac_m: mf,
            sigma_frac_t: sf,
            check_group: conj.iter().all(|&h| grp.contains(h, 1e-5)),
            check_geometry: conj.iter().all(|&h| grp.l.is_symmetry(h)) && !grp.l.is_symmetry(sig),
            centre_map: cm.join(", "),
            sites_to: loc_fn(sig.apply(grp.cell.basis[0])).to_string(),
        });
    }
    let ita = ita_generic_index(grp.name);
    let note = if cont > 0 {
        format!("CONTINUOUS normaliser ({cont} free translation direction(s)): the representatives are samples; the list is NOT complete")
    } else if ita.is_none_or(|v| index == v) {
        "index equals ITA's generic value".to_string()
    } else if let Some(v) = ita {
        if index > v { format!("index {index} > ITA generic {v}: specialised metric (larger normaliser)") }
        else { format!("DISAGREEMENT: index {index} < ITA generic {v}") }
    } else {
        String::new()
    };
    NormaliserInfo {
        index,
        cosets: coset_entries,
        continuous: cont,
        complete: cont == 0,
        ita_generic_index: ita,
        note,
        holohedry_order: hol_len,
        point_group_order: grp.opsf.len(),
    }
}

pub(crate) fn coset_of(grp: &LatticeGroup, norm_info: &NormaliserInfo, sigma: Isometry2) -> Option<String> {
    if grp.contains(sigma, 1e-5) {
        return Some("G".to_string());
    }
    if let Some(r) = norm_info.cosets.iter().find(|r| grp.contains(r.sigma.inverse().compose(&sigma), 1e-5)) {
        return Some(r.id.clone());
    }
    let sig_inv = sigma.inverse();
    grp.ops.iter().all(|h| grp.contains(sigma.compose(h).compose(&sig_inv), 1e-5)).then(|| "N(G), unlisted".to_string())
}

pub(crate) fn compute_domain_walls(grp: &LatticeGroup, norm_info: &NormaliserInfo) -> Vec<DomainWallEntry> {
    let nb = grp.cell.basis.len() as f64;
    let mean_b = grp.cell.basis.iter().fold([0.0, 0.0], |acc, b| [acc[0] + b[0] / nb, acc[1] + b[1] / nb]);
    let pt = [mean_b[0] + 0.137 * grp.cell.a2[0], mean_b[1] + 0.137 * grp.cell.a2[1]];
    let la1 = norm(grp.cell.a1);
    let dir = [grp.cell.a1[0] / la1, grp.cell.a1[1] / la1];
    norm_info
        .cosets
        .iter()
        .enumerate()
        .map(|(k, r)| DomainWallEntry {
            id: format!("Wa{}", k + 1),
            coset: r.id.clone(),
            kind: r.kind.clone(),
            sigma: r.sigma,
            what: r.what.clone(),
            centre_map: r.centre_map.clone(),
            sites_to: r.sites_to.clone(),
            complete: norm_info.complete,
            point: pt,
            direction: dir,
        })
        .collect()
}

fn wall_dirs(grp: &LatticeGroup, prim_vecs: &[[i32; 2]]) -> Vec<[i32; 2]> {
    let len2 = |m: [i32; 2]| {
        let v = mat2_vec(grp.a, [m[0] as f64, m[1] as f64]);
        (v[0] * v[0] + v[1] * v[1]).sqrt()
    };
    let can = |m: [i32; 2]| {
        let v = mat2_vec(grp.a, [m[0] as f64, m[1] as f64]);
        let l = (v[0] * v[0] + v[1] * v[1]).sqrt();
        let cd = canon_dir(v);
        let f = mat2_vec(grp.ainv, [cd[0] * l, cd[1] * l]);
        [round_even(f[0]) as i32, round_even(f[1]) as i32]
    };
    let mut sorted = prim_vecs.to_vec();
    sorted.sort_by(|a, b| len2(*a).total_cmp(&len2(*b)));
    let mut vecs = Vec::new();
    for m in sorted {
        let cm = can(m);
        if !vecs.contains(&cm) {
            vecs.push(cm);
        }
    }
    let mfs: Vec<[[i32; 2]; 2]> = grp.opsf.iter().map(|op| op.mf).collect();
    orbits(&vecs, &mfs, |&m, &mf| can(imat2_vec(mf, m)), |a, b| a == b)
        .into_iter()
        .map(|o| {
            o.into_iter()
                .map(|i| vecs[i])
                .min_by(|&ma, &mb| {
                    let (ca, cb) = (canon_dir(mat2_vec(grp.a, [ma[0] as f64, ma[1] as f64])), canon_dir(mat2_vec(grp.a, [mb[0] as f64, mb[1] as f64])));
                    round_dec(len2(ma), 6).total_cmp(&round_dec(len2(mb), 6)).then_with(|| round_dec(ca[1].atan2(ca[0]), 6).total_cmp(&round_dec(cb[1].atan2(cb[0]), 6)))
                })
                .expect("wall_dir orbit")
        })
        .collect()
}

pub(crate) fn where_interval(d: &TwinDir, c_in: f64) -> usize {
    let c = py_mod(c_in, d.dl);
    for (i, &(is_site, lo, _)) in d.ivs.iter().enumerate() {
        if is_site && (c - lo).abs().min(d.dl - (c - lo).abs()) < 1e-6 {
            return i;
        }
    }
    for (i, &(is_site, lo, hi)) in d.ivs.iter().enumerate() {
        if !is_site && ((lo < c && c < hi) || (lo < c + d.dl && c + d.dl < hi)) {
            return i;
        }
    }
    panic!("twin wall offset {c_in} not placed in intervals");
}

fn map_twin_line(dirs: &[TwinDir], p0: [f64; 2], u0: [f64; 2], h: &Isometry2) -> Option<(usize, i32, usize)> {
    let mut dir_img = h.apply_linear(u0);
    if h.det() < 0.0 {
        dir_img = [-dir_img[0], -dir_img[1]];
    }
    let p = h.apply(p0);
    for (k2, d2) in dirs.iter().enumerate() {
        for o2 in [1i32, -1i32] {
            if np_allclose_vec(dir_img, [o2 as f64 * d2.u[0], o2 as f64 * d2.u[1]], 1e-5, 1e-6) {
                return Some((k2, o2, where_interval(d2, dot(p, d2.n))));
            }
        }
    }
    None
}

pub(crate) fn compute_twin_walls(
    grp: &LatticeGroup,
    all_lines: &[RawLine],
    norm_info: &NormaliserInfo,
    prim_vecs: &[[i32; 2]],
    r: usize,
) -> (Vec<TwinDir>, Vec<TwinWallEntry>) {
    let mut dirs = Vec::new();
    for m in wall_dirs(grp, prim_vecs) {
        let u = canon_dir(mat2_vec(grp.a, [m[0] as f64, m[1] as f64]));
        let n = rot90(u);
        let (per, dl) = line_data(grp, u);
        let mut raw_xs: Vec<f64> = grp
            .cell
            .basis
            .iter()
            .map(|&b| {
                let x = py_mod(dot(b, n), dl);
                if dl - x < 1e-9 { 0.0 } else { x }
            })
            .collect();
        raw_xs.sort_by(f64::total_cmp);
        let mut hs: Vec<f64> = Vec::new();
        for x in raw_xs {
            if !hs.iter().any(|&y| (x - y).abs().min(dl - (x - y).abs()) < 1e-7) {
                hs.push(x);
            }
        }
        let mut ivs: Vec<(bool, f64, f64)> = hs.iter().map(|&h| (true, h, h)).collect();
        for i in 0..hs.len() {
            ivs.push((false, hs[i], if i + 1 < hs.len() { hs[i + 1] } else { hs[0] + dl }));
        }
        let mirrors = all_lines.iter().filter(|ln| ln.glide == 0.0 && np_allclose_vec(ln.direction, u, 1e-5, 1e-6)).map(|ln| ln.offset).collect();
        dirs.push(TwinDir { m, u, n, per, dl, ivs, mirrors });
    }

    type TwinCand = (usize, i32, usize);
    let cands: Vec<TwinCand> = dirs.iter().enumerate().flat_map(|(k, d)| [1i32, -1i32].into_iter().flat_map(move |o| (0..d.ivs.len()).map(move |i| (k, o, i)))).collect();
    let orbs = orbits(
        &cands,
        &grp.ops,
        |&(k, o, i), h| {
            let (d, (is_site, lo, hi)) = (&dirs[k], dirs[k].ivs[i]);
            let c = if is_site { lo } else { 0.5 * (lo + hi) };
            map_twin_line(&dirs, [c * d.n[0], c * d.n[1]], [o as f64 * d.u[0], o as f64 * d.u[1]], h)
        },
        |im, b| im.as_ref() == Some(b),
    );

    let mut by_dir: Vec<Vec<(TwinCand, Vec<TwinCand>)>> = vec![Vec::new(); dirs.len()];
    for o in orbs {
        let mut mem: Vec<_> = o.into_iter().map(|idx| cands[idx]).collect();
        mem.sort_by_key(|&(k, orient, iv)| (k, -orient, iv));
        by_dir[mem[0].0].push((mem[0], mem));
    }

    let perfect_sigs = perfect_wl_sigs(grp, r);
    let mut out = Vec::new();
    for (k, d) in dirs.iter().enumerate() {
        let mut bucket = std::mem::take(&mut by_dir[k]);
        bucket.sort_by(|(ra, _), (rb, _)| {
            let (ia, ib) = (d.ivs[ra.2], d.ivs[rb.2]);
            (-ra.1).cmp(&(-rb.1)).then_with(|| (!ia.0).cmp(&(!ib.0))).then_with(|| ia.1.total_cmp(&ib.1))
        });
        for (rep, mem) in bucket {
            let (_, o, iv_idx) = rep;
            let (is_site, lo, hi) = d.ivs[iv_idx];
            let raw_c = if is_site {
                lo
            } else {
                d.mirrors.iter().copied().find(|&x| [x, x + d.dl].into_iter().any(|y| lo + 1e-6 < y && y < hi - 1e-6)).unwrap_or(0.5 * (lo + hi))
            };
            let c = py_mod(raw_c, d.dl);
            let (p, u) = ([c * d.n[0], c * d.n[1]], [o as f64 * d.u[0], o as f64 * d.u[1]]);
            let sig = reflection_across(p, u, 0.0);
            let crystal = build_wall(&grp.cell, p, u, 0.0, 0.5 * d.per + (2 * r + 3) as f64 * grp.lmax + grp.scale);
            let in_g = crystal.wall_sigma.is_none();
            let n_irregular = if in_g { 0 } else { count_network_irregular(grp, &crystal, p, u, d.per, r, &perfect_sigs) };
            out.push(TwinWallEntry {
                id: format!("Wb{}", out.len() + 1),
                through_site: is_site,
                sigma_in_g: in_g,
                network_invisible: n_irregular == 0,
                n_irregular,
                visible: !in_g && n_irregular > 0,
                domain_coset: coset_of(grp, norm_info, sig),
                direction: u,
                direction_frac: [d.m[0] * o, d.m[1] * o],
                point: p,
                offset: c,
                spacing: d.dl,
                class_size: mem.len(),
                sigma: sig,
                members: mem,
            });
        }
    }
    (dirs, out)
}

pub(crate) fn twin_class_of(grp: &LatticeGroup, dirs: &[TwinDir], twins: &[TwinWallEntry], point: [f64; 2], direction: [f64; 2]) -> Option<String> {
    let ld = norm(direction);
    let u = [direction[0] / ld, direction[1] / ld];
    grp.ops.iter().find_map(|h| {
        let cand = map_twin_line(dirs, point, u, h)?;
        twins.iter().find(|e| e.members.contains(&cand)).map(|e| e.id.clone())
    })
}

pub(crate) fn compute_mobius_seams(grp: &LatticeGroup, mirrors: &[MirrorLineEntry], strip_length: usize) -> Vec<MobiusSeamEntry> {
    let w = 1.6 * grp.scale;
    mirrors
        .iter()
        .enumerate()
        .map(|(k, ln)| MobiusSeamEntry {
            id: format!("Wc{}", k + 1),
            line: ln.id.clone(),
            line_kind: ln.kind.clone(),
            sigma: ln.sigma,
            point: ln.point,
            direction: ln.direction,
            length: if ln.kind == "mirror" { strip_length } else { strip_length.saturating_sub(1).max(1) },
            width: w,
        })
        .collect()
}

pub(crate) fn compute_glide_breaking(
    grp: &LatticeGroup,
    mirrors: &[MirrorLineEntry],
    norm_info: &NormaliserInfo,
    twin_dirs: &[TwinDir],
    twins: &[TwinWallEntry],
) -> Vec<GlideBreakingEntry> {
    mirrors
        .iter()
        .enumerate()
        .map(|(k, ln)| {
            let gl = if ln.kind == "glide" { 0.0 } else { 0.5 * ln.period };
            let sig = reflection_across(ln.point, ln.direction, gl);
            let f = py_mod(dot(sig.compose(&ln.sigma.inverse()).t, ln.direction), ln.period);
            let slip = f.min(ln.period - f);
            GlideBreakingEntry {
                id: format!("Wd{}", k + 1),
                line: ln.id.clone(),
                line_kind: ln.kind.clone(),
                glide: gl,
                slip,
                slip_over_period: slip / ln.period,
                same_domains_as: coset_of(grp, norm_info, sig),
                same_as_twin: (gl == 0.0).then(|| twin_class_of(grp, twin_dirs, twins, ln.point, ln.direction)).flatten(),
                point: ln.point,
                direction: ln.direction,
                sigma: sig,
            }
        })
        .collect()
}
