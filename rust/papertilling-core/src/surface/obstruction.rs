//! Proof (c): lattice-level local cone obstruction theorem for closed surfaces with `chi > 0`.

use std::collections::{HashMap, HashSet};

use crate::defect::catalogue::DefectCatalogue;
use crate::types::{FracOp2D, Isometry2};

use super::pieces::{DartKey, LatticeData};
use super::{SiteLabel, SurfaceError, add2, dot2, mat2_vec, norm2, rot2, sub2};

fn point_in_poly(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let (x, y) = (p[0], p[1]);
    let mut inside = false;
    let n = poly.len();
    for k in 0..n {
        let (x1, y1) = (poly[k][0], poly[k][1]);
        let (x2, y2) = (poly[(k + 1) % n][0], poly[(k + 1) % n][1]);
        if (y1 > y) != (y2 > y) && x < x1 + (y - y1) * (x2 - x1) / (y2 - y1) {
            inside = !inside;
        }
    }
    inside
}

fn containing_face_sides(lat: &LatticeData, c: [f64; 2]) -> Option<usize> {
    let min_a = norm2([lat.a[0][0], lat.a[1][0]]).min(norm2([lat.a[0][1], lat.a[1][1]])).max(1e-6);
    let box_r = ((lat.fmax + 1.0) / min_a).ceil() as i32 + 1;
    let box_r = box_r.max(3);
    for (labs, cen) in &lat.faces {
        let base_poly: Vec<[f64; 2]> = labs.iter().map(|&lab| lat.pos(lab)).collect();
        let fc = mat2_vec(lat.a_inv, sub2(c, *cen));
        let n0 = [fc[0].round() as i32, fc[1].round() as i32];
        for d1 in -box_r..=box_r {
            for d2 in -box_r..=box_r {
                let sh = mat2_vec(lat.a, [(n0[0] + d1) as f64, (n0[1] + d2) as f64]);
                let shifted: Vec<[f64; 2]> = base_poly.iter().map(|&p| add2(p, sh)).collect();
                if point_in_poly(c, &shifted) {
                    return Some(labs.len());
                }
            }
        }
    }
    None
}

fn locate_site(lat: &LatticeData, c: [f64; 2]) -> Option<SiteLabel> {
    for (i, &b) in lat.basis.iter().enumerate() {
        let f = mat2_vec(lat.a_inv, sub2(c, b));
        let r = [f[0].round(), f[1].round()];
        if (f[0] - r[0]).abs() < 1e-5 && (f[1] - r[1]).abs() < 1e-5 {
            return Some((i, r[0] as i32, r[1] as i32));
        }
    }
    None
}

fn site_cone_usable(lat: &LatticeData, c: [f64; 2], n: usize, k: usize) -> (bool, Option<usize>) {
    let Some(s0) = locate_site(lat, c) else {
        return (false, None);
    };
    let (i0, n01, n02) = s0;
    let d = lat.plane_degree[i0];
    if d * (n - k) < 2 * n || !(d * (n - k)).is_multiple_of(n) {
        return (false, Some(d));
    }
    let d_surv = (d * (n - k)) / n;
    let p0 = lat.pos(s0);
    let mut darts: Vec<DartKey> = lat.darts.iter().copied().filter(|dt| dt.0 == i0).collect();
    darts.sort_by(|&a, &b| {
        let va = sub2(lat.pos((a.1, n01 + a.2, n02 + a.3)), p0);
        let vb = sub2(lat.pos((b.1, n01 + b.2, n02 + b.3)), p0);
        va[1].atan2(va[0]).total_cmp(&vb[1].atan2(vb[0]))
    });
    let faces: Vec<Vec<SiteLabel>> = darts
        .iter()
        .map(|dt| {
            let &(f_idx, (o1, o2)) = &lat.dart_face[dt];
            let sh = (n01 - o1, n02 - o2);
            lat.faces[f_idx].0.iter().map(|&(ii, l1, l2)| (ii, l1 + sh.0, l2 + sh.1)).collect()
        })
        .collect();
    let r_mat = rot2(std::f64::consts::TAU * (k as f64) / (n as f64));
    let (mf, tf) = lat.frac(&Isometry2 { m: r_mat, t: sub2(p0, mat2_vec(r_mat, p0)) });
    let gf = FracOp2D {
        mf: [[mf[0][0].round() as i32, mf[0][1].round() as i32], [mf[1][0].round() as i32, mf[1][1].round() as i32]],
        t: tf,
    };
    let m = d / n;
    for r0 in 0..m {
        let b_end = darts[(r0 + d_surv) % d];
        let ray_dir = sub2(lat.pos((b_end.1, n01 + b_end.2, n02 + b_end.3)), p0);
        let ray_len = norm2(ray_dir).max(1e-12);
        let ray_u = [ray_dir[0] / ray_len, ray_dir[1] / ray_len];
        let on_cut_ray = |lab: SiteLabel| -> bool {
            if lab == s0 {
                return true;
            }
            let v = sub2(lat.pos(lab), p0);
            let proj = dot2(v, ray_u);
            proj > 1e-7 && norm2(sub2(v, [proj * ray_u[0], proj * ray_u[1]])) < 1e-5
        };
        let canon_v = |lab: SiteLabel| -> Result<SiteLabel, String> {
            if lab == s0 {
                Ok(s0)
            } else if on_cut_ray(lab) {
                lat.apply_frac(&gf, lab)
            } else {
                Ok(lab)
            }
        };
        let mut ok_wedge = true;
        let mut bonds_seen: HashMap<[SiteLabel; 2], [SiteLabel; 2]> = HashMap::new();
        for t in 0..d_surv {
            let cyc = &faces[(r0 + t) % d];
            let mut distinct = HashSet::with_capacity(cyc.len());
            for &u in cyc {
                let Ok(cu) = canon_v(u) else {
                    ok_wedge = false;
                    break;
                };
                distinct.insert(cu);
            }
            if !ok_wedge || distinct.len() < 3 {
                ok_wedge = false;
                break;
            }
            let lc = cyc.len();
            for idx in 0..lc {
                let (u_raw, v_raw) = (cyc[idx], cyc[(idx + 1) % lc]);
                let (Ok(ua), Ok(vb)) = (canon_v(u_raw), canon_v(v_raw)) else {
                    ok_wedge = false;
                    break;
                };
                if ua == vb {
                    ok_wedge = false;
                    break;
                }
                let plane_key = if on_cut_ray(u_raw) && on_cut_ray(v_raw) {
                    if ua <= vb { [ua, vb] } else { [vb, ua] }
                } else if u_raw <= v_raw {
                    [u_raw, v_raw]
                } else {
                    [v_raw, u_raw]
                };
                let ep = if ua <= vb { [ua, vb] } else { [vb, ua] };
                if let Some(&prev) = bonds_seen.get(&ep) {
                    if prev != plane_key {
                        ok_wedge = false;
                        break;
                    }
                } else {
                    bonds_seen.insert(ep, plane_key);
                }
            }
            if !ok_wedge {
                break;
            }
        }
        if ok_wedge {
            return (true, Some(d));
        }
    }
    (false, Some(d))
}

pub(super) fn check_local_cone_obstruction(
    lat: &LatticeData,
    cat: &DefectCatalogue,
    what: &str,
) -> Result<(), SurfaceError> {
    let mut reasons = Vec::new();
    for rec in &cat.centre_classes {
        let n = rec.order;
        if n < 2 {
            continue;
        }
        match rec.location.as_str() {
            "bond midpoint" | "on a bond" => {
                reasons.push(format!("{} ({}): folds bond into a loop", rec.label, rec.location));
            }
            "face centre" | "inside a face" => {
                let Some(d) = containing_face_sides(lat, rec.point) else {
                    return Ok(());
                };
                if (1..n).any(|k| d * (n - k) >= 3 * n && (d * (n - k)).is_multiple_of(n)) {
                    return Ok(());
                }
                reasons.push(format!("{} ({}, {d}-gon, order {n}): leaves < 3 or non-integer sides", rec.label, rec.location));
            }
            "site" => {
                let mut d_site = None;
                for k in 1..n {
                    let (ok, d_opt) = site_cone_usable(lat, rec.point, n, k);
                    d_site = d_opt;
                    if ok {
                        return Ok(());
                    }
                }
                let deg_str = d_site.map_or_else(|| "?".to_string(), |d| d.to_string());
                reasons.push(format!("{} (site, deg {deg_str}, order {n}): local 1-ring forces loop or double bond", rec.label));
            }
            _ => return Ok(()),
        }
    }
    if reasons.is_empty() {
        return Ok(());
    }
    Err(SurfaceError::impossible_lattice(format!(
        "{what} is impossible (proved for this lattice): every rotation centre fails local cone checks ({})",
        reasons.join("; ")
    )))
}
