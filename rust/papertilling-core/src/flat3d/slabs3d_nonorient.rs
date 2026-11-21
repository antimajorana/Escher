//! Non-orientable flat 3D slab builders (`KleinSlab` and `MoebiusTube`):
//! exact in-plane mirror/glide symmetry discovery and seam gluing.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::f64::consts::TAU;

use super::slabs3d::{FlatSlabKind, SlabPatch3d, slab_meta};
use super::{Cell3D, dot3, frac_to_cart_3d, norm3};

/// Orientation-reversing crystal symmetry `(x, y, z) -> (M (x, y) + t, z)` with `M^2 = I`,
/// `det M = -1`, acting on basis site `s` as `perm[s]` plus lattice `shift[s]`.
pub(crate) struct InPlaneGlide {
    pub m2: [[i32; 2]; 2],
    pub along: [i32; 2],
    pub across: [i32; 2],
    pub perm: Vec<usize>,
    pub shift: Vec<[i32; 3]>,
}

fn gcd_i32(mut a: i32, mut b: i32) -> i32 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// Search the in-plane point operations `M` with entries in `[-2, 2]` (in the `a1, a2` basis, `a3`
/// fixed) and translations fixed by site 0, for a z-preserving mirror/glide symmetry of the crystal.
pub(crate) fn find_inplane_glide(cell: &Cell3D, frac_basis: &[[f64; 3]]) -> Option<InPlaneGlide> {
    let g = [0, 1, 2].map(|i| [0, 1, 2].map(|j| dot3(cell.lattice[i], cell.lattice[j])));
    let tol = 1e-5 * g[0][0].max(g[1][1]).max(g[2][2]).max(1e-12);
    let m_sites = frac_basis.len();
    let mut dart_set: HashSet<(usize, usize, i32, i32, i32)> = HashSet::new();
    for bd in &cell.bonds {
        dart_set.insert((bd.i, bd.j, bd.n1, bd.n2, bd.n3));
        dart_set.insert((bd.j, bd.i, -bd.n1, -bd.n2, -bd.n3));
    }
    let site_kind = |b: usize| cell.site_kinds.get(b).copied().unwrap_or(0);

    let mut cands = Vec::new();
    for a in -2_i32..=2 {
        for b in -2_i32..=2 {
            for c in -2_i32..=2 {
                for d in -2_i32..=2 {
                    if a * d - b * c != -1 || a * a + b * c != 1 || b * (a + d) != 0 || c * (a + d) != 0 || b * c + d * d != 1 {
                        continue;
                    }
                    let r = [[a, b, 0], [c, d, 0], [0, 0, 1]];
                    let ok_g = (0..3).all(|i| {
                        (0..3).all(|j| {
                            let val: f64 = (0..3)
                                .flat_map(|p| (0..3).map(move |q| f64::from(r[p][i]) * g[p][q] * f64::from(r[q][j])))
                                .sum();
                            (val - g[i][j]).abs() <= tol
                        })
                    });
                    if ok_g {
                        cands.push([[a, b], [c, d]]);
                    }
                }
            }
        }
    }
    cands.sort_by_key(|m| if *m == [[1, 0], [0, -1]] { 0 } else if *m == [[-1, 0], [0, 1]] { 1 } else { 2 });

    let mut prim_vecs = Vec::new();
    for x in -2_i32..=2 {
        for y in -2_i32..=2 {
            if (x != 0 || y != 0) && gcd_i32(x, y) == 1 {
                prim_vecs.push([x, y]);
            }
        }
    }
    prim_vecs.sort_by_key(|&[x, y]| x.abs() + y.abs());

    for m2 in cands {
        let Some(&along) = prim_vecs.iter().find(|&&v| [m2[0][0] * v[0] + m2[0][1] * v[1], m2[1][0] * v[0] + m2[1][1] * v[1]] == v) else {
            continue;
        };
        let Some(&across) = prim_vecs.iter().find(|&&v| [m2[0][0] * v[0] + m2[0][1] * v[1], m2[1][0] * v[0] + m2[1][1] * v[1]] == [-v[0], -v[1]]) else {
            continue;
        };
        let mf0 = [f64::from(m2[0][0]) * frac_basis[0][0] + f64::from(m2[0][1]) * frac_basis[0][1], f64::from(m2[1][0]) * frac_basis[0][0] + f64::from(m2[1][1]) * frac_basis[0][1]];
        for b0 in 0..m_sites {
            if site_kind(b0) != site_kind(0) || (frac_basis[b0][2] - frac_basis[0][2]).abs() > 1e-4 {
                continue;
            }
            let t_base = [(frac_basis[b0][0] - mf0[0]).rem_euclid(1.0), (frac_basis[b0][1] - mf0[1]).rem_euclid(1.0)];
            let (mut perm, mut shift, mut ok) = (vec![0usize; m_sites], vec![[0_i32; 3]; m_sites], true);
            for s in 0..m_sites {
                let fx = f64::from(m2[0][0]) * frac_basis[s][0] + f64::from(m2[0][1]) * frac_basis[s][1] + t_base[0];
                let fy = f64::from(m2[1][0]) * frac_basis[s][0] + f64::from(m2[1][1]) * frac_basis[s][1] + t_base[1];
                let hit = (0..m_sites).find_map(|s2| {
                    if site_kind(s2) != site_kind(s) {
                        return None;
                    }
                    let (dx, dy, dz) = (fx - frac_basis[s2][0], fy - frac_basis[s2][1], frac_basis[s][2] - frac_basis[s2][2]);
                    let (rx, ry, rz) = (dx.round(), dy.round(), dz.round());
                    ((dx - rx).abs() < 1e-4 && (dy - ry).abs() < 1e-4 && (dz - rz).abs() < 1e-4).then_some((s2, [rx as i32, ry as i32, rz as i32]))
                });
                if let Some((s2, sh)) = hit {
                    perm[s] = s2;
                    shift[s] = sh;
                } else {
                    ok = false;
                    break;
                }
            }
            if ok
                && cell.bonds.iter().all(|bd| {
                    let n1p = m2[0][0] * bd.n1 + m2[0][1] * bd.n2 + shift[bd.j][0] - shift[bd.i][0];
                    let n2p = m2[1][0] * bd.n1 + m2[1][1] * bd.n2 + shift[bd.j][1] - shift[bd.i][1];
                    let n3p = bd.n3 + shift[bd.j][2] - shift[bd.i][2];
                    dart_set.contains(&(perm[bd.i], perm[bd.j], n1p, n2p, n3p))
                })
            {
                return Some(InPlaneGlide { m2, along, across, perm, shift });
            }
        }
    }
    None
}

pub(crate) fn build_non_orientable_slab(
    cell: &Cell3D,
    frac_basis: &[[f64; 3]],
    kind: FlatSlabKind,
    repeats: [usize; 3],
    glide: &InPlaneGlide,
) -> SlabPatch3d {
    let [l_along, l_across, l_z] = repeats;
    let m = frac_basis.len();
    let v_along = [0, 1, 2].map(|k| glide.along[0] as f64 * cell.lattice[0][k] + glide.along[1] as f64 * cell.lattice[1][k]);
    let v_across = [0, 1, 2].map(|k| glide.across[0] as f64 * cell.lattice[0][k] + glide.across[1] as f64 * cell.lattice[1][k]);
    let inv_aa = 1.0 / f64::from(glide.along[0] * glide.across[1] - glide.along[1] * glide.across[0]);
    let max_span = (l_along + l_across + 2) as i32 * 2;
    let mut keys = Vec::new();
    let mut key_map = HashMap::new();
    for c1 in -max_span..=max_span {
        for c2 in -max_span..=max_span {
            let u = f64::from(glide.across[1] * c1 - glide.across[0] * c2) * inv_aa;
            let v = f64::from(-glide.along[1] * c1 + glide.along[0] * c2) * inv_aa;
            if u < -1e-9 || u >= (l_along as f64) - 1e-9 || v < -1e-9 || v >= (l_across as f64) - 1e-9 {
                continue;
            }
            for iz in 0..(l_z as i32) {
                for b in 0..m {
                    key_map.insert((b, c1, c2, iz), keys.len());
                    keys.push((b, c1, c2, iz));
                }
            }
        }
    }
    let (mid_across, mid_z) = (0.5 * (l_across as f64), 0.5 * (l_z as f64));
    let len_along = (l_along as f64) * norm3(v_along);
    let r_tube = (len_along / TAU).max(0.5);
    let u_along = [0, 1, 2].map(|k| v_along[k] / norm3(v_along).max(1e-12));
    let u_across = [0, 1, 2].map(|k| v_across[k] / norm3(v_across).max(1e-12));

    let mut flat_positions = Vec::with_capacity(keys.len());
    let mut positions = Vec::with_capacity(keys.len());
    let mut site_kinds = Vec::with_capacity(keys.len());
    for &(b, c1, c2, iz) in &keys {
        let p = frac_to_cart_3d(&cell.lattice, [frac_basis[b][0] + f64::from(c1), frac_basis[b][1] + f64::from(c2), frac_basis[b][2] + f64::from(iz) - mid_z]);
        flat_positions.push(p);
        let s_ang = (dot3(p, u_along) / len_along.max(1e-12)) * TAU;
        let w = dot3(p, u_across) - mid_across * norm3(v_across);
        let p3 = if kind == FlatSlabKind::MoebiusTube {
            let tw = 0.5 * s_ang;
            let rad = r_tube + w * tw.cos();
            [rad * s_ang.cos(), rad * s_ang.sin(), p[2] + w * tw.sin()]
        } else {
            p
        };
        positions.push(p3);
        site_kinds.push(cell.site_kinds.get(b).copied().unwrap_or(0));
    }

    // Exact image of lattice cell `(r1, r2)` of site `bc` under the glide: `M (fb + r) + t = fb[perm] + shift + M r`.
    let refl_pos_base = |bc: usize, r1: i32, r2: i32| -> (usize, i32, i32, i32) {
        let m = glide.m2;
        let sh = glide.shift[bc];
        (glide.perm[bc], m[0][0] * r1 + m[0][1] * r2 + sh[0], m[1][0] * r1 + m[1][1] * r2 + sh[1], sh[2])
    };
    let v_min = keys
        .iter()
        .map(|&(b, c1, c2, _)| {
            let (_, nr1, nr2, _) = refl_pos_base(b, c1, c2);
            f64::from(-glide.along[1] * nr1 + glide.along[0] * nr2) * inv_aa
        })
        .fold(f64::INFINITY, f64::min);
    let k_across = -((v_min + 1e-6).floor() as i32);

    let mut seen_bonds = BTreeSet::new();
    let mut bonds = Vec::new();
    let mut deg = vec![0usize; keys.len()];
    let wrap_along = |b: usize, c1: i32, c2: i32, iz: i32| -> Option<(usize, bool)> {
        let z_wrapped = if kind == FlatSlabKind::MoebiusTube {
            iz.rem_euclid(l_z as i32)
        } else if iz < 0 || iz >= l_z as i32 {
            return None;
        } else {
            iz
        };
        let u = f64::from(glide.across[1] * c1 - glide.across[0] * c2) * inv_aa;
        let q_steps = ((u + 1e-9).floor() as i32).div_euclid(l_along as i32);
        let mut bc = b;
        let (mut r1, mut r2, mut rz) = (c1 - q_steps * (l_along as i32) * glide.along[0], c2 - q_steps * (l_along as i32) * glide.along[1], z_wrapped);
        if q_steps.rem_euclid(2) != 0 {
            let (bc2, nr1, nr2, dz) = refl_pos_base(bc, r1, r2);
            bc = bc2;
            r1 = nr1 + k_across * glide.across[0];
            r2 = nr2 + k_across * glide.across[1];
            let qs_u = (((f64::from(glide.across[1] * r1 - glide.across[0] * r2) * inv_aa) + 1e-9).floor() as i32).div_euclid(l_along as i32);
            r1 -= qs_u * (l_along as i32) * glide.along[0];
            r2 -= qs_u * (l_along as i32) * glide.along[1];
            rz = if kind == FlatSlabKind::MoebiusTube { (rz + dz).rem_euclid(l_z as i32) } else { rz + dz };
            if rz < 0 || rz >= l_z as i32 {
                return None;
            }
        }
        let v_frac = f64::from(-glide.along[1] * r1 + glide.along[0] * r2) * inv_aa;
        let v_oob = v_frac < -1e-9 || v_frac >= (l_across as f64) - 1e-9;
        let (r1, r2) = if kind == FlatSlabKind::KleinSlab {
            let qs = ((v_frac + 1e-9).floor() as i32).div_euclid(l_across as i32);
            (r1 - qs * (l_across as i32) * glide.across[0], r2 - qs * (l_across as i32) * glide.across[1])
        } else if v_oob {
            return None;
        } else {
            (r1, r2)
        };
        let tid = *key_map.get(&(bc, r1, r2, rz))?;
        Some((tid, q_steps != 0 || (kind == FlatSlabKind::MoebiusTube && iz != z_wrapped) || (kind == FlatSlabKind::KleinSlab && v_oob)))
    };

    for (uid, &(b, c1, c2, iz)) in keys.iter().enumerate() {
        for bd in &cell.bonds {
            for (src, dst, d1, d2, d3) in [(bd.i, bd.j, bd.n1, bd.n2, bd.n3), (bd.j, bd.i, -bd.n1, -bd.n2, -bd.n3)] {
                if src != b {
                    continue;
                }
                if let Some((vid, seam)) = wrap_along(dst, c1 + d1, c2 + d2, iz + d3)
                    && uid != vid
                {
                    let pair = (uid.min(vid), uid.max(vid));
                    if seen_bonds.insert(pair) {
                        bonds.push((pair.0, pair.1, seam, bd.kind));
                        deg[pair.0] += 1;
                        deg[pair.1] += 1;
                    }
                }
            }
        }
    }
    let max_d = deg.iter().copied().max().unwrap_or(0);
    let boundary_sites = (0..keys.len()).filter(|&u| deg[u] < max_d).collect();
    let (cd, nd, orientable, h1, b1) = slab_meta(kind);
    let periodic_vectors = if kind == FlatSlabKind::KleinSlab {
        vec![[0, 1, 2].map(|k| v_along[k] * l_along as f64), [0, 1, 2].map(|k| v_across[k] * l_across as f64)]
    } else {
        vec![[0, 1, 2].map(|k| v_along[k] * l_along as f64), [0, 1, 2].map(|k| cell.lattice[2][k] * l_z as f64)]
    };
    SlabPatch3d {
        kind,
        status: "OK".to_string(),
        reason: format!("OK: {kind:?} with {} sites, {} bonds", positions.len(), bonds.len()),
        manifold_dim: 3,
        compact_dims: cd,
        noncompact_dims: nd,
        orientable,
        h1_label: h1.to_string(),
        b1,
        repeats,
        positions,
        flat_positions,
        bonds,
        site_kinds,
        boundary_sites,
        periodic_vectors,
    }
}
