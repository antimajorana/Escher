//! Known-answer tests: hand counts, closed-form strains, hand-derived square moduli, analytic isotropy relations, the
//! Stroh field's jump / equilibrium / energy identities, and an exact GIF round trip.

use std::f64::consts::PI;

use super::anim_gif::lzw_encode;
use super::gsv_stroh::{Tensor4, cauchy_born_tensor_opt, rotate_tensor};
use super::strain_vitek_nye::wrap_period;
use super::*;
use crate::types::{Bond2D, Cell2D, Frame2D};

fn cell(a1: [f64; 2], a2: [f64; 2], basis: Vec<[f64; 2]>, bonds: &[[i32; 4]]) -> Cell2D {
    let bonds = bonds.iter().map(|b| Bond2D { i: b[0] as usize, j: b[1] as usize, n1: b[2], n2: b[3], kind: 0 }).collect();
    Cell2D { a1, a2, basis, bonds }
}

// Primitive cells as returned by Python `lattice_data(name).cell` (catalogue names, 2026-11-19).
fn square() -> Cell2D { cell([1.0, 0.0], [0.0, 1.0], vec![[0.0, 0.0]], &[[0, 0, -1, 0], [0, 0, 0, -1]]) }
fn triangular() -> Cell2D { cell([1.0, 0.0], [0.5, 0.8660254037844386], vec![[0.0, 0.0]], &[[0, 0, -1, 0], [0, 0, -1, 1], [0, 0, 0, -1]]) }
fn honeycomb() -> Cell2D {
    cell([1.7320508075688772, 0.0], [0.8660254037844386, 1.5], vec![[0.0, 0.0], [0.0, 1.0]], &[[0, 1, 0, -1], [0, 1, 0, 0], [0, 1, 1, -1]])
}
fn kagome() -> Cell2D {
    cell([2.0, 0.0], [1.0, 1.7320508075688772], vec![[0.0, 0.0], [1.0, 0.0], [0.5, 0.8660254037844386]],
        &[[0, 1, -1, 0], [0, 1, 0, 0], [0, 2, 0, -1], [0, 2, 0, 0], [1, 2, 0, 0], [1, 2, 1, -1]])
}

/// Sites with `|n1|, |n2| <= n` and their bonds.
fn patch(c: &Cell2D, n: i32) -> (Vec<[f64; 2]>, Vec<(usize, usize)>) {
    let mut idx = std::collections::HashMap::new();
    let mut pos = Vec::new();
    for n1 in -n..=n { for n2 in -n..=n { for (s, b) in c.basis.iter().enumerate() {
        idx.insert((n1, n2, s), pos.len());
        pos.push([b[0] + f64::from(n1) * c.a1[0] + f64::from(n2) * c.a2[0], b[1] + f64::from(n1) * c.a1[1] + f64::from(n2) * c.a2[1]]);
    } } }
    let mut edges = Vec::new();
    for n1 in -n..=n { for n2 in -n..=n { for b in &c.bonds {
        if let (Some(&u), Some(&v)) = (idx.get(&(n1, n2, b.i)), idx.get(&(n1 + b.n1, n2 + b.n2, b.j))) { edges.push((u, v)); }
    } } }
    (pos, edges)
}

fn interior(pos: &[[f64; 2]], r: f64) -> impl Fn(usize) -> bool + '_ { move |i| norm(pos[i]) < r }

#[test]
fn wigner_seitz_vacancy_and_interstitial() {
    let (refp, _) = patch(&square(), 3);
    let origin = refp.iter().position(|p| norm(*p) < 1e-12).unwrap();
    let mut vac = refp.clone();
    vac.remove(origin);
    let ws = wigner_seitz_occupancy(&refp, &vac).unwrap();
    assert_eq!((ws.vacancies.clone(), ws.interstitial_hosts.len()), (vec![origin], 0));
    let mut inter = refp.clone();
    inter.push([0.3, 0.2]);
    let ws = wigner_seitz_occupancy(&refp, &inter).unwrap();
    assert_eq!((ws.vacancies.len(), ws.interstitial_hosts.clone()), (0, vec![origin]));
    assert_eq!(ws.occupancy[origin], 2);
}

#[test]
fn cna_hand_counts() {
    // triangular bond: 2 common neighbours, not bonded; square / honeycomb: none; kagome: the triangle's third vertex
    for (c, want) in [(triangular(), (2, 0, 0)), (square(), (0, 0, 0)), (honeycomb(), (0, 0, 0)), (kagome(), (1, 0, 0))] {
        let (pos, edges) = patch(&c, 4);
        let g = BondGraph::from_edges(&pos, &edges);
        let (sigs, _) = cna_2d(&g);
        let inner = interior(&pos, 2.0);
        let checked: Vec<_> = g.edges.iter().zip(&sigs).filter(|((u, v), _)| inner(*u) && inner(*v)).collect();
        assert!(!checked.is_empty());
        assert!(checked.iter().all(|(_, s)| **s == want), "{want:?}");
    }
}

#[test]
fn psi_n_pristine_and_isolated() {
    for (c, n) in [(square(), 4), (triangular(), 6), (honeycomb(), 3), (honeycomb(), 6)] {
        let (pos, edges) = patch(&c, 4);
        let g = BondGraph::from_edges(&pos, &edges);
        let psi = psi_n(&g, n);
        let inner = interior(&pos, 3.0);
        for i in (0..pos.len()).filter(|&i| inner(i)) {
            assert!((psi[i][0].hypot(psi[i][1]) - 1.0).abs() < 1e-12, "n = {n}");
        }
    }
    let g = BondGraph::from_edges(&[[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], &[(0, 1)]);
    assert_eq!(psi_n(&g, 6)[2], [0.0, 0.0]);
}

#[test]
fn strain_of_affine_maps() {
    let (refp, edges) = patch(&triangular(), 2);
    let gam = 0.1;
    let sheared: Vec<[f64; 2]> = refp.iter().map(|p| [p[0] + gam * p[1] + 0.3, p[1] - 0.2]).collect();
    for s in local_deformation_and_strain(&refp, &sheared, &edges).unwrap() {
        let (e, g) = (s.eps.unwrap(), s.green.unwrap());
        let want_e = [[0.0, gam / 2.0], [gam / 2.0, 0.0]];
        let want_g = [[0.0, gam / 2.0], [gam / 2.0, gam * gam / 2.0]];
        for r in 0..2 { for c in 0..2 {
            assert!((e[r][c] - want_e[r][c]).abs() < 1e-12 && (g[r][c] - want_g[r][c]).abs() < 1e-12);
        } }
    }
    let rotd: Vec<[f64; 2]> = refp.iter().map(|&p| rot(p, PI / 6.0)).collect();
    for s in local_deformation_and_strain(&refp, &rotd, &edges).unwrap() {
        let (e, g) = (s.eps.unwrap(), s.green.unwrap());
        assert!((e[0][0] - ((PI / 6.0).cos() - 1.0)).abs() < 1e-12 && e[0][1].abs() < 1e-12);
        assert!(g.iter().flatten().all(|x| x.abs() < 1e-12));
    }
}

#[test]
fn vitek_wrapping() {
    assert_eq!(wrap_period(0.5, 1.0), 0.5);
    assert_eq!(wrap_period(-0.5, 1.0), 0.5);
    assert!((wrap_period(0.7, 1.0) + 0.3).abs() < 1e-15);
    let refp = vec![[0.0, 0.0], [1.0, 0.0]];
    let cur = vec![[0.2, 0.1], [1.2 + 0.7, 0.1]];
    let v = vitek_differential(&refp, &cur, &[(0, 1)], [1.0, 0.0]).unwrap();
    assert!((v[0].along_b - 0.7).abs() < 1e-12 && (v[0].wrapped + 0.3).abs() < 1e-12);
}

fn close4(a: &Tensor4, b: &[[f64; 4]; 4], tol: f64) -> bool {
    (0..16).all(|k| (a[k / 8][(k / 4) % 2][(k / 2) % 2][k % 2] - b[k / 4][k % 4]).abs() < tol)
}

#[test]
fn square_moduli_by_hand() {
    // stretch rows (1,0,0,0), (0,0,0,1); six angle pairs with c = 1/2 -> C1111 = C2222 = 2, C1212 = C1221 = 1, C1122 = 0
    let c = cauchy_born_tensor(&square());
    assert!(close4(&c, &[[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 2.0]], 1e-12));
    let b = barnett_lothe_tensor(&c, 128);
    assert!((b[0][0] - 1.0).abs() < 1e-12 && (b[1][1] - 1.0).abs() < 1e-12 && b[0][1].abs() < 1e-12);
    assert!((dipole_prelog_coeff(&c, [1.0, 0.0], 128) - 1.0 / (2.0 * PI)).abs() < 1e-12);
    assert!((acoustic_tensor(&c, 0.0)[0][0] - 2.0).abs() < 1e-12);
}

#[test]
fn moduli_by_hand_and_isotropy() {
    // Hand sums, unit bonds, angle rows with 2c = 1: C_iajb = (sum_bonds d_i d_a d_j d_b + sum_sites sum_{p<q} R_ia R_jb)
    // / area with R = (d_p d_q^T + d_q d_p^T) / 2; sum_{p<q} (x_p x_q)^2 = ((sum x^2)^2 - sum x^4) / 2.
    // Triangular (area sqrt3/2, bonds 0/60/120 deg, 6 stars): C1111 = (9/8 + 27/8) / (sqrt3/2) = 3 sqrt3,
    //   C1212 = (3/8 + 15/8) / (sqrt3/2) = 3 sqrt3 / 2, C1122 = (3/8 - 3/8) / (sqrt3/2) = 0.
    // Honeycomb (area 3 sqrt3/2, bonds 90/-30/210 deg): C1111 = (9/8 + 9/8) / area = sqrt3/2, C1212 = (3/8 + 3/4) / area
    //   = sqrt3/4, C1122 = (3/8 - 3/8) / area = 0.
    // Kagome (area 2 sqrt3, 6 bonds = each of 0/60/120 deg twice, sites with line pairs {0,60}, {0,120}, {60,120}):
    //   C1111 = (9/4 + 33/16 + 33/16 + 3/8) / area = 9 sqrt3/8, C1212 = (3/4 + 15/16 + 15/16 + 3/8) / area = sqrt3/2,
    //   C1122 = (3/4 + 3/16 + 3/16 - 3/8) / area = sqrt3/8.
    // Each is isotropic with lambda = C1122, mu = C1212: C1111 = lambda + 2 mu.
    let s3 = 3f64.sqrt();
    let iso = |l: f64, m: f64| [[l + 2.0 * m, 0.0, 0.0, l], [0.0, m, m, 0.0], [0.0, m, m, 0.0], [l, 0.0, 0.0, l + 2.0 * m]];
    let refs = [(triangular(), iso(0.0, 1.5 * s3)), (honeycomb(), iso(0.0, 0.25 * s3)), (kagome(), iso(s3 / 8.0, 0.5 * s3))];
    for (c, want) in &refs {
        let got = cauchy_born_tensor(c);
        assert!(close4(&got, want, 1e-12), "{got:?}");
        // hexagonal symmetry => isotropic: B = 2 mu (lambda + mu) / (lambda + 2 mu) I with lambda = C1122, mu = C1212
        assert_eq!(classify_elastic(&got), ElasticClass::Isotropic);
        let (lam, mu) = (got[0][0][1][1], got[0][1][0][1]);
        let b = barnett_lothe_tensor(&got, 128);
        let want_b = 2.0 * mu * (lam + mu) / (lam + 2.0 * mu);
        assert!((b[0][0] - want_b).abs() < 1e-10 && (b[1][1] - want_b).abs() < 1e-10 && b[0][1].abs() < 1e-10);
    }
    // Born-Huang shift: neither lattice relaxes internally, so frozen = relaxed exactly.
    // Kagome: every site is an inversion centre, so the strain-induced force on it is odd and vanishes.
    // Honeycomb (bonds d_k from A to B): the strain-shift coupling is sum_bonds d_k d_i d_a (yyy part +3/4) plus
    //   2 sites x sum_{p<q} c^2 (d_p + d_q)_k (d_p d_q + d_q d_p)_ia (yyy part -3/4). D3 leaves one free component, so
    //   the coupling is zero and the shift is not driven.
    for c in [honeycomb(), kagome()] {
        assert!(close4(&cauchy_born_tensor_opt(&c, false), &flat(&cauchy_born_tensor(&c)), 1e-12));
    }
}

fn flat(c: &Tensor4) -> [[f64; 4]; 4] {
    std::array::from_fn(|p| std::array::from_fn(|q| c[p / 2][p % 2][q / 2][q % 2]))
}

fn voigt(c11: f64, c22: f64, c12: f64, c66: f64, c16: f64, c26: f64) -> Tensor4 {
    let v = [[c11, c12, c16], [c12, c22, c26], [c16, c26, c66]];
    let m = |i: usize, a: usize| if i == a { i } else { 2 };
    let mut c = [[[[0.0; 2]; 2]; 2]; 2];
    for i in 0..2 { for a in 0..2 { for j in 0..2 { for b in 0..2 { c[i][a][j][b] = v[m(i, a)][m(j, b)]; } } } }
    c
}

#[test]
fn elastic_classes() {
    assert_eq!(classify_elastic(&voigt(3.0, 3.0, 1.0, 1.0, 0.0, 0.0)), ElasticClass::Isotropic);
    assert_eq!(classify_elastic(&voigt(3.0, 3.0, 1.0, 0.5, 0.0, 0.0)), ElasticClass::SquareCubic);
    assert_eq!(classify_elastic(&voigt(3.0, 2.0, 1.0, 0.7, 0.0, 0.0)), ElasticClass::Orthotropic);
    assert_eq!(classify_elastic(&rotate_tensor(&voigt(3.0, 2.0, 1.0, 0.7, 0.0, 0.0), 0.5)), ElasticClass::Orthotropic);
    assert_eq!(classify_elastic(&rotate_tensor(&voigt(3.0, 3.0, 1.0, 0.5, 0.0, 0.0), 0.3)), ElasticClass::SquareCubic);
    // Harmonics of E(phi) = c11 c^4 + c22 s^4 + (2 c12 + 4 c66) c^2 s^2 + 4 c16 c^3 s + 4 c26 c s^3:
    // cos 2phi: (c11 - c22) / 2, sin 2phi: c16 + c26, cos 4phi: (c11 + c22 - 2 c12 - 4 c66) / 8, sin 4phi: (c16 - c26) / 2.
    // (3, 2, 0.5, 1, 0.3, 0.2): (0.5, 0.5, 0, 0.05) -> p2 = 45 deg, p4 = 90 deg = 2 p2: mirror at 22.5 deg (orthotropic).
    assert_eq!(classify_elastic(&voigt(3.0, 2.0, 0.5, 1.0, 0.3, 0.2)), ElasticClass::Orthotropic);
    let unrot = rotate_tensor(&voigt(3.0, 2.0, 0.5, 1.0, 0.3, 0.2), -PI / 8.0);
    assert!(unrot[0][0][0][1].abs() < 1e-12 && unrot[1][1][0][1].abs() < 1e-12);
    // (3, 2, 1, 0.7, 0.3, 0): (0.5, 0.3, 0.025, 0.15) -> tan p2 = 0.6, tan 2 p2 = 1.875, tan p4 = 6, all angles in
    // (0, 90 deg): p4 != 2 p2 mod 180 deg, no common mirror (monoclinic).
    assert_eq!(classify_elastic(&voigt(3.0, 2.0, 1.0, 0.7, 0.3, 0.0)), ElasticClass::Monoclinic);
    let rect = cell([1.0, 0.0], [0.0, 1.5], vec![[0.0, 0.0]], &[[0, 0, 1, 0], [0, 0, 0, 1]]);
    assert_eq!(classify_elastic(&cauchy_born_tensor(&rect)), ElasticClass::Orthotropic);
}

fn check_stroh(c: &Tensor4, b: [f64; 2]) {
    let n = 4096;
    let pts: Vec<[f64; 2]> = (0..n).map(|k| { let t = -PI + 2.0 * PI * (k as f64 + 0.5) / n as f64; [t.cos(), t.sin()] }).collect();
    let f = stroh_dislocation_field(c, b, &pts).unwrap();
    // closed-circuit integral of du along the CCW unit circle = b; energy identity b.B b = 2 pi int r^2 sigma_ia du_i/dx_a
    let (mut jump, mut energy) = ([0.0; 2], 0.0);
    for (k, p) in pts.iter().enumerate() {
        let tang = [-p[1], p[0]];
        for (j, jv) in jump.iter_mut().enumerate() {
            *jv += (f.grad[k][j][0] * tang[0] + f.grad[k][j][1] * tang[1]) * 2.0 * PI / n as f64;
        }
        for i in 0..2 { for a in 0..2 { energy += f.stress[k][i][a] * f.grad[k][i][a] * 2.0 * PI / n as f64; } }
    }
    assert!((jump[0] - b[0]).abs() < 1e-9 && (jump[1] - b[1]).abs() < 1e-9, "{} jump {jump:?}", f.method);
    let bt = barnett_lothe_tensor(c, 256);
    let bbb: f64 = (0..4).map(|k| b[k / 2] * bt[k / 2][k % 2] * b[k % 2]).sum();
    assert!((2.0 * PI * energy - bbb).abs() < 1e-8 * bbb.max(1.0), "{} energy {} vs b.B.b {bbb}", f.method, 2.0 * PI * energy);
    // displacement itself jumps by b across the cut (negative x axis), and equilibrium d_a sigma_ia = 0 off the core
    let e = 1e-9;
    let uj = stroh_dislocation_field(c, b, &[[-0.8, e], [-0.8, -e]]).unwrap().u;
    assert!((uj[0][0] - uj[1][0] - b[0]).abs() < 1e-6 && (uj[0][1] - uj[1][1] - b[1]).abs() < 1e-6, "{} cut jump", f.method);
    let h = 1e-5;
    for x in [[0.7, 0.4], [-0.3, 0.9], [0.2, -1.3]] {
        let s = |dx: f64, dy: f64| stroh_dislocation_field(c, b, &[[x[0] + dx, x[1] + dy]]).unwrap().stress[0];
        let (sxp, sxm, syp, sym) = (s(h, 0.0), s(-h, 0.0), s(0.0, h), s(0.0, -h));
        for i in 0..2 {
            let div = (sxp[i][0] - sxm[i][0]) / (2.0 * h) + (syp[i][1] - sym[i][1]) / (2.0 * h);
            assert!(div.abs() < 1e-5, "{} div sigma {div}", f.method);
        }
    }
}

#[test]
fn stroh_field_identities() {
    // rectangular GSV model: orthotropic with C1122 = 0 and a double Stroh root (the sextic route cannot handle it)
    let rect = cell([1.0, 0.0], [0.0, 1.5], vec![[0.0, 0.0]], &[[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, -1]]);
    let degenerate = cauchy_born_tensor(&rect);
    assert_eq!(classify_elastic(&degenerate), ElasticClass::Orthotropic);
    assert!(stroh_field::stroh_sextic_field(&degenerate, [1.0, 0.0], &[[1.0, 0.0]]).is_err());
    let oblique = rotate_tensor(&voigt(3.0, 2.0, 0.5, 1.0, 0.0, 0.0), 0.4);
    let (sq, hc) = (cauchy_born_tensor(&square()), cauchy_born_tensor(&honeycomb()));
    for c in [sq, hc, degenerate, oblique] {
        for b in [[1.0, 0.0], [0.3, -0.8]] {
            check_stroh(&c, b);
        }
    }
    // the angular solver against the textbook isotropic field and the Stroh sextic (distinct roots)
    let pts = [[0.7, 0.4], [-0.3, 0.9], [0.2, -1.3], [-1.1, -0.2], [2.5, 0.1]];
    let b = [0.3, -0.8];
    let pairs = [
        (sq, isotropic_dislocation_field(&sq, b, &pts).unwrap()),
        (hc, isotropic_dislocation_field(&hc, b, &pts).unwrap()),
        (oblique, stroh_field::stroh_sextic_field(&oblique, b, &pts).unwrap()),
    ];
    for (c, other) in pairs {
        let ang = stroh_dislocation_field(&c, b, &pts).unwrap();
        let d0 = [ang.u[0][0] - other.u[0][0], ang.u[0][1] - other.u[0][1]];
        for (k, j) in (0..pts.len()).flat_map(|k| [(k, 0), (k, 1)]) {
            assert!((ang.u[k][j] - other.u[k][j] - d0[j]).abs() < 1e-10, "{}: u differs by more than a constant", other.method);
            for a in 0..2 {
                assert!((ang.grad[k][j][a] - other.grad[k][j][a]).abs() < 1e-10, "{}: grad", other.method);
            }
        }
    }
    // the closed form refuses an anisotropic C instead of approximating it
    assert!(isotropic_dislocation_field(&oblique, b, &pts).is_err());
    // Hirth & Lothe isotropic edge (b along x): u_y = -(b / 2 pi) (1 - 2 nu) / (4 (1 - nu)) ln r^2 + ..., so the ln r
    // coefficient is c = (0, -b mu / (2 pi (lambda + 2 mu))); square GSV model: lambda = 0, mu = 1 -> c_y = -1 / (4 pi)
    let lc = stroh_dislocation_field(&sq, [1.0, 0.0], &[[1.0, 0.0]]).unwrap().log_coeff;
    assert!(lc[0].abs() < 1e-12 && (lc[1] + 1.0 / (4.0 * PI)).abs() < 1e-12, "log coefficient {lc:?}");
}

#[test]
fn antiplane_screw_known_answers() {
    // K = mu I: the Volterra screw u_z = b theta / (2 pi) (Hirth & Lothe 1982 ch. 3), grad = b (-y, x) / (2 pi r^2)
    let (mu, bz) = (0.7, 1.3);
    let pts = [[1.0, 0.0], [0.0, 2.0], [-1.0, 1e-9], [-1.0, -1e-9], [0.3, -0.4]];
    let f = antiplane_screw_field([[mu, 0.0], [0.0, mu]], bz, &pts).unwrap();
    let want = [0.0, bz / 4.0, bz / 2.0, -bz / 2.0, bz * (-0.4f64).atan2(0.3) / (2.0 * PI)];
    for (k, w) in want.iter().enumerate() {
        assert!((f.uz[k] - w).abs() < 1e-8, "u_z at {:?}", pts[k]);
    }
    let (x, y) = (0.3, -0.4);
    let g = [-bz * y / (2.0 * PI * 0.25), bz * x / (2.0 * PI * 0.25)];
    assert!((f.grad[4][0] - g[0]).abs() < 1e-12 && (f.grad[4][1] - g[1]).abs() < 1e-12);
    assert!((f.stress[4][0] - mu * g[0]).abs() < 1e-12 && (f.stress[4][1] - mu * g[1]).abs() < 1e-12);
    // anisotropic K: jump +b across the cut and equilibrium K_ab d_a d_b u_z = 0 (central differences)
    let k = [[2.0, 0.5], [0.5, 1.0]];
    let u = |p: [f64; 2]| antiplane_screw_field(k, bz, &[p]).unwrap().uz[0];
    assert!((u([-0.8, 1e-9]) - u([-0.8, -1e-9]) - bz).abs() < 1e-6);
    let (p, h) = ([0.6, 0.7], 1e-3);
    let d2 = |a: [f64; 2], b: [f64; 2]| {
        (u([p[0] + h * (a[0] + b[0]), p[1] + h * (a[1] + b[1])]) - u([p[0] + h * (a[0] - b[0]), p[1] + h * (a[1] - b[1])])
            - u([p[0] - h * (a[0] - b[0]), p[1] - h * (a[1] - b[1])]) + u([p[0] - h * (a[0] + b[0]), p[1] - h * (a[1] + b[1])]))
            / (4.0 * h * h)
    };
    let lap = k[0][0] * d2([1.0, 0.0], [1.0, 0.0]) + 2.0 * k[0][1] * d2([1.0, 0.0], [0.0, 1.0]) + k[1][1] * d2([0.0, 1.0], [0.0, 1.0]);
    assert!(lap.abs() < 1e-5, "K_ab u_ab = {lap}");
    assert!(antiplane_screw_field([[1.0, 0.0], [0.0, -1.0]], 1.0, &pts).is_err());
}

#[test]
fn voronoi_coordination_hand_counts() {
    // perfect triangular crystal: z = 6, square crystal: z = 4 (the zero-length diagonal edges of the cocircular
    // quadruples are dropped by the tolerance)
    for (c, zi) in [(triangular(), 6), (square(), 4)] {
        let (pos, _) = patch(&c, 5);
        let v = voronoi_coordination(&pos, zi, 0.05).unwrap();
        let inner = interior(&pos, 3.0);
        assert!((0..pos.len()).filter(|&i| inner(i)).all(|i| v.z[i] == zi && v.charge[i] == Some(0)));
        assert!(v.rim.iter().any(|&r| r));
    }
    // triangular crystal displaced by the isotropic edge field: one 5-fold (+1) and one 7-fold (-1) site at the core,
    // net coordination charge 0 (a 5-7 pair is a neutral disclination dipole)
    let (refp, _) = patch(&triangular(), 6);
    let x0 = [0.37, 0.41];
    let c = cauchy_born_tensor(&triangular());
    let rel: Vec<[f64; 2]> = refp.iter().map(|&p| sub(p, x0)).collect();
    let f = isotropic_dislocation_field(&c, [1.0, 0.0], &rel).unwrap();
    let pos: Vec<[f64; 2]> = refp.iter().zip(&f.u).map(|(p, u)| [p[0] + u[0], p[1] + u[1]]).collect();
    let v = voronoi_coordination(&pos, 6, 0.05).unwrap();
    let mut core: Vec<i64> = (0..pos.len()).filter(|&i| norm(rel[i]) < 4.0).filter_map(|i| v.charge[i]).filter(|&q| q != 0).collect();
    core.sort_unstable();
    assert_eq!(core, vec![-1, 1]);
    // FS/RH: the same crystal read by the coordinate detector closes to -b, i.e. burgers_fs_rh = b
    let det = detect_defects_from_coords(&pos, "p6m", &triangular(), Frame2D::default(), None, None, 1).unwrap();
    let ds: Vec<&DetectedDefect> = det.defects.iter().filter(|d| d.kind == "dislocation").collect();
    assert_eq!(ds.len(), 1);
    let (bb, fs) = (ds[0].burgers.unwrap(), ds[0].burgers_fs_rh.unwrap());
    assert!((bb[0] + 1.0).abs() < 1e-9 && bb[1].abs() < 1e-9 && (fs[0] - 1.0).abs() < 1e-9 && fs[1].abs() < 1e-9);
}

#[test]
fn refusals_and_degenerate_inputs() {
    // collinear reference sites: Wigner-Seitz still assigns exactly (bucket size from the extent, not the zero area)
    let line: Vec<[f64; 2]> = (0..50i32).map(|k| [f64::from(k), 0.0]).collect();
    let ws = wigner_seitz_occupancy(&line, &[[10.2, 0.3], [10.4, -0.1], [33.9, 0.0]]).unwrap();
    assert_eq!(ws.assignment, vec![10, 10, 34]);
    assert_eq!(ws.interstitial_hosts, vec![10]);
    let (pos, edges) = patch(&square(), 2);
    let r = nye_tensor(&pos, &square(), Frame2D::default(), Some(&edges), None, Some(&[0usize, 9999][..]));
    assert!(r.is_err_and(|e| e.starts_with("REFUSED")));
    assert!(voronoi_coordination(&[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 6, 0.05).is_err());
}

fn lzw_decode(data: &[u8], min: u32) -> Vec<u8> {
    let (clear, eoi) = (1usize << min, (1usize << min) + 1);
    let fresh = || -> Vec<Vec<u8>> { (0..clear).map(|i| vec![i as u8]).chain([vec![], vec![]]).collect() };
    let (mut dict, mut width, mut prev, mut out) = (fresh(), min + 1, None::<usize>, Vec::new());
    let (mut acc, mut nb, mut it) = (0u32, 0u32, data.iter());
    loop {
        while nb < width {
            let Some(&byte) = it.next() else { return out };
            acc |= u32::from(byte) << nb;
            nb += 8;
        }
        let code = (acc & ((1 << width) - 1)) as usize;
        acc >>= width;
        nb -= width;
        if code == clear { dict = fresh(); width = min + 1; prev = None; continue; }
        if code == eoi { return out; }
        let entry = if code < dict.len() { dict[code].clone() } else {
            let p = dict[prev.expect("KwKwK needs a previous code")].clone();
            let f = p[0];
            [p, vec![f]].concat()
        };
        out.extend_from_slice(&entry);
        if let Some(p) = prev && dict.len() < 4096 {
            let new = [dict[p].clone(), vec![entry[0]]].concat();
            dict.push(new);
        }
        prev = Some(code);
        if dict.len() == (1 << width) && width < 12 { width += 1; }
    }
}

fn gif_frames(gif: &[u8]) -> Vec<Vec<u8>> {
    let k = u32::from(gif[10] & 7) + 1;
    let mut i = 13 + 3 * (1usize << k);
    let mut frames = Vec::new();
    loop {
        match gif[i] {
            0x21 => { i += 2; while gif[i] != 0 { i += 1 + usize::from(gif[i]); } i += 1; }
            0x2C => {
                let min = u32::from(gif[i + 10]);
                i += 11;
                let mut data = Vec::new();
                while gif[i] != 0 { data.extend_from_slice(&gif[i + 1..i + 1 + usize::from(gif[i])]); i += 1 + usize::from(gif[i]); }
                i += 1;
                frames.push(lzw_decode(&data, min));
            }
            0x3B => return frames,
            b => panic!("unexpected block 0x{b:02x}"),
        }
    }
}

#[test]
fn gif_round_trip_and_interpolation() {
    let fr = interpolate_crystal_frames(&[[0.0, 0.0], [1.0, 0.0]], &[[0.0, 1.0], [2.0, 0.0]], 5).unwrap();
    assert_eq!(fr.len(), 5);
    assert_eq!((fr[0][1], fr[4][1], fr[2][0]), ([1.0, 0.0], [2.0, 0.0], [0.0, 0.5]));
    let mut seed = 12345u32;
    let noise: Vec<u8> = (0..120 * 90).map(|_| { seed = seed.wrapping_mul(1103515245).wrapping_add(12345); ((seed >> 16) % 3) as u8 }).collect();
    let flat = vec![1u8; 120 * 90];
    let pal = [[0, 0, 0], [255, 255, 255], [255, 0, 0]];
    let gif = encode_gif89a(120, 90, &pal, &[noise.clone(), flat.clone()], 7).unwrap();
    assert_eq!(&gif[..6], b"GIF89a");
    assert!(gif.windows(11).any(|w| w == b"NETSCAPE2.0"));
    assert_eq!(*gif.last().unwrap(), 0x3B);
    assert_eq!(gif_frames(&gif), vec![noise.clone(), flat]);
    // a long incompressible stream crosses the 4096-entry table limit and must clear and resume exactly
    let mut long = noise.clone();
    long.extend(noise.iter().rev());
    assert_eq!(lzw_decode(&lzw_encode(&long, 2), 2), long);
    let imgs = render_frames(&fr, &[(0, 1)], 40, 30).unwrap();
    assert!(imgs.iter().all(|im| im.contains(&1) && im.contains(&2)));
    let g = export_motion_gif(&fr, &[(0, 1)], 40, 30, 5).unwrap();
    assert_eq!(gif_frames(&g), imgs);
}
