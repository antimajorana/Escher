//! Known-answer tests of the canonical 3D net motifs (C1 WP-N1). Reference site lists are written out here from
//! the International Tables (Vol. A, 2016: #227 origin choice 2, #180, #136, #221), not read from the table under
//! test; the exact space group of every motif is checked with spglib in `tests/test_net_motifs.py`.

use super::{net_motif, NetMotif};
use crate::flat3d::Cell3D;
use crate::space3d::nets3d::{build_net_3d, CanonicalNet3d, LineDefect3d};

fn same_mod1(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|k| {
        let d = (a[k] - b[k]).rem_euclid(1.0);
        d.min(1.0 - d) < 1e-9
    })
}

/// `sites` and `want` are the same set modulo lattice translations (and `sites` has no repeats).
fn same_set_mod1(sites: &[[f64; 3]], want: &[[f64; 3]]) -> bool {
    sites.len() == want.len()
        && want.iter().all(|w| sites.iter().filter(|s| same_mod1(**s, *w)).count() == 1)
}

fn frac(m: &NetMotif) -> Vec<[f64; 3]> {
    m.sites.iter().map(|s| s.0).collect()
}

fn cart(lat: &[[f64; 3]; 3], f: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|c| f[0] * lat[0][c] + f[1] * lat[1][c] + f[2] * lat[2][c])
}

/// Sorted distances from site `i` to every other site image in a 5x5x5 block (independent of the bond cutoff).
fn distances_from(m: &NetMotif, i: usize) -> Vec<f64> {
    let p = cart(&m.lattice, m.sites[i].0);
    let mut out = Vec::new();
    for (f, _) in &m.sites {
        for n in 0..125_i32 {
            let sh = [f64::from(n / 25 - 2), f64::from(n / 5 % 5 - 2), f64::from(n % 5 - 2)];
            let q = cart(&m.lattice, [f[0] + sh[0], f[1] + sh[1], f[2] + sh[2]]);
            let d = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2)).sqrt();
            if d > 1e-9 {
                out.push(d);
            }
        }
    }
    out.sort_by(f64::total_cmp);
    out
}

/// Per-site degree of the periodic bond graph of `Cell3D::from_canonical_net`.
fn degrees(net: CanonicalNet3d) -> Vec<usize> {
    let c = Cell3D::from_canonical_net(net);
    let mut deg = vec![0_usize; c.basis.len()];
    for b in &c.bonds {
        deg[b.i] += 1;
        deg[b.j] += 1;
    }
    deg
}

#[test]
fn site_counts_and_kind_convention() {
    // Hand counts per conventional cell; kinds per the module doc (a label names the coloured structure).
    for (net, n_sites, n_kinds) in [
        (CanonicalNet3d::SimpleCubic, 1, 1),
        (CanonicalNet3d::BCC, 2, 1),
        (CanonicalNet3d::FCC, 4, 1),
        (CanonicalNet3d::Diamond, 8, 1),
        (CanonicalNet3d::HCP, 2, 1),
        (CanonicalNet3d::Tetragonal123, 2, 2),
        (CanonicalNet3d::Rutile136, 6, 2),
        (CanonicalNet3d::Perovskite, 5, 3),
        (CanonicalNet3d::Pyrochlore, 16, 1),
        (CanonicalNet3d::BetaQuartz, 3, 1),
    ] {
        let m = net_motif(net);
        assert_eq!(m.sites.len(), n_sites, "{net:?}");
        let mut kinds: Vec<u8> = m.sites.iter().map(|s| s.1).collect();
        kinds.sort_unstable();
        kinds.dedup();
        assert_eq!(kinds.len(), n_kinds, "{net:?}");
        assert!(m.sites.iter().all(|s| s.0.iter().all(|&x| (0.0..1.0).contains(&x))), "{net:?} not in [0,1)");
        // The pristine patch is a 2x2x2 block of conventional cells.
        assert_eq!(build_net_3d(net, LineDefect3d::None).positions.len(), 8 * n_sites, "{net:?}");
    }
}

#[test]
fn pyrochlore_is_16c_the_diamond_bond_midpoints() {
    // ITA #227 origin choice 2, 16c: 0,0,0; 3/4,1/4,1/2; 1/4,1/2,3/4; 1/2,3/4,1/4 (+ F-centring), shifted by
    // (1/8,1/8,1/8) to the diamond-at-(0,0,0) setting used by the Diamond motif (8a at 1/8,1/8,1/8 in choice 2).
    let fc = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]];
    let c16 = [[0.0, 0.0, 0.0], [0.75, 0.25, 0.5], [0.25, 0.5, 0.75], [0.5, 0.75, 0.25]];
    let ita: Vec<[f64; 3]> =
        fc.iter().flat_map(|t| c16.iter().map(move |s| core::array::from_fn(|k| s[k] + t[k] + 0.125))).collect();
    let pyro = net_motif(CanonicalNet3d::Pyrochlore);
    assert!(same_set_mod1(&frac(&pyro), &ita), "pyrochlore != 16c");

    // Independent route: midpoints of the nearest-neighbour bonds of diamond (8a at 0,0,0 and 1/4,1/4,1/4 + F),
    // each A atom bonded to A + (1/4)(+-1,+-1,+-1) with an even number of minus signs.
    let a_sites: Vec<[f64; 3]> = fc.to_vec();
    let dirs = [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]];
    let mut mids: Vec<[f64; 3]> = Vec::new();
    for a in &a_sites {
        for d in &dirs {
            let m: [f64; 3] = core::array::from_fn(|k| (a[k] + d[k] / 8.0).rem_euclid(1.0));
            if !mids.iter().any(|x| same_mod1(*x, m)) {
                mids.push(m);
            }
        }
    }
    assert_eq!(mids.len(), 16);
    assert!(same_set_mod1(&frac(&pyro), &mids), "pyrochlore != diamond bond midpoints");
}

#[test]
fn beta_quartz_is_3c_of_p6_222() {
    // ITA #180 Wyckoff 3c: 1/2,0,0; 0,1/2,2/3; 1/2,1/2,1/3.
    let bq = net_motif(CanonicalNet3d::BetaQuartz);
    let ita = [[0.5, 0.0, 0.0], [0.0, 0.5, 2.0 / 3.0], [0.5, 0.5, 1.0 / 3.0]];
    assert!(same_set_mod1(&frac(&bq), &ita));
    // #180's six-fold is 6_2: (x - y, x, z + 1/3) maps the set to itself; #181's 6_4 (z + 2/3) does not.
    let six = |p: [f64; 3], tz: f64| [p[0] - p[1], p[0], p[2] + tz];
    let s = frac(&bq);
    let mapped = |tz: f64| s.iter().map(|&p| six(p, tz)).collect::<Vec<_>>();
    assert!(same_set_mod1(&mapped(1.0 / 3.0), &s), "not invariant under 6_2");
    assert!(!same_set_mod1(&mapped(2.0 / 3.0), &s), "invariant under 6_4: this would be #181");
}

#[test]
fn rutile_and_perovskite_match_ita() {
    // #136 2a: 0,0,0; 1/2,1/2,1/2. 4f (x = 0.3): x,x,0; -x,-x,0; -x+1/2,x+1/2,1/2; x+1/2,-x+1/2,1/2.
    let ru = net_motif(CanonicalNet3d::Rutile136);
    let x = 0.3;
    let o = [[x, x, 0.0], [-x, -x, 0.0], [0.5 - x, 0.5 + x, 0.5], [0.5 + x, 0.5 - x, 0.5]];
    let pick = |m: &NetMotif, k: u8| m.sites.iter().filter(|s| s.1 == k).map(|s| s.0).collect::<Vec<_>>();
    assert!(same_set_mod1(&pick(&ru, 0), &[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]));
    assert!(same_set_mod1(&pick(&ru, 1), &o));
    // #221: B at 1a, A at 1b, O at 3d (1/2,0,0; 0,1/2,0; 0,0,1/2).
    let pv = net_motif(CanonicalNet3d::Perovskite);
    assert!(same_set_mod1(&pick(&pv, 0), &[[0.0, 0.0, 0.0]]));
    assert!(same_set_mod1(&pick(&pv, 1), &[[0.5, 0.5, 0.5]]));
    assert!(same_set_mod1(&pick(&pv, 2), &[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]));
}

#[test]
fn coordination_numbers_and_bond_shells() {
    // Hand values (a = 1): sc 6 at 1; bcc 8 at sqrt(3)/2; fcc 12 at sqrt(2)/2; diamond 4 at sqrt(3)/4; pyrochlore
    // 6 at sqrt(2)/4 (corner-sharing tetrahedra on the diamond bond midpoints); hcp (c = 1.35, not ideal) 6 at
    // sqrt(1/3 + (c/2)^2) between layers plus 6 at 1 in-plane; beta-quartz Si 4 at sqrt(1/4 + (c/3)^2): two at
    // in-plane offset 1/2 and dz = +c/3, two at dz = -c/3.
    let r3 = 3.0_f64.sqrt();
    let r2 = 2.0_f64.sqrt();
    let c = super::HEX_C;
    let hcp_out = (1.0 / 3.0 + (c / 2.0).powi(2)).sqrt();
    let bq = (0.25 + (c / 3.0).powi(2)).sqrt();
    let cases: [(CanonicalNet3d, &[(usize, f64)]); 7] = [
        (CanonicalNet3d::SimpleCubic, &[(6, 1.0)]),
        (CanonicalNet3d::BCC, &[(8, r3 / 2.0)]),
        (CanonicalNet3d::FCC, &[(12, r2 / 2.0)]),
        (CanonicalNet3d::Diamond, &[(4, r3 / 4.0)]),
        (CanonicalNet3d::Pyrochlore, &[(6, r2 / 4.0)]),
        (CanonicalNet3d::HCP, &[(6, hcp_out), (6, 1.0)]),
        (CanonicalNet3d::BetaQuartz, &[(4, bq)]),
    ];
    for (net, shells) in cases {
        let want: usize = shells.iter().map(|s| s.0).sum();
        assert!(degrees(net).iter().all(|&d| d == want), "{net:?}: {:?}", degrees(net));
        let m = net_motif(net);
        let hand: Vec<f64> = shells.iter().flat_map(|&(n, d)| std::iter::repeat_n(d, n)).collect();
        for i in 0..m.sites.len() {
            let d = distances_from(&m, i);
            for (k, h) in hand.iter().enumerate() {
                assert!((d[k] - h).abs() < 1e-9, "{net:?} site {i} neighbour {k}: {} vs hand {h}", d[k]);
            }
            // The cutoff keeps exactly these `want` neighbours.
            assert!(d[want - 1] <= m.bond_cutoff && m.bond_cutoff < d[want] - 1e-6, "{net:?} cutoff");
        }
    }
}
