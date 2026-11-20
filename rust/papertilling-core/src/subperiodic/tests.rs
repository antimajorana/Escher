//! Frieze tests (C1 WP-B2). Known answers are hand-derived: the standard frieze table of which
//! isometry types each group contains, the [PROVED] Möbius rule in `frieze_patch.rs`, and the counts
//! V = 3N, E = 5N, F = 2N of an N-column, 3-row closed quad strip. Everything is measured from the
//! built patch (quads, edges, positions), not read from the builder's tables.

use std::collections::{BTreeMap, BTreeSet};

use super::*;

const DEFECTS: [FriezeDefect; 3] = [FriezeDefect::None, FriezeDefect::HalfTurn180, FriezeDefect::GlideDispiration];
const TOPOS: [FriezeTopology; 3] = [FriezeTopology::OpenStrip, FriezeTopology::Cylinder, FriezeTopology::Mobius];

#[test]
fn seven_friezes_have_exact_five_split_and_two_nonsymmorphic() {
    let mut split = 0usize;
    let mut nonsym = Vec::new();
    for &sym in &FRIEZE_SYMBOLS {
        let (s, _) = analyze_frieze_group(sym).unwrap();
        if s.is_split {
            split += 1;
        } else {
            nonsym.push(sym);
        }
    }
    assert_eq!(split, 5);
    assert_eq!(nonsym, vec!["p11g", "p2mg"]);
    assert_eq!(analyze_frieze_group("p11m").unwrap().0.ab_full_label, "Z_2 x Z");
    assert_eq!(analyze_frieze_group("p11g").unwrap().0.ab_full_label, "Z");
    let mob = build_frieze_patch("p2mg", 10, FriezeTopology::Mobius, FriezeDefect::GlideDispiration).unwrap();
    assert!(!mob.pos3d.is_empty() && !mob.quads.is_empty());
}

/// Quad-complex counts: (V, E, F, boundary cycle lengths sorted, orientable).
fn complex_of(p: &FriezePatch) -> (usize, usize, usize, Vec<usize>, bool) {
    let mut verts = BTreeSet::new();
    let mut side_count: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for q in &p.quads {
        for k in 0..4 {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            verts.insert(a);
            *side_count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let bnd: Vec<(usize, usize)> = side_count.iter().filter(|&(_, &c)| c == 1).map(|(&e, _)| e).collect();
    // Boundary cycles: every boundary vertex has degree 2 in the boundary graph of a surface strip.
    let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &(a, b) in &bnd {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    assert!(adj.values().all(|n| n.len() == 2), "boundary is not a union of cycles");
    let mut seen = BTreeSet::new();
    let mut cycles = Vec::new();
    for &start in adj.keys() {
        if !seen.insert(start) {
            continue;
        }
        let (mut prev, mut cur, mut len) = (start, adj[&start][0], 1usize);
        while cur != start {
            seen.insert(cur);
            let nxt = if adj[&cur][0] == prev { adj[&cur][1] } else { adj[&cur][0] };
            prev = cur;
            cur = nxt;
            len += 1;
        }
        cycles.push(len);
    }
    cycles.sort_unstable();
    (verts.len(), side_count.len(), p.quads.len(), cycles, orientable(&p.quads))
}

/// Try to orient all quads coherently (each shared side traversed in opposite directions).
fn orientable(quads: &[[usize; 4]]) -> bool {
    let mut by_side: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (i, q) in quads.iter().enumerate() {
        for k in 0..4 {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            by_side.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    let dir = |q: &[usize; 4], a: usize, b: usize| (0..4).any(|k| q[k] == a && q[(k + 1) % 4] == b);
    let mut sign: Vec<Option<bool>> = vec![None; quads.len()];
    for root in 0..quads.len() {
        if sign[root].is_some() {
            continue;
        }
        sign[root] = Some(true);
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            for (&(a, b), qs) in &by_side {
                if !qs.contains(&i) || qs.len() != 2 {
                    continue;
                }
                let j = if qs[0] == i { qs[1] } else { qs[0] };
                // Same stored direction on the shared side means opposite orientations are needed.
                let same = dir(&quads[i], a, b) == dir(&quads[j], a, b);
                let want = if same { !sign[i].unwrap() } else { sign[i].unwrap() };
                match sign[j] {
                    None => {
                        sign[j] = Some(want);
                        stack.push(j);
                    }
                    Some(s) if s != want => return false,
                    _ => {}
                }
            }
        }
    }
    true
}

#[test]
fn closed_strips_are_cylinder_and_mobius_with_hand_counts() {
    for &sym in &FRIEZE_SYMBOLS {
        for cells in [4usize, 7, 12] {
            let cyl = build_frieze_patch(sym, cells, FriezeTopology::Cylinder, FriezeDefect::None).unwrap();
            let n = cyl.columns;
            assert_eq!(n, 2 * cells, "cylinder of {cells} periods has 2·cells glide columns");
            // Hand: N columns × 3 rows closed: V = 3N, E = 5N, F = 2N, χ = 0; two boundary circles of N.
            assert_eq!(complex_of(&cyl), (3 * n, 5 * n, 2 * n, vec![n, n], true), "{sym} cylinder");
            let open = build_frieze_patch(sym, cells, FriezeTopology::OpenStrip, FriezeDefect::None).unwrap();
            let m = open.columns;
            let (v, e, f, b, o) = complex_of(&open);
            assert_eq!(v as i64 - e as i64 + f as i64, 1, "{sym} open strip is a disk");
            assert_eq!((b, o), (vec![2 * (m - 1) + 4], true), "{sym} open strip has one boundary circle");
            match build_frieze_patch(sym, cells, FriezeTopology::Mobius, FriezeDefect::None) {
                Ok(mob) => {
                    let n = mob.columns;
                    // Hand: same counts, χ = 0; one boundary circle of length 2N; non-orientable.
                    assert_eq!(complex_of(&mob), (3 * n, 5 * n, 2 * n, vec![2 * n], false), "{sym} Möbius");
                }
                Err(why) => assert!(why.starts_with("[PROVED]"), "{sym}: {why}"),
            }
        }
    }
}

#[test]
fn mobius_refusal_set_is_exactly_p1_p2_p1m1_and_glide_lengths_are_odd() {
    let refused: Vec<&str> = FRIEZE_SYMBOLS
        .iter()
        .copied()
        .filter(|s| build_frieze_patch(s, 8, FriezeTopology::Mobius, FriezeDefect::None).is_err())
        .collect();
    assert_eq!(refused, vec!["p1", "p1m1", "p2"]);
    // Hand: p11g and p2mg glue after an odd number 2n+1 of glides (n + 1/2 periods);
    // p11m and p2mm after n whole periods (mirror composed with n translations).
    for (sym, cols, len) in [("p11g", 21, 10.5), ("p2mg", 21, 10.5), ("p11m", 20, 10.0), ("p2mm", 20, 10.0)] {
        let p = build_frieze_patch(sym, 10, FriezeTopology::Mobius, FriezeDefect::None).unwrap();
        assert_eq!((p.columns, p.length_periods), (cols, len), "{sym}");
        assert_eq!(p.columns % 2 == 1, sym == "p11g" || sym == "p2mg");
    }
}

/// Unordered decoration edges ((ix, iy), (ix, iy)).
type Decoration = BTreeSet<((i64, i64), (i64, i64))>;

/// Kind-2 decoration edges of a long open strip as unordered ((ix, iy), (ix, iy)) pairs.
fn decoration(sym: &str) -> (Decoration, i64) {
    let p = build_frieze_patch(sym, 24, FriezeTopology::OpenStrip, FriezeDefect::None).unwrap();
    let at = |v: usize| ((v / 3) as i64, (v % 3) as i64);
    let set = p
        .edges
        .iter()
        .filter(|e| e.2 == 2)
        .map(|&(u, v, _)| {
            let (a, b) = (at(u), at(v));
            (a.min(b), a.max(b))
        })
        .collect();
    (set, p.columns as i64)
}

/// Is (ix, iy) -> (s·ix + t, flip? 2 - iy) a symmetry of the decoration (checked inside the window)?
fn is_symmetry(dec: &Decoration, cols: i64, s: i64, t: i64, flip: bool) -> bool {
    let map = |(x, y): (i64, i64)| (s * x + t, if flip { 2 - y } else { y });
    let mut checked = 0;
    for &(a, b) in dec {
        let (ma, mb) = (map(a), map(b));
        if [ma.0, mb.0].iter().all(|&x| (0..cols).contains(&x)) {
            checked += 1;
            if !dec.contains(&(ma.min(mb), ma.max(mb))) {
                return false;
            }
        }
    }
    assert!(checked > 10, "window too small");
    true
}

#[test]
fn decorations_realise_exactly_their_frieze_group() {
    // Standard frieze table (hand): (horizontal mirror, glide, vertical mirror, half-turn).
    let table = [
        ("p1", [false, false, false, false]),
        ("p11m", [true, false, false, false]),
        ("p1m1", [false, false, true, false]),
        ("p11g", [false, true, false, false]),
        ("p2", [false, false, false, true]),
        ("p2mm", [true, false, true, true]),
        ("p2mg", [false, true, true, true]),
    ];
    for (sym, want) in table {
        let (dec, cols) = decoration(sym);
        // Period is 2 columns (one cell), not 1.
        assert!(!is_symmetry(&dec, cols, 1, 1, false), "{sym} has period one column");
        assert!(is_symmetry(&dec, cols, 1, 2, false), "{sym} lacks its period");
        let my = [0, 2].iter().any(|&t| is_symmetry(&dec, cols, 1, t, true));
        let glide = [1, 3].iter().any(|&t| is_symmetry(&dec, cols, 1, t, true));
        let mx = (40..44).any(|t| is_symmetry(&dec, cols, -1, t, false));
        let half = (40..44).any(|t| is_symmetry(&dec, cols, -1, t, true));
        assert_eq!([my, glide, mx, half], want, "{sym}");
    }
}

#[test]
fn every_gluing_used_is_a_symmetry_of_the_decoration() {
    for &sym in &FRIEZE_SYMBOLS {
        let (dec, cols) = decoration(sym);
        for cells in 4..=10 {
            for topo in [FriezeTopology::Cylinder, FriezeTopology::Mobius] {
                let Ok(p) = build_frieze_patch(sym, cells, topo, FriezeDefect::None) else { continue };
                let flip = topo == FriezeTopology::Mobius;
                assert!(is_symmetry(&dec, cols, 1, p.columns as i64, flip), "{sym} {topo:?} {cells}");
            }
            // Computational evidence for the refusal on the drawn pattern: no flip-translation
            // of 1..=21 columns is a symmetry of p1, p2 or p1m1.
            if matches!(sym, "p1" | "p2" | "p1m1") {
                assert!((1..=21).all(|t| !is_symmetry(&dec, cols, 1, t, true)), "{sym}");
            }
        }
    }
}

#[test]
fn no_two_vertices_coincide_on_any_allowed_patch() {
    for &sym in &FRIEZE_SYMBOLS {
        for topo in TOPOS {
            for defect in DEFECTS {
                for cells in [4usize, 5, 13, 24] {
                    let Ok(p) = build_frieze_patch(sym, cells, topo, defect) else { continue };
                    let mut min_d = f64::INFINITY;
                    for i in 0..p.pos3d.len() {
                        for j in (i + 1)..p.pos3d.len() {
                            let d: f64 = (0..3).map(|k| (p.pos3d[i][k] - p.pos3d[j][k]).powi(2)).sum::<f64>().sqrt();
                            min_d = min_d.min(d);
                        }
                    }
                    assert!(min_d > 0.1, "{sym} {topo:?} {defect:?} {cells}: min distance {min_d}");
                }
            }
        }
    }
}

#[test]
fn pictures_are_labelled_and_the_cylinder_is_isometric() {
    let mob = build_frieze_patch("p11m", 6, FriezeTopology::Mobius, FriezeDefect::None).unwrap();
    assert_eq!(mob.picture_note.as_deref(), Some("topological picture, not isometric"));
    let cyl = build_frieze_patch("p11m", 6, FriezeTopology::Cylinder, FriezeDefect::None).unwrap();
    assert_eq!(cyl.picture_note, None);
    // Hand: circumference = strip length = 6 periods (a = 1), so radius 6 / 2π; rungs keep width 0.7.
    for p in &cyl.pos3d {
        assert!(((p[0] * p[0] + p[2] * p[2]).sqrt() - 6.0 / (2.0 * std::f64::consts::PI)).abs() < 1e-12);
    }
    let rung = |u: usize, v: usize| (0..3).map(|k| (cyl.pos3d[u][k] - cyl.pos3d[v][k]).powi(2)).sum::<f64>().sqrt();
    assert!((rung(0, 1) - 0.7).abs() < 1e-12);
    // The seam is listed: 3 lattice + decoration edges and 2 quads cross it.
    assert_eq!(cyl.wrap_quads.len(), 2);
    assert!(cyl.wrap_edges.len() >= 3);
}
