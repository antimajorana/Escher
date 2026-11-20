//! Seeds for BDF vertex completion on a regular hyperbolic `{p, q}` tiling (C1 WP-G3, plan §2.1).
//!
//! The touching rule is unchanged (`S_{n+1}` = `S_n` plus every tile sharing a vertex with it); only the seed set
//! `S_0` differs:
//! * `Cell` (or `Face`, the same thing in 2D): the base `p`-gon (1 tile);
//! * `Vertex`: the `q` tiles at base vertex 0 (the vertex star);
//! * `Edge`: the 2 tiles on base edge 0 (the base tile and its mirror image across that edge).
//!
//! `Dirichlet` is an E² lattice seed and is refused here. Layers are 0-based: layer 0 is the seed set.

use super::boundary_counts::{letter_counts_prefix, seed_letters};
use super::inflation::is_hyperbolic;
use super::inflation::{
    BaseFace, HypPointSet, IDENTITY, Mat3, T_PRECISION_LIMIT, VcPatch, Vec3, gamma_pq, mat_mul, mat_vec,
};
use crate::growth::SeedKind;

/// The seed tiles as Lorentz matrices (tile = matrix applied to the base tile), base tile first.
pub fn seed_tiles(base: &BaseFace, seed: SeedKind) -> Result<Vec<Mat3>, String> {
    match seed {
        SeedKind::Cell | SeedKind::Face => Ok(vec![IDENTITY]),
        SeedKind::Vertex => Ok(std::iter::once(IDENTITY).chain(base.star[0].iter().copied()).collect()),
        SeedKind::Edge => Ok(vec![IDENTITY, base.reflections[0]]),
        SeedKind::Dirichlet => Err("the Dirichlet seed is defined for E² lattices only; use a vertex, edge or tile seed".into()),
    }
}

/// The point a drawing of this seed is centred on: the base tile centre (tile seed), base vertex 0 (vertex seed)
/// or the midpoint of base edge 0 (edge seed), so that the drawing caps, measured from the disk centre, cut the
/// patch symmetrically about the seed.
pub fn seed_centre(base: &BaseFace, seed: SeedKind) -> Vec3 {
    match seed {
        SeedKind::Vertex => base.vertices[0],
        SeedKind::Edge => base.edge_midpoints[0],
        _ => [0.0, 0.0, 1.0],
    }
}

/// The Lorentz boost taking the hyperboloid point `x` to the origin `(0, 0, 1)` along the geodesic joining them:
/// with `s = (x0, x1)` and `t = x2`, `M = [[I + s s^T / (1 + t), -s], [-s^T, t]]`. Check: `M x = (s (1 + (t^2 - 1) /
/// (1 + t) - t), t^2 - |s|^2) = (0, 0, 1)`. For `x` = the origin it is exactly the identity.
pub fn recentre(x: Vec3) -> Mat3 {
    let (a, b, t) = (x[0], x[1], x[2]);
    let k = 1.0 / (1.0 + t);
    [[1.0 + a * a * k, a * b * k, -a], [a * b * k, 1.0 + b * b * k, -b], [-a, -b, t]]
}

/// Short description of the seed for readouts.
pub fn seed_label(p: usize, q: usize, seed: SeedKind) -> String {
    match seed {
        SeedKind::Cell | SeedKind::Face => format!("one {p}-gon"),
        SeedKind::Vertex => format!("the {q} tiles at a vertex"),
        SeedKind::Edge => "the 2 tiles on an edge".to_string(),
        SeedKind::Dirichlet => "a Dirichlet seed (refused in H²)".to_string(),
    }
}

/// Grow `S_0 .. S_{n_max}` from `seed` (see [`super::inflation::grow_vertex_completion`] for the stops: face
/// budget, f64 precision limit, ambiguous dedup; the partial shell is always dropped).
///
/// Budget estimate of shell `n + 1`: for the tile seed the original estimate (`p (q - 2)` for shell 1, then
/// `2 gamma a_n`), so tile-seed results are unchanged; for vertex and edge seeds the exact shell size from the
/// boundary-letter counter ([`super::boundary_counts::vertex_completion_counts_seeded`], prefix up to `u64`
/// overflow), falling back to `a_0 p (q - 1)` for shell 1
/// (each seed tile has `p` vertices with at most `q - 1` other tiles) and `2 gamma a_n` later if it returns `None`.
pub fn grow_vertex_completion_seeded(
    p: usize,
    q: usize,
    seed: SeedKind,
    n_max: usize,
    face_budget: usize,
) -> Result<VcPatch, String> {
    let base = BaseFace::new(p, q)?;
    let seed_mats = seed_tiles(&base, seed)?;
    // Exact shell sizes as far as u64 holds them (the budget stops growth long before that).
    let exact = match seed {
        SeedKind::Cell | SeedKind::Face => None,
        _ if !is_hyperbolic(p, q) => None,
        _ => seed_letters(p, q, seed).and_then(|(a0, cnt)| letter_counts_prefix(p, q, a0, cnt, n_max)).map(|(c, _)| c),
    };
    let mut centres = HypPointSet::new(2.0 * base.inradius);
    let mut verts = HypPointSet::new(base.edge_length());
    // Vertex indices of every face (`p` per face, in `verts`), and `completed[v]`: vertex v's star
    // has been added (v is a vertex of an expanded face).
    let (mut face_verts, mut completed): (Vec<u32>, Vec<bool>) = (Vec::new(), Vec::new());
    let add_vertices = |verts: &mut HypPointSet, face_verts: &mut Vec<u32>, completed: &mut Vec<bool>, g: &Mat3| {
        for &v in &base.vertices {
            let (vi, new) = verts.insert(mat_vec(g, v));
            if new {
                completed.push(false);
            }
            face_verts.push(vi as u32);
        }
    };
    let mut faces = Vec::new();
    for g in seed_mats {
        if centres.insert(mat_vec(&g, [0.0, 0.0, 1.0])).1 {
            add_vertices(&mut verts, &mut face_verts, &mut completed, &g);
            faces.push(g);
        }
    }
    let (mut shell, mut shell_sizes, mut stopped) = (vec![0_usize; faces.len()], vec![faces.len()], None);
    let two_gamma = 2.0 * gamma_pq(p, q);
    let star_centres: Vec<Vec<[f64; 3]>> =
        base.star.iter().map(|ring| ring.iter().map(|w| [w[0][2], w[1][2], w[2][2]]).collect()).collect();
    for n in 0..n_max {
        let a_n = shell_sizes[n];
        let estimate = match (&exact, seed, n) {
            (Some(c), _, _) if c.len() > n + 1 => usize::try_from(c[n + 1]).unwrap_or(usize::MAX),
            (_, SeedKind::Cell | SeedKind::Face, 0) => p * (q - 2),
            (_, _, 0) => a_n * p * (q - 1),
            _ => (two_gamma * a_n as f64).ceil() as usize,
        };
        if faces.len().saturating_add(estimate) > face_budget {
            stopped = Some(format!("face budget {face_budget}: shell {} (~{estimate} faces) not built", n + 1));
            break;
        }
        let (start, end) = (faces.len() - a_n, faces.len());
        let mut precision_hit = None;
        'shell: for fi in start..end {
            let g = faces[fi];
            for j in 0..p {
                let vi = face_verts[fi * p + j] as usize;
                if completed[vi] {
                    continue; // vertex already completed (it belongs to an earlier expanded face)
                }
                completed[vi] = true;
                for (m, &c0) in star_centres[j].iter().enumerate() {
                    let c = mat_vec(&g, c0);
                    // Written so that a NaN coordinate also stops.
                    if !c.iter().all(|x| x.is_finite()) || c[2] > T_PRECISION_LIMIT {
                        precision_hit = Some("f64 precision limit");
                        break 'shell;
                    }
                    if centres.insert(c).1 {
                        let h = mat_mul(&g, &base.star[j][m]);
                        add_vertices(&mut verts, &mut face_verts, &mut completed, &h);
                        faces.push(h);
                        shell.push(n + 1);
                    }
                }
            }
        }
        if precision_hit.is_none() && centres.suspect + verts.suspect > 0 {
            precision_hit = Some("f64 precision: ambiguous dedup (a distance within [sep/4, 3 sep/4])");
        }
        if let Some(why) = precision_hit {
            faces.truncate(end);
            shell.truncate(end);
            stopped = Some(format!("{why}: shell {} not built", n + 1));
            break;
        }
        shell_sizes.push(faces.len() - end);
    }
    let (max_merged_centre, min_distinct_centre) = (centres.max_merged, centres.min_distinct);
    Ok(VcPatch { base, faces, shell, shell_sizes, max_merged_centre, min_distinct_centre, stopped })
}
