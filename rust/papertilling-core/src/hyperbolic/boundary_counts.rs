//! Exact integer shell counts of BDF vertex completion on `{p, q}` from the boundary alone.
//!
//! The boundary of `S_n` is the cyclic word of `k_i` = number of patch faces at boundary vertex `i`;
//! `o_i = q - k_i` faces at vertex `i` lie outside. Two counters read the same local rule:
//!
//! - [`vertex_completion_counts_boundary_word`] keeps the whole word (memory ~ boundary length, so
//!   only small `n`); it is the transcription of the rule, used as a cross-check.
//! - [`vertex_completion_counts_combinatorial`] keeps only the letter counts `#{i : k_i = k}`. This is
//!   exact because, in the hyperbolic range, the new boundary is a substitution of the old letters
//!   (each letter maps to a fixed block), derived below. It is O(n q) and is what the UI table uses.
//!
//! Derivation of the letter substitution (one step). Around boundary vertex `i` the new faces are,
//! in cyclic order, `o_i - 2` corner faces (touching the patch only at `i`; `p - 1` new vertices)
//! and then the edge face across the boundary edge after `i` (`p - m - 1` new vertices, `m` = number
//! of old edges it touches). Consecutive new faces share one new vertex; a face with `c >= 2` new
//! vertices has `c - 2` private ones (`k = 1`), and a face with `c = 1` makes the shared vertex before
//! and after it the same vertex.
//! - `q > 3`, `p >= 4`: if every `o_i >= 2`, every edge face has `m = 1` (`c = p - 2 >= 2`) and corner
//!   faces `c = p - 1`, so all shared vertices have `k = 2`: letter `k` (with `o = q - k`) gives `o - 1`
//!   faces, `o - 1` new letters 2 and `(o - 2)(p - 3) + (p - 4)` new letters 1.
//! - `q > 3`, `p = 3`: edge faces have `c = 1`. If every `o_i >= 3` each edge face sits between two
//!   corner faces and its one new vertex lies in 3 new faces: letter `k` gives `o - 1` faces,
//!   `o - 3` letters 2 and one letter 3 (corner faces have `c - 2 = 0` private vertices).
//! - `q = 3`: `o_i in {1, 2}`; a letter 2 (`o = 1`) joins the edge faces on both its sides into one
//!   face with `m = 2`. If no two letters 2 are adjacent, there is one face per letter 1, `#2` of them
//!   with `m = 2` (`c = p - 3`) and the rest with `m = 1` (`c = p - 2`); new letters: one 2 per face and
//!   `c - 2` letters 1. Letters 2 stay non-adjacent while every face has `c >= 3`.
//!
//! These conditions hold for every hyperbolic pair at every step: for `q > 3, p >= 4` the letters
//! after step 1 are 1 and 2, so `o >= q - 2 >= 2`; for `p = 3` (then `q >= 7`) they are 2 and 3, so
//! `o >= q - 3 >= 4`; for `q = 3` (then `p >= 7`) `c >= p - 3 >= 4`. The seed (all letters 1) meets
//! them too. So the letter counter returns `None` only on `u64` overflow (or for non-hyperbolic pairs);
//! a failed condition also returns `None` (never a wrong count).

use super::inflation::is_hyperbolic;

/// Exact integer shell sizes `[a_0, .., a_{n_max}]` (`a_0 = 1`) of vertex completion from one
/// `p`-gon, from boundary letter counts (module doc). `None` for a non-hyperbolic pair, on `u64`
/// overflow, or if a step leaves the convex regime of the derivation (which does not happen for
/// hyperbolic pairs; see the module doc).
pub fn vertex_completion_counts_combinatorial(p: usize, q: usize, n_max: usize) -> Option<Vec<u64>> {
    if !is_hyperbolic(p, q) {
        return None;
    }
    // cnt[k] = number of boundary vertices lying in exactly k patch faces.
    let mut cnt = vec![0_u64; q + 2];
    cnt[1] = p as u64;
    let mut out = vec![1_u64];
    for _ in 0..n_max {
        let (faces, next) = if q == 3 { step_q3(p, &cnt)? } else { step_q_gt3(p, q, &cnt)? };
        out.push(faces);
        cnt = next;
    }
    Some(out)
}

/// `a + b * c` with overflow check.
fn mac(a: u64, b: u64, c: u64) -> Option<u64> { a.checked_add(b.checked_mul(c)?) }

fn step_q_gt3(p: usize, q: usize, cnt: &[u64]) -> Option<(u64, Vec<u64>)> {
    let mut next = vec![0_u64; q + 2];
    let mut faces = 0_u64;
    let (p, qq) = (p as u64, q as u64);
    for (k, &n) in cnt.iter().enumerate().filter(|&(_, &n)| n > 0) {
        let o = qq.checked_sub(k as u64)?;
        if o < 2 || (p == 3 && o < 3) {
            return None; // outside the derivation (not convex, or adjacent edge faces for p = 3)
        }
        faces = mac(faces, n, o - 1)?;
        if p >= 4 {
            next[2] = mac(next[2], n, o - 1)?;
            next[1] = mac(next[1], n, (o - 2) * (p - 3) + (p - 4))?;
        } else {
            next[2] = mac(next[2], n, o - 3)?;
            next[3] = next[3].checked_add(n)?;
        }
    }
    Some((faces, next))
}

fn step_q3(p: usize, cnt: &[u64]) -> Option<(u64, Vec<u64>)> {
    if cnt.iter().skip(3).any(|&n| n > 0) || cnt[1] < cnt[2] || cnt[1] == 0 {
        return None;
    }
    let (n1, n2) = (cnt[1], cnt[2]);
    // c = new vertices of faces touching m = 1 (n1 - n2 faces) and m = 2 (n2 faces) old edges.
    let (c1, c2) = ((p as u64).checked_sub(2)?, (p as u64).checked_sub(3)?);
    // c >= 3 keeps the new letters 2 non-adjacent (module doc); p >= 7 for q = 3 gives c >= 4.
    if (n1 > n2 && c1 < 3) || (n2 > 0 && c2 < 3) {
        return None;
    }
    let mut next = vec![0_u64; 5];
    next[2] = n1;
    next[1] = mac(mac(0, n1 - n2, c1 - 2)?, n2, c2.saturating_sub(2))?;
    Some((n1, next))
}

/// Exact integer shell counts `[a_0, .., a_{n_max}]` from the whole boundary word (cross-check of
/// [`vertex_completion_counts_combinatorial`]; memory grows like the boundary, ~`lambda_+^n`).
///
/// Around vertex `i` there are `o_i = q - k_i` outside faces: the faces across its two boundary edges
/// and `o_i - 2` corner faces; if `o_i = 1` the faces across both edges coincide. A new face touching
/// `m` old boundary edges gets `p - m - 1` new vertices; consecutive new faces share one new vertex.
/// Valid while the patch is convex (BDF Sec. III): `o_i >= 2` for `q > 3`, and no two adjacent
/// `o_i = 1` for `q = 3`. Returns `None` if that fails.
pub fn vertex_completion_counts_boundary_word(p: usize, q: usize, n_max: usize) -> Option<Vec<u64>> {
    if !is_hyperbolic(p, q) {
        return None;
    }
    let mut boundary: Vec<usize> = vec![1; p];
    let mut out = vec![1_u64];
    for _ in 0..n_max {
        let l = boundary.len();
        let o: Vec<usize> = boundary.iter().map(|&k| q.checked_sub(k).filter(|&x| x >= 1)).collect::<Option<_>>()?;
        let convex = if q > 3 { o.iter().all(|&x| x >= 2) } else { (0..l).all(|i| o[i] >= 2 || o[(i + 1) % l] >= 2) };
        if !convex {
            return None;
        }
        // New faces in cyclic order, as their new-vertex counts c = p - m - 1.
        let i0 = (0..l).find(|&i| o[i] >= 2)?;
        let mut c_list: Vec<usize> = Vec::new();
        let mut open_m = 0_usize; // edges touched by the currently open edge face
        for t in 0..l {
            let i = (i0 + t) % l;
            if o[i] == 1 {
                open_m += 1;
                continue;
            }
            if open_m > 0 {
                c_list.push(p.checked_sub(open_m + 1).filter(|&c| c >= 1)?);
            }
            c_list.extend(std::iter::repeat_n(p - 1, o[i] - 2));
            open_m = 1;
        }
        c_list.push(p.checked_sub(open_m + 1).filter(|&c| c >= 1)?);
        out.push(c_list.len() as u64);
        // New boundary: private vertices (k = 1) and shared vertices (k = faces containing them).
        let f0 = (0..c_list.len()).find(|&j| c_list[j] >= 2)?;
        let mut next = Vec::with_capacity(c_list.len() * p);
        let mut pending = 0_usize;
        for t in 0..c_list.len() {
            let c = c_list[(f0 + t) % c_list.len()];
            if c == 1 {
                pending += 1; // its single new vertex is also in the following face
                continue;
            }
            if t > 0 {
                next.push(pending);
            }
            next.extend(std::iter::repeat_n(1, c - 2));
            pending = 2;
        }
        next.push(pending);
        boundary = next;
    }
    Some(out)
}
