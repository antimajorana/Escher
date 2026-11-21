//! A finite defected crystal (`BuiltCrystal`) as a [`FiniteComplex`], with its rim (C1 WP-D1, plan §5 T7).
//!
//! Faces are the combinatorial face walks read from the bond reference frames
//! ([`crate::defect::charge::frame_faces`]). Every directed bond lies on exactly one walk, so the walks of a connected
//! planar crystal satisfy V − E + F = 2, and exactly one of them is the outer walk round the whole crystal. The outer
//! walk is taken to be the longest walk (plan §5 T7: "the boundary cycle ... is the longest such cycle"); every other
//! walk is a bounded face, and becomes one top cell of the complex.
//!
//! The rim is the set of vertices on the outer walk that lie in at least one bounded face. A dangling site on the outer
//! walk (a spur) lies in no bounded face and is left out, because no cell can contain it. Vertex degree is not used
//! to find the rim: it misfires at dislocation cores and cone sites.
//!
//! A cell is built from its walk as a table row (distinct vertices, distinct edge ids), not as a vertex cycle, so a walk
//! that visits a site twice (a spur inside a face) is kept instead of refused.

use std::collections::{HashMap, HashSet};

use crate::defect::charge::{frame_faces, outer_face_index};
use crate::defect::cone_geometry::add2;
use crate::types::{BuiltCrystal, Isometry2};

use super::finite::{FiniteComplex, FiniteTables};

/// A crystal's bounded faces as a complex. Cell `c` of [`CrystalComplex::complex`] is `faces[c]`; vertex ids are site
/// indices of the crystal.
#[derive(Clone, Debug)]
pub struct CrystalComplex {
    pub complex: FiniteComplex,
    /// Bounded face walks, in the frame-walk direction (opposite to the outer walk, so counter-clockwise in the
    /// convention of [`crate::defect::charge::total_charge`]).
    pub faces: Vec<Vec<usize>>,
    /// Rim sites (module docs), sorted.
    pub rim: Vec<usize>,
}

/// Build the complex of a planar, orientable, connected crystal. Refused (with the reason) for a non-orientable crystal,
/// a crystal whose walks do not give V − E + F = 2 (disconnected or not planar), or one with no bounded face.
pub fn crystal_complex(crystal: &BuiltCrystal) -> Result<CrystalComplex, String> {
    if crystal.is_non_orientable {
        return Err("growth needs an orientable crystal: a non-orientable seam has no consistent face walks".into());
    }
    let walks = frame_faces(crystal);
    let (v, e, f) = (crystal.pos.len() as i64, crystal.edges.len() as i64, walks.len() as i64);
    if v - e + f != 2 {
        return Err(format!(
            "the crystal's face walks give V - E + F = {} (a connected planar crystal gives 2); it is disconnected or not planar",
            v - e + f
        ));
    }
    if walks.len() < 2 {
        return Err("the crystal has no bounded face".into());
    }
    let outer = outer_face_index(&walks);
    let faces: Vec<Vec<usize>> = walks.iter().enumerate().filter(|&(i, _)| i != outer).map(|(_, w)| w.clone()).collect();
    let edge_id: HashMap<(usize, usize), u32> =
        crystal.edges.iter().enumerate().map(|(i, &(a, b))| ((a.min(b), a.max(b)), i as u32)).collect();
    let mut t = FiniteTables {
        edge_vertices: crystal.edges.iter().map(|&(a, b)| [a as u32, b as u32]).collect(),
        ..Default::default()
    };
    let mut in_face: HashSet<usize> = HashSet::new();
    for w in &faces {
        let (mut vs, mut es) = (Vec::new(), Vec::new());
        for k in 0..w.len() {
            let (a, b) = (w[k], w[(k + 1) % w.len()]);
            if !vs.contains(&(a as u32)) {
                vs.push(a as u32);
            }
            let id = *edge_id.get(&(a.min(b), a.max(b))).ok_or_else(|| format!("face walk uses a bond {a}-{b} missing from the edge list"))?;
            if !es.contains(&id) {
                es.push(id);
            }
            in_face.insert(a);
        }
        t.cell_vertices.push(vs);
        t.cell_edges.push(es);
    }
    let mut rim: Vec<usize> = walks[outer].iter().copied().filter(|s| in_face.contains(s)).collect();
    rim.sort_unstable();
    rim.dedup();
    t.rim = rim.iter().map(|&s| s as u32).collect();
    let complex = FiniteComplex::new(t)?;
    Ok(CrystalComplex { complex, faces, rim })
}

/// The bond from `u` to `v`.
fn bond(crystal: &BuiltCrystal, u: usize, v: usize) -> Option<&crate::types::DirectedBondData> {
    crystal.adj[u].iter().find(|b| b.to == v)
}

/// The polygon of face walk `walk` developed in the reference chart of its `base`-th site: start at that site's
/// `ref_pos` and add the bond vectors, carried back through each seam crossed (as in
/// [`crate::defect::charge::loop_charge`]). Unlike the raw `ref_pos` of the corners, this is one consistent picture
/// even for a face glued across a surgery cut.
pub fn develop_face(crystal: &BuiltCrystal, walk: &[usize], base: usize) -> Option<Vec<[f64; 2]>> {
    let n = walk.len();
    let mut p = crystal.ref_pos[walk[base]];
    let mut dm = Isometry2::IDENTITY;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        out.push(p);
        let (u, v) = (walk[(base + k) % n], walk[(base + k + 1) % n]);
        let e = bond(crystal, u, v)?;
        p = add2(p, dm.apply_linear(e.vec));
        dm = dm.compose(&e.seam.unwrap_or(Isometry2::IDENTITY).inverse());
    }
    ((p[0] - out[0][0]).hypot(p[1] - out[0][1]) <= 1e-5).then_some(out)
}

/// The boundary of a set of bounded faces, as one closed site loop in the face-walk direction (counter-clockwise in
/// the [`crate::defect::charge::total_charge`] convention). A directed bond is on the boundary when its reverse is in
/// no chosen face. Refused when the boundary is not a single simple loop (the set is not a disk, or is pinched).
pub fn boundary_loop(cc: &CrystalComplex, cells: &[u32]) -> Result<Vec<usize>, String> {
    let mut darts: HashSet<(usize, usize)> = HashSet::new();
    for &c in cells {
        let w = &cc.faces[c as usize];
        for k in 0..w.len() {
            darts.insert((w[k], w[(k + 1) % w.len()]));
        }
    }
    let mut next: HashMap<usize, usize> = HashMap::new();
    let mut count = 0usize;
    for &(a, b) in &darts {
        if !darts.contains(&(b, a)) {
            count += 1;
            if next.insert(a, b).is_some() {
                return Err(format!("site {a} has two boundary bonds out: the boundary is pinched"));
            }
        }
    }
    let Some(&start) = next.keys().min() else { return Err("the face set has no boundary".into()) };
    let mut lp = vec![start];
    let mut cur = next[&start];
    while cur != start {
        lp.push(cur);
        cur = *next.get(&cur).ok_or("the boundary does not close")?;
        if lp.len() > count {
            return Err("the boundary walk does not return to its start".into());
        }
    }
    if lp.len() != count {
        return Err(format!("the boundary has {count} bonds but its first loop has {}: it is not one loop", lp.len()));
    }
    Ok(lp)
}
