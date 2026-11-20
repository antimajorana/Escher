//! Boundary-condition builders added in C1 WP-B1 (plan §2.5, §5 T3): the cylinder on any non-zero circumference
//! `(n, m)`, the torus drawn on the Wigner-Seitz cell of its superlattice (3 pairs of sides), the open patch grown by
//! the touching rule, and the simplicity test of a quotient (`NOT FOUND (too small)`). Mirrored in Python by
//! `papertilling/surfaces/_flat_bc.py` (except the open patch: E² inflation is Rust-only, plan §0).

use std::collections::{BTreeMap, HashMap};

use crate::plane::inflation::{InflationRule, PlaneTiling, VertexId, inflate_with};
use crate::surface::flat::{cylinder, torus};
use crate::surface::glides::gauss_reduce;
use crate::surface::quotient::{
    CellGeom, add2, apply_to_site, dom_edge, gcd_i32, mat2_vec, mat2i_vec, plane_faces, sample_bond_curves, sub2,
    vec2_dot,
};
use crate::types::{Cell2D, SurfaceMesh};

/// Prefix of every refusal of a quotient that is not a simple graph (plan §2.5: never `IMPOSSIBLE`).
pub const TOO_SMALL: &str = "NOT FOUND (too small)";

/// `d` in the lattice spanned by `gens` (one or two integer vectors; two must be independent).
fn in_lattice(d: [i32; 2], gens: &[[i32; 2]]) -> bool {
    let cross = |u: [i32; 2], v: [i32; 2]| i64::from(u[0]) * i64::from(v[1]) - i64::from(u[1]) * i64::from(v[0]);
    match gens {
        [g] => {
            let gg = i64::from(g[0]) * i64::from(g[0]) + i64::from(g[1]) * i64::from(g[1]);
            let dg = i64::from(d[0]) * i64::from(g[0]) + i64::from(d[1]) * i64::from(g[1]);
            cross(d, *g) == 0 && gg > 0 && dg % gg == 0
        }
        [g0, g1] => {
            let det = cross(*g0, *g1);
            det != 0 && cross(d, *g1) % det == 0 && cross(*g0, d) % det == 0
        }
        _ => d == [0, 0],
    }
}

/// Simplicity of the quotient of the decorated plane by `Λ = span(gens)` and, when given, the glide coset
/// `g Λ` (`g = (m, t)` fractional, with `g² ∈ Λ` and `m Λ = Λ`): for every basis site `i`, no bond dart from
/// `(i, 0)` ends in the class of `(i, 0)` (a loop), and no two darts end in the same class (a double bond). This is
/// the infinite quotient (an infinite cylinder or Möbius band for one generator); a strip cut from it is refused
/// whenever the infinite quotient is not simple, which can only over-refuse. Exact integer tests, no tolerance.
pub fn simple_quotient(cell: &Cell2D, gens: &[[i32; 2]], glide: Option<([[i32; 2]; 2], [f64; 2])>) -> Result<(), String> {
    let geom = CellGeom::from_cell(cell)?;
    let image = |j: usize, d: [i32; 2]| -> Result<(usize, [i32; 2]), String> {
        let (m, t) = glide.expect("glide image asked without a glide");
        let (jj, c) = apply_to_site(m, t, &geom.frac_basis, j)
            .ok_or_else(|| "NOT FOUND: the gluing is not a symmetry of the cell".to_string())?;
        let md = mat2i_vec(m, d);
        Ok((jj, [md[0] + c[0], md[1] + c[1]]))
    };
    let same = |p: (usize, [i32; 2]), q: (usize, [i32; 2])| -> Result<bool, String> {
        if p.0 == q.0 && in_lattice([q.1[0] - p.1[0], q.1[1] - p.1[1]], gens) {
            return Ok(true);
        }
        if glide.is_some() {
            let gp = image(p.0, p.1)?;
            return Ok(gp.0 == q.0 && in_lattice([q.1[0] - gp.1[0], q.1[1] - gp.1[1]], gens));
        }
        Ok(false)
    };
    for i in 0..cell.basis.len() {
        let ends: Vec<(usize, [i32; 2])> = geom
            .bonds
            .iter()
            .flat_map(|&(p, q, n1, n2)| [(p, q, [n1, n2]), (q, p, [-n1, -n2])])
            .filter(|&(x, y, d)| x == i && !(x == y && d == [0, 0]))
            .map(|(_, y, d)| (y, d))
            .collect();
        for (k, &e) in ends.iter().enumerate() {
            if same((i, [0, 0]), e)? {
                return Err(format!("{TOO_SMALL}: a bond of site {i} closes into a loop on this quotient; take a larger one"));
            }
            for &f in &ends[k + 1..] {
                if same(e, f)? {
                    return Err(format!("{TOO_SMALL}: two bonds of site {i} join the same pair of sites on this quotient; take a larger one"));
                }
            }
        }
    }
    Ok(())
}

/// Extended Euclid: `(g, x, y)` with `a x + b y = g = gcd(a, b) >= 0`.
fn ext_gcd(a: i32, b: i32) -> (i32, i32, i32) {
    if b == 0 {
        return if a < 0 { (-a, -1, 0) } else { (a, 1, 0) };
    }
    let (g, x, y) = ext_gcd(b, a.rem_euclid(b));
    (g, y, x - a.div_euclid(b) * y)
}

/// The lattice vector `T` with `det(c / g, T) = 1` (so `(c / g, T)` is a basis of the lattice, `g = gcd(c)`),
/// chosen in its coset `T + Z c / g` as the most perpendicular to `c` (exact, no search window).
pub fn wrap_complement(a: [[f64; 2]; 2], c: [i32; 2]) -> Result<[i32; 2], String> {
    if c == [0, 0] {
        return Err("the circumference vector must not be zero".to_string());
    }
    let g = gcd_i32(c[0], c[1]).abs();
    let p = [c[0] / g, c[1] / g];
    // det(p, T) = p0 T1 - p1 T0 = 1: ext_gcd(p0, -p1) gives p0 x - p1 y = 1, so T = (y, x).
    let (one, x, y) = ext_gcd(p[0], -p[1]);
    debug_assert_eq!(one, 1);
    let mut t = [y, x];
    let pc = mat2_vec(a, [p[0] as f64, p[1] as f64]);
    let k = (vec2_dot(mat2_vec(a, [t[0] as f64, t[1] as f64]), pc) / vec2_dot(pc, pc)).round_ties_even() as i32;
    t = [t[0] - k * p[0], t[1] - k * p[1]];
    Ok(t)
}

/// The lattice rolled up along any non-zero circumference `wrap = (n, m)` (cell units, a nanotube's chiral vector),
/// `rows` complement steps long along the axis (the strip height is `rows` x cell area / |wrap| when wrap is
/// primitive, so a chiral wrap needs more rows), open at both ends (χ = 0, two boundary circles). Isometric picture.
pub fn cylinder_wrap(cell: &Cell2D, wrap: [i32; 2], rows: i32) -> Result<SurfaceMesh, String> {
    if rows < 1 {
        return Err(format!("rows must be >= 1 (got {rows})"));
    }
    let geom = CellGeom::from_cell(cell)?;
    let t = wrap_complement(geom.a, wrap)?;
    simple_quotient(cell, &[wrap], None)?;
    cylinder(cell, wrap, [rows * t[0], rows * t[1]])
}

/// Circumcentre of the triangle `0, a, b`: the point `c` with `c . a = |a|^2 / 2` and `c . b = |b|^2 / 2`.
fn circumcentre(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let det = a[0] * b[1] - a[1] * b[0];
    let (ra, rb) = (0.5 * vec2_dot(a, a), 0.5 * vec2_dot(b, b));
    [(ra * b[1] - rb * a[1]) / det, (a[0] * rb - b[0] * ra) / det]
}

/// Wigner-Seitz cell of the 2D lattice spanned by the Cartesian `v1, v2`: its vertices (counter-clockwise), the
/// edges as `(p, q, pair label 1..3)`, and `true` when it is a rectangle (the third pair has length 0).
pub fn wigner_seitz(v1: [f64; 2], v2: [f64; 2]) -> (Vec<[f64; 2]>, Vec<([f64; 2], [f64; 2], usize)>, bool) {
    let u = gauss_reduce([[v1[0], v2[0]], [v1[1], v2[1]]]);
    let cart = |k: usize| add2([v1[0] * u[0][k] as f64, v1[1] * u[0][k] as f64], [v2[0] * u[1][k] as f64, v2[1] * u[1][k] as f64]);
    let (w1, mut w2) = (cart(0), cart(1));
    if vec2_dot(w1, w2) > 0.0 {
        w2 = [-w2[0], -w2[1]];
    }
    let w12 = add2(w1, w2);
    let rect = vec2_dot(w1, w2).abs() <= 1e-9 * vec2_dot(w1, w1).max(vec2_dot(w2, w2));
    // Facet normals in angular order (w1 and w2 are at least 90 degrees apart, w1 + w2 lies between them).
    let mut facets: Vec<([f64; 2], usize)> = vec![(w1, 1), (w12, 3), (w2, 2), ([-w1[0], -w1[1]], 1), ([-w12[0], -w12[1]], 3), ([-w2[0], -w2[1]], 2)];
    if rect {
        facets.retain(|f| f.1 != 3);
    }
    if w1[0] * w2[1] - w1[1] * w2[0] < 0.0 {
        facets.reverse();
    }
    let n = facets.len();
    // Vertex between facets k and k+1; edge k runs from vertex k-1 to vertex k.
    let verts: Vec<[f64; 2]> = (0..n).map(|k| circumcentre(facets[k].0, facets[(k + 1) % n].0)).collect();
    let edges = (0..n).map(|k| (verts[(k + n - 1) % n], verts[k], facets[k].1)).collect();
    (verts, edges, rect)
}

/// The torus `R^2 / Λ'`, `Λ' = span(t1, t2)` (cell units), drawn on the Wigner-Seitz cell of `Λ'` with its three
/// pairs of sides identified by translations (plan §2.5): the same quotient as [`torus`] (same sites, bonds and
/// faces), only the drawn domain differs. Returns the mesh and `true` when that cell is a rectangle, so that the
/// three pairs reduce to two for this supercell.
pub fn torus_hex_domain(cell: &Cell2D, t1: [i32; 2], t2: [i32; 2]) -> Result<(SurfaceMesh, bool), String> {
    simple_quotient(cell, &[t1, t2], None)?;
    let mut mesh = torus(cell, t1, t2)?;
    let geom = CellGeom::from_cell(cell)?;
    let (v1, v2) = (mat2_vec(geom.a, [t1[0] as f64, t1[1] as f64]), mat2_vec(geom.a, [t2[0] as f64, t2[1] as f64]));
    let (poly, edges, rect) = wigner_seitz(v1, v2);
    let det = v1[0] * v2[1] - v1[1] * v2[0];
    let inv = [[v2[1] / det, -v2[0] / det], [-v1[1] / det, v1[0] / det]];
    let nearest = |p: [f64; 2]| -> [f64; 2] {
        let f = mat2_vec(inv, p);
        let mut best: Option<(f64, [f64; 2])> = None;
        for da in -2..=2 {
            for db in -2..=2 {
                let (ka, kb) = (f[0].round() + da as f64, f[1].round() + db as f64);
                let q = sub2(p, add2([ka * v1[0], ka * v1[1]], [kb * v2[0], kb * v2[1]]));
                let r = vec2_dot(q, q);
                if best.is_none_or(|(rb, _)| r < rb - 1e-9) {
                    best = Some((r, q));
                }
            }
        }
        best.map(|b| b.1).unwrap_or(p)
    };
    mesh.uv = mesh.uv.iter().map(|&p| nearest(p)).collect();
    mesh.name = "torus (hexagonal domain)".to_string();
    mesh.domain_polygons = vec![poly];
    mesh.domain_edges = edges.into_iter().map(|(p, q, lab)| dom_edge(p, q, &lab.to_string(), lab - 1)).collect();
    Ok((mesh, rect))
}

/// The open patch: the vertex star of the most symmetric vertex ([`PlaneTiling::default_seed`]) grown `layers`
/// times by the touching rule (BDF vertex completion, the WP-G3 E² grower), as a flat disk mesh. `chi` is the
/// measured `V - E + F` (1 for a disk; it is not assumed: the caller flags any other value).
pub fn open_patch(cell: &Cell2D, layers: usize) -> Result<SurfaceMesh, String> {
    let tiling = PlaneTiling::new(cell).map_err(|e| format!("NOT FOUND: the cell is not a plane tiling ({e})"))?;
    let seed = tiling.default_seed();
    let res = inflate_with(cell, seed, &InflationRule::Bdf, layers).map_err(|e| format!("NOT FOUND: {e}"))?;
    let geom = CellGeom::from_cell(cell)?;
    let (_, deg_basis) = plane_faces(cell, &geom)?;
    let mut index: HashMap<VertexId, usize> = HashMap::new();
    let (mut sites, mut uv) = (Vec::new(), Vec::new());
    let mut edge_faces: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut faces = Vec::with_capacity(res.faces.len());
    for f in &res.faces {
        let ids: Vec<usize> = tiling
            .face_vertices((f.orbit, f.offset))
            .into_iter()
            .zip(&f.polygon)
            .map(|(v, &p)| {
                *index.entry(v).or_insert_with(|| {
                    sites.push((v.0, v.1[0], v.1[1], 0));
                    uv.push(p);
                    uv.len() - 1
                })
            })
            .collect();
        for k in 0..ids.len() {
            let (a, b) = (ids[k], ids[(k + 1) % ids.len()]);
            *edge_faces.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
        faces.push(ids);
    }
    let bonds: Vec<(usize, usize, [f64; 2])> = edge_faces.keys().map(|&(a, b)| (a, b, sub2(uv[b], uv[a]))).collect();
    let mut rim: Vec<usize> = edge_faces.iter().filter(|e| *e.1 == 1).flat_map(|(&(a, b), _)| [a, b]).collect();
    rim.sort_unstable();
    rim.dedup();
    let plane_degree = sites.iter().map(|s| deg_basis[s.0]).collect();
    let xyz: Vec<[f64; 3]> = uv.iter().map(|p| [p[0], p[1], 0.0]).collect();
    let bond_curves_3d = sample_bond_curves(&uv, &xyz, &bonds, |p| [p[0], p[1], 0.0]);
    let chi = sites.len() as i32 - bonds.len() as i32 + faces.len() as i32;
    Ok(SurfaceMesh {
        name: format!("open patch ({layers} layers, seed {seed:?})"),
        sites,
        bonds,
        faces,
        plane_degree,
        edge_sites: rim,
        chi,
        orientable: true,
        closed: false,
        cones: Vec::new(),
        uv,
        domain_polygons: Vec::new(),
        domain_edges: Vec::new(),
        slits: Vec::new(),
        xyz,
        bond_curves_3d,
        cover: None,
    })
}
