//! Coset adapter for the regular honeycombs {p,q,r} of H³ and the six polychora of S³
//! (C1 WP-G2; plan §2.1-§2.3, the coset method of the growth map).
//!
//! Let W = ⟨R0,R1,R2,R3⟩ act on R⁴ (S³) or R^{3,1} (H³) as in [`ldl_realisation`] (accurate
//! Coxeter relations; its docs say why the eigen-based realisation is not used), with
//! fundamental points w0 (vertex), w1 (edge midpoint), w2 (face centre), w3 (cell centre).
//! Every cell is M·C0 for some M ∈ W, and the incidences are cosets of the stabilisers:
//! - the vertices, edges and faces of C0 are h·w0, h·w1, h·w2 for h ∈ ⟨R0,R1,R2⟩;
//! - the cells at the vertex w0 are h·w3 for h ∈ ⟨R1,R2,R3⟩ (c_v of them, finite unless the
//!   vertex figure {q,r} is Euclidean, i.e. the vertex is ideal);
//! - the r cells around the edge w1 are h·w3 for h ∈ ⟨R0,R2,R3⟩; the 2 cells on the face w2
//!   are w3 and R3·w3.
//!
//! Cells and vertices are interned by position with [`KeyIndex`] (tolerance [`KEY_TOL`], gap
//! probe 100 × tolerance): unit vectors in S³, spatial hyperboloid coordinates in H³
//! (`honeycomb3d::key_index` gives the [PROVED] reason they stay separated at depth), and
//! ideal vertices by their direction on the sphere at infinity. Points M·w are used as
//! computed, never renormalised: dividing by √(−⟨x,x⟩) costs ε·x₀² relative, which at
//! x₀ ~ 10⁴ was the measured error of 10⁻⁴. A gap below the probe turns every later query
//! into `GrowthError::Precision`.
//!
//! Edges and faces need no float keys: in a regular {p,q,r} with p, q, r ≥ 3 two vertices
//! span at most one edge and two distinct faces share at most an edge (faces are convex
//! polygons meeting face-to-face) [PROVED], so an edge is its vertex pair and a face is its
//! three smallest vertex ids.
//!
//! Memory: a cell is stored as a recipe (the vertex, edge or face it was found from, plus a
//! coset representative) and gets a matrix and its incidences only when the grower asks for
//! its vertices ("materialised"). The engine lists the cells at every vertex of each layer
//! (for the boundary counts); in its last step those are looked up, not interned
//! ([`CoxeterComplex::begin_growth`]).
//!
//! Infinite cells ({4,4,3}, {4,4,4}, {3,6,3}, {6,3,q}) are refused by [`CoxeterComplex::new`]
//! with [`INFINITE_CELL_REASON`]; ideal vertices answer `cells_at` with `Infinite`.

use std::cell::RefCell;

use super::engine::SeedAt;
use crate::honeycomb3d::key_index::{FastMap, KeyIndex};
use super::{CellComplex, GrowthCaps, GrowthError, SeedKind};
use crate::honeycomb3d::coxeter_gram::{classify_schlafli, form, identity4, mat_mul, mat_vec, HoneycombRegime, Mat4, Vec4};
use crate::honeycomb3d::coxeter_ldl::ldl_realisation;

/// Merge tolerance of the position keys (absolute; keys are O(1) apart, `honeycomb3d::key_index`).
pub const KEY_TOL: f64 = 1e-4;

/// [PROVED] reason for an ideal vertex (plan §1 Q4).
pub const IDEAL_VERTEX_REASON: &str = "[PROVED] an ideal vertex lies in infinitely many cells: its vertex figure {q,r} \
     is a Euclidean tiling ((q-2)(r-2) = 4)";
/// [PROVED] reason for an infinite cell (plan §1 Q4).
pub const INFINITE_CELL_REASON: &str = "[PROVED] every cell is an infinite horospherical {4,4} or {6,3} tiling \
     ((p-2)(q-2) = 4), so a cell has infinitely many vertices and no layer can be built whole";

/// The id returned for a cell of the engine's final look-ahead that is not interned
/// ([`CoxeterComplex::begin_growth`]).
pub const LOOKAHEAD: u32 = u32::MAX;

/// Face keys carry this bit; edge keys do not.
pub const FACE_TAG: u32 = 1 << 31;

#[derive(Clone, Copy)]
enum Src {
    Root,
    Vertex(u32),
    Edge(u32),
    Face(u32),
}

/// Coset representatives, as matrices and as the points they carry the base point to.
struct Reps {
    mats: Vec<Mat4>,
    pts: Vec<Vec4>,
}

struct Tables {
    w: [Vec4; 4],
    cell_v: Reps,
    cell_e: Reps,
    cell_f: Reps,
    /// For each face representative: indices into `cell_v`, in cyclic order.
    face_cycles: Vec<Vec<usize>>,
    /// For each edge representative: its two end indices into `cell_v`.
    edge_ends: Vec<(usize, usize)>,
    vert_c: Option<Reps>,
    edge_c: Reps,
    face_c: Reps,
}

struct Mat {
    m: Mat4,
    v: Vec<u32>,
    e: Vec<u32>,
    f: Vec<u32>,
}

struct State {
    cells: KeyIndex,
    cell_src: Vec<(Src, u16)>,
    mats: Vec<Option<Box<Mat>>>,
    verts: KeyIndex,
    vert_src: Vec<(u32, u16)>,
    edge_ids: FastMap<[u32; 2], u32>,
    edge_src: Vec<(u32, u16)>,
    face_src: Vec<(u32, u16)>,
    /// A face by its three smallest vertex ids.
    face_by_verts: FastMap<[u32; 3], u32>,
    /// Engine step tracking for the look-ahead (see [`CoxeterComplex::begin_growth`]).
    step: u32,
    querying: bool,
    final_step: Option<u32>,
    /// Once more than this many cells are interned, star queries answer by lookup (cap guard).
    cap: Option<usize>,
    force_lookup: bool,
}

impl State {
    /// A vertex/edge/face query after star queries starts the engine's next step.
    fn note_cell_query(&mut self) {
        if self.querying {
            self.step += 1;
            self.querying = false;
        }
    }
}

/// Precision readout of the four key indexes (§2.3).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct KeyPrecision {
    pub tol: f64,
    /// Largest distance between two merged representations (cells and vertices).
    pub max_merge: f64,
    /// Smallest distance below 100 × tol between distinct keys (`None`: there is none).
    pub min_gap: Option<f64>,
    /// Ids interned: cells (including the look-ahead layer), vertices, edges, faces.
    pub keys: [usize; 4],
}

/// The adapter. Cells, vertices and keys are interned `u32` ids.
pub struct CoxeterComplex {
    pub pqr: (u32, u32, u32),
    pub spherical: bool,
    pub ideal_vertices: bool,
    t: Tables,
    st: RefCell<State>,
}

fn finite_group(gens: &[Mat4], xi: &Vec4) -> Vec<Mat4> {
    // xi is an interior point of the chamber, so g ↦ g·xi is injective on the group.
    let mut idx = KeyIndex::new(4, 1e-7);
    idx.insert(*xi);
    let mut out = vec![identity4()];
    let mut i = 0;
    while i < out.len() {
        let g = out[i];
        for s in gens {
            let h = mat_mul(&g, s);
            if idx.insert(mat_vec(&h, xi)).1 {
                out.push(h);
            }
        }
        i += 1;
        assert!(out.len() <= 20_000, "a stabiliser of a compact or spherical {{p,q,r}} is finite");
    }
    out
}

impl CoxeterComplex {
    /// The adapter for {p,q,r}: S³, compact H³, or paracompact H³ with finite cells.
    pub fn new(p: u32, q: u32, r: u32) -> Result<Self, String> {
        let cls = classify_schlafli(p, q, r);
        let spherical = match cls.regime {
            HoneycombRegime::SphericalS3 => true,
            HoneycombRegime::HyperbolicCompactH3 | HoneycombRegime::HyperbolicParacompactH3 => false,
            HoneycombRegime::EuclideanE3 => {
                return Err("{4,3,4} is Euclidean: grow it with growth::e3_complex (cubic)".into());
            }
            _ => return Err(cls.reason),
        };
        if cls.has_infinite_euclidean_cells {
            return Err(INFINITE_CELL_REASON.into());
        }
        let real = ldl_realisation(p, q, r)?;
        let ideal = cls.has_ideal_vertices;
        let rr = &real.reflections;
        let xi: Vec4 = std::array::from_fn(|i| real.w.iter().map(|w| w[i]).sum::<f64>());
        let key = |x: &Vec4| point_key(spherical, ideal && form(&real.eta, x, x).abs() < 1e-9, x);
        let reps = |gens: &[Mat4], base: &Vec4| -> Reps {
            let mut idx = KeyIndex::new(4, 1e-7);
            let mut out = Reps { mats: Vec::new(), pts: Vec::new() };
            for g in finite_group(gens, &xi) {
                let x = mat_vec(&g, base);
                if idx.insert(key(&x)).1 {
                    out.mats.push(g);
                    out.pts.push(x);
                }
            }
            out
        };
        let w = real.w;
        let cell_g = [rr[0], rr[1], rr[2]];
        let cell_v = reps(&cell_g, &w[0]);
        let cell_e = reps(&cell_g, &w[1]);
        let cell_f = reps(&cell_g, &w[2]);
        let rot = mat_mul(&rr[0], &rr[1]);
        let mut local = KeyIndex::new(4, 1e-7);
        for x in &cell_v.pts {
            local.insert(key(x));
        }
        let face_cycles = cell_f
            .mats
            .iter()
            .map(|h| {
                let mut g = *h;
                (0..p)
                    .map(|_| {
                        let i = local.find(&key(&mat_vec(&g, &w[0]))).expect("face vertex is a cell vertex") as usize;
                        g = mat_mul(&g, &rot);
                        i
                    })
                    .collect()
            })
            .collect();
        let edge_ends = cell_e
            .mats
            .iter()
            .map(|h| {
                let end = |g: &Mat4| local.find(&key(&mat_vec(g, &w[0]))).expect("edge end is a cell vertex") as usize;
                (end(h), end(&mat_mul(h, &rr[0])))
            })
            .collect();
        let vert_c = (!cls.has_ideal_vertices).then(|| reps(&[rr[1], rr[2], rr[3]], &w[3]));
        let edge_c = reps(&[rr[0], rr[2], rr[3]], &w[3]);
        let face_c = reps(&[rr[0], rr[1], rr[3]], &w[3]);
        let t = Tables { w, cell_v, cell_e, cell_f, face_cycles, edge_ends, vert_c, edge_c, face_c };
        let dims = if spherical { 4 } else { 3 };
        let st = State {
            cells: KeyIndex::new(dims, KEY_TOL),
            cell_src: Vec::new(),
            mats: Vec::new(),
            verts: KeyIndex::new(dims, KEY_TOL),
            vert_src: Vec::new(),
            edge_ids: FastMap::default(),
            edge_src: Vec::new(),
            face_src: Vec::new(),
            face_by_verts: FastMap::default(),
            step: 0,
            querying: false,
            final_step: None,
            cap: None,
            force_lookup: false,
        };
        let out = Self { pqr: (p, q, r), spherical, ideal_vertices: cls.has_ideal_vertices, t, st: RefCell::new(st) };
        {
            let mut st = out.st.borrow_mut();
            let root = out.intern_cell(&mut st, &w[3], Src::Root, 0);
            debug_assert_eq!(root, 0);
            out.materialise(&mut st, 0);
        }
        Ok(out)
    }

    fn vkey(&self, x: &Vec4) -> Vec4 {
        point_key(self.spherical, self.ideal_vertices, x)
    }

    fn intern_cell(&self, st: &mut State, x: &Vec4, src: Src, rep: usize) -> u32 {
        let (i, new) = st.cells.insert(key_of(self.spherical, x));
        if new {
            st.cell_src.push((src, rep as u16));
        }
        i
    }

    fn cell_mat(&self, st: &mut State, c: u32) -> Mat4 {
        if let Some(Some(m)) = st.mats.get(c as usize) {
            return m.m;
        }
        let (src, j) = st.cell_src[c as usize];
        let j = j as usize;
        match src {
            Src::Root => identity4(),
            Src::Vertex(v) => {
                let n = self.vert_mat(st, v);
                mat_mul(&n, &self.t.vert_c.as_ref().expect("finite vertex star").mats[j])
            }
            Src::Edge(e) => mat_mul(&self.edge_mat(st, e), &self.t.edge_c.mats[j]),
            Src::Face(f) => mat_mul(&self.face_mat(st, f), &self.t.face_c.mats[j]),
        }
    }

    fn vert_mat(&self, st: &mut State, v: u32) -> Mat4 {
        let (c, i) = st.vert_src[v as usize];
        mat_mul(&self.cell_mat(st, c), &self.t.cell_v.mats[i as usize])
    }

    fn edge_mat(&self, st: &mut State, e: u32) -> Mat4 {
        let (c, i) = st.edge_src[e as usize];
        mat_mul(&self.cell_mat(st, c), &self.t.cell_e.mats[i as usize])
    }

    fn face_mat(&self, st: &mut State, f: u32) -> Mat4 {
        let (c, i) = st.face_src[f as usize];
        mat_mul(&self.cell_mat(st, c), &self.t.cell_f.mats[i as usize])
    }

    /// Give cell `c` a matrix and intern its vertices, edges and faces.
    fn materialise(&self, st: &mut State, c: u32) {
        if st.mats.get(c as usize).is_some_and(Option::is_some) {
            return;
        }
        let m = self.cell_mat(st, c);
        let mut v = Vec::with_capacity(self.t.cell_v.pts.len());
        for (i, x) in self.t.cell_v.pts.iter().enumerate() {
            let (id, new) = st.verts.insert(self.vkey(&mat_vec(&m, x)));
            if new {
                st.vert_src.push((c, i as u16));
            }
            v.push(id);
        }
        let mut e = Vec::with_capacity(self.t.edge_ends.len());
        for (i, &(a, b)) in self.t.edge_ends.iter().enumerate() {
            let key = [v[a].min(v[b]), v[a].max(v[b])];
            let next = st.edge_src.len() as u32;
            let id = *st.edge_ids.entry(key).or_insert(next);
            if id == next {
                st.edge_src.push((c, i as u16));
            }
            e.push(id);
        }
        let mut f = Vec::with_capacity(self.t.face_cycles.len());
        for (i, cyc) in self.t.face_cycles.iter().enumerate() {
            let next = st.face_src.len() as u32;
            let id = *st.face_by_verts.entry(three_smallest(cyc.iter().map(|&k| v[k]))).or_insert(next);
            if id == next {
                st.face_src.push((c, i as u16));
            }
            f.push(id);
        }
        if st.mats.len() <= c as usize {
            st.mats.resize_with(c as usize + 1, || None);
        }
        st.mats[c as usize] = Some(Box::new(Mat { m, v, e, f }));
    }

    fn check(st: &State) -> Result<(), GrowthError> {
        let gap = st.cells.min_gap.min(st.verts.min_gap);
        if gap.is_finite() { Err(GrowthError::Precision { gap }) } else { Ok(()) }
    }

    fn cells_from(&self, src: Src, m: &Mat4, reps: &Reps) -> Result<Vec<u32>, GrowthError> {
        let mut st = self.st.borrow_mut();
        Self::check(&st)?;
        st.querying = true;
        let over_cap = st.cap.is_some_and(|m| st.cells.len() > m);
        let out = if st.force_lookup || over_cap || st.final_step == Some(st.step) {
            let sp = self.spherical;
            reps.pts.iter().map(|x| st.cells.lookup(&key_of(sp, &mat_vec(m, x))).unwrap_or(LOOKAHEAD)).collect()
        } else {
            reps.pts.iter().enumerate().map(|(j, x)| self.intern_cell(&mut st, &mat_vec(m, x), src, j)).collect()
        };
        Self::check(&st)?;
        Ok(out)
    }

    /// Call right before [`super::grow`] with its `max_layers` (after the seed is listed).
    ///
    /// The engine lists the cells at every vertex of its last layer to count the boundary
    /// (`growth::engine` step 2), which in H³ is 30-40 times that layer. In the engine's step
    /// `max_layers` those cells are only tested for membership in S_k, so the adapter answers
    /// them by lookup, returning the placeholder id [`LOOKAHEAD`] (never a cell of S_k) for a
    /// cell not interned yet, instead of interning millions of cells. The step is tracked by
    /// the engine's call order: star queries, then the vertex/edge/face queries of the next
    /// layer. A test checks that the readout is the same with and without this.
    ///
    /// Cap guard: once more than `max_cells` cells are interned, star queries also answer by
    /// lookup. Every cell interned while building S_{k+1} lies in S_{k+1} (it is at a vertex or
    /// on an edge of S_k), so |S_{k+1}| > `max_cells` is already certain and the engine stops
    /// with the same `CellCap` and the same layer-k readout, without interning the rest of the
    /// over-cap layer (which in H³ can be millions of cells).
    pub fn begin_growth(&mut self, caps: GrowthCaps) {
        let st = self.st.get_mut();
        st.step = 0;
        st.querying = false;
        st.final_step = Some(caps.max_layers);
        st.cap = Some(caps.max_cells);
    }

    /// `cells_at` answered by lookup ([`LOOKAHEAD`] for a cell not interned): for boundary
    /// checks that only test membership (the face-shell rule).
    pub fn cells_at_lookup(&mut self, v: u32) -> Result<Vec<u32>, GrowthError> {
        self.st.get_mut().force_lookup = true;
        let out = self.cells_at(v);
        self.st.get_mut().force_lookup = false;
        out
    }

    /// `cells_on_edge` answered by lookup, as [`Self::cells_at_lookup`].
    pub fn cells_on_edge_lookup(&mut self, e: u32) -> Result<Vec<u32>, GrowthError> {
        self.st.get_mut().force_lookup = true;
        let out = self.cells_on_edge(e);
        self.st.get_mut().force_lookup = false;
        out
    }

    /// Stop answering by lookup (every later star query interns its cells).
    pub fn end_growth(&mut self) {
        let st = self.st.get_mut();
        st.final_step = None;
        st.cap = None;
    }

    /// The cells on the face with key `f` (a key from `faces_of`): 2 cells.
    pub fn cells_on_face_key(&mut self, f: u32) -> Result<Vec<u32>, GrowthError> {
        if f & FACE_TAG == 0 {
            return Err(GrowthError::OutOfRange { reason: "not a face key" });
        }
        let fid = f & !FACE_TAG;
        let m = {
            let mut st = self.st.borrow_mut();
            if fid as usize >= st.face_src.len() {
                return Err(GrowthError::OutOfRange { reason: "face not interned" });
            }
            self.face_mat(&mut st, fid)
        };
        self.cells_from(Src::Face(fid), &m, &self.t.face_c)
    }

    /// Seed position of a seed kind: the vertex w0, the edge w1, the face w2 or the cell C0
    /// (the cell C0 is centred at the origin of the ball in H³). Dirichlet is refused.
    pub fn default_seed(&self, seed: SeedKind) -> Result<SeedAt<u32, u32, u32>, String> {
        let st = self.st.borrow();
        let root = &st.mats[0].as_ref().expect("root cell");
        Ok(match seed {
            SeedKind::Vertex => SeedAt::Vertex(root.v[0]),
            SeedKind::Edge => SeedAt::Edge(root.e[0]),
            SeedKind::Face => SeedAt::Face(self.t.face_cycles[0].iter().map(|&k| root.v[k]).collect()),
            SeedKind::Cell => SeedAt::Cell(0),
            SeedKind::Dirichlet => {
                return Err("Dirichlet seed is not offered for {p,q,r}: the Dirichlet cell of a cell centre is \
                            the cell itself (use cell)"
                    .into())
            }
        })
    }

    /// Faces of a cell as cycles of vertex ids (the cell is materialised if needed).
    pub fn face_cycles_of(&self, c: u32) -> Vec<Vec<u32>> {
        let mut st = self.st.borrow_mut();
        self.materialise(&mut st, c);
        let v = &st.mats[c as usize].as_ref().expect("materialised").v;
        self.t.face_cycles.iter().map(|cyc| cyc.iter().map(|&k| v[k]).collect()).collect()
    }

    /// Ambient coordinates of a vertex: a unit vector (S³), (x₀, X) with ⟨x,x⟩ = −1 (H³), or
    /// (1, s) with |s| = 1 for an ideal vertex.
    pub fn vertex_coords4(&self, v: u32) -> Vec4 {
        let k = self.st.borrow().verts.point(v);
        if self.spherical {
            k
        } else if self.ideal_vertices {
            [1.0, k[0], k[1], k[2]]
        } else {
            [(1.0 + k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt(), k[0], k[1], k[2]]
        }
    }

    /// The fundamental points w0..w3 of the realisation (S³ uses them to pick a pole).
    pub fn fundamental_points(&self) -> [Vec4; 4] {
        self.t.w
    }

    pub fn precision(&self) -> KeyPrecision {
        let st = self.st.borrow();
        let all = [&st.cells, &st.verts];
        let gap = all.iter().map(|i| i.min_gap).fold(f64::INFINITY, f64::min);
        KeyPrecision {
            tol: KEY_TOL,
            max_merge: all.iter().map(|i| i.max_merge).fold(0.0, f64::max),
            min_gap: gap.is_finite().then_some(gap),
            keys: [st.cells.len(), st.verts.len(), st.edge_src.len(), st.face_src.len()],
        }
    }

    /// (vertices, edges, faces) of one cell and c_v (`None` for an ideal vertex): for checks.
    pub fn local_counts(&self) -> (usize, usize, usize, Option<usize>) {
        let t = &self.t;
        (t.cell_v.pts.len(), t.cell_e.pts.len(), t.cell_f.pts.len(), t.vert_c.as_ref().map(|r| r.pts.len()))
    }
}

fn three_smallest(it: impl Iterator<Item = u32>) -> [u32; 3] {
    let mut v: Vec<u32> = it.collect();
    v.sort_unstable();
    [v[0], v[1], v[2]]
}

/// Key of an ideal (`ideal`) or ordinary point: its direction on the sphere at infinity, or
/// [`key_of`].
fn point_key(spherical: bool, ideal: bool, x: &Vec4) -> Vec4 {
    if spherical || !ideal {
        return key_of(spherical, x);
    }
    let s = [x[1] / x[0], x[2] / x[0], x[3] / x[0]];
    let n = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    [s[0] / n, s[1] / n, s[2] / n, 0.0]
}

fn key_of(spherical: bool, x: &Vec4) -> Vec4 {
    if spherical {
        let n = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]).sqrt();
        x.map(|c| c / n)
    } else {
        [x[1], x[2], x[3], 0.0]
    }
}

impl CellComplex for CoxeterComplex {
    type Cell = u32;
    type Vertex = u32;
    type Key = u32;

    fn vertices_of(&self, c: u32) -> Vec<u32> {
        let mut st = self.st.borrow_mut();
        st.note_cell_query();
        self.materialise(&mut st, c);
        st.mats[c as usize].as_ref().expect("materialised").v.clone()
    }
    fn edges_of(&self, c: u32) -> Vec<u32> {
        let mut st = self.st.borrow_mut();
        st.note_cell_query();
        self.materialise(&mut st, c);
        st.mats[c as usize].as_ref().expect("materialised").e.clone()
    }
    fn faces_of(&self, c: u32) -> Vec<u32> {
        let mut st = self.st.borrow_mut();
        st.note_cell_query();
        self.materialise(&mut st, c);
        st.mats[c as usize].as_ref().expect("materialised").f.iter().map(|f| f | FACE_TAG).collect()
    }
    fn cells_at(&mut self, v: u32) -> Result<Vec<u32>, GrowthError> {
        let Some(reps) = self.t.vert_c.as_ref() else {
            return Err(GrowthError::Infinite { reason: IDEAL_VERTEX_REASON });
        };
        let m = {
            let mut st = self.st.borrow_mut();
            if v as usize >= st.vert_src.len() {
                return Err(GrowthError::OutOfRange { reason: "vertex not interned" });
            }
            self.vert_mat(&mut st, v)
        };
        self.cells_from(Src::Vertex(v), &m, reps)
    }
    fn cells_on_edge(&mut self, e: u32) -> Result<Vec<u32>, GrowthError> {
        let m = {
            let mut st = self.st.borrow_mut();
            if e & FACE_TAG != 0 || e as usize >= st.edge_src.len() {
                return Err(GrowthError::OutOfRange { reason: "not an interned edge key" });
            }
            self.edge_mat(&mut st, e)
        };
        self.cells_from(Src::Edge(e), &m, &self.t.edge_c)
    }
    fn cells_on_face(&mut self, f: &[u32]) -> Result<Vec<u32>, GrowthError> {
        if f.len() < 3 {
            return Err(GrowthError::OutOfRange { reason: "a face needs at least 3 vertices" });
        }
        let id = {
            let st = self.st.borrow();
            st.face_by_verts.get(&three_smallest(f.iter().copied())).copied().filter(|&fid| {
                let (c, i) = st.face_src[fid as usize];
                let v = &st.mats[c as usize].as_ref().expect("materialised").v;
                let mut have: Vec<u32> = self.t.face_cycles[i as usize].iter().map(|&k| v[k]).collect();
                let mut want = f.to_vec();
                have.sort_unstable();
                want.sort_unstable();
                have == want
            })
        };
        match id {
            Some(fid) => self.cells_on_face_key(fid | FACE_TAG),
            None => Err(GrowthError::OutOfRange { reason: "no interned face on these vertices" }),
        }
    }
}
