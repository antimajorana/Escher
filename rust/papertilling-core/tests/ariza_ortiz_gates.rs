use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Instant;

use faer::linalg::solvers::DenseSolveCore;
use faer::{Mat, Side};
use papertilling_core::defect::catalogue::DefectCatalogue;
use papertilling_core::defect::core_graph::LatticeCoreInfo;
use papertilling_core::defect::gauge::{
    SpaceGroupExtension, apply_gauge_transform, fusion_channels, plaquette_holonomies,
    ribbon_faces, verify_d_r_squared, wyckoff_rotocentres,
};
use papertilling_core::defect::gsv_elastic::{gsv_eval_2d, gsv_hessian_2d, relax_elastic_2d};
use papertilling_core::defect::gsv_mobius::min_twisted_hinge_alignment;
use papertilling_core::defect::sheet3d::{Sheet3DOptions, relax_sheet_3d};
use papertilling_core::defect::surgery::{DefectSpec, build_point_defects, build_point_defects_with_cuts};
use papertilling_core::defect::wall_build::build_mobius_seam;
use papertilling_core::plane::catalogue::parse_named_str;
use papertilling_core::types::{BuiltCrystal, Cell2D, DirectedBondData, Frame2D, Isometry2};

const NAMED_JSON: &str = include_str!("../../../papertilling/data/named.json");

fn named_cell(key: &str) -> (String, Cell2D, Frame2D) {
    let cat = parse_named_str(NAMED_JSON).expect("valid named.json");
    let nl = cat.named.into_iter().find(|n| n.key == key).expect("named lattice exists");
    (nl.group, nl.cell, nl.frame)
}

fn induce_subset(c: &BuiltCrystal, keep: &[bool]) -> BuiltCrystal {
    let mut old_to_new = vec![usize::MAX; c.pos.len()];
    let new_to_old: Vec<usize> = keep.iter().enumerate().filter(|&(_, &k)| k).map(|(u, _)| u).collect();
    for (nu, &ou) in new_to_old.iter().enumerate() {
        old_to_new[ou] = nu;
    }
    let mut edges = Vec::new();
    let mut adj = vec![Vec::new(); new_to_old.len()];
    for (nu, &ou) in new_to_old.iter().enumerate() {
        for e in &c.adj[ou] {
            let nv = old_to_new[e.to];
            if nv != usize::MAX {
                let mut e2 = e.clone();
                e2.to = nv;
                adj[nu].push(e2);
                if nu < nv { edges.push((nu, nv)); }
            }
        }
    }
    BuiltCrystal {
        pos: new_to_old.iter().map(|&u| c.pos[u]).collect(),
        ref_pos: new_to_old.iter().map(|&u| c.ref_pos[u]).collect(),
        basis_idx: new_to_old.iter().map(|&u| c.basis_idx[u]).collect(),
        side: new_to_old.iter().map(|&u| c.side[u]).collect(),
        cone_sites: c.cone_sites.iter().filter_map(|&u| (old_to_new[u] != usize::MAX).then_some(old_to_new[u])).collect(),
        edges,
        adj,
        wall_sigma: c.wall_sigma,
        is_non_orientable: c.is_non_orientable,
    }
}

fn square_cone_7v(cell: &Cell2D) -> BuiltCrystal {
    let full = build_point_defects(cell, &[DefectSpec::disclination(90.0, [0.0, 0.0])], Some(4.0), None).expect("+90 square cone");
    let c = full.cone_sites[0];
    let mut keep = vec![false; full.pos.len()];
    ribbon_faces(&full).into_iter().filter(|f| f.len() == 4 && f.contains(&c)).flatten().for_each(|u| keep[u] = true);
    induce_subset(&full, &keep)
}

fn kabsch_max_err<const D: usize>(a: &[[f64; D]], b: &[[f64; D]], perm: &[usize]) -> f64 {
    let n = a.len() as f64;
    let (mut ca, mut cb) = ([0.0; D], [0.0; D]);
    for (u, pa) in a.iter().enumerate() {
        for k in 0..D { ca[k] += pa[k] / n; cb[k] += b[perm[u]][k] / n; }
    }
    let mut cov = Mat::<f64>::zeros(D, D);
    for (u, pa) in a.iter().enumerate() {
        for r in 0..D { for c in 0..D { cov[(r, c)] += (pa[r] - ca[r]) * (b[perm[u]][c] - cb[c]); } }
    }
    let svd = cov.svd().expect("svd");
    let (u_m, v_m) = (svd.U(), svd.V());
    let r_raw = u_m * v_m.transpose();
    let det = if D == 2 {
        r_raw[(0, 0)] * r_raw[(1, 1)] - r_raw[(0, 1)] * r_raw[(1, 0)]
    } else {
        r_raw[(0, 0)] * (r_raw[(1, 1)] * r_raw[(2, 2)] - r_raw[(1, 2)] * r_raw[(2, 1)])
            - r_raw[(0, 1)] * (r_raw[(1, 0)] * r_raw[(2, 2)] - r_raw[(1, 2)] * r_raw[(2, 0)])
            + r_raw[(0, 2)] * (r_raw[(1, 0)] * r_raw[(2, 1)] - r_raw[(1, 1)] * r_raw[(2, 0)])
    };
    let mut d_pos = Mat::<f64>::identity(D, D);
    if det < 0.0 { d_pos[(D - 1, D - 1)] = -1.0; }
    let align_err = |rot: &Mat<f64>| {
        a.iter().enumerate().map(|(u, pa)| {
            let pb = b[perm[u]];
            (0..D).map(|r| (pa[r] - ca[r] - (0..D).map(|c| rot[(r, c)] * (pb[c] - cb[c])).sum::<f64>()).powi(2)).sum::<f64>().sqrt()
        }).fold(0.0_f64, f64::max)
    };
    let err_pos = align_err(&(u_m * d_pos * v_m.transpose()));
    if D == 3 {
        // Also allow proper/improper z <-> -z buckling reflection in 3D when matching unconstrained sheets.
        let mut d_neg = Mat::<f64>::identity(D, D);
        if det >= 0.0 { d_neg[(D - 1, D - 1)] = -1.0; }
        let err_neg = align_err(&(u_m * d_neg * v_m.transpose()));
        return err_pos.min(err_neg);
    }
    err_pos
}

fn find_graph_iso(a: &BuiltCrystal, b: &BuiltCrystal) -> Option<Vec<usize>> {
    if a.pos.len() != b.pos.len() || a.edges.len() != b.edges.len() { return None; }
    use petgraph::algo::subgraph_isomorphisms_iter;
    use petgraph::graph::DiGraph;
    let to_pg = |c: &BuiltCrystal| {
        let mut g = DiGraph::<usize, ()>::new();
        let nodes: Vec<_> = (0..c.pos.len()).map(|u| g.add_node(c.basis_idx[u])).collect();
        for &(u, v) in &c.edges { g.add_edge(nodes[u], nodes[v], ()); g.add_edge(nodes[v], nodes[u], ()); }
        g
    };
    let (ga, gb) = (to_pg(a), to_pg(b));
    let mut nm = |x: &usize, y: &usize| x == y;
    let mut em = |_: &(), _: &()| true;
    let (ra, rb) = (&ga, &gb);
    let mut it = subgraph_isomorphisms_iter(&ra, &rb, &mut nm, &mut em)?;
    it.next()
}

#[test]
fn gate1_seven_vertex_square_cone_exact() {
    let t0 = Instant::now();
    let (_, cell, _) = named_cell("square");
    let cone7 = square_cone_7v(&cell);
    assert_eq!((cone7.pos.len(), cone7.edges.len(), cone7.cone_sites.len()), (7, 9, 1));

    let res2 = relax_elastic_2d(&cone7).expect("2D relax");
    assert!(res2.converged);
    let (e_eval, grad) = gsv_eval_2d(&cone7, &res2.pos).expect("eval");
    let g_inf = grad.iter().flatten().fold(0.0_f64, |m, &x| m.max(x.abs()));
    assert!(g_inf < 1e-11 && (e_eval - res2.total_energy).abs() < 1e-13);

    let hess = gsv_hessian_2d(&cone7, &res2.pos).expect("hess");
    let eig = hess.self_adjoint_eigen(Side::Lower).expect("eigen");
    let mut vals: Vec<f64> = (0..14).map(|i| eig.S().column_vector()[i]).collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    assert_eq!(vals.iter().filter(|&&l| l.abs() < 1e-12).count(), 3);
    assert!(vals[3] > 1e-3);

    let r0 = relax_sheet_3d(
        &cone7,
        &cell,
        &Sheet3DOptions { kappa_b: 0.0, max_iters: 1500, grad_tol: 1e-12 },
        None,
    );
    assert!(r0.stretch_energy < 1e-12);
    let totals: Vec<f64> = [1e-4, 1e-3, 1e-2, 1e-1]
        .iter()
        .map(|&kb| {
            let rk = relax_sheet_3d(&cone7, &cell, &Sheet3DOptions { kappa_b: kb, max_iters: 1500, grad_tol: 1e-11 }, None);
            rk.stretch_energy + rk.bend_energy
        })
        .collect();
    assert!(totals.windows(2).all(|w| w[1] > w[0]));
    println!(
        "# time Gate 1: {:.2} ms | E2D={:.6} ||g||_inf={g_inf:.2e} max3|lam|={:.2e} E3D(kb=0)={:.2e} totals={totals:.5?}",
        t0.elapsed().as_secs_f64() * 1e3,
        res2.total_energy,
        vals.iter().take(3).fold(0.0_f64, |m, &v| m.max(v.abs())),
        r0.stretch_energy,
    );
}

#[test]
fn gate2_gauge_and_cut_routing_invariance() {
    let t0 = Instant::now();
    let (_, sq_cell, _) = named_cell("square");
    let (p4g_grp, p4g_cell, p4g_frame) = named_cell("shastry_sutherland");
    let p4g_ext = SpaceGroupExtension::from_cell(&p4g_grp, &p4g_cell).expect("p4g ext");
    let sq_ext = SpaceGroupExtension::from_cell("p4m", &sq_cell).expect("p4m ext");
    let p4g_cat = DefectCatalogue::new(&p4g_grp, &p4g_cell, p4g_frame);
    let p4g_centre = p4g_cat.centre_classes.iter().find(|c| c.order == 4).expect("4-fold").point;
    let p4g_disloc_b = [p4g_cell.a1[0], p4g_cell.a1[1]];

    let disloc = build_point_defects(&sq_cell, &[DefectSpec::dislocation([1.0, 0.0], Some([0.5, 0.5]))], Some(5.0), None).unwrap();
    let dipole = build_point_defects(&sq_cell, &[DefectSpec::disclination(90.0, [0.0, 0.0]), DefectSpec::disclination(-90.0, [1.5, 0.5])], Some(6.0), None).unwrap();
    let p4g_def = build_point_defects(&p4g_cell, &[DefectSpec::disclination(90.0, p4g_centre), DefectSpec::dislocation(p4g_disloc_b, Some([p4g_centre[0] + 1.5, p4g_centre[1]]))], Some(6.5), None).unwrap();

    // Verify nontrivial cut rays ALSO preserve crystal invariance when graph is isomorphic
    let cut_a = build_point_defects_with_cuts(&sq_cell, &[DefectSpec::Dislocation { b: [1.0, 0.0], at: Some([0.5, 0.5]), cut: Some([1.0, 0.25]) }], Some(5.0), None).unwrap().0;
    let cut_b = build_point_defects_with_cuts(&sq_cell, &[DefectSpec::Dislocation { b: [1.0, 0.0], at: Some([0.5, 0.5]), cut: Some([1.0, -0.25]) }], Some(5.0), None).unwrap().0;
    if let Some(perm) = find_graph_iso(&cut_a, &cut_b) {
        let ra = relax_elastic_2d(&cut_a).unwrap();
        let rb = relax_elastic_2d(&cut_b).unwrap();
        let err = kabsch_max_err(&ra.pos, &rb.pos, &perm);
        assert!(err < 1e-8, "cut-ray invariance error {err:.2e}");
    }

    let cases: [(&str, &BuiltCrystal, &Cell2D, &SpaceGroupExtension); 3] = [
        ("sq_disloc", &disloc, &sq_cell, &sq_ext),
        ("sq_dipole", &dipole, &sq_cell, &sq_ext),
        ("p4g_defect", &p4g_def, &p4g_cell, &p4g_ext),
    ];
    for (label, c1, cell, ext) in cases {
        let gauges: Vec<Isometry2> = (0..c1.pos.len())
            .map(|u| {
                let r_idx = ((u * 5 + 3) * 7) % ext.point_group.len();
                let lx = ((u as i32) % 3) - 1;
                let ly = (((u / 3) as i32) % 3) - 1;
                let el = ext.element([lx, ly], ext.point_group[r_idx]).unwrap();
                ext.to_isometry(&el)
            })
            .collect();
        let c2 = apply_gauge_transform(c1, &gauges);
        let perm: Vec<usize> = (0..c1.pos.len()).collect();

        let r1 = relax_elastic_2d(c1).unwrap();
        let r2 = relax_elastic_2d(&c2).unwrap();
        let err2 = kabsch_max_err(&r1.pos, &r2.pos, &perm);
        let esite2 = r1.site_gsv_energy.iter().zip(&r2.site_gsv_energy).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        let ebond2 = r1.bond_gsv_energy.iter().zip(&r2.bond_gsv_energy).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(r1.converged && r2.converged && err2 < 1e-8 && esite2 < 1e-10 && ebond2 < 1e-10,
            "{label} 2D gauge mismatch: err={err2:.2e}, esite={esite2:.2e}, ebond={ebond2:.2e}");

        let s_opts = Sheet3DOptions { kappa_b: 0.02, max_iters: 5000, grad_tol: 1e-13 };
        let s1 = relax_sheet_3d(c1, cell, &s_opts, None);
        let s2 = relax_sheet_3d(&c2, cell, &s_opts, None);
        let err3 = kabsch_max_err(&s1.xyz, &s2.xyz, &perm);
        let esite3 = s1.site_gsv_energy.iter().zip(&s2.site_gsv_energy).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        let ebond3 = s1.bond_gsv_energy.iter().zip(&s2.bond_gsv_energy).fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        println!("# Gate 2 {label}: err2D={err2:.2e} err3D={err3:.2e} it=({},{}) conv=({},{}) dSite={esite2:.2e}/{esite3:.2e}",
            s1.iterations, s2.iterations, s1.converged, s2.converged);
        assert!(s1.converged && s2.converged && err3 < 1e-8 && esite3 < 1e-10 && ebond3 < 1e-10,
            "{label} 3D gauge mismatch: conv=({},{}) err={err3:.2e}, esite={esite3:.2e}, ebond={ebond3:.2e}",
            s1.converged, s2.converged);
    }
    println!("# time Gate 2: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
}

fn pristine_disk(cell: &Cell2D, radius: f64) -> BuiltCrystal {
    let info = LatticeCoreInfo::new(cell);
    let keys = info.lat.sites_in_disk([0.0, 0.0], radius);
    let idx_map: HashMap<_, usize> = keys.iter().copied().enumerate().map(|(i, k)| (k, i)).collect();
    let mut adj = vec![Vec::new(); keys.len()];
    let mut edges = Vec::new();
    for (i, &k) in keys.iter().enumerate() {
        for (k2, kind) in info.lat.nbrs(k) {
            if let Some(&j) = idx_map.get(&k2) {
                if i < j { edges.push((i, j)); }
                let (p, q) = (info.lat.pos(k), info.lat.pos(k2));
                adj[i].push(DirectedBondData { to: j, vec: [q[0] - p[0], q[1] - p[1]], kind, seam: None, ang: None, across: false, ghost_stub: None });
            }
        }
    }
    let pos: Vec<[f64; 2]> = keys.iter().map(|&k| info.lat.pos(k)).collect();
    BuiltCrystal { pos: pos.clone(), ref_pos: pos, basis_idx: keys.iter().map(|&k| k.2).collect(), side: vec![None; keys.len()], cone_sites: Vec::new(), edges, adj, wall_sigma: None, is_non_orientable: false }
}

fn continuum_c_voigt(cell: &Cell2D) -> [[f64; 3]; 3] {
    let c = pristine_disk(cell, 5.5);
    let keys = LatticeCoreInfo::new(cell).lat.sites_in_disk([0.0, 0.0], 5.5);
    let center_sites: Vec<usize> = keys.iter().enumerate().filter(|&(_, &(n1, n2, _))| n1 == 0 && n2 == 0).map(|(i, _)| i).collect();
    let area = (cell.a1[0] * cell.a2[1] - cell.a1[1] * cell.a2[0]).abs();
    let dim_xi = 2 * cell.basis.len().saturating_sub(1);
    let n_var = 3 + dim_xi;
    let eval_w = |v: &[f64]| -> f64 {
        let eps = [[v[0], 0.5 * v[2]], [0.5 * v[2], v[1]]];
        let pos: Vec<[f64; 2]> = c.pos.iter().enumerate().map(|(u, &p)| {
            let bi = c.basis_idx[u];
            let sh = if bi > 0 { [v[3 + 2 * (bi - 1)], v[3 + 2 * (bi - 1) + 1]] } else { [0.0, 0.0] };
            [p[0] + eps[0][0] * p[0] + eps[0][1] * p[1] + sh[0], p[1] + eps[1][0] * p[0] + eps[1][1] * p[1] + sh[1]]
        }).collect();
        let mut w = 0.0;
        for &u in &center_sites {
            for b in &c.adj[u] {
                let l0 = b.vec[0].hypot(b.vec[1]);
                let d = [pos[b.to][0] - pos[u][0], pos[b.to][1] - pos[u][1]];
                w += 0.25 * ((d[0] * d[0] + d[1] * d[1] - l0 * l0) / (2.0 * l0)).powi(2);
            }
            for ai in 0..c.adj[u].len() {
                let ea = &c.adj[u][ai];
                let da = [pos[ea.to][0] - pos[u][0], pos[ea.to][1] - pos[u][1]];
                for eb in &c.adj[u][(ai + 1)..] {
                    let db = [pos[eb.to][0] - pos[u][0], pos[eb.to][1] - pos[u][1]];
                    let k_is = 1.0 / (ea.vec[0].hypot(ea.vec[1]) * eb.vec[0].hypot(eb.vec[1])).sqrt();
                    let r = 0.5 * k_is * (da[0] * db[0] + da[1] * db[1] - (ea.vec[0] * eb.vec[0] + ea.vec[1] * eb.vec[1]));
                    w += 0.5 * r * r;
                }
            }
        }
        w / area
    };
    let h = 1e-5;
    let w0 = eval_w(&vec![0.0; n_var]);
    let mut k_mat = Mat::<f64>::zeros(n_var, n_var);
    for i in 0..n_var {
        let (mut vp, mut vm) = (vec![0.0; n_var], vec![0.0; n_var]);
        vp[i] = h; vm[i] = -h;
        k_mat[(i, i)] = (eval_w(&vp) - 2.0 * w0 + eval_w(&vm)) / (h * h);
        for j in (i + 1)..n_var {
            let (mut pp, mut pm, mut mp, mut mm) = (vec![0.0; n_var], vec![0.0; n_var], vec![0.0; n_var], vec![0.0; n_var]);
            pp[i] = h; pp[j] = h; pm[i] = h; pm[j] = -h; mp[i] = -h; mp[j] = h; mm[i] = -h; mm[j] = -h;
            let hij = (eval_w(&pp) - eval_w(&pm) - eval_w(&mp) + eval_w(&mm)) / (4.0 * h * h);
            k_mat[(i, j)] = hij; k_mat[(j, i)] = hij;
        }
    }
    let mut c_out = [[0.0; 3]; 3];
    if dim_xi == 0 {
        for i in 0..3 { for j in 0..3 { c_out[i][j] = k_mat[(i, j)]; } }
        return c_out;
    }
    let c_eff = k_mat.submatrix(0, 0, 3, 3).to_owned() - k_mat.submatrix(0, 3, 3, dim_xi) * k_mat.submatrix(3, 3, dim_xi, dim_xi).to_owned().partial_piv_lu().inverse() * k_mat.submatrix(0, 3, 3, dim_xi).transpose();
    for i in 0..3 { for j in 0..3 { c_out[i][j] = c_eff[(i, j)]; } }
    c_out
}

#[test]
fn gate3_ariza_ortiz_linear_dislocation_limit() {
    let t0 = Instant::now();
    for key in ["square", "honeycomb"] {
        let (_, cell, _) = named_cell(key);
        let patch = pristine_disk(&cell, 4.0);
        let n = patch.pos.len();
        let u_disp: Vec<[f64; 2]> = (0..n).map(|i| [(i as f64 * 0.71).sin(), (i as f64 * 1.13).cos()]).collect();
        let eps = 1e-5;
        let pos_p: Vec<[f64; 2]> = patch.pos.iter().zip(&u_disp).map(|(p, d)| [p[0] + eps * d[0], p[1] + eps * d[1]]).collect();
        let pos_m: Vec<[f64; 2]> = patch.pos.iter().zip(&u_disp).map(|(p, d)| [p[0] - eps * d[0], p[1] - eps * d[1]]).collect();
        let (e_p, _) = gsv_eval_2d(&patch, &pos_p).unwrap();
        let (e_m, _) = gsv_eval_2d(&patch, &pos_m).unwrap();
        let h_exact = gsv_hessian_2d(&patch, &patch.pos).unwrap();
        let mut u_h_u = 0.0;
        for i in 0..n { for j in 0..n { for a in 0..2 { for b in 0..2 { u_h_u += u_disp[i][a] * h_exact[(2 * i + a, 2 * j + b)] * u_disp[j][b]; } } } }
        let sym_quad_err = ((e_p + e_m) / (eps * eps) - u_h_u).abs();
        assert!(sym_quad_err < 1e-7 && ((2.0 * e_p / (eps * eps) - u_h_u).abs() / u_h_u) < 1e-4, "{key} quad err sym={sym_quad_err:.2e}");

        let fd_h = 1e-6;
        let (mut p_fwd, mut p_bwd, k_mid) = (patch.pos.clone(), patch.pos.clone(), n / 2);
        p_fwd[k_mid][0] += fd_h; p_bwd[k_mid][0] -= fd_h;
        let (_, gf) = gsv_eval_2d(&patch, &p_fwd).unwrap();
        let (_, gb) = gsv_eval_2d(&patch, &p_bwd).unwrap();
        let max_fd_err = (0..n).map(|i| ((gf[i][0] - gb[i][0]) / (2.0 * fd_h) - h_exact[(2 * i, 2 * k_mid)]).abs()).fold(0.0_f64, f64::max);
        assert!(max_fd_err < 1e-9, "{key} FD Hessian err={max_fd_err:.2e}");

        let voigt = continuum_c_voigt(&cell);
        let (mut min_gam, mut k_dip, b, n_ang) = (f64::INFINITY, 0.0, [cell.a1[0], cell.a1[1]], 120usize);
        let dth = 2.0 * PI / (n_ang as f64);
        let c_t = |i, j, k, l| voigt[match (i, j) { (0, 0) => 0, (1, 1) => 1, _ => 2 }][match (k, l) { (0, 0) => 0, (1, 1) => 1, _ => 2 }];
        for s in 0..n_ang {
            let th = (s as f64 + 0.5) * dth;
            let (nv, mv) = ([th.cos(), th.sin()], [-th.sin(), th.cos()]);
            let (mut gam, mut bmat, mut mmat) = ([[0.0; 2]; 2], [[0.0; 2]; 2], [[0.0; 2]; 2]);
            for i in 0..2 { for k in 0..2 { for j in 0..2 { for l in 0..2 {
                let c = c_t(i, j, k, l);
                gam[i][k] += c * nv[j] * nv[l]; bmat[i][k] += c * nv[j] * mv[l]; mmat[i][k] += c * mv[j] * mv[l];
            } } } }
            let (tr, det) = (gam[0][0] + gam[1][1], gam[0][0] * gam[1][1] - gam[0][1] * gam[1][0]);
            min_gam = min_gam.min(0.5 * (tr - (tr * tr - 4.0 * det).max(0.0).sqrt()));
            let gi = [[gam[1][1] / det, -gam[0][1] / det], [-gam[1][0] / det, gam[0][0] / det]];
            for i in 0..2 { for j in 0..2 {
                let mut sch = mmat[i][j];
                for p in 0..2 { for q in 0..2 { sch -= bmat[p][i] * gi[p][q] * bmat[q][j]; } }
                k_dip += b[i] * sch * b[j] * dth / (4.0 * PI * PI);
            } }
        }
        assert!(min_gam > 0.1, "{key} acoustic tensor not SPD: min_gam={min_gam}");
        println!("# Gate 3 {key}: fd_err={max_fd_err:.2e} min_Gamma_eig={min_gam:.4} K_dip={k_dip:.6}");

        let (b_norm, ks) = (b[0].hypot(b[1]), [4usize, 6, 8, 10]);
        let r_disk = if key == "square" { 26.0 } else { 32.0 };
        let c0 = if key == "square" { [0.0, 0.5 * b_norm] } else { [0.3 * cell.a1[0] + 0.5 * cell.a2[0], 0.3 * cell.a1[1] + 0.5 * cell.a2[1]] };
        let (mut e_vals, mut ln_hs) = (Vec::new(), Vec::new());
        for &k in &ks {
            let half = (k / 2) as f64;
            let p_l = [c0[0] - half * b[0], c0[1] - half * b[1]];
            let p_r = [c0[0] + half * b[0], c0[1] + half * b[1]];
            let dip = build_point_defects(&cell, &[DefectSpec::dislocation(b, Some(p_l)), DefectSpec::dislocation([-b[0], -b[1]], Some(p_r))], Some(r_disk), None).unwrap();
            let res = relax_elastic_2d(&dip).unwrap();
            let (_, gg) = gsv_eval_2d(&dip, &res.pos).unwrap();
            let g_inf = gg.iter().flatten().fold(0.0_f64, |m, &x| m.max(x.abs()));
            assert!(g_inf < 1e-9, "{key} k={k} conv={} g_inf={g_inf:.2e}", res.converged);
            e_vals.push(res.total_energy);
            ln_hs.push(((k as f64) * b_norm).ln());
        }
        let ml = ln_hs.iter().sum::<f64>() / (ks.len() as f64);
        let me = e_vals.iter().sum::<f64>() / (ks.len() as f64);
        let slope = ln_hs.iter().zip(&e_vals).map(|(x, y)| (x - ml) * (y - me)).sum::<f64>() / ln_hs.iter().map(|x| (x - ml).powi(2)).sum::<f64>();
        let rel = (slope - k_dip).abs() / k_dip;
        println!("# Gate 3 {key} dipole: E={e_vals:.5?} slope={slope:.5} vs K_cont={k_dip:.5} (rel={:.1}%)", rel * 100.0);
        assert!(rel < 0.20, "{key} dilute dipole slope {slope:.5} vs continuum {k_dip:.5}");
    }
    println!("# time Gate 3: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
}

#[test]
fn gate4_p4m_vs_p4g_nonsymmorphic_and_fusion() {
    let t0 = Instant::now();
    let (_, sq_cell, _) = named_cell("square");
    let (p4g_grp, p4g_cell, _) = named_cell("shastry_sutherland");
    let w_p4m = wyckoff_rotocentres("p4m", &sq_cell).unwrap();
    let c1a = w_p4m.iter().find(|w| w.order == 4 && w.coset_rep == [0, 0]).expect("p4m 1a");
    let c1b = w_p4m.iter().find(|w| w.order == 4 && w.coset_rep != [0, 0]).expect("p4m 1b");
    assert_eq!((c1a.quotient_order, c1b.quotient_order), (2, 2));

    let ext_p4g = SpaceGroupExtension::from_cell(&p4g_grp, &p4g_cell).unwrap();
    assert!(ext_p4g.is_nonsymmorphic);
    let gx = ext_p4g.element([0, 0], [[-1, 0], [0, 1]]).or_else(|| ext_p4g.element([0, 0], [[1, 0], [0, -1]])).expect("p4g axial glide");
    let gx2 = ext_p4g.compose(&gx, &gx);
    assert_eq!(gx2.r, [[1, 0], [0, 1]]);
    assert_ne!(gx2.l_vec, [0, 0], "p4g axial glide squared must be a nonzero lattice translation");
    let ext_p4m = SpaceGroupExtension::from_group("p4m").unwrap();
    let mx = ext_p4m.element([0, 0], [[-1, 0], [0, 1]]).unwrap();
    assert_eq!(ext_p4m.compose(&mx, &mx), ext_p4m.identity(), "p4m mirror squares to identity");
    let c4 = ext_p4g.element([0, 0], [[0, -1], [1, 0]]).unwrap();
    let conj = ext_p4g.conjugate(&c4, &gx);
    assert_eq!(conj.r, [[0, 1], [-1, 0]], "gx C4 gx^-1 = C4^-1");

    let w_p4g = wyckoff_rotocentres(&p4g_grp, &p4g_cell).unwrap();
    assert!(w_p4g.iter().any(|w| w.order == 4 && w.location.contains("2a") || w.gclass.contains("2a") || w.label.contains("2a")));

    let ch_same = fusion_channels("p4m", &sq_cell, &[DefectSpec::disclination(90.0, c1a.point), DefectSpec::disclination(-90.0, c1a.point)]).unwrap();
    assert!(ch_same.iter().filter(|c| c.burgers_coset_rep.is_some()).all(|c| c.parity_mod2 == Some(0)));
    let ch_cross = fusion_channels("p4m", &sq_cell, &[DefectSpec::disclination(90.0, c1b.point), DefectSpec::disclination(-90.0, c1a.point)]).unwrap();
    assert!(ch_cross.iter().any(|c| c.parity_mod2 == Some(1) && c.shortest_nonzero_burgers_frac.iter().any(|b| b.iter().map(|x| x.abs()).sum::<i32>() == 1)));

    let pair_even = build_point_defects(&sq_cell, &[DefectSpec::Disclination { angle_deg: 90.0, at: [-2.0, 0.0], cut: Some([-1.0, 0.0]) }, DefectSpec::Disclination { angle_deg: -90.0, at: [2.0, 0.0], cut: Some([1.0, 0.0]) }], Some(8.0), None).unwrap();
    let pair_odd = build_point_defects(&sq_cell, &[DefectSpec::Disclination { angle_deg: 90.0, at: [-1.5, 0.5], cut: Some([-1.0, 0.0]) }, DefectSpec::Disclination { angle_deg: -90.0, at: [2.0, 0.0], cut: Some([1.0, 0.0]) }], Some(8.0), None).unwrap();
    let e_even = relax_elastic_2d(&pair_even).unwrap().total_energy;
    let e_odd = relax_elastic_2d(&pair_odd).unwrap().total_energy;
    assert!(e_even.is_finite() && e_odd > 0.0);

    let m_coch: Vec<[f64; 2]> = (0..pair_even.pos.len()).map(|i| [(i as f64 * 0.3).cos(), (i as f64 * 0.7).sin()]).collect();
    let dr2 = verify_d_r_squared(&pair_even, &m_coch);
    assert!(dr2.iter().filter(|x| x.1).all(|x| x.2[0].hypot(x.2[1]) < 1e-12));
    let hols = plaquette_holonomies(&pair_odd, Some("p4m"), Some(&sq_cell));
    assert!(hols.iter().any(|h| h.has_torsion || h.has_curvature));
    println!("# time Gate 4: {:.2} ms | gx^2={:?} conj={:?} | e_even={e_even:.4} e_odd={e_odd:.4}", t0.elapsed().as_secs_f64() * 1e3, gx2.l_vec, conj.l_vec);
}

#[test]
fn gate5_mobius_seam_3d() {
    let t0 = Instant::now();
    for (key, long_len) in [("square", 30usize), ("brick_pg", 24usize)] {
        let (grp, cell, frame) = named_cell(key);
        let cat = DefectCatalogue::new(&grp, &cell, frame);
        let wc1 = cat.build_wall("Wc1", Some(4.0)).expect("Wc1");
        let res_wc = relax_sheet_3d(&wc1, &cell, &Sheet3DOptions::default(), None);
        assert!(!res_wc.faces.is_empty() && res_wc.has_finite_geometry() && res_wc.bend_energy > 0.0);
        let seam = &cat.mobius_seams[0];
        let smooth = build_mobius_seam(&cell, seam.sigma, long_len, if key == "square" { 1.0 } else { seam.width });
        let res_sm = relax_sheet_3d(&smooth, &cell, &Sheet3DOptions { max_iters: 800, ..Default::default() }, None);
        let align = min_twisted_hinge_alignment(&res_sm.xyz, &res_sm.faces).expect("mobius align");
        assert!(!res_sm.faces.is_empty() && res_sm.has_finite_geometry() && res_sm.bend_energy > 0.0 && align > 0.85, "{key} align={align}");
        println!("# Gate 5 {key}: Wc1(bend={:.4}) | smooth(len={long_len}, bend={:.4}, min_align={align:.4})", res_wc.bend_energy, res_sm.bend_energy);
    }
    println!("# time Gate 5: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
}
