"""Tests of papertilling.defects.build.

Part A, the GATE: the cone-safe builder (fusion_framework engine, used for everything) against the
independent C01-C07 engine lab_defects, on nine lattices, for every disclination class with
|angle| <= 180 at every rotation-centre class and for the shortest dislocation classes:
  (i)  the charge read from the built graph equals the requested class;
  (ii) the degree-labelled core graph (radius 2, graph-only rule of build.py, applied identically to
       both engines' graphs) is isomorphic to the lab_defects build with the SAME centre / core point,
       cut direction and patch radius.
The expected values are the requested class (input) and the lab_defects graph (independent engine).

Part B: controls of the builder (negative controls, close-core stress, order independence).
Part C: the fast path (papertilling.defects.fast, fast_relax) equals the legacy path.

Speed (2026-11-13): the default run gates one class per lattice (disclination and dislocation in
turn over the nine lattices), serially, with the lab engine's fast twins (fast.disclination_fast / dislocation_fast: the same
algorithm, outputs tested identical to legacy in Part C and in test_config.py). `pytest -m slow`
runs the full gate (every class, legacy lab engine itself, worker processes).
"""
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import numpy as np
import pytest

from papertilling.defects.build import (build, lattice_data, graph_core, isomorphic_cores, classify_charge,
                                        same_charge, _run, validate_specs, patch_rim)
from papertilling.legacy.lab_defects import disclination, dislocation, frame_faces, loop_charge, rotation_about
from papertilling.legacy import fusion_framework as ff
from papertilling.defects import fast as fastmod

GATE = ["square", "triangular", "honeycomb", "kagome", "shastry_sutherland", "chiral_square",
        "chiral_triangular", "breathing_kagome", "zigzag"]
TH0 = 0.123                                   # seam direction (rad) given to both engines


def _R(lat, b=0.0):
    """First patch radius tried: room for r_det (detection) + 2 (core ball) + r_det (rim exclusion)
    bonds of mean length. graph_core refuses (never truncates) when it is too small; then x 1.25."""
    return (2 * lat.r_det + 4) * 0.5 * (lat.lmean + lat.lmax) + 0.5 * lat.scale + b


_TRIES = []


def _with_room(fn, R):
    """Run fn(R), enlarging R by 1.25 while the core ball reaches the rim (at most 4 times)."""
    for k in range(5):
        try:
            _TRIES.append(k + 1)
            return fn(R)
        except ValueError as e:
            if "rim" not in str(e):
                raise
            R *= 1.25
    raise AssertionError("no patch radius gave room for the core")


def _lab_core(x, keyof, centre, R, lat, radius=2):
    """Core graph of a lab_defects build, by the same graph-only rule; rim by reference position."""
    G = nx.Graph()
    G.add_nodes_from(range(len(x.sites)))
    G.add_edges_from(x.edges)
    keys = [keyof(s) for s in x.sites]
    rim = patch_rim(lat, [lat.L.pos(k) for k in keys], [k[2] for k in keys], [G.degree(i) for i in range(len(keys))], centre, R)
    return graph_core(G, rim, lat, radius)


def _disclination_cases():
    out = []
    for name in GATE:
        lat = lattice_data(name)
        for rec in lat.centres:
            for ang in rec["angles"]:
                if abs(ang) <= 180 + 1e-9:
                    out.append((name, rec["label"], ang))
    return out


def _dislocation_cases():
    out = []
    for name in GATE:
        lat = lattice_data(name)
        vecs = [lat.A @ np.array(m, float) for m in [(i, j) for i in range(-3, 4) for j in range(-3, 4)] if m != (0, 0)]
        bmin = min(np.linalg.norm(v) for v in vecs)
        shortest = [v for v in vecs if np.linalg.norm(v) < bmin + 1e-6]
        reps = []
        for v in shortest:                    # one representative per P+ orbit
            if not any(any(np.allclose(M @ v, r) for M, _ in lat.ops_plus) for r in reps):
                reps.append(v)
        for k, b in enumerate(reps):
            out.append((name, k, tuple(b)))
    return out


def _cuts(lat, b, n=3):
    """Perpendicular cut with cross2(c, b) < 0 (the builder's Volterra rule) + lattice directions."""
    b = np.asarray(b, float)
    out = [np.array([-b[1], b[0]]) / np.linalg.norm(b)]
    vecs = sorted((lat.A @ np.array((i, j), float) for i in range(-2, 3) for j in range(-2, 3) if (i, j) != (0, 0)),
                  key=lambda v: round(float(np.linalg.norm(v)), 6))
    for v in vecs:
        u = v / np.linalg.norm(v)
        if u[0] * b[1] - u[1] * b[0] < -1e-6 and not any(np.allclose(u, w) for w in out) and len(out) < n:
            out.append(u)
    return out


FULL = os.environ.get("PAPERTILLING_FULL_GATE") == "1"
POINTS = [(0.371, 0.413)] + ([(0.62, 0.18), (0.12, 0.77), (0.87, 0.55)] if FULL else [])
NCUTS = 3 if FULL else 2


def gate_case(case, lab_fast=False):
    """One gate comparison (runs in a worker process). Returns a dict, never raises.
    lab_fast: the lab engine's fast twins (identical outputs, Part C) instead of lab_defects itself."""
    t0 = time.time()
    disc_lab = fastmod.disclination_fast if lab_fast else disclination
    disl_lab = fastmod.dislocation_fast if lab_fast else dislocation
    try:
        if case[0] == "disc":
            _, name, label, angle = case
            lat = lattice_data(name)
            c = next(r for r in lat.centres if r["label"] == label)["point"]

            def both(R):
                conf = build(lat, dict(kind="disclination", angle=angle, at=c, cut=(math.cos(TH0), math.sin(TH0))),
                             R=R, centre=c, picture="raw")
                x = disc_lab(lat.cell, c, math.radians(angle), R=R, theta0=TH0)
                return conf, conf.core_graph(2), _lab_core(x, lambda s: s[0], c, R, lat)[1]
            conf, Hf, Hl = _with_room(both, _R(lat))
            q = conf.total_charge()
            want = classify_charge(lat, angle, rotation_about(c, math.radians(angle)))
            charge_ok = q["kind"] == "disclination" and same_charge(lat, q, want) and q["centre_class"] == label
            extra = ""
        else:
            _, name, b, f, cut = case
            lat = lattice_data(name)
            b, p, cut = np.asarray(b, float), lat.A @ np.asarray(f, float), np.asarray(cut, float)

            def both(R):
                conf = build(lat, dict(kind="dislocation", b=b, at=p, cut=cut), R=R, centre=p, picture="raw")
                x = disl_lab(lat.cell, b, core=p, cut=cut, R=R)
                return conf, x, conf.core_graph(2), _lab_core(x, lambda s: s, p, R, lat)[1]
            conf, x, Hf, Hl = _with_room(both, _R(lat, np.linalg.norm(b)))
            q = conf.total_charge()
            fl = (fastmod.frame_faces_fast if lab_fast else frame_faces)(x)
            kout = max(range(len(fl)), key=lambda i: len(fl[i]))
            charge_ok = (q["kind"] == "dislocation" and np.allclose(q["b"], b)                     # requested b
                         and np.allclose((fastmod.loop_charge_fast if lab_fast else loop_charge)(x, fl[kout][::-1])[1], b))                    # lab: same K5 b
            extra = str(sorted(c["ring"] for c in conf.cores()))
        return dict(error=None, charge_ok=bool(charge_ok), core_ok=isomorphic_cores(Hf, Hl), label=q["label"], tries=_TRIES[-1],
                    nodes=(Hf.number_of_nodes(), Hl.number_of_nodes()), rings=extra, time=time.time() - t0)
    except Exception as e:                                       # reported by the test, not swallowed
        return dict(error=f"{type(e).__name__}: {e}", time=time.time() - t0)


CASES = [("disc",) + c for c in _disclination_cases()]
for _name, _k, _b in _dislocation_cases():
    _lat = lattice_data(_name)
    for _f in POINTS:
        for _c in _cuts(_lat, _b, NCUTS):
            CASES.append(("disl", _name, _b, _f, tuple(_c)))


def _case_id(c):
    if c[0] == "disc":
        return f"{c[1]}-D({c[2]},{c[3]:+.0f})"
    return f"{c[1]}-b{np.round(c[2], 2).tolist()}-at{list(c[3])}-cut{np.round(c[4], 2).tolist()}"


def _quick_ids():
    """Default gate: one case per lattice, the first disclination class for GATE[0], GATE[2], ...
    and the first dislocation case for GATE[1], GATE[3], ... (technical choice, speed)."""
    out = set()
    for i, name in enumerate(GATE):
        kind = "disc" if i % 2 == 0 else "disl"
        out.add(_case_id(next(c for c in CASES if c[0] == kind and c[1] == name)))
    return out


QUICK = _quick_ids()


@pytest.fixture(scope="module")
def gate_quick():
    """The default gate cases, serially, lab engine = fast twins (# time printed)."""
    t0 = time.time()
    out = {_case_id(c): gate_case(c, lab_fast=True) for c in CASES if _case_id(c) in QUICK}
    print(f"# time quick gate: {len(out)} comparisons, serial, {time.time() - t0:.1f} s")
    return out


@pytest.mark.parametrize("cid", sorted(QUICK))
def test_gate_quick(gate_quick, cid):
    r = gate_quick[cid]
    assert r["error"] is None, r["error"]
    assert r["charge_ok"], r
    assert r["nodes"][0] > 0 and r["nodes"][1] > 0, r
    assert r["core_ok"], r


@pytest.fixture(scope="module")
def gate():
    """All gate comparisons (legacy lab engine), computed once, in parallel worker processes (# time printed)."""
    t0 = time.time()
    workers = max(1, min(10, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(gate_case, CASES, chunksize=2))
    out = {_case_id(c): r for c, r in zip(CASES, res)}
    print(f"# time gate: {len(CASES)} comparisons, {workers} workers, wall {time.time() - t0:.1f} s, "
          f"cpu {sum(r['time'] for r in res):.1f} s; patch radius retries {sum(r.get('tries', 1) - 1 for r in res)}")
    return out


@pytest.mark.slow
@pytest.mark.parametrize("cid", [_case_id(c) for c in CASES])
def test_gate(gate, cid):
    r = gate[cid]
    assert r["error"] is None, r["error"]
    assert r["charge_ok"], r
    assert r["nodes"][0] > 0 and r["nodes"][1] > 0, r          # two empty cores would be 'isomorphic'
    assert r["core_ok"], r


@pytest.mark.slow
def test_gate_table(gate):
    """Print the gate table (lattice x class: charge ok / core ok)."""
    rows = {}
    for c in CASES:
        r = gate[_case_id(c)]
        key = (c[1], f"D({c[2]},{c[3]:+.0f})" if c[0] == "disc" else f"T b={np.round(c[2], 3).tolist()}")
        ok = rows.setdefault(key, [0, 0, 0])
        ok[0] += 1
        ok[1] += bool(r.get("charge_ok"))
        ok[2] += bool(r.get("core_ok"))
    for (n, k), (m, a, b) in rows.items():
        print(f"GATE {n:20s} {k:28s} builds {m:2d}  charge ok {a:2d}  core ok {b:2d}")
    assert all(m == a == b for m, a, b in rows.values())


def test_honeycomb_dangling_core_survives():
    """C04 (cited result): the cut-and-reglue core K1 is a 14-ring with a 1-bond site. Placement
    found by scanning C04's grid (frac (0.0618, 0.7355) = ((0+0.371)/6, (4+0.413)/6), cut +a2 side).
    Fixed builder: keeps the 1-bond site and matches lab_defects. Legacy builder (negative control of
    the fix): peels the site and silently returns a 12-ring core."""
    lat = lattice_data("honeycomb")
    b = np.array([math.sqrt(3), 0.0])
    p = lat.A @ np.array([0.371 / 6, 4.413 / 6])
    cut = np.array([0.0, 1.0])
    R = 12.0
    conf = build(lat, dict(kind="dislocation", b=b, at=p, cut=cut), R=R, centre=p, picture="raw")
    assert [c["ring"] for c in conf.cores()] == [14]
    H = conf.core_graph(2)
    assert min(d for _, d in H.nodes(data="deg")) == 1
    x = dislocation(lat.cell, b, core=p, cut=cut, R=R)
    assert isomorphic_cores(H, _lab_core(x, lambda s: s, p, R, lat)[1])
    old = ff.build_configuration(lat.cell, [dict(kind="dislocation", b=b, at=p, cut=cut)], L=lat.L, R=R, centre=p, picture="raw")
    assert [c["ring"] for c in ff.cores(old)] == [12]


# ---------------------------------------------------------------------------------------------
# Part B: controls
# ---------------------------------------------------------------------------------------------
def test_negative_controls():
    sq = lattice_data("square")
    with pytest.raises(ValueError, match="not a rotation centre"):
        build(sq, dict(kind="disclination", angle=90, at=(0.3, 0.1)))
    with pytest.raises(ValueError, match="2-fold"):        # (0, 1/2): bond midpoint, 2-fold only
        build(sq, dict(kind="disclination", angle=90, at=(0.0, 0.5)))
    with pytest.raises(ValueError, match="lattice translation"):
        build(sq, dict(kind="dislocation", b=(0.5, 0.0), at=(0.37, 0.41)))
    with pytest.raises(ValueError, match="360"):           # +180 + +180 closes the cone
        build(sq, dict(kind="disclination", angle=180, at=(0, 0)), dict(kind="disclination", angle=180, at=(3.5, 0.5)))
    with pytest.raises(ValueError):                        # cut on the wrong side of b (would insert a row)
        build(sq, dict(kind="dislocation", b=(1, 0), at=(0.37, 0.41), cut=(0, -1)))
    # snap=True moves a near miss to the nearest allowed centre
    c = build(sq, dict(kind="disclination", angle=90, at=(0.45, 0.6)), snap=True, picture="raw")
    assert np.allclose(c.specs[0]["at"], (0.5, 0.5))


def test_close_core_stress():
    """Two disclinations at neighbouring centres, 3 x 3 forced seam directions, built WITHOUT the
    builder's own refusal. A forced seam may be refused (it runs into the other core: honest refusal);
    every crystal that IS built must be a disk (V - E + F = 2 with the outside face), have consistent
    frames, and total Frank angle = sum of the requested angles (exact: Gauss-Bonnet on the disk).
    C09's builder gave -540 / -810 here. Each pair must also build with automatic seams."""
    sq = lattice_data("square")
    dirs = [(math.cos(a), math.sin(a)) for a in (0.123, 0.123 + 2 * math.pi / 3, 0.123 + 4 * math.pi / 3)]
    pairs = [((90, (0, 0)), (-90, (1, 0))), ((90, (0, 0)), (-90, (0.5, 0.5))), ((-90, (0, 0)), (90, (0.5, 0.5))),
             ((90, (0, 0)), (90, (0.5, 0.5))), ((180, (0, 0)), (-90, (0.5, 0.5)))]
    total_built = 0
    for (a1, p1), (a2, p2) in pairs:
        built = refused = 0
        for c1 in dirs:
            for c2 in dirs:
                specs = validate_specs(sq, [dict(kind="disclination", angle=a1, at=p1, cut=c1),
                                            dict(kind="disclination", angle=a2, at=p2, cut=c2)])
                try:
                    d, rep = _run(sq, specs, R=6.5, check=False)
                except ValueError:
                    refused += 1
                    continue
                built += 1
                assert rep["euler"] == 2 and rep["frames_bad"] == 0, rep
                assert abs(rep["frank"] - (a1 + a2)) < 1e-6, rep
        auto = build(sq, dict(kind="disclination", angle=a1, at=p1), dict(kind="disclination", angle=a2, at=p2),
                     R=6.5, picture="raw")
        r = auto.report
        assert r["euler"] == 2 and r["frames_bad"] == 0 and abs(r["frank"] - (a1 + a2)) < 1e-6, r
        print(f"close-core {a1:+d}@{p1} {a2:+d}@{p2}: forced seams built {built}, refused {refused}; auto seams built")
        total_built += built
    assert total_built >= 1          # the forced-seam check is not vacuous


def test_order_independence():
    """A dislocation and a disclination with explicit, non-crossing cuts, built in both orders:
    same charge class and isomorphic graphs (whole patch)."""
    sq = lattice_data("square")
    s1 = dict(kind="dislocation", b=(1, 0), at=(-2.63, 0.41), cut=(0, 1))
    s2 = dict(kind="disclination", angle=90, at=(2, 0), cut=(0.9, -0.43))
    c12 = build(sq, s1, s2, R=11, centre=(0, 0), picture="raw")
    c21 = build(sq, s2, s1, R=11, centre=(0, 0), picture="raw")
    assert same_charge(sq, c12.total_charge(), c21.total_charge())
    G1, G2 = c12.defect.to_networkx(), c21.defect.to_networkx()
    assert nx.is_isomorphic(G1, G2)


def test_relaxed_picture_and_inspect(capsys):
    t0 = time.time()
    c = build("square", dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)), R=7)
    info = c.inspect()
    assert info["euler"] == 2 and info["frames_consistent"] and abs(info["total_frank"]) < 1e-9
    import matplotlib
    matplotlib.use("Agg")
    ax = c.plot()
    assert "dislocation" in ax.get_title()
    print(f"# time relaxed build {time.time() - t0:.2f} s")


def test_lattice_inputs():
    """UnitCell, legacy Family and recipe dict give the same lattice data, and a +90 build on each reads
    back the class D(1a, +90) (hand: the site of the square lattice is the 4-fold centre, ITA p4m 1a)."""
    from papertilling.legacy.lattice_families import Family
    from papertilling.legacy.wallpaper_lattices import make_cell
    fam = Family("p4m", sites=[(0, 0)], bonds=[((0, 0), (1, 0))])
    for L in (make_cell("square"), fam, fam.recipe()):
        lat = lattice_data(L)
        c = next(r["point"] for r in lat.centres if r["on_site"] and r["order"] == 4)
        conf = build(L, dict(kind="disclination", angle=90, at=c), R=6, picture="raw")
        q = conf.total_charge()
        assert q["kind"] == "disclination" and q["angle"] == 90 and lat.centre_class(q["centre"])["on_site"]
    with pytest.raises(TypeError):
        build(3.0, dict(kind="disclination", angle=90, at=(0, 0)))


def _exact_irregular(conf, lat):
    """Second route for the core set: exact degree-labelled isomorphism (VF2++) of every candidate's
    r_det-ball against the perfect balls (one per sublattice), instead of layered signatures."""
    from papertilling.legacy.lab_defects import perfect_patch
    G = conf.defect.to_networkx()
    for v in G:
        G.nodes[v]["deg"] = G.degree(v)
    pp = perfect_patch(lat.cell, (lat.r_det + 3) * lat.lmax + 2 * lat.scale)
    P = pp.to_networkx()
    for v in P:
        P.nodes[v]["deg"] = P.degree(v)
    centre_site = {s: next(i for i, k in enumerate(pp.sites) if k == (0, 0, s)) for s in lat.L.rule}
    ball = lambda H, v: H.subgraph(nx.single_source_shortest_path_length(H, v, cutoff=lat.r_det)).copy()
    perfect = [ball(P, i) for i in centre_site.values()]
    rim = conf.rim()
    drim = nx.multi_source_dijkstra_path_length(G, rim)
    return {v for v in G if drim.get(v, 10**9) > lat.r_det and not any(isomorphic_cores(ball(G, v), B) for B in perfect)}


@pytest.mark.parametrize("name,specs", [
    ("square", [dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5))]),
    ("honeycomb", [dict(kind="dislocation", b=(math.sqrt(3), 0), at=(0.7441, 1.1032), cut=(0, 1))]),
    pytest.param("kagome", [dict(kind="disclination", angle=-60, at=(1.5, 0.8660254037844386))], marks=pytest.mark.slow),
])
def test_core_set_exact_second_route(name, specs):
    lat = lattice_data(name)
    conf = build(lat, *specs, picture="raw")
    K = conf.core_sites()
    assert K and K == _exact_irregular(conf, lat)


# ---------------------------------------------------------------------------------------------
# Part C: fast path == legacy path (expected values: the legacy functions themselves)
# ---------------------------------------------------------------------------------------------
import copy

from papertilling.defects import fast_relax
from papertilling.defects.build import layer_signature
from papertilling.legacy.lab_defects import Perfect, perfect_patch


def same_edge_data(e1, e2, tol=1e-12):
    """Same keys in the same order; vec / j / g / ang / type equal (arrays within tol)."""
    assert list(e1) == list(e2), "bond keys or their order differ"
    for k in e1:
        a, b = e1[k], e2[k]
        assert a.keys() == b.keys(), (k, a.keys(), b.keys())
        for f in a:
            x, y = a[f], b[f]
            if f == "g":
                assert np.allclose(x[0], y[0], rtol=0, atol=tol) and np.allclose(x[1], y[1], rtol=0, atol=tol), (k, f)
            elif isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                assert np.allclose(x, y, rtol=0, atol=tol), (k, f, x, y)
            else:
                assert x == y, (k, f, x, y)


def same_crystal(d1, d2, tol=1e-12):
    assert list(d1.sites) == list(d2.sites)
    assert np.allclose(d1.positions, d2.positions, rtol=0, atol=tol)
    assert d1.edges == d2.edges
    same_edge_data(d1.edge, d2.edge, tol)


@pytest.mark.parametrize("name", ["square", "honeycomb", "kagome", "zigzag"] + [pytest.param(n, marks=pytest.mark.slow) for n in ("shastry_sutherland", "chiral_triangular")])
def test_fast_perfect_primitives(name):
    """FastPerfect (memoised / vectorised) against lab_defects.Perfect: same disk keys in the same
    order, same positions, same locate (sites and, negative control, points between sites), same
    is_symmetry answers (a symmetry and, negative control, a shift by a third of a1)."""
    lat = lattice_data(name)
    P, F = Perfect(lat.cell), fastmod.FastPerfect(lat.cell)
    for c, R in (((0.0, 0.0), 4.0), ((0.37, -1.21), 6.3), (tuple(lat.cell.a1 * 2.0), 3.0 * lat.lmax)):
        keys = P.sites_in_disk(c, R)
        assert F.sites_in_disk(c, R) == keys and keys
        for k in keys:
            assert np.array_equal(P.pos(k), F.pos(k))
            assert P.locate(P.pos(k)) == F.locate(P.pos(k)) == k
            q = P.pos(k) + np.array([0.123 * lat.lmax, 0.071 * lat.lmax])        # not a site
            assert P.locate(q) == F.locate(q)
    g_yes = (np.eye(2), lat.cell.a1.copy())
    g_no = (np.eye(2), lat.cell.a1 / 3.0)
    assert P.is_symmetry(g_yes) and F.is_symmetry(g_yes)
    assert not P.is_symmetry(g_no) and not F.is_symmetry(g_no)


_EQ_BUILDS = [
    ("square", [dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)),
                dict(kind="disclination", angle=90, at=(5, 0)), dict(kind="disclination", angle=-90, at=(4.5, -0.5))]),
    ("square", [dict(kind="dislocation", b=(1, 0), at=(-2.63, 0.41), cut=(0, 1)),
                dict(kind="disclination", angle=90, at=(2, 0), cut=(0.9, -0.43)), dict(R=11, centre=(0, 0))]),
    ("honeycomb", "57"),
    ("kagome", [dict(kind="disclination", angle=-60, at=(1.5, 0.8660254037844386))]),
    ("shastry_sutherland", "disl"),
    ("zigzag", "disc180"),
    ("square", [dict(kind="disclination", angle=180, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5)),
                dict(R=8.5)]),            # automatic seams: the first cut is refused, snapshot/restore runs
]


def _eq_specs(name, specs):
    lat = lattice_data(name)
    if specs == "57":
        c1 = next(r["point"] for r in lat.centres if r["order"] == 6)
        return [dict(kind="disclination", angle=60, at=c1), dict(kind="disclination", angle=-60, at=c1 + lat.cell.a1)]
    if specs == "disl":
        return [dict(kind="dislocation", b=lat.cell.a1, at=0.371 * lat.cell.a1 + 0.413 * lat.cell.a2)]
    if specs == "disc180":
        return [dict(kind="disclination", angle=180, at=lat.centres[0]["point"])]
    return list(specs)


_EQ_QUICK = {1, 2}                       # default run: mixed square, honeycomb 5-7; the rest -m slow


@pytest.mark.parametrize("name,specs", [pytest.param(n, sp, id=f"{n}-{i}", marks=() if i in _EQ_QUICK else pytest.mark.slow)
                                        for i, (n, sp) in enumerate(_EQ_BUILDS)])
def test_build_fast_equals_legacy(name, specs, monkeypatch):
    """The fast path (FastPerfect, FastBuilder.snapshot, fast readers, sparse irregular sites) against
    the full legacy path (PAPERTILLING_LEGACY=1: legacy Perfect, _Builder, fusion_framework readers,
    layer_signature): identical crystal, report, charge and core sites; picture 'raw'."""
    sp = _eq_specs(name, specs)
    kw = sp.pop() if "kind" not in sp[-1] else {}
    monkeypatch.delenv("PAPERTILLING_LEGACY", raising=False)
    t0 = time.time()
    a = build(name, *sp, picture="raw", **kw)
    qa, ka = a.total_charge()["label"], a.core_sites()
    t1 = time.time()
    monkeypatch.setenv("PAPERTILLING_LEGACY", "1")
    b = build(name, *sp, picture="raw", **kw)
    qb, kb = b.total_charge()["label"], b.core_sites()
    t2 = time.time()
    assert type(a.lattice.L) is fastmod.FastPerfect and type(b.lattice.L) is Perfect     # both paths really ran
    same_crystal(a.defect, b.defect)
    ra, rb = dict(a.report), dict(b.report)
    assert np.array_equal(ra.pop("centre"), rb.pop("centre")) and ra == rb
    assert qa == qb and ka == kb
    print(f"# time build {name}: fast {t1 - t0:.2f} s, legacy {t2 - t1:.2f} s")


def test_check_frames_fast_equals_legacy():
    """check_frames_fast vs fusion_framework.check_frames: 0 on a good crystal (positive control);
    the same non-zero count after corrupting two bond vectors and one frame change (negative control)."""
    c = build("square", *_eq_specs("square", _EQ_BUILDS[1][1])[:2], R=11, centre=(0, 0), picture="raw")
    d = c.defect
    assert fastmod.check_frames_fast(d) == ff.check_frames(d) == 0
    bad = copy.deepcopy(d)
    keys = list(bad.edge)
    bad.edge[keys[3]]["vec"] = bad.edge[keys[3]]["vec"] + np.array([0.0, 1e-3])
    bad.edge[keys[40]]["vec"] = bad.edge[keys[40]]["vec"] * 1.5
    gk = next(k for k in keys if "g" in bad.edge[k])
    M, t = bad.edge[gk]["g"]
    bad.edge[gk]["g"] = (M, t + np.array([0.01, 0.0]))
    n_legacy = ff.check_frames(bad)
    assert n_legacy > 0 and fastmod.check_frames_fast(bad) == n_legacy


@pytest.mark.parametrize("name", ["square", "honeycomb", "kagome", "shastry_sutherland", "breathing_kagome"])
def test_irregular_fast_equals_layer_signature(name):
    """fast.irregular (sparse-matrix layer signatures) against the definition, build.layer_signature,
    on every site of a dislocated patch and of a perfect patch (rim sites included), then with one
    bond deleted at the perfect patch's centre (negative control: both ends become irregular)."""
    lat = lattice_data(name)
    x = fastmod.dislocation_fast(lat.cell, lat.cell.a1, core=0.371 * lat.cell.a1 + 0.413 * lat.cell.a2, R=9.0)
    pp = perfect_patch(lat.cell, 9.0)
    for d in (x, pp):
        G = d.to_networkx()
        adj = [list(G.adj[v]) for v in G.nodes]
        deg = [len(a) for a in adj]
        cand = list(range(len(adj)))
        ref = [v for v in cand if layer_signature(adj, deg, v, lat.r_det) not in lat.perfect_sigs]
        assert fastmod.irregular(adj, deg, cand, lat.r_det, lat.perfect_sigs, fast=True) == ref
        assert fastmod.irregular(adj, deg, cand, lat.r_det, lat.perfect_sigs, fast=False) == ref
    u = min(range(len(pp.sites)), key=lambda i: np.linalg.norm(pp.positions[i]))
    assert u not in ref                                          # perfect patch: the centre is regular
    w = adj[u][0]
    adj[u].remove(w); adj[w].remove(u)
    deg = [len(a) for a in adj]
    ref2 = [v for v in cand if layer_signature(adj, deg, v, lat.r_det) not in lat.perfect_sigs]
    assert u in ref2 and w in ref2
    assert fastmod.irregular(adj, deg, cand, lat.r_det, lat.perfect_sigs, fast=True) == ref2


@pytest.mark.parametrize("name", ["square"] + [pytest.param(n, marks=pytest.mark.slow)
                                                for n in ("shastry_sutherland", "breathing_kagome")])
def test_relax_fast_vs_legacy(name):
    """Relaxed picture. (i) fast_relax.relax_reference (scipy trf/lsmr on relax_setup's copy of the
    legacy springs, triangles and anchor) reproduces legacy relax_picture's positions to 1e-12: the
    copy of the energy is exact. (ii) The fast minimiser (Gauss-Newton, sparse LU) leaves the graph
    and every bond's vec/g data untouched and reaches the same energy within 1e-6 relative
    (positions are a drawing only; the energy is flat, so positions are not compared)."""
    lat = lattice_data(name)
    if name == "square":
        sp, kw = [dict(kind="disclination", angle=90, at=(0, 0)), dict(kind="disclination", angle=-90, at=(0.5, 0.5))], dict(R=6)
    elif name == "shastry_sutherland":
        sp, kw = _eq_specs(name, "disl") + [dict(kind="dislocation", b=-lat.cell.a1,
                                                 at=0.371 * lat.cell.a1 + 4.413 * lat.cell.a2)], {}
    else:                                       # four disclinations (the benchmark case, 1301 sites)
        r = sorted(lat.centres, key=lambda r: -r["order"])[0]
        a, c, a1, a2 = 360 / r["order"], r["point"], lat.cell.a1, lat.cell.a2
        sp, kw = [dict(kind="disclination", angle=a, at=c), dict(kind="disclination", angle=-a, at=c + 2 * a1),
                  dict(kind="disclination", angle=a, at=c + 3 * a2),
                  dict(kind="disclination", angle=-a, at=c + 3 * a2 + 2 * a1)], {}
    raw = build(name, *sp, picture="raw", **kw).defect
    d_leg, d_fast = copy.deepcopy(raw), copy.deepcopy(raw)
    t0 = time.time()
    ff.relax_picture(d_leg)
    t1 = time.time()
    fast_relax.relax_picture(d_fast, fast=True)
    t2 = time.time()
    z_ref, _ = fast_relax.relax_reference(copy.deepcopy(raw))
    assert np.allclose(z_ref.reshape(-1, 2), d_leg.positions, rtol=0, atol=1e-12)
    assert d_fast.edges == d_leg.edges == raw.edges
    strip = lambda E: {k: {f: v for f, v in e.items() if f != "ghost"} for k, e in E.items()}
    same_edge_data(strip(d_fast.edge), strip(raw.edge))
    S = fast_relax.relax_setup(raw)
    e_leg, e_fast = fast_relax.cost(S, d_leg.positions.ravel()), fast_relax.cost(S, d_fast.positions.ravel())
    assert abs(e_fast - e_leg) <= 1e-6 * e_leg, (e_fast, e_leg)
    assert d_fast.info["picture"] == d_leg.info["picture"] == "relaxed"
    print(f"# time relax {name} ({len(raw.sites)} sites): legacy {t1 - t0:.2f} s, fast {t2 - t1:.2f} s; "
          f"energy legacy {e_leg:.10g}, fast {e_fast:.10g}")


def test_legacy_switch(monkeypatch):
    """PAPERTILLING_LEGACY=1 (read at call time) selects the legacy path everywhere; the caches are
    keyed on the path, so the two never mix."""
    monkeypatch.delenv("PAPERTILLING_LEGACY", raising=False)
    assert fastmod.enabled() and fastmod.enabled(True) and not fastmod.enabled(False)
    monkeypatch.setenv("PAPERTILLING_LEGACY", "1")
    assert not fastmod.enabled()
    lat = lattice_data("square")
    assert type(lat.L) is Perfect
    c = build("square", dict(kind="disclination", angle=90, at=(0, 0)), picture="raw")
    assert type(c.lattice.L) is Perfect
    monkeypatch.delenv("PAPERTILLING_LEGACY")
    lat2 = lattice_data("square")
    assert lat2 is not lat and type(lat2.L) is fastmod.FastPerfect
