"""Layer-group slabs and the periodic Dirichlet complex (C1 WP-L1; Python mirror of space3d/layer_slab.rs and
growth/periodic_net.rs). Expected values are hand-derived (stated per test) or come from an independent method
(scipy's Delaunay); the parity test compares with a dump written by the Rust test `layer_slab_parity_dump`."""
from __future__ import annotations

import itertools
import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from papertilling import _layer_slab as LS
from papertilling._growth_engine import grow
from papertilling._periodic_net import PeriodicNet, _canonical

SQ3_2 = 0.8660254037844386


def _counts(net, kind, layers):
    g = grow(net, net.seed_cells(kind), max_layers=layers, max_cells=10**6)
    assert g["stopped"] is None
    return [s["cells"] for s in g["stats"]], [s["euler"] for s in g["stats"]]


@pytest.mark.parametrize("name,L,P,forms", [
    # square net: unit-square cells, box of whole squares (plan §2.4 {4,4})
    ("square", [[1, 0, 0], [0, 1, 0]], [[0, 0, 0]],
     {"cell": lambda n: (2 * n + 1) ** 2, "vertex": lambda n: (2 * n + 2) ** 2, "edge": lambda n: (2 * n + 1) * (2 * n + 2)}),
    # triangular net: hexagonal cells, {6,3} forms
    ("tri", [[1, 0, 0], [-0.5, SQ3_2, 0]], [[0, 0, 0]],
     {"cell": lambda n: 3 * n * n + 3 * n + 1, "vertex": lambda n: 3 * (n + 1) ** 2, "edge": lambda n: (3 * n + 2) * (n + 1)}),
    # honeycomb net: triangular cells, {3,6} forms
    ("honey", [[1, 0, 0], [-0.5, SQ3_2, 0]], [[0.5, SQ3_2 / 3, 0], [0, 2 * SQ3_2 / 3, 0]],
     {"cell": lambda n: 6 * n * n + 6 * n + 1, "vertex": lambda n: 6 * (n + 1) ** 2, "edge": lambda n: 2 * (3 * n + 1) * (n + 1)}),
])
def test_flat_nets_match_the_closed_forms(name, L, P, forms):
    net = PeriodicNet(L, P)
    assert net.mode == "flat" and net.euler_defined
    for kind, f in forms.items():
        cells, chi = _counts(net, kind, 6)
        assert cells == [f(n) for n in range(7)], (name, kind)
        assert set(chi) == {1}


def test_mirror_bilayer_and_simple_cubic():
    # bilayer (0,0,+-h): square prisms, 8 cells per vertex -> whole 2-site columns (hand)
    b = PeriodicNet([[1, 0, 0], [0, 1, 0]], [[0, 0, 0.3], [0, 0, -0.3]])
    assert b.mode == "slab" and not b.euler_defined and [len(t) for t in b.verts] == [8]
    assert _counts(b, "cell", 4)[0] == [1] + [2 * (2 * n + 1) ** 2 for n in range(1, 5)]
    assert _counts(b, "vertex", 4)[0] == [2 * (2 * n + 2) ** 2 for n in range(5)]
    # simple cubic: cubes (plan §2.4 E3 forms)
    sc = PeriodicNet(np.eye(3), [[0, 0, 0]])
    for kind, f in {"cell": lambda n: (2 * n + 1) ** 3, "vertex": lambda n: (2 * n + 2) ** 3,
                    "face": lambda n: (2 * n + 1) ** 2 * (2 * n + 2)}.items():
        cells, chi = _counts(sc, kind, 3)
        assert cells == [f(n) for n in range(4)] and set(chi) == {1}


def test_slab_polytopes_match_scipy_delaunay():
    """Independent method: scipy's Delaunay of a large patch of the p1 layer (3 generic sites at 3 heights), the
    tetrahedra meeting cell (0, 0) and well inside the patch, translation-normalised."""
    from scipy.spatial import Delaunay
    lay = LS.layer(1)
    L, P = lay["L"], lay["P"]
    ids, pts = [], []
    for n in itertools.product(range(-6, 7), repeat=2):
        for i, p in enumerate(P):
            ids.append((i, n[0], n[1], 0))
            pts.append(p + n[0] * L[0] + n[1] * L[1])
    tri = Delaunay(np.array(pts))
    ref = set()
    for s in tri.simplices:
        mem = [ids[k] for k in s]
        if any(m[1:3] == (0, 0) for m in mem) and all(max(abs(m[1]), abs(m[2])) <= 2 for m in mem):
            ref.add(_canonical(mem))
    ours = set(PeriodicNet(L, P).verts)
    assert all(len(t) == 4 for t in ours)          # generic: every Delaunay cell is a tetrahedron
    assert ours == ref


def _connected_two_periodic(m, bonds, lim=6):
    """Independent of the HNF: BFS on cells |n| <= lim reaches all sites of cell 0 and site 0 of (1,0), (0,1)."""
    adj = [[] for _ in range(m)]
    for i, j, n in bonds:
        adj[i].append((j, n))
        adj[j].append((i, (-n[0], -n[1])))
    seen, q = {(0, (0, 0))}, deque([(0, (0, 0))])
    while q:
        u, c = q.popleft()
        for v, n in adj[u]:
            w = (v, (c[0] + n[0], c[1] + n[1]))
            if max(map(abs, w[1])) <= lim and w not in seen:
                seen.add(w)
                q.append(w)
    return all((s, (0, 0)) in seen for s in range(m)) and (0, (1, 0)) in seen and (0, (0, 1)) in seen


def test_all_80_groups_exact_connected_and_invariant():
    for num in range(1, 81):
        g = LS.group(num)
        lay = LS.layer(num)
        general = g["point_order"] * (2 if g["centred"] else 1)      # from the group table
        assert len(lay["P"]) == (lay["labels"].max() + 1) * general, num
        assert lay["exact"] and lay["bonds"], num
        assert _connected_two_periodic(len(lay["P"]), lay["bonds"]), num
        assert LS.quotient_state(len(lay["P"]), lay["bonds"], 2) == (1, 2, 1)
        assert LS.bond_set_invariant(lay), num


def test_rank_index_and_offsets_by_hand():
    assert LS.lattice_rank_index([(2, 0), (0, 3)], 2) == (2, 6)
    assert LS.lattice_rank_index([(1, 1), (1, -1)], 2) == (2, 2)
    assert LS.lattice_rank_index([(2, 4), (-1, -2)], 2) == (1, 0)
    ok = lambda n, s: LS.offset_keeps_group(n, s, 3)   # odd m: only the lambda condition can fail
    # p1, p-1 any; p11m and p112 need 2s in Z^2; p4 needs s in {0, (1/2,1/2)}; p3 s in {0, (1/3,2/3), (2/3,1/3)}
    assert ok(1, (0.3, 0.1)) and ok(2, (0.37, 0.81))
    assert ok(4, (0.5, 0.0)) and not ok(4, (0.3, 0.0))
    assert ok(3, (0.5, 0.5)) and not ok(3, (0.25, 0.0))
    assert ok(49, (0.5, 0.5)) and not ok(49, (0.5, 0.0))
    assert ok(65, (1 / 3, 2 / 3)) and ok(65, (2 / 3, 1 / 3)) and not ok(65, (0.5, 0.0))
    # even m by hand (see the Rust test offset_rule_by_hand): p11m s = (1/2, 0) refused for m = 2, 4, kept for 3;
    # p112 s = (1/2, 0) kept for m = 2; p-1 kept for any s; one layer has nothing to check
    k = LS.offset_keeps_group
    assert not k(4, (0.5, 0), 2) and k(4, (0.5, 0), 3) and not k(4, (0.5, 0), 4)
    assert k(3, (0.5, 0), 2) and k(2, (0.37, 0.81), 2) and k(49, (0.3, 0), 1)
    with pytest.raises(LS.SlabRefusal, match="breaks"):
        LS.stack(LS.layer(4), 2, LS.layer(4)["thickness"] + 0.5, (0.5, 0.0))
    LS.stack(LS.layer(4), 3, LS.layer(4)["thickness"] + 0.5, (0.5, 0.0))


def test_one_site_square_net_has_coordination_four():
    # nearest shell of the unit square lattice: 4 neighbours at 1, connected at rank 2: 2 bonds per cell (hand)
    bonds, lengths = LS.shell_bonds(np.zeros((1, 3)), np.array([[1.0, 0, 0], [0, 1.0, 0]]), np.array([0]))
    assert len(bonds) == 2 and len(lengths) == 1 and abs(lengths[0] - 1) < 1e-12


def test_stacks():
    lay = LS.layer(61)
    d = lay["thickness"] + 0.1
    st = LS.stack(lay, 2, d)
    zmax = lay["P"][:, 2].max()
    top = int((np.abs(lay["P"][:, 2] - zmax) < 1e-9).sum())
    assert len(st["interlayer"]) == top == len(lay["P"]) // 2   # mirror z: each top site to its image above
    with pytest.raises(LS.SlabRefusal, match="interpenetrate"):
        LS.stack(lay, 2, lay["thickness"])
    with pytest.raises(LS.SlabRefusal, match="breaks"):
        LS.stack(LS.layer(49), 2, 2.0, (0.5, 0.0))
    one = LS.grow_slab(48, layers=2, m=1, spacing=1.3)
    two = LS.grow_slab(48, layers=2, m=2, spacing=1.3)
    s = sum(1 for b in two["bonds"] if b[2] == 1)
    for m in (1, 3):
        g = LS.grow_slab(48, layers=2, m=m, spacing=1.3)
        n = len(one["positions"])
        assert len(g["positions"]) == m * n
        assert sum(b[2] == 0 for b in g["bonds"]) == m * len(one["bonds"])
        assert sum(b[2] == 1 for b in g["bonds"]) == (m - 1) * s
        for k in range(1, m):
            assert np.allclose(g["positions"][k * n:(k + 1) * n, :2], g["positions"][:n, :2])


def test_slab_readout_flags_chi():
    g = LS.grow_slab(61, seed="cell", layers=3)
    assert g["mode"] == "slab" and not g["euler_defined"] and g["exact"]
    assert [s["cells"] for s in g["stats"]] == sorted(s["cells"] for s in g["stats"])


def test_parity_with_rust_dump():
    d = os.environ.get("PAPERTILLING_PARITY_DIR")
    path = Path(d) / "layer_slab_parity.json" if d else None
    if path is None or not path.exists():
        pytest.skip("no Rust dump (run the cargo test layer_slab_parity_dump with PAPERTILLING_PARITY_DIR set)")
    rows = json.loads(path.read_text("utf-8"))
    assert len(rows) == 80
    for row in rows:
        lay = LS.layer(row["number"])
        assert len(lay["P"]) == row["sites"]
        assert [[i, j, list(n)] for i, j, n in lay["bonds"]] == row["bonds"], row["number"]
        assert np.allclose(lay["shells"], row["shells"], rtol=1e-12)
        if "growth" in row:
            net = PeriodicNet(lay["L"], lay["P"])
            assert [[[t[0], list(t[1:])] for t in T] for T in net.verts] == row["polytopes"], row["number"]
            g = LS.grow_slab(row["number"], layers=3)
            assert [s["cells"] for s in g["stats"]] == row["growth"], row["number"]
            st = LS.stack(lay, 2, lay["thickness"] + 0.3)
            assert [[i, j, list(n)] for i, j, n in st["interlayer"]] == row["interlayer"], row["number"]
