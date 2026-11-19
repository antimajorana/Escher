"""Pure-Python fallback, part 2 of `honeycomb3d`: Dirichlet domains of closed 3-manifolds and the
clean-room P / D / G labyrinth nets and periodic surfaces (same dict keys as the Rust core).

Sources (published facts only; no EPINET / RCSR data): Weber & Seifert, Math. Z. 37 (1933) 237-253
(dodecahedral spaces); International Tables for Crystallography Vol. A, Wyckoff positions 221 1a,
227 8a, 214 8a (net sites); von Schnering & Nesper, Z. Phys. B 83 (1991) 407-412 (nodal surfaces);
Coxeter, Proc. London Math. Soc. (2) 43 (1937) 33-62 (regular skew polyhedra); Schoen, NASA TN
D-5541 (1970) (genus 3 of P, D, G per side-preserving primitive cell)."""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from .bieberbach import _snf

# Evidence stamps (identical strings in the Rust core, so Rust/Python dicts compare equal).
_QUOTIENT_EVIDENCE = "[PROVED: exact cell counts of the quotient complex; H_1 from the Poincare presentation (one generator per face pair, one relator per edge cycle) reduced by integer Smith normal form]"
_PC_EVIDENCE = ("[PROVED: exact V, E, F of the finite complex on the 3-torus of the conventional cell; genus 3 per "
                "side-preserving primitive cell, as for Coxeter's regular skew polyhedra (Proc. London Math. Soc. 43, 1937)]")
_NODAL_EVIDENCE = ("[COMPUTATIONAL EVIDENCE: topology of a marching-tetrahedra mesh of the nodal approximation on an n^3"
                   " grid; resolution-dependent, not a proof about the minimal surface]")
_TORI = {"cubic_3_torus": "SC", "fcc_3_torus": "FCC", "bcc_3_torus": "BCC"}
_TWISTS = {
    "seifert_weber": (3 * math.pi / 5, "H3", "regular {5,3,5} dodecahedron (dihedral angle 2π/5) in the Poincaré ball; "
                      "opposite faces glued with a 3π/5 twist"),
    "poincare_homology_sphere": (math.pi / 5, "S3", "regular spherical dodecahedron of the 120-cell {5,3,3} (dihedral angle "
                                 "2π/3); opposite faces glued with a π/5 twist; combinatorial model drawn with the same "
                                 "centred dodecahedron"),
}


def _hc():
    from . import honeycomb3d
    return honeycomb3d


def dirichlet_manifold(name: str) -> dict[str, Any]:
    """`{"domain": ..., "quotient": ...}` for 'seifert_weber' (H³), 'poincare_homology_sphere' (S³) or the
    Euclidean 3-tori 'cubic_3_torus', 'fcc_3_torus', 'bcc_3_torus'. The quotient has counts, Euler
    characteristic and H_1 (Poincaré presentation: face pairs / edge cycles, Smith normal form)."""
    out = _hc()._call("honeycomb3d_dirichlet_json", name)
    if out is not None:
        return out
    if name in _TORI:
        cell = _hc().lattice_voronoi_cell(_TORI[name])
        vs, fn = np.array(cell["vertices"]), np.array(cell["face_neighbours"])
        pairings = []
        for f, b in enumerate(fn):
            partner = int(np.argmin(np.linalg.norm(fn + b, axis=1)))
            if partner > f:
                vm = [[v, min(cell["faces"][partner], key=lambda w: np.linalg.norm(vs[w] - vs[v] + b))] for v in cell["faces"][f]]
                pairings.append(dict(face=f, partner=partner, vertex_map=vm))
        dom = dict(name=name, geometry="E3", vertices=cell["vertices"], faces=cell["faces"], pairings=pairings,
                   note=f"Voronoi cell of the {_TORI[name]} lattice, opposite faces identified by lattice translations")
    elif name in _TWISTS:
        phi, geo, note = _TWISTS[name]
        cell = _hc().build_honeycomb(5, 3, 5, 0, 1)
        dom = dict(name=name, geometry=geo, note=note, vertices=cell["points"], faces=cell["faces"],
                   pairings=_twisted_pairings(np.array(cell["points"]), cell["faces"], phi))
    else:
        raise ValueError(f"unknown manifold {name!r} (seifert_weber, poincare_homology_sphere, cubic_3_torus, "
                         "fcc_3_torus, bcc_3_torus)")
    return dict(domain=dom, quotient=_quotient(dom))


def _rotate(v, k, phi):
    return v * math.cos(phi) + np.cross(k, v) * math.sin(phi) + k * (k @ v) * (1 - math.cos(phi))


def _twisted_pairings(vs, faces, phi):
    cents = np.array([vs[f].mean(axis=0) for f in faces])
    edge = np.linalg.norm(vs[faces[0][0]] - vs[faces[0][1]])
    out = []
    for f, c in enumerate(cents):
        partner = int(np.argmin(np.linalg.norm(cents + c, axis=1)))
        if partner < f:
            continue
        axis = (cents[partner] - c) / np.linalg.norm(cents[partner] - c)
        vm = []
        for v in faces[f]:
            target = cents[partner] + _rotate(vs[v] - c, axis, phi)
            w = min(faces[partner], key=lambda u: np.linalg.norm(vs[u] - target))
            if np.linalg.norm(vs[w] - target) > 0.2 * edge:
                raise ValueError(f"twist {phi} does not map face {f} onto face {partner} vertex to vertex")
            vm.append([v, w])
        out.append(dict(face=f, partner=partner, vertex_map=vm))
    return out


def _find(par, x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


def _sizes(par):
    roots = [_find(par, i) for i in range(len(par))]
    return sorted(roots.count(r) for r in set(roots))


def _quotient(dom):
    faces, pairs = dom["faces"], dom["pairings"]
    info = {}
    for j, pr in enumerate(pairs):
        fwd = {a: b for a, b in pr["vertex_map"]}
        info[pr["face"]] = (j, 1, pr["partner"], fwd)
        info[pr["partner"]] = (j, -1, pr["face"], {b: a for a, b in fwd.items()})
    if len(info) != len(faces):
        raise ValueError("faces are not paired exactly once")
    eid, efaces = {}, []
    for f, cyc in enumerate(faces):
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            e = (min(a, b), max(a, b))
            if e not in eid:
                eid[e] = len(efaces)
                efaces.append([])
            efaces[eid[e]].append(f)
    edges = sorted(eid, key=eid.get)
    vpar, epar = list(range(len(dom["vertices"]))), list(range(len(edges)))

    def map_edge(e, m):
        a, b = m[e[0]], m[e[1]]
        return eid[(min(a, b), max(a, b))]

    for pr in pairs:
        for a, b in pr["vertex_map"]:
            vpar[_find(vpar, a)] = _find(vpar, b)
        cyc = faces[pr["face"]]
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            e0 = eid[(min(a, b), max(a, b))]
            epar[_find(epar, e0)] = _find(epar, map_edge(edges[e0], info[pr["face"]][3]))
    rows, seen = [], set()
    for e0 in range(len(edges)):
        root = _find(epar, e0)
        if root in seen:
            continue
        seen.add(root)
        row, e, f = [0] * len(pairs), e0, efaces[e0][0]
        for _ in range(10000):
            j, s, partner, m = info[f]
            row[j] += s
            e = map_edge(edges[e], m)
            others = [g for g in efaces[e] if g != partner]
            f = others[0]
            if (e, f) == (e0, efaces[e0][0]):
                break
        else:
            raise ValueError("edge cycle did not close")
        rows.append(row)
    diag = [abs(d) for d in _snf(rows, track_uv=False)[0] if d != 0]
    free = len(pairs) - len(diag)
    tors = [d for d in diag if d > 1]
    parts = ([] if free == 0 else ["Z" if free == 1 else f"Z^{free}"])
    for d, grp in itertools.groupby(tors):
        c = len(list(grp))
        parts.append(f"Z_{d}" if c == 1 else f"Z_{d}^{c}")
    es, vsz = _sizes(epar), _sizes(vpar)
    return dict(cells=1, faces=len(pairs), edges=len(es), vertices=len(vsz), euler=len(vsz) - len(es) + len(pairs) - 1,
                edge_class_sizes=es, vertex_class_sizes=vsz, h1_torsion=tors, h1_free_rank=free,
                h1_label=" + ".join(parts) if parts else "0", evidence=_QUOTIENT_EVIDENCE)


# ---------------------------------------------------------------- P / D / G

_NETS = {("P", 1): ("pcu (Schwarz P labyrinth)", "Pm-3m", 221), ("P", 2): ("pcu pair (both Schwarz P labyrinths)", "Im-3m", 229),
         ("D", 1): ("dia (Schwarz D labyrinth)", "Fd-3m", 227), ("D", 2): ("double diamond (both Schwarz D labyrinths)", "Pn-3m", 224),
         ("G", 1): ("srs (gyroid labyrinth, chiral)", "I4_132", 214), ("G", 2): ("double gyroid (both gyroid labyrinths)", "Ia-3d", 230)}
_SIDE_CELLS = {"P": 1, "D": 4, "G": 2}
_FCC = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


def _kind(kind):
    k = str(kind).upper()
    if k not in _SIDE_CELLS:
        raise ValueError(f"unknown triply periodic surface {kind!r} (P, D, G)")
    return k


def nodal(kind: str, x) -> np.ndarray:
    """Classical nodal approximation of P / D / G at fractional coordinates `x` (last axis 3)."""
    X, Y, Z = np.moveaxis(2 * np.pi * np.asarray(x, float), -1, 0)
    k = _kind(kind)
    if k == "P":
        return np.cos(X) + np.cos(Y) + np.cos(Z)
    if k == "D":
        return np.cos(X) * np.cos(Y) * np.cos(Z) + np.sin(X) * np.sin(Y) * np.sin(Z)
    return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)


def tpms_net(kind: str = "G", labyrinths: int = 1) -> dict[str, Any]:
    """Labyrinth net(s) of P (pcu), D (dia) or G (srs) in the conventional cubic cell; `labyrinths` 1 or 2."""
    out = _hc()._call("honeycomb3d_tpms_net_json", kind, labyrinths)
    if out is not None:
        return out
    k = _kind(kind)
    if labyrinths not in (1, 2):
        raise ValueError("labyrinths must be 1 or 2")
    if k == "P":
        a = np.zeros((1, 3))
    elif k == "D":
        a = np.vstack([_FCC, _FCC + 0.25])
    else:
        base = np.array([[1, 1, 1], [3, 7, 5], [7, 5, 3], [5, 3, 7]]) / 8
        a = np.vstack([base, base + 0.5]) % 1.0
    pos, lab = [a], [0] * len(a)
    if labyrinths == 2:
        pos.append((-a) % 1.0 if k == "G" else (a + 0.5) % 1.0)
        lab += [1] * len(a)
    pos = np.vstack(pos)
    cands = []
    for i, j in itertools.product(range(len(pos)), repeat=2):
        if lab[i] == lab[j]:
            for off in itertools.product((-1, 0, 1), repeat=3):
                if i != j or any(off):
                    cands.append((i, j, list(off), float(np.linalg.norm(pos[j] + off - pos[i]))))
    dmin = min(c[3] for c in cands)
    coord, edges = [0] * len(pos), []
    for i, j, off, d in cands:
        if d <= dmin * (1 + 1e-6):
            coord[i] += 1
            if i < j or (i == j and off > [0, 0, 0]):
                edges.append([i, j, off])
    name, sg, num = _NETS[(k, labyrinths)]
    return dict(name=name, kind=k, space_group=sg, space_group_number=num, labyrinths=labyrinths, positions=pos.tolist(),
                labyrinth_of=lab, edges=edges, coordination=coord, bond_length=dmin)


def _wrap(x):
    w = np.mod(np.asarray(x, float), 1.0)
    return np.where(w > 1 - 1e-9, 0.0, w)


def _surface(name, k, verts, faces, evidence):
    count, par = {}, list(range(len(verts)))
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            e = (min(a, b), max(a, b))
            count[e] = count.get(e, 0) + 1
            par[_find(par, a)] = _find(par, b)
    chi = len(verts) - len(count) + len(faces)
    cp = chi // _SIDE_CELLS[k]
    return dict(name=name, kind=k, vertices=[list(map(float, v)) for v in verts], faces=faces,
                edges=[list(e) for e in sorted(count)], euler_conventional=chi, side_preserving_cells=_SIDE_CELLS[k],
                euler_primitive=cp, genus_primitive=1 - cp // 2,
                connected_components=sum(_find(par, i) == i for i in range(len(verts))),
                closed_manifold=all(c == 2 for c in count.values()), evidence=evidence)


def petrie_coxeter_surface(kind: str = "P") -> dict[str, Any]:
    """{6,4|4} (P: hexagons of the Kelvin honeycomb) or {6,6|3} (D: hexagons between truncated tetrahedra
    on the diamond sites) on the 3-torus of the conventional cubic cell."""
    out = _hc()._call("honeycomb3d_petrie_coxeter_json", kind)
    if out is not None:
        return out
    k = _kind(kind)
    hexes = []
    if k == "P":
        cell = _hc().lattice_voronoi_cell("BCC")
        vs = np.array(cell["vertices"])
        hexes = [[(vs[v] + t) / 2 for v in f] for t in (np.zeros(3), np.ones(3)) for f in cell["faces"] if len(f) == 6]
        name = "{6,4|4} Petrie-Coxeter polyhedron (hexagons of the Kelvin honeycomb, P surface)"
    elif k == "D":
        tt = [np.array(p) * s for p in set(itertools.permutations((1, 1, 3))) for s in itertools.product((1, -1), repeat=3)]
        even = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
        for a in _FCC:
            for s in even:
                for centre, face, parity in ((a, s, 1), (a + 0.25, -s, 0)):
                    pts = [x / 8 for x in tt if int((x < 0).sum()) % 2 == parity and abs(x @ face - 3) < 1e-9]
                    hexes.append([centre + x for x in _hc()._order_around(pts, face)])
        name = "{6,6|3} Petrie-Coxeter polyhedron (hexagons between truncated tetrahedra on diamond sites, D surface)"
    else:
        raise ValueError("no regular Petrie-Coxeter polyhedron lies on the gyroid; use nodal_surface_mesh")
    idx, verts, faces, keys = _hc()._Index(), [], [], set()
    for h in hexes:
        cyc = []
        for x in h:
            i, new = idx.insert(_wrap(x))
            if new:
                verts.append(_wrap(x))
            cyc.append(i)
        if tuple(sorted(cyc)) not in keys:
            keys.add(tuple(sorted(cyc)))
            faces.append(cyc)
    return _surface(name, k, verts, faces, _PC_EVIDENCE)


def nodal_surface_mesh(kind: str = "G", n: int = 24) -> dict[str, Any]:
    """Marching-tetrahedra mesh (Kuhn subdivision) of the nodal P / D / G surface on a periodic n³ grid."""
    out = _hc()._call("honeycomb3d_nodal_mesh_json", kind, n)
    if out is not None:
        return out
    k = _kind(kind)
    if not 4 <= n <= 96:
        raise ValueError("n must be in 4..=96")
    delta = 0.37
    g = (np.indices((n, n, n)).transpose(1, 2, 3, 0) + delta) / n
    vals = nodal(k, g)
    vert_of, verts, tris = {}, [], []
    tets = []
    for perm in itertools.permutations(range(3)):
        c = [0]
        for s in range(3):
            c.append(c[-1] | (1 << perm[s]))
        tets.append(c)
    bits = [np.array([b & 1, b >> 1 & 1, b >> 2 & 1]) for b in range(8)]

    def crossing(p0, a, b):
        lo, hi = min(a, b), max(a, b)
        p = (p0 + bits[lo]) % n
        key = (tuple(p), hi ^ lo)
        if key not in vert_of:
            fa, fb = vals[tuple(p)], vals[tuple((p0 + bits[hi]) % n)]
            vert_of[key] = len(verts)
            verts.append((p + delta + fa / (fa - fb) * bits[hi ^ lo]) / n)
        return vert_of[key]

    for p0 in itertools.product(range(n), repeat=3):
        p0 = np.array(p0)
        cv = [vals[tuple((p0 + bits[b]) % n)] for b in range(8)]
        if max(cv) <= 0 or min(cv) > 0:
            continue
        for tet in tets:
            ins = [b for b in tet if cv[b] > 0]
            outs = [b for b in tet if cv[b] <= 0]
            if len(ins) == 1:
                tris.append([crossing(p0, ins[0], o) for o in outs])
            elif len(ins) == 3:
                tris.append([crossing(p0, outs[0], o) for o in ins])
            elif len(ins) == 2:
                a, b, c, d = ins + outs
                ac, ad, bd, bc = (crossing(p0, x, y) for x, y in ((a, c), (a, d), (b, d), (b, c)))
                tris += [[ac, ad, bd], [ac, bd, bc]]
    names = {"P": "Schwarz P nodal surface cos X + cos Y + cos Z = 0",
             "D": "Schwarz D nodal surface cos X cos Y cos Z + sin X sin Y sin Z = 0",
             "G": "Schoen gyroid nodal surface sin X cos Y + sin Y cos Z + sin Z cos X = 0"}
    return _surface(names[k], k, [_wrap(v) for v in verts], tris, _NODAL_EVIDENCE)
