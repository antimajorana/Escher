"""Solver-facing exports: extended XYZ writers, periodic neighbour shells & space-group bond orbits,
and site-permutation symmetry groups for periodic tori and finite defect clusters.

Delegates to `papertilling_rs` when available and `PAPERTILLING_NO_RUST` is unset, with an exact
pure-Python + NumPy fallback producing identical output formats and schemas.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from . import _rust
from .legacy.wallpaper_lattices import UnitCell, _symmetry_ops

DEFAULT_SPECIES = ("C", "B", "N", "Si", "P", "S", "Ge", "As")
OPEN_BOX_PADDING, SLAB_C, CLUSTER_TOL = 10.0, 20.0, 1e-4
_PROPS = "Properties=species:S:1:pos:R:3:sublattice:I:1:coordination:I:1"


def _fmt_f8(x: float) -> str:
    s = f"{(0.0 if abs(float(x)) < 5e-13 else float(x)):.8f}"
    return "0.00000000" if s == "-0.00000000" else s


def _validate_element_symbol(sym: str) -> str:
    ok = isinstance(sym, str) and len(sym) in (1, 2) and sym.isascii() and sym[0].isupper()
    if not (ok and (len(sym) == 1 or sym[1].islower())):
        raise ValueError(f"invalid element symbol {sym!r} (expected e.g. 'C', 'Si')")
    return sym


def _pick_species(n: int, labels: list[int], species: Any) -> list[str]:
    if species is None:
        return [DEFAULT_SPECIES[int(b) % 8] for b in labels]
    sp = [str(s) for s in species]
    if len(sp) != n:
        raise ValueError(f"species has {len(sp)} entries, expected {n}")
    return [_validate_element_symbol(s) for s in sp]


def _is_periodic_obj(obj: Any) -> bool:
    if isinstance(obj, UnitCell):
        return True
    if isinstance(obj, dict):
        return "a1" in obj and "a2" in obj and "basis" in obj
    return hasattr(obj, "cell") and callable(obj.cell) and hasattr(obj, "group")


def _extract_cell(obj: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(obj, dict) and "a1" in obj and "a2" in obj and "basis" in obj:
        bonds = [
            {"i": int(b["i"]), "j": int(b["j"]), "n1": int(b["n1"]), "n2": int(b["n2"]), "kind": int(b.get("kind", 0))}
            if isinstance(b, dict)
            else {"i": int(b[0]), "j": int(b[1]), "n1": int(b[2]), "n2": int(b[3]), "kind": int(b[4]) if len(b) > 4 else 0}
            for b in obj.get("bonds", ())
        ]
        return str(obj.get("group", "p1")), {
            "a1": [float(obj["a1"][0]), float(obj["a1"][1])],
            "a2": [float(obj["a2"][0]), float(obj["a2"][1])],
            "basis": [[float(p[0]), float(p[1])] for p in obj["basis"]],
            "bonds": bonds,
        }
    uc = obj.cell() if (hasattr(obj, "cell") and callable(obj.cell)) else obj
    if not isinstance(uc, UnitCell):
        raise TypeError(f"expected Lattice, UnitCell, or cell dict, got {type(obj).__name__}")
    group = str(getattr(obj, "group", None) or getattr(uc, "group", "p1"))
    return group, {
        "a1": [float(uc.a1[0]), float(uc.a1[1])], "a2": [float(uc.a2[0]), float(uc.a2[1])],
        "basis": [[float(p[0]), float(p[1])] for p in uc.basis],
        "bonds": [{"i": int(b[0]), "j": int(b[1]), "n1": int(b[2]), "n2": int(b[3]), "kind": int(b[4])} for b in uc.bonds],
    }


def _cell_ops_with_action(cell: dict[str, Any], group: str = "p1") -> list[tuple[list[list[int]], list[int], list[list[int]]]]:
    a1, a2 = cell["a1"], cell["a2"]
    if abs(a1[0] * a2[1] - a1[1] * a2[0]) < 1e-12:
        raise ValueError("singular cell")
    uc = UnitCell(
        "tmp", "tmp", group, np.asarray(a1, float), np.asarray(a2, float),
        np.asarray(cell["basis"], float), [(b["i"], b["j"], b["n1"], b["n2"], b["kind"]) for b in cell["bonds"]],
    )
    ops = _symmetry_ops(uc, use_types=True)
    if not ops:
        raise ValueError("cell has no symmetry operations")
    fr = [np.linalg.inv(uc.A) @ np.asarray(p, float) for p in cell["basis"]]
    out = []
    for mf, t in ops:
        perm, shift = [], []
        for f in fr:
            g = mf @ f + t
            hit = next(
                ((j, [int(r[0]), int(r[1])]) for j, fj in enumerate(fr)
                 for d in (g - fj,) for r in (np.round(d),)
                 if abs(d[0] - r[0]) < 1e-6 and abs(d[1] - r[1]) < 1e-6),
                None,
            )
            if hit is None:
                raise ValueError("symmetry op does not permute the basis")
            perm.append(hit[0])
            shift.append(hit[1])
        out.append(([[int(mf[0, 0]), int(mf[0, 1])], [int(mf[1, 0]), int(mf[1, 1])]], perm, shift))
    return out


def _extract_crystal(obj: Any) -> tuple[dict[str, Any], list[float] | None, list[str] | None, list[int]]:
    if isinstance(obj, dict) and "crystal" in obj and isinstance(obj["crystal"], dict):
        cr = obj["crystal"]
        z_vals = [float(p[2]) for p in obj["pos3d"]] if "pos3d" in obj else None
        sp_vals = [str(s) for s in obj["species"]] if "species" in obj else None
        uniq: dict[str, int] = {}
        labels = [uniq.setdefault(s, len(uniq)) for s in sp_vals] if sp_vals is not None else [0] * len(cr["pos"])
        return cr, z_vals, sp_vals, labels
    if isinstance(obj, dict) and "pos" in obj and "adj" in obj:
        return obj, None, obj.get("species"), [int(x) for x in obj.get("site_labels", [0] * len(obj["pos"]))]

    lat_obj, d = getattr(obj, "lattice", None), getattr(obj, "defect", obj)
    if not (hasattr(d, "positions") and hasattr(d, "edge") and hasattr(d, "edges")):
        raise TypeError(f"unsupported object for defect/cluster export: {type(obj).__name__}")
    pos = np.asarray(d.positions, float)
    n, ref = len(pos), np.asarray(d.info.get("reference", pos), float)
    non_ori = bool(d.info.get("mobius") or d.info.get("klein") or d.info.get("_orientable") is False)
    basis_idx, side = [], []
    for s in d.sites:
        key = s[0] if isinstance(s, (tuple, list)) and len(s) >= 1 and isinstance(s[0], (tuple, list)) else s
        basis_idx.append(int(key[2]) if isinstance(key, (tuple, list)) and len(key) >= 3 else 0)
        tag = s[1] if (not non_ori and isinstance(s, (tuple, list)) and len(s) >= 2 and not isinstance(s[0], int)) else None
        side.append("A" if tag in ("A", 0) else ("B" if tag in ("B", 1) else None))

    from .defects.circuit import _edge_vec_and_g
    adj: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for (u, v), e in sorted(d.edge.items()):
        vec, g = _edge_vec_and_g(d, int(u), int(v))
        seam = {"m": [[float(g[0][0, 0]), float(g[0][0, 1])], [float(g[0][1, 0]), float(g[0][1, 1])]],
                "t": [float(g[1][0]), float(g[1][1])]} if g is not None else None
        ghost = e.get("ghost")
        adj[int(u)].append({
            "to": int(v), "vec": [float(vec[0]), float(vec[1])], "kind": int(e.get("kind", e.get("type", 0)) or 0),
            "seam": seam, "ang": float(e["ang"]) if e.get("ang") is not None else None,
            "across": bool(e.get("across", False)),
            "ghost_stub": [float(ghost[0]), float(ghost[1])] if ghost is not None else None,
        })
    crystal = {
        "pos": pos.tolist(), "ref_pos": ref.tolist(), "basis_idx": basis_idx, "side": side,
        "cone_sites": sorted(int(s) for s in d.info.get("cone_sites", ())),
        "edges": [[int(u), int(v)] for u, v in d.edges], "adj": adj, "wall_sigma": None, "is_non_orientable": non_ori,
    }
    if lat_obj is not None and hasattr(lat_obj, "cell"):
        _, cell_d = _extract_cell(lat_obj.cell)
        ops = _cell_ops_with_action(cell_d)
        orbit_map = [min(perm[s] for _, perm, _ in ops) for s in range(len(cell_d["basis"]))]
        labels = [orbit_map[b % len(orbit_map)] for b in basis_idx]
    else:
        labels = [0] * n
    return crystal, None, None, labels


def _write_xyz(lat9: list[float], pbc: str, rows: list[tuple[str, list[float], int, int]]) -> str:
    lat_str = " ".join(_fmt_f8(v) for v in lat9)
    lines = [str(len(rows)), f'Lattice="{lat_str}" {_PROPS} pbc="{pbc}"']
    lines.extend(f"{sp} {_fmt_f8(p[0])} {_fmt_f8(p[1])} {_fmt_f8(p[2])} {sub} {coord}" for sp, p, sub, coord in rows)
    return "\n".join(lines) + "\n"


def _open_box(pts: list[list[float]]) -> list[float]:
    if not pts:
        return [OPEN_BOX_PADDING, 0.0, 0.0, 0.0, OPEN_BOX_PADDING, 0.0, 0.0, 0.0, OPEN_BOX_PADDING]
    arr = np.asarray(pts, float)
    ext = arr.max(axis=0) - arr.min(axis=0) + OPEN_BOX_PADDING
    return [float(ext[0]), 0.0, 0.0, 0.0, float(ext[1]), 0.0, 0.0, 0.0, float(ext[2])]


def export_extxyz(obj: Any, *, supercell: tuple[int, int] = (1, 1), species: Any = None, z: Any = None) -> str:
    """Export a periodic `Lattice`/`UnitCell` (`pbc="T T F"`) or open defect crystal (`pbc="F F F"`) to Extended XYZ."""
    rs = _rust.get()
    if _is_periodic_obj(obj):
        nx, ny = int(supercell[0]), int(supercell[1])
        if nx < 1 or ny < 1:
            raise ValueError("supercell sizes nx, ny must be >= 1")
        _, cell = _extract_cell(obj)
        sp_arg = None if species is None else [str(s) for s in species]
        if rs is not None and hasattr(rs, "export_cell_extxyz_rs"):
            return str(rs.export_cell_extxyz_rs(json.dumps(cell), nx, ny, None if sp_arg is None else json.dumps(sp_arg)))
        nb = len(cell["basis"])
        sp, coord = _pick_species(nb, list(range(nb)), sp_arg), [0] * nb
        for b in cell["bonds"]:
            coord[b["i"]] += 1
            coord[b["j"]] += 1
        a1, a2 = cell["a1"], cell["a2"]
        rows = [
            (sp[s], [p[0] + ix * a1[0] + iy * a2[0], p[1] + ix * a1[1] + iy * a2[1], 0.0], s, coord[s])
            for iy in range(ny) for ix in range(nx) for s, p in enumerate(cell["basis"])
        ]
        return _write_xyz([nx * a1[0], nx * a1[1], 0.0, ny * a2[0], ny * a2[1], 0.0, 0.0, 0.0, SLAB_C], "T T F", rows)

    crystal, z_auto, sp_auto, _ = _extract_crystal(obj)
    n = len(crystal["pos"])
    sp_use = sp_auto if species is None else [str(s) for s in species]
    z_use = z_auto if z is None else [float(v) for v in z]
    if rs is not None and hasattr(rs, "export_crystal_extxyz_rs"):
        return str(rs.export_crystal_extxyz_rs(
            json.dumps(crystal), None if sp_use is None else json.dumps(sp_use), None if z_use is None else json.dumps(z_use),
        ))
    if z_use is not None and len(z_use) != n:
        raise ValueError(f"z has {len(z_use)} entries, expected {n}")
    sp = _pick_species(n, crystal["basis_idx"], sp_use)
    pts = [[float(crystal["pos"][i][0]), float(crystal["pos"][i][1]), 0.0 if z_use is None else float(z_use[i])]
           for i in range(n)]
    rows = [(sp[i], pts[i], int(crystal["basis_idx"][i]), len(crystal["adj"][i])) for i in range(n)]
    return _write_xyz(_open_box(pts), "F F F", rows)


def _canon_bond(i: int, j: int, n1: int, n2: int) -> tuple[int, int, int, int]:
    return (i, j, n1, n2) if (i, j, n1, n2) <= (j, i, -n1, -n2) else (j, i, -n1, -n2)


def _classify_shells_py(group: str, cell: dict[str, Any], max_shells: int) -> dict[str, Any]:
    nb = len(cell["basis"])
    if nb == 0 or max_shells == 0:
        raise ValueError("need at least one basis site and max_shells >= 1")
    ops = _cell_ops_with_action(cell, group)
    a1, a2, basis = np.asarray(cell["a1"], float), np.asarray(cell["a2"], float), np.asarray(cell["basis"], float)
    h = abs(a1[0] * a2[1] - a1[1] * a2[0]) / max(float(np.linalg.norm(a1)), float(np.linalg.norm(a2)))
    diam = max((float(np.linalg.norm(p - q)) for p in basis for q in basis), default=0.0)
    k = 2
    while True:
        r_safe = k * h - diam
        pairs: list[tuple[int, int, int, int, float]] = []
        if r_safe > 0.0:
            ns = np.arange(-k, k + 1)
            g1, g2 = np.meshgrid(ns, ns, indexing="ij")
            g1f, g2f = g1.ravel(), g2.ravel()
            shifts = np.outer(g1f, a1) + np.outer(g2f, a2)
            for i in range(nb):
                for j in range(nb):
                    ds = np.linalg.norm(shifts + (basis[j] - basis[i]), axis=1)
                    for idx_m in np.nonzero((ds > 1e-8) & (ds <= r_safe))[0]:
                        pairs.append((i, j, int(g1f[idx_m]), int(g2f[idx_m]), float(ds[idx_m])))
        all_d = sorted(p[4] for p in pairs)
        dist: list[float] = []
        if all_d:
            d1_tol = 1e-5 * all_d[0]
            for d in all_d:
                if not dist or d - dist[-1] > d1_tol:
                    dist.append(d)
        if len(dist) > max_shells:
            break
        if k > 4096:
            raise ValueError("could not resolve the requested number of shells")
        k *= 2

    tol, shells = 1e-5 * dist[0], []
    for s, ds in enumerate(dist[:max_shells]):
        members = [p for p in pairs if abs(p[4] - ds) <= tol]
        coord = [sum(1 for p in members if p[0] == i) for i in range(nb)]
        bonds = sorted({_canon_bond(p[0], p[1], p[2], p[3]) for p in members})
        b_idx, orbit_id, orbits = {b: idx for idx, b in enumerate(bonds)}, [None] * len(bonds), []
        for start, b0 in enumerate(bonds):
            if orbit_id[start] is not None:
                continue
            oid, mem_o = len(orbits), set()
            for mf, perm, shift in ops:
                n1 = mf[0][0] * b0[2] + mf[0][1] * b0[3] + shift[b0[1]][0] - shift[b0[0]][0]
                n2 = mf[1][0] * b0[2] + mf[1][1] * b0[3] + shift[b0[1]][1] - shift[b0[0]][1]
                img = _canon_bond(perm[b0[0]], perm[b0[1]], n1, n2)
                idx = b_idx.get(img)
                if idx is None:
                    raise ValueError("symmetry image left the shell")
                orbit_id[idx] = oid
                mem_o.add(img)
            ob, oc = sorted(mem_o), [0] * nb
            for bi, bj, _, _ in ob:
                oc[bi] += 1; oc[bj] += 1
            orbits.append({
                "representative": {"i": ob[0][0], "j": ob[0][1], "n1": ob[0][2], "n2": ob[0][3]},
                "bonds_per_cell": len(ob), "coordination_by_basis": oc, "mean_coordination": float(sum(oc)) / nb,
                "bonds": [{"i": b[0], "j": b[1], "n1": b[2], "n2": b[3]} for b in ob],
            })
        orbits.sort(key=lambda o: tuple(o["representative"].values()))
        shells.append({
            "shell": s + 1, "distance": round(ds, 10), "total_directed_per_cell": len(members),
            "coordination_by_basis": coord, "mean_coordination": float(len(members)) / nb, "orbits": orbits,
        })
    return {
        "group": str(group), "n_basis": nb, "n_symmetry_ops": len(ops), "shells": shells,
        "provenance": "[PROVED: exact integer orbit closure under the bond-preserving cell symmetry]",
    }


def classify_bond_shells(lattice: Any, *, max_shells: int = 3) -> dict[str, Any]:
    """Classify periodic neighbour shells `1..=max_shells` and split each into exact space-group bond orbits."""
    group, cell = _extract_cell(lattice)
    rs = _rust.get()
    if rs is not None and hasattr(rs, "classify_cell_shells_json"):
        return json.loads(rs.classify_cell_shells_json(group, json.dumps(cell), int(max_shells)))
    return _classify_shells_py(group, cell, int(max_shells))


def _group_closed(perms: list[list[int]]) -> bool:
    pset = {tuple(p) for p in perms}
    return bool(perms) and all(tuple(g[k] for k in h) in pset for g in perms for h in perms)


def _frame3(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    e1 = a / np.linalg.norm(a)
    helper = np.array([1.0, 0.0, 0.0] if abs(e1[0]) < 0.9 else [0.0, 1.0, 0.0]) if b is None else b
    w = helper - np.dot(helper, e1) * e1
    e2 = w / np.linalg.norm(w)
    return np.vstack([e1, e2, np.cross(e1, e2)])


def _cluster_perms_py(positions: list[list[float]], labels: list[int], shells: list[list[tuple[int, int]]]) -> dict[str, Any]:
    n, tol = len(positions), CLUSTER_TOL
    if n == 0 or len(labels) != n:
        raise ValueError("need >= 1 site and one label per site")
    edge_sets = [{(min(a, b), max(a, b)) for a, b in sh} for sh in shells]
    x = np.asarray(positions, float)
    x = x - x.mean(axis=0)
    r = np.linalg.norm(x, axis=1)
    cands = lambda i: [q for q in range(n) if labels[q] == labels[i] and abs(r[q] - r[i]) < tol]
    off = [i for i in range(n) if r[i] > 10.0 * tol]
    p0 = min(off, key=lambda i: (len(cands(i)), i)) if off else None
    p1 = min((j for j in off if np.linalg.norm(np.cross(x[p0], x[j])) > 1e-3 * r[p0] * r[j]),
             key=lambda j: (len(cands(j)), j), default=None) if p0 is not None else None
    cell_w, grid = 10.0 * tol, {}
    for i, p in enumerate(x):
        grid.setdefault((math.floor(p[0] / cell_w), math.floor(p[1] / cell_w), math.floor(p[2] / cell_w)), []).append(i)
    found: set[tuple[int, ...]] = set()

    def try_map(o: np.ndarray) -> None:
        perm: list[int] = []
        for i, q in enumerate(x @ o.T):
            kx, ky, kz = math.floor(q[0] / cell_w), math.floor(q[1] / cell_w), math.floor(q[2] / cell_w)
            c_idx = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                     for j in grid.get((kx + dx, ky + dy, kz + dz), ())]
            best_d, best_j = min(((float(np.linalg.norm(x[j] - q)), j) for j in c_idx), default=(tol, -1))
            if best_d >= tol or best_j < 0 or labels[best_j] != labels[i]:
                return
            perm.append(best_j)
        if len(set(perm)) == n and all(all((min(perm[a], perm[b]), max(perm[a], perm[b])) in es for a, b in es) for es in edge_sets):
            found.add(tuple(perm))

    def try_pair(e: np.ndarray, q0: int, q1: int | None) -> None:
        f = _frame3(x[q0], None if q1 is None else x[q1])
        for flip in (1.0, -1.0):
            try_map(np.vstack([f[0], f[1], flip * f[2]]).T @ e)

    if p0 is None:
        found.add(tuple(range(n)))
    elif p1 is None:
        e = _frame3(x[p0])
        for q0 in cands(p0):
            try_pair(e, q0, None)
    else:
        e, g01, c1, lim = _frame3(x[p0], x[p1]), float(np.dot(x[p0], x[p1])), cands(p1), tol * (1.0 + r[p0] + r[p1])
        for q0 in cands(p0):
            for q1 in c1:
                if q1 != q0 and abs(float(np.dot(x[q0], x[q1])) - g01) < lim:
                    try_pair(e, q0, q1)

    perms = [list(p) for p in sorted(found)]
    return {
        "n_sites": n, "order": len(perms), "permutations": perms, "group_closed": _group_closed(perms),
        "provenance": "[COMPUTATIONAL EVIDENCE: geometric search, tol 1e-4; closure checked exactly]",
    }


def _periodic_perms_py(group: str, cell: dict[str, Any], nx: int, ny: int, max_shells: int) -> dict[str, Any]:
    if nx < 1 or ny < 1:
        raise ValueError("supercell sizes nx, ny must be >= 1")
    shells, ops = _classify_shells_py(group, cell, max_shells), _cell_ops_with_action(cell, group)
    nb, a1, a2 = len(cell["basis"]), cell["a1"], cell["a2"]
    n = nx * ny * nb
    site_label = [min(perm[s] for _, perm, _ in ops) for s in range(nb)]
    positions = [[p[0] + ix * a1[0] + iy * a2[0], p[1] + ix * a1[1] + iy * a2[1], 0.0]
                 for iy in range(ny) for ix in range(nx) for p in cell["basis"]]
    sublattice = [s for _ in range(nx * ny) for s in range(nb)]
    site_labels = [site_label[s] for s in sublattice]
    idx = lambda ix, iy, s: ((iy % ny) * nx + (ix % nx)) * nb + s

    adjs, shell_edges = [], []
    for sh in shells["shells"]:
        a = np.zeros((n, n), dtype=np.int64)
        for orb in sh["orbits"]:
            for bd in orb["bonds"]:
                for iy in range(ny):
                    for ix in range(nx):
                        u, v = idx(ix, iy, bd["i"]), idx(ix + bd["n1"], iy + bd["n2"], bd["j"])
                        a[u, v] += 1
                        a[v, u] += 1
        shell_edges.append([[u, v, int(a[u, v])] for u in range(n) for v in range(u, n) if a[u, v] > 0])
        adjs.append(a)

    found: set[tuple[int, ...]] = set()
    for mf, perm, shift in ops:
        if (mf[1][0] * nx) % ny != 0 or (mf[0][1] * ny) % nx != 0:
            continue
        for t2 in range(ny):
            for t1 in range(nx):
                p = [
                    idx(mf[0][0] * ix + mf[0][1] * iy + shift[s][0] + t1,
                        mf[1][0] * ix + mf[1][1] * iy + shift[s][1] + t2, perm[s])
                    for iy in range(ny) for ix in range(nx) for s in range(nb)
                ]
                pa = np.asarray(p, dtype=np.int64)
                if len(set(p)) != n or any(site_labels[p[i]] != site_labels[i] for i in range(n)):
                    raise ValueError("space-group op failed torus site-label invariance")
                if any(not np.array_equal(a[np.ix_(pa, pa)], a) for a in adjs):
                    raise ValueError("space-group op failed torus adjacency invariance")
                found.add(tuple(p))

    perms = [list(p) for p in sorted(found)]
    report = {
        "n_sites": n, "order": len(perms), "permutations": perms, "group_closed": _group_closed(perms),
        "provenance": "[PROVED: integer space-group action, adjacency invariance and closure checked exactly]",
    }
    return {
        "group": str(group), "nx": nx, "ny": ny, "n_sites": n,
        "lattice": [[nx * a1[0], nx * a1[1]], [ny * a2[0], ny * a2[1]]],
        "positions": positions, "sublattice": sublattice, "site_labels": site_labels,
        "shell_distances": [sh["distance"] for sh in shells["shells"]],
        "shell_edges": shell_edges, "shells": shells, "report": report, **report,
    }


def cluster_site_permutations(obj: Any, *, supercell: tuple[int, int] = (4, 4), max_shells: int = 3) -> dict[str, Any]:
    """Compute exact site-permutation symmetry group for an `nx x ny` periodic torus or open defect crystal."""
    rs = _rust.get()
    if _is_periodic_obj(obj):
        nx, ny = int(supercell[0]), int(supercell[1])
        group, cell = _extract_cell(obj)
        if rs is not None and hasattr(rs, "periodic_supercell_solver_json"):
            res = json.loads(rs.periodic_supercell_solver_json(group, json.dumps(cell), nx, ny, int(max_shells)))
            return {**res, **res["report"]}
        return _periodic_perms_py(group, cell, nx, ny, int(max_shells))

    crystal, z_vals, _, labels = _extract_crystal(obj)
    pos3 = [[float(crystal["pos"][i][0]), float(crystal["pos"][i][1]), 0.0 if z_vals is None else float(z_vals[i])]
            for i in range(len(crystal["pos"]))]
    edges = [(int(u), int(v)) for u, v in crystal["edges"]]
    if rs is not None and hasattr(rs, "cluster_site_permutations_json"):
        rep = json.loads(rs.cluster_site_permutations_json(json.dumps(pos3), json.dumps(labels), json.dumps([edges])))
    else:
        rep = _cluster_perms_py(pos3, labels, [edges])
    rep["positions"], rep["edges"] = pos3, [list(e) for e in edges]
    return rep
