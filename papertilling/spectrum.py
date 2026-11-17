"""Tight-binding nearest-neighbour spectrum, multiplet LDOS, zero modes, and disclination charges."""
from __future__ import annotations

from collections import deque
import json
import math

import numpy as np
from scipy.linalg import eigh

from . import _rust
from .lattice import Lattice
from .surfaces.flat import torus


class SpectrumResult(dict):
    """Tight-binding spectrum and charge diagnostics with dict-key and attribute access."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def ldos_for_level(self, idx: int, multiplet: bool = True) -> np.ndarray:
        """Local density of states for eigenstate `idx` (or its full near-degenerate multiplet)."""
        evecs = self["eigenvectors"]
        n = evecs.shape[0]
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"level index {idx} out of range for N={n}")
        if not multiplet:
            return np.abs(evecs[:, idx]) ** 2
        for _, start, end, _ in self["multiplets"]:
            if start <= idx < end:
                return np.sum(np.abs(evecs[:, start:end]) ** 2, axis=1)
        return np.abs(evecs[:, idx]) ** 2


def _cluster_multiplets(evals: np.ndarray, tol: float = 1e-8) -> list[tuple[float, int, int, int]]:
    n = len(evals)
    out: list[tuple[float, int, int, int]] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and (evals[j] - evals[j - 1]) < tol and (evals[j] - evals[i]) < 10.0 * tol:
            j += 1
        out.append((float(np.mean(evals[i:j])), i, j, j - i))
        i = j
    return out


def _cone_seeds_from_surface(surface) -> list[list[int]]:
    seeds: list[list[int]] = []
    for c in getattr(surface, "cones", ()):
        if not isinstance(c, dict):
            continue
        if c.get("site") is not None:
            seeds.append([int(c["site"])])
        elif c.get("sites") is not None:
            seeds.append([int(s) for s in c["sites"]])
        elif c.get("face") is not None:
            seeds.append([int(s) for s in surface.faces[int(c["face"])]])
        elif c.get("faces") is not None:
            seeds.append([int(s) for fi in c["faces"] for s in surface.faces[int(fi)]])
    return seeds


def _extract_defect_graph(obj, conf=None) -> tuple[int, bool, list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    d = obj
    n = len(d.sites)
    pos3 = getattr(conf, "xyz", None) if conf is not None else None
    pos2 = np.asarray(d.positions, float)
    pos = np.asarray(pos3, float) if pos3 is not None else pos2
    edges = [(int(u), int(v)) for u, v in d.edges]
    h_iso = d.info.get("h")
    r_ij = np.zeros(len(edges), float)
    p_i = np.zeros((len(edges), 2), float)
    p_j = np.zeros((len(edges), 2), float)
    seam = np.zeros(len(edges), bool)
    is_mobius = (d.kind == "mobius") or bool(d.info.get("mobius"))
    for k, (u, v) in enumerate(edges):
        ed = d.edge[(u, v)]
        j_val = int(ed.get("j", 0))
        is_seam = (is_mobius and abs(j_val) % 2 == 1) or bool(ed.get("seam"))
        seam[k] = is_seam
        if is_mobius and abs(j_val) % 2 == 1 and pos3 is None and h_iso is not None:
            M, t = np.asarray(h_iso[0], float), np.asarray(h_iso[1], float)
            q_img = (M @ pos2[v] + t) if j_val > 0 else (M.T @ (pos2[v] - t))
            r_ij[k] = float(np.linalg.norm(q_img - pos2[u]))
            p_i[k], p_j[k] = pos2[u], q_img
        else:
            r_ij[k] = float(np.linalg.norm(pos[v] - pos[u]))
            p_i[k], p_j[k] = pos2[u], pos2[v]
    seeds: list[list[int]] = []
    if conf is not None:
        seeds = [[int(s) for s in c["sites"]] for c in conf.cores() if c.get("sites")]
        if not seeds:
            cs = sorted(conf.core_sites())
            if cs:
                seeds = [[int(s)] for s in cs]
    return n, False, edges, r_ij, p_i, p_j, seam, seeds


def _extract_surface_graph(S) -> tuple[int, bool, list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    n = len(S.sites)
    is_closed = bool(S.closed)
    is_proj = bool(getattr(S, "projective", False)
                   or (getattr(S, "cover", None) is not None and not S.orientable and S.chi == 1))
    relaxed = bool(getattr(S, "_elastic_relaxed", False))
    flat_kind = any(tag in S.kind for tag in ("torus", "cylinder", "Klein", "Moebius", "genus"))
    use_3d = relaxed or (S.xyz is not None and not flat_kind)

    edges: list[tuple[int, int]] = []
    r_ij = np.zeros(len(S.bonds), float)
    p_i = np.zeros((len(S.bonds), 2), float)
    p_j = np.zeros((len(S.bonds), 2), float)
    seam = np.zeros(len(S.bonds), bool)
    uv = np.asarray(S.uv, float)
    cbonds = S.cover.bonds if is_proj and getattr(S, "cover", None) is not None else None
    cxyz = np.asarray(S.cover.xyz, float) if is_proj and getattr(S, "cover", None) is not None else None
    xyz = np.asarray(S.xyz, float) if S.xyz is not None else None

    for k, (a, b, v) in enumerate(S.bonds):
        a, b = int(a), int(b)
        edges.append((a, b))
        vv = np.asarray(v, float)
        if "Moebius" in S.kind and float(np.linalg.norm(uv[a] + vv[:2] - uv[b])) > 1e-4:
            seam[k] = True
        if is_proj and use_3d and cxyz is not None and cbonds is not None:
            ca, cb = (cbonds[2 * k][0], cbonds[2 * k][1]) if 2 * k < len(cbonds) else (a, b)
            if {ca % n, cb % n} != {a % n, b % n}:
                ca, cb = next((u, w) for u, w, _ in cbonds if {u % n, w % n} == {a % n, b % n})
            r_ij[k] = float(np.linalg.norm(cxyz[cb] - cxyz[ca]))
            p_i[k], p_j[k] = cxyz[ca, :2], cxyz[cb, :2]
        elif use_3d and xyz is not None:
            r_ij[k] = float(np.linalg.norm(xyz[b] - xyz[a]))
            p_i[k], p_j[k] = xyz[a, :2], xyz[b, :2]
        else:
            r_ij[k] = math.hypot(float(vv[0]), float(vv[1]))
            p_i[k], p_j[k] = uv[a], uv[a] + vv[:2]
    return n, is_closed, edges, r_ij, p_i, p_j, seam, _cone_seeds_from_surface(S)


def _normalize_target(target):
    if isinstance(target, Lattice):
        return _extract_surface_graph(torus(target, (4, 0), (0, 4)))
    if hasattr(target, "build") and callable(target.build) and isinstance(target, dict):
        target = target.build()
    if hasattr(target, "defect"):
        return _extract_defect_graph(target.defect, conf=target)
    if hasattr(target, "edges") and hasattr(target, "edge"):
        return _extract_defect_graph(target, conf=None)
    if hasattr(target, "bonds") and hasattr(target, "sites"):
        return _extract_surface_graph(target)
    raise TypeError(f"unsupported target type for spectrum(): {type(target).__name__}")


def _geodesic_dists(n: int, edges: list[tuple[int, int]], seeds: list[list[int]]) -> list[np.ndarray]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    dists: list[np.ndarray] = []
    for sd in seeds:
        dist = np.full(n, -1, int)
        q = deque()
        for s in sd:
            if 0 <= s < n and dist[s] < 0:
                dist[s] = 0
                q.append(s)
        while q:
            u = q.popleft()
            for w in adj[u]:
                if dist[w] < 0:
                    dist[w] = dist[u] + 1
                    q.append(w)
        dists.append(dist)
    return dists


def _resolve_mu(evals: np.ndarray, mu: float, filling: float | None) -> float:
    if filling is None:
        return float(mu)
    n = len(evals)
    if n == 0:
        return float(mu)
    target_e = float(filling) * n
    if target_e <= 0.0:
        return float(evals[0] - 1.0)
    if target_e >= n:
        return float(evals[-1] + 1.0)
    k = int(round(target_e))
    if abs(target_e - k) < 1e-9 and 0 < k < n and (evals[k] - evals[k - 1]) > 1e-8:
        return float(0.5 * (evals[k - 1] + evals[k]))
    return float(evals[min(max(int(math.floor(target_e)), 0), n - 1)])


def _result_from_json(raw: str, level_idx: int | None, multiplet: bool) -> SpectrumResult:
    d = json.loads(raw)
    for k in ("eigenvalues", "core_weight", "ldos", "delta_rho", "cone_charges", "delta_Q"):
        if d.get(k) is not None:
            d[k] = np.asarray(d[k], float)
    if "eigenvectors" in d:
        ev = d["eigenvectors"]
        if ev and isinstance(ev[0][0], list):
            arr = np.asarray(ev, float)
            d["eigenvectors"] = arr[..., 0] + 1j * arr[..., 1]
        else:
            d["eigenvectors"] = np.asarray(ev, float)
    d["multiplets"] = [tuple(m) for m in d["multiplets"]]
    res = SpectrumResult(d)
    if level_idx is not None and "eigenvectors" in res:
        res["ldos"] = res.ldos_for_level(level_idx, multiplet=multiplet)
    return res


def spectrum(
    target,
    beta: float = 0.0,
    flux: float = 0.0,
    *,
    mu: float = 0.0,
    filling: float | None = None,
    seam_sign: int = 1,
    multiplet: bool = True,
    t0: float = 1.0,
    core_radius: float | None = None,
    level_idx: int | None = None,
) -> SpectrumResult:
    """Compute nearest-neighbour tight-binding spectrum, multiplet LDOS, and cone charges."""
    if seam_sign not in (1, -1):
        raise ValueError("seam_sign must be +1 or -1")
    if filling is not None and not (0.0 <= filling <= 1.0):
        raise ValueError("filling must be in [0, 1]")
    if core_radius is not None and core_radius < 0.0:
        raise ValueError("core_radius must be non-negative")

    n, is_closed, edges, r_ij, p_i, p_j, seam, seeds = _normalize_target(target)
    rs = _rust.get()
    if rs is not None and hasattr(rs, "spectrum_json"):
        payload = json.dumps(dict(
            n=int(n), closed=bool(is_closed), edges=[(int(u), int(v)) for u, v in edges],
            r_ij=r_ij.tolist(), p_i=p_i.tolist(), p_j=p_j.tolist(), seam=seam.tolist(), seeds=seeds,
            beta=float(beta), flux=float(flux), mu=float(mu), filling=filling,
            seam_sign=int(seam_sign), multiplet=bool(multiplet), t0=float(t0),
            core_radius=core_radius, level_idx=level_idx,
        ))
        return _result_from_json(rs.spectrum_json(payload), level_idx, multiplet)

    if n == 0:
        empty = np.zeros(0, float)
        return SpectrumResult(
            eigenvalues=empty, eigenvectors=np.zeros((0, 0), float), zero_mode_count=0,
            core_weight=empty, multiplets=[], ldos=empty, mu=float(mu),
            delta_rho=empty, cone_charges=empty, delta_Q=empty,
        )

    l_mean = float(np.mean(r_ij)) if len(r_ij) else 1.0
    t_ij = float(t0) * np.exp(-float(beta) * (r_ij / max(l_mean, 1e-12) - 1.0))
    if seam_sign == -1 and len(seam):
        t_ij = np.where(seam, -t_ij, t_ij)

    use_flux = abs(float(flux)) > 0.0
    H = np.zeros((n, n), dtype=np.complex128 if use_flux else np.float64)
    if use_flux:
        A_ij = 2.0 * math.pi * float(flux) * 0.5 * (p_i[:, 0] * p_j[:, 1] - p_i[:, 1] * p_j[:, 0])
        hops = -t_ij * np.exp(1j * A_ij)
        for (u, v), h_val in zip(edges, hops):
            H[u, v] += h_val
            H[v, u] += np.conj(h_val)
    else:
        for (u, v), t_val in zip(edges, t_ij):
            H[u, v] -= t_val
            H[v, u] -= t_val

    evals, evecs = eigh(H)
    evals = np.asarray(evals, float)
    zero_mode_count = int(np.sum(np.abs(evals) < 1e-6))
    multiplets = _cluster_multiplets(evals, tol=1e-8)
    mu_used = _resolve_mu(evals, mu, filling)

    below = evals < (mu_used - 1e-8)
    at_mu = np.abs(evals - mu_used) <= 1e-8
    occ = np.sum(np.abs(evecs[:, below]) ** 2, axis=1) + 0.5 * np.sum(np.abs(evecs[:, at_mu]) ** 2, axis=1)
    nu_ref = float(np.mean(occ)) if (is_closed or abs(mu_used) > 1e-12 or filling is not None) else 0.5
    delta_rho = occ - nu_ref

    dists = _geodesic_dists(n, edges, seeds)
    r_ball = float(core_radius) if core_radius is not None else 2.0
    if dists:
        d_stack = np.stack(dists, axis=0)
        d_valid = np.where(d_stack >= 0, d_stack, n + 1000)
        in_ball = np.min(d_valid, axis=0) <= r_ball + 1e-12
        core_weight = np.sum(np.abs(evecs[in_ball, :]) ** 2, axis=0)
        sigma = max(r_ball, 1e-6)
        weights = [np.where(dc >= 0, np.exp(-0.5 * (dc / sigma) ** 2), 0.0) for dc in dists]
        delta_Q = np.array([float(w @ delta_rho) for w in weights], float)
    else:
        core_weight = np.zeros(n, float)
        delta_Q = np.zeros(0, float)

    res = SpectrumResult(
        eigenvalues=evals,
        eigenvectors=evecs,
        zero_mode_count=zero_mode_count,
        core_weight=core_weight,
        multiplets=multiplets,
        ldos=np.zeros(n, float),
        mu=mu_used,
        delta_rho=delta_rho,
        cone_charges=delta_Q,
        delta_Q=delta_Q,
    )
    idx_use = int(np.argmin(np.abs(evals - mu_used))) if level_idx is None else int(level_idx)
    res["ldos"] = res.ldos_for_level(idx_use, multiplet=multiplet)
    return res
