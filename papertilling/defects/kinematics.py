"""Particle-conserving defect kinematics on 2D crystals (1D glide and disclination-dislocation reactions)."""
from __future__ import annotations

import math

import numpy as np

from .build import build, lattice_data, rotation_about, same_charge, validate_specs


def _rot(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s], [s, c]], float)


def _check_radius(lat, specs_a: list[dict], specs_b: list[dict]) -> tuple[np.ndarray, float]:
    pts = np.vstack([np.array([s["at"] for s in specs_a], float), np.array([s["at"] for s in specs_b], float)])
    ctr = pts.mean(axis=0)
    max_d = float(np.max(np.linalg.norm(pts - ctr, axis=1)))
    max_b = max((float(np.linalg.norm(s["b"])) for s in specs_a + specs_b if s["kind"] == "dislocation"), default=0.0)
    r_check = max_d + max(14.5 * lat.lmax + max_b, 6.0 * lat.scale)
    return ctr, r_check


def glide_line_spacing(lattice, b) -> float:
    """Exact perpendicular spacing between adjacent Delta_N = const atomic glide planes for Burgers vector b."""
    lat = lattice_data(lattice)
    bv = np.asarray(b, float)
    area = abs(float(lat.cell.a1[0] * lat.cell.a2[1] - lat.cell.a1[1] * lat.cell.a2[0]))
    f = np.round(lat.Ainv @ bv).astype(int)
    g = math.gcd(abs(int(f[0])), abs(int(f[1]))) or 1
    b_prim = bv / g
    return float(area / max(float(np.linalg.norm(b_prim)), 1e-12))


def project_glide(spec: dict, target_at, anchor_at=None) -> dict:
    """Project a dislocation drag target strictly onto its 1D glide line (Delta_N = 0, zero climb)."""
    if spec.get("kind") != "dislocation":
        raise ValueError("disclinations are immobile without emitting or absorbing a dislocation")
    b = np.asarray(spec["b"], float)
    lb = float(np.linalg.norm(b))
    if lb < 1e-12:
        raise ValueError("dislocation Burgers vector must be non-zero")
    ub = b / lb
    raw_anc = spec.get("at") if anchor_at is None else anchor_at
    if raw_anc is None:
        raise ValueError("give an explicit dislocation anchor position before projecting glide")
    anc = np.asarray(raw_anc, float)
    tgt = np.asarray(target_at, float)
    s = float((tgt - anc) @ ub)
    out = dict(spec)
    out["at"] = tuple((anc + s * ub).tolist())
    return out


def step_disclination_emit(
    lattice,
    specs: list[dict],
    disc_idx: int,
    target_at,
    *,
    snap: bool = False,
) -> tuple[list[dict], dict | None]:
    """Step disclination `disc_idx` to `target_at` by emitting a Delta_N = 0, G~+-conserving dislocation."""
    lat = lattice_data(lattice)
    norm = validate_specs(lat, specs, snap=False)
    if disc_idx < 0 or disc_idx >= len(norm) or norm[disc_idx]["kind"] != "disclination":
        raise ValueError(f"spec {disc_idx + 1} is not a disclination")
    s_disc = norm[disc_idx]
    ang = float(s_disc["angle"])
    c_old = np.asarray(s_disc["at"], float)
    c_new = lat.nearest_centre(ang, target_at) if snap else np.asarray(target_at, float)
    if not snap and not lat.L.is_symmetry(rotation_about(c_new, math.radians(ang)), probe=2.2):
        raise ValueError(f"{np.round(c_new, 4).tolist()} is not a {ang:g}-deg rotation centre")
    if float(np.linalg.norm(c_new - c_old)) <= 1e-6:
        return [dict(s) for s in norm], None

    q_before = build(lat, *norm, picture="raw").total_charge()
    ra = _rot(ang)
    dc = c_new - c_old
    mid = 0.5 * (c_old + c_new)
    b_base = (np.eye(2) - ra) @ dc
    cand_bs = [b_base, (ra.T - np.eye(2)) @ dc]
    for m_op, _ in lat.ops_plus:
        bm = m_op @ b_base
        if not any(float(np.linalg.norm(bm - prev)) < 1e-6 for prev in cand_bs):
            cand_bs.append(bm)

    for b_em in cand_bs:
        lb = float(np.linalg.norm(b_em))
        if lb < 1e-6:
            continue
        ub = b_em / lb
        for s_mul in (0.0, -1.35, 1.35, -2.4, 2.4, -3.5, 3.5):
            core_d = mid + (s_mul * max(lat.lmax, lb) + 0.037 * lat.lmax) * ub
            at_d = core_d - 0.5 * b_em
            new_specs = [dict(s) for s in norm]
            new_specs[disc_idx] = dict(s_disc, at=(float(c_new[0]), float(c_new[1])), cut=None)
            emitted = dict(kind="dislocation", b=(float(b_em[0]), float(b_em[1])), at=(float(at_d[0]), float(at_d[1])))
            new_specs.append(emitted)
            ctr, r_check = _check_radius(lat, norm, new_specs)
            try:
                c_after = build(lat, *new_specs, R=r_check, centre=ctr, picture="raw")
                if same_charge(lat, q_before, c_after.total_charge(), mirror=False):
                    return new_specs, emitted
            except ValueError:
                continue
    raise ValueError("no particle-conserving dislocation emission found for this disclination step")


def absorb_dislocation(
    lattice,
    specs: list[dict],
    disl_idx: int,
    disc_idx: int,
) -> tuple[list[dict], np.ndarray]:
    """Absorb dislocation `disl_idx` into disclination `disc_idx` along a Delta_N = 0 glide channel."""
    lat = lattice_data(lattice)
    norm = validate_specs(lat, specs, snap=False)
    if disl_idx < 0 or disl_idx >= len(norm) or norm[disl_idx]["kind"] != "dislocation":
        raise ValueError(f"spec {disl_idx + 1} is not a dislocation")
    if disc_idx < 0 or disc_idx >= len(norm) or norm[disc_idx]["kind"] != "disclination":
        raise ValueError(f"spec {disc_idx + 1} is not a disclination")

    s_disl, s_disc = norm[disl_idx], norm[disc_idx]
    b = np.asarray(s_disl["b"], float)
    lb = float(np.linalg.norm(b))
    ub = b / lb
    nb = np.array([-b[1] / lb, b[0] / lb], float)
    x_disl = np.asarray(s_disl["at"], float) + 0.5 * b
    ang = float(s_disc["angle"])
    c_old = np.asarray(s_disc["at"], float)
    ra = _rot(ang)
    inv_i_ra = np.linalg.inv(np.eye(2) - ra)
    dp = glide_line_spacing(lat, b)

    q_before = build(lat, *norm, picture="raw").total_charge()
    cand_bs = [b, ra @ b]
    for m_op, _ in lat.ops_plus:
        bm = m_op @ b
        if not any(float(np.linalg.norm(bm - prev)) < 1e-6 for prev in cand_bs):
            cand_bs.append(bm)

    valid_cands = []
    for b_eff in cand_bs:
        c_new = c_old - inv_i_ra @ b_eff
        if not lat.L.is_symmetry(rotation_about(c_new, math.radians(ang)), probe=2.2):
            continue
        mid = 0.5 * (c_old + c_new)
        if abs(float((mid - x_disl) @ nb)) <= 0.45 * dp:
            along = abs(float((mid - x_disl) @ ub))
            valid_cands.append((along, c_new))
    if not valid_cands:
        raise ValueError("cannot absorb: dislocation glide line misses the ΔN = 0 channel of this disclination (would require climb)")

    valid_cands.sort(key=lambda item: item[0])
    target_k = disc_idx if disc_idx < disl_idx else disc_idx - 1
    for _, c_new in valid_cands:
        new_specs = [dict(s) for k, s in enumerate(norm) if k != disl_idx]
        new_specs[target_k] = dict(s_disc, at=(float(c_new[0]), float(c_new[1])), cut=None)
        ctr, r_check = _check_radius(lat, norm, new_specs)
        try:
            c_after = build(lat, *new_specs, R=r_check, centre=ctr, picture="raw")
            if same_charge(lat, q_before, c_after.total_charge(), mirror=False):
                return new_specs, c_new
        except ValueError:
            continue
    raise ValueError("cannot absorb: outer holonomy conservation failed")


def glide_dislocation(
    lattice,
    specs: list[dict],
    disl_idx: int,
    target_at,
    *,
    anchor_at=None,
) -> tuple[list[dict], bool]:
    """Glide dislocation `disl_idx` along b (`Delta_N = 0`), parallel-transporting b across branch cuts."""
    lat = lattice_data(lattice)
    norm = validate_specs(lat, specs, snap=False)
    if disl_idx < 0 or disl_idx >= len(norm) or norm[disl_idx]["kind"] != "dislocation":
        raise ValueError(f"spec {disl_idx + 1} is not a dislocation")
    moved = project_glide(norm[disl_idx], target_at, anchor_at=anchor_at)
    after = [dict(s) for s in norm]
    after[disl_idx] = moved
    if not any(s["kind"] == "disclination" for s in norm):
        build(lat, *after, picture="raw")
        return after, False

    q_before = build(lat, *norm, picture="raw").total_charge()
    ctr, r_check = _check_radius(lat, norm, after)
    try:
        c_after = build(lat, *after, R=r_check, centre=ctr, picture="raw")
        if same_charge(lat, q_before, c_after.total_charge(), mirror=False):
            return after, False
    except ValueError:
        pass

    b = np.asarray(moved["b"], float)
    for m_op, _ in lat.ops_plus:
        bm = m_op @ b
        if float(np.linalg.norm(bm - b)) < 1e-6:
            continue
        trial = [dict(s) for s in after]
        trial[disl_idx] = dict(moved, b=(float(bm[0]), float(bm[1])), cut=None)
        ctr, r_check = _check_radius(lat, norm, trial)
        try:
            c_trial = build(lat, *trial, R=r_check, centre=ctr, picture="raw")
            if same_charge(lat, q_before, c_trial.total_charge(), mirror=False):
                return trial, True
        except ValueError:
            continue
    raise ValueError("glide path blocked by defect core or branch cut")
